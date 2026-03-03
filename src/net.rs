// Architecture: Listen-server with server-authoritative world.
//   - Server thread owns tick loop at fixed 60 Hz (configurable)
//   - TCP: reliable ordered (block changes, chunk data, join/leave)
//   - UDP: unreliable fast (player positions/rotations)
//   - Clients send inputs → server simulates → broadcasts state
//   - Dirty tracking: only broadcast players whose state changed
//

use std::{
    collections::HashMap,
    io::{self, Read, Write},
    net::{TcpListener, TcpStream, UdpSocket, SocketAddr, Shutdown},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use crossbeam::channel::{self, Receiver, Sender, TryRecvError};

use crate::world::{BlockType, ChunkPos, CHUNK_X, CHUNK_Y, CHUNK_Z};

// ===== Constants =====

pub const DEFAULT_TCP_PORT: u16 = 25565;
pub const DEFAULT_UDP_PORT: u16 = 25566;
pub const SERVER_TICK_RATE: u32 = 60;
pub const TICK_DURATION: Duration = Duration::from_micros(16_667); // ~16.67ms for 60Hz
pub const MAX_PLAYERS: usize = 8;
pub const PROTOCOL_MAGIC: u32 = 0x534D5052; // "SMPR"
pub const PROTOCOL_VERSION: u8 = 1;

const MAX_UDP_PACKET: usize = 1400;
const MAX_TCP_PAYLOAD: usize = 512 * 1024;

// Delta thresholds for dirty detection
const POS_EPSILON: f32 = 0.001;
const ANGLE_EPSILON: f32 = 0.001;
// Time sync broadcast interval (ticks) — ~3Hz at 60 tick/s
const TIME_SYNC_INTERVAL: u32 = 20;

pub type PlayerId = u8;

// ===== Network Events =====

#[derive(Debug, Clone)]
pub enum NetEvent {
    PlayerJoined { id: PlayerId, name: String },
    PlayerLeft { id: PlayerId },
    PlayerState { id: PlayerId, position: [f32; 3], yaw: f32, pitch: f32, on_ground: bool },
    BlockChange { wx: i32, wy: i32, wz: i32, block: BlockType },
    ChunkData { pos: ChunkPos, blocks: Vec<BlockType> },
    AssignedId { id: PlayerId },
    Disconnected { reason: String },
    WorldSeed { seed: u32 },
    TimeSync { game_time: f32 },
}

#[derive(Debug, Clone)]
pub enum NetCommand {
    SendPlayerState { position: [f32; 3], yaw: f32, pitch: f32, on_ground: bool },
    RequestBlockChange { wx: i32, wy: i32, wz: i32, block: BlockType },
    RequestChunk { pos: ChunkPos },
    SendTimeSync { game_time: f32 },
    Disconnect,
}

// ===== Wire Protocol =====

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacketType {
    Handshake      = 0x01,
    HandshakeAck   = 0x02,
    PlayerJoin     = 0x03,
    PlayerLeave    = 0x04,
    BlockChange    = 0x05,
    ChunkRequest   = 0x06,
    ChunkResponse  = 0x07,
    Disconnect     = 0x08,
    PlayerPosition = 0x10,
    TimeSync      = 0x11,
}

impl PacketType {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x01 => Some(Self::Handshake),
            0x02 => Some(Self::HandshakeAck),
            0x03 => Some(Self::PlayerJoin),
            0x04 => Some(Self::PlayerLeave),
            0x05 => Some(Self::BlockChange),
            0x06 => Some(Self::ChunkRequest),
            0x07 => Some(Self::ChunkResponse),
            0x08 => Some(Self::Disconnect),
            0x10 => Some(Self::PlayerPosition),
            0x11 => Some(Self::TimeSync),
            _ => None,
        }
    }
}

// ===== TCP Framing =====

fn tcp_send(stream: &mut TcpStream, ptype: PacketType, payload: &[u8]) -> io::Result<()> {
    let total_len = 1 + payload.len();
    if total_len > MAX_TCP_PAYLOAD {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "packet too large"));
    }
    stream.write_all(&(total_len as u32).to_le_bytes())?;
    stream.write_all(&[ptype as u8])?;
    stream.write_all(payload)?;
    stream.flush()
}

fn tcp_recv(stream: &mut TcpStream) -> io::Result<(PacketType, Vec<u8>)> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let total_len = u32::from_le_bytes(len_buf) as usize;
    if total_len == 0 || total_len > MAX_TCP_PAYLOAD {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad packet length"));
    }
    let mut buf = vec![0u8; total_len];
    stream.read_exact(&mut buf)?;
    let ptype = PacketType::from_u8(buf[0])
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "unknown packet type"))?;
    Ok((ptype, buf[1..].to_vec()))
}

// ===== Byte Helpers =====

fn write_f32(buf: &mut Vec<u8>, v: f32) { buf.extend_from_slice(&v.to_le_bytes()); }
fn write_i32(buf: &mut Vec<u8>, v: i32) { buf.extend_from_slice(&v.to_le_bytes()); }
fn write_u32(buf: &mut Vec<u8>, v: u32) { buf.extend_from_slice(&v.to_le_bytes()); }
fn write_u8(buf: &mut Vec<u8>, v: u8)   { buf.push(v); }

fn write_string(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    let len = bytes.len().min(255);
    buf.push(len as u8);
    buf.extend_from_slice(&bytes[..len]);
}

fn read_f32(data: &[u8], off: &mut usize) -> f32 {
    let v = f32::from_le_bytes(data[*off..*off+4].try_into().unwrap_or([0;4]));
    *off += 4; v
}
fn read_i32(data: &[u8], off: &mut usize) -> i32 {
    let v = i32::from_le_bytes(data[*off..*off+4].try_into().unwrap_or([0;4]));
    *off += 4; v
}
fn read_u32(data: &[u8], off: &mut usize) -> u32 {
    let v = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap_or([0;4]));
    *off += 4; v
}
fn read_u8(data: &[u8], off: &mut usize) -> u8 {
    let v = data.get(*off).copied().unwrap_or(0);
    *off += 1; v
}
fn read_string(data: &[u8], off: &mut usize) -> String {
    let len = read_u8(data, off) as usize;
    let end = (*off + len).min(data.len());
    let s = String::from_utf8_lossy(&data[*off..end]).to_string();
    *off = end; s
}

// ===== UDP Position Packet (24 bytes) =====
// [u8 type][u8 id][f32 x][f32 y][f32 z][f32 yaw][f32 pitch][u8 flags][u8 pad]

fn encode_position_udp(id: PlayerId, pos: [f32;3], yaw: f32, pitch: f32, on_ground: bool) -> [u8; 24] {
    let mut buf = [0u8; 24];
    buf[0] = PacketType::PlayerPosition as u8;
    buf[1] = id;
    buf[2..6].copy_from_slice(&pos[0].to_le_bytes());
    buf[6..10].copy_from_slice(&pos[1].to_le_bytes());
    buf[10..14].copy_from_slice(&pos[2].to_le_bytes());
    buf[14..18].copy_from_slice(&yaw.to_le_bytes());
    buf[18..22].copy_from_slice(&pitch.to_le_bytes());
    buf[22] = if on_ground { 1 } else { 0 };
    buf
}

fn decode_position_udp(buf: &[u8]) -> Option<(PlayerId, [f32;3], f32, f32, bool)> {
    if buf.len() < 23 || buf[0] != PacketType::PlayerPosition as u8 { return None; }
    let id = buf[1];
    let x = f32::from_le_bytes(buf[2..6].try_into().ok()?);
    let y = f32::from_le_bytes(buf[6..10].try_into().ok()?);
    let z = f32::from_le_bytes(buf[10..14].try_into().ok()?);
    let yaw = f32::from_le_bytes(buf[14..18].try_into().ok()?);
    let pitch = f32::from_le_bytes(buf[18..22].try_into().ok()?);
    Some((id, [x,y,z], yaw, pitch, buf[22] != 0))
}

// ===== UDP TimeSync Packet (5 bytes) =====
// [u8 type=0x11][f32 game_time]

fn encode_time_sync_udp(game_time: f32) -> [u8; 5] {
    let mut buf = [0u8; 5];
    buf[0] = PacketType::TimeSync as u8;
    buf[1..5].copy_from_slice(&game_time.to_le_bytes());
    buf
}

fn decode_time_sync_udp(buf: &[u8]) -> Option<f32> {
    if buf.len() < 5 || buf[0] != PacketType::TimeSync as u8 { return None; }
    Some(f32::from_le_bytes(buf[1..5].try_into().ok()?))
}

// ===== TCP Block Change Packet (13 bytes) =====

fn encode_block_change(wx: i32, wy: i32, wz: i32, block: BlockType) -> Vec<u8> {
    let mut buf = Vec::with_capacity(13);
    write_i32(&mut buf, wx);
    write_i32(&mut buf, wy);
    write_i32(&mut buf, wz);
    write_u8(&mut buf, block as u8);
    buf
}

fn decode_block_change(data: &[u8]) -> Option<(i32, i32, i32, BlockType)> {
    if data.len() < 13 { return None; }
    let mut off = 0;
    let wx = read_i32(data, &mut off);
    let wy = read_i32(data, &mut off);
    let wz = read_i32(data, &mut off);
    let block = BlockType::from_u8(read_u8(data, &mut off))?;
    Some((wx, wy, wz, block))
}

// ===== Chunk RLE =====
// [u8 block][u16 count] per run. Typical 32K chunk → ~200 bytes.

pub fn rle_encode_chunk(blocks: &[BlockType]) -> Vec<u8> {
    let mut out = Vec::with_capacity(256);
    if blocks.is_empty() { return out; }
    let mut current = blocks[0];
    let mut count: u16 = 1;
    for &b in &blocks[1..] {
        if b == current && count < u16::MAX { count += 1; }
        else {
            out.push(current as u8);
            out.extend_from_slice(&count.to_le_bytes());
            current = b; count = 1;
        }
    }
    out.push(current as u8);
    out.extend_from_slice(&count.to_le_bytes());
    out
}

pub fn rle_decode_chunk(data: &[u8]) -> Option<Vec<BlockType>> {
    let expected = CHUNK_X * CHUNK_Y * CHUNK_Z;
    let mut blocks = Vec::with_capacity(expected);
    let mut i = 0;
    while i + 2 < data.len() {
        let block = BlockType::from_u8(data[i])?;
        let count = u16::from_le_bytes([data[i+1], data[i+2]]) as usize;
        i += 3;
        for _ in 0..count {
            blocks.push(block);
            if blocks.len() > expected { return None; }
        }
    }
    if blocks.len() == expected { Some(blocks) } else { None }
}

// ===== Remote Player =====

#[derive(Debug, Clone)]
pub struct RemotePlayer {
    pub id: PlayerId,
    pub name: String,
    pub position: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub on_ground: bool,
    pub last_update: Instant,
}

impl RemotePlayer {
    pub fn new(id: PlayerId, name: String) -> Self {
        Self { id, name, position: [0.0, 80.0, 0.0], yaw: 0.0, pitch: 0.0, on_ground: false, last_update: Instant::now() }
    }

    pub fn interpolate_toward(&mut self, target_pos: [f32;3], target_yaw: f32, alpha: f32) {
        let a = alpha.clamp(0.0, 1.0);
        for i in 0..3 { self.position[i] += (target_pos[i] - self.position[i]) * a; }
        let mut dy = target_yaw - self.yaw;
        if dy > std::f32::consts::PI { dy -= 2.0 * std::f32::consts::PI; }
        if dy < -std::f32::consts::PI { dy += 2.0 * std::f32::consts::PI; }
        self.yaw += dy * a;
    }

    pub fn model_matrix(&self) -> [[f32;4];4] {
        let (c, s, p) = (self.yaw.cos(), self.yaw.sin(), self.position);
        [
            [ c,   0.0, -s,  0.0],
            [ 0.0, 1.0,  0.0, 0.0],
            [ s,   0.0,  c,  0.0],
            [p[0], p[1], p[2], 1.0],
        ]
    }
}

// ===== Player State Snapshot for Dirty Detection =====

#[derive(Clone, Copy, Default)]
struct PlayerSnapshot {
    position: [f32; 3],
    yaw: f32,
    pitch: f32,
    on_ground: bool,
}

impl PlayerSnapshot {
    /// Returns true if current state differs enough from snapshot to warrant broadcast.
    fn differs_from(&self, pos: [f32; 3], yaw: f32, pitch: f32, on_ground: bool) -> bool {
        let dp = (self.position[0] - pos[0]).abs()
               + (self.position[1] - pos[1]).abs()
               + (self.position[2] - pos[2]).abs();
        let dy = (self.yaw - yaw).abs();
        let dpi = (self.pitch - pitch).abs();
        dp > POS_EPSILON || dy > ANGLE_EPSILON || dpi > ANGLE_EPSILON || self.on_ground != on_ground
    }
    
    fn update(&mut self, pos: [f32; 3], yaw: f32, pitch: f32, on_ground: bool) {
        self.position = pos;
        self.yaw = yaw;
        self.pitch = pitch;
        self.on_ground = on_ground;
    }
}

// =====================================================================
//  SERVER
// =====================================================================

struct ServerClient {
    id: PlayerId,
    name: String,
    tcp_stream: TcpStream,
    udp_addr: Option<SocketAddr>,
    position: [f32; 3],
    yaw: f32,
    pitch: f32,
    on_ground: bool,
    // Dirty tracking for mutation-based broadcasts
    dirty: bool,
    last_snapshot: PlayerSnapshot,
}

pub struct Server {
    cmd_rx: Receiver<NetCommand>,
    event_tx: Sender<NetEvent>,
    #[allow(dead_code)]
    tcp_listener: TcpListener,
    udp_socket: UdpSocket,
    clients: HashMap<PlayerId, ServerClient>,
    next_id: PlayerId,
    seed: u32,
    game_time: f32,
    shutdown: Arc<AtomicBool>,
    _threads: Vec<thread::JoinHandle<()>>,
    tcp_incoming_tx: Sender<(PlayerId, PacketType, Vec<u8>)>,
    tcp_incoming_rx: Receiver<(PlayerId, PacketType, Vec<u8>)>,
    new_conn_rx: Receiver<TcpStream>,
    // Sub-tick scheduling counter
    tick_counter: u32,
    // Host player dirty tracking
    host_dirty: bool,
    host_snapshot: PlayerSnapshot,
}

impl Server {
    pub fn start(seed: u32, tcp_port: u16, udp_port: u16)
        -> io::Result<(Sender<NetCommand>, Receiver<NetEvent>, thread::JoinHandle<()>)>
    {
        let (cmd_tx, cmd_rx) = channel::unbounded();
        let (event_tx, event_rx) = channel::unbounded();
        let (tcp_inc_tx, tcp_inc_rx) = channel::unbounded();
        let (new_conn_tx, new_conn_rx) = channel::unbounded();

        let tcp_listener = TcpListener::bind(format!("0.0.0.0:{}", tcp_port))?;
        tcp_listener.set_nonblocking(true)?;
        let udp_socket = UdpSocket::bind(format!("0.0.0.0:{}", udp_port))?;
        udp_socket.set_nonblocking(true)?;

        let shutdown = Arc::new(AtomicBool::new(false));

        let accept_listener = tcp_listener.try_clone()?;
        let accept_shutdown = shutdown.clone();
        let accept_handle = thread::Builder::new().name("net-accept".into()).spawn(move || {
            while !accept_shutdown.load(Ordering::Relaxed) {
                match accept_listener.accept() {
                    Ok((stream, addr)) => {
                        println!("[server] Connection from {}", addr);
                        let _ = stream.set_nonblocking(false);
                        let _ = stream.set_nodelay(true);
                        let _ = new_conn_tx.send(stream);
                    }
                    Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(50));
                    }
                    Err(_) => break,
                }
            }
        })?;

        let mut server = Server {
            cmd_rx, event_tx, tcp_listener, udp_socket,
            clients: HashMap::new(), next_id: 1, seed,
            game_time: 0.30,
            shutdown: shutdown.clone(),
            _threads: vec![accept_handle],
            tcp_incoming_tx: tcp_inc_tx, tcp_incoming_rx: tcp_inc_rx,
            new_conn_rx,
            tick_counter: 0,
            host_dirty: false,
            host_snapshot: PlayerSnapshot::default(),
        };

        let handle = thread::Builder::new().name("net-server".into())
            .spawn(move || server.run())?;
        Ok((cmd_tx, event_rx, handle))
    }

    fn run(&mut self) {
        println!("[server] Listening TCP:{} UDP:{} @ {}Hz", DEFAULT_TCP_PORT, DEFAULT_UDP_PORT, SERVER_TICK_RATE);
        while !self.shutdown.load(Ordering::Relaxed) {
            let tick = Instant::now();
            
            self.accept_new_connections();
            self.process_tcp_incoming();
            self.process_udp_incoming();
            self.process_host_commands();
            
            // Only broadcast dirty players (mutation-based)
            self.broadcast_dirty_player_states();
            
            // Time sync at reduced rate (~3Hz)
            self.tick_counter = self.tick_counter.wrapping_add(1);
            if self.tick_counter % TIME_SYNC_INTERVAL == 0 {
                self.broadcast_time_sync();
            }
            
            let elapsed = tick.elapsed();
            if elapsed < TICK_DURATION { 
                thread::sleep(TICK_DURATION - elapsed); 
            }
        }
        let ids: Vec<_> = self.clients.keys().copied().collect();
        for id in ids { self.disconnect_client(id, "Server shutting down"); }
        println!("[server] Shut down");
    }

    fn accept_new_connections(&mut self) {
        while let Ok(mut stream) = self.new_conn_rx.try_recv() {
            let handshake = match tcp_recv(&mut stream) {
                Ok((PacketType::Handshake, data)) => data,
                _ => { let _ = stream.shutdown(Shutdown::Both); continue; }
            };
            let mut off = 0;
            let magic = read_u32(&handshake, &mut off);
            let version = read_u8(&handshake, &mut off);
            let name = read_string(&handshake, &mut off);

            if magic != PROTOCOL_MAGIC || version != PROTOCOL_VERSION {
                let _ = stream.shutdown(Shutdown::Both); continue;
            }
            if self.clients.len() >= MAX_PLAYERS - 1 {
                let _ = stream.shutdown(Shutdown::Both); continue;
            }

            let id = self.next_id;
            self.next_id = self.next_id.wrapping_add(1).max(1);

            let mut ack = Vec::with_capacity(9);
            write_u8(&mut ack, id);
            write_u32(&mut ack, self.seed);
            write_f32(&mut ack, self.game_time);
            if tcp_send(&mut stream, PacketType::HandshakeAck, &ack).is_err() {
                let _ = stream.shutdown(Shutdown::Both); continue;
            }

            // Broadcast join
            let mut join_pkt = Vec::new();
            write_u8(&mut join_pkt, id);
            write_string(&mut join_pkt, &name);
            for c in self.clients.values_mut() {
                let _ = tcp_send(&mut c.tcp_stream, PacketType::PlayerJoin, &join_pkt);
            }
            let _ = self.event_tx.send(NetEvent::PlayerJoined { id, name: name.clone() });

            // Tell new client about existing players
            let mut host_pkt = Vec::new();
            write_u8(&mut host_pkt, 0);
            write_string(&mut host_pkt, "Host");
            let _ = tcp_send(&mut stream, PacketType::PlayerJoin, &host_pkt);
            for c in self.clients.values() {
                let mut pkt = Vec::new();
                write_u8(&mut pkt, c.id);
                write_string(&mut pkt, &c.name);
                let _ = tcp_send(&mut stream, PacketType::PlayerJoin, &pkt);
            }

            // Per-client recv thread
            let recv = stream.try_clone().expect("clone TCP");
            let inc_tx = self.tcp_incoming_tx.clone();
            let sd = self.shutdown.clone();
            let cid = id;
            self._threads.push(thread::Builder::new()
                .name(format!("net-recv-{}", id))
                .spawn(move || {
                    let mut s = recv;
                    while !sd.load(Ordering::Relaxed) {
                        match tcp_recv(&mut s) {
                            Ok((pt, d)) => { if inc_tx.send((cid, pt, d)).is_err() { break; } }
                            Err(_) => { let _ = inc_tx.send((cid, PacketType::Disconnect, vec![])); break; }
                        }
                    }
                }).expect("spawn recv"));

            println!("[server] '{}' joined as id={}", name, id);
            self.clients.insert(id, ServerClient {
                id, name, tcp_stream: stream, udp_addr: None,
                position: [0.0, 80.0, 0.0], yaw: 0.0, pitch: 0.0, on_ground: false,
                dirty: true,  // Broadcast initial state
                last_snapshot: PlayerSnapshot::default(),
            });
        }
    }

    fn process_tcp_incoming(&mut self) {
        loop {
            match self.tcp_incoming_rx.try_recv() {
                Ok((sid, pt, data)) => self.handle_tcp(sid, pt, &data),
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
            }
        }
    }

    fn handle_tcp(&mut self, sender: PlayerId, ptype: PacketType, data: &[u8]) {
        match ptype {
            PacketType::BlockChange => {
                if let Some((wx, wy, wz, block)) = decode_block_change(data) {
                    let _ = self.event_tx.send(NetEvent::BlockChange { wx, wy, wz, block });
                    let pkt = encode_block_change(wx, wy, wz, block);
                    for c in self.clients.values_mut() {
                        if c.id != sender { let _ = tcp_send(&mut c.tcp_stream, PacketType::BlockChange, &pkt); }
                    }
                }
            }
            PacketType::ChunkRequest if data.len() >= 8 => {
                let mut off = 0;
                let cx = read_i32(data, &mut off);
                let cz = read_i32(data, &mut off);
                let _ = self.event_tx.send(NetEvent::ChunkData { pos: ChunkPos::new(cx,cz), blocks: vec![] });
            }
            PacketType::Disconnect => { self.disconnect_client(sender, "Client disconnected"); }
            _ => {}
        }
    }

    fn process_udp_incoming(&mut self) {
        let mut buf = [0u8; MAX_UDP_PACKET];
        loop {
            match self.udp_socket.recv_from(&mut buf) {
                Ok((len, addr)) => {
                    if let Some((id, pos, yaw, pitch, og)) = decode_position_udp(&buf[..len]) {
                        if let Some(c) = self.clients.get_mut(&id) {
                            if c.udp_addr.is_none() { c.udp_addr = Some(addr); }
                            // Check if state actually changed before marking dirty
                            if c.last_snapshot.differs_from(pos, yaw, pitch, og) {
                                c.position = pos;
                                c.yaw = yaw;
                                c.pitch = pitch;
                                c.on_ground = og;
                                c.dirty = true;
                            }
                        }
                        // Always forward to host game loop for interpolation
                        let _ = self.event_tx.send(NetEvent::PlayerState { 
                            id, position: pos, yaw, pitch, on_ground: og 
                        });
                    }
                }
                Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(_) => break,
            }
        }
    }

    fn process_host_commands(&mut self) {
        loop {
            match self.cmd_rx.try_recv() {
                Ok(cmd) => match cmd {
                    NetCommand::SendPlayerState { position, yaw, pitch, on_ground } => {
                        // Only broadcast if host state changed
                        if self.host_snapshot.differs_from(position, yaw, pitch, on_ground) {
                            self.host_dirty = true;
                            self.host_snapshot.update(position, yaw, pitch, on_ground);
                        }
                        // Broadcast host position to all clients if dirty
                        if self.host_dirty {
                            let pkt = encode_position_udp(0, position, yaw, pitch, on_ground);
                            for c in self.clients.values() {
                                if let Some(a) = c.udp_addr { let _ = self.udp_socket.send_to(&pkt, a); }
                            }
                            self.host_dirty = false;
                        }
                    }
                    NetCommand::RequestBlockChange { wx, wy, wz, block } => {
                        let pkt = encode_block_change(wx, wy, wz, block);
                        for c in self.clients.values_mut() { 
                            let _ = tcp_send(&mut c.tcp_stream, PacketType::BlockChange, &pkt); 
                        }
                    }
                    NetCommand::SendTimeSync { game_time } => {
                        self.game_time = game_time;
                    }
                    NetCommand::Disconnect => { self.shutdown.store(true, Ordering::Release); }
                    _ => {}
                },
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => { self.shutdown.store(true, Ordering::Release); break; }
            }
        }
    }

    /// Broadcast only players whose state has changed since last broadcast.
    fn broadcast_dirty_player_states(&mut self) {
        // Collect dirty client data first to avoid borrow issues
        let dirty_states: Vec<(PlayerId, [f32;3], f32, f32, bool)> = self.clients.values()
            .filter(|c| c.dirty)
            .map(|c| (c.id, c.position, c.yaw, c.pitch, c.on_ground))
            .collect();
        
        if dirty_states.is_empty() {
            return;
        }
        
        // Collect UDP addresses
        let addrs: Vec<(PlayerId, Option<SocketAddr>)> = self.clients.values()
            .map(|c| (c.id, c.udp_addr))
            .collect();
        
        // Broadcast each dirty player to all other players
        for (sid, pos, yaw, pitch, on_ground) in &dirty_states {
            let pkt = encode_position_udp(*sid, *pos, *yaw, *pitch, *on_ground);
            for (rid, addr) in &addrs {
                if rid != sid {
                    if let Some(a) = addr { 
                        let _ = self.udp_socket.send_to(&pkt, *a); 
                    }
                }
            }
            // Notify host game loop
            let _ = self.event_tx.send(NetEvent::PlayerState {
                id: *sid, position: *pos, yaw: *yaw, pitch: *pitch, on_ground: *on_ground,
            });
        }
        
        // Clear dirty flags and update snapshots
        for c in self.clients.values_mut() {
            if c.dirty {
                c.last_snapshot.update(c.position, c.yaw, c.pitch, c.on_ground);
                c.dirty = false;
            }
        }
    }

    fn broadcast_time_sync(&self) {
        let pkt = encode_time_sync_udp(self.game_time);
        for c in self.clients.values() {
            if let Some(a) = c.udp_addr {
                let _ = self.udp_socket.send_to(&pkt, a);
            }
        }
    }

    fn disconnect_client(&mut self, id: PlayerId, reason: &str) {
        if let Some(mut c) = self.clients.remove(&id) {
            println!("[server] '{}' (id={}) left: {}", c.name, id, reason);
            let _ = c.tcp_stream.shutdown(Shutdown::Both);
            let mut pkt = Vec::new();
            write_u8(&mut pkt, id);
            for o in self.clients.values_mut() { let _ = tcp_send(&mut o.tcp_stream, PacketType::PlayerLeave, &pkt); }
            let _ = self.event_tx.send(NetEvent::PlayerLeft { id });
        }
    }
}

// =====================================================================
//  CLIENT
// =====================================================================

pub struct Client;

impl Client {
    pub fn connect(server_ip: &str, player_name: &str, tcp_port: u16, udp_port: u16)
        -> io::Result<(Sender<NetCommand>, Receiver<NetEvent>)>
    {
        let (cmd_tx, cmd_rx) = channel::unbounded();
        let (event_tx, event_rx) = channel::unbounded();
        let shutdown = Arc::new(AtomicBool::new(false));

        let addr = format!("{}:{}", server_ip, tcp_port);
        let mut tcp = TcpStream::connect_timeout(
            &addr.parse().map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?,
            Duration::from_secs(5),
        )?;
        tcp.set_nodelay(true)?;

        let mut hs = Vec::new();
        write_u32(&mut hs, PROTOCOL_MAGIC);
        write_u8(&mut hs, PROTOCOL_VERSION);
        write_string(&mut hs, player_name);
        tcp_send(&mut tcp, PacketType::Handshake, &hs)?;

        let (pt, ack) = tcp_recv(&mut tcp)?;
        if pt != PacketType::HandshakeAck || ack.len() < 5 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad ack"));
        }
        let mut off = 0;
        let my_id = read_u8(&ack, &mut off);
        let seed = read_u32(&ack, &mut off);
        let initial_time = if ack.len() >= 9 { read_f32(&ack, &mut off) } else { 0.30 };
        println!("[client] Connected id={} seed={} time={:.3}", my_id, seed, initial_time);

        let _ = event_tx.send(NetEvent::AssignedId { id: my_id });
        let _ = event_tx.send(NetEvent::WorldSeed { seed });
        let _ = event_tx.send(NetEvent::TimeSync { game_time: initial_time });

        let udp = UdpSocket::bind("0.0.0.0:0")?;
        let server_udp: SocketAddr = format!("{}:{}", server_ip, udp_port)
            .parse().map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        let _ = udp.send_to(&encode_position_udp(my_id, [0.0,80.0,0.0], 0.0, 0.0, false), server_udp);

        // TCP recv thread
        let tr = tcp.try_clone()?;
        let te = event_tx.clone();
        let ts = shutdown.clone();
        thread::Builder::new().name("net-c-tcp".into()).spawn(move || {
            let mut s = tr;
            while !ts.load(Ordering::Relaxed) {
                match tcp_recv(&mut s) {
                    Ok((pt, d)) => { if let Some(e) = Self::decode_tcp(pt, &d) { if te.send(e).is_err() { break; } } }
                    Err(_) => { let _ = te.send(NetEvent::Disconnected { reason: "Lost".into() }); break; }
                }
            }
        })?;

        // UDP recv thread
        let ur = udp.try_clone()?;
        let _ = ur.set_read_timeout(Some(Duration::from_millis(100)));
        let ue = event_tx.clone();
        let us = shutdown.clone();
        thread::Builder::new().name("net-c-udp".into()).spawn(move || {
            let mut buf = [0u8; MAX_UDP_PACKET];
            while !us.load(Ordering::Relaxed) {
                match ur.recv_from(&mut buf) {
                    Ok((len, _)) => {
                        let data = &buf[..len];
                        if len >= 5 && data[0] == PacketType::TimeSync as u8 {
                            if let Some(gt) = decode_time_sync_udp(data) {
                                let _ = ue.send(NetEvent::TimeSync { game_time: gt });
                            }
                        } else if let Some((id,pos,yaw,pitch,og)) = decode_position_udp(data) {
                            let _ = ue.send(NetEvent::PlayerState { id, position: pos, yaw, pitch, on_ground: og });
                        }
                    }
                    Err(ref e) if e.kind() == io::ErrorKind::WouldBlock || e.kind() == io::ErrorKind::TimedOut => continue,
                    Err(_) => break,
                }
            }
        })?;

        // Send thread
        let st = tcp;
        let su = udp;
        let ss = shutdown.clone();
        thread::Builder::new().name("net-c-send".into()).spawn(move || {
            let mut t = st;
            while !ss.load(Ordering::Relaxed) {
                match cmd_rx.recv_timeout(Duration::from_millis(50)) {
                    Ok(cmd) => match cmd {
                        NetCommand::SendPlayerState { position, yaw, pitch, on_ground } => {
                            let _ = su.send_to(&encode_position_udp(my_id, position, yaw, pitch, on_ground), server_udp);
                        }
                        NetCommand::RequestBlockChange { wx, wy, wz, block } => {
                            let _ = tcp_send(&mut t, PacketType::BlockChange, &encode_block_change(wx, wy, wz, block));
                        }
                        NetCommand::Disconnect => {
                            let _ = tcp_send(&mut t, PacketType::Disconnect, &[]);
                            ss.store(true, Ordering::Release); break;
                        }
                        _ => {}
                    },
                    Err(channel::RecvTimeoutError::Timeout) => continue,
                    Err(_) => break,
                }
            }
        })?;

        Ok((cmd_tx, event_rx))
    }

    fn decode_tcp(ptype: PacketType, data: &[u8]) -> Option<NetEvent> {
        match ptype {
            PacketType::PlayerJoin => {
                let mut off = 0;
                Some(NetEvent::PlayerJoined { id: read_u8(data, &mut off), name: read_string(data, &mut off) })
            }
            PacketType::PlayerLeave => {
                let mut off = 0;
                Some(NetEvent::PlayerLeft { id: read_u8(data, &mut off) })
            }
            PacketType::BlockChange => {
                let (wx,wy,wz,b) = decode_block_change(data)?;
                Some(NetEvent::BlockChange { wx, wy, wz, block: b })
            }
            PacketType::ChunkResponse if data.len() >= 8 => {
                let mut off = 0;
                let cx = read_i32(data, &mut off);
                let cz = read_i32(data, &mut off);
                Some(NetEvent::ChunkData { pos: ChunkPos::new(cx,cz), blocks: rle_decode_chunk(&data[off..])? })
            }
            _ => None,
        }
    }
}

// =====================================================================
//  NETWORK HANDLE — Unified game loop interface
// =====================================================================

pub enum NetworkHandle {
    Host { cmd_tx: Sender<NetCommand>, event_rx: Receiver<NetEvent>, _thread: thread::JoinHandle<()> },
    Client { cmd_tx: Sender<NetCommand>, event_rx: Receiver<NetEvent> },
    Offline,
}

impl NetworkHandle {
    pub fn host(seed: u32) -> io::Result<Self> {
        let (tx, rx, h) = Server::start(seed, DEFAULT_TCP_PORT, DEFAULT_UDP_PORT)?;
        Ok(Self::Host { cmd_tx: tx, event_rx: rx, _thread: h })
    }

    pub fn connect(ip: &str, name: &str) -> io::Result<Self> {
        let (tx, rx) = Client::connect(ip, name, DEFAULT_TCP_PORT, DEFAULT_UDP_PORT)?;
        Ok(Self::Client { cmd_tx: tx, event_rx: rx })
    }

    pub fn drain_events(&self) -> Vec<NetEvent> {
        let rx = match self {
            Self::Host { event_rx, .. } | Self::Client { event_rx, .. } => event_rx,
            Self::Offline => return vec![],
        };
        let mut out = Vec::new();
        while let Ok(e) = rx.try_recv() { out.push(e); }
        out
    }

    pub fn send(&self, cmd: NetCommand) {
        match self {
            Self::Host { cmd_tx, .. } | Self::Client { cmd_tx, .. } => { let _ = cmd_tx.send(cmd); }
            Self::Offline => {}
        }
    }

    pub fn send_player_state(&self, pos: [f32;3], yaw: f32, pitch: f32, on_ground: bool) {
        self.send(NetCommand::SendPlayerState { position: pos, yaw, pitch, on_ground });
    }

    pub fn request_block_change(&self, wx: i32, wy: i32, wz: i32, block: BlockType) {
        self.send(NetCommand::RequestBlockChange { wx, wy, wz, block });
    }

    pub fn send_time_sync(&self, game_time: f32) {
        self.send(NetCommand::SendTimeSync { game_time });
    }

    pub fn is_online(&self) -> bool { !matches!(self, Self::Offline) }
    pub fn is_host(&self) -> bool { matches!(self, Self::Host { .. }) }
    pub fn disconnect(&self) { self.send(NetCommand::Disconnect); }
}