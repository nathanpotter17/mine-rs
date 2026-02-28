use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::thread;

// ===== Block Types =====

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BlockType {
    Air = 0, Grass = 1, Dirt = 2, Stone = 3, Sand = 4, Water = 5,
    Wood = 6, Leaves = 7, Snow = 8, Bedrock = 9, Gravel = 10,
    Coal = 11, Iron = 12, Gold = 13, Diamond = 14, Torch = 15,
}

impl BlockType {
    #[inline(always)]
    pub fn is_solid(self) -> bool {
        const S: [bool; 16] = [false,true,true,true,true,false,true,true,true,true,true,true,true,true,true,false];
        S[self as usize]
    }
    #[inline(always)]
    pub fn is_transparent(self) -> bool {
        const T: [bool; 16] = [true,false,false,false,false,true,false,true,false,false,false,false,false,false,false,true];
        T[self as usize]
    }
    pub fn is_light_source(self) -> bool { matches!(self, BlockType::Torch) }
    pub fn light_level(self) -> u8 { match self { BlockType::Torch => 15, _ => 0 } }

    /// Tile index into the 16×16 texture atlas.
    /// Row 0 (tiles 0-15)  = top faces
    /// Row 1 (tiles 16-31) = side faces
    /// Row 2 (tiles 32-47) = bottom faces
    #[inline(always)]
    pub fn tile_index(self, face: Face) -> u8 {
        let base = self as u8;
        match face {
            Face::Top    => base,       // row 0
            Face::Bottom => 32 + base,  // row 2
            _            => 16 + base,  // row 1: all sides
        }
    }

    /// Per-face tint color that modulates the atlas texture.
    /// These are intentionally brighter than before since the texture provides
    /// its own color variation — the tint is for face-dependent shading and biome hints.
    pub fn colors(self) -> ([f32; 3], [f32; 3], [f32; 3]) {
        match self {
            BlockType::Grass   => ([0.30,0.65,0.20],[0.45,0.35,0.20],[0.45,0.32,0.18]),
            BlockType::Dirt    => ([0.45,0.32,0.18],[0.45,0.32,0.18],[0.45,0.32,0.18]),
            BlockType::Stone   => ([0.50,0.50,0.52],[0.48,0.48,0.50],[0.46,0.46,0.48]),
            BlockType::Sand    => ([0.85,0.80,0.55],[0.82,0.77,0.52],[0.80,0.75,0.50]),
            BlockType::Water   => ([0.15,0.35,0.75],[0.12,0.30,0.70],[0.10,0.25,0.65]),
            BlockType::Wood    => ([0.55,0.40,0.22],[0.50,0.35,0.18],[0.55,0.40,0.22]),
            BlockType::Leaves  => ([0.20,0.55,0.15],[0.18,0.50,0.12],[0.20,0.55,0.15]),
            BlockType::Snow    => ([0.95,0.97,1.00],[0.90,0.92,0.95],[0.88,0.90,0.93]),
            BlockType::Bedrock => ([0.20,0.20,0.22],[0.18,0.18,0.20],[0.15,0.15,0.17]),
            BlockType::Gravel  => ([0.55,0.52,0.48],[0.52,0.49,0.45],[0.50,0.47,0.43]),
            BlockType::Coal    => ([0.25,0.25,0.28],[0.22,0.22,0.25],[0.20,0.20,0.23]),
            BlockType::Iron    => ([0.60,0.55,0.50],[0.58,0.52,0.47],[0.55,0.50,0.45]),
            BlockType::Gold    => ([0.90,0.78,0.20],[0.85,0.73,0.18],[0.80,0.68,0.15]),
            BlockType::Diamond => ([0.45,0.85,0.90],[0.40,0.80,0.85],[0.35,0.75,0.80]),
            BlockType::Torch   => ([0.95,0.85,0.45],[0.90,0.75,0.30],[0.90,0.75,0.30]),
            BlockType::Air     => ([0.0;3],[0.0;3],[0.0;3]),
        }
    }
    pub fn color_for_face(self, face: Face) -> [f32; 3] {
        let (top, side, bottom) = self.colors();
        match face { Face::Top => top, Face::Bottom => bottom, _ => side }
    }
}

// ===== Faces =====

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Face { Top = 0, Bottom = 1, North = 2, South = 3, East = 4, West = 5 }

impl Face {
    #[inline(always)]
    pub fn normal(self) -> [f32; 3] {
        const N: [[f32;3];6] = [[0.0,1.0,0.0],[0.0,-1.0,0.0],[0.0,0.0,1.0],[0.0,0.0,-1.0],[1.0,0.0,0.0],[-1.0,0.0,0.0]];
        N[self as usize]
    }
    #[inline(always)]
    pub fn offset(self) -> [i32; 3] {
        const O: [[i32;3];6] = [[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[1,0,0],[-1,0,0]];
        O[self as usize]
    }
    pub const ALL: [Face; 6] = [Face::Top, Face::Bottom, Face::North, Face::South, Face::East, Face::West];
}

// ===== Chunk =====

pub const CHUNK_X: usize = 16;
pub const CHUNK_Y: usize = 128;
pub const CHUNK_Z: usize = 16;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ChunkPos { pub x: i32, pub z: i32 }

impl ChunkPos {
    pub fn new(x: i32, z: i32) -> Self { Self { x, z } }
    pub fn world_offset(&self) -> (f32, f32) {
        (self.x as f32 * CHUNK_X as f32, self.z as f32 * CHUNK_Z as f32)
    }
}

pub struct Chunk {
    blocks: Vec<BlockType>,
    light_map: Vec<u8>,
    pub pos: ChunkPos,
    pub dirty: bool,
    pub light_dirty: bool,
    pub min_y: usize,
    pub max_y: usize,
}

impl Chunk {
    pub fn new(pos: ChunkPos) -> Self {
        Self {
            blocks: vec![BlockType::Air; CHUNK_X * CHUNK_Y * CHUNK_Z],
            light_map: vec![0u8; CHUNK_X * CHUNK_Y * CHUNK_Z],
            pos, dirty: true, light_dirty: true, min_y: CHUNK_Y, max_y: 0,
        }
    }

    #[inline(always)]
    fn index(x: usize, y: usize, z: usize) -> usize { y * CHUNK_X * CHUNK_Z + z * CHUNK_X + x }

    #[inline(always)]
    pub fn get(&self, x: usize, y: usize, z: usize) -> BlockType {
        if x < CHUNK_X && y < CHUNK_Y && z < CHUNK_Z {
            unsafe { *self.blocks.get_unchecked(Self::index(x, y, z)) }
        } else { BlockType::Air }
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, block: BlockType) {
        if x < CHUNK_X && y < CHUNK_Y && z < CHUNK_Z {
            self.blocks[Self::index(x, y, z)] = block;
            self.dirty = true;
            self.light_dirty = true;
            if block != BlockType::Air {
                if y < self.min_y { self.min_y = y; }
                if y > self.max_y { self.max_y = y; }
            }
        }
    }

    #[inline(always)]
    pub fn get_light(&self, x: usize, y: usize, z: usize) -> u8 {
        if x < CHUNK_X && y < CHUNK_Y && z < CHUNK_Z {
            unsafe { *self.light_map.get_unchecked(Self::index(x, y, z)) }
        } else { 0 }
    }

    #[inline(always)]
    pub fn set_light(&mut self, x: usize, y: usize, z: usize, level: u8) {
        if x < CHUNK_X && y < CHUNK_Y && z < CHUNK_Z {
            unsafe { *self.light_map.get_unchecked_mut(Self::index(x, y, z)) = level; }
        }
    }

    pub fn propagate_light(&mut self) {
        self.light_map.fill(0);
        let mut queue: VecDeque<(usize, usize, usize, u8)> = VecDeque::with_capacity(512);
        let scan_max = (self.max_y + 15).min(CHUNK_Y - 1);
        for y in self.min_y.saturating_sub(1)..=scan_max {
            for z in 0..CHUNK_Z { for x in 0..CHUNK_X {
                let block = self.get(x, y, z);
                if block.is_light_source() {
                    let level = block.light_level();
                    self.set_light(x, y, z, level);
                    queue.push_back((x, y, z, level));
                }
            }}
        }
        for z in 0..CHUNK_Z { for x in 0..CHUNK_X {
            let mut in_sky = true;
            for y in (0..CHUNK_Y).rev() {
                if in_sky {
                    if self.get(x, y, z).is_solid() { in_sky = false; }
                    else {
                        let sky_level = 12u8;
                        if sky_level > self.get_light(x, y, z) {
                            self.set_light(x, y, z, sky_level);
                            queue.push_back((x, y, z, sky_level));
                        }
                    }
                }
            }
        }}
        const OFF: [(i32,i32,i32);6] = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)];
        while let Some((x, y, z, level)) = queue.pop_front() {
            if level <= 1 { continue; }
            let new_level = level - 1;
            for &(dx, dy, dz) in &OFF {
                let (nx, ny, nz) = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
                if nx < 0 || nx >= CHUNK_X as i32 || ny < 0 || ny >= CHUNK_Y as i32
                    || nz < 0 || nz >= CHUNK_Z as i32 { continue; }
                let (ux, uy, uz) = (nx as usize, ny as usize, nz as usize);
                if !self.get(ux, uy, uz).is_solid() && self.get_light(ux, uy, uz) < new_level {
                    self.set_light(ux, uy, uz, new_level);
                    queue.push_back((ux, uy, uz, new_level));
                }
            }
        }
        self.light_dirty = false;
    }
}

// ===== Mesh Vertex =====
// Compact 20-byte vertex (unchanged size from pre-texture version)
//   position:     [f32; 3]  = 12 bytes
//   color_ao:     u32       = 4 bytes  (R8G8B8 tint color + A8 AO, UNORM → auto 0.0-1.0)
//   normal_light: u32       = 4 bytes  (R8=normal_idx, G8=light_u8, B8=tile_index, A8=uv_corner)

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockVertex {
    pub position: [f32; 3],
    pub color_ao: u32,
    pub normal_light: u32,
}

impl BlockVertex {
    /// Pack tint color, AO, normal, light, tile_index, uv_corner into 20 bytes.
    ///
    /// - `color`: [f32;3] tint that modulates the atlas texture
    /// - `normal_idx`: 0-5 (Face enum order)
    /// - `ao`: 0.0-1.0 ambient occlusion
    /// - `light`: 0.0-1.0 light level
    /// - `tile_index`: 0-255 atlas tile (row*16 + col in 16×16 grid)
    /// - `uv_corner`: 0-3 which quad corner (0=TL, 1=TR, 2=BR, 3=BL)
    #[inline]
    pub fn new(
        position: [f32; 3], color: [f32; 3], normal_idx: u8,
        ao: f32, light: f32, tile_index: u8, uv_corner: u8,
    ) -> Self {
        let r = (color[0] * 255.0 + 0.5) as u8;
        let g = (color[1] * 255.0 + 0.5) as u8;
        let b = (color[2] * 255.0 + 0.5) as u8;
        let a = (ao.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        let color_ao = (r as u32) | ((g as u32) << 8) | ((b as u32) << 16) | ((a as u32) << 24);
        let light_u8 = (light.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        let normal_light = (normal_idx as u32)
            | ((light_u8 as u32) << 8)
            | ((tile_index as u32) << 16)
            | ((uv_corner as u32) << 24);
        Self { position, color_ao, normal_light }
    }
}

pub struct ChunkMesh {
    pub vertices: Vec<BlockVertex>,
    pub indices: Vec<u32>,
    pub pos: ChunkPos,
}

// ===== Neighbor lookup =====

pub struct ChunkNeighbors<'a> {
    pub north: Option<&'a Chunk>, pub south: Option<&'a Chunk>,
    pub east: Option<&'a Chunk>, pub west: Option<&'a Chunk>,
    pub northeast: Option<&'a Chunk>, pub northwest: Option<&'a Chunk>,
    pub southeast: Option<&'a Chunk>, pub southwest: Option<&'a Chunk>,
}

#[inline(always)]
fn resolve_chunk<'a>(chunk: &'a Chunk, neighbors: &'a ChunkNeighbors<'a>, x: i32, z: i32) -> (Option<&'a Chunk>, usize, usize) {
    let (xn, xp, zn, zp) = (x < 0, x >= CHUNK_X as i32, z < 0, z >= CHUNK_Z as i32);
    let lx = if xn {(CHUNK_X as i32+x) as usize} else if xp {(x-CHUNK_X as i32) as usize} else {x as usize};
    let lz = if zn {(CHUNK_Z as i32+z) as usize} else if zp {(z-CHUNK_Z as i32) as usize} else {z as usize};
    let t = match (xn,xp,zn,zp) {
        (false,false,false,false) => Some(chunk),
        (true,false,false,false)  => neighbors.west,  (false,true,false,false)  => neighbors.east,
        (false,false,true,false)  => neighbors.south,  (false,false,false,true)  => neighbors.north,
        (true,false,true,false)   => neighbors.southwest, (true,false,false,true)   => neighbors.northwest,
        (false,true,true,false)   => neighbors.southeast, (false,true,false,true)   => neighbors.northeast,
        _ => None,
    };
    (t, lx, lz)
}

#[inline(always)]
fn get_block_with_neighbors(chunk: &Chunk, neighbors: &ChunkNeighbors, x: i32, y: i32, z: i32) -> BlockType {
    if y < 0 || y >= CHUNK_Y as i32 { return BlockType::Air; }
    let (t, lx, lz) = resolve_chunk(chunk, neighbors, x, z);
    t.map_or(BlockType::Air, |c| c.get(lx, y as usize, lz))
}

#[inline(always)]
fn get_light_with_neighbors(chunk: &Chunk, neighbors: &ChunkNeighbors, x: i32, y: i32, z: i32) -> u8 {
    if y < 0 { return 0; } if y >= CHUNK_Y as i32 { return 12; }
    let (t, lx, lz) = resolve_chunk(chunk, neighbors, x, z);
    t.map_or(0, |c| c.get_light(lx, y as usize, lz))
}

// ===== Precomputed AO + Light Tables =====

const AO_LUT: [u8; 8] = { let mut l = [0u8;8]; let mut i=0u8; while i<8 {
    let (s1,s2,c) = ((i>>2)&1, (i>>1)&1, i&1);
    l[i as usize] = if s1==1 && s2==1 {0} else {3-(s1+s2+c)}; i+=1; } l };

static AO_OFFSETS: [[([i32;3],[i32;3],[i32;3]);4];6] = [
    [([0,1,-1],[-1,1,0],[-1,1,-1]),([0,1,-1],[1,1,0],[1,1,-1]),([0,1,1],[1,1,0],[1,1,1]),([0,1,1],[-1,1,0],[-1,1,1])],
    [([0,-1,1],[-1,-1,0],[-1,-1,1]),([0,-1,1],[1,-1,0],[1,-1,1]),([0,-1,-1],[1,-1,0],[1,-1,-1]),([0,-1,-1],[-1,-1,0],[-1,-1,-1])],
    [([1,0,1],[0,-1,1],[1,-1,1]),([-1,0,1],[0,-1,1],[-1,-1,1]),([-1,0,1],[0,1,1],[-1,1,1]),([1,0,1],[0,1,1],[1,1,1])],
    [([-1,0,-1],[0,-1,-1],[-1,-1,-1]),([1,0,-1],[0,-1,-1],[1,-1,-1]),([1,0,-1],[0,1,-1],[1,1,-1]),([-1,0,-1],[0,1,-1],[-1,1,-1])],
    [([1,0,-1],[1,-1,0],[1,-1,-1]),([1,0,1],[1,-1,0],[1,-1,1]),([1,0,1],[1,1,0],[1,1,1]),([1,0,-1],[1,1,0],[1,1,-1])],
    [([-1,0,1],[-1,-1,0],[-1,-1,1]),([-1,0,-1],[-1,-1,0],[-1,-1,-1]),([-1,0,-1],[-1,1,0],[-1,1,-1]),([-1,0,1],[-1,1,0],[-1,1,1])],
];

static LIGHT_OFFSETS: [[[(i32,i32,i32);3];4];6] = [
    [[(-1,0,0),(0,0,-1),(-1,0,-1)],[(1,0,0),(0,0,-1),(1,0,-1)],[(1,0,0),(0,0,1),(1,0,1)],[(-1,0,0),(0,0,1),(-1,0,1)]],
    [[(-1,0,0),(0,0,1),(-1,0,1)],[(1,0,0),(0,0,1),(1,0,1)],[(1,0,0),(0,0,-1),(1,0,-1)],[(-1,0,0),(0,0,-1),(-1,0,-1)]],
    [[(1,0,0),(0,-1,0),(1,-1,0)],[(-1,0,0),(0,-1,0),(-1,-1,0)],[(-1,0,0),(0,1,0),(-1,1,0)],[(1,0,0),(0,1,0),(1,1,0)]],
    [[(-1,0,0),(0,-1,0),(-1,-1,0)],[(1,0,0),(0,-1,0),(1,-1,0)],[(1,0,0),(0,1,0),(1,1,0)],[(-1,0,0),(0,1,0),(-1,1,0)]],
    [[(0,0,-1),(0,-1,0),(0,-1,-1)],[(0,0,1),(0,-1,0),(0,-1,1)],[(0,0,1),(0,1,0),(0,1,1)],[(0,0,-1),(0,1,0),(0,1,-1)]],
    [[(0,0,1),(0,-1,0),(0,-1,1)],[(0,0,-1),(0,-1,0),(0,-1,-1)],[(0,0,-1),(0,1,0),(0,1,-1)],[(0,0,1),(0,1,0),(0,1,1)]],
];

const INV_3: f32 = 1.0 / 3.0;
const INV_LX15: f32 = 1.0 / (4.0 * 15.0);

// ===== Generic AO/Light computation =====

#[inline]
fn compute_ao_generic(x: i32, y: i32, z: i32, face: Face, get: &impl Fn(i32,i32,i32)->BlockType) -> [u8;4] {
    let s = &AO_OFFSETS[face as usize];
    let mut ao = [0u8;4];
    for (i,(e1,e2,cr)) in s.iter().enumerate() {
        let s1 = get(x+e1[0],y+e1[1],z+e1[2]).is_solid() as u8;
        let s2 = get(x+e2[0],y+e2[1],z+e2[2]).is_solid() as u8;
        let c  = get(x+cr[0],y+cr[1],z+cr[2]).is_solid() as u8;
        ao[i] = AO_LUT[((s1<<2)|(s2<<1)|c) as usize];
    }
    ao
}

#[inline]
fn compute_light_generic(x: i32, y: i32, z: i32, face: Face, get: &impl Fn(i32,i32,i32)->u8) -> [f32;4] {
    let off = face.offset();
    let (fx,fy,fz) = (x+off[0], y+off[1], z+off[2]);
    let center = get(fx,fy,fz) as u32;
    let offsets = &LIGHT_OFFSETS[face as usize];
    let mut lights = [0.0f32;4];
    for (i, samples) in offsets.iter().enumerate() {
        let mut total = center;
        for &(sx,sy,sz) in samples { total += get(fx+sx,fy+sy,fz+sz) as u32; }
        lights[i] = total as f32 * INV_LX15;
    }
    lights
}

// ===== Mesh geometry emission =====

fn emit_torch(verts: &mut Vec<BlockVertex>, idxs: &mut Vec<u32>, x: f32, y: f32, z: f32, _light_val: f32) {
    // Proper 3D box torch: 4 sides + top. No cross-billboard nonsense.
    // Self-illuminated: AO=1.0, light=1.0 for full brightness.
    let color = [0.95, 0.85, 0.50];
    let cx = x + 0.5; let cz = z + 0.5;
    let r = 0.0625;    // half-width: 1 pixel at 16px/block
    let h = 0.625;     // 10/16 of a block tall
    let tile = BlockType::Torch.tile_index(Face::South);

    // Helper: emit one quad with correct winding
    let mut quad = |v0: [f32;3], v1: [f32;3], v2: [f32;3], v3: [f32;3], ni: u8, c: [f32;3], ti: u8| {
        let base = verts.len() as u32;
        verts.push(BlockVertex::new(v0, c, ni, 1.0, 1.0, ti, 0));
        verts.push(BlockVertex::new(v1, c, ni, 1.0, 1.0, ti, 1));
        verts.push(BlockVertex::new(v2, c, ni, 1.0, 1.0, ti, 2));
        verts.push(BlockVertex::new(v3, c, ni, 1.0, 1.0, ti, 3));
        idxs.extend_from_slice(&[base, base+1, base+2, base+2, base+3, base]);
    };

    let (x0, x1) = (cx - r, cx + r);
    let (z0, z1) = (cz - r, cz + r);
    let (y0, y1) = (y, y + h);

    // North face (+Z, normal_idx=2)
    quad([x1,y0,z1],[x0,y0,z1],[x0,y1,z1],[x1,y1,z1], 2, color, tile);
    // South face (-Z, normal_idx=3)
    quad([x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0], 3, color, tile);
    // East face (+X, normal_idx=4)
    quad([x1,y0,z0],[x1,y0,z1],[x1,y1,z1],[x1,y1,z0], 4, color, tile);
    // West face (-X, normal_idx=5)
    quad([x0,y0,z1],[x0,y0,z0],[x0,y1,z0],[x0,y1,z1], 5, color, tile);

    // Top face — flame (brighter orange, normal_idx=0)
    let flame = [1.0, 0.7, 0.2];
    let flame_tile = BlockType::Torch.tile_index(Face::Top);
    quad([x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1], 0, flame, flame_tile);
}

fn add_face_ao(
    verts: &mut Vec<BlockVertex>, idxs: &mut Vec<u32>,
    x: f32, y: f32, z: f32, face: Face, color: [f32;3],
    ao: [f32;4], light: [f32;4], ao_raw: [u8;4], tile_index: u8,
) {
    let base = verts.len() as u32;
    let ni = face as u8;
    let (v0,v1,v2,v3) = match face {
        Face::Top    => ([x,y+1.0,z],[x+1.0,y+1.0,z],[x+1.0,y+1.0,z+1.0],[x,y+1.0,z+1.0]),
        Face::Bottom => ([x,y,z+1.0],[x+1.0,y,z+1.0],[x+1.0,y,z],[x,y,z]),
        Face::North  => ([x+1.0,y,z+1.0],[x,y,z+1.0],[x,y+1.0,z+1.0],[x+1.0,y+1.0,z+1.0]),
        Face::South  => ([x,y,z],[x+1.0,y,z],[x+1.0,y+1.0,z],[x,y+1.0,z]),
        Face::East   => ([x+1.0,y,z],[x+1.0,y,z+1.0],[x+1.0,y+1.0,z+1.0],[x+1.0,y+1.0,z]),
        Face::West   => ([x,y,z+1.0],[x,y,z],[x,y+1.0,z],[x,y+1.0,z+1.0]),
    };
    // uv_corner 0-3 maps to quad corners: v0=TL, v1=TR, v2=BR, v3=BL
    verts.push(BlockVertex::new(v0, color, ni, ao[0], light[0], tile_index, 0));
    verts.push(BlockVertex::new(v1, color, ni, ao[1], light[1], tile_index, 1));
    verts.push(BlockVertex::new(v2, color, ni, ao[2], light[2], tile_index, 2));
    verts.push(BlockVertex::new(v3, color, ni, ao[3], light[3], tile_index, 3));
    if ao_raw[0] + ao_raw[2] > ao_raw[1] + ao_raw[3] {
        idxs.extend_from_slice(&[base,base+1,base+2, base+2,base+3,base]);
    } else {
        idxs.extend_from_slice(&[base+1,base+2,base+3, base+3,base,base+1]);
    }
}

// ===== Meshing: generic inner loop =====

fn mesh_inner(
    verts: &mut Vec<BlockVertex>, idxs: &mut Vec<u32>,
    ox: f32, oz: f32, scan_min: usize, scan_max: usize,
    get_nb_block: impl Fn(i32,i32,i32)->BlockType,
    get_nb_light: impl Fn(i32,i32,i32)->u8,
    get_local: impl Fn(usize,usize,usize)->BlockType,
) {
    for y in scan_min..=scan_max { for z in 0..CHUNK_Z { for x in 0..CHUNK_X {
        let block = get_local(x, y, z);
        if block == BlockType::Air { continue; }
        let (wx, wy, wz) = (ox + x as f32, y as f32, oz + z as f32);
        let (ix, iy, iz) = (x as i32, y as i32, z as i32);
        if block == BlockType::Torch {
            emit_torch(verts, idxs, wx, wy, wz, (get_nb_light(ix,iy,iz) as f32 / 15.0).max(0.9));
            continue;
        }
        for face in Face::ALL {
            let off = face.offset();
            let nb = get_nb_block(ix+off[0], iy+off[1], iz+off[2]);
            if nb.is_transparent() && nb != block {
                let ao_raw = compute_ao_generic(ix,iy,iz,face,&get_nb_block);
                let ao = [ao_raw[0] as f32*INV_3, ao_raw[1] as f32*INV_3, ao_raw[2] as f32*INV_3, ao_raw[3] as f32*INV_3];
                let light = compute_light_generic(ix,iy,iz,face,&get_nb_light);
                let tile = block.tile_index(face);
                add_face_ao(verts, idxs, wx, wy, wz, face, block.color_for_face(face), ao, light, ao_raw, tile);
            }
        }
    }}}
}

/// On-thread meshing (startup + set_block immediate re-mesh)
pub fn mesh_chunk(chunk: &Chunk, neighbors: &ChunkNeighbors) -> ChunkMesh {
    let mut verts = Vec::with_capacity(1024);
    let mut idxs = Vec::with_capacity(2048);
    let (ox, oz) = chunk.pos.world_offset();
    let scan_min = chunk.min_y.saturating_sub(1);
    let scan_max = (chunk.max_y + 1).min(CHUNK_Y - 1);
    if scan_min <= scan_max {
        mesh_inner(&mut verts, &mut idxs, ox, oz, scan_min, scan_max,
            |x,y,z| get_block_with_neighbors(chunk, neighbors, x, y, z),
            |x,y,z| get_light_with_neighbors(chunk, neighbors, x, y, z),
            |x,y,z| chunk.get(x, y, z));
    }
    ChunkMesh { vertices: verts, indices: idxs, pos: chunk.pos }
}

// ===== Noise =====

pub struct Noise { seed: u32 }
impl Noise {
    pub fn new(seed: u32) -> Self { Self { seed } }
    fn hash2d(&self, x: i32, z: i32) -> f32 {
        let mut h = self.seed;
        h ^= x as u32; h = h.wrapping_mul(0x85ebca6b);
        h ^= z as u32; h = h.wrapping_mul(0xc2b2ae35);
        h ^= h >> 16;  h = h.wrapping_mul(0x45d9f3b);
        h ^= h >> 16;
        (h & 0xFFFF) as f32 / 65535.0
    }
    fn smoothed(&self, x: f32, z: f32) -> f32 {
        let ix = x.floor() as i32; let iz = z.floor() as i32;
        let fx = x - x.floor(); let fz = z - z.floor();
        let sx = fx*fx*(3.0-2.0*fx); let sz = fz*fz*(3.0-2.0*fz);
        let (v00,v10,v01,v11) = (self.hash2d(ix,iz),self.hash2d(ix+1,iz),self.hash2d(ix,iz+1),self.hash2d(ix+1,iz+1));
        let a = v00 + sx*(v10-v00); let b = v01 + sx*(v11-v01);
        a + sz*(b-a)
    }
    pub fn fbm(&self, x: f32, z: f32, octaves: u32) -> f32 {
        let (mut value, mut amplitude, mut frequency, mut max_val) = (0.0, 1.0, 1.0, 0.0);
        for _ in 0..octaves {
            value += amplitude * self.smoothed(x*frequency, z*frequency);
            max_val += amplitude; amplitude *= 0.5; frequency *= 2.0;
        }
        value / max_val
    }
}

// ===== Heightmap Cache =====

#[derive(Clone, Copy)]
struct ColumnData {
    height: usize, surface_block: BlockType, sub_surface_block: BlockType,
    is_water: bool, can_have_tree: bool,
}

const HEIGHTMAP_SHARDS: usize = 16;
struct HeightmapCache {
    shards: Vec<std::sync::Mutex<HashMap<(i32,i32), ColumnData>>>,
}

impl HeightmapCache {
    fn new() -> Self {
        Self { shards: (0..HEIGHTMAP_SHARDS).map(|_| std::sync::Mutex::new(HashMap::with_capacity(4096))).collect() }
    }
    #[inline]
    fn shard(wx: i32, wz: i32) -> usize {
        ((wx as u32).wrapping_mul(0x9e3779b9) ^ (wz as u32).wrapping_mul(0x517cc1b7)) as usize % HEIGHTMAP_SHARDS
    }
    fn get(&self, wx: i32, wz: i32) -> Option<ColumnData> {
        self.shards[Self::shard(wx,wz)].lock().unwrap().get(&(wx,wz)).copied()
    }
    fn precompute_chunk(&self, pos: ChunkPos, noise: &Noise, tree_noise: &Noise) {
        for z in 0..CHUNK_Z { for x in 0..CHUNK_X {
            let wx = pos.x as i32 * CHUNK_X as i32 + x as i32;
            let wz = pos.z as i32 * CHUNK_Z as i32 + z as i32;
            let si = Self::shard(wx, wz);
            { let s = self.shards[si].lock().unwrap(); if s.contains_key(&(wx,wz)) { continue; } }
            let data = compute_column(wx as f32, wz as f32, x, z, noise, tree_noise);
            self.shards[si].lock().unwrap().insert((wx,wz), data);
        }}
    }
    fn evict_far(&self, pcx: i32, pcz: i32, keep: i32) {
        let (mnx,mxx) = ((pcx-keep)*CHUNK_X as i32, (pcx+keep+1)*CHUNK_X as i32);
        let (mnz,mxz) = ((pcz-keep)*CHUNK_Z as i32, (pcz+keep+1)*CHUNK_Z as i32);
        for s in &self.shards { s.lock().unwrap().retain(|&(wx,wz),_| wx>=mnx && wx<mxx && wz>=mnz && wz<mxz); }
    }
}

fn compute_column(wx: f32, wz: f32, lx: usize, lz: usize, noise: &Noise, tree_noise: &Noise) -> ColumnData {
    let bh = noise.fbm(wx*0.008, wz*0.008, 4);
    let detail = noise.fbm(wx*0.03, wz*0.03, 3) * 0.3;
    let mountains = noise.fbm(wx*0.004, wz*0.004, 3);
    let mf = (mountains - 0.5).max(0.0) * 2.0;
    let height = ((bh*30.0 + detail*8.0 + mf*40.0 + 20.0) as usize).min(CHUNK_Y-1);
    let surface_block = if height < SEA_LEVEL-1 { BlockType::Sand } else if height > 70 { BlockType::Snow } else { BlockType::Grass };
    let sub_surface_block = if height < SEA_LEVEL-1 { BlockType::Sand } else { BlockType::Dirt };
    let is_water = height + 1 < SEA_LEVEL;
    let can_have_tree = height >= SEA_LEVEL && height < 65 && {
        let tv = tree_noise.hash2d(wx as i32, wz as i32);
        tv < 0.012 && lx >= 2 && lx < CHUNK_X-2 && lz >= 2 && lz < CHUNK_Z-2
    };
    ColumnData { height, surface_block, sub_surface_block, is_water, can_have_tree }
}

// ===== Chunk Snapshot for off-thread meshing =====

struct ChunkSnapshot {
    blocks: Vec<BlockType>, light_map: Vec<u8>,
    pos: ChunkPos, min_y: usize, max_y: usize,
}

impl ChunkSnapshot {
    fn from_chunk(c: &Chunk) -> Self {
        Self { blocks: c.blocks.clone(), light_map: c.light_map.clone(), pos: c.pos, min_y: c.min_y, max_y: c.max_y }
    }
    #[inline(always)]
    fn get(&self, x: usize, y: usize, z: usize) -> BlockType {
        if x < CHUNK_X && y < CHUNK_Y && z < CHUNK_Z {
            unsafe { *self.blocks.get_unchecked(Chunk::index(x,y,z)) }
        } else { BlockType::Air }
    }
    #[inline(always)]
    fn get_light(&self, x: usize, y: usize, z: usize) -> u8 {
        if x < CHUNK_X && y < CHUNK_Y && z < CHUNK_Z {
            unsafe { *self.light_map.get_unchecked(Chunk::index(x,y,z)) }
        } else { 0 }
    }
}

struct MeshJob {
    center: ChunkSnapshot,
    north: Option<ChunkSnapshot>, south: Option<ChunkSnapshot>,
    east: Option<ChunkSnapshot>,  west: Option<ChunkSnapshot>,
    northeast: Option<ChunkSnapshot>, northwest: Option<ChunkSnapshot>,
    southeast: Option<ChunkSnapshot>, southwest: Option<ChunkSnapshot>,
}

#[inline(always)]
fn snap_resolve<'a>(c: &'a ChunkSnapshot, n: &'a MeshJob, x: i32, z: i32) -> (Option<&'a ChunkSnapshot>, usize, usize) {
    let (xn,xp,zn,zp) = (x<0, x>=CHUNK_X as i32, z<0, z>=CHUNK_Z as i32);
    let lx = if xn {(CHUNK_X as i32+x) as usize} else if xp {(x-CHUNK_X as i32) as usize} else {x as usize};
    let lz = if zn {(CHUNK_Z as i32+z) as usize} else if zp {(z-CHUNK_Z as i32) as usize} else {z as usize};
    let t = match (xn,xp,zn,zp) {
        (false,false,false,false) => Some(c),
        (true,false,false,false)  => n.west.as_ref(),   (false,true,false,false)  => n.east.as_ref(),
        (false,false,true,false)  => n.south.as_ref(),  (false,false,false,true)  => n.north.as_ref(),
        (true,false,true,false)   => n.southwest.as_ref(), (true,false,false,true)   => n.northwest.as_ref(),
        (false,true,true,false)   => n.southeast.as_ref(), (false,true,false,true)   => n.northeast.as_ref(),
        _ => None,
    };
    (t, lx, lz)
}

fn mesh_from_snapshot(job: &MeshJob) -> ChunkMesh {
    let c = &job.center;
    let (ox, oz) = c.pos.world_offset();
    let mut verts = Vec::with_capacity(1024);
    let mut idxs = Vec::with_capacity(2048);
    let scan_min = c.min_y.saturating_sub(1);
    let scan_max = (c.max_y + 1).min(CHUNK_Y - 1);
    if scan_min <= scan_max {
        mesh_inner(&mut verts, &mut idxs, ox, oz, scan_min, scan_max,
            |x,y,z| {
                if y<0||y>=CHUNK_Y as i32 { return BlockType::Air; }
                let (t,lx,lz) = snap_resolve(c, job, x, z);
                t.map_or(BlockType::Air, |s| s.get(lx, y as usize, lz))
            },
            |x,y,z| {
                if y<0 { return 0; } if y>=CHUNK_Y as i32 { return 12; }
                let (t,lx,lz) = snap_resolve(c, job, x, z);
                t.map_or(0, |s| s.get_light(lx, y as usize, lz))
            },
            |x,y,z| c.get(x,y,z));
    }
    ChunkMesh { vertices: verts, indices: idxs, pos: c.pos }
}

// ===== Background Generation Pool =====

struct ChunkGenRequest { pos: ChunkPos }
struct ChunkGenResult { chunk: Chunk }

pub struct ChunkGenPool {
    tx: crossbeam::channel::Sender<ChunkGenRequest>,
    rx: crossbeam::channel::Receiver<ChunkGenResult>,
    in_flight: HashMap<ChunkPos, ()>,
    _workers: Vec<thread::JoinHandle<()>>,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
}

impl ChunkGenPool {
    pub fn new(n: usize, seed: u32, hm: Arc<HeightmapCache>) -> Self {
        let (req_tx, req_rx) = crossbeam::channel::bounded::<ChunkGenRequest>(64);
        let (res_tx, res_rx) = crossbeam::channel::bounded::<ChunkGenResult>(64);
        let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut workers = Vec::with_capacity(n);
        for _ in 0..n {
            let (rx, tx, h, sd) = (req_rx.clone(), res_tx.clone(), hm.clone(), shutdown.clone());
            let (noise, tn) = (Noise::new(seed), Noise::new(seed.wrapping_add(12345)));
            workers.push(thread::Builder::new().name("chunk-gen".into()).spawn(move || {
                while !sd.load(std::sync::atomic::Ordering::Relaxed) {
                    match rx.recv_timeout(std::time::Duration::from_millis(50)) {
                        Ok(req) => {
                            h.precompute_chunk(req.pos, &noise, &tn);
                            let mut chunk = gen_chunk_cached(req.pos, &noise, &tn, &h);
                            chunk.propagate_light();
                            let _ = tx.send(ChunkGenResult { chunk });
                        }
                        Err(crossbeam::channel::RecvTimeoutError::Timeout) => continue,
                        Err(_) => break,
                    }
                }
            }).expect("spawn chunk-gen"));
        }
        Self { tx: req_tx, rx: res_rx, in_flight: HashMap::new(), _workers: workers, shutdown }
    }
    pub fn request(&mut self, pos: ChunkPos) {
        if !self.in_flight.contains_key(&pos) {
            if self.tx.try_send(ChunkGenRequest { pos }).is_ok() { self.in_flight.insert(pos, ()); }
        }
    }
    pub fn drain(&mut self) -> Vec<Chunk> {
        let mut out = Vec::new();
        while let Ok(r) = self.rx.try_recv() { self.in_flight.remove(&r.chunk.pos); out.push(r.chunk); }
        out
    }
}

impl Drop for ChunkGenPool {
    fn drop(&mut self) {
        self.shutdown.store(true, std::sync::atomic::Ordering::Release);
    }
}

// ===== Background Mesh Pool =====

pub struct MeshPool {
    tx: crossbeam::channel::Sender<MeshJob>,
    rx: crossbeam::channel::Receiver<ChunkMesh>,
    in_flight: HashMap<ChunkPos, ()>,
    _workers: Vec<thread::JoinHandle<()>>,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
}

impl MeshPool {
    pub fn new(n: usize) -> Self {
        let (req_tx, req_rx) = crossbeam::channel::bounded::<MeshJob>(32);
        let (res_tx, res_rx) = crossbeam::channel::bounded::<ChunkMesh>(32);
        let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut workers = Vec::with_capacity(n);
        for _ in 0..n {
            let (rx, tx, sd) = (req_rx.clone(), res_tx.clone(), shutdown.clone());
            workers.push(thread::Builder::new().name("chunk-mesh".into()).spawn(move || {
                while !sd.load(std::sync::atomic::Ordering::Relaxed) {
                    match rx.recv_timeout(std::time::Duration::from_millis(50)) {
                        Ok(job) => { let _ = tx.send(mesh_from_snapshot(&job)); }
                        Err(crossbeam::channel::RecvTimeoutError::Timeout) => continue,
                        Err(_) => break,
                    }
                }
            }).expect("spawn chunk-mesh"));
        }
        Self { tx: req_tx, rx: res_rx, in_flight: HashMap::new(), _workers: workers, shutdown }
    }

    /// Returns true if the mesh job was actually sent to the worker pool.
    pub fn submit(&mut self, pos: ChunkPos, chunks: &HashMap<ChunkPos, Chunk>) -> bool {
        if self.in_flight.contains_key(&pos) { return true; } // already queued
        let center = match chunks.get(&pos) {
            Some(c) => ChunkSnapshot::from_chunk(c),
            None => return false,
        };
        let snap = |p: ChunkPos| chunks.get(&p).map(ChunkSnapshot::from_chunk);
        let job = MeshJob {
            center,
            north: snap(ChunkPos::new(pos.x,pos.z+1)), south: snap(ChunkPos::new(pos.x,pos.z-1)),
            east: snap(ChunkPos::new(pos.x+1,pos.z)),  west: snap(ChunkPos::new(pos.x-1,pos.z)),
            northeast: snap(ChunkPos::new(pos.x+1,pos.z+1)), northwest: snap(ChunkPos::new(pos.x-1,pos.z+1)),
            southeast: snap(ChunkPos::new(pos.x+1,pos.z-1)), southwest: snap(ChunkPos::new(pos.x-1,pos.z-1)),
        };
        if self.tx.try_send(job).is_ok() {
            self.in_flight.insert(pos, ());
            true
        } else {
            false  // channel full — caller must NOT clear dirty
        }
    }

    pub fn drain(&mut self) -> Vec<ChunkMesh> {
        let mut out = Vec::new();
        while let Ok(m) = self.rx.try_recv() { self.in_flight.remove(&m.pos); out.push(m); }
        out
    }

    pub fn is_in_flight(&self, pos: &ChunkPos) -> bool { self.in_flight.contains_key(pos) }
}

impl Drop for MeshPool {
    fn drop(&mut self) { self.shutdown.store(true, std::sync::atomic::Ordering::Release); }
}

// ===== Terrain generation from cached heightmap =====

fn gen_chunk_cached(pos: ChunkPos, noise: &Noise, _tn: &Noise, hm: &HeightmapCache) -> Chunk {
    let mut chunk = Chunk::new(pos);
    for z in 0..CHUNK_Z { for x in 0..CHUNK_X {
        let wx = pos.x as i32 * CHUNK_X as i32 + x as i32;
        let wz = pos.z as i32 * CHUNK_Z as i32 + z as i32;
        let col = hm.get(wx, wz).unwrap_or_else(|| {
            let n = Noise::new(noise.seed); let tn = Noise::new(noise.seed.wrapping_add(12345));
            compute_column(wx as f32, wz as f32, x, z, &n, &tn)
        });
        chunk.set(x, 0, z, BlockType::Bedrock);
        for y in 1..=col.height {
            let block = if y == col.height { col.surface_block }
                else if y > col.height.saturating_sub(4) { col.sub_surface_block }
                else if y < 5 { BlockType::Bedrock }
                else {
                    let on = noise.hash2d(wx + y as i32 * 137, wz + y as i32 * 251);
                    if y<12 && on<0.02 { BlockType::Diamond } else if y<32 && on<0.04 { BlockType::Gold }
                    else if y<48 && on<0.08 { BlockType::Iron } else if on<0.10 { BlockType::Coal }
                    else if on<0.14 { BlockType::Gravel } else { BlockType::Stone }
                };
            chunk.set(x, y, z, block);
        }
        if col.is_water { for y in (col.height+1)..SEA_LEVEL { chunk.set(x, y, z, BlockType::Water); } }
        if col.can_have_tree { place_tree(&mut chunk, x, col.height+1, z); }
    }}
    chunk
}

fn place_tree(chunk: &mut Chunk, x: usize, y: usize, z: usize) {
    let th = 5;
    for dy in 0..th { if y+dy < CHUNK_Y { chunk.set(x, y+dy, z, BlockType::Wood); } }
    for dy in (th-2)..(th+2) {
        let r = if dy >= th { 1 } else { 2 };
        for dz in -(r as i32)..=(r as i32) { for dx in -(r as i32)..=(r as i32) {
            if dx*dx+dz*dz <= r*r+1 {
                let (lx,ly,lz) = ((x as i32+dx) as usize, y+dy, (z as i32+dz) as usize);
                if lx<CHUNK_X && ly<CHUNK_Y && lz<CHUNK_Z && chunk.get(lx,ly,lz)==BlockType::Air {
                    chunk.set(lx, ly, lz, BlockType::Leaves);
                }
            }
        }}
    }
}

// ===== World =====

pub const RENDER_DISTANCE: i32 = 6;
pub const GENERATION_DISTANCE: i32 = RENDER_DISTANCE + 2;
pub const SEA_LEVEL: usize = 32;
const MAX_CHUNKS_PER_FRAME: usize = 4;
const GEN_WORKERS: usize = 2;
const MESH_WORKERS: usize = 2;
const HM_PREFETCH: i32 = GENERATION_DISTANCE + 3;

pub struct World {
    pub chunks: HashMap<ChunkPos, Chunk>,
    noise: Noise,
    tree_noise: Noise,
    heightmap: Arc<HeightmapCache>,
    gen_pool: ChunkGenPool,
    pub mesh_pool: MeshPool,
    cleanup_counter: u32,
    hm_prefetch_idx: i32,
}

impl World {
    pub fn new(seed: u32) -> Self {
        let hm = Arc::new(HeightmapCache::new());
        Self {
            chunks: HashMap::new(), noise: Noise::new(seed),
            tree_noise: Noise::new(seed.wrapping_add(12345)),
            heightmap: hm.clone(),
            gen_pool: ChunkGenPool::new(GEN_WORKERS, seed, hm),
            mesh_pool: MeshPool::new(MESH_WORKERS),
            cleanup_counter: 0,
            hm_prefetch_idx: 0,
        }
    }

    pub fn generate_around_immediate(&mut self, px: f32, pz: f32) {
        let cx = (px / CHUNK_X as f32).floor() as i32;
        let cz = (pz / CHUNK_Z as f32).floor() as i32;
        for dz in -GENERATION_DISTANCE..=GENERATION_DISTANCE {
            for dx in -GENERATION_DISTANCE..=GENERATION_DISTANCE {
                self.heightmap.precompute_chunk(ChunkPos::new(cx+dx,cz+dz), &self.noise, &self.tree_noise);
            }
        }
        for dz in -GENERATION_DISTANCE..=GENERATION_DISTANCE {
            for dx in -GENERATION_DISTANCE..=GENERATION_DISTANCE {
                let pos = ChunkPos::new(cx+dx, cz+dz);
                if !self.chunks.contains_key(&pos) {
                    let mut chunk = gen_chunk_cached(pos, &self.noise, &self.tree_noise, &self.heightmap);
                    chunk.propagate_light();
                    self.chunks.insert(pos, chunk);
                }
            }
        }
        self.remove_far_chunks(cx, cz);
    }

    pub fn generate_around(&mut self, px: f32, pz: f32) {
        let cx = (px / CHUNK_X as f32).floor() as i32;
        let cz = (pz / CHUNK_Z as f32).floor() as i32;

        let ta = std::time::Instant::now();

        for chunk in self.gen_pool.drain() {
            let pos = chunk.pos;
            self.chunks.insert(pos, chunk);
            for &(dx,dz) in &[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)] {
                if let Some(nb) = self.chunks.get_mut(&ChunkPos::new(pos.x+dx,pos.z+dz)) { nb.dirty = true; }
            }
        }

        let tb = std::time::Instant::now();

        let mut req = 0;
        'outer: for dist in 0..=GENERATION_DISTANCE { for dz in -dist..=dist { for dx in -dist..=dist {
            if dx.abs() != dist && dz.abs() != dist { continue; }
            let pos = ChunkPos::new(cx+dx, cz+dz);
            if !self.chunks.contains_key(&pos) {
                self.gen_pool.request(pos);
                req += 1;
                if req >= MAX_CHUNKS_PER_FRAME * 2 { break 'outer; }
            }
        }}}

        let tc = std::time::Instant::now();

        let mut light_count = 0;
        let light_dirty: Vec<ChunkPos> = self.chunks.iter()
            .filter(|(_,c)| c.light_dirty).map(|(p,_)| *p).take(2).collect();
        for pos in light_dirty {
            if let Some(chunk) = self.chunks.get_mut(&pos) {
                chunk.propagate_light();
                light_count += 1;
            }
        }

        let td = std::time::Instant::now();

        self.submit_mesh_jobs(cx, cz);

        let te = std::time::Instant::now();

        self.hm_prefetch_step(cx, cz);

        let tf = std::time::Instant::now();

        self.cleanup_counter += 1;
        if self.cleanup_counter >= 30 {
            self.cleanup_counter = 0;
            self.remove_far_chunks(cx, cz);
            self.heightmap.evict_far(cx, cz, HM_PREFETCH + 2);
        }

        // let tg = std::time::Instant::now();

        // let drain = (tb-ta).as_secs_f64()*1000.0;
        // let genreq = (tc-tb).as_secs_f64()*1000.0;
        // let light = (td-tc).as_secs_f64()*1000.0;
        // let mesh = (te-td).as_secs_f64()*1000.0;
        // let hmpre = (tf-te).as_secs_f64()*1000.0;
        // let clean = (tg-tf).as_secs_f64()*1000.0;
        // let total = (tg-ta).as_secs_f64()*1000.0;
        // if total > 2.0 {
        //     println!("    [world] drain={:.1} genreq={:.1} light={:.1} mesh={:.1} hmpre={:.1} clean={:.1} TOTAL={:.1}ms (lit={})",
        //         drain, genreq, light, mesh, hmpre, clean, total, light_count);
        // }
    }

    fn hm_prefetch_step(&mut self, cx: i32, cz: i32) {
        let ring_size = HM_PREFETCH * 2 + 1;
        let total = ring_size * ring_size;
        for _ in 0..1 {
            let idx = self.hm_prefetch_idx % total;
            self.hm_prefetch_idx += 1;
            let dx = (idx % ring_size) - HM_PREFETCH;
            let dz = (idx / ring_size) - HM_PREFETCH;
            if dx.abs() > GENERATION_DISTANCE || dz.abs() > GENERATION_DISTANCE {
                self.heightmap.precompute_chunk(
                    ChunkPos::new(cx + dx, cz + dz), &self.noise, &self.tree_noise);
            }
        }
    }

    fn submit_mesh_jobs(&mut self, pcx: i32, pcz: i32) {
        let mut dirty: Vec<ChunkPos> = self.chunks.iter()
            .filter(|(p,c)| c.dirty && !c.light_dirty
                && (p.x-pcx).abs() <= RENDER_DISTANCE && (p.z-pcz).abs() <= RENDER_DISTANCE
                && !self.mesh_pool.is_in_flight(p))
            .map(|(p,_)| *p).collect();
        dirty.sort_unstable_by_key(|p| { let (dx,dz) = (p.x-pcx, p.z-pcz); dx*dx + dz*dz });
        let mut submitted = 0;
        for pos in dirty {
            if submitted >= 4 { break; }
            if !self.has_all_neighbors(pos) { continue; }
            let actually_sent = self.mesh_pool.submit(pos, &self.chunks);
            if actually_sent {
                if let Some(c) = self.chunks.get_mut(&pos) { c.dirty = false; }
                submitted += 1;
            }
            // If not sent (channel full), dirty stays true → retried next frame
        }
    }

    pub fn drain_completed_meshes(&mut self) -> Vec<ChunkMesh> {
        self.mesh_pool.drain()
    }

    fn remove_far_chunks(&mut self, cx: i32, cz: i32) {
        self.chunks.retain(|pos, _| {
            (pos.x - cx).abs() <= GENERATION_DISTANCE + 2 && (pos.z - cz).abs() <= GENERATION_DISTANCE + 2
        });
    }

    pub fn repropagate_light_around(&mut self, cx: i32, cz: i32) {
        for pos in [ChunkPos::new(cx,cz),ChunkPos::new(cx-1,cz),ChunkPos::new(cx+1,cz),
                    ChunkPos::new(cx,cz-1),ChunkPos::new(cx,cz+1)] {
            if let Some(c) = self.chunks.get_mut(&pos) { c.propagate_light(); }
        }
    }

    pub fn get_block(&self, wx: i32, wy: i32, wz: i32) -> BlockType {
        if wy < 0 || wy >= CHUNK_Y as i32 { return BlockType::Air; }
        let cx = if wx < 0 { (wx+1)/CHUNK_X as i32-1 } else { wx/CHUNK_X as i32 };
        let cz = if wz < 0 { (wz+1)/CHUNK_Z as i32-1 } else { wz/CHUNK_Z as i32 };
        let lx = ((wx % CHUNK_X as i32) + CHUNK_X as i32) as usize % CHUNK_X;
        let lz = ((wz % CHUNK_Z as i32) + CHUNK_Z as i32) as usize % CHUNK_Z;
        self.chunks.get(&ChunkPos::new(cx,cz)).map(|c| c.get(lx, wy as usize, lz)).unwrap_or(BlockType::Air)
    }

    pub fn set_block(&mut self, wx: i32, wy: i32, wz: i32, block: BlockType) {
        if wy < 0 || wy >= CHUNK_Y as i32 { return; }
        let cx = if wx < 0 { (wx+1)/CHUNK_X as i32-1 } else { wx/CHUNK_X as i32 };
        let cz = if wz < 0 { (wz+1)/CHUNK_Z as i32-1 } else { wz/CHUNK_Z as i32 };
        let lx = ((wx % CHUNK_X as i32) + CHUNK_X as i32) as usize % CHUNK_X;
        let lz = ((wz % CHUNK_Z as i32) + CHUNK_Z as i32) as usize % CHUNK_Z;
        if let Some(c) = self.chunks.get_mut(&ChunkPos::new(cx,cz)) { c.set(lx, wy as usize, lz, block); }
        self.repropagate_light_around(cx, cz);
        if lx == 0          { self.mark_dirty(ChunkPos::new(cx-1, cz)); }
        if lx == CHUNK_X-1  { self.mark_dirty(ChunkPos::new(cx+1, cz)); }
        if lz == 0          { self.mark_dirty(ChunkPos::new(cx, cz-1)); }
        if lz == CHUNK_Z-1  { self.mark_dirty(ChunkPos::new(cx, cz+1)); }
    }

    fn mark_dirty(&mut self, pos: ChunkPos) {
        if let Some(c) = self.chunks.get_mut(&pos) { c.dirty = true; }
    }

    pub fn get_neighbors(&self, pos: ChunkPos) -> ChunkNeighbors {
        ChunkNeighbors {
            north: self.chunks.get(&ChunkPos::new(pos.x,pos.z+1)),
            south: self.chunks.get(&ChunkPos::new(pos.x,pos.z-1)),
            east:  self.chunks.get(&ChunkPos::new(pos.x+1,pos.z)),
            west:  self.chunks.get(&ChunkPos::new(pos.x-1,pos.z)),
            northeast: self.chunks.get(&ChunkPos::new(pos.x+1,pos.z+1)),
            northwest: self.chunks.get(&ChunkPos::new(pos.x-1,pos.z+1)),
            southeast: self.chunks.get(&ChunkPos::new(pos.x+1,pos.z-1)),
            southwest: self.chunks.get(&ChunkPos::new(pos.x-1,pos.z-1)),
        }
    }

    pub fn has_all_neighbors(&self, pos: ChunkPos) -> bool {
        [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)].iter()
            .all(|&(dx,dz)| self.chunks.contains_key(&ChunkPos::new(pos.x+dx, pos.z+dz)))
    }

    pub fn raycast(&self, origin: [f32;3], direction: [f32;3], max_dist: f32) -> Option<RayHit> {
        let mut x = origin[0].floor() as i32;
        let mut y = origin[1].floor() as i32;
        let mut z = origin[2].floor() as i32;
        let step_x = if direction[0]>0.0 {1i32} else {-1};
        let step_y = if direction[1]>0.0 {1i32} else {-1};
        let step_z = if direction[2]>0.0 {1i32} else {-1};
        let inv_dx = if direction[0]!=0.0 {1.0/direction[0].abs()} else {f32::MAX};
        let inv_dy = if direction[1]!=0.0 {1.0/direction[1].abs()} else {f32::MAX};
        let inv_dz = if direction[2]!=0.0 {1.0/direction[2].abs()} else {f32::MAX};
        let mut tmx = if direction[0]>0.0 {((x as f32+1.0)-origin[0])*inv_dx} else {(origin[0]-x as f32)*inv_dx};
        let mut tmy = if direction[1]>0.0 {((y as f32+1.0)-origin[1])*inv_dy} else {(origin[1]-y as f32)*inv_dy};
        let mut tmz = if direction[2]>0.0 {((z as f32+1.0)-origin[2])*inv_dz} else {(origin[2]-z as f32)*inv_dz};
        let (mut px, mut py, mut pz) = (x, y, z);
        let mut dist = 0.0;
        while dist < max_dist {
            let block = self.get_block(x, y, z);
            if block.is_solid() || block == BlockType::Torch {
                return Some(RayHit { block_pos: [x,y,z], place_pos: [px,py,pz], block_type: block });
            }
            px=x; py=y; pz=z;
            if tmx < tmy {
                if tmx < tmz { dist=tmx; x+=step_x; tmx+=inv_dx; }
                else { dist=tmz; z+=step_z; tmz+=inv_dz; }
            } else {
                if tmy < tmz { dist=tmy; y+=step_y; tmy+=inv_dy; }
                else { dist=tmz; z+=step_z; tmz+=inv_dz; }
            }
        }
        None
    }
}

pub struct RayHit {
    pub block_pos: [i32; 3],
    pub place_pos: [i32; 3],
    pub block_type: BlockType,
}