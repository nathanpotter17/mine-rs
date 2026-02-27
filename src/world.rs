use std::collections::{HashMap, VecDeque};

// ===== Block Types =====

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BlockType {
    Air = 0,
    Grass = 1,
    Dirt = 2,
    Stone = 3,
    Sand = 4,
    Water = 5,
    Wood = 6,
    Leaves = 7,
    Snow = 8,
    Bedrock = 9,
    Gravel = 10,
    Coal = 11,
    Iron = 12,
    Gold = 13,
    Diamond = 14,
    Torch = 15,
}

impl BlockType {
    pub fn is_solid(self) -> bool {
        !matches!(self, BlockType::Air | BlockType::Water | BlockType::Torch)
    }

    pub fn is_transparent(self) -> bool {
        matches!(self, BlockType::Air | BlockType::Water | BlockType::Leaves | BlockType::Torch)
    }

    pub fn is_light_source(self) -> bool {
        matches!(self, BlockType::Torch)
    }

    pub fn light_level(self) -> u8 {
        match self {
            BlockType::Torch => 14,
            _ => 0,
        }
    }

    pub fn colors(self) -> ([f32; 3], [f32; 3], [f32; 3]) {
        match self {
            BlockType::Grass => ([0.30, 0.65, 0.20], [0.45, 0.35, 0.20], [0.45, 0.32, 0.18]),
            BlockType::Dirt => ([0.45, 0.32, 0.18], [0.45, 0.32, 0.18], [0.45, 0.32, 0.18]),
            BlockType::Stone => ([0.50, 0.50, 0.52], [0.48, 0.48, 0.50], [0.46, 0.46, 0.48]),
            BlockType::Sand => ([0.85, 0.80, 0.55], [0.82, 0.77, 0.52], [0.80, 0.75, 0.50]),
            BlockType::Water => ([0.15, 0.35, 0.75], [0.12, 0.30, 0.70], [0.10, 0.25, 0.65]),
            BlockType::Wood => ([0.55, 0.40, 0.22], [0.50, 0.35, 0.18], [0.55, 0.40, 0.22]),
            BlockType::Leaves => ([0.20, 0.55, 0.15], [0.18, 0.50, 0.12], [0.20, 0.55, 0.15]),
            BlockType::Snow => ([0.95, 0.97, 1.00], [0.90, 0.92, 0.95], [0.88, 0.90, 0.93]),
            BlockType::Bedrock => ([0.20, 0.20, 0.22], [0.18, 0.18, 0.20], [0.15, 0.15, 0.17]),
            BlockType::Gravel => ([0.55, 0.52, 0.48], [0.52, 0.49, 0.45], [0.50, 0.47, 0.43]),
            BlockType::Coal => ([0.25, 0.25, 0.28], [0.22, 0.22, 0.25], [0.20, 0.20, 0.23]),
            BlockType::Iron => ([0.60, 0.55, 0.50], [0.58, 0.52, 0.47], [0.55, 0.50, 0.45]),
            BlockType::Gold => ([0.90, 0.78, 0.20], [0.85, 0.73, 0.18], [0.80, 0.68, 0.15]),
            BlockType::Diamond => ([0.45, 0.85, 0.90], [0.40, 0.80, 0.85], [0.35, 0.75, 0.80]),
            BlockType::Torch => ([0.95, 0.85, 0.45], [0.90, 0.75, 0.30], [0.90, 0.75, 0.30]),
            BlockType::Air => ([0.0; 3], [0.0; 3], [0.0; 3]),
        }
    }

    pub fn color_for_face(self, face: Face) -> [f32; 3] {
        let (top, side, bottom) = self.colors();
        match face {
            Face::Top => top,
            Face::Bottom => bottom,
            _ => side,
        }
    }
}

// ===== Faces =====

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Face {
    Top, Bottom, North, South, East, West,
}

impl Face {
    pub fn normal(self) -> [f32; 3] {
        match self {
            Face::Top => [0.0, 1.0, 0.0],
            Face::Bottom => [0.0, -1.0, 0.0],
            Face::North => [0.0, 0.0, 1.0],
            Face::South => [0.0, 0.0, -1.0],
            Face::East => [1.0, 0.0, 0.0],
            Face::West => [-1.0, 0.0, 0.0],
        }
    }

    pub fn offset(self) -> [i32; 3] {
        match self {
            Face::Top => [0, 1, 0],
            Face::Bottom => [0, -1, 0],
            Face::North => [0, 0, 1],
            Face::South => [0, 0, -1],
            Face::East => [1, 0, 0],
            Face::West => [-1, 0, 0],
        }
    }

    pub const ALL: [Face; 6] = [
        Face::Top, Face::Bottom, Face::North, Face::South, Face::East, Face::West,
    ];
}

// ===== Chunk =====

pub const CHUNK_X: usize = 16;
pub const CHUNK_Y: usize = 128;
pub const CHUNK_Z: usize = 16;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ChunkPos {
    pub x: i32,
    pub z: i32,
}

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
}

impl Chunk {
    pub fn new(pos: ChunkPos) -> Self {
        let size = CHUNK_X * CHUNK_Y * CHUNK_Z;
        Self {
            blocks: vec![BlockType::Air; size],
            light_map: vec![0u8; size],
            pos,
            dirty: true,
            light_dirty: true,
        }
    }

    #[inline]
    fn index(x: usize, y: usize, z: usize) -> usize {
        y * CHUNK_X * CHUNK_Z + z * CHUNK_X + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> BlockType {
        if x < CHUNK_X && y < CHUNK_Y && z < CHUNK_Z {
            self.blocks[Self::index(x, y, z)]
        } else {
            BlockType::Air
        }
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, block: BlockType) {
        if x < CHUNK_X && y < CHUNK_Y && z < CHUNK_Z {
            self.blocks[Self::index(x, y, z)] = block;
            self.dirty = true;
            self.light_dirty = true;
        }
    }

    #[inline]
    pub fn get_light(&self, x: usize, y: usize, z: usize) -> u8 {
        if x < CHUNK_X && y < CHUNK_Y && z < CHUNK_Z {
            self.light_map[Self::index(x, y, z)]
        } else {
            0
        }
    }

    #[inline]
    pub fn set_light(&mut self, x: usize, y: usize, z: usize, level: u8) {
        if x < CHUNK_X && y < CHUNK_Y && z < CHUNK_Z {
            self.light_map[Self::index(x, y, z)] = level;
        }
    }

    pub fn propagate_light(&mut self) {
        self.light_map.fill(0);
        let mut queue: VecDeque<(usize, usize, usize, u8)> = VecDeque::new();

        for y in 0..CHUNK_Y {
            for z in 0..CHUNK_Z {
                for x in 0..CHUNK_X {
                    let block = self.get(x, y, z);
                    if block.is_light_source() {
                        let level = block.light_level();
                        self.set_light(x, y, z, level);
                        queue.push_back((x, y, z, level));
                    }
                }
            }
        }

        // Sky light downward
        for z in 0..CHUNK_Z {
            for x in 0..CHUNK_X {
                let mut in_sky = true;
                for y in (0..CHUNK_Y).rev() {
                    if in_sky {
                        if self.get(x, y, z).is_solid() {
                            in_sky = false;
                        } else {
                            let sky_level = 12u8;
                            if sky_level > self.get_light(x, y, z) {
                                self.set_light(x, y, z, sky_level);
                                queue.push_back((x, y, z, sky_level));
                            }
                        }
                    }
                }
            }
        }

        let offsets: [(i32, i32, i32); 6] = [
            (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1),
        ];

        while let Some((x, y, z, level)) = queue.pop_front() {
            if level <= 1 { continue; }
            let new_level = level - 1;
            for (dx, dy, dz) in &offsets {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                let nz = z as i32 + dz;
                if nx < 0 || nx >= CHUNK_X as i32
                    || ny < 0 || ny >= CHUNK_Y as i32
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

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockVertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub normal: [f32; 3],
    pub ao: f32,
    pub light: f32,
}

pub struct ChunkMesh {
    pub vertices: Vec<BlockVertex>,
    pub indices: Vec<u32>,
}

// ===== Neighbor lookup with all 8 directions =====

pub struct ChunkNeighbors<'a> {
    pub north: Option<&'a Chunk>,      // +Z
    pub south: Option<&'a Chunk>,      // -Z
    pub east: Option<&'a Chunk>,       // +X
    pub west: Option<&'a Chunk>,       // -X
    pub northeast: Option<&'a Chunk>,  // +X +Z
    pub northwest: Option<&'a Chunk>,  // -X +Z
    pub southeast: Option<&'a Chunk>,  // +X -Z
    pub southwest: Option<&'a Chunk>,  // -X -Z
}

/// Resolve which chunk a local coordinate falls in, remapping to that chunk's local coords.
#[inline]
fn resolve_chunk<'a>(
    chunk: &'a Chunk,
    neighbors: &'a ChunkNeighbors<'a>,
    x: i32,
    z: i32,
) -> (Option<&'a Chunk>, usize, usize) {
    let x_neg = x < 0;
    let x_pos = x >= CHUNK_X as i32;
    let z_neg = z < 0;
    let z_pos = z >= CHUNK_Z as i32;

    let lx = if x_neg { (CHUNK_X as i32 + x) as usize }
             else if x_pos { (x - CHUNK_X as i32) as usize }
             else { x as usize };
    let lz = if z_neg { (CHUNK_Z as i32 + z) as usize }
             else if z_pos { (z - CHUNK_Z as i32) as usize }
             else { z as usize };

    let target = match (x_neg, x_pos, z_neg, z_pos) {
        (false, false, false, false) => Some(chunk),
        (true,  false, false, false) => neighbors.west,
        (false, true,  false, false) => neighbors.east,
        (false, false, true,  false) => neighbors.south,
        (false, false, false, true)  => neighbors.north,
        (true,  false, true,  false) => neighbors.southwest,
        (true,  false, false, true)  => neighbors.northwest,
        (false, true,  true,  false) => neighbors.southeast,
        (false, true,  false, true)  => neighbors.northeast,
        _ => None,
    };

    (target, lx, lz)
}

fn get_block_with_neighbors(
    chunk: &Chunk, neighbors: &ChunkNeighbors,
    x: i32, y: i32, z: i32,
) -> BlockType {
    if y < 0 || y >= CHUNK_Y as i32 { return BlockType::Air; }
    let (target, lx, lz) = resolve_chunk(chunk, neighbors, x, z);
    match target {
        Some(c) => c.get(lx, y as usize, lz),
        None => BlockType::Air,
    }
}

fn get_light_with_neighbors(
    chunk: &Chunk, neighbors: &ChunkNeighbors,
    x: i32, y: i32, z: i32,
) -> u8 {
    if y < 0 { return 0; }
    if y >= CHUNK_Y as i32 { return 12; }
    let (target, lx, lz) = resolve_chunk(chunk, neighbors, x, z);
    match target {
        Some(c) => c.get_light(lx, y as usize, lz),
        None => 0,
    }
}

// ===== Ambient Occlusion =====

fn compute_face_ao(
    chunk: &Chunk, neighbors: &ChunkNeighbors,
    x: i32, y: i32, z: i32, face: Face,
) -> [u8; 4] {
    let samples = ao_neighbor_offsets(face);
    let mut ao = [0u8; 4];
    for (i, (e1, e2, corner)) in samples.iter().enumerate() {
        let s1 = get_block_with_neighbors(chunk, neighbors, x+e1[0], y+e1[1], z+e1[2]).is_solid();
        let s2 = get_block_with_neighbors(chunk, neighbors, x+e2[0], y+e2[1], z+e2[2]).is_solid();
        let c  = get_block_with_neighbors(chunk, neighbors, x+corner[0], y+corner[1], z+corner[2]).is_solid();
        ao[i] = if s1 && s2 { 0 } else { 3 - (s1 as u8 + s2 as u8 + c as u8) };
    }
    ao
}

fn ao_neighbor_offsets(face: Face) -> [([i32;3],[i32;3],[i32;3]); 4] {
    match face {
        Face::Top => [
            ([0,1,-1], [-1,1,0], [-1,1,-1]),
            ([0,1,-1], [1,1,0],  [1,1,-1]),
            ([0,1,1],  [1,1,0],  [1,1,1]),
            ([0,1,1],  [-1,1,0], [-1,1,1]),
        ],
        Face::Bottom => [
            ([0,-1,1],  [-1,-1,0], [-1,-1,1]),
            ([0,-1,1],  [1,-1,0],  [1,-1,1]),
            ([0,-1,-1], [1,-1,0],  [1,-1,-1]),
            ([0,-1,-1], [-1,-1,0], [-1,-1,-1]),
        ],
        Face::North => [
            ([1,0,1],  [0,-1,1], [1,-1,1]),
            ([-1,0,1], [0,-1,1], [-1,-1,1]),
            ([-1,0,1], [0,1,1],  [-1,1,1]),
            ([1,0,1],  [0,1,1],  [1,1,1]),
        ],
        Face::South => [
            ([-1,0,-1], [0,-1,-1], [-1,-1,-1]),
            ([1,0,-1],  [0,-1,-1], [1,-1,-1]),
            ([1,0,-1],  [0,1,-1],  [1,1,-1]),
            ([-1,0,-1], [0,1,-1],  [-1,1,-1]),
        ],
        Face::East => [
            ([1,0,-1], [1,-1,0], [1,-1,-1]),
            ([1,0,1],  [1,-1,0], [1,-1,1]),
            ([1,0,1],  [1,1,0],  [1,1,1]),
            ([1,0,-1], [1,1,0],  [1,1,-1]),
        ],
        Face::West => [
            ([-1,0,1],  [-1,-1,0], [-1,-1,1]),
            ([-1,0,-1], [-1,-1,0], [-1,-1,-1]),
            ([-1,0,-1], [-1,1,0],  [-1,1,-1]),
            ([-1,0,1],  [-1,1,0],  [-1,1,1]),
        ],
    }
}

// ===== Per-vertex light sampling =====

fn compute_face_light(
    chunk: &Chunk, neighbors: &ChunkNeighbors,
    x: i32, y: i32, z: i32, face: Face,
) -> [f32; 4] {
    let off = face.offset();
    let fx = x + off[0];
    let fy = y + off[1];
    let fz = z + off[2];

    let sample_offsets = light_vertex_offsets(face);
    let mut lights = [0.0f32; 4];
    for (i, samples) in sample_offsets.iter().enumerate() {
        let mut total = 0u32;
        let mut count = 0u32;
        for &(sx, sy, sz) in samples {
            total += get_light_with_neighbors(chunk, neighbors, fx+sx, fy+sy, fz+sz) as u32;
            count += 1;
        }
        total += get_light_with_neighbors(chunk, neighbors, fx, fy, fz) as u32;
        count += 1;
        lights[i] = (total as f32 / count as f32) / 15.0;
    }
    lights
}

fn light_vertex_offsets(face: Face) -> [[(i32,i32,i32); 3]; 4] {
    match face {
        Face::Top => [
            [(-1,0,0), (0,0,-1), (-1,0,-1)],
            [(1,0,0),  (0,0,-1), (1,0,-1)],
            [(1,0,0),  (0,0,1),  (1,0,1)],
            [(-1,0,0), (0,0,1),  (-1,0,1)],
        ],
        Face::Bottom => [
            [(-1,0,0), (0,0,1),  (-1,0,1)],
            [(1,0,0),  (0,0,1),  (1,0,1)],
            [(1,0,0),  (0,0,-1), (1,0,-1)],
            [(-1,0,0), (0,0,-1), (-1,0,-1)],
        ],
        Face::North => [
            [(1,0,0),  (0,-1,0), (1,-1,0)],
            [(-1,0,0), (0,-1,0), (-1,-1,0)],
            [(-1,0,0), (0,1,0),  (-1,1,0)],
            [(1,0,0),  (0,1,0),  (1,1,0)],
        ],
        Face::South => [
            [(-1,0,0), (0,-1,0), (-1,-1,0)],
            [(1,0,0),  (0,-1,0), (1,-1,0)],
            [(1,0,0),  (0,1,0),  (1,1,0)],
            [(-1,0,0), (0,1,0),  (-1,1,0)],
        ],
        Face::East => [
            [(0,0,-1), (0,-1,0), (0,-1,-1)],
            [(0,0,1),  (0,-1,0), (0,-1,1)],
            [(0,0,1),  (0,1,0),  (0,1,1)],
            [(0,0,-1), (0,1,0),  (0,1,-1)],
        ],
        Face::West => [
            [(0,0,1),  (0,-1,0), (0,-1,1)],
            [(0,0,-1), (0,-1,0), (0,-1,-1)],
            [(0,0,-1), (0,1,0),  (0,1,-1)],
            [(0,0,1),  (0,1,0),  (0,1,1)],
        ],
    }
}

// ===== Meshing =====

pub fn mesh_chunk(chunk: &Chunk, neighbors: &ChunkNeighbors) -> ChunkMesh {
    let mut vertices = Vec::with_capacity(1024);
    let mut indices = Vec::with_capacity(2048);
    let (ox, oz) = chunk.pos.world_offset();

    for y in 0..CHUNK_Y {
        for z in 0..CHUNK_Z {
            for x in 0..CHUNK_X {
                let block = chunk.get(x, y, z);
                if block == BlockType::Air { continue; }

                let wx = ox + x as f32;
                let wy = y as f32;
                let wz = oz + z as f32;
                let ix = x as i32;
                let iy = y as i32;
                let iz = z as i32;

                if block == BlockType::Torch {
                    add_torch_geometry(&mut vertices, &mut indices, wx, wy, wz, chunk, neighbors, ix, iy, iz);
                    continue;
                }

                for face in Face::ALL {
                    let off = face.offset();
                    let neighbor = get_block_with_neighbors(chunk, neighbors, ix+off[0], iy+off[1], iz+off[2]);
                    if neighbor.is_transparent() && neighbor != block {
                        let ao_raw = compute_face_ao(chunk, neighbors, ix, iy, iz, face);
                        let ao = [
                            ao_raw[0] as f32 / 3.0,
                            ao_raw[1] as f32 / 3.0,
                            ao_raw[2] as f32 / 3.0,
                            ao_raw[3] as f32 / 3.0,
                        ];
                        let light = compute_face_light(chunk, neighbors, ix, iy, iz, face);
                        add_face_ao(&mut vertices, &mut indices,
                            wx, wy, wz, face,
                            block.color_for_face(face), ao, light, ao_raw);
                    }
                }
            }
        }
    }

    ChunkMesh { vertices, indices }
}

fn add_torch_geometry(
    vertices: &mut Vec<BlockVertex>, indices: &mut Vec<u32>,
    x: f32, y: f32, z: f32,
    chunk: &Chunk, neighbors: &ChunkNeighbors,
    ix: i32, iy: i32, iz: i32,
) {
    let color = [0.95, 0.82, 0.35];
    let light_val = (get_light_with_neighbors(chunk, neighbors, ix, iy, iz) as f32 / 15.0).max(0.9);
    let cx = x + 0.5;
    let cz = z + 0.5;
    let r = 0.1;
    let h = 0.65;

    // Quad 1: X-aligned (both sides)
    let base = vertices.len() as u32;
    vertices.push(BlockVertex { position: [cx-r,y,cz],   color, normal: [0.0,0.0,1.0], ao: 1.0, light: light_val });
    vertices.push(BlockVertex { position: [cx+r,y,cz],   color, normal: [0.0,0.0,1.0], ao: 1.0, light: light_val });
    vertices.push(BlockVertex { position: [cx+r,y+h,cz], color, normal: [0.0,0.0,1.0], ao: 1.0, light: light_val });
    vertices.push(BlockVertex { position: [cx-r,y+h,cz], color, normal: [0.0,0.0,1.0], ao: 1.0, light: light_val });
    indices.extend_from_slice(&[base,base+1,base+2, base+2,base+3,base]);
    indices.extend_from_slice(&[base+2,base+1,base, base,base+3,base+2]);

    // Quad 2: Z-aligned (both sides)
    let base = vertices.len() as u32;
    vertices.push(BlockVertex { position: [cx,y,cz-r],   color, normal: [1.0,0.0,0.0], ao: 1.0, light: light_val });
    vertices.push(BlockVertex { position: [cx,y,cz+r],   color, normal: [1.0,0.0,0.0], ao: 1.0, light: light_val });
    vertices.push(BlockVertex { position: [cx,y+h,cz+r], color, normal: [1.0,0.0,0.0], ao: 1.0, light: light_val });
    vertices.push(BlockVertex { position: [cx,y+h,cz-r], color, normal: [1.0,0.0,0.0], ao: 1.0, light: light_val });
    indices.extend_from_slice(&[base,base+1,base+2, base+2,base+3,base]);
    indices.extend_from_slice(&[base+2,base+1,base, base,base+3,base+2]);

    // Flame cap
    let flame = [1.0, 0.65, 0.15];
    let base = vertices.len() as u32;
    vertices.push(BlockVertex { position: [cx-r,y+h,cz-r], color: flame, normal: [0.0,1.0,0.0], ao: 1.0, light: 1.0 });
    vertices.push(BlockVertex { position: [cx+r,y+h,cz-r], color: flame, normal: [0.0,1.0,0.0], ao: 1.0, light: 1.0 });
    vertices.push(BlockVertex { position: [cx+r,y+h,cz+r], color: flame, normal: [0.0,1.0,0.0], ao: 1.0, light: 1.0 });
    vertices.push(BlockVertex { position: [cx-r,y+h,cz+r], color: flame, normal: [0.0,1.0,0.0], ao: 1.0, light: 1.0 });
    indices.extend_from_slice(&[base,base+1,base+2, base+2,base+3,base]);
}

fn add_face_ao(
    vertices: &mut Vec<BlockVertex>, indices: &mut Vec<u32>,
    x: f32, y: f32, z: f32,
    face: Face, color: [f32; 3],
    ao: [f32; 4], light: [f32; 4], ao_raw: [u8; 4],
) {
    let base = vertices.len() as u32;
    let n = face.normal();
    let (v0,v1,v2,v3) = match face {
        Face::Top    => ([x,y+1.0,z], [x+1.0,y+1.0,z], [x+1.0,y+1.0,z+1.0], [x,y+1.0,z+1.0]),
        Face::Bottom => ([x,y,z+1.0], [x+1.0,y,z+1.0], [x+1.0,y,z],         [x,y,z]),
        Face::North  => ([x+1.0,y,z+1.0], [x,y,z+1.0], [x,y+1.0,z+1.0], [x+1.0,y+1.0,z+1.0]),
        Face::South  => ([x,y,z], [x+1.0,y,z], [x+1.0,y+1.0,z], [x,y+1.0,z]),
        Face::East   => ([x+1.0,y,z], [x+1.0,y,z+1.0], [x+1.0,y+1.0,z+1.0], [x+1.0,y+1.0,z]),
        Face::West   => ([x,y,z+1.0], [x,y,z], [x,y+1.0,z], [x,y+1.0,z+1.0]),
    };

    vertices.push(BlockVertex { position: v0, color, normal: n, ao: ao[0], light: light[0] });
    vertices.push(BlockVertex { position: v1, color, normal: n, ao: ao[1], light: light[1] });
    vertices.push(BlockVertex { position: v2, color, normal: n, ao: ao[2], light: light[2] });
    vertices.push(BlockVertex { position: v3, color, normal: n, ao: ao[3], light: light[3] });

    // Flip quad diagonal when AO is anisotropic to prevent interpolation artifacts
    if ao_raw[0] + ao_raw[2] > ao_raw[1] + ao_raw[3] {
        indices.extend_from_slice(&[base, base+1, base+2, base+2, base+3, base]);
    } else {
        indices.extend_from_slice(&[base+1, base+2, base+3, base+3, base, base+1]);
    }
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
        let ix = x.floor() as i32;
        let iz = z.floor() as i32;
        let fx = x - x.floor();
        let fz = z - z.floor();
        let sx = fx * fx * (3.0 - 2.0 * fx);
        let sz = fz * fz * (3.0 - 2.0 * fz);
        let v00 = self.hash2d(ix, iz);
        let v10 = self.hash2d(ix+1, iz);
        let v01 = self.hash2d(ix, iz+1);
        let v11 = self.hash2d(ix+1, iz+1);
        let a = v00 + sx * (v10 - v00);
        let b = v01 + sx * (v11 - v01);
        a + sz * (b - a)
    }

    pub fn fbm(&self, x: f32, z: f32, octaves: u32) -> f32 {
        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_val = 0.0;
        for _ in 0..octaves {
            value += amplitude * self.smoothed(x * frequency, z * frequency);
            max_val += amplitude;
            amplitude *= 0.5;
            frequency *= 2.0;
        }
        value / max_val
    }
}

// ===== World =====

pub const RENDER_DISTANCE: i32 = 6;
pub const GENERATION_DISTANCE: i32 = RENDER_DISTANCE + 2; // Pre-generate buffer ring
pub const SEA_LEVEL: usize = 32;
const MAX_CHUNKS_PER_FRAME: usize = 4;

pub struct World {
    pub chunks: HashMap<ChunkPos, Chunk>,
    noise: Noise,
    tree_noise: Noise,
}

impl World {
    pub fn new(seed: u32) -> Self {
        Self {
            chunks: HashMap::new(),
            noise: Noise::new(seed),
            tree_noise: Noise::new(seed.wrapping_add(12345)),
        }
    }

    /// Blocking generation of all chunks (used at startup only).
    pub fn generate_around_immediate(&mut self, player_x: f32, player_z: f32) {
        let cx = (player_x / CHUNK_X as f32).floor() as i32;
        let cz = (player_z / CHUNK_Z as f32).floor() as i32;
        for dz in -GENERATION_DISTANCE..=GENERATION_DISTANCE {
            for dx in -GENERATION_DISTANCE..=GENERATION_DISTANCE {
                let pos = ChunkPos::new(cx + dx, cz + dz);
                if !self.chunks.contains_key(&pos) {
                    let mut chunk = self.generate_chunk(pos);
                    chunk.propagate_light();
                    self.chunks.insert(pos, chunk);
                }
            }
        }
        self.remove_far_chunks(cx, cz);
    }

    /// Rate-limited generation (max MAX_CHUNKS_PER_FRAME per call), spiral from center.
    pub fn generate_around(&mut self, player_x: f32, player_z: f32) {
        let cx = (player_x / CHUNK_X as f32).floor() as i32;
        let cz = (player_z / CHUNK_Z as f32).floor() as i32;
        let mut generated = 0;

        'outer: for dist in 0..=GENERATION_DISTANCE {
            for dz in -dist..=dist {
                for dx in -dist..=dist {
                    if dx.abs() != dist && dz.abs() != dist { continue; }
                    let pos = ChunkPos::new(cx + dx, cz + dz);
                    if !self.chunks.contains_key(&pos) {
                        let mut chunk = self.generate_chunk(pos);
                        chunk.propagate_light();
                        self.chunks.insert(pos, chunk);
                        generated += 1;

                        // Mark all existing neighbors dirty so they re-mesh
                        // with correct cross-boundary AO and lighting
                        for &(ndx, ndz) in &[
                            (-1, 0), (1, 0), (0, -1), (0, 1),
                            (-1, -1), (-1, 1), (1, -1), (1, 1),
                        ] {
                            let npos = ChunkPos::new(pos.x + ndx, pos.z + ndz);
                            if let Some(neighbor) = self.chunks.get_mut(&npos) {
                                neighbor.dirty = true;
                            }
                        }

                        if generated >= MAX_CHUNKS_PER_FRAME { break 'outer; }
                    }
                }
            }
        }

        self.remove_far_chunks(cx, cz);
    }

    fn remove_far_chunks(&mut self, cx: i32, cz: i32) {
        let keys: Vec<ChunkPos> = self.chunks.keys().copied().collect();
        for pos in keys {
            if (pos.x - cx).abs() > GENERATION_DISTANCE + 2
            || (pos.z - cz).abs() > GENERATION_DISTANCE + 2 {
                self.chunks.remove(&pos);
            }
        }
    }

    pub fn repropagate_light_around(&mut self, cx: i32, cz: i32) {
        for pos in [
            ChunkPos::new(cx, cz),
            ChunkPos::new(cx-1, cz), ChunkPos::new(cx+1, cz),
            ChunkPos::new(cx, cz-1), ChunkPos::new(cx, cz+1),
        ] {
            if let Some(chunk) = self.chunks.get_mut(&pos) {
                chunk.propagate_light();
            }
        }
    }

    fn generate_chunk(&self, pos: ChunkPos) -> Chunk {
        let mut chunk = Chunk::new(pos);
        for z in 0..CHUNK_Z {
            for x in 0..CHUNK_X {
                let wx = pos.x as f32 * CHUNK_X as f32 + x as f32;
                let wz = pos.z as f32 * CHUNK_Z as f32 + z as f32;
                let base_height = self.noise.fbm(wx * 0.008, wz * 0.008, 4);
                let detail = self.noise.fbm(wx * 0.03, wz * 0.03, 3) * 0.3;
                let mountains = self.noise.fbm(wx * 0.004, wz * 0.004, 3);
                let mountain_factor = (mountains - 0.5).max(0.0) * 2.0;
                let height = ((base_height * 30.0 + detail * 8.0 + mountain_factor * 40.0 + 20.0) as usize).min(CHUNK_Y - 1);

                chunk.set(x, 0, z, BlockType::Bedrock);
                for y in 1..=height {
                    let block = if y == height {
                        if height < SEA_LEVEL - 1 { BlockType::Sand }
                        else if height > 70 { BlockType::Snow }
                        else { BlockType::Grass }
                    } else if y > height.saturating_sub(4) {
                        if height < SEA_LEVEL - 1 { BlockType::Sand } else { BlockType::Dirt }
                    } else if y < 5 {
                        BlockType::Bedrock
                    } else {
                        let ore_noise = self.noise.hash2d(wx as i32 + y as i32 * 137, wz as i32 + y as i32 * 251);
                        if y < 12 && ore_noise < 0.02 { BlockType::Diamond }
                        else if y < 32 && ore_noise < 0.04 { BlockType::Gold }
                        else if y < 48 && ore_noise < 0.08 { BlockType::Iron }
                        else if ore_noise < 0.10 { BlockType::Coal }
                        else if ore_noise < 0.14 { BlockType::Gravel }
                        else { BlockType::Stone }
                    };
                    chunk.set(x, y, z, block);
                }
                for y in (height + 1)..SEA_LEVEL {
                    chunk.set(x, y, z, BlockType::Water);
                }
                if height >= SEA_LEVEL && height < 65 {
                    let tree_val = self.tree_noise.hash2d(wx as i32, wz as i32);
                    if tree_val < 0.012 && x >= 2 && x < CHUNK_X-2 && z >= 2 && z < CHUNK_Z-2 {
                        self.place_tree(&mut chunk, x, height + 1, z);
                    }
                }
            }
        }
        chunk
    }

    fn place_tree(&self, chunk: &mut Chunk, x: usize, y: usize, z: usize) {
        let trunk_height = 5;
        for dy in 0..trunk_height {
            if y + dy < CHUNK_Y { chunk.set(x, y + dy, z, BlockType::Wood); }
        }
        let leaf_start = trunk_height - 2;
        for dy in leaf_start..(trunk_height + 2) {
            let radius = if dy >= trunk_height { 1 } else { 2 };
            for dz in -(radius as i32)..=(radius as i32) {
                for dx in -(radius as i32)..=(radius as i32) {
                    if dx*dx + dz*dz <= radius*radius + 1 {
                        let lx = (x as i32 + dx) as usize;
                        let ly = y + dy;
                        let lz = (z as i32 + dz) as usize;
                        if lx < CHUNK_X && ly < CHUNK_Y && lz < CHUNK_Z && chunk.get(lx, ly, lz) == BlockType::Air {
                            chunk.set(lx, ly, lz, BlockType::Leaves);
                        }
                    }
                }
            }
        }
    }

    pub fn get_block(&self, wx: i32, wy: i32, wz: i32) -> BlockType {
        if wy < 0 || wy >= CHUNK_Y as i32 { return BlockType::Air; }
        let cx = if wx < 0 { (wx+1) / CHUNK_X as i32 - 1 } else { wx / CHUNK_X as i32 };
        let cz = if wz < 0 { (wz+1) / CHUNK_Z as i32 - 1 } else { wz / CHUNK_Z as i32 };
        let lx = ((wx % CHUNK_X as i32) + CHUNK_X as i32) as usize % CHUNK_X;
        let lz = ((wz % CHUNK_Z as i32) + CHUNK_Z as i32) as usize % CHUNK_Z;
        self.chunks.get(&ChunkPos::new(cx, cz)).map(|c| c.get(lx, wy as usize, lz)).unwrap_or(BlockType::Air)
    }

    pub fn set_block(&mut self, wx: i32, wy: i32, wz: i32, block: BlockType) {
        if wy < 0 || wy >= CHUNK_Y as i32 { return; }
        let cx = if wx < 0 { (wx+1) / CHUNK_X as i32 - 1 } else { wx / CHUNK_X as i32 };
        let cz = if wz < 0 { (wz+1) / CHUNK_Z as i32 - 1 } else { wz / CHUNK_Z as i32 };
        let lx = ((wx % CHUNK_X as i32) + CHUNK_X as i32) as usize % CHUNK_X;
        let lz = ((wz % CHUNK_Z as i32) + CHUNK_Z as i32) as usize % CHUNK_Z;

        if let Some(chunk) = self.chunks.get_mut(&ChunkPos::new(cx, cz)) {
            chunk.set(lx, wy as usize, lz, block);
        }
        self.repropagate_light_around(cx, cz);

        if lx == 0          { self.mark_dirty(ChunkPos::new(cx-1, cz)); }
        if lx == CHUNK_X - 1 { self.mark_dirty(ChunkPos::new(cx+1, cz)); }
        if lz == 0          { self.mark_dirty(ChunkPos::new(cx, cz-1)); }
        if lz == CHUNK_Z - 1 { self.mark_dirty(ChunkPos::new(cx, cz+1)); }
    }

    fn mark_dirty(&mut self, pos: ChunkPos) {
        if let Some(chunk) = self.chunks.get_mut(&pos) { chunk.dirty = true; }
    }

    pub fn get_neighbors(&self, pos: ChunkPos) -> ChunkNeighbors {
        ChunkNeighbors {
            north:     self.chunks.get(&ChunkPos::new(pos.x,   pos.z+1)),
            south:     self.chunks.get(&ChunkPos::new(pos.x,   pos.z-1)),
            east:      self.chunks.get(&ChunkPos::new(pos.x+1, pos.z)),
            west:      self.chunks.get(&ChunkPos::new(pos.x-1, pos.z)),
            northeast: self.chunks.get(&ChunkPos::new(pos.x+1, pos.z+1)),
            northwest: self.chunks.get(&ChunkPos::new(pos.x-1, pos.z+1)),
            southeast: self.chunks.get(&ChunkPos::new(pos.x+1, pos.z-1)),
            southwest: self.chunks.get(&ChunkPos::new(pos.x-1, pos.z-1)),
        }
    }

    /// Returns true if all 8 surrounding chunks exist (needed for correct AO meshing).
    pub fn has_all_neighbors(&self, pos: ChunkPos) -> bool {
        self.chunks.contains_key(&ChunkPos::new(pos.x-1, pos.z))
        && self.chunks.contains_key(&ChunkPos::new(pos.x+1, pos.z))
        && self.chunks.contains_key(&ChunkPos::new(pos.x, pos.z-1))
        && self.chunks.contains_key(&ChunkPos::new(pos.x, pos.z+1))
        && self.chunks.contains_key(&ChunkPos::new(pos.x-1, pos.z-1))
        && self.chunks.contains_key(&ChunkPos::new(pos.x-1, pos.z+1))
        && self.chunks.contains_key(&ChunkPos::new(pos.x+1, pos.z-1))
        && self.chunks.contains_key(&ChunkPos::new(pos.x+1, pos.z+1))
    }

    pub fn raycast(&self, origin: [f32; 3], direction: [f32; 3], max_dist: f32) -> Option<RayHit> {
        let mut x = origin[0].floor() as i32;
        let mut y = origin[1].floor() as i32;
        let mut z = origin[2].floor() as i32;
        let step_x = if direction[0] > 0.0 { 1i32 } else { -1 };
        let step_y = if direction[1] > 0.0 { 1i32 } else { -1 };
        let step_z = if direction[2] > 0.0 { 1i32 } else { -1 };
        let inv_dx = if direction[0] != 0.0 { 1.0 / direction[0].abs() } else { f32::MAX };
        let inv_dy = if direction[1] != 0.0 { 1.0 / direction[1].abs() } else { f32::MAX };
        let inv_dz = if direction[2] != 0.0 { 1.0 / direction[2].abs() } else { f32::MAX };
        let mut t_max_x = if direction[0] > 0.0 { ((x as f32+1.0)-origin[0])*inv_dx } else { (origin[0]-x as f32)*inv_dx };
        let mut t_max_y = if direction[1] > 0.0 { ((y as f32+1.0)-origin[1])*inv_dy } else { (origin[1]-y as f32)*inv_dy };
        let mut t_max_z = if direction[2] > 0.0 { ((z as f32+1.0)-origin[2])*inv_dz } else { (origin[2]-z as f32)*inv_dz };
        let (mut prev_x, mut prev_y, mut prev_z) = (x, y, z);
        let mut dist = 0.0;

        while dist < max_dist {
            let block = self.get_block(x, y, z);
            if block.is_solid() || block == BlockType::Torch {
                return Some(RayHit { block_pos: [x,y,z], place_pos: [prev_x,prev_y,prev_z], block_type: block });
            }
            prev_x = x; prev_y = y; prev_z = z;
            if t_max_x < t_max_y {
                if t_max_x < t_max_z { dist = t_max_x; x += step_x; t_max_x += inv_dx; }
                else { dist = t_max_z; z += step_z; t_max_z += inv_dz; }
            } else {
                if t_max_y < t_max_z { dist = t_max_y; y += step_y; t_max_y += inv_dy; }
                else { dist = t_max_z; z += step_z; t_max_z += inv_dz; }
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