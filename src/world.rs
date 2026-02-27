use std::collections::HashMap;

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
}

impl BlockType {
    pub fn is_solid(self) -> bool {
        !matches!(self, BlockType::Air | BlockType::Water)
    }

    pub fn is_transparent(self) -> bool {
        matches!(self, BlockType::Air | BlockType::Water | BlockType::Leaves)
    }

    /// Returns (top_color, side_color, bottom_color)
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
    Top,    // +Y
    Bottom, // -Y
    North,  // +Z
    South,  // -Z
    East,   // +X
    West,   // -X
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
    pub fn new(x: i32, z: i32) -> Self {
        Self { x, z }
    }

    pub fn world_offset(&self) -> (f32, f32) {
        (self.x as f32 * CHUNK_X as f32, self.z as f32 * CHUNK_Z as f32)
    }
}

pub struct Chunk {
    blocks: Vec<BlockType>, // CHUNK_X * CHUNK_Y * CHUNK_Z
    pub pos: ChunkPos,
    pub dirty: bool,
}

impl Chunk {
    pub fn new(pos: ChunkPos) -> Self {
        Self {
            blocks: vec![BlockType::Air; CHUNK_X * CHUNK_Y * CHUNK_Z],
            pos,
            dirty: true,
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
        }
    }
}

// ===== Mesh Vertex =====

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockVertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub normal: [f32; 3],
}

pub struct ChunkMesh {
    pub vertices: Vec<BlockVertex>,
    pub indices: Vec<u32>,
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
                if block == BlockType::Air {
                    continue;
                }

                let wx = ox + x as f32;
                let wy = y as f32;
                let wz = oz + z as f32;

                for face in Face::ALL {
                    let off = face.offset();
                    let nx = x as i32 + off[0];
                    let ny = y as i32 + off[1];
                    let nz = z as i32 + off[2];

                    let neighbor = get_block_with_neighbors(chunk, neighbors, nx, ny, nz);
                    if neighbor.is_transparent() && neighbor != block {
                        add_face(
                            &mut vertices,
                            &mut indices,
                            wx, wy, wz,
                            face,
                            block.color_for_face(face),
                        );
                    }
                }
            }
        }
    }

    ChunkMesh { vertices, indices }
}

pub struct ChunkNeighbors<'a> {
    pub north: Option<&'a Chunk>,  // +Z
    pub south: Option<&'a Chunk>,  // -Z
    pub east: Option<&'a Chunk>,   // +X
    pub west: Option<&'a Chunk>,   // -X
}

fn get_block_with_neighbors(
    chunk: &Chunk,
    neighbors: &ChunkNeighbors,
    x: i32, y: i32, z: i32,
) -> BlockType {
    if y < 0 || y >= CHUNK_Y as i32 {
        return BlockType::Air;
    }

    let yu = y as usize;

    if x >= 0 && x < CHUNK_X as i32 && z >= 0 && z < CHUNK_Z as i32 {
        return chunk.get(x as usize, yu, z as usize);
    }

    // Check neighbor chunks
    if x < 0 {
        if let Some(west) = neighbors.west {
            return west.get((CHUNK_X as i32 + x) as usize, yu, z as usize);
        }
    } else if x >= CHUNK_X as i32 {
        if let Some(east) = neighbors.east {
            return east.get((x - CHUNK_X as i32) as usize, yu, z as usize);
        }
    }

    if z < 0 {
        if let Some(south) = neighbors.south {
            return south.get(x as usize, yu, (CHUNK_Z as i32 + z) as usize);
        }
    } else if z >= CHUNK_Z as i32 {
        if let Some(north) = neighbors.north {
            return north.get(x as usize, yu, (z - CHUNK_Z as i32) as usize);
        }
    }

    BlockType::Air
}

fn add_face(
    vertices: &mut Vec<BlockVertex>,
    indices: &mut Vec<u32>,
    x: f32, y: f32, z: f32,
    face: Face,
    color: [f32; 3],
) {
    let base = vertices.len() as u32;
    let n = face.normal();

    let (v0, v1, v2, v3) = match face {
        Face::Top => (
            [x, y + 1.0, z],
            [x + 1.0, y + 1.0, z],
            [x + 1.0, y + 1.0, z + 1.0],
            [x, y + 1.0, z + 1.0],
        ),
        Face::Bottom => (
            [x, y, z + 1.0],
            [x + 1.0, y, z + 1.0],
            [x + 1.0, y, z],
            [x, y, z],
        ),
        Face::North => (
            [x + 1.0, y, z + 1.0],
            [x, y, z + 1.0],
            [x, y + 1.0, z + 1.0],
            [x + 1.0, y + 1.0, z + 1.0],
        ),
        Face::South => (
            [x, y, z],
            [x + 1.0, y, z],
            [x + 1.0, y + 1.0, z],
            [x, y + 1.0, z],
        ),
        Face::East => (
            [x + 1.0, y, z],
            [x + 1.0, y, z + 1.0],
            [x + 1.0, y + 1.0, z + 1.0],
            [x + 1.0, y + 1.0, z],
        ),
        Face::West => (
            [x, y, z + 1.0],
            [x, y, z],
            [x, y + 1.0, z],
            [x, y + 1.0, z + 1.0],
        ),
    };

    vertices.push(BlockVertex { position: v0, color, normal: n });
    vertices.push(BlockVertex { position: v1, color, normal: n });
    vertices.push(BlockVertex { position: v2, color, normal: n });
    vertices.push(BlockVertex { position: v3, color, normal: n });

    indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
}

// ===== Noise / Terrain Generation =====

/// Simple hash-based noise for terrain generation
pub struct Noise {
    seed: u32,
}

impl Noise {
    pub fn new(seed: u32) -> Self {
        Self { seed }
    }

    fn hash2d(&self, x: i32, z: i32) -> f32 {
        let mut h = self.seed;
        h ^= x as u32;
        h = h.wrapping_mul(0x85ebca6b);
        h ^= z as u32;
        h = h.wrapping_mul(0xc2b2ae35);
        h ^= h >> 16;
        h = h.wrapping_mul(0x45d9f3b);
        h ^= h >> 16;
        (h & 0xFFFF) as f32 / 65535.0
    }

    fn smoothed(&self, x: f32, z: f32) -> f32 {
        let ix = x.floor() as i32;
        let iz = z.floor() as i32;
        let fx = x - x.floor();
        let fz = z - z.floor();

        // Smoothstep
        let sx = fx * fx * (3.0 - 2.0 * fx);
        let sz = fz * fz * (3.0 - 2.0 * fz);

        let v00 = self.hash2d(ix, iz);
        let v10 = self.hash2d(ix + 1, iz);
        let v01 = self.hash2d(ix, iz + 1);
        let v11 = self.hash2d(ix + 1, iz + 1);

        let a = v00 + sx * (v10 - v00);
        let b = v01 + sx * (v11 - v01);
        a + sz * (b - a)
    }

    /// Fractal Brownian Motion noise
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
pub const SEA_LEVEL: usize = 32;

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

    pub fn generate_around(&mut self, player_x: f32, player_z: f32) {
        let cx = (player_x / CHUNK_X as f32).floor() as i32;
        let cz = (player_z / CHUNK_Z as f32).floor() as i32;

        for dz in -RENDER_DISTANCE..=RENDER_DISTANCE {
            for dx in -RENDER_DISTANCE..=RENDER_DISTANCE {
                let pos = ChunkPos::new(cx + dx, cz + dz);
                if !self.chunks.contains_key(&pos) {
                    let chunk = self.generate_chunk(pos);
                    self.chunks.insert(pos, chunk);
                }
            }
        }

        // Remove far chunks
        let keys: Vec<ChunkPos> = self.chunks.keys().copied().collect();
        for pos in keys {
            if (pos.x - cx).abs() > RENDER_DISTANCE + 2 || (pos.z - cz).abs() > RENDER_DISTANCE + 2 {
                self.chunks.remove(&pos);
            }
        }
    }

    fn generate_chunk(&self, pos: ChunkPos) -> Chunk {
        let mut chunk = Chunk::new(pos);

        for z in 0..CHUNK_Z {
            for x in 0..CHUNK_X {
                let wx = pos.x as f32 * CHUNK_X as f32 + x as f32;
                let wz = pos.z as f32 * CHUNK_Z as f32 + z as f32;

                // Multi-octave terrain height
                let base_height = self.noise.fbm(wx * 0.008, wz * 0.008, 4);
                let detail = self.noise.fbm(wx * 0.03, wz * 0.03, 3) * 0.3;
                let mountains = self.noise.fbm(wx * 0.004, wz * 0.004, 3);
                let mountain_factor = (mountains - 0.5).max(0.0) * 2.0;

                let height = (base_height * 30.0 + detail * 8.0 + mountain_factor * 40.0 + 20.0) as usize;
                let height = height.min(CHUNK_Y - 1);

                // Bedrock
                chunk.set(x, 0, z, BlockType::Bedrock);

                for y in 1..=height {
                    let block = if y == height {
                        if height < SEA_LEVEL - 1 {
                            BlockType::Sand
                        } else if height > 70 {
                            BlockType::Snow
                        } else {
                            BlockType::Grass
                        }
                    } else if y > height.saturating_sub(4) {
                        if height < SEA_LEVEL - 1 {
                            BlockType::Sand
                        } else {
                            BlockType::Dirt
                        }
                    } else if y < 5 {
                        BlockType::Bedrock
                    } else {
                        // Ore generation
                        let ore_noise = self.noise.hash2d(
                            wx as i32 + y as i32 * 137,
                            wz as i32 + y as i32 * 251,
                        );
                        if y < 12 && ore_noise < 0.02 {
                            BlockType::Diamond
                        } else if y < 32 && ore_noise < 0.04 {
                            BlockType::Gold
                        } else if y < 48 && ore_noise < 0.08 {
                            BlockType::Iron
                        } else if ore_noise < 0.10 {
                            BlockType::Coal
                        } else if ore_noise < 0.14 {
                            BlockType::Gravel
                        } else {
                            BlockType::Stone
                        }
                    };
                    chunk.set(x, y, z, block);
                }

                // Water fill
                for y in (height + 1)..SEA_LEVEL {
                    chunk.set(x, y, z, BlockType::Water);
                }

                // Trees
                if height >= SEA_LEVEL && height < 65 {
                    let tree_val = self.tree_noise.hash2d(wx as i32, wz as i32);
                    if tree_val < 0.012 && x >= 2 && x < CHUNK_X - 2 && z >= 2 && z < CHUNK_Z - 2 {
                        self.place_tree(&mut chunk, x, height + 1, z);
                    }
                }
            }
        }

        chunk
    }

    fn place_tree(&self, chunk: &mut Chunk, x: usize, y: usize, z: usize) {
        let trunk_height = 5;

        // Trunk
        for dy in 0..trunk_height {
            if y + dy < CHUNK_Y {
                chunk.set(x, y + dy, z, BlockType::Wood);
            }
        }

        // Leaves (sphere-ish)
        let leaf_start = trunk_height - 2;
        for dy in leaf_start..(trunk_height + 2) {
            let radius = if dy >= trunk_height { 1 } else { 2 };
            for dz in -(radius as i32)..=(radius as i32) {
                for dx in -(radius as i32)..=(radius as i32) {
                    if dx * dx + dz * dz <= radius * radius + 1 {
                        let lx = (x as i32 + dx) as usize;
                        let ly = y + dy;
                        let lz = (z as i32 + dz) as usize;
                        if lx < CHUNK_X && ly < CHUNK_Y && lz < CHUNK_Z {
                            if chunk.get(lx, ly, lz) == BlockType::Air {
                                chunk.set(lx, ly, lz, BlockType::Leaves);
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn get_block(&self, wx: i32, wy: i32, wz: i32) -> BlockType {
        if wy < 0 || wy >= CHUNK_Y as i32 {
            return BlockType::Air;
        }

        let cx = if wx < 0 { (wx + 1) / CHUNK_X as i32 - 1 } else { wx / CHUNK_X as i32 };
        let cz = if wz < 0 { (wz + 1) / CHUNK_Z as i32 - 1 } else { wz / CHUNK_Z as i32 };
        let lx = ((wx % CHUNK_X as i32) + CHUNK_X as i32) as usize % CHUNK_X;
        let lz = ((wz % CHUNK_Z as i32) + CHUNK_Z as i32) as usize % CHUNK_Z;

        self.chunks
            .get(&ChunkPos::new(cx, cz))
            .map(|c| c.get(lx, wy as usize, lz))
            .unwrap_or(BlockType::Air)
    }

    pub fn set_block(&mut self, wx: i32, wy: i32, wz: i32, block: BlockType) {
        if wy < 0 || wy >= CHUNK_Y as i32 {
            return;
        }

        let cx = if wx < 0 { (wx + 1) / CHUNK_X as i32 - 1 } else { wx / CHUNK_X as i32 };
        let cz = if wz < 0 { (wz + 1) / CHUNK_Z as i32 - 1 } else { wz / CHUNK_Z as i32 };
        let lx = ((wx % CHUNK_X as i32) + CHUNK_X as i32) as usize % CHUNK_X;
        let lz = ((wz % CHUNK_Z as i32) + CHUNK_Z as i32) as usize % CHUNK_Z;

        if let Some(chunk) = self.chunks.get_mut(&ChunkPos::new(cx, cz)) {
            chunk.set(lx, wy as usize, lz, block);
        }

        // Mark neighbor chunks dirty if on border
        if lx == 0 { self.mark_dirty(ChunkPos::new(cx - 1, cz)); }
        if lx == CHUNK_X - 1 { self.mark_dirty(ChunkPos::new(cx + 1, cz)); }
        if lz == 0 { self.mark_dirty(ChunkPos::new(cx, cz - 1)); }
        if lz == CHUNK_Z - 1 { self.mark_dirty(ChunkPos::new(cx, cz + 1)); }
    }

    fn mark_dirty(&mut self, pos: ChunkPos) {
        if let Some(chunk) = self.chunks.get_mut(&pos) {
            chunk.dirty = true;
        }
    }

    pub fn get_neighbors(&self, pos: ChunkPos) -> ChunkNeighbors {
        ChunkNeighbors {
            north: self.chunks.get(&ChunkPos::new(pos.x, pos.z + 1)),
            south: self.chunks.get(&ChunkPos::new(pos.x, pos.z - 1)),
            east: self.chunks.get(&ChunkPos::new(pos.x + 1, pos.z)),
            west: self.chunks.get(&ChunkPos::new(pos.x - 1, pos.z)),
        }
    }

    /// Raycast for block interaction - returns (hit_pos, face_pos_for_placement)
    pub fn raycast(
        &self,
        origin: [f32; 3],
        direction: [f32; 3],
        max_dist: f32,
    ) -> Option<(RayHit)> {
        // DDA voxel traversal
        let mut x = origin[0].floor() as i32;
        let mut y = origin[1].floor() as i32;
        let mut z = origin[2].floor() as i32;

        let step_x = if direction[0] > 0.0 { 1i32 } else { -1 };
        let step_y = if direction[1] > 0.0 { 1i32 } else { -1 };
        let step_z = if direction[2] > 0.0 { 1i32 } else { -1 };

        let inv_dx = if direction[0] != 0.0 { 1.0 / direction[0].abs() } else { f32::MAX };
        let inv_dy = if direction[1] != 0.0 { 1.0 / direction[1].abs() } else { f32::MAX };
        let inv_dz = if direction[2] != 0.0 { 1.0 / direction[2].abs() } else { f32::MAX };

        let mut t_max_x = if direction[0] > 0.0 {
            ((x as f32 + 1.0) - origin[0]) * inv_dx
        } else {
            (origin[0] - x as f32) * inv_dx
        };
        let mut t_max_y = if direction[1] > 0.0 {
            ((y as f32 + 1.0) - origin[1]) * inv_dy
        } else {
            (origin[1] - y as f32) * inv_dy
        };
        let mut t_max_z = if direction[2] > 0.0 {
            ((z as f32 + 1.0) - origin[2]) * inv_dz
        } else {
            (origin[2] - z as f32) * inv_dz
        };

        let mut prev_x = x;
        let mut prev_y = y;
        let mut prev_z = z;
        let mut dist = 0.0;

        while dist < max_dist {
            let block = self.get_block(x, y, z);
            if block.is_solid() {
                return Some(RayHit {
                    block_pos: [x, y, z],
                    place_pos: [prev_x, prev_y, prev_z],
                    block_type: block,
                });
            }

            prev_x = x;
            prev_y = y;
            prev_z = z;

            if t_max_x < t_max_y {
                if t_max_x < t_max_z {
                    dist = t_max_x;
                    x += step_x;
                    t_max_x += inv_dx;
                } else {
                    dist = t_max_z;
                    z += step_z;
                    t_max_z += inv_dz;
                }
            } else {
                if t_max_y < t_max_z {
                    dist = t_max_y;
                    y += step_y;
                    t_max_y += inv_dy;
                } else {
                    dist = t_max_z;
                    z += step_z;
                    t_max_z += inv_dz;
                }
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