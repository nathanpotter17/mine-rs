use crate::world::{World, BlockType, CHUNK_Y};

// ===== Camera Mode =====

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CameraMode {
    FirstPerson,
    ThirdPerson,
}

// ===== Player Model Vertex =====
// 36-byte vertex for the player model box (position + color + normal)

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PlayerVertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub normal: [f32; 3],
}

pub struct Player {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub yaw: f32,   // Horizontal rotation (radians)
    pub pitch: f32,  // Vertical rotation (radians)
    pub on_ground: bool,
    pub flying: bool,
    pub selected_block: BlockType,
    
    // Movement config
    pub move_speed: f32,
    pub fly_speed: f32,
    pub jump_velocity: f32,
    pub gravity: f32,
    pub mouse_sensitivity: f32,
    
    // Player dimensions
    pub width: f32,
    pub height: f32,
    pub eye_height: f32,

    // Camera
    pub camera_mode: CameraMode,
    pub third_person_distance: f32,
    pub third_person_height: f32,
}

impl Player {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            position: [x, y, z],
            velocity: [0.0; 3],
            yaw: 0.0,
            pitch: 0.0,
            on_ground: false,
            flying: true, // Start flying for easier exploration
            selected_block: BlockType::Stone,
            move_speed: 5.5,
            fly_speed: 15.0,
            jump_velocity: 8.5,
            gravity: 28.0,
            mouse_sensitivity: 0.003,
            width: 0.6,
            height: 1.8,
            eye_height: 1.62,
            camera_mode: CameraMode::FirstPerson,
            third_person_distance: 5.0,
            third_person_height: 1.5,
        }
    }

    pub fn eye_position(&self) -> [f32; 3] {
        [
            self.position[0],
            self.position[1] + self.eye_height,
            self.position[2],
        ]
    }

    pub fn forward(&self) -> [f32; 3] {
        [
            -(self.yaw.sin()) * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.cos() * self.pitch.cos(),
        ]
    }

    fn flat_forward(&self) -> [f32; 3] {
        let f = [-(self.yaw.sin()), 0.0, self.yaw.cos()];
        let len = (f[0] * f[0] + f[2] * f[2]).sqrt();
        if len > 0.001 {
            [f[0] / len, 0.0, f[2] / len]
        } else {
            [0.0, 0.0, 1.0]
        }
    }

    fn right(&self) -> [f32; 3] {
        let f = self.flat_forward();
        [-f[2], 0.0, f[0]]
    }

    pub fn look(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * self.mouse_sensitivity;
        self.pitch -= dy * self.mouse_sensitivity;
        self.pitch = self.pitch.clamp(-1.55, 1.55); // ~89 degrees
    }

    pub fn update(&mut self, dt: f32, input: &PlayerInput, world: &World) {
        let speed = if self.flying { self.fly_speed } else { self.move_speed };
        let forward = self.flat_forward();
        let right = self.right();

        // Calculate desired horizontal movement
        let mut move_x = 0.0f32;
        let mut move_z = 0.0f32;

        if input.forward {
            move_x += forward[0];
            move_z += forward[2];
        }
        if input.backward {
            move_x -= forward[0];
            move_z -= forward[2];
        }
        if input.left {
            move_x -= right[0];
            move_z -= right[2];
        }
        if input.right {
            move_x += right[0];
            move_z += right[2];
        }

        // Normalize horizontal movement
        let len = (move_x * move_x + move_z * move_z).sqrt();
        if len > 0.001 {
            move_x = (move_x / len) * speed;
            move_z = (move_z / len) * speed;
        }

        if self.flying {
            self.velocity[0] = move_x;
            self.velocity[2] = move_z;
            self.velocity[1] = 0.0;

            if input.jump { self.velocity[1] = speed; }
            if input.sneak { self.velocity[1] = -speed; }

            // Apply movement directly in fly mode
            self.position[0] += self.velocity[0] * dt;
            self.position[1] += self.velocity[1] * dt;
            self.position[2] += self.velocity[2] * dt;
        } else {
            // Walking mode with gravity
            self.velocity[0] = move_x;
            self.velocity[2] = move_z;

            // Gravity
            self.velocity[1] -= self.gravity * dt;

            // Jump
            if input.jump && self.on_ground {
                self.velocity[1] = self.jump_velocity;
                self.on_ground = false;
            }

            // Apply movement with collision
            self.move_with_collision(dt, world);
        }

        // Toggle flying
        if input.toggle_fly {
            self.flying = !self.flying;
            self.velocity[1] = 0.0;
        }

        // Keep in world bounds
        if self.position[1] < -10.0 {
            self.position[1] = 80.0;
            self.velocity[1] = 0.0;
        }
    }

    fn move_with_collision(&mut self, dt: f32, world: &World) {
        let half_w = self.width / 2.0;
        
        // Move X
        let new_x = self.position[0] + self.velocity[0] * dt;
        if !self.check_collision(world, new_x, self.position[1], self.position[2], half_w) {
            self.position[0] = new_x;
        } else {
            self.velocity[0] = 0.0;
        }

        // Move Y
        let new_y = self.position[1] + self.velocity[1] * dt;
        if !self.check_collision(world, self.position[0], new_y, self.position[2], half_w) {
            self.position[1] = new_y;
            self.on_ground = false;
        } else {
            if self.velocity[1] < 0.0 {
                self.on_ground = true;
                // Snap to ground
                self.position[1] = (self.position[1]).floor() + 0.001;
            }
            self.velocity[1] = 0.0;
        }

        // Move Z
        let new_z = self.position[2] + self.velocity[2] * dt;
        if !self.check_collision(world, self.position[0], self.position[1], new_z, half_w) {
            self.position[2] = new_z;
        } else {
            self.velocity[2] = 0.0;
        }
    }

    fn check_collision(&self, world: &World, x: f32, y: f32, z: f32, half_w: f32) -> bool {
        // Check all blocks the player AABB overlaps
        let min_x = (x - half_w).floor() as i32;
        let max_x = (x + half_w).floor() as i32;
        let min_y = y.floor() as i32;
        let max_y = (y + self.height).floor() as i32;
        let min_z = (z - half_w).floor() as i32;
        let max_z = (z + half_w).floor() as i32;

        for by in min_y..=max_y {
            for bz in min_z..=max_z {
                for bx in min_x..=max_x {
                    if world.get_block(bx, by, bz).is_solid() {
                        return true;
                    }
                }
            }
        }
        false
    }

    // ===== Camera Matrix Methods =====

    /// First-person view matrix (existing behavior)
    pub fn get_view_matrix(&self) -> [[f32; 4]; 4] {
        let eye = self.eye_position();
        let dir = self.forward();
        let target = [eye[0] + dir[0], eye[1] + dir[1], eye[2] + dir[2]];
        let up = [0.0f32, 1.0, 0.0];

        look_at(eye, target, up)
    }

    pub fn get_projection_matrix(&self, aspect: f32) -> [[f32; 4]; 4] {
        let fov = 70.0f32.to_radians();
        let near = 0.1;
        let far = 500.0;
        perspective(fov, aspect, near, far)
    }

    /// Third-person camera position: behind and above the player along the look direction.
    /// Returns (camera_pos, look_target) pair.
    pub fn get_third_person_camera(&self) -> ([f32; 3], [f32; 3]) {
        let eye = self.eye_position();
        let fwd = self.forward();

        let cam_pos = [
            eye[0] - fwd[0] * self.third_person_distance,
            eye[1] - fwd[1] * self.third_person_distance + self.third_person_height,
            eye[2] - fwd[2] * self.third_person_distance,
        ];

        (cam_pos, eye)
    }

    /// Third-person view matrix: camera orbits behind player.
    pub fn get_third_person_view_matrix(&self) -> [[f32; 4]; 4] {
        let (cam_pos, target) = self.get_third_person_camera();
        look_at(cam_pos, target, [0.0, 1.0, 0.0])
    }

    /// Returns the camera position for the current mode (used for ViewUBO.camera_pos).
    pub fn camera_position(&self) -> [f32; 3] {
        match self.camera_mode {
            CameraMode::FirstPerson => self.eye_position(),
            CameraMode::ThirdPerson => self.get_third_person_camera().0,
        }
    }

    /// Returns the view matrix for the current camera mode.
    pub fn current_view_matrix(&self) -> [[f32; 4]; 4] {
        match self.camera_mode {
            CameraMode::FirstPerson => self.get_view_matrix(),
            CameraMode::ThirdPerson => self.get_third_person_view_matrix(),
        }
    }

    // ===== Player Model =====

    /// Model matrix: translates to player.position and rotates by yaw around Y.
    /// The mesh is defined with feet at origin, so translation places feet at position.
    pub fn get_model_matrix(&self) -> [[f32; 4]; 4] {
        let c = self.yaw.cos();
        let s = self.yaw.sin();
        let p = self.position;
        // Column-major: rotation around Y then translate
        [
            [ c,   0.0, -s,  0.0],
            [ 0.0, 1.0,  0.0, 0.0],
            [ s,   0.0,  c,  0.0],
            [p[0], p[1], p[2], 1.0],
        ]
    }

    pub fn toggle_camera_mode(&mut self) {
        self.camera_mode = match self.camera_mode {
            CameraMode::FirstPerson => CameraMode::ThirdPerson,
            CameraMode::ThirdPerson => CameraMode::FirstPerson,
        };
    }

    /// Generate a static player model mesh: a box matching player dimensions.
    /// Feet at origin (0,0,0), top at (0, height, 0).
    /// Returns (vertices, indices).
    pub fn generate_player_mesh(&self) -> (Vec<PlayerVertex>, Vec<u32>) {
        let hw = self.width / 2.0;   // half-width  (X)
        let hd = self.width / 2.0;   // half-depth  (Z), same as width for square cross-section
        let h  = self.height;         // full height (Y)

        // 8 corner positions: bottom 4 (y=0), top 4 (y=h)
        let p = [
            [-hw, 0.0, -hd], // 0: bottom-left-back
            [ hw, 0.0, -hd], // 1: bottom-right-back
            [ hw, 0.0,  hd], // 2: bottom-right-front
            [-hw, 0.0,  hd], // 3: bottom-left-front
            [-hw,   h, -hd], // 4: top-left-back
            [ hw,   h, -hd], // 5: top-right-back
            [ hw,   h,  hd], // 6: top-right-front
            [-hw,   h,  hd], // 7: top-left-front
        ];

        // Face definitions: (corner_indices, normal, color)
        // Colors: teal body with subtle per-face variation for visual clarity
        let faces: [([usize; 4], [f32; 3], [f32; 3]); 6] = [
            // Front (+Z): teal
            ([3, 2, 6, 7], [ 0.0,  0.0,  1.0], [0.22, 0.62, 0.72]),
            // Back (-Z): darker teal
            ([1, 0, 4, 5], [ 0.0,  0.0, -1.0], [0.18, 0.50, 0.58]),
            // Top (+Y): lighter / head area
            ([7, 6, 5, 4], [ 0.0,  1.0,  0.0], [0.75, 0.60, 0.48]),
            // Bottom (-Y): dark
            ([0, 1, 2, 3], [ 0.0, -1.0,  0.0], [0.20, 0.20, 0.25]),
            // Right (+X): medium teal
            ([2, 1, 5, 6], [ 1.0,  0.0,  0.0], [0.20, 0.55, 0.65]),
            // Left (-X): medium teal
            ([0, 3, 7, 4], [-1.0,  0.0,  0.0], [0.20, 0.55, 0.65]),
        ];

        let mut vertices = Vec::with_capacity(24);
        let mut indices = Vec::with_capacity(36);

        for (corners, normal, color) in &faces {
            let base = vertices.len() as u32;
            for &ci in corners {
                vertices.push(PlayerVertex {
                    position: p[ci],
                    color: *color,
                    normal: *normal,
                });
            }
            // Two triangles per face: 0-1-2, 2-3-0
            indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
        }

        (vertices, indices)
    }

    /// Cycle through available block types
    pub fn next_block(&mut self) {
        self.selected_block = match self.selected_block {
            BlockType::Stone => BlockType::Dirt,
            BlockType::Dirt => BlockType::Grass,
            BlockType::Grass => BlockType::Sand,
            BlockType::Sand => BlockType::Wood,
            BlockType::Wood => BlockType::Leaves,
            BlockType::Leaves => BlockType::Gravel,
            BlockType::Gravel => BlockType::Snow,
            BlockType::Snow => BlockType::Torch,
            BlockType::Torch => BlockType::Stone,
            _ => BlockType::Stone,
        };
    }

    pub fn prev_block(&mut self) {
        self.selected_block = match self.selected_block {
            BlockType::Stone => BlockType::Torch,
            BlockType::Torch => BlockType::Snow,
            BlockType::Snow => BlockType::Gravel,
            BlockType::Gravel => BlockType::Leaves,
            BlockType::Leaves => BlockType::Wood,
            BlockType::Wood => BlockType::Sand,
            BlockType::Sand => BlockType::Grass,
            BlockType::Grass => BlockType::Dirt,
            BlockType::Dirt => BlockType::Stone,
            _ => BlockType::Stone,
        };
    }
}

pub struct PlayerInput {
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub jump: bool,
    pub sneak: bool,
    pub toggle_fly: bool,
}

impl Default for PlayerInput {
    fn default() -> Self {
        Self {
            forward: false,
            backward: false,
            left: false,
            right: false,
            jump: false,
            sneak: false,
            toggle_fly: false,
        }
    }
}

// ===== Math helpers =====

fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = normalize(sub(target, eye));
    let s = normalize(cross(f, up));
    let u = cross(s, f);

    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0],
    ]
}

fn perspective(fov: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov / 2.0).tan();
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, -f, 0.0, 0.0], // Vulkan Y-flip
        [0.0, 0.0, far / (near - far), -1.0],
        [0.0, 0.0, (near * far) / (near - far), 0.0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0 { [v[0] / len, v[1] / len, v[2] / len] } else { v }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
}