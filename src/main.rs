use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use sdl2::mouse::MouseButton;
use std::time::Instant;

use simmerlib::{
    device::{DeviceContext, WINDOW_NAME},
    world::{World, BlockType, CHUNK_X, CHUNK_Z, RENDER_DISTANCE, mesh_chunk, ChunkPos},
    player::{Player, PlayerInput},
    renderer::{Renderer, ViewUBO},
};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const ENABLE_VALIDATION: bool = cfg!(debug_assertions);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔════════════════════════════════════╗");
    println!("║      VOXEL WORLD - Minecraft       ║");
    println!("║        Vulkan Renderer              ║");
    println!("╚════════════════════════════════════╝\n");

    // SDL2
    let sdl = sdl2::init()?;
    let video = sdl.video()?;

    let mut window = video.window(WINDOW_NAME, WIDTH, HEIGHT)
        .position_centered().resizable().vulkan().build()?;

    // Vulkan
    let mut device_ctx = DeviceContext::new(&window, ENABLE_VALIDATION)?;
    println!("✓ Vulkan initialized");

    let mut renderer = Renderer::new(&device_ctx)?;
    println!("✓ Renderer initialized");

    // World
    let seed = 42;
    let mut world = World::new(seed);
    println!("✓ World created (seed: {})", seed);

    // Player - spawn above terrain
    let spawn_x = 8.0;
    let spawn_z = 8.0;
    let mut player = Player::new(spawn_x, 80.0, spawn_z);

    // Generate initial chunks
    print!("  Generating terrain...");
    world.generate_around(player.position[0], player.position[2]);
    println!(" done ({} chunks)", world.chunks.len());

    // Find spawn height
    for y in (0..128).rev() {
        if world.get_block(spawn_x as i32, y, spawn_z as i32).is_solid() {
            player.position[1] = y as f32 + 2.0;
            break;
        }
    }

    // Initial mesh upload
    print!("  Meshing chunks...");
    let chunk_positions: Vec<ChunkPos> = world.chunks.keys().copied().collect();
    for pos in &chunk_positions {
        if let Some(chunk) = world.chunks.get(pos) {
            let neighbors = world.get_neighbors(*pos);
            let mesh = mesh_chunk(chunk, &neighbors);
            renderer.upload_chunk_mesh(&device_ctx, *pos, &mesh)?;
        }
    }
    // Mark all as clean
    for chunk in world.chunks.values_mut() {
        chunk.dirty = false;
    }
    println!(" done");

    // Mouse capture
    sdl.mouse().set_relative_mouse_mode(true);
    let mut mouse_captured = true;

    let mut event_pump = sdl.event_pump()?;
    let mut last_frame = Instant::now();
    let mut frame_count = 0u64;
    let mut fps_timer = Instant::now();
    let mut last_fps = 0;

    let mut input = PlayerInput::default();
    let mut meshes_per_frame = 2; // Limit mesh rebuilds per frame

    println!("\n═══════════════════════════════════");
    println!("  Controls:");
    println!("  WASD       - Move");
    println!("  Mouse      - Look");
    println!("  Space      - Jump / Fly up");
    println!("  LShift     - Fly down");
    println!("  F          - Toggle flying");
    println!("  LClick     - Break block");
    println!("  RClick     - Place block");
    println!("  Scroll     - Change block type");
    println!("  Escape     - Release mouse");
    println!("  Q          - Quit");
    println!("═══════════════════════════════════\n");
    println!("Starting render loop...\n");

    'running: loop {
        let now = Instant::now();
        let dt = now.duration_since(last_frame).as_secs_f32().min(0.1);
        last_frame = now;

        // FPS counter
        frame_count += 1;
        if fps_timer.elapsed().as_secs_f32() >= 1.0 {
            last_fps = frame_count;
            frame_count = 0;
            fps_timer = Instant::now();
            
            let pos = player.position;
            let mode = if player.flying { "FLY" } else { "WALK" };
            let looking_at = world.raycast(player.eye_position(), player.forward(), 6.0)
                .map(|hit| format!("{:?}", hit.block_type))
                .unwrap_or_else(|| "---".to_string());
            window.set_title(&format!(
                "Voxel World | FPS: {} | Pos: ({:.0}, {:.0}, {:.0}) | {} | Chunks: {} | Looking: {} | Place: {:?}",
                last_fps, pos[0], pos[1], pos[2], mode,
                renderer.chunk_count(), looking_at, player.selected_block,
            )).ok();
        }

        // Reset per-frame input
        input.toggle_fly = false;

        // Events
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'running,

                Event::KeyDown { keycode: Some(key), repeat: false, .. } => match key {
                    Keycode::Q => break 'running,
                    Keycode::Escape => {
                        if mouse_captured {
                            sdl.mouse().set_relative_mouse_mode(false);
                            mouse_captured = false;
                        } else {
                            break 'running;
                        }
                    }
                    Keycode::W => input.forward = true,
                    Keycode::S => input.backward = true,
                    Keycode::A => input.left = true,
                    Keycode::D => input.right = true,
                    Keycode::Space => input.jump = true,
                    Keycode::LShift => input.sneak = true,
                    Keycode::F => input.toggle_fly = true,
                    _ => {}
                },

                Event::KeyUp { keycode: Some(key), .. } => match key {
                    Keycode::W => input.forward = false,
                    Keycode::S => input.backward = false,
                    Keycode::A => input.left = false,
                    Keycode::D => input.right = false,
                    Keycode::Space => input.jump = false,
                    Keycode::LShift => input.sneak = false,
                    _ => {}
                },

                Event::MouseMotion { xrel, yrel, .. } if mouse_captured => {
                    player.look(xrel as f32, yrel as f32);
                }

                Event::MouseButtonDown { mouse_btn, .. } => {
                    if !mouse_captured {
                        sdl.mouse().set_relative_mouse_mode(true);
                        mouse_captured = true;
                    } else {
                        let eye = player.eye_position();
                        let dir = player.forward();

                        match mouse_btn {
                            MouseButton::Left => {
                                // Break block
                                if let Some(hit) = world.raycast(eye, dir, 6.0) {
                                    world.set_block(
                                        hit.block_pos[0], hit.block_pos[1], hit.block_pos[2],
                                        BlockType::Air,
                                    );
                                }
                            }
                            MouseButton::Right => {
                                // Place block
                                if let Some(hit) = world.raycast(eye, dir, 6.0) {
                                    world.set_block(
                                        hit.place_pos[0], hit.place_pos[1], hit.place_pos[2],
                                        player.selected_block,
                                    );
                                }
                            }
                            _ => {}
                        }
                    }
                }

                Event::MouseWheel { y, .. } => {
                    if y > 0 { player.next_block(); }
                    else if y < 0 { player.prev_block(); }
                }

                Event::Window { win_event: WindowEvent::Resized(w, h), .. } => {
                    device_ctx.recreate_swapchain(w as u32, h as u32);
                    renderer.recreate_framebuffers(&device_ctx)?;
                }

                _ => {}
            }
        }

        // Update player
        player.update(dt, &input, &world);

        // Generate new chunks around player
        world.generate_around(player.position[0], player.position[2]);

        // Rebuild dirty chunk meshes (limited per frame to avoid stuttering)
        let mut rebuilt = 0;
        let dirty_chunks: Vec<ChunkPos> = world.chunks.iter()
            .filter(|(_, c)| c.dirty)
            .map(|(p, _)| *p)
            .collect();

        for pos in dirty_chunks {
            if rebuilt >= meshes_per_frame { break; }
            if let Some(chunk) = world.chunks.get(&pos) {
                let neighbors = world.get_neighbors(pos);
                let mesh = mesh_chunk(chunk, &neighbors);
                renderer.upload_chunk_mesh(&device_ctx, pos, &mesh)?;
            }
            if let Some(chunk) = world.chunks.get_mut(&pos) {
                chunk.dirty = false;
            }
            rebuilt += 1;
        }

        // Remove GPU data for unloaded chunks
        // (handled naturally - chunks removed from world won't get new uploads)

        // Render
        let aspect = device_ctx.swapchain_extent.width as f32 / device_ctx.swapchain_extent.height as f32;
        let eye = player.eye_position();

        let ubo = ViewUBO {
            view: player.get_view_matrix(),
            proj: player.get_projection_matrix(aspect),
            camera_pos: [eye[0], eye[1], eye[2], 0.0],
        };

        renderer.render(&device_ctx, ubo)?;
    }

    println!("\nShutting down...");
    unsafe { device_ctx.device.device_wait_idle()?; }
    Ok(())
}