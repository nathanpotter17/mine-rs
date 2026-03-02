use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use sdl2::mouse::MouseButton;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use simmerlib::{
    device::{DeviceContext, WINDOW_NAME},
    world::{World, BlockType, CHUNK_X, CHUNK_Z, RENDER_DISTANCE, GENERATION_DISTANCE, mesh_chunk, ChunkPos},
    player::{Player, PlayerInput},
    renderer::{Renderer, ViewUBO, Frustum},
    ui::{UIManager, UIElement, Button},
};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const ENABLE_VALIDATION: bool = cfg!(debug_assertions);

// ===== UI Button IDs =====

const BTN_RESUME: u32 = 1;
const BTN_TOGGLE_FLY: u32 = 2;
const BTN_QUIT: u32 = 3;

// ===== UI Action Queue =====
// Callbacks fire inside UIManager::update_buttons() which holds &mut UIManager.
// We can't mutate game state from inside a callback closure, so callbacks push
// lightweight action tokens into a shared queue. The game loop drains it each frame.

#[derive(Debug, Clone, Copy)]
enum UIAction {
    Resume,
    ToggleFly,
    Quit,
}

type ActionQueue = Arc<Mutex<Vec<UIAction>>>;

fn new_action_queue() -> ActionQueue {
    Arc::new(Mutex::new(Vec::new()))
}

// ===== UI Setup =====

/// Populate the UIManager with a pause-menu layout.
/// Called once at init; visibility toggled at runtime with U key.
///
/// Layout (normalized screen coords, 0-1):
///   - Full-screen dim overlay
///   - Dark centered panel
///   - Three stacked buttons: Resume (green), Toggle Fly (blue), Quit (red)
fn setup_pause_menu(manager: &mut UIManager, actions: &ActionQueue) {
    manager.clear_elements();
    manager.clear_buttons();

    // Full-screen dim overlay
    manager.add_element(UIElement::new(
        0.0, 0.0, 1.0, 1.0,
        [0.0, 0.0, 0.0, 0.45],
    ));

    // Dark panel behind buttons
    manager.add_element(UIElement::new(
        0.34, 0.28, 0.32, 0.34,
        [0.08, 0.08, 0.12, 0.75],
    ));

    // Accent bar at top of panel
    manager.add_element(UIElement::new(
        0.34, 0.28, 0.32, 0.008,
        [0.4, 0.6, 1.0, 0.9],
    ));

    // --- Buttons ---
    // Centered horizontally: x = (1.0 - 0.20) / 2 = 0.40

    // Resume — green
    manager.add_button(Button::new(
        BTN_RESUME,
        0.40, 0.34, 0.20, 0.055,
        [0.18, 0.55, 0.22, 0.92],
    ));

    // Toggle Flying — blue
    manager.add_button(Button::new(
        BTN_TOGGLE_FLY,
        0.40, 0.42, 0.20, 0.055,
        [0.20, 0.35, 0.65, 0.92],
    ));

    // Quit — red
    manager.add_button(Button::new(
        BTN_QUIT,
        0.40, 0.50, 0.20, 0.055,
        [0.65, 0.18, 0.18, 0.92],
    ));

    // Wire up callbacks → action queue
    let a = actions.clone();
    manager.set_button_callback(BTN_RESUME, move || {
        a.lock().unwrap().push(UIAction::Resume);
    });

    let a = actions.clone();
    manager.set_button_callback(BTN_TOGGLE_FLY, move || {
        a.lock().unwrap().push(UIAction::ToggleFly);
    });

    let a = actions.clone();
    manager.set_button_callback(BTN_QUIT, move || {
        a.lock().unwrap().push(UIAction::Quit);
    });

    // Start hidden — user presses U to open
    manager.visible = false;
}

// ===== UI Toggle =====

/// Toggle UI on/off and manage mouse capture state accordingly.
/// When UI opens: release mouse so cursor is visible for button interaction.
/// When UI closes: recapture mouse for gameplay.
fn toggle_ui(
    manager: &mut UIManager,
    sdl_mouse: &sdl2::mouse::MouseUtil,
    mouse_captured: &mut bool,
) {
    let opening = !manager.visible;
    manager.visible = opening;
    manager.dirty = true; // force vertex rebuild

    if opening {
        // Release mouse for UI interaction
        if *mouse_captured {
            sdl_mouse.set_relative_mouse_mode(false);
            *mouse_captured = false;
        }
    } else {
        // Recapture mouse for gameplay
        if !*mouse_captured {
            sdl_mouse.set_relative_mouse_mode(true);
            *mouse_captured = true;
        }
    }
}

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
    world.generate_around_immediate(player.position[0], player.position[2]);
    println!(" done ({} chunks)", world.chunks.len());

    // Find spawn height
    for y in (0..128).rev() {
        if world.get_block(spawn_x as i32, y, spawn_z as i32).is_solid() {
            player.position[1] = y as f32 + 2.0;
            break;
        }
    }

    // Initial mesh upload (only mesh within render distance, buffer ring stays unmeshed)
    print!("  Meshing chunks...");
    let player_cx = (player.position[0] / CHUNK_X as f32).floor() as i32;
    let player_cz = (player.position[2] / CHUNK_Z as f32).floor() as i32;
    let chunk_positions: Vec<ChunkPos> = world.chunks.keys()
        .filter(|p| (p.x - player_cx).abs() <= RENDER_DISTANCE && (p.z - player_cz).abs() <= RENDER_DISTANCE)
        .copied().collect();
    for pos in &chunk_positions {
        if world.has_all_neighbors(*pos) {
            if let Some(chunk) = world.chunks.get(pos) {
                let neighbors = world.get_neighbors(*pos);
                let mesh = mesh_chunk(chunk, &neighbors);
                renderer.upload_chunk_mesh(&device_ctx, *pos, &mesh)?;
            }
        }
    }
    // Flush all initial staging uploads to DEVICE_LOCAL
    renderer.flush_uploads(&device_ctx)?;
    // Mark only meshed chunks as clean (buffer ring stays dirty until it enters view)
    for pos in &chunk_positions {
        if let Some(chunk) = world.chunks.get_mut(pos) {
            chunk.dirty = false;
        }
    }
    println!(" done");

    // UI setup
    let ui_actions = new_action_queue();
    setup_pause_menu(&mut renderer.ui_manager, &ui_actions);
    println!("✓ UI system ready (press U to toggle)");

    // Mouse capture
    sdl.mouse().set_relative_mouse_mode(true);
    let mut mouse_captured = true;

    let mut event_pump = sdl.event_pump()?;
    let mut last_frame = Instant::now();
    let mut frame_count = 0u64;
    let mut fps_timer = Instant::now();
    let mut last_fps = 0;

    let mut game_time: f32 = 0.30;  // start at "morning" (sunrise = 0.25, noon = 0.5)
    let day_length_seconds: f32 = 600.0;  // 10 real minutes = 1 game day

    let mut input = PlayerInput::default();

    // Frame timing accumulators (printed every second)
    let mut t_events = 0.0f64;
    let mut t_update = 0.0f64;
    let mut t_world = 0.0f64;
    let mut t_upload = 0.0f64;
    let mut t_unload = 0.0f64;
    let mut t_render = 0.0f64;
    let mut t_frames = 0u32;

    println!("\n═══════════════════════════════════");
    println!("  Controls:");
    println!("  WASD       - Move");
    println!("  Mouse      - Look");
    println!("  Space      - Jump / Fly up");
    println!("  LShift     - Fly down");
    println!("  F          - Toggle flying");
    println!("  U          - Toggle UI menu");
    println!("  LClick     - Break block");
    println!("  RClick     - Place block");
    println!("  T          - Place torch");
    println!("  Scroll     - Change block type");
    println!("  Escape     - Close UI / Release mouse / Quit");
    println!("  Q          - Quit");
    println!("═══════════════════════════════════\n");
    println!("Starting render loop...\n");

    'running: loop {
        let now = Instant::now();
        let dt = now.duration_since(last_frame).as_secs_f32().min(0.1);
        last_frame = now;
        game_time = (game_time + dt / day_length_seconds) % 1.0;

        // FPS counter
        frame_count += 1;
        t_frames += 1;
        if fps_timer.elapsed().as_secs_f32() >= 1.0 {
            last_fps = frame_count;
            frame_count = 0;
            fps_timer = Instant::now();

            let pos = player.position;
            let mode = if player.flying { "FLY" } else { "WALK" };
            let _looking_at = world.raycast(player.eye_position(), player.forward(), 6.0)
                .map(|hit| format!("{:?}", hit.block_type))
                .unwrap_or_else(|| "---".to_string());

            t_events = 0.0; t_update = 0.0; t_world = 0.0;
            t_upload = 0.0; t_unload = 0.0; t_render = 0.0; t_frames = 0;

            let time_hours = (game_time * 24.0) as u32;
            let time_mins = ((game_time * 24.0 - time_hours as f32) * 60.0) as u32;
            let ui_tag = if renderer.ui_manager.visible { " | [UI]" } else { "" };
            window.set_title(&format!(
                "Voxel World | FPS: {} | Pos: ({:.0}, {:.0}, {:.0}) | {} | Chunks: {} | {:02}:{:02} | Place: {:?}{}",
                last_fps, pos[0], pos[1], pos[2], mode,
                renderer.chunk_count(), time_hours, time_mins, player.selected_block, ui_tag,
            )).ok();
        }

        // Reset per-frame input
        input.toggle_fly = false;
        let ui_visible = renderer.ui_manager.visible;

        // ===== Events =====
        let t0 = Instant::now();
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'running,

                Event::KeyDown { keycode: Some(key), repeat: false, .. } => match key {
                    Keycode::Q => break 'running,

                    Keycode::U => {
                        toggle_ui(&mut renderer.ui_manager, &sdl.mouse(), &mut mouse_captured);
                    }

                    Keycode::Escape => {
                        if renderer.ui_manager.visible {
                            // Close UI, recapture mouse
                            toggle_ui(&mut renderer.ui_manager, &sdl.mouse(), &mut mouse_captured);
                        } else if mouse_captured {
                            sdl.mouse().set_relative_mouse_mode(false);
                            mouse_captured = false;
                        } else {
                            break 'running;
                        }
                    }

                    // Movement keys — only process during gameplay (not when UI is open)
                    Keycode::W if !ui_visible => input.forward = true,
                    Keycode::S if !ui_visible => input.backward = true,
                    Keycode::A if !ui_visible => input.left = true,
                    Keycode::D if !ui_visible => input.right = true,
                    Keycode::Space if !ui_visible => input.jump = true,
                    Keycode::LShift if !ui_visible => input.sneak = true,
                    Keycode::F if !ui_visible => input.toggle_fly = true,
                    Keycode::T if !ui_visible => {
                        // Quick-place torch
                        if mouse_captured {
                            let eye = player.eye_position();
                            let dir = player.forward();
                            if let Some(hit) = world.raycast(eye, dir, 6.0) {
                                world.set_block(
                                    hit.place_pos[0], hit.place_pos[1], hit.place_pos[2],
                                    BlockType::Torch,
                                );
                            }
                        }
                    }
                    _ => {}
                },

                Event::KeyUp { keycode: Some(key), .. } => {
                    // Always release keys (prevents stuck keys if UI opened mid-press)
                    match key {
                        Keycode::W => input.forward = false,
                        Keycode::S => input.backward = false,
                        Keycode::A => input.left = false,
                        Keycode::D => input.right = false,
                        Keycode::Space => input.jump = false,
                        Keycode::LShift => input.sneak = false,
                        _ => {}
                    }
                },

                // ── Mouse motion ──
                Event::MouseMotion { x, y, xrel, yrel, .. } => {
                    if mouse_captured {
                        // Gameplay look
                        player.look(xrel as f32, yrel as f32);
                    } else if ui_visible {
                        // Feed absolute position to UI for hover tracking
                        renderer.ui_manager.update_mouse_position(
                            x as f32, y as f32,
                            device_ctx.swapchain_extent.width as f32,
                            device_ctx.swapchain_extent.height as f32,
                        );
                    }
                }

                // ── Mouse button down ──
                Event::MouseButtonDown { mouse_btn, x, y, .. } => {
                    if ui_visible && !mouse_captured {
                        // UI interaction mode: update position and press state
                        if mouse_btn == MouseButton::Left {
                            renderer.ui_manager.update_mouse_position(
                                x as f32, y as f32,
                                device_ctx.swapchain_extent.width as f32,
                                device_ctx.swapchain_extent.height as f32,
                            );
                            renderer.ui_manager.set_mouse_down(true);
                        }
                    } else if !mouse_captured {
                        // No UI visible, click recaptures mouse for gameplay
                        sdl.mouse().set_relative_mouse_mode(true);
                        mouse_captured = true;
                    } else {
                        // Gameplay mode — block interaction
                        let eye = player.eye_position();
                        let dir = player.forward();

                        match mouse_btn {
                            MouseButton::Left => {
                                if let Some(hit) = world.raycast(eye, dir, 6.0) {
                                    world.set_block(
                                        hit.block_pos[0], hit.block_pos[1], hit.block_pos[2],
                                        BlockType::Air,
                                    );
                                }
                            }
                            MouseButton::Right => {
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

                // ── Mouse button up ──
                Event::MouseButtonUp { mouse_btn: MouseButton::Left, .. } => {
                    if ui_visible && !mouse_captured {
                        renderer.ui_manager.set_mouse_down(false);
                    }
                }

                Event::MouseWheel { y, .. } if !ui_visible => {
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

        // ===== Process UI button state transitions =====
        renderer.ui_manager.update_buttons();

        // ===== Drain UI action queue =====
        {
            let mut actions = ui_actions.lock().unwrap();
            for action in actions.drain(..) {
                match action {
                    UIAction::Resume => {
                        // Close UI and return to gameplay
                        if renderer.ui_manager.visible {
                            toggle_ui(&mut renderer.ui_manager, &sdl.mouse(), &mut mouse_captured);
                        }
                    }
                    UIAction::ToggleFly => {
                        player.flying = !player.flying;
                        player.velocity[1] = 0.0;
                    }
                    UIAction::Quit => {
                        break 'running;
                    }
                }
            }
        }

        // ===== Update player =====
        t_events += t0.elapsed().as_secs_f64();
        let t0 = Instant::now();

        // Only update player physics when UI is closed
        if !renderer.ui_manager.visible {
            player.update(dt, &input, &world);
        }

        // === Async world update ===
        t_update += t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        world.generate_around(player.position[0], player.position[2]);

        // === Drain completed meshes from background workers and upload to GPU ===
        t_world += t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let completed_meshes = world.drain_completed_meshes();
        for mesh in &completed_meshes {
            renderer.upload_chunk_mesh(&device_ctx, mesh.pos, mesh)?;
        }
        // Submit all staging→device copies in one command buffer
        renderer.flush_uploads(&device_ctx)?;

        // Remove GPU data for chunks that left render distance
        t_upload += t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let pcx = (player.position[0] / CHUNK_X as f32).floor() as i32;
        let pcz = (player.position[2] / CHUNK_Z as f32).floor() as i32;
        let gpu_chunks: Vec<ChunkPos> = renderer.loaded_chunk_positions();
        for pos in gpu_chunks {
            if (pos.x - pcx).abs() > RENDER_DISTANCE + 1
            || (pos.z - pcz).abs() > RENDER_DISTANCE + 1 {
                renderer.remove_chunk(pos);
                // Re-mark dirty so chunk gets re-meshed when back in range
                if let Some(chunk) = world.chunks.get_mut(&pos) {
                    chunk.dirty = true;
                }
            }
        }

        // Render
        t_unload += t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let aspect = device_ctx.swapchain_extent.width as f32 / device_ctx.swapchain_extent.height as f32;
        let eye = player.eye_position();

        let view = player.get_view_matrix();
        let proj = player.get_projection_matrix(aspect);
        let frustum = Frustum::from_view_proj(&view, &proj);

        let ubo = ViewUBO {
            view,
            proj,
            camera_pos: [eye[0], eye[1], eye[2], 0.0],
        };

        renderer.render(&device_ctx, ubo, &frustum, game_time)?;
        t_render += t0.elapsed().as_secs_f64();
    }

    println!("\nShutting down...");
    unsafe { device_ctx.device.device_wait_idle()?; }
    Ok(())
}