use ash::{vk, Device};
use std::sync::Arc;
use crate::device::DeviceContext;

// ===== Constants =====

/// Maximum UI elements per frame. 128 quads × 6 verts × 24 bytes = 18,432 bytes.
const MAX_UI_ELEMENTS: usize = 128;
const VERTS_PER_QUAD: usize = 6;
const MAX_VERTICES: usize = MAX_UI_ELEMENTS * VERTS_PER_QUAD;

// ===== Vertex Layout =====

/// Per-vertex data written to the dynamic vertex buffer.
/// Stride = 24 bytes. Matches pipeline vertex input:
///   location 0: R32G32_SFLOAT   (pos,   offset 0)
///   location 1: R32G32B32A32_SFLOAT (color, offset 8)
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct UIVertex {
    pos: [f32; 2],   // clip-space position
    color: [f32; 4], // RGBA
}

impl UIVertex {
    const STRIDE: u32 = std::mem::size_of::<Self>() as u32; // 24
}

// ===== Coordinate Helpers =====

/// Convert normalized screen coords (0-1, top-left origin) to Vulkan clip space.
/// Standard viewport (y=0, height=+H): clip Y=-1 is screen top, Y=+1 is screen bottom.
/// x: 0→-1, 1→+1
/// y: 0→-1 (top), 1→+1 (bottom)
#[inline]
fn screen_to_clip(sx: f32, sy: f32) -> [f32; 2] {
    [sx * 2.0 - 1.0, sy * 2.0 - 1.0]
}

/// Build 6 vertices (two triangles) for one axis-aligned quad.
/// `x, y, w, h` in normalized screen coords. Color applied to all vertices.
fn build_quad_vertices(x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) -> [UIVertex; 6] {
    let tl = screen_to_clip(x, y);
    let tr = screen_to_clip(x + w, y);
    let bl = screen_to_clip(x, y + h);
    let br = screen_to_clip(x + w, y + h);

    [
        // Triangle 1: TL → TR → BR
        UIVertex { pos: tl, color },
        UIVertex { pos: tr, color },
        UIVertex { pos: br, color },
        // Triangle 2: TL → BR → BL
        UIVertex { pos: tl, color },
        UIVertex { pos: br, color },
        UIVertex { pos: bl, color },
    ]
}

// ===== UI Element =====

/// A single colored rectangle in screen space.
#[derive(Copy, Clone, Debug)]
pub struct UIElement {
    /// Position: (x, y) top-left corner in normalized screen coords [0..1]
    pub x: f32,
    pub y: f32,
    /// Size in normalized screen coords
    pub width: f32,
    pub height: f32,
    /// RGBA color [0..1]
    pub color: [f32; 4],
    /// Whether this element is visible
    pub visible: bool,
}

impl UIElement {
    pub fn new(x: f32, y: f32, width: f32, height: f32, color: [f32; 4]) -> Self {
        Self { x, y, width, height, color, visible: true }
    }

    /// Check if a point (in normalized screen coords) is inside this element
    #[inline]
    pub fn contains_point(&self, px: f32, py: f32) -> bool {
        px >= self.x && px <= self.x + self.width
            && py >= self.y && py <= self.y + self.height
    }
}

// ===== Button =====

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ButtonState {
    Normal,
    Hover,
    Pressed,
    Disabled,
}

/// Interactive button wrapping a UIElement with state-driven color transitions.
/// Callbacks stored externally (in UIManager) to keep Button Copy-friendly for the
/// common case. Use `UIManager::set_button_callback` for click handlers.
pub struct Button {
    id: u32,
    element: UIElement,
    state: ButtonState,
    // Color per state
    color_normal: [f32; 4],
    color_hover: [f32; 4],
    color_pressed: [f32; 4],
    color_disabled: [f32; 4],
}

impl Button {
    /// Create a button with automatic hover/press color derivation.
    /// `color` is the normal state color; hover brightens, press darkens.
    pub fn new(id: u32, x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) -> Self {
        let hover = [
            (color[0] * 1.25).min(1.0),
            (color[1] * 1.25).min(1.0),
            (color[2] * 1.25).min(1.0),
            color[3],
        ];
        let pressed = [
            color[0] * 0.75,
            color[1] * 0.75,
            color[2] * 0.75,
            color[3],
        ];
        let disabled = [
            color[0] * 0.5,
            color[1] * 0.5,
            color[2] * 0.5,
            color[3] * 0.6,
        ];

        let mut btn = Self {
            id,
            element: UIElement::new(x, y, w, h, color),
            state: ButtonState::Normal,
            color_normal: color,
            color_hover: hover,
            color_pressed: pressed,
            color_disabled: disabled,
        };
        btn.apply_state_color();
        btn
    }

    /// Create with explicit per-state colors.
    pub fn new_explicit(
        id: u32,
        x: f32, y: f32, w: f32, h: f32,
        normal: [f32; 4],
        hover: [f32; 4],
        pressed: [f32; 4],
        disabled: [f32; 4],
    ) -> Self {
        let mut btn = Self {
            id,
            element: UIElement::new(x, y, w, h, normal),
            state: ButtonState::Normal,
            color_normal: normal,
            color_hover: hover,
            color_pressed: pressed,
            color_disabled: disabled,
        };
        btn.apply_state_color();
        btn
    }

    #[inline]
    pub fn id(&self) -> u32 { self.id }

    #[inline]
    pub fn state(&self) -> ButtonState { self.state }

    #[inline]
    pub fn element(&self) -> &UIElement { &self.element }

    #[inline]
    pub fn contains_point(&self, px: f32, py: f32) -> bool {
        self.element.contains_point(px, py)
    }

    pub fn set_state(&mut self, state: ButtonState) {
        self.state = state;
        self.apply_state_color();
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        if enabled && self.state == ButtonState::Disabled {
            self.state = ButtonState::Normal;
        } else if !enabled {
            self.state = ButtonState::Disabled;
        }
        self.apply_state_color();
    }

    pub fn set_visible(&mut self, visible: bool) {
        self.element.visible = visible;
    }

    fn apply_state_color(&mut self) {
        self.element.color = match self.state {
            ButtonState::Normal => self.color_normal,
            ButtonState::Hover => self.color_hover,
            ButtonState::Pressed => self.color_pressed,
            ButtonState::Disabled => self.color_disabled,
        };
    }
}

// ===== UI Manager (CPU-side state) =====

/// Manages UI elements, buttons, and mouse interaction state.
/// Produces a flat list of UIElements for the renderer each frame.
pub struct UIManager {
    /// Static (non-interactive) elements rendered below buttons
    pub elements: Vec<UIElement>,
    /// Interactive buttons
    pub buttons: Vec<Button>,
    /// Button click callbacks, keyed by button id.
    /// Stored separately so Button doesn't need Box<dyn Fn()>.
    callbacks: Vec<(u32, Box<dyn Fn() + Send>)>,
    /// Current mouse position in normalized screen coords (0-1)
    mouse_x: f32,
    mouse_y: f32,
    /// Whether left mouse button is currently held
    mouse_down: bool,
    /// Button ID that received the initial press (for drag tracking)
    pressed_button_id: Option<u32>,
    /// Set true whenever any element/button changes, cleared after vertex rebuild
    pub dirty: bool,
    /// Master visibility toggle
    pub visible: bool,
}

impl Default for UIManager {
    fn default() -> Self {
        Self {
            elements: Vec::new(),
            buttons: Vec::new(),
            callbacks: Vec::new(),
            mouse_x: 0.0,
            mouse_y: 0.0,
            mouse_down: false,
            pressed_button_id: None,
            dirty: true,
            visible: false,
        }
    }
}

impl UIManager {
    pub fn new() -> Self {
        Self::default()
    }

    // ── Element management ──

    pub fn add_element(&mut self, elem: UIElement) {
        self.elements.push(elem);
        self.dirty = true;
    }

    pub fn clear_elements(&mut self) {
        self.elements.clear();
        self.dirty = true;
    }

    // ── Button management ──

    pub fn add_button(&mut self, button: Button) {
        self.buttons.push(button);
        self.dirty = true;
    }

    /// Register a click callback for a button id.
    /// Replaces any existing callback for that id.
    pub fn set_button_callback<F: Fn() + Send + 'static>(&mut self, button_id: u32, f: F) {
        // Remove existing
        self.callbacks.retain(|(id, _)| *id != button_id);
        self.callbacks.push((button_id, Box::new(f)));
    }

    pub fn find_button(&self, id: u32) -> Option<&Button> {
        self.buttons.iter().find(|b| b.id() == id)
    }

    pub fn find_button_mut(&mut self, id: u32) -> Option<&mut Button> {
        self.buttons.iter_mut().find(|b| b.id() == id)
    }

    pub fn remove_button(&mut self, id: u32) -> bool {
        let before = self.buttons.len();
        if self.pressed_button_id == Some(id) {
            self.pressed_button_id = None;
        }
        self.buttons.retain(|b| b.id() != id);
        self.callbacks.retain(|(cid, _)| *cid != id);
        let removed = self.buttons.len() < before;
        if removed { self.dirty = true; }
        removed
    }

    pub fn clear_buttons(&mut self) {
        self.buttons.clear();
        self.callbacks.clear();
        self.pressed_button_id = None;
        self.dirty = true;
    }

    // ── Mouse input ──

    /// Update mouse position from raw screen pixel coordinates.
    /// Call this from MouseMotion events.
    pub fn update_mouse_position(&mut self, pixel_x: f32, pixel_y: f32, screen_w: f32, screen_h: f32) {
        self.mouse_x = (pixel_x / screen_w).clamp(0.0, 1.0);
        self.mouse_y = (pixel_y / screen_h).clamp(0.0, 1.0);
    }

    /// Update mouse button state.
    /// Call from MouseButtonDown (pressed=true) and MouseButtonUp (pressed=false).
    pub fn set_mouse_down(&mut self, pressed: bool) {
        self.mouse_down = pressed;
    }

    /// Returns true if the mouse is currently over any button (useful for consuming clicks).
    pub fn mouse_over_button(&self) -> bool {
        let (mx, my) = (self.mouse_x, self.mouse_y);
        self.buttons.iter().any(|b| {
            b.state() != ButtonState::Disabled
                && b.element().visible
                && b.contains_point(mx, my)
        })
    }

    // ── Per-frame update ──

    /// Process button state transitions based on current mouse state.
    /// Returns true if any button state changed (caller should rebuild vertices).
    /// Call once per frame after processing all input events.
    pub fn update_buttons(&mut self) -> bool {
        if !self.visible { return false; }

        let mx = self.mouse_x;
        let my = self.mouse_y;
        let mouse_down = self.mouse_down;
        let mut changed = false;

        // ── Press initiation ──
        if mouse_down && self.pressed_button_id.is_none() {
            for button in &mut self.buttons {
                if button.state() != ButtonState::Disabled
                    && button.element().visible
                    && button.contains_point(mx, my)
                {
                    button.set_state(ButtonState::Pressed);
                    self.pressed_button_id = Some(button.id());
                    changed = true;
                    break;
                }
            }
        }
        // ── Release ──
        else if !mouse_down && self.pressed_button_id.is_some() {
            let pressed_id = self.pressed_button_id.take().unwrap();

            if let Some(button) = self.buttons.iter_mut().find(|b| b.id() == pressed_id) {
                let over = button.contains_point(mx, my);

                // Fire callback if released over the button
                if over {
                    if let Some((_, cb)) = self.callbacks.iter().find(|(id, _)| *id == pressed_id) {
                        cb();
                    }
                }

                // Transition to hover or normal
                let new_state = if button.state() == ButtonState::Disabled {
                    ButtonState::Disabled
                } else if over {
                    ButtonState::Hover
                } else {
                    ButtonState::Normal
                };
                button.set_state(new_state);
                changed = true;
            }
        }
        // ── Hover tracking (no button pressed) ──
        else if !mouse_down {
            for button in &mut self.buttons {
                if button.state() == ButtonState::Disabled || !button.element().visible {
                    continue;
                }
                let over = button.contains_point(mx, my);
                let old = button.state();
                let new = if over { ButtonState::Hover } else { ButtonState::Normal };
                if new != old {
                    button.set_state(new);
                    changed = true;
                }
            }
        }
        // ── Drag tracking (button pressed, mouse still held) ──
        else if mouse_down && self.pressed_button_id.is_some() {
            let pressed_id = self.pressed_button_id.unwrap();
            if let Some(button) = self.buttons.iter_mut().find(|b| b.id() == pressed_id) {
                let old = button.state();
                let new = if button.contains_point(mx, my) {
                    ButtonState::Pressed
                } else {
                    ButtonState::Normal
                };
                if new != old {
                    button.set_state(new);
                    changed = true;
                }
            }
        }

        if changed {
            self.dirty = true;
        }
        changed
    }

    // ── Vertex generation ──

    /// Collect all visible elements + buttons into flat UIElement list.
    fn collect_visible_elements(&self) -> Vec<UIElement> {
        let mut out = Vec::with_capacity(self.elements.len() + self.buttons.len());
        for elem in &self.elements {
            if elem.visible {
                out.push(*elem);
            }
        }
        for button in &self.buttons {
            if button.element().visible {
                out.push(*button.element());
            }
        }
        out
    }

    /// Build vertex data for all visible elements. Returns vertex count.
    /// Writes into `dst` which must have capacity for MAX_VERTICES.
    fn build_vertices(&self, dst: &mut Vec<UIVertex>) -> u32 {
        dst.clear();
        if !self.visible { return 0; }

        let elements = self.collect_visible_elements();
        let count = elements.len().min(MAX_UI_ELEMENTS);

        for elem in &elements[..count] {
            let verts = build_quad_vertices(elem.x, elem.y, elem.width, elem.height, elem.color);
            dst.extend_from_slice(&verts);
        }
        dst.len() as u32
    }
}

// ===== GPU Resources (UIOverlay) =====

/// Owns all Vulkan resources for UI rendering.
/// Created once, destroyed on Drop. Vertex data updated per-frame.
///
/// Integration into Renderer:
///   - Store as field: `ui_overlay: UIOverlay`
///   - Create in Renderer::new after crosshair setup
///   - Call `ui_overlay.update_and_render(...)` in render() after crosshair draw
///   - Drop handles cleanup automatically
pub struct UIOverlay {
    device: Arc<Device>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    vertex_buffer: vk::Buffer,
    vertex_memory: vk::DeviceMemory,
    vertex_mapped: *mut u8,  // persistently mapped
    vertex_count: u32,
    // Scratch buffer for building vertices (avoids per-frame allocation)
    vertex_scratch: Vec<UIVertex>,
}

// SAFETY: vertex_mapped is a persistently-mapped GPU pointer used only on the
// main thread (same thread that created the device). Not accessed from other threads.
unsafe impl Send for UIOverlay {}

impl UIOverlay {
    /// Buffer size in bytes for the vertex buffer.
    const BUFFER_SIZE: u64 = (MAX_VERTICES * std::mem::size_of::<UIVertex>()) as u64;

    /// Create the UI overlay pipeline and vertex buffer.
    /// Call after render_pass is created. Reuses the same render_pass as the crosshair.
    ///
    /// # Renderer::new integration point (after crosshair creation, ~line 214):
    /// ```ignore
    /// let ui_overlay = UIOverlay::new(&device, &device_ctx, render_pass)?;
    /// println!("  ✓ UI overlay created");
    /// ```
    pub fn new(
        device: &Arc<Device>,
        device_ctx: &DeviceContext,
        render_pass: vk::RenderPass,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let device = Arc::clone(device);

        let (pipeline, pipeline_layout) = Self::create_pipeline(&device, render_pass)?;
        let (vertex_buffer, vertex_memory, vertex_mapped) =
            Self::create_vertex_buffer(&device, device_ctx)?;

        Ok(Self {
            device,
            pipeline,
            pipeline_layout,
            vertex_buffer,
            vertex_memory,
            vertex_mapped,
            vertex_count: 0,
            vertex_scratch: Vec::with_capacity(MAX_VERTICES),
        })
    }

    /// Rebuild vertex buffer from UIManager state (only if dirty), then record draw commands.
    ///
    /// # Renderer::render integration point (after crosshair draw, ~line 642):
    /// ```ignore
    /// // === Pass 4: UI overlay ===
    /// self.ui_overlay.update_and_render(cmd, &mut self.ui_manager);
    /// ```
    pub fn update_and_render(
        &mut self,
        cmd: vk::CommandBuffer,
        manager: &mut UIManager,
    ) {
        if !manager.visible { return; }

        // Rebuild vertex data when dirty
        if manager.dirty {
            self.vertex_count = manager.build_vertices(&mut self.vertex_scratch);
            manager.dirty = false;

            if self.vertex_count > 0 {
                let byte_count = self.vertex_count as usize * std::mem::size_of::<UIVertex>();
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.vertex_scratch.as_ptr() as *const u8,
                        self.vertex_mapped,
                        byte_count,
                    );
                }
            }
        }

        if self.vertex_count == 0 { return; }

        unsafe {
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            self.device.cmd_bind_vertex_buffers(cmd, 0, &[self.vertex_buffer], &[0]);
            self.device.cmd_draw(cmd, self.vertex_count, 1, 0, 0);
        }
    }

    /// Record draw commands only (vertex data must already be uploaded).
    /// Use when you need finer control over update vs render timing.
    pub fn render_only(&self, cmd: vk::CommandBuffer) {
        if self.vertex_count == 0 { return; }
        unsafe {
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            self.device.cmd_bind_vertex_buffers(cmd, 0, &[self.vertex_buffer], &[0]);
            self.device.cmd_draw(cmd, self.vertex_count, 1, 0, 0);
        }
    }

    // ── Pipeline creation ──
    // Mirrors create_crosshair_pipeline() in renderer.rs lines 877-940

    fn create_pipeline(
        device: &Device,
        render_pass: vk::RenderPass,
    ) -> Result<(vk::Pipeline, vk::PipelineLayout), Box<dyn std::error::Error>> {
        unsafe {
            let vert_code = align_spv(include_bytes!("../shaders/compiled/ui.vert.spv"));
            let frag_code = align_spv(include_bytes!("../shaders/compiled/ui.frag.spv"));

            let vert_mod = device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&vert_code), None)?;
            let frag_mod = device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&frag_code), None)?;

            let stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX).module(vert_mod).name(c"main"),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT).module(frag_mod).name(c"main"),
            ];

            // Vertex input: pos (vec2) + color (vec4)
            let binding = vk::VertexInputBindingDescription::default()
                .binding(0)
                .stride(UIVertex::STRIDE)
                .input_rate(vk::VertexInputRate::VERTEX);

            let attrs = [
                // location 0: position (R32G32_SFLOAT, offset 0)
                vk::VertexInputAttributeDescription::default()
                    .binding(0).location(0)
                    .format(vk::Format::R32G32_SFLOAT)
                    .offset(0),
                // location 1: color (R32G32B32A32_SFLOAT, offset 8)
                vk::VertexInputAttributeDescription::default()
                    .binding(0).location(1)
                    .format(vk::Format::R32G32B32A32_SFLOAT)
                    .offset(8),
            ];

            let vi = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(std::slice::from_ref(&binding))
                .vertex_attribute_descriptions(&attrs);

            let ia = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

            let vp = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1);

            let rs = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0)
                .cull_mode(vk::CullModeFlags::NONE);

            let ms = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);

            // No depth test/write — overlay on top of everything
            let ds = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(false)
                .depth_write_enable(false);

            // Standard alpha blending: srcAlpha, 1-srcAlpha
            let cba = vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                .alpha_blend_op(vk::BlendOp::ADD);

            let cb = vk::PipelineColorBlendStateCreateInfo::default()
                .attachments(std::slice::from_ref(&cba));

            let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dyn_state = vk::PipelineDynamicStateCreateInfo::default()
                .dynamic_states(&dyn_states);

            // Empty layout — no push constants, no descriptor sets
            let layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default(), None)?;

            let pi = vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&vi)
                .input_assembly_state(&ia)
                .viewport_state(&vp)
                .rasterization_state(&rs)
                .multisample_state(&ms)
                .depth_stencil_state(&ds)
                .color_blend_state(&cb)
                .dynamic_state(&dyn_state)
                .layout(layout)
                .render_pass(render_pass)
                .subpass(0);

            let pipelines = device.create_graphics_pipelines(
                vk::PipelineCache::null(), &[pi], None)
                .map_err(|(_, e)| e)?;

            device.destroy_shader_module(vert_mod, None);
            device.destroy_shader_module(frag_mod, None);

            Ok((pipelines[0], layout))
        }
    }

    // ── Vertex buffer creation ──
    // Mirrors create_crosshair_buffer() in renderer.rs lines 1016-1038
    // but HOST_VISIBLE + persistently mapped for dynamic updates.

    fn create_vertex_buffer(
        device: &Device,
        device_ctx: &DeviceContext,
    ) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8), Box<dyn std::error::Error>> {
        unsafe {
            let buf = device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(Self::BUFFER_SIZE)
                    .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;

            let req = device.get_buffer_memory_requirements(buf);
            let mem_type = find_mem_type(
                device_ctx.memory_properties,
                req.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;

            let mem = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mem_type),
                None,
            )?;

            device.bind_buffer_memory(buf, mem, 0)?;

            // Persistently map — stays valid until free_memory
            let mapped = device.map_memory(
                mem, 0, Self::BUFFER_SIZE, vk::MemoryMapFlags::empty(),
            )? as *mut u8;

            // Zero-fill for safety
            std::ptr::write_bytes(mapped, 0, Self::BUFFER_SIZE as usize);

            Ok((buf, mem, mapped))
        }
    }
}

/// Cleanup GPU resources. Matches Renderer::drop pattern (renderer.rs lines 1108-1167).
impl Drop for UIOverlay {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            // Unmap happens implicitly on free_memory, but explicit unmap is cleaner
            self.device.unmap_memory(self.vertex_memory);
            self.device.destroy_buffer(self.vertex_buffer, None);
            self.device.free_memory(self.vertex_memory, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

// ===== Helpers =====
// Duplicated from renderer.rs to keep ui.rs self-contained.
// If you refactor these into a shared module, remove these.

fn find_mem_type(
    props: vk::PhysicalDeviceMemoryProperties,
    filter: u32,
    flags: vk::MemoryPropertyFlags,
) -> Result<u32, Box<dyn std::error::Error>> {
    for i in 0..props.memory_type_count {
        if (filter & (1 << i)) != 0
            && props.memory_types[i as usize].property_flags.contains(flags)
        {
            return Ok(i);
        }
    }
    Err("No suitable memory type for UI vertex buffer".into())
}

fn align_spv(code: &[u8]) -> Vec<u32> {
    code.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}