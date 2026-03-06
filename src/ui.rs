use ash::{vk, Device};
use image::GenericImageView;
use std::sync::Arc;
use crate::device::DeviceContext;

// ===== Constants =====

/// Maximum UI elements per frame. 128 quads × 6 verts × 32 bytes = 24,576 bytes.
const MAX_UI_ELEMENTS: usize = 128;
const VERTS_PER_QUAD: usize = 6;
const MAX_VERTICES: usize = MAX_UI_ELEMENTS * VERTS_PER_QUAD;

// ===== Vertex Layout =====

/// Per-vertex data written to the dynamic vertex buffer.
/// Stride = 32 bytes. Matches pipeline vertex input:
///   location 0: R32G32_SFLOAT       (pos,   offset 0)
///   location 1: R32G32_SFLOAT       (uv,    offset 8)
///   location 2: R32G32B32A32_SFLOAT (color, offset 16)
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct UIVertex {
    pos: [f32; 2],   // clip-space position
    uv: [f32; 2],    // texture coordinates (0,0 → sample 1×1 white for color-only)
    color: [f32; 4], // RGBA
}

impl UIVertex {
    const STRIDE: u32 = std::mem::size_of::<Self>() as u32; // 32
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
/// `uv_rect`: if Some([u0, v0, u1, v1]), map quad corners to the UV sub-region.
///            if None, all UVs = (0.5, 0.5) — samples 1×1 white fallback.
fn build_quad_vertices(
    x: f32, y: f32, w: f32, h: f32,
    color: [f32; 4],
    uv_rect: Option<[f32; 4]>,
) -> [UIVertex; 6] {
    let tl = screen_to_clip(x, y);
    let tr = screen_to_clip(x + w, y);
    let bl = screen_to_clip(x, y + h);
    let br = screen_to_clip(x + w, y + h);

    let (uv_tl, uv_tr, uv_bl, uv_br) = match uv_rect {
        Some([u0, v0, u1, v1]) => (
            [u0, v0], // top-left
            [u1, v0], // top-right
            [u0, v1], // bottom-left
            [u1, v1], // bottom-right
        ),
        None => {
            let c = [0.5, 0.5];
            (c, c, c, c)
        }
    };

    [
        // Triangle 1: TL → TR → BR
        UIVertex { pos: tl, uv: uv_tl, color },
        UIVertex { pos: tr, uv: uv_tr, color },
        UIVertex { pos: br, uv: uv_br, color },
        // Triangle 2: TL → BR → BL
        UIVertex { pos: tl, uv: uv_tl, color },
        UIVertex { pos: br, uv: uv_br, color },
        UIVertex { pos: bl, uv: uv_bl, color },
    ]
}

// ===== UI Element =====

/// A single rectangle in screen space, optionally textured.
#[derive(Copy, Clone, Debug)]
pub struct UIElement {
    /// Position: (x, y) top-left corner in normalized screen coords [0..1]
    pub x: f32,
    pub y: f32,
    /// Size in normalized screen coords
    pub width: f32,
    pub height: f32,
    /// RGBA color / tint [0..1]
    pub color: [f32; 4],
    /// Whether this element is visible
    pub visible: bool,
    /// Texture to sample. None = solid color (uses 1×1 white fallback).
    pub texture_id: Option<u32>,
    /// UV sub-region [u0, v0, u1, v1]. Only meaningful when texture_id is Some.
    pub uv_rect: [f32; 4],
}

impl UIElement {
    /// Create a solid-color (untextured) element.
    pub fn new(x: f32, y: f32, width: f32, height: f32, color: [f32; 4]) -> Self {
        Self { x, y, width, height, color, visible: true, texture_id: None, uv_rect: [0.0; 4] }
    }

    /// Create a textured element with a UV sub-region and tint color.
    pub fn new_textured(
        x: f32, y: f32, width: f32, height: f32,
        color: [f32; 4], texture_id: u32, uv_rect: [f32; 4],
    ) -> Self {
        Self { x, y, width, height, color, visible: true, texture_id: Some(texture_id), uv_rect }
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
///
/// Two rendering layers:
///   - `elements` + `buttons`: menu layer, gated behind `visible`
///   - `hud_elements`: always-on HUD layer, rendered regardless of `visible`
pub struct UIManager {
    /// Static (non-interactive) menu elements rendered below buttons
    pub elements: Vec<UIElement>,
    /// Interactive buttons (menu layer)
    pub buttons: Vec<Button>,
    /// Button click callbacks, keyed by button id.
    /// Stored separately so Button doesn't need Box<dyn Fn()>.
    callbacks: Vec<(u32, Box<dyn Fn() + Send>)>,
    /// Always-visible HUD elements (rendered even when menu is hidden)
    pub hud_elements: Vec<UIElement>,
    /// Screen dimensions in pixels (for aspect-ratio-correct element sizing)
    screen_width: f32,
    screen_height: f32,
    /// Current mouse position in normalized screen coords (0-1)
    mouse_x: f32,
    mouse_y: f32,
    /// Whether left mouse button is currently held
    mouse_down: bool,
    /// Button ID that received the initial press (for drag tracking)
    pressed_button_id: Option<u32>,
    /// Set true whenever any element/button changes, cleared after vertex rebuild
    pub dirty: bool,
    /// Menu visibility toggle (does NOT affect hud_elements)
    pub visible: bool,
}

impl Default for UIManager {
    fn default() -> Self {
        Self {
            elements: Vec::new(),
            buttons: Vec::new(),
            callbacks: Vec::new(),
            hud_elements: Vec::new(),
            screen_width: 1280.0,
            screen_height: 720.0,
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

    // ── Screen size ──

    /// Store current screen dimensions for aspect-ratio calculations.
    /// Call on init and on window resize.
    pub fn set_screen_size(&mut self, w: f32, h: f32) {
        self.screen_width = w;
        self.screen_height = h;
    }

    // ── Element management (menu layer) ──

    pub fn add_element(&mut self, elem: UIElement) {
        self.elements.push(elem);
        self.dirty = true;
    }

    pub fn clear_elements(&mut self) {
        self.elements.clear();
        self.dirty = true;
    }

    // ── HUD element management (always-visible layer) ──

    /// Add an element to the always-visible HUD layer.
    /// Renders regardless of the menu `visible` toggle.
    pub fn add_hud_element(&mut self, elem: UIElement) {
        self.hud_elements.push(elem);
        self.dirty = true;
    }

    pub fn clear_hud_elements(&mut self) {
        self.hud_elements.clear();
        self.dirty = true;
    }

    // ── Textured element helpers ──
    //
    // "Textured" elements are colored quads whose dimensions are computed from
    // a loaded texture's aspect ratio. The current UI pipeline renders solid
    // color — the texture metadata is used only for sizing. When the pipeline
    // gains texture sampling support, these quads will display the actual image.

    /// Compute aspect-ratio-fitted dimensions within a bounding box.
    /// Returns (width, height) in normalized screen coords.
    fn fit_to_aspect(
        &self,
        overlay: &UIOverlay,
        max_w: f32,
        max_h: f32,
        texture_id: u32,
        uv_rect: [f32; 4],
    ) -> (f32, f32) {
        let (tex_w, tex_h) = overlay.get_texture_size(texture_id);
        if tex_w == 0 || tex_h == 0 {
            return (max_w, max_h);
        }

        // UV rect: [u0, v0, u1, v1] — sub-region of the texture
        let uv_w = (uv_rect[2] - uv_rect[0]).abs().max(0.001);
        let uv_h = (uv_rect[3] - uv_rect[1]).abs().max(0.001);

        // Pixel aspect ratio of the selected UV region
        let tex_aspect = (uv_w * tex_w as f32) / (uv_h * tex_h as f32);

        // Convert to normalized-screen-coord ratio accounting for screen aspect
        // A square pixel on a 16:9 screen needs w/h = 9/16 in normalized coords
        let screen_ratio = self.screen_height / self.screen_width.max(1.0);
        let desired_ratio = tex_aspect * screen_ratio;

        // Fit within bounding box
        if max_w / max_h.max(0.001) > desired_ratio {
            // Height-limited
            let h = max_h;
            let w = max_h * desired_ratio;
            (w, h)
        } else {
            // Width-limited
            let w = max_w;
            let h = max_w / desired_ratio.max(0.001);
            (w, h)
        }
    }

    /// Add an element to the **menu layer** whose dimensions are fitted to
    /// the texture's aspect ratio within the given bounding box.
    /// Positioned with top-left at (x, y).
    pub fn add_textured_element_fitted(
        &mut self,
        overlay: &UIOverlay,
        x: f32,
        y: f32,
        max_w: f32,
        max_h: f32,
        texture_id: u32,
        uv_rect: [f32; 4],
        tint: [f32; 4],
    ) {
        let (w, h) = self.fit_to_aspect(overlay, max_w, max_h, texture_id, uv_rect);
        self.add_element(UIElement::new_textured(x, y, w, h, tint, texture_id, uv_rect));
    }

    /// Add an element to the **menu layer** centered horizontally at center_x
    /// with its top edge at top_y. Dimensions fitted to texture aspect ratio.
    pub fn add_textured_element_centered(
        &mut self,
        overlay: &UIOverlay,
        center_x: f32,
        top_y: f32,
        max_w: f32,
        max_h: f32,
        texture_id: u32,
        uv_rect: [f32; 4],
        tint: [f32; 4],
    ) {
        let (w, h) = self.fit_to_aspect(overlay, max_w, max_h, texture_id, uv_rect);
        let x = center_x - w * 0.5;
        self.add_element(UIElement::new_textured(x, top_y, w, h, tint, texture_id, uv_rect));
    }

    /// Add a textured element to the **always-visible HUD layer**.
    /// Positioned with top-left at (x, y), fitted to texture aspect ratio.
    pub fn add_hud_textured_element_fitted(
        &mut self,
        overlay: &UIOverlay,
        x: f32,
        y: f32,
        max_w: f32,
        max_h: f32,
        texture_id: u32,
        uv_rect: [f32; 4],
        tint: [f32; 4],
    ) {
        let (w, h) = self.fit_to_aspect(overlay, max_w, max_h, texture_id, uv_rect);
        self.add_hud_element(UIElement::new_textured(x, y, w, h, tint, texture_id, uv_rect));
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

    /// Collect all visible menu-layer elements + buttons into flat UIElement list.
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

    /// Build vertex data for all renderable elements. Returns vertex count.
    /// HUD elements always emitted. Menu elements only when `self.visible`.
    /// Writes into `dst` which must have capacity for MAX_VERTICES.
    /// Populates `batches` with draw-call ranges grouped by texture_id.
    /// Draw order is preserved (painter's algorithm); batches split on texture transitions.
    fn build_vertices(&self, dst: &mut Vec<UIVertex>, batches: &mut Vec<DrawBatch>) -> u32 {
        dst.clear();
        batches.clear();

        // Collect all elements to render into a single flat list (order = draw order)
        let mut all: Vec<UIElement> = Vec::with_capacity(
            self.hud_elements.len() + self.elements.len() + self.buttons.len(),
        );

        // Always emit HUD elements (persistent layer)
        for elem in &self.hud_elements {
            if elem.visible {
                all.push(*elem);
            }
        }

        // Menu layer — only when visible
        if self.visible {
            let menu_elems = self.collect_visible_elements();
            let remaining = MAX_UI_ELEMENTS.saturating_sub(all.len());
            let count = menu_elems.len().min(remaining);
            all.extend_from_slice(&menu_elems[..count]);
        }

        if all.is_empty() {
            return 0;
        }

        // Generate vertices in insertion order, splitting batches on texture transitions.
        // No sorting — preserves painter's algorithm draw order.
        let mut current_tex: Option<u32> = Some(u32::MAX); // sentinel
        let mut batch_start: u32 = 0;

        for elem in &all {
            let tex = elem.texture_id;
            if tex != current_tex {
                // Close previous batch
                let vert_now = dst.len() as u32;
                if vert_now > batch_start {
                    batches.push(DrawBatch {
                        texture_id: current_tex.filter(|&id| id != u32::MAX),
                        first_vertex: batch_start,
                        vertex_count: vert_now - batch_start,
                    });
                }
                current_tex = tex;
                batch_start = vert_now;
            }

            let uv = elem.texture_id.map(|_| elem.uv_rect);
            let verts = build_quad_vertices(
                elem.x, elem.y, elem.width, elem.height, elem.color, uv,
            );
            dst.extend_from_slice(&verts);
        }

        // Close final batch
        let vert_now = dst.len() as u32;
        if vert_now > batch_start {
            batches.push(DrawBatch {
                texture_id: current_tex.filter(|&id| id != u32::MAX),
                first_vertex: batch_start,
                vertex_count: vert_now - batch_start,
            });
        }

        dst.len() as u32
    }
}

// ===== Draw Batching =====

/// A contiguous range of vertices sharing the same texture.
/// Used to minimize descriptor set rebinds during rendering.
#[derive(Clone, Debug)]
struct DrawBatch {
    /// Which texture to bind. None = 1×1 white fallback (solid color elements).
    texture_id: Option<u32>,
    /// First vertex index in the shared vertex buffer.
    first_vertex: u32,
    /// Number of vertices in this batch.
    vertex_count: u32,
}

// ===== Texture Metadata =====

/// Dimensions of a loaded texture. Used by UIManager for aspect-ratio fitting.
/// The actual pixel data is not stored here — only the dimensions extracted
/// from the image header at load time.
struct TextureInfo {
    width: u32,
    height: u32,
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
/// GPU resources for a single loaded UI texture.
struct LoadedTexture {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    descriptor_set: vk::DescriptorSet,
}

/// Maximum number of distinct textures that can be loaded via load_texture().
const MAX_UI_TEXTURES: u32 = 16;

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
    // Loaded texture metadata (dimensions for aspect-ratio fitting)
    texture_infos: Vec<TextureInfo>,
    // Descriptor resources for the UI texture sampler (binding 0)
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    // Descriptor set for the 1×1 white fallback (solid-color elements)
    white_descriptor_set: vk::DescriptorSet,
    // 1×1 white fallback texture — color-only elements sample this so
    // texture(tex, uv) * vertexColor == vertexColor
    white_image: vk::Image,
    white_image_memory: vk::DeviceMemory,
    white_image_view: vk::ImageView,
    white_sampler: vk::Sampler,
    // Per-texture GPU resources (indexed by texture_id from load_texture)
    loaded_textures: Vec<LoadedTexture>,
    // Draw batches from last build_vertices (texture-grouped draw ranges)
    draw_batches: Vec<DrawBatch>,
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

        // ── 1×1 white fallback texture ──
        let (white_image, white_image_memory, white_image_view, white_sampler) =
            Self::create_white_texture(&device, device_ctx)?;

        // ── Descriptor set layout: binding 0 = combined image sampler (fragment) ──
        let descriptor_set_layout = unsafe {
            let binding = vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT);
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(std::slice::from_ref(&binding)),
                None,
            )?
        };

        // ── Descriptor pool + white fallback set ──
        let (descriptor_pool, white_descriptor_set) = unsafe {
            let pool_size = vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1 + MAX_UI_TEXTURES); // 1 white + N loaded textures
            let pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(std::slice::from_ref(&pool_size))
                    .max_sets(1 + MAX_UI_TEXTURES),
                None,
            )?;
            let sets = device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(std::slice::from_ref(&descriptor_set_layout)),
            )?;
            let set = sets[0];

            // Write: point binding 0 at the white texture
            let img_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(white_image_view)
                .sampler(white_sampler);
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&img_info))],
                &[],
            );

            (pool, set)
        };

        let (pipeline, pipeline_layout) =
            Self::create_pipeline(&device, render_pass, descriptor_set_layout)?;
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
            texture_infos: Vec::new(),
            descriptor_set_layout,
            descriptor_pool,
            white_descriptor_set,
            white_image,
            white_image_memory,
            white_image_view,
            white_sampler,
            loaded_textures: Vec::new(),
            draw_batches: Vec::new(),
        })
    }

    // ── Texture loading ──

    /// Load an image file, decode to RGBA, upload to GPU, and create a descriptor set.
    /// Returns a texture id used to reference this texture in UIManager methods.
    pub fn load_texture(
        &mut self,
        device_ctx: &DeviceContext,
        path: &str,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        if self.loaded_textures.len() >= MAX_UI_TEXTURES as usize {
            return Err(format!(
                "Maximum UI textures ({}) exceeded loading: {}", MAX_UI_TEXTURES, path
            ).into());
        }

        // Decode image to RGBA8
        let img = image::open(path)
            .map_err(|e| format!("Failed to load image '{}': {}", path, e))?;
        let (width, height) = img.dimensions();
        let rgba = img.to_rgba8();
        let pixels = rgba.as_raw();
        let image_size = (width * height * 4) as u64;

        // Upload to GPU — mirrors create_white_texture / create_texture_atlas pattern
        let (gpu_image, gpu_memory, gpu_view) = unsafe {
            let image = self.device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_SRGB)
                    .extent(vk::Extent3D { width, height, depth: 1 })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )?;

            let mem_req = self.device.get_image_memory_requirements(image);
            let mem_type = find_mem_type(
                device_ctx.memory_properties,
                mem_req.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            let memory = self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(mem_req.size)
                    .memory_type_index(mem_type),
                None,
            )?;
            self.device.bind_image_memory(image, memory, 0)?;

            // Staging buffer
            let staging_buf = self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(image_size)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;
            let staging_req = self.device.get_buffer_memory_requirements(staging_buf);
            let staging_type = find_mem_type(
                device_ctx.memory_properties,
                staging_req.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let staging_mem = self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(staging_req.size)
                    .memory_type_index(staging_type),
                None,
            )?;
            self.device.bind_buffer_memory(staging_buf, staging_mem, 0)?;
            let mapped = self.device.map_memory(
                staging_mem, 0, image_size, vk::MemoryMapFlags::empty(),
            )?;
            std::ptr::copy_nonoverlapping(
                pixels.as_ptr(), mapped as *mut u8, image_size as usize,
            );
            self.device.unmap_memory(staging_mem);

            // One-shot command buffer for upload
            let cmd = self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(device_ctx.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];

            self.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            // UNDEFINED → TRANSFER_DST
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[], &[],
                &[vk::ImageMemoryBarrier::default()
                    .image(image)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0, level_count: 1,
                        base_array_layer: 0, layer_count: 1,
                    })],
            );

            // Copy buffer → image
            self.device.cmd_copy_buffer_to_image(
                cmd, staging_buf, image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0, base_array_layer: 0, layer_count: 1,
                    })
                    .image_extent(vk::Extent3D { width, height, depth: 1 })],
            );

            // TRANSFER_DST → SHADER_READ_ONLY
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[], &[],
                &[vk::ImageMemoryBarrier::default()
                    .image(image)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0, level_count: 1,
                        base_array_layer: 0, layer_count: 1,
                    })],
            );

            self.device.end_command_buffer(cmd)?;

            let fence = self.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            self.device.queue_submit(
                device_ctx.queue,
                &[vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd))],
                fence,
            )?;
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;

            // Cleanup staging
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(device_ctx.command_pool, &[cmd]);
            self.device.destroy_buffer(staging_buf, None);
            self.device.free_memory(staging_mem, None);

            // Image view
            let view = self.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_SRGB)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0, level_count: 1,
                        base_array_layer: 0, layer_count: 1,
                    }),
                None,
            )?;

            (image, memory, view)
        };

        // Allocate a descriptor set for this texture
        let descriptor_set = unsafe {
            let sets = self.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptor_pool)
                    .set_layouts(std::slice::from_ref(&self.descriptor_set_layout)),
            )?;
            let set = sets[0];

            // Reuse the white_sampler (NEAREST, CLAMP) for loaded textures too
            let img_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(gpu_view)
                .sampler(self.white_sampler);
            self.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&img_info))],
                &[],
            );

            set
        };

        let id = self.texture_infos.len() as u32;
        self.texture_infos.push(TextureInfo { width, height });
        self.loaded_textures.push(LoadedTexture {
            image: gpu_image,
            memory: gpu_memory,
            view: gpu_view,
            descriptor_set,
        });

        println!("  ✓ UI texture loaded: {} ({}×{}, id={})", path, width, height, id);
        Ok(id)
    }

    /// Return (width, height) in pixels for a previously loaded texture.
    /// Returns (0, 0) if the id is invalid.
    pub fn get_texture_size(&self, id: u32) -> (u32, u32) {
        self.texture_infos
            .get(id as usize)
            .map(|t| (t.width, t.height))
            .unwrap_or((0, 0))
    }

    // ── Rendering ──

    /// Rebuild vertex buffer from UIManager state (only if dirty), then record draw commands.
    /// Always renders HUD elements; menu elements gated by manager.visible.
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
        // Rebuild vertex data when dirty
        if manager.dirty {
            self.vertex_count = manager.build_vertices(
                &mut self.vertex_scratch, &mut self.draw_batches,
            );
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

            for batch in &self.draw_batches {
                // Select the descriptor set for this batch's texture
                let desc_set = match batch.texture_id {
                    Some(id) => {
                        if let Some(tex) = self.loaded_textures.get(id as usize) {
                            tex.descriptor_set
                        } else {
                            self.white_descriptor_set
                        }
                    }
                    None => self.white_descriptor_set,
                };

                self.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    0,
                    &[desc_set],
                    &[],
                );
                self.device.cmd_draw(cmd, batch.vertex_count, 1, batch.first_vertex, 0);
            }
        }
    }

    /// Record draw commands only (vertex data must already be uploaded).
    /// Use when you need finer control over update vs render timing.
    pub fn render_only(&self, cmd: vk::CommandBuffer) {
        if self.vertex_count == 0 { return; }
        unsafe {
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            self.device.cmd_bind_vertex_buffers(cmd, 0, &[self.vertex_buffer], &[0]);

            for batch in &self.draw_batches {
                let desc_set = match batch.texture_id {
                    Some(id) => {
                        self.loaded_textures.get(id as usize)
                            .map(|t| t.descriptor_set)
                            .unwrap_or(self.white_descriptor_set)
                    }
                    None => self.white_descriptor_set,
                };

                self.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    0,
                    &[desc_set],
                    &[],
                );
                self.device.cmd_draw(cmd, batch.vertex_count, 1, batch.first_vertex, 0);
            }
        }
    }

    // ── Pipeline creation ──
    // Mirrors create_crosshair_pipeline() in renderer.rs lines 877-940

    fn create_pipeline(
        device: &Device,
        render_pass: vk::RenderPass,
        desc_layout: vk::DescriptorSetLayout,
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

            // Vertex input: pos (vec2) + uv (vec2) + color (vec4) = 32 bytes
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
                // location 1: uv (R32G32_SFLOAT, offset 8)
                vk::VertexInputAttributeDescription::default()
                    .binding(0).location(1)
                    .format(vk::Format::R32G32_SFLOAT)
                    .offset(8),
                // location 2: color (R32G32B32A32_SFLOAT, offset 16)
                vk::VertexInputAttributeDescription::default()
                    .binding(0).location(2)
                    .format(vk::Format::R32G32B32A32_SFLOAT)
                    .offset(16),
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

            // Layout with descriptor set 0 for the UI texture sampler
            let layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(std::slice::from_ref(&desc_layout)), None)?;

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

    // ── 1×1 white fallback texture ──
    // Mirrors create_texture_atlas() pattern in renderer.rs.
    // Creates a 1×1 RGBA white pixel so color-only UI quads produce:
    //   texture(tex, uv) * vertexColor == vertexColor

    fn create_white_texture(
        device: &Device,
        device_ctx: &DeviceContext,
    ) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView, vk::Sampler), Box<dyn std::error::Error>> {
        unsafe {
            let pixel: [u8; 4] = [255, 255, 255, 255];
            let image_size: u64 = 4;

            // Device-local image
            let image = device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .extent(vk::Extent3D { width: 1, height: 1, depth: 1 })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )?;

            let mem_req = device.get_image_memory_requirements(image);
            let mem_type = find_mem_type(
                device_ctx.memory_properties,
                mem_req.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            let memory = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(mem_req.size)
                    .memory_type_index(mem_type),
                None,
            )?;
            device.bind_image_memory(image, memory, 0)?;

            // Staging buffer
            let staging_buf = device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(image_size)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;
            let staging_req = device.get_buffer_memory_requirements(staging_buf);
            let staging_type = find_mem_type(
                device_ctx.memory_properties,
                staging_req.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let staging_mem = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(staging_req.size)
                    .memory_type_index(staging_type),
                None,
            )?;
            device.bind_buffer_memory(staging_buf, staging_mem, 0)?;
            let mapped = device.map_memory(staging_mem, 0, image_size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(pixel.as_ptr(), mapped as *mut u8, 4);
            device.unmap_memory(staging_mem);

            // Upload via one-shot command buffer
            let cmd = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(device_ctx.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];

            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            // UNDEFINED → TRANSFER_DST
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[], &[],
                &[vk::ImageMemoryBarrier::default()
                    .image(image)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0, level_count: 1,
                        base_array_layer: 0, layer_count: 1,
                    })],
            );

            // Copy buffer → image
            device.cmd_copy_buffer_to_image(
                cmd, staging_buf, image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0, base_array_layer: 0, layer_count: 1,
                    })
                    .image_extent(vk::Extent3D { width: 1, height: 1, depth: 1 })],
            );

            // TRANSFER_DST → SHADER_READ_ONLY
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[], &[],
                &[vk::ImageMemoryBarrier::default()
                    .image(image)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0, level_count: 1,
                        base_array_layer: 0, layer_count: 1,
                    })],
            );

            device.end_command_buffer(cmd)?;

            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            device.queue_submit(
                device_ctx.queue,
                &[vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd))],
                fence,
            )?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;

            // Cleanup staging
            device.destroy_fence(fence, None);
            device.free_command_buffers(device_ctx.command_pool, &[cmd]);
            device.destroy_buffer(staging_buf, None);
            device.free_memory(staging_mem, None);

            // Image view
            let view = device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0, level_count: 1,
                        base_array_layer: 0, layer_count: 1,
                    }),
                None,
            )?;

            // Sampler — NEAREST, clamp
            let sampler = device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_anisotropy(1.0)
                    .border_color(vk::BorderColor::INT_OPAQUE_WHITE),
                None,
            )?;

            Ok((image, memory, view, sampler))
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
            // Loaded texture GPU resources (descriptor sets freed implicitly by pool destroy)
            for tex in &self.loaded_textures {
                self.device.destroy_image_view(tex.view, None);
                self.device.destroy_image(tex.image, None);
                self.device.free_memory(tex.memory, None);
            }
            // Descriptor resources (pool destroy frees all allocated sets)
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            // 1×1 white fallback texture
            self.device.destroy_sampler(self.white_sampler, None);
            self.device.destroy_image_view(self.white_image_view, None);
            self.device.destroy_image(self.white_image, None);
            self.device.free_memory(self.white_image_memory, None);
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