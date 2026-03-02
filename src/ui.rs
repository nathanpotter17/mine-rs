use ash::{vk, Device};
use std::collections::HashMap;
use std::sync::Arc;
use crate::device::DeviceContext;

// ===== Constants =====

/// Maximum UI elements per frame. 128 quads × 6 verts × 32 bytes = 24,576 bytes.
const MAX_UI_ELEMENTS: usize = 128;
const VERTS_PER_QUAD: usize = 6;
const MAX_VERTICES: usize = MAX_UI_ELEMENTS * VERTS_PER_QUAD;

/// Maximum number of UI textures that can be loaded simultaneously.
/// Controls descriptor pool pre-allocation.
const MAX_UI_TEXTURES: u32 = 32;

/// Reserved texture ID for the built-in 1×1 white pixel.
/// All color-only elements sample this so the shader path is always:
///   outColor = texture(tex, uv) * vertexColor
const WHITE_TEXTURE_ID: u32 = 0;

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
    uv: [f32; 2],    // texture coordinates
    color: [f32; 4], // RGBA tint / solid color
}

impl UIVertex {
    const STRIDE: u32 = std::mem::size_of::<Self>() as u32; // 32
}

// ===== Coordinate Helpers =====

/// Convert normalized screen coords (0-1, top-left origin) to Vulkan clip space.
/// x: 0→-1, 1→+1
/// y: 0→-1 (top), 1→+1 (bottom)
#[inline]
fn screen_to_clip(sx: f32, sy: f32) -> [f32; 2] {
    [sx * 2.0 - 1.0, sy * 2.0 - 1.0]
}

/// Build 6 vertices (two triangles) for one axis-aligned quad.
/// `x, y, w, h` in normalized screen coords. Color applied to all vertices.
/// `uv_rect` = [u0, v0, u1, v1] defining the texture region.
fn build_quad_vertices(
    x: f32, y: f32, w: f32, h: f32,
    color: [f32; 4],
    uv_rect: [f32; 4],
) -> [UIVertex; 6] {
    let tl = screen_to_clip(x, y);
    let tr = screen_to_clip(x + w, y);
    let bl = screen_to_clip(x, y + h);
    let br = screen_to_clip(x + w, y + h);

    let [u0, v0, u1, v1] = uv_rect;

    [
        // Triangle 1: TL → TR → BR
        UIVertex { pos: tl, uv: [u0, v0], color },
        UIVertex { pos: tr, uv: [u1, v0], color },
        UIVertex { pos: br, uv: [u1, v1], color },
        // Triangle 2: TL → BR → BL
        UIVertex { pos: tl, uv: [u0, v0], color },
        UIVertex { pos: br, uv: [u1, v1], color },
        UIVertex { pos: bl, uv: [u0, v1], color },
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
    /// RGBA color [0..1]. For textured elements this acts as a tint/multiply.
    pub color: [f32; 4],
    /// Whether this element is visible
    pub visible: bool,
    /// Texture ID to sample from. `WHITE_TEXTURE_ID` (0) = solid color only.
    /// Obtain IDs from `UIOverlay::load_texture()`.
    pub texture_id: u32,
    /// UV rectangle [u0, v0, u1, v1] defining which region of the texture to sample.
    /// Defaults to [0, 0, 1, 1] (full texture).
    pub uv_rect: [f32; 4],
}

impl UIElement {
    /// Create a solid-color element (no texture).
    pub fn new(x: f32, y: f32, width: f32, height: f32, color: [f32; 4]) -> Self {
        Self {
            x, y, width, height, color, visible: true,
            texture_id: WHITE_TEXTURE_ID,
            uv_rect: [0.0, 0.0, 1.0, 1.0],
        }
    }

    /// Create a textured element.
    /// `color` acts as a tint/multiply — use `[1,1,1,1]` for no tint.
    pub fn new_textured(
        x: f32, y: f32, width: f32, height: f32,
        texture_id: u32,
        uv_rect: [f32; 4],
        color: [f32; 4],
    ) -> Self {
        Self {
            x, y, width, height, color, visible: true,
            texture_id,
            uv_rect,
        }
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

    /// Create a textured button. State colors act as tint multipliers.
    pub fn new_textured(
        id: u32,
        x: f32, y: f32, w: f32, h: f32,
        texture_id: u32,
        uv_rect: [f32; 4],
        color: [f32; 4],
    ) -> Self {
        let hover = [
            (color[0] * 1.15).min(1.0),
            (color[1] * 1.15).min(1.0),
            (color[2] * 1.15).min(1.0),
            color[3],
        ];
        let pressed = [
            color[0] * 0.80,
            color[1] * 0.80,
            color[2] * 0.80,
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
            element: UIElement::new_textured(x, y, w, h, texture_id, uv_rect, color),
            state: ButtonState::Normal,
            color_normal: color,
            color_hover: hover,
            color_pressed: pressed,
            color_disabled: disabled,
        };
        btn.apply_state_color();
        btn
    }

    /// Create a textured button that auto-fits to the image's aspect ratio.
    /// The button fits inside `(max_w, max_h)` without stretching, centered
    /// horizontally at `center_x`.
    ///
    /// # Example (main.rs)
    /// ```ignore
    /// let btn = Button::new_textured_fitted(
    ///     &renderer.ui_overlay, &renderer.ui_manager,
    ///     42,
    ///     0.50, 0.45,          // center_x, y
    ///     0.30, 0.08,          // max_w, max_h
    ///     hud_text_id,
    ///     [0.0, 0.75, 1.0, 1.0],
    ///     [1.0, 1.0, 1.0, 1.0],
    /// );
    /// renderer.ui_manager.add_button(btn);
    /// ```
    pub fn new_textured_fitted(
        overlay: &UIOverlay,
        manager: &UIManager,
        id: u32,
        center_x: f32, y: f32,
        max_w: f32, max_h: f32,
        texture_id: u32,
        uv_rect: [f32; 4],
        color: [f32; 4],
    ) -> Self {
        let (w, h) = manager.fit_to_aspect(overlay, texture_id, max_w, max_h, &uv_rect);
        let x = center_x - w / 2.0;
        Self::new_textured(id, x, y, w, h, texture_id, uv_rect, color)
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
    /// Current screen aspect ratio (width / height). Updated via set_screen_size()
    /// or automatically in update_mouse_position(). Used by fitted placement methods.
    screen_aspect: f32,
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
            screen_aspect: 16.0 / 9.0, // sensible default, overwritten on first resize/input
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

    /// Convenience: add a textured quad (e.g. pre-rendered text image).
    /// `color` = [1,1,1,1] for no tint, or use as alpha/tint multiplier.
    pub fn add_textured_element(
        &mut self,
        x: f32, y: f32, width: f32, height: f32,
        texture_id: u32,
        uv_rect: [f32; 4],
        color: [f32; 4],
    ) {
        self.elements.push(UIElement::new_textured(
            x, y, width, height, texture_id, uv_rect, color,
        ));
        self.dirty = true;
    }

    /// Add a textured quad that preserves the image's native aspect ratio.
    /// The quad is sized to fit inside `(max_w, max_h)` without stretching.
    /// Position `(x, y)` is the top-left corner.
    ///
    /// Requires `overlay` to query the texture's pixel dimensions.
    ///
    /// # Example (main.rs)
    /// ```ignore
    /// renderer.ui_manager.add_textured_element_fitted(
    ///     &renderer.ui_overlay,
    ///     0.35, 0.29,                  // top-left position
    ///     0.30, 0.08,                  // max bounding box
    ///     hud_text_id,
    ///     [0.0, 0.0, 1.0, 1.0],       // full texture
    ///     [1.0, 1.0, 1.0, 1.0],       // no tint
    /// );
    /// ```
    pub fn add_textured_element_fitted(
        &mut self,
        overlay: &UIOverlay,
        x: f32, y: f32,
        max_w: f32, max_h: f32,
        texture_id: u32,
        uv_rect: [f32; 4],
        color: [f32; 4],
    ) {
        let (w, h) = self.fit_to_aspect(overlay, texture_id, max_w, max_h, &uv_rect);
        self.elements.push(UIElement::new_textured(x, y, w, h, texture_id, uv_rect, color));
        self.dirty = true;
    }

    /// Add a textured quad that preserves aspect ratio AND centers horizontally.
    /// `center_x` = horizontal center position (0.5 = screen center).
    /// `y` = top edge.
    ///
    /// # Example (main.rs)
    /// ```ignore
    /// renderer.ui_manager.add_textured_element_centered(
    ///     &renderer.ui_overlay,
    ///     0.50, 0.29,                  // center_x, top y
    ///     0.30, 0.08,                  // max bounding box
    ///     hud_text_id,
    ///     [0.0, 0.0, 1.0, 1.0],
    ///     [1.0, 1.0, 1.0, 1.0],
    /// );
    /// ```
    pub fn add_textured_element_centered(
        &mut self,
        overlay: &UIOverlay,
        center_x: f32, y: f32,
        max_w: f32, max_h: f32,
        texture_id: u32,
        uv_rect: [f32; 4],
        color: [f32; 4],
    ) {
        let (w, h) = self.fit_to_aspect(overlay, texture_id, max_w, max_h, &uv_rect);
        let x = center_x - w / 2.0;
        self.elements.push(UIElement::new_textured(x, y, w, h, texture_id, uv_rect, color));
        self.dirty = true;
    }

    /// Compute aspect-ratio-preserving (width, height) in normalized screen coords.
    /// Fits inside (max_w, max_h) without stretching, accounting for screen aspect ratio.
    /// If uv_rect selects a sub-region, the sub-region's proportions are used.
    fn fit_to_aspect(
        &self,
        overlay: &UIOverlay,
        texture_id: u32,
        max_w: f32,
        max_h: f32,
        uv_rect: &[f32; 4],
    ) -> (f32, f32) {
        let (tex_w, tex_h) = overlay.texture_size(texture_id).unwrap_or((1, 1));

        // Account for UV sub-region: effective pixel dimensions
        let region_w = (uv_rect[2] - uv_rect[0]).abs() * tex_w as f32;
        let region_h = (uv_rect[3] - uv_rect[1]).abs() * tex_h as f32;

        // Image aspect in pixel space, corrected for non-square screen
        // In normalized coords, 0.1 horizontal ≠ 0.1 vertical unless screen is square.
        // Dividing by screen_aspect converts pixel-space ratio to normalized-coord ratio.
        let img_aspect = (region_w / region_h.max(1.0)) / self.screen_aspect;

        let box_aspect = max_w / max_h.max(0.001);
        if img_aspect > box_aspect {
            // Width-limited
            (max_w, max_w / img_aspect)
        } else {
            // Height-limited
            (max_h * img_aspect, max_h)
        }
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

    // ── Screen geometry ──

    /// Set screen dimensions. Call once at init and on resize.
    /// Also called implicitly by update_mouse_position().
    pub fn set_screen_size(&mut self, screen_w: f32, screen_h: f32) {
        if screen_h > 0.0 {
            self.screen_aspect = screen_w / screen_h;
        }
    }

    /// Current screen aspect ratio (width / height).
    #[inline]
    pub fn screen_aspect(&self) -> f32 {
        self.screen_aspect
    }

    // ── Mouse input ──

    /// Update mouse position from raw screen pixel coordinates.
    /// Call this from MouseMotion events.
    pub fn update_mouse_position(&mut self, pixel_x: f32, pixel_y: f32, screen_w: f32, screen_h: f32) {
        self.mouse_x = (pixel_x / screen_w).clamp(0.0, 1.0);
        self.mouse_y = (pixel_y / screen_h).clamp(0.0, 1.0);
        if screen_h > 0.0 {
            self.screen_aspect = screen_w / screen_h;
        }
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

    /// Build vertex data for all visible elements, batched by texture_id.
    /// Returns Vec of (texture_id, start_vertex, vertex_count) draw batches.
    /// Writes into `dst` which must have capacity for MAX_VERTICES.
    fn build_vertices(&self, dst: &mut Vec<UIVertex>) -> Vec<DrawBatch> {
        dst.clear();
        if !self.visible { return Vec::new(); }

        let elements = self.collect_visible_elements();
        let count = elements.len().min(MAX_UI_ELEMENTS);

        // Sort by texture_id to batch draws and minimize descriptor set swaps
        let mut sorted: Vec<&UIElement> = elements[..count].iter().collect();
        sorted.sort_by_key(|e| e.texture_id);

        let mut batches: Vec<DrawBatch> = Vec::new();
        let mut current_tex = u32::MAX;
        let mut batch_start = 0u32;

        for elem in &sorted {
            if elem.texture_id != current_tex {
                // Close previous batch
                let current_vert = dst.len() as u32;
                if current_vert > batch_start {
                    batches.push(DrawBatch {
                        texture_id: current_tex,
                        start_vertex: batch_start,
                        vertex_count: current_vert - batch_start,
                    });
                }
                current_tex = elem.texture_id;
                batch_start = current_vert;
            }
            let verts = build_quad_vertices(
                elem.x, elem.y, elem.width, elem.height,
                elem.color, elem.uv_rect,
            );
            dst.extend_from_slice(&verts);
        }

        // Close final batch
        let final_vert = dst.len() as u32;
        if final_vert > batch_start {
            batches.push(DrawBatch {
                texture_id: current_tex,
                start_vertex: batch_start,
                vertex_count: final_vert - batch_start,
            });
        }

        batches
    }
}

/// Describes one draw call within a frame: a range of vertices sharing one texture.
#[derive(Copy, Clone, Debug)]
struct DrawBatch {
    texture_id: u32,
    start_vertex: u32,
    vertex_count: u32,
}

// ===== GPU Texture Resource =====

/// All Vulkan resources for a single UI texture.
struct UITextureGPU {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    descriptor_set: vk::DescriptorSet,
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
pub struct UIOverlay {
    device: Arc<Device>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    vertex_buffer: vk::Buffer,
    vertex_memory: vk::DeviceMemory,
    vertex_mapped: *mut u8,  // persistently mapped

    // Scratch buffers (avoids per-frame allocation)
    vertex_scratch: Vec<UIVertex>,
    batch_scratch: Vec<DrawBatch>,

    // Texture infrastructure
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    sampler: vk::Sampler,
    textures: HashMap<u32, UITextureGPU>,
    next_texture_id: u32,

    // Cached for texture uploads
    memory_properties: vk::PhysicalDeviceMemoryProperties,
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
    /// # Renderer::new integration point (after crosshair creation, ~line 233):
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

        let descriptor_set_layout = Self::create_descriptor_set_layout(&device)?;
        let descriptor_pool = Self::create_descriptor_pool(&device)?;
        let sampler = Self::create_sampler(&device)?;

        let (pipeline, pipeline_layout) =
            Self::create_pipeline(&device, render_pass, descriptor_set_layout)?;
        let (vertex_buffer, vertex_memory, vertex_mapped) =
            Self::create_vertex_buffer(&device, device_ctx)?;

        let mut overlay = Self {
            device,
            pipeline,
            pipeline_layout,
            vertex_buffer,
            vertex_memory,
            vertex_mapped,
            vertex_scratch: Vec::with_capacity(MAX_VERTICES),
            batch_scratch: Vec::new(),
            descriptor_set_layout,
            descriptor_pool,
            sampler,
            textures: HashMap::new(),
            next_texture_id: WHITE_TEXTURE_ID + 1, // 0 is reserved for white
            memory_properties: device_ctx.memory_properties,
        };

        // Create the built-in 1×1 white pixel texture (ID 0).
        // Every color-only element samples this: texture(white) × color = color.
        overlay.create_white_texture(device_ctx)?;

        Ok(overlay)
    }

    // ===== Public Texture API =====

    /// Load an image file (BMP, PNG, JPG, TGA, etc.) and upload it as a UI texture.
    /// Returns a texture ID for use in `UIElement::new_textured()`.
    ///
    /// # Example
    /// ```ignore
    /// let tex_id = renderer.ui_overlay.load_texture(&device_ctx, "assets/hud_text.png")?;
    /// renderer.ui_manager.add_textured_element(
    ///     0.3, 0.02, 0.4, 0.06,   // x, y, w, h (normalized screen)
    ///     tex_id,
    ///     [0.0, 0.0, 1.0, 1.0],   // full texture
    ///     [1.0, 1.0, 1.0, 1.0],   // no tint
    /// );
    /// ```
    pub fn load_texture(
        &mut self,
        device_ctx: &DeviceContext,
        path: &str,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        let (width, height, rgba_pixels) = load_image_rgba8(path)?;
        let id = self.next_texture_id;
        self.next_texture_id += 1;
        self.upload_texture_rgba8(device_ctx, id, width, height, &rgba_pixels)?;
        println!("  ✓ UI texture loaded: {} ({}×{}, id={})", path, width, height, id);
        Ok(id)
    }

    /// Upload raw RGBA8 pixel data as a UI texture. Use when you generate pixels
    /// in code (e.g. CPU-rasterized text, procedural icons).
    /// Returns the assigned texture ID.
    pub fn load_texture_from_rgba8(
        &mut self,
        device_ctx: &DeviceContext,
        width: u32,
        height: u32,
        rgba_pixels: &[u8],
    ) -> Result<u32, Box<dyn std::error::Error>> {
        assert_eq!(rgba_pixels.len(), (width * height * 4) as usize,
            "Pixel data length must equal width × height × 4");
        let id = self.next_texture_id;
        self.next_texture_id += 1;
        self.upload_texture_rgba8(device_ctx, id, width, height, rgba_pixels)?;
        Ok(id)
    }

    /// Destroy a previously loaded texture and free its GPU resources.
    pub fn destroy_texture(&mut self, id: u32) {
        if id == WHITE_TEXTURE_ID { return; } // never destroy the white pixel
        if let Some(tex) = self.textures.remove(&id) {
            unsafe {
                let _ = self.device.device_wait_idle();
                self.device.destroy_image_view(tex.view, None);
                self.device.destroy_image(tex.image, None);
                self.device.free_memory(tex.memory, None);
                // descriptor_set freed when pool is destroyed/reset
            }
        }
    }

    /// Check whether a texture ID is currently loaded.
    pub fn has_texture(&self, id: u32) -> bool {
        self.textures.contains_key(&id)
    }

    /// Get the pixel dimensions of a loaded texture.
    pub fn texture_size(&self, id: u32) -> Option<(u32, u32)> {
        self.textures.get(&id).map(|t| (t.width, t.height))
    }

    // ===== Rendering =====

    /// Rebuild vertex buffer from UIManager state (only if dirty), then record draw commands.
    ///
    /// # Renderer::render integration point (after crosshair draw, ~line 864):
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
            self.batch_scratch = manager.build_vertices(&mut self.vertex_scratch);
            manager.dirty = false;

            if !self.vertex_scratch.is_empty() {
                let byte_count =
                    self.vertex_scratch.len() * std::mem::size_of::<UIVertex>();
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.vertex_scratch.as_ptr() as *const u8,
                        self.vertex_mapped,
                        byte_count,
                    );
                }
            }
        }

        if self.batch_scratch.is_empty() { return; }

        unsafe {
            self.device.cmd_bind_pipeline(
                cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline,
            );
            self.device.cmd_bind_vertex_buffers(cmd, 0, &[self.vertex_buffer], &[0]);

            for batch in &self.batch_scratch {
                // Bind the descriptor set for this batch's texture
                let tex_id = batch.texture_id;
                let ds = match self.textures.get(&tex_id) {
                    Some(tex) => tex.descriptor_set,
                    None => {
                        // Fallback to white pixel if texture was destroyed
                        match self.textures.get(&WHITE_TEXTURE_ID) {
                            Some(tex) => tex.descriptor_set,
                            None => continue, // should never happen
                        }
                    }
                };

                self.device.cmd_bind_descriptor_sets(
                    cmd, vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout, 0,
                    &[ds], &[],
                );

                self.device.cmd_draw(
                    cmd, batch.vertex_count, 1, batch.start_vertex, 0,
                );
            }
        }
    }

    /// Record draw commands only (vertex data must already be uploaded).
    /// Use when you need finer control over update vs render timing.
    pub fn render_only(&self, cmd: vk::CommandBuffer) {
        if self.batch_scratch.is_empty() { return; }

        unsafe {
            self.device.cmd_bind_pipeline(
                cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline,
            );
            self.device.cmd_bind_vertex_buffers(cmd, 0, &[self.vertex_buffer], &[0]);

            for batch in &self.batch_scratch {
                let tex_id = batch.texture_id;
                let ds = self.textures.get(&tex_id)
                    .or_else(|| self.textures.get(&WHITE_TEXTURE_ID))
                    .map(|t| t.descriptor_set);
                let Some(ds) = ds else { continue; };

                self.device.cmd_bind_descriptor_sets(
                    cmd, vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout, 0,
                    &[ds], &[],
                );

                self.device.cmd_draw(
                    cmd, batch.vertex_count, 1, batch.start_vertex, 0,
                );
            }
        }
    }

    // ===== Internal: texture upload =====

    /// Create the built-in 1×1 white pixel texture at ID 0.
    fn create_white_texture(
        &mut self,
        device_ctx: &DeviceContext,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let white_pixel: [u8; 4] = [255, 255, 255, 255];
        self.upload_texture_rgba8(device_ctx, WHITE_TEXTURE_ID, 1, 1, &white_pixel)
    }

    /// Core upload: creates DEVICE_LOCAL image, stages pixel data, transitions layout,
    /// creates image view, allocates descriptor set, writes descriptor.
    /// Mirrors Renderer::create_texture_atlas pattern.
    fn upload_texture_rgba8(
        &mut self,
        device_ctx: &DeviceContext,
        id: u32,
        width: u32,
        height: u32,
        pixels: &[u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let image_size = (width * height * 4) as u64;

        unsafe {
            // --- Create DEVICE_LOCAL image ---
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
                self.memory_properties,
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

            // --- Create staging buffer + copy pixel data ---
            let staging_buf = self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(image_size)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;

            let staging_req = self.device.get_buffer_memory_requirements(staging_buf);
            let staging_mem_type = find_mem_type(
                self.memory_properties,
                staging_req.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let staging_mem = self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(staging_req.size)
                    .memory_type_index(staging_mem_type),
                None,
            )?;
            self.device.bind_buffer_memory(staging_buf, staging_mem, 0)?;

            let mapped = self.device.map_memory(
                staging_mem, 0, image_size, vk::MemoryMapFlags::empty(),
            )?;
            std::ptr::copy_nonoverlapping(
                pixels.as_ptr(),
                mapped as *mut u8,
                image_size as usize,
            );
            self.device.unmap_memory(staging_mem);

            // --- One-shot command buffer: transitions + copy ---
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

            // Transition UNDEFINED → TRANSFER_DST_OPTIMAL
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

            // Transition TRANSFER_DST → SHADER_READ_ONLY_OPTIMAL
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

            // --- Image view ---
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

            // --- Allocate + write descriptor set ---
            let descriptor_set = self.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptor_pool)
                    .set_layouts(std::slice::from_ref(&self.descriptor_set_layout)),
            )?[0];

            let img_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(view)
                .sampler(self.sampler);

            self.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&img_info))],
                &[],
            );

            // Store
            self.textures.insert(id, UITextureGPU {
                image, memory, view, descriptor_set, width, height,
            });
        }

        Ok(())
    }

    // ===== Internal: Vulkan resource creation =====

    fn create_descriptor_set_layout(
        device: &Device,
    ) -> Result<vk::DescriptorSetLayout, Box<dyn std::error::Error>> {
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        unsafe {
            Ok(device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(std::slice::from_ref(&binding)),
                None,
            )?)
        }
    }

    fn create_descriptor_pool(
        device: &Device,
    ) -> Result<vk::DescriptorPool, Box<dyn std::error::Error>> {
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(MAX_UI_TEXTURES);

        unsafe {
            Ok(device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(std::slice::from_ref(&pool_size))
                    .max_sets(MAX_UI_TEXTURES)
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET),
                None,
            )?)
        }
    }

    /// Shared sampler for all UI textures. LINEAR filtering for smooth text rendering.
    fn create_sampler(device: &Device) -> Result<vk::Sampler, Box<dyn std::error::Error>> {
        unsafe {
            Ok(device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_anisotropy(1.0)
                    .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK),
                None,
            )?)
        }
    }

    // ── Pipeline creation ──

    fn create_pipeline(
        device: &Device,
        render_pass: vk::RenderPass,
        descriptor_set_layout: vk::DescriptorSetLayout,
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

            // Vertex input: pos (vec2) + uv (vec2) + color (vec4)
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
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .alpha_blend_op(vk::BlendOp::ADD);

            let cb = vk::PipelineColorBlendStateCreateInfo::default()
                .attachments(std::slice::from_ref(&cba));

            let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dyn_state = vk::PipelineDynamicStateCreateInfo::default()
                .dynamic_states(&dyn_states);

            // Layout with one descriptor set (binding 0 = sampler)
            let layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(std::slice::from_ref(&descriptor_set_layout)),
                None,
            )?;

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

/// Cleanup GPU resources.
impl Drop for UIOverlay {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();

            // Destroy all loaded textures (images, views, memory)
            for (_, tex) in self.textures.drain() {
                self.device.destroy_image_view(tex.view, None);
                self.device.destroy_image(tex.image, None);
                self.device.free_memory(tex.memory, None);
            }

            // Sampler + descriptors
            self.device.destroy_sampler(self.sampler, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            // Vertex buffer
            self.device.unmap_memory(self.vertex_memory);
            self.device.destroy_buffer(self.vertex_buffer, None);
            self.device.free_memory(self.vertex_memory, None);

            // Pipeline
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

// ===== Image File Loading =====

/// Load an image file and decode to RGBA8 pixels.
/// Supports BMP, PNG, JPG, TGA, and other formats via the `image` crate.
///
/// Returns (width, height, rgba_pixel_data).
pub fn load_image_rgba8(
    path: &str,
) -> Result<(u32, u32, Vec<u8>), Box<dyn std::error::Error>> {
    let img = image::open(path)
        .map_err(|e| format!("Failed to load image '{}': {}", path, e))?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    Ok((w, h, rgba.into_raw()))
}

// ===== Helpers =====
// Duplicated from renderer.rs to keep ui.rs self-contained.

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
    Err("No suitable memory type for UI resource".into())
}

fn align_spv(code: &[u8]) -> Vec<u32> {
    code.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}