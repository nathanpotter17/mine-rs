use ash::{vk, Device};
use std::collections::HashMap;
use std::sync::Arc;
use crate::device::DeviceContext;
use crate::world::{BlockVertex, ChunkMesh, ChunkPos};

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// Push constant for fog / view info
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PushConstants {
    pub fog_color: [f32; 4],
    pub fog_start: f32,
    pub fog_end: f32,
    pub time_of_day: f32,    // 0.0=midnight, 0.25=sunrise, 0.5=noon, 0.75=sunset
    pub sun_intensity: f32,  // 0.0=full night, 1.0=full day (smooth, derived from time)
}

// UBO: just view-projection
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ViewUBO {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4], // w unused
}

// ===== Frustum Culling =====

pub struct Frustum {
    planes: [[f32; 4]; 6],
}

impl Frustum {
    pub fn from_view_proj(v: &[[f32; 4]; 4], p: &[[f32; 4]; 4]) -> Self {
        let mut m = [[0.0f32; 4]; 4];
        for col in 0..4 {
            for row in 0..4 {
                m[col][row] = p[0][row]*v[col][0] + p[1][row]*v[col][1]
                            + p[2][row]*v[col][2] + p[3][row]*v[col][3];
            }
        }
        let row = |r: usize| -> [f32; 4] { [m[0][r], m[1][r], m[2][r], m[3][r]] };
        let r0 = row(0); let r1 = row(1); let r2 = row(2); let r3 = row(3);
        let mut planes = [[0.0f32; 4]; 6];
        for i in 0..4 {
            planes[0][i] = r3[i] + r0[i];
            planes[1][i] = r3[i] - r0[i];
            planes[2][i] = r3[i] + r1[i];
            planes[3][i] = r3[i] - r1[i];
            planes[4][i] = r3[i] + r2[i];
            planes[5][i] = r3[i] - r2[i];
        }
        for plane in &mut planes {
            let len = (plane[0]*plane[0] + plane[1]*plane[1] + plane[2]*plane[2]).sqrt();
            if len > 1e-8 {
                let inv = 1.0 / len;
                plane[0] *= inv; plane[1] *= inv; plane[2] *= inv; plane[3] *= inv;
            }
        }
        Self { planes }
    }

    #[inline]
    pub fn test_aabb(&self, min: [f32; 3], max: [f32; 3]) -> bool {
        for plane in &self.planes {
            let px = if plane[0] >= 0.0 { max[0] } else { min[0] };
            let py = if plane[1] >= 0.0 { max[1] } else { min[1] };
            let pz = if plane[2] >= 0.0 { max[2] } else { min[2] };
            if plane[0]*px + plane[1]*py + plane[2]*pz + plane[3] < 0.0 {
                return false;
            }
        }
        true
    }
}

struct ChunkGPUData {
    vertex_buffer: vk::Buffer,
    vertex_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_memory: vk::DeviceMemory,
    index_count: u32,
}

struct StagingCopy {
    staging_buf: vk::Buffer,
    staging_mem: vk::DeviceMemory,
    dst_buf: vk::Buffer,
    size: u64,
}

pub struct Renderer {
    device: Arc<Device>,

    // Per-chunk GPU data
    chunk_data: HashMap<ChunkPos, ChunkGPUData>,

    // Pipeline
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,

    // Uniform
    uniform_buffer: vk::Buffer,
    uniform_memory: vk::DeviceMemory,
    uniform_mapped: *mut std::ffi::c_void,

    // Texture atlas
    atlas_image: vk::Image,
    atlas_memory: vk::DeviceMemory,
    atlas_view: vk::ImageView,
    atlas_sampler: vk::Sampler,

    // Render pass & framebuffers
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,

    // Commands
    command_buffers: Vec<vk::CommandBuffer>,

    // Sync
    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,

    current_frame: usize,

    // Crosshair overlay
    crosshair_pipeline: vk::Pipeline,
    crosshair_pipeline_layout: vk::PipelineLayout,
    crosshair_vertex_buffer: vk::Buffer,
    crosshair_vertex_memory: vk::DeviceMemory,

    // Sky pass
    sky_pipeline: vk::Pipeline,
    sky_pipeline_layout: vk::PipelineLayout,

    // Cached
    memory_properties: vk::PhysicalDeviceMemoryProperties,

    // Deferred deletion
    deletion_queue: Vec<(u64, vk::Buffer, vk::DeviceMemory)>,
    frame_counter: u64,

    // Staging upload batch
    pending_copies: Vec<StagingCopy>,
    upload_fence: vk::Fence,
}

const MAX_FRAMES: usize = 2;

// Atlas constants
const ATLAS_SIZE: u32 = 256;    // 256×256 pixels
const TILE_SIZE: u32 = 16;      // 16×16 per tile
const TILES_PER_ROW: u32 = ATLAS_SIZE / TILE_SIZE; // = 16

impl Renderer {
    pub fn new(device_ctx: &DeviceContext) -> Result<Self, Box<dyn std::error::Error>> {
        let device = Arc::new(device_ctx.device.clone());

        // Create texture atlas
        let (atlas_image, atlas_memory, atlas_view, atlas_sampler) =
            Self::create_texture_atlas(&device, device_ctx)?;
        println!("  ✓ Texture atlas generated ({}×{}, {} tiles)", ATLAS_SIZE, ATLAS_SIZE, TILES_PER_ROW * TILES_PER_ROW);

        let descriptor_set_layout = Self::create_descriptor_set_layout(&device)?;
        let render_pass = Self::create_render_pass(&device, device_ctx.surface_format.format)?;
        let (pipeline, pipeline_layout) = Self::create_pipeline(&device, render_pass, descriptor_set_layout)?;
        let framebuffers = Self::create_framebuffers(
            &device, render_pass,
            &device_ctx.swapchain_image_views,
            device_ctx.depth_image_view,
            device_ctx.swapchain_extent,
        )?;

        let (uniform_buffer, uniform_memory, uniform_mapped) =
            Self::create_uniform_buffer(&device, device_ctx)?;

        let (descriptor_pool, descriptor_sets) =
            Self::create_descriptor_sets(&device, descriptor_set_layout, uniform_buffer,
                                         atlas_view, atlas_sampler)?;

        let command_buffers = Self::allocate_command_buffers(&device, device_ctx.command_pool)?;

        let mut image_available = Vec::new();
        let mut render_finished = Vec::new();
        let mut in_flight_fences = Vec::new();
        for _ in 0..MAX_FRAMES {
            unsafe {
                image_available.push(device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?);
                render_finished.push(device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?);
                in_flight_fences.push(device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None)?);
            }
        }

        // Crosshair
        let (crosshair_pipeline, crosshair_pipeline_layout) =
            Self::create_crosshair_pipeline(&device, render_pass)?;
        let (crosshair_vertex_buffer, crosshair_vertex_memory) =
            Self::create_crosshair_buffer(&device, device_ctx)?;

        let (sky_pipeline, sky_pipeline_layout) =
            Self::create_sky_pipeline(&device, render_pass, descriptor_set_layout)?;
        println!("  ✓ Sky pipeline created");

        let upload_fence = unsafe {
            device.create_fence(&vk::FenceCreateInfo::default(), None)?
        };

        Ok(Self {
            device,
            chunk_data: HashMap::new(),
            pipeline, pipeline_layout,
            descriptor_set_layout, descriptor_pool, descriptor_sets,
            uniform_buffer, uniform_memory, uniform_mapped,
            atlas_image, atlas_memory, atlas_view, atlas_sampler,
            render_pass, framebuffers,
            command_buffers,
            image_available, render_finished, in_flight_fences,
            current_frame: 0,
            crosshair_pipeline, crosshair_pipeline_layout,
            crosshair_vertex_buffer, crosshair_vertex_memory,
            sky_pipeline, sky_pipeline_layout,
            memory_properties: device_ctx.memory_properties,
            deletion_queue: Vec::new(),
            frame_counter: 0,
            pending_copies: Vec::new(),
            upload_fence,
        })
    }

    // ===== Texture Atlas Generation =====

    fn create_texture_atlas(
        device: &Device, device_ctx: &DeviceContext,
    ) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView, vk::Sampler), Box<dyn std::error::Error>> {
        // Generate procedural pixel data
        let pixels = generate_atlas_pixels();
        let image_size = (ATLAS_SIZE * ATLAS_SIZE * 4) as u64;

        unsafe {
            // Create DEVICE_LOCAL image
            let image = device.create_image(&vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_SRGB)
                .extent(vk::Extent3D { width: ATLAS_SIZE, height: ATLAS_SIZE, depth: 1 })
                .mip_levels(1).array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED), None)?;

            let mem_req = device.get_image_memory_requirements(image);
            let mem_type = find_mem_type(device_ctx.memory_properties, mem_req.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            let memory = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(mem_req.size)
                    .memory_type_index(mem_type), None)?;
            device.bind_image_memory(image, memory, 0)?;

            // Create staging buffer
            let (staging_buf, staging_mem) = Self::create_buffer(device, device_ctx, image_size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)?;
            let mapped = device.map_memory(staging_mem, 0, image_size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(pixels.as_ptr(), mapped as *mut u8, image_size as usize);
            device.unmap_memory(staging_mem);

            // Upload via one-shot command buffer
            let cmd = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(device_ctx.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1))?[0];

            device.begin_command_buffer(cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;

            // Transition UNDEFINED → TRANSFER_DST_OPTIMAL
            device.cmd_pipeline_barrier(cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(), &[], &[],
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
                    })]);

            // Copy buffer → image
            device.cmd_copy_buffer_to_image(cmd, staging_buf, image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0, base_array_layer: 0, layer_count: 1,
                    })
                    .image_extent(vk::Extent3D { width: ATLAS_SIZE, height: ATLAS_SIZE, depth: 1 })]);

            // Transition TRANSFER_DST → SHADER_READ_ONLY
            device.cmd_pipeline_barrier(cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(), &[], &[],
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
                    })]);

            device.end_command_buffer(cmd)?;

            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            device.queue_submit(device_ctx.queue,
                &[vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd))],
                fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;

            // Cleanup staging
            device.destroy_fence(fence, None);
            device.free_command_buffers(device_ctx.command_pool, &[cmd]);
            device.destroy_buffer(staging_buf, None);
            device.free_memory(staging_mem, None);

            // Image view
            let view = device.create_image_view(&vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_SRGB)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0, level_count: 1,
                    base_array_layer: 0, layer_count: 1,
                }), None)?;

            // Sampler — NEAREST for that crisp pixel-art voxel look
            let sampler = device.create_sampler(&vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::NEAREST)
                .min_filter(vk::Filter::NEAREST)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .max_anisotropy(1.0)
                .border_color(vk::BorderColor::INT_OPAQUE_BLACK), None)?;

            Ok((image, memory, view, sampler))
        }
    }

    pub fn upload_chunk_mesh(
        &mut self,
        device_ctx: &DeviceContext,
        pos: ChunkPos,
        mesh: &ChunkMesh,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.remove_chunk(pos);

        if mesh.vertices.is_empty() || mesh.indices.is_empty() {
            return Ok(());
        }

        let vert_size = (std::mem::size_of::<BlockVertex>() * mesh.vertices.len()) as u64;
        let idx_size = (std::mem::size_of::<u32>() * mesh.indices.len()) as u64;

        let (vb, vm) = self.stage_buffer_upload(
            device_ctx, &mesh.vertices, vert_size, vk::BufferUsageFlags::VERTEX_BUFFER)?;
        let (ib, im) = self.stage_buffer_upload(
            device_ctx, &mesh.indices, idx_size, vk::BufferUsageFlags::INDEX_BUFFER)?;

        self.chunk_data.insert(pos, ChunkGPUData {
            vertex_buffer: vb,
            vertex_memory: vm,
            index_buffer: ib,
            index_memory: im,
            index_count: mesh.indices.len() as u32,
        });

        Ok(())
    }

    fn stage_buffer_upload<T: Copy>(
        &mut self,
        device_ctx: &DeviceContext,
        data: &[T], size: u64, usage: vk::BufferUsageFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), Box<dyn std::error::Error>> {
        unsafe {
            let (staging_buf, staging_mem) = Self::create_buffer(&self.device, device_ctx, size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)?;
            let mapped = self.device.map_memory(staging_mem, 0, size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, mapped as *mut u8, size as usize);
            self.device.unmap_memory(staging_mem);

            let (buf, mem) = Self::create_buffer(&self.device, device_ctx, size,
                usage | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

            self.pending_copies.push(StagingCopy {
                staging_buf, staging_mem, dst_buf: buf, size,
            });

            Ok((buf, mem))
        }
    }

    pub fn flush_uploads(&mut self, device_ctx: &DeviceContext) -> Result<(), Box<dyn std::error::Error>> {
        if self.pending_copies.is_empty() { return Ok(()); }

        unsafe {
            let cmd_buf = self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(device_ctx.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1))?[0];

            self.device.begin_command_buffer(cmd_buf,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;

            for copy in &self.pending_copies {
                self.device.cmd_copy_buffer(cmd_buf, copy.staging_buf, copy.dst_buf,
                    &[vk::BufferCopy::default().size(copy.size)]);
            }

            self.device.end_command_buffer(cmd_buf)?;

            self.device.reset_fences(&[self.upload_fence])?;
            let submit = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&cmd_buf));
            self.device.queue_submit(device_ctx.queue, &[submit], self.upload_fence)?;
            self.device.wait_for_fences(&[self.upload_fence], true, u64::MAX)?;

            for copy in self.pending_copies.drain(..) {
                self.device.destroy_buffer(copy.staging_buf, None);
                self.device.free_memory(copy.staging_mem, None);
            }

            self.device.free_command_buffers(device_ctx.command_pool, &[cmd_buf]);
        }
        Ok(())
    }

    pub fn remove_chunk(&mut self, pos: ChunkPos) {
        if let Some(data) = self.chunk_data.remove(&pos) {
            let frame = self.frame_counter;
            self.deletion_queue.push((frame, data.vertex_buffer, data.vertex_memory));
            self.deletion_queue.push((frame, data.index_buffer, data.index_memory));
        }
    }

    fn flush_deletion_queue(&mut self) {
        let safe_frame = self.frame_counter.saturating_sub(MAX_FRAMES as u64);
        self.deletion_queue.retain(|&(queued_frame, buffer, memory)| {
            if queued_frame <= safe_frame {
                unsafe {
                    self.device.destroy_buffer(buffer, None);
                    self.device.free_memory(memory, None);
                }
                false
            } else {
                true
            }
        });
    }

    pub fn render(
        &mut self,
        device_ctx: &DeviceContext,
        view_ubo: ViewUBO,
        frustum: &Frustum,
        time_of_day: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            self.device.wait_for_fences(
                &[self.in_flight_fences[self.current_frame]], true, u64::MAX)?;

            self.flush_deletion_queue();

            let (image_index, _) = match device_ctx.swapchain_loader.acquire_next_image(
                device_ctx.swapchain, u64::MAX,
                self.image_available[self.current_frame], vk::Fence::null(),
            ) {
                Ok(result) => result,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(()),
                Err(e) => return Err(e.into()),
            };

            self.device.reset_fences(&[self.in_flight_fences[self.current_frame]])?;

            // Update UBO
            std::ptr::copy_nonoverlapping(
                &view_ubo as *const ViewUBO,
                self.uniform_mapped as *mut ViewUBO,
                1,
            );

            let cmd = self.command_buffers[self.current_frame];
            self.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;

            // === Compute time-of-day lighting parameters ===
            let sun_angle = time_of_day * std::f32::consts::TAU;
            let sun_elevation = sun_angle.sin(); // -1..1

            // sun_intensity: smooth ramp — matches sky.frag's dayFactor
            let sun_intensity = smoothstep(-0.1, 0.3, sun_elevation);

            // === Fog color — MUST match sky.frag horizon calculation exactly ===
            // These constants are duplicated from sky.frag. If you change one, change both.
            let day_horizon:    [f32; 3] = [0.65, 0.78, 0.95];
            let night_horizon:  [f32; 3] = [0.03, 0.04, 0.08];
            let sunset_color:   [f32; 3] = [0.95, 0.45, 0.15];

            // Sunset band factor — same formula as sky.frag
            let sunset_band = smoothstep(-0.05, 0.15, sun_elevation)
                            * (1.0 - smoothstep(0.15, 0.45, sun_elevation));

            // Horizon color = mix(night, day, intensity) then blend sunset
            let mut fog = [0.0f32; 4];
            for i in 0..3 {
                let base = lerp(night_horizon[i], day_horizon[i], sun_intensity);
                fog[i] = lerp(base, sunset_color[i], sunset_band * 0.7);
            }
            fog[3] = 1.0;

            // Fog distances: shorter at night for atmosphere
            let fog_start = lerp(35.0, 60.0, sun_intensity);
            let fog_end   = lerp(55.0, 90.0, sun_intensity);

            let clear_values = [
                vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } },
                vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } },
            ];

            let rp_info = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.framebuffers[image_index as usize])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: device_ctx.swapchain_extent,
                })
                .clear_values(&clear_values);

            self.device.cmd_begin_render_pass(cmd, &rp_info, vk::SubpassContents::INLINE);

            let viewport = vk::Viewport {
                x: 0.0, y: 0.0,
                width: device_ctx.swapchain_extent.width as f32,
                height: device_ctx.swapchain_extent.height as f32,
                min_depth: 0.0, max_depth: 1.0,
            };
            self.device.cmd_set_viewport(cmd, 0, &[viewport]);
            self.device.cmd_set_scissor(cmd, 0, &[vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: device_ctx.swapchain_extent,
            }]);

            let push = PushConstants {
                fog_color: fog,
                fog_start,
                fog_end,
                time_of_day,
                sun_intensity,
            };
            let push_bytes: &[u8] = std::slice::from_raw_parts(
                &push as *const PushConstants as *const u8,
                std::mem::size_of::<PushConstants>(),
            );

            // === Pass 1: Sky (fullscreen triangle — no vertex buffer) ===
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.sky_pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::GRAPHICS,
                self.sky_pipeline_layout, 0,
                &[self.descriptor_sets[self.current_frame]], &[],
            );
            self.device.cmd_push_constants(
                cmd, self.sky_pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0, push_bytes,
            );
            self.device.cmd_draw(cmd, 3, 1, 0, 0); // 3 verts, no buffer

            // === Pass 2: World chunks ===
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout, 0,
                &[self.descriptor_sets[self.current_frame]], &[],
            );
            self.device.cmd_push_constants(
                cmd, self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0, push_bytes,
            );

            for (pos, data) in &self.chunk_data {
                if data.index_count == 0 { continue; }
                let min_x = pos.x as f32 * 16.0;
                let min_z = pos.z as f32 * 16.0;
                let aabb_min = [min_x, 0.0, min_z];
                let aabb_max = [min_x + 16.0, 128.0, min_z + 16.0];
                if !frustum.test_aabb(aabb_min, aabb_max) { continue; }
                self.device.cmd_bind_vertex_buffers(cmd, 0, &[data.vertex_buffer], &[0]);
                self.device.cmd_bind_index_buffer(cmd, data.index_buffer, 0, vk::IndexType::UINT32);
                self.device.cmd_draw_indexed(cmd, data.index_count, 1, 0, 0, 0);
            }

            // === Pass 3: Crosshair overlay ===
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.crosshair_pipeline);
            let aspect = device_ctx.swapchain_extent.width as f32
                       / device_ctx.swapchain_extent.height as f32;
            self.device.cmd_push_constants(
                cmd,
                self.crosshair_pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                std::slice::from_raw_parts(&aspect as *const f32 as *const u8, std::mem::size_of::<f32>()),
            );
            self.device.cmd_bind_vertex_buffers(cmd, 0, &[self.crosshair_vertex_buffer], &[0]);
            self.device.cmd_draw(cmd, 12, 1, 0, 0);

            self.device.cmd_end_render_pass(cmd);
            self.device.end_command_buffer(cmd)?;

            let wait_semaphores = [self.image_available[self.current_frame]];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let command_buffers = [cmd];
            let signal_semaphores = [self.render_finished[self.current_frame]];

            let submit = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);

            self.device.queue_submit(device_ctx.queue, &[submit], self.in_flight_fences[self.current_frame])?;

            let present_wait = [self.render_finished[self.current_frame]];
            let swapchains = [device_ctx.swapchain];
            let image_indices = [image_index];

            let present = vk::PresentInfoKHR::default()
                .wait_semaphores(&present_wait)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            match device_ctx.swapchain_loader.queue_present(device_ctx.queue, &present) {
                Ok(_) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {},
                Err(e) => return Err(e.into()),
            }

            self.current_frame = (self.current_frame + 1) % MAX_FRAMES;
            self.frame_counter += 1;
        }
        Ok(())
    }

    // ===== Setup helpers =====

    fn create_descriptor_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, Box<dyn std::error::Error>> {
        let bindings = [
            // Binding 0: UBO (view/proj/camera)
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            // Binding 1: texture atlas sampler
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];

        unsafe { Ok(device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)?) }
    }

    fn create_descriptor_sets(
        device: &Device,
        layout: vk::DescriptorSetLayout,
        uniform_buffer: vk::Buffer,
        atlas_view: vk::ImageView,
        atlas_sampler: vk::Sampler,
    ) -> Result<(vk::DescriptorPool, Vec<vk::DescriptorSet>), Box<dyn std::error::Error>> {
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(MAX_FRAMES as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(MAX_FRAMES as u32),
        ];

        let pool = unsafe { device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(MAX_FRAMES as u32), None)? };

        let layouts = vec![layout; MAX_FRAMES];
        let sets = unsafe { device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts))? };

        for set in &sets {
            let buf_info = vk::DescriptorBufferInfo::default()
                .buffer(uniform_buffer).offset(0)
                .range(std::mem::size_of::<ViewUBO>() as u64);

            let img_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(atlas_view)
                .sampler(atlas_sampler);

            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(*set).dst_binding(0).dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(&buf_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(*set).dst_binding(1).dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&img_info)),
            ];

            unsafe { device.update_descriptor_sets(&writes, &[]); }
        }
        Ok((pool, sets))
    }

    fn create_uniform_buffer(
        device: &Device, device_ctx: &DeviceContext,
    ) -> Result<(vk::Buffer, vk::DeviceMemory, *mut std::ffi::c_void), Box<dyn std::error::Error>> {
        let size = std::mem::size_of::<ViewUBO>() as u64;
        let (buf, mem) = Self::create_buffer(device, device_ctx, size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)?;
        let mapped = unsafe { device.map_memory(mem, 0, size, vk::MemoryMapFlags::empty())? };
        Ok((buf, mem, mapped))
    }

    fn create_render_pass(device: &Device, format: vk::Format) -> Result<vk::RenderPass, Box<dyn std::error::Error>> {
        let attachments = [
            vk::AttachmentDescription::default()
                .format(format).samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR).store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE).stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR),
            vk::AttachmentDescription::default()
                .format(vk::Format::D32_SFLOAT).samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR).store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE).stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        ];
        let color_ref = vk::AttachmentReference::default().attachment(0).layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let depth_ref = vk::AttachmentReference::default().attachment(1).layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_ref))
            .depth_stencil_attachment(&depth_ref);
        let dep = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL).dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE);

        unsafe { Ok(device.create_render_pass(
            &vk::RenderPassCreateInfo::default()
                .attachments(&attachments)
                .subpasses(std::slice::from_ref(&subpass))
                .dependencies(std::slice::from_ref(&dep)), None)?) }
    }

    fn create_pipeline(
        device: &Device, render_pass: vk::RenderPass,
        desc_layout: vk::DescriptorSetLayout,
    ) -> Result<(vk::Pipeline, vk::PipelineLayout), Box<dyn std::error::Error>> {
        unsafe {
            let vert_code = align_spv(include_bytes!("../shaders/compiled/basic.vert.spv"));
            let frag_code = align_spv(include_bytes!("../shaders/compiled/basic.frag.spv"));
            let vert_mod = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vert_code), None)?;
            let frag_mod = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&frag_code), None)?;

            let stages = [
                vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vert_mod).name(c"main"),
                vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(frag_mod).name(c"main"),
            ];

            let binding = vk::VertexInputBindingDescription::default()
                .binding(0).stride(std::mem::size_of::<BlockVertex>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX);

            // 3 attributes: position (vec3), color_ao (rgba8 unorm), normal_light (uvec4 as u8x4)
            // normal_light now packs: .r=normal_idx, .g=light_u8, .b=tile_index, .a=uv_corner
            let attrs = [
                vk::VertexInputAttributeDescription::default().binding(0).location(0)
                    .format(vk::Format::R32G32B32_SFLOAT).offset(0),         // position: 12 bytes
                vk::VertexInputAttributeDescription::default().binding(0).location(1)
                    .format(vk::Format::R8G8B8A8_UNORM).offset(12),          // color_ao: auto 0.0-1.0
                vk::VertexInputAttributeDescription::default().binding(0).location(2)
                    .format(vk::Format::R8G8B8A8_UINT).offset(16),           // normal_light: integer
            ];

            let vi = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(std::slice::from_ref(&binding))
                .vertex_attribute_descriptions(&attrs);
            let ia = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let vp = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
            let rs = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL).line_width(1.0)
                .cull_mode(vk::CullModeFlags::BACK).front_face(vk::FrontFace::CLOCKWISE);
            let ms = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let ds = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true).depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS);
            let cba = vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA).blend_enable(false);
            let cb = vk::PipelineColorBlendStateCreateInfo::default()
                .attachments(std::slice::from_ref(&cba));
            let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dyn_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyn_states);

            let push_range = vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                .offset(0)
                .size(std::mem::size_of::<PushConstants>() as u32);

            let layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(std::slice::from_ref(&desc_layout))
                    .push_constant_ranges(std::slice::from_ref(&push_range)), None)?;

            let pi = vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages).vertex_input_state(&vi).input_assembly_state(&ia)
                .viewport_state(&vp).rasterization_state(&rs).multisample_state(&ms)
                .depth_stencil_state(&ds).color_blend_state(&cb).dynamic_state(&dyn_state)
                .layout(layout).render_pass(render_pass).subpass(0);

            let pipelines = device.create_graphics_pipelines(vk::PipelineCache::null(), &[pi], None)
                .map_err(|(_, e)| e)?;

            device.destroy_shader_module(vert_mod, None);
            device.destroy_shader_module(frag_mod, None);

            Ok((pipelines[0], layout))
        }
    }

    fn create_crosshair_pipeline(
        device: &Device, render_pass: vk::RenderPass,
    ) -> Result<(vk::Pipeline, vk::PipelineLayout), Box<dyn std::error::Error>> {
        unsafe {
            let vert_code = align_spv(include_bytes!("../shaders/compiled/crosshair.vert.spv"));
            let frag_code = align_spv(include_bytes!("../shaders/compiled/crosshair.frag.spv"));
            let vert_mod = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vert_code), None)?;
            let frag_mod = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&frag_code), None)?;

            let stages = [
                vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vert_mod).name(c"main"),
                vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(frag_mod).name(c"main"),
            ];

            let binding = vk::VertexInputBindingDescription::default()
                .binding(0).stride(8).input_rate(vk::VertexInputRate::VERTEX);
            let attr = vk::VertexInputAttributeDescription::default()
                .binding(0).location(0).format(vk::Format::R32G32_SFLOAT).offset(0);
            let vi = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(std::slice::from_ref(&binding))
                .vertex_attribute_descriptions(std::slice::from_ref(&attr));

            let ia = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let vp = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
            let rs = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL).line_width(1.0)
                .cull_mode(vk::CullModeFlags::NONE);
            let ms = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let ds = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(false).depth_write_enable(false);

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
            let dyn_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyn_states);

            let push_range = vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::VERTEX)
                .offset(0)
                .size(std::mem::size_of::<f32>() as u32);

            let layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .push_constant_ranges(std::slice::from_ref(&push_range)),
                None,
            )?;

            let pi = vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages).vertex_input_state(&vi).input_assembly_state(&ia)
                .viewport_state(&vp).rasterization_state(&rs).multisample_state(&ms)
                .depth_stencil_state(&ds).color_blend_state(&cb).dynamic_state(&dyn_state)
                .layout(layout).render_pass(render_pass).subpass(0);

            let pipelines = device.create_graphics_pipelines(vk::PipelineCache::null(), &[pi], None)
                .map_err(|(_, e)| e)?;

            device.destroy_shader_module(vert_mod, None);
            device.destroy_shader_module(frag_mod, None);

            Ok((pipelines[0], layout))
        }
    }

    fn create_sky_pipeline(
        device: &Device, render_pass: vk::RenderPass,
        desc_layout: vk::DescriptorSetLayout,
    ) -> Result<(vk::Pipeline, vk::PipelineLayout), Box<dyn std::error::Error>> {
        unsafe {
            let vert_code = align_spv(include_bytes!("../shaders/compiled/sky.vert.spv"));
            let frag_code = align_spv(include_bytes!("../shaders/compiled/sky.frag.spv"));
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

            // No vertex input — fullscreen triangle generated from gl_VertexIndex
            let vi = vk::PipelineVertexInputStateCreateInfo::default();
            let ia = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let vp = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1).scissor_count(1);
            let rs = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL).line_width(1.0)
                .cull_mode(vk::CullModeFlags::NONE);
            let ms = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);

            // Depth: write enabled at far plane (0.9999) so blocks draw in front,
            // but test disabled so the sky always draws as background
            let ds = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(false)
                .depth_write_enable(false);

            let cba = vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(false);
            let cb = vk::PipelineColorBlendStateCreateInfo::default()
                .attachments(std::slice::from_ref(&cba));
            let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dyn_state = vk::PipelineDynamicStateCreateInfo::default()
                .dynamic_states(&dyn_states);

            // Shares the same layout as the block pipeline (UBO + push constants)
            let push_range = vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                .offset(0)
                .size(std::mem::size_of::<PushConstants>() as u32);

            let layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(std::slice::from_ref(&desc_layout))
                    .push_constant_ranges(std::slice::from_ref(&push_range)), None)?;

            let pi = vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages).vertex_input_state(&vi).input_assembly_state(&ia)
                .viewport_state(&vp).rasterization_state(&rs).multisample_state(&ms)
                .depth_stencil_state(&ds).color_blend_state(&cb).dynamic_state(&dyn_state)
                .layout(layout).render_pass(render_pass).subpass(0);

            let pipelines = device.create_graphics_pipelines(
                vk::PipelineCache::null(), &[pi], None)
                .map_err(|(_, e)| e)?;

            device.destroy_shader_module(vert_mod, None);
            device.destroy_shader_module(frag_mod, None);

            Ok((pipelines[0], layout))
        }
    }

    fn create_crosshair_buffer(
        device: &Device, device_ctx: &DeviceContext,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), Box<dyn std::error::Error>> {
        let s = 0.018f32;
        let t = 0.003f32;
        #[rustfmt::skip]
        let verts: [f32; 24] = [
            -s, -t,   s, -t,   s,  t,
            -s, -t,   s,  t,  -s,  t,
            -t, -s,   t, -s,   t,  s,
            -t, -s,   t,  s,  -t,  s,
        ];
        let size = std::mem::size_of_val(&verts) as u64;
        let (buf, mem) = Self::create_buffer(device, device_ctx, size,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)?;
        unsafe {
            let data = device.map_memory(mem, 0, size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(verts.as_ptr() as *const u8, data as *mut u8, size as usize);
            device.unmap_memory(mem);
        }
        Ok((buf, mem))
    }

    fn create_framebuffers(
        device: &Device, render_pass: vk::RenderPass,
        views: &[vk::ImageView], depth: vk::ImageView, extent: vk::Extent2D,
    ) -> Result<Vec<vk::Framebuffer>, Box<dyn std::error::Error>> {
        views.iter().map(|&view| {
            let attachments = [view, depth];
            unsafe { Ok(device.create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass).attachments(&attachments)
                    .width(extent.width).height(extent.height).layers(1), None)?) }
        }).collect()
    }

    fn allocate_command_buffers(device: &Device, pool: vk::CommandPool) -> Result<Vec<vk::CommandBuffer>, Box<dyn std::error::Error>> {
        unsafe { Ok(device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool).level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(MAX_FRAMES as u32))?) }
    }

    fn create_buffer(
        device: &Device, device_ctx: &DeviceContext,
        size: u64, usage: vk::BufferUsageFlags, properties: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), Box<dyn std::error::Error>> {
        let buf = unsafe { device.create_buffer(
            &vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE), None)? };
        let req = unsafe { device.get_buffer_memory_requirements(buf) };
        let mti = find_mem_type(device_ctx.memory_properties, req.memory_type_bits, properties)?;
        let mem = unsafe { device.allocate_memory(
            &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mti), None)? };
        unsafe { device.bind_buffer_memory(buf, mem, 0)?; }
        Ok((buf, mem))
    }

    pub fn recreate_framebuffers(&mut self, device_ctx: &DeviceContext) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            self.device.device_wait_idle()?;
            for &fb in &self.framebuffers { self.device.destroy_framebuffer(fb, None); }
            self.framebuffers = Self::create_framebuffers(
                &self.device, self.render_pass,
                &device_ctx.swapchain_image_views, device_ctx.depth_image_view,
                device_ctx.swapchain_extent)?;
        }
        Ok(())
    }

    pub fn chunk_count(&self) -> usize {
        self.chunk_data.len()
    }

    pub fn loaded_chunk_positions(&self) -> Vec<ChunkPos> {
        self.chunk_data.keys().copied().collect()
    }
}

fn find_mem_type(props: vk::PhysicalDeviceMemoryProperties, filter: u32, flags: vk::MemoryPropertyFlags) -> Result<u32, Box<dyn std::error::Error>> {
    for i in 0..props.memory_type_count {
        if (filter & (1 << i)) != 0 && props.memory_types[i as usize].property_flags.contains(flags) {
            return Ok(i);
        }
    }
    Err("No suitable memory type".into())
}

fn align_spv(code: &[u8]) -> Vec<u32> {
    code.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();

            // Flush any remaining deletion queue items
            for &(_, buffer, memory) in &self.deletion_queue {
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
            }

            // Destroy chunk GPU data
            for (_, data) in self.chunk_data.drain() {
                self.device.destroy_buffer(data.vertex_buffer, None);
                self.device.free_memory(data.vertex_memory, None);
                self.device.destroy_buffer(data.index_buffer, None);
                self.device.free_memory(data.index_memory, None);
            }

            // Pipelines
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_pipeline(self.crosshair_pipeline, None);
            self.device.destroy_pipeline_layout(self.crosshair_pipeline_layout, None);
            self.device.destroy_pipeline(self.sky_pipeline, None);
            self.device.destroy_pipeline_layout(self.sky_pipeline_layout, None);

            // Crosshair vertex buffer
            self.device.destroy_buffer(self.crosshair_vertex_buffer, None);
            self.device.free_memory(self.crosshair_vertex_memory, None);

            // Texture atlas
            self.device.destroy_sampler(self.atlas_sampler, None);
            self.device.destroy_image_view(self.atlas_view, None);
            self.device.destroy_image(self.atlas_image, None);
            self.device.free_memory(self.atlas_memory, None);

            // Descriptors
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            // Uniform
            self.device.destroy_buffer(self.uniform_buffer, None);
            self.device.free_memory(self.uniform_memory, None);

            // Framebuffers + render pass
            for &fb in &self.framebuffers {
                self.device.destroy_framebuffer(fb, None);
            }
            self.device.destroy_render_pass(self.render_pass, None);

            // Sync
            for i in 0..self.image_available.len() {
                self.device.destroy_semaphore(self.image_available[i], None);
                self.device.destroy_semaphore(self.render_finished[i], None);
                self.device.destroy_fence(self.in_flight_fences[i], None);
            }
            self.device.destroy_fence(self.upload_fence, None);
        }
    }
}

// ===== Procedural Texture Atlas Generation =====
//
// Layout: 16×16 grid of 16×16px tiles in a 256×256 RGBA8 image.
//   Row 0 (tiles 0-15):  top faces   — indexed by BlockType as u8
//   Row 1 (tiles 16-31): side faces
//   Row 2 (tiles 32-47): bottom faces
//   Rows 3-15: reserved (filled with magenta for debugging)

/// Fast integer hash for procedural noise lattice points.
#[inline]
fn tex_hash(x: i32, y: i32, seed: u32) -> u32 {
    let mut h = seed;
    h ^= (x as u32).wrapping_mul(0x9e3779b9);
    h ^= (y as u32).wrapping_mul(0x517cc1b7);
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= h >> 16;
    h
}

/// Hash to float in [0.0, 1.0)
#[inline]
fn hash_f(x: i32, y: i32, seed: u32) -> f32 {
    (tex_hash(x, y, seed) & 0xFFFF) as f32 / 65536.0
}

/// Smooth tileable noise: bilinear interpolation on a lattice that wraps at `period`.
/// Input (px, py) in pixel coords [0, TILE_SIZE). Output in [0.0, 1.0).
/// `scale` controls how many lattice cells fit in one tile (lower = larger features).
#[inline]
fn smooth_noise(px: u32, py: u32, seed: u32, scale: f32) -> f32 {
    let period = TILE_SIZE as f32;
    // Fractional lattice position that wraps
    let fx = (px as f32 / period) * scale;
    let fy = (py as f32 / period) * scale;

    let cells = scale.ceil() as i32; // number of lattice cells in one tile

    let ix = fx.floor() as i32;
    let iy = fy.floor() as i32;
    let frac_x = fx - fx.floor();
    let frac_y = fy - fy.floor();

    // Hermite smoothstep for smooth interpolation
    let sx = frac_x * frac_x * (3.0 - 2.0 * frac_x);
    let sy = frac_y * frac_y * (3.0 - 2.0 * frac_y);

    // Wrap lattice coordinates for seamless tiling
    let x0 = ((ix % cells) + cells) % cells;
    let x1 = ((ix + 1) % cells + cells) % cells;
    let y0 = ((iy % cells) + cells) % cells;
    let y1 = ((iy + 1) % cells + cells) % cells;

    let v00 = hash_f(x0, y0, seed);
    let v10 = hash_f(x1, y0, seed);
    let v01 = hash_f(x0, y1, seed);
    let v11 = hash_f(x1, y1, seed);

    let a = v00 + sx * (v10 - v00);
    let b = v01 + sx * (v11 - v01);
    a + sy * (b - a)
}

/// Multi-octave tileable noise (fBm). Available for richer detail if needed.
#[inline]
#[allow(dead_code)]
fn fbm_noise(px: u32, py: u32, seed: u32, base_scale: f32, octaves: u32) -> f32 {
    let mut value = 0.0f32;
    let mut amplitude = 1.0f32;
    let mut scale = base_scale;
    let mut max_val = 0.0f32;
    for oct in 0..octaves {
        value += amplitude * smooth_noise(px, py, seed.wrapping_add(oct * 137), scale);
        max_val += amplitude;
        amplitude *= 0.5;
        scale *= 2.0;
    }
    value / max_val
}

/// Write a single pixel into the atlas buffer
#[inline]
fn set_pixel(pixels: &mut [u8], x: u32, y: u32, r: u8, g: u8, b: u8, a: u8) {
    let idx = ((y * ATLAS_SIZE + x) * 4) as usize;
    if idx + 3 < pixels.len() {
        pixels[idx]     = r;
        pixels[idx + 1] = g;
        pixels[idx + 2] = b;
        pixels[idx + 3] = a;
    }
}

/// Mix two u8 colors by factor t (0.0-1.0)
#[inline]
fn mix_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 * (1.0 - t) + b as f32 * t + 0.5).clamp(0.0, 255.0) as u8
}

/// Clamp f32 to u8 range
#[inline]
fn clamp_u8(v: f32) -> u8 {
    v.clamp(0.0, 255.0) as u8
}

fn generate_atlas_pixels() -> Vec<u8> {
    let total = (ATLAS_SIZE * ATLAS_SIZE * 4) as usize;
    let mut pixels = vec![0u8; total];

    // Fill entire atlas with debug magenta so unused tiles are visible
    for y in 0..ATLAS_SIZE {
        for x in 0..ATLAS_SIZE {
            set_pixel(&mut pixels, x, y, 255, 0, 255, 255);
        }
    }

    // Generate each BlockType's 3 tile variants (top, side, bottom)
    for block_id in 0..16u8 {
        let col = block_id as u32;

        let top_origin    = (col * TILE_SIZE, 0 * TILE_SIZE);
        let side_origin   = (col * TILE_SIZE, 1 * TILE_SIZE);
        let bottom_origin = (col * TILE_SIZE, 2 * TILE_SIZE);

        match block_id {
            0 => { // Air — white
                fill_solid(&mut pixels, col, 0, [255, 255, 255]);
                fill_solid(&mut pixels, col, 1, [255, 255, 255]);
                fill_solid(&mut pixels, col, 2, [255, 255, 255]);
            }
            1 => { // Grass — top: crunchy green, side: dirt+green strip, bottom: dirt
                // Top: lush green with per-pixel crunch
                for py in 0..TILE_SIZE { for px in 0..TILE_SIZE {
                    let n = hash_f(px as i32, py as i32, 100);
                    let n2 = hash_f(px as i32, py as i32, 105);
                    let r = clamp_u8(55.0 + n * 55.0 - n2 * 15.0);
                    let g = clamp_u8(120.0 + n * 80.0 + n2 * 25.0);
                    let b = clamp_u8(25.0 + n * 30.0);
                    set_pixel(&mut pixels, top_origin.0 + px, top_origin.1 + py, r, g, b, 255);
                }}
                // Side: dirt with green strip (top 3 rows)
                for py in 0..TILE_SIZE { for px in 0..TILE_SIZE {
                    let n = hash_f(px as i32, py as i32, 200);
                    if py < 3 {
                        let blend = 1.0 - (py as f32 / 3.0);
                        let r = clamp_u8(60.0 * blend + 140.0 * (1.0 - blend) + n * 30.0);
                        let g = clamp_u8(120.0 * blend + 95.0 * (1.0 - blend) + n * 35.0);
                        let b = clamp_u8(25.0 * blend + 60.0 * (1.0 - blend) + n * 12.0);
                        set_pixel(&mut pixels, side_origin.0 + px, side_origin.1 + py, r, g, b, 255);
                    } else {
                        gen_dirt_pixel(&mut pixels, side_origin.0 + px, side_origin.1 + py, px, py, 210);
                    }
                }}
                // Bottom: dirt
                gen_dirt_tile(&mut pixels, bottom_origin.0, bottom_origin.1, 300);
            }
            2 => { // Dirt
                gen_dirt_tile(&mut pixels, top_origin.0, top_origin.1, 400);
                gen_dirt_tile(&mut pixels, side_origin.0, side_origin.1, 410);
                gen_dirt_tile(&mut pixels, bottom_origin.0, bottom_origin.1, 420);
            }
            3 => { // Stone — gray with crunchy texture, smooth crack placement
                for face in 0..3u32 {
                    let (ox, oy) = [top_origin, side_origin, bottom_origin][face as usize];
                    let seed_base = 500 + face * 100;
                    for py in 0..TILE_SIZE { for px in 0..TILE_SIZE {
                        let n = hash_f(px as i32, py as i32, seed_base);
                        // Smooth noise only for crack region detection (organic shapes)
                        let crack_region = smooth_noise(px, py, seed_base + 50, 3.0);
                        let crack = if crack_region < 0.25 { 0.7 } else { 1.0 };
                        let base = (125.0 + n * 55.0) * crack;
                        let r = clamp_u8(base);
                        let g = clamp_u8(base);
                        let b = clamp_u8(base * 1.04);
                        set_pixel(&mut pixels, ox + px, oy + py, r, g, b, 255);
                    }}
                }
            }
            4 => { // Sand — warm beige, crunchy grain
                for face in 0..3u32 {
                    let (ox, oy) = [top_origin, side_origin, bottom_origin][face as usize];
                    for py in 0..TILE_SIZE { for px in 0..TILE_SIZE {
                        let n = hash_f(px as i32, py as i32, 600 + face * 100);
                        let n2 = hash_f(px as i32, py as i32, 650 + face * 100);
                        let r = clamp_u8(205.0 + n * 35.0 + n2 * 10.0);
                        let g = clamp_u8(190.0 + n * 32.0 + n2 * 8.0);
                        let b = clamp_u8(135.0 + n * 28.0 + n2 * 8.0);
                        set_pixel(&mut pixels, ox + px, oy + py, r, g, b, 255);
                    }}
                }
            }
            5 => { // Water — blue, smooth waves + crunchy sparkle
                for face in 0..3u32 {
                    let (ox, oy) = [top_origin, side_origin, bottom_origin][face as usize];
                    for py in 0..TILE_SIZE { for px in 0..TILE_SIZE {
                        let n = hash_f(px as i32, py as i32, 700 + face * 100);
                        // Smooth wave pattern (structural)
                        let wave = smooth_noise(px, py, 750 + face * 100, 2.0);
                        let r = clamp_u8(25.0 + wave * 25.0 + n * 18.0);
                        let g = clamp_u8(65.0 + wave * 35.0 + n * 28.0);
                        let b = clamp_u8(155.0 + n * 50.0 + wave * 20.0);
                        set_pixel(&mut pixels, ox + px, oy + py, r, g, b, 220);
                    }}
                }
            }
            6 => { // Wood — top/bottom: rings, side: bark
                // Top: concentric rings + crunch
                for py in 0..TILE_SIZE { for px in 0..TILE_SIZE {
                    let cx = px as f32 - 7.5;
                    let cy = py as f32 - 7.5;
                    let dist = (cx * cx + cy * cy).sqrt();
                    let ring = (dist * 0.9).sin() * 0.5 + 0.5;
                    let n = hash_f(px as i32, py as i32, 800);
                    let base = 135.0 + ring * 65.0 + n * 25.0;
                    let r = clamp_u8(base);
                    let g = clamp_u8(base * 0.65);
                    let b = clamp_u8(base * 0.35);
                    set_pixel(&mut pixels, top_origin.0 + px, top_origin.1 + py, r, g, b, 255);
                }}
                // Side: vertical bark — smooth streaks for structure, hash for crunch
                for py in 0..TILE_SIZE { for px in 0..TILE_SIZE {
                    let n = hash_f(px as i32, py as i32, 810);
                    // Smooth vertical streaks (structural)
                    let streak = smooth_noise(px, py / 3, 820, 3.0);
                    let base = 110.0 + streak * 60.0 + n * 25.0;
                    let r = clamp_u8(base);
                    let g = clamp_u8(base * 0.65);
                    let b = clamp_u8(base * 0.35);
                    set_pixel(&mut pixels, side_origin.0 + px, side_origin.1 + py, r, g, b, 255);
                }}
                // Bottom: same as top
                for py in 0..TILE_SIZE { for px in 0..TILE_SIZE {
                    let cx = px as f32 - 7.5;
                    let cy = py as f32 - 7.5;
                    let dist = (cx * cx + cy * cy).sqrt();
                    let ring = (dist * 0.9).sin() * 0.5 + 0.5;
                    let n = hash_f(px as i32, py as i32, 830);
                    let base = 125.0 + ring * 65.0 + n * 25.0;
                    let r = clamp_u8(base);
                    let g = clamp_u8(base * 0.65);
                    let b = clamp_u8(base * 0.35);
                    set_pixel(&mut pixels, bottom_origin.0 + px, bottom_origin.1 + py, r, g, b, 255);
                }}
            }
            7 => { // Leaves — crunchy green, smooth hole placement
                for face in 0..3u32 {
                    let (ox, oy) = [top_origin, side_origin, bottom_origin][face as usize];
                    for py in 0..TILE_SIZE { for px in 0..TILE_SIZE {
                        let n = hash_f(px as i32, py as i32, 900 + face * 100);
                        // Smooth noise only for hole region detection (organic shapes)
                        let hole = smooth_noise(px, py, 950 + face * 100, 3.0);
                        if hole < 0.22 {
                            set_pixel(&mut pixels, ox + px, oy + py, 90, 170, 60, 200);
                        } else {
                            let g = clamp_u8(95.0 + n * 85.0);
                            let r = clamp_u8(25.0 + n * 45.0);
                            let b = clamp_u8(12.0 + n * 28.0);
                            set_pixel(&mut pixels, ox + px, oy + py, r, g, b, 255);
                        }
                    }}
                }
            }
            8 => { // Snow — crunchy white with blue tint
                for face in 0..3u32 {
                    let (ox, oy) = [top_origin, side_origin, bottom_origin][face as usize];
                    for py in 0..TILE_SIZE { for px in 0..TILE_SIZE {
                        let n = hash_f(px as i32, py as i32, 1000 + face * 100);
                        let sparkle = hash_f(px as i32, py as i32, 1050 + face * 100);
                        let base = 225.0 + n * 25.0 + sparkle * 5.0;
                        let r = clamp_u8(base - 5.0);
                        let g = clamp_u8(base - 2.0);
                        let b = clamp_u8(base);
                        set_pixel(&mut pixels, ox + px, oy + py, r, g, b, 255);
                    }}
                }
            }
            9 => { // Bedrock — very dark, chunky hash
                for face in 0..3u32 {
                    let (ox, oy) = [top_origin, side_origin, bottom_origin][face as usize];
                    for py in 0..TILE_SIZE { for px in 0..TILE_SIZE {
                        let n = hash_f(px as i32, py as i32, 1100 + face * 100);
                        // 2x2 pixel blocks for chunkier look
                        let n2 = hash_f(px as i32 / 2, py as i32 / 2, 1150 + face * 100);
                        let base = 30.0 + n * 30.0 + n2 * 25.0;
                        let r = clamp_u8(base);
                        let g = clamp_u8(base);
                        let b = clamp_u8(base * 1.06);
                        set_pixel(&mut pixels, ox + px, oy + py, r, g, b, 255);
                    }}
                }
            }
            10 => { // Gravel — crunchy mixed pebbles, smooth pebble shapes
                for face in 0..3u32 {
                    let (ox, oy) = [top_origin, side_origin, bottom_origin][face as usize];
                    for py in 0..TILE_SIZE { for px in 0..TILE_SIZE {
                        let n = hash_f(px as i32, py as i32, 1200 + face * 100);
                        // Smooth noise for pebble shape regions
                        let pebble = smooth_noise(px, py, 1250 + face * 100, 3.0);
                        let base = 95.0 + pebble * 70.0 + n * 20.0;
                        let r = clamp_u8(base);
                        let g = clamp_u8(base * 0.95);
                        let b = clamp_u8(base * 0.88);
                        set_pixel(&mut pixels, ox + px, oy + py, r, g, b, 255);
                    }}
                }
            }
            11 => { // Coal — stone with dark specks
                gen_ore_tile(&mut pixels, top_origin.0, top_origin.1, 1300, [30, 30, 35]);
                gen_ore_tile(&mut pixels, side_origin.0, side_origin.1, 1310, [30, 30, 35]);
                gen_ore_tile(&mut pixels, bottom_origin.0, bottom_origin.1, 1320, [30, 30, 35]);
            }
            12 => { // Iron — stone with tan specks
                gen_ore_tile(&mut pixels, top_origin.0, top_origin.1, 1400, [180, 140, 100]);
                gen_ore_tile(&mut pixels, side_origin.0, side_origin.1, 1410, [180, 140, 100]);
                gen_ore_tile(&mut pixels, bottom_origin.0, bottom_origin.1, 1420, [180, 140, 100]);
            }
            13 => { // Gold — stone with gold specks
                gen_ore_tile(&mut pixels, top_origin.0, top_origin.1, 1500, [240, 200, 50]);
                gen_ore_tile(&mut pixels, side_origin.0, side_origin.1, 1510, [240, 200, 50]);
                gen_ore_tile(&mut pixels, bottom_origin.0, bottom_origin.1, 1520, [240, 200, 50]);
            }
            14 => { // Diamond — stone with cyan specks
                gen_ore_tile(&mut pixels, top_origin.0, top_origin.1, 1600, [100, 220, 235]);
                gen_ore_tile(&mut pixels, side_origin.0, side_origin.1, 1610, [100, 220, 235]);
                gen_ore_tile(&mut pixels, bottom_origin.0, bottom_origin.1, 1620, [100, 220, 235]);
            }
            15 => { // Torch — warm crunchy self-illuminated
                // Top (flame)
                for py in 0..TILE_SIZE { for px in 0..TILE_SIZE {
                    let n = hash_f(px as i32, py as i32, 1700);
                    let r = 255;
                    let g = clamp_u8(160.0 + n * 60.0);
                    let b = clamp_u8(30.0 + n * 55.0);
                    set_pixel(&mut pixels, top_origin.0 + px, top_origin.1 + py, r, g, b, 255);
                }}
                // Side (torch body)
                for py in 0..TILE_SIZE { for px in 0..TILE_SIZE {
                    let n = hash_f(px as i32, py as i32, 1710);
                    let base = 150.0 + n * 50.0;
                    let r = clamp_u8(base);
                    let g = clamp_u8(base * 0.7);
                    let b = clamp_u8(base * 0.32);
                    set_pixel(&mut pixels, side_origin.0 + px, side_origin.1 + py, r, g, b, 255);
                }}
                // Bottom
                for py in 0..TILE_SIZE { for px in 0..TILE_SIZE {
                    let n = hash_f(px as i32, py as i32, 1720);
                    let base = 135.0 + n * 45.0;
                    set_pixel(&mut pixels, bottom_origin.0 + px, bottom_origin.1 + py,
                        clamp_u8(base), clamp_u8(base * 0.65), clamp_u8(base * 0.3), 255);
                }}
            }
            _ => {}
        }
    }

    pixels
}

/// Fill a tile slot with a solid color (for Air, etc.)
fn fill_solid(pixels: &mut [u8], col: u32, row: u32, rgb: [u8; 3]) {
    let ox = col * TILE_SIZE;
    let oy = row * TILE_SIZE;
    for py in 0..TILE_SIZE { for px in 0..TILE_SIZE {
        set_pixel(pixels, ox + px, oy + py, rgb[0], rgb[1], rgb[2], 255);
    }}
}

/// Generate a single dirt pixel — raw hash crunch
fn gen_dirt_pixel(pixels: &mut [u8], gx: u32, gy: u32, px: u32, py: u32, seed: u32) {
    let n = hash_f(px as i32, py as i32, seed);
    let n2 = hash_f(px as i32, py as i32, seed + 50);
    let base = 115.0 + n * 50.0 + n2 * 18.0;
    let r = clamp_u8(base);
    let g = clamp_u8(base * 0.72);
    let b = clamp_u8(base * 0.45);
    set_pixel(pixels, gx, gy, r, g, b, 255);
}

/// Generate a full dirt tile at the given pixel origin
fn gen_dirt_tile(pixels: &mut [u8], ox: u32, oy: u32, seed: u32) {
    for py in 0..TILE_SIZE { for px in 0..TILE_SIZE {
        gen_dirt_pixel(pixels, ox + px, oy + py, px, py, seed);
    }}
}

/// Generate an ore tile: crunchy stone base with smooth ore blob boundaries
fn gen_ore_tile(pixels: &mut [u8], ox: u32, oy: u32, seed: u32, ore_color: [u8; 3]) {
    for py in 0..TILE_SIZE { for px in 0..TILE_SIZE {
        let n = hash_f(px as i32, py as i32, seed);
        // Smooth noise ONLY for ore blob boundary detection (organic shapes)
        let ore_n = smooth_noise(px, py, seed + 50, 3.0);
        // Crunchy stone base
        let base = 122.0 + n * 50.0;
        let mut r = clamp_u8(base);
        let mut g = clamp_u8(base);
        let mut b = clamp_u8(base * 1.03);

        // Ore blobs: smooth regions where noise < threshold
        if ore_n < 0.28 {
            let t = (1.0 - ore_n / 0.28).clamp(0.0, 1.0) * 0.85;
            r = mix_u8(r, ore_color[0], t);
            g = mix_u8(g, ore_color[1], t);
            b = mix_u8(b, ore_color[2], t);
        }
        set_pixel(pixels, ox + px, oy + py, r, g, b, 255);
    }}
}