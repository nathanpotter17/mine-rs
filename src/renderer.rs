use ash::{vk, Device};
use std::collections::HashMap;
use std::sync::Arc;
use crate::device::DeviceContext;
use crate::world::{BlockVertex, ChunkMesh, ChunkPos};

// Push constant for fog / view info
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PushConstants {
    pub fog_color: [f32; 4],
    pub fog_start: f32,
    pub fog_end: f32,
    pub sun_dir: [f32; 2], // packed x,z (y derived)
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

/// 6 planes in [a,b,c,d] form (ax+by+cz+d >= 0 means inside).
/// Extracted from the combined view-projection matrix each frame.
pub struct Frustum {
    planes: [[f32; 4]; 6],
}

impl Frustum {
    /// Extract 6 frustum planes from view and projection matrices.
    ///
    /// Our matrices are stored **column-major**: `m[col][row]`.
    /// look_at stores basis vectors as columns, perspective likewise.
    /// Gribb-Hartmann expects row indexing on the combined clip matrix,
    /// so we index as m[col][row] and extract accordingly.
    pub fn from_view_proj(v: &[[f32; 4]; 4], p: &[[f32; 4]; 4]) -> Self {
        // Compute clip = P * V in column-major layout.
        // m[col][row] = sum_k p[k][row] * v[col][k]
        let mut m = [[0.0f32; 4]; 4];
        for col in 0..4 {
            for row in 0..4 {
                m[col][row] = p[0][row]*v[col][0] + p[1][row]*v[col][1]
                            + p[2][row]*v[col][2] + p[3][row]*v[col][3];
            }
        }

        // Gribb-Hartmann: extract planes from the clip matrix.
        // For column-major m[col][row], "row i" is accessed as m[0][i], m[1][i], m[2][i], m[3][i].
        // Plane equations use rows of the clip matrix:
        //   Left:   row3 + row0
        //   Right:  row3 - row0
        //   Bottom: row3 + row1
        //   Top:    row3 - row1
        //   Near:   row3 + row2
        //   Far:    row3 - row2
        let row = |r: usize| -> [f32; 4] { [m[0][r], m[1][r], m[2][r], m[3][r]] };
        let r0 = row(0); let r1 = row(1); let r2 = row(2); let r3 = row(3);

        let mut planes = [[0.0f32; 4]; 6];
        for i in 0..4 {
            planes[0][i] = r3[i] + r0[i]; // left
            planes[1][i] = r3[i] - r0[i]; // right
            planes[2][i] = r3[i] + r1[i]; // bottom
            planes[3][i] = r3[i] - r1[i]; // top
            planes[4][i] = r3[i] + r2[i]; // near
            planes[5][i] = r3[i] - r2[i]; // far
        }

        // Normalize each plane
        for plane in &mut planes {
            let len = (plane[0]*plane[0] + plane[1]*plane[1] + plane[2]*plane[2]).sqrt();
            if len > 1e-8 {
                let inv = 1.0 / len;
                plane[0] *= inv;
                plane[1] *= inv;
                plane[2] *= inv;
                plane[3] *= inv;
            }
        }
        Self { planes }
    }

    /// Test AABB against frustum. Returns true if the box is at least partially inside.
    #[inline]
    pub fn test_aabb(&self, min: [f32; 3], max: [f32; 3]) -> bool {
        for plane in &self.planes {
            // Find the p-vertex: corner most in the direction of the plane normal
            let px = if plane[0] >= 0.0 { max[0] } else { min[0] };
            let py = if plane[1] >= 0.0 { max[1] } else { min[1] };
            let pz = if plane[2] >= 0.0 { max[2] } else { min[2] };
            // If the p-vertex is outside this plane, the entire AABB is outside
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

/// Pending staging copy: staging buffer/mem to destroy after transfer completes
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

impl Renderer {
    pub fn new(device_ctx: &DeviceContext) -> Result<Self, Box<dyn std::error::Error>> {
        let device = Arc::new(device_ctx.device.clone());

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
            Self::create_descriptor_sets(&device, descriptor_set_layout, uniform_buffer)?;

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

        let upload_fence = unsafe {
            device.create_fence(&vk::FenceCreateInfo::default(), None)?
        };

        Ok(Self {
            device,
            chunk_data: HashMap::new(),
            pipeline, pipeline_layout,
            descriptor_set_layout, descriptor_pool, descriptor_sets,
            uniform_buffer, uniform_memory, uniform_mapped,
            render_pass, framebuffers,
            command_buffers,
            image_available, render_finished, in_flight_fences,
            current_frame: 0,
            crosshair_pipeline, crosshair_pipeline_layout,
            crosshair_vertex_buffer, crosshair_vertex_memory,
            memory_properties: device_ctx.memory_properties,
            deletion_queue: Vec::new(),
            frame_counter: 0,
            pending_copies: Vec::new(),
            upload_fence,
        })
    }

    pub fn upload_chunk_mesh(
        &mut self,
        device_ctx: &DeviceContext,
        pos: ChunkPos,
        mesh: &ChunkMesh,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Remove old data
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

    /// Stage a buffer upload: creates staging + DEVICE_LOCAL buffers, memcpys data into staging,
    /// and queues a copy. Call `flush_uploads` to submit all pending copies in one batch.
    fn stage_buffer_upload<T: Copy>(
        &mut self,
        device_ctx: &DeviceContext,
        data: &[T], size: u64, usage: vk::BufferUsageFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), Box<dyn std::error::Error>> {
        unsafe {
            // HOST_VISIBLE staging
            let (staging_buf, staging_mem) = Self::create_buffer(&self.device, device_ctx, size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)?;
            let mapped = self.device.map_memory(staging_mem, 0, size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, mapped as *mut u8, size as usize);
            self.device.unmap_memory(staging_mem);

            // DEVICE_LOCAL target
            let (buf, mem) = Self::create_buffer(&self.device, device_ctx, size,
                usage | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

            self.pending_copies.push(StagingCopy {
                staging_buf, staging_mem, dst_buf: buf, size,
            });

            Ok((buf, mem))
        }
    }

    /// Submit all pending staging→device copies in one command buffer, wait once.
    /// Call after all upload_chunk_mesh calls for this frame, before render().
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

            // Cleanup all staging buffers
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

            // Sky color / fog color
            let sky = [0.53, 0.72, 0.92, 1.0];

            let clear_values = [
                vk::ClearValue { color: vk::ClearColorValue { float32: sky } },
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

            // Viewport + scissor
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

            // Draw world chunks
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout, 0,
                &[self.descriptor_sets[self.current_frame]], &[],
            );

            // Push constants (fog)
            let push = PushConstants {
                fog_color: sky,
                fog_start: 60.0,
                fog_end: 90.0,
                sun_dir: [0.4, 0.3],
            };
            let push_bytes: &[u8] = std::slice::from_raw_parts(
                &push as *const PushConstants as *const u8,
                std::mem::size_of::<PushConstants>(),
            );
            self.device.cmd_push_constants(
                cmd, self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0, push_bytes,
            );

            // Draw each chunk (frustum culled)
            for (pos, data) in &self.chunk_data {
                if data.index_count == 0 { continue; }
                // Chunk AABB: world-space bounding box
                let min_x = pos.x as f32 * 16.0;
                let min_z = pos.z as f32 * 16.0;
                let aabb_min = [min_x, 0.0, min_z];
                let aabb_max = [min_x + 16.0, 128.0, min_z + 16.0];
                if !frustum.test_aabb(aabb_min, aabb_max) { continue; }
                self.device.cmd_bind_vertex_buffers(cmd, 0, &[data.vertex_buffer], &[0]);
                self.device.cmd_bind_index_buffer(cmd, data.index_buffer, 0, vk::IndexType::UINT32);
                self.device.cmd_draw_indexed(cmd, data.index_count, 1, 0, 0, 0);
            }

            // Draw crosshair
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.crosshair_pipeline);
            self.device.cmd_bind_vertex_buffers(cmd, 0, &[self.crosshair_vertex_buffer], &[0]);
            self.device.cmd_draw(cmd, 12, 1, 0, 0);

            self.device.cmd_end_render_pass(cmd);
            self.device.end_command_buffer(cmd)?;

            // Submit
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
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);

        unsafe { Ok(device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(std::slice::from_ref(&binding)), None)?) }
    }

    fn create_descriptor_sets(
        device: &Device,
        layout: vk::DescriptorSetLayout,
        uniform_buffer: vk::Buffer,
    ) -> Result<(vk::DescriptorPool, Vec<vk::DescriptorSet>), Box<dyn std::error::Error>> {
        let pool = unsafe { device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&[vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(MAX_FRAMES as u32)])
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
            let write = vk::WriteDescriptorSet::default()
                .dst_set(*set).dst_binding(0).dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_info));
            unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]); }
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

            let layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)?;

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

            // Cleanup any pending staging buffers that weren't flushed
            for copy in self.pending_copies.drain(..) {
                self.device.destroy_buffer(copy.staging_buf, None);
                self.device.free_memory(copy.staging_mem, None);
            }

            for &(_, buffer, memory) in &self.deletion_queue {
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
            }

            for data in self.chunk_data.values() {
                self.device.destroy_buffer(data.vertex_buffer, None);
                self.device.free_memory(data.vertex_memory, None);
                self.device.destroy_buffer(data.index_buffer, None);
                self.device.free_memory(data.index_memory, None);
            }

            self.device.destroy_fence(self.upload_fence, None);
            for &f in &self.in_flight_fences { self.device.destroy_fence(f, None); }
            for &s in &self.render_finished { self.device.destroy_semaphore(s, None); }
            for &s in &self.image_available { self.device.destroy_semaphore(s, None); }

            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            for &fb in &self.framebuffers { self.device.destroy_framebuffer(fb, None); }

            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_pipeline(self.crosshair_pipeline, None);
            self.device.destroy_pipeline_layout(self.crosshair_pipeline_layout, None);

            self.device.destroy_render_pass(self.render_pass, None);

            self.device.unmap_memory(self.uniform_memory);
            self.device.destroy_buffer(self.uniform_buffer, None);
            self.device.free_memory(self.uniform_memory, None);

            self.device.destroy_buffer(self.crosshair_vertex_buffer, None);
            self.device.free_memory(self.crosshair_vertex_memory, None);
        }
    }
}