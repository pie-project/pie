//! What the fused `rms_rope` costs, against the two kernels it replaces.
//!
//! `tests/sdpa_bench.rs` sets the method and its device harness is copied
//! here verbatim -- device timestamps, one recorded command buffer per shape,
//! warm-ups then a median over many barrier-separated dispatches,
//! device-local buffers. Read `tests/qmv_bench.rs`'s header for why each of
//! those is so; only what is DIFFERENT is recorded below.
//!
//! # What this file is for, and what would make it say no
//!
//! A decode step fires `rms_strided_head_row` then a barrier then `neox`,
//! twice a layer (q and k). Suppressing only the barrier in front of every
//! `neox` was measured at 0.099 ms a step over 28 layers, and that number is
//! the ENTIRE case for fusing the two: the arithmetic saved is one store and
//! one load of a 128-wide head, which is nothing. So the question this file
//! answers is not "is the fused kernel fast" but "does the fused kernel give
//! back more than the barrier was worth".
//!
//! It can. The fused grid is one workgroup per (head, row) where `neox`'s is
//! one THREAD per rotary pair, so at a 128-wide head the rotation half of the
//! fused kernel has 64 pairs of work for a group that the norm half wanted
//! 256 threads for. If that idleness costs more than 3.5 us a fire, the pair
//! stays and this file is the record of why.
//!
//! # How to run it
//!
//! ```sh
//! PIE_SLANGC=$HOME/slang/bin/slangc cargo test -p kernels-vulkan \
//!   --features native --test norm_bench -- --ignored --nocapture
//! ```
//!
//! `#[ignore]`, for the same reason the other two sweeps are: it prints a
//! table and proves no property. The correctness claim lives in
//! `tests/gpu.rs`.
//!
//! # THE ANSWER, measured
//!
//! qwen3-0.6B's heads, 128 wide, fully rotated, median of 128 intervals of 32
//! passes each. `groups` is the widest dispatch in the chain.
//!
//! | heads | rows | rms + neox | rms_rope | delta |
//! |---|---|---|---|---|
//! | 16 | 1 | 6.15 | 4.10 | -2.05 |
//! | 8 | 1 | 6.14 | 4.10 | -2.05 |
//! | 16 | 8 | 10.31 | 4.10 | -6.21 |
//! | 8 | 8 | 8.19 | 4.10 | -4.10 |
//! | 16 | 64 | 42.65 | 6.14 | -36.51 |
//! | 8 | 64 | 23.48 | 4.16 | -19.32 |
//! | 16 | 512 | 290.56 | 20.45 | -270.11 |
//! | 8 | 512 | 147.97 | 12.29 | -135.68 |
//!
//! The fused kernel wins at every shape and it never once loses, so the
//! occupancy worry this file was written to test -- 64 pairs of rotation work
//! in a group the norm wanted 256 threads for -- is real but far too small to
//! see. It is worth being clear that the decode rows are pinned to a floor:
//! every figure at 1 to 8 rows is an exact multiple of 2.048 us, and timing
//! 32 passes per interval did not break the grid, so what those rows report
//! is a per-dispatch floor and not a cost. Two runs disagreed by one whole
//! tick on `16 x 1` (6.15 against 8.19) while the fused column sat at 4.10 in
//! both. **Read the decode rows as "one dispatch cheaper", not as a
//! microsecond count.**
//!
//! # The prefill column is the surprise, and it is not about the fusion
//!
//! 290 us to rotate 512 tokens is not a rotation; 512 x 16 heads x 128 dims
//! at bf16 is 2 MB read and written, which this card does in about 4 us. The
//! cause is in `rope.rs`'s own grid doc: `neox.slang` is
//! `[numthreads(1, 1, 1)]` and the launch is `[rotary/2, heads, rows]`, so a
//! 512-token prefill dispatches 524288 workgroups of ONE THREAD, each doing
//! two loads, two multiplies and two stores. The fused kernel's 20.45 us is
//! what the same work costs when a workgroup is 256 threads wide.
//!
//! So most of the prefill delta belongs to `neox`'s one-thread grid and would
//! be recovered by widening that grid alone, WITHOUT any fusion. That does
//! not make the fusion wrong -- it subsumes the fix -- but it does mean this
//! table must not be quoted as "fusing norm and rope is worth 270 us". The
//! honest split: the fusion is worth one dispatch and one barrier a fire,
//! which is what the decode rows show, and it happens to also carry a
//! long-standing prefill defect out with it.


use ash::vk;
use kernels_vulkan::Capability;

/// Where a `native` build left the modules, or `None` if this is not one.
const SPV_DIR: Option<&str> = option_env!("PIE_KERNELS_VULKAN_SPV_DIR");

/// Timed dispatches per shape.
const ITERATIONS: usize = 128;

/// Untimed dispatches first.
const WARMUPS: usize = 16;

/// Passes per timed interval. See `measure` for why this is not one.
const REPEATS: usize = 32;

// ---------------------------------------------------------------------------
// the device
// ---------------------------------------------------------------------------

/// The same cut-down harness `qmv_bench.rs` opens, plus the 8- and 16-bit
/// storage features: the paged kernel reads `uint8_t` masks and `bfloat16`
/// planes, and a module may not declare a capability whose feature is off.
struct Bench {
    _entry: ash::Entry,
    _instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    family: u32,
    memory: vk::PhysicalDeviceMemoryProperties,
    name: String,
    /// Nanoseconds per timestamp tick.
    period: f32,
    /// What the device says its subgroup is, which is what decides how many
    /// keys the rewritten kernel walks in parallel.
    subgroup: u32,
}

fn unavailable() -> Option<&'static str> {
    if SPV_DIR.is_none() {
        return Some("built without --features native, so there are no SPIR-V modules");
    }
    None
}

impl Bench {
    fn open() -> Result<Self, String> {
        let entry = unsafe { ash::Entry::load() }.map_err(|e| format!("no Vulkan loader: {e}"))?;
        let app = vk::ApplicationInfo::default()
            .application_name(c"kernels-vulkan sdpa bench")
            .api_version(vk::API_VERSION_1_3);
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )
        }
        .map_err(|e| format!("no Vulkan instance: {e}"))?;

        let devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| format!("cannot enumerate devices: {e}"))?;
        let seen: Vec<(vk::PhysicalDevice, String, u8, Option<u32>)> = devices
            .iter()
            .map(|&d| {
                let props = unsafe { instance.get_physical_device_properties(d) };
                let name = props
                    .device_name_as_c_str()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|_| "<unnamed>".into());
                let rank = match props.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                    vk::PhysicalDeviceType::CPU => 4,
                    _ => 3,
                };
                let family = unsafe { instance.get_physical_device_queue_family_properties(d) }
                    .iter()
                    .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                    .map(|i| i as u32);
                (d, name, rank, family)
            })
            .collect();
        let pin = std::env::var("PIE_VULKAN_DEVICE").ok();
        let pin = pin.as_deref().map(str::trim).filter(|p| !p.is_empty());
        let usable = || seen.iter().filter(|(_, _, _, f)| f.is_some());
        let chosen = match pin {
            Some(want) => {
                let want = want.to_ascii_lowercase();
                usable().find(|(_, n, _, _)| n.to_ascii_lowercase().contains(&want))
            }
            None => usable().min_by_key(|(_, _, r, _)| *r),
        };
        let Some((physical, name, _, family)) = chosen else {
            return Err("no Vulkan device with a compute queue".into());
        };
        let (physical, name, family) = (*physical, name.clone(), family.expect("filtered"));

        let mut sub = vk::PhysicalDeviceSubgroupProperties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sub);
        unsafe { instance.get_physical_device_properties2(physical, &mut props2) };
        let props = props2.properties;
        if props.limits.timestamp_compute_and_graphics != vk::TRUE
            || props.limits.timestamp_period <= 0.0
        {
            return Err(format!("{name} does not report usable timestamps"));
        }

        let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut query = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut f11)
            .push_next(&mut f12);
        unsafe { instance.get_physical_device_features2(physical, &mut query) };
        let core = query.features;

        let mut e11 = vk::PhysicalDeviceVulkan11Features::default()
            .storage_buffer16_bit_access(f11.storage_buffer16_bit_access == vk::TRUE)
            .uniform_and_storage_buffer16_bit_access(
                f11.uniform_and_storage_buffer16_bit_access == vk::TRUE,
            );
        let mut e12 = vk::PhysicalDeviceVulkan12Features::default()
            .shader_float16(f12.shader_float16 == vk::TRUE)
            .shader_int8(f12.shader_int8 == vk::TRUE)
            .storage_buffer8_bit_access(f12.storage_buffer8_bit_access == vk::TRUE)
            .uniform_and_storage_buffer8_bit_access(
                f12.uniform_and_storage_buffer8_bit_access == vk::TRUE,
            )
            .vulkan_memory_model(f12.vulkan_memory_model == vk::TRUE)
            .vulkan_memory_model_device_scope(f12.vulkan_memory_model_device_scope == vk::TRUE);
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .features(
                vk::PhysicalDeviceFeatures::default()
                    .shader_int16(core.shader_int16 == vk::TRUE)
                    .robust_buffer_access(core.robust_buffer_access == vk::TRUE),
            )
            .push_next(&mut e11)
            .push_next(&mut e12);

        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let device = unsafe {
            instance.create_device(
                physical,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queues)
                    .push_next(&mut features),
                None,
            )
        }
        .map_err(|e| format!("cannot create a device on {name}: {e}"))?;
        let queue = unsafe { device.get_device_queue(family, 0) };
        let memory = unsafe { instance.get_physical_device_memory_properties(physical) };

        Ok(Self {
            _entry: entry,
            _instance: instance,
            device,
            queue,
            family,
            memory,
            name,
            period: props.limits.timestamp_period,
            subgroup: sub.subgroup_size,
        })
    }

    fn memory_type(&self, bits: u32, want: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.memory.memory_type_count).find(|i| {
            bits & (1 << i) != 0
                && self.memory.memory_types[*i as usize]
                    .property_flags
                    .contains(want)
        })
    }

    /// A DEVICE_LOCAL storage buffer of `size` bytes, filled by repeating
    /// `fill` (so an exact-length `fill` is written verbatim, which is what
    /// the page table and the position buffers need).
    fn buffer(&self, size: u64, fill: &[u8]) -> Owned {
        let info = vk::BufferCreateInfo::default()
            .size(size.max(4))
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::TRANSFER_SRC,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { self.device.create_buffer(&info, None) }.expect("create buffer");
        let need = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let index = self
            .memory_type(need.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .expect("a device-local memory type");
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(need.size)
            .memory_type_index(index);
        let memory = unsafe { self.device.allocate_memory(&alloc, None) }.expect("allocate");
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0) }.expect("bind");
        if !fill.is_empty() {
            self.upload(buffer, size.max(4), fill);
        }
        Owned {
            buffer,
            memory,
            size: size.max(4),
        }
    }

    fn upload(&self, dst: vk::Buffer, size: u64, pattern: &[u8]) {
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging = unsafe { self.device.create_buffer(&info, None) }.expect("staging buffer");
        let need = unsafe { self.device.get_buffer_memory_requirements(staging) };
        let want = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let index = self
            .memory_type(need.memory_type_bits, want)
            .expect("a host-visible memory type");
        let memory = unsafe {
            self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(need.size)
                    .memory_type_index(index),
                None,
            )
        }
        .expect("staging memory");
        unsafe { self.device.bind_buffer_memory(staging, memory, 0) }.expect("bind staging");

        unsafe {
            let ptr = self
                .device
                .map_memory(memory, 0, need.size, vk::MemoryMapFlags::empty())
                .expect("map staging") as *mut u8;
            let bytes = std::slice::from_raw_parts_mut(ptr, size as usize);
            for chunk in bytes.chunks_mut(pattern.len()) {
                let n = chunk.len().min(pattern.len());
                chunk[..n].copy_from_slice(&pattern[..n]);
            }
            self.device.unmap_memory(memory);
        }

        self.once(|cmd| unsafe {
            self.device
                .cmd_copy_buffer(cmd, staging, dst, &[vk::BufferCopy::default().size(size)]);
        });

        unsafe {
            self.device.destroy_buffer(staging, None);
            self.device.free_memory(memory, None);
        }
    }

    /// Record, submit and wait on one throwaway command buffer.
    fn once(&self, record: impl FnOnce(vk::CommandBuffer)) {
        unsafe {
            let pool = self
                .device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(self.family),
                    None,
                )
                .expect("command pool");
            let cmd = self
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .expect("command buffer")[0];
            self.device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("begin");
            record(cmd);
            self.device.end_command_buffer(cmd).expect("end");
            let buffers = [cmd];
            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .expect("fence");
            self.device
                .queue_submit(
                    self.queue,
                    &[vk::SubmitInfo::default().command_buffers(&buffers)],
                    fence,
                )
                .expect("submit");
            self.device
                .wait_for_fences(&[fence], true, 60_000_000_000)
                .expect("the submit finished within a minute");
            self.device.destroy_fence(fence, None);
            self.device.destroy_command_pool(pool, None);
        }
    }
}

/// A device buffer and the allocation under it, destroyed by hand.
struct Owned {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
}

impl Owned {
    fn free(self, device: &ash::Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}
/// One entrypoint, built and bound: the six Vulkan objects a dispatch needs.
///
/// A struct rather than six locals because a flash decode measures TWO
/// entrypoints in one shape, and six locals twice over is where a benchmark
/// starts destroying the wrong pipeline.
struct Program {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}

impl Program {
    /// Build `name` at the baseline tier over `operands`, with a push range
    /// of `push_bytes`.
    ///
    /// The descriptor set layout declares one binding per operand IN ORDER,
    /// which is the shader's binding order only because the operand lists
    /// below are written to match it. A module whose unused bindings slangc
    /// dropped still accepts a layout that declares them -- a set may be a
    /// superset of what the shader reads -- which is what lets the split
    /// pass bind a `sinks` at 10 it never touches.
    fn build(bench: &Bench, name: &str, operands: &[&Owned], push_bytes: u32) -> Self {
        let device = &bench.device;
        let path = std::path::Path::new(SPV_DIR.expect("checked by the caller"))
            .join(Capability::Baseline.module(name));
        let code =
            std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let words: Vec<u32> = code
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let bindings: Vec<_> = (0..operands.len() as u32)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .expect("descriptor set layout");
        let set_layouts = [set_layout];

        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(push_bytes)];
        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&ranges),
                None,
            )
        }
        .expect("pipeline layout");
        let module = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
        }
        .expect("shader module");
        let pipeline = unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(module)
                            .name(c"main"),
                    )
                    .layout(layout)],
                None,
            )
        }
        .unwrap_or_else(|(_, e)| panic!("cannot build a pipeline for {name}: {e}"))[0];

        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(operands.len() as u32)];
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )
        }
        .expect("descriptor pool");
        let set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&set_layouts),
            )
        }
        .expect("descriptor set")[0];
        let infos: Vec<_> = operands
            .iter()
            .map(|b| {
                [vk::DescriptorBufferInfo::default()
                    .buffer(b.buffer)
                    .offset(0)
                    .range(b.size)]
            })
            .collect();
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        Self {
            set_layout,
            layout,
            module,
            pipeline,
            pool,
            set,
        }
    }

    /// Bind the pipeline, its set and its push block.
    ///
    /// # Safety
    ///
    /// `cmd` must be recording.
    unsafe fn bind(&self, bench: &Bench, cmd: vk::CommandBuffer, push: &[u8]) {
        let device = &bench.device;
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.set],
                &[],
            );
            device.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::COMPUTE, 0, push);
        }
    }

    fn free(self, device: &ash::Device) {
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_shader_module(self.module, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

// ---------------------------------------------------------------------------
// one shape
// ---------------------------------------------------------------------------

/// One measured point: a norm-then-rotate of `heads` heads for `rows` tokens.
#[derive(Clone, Copy)]
struct Shape {
    heads: usize,
    head_dim: usize,
    rotary: usize,
    rows: usize,
}

impl Shape {
    fn pitch(&self) -> usize {
        self.heads * self.head_dim
    }
}

/// What a measured chain of dispatches costs.
struct Row {
    what: &'static str,
    shape: Shape,
    /// Median microseconds for ONE pass of the whole chain, barrier included.
    micros: f64,
    /// Workgroups the widest dispatch in the chain launches.
    groups: u32,
}

/// A stage in a chain: a built program, its push block, and its grid.
struct Stage<'a> {
    program: &'a Program,
    push: Vec<u8>,
    grid: [u32; 3],
}

/// Time one chain of dispatches, barrier-separated, as a single unit.
///
/// The barrier placement is the whole point of this file, so it is worth
/// being exact: a barrier is issued after EVERY stage including the last.
/// The two-kernel path therefore pays two and the fused path pays one, which
/// is precisely the difference a decode step sees -- the trailing barrier is
/// the one both paths owe whatever comes next, and the interior one is the
/// one the fusion deletes. Charging the trailing barrier to both is what
/// keeps this a comparison of the two paths rather than a comparison of two
/// barrier counts.
///
/// The reported figure is a MEDIAN over `ITERATIONS`, which is the right
/// summary for a number whose tail is other people's work on a shared card.
///
/// # Why each interval covers `REPEATS` passes and not one
///
/// The first run of this file returned 4.10, 6.14, 8.19 and 10.24 us and
/// nothing else -- every figure an exact multiple of 2.048 us, for every
/// shape, with the fused kernel flat at 4.10 whether it launched 8
/// workgroups or 128. That is not a kernel; that is the instrument. This
/// card reports a timestamp period of 1 ns and then writes timestamps that
/// only ever land on a 2048 ns grid, so an interval containing a couple of
/// microseconds of work can only be reported as one tick or two.
///
/// Timing `REPEATS` passes per interval and dividing pushes the grid down to
/// 2048/REPEATS ns, which is what makes the difference between two paths
/// that both cost a few microseconds readable at all. It is the same trick
/// `Order::Overlapped` uses in `qmv_bench.rs` and it has the same cost: the
/// figure becomes a mean over the repeats before it is a median over the
/// iterations.
fn measure(bench: &Bench, what: &'static str, shape: Shape, stages: &[Stage]) -> Row {
    let device = &bench.device;
    let groups = stages
        .iter()
        .map(|s| s.grid[0] * s.grid[1] * s.grid[2])
        .max()
        .expect("a chain has at least one stage");

    let query_pool = unsafe {
        device.create_query_pool(
            &vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(ITERATIONS as u32 + 1),
            None,
        )
    }
    .expect("query pool");

    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);

    bench.once(|cmd| unsafe {
        device.cmd_reset_query_pool(cmd, query_pool, 0, ITERATIONS as u32 + 1);
        let mut pass = |cmd: vk::CommandBuffer| {
            for stage in stages {
                stage.program.bind(bench, cmd, &stage.push);
                device.cmd_dispatch(cmd, stage.grid[0], stage.grid[1], stage.grid[2]);
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
            }
        };
        for _ in 0..WARMUPS {
            pass(cmd);
        }
        device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, query_pool, 0);
        for i in 0..ITERATIONS {
            for _ in 0..REPEATS {
                pass(cmd);
            }
            device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                query_pool,
                i as u32 + 1,
            );
        }
    });

    let mut stamps = vec![0u64; ITERATIONS + 1];
    unsafe {
        device.get_query_pool_results(
            query_pool,
            0,
            &mut stamps,
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
        )
    }
    .expect("timestamps");
    unsafe { device.destroy_query_pool(query_pool, None) };

    let mut deltas: Vec<f64> = stamps
        .windows(2)
        .map(|w| (w[1] - w[0]) as f64 * bench.period as f64 / 1000.0 / REPEATS as f64)
        .collect();
    deltas.sort_by(|a, b| a.partial_cmp(b).expect("no NaN from a subtraction"));

    Row {
        what,
        shape,
        micros: deltas[deltas.len() / 2],
        groups,
    }
}

/// `RmsRopeParams`: the five `RmsParams` fields, then the rotation's four.
fn rms_rope_params(
    axis: usize,
    row_pitch: usize,
    rotary: usize,
    scale: f32,
    base: f32,
) -> Vec<u8> {
    let mut p = rms_params(axis);
    p.extend_from_slice(&(row_pitch as u32).to_le_bytes());
    p.extend_from_slice(&(rotary as u32).to_le_bytes());
    p.extend_from_slice(&scale.to_le_bytes());
    p.extend_from_slice(&base.to_le_bytes());
    p
}

/// The `RmsParams` block, in the order the struct declares it.
fn rms_params(axis: usize) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&1e-5f32.to_le_bytes());
    p.extend_from_slice(&(axis as u32).to_le_bytes());
    p.extend_from_slice(&1u32.to_le_bytes()); // w_stride
    p.extend_from_slice(&0u32.to_le_bytes()); // plus_one
    p.extend_from_slice(&1.0f32.to_le_bytes()); // gain
    p
}

#[test]
#[ignore = "prints a table, proves no property"]
fn the_fused_norm_and_rope_against_the_pair_it_replaces() {
    if let Some(why) = unavailable() {
        eprintln!("skipped: {why}");
        return;
    }
    let bench = match Bench::open() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipped: {e}");
            return;
        }
    };
    eprintln!(
        "device: {} (timestamp {} ns, subgroup {})",
        bench.name, bench.period, bench.subgroup
    );

    // qwen3-0.6B: 16 query heads and 8 KV heads, both 128 wide, both fully
    // rotated. The q fire and the k fire are measured separately because they
    // differ only in workgroup count, and whether the fused form's idle
    // threads matter is exactly a question about workgroup count.
    // 64 and 512 rows are CONTROLS, not decode shapes. The first run of this
    // sweep put every decode figure on an exact 2.048 us grid and left the
    // fused kernel flat at 4.10 us from 8 workgroups to 128, which is the
    // signature of a measurement that is reporting a floor rather than a
    // cost. Timing 32 passes per interval did not move it, so the floor is
    // the dispatch's and not the instrument's. These two rows are here to
    // make the sweep able to leave that floor: at 512 rows the fused kernel
    // launches 8192 workgroups and must cost real time, and if THOSE figures
    // are also exact multiples of 2.048 then the grid is an artefact of the
    // harness and every conclusion below is suspect.
    let shapes: Vec<Shape> = [1usize, 2, 4, 8, 64, 512]
        .iter()
        .flat_map(|&rows| {
            [16usize, 8].into_iter().map(move |heads| Shape {
                heads,
                head_dim: 128,
                rotary: 128,
                rows,
            })
        })
        .collect();

    let mut rows = Vec::new();
    for shape in shapes {
        let pitch = shape.pitch();
        let bytes = (shape.rows * pitch * 2) as u64;
        let x = bench.buffer(bytes, &0x3f80_3f80u32.to_le_bytes());
        let out = bench.buffer(bytes, &[]);
        let w = bench.buffer((shape.head_dim * 2) as u64, &0x3f80_3f80u32.to_le_bytes());
        let params = bench.buffer(20, &rms_params(shape.head_dim));
        let fused_params = bench.buffer(
            36,
            &rms_rope_params(shape.head_dim, pitch, shape.rotary, 1.0, (10000.0f32).log2()),
        );
        let position = bench.buffer((shape.rows * 4) as u64, &7i32.to_le_bytes());

        let rms = Program::build(
            &bench,
            "rms_strided_head_row_bfloat16",
            &[&x, &w, &out, &params],
            4,
        );
        let neox = Program::build(&bench, "neox_mb_bfloat16", &[&out, &position], 12);
        let fused = Program::build(
            &bench,
            "rms_rope_bfloat16",
            &[&x, &w, &fused_params, &position],
            0,
        );

        let base = (10000.0f32).log2();
        let mut rope_push = Vec::new();
        rope_push.extend_from_slice(&1.0f32.to_le_bytes());
        rope_push.extend_from_slice(&base.to_le_bytes());
        rope_push.extend_from_slice(&(shape.head_dim as i32).to_le_bytes());

        rows.push(measure(
            &bench,
            "rms + neox",
            shape,
            &[
                Stage {
                    program: &rms,
                    push: (pitch as i32).to_le_bytes().to_vec(),
                    grid: [1, shape.heads as u32, shape.rows as u32],
                },
                Stage {
                    program: &neox,
                    push: rope_push,
                    grid: [
                        (shape.rotary / 2) as u32,
                        shape.heads as u32,
                        shape.rows as u32,
                    ],
                },
            ],
        ));
        rows.push(measure(
            &bench,
            "rms_rope",
            shape,
            &[Stage {
                program: &fused,
                push: Vec::new(),
                grid: [1, shape.heads as u32, shape.rows as u32],
            }],
        ));

        fused.free(&bench.device);
        fused_params.free(&bench.device);
        neox.free(&bench.device);
        rms.free(&bench.device);
        position.free(&bench.device);
        params.free(&bench.device);
        w.free(&bench.device);
        out.free(&bench.device);
        x.free(&bench.device);
    }

    eprintln!();
    eprintln!("  {:<12} {:>5} {:>5} {:>8} {:>8}", "what", "heads", "rows", "groups", "us");
    for row in &rows {
        eprintln!(
            "  {:<12} {:>5} {:>5} {:>8} {:>8.2}",
            row.what, row.shape.heads, row.shape.rows, row.groups, row.micros
        );
    }
    eprintln!();
    for pair in rows.chunks(2) {
        let (a, b) = (&pair[0], &pair[1]);
        eprintln!(
            "  heads {:>2} rows {}: fused is {:+.2} us ({:+.1}%)",
            a.shape.heads,
            a.shape.rows,
            b.micros - a.micros,
            100.0 * (b.micros - a.micros) / a.micros
        );
    }
}
