//! What `sdpa_paged_decode` costs per dispatch, against history length.
//!
//! `tests/qmv_bench.rs` did this for the decode matvec and its method is
//! copied here wholesale -- device timestamps, one recorded command buffer per
//! shape, warm-ups then a median over many barrier-separated dispatches,
//! device-local buffers so the traffic is the card's memory and not PCIe. Read
//! that file's header for why each of those is so; only what is DIFFERENT is
//! recorded below.
//!
//! # What this file is for
//!
//! At 384 tokens of context a qwen3-0.6B decode step spends about three
//! quarters of its device time in this one kernel, and the arithmetic says it
//! should not: 384 positions x 8 KV heads x 128 dims x 2 planes x 2 bytes is
//! 1.57 MB per layer, which at the ~1000 GB/s this card was measured at is
//! 1.6 us. Anything above that is not bandwidth. This file prices the gap
//! directly and, by sweeping the ROW count as well as the history, says
//! whether what is left after a change is bandwidth or occupancy.
//!
//! # How to run it
//!
//! ```sh
//! PIE_SLANGC=$HOME/.local/share/slang-2026.14.1/bin/slangc VK_LAYER_PATH= \
//!   cargo test -p kernels-vulkan --features native --test sdpa_bench \
//!   -- --ignored --nocapture
//! ```
//!
//! `#[ignore]`, for the same reason the qmv sweep is: it prints a table and
//! proves no property.
//!
//! # The shape measured
//!
//! qwen3-0.6B's attention: 8 KV heads, GQA 2, so 16 query heads, head width
//! 128, page size 32. One decode row is therefore 16 workgroups of
//! `PIE_HEAD_DIM` threads, which is the number the occupancy column is about
//! -- a 4090 has 128 SMs, so a single-row decode leaves seven eighths of the
//! card with nothing to do, and only a sweep over rows can say whether that is
//! what the remaining microseconds are.
//!
//! ## The bytes column counts UNIQUE bytes
//!
//! The two query heads that share a KV head read the same key and value
//! planes, so a dispatch issues twice the loads this counts. The figure to
//! compare against the card's bandwidth is the unique one -- the second read
//! is an L2 hit by construction -- and it is also the figure the diagnosis in
//! `.wiki/kernel-x/vulkan-refactor.md` §15 quotes, so the two are directly
//! comparable.

use ash::vk;
use kernels_vulkan::Capability;

/// Where a `native` build left the modules, or `None` if this is not one.
const SPV_DIR: Option<&str> = option_env!("PIE_KERNELS_VULKAN_SPV_DIR");

/// Timed dispatches per shape.
const ITERATIONS: usize = 128;

/// Untimed dispatches first.
const WARMUPS: usize = 16;

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

// ---------------------------------------------------------------------------
// one shape
// ---------------------------------------------------------------------------

/// One measured point.
#[derive(Clone, Copy)]
struct Shape {
    head_dim: usize,
    /// KV heads; query heads are `kv_heads * gqa`.
    kv_heads: usize,
    gqa: usize,
    /// Decode rows in the dispatch -- one per sequence in the batch.
    rows: usize,
    /// Keys attended per row, including the query's own position.
    history: usize,
    /// How many ways the key range is split across workgroups.
    ///
    /// One is the single-pass `sdpa_paged_decode`, unchanged. Anything above
    /// one is the flash decode: a `split` dispatch of `q_heads * rows *
    /// splits` workgroups leaving unnormalised partials, then a `combine`
    /// dispatch of `q_heads * rows` folding them.
    splits: usize,
}

impl Shape {
    fn q_heads(&self) -> usize {
        self.kv_heads * self.gqa
    }
    /// Workgroups the WIDEST dispatch of this configuration launches, which
    /// is the split pass when there is one.
    fn groups(&self) -> u32 {
        (self.q_heads() * self.rows * self.splits.max(1)) as u32
    }
    /// Unique KV bytes a dispatch must read: keys and values, for every row's
    /// own history, at bf16.
    fn bytes(&self) -> u64 {
        (self.rows * self.history * self.kv_heads * self.head_dim * 2 * 2) as u64
    }
    fn entrypoint(&self) -> String {
        let stem = match self.splits {
            0 | 1 => "sdpa_paged_decode_bfloat16_d",
            _ => "sdpa_paged_decode_split_bfloat16_d",
        };
        assert!(
            matches!(self.head_dim, 64 | 128 | 256 | 512),
            "no paged decode module at head width {}",
            self.head_dim
        );
        format!("{stem}_{}", self.head_dim)
    }

    fn combine_entrypoint(&self) -> String {
        format!("sdpa_paged_decode_combine_bfloat16_d_{}", self.head_dim)
    }

    /// Floats the partial buffer holds: `splits * rows * heads` accumulators
    /// of `head_dim`, then the same many `(max, sum_exp)` pairs.
    fn partial_floats(&self) -> usize {
        self.splits * self.rows * self.q_heads() * (self.head_dim + 2)
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

struct Row {
    what: &'static str,
    shape: Shape,
    micros: f64,
    wall_micros: f64,
    gbps: f64,
}

/// The page size the driver uses for qwen-class checkpoints.
const PAGE: usize = 32;

fn measure(bench: &Bench, what: &'static str, shape: Shape) -> Row {
    let device = &bench.device;
    let Shape {
        head_dim,
        kv_heads,
        gqa,
        rows,
        history,
        splits,
    } = shape;
    let q_heads = shape.q_heads();
    let split = splits > 1;

    // One request per row, each with its own pages, handed out in the order
    // an allocator that never freed anything would: physical page `i` for
    // logical page `i`. A fragmented table would be slower and this is
    // therefore the optimistic end of the range.
    let pages_per = history.div_ceil(PAGE);
    let total_pages = pages_per * rows;
    let indices: Vec<u32> = (0..total_pages as u32).collect();
    let indptr: Vec<u32> = (0..=rows as u32).map(|r| r * pages_per as u32).collect();
    let slots = total_pages * PAGE;
    let kv_elems = slots * kv_heads * head_dim;

    // bf16 1.0 for the queries, bf16 0.0078125 for the planes: the content
    // cannot change the timing, but scores that overflow to infinity would
    // make the online update's `exp` do something unrepresentative.
    let q = bench.buffer(
        (rows * q_heads * head_dim * 2) as u64,
        &0x3f80_3f80u32.to_le_bytes(),
    );
    let k = bench.buffer((kv_elems * 2) as u64, &0x3c00_3c00u32.to_le_bytes());
    let v = bench.buffer((kv_elems * 2) as u64, &0x3c00_3c00u32.to_le_bytes());
    let out = bench.buffer((rows * q_heads * head_dim * 2) as u64, &[]);
    let positions: Vec<u8> = (0..rows)
        .flat_map(|_| (history as i32 - 1).to_le_bytes())
        .collect();
    let reqs: Vec<u8> = (0..rows as i32).flat_map(i32::to_le_bytes).collect();
    let pos = bench.buffer(positions.len() as u64, &positions);
    let req = bench.buffer(reqs.len() as u64, &reqs);
    let page_indices = bench.buffer(
        (indices.len() * 4) as u64,
        &indices
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    let page_indptr = bench.buffer(
        (indptr.len() * 4) as u64,
        &indptr
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    let mask = bench.buffer(rows as u64, &vec![0u8; rows.max(4)]);
    let mask_on = bench.buffer(rows as u64, &vec![0u8; rows.max(4)]);
    let sinks = bench.buffer((q_heads * 2) as u64, &[0u8, 0u8]);
    // The split pass's scratch. Never read before it is written -- every
    // workgroup of the split grid writes its whole entry -- so it is left
    // uninitialised, which is also what the driver does with it.
    let partials = bench.buffer((shape.partial_floats().max(1) * 4) as u64, &[]);
    let mut operands = vec![
        &q,
        &k,
        &v,
        &out,
        &pos,
        &req,
        &page_indices,
        &page_indptr,
        &mask,
        &mask_on,
        &sinks,
    ];
    if split {
        operands.push(&partials);
    }

    // gqa_factor, page_size, n_kv_heads, scale, attention_mask_stride, window.
    let mut push = Vec::new();
    push.extend_from_slice(&(gqa as i32).to_le_bytes());
    push.extend_from_slice(&(PAGE as i32).to_le_bytes());
    push.extend_from_slice(&(kv_heads as i32).to_le_bytes());
    push.extend_from_slice(&(1.0f32 / (head_dim as f32).sqrt()).to_le_bytes());
    push.extend_from_slice(&0u32.to_le_bytes());
    push.extend_from_slice(&0i32.to_le_bytes());

    let first = Program::build(bench, &shape.entrypoint(), &operands, push.len() as u32);
    // The fold's own three, in the order `attn::sdpa_paged_decode` takes
    // them: the output it writes, the sinks it does not (this bench measures
    // the sinkless module, so binding 1 is a hole) and the partials it reads.
    // Its push block is one word, the split count the grid cannot carry.
    let fold_push = (splits as i32).to_le_bytes().to_vec();
    let fold = split.then(|| {
        Program::build(
            bench,
            &shape.combine_entrypoint(),
            &[&out, &sinks, &partials],
            fold_push.len() as u32,
        )
    });

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
    let grid = [q_heads as u32, rows as u32, splits.max(1) as u32];

    let started = std::time::Instant::now();
    bench.once(|cmd| unsafe {
        device.cmd_reset_query_pool(cmd, query_pool, 0, ITERATIONS as u32 + 1);
        let flush = |cmd| {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        };
        // One "fire" is a whole attention, which for a flash decode is TWO
        // dispatches with a barrier between them -- so the number printed is
        // directly comparable with the single-pass one.
        let fire = |cmd| {
            first.bind(bench, cmd, &push);
            device.cmd_dispatch(cmd, grid[0], grid[1], grid[2]);
            if let Some(fold) = &fold {
                flush(cmd);
                fold.bind(bench, cmd, &fold_push);
                device.cmd_dispatch(cmd, grid[0], grid[1], 1);
            }
            flush(cmd);
        };
        for _ in 0..WARMUPS {
            fire(cmd);
        }
        device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, query_pool, 0);
        for i in 0..ITERATIONS {
            fire(cmd);
            device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                query_pool,
                i as u32 + 1,
            );
        }
    });
    let wall = started.elapsed();

    let mut ticks = vec![0u64; ITERATIONS + 1];
    unsafe {
        device.get_query_pool_results(
            query_pool,
            0,
            &mut ticks,
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
        )
    }
    .expect("timestamps");
    let mut micros: Vec<f64> = ticks
        .windows(2)
        .map(|w| (w[1].saturating_sub(w[0]) as f64) * f64::from(bench.period) / 1000.0)
        .collect();
    micros.sort_by(f64::total_cmp);
    let median = micros[micros.len() / 2];

    unsafe { device.destroy_query_pool(query_pool, None) };
    first.free(device);
    if let Some(fold) = fold {
        fold.free(device);
    }
    drop(operands);
    for b in [
        q,
        k,
        v,
        out,
        pos,
        req,
        page_indices,
        page_indptr,
        mask,
        mask_on,
        sinks,
        partials,
    ] {
        b.free(device);
    }

    Row {
        what,
        shape,
        micros: median,
        wall_micros: wall.as_secs_f64() * 1.0e6 / (ITERATIONS + WARMUPS) as f64,
        gbps: shape.bytes() as f64 / (median * 1000.0),
    }
}

fn print(rows: &[Row]) {
    eprintln!(
        "{:<20} {:>4} {:>5} {:>7} {:>7} {:>7} {:>8} {:>10} {:>9}",
        "purpose", "d", "rows", "history", "splits", "groups", "KV KiB", "us/attn", "GB/s"
    );
    for r in rows {
        eprintln!(
            "{:<20} {:>4} {:>5} {:>7} {:>7} {:>7} {:>8.0} {:>10.2} {:>9.1}",
            r.what,
            r.shape.head_dim,
            r.shape.rows,
            r.shape.history,
            r.shape.splits.max(1),
            r.shape.groups(),
            r.shape.bytes() as f64 / 1024.0,
            r.micros,
            r.gbps,
        );
    }
}

// ---------------------------------------------------------------------------
// the sweep
// ---------------------------------------------------------------------------

/// `sdpa_paged_decode` against history, rows and head width.
///
/// Three sub-sweeps, and each answers a different question:
///
///   - **history** at one row is the number the decode step pays. If it grows
///     linearly with a slope far above what the bytes cost, the loop body is
///     the problem; if it is FLAT at the short end, there is a per-dispatch
///     floor no loop change can remove.
///   - **rows** holds the history at 384 and multiplies the workgroup count.
///     A single decode row is 16 workgroups on a 128-SM card. If sixteen rows
///     cost the same as one, the kernel was occupancy-starved and splitting
///     the KV range across workgroups (flash-decoding) is the next move; if
///     the cost rises in proportion, the card was already busy and it is not.
///   - **width** checks the other three instantiations, which is where a
///     subgroup-shaped inner loop could go wrong: 64 is narrower than two
///     subgroups on this card and 512 is sixteen of them.
#[test]
#[ignore = "a measurement, not a property: prints a table and takes tens of seconds"]
fn sdpa_paged_decode_cost_against_history() {
    if let Some(why) = unavailable() {
        eprintln!("SKIP: {why}");
        return;
    }
    let bench = match Bench::open() {
        Ok(b) => b,
        Err(why) => {
            eprintln!("SKIP: {why}");
            return;
        }
    };
    eprintln!(
        "device: {} ({} ns per tick, subgroup {})\n\
         {ITERATIONS} timed dispatches per shape after {WARMUPS} warm-ups, one submit, \
         median of the per-dispatch timestamp intervals.\n\
         The instrument perturbs: each interval carries one bottom-of-pipe timestamp \
         write between two barriers that were there anyway, and the wall column below \
         is the cross-check.\n",
        bench.name, bench.period, bench.subgroup
    );

    let qwen = |rows: usize, history: usize| Shape {
        head_dim: 128,
        kv_heads: 8,
        gqa: 2,
        rows,
        history,
        splits: 1,
    };
    let split = |rows: usize, history: usize, splits: usize| Shape {
        splits,
        ..qwen(rows, history)
    };

    let mut rows: Vec<Row> = Vec::new();
    for h in [24usize, 128, 384, 1024] {
        rows.push(measure(&bench, "qwen3-0.6b decode", qwen(1, h)));
    }
    for r in [1usize, 2, 4, 8, 16, 32] {
        rows.push(measure(&bench, "rows at 384", qwen(r, 384)));
    }
    for d in [64usize, 128, 256, 512] {
        rows.push(measure(
            &bench,
            "width at 384",
            Shape {
                head_dim: d,
                kv_heads: 8,
                gqa: 2,
                rows: 1,
                history: 384,
                splits: 1,
            },
        ));
    }
    // The flash decode, against the same four histories and over the split
    // counts either side of the one a 128-SM card wants. `S = 1` is not
    // measured here because it IS the single-pass row above -- the driver
    // falls back to `sdpa_paged_decode` rather than launching a fold over one
    // partial.
    for h in [24usize, 128, 384, 1024] {
        for s in [2usize, 4, 8, 16, 32, 64] {
            rows.push(measure(&bench, "flash decode", split(1, h, s)));
        }
    }
    for r in [1usize, 2, 4, 8, 16, 32] {
        rows.push(measure(&bench, "flash rows at 384", split(r, 384, 8)));
    }
    for (r, s) in [(16usize, 8usize), (32, 4), (32, 2)] {
        rows.push(measure(&bench, "flash batch rule", split(r, 384, s)));
    }
    for d in [64usize, 128, 256, 512] {
        rows.push(measure(
            &bench,
            "flash width at 384",
            Shape {
                head_dim: d,
                kv_heads: 8,
                gqa: 2,
                rows: 1,
                history: 384,
                splits: 8,
            },
        ));
    }
    print(&rows);

    eprintln!("\nwall-clock cross-check (whole submit / dispatches, carries submit overhead):");
    for r in &rows {
        eprintln!(
            "  {:<16} d={:<4} rows={:<3} history={:<5} timestamps {:>8.2} us   wall {:>8.2} us",
            r.what, r.shape.head_dim, r.shape.rows, r.shape.history, r.micros, r.wall_micros
        );
    }

    assert!(
        rows.iter().all(|r| r.micros > 0.0),
        "a dispatch that took no measurable time means the timestamps did not work",
    );
}
