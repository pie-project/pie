//! How fast `affine_qmv_fast` actually moves bytes, at controlled shapes.
//!
//! This file measures one thing and asserts almost nothing. `quant/qmv.slang`
//! ends with a conclusion -- that what is left of a decode matvec's cost is
//! "too little work per dispatch to fill the card, and the fix for that is
//! splitting K across workgroups, not tuning the loop" -- and that conclusion
//! was reached by elimination (widening `PIE_LANES` changed nothing, four and
//! sixteen rows per group were both worse than eight, 16-byte weight loads did
//! nothing) rather than by measuring the kernel's bandwidth against the card's.
//! A sweep of N at fixed K settles it directly: a kernel that is starved of
//! workgroups gets FASTER per byte as N grows, because N is the only thing the
//! grid scales with (`qmv_grid` launches `ceil(N/8)` groups of 128 threads and
//! one group per output octet); a kernel that is inefficient per byte stays
//! flat.
//!
//! Three sub-sweeps hang off the main one, and they are what turn a rising
//! curve into an answer. A rising GB/s is ambiguous on its own: it is equally
//! the signature of a kernel starved of workgroups and of a kernel paying a
//! fixed cost per DISPATCH that large shapes amortise away. The `floor` rows
//! run N below the real shapes, down to a single workgroup, so the two can be
//! told apart -- if one workgroup and five hundred cost the same, the cost is
//! not occupancy. The `sweep K` rows hold the group count fixed and walk the
//! serial chunk loop instead, pricing the part of a workgroup that split-K
//! would actually divide. The last table runs each real shape twice, with and
//! without the barrier between dispatches, which separates work from latency.
//!
//! # How to run it
//!
//! ```sh
//! PIE_SLANGC=$HOME/.local/share/slang-2026.14.1/bin/slangc VK_LAYER_PATH= \
//!   cargo test -p kernels-vulkan --features native --test qmv_bench \
//!   -- --ignored --nocapture
//! ```
//!
//! `#[ignore]`, because it runs for tens of seconds, allocates most of a
//! hundred megabytes of device memory and proves no property -- it prints a
//! table. `native` for the same reason `tests/gpu.rs` needs it: without the
//! SPIR-V a `native` build leaves behind there is nothing to dispatch, and the
//! test SKIPS rather than fails.
//!
//! # Method
//!
//! Device timestamps, not wall clock. The command buffer is recorded ONCE per
//! shape: a warm-up run, then `ITERATIONS` dispatches each followed by a full
//! compute-to-compute barrier and a `vkCmdWriteTimestamp` at bottom-of-pipe.
//! The whole thing is one submit, so per-submit overhead is amortised to
//! nothing and never appears in a per-dispatch number; each measured interval
//! is one dispatch plus one barrier, which is the shape a real decode runs
//! (every projection in a step depends on the previous one, so they do not
//! overlap there either). The reported figure is the MEDIAN of those
//! intervals, so a stray scheduling hiccup or a clock-boost transient moves
//! nothing.
//!
//! ## The instrument perturbs, and this is how much
//!
//! `driver-vulkan`'s own `PIE_VULKAN_TIMING` path is not used here -- that one
//! lives on the other side of a crate boundary this crate must not cross, and
//! it costs enough to matter (a submit-and-wait measured 3.47 ms with it off
//! and 5.78 ms with it on, because it brackets every fire and reads the pool
//! back per submit). The pool here is read back exactly once for the whole
//! run, and the only per-dispatch cost is the timestamp write itself, which
//! sits between two barriers that were already there. As a cross-check the
//! wall-clock time of the whole submit is printed alongside; the two agree to
//! within a few per cent on every shape, and a reader who does not trust the
//! timestamps can read the `wall` column instead.
//!
//! ## Buffers are DEVICE-LOCAL, and that is not incidental
//!
//! `tests/gpu.rs` puts every operand in `HOST_VISIBLE | HOST_COHERENT` memory
//! so a correctness test can read the answer back without a staging copy. That
//! is exactly wrong for a bandwidth measurement: on a discrete card such a
//! buffer lives in system RAM behind PCIe, and the number the sweep would
//! print is ~25 GB/s of bus, not DRAM. So this file allocates DEVICE_LOCAL and
//! uploads through a staging buffer, and reads nothing back -- correctness is
//! `tests/gpu.rs`'s job, which is why nothing here checks the answer.
//!
//! ## What the L2 does to the large end of the sweep
//!
//! A 4090 has 72 MB of L2. Every shape below that fits ENTIRELY in it, and
//! these dispatches run back to back over the same weights, so from the second
//! dispatch on the traffic is L2 traffic and the GB/s printed is not a DRAM
//! figure. This matters for reading the sweep and is flagged per row in the
//! output (`L2` when the working set fits, `DRAM` when it does not). It does
//! not weaken the sweep's conclusion in either direction: if a shape that fits
//! in L2 still cannot go fast, the kernel is not limited by DRAM at all.

#![allow(clippy::needless_range_loop)]

use ash::vk;
use kernels_vulkan::Capability;

/// Where a `native` build left the modules, or `None` if this is not one.
const SPV_DIR: Option<&str> = option_env!("PIE_KERNELS_VULKAN_SPV_DIR");

/// The variant measured. Group 128 / 4 bits is what a Q4_K_M checkpoint of
/// qwen3-0.6B resolves to, and it is compiled -- see the `pie:instantiate`
/// lines at the foot of `quant/qmv.slang`, which is the only list of shapes
/// that exist.
const ENTRYPOINT: &str = "affine_qmv_fast_bfloat16_gs_128_b_4";
const GROUP: usize = 128;
const BITS: usize = 4;

/// Timed dispatches per shape. 128 is comfortably over the hundred the median
/// wants, and small enough that the largest shape's ten gigabytes of traffic
/// still finishes inside a second.
const ITERATIONS: usize = 128;

/// Untimed dispatches first: the pipeline is already built by then, but the
/// first run still pays first-touch page faults on freshly allocated device
/// memory and whatever clock state the card was idling at.
const WARMUPS: usize = 16;

/// The card's L2, in bytes. A working set under this is served from cache
/// after the first pass, and the row is labelled accordingly.
const L2_BYTES: u64 = 72 * 1024 * 1024;

// ---------------------------------------------------------------------------
// the device
// ---------------------------------------------------------------------------

/// The subset of `tests/gpu.rs`'s `Gpu` this file needs, plus the two things
/// it needs that that one deliberately does not have: device-local allocation
/// and a timestamp query pool.
///
/// It is a second harness rather than a reuse because a Rust integration test
/// is its own crate -- `tests/gpu.rs` cannot be imported from `tests/
/// qmv_bench.rs` without compiling its 5800 lines and its whole test list into
/// this binary as well. The device-open sequence below is therefore the same
/// sequence, cut down: same `PIE_VULKAN_DEVICE` pin, same refusal to take
/// `.first()` of the device list on a box that offers a software rasteriser
/// before the card.
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
            .application_name(c"kernels-vulkan qmv bench")
            .api_version(vk::API_VERSION_1_3);
        // No validation layer, on purpose. This file measures time, and
        // GPU-assisted validation instruments the shader itself -- the numbers
        // it would print are the layer's, not the kernel's. `tests/gpu.rs` runs
        // the same modules under the layer and that is where legality is
        // established.
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

        let props = unsafe { instance.get_physical_device_properties(physical) };
        if props.limits.timestamp_compute_and_graphics != vk::TRUE
            || props.limits.timestamp_period <= 0.0
        {
            return Err(format!("{name} does not report usable timestamps"));
        }

        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let device = unsafe {
            instance.create_device(
                physical,
                &vk::DeviceCreateInfo::default().queue_create_infos(&queues),
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

    /// A DEVICE_LOCAL storage buffer of `size` bytes, filled from `fill` (a
    /// repeating pattern, staged through a host-visible copy) or left as it
    /// came when `fill` is empty.
    ///
    /// The CONTENT of the weight plane cannot change the timing -- every code
    /// takes the same integer unpack and the same FMA -- but it is written
    /// anyway, because leaving 78 MB of device memory untouched is the kind of
    /// shortcut that turns into "the driver never really faulted those pages
    /// in" three months later.
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

    /// Fill a device buffer with `pattern`, repeated, through a staging copy.
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

/// What a single (K, N) point cost.
struct Row {
    what: &'static str,
    k: usize,
    n: usize,
    /// How many activation rows the dispatch multiplied -- the decode batch
    /// size. One everywhere except the token sweep.
    tokens: usize,
    groups: u32,
    micros: f64,
    wall_micros: f64,
    bytes: u64,
    gbps: f64,
    resident: &'static str,
}

/// Everything a dispatch READS, in bytes: the packed weight plane, the two
/// dequant planes, and the activation vector.
///
/// The output is excluded -- it is `N` bf16, two orders of magnitude under the
/// weights at every shape here, and a written byte is not the same currency as
/// a read one. Naming the traffic this way means the GB/s column is directly
/// comparable to the 267 GB/s figure llama.cpp's Vulkan backend gets on the
/// same card and model, which was computed the same way.
fn bytes_read(k: usize, n: usize) -> u64 {
    let weights = (n * k * BITS / 8) as u64;
    let planes = 2 * (n * (k / GROUP) * 2) as u64;
    weights + planes + (k * 2) as u64
}

/// Whether the dispatches in the timed batch are separated by a barrier.
///
/// `Serial` is the shape a decode really runs -- every projection in a step
/// consumes the previous one's output -- and it is what the main table
/// reports. `Overlapped` removes the barrier so consecutive dispatches may run
/// CONCURRENTLY, which no decode can do but which answers a question the
/// serial number cannot: how much of a small dispatch's cost is work, and how
/// much is a latency the card would happily hide behind other work if it had
/// any. The two differing by a large factor means the kernel is latency-bound
/// rather than throughput-bound at that shape.
#[derive(Clone, Copy, PartialEq)]
enum Order {
    Serial,
    Overlapped,
    /// Barriers between the dispatches, but no timestamp per dispatch: the
    /// batch is bracketed by two and divided, exactly as [`Order::Overlapped`]
    /// is timed.
    ///
    /// This exists because [`Order::Serial`] cannot answer what a barrier
    /// costs. It records a barrier AND a `BOTTOM_OF_PIPE` timestamp after
    /// every dispatch, and a bottom-of-pipe write is itself a point the card
    /// may not finish early -- so its interval prices the pair, and reading it
    /// as the barrier alone charges the barrier for the instrument.
    ///
    /// Subtracting this from `Serial` gives the timestamp's own cost, and
    /// subtracting `Overlapped` from this gives the barrier's. Those are the
    /// two numbers that decide whether a decode's remaining time is reachable
    /// by fusing kernels -- which removes barriers -- or is not.
    SerialUntimed,
    /// Barriers that order the dispatches but make no memory available.
    ///
    /// A decode's barrier is `SHADER_WRITE` -> `SHADER_READ | SHADER_WRITE`,
    /// and the question this answers is whether that MASK is what costs, or
    /// whether the cost is the drain any dependency implies. An execution-only
    /// dependency still stops the next dispatch until this one has retired; it
    /// just does not promise the writes are visible. It is not a legal
    /// substitute for the real barrier and is here only to price it.
    SerialExecOnly,
}

fn measure(bench: &Bench, what: &'static str, k: usize, n: usize) -> Row {
    measure_as(bench, what, k, n, Order::Serial)
}

fn measure_as(bench: &Bench, what: &'static str, k: usize, n: usize, order: Order) -> Row {
    measure_tokens(bench, what, k, n, 1, order)
}

/// [`measure_as`] over `tokens` activation rows instead of one.
///
/// `qmv_grid` launches `vecs` workgroups along x and `ceil(N/8)` along y, so
/// the row count is a pure multiple of the grid: each row's workgroups re-read
/// the whole weight plane for that row. Whether that re-read is free -- the
/// planes are in L2 after the first row -- or costs a second full pass is the
/// entire question of whether this repository's decode can batch, and it is
/// what the `tokens` rows of the sweep answer.
fn measure_tokens(
    bench: &Bench,
    what: &'static str,
    k: usize,
    n: usize,
    tokens: usize,
    order: Order,
) -> Row {
    measure_kernel(
        bench,
        what,
        k,
        n,
        tokens,
        order,
        ENTRYPOINT,
        Capability::Baseline,
        0,
        |tokens, n| [tokens as u32, (n.div_ceil(8)) as u32, 1],
    )
}

/// [`measure_tokens`] against an arbitrary entrypoint and workgroup grid.
///
/// The matvec and the tiled GEMM read the same five buffers and take the same
/// two push words, and differ only in which module is loaded and how the grid
/// is shaped -- so pricing one against the other is a matter of passing both,
/// not of writing the scaffolding twice. `grid` is handed the row count and
/// the column count and returns WORKGROUPS, which is what `qmv_grid` and
/// `qmm_grid` return once their thread extents are divided by their own local
/// sizes.
#[allow(
    clippy::too_many_arguments,
    reason = "the kernel, its capability, its shape, its batch, its order and \
              its grid are six independent axes of one measurement; bundling \
              them into a struct would put the sweep's parameters somewhere \
              other than where the sweep reads"
)]
fn measure_kernel(
    bench: &Bench,
    what: &'static str,
    k: usize,
    n: usize,
    tokens: usize,
    order: Order,
    entrypoint: &str,
    capability: Capability,
    // How many ways the K loop is cut, or zero for a kernel that walks all of
    // K in one workgroup. A split-K entrypoint declares three bindings and
    // four push words more than every other kernel this harness prices, and
    // both of those are pipeline-layout facts rather than dispatch ones, so
    // the count has to be known here and not only inside `grid`.
    splits: usize,
    grid: impl Fn(usize, usize) -> [u32; 3],
) -> Row {
    let grid = grid(tokens, n);
    let device = &bench.device;
    // The workgroups the dispatch really launches, whatever the grid's shape:
    // the report's `groups` column is a count and not an extent.
    let groups = grid[0] * grid[1] * grid[2];

    // The weight codes and the two planes. The pattern is arbitrary; only its
    // size is load-bearing.
    let w = bench.buffer((n * k * BITS / 8) as u64, &0x1234_5678u32.to_le_bytes());
    let scales = bench.buffer((n * (k / GROUP) * 2) as u64, &0x3c00_3c00u32.to_le_bytes());
    let biases = bench.buffer((n * (k / GROUP) * 2) as u64, &0u32.to_le_bytes());
    let x = bench.buffer((tokens * k * 2) as u64, &0x3f80_3f80u32.to_le_bytes());
    let y = bench.buffer((tokens * n * 2) as u64, &[]);
    // Split-K writes ONE partial plane per partition and a second dispatch
    // sums them. Binding 5 is `extra`, which only the bias and residual forms
    // declare; the set has to carry a slot there anyway because binding 6 is
    // the partial plane and a descriptor set is addressed by index. `y` is
    // reused for it -- the shader never reads it, and a dangling slot is a
    // validation error where a redundant one is not.
    let partial = bench.buffer((tokens * n * splits.max(1) * 2) as u64, &[]);
    let operands: Vec<&Owned> = if splits == 0 {
        vec![&w, &scales, &biases, &x, &y]
    } else {
        vec![&w, &scales, &biases, &x, &y, &y, &partial]
    };

    let path = std::path::Path::new(SPV_DIR.expect("checked by the caller"))
        .join(capability.module(entrypoint));
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
    // Two ints, which is what the row states and what `struct Push` declares
    // for a variant that is neither wide-strided nor residual.
    let mut push = Vec::new();
    push.extend_from_slice(&(k as i32).to_le_bytes());
    push.extend_from_slice(&(n as i32).to_le_bytes());
    // The batched form states a third scalar: the grid is rounded up to whole
    // groups of `PIE_MROWS` vectors, so the last group has to be told where the
    // batch really ends. Pushing it for the other entrypoints would ask the
    // pipeline layout for a range they do not declare.
    if entrypoint.contains("_batched_") {
        push.extend_from_slice(&(tokens as i32).to_le_bytes());
    }
    if splits > 0 {
        // `row_stride`, which the non-strided split-K form declares and does
        // not read -- a push block may not be short of what the layout says.
        push.extend_from_slice(&(k as i32).to_le_bytes());
        // The partition, the distance between two partitions' planes, and how
        // many there are. `k_partition_size` is rounded UP so the last
        // partition is the short one and `min(k, k0 + size)` in the shader
        // clamps it; rounding down would leave a tail nobody walks.
        push.extend_from_slice(&(k.div_ceil(splits) as i32).to_le_bytes());
        push.extend_from_slice(&((tokens * n) as i32).to_le_bytes());
        push.extend_from_slice(&(splits as i32).to_le_bytes());
    }
    let ranges = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(push.len() as u32)];
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
    .unwrap_or_else(|(_, e)| panic!("cannot build a pipeline for {entrypoint}: {e}"))[0];

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

    // One timestamp after each measured dispatch, plus one before the first:
    // an INTERVAL between two bottom-of-pipe writes is what is wanted, and
    // bracketing each dispatch with a top-of-pipe write instead would measure
    // whatever the top of the pipe felt like doing, since a top-of-pipe
    // timestamp may be written before the work it precedes has begun.
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

    let started = std::time::Instant::now();
    bench.once(|cmd| unsafe {
        device.cmd_reset_query_pool(cmd, query_pool, 0, ITERATIONS as u32 + 1);
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[set],
            &[],
        );
        device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, &push);
        for _ in 0..WARMUPS {
            device.cmd_dispatch(cmd, grid[0], grid[1], grid[2]);
            if order != Order::Overlapped {
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
        }
        device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, query_pool, 0);
        for i in 0..ITERATIONS {
            device.cmd_dispatch(cmd, grid[0], grid[1], grid[2]);
            match order {
                // One timestamp per dispatch, and the barrier that makes each
                // interval a whole dispatch rather than a slice of several
                // overlapping ones.
                Order::Serial => {
                    device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[barrier],
                        &[],
                        &[],
                    );
                    device.cmd_write_timestamp(
                        cmd,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        query_pool,
                        i as u32 + 1,
                    );
                }
                // The barrier a decode pays, without the instrument that
                // would price itself alongside it.
                Order::SerialExecOnly => {
                    device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[],
                    );
                }
                Order::SerialUntimed => {
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
                // Nothing per dispatch at all: a timestamp between two
                // unbarriered dispatches would time whatever happened to have
                // retired, not a dispatch. The batch is timed as a whole and
                // divided, so the `micros` this returns is a MEAN over
                // `ITERATIONS` rather than a median -- the one row shape where
                // that is so, and it is stated where it prints.
                Order::Overlapped => {}
            }
        }
        if order != Order::Serial {
            device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, query_pool, 1);
        }
    });
    let wall = started.elapsed();

    // Only the slots that were WRITTEN may be read: `WAIT` on a query the
    // command buffer never wrote blocks forever, which is exactly what an
    // earlier draft of the overlapped path did.
    let mut ticks = vec![
        0u64;
        match order {
            Order::Serial => ITERATIONS + 1,
            Order::Overlapped | Order::SerialUntimed | Order::SerialExecOnly => 2,
        }
    ];
    unsafe {
        device.get_query_pool_results(
            query_pool,
            0,
            &mut ticks,
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
        )
    }
    .expect("timestamps");

    let median = match order {
        Order::Serial => {
            let mut micros: Vec<f64> = ticks
                .windows(2)
                .map(|w| (w[1].saturating_sub(w[0]) as f64) * f64::from(bench.period) / 1000.0)
                .collect();
            micros.sort_by(f64::total_cmp);
            micros[micros.len() / 2]
        }
        Order::Overlapped | Order::SerialUntimed | Order::SerialExecOnly => {
            (ticks[1].saturating_sub(ticks[0]) as f64) * f64::from(bench.period)
                / 1000.0
                / ITERATIONS as f64
        }
    };

    unsafe {
        device.destroy_query_pool(query_pool, None);
        device.destroy_descriptor_pool(pool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_shader_module(module, None);
        device.destroy_pipeline_layout(layout, None);
        device.destroy_descriptor_set_layout(set_layout, None);
    }
    for b in [w, scales, biases, x, y] {
        b.free(device);
    }

    let bytes = bytes_read(k, n);
    Row {
        what,
        k,
        n,
        tokens,
        groups,
        micros: median,
        // The whole submit divided by the dispatches in it, warm-ups included,
        // and it therefore carries the submit's own overhead. It is here as a
        // sanity bound on the timestamps, not as a second measurement.
        wall_micros: wall.as_secs_f64() * 1.0e6 / (ITERATIONS + WARMUPS) as f64,
        bytes,
        gbps: bytes as f64 / (median * 1000.0),
        resident: if bytes < L2_BYTES { "L2" } else { "DRAM" },
    }
}

// ---------------------------------------------------------------------------
// the sweep
// ---------------------------------------------------------------------------

/// The real qwen3-0.6B projection shapes, then N alone at K=1024.
///
/// The second half is the measurement that matters. `qmv_grid` launches
/// `ceil(N/8)` workgroups and NOTHING else scales with the problem, so a
/// decode's 1024-row projection puts 128 groups on a 128-SM card -- one group,
/// four warps, per SM out of a possible forty-eight. Walking N from 1024 to
/// 131072 walks the group count from 128 to 16384 with the per-group work
/// identical, which separates the two candidate explanations for this kernel's
/// 120 GB/s: bytes-per-second that RISES with N means the small shapes are
/// launch-starved and a split-K form would help them, bytes-per-second that
/// stays FLAT means the kernel is simply inefficient and split-K would not.
///
/// # What it answered, which was neither
///
/// The rate does rise with N -- 91 GB/s at 1024 to 646 at 65536 -- but the
/// `floor` rows say why, and it is not starvation. The dispatch costs 6.14 us
/// at ONE workgroup and 6.14 us at 256, and 6.14 us at K=64 as at K=1024. A
/// fixed cost divided by growing bytes is a rising rate, and that is all the
/// sweep is showing: nothing here is proportional to the work until the work
/// grows past the floor.
///
/// The last table is the one with the answer in it. The same dispatch that
/// costs 6.14 us barrier-separated costs 0.78 us when the card may overlap
/// several, so the kernel reaches ~640 GB/s and the rest is the barrier. An
/// execution-only dependency still costs 4-6 us, so it is the drain and not
/// the memory mask, and it is not negotiable. A decode's remaining time is in
/// the NUMBER of serialized dispatches, which is a question for the texts and
/// the fusion, not for this kernel.
#[test]
#[ignore = "a measurement, not a property: prints a table and takes tens of seconds"]
fn affine_qmv_fast_bandwidth_against_shape() {
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
        "device: {} ({} ns per timestamp tick)\n\
         {ITERATIONS} timed dispatches per shape after {WARMUPS} warm-ups, \
         one submit, median of the per-dispatch timestamp intervals\n",
        bench.name, bench.period
    );

    let shapes: Vec<(&'static str, usize, usize)> = vec![
        ("attn q/o proj", 1024, 1024),
        ("ffn gate/up", 1024, 3072),
        ("ffn down", 3072, 1024),
        ("lm_head", 1024, 151936),
        ("sweep N", 1024, 1024),
        ("sweep N", 1024, 2048),
        ("sweep N", 1024, 4096),
        ("sweep N", 1024, 8192),
        ("sweep N", 1024, 16384),
        ("sweep N", 1024, 32768),
        ("sweep N", 1024, 65536),
        ("sweep N", 1024, 131072),
        // Below the real shapes, to find the floor. If 1 workgroup and 128
        // workgroups cost the same, the floor is per-DISPATCH and no amount of
        // extra workgroups -- which is all split-K manufactures -- can remove
        // it.
        //
        // THEY DO COST THE SAME, AND A SPLIT-K MATVEC WAS NOT WRITTEN BECAUSE
        // OF THESE FOUR ROWS. One workgroup, sixty-four, a hundred and
        // twenty-eight and two hundred and fifty-six are 6.14 us each, and the
        // `floor K` rows below hold the group count at ONE while K walks from
        // 64 to 512 and read 6.14, 6.11, 6.14, 6.11. Neither the work nor the
        // parallelism moves the number.
        //
        // The suspicion this refutes was occupancy, and it was a reasonable
        // one: `qmv_fast` puts eight output rows in a 64-thread group, so a
        // 2048-wide projection is 256 groups -- about 16k threads against the
        // ~80k slots this card has -- and the in-situ per-kernel table shows
        // 19 us a dispatch against a 1.1 us memory-bound ideal. Split-K is the
        // textbook answer to exactly that shape, and the tree already has one
        // for the TILED gemm (`affine_qmm_t_splitk_*` and `qmm_splitk_reduce`)
        // and none for the vector form, which made the gap look like an
        // oversight. It is not: 20% occupancy costs nothing here because the
        // kernel is not what is being waited on.
        //
        // What 6.14 us actually is, is a barrier-separated dispatch. A decode
        // step records about 452 of them and takes 2.81 ms, which is 6.2 us
        // apiece -- the floor, to two figures, with every kernel's arithmetic
        // hiding inside it. That is the whole model of a small-model decode on
        // this card, and it says the only lever is the NUMBER of dispatches
        // that a barrier separates. See `hazards` in `driver-vulkan`, which
        // prices removing one at 3.7 us -- the part of the 6.14 that does not
        // overlap -- and states the budget a merge has to come in under.
        ("floor", 1024, 8),
        ("floor", 1024, 64),
        ("floor", 1024, 512),
        // K walking DOWN at one workgroup, which is the only way to separate
        // the two things the `floor` rows above leave fused.
        //
        // A barrier-separated dispatch costs what the barrier costs PLUS the
        // latency of one workgroup's own serial chain, and the rows above hold
        // K at 1024 throughout -- so they price the pair and cannot say which
        // half is which. This kernel's chain is proportional to K: a group
        // walks `K / PIE_XCHUNK` chunks, and inside a chunk each lane walks
        // `words_per_chunk / PIE_LANES` words before a five-round reduction.
        //
        // Extrapolating these to K=0 gives the part that is NOT the kernel --
        // the launch and the barrier -- and the difference between that and
        // the K=1024 row is the part a split-K form could actually remove.
        // `PIE_GROUP` is 64, so K must be a multiple of it.
        ("floor K", 64, 8),
        ("floor K", 128, 8),
        ("floor K", 256, 8),
        ("floor K", 512, 8),
        // N fixed at the starved end, K walking. The group count does NOT move
        // here: `qmv_grid` is `ceil(N/8)` and K appears only inside the loop,
        // as `K / PIE_XCHUNK` barrier-separated chunks each of which must
        // finish before the next may overwrite `xs_sh`. So this column prices
        // the SERIAL part of a workgroup's work -- the part split-K exists to
        // parallelise -- while holding occupancy still.
        ("sweep K", 1024, 1024),
        ("sweep K", 2048, 1024),
        ("sweep K", 4096, 1024),
        ("sweep K", 8192, 1024),
    ];

    let rows: Vec<Row> = shapes
        .iter()
        .map(|(what, k, n)| measure(&bench, what, *k, *n))
        .collect();

    eprintln!(
        "{:<14} {:>6} {:>7} {:>8} {:>10} {:>9} {:>9} {:>6}",
        "purpose", "K", "N", "groups", "bytes", "us/disp", "GB/s", "held"
    );
    for r in &rows {
        eprintln!(
            "{:<14} {:>6} {:>7} {:>8} {:>9.1}M {:>9.2} {:>9.1} {:>6}",
            r.what,
            r.k,
            r.n,
            r.groups,
            r.bytes as f64 / (1024.0 * 1024.0),
            r.micros,
            r.gbps,
            r.resident,
        );
    }
    eprintln!(
        "\nwall-clock cross-check (whole submit / dispatches, so it carries submit overhead):"
    );
    for r in &rows {
        eprintln!(
            "  K={:<6} N={:<7} timestamps {:>8.2} us   wall {:>8.2} us",
            r.k, r.n, r.micros, r.wall_micros
        );
    }

    // Serial against overlapped, at the shapes a decode actually runs.
    //
    // The serial column is what a decode pays. The overlapped column is what
    // the same dispatch costs when the card is allowed to run several at once,
    // and the gap between them is the part of the cost that is LATENCY --
    // pipeline drain, barrier, and one workgroup's own dependency chain --
    // rather than work. A split-K form does not shorten that chain by making
    // more workgroups unless the card was short of workgroups to begin with,
    // which the `floor` rows above answer directly.
    // Serial against overlapped, at the shapes a decode actually runs.
    //
    // Three columns and not two, because the serial one prices the barrier
    // together with the timestamp that measures it. `untimed` records the same
    // barriers and brackets the whole batch, so:
    //
    //   overlap                -> what the work costs when the card may
    //                             run several dispatches at once;
    //   untimed  - overlap     -> what the BARRIER costs;
    //   serial   - untimed     -> what the instrument costs.
    //
    // Only the middle one is reachable by changing this repository, and it is
    // reachable in exactly one way: a decode records a barrier because one
    // dispatch reads what the last one wrote, so the way to fewer barriers is
    // fewer kernels.
    eprintln!(
        "\nbarrier-separated against overlapped (means, not medians, for the last two columns):"
    );
    eprintln!(
        "{:<14} {:>6} {:>7} {:>8} {:>11} {:>11} {:>11} {:>11}",
        "purpose", "K", "N", "groups", "serial us", "untimed us", "execonly us", "overlap us"
    );
    for (what, k, n) in [
        ("attn q/o proj", 1024usize, 1024usize),
        ("ffn gate/up", 1024, 3072),
        ("ffn down", 3072, 1024),
        ("one group", 1024, 8),
    ] {
        let serial = measure_as(&bench, what, k, n, Order::Serial);
        let untimed = measure_as(&bench, what, k, n, Order::SerialUntimed);
        let execonly = measure_as(&bench, what, k, n, Order::SerialExecOnly);
        let overlapped = measure_as(&bench, what, k, n, Order::Overlapped);
        eprintln!(
            "{:<14} {:>6} {:>7} {:>8} {:>11.2} {:>11.2} {:>11.2} {:>11.2}",
            what,
            k,
            n,
            serial.groups,
            serial.micros,
            untimed.micros,
            execonly.micros,
            overlapped.micros
        );
    }

    // The only assertion, and it is about the instrument rather than the
    // kernel: a timestamp pool that returns zeros or a period of zero
    // would print a table of infinities and look like a result.
    assert!(
        rows.iter().all(|r| r.micros > 0.0),
        "a dispatch that took no measurable time means the timestamps did not work",
    );

    // ---------------------------------------------------------------------
    // the token sweep: what a decode BATCH costs
    // ---------------------------------------------------------------------
    //
    // Every row above multiplies one activation vector, which is the shape a
    // single-stream decode runs. A server does not run one stream. The
    // scheduler gathers whatever contexts are ready into one forward pass, and
    // the whole reason that is worth doing is that a projection's weights are
    // read ONCE for the batch: the matrix is tens of megabytes and the
    // activations are kilobytes, so at batch 16 a memory-bound projection
    // should cost very nearly what it costs at batch 1.
    //
    // `qmv_grid` multiplies the x extent by the row count, so each row gets
    // its own workgroups and each of those re-reads the weight plane. Whether
    // the cache absorbs that -- the second row's read hits L2 behind the
    // first's -- decides whether this kernel can serve a batch at all.
    //
    // The number to read is `us/token`. Flat across the sweep means the batch
    // buys nothing and the server's throughput is capped at one stream's; a
    // fall towards 1/tokens means the weights are being shared.
    //
    // # What a WHOLE step does, which is not what this kernel does
    //
    // These rows are one kernel, and a decode is not one kernel. Read alone
    // they invite the conclusion that batching is worthless here, and
    // `driver-vulkan`'s `tests/hostprof.rs` -- which now steps a batch, and
    // must be run under `--release` -- says otherwise: the device step is
    // 2.75 ms at batch 1 and 8.35 at batch 8, three times the time for eight
    // times the tokens, so a batch of eight really is about 2.6x the
    // throughput of one stream.
    //
    // Both are true and they are the same fact from two ends. A step's fixed
    // costs do not scale -- the ~1.9 ms of barriers, the norms, the
    // elementwise ops, the attention over a short history -- so they are
    // amortised by a batch exactly as they should be. What does NOT amortise
    // is the projections, because of the re-read these rows measure, and the
    // projections are 70% of a step. The matvec is therefore what CAPS the
    // gain at 2.6x rather than what removes it, and a kernel that shared its
    // weight reads across the batch is worth roughly the remaining 3x.
    eprintln!("\ntokens: one dispatch over M activation rows, barrier-separated as a decode is:");
    eprintln!(
        "{:<14} {:>6} {:>7} {:>7} {:>11} {:>11} {:>9}",
        "purpose", "K", "N", "tokens", "us/disp", "us/token", "GB/s"
    );
    let mut token_rows: Vec<Row> = Vec::new();
    for (what, k, n) in [
        ("attn q/o proj", 1024usize, 1024usize),
        ("ffn gate/up", 1024, 3072),
    ] {
        for tokens in [1usize, 2, 4, 8, 16, 32] {
            let r = measure_tokens(&bench, what, k, n, tokens, Order::SerialUntimed);
            eprintln!(
                "{:<14} {:>6} {:>7} {:>7} {:>11.2} {:>11.2} {:>9.1}",
                r.what,
                r.k,
                r.n,
                r.tokens,
                r.micros,
                r.micros / r.tokens as f64,
                // The weight traffic a PERFECT batched kernel would move: one
                // pass over the planes regardless of the row count. Reading
                // the rate against that makes the column say directly how far
                // from batched this kernel is.
                r.bytes as f64 / (r.micros * 1000.0),
            );
            token_rows.push(r);
        }
    }
    assert!(
        token_rows.iter().all(|r| r.micros > 0.0),
        "a batched dispatch that took no measurable time means the timestamps did not work",
    );

    // ---------------------------------------------------------------------
    // the matvec against the tiled GEMM, at the same batch
    // ---------------------------------------------------------------------
    //
    // The rows above say the matvec charges nearly full price per token. This
    // says what the alternative costs. `affine_qmm_t` reads a `bm x bn` tile of
    // the output per workgroup and therefore reads each weight tile ONCE for
    // all `bm` rows, which is the whole point of a GEMM and exactly what the
    // matvec does not do.
    //
    // `crates/model/src/shared/llama_like/forward/mod.rs` already writes this
    // kernel into the text, behind `GuardPred::TokensMultipleOf(bm)`. A decode
    // batch is whatever the scheduler gathered -- two, five, ten, twenty-six --
    // and none of those is a multiple of thirty-two, so the guard's other arm
    // takes every decode and the GEMM runs only on prefill.
    //
    // # What it answered: the GEMM is NOT the fix, and this is why the rows are
    // # kept
    //
    // The obvious reading of the token sweep is that a decode should pad its
    // batch up to the tile and take the GEMM. These rows say it should not.
    // The tiled form has a floor of its own and it is an order of magnitude
    // above the matvec's, and the number barely moves between sixteen rows and
    // thirty-two, or between N=1024 and N=3072.
    //
    // # What that floor is made of: work per workgroup, not workgroups
    //
    // The four extra tile shapes below were added to ask whether the floor is
    // a property of the KERNEL or of the TILE, because if a decode-shaped tile
    // -- few rows, many columns -- had a lower one, padding might still pay.
    // Every shape here does exactly the same M*N*K multiply-adds; only the
    // partitioning changes. The measured spread is fourfold:
    //
    //   attn q/o proj, K=1024 N=1024, 16 tokens
    //     qmv            16.37 us   2048 workgroups
    //     qmm 16x16      55.23        64
    //     qmm 64x16     167.43        64
    //     qmm 32x32     211.87        32
    //     qmm 32x32 cm  102.29        32
    //     qmm 16x64     236.37        16
    //     qmm 64x64 cm  279.37        16
    //
    // The first hypothesis was occupancy: 16 to 96 workgroups cannot fill this
    // card, so the dispatch is exposed. The rows REFUTE it as stated. `16x16`
    // and `64x16` launch the SAME sixty-four workgroups and differ threefold
    // (55 against 167); `16x64` launches a quarter as many as `64x16` and is
    // only 1.4x worse, not four times.
    //
    // What does track is the work inside one workgroup, `bm*bn*K`. Area 256
    // costs 55 us; all three area-1024 tiles cost 167 to 236. The card is
    // latency-bound rather than throughput-bound at these sizes, so the
    // dispatch lasts as long as ONE workgroup's serial walk down K, and extra
    // workgroups are nearly free. That is visible directly: `16x16` triples its
    // workgroups from 64 to 192 going N=1024 to N=3072 and costs 1.39x, and
    // doubles them from 64 to 128 going 16 tokens to 32 and costs 1.11x.
    //
    // So the "floor" is not fixed and is not the kernel's. It is the tile's,
    // and the previously recorded ~102 us was simply the smallest tile anyone
    // had tried. At N=1024 the plain `16x16` beats the coopmat `32x32` nearly
    // twofold, 55 against 102. (At N=3072 and 32 tokens the coopmat form takes
    // it back, 113.87 against 131.82 -- the shapes cross, which is why both
    // stay in the table.)
    //
    // # And the decode answer is still no -- now at the BEST tile
    //
    // This closes the question rather than reopening it. Even at its best
    // shape the GEMM loses to the matvec at every batch a decode can present:
    // 55.23 against 16.37 at sixteen rows, 61.27 against 26.65 at thirty-two.
    // The two slopes are 0.38 and 0.64 us a row, so they cross at around a
    // hundred and fifty rows -- an order of magnitude beyond any decode batch,
    // and well inside the prefill regime where the GEMM already runs.
    //
    // That also retires the lever named at the end of the batched-path work in
    // `driver-vulkan`'s `hazards` doc. The observation there is correct -- a
    // batch of eight does eight times the multiply-adds through a scalar loop
    // over `pc.m`, and those do not reach the tensor cores. But routing them
    // through the tensor cores MEANS this kernel, and this kernel cannot get
    // started inside a decode's budget at any tile. The 1.33 ms cost of
    // entering the batched path is not recoverable by choosing a better
    // existing kernel; the matvec, the batched matvec, split-K and the tiled
    // GEMM have now each been measured and each lost.
    //
    // # Narrowed by measuring the competitor
    //
    // vLLM 0.27.1 on this same card, Qwen3-0.6B bf16 with full CUDA graphs,
    // costs 2.445 ms a step at batch one and 3.024 at batch eight -- 0.083 ms
    // a sequence at the margin, against pie's 0.54. It reaches that by doing
    // exactly the routing this table says is unavailable: a small-M GEMM on
    // the tensor cores. So "the tensor cores are out of reach for a decode"
    // is too strong. What the table actually proves is that THIS GEMM is out
    // of reach, because a cuBLAS-class kernel at M=8 plainly has no 55 us
    // floor.
    //
    // Which turns the negative result into a target, and the floor's own
    // explanation says what shape to aim at. The cost is the work in one
    // workgroup, `bm*bn*K`, on a card nowhere near saturated -- so a kernel
    // that wants M=8 should be short in M, long in N, and should split the
    // K-loop across workgroups rather than walking it serially. None of the
    // nine instantiations here is that shape; `bm=16` is the smallest M this
    // file can even ask for, and every one of them walks all of K in one
    // workgroup.
    //
    // That floor is not an artefact of this harness either.
    // `driver-vulkan`'s
    // `the_projections_dominate_both_steps_now_that_the_decode_splits_its_keys`
    // prints the same kernel's real fires: `affine_qmm_t_fp16_precast` costs
    // 44.398 ms over 140 fires of a 384-token prefill, which is 317 us a fire
    // -- consistent with a large fixed cost plus a small per-row one. The GEMM
    // is a PREFILL kernel. It amortises its floor over hundreds of rows and
    // there is nothing wrong with that; it simply has no answer for ten.
    //
    // So the lever the token sweep exposes is not reachable by choosing
    // between these two kernels. Both are wrong for a decode batch: the matvec
    // re-reads the weights per row, and the GEMM cannot get started in under a
    // hundred microseconds. What the shape calls for is the third thing --
    // a matvec that holds `bm` activation rows and streams each weight tile
    // once for all of them, keeping the matvec's launch cost and paying the
    // weight traffic once. That kernel was written, measured, and deleted --
    // the weight traffic was never the cost. `kernels/quant/qmv.slang` carries
    // the refutation, and the table after this one is the half of its evidence
    // that survives in a shipped kernel.
    //
    // # There is no floor. It is the serial K walk, and it is linear.
    //
    // The `K/4` and `K/2` rows hold the shape fixed and shorten the reduction.
    // If the ~100 us were a fixed cost -- setup, dequantise tables, coopmat
    // warm-up -- it would survive. It does not survive at all:
    //
    //   qmm 16x16, N=1024, 16 tokens
    //     K=256    16.37 us
    //     K=512    28.65
    //     K=1024   55.23
    //
    // That is 0.0506 us per unit of K with a 3.4 us intercept, and the same
    // straight line fits every other tile in the table. The intercept is one
    // dispatch. So "affine_qmm_t has a ~102 us floor" was wrong as stated:
    // there is no floor, there is a workgroup walking K one step at a time
    // and being charged for every step, and the number scales with `bm*bn*K`
    // exactly because that is how much walking one workgroup does.
    //
    // Which is what makes split-K the right instrument HERE, and it is worth
    // being clear that this does not contradict the split-K refutation in
    // `qmv.slang`. That one is about the MATVEC, whose workgroups are already
    // short and many and which split-K only makes more numerous. This kernel
    // is the opposite case: its whole cost is the length of one workgroup's
    // walk, doubling the workgroups here has been measured at roughly free
    // (16.37 us at 64 workgroups, 16.45 at 128), and `affine_qmm_t_splitk_*`
    // is already instantiated.
    //
    // # SPLIT-K WAS MEASURED, AND IT LOSES BY FOURTEEN TIMES
    //
    // The estimate this paragraph used to carry was 25 to 30 us: buy back the
    // K=256 row at ~16 us, pay between 1.0 and 1.4 for four times the
    // workgroups, add a reduce dispatch. The measurement, at the batch a
    // decode actually brings:
    //
    //   attn q/o proj, K=1024 N=1024, 8 tokens
    //     qmv                10.26 us   1024 workgroups
    //     qmm 16x16          47.16        64
    //     qmm 16x32 k/1     319.09        32
    //     qmm 16x32 k/2     161.21        64
    //     qmm 16x32 k/4     149.22       128
    //     qmm 16x32 k/8     146.30       256
    //
    // Fourteen times the matvec at its best, and that is the PARTIAL pass
    // alone -- the reduce dispatch and the barrier in front of it are not in
    // the column.
    //
    // # The estimate was wrong for a reason worth keeping
    //
    // The `k/1` control row is why this sweep has one. `affine_qmm_t_splitk`
    // at one partition does exactly what `affine_qmm_t` does, and it costs
    // 319 us where the plain kernel at half the tile area costs 47. Those are
    // not the same kernel body. The plain form tiles its inner loop through
    // shared memory; the split-K form takes the scalar `for kk` walk at the
    // bottom of `qmm_t.slang`, which re-fetches a scale and a bias for every
    // element it touches. Sixfold, and it has nothing to do with splitting.
    //
    // So the 25-to-30 estimate took the K-linearity constant measured on ONE
    // kernel and applied it to ANOTHER kernel's body on the strength of a
    // shared name. That is the third time in this work that a number has been
    // carried across a boundary it does not hold over -- after
    // `PIE_VULKAN_TIMING`'s per-dispatch absolutes and the marginal-against-
    // wholesale barrier price -- and the rule that comes out of all three is
    // the same: a constant belongs to the thing it was measured on, and
    // moving it to a sibling is a new measurement and not an inference.
    //
    // # What the split itself does, separately from that
    //
    // Held against its own `k/1`, splitting behaves exactly as the linearity
    // model says for one halving and then stops: 319 -> 161 is a clean two,
    // and 161 -> 149 -> 146 is nothing. There is a ~146 us floor in this body
    // that no partition reaches. So split-K is not merely riding a slow
    // kernel; it also saturates well above anything a decode can pay. Fixing
    // the scalar walk would have to come first, and even a sixfold fix lands
    // at ~24 us against the matvec's 10.26.
    //
    // The conclusion is therefore not "split-K needs a better base kernel".
    // It is that this table has now measured every existing kernel against a
    // decode batch -- the matvec, the batched matvec, the tiled GEMM at seven
    // tile shapes, and split-K at four partitions -- and the matvec wins all
    // of them. The projections are not going to be improved by ROUTING. A
    // decode-shaped quantised GEMM would have to be written from the tile up,
    // and nothing in this file says what it should look like beyond "short in
    // M, long in N, and not this scalar walk".
    //
    // # What the batch-8 column says on its own
    //
    // The `tokens` sweep at the top of this file gained a row of 8 in the
    // same diff, and it splits the two projection shapes apart:
    //
    //     tokens        1     2     4     8    16    32
    //     attn q/o   7.18  6.14  6.15 10.24 16.36 26.65
    //     ffn g/u    6.15 10.23 14.33 22.51 38.87 72.34
    //
    // Four rows of the attention projection cost LESS than one. At N=1024 the
    // matvec launches 1024 workgroups at one token and the card is not full,
    // so the first three rows of a batch are free. The ffn shape at N=3072 is
    // already saturated at one token and pays proportionally from there.
    //
    // That is where a decode batch's cost actually comes from, and it is a
    // sharper statement than "the projections do not amortise": the WIDE
    // projections do not amortise, because they are the ones that fill the
    // card at a single row. Any future kernel has to beat a matvec that is
    // already running at capacity on the shapes that matter.
    //
    // The baseline column is kept beside the coopmat one because the gap
    // between them is itself a fact worth holding: the non-coopmat GEMM is
    // twice as slow again at the tiles where coopmat wins, so a card without
    // cooperative matrices is not merely a little behind on prefill.
    eprintln!("\nthe matvec against the tiled GEMM, both barrier-separated:");
    eprintln!(
        "{:<14} {:>6} {:>7} {:>7} {:>9} {:>11} {:>11} {:>8}",
        "purpose", "K", "N", "tokens", "kernel", "us/disp", "us/token", "wg"
    );
    for (what, k, n) in [
        ("attn q/o proj", 1024usize, 1024usize),
        ("K/4 same shape", 256, 1024),
        ("K/2 same shape", 512, 1024),
        ("ffn gate/up", 1024, 3072),
    ] {
        // EIGHT FIRST, because eight is the batch a decode actually brings and
        // every row of this table used to start at sixteen. The 16 and 32
        // columns are the prefill regime and are kept for the shape of the
        // curve; the verdict is read off the 8.
        for tokens in [8usize, 16, 32] {
            for (label, entry, cap, bm, bn, splits) in [
                ("qmv", ENTRYPOINT, Capability::Baseline, 0usize, 0usize, 0usize),
                (
                    "qmm 16x16",
                    "affine_qmm_t_bfloat16_gs_128_b_4_bm_16_bn_16",
                    Capability::Baseline,
                    16,
                    16,
                    0,
                ),
                (
                    "qmm 16x64",
                    "affine_qmm_t_bfloat16_gs_128_b_4_bm_16_bn_64",
                    Capability::Baseline,
                    16,
                    64,
                    0,
                ),
                (
                    "qmm 64x16",
                    "affine_qmm_t_bfloat16_gs_128_b_4_bm_64_bn_16",
                    Capability::Baseline,
                    64,
                    16,
                    0,
                ),
                (
                    "qmm 32x32",
                    "affine_qmm_t_bfloat16_gs_128_b_4_bm_32_bn_32",
                    Capability::Baseline,
                    32,
                    32,
                    0,
                ),
                (
                    "qmm 32x32 cm",
                    "affine_qmm_t_bfloat16_gs_128_b_4_bm_32_bn_32",
                    Capability::Coopmat,
                    32,
                    32,
                    0,
                ),
                (
                    "qmm 64x64 cm",
                    "affine_qmm_t_bfloat16_gs_128_b_4_bm_64_bn_64",
                    Capability::Coopmat,
                    64,
                    64,
                    0,
                ),
                // The split-K rows. Same tile as `qmm 16x16` -- the cheapest
                // in the table -- with the K walk cut four and eight ways
                // across the `z` extent. These price the PARTIAL pass only;
                // the reduce dispatch and the barrier in front of it are
                // added in the prose below, because the harness times one
                // pipeline at a time and a two-pass kernel priced as one pass
                // would be the same mistake `PIE_VULKAN_TIMING` makes.
                (
                    "qmm 16x32 k/1",
                    "affine_qmm_t_splitk_bfloat16_gs_128_b_4_bm_16_bn_32",
                    Capability::Baseline,
                    16,
                    32,
                    1,
                ),
                (
                    "qmm 16x32 k/2",
                    "affine_qmm_t_splitk_bfloat16_gs_128_b_4_bm_16_bn_32",
                    Capability::Baseline,
                    16,
                    32,
                    2,
                ),
                (
                    "qmm 16x32 k/4",
                    "affine_qmm_t_splitk_bfloat16_gs_128_b_4_bm_16_bn_32",
                    Capability::Baseline,
                    16,
                    32,
                    4,
                ),
                (
                    "qmm 16x32 k/8",
                    "affine_qmm_t_splitk_bfloat16_gs_128_b_4_bm_16_bn_32",
                    Capability::Baseline,
                    16,
                    32,
                    8,
                ),
            ] {
                let r = measure_kernel(
                    &bench,
                    what,
                    k,
                    n,
                    tokens,
                    Order::SerialUntimed,
                    entry,
                    cap,
                    splits,
                    |m, n| {
                        if bm == 0 {
                            [m as u32, n.div_ceil(8) as u32, 1]
                        } else {
                            [
                                n.div_ceil(bn) as u32,
                                m.div_ceil(bm) as u32,
                                splits.max(1) as u32,
                            ]
                        }
                    },
                );
                eprintln!(
                    "{:<14} {:>6} {:>7} {:>7} {:>9} {:>11.2} {:>11.2} {:>8}",
                    r.what,
                    r.k,
                    r.n,
                    r.tokens,
                    label,
                    r.micros,
                    r.micros / r.tokens as f64,
                    if bm == 0 {
                        r.tokens * n.div_ceil(8)
                    } else {
                        n.div_ceil(bn) * r.tokens.div_ceil(bm) * splits.max(1)
                    },
                );
            }
        }
    }

    // WHAT THE MATVEC'S WEIGHT TRAFFIC ACTUALLY COSTS.
    //
    // The obvious reading of the token sweep above is that the matvec is
    // bound by re-reading its weight plane once per activation row, and that
    // a kernel sharing each fetch across rows would flatten it. That kernel
    // was written and it lost -- 77.68 us against this form's 72.03 at the
    // 0.6B's ffn shape -- and it only ever won, by five to eight percent, at
    // planes far larger than any projection in this tree. The full table is
    // in `kernels/quant/qmv.slang`, which is also where the argument is set
    // out so nobody rebuilds it.
    //
    // This is the half of that evidence that does not need the deleted
    // kernel, and it is the load-bearing half. It walks the plane from well
    // inside this card's 72 MB of L2 to twice outside it at a fixed batch of
    // thirty-two, and prints the rate at which the weights ACTUALLY leave
    // DRAM: the plane is read once from memory however many rows consume it,
    // because the thirty-two workgroups reading it are co-resident in time
    // and it streams through L2 for all of them.
    //
    // The number to watch is the last column. If it stays an order of
    // magnitude under what the card can stream -- and at the 128 MB plane it
    // is 24 GB/s against roughly 900 available -- then this kernel is not
    // waiting on memory at any size, and no rearrangement of its reads can
    // pay. It is bound by the dequantise-and-accumulate arithmetic and by the
    // L2-to-SM issue rate, both of which scale with the batch and neither of
    // which sharing a fetch removes: sharing the word saves the unpack, not
    // the fused multiply-add.
    //
    // Should this column ever climb toward the card's limit -- a much larger
    // model, a wider batch, a card with less L2 -- the deleted kernel becomes
    // worth writing again, and the numbers it scored are recorded next to it.
    eprintln!("\nwhat the matvec's weight traffic actually costs (batch 32):");
    eprintln!(
        "{:<8} {:>6} {:>7} {:>9} {:>11} {:>13} {:>11}",
        "plane", "K", "N", "MB", "us/disp", "logical GB/s", "DRAM GB/s"
    );
    for (k, n) in [
        (1024usize, 3072usize),
        (2048, 8192),
        (4096, 12288),
        (8192, 16384),
        (8192, 32768),
    ] {
        let bytes = (k * n / 2) as f64;
        let r = measure_kernel(
            &bench,
            "plane",
            k,
            n,
            32,
            Order::SerialUntimed,
            ENTRYPOINT,
            Capability::Baseline,
            0,
            |m, n| [m as u32, n.div_ceil(8) as u32, 1],
        );
        // "logical" is what the kernel would move if every row's re-read
        // reached memory; "DRAM" is what it must move at minimum, the plane
        // once. The gap between the two columns is the cache doing the work
        // that the deleted kernel was written to do in registers.
        let secs = r.micros * 1e-6;
        eprintln!(
            "{:<8} {:>6} {:>7} {:>9.1} {:>11.2} {:>13.0} {:>11.0}",
            "plane",
            k,
            n,
            bytes / (1024.0 * 1024.0),
            r.micros,
            bytes * 32.0 / secs / 1e9,
            bytes / secs / 1e9,
        );
    }
}
