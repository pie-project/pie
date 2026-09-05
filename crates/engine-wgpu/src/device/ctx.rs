use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use crate::api::DeviceBoot;
use crate::error::{Fault, Result};

use super::alloc::{Memory, Raw, Slab};

static RESERVATIONS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn note_reservation() {
    RESERVATIONS.fetch_add(1, Ordering::Relaxed);
}

#[must_use]
pub fn reservations() -> u64 {
    RESERVATIONS.load(Ordering::Relaxed)
}

fn instance(backends: wgpu::Backends) -> &'static wgpu::Instance {
    static INSTANCE: OnceLock<wgpu::Instance> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        })
    })
}

fn gate() -> std::sync::MutexGuard<'static, ()> {
    static GATE: Mutex<()> = Mutex::new(());
    GATE.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[must_use]
pub fn present() -> bool {
    let _gate = gate();
    let instance = instance(wgpu::Backends::all());
    !pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all())).is_empty()
}

pub(crate) const STAGING_BYTES: u64 = 256 << 20;

pub(crate) const DUMMY_BYTES: u64 = 256;

const UNIFORM_CHUNK: u64 = 1 << 20;

const SCRATCH_CHUNK: u64 = 8 << 20;

const WAIT_TIMEOUT: Duration = Duration::from_secs(120);

const TIMING_QUERIES: u32 = 4096;

const SPAN_AT: u64 = (TIMING_QUERIES as u64) * 8;

pub(crate) struct Timing {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    names: Vec<String>,
}

impl Timing {
    fn open(core: &Core) -> Timing {
        let bytes = SPAN_AT + 256;
        Timing {
            set: core.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("pie timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: TIMING_QUERIES,
            }),
            resolve: core.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pie timestamps resolve"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback: core.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pie timestamps readback"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            names: Vec::new(),
        }
    }

    fn harvest(&self, core: &Core, first: bool) {
        if self.names.is_empty() {
            return;
        }
        let period = f64::from(core.queue.get_timestamp_period());
        let bytes = if first {
            SPAN_AT + 16
        } else {
            u64::from(self.names.len() as u32) * 16
        };
        let slice = self.readback.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = core.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(WAIT_TIMEOUT),
        });
        if !matches!(rx.recv(), Ok(Ok(()))) {
            return;
        }
        if let Ok(view) = slice.get_mapped_range() {
            let word = |i: usize| {
                let mut b = [0u8; 8];
                b.copy_from_slice(&view[i * 8..i * 8 + 8]);
                u64::from_le_bytes(b)
            };
            for (at, name) in self.names.iter().enumerate() {
                let (begin, end) = (word(2 * at), word(2 * at + 1));
                let ns = (end.saturating_sub(begin) as f64 * period) as u64;
                crate::encode::record_timing(name, ns);
            }
            if first {
                let at = (SPAN_AT / 8) as usize;
                let ns = (word(at + 1).saturating_sub(word(at)) as f64 * period) as u64;
                crate::encode::record_timing("frame.span", ns);
            }
        }
        self.readback.unmap();
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Enabled {
    pub subgroups: bool,
    pub f16: bool,
    pub timestamps: bool,
    pub mappable_primary: bool,
    pub pipeline_cache: bool,
}

pub(crate) struct Core {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) info: wgpu::AdapterInfo,
    pub(crate) limits: wgpu::Limits,
    pub(crate) enabled: Enabled,

    pub(crate) staging: Mutex<Option<wgpu::Buffer>>,

    pub(crate) uniforms: Mutex<Vec<wgpu::Buffer>>,
    pub(crate) scratch: Mutex<Vec<wgpu::Buffer>>,

    pub(crate) last_error: Arc<Mutex<Option<String>>>,
    pub(crate) allocated: AtomicU64,
    pub(crate) name: String,
    pub(crate) cores: u32,
    pub(crate) subgroup_size: u32,
    pub(crate) device_local: u64,
}

impl Core {
    pub(crate) fn fault(&self, what: &'static str, why: impl ToString) -> Fault {
        Fault::Wgpu {
            what,
            why: why.to_string(),
        }
    }

    pub(crate) fn take_error(&self, what: &'static str) -> Result<()> {
        let taken = self
            .last_error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        match taken {
            Some(why) => Err(Fault::Wgpu { what, why }),
            None => Ok(()),
        }
    }

    pub(crate) fn wait_for(&self, index: &wgpu::SubmissionIndex) -> Result<()> {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(index.clone()),
                timeout: Some(WAIT_TIMEOUT),
            })
            .map_err(|e| self.fault("Device::poll", e))?;
        self.take_error("submission")
    }

    pub(crate) fn submit_once(&self, encoder: wgpu::CommandEncoder) -> Result<()> {
        let index = self.queue.submit(std::iter::once(encoder.finish()));
        self.wait_for(&index)
    }

    pub(crate) fn staging<'a>(&self, guard: &'a mut Option<wgpu::Buffer>) -> &'a wgpu::Buffer {
        guard.get_or_insert_with(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pie staging"),
                size: STAGING_BYTES,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
    }

    pub(crate) fn uniform_chunk(&self) -> wgpu::Buffer {
        if let Some(chunk) = self
            .uniforms
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop()
        {
            return chunk;
        }
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pie uniforms"),
            size: UNIFORM_CHUNK,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub(crate) fn return_uniforms(&self, chunks: Vec<wgpu::Buffer>) {
        self.uniforms
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .extend(chunks);
    }

    pub(crate) fn scratch_chunk(&self, bytes: u64) -> wgpu::Buffer {
        if bytes <= SCRATCH_CHUNK
            && let Some(chunk) = self
                .scratch
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .pop()
        {
            return chunk;
        }
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pie scratch"),
            size: bytes.max(SCRATCH_CHUNK),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub(crate) fn return_scratch(&self, chunks: Vec<wgpu::Buffer>) {
        self.scratch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .extend(chunks.into_iter().filter(|c| c.size() == SCRATCH_CHUNK));
    }
}

pub struct Context {
    pub(crate) core: Arc<Core>,
    working_set: u64,
    max_buffer: u64,
    pub(crate) dummy: Slab,
    pipeline_cache: Option<std::path::PathBuf>,
    device_index: u32,
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context")
            .field("name", &self.core.name)
            .field("backend", &self.core.info.backend)
            .field("working_set", &self.working_set)
            .finish()
    }
}

fn rank(kind: wgpu::DeviceType, preference: wgpu::PowerPreference) -> u8 {
    use wgpu::DeviceType as D;
    match preference {
        wgpu::PowerPreference::LowPower => match kind {
            D::IntegratedGpu => 0,
            D::DiscreteGpu => 1,
            D::VirtualGpu => 2,
            D::Other => 3,
            D::Cpu => 4,
        },
        _ => match kind {
            D::DiscreteGpu => 0,
            D::IntegratedGpu => 1,
            D::VirtualGpu => 2,
            D::Other => 3,
            D::Cpu => 4,
        },
    }
}

fn vulkan_facts(adapter: &wgpu::Adapter) -> (Option<u64>, Option<u32>, Option<u32>) {
    use ash::vk;
    let Some(hal) = (unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }) else {
        return (None, None, None);
    };
    let instance = hal.shared_instance().raw_instance();
    let physical = hal.raw_physical_device();
    let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
    let heaps = memory.memory_heap_count as usize;
    let local = (0..heaps)
        .filter(|&i| {
            memory.memory_heaps[i]
                .flags
                .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
        })
        .map(|i| memory.memory_heaps[i].size)
        .max();
    let exts =
        unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap_or_default();
    let has = |name: &std::ffi::CStr| {
        exts.iter()
            .any(|e| e.extension_name_as_c_str().is_ok_and(|s| s == name))
    };
    let cores = if has(ash::nv::shader_sm_builtins::NAME) {
        let mut sm = vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV::default();
        let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sm);
        unsafe { instance.get_physical_device_properties2(physical, &mut p2) };
        Some(sm.shader_sm_count)
    } else if has(ash::amd::shader_core_properties::NAME) {
        let mut cu = vk::PhysicalDeviceShaderCorePropertiesAMD::default();
        let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut cu);
        unsafe { instance.get_physical_device_properties2(physical, &mut p2) };
        Some(
            cu.shader_engine_count
                * cu.shader_arrays_per_engine_count
                * cu.compute_units_per_shader_array,
        )
    } else {
        None
    };
    let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
    let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
    unsafe { instance.get_physical_device_properties2(physical, &mut p2) };
    (
        local,
        cores.filter(|&n| n > 0),
        Some(subgroup.subgroup_size).filter(|&n| n > 0),
    )
}

impl Context {
    pub fn bind(boot: &DeviceBoot) -> Result<Context> {
        let backends = boot
            .backends
            .as_deref()
            .map_or(wgpu::Backends::all(), wgpu::Backends::from_comma_list);
        let preference = match boot.power_preference.as_str() {
            "low-power" | "low_power" => wgpu::PowerPreference::LowPower,
            "none" => wgpu::PowerPreference::None,
            _ => wgpu::PowerPreference::HighPerformance,
        };
        let _gate = gate();
        let instance = instance(backends);
        let mut adapters = pollster::block_on(instance.enumerate_adapters(backends));
        adapters.sort_by_key(|a| rank(a.get_info().device_type, preference));
        let roster: Vec<String> = adapters
            .iter()
            .map(|a| {
                let i = a.get_info();
                format!("{} ({:?}, {:?})", i.name, i.device_type, i.backend)
            })
            .collect();
        if adapters.is_empty() {
            return Err(Fault::NoDevice {
                detail: format!("no wgpu adapter on backends {backends:?}"),
            });
        }
        let adapter = adapters
            .into_iter()
            .nth(boot.adapter_index as usize)
            .ok_or_else(|| Fault::NoDevice {
                detail: format!(
                    "no adapter at index {} (seen: {roster:?})",
                    boot.adapter_index
                ),
            })?;
        let info = adapter.get_info();
        let offered = adapter.features();
        let wanted = wgpu::Features::SUBGROUP
            | wgpu::Features::SHADER_F16
            | wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | wgpu::Features::MAPPABLE_PRIMARY_BUFFERS
            | wgpu::Features::PIPELINE_CACHE;
        let features = offered & wanted;
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("pie"),
            required_features: features,
            required_limits: limits.clone(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| Fault::NoDevice {
            detail: format!("request_device on {}: {e}", info.name),
        })?;
        let last_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        {
            let sink = Arc::clone(&last_error);
            device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
                let text = error.to_string();
                eprintln!("wgpu: {text}");
                *sink
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(text);
            }));
        }
        let (heap, sm, subgroup) = vulkan_facts(&adapter);
        let device_local = boot
            .device_memory
            .or(heap)
            .unwrap_or(crate::boot::DEFAULT_DEVICE_MEMORY);
        let enabled = Enabled {
            subgroups: features.contains(wgpu::Features::SUBGROUP),
            f16: features.contains(wgpu::Features::SHADER_F16),
            timestamps: features.contains(wgpu::Features::TIMESTAMP_QUERY),
            mappable_primary: features.contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS),
            pipeline_cache: features.contains(wgpu::Features::PIPELINE_CACHE),
        };
        let max_buffer = limits.max_buffer_size;
        let core = Arc::new(Core {
            device,
            queue,
            name: info.name.clone(),
            info,
            limits,
            enabled,
            staging: Mutex::new(None),
            uniforms: Mutex::new(Vec::new()),
            scratch: Mutex::new(Vec::new()),
            last_error,
            allocated: AtomicU64::new(0),
            cores: sm.unwrap_or(16),
            subgroup_size: subgroup.unwrap_or(32),
            device_local,
        });
        let dummy = Raw::new(&core, DUMMY_BYTES, Memory::Device)?;
        let fraction = boot.gpu_mem_utilization.clamp(0.0, 1.0);
        let context = Context {
            working_set: (device_local as f64 * fraction) as u64,
            max_buffer,
            core,
            dummy,
            pipeline_cache: boot.pipeline_cache.clone(),
            device_index: boot.adapter_index,
        };

        kernels_wgpu::tuning::describe(context.info());
        Ok(context)
    }

    pub(crate) fn core(&self) -> &Arc<Core> {
        &self.core
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.core.name
    }

    #[must_use]
    pub fn backend(&self) -> &'static str {
        self.core.info.backend.to_str()
    }

    #[must_use]
    pub fn working_set(&self) -> u64 {
        self.working_set
    }

    #[must_use]
    pub fn max_buffer(&self) -> u64 {
        self.max_buffer
    }

    #[must_use]
    pub fn cores(&self) -> u32 {
        self.core.cores
    }

    #[must_use]
    pub fn subgroup_size(&self) -> u32 {
        self.core.subgroup_size
    }

    #[must_use]
    pub fn tiers(&self) -> Vec<kernels_wgpu::Capability> {
        let mut out = Vec::new();
        if self.core.enabled.subgroups && self.core.subgroup_size >= 32 {
            out.push(kernels_wgpu::Capability::Subgroup);
        }
        if self.core.enabled.f16 {
            out.push(kernels_wgpu::Capability::Fp16);
        }
        out
    }

    #[must_use]
    pub fn api_version(&self) -> u32 {
        0
    }

    #[must_use]
    pub fn device_index(&self) -> u32 {
        self.device_index
    }

    #[must_use]
    pub fn pipeline_cache_path(&self) -> Option<&std::path::Path> {
        self.pipeline_cache.as_deref()
    }

    #[must_use]
    pub fn enabled(&self) -> Enabled {
        self.core.enabled
    }

    #[must_use]
    pub fn info(&self) -> kernels_wgpu::DeviceInfo {
        kernels_wgpu::DeviceInfo {
            vendor: match self.core.info.vendor {
                0x10DE => kernels_wgpu::tuning::Vendor::Nvidia,
                0x1002 => kernels_wgpu::tuning::Vendor::Amd,
                0x8086 => kernels_wgpu::tuning::Vendor::Intel,
                0x106B => kernels_wgpu::tuning::Vendor::Apple,
                _ => kernels_wgpu::tuning::Vendor::Other,
            },
            subgroups: self.core.enabled.subgroups,
            f16: self.core.enabled.f16,
            cores: self.core.cores,
            max_workgroup_invocations: self.core.limits.max_compute_invocations_per_workgroup,
            max_shared_bytes: self.core.limits.max_compute_workgroup_storage_size,
            max_storage_buffer_binding: self.core.limits.max_storage_buffer_binding_size,
        }
    }

    #[must_use]
    pub fn used(&self) -> u64 {
        self.core.allocated.load(Ordering::Relaxed)
    }

    pub fn bind_thread(&self) -> Result<()> {
        Ok(())
    }

    pub fn frame(&self) -> Result<Frame> {
        let core = &self.core;
        let encoder = core
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pie frame"),
            });
        Ok(Frame {
            inner: Arc::new(Inner {
                core: Arc::clone(core),
                encoder: Mutex::new(Some(encoder)),
                uniforms: Mutex::new(UniformRing::default()),
                scratch: Mutex::new(UniformRing::default()),
                index: Mutex::new(None),
                done: AtomicBool::new(false),
                timing: Mutex::new(Vec::new()),
                pass: Mutex::new(None),
            }),
            open: true,
            dispatches: std::cell::Cell::new(0),
        })
    }
}

#[derive(Default)]
pub(crate) struct UniformRing {
    chunks: Vec<wgpu::Buffer>,
    used: u64,
}

pub(crate) struct Inner {
    pub(crate) core: Arc<Core>,
    encoder: Mutex<Option<wgpu::CommandEncoder>>,
    uniforms: Mutex<UniformRing>,
    scratch: Mutex<UniformRing>,
    index: Mutex<Option<wgpu::SubmissionIndex>>,
    done: AtomicBool,
    timing: Mutex<Vec<Timing>>,

    pass: Mutex<Option<wgpu::ComputePass<'static>>>,
}

impl Inner {
    fn harvest(&self) {
        let planes = std::mem::take(
            &mut *self
                .timing
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        for (at, plane) in planes.iter().enumerate() {
            plane.harvest(&self.core, at == 0);
        }
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        let ring = std::mem::take(
            self.uniforms
                .get_mut()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        self.core.return_uniforms(ring.chunks);
        let scratch = std::mem::take(
            self.scratch
                .get_mut()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        self.core.return_scratch(scratch.chunks);
    }
}

pub struct Frame {
    pub(crate) inner: Arc<Inner>,
    open: bool,
    pub(crate) dispatches: std::cell::Cell<u64>,
}

impl Frame {
    pub(crate) fn close_pass(&self) {
        self.inner
            .pass
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
    }

    pub(crate) fn dispatch(
        &self,
        label: &'static str,
        pipeline: &wgpu::ComputePipeline,
        group: &wgpu::BindGroup,
        groups: [u32; 3],
    ) -> Result<()> {
        let mut pass = self
            .inner
            .pass
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if pass.is_none() {
            let mut guard = self
                .inner
                .encoder
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let encoder = guard.as_mut().ok_or(Fault::Device {
                call: "frame",
                why: "this frame was already submitted".to_string(),
            })?;
            *pass = Some(
                encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("pie frame"),
                        timestamp_writes: None,
                    })
                    .forget_lifetime(),
            );
        }
        let pass = pass.as_mut().expect("just opened");
        let _ = label;
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(groups[0].max(1), groups[1].max(1), groups[2].max(1));
        Ok(())
    }

    pub(crate) fn encode<T>(&self, f: impl FnOnce(&mut wgpu::CommandEncoder) -> T) -> Result<T> {
        self.close_pass();
        let mut guard = self
            .inner
            .encoder
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let encoder = guard.as_mut().ok_or(Fault::Device {
            call: "frame",
            why: "this frame was already submitted".to_string(),
        })?;
        Ok(f(encoder))
    }

    pub(crate) fn uniform_slot(&self, bytes: u64) -> (wgpu::Buffer, u64) {
        let core = &self.inner.core;
        let align = u64::from(core.limits.min_uniform_buffer_offset_alignment).max(16);
        let mut ring = self
            .inner
            .uniforms
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let at = ring.used.next_multiple_of(align);
        if ring.chunks.is_empty() || at + bytes > UNIFORM_CHUNK {
            ring.chunks.push(core.uniform_chunk());
            ring.used = bytes;
            return (ring.chunks.last().expect("just pushed").clone(), 0);
        }
        ring.used = at + bytes;
        (ring.chunks.last().expect("non-empty").clone(), at)
    }

    pub(crate) fn timestamp_slot(&self, name: &str) -> Option<(wgpu::QuerySet, u32)> {
        let core = &self.inner.core;
        if !core.enabled.timestamps {
            return None;
        }
        let mut planes = self
            .inner
            .timing
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let first = planes.is_empty();
        if planes
            .last()
            .is_none_or(|plane| 2 * plane.names.len() as u32 + 2 > TIMING_QUERIES - 2)
        {
            planes.push(Timing::open(core));
        }
        let plane = planes.last_mut().expect("just opened");
        let at = plane.names.len() as u32;
        plane.names.push(name.to_string());
        let set = plane.set.clone();
        if first {
            drop(planes);
            let _ = self.encode(|encoder| encoder.write_timestamp(&set, TIMING_QUERIES - 2));
        }
        Some((set, 2 * at))
    }

    pub(crate) fn scratch_slot(&self, bytes: u64) -> (wgpu::Buffer, u64) {
        let core = &self.inner.core;
        let align = u64::from(core.limits.min_storage_buffer_offset_alignment).max(256);
        let mut ring = self
            .inner
            .scratch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let at = ring.used.next_multiple_of(align);
        if ring.chunks.is_empty() || at + bytes > SCRATCH_CHUNK {
            ring.chunks.push(core.scratch_chunk(bytes));
            ring.used = bytes;
            return (ring.chunks.last().expect("just pushed").clone(), 0);
        }
        ring.used = at + bytes;
        (ring.chunks.last().expect("non-empty").clone(), at)
    }

    pub fn copy(
        &mut self,
        source: &super::Buffer,
        source_at: u64,
        into: &super::Buffer,
        into_at: u64,
        len: u64,
    ) -> Result<()> {
        source.span(source_at, len)?;
        into.span(into_at, len)?;
        if len == 0 {
            return Ok(());
        }
        if !source_at.is_multiple_of(4) || !into_at.is_multiple_of(4) || !len.is_multiple_of(4) {
            return Err(Fault::Device {
                call: "copy_buffer_to_buffer",
                why: format!("copy of {len} bytes at {source_at} -> {into_at} is not 4-aligned"),
            });
        }
        crate::encode::record_copy();
        self.encode(|encoder| {
            encoder.copy_buffer_to_buffer(
                &source.slab().buffer,
                source_at,
                &into.slab().buffer,
                into_at,
                Some(len),
            );
        })
    }

    pub fn fill_zero(&mut self, into: &super::Buffer, at: u64, len: u64) -> Result<()> {
        into.span(at, len)?;
        if len == 0 {
            return Ok(());
        }
        let start = at & !3;
        let end = (at + len).next_multiple_of(4).min(into.slab().size);
        self.encode(|encoder| {
            encoder.clear_buffer(&into.slab().buffer, start, Some(end - start));
        })
    }

    #[must_use]
    pub fn dispatches(&self) -> u64 {
        self.dispatches.get()
    }

    pub fn flush(&self) -> Result<()> {
        self.close_pass();
        let core = &self.inner.core;
        let taken = self
            .inner
            .encoder
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        let Some(encoder) = taken else {
            return Err(Fault::Device {
                call: "flush",
                why: "this frame was already submitted".to_string(),
            });
        };
        core.submit_once(encoder)?;
        *self
            .inner
            .encoder
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(
            core.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pie frame"),
                }),
        );
        Ok(())
    }

    fn submit(&mut self) -> Result<()> {
        if !self.open {
            return Ok(());
        }
        self.open = false;
        self.close_pass();
        let core = &self.inner.core;
        let taken = self
            .inner
            .encoder
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        let Some(mut encoder) = taken else {
            return Ok(());
        };
        for (at, timing) in self
            .inner
            .timing
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .enumerate()
            .filter(|(_, plane)| !plane.names.is_empty())
        {
            let n = timing.names.len() as u32;
            encoder.resolve_query_set(&timing.set, 0..2 * n, &timing.resolve, 0);
            encoder.copy_buffer_to_buffer(
                &timing.resolve,
                0,
                &timing.readback,
                0,
                Some(u64::from(n) * 16),
            );
            if at == 0 {
                encoder.write_timestamp(&timing.set, TIMING_QUERIES - 1);
                encoder.resolve_query_set(
                    &timing.set,
                    TIMING_QUERIES - 2..TIMING_QUERIES,
                    &timing.resolve,
                    SPAN_AT,
                );
                encoder.copy_buffer_to_buffer(
                    &timing.resolve,
                    SPAN_AT,
                    &timing.readback,
                    SPAN_AT,
                    Some(16),
                );
            }
        }
        let started = std::time::Instant::now();
        let finished = encoder.finish();
        crate::encode::record_read_phase(0, started.elapsed().as_nanos() as u64);
        let index = core.queue.submit(std::iter::once(finished));
        crate::encode::record_submit(started.elapsed().as_nanos() as u64, 0);
        let inner = Arc::clone(&self.inner);
        core.queue.on_submitted_work_done(move || {
            inner.done.store(true, Ordering::Release);
        });
        *self
            .inner
            .index
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(index);
        core.take_error("submit")
    }

    pub fn commit(mut self) -> Result<()> {
        self.submit()?;
        Pending {
            inner: Arc::clone(&self.inner),
        }
        .wait()
    }

    pub fn commit_timed(mut self) -> Result<f64> {
        let started = std::time::Instant::now();
        self.submit()?;
        Pending {
            inner: Arc::clone(&self.inner),
        }
        .wait()?;
        Ok(started.elapsed().as_secs_f64())
    }

    pub fn commit_async(
        mut self,
        on_done: Option<Box<dyn Fn(Option<String>) + Send + 'static>>,
    ) -> Result<Pending> {
        self.submit()?;
        let pending = Pending {
            inner: Arc::clone(&self.inner),
        };
        if let Some(on_done) = on_done {
            let watched = Pending {
                inner: Arc::clone(&self.inner),
            };
            std::thread::spawn(move || {
                on_done(watched.wait().err().map(|fault| fault.to_string()));
            });
        }
        Ok(pending)
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        self.open = false;
    }
}

pub struct Pending {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for Pending {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pending").finish()
    }
}

impl Pending {
    #[must_use]
    pub fn landed(&self) -> bool {
        if self.inner.done.load(Ordering::Acquire) {
            return true;
        }
        let _ = self.inner.core.device.poll(wgpu::PollType::Poll);
        self.inner.done.load(Ordering::Acquire)
    }

    pub fn wait(&self) -> Result<()> {
        let index = self
            .inner
            .index
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        match index {
            Some(index) => {
                let started = std::time::Instant::now();
                self.inner.core.wait_for(&index)?;
                crate::encode::record_submit(0, started.elapsed().as_nanos() as u64);
                crate::encode::record_wait_call();
                self.inner.harvest();
                Ok(())
            }
            None => Ok(()),
        }
    }
}
