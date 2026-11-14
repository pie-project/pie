#![allow(unsafe_code)]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::api::DeviceBoot;
use crate::encode::Mark;
use engine_gpu::Owner;
use engine_gpu::error::{Fault, Result};

use super::alloc::{Memory, Raw, Slab};
use super::host::{self, Signal};

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
    GATE.lock().expect("gate lock poisoned")
}

#[cfg(not(target_arch = "wasm32"))]
#[must_use]
pub fn present() -> bool {
    let _gate = gate();
    let instance = instance(wgpu::Backends::all());
    host::block_on(
        "enumerate_adapters",
        instance.enumerate_adapters(wgpu::Backends::all()),
    )
    .is_ok_and(|adapters| !adapters.is_empty())
}

#[cfg(target_arch = "wasm32")]
#[must_use]
pub fn present() -> bool {
    wgpu::Instance::enabled_backend_features().contains(wgpu::Backends::BROWSER_WEBGPU)
}

pub(crate) const DUMMY_BYTES: u64 = 256;

const UNIFORM_CHUNK: u64 = 1 << 20;

const SCRATCH_CHUNK: u64 = 8 << 20;

const TIMING_QUERIES: u32 = 4096;

const SPAN_AT: u64 = (TIMING_QUERIES as u64) * 8;

pub(crate) struct Timing {
    set: wgpu::QuerySet,
    resolve: Held,
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
            resolve: core.held(
                &wgpu::BufferDescriptor {
                    label: Some("pie timestamps resolve"),
                    size: bytes,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                },
                engine_gpu::books::owners::TIMESTAMPS,
            ),
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
        let mapped = Signal::new();
        let deliver = mapped.clone();
        slice.map_async(wgpu::MapMode::Read, move |r| deliver.notify(r));
        core.kick();
        if !matches!(mapped.park("timestamps"), Ok(Ok(()))) {
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

    pub(crate) spent: Mutex<Vec<wgpu::Buffer>>,

    poller: host::Poller,

    pub(crate) uniforms: Mutex<Vec<Held>>,
    pub(crate) scratch: Mutex<Vec<Held>>,

    pub(crate) last_error: Arc<Mutex<Option<String>>>,
    pub(crate) books: Arc<engine_gpu::Books>,
    pub(crate) name: String,
    pub(crate) cores: Option<u32>,
    /// `None` without the subgroup feature.
    pub(crate) subgroup_min: Option<u32>,
    /// The device-local heap's bytes, where the adapter reports one; else 0.
    pub(crate) device_local: u64,
    adapter: wgpu::Adapter,
    heap: Option<Heap>,
    /// The device's bytes as the host states them (`DeviceBoot::device_memory`).
    stated: Option<u64>,
}

impl Core {
    /// The device bytes free to this process now: the heap's budget less its
    /// usage where the driver reports them, else the heap less every
    /// reservation (blind to other processes). Where wgpu reports no heap
    /// (Metal, DX12, GL, a browser): the size the host states less every
    /// reservation, else the pool's own binding ceiling, four segments of
    /// the largest storage binding.
    fn free(&self) -> u64 {
        let Some(heap) = self.heap else {
            if let Some(stated) = self.stated {
                return stated.saturating_sub(engine_gpu::books::process_reserved());
            }
            let binding = self
                .limits
                .max_buffer_size
                .min(self.limits.max_storage_buffer_binding_size);
            return binding.saturating_mul(kernels_slang::SEGMENTS as u64);
        };
        match vulkan_budget(&self.adapter, heap) {
            Some((budget, usage)) => budget.saturating_sub(usage),
            None => heap
                .bytes
                .saturating_sub(engine_gpu::books::process_reserved()),
        }
    }

    /// What the driver (wgpu's allocator blocks among it) holds for this
    /// process in the heap beside every reservation of this process, where it
    /// reports it.
    fn outside(&self) -> u64 {
        self.heap
            .and_then(|heap| vulkan_budget(&self.adapter, heap))
            .map_or(0, |(_, usage)| {
                usage.saturating_sub(engine_gpu::books::process_reserved())
            })
    }

    /// Whether the driver reports the heap's usage, so [`Core::outside`]
    /// measures what it holds.
    fn measures_outside(&self) -> bool {
        self.heap
            .is_some_and(|heap| vulkan_budget(&self.adapter, heap).is_some())
    }

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
            .expect("last error lock poisoned")
            .take();
        match taken {
            Some(why) => Err(Fault::Wgpu { what, why }),
            None => Ok(()),
        }
    }

    pub(crate) fn kick(&self) {
        self.poller.kick();
    }

    pub(crate) fn wait_for(&self, _index: &wgpu::SubmissionIndex) -> Result<()> {
        let done = Signal::new();
        let landed = done.clone();
        self.queue.on_submitted_work_done(move || landed.notify(()));
        self.kick();
        done.park("Device::poll")?;
        self.take_error("submission")
    }

    pub(crate) fn destroy_spent(&self) {
        let spent = std::mem::take(&mut *self.spent.lock().expect("spent lock poisoned"));
        for buffer in spent {
            buffer.destroy();
        }
    }

    pub(crate) fn spend(&self, buffer: wgpu::Buffer) {
        self.spent.lock().expect("spent lock poisoned").push(buffer);
    }

    pub(crate) fn submit_once(&self, encoder: wgpu::CommandEncoder) -> Result<()> {
        let index = self.queue.submit(std::iter::once(encoder.finish()));
        self.wait_for(&index)
    }

    /// A buffer outside any slab, entered in the books under `owner` while
    /// it stands.
    pub(crate) fn held(&self, descriptor: &wgpu::BufferDescriptor<'_>, owner: Owner) -> Held {
        let buffer = self.device.create_buffer(descriptor);
        self.books.take(owner, buffer.size());
        Held {
            buffer,
            books: Arc::clone(&self.books),
            owner,
        }
    }

    pub(crate) fn uniform_chunk(&self) -> Held {
        if let Some(chunk) = self.uniforms.lock().expect("uniforms lock poisoned").pop() {
            return chunk;
        }
        self.held(
            &wgpu::BufferDescriptor {
                label: Some("pie uniforms"),
                size: UNIFORM_CHUNK,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
            engine_gpu::books::owners::UNIFORMS,
        )
    }

    pub(crate) fn return_uniforms(&self, chunks: Vec<Held>) {
        self.uniforms
            .lock()
            .expect("uniforms lock poisoned")
            .extend(chunks);
    }

    pub(crate) fn scratch_chunk(&self, bytes: u64) -> Held {
        if bytes <= SCRATCH_CHUNK
            && let Some(chunk) = self.scratch.lock().expect("scratch lock poisoned").pop()
        {
            return chunk;
        }
        self.held(
            &wgpu::BufferDescriptor {
                label: Some("pie scratch"),
                size: bytes.max(SCRATCH_CHUNK),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
            engine_gpu::books::owners::FRAME_SCRATCH,
        )
    }

    pub(crate) fn return_scratch(&self, chunks: Vec<Held>) {
        self.scratch
            .lock()
            .expect("scratch lock poisoned")
            .extend(chunks.into_iter().filter(|c| c.size() == SCRATCH_CHUNK));
    }
}

/// A buffer the device gave outside any slab (a frame's uniform or scratch
/// chunk, a timestamp plane, a bind group's uniforms), entered in the books
/// while it stands.
pub(crate) struct Held {
    buffer: wgpu::Buffer,
    books: Arc<engine_gpu::Books>,
    owner: Owner,
}

impl std::ops::Deref for Held {
    type Target = wgpu::Buffer;

    fn deref(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        self.books.give(self.owner, self.buffer.size());
    }
}

#[derive(Clone)]
pub struct Context {
    pub(crate) core: Arc<Core>,
    /// The operator's cap on what the load takes, all of it together.
    budget: Option<u64>,
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
            .field("budget", &self.budget)
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

/// The device-local heap device reservations take, where the adapter is a
/// Vulkan one: its index, its bytes, and whether the driver reports a
/// budget of it (`VK_EXT_memory_budget`, which wgpu enables where offered).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Heap {
    index: usize,
    bytes: u64,
    budgeted: bool,
}

#[cfg(not(any(target_vendor = "apple", target_arch = "wasm32")))]
fn vulkan_facts(adapter: &wgpu::Adapter) -> (Option<Heap>, Option<u32>) {
    use ash::vk;
    // SAFETY: the guard is only read through, and dropped before the adapter;
    // nothing here destroys the instance or physical device it hands out.
    let Some(hal) = (unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }) else {
        return (None, None);
    };
    let instance = hal.shared_instance().raw_instance();
    let physical = hal.raw_physical_device();
    // SAFETY: `physical` is a live handle of `instance`, both kept alive by
    // `hal`, and the call only fills the returned struct.
    let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
    let heaps = memory.memory_heap_count as usize;
    // SAFETY: as above — a query on a live physical device of `instance`.
    let exts =
        unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap_or_default();
    let budgeted = exts
        .iter()
        .any(|e| e.extension_name_as_c_str() == Ok(ash::ext::memory_budget::NAME));
    let local = (0..heaps)
        .filter(|&i| {
            memory.memory_heaps[i]
                .flags
                .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
        })
        .max_by_key(|&i| memory.memory_heaps[i].size)
        .map(|index| Heap {
            index,
            bytes: memory.memory_heaps[index].size,
            budgeted,
        });
    // SAFETY: as above; `shader_cores` chains onto `p2` only the struct of an
    // extension `exts` says this physical device supports.
    let cores = kernels_slang::cores::shader_cores(&exts, &mut |p2| unsafe {
        instance.get_physical_device_properties2(physical, p2);
    });
    (local, cores)
}

#[cfg(any(target_vendor = "apple", target_arch = "wasm32"))]
fn vulkan_facts(_adapter: &wgpu::Adapter) -> (Option<Heap>, Option<u32>) {
    (None, None)
}

/// `heap`'s budget for this process and what this process holds of it, as
/// the Vulkan driver under the adapter reports them now.
#[cfg(not(any(target_vendor = "apple", target_arch = "wasm32")))]
fn vulkan_budget(adapter: &wgpu::Adapter, heap: Heap) -> Option<(u64, u64)> {
    use ash::vk;
    if !heap.budgeted {
        return None;
    }
    // SAFETY: the guard is only read through, and dropped before the adapter.
    let hal = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }?;
    let instance = hal.shared_instance().raw_instance();
    let physical = hal.raw_physical_device();
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut props = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
    // SAFETY: `physical` is a live handle of `instance`, both kept alive by
    // `hal`; the budget struct is chained only where the physical device
    // offers `VK_EXT_memory_budget`, and both structs outlive the call.
    unsafe { instance.get_physical_device_memory_properties2(physical, &mut props) };
    Some((
        budget.heap_budget[heap.index],
        budget.heap_usage[heap.index],
    ))
}

#[cfg(any(target_vendor = "apple", target_arch = "wasm32"))]
fn vulkan_budget(_adapter: &wgpu::Adapter, _heap: Heap) -> Option<(u64, u64)> {
    None
}

pub struct Handed {
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

fn backends_of(boot: &DeviceBoot) -> wgpu::Backends {
    boot.backends
        .as_deref()
        .map_or(wgpu::Backends::all(), wgpu::Backends::from_comma_list)
}

fn preference_of(boot: &DeviceBoot) -> wgpu::PowerPreference {
    match boot.power_preference.as_str() {
        "low-power" | "low_power" => wgpu::PowerPreference::LowPower,
        "none" => wgpu::PowerPreference::None,
        _ => wgpu::PowerPreference::HighPerformance,
    }
}

#[must_use]
pub fn wanted_features(offered: wgpu::Features) -> wgpu::Features {
    let wanted = wgpu::Features::SUBGROUP
        | wgpu::Features::SHADER_F16
        | wgpu::Features::TIMESTAMP_QUERY
        | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
        | wgpu::Features::MAPPABLE_PRIMARY_BUFFERS
        | wgpu::Features::PIPELINE_CACHE;
    offered & wanted
}

pub async fn request_device(adapter: &wgpu::Adapter) -> Result<(wgpu::Device, wgpu::Queue)> {
    let info = adapter.get_info();
    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("pie"),
            required_features: wanted_features(adapter.features()),
            required_limits: adapter.limits(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|e| Fault::NoDevice {
            detail: format!("request_device on {}: {e}", info.name),
        })
}

#[cfg(not(target_arch = "wasm32"))]
async fn request_adapter(boot: &DeviceBoot) -> Result<wgpu::Adapter> {
    let backends = backends_of(boot);
    let preference = preference_of(boot);
    let instance = instance(backends);
    let mut adapters = instance.enumerate_adapters(backends).await;
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
    adapters
        .into_iter()
        .nth(boot.adapter_index as usize)
        .ok_or_else(|| Fault::NoDevice {
            detail: format!(
                "no adapter at index {} (seen: {roster:?})",
                boot.adapter_index
            ),
        })
}

#[cfg(target_arch = "wasm32")]
async fn request_adapter(boot: &DeviceBoot) -> Result<wgpu::Adapter> {
    let _ = rank;
    let instance = instance(backends_of(boot));
    instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: preference_of(boot),
            ..wgpu::RequestAdapterOptions::default()
        })
        .await
        .map_err(|e| Fault::NoDevice {
            detail: format!("request_adapter: {e}"),
        })
}

pub async fn request(boot: &DeviceBoot) -> Result<Handed> {
    let adapter = request_adapter(boot).await?;
    let (device, queue) = request_device(&adapter).await?;
    Ok(Handed {
        adapter,
        device,
        queue,
    })
}

impl Context {
    /// The device `boot` names, of which a load takes what it has, no more
    /// than `allowance.budget` where the operator states one.
    pub fn bind(boot: &DeviceBoot, allowance: engine::Allowance) -> Result<Context> {
        let _gate = gate();
        let handed = host::block_on("request_device", request(boot))??;
        Context::adopt(boot, allowance, handed)
    }

    /// A device the host already holds, of which a load takes what it has,
    /// no more than `allowance.budget` where stated. wgpu reports no memory
    /// of its own: what
    /// the device has is read from the Vulkan driver under the adapter
    /// where there is one ([`Core::free`] says what stands in elsewhere).
    pub fn adopt(
        boot: &DeviceBoot,
        allowance: engine::Allowance,
        handed: Handed,
    ) -> Result<Context> {
        let budget = allowance.budget;
        let Handed {
            adapter,
            device,
            queue,
        } = handed;
        let info = adapter.get_info();
        let features = device.features();
        let limits = device.limits();
        let last_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        {
            let sink = Arc::clone(&last_error);
            device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
                let text = error.to_string();
                eprintln!("wgpu: {text}");
                *sink.lock().expect("sink lock poisoned") = Some(text);
            }));
        }
        let (heap, sm) = vulkan_facts(&adapter);
        let device_local = heap
            .map(|heap| heap.bytes)
            .or(boot.device_memory)
            .unwrap_or(0);
        let enabled = Enabled {
            subgroups: features.contains(wgpu::Features::SUBGROUP),
            f16: features.contains(wgpu::Features::SHADER_F16),
            timestamps: features.contains(wgpu::Features::TIMESTAMP_QUERY),
            mappable_primary: features.contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS),
            pipeline_cache: features.contains(wgpu::Features::PIPELINE_CACHE),
        };
        // A slab is bound whole, so no larger than one storage binding takes:
        // past it a fire refuses the view rather than reading a cut slab.
        let max_buffer = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size);
        let subgroup_min = enabled.subgroups.then_some(info.subgroup_min_size);
        let poller = host::Poller::start(device.clone());
        let core = Arc::new(Core {
            device,
            queue,
            name: info.name.clone(),
            info,
            limits,
            enabled,
            spent: Mutex::new(Vec::new()),
            poller,
            uniforms: Mutex::new(Vec::new()),
            scratch: Mutex::new(Vec::new()),
            last_error,
            books: Arc::new(engine_gpu::Books::new()),
            cores: sm,
            subgroup_min,
            device_local,
            adapter,
            heap,
            stated: boot.device_memory,
        });
        let dummy = Raw::new(
            &core,
            DUMMY_BYTES,
            Memory::Device,
            engine_gpu::books::owners::DUMMY,
        )?;
        let context = Context {
            budget,
            max_buffer,
            core,
            dummy,
            pipeline_cache: boot.pipeline_cache.clone(),
            device_index: boot.adapter_index,
        };

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

    /// What the device has for this load, measured now: every reservation,
    /// what the driver holds beside them and what is free, capped by the
    /// operator's budget.
    #[must_use]
    pub fn capacity(&self) -> u64 {
        let core = &self.core;
        let measured = core.books.device() + core.outside() + core.free();
        self.budget.map_or(measured, |budget| measured.min(budget))
    }

    /// What the driver holds for this process beside every reservation.
    #[must_use]
    pub fn outside(&self) -> u64 {
        self.core.outside()
    }

    /// Whether [`Context::outside`] is measured: the driver reports the
    /// heap's usage.
    #[must_use]
    pub fn measures_outside(&self) -> bool {
        self.core.measures_outside()
    }

    #[must_use]
    pub fn max_buffer(&self) -> u64 {
        self.max_buffer
    }

    #[must_use]
    pub fn cores(&self) -> u32 {
        self.core.cores.unwrap_or(crate::chip::UNREPORTED_CORES)
    }

    /// The alignment a guest table takes in its pool slot: 16 at least, and
    /// what a storage or uniform binding's offset asks.
    #[must_use]
    pub fn table_align(&self) -> u64 {
        let limits = &self.core.limits;
        u64::from(
            limits
                .min_storage_buffer_offset_alignment
                .max(limits.min_uniform_buffer_offset_alignment),
        )
        .max(engine_gpu::guest::held::TABLE_ALIGN)
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
    pub fn chip(&self) -> crate::chip::Device {
        crate::chip::Device {
            name: self.core.name.clone(),
            vendor: profile::vendor::of_pci(self.core.info.vendor),
            subgroup_min: self.core.subgroup_min,
            cores: self.core.cores,
            f16: self.core.enabled.f16,
        }
    }

    #[must_use]
    pub fn storage_buffers(&self) -> u32 {
        self.core.limits.max_storage_buffers_per_shader_stage
    }

    /// What the capacity has left now, past every reservation and the
    /// driver's own.
    #[must_use]
    pub fn left(&self) -> u64 {
        self.capacity()
            .saturating_sub(self.core.books.device())
            .saturating_sub(self.core.outside())
    }

    /// The device's books: every reservation under its owner.
    #[must_use]
    pub fn books(&self) -> &engine_gpu::Books {
        &self.core.books
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
    chunks: Vec<Held>,
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
        let planes = std::mem::take(&mut *self.timing.lock().expect("timing lock poisoned"));
        for (at, plane) in planes.iter().enumerate() {
            plane.harvest(&self.core, at == 0);
        }
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        let ring = std::mem::take(self.uniforms.get_mut().expect("uniforms lock poisoned"));
        self.core.return_uniforms(ring.chunks);
        let scratch = std::mem::take(self.scratch.get_mut().expect("scratch lock poisoned"));
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
        self.inner.pass.lock().expect("pass lock poisoned").take();
    }

    pub(crate) fn dispatch(
        &self,
        label: &'static str,
        pipeline: &wgpu::ComputePipeline,
        group: &wgpu::BindGroup,
        groups: [u32; 3],
    ) -> Result<()> {
        let mut pass = self.inner.pass.lock().expect("pass lock poisoned");
        if pass.is_none() {
            let mut guard = self.inner.encoder.lock().expect("encoder lock poisoned");
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
        let mut guard = self.inner.encoder.lock().expect("encoder lock poisoned");
        let encoder = guard.as_mut().ok_or(Fault::Device {
            call: "frame",
            why: "this frame was already submitted".to_string(),
        })?;
        Ok(f(encoder))
    }

    pub(crate) fn uniform_slot(&self, bytes: u64) -> (wgpu::Buffer, u64) {
        let core = &self.inner.core;
        let align = u64::from(core.limits.min_uniform_buffer_offset_alignment).max(16);
        let mut ring = self.inner.uniforms.lock().expect("uniforms lock poisoned");
        let at = ring.used.next_multiple_of(align);
        if ring.chunks.is_empty() || at + bytes > UNIFORM_CHUNK {
            ring.chunks.push(core.uniform_chunk());
            ring.used = bytes;
            return ((*ring.chunks.last().expect("just pushed")).clone(), 0);
        }
        ring.used = at + bytes;
        ((*ring.chunks.last().expect("non-empty")).clone(), at)
    }

    pub(crate) fn timestamp_slot(&self, name: &str) -> Option<(wgpu::QuerySet, u32)> {
        let core = &self.inner.core;
        if !core.enabled.timestamps {
            return None;
        }
        let mut planes = self.inner.timing.lock().expect("timing lock poisoned");
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
        let mut ring = self.inner.scratch.lock().expect("scratch lock poisoned");
        let at = ring.used.next_multiple_of(align);
        if ring.chunks.is_empty() || at + bytes > SCRATCH_CHUNK {
            ring.chunks.push(core.scratch_chunk(bytes));
            ring.used = bytes;
            return ((*ring.chunks.last().expect("just pushed")).clone(), 0);
        }
        ring.used = at + bytes;
        ((*ring.chunks.last().expect("non-empty")).clone(), at)
    }

    /// Writes `bytes` into `into` from `at` in queue order: the queue lands
    /// the write after every frame submitted before and before this one
    /// (WebGPU's queue timeline), so no fire in flight reads it changed and
    /// no kernel of this frame reads it unwritten.
    pub(crate) fn update(&self, into: &super::Buffer, at: u64, bytes: &[u8]) -> Result<()> {
        into.write_shared(at, bytes)
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
        let (from, to) = (&source.slab().buffer, &into.slab().buffer);
        // wgpu refuses a copy whose ends are one buffer: it goes through
        // scratch, which also reads overlapping ends as they were.
        let stage = (from == to).then(|| self.scratch_slot(len));
        self.encode(|encoder| match &stage {
            None => encoder.copy_buffer_to_buffer(from, source_at, to, into_at, Some(len)),
            Some((scratch, at)) => {
                encoder.copy_buffer_to_buffer(from, source_at, scratch, *at, Some(len));
                encoder.copy_buffer_to_buffer(scratch, *at, to, into_at, Some(len));
            }
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

    /// [`Frame::flush`]: the queue orders a write after every submission
    /// before it (WebGPU's queue timeline), so landing this frame settles
    /// what the host writes next through the queue.
    pub fn flush_settled(&self) -> Result<()> {
        self.flush()
    }

    pub fn flush(&self) -> Result<()> {
        self.close_pass();
        let core = &self.inner.core;
        let taken = self
            .inner
            .encoder
            .lock()
            .expect("encoder lock poisoned")
            .take();
        let Some(encoder) = taken else {
            return Err(Fault::Device {
                call: "flush",
                why: "this frame was already submitted".to_string(),
            });
        };
        core.submit_once(encoder)?;
        *self.inner.encoder.lock().expect("encoder lock poisoned") = Some(
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
            .expect("encoder lock poisoned")
            .take();
        let Some(mut encoder) = taken else {
            return Ok(());
        };
        for (at, timing) in self
            .inner
            .timing
            .lock()
            .expect("timing lock poisoned")
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
        let started = Mark::now();
        let finished = encoder.finish();
        crate::encode::record_read_phase(0, started.ns());
        let index = core.queue.submit(std::iter::once(finished));
        crate::encode::record_submit(started.ns(), 0);
        let inner = Arc::clone(&self.inner);
        core.queue.on_submitted_work_done(move || {
            inner.done.store(true, Ordering::Release);
        });
        core.kick();
        *self.inner.index.lock().expect("index lock poisoned") = Some(index);
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
        let started = Mark::now();
        self.submit()?;
        Pending {
            inner: Arc::clone(&self.inner),
        }
        .wait()?;
        Ok(started.secs())
    }

    pub fn commit_async(mut self, on_done: Option<OnDone>) -> Result<Pending> {
        self.submit()?;
        let pending = Pending {
            inner: Arc::clone(&self.inner),
        };
        if let Some(on_done) = on_done {
            self.watch(on_done);
        }
        Ok(pending)
    }

    fn watch(&self, on_done: OnDone) {
        let inner = Arc::clone(&self.inner);
        self.inner.core.queue.on_submitted_work_done(move || {
            let refused = inner.core.take_error("submission").err();
            on_done(refused.map(|fault| fault.to_string()));
        });
        self.inner.core.kick();
    }
}

pub use engine_gpu::backend::OnDone;

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
    pub fn wait(&self) -> Result<()> {
        if !self.inner.done.load(Ordering::Acquire) {
            let done = Signal::new();
            let landed = done.clone();
            self.inner
                .core
                .queue
                .on_submitted_work_done(move || landed.notify(()));
            self.inner.core.kick();
            done.park("Pending::wait")?;
        }
        crate::encode::record_wait_call();
        self.inner.harvest();
        self.inner.core.take_error("submission")
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::device::Buffer;

    #[test]
    fn a_copy_between_two_spans_of_one_buffer_lands() {
        if !present() {
            return;
        }
        let device = Context::bind(&DeviceBoot::default(), engine::Allowance::WHOLE).unwrap();
        let mut buffer = Buffer::zeroed(&device, 64, engine_gpu::books::owners::TEST).unwrap();
        let ramp: Vec<u8> = (1..=24).collect();
        buffer.write(0, &ramp).unwrap();
        let mut frame = device.frame().unwrap();
        frame.copy(&buffer, 0, &buffer, 32, 16).unwrap();
        // Overlapping ends read the source as it was before the copy.
        frame.copy(&buffer, 0, &buffer, 8, 16).unwrap();
        frame.commit().unwrap();
        let mut got = [0u8; 64];
        buffer.read(0, &mut got).unwrap();
        assert_eq!(got[32..48], ramp[..16]);
        assert_eq!(got[8..24], ramp[..16]);
    }
}
