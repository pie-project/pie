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

/// A spanned frame's two timestamps ([`Context::spanned`]): the set they
/// are written to, the buffer it resolves into, and the host's copy;
/// `busy` from the frame that writes them opening until their readback is
/// read and unmapped, so one frame writes them at a time.
pub(crate) struct Span {
    set: wgpu::QuerySet,
    resolve: Held,
    readback: wgpu::Buffer,
    busy: AtomicBool,
}

/// The timestamps timed frames write ([`Context::stamps`]), laid at load:
/// sets, each with its resolve and readback, each written by one frame at a
/// time, where the device enabled timestamp queries; none where it did not
/// (a timed frame is then timed by the host's clock).
#[derive(Clone)]
pub struct Stamps(Vec<Arc<Span>>);

impl std::fmt::Debug for Stamps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Stamps").field(&self.0.len()).finish()
    }
}

/// The bytes a span's buffers take: two timestamps, at the resolve's
/// alignment.
const SPAN_BYTES: u64 = wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT;

#[derive(Clone, Copy, Debug, Default)]
pub struct Enabled {
    pub subgroups: bool,
    pub f16: bool,
    pub timestamps: bool,
}

pub(crate) struct Core {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) info: wgpu::AdapterInfo,
    pub(crate) limits: wgpu::Limits,
    pub(crate) enabled: Enabled,

    pub(crate) spent: Mutex<Vec<wgpu::Buffer>>,

    poller: host::Poller,
    /// The submissions on the queue not yet seen done ([`Core::submit`]).
    in_flight: Arc<std::sync::atomic::AtomicU64>,

    /// The uniform buffer frames write their dispatches' scalars to, made
    /// at load, a region of it each frame in flight; and the regions free.
    pub(crate) uniform_buffer: std::sync::OnceLock<Held>,
    pub(crate) uniforms: Mutex<Vec<u64>>,
    pub(crate) scratch: Mutex<Vec<Held>>,
    /// The uniform regions and scratch chunks stocked, free or held.
    uniform_stock: std::sync::atomic::AtomicUsize,
    scratch_stock: std::sync::atomic::AtomicUsize,

    pub(crate) last_error: Arc<Mutex<Option<String>>>,
    /// The device's loss, once seen ([`Lost`]): every call after fails by
    /// name ([`Fault::DeviceLost`]).
    pub(crate) lost: Lost,
    pub(crate) books: Arc<engine_gpu::Books>,
    pub(crate) name: String,
    /// `None` without the subgroup feature.
    pub(crate) subgroup_min: Option<u32>,
    /// What the load takes, all of it together: the device memory read at
    /// load ([`super::memory::at_load`]), or `[engine.memory]
    /// budget_device` where stated.
    pub(crate) budget: u64,
    /// Whether `budget` is the host's room ([`super::memory::Source::HostAvailable`]):
    /// the device shares the host's memory, so what the host can still give
    /// is read afresh while serving.
    pub(crate) shares_host: bool,
    /// The most bytes one reservation takes on this device
    /// ([`super::memory::reservation_bound`]): every buffer made is no
    /// larger, and one past it is refused by name, never made invalid.
    pub(crate) bound: u64,
}

/// A wgpu device's loss, once seen, and where: its lost callback (a driver
/// fault, or `Device::destroy`), or a mapping it refused (wgpu refuses a map
/// of a buffer the load made only once the device is lost: a submission the
/// driver faulted). Nothing recovers it: once set it stands, and every call
/// on the device after fails as [`Fault::DeviceLost`], never as the
/// validation errors a lost device's objects raise.
#[derive(Clone, Default)]
pub(crate) struct Lost(Arc<Mutex<Option<(&'static str, String)>>>);

impl Lost {
    /// The device was lost at `what` for `why`: the first loss seen stands.
    pub(crate) fn mark(&self, what: &'static str, why: impl ToString) {
        let mut lost = self.0.lock().expect("lost lock poisoned");
        if lost.is_none() {
            let why = why.to_string();
            tracing::error!(what, %why, "wgpu: the device was lost; every call on it fails from here");
            *lost = Some((what, why));
        }
    }

    /// The loss, by name, where the device was lost.
    pub(crate) fn fault(&self) -> Option<Fault> {
        self.0
            .lock()
            .expect("lost lock poisoned")
            .as_ref()
            .map(|(what, why)| Fault::DeviceLost {
                what,
                why: why.clone(),
            })
    }

    /// `Ok` while the device stands; its loss by name once lost.
    pub(crate) fn alive(&self) -> Result<()> {
        self.fault().map_or(Ok(()), Err)
    }
}

impl Core {
    /// `what` refused for `why`: a mapping refused marks the device lost,
    /// and a lost device answers with its loss.
    pub(crate) fn fault(&self, what: &'static str, why: impl ToString) -> Fault {
        if what == "map_async" {
            self.lost.mark(what, why.to_string());
        }
        self.lost.fault().unwrap_or_else(|| Fault::Wgpu {
            what,
            why: why.to_string(),
        })
    }

    /// `Ok` while the device stands; its loss by name once lost.
    pub(crate) fn alive(&self) -> Result<()> {
        self.lost.alive()
    }

    pub(crate) fn take_error(&self, what: &'static str) -> Result<()> {
        let taken = self
            .last_error
            .lock()
            .expect("last error lock poisoned")
            .take();
        // A lost device's objects raise validation errors of their own (a
        // buffer made on it is invalid): the loss is what names them.
        self.alive()?;
        match taken {
            Some(why) => Err(Fault::Wgpu { what, why }),
            None => Ok(()),
        }
    }

    pub(crate) fn kick(&self) {
        self.poller.kick();
    }

    /// `buffer` submitted, counted on the queue until its work is done (the
    /// poller polls the device from the kick until its queue is empty).
    pub(crate) fn submit(&self, buffer: wgpu::CommandBuffer) -> wgpu::SubmissionIndex {
        self.in_flight
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        let index = self.queue.submit(std::iter::once(buffer));
        let in_flight = Arc::clone(&self.in_flight);
        self.queue.on_submitted_work_done(move || {
            in_flight.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        });
        self.kick();
        index
    }

    /// Whether every submission on the queue is done, read now without
    /// waiting.
    pub(crate) fn idle(&self) -> bool {
        self.in_flight.load(std::sync::atomic::Ordering::Acquire) == 0
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
        let index = self.submit(encoder.finish());
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

    /// Makes the uniform buffer frames write their dispatches' scalars to,
    /// with the device: one buffer of a region a frame, so a bind group made
    /// once binds every frame's scalars at a dynamic offset into its region,
    /// for the most frames in flight any load of this engine takes
    /// ([`engine::runahead::Runahead::MAX_FRAMES`]) and the one beside them.
    pub(crate) fn stock_uniforms(&self) {
        let frames = u64::from(engine::runahead::Runahead::MAX_FRAMES) + 1;
        let buffer = self.held(
            &wgpu::BufferDescriptor {
                label: Some("pie uniforms"),
                size: UNIFORM_CHUNK * frames,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
            engine_gpu::books::owners::UNIFORMS,
        );
        if self.uniform_buffer.set(buffer).is_ok() {
            self.uniform_stock.fetch_add(
                usize::try_from(frames).unwrap_or(usize::MAX),
                Ordering::Relaxed,
            );
            self.return_uniforms((0..frames).map(|at| at * UNIFORM_CHUNK).collect());
        }
    }

    /// Stocks a scratch chunk for each of `frames` frames in flight, at load:
    /// a frame takes at most one, and gives it back when it is dropped.
    pub(crate) fn stock_rings(&self, frames: usize) {
        let scratch = (0..frames).map(|_| {
            self.held(
                &wgpu::BufferDescriptor {
                    label: Some("pie scratch"),
                    size: self.pooled_scratch(),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
                engine_gpu::books::owners::FRAME_SCRATCH,
            )
        });
        self.scratch_stock.fetch_add(frames, Ordering::Relaxed);
        self.return_scratch(scratch.collect());
    }

    /// The uniform buffer's region a frame takes, its first byte.
    pub(crate) fn uniform_chunk(&self) -> Result<u64> {
        self.uniforms
            .lock()
            .expect("uniforms lock poisoned")
            .pop()
            .ok_or_else(|| {
                let stock = self.uniform_stock.load(Ordering::Relaxed);
                Fault::Device {
                    call: "uniform region",
                    why: format!("none free ({stock} held by frames in flight, stock {stock})"),
                }
            })
    }

    pub(crate) fn return_uniforms(&self, chunks: Vec<u64>) {
        self.uniforms
            .lock()
            .expect("uniforms lock poisoned")
            .extend(chunks);
    }

    /// A scratch chunk for a frame, stocked at load: one holds at most
    /// [`Core::pooled_scratch`] bytes, refused by name past it.
    pub(crate) fn scratch_chunk(&self, bytes: u64) -> Result<Held> {
        let pooled = self.pooled_scratch();
        if bytes > pooled {
            return Err(Fault::Ceiling {
                what: "bytes of one scratch reservation",
                need: bytes,
                have: pooled,
            });
        }
        self.scratch
            .lock()
            .expect("scratch lock poisoned")
            .pop()
            .ok_or_else(|| {
                let stock = self.scratch_stock.load(Ordering::Relaxed);
                Fault::Device {
                    call: "scratch chunk",
                    why: format!("none free ({stock} held by frames in flight, stock {stock})"),
                }
            })
    }

    pub(crate) fn return_scratch(&self, chunks: Vec<Held>) {
        self.scratch
            .lock()
            .expect("scratch lock poisoned")
            .extend(chunks);
    }

    /// The size of a scratch chunk the ring keeps between frames: no larger
    /// than one reservation takes.
    fn pooled_scratch(&self) -> u64 {
        SCRATCH_CHUNK.min(self.bound)
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
    /// The most bytes one reservation the load plans takes: the device's
    /// bound, or a test's cap under it.
    max_buffer: u64,
    pub(crate) dummy: Slab,
    device_index: u32,
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context")
            .field("name", &self.core.name)
            .field("backend", &self.core.info.backend)
            .field("budget", &self.core.budget)
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
        | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
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

/// Refuses a load on a target with no native wgpu backend: macOS, as on
/// wgpu's Metal backend a large first submission never completes.
fn native() -> Result<()> {
    if cfg!(target_os = "macos") {
        return Err(Fault::NoDevice {
            detail: "engine type wgpu has no backend on macOS in this build; use type = \"metal\""
                .to_string(),
        });
    }
    Ok(())
}

impl Context {
    /// The device `boot` names, of which a load takes what it states it
    /// has, or `allowance.budget` where stated.
    pub fn bind(boot: &DeviceBoot, allowance: engine::Allowance) -> Result<Context> {
        native()?;
        let _gate = gate();
        let handed = host::block_on("request_device", request(boot))??;
        Context::adopt(boot, allowance, handed)
    }

    /// A device the host already holds, of which a load takes what its
    /// adapter states it has, or `allowance.budget` where stated; refused
    /// where neither states it.
    pub fn adopt(
        boot: &DeviceBoot,
        allowance: engine::Allowance,
        handed: Handed,
    ) -> Result<Context> {
        let Handed {
            adapter,
            device,
            queue,
        } = handed;
        let info = adapter.get_info();
        let reading = super::memory::at_load(&adapter, allowance)?;
        tracing::info!(device = %info.name, "{reading}");
        let budget = reading.bytes;
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
        let lost = Lost::default();
        {
            let lost = lost.clone();
            device.set_device_lost_callback(move |reason, message| {
                lost.mark("device lost callback", format!("{reason:?}: {message}"));
            });
        }
        let enabled = Enabled {
            subgroups: features.contains(wgpu::Features::SUBGROUP),
            f16: features.contains(wgpu::Features::SHADER_F16),
            // Every timestamp this backend writes is written into an
            // encoder: both features, or none is written.
            timestamps: features.contains(
                wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
            ),
        };
        // A slab is bound whole, so no larger than one storage binding takes
        // (past it a fire refuses the view rather than reading a cut slab),
        // nor than one allocation backs: past either, the buffer is made
        // invalid, so every reservation is refused there by name instead.
        let bound =
            super::memory::reservation_bound(&limits, super::memory::allocation_limit(&adapter));
        let max_buffer = bound;
        // A test's cap narrows what the load plans by, never what the device
        // can make.
        #[cfg(feature = "test-limits")]
        let max_buffer = max_buffer.min(boot.max_buffer.unwrap_or(u64::MAX));
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
            in_flight: Arc::default(),
            uniform_buffer: std::sync::OnceLock::new(),
            uniforms: Mutex::new(Vec::new()),
            scratch: Mutex::new(Vec::new()),
            uniform_stock: std::sync::atomic::AtomicUsize::new(0),
            scratch_stock: std::sync::atomic::AtomicUsize::new(0),
            last_error,
            lost,
            books: Arc::new(engine_gpu::Books::new()),
            subgroup_min,
            budget,
            shares_host: reading.source == super::memory::Source::HostAvailable,
            bound,
        });
        let dummy = Raw::new(
            &core,
            DUMMY_BYTES,
            Memory::Device,
            engine_gpu::books::owners::DUMMY,
        )?;
        // Every kernel dispatch reads its scalars from this buffer, the
        // first at the load's rates pass, so it is made with the device.
        core.stock_uniforms();
        let context = Context {
            max_buffer,
            core,
            dummy,
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

    /// What the device has for this load, as read at load.
    #[must_use]
    pub fn capacity(&self) -> u64 {
        self.core.budget
    }

    /// Where the load's memory was read from the host's room, the reader of
    /// what the host can still give the process.
    #[must_use]
    pub fn host_room(&self) -> Option<engine::HostRoom> {
        self.core
            .shares_host
            .then(engine_gpu::host_memory::room_reader)
    }

    /// What the driver holds for this process beside every reservation:
    /// wgpu reports none.
    #[must_use]
    pub fn outside(&self) -> u64 {
        0
    }

    /// Whether [`Context::outside`] is measured: never under wgpu.
    #[must_use]
    pub fn measures_outside(&self) -> bool {
        false
    }

    #[must_use]
    pub fn max_buffer(&self) -> u64 {
        self.max_buffer
    }

    #[must_use]
    pub fn cores(&self) -> u32 {
        crate::chip::UNREPORTED_CORES
    }

    /// The workgroup memory the adapter states a kernel may take.
    #[must_use]
    pub fn workgroup_bytes(&self) -> u32 {
        self.core.limits.max_compute_workgroup_storage_size
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
    pub fn enabled(&self) -> Enabled {
        self.core.enabled
    }

    #[must_use]
    pub fn chip(&self) -> crate::chip::Device {
        crate::chip::Device {
            name: self.core.name.clone(),
            vendor: profile::vendor::of_pci(self.core.info.vendor),
            subgroup_min: self.core.subgroup_min,
            cores: None,
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
        self.capacity().saturating_sub(self.core.books.device())
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
        core.alive()?;
        let encoder = core
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pie frame"),
            });
        Ok(Frame {
            inner: Arc::new(Inner {
                core: Arc::clone(core),
                encoder: Mutex::new(Some(encoder)),
                uniforms: Mutex::new(UniformRegion::default()),
                scratch: Mutex::new(UniformRing::default()),
                index: Mutex::new(None),
                done: AtomicBool::new(false),
                timing: Mutex::new(Vec::new()),
                span: Mutex::new(None),
                pass: Mutex::new(None),
                time: engine_gpu::submit::DeviceTime::default(),
                wall: AtomicBool::new(false),
            }),
            open: true,
            dispatches: std::cell::Cell::new(0),
            bind_only: false,
        })
    }

    /// A frame whose device time [`Frame::commit_spanned`] reads back: a
    /// timestamp written into its encoder before its work and one after it,
    /// where the device enabled timestamp queries; a plain frame elsewhere.
    pub fn spanned(&self) -> Result<Frame> {
        let stamps = self.stamps(1)?;
        self.spanned_into(&stamps)
    }

    /// The timestamps timed frames write, laid now ([`Stamps`]): `sets`
    /// sets of two, each with its resolve and its readback, where the
    /// device enabled timestamp queries, else none.
    pub fn stamps(&self, sets: u32) -> Result<Stamps> {
        let core = &self.core;
        core.alive()?;
        if !core.enabled.timestamps {
            return Ok(Stamps(Vec::new()));
        }
        Ok(Stamps((0..sets.max(1)).map(|_| self.span()).collect()))
    }

    /// One set of two timestamps, its resolve and its readback.
    fn span(&self) -> Arc<Span> {
        let core = &self.core;
        Arc::new(Span {
            set: core.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("pie span"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            }),
            resolve: core.held(
                &wgpu::BufferDescriptor {
                    label: Some("pie span resolve"),
                    size: SPAN_BYTES,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                },
                engine_gpu::books::owners::TIMESTAMPS,
            ),
            readback: core.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pie span readback"),
                size: SPAN_BYTES,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            busy: AtomicBool::new(false),
        })
    }

    /// A frame writing its two timestamps into a set of `stamps` (laid
    /// before, [`Context::stamps`]) no frame before still reads (its
    /// readback not yet unmapped); a plain frame where they hold none.
    /// Where every set is still read the frame goes out unspanned and says
    /// so, its time told unread.
    pub fn spanned_into(&self, stamps: &Stamps) -> Result<Frame> {
        let frame = self.frame()?;
        if stamps.0.is_empty() {
            return Ok(frame);
        }
        let Some(span) = stamps.0.iter().find(|span| {
            span.busy
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
        }) else {
            tracing::warn!(
                sets = stamps.0.len(),
                "timestamps: every laid set is still read for the frames before; this frame \
                 goes out untimed"
            );
            frame.inner.time.unread();
            return Ok(frame);
        };
        frame.encode(|encoder| encoder.write_timestamp(&span.set, 0))?;
        *frame.inner.span.lock().expect("span lock poisoned") = Some(Arc::clone(span));
        Ok(frame)
    }

    /// Whether every submission made on the device so far is done, read now
    /// without waiting: the queue is empty.
    #[must_use]
    pub fn idle(&self) -> bool {
        self.core.idle()
    }

    /// A frame whose device time [`Pending::device_time`] reads once its
    /// work is done, filled where that is seen (its work-done callback),
    /// never by a wait of its own: spanned ([`Context::spanned`]) where the
    /// device enabled timestamp queries, else timed by the host's clock
    /// from its submit to its work done.
    pub fn timed_frame(&self, stamps: &Stamps) -> Result<Frame> {
        if !stamps.0.is_empty() {
            return self.spanned_into(stamps);
        }
        let frame = self.frame()?;
        frame.inner.wall.store(true, Ordering::Release);
        Ok(frame)
    }

    /// A frame a load's bind walk walks into ([`Frame::bind_only`]).
    pub fn bind_frame(&self) -> Result<Frame> {
        let mut frame = self.frame()?;
        frame.bind_only = true;
        Ok(frame)
    }
}

/// The uniform buffer's region a frame holds, the bytes of it taken, and
/// their image on the host: the frame writes the image through the queue
/// once before it submits ([`Frame::land_uniforms`]), not a write a dispatch.
#[derive(Default)]
pub(crate) struct UniformRegion {
    region: Option<u64>,
    used: u64,
    image: Vec<u8>,
    /// The image's bytes already written through the queue.
    written: u64,
}

#[derive(Default)]
pub(crate) struct UniformRing {
    chunks: Vec<Held>,
    used: u64,
    /// The end of the most the frame's encoder has staged through its
    /// chunk: a copy gives its slot back once encoded, and a staging the
    /// queue writes lies past this, where no command of the frame writes.
    high: u64,
}

pub(crate) struct Inner {
    pub(crate) core: Arc<Core>,
    encoder: Mutex<Option<wgpu::CommandEncoder>>,
    uniforms: Mutex<UniformRegion>,
    scratch: Mutex<UniformRing>,
    index: Mutex<Option<wgpu::SubmissionIndex>>,
    done: AtomicBool,
    timing: Mutex<Vec<Timing>>,
    span: Mutex<Option<Arc<Span>>>,

    pass: Mutex<Option<wgpu::ComputePass<'static>>>,

    /// A timed frame's device time, filled once its work is done
    /// ([`Context::timed_frame`]).
    time: engine_gpu::submit::DeviceTime,
    /// A timed frame on a device that writes no timestamps: its time is the
    /// host's clock from its submit to its work done.
    wall: AtomicBool,
}

/// Submitted frames whose dispatches wrote timestamps: a served frame lands
/// by callback and is never waited on, so its times are read here, off the
/// callback, once its work is done ([`harvest_landed`]).
static TIMED: Mutex<Vec<Arc<Inner>>> = Mutex::new(Vec::new());

/// Reads the times of every timed frame whose work has landed into the
/// kernel profile; the rest stay for a later call.
pub(crate) fn harvest_landed() {
    let landed: Vec<Arc<Inner>> = {
        let mut timed = TIMED.lock().expect("timed frames lock poisoned");
        let (landed, pending): (Vec<_>, Vec<_>) = timed
            .drain(..)
            .partition(|inner| inner.done.load(Ordering::Acquire));
        *timed = pending;
        landed
    };
    for inner in landed {
        inner.harvest();
    }
}

impl Inner {
    fn harvest(&self) {
        let planes = std::mem::take(&mut *self.timing.lock().expect("timing lock poisoned"));
        for (at, plane) in planes.iter().enumerate() {
            plane.harvest(&self.core, at == 0);
        }
    }
}

impl Inner {
    /// Gives the frame's uniform region and scratch chunk back to the
    /// load's stock: once it is submitted, or where it is dropped
    /// unsubmitted.
    fn release(&self) {
        let ring = std::mem::take(&mut *self.uniforms.lock().expect("uniforms lock poisoned"));
        self.core.return_uniforms(ring.region.into_iter().collect());
        let scratch = std::mem::take(&mut *self.scratch.lock().expect("scratch lock poisoned"));
        self.core.return_scratch(scratch.chunks);
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.release();
        // Timestamps a frame dropped unsubmitted held are free again: a
        // submitted frame took them, and frees them once read.
        if let Some(span) = self.span.get_mut().expect("span lock poisoned").take() {
            span.busy.store(false, Ordering::Release);
        }
    }
}

pub struct Frame {
    pub(crate) inner: Arc<Inner>,
    open: bool,
    pub(crate) dispatches: std::cell::Cell<u64>,
    /// A load's bind walk's ([`engine_gpu::serve::Shell::walk_binds`]): its
    /// sink builds and keeps each dispatch's bind group, encodes nothing, and
    /// the frame is never submitted.
    pub(crate) bind_only: bool,
}

impl Frame {
    pub(crate) fn close_pass(&self) {
        self.inner.pass.lock().expect("pass lock poisoned").take();
    }

    /// Dispatches `pipeline` over `group`, read at its dynamic `offsets`.
    pub(crate) fn dispatch_at(
        &self,
        label: &'static str,
        pipeline: &wgpu::ComputePipeline,
        group: &wgpu::BindGroup,
        offsets: &[u32],
        groups: [u32; 3],
    ) -> Result<()> {
        let groups = engine_gpu::workgroups(
            label,
            groups,
            [self.inner.core.limits.max_compute_workgroups_per_dimension; 3],
        )?;
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
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, offsets);
        pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
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

    /// `bytes` of this frame's region of the uniform buffer: the buffer and
    /// the slot's first byte in it.
    pub(crate) fn uniform_slot(&self, bytes: u64) -> Result<(wgpu::Buffer, u64)> {
        let core = &self.inner.core;
        let buffer = core.uniform_buffer.get().ok_or(Fault::Device {
            call: "uniform region",
            why: "the device made no uniform buffer".into(),
        })?;
        let align = u64::from(core.limits.min_uniform_buffer_offset_alignment).max(16);
        let mut ring = self.inner.uniforms.lock().expect("uniforms lock poisoned");
        let region = match ring.region {
            Some(region) => region,
            None => {
                let region = core.uniform_chunk()?;
                ring.region = Some(region);
                ring.used = 0;
                ring.image.clear();
                ring.written = 0;
                region
            }
        };
        let at = ring.used.next_multiple_of(align);
        if at + bytes > UNIFORM_CHUNK {
            return Err(Fault::Ceiling {
                what: "bytes of uniforms one frame writes (its region)",
                need: at + bytes,
                have: UNIFORM_CHUNK,
            });
        }
        ring.used = at + bytes;
        Ok(((**buffer).clone(), region + at))
    }

    /// Puts `bytes` at byte `at` of the uniform buffer (a slot
    /// [`Frame::uniform_slot`] gave), in the frame's image of its region.
    pub(crate) fn write_uniform(&self, at: u64, bytes: &[u8]) -> Result<()> {
        let mut ring = self.inner.uniforms.lock().expect("uniforms lock poisoned");
        let region = ring.region.ok_or(Fault::Device {
            call: "write_uniform",
            why: "a uniform written before its frame took a region".into(),
        })?;
        let from = usize::try_from(at - region).unwrap_or(usize::MAX);
        let to = from + bytes.len();
        if ring.image.len() < to {
            ring.image.resize(to, 0);
        }
        ring.image[from..to].copy_from_slice(bytes);
        Ok(())
    }

    /// Writes what the frame's image holds past what it wrote, in one write
    /// through the queue, which lands it before the submission that follows.
    fn land_uniforms(&self) {
        let core = &self.inner.core;
        let mut ring = self.inner.uniforms.lock().expect("uniforms lock poisoned");
        let (Some(region), Some(buffer)) = (ring.region, core.uniform_buffer.get()) else {
            return;
        };
        let from = usize::try_from(ring.written).unwrap_or(usize::MAX);
        if from < ring.image.len() {
            core.queue
                .write_buffer(buffer, region + ring.written, &ring.image[from..]);
            ring.written = ring.image.len() as u64;
        }
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

    /// The bytes one slot of this frame's scratch can still hold: its
    /// chunk's room past what it took, or a whole chunk before it takes one.
    fn scratch_room(&self) -> u64 {
        let core = &self.inner.core;
        let align = u64::from(core.limits.min_storage_buffer_offset_alignment).max(256);
        let ring = self.inner.scratch.lock().expect("scratch lock poisoned");
        match ring.chunks.last() {
            None => core.pooled_scratch(),
            Some(chunk) => chunk
                .size()
                .saturating_sub(ring.used.next_multiple_of(align)),
        }
    }

    /// `bytes` of this frame's scratch the frame's encoder writes: the
    /// chunk and the slot's first byte in it.
    pub(crate) fn scratch_slot(&self, bytes: u64) -> Result<(wgpu::Buffer, u64)> {
        self.scratch_take(bytes, false)
    }

    /// `bytes` of this frame's scratch the queue writes, which lands before
    /// any command of the frame runs: past every slot the encoder has
    /// written, so no command of the frame writes over it.
    pub(crate) fn scratch_slot_queued(&self, bytes: u64) -> Result<(wgpu::Buffer, u64)> {
        self.scratch_take(bytes, true)
    }

    fn scratch_take(&self, bytes: u64, queued: bool) -> Result<(wgpu::Buffer, u64)> {
        let core = &self.inner.core;
        let align = u64::from(core.limits.min_storage_buffer_offset_alignment).max(256);
        let mut ring = self.inner.scratch.lock().expect("scratch lock poisoned");
        if ring.chunks.is_empty() {
            ring.chunks.push(core.scratch_chunk(bytes)?);
        }
        let from = if queued {
            ring.used.max(ring.high)
        } else {
            ring.used
        };
        let at = from.next_multiple_of(align);
        let room = ring.chunks.last().map_or(0, |chunk| chunk.size());
        if at + bytes > room {
            return Err(Fault::Ceiling {
                what: "bytes one frame stages through scratch (its chunk)",
                need: at + bytes,
                have: room,
            });
        }
        ring.used = at + bytes;
        ring.high = ring.high.max(ring.used);
        Ok(((*ring.chunks.last().expect("non-empty")).clone(), at))
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
        if from != to {
            return self.encode(|encoder| {
                encoder.copy_buffer_to_buffer(from, source_at, to, into_at, Some(len))
            });
        }
        // wgpu refuses a copy whose ends are one buffer: it goes through
        // scratch a piece at a time, every piece through one slot (a frame's
        // encoder runs its copies in order), so a copy of any length stages
        // in what room the frame's chunk has. Where the destination lies past
        // an overlapping source the pieces go from the end, so each reads its
        // source as it was.
        // The slot is given back once the copy is encoded: a later slot the
        // encoder writes runs after the copy's pieces, and one the queue
        // writes lies past it (`scratch_slot_queued`).
        let piece = len.min(self.scratch_room() & !3).max(4);
        let given = self
            .inner
            .scratch
            .lock()
            .expect("scratch lock poisoned")
            .used;
        let (scratch, at) = self.scratch_slot(piece)?;
        let mut starts: Vec<u64> = (0..len).step_by(piece as usize).collect();
        if into_at > source_at && into_at < source_at + len {
            starts.reverse();
        }
        self.encode(|encoder| {
            for start in starts {
                let bytes = piece.min(len - start);
                encoder.copy_buffer_to_buffer(from, source_at + start, &scratch, at, Some(bytes));
                encoder.copy_buffer_to_buffer(&scratch, at, to, into_at + start, Some(bytes));
            }
        })?;
        self.inner
            .scratch
            .lock()
            .expect("scratch lock poisoned")
            .used = given;
        Ok(())
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

    /// [`engine_gpu::backend::Frame::submit_segment`]: the frame's encoder
    /// submitted unawaited and a new one opened. WebGPU's queue orders the
    /// next submission after it, and the frame's uniform region and scratch
    /// chunk stay the frame's (given back at its commit).
    pub fn submit_segment(&self) -> Result<()> {
        self.close_pass();
        self.land_uniforms();
        let core = &self.inner.core;
        core.alive()?;
        let taken = self
            .inner
            .encoder
            .lock()
            .expect("encoder lock poisoned")
            .take();
        let Some(encoder) = taken else {
            return Err(Fault::Device {
                call: "submit_segment",
                why: "this frame was already submitted".to_string(),
            });
        };
        core.submit(encoder.finish());
        core.kick();
        *self.inner.encoder.lock().expect("encoder lock poisoned") = Some(
            core.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pie frame"),
                }),
        );
        core.take_error("submit_segment")
    }

    pub fn flush(&self) -> Result<()> {
        self.close_pass();
        self.land_uniforms();
        let core = &self.inner.core;
        core.alive()?;
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
        self.land_uniforms();
        let core = &self.inner.core;
        core.alive()?;
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
        let index = core.submit(finished);
        crate::encode::record_submit(started.ns(), 0);
        // What the frame staged through goes back to the load's stock now,
        // submitted, not when its work completes: the queue runs what any
        // frame writes or dispatches after this one's (WebGPU's queue
        // timeline), so the next frame to take them cannot reach them before
        // this one's work is done with them. That holds for these two only
        // because they are device-only: a scratch chunk (STORAGE and copies)
        // and a uniform region (UNIFORM, written through the queue), neither
        // mappable, so the host never reads one back. What the host maps or
        // reads (the readout mirrors, the guest readback buffers, the read
        // stagings) stays held until its work is done. The stock then covers
        // the frames a step encodes at once, however many it has in flight:
        // a step's copies (a same-buffer copy stages through scratch) and
        // its walk, each submitted before the next is opened.
        self.inner.release();
        if !self
            .inner
            .timing
            .lock()
            .expect("timing lock poisoned")
            .is_empty()
        {
            TIMED
                .lock()
                .expect("timed frames lock poisoned")
                .push(Arc::clone(&self.inner));
        }
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

    /// Submits and waits: the device's time between the two timestamps a
    /// [`Context::spanned`] frame writes, ns; `None` for another frame.
    pub fn commit_spanned(mut self) -> Result<Option<u64>> {
        let span = self.inner.span.lock().expect("span lock poisoned").take();
        if let Some(span) = &span
            && self.open
        {
            self.encode(|encoder| {
                encoder.write_timestamp(&span.set, 1);
                encoder.resolve_query_set(&span.set, 0..2, &span.resolve, 0);
                encoder.copy_buffer_to_buffer(&span.resolve, 0, &span.readback, 0, Some(16));
            })?;
        }
        self.submit()?;
        Pending {
            inner: Arc::clone(&self.inner),
        }
        .wait()?;
        let Some(span) = span else {
            return Ok(None);
        };
        let core = &self.inner.core;
        let slice = span.readback.slice(0..16);
        let mapped = Signal::new();
        let deliver = mapped.clone();
        slice.map_async(wgpu::MapMode::Read, move |r| deliver.notify(r));
        core.kick();
        mapped
            .park("span readback")?
            .map_err(|e| Fault::Device {
                call: "span readback",
                why: e.to_string(),
            })?;
        let ticks = {
            let view = slice.get_mapped_range().map_err(|e| Fault::Device {
                call: "span readback",
                why: e.to_string(),
            })?;
            let word = |i: usize| {
                let mut b = [0u8; 8];
                b.copy_from_slice(&view[i * 8..i * 8 + 8]);
                u64::from_le_bytes(b)
            };
            word(1).saturating_sub(word(0))
        };
        span.readback.unmap();
        span.busy.store(false, Ordering::Release);
        let period = f64::from(core.queue.get_timestamp_period());
        Ok(Some((ticks as f64 * period) as u64))
    }

    pub fn commit_async(mut self, on_done: Option<OnDone>) -> Result<Pending> {
        // A timed frame ([`Context::timed_frame`]): its closing timestamp
        // resolved into its readback before the submit, the readback mapped
        // once the work is done; or the host's clock across the submit.
        let span = self.inner.span.lock().expect("span lock poisoned").take();
        if let Some(span) = &span
            && self.open
        {
            self.encode(|encoder| {
                encoder.write_timestamp(&span.set, 1);
                encoder.resolve_query_set(&span.set, 0..2, &span.resolve, 0);
                encoder.copy_buffer_to_buffer(&span.resolve, 0, &span.readback, 0, Some(16));
            })?;
        }
        let wall = self.inner.wall.load(Ordering::Acquire);
        let submitted = std::time::Instant::now();
        self.submit()?;
        if let Some(span) = span {
            let (core, time) = (Arc::clone(&self.inner.core), self.inner.time.clone());
            self.inner.core.queue.on_submitted_work_done(move || {
                read_span(core, span, time);
            });
            self.inner.core.kick();
        } else if wall {
            let time = self.inner.time.clone();
            self.inner.core.queue.on_submitted_work_done(move || {
                time.set(submitted.elapsed().as_nanos() as u64);
            });
            self.inner.core.kick();
        }
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

/// A timed frame's span read once its work is done: its readback mapped
/// and its two timestamps' distance filled into `time`, or `time` told
/// unread where the map failed.
fn read_span(core: Arc<Core>, span: Arc<Span>, time: engine_gpu::submit::DeviceTime) {
    let period = f64::from(core.queue.get_timestamp_period());
    let readback = span.readback.clone();
    readback.slice(0..16).map_async(wgpu::MapMode::Read, move |mapped| {
        if mapped.is_err() {
            span.busy.store(false, Ordering::Release);
            time.unread();
            return;
        }
        let slice = span.readback.slice(0..16);
        let ticks = match slice.get_mapped_range() {
            Ok(view) => {
                let word = |i: usize| {
                    let mut b = [0u8; 8];
                    b.copy_from_slice(&view[i * 8..i * 8 + 8]);
                    u64::from_le_bytes(b)
                };
                Some(word(1).saturating_sub(word(0)))
            }
            Err(_) => None,
        };
        span.readback.unmap();
        span.busy.store(false, Ordering::Release);
        match ticks {
            Some(ticks) => time.set((ticks as f64 * period) as u64),
            None => time.unread(),
        }
    });
    core.kick();
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
    /// The frame's device time, filled once its work is done where it was
    /// opened timed ([`Context::timed_frame`]); never filled otherwise.
    #[must_use]
    pub fn device_time(&self) -> engine_gpu::submit::DeviceTime {
        self.inner.time.clone()
    }

    /// Whether the frame has landed, read now without waiting: its
    /// submitted-work-done callback has run (the poller polls the device
    /// from the submit until its queue is empty).
    #[must_use]
    pub fn landed(&self) -> bool {
        self.inner.done.load(Ordering::Acquire)
    }

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

    fn lost_by_name(fault: Fault) {
        assert!(
            matches!(fault, Fault::DeviceLost { .. }),
            "refused as {fault}, not as DeviceLost"
        );
        assert!(fault.to_string().starts_with("DeviceLost: "), "{fault}");
    }

    /// The loss's state: none until marked; the first loss marked stands,
    /// and every fault after names it.
    #[test]
    fn a_loss_once_marked_stands_and_names_every_fault_after() {
        let lost = Lost::default();
        assert!(lost.alive().is_ok());
        lost.mark("map_async", "Validation");
        lost.mark("device lost callback", "a later loss");
        match lost.fault().expect("lost") {
            Fault::DeviceLost { what, why } => {
                assert_eq!(what, "map_async");
                assert_eq!(why, "Validation");
            }
            other => panic!("{other}"),
        }
        lost_by_name(lost.alive().unwrap_err());
    }

    /// A device lost (here by `Device::destroy`, as a driver fault loses
    /// it) fails every later call by name: a buffer, a frame, a write, a
    /// read, a submit, never wgpu's "Buffer with '' label is invalid".
    #[test]
    fn a_lost_device_fails_every_later_call_as_device_lost() {
        if !present() {
            return;
        }
        let allowance = engine::Allowance {
            budget: Some(1 << 30),
        };
        let device = Context::bind(&DeviceBoot::default(), allowance).unwrap();
        device.core().stock_rings(1);
        let owner = engine_gpu::books::owners::TEST;
        let mut buffer = Buffer::zeroed(&device, 64, owner).unwrap();
        // A frame opened before the loss and submitted after it.
        let mut before = device.frame().unwrap();
        before.copy(&buffer, 0, &buffer, 32, 16).unwrap();
        device.core.device.destroy();
        // The lost callback runs as the device is polled.
        let _ = device.core.device.poll(wgpu::PollType::wait_indefinitely());
        lost_by_name(before.commit().err().expect("refused"));
        lost_by_name(Buffer::zeroed(&device, 64, owner).err().expect("refused"));
        lost_by_name(device.frame().err().expect("refused"));
        lost_by_name(buffer.write(0, &[1, 2, 3, 4]).err().expect("refused"));
        let mut got = [0u8; 4];
        lost_by_name(buffer.read(0, &mut got).err().expect("refused"));
    }

    #[test]
    fn a_copy_between_two_spans_of_one_buffer_lands() {
        if !present() {
            return;
        }
        let allowance = engine::Allowance {
            budget: Some(1 << 30),
        };
        let device = Context::bind(&DeviceBoot::default(), allowance).unwrap();
        device.core().stock_rings(1);
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

    /// A copy within one buffer longer than the whole scratch chunk lands,
    /// in pieces through what room the frame's chunk has (some taken first
    /// by a smaller staged copy), its overlapping ends read as they were
    /// whichever side the destination lies on.
    #[test]
    fn a_copy_within_one_buffer_past_its_scratch_chunk_lands_in_pieces() {
        if !present() {
            return;
        }
        let allowance = engine::Allowance {
            budget: Some(1 << 30),
        };
        let device = Context::bind(&DeviceBoot::default(), allowance).unwrap();
        device.core().stock_rings(1);
        let pooled = device.core().pooled_scratch();
        let len = (pooled + pooled / 2) & !3;
        let size = 3 * pooled + 8192;
        let words: Vec<u8> = (0..size / 4)
            .flat_map(|w| (w as u32).wrapping_mul(2_654_435_761).to_le_bytes())
            .collect();
        // (source, destination): past an overlapping source, before one,
        // and apart from it.
        for (from, into) in [(0, 4096), (4096, 0), (0, len + 4096)] {
            let mut buffer =
                Buffer::zeroed(&device, size, engine_gpu::books::owners::TEST).unwrap();
            buffer.write(0, &words).unwrap();
            let mut frame = device.frame().unwrap();
            frame
                .copy(&buffer, size - 64, &buffer, size - 32, 16)
                .unwrap();
            frame.copy(&buffer, from, &buffer, into, len).unwrap();
            // The copy gave its slot back: a view staged after it in the frame
            // still finds the chunk's room.
            frame.scratch_slot(pooled - 4096).unwrap();
            frame.commit().unwrap();
            let mut want = words.clone();
            want.copy_within(
                (size - 64) as usize..(size - 48) as usize,
                (size - 32) as usize,
            );
            want.copy_within(from as usize..(from + len) as usize, into as usize);
            let mut got = vec![0u8; size as usize];
            buffer.read(0, &mut got).unwrap();
            assert!(
                got == want,
                "a copy of {len} bytes from {from} to {into} (scratch chunk {pooled}) did not land \
                 as one copy of the bytes as they were"
            );
        }
        device.core.take_error("the copies").unwrap();
    }

    /// No buffer is made past the device's bound on one reservation: a
    /// slab or a scratch slot past it is refused by name before anything
    /// reaches the device, and nothing invalid is made.
    #[test]
    fn no_buffer_is_made_past_the_bound_of_one_reservation() {
        if !present() {
            return;
        }
        let allowance = engine::Allowance {
            budget: Some(1 << 30),
        };
        let device = Context::bind(&DeviceBoot::default(), allowance).unwrap();
        let bound = device.core.bound;
        assert!(device.max_buffer() <= bound);
        let past = |fault: Fault, named: &str, bound: u64| match fault {
            Fault::Ceiling { what, need, have } => {
                assert_eq!(what, named);
                assert!(need > bound, "{need}");
                assert_eq!(have, bound);
            }
            other => panic!("refused as {other}, not by the bound"),
        };
        let owner = engine_gpu::books::owners::TEST;
        let before = device.core.books.device();
        past(
            Buffer::zeroed(&device, bound + 4, owner)
                .err()
                .expect("refused"),
            "bytes of one device reservation",
            bound,
        );
        assert_eq!(device.core.books.device(), before, "nothing was reserved");
        // Scratch is stocked at load in chunks no larger than one
        // reservation, and a frame takes no more than its chunk.
        device.core.stock_rings(1);
        let pooled = device.core.pooled_scratch();
        assert!(pooled <= bound);
        let held = device.core.books.device();
        let frame = device.frame().unwrap();
        past(
            frame.scratch_slot(pooled + 256).err().expect("refused"),
            "bytes of one scratch reservation",
            pooled,
        );
        assert_eq!(device.core.books.device(), held, "nothing was reserved");
        frame.commit().unwrap();
        device.core.take_error("the refused buffers").unwrap();
    }
}
