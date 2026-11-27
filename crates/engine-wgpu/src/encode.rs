use engine_gpu::{Handles, NIL};
use kernels_slang::{ArgValue, Encode, Error, Fire, Target};

use crate::device::ctx::Frame;
use crate::device::{Api, Context, Pipelines};
use engine_gpu::error::Fault;

static KERNEL_PROFILE: engine_gpu::kernel_profile::KernelProfile =
    engine_gpu::kernel_profile::KernelProfile::new();

static TIMING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

static SHAPES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

static HOST_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[must_use]
pub fn host_encode_ns() -> u64 {
    HOST_NS.load(std::sync::atomic::Ordering::Relaxed)
}

static SUBMIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[must_use]
pub fn host_submit_ns() -> (u64, u64) {
    (
        SUBMIT_NS.load(std::sync::atomic::Ordering::Relaxed),
        WAIT_NS.load(std::sync::atomic::Ordering::Relaxed),
    )
}

static COPIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[must_use]
pub fn host_copies() -> u64 {
    COPIES.load(std::sync::atomic::Ordering::Relaxed)
}

static IO: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

#[must_use]
pub fn host_io() -> (u64, u64, u64, u64) {
    let at = |i: usize| IO[i].load(std::sync::atomic::Ordering::Relaxed);
    (at(0), at(1), at(2), at(3))
}

#[cfg(feature = "wgpu")]
pub(crate) fn record_io(write: bool, ns: u64) {
    let base = if write { 0 } else { 2 };
    IO[base].fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
    IO[base + 1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

static WAIT_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[must_use]
pub fn host_wait_calls() -> u64 {
    WAIT_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(feature = "wgpu")]
pub(crate) fn record_wait_call() {
    WAIT_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

static READ_PHASES: [std::sync::atomic::AtomicU64; 3] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

#[must_use]
pub fn host_read_phases() -> (u64, u64, u64) {
    let at = |i: usize| READ_PHASES[i].load(std::sync::atomic::Ordering::Relaxed);
    (at(0), at(1), at(2))
}

#[cfg(feature = "wgpu")]
pub(crate) fn record_read_phase(phase: usize, ns: u64) {
    READ_PHASES[phase].fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "wgpu")]
pub(crate) fn record_copy() {
    COPIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "wgpu")]
pub(crate) fn record_submit(submit_ns: u64, wait_ns: u64) {
    SUBMIT_NS.fetch_add(submit_ns, std::sync::atomic::Ordering::Relaxed);
    WAIT_NS.fetch_add(wait_ns, std::sync::atomic::Ordering::Relaxed);
}

pub fn profile_timing(on: bool) {
    TIMING.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether frames are timed on the device ([`profile_timing`]).
#[must_use]
pub fn timing_on() -> bool {
    TIMING.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn profile_shapes(on: bool) {
    SHAPES.store(on, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "wgpu")]
#[derive(Clone, Copy)]
pub(crate) struct Mark {
    at: crate::device::host::Instant,
}

#[cfg(feature = "wgpu")]
impl Mark {
    pub(crate) fn now() -> Mark {
        Mark {
            at: crate::device::host::Instant::now(),
        }
    }

    pub(crate) fn ns(self) -> u64 {
        self.at.elapsed().as_nanos() as u64
    }

    pub(crate) fn secs(self) -> f64 {
        self.ns() as f64 * 1e-9
    }
}

#[cfg(feature = "wgpu")]
fn timing() -> bool {
    TIMING.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(feature = "wgpu")]
fn shapes() -> bool {
    SHAPES.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(feature = "wgpu")]
fn label(name: &'static str, fire: Fire, args: &[ArgValue]) -> std::borrow::Cow<'static, str> {
    use std::fmt::Write;
    if !shapes() {
        return std::borrow::Cow::Borrowed(name);
    }
    let mut key = String::with_capacity(name.len() + 48);
    key.push_str(name);
    let _ = write!(
        key,
        " g={}x{}x{}",
        fire.groups[0].max(1),
        fire.groups[1].max(1),
        fire.groups[2].max(1)
    );
    let mut first = true;
    for arg in args {
        let scalar = match *arg {
            ArgValue::I32(v) => v.to_string(),
            ArgValue::U32(v) => v.to_string(),
            ArgValue::F32(v) => format!("{v}"),
            ArgValue::Zeros(words) => format!("0x{words}"),
            ArgValue::Buffer(_) | ArgValue::BufferMut(_) => continue,
        };
        let _ = write!(key, "{}{scalar}", if first { " s=" } else { "," });
        first = false;
    }
    std::borrow::Cow::Owned(key)
}

#[must_use]
pub fn kernel_profile() -> Vec<(String, u64, u64)> {
    KERNEL_PROFILE.rows()
}

#[cfg(feature = "wgpu")]
pub(crate) fn record_timing(name: &str, ns: u64) {
    KERNEL_PROFILE.time(name, ns);
}

pub fn reset_kernel_profile() {
    KERNEL_PROFILE.clear();
    HOST_NS.store(0, std::sync::atomic::Ordering::Relaxed);
    SUBMIT_NS.store(0, std::sync::atomic::Ordering::Relaxed);
    WAIT_NS.store(0, std::sync::atomic::Ordering::Relaxed);
    COPIES.store(0, std::sync::atomic::Ordering::Relaxed);
    for slot in &IO {
        slot.store(0, std::sync::atomic::Ordering::Relaxed);
    }
    for slot in &READ_PHASES {
        slot.store(0, std::sync::atomic::Ordering::Relaxed);
    }
    WAIT_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
    #[cfg(feature = "wgpu")]
    crate::device::pipelines::reset_bind_traffic();
}

#[cfg(feature = "wgpu")]
fn record_kernel(name: &str) {
    KERNEL_PROFILE.fire(name);
}

#[cfg(feature = "wgpu")]
pub struct Sink<'a> {
    device: &'a Context,
    frame: &'a Frame,
    pipelines: &'a Pipelines,
    handles: &'a Handles<Api>,
    outer: engine_gpu::probe::Recording,
}

#[cfg(feature = "wgpu")]
impl Drop for Sink<'_> {
    fn drop(&mut self) {
        engine_gpu::probe::restore(self.outer);
    }
}

#[cfg(feature = "wgpu")]
impl<'a> Sink<'a> {
    #[must_use]
    pub fn new(
        device: &'a Context,
        frame: &'a Frame,
        pipelines: &'a Pipelines,
        handles: &'a Handles<Api>,
    ) -> Sink<'a> {
        let outer = engine_gpu::probe::set_frame(frame);
        Sink {
            device,
            frame,
            pipelines,
            handles,
            outer,
        }
    }

    fn refuse(fire: Fire, fault: Fault) -> Error {
        Error::Backend {
            op: fire.entrypoint,
            detail: fault.to_string(),
        }
    }

    fn backend(fire: Fire, detail: String) -> Error {
        Error::Backend {
            op: fire.entrypoint,
            detail,
        }
    }
}

#[cfg(feature = "wgpu")]
impl Encode for Sink<'_> {
    fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Error> {
        let started = timing().then(Mark::now);
        let out = self.fire_inner(fire, args);
        if let Some(started) = started {
            HOST_NS.fetch_add(started.ns(), std::sync::atomic::Ordering::Relaxed);
        }
        out
    }

    fn absent(&self) -> Result<ArgValue, Error> {
        Ok(ArgValue::Buffer(NIL))
    }

    fn target(&self) -> Target {
        Target::Wgsl
    }
}

#[cfg(feature = "wgpu")]
impl Sink<'_> {
    /// The view an argument that names no buffer binds: the dummy whole,
    /// or, where no two views may overlap (`exact`), the `nils`th window of
    /// it, so a dispatch with several binds no window twice.
    fn dummy_view(&self, exact: bool, nils: &mut u64) -> (wgpu::Buffer, u64, u64, bool) {
        let dummy = &self.device.dummy;
        if !exact {
            return (dummy.buffer.clone(), 0, dummy.size, false);
        }
        let at = *nils % crate::device::ctx::DUMMY_WINDOWS;
        *nils += 1;
        let bytes = crate::device::ctx::DUMMY_BYTES;
        (dummy.buffer.clone(), at * bytes, bytes, false)
    }

    fn fire_inner(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Error> {
        let name = kernels_slang::key(fire.entrypoint, fire.tier).ok_or_else(|| {
            Sink::backend(
                fire,
                format!("not instantiated at the {} tier", fire.tier.name()),
            )
        })?;
        let pipeline = self
            .pipelines
            .get(self.device, fire.file, name)
            .map_err(|fault| Sink::refuse(fire, fault))?;
        let core = self.device.core();
        let limits = &core.limits;
        let groups = engine_gpu::workgroups(
            name,
            fire.groups,
            [limits.max_compute_workgroups_per_dimension; 3],
        )
        .map_err(|fault| Sink::refuse(fire, fault))?;
        let storage_align = u64::from(limits.min_storage_buffer_offset_alignment);
        let max_binding = limits.max_storage_buffer_binding_size;
        let exact_views = crate::device::pipelines::exact_views(core.one_segment());
        // Arguments that name no buffer, as many as seen: each its own
        // window of the dummy where no two views may overlap.
        let mut nils = 0u64;

        let mut views: Vec<(wgpu::Buffer, u64, u64, bool)> = Vec::with_capacity(args.len());

        // A window of a value binds the value from its aligned first byte,
        // the kernel's rebased twin stepping each binding over its element
        // offset here ([`crate::device::pipelines::Rebased`]).
        let mut twin: Option<std::sync::Arc<crate::device::pipelines::Rebased>> = None;
        let mut rebase: Vec<u32> = Vec::new();
        let mut words: Vec<u8> = Vec::with_capacity(64);
        for arg in args {
            match *arg {
                ArgValue::Buffer(handle) | ArgValue::BufferMut(handle) => {
                    let mutable = matches!(arg, ArgValue::BufferMut(_));
                    if handle == NIL {
                        views.push(self.dummy_view(exact_views, &mut nils));
                        continue;
                    }
                    let binding = self.handles.get(handle).ok_or_else(|| {
                        Sink::refuse(
                            fire,
                            Fault::Unbound {
                                what: format!(
                                    "handle {handle} at argument {}, which no row answers",
                                    views.len()
                                ),
                            },
                        )
                    })?;
                    let remaining = binding.extent().min(binding.remaining());
                    if remaining == 0 {
                        if mutable {
                            return Err(Sink::backend(
                                fire,
                                format!("argument {} writes an empty view", views.len()),
                            ));
                        }
                        views.push(self.dummy_view(exact_views, &mut nils));
                        continue;
                    }
                    let view = remaining.next_multiple_of(4).min(
                        binding
                            .buffer()
                            .slab()
                            .size
                            .saturating_sub(binding.offset()),
                    );
                    storage_view_fits(views.len(), view, max_binding)
                        .map_err(|detail| Sink::backend(fire, detail))?;
                    // A view binds its slab from its first byte to the end
                    // (as Vulkan's descriptors do), up to the largest binding,
                    // holding at least the `need` bytes the kernel reads: the
                    // group a dispatch binds names where a value starts, not
                    // how many rows a fire's window cuts of it, so the load's
                    // walk builds the one every fire over it binds.
                    let slab_bytes = binding.buffer().slab().size;
                    // Where no two views of a buffer may overlap (the tab's
                    // WebGPU, `pipelines::aliased`), a view ends where its
                    // value does: still one group for every window of it.
                    let value_end = if exact_views {
                        binding.end().next_multiple_of(4).min(slab_bytes)
                    } else {
                        slab_bytes
                    };
                    let whole = |from: u64, need: u64| {
                        let to_end = (value_end.saturating_sub(from) & !3).min(max_binding & !3);
                        to_end.max(need)
                    };
                    // A window of a value (a class region's rows of it) binds
                    // the value from its aligned first byte and steps the
                    // kernel over the window's lead, so every region of the
                    // value binds the one group whatever row it starts at.
                    let at = binding.offset();
                    let origin = binding.origin().min(at);
                    let base = origin - origin % storage_align;
                    if at == base {
                        views.push((
                            binding.buffer().slab().buffer.clone(),
                            at,
                            whole(at, view),
                            mutable,
                        ));
                        continue;
                    }

                    let slot = views.len();
                    let lead = at - base;
                    // A binding the kernel never reads binds no view at all.
                    if !pipeline.used.get(slot).copied().unwrap_or(false) {
                        views.push((
                            binding.buffer().slab().buffer.clone(),
                            base,
                            whole(base, lead + view),
                            mutable,
                        ));
                        continue;
                    }
                    let step = u64::from(pipeline.stride.get(slot).copied().unwrap_or(0));
                    if step == 0 || !lead.is_multiple_of(step) || lead + view > max_binding {
                        return Err(Sink::backend(
                            fire,
                            format!(
                                "argument {slot} (handle {handle}) is a window at byte {at} of \
                                 a value from byte {origin}, {lead} bytes past its aligned \
                                 first byte {base}: {}",
                                if step == 0 {
                                    "the binding is no runtime-sized array, which no element \
                                     offset steps"
                                        .to_string()
                                } else if !lead.is_multiple_of(step) {
                                    format!("not a whole number of its {step}-byte elements")
                                } else {
                                    format!(
                                        "the value bound from there runs past this device's \
                                         largest binding of {max_binding} bytes"
                                    )
                                }
                            ),
                        ));
                    }
                    if twin.is_none() {
                        twin = Some(
                            self.pipelines
                                .rebased(self.device, &pipeline, self.frame.bind_only)
                                .map_err(|fault| Sink::refuse(fire, fault))?,
                        );
                    }
                    if rebase.len() <= slot {
                        rebase.resize(slot + 1, 0);
                    }
                    rebase[slot] = u32::try_from(lead / step).unwrap_or(u32::MAX);
                    views.push((
                        binding.buffer().slab().buffer.clone(),
                        base,
                        whole(base, lead + view),
                        mutable,
                    ));
                }
                ArgValue::I32(v) => words.extend_from_slice(&v.to_le_bytes()),
                ArgValue::U32(v) => words.extend_from_slice(&v.to_le_bytes()),
                ArgValue::F32(v) => words.extend_from_slice(&v.to_le_bytes()),
                ArgValue::Zeros(count) => words.resize(words.len() + count as usize * 4, 0),
            }
        }

        let declared = pipeline.bindings as usize;
        if let Some(at) = (views.len()..declared).find(|&at| pipeline.used[at]) {
            return Err(Sink::backend(
                fire,
                format!(
                    "the module declares binding {at} but the entry passed only {} buffers",
                    views.len()
                ),
            ));
        }
        for (at, view) in views.iter().enumerate() {
            if at >= declared {
                // A one-segment form's undeclared segments are passed and
                // not bound: the pool lies whole in the first.
                if view.3 && !pipeline.one_segment {
                    return Err(Sink::backend(
                        fire,
                        format!(
                            "argument {at} is a write target the module does not declare (it \
                             declares {declared} bindings)"
                        ),
                    ));
                }
                // Only where it is none, or the first again: a pool of
                // planes of their own (`KvPool::planes`) is not one segment,
                // and the form reads the first in its place.
                if pipeline.one_segment
                    && declared > 0
                    && view.0 != self.device.dummy.buffer
                    && view.0 != views[declared - 1].0
                {
                    return Err(Sink::backend(
                        fire,
                        format!(
                            "argument {at} is a pool segment in a buffer of its own (a pool of \
                             planes of their own), and the kernel's one-segment form binds the \
                             pool in one buffer"
                        ),
                    ));
                }
            } else if pipeline.used[at] && view.3 && pipeline.read_only[at] {
                return Err(Sink::backend(
                    fire,
                    format!(
                        "argument {at} is a write target but the module reads binding {at} only"
                    ),
                ));
            }
        }
        let expected = words.len() as u32;
        let rounded = expected.next_multiple_of(16);
        match pipeline.uniform {
            None if expected > 0 => {
                return Err(Sink::backend(
                    fire,
                    format!(
                        "the entry passes {expected} bytes of scalars but the module declares no uniform block"
                    ),
                ));
            }
            Some(_) if pipeline.push_bytes != expected && pipeline.push_bytes != rounded => {
                return Err(Sink::backend(
                    fire,
                    format!(
                        "the entry passes {expected} bytes of scalars but the module declares a {}-byte uniform block",
                        pipeline.push_bytes
                    ),
                ));
            }
            _ => {}
        }

        let stated = fire.group.iter().product::<u32>();
        if stated > 1 && fire.group != pipeline.local {
            return Err(Sink::backend(
                fire,
                format!(
                    "the entry states workgroup {:?} but the module was expanded with {:?}",
                    fire.group, pipeline.local
                ),
            ));
        }

        let keyed: Vec<crate::device::pipelines::View> = views
            .iter()
            .map(|(buffer, offset, size, _)| (buffer.clone(), *offset, *size))
            .collect();
        let uniform_bytes = u64::from(pipeline.push_bytes.max(rounded).max(16));
        // The twin where a view was rebased: its offsets block beside the
        // kernel's scalars.
        let rebased = twin;
        if self.frame.bind_only {
            // A load's bind walk: the group a dispatch over these views binds
            // is built and kept (and the kernel's rebased twin, where a view
            // is a window); nothing is encoded.
            self.pipelines
                .bind_group(core, &pipeline, &keyed, uniform_bytes, rebased.as_deref())
                .map_err(|fault| Sink::refuse(fire, fault))?;
            return Ok(());
        }
        let group = self
            .pipelines
            .bind_group(core, &pipeline, &keyed, uniform_bytes, rebased.as_deref())
            .map_err(|fault| Sink::refuse(fire, fault))?;
        // The scalars in this frame's slot, the group reading them there.
        let mut offsets: Vec<u32> = Vec::with_capacity(1);
        if pipeline.uniform.is_some() {
            let (_, at) = self
                .frame
                .uniform_slot(uniform_bytes)
                .map_err(|fault| Sink::refuse(fire, fault))?;
            let mut padded = words.clone();
            padded.resize(uniform_bytes as usize, 0);
            self.frame
                .write_uniform(at, &padded)
                .map_err(|fault| Sink::refuse(fire, fault))?;
            offsets.push(u32::try_from(at).map_err(|_| {
                Sink::backend(
                    fire,
                    format!("a uniform slot at byte {at} is past a u32 offset"),
                )
            })?);
        }
        // The rebased views' element offsets, in their block's slot: bound
        // past the kernel's uniform, so its dynamic offset follows.
        if let Some(rebased) = &rebased {
            let bytes = u64::from(rebased.bytes);
            let (_, at) = self
                .frame
                .uniform_slot(bytes)
                .map_err(|fault| Sink::refuse(fire, fault))?;
            rebase.resize(bytes as usize / 4, 0);
            let block: Vec<u8> = rebase.iter().flat_map(|lead| lead.to_le_bytes()).collect();
            self.frame
                .write_uniform(at, &block)
                .map_err(|fault| Sink::refuse(fire, fault))?;
            offsets.push(u32::try_from(at).map_err(|_| {
                Sink::backend(
                    fire,
                    format!("a uniform slot at byte {at} is past a u32 offset"),
                )
            })?);
        }

        let compute = rebased
            .as_ref()
            .map_or(&pipeline.pipeline, |rebased| &rebased.pipeline);
        let key = label(name, fire, args);
        if let Some((set, at)) = timing().then(|| self.frame.timestamp_slot(&key)).flatten() {
            self.frame
                .encode(|encoder| {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(name),
                        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                            query_set: &set,
                            beginning_of_pass_write_index: Some(at),
                            end_of_pass_write_index: Some(at + 1),
                        }),
                    });
                    pass.set_pipeline(compute);
                    pass.set_bind_group(0, &group, &offsets);
                    pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
                })
                .map_err(|fault| Sink::refuse(fire, fault))?;
        } else {
            self.frame
                .dispatch_at(name, compute, &group, &offsets, groups)
                .map_err(|fault| Sink::refuse(fire, fault))?;
        }
        self.frame.dispatches.set(self.frame.dispatches.get() + 1);
        record_kernel(&key);
        Ok(())
    }
}

#[cfg(not(feature = "wgpu"))]
pub struct Sink<'a> {
    _device: &'a Context,
    _frame: &'a Frame,
    _pipelines: &'a Pipelines,
    _handles: &'a Handles<Api>,
}

#[cfg(not(feature = "wgpu"))]
impl<'a> Sink<'a> {
    #[must_use]
    pub fn new(
        device: &'a Context,
        frame: &'a Frame,
        pipelines: &'a Pipelines,
        handles: &'a Handles<Api>,
    ) -> Sink<'a> {
        Sink {
            _device: device,
            _frame: frame,
            _pipelines: pipelines,
            _handles: handles,
        }
    }
}

#[cfg(not(feature = "wgpu"))]
impl Encode for Sink<'_> {
    fn fire(&self, fire: Fire, _args: &[ArgValue]) -> Result<(), Error> {
        Err(Error::Backend {
            op: fire.entrypoint,
            detail: Fault::Deviceless.to_string(),
        })
    }

    fn absent(&self) -> Result<ArgValue, Error> {
        Ok(ArgValue::Buffer(NIL))
    }

    fn target(&self) -> Target {
        Target::Wgsl
    }
}

/// Whether argument `at`'s view of `bytes` fits the device's largest storage
/// binding. A view past it is refused rather than clamped: a clamped view
/// binds a truncated slab, and the kernel's reads past the cut come back as
/// whatever robust access answers.
#[cfg_attr(not(feature = "wgpu"), allow(dead_code))]
fn storage_view_fits(at: usize, bytes: u64, limit: u64) -> Result<(), String> {
    if bytes <= limit {
        return Ok(());
    }
    Err(format!(
        "argument {at} binds a {bytes}-byte view, past this device's \
         max_storage_buffer_binding_size of {limit} bytes; a view is never \
         clamped, so the slab must be split to fit the binding"
    ))
}
