use kernels_wgpu::{ArgValue, Encode, Error, Fire};

use crate::device::ctx::Frame;
use crate::device::handles::NIL;
use crate::device::{Context, Handles, Pipelines};
use crate::error::Fault;

static KERNEL_PROFILE: std::sync::Mutex<std::collections::BTreeMap<String, (u64, u64)>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

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
static STAGED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[must_use]
pub fn host_copies() -> (u64, u64) {
    (
        COPIES.load(std::sync::atomic::Ordering::Relaxed),
        STAGED.load(std::sync::atomic::Ordering::Relaxed),
    )
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

pub fn profile_shapes(on: bool) {
    SHAPES.store(on, std::sync::atomic::Ordering::Relaxed);
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
fn label(fire: Fire, args: &[ArgValue]) -> std::borrow::Cow<'static, str> {
    use std::fmt::Write;
    if !shapes() {
        return std::borrow::Cow::Borrowed(fire.entrypoint);
    }
    let mut key = String::with_capacity(fire.entrypoint.len() + 48);
    key.push_str(fire.entrypoint);
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
            ArgValue::Buffer(_) | ArgValue::BufferMut(_) => continue,
        };
        let _ = write!(key, "{}{scalar}", if first { " s=" } else { "," });
        first = false;
    }
    std::borrow::Cow::Owned(key)
}

#[must_use]
pub fn kernel_profile() -> Vec<(String, u64, u64)> {
    let mut rows: Vec<(String, u64, u64)> = KERNEL_PROFILE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .map(|(name, &(ns, n))| (name.clone(), ns, n))
        .collect();
    rows.sort_by_key(|row| std::cmp::Reverse((row.1, row.2)));
    rows
}

#[cfg(feature = "wgpu")]
pub(crate) fn record_timing(name: &str, ns: u64) {
    let mut table = KERNEL_PROFILE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    table.entry(name.to_string()).or_insert((0, 0)).0 += ns;
}

pub fn reset_kernel_profile() {
    KERNEL_PROFILE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
    HOST_NS.store(0, std::sync::atomic::Ordering::Relaxed);
    SUBMIT_NS.store(0, std::sync::atomic::Ordering::Relaxed);
    WAIT_NS.store(0, std::sync::atomic::Ordering::Relaxed);
    COPIES.store(0, std::sync::atomic::Ordering::Relaxed);
    STAGED.store(0, std::sync::atomic::Ordering::Relaxed);
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
    let mut table = KERNEL_PROFILE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    table.entry(name.to_string()).or_insert((0, 0)).1 += 1;
}

#[cfg(feature = "wgpu")]
pub struct Sink<'a> {
    device: &'a Context,
    frame: &'a Frame,
    pipelines: &'a Pipelines,
    handles: &'a Handles,
}

#[cfg(feature = "wgpu")]
impl<'a> Sink<'a> {
    #[must_use]
    pub fn new(
        device: &'a Context,
        frame: &'a Frame,
        pipelines: &'a Pipelines,
        handles: &'a Handles,
    ) -> Sink<'a> {
        crate::probe::set_frame(frame);
        Sink {
            device,
            frame,
            pipelines,
            handles,
        }
    }

    #[must_use]
    pub fn into_frame(self) -> Option<Frame> {
        None
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
        let started = timing().then(std::time::Instant::now);
        let out = self.fire_inner(fire, args);
        if let Some(started) = started {
            HOST_NS.fetch_add(
                started.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        out
    }

    fn absent(&self) -> Result<ArgValue, Error> {
        Ok(ArgValue::Buffer(NIL))
    }
}

#[cfg(feature = "wgpu")]
impl Sink<'_> {
    fn fire_inner(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Error> {
        let pipeline = self
            .pipelines
            .get(self.device, fire)
            .map_err(|fault| Sink::refuse(fire, fault))?;
        let core = self.device.core();
        let limits = &core.limits;
        let storage_align = u64::from(limits.min_storage_buffer_offset_alignment);
        let max_binding = limits.max_storage_buffer_binding_size;

        let mut views: Vec<(wgpu::Buffer, u64, u64, bool)> = Vec::with_capacity(args.len());

        let mut staged: Vec<(wgpu::Buffer, u64, wgpu::Buffer, u64, u64)> = Vec::new();
        let mut scratch_views = 0usize;
        let mut words: Vec<u8> = Vec::with_capacity(64);
        for arg in args {
            match *arg {
                ArgValue::Buffer(handle) | ArgValue::BufferMut(handle) => {
                    let mutable = matches!(arg, ArgValue::BufferMut(_));
                    if handle == NIL {
                        let dummy = &self.device.dummy;
                        views.push((dummy.buffer.clone(), 0, dummy.size, false));
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
                    let remaining = binding.remaining();
                    if remaining == 0 {
                        if mutable {
                            return Err(Sink::backend(
                                fire,
                                format!("argument {} writes an empty view", views.len()),
                            ));
                        }
                        let dummy = &self.device.dummy;
                        views.push((dummy.buffer.clone(), 0, dummy.size, false));
                        continue;
                    }
                    if binding.offset().is_multiple_of(storage_align) {
                        let size = remaining
                            .next_multiple_of(4)
                            .min(binding.slab().size.saturating_sub(binding.offset()))
                            .min(max_binding);
                        views.push((
                            binding.slab().buffer.clone(),
                            binding.offset(),
                            size,
                            mutable,
                        ));
                        continue;
                    }

                    let source = binding.slab().buffer.clone();
                    let at = binding.offset();
                    if !at.is_multiple_of(4) {
                        if mutable {
                            return Err(Sink::backend(
                                fire,
                                format!(
                                    "argument {} is a write target at offset {at}, which is \
                                     not 4-aligned; a sub-word cut cannot be staged back",
                                    views.len()
                                ),
                            ));
                        }
                        let len = remaining.min(binding.slab().size.saturating_sub(at));
                        let mut bytes = vec![0u8; len.next_multiple_of(4) as usize];
                        binding
                            .slab()
                            .read(at, &mut bytes[..len as usize])
                            .map_err(|fault| Sink::refuse(fire, fault))?;
                        let staged_len = bytes.len() as u64;
                        let (chunk, slot) = self.frame.scratch_slot(staged_len);
                        core.queue.write_buffer(&chunk, slot, &bytes);
                        scratch_views += 1;
                        STAGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        views.push((chunk, slot, staged_len, false));
                        continue;
                    }
                    let len = remaining
                        .next_multiple_of(4)
                        .min(binding.slab().size.saturating_sub(at) & !3);
                    let (chunk, slot) = self.frame.scratch_slot(len);
                    self.frame
                        .encode(|encoder| {
                            encoder.copy_buffer_to_buffer(&source, at, &chunk, slot, len);
                        })
                        .map_err(|fault| Sink::refuse(fire, fault))?;
                    if mutable {
                        staged.push((chunk.clone(), slot, source, at, len));
                    }
                    scratch_views += 1;
                    STAGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    views.push((chunk, slot, len, mutable));
                }
                ArgValue::I32(v) => words.extend_from_slice(&v.to_le_bytes()),
                ArgValue::U32(v) => words.extend_from_slice(&v.to_le_bytes()),
                ArgValue::F32(v) => words.extend_from_slice(&v.to_le_bytes()),
            }
        }

        let declared = pipeline.bindings as usize;
        for at in views.len()..declared {
            if pipeline.used[at] {
                return Err(Sink::backend(
                    fire,
                    format!(
                        "the module declares binding {at} but the entry passed only {} buffers",
                        views.len()
                    ),
                ));
            }
        }
        for (at, view) in views.iter().enumerate() {
            if at >= declared {
                if view.3 {
                    return Err(Sink::backend(
                        fire,
                        format!(
                            "argument {at} is a write target the module does not declare (it \
                             declares {declared} bindings)"
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
        if stated > 1
            && pipeline.tier == kernels_wgpu::Capability::Baseline
            && fire.group != pipeline.local
        {
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
        let group = if scratch_views == 0 {
            self.pipelines
                .bind_group(core, &pipeline, &keyed, &words, uniform_bytes)
        } else {
            let mut entries: Vec<wgpu::BindGroupEntry<'_>> = Vec::with_capacity(declared + 1);
            for (at, (buffer, offset, size, _)) in views.iter().enumerate().take(declared) {
                if !pipeline.used[at] {
                    continue;
                }
                entries.push(wgpu::BindGroupEntry {
                    binding: at as u32,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: *offset,
                        size: std::num::NonZeroU64::new(*size),
                    }),
                });
            }
            let uniform;
            if let Some(binding) = pipeline.uniform {
                let (chunk, at) = self.frame.uniform_slot(uniform_bytes);
                let mut padded = words.clone();
                padded.resize(uniform_bytes as usize, 0);
                core.queue.write_buffer(&chunk, at, &padded);
                uniform = (chunk, at, uniform_bytes);
                entries.push(wgpu::BindGroupEntry {
                    binding,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform.0,
                        offset: uniform.1,
                        size: std::num::NonZeroU64::new(uniform.2),
                    }),
                });
            }
            core.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(fire.entrypoint),
                layout: &pipeline.layout,
                entries: &entries,
            })
        };
        core.take_error("create_bind_group")
            .map_err(|fault| Sink::refuse(fire, fault))?;

        let key = label(fire, args);
        if let Some((set, at)) = timing().then(|| self.frame.timestamp_slot(&key)).flatten() {
            self.frame
                .encode(|encoder| {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(fire.entrypoint),
                        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                            query_set: &set,
                            beginning_of_pass_write_index: Some(at),
                            end_of_pass_write_index: Some(at + 1),
                        }),
                    });
                    pass.set_pipeline(&pipeline.pipeline);
                    pass.set_bind_group(0, &group, &[]);
                    pass.dispatch_workgroups(
                        fire.groups[0].max(1),
                        fire.groups[1].max(1),
                        fire.groups[2].max(1),
                    );
                })
                .map_err(|fault| Sink::refuse(fire, fault))?;
        } else {
            self.frame
                .dispatch(fire.entrypoint, &pipeline.pipeline, &group, fire.groups)
                .map_err(|fault| Sink::refuse(fire, fault))?;
        }
        if !staged.is_empty() {
            self.frame
                .encode(|encoder| {
                    for (chunk, slot, home, home_at, len) in &staged {
                        encoder.copy_buffer_to_buffer(chunk, *slot, home, *home_at, *len);
                    }
                })
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
    _handles: &'a Handles,
}

#[cfg(not(feature = "wgpu"))]
impl<'a> Sink<'a> {
    #[must_use]
    pub fn new(
        device: &'a Context,
        frame: &'a Frame,
        pipelines: &'a Pipelines,
        handles: &'a Handles,
    ) -> Sink<'a> {
        Sink {
            _device: device,
            _frame: frame,
            _pipelines: pipelines,
            _handles: handles,
        }
    }

    #[must_use]
    pub fn into_frame(self) -> Option<Frame> {
        None
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
}
