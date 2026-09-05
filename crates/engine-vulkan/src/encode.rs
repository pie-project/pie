use kernels_vulkan::{ArgValue, Encode, Error, Fire};

use crate::device::ctx::Frame;
use crate::device::handles::NIL;
use crate::device::{Context, Handles, Pipelines};
use crate::error::Fault;

static KERNEL_PROFILE: std::sync::Mutex<std::collections::BTreeMap<String, (u64, u64)>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

static PROFILING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static HOST_ENCODE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BARRIERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BETWEEN_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FRAME_SPAN_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[must_use]
pub fn frame_span_ns() -> u64 {
    FRAME_SPAN_NS.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(feature = "vulkan")]
pub(crate) fn credit_frame_span(ns: u64) {
    FRAME_SPAN_NS.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
}

static SUBMIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FENCE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[must_use]
pub fn submit_ns() -> u64 {
    SUBMIT_NS.load(std::sync::atomic::Ordering::Relaxed)
}

#[must_use]
pub fn fence_ns() -> u64 {
    FENCE_NS.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(feature = "vulkan")]
pub(crate) fn credit_submit(ns: u64) {
    SUBMIT_NS.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "vulkan")]
pub(crate) fn credit_fence(ns: u64) {
    FENCE_NS.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
}
thread_local! {
    static LAST_FIRE: std::cell::Cell<Option<std::time::Instant>> = const {
        std::cell::Cell::new(None)
    };
}

#[must_use]
pub fn barriers() -> u64 {
    BARRIERS.load(std::sync::atomic::Ordering::Relaxed)
}

#[must_use]
pub fn between_fires_ns() -> u64 {
    BETWEEN_NS.load(std::sync::atomic::Ordering::Relaxed)
}

#[must_use]
pub fn host_encode_ns() -> u64 {
    HOST_ENCODE_NS.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn profile_kernels(on: bool) {
    PROFILING.store(on, std::sync::atomic::Ordering::Release);
}

#[must_use]
pub fn profiling() -> bool {
    PROFILING.load(std::sync::atomic::Ordering::Acquire)
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

#[cfg(feature = "vulkan")]
pub(crate) fn credit(name: &str, ns: u64) {
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
    HOST_ENCODE_NS.store(0, std::sync::atomic::Ordering::Relaxed);
    BARRIERS.store(0, std::sync::atomic::Ordering::Relaxed);
    BETWEEN_NS.store(0, std::sync::atomic::Ordering::Relaxed);
    FRAME_SPAN_NS.store(0, std::sync::atomic::Ordering::Relaxed);
    SUBMIT_NS.store(0, std::sync::atomic::Ordering::Relaxed);
    FENCE_NS.store(0, std::sync::atomic::Ordering::Relaxed);
    LAST_FIRE.with(|cell| cell.set(None));
}

#[cfg(feature = "vulkan")]
fn record_kernel(name: &str) {
    let mut table = KERNEL_PROFILE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    table.entry(name.to_string()).or_insert((0, 0)).1 += 1;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DispatchShape {
    pub entrypoint: &'static str,

    pub bindings: Vec<(u64, u64, u64)>,
    pub push: Vec<u8>,
    pub groups: [u32; 3],
    pub barrier: bool,
}

static SHAPES: std::sync::Mutex<Vec<DispatchShape>> = std::sync::Mutex::new(Vec::new());
static CAPTURING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn capture_shapes(on: bool) {
    CAPTURING.store(on, std::sync::atomic::Ordering::Release);
}

#[must_use]
pub fn take_shapes() -> Vec<DispatchShape> {
    std::mem::take(
        &mut *SHAPES
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    )
}

#[cfg(feature = "vulkan")]
#[derive(Default)]
struct Scratch {
    infos: Vec<ash::vk::DescriptorBufferInfo>,
    writes_at: Vec<bool>,
    extents: Vec<u64>,
    push: Vec<u8>,
    reads: Vec<crate::device::ctx::Span>,
    writes: Vec<crate::device::ctx::Span>,
    sets: Vec<ash::vk::WriteDescriptorSet<'static>>,
}

#[cfg(feature = "vulkan")]
pub struct Sink<'a> {
    device: &'a Context,
    frame: &'a Frame,
    pipelines: &'a Pipelines,
    handles: &'a Handles,

    scratch: std::cell::RefCell<Scratch>,

    comm: Option<&'a crate::comm::Comm>,
}

#[cfg(feature = "vulkan")]
impl<'a> Sink<'a> {
    #[must_use]
    pub fn new(
        device: &'a Context,
        frame: &'a Frame,
        pipelines: &'a Pipelines,
        handles: &'a Handles,
    ) -> Sink<'a> {
        crate::probe::set_frame(frame);

        LAST_FIRE.with(|cell| cell.set(None));
        Sink {
            device,
            frame,
            pipelines,
            handles,
            scratch: std::cell::RefCell::new(Scratch::default()),
            comm: None,
        }
    }

    #[must_use]
    pub fn with_comm(mut self, comm: &'a crate::comm::Comm) -> Sink<'a> {
        self.comm = Some(comm);
        self
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

#[cfg(feature = "vulkan")]
impl Encode for Sink<'_> {
    fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Error> {
        let started = profiling().then(std::time::Instant::now);
        if let Some(at) = started {
            LAST_FIRE.with(|cell| {
                if let Some(prior) = cell.get() {
                    BETWEEN_NS.fetch_add(
                        at.saturating_duration_since(prior).as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
            });
        }
        let answer = self.fire_inner(fire, args);
        if let Some(at) = started {
            HOST_ENCODE_NS.fetch_add(
                at.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            LAST_FIRE.with(|cell| cell.set(Some(std::time::Instant::now())));
        }
        answer
    }

    fn absent(&self) -> Result<ArgValue, Error> {
        Ok(ArgValue::Buffer(NIL))
    }

    fn comm(&self, op: &'static str) -> Result<kernels_vulkan::Comm, Error> {
        let comm = self.comm.ok_or(Error::Unsupported { op })?;

        let band = self
            .handles
            .bind(comm.band(), 0, comm.band().bytes())
            .map_err(|fault| Error::Backend {
                op,
                detail: fault.to_string(),
            })?;
        let slot_bytes = comm.slot_bytes_u32().map_err(|fault| Error::Backend {
            op,
            detail: fault.to_string(),
        })?;
        Ok(kernels_vulkan::Comm {
            rank: comm.rank(),
            world: comm.world(),

            band: kernels_vulkan::Tensor::new(band, 1, 0, model_ir::Dtype::Bf16),
            slot_bytes,
        })
    }

    fn rendezvous(&self, op: &'static str) -> Result<(), Error> {
        let comm = self.comm.ok_or(Error::Unsupported { op })?;

        self.frame.flush().map_err(|fault| Error::Backend {
            op,
            detail: fault.to_string(),
        })?;
        comm.wait();
        Ok(())
    }
}

#[cfg(feature = "vulkan")]
impl Sink<'_> {
    fn fire_inner(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Error> {
        use ash::vk;

        let pipeline = self
            .pipelines
            .get(self.device, fire)
            .map_err(|fault| Sink::refuse(fire, fault))?;

        let dummy = &self.device.dummy;
        let mut scratch = self.scratch.borrow_mut();
        let scratch = &mut *scratch;

        scratch.infos.clear();
        scratch.writes_at.clear();
        scratch.extents.clear();
        scratch.push.clear();
        scratch.reads.clear();
        scratch.writes.clear();
        let (infos, writes_at, extents, push) = (
            &mut scratch.infos,
            &mut scratch.writes_at,
            &mut scratch.extents,
            &mut scratch.push,
        );
        for arg in args {
            match *arg {
                ArgValue::Buffer(handle) | ArgValue::BufferMut(handle) => {
                    let mutable = matches!(arg, ArgValue::BufferMut(_));
                    if handle == NIL {
                        infos.push(
                            vk::DescriptorBufferInfo::default()
                                .buffer(dummy.buffer)
                                .offset(0)
                                .range(dummy.size),
                        );
                        writes_at.push(false);
                        extents.push(0);
                        continue;
                    }
                    let binding = self.handles.get(handle).ok_or_else(|| {
                        Sink::refuse(
                            fire,
                            Fault::Unbound {
                                what: format!(
                                    "handle {handle} at argument {}, which no row answers",
                                    infos.len()
                                ),
                            },
                        )
                    })?;
                    let remaining = binding.remaining();
                    if remaining == 0 {
                        return Err(Sink::backend(
                            fire,
                            format!("argument {} binds an empty view", infos.len()),
                        ));
                    }
                    infos.push(
                        vk::DescriptorBufferInfo::default()
                            .buffer(binding.slab().buffer)
                            .offset(binding.offset())
                            .range(remaining),
                    );
                    writes_at.push(mutable);
                    extents.push(binding.extent());
                    self.frame.retain(binding.slab());
                }
                ArgValue::I32(v) => push.extend_from_slice(&v.to_le_bytes()),
                ArgValue::U32(v) => push.extend_from_slice(&v.to_le_bytes()),
                ArgValue::F32(v) => push.extend_from_slice(&v.to_le_bytes()),
            }
        }

        let declared = pipeline.bindings as usize;
        for at in infos.len()..declared {
            if pipeline.used[at] {
                return Err(Sink::backend(
                    fire,
                    format!(
                        "the module declares binding {at} but the entry passed only {} buffers",
                        infos.len()
                    ),
                ));
            }
        }
        for (at, &writes) in writes_at.iter().enumerate().skip(declared) {
            if writes {
                return Err(Sink::backend(
                    fire,
                    format!(
                        "argument {at} is a write target the module does not declare (it \
                         declares {declared} bindings)"
                    ),
                ));
            }
        }
        if push.len() != pipeline.push_bytes as usize {
            return Err(Sink::backend(
                fire,
                format!(
                    "the entry pushes {} bytes of scalars but the module declares a {}-byte push block",
                    push.len(),
                    pipeline.push_bytes
                ),
            ));
        }
        let stated = fire.group.iter().product::<u32>();
        if stated > 1 && fire.group != pipeline.local {
            return Err(Sink::backend(
                fire,
                format!(
                    "the entry states workgroup {:?} but the module was compiled with {:?}",
                    fire.group, pipeline.local
                ),
            ));
        }

        let (reads, writes) = (&mut scratch.reads, &mut scratch.writes);
        for (at, info) in infos.iter().enumerate().take(declared) {
            if !pipeline.used[at] || info.buffer == dummy.buffer {
                continue;
            }
            let span = crate::device::ctx::Span::new(info.buffer, info.offset, extents[at]);
            let written = writes_at[at] || pipeline.writable.get(at).copied().unwrap_or(true);
            if written {
                writes.push(span);
            } else {
                reads.push(span);
            }
        }
        let owed = self.frame.barrier_owed(reads, writes);
        if owed {
            BARRIERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        let core = self.frame.core();
        let d = &core.device;
        let cmd = self.frame.cmd();
        let set = self
            .frame
            .descriptor_set(pipeline.set_layout)
            .map_err(|fault| Sink::refuse(fire, fault))?;
        scratch.sets.clear();
        for (at, info) in infos.iter().enumerate().take(declared) {
            if !pipeline.used[at] {
                continue;
            }

            let info: &'static vk::DescriptorBufferInfo = unsafe { std::mem::transmute(info) };
            scratch.sets.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(at as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info)),
            );
        }
        unsafe {
            if !scratch.sets.is_empty() {
                d.update_descriptor_sets(&scratch.sets, &[]);
            }
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
            if declared > 0 {
                d.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.layout,
                    0,
                    &[set],
                    &[],
                );
            }
            if !push.is_empty() {
                d.cmd_push_constants(cmd, pipeline.layout, vk::ShaderStageFlags::COMPUTE, 0, push);
            }

            if owed {
                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
                d.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
            }
            let timed = {
                let mut slot = self
                    .frame
                    .inner
                    .timing
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                match slot.as_mut() {
                    Some(t) if (t.names.len() as u32) < t.capacity => {
                        let at = t.names.len() as u32;
                        t.names.push(fire.entrypoint);
                        Some((t.pool, at))
                    }
                    _ => None,
                }
            };
            if let Some((pool, at)) = timed {
                d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, pool, at * 2);
            }
            d.cmd_dispatch(
                cmd,
                fire.groups[0].max(1),
                fire.groups[1].max(1),
                fire.groups[2].max(1),
            );
            if let Some((pool, at)) = timed {
                d.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    pool,
                    at * 2 + 1,
                );
            }
        }
        self.frame.dispatches.set(self.frame.dispatches.get() + 1);
        record_kernel(fire.entrypoint);
        if CAPTURING.load(std::sync::atomic::Ordering::Acquire) {
            use ash::vk::Handle;
            SHAPES
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(DispatchShape {
                    entrypoint: fire.entrypoint,
                    bindings: infos
                        .iter()
                        .take(declared)
                        .map(|info| (info.buffer.as_raw(), info.offset, info.range))
                        .collect(),
                    push: push.clone(),
                    groups: fire.groups,
                    barrier: owed,
                });
        }
        Ok(())
    }
}

#[cfg(not(feature = "vulkan"))]
pub struct Sink<'a> {
    _device: &'a Context,
    _frame: &'a Frame,
    _pipelines: &'a Pipelines,
    _handles: &'a Handles,
}

#[cfg(not(feature = "vulkan"))]
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

#[cfg(not(feature = "vulkan"))]
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
