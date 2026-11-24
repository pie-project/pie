#![allow(unsafe_code)]
use engine_gpu::{Handles, NIL};
use kernels_slang::{ArgValue, Encode, Error, Fire, Target};

use crate::device::ctx::Frame;
use crate::device::{Api, Context, Pipelines};
use engine_gpu::error::Fault;

static KERNEL_PROFILE: engine_gpu::kernel_profile::KernelProfile =
    engine_gpu::kernel_profile::KernelProfile::new();

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
    LONGEST_SPAN_NS.with(|longest| longest.set(longest.get().max(ns)));
}

thread_local! {
    /// The longest device span of a frame this thread waited on since it
    /// last took it ([`take_longest_frame_span_ns`]).
    static LONGEST_SPAN_NS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// The longest device span (first dispatch's start to last's end, by the
/// device's timestamps) of a frame this thread waited on since it last
/// asked, while kernels were profiled ([`profile_kernels`]); 0 where none.
/// What one submission took on the device, apart from any queued ahead of it.
pub fn take_longest_frame_span_ns() -> u64 {
    LONGEST_SPAN_NS.with(|longest| longest.replace(0))
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
    KERNEL_PROFILE.rows()
}

#[cfg(feature = "vulkan")]
pub(crate) fn credit(name: &str, ns: u64) {
    KERNEL_PROFILE.time(name, ns);
}

pub fn reset_kernel_profile() {
    KERNEL_PROFILE.clear();
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
    KERNEL_PROFILE.fire(name);
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
    std::mem::take(&mut *SHAPES.lock().expect("shapes lock poisoned"))
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
    baked: Vec<(u64, u64, u64)>,
    /// Each binding's pool segment, where it binds one (below the span's),
    /// and whether it binds the buffer whole.
    segments: Vec<Option<(u32, bool)>>,
    /// Each binding's place in the frame's score room, where it binds the
    /// room at a binding laid for it: the bytes past the room's byte.
    rooms: Vec<Option<u64>>,
}

/// What stands for a pool segment's buffer in a frame's hash: its place,
/// never the buffer, which a commit or a release changes under a recording
/// that is rebound ([`crate::device::Context::rebind_segments`]).
const SEGMENT_TAG: u64 = 0xffff_ffff_0000_0000;

/// What stands for a score room's buffer in a frame's hash: the room's place
/// is the fire's ([`crate::device::Kept::bind_room`]).
const ROOM_TAG: u64 = 0xffff_fffe_0000_0000;

#[cfg(feature = "vulkan")]
pub struct Sink<'a> {
    device: &'a Context,
    frame: &'a Frame,
    pipelines: &'a Pipelines,
    handles: &'a Handles<Api>,

    scratch: std::cell::RefCell<Scratch>,

    comm: Option<&'a crate::comm::Comm>,

    outer: engine_gpu::probe::Recording,

    /// The pool's segments below the span's, by raw buffer, as the walk
    /// found them placed.
    segments: std::collections::HashMap<u64, u32>,
}

#[cfg(feature = "vulkan")]
impl<'a> Sink<'a> {
    #[must_use]
    pub fn new(
        device: &'a Context,
        frame: &'a Frame,
        pipelines: &'a Pipelines,
        handles: &'a Handles<Api>,
    ) -> Sink<'a> {
        let outer = engine_gpu::probe::set_frame(frame);

        LAST_FIRE.with(|cell| cell.set(None));
        let segments = device
            .core
            .segments
            .lock()
            .expect("segments lock poisoned")
            .clone();
        Sink {
            device,
            frame,
            pipelines,
            handles,
            scratch: std::cell::RefCell::new(Scratch::default()),
            comm: None,
            outer,
            segments,
        }
    }

    #[must_use]
    pub fn with_comm(mut self, comm: &'a crate::comm::Comm) -> Sink<'a> {
        self.comm = Some(comm);
        self
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
impl Drop for Sink<'_> {
    fn drop(&mut self) {
        engine_gpu::probe::restore(self.outer);
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

    fn target(&self) -> Target {
        Target::Spirv
    }

    fn fixed_rows(&self) -> bool {
        self.frame.fixed_rows()
    }

    fn comm(&self, op: &'static str) -> Result<kernels_slang::Comm, Error> {
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
        Ok(kernels_slang::Comm {
            rank: comm.rank(),
            world: comm.world(),
            band,
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

        let key = kernels_slang::key(fire.entrypoint, fire.tier).ok_or_else(|| {
            Sink::backend(
                fire,
                format!("not instantiated at the {} tier", fire.tier.name()),
            )
        })?;
        let groups = engine_gpu::workgroups(
            key,
            fire.groups,
            self.device.core.limits.max_compute_work_group_count,
        )
        .map_err(|fault| Sink::refuse(fire, fault))?;
        let pipeline = self
            .pipelines
            .get(self.device, fire.file, key)
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
        scratch.segments.clear();
        scratch.rooms.clear();
        let (infos, writes_at, extents, push, segments, rooms) = (
            &mut scratch.infos,
            &mut scratch.writes_at,
            &mut scratch.extents,
            &mut scratch.push,
            &mut scratch.segments,
            &mut scratch.rooms,
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
                        segments.push(None);
                        rooms.push(None);
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
                    let range = u64::from(self.device.core.limits.max_storage_buffer_range);
                    let view = storage_view(infos.len(), remaining, binding.extent(), range)
                        .map_err(|detail| Sink::backend(fire, detail))?;
                    infos.push(
                        vk::DescriptorBufferInfo::default()
                            .buffer(binding.buffer().slab().buffer)
                            .offset(binding.offset())
                            .range(view),
                    );
                    writes_at.push(mutable);
                    extents.push(binding.extent());
                    let slab = binding.buffer().slab();
                    // The score room at a binding laid for it is the fire's
                    // (a replay binds the room it lays its rows in); a pool
                    // segment is bound by its place.
                    let at = infos.len() - 1;
                    let room = (at < 64 && pipeline.rooms & (1 << at) != 0)
                        .then_some(self.frame.room)
                        .flatten()
                        .filter(|&(room, _)| room == slab.buffer)
                        .map(|(_, at)| binding.offset().saturating_sub(at));
                    let segment = room.is_none().then(|| {
                        self.segments.get(&ash::vk::Handle::as_raw(slab.buffer)).map(|&at| {
                            (at, binding.offset() == 0 && binding.extent() == binding.buffer().bytes())
                        })
                    }).flatten();
                    segments.push(segment);
                    rooms.push(room);
                    // A recording holds neither a pool segment's buffer nor
                    // its room's: a segment's release gives the memory back
                    // (the recording rebound to its stand-in first), and a
                    // room is its pair's. A walk run once holds them until
                    // it lands.
                    if (segment.is_none() && room.is_none()) || !self.frame.kept() {
                        self.frame.retain(slab);
                    }
                }
                ArgValue::I32(v) => push.extend_from_slice(&v.to_le_bytes()),
                ArgValue::U32(v) => push.extend_from_slice(&v.to_le_bytes()),
                ArgValue::F32(v) => push.extend_from_slice(&v.to_le_bytes()),
            }
        }

        let declared = pipeline.bindings as usize;
        let params_at = pipeline.fire.as_ref().map(|fire| fire.binding as usize);
        for at in infos.len()..declared {
            if pipeline.used[at] && Some(at) != params_at {
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
        // A module that reads every scalar from the fire block pushes none.
        let scalars = pipeline
            .fire
            .as_ref()
            .map_or(0, |fire| fire.bytes)
            .max(pipeline.push_bytes);
        if push.len() != scalars as usize {
            return Err(Sink::backend(
                fire,
                format!(
                    "the entry passes {} bytes of scalars but the module declares a {}-byte block",
                    push.len(),
                    scalars
                ),
            ));
        }
        // The scalars the kernel reads from the fire block lie there, at a
        // place the commands name; the rest are pushed and kept with the
        // recording. The group counts lie there too, for an indirect
        // dispatch where the frame's shapes differ in them.
        let limits = &self.device.core.limits;
        let mut fired_buffer: Option<vk::Buffer> = None;
        let fired = match pipeline.fire.as_ref() {
            Some(block) => {
                let at = block.binding as usize;
                if at < infos.len() {
                    return Err(Sink::backend(
                        fire,
                        format!("argument {at} is a buffer at the module's fire block"),
                    ));
                }
                let (buffer, place) = self
                    .frame
                    .lay(push, limits.min_uniform_buffer_offset_alignment)
                    .map_err(|fault| Sink::refuse(fire, fault))?;
                fired_buffer = Some(buffer);
                infos.resize(
                    at,
                    vk::DescriptorBufferInfo::default()
                        .buffer(dummy.buffer)
                        .offset(0)
                        .range(dummy.size),
                );
                writes_at.resize(at, false);
                extents.resize(at, 0);
                segments.resize(at, None);
                rooms.resize(at, None);
                infos.push(
                    vk::DescriptorBufferInfo::default()
                        .buffer(buffer)
                        .offset(place.1)
                        .range(u64::from(block.bytes)),
                );
                writes_at.push(false);
                extents.push(0);
                segments.push(None);
                rooms.push(None);
                Some((place, &block.words))
            }
            None => None,
        };
        let group_words: Vec<u8> = groups.iter().flat_map(|g| g.to_le_bytes()).collect();
        let (indirect, groups_place) = self
            .frame
            .lay(&group_words, 4)
            .map_err(|fault| Sink::refuse(fire, fault))?;
        let indirect = self.frame.dispatched(groups).then_some(indirect);
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
            if !pipeline.used[at] || info.buffer == dummy.buffer || Some(at) == params_at {
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
        {
            // What the commands carry, but what a fire changes: the kernel,
            // every binding but the block's (which lies at its place in every
            // walk laid alike), the pushed scalars the kernel reads, the
            // barrier. The group counts are the frame's to weigh.
            use ash::vk::Handle;
            scratch.baked.clear();
            scratch.baked.extend(
                infos
                    .iter()
                    .enumerate()
                    .take(declared)
                    .filter(|&(at, _)| pipeline.used[at] && Some(at) != params_at)
                    .map(|(at, info)| {
                        // A room's view is the fire's: where the room lies and
                        // where in it the fire's rows do (the fill's
                        // [`crate::device::Frame::room_past`]).
                        if scratch.rooms.get(at).copied().flatten().is_some() {
                            return (ROOM_TAG, 0, 0);
                        }
                        match scratch.segments.get(at).copied().flatten() {
                            Some((segment, true)) => {
                                (SEGMENT_TAG | u64::from(segment), 0, u64::MAX)
                            }
                            Some((segment, false)) => {
                                (SEGMENT_TAG | u64::from(segment), info.offset, info.range)
                            }
                            None => (info.buffer.as_raw(), info.offset, info.range),
                        }
                    }),
            );
            let mut pushed = push.clone();
            if let Some((_, words)) = fired {
                for &(at, len) in words {
                    pushed[at as usize..(at + len) as usize].fill(0);
                }
            }
            if tracing::enabled!(target: "vkbake", tracing::Level::TRACE) {
                tracing::trace!(
                    target: "vkbake",
                    key,
                    baked = ?scratch.baked,
                    fired = ?fired.map(|(place, _)| place),
                    groups = ?groups_place,
                    owed,
                    pushed = ?pushed,
                    "bake"
                );
            }
            self.frame.bake(&(
                key,
                &scratch.baked,
                fired.map(|(place, _)| place),
                groups_place,
                owed,
                pushed,
            ));
        }

        // A kept frame's buffer set is the device's for every recording
        // that binds alike on the arm (its fire block's buffer): keyed by
        // what the commands bake of its bindings (a pool segment by its
        // place, a score room by its tag, both written again by a rebind
        // or a replay).
        let shared_key = self.frame.kept().then(|| {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            scratch.baked.hash(&mut h);
            fired_buffer.map(ash::vk::Handle::as_raw).hash(&mut h);
            let key = h.finish();
            crate::device::ctx::note_tuple(key ^ ash::vk::Handle::as_raw(pipeline.set_layout));
            key
        });
        // Where in the room each room binding lies, the fire's data.
        for past in scratch.rooms.iter().take(declared).flatten() {
            self.frame.room_past(*past);
        }
        // A fill walk records nothing: what it laid and hashed is its fill.
        if self.frame.dry {
            self.frame.dispatches.set(self.frame.dispatches.get() + 1);
            return Ok(());
        }
        let core = self.frame.core();
        let d = &core.device;
        let cmd = self.frame.cmd();
        let set = match shared_key {
            Some(key) => self.frame.shared_set(pipeline.set_layout, key),
            None => self.frame.descriptor_set(pipeline.set_layout),
        }
        .map_err(|fault| Sink::refuse(fire, fault))?;
        // The fire block's set (set 1), the dispatch's place its dynamic
        // offset: an arm's block's the device's one for the kernel, an
        // arm-less frame's its own.
        let fire_set = match (pipeline.fire_layout, fired, fired_buffer) {
            (Some(layout), Some(((_, at), _)), Some(buffer)) => {
                let binding = params_at.expect("a fire layout has its binding") as u32;
                let range = u64::from(pipeline.fire.as_ref().map_or(0, |block| block.bytes));
                let set = if self.frame.arms_block() {
                    self.frame.fire_set(layout, binding, buffer, range)
                } else {
                    self.frame.own_fire_set(layout, binding, buffer, range)
                }
                .map_err(|fault| Sink::refuse(fire, fault))?;
                Some((set, u32::try_from(at).map_err(|_| {
                    Sink::backend(fire, format!("the fire block's place {at} past a u32"))
                })?))
            }
            _ => None,
        };
        // A recording's pool segments are rebound as they are backed and
        // given back: where its layout lays the binding update-after-bind;
        // elsewhere it stands only while the segments do.
        if self.frame.kept() {
            for (at, info) in infos.iter().enumerate().take(declared) {
                if !pipeline.used[at] {
                    continue;
                }
                if let Some(past) = scratch.rooms.get(at).copied().flatten() {
                    // 0 bytes: to the room's buffer's end, as a view binds.
                    self.frame.room_binds(set, at as u32, past, 0);
                    continue;
                }
                let Some((segment, whole)) = scratch.segments.get(at).copied().flatten() else {
                    continue;
                };
                if at < 64 && pipeline.segments & (1 << at) != 0 {
                    let bytes = if whole { u64::MAX } else { info.range };
                    self.frame.rebinds(set, at as u32, segment, info.offset, bytes);
                } else {
                    self.frame.pin(format!(
                        "kernel {key} binds pool segment {segment} at binding {at}, which its \
                         layout does not lay update-after-bind"
                    ));
                }
            }
        }
        scratch.sets.clear();
        for (at, info) in infos.iter().enumerate().take(declared) {
            if !pipeline.used[at] || Some(at) == params_at {
                continue;
            }

            // SAFETY: only the lifetime changes, so `sets` can live in the
            // reused scratch. `WriteDescriptorSet` keeps a raw pointer to
            // `info`, which is read only by `update_descriptor_sets` below,
            // while `infos` is untouched; `sets` is cleared before `infos`
            // changes again on the next fire, and no one reads it in between.
            let info: &'static vk::DescriptorBufferInfo = unsafe { std::mem::transmute(info) };
            let ty = vk::DescriptorType::STORAGE_BUFFER;
            scratch.sets.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(at as u32)
                    .descriptor_type(ty)
                    .buffer_info(std::slice::from_ref(info)),
            );
        }
        // SAFETY: `set` was handed out by `descriptor_set` to this dispatch
        // alone and no submitted command uses it yet, so updating it needs no
        // further sync; every buffer it names is a live slab the frame retains,
        // with offsets and ranges from bound views. `cmd` is the open frame's
        // buffer, recorded only by this sink's thread (`Frame` is not `Sync`).
        // The push block's length was checked against the layout's range;
        // the fire block's uniform and group counts lie in chunks the frame
        // retains, made with the uniform and indirect usages, at offsets the
        // device's alignments allow; a timestamp index stays below the
        // pool's `capacity * 2`.
        unsafe {
            if !scratch.sets.is_empty() {
                d.update_descriptor_sets(&scratch.sets, &[]);
            }
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
            if declared > 0 {
                match fire_set {
                    Some((fire_set, at)) => d.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        pipeline.layout,
                        0,
                        &[set, fire_set],
                        &[at],
                    ),
                    None => d.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        pipeline.layout,
                        0,
                        &[set],
                        &[],
                    ),
                }
            }
            if pipeline.push_bytes > 0 {
                d.cmd_push_constants(
                    cmd,
                    pipeline.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &push[..pipeline.push_bytes as usize],
                );
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
                    .expect("timing lock poisoned");
                match slot.as_mut() {
                    Some(t) if (t.names.len() as u32) < t.capacity => {
                        let at = t.names.len() as u32;
                        t.names.push(key);
                        Some((t.pool, at))
                    }
                    _ => None,
                }
            };
            if let Some((pool, at)) = timed {
                d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, pool, at * 2);
            }
            match indirect {
                Some(buffer) => d.cmd_dispatch_indirect(cmd, buffer, groups_place.1),
                None => d.cmd_dispatch(cmd, groups[0], groups[1], groups[2]),
            }
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
        record_kernel(key);
        if CAPTURING.load(std::sync::atomic::Ordering::Acquire) {
            use ash::vk::Handle;
            SHAPES
                .lock()
                .expect("shapes lock poisoned")
                .push(DispatchShape {
                    entrypoint: key,
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
    _handles: &'a Handles<Api>,
}

#[cfg(not(feature = "vulkan"))]
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

    fn target(&self) -> Target {
        Target::Spirv
    }
}

/// The device's `kernel.qmm` ladder at this shape, as kernels-slang's rungs.
pub(crate) fn qmm_ladder(
    tuning: &crate::tuning::Tuning,
    rows: u32,
    columns: u32,
    contraction: u32,
    bank: kernels_slang::Bank,
) -> Vec<kernels_slang::linear::qmm::Qmm> {
    let at = tuning.qmm_at(rows, columns, contraction, bank.bits, bank.group);
    tuning.qmm(&at).into_iter().map(qmm_rung).collect()
}

/// Every rung of the device's `kernel.qmm` ladder a batch of 1 to
/// `capacity` rows reaches at this shape, as kernels-slang's rungs.
pub(crate) fn qmm_rungs(
    tuning: &crate::tuning::Tuning,
    capacity: u32,
    columns: u32,
    contraction: u32,
    bank: kernels_slang::Bank,
) -> Vec<kernels_slang::linear::qmm::Qmm> {
    let at = |rows| tuning.qmm_at(rows, columns, contraction, bank.bits, bank.group);
    tuning
        .qmm_between(&at(1), &at(capacity.max(1)))
        .into_iter()
        .map(qmm_rung)
        .collect()
}

fn qmm_rung(rung: crate::tuning::Qmm) -> kernels_slang::linear::qmm::Qmm {
    use crate::tuning::Qmm;
    use kernels_slang::linear::qmm::Qmm as Rung;
    match rung {
        Qmm::Coopmat => Rung::Coopmat,
        Qmm::RegisterTiled { bm, k16 } => Rung::RegisterTiled { bm, k16 },
        Qmm::Tiled { crossover_tg } => Rung::Tiled { crossover_tg },
        Qmm::OneRow { lanes, rows } => Rung::OneRow { lanes, rows },
        Qmm::Qmv { vecs } => Rung::Qmv { vecs },
    }
}

/// The range argument `at` binds: the `remaining` bytes of its slab from its
/// offset, or, where those run past `maxStorageBufferRange` (a range drivers
/// answer by binding a truncated view) and the `extent` it states fits, the
/// largest range the device binds, which holds every byte the kernel reads.
/// A view whose own extent is past the limit is refused: no descriptor
/// holds it, so its slab must be split.
#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
fn storage_view(at: usize, remaining: u64, extent: u64, limit: u64) -> Result<u64, String> {
    if remaining <= limit {
        return Ok(remaining);
    }
    if extent <= limit {
        return Ok(limit);
    }
    Err(format!(
        "argument {at} binds a {extent}-byte view, past this device's \
         maxStorageBufferRange of {limit} bytes; the slab must be split to fit \
         the binding"
    ))
}
