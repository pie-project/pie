#![allow(unsafe_code)]
use crate::error::Error;

use crate::attn::{sched_decode, sched_mla, sched_prefill};

use crate::jit::{Ctx, refuse};
use crate::tensor::Tensor;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Device {
    pub num_sm: u32,
    pub cc_major: u32,
    pub max_smem_per_sm: u32,
    pub max_smem_per_block_optin: u32,
}

impl Device {
    pub const FLOOR_MAJOR: u32 = 8;

    /// For tests; never a stand-in for a device the probe could not read.
    pub const L40S: Self = Self {
        num_sm: 142,
        cc_major: 8,
        max_smem_per_sm: 102_400,
        max_smem_per_block_optin: 101_376,
    };

    #[must_use]
    pub fn probe(ctx: &Ctx) -> Option<Self> {
        #[cfg(feature = "rocm")]
        {
            let _ = ctx;
            Some(Self {
                num_sm: crate::jit::device::multiprocessors()?.max(1),
                cc_major: crate::jit::device::compute_capability_major()?,
                max_smem_per_sm: crate::jit::device::max_shared_memory_per_sm()?,
                max_smem_per_block_optin: crate::jit::device::max_shared_memory_per_block_optin()?,
            })
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = ctx;
            None
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Workspace {
    pub int_ptr: u64,
    pub int_bytes: usize,
    pub float_ptr: u64,
    pub float_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub num_requests: u32,
    pub lane_offset: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub page_size: u32,
    pub hnd_layout: bool,
}

impl Shape {
    #[must_use]
    pub const fn group_size(&self) -> u32 {
        #[allow(clippy::manual_checked_ops)]
        if self.num_kv_heads > 0 {
            self.num_q_heads / self.num_kv_heads
        } else {
            1
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Live {
    pub requests: u32,
    pub lane_offset: u32,
    pub row_offset: u32,
    pub rows: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Built<I> {
    pub info: I,
    pub int_upload: Vec<u8>,
    pub int_bytes: usize,
    pub float_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodePlan {
    pub info: DecodePlanInfo,
    pub int_upload: Vec<u8>,
    pub int_bytes: usize,
    pub float_bytes: usize,
    pub workspace: Workspace,
    pub shape: Shape,
    pub window: Option<u32>,
    pub device: Device,
    /// The fire's mask offsets (i64, one a request and one past) and words
    /// (u32 a request), where it staged them: a decode reading a lane's mask
    /// finds the lane's here. `plan_decode` leaves them `None`; the engine
    /// sets them.
    pub mask_indptr: Option<Tensor>,
    pub mask_origin: Option<Tensor>,
}

impl DecodePlan {
    #[must_use]
    pub const fn full_attention_variant(&self) -> bool {
        self.window.is_none()
    }

    pub fn accepts(
        &self,
        op: &'static str,
        head_dim: u32,
        window: Option<u32>,
    ) -> Result<(), Error> {
        planned_head_dim(op, self.shape.head_dim, head_dim)?;
        if self.full_attention_variant() != window.is_none() {
            return Err(refuse(
                op,
                "the stated window is not the reading this fire's attention schedule was \
                 planned for",
            ));
        }
        Ok(())
    }

    pub fn stage(&self, ctx: &Ctx) -> Result<(), Error> {
        upload(
            ctx,
            "attn.plan_decode",
            &self.int_upload,
            self.workspace.int_ptr,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefillPlan {
    pub info: PrefillPlanInfo,
    pub int_upload: Vec<u8>,
    pub int_bytes: usize,
    pub float_bytes: usize,
    pub workspace: Workspace,
    pub shape: Shape,
    pub total_tokens: u32,
    pub window: Option<u32>,
    pub causal: bool,
    pub graph_capturable: bool,
    pub graph_refusal: Option<String>,
    pub mask_indptr: Option<Tensor>,
    /// Per request its mask word (`kernels_cuda`'s `PrefillPlan::mask_origin`):
    /// the engine binds it beside `mask_indptr`; ROCm's paged prefill takes
    /// no mask, so nothing here reads it.
    pub mask_origin: Option<Tensor>,
    pub device: Device,
}

impl PrefillPlan {
    pub fn accepts(
        &self,
        op: &'static str,
        head_dim: u32,
        kv_heads: Option<u32>,
        window: Option<u32>,
    ) -> Result<(), Error> {
        planned_head_dim(op, self.shape.head_dim, head_dim)?;
        if let Some(kv_heads) = kv_heads
            && self.shape.num_kv_heads != kv_heads
        {
            return Err(refuse(
                op,
                format!(
                    "the stated kv head count {kv_heads} is not the {} this fire's \
                         prefill schedule was planned at",
                    self.shape.num_kv_heads
                ),
            ));
        }
        if self.window != window {
            return Err(refuse(
                op,
                format!(
                    "the stated window {window:?} is not the {:?} this fire's prefill \
                     schedule carved its kv spans for",
                    self.window
                ),
            ));
        }
        Ok(())
    }

    #[must_use]
    pub const fn cta_tile_q(&self) -> u32 {
        self.info.cta_tile_q as u32
    }

    pub fn stage(&self, ctx: &Ctx) -> Result<(), Error> {
        upload(
            ctx,
            "attn.plan_prefill",
            &self.int_upload,
            self.workspace.int_ptr,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlaPlan {
    pub info: MlaPlanInfo,
    pub int_upload: Vec<u8>,
    pub int_bytes: usize,
    pub float_bytes: usize,
    pub workspace: Workspace,
    pub num_heads: u32,
    pub causal: bool,
    pub device: Device,
}

impl MlaPlan {
    pub fn stage(&self, ctx: &Ctx) -> Result<(), Error> {
        upload(
            ctx,
            "attn.mla_plan",
            &self.int_upload,
            self.workspace.int_ptr,
        )
    }
}

const HEAD_DIMS: [u32; 4] = [64, 128, 256, 512];

#[must_use]
pub fn head_dim_instantiated(head_dim: u32) -> bool {
    HEAD_DIMS.contains(&head_dim)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Toggles {
    /// The fewest keys a split decode chunk carries; 0 takes 256.
    pub chunk_floor_tokens: u32,

    pub prefill_ctas_per_sm: u32,
}

impl Toggles {
    #[must_use]
    pub fn of(tuning: crate::tuning::Tuning) -> Self {
        Self {
            chunk_floor_tokens: tuning.decode_chunk_floor_tokens(),
            prefill_ctas_per_sm: tuning.prefill_ctas_per_sm(),
        }
    }
}

pub(crate) fn window_left(op: &'static str, window: Option<u32>) -> Result<i32, Error> {
    match window {
        None => Ok(-1),
        Some(0) => Err(refuse(op, "the stated sliding window is zero")),
        Some(w) => i32::try_from(w - 1).map_err(|_| {
            refuse(
                op,
                format!("the {w}-wide window does not fit the kernel's int"),
            )
        }),
    }
}

fn planned_head_dim(op: &'static str, planned: u32, stated: u32) -> Result<(), Error> {
    if planned == stated {
        return Ok(());
    }
    Err(refuse(
        op,
        format!(
            "the stated head width {stated} is not the {planned} this fire's attention \
             schedule was planned at"
        ),
    ))
}

fn instantiated(op: &'static str, head_dim: u32) -> Result<(), Error> {
    if head_dim_instantiated(head_dim) {
        return Ok(());
    }
    Err(refuse(
        op,
        format!(
            "no fa2 unit is stamped at head width {head_dim}; the lattice holds 64/128/256/512"
        ),
    ))
}

fn some_requests(op: &'static str, shape: &Shape) -> Result<(), Error> {
    if shape.num_requests == 0 {
        return Err(refuse(op, "the batch is empty"));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn plan_decode(
    kv_indptr: &[i32],
    kv_len: &[i32],
    shape: Shape,
    live: Live,
    window: Option<u32>,
    enable_cuda_graph: bool,
    max_grid_size: u32,
    toggles: Toggles,
    device: &Device,
    workspace: Workspace,
) -> Result<DecodePlan, Error> {
    const OP: &str = "attn.plan_decode";
    let _ = kv_len;
    instantiated(OP, shape.head_dim)?;
    some_requests(OP, &shape)?;
    let window_left = window_left(OP, window)?;

    let _ = window_left;
    // An eager decode plans as a body does: its keys split while the
    // lanes' head groups leave the grid unfilled (`sched_decode::estimate`).
    let req = sched_decode::Request {
        kv_indptr,
        batch_size: shape.num_requests,
        live,
        num_qo_heads: shape.num_q_heads,
        gqa_group_size: shape.group_size(),
        page_size: shape.page_size,
        head_dim: shape.head_dim,
        enable_cuda_graph,
        chunk_floor_tokens: toggles.chunk_floor_tokens,
    };
    let built = sched_decode::plan(
        OP,
        &req,
        max_grid_size,
        workspace.int_bytes,
        workspace.float_bytes,
    )?;

    Ok(DecodePlan {
        info: built.info,
        int_upload: built.int_upload,
        int_bytes: built.int_bytes,
        float_bytes: built.float_bytes,
        workspace,
        shape,
        window,
        device: *device,
        mask_indptr: None,
        mask_origin: None,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn plan_prefill(
    qo_indptr: &[i32],
    kv_indptr: &[i32],
    kv_len: &[i32],
    total_tokens: u32,
    shape: Shape,
    live: Live,
    window: Option<u32>,
    causal: bool,
    enable_cuda_graph: bool,
    mask_indptr: Option<Tensor>,
    device: &Device,
    toggles: Toggles,
    workspace: Workspace,
) -> Result<PrefillPlan, Error> {
    const OP: &str = "attn.plan_prefill";
    let _ = kv_len;
    instantiated(OP, shape.head_dim)?;
    some_requests(OP, &shape)?;
    window_left(OP, window)?;

    let req = sched_prefill::Request {
        qo_indptr,
        kv_indptr,
        total_num_rows: total_tokens,
        batch_size: shape.num_requests,
        lane_offset: shape.lane_offset,
        live,
        num_qo_heads: shape.num_q_heads,
        num_kv_heads: shape.num_kv_heads,
        head_dim: shape.head_dim,
        page_size: shape.page_size,
        enable_cuda_graph,
        window_left: window.map(|w| w - 1),
        ctas_per_sm: toggles.prefill_ctas_per_sm,
    };

    let (built, capturable, graph_refusal) =
        match sched_prefill::plan(OP, &req, device, workspace.int_bytes, workspace.float_bytes) {
            Ok(built) => (built, enable_cuda_graph, None),
            Err(declined) if enable_cuda_graph => {
                let req = sched_prefill::Request {
                    enable_cuda_graph: false,
                    ..req
                };
                let built = sched_prefill::plan(
                    OP,
                    &req,
                    device,
                    workspace.int_bytes,
                    workspace.float_bytes,
                )?;
                (built, false, Some(declined.to_string()))
            }
            Err(declined) => return Err(declined),
        };

    Ok(PrefillPlan {
        info: built.info,
        int_upload: built.int_upload,
        int_bytes: built.int_bytes,
        float_bytes: built.float_bytes,
        workspace,
        shape,
        total_tokens,
        window,
        causal,
        graph_capturable: capturable,
        graph_refusal,
        mask_indptr,
        mask_origin: None,
        device: *device,
    })
}

/// The float workspace a captured prefill plan of `rows` rows over `lanes`
/// lanes lays: its split's partials, or nothing where its layout runs
/// unsplit. It is [`plan_prefill`]'s own layout of that graph form (one
/// live request of a row, one page of keys): a captured layout's tile,
/// split and padded grid follow from its carved rows and lanes alone,
/// never from the keys or the live rows, so every replay of that body asks
/// exactly this. The load sizes the prefill grant as the most of it over
/// the compositions its lattice arms (`engine_cuda::inputs`).
#[allow(clippy::too_many_arguments)]
pub fn prefill_graph_float_bytes(
    rows: u32,
    lanes: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    page_size: u32,
    window: Option<u32>,
    device: &Device,
    toggles: Toggles,
) -> Result<usize, Error> {
    const OP: &str = "attn.plan_prefill";
    window_left(OP, window)?;
    // The live request carries a row: the layout reads the carve's rows
    // (`total_num_rows`), and the live rows only fill its index tables.
    let qo_indptr = [0, 1];
    let req = sched_prefill::Request {
        qo_indptr: &qo_indptr,
        kv_indptr: &[0, 1],
        total_num_rows: rows.max(1),
        batch_size: lanes.max(1),
        lane_offset: 0,
        live: Live {
            requests: 1,
            lane_offset: 0,
            row_offset: 0,
            rows: 1,
        },
        num_qo_heads: num_q_heads,
        num_kv_heads,
        head_dim,
        page_size: page_size.max(1),
        enable_cuda_graph: true,
        window_left: window.map(|w| w - 1),
        ctas_per_sm: toggles.prefill_ctas_per_sm,
    };
    Ok(sched_prefill::plan(OP, &req, device, usize::MAX, usize::MAX)?.float_bytes)
}

#[allow(clippy::too_many_arguments)]
pub fn plan_mla(
    qo_indptr: &[i32],
    kv_indptr: &[i32],
    kv_len: &[i32],
    total_tokens: u32,
    num_requests: u32,
    live: Live,
    num_heads: u32,
    head_dim_o: u32,
    causal: bool,
    device: &Device,
    workspace: Workspace,
) -> Result<MlaPlan, Error> {
    const OP: &str = "attn.mla_plan";
    let req = sched_mla::Request {
        qo_indptr,
        kv_indptr,
        kv_len_arr: kv_len,
        total_num_rows: total_tokens,
        batch_size: num_requests,
        live,
        num_heads,
        head_dim_o,
        causal,
    };
    let built = sched_mla::plan(OP, &req, device, workspace.int_bytes, workspace.float_bytes)?;
    Ok(MlaPlan {
        info: built.info,
        int_upload: built.int_upload,
        int_bytes: built.int_bytes,
        float_bytes: built.float_bytes,
        workspace,
        num_heads,
        causal,
        device: *device,
    })
}

pub fn upload(ctx: &Ctx, op: &'static str, bytes: &[u8], int_ptr: u64) -> Result<(), Error> {
    if bytes.is_empty() {
        return Ok(());
    }
    if int_ptr == 0 {
        return Err(refuse(op, "the plan's int workspace is null"));
    }
    #[cfg(feature = "rocm")]
    {
        // The plan's bytes are the caller's, dropped when its run ends: a
        // copy from them may run after that, since HIP stages a pageable
        // source only as it reaches it. A pinned slot of the stream's ring
        // holds them until an event recorded after the copy fires.
        let from = pinned::stage(op, ctx.stream(), bytes)?;
        if crate::sizing::active() {
            // A size-only walk (`crate::sizing`) moves nothing.
            return pinned::sent(op, ctx.stream());
        }
        // SAFETY: `int_ptr` is the plan's device workspace, sized for the
        // plan the host bytes describe; `from` is a pinned slot holding a
        // copy of them, which the ring does not reuse until an event
        // recorded after this copy fires.
        unsafe {
            crate::hip::memcpy_async(
                int_ptr as usize as *mut core::ffi::c_void,
                from.cast(),
                bytes.len(),
                crate::hip::Direction::HostToDevice,
                ctx.stream(),
            )
        }
        .map_err(|code| {
            refuse(
                op,
                format!("`hipMemcpyAsync` answered {code} staging the plan"),
            )
        })?;
        pinned::sent(op, ctx.stream())
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = ctx;
        Err(crate::jit::runtimeless(op))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecodePlanInfo {
    pub padded_batch_size: i64,
    pub v_offset: Option<u32>,
    pub s_offset: Option<u32>,
    pub request_indices_offset: Option<u32>,
    pub kv_tile_indices_offset: Option<u32>,
    pub o_indptr_offset: Option<u32>,
    pub block_valid_mask_offset: Option<u32>,
    pub kv_chunk_size_ptr_offset: Option<u32>,
    pub enable_cuda_graph: bool,
    pub split_kv: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefillPlanInfo {
    pub padded_batch_size: i64,
    pub total_num_rows: i64,
    pub total_num_rows_offset: Option<u32>,
    pub cta_tile_q: i64,
    pub request_indices_offset: Option<u32>,
    pub qo_tile_indices_offset: Option<u32>,
    pub kv_tile_indices_offset: Option<u32>,
    pub merge_indptr_offset: Option<u32>,
    pub o_indptr_offset: Option<u32>,
    pub kv_chunk_size_ptr_offset: Option<u32>,
    pub v_offset: Option<u32>,
    pub s_offset: Option<u32>,
    pub block_valid_mask_offset: Option<u32>,
    pub enable_cuda_graph: bool,
    pub split_kv: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MlaPlanInfo {
    pub num_blks_x: i64,
    pub num_blks_y: i64,
    pub q_indptr_offset: Option<u32>,
    pub kv_indptr_offset: Option<u32>,
    pub partial_indptr_offset: Option<u32>,
    pub merge_packed_offset_start_offset: Option<u32>,
    pub merge_packed_offset_end_offset: Option<u32>,
    pub merge_partial_packed_offset_start_offset: Option<u32>,
    pub merge_partial_packed_offset_end_offset: Option<u32>,
    pub merge_partial_stride_offset: Option<u32>,
    pub q_len_offset: Option<u32>,
    pub kv_len_offset: Option<u32>,
    pub q_start_offset: Option<u32>,
    pub kv_start_offset: Option<u32>,
    pub kv_end_offset: Option<u32>,
    pub work_indptr_offset: Option<u32>,
    pub partial_o_offset: Option<u32>,
    pub partial_lse_offset: Option<u32>,
}

/// Forgets the plan ring of `stream` before the caller destroys it
/// (`pinned::forget_stream`). A no-op without the device runtime.
pub fn forget_stream(stream: *mut core::ffi::c_void) {
    #[cfg(feature = "rocm")]
    pinned::forget_stream(stream.cast());
    #[cfg(not(feature = "rocm"))]
    let _ = stream;
}

/// The pinned host slots a plan's bytes are staged through, a ring a stream,
/// each slot guarded by an event recorded after the copy that reads it.
#[cfg(feature = "rocm")]
mod pinned {
    use super::{Error, refuse};
    use crate::hip::rt;
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    /// Slots before a stage waits for the oldest copy rather than grow.
    const MOST: usize = 64;
    /// The least a slot holds, so a batch's plans growing by a few rows
    /// reuse what warm launches allocated.
    const LEAST: usize = 64 << 10;
    /// `hipErrorNotReady`: an event's work is still queued.
    const NOT_READY: i32 = 600;
    /// `hipEventDisableTiming`.
    const DISABLE_TIMING: u32 = 2;

    struct Slot {
        host: *mut u8,
        bytes: usize,
        read: rt::Event,
    }

    struct Ring {
        slots: Vec<Slot>,
        next: usize,
    }

    // SAFETY: a ring holds page-locked allocations and events, runtime
    // handles any thread may use; it is reached only under its lock.
    unsafe impl Send for Ring {}

    /// Every stream's ring, by its device and handle: each its own lock, so
    /// a stage waiting out its stream's oldest copy holds back no other's.
    static RINGS: Mutex<BTreeMap<(i32, usize), &'static Mutex<Ring>>> = Mutex::new(BTreeMap::new());

    fn on_ring<R>(
        op: &'static str,
        stream: rt::Stream,
        f: impl FnOnce(&mut Ring) -> Result<R, Error>,
    ) -> Result<R, Error> {
        let device = crate::jit::cache::current_device()
            .map_err(|fault| refuse(op, format!("staging the plan: {fault}")))?;
        let ring: &'static Mutex<Ring> = RINGS
            .lock()
            .expect("plan rings lock poisoned")
            .entry((device, stream as usize))
            .or_insert_with(|| {
                Box::leak(Box::new(Mutex::new(Ring {
                    slots: Vec::new(),
                    next: 0,
                })))
            });
        f(&mut ring.lock().expect("plan ring lock poisoned"))
    }

    fn check(op: &'static str, what: &str, code: i32) -> Result<(), Error> {
        if code == 0 {
            Ok(())
        } else {
            Err(refuse(
                op,
                format!("`{what}` answered {code} staging the plan"),
            ))
        }
    }

    /// `bytes` copied into the ring's next slot, waiting out the copy that
    /// last read it; the slot's address.
    pub(super) fn stage(
        op: &'static str,
        stream: rt::Stream,
        bytes: &[u8],
    ) -> Result<*const u8, Error> {
        on_ring(op, stream, |ring| {
            let at = ring.next;
            // The slot at `at` is the oldest a copy read; one still queued
            // gets a new slot before it, which keeps it the oldest.
            let queued = at < ring.slots.len() && {
                // SAFETY: an event this ring created.
                let code = unsafe { rt::hipEventQuery(ring.slots[at].read) };
                if code == NOT_READY {
                    true
                } else {
                    check(op, "hipEventQuery", code)?;
                    false
                }
            };
            if queued && ring.slots.len() >= MOST {
                // SAFETY: an event this ring created.
                check(op, "hipEventSynchronize", unsafe {
                    rt::hipEventSynchronize(ring.slots[at].read)
                })?;
            } else if queued || ring.slots.len() <= at {
                let mut read: rt::Event = core::ptr::null_mut();
                // SAFETY: an out-pointer for a new event.
                check(op, "hipEventCreateWithFlags", unsafe {
                    rt::hipEventCreateWithFlags(&raw mut read, DISABLE_TIMING)
                })?;
                ring.slots.insert(
                    at,
                    Slot {
                        host: core::ptr::null_mut(),
                        bytes: 0,
                        read,
                    },
                );
            }
            let slot = &mut ring.slots[at];
            if slot.bytes < bytes.len() {
                if !slot.host.is_null() {
                    // SAFETY: a page-locked allocation this slot owns, its
                    // last copy run (the event above).
                    unsafe { rt::hipHostFree(slot.host.cast()) };
                    slot.host = core::ptr::null_mut();
                    slot.bytes = 0;
                }
                let size = bytes.len().next_power_of_two().max(LEAST);
                let mut host: *mut core::ffi::c_void = core::ptr::null_mut();
                // SAFETY: an out-pointer for a new page-locked allocation.
                check(op, "hipHostMalloc", unsafe {
                    rt::hipHostMalloc(&raw mut host, size, 0)
                })?;
                slot.host = host.cast();
                slot.bytes = size;
            }
            // SAFETY: the slot holds at least `bytes.len()` and no copy reads it.
            unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), slot.host, bytes.len()) };
            Ok(slot.host.cast_const())
        })
    }

    /// Forgets what `stream` recorded in its ring, before it is destroyed:
    /// each slot's event waits (the stream still live) and is made again,
    /// unrecorded, and the ring leaves the map, so a later stream at the
    /// same address starts a ring of its own. A query of an event whose
    /// stream was destroyed reads freed state (on ROCm, a lock no thread
    /// holds, so the query waits forever).
    pub(crate) fn forget_stream(stream: rt::Stream) {
        let Ok(device) = crate::jit::cache::current_device() else {
            return;
        };
        let ring = RINGS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&(device, stream as usize));
        let Some(ring) = ring else {
            return;
        };
        let mut ring = ring.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        for slot in &mut ring.slots {
            // SAFETY: the slot's own event, recorded only on `stream`, which
            // is live until its caller destroys it after this returns.
            unsafe {
                let _ = rt::hipEventSynchronize(slot.read);
                let _ = rt::hipEventDestroy(slot.read);
                if !slot.host.is_null() {
                    // The slot's last copy ran (the event above).
                    let _ = rt::hipHostFree(slot.host.cast());
                }
            }
            slot.host = core::ptr::null_mut();
            slot.bytes = 0;
        }
        ring.slots.clear();
        ring.next = 0;
    }

    /// The slot `stage` handed out is read once `stream` runs to here.
    pub(super) fn sent(op: &'static str, stream: rt::Stream) -> Result<(), Error> {
        on_ring(op, stream, |ring| {
            let at = ring.next;
            ring.next = (at + 1) % ring.slots.len();
            // SAFETY: an event this ring created, on the caller's stream.
            check(op, "hipEventRecord", unsafe {
                rt::hipEventRecord(ring.slots[at].read, stream)
            })
        })
    }
}
