//! Host-only: the work plans the FlashAttention-2 decode, prefill and MLA
//! arms launch from (`sched` the shared allocator and splitter).
#![allow(unsafe_code)]

use crate::error::Error;

use crate::jit::{Ctx, refuse};
use crate::tensor::Tensor;

pub(crate) mod decode;
pub(crate) mod mla;
pub(crate) mod prefill;
pub(crate) mod sched;

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
        #[cfg(feature = "cuda")]
        {
            let _ = ctx;
            Some(Self {
                num_sm: crate::jit::device::multiprocessors()?.max(1),
                cc_major: crate::jit::device::compute_capability_major()?,
                max_smem_per_sm: crate::jit::device::max_shared_memory_per_sm()?,
                max_smem_per_block_optin: crate::jit::device::max_shared_memory_per_block_optin()?,
            })
        }
        #[cfg(not(feature = "cuda"))]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sizes {
    pub float_bytes: usize,
    pub int_bytes: usize,
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
    pub device: Device,
}

impl PrefillPlan {
    #[must_use]
    pub const fn full_attention_variant(&self) -> bool {
        self.window.is_none()
    }

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
    pub force_split_small: bool,

    pub window_split: bool,

    pub chunk_floor_128: bool,

    pub prefill_ctas_per_sm: u32,

    pub static_max_requests: u32,

    pub decode_gqa: crate::attn::decode_gqa::Gate,
}

impl Toggles {
    #[must_use]
    pub const fn of(tuning: crate::tuning::Tuning) -> Self {
        Self {
            force_split_small: tuning.decode_split_small,
            window_split: tuning.decode_split_window,
            chunk_floor_128: tuning.decode_chunk_floor_128,
            prefill_ctas_per_sm: tuning.prefill_ctas_per_sm,
            static_max_requests: tuning.decode_static_max_requests,
            decode_gqa: crate::attn::decode_gqa::Gate::of(tuning),
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

    // The tensor-core decode's block holds a whole stamp of key tiles; its
    // split fills the SMs once, not the FA2 kernel's occupancy.
    let max_grid_size = match toggles.decode_gqa.stamp(shape.head_dim, false) {
        Some(stamp)
            if toggles
                .decode_gqa
                .takes(shape.head_dim, shape.group_size(), shape.num_requests) =>
        {
            stamp.grid(device)
        }
        _ => max_grid_size,
    };
    // A cluster folds through distributed shared memory, sm_90 on; it fills
    // the SMs once with the stamp's blocks.
    let ranks = match toggles.decode_gqa.stamp(shape.head_dim, false) {
        Some(stamp) if device.cc_major >= 9 => toggles.decode_gqa.ranks(
            shape.head_dim,
            shape.group_size(),
            shape.num_requests,
            shape.num_kv_heads,
            window_left,
            stamp.grid(device),
        ),
        _ => 0,
    };
    let windowed_split = toggles.window_split && window_left >= 0;
    let built = if !windowed_split
        && decode::can_use_static_nonsplit(shape.num_requests, enable_cuda_graph, toggles)
    {
        decode::static_nonsplit(
            OP,
            shape.num_requests,
            live,
            shape.page_size,
            enable_cuda_graph,
            workspace.int_bytes,
        )?
    } else {
        let req = decode::Request {
            kv_indptr,
            batch_size: shape.num_requests,
            live,
            num_qo_heads: shape.num_q_heads,
            gqa_group_size: shape.group_size(),
            page_size: shape.page_size,
            head_dim: shape.head_dim,
            enable_cuda_graph,
            chunk_floor_128: toggles.chunk_floor_128,
            window_left: u32::try_from(window_left).ok(),
            whole: ranks > 0
                || toggles
                    .decode_gqa
                    .whole(shape.head_dim, shape.group_size(), window_left),
        };
        decode::plan(
            OP,
            &req,
            max_grid_size,
            workspace.int_bytes,
            workspace.float_bytes,
        )?
    };

    let mut info = built.info;
    if info.whole {
        info.ranks = ranks;
    }
    Ok(DecodePlan {
        info,
        int_upload: built.int_upload,
        int_bytes: built.int_bytes,
        float_bytes: built.float_bytes,
        workspace,
        shape,
        window,
        device: *device,
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

    let req = prefill::Request {
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
        match prefill::plan(OP, &req, device, workspace.int_bytes, workspace.float_bytes) {
            Ok(built) => (built, enable_cuda_graph, None),
            Err(declined) if enable_cuda_graph => {
                let req = prefill::Request {
                    enable_cuda_graph: false,
                    ..req
                };
                let built =
                    prefill::plan(OP, &req, device, workspace.int_bytes, workspace.float_bytes)?;
                (built, false, Some(declined.to_string()))
            }
            Err(declined) => return Err(declined),
        };

    let cta_tile_q = u32::try_from(built.info.cta_tile_q).unwrap_or(0);
    if shape.head_dim >= 256 && cta_tile_q == 128 {
        return Err(refuse(
            OP,
            format!(
                "head width {} with CTA_TILE_Q {cta_tile_q} has no valid KernelTraits — \
                 `IsInvalid()` is true for every NUM_MMA_KV, so no unit exists and none can",
                shape.head_dim
            ),
        ));
    }

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
        device: *device,
    })
}

pub use crate::attn::plan::prefill::{
    graph_padding as prefill_graph_padding, partial_rows as prefill_partial_rows,
};

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
    let req = mla::Request {
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
    let built = mla::plan(OP, &req, device, workspace.int_bytes, workspace.float_bytes)?;
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
    #[cfg(feature = "cuda")]
    {
        use cudarc::runtime::sys as rt;

        // From pageable memory the copy holds the host until the stream
        // reaches it (the fire before's kernels), so a plan the next fire
        // stages ahead would wait out the card; a pinned slot leaves it
        // queued.
        let from = pinned::stage(op, bytes)?;
        // SAFETY: `from` is a pinned slot holding a copy of `bytes`, which
        // the ring does not reuse until an event recorded after this copy
        // fires; the stream is `ctx`'s, live by `Ctx::on`'s contract. That
        // `int_ptr` addresses `bytes.len()` writable device bytes is the
        // caller's to vouch for.
        let code = unsafe {
            rt::cudaMemcpyAsync(
                int_ptr as usize as *mut core::ffi::c_void,
                from.cast(),
                bytes.len(),
                rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                ctx.stream().cast(),
            )
        };
        if code != rt::cudaError::cudaSuccess {
            return Err(refuse(
                op,
                format!(
                    "`cudaMemcpyAsync` answered {} staging the plan",
                    code as i32
                ),
            ));
        }
        pinned::sent(op, ctx.stream().cast())
    }
    #[cfg(not(feature = "cuda"))]
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
    /// Each request one work item over its whole window, which the
    /// tensor-core decode lands unsplit (`decode_gqa::Gate::whole`).
    pub whole: bool,
    /// The blocks a cluster reads each (request, kv head) of a whole plan
    /// with, folding their slices itself (`decode_gqa::Gate::ranks`); 0 for none.
    pub ranks: u32,
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
    /// The fire's live lanes, `[first lane, requests]`, as two i32s: a
    /// captured launch reads them at replay.
    pub live_lanes_offset: Option<u32>,
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

/// A ring of page-locked slots the plans' uploads stage through, each
/// reused once the copy that read it has run. A slot whose copy is still
/// queued is passed over for a new one, so staging a plan never waits for
/// the device: the ring grows to the copies in flight, warm launches size
/// its slots, and only past `MOST` does a stage wait out the oldest.
#[cfg(feature = "cuda")]
mod pinned {
    use super::{Error, refuse};
    use core::cell::RefCell;
    use cudarc::runtime::sys as rt;

    /// Slots before a stage waits for the oldest copy rather than grow.
    const MOST: usize = 64;
    /// The least a slot holds, so a batch's plans growing by a few rows
    /// reuse what warm launches allocated.
    const LEAST: usize = 64 << 10;

    struct Slot {
        host: *mut u8,
        bytes: usize,
        read: rt::cudaEvent_t,
    }

    struct Ring {
        slots: Vec<Slot>,
        next: usize,
    }

    thread_local! {
        static RING: RefCell<Ring> = const { RefCell::new(Ring { slots: Vec::new(), next: 0 }) };
    }

    fn check(op: &'static str, what: &str, code: rt::cudaError_t) -> Result<(), Error> {
        if code == rt::cudaError::cudaSuccess {
            Ok(())
        } else {
            Err(refuse(
                op,
                format!("`{what}` answered {} staging the plan", code as i32),
            ))
        }
    }

    /// `bytes` copied into the ring's next slot, waiting out the copy that
    /// last read it; the slot's address.
    pub(super) fn stage(op: &'static str, bytes: &[u8]) -> Result<*const u8, Error> {
        RING.with(|ring| {
            let mut ring = ring.borrow_mut();
            let at = ring.next;
            // The slot at `at` is the oldest a copy read; one still queued
            // gets a new slot before it, which keeps it the oldest.
            let queued = at < ring.slots.len() && {
                // SAFETY: an event this ring created.
                let code = unsafe { rt::cudaEventQuery(ring.slots[at].read) };
                if code == rt::cudaError::cudaErrorNotReady {
                    true
                } else {
                    check(op, "cudaEventQuery", code)?;
                    false
                }
            };
            if queued && ring.slots.len() >= MOST {
                // SAFETY: an event this ring created.
                check(op, "cudaEventSynchronize", unsafe {
                    rt::cudaEventSynchronize(ring.slots[at].read)
                })?;
            } else if queued || ring.slots.len() <= at {
                let mut read: rt::cudaEvent_t = core::ptr::null_mut();
                // SAFETY: an out-pointer for a new event.
                check(op, "cudaEventCreateWithFlags", unsafe {
                    rt::cudaEventCreateWithFlags(&raw mut read, rt::cudaEventDisableTiming)
                })?;
                let fresh = Slot {
                    host: core::ptr::null_mut(),
                    bytes: 0,
                    read,
                };
                ring.slots.insert(at, fresh);
            }
            let slot = &mut ring.slots[at];
            if slot.bytes < bytes.len() {
                if !slot.host.is_null() {
                    // SAFETY: a page-locked allocation this slot owns, its
                    // last copy run (the event above).
                    unsafe { rt::cudaFreeHost(slot.host.cast()) };
                    slot.host = core::ptr::null_mut();
                    slot.bytes = 0;
                }
                let size = bytes.len().next_power_of_two().max(LEAST);
                let mut host: *mut core::ffi::c_void = core::ptr::null_mut();
                // SAFETY: an out-pointer for a new page-locked allocation.
                check(op, "cudaMallocHost", unsafe {
                    rt::cudaMallocHost(&raw mut host, size)
                })?;
                slot.host = host.cast();
                slot.bytes = size;
            }
            // SAFETY: the slot holds at least `bytes.len()` and no copy reads it.
            unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), slot.host, bytes.len()) };
            Ok(slot.host.cast_const())
        })
    }

    /// The slot `stage` handed out is read once `stream` runs to here.
    pub(super) fn sent(op: &'static str, stream: rt::cudaStream_t) -> Result<(), Error> {
        RING.with(|ring| {
            let mut ring = ring.borrow_mut();
            let at = ring.next;
            ring.next = (at + 1) % ring.slots.len();
            // SAFETY: an event this ring created, on the caller's stream.
            check(op, "cudaEventRecord", unsafe {
                rt::cudaEventRecord(ring.slots[at].read, stream)
            })
        })
    }
}
