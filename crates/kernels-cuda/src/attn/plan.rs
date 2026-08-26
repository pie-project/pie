//! The plan structs and their builders — the §6 payoff. What the old driver
//! conjured behind `Ctx.raised` (`Fa2Decode`, `Fa2Prefill`, `MlaPlanned`,
//! mutable `*PlanCache`s) is now four owned structs, each defined by an
//! explicit plan op in the IR and built by a **pure function** of:
//!
//! - host copies of the geometry the IR names (`kv_indptr`, `kv_len`,
//!   `qo_indptr`) — the planners walk their contents, and device handles
//!   cannot be read host-side (see the MENLO-SEAM notes on each builder);
//! - the shape facts of the attention being planned ([`Shape`]);
//! - the device facts ([`Device`]) and the workspace grant ([`Workspace`]) —
//!   the driver owns where both come from;
//! - the operator toggles ([`Toggles`]) — the driver resolves them from the
//!   environment once ([`Toggles::from_env`]) and hands them in, so a
//!   builder never reads the environment itself.
//!
//! A build stages the schedule's index vectors into `int_upload` and records
//! byte offsets into the granted workspace. Nothing device-side happens at
//! build time: the driver copies `int_upload` to `workspace.int_ptr` in the
//! prepare phase (a [`stage`](DecodePlan::stage) call, or its own memcpy),
//! and the capture-phase attention entries then only read.

use kernels::KernelError;

use crate::attn::{sched_decode, sched_mla, sched_prefill, sched_sm90};

use crate::jit::{Ctx, refuse};
use crate::tensor::Tensor;

/// The device facts every builder takes as an argument (never probed inside
/// a builder — purity is the design). Schedule math reads `num_sm` and
/// `cc_major`; the fa2 launch geometry carried on the plan reads the two
/// shared-memory bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Device {
    pub num_sm: u32,
    pub cc_major: u32,
    pub max_smem_per_sm: u32,
    pub max_smem_per_block_optin: u32,
}

impl Device {
    /// The fallback facts the old plane assumed when the probe failed: an
    /// L40S.
    pub const L40S: Self = Self {
        num_sm: 148,
        cc_major: 8,
        max_smem_per_sm: 102_400,
        max_smem_per_block_optin: 101_376,
    };

    /// Probes the current device once. A convenience for the driver — the
    /// builders themselves never call it.
    #[must_use]
    pub fn probe(ctx: &Ctx) -> Option<Self> {
        #[cfg(feature = "_cuda")]
        {
            let _ = ctx;
            Some(Self {
                num_sm: crate::jit::device::multiprocessors()?.max(1),
                cc_major: crate::jit::device::compute_capability_major()?,
                max_smem_per_sm: crate::jit::device::max_shared_memory_per_sm()?,
                max_smem_per_block_optin: crate::jit::device::max_shared_memory_per_block_optin()?,
            })
        }
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = ctx;
            None
        }
    }
}

/// The workspace grant a plan is carved into: the driver's bounded pool,
/// as base addresses plus byte bounds. The builders read only the bounds
/// (an offset past them is a refusal at build time, not a device fault at
/// fire time); the addresses ride the plan so the launches can resolve the
/// staged offsets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Workspace {
    pub int_ptr: u64,
    pub int_bytes: usize,
    pub float_ptr: u64,
    pub float_bytes: usize,
}

/// The kv-side shape a plan is built at. The attention entries restate
/// `head_dim` and `kv_heads` from the IR op and refuse a plan built at a
/// different shape — the old `agrees` check, kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub num_requests: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub page_size: u32,
    /// Whether the pool pages are laid out `[heads][tokens][dim]`.
    pub hnd_layout: bool,
}

impl Shape {
    #[must_use]
    pub const fn group_size(&self) -> u32 {
        if self.num_kv_heads > 0 {
            self.num_q_heads / self.num_kv_heads
        } else {
            1
        }
    }
}

/// What a raw schedule build hands back before it is folded into a plan
/// struct: the offset table, the staged bytes, and the sizes actually used.
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

/// The fa2 decode plan: the schedule the decode kernels walk, plus the
/// shape and workspace it was carved for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodePlan {
    pub info: DecodePlanInfo,
    /// The int-workspace image; the driver copies it to
    /// `workspace.int_ptr` each prepare phase (see [`DecodePlan::stage`]).
    pub int_upload: Vec<u8>,
    pub int_bytes: usize,
    pub float_bytes: usize,
    pub workspace: Workspace,
    pub shape: Shape,
    /// The sliding window this plan was carved for; the entries check the
    /// stated window against it (the old `variant_agrees`).
    pub window: Option<u32>,
    pub device: Device,
}

impl DecodePlan {
    #[must_use]
    pub const fn full_attention_variant(&self) -> bool {
        self.window.is_none()
    }

    /// The op's restated facts must be the ones this plan was carved at —
    /// plan facts are driver-supplied, so disagreement is refused, not
    /// asserted (the old `agrees`/`variant_agrees`). Decode states no kv
    /// head count and no exact window: only the windowed/full reading has
    /// to match.
    pub fn accepts(
        &self,
        op: &'static str,
        head_dim: u32,
        window: Option<u32>,
    ) -> Result<(), KernelError> {
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

    /// Copies the staged int workspace to the device. Prepare-phase work.
    pub fn stage(&self, ctx: &Ctx) -> Result<(), KernelError> {
        upload(
            ctx,
            "attention.plan_decode",
            &self.int_upload,
            self.workspace.int_ptr,
        )
    }
}

/// The fa2 prefill plan.
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
    /// Whether the schedule kept its graph-shaped padding. A build asked to
    /// be capturable may fall back to an uncapturable schedule rather than
    /// decline (the old retry, kept); the driver reads this before capture.
    pub graph_capturable: bool,
    /// `i32`, `[lanes + 1]`: each request's span of the mask bits
    /// `attention.masked` names. Bound only when this plan serves that op.
    // MENLO-SEAM: the op states the mask bits themselves, but this span
    // table has no IR seat — the driver derives it and binds it here at
    // build time.
    pub mask_indptr: Option<Tensor>,
    pub device: Device,
}

impl PrefillPlan {
    #[must_use]
    pub const fn full_attention_variant(&self) -> bool {
        self.window.is_none()
    }

    /// As [`DecodePlan::accepts`], at prefill's stricter reading: the kv
    /// head count (when the op states one — `attention.masked` does not)
    /// and the exact window must be the ones the schedule carved its kv
    /// spans for.
    pub fn accepts(
        &self,
        op: &'static str,
        head_dim: u32,
        kv_heads: Option<u32>,
        window: Option<u32>,
    ) -> Result<(), KernelError> {
        planned_head_dim(op, self.shape.head_dim, head_dim)?;
        if let Some(kv_heads) = kv_heads {
            if self.shape.num_kv_heads != kv_heads {
                return Err(refuse(
                    op,
                    format!(
                        "the stated kv head count {kv_heads} is not the {} this fire's \
                         prefill schedule was planned at",
                        self.shape.num_kv_heads
                    ),
                ));
            }
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

    pub fn stage(&self, ctx: &Ctx) -> Result<(), KernelError> {
        upload(
            ctx,
            "attention.plan_prefill",
            &self.int_upload,
            self.workspace.int_ptr,
        )
    }
}

/// The sm90 prefill plan. The builder is real (the schedule and its staging
/// are ported); the launcher that would consume it was never part of this
/// lattice, so `attn::prefill_sm90` answers a typed refusal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefillPlanSm90 {
    pub info: PrefillPlanSm90Info,
    pub int_upload: Vec<u8>,
    pub int_bytes: usize,
    pub workspace: Workspace,
    pub shape: Shape,
    pub total_tokens: u32,
    pub causal: bool,
    pub device: Device,
}

impl PrefillPlanSm90 {
    pub fn stage(&self, ctx: &Ctx) -> Result<(), KernelError> {
        upload(
            ctx,
            "attention.plan_prefill_sm90",
            &self.int_upload,
            self.workspace.int_ptr,
        )
    }
}

/// The latent-attention plan, shared by mla decode and prefill.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlaPlan {
    pub info: MlaPlanInfo,
    pub int_upload: Vec<u8>,
    pub int_bytes: usize,
    pub float_bytes: usize,
    pub workspace: Workspace,
    pub num_heads: u32,
    pub device: Device,
}

impl MlaPlan {
    pub fn stage(&self, ctx: &Ctx) -> Result<(), KernelError> {
        upload(ctx, "attention.mla_plan", &self.int_upload, self.workspace.int_ptr)
    }
}

/// The fa2 lattice's instantiated head widths.
const HEAD_DIMS: [u32; 4] = [64, 128, 256, 512];

#[must_use]
pub fn head_dim_instantiated(head_dim: u32) -> bool {
    HEAD_DIMS.contains(&head_dim)
}

/// The operator toggles a decode build takes as an argument — like
/// [`Device`], never probed inside a builder (purity is the design). The
/// driver resolves them once with [`Toggles::from_env`] and threads the
/// value through every [`plan_decode`] call.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Toggles {
    /// `PIE_CUDA_FORCE_SPLIT_KV_SMALL`: send even the small batches that
    /// qualify for the static non-split fast path through the split-kv
    /// planner.
    pub force_split_small: bool,

    /// `PIE_CUDA_WINDOW_SPLIT_KV`: plan windowed decodes through the
    /// split-kv planner rather than the static fast path.
    pub window_split: bool,
}

impl Toggles {
    /// Reads both toggles from the environment. Driver-side, once — the
    /// builders only ever see the resulting value.
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            force_split_small: truthy("PIE_CUDA_FORCE_SPLIT_KV_SMALL"),
            window_split: truthy("PIE_CUDA_WINDOW_SPLIT_KV"),
        }
    }
}

fn truthy(key: &str) -> bool {
    match std::env::var(key) {
        Ok(v) => v != "0",
        Err(_) => false,
    }
}

/// The sliding extent as the device text reads it: `-1` is "no window", and
/// a stated window of zero is a degenerate statement, not an unwindowed one.
pub(crate) fn window_left(op: &'static str, window: Option<u32>) -> Result<i32, KernelError> {
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

/// The old `agrees`: a stated head width must be the one the plan was
/// carved at.
fn planned_head_dim(op: &'static str, planned: u32, stated: u32) -> Result<(), KernelError> {
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

fn instantiated(op: &'static str, head_dim: u32) -> Result<(), KernelError> {
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

fn some_requests(op: &'static str, shape: &Shape) -> Result<(), KernelError> {
    if shape.num_requests == 0 {
        return Err(refuse(op, "the batch is empty"));
    }
    Ok(())
}

/// Builds the fa2 decode plan.
///
/// `max_grid_size` is a device fact the driver obtains from
/// [`crate::attn::fa2::decode_max_grid_size`] (an occupancy probe, so it
/// cannot live inside a pure builder); `toggles` is the driver's one
/// [`Toggles::from_env`] read, for the same reason.
///
// MENLO-SEAM: `attention.plan_decode` in the IR names on-device geometry
// (kv_indptr/kv_indices/last_page_len/kv_len); this builder walks
// `kv_indptr`'s CONTENTS, so the driver binds a host copy beside the device
// tensor — and the same duality serves `kv_len` here. kv_indices and
// last_page_len are never read at build time (they ride the pool row into
// the launch), and the fa2 decode schedule derives its extents from
// kv_indptr alone, so the op-named `kv_len` goes unread on this builder.
#[allow(clippy::too_many_arguments)]
pub fn plan_decode(
    kv_indptr: &[i32],
    kv_len: &[i32],
    shape: Shape,
    window: Option<u32>,
    enable_cuda_graph: bool,
    max_grid_size: u32,
    toggles: Toggles,
    device: &Device,
    workspace: Workspace,
) -> Result<DecodePlan, KernelError> {
    const OP: &str = "attention.plan_decode";
    let _ = kv_len;
    instantiated(OP, shape.head_dim)?;
    some_requests(OP, &shape)?;
    let window_left = window_left(OP, window)?;

    let windowed_split = toggles.window_split && window_left >= 0;
    let built = if !windowed_split
        && sched_decode::can_use_static_nonsplit(
            shape.num_requests,
            device.cc_major,
            enable_cuda_graph,
            toggles,
        ) {
        sched_decode::static_nonsplit(
            OP,
            shape.num_requests,
            shape.page_size,
            enable_cuda_graph,
            workspace.int_bytes,
        )?
    } else {
        let req = sched_decode::Request {
            kv_indptr,
            batch_size: shape.num_requests,
            num_qo_heads: shape.num_q_heads,
            gqa_group_size: shape.group_size(),
            page_size: shape.page_size,
            head_dim: shape.head_dim,
            enable_cuda_graph,
        };
        sched_decode::plan(
            OP,
            &req,
            max_grid_size,
            workspace.int_bytes,
            workspace.float_bytes,
        )?
    };

    Ok(DecodePlan {
        info: built.info,
        int_upload: built.int_upload,
        int_bytes: built.int_bytes,
        float_bytes: built.float_bytes,
        workspace,
        shape,
        window,
        device: *device,
    })
}

/// Builds the fa2 prefill plan. When a graph-shaped schedule does not fit
/// the workspace, the build falls back to an uncapturable one and records
/// that on `graph_capturable` — the old retry, kept.
///
// MENLO-SEAM: as `plan_decode`, plus the qo side — the builder walks host
// copies of both indptrs, bound by the driver beside the device tensors the
// IR names. The fa2 prefill schedule, too, derives its extents from
// kv_indptr alone, so the op-named `kv_len` goes unread here (the sm90
// twin reads it); `mask_indptr` is the span table `attention.masked`'s
// op-named bits ride with — driver-derived, see the field's seam note.
#[allow(clippy::too_many_arguments)]
pub fn plan_prefill(
    qo_indptr: &[i32],
    kv_indptr: &[i32],
    kv_len: &[i32],
    total_tokens: u32,
    shape: Shape,
    window: Option<u32>,
    causal: bool,
    enable_cuda_graph: bool,
    mask_indptr: Option<Tensor>,
    device: &Device,
    workspace: Workspace,
) -> Result<PrefillPlan, KernelError> {
    const OP: &str = "attention.plan_prefill";
    let _ = kv_len;
    instantiated(OP, shape.head_dim)?;
    some_requests(OP, &shape)?;
    window_left(OP, window)?;

    let req = sched_prefill::Request {
        qo_indptr,
        kv_indptr,
        total_num_rows: total_tokens,
        batch_size: shape.num_requests,
        num_qo_heads: shape.num_q_heads,
        num_kv_heads: shape.num_kv_heads,
        head_dim: shape.head_dim,
        page_size: shape.page_size,
        enable_cuda_graph,
        window_left: window.map(|w| w - 1),
    };

    let (built, capturable) =
        match sched_prefill::plan(OP, &req, device, workspace.int_bytes, workspace.float_bytes) {
            Ok(built) => (built, enable_cuda_graph),
            Err(_) if enable_cuda_graph => {
                let req = sched_prefill::Request {
                    enable_cuda_graph: false,
                    ..req
                };
                let built =
                    sched_prefill::plan(OP, &req, device, workspace.int_bytes, workspace.float_bytes)?;
                (built, false)
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
        mask_indptr,
        device: *device,
    })
}

/// Builds the sm90 prefill plan — the schedule the `AttnPrefillPlanSm90`
/// struct kind names. `kv_len` is the op's own named input now (per-request
/// kv lengths in tokens); as with the indptrs, the builder walks the host
/// twin the driver binds beside the device tensor.
#[allow(clippy::too_many_arguments)]
pub fn plan_prefill_sm90(
    qo_indptr: &[i32],
    kv_indptr: &[i32],
    kv_len: &[i32],
    total_tokens: u32,
    shape: Shape,
    causal: bool,
    enable_cuda_graph: bool,
    device: &Device,
    workspace: Workspace,
) -> Result<PrefillPlanSm90, KernelError> {
    const OP: &str = "attention.plan_prefill_sm90";
    some_requests(OP, &shape)?;
    let req = sched_sm90::Request {
        qo_indptr,
        kv_indptr,
        kv_len_arr: kv_len,
        total_num_rows: total_tokens,
        batch_size: shape.num_requests,
        num_qo_heads: shape.num_q_heads,
        num_kv_heads: shape.num_kv_heads,
        head_dim: shape.head_dim,
        causal,
        enable_cuda_graph,
    };
    let built = sched_sm90::plan(OP, &req, device, workspace.int_bytes)?;
    Ok(PrefillPlanSm90 {
        info: built.info,
        int_upload: built.int_upload,
        int_bytes: built.int_bytes,
        workspace,
        shape,
        total_tokens,
        causal,
        device: *device,
    })
}

/// Builds the latent-attention plan, shared by mla decode (`causal` false,
/// one token per lane) and prefill. As on `plan_prefill_sm90`, `kv_len` is
/// the op's named input, walked as the host twin beside qo_indptr and the
/// kv element offsets.
#[allow(clippy::too_many_arguments)]
pub fn plan_mla(
    qo_indptr: &[i32],
    kv_indptr: &[i32],
    kv_len: &[i32],
    num_requests: u32,
    num_heads: u32,
    head_dim_o: u32,
    causal: bool,
    device: &Device,
    workspace: Workspace,
) -> Result<MlaPlan, KernelError> {
    const OP: &str = "attention.mla_plan";
    let req = sched_mla::Request {
        qo_indptr,
        kv_indptr,
        kv_len_arr: kv_len,
        batch_size: num_requests,
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
        device: *device,
    })
}

/// Copies a plan's staged int workspace to the device, async on the
/// context's stream. Prepare-phase work: the source is pageable host
/// memory, which CUDA graph capture would refuse — the driver runs this
/// before capture, never inside it (the old plane pinned these bytes to
/// smuggle the copy into capture; the prepare/capture split makes that
/// machinery unnecessary).
pub fn upload(ctx: &Ctx, op: &'static str, bytes: &[u8], int_ptr: u64) -> Result<(), KernelError> {
    if bytes.is_empty() {
        return Ok(());
    }
    if int_ptr == 0 {
        return Err(refuse(op, "the plan's int workspace is null"));
    }
    #[cfg(feature = "_cuda")]
    {
        use cudarc::runtime::sys as rt;

        let code = unsafe {
            rt::cudaMemcpyAsync(
                int_ptr as usize as *mut core::ffi::c_void,
                bytes.as_ptr().cast(),
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
        Ok(())
    }
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = ctx;
        Err(crate::jit::runtimeless(op))
    }
}

// ─── the offset tables a plan build produces (was plan/info.rs) ────────────

// The offset tables a plan build produces: every field is either a launch
// shape fact or a byte offset into the int/float workspace the schedule was
// staged for. These are the payloads the plan structs carry across the
// prepare/capture boundary (design §6) — pure data, device-pointer-free.

/// The fa2 decode schedule's workspace map. An offset is `Some` exactly
/// when the schedule laid that table out; `None` seats resolve to the
/// workspace base and are guarded off by `split_kv`/`enable_cuda_graph`
/// before the device ever reads them.
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

/// The fa2 prefill schedule's workspace map; offsets as on
/// [`DecodePlanInfo`].
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

/// The sm90 prefill schedule's workspace map. The builder exists so the
/// `AttnPrefillPlanSm90` struct kind has an honest payload; the sm90
/// launcher itself was never part of this lattice (see `attn::prefill_sm90`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefillPlanSm90Info {
    pub qo_tile_indices_offset: Option<u32>,
    pub qo_indptr_offset: Option<u32>,
    pub kv_indptr_offset: Option<u32>,
    pub qo_len_offset: Option<u32>,
    pub kv_len_offset: Option<u32>,
    pub head_indices_offset: Option<u32>,
    pub work_indptr_offset: Option<u32>,
    pub batch_indices_offset: Option<u32>,
    pub same_schedule_for_all_heads: bool,
}

/// The latent-attention schedule's workspace map; offsets as on
/// [`DecodePlanInfo`] (every seat is laid out on this schedule).
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


