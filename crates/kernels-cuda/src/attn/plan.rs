use crate::error::Error;

use crate::attn::{sched_decode, sched_mla, sched_prefill, sched_sm90};

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
    pub const L40S: Self = Self {
        num_sm: 148,
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
            "attention.plan_decode",
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
            && self.shape.num_kv_heads != kv_heads {
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
            "attention.plan_prefill",
            &self.int_upload,
            self.workspace.int_ptr,
        )
    }
}

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
    pub fn stage(&self, ctx: &Ctx) -> Result<(), Error> {
        upload(
            ctx,
            "attention.plan_prefill_sm90",
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
        upload(ctx, "attention.mla_plan", &self.int_upload, self.workspace.int_ptr)
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
}

impl Toggles {
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
            live,
            shape.page_size,
            enable_cuda_graph,
            workspace.int_bytes,
        )?
    } else {
        let req = sched_decode::Request {
            kv_indptr,
            batch_size: shape.num_requests,
            live,
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
    workspace: Workspace,
) -> Result<PrefillPlan, Error> {
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
        lane_offset: shape.lane_offset,
        live,
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

pub use crate::attn::sched_prefill::graph_padding as prefill_graph_padding;

#[allow(clippy::too_many_arguments)]
pub fn plan_prefill_sm90(
    qo_indptr: &[i32],
    kv_indptr: &[i32],
    kv_len: &[i32],
    total_tokens: u32,
    shape: Shape,
    live: Live,
    causal: bool,
    enable_cuda_graph: bool,
    device: &Device,
    workspace: Workspace,
) -> Result<PrefillPlanSm90, Error> {
    const OP: &str = "attention.plan_prefill_sm90";
    some_requests(OP, &shape)?;
    let req = sched_sm90::Request {
        qo_indptr,
        kv_indptr,
        kv_len_arr: kv_len,
        total_num_rows: total_tokens,
        batch_size: shape.num_requests,
        live,
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
    const OP: &str = "attention.mla_plan";
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
    #[cfg(feature = "cuda")]
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
