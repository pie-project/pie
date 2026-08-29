//! The fa2 decode work split: given the host copy of the kv page indptr it
//! decides whether to split kv, partitions requests into `(request, kv
//! tile)` work items, and stages the index vectors the decode kernel walks.
//! A native reimplementation of FlashInfer's host planner — the schedule is
//! valid and deterministic, not byte-identical to the C++ reference (see
//! [`sched`](crate::attn::sched)).

use kernels::KernelError;

use crate::attn::plan::{Built, DecodePlanInfo, Sizes, Toggles};
use crate::attn::sched::{at, AlignedAllocator, Staging, narrow, narrow_all, spans};
use crate::jit::refuse;

#[derive(Clone, Copy, Debug)]
pub struct Request<'a> {
    /// Host copy of the kv page indptr — `[batch_size + 1]`.
    pub kv_indptr: &'a [i32],
    pub batch_size: u32,
    pub num_qo_heads: u32,
    pub gqa_group_size: u32,
    pub page_size: u32,
    pub head_dim: u32,
    pub enable_cuda_graph: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkEstimate {
    pub split_kv: bool,
    pub kv_chunk_size_in_pages: u32,
    pub new_batch_size: u32,
    pub gdy: u32,
}

/// Decides the kv split: the chunk width in pages and the work-item count
/// it produces against the grid the occupancy probe granted.
pub fn estimate(
    op: &'static str,
    req: &Request<'_>,
    max_grid_size: u32,
) -> Result<WorkEstimate, KernelError> {
    let pages = spans(op, "kv_indptr", req.kv_indptr, req.batch_size as usize)?;
    if req.gqa_group_size == 0 {
        return Err(refuse(op, "the GQA group this schedule states is zero"));
    }
    let gdy = req.num_qo_heads / req.gqa_group_size;
    if gdy == 0 {
        return Err(refuse(op, "this schedule states no query heads"));
    }
    if req.page_size == 0 {
        return Err(refuse(op, "the pool's page size is zero"));
    }

    // A batch that already fills the grid takes one work item per request.
    if u64::from(req.batch_size) * u64::from(gdy) >= u64::from(max_grid_size) {
        return Ok(WorkEstimate {
            split_kv: false,
            kv_chunk_size_in_pages: pages.iter().copied().max().unwrap_or(0).max(1),
            new_batch_size: req.batch_size,
            gdy,
        });
    }

    // Otherwise, the smallest chunk whose work items still fit the grid.
    let mut low = (128 / req.page_size).max(1);
    let mut high = pages.iter().copied().max().unwrap_or(0);
    while low < high {
        let mid = u32::midpoint(low, high);
        let filled: u64 = pages.iter().map(|&p| u64::from(p.div_ceil(mid))).sum();
        if filled * u64::from(gdy) > u64::from(max_grid_size) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    let filled: u64 = pages.iter().map(|&p| u64::from(p.max(1).div_ceil(low))).sum();
    let new_batch_size = u32::try_from(filled)
        .map_err(|_| refuse(op, "the split's work items do not fit a 32-bit count"))?;

    Ok(WorkEstimate {
        split_kv: req.enable_cuda_graph || new_batch_size != req.batch_size,
        kv_chunk_size_in_pages: low,
        new_batch_size,
        gdy,
    })
}

/// The computed schedule: pure data, laid out and staged by [`plan`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schedule {
    pub split_kv: bool,
    pub enable_cuda_graph: bool,
    pub kv_chunk_size_in_pages: u32,
    pub new_batch_size: u32,
    pub padded_batch_size: usize,
    pub request_indices: Vec<i32>,
    pub kv_tile_indices: Vec<i32>,
    pub o_indptr: Vec<i32>,
}

pub fn schedule(
    op: &'static str,
    req: &Request<'_>,
    max_grid_size: u32,
) -> Result<Schedule, KernelError> {
    let est = estimate(op, req, max_grid_size)?;
    let pages = spans(op, "kv_indptr", req.kv_indptr, req.batch_size as usize)?;

    let mut request_indices = Vec::new();
    let mut kv_tile_indices = Vec::new();
    let mut o_indptr = vec![0i64];
    // Every index pushed here is bounded by the work-item total, which the
    // narrowed o_indptr tail below bounds at the device's i32 — the casts
    // are plain.
    for (batch_idx, &p) in pages.iter().enumerate() {
        let chunks = p.max(1).div_ceil(est.kv_chunk_size_in_pages);
        for kv_tile in 0..chunks {
            request_indices.push(batch_idx as i32);
            kv_tile_indices.push(kv_tile as i32);
        }
        o_indptr.push(o_indptr.last().expect("o_indptr starts with a zero") + i64::from(chunks));
    }
    let o_indptr = narrow_all(op, "batch_decode_o_indptr", &o_indptr)?;

    let padded_batch_size = if req.enable_cuda_graph {
        if est.split_kv {
            (max_grid_size / est.gdy) as usize
        } else {
            req.batch_size as usize
        }
    } else {
        est.new_batch_size as usize
    };
    if request_indices.len() > padded_batch_size {
        return Err(refuse(
            op,
            format!(
                "the split produced {} work items over a padded batch of {padded_batch_size}",
                request_indices.len()
            ),
        ));
    }

    Ok(Schedule {
        split_kv: est.split_kv,
        enable_cuda_graph: req.enable_cuda_graph,
        kv_chunk_size_in_pages: est.kv_chunk_size_in_pages,
        new_batch_size: est.new_batch_size,
        padded_batch_size,
        request_indices,
        kv_tile_indices,
        o_indptr,
    })
}

/// The offsets a schedule occupies in the granted workspace, assigned but
/// not yet written.
struct Laid {
    info: DecodePlanInfo,
    int_bytes: usize,
    float_bytes: usize,
}

fn layout(
    op: &'static str,
    num_qo_heads: u32,
    head_dim: u32,
    sched: &Schedule,
    int_space: usize,
    float_space: usize,
) -> Result<Laid, KernelError> {
    let padded = sched.padded_batch_size;
    let mut info = DecodePlanInfo {
        enable_cuda_graph: sched.enable_cuda_graph,
        split_kv: sched.split_kv,
        padded_batch_size: padded as i64,
        ..DecodePlanInfo::default()
    };

    let mut ints = AlignedAllocator::new(op, int_space);
    info.request_indices_offset = Some(ints.alloc(padded * 4, 16, "batch_decode_request_indices")?);
    info.kv_tile_indices_offset = Some(ints.alloc(padded * 4, 16, "batch_decode_kv_tile_indices")?);
    info.o_indptr_offset = Some(ints.alloc((padded + 1) * 4, 16, "batch_decode_o_indptr")?);
    info.kv_chunk_size_ptr_offset = Some(ints.alloc(4, 1, "batch_decode_kv_chunk_size_ptr")?);

    let mut floats = AlignedAllocator::new(op, float_space);
    if sched.split_kv {
        let heads = u64::from(num_qo_heads);
        let head_dim = u64::from(head_dim);
        info.v_offset = Some(floats.alloc((heads * padded as u64 * head_dim * 4) as usize, 16, "batch_decode_tmp_v")?);
        info.s_offset = Some(floats.alloc((heads * padded as u64 * 4) as usize, 16, "batch_decode_tmp_s")?);
        info.block_valid_mask_offset = Some(ints.alloc(padded, 16, "batch_decode_block_valid_mask")?);
    }

    Ok(Laid {
        info,
        int_bytes: ints.used(),
        float_bytes: floats.used(),
    })
}

fn stage(
    op: &'static str,
    sched: &Schedule,
    page_size: u32,
    laid: &Laid,
) -> Result<Vec<u8>, KernelError> {
    let info = &laid.info;
    let mut staging = Staging::new(op, laid.int_bytes);
    staging.put_i32s(
        at(info.request_indices_offset),
        &sched.request_indices,
        "batch_decode_request_indices",
    )?;
    staging.put_i32s(
        at(info.kv_tile_indices_offset),
        &sched.kv_tile_indices,
        "batch_decode_kv_tile_indices",
    )?;
    staging.put_i32s(
        at(info.o_indptr_offset),
        &sched.o_indptr,
        "batch_decode_o_indptr",
    )?;
    staging.put_i32(
        at(info.kv_chunk_size_ptr_offset),
        narrow(
            op,
            "batch_decode_kv_chunk_size_ptr",
            i64::from(sched.kv_chunk_size_in_pages) * i64::from(page_size),
        )?,
        "batch_decode_kv_chunk_size_ptr",
    )?;
    if sched.split_kv {
        staging.put_bools(
            at(info.block_valid_mask_offset),
            (0..sched.padded_batch_size).map(|i| i < sched.new_batch_size as usize),
            "batch_decode_block_valid_mask",
        )?;
    }
    Ok(staging.into_upload(laid.int_bytes))
}

pub fn plan(
    op: &'static str,
    req: &Request<'_>,
    max_grid_size: u32,
    int_bytes: usize,
    float_bytes: usize,
) -> Result<Built<DecodePlanInfo>, KernelError> {
    let sched = schedule(op, req, max_grid_size)?;
    let laid = layout(op, req.num_qo_heads, req.head_dim, &sched, int_bytes, float_bytes)?;
    let int_upload = stage(op, &sched, req.page_size, &laid)?;
    Ok(Built {
        info: laid.info,
        int_upload,
        int_bytes: laid.int_bytes,
        float_bytes: laid.float_bytes,
    })
}

/// The sizing pass: schedule and layout only, unbounded, so the engine can
/// learn the workspace a plan would need before granting one.
pub fn workspace_size(
    op: &'static str,
    req: &Request<'_>,
    max_grid_size: u32,
) -> Result<Sizes, KernelError> {
    let sched = schedule(op, req, max_grid_size)?;
    let laid = layout(
        op,
        req.num_qo_heads,
        req.head_dim,
        &sched,
        usize::MAX,
        usize::MAX,
    )?;
    Ok(Sizes {
        float_bytes: laid.float_bytes,
        int_bytes: laid.int_bytes,
    })
}

/// The static non-split fast path: small batches on cc >= 8 skip the
/// planner entirely — one work item per request, no kv split.
#[must_use]
pub fn can_use_static_nonsplit(
    num_requests: u32,
    cc_major: u32,
    enable_cuda_graph: bool,
    toggles: Toggles,
) -> bool {
    !enable_cuda_graph && !toggles.force_split_small && cc_major >= 8 && num_requests <= 512
}

/// Builds the static schedule [`can_use_static_nonsplit`] admits, through
/// the same layout and staging the planned path takes.
pub fn static_nonsplit(
    op: &'static str,
    num_requests: u32,
    page_size: u32,
    enable_cuda_graph: bool,
    int_bytes: usize,
) -> Result<Built<DecodePlanInfo>, KernelError> {
    let n = narrow(op, "batch_decode_request_indices", i64::from(num_requests))?;
    let sched = Schedule {
        split_kv: false,
        enable_cuda_graph,
        kv_chunk_size_in_pages: 1,
        new_batch_size: num_requests,
        padded_batch_size: num_requests as usize,
        request_indices: (0..n).collect(),
        kv_tile_indices: vec![0; num_requests as usize],
        o_indptr: (0..=n).collect(),
    };
    let laid = layout(op, 0, 0, &sched, int_bytes, usize::MAX)?;
    let int_upload = stage(op, &sched, page_size, &laid)?;
    Ok(Built {
        info: laid.info,
        int_upload,
        int_bytes: laid.int_bytes,
        float_bytes: laid.float_bytes,
    })
}
