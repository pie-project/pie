//! The fa2 prefill work split: tiles the packed query axis, binary-searches
//! the kv chunk size that fills the grid, and stages the tile/merge index
//! vectors the prefill kernel and the cascade merge walk. A native
//! reimplementation of FlashInfer's host planner — the schedule is valid
//! and deterministic, not byte-identical to the C++ reference (see
//! [`sched`](crate::attn::sched)).

use kernels::KernelError;

use crate::attn::plan::{Built, Device, PrefillPlanInfo, Sizes};
use crate::attn::sched::{at, AlignedAllocator, Staging, narrow, narrow_all, spans};
use crate::jit::refuse;

#[derive(Clone, Copy, Debug)]
pub struct Request<'a> {
    /// Host copy of the query indptr — `[batch_size + 1]`.
    pub qo_indptr: &'a [i32],
    /// Host copy of the kv page indptr — `[batch_size + 1]`.
    pub kv_indptr: &'a [i32],
    pub total_num_rows: u32,
    pub batch_size: u32,
    pub num_qo_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub page_size: u32,
    pub enable_cuda_graph: bool,
    /// The sliding extent as the device reads it — `window - 1`, already
    /// validated by `plan::window_left`; `None` is the full reading.
    pub window_left: Option<u32>,
}

impl Request<'_> {
    fn check(&self, op: &'static str) -> Result<(), KernelError> {
        if self.batch_size == 0 {
            return Err(refuse(op, "the batch is empty"));
        }
        if self.num_kv_heads == 0 || !self.num_qo_heads.is_multiple_of(self.num_kv_heads) {
            return Err(refuse(
                op,
                format!(
                    "{} query heads are not a whole number of the {} kv heads",
                    self.num_qo_heads, self.num_kv_heads
                ),
            ));
        }
        if self.num_qo_heads == 0 {
            return Err(refuse(op, "this schedule states no query heads"));
        }
        if self.page_size == 0 {
            return Err(refuse(op, "the pool's page size is zero"));
        }
        Ok(())
    }
}

/// FlashInfer's CTA tile chooser for the fa2 prefill query axis.
#[must_use]
const fn determine_cta_tile_q(avg_packed_qo_len: u64, head_dim: u32, cc_major: u32) -> u32 {
    if head_dim >= 512 {
        if avg_packed_qo_len <= 32 {
            return 16;
        }
        return 32;
    }
    if avg_packed_qo_len > 64 && head_dim < 256 {
        128
    } else if cc_major >= 8 {
        if avg_packed_qo_len > 16 { 64 } else { 16 }
    } else {
        64
    }
}

/// The smallest kv chunk (in pages) whose work items still fit the grid,
/// and whether that chunking actually splits anything.
fn search_kv_chunk_size(
    enable_cuda_graph: bool,
    max_batch_size_if_split: u32,
    packed_qo_lens: &[u64],
    effective_kv_lens: &[u64],
    qo_chunk_size: u32,
    min_kv_chunk_size: u64,
) -> (bool, u64) {
    let max_kv_len = effective_kv_lens.iter().copied().max().unwrap_or(0).max(1);
    let mut low = min_kv_chunk_size;
    let mut high = max_kv_len;
    while low < high {
        let mid = u64::midpoint(low, high);
        let work_items: u64 = packed_qo_lens
            .iter()
            .zip(effective_kv_lens)
            .map(|(&packed, &kv)| {
                packed.div_ceil(u64::from(qo_chunk_size)) * kv.max(1).div_ceil(mid)
            })
            .sum();
        if work_items > u64::from(max_batch_size_if_split) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    (enable_cuda_graph || low < max_kv_len, low)
}

/// The computed schedule: pure data, laid out and staged by [`plan`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schedule {
    pub split_kv: bool,
    pub new_batch_size: u32,
    pub padded_batch_size: usize,
    pub cta_tile_q: u32,
    /// The chunk width in tokens, as the staged scalar spells it.
    pub kv_chunk_size: u64,
    pub request_indices: Vec<i32>,
    pub qo_tile_indices: Vec<i32>,
    pub kv_tile_indices: Vec<i32>,
    pub merge_indptr: Vec<i32>,
    pub o_indptr: Vec<i32>,
}

#[allow(clippy::too_many_lines)]
pub fn schedule(
    op: &'static str,
    req: &Request<'_>,
    device: &Device,
) -> Result<Schedule, KernelError> {
    req.check(op)?;
    let batch = req.batch_size as usize;
    let qo_lens = spans(op, "qo_indptr", req.qo_indptr, batch)?;
    let kv_pages = spans(op, "kv_indptr", req.kv_indptr, batch)?;
    let group = u64::from(req.num_qo_heads / req.num_kv_heads);

    let max_grid_size = 2 * u64::from(device.num_sm);
    let max_batch_size_if_split =
        u32::try_from(max_grid_size / u64::from(req.num_kv_heads)).unwrap_or(u32::MAX);

    let packed_qo_lens: Vec<u64> = qo_lens.iter().map(|&q| u64::from(q) * group).collect();
    let min_kv_chunk_size = u64::from((128 / req.page_size).max(1));

    let (cta_tile_q, total_num_tiles_q) = if req.enable_cuda_graph {
        // The graph shape assumes the worst single request: all rows on one
        // lane, the rest empty.
        let max_seq_len =
            u64::from(req.total_num_rows).saturating_sub(u64::from(req.batch_size) - 1);
        let tile = determine_cta_tile_q(max_seq_len * group, req.head_dim, device.cc_major);
        let tiles = (u64::from(req.total_num_rows) * group).div_ceil(u64::from(tile))
            + u64::from(req.batch_size)
            - 1;
        (tile, tiles)
    } else {
        let sum: u64 = packed_qo_lens.iter().sum();
        let tile = determine_cta_tile_q(
            sum / u64::from(req.batch_size),
            req.head_dim,
            device.cc_major,
        );
        let tiles = packed_qo_lens
            .iter()
            .map(|&packed| packed.div_ceil(u64::from(tile)))
            .sum();
        (tile, tiles)
    };

    // A window shortens every prefix to the pages the sliding extent can
    // actually touch.
    let effective_kv_lens: Vec<u64> = kv_pages
        .iter()
        .map(|&pages| {
            let full = u64::from(pages);
            match req.window_left {
                Some(window_left) => (u64::from(window_left) + u64::from(cta_tile_q))
                    .div_ceil(u64::from(req.page_size))
                    .min(full),
                None => full,
            }
        })
        .collect();

    let (split_kv, kv_chunk_size_in_pages) = search_kv_chunk_size(
        req.enable_cuda_graph,
        max_batch_size_if_split,
        &packed_qo_lens,
        &effective_kv_lens,
        cta_tile_q,
        min_kv_chunk_size,
    );

    let mut request_indices = Vec::new();
    let mut qo_tile_indices = Vec::new();
    let mut kv_tile_indices = Vec::new();
    let mut merge_indptr = vec![0i64];
    let mut o_indptr = vec![0i64];
    let mut new_batch_size: u64 = 0;
    for (request_idx, (&packed, &kv)) in
        packed_qo_lens.iter().zip(&effective_kv_lens).enumerate()
    {
        let num_tiles_q = packed.div_ceil(u64::from(cta_tile_q));
        let num_chunks_kv = kv.max(1).div_ceil(kv_chunk_size_in_pages);
        // Every index pushed here is bounded by the work-item total, which
        // the refusal below bounds at the device's i32 — the casts are
        // plain.
        for q_tile_idx in 0..num_tiles_q {
            for kv_tile_idx in 0..num_chunks_kv {
                new_batch_size += 1;
                request_indices.push(request_idx as i32);
                qo_tile_indices.push(q_tile_idx as i32);
                kv_tile_indices.push(kv_tile_idx as i32);
            }
        }

        let qo_len = packed / group;
        let merge_step = num_chunks_kv as i64;
        for _ in 0..qo_len {
            merge_indptr
                .push(merge_indptr.last().expect("merge_indptr starts with a zero") + merge_step);
        }
        o_indptr.push(
            o_indptr.last().expect("o_indptr starts with a zero") + qo_len as i64 * merge_step,
        );
    }
    let merge_indptr = narrow_all(op, "batch_prefill_merge_indptr", &merge_indptr)?;
    let o_indptr = narrow_all(op, "batch_prefill_o_indptr", &o_indptr)?;

    let padded_batch_size = if req.enable_cuda_graph {
        u64::from(max_batch_size_if_split).max(total_num_tiles_q)
    } else {
        new_batch_size
    };
    if new_batch_size > padded_batch_size {
        return Err(refuse(
            op,
            format!(
                "the split produced {new_batch_size} work items over a padded batch of \
                 {padded_batch_size}"
            ),
        ));
    }
    let padded_batch_size = usize::try_from(padded_batch_size)
        .map_err(|_| refuse(op, "the padded batch does not fit this host's address space"))?;

    Ok(Schedule {
        split_kv,
        new_batch_size: u32::try_from(new_batch_size)
            .ok()
            .filter(|&n| n <= i32::MAX as u32)
            .ok_or_else(|| refuse(op, "the split's work items do not fit the device's i32"))?,
        padded_batch_size,
        cta_tile_q,
        kv_chunk_size: kv_chunk_size_in_pages * u64::from(req.page_size),
        request_indices,
        qo_tile_indices,
        kv_tile_indices,
        merge_indptr,
        o_indptr,
    })
}

/// The offsets a schedule occupies in the granted workspace, assigned but
/// not yet written.
struct Laid {
    info: PrefillPlanInfo,
    int_bytes: usize,
    float_bytes: usize,
}

fn layout(
    op: &'static str,
    req: &Request<'_>,
    sched: &Schedule,
    int_space: usize,
    float_space: usize,
) -> Result<Laid, KernelError> {
    let padded = sched.padded_batch_size;
    let mut info = PrefillPlanInfo {
        cta_tile_q: i64::from(sched.cta_tile_q),
        total_num_rows: i64::from(req.total_num_rows),
        enable_cuda_graph: req.enable_cuda_graph,
        padded_batch_size: padded as i64,
        split_kv: sched.split_kv,
        ..PrefillPlanInfo::default()
    };

    let mut ints = AlignedAllocator::new(op, int_space);
    info.request_indices_offset = Some(ints.alloc(4 * padded, 16, "batch_prefill_request_indices")?);
    info.qo_tile_indices_offset = Some(ints.alloc(4 * padded, 16, "batch_prefill_qo_tile_indices")?);
    info.kv_tile_indices_offset = Some(ints.alloc(4 * padded, 16, "batch_prefill_kv_tile_indices")?);
    info.o_indptr_offset = Some(ints.alloc(4 * (req.batch_size as usize + 1), 16, "batch_prefill_o_indptr")?);
    info.kv_chunk_size_ptr_offset = Some(ints.alloc(4, 1, "batch_prefill_kv_chunk_size_ptr")?);
    if req.enable_cuda_graph {
        info.total_num_rows_offset = Some(ints.alloc(4, 16, "batch_prefill_total_num_rows")?);
    }

    let mut floats = AlignedAllocator::new(op, float_space);
    if sched.split_kv {
        let heads = u64::from(req.num_qo_heads);
        let tile_q = u64::from(sched.cta_tile_q);
        let head_dim = u64::from(req.head_dim);
        info.v_offset = Some(floats.alloc(
                (heads * padded as u64 * tile_q * head_dim * 4) as usize,
                16,
                "batch_prefill_tmp_v",
            )?);
        info.s_offset = Some(floats.alloc((heads * padded as u64 * tile_q * 4) as usize, 16, "batch_prefill_tmp_s")?);
        info.merge_indptr_offset = Some(ints.alloc(4 * (req.total_num_rows as usize + 1), 16, "batch_prefill_merge_indptr")?);
        info.block_valid_mask_offset = Some(ints.alloc(padded, 16, "batch_prefill_block_valid_mask")?);
    }

    Ok(Laid {
        info,
        int_bytes: ints.used(),
        float_bytes: floats.used(),
    })
}

fn stage(
    op: &'static str,
    req: &Request<'_>,
    sched: &Schedule,
    laid: &Laid,
) -> Result<Vec<u8>, KernelError> {
    let info = &laid.info;
    let mut staging = Staging::new(op, laid.int_bytes);
    staging.put_i32s(
        at(info.request_indices_offset),
        &sched.request_indices,
        "batch_prefill_request_indices",
    )?;
    staging.put_i32s(
        at(info.qo_tile_indices_offset),
        &sched.qo_tile_indices,
        "batch_prefill_qo_tile_indices",
    )?;
    staging.put_i32s(
        at(info.kv_tile_indices_offset),
        &sched.kv_tile_indices,
        "batch_prefill_kv_tile_indices",
    )?;
    staging.put_i32s(
        at(info.o_indptr_offset),
        &sched.o_indptr,
        "batch_prefill_o_indptr",
    )?;
    staging.put_i32(
        at(info.kv_chunk_size_ptr_offset),
        narrow(
            op,
            "batch_prefill_kv_chunk_size_ptr",
            i64::try_from(sched.kv_chunk_size)
                .map_err(|_| refuse(op, "the kv chunk width does not fit the device's i32"))?,
        )?,
        "batch_prefill_kv_chunk_size_ptr",
    )?;
    if req.enable_cuda_graph {
        staging.put_i32(
            at(info.total_num_rows_offset),
            req.qo_indptr[req.batch_size as usize],
            "batch_prefill_total_num_rows",
        )?;
    }
    if sched.split_kv {
        staging.put_i32s(
            at(info.merge_indptr_offset),
            &sched.merge_indptr,
            "batch_prefill_merge_indptr",
        )?;
        staging.put_bools(
            at(info.block_valid_mask_offset),
            (0..sched.padded_batch_size).map(|i| i < sched.new_batch_size as usize),
            "batch_prefill_block_valid_mask",
        )?;
    }
    Ok(staging.into_upload(laid.int_bytes))
}

pub fn plan(
    op: &'static str,
    req: &Request<'_>,
    device: &Device,
    int_bytes: usize,
    float_bytes: usize,
) -> Result<Built<PrefillPlanInfo>, KernelError> {
    let sched = schedule(op, req, device)?;
    let laid = layout(op, req, &sched, int_bytes, float_bytes)?;
    let int_upload = stage(op, req, &sched, &laid)?;
    Ok(Built {
        info: laid.info,
        int_upload,
        int_bytes: laid.int_bytes,
        float_bytes: laid.float_bytes,
    })
}

/// The sizing pass: schedule and layout only, unbounded, so the driver can
/// learn the workspace a plan would need before granting one.
pub fn workspace_size(
    op: &'static str,
    req: &Request<'_>,
    device: &Device,
) -> Result<Sizes, KernelError> {
    let sched = schedule(op, req, device)?;
    let laid = layout(op, req, &sched, usize::MAX, usize::MAX)?;
    Ok(Sizes {
        float_bytes: laid.float_bytes,
        int_bytes: laid.int_bytes,
    })
}
