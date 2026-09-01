//! The fa2 decode work split: given the host copy of the kv page indptr it
//! decides whether to split kv, partitions requests into `(request, kv
//! tile)` work items, and stages the index vectors the decode kernel walks.
//! A native reimplementation of FlashInfer's host planner — the schedule is
//! valid and deterministic, not byte-identical to the C++ reference (see
//! [`sched`](crate::attn::sched)).

use crate::error::Error;

use crate::attn::plan::{Built, DecodePlanInfo, Live, Sizes, Toggles};
use crate::attn::sched::{at, AlignedAllocator, Staging, narrow, narrow_all, spans};
use crate::jit::refuse;

#[derive(Clone, Copy, Debug)]
pub struct Request<'a> {
    /// Host copy of the kv page indptr — `[batch_size + 1]`.
    pub kv_indptr: &'a [i32],
    /// The lane count this schedule is CARVED for: what the padding, the
    /// allocations and the grid arithmetic below are sized at.
    ///
    /// **AND SINCE THE DECODE CEILING IT IS ROUTINELY LARGER THAN THE FIRE'S**
    /// (`engine_cuda::run::Run::planning`): a bodied fire carves at the
    /// bucket's lane ceiling so that the hashed image of the plan stops
    /// following the batch. Nothing in this module moves with that except the
    /// allocations and the padding — see [`schedule`] for the argument that
    /// every work item past the live ones is masked off.
    pub batch_size: u32,
    /// **AND WHAT THIS FIRE ACTUALLY BROUGHT** — the origin-and-extent
    /// channel ([`Live`]). Three readers here and no fourth: the staged
    /// request ids, which are `live.lane_offset + r` (and see `schedule` for
    /// why `o_indptr` does not move with them); the walk over `kv_indptr`'s
    /// live contents; and the work-item count `block_valid_mask` retires the
    /// padding against.
    pub live: Live,
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
) -> Result<WorkEstimate, Error> {
    let pages = spans(op, "kv_indptr", req.kv_indptr, req.live.requests as usize)?;
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
            // **AND IT STILL SPLITS UNDER CAPTURE**, which is the one word
            // this arm used to answer without looking at the toggle. Two
            // things hang off `split_kv` that a captured launch cannot do
            // without: `layout` lays `block_valid_mask` only when it is set,
            // and that mask is the ONLY thing that retires a work item the
            // recorded grid over-launched; and `schedule` pads the batch to
            // the grid only when it is set, so a `false` here bakes THIS
            // fire's live lane count into a graph the next fire replays at
            // another. Every other decode arm already splits under capture —
            // the second return below does, and `can_use_static_nonsplit`
            // refuses the fast path outright — so this was the one corner
            // where over-launched work items would run unretired. The arm
            // itself is unchanged: the chunk still spans the longest lane, so
            // the split it now declares is one work item per request, exactly
            // the schedule this arm has always produced.
            split_kv: req.enable_cuda_graph,
            kv_chunk_size_in_pages: pages.iter().copied().max().unwrap_or(0).max(1),
            // The count of work items this arm actually emits — one per LIVE
            // lane, because the chunk spans the longest of them — and so the
            // count `block_valid_mask` retires the padding against. The test
            // above is the other half: which arm to take is grid arithmetic,
            // and grid arithmetic reads the carved count.
            new_batch_size: req.live.requests,
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
) -> Result<Schedule, Error> {
    let est = estimate(op, req, max_grid_size)?;
    let pages = spans(op, "kv_indptr", req.kv_indptr, req.live.requests as usize)?;

    let mut request_indices = Vec::new();
    let mut kv_tile_indices = Vec::new();
    // **AND `o_indptr` STAYS WINDOW-LOCAL WHERE THE REQUEST IDS GO
    // ABSOLUTE**, which is the opposite of `sched_prefill`'s and for the
    // opposite reason: `decode.cuh` never reads it. The decode kernel writes
    // its output at the WORK ITEM (`o + bx * ...`), and this vector's one
    // consumer is the cascade fold, which walks it at `pos` in
    // `[0, num_requests)` — the launch's own request number, from its own
    // zero. Adding `lane_offset` here would send the fold's first lane to a
    // dead entry.
    let mut o_indptr = vec![0i64];
    // Every index pushed here is bounded by the work-item total, which the
    // narrowed o_indptr tail below bounds at the device's i32 — the casts
    // are plain.
    for (batch_idx, &p) in pages.iter().enumerate() {
        let chunks = p.max(1).div_ceil(est.kv_chunk_size_in_pages);
        for kv_tile in 0..chunks {
            request_indices.push((req.live.lane_offset as u64 + batch_idx as u64) as i32);
            kv_tile_indices.push(kv_tile as i32);
        }
        o_indptr.push(o_indptr.last().expect("o_indptr starts with a zero") + i64::from(chunks));
    }
    let o_indptr = narrow_all(op, "batch_decode_o_indptr", &o_indptr)?;

    // **THE PADDED BATCH IS THE GRID OR THE LANE COUNT, WHICHEVER IS LARGER.**
    // `estimate` splits unconditionally under capture now, so the padding is
    // the grid the occupancy probe granted — but the loop above floors every
    // lane at ONE work item (`p.max(1)`), zero-page lanes included, where the
    // search that sized the chunk counted a zero-page lane as zero. A fire
    // with more lanes than `max_grid_size / gdy` therefore emits more work
    // items than that grid holds and trips the refusal below, which is a fire
    // the shell has to serve rather than refuse. The lane count is this
    // window's `num_requests`, which the graph key holds fixed, so the larger
    // of the two is still a function of the KEY and not of this fire's kv
    // lengths — which is the whole property the graph arm exists to keep.
    //
    // What it does not cover is the SUM of the two: a fire whose zero-page
    // lanes and whose split live lanes together outrun both numbers. That one
    // still refuses, and the refusal is the honest answer — padding every
    // capture's partial planes to `grid + lanes` would size the workspace for
    // a case the key cannot promise, and a refusal here is a named fault
    // rather than a wrong schedule.
    //
    // **AND EVERY WORK ITEM THIS PADDING BUYS IS PROVABLY DEAD** — which is
    // the question a ceiling carve (`Run::planning`) asks of this loop,
    // because `batch_size` is then the bucket's lane count and the walk above
    // is over `live.requests` of them. The walk STAYS LIVE and the grid is
    // covered by the padding, rather than the walk running out to the ceiling
    // and emitting a one-page item per empty lane; the argument that this is
    // the safe half of the choice is three lines long:
    //
    // * `request_indices.len()` IS `est.new_batch_size`. In the first arm the
    //   chunk spans the longest lane, so every lane emits exactly one item and
    //   the arm returns `live.requests`; in the second the arm returns the
    //   same `sum p.max(1).div_ceil(chunk)` this loop pushes. One expression,
    //   written twice, and the two are checked against each other by the
    //   refusal just below.
    // * `stage` writes `block_valid_mask[i] = i < new_batch_size` over the
    //   WHOLE padded batch, so every index past the emitted items is `false`,
    //   and `decode.cuh` retires those blocks before it reads anything else.
    //   The mask is laid out and bound exactly when `split_kv` is, and
    //   `estimate` sets `split_kv` unconditionally under capture — which a
    //   bodied fire always is (`FireBindings::capture` is the load's
    //   `Graphs::shaped()`), so a ceiling carve never reaches an arm without
    //   one.
    // * and what a masked block would have read is zeros anyway: `Staging`
    //   zero-fills the whole grant before a single `put`, so the padded tail
    //   of `request_indices` and `kv_tile_indices` names lane 0's tile 0
    //   rather than the last fire's numbers.
    //
    // The other half — walking out to the ceiling and letting `p.max(1)` emit
    // an item per EMPTY lane — would have to prove those items masked too, and
    // they would not be: they would fall inside `new_batch_size` and run,
    // reading the empty lane chunk 2 staged. Correct, but wasted grid and one
    // more thing to argue; this way there is nothing to argue.
    let padded_batch_size = if req.enable_cuda_graph {
        ((max_grid_size / est.gdy) as usize).max(req.batch_size as usize)
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
) -> Result<Laid, Error> {
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
) -> Result<Vec<u8>, Error> {
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
) -> Result<Built<DecodePlanInfo>, Error> {
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
) -> Result<Sizes, Error> {
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
    live: Live,
    page_size: u32,
    enable_cuda_graph: bool,
    int_bytes: usize,
) -> Result<Built<DecodePlanInfo>, Error> {
    // `n` is the carved count — what the padding and the allocations behind
    // it are sized at; `live_n` is the count this fire actually stages.
    let n = narrow(op, "batch_decode_request_indices", i64::from(num_requests))?;
    let live_n = narrow(op, "batch_decode_request_indices", i64::from(live.requests))?;
    // The absolute ids of `schedule` above, at one work item per request; and
    // `o_indptr` window-local for that arm's reason exactly.
    let first = narrow(op, "batch_decode_request_indices", i64::from(live.lane_offset))?;
    let sched = Schedule {
        split_kv: false,
        enable_cuda_graph,
        kv_chunk_size_in_pages: 1,
        new_batch_size: live.requests,
        padded_batch_size: n as usize,
        request_indices: (first..first + live_n).collect(),
        kv_tile_indices: vec![0; live.requests as usize],
        o_indptr: (0..=live_n).collect(),
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
