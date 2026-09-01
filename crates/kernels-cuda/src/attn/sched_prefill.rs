//! The fa2 prefill work split: tiles the packed query axis, binary-searches
//! the kv chunk size that fills the grid, and stages the tile/merge index
//! vectors the prefill kernel and the cascade merge walk. A native
//! reimplementation of FlashInfer's host planner — the schedule is valid
//! and deterministic, not byte-identical to the C++ reference (see
//! [`sched`](crate::attn::sched)).

use crate::error::Error;

use crate::attn::plan::{Built, Device, Live, PrefillPlanInfo, Sizes};
use crate::attn::sched::{at, AlignedAllocator, Staging, narrow, narrow_all, spans};
use crate::jit::refuse;

#[derive(Clone, Copy, Debug)]
pub struct Request<'a> {
    /// Host copy of the query indptr — `[batch_size + 1]`.
    pub qo_indptr: &'a [i32],
    /// Host copy of the kv page indptr — `[batch_size + 1]`.
    pub kv_indptr: &'a [i32],
    /// The row and lane counts this schedule is CARVED for: the graph
    /// shape's tile arithmetic, every allocation, and the padding.
    pub total_num_rows: u32,
    pub batch_size: u32,
    /// **HOW FAR IN THE FIRE'S LANE ORDER THE `o_indptr` ALLOCATION HAS TO
    /// REACH** ([`plan::Shape::lane_offset`](crate::attn::plan::Shape)) —
    /// that vector is indexed by the absolute request id, so `layout` sizes
    /// it `[lane_offset + batch_size + 1]`. The dead prefix the SCHEDULE
    /// stages in front of its own numbers is `live.lane_offset`, which is
    /// the same number today and the origin half of it always.
    pub lane_offset: u32,
    /// **AND WHAT THIS FIRE ACTUALLY BROUGHT** ([`Live`]): the ids the work
    /// items are staged under (`live.lane_offset + r`), the dead prefix and
    /// the row those unsplit `o_indptr` entries count from
    /// (`live.row_offset`), the walk over the two indptrs' live contents,
    /// and the staged row-total word.
    pub live: Live,
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
    fn check(&self, op: &'static str) -> Result<(), Error> {
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
///
/// **AND ITS ANSWER PICKS THE KERNEL SYMBOL** — `PrefillPlan::cta_tile_q`
/// reaches `fa2::PrefillPoint`, which spells `NUM_MMA_Q`, `NUM_WARPS_Q` and
/// `NUM_WARPS_KV` into the instantiation a launch resolves. So under the
/// bucket ceiling this is a HEURISTIC THAT HAS BEEN FROZEN: the graph shape
/// feeds it the carved row and lane counts, those do not move inside a
/// `record::BodyKey`, and one key therefore captures ONE symbol.
///
/// **THE PERFORMANCE CONSEQUENCE IS REAL AND THE CORRECTNESS ONE IS NOT.**
/// A fire whose LIVE rows would have chosen a narrower tile replays the
/// ceiling's wider one — a small fire in a big bucket runs the big tile, whose
/// work items are mostly masked rows, and pays for them. What it does not pay
/// is a wrong answer: every tile in `DISPATCH_CTA_TILE_Q` computes the same
/// attention over the same rows, the tile only says how many query rows one
/// CTA carries, and the `qo_upper_bound` inside the kernel is `qo_len`'s and
/// not the tile's. Which is exactly why this number is allowed to be a
/// function of the key rather than of the fire.
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

/// **THE GRAPH SHAPE'S TILE AND WORK-ITEM COUNT**, at a stated row and lane
/// count — the branch [`schedule`] takes under `enable_cuda_graph`, lifted out
/// because it has a second caller ([`graph_padding`]) and one arithmetic is
/// the whole point of lifting it.
///
/// The graph shape assumes the worst single request: all rows on one lane, the
/// rest empty. That is what makes the tile a function of the CARVED counts
/// rather than of how this fire happened to split its rows between lanes.
fn graph_tiles(rows: u32, batch: u32, group: u64, head_dim: u32, cc_major: u32) -> (u32, u64) {
    let batch = u64::from(batch.max(1));
    let max_seq_len = u64::from(rows).saturating_sub(batch - 1);
    let tile = determine_cta_tile_q(max_seq_len * group, head_dim, cc_major);
    let tiles = (u64::from(rows) * group).div_ceil(u64::from(tile)) + batch - 1;
    (tile, tiles)
}

/// **THE TILE AND THE PADDED WORK-ITEM COUNT A GRAPH-SHAPED SCHEDULE WOULD PAD
/// TO** at a stated row and lane CEILING — `(cta_tile_q, padded_batch_size)`,
/// which is the pair the float grant is a function of.
///
/// **EXPORTED BECAUSE THE ENGINE HAS TO ASK BEFORE IT GRANTS.** A prefill
/// schedule that does not fit its float grant does not fail — it silently
/// retries without the graph shape and reports `graph_capturable = false`, and
/// a body that reads that never captures. So the shell sizes the grant from
/// the ceiling the plan-at-bucket-ceiling design carves at
/// (`engine_cuda::inputs`'s `prefill_float_bytes`), and the arithmetic it
/// sizes from is THIS one rather than a restatement of it: `layout` allocates
/// `q_heads x padded x cta_tile_q x head_dim` floats for the partials and
/// `q_heads x padded x cta_tile_q` for their log-sum-exps.
///
/// `lanes` is the lane ceiling and `rows` the row ceiling; a caller with only
/// one of them passes the fire's own number for the other, and gets the
/// schedule that fire would carve.
#[must_use]
pub fn graph_padding(
    rows: u32,
    lanes: u32,
    num_qo_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    device: &Device,
) -> (u32, u64) {
    let kv_heads = num_kv_heads.max(1);
    let group = u64::from((num_qo_heads / kv_heads).max(1));
    let (tile, tiles) = graph_tiles(rows, lanes, group, head_dim, device.cc_major);
    // `schedule`'s `max_batch_size_if_split`, which is the floor the padding
    // takes when the tiles do not reach it.
    let floor = u64::from(2 * device.num_sm.max(1)) / u64::from(kv_heads);
    (tile, floor.max(tiles))
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
pub fn schedule(op: &'static str, req: &Request<'_>, device: &Device) -> Result<Schedule, Error> {
    req.check(op)?;
    // The two indptrs bound THIS FIRE's rectangle, so the walk over them is
    // the live lane count; the carved `batch_size` beside it is what the
    // graph shape and the padding below are sized at.
    let batch = req.live.requests as usize;
    let qo_lens = spans(op, "qo_indptr", req.qo_indptr, batch)?;
    let kv_pages = spans(op, "kv_indptr", req.kv_indptr, batch)?;
    let group = u64::from(req.num_qo_heads / req.num_kv_heads);

    let max_grid_size = 2 * u64::from(device.num_sm);
    let max_batch_size_if_split =
        u32::try_from(max_grid_size / u64::from(req.num_kv_heads)).unwrap_or(u32::MAX);

    let packed_qo_lens: Vec<u64> = qo_lens.iter().map(|&q| u64::from(q) * group).collect();
    let min_kv_chunk_size = u64::from((128 / req.page_size).max(1));

    let (cta_tile_q, total_num_tiles_q) = if req.enable_cuda_graph {
        graph_tiles(
            req.total_num_rows,
            req.batch_size,
            group,
            req.head_dim,
            device.cc_major,
        )
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
    // **`o_indptr` IS INDEXED BY THE ABSOLUTE REQUEST ID, SO IT BEGINS WITH
    // `lane_offset` DEAD ENTRIES.** The prefill kernel reads
    // `o + o_indptr[request_idx] * stride` (and `o_indptr[request_idx] +
    // kv_tile_idx` when the schedule split kv),
    // and `request_idx` is now `lane_offset + r` — so the vector is
    // `[lane_offset + batch + 1]` and this window's own numbers sit at
    // `lane_offset`. The entries in front are dead: no work item names a lane
    // below `lane_offset`, and they are zeros rather than a fork in the
    // layout, because a layout that moved between the bodied and the keyed
    // path would move the schedule hash with it and the A/B would stop
    // comparing the same carving.
    //
    // Decode's `o_indptr` is the opposite reading and stays window-local; see
    // `sched_decode::schedule`.
    let mut o_indptr = vec![0i64; req.live.lane_offset as usize + 1];
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
                request_indices.push((req.live.lane_offset as u64 + request_idx as u64) as i32);
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
    // **AND AN UNSPLIT SCHEDULE'S `o_indptr` IS A ROW OF THE FIRE'S OUTPUT
    // PLANE, NOT OF THE PLAN'S WORKSPACE.** WITHOUT `split_kv` the chunk
    // width covers every request whole, so each one contributes exactly one
    // entry per query row and these numbers ARE the rows — which the kernel
    // then adds to whatever `o` points at. A launch handed the PLANE's base
    // therefore needs the fire's row and not the window's, and `row_offset`
    // is that difference (zero on every path where `o` was sliced for the
    // launch). A SPLIT schedule's numbers address the plan's partial planes,
    // which begin at zero whatever the window is, so nothing is added there
    // and the fold behind them carries the window on the staged seat instead.
    let o_indptr: Vec<i64> = if split_kv {
        o_indptr
    } else {
        o_indptr
            .into_iter()
            .map(|at| at + i64::from(req.live.row_offset))
            .collect()
    };
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
) -> Result<Laid, Error> {
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
    // `[lane_offset + batch + 1]`, because the kernel indexes this one
    // absolutely — see the vector's own note in `schedule`.
    info.o_indptr_offset = Some(ints.alloc(
        4 * (req.lane_offset as usize + req.batch_size as usize + 1),
        16,
        "batch_prefill_o_indptr",
    )?);
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
) -> Result<Vec<u8>, Error> {
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
        // **THE WORD THE FOLD READS IN PLACE OF ITS BAKED BOUND**
        // (`fa2_abi`'s `seq_len` pointer), so it is this fire's own row
        // total: the last LIVE boundary, where `info.total_num_rows` beside
        // it is the carved count the params bake.
        staging.put_i32(
            at(info.total_num_rows_offset),
            req.qo_indptr[req.live.requests as usize],
            "batch_prefill_total_num_rows",
        )?;
    }
    if sched.split_kv {
        // The staged vector is the LIVE walk's `[live rows + 1]`; `layout`
        // allocated `[carved rows + 1]` above it. Nothing reads the tail: the
        // fold walks `pos < min(win[0], *seq_len_ptr)`, which is this fire's
        // rows, and no work item names a row past them.
        staging.put_i32s(
            at(info.merge_indptr_offset),
            &sched.merge_indptr,
            "batch_prefill_merge_indptr",
        )?;
        // **AND THE PADDED WORK ITEMS ARE RETIRED BY A MASK LAID OVER THE
        // WHOLE PADDED BATCH**, which is what makes a ceiling carve safe on
        // this axis too (plan-at-bucket-ceiling, chunk 4). Three facts, and
        // each is checkable from here:
        //
        // * the mask is `padded_batch_size` long — `layout` allocates exactly
        //   that many bytes and this writes exactly that many — so every
        //   `bx` the grid runs has an entry, and the entries past
        //   `new_batch_size` are `false`.
        // * `new_batch_size` counts the LIVE work items, because the loop that
        //   emits them walks `packed_qo_lens`, which is `live.requests` long.
        //   So the true prefix is the fire's own and the padding is the
        //   ceiling's.
        // * the kernel's `if (block_valid_mask && !block_valid_mask[bx])
        //   return;` stands BEFORE it reads `request_indices[bx]`
        //   (`prefill.cuh`), so a retired item touches neither the zeros in
        //   the index vectors' tails nor the output plane.
        //
        // And the mask always exists where the padding does: padding happens
        // only under `enable_cuda_graph`, and `search_kv_chunk_size` returns
        // `split_kv` unconditionally true under that same word — so there is
        // no arm where `padded_batch_size > new_batch_size` and the mask is
        // null.
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
) -> Result<Built<PrefillPlanInfo>, Error> {
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

/// The sizing pass: schedule and layout only, unbounded, so the engine can
/// learn the workspace a plan would need before granting one.
pub fn workspace_size(
    op: &'static str,
    req: &Request<'_>,
    device: &Device,
) -> Result<Sizes, Error> {
    let sched = schedule(op, req, device)?;
    let laid = layout(op, req, &sched, usize::MAX, usize::MAX)?;
    Ok(Sizes {
        float_bytes: laid.float_bytes,
        int_bytes: laid.int_bytes,
    })
}
