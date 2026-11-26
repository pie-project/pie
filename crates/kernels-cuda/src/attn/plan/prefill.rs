use crate::error::Error;

use crate::attn::plan::sched::{AlignedAllocator, Staging, at, narrow, narrow_all, spans};
use crate::attn::plan::{Built, Device, Live, PrefillPlanInfo};
use crate::jit::refuse;

#[derive(Clone, Copy, Debug)]
pub struct Request<'a> {
    pub qo_indptr: &'a [i32],
    pub kv_indptr: &'a [i32],
    pub total_num_rows: u32,
    pub batch_size: u32,
    pub lane_offset: u32,
    pub live: Live,
    pub num_qo_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub page_size: u32,
    pub enable_cuda_graph: bool,
    pub window_left: Option<u32>,
    /// The blocks a multiprocessor holds of an eager plan's launch (the
    /// profile's `kernel.sdpa.prefill.ctas_per_sm`).
    pub ctas_per_sm: u32,
    /// The blocks a multiprocessor holds of the kernel a captured plan's
    /// launch runs, at `(head_dim, cta_tile_q)`, as the device states them
    /// (`fa2::prefill_blocks_per_sm`); `None` where it states none, and
    /// the captured plan then holds `ctas_per_sm`.
    pub blocks_per_sm: fn(u32, u32, &Device) -> Option<u32>,
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

#[must_use]
const fn determine_cta_tile_q(avg_packed_qo_len: u64, head_dim: u32) -> u32 {
    if head_dim >= 512 {
        if avg_packed_qo_len <= 32 {
            return 16;
        }
        return 32;
    }
    if avg_packed_qo_len > 64 && head_dim < 256 {
        128
    } else if avg_packed_qo_len > 16 {
        64
    } else {
        16
    }
}

fn search_kv_chunk_size(
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
    (low < max_kv_len, low)
}

/// The key chunks a captured fire whose live tiles span `blocks` blocks
/// (its tiles times the kv heads) splits its keys into, at most `cap`: the
/// count whose waves over the device's `concurrent` blocks take least, a
/// chunk count `c` costing `ceil(blocks · c / concurrent) / c` of one
/// unsplit block's time; ties go to the fewer chunks, and the keys split
/// only where that beats leaving them whole.
fn fill_chunks(blocks: u64, concurrent: u64, cap: u64) -> u64 {
    let concurrent = concurrent.max(1);
    let waves = |chunks: u64| (blocks.max(1) * chunks).div_ceil(concurrent);
    let mut best = 1;
    for chunks in 2..=cap {
        if waves(chunks) * best < waves(best) * chunks {
            best = chunks;
        }
    }
    best
}

/// The most work items a captured layout's split lays over its replays: for
/// every live tile count its grid of `grid` items holds a split of,
/// [`fill_chunks`]'s chunks of it. 0 where no replay splits.
fn split_capacity(grid: u64, heads: u64, concurrent: u64) -> u64 {
    (1..=grid / 2)
        .filter_map(|tiles| {
            let chunks = fill_chunks(tiles * heads.max(1), concurrent, grid / tiles);
            (chunks > 1).then_some(tiles * chunks)
        })
        .max()
        .unwrap_or(0)
}

fn graph_tiles(rows: u32, batch: u32, group: u64, head_dim: u32) -> (u32, u64) {
    let batch = u64::from(batch.max(1));
    let max_seq_len = u64::from(rows).saturating_sub(batch - 1);
    let tile = determine_cta_tile_q(max_seq_len * group, head_dim);
    let tiles = (u64::from(rows) * group).div_ceil(u64::from(tile)) + batch - 1;
    (tile, tiles)
}

/// The query rows one work item's partials hold: its CTA tile packs `tile`
/// rows of a request's query heads, so it spans `tile / group` query rows,
/// and the partials are laid out by query row across every head (the
/// kernel strides them by `o_indptr`), not by packed row.
#[must_use]
pub fn partial_rows(tile: u32, num_qo_heads: u32, num_kv_heads: u32) -> u64 {
    let group = (num_qo_heads / num_kv_heads.max(1)).max(1);
    u64::from(tile.div_ceil(group))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schedule {
    /// Whether the layout splits keys: a captured layout's split stands for
    /// every replay, each fire's own chunks restaged ([`Schedule::split_live`]).
    pub split_kv: bool,
    /// Whether this fire's keys split: a split layout's fire of one chunk a
    /// tile lands its rows whole, the fold folding none.
    pub split_live: bool,
    /// The work items the partials hold: a captured layout's most over its
    /// replays ([`split_capacity`]), an eager one's own.
    pub split_items: u64,
    pub new_batch_size: u32,
    pub padded_batch_size: usize,
    pub cta_tile_q: u32,
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
    let batch = req.live.requests as usize;
    let qo_lens = spans(op, "qo_indptr", req.qo_indptr, batch)?;
    let kv_pages = spans(op, "kv_indptr", req.kv_indptr, batch)?;
    let group = u64::from(req.num_qo_heads / req.num_kv_heads);

    let max_grid_size = u64::from(req.ctas_per_sm.max(1)) * u64::from(device.num_sm);
    let max_batch_size_if_split =
        u32::try_from(max_grid_size / u64::from(req.num_kv_heads)).unwrap_or(u32::MAX);

    let packed_qo_lens: Vec<u64> = qo_lens.iter().map(|&q| u64::from(q) * group).collect();
    let min_kv_chunk_size = u64::from((128 / req.page_size).max(1));

    let (cta_tile_q, total_num_tiles_q) = if req.enable_cuda_graph {
        graph_tiles(req.total_num_rows, req.batch_size, group, req.head_dim)
    } else {
        let sum: u64 = packed_qo_lens.iter().sum();
        let tile = determine_cta_tile_q(sum / u64::from(req.batch_size), req.head_dim);
        let tiles = packed_qo_lens
            .iter()
            .map(|&packed| packed.div_ceil(u64::from(tile)))
            .sum();
        (tile, tiles)
    };

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

    // A captured layout is split wherever a replay can split: each fire
    // splits its keys into the chunks whose waves over the device's
    // concurrent blocks (`Request::blocks_per_sm`) take least, as many as
    // the captured grid holds work items of its live tiles and the keys
    // hold chunks of the least width ([`fill_chunks`]). The chunks are the
    // fire's own data, restaged as the plan's tables are; a fire of one
    // chunk lands its rows whole.
    let graph_capacity = req.enable_cuda_graph.then(|| {
        let per_sm = (req.blocks_per_sm)(req.head_dim, cta_tile_q, device)
            .unwrap_or(req.ctas_per_sm)
            .max(1);
        let concurrent = u64::from(per_sm) * u64::from(device.num_sm.max(1));
        let heads = u64::from(req.num_kv_heads.max(1));
        let grid = total_num_tiles_q.max(concurrent.div_ceil(heads));
        (concurrent, grid, split_capacity(grid, heads, concurrent))
    });
    let (split_kv, kv_chunk_size_in_pages) = match graph_capacity {
        Some((concurrent, grid, capacity)) => {
            let max_kv = effective_kv_lens.iter().copied().max().unwrap_or(0).max(1);
            let live_tiles: u64 = packed_qo_lens
                .iter()
                .map(|&packed| packed.div_ceil(u64::from(cta_tile_q)))
                .sum::<u64>()
                .max(1);
            let cap = (grid / live_tiles).min(max_kv.div_ceil(min_kv_chunk_size));
            let chunks = if capacity > 0 {
                fill_chunks(live_tiles * u64::from(req.num_kv_heads), concurrent, cap)
            } else {
                1
            };
            let pages = if chunks > 1 {
                max_kv.div_ceil(chunks).max(min_kv_chunk_size)
            } else {
                max_kv
            };
            (capacity > 0, pages)
        }
        None => search_kv_chunk_size(
            max_batch_size_if_split,
            &packed_qo_lens,
            &effective_kv_lens,
            cta_tile_q,
            min_kv_chunk_size,
        ),
    };

    let mut request_indices = Vec::new();
    let mut qo_tile_indices = Vec::new();
    let mut kv_tile_indices = Vec::new();
    let mut merge_indptr = vec![0i64];
    let mut o_indptr = vec![0i64; req.live.lane_offset as usize + 1];
    let mut new_batch_size: u64 = 0;
    for (request_idx, (&packed, &kv)) in packed_qo_lens.iter().zip(&effective_kv_lens).enumerate() {
        let num_tiles_q = packed.div_ceil(u64::from(cta_tile_q));
        let num_chunks_kv = kv.max(1).div_ceil(kv_chunk_size_in_pages);
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
            merge_indptr.push(
                merge_indptr
                    .last()
                    .expect("merge_indptr starts with a zero")
                    + merge_step,
            );
        }
        o_indptr.push(
            o_indptr.last().expect("o_indptr starts with a zero") + qo_len as i64 * merge_step,
        );
    }
    let live_tiles: u64 = packed_qo_lens
        .iter()
        .map(|&packed| packed.div_ceil(u64::from(cta_tile_q)))
        .sum();
    let split_live = split_kv && (!req.enable_cuda_graph || new_batch_size > live_tiles);
    let merge_indptr = narrow_all(op, "batch_prefill_merge_indptr", &merge_indptr)?;
    let o_indptr: Vec<i64> = if split_live {
        o_indptr
    } else {
        o_indptr
            .into_iter()
            .map(|at| at + i64::from(req.live.row_offset))
            .collect()
    };
    let o_indptr = narrow_all(op, "batch_prefill_o_indptr", &o_indptr)?;

    // A captured grid is laid for the widest plan its replays can take. A
    // split layout's items reach the grid's split capacity; an unsplit one's
    // never pass the rows' tiles, so its grid is those tiles: padding it to
    // the split capacity re-pairs the CTAs on the SMs, and a replay ran
    // attention up to a fifth slower than the same plan walked.
    let padded_batch_size = match graph_capacity {
        Some((_, grid, _)) => grid,
        None => new_batch_size,
    };
    let split_items = match graph_capacity {
        Some((_, _, capacity)) => capacity,
        None => new_batch_size,
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
    let padded_batch_size = usize::try_from(padded_batch_size).map_err(|_| {
        refuse(
            op,
            "the padded batch does not fit this host's address space",
        )
    })?;

    Ok(Schedule {
        split_kv,
        split_live,
        split_items,
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
    info.request_indices_offset =
        Some(ints.alloc(4 * padded, 16, "batch_prefill_request_indices")?);
    info.qo_tile_indices_offset =
        Some(ints.alloc(4 * padded, 16, "batch_prefill_qo_tile_indices")?);
    info.kv_tile_indices_offset =
        Some(ints.alloc(4 * padded, 16, "batch_prefill_kv_tile_indices")?);
    info.o_indptr_offset = Some(ints.alloc(
        4 * (req.lane_offset as usize + req.batch_size as usize + 1),
        16,
        "batch_prefill_o_indptr",
    )?);
    info.kv_chunk_size_ptr_offset = Some(ints.alloc(4, 1, "batch_prefill_kv_chunk_size_ptr")?);
    info.live_lanes_offset = Some(ints.alloc(8, 16, "batch_prefill_live_lanes")?);
    if req.enable_cuda_graph {
        info.total_num_rows_offset = Some(ints.alloc(4, 16, "batch_prefill_total_num_rows")?);
    }

    let mut floats = AlignedAllocator::new(op, float_space);
    if sched.split_kv {
        let heads = u64::from(req.num_qo_heads);
        let rows = partial_rows(sched.cta_tile_q, req.num_qo_heads, req.num_kv_heads);
        let head_dim = u64::from(req.head_dim);
        let items = sched.split_items;
        info.v_offset = Some(floats.alloc(
            (heads * items * rows * head_dim * 4) as usize,
            16,
            "batch_prefill_tmp_v",
        )?);
        info.s_offset = Some(floats.alloc(
            (heads * items * rows * 4) as usize,
            16,
            "batch_prefill_tmp_s",
        )?);
        info.merge_indptr_offset = Some(ints.alloc(
            4 * (req.total_num_rows as usize + 1),
            16,
            "batch_prefill_merge_indptr",
        )?);
        if req.enable_cuda_graph {
            info.split_live_offset = Some(ints.alloc(4, 16, "batch_prefill_split_live")?);
        }
    }
    if sched.split_kv || req.enable_cuda_graph {
        info.block_valid_mask_offset =
            Some(ints.alloc(padded, 16, "batch_prefill_block_valid_mask")?);
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
    staging.put_i32s(
        at(info.live_lanes_offset),
        &[
            narrow(
                op,
                "batch_prefill_live_lanes",
                i64::from(req.live.lane_offset),
            )?,
            narrow(op, "batch_prefill_live_lanes", i64::from(req.live.requests))?,
        ],
        "batch_prefill_live_lanes",
    )?;
    if req.enable_cuda_graph {
        // The rows the fold folds: none where the fire lands them whole.
        let folded = if sched.split_live {
            req.qo_indptr[req.live.requests as usize]
        } else {
            0
        };
        staging.put_i32(
            at(info.total_num_rows_offset),
            folded,
            "batch_prefill_total_num_rows",
        )?;
    }
    if info.split_live_offset.is_some() {
        staging.put_i32(
            at(info.split_live_offset),
            i32::from(sched.split_live),
            "batch_prefill_split_live",
        )?;
    }
    if sched.split_kv {
        staging.put_i32s(
            at(info.merge_indptr_offset),
            &sched.merge_indptr,
            "batch_prefill_merge_indptr",
        )?;
    }
    if sched.split_kv || req.enable_cuda_graph {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The PRO 4500's 164 blocks (82 SMs, two each): Qwen3-0.6B's 2048-row
    /// fire (256 blocks) splits only past the 95 items its grid holds; a
    /// fire one tile wide fills the device; one that fills whole waves
    /// stays whole.
    #[test]
    fn a_fire_splits_where_its_waves_take_less() {
        assert_eq!(fill_chunks(256, 164, 2), 1, "two chunks tie one");
        assert_eq!(fill_chunks(256, 164, 3), 3);
        assert_eq!(fill_chunks(128, 164, 4), 1, "one wave either way");
        assert_eq!(fill_chunks(128, 164, 5), 5);
        assert_eq!(fill_chunks(8, 164, 20), 20);
        assert_eq!(fill_chunks(328, 164, 8), 1, "whole waves");
        assert_eq!(fill_chunks(8, 164, 1), 1, "no room for a second chunk");
    }

    #[test]
    fn the_partials_hold_every_row_the_split_writes() {
        for (q_heads, kv_heads, head_dim) in [(16, 2, 512), (16, 8, 256), (32, 8, 128), (8, 8, 64)]
        {
            for qo in [vec![1u32; 64], vec![1, 7, 300, 1, 33], vec![2048]] {
                for graph in [true, false] {
                    let qo_indptr: Vec<i32> = std::iter::once(0)
                        .chain(qo.iter().scan(0, |at, len| {
                            *at += *len as i32;
                            Some(*at)
                        }))
                        .collect();
                    let kv_indptr: Vec<i32> =
                        (0..=qo.len() as i32).map(|lane| lane * 384).collect();
                    let rows = *qo_indptr.last().unwrap() as u32;
                    let req = Request {
                        qo_indptr: &qo_indptr,
                        kv_indptr: &kv_indptr,
                        total_num_rows: rows,
                        batch_size: qo.len() as u32,
                        lane_offset: 0,
                        live: Live {
                            requests: qo.len() as u32,
                            lane_offset: 0,
                            row_offset: 0,
                            rows,
                        },
                        num_qo_heads: q_heads,
                        num_kv_heads: kv_heads,
                        head_dim,
                        page_size: 16,
                        enable_cuda_graph: graph,
                        window_left: None,
                        ctas_per_sm: 2,
                        blocks_per_sm: |_, _, _| Some(2),
                    };
                    let sched = schedule("test", &req, &Device::L40S).unwrap();
                    if !sched.split_live {
                        continue;
                    }
                    let laid = layout("test", &req, &sched, usize::MAX, usize::MAX).unwrap();
                    let (Some(v), Some(s)) = (laid.info.v_offset, laid.info.s_offset) else {
                        continue;
                    };
                    let written = u64::from(*sched.o_indptr.last().unwrap() as u32)
                        * u64::from(q_heads)
                        * u64::from(head_dim)
                        * 4;
                    assert!(written <= u64::from(s - v), "{q_heads}/{kv_heads} {qo:?}");
                }
            }
        }
    }

}
