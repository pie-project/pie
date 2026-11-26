use crate::error::Error;

use crate::attn::plan::{Built, Device, Live, PrefillPlanInfo};
use crate::attn::sched::{AlignedAllocator, Staging, at, narrow, narrow_all, spans};
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
    /// The prefill blocks one compute unit holds at once: HIP's occupancy
    /// calculator over the kernel this schedule lays
    /// (`attn::paged::prefill_resident_blocks`).
    pub resident_blocks: u32,
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

/// The query tile every prefill work item covers: the one the kernel walks
/// (`attn::paged::PREFILL_TILE_Q`), whatever the batch's shape.
#[must_use]
const fn determine_cta_tile_q(_avg_packed_qo_len: u64, _head_dim: u32) -> u32 {
    crate::attn::paged::PREFILL_TILE_Q
}

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

/// The keys a windowed tile of `tile` packed rows reads from its first:
/// the window, then its rows' tokens (⌈tile / group⌉, and one more where a
/// tile starts mid-token), as the prefill kernels' `window_reach`.
const fn window_reach(window_left: u32, tile: u32, group: u64) -> u64 {
    window_left as u64 + (tile as u64).div_ceil(group) + 1
}

/// Windowed walks no longer than this many least chunks run whole, unsplit.
const WINDOW_WHOLE: u64 = 2;

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
    pub split_kv: bool,
    /// A split layout's replay that runs whole: it writes the output and
    /// the fold leaves (`block_direct_offset`).
    pub direct: bool,
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

/// The chunks `blocks` work blocks are best cut into on a device running
/// `slots` blocks at once, at most `cap`: the cut whose waves take least
/// time, each wave a chunk's walk (`ceil(blocks * c / slots) / c` whole
/// walks), the fewer chunks on a tie. One where no cut beats the whole walk.
fn split_chunks(blocks: u64, slots: u64, cap: u64) -> u64 {
    let slots = slots.max(1);
    let waves = |c: u64| (blocks * c).div_ceil(slots);
    let mut best = 1;
    for c in 2..=cap {
        if waves(c) * best < waves(best) * c {
            best = c;
        }
    }
    best
}

/// Whether a captured grid of `grid` tiles a kv head, over `heads` kv heads,
/// has a replay (of `fewest` tiles or more) that `split_chunks` cuts.
fn splits_somewhere(fewest: u64, grid: u64, heads: u64, slots: u64) -> bool {
    (fewest.max(1)..=grid / 2).any(|tiles| split_chunks(tiles * heads, slots, grid / tiles) > 1)
}

#[allow(clippy::too_many_lines)]
pub fn schedule(op: &'static str, req: &Request<'_>, device: &Device) -> Result<Schedule, Error> {
    req.check(op)?;
    let batch = req.live.requests as usize;
    let qo_lens = spans(op, "qo_indptr", req.qo_indptr, batch)?;
    let kv_pages = spans(op, "kv_indptr", req.kv_indptr, batch)?;
    let group = u64::from(req.num_qo_heads / req.num_kv_heads);

    // The blocks the device runs at once: every compute unit holding as many
    // prefill blocks as the occupancy calculator seats.
    let max_grid_size = u64::from(req.resident_blocks.max(1)) * u64::from(device.num_sm);
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

    // Under a window a tile reads the window behind its first query row
    // and its own rows' keys: `window_reach` keys from its first (the
    // kernels count a request's chunks over the same reach).
    let effective_kv_lens: Vec<u64> = kv_pages
        .iter()
        .map(|&pages| {
            let full = u64::from(pages);
            match req.window_left {
                Some(window_left) => window_reach(window_left, cta_tile_q, group)
                    .div_ceil(u64::from(req.page_size))
                    .min(full),
                None => full,
            }
        })
        .collect();

    // A short window's walk is a few kv tiles: splitting it would cost a
    // fold of the partials for a chunk or two each. The window alone decides
    // (no length does), so a captured plan keeps its layout across replays.
    let whole = req.window_left.is_some_and(|window_left| {
        window_reach(window_left, cta_tile_q, group).div_ceil(u64::from(req.page_size))
            <= WINDOW_WHOLE * min_kv_chunk_size
    });
    let max_kv = effective_kv_lens.iter().copied().max().unwrap_or(1).max(1);
    let (split_kv, direct, kv_chunk_size_in_pages) = if whole {
        (false, false, max_kv)
    } else if req.enable_cuda_graph {
        // A captured layout holds its rung's every replay, so it is split
        // (partials and their fold laid) wherever some replay's tiles would
        // run in fewer waves cut into chunks (`split_chunks`) within the
        // captured grid; each replay then takes its own chunks, and one that
        // is best whole is staged direct: it writes the output as an unsplit
        // walk does, and the fold leaves at entry. The rows alone decide the
        // layout, never a replay's keys or requests.
        let grid = total_num_tiles_q;
        let heads = u64::from(req.num_kv_heads);
        let fewest = (u64::from(req.total_num_rows) * group)
            .div_ceil(u64::from(cta_tile_q))
            .max(1);
        if !splits_somewhere(fewest, grid, heads, max_grid_size) {
            (false, false, max_kv)
        } else {
            let live_tiles: u64 = packed_qo_lens
                .iter()
                .map(|&packed| packed.div_ceil(u64::from(cta_tile_q)))
                .sum::<u64>()
                .max(1);
            let cap = (grid / live_tiles)
                .min(max_kv.div_ceil(min_kv_chunk_size))
                .max(1);
            match split_chunks(live_tiles * heads, max_grid_size, cap) {
                1 => (true, true, max_kv),
                chunks => (true, false, max_kv.div_ceil(chunks).max(min_kv_chunk_size)),
            }
        }
    } else if total_num_tiles_q >= u64::from(max_batch_size_if_split) {
        // Eager query tiles that already fill the grid read each request's
        // kv whole: a split would only add partials and their fold.
        (false, false, max_kv)
    } else {
        let (split, chunk) = search_kv_chunk_size(
            false,
            max_batch_size_if_split,
            &packed_qo_lens,
            &effective_kv_lens,
            cta_tile_q,
            min_kv_chunk_size,
        );
        (split, false, chunk)
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
    let merge_indptr = narrow_all(op, "batch_prefill_merge_indptr", &merge_indptr)?;
    let o_indptr: Vec<i64> = if split_kv && !direct {
        o_indptr
    } else {
        o_indptr
            .into_iter()
            .map(|at| at + i64::from(req.live.row_offset))
            .collect()
    };
    let o_indptr = narrow_all(op, "batch_prefill_o_indptr", &o_indptr)?;

    // A captured grid is its rung's tiles, padded for a replay whose every
    // lane is its own request; a split replay cuts its chunks within it
    // (`split_chunks`'s cap), so no layout pads past them.
    let padded_batch_size = if req.enable_cuda_graph {
        total_num_tiles_q
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
    let padded_batch_size = usize::try_from(padded_batch_size).map_err(|_| {
        refuse(
            op,
            "the padded batch does not fit this host's address space",
        )
    })?;

    Ok(Schedule {
        split_kv,
        direct,
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
    if req.enable_cuda_graph {
        info.total_num_rows_offset = Some(ints.alloc(4, 16, "batch_prefill_total_num_rows")?);
        info.block_live_offset = Some(ints.alloc(4, 16, "batch_prefill_block_live")?);
        if sched.split_kv {
            info.block_direct_offset = Some(ints.alloc(4, 16, "batch_prefill_block_direct")?);
        }
    }

    let mut floats = AlignedAllocator::new(op, float_space);
    if sched.split_kv {
        let heads = u64::from(req.num_qo_heads);
        let rows = partial_rows(sched.cta_tile_q, req.num_qo_heads, req.num_kv_heads);
        let head_dim = u64::from(req.head_dim);
        info.v_offset = Some(floats.alloc(
            (heads * padded as u64 * rows * head_dim * 4) as usize,
            16,
            "batch_prefill_tmp_v",
        )?);
        info.s_offset = Some(floats.alloc(
            (heads * padded as u64 * rows * 4) as usize,
            16,
            "batch_prefill_tmp_s",
        )?);
        info.merge_indptr_offset = Some(ints.alloc(
            4 * (req.total_num_rows as usize + 1),
            16,
            "batch_prefill_merge_indptr",
        )?);
    }
    // A captured grid is padded past the work: its extra blocks leave on the
    // live count (`block_live_offset`) or the mask.
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
    if req.enable_cuda_graph {
        staging.put_i32(
            at(info.total_num_rows_offset),
            req.qo_indptr[req.live.requests as usize],
            "batch_prefill_total_num_rows",
        )?;
        staging.put_i32(
            at(info.block_live_offset),
            narrow(
                op,
                "batch_prefill_block_live",
                i64::from(sched.new_batch_size),
            )?,
            "batch_prefill_block_live",
        )?;
        if sched.split_kv {
            staging.put_i32(
                at(info.block_direct_offset),
                i32::from(sched.direct),
                "batch_prefill_block_direct",
            )?;
        }
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
                        resident_blocks: 2,
                    };
                    let sched = schedule("test", &req, &Device::L40S).unwrap();
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

    /// The MI300X: 304 compute units of two prefill blocks each.
    const MI300X: Device = Device {
        num_sm: 304,
        cc_major: 9,
        max_smem_per_sm: 65_536,
        max_smem_per_block_optin: 65_536,
    };

    /// A captured fire of one request of `rows` rows over `keys` keys under
    /// a 256-lane carve, or of `rows / 8` requests of eight rows.
    fn captured(rows: u32, keys: u32, q_heads: u32, kv_heads: u32, one: bool) -> Schedule {
        let n = if one { 1 } else { rows / 8 };
        let qo_indptr: Vec<i32> = (0..=n).map(|i| (i * rows / n) as i32).collect();
        let pages = keys.div_ceil(16) as i32;
        let kv_indptr: Vec<i32> = (0..=n as i32).map(|i| i * pages).collect();
        let req = Request {
            qo_indptr: &qo_indptr,
            kv_indptr: &kv_indptr,
            total_num_rows: rows,
            batch_size: 256,
            lane_offset: 0,
            live: Live {
                requests: n,
                lane_offset: 0,
                row_offset: 0,
                rows,
            },
            num_qo_heads: q_heads,
            num_kv_heads: kv_heads,
            head_dim: 128,
            page_size: 16,
            enable_cuda_graph: true,
            window_left: None,
            resident_blocks: 2,
        };
        schedule("test", &req, &MI300X).unwrap()
    }

    fn chunks(s: &Schedule) -> i32 {
        s.kv_tile_indices.iter().max().map_or(0, |&k| k + 1)
    }

    /// The cut in waves: the fewest whole walks a wave each, ties to fewer.
    #[test]
    fn a_split_cuts_where_its_waves_take_less() {
        // 0.6B's one 2048-row request (32 tiles x 8 kv heads) on 608 slots:
        // seven chunks run in three waves of a seventh of the walk.
        assert_eq!(split_chunks(256, 608, 8), 7);
        // 30B-A3B's (128 x 4) and 0.6B's 4096 rows (64 x 8) fill the device:
        // a cut only takes as many more waves as chunks.
        assert_eq!(split_chunks(512, 608, 2), 1);
        assert_eq!(split_chunks(512, 608, 4), 1);
        // Past one wave by a little, two chunks take three waves of a half.
        assert_eq!(split_chunks(612, 608, 2), 2);
        assert_eq!(split_chunks(10, 608, 1), 1);
    }

    /// The four served shapes of a 32k prompt: only 0.6B's 2048-row fire
    /// is cut, within its captured grid; a fire of the same layout whose
    /// tiles fill the device runs whole, staged direct.
    #[test]
    fn a_captured_fire_splits_only_to_fill_the_device() {
        let small = captured(2048, 32768, 16, 8, true);
        assert!(small.split_kv && !small.direct);
        assert_eq!(chunks(&small), 7);
        assert!(small.new_batch_size as usize <= small.padded_batch_size);
        assert_eq!(small.padded_batch_size, 287, "the rung's tiles, not more");
        let full = captured(2048, 32768, 16, 8, false);
        assert!(full.split_kv && full.direct, "one layout, a whole replay");
        assert_eq!(chunks(&full), 1);
        assert_eq!(full.padded_batch_size, small.padded_batch_size);
        // A short prefix cannot be cut: its keys bound the chunks.
        let short = captured(2048, 2048 + 64, 16, 8, true);
        assert!(short.split_kv);
        for (rows, q_heads, kv_heads) in [(4096, 16, 8), (2048, 32, 4), (4096, 32, 4)] {
            let one = captured(rows, 32768, q_heads, kv_heads, true);
            assert_eq!(chunks(&one), 1, "{rows} rows {q_heads}/{kv_heads}");
            assert!(one.direct || !one.split_kv);
        }
        // 30B-A3B's 4096-row rung: no replay is ever cut, so no partials.
        assert!(!captured(4096, 32768, 32, 4, true).split_kv);
    }
}
