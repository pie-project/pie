//! Latent-attention scheduler: splits packed query tiles over CTA clusters by a cost heap, carves kv spans at `kv_len_limit`, and books the partial-output merge the split spans need.

use crate::error::Error;

use crate::attn::plan::{Built, Device, Live, MlaPlanInfo};
use crate::attn::sched::{
    AlignedAllocator, at, CostHeap, Staging, cost_function, lengths, narrow, packed_causal_kv_end,
    spans,
};
use crate::jit::refuse;

/// The device text's bound on the schedule's work items.
pub const MAX_TOTAL_NUM_WORKS: usize = 16384;

#[derive(Clone, Copy, Debug)]
pub struct Request<'a> {
    /// Host copy of the query indptr — `[batch_size + 1]`.
    pub qo_indptr: &'a [i32],
    /// Host copy of the kv element offsets — `[batch_size + 1]`.
    pub kv_indptr: &'a [i32],
    /// Host per-request kv lengths, in tokens — `[batch_size]`.
    pub kv_len_arr: &'a [i32],
    /// Row and lane counts this schedule is carved for; the cluster split averages `rows * heads / lanes` from this pair.
    pub total_num_rows: u32,
    pub batch_size: u32,
    /// This fire's own host vectors; drives the kv-span carve and work-list staging.
    pub live: Live,
    pub num_heads: u32,
    pub head_dim_o: u32,
    pub causal: bool,
}

#[derive(Clone, Debug, Default)]
struct ClusterWork {
    q_indptr: Vec<i32>,
    kv_indptr: Vec<i32>,
    partial_indptr: Vec<i32>,
    q_len: Vec<i32>,
    kv_len: Vec<i32>,
    q_start: Vec<i32>,
    kv_start: Vec<i32>,
    kv_end: Vec<i32>,
}

/// The computed schedule: pure data, laid out and staged by [`plan`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schedule {
    pub cluster_size: u32,
    pub num_clusters: u32,
    pub cluster_tile_q: u32,
    pub kv_len_limit: u64,
    pub total_num_works: usize,
    pub partial_o_nnz: i64,
    pub q_indptr: Vec<i32>,
    pub kv_indptr: Vec<i32>,
    pub partial_indptr: Vec<i32>,
    pub q_len: Vec<i32>,
    pub kv_len: Vec<i32>,
    pub q_start: Vec<i32>,
    pub kv_start: Vec<i32>,
    pub kv_end: Vec<i32>,
    pub work_indptr: Vec<i32>,
    pub merge_packed_offset_start: Vec<i32>,
    pub merge_packed_offset_end: Vec<i32>,
    pub merge_partial_packed_offset_start: Vec<i32>,
    pub merge_partial_packed_offset_end: Vec<i32>,
    pub merge_partial_stride: Vec<i32>,
}

/// The kv split threshold: the per-cluster average kv walk, stepped up to
/// the tier the device text tiles at.
#[must_use]
pub fn kv_len_limit_step(x: u64) -> u64 {
    if x <= 8 {
        32
    } else if x <= 16 {
        64
    } else if x <= 32 {
        128
    } else if x <= 64 {
        192
    } else {
        x.div_ceil(256) * 256
    }
}

#[allow(clippy::too_many_lines)]
pub fn schedule(op: &'static str, req: &Request<'_>, device: &Device) -> Result<Schedule, Error> {
    let batch = req.live.requests as usize;
    if batch == 0 {
        return Err(refuse(op, "the batch is empty"));
    }
    if device.num_sm == 0 {
        return Err(refuse(op, "the stated device has no multiprocessors"));
    }
    let qo_lens = spans(op, "qo_indptr", req.qo_indptr, batch)?;
    spans(op, "kv_indptr", req.kv_indptr, batch)?;
    let kv_lens = lengths(op, "kv length table", req.kv_len_arr, batch)?;

    // packed extent of every request must fit the device's i32; merge offsets are staged in it.
    let packed_qo_lens: Vec<u32> = qo_lens
        .iter()
        .map(|&q| {
            narrow(
                op,
                "the packed query extent",
                i64::from(q) * i64::from(req.num_heads),
            )
            .map(|packed| packed as u32)
        })
        .collect::<Result<_, _>>()?;
    for (&qo_len, &kv_len) in qo_lens.iter().zip(&kv_lens) {
        narrow(op, "mla_q_len", i64::from(qo_len))?;
        narrow(op, "mla_kv_len", i64::from(kv_len))?;
    }

    // avg packed extent is exactly rows * heads / lanes, a function of the carve rather than of per-request lengths.
    let avg_packed_qo_len =
        u64::from(req.total_num_rows) * u64::from(req.num_heads) / u64::from(req.batch_size);

    // cluster_size/num_clusters (num_blks_x/num_blks_y) affect only performance, not correctness: kv extents and the merge are booked off the actual walk regardless.
    let cluster_size: u32 = if avg_packed_qo_len > 64 { 2 } else { 1 };
    let num_clusters = device.num_sm / cluster_size;
    if num_clusters == 0 {
        return Err(refuse(
            op,
            format!(
                "{} multiprocessors do not hold one {cluster_size}-CTA cluster",
                device.num_sm
            ),
        ));
    }
    const CTA_TILE_Q: u32 = 64;
    let cluster_tile_q = cluster_size * CTA_TILE_Q;

    let effective = |qo_len: u32, kv_len: u32, qo_tile_idx: u32, num_qo_tiles: u32| {
        if req.causal {
            packed_causal_kv_end(
                qo_len,
                kv_len,
                qo_tile_idx,
                cluster_tile_q,
                num_qo_tiles,
                req.num_heads,
            )
        } else {
            kv_len
        }
    };

    let mut total_kv_lens: u64 = 0;
    for i in 0..batch {
        let num_qo_tiles = packed_qo_lens[i].div_ceil(cluster_tile_q);
        for qo_tile_idx in 0..num_qo_tiles {
            total_kv_lens +=
                u64::from(effective(qo_lens[i], kv_lens[i], qo_tile_idx, num_qo_tiles));
        }
    }
    let kv_len_limit = kv_len_limit_step(total_kv_lens.div_ceil(u64::from(num_clusters)).max(1));

    let mut heap = CostHeap::new(num_clusters);
    let mut clusters = vec![ClusterWork::default(); num_clusters as usize];
    let num_sm = device.num_sm as usize;
    let mut merge_packed_offset_start = vec![0i32; num_sm];
    let mut merge_packed_offset_end = vec![0i32; num_sm];
    let mut merge_partial_packed_offset_start = vec![0i32; num_sm];
    let mut merge_partial_packed_offset_end = vec![0i32; num_sm];
    let mut merge_partial_stride = vec![0i32; num_sm];

    let mut merge_cta_counter: usize = 0;
    let mut partial_o_nnz: i64 = 0;

    for i in 0..batch {
        let (qo_len, kv_len, packed) = (qo_lens[i], kv_lens[i], packed_qo_lens[i]);
        let num_qo_tiles = packed.div_ceil(cluster_tile_q);
        for qo_tile_idx in (0..num_qo_tiles).rev() {
            let mut remaining = u64::from(effective(qo_len, kv_len, qo_tile_idx, num_qo_tiles));
            let mut kv_start: u64 = 0;
            let split_kv = remaining > kv_len_limit;
            let tile_start = u64::from(qo_tile_idx) * u64::from(cluster_tile_q);
            let row_tile_size = u64::from(cluster_tile_q).min(u64::from(packed) - tile_start);
            if split_kv {
                let num_qo_chunks = (remaining * u64::from(cluster_size) / kv_len_limit).max(1);
                let row_chunk_size = row_tile_size.div_ceil(num_qo_chunks);
                let partial_span = remaining.div_ceil(kv_len_limit) * row_tile_size;
                let mut offset_start: u64 = 0;
                while offset_start < row_tile_size {
                    if merge_cta_counter >= num_sm {
                        return Err(refuse(
                            op,
                            format!(
                                "the schedule asks for merge CTA {} on a device with {}",
                                merge_cta_counter + 1,
                                device.num_sm
                            ),
                        ));
                    }
                    let slot = merge_cta_counter;
                    let base = i64::from(req.qo_indptr[i]) * i64::from(req.num_heads)
                        + tile_start as i64;
                    merge_packed_offset_start[slot] =
                        narrow(op, "mla_merge_packed_offset_start", base + offset_start as i64)?;
                    merge_packed_offset_end[slot] = narrow(
                        op,
                        "mla_merge_packed_offset_end",
                        base + (offset_start + row_chunk_size).min(row_tile_size) as i64,
                    )?;
                    merge_partial_packed_offset_start[slot] = narrow(
                        op,
                        "mla_merge_partial_packed_offset_start",
                        partial_o_nnz + offset_start as i64,
                    )?;
                    merge_partial_packed_offset_end[slot] = narrow(
                        op,
                        "mla_merge_partial_packed_offset_end",
                        partial_o_nnz + partial_span as i64,
                    )?;
                    merge_partial_stride[slot] =
                        narrow(op, "mla_merge_partial_stride", row_tile_size as i64)?;
                    merge_cta_counter += 1;
                    offset_start += row_chunk_size;
                }
            }
            let zero_kv_len = remaining == 0;
            loop {
                let (cluster_idx, accum_cost) = heap.pop();
                let actual_len = remaining.min(kv_len_limit);
                heap.insert(
                    cluster_idx,
                    accum_cost + cost_function(cluster_tile_q, actual_len),
                );
                let cluster = &mut clusters[cluster_idx as usize];
                // Narrowed above: qo_len, kv_len, and every packed offset
                // fit the device's i32.
                cluster.q_len.push(qo_len as i32);
                cluster.kv_len.push(kv_len as i32);
                cluster.q_indptr.push(req.qo_indptr[i]);
                cluster.kv_indptr.push(req.kv_indptr[i]);
                if split_kv {
                    cluster
                        .partial_indptr
                        .push(narrow(op, "mla_partial_indptr", partial_o_nnz)?);
                    partial_o_nnz += row_tile_size as i64;
                } else {
                    cluster.partial_indptr.push(-1);
                }
                cluster.q_start.push(tile_start as i32);
                cluster.kv_start.push(narrow(op, "mla_kv_start", kv_start as i64)?);
                cluster
                    .kv_end
                    .push(narrow(op, "mla_kv_end", (kv_start + actual_len) as i64)?);
                remaining -= actual_len;
                kv_start += actual_len;
                if remaining == 0 || zero_kv_len {
                    break;
                }
            }
        }
    }

    let mut work_indptr = vec![0i32; num_clusters as usize + 1];
    for i in 0..num_clusters as usize {
        work_indptr[i + 1] = work_indptr[i] + clusters[i].q_indptr.len() as i32;
    }
    let total_num_works =
        *work_indptr.last().expect("work_indptr has num_clusters + 1 entries") as usize;
    if total_num_works > MAX_TOTAL_NUM_WORKS {
        return Err(refuse(
            op,
            format!(
                "the schedule produced {total_num_works} work items over the device text's \
                 bound of {MAX_TOTAL_NUM_WORKS}"
            ),
        ));
    }

    let flatten = |f: fn(&ClusterWork) -> &Vec<i32>| -> Vec<i32> {
        clusters.iter().flat_map(|c| f(c).iter().copied()).collect()
    };
    Ok(Schedule {
        cluster_size,
        num_clusters,
        cluster_tile_q,
        kv_len_limit,
        total_num_works,
        partial_o_nnz,
        q_indptr: flatten(|c| &c.q_indptr),
        kv_indptr: flatten(|c| &c.kv_indptr),
        partial_indptr: flatten(|c| &c.partial_indptr),
        q_len: flatten(|c| &c.q_len),
        kv_len: flatten(|c| &c.kv_len),
        q_start: flatten(|c| &c.q_start),
        kv_start: flatten(|c| &c.kv_start),
        kv_end: flatten(|c| &c.kv_end),
        work_indptr,
        merge_packed_offset_start,
        merge_packed_offset_end,
        merge_partial_packed_offset_start,
        merge_partial_packed_offset_end,
        merge_partial_stride,
    })
}

pub fn plan(
    op: &'static str,
    req: &Request<'_>,
    device: &Device,
    int_bytes: usize,
    float_bytes: usize,
) -> Result<Built<MlaPlanInfo>, Error> {
    let sched = schedule(op, req, device)?;
    let mut info = MlaPlanInfo {
        num_blks_x: i64::from(sched.cluster_size),
        num_blks_y: i64::from(sched.num_clusters),
        ..MlaPlanInfo::default()
    };

    let works = 4 * MAX_TOTAL_NUM_WORKS;
    let per_sm = 4 * device.num_sm as usize;
    let mut ints = AlignedAllocator::new(op, int_bytes);
    info.q_indptr_offset = Some(ints.alloc(works, 16, "mla_q_indptr")?);
    info.kv_indptr_offset =
        Some(ints.alloc(works, 16, "mla_kv_indptr")?);
    info.partial_indptr_offset =
        Some(ints.alloc(works, 16, "mla_partial_indptr")?);
    info.merge_packed_offset_start_offset = Some(ints.alloc(per_sm, 16, "mla_merge_packed_offset_start")?);
    info.merge_packed_offset_end_offset = Some(ints.alloc(per_sm, 16, "mla_merge_packed_offset_end")?);
    info.merge_partial_packed_offset_start_offset = Some(ints.alloc(per_sm, 16, "mla_merge_partial_packed_offset_start")?);
    info.merge_partial_packed_offset_end_offset = Some(ints.alloc(per_sm, 16, "mla_merge_partial_packed_offset_end")?);
    info.merge_partial_stride_offset = Some(ints.alloc(per_sm, 16, "mla_merge_partial_stride")?);
    info.q_len_offset = Some(ints.alloc(works, 16, "mla_q_len")?);
    info.kv_len_offset = Some(ints.alloc(works, 16, "mla_kv_len")?);
    info.q_start_offset = Some(ints.alloc(works, 16, "mla_q_start")?);
    info.kv_start_offset = Some(ints.alloc(works, 16, "mla_kv_start")?);
    info.kv_end_offset = Some(ints.alloc(works, 16, "mla_kv_end")?);
    info.work_indptr_offset =
        Some(ints.alloc(works, 16, "mla_work_indptr")?);

    let writes: [(Option<u32>, &Vec<i32>, &'static str); 14] = [
        (info.q_indptr_offset, &sched.q_indptr, "mla_q_indptr"),
        (info.kv_indptr_offset, &sched.kv_indptr, "mla_kv_indptr"),
        (
            info.partial_indptr_offset,
            &sched.partial_indptr,
            "mla_partial_indptr",
        ),
        (
            info.merge_packed_offset_start_offset,
            &sched.merge_packed_offset_start,
            "mla_merge_packed_offset_start",
        ),
        (
            info.merge_packed_offset_end_offset,
            &sched.merge_packed_offset_end,
            "mla_merge_packed_offset_end",
        ),
        (
            info.merge_partial_packed_offset_start_offset,
            &sched.merge_partial_packed_offset_start,
            "mla_merge_partial_packed_offset_start",
        ),
        (
            info.merge_partial_packed_offset_end_offset,
            &sched.merge_partial_packed_offset_end,
            "mla_merge_partial_packed_offset_end",
        ),
        (
            info.merge_partial_stride_offset,
            &sched.merge_partial_stride,
            "mla_merge_partial_stride",
        ),
        (info.q_len_offset, &sched.q_len, "mla_q_len"),
        (info.kv_len_offset, &sched.kv_len, "mla_kv_len"),
        (info.q_start_offset, &sched.q_start, "mla_q_start"),
        (info.kv_start_offset, &sched.kv_start, "mla_kv_start"),
        (info.kv_end_offset, &sched.kv_end, "mla_kv_end"),
        (
            info.work_indptr_offset,
            &sched.work_indptr,
            "mla_work_indptr",
        ),
    ];
    let int_used = ints.used();
    let mut staging = Staging::new(op, int_used);
    for (offset, values, what) in writes {
        staging.put_i32s(at(offset), values, what)?;
    }

    const SIZEOF_DTYPE_O: usize = 2;
    let mut floats = AlignedAllocator::new(op, float_bytes);
    let rows = 2 * sched.num_clusters as usize * sched.cluster_tile_q as usize;
    info.partial_o_offset = Some(floats.alloc(rows * SIZEOF_DTYPE_O * req.head_dim_o as usize, 16, "mla_partial_o")?);
    info.partial_lse_offset =
        Some(floats.alloc(rows * 4, 16, "mla_partial_lse")?);

    Ok(Built {
        info,
        int_upload: staging.into_upload(int_used),
        int_bytes: int_used,
        float_bytes: floats.used(),
    })
}
