use super::alloc::{AlignedAllocator, Staging};
use super::arith::{ceil_div_i32, ceil_div_i64, cost_function, packed_causal_kv_end};
use super::heap::MinHeap;
use super::info::MlaPlanInfo;
use super::{Device, Error, Plan, Workspace};

/// The cap on work items MLA's index arrays are sized for.
pub const MAX_TOTAL_NUM_WORKS: i32 = 16384;

/// The batch an MLA plan is built for.
#[derive(Clone, Copy, Debug)]
pub struct Request<'a> {
    /// `batch_size + 1` QO row offsets.
    pub qo_indptr: &'a [i32],
    /// `batch_size + 1` KV offsets.
    pub kv_indptr: &'a [i32],
    /// `batch_size` KV lengths.
    pub kv_len_arr: &'a [i32],
    /// Requests in the batch.
    pub batch_size: u32,
    /// Heads, which pack into the QO tile — MLA fuses them, unlike FA3.
    pub num_heads: u32,
    /// Output head dimension, which sizes the partial-output carve.
    pub head_dim_o: u32,
    /// Whether a QO tile reads only up to the diagonal.
    pub causal: bool,
}

/// One cluster's work list, before it is flattened.
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

/// The MLA schedule: per-work arrays, per-merge-CTA arrays, and the grid shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schedule {
    /// CTAs per cluster — `num_blks_x`, and the grid's x extent.
    pub cluster_size: i32,
    /// Clusters — `num_blks_y`.
    pub num_clusters: u32,
    /// QO rows per cluster tile: `cluster_size * 64`.
    pub cluster_tile_q: i32,
    /// The KV chunk limit `f(ceil(total_kv / num_clusters))`.
    pub kv_len_limit: i32,
    /// Work items produced.
    pub total_num_works: i32,
    /// Rows of partial output the split tiles will write.
    pub partial_o_nnz: i32,
    /// `q_indptr[work]`.
    pub q_indptr: Vec<i32>,
    /// `kv_indptr[work]`.
    pub kv_indptr: Vec<i32>,
    /// `partial_indptr[work]` — the partial-output row this work item writes
    pub partial_indptr: Vec<i32>,
    /// `q_len[work]`.
    pub q_len: Vec<i32>,
    /// `kv_len[work]`.
    pub kv_len: Vec<i32>,
    /// `q_start[work]`.
    pub q_start: Vec<i32>,
    /// `kv_start[work]`.
    pub kv_start: Vec<i32>,
    /// `kv_end[work]`.
    pub kv_end: Vec<i32>,
    /// `work_indptr[cluster]`, `num_clusters + 1` entries.
    pub work_indptr: Vec<i32>,
    /// `merge_packed_offset_start[cta]`, `num_sm` entries.
    pub merge_packed_offset_start: Vec<i32>,
    /// `merge_packed_offset_end[cta]`.
    pub merge_packed_offset_end: Vec<i32>,
    /// `merge_partial_packed_offset_start[cta]`.
    pub merge_partial_packed_offset_start: Vec<i32>,
    /// `merge_partial_packed_offset_end[cta]`.
    pub merge_partial_packed_offset_end: Vec<i32>,
    /// `merge_partial_stride[cta]`.
    pub merge_partial_stride: Vec<i32>,
}

/// The KV chunk limit's step function.
#[must_use]
pub const fn kv_len_limit_step(x: i32) -> i32 {
    if x <= 8 {
        32
    } else if x <= 16 {
        64
    } else if x <= 32 {
        128
    } else if x <= 64 {
        192
    } else {
        ceil_div_i32(x, 256).wrapping_mul(256)
    }
}

/// Build the MLA schedule without laying it out in a workspace.
#[allow(clippy::too_many_lines)]
pub fn schedule(req: &Request<'_>, device: &Device) -> Result<Schedule, Error> {
    let batch_size = req.batch_size as usize;
    if batch_size == 0 {
        return Err(Error::EmptyBatch);
    }
    if device.num_sm == 0 {
        return Err(Error::MergeCtasExceedSm { counter: 0, num_sm: 0 });
    }
    for (array, len, needed) in [
        ("qo_indptr", req.qo_indptr.len(), batch_size + 1),
        ("kv_indptr", req.kv_indptr.len(), batch_size + 1),
        ("kv_len_arr", req.kv_len_arr.len(), batch_size),
    ] {
        if len < needed {
            return Err(Error::IndptrTooShort { array, needed, got: len });
        }
    }

    let mut accum_packed_qo_len: i32 = 0;
    let mut idx_qo_kv_len: Vec<(i32, i32, i32)> = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let qo_len = req.qo_indptr[i + 1].wrapping_sub(req.qo_indptr[i]);
        if qo_len < 0 {
            return Err(Error::NegativeSpan {
                array: "qo_indptr",
                index: i,
                begin: i64::from(req.qo_indptr[i]),
                end: i64::from(req.qo_indptr[i + 1]),
            });
        }
        let packed_qo_len = (qo_len as u32).wrapping_mul(req.num_heads) as i32;
        accum_packed_qo_len = accum_packed_qo_len.wrapping_add(packed_qo_len);
        idx_qo_kv_len.push((i as i32, qo_len, req.kv_len_arr[i]));
    }
    let avg_packed_qo_len = ((accum_packed_qo_len as u32) / req.batch_size) as i32;

    let cluster_size: i32 = if avg_packed_qo_len > 64 { 2 } else { 1 };
    let num_clusters = device.num_sm / cluster_size as u32;
    if num_clusters == 0 {
        return Err(Error::MergeCtasExceedSm { counter: 0, num_sm: i64::from(device.num_sm) });
    }
    const CTA_TILE_Q: i32 = 64;
    let cluster_tile_q = cluster_size * CTA_TILE_Q;

    let mut total_kv_lens: i64 = 0;
    for &(_, qo_len, kv_len) in &idx_qo_kv_len {
        let packed_qo_len = (qo_len as u32).wrapping_mul(req.num_heads) as i32;
        let num_qo_tiles = ceil_div_i32(packed_qo_len, cluster_tile_q);
        for qo_tile_idx in (0..num_qo_tiles).rev() {
            let effective_kv_len = if req.causal {
                packed_causal_kv_end(
                    qo_len,
                    kv_len,
                    qo_tile_idx,
                    cluster_tile_q,
                    num_qo_tiles,
                    req.num_heads as i32,
                )
            } else {
                kv_len
            };
            total_kv_lens = total_kv_lens.wrapping_add(i64::from(effective_kv_len));
        }
    }
    let kv_len_limit =
        kv_len_limit_step(ceil_div_i64(total_kv_lens, i64::from(num_clusters)).max(1) as i32);

    let mut heap = MinHeap::new(num_clusters);
    let mut clusters = vec![ClusterWork::default(); num_clusters as usize];
    let num_sm = device.num_sm as usize;
    let mut merge_packed_offset_start = vec![0i32; num_sm];
    let mut merge_packed_offset_end = vec![0i32; num_sm];
    let mut merge_partial_packed_offset_start = vec![0i32; num_sm];
    let mut merge_partial_packed_offset_end = vec![0i32; num_sm];
    let mut merge_partial_stride = vec![0i32; num_sm];

    let mut merge_cta_counter: i32 = 0;
    let mut partial_o_nnz: i32 = 0;

    for &(i, qo_len, kv_len) in &idx_qo_kv_len {
        let packed_qo_len = (qo_len as u32).wrapping_mul(req.num_heads) as i32;
        let num_qo_tiles = ceil_div_i32(packed_qo_len, cluster_tile_q);
        for qo_tile_idx in (0..num_qo_tiles).rev() {
            let mut remaining_len = if req.causal {
                packed_causal_kv_end(
                    qo_len,
                    kv_len,
                    qo_tile_idx,
                    cluster_tile_q,
                    num_qo_tiles,
                    req.num_heads as i32,
                )
            } else {
                kv_len
            };
            let mut kv_start: i32 = 0;
            let split_kv = remaining_len > kv_len_limit;
            let row_tile_size =
                cluster_tile_q.min(packed_qo_len.wrapping_sub(qo_tile_idx * cluster_tile_q));
            if split_kv {
                let num_qo_chunks =
                    (remaining_len.wrapping_mul(cluster_size) / kv_len_limit).max(1);
                let row_chunk_size = ceil_div_i32(row_tile_size, num_qo_chunks);
                let current_q_tile_end =
                    cluster_tile_q.min(packed_qo_len.wrapping_sub(qo_tile_idx * cluster_tile_q));
                let mut offset_start: i32 = 0;
                while offset_start < row_tile_size {
                    let slot = merge_cta_counter as usize;
                    if slot >= num_sm {
                        return Err(Error::MergeCtasExceedSm {
                            counter: i64::from(merge_cta_counter) + 1,
                            num_sm: i64::from(device.num_sm),
                        });
                    }
                    let base = (req.qo_indptr[i as usize] as u32)
                        .wrapping_mul(req.num_heads)
                        .wrapping_add((qo_tile_idx * cluster_tile_q) as u32);
                    merge_packed_offset_start[slot] = base.wrapping_add(offset_start as u32) as i32;
                    merge_packed_offset_end[slot] = base.wrapping_add(
                        offset_start.wrapping_add(row_chunk_size).min(current_q_tile_end) as u32,
                    ) as i32;
                    merge_partial_packed_offset_start[slot] =
                        partial_o_nnz.wrapping_add(offset_start);
                    merge_partial_packed_offset_end[slot] = partial_o_nnz.wrapping_add(
                        ceil_div_i32(remaining_len, kv_len_limit).wrapping_mul(row_tile_size),
                    );
                    merge_partial_stride[slot] = row_tile_size;
                    merge_cta_counter += 1;
                    offset_start = offset_start.wrapping_add(row_chunk_size);
                }
            }
            let zero_kv_len = remaining_len == 0;
            while remaining_len > 0 || zero_kv_len {
                let (cluster_idx, accum_cost) = heap.pop();
                let actual_len = remaining_len.min(kv_len_limit);
                heap.insert((cluster_idx, accum_cost + cost_function(cluster_tile_q, actual_len)));
                let cluster = &mut clusters[cluster_idx as usize];
                cluster.q_len.push(qo_len);
                cluster.kv_len.push(kv_len);
                cluster.q_indptr.push(req.qo_indptr[i as usize]);
                cluster.kv_indptr.push(req.kv_indptr[i as usize]);
                if split_kv {
                    cluster.partial_indptr.push(partial_o_nnz);
                    partial_o_nnz = partial_o_nnz.wrapping_add(row_tile_size);
                } else {
                    cluster.partial_indptr.push(-1);
                }
                cluster.q_start.push(qo_tile_idx * cluster_tile_q);
                cluster.kv_start.push(kv_start);
                cluster.kv_end.push(kv_start.wrapping_add(actual_len));
                remaining_len -= actual_len;
                kv_start = kv_start.wrapping_add(actual_len);
                if zero_kv_len {
                    break;
                }
            }
        }
    }

    if i64::from(merge_cta_counter) > i64::from(device.num_sm) {
        return Err(Error::MergeCtasExceedSm {
            counter: i64::from(merge_cta_counter),
            num_sm: i64::from(device.num_sm),
        });
    }

    let mut work_indptr = vec![0i32; num_clusters as usize + 1];
    for i in 0..num_clusters as usize {
        work_indptr[i + 1] = work_indptr[i].wrapping_add(clusters[i].q_indptr.len() as i32);
    }
    let total_num_works = *work_indptr.last().expect("work_indptr has num_clusters + 1 entries");
    if total_num_works > MAX_TOTAL_NUM_WORKS {
        return Err(Error::TooManyWorks {
            total: i64::from(total_num_works),
            max: i64::from(MAX_TOTAL_NUM_WORKS),
        });
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

/// `MLAPlan` — the plan, and the bytes to upload under it.
pub fn plan(
    req: &Request<'_>,
    device: &Device,
    workspace: Workspace,
) -> Result<Plan<MlaPlanInfo>, Error> {
    let sched = schedule(req, device)?;
    let mut staging = Staging::new(workspace.int_bytes);
    let mut info = MlaPlanInfo {
        num_blks_x: i64::from(sched.cluster_size),
        num_blks_y: i64::from(sched.num_clusters),
        ..MlaPlanInfo::default()
    };

    let works = 4 * MAX_TOTAL_NUM_WORKS as usize;
    let per_sm = 4 * device.num_sm as usize;
    let mut int_alloc = AlignedAllocator::new(workspace.int_bytes);
    info.q_indptr_offset = int_alloc.alloc(works, 16, "mla_q_indptr")? as i64;
    info.kv_indptr_offset = int_alloc.alloc(works, 16, "mla_kv_indptr")? as i64;
    info.partial_indptr_offset = int_alloc.alloc(works, 16, "mla_partial_indptr")? as i64;
    info.merge_packed_offset_start_offset =
        int_alloc.alloc(per_sm, 16, "mla_merge_packed_offset_start")? as i64;
    info.merge_packed_offset_end_offset =
        int_alloc.alloc(per_sm, 16, "mla_merge_packed_offset_end")? as i64;
    info.merge_partial_packed_offset_start_offset =
        int_alloc.alloc(per_sm, 16, "mla_merge_partial_packed_offset_start")? as i64;
    info.merge_partial_packed_offset_end_offset =
        int_alloc.alloc(per_sm, 16, "mla_merge_partial_packed_offset_end")? as i64;
    info.merge_partial_stride_offset =
        int_alloc.alloc(per_sm, 16, "mla_merge_partial_stride")? as i64;
    info.q_len_offset = int_alloc.alloc(works, 16, "mla_q_len")? as i64;
    info.kv_len_offset = int_alloc.alloc(works, 16, "mla_kv_len")? as i64;
    info.q_start_offset = int_alloc.alloc(works, 16, "mla_q_start")? as i64;
    info.kv_start_offset = int_alloc.alloc(works, 16, "mla_kv_start")? as i64;
    info.kv_end_offset = int_alloc.alloc(works, 16, "mla_kv_end")? as i64;
    info.work_indptr_offset = int_alloc.alloc(works, 16, "mla_work_indptr")? as i64;

    let writes: [(i64, &Vec<i32>, &str); 14] = [
        (info.q_indptr_offset, &sched.q_indptr, "mla_q_indptr"),
        (info.kv_indptr_offset, &sched.kv_indptr, "mla_kv_indptr"),
        (info.partial_indptr_offset, &sched.partial_indptr, "mla_partial_indptr"),
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
        (info.merge_partial_stride_offset, &sched.merge_partial_stride, "mla_merge_partial_stride"),
        (info.q_len_offset, &sched.q_len, "mla_q_len"),
        (info.kv_len_offset, &sched.kv_len, "mla_kv_len"),
        (info.q_start_offset, &sched.q_start, "mla_q_start"),
        (info.kv_start_offset, &sched.kv_start, "mla_kv_start"),
        (info.kv_end_offset, &sched.kv_end, "mla_kv_end"),
        (info.work_indptr_offset, &sched.work_indptr, "mla_work_indptr"),
    ];
    for (offset, values, what) in writes {
        staging.put_i32s(offset as usize, values, what)?;
    }

    const SIZEOF_DTYPE_O: usize = 2;
    let mut float_alloc = AlignedAllocator::new(workspace.float_bytes);
    let rows = 2 * sched.num_clusters as usize * sched.cluster_tile_q as usize;
    info.partial_o_offset =
        float_alloc.alloc(rows * SIZEOF_DTYPE_O * req.head_dim_o as usize, 16, "mla_partial_o")?
            as i64;
    info.partial_lse_offset = float_alloc.alloc(rows * 4, 16, "mla_partial_lse")? as i64;

    let int_bytes = int_alloc.used();
    Ok(Plan {
        info,
        int_upload: staging.into_upload(int_bytes),
        int_bytes,
        float_bytes: float_alloc.used(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const H100: Device = Device::new(132, 9);

    fn request<'a>(qo: &'a [i32], kv: &'a [i32], lens: &'a [i32]) -> Request<'a> {
        Request {
            qo_indptr: qo,
            kv_indptr: kv,
            kv_len_arr: lens,
            batch_size: lens.len() as u32,
            num_heads: 128,
            head_dim_o: 512,
            causal: true,
        }
    }

    /// A batch of single-token decodes gets one-CTA clusters and 64-row tiles;
    #[test]
    fn a_decode_batch_gets_one_cta_clusters() {
        let qo: Vec<i32> = (0..=8).collect();
        let kv: Vec<i32> = (0..=8).map(|i| i * 1024).collect();
        let lens = vec![1024i32; 8];
        let sched = schedule(&request(&qo, &kv, &lens), &H100).unwrap();
        assert_eq!(sched.cluster_size, 2, "128 heads x 1 token packs to 128 rows");
        assert_eq!(sched.cluster_tile_q, 128);
        assert_eq!(*sched.work_indptr.last().unwrap(), sched.total_num_works);
    }

    /// The work arrays are a partition of the schedule, whatever the split did.
    #[test]
    fn work_indptr_partitions_the_work_items() {
        let qo = [0i32, 1, 2, 3, 4];
        let kv = [0i32, 8192, 16384, 24576, 32768];
        let lens = [8192i32, 8192, 8192, 8192];
        let sched = schedule(&request(&qo, &kv, &lens), &H100).unwrap();
        assert_eq!(sched.q_indptr.len(), sched.total_num_works as usize);
        assert_eq!(sched.kv_end.len(), sched.total_num_works as usize);
        assert!(sched.work_indptr.windows(2).all(|w| w[1] >= w[0]));
    }

    /// One very long sequence beside short ones is the case the KV split
    #[test]
    fn a_long_sequence_is_chopped_into_contiguous_kv_ranges() {
        let qo = [0i32, 1, 2];
        let kv = [0i32, 131_072, 131_073];
        let lens = [131_072i32, 1];
        let sched = schedule(&request(&qo, &kv, &lens), &H100).unwrap();
        assert!(sched.partial_indptr.iter().any(|&p| p >= 0), "nothing was split");
        let long: Vec<(i32, i32)> = sched
            .kv_start
            .iter()
            .zip(&sched.kv_end)
            .zip(&sched.kv_len)
            .filter(|&((_, _), &len)| len == 131_072)
            .map(|((&s, &e), _)| (s, e))
            .collect();
        let mut sorted = long.clone();
        sorted.sort_unstable();
        assert_eq!(sorted[0].0, 0);
        assert!(sorted.windows(2).all(|w| w[0].1 == w[1].0), "kv ranges are not contiguous");
        assert_eq!(sorted.last().unwrap().1, 131_072);
    }

    /// A zero-length KV still gets a work item, because the kernel must write
    #[test]
    fn a_zero_length_request_still_gets_one_work_item() {
        let qo = [0i32, 1];
        let kv = [0i32, 0];
        let lens = [0i32];
        let sched = schedule(&request(&qo, &kv, &lens), &H100).unwrap();
        assert_eq!(sched.total_num_works, 1);
        assert_eq!(sched.kv_start, vec![0]);
        assert_eq!(sched.kv_end, vec![0]);
    }

    /// The chunk limit snaps, which is why the kernel's KV loop has no tail.
    #[test]
    fn the_chunk_limit_snaps_to_a_tileable_size() {
        assert_eq!(kv_len_limit_step(1), 32);
        assert_eq!(kv_len_limit_step(8), 32);
        assert_eq!(kv_len_limit_step(9), 64);
        assert_eq!(kv_len_limit_step(33), 192);
        assert_eq!(kv_len_limit_step(65), 256);
        assert_eq!(kv_len_limit_step(257), 512);
    }

    /// The empty batch is upstream's division by zero.
    #[test]
    fn the_empty_batch_is_refused_rather_than_dividing_by_zero() {
        let qo = [0i32];
        let kv = [0i32];
        let lens: [i32; 0] = [];
        assert_eq!(schedule(&request(&qo, &kv, &lens), &H100).unwrap_err(), Error::EmptyBatch);
    }

    /// The int workspace does not depend on the batch — it is `max_total_num_works`
    #[test]
    fn the_int_workspace_is_a_constant_size() {
        let small_qo = [0i32, 1];
        let small_kv = [0i32, 16];
        let small_lens = [16i32];
        let big_qo: Vec<i32> = (0..=64).collect();
        let big_kv: Vec<i32> = (0..=64).map(|i| i * 4096).collect();
        let big_lens = vec![4096i32; 64];
        let ws = Workspace::new(1 << 25, 1 << 22);
        let small = plan(&request(&small_qo, &small_kv, &small_lens), &H100, ws).unwrap();
        let big = plan(&request(&big_qo, &big_kv, &big_lens), &H100, ws).unwrap();
        assert_eq!(small.int_bytes, big.int_bytes);
        assert_eq!(small.info.work_indptr_offset, big.info.work_indptr_offset);
    }

    /// A workspace that cannot hold the fixed-size arrays is refused by name.
    #[test]
    fn a_workspace_that_cannot_hold_the_plan_is_refused() {
        let qo = [0i32, 1];
        let kv = [0i32, 16];
        let lens = [16i32];
        let err =
            plan(&request(&qo, &kv, &lens), &H100, Workspace::new(1 << 24, 1024)).unwrap_err();
        assert!(matches!(err, Error::WorkspaceOverflow { what: "mla_q_indptr", .. }));
    }
}
