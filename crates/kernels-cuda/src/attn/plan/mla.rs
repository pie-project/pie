use super::alloc::{AlignedAllocator, Staging};
use super::arith::{ceil_div_i32, ceil_div_i64, cost_function, packed_causal_kv_end};
use super::heap::MinHeap;
use super::info::MlaPlanInfo;
use super::{Device, Error, Plan, Workspace};

pub const MAX_TOTAL_NUM_WORKS: i32 = 16384;

#[derive(Clone, Copy, Debug)]
pub struct Request<'a> {
    pub qo_indptr: &'a [i32],
    pub kv_indptr: &'a [i32],
    pub kv_len_arr: &'a [i32],
    pub batch_size: u32,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schedule {
    pub cluster_size: i32,
    pub num_clusters: u32,
    pub cluster_tile_q: i32,
    pub kv_len_limit: i32,
    pub total_num_works: i32,
    pub partial_o_nnz: i32,
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

#[allow(clippy::too_many_lines)]
pub fn schedule(req: &Request<'_>, device: &Device) -> Result<Schedule, Error> {
    let batch_size = req.batch_size as usize;
    if batch_size == 0 {
        return Err(Error::EmptyBatch);
    }
    if device.num_sm == 0 {
        return Err(Error::MergeCtasExceedSm {
            counter: 0,
            num_sm: 0,
        });
    }
    for (array, len, needed) in [
        ("qo_indptr", req.qo_indptr.len(), batch_size + 1),
        ("kv_indptr", req.kv_indptr.len(), batch_size + 1),
        ("kv_len_arr", req.kv_len_arr.len(), batch_size),
    ] {
        if len < needed {
            return Err(Error::IndptrTooShort {
                array,
                needed,
                got: len,
            });
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
        return Err(Error::MergeCtasExceedSm {
            counter: 0,
            num_sm: i64::from(device.num_sm),
        });
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
                        offset_start
                            .wrapping_add(row_chunk_size)
                            .min(current_q_tile_end) as u32,
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
                heap.insert((
                    cluster_idx,
                    accum_cost + cost_function(cluster_tile_q, actual_len),
                ));
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
    let total_num_works = *work_indptr
        .last()
        .expect("work_indptr has num_clusters + 1 entries");
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
    for (offset, values, what) in writes {
        staging.put_i32s(offset as usize, values, what)?;
    }

    const SIZEOF_DTYPE_O: usize = 2;
    let mut float_alloc = AlignedAllocator::new(workspace.float_bytes);
    let rows = 2 * sched.num_clusters as usize * sched.cluster_tile_q as usize;
    info.partial_o_offset = float_alloc.alloc(
        rows * SIZEOF_DTYPE_O * req.head_dim_o as usize,
        16,
        "mla_partial_o",
    )? as i64;
    info.partial_lse_offset = float_alloc.alloc(rows * 4, 16, "mla_partial_lse")? as i64;

    let int_bytes = int_alloc.used();
    Ok(Plan {
        info,
        int_upload: staging.into_upload(int_bytes),
        int_bytes,
        float_bytes: float_alloc.used(),
    })
}
