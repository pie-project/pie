use super::alloc::{AlignedAllocator, Staging};
use super::arith::{ceil_div_i32, ceil_div_u32, cost_function, packed_causal_kv_end};
use super::heap::MinHeap;
use super::info::PrefillPlanSm90Info;
use super::sort::sort;
use super::{Device, Error, Plan, Workspace};

#[derive(Clone, Copy, Debug)]
pub struct Request<'a> {
    pub qo_indptr: &'a [i32],
    pub kv_indptr: &'a [i32],
    pub kv_len_arr: &'a [i32],
    pub total_num_rows: u32,
    pub batch_size: u32,
    pub num_qo_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim_qk: u32,
    pub head_dim_vo: u32,
    pub page_size: u32,
    pub causal: bool,
    pub enable_cuda_graph: bool,
    pub sizeof_dtype_o: u32,
}

#[derive(Clone, Debug, Default)]
struct CtaWork {
    qo_tile_indices: Vec<i32>,
    qo_indptr: Vec<i32>,
    kv_indptr: Vec<i32>,
    qo_len: Vec<i32>,
    kv_len: Vec<i32>,
    head_indices: Vec<i32>,
    batch_indices: Vec<i32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schedule {
    pub same_schedule_for_all_heads: bool,
    pub total_num_works: i32,
    pub max_total_num_works: i32,
    pub qo_tile_indices: Vec<i32>,
    pub qo_indptr: Vec<i32>,
    pub kv_indptr: Vec<i32>,
    pub qo_len: Vec<i32>,
    pub kv_len: Vec<i32>,
    pub head_indices: Vec<i32>,
    pub batch_indices: Vec<i32>,
    pub work_indptr: Vec<i32>,
}

pub fn schedule(req: &Request<'_>, device: &Device) -> Result<Schedule, Error> {
    if req.num_kv_heads == 0 || !req.num_qo_heads.is_multiple_of(req.num_kv_heads) {
        return Err(Error::HeadsNotDivisible {
            num_qo_heads: req.num_qo_heads,
            num_kv_heads: req.num_kv_heads,
        });
    }
    let batch_size = req.batch_size as usize;
    for (array, len) in [
        ("qo_indptr", req.qo_indptr.len()),
        ("kv_indptr", req.kv_indptr.len()),
    ] {
        if len < batch_size + 1 {
            return Err(Error::IndptrTooShort {
                array,
                needed: batch_size + 1,
                got: len,
            });
        }
    }
    if req.kv_len_arr.len() < batch_size {
        return Err(Error::IndptrTooShort {
            array: "kv_len_arr",
            needed: batch_size,
            got: req.kv_len_arr.len(),
        });
    }

    let mut idx_qo_kv_len: Vec<(i32, i32, i32)> = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let qo_len = req.qo_indptr[i + 1].wrapping_sub(req.qo_indptr[i]);
        let kv_len = req.kv_len_arr[i];
        if kv_len < 0 {
            return Err(Error::NegativeSpan {
                array: "kv_len_arr",
                index: i,
                begin: 0,
                end: i64::from(kv_len),
            });
        }
        if qo_len < 0 {
            return Err(Error::NegativeSpan {
                array: "qo_indptr",
                index: i,
                begin: i64::from(req.qo_indptr[i]),
                end: i64::from(req.qo_indptr[i + 1]),
            });
        }
        idx_qo_kv_len.push((i as i32, qo_len, kv_len));
    }

    sort(
        &mut idx_qo_kv_len,
        &|a: &(i32, i32, i32), b: &(i32, i32, i32)| a.2 > b.2,
    );

    let cta_tile_q: i32 = if req.head_dim_vo == 64 { 192 } else { 128 };
    let num_ctas = device.num_sm;

    let mut heap = MinHeap::new(num_ctas);
    let mut ctas = vec![CtaWork::default(); num_ctas as usize];

    let max_num_works_per_head = ceil_div_u32(req.total_num_rows, cta_tile_q as u32)
        .wrapping_add(req.batch_size)
        .wrapping_sub(1) as i32;
    let same_schedule_for_all_heads = max_num_works_per_head > 4096;

    let heads_scheduled = if same_schedule_for_all_heads {
        1
    } else {
        req.num_qo_heads as i32
    };
    for qo_head_idx in 0..heads_scheduled {
        for &(i, qo_len, kv_len) in &idx_qo_kv_len {
            let num_qo_tiles = ceil_div_i32(qo_len, cta_tile_q);
            for qo_tile_idx in (0..num_qo_tiles).rev() {
                let (cta_idx, accum_cost) = heap.pop();
                let effective_kv_len = if req.causal {
                    packed_causal_kv_end(qo_len, kv_len, qo_tile_idx, cta_tile_q, num_qo_tiles, 1)
                } else {
                    kv_len
                };
                heap.insert((
                    cta_idx,
                    accum_cost + cost_function(cta_tile_q, effective_kv_len),
                ));
                let cta = &mut ctas[cta_idx as usize];
                cta.qo_tile_indices.push(qo_tile_idx);
                cta.qo_indptr.push(req.qo_indptr[i as usize]);
                cta.qo_len.push(qo_len);
                cta.kv_indptr.push(req.kv_indptr[i as usize]);
                cta.kv_len.push(kv_len);
                cta.head_indices.push(qo_head_idx);
                cta.batch_indices.push(i);
            }
        }
    }

    let mut work_indptr = vec![0i32; num_ctas as usize + 1];
    for i in 0..num_ctas as usize {
        work_indptr[i + 1] = work_indptr[i].wrapping_add(ctas[i].qo_tile_indices.len() as i32);
    }
    let total_num_works = *work_indptr
        .last()
        .expect("work_indptr has num_sm + 1 entries");

    let max_total_num_works = if req.enable_cuda_graph {
        if same_schedule_for_all_heads {
            max_num_works_per_head
        } else {
            max_num_works_per_head.wrapping_mul(req.num_qo_heads as i32)
        }
    } else {
        total_num_works
    };

    let flatten = |f: fn(&CtaWork) -> &Vec<i32>| -> Vec<i32> {
        ctas.iter().flat_map(|c| f(c).iter().copied()).collect()
    };
    Ok(Schedule {
        same_schedule_for_all_heads,
        total_num_works,
        max_total_num_works,
        qo_tile_indices: flatten(|c| &c.qo_tile_indices),
        qo_indptr: flatten(|c| &c.qo_indptr),
        kv_indptr: flatten(|c| &c.kv_indptr),
        qo_len: flatten(|c| &c.qo_len),
        kv_len: flatten(|c| &c.kv_len),
        head_indices: flatten(|c| &c.head_indices),
        batch_indices: flatten(|c| &c.batch_indices),
        work_indptr,
    })
}

pub fn plan(
    req: &Request<'_>,
    device: &Device,
    workspace: Workspace,
) -> Result<Plan<PrefillPlanSm90Info>, Error> {
    let sched = schedule(req, device)?;
    let mut staging = Staging::new(workspace.int_bytes);
    let mut info = PrefillPlanSm90Info {
        same_schedule_for_all_heads: sched.same_schedule_for_all_heads,
        ..PrefillPlanSm90Info::default()
    };

    let works = 4 * sched.max_total_num_works.max(0) as usize;
    let mut int_alloc = AlignedAllocator::new(workspace.int_bytes);
    info.qo_tile_indices_offset =
        int_alloc.alloc(works, 16, "batch_prefill_sm90_qo_tile_indices")? as i64;
    info.qo_indptr_offset = int_alloc.alloc(works, 16, "batch_prefill_sm90_qo_offset")? as i64;
    info.kv_indptr_offset = int_alloc.alloc(works, 16, "batch_prefill_sm90_kv_offset")? as i64;
    info.qo_len_offset = int_alloc.alloc(works, 16, "batch_prefill_sm90_qo_len")? as i64;
    info.kv_len_offset = int_alloc.alloc(works, 16, "batch_prefill_sm90_kv_len")? as i64;
    info.head_indices_offset =
        int_alloc.alloc(works, 16, "batch_prefill_sm90_head_indices")? as i64;
    info.work_indptr_offset = int_alloc.alloc(
        4 * (device.num_sm as usize + 1),
        16,
        "batch_prefill_sm90_work_indptr",
    )? as i64;
    info.batch_indices_offset =
        int_alloc.alloc(works, 16, "batch_prefill_sm90_batch_indices")? as i64;

    let writes: [(i64, &Vec<i32>, &str); 8] = [
        (
            info.qo_tile_indices_offset,
            &sched.qo_tile_indices,
            "batch_prefill_sm90_qo_tile_indices",
        ),
        (
            info.qo_indptr_offset,
            &sched.qo_indptr,
            "batch_prefill_sm90_qo_offset",
        ),
        (
            info.kv_indptr_offset,
            &sched.kv_indptr,
            "batch_prefill_sm90_kv_offset",
        ),
        (
            info.qo_len_offset,
            &sched.qo_len,
            "batch_prefill_sm90_qo_len",
        ),
        (
            info.kv_len_offset,
            &sched.kv_len,
            "batch_prefill_sm90_kv_len",
        ),
        (
            info.head_indices_offset,
            &sched.head_indices,
            "batch_prefill_sm90_head_indices",
        ),
        (
            info.work_indptr_offset,
            &sched.work_indptr,
            "batch_prefill_sm90_work_indptr",
        ),
        (
            info.batch_indices_offset,
            &sched.batch_indices,
            "batch_prefill_sm90_batch_indices",
        ),
    ];
    for (offset, values, what) in writes {
        staging.put_i32s(offset as usize, values, what)?;
    }

    let int_bytes = int_alloc.used();
    Ok(Plan {
        info,
        int_upload: staging.into_upload(int_bytes),
        int_bytes,
        float_bytes: 0,
    })
}
