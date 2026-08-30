//! The sm90 prefill scheduler: per-CTA work lists balanced over a cost
//! heap, one list per SM, longest prefixes first. A native reimplementation
//! of FlashInfer's host planner (see [`sched`](crate::attn::sched)), kept
//! so `Struct(AttnPrefillPlanSm90)` has an honest payload; the launcher
//! this schedule feeds was never part of the lattice (the entry answers a
//! typed refusal — see `attn::prefill_sm90`).

use core::cmp::Reverse;

use crate::error::Error;

use crate::attn::plan::{Built, Device, PrefillPlanSm90Info};
use crate::attn::sched::{
    AlignedAllocator, at, CostHeap, Staging, cost_function, lengths, narrow, narrow_all,
    packed_causal_kv_end, spans,
};
use crate::jit::refuse;

#[derive(Clone, Copy, Debug)]
pub struct Request<'a> {
    /// Host copy of the query indptr — `[batch_size + 1]`.
    pub qo_indptr: &'a [i32],
    /// Host copy of the kv element offsets — `[batch_size + 1]`.
    pub kv_indptr: &'a [i32],
    /// Host per-request kv lengths, in tokens — `[batch_size]`.
    pub kv_len_arr: &'a [i32],
    pub total_num_rows: u32,
    pub batch_size: u32,
    pub num_qo_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub causal: bool,
    pub enable_cuda_graph: bool,
}

/// One request as the balancer walks it.
#[derive(Clone, Copy, Debug)]
struct Lane {
    request: i32,
    qo_len: u32,
    kv_len: u32,
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

/// The computed schedule: pure data, laid out and staged by [`plan`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schedule {
    pub same_schedule_for_all_heads: bool,
    pub total_num_works: usize,
    pub max_total_num_works: usize,
    pub qo_tile_indices: Vec<i32>,
    pub qo_indptr: Vec<i32>,
    pub kv_indptr: Vec<i32>,
    pub qo_len: Vec<i32>,
    pub kv_len: Vec<i32>,
    pub head_indices: Vec<i32>,
    pub batch_indices: Vec<i32>,
    pub work_indptr: Vec<i32>,
}

pub fn schedule(op: &'static str, req: &Request<'_>, device: &Device) -> Result<Schedule, Error> {
    if req.batch_size == 0 {
        return Err(refuse(op, "the batch is empty"));
    }
    if req.num_kv_heads == 0 || !req.num_qo_heads.is_multiple_of(req.num_kv_heads) {
        return Err(refuse(
            op,
            format!(
                "{} query heads are not a whole number of the {} kv heads",
                req.num_qo_heads, req.num_kv_heads
            ),
        ));
    }
    let batch = req.batch_size as usize;
    let qo_lens = spans(op, "qo_indptr", req.qo_indptr, batch)?;
    spans(op, "kv_indptr", req.kv_indptr, batch)?;
    let kv_lens = lengths(op, "kv length table", req.kv_len_arr, batch)?;
    narrow(op, "batch_prefill_sm90_head_indices", i64::from(req.num_qo_heads))?;

    // Longest prefixes place first; the stable sort keeps request order
    // between equal lengths.
    let mut lanes: Vec<Lane> = Vec::with_capacity(batch);
    for i in 0..batch {
        narrow(op, "batch_prefill_sm90_qo_len", i64::from(qo_lens[i]))?;
        narrow(op, "batch_prefill_sm90_kv_len", i64::from(kv_lens[i]))?;
        lanes.push(Lane {
            request: narrow(op, "batch_prefill_sm90_batch_indices", i as i64)?,
            qo_len: qo_lens[i],
            kv_len: kv_lens[i],
        });
    }
    lanes.sort_by_key(|lane| Reverse(lane.kv_len));

    let cta_tile_q: u32 = if req.head_dim == 64 { 192 } else { 128 };
    let num_ctas = device.num_sm;
    if num_ctas == 0 {
        return Err(refuse(op, "the stated device has no multiprocessors"));
    }

    let mut heap = CostHeap::new(num_ctas);
    let mut ctas = vec![CtaWork::default(); num_ctas as usize];

    let max_num_works_per_head = (u64::from(req.total_num_rows).div_ceil(u64::from(cta_tile_q))
        + u64::from(req.batch_size)
        - 1) as usize;
    let same_schedule_for_all_heads = max_num_works_per_head > 4096;

    let heads_scheduled = if same_schedule_for_all_heads {
        1
    } else {
        req.num_qo_heads
    };
    for qo_head_idx in 0..heads_scheduled {
        for lane in &lanes {
            let num_qo_tiles = lane.qo_len.div_ceil(cta_tile_q);
            for qo_tile_idx in (0..num_qo_tiles).rev() {
                let (cta_idx, accum_cost) = heap.pop();
                let effective_kv_len = if req.causal {
                    packed_causal_kv_end(
                        lane.qo_len,
                        lane.kv_len,
                        qo_tile_idx,
                        cta_tile_q,
                        num_qo_tiles,
                        1,
                    )
                } else {
                    lane.kv_len
                };
                heap.insert(
                    cta_idx,
                    accum_cost + cost_function(cta_tile_q, u64::from(effective_kv_len)),
                );
                let request = lane.request as usize;
                let cta = &mut ctas[cta_idx as usize];
                // Narrowed above: tile < tiles <= qo_len, head < heads.
                cta.qo_tile_indices.push(qo_tile_idx as i32);
                cta.qo_indptr.push(req.qo_indptr[request]);
                cta.qo_len.push(lane.qo_len as i32);
                cta.kv_indptr.push(req.kv_indptr[request]);
                cta.kv_len.push(lane.kv_len as i32);
                cta.head_indices.push(qo_head_idx as i32);
                cta.batch_indices.push(lane.request);
            }
        }
    }

    let mut work_indptr = vec![0i64];
    for cta in &ctas {
        work_indptr.push(
            work_indptr.last().expect("work_indptr starts with a zero")
                + cta.qo_tile_indices.len() as i64,
        );
    }
    let total_num_works = *work_indptr.last().expect("work_indptr has num_sm + 1 entries") as usize;
    let work_indptr = narrow_all(op, "batch_prefill_sm90_work_indptr", &work_indptr)?;

    let max_total_num_works = if req.enable_cuda_graph {
        if same_schedule_for_all_heads {
            max_num_works_per_head
        } else {
            max_num_works_per_head * req.num_qo_heads as usize
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
    op: &'static str,
    req: &Request<'_>,
    device: &Device,
    int_bytes: usize,
) -> Result<Built<PrefillPlanSm90Info>, Error> {
    let sched = schedule(op, req, device)?;
    let mut info = PrefillPlanSm90Info {
        same_schedule_for_all_heads: sched.same_schedule_for_all_heads,
        ..PrefillPlanSm90Info::default()
    };

    let works = 4 * sched.max_total_num_works;
    let mut ints = AlignedAllocator::new(op, int_bytes);
    info.qo_tile_indices_offset = Some(ints.alloc(works, 16, "batch_prefill_sm90_qo_tile_indices")?);
    info.qo_indptr_offset = Some(ints.alloc(works, 16, "batch_prefill_sm90_qo_offset")?);
    info.kv_indptr_offset = Some(ints.alloc(works, 16, "batch_prefill_sm90_kv_offset")?);
    info.qo_len_offset = Some(ints.alloc(works, 16, "batch_prefill_sm90_qo_len")?);
    info.kv_len_offset = Some(ints.alloc(works, 16, "batch_prefill_sm90_kv_len")?);
    info.head_indices_offset = Some(ints.alloc(works, 16, "batch_prefill_sm90_head_indices")?);
    info.work_indptr_offset = Some(ints.alloc(4 * (device.num_sm as usize + 1), 16, "batch_prefill_sm90_work_indptr")?);
    info.batch_indices_offset = Some(ints.alloc(works, 16, "batch_prefill_sm90_batch_indices")?);

    let writes: [(Option<u32>, &Vec<i32>, &'static str); 8] = [
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
    let int_bytes = ints.used();
    let mut staging = Staging::new(op, int_bytes);
    for (offset, values, what) in writes {
        staging.put_i32s(at(offset), values, what)?;
    }

    Ok(Built {
        info,
        int_upload: staging.into_upload(int_bytes),
        int_bytes,
        float_bytes: 0,
    })
}
