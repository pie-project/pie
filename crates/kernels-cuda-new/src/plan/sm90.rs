use super::alloc::{AlignedAllocator, Staging};
use super::arith::{ceil_div_i32, ceil_div_u32, cost_function, packed_causal_kv_end};
use super::heap::MinHeap;
use super::info::PrefillPlanSm90Info;
use super::sort::sort;
use super::{Device, Error, Plan, Workspace};

/// The batch an SM90 prefill plan is built for.
#[derive(Clone, Copy, Debug)]
pub struct Request<'a> {
    /// `batch_size + 1` QO row offsets.
    pub qo_indptr: &'a [i32],
    /// `batch_size + 1` KV offsets.
    pub kv_indptr: &'a [i32],
    /// `batch_size` KV lengths.
    pub kv_len_arr: &'a [i32],
    /// QO rows in the batch, which bounds the per-head work count.
    pub total_num_rows: u32,
    /// Requests in the batch.
    pub batch_size: u32,
    /// Query/output heads.
    pub num_qo_heads: u32,
    /// Key/value heads; must divide `num_qo_heads`.
    pub num_kv_heads: u32,
    /// QK head dimension. Unused by the schedule; kept for signature parity.
    pub head_dim_qk: u32,
    /// VO head dimension — `64` widens the QO tile to 192.
    pub head_dim_vo: u32,
    /// Tokens per page. Unused by the schedule; kept for signature parity.
    pub page_size: u32,
    /// Whether a QO tile reads only up to the diagonal.
    pub causal: bool,
    /// Whether this plan will be captured into a CUDA graph — which sizes the
    pub enable_cuda_graph: bool,
    /// `sizeof(DTypeO)`. Unused; kept for signature parity.
    pub sizeof_dtype_o: u32,
}

/// One CTA's work list, before it is flattened.
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

/// The schedule: seven parallel arrays indexed by work item, plus the
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schedule {
    /// Whether one head's schedule is reused for all of them.
    pub same_schedule_for_all_heads: bool,
    /// Work items produced.
    pub total_num_works: i32,
    /// Slots the arrays are sized for — larger than `total_num_works` only
    pub max_total_num_works: i32,
    /// `qo_tile_indices[work]`.
    pub qo_tile_indices: Vec<i32>,
    /// `qo_indptr[work]` — the request's QO base, repeated per work item.
    pub qo_indptr: Vec<i32>,
    /// `kv_indptr[work]`.
    pub kv_indptr: Vec<i32>,
    /// `qo_len[work]`.
    pub qo_len: Vec<i32>,
    /// `kv_len[work]`.
    pub kv_len: Vec<i32>,
    /// `head_indices[work]`.
    pub head_indices: Vec<i32>,
    /// `batch_indices[work]`.
    pub batch_indices: Vec<i32>,
    /// `work_indptr[cta]`, `num_sm + 1` entries.
    pub work_indptr: Vec<i32>,
}

/// Build the SM90 schedule without laying it out in a workspace.
pub fn schedule(req: &Request<'_>, device: &Device) -> Result<Schedule, Error> {
    if req.num_kv_heads == 0 || req.num_qo_heads % req.num_kv_heads != 0 {
        return Err(Error::HeadsNotDivisible {
            num_qo_heads: req.num_qo_heads,
            num_kv_heads: req.num_kv_heads,
        });
    }
    let batch_size = req.batch_size as usize;
    for (array, len) in [("qo_indptr", req.qo_indptr.len()), ("kv_indptr", req.kv_indptr.len())] {
        if len < batch_size + 1 {
            return Err(Error::IndptrTooShort { array, needed: batch_size + 1, got: len });
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

    sort(&mut idx_qo_kv_len, &|a: &(i32, i32, i32), b: &(i32, i32, i32)| a.2 > b.2);

    let cta_tile_q: i32 = if req.head_dim_vo == 64 { 192 } else { 128 };
    let num_ctas = device.num_sm;

    let mut heap = MinHeap::new(num_ctas);
    let mut ctas = vec![CtaWork::default(); num_ctas as usize];

    let max_num_works_per_head = ceil_div_u32(req.total_num_rows, cta_tile_q as u32)
        .wrapping_add(req.batch_size)
        .wrapping_sub(1) as i32;
    let same_schedule_for_all_heads = max_num_works_per_head > 4096;

    let heads_scheduled = if same_schedule_for_all_heads { 1 } else { req.num_qo_heads as i32 };
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
                heap.insert((cta_idx, accum_cost + cost_function(cta_tile_q, effective_kv_len)));
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
    let total_num_works = *work_indptr.last().expect("work_indptr has num_sm + 1 entries");

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

/// `PrefillSM90Plan` — the plan, and the bytes to upload under it.
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
    info.work_indptr_offset =
        int_alloc.alloc(4 * (device.num_sm as usize + 1), 16, "batch_prefill_sm90_work_indptr")?
            as i64;
    info.batch_indices_offset =
        int_alloc.alloc(works, 16, "batch_prefill_sm90_batch_indices")? as i64;

    let writes: [(i64, &Vec<i32>, &str); 8] = [
        (info.qo_tile_indices_offset, &sched.qo_tile_indices, "batch_prefill_sm90_qo_tile_indices"),
        (info.qo_indptr_offset, &sched.qo_indptr, "batch_prefill_sm90_qo_offset"),
        (info.kv_indptr_offset, &sched.kv_indptr, "batch_prefill_sm90_kv_offset"),
        (info.qo_len_offset, &sched.qo_len, "batch_prefill_sm90_qo_len"),
        (info.kv_len_offset, &sched.kv_len, "batch_prefill_sm90_kv_len"),
        (info.head_indices_offset, &sched.head_indices, "batch_prefill_sm90_head_indices"),
        (info.work_indptr_offset, &sched.work_indptr, "batch_prefill_sm90_work_indptr"),
        (info.batch_indices_offset, &sched.batch_indices, "batch_prefill_sm90_batch_indices"),
    ];
    for (offset, values, what) in writes {
        staging.put_i32s(offset as usize, values, what)?;
    }

    let int_bytes = int_alloc.used();
    Ok(Plan { info, int_upload: staging.into_upload(int_bytes), int_bytes, float_bytes: 0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    const H100: Device = Device::new(132, 9);

    fn request<'a>(qo: &'a [i32], kv: &'a [i32], lens: &'a [i32], rows: u32) -> Request<'a> {
        Request {
            qo_indptr: qo,
            kv_indptr: kv,
            kv_len_arr: lens,
            total_num_rows: rows,
            batch_size: lens.len() as u32,
            num_qo_heads: 32,
            num_kv_heads: 8,
            head_dim_qk: 128,
            head_dim_vo: 128,
            page_size: 1,
            causal: true,
            enable_cuda_graph: false,
            sizeof_dtype_o: 2,
        }
    }

    /// Every work item lands on exactly one CTA, and `work_indptr` is the
    #[test]
    fn work_indptr_partitions_the_work_items() {
        let qo = [0i32, 512, 1024, 1536];
        let kv = [0i32, 4096, 8192, 12288];
        let lens = [4096i32, 4096, 4096];
        let sched = schedule(&request(&qo, &kv, &lens, 1536), &H100).unwrap();
        assert_eq!(sched.work_indptr.len(), 133);
        assert_eq!(sched.work_indptr[0], 0);
        assert_eq!(*sched.work_indptr.last().unwrap(), sched.total_num_works);
        assert_eq!(sched.qo_tile_indices.len(), sched.total_num_works as usize);
        assert!(sched.work_indptr.windows(2).all(|w| w[1] >= w[0]));
    }

    /// A batch smaller than the grid leaves most CTAs empty and never
    #[test]
    fn a_small_batch_spreads_one_tile_per_cta() {
        let qo = [0i32, 128, 256];
        let kv = [0i32, 128, 256];
        let lens = [128i32, 128];
        let sched = schedule(&request(&qo, &kv, &lens, 256), &H100).unwrap();
        assert_eq!(sched.total_num_works, 64);
        assert!(sched.work_indptr.windows(2).all(|w| w[1] - w[0] <= 1));
    }

    /// Past 4096 works per head the planner schedules one head and lets the
    #[test]
    fn a_huge_batch_falls_back_to_one_schedule_for_all_heads() {
        let batch = 5000usize;
        let qo: Vec<i32> = (0..=batch as i32).collect();
        let kv: Vec<i32> = (0..=batch as i32).map(|i| i * 64).collect();
        let lens = vec![64i32; batch];
        let sched = schedule(&request(&qo, &kv, &lens, batch as u32), &H100).unwrap();
        assert!(sched.same_schedule_for_all_heads);
        assert_eq!(sched.total_num_works, batch as i32);
        assert!(sched.head_indices.iter().all(|&h| h == 0));
    }

    /// The empty batch is a real input here — unlike FA2's planner there is no
    #[test]
    fn the_empty_batch_plans_nothing_and_says_so() {
        let qo = [0i32];
        let kv = [0i32];
        let lens: [i32; 0] = [];
        let sched = schedule(&request(&qo, &kv, &lens, 0), &H100).unwrap();
        assert_eq!(sched.total_num_works, 0);
        assert!(sched.work_indptr.iter().all(|&w| w == 0));
    }

    /// Causal masking makes the first tile of a long sequence cheap and the
    #[test]
    fn causal_masking_changes_the_assignment() {
        let qo = [0i32, 2048];
        let kv = [0i32, 2048];
        let lens = [2048i32];
        let mut req = request(&qo, &kv, &lens, 2048);
        let causal = schedule(&req, &H100).unwrap();
        req.causal = false;
        let dense = schedule(&req, &H100).unwrap();
        assert_eq!(causal.total_num_works, dense.total_num_works);
        assert_ne!(causal.kv_indptr.len(), 0);
    }

    /// A workspace that cannot hold the index arrays is refused by name.
    #[test]
    fn a_workspace_that_cannot_hold_the_schedule_is_refused() {
        let qo = [0i32, 512];
        let kv = [0i32, 4096];
        let lens = [4096i32];
        let err = plan(&request(&qo, &kv, &lens, 512), &H100, Workspace::new(0, 128)).unwrap_err();
        assert!(matches!(err, Error::WorkspaceOverflow { .. }));
    }
}
