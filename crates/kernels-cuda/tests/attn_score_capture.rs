//! **THE OBSERVABILITY DOOR'S SCORE CAPTURE, AGAINST A CPU REFERENCE.**
//!
//! ```text
//! cargo test -p kernels-cuda --features cuda-13 --release \
//!     --test attn_score_capture -- --ignored --nocapture
//! ```
//!
//! `attn_score::capture` writes per-key attention mass into a caller-owned
//! F32 slab (`.wiki/alto/attn-score.md` §4). Its whole contract is one
//! sentence — *each output row is a probability distribution over
//! `[0, kv_len)` summing to one, and exactly `0.0` on `[kv_len, kv_max)`* —
//! and every property below is a way that sentence can be silently false on
//! a machine that never faults:
//!
//! ```text
//! (1) every captured row sums to 1.0 over [0, kv_len), across page counts,
//!     grouped heads, window widths and a kv_len that is not a whole
//!     number of pages
//! (2) the tail [kv_len, kv_max) is EXACTLY zero -- including after a
//!     second, shorter capture into the same slab. A stale tail is the
//!     previous fire's mass on keys that no longer exist, and an eviction
//!     policy would rank on it
//! (3) two lanes at different `lane_offset` write disjoint rows: neither
//!     lane's mass appears in the other's block, and neither disturbs it
//! (4) a hand-computed 2-key, 1-head, head_dim-4 softmax matches to 1e-4 --
//!     the arithmetic itself, with no reference implementation in the way
//! (5) `observe = 1` on a one-row query IS the plain softmax of that row
//!     (the TOVA quantity), and `observe = 4` over four rows is the MEAN of
//!     the four rows' own causal distributions (the SnapKV one)
//! (6) the refusals fire by name -- the sliding window above all, which is
//!     semantic and not a missing instantiation
//! ```

#![cfg(feature = "_cuda")]

mod common;

use common::{Gpu, Lcg, to_bf16};

use dtype::Dtype;
use kernels_cuda::attn::plan::{Device, PrefillPlan, PrefillPlanInfo, Shape, Workspace};
use kernels_cuda::attn_score;
use kernels_cuda::tensor::{KvPool, RaggedTensor, Tensor};

/// How far a captured mass may sit from the f32 reference computed on the
/// same bf16 inputs. The kernel reads bf16 keys and queries and sums in f32,
/// exactly as the reference does; what is left is `__expf`'s own error and
/// the order the two add in.
const TOLERANCE: f32 = 1.0e-4;

/// The sum a distribution must land on.
const SUM_TOLERANCE: f32 = 1.0e-3;

/// One capture's geometry: a hand-built paged cache and the window over it.
struct Case {
    what: &'static str,
    page_size: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    kv_max: u32,
    observe: u32,
    /// Whether the pages are laid out `[page][head][token][dim]`. Every pool
    /// `engine-cuda` binds is NHD (`store.rs`'s `NHD`), but the entry reads
    /// the enumerator off the pool and stamps either arm, so both are walked
    /// here.
    hnd: bool,
    /// `(kv_len, qo_len)` per request.
    lanes: &'static [(u32, u32)],
}

const CASES: &[Case] = &[
    Case {
        what: "one request, one head, one whole page",
        page_size: 4,
        q_heads: 1,
        kv_heads: 1,
        head_dim: 64,
        kv_max: 8,
        observe: 1,
        hnd: false,
        lanes: &[(4, 1)],
    },
    Case {
        what: "two requests, grouped heads, a ragged last page",
        page_size: 4,
        q_heads: 4,
        kv_heads: 2,
        head_dim: 64,
        kv_max: 16,
        observe: 2,
        hnd: false,
        lanes: &[(6, 3), (3, 2)],
    },
    Case {
        what: "an observation window wider than the query rows it has",
        page_size: 2,
        q_heads: 2,
        kv_heads: 2,
        head_dim: 72,
        kv_max: 12,
        observe: 8,
        hnd: false,
        lanes: &[(7, 3)],
    },
    Case {
        what: "a 128-wide head over more keys than the block has warps",
        page_size: 8,
        q_heads: 2,
        kv_heads: 1,
        head_dim: 128,
        kv_max: 40,
        observe: 4,
        hnd: false,
        lanes: &[(33, 5)],
    },
];

impl Case {
    fn requests(&self) -> usize {
        self.lanes.len()
    }

    fn group(&self) -> usize {
        (self.q_heads / self.kv_heads) as usize
    }

    /// Pages per request, and the running page table the pool reads.
    fn pages_of(&self, kv_len: u32) -> u32 {
        kv_len.div_ceil(self.page_size)
    }

    fn page_indptr(&self) -> Vec<i32> {
        let mut out = vec![0i32];
        for (kv_len, _) in self.lanes {
            let last = *out.last().expect("seeded");
            out.push(last + self.pages_of(*kv_len) as i32);
        }
        out
    }

    fn last_page_lens(&self) -> Vec<i32> {
        self.lanes
            .iter()
            .map(|(kv_len, _)| {
                let rem = kv_len % self.page_size;
                (if rem == 0 { self.page_size } else { rem }) as i32
            })
            .collect()
    }

    /// **THE PAGE TABLE IS NOT THE IDENTITY**: physical pages are handed out
    /// backwards, so a kernel that ignored `kv_page_indices` and addressed
    /// pages by their logical order would read the wrong keys everywhere.
    fn page_indices(&self) -> Vec<i32> {
        let total = *self.page_indptr().last().expect("seeded") as usize;
        (0..total).map(|n| (total - 1 - n) as i32).collect()
    }

    fn qo_indptr(&self) -> Vec<i32> {
        let mut out = vec![0i32];
        for (_, qo_len) in self.lanes {
            let last = *out.last().expect("seeded");
            out.push(last + *qo_len as i32);
        }
        out
    }

    fn q_rows(&self) -> usize {
        self.lanes.iter().map(|(_, qo)| *qo as usize).sum()
    }

    fn sm_scale(&self) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        let width = self.head_dim as f32;
        1.0 / width.sqrt()
    }
}

/// A hand-built cache: the bf16 page slab as the device reads it, the query
/// rows, and the f32 twins of both for the reference.
struct Built {
    k_pages: Vec<u16>,
    /// `keys[request][key][kv_head * head_dim + d]`, in f32.
    keys: Vec<Vec<Vec<f32>>>,
    q_raw: Vec<u16>,
    q: Vec<f32>,
}

fn build(case: &Case, seed: u64) -> Built {
    let mut rng = Lcg::seeded(seed);
    let page_indptr = case.page_indptr();
    let page_indices = case.page_indices();
    let total_pages = *page_indptr.last().expect("seeded") as usize;
    let head_dim = case.head_dim as usize;
    let kv_width = (case.kv_heads * case.head_dim) as usize;
    let page_elems = case.page_size as usize * kv_width;

    let mut k_pages = vec![0u16; total_pages * page_elems];
    let mut keys = Vec::with_capacity(case.requests());

    for (r, (kv_len, _)) in case.lanes.iter().enumerate() {
        let first = page_indptr[r] as usize;
        let mut rows = Vec::with_capacity(*kv_len as usize);
        for j in 0..*kv_len as usize {
            let (raw, exact) = rng.row(kv_width);
            let page = page_indices[first + j / case.page_size as usize] as usize;
            let offset = j % case.page_size as usize;
            if case.hnd {
                // HND: `[page][head][token][dim]` -- one head's plane is
                // contiguous in tokens, so a key's row is scattered.
                for h in 0..case.kv_heads as usize {
                    let at = ((page * case.kv_heads as usize + h) * case.page_size as usize
                        + offset)
                        * head_dim;
                    k_pages[at..at + head_dim]
                        .copy_from_slice(&raw[h * head_dim..(h + 1) * head_dim]);
                }
            } else {
                // NHD: `[page][token][head][dim]`, the one layout
                // `engine-cuda`'s store binds (`store.rs`'s `NHD`).
                let at = (page * case.page_size as usize + offset) * kv_width;
                k_pages[at..at + kv_width].copy_from_slice(&raw);
            }
            rows.push(exact);
        }
        keys.push(rows);
    }

    let (q_raw, q) = rng.row(case.q_rows() * (case.q_heads * case.head_dim) as usize);
    Built {
        k_pages,
        keys,
        q_raw,
        q,
    }
}

/// The distributions the kernel claims, one per `(request, head)`, each
/// `kv_max` wide — the whole row, tail included.
fn reference(case: &Case, built: &Built) -> Vec<Vec<f32>> {
    let qo_indptr = case.qo_indptr();
    let head_dim = case.head_dim as usize;
    let q_heads = case.q_heads as usize;
    let group = case.group();
    let mut out = Vec::new();

    for (r, (kv_len, qo_len)) in case.lanes.iter().enumerate() {
        let kv_len = *kv_len as i32;
        let rows = (case.observe).min(*qo_len) as i32;
        for head in 0..q_heads {
            let kv_head = head / group;
            let mut row = vec![0.0f32; case.kv_max as usize];
            if kv_len > 0 && rows > 0 {
                for w in 0..rows {
                    let q_index = (qo_indptr[r + 1] - rows + w) as usize;
                    let limit = (kv_len - rows + w + 1).min(kv_len);
                    if limit <= 0 {
                        continue;
                    }
                    let q_row = &built.q[(q_index * q_heads + head) * head_dim..][..head_dim];
                    let scores: Vec<f32> = (0..limit as usize)
                        .map(|j| {
                            let k_row = &built.keys[r][j][kv_head * head_dim..][..head_dim];
                            let dot: f32 = q_row.iter().zip(k_row).map(|(a, b)| a * b).sum();
                            dot * case.sm_scale()
                        })
                        .collect();
                    let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let weights: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
                    let total: f32 = weights.iter().sum();
                    #[allow(clippy::cast_precision_loss)]
                    let per_row = total * rows as f32;
                    for (j, weight) in weights.iter().enumerate() {
                        if j < case.kv_max as usize {
                            row[j] += weight / per_row;
                        }
                    }
                }
            }
            out.push(row);
        }
    }
    out
}

/// One fire's slab, read back whole. `lane_offset`, `plane_stride` and
/// `plane` are the caller's own; `slab_lanes` is how many lanes' blocks the
/// slab holds.
#[allow(clippy::too_many_arguments)]
fn fire(
    case: &Case,
    built: &Built,
    lane_offset: u32,
    plane_stride: u32,
    plane: u32,
    slab_lanes: u32,
    prefill: Option<f32>,
) -> Vec<f32> {
    let mut gpu = Gpu::open();
    let slab_rows = slab_lanes * plane_stride;
    let slab = vec![prefill.unwrap_or(0.0); (slab_rows * case.kv_max) as usize];
    let scores_at = gpu.up(&slab);
    let got = fire_into(case, built, &mut gpu, scores_at, slab_rows, lane_offset, plane_stride, plane);
    assert!(got, "the capture enqueues");
    gpu.sync();
    gpu.down(scores_at, slab.len())
}

/// The fire itself, against a slab the caller already owns.
#[allow(clippy::too_many_arguments)]
fn fire_into(
    case: &Case,
    built: &Built,
    gpu: &mut Gpu,
    scores_at: u64,
    slab_rows: u32,
    lane_offset: u32,
    plane_stride: u32,
    plane: u32,
) -> bool {
    let q_at = gpu.up(&built.q_raw);
    let qo_at = gpu.up(&case.qo_indptr());
    let k_at = gpu.up(&built.k_pages);
    let indices_at = gpu.up(&case.page_indices());
    let indptr_at = gpu.up(&case.page_indptr());
    let lens_at = gpu.up(&case.last_page_lens());

    let pool = pool_of(case, k_at, indices_at, indptr_at, lens_at);
    let plan = plan_of(case);
    let mut scores = Tensor::new(scores_at, slab_rows, case.kv_max, Dtype::F32);
    let q = RaggedTensor {
        data: Tensor::new(
            q_at,
            case.q_rows() as u32,
            case.q_heads * case.head_dim,
            Dtype::Bf16,
        ),
        indptr: Tensor::new(qo_at, case.requests() as u32 + 1, 1, Dtype::I32),
    };
    attn_score::capture(
        &gpu.ctx(),
        q,
        &plan,
        &pool,
        None,
        case.head_dim,
        case.kv_heads,
        case.sm_scale(),
        case.observe,
        lane_offset,
        plane_stride,
        plane,
        case.kv_max,
        &mut scores,
    )
    .is_ok()
}

fn pool_of(case: &Case, keys: u64, indices: u64, indptr: u64, lens: u64) -> KvPool {
    let kv_width = case.kv_heads * case.head_dim;
    let total_pages = *case.page_indptr().last().expect("seeded") as u32;
    KvPool {
        keys: Tensor::new(keys, total_pages * case.page_size, kv_width, Dtype::Bf16),
        values: Tensor::new(keys, total_pages * case.page_size, kv_width, Dtype::Bf16),
        bf16_keys: Tensor::ABSENT,
        bf16_values: Tensor::ABSENT,
        key_scales: Tensor::ABSENT,
        value_scales: Tensor::ABSENT,
        page_indices: Tensor::new(indices, total_pages, 1, Dtype::I32),
        page_indptr: Tensor::new(indptr, case.requests() as u32 + 1, 1, Dtype::I32),
        last_page_lens: Tensor::new(lens, case.requests() as u32, 1, Dtype::I32),
        row_valid: Tensor::ABSENT,
        env_min: Tensor::ABSENT,
        env_max: Tensor::ABSENT,
        has_envelopes: false,
        page_size: case.page_size as i32,
        // The stride pair spells the layout the enumerator names -- under
        // NHD one token's step is the plane's whole width and a head plane
        // is a share of it; under HND the two swap.
        seq_stride: if case.hnd { i64::from(case.head_dim) } else { i64::from(kv_width) },
        head_stride: if case.hnd {
            i64::from(case.page_size) * i64::from(case.head_dim)
        } else {
            i64::from(case.head_dim)
        },
        layout: i32::from(case.hnd),
        scheme_byte: 0,
        block_size: 0,
        max_pages_per_request: 64,
        pages_in_batch: total_pages as i32,
    }
}

/// The plan is read for ONE thing — the shape triple `accepts` checks — so a
/// hand-built one with an empty schedule is the honest fixture here.
fn plan_of(case: &Case) -> PrefillPlan {
    PrefillPlan {
        info: PrefillPlanInfo::default(),
        int_upload: Vec::new(),
        int_bytes: 0,
        float_bytes: 0,
        workspace: Workspace {
            int_ptr: 0,
            int_bytes: 0,
            float_ptr: 0,
            float_bytes: 0,
        },
        shape: Shape {
            num_requests: case.requests() as u32,
            num_q_heads: case.q_heads,
            num_kv_heads: case.kv_heads,
            head_dim: case.head_dim,
            page_size: case.page_size,
            hnd_layout: case.hnd,
        },
        total_tokens: case.q_rows() as u32,
        window: None,
        causal: true,
        graph_capturable: false,
        mask_indptr: None,
        device: Device::L40S,
    }
}

/// (1) and (4)'s general half: every captured row is the reference
/// distribution, and it sums to one over the live keys.
#[test]
#[ignore = "real-hardware: needs a CUDA device and `--features cuda-13`; run it with `-- --ignored`"]
fn every_captured_row_is_a_distribution_over_the_live_keys() {
    for case in CASES {
        let built = build(case, 17);
        let plane_stride = case.q_heads;
        let got = fire(case, &built, 0, plane_stride, 0, case.requests() as u32, None);
        let want = reference(case, &built);

        for (r, (kv_len, _)) in case.lanes.iter().enumerate() {
            for head in 0..case.q_heads as usize {
                let row = (r * plane_stride as usize + head) * case.kv_max as usize;
                let landed = &got[row..row + case.kv_max as usize];
                let expected = &want[r * case.q_heads as usize + head];

                let sum: f32 = landed[..*kv_len as usize].iter().sum();
                assert!(
                    (sum - 1.0).abs() <= SUM_TOLERANCE,
                    "{}: request {r} head {head} sums to {sum}, not one",
                    case.what
                );
                for j in 0..case.kv_max as usize {
                    assert!(
                        (landed[j] - expected[j]).abs() <= TOLERANCE,
                        "{}: request {r} head {head} key {j} landed {} and the reference \
                         says {}",
                        case.what,
                        landed[j],
                        expected[j]
                    );
                }
            }
        }
    }
}

/// (2) THE TAIL IS ZERO, AND STAYS ZERO. The first half is easy; the second
/// is the one that matters — a shorter capture into a slab a longer one
/// already wrote must leave no mass past its own `kv_len`.
#[test]
#[ignore = "real-hardware: needs a CUDA device and `--features cuda-13`; run it with `-- --ignored`"]
fn a_shorter_second_capture_leaves_no_stale_tail() {
    const LONG: Case = Case {
        what: "the first, long capture",
        page_size: 4,
        q_heads: 2,
        kv_heads: 1,
        head_dim: 64,
        kv_max: 32,
        observe: 2,
        hnd: false,
        lanes: &[(20, 3)],
    };
    const SHORT: Case = Case {
        what: "the second, shorter capture into the same slab",
        page_size: 4,
        q_heads: 2,
        kv_heads: 1,
        head_dim: 64,
        kv_max: 32,
        observe: 2,
        hnd: false,
        lanes: &[(5, 3)],
    };

    let mut gpu = Gpu::open();
    let slab_rows = LONG.q_heads;
    let slab = vec![9.75f32; (slab_rows * LONG.kv_max) as usize];
    let scores_at = gpu.up(&slab);

    let long_built = build(&LONG, 23);
    assert!(
        fire_into(&LONG, &long_built, &mut gpu, scores_at, slab_rows, 0, LONG.q_heads, 0),
        "the long capture enqueues"
    );
    gpu.sync();
    let after_long: Vec<f32> = gpu.down(scores_at, slab.len());
    for head in 0..LONG.q_heads as usize {
        let row = head * LONG.kv_max as usize;
        for j in 20..LONG.kv_max as usize {
            assert_eq!(
                after_long[row + j],
                0.0,
                "head {head} key {j} is not zero past the first capture's kv_len"
            );
        }
        assert!(
            after_long[row + 19] > 0.0,
            "head {head} key 19 carries no mass, so the long capture never ran"
        );
    }

    let short_built = build(&SHORT, 29);
    assert!(
        fire_into(&SHORT, &short_built, &mut gpu, scores_at, slab_rows, 0, SHORT.q_heads, 0),
        "the short capture enqueues"
    );
    gpu.sync();
    let after_short: Vec<f32> = gpu.down(scores_at, slab.len());
    for head in 0..SHORT.q_heads as usize {
        let row = head * SHORT.kv_max as usize;
        let sum: f32 = after_short[row..row + 5].iter().sum();
        assert!(
            (sum - 1.0).abs() <= SUM_TOLERANCE,
            "head {head} sums to {sum} after the short capture"
        );
        for j in 5..SHORT.kv_max as usize {
            assert_eq!(
                after_short[row + j],
                0.0,
                "head {head} key {j} kept the LONG capture's mass -- a stale tail"
            );
        }
    }
}

/// (3): two lanes into one slab. The second lane's fire must not move the
/// first lane's block, and the first lane's mass must not appear in the
/// second's.
#[test]
#[ignore = "real-hardware: needs a CUDA device and `--features cuda-13`; run it with `-- --ignored`"]
fn two_lanes_write_disjoint_rows_and_leave_each_other_alone() {
    const CASE: Case = Case {
        what: "one lane's block",
        page_size: 4,
        q_heads: 2,
        kv_heads: 2,
        head_dim: 64,
        kv_max: 16,
        observe: 2,
        hnd: false,
        lanes: &[(6, 2)],
    };
    // Four planes per lane, this layer at plane 1: a lane's block is wider
    // than one layer's heads, which is the geometry the slab is carved at.
    const PLANE_STRIDE: u32 = 4;
    const PLANE: u32 = 1;
    const LANES: u32 = 3;

    let mut gpu = Gpu::open();
    let slab_rows = LANES * PLANE_STRIDE;
    let slab = vec![-1.0f32; (slab_rows * CASE.kv_max) as usize];
    let scores_at = gpu.up(&slab);

    let first = build(&CASE, 31);
    let second = build(&CASE, 37);
    assert!(
        fire_into(&CASE, &first, &mut gpu, scores_at, slab_rows, 0, PLANE_STRIDE, PLANE),
        "lane 0's capture enqueues"
    );
    gpu.sync();
    let after_first: Vec<f32> = gpu.down(scores_at, slab.len());

    assert!(
        fire_into(&CASE, &second, &mut gpu, scores_at, slab_rows, 2, PLANE_STRIDE, PLANE),
        "lane 2's capture enqueues"
    );
    gpu.sync();
    let after_second: Vec<f32> = gpu.down(scores_at, slab.len());

    let width = CASE.kv_max as usize;
    for row in 0..slab_rows as usize {
        let lane = row as u32 / PLANE_STRIDE;
        let plane = row as u32 % PLANE_STRIDE;
        let head = plane.checked_sub(PLANE);
        let written_first = lane == 0 && matches!(head, Some(h) if h < CASE.q_heads);
        let written_second = lane == 2 && matches!(head, Some(h) if h < CASE.q_heads);

        let before = &after_first[row * width..][..width];
        let after = &after_second[row * width..][..width];

        if !written_first {
            assert!(
                before.iter().all(|v| *v == -1.0),
                "row {row} was written by lane 0's capture and is not its rows"
            );
        }
        assert_eq!(
            before == after,
            !written_second,
            "row {row} {} when lane 2 captured",
            if written_second { "did not move" } else { "moved" }
        );
        if written_first || written_second {
            let sum: f32 = after[..6].iter().sum();
            assert!(
                (sum - 1.0).abs() <= SUM_TOLERANCE,
                "row {row} sums to {sum}"
            );
        }
    }
}

/// (4): the arithmetic itself. Two keys, one head, four lanes of head — the
/// softmax is a number a reader can check by hand, and the test computes it
/// from the bf16 words rather than from a reference implementation.
#[test]
#[ignore = "real-hardware: needs a CUDA device and `--features cuda-13`; run it with `-- --ignored`"]
fn a_two_key_row_is_the_softmax_written_out_by_hand() {
    const CASE: Case = Case {
        what: "two keys, one head, head_dim 4",
        page_size: 2,
        q_heads: 1,
        kv_heads: 1,
        head_dim: 4,
        kv_max: 4,
        observe: 1,
        hnd: false,
        lanes: &[(2, 1)],
    };

    // Exact in bf16, so the device and this test read the same numbers.
    let q_exact = [1.0f32, 0.5, -0.25, 2.0];
    let k0 = [0.5f32, 1.0, 2.0, -0.5];
    let k1 = [-1.0f32, 0.25, 0.5, 1.5];

    let raw = |xs: &[f32]| xs.iter().map(|x| to_bf16(*x)).collect::<Vec<u16>>();
    let mut k_pages = raw(&k0);
    k_pages.extend(raw(&k1));

    let built = Built {
        k_pages,
        keys: vec![vec![k0.to_vec(), k1.to_vec()]],
        q_raw: raw(&q_exact),
        q: q_exact.to_vec(),
    };

    let got = fire(&CASE, &built, 0, 1, 0, 1, None);

    // s_j = sm_scale * <q, k_j>, sm_scale = 1 / sqrt(4) = 0.5.
    let scale = 0.5f32;
    let s0 = scale * (1.0 * 0.5 + 0.5 * 1.0 + (-0.25) * 2.0 + 2.0 * (-0.5));
    let s1 = scale * (1.0 * (-1.0) + 0.5 * 0.25 + (-0.25) * 0.5 + 2.0 * 1.5);
    let top = s0.max(s1);
    let (e0, e1) = ((s0 - top).exp(), (s1 - top).exp());
    let want = [e0 / (e0 + e1), e1 / (e0 + e1)];

    for (j, expected) in want.iter().enumerate() {
        assert!(
            (got[j] - expected).abs() <= TOLERANCE,
            "key {j} landed {} and the hand arithmetic says {expected}",
            got[j]
        );
    }
    assert_eq!(got[2], 0.0, "the tail past kv_len is not zero");
    assert_eq!(got[3], 0.0, "the tail past kv_len is not zero");
}

/// (5): the two quantities the papers name. `observe = 1` on a one-row query
/// is the plain softmax over every key (TOVA); `observe = 4` over four rows
/// is the MEAN of the four rows' own causal distributions (SnapKV) — and
/// with `qo_len == kv_len == 4` those limits are 1, 2, 3 and 4, so the mean
/// is a shape a reader can check: key 0 carries mass from all four rows and
/// key 3 from one.
#[test]
#[ignore = "real-hardware: needs a CUDA device and `--features cuda-13`; run it with `-- --ignored`"]
fn one_observed_row_is_the_softmax_and_four_are_their_mean() {
    const ONE: Case = Case {
        what: "observe = 1 on a one-row query",
        page_size: 4,
        q_heads: 1,
        kv_heads: 1,
        head_dim: 64,
        kv_max: 8,
        observe: 1,
        hnd: false,
        lanes: &[(4, 1)],
    };
    const FOUR: Case = Case {
        what: "observe = 4 over four rows",
        page_size: 4,
        q_heads: 1,
        kv_heads: 1,
        head_dim: 64,
        kv_max: 8,
        observe: 4,
        hnd: false,
        lanes: &[(4, 4)],
    };

    // (a) one row, one softmax over all four keys.
    let built = build(&ONE, 41);
    let got = fire(&ONE, &built, 0, 1, 0, 1, None);
    let want = softmax_of_row(&ONE, &built, 0, 4);
    for j in 0..4 {
        assert!(
            (got[j] - want[j]).abs() <= TOLERANCE,
            "TOVA: key {j} landed {} and the plain softmax says {}",
            got[j],
            want[j]
        );
    }
    let sum: f32 = got[..4].iter().sum();
    assert!((sum - 1.0).abs() <= SUM_TOLERANCE, "TOVA: sums to {sum}");

    // (b) four rows, the mean of four distributions taken one at a time.
    let built = build(&FOUR, 43);
    let got = fire(&FOUR, &built, 0, 1, 0, 1, None);
    let mut want = [0.0f32; 8];
    for w in 0..4usize {
        // rows = 4, kv_len = 4, so row `w`'s causal limit is `w + 1`.
        let row = softmax_of_row(&FOUR, &built, w, w + 1);
        for (j, p) in row.iter().enumerate() {
            want[j] += p / 4.0;
        }
    }
    for j in 0..8 {
        assert!(
            (got[j] - want[j]).abs() <= TOLERANCE,
            "SnapKV: key {j} landed {} and the mean of four rows says {}",
            got[j],
            want[j]
        );
    }
    let sum: f32 = got[..4].iter().sum();
    assert!((sum - 1.0).abs() <= SUM_TOLERANCE, "SnapKV: sums to {sum}");
    assert!(
        want[0] > want[3],
        "the causal ladder puts more mass on key 0 than on key 3"
    );
}

/// Query row `w`'s own softmax over `[0, limit)`, head 0 of request 0.
fn softmax_of_row(case: &Case, built: &Built, w: usize, limit: usize) -> Vec<f32> {
    let head_dim = case.head_dim as usize;
    let q_row = &built.q[w * head_dim..][..head_dim];
    let scores: Vec<f32> = (0..limit)
        .map(|j| {
            let dot: f32 = q_row.iter().zip(&built.keys[0][j]).map(|(a, b)| a * b).sum();
            dot * case.sm_scale()
        })
        .collect();
    let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
    let total: f32 = weights.iter().sum();
    weights.iter().map(|w| w / total).collect()
}

/// (6): the refusals, by name. The sliding window leads because it is the
/// only SEMANTIC one — nothing is missing, the answer would just not be the
/// quantity the papers define.
#[test]
#[ignore = "real-hardware: needs a CUDA device and `--features cuda-13`; run it with `-- --ignored`"]
fn the_refusals_fire_by_name() {
    const CASE: Case = Case {
        what: "the refusal fixture",
        page_size: 4,
        q_heads: 2,
        kv_heads: 1,
        head_dim: 64,
        kv_max: 8,
        observe: 2,
        hnd: false,
        lanes: &[(4, 2)],
    };

    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let built = build(&CASE, 47);
    let q_at = gpu.up(&built.q_raw);
    let qo_at = gpu.up(&CASE.qo_indptr());
    let k_at = gpu.up(&built.k_pages);
    let indices_at = gpu.up(&CASE.page_indices());
    let indptr_at = gpu.up(&CASE.page_indptr());
    let lens_at = gpu.up(&CASE.last_page_lens());
    let scores_at = gpu.zeros(CASE.q_heads as usize * CASE.kv_max as usize * 4);

    let pool = pool_of(&CASE, k_at, indices_at, indptr_at, lens_at);
    let plan = plan_of(&CASE);
    let q = RaggedTensor {
        data: Tensor::new(
            q_at,
            CASE.q_rows() as u32,
            CASE.q_heads * CASE.head_dim,
            Dtype::Bf16,
        ),
        indptr: Tensor::new(qo_at, CASE.requests() as u32 + 1, 1, Dtype::I32),
    };
    let slab = |dtype: Dtype, width: u32| Tensor::new(scores_at, CASE.q_heads, width, dtype);

    let go = |q: RaggedTensor,
                  pool: KvPool,
                  window: Option<u32>,
                  head_dim: u32,
                  kv_heads: u32,
                  observe: u32,
                  kv_max: u32,
                  scores: &mut Tensor| {
        attn_score::capture(
            &ctx,
            q,
            &plan,
            &pool,
            window,
            head_dim,
            kv_heads,
            CASE.sm_scale(),
            observe,
            0,
            CASE.q_heads,
            0,
            kv_max,
            scores,
        )
    };

    let mut ok = slab(Dtype::F32, CASE.kv_max);
    assert!(
        go(q, pool, None, CASE.head_dim, CASE.kv_heads, CASE.observe, CASE.kv_max, &mut ok).is_ok(),
        "the fixture itself is admissible"
    );

    // THE SEMANTIC REFUSAL. Asked through a plan carved AT that window, so
    // `accepts` is satisfied and nothing but the meaning is left to refuse:
    // a windowed row is a distribution over the window, not over the
    // request's keys, and every key outside it would read as unattended
    // rather than as excluded.
    let windowed_plan = PrefillPlan {
        window: Some(4),
        ..plan_of(&CASE)
    };
    let windowed = attn_score::capture(
        &ctx,
        q,
        &windowed_plan,
        &pool,
        Some(4),
        CASE.head_dim,
        CASE.kv_heads,
        CASE.sm_scale(),
        CASE.observe,
        0,
        CASE.q_heads,
        0,
        CASE.kv_max,
        &mut ok,
    );
    assert!(
        format!("{:?}", windowed.expect_err("a sliding window is refused"))
            .contains("not the softmax the"),
        "a sliding window is refused by name, and for its meaning"
    );

    // And the plan's own guard still stands in front of it: a window stated
    // against an unwindowed schedule never reaches the semantic refusal.
    let unplanned = go(
        q,
        pool,
        Some(4),
        CASE.head_dim,
        CASE.kv_heads,
        CASE.observe,
        CASE.kv_max,
        &mut ok,
    );
    assert!(
        format!("{:?}", unplanned.expect_err("an unplanned window is refused"))
            .contains("carved its kv spans for"),
        "a window the schedule was not carved for is refused first"
    );

    let mut quantized = pool;
    quantized.keys = Tensor::new(k_at, pool.keys.rows, pool.keys.width, Dtype::Fp8E4m3);
    let dequantized = go(
        q,
        quantized,
        None,
        CASE.head_dim,
        CASE.kv_heads,
        CASE.observe,
        CASE.kv_max,
        &mut ok,
    );
    assert!(
        format!("{:?}", dequantized.expect_err("an fp8 pool is refused"))
            .contains("dequantizes nothing"),
        "a quantized pool is refused by name"
    );

    let mut wrong_dtype = slab(Dtype::Bf16, CASE.kv_max);
    let not_f32 = go(
        q,
        pool,
        None,
        CASE.head_dim,
        CASE.kv_heads,
        CASE.observe,
        CASE.kv_max,
        &mut wrong_dtype,
    );
    assert!(
        format!("{:?}", not_f32.expect_err("a bf16 slab is refused")).contains("f32 rectangle"),
        "a slab that is not F32 is refused by name"
    );

    let mut narrow = slab(Dtype::F32, CASE.kv_max - 1);
    let mismatched = go(
        q,
        pool,
        None,
        CASE.head_dim,
        CASE.kv_heads,
        CASE.observe,
        CASE.kv_max,
        &mut narrow,
    );
    assert!(
        format!("{:?}", mismatched.expect_err("a narrow slab is refused")).contains("the row IS"),
        "a slab row that is not the kv ceiling is refused by name"
    );

    let nothing = go(
        q,
        pool,
        None,
        CASE.head_dim,
        CASE.kv_heads,
        0,
        CASE.kv_max,
        &mut ok,
    );
    assert!(
        format!("{:?}", nothing.expect_err("observe = 0 is refused")).contains("observes nothing"),
        "an empty observation window is refused by name"
    );

    let ragged = RaggedTensor {
        data: Tensor::new(q_at, CASE.q_rows() as u32, CASE.q_heads * CASE.head_dim + 1, Dtype::Bf16),
        indptr: q.indptr,
    };
    let uneven = go(
        ragged,
        pool,
        None,
        CASE.head_dim,
        CASE.kv_heads,
        CASE.observe,
        CASE.kv_max,
        &mut ok,
    );
    assert!(
        format!("{:?}", uneven.expect_err("a ragged q row is refused")).contains("does not divide"),
        "a query row that is not a whole number of heads is refused by name"
    );

    let f32_q = RaggedTensor {
        data: Tensor::new(q_at, CASE.q_rows() as u32, CASE.q_heads * CASE.head_dim, Dtype::F32),
        indptr: q.indptr,
    };
    let wrong_element = go(
        f32_q,
        pool,
        None,
        CASE.head_dim,
        CASE.kv_heads,
        CASE.observe,
        CASE.kv_max,
        &mut ok,
    );
    assert!(
        wrong_element.is_err(),
        "a query that is not bf16 is refused by dtype"
    );
}

/// **BOTH PAGE LAYOUTS, ONE ANSWER.** Every pool `engine-cuda` binds is NHD
/// (`store.rs` states the enumerator and the stride pair together, and
/// crosschecks them), but the entry reads `pool.layout` and stamps either
/// arm of `kv_dst_index` — so the HND arm has to be walked here or it is
/// text nobody ever compiles. The same keys, laid out both ways, must give
/// the same distribution.
#[test]
#[ignore = "real-hardware: needs a CUDA device and `--features cuda-13`; run it with `-- --ignored`"]
fn the_hnd_arm_reads_the_same_keys_the_nhd_arm_does() {
    const NHD: Case = Case {
        what: "the pages this tree actually binds",
        page_size: 4,
        q_heads: 4,
        kv_heads: 2,
        head_dim: 64,
        kv_max: 16,
        observe: 2,
        hnd: false,
        lanes: &[(6, 3)],
    };
    const HND: Case = Case {
        what: "the same cache, laid out `[page][head][token][dim]`",
        hnd: true,
        ..NHD
    };

    let nhd = build(&NHD, 53);
    let hnd = build(&HND, 53);
    assert_eq!(nhd.keys, hnd.keys, "both builders wrote the same keys");
    assert_ne!(
        nhd.k_pages, hnd.k_pages,
        "the two layouts put those keys in different places"
    );

    let from_nhd = fire(&NHD, &nhd, 0, NHD.q_heads, 0, 1, None);
    let from_hnd = fire(&HND, &hnd, 0, HND.q_heads, 0, 1, None);
    let want = reference(&NHD, &nhd);

    for head in 0..NHD.q_heads as usize {
        for j in 0..NHD.kv_max as usize {
            let at = head * NHD.kv_max as usize + j;
            assert!(
                (from_hnd[at] - want[head][j]).abs() <= TOLERANCE,
                "HND: head {head} key {j} landed {} and the reference says {}",
                from_hnd[at],
                want[head][j]
            );
            assert_eq!(
                from_nhd[at], from_hnd[at],
                "head {head} key {j} differs between the two layouts"
            );
        }
    }
}
