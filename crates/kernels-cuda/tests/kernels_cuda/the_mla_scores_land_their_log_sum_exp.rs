#![cfg(feature = "cuda")]

// A latent attention asked for its log-sum-exp lands, beside its scores, each
// row's per-head base-2 log of the softmax denominator a sink rescales by.
// Every arm is read against a host reference: the FA2 arm the plan takes, the
// tensor-core and the naive arm (`kernel.mla.path = naive`), the naive arm
// reading a selection, and the naive arm a window sends any read to.

use crate::common::{Gpu, Lcg, close, from_bf16};
use dtype::Dtype;
use kernels_cuda::attn::mla;
use kernels_cuda::attn::plan::{self, Device, Live, MlaPlan, Workspace};
use kernels_cuda::tensor::{KvPool, RaggedTensor, Tensor};
use kernels_cuda::tuning::Tuning;

const HEADS: usize = 16;
const RANK: usize = 512;
const ROPE: usize = 64;
const PAGE: usize = 16;

// One request continues a 37-token context with 40 new rows; the other is a
// fresh 21-row prefill.
const Q_LENS: [usize; 2] = [40, 21];
const KV_LENS: [usize; 2] = [77, 21];
const PAGE_INDICES: [i32; 7] = [3, 0, 5, 1, 6, 2, 4];
const PAGE_INDPTR: [i32; 3] = [0, 5, 7];
const LAST_PAGE_LENS: [i32; 2] = [13, 5];
// The selection reads every other key of the causal prefix.
const TOP_K: usize = 48;

const INT_BYTES: usize = 4 << 20;
const FLOAT_BYTES: usize = 64 << 20;

struct Case {
    q_nope: Vec<f32>,
    q_pe: Vec<f32>,
    ckv: Vec<f32>,
    kpe: Vec<f32>,
    qo_indptr: Vec<i32>,
    sm_scale: f32,
}

impl Case {
    fn tokens(&self) -> usize {
        Q_LENS.iter().sum()
    }

    fn slot(&self, request: usize, position: usize) -> usize {
        let page = PAGE_INDICES[PAGE_INDPTR[request] as usize + position / PAGE] as usize;
        page * PAGE + position % PAGE
    }

    fn picked(abs: usize) -> Vec<usize> {
        (0..=abs).filter(|j| j % 2 == 0).take(TOP_K).collect()
    }

    fn selection(&self) -> Vec<i32> {
        let mut out = vec![-1i32; self.tokens() * TOP_K];
        for r in 0..Q_LENS.len() {
            for i in 0..Q_LENS[r] {
                let t = self.qo_indptr[r] as usize + i;
                let abs = KV_LENS[r] - Q_LENS[r] + i;
                for (n, j) in Self::picked(abs).into_iter().enumerate() {
                    out[t * TOP_K + n] = j as i32;
                }
            }
        }
        out
    }

    /// The scores and the base-2 log-sum-exp, over every causal key (the
    /// last `window` of them, given one) or the selection.
    fn reference(&self, selected: bool, window: Option<usize>) -> (Vec<f32>, Vec<f32>) {
        let tokens = self.tokens();
        let mut out = vec![0f32; tokens * HEADS * RANK];
        let mut lse = vec![0f32; tokens * HEADS];
        for r in 0..Q_LENS.len() {
            let q0 = self.qo_indptr[r] as usize;
            for i in 0..Q_LENS[r] {
                let t = q0 + i;
                let abs = KV_LENS[r] - Q_LENS[r] + i;
                let keys: Vec<usize> = if selected {
                    Self::picked(abs)
                } else {
                    ((abs + 1).saturating_sub(window.unwrap_or(abs + 1))..=abs).collect()
                };
                for h in 0..HEADS {
                    let qn = &self.q_nope[(t * HEADS + h) * RANK..][..RANK];
                    let qp = &self.q_pe[(t * HEADS + h) * ROPE..][..ROPE];
                    let mut scores: Vec<f32> = keys
                        .iter()
                        .map(|&j| {
                            let slot = self.slot(r, j);
                            let ckv = &self.ckv[slot * RANK..][..RANK];
                            let kpe = &self.kpe[slot * ROPE..][..ROPE];
                            let nope: f32 = qn.iter().zip(ckv).map(|(a, b)| a * b).sum();
                            let rope: f32 = qp.iter().zip(kpe).map(|(a, b)| a * b).sum();
                            (nope + rope) * self.sm_scale
                        })
                        .collect();
                    let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0f32;
                    for s in &mut scores {
                        *s = (*s - m).exp();
                        sum += *s;
                    }
                    lse[t * HEADS + h] = (m + sum.ln()) * std::f32::consts::LOG2_E;
                    let o = &mut out[(t * HEADS + h) * RANK..][..RANK];
                    for (&j, p) in keys.iter().zip(&scores) {
                        let ckv = &self.ckv[self.slot(r, j) * RANK..][..RANK];
                        for (d, x) in o.iter_mut().zip(ckv) {
                            *d += p / sum * x;
                        }
                    }
                }
            }
        }
        (out, lse)
    }
}

struct Raw {
    q_nope: u64,
    q_pe: u64,
    ckv: u64,
    kpe: u64,
    qo_indptr: u64,
    page_indices: u64,
    page_indptr: u64,
    last_page_lens: u64,
    selection: u64,
}

fn case() -> (Case, [Vec<u16>; 4]) {
    let mut rng = Lcg::seeded(0x6c73_6532);
    let tokens: usize = Q_LENS.iter().sum();
    let slots = PAGE_INDICES.len() * PAGE;
    let (q_nope_raw, q_nope) = rng.row(tokens * HEADS * RANK);
    let (q_pe_raw, q_pe) = rng.row(tokens * HEADS * ROPE);
    let (ckv_raw, ckv) = rng.row(slots * RANK);
    let (kpe_raw, kpe) = rng.row(slots * ROPE);
    let mut qo_indptr = vec![0i32];
    for q in Q_LENS {
        qo_indptr.push(qo_indptr.last().unwrap() + q as i32);
    }
    #[allow(clippy::cast_precision_loss)]
    let sm_scale = 1.0 / ((192 + ROPE) as f32).sqrt();
    (
        Case {
            q_nope,
            q_pe,
            ckv,
            kpe,
            qo_indptr,
            sm_scale,
        },
        [q_nope_raw, q_pe_raw, ckv_raw, kpe_raw],
    )
}

fn plan(gpu: &mut Gpu, case: &Case, device: &Device) -> MlaPlan {
    let tokens = case.tokens() as u32;
    let workspace = Workspace {
        int_ptr: gpu.zeros(INT_BYTES),
        int_bytes: INT_BYTES,
        float_ptr: gpu.zeros(FLOAT_BYTES),
        float_bytes: FLOAT_BYTES,
    };
    let kv_len: Vec<i32> = KV_LENS.iter().map(|&n| n as i32).collect();
    let plan = plan::plan_mla(
        &case.qo_indptr,
        &PAGE_INDPTR,
        &kv_len,
        tokens,
        Q_LENS.len() as u32,
        Live {
            requests: Q_LENS.len() as u32,
            lane_offset: 0,
            row_offset: 0,
            rows: tokens,
        },
        HEADS as u32,
        RANK as u32,
        true,
        device,
        workspace,
    )
    .expect("the mla plan schedules two requests");
    plan.stage(&gpu.ctx())
        .expect("the plan stages its int workspace");
    plan
}

fn pool(raw: &Raw) -> KvPool {
    let slots = (PAGE_INDICES.len() * PAGE) as u32;
    KvPool {
        keys: Tensor::new(raw.ckv, slots, RANK as u32, Dtype::Bf16),
        values: Tensor::new(raw.kpe, slots, ROPE as u32, Dtype::Bf16),
        bf16_keys: Tensor::ABSENT,
        bf16_values: Tensor::ABSENT,
        key_scales: Tensor::ABSENT,
        value_scales: Tensor::ABSENT,
        page_indices: Tensor::new(raw.page_indices, PAGE_INDICES.len() as u32, 1, Dtype::I32),
        page_indptr: Tensor::new(raw.page_indptr, PAGE_INDPTR.len() as u32, 1, Dtype::I32),
        last_page_lens: Tensor::new(
            raw.last_page_lens,
            LAST_PAGE_LENS.len() as u32,
            1,
            Dtype::I32,
        ),
        row_valid: Tensor::ABSENT,
        env_min: Tensor::ABSENT,
        env_max: Tensor::ABSENT,
        has_envelopes: false,
        page_size: PAGE as i32,
        seq_stride: RANK as i64,
        head_stride: RANK as i64,
        layout: 0,
        scheme_byte: 0,
        block_size: 0,
        max_pages_per_request: 5,
        pages_in_batch: PAGE_INDICES.len() as i32,
    }
}

/// The scores and the log-sum-exp one arm lands: the selected naive arm, or
/// the dense arm `naive` picks.
fn score(
    gpu: &mut Gpu,
    case: &Case,
    raw: &Raw,
    naive: bool,
    selected: bool,
    window: Option<u32>,
) -> (Vec<u16>, Vec<f32>) {
    let device = Device::probe(&gpu.ctx()).expect("the device answers its attributes");
    let plan = plan(gpu, case, &device);
    let tokens = case.tokens() as u32;
    let ctx = gpu.ctx().with_tuning(Tuning {
        mla_naive: naive,
        ..Tuning::DEFAULT
    });
    let pool = pool(raw);
    let q = RaggedTensor {
        data: Tensor::new(raw.q_nope, tokens, (HEADS * RANK) as u32, Dtype::Bf16),
        indptr: Tensor::new(raw.qo_indptr, case.qo_indptr.len() as u32, 1, Dtype::I32),
    };
    let q_pe = Tensor::new(raw.q_pe, tokens, (HEADS * ROPE) as u32, Dtype::Bf16);
    let out = gpu.zeros(case.tokens() * HEADS * RANK * 2);
    let lse_at = gpu.zeros(case.tokens() * HEADS * 4);
    let mut o = Tensor::new(out, tokens, (HEADS * RANK) as u32, Dtype::Bf16);
    let mut lse = Tensor::new(lse_at, tokens, HEADS as u32, Dtype::F32);
    if selected {
        let selection = Tensor::new(raw.selection, tokens, TOP_K as u32, Dtype::I32);
        mla::attention_prefill_selected(
            &ctx,
            q,
            &plan,
            q_pe,
            selection,
            &pool,
            HEADS as u32,
            RANK as u32,
            case.sm_scale,
            &mut o,
            Some(&mut lse),
        )
        .expect("attention.mla_prefill_selected enqueues");
    } else {
        mla::attention_prefill(
            &ctx,
            q,
            &plan,
            q_pe,
            &pool,
            HEADS as u32,
            RANK as u32,
            case.sm_scale,
            window,
            &mut o,
            Some(&mut lse),
        )
        .expect("attention.mla_prefill enqueues");
    }
    gpu.sync();
    (
        gpu.down::<u16>(out, case.tokens() * HEADS * RANK),
        gpu.down::<f32>(lse_at, case.tokens() * HEADS),
    )
}

fn assert_landed(got: &(Vec<u16>, Vec<f32>), want: &(Vec<f32>, Vec<f32>), what: &str) {
    for (i, (&g, &w)) in got.0.iter().zip(&want.0).enumerate() {
        let g = from_bf16(g);
        assert!(
            close(g, w),
            "{what}: row {} head {} dim {} scored {g} against {w}",
            i / (HEADS * RANK),
            (i / RANK) % HEADS,
            i % RANK
        );
    }
    let mut worst = 0f32;
    for (i, (&g, &w)) in got.1.iter().zip(&want.1).enumerate() {
        assert!(
            (g - w).abs() <= 1.0e-2 * w.abs().max(1.0),
            "{what}: row {} head {} landed log-sum-exp {g} against {w}",
            i / HEADS,
            i % HEADS
        );
        worst = worst.max((g - w).abs());
    }
    eprintln!("{what}: worst log-sum-exp error {worst}");
}

#[test]
fn the_mla_scores_land_their_log_sum_exp() {
    let mut gpu = Gpu::open();
    let (case, [q_nope, q_pe, ckv, kpe]) = case();
    let raw = Raw {
        q_nope: gpu.up(&q_nope),
        q_pe: gpu.up(&q_pe),
        ckv: gpu.up(&ckv),
        kpe: gpu.up(&kpe),
        qo_indptr: gpu.up(&case.qo_indptr),
        page_indices: gpu.up(&PAGE_INDICES),
        page_indptr: gpu.up(&PAGE_INDPTR),
        last_page_lens: gpu.up(&LAST_PAGE_LENS),
        selection: gpu.up(&case.selection()),
    };
    let dense = case.reference(false, None);
    let fa2 = score(&mut gpu, &case, &raw, false, false, None);
    assert_landed(&fa2, &dense, "the FA2 arm");
    let mma = score(&mut gpu, &case, &raw, true, false, None);
    assert_landed(&mma, &dense, "the tensor-core arm");
    let picked = score(&mut gpu, &case, &raw, true, true, None);
    assert_landed(
        &picked,
        &case.reference(true, None),
        "the selected naive arm",
    );
    let windowed = score(&mut gpu, &case, &raw, false, false, Some(24));
    assert_landed(
        &windowed,
        &case.reference(false, Some(24)),
        "the windowed naive arm",
    );
}
