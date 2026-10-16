#![cfg(feature = "cuda")]

//! GLM-5.3's DSA attention (`kernel.sdpa.mla_sparse`, `attn/mla_sparse.cuh`):
//! 64 absorbed heads over a 512-wide latent and a 64-wide rotated part, the
//! token ids a row's index selected, a block a row on tcgen05. Read against a host reference, scores and base-2
//! log-sum-exp, over two requests (one continuing a context), shuffled
//! pages, selections longer than a tile with whole page runs in them, a row
//! that selects nothing visible; on a staged window and on a pooled index's
//! selection, against the warp-a-key form over the same rows. Two passes
//! land the same bits.

use crate::common::{Gpu, Lcg, close, from_bf16};
use dtype::Dtype;
use kernels_cuda::attn::mla;
use kernels_cuda::attn::plan::{self, Device, Live, MlaPlan, Workspace};
use kernels_cuda::tensor::{KvPool, RaggedTensor, Tensor};
use kernels_cuda::tuning::Tuning;

const HEADS: usize = 64;
const RANK: usize = 512;
const PAGE: usize = 16;
const Q_LENS: [usize; 2] = [40, 21];
const KV_LENS: [usize; 2] = [157, 21];
const PAGE_INDICES: [i32; 12] = [3, 0, 5, 1, 6, 2, 4, 9, 11, 8, 7, 10];
const PAGE_INDPTR: [i32; 3] = [0, 10, 12];
const LAST_PAGE_LENS: [i32; 2] = [13, 5];
/// A row's selection width: up to 96 ids with -1 holes among them (as a
/// pooled index's dropped blocks leave), not a whole number of tiles.
const TOP_K: usize = 131;
const PICKS: usize = 96;
const INT_BYTES: usize = 4 << 20;
const FLOAT_BYTES: usize = 64 << 20;

struct Case {
    rope: usize,
    pooled: bool,
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

    fn slot(request: usize, position: usize) -> usize {
        let page = PAGE_INDICES[PAGE_INDPTR[request] as usize + position / PAGE] as usize;
        page * PAGE + position % PAGE
    }

    /// The first 40 keys whole (whole pages among them), then every other,
    /// at most `PICKS`; the very first row of request 1 selects nothing.
    fn picked(request: usize, abs: usize) -> Vec<usize> {
        if request == 1 && abs == 0 {
            return Vec::new();
        }
        (0..=abs)
            .filter(|&j| j < 40 || j % 2 == 0)
            .take(PICKS)
            .collect()
    }

    fn selection(&self) -> Vec<i32> {
        if self.pooled {
            return self.pooled_selection();
        }
        let mut out = vec![-1i32; self.tokens() * TOP_K];
        for r in 0..Q_LENS.len() {
            for i in 0..Q_LENS[r] {
                let t = self.qo_indptr[r] as usize + i;
                let abs = KV_LENS[r] - Q_LENS[r] + i;
                let mut n = 0;
                for (i, j) in Self::picked(r, abs).into_iter().enumerate() {
                    if i % 5 == 2 {
                        n += 1; // a hole
                    }
                    out[t * TOP_K + n] = j as i32;
                    n += 1;
                }
            }
        }
        out
    }

    /// A pooled index's: blocks of four tokens a row ranks, expanded whole
    /// (some of a block past the row), -1 for a dropped block, then the
    /// row's last three tokens, which may repeat a block's.
    fn pooled_selection(&self) -> Vec<i32> {
        let mut out = vec![-1i32; self.tokens() * TOP_K];
        for r in 0..Q_LENS.len() {
            for i in 0..Q_LENS[r] {
                let t = self.qo_indptr[r] as usize + i;
                let abs = KV_LENS[r] - Q_LENS[r] + i;
                let blocks = (abs + 1).div_ceil(4).min((TOP_K - 3) / 4);
                for b in 0..blocks {
                    if b % 3 == 1 {
                        continue; // dropped
                    }
                    for k in 0..4 {
                        out[t * TOP_K + 4 * b + k] = (4 * b + k) as i32;
                    }
                }
                for k in 0..3 {
                    if abs >= k {
                        out[t * TOP_K + TOP_K - 3 + k] = (abs - k) as i32;
                    }
                }
            }
        }
        out
    }

    fn reference(&self) -> (Vec<f32>, Vec<f32>) {
        let tokens = self.tokens();
        let rope = self.rope;
        let mut out = vec![0f32; tokens * HEADS * RANK];
        let mut lse = vec![f32::NEG_INFINITY; tokens * HEADS];
        for r in 0..Q_LENS.len() {
            let q0 = self.qo_indptr[r] as usize;
            for i in 0..Q_LENS[r] {
                let t = q0 + i;
                let abs = KV_LENS[r] - Q_LENS[r] + i;
                let keys = Self::picked(r, abs);
                if keys.is_empty() {
                    continue;
                }
                for h in 0..HEADS {
                    let qn = &self.q_nope[(t * HEADS + h) * RANK..][..RANK];
                    let qp = &self.q_pe[(t * HEADS + h) * rope..][..rope];
                    let mut scores: Vec<f64> = keys
                        .iter()
                        .map(|&j| {
                            let slot = Self::slot(r, j);
                            let ckv = &self.ckv[slot * RANK..][..RANK];
                            let kpe = &self.kpe[slot * rope..][..rope];
                            let nope: f64 = qn.iter().zip(ckv).map(|(a, b)| f64::from(a * b)).sum();
                            let pe: f64 = qp.iter().zip(kpe).map(|(a, b)| f64::from(a * b)).sum();
                            (nope + pe) * f64::from(self.sm_scale)
                        })
                        .collect();
                    let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let mut sum = 0f64;
                    for s in &mut scores {
                        *s = (*s - m).exp();
                        sum += *s;
                    }
                    lse[t * HEADS + h] = ((m + sum.ln()) * std::f64::consts::LOG2_E) as f32;
                    let o = &mut out[(t * HEADS + h) * RANK..][..RANK];
                    for (&j, p) in keys.iter().zip(&scores) {
                        let ckv = &self.ckv[Self::slot(r, j) * RANK..][..RANK];
                        for (d, x) in o.iter_mut().zip(ckv) {
                            *d += (p / sum) as f32 * x;
                        }
                    }
                }
            }
        }
        (out, lse)
    }
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

/// Scores and log-sum-exp, `tc` choosing the tcgen05 pass; `window` is
/// `(live, base)` over the rows.
fn score(
    rope: usize,
    tc: bool,
    window: Option<(u32, u32)>,
    decode: bool,
    pooled: bool,
) -> ((Vec<u16>, Vec<f32>), Case) {
    let mut rng = Lcg::seeded(0x6c73_6533 + rope as u64);
    let tokens: usize = Q_LENS.iter().sum();
    let slots = PAGE_INDICES.len() * PAGE;
    let (q_nope_raw, q_nope) = rng.row(tokens * HEADS * RANK);
    let (q_pe_raw, q_pe) = rng.row(tokens * HEADS * rope.max(1));
    let (ckv_raw, ckv) = rng.row(slots * RANK);
    let (kpe_raw, kpe) = rng.row(slots * rope.max(1));
    let mut qo_indptr = vec![0i32];
    for q in Q_LENS {
        qo_indptr.push(qo_indptr.last().unwrap() + q as i32);
    }
    #[allow(clippy::cast_precision_loss)]
    let case = Case {
        rope,
        pooled,
        q_nope,
        q_pe,
        ckv,
        kpe,
        qo_indptr,
        sm_scale: 1.0 / ((192 + rope) as f32).sqrt(),
    };
    let mut gpu = Gpu::open();
    let device = Device::probe(&gpu.ctx()).expect("the device answers its attributes");
    let plan = plan(&mut gpu, &case, &device);
    let ctx = gpu.ctx().with_tuning(Tuning {
        mla_naive: true,
        mla_sparse: tc,
        ..Tuning::DEFAULT
    });
    if let Some((live, base)) = window {
        let win_at = gpu.up(&[live, base, Q_LENS.len() as u32, 0u32]);
        ctx.arm_stage(win_at);
    }
    let pool = KvPool {
        keys: Tensor::new(gpu.up(&ckv_raw), slots as u32, RANK as u32, Dtype::Bf16),
        values: Tensor::new(gpu.up(&kpe_raw), slots as u32, rope as u32, Dtype::Bf16),
        bf16_keys: Tensor::ABSENT,
        bf16_values: Tensor::ABSENT,
        key_scales: Tensor::ABSENT,
        value_scales: Tensor::ABSENT,
        page_indices: Tensor::new(
            gpu.up(&PAGE_INDICES),
            PAGE_INDICES.len() as u32,
            1,
            Dtype::I32,
        ),
        page_indptr: Tensor::new(
            gpu.up(&PAGE_INDPTR),
            PAGE_INDPTR.len() as u32,
            1,
            Dtype::I32,
        ),
        last_page_lens: Tensor::new(gpu.up(&LAST_PAGE_LENS), 2, 1, Dtype::I32),
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
        max_pages_per_request: 10,
        pages_in_batch: PAGE_INDICES.len() as i32,
    };
    let q = RaggedTensor {
        data: Tensor::new(
            gpu.up(&q_nope_raw),
            tokens as u32,
            (HEADS * RANK) as u32,
            Dtype::Bf16,
        ),
        indptr: Tensor::new(gpu.up(&case.qo_indptr), 3, 1, Dtype::I32),
    };
    let q_pe_t = Tensor::new(
        gpu.up(&q_pe_raw),
        tokens as u32,
        (HEADS * rope) as u32,
        Dtype::Bf16,
    );
    let selection = Tensor::new(
        gpu.up(&case.selection()),
        tokens as u32,
        TOP_K as u32,
        Dtype::I32,
    );
    let out = gpu.up(&vec![0x7fc0u16; tokens * HEADS * RANK]);
    let lse_at = gpu.up(&vec![7.0f32; tokens * HEADS]);
    let mut o = Tensor::new(out, tokens as u32, (HEADS * RANK) as u32, Dtype::Bf16);
    let mut lse = Tensor::new(lse_at, tokens as u32, HEADS as u32, Dtype::F32);
    let arm = if decode {
        mla::attention_decode_selected
    } else {
        mla::attention_prefill_selected
    };
    arm(
        &ctx,
        q,
        &plan,
        q_pe_t,
        selection,
        &pool,
        HEADS as u32,
        RANK as u32,
        case.sm_scale,
        &mut o,
        Some(&mut lse),
    )
    .expect("the selected attention enqueues");
    gpu.sync();
    (
        (
            gpu.down::<u16>(out, tokens * HEADS * RANK),
            gpu.down::<f32>(lse_at, tokens * HEADS),
        ),
        case,
    )
}

fn check(rope: usize, window: Option<(u32, u32)>, decode: bool, pooled: bool) {
    let (got, case) = score(rope, true, window, decode, pooled);
    let (again, _) = score(rope, true, window, decode, pooled);
    assert!(
        got.0 == again.0
            && got
                .1
                .iter()
                .zip(&again.1)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
        "rope {rope} {window:?} decode {decode} pooled {pooled}: two passes land apart"
    );
    // On a window the warp-a-key form is the reference, over the same rows.
    let want = match window {
        None if !pooled => case.reference(),
        _ => {
            let (naive, _) = score(rope, false, window, decode, pooled);
            (naive.0.iter().map(|&b| from_bf16(b)).collect(), naive.1)
        }
    };
    let (live, base) = window.map_or((case.tokens(), 0), |(l, b)| (l as usize, b as usize));
    for (i, &g) in got.0.iter().enumerate() {
        let row = i / (HEADS * RANK);
        let g = from_bf16(g);
        if row < base || row >= base + live {
            assert!(
                g.is_nan(),
                "rope {rope}: row {row} past the window was written"
            );
            continue;
        }
        let w = want.0[i];
        assert!(
            close(g, w),
            "rope {rope} {window:?}: row {row} head {} dim {} scored {g} against {w}",
            (i / RANK) % HEADS,
            i % RANK
        );
    }
    for (i, (&g, &w)) in got.1.iter().zip(&want.1).enumerate() {
        let row = i / HEADS;
        if row < base || row >= base + live {
            continue;
        }
        assert!(
            (g == w) || (g - w).abs() <= 1.0e-2 * w.abs().max(1.0),
            "rope {rope} {window:?}: row {row} head {} landed log-sum-exp {g} against {w}",
            i % HEADS
        );
    }
}

#[test]
fn the_sparse_mla_lands_on_tcgen05() {
    check(64, None, false, false);
    // A pooled index's selection (holes, repeats, ids past the row),
    // against the warp-a-key form.
    check(64, None, false, true);
    check(64, None, true, true);
    check(64, Some((30, 25)), false, false);
    // The decode arm sees the request's keys, not the row's own prefix: the
    // selections here read none past a row's position, so both answer alike.
    check(64, None, true, false);
}

/// One 4096-row prefill selecting up to 2048 tokens a row, both forms.
#[test]
#[ignore = "pacing; run with --ignored --nocapture"]
fn the_sparse_mla_paces_a_prefill() {
    const ROWS: usize = 4096;
    const K: usize = 2048;
    const PG: usize = 64;
    let pages = ROWS / PG;
    let mut gpu = Gpu::open();
    let page_ids: Vec<i32> = (0..pages as i32).rev().collect();
    let mut sel = vec![-1i32; ROWS * K];
    for t in 0..ROWS {
        let lo = (t + 1).saturating_sub(K);
        for (n, j) in (lo..=t).enumerate() {
            sel[t * K + n] = j as i32;
        }
    }
    let ckv = gpu.zeros(ROWS * RANK * 2);
    let kpe = gpu.zeros(ROWS * 64 * 2);
    let pool = KvPool {
        keys: Tensor::new(ckv, ROWS as u32, RANK as u32, Dtype::Bf16),
        values: Tensor::new(kpe, ROWS as u32, 64, Dtype::Bf16),
        bf16_keys: Tensor::ABSENT,
        bf16_values: Tensor::ABSENT,
        key_scales: Tensor::ABSENT,
        value_scales: Tensor::ABSENT,
        page_indices: Tensor::new(gpu.up(&page_ids), pages as u32, 1, Dtype::I32),
        page_indptr: Tensor::new(gpu.up(&[0i32, pages as i32]), 2, 1, Dtype::I32),
        last_page_lens: Tensor::new(gpu.up(&[PG as i32]), 1, 1, Dtype::I32),
        row_valid: Tensor::ABSENT,
        env_min: Tensor::ABSENT,
        env_max: Tensor::ABSENT,
        has_envelopes: false,
        page_size: PG as i32,
        seq_stride: RANK as i64,
        head_stride: RANK as i64,
        layout: 0,
        scheme_byte: 0,
        block_size: 0,
        max_pages_per_request: pages as i32,
        pages_in_batch: pages as i32,
    };
    let qo = [0i32, ROWS as i32];
    let case = Case {
        rope: 64,
        pooled: false,
        q_nope: Vec::new(),
        q_pe: Vec::new(),
        ckv: Vec::new(),
        kpe: Vec::new(),
        qo_indptr: qo.to_vec(),
        sm_scale: 0.05,
    };
    let device = Device::probe(&gpu.ctx()).expect("the device answers its attributes");
    let workspace = Workspace {
        int_ptr: gpu.zeros(INT_BYTES),
        int_bytes: INT_BYTES,
        float_ptr: gpu.zeros(FLOAT_BYTES),
        float_bytes: FLOAT_BYTES,
    };
    let plan = plan::plan_mla(
        &case.qo_indptr,
        &[0, pages as i32],
        &[ROWS as i32],
        ROWS as u32,
        1,
        Live {
            requests: 1,
            lane_offset: 0,
            row_offset: 0,
            rows: ROWS as u32,
        },
        HEADS as u32,
        RANK as u32,
        true,
        &device,
        workspace,
    )
    .expect("the plan schedules");
    let q = gpu.zeros(ROWS * HEADS * RANK * 2);
    let q_pe = gpu.zeros(ROWS * HEADS * 64 * 2);
    let qo_at = gpu.up(&qo);
    let sel_at = gpu.up(&sel);
    let out = gpu.zeros(ROWS * HEADS * RANK * 2);
    for tc in [false, true] {
        let ctx = gpu.ctx().with_tuning(Tuning {
            mla_sparse: tc,
            ..Tuning::DEFAULT
        });
        let fire = || {
            mla::attention_prefill_selected(
                &ctx,
                RaggedTensor {
                    data: Tensor::new(q, ROWS as u32, (HEADS * RANK) as u32, Dtype::Bf16),
                    indptr: Tensor::new(qo_at, 2, 1, Dtype::I32),
                },
                &plan,
                Tensor::new(q_pe, ROWS as u32, (HEADS * 64) as u32, Dtype::Bf16),
                Tensor::new(sel_at, ROWS as u32, K as u32, Dtype::I32),
                &pool,
                HEADS as u32,
                RANK as u32,
                0.05,
                &mut Tensor::new(out, ROWS as u32, (HEADS * RANK) as u32, Dtype::Bf16),
                None,
            )
            .unwrap();
        };
        fire();
        gpu.sync();
        let iters = if tc { 10 } else { 1 };
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            fire();
        }
        gpu.sync();
        let us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(iters);
        eprintln!("| {ROWS} rows, top {K} | tcgen05 {tc:5} | {us:9.0} us |");
    }
}
