#![cfg(feature = "cuda")]

// The FA2 MLA kernel's smallest `DISPATCH_SMEM_CONFIG` arm (`CTA_TILE_KV = 16`)
// is the only one that fits an Ada-class 99 KiB block. A plan scheduled for the
// L40S ceiling takes that arm on any device, so this drives it on the local
// GPU and reads it against a host reference and against the plan the probed
// device would take.

mod common;

use common::{Gpu, Lcg, close, from_bf16};
use dtype::Dtype;
use kernels_cuda::attn::mla;
use kernels_cuda::attn::plan::{self, Device, Live, Workspace};
use kernels_cuda::tensor::{KvPool, RaggedTensor, Tensor};

const HEADS: usize = 16;
const RANK: usize = 512;
const ROPE: usize = 64;
const PAGE: usize = 16;

// One request continues a 37-token context with 40 new rows; the other is a
// fresh 21-row prefill. The page table is deliberately out of order.
const Q_LENS: [usize; 2] = [40, 21];
const KV_LENS: [usize; 2] = [77, 21];
const PAGE_INDICES: [i32; 7] = [3, 0, 5, 1, 6, 2, 4];
const PAGE_INDPTR: [i32; 3] = [0, 5, 7];
const LAST_PAGE_LENS: [i32; 2] = [13, 5];

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

    fn reference(&self) -> Vec<f32> {
        let tokens = self.tokens();
        let mut out = vec![0f32; tokens * HEADS * RANK];
        for r in 0..Q_LENS.len() {
            let q0 = self.qo_indptr[r] as usize;
            for i in 0..Q_LENS[r] {
                let t = q0 + i;
                let abs = KV_LENS[r] - Q_LENS[r] + i;
                for h in 0..HEADS {
                    let qn = &self.q_nope[(t * HEADS + h) * RANK..][..RANK];
                    let qp = &self.q_pe[(t * HEADS + h) * ROPE..][..ROPE];
                    let mut scores: Vec<f32> = (0..=abs)
                        .map(|j| {
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
                    let o = &mut out[(t * HEADS + h) * RANK..][..RANK];
                    for (j, p) in scores.iter().enumerate() {
                        let ckv = &self.ckv[self.slot(r, j) * RANK..][..RANK];
                        for (d, x) in o.iter_mut().zip(ckv) {
                            *d += p / sum * x;
                        }
                    }
                }
            }
        }
        out
    }
}

fn case() -> (Case, Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
    let mut rng = Lcg::seeded(0x4b69_6d69);
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
    let sm_scale = 1.0 / ((128 + ROPE) as f32).sqrt();
    (
        Case {
            q_nope,
            q_pe,
            ckv,
            kpe,
            qo_indptr,
            sm_scale,
        },
        q_nope_raw,
        q_pe_raw,
        ckv_raw,
        kpe_raw,
    )
}

fn prefill(gpu: &mut Gpu, case: &Case, raw: &Raw, device: &Device) -> Vec<u16> {
    let ctx = gpu.ctx();
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
    plan.stage(&ctx).expect("the plan stages its int workspace");

    let slots = (PAGE_INDICES.len() * PAGE) as u32;
    let pool = KvPool {
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
    };
    let q = RaggedTensor {
        data: Tensor::new(raw.q_nope, tokens, (HEADS * RANK) as u32, Dtype::Bf16),
        indptr: Tensor::new(raw.qo_indptr, case.qo_indptr.len() as u32, 1, Dtype::I32),
    };
    let q_pe = Tensor::new(raw.q_pe, tokens, (HEADS * ROPE) as u32, Dtype::Bf16);
    let out = gpu.zeros(case.tokens() * HEADS * RANK * 2);
    let mut o = Tensor::new(out, tokens, (HEADS * RANK) as u32, Dtype::Bf16);
    mla::attention_prefill(
        &ctx,
        q,
        &plan,
        q_pe,
        &pool,
        HEADS as u32,
        RANK as u32,
        case.sm_scale,
        &mut o,
    )
    .expect("attention.mla_prefill enqueues");
    gpu.sync();
    gpu.down::<u16>(out, case.tokens() * HEADS * RANK)
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
}

fn assert_close(got: &[u16], want: &[f32], what: &str) {
    let mut worst = 0f32;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let g = from_bf16(g);
        assert!(
            close(g, w),
            "{what}: row {} head {} dim {} landed {g} against {w}",
            i / (HEADS * RANK),
            (i / RANK) % HEADS,
            i % RANK
        );
        worst = worst.max((g - w).abs());
    }
    eprintln!("{what}: worst absolute error {worst}");
}

#[test]
fn the_mla_prefill_ada_arm_answers_the_reference() {
    let mut gpu = Gpu::open();
    let (case, q_nope, q_pe, ckv, kpe) = case();
    let raw = Raw {
        q_nope: gpu.up(&q_nope),
        q_pe: gpu.up(&q_pe),
        ckv: gpu.up(&ckv),
        kpe: gpu.up(&kpe),
        qo_indptr: gpu.up(&case.qo_indptr),
        page_indices: gpu.up(&PAGE_INDICES),
        page_indptr: gpu.up(&PAGE_INDPTR),
        last_page_lens: gpu.up(&LAST_PAGE_LENS),
    };
    let want = case.reference();

    let probed = Device::probe(&gpu.ctx()).expect("the device answers its attributes");
    // The Ada ceiling on this device's SM count: the cooperative grid must be
    // co-resident here, while the arm choice is the L40S's.
    let ada = Device {
        num_sm: probed.num_sm,
        ..Device::L40S
    };

    let native = prefill(&mut gpu, &case, &raw, &probed);
    assert_close(&native, &want, "the probed device's arm");

    let sixteen = prefill(&mut gpu, &case, &raw, &ada);
    assert_close(
        &sixteen,
        &want,
        "the CTA_TILE_KV = 16 arm under the Ada ceiling",
    );
}
