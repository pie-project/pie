#![cfg(feature = "cuda")]

// A latent attention's prefill expanded (the fire's latent rows through
// `kv_b` into a key and value head a query head, then the tcgen05 prefill
// at key width 192 and value width 128) against the same prefill absorbed
// (the query absorb, the latent scoring, the output absorb): one request
// over a cached prefix, one fresh, the page table out of order; the queries
// both whole (`q_b`'s rows) and as their two halves.

use crate::common::{Gpu, Lcg, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::attn::mla::{self, ExpandedQ, Heads};
use kernels_cuda::attn::plan::{self, Device, Live, Workspace};
use kernels_cuda::tensor::{KvPool, RaggedTensor, Tensor};

const HEADS: usize = 16;
const RANK: usize = 512;
const NOPE: usize = 128;
const ROPE: usize = 64;
const V: usize = 128;
const PAGE: usize = 16;

const Q_LENS: [usize; 2] = [40, 150];
const KV_LENS: [usize; 2] = [77, 150];
const PAGE_INDICES: [i32; 15] = [3, 0, 5, 1, 6, 14, 2, 4, 13, 7, 12, 8, 11, 9, 10];
const PAGE_INDPTR: [i32; 3] = [0, 5, 15];
const LAST_PAGE_LENS: [i32; 2] = [13, 6];

const INT_BYTES: usize = 4 << 20;
const FLOAT_BYTES: usize = 64 << 20;

#[test]
fn the_expanded_mla_prefill_lands_what_the_absorbed_one_does() {
    let mut gpu = Gpu::open();
    let probed = Device::probe(&gpu.ctx()).expect("the device answers its attributes");
    let h = Heads {
        heads: HEADS as u32,
        kv_lora_rank: RANK as u32,
        nope: NOPE as u32,
        rope: ROPE as u32,
        v: V as u32,
    };
    if !mla::expanded_takes(&h) {
        eprintln!("no tcgen05 here: skipped");
        return;
    }
    let ctx = gpu.ctx();
    let mut rng = Lcg::seeded(0x6d6c_6178);
    let tokens: usize = Q_LENS.iter().sum();
    let slots = PAGE_INDICES.len() * PAGE;
    let (q_b, _) = rng.row(tokens * HEADS * (NOPE + ROPE));
    let (ckv, _) = rng.row(slots * RANK);
    let (kpe, _) = rng.row(slots * ROPE);
    // kv_b small, so the scores stay in a softmax's useful range.
    let kv_b: Vec<u16> = (0..HEADS * (NOPE + V) * RANK)
        .map(|_| to_bf16(0.08 * rng.unit()))
        .collect();
    let mut qo_indptr = vec![0i32];
    for q in Q_LENS {
        qo_indptr.push(qo_indptr.last().unwrap() + q as i32);
    }
    let sm_scale = 1.0 / ((NOPE + ROPE) as f32).sqrt();
    let rows = tokens as u32;
    let q_b_t = Tensor::new(
        gpu.up(&q_b),
        rows,
        (HEADS * (NOPE + ROPE)) as u32,
        Dtype::Bf16,
    );
    let kv_b_t = Tensor::new(
        gpu.up(&kv_b),
        (HEADS * (NOPE + V)) as u32,
        RANK as u32,
        Dtype::Bf16,
    );
    let qo = Tensor::new(gpu.up(&qo_indptr), qo_indptr.len() as u32, 1, Dtype::I32);
    let pool = KvPool {
        keys: Tensor::new(gpu.up(&ckv), slots as u32, RANK as u32, Dtype::Bf16),
        values: Tensor::new(gpu.up(&kpe), slots as u32, ROPE as u32, Dtype::Bf16),
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
        last_page_lens: Tensor::new(
            gpu.up(&LAST_PAGE_LENS),
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
        max_pages_per_request: 10,
        pages_in_batch: PAGE_INDICES.len() as i32,
        live: [0, 0],
    };

    // Absorbed: split, absorb the queries, score the latents, absorb out.
    let mut q_nope = Tensor::new(
        gpu.zeros(tokens * HEADS * NOPE * 2),
        rows,
        (HEADS * NOPE) as u32,
        Dtype::Bf16,
    );
    let mut q_pe = Tensor::new(
        gpu.zeros(tokens * HEADS * ROPE * 2),
        rows,
        (HEADS * ROPE) as u32,
        Dtype::Bf16,
    );
    mla::split_q_b(&ctx, q_b_t, h.heads, h.nope, h.rope, &mut q_nope, &mut q_pe).expect("splits");
    let mut q_lat = Tensor::new(
        gpu.zeros(tokens * HEADS * RANK * 2),
        rows,
        (HEADS * RANK) as u32,
        Dtype::Bf16,
    );
    mla::absorb_q(
        &ctx,
        q_nope,
        kv_b_t,
        h.heads,
        h.kv_lora_rank,
        h.nope,
        h.v,
        &mut q_lat,
    )
    .expect("absorbs");
    let workspace = Workspace {
        int_ptr: gpu.zeros(INT_BYTES),
        int_bytes: INT_BYTES,
        float_ptr: gpu.zeros(FLOAT_BYTES),
        float_bytes: FLOAT_BYTES,
    };
    let kv_len: Vec<i32> = KV_LENS.iter().map(|&n| n as i32).collect();
    let plan = plan::plan_mla(
        &qo_indptr,
        &PAGE_INDPTR,
        &kv_len,
        rows,
        Q_LENS.len() as u32,
        Live {
            requests: Q_LENS.len() as u32,
            lane_offset: 0,
            row_offset: 0,
            rows,
        },
        HEADS as u32,
        RANK as u32,
        true,
        &probed,
        workspace,
    )
    .expect("the plan schedules");
    plan.stage(&ctx).expect("stages");
    let mut scored = Tensor::new(
        gpu.zeros(tokens * HEADS * RANK * 2),
        rows,
        (HEADS * RANK) as u32,
        Dtype::Bf16,
    );
    mla::attention_prefill(
        &ctx,
        RaggedTensor {
            data: q_lat,
            indptr: qo,
        },
        &plan,
        q_pe,
        &pool,
        h.heads,
        h.kv_lora_rank,
        sm_scale,
        None,
        &mut scored,
        None,
    )
    .expect("scores");
    let mut absorbed = Tensor::new(
        gpu.zeros(tokens * HEADS * V * 2),
        rows,
        (HEADS * V) as u32,
        Dtype::Bf16,
    );
    mla::absorb_out(
        &ctx,
        scored,
        kv_b_t,
        h.heads,
        h.kv_lora_rank,
        h.v,
        h.nope,
        &mut absorbed,
    )
    .expect("absorbs");
    gpu.sync();
    let want: Vec<u16> = gpu.down(absorbed.ptr, tokens * HEADS * V);

    for packed in [true, false] {
        let mut o = Tensor::new(
            gpu.zeros(tokens * HEADS * V * 2),
            rows,
            (HEADS * V) as u32,
            Dtype::Bf16,
        );
        let q = if packed {
            ExpandedQ::Packed(q_b_t)
        } else {
            ExpandedQ::Split { q_nope, q_pe }
        };
        let took =
            mla::prefill_expanded(&ctx, q, qo, &pool, kv_b_t, h, sm_scale, &mut o).expect("fires");
        assert!(took, "the expanded prefill takes the fire");
        gpu.sync();
        let got: Vec<u16> = gpu.down(o.ptr, tokens * HEADS * V);
        let scale = want.iter().map(|w| from_bf16(*w).abs()).sum::<f32>() / want.len() as f32;
        let mut worst = 0f32;
        let mut total = 0f32;
        let mut at = 0usize;
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let (g, w) = (from_bf16(*g), from_bf16(*w));
            let err = (g - w).abs();
            total += err;
            if err > worst {
                worst = err;
                at = i;
            }
        }
        let mean = total / want.len() as f32;
        eprintln!(
            "packed {packed}: mean error {mean}, worst {worst} at row {} head {}, mean magnitude {scale}",
            at / (HEADS * V),
            (at / V) % HEADS
        );
        // On average within a percent of the rows' magnitude; at worst a
        // couple of bf16 steps of a row over its first key or two, which
        // lands that key's value row nearly whole.
        assert!(
            mean <= 0.01 * scale && worst <= 0.15 * scale,
            "packed {packed}: the expanded prefill sits {mean} on average and {worst} at worst \
             from the absorbed one (mean magnitude {scale})"
        );
    }
}
