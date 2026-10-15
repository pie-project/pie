#![cfg(feature = "cuda")]

//! DeepSeek-V4's compressed-pool attention, the whole visible pool and the
//! rows an index selected, on the scalar walk and on the tensor cores: every
//! head's output and log-sum-exp alike up to the bf16 the tensor-core walk
//! rounds its probabilities to.

use crate::common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::attn::pool;
use kernels_cuda::tensor::{KvPool, Tensor};
use kernels_cuda::tuning::Tuning;

const PAGE: usize = 64;

fn pool_of(gpu: &mut Gpu, keys: u64, cells: usize, d: usize, pages: &[u32], indptr: &[u32]) -> KvPool {
    KvPool {
        keys: Tensor::new(keys, cells as u32, d as u32, Dtype::Bf16),
        values: Tensor::ABSENT,
        bf16_keys: Tensor::ABSENT,
        bf16_values: Tensor::ABSENT,
        key_scales: Tensor::ABSENT,
        value_scales: Tensor::ABSENT,
        page_indices: Tensor::new(gpu.up(pages), pages.len() as u32, 1, Dtype::U32),
        page_indptr: Tensor::new(gpu.up(indptr), indptr.len() as u32, 1, Dtype::U32),
        last_page_lens: Tensor::ABSENT,
        row_valid: Tensor::ABSENT,
        env_min: Tensor::ABSENT,
        env_max: Tensor::ABSENT,
        has_envelopes: false,
        page_size: PAGE as i32,
        seq_stride: d as i64,
        head_stride: d as i64,
        layout: 0,
        scheme_byte: 0,
        block_size: 0,
        max_pages_per_request: (pages.len() / 2) as i32,
        pages_in_batch: pages.len() as i32,
    }
}

#[allow(clippy::too_many_arguments)]
fn check(heads: usize, d: usize, ratio: usize, top_k: usize, selected: bool) {
    // Two requests of 2304 positions each, their pages interleaved.
    let span = 2304usize;
    let per_req = span / PAGE;
    let pages: Vec<u32> = (0..2 * per_req as u32)
        .map(|i| if i < per_req as u32 { 2 * i } else { 2 * (i - per_req as u32) + 1 })
        .collect();
    let indptr: Vec<u32> = vec![0, per_req as u32, 2 * per_req as u32];
    let cells = 2 * per_req * PAGE;

    let positions: Vec<i32> = vec![2, 3, 50, 700, 2047, 2100, 1500, 9];
    let reqs: Vec<i32> = vec![0, 1, 0, 1, 0, 1, 1, 0];
    let rows = positions.len();

    let mut lcg = Lcg::seeded(0x9001 + d as u64 + heads as u64);
    let (q_bits, _) = lcg.row(rows * heads * d);
    let (k_bits, _) = lcg.row(cells * d);
    let mut sel: Vec<i32> = Vec::with_capacity(rows * top_k);
    for (r, &pos) in positions.iter().enumerate() {
        let visible = ((pos + 1) as usize) / ratio;
        for n in 0..top_k {
            let id = match (r + n) % 9 {
                // An id past the visible rows (read as nothing).
                0 => (visible + n % 7) as i32,
                _ => ((n * 37 + r * 11) % (visible.max(1) + 3)) as i32,
            };
            sel.push(id);
        }
        // Rows 3 and 5 end their selection early.
        if r == 3 || r == 5 {
            for n in top_k / 2..top_k {
                sel[r * top_k + n] = if n == top_k / 2 { -1 } else { 0 };
            }
        }
    }

    let mut gpu = Gpu::open();
    let q = Tensor::new(gpu.up(&q_bits), rows as u32, (heads * d) as u32, Dtype::Bf16);
    let keys = gpu.up(&k_bits);
    let pos_t = Tensor::new(gpu.up(&positions), rows as u32, 1, Dtype::I32);
    let req_t = Tensor::new(gpu.up(&reqs), rows as u32, 1, Dtype::I32);
    let sel_t = Tensor::new(gpu.up(&sel), rows as u32, top_k as u32, Dtype::I32);
    let pool_t = pool_of(&mut gpu, keys, cells, d, &pages, &indptr);
    let scale = 1.0 / (d as f32).sqrt();

    let run = |gpu: &mut Gpu, mma: bool| -> (Vec<u16>, Vec<f32>) {
        let o_at = gpu.zeros(rows * heads * d * 2);
        let lse_at = gpu.zeros(rows * heads * 4);
        let mut o = Tensor::new(o_at, rows as u32, (heads * d) as u32, Dtype::Bf16);
        let mut lse = Tensor::new(lse_at, rows as u32, heads as u32, Dtype::F32);
        let ctx = gpu.ctx().with_tuning(Tuning {
            pool_mma: mma,
            ..Tuning::DEFAULT
        });
        if selected {
            pool::attention_lse_selected(
                &ctx, q, pos_t, req_t, sel_t, &pool_t, ratio as u32, top_k as u32,
                heads as u32, d as u32, scale, &mut o, &mut lse,
            )
            .expect("the selected pool attention fires");
        } else {
            pool::attention_lse(
                &ctx, q, pos_t, req_t, &pool_t, ratio as u32, heads as u32, d as u32, scale,
                &mut o, &mut lse,
            )
            .expect("the pool attention fires");
        }
        gpu.sync();
        (gpu.down(o_at, rows * heads * d), gpu.down(lse_at, rows * heads))
    };
    let (o_ref, lse_ref) = run(&mut gpu, false);
    let (o_mma, lse_mma) = run(&mut gpu, true);
    let label = format!("{heads} heads at {d}, ratio {ratio}, selected {selected}");
    assert!(o_ref.iter().any(|&b| b != 0), "{label}: the scalar walk landed rows");
    for (i, (&a, &b)) in lse_ref.iter().zip(&lse_mma).enumerate() {
        if a.is_infinite() || b.is_infinite() {
            assert_eq!(a, b, "{label}: row {} head {} log-sum-exp", i / heads, i % heads);
        } else {
            assert!((a - b).abs() <= 2e-2 * a.abs().max(1.0), "{label}: lse {i}: {a} vs {b}");
        }
    }
    let mut worst = 0f32;
    for (&a, &b) in o_ref.iter().zip(&o_mma) {
        let (a, b) = (from_bf16(a), from_bf16(b));
        worst = worst.max((a - b).abs() / a.abs().max(0.05));
    }
    assert!(worst <= 3e-2, "{label}: an output element differs by {worst} relative");
}

#[test]
fn the_pool_attention_lands_alike_on_the_tensor_cores() {
    // DeepSeek-V4-Flash's reading: 64 heads at 512, ratio 4 selected (512) and
    // ratio 128 whole.
    check(64, 512, 4, 512, true);
    check(64, 512, 128, 512, false);
    check(64, 512, 4, 512, false);
    // A head count that leaves a partial block, and the narrower widths.
    check(20, 256, 4, 64, true);
    check(24, 128, 16, 32, false);
}
