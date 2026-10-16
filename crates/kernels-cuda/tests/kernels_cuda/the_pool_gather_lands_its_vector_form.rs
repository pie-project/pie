#![cfg(feature = "cuda")]

//! `kernel.elemwise.rows_vec` swaps DeepSeek-V4's pool gather for its
//! one-pass form over 16-byte runs: at both pooling ratios (4 with its
//! overlapping two-column window, 128), at the entry widths it gathers (512
//! and the indexer's 128), with and without the position bias, over shuffled
//! pages and on a staged window, it lands what the two-pass form lands
//! within rounding, zeros where no entry closes and leaves rows past the
//! window alone.

use crate::common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::attn::pool;
use kernels_cuda::tensor::{KvPool, Tensor};
use kernels_cuda::tuning::Tuning;

const PAGE: usize = 16;
const REQUESTS: usize = 3;
const PAGES_PER: usize = 12;

fn pool_of(gpu: &mut Gpu, keys: u64, pages: &[u32], indptr: &[u32], width: usize) -> KvPool {
    KvPool {
        keys: Tensor::new(keys, (pages.len() * PAGE) as u32, width as u32, Dtype::Bf16),
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
        seq_stride: width as i64,
        head_stride: width as i64,
        layout: 0,
        scheme_byte: 0,
        block_size: 0,
        max_pages_per_request: PAGES_PER as i32,
        pages_in_batch: pages.len() as i32,
    }
}

/// The gathered entries, `vec` choosing the form; `window` is `(live, base)`.
fn gather(vec: bool, ratio: usize, d: usize, ape: bool, window: Option<(u32, u32)>) -> Vec<u16> {
    let coff = if ratio == 4 { 2 } else { 1 };
    let width = coff * d;
    let positions = PAGES_PER * PAGE;
    let total = REQUESTS * PAGES_PER;
    // Shuffled pages: request r's page j is (7 * (r * PAGES_PER + j)) mod total.
    let pages: Vec<u32> = (0..total).map(|i| ((7 * i) % total) as u32).collect();
    let indptr: Vec<u32> = (0..=REQUESTS).map(|r| (r * PAGES_PER) as u32).collect();
    let mut bpos = Vec::new();
    let mut breq = Vec::new();
    for r in 0..REQUESTS {
        for p in 0..positions {
            bpos.push(if (p + 1) % ratio == 0 { p as i32 } else { -1 });
            breq.push(r as i32);
        }
    }
    let rows = bpos.len();
    let mut lcg = Lcg::seeded(0x9a7 + (ratio * d) as u64 + u64::from(ape));
    let (kv_bits, _) = lcg.row(total * PAGE * width);
    let score_bits: Vec<u16> = (0..total * PAGE * width)
        .map(|_| crate::common::to_bf16(lcg.unit() * 4.0))
        .collect();
    let ape_vals: Vec<f32> = (0..ratio * width).map(|_| lcg.unit()).collect();
    let mut gpu = Gpu::open();
    let kv = gpu.up(&kv_bits);
    let score = gpu.up(&score_bits);
    let state = pool_of(&mut gpu, kv, &pages, &indptr, width);
    let bpos_t = Tensor::new(gpu.up(&bpos), rows as u32, 1, Dtype::I32);
    let breq_t = Tensor::new(gpu.up(&breq), rows as u32, 1, Dtype::I32);
    let ape_t = if ape {
        Tensor::new(gpu.up(&ape_vals), ratio as u32, width as u32, Dtype::F32)
    } else {
        Tensor::ABSENT
    };
    let out_at = gpu.up(&vec![0x3f80u16; rows * d]);
    let ctx = gpu.ctx().with_tuning(Tuning {
        rows_vec: vec,
        ..Tuning::DEFAULT
    });
    if let Some((live, base)) = window {
        let win_at = gpu.up(&[live, base, 0u32, 0u32]);
        ctx.arm_stage(win_at);
    }
    let cells = (total * PAGE) as u32;
    pool::gather(
        &ctx,
        bpos_t,
        breq_t,
        &state,
        d as u32,
        ratio as u32,
        Tensor::new(kv, cells, width as u32, Dtype::Bf16),
        Tensor::new(score, cells, width as u32, Dtype::Bf16),
        ape_t,
        &mut Tensor::new(out_at, rows as u32, d as u32, Dtype::Bf16),
    )
    .expect("the gather fires");
    gpu.sync();
    gpu.down(out_at, rows * d)
}

fn check(ratio: usize, d: usize, ape: bool, window: Option<(u32, u32)>) {
    let old = gather(false, ratio, d, ape, window);
    let new = gather(true, ratio, d, ape, window);
    let mut closed = 0;
    for (i, (&g, &w)) in new.iter().zip(&old).enumerate() {
        let (gf, wf) = (from_bf16(g), from_bf16(w));
        assert!(
            (gf - wf).abs() <= wf.abs() / 64.0 + 1e-3,
            "ratio {ratio} d {d} ape {ape} {window:?}: element {i} is {gf} against {wf}"
        );
        closed += usize::from(w != 0 && w != 0x3f80);
    }
    assert!(closed > 0, "some entry closes");
}

#[test]
fn the_pool_gather_lands_its_vector_form() {
    for (ratio, d) in [(4, 512), (4, 128), (128, 512)] {
        for ape in [false, true] {
            check(ratio, d, ape, None);
        }
    }
    check(4, 512, true, Some((300, 100)));
    check(128, 512, true, Some((400, 50)));
}
