//! Qwen's gated attention prologue in one launch
//! (`custom::q_gate_k_norm_rope_append`) against its primitives on the
//! device (`split_q_gate`, two `rmsnorm_per_head_plus_one`, `rope_partial`,
//! `kv_append`): bit for bit on q, the gate and both pages, windowed, with a
//! dead row and a slot past the page.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, KV, bf16s};
use dtype::Dtype;
use kernels_rocm::{KvPool, RaggedTensor, Tensor};

fn pool(gpu: &Gpu, slots: u32, row: u32, d: u32, ps: u32, row_valid: Tensor) -> KvPool {
    KvPool {
        keys: gpu.zeros(slots, row, Dtype::Bf16),
        values: gpu.zeros(slots, row, Dtype::Bf16),
        row_valid,
        page_size: ps as i32,
        seq_stride: i64::from(row),
        head_stride: i64::from(d),
        ..KV
    }
}

#[test]
fn the_fused_prologue_lands_what_its_primitives_land() {
    let Some(gpu) = Gpu::open() else { return };
    let (theta, ps) = (1e7f32, 16u32);
    // Qwen3.8-27B's attention: 24 q heads, 4 kv heads of 256, a quarter
    // rotated; and a narrower head with a proportional span.
    for (hq, hk, d, rotary, span) in [(24u32, 4u32, 256u32, 64u32, 32u32), (8, 2, 128, 64, 64)] {
        for (rows, win, dead) in [
            (1u32, None, None),
            (3, None, Some(1usize)),
            (7, Some([4, 2, 4, 2]), None),
        ] {
            let run = |fused: bool| {
                let qg = gpu.tensor(
                    &bf16s((rows * 2 * hq * d) as usize, 1, 2.0),
                    rows,
                    2 * hq * d,
                    Dtype::Bf16,
                );
                let k = gpu.tensor(
                    &bf16s((rows * hk * d) as usize, 2, 2.0),
                    rows,
                    hk * d,
                    Dtype::Bf16,
                );
                let v = gpu.tensor(
                    &bf16s((rows * hk * d) as usize, 3, 2.0),
                    rows,
                    hk * d,
                    Dtype::Bf16,
                );
                let qw = gpu.tensor(&bf16s(d as usize, 4, 0.5), 1, d, Dtype::Bf16);
                let kw = gpu.tensor(&bf16s(d as usize, 5, 0.5), 1, d, Dtype::Bf16);
                let positions: Vec<i32> = (0..rows as i32).map(|r| (r * 977 + 3) % 9000).collect();
                let pages = rows.div_ceil(2) + 1;
                let w_page: Vec<i32> = (0..rows as i32).map(|r| pages as i32 - 1 - r / 2).collect();
                // The last row's slot is past its page: it lands nothing.
                let w_off: Vec<i32> = (0..rows as i32)
                    .map(|r| {
                        if r == rows as i32 - 1 && rows > 1 {
                            ps as i32
                        } else {
                            3 + r % 2
                        }
                    })
                    .collect();
                let valid: Vec<i32> = (0..rows.div_ceil(4))
                    .map(|w| {
                        (0..4).fold(0i32, |acc, b| {
                            let r = (w * 4 + b) as usize;
                            acc | (i32::from(Some(r) != dead) << (8 * b))
                        })
                    })
                    .collect();
                let rv = gpu.ints(&valid);
                let rv = Tensor::new(rv.ptr, rows, 1, Dtype::U8);
                let pool = pool(
                    &gpu,
                    pages * ps,
                    hk * d,
                    d,
                    ps,
                    if dead.is_some() { rv } else { Tensor::ABSENT },
                );
                let mut q = gpu.zeros(rows, hq * d, Dtype::Bf16);
                let mut gate = gpu.zeros(rows, hq * d, Dtype::Bf16);
                let mut qs = gpu.zeros(rows, hq * d, Dtype::Bf16);
                let (positions, w_page, w_off) = (
                    gpu.column(&positions),
                    gpu.column(&w_page),
                    gpu.column(&w_off),
                );
                let ctx = gpu.ctx();
                let win_t = win.map(|w| gpu.ints(&w));
                if let Some(w) = win_t {
                    ctx.arm_stage(w.ptr);
                }
                if fused {
                    assert!(
                        kernels_rocm::attn::qkv_append::q_gate_k_norm_rope_append(
                            &ctx, qg, k, v, qw, kw, positions, 1e-6, 1e-6, &pool, w_page, w_off, d,
                            rotary, span, theta, &mut q, &mut gate, &mut qs,
                        )
                        .unwrap(),
                        "the fused prologue covers this shape"
                    );
                } else {
                    let mut qn = gpu.zeros(rows, hq * d, Dtype::Bf16);
                    let mut kn = gpu.zeros(rows, hk * d, Dtype::Bf16);
                    kernels_rocm::gather::split_q_gate(&ctx, qg, d, &mut qs, &mut gate).unwrap();
                    kernels_rocm::elemwise::norm::rmsnorm_per_head_plus_one(
                        &ctx, qs, qw, d, 1e-6, &mut qn,
                    )
                    .unwrap();
                    kernels_rocm::elemwise::norm::rmsnorm_per_head_plus_one(
                        &ctx, k, kw, d, 1e-6, &mut kn,
                    )
                    .unwrap();
                    kernels_rocm::elemwise::rope::partial(
                        &ctx, &mut qn, &mut kn, positions, rotary, d, theta, span,
                    )
                    .unwrap();
                    let indptr = gpu.column(&(0..=rows as i32).collect::<Vec<_>>());
                    kernels_rocm::attn::kv_append(
                        &ctx,
                        RaggedTensor { data: kn, indptr },
                        v,
                        &pool,
                        w_page,
                        w_off,
                    )
                    .unwrap();
                    q = qn;
                }
                vec![
                    gpu.read(&q),
                    gpu.read(&gate),
                    gpu.read(&pool.keys),
                    gpu.read(&pool.values),
                    gpu.read(&qs),
                ]
            };
            let (have, want) = (run(true), run(false));
            let (r0, live) = win.map_or((0, rows as usize), |w| (w[1] as usize, w[0] as usize));
            let names = ["q", "gate", "keys", "values", "split q"];
            for (i, (h, w)) in have.iter().zip(&want).enumerate() {
                let rowwise = i < 2 || i == 4;
                let width = h.len() / if rowwise { rows as usize } else { 1 };
                let range = if rowwise {
                    r0 * width..(r0 + live) * width
                } else {
                    0..h.len()
                };
                for e in range {
                    assert!(
                        h[e].to_bits() == w[e].to_bits(),
                        "{hq}x{hk}x{d} rows {rows} window {win:?}: {} element {e}: fused {} vs apart {}",
                        names[i],
                        h[e],
                        w[e]
                    );
                }
            }
            assert!(have[2].iter().any(|&x| x != 0.0), "some key landed");
        }
    }
}
