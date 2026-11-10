#![cfg(feature = "cuda")]

//! The kernels that write paged kv rows, each over a packed pool and over
//! the same pages spread apart as a state chunk lays them (layer-major
//! structs a stride apart, shuffled, garbage between): byte for byte the
//! same rows land, nothing lands past them, and a wrong plane offset
//! lands something else (`common::spread::agrees`). The readers are held
//! to the same in their own families' tests.

use crate::common::attn_ref::{col, pool, ten, tuned};
use crate::common::spread::agrees;
use crate::common::{Gpu, Lcg};
use dtype::Dtype::{self, Bf16, E4m3, I32, U8, U32};
use kernels_cuda::attn;
use kernels_cuda::tensor::{KvPool, RaggedTensor, Tensor};
use kernels_cuda::tuning::Tuning;

const PAGE: usize = 16;

/// Random bytes on the device, `bytes` of them.
fn noise(gpu: &mut Gpu, lcg: &mut Lcg, bytes: usize) -> u64 {
    let raw: Vec<u8> = (0..bytes)
        .map(|_| (lcg.unit().to_bits() >> 9) as u8)
        .collect();
    gpu.up(&raw)
}

/// A bf16 `[rows][width]` plane of random values.
fn rows(gpu: &mut Gpu, lcg: &mut Lcg, rows: usize, width: usize) -> Tensor {
    let (raw, _) = lcg.row(rows * width);
    ten(gpu.up(&raw), rows, width, Bf16)
}

/// A pool of `pages` pages of `width`-wide `dtype` rows (random), two
/// requests over `lens` positions each, their pages shuffled through the
/// pool; keys and values planes of their own where `values`.
fn paged(
    gpu: &mut Gpu,
    lcg: &mut Lcg,
    dtype: Dtype,
    width: usize,
    values: Option<(Dtype, usize)>,
    lens: [usize; 2],
) -> KvPool {
    let pages = 9;
    let cells = pages * PAGE;
    let bytes = |d: Dtype, w: usize| cells * w * (d.bits() as usize).div_ceil(8);
    let keys = ten(noise(gpu, lcg, bytes(dtype, width)), cells, width, dtype);
    let vals = values.map_or(keys, |(d, w)| {
        ten(noise(gpu, lcg, bytes(d, w)), cells, w, d)
    });
    let ids: Vec<i32> = vec![6, 1, 4, 8, 2, 0, 7, 3];
    let need = |n: usize| n.div_ceil(PAGE);
    let (a, b) = (need(lens[0]), need(lens[1]));
    let indptr = [0, a as i32, (a + b) as i32];
    let last = lens.map(|n| ((n - 1) % PAGE + 1) as i32);
    let p = pool(
        keys,
        col(gpu, &ids[..a + b], I32),
        col(gpu, &indptr, I32),
        PAGE,
    );
    KvPool {
        values: vals,
        last_page_lens: col(gpu, &last, I32),
        seq_stride: width as i64,
        head_stride: width as i64,
        ..p
    }
    .with_packed_stride()
}

/// Each of `n` rows' page and offset: rows `0..16` on page 5, then page 2.
fn written(gpu: &mut Gpu, n: usize) -> (Tensor, Tensor) {
    let page: Vec<u32> = (0..n).map(|t| if t < PAGE { 5 } else { 2 }).collect();
    let off: Vec<u32> = (0..n).map(|t| (t % PAGE) as u32).collect();
    (col(gpu, &page, U32), col(gpu, &off, U32))
}

#[test]
fn a_kv_append_lands_the_same_rows_over_spread_pages() {
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let mut lcg = Lcg::seeded(0xa99e);
    let (heads, d, n) = (2, 64, 20);
    for dtype in [Bf16, E4m3] {
        for shared in [false, true] {
            let values = (!shared).then_some((dtype, heads * d));
            let p = paged(&mut gpu, &mut lcg, dtype, heads * d, values, [40, 30]);
            let k = rows(&mut gpu, &mut lcg, n, heads * d);
            let v = rows(&mut gpu, &mut lcg, n, heads * d);
            let indptr = col(&mut gpu, &[0i32, n as i32], I32);
            let (page, off) = written(&mut gpu, n);
            let what = format!("kv_append {dtype:?} shared {shared}");
            agrees(
                &mut gpu,
                &what,
                &[p],
                &[vec![page]],
                &[],
                |_, pools, tables| {
                    let k = RaggedTensor { data: k, indptr };
                    let fired = if shared {
                        attn::kv_append_shared(&ctx, k, &pools[0], tables[0][0], off)
                    } else {
                        attn::kv_append(&ctx, k, v, &pools[0], tables[0][0], off)
                    };
                    fired.expect("the append fires");
                },
            );
        }
    }
}

#[test]
fn a_fused_qkv_write_lands_the_same_rows_over_spread_pages() {
    let mut gpu = Gpu::open();
    let mut lcg = Lcg::seeded(0x9c5);
    // The row kernel (staged and not), the warp kernel, and the block-a-head one.
    let rows_from = |from: u32, vec: bool| tuned(|t| (t.qkv_rows_from, t.rows_vec) = (from, vec));
    #[rustfmt::skip]
    let cases: [(usize, Tuning, Dtype); 6] = [
        (128, rows_from(1, false), Bf16),
        (128, rows_from(1, true), Bf16),
        (128, rows_from(1, false), E4m3),
        (64, rows_from(4096, false), Bf16),
        (128, rows_from(4096, false), E4m3),
        (32, rows_from(4096, false), Bf16),
    ];
    for (d, tuning, dtype) in cases {
        let ctx = gpu.ctx().with_tuning(tuning);
        let (qh, kvh, n) = (4, 2, 20);
        let p = paged(
            &mut gpu,
            &mut lcg,
            dtype,
            kvh * d,
            Some((dtype, kvh * d)),
            [40, 30],
        );
        let packed = rows(&mut gpu, &mut lcg, n, (qh + 2 * kvh) * d);
        let positions: Vec<i32> = (0..n as i32).map(|t| 3 * t + 1).collect();
        let positions = col(&mut gpu, &positions, I32);
        let weight = rows(&mut gpu, &mut lcg, 1, d);
        let (page, off) = written(&mut gpu, n);
        let q_at = gpu.zeros(n * qh * d * 2);
        let what = format!("qkv write {d} {dtype:?} {tuning:?}");
        let outs = [(q_at, n * qh * d * 2)];
        let norms = d == 32;
        agrees(
            &mut gpu,
            &what,
            &[p],
            &[vec![page]],
            &outs,
            |_, pools, tables| {
                let mut q = ten(q_at, n, qh * d, Bf16);
                let w = norms.then_some(weight);
                attn::qkv_append::qkv_fused_qknorm_rope_vnorm_write(
                    &ctx,
                    packed,
                    Some(positions),
                    w,
                    w,
                    false,
                    1e-6,
                    &pools[0],
                    tables[0][0],
                    off,
                    kvh as u32,
                    d as u32,
                    10_000.0,
                    d as u32,
                    d as u32 / 2,
                    norms,
                    1.0,
                    [d as u32; 2],
                    &mut q,
                )
                .expect("the fused write fires");
            },
        );
    }
}

#[test]
fn an_mla_append_lands_the_same_rows_over_spread_pages() {
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let mut lcg = Lcg::seeded(0x31a);
    let (lens, n) = ([37, 21], 9);
    // bf16 latents beside their rotated plane, and vLLM's fp8 rows (one plane).
    for fp8 in [false, true] {
        let p = if fp8 {
            paged(&mut gpu, &mut lcg, U8, 576, None, lens)
        } else {
            paged(&mut gpu, &mut lcg, Bf16, 512, Some((Bf16, 64)), lens)
        };
        let ckv = rows(&mut gpu, &mut lcg, n, 512);
        let kpe = rows(&mut gpu, &mut lcg, n, 64);
        // The rows close each request: 5 and 4 new ones.
        let qo = col(&mut gpu, &[0i32, 5, 9], I32);
        let what = format!("mla append fp8 {fp8}");
        agrees(&mut gpu, &what, &[p], &[], &[], |_, pools, _| {
            let kv_c = RaggedTensor {
                data: ckv,
                indptr: qo,
            };
            attn::mla::kv_append(&ctx, kv_c, kpe, &pools[0], Tensor::ABSENT, Tensor::ABSENT)
                .expect("the append fires");
        });
    }
}

/// Each entry's boundary: entry `c` closes request `c % 2`'s group at
/// position `(c / 2 + 1) * ratio - 1`.
fn boundaries(gpu: &mut Gpu, n: usize, ratio: usize) -> (Tensor, Tensor) {
    let pos: Vec<i32> = (0..n).map(|c| ((c / 2 + 1) * ratio - 1) as i32).collect();
    let req: Vec<i32> = (0..n).map(|c| (c % 2) as i32).collect();
    (col(gpu, &pos, I32), col(gpu, &req, I32))
}

#[test]
fn a_compressed_store_lands_the_same_rows_over_spread_pages() {
    use attn::compressed::{kv_fp8, pool as cpool};
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let mut lcg = Lcg::seeded(0xc0);
    let lens = [64, 48];
    // DeepSeek's 584-byte rows: the window's append, and the entries' store.
    let p = paged(&mut gpu, &mut lcg, U8, 584, None, lens);
    let n = 20;
    let plane = rows(&mut gpu, &mut lcg, n, 512);
    let (page, off) = written(&mut gpu, n);
    agrees(
        &mut gpu,
        "fp8 rows append",
        &[p],
        &[vec![page]],
        &[],
        |_, pools, tables| {
            kv_fp8::append(&ctx, plane, &pools[0], tables[0][0], off).expect("the append fires");
        },
    );
    for (ratio, compact) in [(4, false), (4, true)] {
        let n = 10;
        let entries = rows(&mut gpu, &mut lcg, n, 512);
        let (pos, req) = boundaries(&mut gpu, n, ratio);
        // Fp8 rows a position, or compact: a row an entry, its positions a quarter wide.
        let width = if compact { 584 / ratio } else { 584 };
        let p = paged(&mut gpu, &mut lcg, U8, width, None, lens);
        let what = format!("fp8 entries store compact {compact}");
        agrees(&mut gpu, &what, &[p], &[], &[], |_, pools, _| {
            kv_fp8::store_entries(&ctx, entries, pos, req, &pools[0]).expect("the store fires");
        });
        // bf16 entries, a row a position or compact.
        let width = if compact { 512 / ratio } else { 512 };
        let p = paged(&mut gpu, &mut lcg, Bf16, width, None, lens);
        let what = format!("bf16 entries store compact {compact}");
        agrees(&mut gpu, &what, &[p], &[], &[], |_, pools, _| {
            cpool::kv_append(
                &ctx,
                entries,
                pos,
                req,
                &pools[0],
                Tensor::ABSENT,
                Tensor::ABSENT,
            )
            .expect("the store fires");
        });
    }
    // DeepSeek-V4.1's FP4 index keys, a row a position.
    let keys = rows(&mut gpu, &mut lcg, 10, 128);
    let (pos, req) = boundaries(&mut gpu, 10, 4);
    let p = paged(&mut gpu, &mut lcg, U8, 68, None, lens);
    agrees(
        &mut gpu,
        "fp4 index keys store",
        &[p],
        &[],
        &[],
        |_, pools, _| {
            attn::index::store_fp4(&ctx, keys, pos, req, &pools[0]).expect("the store fires");
        },
    );
}

#[test]
fn a_normed_rotated_append_lands_the_same_rows_over_spread_pages() {
    use kernels_cuda::elemwise::norm_rope::{Append, norm_rope_last_append};
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx().with_tuning(tuned(|t| t.rows_vec = true));
    let mut lcg = Lcg::seeded(0x4e);
    let n = 20;
    let p = paged(&mut gpu, &mut lcg, U8, 584, None, [64, 48]);
    let x = rows(&mut gpu, &mut lcg, n, 512);
    let weight = rows(&mut gpu, &mut lcg, 1, 512);
    let positions: Vec<i32> = (0..n as i32).collect();
    let positions = col(&mut gpu, &positions, I32);
    let (page, off) = written(&mut gpu, n);
    let y_at = gpu.zeros(n * 512 * 2);
    let outs = [(y_at, n * 512 * 2)];
    agrees(
        &mut gpu,
        "norm rope append",
        &[p],
        &[vec![page]],
        &outs,
        |_, pools, tables| {
            let mut y = ten(y_at, n, 512, Bf16);
            let append = Append {
                pool: &pools[0],
                write_page: tables[0][0],
                write_offset: off,
            };
            let fired = norm_rope_last_append(
                &ctx,
                x,
                weight,
                512,
                1e-6,
                positions,
                64,
                10_000.0,
                true,
                false,
                None,
                &mut y,
                Some(append),
            );
            assert!(
                fired.expect("the pass fires"),
                "the one pass serves DeepSeek's append"
            );
        },
    );
}
