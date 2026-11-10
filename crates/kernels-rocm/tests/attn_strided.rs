//! The kernels that write paged kv rows, each over a packed pool and over
//! the same pages spread apart as a state chunk lays them (layer-major
//! structs a stride apart, shuffled, garbage between): byte for byte the
//! same rows land, nothing lands past them, and a wrong plane offset lands
//! something else (`common::spread::agrees`). The readers are held to the
//! same in their own families' tests.

#![cfg(feature = "rocm")]

mod common;

use common::spread::{agrees, out};
use common::{Gpu, KV, bf16s};
use dtype::Dtype::{self, Bf16, E4m3, F32, U8};
use kernels_rocm::attn::compressed::pool;
use kernels_rocm::{KvPool, RaggedTensor, Tensor};

const PAGE: usize = 16;

/// `bytes` bytes of noise, a xorshift stream from `seed`.
fn noise(seed: u64, bytes: usize) -> Vec<u8> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..bytes)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 32) as u8
        })
        .collect()
}

/// A bf16 `[rows, width]` plane of values in [-2, 2).
fn rows(gpu: &Gpu, seed: u64, rows: usize, width: usize) -> Tensor {
    gpu.tensor(
        &bf16s(rows * width, seed, 2.0),
        rows as u32,
        width as u32,
        Bf16,
    )
}

/// A pool of nine pages of `width`-wide `dtype` rows (noise), two requests
/// over `lens` positions each, their pages shuffled through the pool; a
/// value plane of its own where `values`. `hnd` lays heads `d` wide
/// head-major in a page.
fn paged(
    gpu: &Gpu,
    seed: u64,
    dtype: Dtype,
    width: usize,
    values: Option<(Dtype, usize)>,
    lens: [usize; 2],
    hnd: Option<usize>,
) -> KvPool {
    let cells = 9 * PAGE;
    let plane = |seed: u64, d: Dtype, w: usize| {
        let bytes = cells * w * (d.bits() as usize).div_ceil(8);
        gpu.upload(&noise(seed, bytes), cells as u32, w as u32, d)
    };
    let keys = plane(seed, dtype, width);
    let vals = values.map_or(keys, |(d, w)| plane(seed + 1, d, w));
    let ids: Vec<i32> = vec![6, 1, 4, 8, 2, 0, 7, 3];
    let need = |n: usize| n.div_ceil(PAGE);
    let (a, b) = (need(lens[0]), need(lens[1]));
    let indptr = [0, a as i32, (a + b) as i32];
    let last = lens.map(|n| ((n - 1) % PAGE + 1) as i32);
    let (seq, head) = match hnd {
        Some(d) => (d as i64, (PAGE * d) as i64),
        None => (width as i64, width as i64),
    };
    KvPool {
        keys,
        values: vals,
        page_indices: gpu.column(&ids[..a + b]),
        page_indptr: gpu.column(&indptr),
        last_page_lens: gpu.column(&last),
        page_size: PAGE as i32,
        seq_stride: seq,
        head_stride: head,
        layout: i32::from(hnd.is_some()),
        max_pages_per_request: a.max(b) as i32,
        pages_in_batch: (a + b) as i32,
        ..KV
    }
    .with_packed_stride()
}

/// Each of `n` rows' page and offset: rows `0..16` on page 5, then page 2.
fn written(gpu: &Gpu, n: usize) -> (Tensor, Tensor) {
    let page: Vec<i32> = (0..n).map(|t| if t < PAGE { 5 } else { 2 }).collect();
    let off: Vec<i32> = (0..n).map(|t| (t % PAGE) as i32).collect();
    (gpu.column(&page), gpu.column(&off))
}

#[test]
fn a_kv_append_lands_the_same_rows_over_spread_pages() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (heads, d, n) = (2usize, 64usize, 20usize);
    for dtype in [Bf16, E4m3] {
        for hnd in [false, true] {
            for shared in [false, true] {
                let values = (!shared).then_some((dtype, heads * d));
                let p = paged(
                    &gpu,
                    1,
                    dtype,
                    heads * d,
                    values,
                    [40, 30],
                    hnd.then_some(d),
                );
                let k = rows(&gpu, 2, n, heads * d);
                let v = rows(&gpu, 3, n, heads * d);
                let indptr = gpu.column(&[0i32, n as i32]);
                let (page, off) = written(&gpu, n);
                let what = format!("kv_append {dtype:?} hnd {hnd} shared {shared}");
                agrees(&gpu, &what, &[p], &[vec![page]], &[], |_, pools, tables| {
                    let k = RaggedTensor { data: k, indptr };
                    let fired = if shared {
                        kernels_rocm::attn::kv_append_shared(&ctx, k, &pools[0], tables[0][0], off)
                    } else {
                        kernels_rocm::attn::kv_append(&ctx, k, v, &pools[0], tables[0][0], off)
                    };
                    fired.expect("the append fires");
                });
            }
        }
    }
}

#[test]
fn a_fused_qkv_write_lands_the_same_rows_over_spread_pages() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    // The wavefront kernel (bf16 and e4m3, either layout) and the
    // block-a-head one.
    for (d, dtype, hnd) in [
        (128usize, Bf16, false),
        (64, Bf16, true),
        (128, E4m3, false),
        (64, E4m3, true),
        (96, Bf16, true),
    ] {
        let (qh, kvh, n) = (4usize, 2usize, 20usize);
        let p = paged(
            &gpu,
            4,
            dtype,
            kvh * d,
            Some((dtype, kvh * d)),
            [40, 30],
            hnd.then_some(d),
        );
        let packed = rows(&gpu, 5, n, (qh + 2 * kvh) * d);
        let positions: Vec<i32> = (0..n as i32).map(|t| 3 * t + 1).collect();
        let positions = gpu.column(&positions);
        let weight = rows(&gpu, 6, 1, d);
        let (page, off) = written(&gpu, n);
        let q = gpu.zeros(n as u32, (qh * d) as u32, Bf16);
        let what = format!("qkv write {d} {dtype:?} hnd {hnd}");
        agrees(
            &gpu,
            &what,
            &[p],
            &[vec![page]],
            &[out(&q)],
            |_, pools, tables| {
                let mut q = q;
                kernels_rocm::attn::qkv_append::qkv_fused_qknorm_rope_vnorm_write(
                    &ctx,
                    packed,
                    Some(positions),
                    Some(weight),
                    Some(weight),
                    true,
                    1e-6,
                    &pools[0],
                    tables[0][0],
                    off,
                    kvh as u32,
                    d as u32,
                    1e6,
                    d as u32,
                    d as u32 / 2,
                    true,
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
fn a_gated_prologue_lands_the_same_rows_over_spread_pages() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (hq, hk, d, n) = (8usize, 2usize, 128usize, 20usize);
    for hnd in [false, true] {
        let p = paged(
            &gpu,
            7,
            Bf16,
            hk * d,
            Some((Bf16, hk * d)),
            [40, 30],
            hnd.then_some(d),
        );
        let qg = rows(&gpu, 8, n, 2 * hq * d);
        let k = rows(&gpu, 9, n, hk * d);
        let v = rows(&gpu, 10, n, hk * d);
        let w = rows(&gpu, 11, 1, d);
        let positions: Vec<i32> = (0..n as i32).map(|t| 5 * t + 2).collect();
        let positions = gpu.column(&positions);
        let (page, off) = written(&gpu, n);
        let q = gpu.zeros(n as u32, (hq * d) as u32, Bf16);
        let gate = gpu.zeros(n as u32, (hq * d) as u32, Bf16);
        let split = gpu.zeros(n as u32, (hq * d) as u32, Bf16);
        let what = format!("gated prologue hnd {hnd}");
        agrees(
            &gpu,
            &what,
            &[p],
            &[vec![page]],
            &[out(&q), out(&gate), out(&split)],
            |_, pools, tables| {
                let (mut q, mut gate, mut split) = (q, gate, split);
                let covered = kernels_rocm::attn::qkv_append::q_gate_k_norm_rope_append(
                    &ctx,
                    qg,
                    k,
                    v,
                    w,
                    w,
                    positions,
                    1e-6,
                    1e-6,
                    &pools[0],
                    tables[0][0],
                    off,
                    d as u32,
                    64,
                    64,
                    1e7,
                    &mut q,
                    &mut gate,
                    &mut split,
                )
                .expect("the prologue fires");
                assert!(covered, "the fused prologue covers this shape");
            },
        );
    }
}

#[test]
fn an_mla_append_lands_the_same_rows_over_spread_pages() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let n = 20usize;
    let indptr = gpu.column(&[0i32, 12, n as i32]);
    // bf16 latent and rope planes; fp8 rows of the 512 latent and its rope.
    for (rank, rope, fp8) in [(128usize, 64usize, false), (512, 64, true)] {
        let p = if fp8 {
            paged(&gpu, 12, U8, rank + rope, None, [40, 30], None)
        } else {
            paged(&gpu, 12, Bf16, rank, Some((Bf16, rope)), [40, 30], None)
        };
        let p = KvPool {
            values: if fp8 { Tensor::ABSENT } else { p.values },
            ..p
        }
        .with_packed_stride();
        let ckv = rows(&gpu, 13, n, rank);
        let kpe = rows(&gpu, 14, n, rope);
        let what = format!("mla append rank {rank} fp8 {fp8}");
        agrees(&gpu, &what, &[p], &[], &[], |_, pools, _| {
            kernels_rocm::attn::mla::kv_append(
                &ctx,
                RaggedTensor { data: ckv, indptr },
                kpe,
                &pools[0],
                Tensor::ABSENT,
                Tensor::ABSENT,
            )
            .expect("the append fires");
        });
    }
}

#[test]
fn an_index_key_append_lands_the_same_rows_over_spread_pages() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (n, width) = (20usize, 128usize);
    let p = paged(&gpu, 15, Bf16, width, None, [40, 30], None);
    let p = KvPool {
        values: Tensor::ABSENT,
        ..p
    }
    .with_packed_stride();
    let k = rows(&gpu, 16, n, width);
    let indptr = gpu.column(&[0i32, 7, n as i32]);
    agrees(&gpu, "index key append", &[p], &[], &[], |_, pools, _| {
        kernels_rocm::attn::index::kv_append(
            &ctx,
            RaggedTensor { data: k, indptr },
            &pools[0],
            Tensor::ABSENT,
            Tensor::ABSENT,
        )
        .expect("the append fires");
    });
}

#[test]
fn a_pool_entry_store_lands_the_same_rows_over_spread_pages() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let d = 64usize;
    // An entry a position's row, and compact (four positions' rows a row).
    for div in [1usize, 4] {
        let p = paged(&gpu, 17, Bf16, d / div, None, [40, 30], None);
        let p = KvPool {
            values: Tensor::ABSENT,
            ..p
        }
        .with_packed_stride();
        let bpos = gpu.column(&[3i32, 35, -1, 27, 11]);
        let breq = gpu.column(&[0i32, 0, 0, 1, 1]);
        let entries = rows(&gpu, 18, 5, d);
        let what = format!("pool entry store div {div}");
        agrees(&gpu, &what, &[p], &[], &[], |_, pools, _| {
            pool::kv_append(
                &ctx,
                entries,
                bpos,
                breq,
                &pools[0],
                Tensor::ABSENT,
                Tensor::ABSENT,
            )
            .expect("the store fires");
        });
    }
}

/// A compressor state as a pool: its kv and score planes `width` wide.
fn state(gpu: &Gpu, seed: u64, width: usize) -> KvPool {
    paged(gpu, seed, Bf16, width, Some((Bf16, width)), [40, 30], None)
}

/// A state pool's page stride in its elements, as the kernels take it.
fn state_stride(p: &KvPool) -> i64 {
    assert_eq!(p.page_stride[0], p.page_stride[1]);
    (p.page_stride[0] / 2) as i64
}

#[test]
fn a_compressor_state_lands_and_gathers_the_same_over_spread_pages() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (d, ratio, n) = (64usize, 4usize, 20usize);
    for coff in [1usize, 2] {
        let width = coff * d;
        let p = state(&gpu, 19, width);
        let kv = rows(&gpu, 20, n, width);
        let sc = rows(&gpu, 21, n, width);
        let (page, off) = written(&gpu, n);
        let what = format!("state write coff {coff}");
        agrees(&gpu, &what, &[p], &[vec![page]], &[], |_, pools, tables| {
            let s = &pools[0];
            pool::state_write(
                &ctx,
                kv,
                sc,
                s,
                tables[0][0],
                off,
                d as u32,
                ratio as u32,
                s.keys,
                s.values,
                state_stride(s),
            )
            .expect("the state write fires");
        });

        let ape = gpu.tensor(
            &bf16s(ratio * width, 22, 0.5),
            ratio as u32,
            width as u32,
            F32,
        );
        let bpos = gpu.column(&[3i32, 39, -1, 11, 27]);
        let breq = gpu.column(&[0i32, 0, 0, 1, 1]);
        let entries = gpu.zeros(5, d as u32, Bf16);
        let what = format!("state gather coff {coff}");
        agrees(&gpu, &what, &[p], &[], &[out(&entries)], |_, pools, _| {
            let s = &pools[0];
            let mut entries = entries;
            pool::gather(
                &ctx,
                bpos,
                breq,
                s,
                d as u32,
                ratio as u32,
                s.keys,
                s.values,
                state_stride(s),
                ape,
                &mut entries,
            )
            .expect("the gather fires");
        });
    }
}

/// The key envelopes an append refreshes, over a pool whose pages lie a
/// stride apart (garbage between): the same envelopes, and the same rows.
#[test]
fn an_appends_envelopes_are_the_same_over_strided_pages() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (heads, d, n) = (2usize, 64usize, 20usize);
    let row = heads * d;
    let p = paged(&gpu, 23, Bf16, row, Some((Bf16, row)), [40, 30], None);
    // Each page's rows `GAP` rows of garbage past the page before's.
    const GAP: usize = 3;
    let pages = 9;
    let page_bytes = PAGE * row * 2;
    let stride = page_bytes + GAP * row * 2;
    let spread = |plane: Tensor, seed: u64| {
        let packed: Vec<u8> = gpu.down(plane.ptr, pages * page_bytes);
        let mut image = noise(seed, pages * stride);
        for page in 0..pages {
            image[page * stride..page * stride + page_bytes]
                .copy_from_slice(&packed[page * page_bytes..(page + 1) * page_bytes]);
        }
        (gpu.up(&image), image)
    };
    let ((kb, k_image), (vb, v_image)) = (spread(p.keys, 24), spread(p.values, 25));
    let env = |p: &KvPool| KvPool {
        env_min: gpu.zeros((pages * heads) as u32, d as u32, Bf16),
        env_max: gpu.zeros((pages * heads) as u32, d as u32, Bf16),
        has_envelopes: true,
        ..*p
    };
    let packed = env(&p);
    let strided = env(&KvPool {
        keys: Tensor { ptr: kb, ..p.keys },
        values: Tensor {
            ptr: vb,
            ..p.values
        },
        page_stride: [stride as u64; 2],
        ..p
    });
    let k = rows(&gpu, 26, n, row);
    let v = rows(&gpu, 27, n, row);
    let indptr = gpu.column(&[0i32, 12, n as i32]);
    let (page, off) = written(&gpu, n);
    for pool in [&packed, &strided] {
        kernels_rocm::attn::kv_append(&ctx, RaggedTensor { data: k, indptr }, v, pool, page, off)
            .expect("the append fires");
    }
    let bytes = |t: &Tensor| gpu.down::<u8>(t.ptr, (t.elements() * 2) as usize);
    assert!(
        bytes(&packed.env_min).iter().any(|&b| b != 0),
        "the append refreshed no envelope, so the check checks nothing"
    );
    assert!(
        bytes(&packed.env_min) == bytes(&strided.env_min)
            && bytes(&packed.env_max) == bytes(&strided.env_max),
        "the envelopes over strided pages are not the packed pool's"
    );
    for ((plane, base), image) in [(packed.keys, kb), (packed.values, vb)]
        .into_iter()
        .zip([k_image, v_image])
    {
        let want: Vec<u8> = gpu.down(plane.ptr, pages * page_bytes);
        let now: Vec<u8> = gpu.down(base, pages * stride);
        for page in 0..pages {
            let at = page * stride;
            assert!(
                now[at..at + page_bytes] == want[page * page_bytes..(page + 1) * page_bytes],
                "page {page}'s rows over strided pages are not the packed pool's"
            );
            assert!(
                now[at + page_bytes..at + stride] == image[at + page_bytes..at + stride],
                "a write landed past page {page}'s rows"
            );
        }
    }
}
