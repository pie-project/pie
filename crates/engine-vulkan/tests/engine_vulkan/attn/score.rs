//! Attention mass capture on the first Vulkan device, against an f64 host
//! reference: two requests over a paged pool with pages out of order, the
//! mean causal softmax of each request's last query rows landed per key in
//! its lane's planes of an f32 slab.

use kernels_slang::{KvPool, RaggedTensor, Tensor};
use poem_ir::Dtype;

use crate::gpu::{Gpu, close, noise};

#[test]
fn score_capture_answers_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let (qh, kvh, hd, page) = (4usize, 2usize, 64usize, 4usize);
    let scale = 0.125f64;
    // Request 0 reads pages [2, 0] with 3 query rows at positions 5..8;
    // request 1 reads page [1] with 1 query row at position 2.
    let page_indices = [2u32, 0, 1];
    let page_indptr = [0u32, 2, 3];
    let qo_indptr = [0i32, 3, 4];
    let positions = [5i32, 6, 7, 2];
    let (observe, kv_max, lane_offset, plane_stride) = (2usize, 8usize, 1usize, 4usize);
    let slots = 3 * page;
    let row = kvh * hd;
    let (keys, kr) = gpu.up(&noise(1, slots * row));
    let (q, qr) = gpu.up(&noise(2, 4 * qh * hd));
    let kr: Vec<f64> = kr.iter().map(|v| f64::from(*v)).collect();
    let qr: Vec<f64> = qr.iter().map(|v| f64::from(*v)).collect();
    let slab_rows = (lane_offset + 2) * plane_stride;
    let mut want = vec![7.0; slab_rows * kv_max];
    for r in 0..2 {
        let pages = &page_indices[page_indptr[r] as usize..page_indptr[r + 1] as usize];
        let capacity = pages.len() * page;
        let (lo, hi) = (qo_indptr[r] as usize, qo_indptr[r + 1] as usize);
        let rows = observe.min(hi - lo);
        for h in 0..qh {
            let base = ((lane_offset + r) * plane_stride + h) * kv_max;
            want[base..base + kv_max].fill(0.0);
            let kh = h / (qh / kvh);
            for w in 0..rows {
                let qi = hi - rows + w;
                let limit = (positions[qi] as usize + 1).min(capacity);
                let qv = &qr[(qi * qh + h) * hd..][..hd];
                let s: Vec<f64> = (0..limit)
                    .map(|j| {
                        let slot = pages[j / page] as usize * page + j % page;
                        qv.iter()
                            .zip(&kr[slot * row + kh * hd..][..hd])
                            .map(|(a, b)| a * b)
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let top = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = s.iter().map(|v| (v - top).exp()).sum();
                for (j, v) in s.iter().enumerate() {
                    if j < kv_max {
                        want[base + j] += (v - top).exp() / sum / rows as f64;
                    }
                }
            }
        }
    }
    let t = |b: u32, r: usize, w: usize, d: Dtype| Tensor::new(b, r as u32, w as u32, d);
    let pi = gpu.up_u32(&page_indices);
    let pp = gpu.up_u32(&page_indptr);
    let qp = gpu.up_i32(&qo_indptr);
    let pos = gpu.up_i32(&positions);
    let pool = KvPool {
        keys: t(keys, slots, row, Dtype::Bf16),
        values: t(keys, slots, row, Dtype::Bf16),
        page_indices: t(pi, 3, 1, Dtype::U32),
        page_indptr: t(pp, 3, 1, Dtype::U32),
        page_size: page as i32,
        seq_stride: row as u64,
        head_stride: hd as u64,
    };
    let qrag = RaggedTensor {
        data: t(q, 4, qh * hd, Dtype::Bf16),
        indptr: t(qp, 3, 1, Dtype::I32),
    };
    let post = t(pos, 4, 1, Dtype::I32);
    let slab = gpu.up_f32(&vec![7.0; slab_rows * kv_max]);
    let sl = t(slab, slab_rows, kv_max, Dtype::F32);
    let args = (
        hd as u32,
        kvh as u32,
        scale as f32,
        observe as u32,
        lane_offset as u32,
        plane_stride as u32,
    );
    gpu.run(|c| {
        kernels_slang::attn::score::capture(
            c,
            qrag,
            post,
            &pool,
            None,
            args.0,
            args.1,
            args.2,
            args.3,
            args.4,
            args.5,
            0,
            kv_max as u32,
            2,
            sl,
        )
    });

    // Lane 0's planes are not this capture's and keep their sevens.
    close(
        &gpu.down_f32(slab, slab_rows * kv_max),
        &want,
        "score capture",
    );
}
