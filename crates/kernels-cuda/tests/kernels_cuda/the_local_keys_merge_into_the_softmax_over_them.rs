#![cfg(feature = "cuda")]

//! `attention.local_lse` over each row's own extra keys, merged by
//! `attention.merge_lse`, is the softmax attention over those keys: the MTP
//! chain's steps read the keys of the steps before them this way.

use crate::common::{Gpu, Lcg, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::attn;
use kernels_cuda::tensor::Tensor;

#[test]
fn two_local_keys_merge_into_their_softmax() {
    let (rows, heads, kv_heads, d) = (3usize, 8usize, 2usize, 128usize);
    let group = heads / kv_heads;
    let scale = 1.0 / (d as f32).sqrt();
    let mut lcg = Lcg::seeded(0x10ca1);
    let mut draw = |n: usize| -> (Vec<u16>, Vec<f32>) {
        let raw: Vec<u16> = (0..n).map(|_| to_bf16(lcg.unit() * 2.0 - 1.0)).collect();
        let f = raw.iter().map(|&b| from_bf16(b)).collect();
        (raw, f)
    };
    let (q_raw, q) = draw(rows * heads * d);
    let (k1_raw, k1) = draw(rows * kv_heads * d);
    let (v1_raw, v1) = draw(rows * kv_heads * d);
    let (k2_raw, k2) = draw(rows * kv_heads * d);
    let (v2_raw, v2) = draw(rows * kv_heads * d);

    let mut gpu = Gpu::open();
    let up = |gpu: &mut Gpu, raw: &[u16], width: usize| {
        Tensor::new(gpu.up(raw), rows as u32, width as u32, Dtype::Bf16)
    };
    let qt = up(&mut gpu, &q_raw, heads * d);
    let k1t = up(&mut gpu, &k1_raw, kv_heads * d);
    let v1t = up(&mut gpu, &v1_raw, kv_heads * d);
    let k2t = up(&mut gpu, &k2_raw, kv_heads * d);
    let v2t = up(&mut gpu, &v2_raw, kv_heads * d);
    let plane = |gpu: &mut Gpu, width: usize, dtype: Dtype, elem: usize| {
        Tensor::new(gpu.zeros(rows * width * elem), rows as u32, width as u32, dtype)
    };
    let mut o1 = plane(&mut gpu, heads * d, Dtype::Bf16, 2);
    let mut l1 = plane(&mut gpu, heads, Dtype::F32, 4);
    let mut o2 = plane(&mut gpu, heads * d, Dtype::Bf16, 2);
    let mut l2 = plane(&mut gpu, heads, Dtype::F32, 4);
    let mut o = plane(&mut gpu, heads * d, Dtype::Bf16, 2);
    let mut l = plane(&mut gpu, heads, Dtype::F32, 4);
    let (h, kh, hd) = (heads as u32, kv_heads as u32, d as u32);
    attn::local_lse(&gpu.ctx(), qt, k1t, v1t, h, kh, hd, scale, &mut o1, &mut l1)
        .expect("the first local key fires");
    attn::local_lse(&gpu.ctx(), qt, k2t, v2t, h, kh, hd, scale, &mut o2, &mut l2)
        .expect("the second local key fires");
    attn::merge_lse(&gpu.ctx(), o1, l1, o2, l2, h, hd, &mut o, &mut l).expect("the merge fires");
    gpu.sync();
    let got: Vec<u16> = gpu.down(o.ptr, rows * heads * d);

    for r in 0..rows {
        for hh in 0..heads {
            let g = hh / group;
            let qr = &q[(r * heads + hh) * d..][..d];
            let at = (r * kv_heads + g) * d;
            let s1: f32 = qr.iter().zip(&k1[at..at + d]).map(|(a, b)| a * b).sum::<f32>() * scale;
            let s2: f32 = qr.iter().zip(&k2[at..at + d]).map(|(a, b)| a * b).sum::<f32>() * scale;
            let m = s1.max(s2);
            let (w1, w2) = ((s1 - m).exp(), (s2 - m).exp());
            for c in 0..d {
                let want = (v1[at + c] * w1 + v2[at + c] * w2) / (w1 + w2);
                let have = from_bf16(got[(r * heads + hh) * d + c]);
                assert!(
                    (have - want).abs() <= 1e-2 + 1e-2 * want.abs(),
                    "row {r} head {hh} column {c}: {have} vs {want}"
                );
            }
        }
    }
}
