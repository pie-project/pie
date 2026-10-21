#![cfg(target_vendor = "apple")]
//! A per-head norm pair (a layer's query and key heads in one launch) lands
//! each norm's own launch's rows, bit for bit.

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use kernels_metal::elemwise::norm;
use poem_ir::Dtype;

fn noise(at: u64) -> u8 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 40) as u8
}

fn bf16(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    [(bits >> 16) as u8, (bits >> 24) as u8]
}

#[test]
fn a_norm_pair_lands_each_norms_rows() {
    let Ok(device) = Context::bind() else {
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let upload = |bytes: &[u8]| {
        let mut b = Buffer::zeroed(&device, bytes.len().max(16) as u64).expect("a plane");
        b.write(0, bytes).expect("write");
        b
    };
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("a handle");
    let plane = |n: u32, salt: u64, scale: f32| -> Vec<u8> {
        (0..n)
            .flat_map(|at| bf16((f32::from(noise(u64::from(at) ^ salt)) - 127.0) * scale))
            .collect()
    };
    // (head width, query heads, key heads, rows): the 35B's attention and its draft's
    for (d, qh, kh, m) in [
        (256u32, 16u32, 2u32, 8u32),
        (128, 32, 8, 8),
        (128, 32, 8, 5),
    ] {
        let (q, k) = (
            upload(&plane(m * qh * d, 0x11, 0.03)),
            upload(&plane(m * kh * d, 0x22, 0.05)),
        );
        let (wq, wk) = (
            upload(&plane(d, 0x33, 0.004)),
            upload(&plane(d, 0x44, 0.004)),
        );
        let out = |n: u32| upload(&vec![0x5a; (m * n * d * 2) as usize]);
        let (q_own, k_own, q_one, k_one) = (out(qh), out(kh), out(qh), out(kh));
        let t = |b: &Buffer, heads: u32| Tensor::new(bind(b), m, heads * d, Dtype::Bf16);
        let w = |b: &Buffer| Tensor::new(bind(b), 1, d, Dtype::Bf16);
        let frame = device.frame().expect("a frame");
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        norm::rmsnorm_per_head_plus_one(&sink, t(&q, qh), w(&wq), d, 1e-6, t(&q_own, qh))
            .expect("query");
        norm::rmsnorm_per_head_plus_one(&sink, t(&k, kh), w(&wk), d, 1e-6, t(&k_own, kh))
            .expect("key");
        norm::rmsnorm_per_head_plus_one_pair(
            &sink,
            t(&q, qh),
            w(&wq),
            t(&q_one, qh),
            t(&k, kh),
            w(&wk),
            t(&k_one, kh),
            d,
            1e-6,
        )
        .expect("pair");
        frame.commit().expect("the commit");
        let read = |b: &Buffer| handles.read(bind(b), b.bytes()).expect("read");
        assert!(
            read(&q_own) == read(&q_one),
            "d={d} M={m}: the pair lands other query rows"
        );
        assert!(
            read(&k_own) == read(&k_one),
            "d={d} M={m}: the pair lands other key rows"
        );
    }
}
