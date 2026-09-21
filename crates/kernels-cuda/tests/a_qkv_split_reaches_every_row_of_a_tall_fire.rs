#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::layout;
use kernels_cuda::tensor::Tensor;

const TALL: u32 = 65_536;

#[test]
fn a_qkv_split_reaches_every_row_of_a_tall_fire() {
    common::arm_cache();
    let (q_width, kv_width) = (8u32, 4u32);
    let stride = (q_width + 2 * kv_width) as usize;

    let mut lcg = Lcg::seeded(0x51);
    let (packed, _) = lcg.row(TALL as usize * stride);

    let mut gpu = Gpu::open();
    let packed_at = gpu.up(&packed);
    let q_at = gpu.zeros(TALL as usize * q_width as usize * 2);
    let k_at = gpu.zeros(TALL as usize * kv_width as usize * 2);
    let v_at = gpu.zeros(TALL as usize * kv_width as usize * 2);

    let src = Tensor::new(packed_at, TALL, stride as u32, Dtype::Bf16);
    let mut q = Tensor::new(q_at, TALL, q_width, Dtype::Bf16);
    let mut k = Tensor::new(k_at, TALL, kv_width, Dtype::Bf16);
    let mut v = Tensor::new(v_at, TALL, kv_width, Dtype::Bf16);
    layout::split_qkv(&gpu.ctx(), src, q_width, kv_width, &mut q, &mut k, &mut v)
        .expect("a tall split is a split");
    gpu.sync();

    let q_out: Vec<u16> = gpu.down(q_at, TALL as usize * q_width as usize);
    let k_out: Vec<u16> = gpu.down(k_at, TALL as usize * kv_width as usize);
    let v_out: Vec<u16> = gpu.down(v_at, TALL as usize * kv_width as usize);

    let mut first_wrong = None;
    for n in 0..TALL as usize {
        let row = &packed[n * stride..(n + 1) * stride];
        let q_ok =
            q_out[n * q_width as usize..(n + 1) * q_width as usize] == row[..q_width as usize];
        let k_ok = k_out[n * kv_width as usize..(n + 1) * kv_width as usize]
            == row[q_width as usize..(q_width + kv_width) as usize];
        let v_ok = v_out[n * kv_width as usize..(n + 1) * kv_width as usize]
            == row[(q_width + kv_width) as usize..];
        if !(q_ok && k_ok && v_ok) {
            first_wrong = Some(n);
            break;
        }
    }
    assert_eq!(
        first_wrong, None,
        "the split stopped at row {:?} of {TALL}; `gridDim.y` caps at 65535, so a \
         split whose rows ride `y` silently drops everything past it",
        first_wrong
    );
}
