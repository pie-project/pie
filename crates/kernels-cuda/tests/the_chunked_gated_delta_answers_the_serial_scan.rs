#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::attn::ssm;
use kernels_cuda::tensor::{RaggedTensor, RecurrentPool, Tensor};

#[test]
fn the_chunked_gated_delta_answers_the_serial_scan() {
    let lens: [usize; 3] = [70, 5, 130];
    let (k_heads, v_heads, k_dim, v_dim) = (2u32, 4u32, 128u32, 128u32);
    let conv_dim = (2 * k_heads * k_dim + v_heads * v_dim) as usize;
    let rows: usize = lens.iter().sum();
    let slot_of: [i32; 3] = [2, 0, 1];
    let stride = (v_heads * k_dim * v_dim) as usize;
    let mut indptr: Vec<i32> = vec![0];
    for len in lens {
        indptr.push(indptr.last().unwrap() + len as i32);
    }
    let mut lcg = Lcg::seeded(0x9d);
    let (qkv_raw, _) = lcg.row(rows * conv_dim);
    let mut gates: Vec<f32> = Vec::with_capacity(rows * 2 * v_heads as usize);
    for _ in 0..rows {
        for _ in 0..v_heads {
            gates.push(-0.15 * (lcg.unit() * 0.5 + 0.5));
        }
        for _ in 0..v_heads {
            gates.push(0.2 + 0.7 * (lcg.unit() * 0.5 + 0.5));
        }
    }
    let (slab_raw, _) = lcg.row(3 * stride);
    let slab_raw: Vec<u16> = slab_raw
        .iter()
        .map(|&v| common::to_bf16(common::from_bf16(v) * 0.05))
        .collect();
    let mut gpu = Gpu::open();
    let qkv_at = gpu.up(&qkv_raw);
    let gates_at = gpu.up(&gates);
    let indptr_at = gpu.up(&indptr);
    let slots_at = gpu.up(&slot_of);
    let mut run = |serial: bool| -> (Vec<f32>, Vec<u16>) {
        let slab_at = gpu.up(&slab_raw);
        let y_at = gpu.zeros(rows * (v_heads * v_dim) as usize * 4);
        let pool = RecurrentPool {
            slab: Tensor::new(slab_at, 3, stride as u32, Dtype::Bf16),
            slot_ids: Tensor::new(slots_at, 3, 1, Dtype::I32),
            slot_stride_elems: stride as i64,
            conv_slab: Tensor::ABSENT,
            conv_stride: 0,
            write_state: true,
            write_state_mask: Tensor::ABSENT,
            commit_len: Tensor::ABSENT,
            begin_at: Tensor::ABSENT,
            fused_decay: false,
        };
        let mut y = Tensor::new(y_at, rows as u32, v_heads * v_dim, Dtype::F32);
        let qkv = RaggedTensor {
            data: Tensor::new(qkv_at, rows as u32, conv_dim as u32, Dtype::Bf16),
            indptr: Tensor::new(indptr_at, 4, 1, Dtype::I32),
        };
        let gates_t = Tensor::new(gates_at, rows as u32, 2 * v_heads, Dtype::F32);
        let fired = if serial {
            ssm::gated_delta_chunked_serial(
                &gpu.ctx(), qkv, Tensor::ABSENT, gates_t, &pool, k_heads, v_heads, k_dim, v_dim, &mut y,
            )
        } else {
            ssm::gated_delta_chunked(
                &gpu.ctx(), qkv, Tensor::ABSENT, gates_t, &pool, k_heads, v_heads, k_dim, v_dim, &mut y,
            )
        };
        fired.expect("the gated delta fires");
        gpu.sync();
        (gpu.down(y_at, rows * (v_heads * v_dim) as usize), gpu.down(slab_at, 3 * stride))
    };
    let (y_serial, state_serial) = run(true);
    let (y_chunked, state_chunked) = run(false);
    let (y_again, state_again) = run(false);
    let unstable = y_chunked.iter().zip(&y_again).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    let unstable_state = state_chunked.iter().zip(&state_again).filter(|(a, b)| a != b).count();
    assert!(
        unstable == 0 && unstable_state == 0,
        "two chunked fires disagree with each other: {unstable} output cells, {unstable_state} state cells"
    );
    let mut worst = 0f32;
    let mut worst_at = 0usize;
    for (at, (&a, &b)) in y_serial.iter().zip(&y_chunked).enumerate() {
        let err = (a - b).abs() / a.abs().max(b.abs()).max(1.0);
        if err > worst {
            worst = err;
            worst_at = at;
        }
    }
    let width = (v_heads * v_dim) as usize;
    assert!(
        worst < 3e-2,
        "output row {} col {}: serial {} against chunked {} (relative error {worst})",
        worst_at / width,
        worst_at % width,
        y_serial[worst_at],
        y_chunked[worst_at]
    );
    let mut worst_state = 0f32;
    for (&a, &b) in state_serial.iter().zip(&state_chunked) {
        let (a, b) = (common::from_bf16(a), common::from_bf16(b));
        let err = (a - b).abs() / a.abs().max(b.abs()).max(1.0);
        worst_state = worst_state.max(err);
    }
    assert!(worst_state < 3e-2, "the final states differ by {worst_state}");
    eprintln!("chunked vs serial: worst output error {worst}, worst state error {worst_state}");
}
