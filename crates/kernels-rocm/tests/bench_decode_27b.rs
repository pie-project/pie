//! Device time of one Qwen3.8-27B decode step's kernels by layer kind, at
//! the batch sizes a serving run reaches.
//! `cargo test --release --features rocm --test bench_decode_27b -- --ignored --nocapture`

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::attn::ssm;
use kernels_rocm::linear::gemm;
use kernels_rocm::{RecurrentPool, Tensor};

/// (name, n, k, per step)
const GEMMS: [(&str, u32, u32, u32); 8] = [
    ("gdn_in_qkvz", 16384, 5120, 48),
    ("gdn_in_ba", 96, 5120, 48),
    ("gdn_out", 5120, 6144, 48),
    ("attn_q_gate", 12288, 5120, 16),
    ("attn_kv", 2048, 5120, 16),
    ("attn_o", 5120, 6144, 16),
    ("mlp_gate_up", 34816, 5120, 64),
    ("mlp_down", 5120, 17408, 64),
];

#[test]
#[ignore = "benchmark"]
fn bench_decode_27b() {
    let Some(gpu) = Gpu::open() else { return };
    for m in [1u32, 64, 128, 256] {
        let mut total = 0.0;
        for (name, n, k, per) in GEMMS {
            // Copies enough to overflow the 256 MiB last-level cache, cycled,
            // so each launch streams its weight from memory as serving does.
            let copies = (512u64 << 20)
                .div_ceil(u64::from(n) * u64::from(k) * 2)
                .max(2) as usize;
            let host = values((n * k) as usize, 2, 0.05);
            let ws: Vec<_> = (0..copies)
                .map(|_| gpu.tensor(&host, n, k, Dtype::Bf16))
                .collect();
            let a = gpu.tensor(&values((m * k) as usize, 1, 1.0), m, k, Dtype::Bf16);
            let mut y = gpu.zeros(m, n, Dtype::Bf16);
            let mut at = 0usize;
            let us = gpu.time_graph(10 * copies as u32, || {
                at = (at + 1) % copies;
                gemm::matmul(&gpu.ctx(), a, ws[at], &mut y).unwrap()
            });
            let gbs = f64::from(n) * f64::from(k) * 2.0 / us / 1e3;
            total += us * f64::from(per);
            eprintln!("m={m:3} {name:12} n={n:6} k={k:5}: {us:8.1} us {gbs:6.0} GB/s x{per}");
        }
        let gdn = gdn_step(&gpu, m);
        total += gdn * 48.0;
        eprintln!("m={m:3} gdn step (vh=48)            : {gdn:8.1} us x48");
        eprintln!(
            "m={m:3} step total (gemm + gdn)     : {:8.2} ms",
            total / 1e3
        );
    }
}

fn gdn_step(gpu: &Gpu, n: u32) -> f64 {
    let bf = |rows: u32, width: u32, seed: u64| {
        gpu.tensor(
            &values((rows * width) as usize, seed, 0.5)
                .into_iter()
                .map(round_bf16)
                .collect::<Vec<_>>(),
            rows,
            width,
            Dtype::Bf16,
        )
    };
    let (kh, vh, kd, vd) = (16u32, 48u32, 128u32, 128u32);
    let c = 2 * kh * kd + vh * vd;
    let pool = RecurrentPool {
        slab: bf(n + 1, vh * kd * vd, 4),
        slot_ids: gpu.column(&(0..n as i32).collect::<Vec<_>>()),
        slot_stride_elems: i64::from(vh * kd * vd),
        conv_slab: bf((n + 1) * 4, c, 5),
        conv_stride: i64::from(4 * c),
        write_state: true,
        write_state_mask: Tensor::ABSENT,
        commit_len: Tensor::ABSENT,
        begin_at: Tensor::ABSENT,
        fused_decay: false,
    };
    let qkv = bf(n, c, 6);
    let gates = gpu.tensor(
        &values((n * 2 * vh) as usize, 9, 0.5),
        n,
        2 * vh,
        Dtype::F32,
    );
    let mut core = gpu.zeros(n, vh * vd, Dtype::F32);
    gpu.time_graph(10, || {
        ssm::gated_delta(
            &gpu.ctx(),
            qkv,
            Tensor::ABSENT,
            gates,
            &pool,
            kh,
            vh,
            kd,
            vd,
            &mut core,
        )
        .unwrap()
    })
}
