//! Device time of the small kernels a Qwen 3.5 decode step fires once per
//! layer, at one row, each replayed from a graph.
//! `cargo test --release --features rocm --test bench_decode_small -- --ignored --nocapture`

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::attn::ssm;
use kernels_rocm::elemwise::norm;
use kernels_rocm::{RecurrentPool, Tensor};

#[test]
#[ignore = "benchmark"]
fn bench_decode_small() {
    for rows in [1u32, 64, 256] {
        bench_at(rows);
    }
}

fn bench_at(n: u32) {
    let Some(gpu) = Gpu::open() else { return };
    let r = |name: &str, us: f64| eprintln!("small rows={n:3} {name:28}: {us:6.1} us");
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
    let f32t = |rows: u32, width: u32, seed: u64| {
        gpu.tensor(
            &values((rows * width) as usize, seed, 0.5),
            rows,
            width,
            Dtype::F32,
        )
    };

    // The residual stream's add + norm.
    let x = bf(n, 1024, 1);
    let mut y = bf(n, 1024, 2);
    let w = bf(1, 1024, 3);
    let mut out = gpu.zeros(n, 1024, Dtype::Bf16);
    r(
        "residual_add_rmsnorm",
        gpu.time_graph(50, || {
            norm::residual_add_rmsnorm(&gpu.ctx(), x, &mut y, w, true, 1e-6, &mut out).unwrap()
        }),
    );
    r(
        "mul_scalar (floor)",
        gpu.time_graph(50, || norm::mul_scalar(&gpu.ctx(), 1.0, &mut out).unwrap()),
    );
    if n > 1 {
        // The batch step's traffic as a plain read and write of 32 MiB.
        let mut big = gpu.zeros(n * 16, 128 * 128, Dtype::Bf16);
        r(
            "mul_scalar 32 MiB",
            gpu.time_graph(20, || norm::mul_scalar(&gpu.ctx(), 1.0, &mut big).unwrap()),
        );
    }

    // The GDN decode chain: conv, gates, step, gated norm.
    let (kh, vh, kd, vd) = (16u32, 16u32, 128u32, 128u32);
    let c = 2 * kh * kd + vh * vd;
    let slab = bf(n + 1, vh * kd * vd, 4);
    let conv_slab = bf((n + 1) * 4, c, 5);
    let pool = RecurrentPool {
        slab,
        slot_ids: gpu.column(&(0..n as i32).collect::<Vec<_>>()),
        slot_stride_elems: i64::from(vh * kd * vd),
        conv_slab,
        conv_stride: i64::from(4 * c),
        write_state: true,
        write_state_mask: Tensor::ABSENT,
        commit_len: Tensor::ABSENT,
        begin_at: Tensor::ABSENT,
        fused_decay: false,
    };
    let qkv = bf(n, c, 6);
    let cw = bf(c, 4, 7);
    let mut qkv_out = gpu.zeros(n, c, Dtype::Bf16);
    r(
        "ssm_causal_conv1d",
        gpu.time_graph(50, || {
            ssm::causal_conv1d(&gpu.ctx(), qkv, cw, &pool, 4, 1, &mut qkv_out).unwrap()
        }),
    );
    let ba = bf(n, 2 * vh, 8);
    let dt = f32t(1, vh, 9);
    let alog = f32t(1, vh, 10);
    let mut gates = gpu.zeros(n, 2 * vh, Dtype::F32);
    r(
        "ssm_gdn_prep",
        gpu.time_graph(50, || {
            ssm::gdn_prep(&gpu.ctx(), ba, dt, alog, &mut gates).unwrap()
        }),
    );
    let mut core = gpu.zeros(n, vh * vd, Dtype::F32);
    r(
        "ssm_gated_delta (step)",
        gpu.time_graph(50, || {
            ssm::gated_delta(
                &gpu.ctx(),
                qkv_out,
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
        }),
    );
    let z = bf(n, vh * vd, 11);
    let nw = f32t(1, vd, 12);
    let mut o = gpu.zeros(n, vh * vd, Dtype::Bf16);
    r(
        "rmsnorm_gated",
        gpu.time_graph(50, || {
            norm::rmsnorm_gated(&gpu.ctx(), core, z, nw, vd, 1e-6, false, &mut o).unwrap()
        }),
    );
}
