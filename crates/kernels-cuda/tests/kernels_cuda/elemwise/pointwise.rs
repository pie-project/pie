#![cfg(feature = "cuda")]

use crate::common::{Gpu, Lcg, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::elemwise::pointwise;
use kernels_cuda::tensor::Tensor;

const ROWS: usize = 9;
const WIDTH: usize = 96;
const N: usize = ROWS * WIDTH;

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

fn gelu_tanh(v: f32) -> f32 {
    const C: f32 = 0.797_884_6;
    0.5 * v * (1.0 + (C * v.mul_add(0.044_715 * v * v, v)).tanh())
}

fn close(got: f32, want: f32, what: &str, at: usize) {
    assert!(
        (got - want).abs() <= 1e-6 + want.abs() / 128.0,
        "{what} at {at}: {got} against {want}"
    );
}

#[test]
fn the_bare_binary_and_activation_ops_answer_the_reference_every_case() {
    the_bare_ops_answer_the_reference();
    each_may_write_the_rectangle_it_read();
    div_mod_and_sqrt_answer_the_reference_in_f32();
}

fn the_bare_ops_answer_the_reference() {
    let mut lcg = Lcg::seeded(0xba4e);
    let (x_raw, x) = lcg.row(N);
    let (y_raw, y) = lcg.row(N);

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let y_at = gpu.up(&y_raw);
    let out = [(); 5].map(|()| gpu.zeros(N * 2));
    let ctx = gpu.ctx();
    let rect = |at: u64| Tensor::new(at, ROWS as u32, WIDTH as u32, Dtype::Bf16);

    pointwise::add(&ctx, rect(x_at), rect(y_at), &mut rect(out[0])).expect("fires");
    pointwise::mul(&ctx, rect(x_at), rect(y_at), &mut rect(out[1])).expect("fires");
    pointwise::silu(&ctx, rect(x_at), &mut rect(out[2])).expect("fires");
    pointwise::gelu_tanh(&ctx, rect(x_at), &mut rect(out[3])).expect("fires");
    pointwise::tanh(&ctx, rect(x_at), &mut rect(out[4])).expect("fires");
    // The residual add into the stream it reads, eight lanes a thread.
    let acc = gpu.up(&y_raw);
    pointwise::residual_add(&ctx, rect(x_at), &mut rect(acc)).expect("fires");
    gpu.sync();

    let got: Vec<Vec<u16>> = out.iter().map(|&at| gpu.down(at, N)).collect();
    let summed: Vec<u16> = gpu.down(acc, N);
    for at in 0..N {
        close(from_bf16(summed[at]), x[at] + y[at], "residual_add", at);
        close(from_bf16(got[0][at]), x[at] + y[at], "add", at);
        close(from_bf16(got[1][at]), x[at] * y[at], "mul", at);
        close(from_bf16(got[2][at]), silu(x[at]), "silu", at);
        close(from_bf16(got[3][at]), gelu_tanh(x[at]), "gelu_tanh", at);
        close(from_bf16(got[4][at]), x[at].tanh(), "tanh", at);
    }
}

fn each_may_write_the_rectangle_it_read() {
    let mut lcg = Lcg::seeded(0xa11a52);
    let (x_raw, x) = lcg.row(N);
    let (y_raw, y) = lcg.row(N);

    let mut gpu = Gpu::open();
    let sum = gpu.up(&x_raw);
    let gated = gpu.up(&x_raw);
    let y_at = gpu.up(&y_raw);
    let ctx = gpu.ctx();
    let rect = |at: u64| Tensor::new(at, ROWS as u32, WIDTH as u32, Dtype::Bf16);

    pointwise::add(&ctx, rect(sum), rect(y_at), &mut rect(sum)).expect("fires");
    pointwise::tanh(&ctx, rect(gated), &mut rect(gated)).expect("fires");
    gpu.sync();

    let got_sum: Vec<u16> = gpu.down(sum, N);
    let got_tanh: Vec<u16> = gpu.down(gated, N);
    for at in 0..N {
        assert_eq!(got_sum[at], to_bf16(x[at] + y[at]), "add in place at {at}");
        assert_eq!(got_tanh[at], to_bf16(x[at].tanh()), "tanh in place at {at}");
    }
}

/// Position arithmetic: f32 quotients, floored moduli and square roots.
fn div_mod_and_sqrt_answer_the_reference_in_f32() {
    let x: Vec<f32> = (0..N).map(|i| i as f32 * 0.75 - 200.0).collect();
    let y: Vec<f32> = (0..N)
        .map(|i| {
            let m = (i % 7) as f32 + 1.0;
            if i % 3 == 0 { -m } else { m }
        })
        .collect();
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x);
    let y_at = gpu.up(&y);
    let out = [(); 3].map(|()| gpu.zeros(N * 4));
    let sq = gpu.up(&x.iter().map(|v| v.abs()).collect::<Vec<f32>>());
    let ctx = gpu.ctx();
    let rect = |at: u64| Tensor::new(at, ROWS as u32, WIDTH as u32, Dtype::F32);

    pointwise::div(&ctx, rect(x_at), rect(y_at), &mut rect(out[0])).expect("fires");
    pointwise::modulo(&ctx, rect(x_at), rect(y_at), &mut rect(out[1])).expect("fires");
    pointwise::sqrt(&ctx, rect(sq), &mut rect(out[2])).expect("fires");
    gpu.sync();

    let got: Vec<Vec<f32>> = out.iter().map(|&at| gpu.down(at, N)).collect();
    for at in 0..N {
        let (a, b) = (x[at], y[at]);
        close(got[0][at], a / b, "div", at);
        let want = a - b * (a / b).floor();
        assert!(
            (got[1][at] - want).abs() <= 1e-4,
            "mod at {at}: {} against {want}",
            got[1][at]
        );
        assert!(
            want == 0.0 || want.signum() == b.signum(),
            "the floored modulo takes the divisor's sign"
        );
        close(got[2][at], a.abs().sqrt(), "sqrt", at);
    }
}
