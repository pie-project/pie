mod common;

use common::data;
use engine_cerebras::bench::{Bench, assert_close, round_bf16};
use kernels_cerebras::elemwise::gate;

#[test]
fn a_sigmoid_gate_scales_its_operand_in_place() {
    let (rows, width) = (3u32, 22u32);
    let xs = data((rows * width) as usize, 71);
    let gs: Vec<f32> = data((rows * width) as usize, 72)
        .iter()
        .map(|v| round_bf16(v * 4.0))
        .collect();
    let mut b = Bench::new();
    let x = b.bf16(rows, width, &xs);
    let g = b.bf16(rows, width, &gs);
    if !b.run(|ctx| gate::sigmoid_mul(ctx, g, x)).unwrap() {
        return;
    }
    let want: Vec<f32> = xs
        .iter()
        .zip(&gs)
        .map(|(v, g)| round_bf16(v / (1.0 + (-g).exp())))
        .collect();
    assert_close(&b.read_f32(x), &want, 1e-2, 1e-2);
}

/// `y = gelu_tanh(x)` element for element.
#[test]
fn a_gelu_tanh_bends_each_element() {
    let (rows, width) = (3u32, 10u32);
    let xs: Vec<f32> = common::data((rows * width) as usize, 67)
        .iter()
        .map(|v| v * 4.0)
        .collect();
    let mut b = Bench::new();
    let x = b.f32(rows, width, &xs);
    let y = b.zeros(dtype::Dtype::F32, rows, width);
    if !b
        .run(|ctx| kernels_cerebras::linear::mlp::gelu_tanh(ctx, x, y))
        .unwrap()
    {
        return;
    }
    let want: Vec<f32> = xs
        .iter()
        .map(|x| 0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x * x * x)).tanh()))
        .collect();
    assert_close(&b.read_f32(y), &want, 1e-3, 1e-3);
}
