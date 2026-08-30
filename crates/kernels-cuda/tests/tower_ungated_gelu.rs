//! **THE UNGATED GELU, AGAINST A CPU REFERENCE AND AGAINST THE BAKE IT
//! REPLACES.**
//!
//! ```text
//! cargo test -p kernels-cuda --features cuda-13 --test tower_ungated_gelu -- --nocapture
//! ```
//!
//! `linear::mlp::gelu_tanh` is `.wiki/alto/multimodal.md` §6.2's cost
//! decision, landed. `Qwen3_5VisionMLP` is `fc2(act(fc1(x)))` with
//! `hidden_act: gelu_pytorch_tanh` and NOT gated, and every gelu arm this
//! plane had multiplies by an `up` half. The alternative was a bake — a
//! `gate_up` bank at `[2·inter, hidden]` with the `up` half zero and the `up`
//! half of the bias one — so the second gate below fires that bake and
//! demands the same numbers, which is what makes the arm a saving rather than
//! a change:
//!
//! ```text
//! (a) every element matches `0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))`
//!     in f32, from the same bf16 inputs
//! (b) THE BAKE ANSWERS THE SAME NUMBERS: `mlp_geglu_tanh_packed` over a
//!     packed row whose `up` half is ones agrees with this arm word for word
//!     up to one rounding — so declining the op would have been correct, and
//!     0.5 GiB more expensive on qwen36
//! (c) the refusals fire by name: an empty rectangle, an element with no
//!     kernel
//! ```

#![cfg(feature = "_cuda")]

mod common;

use common::{Gpu, Lcg, close, from_bf16, to_bf16};

use dtype::Dtype;
use kernels_cuda::linear::mlp;
use kernels_cuda::tensor::Tensor;

const INTERMEDIATE: u32 = 512;

const ROWS: u32 = 4;

fn gelu_tanh(x: f32) -> f32 {
    const C: f32 = 0.797_884_56;
    0.5 * x * (1.0 + (C * (x + 0.044_715 * x * x * x)).tanh())
}

fn fire_gelu(x: &[u16], rows: u32, width: u32) -> Vec<u16> {
    let mut gpu = Gpu::open();
    let x_at = gpu.up(x);
    let y_at = gpu.zeros(x.len() * 2);
    let mut y = Tensor::new(y_at, rows, width, Dtype::Bf16);
    mlp::gelu_tanh(
        &gpu.ctx(),
        Tensor::new(x_at, rows, width, Dtype::Bf16),
        &mut y,
    )
    .expect("the ungated gelu enqueues");
    gpu.sync();
    gpu.down(y_at, x.len())
}

/// The bake §6.2 prices: a packed `[rows, 2·inter]` whose gate half is `x`
/// and whose `up` half is ones, through the arm that already shipped.
fn fire_baked(x: &[u16], rows: u32, intermediate: u32) -> Vec<u16> {
    let one = to_bf16(1.0);
    let mut packed = Vec::with_capacity(x.len() * 2);
    for row in 0..rows as usize {
        let at = row * intermediate as usize;
        packed.extend_from_slice(&x[at..at + intermediate as usize]);
        packed.extend(std::iter::repeat_n(one, intermediate as usize));
    }

    let mut gpu = Gpu::open();
    let packed_at = gpu.up(&packed);
    let y_at = gpu.zeros(x.len() * 2);
    let mut y = Tensor::new(y_at, rows, intermediate, Dtype::Bf16);
    mlp::geglu_tanh_packed(
        &gpu.ctx(),
        Tensor::new(packed_at, rows, 2 * intermediate, Dtype::Bf16),
        intermediate,
        &mut y,
    )
    .expect("the packed gated arm enqueues");
    gpu.sync();
    gpu.down(y_at, x.len())
}

/// (a): the activation is the activation.
#[test]
fn every_element_matches_a_gelu_tanh_reference() {
    let (raw, exact) = Lcg::seeded(41).row((ROWS * INTERMEDIATE) as usize);
    let landed = fire_gelu(&raw, ROWS, INTERMEDIATE);

    for (at, value) in exact.iter().enumerate() {
        let got = from_bf16(landed[at]);
        let want = gelu_tanh(*value);
        assert!(
            close(got, want),
            "element {at} landed {got} and gelu_tanh({value}) is {want}"
        );
    }
}

/// (b) THE BAKE ANSWERS THE SAME NUMBERS — which is what makes the op a
/// saving and not a correction.
#[test]
fn the_zero_up_bake_answers_what_the_op_does() {
    let (raw, _) = Lcg::seeded(83).row((ROWS * INTERMEDIATE) as usize);
    let direct = fire_gelu(&raw, ROWS, INTERMEDIATE);
    let baked = fire_baked(&raw, ROWS, INTERMEDIATE);

    for at in 0..direct.len() {
        let apart = i32::from(direct[at]) - i32::from(baked[at]);
        assert!(
            apart.abs() <= 1,
            "element {at}: the ungated arm answered {:#06x} and the zero-up bake {:#06x}",
            direct[at],
            baked[at]
        );
    }
}

/// (c): the refusals, by name.
#[test]
fn the_refusals_fire_by_name() {
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let x_at = gpu.zeros((ROWS * INTERMEDIATE) as usize * 2);
    let y_at = gpu.zeros((ROWS * INTERMEDIATE) as usize * 2);

    let mut empty = Tensor::new(y_at, 0, INTERMEDIATE, Dtype::Bf16);
    let none = mlp::gelu_tanh(
        &ctx,
        Tensor::new(x_at, 0, INTERMEDIATE, Dtype::Bf16),
        &mut empty,
    );
    assert!(
        none.is_err(),
        "a rectangle with no elements is refused rather than launched empty"
    );

    let mut wrong = Tensor::new(y_at, ROWS, INTERMEDIATE, Dtype::F32);
    let dtype = mlp::gelu_tanh(
        &ctx,
        Tensor::new(x_at, ROWS, INTERMEDIATE, Dtype::F32),
        &mut wrong,
    );
    assert!(
        matches!(
            dtype.expect_err("f32 has no ungated gelu here"),
            kernels_cuda::Error::DtypeUnsupported { .. }
        ),
        "an element with no kernel is refused as a dtype"
    );
}
