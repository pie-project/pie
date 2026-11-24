//! The flat elementwise kernels over planes of any element count: a plane
//! past 2^31 elements lands every element.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, bf16, from_bf16};
use dtype::Dtype;
use kernels_rocm::elemwise::{gate, pointwise};
use kernels_rocm::linear::glu;

/// One plane of 2^31 + 8 elements, 8 wide: each flat kernel lands its last
/// elements (past any 32-bit index) as it lands its first.
#[test]
fn a_plane_past_two_to_the_31_lands_every_element() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    const WIDTH: u32 = 8;
    const ROWS: u32 = (1 << 28) + 1;
    let n = u64::from(ROWS) * u64::from(WIDTH);
    assert!(n > 1 << 31);
    let bytes = (n * 2) as usize;
    // x = 0x3f3f (0.746...), y = 0x4040 (3.0), byte-filled.
    let x = gpu.zeros(ROWS, WIDTH, Dtype::Bf16);
    let y = gpu.zeros(ROWS, WIDTH, Dtype::Bf16);
    let mut o = gpu.zeros(ROWS, WIDTH, Dtype::Bf16);
    gpu.fill(x.ptr, 0x3f, bytes);
    gpu.fill(y.ptr, 0x40, bytes);
    let (xv, yv) = (from_bf16(0x3f3f), from_bf16(0x4040));
    let silu = |v: f32| v / (1.0 + (-v).exp());
    let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
    // The elements read: the first, about 2^31, and the last.
    let at = [0u64, (1 << 31) - 8, 1 << 31, n - 1];
    let check = |what: &str, t: &kernels_rocm::Tensor, want: f32| {
        for &i in &at {
            let got = from_bf16(gpu.down::<u16>(t.ptr + 2 * i, 1)[0]);
            let ulp = (from_bf16(bf16(want) + 1) - from_bf16(bf16(want))).abs();
            assert!(
                (got - want).abs() <= ulp,
                "{what}[{i}] = {got}, want {want}"
            );
        }
    };
    pointwise::add(&ctx, x, y, &mut o).unwrap();
    check("add", &o, xv + yv);
    pointwise::mul(&ctx, x, y, &mut o).unwrap();
    check("mul", &o, xv * yv);
    pointwise::silu(&ctx, x, &mut o).unwrap();
    check("silu", &o, silu(xv));
    pointwise::mul_scalar(&ctx, 2.0, &mut o).unwrap();
    check("mul_scalar", &o, from_bf16(bf16(silu(xv))) * 2.0);
    pointwise::clamp(&ctx, -0.5, 0.5, &mut o).unwrap();
    check("clamp", &o, 0.5);
    glu::swiglu_clamp_split(&ctx, x, y, 1, 7.0, &mut o).unwrap();
    check("swiglu_clamp", &o, silu(xv) * yv);
    glu::gelu_tanh(&ctx, y, 1, &mut o).unwrap();
    check(
        "gelu",
        &o,
        0.5 * yv * (1.0 + (0.797_884_6 * (yv + 0.044_715 * yv * yv * yv)).tanh()),
    );
    glu::situ_split(&ctx, x, y, 1, 2.0, None, &mut o).unwrap();
    check("situ", &o, 2.0 * (xv / 2.0).tanh() * sigmoid(xv) * yv);
    gpu.fill(o.ptr, 0x3f, bytes);
    gate::sigmoid_mul(&ctx, y, 1, &mut o).unwrap();
    check("gate", &o, xv * sigmoid(yv));
}
