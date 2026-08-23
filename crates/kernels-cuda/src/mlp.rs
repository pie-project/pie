use crate::jit::abi::Tensor;
use crate::jit::abi::{bf16, f16};
use crate::jit::{Ctx, Launch};
use kernels::Refusal;
use kernels::routine::{Const, In, InOut, Out};
use kernels::{Bind, Fire};
use kernels_macros::routine;

const BLOCK: u32 = 256;

const WARP: u32 = 32;

#[must_use]
const fn elementwise(n: i32) -> Launch {
    Launch::flat(n.unsigned_abs(), BLOCK)
}

#[must_use]
const fn elementwise_rows(rows: i32, width: i32) -> Launch {
    Launch::grid(
        [rows.unsigned_abs(), width.unsigned_abs().div_ceil(BLOCK), 1],
        [BLOCK, 1, 1],
    )
}

#[must_use]
const fn rms(rows: i32) -> Launch {
    const RMS_SMEM: u32 = (BLOCK / WARP) * 4;

    Launch::per_row(rows.unsigned_abs(), BLOCK).smem(RMS_SMEM)
}

pub const GPT_OSS_GLU_ALPHA: f32 = 1.702;

#[routine(bf16, out(y = like(gate)))]
pub fn swiglu<T>(
    ctx: &Ctx<'_>,
    gate: In<Tensor<T>>,
    up: In<Tensor<T>>,
    y: Out<Tensor<T>>,
) -> Result<(), Refusal> {
    let n = y.rows.saturating_mul(y.width);

    ctx.fire(
        Fire::at(
            "mlp/swiglu.cuh",
            crate::jit::symbol(&format!("::pie::mlp::swiglu<{}>", T::CPP)),
        )
        .apply(elementwise(n)),
        &[gate.arg(), up.arg(), y.arg(), n.arg()],
    )
}

#[routine(bf16, out(y = like(gate)))]
pub fn swiglu_clamp<T>(
    ctx: &Ctx<'_>,
    gate: In<Tensor<T>>,
    up: In<Tensor<T>>,
    y: Out<Tensor<T>>,
    limit: Const<f32>,
) -> Result<(), Refusal> {
    let limit = *limit;

    let n = y.rows.saturating_mul(y.width);
    ctx.fire(
        Fire::at(
            "mlp/swiglu.cuh",
            crate::jit::symbol(&format!("::pie::mlp::swiglu_clamp<{}>", T::CPP)),
        )
        .apply(elementwise(n)),
        &[gate.arg(), up.arg(), y.arg(), n.arg(), limit.arg()],
    )
}

#[routine(bf16, out(y = like(gate)))]
pub fn situ<T>(
    ctx: &Ctx<'_>,
    gate: In<Tensor<T>>,
    up: In<Tensor<T>>,
    y: Out<Tensor<T>>,
    beta: Const<f32>,
    linear_beta: Const<f32>,
) -> Result<(), Refusal> {
    let beta = *beta;
    let linear_beta = *linear_beta;
    let n = y.rows.saturating_mul(y.width);
    ctx.fire(
        Fire::at(
            "mlp/swiglu.cuh",
            crate::jit::symbol(&format!("::pie::mlp::situ<{}>", T::CPP)),
        )
        .apply(elementwise(n)),
        &[
            gate.arg(),
            up.arg(),
            y.arg(),
            n.arg(),
            beta.arg(),
            linear_beta.arg(),
        ],
    )
}

#[routine(bf16, canon = "swiglu.geglu_tanh", out(y = like(gate)))]
pub fn geglu_tanh<T>(
    ctx: &Ctx<'_>,
    gate: In<Tensor<T>>,
    up: In<Tensor<T>>,
    y: Out<Tensor<T>>,
) -> Result<(), Refusal> {
    let n = y.rows.saturating_mul(y.width);
    ctx.fire(
        Fire::at(
            "mlp/swiglu.cuh",
            crate::jit::symbol(&format!("::pie::mlp::geglu_tanh<{}>", T::CPP)),
        )
        .apply(elementwise(n)),
        &[gate.arg(), up.arg(), y.arg(), n.arg()],
    )
}

#[routine(dtypes(bf16, f16), out(y = like(x)))]
pub fn relu2<T>(ctx: &Ctx<'_>, x: In<Tensor<T>>, y: Out<Tensor<T>>) -> Result<(), Refusal> {
    let n = y.rows.saturating_mul(y.width);
    ctx.fire(
        Fire::at(
            "mlp/swiglu.cuh",
            crate::jit::symbol(&format!("::pie::mlp::relu2<{}>", T::CPP)),
        )
        .apply(elementwise(n)),
        &[x.arg(), y.arg(), n.arg()],
    )
}

#[routine(bf16, out(y = like(gate)))]
pub fn gpt_oss_glu<T>(
    ctx: &Ctx<'_>,
    gate: In<Tensor<T>>,
    up: In<Tensor<T>>,
    y: Out<Tensor<T>>,
    limit: Const<f32>,
    alpha: Const<f32>,
) -> Result<(), Refusal> {
    let limit = *limit;
    let alpha = *alpha;

    let n = y.rows.saturating_mul(y.width);
    ctx.fire(
        Fire::at(
            "mlp/swiglu.cuh",
            crate::jit::symbol(&format!("::pie::mlp::gpt_oss_glu<{}>", T::CPP)),
        )
        .apply(elementwise(n)),
        &[
            gate.arg(),
            up.arg(),
            y.arg(),
            ::core::ptr::null_mut::<f16>().arg(),
            n.arg(),
            limit.arg(),
            alpha.arg(),
        ],
    )
}

#[routine(bf16, canon = swiglu, out(y = rows(packed) x half(packed)))]
pub fn chunked_swiglu<T>(
    ctx: &Ctx<'_>,
    packed: In<Tensor<T>>,
    y: Out<Tensor<T>>,
) -> Result<(), Refusal> {
    ctx.fire(
        Fire::at(
            "mlp/swiglu.cuh",
            crate::jit::symbol(&format!("::pie::mlp::chunked_swiglu<{}>", T::CPP)),
        )
        .apply(elementwise_rows(y.rows, y.width)),
        &[packed.arg(), y.arg(), y.width.arg()],
    )
}

#[routine(bf16, out(y = like(y)))]
pub fn chunked_swiglu_into<T>(
    ctx: &Ctx<'_>,
    packed: In<Tensor<T>>,
    y: InOut<Tensor<T>>,
) -> Result<(), Refusal> {
    ctx.fire(
        Fire::at(
            "mlp/swiglu.cuh",
            crate::jit::symbol(&format!("::pie::mlp::chunked_swiglu<{}>", T::CPP)),
        )
        .apply(elementwise_rows(y.rows, y.width)),
        &[packed.arg(), y.arg(), y.width.arg()],
    )
}

#[routine(bf16, canon = "swiglu.clamp", out(y = rows(packed) x half(packed)))]
pub fn chunked_swiglu_clamp<T>(
    ctx: &Ctx<'_>,
    packed: In<Tensor<T>>,
    y: Out<Tensor<T>>,
    limit: Const<f32>,
) -> Result<(), Refusal> {
    let limit = *limit;

    ctx.fire(
        Fire::at(
            "mlp/swiglu.cuh",
            crate::jit::symbol(&format!("::pie::mlp::chunked_swiglu_clamp<{}>", T::CPP)),
        )
        .apply(elementwise_rows(y.rows, y.width)),
        &[packed.arg(), y.arg(), y.width.arg(), limit.arg()],
    )
}

#[routine(bf16, canon = situ, out(y = rows(packed) x half(packed)))]
pub fn chunked_situ<T>(
    ctx: &Ctx<'_>,
    packed: In<Tensor<T>>,
    y: Out<Tensor<T>>,
    beta: Const<f32>,
    linear_beta: Const<f32>,
) -> Result<(), Refusal> {
    let beta = *beta;
    let linear_beta = *linear_beta;
    ctx.fire(
        Fire::at(
            "mlp/swiglu.cuh",
            crate::jit::symbol(&format!("::pie::mlp::chunked_situ<{}>", T::CPP)),
        )
        .apply(elementwise_rows(y.rows, y.width)),
        &[
            packed.arg(),
            y.arg(),
            y.width.arg(),
            beta.arg(),
            linear_beta.arg(),
        ],
    )
}

#[routine(bf16, canon = "swiglu.geglu_tanh_packed", out(y = rows(packed) x half(packed)))]
pub fn chunked_geglu_tanh<T>(
    ctx: &Ctx<'_>,
    packed: In<Tensor<T>>,
    y: Out<Tensor<T>>,
) -> Result<(), Refusal> {
    ctx.fire(
        Fire::at(
            "mlp/swiglu.cuh",
            crate::jit::symbol(&format!("::pie::mlp::chunked_geglu_tanh<{}>", T::CPP)),
        )
        .apply(elementwise_rows(y.rows, y.width)),
        &[packed.arg(), y.arg(), y.width.arg()],
    )
}

#[routine(bf16, canon = sigmoid_gate_add, out(out = like(out)))]
pub fn sigmoid_dot_scalar_gate_add<T>(
    ctx: &Ctx<'_>,
    x: In<Tensor<T>>,
    gate_w: Const<Tensor<T>>,
    out: InOut<Tensor<T>>,
    y: In<Tensor<T>>,
) -> Result<(), Refusal> {
    let row = crate::layout::stated(out.all("the row width"))?;
    let h = row.stride;
    ctx.fire(
        Fire::at(
            "mlp/swiglu.cuh",
            crate::jit::symbol(&format!(
                "::pie::mlp::sigmoid_dot_scalar_gate_add<{}>",
                T::CPP
            )),
        )
        .apply(rms(row.rows)),
        &[x.arg(), gate_w.arg(), out.arg(), y.arg(), h.arg()],
    )
}

#[routine(bf16, out(x = like(x)))]
pub fn gaussian_topk<T>(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<T>>,
    std_multiplier: Const<f32>,
) -> Result<(), Refusal> {
    let std_multiplier = *std_multiplier;

    let row = crate::layout::stated(x.all("the row width"))?;
    let dim = row.stride;
    ctx.fire(
        Fire::at(
            "mlp/gaussian_topk.cuh",
            crate::jit::symbol(&format!("::pie::mlp::gaussian_topk<{}>", T::CPP)),
        )
        .apply(rms(row.rows)),
        &[x.arg(), dim.arg(), std_multiplier.arg()],
    )
}
