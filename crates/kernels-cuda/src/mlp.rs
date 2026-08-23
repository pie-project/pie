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

/// The `Mlp` family, claimed. Each body is a delegation to the `chunked_*`
/// routine below that already fires the point — the packed forms read one
/// `[gate | up]` row and write a row half as wide.
///
/// Every body drops the stated `intermediate`. The declaration states it
/// because the geometry is not derivable from the first `In` (the `Out` is
/// HALF its width), but the `Out` reaching a body has already been sized by
/// the `out(y = rows(packed) x half(packed))` rule on the routine, and the
/// launch reads `y.width` from it. When `#[shape]` replaces those rules the
/// stated width becomes their input; until then it is recorded and unread.
///
/// One point stays on the floor's default body, and the absence is measured
/// rather than an oversight:
///
/// * `mlp.swiglu_clamp_alpha` — `gpt_oss_glu` below computes the same
///   activation, but from gate and up as two separate rows. No cuda kernel
///   reads the packed form the text states, so there is nothing to delegate
///   to and the point reports itself unclaimed.
#[kernels_macros::claims]
impl kernels::points::Mlp for Ctx<'_> {
    fn swiglu<T: kernels::points::Scalar>(
        &self,
        packed: In<Tensor<T>>,
        intermediate: u32,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = intermediate;
        chunked_swiglu(self, packed, y)
    }

    fn swiglu_clamp<T: kernels::points::Scalar>(
        &self,
        packed: In<Tensor<T>>,
        intermediate: u32,
        limit: f32,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = intermediate;
        chunked_swiglu_clamp(self, packed, y, Const::new(limit))
    }

    fn geglu_tanh<T: kernels::points::Scalar>(
        &self,
        gate: In<Tensor<T>>,
        up: In<Tensor<T>>,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        geglu_tanh(self, gate, up, y)
    }

    fn geglu_tanh_packed<T: kernels::points::Scalar>(
        &self,
        packed: In<Tensor<T>>,
        intermediate: u32,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = intermediate;
        chunked_geglu_tanh(self, packed, y)
    }

    fn situ<T: kernels::points::Scalar>(
        &self,
        packed: In<Tensor<T>>,
        intermediate: u32,
        beta: f32,
        up_cap: f32,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = intermediate;
        chunked_situ(self, packed, y, Const::new(beta), Const::new(up_cap))
    }
}

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

#[routine(bf16, canon = "mlp.geglu_tanh", out(y = like(gate)))]
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

#[routine(bf16, canon = "mlp.swiglu", out(y = rows(packed) x half(packed)))]
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

#[routine(bf16, canon = "mlp.swiglu_clamp", out(y = rows(packed) x half(packed)))]
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

#[routine(bf16, canon = "mlp.situ", out(y = rows(packed) x half(packed)))]
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

#[routine(bf16, canon = "mlp.geglu_tanh_packed", out(y = rows(packed) x half(packed)))]
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

/// The `Gate` family, claimed. One point, and the delegation crosses a
/// dtype pin to reach it: `driver_internal::sigmoid_gate_inplace_bf16` is a
/// host program spelled at bf16, while a point quantifies over `Scalar`, so
/// the body states the pin as a refusal by name and casts the two addresses
/// it was handed. That is the whole of cuda's gating today; a second dtype
/// wants a second spelling of the routine, not a cast here.
///
/// The impl lives beside `sigmoid_dot_scalar_gate_add` — the other gate
/// kernel, and the module the C++ namespace names — rather than in
/// `driver_internal.rs`, which collects host programs by CALLER. `GATE_CLAIMS`
/// therefore reads `kernels_cuda::mlp::GATE_CLAIMS`.
#[kernels_macros::claims]
impl kernels::points::Gate for Ctx<'_> {
    fn sigmoid_mul<T: kernels::points::Scalar>(
        &self,
        x: InOut<Tensor<T>>,
        gate: In<Tensor<T>>,
    ) -> Result<(), Refusal> {
        if T::CPP != <bf16 as kernels::Elem>::CPP {
            return Err(Refusal::Absent {
                what: "gate.sigmoid_mul at an element other than bf16",
            });
        }
        crate::driver_internal::sigmoid_gate_inplace_bf16(
            self,
            InOut {
                ptr: x.ptr.cast::<bf16>(),
                rows: x.rows,
                width: x.width,
            },
            In {
                ptr: gate.ptr.cast::<bf16>(),
                rows: gate.rows,
                width: gate.width,
            },
        )
    }
}

#[routine(bf16, canon = "moe.sigmoid_gate_add", out(out = like(out)))]
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
