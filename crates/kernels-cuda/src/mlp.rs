use crate::jit::abi::Tensor;
use crate::jit::abi::{bf16, f16};
use crate::jit::{Ctx, Launch};
use kernels::Refusal;
use kernels::plane::{Const, In, InOut, Out};
use kernels::{Bind, Fire};

const BLOCK: u32 = 256;

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

pub const GPT_OSS_GLU_ALPHA: f32 = 1.702;

#[kernels_macros::claims]
impl kernels::points::Mlp for Ctx<'_> {
    fn swiglu<T: kernels::points::Scalar>(
        &self,
        packed: In<Tensor<T>>,
        intermediate: u32,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = intermediate;
        self.fire(
            Fire::at(
                "mlp/swiglu.cuh",
                crate::jit::symbol(&format!("::pie::mlp::chunked_swiglu<{}>", T::CPP)),
            )
            .apply(elementwise_rows(y.rows, y.width)),
            &[packed.arg(), y.arg(), y.width.arg()],
        )
    }

    fn swiglu_clamp<T: kernels::points::Scalar>(
        &self,
        packed: In<Tensor<T>>,
        intermediate: u32,
        limit: f32,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = intermediate;
        self.fire(
            Fire::at(
                "mlp/swiglu.cuh",
                crate::jit::symbol(&format!("::pie::mlp::chunked_swiglu_clamp<{}>", T::CPP)),
            )
            .apply(elementwise_rows(y.rows, y.width)),
            &[packed.arg(), y.arg(), y.width.arg(), limit.arg()],
        )
    }

    fn swiglu_clamp_alpha<T: kernels::points::Scalar>(
        &self,
        packed: In<Tensor<T>>,
        intermediate: u32,
        limit: f32,
        alpha: f32,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = intermediate;
        self.fire(
            Fire::at(
                "mlp/swiglu.cuh",
                crate::jit::symbol(&format!("::pie::mlp::chunked_gpt_oss_glu<{}>", T::CPP)),
            )
            .apply(elementwise_rows(y.rows, y.width)),
            &[
                packed.arg(),
                y.arg(),
                y.width.arg(),
                limit.arg(),
                alpha.arg(),
            ],
        )
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
        self.fire(
            Fire::at(
                "mlp/swiglu.cuh",
                crate::jit::symbol(&format!("::pie::mlp::chunked_geglu_tanh<{}>", T::CPP)),
            )
            .apply(elementwise_rows(y.rows, y.width)),
            &[packed.arg(), y.arg(), y.width.arg()],
        )
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
        self.fire(
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
                up_cap.arg(),
            ],
        )
    }
}

pub(crate) fn geglu_tanh<T: crate::RoutineElem>(
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

pub fn gpt_oss_glu<T: crate::RoutineElem>(
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
        let x: InOut<Tensor<bf16>> = InOut {
            ptr: x.ptr.cast::<bf16>(),
            rows: x.rows,
            width: x.width,
        };
        let gate: In<Tensor<bf16>> = In {
            ptr: gate.ptr.cast::<bf16>(),
            rows: gate.rows,
            width: gate.width,
        };
        let num_elements = x.all("the gated rectangle")?.elements();
        self.fire(
            Fire::at(
                "mlp/swiglu.cuh",
                "::pie::mlp::sigmoid_gate_inplace<::pie::bf16>",
            )
            .apply(Launch::flat(num_elements.unsigned_abs(), BLOCK)),
            &[x.arg(), gate.arg(), num_elements.arg()],
        )
    }
}
