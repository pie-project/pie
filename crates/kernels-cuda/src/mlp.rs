use crate::jit::abi::Tensor;
use crate::jit::abi::{bf16, f16};
use crate::jit::{Ctx, Launch};
use kernels::Refusal;
use kernels::routine::{Const, In, InOut, Out};
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

/// The `Mlp` family, claimed. Every body is the launch itself: one
/// `__global__` out of `mlp/swiglu.cuh`, one thread block per `BLOCK`-wide
/// stripe of the result, and the packed forms read one `[gate | up]` row and
/// write a row half as wide.
///
/// Every body drops the stated `intermediate`. The declaration states it
/// because the geometry is not derivable from the first `In` (the `Out` is
/// HALF its width), but the `Out` reaching a body has already been sized by
/// the sweep's width rule, and the launch reads `y.width` from it. When
/// `#[shape]` replaces that rule the stated width becomes its input; until
/// then it is recorded and unread.
///
/// EVERY POINT OF THIS FAMILY IS ANSWERED, and the last absence was a
/// MISSING OPERAND SHAPE rather than a missing activation.
/// `mlp.swiglu_clamp_alpha` stood on the floor's default body because
/// [`gpt_oss_glu`] computes exactly its arithmetic but from gate and up as
/// two separate rows, and no cuda kernel read the packed form the text
/// states. `::pie::mlp::chunked_gpt_oss_glu` is that kernel.
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

    /// The clamped swiglu whose sigmoid carries a stated `alpha`, over the
    /// packed row the text states.
    ///
    /// A ROW WHERE THE ACTIVATION HAD TWO PLANES, and that was the whole
    /// gap: [`gpt_oss_glu`] computes this arithmetic and has since gpt-oss
    /// landed, but from `gate` and `up` as two rectangles, and a text that
    /// projects `[gate | up]` in one matmul has one. The `.cuh` grew
    /// `chunked_gpt_oss_glu` beside it — the same clamp, the same
    /// `(u + 1) * glu`, the same `expf` — over the packed indexing.
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

/// The two-plane tanh-GELU gate.
///
/// TWO CALLERS, WHICH IS WHY IT IS A FUNCTION: `Mlp::geglu_tanh` above, and
/// the gemma vision tower's MLP, which holds bare pointers rather than a
/// statement's rectangles.
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

/// gpt-oss's GLU over gate and up as TWO planes, and the second output plane
/// its fp16 arm writes.
///
/// `Mlp::swiglu_clamp_alpha` is the same arithmetic over one packed row and
/// is the form every text states. This one is the oracle
/// `tests/swiglu_clamp_alpha.rs` measures that body against — the packed
/// kernel is the flat kernel's arithmetic with packed indexing around it,
/// and that is a claim a test checks rather than a comment.
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

/// The `Gate` family, claimed. One point, and the body crosses a dtype pin
/// to reach its kernel: `::pie::mlp::sigmoid_gate_inplace` is spelled at
/// bf16 while a point quantifies over `Scalar`, so the body states the pin
/// as a refusal by name and casts the two addresses it was handed. That is
/// the whole of cuda's gating today; a second dtype wants a second
/// instantiation in the `.cuh`, not a cast here.
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
