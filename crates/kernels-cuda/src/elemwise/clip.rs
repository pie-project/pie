//! `Clip`: the free-standing activation clamp.
//!
//! gemma4's `vision_config.use_clipped_linears: true` publishes `{input,output}_{min,max}` as scalars beside every `encoder.layers.{l}.*.linear.weight`, so each projection clamps what it reads and writes. This is a standalone clamp (unlike the one fused inside `linear::mlp::swiglu_clamp`), since it serves sites on both sides of an ordinary matmul.

use crate::error::Error;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "elemwise/clip.cuh";

const BLOCK: u32 = 256;

/// `x = min(max(x, lo), hi)`, in place, one thread per element. Both bounds are trace constants, rounded through the element before the comparison.
///
/// Errs [`Error::DtypeUnsupported`] for anything but bf16 and f16; a refusal for crossed bounds (`lo > hi`), an empty rectangle, or an extent past a 32-bit launch.
pub fn clamp(ctx: &Ctx, lo: f32, hi: f32, x: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.clamp";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    if !(lo <= hi) {
        return Err(refuse(
            OP,
            format!("the bounds {lo} and {hi} cross, and a clamp between them is the constant {hi}"),
        ));
    }
    let n = x.elements();
    let lanes = u32::try_from(n).map_err(|_| {
        refuse(
            OP,
            format!("{n} elements do not fit a 32-bit launch extent"),
        )
    })?;
    nonzero(OP, "the element count", lanes)?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::elemwise::clamp<{t}>")))
            .apply(Launch::flat(lanes, BLOCK)),
        &[
            x.arg(),
            lo.arg(),
            hi.arg(),
            n.arg(),
            // flat over rows * width; the kernel needs the row width to read the staged row count/start as elements.
            stated(OP, x.width)?.arg(),
            // region's live-rows word when a body replay armed one, else the null seat.
            ctx.stage(),
        ],
    )
}

/// The same clamp, with the bounds the checkpoint states: `lo` and `hi` are one-element planes read on the device (a property of the artifact, not the text).
///
/// No crossed-bounds refusal here, unlike [`clamp`]: comparing would require reading device memory. A checkpoint whose `input_min` exceeds its `input_max` is an import-time refusal instead.
///
/// Errs [`Error::DtypeUnsupported`] for anything but bf16 and f16; a refusal for a bound plane that isn't one element of the activation's own type, an empty rectangle, or an extent past a 32-bit launch.
pub fn clamp_learned(
    ctx: &Ctx,
    lo: Tensor,
    hi: Tensor,
    x: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.clamp_learned";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    for (what, bound) in [("lower", lo), ("upper", hi)] {
        if bound.dtype != x.dtype {
            return Err(refuse(
                OP,
                format!(
                    "the {what} bound is {:?} and the rows it clamps are {:?}; a learned bound \
                     rides the activation's element",
                    bound.dtype, x.dtype
                ),
            ));
        }
        if bound.elements() != 1 {
            return Err(refuse(
                OP,
                format!(
                    "the {what} bound is a {} x {} plane, and this clamp reads one scalar",
                    bound.rows, bound.width
                ),
            ));
        }
    }
    let n = x.elements();
    let lanes = u32::try_from(n).map_err(|_| {
        refuse(
            OP,
            format!("{n} elements do not fit a 32-bit launch extent"),
        )
    })?;
    nonzero(OP, "the element count", lanes)?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::elemwise::clamp_learned<{t}>")))
            .apply(Launch::flat(lanes, BLOCK)),
        &[
            x.arg(),
            lo.arg(),
            hi.arg(),
            n.arg(),
            // flat over rows * width; the kernel needs the row width to read the staged row count/start as elements.
            stated(OP, x.width)?.arg(),
            // region's live-rows word when a body replay armed one, else the null seat.
            ctx.stage(),
        ],
    )
}
