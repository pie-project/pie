//! `Clip`: the free-standing activation clamp (`.wiki/alto/multimodal.md`
//! §6.5).
//!
//! gemma4's `vision_config.use_clipped_linears: true` publishes
//! `{input,output}_{min,max}` as scalars beside every
//! `encoder.layers.{l}.*.linear.weight`, so each projection clamps what it
//! reads and what it writes. The only clamp this plane had was FUSED inside
//! `linear::mlp::swiglu_clamp`, which is a swiglu and not a projection; this
//! is the clamp on its own, because the sites it serves sit on both sides of
//! an ordinary matmul and a fused spelling would need one fusion per
//! projection shape.

use crate::error::Error;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "elemwise/clip.cuh";

const BLOCK: u32 = 256;

/// `x = min(max(x, lo), hi)`, in place, one thread per element.
///
/// Both bounds are TRACE CONSTANTS — the checkpoint's own scalars — and are
/// rounded through the element before the comparison, so a value already at
/// the stated bound is left where it is.
///
/// # Errors
///
/// [`Error::DtypeUnsupported`] for anything but bf16 and f16; a refusal for
/// bounds that cross (`lo > hi`, which would collapse the row to `hi`
/// silently), for an empty rectangle, and for an extent past a 32-bit launch
/// — clamped rather than truncated is the one thing this must not do.
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
            // The element-form seat's width: this launch is flat over
            // `rows * width`, so the kernel needs the row's width to read the
            // staged row count and row start as elements.
            stated(OP, x.width)?.arg(),
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )
}

/// **THE SAME CLAMP, WITH THE BOUNDS THE CHECKPOINT STATES** (multimodal
/// §12.2): `lo` and `hi` are one-element planes read on the device.
///
/// gemma4 ships 448 of them over its vision tower — `input_min`/`input_max`
/// and `output_min`/`output_max` beside every `.linear.weight`, all finite,
/// all different — so they are a property of the artifact and not of the text.
/// [`crate::elemwise::norm::scale`] reads its scalar this way for the same
/// reason, and this entry is that argument applied to a bound.
///
/// **NO CROSSED-BOUNDS REFUSAL HERE**, and that is the one difference from
/// [`clamp`]: the plain form compares two numbers the host holds, and this one
/// would have to read device memory to compare anything. A checkpoint whose
/// `input_min` exceeds its `input_max` is an import to refuse, where the two
/// planes are still on the host.
///
/// # Errors
///
/// [`Error::DtypeUnsupported`] for anything but bf16 and f16; a refusal for a
/// bound plane that is not one element of the activation's own type, for an
/// empty rectangle, and for an extent past a 32-bit launch.
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
            // The element-form seat's width: this launch is flat over
            // `rows * width`, so the kernel needs the row's width to read the
            // staged row count and row start as elements.
            stated(OP, x.width)?.arg(),
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )
}
