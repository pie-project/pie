//! What a trace that states one of `quant`'s symbols binds to.
//!
//! One hand arm is left, and it is not a weight walker: `mxfp4_moe_gate_up_
//! decode_bf16` must bind NULL for a bias plane the loaded export does not
//! carry, and its two `Env`s name no key for the column to read.
//!
//! `dequant_fp8_e4m3_to_bf16_per_channel` WAS THE OTHER, and it crossed. The
//! prose here called it a weight walker — reading a CHECKPOINT MATRIX's
//! extents where `Cx` answers a fire's rectangle — and that stopped being true
//! when its signature took `rows: Param<1, i32>`/`cols: Param<2, i32>`:
//! `model-dsl`'s `dequant_fp8_e4m3` records `vec![0, rows, cols]`, so the
//! STATEMENT carries the matrix's extents and the column reads them. The arm
//! passed `cx.rows().count` and `cx.out_width(0)?` instead, which are the same
//! two numbers on this row and would not be on a row whose value shape stopped
//! being the weight's.
//!
//! The loader's two quantisers were here and are not. Nothing states them —
//! `model-loader`'s transform plan runs them once at load — so they carried a
//! row that existed only to account for a symbol no fire could ever reach.
//! They are plain `unsafe fn`s in `kernels_cuda::quant` now, out of that
//! crate's routine registry too, and the loader reaches them by path.
//!
//! Two `Lit` temptations are declined: `wna16_*`'s `group_size` comes off the
//! checkpoint despite being 128 almost everywhere, and a `Lit(Null)` for
//! `mxfp4_moe_down_decode`'s `bias_ptrs` would have frozen a null over a plane
//! the checkpoint ships.

use core::ffi::c_void;

use kernels::Refusal;
use kernels::routine::{Const, In, Out};
use kernels_cuda::jit::Ctx;
use kernels_cuda::jit::abi::{bf16, f16};
use kernels_cuda::quant::*;

use super::super::cx::Cx;
use super::Bound;


/// `quant::mxfp4_moe_gate_up_decode_bf16`
fn mxfp4_moe_gate_up_decode_bf16_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // THE THREE PLANES ARE ASKED FOR NOW, so this arm LENDS the fire's facts
    // rather than resolving them by hand: `_scales` is `keys::WeightScales`
    // and the two bias planes are `ctx.absent()`. Absence there is which
    // export was loaded -- not something a CUDA statement sees -- and the
    // body binds null for it, which is what `unwrap_or(null_mut())` did here.
    // SAFETY: `stream` is the fire's own, live across the launch.
    let answers = crate::bind::table::Answering::over_facts(cx);
    let ctx = unsafe { Ctx::on(stream) }.with_env(&answers);
    mxfp4_moe_gate_up_decode_bf16(
        &ctx,
        // Operand 0 is the INDEX run and operand 1 the activation; this
        // family states `vec![experts.id, x.id]`.
        In {
            ptr: cx.arg_in(0)?.cast_const().cast::<i32>(),
            rows: cx.rows().count,
            width: cx.in_width(0).unwrap_or(0),
        },
        In {
            ptr: cx.arg_in(1)?.cast_const().cast::<f16>(),
            rows: cx.rows().count,
            width: cx.in_width(1).unwrap_or(0),
        },
        Const { v: cx.weight(0)?.cast_const().cast::<u8>() },
        Out {
            ptr: cx.arg_out(0)?.cast::<bf16>(),
            rows: cx.rows().count,
            width: cx.out_width(0).unwrap_or(0),
        },
        Out {
            ptr: cx.arg_out(1)?.cast::<bf16>(),
            rows: cx.rows().count,
            width: cx.out_width(1).unwrap_or(0),
        },
        // THE TWO GLU CONSTANTS ARE ASKED FOR INSIDE THE CALLEE NOW: `ctx`
        // carries `Answering::over_facts(cx)`, so `keys::GluLimit` and
        // `keys::GluAlpha` reach the body off the same `Cx` this arm would
        // have read them from.
    )
}

/// Every symbol this family binds.
pub static ARMS: &[Bound] = &[
    // A weight walker: `out_dim`/`in_dim` are the checkpoint matrix's extents.
    Bound::derived("quant::dequant_wna16_int4b8_to_bf16"),
    // Its `n` is a `usize` and `operand()` mints `ArgValue::I32`, so the call
    // refuses `Refusal::Kind`. Narrowing is wrong here: `model-loader` casts
    // whole checkpoint tensors, whose counts are genuinely `usize`.
    Bound::derived("quant::cast_fp32_to_bf16"),
    // `selected_rows` is the trap: `out.rows` is the same number today for a
    // reason unrelated to what the parameter means.
    Bound::derived("quant::mxfp4_scales_to_marlin_e8m0"),
    // A weight walker, plus `scale` from `cx.param_f32(0)?`: two independent
    // refusals on one row.
    Bound::derived("quant::dequant_fp8_e4m3_to_bf16"),
    // The extents are the MATRIX's and the STATEMENT carries them, in
    // `params[1..3]` — so the pointers and the rectangle derive together. See
    // the header for what this row used to say and why it stopped being true.
    Bound::derived("quant::dequant_fp8_e4m3_to_bf16_per_channel"),
    Bound::derived("quant::dequant_fp8_e4m3_to_bf16_per_group"),
    Bound::derived("quant::dequant_mxfp4_to_bf16"),

    // A non-positive width now refuses `Absent`, not `Empty`.
    Bound::derived("quant::bf16_to_fp16"),

    // No `in_place` needed: that is a claim about ALIASING, where
    // `#[source(In(1))]` is a claim about WHICH INPUT. `x` is scaled in place,
    // so the first genuine `*const` is the second input.
    Bound::derived("quant::scale_rows_bf16"),

    // The operand order inverts against the W4A16 twin below: the parameter
    // order is the CUDA signature's, the operand order the model text's. Both
    // are stated. `packed_ptrs` reads the POSITIONAL bank, not `Weight<0, _>`.
    // What blocks the row is `_gate_bias`/`_up_bias`, which must bind NULL on
    // the export that has neither.
    Bound {
        symbol: "quant::mxfp4_moe_gate_up_decode_bf16",
        arm: None,
        unbound: Some("a per-expert POINTER ARRAY for the packed bank and its scales. Measured with \
             `compute-sanitizer` on gpt-oss-20b: the kernel does \
             `packed_ptrs[expert]`, so both parameters are `const u8* const*` \
             over `num_experts` entries, and the launcher binds the BANK's own \
             base address instead -- the first eight bytes of MXFP4 weight data \
             read as a pointer. That is an illegal address that poisons the \
             context, and the next module load reports it, so the failure names \
             an unrelated kernel. Nothing in this tree builds the array: \
             `build_moe_ptrs_aligned` is qwen3.5's statement-side one and gpt-oss \
             states only the bank by name. Refusing at LOAD is the honest \
             answer until something does"),
    },
    // `scale_ptrs` is NOT `Weight(1)`: the positional bank is a separate slice
    // from the suffixed lookup `keys::WeightScales` names. Its kernel takes
    // the same two pointer ARRAYS as the gate/up twin above and is unfireable
    // for the same measured reason.
    Bound {
        symbol: "quant::mxfp4_moe_down_decode_bf16",
        arm: None,
        unbound: Some("the same per-expert pointer arrays the gate/up twin above needs"),
    },

    // The positional weights are marked because `In(2)`..`In(5)` are ANSWERABLE
    // -- they resolve the moment a statement places six operands, to buffers
    // these launchers do not want. Both rows are `arm: None` so they refuse at
    // LOAD rather than dying mid-fire with `NoArm`.
    Bound {
        symbol: "quant::wna16_gate_up_decode_bf16",
        arm: None,
        unbound: Some(
            "a WNA16 group size, which no model contract states and no `QuantMeta` here builds",
        ),
    },
    Bound {
        symbol: "quant::wna16_down_decode_bf16",
        arm: None,
        unbound: Some(
            "a WNA16 group size; see `quant::wna16_gate_up_decode_bf16`.",
        ),
    },
];
