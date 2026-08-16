#![allow(clippy::too_many_arguments)]
//! Rotary embeddings.
//!
//! Four spellings of the schedule (`neox`, `freqs`, `prop`, and the strided
//! form), each in a decode and a multi-batch shape. The `freqs` pair reads a
//! host-computed table, which is what llama-3.1's wavelength ramp needs.

use kernels::{KernelSig, kernel};

use crate::axes::*;

pub static KERNELS: &[KernelSig] = &[
    // 1 in rope.wgsl
    // IN PLACE on ONE tensor: buffer 0 is both the input and the result, and
    // the template is `rope_neox_decode`.
    //
    // Which makes a TEXT gap visible that no row can close. `dsl::metal::rope`
    // states one launch carrying q and k — two inputs and two results — and
    // this kernel rotates one buffer. The statement should be two, one per
    // tensor, and until it is, the second tensor is not rotated at all.
    kernel!(neox_decode "neox_decode", file = Some("rope/neox.wgsl"), launch = kernels::LaunchRule::Rope,
        operands = kernels::operands![
            x: BufMut <- kernels::Source::Out(0),
            position: I32s <- kernels::Source::Positions,
            scale: F32 <- kernels::Source::ParamF32(0),
            base: F32 <- kernels::Source::ParamF32(1),
            head_dim: I32 <- kernels::Source::Param(2),
        ],
        // The rotation's extent is the STATEMENT's, not the fire's: gemma-4
        // rotates a quarter of each full-attention head and all of each
        // sliding one. The kernel never reads param 3 -- its operand list
        // stops at 2 -- but `Rule::Rope`'s grid is half of it.
        grid_param = Some(3),
        // The heads are counted by the SAME width the kernel is told.
        head_param = Some(2),
        axes = &[BF16]),
    // 1 in rope.wgsl
    // The rotation a deployment that RESCALES its ladder takes. Same body as
    // `neox_decode` with the frequencies read from a buffer instead of raised
    // from a base -- which is the only form that can express llama-3's
    // piecewise rescaling or YaRN's, because neither is a base.
    kernel!(neox_freqs_decode "neox_freqs_decode", file = Some("rope/neox.wgsl"),
        launch = kernels::LaunchRule::Rope,
        operands = kernels::operands![
            x: BufMut <- kernels::Source::Out(0),
            position: I32s <- kernels::Source::Positions,
            scale: F32 <- kernels::Source::ParamF32(0),
            inv_freq: Buf <- kernels::Source::RopeFrequencies,
            head_dim: I32 <- kernels::Source::Param(1),
            // YaRN's attention-temperature correction. One for a deployment
            // that has none, which is every llama-3 one -- its rescaling is in
            // the frequencies and not in a gain.
            mscale: F32 <- kernels::Source::ParamF32(2),
        ],
        // See `neox_decode`: the extent is the statement's.
        grid_param = Some(3),
        // The heads are counted by the SAME width the kernel is told.
        head_param = Some(1),
        axes = &[BF16]),
    // 1 in rope.wgsl
    // The batched form of the rescaled ladder, and the row a PREFILL on
    // llama-3.1, llama-3.2 or any YaRN deployment needs. It was bare, so the
    // statement had nothing to name and named the decode symbol instead — a
    // single-row kernel over a multi-row grid, which rotates row zero and
    // leaves every row after it untouched. Rope is the identity at position
    // zero, so row zero agreed with the reference either way and the failure
    // was silent.
    //
    // Same operands as `neox_freqs_decode`: the row stride the shader needs is
    // `grid.y * head_dim`, and `Rule::Rope` now takes its head axis from the
    // tensor being turned, so the grid says it.
    kernel!(neox_freqs_mb "neox_freqs_mb", file = Some("rope/neox.wgsl"),
        launch = kernels::LaunchRule::Rope,
        operands = kernels::operands![
            x: BufMut <- kernels::Source::Out(0),
            position: I32s <- kernels::Source::Positions,
            scale: F32 <- kernels::Source::ParamF32(0),
            inv_freq: Buf <- kernels::Source::RopeFrequencies,
            head_dim: I32 <- kernels::Source::Param(1),
            mscale: F32 <- kernels::Source::ParamF32(2),
        ],
        // See `neox_decode`: the extent is the statement's.
        grid_param = Some(3),
        // The heads are counted by the SAME width the kernel is told.
        head_param = Some(1),
        axes = &[BF16]),
    // 1 in rope.wgsl
    // The batched form, and the same shape: one tensor, per-token positions.
    kernel!(neox_mb "neox_mb", file = Some("rope/neox.wgsl"), launch = kernels::LaunchRule::Rope,
        operands = kernels::operands![
            x: BufMut <- kernels::Source::Out(0),
            position: I32s <- kernels::Source::Positions,
            scale: F32 <- kernels::Source::ParamF32(0),
            base: F32 <- kernels::Source::ParamF32(1),
            head_dim: I32 <- kernels::Source::Param(2),
        ],
        // The rotation's extent is the STATEMENT's, not the fire's: gemma-4
        // rotates a quarter of each full-attention head and all of each
        // sliding one. The kernel never reads param 3 -- its operand list
        // stops at 2 -- but `Rule::Rope`'s grid is half of it.
        grid_param = Some(3),
        // The heads are counted by the SAME width the kernel is told.
        head_param = Some(2),
        axes = &[BF16]),
    // 1 in rope.wgsl
    // gemma's rotation: the same neox body over a PROPORTIONAL slice of each
    // head rather than all of it. Same operands as `neox_decode`, and in
    // place like every rotation in this file.
    kernel!(neox_prop_decode "neox_prop_decode", file = Some("rope/neox.wgsl"),
        launch = kernels::LaunchRule::Rope,
        operands = kernels::operands![
            x: BufMut <- kernels::Source::Out(0),
            position: I32s <- kernels::Source::Positions,
            scale: F32 <- kernels::Source::ParamF32(0),
            base: F32 <- kernels::Source::ParamF32(1),
            head_dim: I32 <- kernels::Source::Param(2),
        ],
        // The rotation's extent is the STATEMENT's, not the fire's: gemma-4
        // rotates a quarter of each full-attention head and all of each
        // sliding one. The kernel never reads param 3 -- its operand list
        // stops at 2 -- but `Rule::Rope`'s grid is half of it.
        grid_param = Some(3),
        // The heads are counted by the SAME width the kernel is told.
        head_param = Some(2),
        axes = &[BF16]),
    // 1 in rope.wgsl
    kernel!(neox_prop_mb "neox_prop_mb", axes = &[BF16]),
    // 1 in rope.wgsl
    kernel!(neox_strided "neox_strided", axes = &[BF16]),
];

use crate::routine::{Bind, Buf, BufMut, Ctx, Env, Fire, I32s, Routine};
use kernels::routine::Refusal;

/// One invocation per `(pair, head, row)`.
///
/// `rope/neox.wgsl` is `@workgroup_size(1)`, so the lanes a [`Fire`] states
/// ARE the workgroups — which is why this returns the same three numbers
/// `kernels-vulkan::rope::rope_grid` does, where they are workgroups
/// outright. x is the PAIR: a rotation moves two channels at once, so a
/// `rotary` of 128 is 64 lanes and a grid built on the channel count would
/// rotate the first half twice and the second half not at all.
///
/// # Errors
///
/// [`Refusal::Empty`] for a zero rotary width, head width or row count;
/// [`Refusal::Narrow`] for a rotary width that is not a whole number of pairs
/// or a row that is not a whole number of heads. Both are checked rather than
/// rounded: an odd `rotary` leaves one channel unrotated and a ragged width
/// gives the last head fewer channels than the first, and neither shows up as
/// anything but slightly wrong text.
fn rope_grid(rotary: i32, width: i32, head_dim: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if rotary <= 0 {
        return Err(Refusal::Empty { what: "rotary" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if head_dim <= 0 {
        return Err(Refusal::Empty { what: "head_dim" });
    }
    if rotary % 2 != 0 {
        return Err(Refusal::Narrow {
            what: "rotary is not a whole number of pairs",
            at: i64::from(rotary),
        });
    }
    if width <= 0 || width % head_dim != 0 {
        return Err(Refusal::Narrow {
            what: "width is not a whole number of heads",
            at: i64::from(width),
        });
    }
    Ok([
        rotary.unsigned_abs() / 2,
        width.unsigned_abs() / head_dim.unsigned_abs(),
        rows.unsigned_abs(),
    ])
}

/// NeoX rotary over ONE row, the angle from `base`.
///
/// The rotation is in place: `x` is the only buffer and it is both operand
/// and result. That is why this family states no `in_place` pair the way
/// `norm::add_bias` does — a rotation's statement has no separate input to
/// alias.
///
/// # Errors
///
/// See `rope_grid`.
pub fn neox_decode(
    ctx: &Ctx<'_>,
    x: BufMut,
    position: I32s,
    scale: f32,
    base: f32,
    head_dim: i32,
    rotary: Env<i32>,
    width: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "rope/neox.wgsl",
            entrypoint: "neox_decode_bfloat16",
            lanes: rope_grid(*rotary, *width, head_dim, 1)?,
        },
        &[x.v(), position.v(), scale.v(), base.v(), head_dim.v()],
    )
}

/// [`neox_decode`] over many rows.
///
/// # Errors
///
/// See `rope_grid`.
pub fn neox_mb(
    ctx: &Ctx<'_>,
    x: BufMut,
    position: I32s,
    scale: f32,
    base: f32,
    head_dim: i32,
    rotary: Env<i32>,
    width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "rope/neox.wgsl",
            entrypoint: "neox_mb_bfloat16",
            lanes: rope_grid(*rotary, *width, head_dim, *rows)?,
        },
        &[x.v(), position.v(), scale.v(), base.v(), head_dim.v()],
    )
}

/// [`neox_decode`] with the angles read from a TABLE rather than derived.
///
/// The long-context families (yarn, llama3) precompute an inverse-frequency
/// vector the driver stages; `mscale` is the attention rescale that rides
/// with it.
///
/// # Errors
///
/// See `rope_grid`.
pub fn neox_freqs_decode(
    ctx: &Ctx<'_>,
    x: BufMut,
    position: I32s,
    scale: f32,
    inv_freq: Buf,
    head_dim: i32,
    mscale: f32,
    rotary: Env<i32>,
    width: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "rope/neox.wgsl",
            entrypoint: "neox_freqs_decode_bfloat16",
            lanes: rope_grid(*rotary, *width, head_dim, 1)?,
        },
        &[
            x.v(),
            position.v(),
            scale.v(),
            inv_freq.v(),
            head_dim.v(),
            mscale.v(),
        ],
    )
}

/// [`neox_freqs_decode`] over many rows.
///
/// # Errors
///
/// See `rope_grid`.
pub fn neox_freqs_mb(
    ctx: &Ctx<'_>,
    x: BufMut,
    position: I32s,
    scale: f32,
    inv_freq: Buf,
    head_dim: i32,
    mscale: f32,
    rotary: Env<i32>,
    width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "rope/neox.wgsl",
            entrypoint: "neox_freqs_mb_bfloat16",
            lanes: rope_grid(*rotary, *width, head_dim, *rows)?,
        },
        &[
            x.v(),
            position.v(),
            scale.v(),
            inv_freq.v(),
            head_dim.v(),
            mscale.v(),
        ],
    )
}

/// [`neox_decode`] rotating only the first `rotary` channels of each head.
///
/// qwen3.5's partial rotary. The channels past `rotary` pass through, which
/// is why the grid is built on `rotary` and the head width is still needed:
/// one states how far to rotate and the other how far apart the heads are.
///
/// # Errors
///
/// See `rope_grid`.
pub fn neox_prop_decode(
    ctx: &Ctx<'_>,
    x: BufMut,
    position: I32s,
    scale: f32,
    base: f32,
    head_dim: i32,
    rotary: Env<i32>,
    width: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "rope/neox.wgsl",
            entrypoint: "neox_prop_decode_bfloat16",
            lanes: rope_grid(*rotary, *width, head_dim, 1)?,
        },
        &[x.v(), position.v(), scale.v(), base.v(), head_dim.v()],
    )
}

/// [`neox_prop_decode`] over many rows.
///
/// # Errors
///
/// See `rope_grid`.
pub fn neox_prop_mb(
    ctx: &Ctx<'_>,
    x: BufMut,
    position: I32s,
    scale: f32,
    base: f32,
    head_dim: i32,
    rotary: Env<i32>,
    width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "rope/neox.wgsl",
            entrypoint: "neox_prop_mb_bfloat16",
            lanes: rope_grid(*rotary, *width, head_dim, *rows)?,
        },
        &[x.v(), position.v(), scale.v(), base.v(), head_dim.v()],
    )
}

/// [`neox_mb`] over rows a `row_pitch` apart rather than a width apart.
///
/// # Errors
///
/// See `rope_grid`.
pub fn neox_strided(
    ctx: &Ctx<'_>,
    x: BufMut,
    position: I32s,
    scale: f32,
    base: f32,
    head_dim: i32,
    row_pitch: i32,
    rotary: Env<i32>,
    width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "rope/neox.wgsl",
            entrypoint: "neox_strided_bfloat16",
            lanes: rope_grid(*rotary, *width, head_dim, *rows)?,
        },
        &[
            x.v(),
            position.v(),
            scale.v(),
            base.v(),
            head_dim.v(),
            row_pitch.v(),
        ],
    )
}

pub static ROUTINES: &[Routine] = &[
    crate::routine!(neox_decode),
    crate::routine!(neox_freqs_decode),
    crate::routine!(neox_freqs_mb),
    crate::routine!(neox_mb),
    crate::routine!(neox_prop_decode),
    crate::routine!(neox_prop_mb),
    crate::routine!(neox_strided),
];
