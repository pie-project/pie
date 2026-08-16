#![allow(clippy::too_many_arguments)]
//! Rotary embeddings.
//!
//! Four spellings of the schedule (`neox`, `freqs`, `prop`, and the strided
//! form), each in a decode and a multi-batch shape. The `freqs` pair reads a
//! host-computed table, which is what llama-3.1's wavelength ramp needs.

use kernels::KernelSig;

/// EMPTY: this family's rows have been RETIRED.
///
/// `refactor-bigplan.md` §7 Stage 3. Seven kernels, and the rotation is IN
/// PLACE — one `BufMut` and no `Buf` beside it — so the statement's single
/// widthed operand is an OUTPUT and an arm asking `input(0)` would refuse
/// every rotation in the tree.
pub static KERNELS: &[KernelSig] = &[];

/// The entrypoints this family's routines spell, now that its rows are gone.
///
/// See [`crate::sample::ENTRYPOINTS`].
pub static ENTRYPOINTS: &[&str] = &[
    "neox_decode_bfloat16",
    "neox_freqs_decode_bfloat16",
    "neox_freqs_mb_bfloat16",
    "neox_mb_bfloat16",
    "neox_prop_decode_bfloat16",
    "neox_prop_mb_bfloat16",
    "neox_strided_bfloat16",
];

use crate::routine::{keys, Ask, Bind, Buf, BufMut, Ctx, Fire, I32s, Param, ParamF32, ParamOr, Routine};
use crate::routine::OutSlot;
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
    x: OutSlot<0, BufMut>,
    position: Ask<keys::Positions, I32s>,
    scale: ParamF32<0>,
    base: ParamF32<1>,
    head_dim: ParamOr<2, keys::HeadDim, i32>,
    rotary: ParamOr<3, keys::RotaryWidth, i32>,
    width: Ask<keys::Width, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "rope/neox.wgsl",
            entrypoint: "neox_decode_bfloat16",
            lanes: rope_grid(*rotary, *width, *head_dim, 1)?,
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
    x: OutSlot<0, BufMut>,
    position: Ask<keys::Positions, I32s>,
    scale: ParamF32<0>,
    base: ParamF32<1>,
    head_dim: ParamOr<2, keys::HeadDim, i32>,
    rotary: ParamOr<3, keys::RotaryWidth, i32>,
    width: Ask<keys::Width, i32>,
    rows: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "rope/neox.wgsl",
            entrypoint: "neox_mb_bfloat16",
            lanes: rope_grid(*rotary, *width, *head_dim, *rows)?,
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
    x: OutSlot<0, BufMut>,
    position: Ask<keys::Positions, I32s>,
    scale: ParamF32<0>,
    inv_freq: Ask<keys::RopeFrequencies, Buf>,
    head_dim: ParamOr<1, keys::HeadDim, i32>,
    mscale: ParamF32<2>,
    rotary: ParamOr<3, keys::RotaryWidth, i32>,
    width: Ask<keys::Width, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "rope/neox.wgsl",
            entrypoint: "neox_freqs_decode_bfloat16",
            lanes: rope_grid(*rotary, *width, *head_dim, 1)?,
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
    x: OutSlot<0, BufMut>,
    position: Ask<keys::Positions, I32s>,
    scale: ParamF32<0>,
    inv_freq: Ask<keys::RopeFrequencies, Buf>,
    head_dim: ParamOr<1, keys::HeadDim, i32>,
    mscale: ParamF32<2>,
    rotary: ParamOr<3, keys::RotaryWidth, i32>,
    width: Ask<keys::Width, i32>,
    rows: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "rope/neox.wgsl",
            entrypoint: "neox_freqs_mb_bfloat16",
            lanes: rope_grid(*rotary, *width, *head_dim, *rows)?,
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
    x: OutSlot<0, BufMut>,
    position: Ask<keys::Positions, I32s>,
    scale: ParamF32<0>,
    base: ParamF32<1>,
    head_dim: ParamOr<2, keys::HeadDim, i32>,
    rotary: ParamOr<3, keys::RotaryWidth, i32>,
    width: Ask<keys::Width, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "rope/neox.wgsl",
            entrypoint: "neox_prop_decode_bfloat16",
            lanes: rope_grid(*rotary, *width, *head_dim, 1)?,
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
    x: OutSlot<0, BufMut>,
    position: Ask<keys::Positions, I32s>,
    scale: ParamF32<0>,
    base: ParamF32<1>,
    head_dim: ParamOr<2, keys::HeadDim, i32>,
    rotary: ParamOr<3, keys::RotaryWidth, i32>,
    width: Ask<keys::Width, i32>,
    rows: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "rope/neox.wgsl",
            entrypoint: "neox_prop_mb_bfloat16",
            lanes: rope_grid(*rotary, *width, *head_dim, *rows)?,
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
    x: OutSlot<0, BufMut>,
    position: Ask<keys::Positions, I32s>,
    scale: ParamF32<0>,
    base: ParamF32<1>,
    head_dim: ParamOr<2, keys::HeadDim, i32>,
    row_pitch: Param<4, i32>,
    rotary: ParamOr<3, keys::RotaryWidth, i32>,
    width: Ask<keys::Width, i32>,
    rows: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "rope/neox.wgsl",
            entrypoint: "neox_strided_bfloat16",
            lanes: rope_grid(*rotary, *width, *head_dim, *rows)?,
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

/// This family's routines.
///
/// EVERY ONE STATES `in_place = &[(0, 0)]`, and an earlier draft that removed
/// it had the right gate and the wrong word. The claim then was that a
/// rotation's single operand is an `Out` with no input to alias -- but
/// `dsl::metal::rope_one`, the only place any plane's neox is stated, places
/// that operand as an INPUT and declares no result on purpose. A statement
/// that declared a separate result had `Out(0)` bind the RESULT's slot, which
/// no kernel had written, and the rotated value everything downstream wanted
/// was never produced; position zero makes rope the identity, so the first
/// reference gate agreed anyway. Stating no result is what makes
/// `dispatch::reorder` bind `Out(0)` to the one widthed operand -- the input,
/// the buffer the kernel mutates.
///
/// So `(0, 0)` is not a second spelling of `BufMut`. It is the only thing that
/// says the write and the placement are ONE buffer, which §6.2's arity rule
/// needs on both sides: without it a rotation reads NOTHING against a
/// statement placing one operand, and writes one pointer against a statement
/// declaring none.
pub static ROUTINES: &[Routine] = &[
    crate::routine!(neox_decode, in_place = &[(0, 0)]),
    crate::routine!(neox_freqs_decode, in_place = &[(0, 0)]),
    crate::routine!(neox_freqs_mb, in_place = &[(0, 0)]),
    crate::routine!(neox_mb, in_place = &[(0, 0)]),
    crate::routine!(neox_prop_decode, in_place = &[(0, 0)]),
    crate::routine!(neox_prop_mb, in_place = &[(0, 0)]),
    crate::routine!(neox_strided, in_place = &[(0, 0)]),
];
