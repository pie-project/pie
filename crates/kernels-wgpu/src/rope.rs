//! Rotary embeddings.
//!
//! Four spellings of the schedule (`neox`, `freqs`, `prop`, and the strided
//! form), each in a decode and a multi-batch shape. The `freqs` pair reads a
//! host-computed table, which is what llama-3.1's wavelength ramp needs.

use kernels_macros::routine;
use kernels::KernelSig;

/// EMPTY: this family's rows have been RETIRED.
///
/// `refactor-bigplan.md` §7 Stage 3. Seven kernels, and the rotation is IN
/// PLACE — one `Buf` and no `Buf` beside it — so the statement's single
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

use crate::routine::{Asks, Bind, Const, Ctx, Fire, InOut, Tensor, bf16, keys};
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
#[routine]
pub fn neox_decode(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    base: Const<f32>,
    head_dim: Const<i32>,
    rotary: Const<i32>) -> Result<(), Refusal> {
    let position = ctx.ask::<Tensor<i32>, keys::Positions>()?;
    let width = x.width;
    ctx.fire(
        Fire::at("rope/neox.wgsl", "neox_decode_bfloat16").apply(rope_grid(*rotary, width, *head_dim, 1)?),
        &[x.arg(), position.arg(), scale.arg(), base.arg(), head_dim.arg()],
    )
}

/// [`neox_decode`] over many rows.
///
/// # Errors
///
/// See `rope_grid`.
#[routine]
pub fn neox_mb(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    base: Const<f32>,
    head_dim: Const<i32>,
    rotary: Const<i32>) -> Result<(), Refusal> {
    let position = ctx.ask::<Tensor<i32>, keys::Positions>()?;
    let width = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("rope/neox.wgsl", "neox_mb_bfloat16").apply(rope_grid(*rotary, width, *head_dim, rows)?),
        &[x.arg(), position.arg(), scale.arg(), base.arg(), head_dim.arg()],
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
#[routine]
pub fn neox_freqs_decode(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    head_dim: Const<i32>,
    mscale: Const<f32>,
    rotary: Const<i32>) -> Result<(), Refusal> {
    // THE FIRE'S FREQUENCY TABLE, ASKED FOR. `Ask<keys::RopeFrequencies, Buf>`
    // before the marks: a table the driver builds once per fire, not a weight
    // the checkpoint carries and no builder places one. As a
    // `Const<Tensor<f32>>` it asked the statement for a weight operand that
    // is not there, and every gpt-oss rotation refused.
    let inv_freq = ctx.ask::<Tensor<f32>, keys::RopeFrequencies>()?;

    let position = ctx.ask::<Tensor<i32>, keys::Positions>()?;
    let width = x.width;
    ctx.fire(
        Fire::at("rope/neox.wgsl", "neox_freqs_decode_bfloat16").apply(rope_grid(*rotary, width, *head_dim, 1)?),
        &[
            x.arg(),
            position.arg(),
            scale.arg(),
            inv_freq.arg(),
            head_dim.arg(),
            mscale.arg(),
        ],
    )
}

/// [`neox_freqs_decode`] over many rows.
///
/// # Errors
///
/// See `rope_grid`.
#[routine]
pub fn neox_freqs_mb(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    head_dim: Const<i32>,
    mscale: Const<f32>,
    rotary: Const<i32>) -> Result<(), Refusal> {
    // THE FIRE'S FREQUENCY TABLE, ASKED FOR. `Ask<keys::RopeFrequencies, Buf>`
    // before the marks: a table the driver builds once per fire, not a weight
    // the checkpoint carries and no builder places one. As a
    // `Const<Tensor<f32>>` it asked the statement for a weight operand that
    // is not there, and every gpt-oss rotation refused.
    let inv_freq = ctx.ask::<Tensor<f32>, keys::RopeFrequencies>()?;

    let position = ctx.ask::<Tensor<i32>, keys::Positions>()?;
    let width = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("rope/neox.wgsl", "neox_freqs_mb_bfloat16").apply(rope_grid(*rotary, width, *head_dim, rows)?),
        &[
            x.arg(),
            position.arg(),
            scale.arg(),
            inv_freq.arg(),
            head_dim.arg(),
            mscale.arg(),
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
#[routine]
pub fn neox_prop_decode(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    base: Const<f32>,
    head_dim: Const<i32>,
    rotary: Const<i32>) -> Result<(), Refusal> {
    let position = ctx.ask::<Tensor<i32>, keys::Positions>()?;
    let width = x.width;
    ctx.fire(
        Fire::at("rope/neox.wgsl", "neox_prop_decode_bfloat16").apply(rope_grid(*rotary, width, *head_dim, 1)?),
        &[x.arg(), position.arg(), scale.arg(), base.arg(), head_dim.arg()],
    )
}

/// [`neox_prop_decode`] over many rows.
///
/// # Errors
///
/// See `rope_grid`.
#[routine]
pub fn neox_prop_mb(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    base: Const<f32>,
    head_dim: Const<i32>,
    rotary: Const<i32>) -> Result<(), Refusal> {
    let position = ctx.ask::<Tensor<i32>, keys::Positions>()?;
    let width = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("rope/neox.wgsl", "neox_prop_mb_bfloat16").apply(rope_grid(*rotary, width, *head_dim, rows)?),
        &[x.arg(), position.arg(), scale.arg(), base.arg(), head_dim.arg()],
    )
}

/// [`neox_mb`] over rows a `row_pitch` apart rather than a width apart.
///
/// # Errors
///
/// See `rope_grid`.
#[routine]
pub fn neox_strided(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    scale: Const<f32>,
    base: Const<f32>,
    head_dim: Const<i32>,
    rotary: Const<i32>,
    // THE STATEMENT'S, AND IT WAS `Param<4, i32>`. A pitch is the rectangle
    // the text laid out, not something this batch made -- two fires of one
    // deployment stride the same way -- so it fails `ask`'s own test and no
    // driver answers `keys::RowPitch`. Metal's twin declares it identically;
    // the three planes must ask the binder the same questions.
    row_pitch: Const<i32>) -> Result<(), Refusal> {
    let position = ctx.ask::<Tensor<i32>, keys::Positions>()?;
    let width = x.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("rope/neox.wgsl", "neox_strided_bfloat16").apply(rope_grid(*rotary, width, *head_dim, rows)?),
        &[
            x.arg(),
            position.arg(),
            scale.arg(),
            base.arg(),
            head_dim.arg(),
            row_pitch.arg(),
        ],
    )
}

