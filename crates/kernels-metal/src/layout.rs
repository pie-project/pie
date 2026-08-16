//! Gathers and embeddings -- the kernels that move rows rather than
//! compute over them.
//!
//! A routine's arguments are its kernel's bindings, and a quantised gather
//! binds five buffers, two scalars and two axis facts. Nine is what that
//! kernel takes; collecting them into a struct would restate the binding order
//! somewhere else, which is the thing this refactor removes.
#![allow(clippy::too_many_arguments)]

use kernels::routine::Refusal;

use crate::routine::{
    Bind, Buf, BufMut, Ctx, Env, Fire, I32s, InPacked, Routine, U32s, elementwise, elementwise_rows,
};

/// Threads per threadgroup for every body in this file.
///
/// `driver-metal`'s `grid::embed`, `launch::embed_rows` and
/// `grid::elementwise_mb` all state `[256, 1, 1]`, and all three are rules
/// this family's rows use. Metal declares no group size in the source, so the
/// number has to be stated somewhere and this is the same statement moved.
const GROUP_X: u32 = 256;

/// The shaders this family's routines reach: `(file, entrypoint)`, one pair
/// per instantiated name.
///
/// A row's `axes` GENERATED these names and its `file` column said where they
/// live. Retiring the row moved who NAMES them, not what exists -- the shader
/// is still compiled and still dispatched -- so the pairs are stated here and
/// [`crate::entrypoints`] reads them back. The FILE rides along because Metal
/// compiles from `(path, entry name)` at run time, and `device_kernels.rs`
/// builds every one of them against a real device; a name without its file
/// would leave that sweep nothing to open. See [`crate::RETIRED`].
pub static ENTRYPOINTS: &[(&str, &str)] = &[
    (
        "layout/embed_gather.metal",
        "embed_gather_4bit_bfloat16_gs_128_b_4",
    ),
    (
        "layout/embed_gather.metal",
        "embed_gather_4bit_bfloat16_gs_128_b_8",
    ),
    (
        "layout/embed_gather.metal",
        "embed_gather_4bit_bfloat16_gs_32_b_4",
    ),
    (
        "layout/embed_gather.metal",
        "embed_gather_4bit_bfloat16_gs_32_b_8",
    ),
    (
        "layout/embed_gather.metal",
        "embed_gather_4bit_bfloat16_gs_64_b_4",
    ),
    (
        "layout/embed_gather.metal",
        "embed_gather_4bit_bfloat16_gs_64_b_8",
    ),
    (
        "layout/embed_gather.metal",
        "embed_gather_mb_4bit_bfloat16_gs_128_b_4",
    ),
    (
        "layout/embed_gather.metal",
        "embed_gather_mb_4bit_bfloat16_gs_128_b_8",
    ),
    (
        "layout/embed_gather.metal",
        "embed_gather_mb_4bit_bfloat16_gs_32_b_4",
    ),
    (
        "layout/embed_gather.metal",
        "embed_gather_mb_4bit_bfloat16_gs_32_b_8",
    ),
    (
        "layout/embed_gather.metal",
        "embed_gather_mb_4bit_bfloat16_gs_64_b_4",
    ),
    (
        "layout/embed_gather.metal",
        "embed_gather_mb_4bit_bfloat16_gs_64_b_8",
    ),
    (
        "layout/embed_gather.metal",
        "embed_gather_scaled_4bit_bfloat16_gs_128_b_4",
    ),
    (
        "layout/embed_gather.metal",
        "embed_gather_scaled_4bit_bfloat16_gs_128_b_8",
    ),
    (
        "layout/embed_gather.metal",
        "embed_gather_scaled_4bit_bfloat16_gs_32_b_4",
    ),
    (
        "layout/embed_gather.metal",
        "embed_gather_scaled_4bit_bfloat16_gs_32_b_8",
    ),
    (
        "layout/embed_gather.metal",
        "embed_gather_scaled_4bit_bfloat16_gs_64_b_4",
    ),
    (
        "layout/embed_gather.metal",
        "embed_gather_scaled_4bit_bfloat16_gs_64_b_8",
    ),
    (
        "layout/embed_gather.metal",
        "embed_gather_scaled_mb_4bit_bfloat16_gs_128_b_4",
    ),
    (
        "layout/embed_gather.metal",
        "embed_gather_scaled_mb_4bit_bfloat16_gs_128_b_8",
    ),
    (
        "layout/embed_gather.metal",
        "embed_gather_scaled_mb_4bit_bfloat16_gs_32_b_4",
    ),
    (
        "layout/embed_gather.metal",
        "embed_gather_scaled_mb_4bit_bfloat16_gs_32_b_8",
    ),
    (
        "layout/embed_gather.metal",
        "embed_gather_scaled_mb_4bit_bfloat16_gs_64_b_4",
    ),
    (
        "layout/embed_gather.metal",
        "embed_gather_scaled_mb_4bit_bfloat16_gs_64_b_8",
    ),
    ("layout/ple_combine.metal", "ple_combine_bfloat16"),
    ("layout/row_gather.metal", "row_gather_bfloat16"),
];

/// The point in the affine axis a spelling exists for.
///
/// `metal::instantiate_embed_gather(bfloat16, bfloat, 32, 4)` and five
/// siblings, and nothing else exists to name.
///
/// Written the way `kernels-vulkan` and `kernels-wgpu` write it, and for the
/// same reason: LITERAL tables rather than a paste.
/// `format!("..._gs_{group}_b_{bits}")` would spell `_gs_96_b_3` as readily as
/// `_gs_64_b_4`, and Metal does not find out until `newFunctionWithName:`
/// returns nil at RUN time -- inside a fire, after the plan was accepted. A
/// point the tree does not carry cannot be spelled at all.
///
/// # Errors
///
/// [`Refusal::Narrow`], carrying the extent that was not one of the six, so
/// the caller can fall back rather than fault.
fn affine_point(group: i32, bits: i32) -> Result<usize, Refusal> {
    let g = match group {
        32 => 0,
        64 => 1,
        128 => 2,
        _ => {
            return Err(Refusal::Narrow {
                what: "affine group size",
                at: i64::from(group),
            });
        }
    };
    let b = match bits {
        4 => 0,
        8 => 1,
        _ => {
            return Err(Refusal::Narrow {
                what: "affine bit width",
                at: i64::from(bits),
            });
        }
    };
    Ok(g * 2 + b)
}

/// The six `embed_gather_4bit` entrypoints, in [`affine_point`] order.
///
/// `_4bit` is in the name of the 8-bit ones too. That is the SYMBOL, not a
/// mistake being copied: the name predates the bit axis and the axis suffix is
/// what distinguishes them. The shaders are the authority and nothing holds
/// this list to them any more: `scripts/metal-kernel-audit.py --table` did,
/// and it is retired because the example it read the table with is deleted.
/// The script's census (no flag) still prints what the shaders instantiate, so
/// the comparison is available by eye and by nothing else.
static EMBED_GATHER: [&str; 6] = [
    "embed_gather_4bit_bfloat16_gs_32_b_4",
    "embed_gather_4bit_bfloat16_gs_32_b_8",
    "embed_gather_4bit_bfloat16_gs_64_b_4",
    "embed_gather_4bit_bfloat16_gs_64_b_8",
    "embed_gather_4bit_bfloat16_gs_128_b_4",
    "embed_gather_4bit_bfloat16_gs_128_b_8",
];

/// The same, over M>1 rows.
static EMBED_GATHER_MB: [&str; 6] = [
    "embed_gather_mb_4bit_bfloat16_gs_32_b_4",
    "embed_gather_mb_4bit_bfloat16_gs_32_b_8",
    "embed_gather_mb_4bit_bfloat16_gs_64_b_4",
    "embed_gather_mb_4bit_bfloat16_gs_64_b_8",
    "embed_gather_mb_4bit_bfloat16_gs_128_b_4",
    "embed_gather_mb_4bit_bfloat16_gs_128_b_8",
];

/// The same, with the embedding scale folded in.
static EMBED_GATHER_SCALED: [&str; 6] = [
    "embed_gather_scaled_4bit_bfloat16_gs_32_b_4",
    "embed_gather_scaled_4bit_bfloat16_gs_32_b_8",
    "embed_gather_scaled_4bit_bfloat16_gs_64_b_4",
    "embed_gather_scaled_4bit_bfloat16_gs_64_b_8",
    "embed_gather_scaled_4bit_bfloat16_gs_128_b_4",
    "embed_gather_scaled_4bit_bfloat16_gs_128_b_8",
];

/// The same again, over M>1 rows.
static EMBED_GATHER_SCALED_MB: [&str; 6] = [
    "embed_gather_scaled_mb_4bit_bfloat16_gs_32_b_4",
    "embed_gather_scaled_mb_4bit_bfloat16_gs_32_b_8",
    "embed_gather_scaled_mb_4bit_bfloat16_gs_64_b_4",
    "embed_gather_scaled_mb_4bit_bfloat16_gs_64_b_8",
    "embed_gather_scaled_mb_4bit_bfloat16_gs_128_b_4",
    "embed_gather_scaled_mb_4bit_bfloat16_gs_128_b_8",
];

/// The shader all four gathers are compiled from.
const EMBED_GATHER_FILE: &str = "layout/embed_gather.metal";

/// The gather that reads ONE row of an affine-quantised embedding table.
///
/// The grid is `hidden` lanes on x and the shader takes `m = 0`, so it reads
/// `id[0]` whatever grid it is handed. That is why the M>1 twin exists as a
/// separate symbol rather than this one over a taller rectangle.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty rectangle, [`Refusal::Narrow`] for an
/// affine point the shader tree does not carry.
pub fn embed_gather_4bit(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    id: I32s,
    out: BufMut,
    hidden: i32,
    group: Env<i32>,
    bits: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: EMBED_GATHER[affine_point(*group, *bits)?],
            file: EMBED_GATHER_FILE,
            lanes: elementwise(hidden, 1)?,
            group: [GROUP_X, 1, 1],
        },
        &[w.v(), scales.v(), biases.v(), id.v(), out.v(), hidden.v()],
    )
}

/// The M>1 form, and the one a text should name.
///
/// The rows get their own axis -- `LaunchRule::ElementwiseRows`, which is a
/// different grid from [`embed_gather_4bit`]'s and not a taller one.
/// Flattening it would visit `hidden * rows` lanes with `k >= hidden` for all
/// but the first row, every one of which returns early, and write exactly one
/// row.
///
/// # Errors
///
/// As [`embed_gather_4bit`].
pub fn embed_gather_mb_4bit(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    id: I32s,
    out: BufMut,
    hidden: i32,
    rows: Env<i32>,
    group: Env<i32>,
    bits: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: EMBED_GATHER_MB[affine_point(*group, *bits)?],
            file: EMBED_GATHER_FILE,
            lanes: elementwise_rows(hidden, *rows)?,
            group: [GROUP_X, 1, 1],
        },
        &[w.v(), scales.v(), biases.v(), id.v(), out.v(), hidden.v()],
    )
}

/// [`embed_gather_4bit`] with the embedding scale folded in.
///
/// gemma multiplies its embeddings by `sqrt(hidden)`. That is a number the
/// STATEMENT carries, not one the kernel knows, which is why it is an argument
/// and why the unscaled twin is a different symbol rather than this one with a
/// `1.0`.
///
/// # Errors
///
/// As [`embed_gather_4bit`].
pub fn embed_gather_scaled_4bit(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    id: I32s,
    out: BufMut,
    hidden: i32,
    embed_scale: f32,
    group: Env<i32>,
    bits: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: EMBED_GATHER_SCALED[affine_point(*group, *bits)?],
            file: EMBED_GATHER_FILE,
            lanes: elementwise(hidden, 1)?,
            group: [GROUP_X, 1, 1],
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            id.v(),
            out.v(),
            hidden.v(),
            embed_scale.v(),
        ],
    )
}

/// The M>1 form of [`embed_gather_scaled_4bit`], and the one a text should
/// name.
///
/// # Errors
///
/// As [`embed_gather_4bit`].
pub fn embed_gather_scaled_mb_4bit(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    id: I32s,
    out: BufMut,
    hidden: i32,
    embed_scale: f32,
    rows: Env<i32>,
    group: Env<i32>,
    bits: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: EMBED_GATHER_SCALED_MB[affine_point(*group, *bits)?],
            file: EMBED_GATHER_FILE,
            lanes: elementwise_rows(hidden, *rows)?,
            group: [GROUP_X, 1, 1],
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            id.v(),
            out.v(),
            hidden.v(),
            embed_scale.v(),
        ],
    )
}

/// gemma's PLE join: `(proj + token) * inv_sqrt2`, over the whole
/// `[n_layers, ple_dim]` block at once.
///
/// The scale is `1/sqrt(2)` and it is the JOIN's, not a deployment's -- two
/// streams averaged in the root-mean-square sense -- so it rides the packed
/// `PleCombineParams` buffer rather than an argument.
///
/// `width` and `rows` are [`Env`]: the statement does not carry them and the
/// environment always does. Under `kernel!` they were not arguments at all --
/// `LaunchRule::Elementwise` told the driver to read them off the rectangle.
/// A body that needs them says so.
///
/// # Errors
///
/// [`Refusal::Empty`] when the block is empty.
pub fn ple_combine(
    ctx: &Ctx<'_>,
    proj: Buf,
    token: Buf,
    out: BufMut,
    params: Buf,
    width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "ple_combine_bfloat16",
            file: "layout/ple_combine.metal",
            lanes: elementwise(*width, *rows)?,
            group: [GROUP_X, 1, 1],
        },
        &[proj.v(), token.v(), out.v(), params.v()],
    )
}

/// The readout's gather: one distribution per REQUEST out of one row per
/// TOKEN.
///
/// A prefill's stream is one row per token and its readout is one distribution
/// per request, so the sampled rows are picked out before the lm head runs.
///
/// `count` is [`InPacked`], and on Metal that means "APPEND to the scalar
/// run": `RowGatherParams` is width then count packed into buffer 3, there is
/// no buffer 4, and a packed slot's run already covers every scalar after it.
/// The statement states `[width]`; the driver appends the count, giving
/// `[width, count]` -- exactly the struct. Vulkan reads the same `Ty::InPacked`
/// differently because there the struct is a std430 buffer and the scalars go
/// to a push block, so there is nothing to append to. The TYPE is now the only
/// statement of it; under `kernel!` the fact was said twice, in the row and in
/// the shader's struct, and only one was checked.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty rectangle.
pub fn row_gather(
    ctx: &Ctx<'_>,
    input: Buf,
    out: BufMut,
    rows: U32s,
    params: Buf,
    count: InPacked,
    width: Env<i32>,
    row_count: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "row_gather_bfloat16",
            file: "layout/row_gather.metal",
            lanes: elementwise_rows(*width, *row_count)?,
            group: [GROUP_X, 1, 1],
        },
        &[input.v(), out.v(), rows.v(), params.v(), count.v()],
    )
}

/// This family's routines.
pub static ROUTINES: &[Routine] = &[
    crate::routine!(embed_gather_4bit),
    crate::routine!(embed_gather_mb_4bit),
    crate::routine!(embed_gather_scaled_4bit),
    crate::routine!(embed_gather_scaled_mb_4bit),
    crate::routine!(ple_combine),
    crate::routine!(row_gather),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{ArgValue, Encode};
    use core::cell::RefCell;

    /// One recorded dispatch: the fire, and the argument list.
    type Call = (Fire, Vec<ArgValue>);

    /// An `Encode` that remembers what it was asked to do.
    #[derive(Default)]
    struct Seen(RefCell<Vec<Call>>);

    impl Encode for Seen {
        fn dispatch(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
            self.0.borrow_mut().push((fire, args.to_vec()));
            Ok(())
        }
    }

    /// A body PICKS its spelling out of a table; it does not build one.
    ///
    /// The failure this shape exists to prevent is Metal's own: a name that no
    /// `[[host_name]]` declares makes `newFunctionWithName:` return nil, at
    /// RUN time, inside a fire, after the plan was accepted. `format!` would
    /// spell `_gs_96_b_3` as readily as `_gs_64_b_4`; a literal table cannot.
    #[test]
    fn the_affine_point_picks_a_spelling_the_shader_tree_declares() {
        let seen = Seen::default();
        embed_gather_4bit(
            &seen,
            Buf(1),
            Buf(2),
            Buf(3),
            I32s(4),
            BufMut(5),
            2048,
            Env(64),
            Env(4),
        )
        .expect("a 64/4 checkpoint is one of the six");

        assert_eq!(
            seen.0.borrow()[0].0.entrypoint,
            "embed_gather_4bit_bfloat16_gs_64_b_4"
        );

        let narrow = embed_gather_4bit(
            &seen,
            Buf(1),
            Buf(2),
            Buf(3),
            I32s(4),
            BufMut(5),
            2048,
            Env(96),
            Env(4),
        )
        .expect_err("96 is not a group size this tree carries");
        assert!(
            matches!(
                narrow,
                Refusal::Narrow {
                    what: "affine group size",
                    at: 96
                }
            ),
            "got {narrow:?} -- the extent has to ride the refusal so a caller \
             can fall back rather than fault"
        );
        assert_eq!(
            seen.0.borrow().len(),
            1,
            "and nothing was encoded on the way to refusing"
        );
    }

    /// The two grids in this family are a DIFFERENT one and a taller one is
    /// not what the second is.
    ///
    /// `embed_gather_4bit` puts `hidden` threads on x and the shader takes
    /// `m = 0`, so it reads `id[0]` whatever grid it is handed. Its M>1 twin
    /// puts the rows on their own axis. Handing the single-row symbol a taller
    /// rectangle -- which is what flattening the twin would amount to -- gathers
    /// token zero into every row.
    #[test]
    fn the_single_row_gather_and_its_m_gt_1_twin_are_two_grids_and_not_one() {
        let seen = Seen::default();
        embed_gather_4bit(
            &seen,
            Buf(1),
            Buf(2),
            Buf(3),
            I32s(4),
            BufMut(5),
            2048,
            Env(32),
            Env(4),
        )
        .expect("a launch");
        embed_gather_mb_4bit(
            &seen,
            Buf(1),
            Buf(2),
            Buf(3),
            I32s(4),
            BufMut(5),
            2048,
            Env(7),
            Env(32),
            Env(4),
        )
        .expect("a launch");

        let calls = seen.0.borrow();
        assert_eq!(
            calls[0].0.lanes,
            [2048, 1, 1],
            "one row, and the row count is not multiplied in"
        );
        assert_eq!(
            calls[1].0.lanes,
            [2048, 7, 1],
            "seven rows on their OWN axis -- [2048 * 7, 1, 1] would visit the \
             right number of lanes and write one row, because the shader reads \
             its row off y"
        );
        assert_eq!(calls[0].0.group, [GROUP_X, 1, 1]);
        assert_eq!(calls[1].0.group, [GROUP_X, 1, 1]);
    }

    /// The scaled gathers carry the scale as a trailing argument, after
    /// `hidden`.
    ///
    /// gemma's `sqrt(hidden)` is the STATEMENT's number. Its position is the
    /// shader's argument order and nothing else, and a scale bound where the
    /// width goes is a model that runs and is wrong.
    #[test]
    fn the_embedding_scale_rides_after_the_width_and_not_before_it() {
        let seen = Seen::default();
        embed_gather_scaled_mb_4bit(
            &seen,
            Buf(1),
            Buf(2),
            Buf(3),
            I32s(4),
            BufMut(5),
            2048,
            45.254_834,
            Env(7),
            Env(128),
            Env(8),
        )
        .expect("a launch");

        let calls = seen.0.borrow();
        let (fire, args) = &calls[0];
        assert_eq!(
            fire.entrypoint,
            "embed_gather_scaled_mb_4bit_bfloat16_gs_128_b_8"
        );
        assert_eq!(
            args[5],
            ArgValue::I32(2048),
            "the width is the sixth argument"
        );
        assert_eq!(
            args[6],
            ArgValue::F32(45.254_834),
            "and the scale is the seventh, which is where the shader declares it"
        );
        assert_eq!(args.len(), 7, "and there is no eighth");
    }

    /// `row_gather`'s count is an argument, and it is the LAST one.
    ///
    /// On Metal `Ty::InPacked` means "append to the scalar run": the struct is
    /// `{width, count}` packed into buffer 3 and there is no buffer 4, so the
    /// count is the run's second word rather than a slot of its own. The
    /// statement states the width; the body states that something appends a
    /// count, once, in the type.
    #[test]
    fn the_row_gathers_count_is_the_last_argument_because_it_rides_the_pack() {
        let seen = Seen::default();
        row_gather(
            &seen,
            Buf(1),
            BufMut(2),
            U32s(3),
            Buf(4),
            InPacked(3),
            Env(2048),
            Env(3),
        )
        .expect("three requests is a launch");

        let calls = seen.0.borrow();
        let (fire, args) = &calls[0];
        assert_eq!(fire.entrypoint, "row_gather_bfloat16");
        assert_eq!(fire.file, "layout/row_gather.metal");
        assert_eq!(
            fire.lanes,
            [2048, 3, 1],
            "one row per REQUEST, not one per token -- that is the whole point \
             of the gather"
        );
        assert_eq!(
            args.len(),
            5,
            "input, out, rows, params and the count that rides inside params"
        );
        assert_eq!(
            args[4],
            ArgValue::U32(3),
            "and it carries the request count, which is the struct's second word"
        );
    }
}
