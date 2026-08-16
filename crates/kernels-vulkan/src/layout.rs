//! Gathers and embeddings -- the kernels that move rows rather than
//! compute over them.
//!
//! A routine's arguments are its kernel's bindings, and a quantised gather
//! binds five buffers, two scalars and two axis facts. Nine is what that
//! kernel takes; collecting them into a struct would restate the binding order
//! somewhere else, which is the thing this refactor removes.
#![allow(clippy::too_many_arguments)]

use kernels::KernelSig;
use kernels::routine::Refusal;

use crate::routine::{
    Bind, Buf, BufMut, Ctx, Env, Fire, I32s, InPacked, Routine, U32s, elementwise, elementwise_rows,
};

/// The entrypoints this family's crossed routines spell, now that their
/// rows are gone. See [`crate::RETIRED`].
pub static ENTRYPOINTS: &[&str] = &[
    "embed_gather_4bit_bfloat16_gs_32_b_4",
    "embed_gather_4bit_bfloat16_gs_32_b_8",
    "embed_gather_4bit_bfloat16_gs_64_b_4",
    "embed_gather_4bit_bfloat16_gs_64_b_8",
    "embed_gather_4bit_bfloat16_gs_128_b_4",
    "embed_gather_4bit_bfloat16_gs_128_b_8",
    "embed_gather_mb_4bit_bfloat16_gs_32_b_4",
    "embed_gather_mb_4bit_bfloat16_gs_32_b_8",
    "embed_gather_mb_4bit_bfloat16_gs_64_b_4",
    "embed_gather_mb_4bit_bfloat16_gs_64_b_8",
    "embed_gather_mb_4bit_bfloat16_gs_128_b_4",
    "embed_gather_mb_4bit_bfloat16_gs_128_b_8",
    "embed_gather_scaled_4bit_bfloat16_gs_32_b_4",
    "embed_gather_scaled_4bit_bfloat16_gs_32_b_8",
    "embed_gather_scaled_4bit_bfloat16_gs_64_b_4",
    "embed_gather_scaled_4bit_bfloat16_gs_64_b_8",
    "embed_gather_scaled_4bit_bfloat16_gs_128_b_4",
    "embed_gather_scaled_4bit_bfloat16_gs_128_b_8",
    "embed_gather_scaled_mb_4bit_bfloat16_gs_32_b_4",
    "embed_gather_scaled_mb_4bit_bfloat16_gs_32_b_8",
    "embed_gather_scaled_mb_4bit_bfloat16_gs_64_b_4",
    "embed_gather_scaled_mb_4bit_bfloat16_gs_64_b_8",
    "embed_gather_scaled_mb_4bit_bfloat16_gs_128_b_4",
    "embed_gather_scaled_mb_4bit_bfloat16_gs_128_b_8",
    "ple_combine_bfloat16",
    "row_gather_bfloat16",
];

pub static KERNELS: &[KernelSig] = &[];

/// Which of the six affine points `(group, bits)` names, or a refusal.
///
/// The order is group ascending, 4-bit before 8-bit within each. Six points,
/// and they are the six the shader tree carries: `embed_gather.slang` declares
/// `pie:instantiate embed_gather_4bit_bfloat16_gs_32_b_4` and five siblings,
/// and nothing else exists to name.
///
/// `.wiki/kernel-x/vulkan-refactor.md` §4 made concrete. The traced symbol
/// carried `_gs_64_b_4` and `KernelSig::covers_point` stripped it back to find
/// a row; a body picks the spelling instead, out of facts the driver already
/// holds -- the affine format is a checkpoint fact, read once when the weights
/// are mapped.
///
/// Written the same way `kernels-wgpu` writes it, and for the same reason:
/// LITERAL tables rather than a paste. `format!("..._gs_{group}_b_{bits}")`
/// would spell `_gs_96_b_3` as readily as `_gs_64_b_4`, and this backend does
/// not find out until `vkCreateComputePipelines` is handed a module that was
/// never built -- which, with the validation layer silent, is a SIGSEGV rather
/// than an error. A point the tree does not carry cannot be spelled at all.
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
/// what distinguishes them. The `// pie:instantiate` directives are the
/// authority and `tests/routines.rs` reads them.
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

/// The gather that reads ONE row of an affine-quantised embedding table.
///
/// `hidden` is a push constant here -- `embed_gather.slang` declares
/// `struct Push { int hidden; }` -- but that is not this signature's business.
/// Where a scalar rides is the MODULE's decision and `binding::params` reads
/// it off the SPIR-V; a body states that the kernel takes an `i32`, once.
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
            lanes: elementwise(hidden, 1)?,
        },
        &[w.v(), scales.v(), biases.v(), id.v(), out.v(), hidden.v()],
    )
}

/// The M>1 form, and the one a text should name.
///
/// `[numthreads(16, 16, 1)]` and `m = gid.y`, so the rows get their own axis
/// -- `LaunchRule::ElementwiseRows`, which is a different grid from
/// [`embed_gather_4bit`]'s and not a taller one. Flattening it would visit
/// `hidden * rows` lanes with `k >= hidden` for all but the first row, every
/// one of which returns early, and write exactly one row.
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
            lanes: elementwise_rows(hidden, *rows)?,
        },
        &[w.v(), scales.v(), biases.v(), id.v(), out.v(), hidden.v()],
    )
}

/// [`embed_gather_4bit`] with the embedding scale folded in.
///
/// gemma multiplies its embeddings by `sqrt(hidden)`. That is a number the
/// STATEMENT carries, not one the kernel knows, which is why it is an argument
/// and why the unscaled twin is a different symbol rather than this one with a
/// `1.0`: `PIE_SCALED` changes the push block's layout.
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
            lanes: elementwise(hidden, 1)?,
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
            lanes: elementwise_rows(hidden, *rows)?,
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
            lanes: elementwise(*width, *rows)?,
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
/// `count` is [`InPacked`], and on this backend that word does real work.
/// `row_gather.slang` binds `RowGatherParams` as a std430 buffer at 2 and
/// sends plain scalars to a PUSH block, so there is no trailing scalar slot to
/// append a count to: the driver writes it into the struct's second field and
/// the shader reads `p.count`. `kernels-metal` reads the same `Ty::InPacked`
/// as "append to the scalar run", because there the packed slot IS the buffer.
/// The type is now the only statement of it; under `kernel!` the fact was said
/// twice, in the row and in the shader's struct, and only one was checked.
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
            lanes: elementwise_rows(*width, *row_count)?,
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

    type Call = (String, [u32; 3], Vec<ArgValue>);

    #[derive(Default)]
    struct Seen(RefCell<Vec<Call>>);

    impl Encode for Seen {
        fn dispatch(&self, fire: Fire<'_>, args: &[ArgValue]) -> Result<(), Refusal> {
            self.0
                .borrow_mut()
                .push((fire.entrypoint.to_string(), fire.lanes, args.to_vec()));
            Ok(())
        }
    }

    /// The affine point picks the SPELLING, and the six are in the order the
    /// four tables are written in.
    ///
    /// A wrong index here is not a compile error and not a runtime error: it
    /// is a module built for `_gs_64_b_4` storage being handed `_gs_128_b_8`
    /// bytes, which dequantises with the wrong stride and produces embeddings
    /// that are finite, wrong, and never flagged.
    #[test]
    fn the_affine_point_indexes_the_six_spellings_in_order() {
        let spellings: Vec<&str> = POINTS
            .iter()
            .map(|(g, b)| EMBED_GATHER[affine_point(*g, *b).expect("a carried point")])
            .collect();
        assert_eq!(
            spellings,
            vec![
                "embed_gather_4bit_bfloat16_gs_32_b_4",
                "embed_gather_4bit_bfloat16_gs_32_b_8",
                "embed_gather_4bit_bfloat16_gs_64_b_4",
                "embed_gather_4bit_bfloat16_gs_64_b_8",
                "embed_gather_4bit_bfloat16_gs_128_b_4",
                "embed_gather_4bit_bfloat16_gs_128_b_8",
            ],
            "group ascending, 4-bit before 8-bit within each"
        );
    }

    /// The six points this tree carries, in `affine_point` order. Shared with
    /// `tests/routines.rs`, which sweeps every gather over all of them.
    const POINTS: [(i32, i32); 6] = [(32, 4), (32, 8), (64, 4), (64, 8), (128, 4), (128, 8)];

    /// A point the shader tree does not carry is REFUSED, not spelled.
    ///
    /// This is the one place a body can fail in a way the device turns into a
    /// crash rather than a wrong number. `format!("..._gs_{group}_b_{bits}")`
    /// would produce `embed_gather_4bit_bfloat16_gs_96_b_3` happily; nothing
    /// built it; `vkCreateComputePipelines` is handed a null module and
    /// SIGSEGVs, with the validation layer reporting nothing. So the refusal
    /// carries the extent and the caller falls back.
    #[test]
    fn an_affine_point_the_tree_does_not_carry_is_refused() {
        assert!(
            matches!(
                affine_point(96, 4),
                Err(Refusal::Narrow {
                    what: "affine group size",
                    at: 96
                })
            ),
            "96 is a plausible group size and this tree has no module for it"
        );
        assert!(
            matches!(
                affine_point(64, 3),
                Err(Refusal::Narrow {
                    what: "affine bit width",
                    at: 3
                })
            ),
            "the bit width is checked separately, so the refusal says which"
        );
        assert!(
            affine_point(0, 0).is_err(),
            "and nothing about zero is a special case worth carrying"
        );
    }

    /// The single-row gather and the M>1 gather ask for DIFFERENT grids.
    ///
    /// Not a taller version of the same one. `embed_gather.slang` is
    /// `[numthreads(256, 1, 1)]` with `m = 0` when `PIE_MB` is unset and
    /// `[numthreads(16, 16, 1)]` with `m = gid.y` when it is set, so flattening
    /// the M>1 grid to `hidden * rows` lanes on x would visit every lane with
    /// `k >= hidden` for all but the first row -- each returning early -- and
    /// write exactly one row of the output, successfully.
    #[test]
    fn the_single_row_gather_and_the_multi_row_gather_ask_for_different_grids() {
        let seen = Seen::default();
        embed_gather_4bit(
            &seen,
            Buf(0),
            Buf(1),
            Buf(2),
            I32s(3),
            BufMut(4),
            2048,
            Env(64),
            Env(4),
        )
        .expect("a carried point over a real width");
        embed_gather_mb_4bit(
            &seen,
            Buf(0),
            Buf(1),
            Buf(2),
            I32s(3),
            BufMut(4),
            2048,
            Env(7),
            Env(64),
            Env(4),
        )
        .expect("seven rows is a launch");

        let calls = seen.0.borrow();
        let fired: Vec<(&str, [u32; 3])> = calls
            .iter()
            .map(|(e, lanes, _)| (e.as_str(), *lanes))
            .collect();
        assert_eq!(
            fired,
            vec![
                ("embed_gather_4bit_bfloat16_gs_64_b_4", [2048, 1, 1]),
                ("embed_gather_mb_4bit_bfloat16_gs_64_b_4", [2048, 7, 1]),
            ],
            "the rows are an AXIS in the M>1 form and absent in the other"
        );
    }

    /// The scaled gathers bind the scale, and the unscaled ones do not.
    ///
    /// `PIE_SCALED` changes the push block's LAYOUT -- `{ int hidden; }`
    /// against `{ int hidden; float embed_scale; }` -- so this is not a value
    /// that could default to `1.0`. An extra word pushed at an unscaled module
    /// is a validation error at best; a missing one is `embed_scale` read from
    /// whatever the push range last held, and gemma's embeddings come out
    /// scaled by a number from the previous dispatch.
    #[test]
    fn only_the_scaled_gathers_carry_the_scale() {
        let seen = Seen::default();
        embed_gather_4bit(
            &seen,
            Buf(0),
            Buf(1),
            Buf(2),
            I32s(3),
            BufMut(4),
            2048,
            Env(64),
            Env(4),
        )
        .expect("a launch");
        embed_gather_scaled_4bit(
            &seen,
            Buf(0),
            Buf(1),
            Buf(2),
            I32s(3),
            BufMut(4),
            2048,
            45.25,
            Env(64),
            Env(4),
        )
        .expect("a launch");

        let calls = seen.0.borrow();
        assert_eq!(
            calls[0].2.len(),
            6,
            "w, scales, biases, id, out, hidden -- and nothing else"
        );
        assert_eq!(
            calls[1].2.last(),
            Some(&ArgValue::F32(45.25)),
            "the scale is the LAST argument, after `hidden`, because that is \
             the order the push block declares its two fields in"
        );
    }

    /// `ple_combine` is flat and `row_gather` is not.
    ///
    /// Two rules, one family: `ple_combine.slang` is `[numthreads(256, 1, 1)]`
    /// over the whole `[n_layers, ple_dim]` block, and `row_gather.slang` is
    /// `[numthreads(16, 16, 1)]` with the requests on y. Under `kernel!` the
    /// difference was `LaunchRule::Elementwise` against `ElementwiseRows` and
    /// the driver read it off the row; here each body says which it is.
    #[test]
    fn the_two_join_kernels_ask_for_the_grids_their_shaders_are_written_for() {
        let seen = Seen::default();
        ple_combine(&seen, Buf(0), Buf(1), BufMut(2), Buf(3), Env(256), Env(26))
            .expect("twenty-six layers of PLE is a launch");
        row_gather(
            &seen,
            Buf(0),
            BufMut(1),
            U32s(2),
            Buf(3),
            InPacked(4),
            Env(2048),
            Env(3),
        )
        .expect("three requests is a launch");

        let calls = seen.0.borrow();
        let fired: Vec<(&str, [u32; 3])> = calls
            .iter()
            .map(|(e, lanes, _)| (e.as_str(), *lanes))
            .collect();
        assert_eq!(
            fired,
            vec![
                ("ple_combine_bfloat16", [256 * 26, 1, 1]),
                ("row_gather_bfloat16", [2048, 3, 1]),
            ]
        );
        assert_eq!(
            calls[1].2,
            vec![
                ArgValue::Buffer {
                    handle: 0,
                    writes: false
                },
                ArgValue::Buffer {
                    handle: 1,
                    writes: true
                },
                ArgValue::Buffer {
                    handle: 2,
                    writes: false
                },
                ArgValue::Buffer {
                    handle: 3,
                    writes: false
                },
                ArgValue::U32(4),
            ],
            "`count` is `InPacked`, which reaches the driver as a scalar it \
             writes into the params STRUCT, not as a fifth descriptor"
        );
    }
}
