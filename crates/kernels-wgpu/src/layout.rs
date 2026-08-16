//! Gathers and embeddings -- the kernels that move rows rather than
//! compute over them.

// A routine's arguments are the KERNEL's arguments, and a kernel takes what it
// takes: `embed_gather_scaled_mb_4bit` binds five buffers, two scalars and
// three environment facts because that is the dispatch. Collapsing them into a
// struct to satisfy a count would undo the derivation -- the table row IS the
// signature -- and `kernels-cuda-new` carries the same allow for the same
// reason, at a ceiling of 24.
#![allow(clippy::too_many_arguments)]

use kernels::KernelSig;

/// EMPTY: this family's rows have been RETIRED.
///
/// `refactor-bigplan.md` §7 Stage 3. Six kernels over 26 entrypoints — the
/// four quantized embedding gathers carry a two-axis product (`gs` x `b`) and
/// are the reason this backend's affine point had to be readable from the
/// SYMBOL rather than from a launch fact.
pub static KERNELS: &[KernelSig] = &[];

/// The entrypoints this family's routines spell, now that its rows are gone.
///
/// See [`crate::sample::ENTRYPOINTS`].
pub static ENTRYPOINTS: &[&str] = &[
    "embed_gather_4bit_bfloat16_gs_128_b_4",
    "embed_gather_4bit_bfloat16_gs_128_b_8",
    "embed_gather_4bit_bfloat16_gs_32_b_4",
    "embed_gather_4bit_bfloat16_gs_32_b_8",
    "embed_gather_4bit_bfloat16_gs_64_b_4",
    "embed_gather_4bit_bfloat16_gs_64_b_8",
    "embed_gather_mb_4bit_bfloat16_gs_128_b_4",
    "embed_gather_mb_4bit_bfloat16_gs_128_b_8",
    "embed_gather_mb_4bit_bfloat16_gs_32_b_4",
    "embed_gather_mb_4bit_bfloat16_gs_32_b_8",
    "embed_gather_mb_4bit_bfloat16_gs_64_b_4",
    "embed_gather_mb_4bit_bfloat16_gs_64_b_8",
    "embed_gather_scaled_4bit_bfloat16_gs_128_b_4",
    "embed_gather_scaled_4bit_bfloat16_gs_128_b_8",
    "embed_gather_scaled_4bit_bfloat16_gs_32_b_4",
    "embed_gather_scaled_4bit_bfloat16_gs_32_b_8",
    "embed_gather_scaled_4bit_bfloat16_gs_64_b_4",
    "embed_gather_scaled_4bit_bfloat16_gs_64_b_8",
    "embed_gather_scaled_mb_4bit_bfloat16_gs_128_b_4",
    "embed_gather_scaled_mb_4bit_bfloat16_gs_128_b_8",
    "embed_gather_scaled_mb_4bit_bfloat16_gs_32_b_4",
    "embed_gather_scaled_mb_4bit_bfloat16_gs_32_b_8",
    "embed_gather_scaled_mb_4bit_bfloat16_gs_64_b_4",
    "embed_gather_scaled_mb_4bit_bfloat16_gs_64_b_8",
    "ple_combine_bfloat16",
    "row_gather_bfloat16",
];

// ── The routine shape ────────────────────────────────────────────────
//
// `ple_combine` is the first row of this backend ported to `.wiki/kernel-x`'s
// shape: an ordinary `fn` whose body states the entrypoint and the lanes, and
// whose table row is derived from its signature. It sits BESIDE its `kernel!`
// row rather than replacing it, for exactly as long as it takes to prove the
// two agree -- see `driver-wgpu`'s
// `the_first_ported_routine_asks_for_the_grid_its_row_asked_for`. Two
// statements of one fact is what this refactor is against, so the row goes
// when the family does, and not one commit later.

use crate::routine::{Bind, Buf, BufMut, Ctx, Env, Fire, I32s, InPacked, Routine, U32s};
use kernels::routine::Refusal;
use kernels::shader::{elementwise, elementwise_rows};

/// gemma's PLE join: `(proj + token) * inv_sqrt2`, over the whole
/// `[n_layers, ple_dim]` block at once.
///
/// The scale is `1/sqrt(2)` and it is the JOIN's, not a deployment's -- two
/// streams averaged in the root-mean-square sense -- so it rides the packed
/// `PleCombineParams` buffer rather than an argument.
///
/// # The two arguments that were not operands
///
/// `width` and `rows` are `Env`: the statement does not carry them and the
/// environment always does. In the table shape they were not arguments at all
/// -- `LaunchRule::Elementwise` told `driver-wgpu` to read them off the
/// rectangle, which is why `Dims` exists. A body that needs them says so.
///
/// # Errors
///
/// [`Refusal::Empty`] when the block is empty. Stated rather than dispatched
/// as a zero grid: `dispatch_workgroups(0, 1, 1)` is legal WebGPU that runs
/// nothing and reports success.
pub fn ple_combine(
    ctx: &Ctx,
    proj: Buf,
    token: Buf,
    out: BufMut,
    params: Buf,
    width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    if *width <= 0 {
        return Err(Refusal::Empty { what: "width" });
    }
    if *rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    // `LaunchRule::Elementwise`, whole: one lane per element of the
    // rectangle, on one axis. The division into workgroups is the driver's --
    // `@workgroup_size` is in the WGSL and this crate does not reflect it.
    let lanes = width.unsigned_abs() * rows.unsigned_abs();
    ctx.dispatch(
        Fire {
            module: "layout/ple_combine.wgsl",
            entrypoint: "ple_combine_bfloat16",
            lanes: [lanes, 1, 1],
        },
        &[proj.v(), token.v(), out.v(), params.v()],
    )
}

/// Which of the six affine points `(group, bits)` names, or a refusal.
///
/// The order is the one every table below is written in: group ascending,
/// and 4-bit before 8-bit within each. Six points, and they are the six the
/// shader tree carries -- `embed_gather.wgsl` declares
/// `pie:instantiate embed_gather_4bit_bfloat16_gs_32_b_4` and five siblings,
/// and nothing else exists to name.
///
/// This is §4 of `.wiki/kernel-x/wgpu-refactor.md` made concrete: the traced
/// symbol used to carry `_gs_64_b_4` and `KernelSig::covers_point` stripped it
/// to find the row. A body picks the spelling instead, from facts the driver
/// holds -- `AffineFormat` is a checkpoint fact -- and the tables below are
/// literals rather than a paste, so a point the tree does not carry cannot be
/// spelled at all.
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

/// The gather that reads one row of an affine-quantised embedding table.
///
/// `hidden` rides the `@group(1)` uniform block -- the shader declares
/// `struct Params { hidden: i32 }` -- which is the driver's to place: it reads
/// the split off the argument VARIANTS and needs no second statement of it.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty rectangle, [`Refusal::Narrow`] for an
/// affine point the shader tree does not carry.
pub fn embed_gather_4bit(
    ctx: &Ctx,
    w: Buf,
    scales: Buf,
    biases: Buf,
    id: I32s,
    out: BufMut,
    hidden: i32,
    group: Env<i32>,
    bits: Env<i32>,
) -> Result<(), Refusal> {
    let rows = 1;
    let lanes = elementwise(hidden, rows)?;
    ctx.dispatch(
        Fire {
            module: "layout/embed_gather.wgsl",
            entrypoint: EMBED_GATHER[affine_point(*group, *bits)?],
            lanes,
        },
        &[w.v(), scales.v(), biases.v(), id.v(), out.v(), hidden.v()],
    )
}

/// The M>1 form, and the one a text should name.
///
/// # Errors
///
/// As [`embed_gather_4bit`].
pub fn embed_gather_mb_4bit(
    ctx: &Ctx,
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
    let lanes = elementwise_rows(hidden, *rows)?;
    ctx.dispatch(
        Fire {
            module: "layout/embed_gather.wgsl",
            entrypoint: EMBED_GATHER_MB[affine_point(*group, *bits)?],
            lanes,
        },
        &[w.v(), scales.v(), biases.v(), id.v(), out.v(), hidden.v()],
    )
}

/// [`embed_gather_4bit`] with the embedding scale folded in -- gemma
/// multiplies its embeddings by `sqrt(hidden)`, which the statement carries.
///
/// # Errors
///
/// As [`embed_gather_4bit`].
pub fn embed_gather_scaled_4bit(
    ctx: &Ctx,
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
    let lanes = elementwise(hidden, 1)?;
    ctx.dispatch(
        Fire {
            module: "layout/embed_gather.wgsl",
            entrypoint: EMBED_GATHER_SCALED[affine_point(*group, *bits)?],
            lanes,
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

/// The scaled gather over M>1 rows.
///
/// # Errors
///
/// As [`embed_gather_4bit`].
pub fn embed_gather_scaled_mb_4bit(
    ctx: &Ctx,
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
    let lanes = elementwise_rows(hidden, *rows)?;
    ctx.dispatch(
        Fire {
            module: "layout/embed_gather.wgsl",
            entrypoint: EMBED_GATHER_SCALED_MB[affine_point(*group, *bits)?],
            lanes,
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

/// The readout's gather: one distribution per REQUEST out of one row per
/// TOKEN.
///
/// `count` is [`InPacked`]: it is the second FIELD of the `RowGatherParams`
/// struct that buffer 3 binds, not a uniform-block scalar and not a buffer of
/// its own. The type is what says so; under `kernel!` the same fact was an
/// operand's `Ty::InPacked` and the shader's struct, and only one of the two
/// was checked.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty rectangle.
pub fn row_gather(
    ctx: &Ctx,
    input: Buf,
    out: BufMut,
    rows: U32s,
    params: Buf,
    count: InPacked,
    width: Env<i32>,
    row_count: Env<i32>,
) -> Result<(), Refusal> {
    let lanes = elementwise_rows(*width, *row_count)?;
    ctx.dispatch(
        Fire {
            module: "layout/row_gather.wgsl",
            entrypoint: "row_gather_bfloat16",
            lanes,
        },
        &[input.v(), out.v(), rows.v(), params.v(), count.v()],
    )
}

/// This family's routines.
pub static ROUTINES: &[Routine] = &[
    crate::routine!(ple_combine),
    crate::routine!(embed_gather_4bit),
    crate::routine!(embed_gather_mb_4bit),
    crate::routine!(embed_gather_scaled_4bit),
    crate::routine!(embed_gather_scaled_mb_4bit),
    crate::routine!(row_gather),
];

#[cfg(test)]
mod ported {
    use super::*;
    use crate::routine::ArgValue;
    use core::cell::RefCell;
    use kernels::Ty;
    use kernels::routine::Provenance;

    /// One dispatch, as the recorder kept it.
    type Kept = (String, String, [u32; 3], Vec<ArgValue>);

    /// An `Encode` that records instead of dispatching.
    #[derive(Default)]
    struct Recorder {
        seen: RefCell<Vec<Kept>>,
    }

    impl crate::routine::Encode for Recorder {
        fn dispatch(&self, fire: Fire<'_>, args: &[ArgValue]) -> Result<(), Refusal> {
            self.seen.borrow_mut().push((
                fire.module.to_owned(),
                fire.entrypoint.to_owned(),
                fire.lanes,
                args.to_vec(),
            ));
            Ok(())
        }
    }

    /// The row this routine replaces is derived from its signature.
    ///
    /// Six arguments where the `kernel!` row states four operands, and the
    /// difference is the whole point: `width` and `rows` were never operands.
    /// `LaunchRule::Elementwise` told the driver to read them off the
    /// rectangle, so they were a fact about the launch that the row pointed at
    /// and did not carry. The signature carries them, and their `Env`
    /// provenance says who supplies them.
    #[test]
    fn the_routines_row_is_its_signature_and_names_the_two_that_were_not_operands() {
        let row = ROUTINES
            .iter()
            .find(|r| r.name == "ple_combine")
            .expect("the family declares it");

        assert_eq!(row.args.len(), 6, "four operands and two environment facts");
        assert_eq!(
            row.args.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
            [Ty::Buf, Ty::Buf, Ty::BufMut, Ty::Buf, Ty::I32, Ty::I32],
            "and the first four are the row's operands, in the row's order"
        );
        assert_eq!(
            row.args
                .iter()
                .filter(|(_, p)| *p == Provenance::Env)
                .count(),
            2,
            "the two the statement does not supply"
        );
    }

    /// The body asks for the grid `LaunchRule::Elementwise` asked for.
    ///
    /// `Elementwise` is `[width * rows, 1, 1]` lanes -- `geometry.rs:551` --
    /// and this is the same number reached by code instead of by a rule. The
    /// driver-side half of this check, which compares against `geometry::groups`
    /// itself rather than against a transcription of it, is
    /// `the_first_ported_routine_asks_for_the_grid_its_row_asked_for`.
    #[test]
    fn the_body_asks_for_the_elementwise_grid() {
        let to = Recorder::default();
        ple_combine(&to, Buf(0), Buf(1), BufMut(2), Buf(3), Env(64), Env(7))
            .expect("it dispatches");

        let seen = to.seen.borrow();
        let (module, entrypoint, lanes, args) = seen.first().expect("one dispatch");
        assert_eq!(module, "layout/ple_combine.wgsl");
        assert_eq!(
            entrypoint, "ple_combine_bfloat16",
            "the axis point, pasted by the body"
        );
        assert_eq!(*lanes, [64 * 7, 1, 1], "width * rows on one axis");
        assert_eq!(
            args,
            &[
                ArgValue::Buffer(0),
                ArgValue::Buffer(1),
                ArgValue::Buffer(2),
                ArgValue::Buffer(3)
            ],
            "the four operands, and NOT the two environment facts -- those \
             sized the grid and the kernel never reads them"
        );
    }

    /// Every entrypoint a body in this family can name, exists.
    ///
    /// This is the check that de-risks §4 of
    /// `.wiki/kernel-x/wgpu-refactor.md`. The traced symbol used to carry the
    /// axis suffix and `KernelSig::covers_point` stripped it, so the NAME was
    /// the thing that had to be right and the table was what made it so. A
    /// body picks the spelling instead, and a body that picks a spelling the
    /// tree does not carry fails at first fire on a user's machine.
    ///
    /// So the whole reachable set is resolved here, with no adapter: the four
    /// six-point tables and `row_gather`'s single name, against
    /// `source::entrypoint_source`, which is what a driver will ask.
    #[test]
    fn every_spelling_a_body_here_can_choose_is_one_the_tree_carries() {
        let mut checked = 0usize;
        for table in [
            &EMBED_GATHER,
            &EMBED_GATHER_MB,
            &EMBED_GATHER_SCALED,
            &EMBED_GATHER_SCALED_MB,
        ] {
            for name in table.iter() {
                crate::source::entrypoint_source(name, crate::Capability::Baseline).unwrap_or_else(
                    |e| panic!("a body can name `{name}` and the tree has not got it: {e}"),
                );
                checked += 1;
            }
        }
        for name in ["row_gather_bfloat16", "ple_combine_bfloat16"] {
            crate::source::entrypoint_source(name, crate::Capability::Baseline).unwrap_or_else(
                |e| panic!("a body can name `{name}` and the tree has not got it: {e}"),
            );
            checked += 1;
        }
        assert_eq!(checked, 26, "four six-point tables and two single names");
    }

    /// An affine point the tree does not carry is refused, by name.
    ///
    /// The other half of the same decision: the tables are literals, so a
    /// point outside them cannot be SPELLED -- and the refusal says which
    /// coordinate was wrong rather than failing to find a module later.
    #[test]
    fn an_affine_point_the_tree_does_not_carry_is_refused_before_it_is_spelled() {
        assert_eq!(
            affine_point(96, 4),
            Err(Refusal::Narrow {
                what: "affine group size",
                at: 96
            })
        );
        assert_eq!(
            affine_point(64, 3),
            Err(Refusal::Narrow {
                what: "affine bit width",
                at: 3
            })
        );
        // And the six that do exist map onto the six names, in order.
        for (i, (g, b)) in [(32, 4), (32, 8), (64, 4), (64, 8), (128, 4), (128, 8)]
            .into_iter()
            .enumerate()
        {
            assert_eq!(affine_point(g, b), Ok(i));
            assert!(
                EMBED_GATHER[i].ends_with(&format!("_gs_{g}_b_{b}")),
                "the index and the table disagree at ({g}, {b}): {}",
                EMBED_GATHER[i]
            );
        }
    }

    /// This family declares six routines and each is named by its `fn`.
    #[test]
    fn the_family_declares_the_six_it_has_ported() {
        let names: Vec<&str> = ROUTINES.iter().map(|r| r.name).collect();
        assert_eq!(
            names,
            [
                "ple_combine",
                "embed_gather_4bit",
                "embed_gather_mb_4bit",
                "embed_gather_scaled_4bit",
                "embed_gather_scaled_mb_4bit",
                "row_gather",
            ]
        );
    }

    /// Every body in this family asks for the grid its row's rule names.
    ///
    /// The six split two ways and the split is load-bearing.
    /// `LaunchRule::ElementwiseRows` puts the rows on their own axis; flatten
    /// one of those to `Elementwise` and the dispatch visits `width * rows`
    /// lanes with the row index taken from `gid.y`, which is zero for all of
    /// them -- so it writes exactly ONE row and reports success. The reverse
    /// mistake is a grid `rows` times too small on x.
    ///
    /// Neither is visible from a body: both spell a plausible `lanes`, both
    /// dispatch, both return `Ok`. The rows are still here to be compared
    /// against, so they are.
    #[test]
    fn every_body_here_asks_for_the_grid_its_rows_rule_names() {
        const W: i32 = 64;
        const R: i32 = 7;

        let to = Recorder::default();
        let (b, m) = (Buf(0), BufMut(1));
        ple_combine(&to, b, b, m, b, Env(W), Env(R)).expect("dispatches");
        embed_gather_4bit(&to, b, b, b, I32s(2), m, W, Env(64), Env(4)).expect("dispatches");
        embed_gather_mb_4bit(&to, b, b, b, I32s(2), m, W, Env(R), Env(64), Env(4))
            .expect("dispatches");
        embed_gather_scaled_4bit(&to, b, b, b, I32s(2), m, W, 1.0, Env(64), Env(4))
            .expect("dispatches");
        embed_gather_scaled_mb_4bit(&to, b, b, b, I32s(2), m, W, 1.0, Env(R), Env(64), Env(4))
            .expect("dispatches");
        row_gather(&to, b, m, U32s(3), b, InPacked(1), Env(W), Env(R)).expect("dispatches");

        let seen = to.seen.borrow();
        let order = [
            "ple_combine",
            "embed_gather_4bit",
            "embed_gather_mb_4bit",
            "embed_gather_scaled_4bit",
            "embed_gather_scaled_mb_4bit",
            "row_gather",
        ];
        assert_eq!(seen.len(), order.len(), "one dispatch each");

        // `embed_gather_4bit` and its scaled twin are single-row by
        // construction -- the body passes rows = 1 -- so their `Elementwise`
        // grid is `[W * 1, 1, 1]`. The rest carry the fire's rows.
        for ((name, (_, _, lanes, _)), _) in order.iter().zip(seen.iter()).zip(0..) {
            // The RULE each body's grid must match. It was read off the row
            // until Stage 3 retired them; stated here because the claim is
            // about the BODY and a deleted row is not a reason to stop making
            // it. `driver-wgpu`'s
            // `the_routine_path_plans_what_the_table_path_planned` compared
            // all six against their rows before those rows went.
            let rule = match *name {
                // The single-row gathers pass `rows = 1` themselves, so their
                // grid is flat; the `_mb_` pair carries the fire's rows on y.
                "ple_combine" | "embed_gather_4bit" | "embed_gather_scaled_4bit" => {
                    kernels::LaunchRule::Elementwise
                }
                "embed_gather_mb_4bit" | "embed_gather_scaled_mb_4bit" | "row_gather" => {
                    kernels::LaunchRule::ElementwiseRows
                }
                other => panic!("`{other}` has no stated rule here"),
            };
            let rows = if name.starts_with("embed_gather") && !name.contains("_mb_") {
                1
            } else {
                R
            };
            let want = match rule {
                kernels::LaunchRule::Elementwise => [W.unsigned_abs() * rows.unsigned_abs(), 1, 1],
                kernels::LaunchRule::ElementwiseRows => [W.unsigned_abs(), rows.unsigned_abs(), 1],
                other => panic!("`{name}` states {other:?}, which this test does not model"),
            };
            assert_eq!(
                *lanes, want,
                "`{name}`'s body asks for {lanes:?} where {rule:?} wants {want:?}"
            );
        }
    }

    /// An empty rectangle is refused, not dispatched as a zero grid.
    #[test]
    fn an_empty_block_is_refused_rather_than_launched_as_nothing() {
        let to = Recorder::default();
        for (w, r, what) in [(0, 7, "width"), (64, 0, "rows")] {
            assert_eq!(
                ple_combine(&to, Buf(0), Buf(1), BufMut(2), Buf(3), Env(w), Env(r)),
                Err(Refusal::Empty { what }),
                "`dispatch_workgroups(0, 1, 1)` is legal WebGPU that runs \
                 nothing and reports success"
            );
        }
        assert!(to.seen.borrow().is_empty(), "and nothing was dispatched");
    }
}
