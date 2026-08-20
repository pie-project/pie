//! Gathers and embeddings -- the kernels that move rows rather than
//! compute over them.
//!
//! A routine's arguments are its kernel's bindings, and a quantised gather
//! binds five buffers, two scalars and two axis facts. Nine is what that
//! kernel takes; collecting them into a struct would restate the binding order
//! somewhere else, which is the thing this refactor removes.

use crate::routine::{Asks, Bind, Const, Ctx, Fire, In, Out, Tensor, bf16, keys};
use kernels::routine::Refusal;
use kernels::shader::{elementwise, elementwise_rows};
use kernels_macros::routine;

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
#[routine]
pub fn embed_gather_4bit(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    group: Const<i32>,
    bits: Const<i32>,
) -> Result<(), Refusal> {
    let id = ctx.ask::<Tensor<i32>, keys::TokenIds>()?;
    let hidden = out.width;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(EMBED_GATHER[affine_point(*group, *bits)?], ctx.best()),
            EMBED_GATHER[affine_point(*group, *bits)?],
        )
        .apply(elementwise(hidden, 1)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            id.arg(),
            out.arg(),
            hidden.arg(),
        ],
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
#[routine]
pub fn embed_gather_mb_4bit(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    group: Const<i32>,
    bits: Const<i32>,
) -> Result<(), Refusal> {
    let id = ctx.ask::<Tensor<i32>, keys::TokenIds>()?;
    let hidden = out.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(EMBED_GATHER_MB[affine_point(*group, *bits)?], ctx.best()),
            EMBED_GATHER_MB[affine_point(*group, *bits)?],
        )
        .apply(elementwise_rows(hidden, rows)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            id.arg(),
            out.arg(),
            hidden.arg(),
        ],
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
#[routine]
pub fn embed_gather_scaled_4bit(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    embed_scale: Const<f32>,
    group: Const<i32>,
    bits: Const<i32>,
) -> Result<(), Refusal> {
    let id = ctx.ask::<Tensor<i32>, keys::TokenIds>()?;
    let hidden = out.width;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(
                EMBED_GATHER_SCALED[affine_point(*group, *bits)?],
                ctx.best(),
            ),
            EMBED_GATHER_SCALED[affine_point(*group, *bits)?],
        )
        .apply(elementwise(hidden, 1)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            id.arg(),
            out.arg(),
            hidden.arg(),
            embed_scale.arg(),
        ],
    )
}

/// The M>1 form of [`embed_gather_scaled_4bit`], and the one a text should
/// name.
///
/// # Errors
///
/// As [`embed_gather_4bit`].
#[routine]
pub fn embed_gather_scaled_mb_4bit(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    embed_scale: Const<f32>,
    group: Const<i32>,
    bits: Const<i32>,
) -> Result<(), Refusal> {
    let id = ctx.ask::<Tensor<i32>, keys::TokenIds>()?;
    let hidden = out.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(
                EMBED_GATHER_SCALED_MB[affine_point(*group, *bits)?],
                ctx.best(),
            ),
            EMBED_GATHER_SCALED_MB[affine_point(*group, *bits)?],
        )
        .apply(elementwise_rows(hidden, rows)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            id.arg(),
            out.arg(),
            hidden.arg(),
            embed_scale.arg(),
        ],
    )
}

/// gemma's PLE join: `(proj + token) * inv_sqrt2`, over the whole
/// `[n_layers, ple_dim]` block at once.
///
/// The scale is `1/sqrt(2)` and it is the JOIN's, not a deployment's -- two
/// streams averaged in the root-mean-square sense -- so it is the only scalar
/// this kernel reads, and it is a MARK rather than a struct:
/// `layout/ple_combine.slang` takes it as the one field of its push block. It
/// used to ride a `PleCombineParams { inv_sqrt2, n }` storage buffer -- Metal's
/// layout, ported whole and parked in `norm/rms_params.slang` beside the RMS
/// family's -- whose second word was dead: `n` was one ROW's element count and
/// the elementwise launch covers the whole rectangle, so it never bounded the
/// dispatch it was named for and stayed only to hold the struct's size. That
/// cost this plane a descriptor, a staging slot and the sentinel
/// `Holds::params_block` mints for a buffer with no address, for four live
/// bytes. Word 0 of the statement's run is the same number either way.
///
/// `width` and `rows` are [`Env`]: the statement does not carry them and the
/// environment always does. Under `kernel!` they were not arguments at all --
/// `LaunchRule::Elementwise` told the driver to read them off the rectangle.
/// A body that needs them says so.
///
/// # Errors
///
/// [`Refusal::Empty`] when the block is empty.
#[routine]
pub fn ple_combine(
    ctx: &Ctx<'_>,
    proj: In<Tensor<bf16>>,
    token: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    // THE JOIN'S SCALE, which was `PleCombineParams`'s first field. The
    // struct's other word -- a per-row element count bounding a whole-tensor
    // grid -- was dead, so with the scale stated as a mark there is nothing
    // left for a block to carry.
    inv_sqrt2: Const<f32>,
) -> Result<(), Refusal> {
    let width = proj.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("ple_combine_bfloat16", ctx.best()),
            "ple_combine_bfloat16",
        )
        .apply(elementwise(width, rows)?),
        &[proj.arg(), token.arg(), out.arg(), inv_sqrt2.arg()],
    )
}

/// The readout's gather: one distribution per REQUEST out of one row per
/// TOKEN.
///
/// A prefill's stream is one row per token and its readout is one distribution
/// per request, so the sampled rows are picked out before the lm head runs.
///
/// # Two words, one push block, and the order is the body's
///
/// `count` used to be [`crate::routine::InPacked`], and on this backend that
/// word did real work: `row_gather.slang` bound `RowGatherParams` as a std430
/// buffer at 3 and sent plain scalars to a PUSH block, so there was no trailing
/// scalar slot to append a count to -- the driver wrote it into the struct's
/// second field and the shader read `p.count`. `kernels-metal` read the same
/// `Ty::InPacked` as "append to the scalar run", because there the packed slot
/// IS the buffer. The type was the only statement of it; under `kernel!` the
/// fact was said twice, in the row and in the shader's struct, and only one was
/// ever checked.
///
/// There is no struct now. `width` is the statement's word 0 read through a
/// `Const<u32>` mark, and the request count -- which is the FIRE's and not the
/// statement's, so it is asked for rather than marked -- is an ordinary scalar
/// this body passes straight after it. `driver-vulkan`'s `Encoder::fire` builds
/// the push range out of the body's scalars in the order the body passed them,
/// so `[width, count]` is eight bytes with the layout the struct had, field for
/// field, and `layout/row_gather.slang` reads `pc.width`/`pc.count` where it
/// read `p.width`/`p.count`. The old worry -- that a count folded into the push
/// block would push a word nothing read and leave the struct's field unwritten
/// -- was about a block that was otherwise EMPTY, and this one is not: both its
/// words are read.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty rectangle.
#[routine]
pub fn row_gather(
    ctx: &Ctx<'_>,
    input: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    // THE ROW PITCH, which was `RowGatherParams`'s first field and is the only
    // word of the two the STATEMENT carries. The count below is the fire's.
    width: Const<u32>,
) -> Result<(), Refusal> {
    let rows = ctx.ask::<Tensor<u32>, keys::SamplingIndices>()?;
    // A `u32` and no longer an `InPacked`: there is no struct for it to be a
    // field of, so it is a scalar like any other and takes its place in the
    // push block by the position the body gives it.
    let count = ctx.ask::<u32, keys::RequestCount>()?;
    let row_count = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("row_gather_bfloat16", ctx.best()),
            "row_gather_bfloat16",
        )
        .apply(elementwise_rows(input.width, row_count)?),
        &[input.arg(), out.arg(), rows.arg(), width.arg(), count.arg()],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{ArgValue, Const, Encode, Tensor};
    use core::cell::{Cell, RefCell};

    type Call = (String, [u32; 3], Vec<ArgValue>);

    /// An `Encode` that remembers, and answers the facts this family's bodies
    /// ask for.
    ///
    /// `rows` backs every `ctx.ask::<i32, keys::Rows>()` in this file --
    /// `ple_combine`'s layer count and `row_gather`'s request count are both
    /// this fact under different names, so a test that fires both in one
    /// function sets it again between calls. `token_ids` and
    /// `sampling_indices` are the two buffers every gather reads its indices
    /// through; `request_count` is `row_gather`'s request count, a plain `u32`
    /// now that there is no struct for it to be a packed field of.
    ///
    /// NO `params_handle` ANY MORE. It held `ctx.params()`'s own handle, split
    /// out from the generic `Ty::Buf` catch-all only because
    /// `the_two_join_kernels_ask_for_the_grids_their_shaders_are_written_for`
    /// asserts the exact bound list and needed it to read a specific number.
    /// `ple_combine` and `row_gather` were the two bodies that forwarded that
    /// block; both state their scalars as `Const` marks now, a mark is passed
    /// by the CALLER rather than resolved, and so this probe is never asked for
    /// the params run at all.
    struct Seen {
        calls: RefCell<Vec<Call>>,
        rows: Cell<i32>,
        token_ids: Cell<u32>,
        sampling_indices: Cell<u32>,
        request_count: Cell<u32>,
        /// THE STATEMENT\'S SCALAR RUN, for a body that reads a word by
        /// index. Empty means "4096 at every slot", which is a plausible
        /// stride for the rows these tests build; a case that means a
        /// particular tiling or split count sets its own.
        words: RefCell<Vec<i32>>,
    }

    impl Default for Seen {
        fn default() -> Self {
            Self {
                calls: RefCell::default(),
                rows: Cell::new(1),
                token_ids: Cell::new(800),
                sampling_indices: Cell::new(800),
                request_count: Cell::new(1),
                words: RefCell::default(),
            }
        }
    }

    impl Encode for Seen {
        fn resolve(&self, ty: kernels::Ty, source: kernels::Source) -> Result<ArgValue, Refusal> {
            use kernels::keys::Fact;
            if source == <keys::Rows as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.rows.get()));
            }
            if source == <keys::TokenIds as Fact>::SOURCE {
                return Ok(ArgValue::Buffer {
                    handle: self.token_ids.get(),
                    writes: false,
                    rows: 0,
                    width: 0,
                });
            }
            if source == <keys::SamplingIndices as Fact>::SOURCE {
                return Ok(ArgValue::Buffer {
                    handle: self.sampling_indices.get(),
                    writes: false,
                    rows: 0,
                    width: 0,
                });
            }
            if source == <keys::RequestCount as Fact>::SOURCE {
                return Ok(ArgValue::U32(self.request_count.get()));
            }
            // THE STATEMENT'S OWN SCALARS, which a body reads by index when its
            // params run is a struct and no `Const` mark can name a word inside
            // it -- see `Asks::param`. The probe answers a number that is
            // plausible for every reader: a stride wide enough for the rows
            // these tests build, and a positive tiling.
            if let kernels::Source::Slot(kernels::Kind::Param, n) = source {
                return Ok(ArgValue::I32(
                    self.words
                        .borrow()
                        .get(usize::from(n))
                        .copied()
                        .unwrap_or(4096),
                ));
            }
            if matches!(ty, kernels::Ty::Buf) {
                return Ok(ArgValue::Buffer {
                    handle: 900,
                    writes: false,
                    rows: 0,
                    width: 0,
                });
            }
            // Anything else is refused: a probe that invented an answer to a
            // fact it does not know would let a body pass under test while
            // the same fact went unanswered on a real driver.
            Err(Refusal::Unstated {
                what: "a fact this probe does not answer",
            })
        }

        fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
            self.calls
                .borrow_mut()
                .push((fire.entrypoint.to_owned(), fire.lanes, args.to_vec()));
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
            Const::new(Tensor::<u32>::new(0)),
            Const::new(Tensor::<bf16>::new(1)),
            Const::new(Tensor::<bf16>::new(2)),
            Out {
                ptr: Tensor::<bf16>::new(4),
                rows: 0,
                width: 2048,
            },
            Const::new(64),
            Const::new(4),
        )
        .expect("a carried point over a real width");
        // The M>1 form asks `keys::Rows` for the seven rows the single-row
        // form never reads at all.
        seen.rows.set(7);
        embed_gather_mb_4bit(
            &seen,
            Const::new(Tensor::<u32>::new(0)),
            Const::new(Tensor::<bf16>::new(1)),
            Const::new(Tensor::<bf16>::new(2)),
            Out {
                ptr: Tensor::<bf16>::new(4),
                rows: 0,
                width: 2048,
            },
            Const::new(64),
            Const::new(4),
        )
        .expect("seven rows is a launch");

        let calls = seen.calls.borrow();
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
            Const::new(Tensor::<u32>::new(0)),
            Const::new(Tensor::<bf16>::new(1)),
            Const::new(Tensor::<bf16>::new(2)),
            Out {
                ptr: Tensor::<bf16>::new(4),
                rows: 0,
                width: 2048,
            },
            Const::new(64),
            Const::new(4),
        )
        .expect("a launch");
        embed_gather_scaled_4bit(
            &seen,
            Const::new(Tensor::<u32>::new(0)),
            Const::new(Tensor::<bf16>::new(1)),
            Const::new(Tensor::<bf16>::new(2)),
            Out {
                ptr: Tensor::<bf16>::new(4),
                rows: 0,
                width: 2048,
            },
            Const::new(45.25),
            Const::new(64),
            Const::new(4),
        )
        .expect("a launch");

        let calls = seen.calls.borrow();
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
    ///
    /// The `writes: true` below is load-bearing, not decoration.
    /// `row_gather`'s `out` is an `Out<Tensor<bf16>>`, and the direction
    /// survives to the driver only because `kernels::routine`'s `Bind` impl
    /// for `Out<E>`/`InOut<E>` reaches for `BindMut` -- `Tensor<E>`'s
    /// `arg_mut`, which is `V::buffer_mut` -- where `In<E>` reaches for
    /// `Bind`. On this plane `Elem::Read` and `Elem::Write` are the SAME
    /// type, so the trait is the only thing carrying the fact:
    /// `driver-vulkan::encode::Encode::fire` takes the bit straight off the
    /// fired `ArgValue` and never re-derives it from the declared `Ty`, and
    /// `device::hazards` decides every barrier from what that bit says. A
    /// mark that bound read-only would cost no error and no crash, only a
    /// missing write-then-read barrier and a fluent wrong answer.
    ///
    /// The bound list below is also where `row_gather`'s two words are pinned.
    /// The fourth entry used to be `Buffer { handle: 3 }` -- the handle
    /// `ctx.params()` minted for the staged `RowGatherParams` block -- and the
    /// fifth was the `InPacked` count the driver wrote into that struct's
    /// second field. Both are plain scalars now, in the order
    /// `layout/row_gather.slang` declares its push block, and they are
    /// different numbers on purpose so a swap of the two cannot pass.
    #[test]
    fn the_two_join_kernels_ask_for_the_grids_their_shaders_are_written_for() {
        let seen = Seen::default();
        // `ple_combine`'s own ask: 26 is the layer count `rows` means here.
        seen.rows.set(26);
        ple_combine(
            &seen,
            In {
                ptr: Tensor::<bf16>::new(0),
                rows: 0,
                width: 256,
            },
            In::new(Tensor::<bf16>::new(1)),
            Out::new(Tensor::<bf16>::new(2)),
            Const::new(core::f32::consts::FRAC_1_SQRT_2),
        )
        .expect("twenty-six layers of PLE is a launch");
        // `row_gather` re-purposes `rows` as its own request count, and its
        // remaining three facts get the exact handles the assertion below
        // checks for, continuing the sequential numbering the bound list
        // reads as.
        seen.rows.set(3);
        seen.sampling_indices.set(2);
        seen.request_count.set(4);
        row_gather(
            &seen,
            In {
                ptr: Tensor::<bf16>::new(0),
                rows: 0,
                width: 2048,
            },
            Out::new(Tensor::<bf16>::new(1)),
            Const::new(2048),
        )
        .expect("three requests is a launch");

        let calls = seen.calls.borrow();
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
                    writes: false,
                    rows: 0,
                    width: 0
                },
                ArgValue::Buffer {
                    handle: 1,
                    writes: true,
                    rows: 0,
                    width: 0
                },
                ArgValue::Buffer {
                    handle: 2,
                    writes: false,
                    rows: 0,
                    width: 0
                },
                ArgValue::U32(2048),
                ArgValue::U32(4),
            ],
            "three descriptors and then the push block's two words, in the \
             order `layout/row_gather.slang` declares them: the width the \
             statement carries -- read at slot 0 by a `Const<u32>` mark where \
             it used to be word 0 of a staged `RowGatherParams` -- and then \
             the request count the fire answers, which was `InPacked` and \
             reached the driver as a scalar it wrote into that struct's \
             second field rather than as a value of its own"
        );
    }
}
