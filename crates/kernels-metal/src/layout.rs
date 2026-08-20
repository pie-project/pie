//! Gathers and embeddings -- the kernels that move rows rather than
//! compute over them.
//!
//! A routine's arguments are its kernel's bindings, and a quantised gather
//! binds five buffers, two scalars and two axis facts. Nine is what that
//! kernel takes; collecting them into a struct would restate the binding order
//! somewhere else, which is the thing this refactor removes.

use kernels::Grid;
use kernels_macros::routine;
use kernels::routine::Refusal;

use crate::routine::{Asks, Bind, Const, Ctx, Fire, In, Out, Tensor, bf16, elementwise, elementwise_rows, keys};

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
#[routine]
pub fn embed_gather_4bit(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    group: Const<i32>,
    bits: Const<i32>) -> Result<(), Refusal> {
    let id = ctx.ask::<Tensor<i32>, keys::TokenIds>()?;
    let hidden = out.width;
    ctx.fire(
        Fire::at(EMBED_GATHER_FILE, EMBED_GATHER[affine_point(*group, *bits)?]).apply(Grid::of(elementwise(hidden, 1)?, [GROUP_X, 1, 1])),
        &[w.arg(), scales.arg(), biases.arg(), id.arg(), out.arg(), hidden.arg()],
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
#[routine]
pub fn embed_gather_mb_4bit(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    group: Const<i32>,
    bits: Const<i32>) -> Result<(), Refusal> {
    let id = ctx.ask::<Tensor<i32>, keys::TokenIds>()?;
    let hidden = out.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(EMBED_GATHER_FILE, EMBED_GATHER_MB[affine_point(*group, *bits)?]).apply(Grid::of(elementwise_rows(hidden, rows)?, [GROUP_X, 1, 1])),
        &[w.arg(), scales.arg(), biases.arg(), id.arg(), out.arg(), hidden.arg()],
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
#[routine]
pub fn embed_gather_scaled_4bit(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    embed_scale: Const<f32>,
    group: Const<i32>,
    bits: Const<i32>) -> Result<(), Refusal> {
    let id = ctx.ask::<Tensor<i32>, keys::TokenIds>()?;
    let hidden = out.width;
    ctx.fire(
        Fire::at(EMBED_GATHER_FILE, EMBED_GATHER_SCALED[affine_point(*group, *bits)?]).apply(Grid::of(elementwise(hidden, 1)?, [GROUP_X, 1, 1])),
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
    bits: Const<i32>) -> Result<(), Refusal> {
    let id = ctx.ask::<Tensor<i32>, keys::TokenIds>()?;
    let hidden = out.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(EMBED_GATHER_FILE, EMBED_GATHER_SCALED_MB[affine_point(*group, *bits)?]).apply(Grid::of(elementwise_rows(hidden, rows)?, [GROUP_X, 1, 1])),
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
/// `layout/ple_combine.metal` takes it as buffer 3, a `const constant float&`
/// of its own. It used to ride `PleCombineParams { inv_sqrt2, unused }` at that
/// same buffer -- MLX's layout, the one `kernels-vulkan` and `kernels-wgpu`
/// then copied -- whose second word was dead: it was one ROW's element count
/// bounding a dispatch of `width * rows`, so every row after the first returned
/// immediately and kept whatever the arena held. The bound went and the field
/// stayed, to keep the struct's size. Word 0 of the statement's run is the same
/// number either way, and the routine binds one `setBytes` where it forwarded a
/// staged block.
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
    inv_sqrt2: Const<f32>) -> Result<(), Refusal> {
    let width = proj.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("layout/ple_combine.metal", "ple_combine_bfloat16").apply(Grid::of(elementwise(width, rows)?, [GROUP_X, 1, 1])),
        &[proj.arg(), token.arg(), out.arg(), inv_sqrt2.arg()],
    )
}

/// The readout's gather: one distribution per REQUEST out of one row per
/// TOKEN.
///
/// A prefill's stream is one row per token and its readout is one distribution
/// per request, so the sampled rows are picked out before the lm head runs.
///
/// # Two scalars, two buffers, and the order is the body's
///
/// `count` used to be [`crate::routine::InPacked`], and on Metal that meant
/// "APPEND to the scalar run": `RowGatherParams` was width then count packed
/// into buffer 3, there was no buffer 4, and a packed slot's run already covers
/// every scalar after it. The statement stated `[width]`; the driver appended
/// the count, giving `[width, count]` -- exactly the struct. Vulkan read the
/// same `Ty::InPacked` differently because there the struct was a std430 buffer
/// and the scalars went to a push block, so there was nothing to append to. The
/// TYPE was the only statement of it; under `kernel!` the fact was said twice,
/// in the row and in the shader's struct, and only one was ever checked.
///
/// There is no struct now, and no `layout/row_gather_params.h` for a host and a
/// shader to read separately. `width` is the statement's word 0 read through a
/// `Const<u32>` mark, and the request count -- which is the FIRE's and not the
/// statement's, so it is asked for rather than marked -- is an ordinary scalar
/// this body passes straight after it. `driver-metal`'s `lay_out` gives every
/// scalar its own argument slot at the position the body put it, so the two
/// land as buffers 3 and 4 and `layout/row_gather.metal` declares them there.
/// The order is what has to be right -- a swap binds the count as the pitch --
/// and it is stated here, once, rather than in a shared header.
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
    width: Const<u32>) -> Result<(), Refusal> {
    let rows = ctx.ask::<Tensor<u32>, keys::SamplingIndices>()?;
    // A `u32` and no longer an `InPacked`: there is no block for it to be
    // appended to, so it is a scalar like any other and takes the buffer index
    // the body's argument order gives it.
    let count = ctx.ask::<u32, keys::RequestCount>()?;
    let row_count = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("layout/row_gather.metal", "row_gather_bfloat16").apply(Grid::of(elementwise_rows(input.width, row_count)?, [GROUP_X, 1, 1])),
        &[input.arg(), out.arg(), rows.arg(), width.arg(), count.arg()],
    )
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{ArgValue, Const, Encode, Tensor};
    use core::cell::{Cell, RefCell};

    /// One recorded dispatch: the fire, and the argument list.
    type Call = (Fire, Vec<ArgValue>);

    /// An `Encode` that remembers what it was asked to do, and answers the
    /// facts this family's bodies ask for.
    ///
    /// `rows` backs every `ctx.ask::<i32, keys::Rows>()` in this file --
    /// `embed_gather_mb_4bit`/`embed_gather_scaled_mb_4bit`'s row count and
    /// `row_gather`'s request count are both this fact under different names,
    /// so a test that fires more than one sets it again between calls.
    /// `token_ids` and `sampling_indices` are the two buffers every gather
    /// reads its indices through; `request_count` is `row_gather`'s request
    /// count, a plain `u32` now that there is no struct for it to be appended
    /// to as a packed field.
    ///
    /// NO `params_handle` ANY MORE. It answered `ctx.params()`, and
    /// `ple_combine` and `row_gather` were the two bodies in this file that
    /// asked for it; both state their scalars as `Const` marks now. A mark is
    /// passed by the CALLER rather than resolved, so this probe is never asked
    /// for the params run at all.
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
        // A PROBE HAS NO FIRE BEHIND IT, so it answers only the facts this
        // file's bodies ask for and refuses everything else honestly --
        // answering zero for an unasked fact would let a body under test pass
        // while the fact it asked for went unanswered on a real driver.
        fn resolve(&self, _ty: kernels::Ty, source: kernels::Source) -> Result<ArgValue, Refusal> {
            use kernels::keys::Fact;
            if source == <keys::Rows as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.rows.get()));
            }
            if source == <keys::TokenIds as Fact>::SOURCE {
                return Ok(ArgValue::Buffer(self.token_ids.get()));
            }
            if source == <keys::SamplingIndices as Fact>::SOURCE {
                return Ok(ArgValue::Buffer(self.sampling_indices.get()));
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
                    self.words.borrow().get(usize::from(n)).copied().unwrap_or(4096),
                ));
            }
            Err(Refusal::Unstated { what: "a fact this probe does not answer" })
        }

        fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
            self.calls.borrow_mut().push((fire, args.to_vec()));
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
            Const::new(Tensor::<u32>::new(1)),
            Const::new(Tensor::<bf16>::new(2)),
            Const::new(Tensor::<bf16>::new(3)),
            Out { ptr: Tensor::<bf16>::new(5), rows: 0, width: 2048 },
            Const::new(64),
            Const::new(4))
        .expect("a 64/4 checkpoint is one of the six");

        assert_eq!(
            seen.calls.borrow()[0].0.entrypoint,
            "embed_gather_4bit_bfloat16_gs_64_b_4"
        );

        let narrow = embed_gather_4bit(
            &seen,
            Const::new(Tensor::<u32>::new(1)),
            Const::new(Tensor::<bf16>::new(2)),
            Const::new(Tensor::<bf16>::new(3)),
            Out { ptr: Tensor::<bf16>::new(5), rows: 0, width: 2048 },
            Const::new(96),
            Const::new(4))
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
            seen.calls.borrow().len(),
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
            Const::new(Tensor::<u32>::new(1)),
            Const::new(Tensor::<bf16>::new(2)),
            Const::new(Tensor::<bf16>::new(3)),
            Out { ptr: Tensor::<bf16>::new(5), rows: 0, width: 2048 },
            Const::new(32),
            Const::new(4))
        .expect("a launch");
        seen.rows.set(7);
        embed_gather_mb_4bit(
            &seen,
            Const::new(Tensor::<u32>::new(1)),
            Const::new(Tensor::<bf16>::new(2)),
            Const::new(Tensor::<bf16>::new(3)),
            Out { ptr: Tensor::<bf16>::new(5), rows: 0, width: 2048 },
            Const::new(32),
            Const::new(4))
        .expect("a launch");

        let calls = seen.calls.borrow();
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
            Const::new(Tensor::<u32>::new(1)),
            Const::new(Tensor::<bf16>::new(2)),
            Const::new(Tensor::<bf16>::new(3)),
            Out { ptr: Tensor::<bf16>::new(5), rows: 0, width: 2048 },
            Const::new(45.254_834),
            Const::new(128),
            Const::new(8))
        .expect("a launch");

        let calls = seen.calls.borrow();
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

    /// `row_gather` passes TWO words, and the width comes first.
    ///
    /// On Metal `Ty::InPacked` meant "append to the scalar run": the struct was
    /// `{width, count}` packed into buffer 3, there was no buffer 4, and the
    /// count was the run's second word rather than a slot of its own. The
    /// statement stated the width; the body stated that something appended a
    /// count, once, in the type.
    ///
    /// Both words are ordinary scalars now -- `width` a `Const<u32>` mark
    /// reading the same word 0 of the same statement run by index, `count` the
    /// fire's own answer, asked for and passed straight after it -- so
    /// `driver-metal`'s `lay_out` gives each its own argument slot and
    /// `layout/row_gather.metal` declares them at buffers 3 and 4. The ORDER is
    /// what this pins: a swap binds the request count as the row pitch and the
    /// pitch as the request count, neither is a type error, both are plausible
    /// `uint`s, and the gather would read whole rows out of the wrong place and
    /// report success. The two numbers below differ on purpose so that a swap
    /// cannot pass.
    #[test]
    fn the_row_gathers_two_words_are_the_width_then_the_count() {
        let seen = Seen::default();
        seen.rows.set(3);
        seen.request_count.set(3);
        row_gather(
            &seen,
            In { ptr: Tensor::<bf16>::new(1), rows: 0, width: 2048 },
            Out::new(Tensor::<bf16>::new(2)),
            Const::new(2048))
        .expect("three requests is a launch");

        let calls = seen.calls.borrow();
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
            "input, out, rows, and then the two scalars that were one packed \
             struct"
        );
        assert_eq!(
            args[3],
            ArgValue::U32(2048),
            "the width the statement carries, which the mark reads at slot 0 \
             and the shader takes at buffer 3"
        );
        assert_eq!(
            args[4],
            ArgValue::U32(3),
            "and the request count the fire answers, at buffer 4 -- where the \
             struct had no buffer at all, because a packed slot's run covered \
             every scalar after it"
        );
    }
}
