//! Gathers and embeddings -- the kernels that move rows rather than
//! compute over them.

// A routine's arguments are the KERNEL's arguments, and a kernel takes what it
// takes: `embed_gather_scaled_mb_4bit` binds five buffers, two scalars and
// three environment facts because that is the dispatch. Collapsing them into a
// struct to satisfy a count would undo the derivation -- the table row IS the
// signature -- and `kernels-cuda-new` carries the same allow for the same
// reason, at a ceiling of 24.

use kernels_macros::routine;

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

use crate::routine::{Asks, Bind, Const, Ctx, Fire, In, Out, Tensor, bf16, keys};
use kernels::routine::Refusal;
use kernels::shader::{elementwise, elementwise_rows};

/// gemma's PLE join: `(proj + token) * inv_sqrt2`, over the whole
/// `[n_layers, ple_dim]` block at once.
///
/// The scale is `1/sqrt(2)` and it is the JOIN's, not a deployment's -- two
/// streams averaged in the root-mean-square sense -- so it is the one scalar
/// this kernel reads, and it is a MARK rather than a struct:
/// `layout/ple_combine.wgsl` takes it as the one field of its `@group(1)`
/// uniform block. It used to ride a `PleCombineParams { inv_sqrt2, n }` storage
/// buffer -- Metal's layout, ported twice -- whose second word was dead: `n`
/// was one ROW's element count and the elementwise grid is `width * rows`, so
/// it could not bound the dispatch it was written for, and the field stayed
/// only to hold the struct's size. Word 0 of the statement's run is the same
/// number either way; the mark reaches it by index instead of by field, and
/// with no struct left there is nothing for `n` to hold the size of.
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
    if width <= 0 {
        return Err(Refusal::Empty { what: "width" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    // `LaunchRule::Elementwise`, whole: one lane per element of the
    // rectangle, on one axis. The division into workgroups is the driver's --
    // `@workgroup_size` is in the WGSL and this crate does not reflect it.
    let lanes = width.unsigned_abs() * rows.unsigned_abs();
    ctx.fire(
        Fire::at("layout/ple_combine.wgsl", "ple_combine_bfloat16").apply([lanes, 1, 1]),
        &[proj.arg(), token.arg(), out.arg(), inv_sqrt2.arg()],
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
    let rows = 1;
    let lanes = elementwise(hidden, rows)?;
    ctx.fire(
        Fire::at("layout/embed_gather.wgsl", EMBED_GATHER[affine_point(*group, *bits)?]).apply(lanes),
        &[w.arg(), scales.arg(), biases.arg(), id.arg(), out.arg(), hidden.arg()],
    )
}

/// The M>1 form, and the one a text should name.
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
    let lanes = elementwise_rows(hidden, rows)?;
    ctx.fire(
        Fire::at("layout/embed_gather.wgsl", EMBED_GATHER_MB[affine_point(*group, *bits)?]).apply(lanes),
        &[w.arg(), scales.arg(), biases.arg(), id.arg(), out.arg(), hidden.arg()],
    )
}

/// [`embed_gather_4bit`] with the embedding scale folded in -- gemma
/// multiplies its embeddings by `sqrt(hidden)`, which the statement carries.
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
    let lanes = elementwise(hidden, 1)?;
    ctx.fire(
        Fire::at("layout/embed_gather.wgsl", EMBED_GATHER_SCALED[affine_point(*group, *bits)?]).apply(lanes),
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

/// The scaled gather over M>1 rows.
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
    let lanes = elementwise_rows(hidden, rows)?;
    ctx.fire(
        Fire::at("layout/embed_gather.wgsl", EMBED_GATHER_SCALED_MB[affine_point(*group, *bits)?]).apply(lanes),
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

/// The readout's gather: one distribution per REQUEST out of one row per
/// TOKEN.
///
/// # Two words, one block, and the order is the body's
///
/// `count` used to be [`crate::routine::InPacked`]: the second FIELD of the `RowGatherParams`
/// struct that buffer 3 bound, neither a uniform-block scalar nor a buffer of
/// its own, written by the driver while it filled that buffer. The type was
/// what said so; under `kernel!` the same fact was an operand's `Ty::InPacked`
/// and the shader's struct, and only one of the two was ever checked.
///
/// There is no struct now. `width` is the statement's word 0 read through a
/// `Const<u32>` mark, and the request count -- which is the FIRE's and not the
/// statement's, so it is asked for rather than marked -- is an ordinary scalar
/// this body passes straight after it. `driver-wgpu::lowering::routine::bind`
/// packs body-passed scalars into the `@group(1)` uniform block in the order
/// the body passed them, so `[width, count]` lands with the layout the struct
/// had, field for field, and `layout/row_gather.wgsl` reads
/// `params.width`/`params.count` off the uniform where it read them off the
/// storage block. The old worry -- that folding `count` into a uniform would
/// push a word no shader read and leave `params.count` holding whatever the
/// params buffer contained -- was about a block that was otherwise EMPTY, and
/// this one is not: both its words are read.
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
    // A `u32` and no longer an `InPacked`: there is no struct for it to be a
    // field of, so it is a scalar like any other and takes its place in the
    // block by the position the body gives it.
    let count = ctx.ask::<u32, keys::RequestCount>()?;
    let row_count = ctx.ask::<i32, keys::Rows>()?;
    let lanes = elementwise_rows(input.width, row_count)?;
    ctx.fire(
        Fire::at("layout/row_gather.wgsl", "row_gather_bfloat16").apply(lanes),
        &[input.arg(), out.arg(), rows.arg(), width.arg(), count.arg()],
    )
}

#[cfg(test)]
mod ported {
    use super::*;
    use crate::routine::ArgValue;
    use core::cell::{Cell, RefCell};
    use kernels::Ty;

    /// One dispatch, as the recorder kept it.
    type Kept = (String, String, [u32; 3], Vec<ArgValue>);

    /// An `Encode` that records instead of dispatching, and answers a body's
    /// asks the same way `tests/routines.rs`'s `Seen` does: generically, by
    /// [`kernels::Ty`] alone, with two exceptions.
    ///
    /// `rows` is held in a `Cell` rather than answered by a fixed default,
    /// because `an_empty_block_is_refused_rather_than_launched_as_nothing`
    /// needs a ZERO there while every other test in this module needs a real
    /// row count -- and `rows` is asked for from inside the body now, not a
    /// positional argument a caller can vary directly.
    ///
    /// NO `ctx.params()` ANSWER ANY MORE. This used to hold a fixed handle for
    /// `kernels::Source::Slot(kernels::Kind::Params, 0)`, small and constant so
    /// that `the_body_asks_for_the_elementwise_grid`'s exact dispatched list
    /// stayed legible; `ple_combine` and `row_gather` were the two bodies that
    /// asked for it, and both state their scalars as `Const` marks now. A mark
    /// is passed by the CALLER, not resolved, so the tests below hand the
    /// numbers in and this probe never sees the params run at all.
    struct Recorder {
        seen: RefCell<Vec<Kept>>,
        /// What `ctx.ask::<i32, keys::Rows>()` answers.
        rows: Cell<i32>,
        /// A source of buffer handles for any OTHER asked fact, clear of the
        /// small handles this file's tests hand its routines directly.
        asked: Cell<u32>,
    }

    impl Default for Recorder {
        fn default() -> Self {
            Self {
                seen: RefCell::default(),
                rows: Cell::new(7),
                asked: Cell::new(0),
            }
        }
    }

    impl Recorder {
        /// A fresh handle, clear of every test's own small numbers.
        fn asked_buffer(&self) -> u32 {
            let at = 900 + self.asked.get();
            self.asked.set(self.asked.get() + 1);
            at
        }
    }

    impl crate::routine::Encode for Recorder {
        fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
            self.seen.borrow_mut().push((
                fire.file.to_owned(),
                fire.entrypoint.to_owned(),
                fire.lanes,
                args.to_vec(),
            ));
            Ok(())
        }

        fn resolve(&self, ty: kernels::Ty, source: kernels::Source) -> Result<ArgValue, Refusal> {
            // The statement's own scalars, read by index where the params run
            // is the shader's struct -- see `Asks::param`.
            if let kernels::Source::Slot(kernels::Kind::Param, n) = source {
                let _ = n;
                return Ok(ArgValue::I32(4096));
            }
            use kernels::{Source as Src, Ty as T};
            if source == Src::Named("rows") {
                return Ok(ArgValue::I32(self.rows.get()));
            }
            // THE REQUEST COUNT, which `row_gather` asks for as a plain `u32`
            // now that it is a scalar of the body's own rather than a field
            // the driver wrote into a struct. Answered by name and not by the
            // `T::U32` arm below, so that
            // `the_row_gathers_two_words_are_the_width_then_the_count` can
            // tell it apart from the width the caller passed.
            if source == Src::Named("request_count") {
                return Ok(ArgValue::U32(3));
            }
            Ok(match ty {
                T::Buf
                | T::BufMut
                | T::Bf16s
                | T::Bf16sMut
                | T::F16s
                | T::F16sMut
                | T::I32s
                | T::I32sMut
                | T::U32s
                | T::U32sMut
                | T::U8s
                | T::U8sMut
                | T::F32s
                | T::F32sMut => ArgValue::Buffer(self.asked_buffer()),
                T::I32 => ArgValue::I32(8),
                T::U32 => ArgValue::U32(8),
                T::F32 => ArgValue::F32(1.0),
                T::Usize => ArgValue::Usize(4096),
                T::InPacked => ArgValue::U32(8),
                _ => {
                    return Err(Refusal::Unstated {
                        what: "a fact this recorder does not answer",
                    });
                }
            })
        }
    }

    /// The row this routine replaces is derived from its signature.
    ///
    /// FOUR arguments, not the six a `kernel!` row stated. The row this
    /// replaced was positional over everything a launch read, `Env` included,
    /// so it carried `width` and `rows` beside the four buffers. Neither
    /// scalar is a parameter of any kind any more: `width` comes off
    /// `proj.width`, the mark's own rectangle, and `rows` is asked for from
    /// inside the body (`ctx.ask::<i32, keys::Rows>()` -- see
    /// `Recorder::resolve`, which is what a test answers that ask with now
    /// that `Env` provenance cannot). What is left on the signature, and so in
    /// `row.args`, is the three buffers the shader binds -- two reads and a
    /// write, which is `Ty::Bf16sMut` and not a third `Bf16s`, because
    /// [`kernels::shader::Element::TY_MUT`] is what a mark's write/read split
    /// says once a buffer's own type no longer implies which -- and then the
    /// one SCALAR the shader reads.
    ///
    /// That fourth entry is the migration. `inv_sqrt2` used to arrive inside
    /// `PleCombineParams` on a storage binding, which is a `Ty::Buf` the
    /// signature never mentioned because `ctx.params()` is a call and not a
    /// parameter; as a `Const<f32>` mark it is `Ty::F32` here, resolved to
    /// `Source::Slot(Kind::ParamF32, 0)` -- word 0 of the same statement run
    /// the struct's first field read.
    #[test]
    fn the_routines_row_is_its_signature_and_names_the_two_that_were_not_operands() {
        let row = crate::ROUTINES
            .iter()
            .find(|r| r.name == "ple_combine")
            .expect("the family declares it");

        assert_eq!(
            row.args.iter().copied().collect::<Vec<_>>(),
            [Ty::Bf16s, Ty::Bf16s, Ty::Bf16sMut, Ty::F32],
            "the row's operands, in the row's order: two reads, a write, and \
             the join's scale"
        );
    }

    /// The body asks for the grid `LaunchRule::Elementwise` asked for.
    ///
    /// `Elementwise` is `[width * rows, 1, 1]` lanes -- `geometry.rs:551` --
    /// and this is the same number reached by code instead of by a rule. The
    /// driver-side half of this check, which compares against `geometry::groups`
    /// itself rather than against a transcription of it, is
    /// `the_first_ported_routine_asks_for_the_grid_its_row_asked_for`.
    ///
    /// # No file, any more
    ///
    /// This used to compare `fire.module` against a literal `.wgsl` path.
    /// `Fire` carries no `module` field today -- the shared type's field is
    /// `file`, and it is EMPTY for every wgpu fire, because this backend (like
    /// vulkan) reaches a point through the module registry
    /// [`crate::source::entrypoint_source`] builds from the entrypoint STRING
    /// alone, never by a path a body states. Checking `file` against a
    /// literal would now be checking that it is always the empty string,
    /// which says nothing about `ple_combine` in particular; asking the
    /// registry the same question a driver would is the check that replaces
    /// it.
    #[test]
    fn the_body_asks_for_the_elementwise_grid() {
        let to = Recorder::default();
        ple_combine(
            &to,
            In { ptr: Tensor::<bf16>::new(0), rows: 0, width: 64 },
            In { ptr: Tensor::<bf16>::new(1), rows: 0, width: 64 },
            Out { ptr: Tensor::<bf16>::new(2), rows: 0, width: 64 },
            Const::new(core::f32::consts::FRAC_1_SQRT_2),
        )
        .expect("it dispatches");

        let seen = to.seen.borrow();
        let (file, entrypoint, lanes, args) = seen.first().expect("one dispatch");
        assert_eq!(
            file, "layout/ple_combine.wgsl",
            "the body states the file its point lives in"
        );
        assert_eq!(
            entrypoint, "ple_combine_bfloat16",
            "the axis point, pasted by the body"
        );
        // THE PAIR HAS TO AGREE, which is the whole reason the file is stated.
        // `entrypoint_source` used to recover the file by re-parsing all 38
        // embedded sources and taking the first entrypoint that matched; a
        // body that names both makes the scan a lookup, and makes a mismatch
        // between the two a test failure rather than a silently different
        // module.
        assert!(
            crate::source::source(file)
                .expect("the file the body named is in the tree")
                .contains(entrypoint),
            "`{entrypoint}` is declared by `{file}`"
        );
        crate::source::entrypoint_source(entrypoint, crate::Capability::Baseline)
            .unwrap_or_else(|e| panic!("`{entrypoint}` names no module the tree carries: {e}"));
        assert_eq!(*lanes, [64 * 7, 1, 1], "width * rows on one axis");
        assert_eq!(
            args,
            &[
                ArgValue::Buffer(0),
                ArgValue::Buffer(1),
                ArgValue::Buffer(2),
                ArgValue::F32(core::f32::consts::FRAC_1_SQRT_2)
            ],
            "the three operands and the join's scale -- and NOT `width` or \
             `rows`, which are no longer arguments at all: `width` comes off \
             `proj`'s own rectangle and `rows` is asked for from inside the \
             body (`Recorder::resolve` answers it, here as 7). The fourth \
             entry used to be `Buffer(3)`, the handle `ctx.params()` minted \
             for the staged `PleCombineParams` block; the scale is a \
             `Const<f32>` mark now, so it reaches the driver as the NUMBER \
             the caller passed and `driver-wgpu`'s `bind` packs it into the \
             `@group(1)` uniform instead of staging a storage buffer for it"
        );
    }

    /// `row_gather` passes TWO words, and the width comes first.
    ///
    /// The gather is the one row in this family whose shader reads two
    /// scalars, and the order between them is an ABI: `layout/row_gather.wgsl`
    /// declares `struct Params { width: u32, count: u32 }` and
    /// `driver-wgpu::lowering::routine::bind` packs the body's scalars into
    /// the `@group(1)` block in the order the body passed them, so a swap here
    /// binds the request count as the row pitch and the pitch as the request
    /// count. Neither is a type error, both are plausible `u32`s, and the
    /// gather would read whole rows out of the wrong place and report success.
    ///
    /// What this replaces is the same claim about the other carrier. `width`
    /// was word 0 of a `RowGatherParams` STRUCT that a storage binding bound
    /// and `count` was `Ty::InPacked` -- a value with no slot of its own,
    /// appended to that struct's staged run by the driver. There is no struct
    /// and no packed field now: `width` is a `Const<u32>` mark reading the
    /// same word 0 of the same statement run by index, and `count` is an
    /// ordinary `u32` the body asks the fire for and passes straight after it.
    #[test]
    fn the_row_gathers_two_words_are_the_width_then_the_count() {
        let to = Recorder::default();
        to.rows.set(3);
        row_gather(
            &to,
            In { ptr: Tensor::<bf16>::new(0), rows: 0, width: 2048 },
            Out { ptr: Tensor::<bf16>::new(1), rows: 0, width: 2048 },
            Const::new(2048),
        )
        .expect("three requests is a launch");

        let seen = to.seen.borrow();
        let (_, entrypoint, lanes, args) = seen.first().expect("one dispatch");
        assert_eq!(entrypoint, "row_gather_bfloat16");
        assert_eq!(
            *lanes,
            [2048, 3, 1],
            "one row per REQUEST, not one per token -- that is the whole point \
             of the gather"
        );
        assert_eq!(
            args.len(),
            5,
            "input, out, the index table, and then the two words of the block"
        );
        assert_eq!(
            args[3],
            ArgValue::U32(2048),
            "the width the statement carries, which the mark reads at slot 0"
        );
        assert_eq!(
            args[4],
            ArgValue::U32(3),
            "and the request count the fire answers, which `Recorder::resolve` \
             gives as 3 -- a different number from the width on purpose, so a \
             swap of the two cannot pass this"
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
    ///
    /// FILTERED BY NAMESPACE, and compared UNORDERED, neither of which this
    /// test needed before every other family crossed. `ROUTINES` is a
    /// crate-wide `linkme` distributed slice -- one list every family's
    /// `#[routine]` pushes into, from every object file the linker sees --
    /// and this test predates the other nine families joining it: at the
    /// time it was written this family's six were the only entries there, in
    /// the order the one file declared them, so reading `ROUTINES` whole and
    /// comparing it to a literal Vec answered both "are these the six" and
    /// "in this order" at once. Neither survives a hundred routines from ten
    /// families sharing the slice: the check now narrows to the routines
    /// whose `namespace` (`module_path!()`'s own segment, see
    /// [`kernels::routine::namespace`]) is `"layout"` -- and sorts before
    /// comparing, because a distributed slice's order across TRANSLATION
    /// UNITS is the linker's to choose, not this crate's, and it is in fact
    /// no longer declaration order (`row_gather` links first despite being
    /// declared last). The claim was always "the family declares these six",
    /// never "in this order", so sorting both sides loses nothing the test
    /// meant to check.
    #[test]
    fn the_family_declares_the_six_it_has_ported() {
        let mut names: Vec<&str> = crate::ROUTINES
            .iter()
            .filter(|r| r.namespace == "layout")
            .map(|r| r.name)
            .collect();
        names.sort_unstable();
        let mut want = [
            "ple_combine",
            "embed_gather_4bit",
            "embed_gather_mb_4bit",
            "embed_gather_scaled_4bit",
            "embed_gather_scaled_mb_4bit",
            "row_gather",
        ];
        want.sort_unstable();
        assert_eq!(names, want);
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
        to.rows.set(R);
        let (b, m) = (Tensor::<bf16>::new(0), Tensor::<bf16>::new(1));
        ple_combine(
            &to,
            In { ptr: b, rows: 0, width: W },
            In { ptr: b, rows: 0, width: W },
            Out { ptr: m, rows: 0, width: W },
            Const::new(core::f32::consts::FRAC_1_SQRT_2),
        )
        .expect("dispatches");
        embed_gather_4bit(
            &to,
            Const::new(Tensor::<u32>::new(0)),
            Const::new(b),
            Const::new(b),
            Out { ptr: m, rows: 0, width: W },
            Const::new(64),
            Const::new(4),
        )
        .expect("dispatches");
        embed_gather_mb_4bit(
            &to,
            Const::new(Tensor::<u32>::new(0)),
            Const::new(b),
            Const::new(b),
            Out { ptr: m, rows: 0, width: W },
            Const::new(64),
            Const::new(4),
        )
        .expect("dispatches");
        embed_gather_scaled_4bit(
            &to,
            Const::new(Tensor::<u32>::new(0)),
            Const::new(b),
            Const::new(b),
            Out { ptr: m, rows: 0, width: W },
            Const::new(1.0),
            Const::new(64),
            Const::new(4),
        )
        .expect("dispatches");
        embed_gather_scaled_mb_4bit(
            &to,
            Const::new(Tensor::<u32>::new(0)),
            Const::new(b),
            Const::new(b),
            Out { ptr: m, rows: 0, width: W },
            Const::new(1.0),
            Const::new(64),
            Const::new(4),
        )
        .expect("dispatches");
        row_gather(
            &to,
            In { ptr: b, rows: 0, width: W },
            Out { ptr: m, rows: 0, width: W },
            Const::new(W.unsigned_abs()),
        )
        .expect("dispatches");

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
            // `every_launchs_scalars_land_where_its_module_reads_them` compared
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
    ///
    /// Two refusals, and each needs its own construction now rather than one
    /// loop over a shared pair of positional arguments. `width` used to be a
    /// positional `Env` scalar a caller could vary directly; it comes off
    /// `proj.width`, the mark's own rectangle, so an empty width is an empty
    /// `In`. `rows` used to sit right beside it; it is asked for from inside
    /// the body now, so an empty rows is answered by `Recorder::rows`
    /// instead of passed.
    #[test]
    fn an_empty_block_is_refused_rather_than_launched_as_nothing() {
        let to = Recorder::default();
        assert_eq!(
            ple_combine(
                &to,
                In { ptr: Tensor::<bf16>::new(0), rows: 0, width: 0 },
                In { ptr: Tensor::<bf16>::new(1), rows: 0, width: 0 },
                Out { ptr: Tensor::<bf16>::new(2), rows: 0, width: 0 },
                Const::new(core::f32::consts::FRAC_1_SQRT_2),
            ),
            Err(Refusal::Empty { what: "width" }),
            "`dispatch_workgroups(0, 1, 1)` is legal WebGPU that runs \
             nothing and reports success"
        );

        // `width` is real here (64) so only `rows`, which `Recorder` now
        // answers as zero, can be the empty one.
        to.rows.set(0);
        assert_eq!(
            ple_combine(
                &to,
                In { ptr: Tensor::<bf16>::new(0), rows: 0, width: 64 },
                In { ptr: Tensor::<bf16>::new(1), rows: 0, width: 64 },
                Out { ptr: Tensor::<bf16>::new(2), rows: 0, width: 64 },
                Const::new(core::f32::consts::FRAC_1_SQRT_2),
            ),
            Err(Refusal::Empty { what: "rows" }),
            "the same refusal, for the fact that is asked for rather than \
             read off a mark"
        );

        assert!(to.seen.borrow().is_empty(), "and nothing was dispatched");
    }
}
