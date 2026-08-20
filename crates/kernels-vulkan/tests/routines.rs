//! Every crossed routine, asked the questions a family test cannot ask alone.
//!
//! The bodies are proved one family at a time, against what that kernel means.
//! These are the properties that hold for ALL of them, and they exist because
//! a body can be wrong in ways its own family test is looking straight past.
//!
//! One of them was wrong exactly this way while being written.
//! `mlp::gptoss_swiglu` takes a `params` buffer, states it in its signature,
//! is compared against its row by `kernels/tests/shader_backends_agree.rs` --
//! and bound only its first three arguments. Every check in the tree passed:
//! the signature was right, the row agreed with it, the entrypoint existed.
//! The kernel would have been dispatched against a descriptor set one short of
//! what its layout declares, which on this backend is not an error.

use std::cell::RefCell;

use kernels::Ty;
use kernels::routine::{Refusal, Routine};

/// One recorded dispatch: the entrypoint, the lanes, the argument list.
type Call = (String, [u32; 3], Vec<ArgValue>);
use kernels_vulkan::routine::{ArgValue, Encode, Fire, Vulkan};

/// A `Encode` that remembers, and answers generically.
#[derive(Default)]
struct Seen(RefCell<Vec<Call>>);

impl Encode for Seen {
    fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
        self.0
            .borrow_mut()
            .push((fire.entrypoint.to_owned(), fire.lanes, args.to_vec()));
        Ok(())
    }

    /// Answers generically, by [`kernels::Ty`] alone, with two deliberate
    /// exceptions.
    ///
    /// None of this file's tests reads the VALUE an asked fact resolves to --
    /// only order, arity, entrypoint names and buffer counts -- so a probe
    /// that answers honestly by TYPE, ignoring which specific fact was named,
    /// is enough for every routine but one. `ssm::gdn_core_recurrent_prefill`
    /// turns two asked scalars into a compiled-shape LOOKUP (`scan_point`)
    /// rather than a bare forward: only nine `(LANES, VROWS)` pairs are
    /// compiled, and a generic default for each would almost certainly name
    /// none of them, faulting inside `vkCreateComputePipelines` rather than
    /// erring at compile time. `(32, 4)` is one of the nine, so answering it
    /// unconditionally keeps this routine's body running whole -- at the cost
    /// of the eight other compiled shapes, which `SWEEP` covered when `lanes`
    /// and `vrows` were the positional pair it overrode and cannot reach now
    /// that both are asked from inside the body instead of read off the
    /// argument list.
    fn resolve(&self, ty: kernels::Ty, source: kernels::Source) -> Result<ArgValue, Refusal> {
        use kernels::{Source as Src, Ty as T};
        // THE SAME TWO NUMBERS BY THE OTHER ROUTE. The tiling was
        // `Source::Named("lanes")` while the body asked for it as a fact; it
        // is the statement's eleventh and twelfth WORDS now -- HEAD's
        // `Param<11>`/`Param<12>` -- because the run this body forwards is the
        // shader's struct and no `Const` mark can name a word inside it.
        if let Src::Slot(kernels::Kind::Param, n) = source {
            return Ok(ArgValue::I32(match n {
                11 => 32,
                12 => 4,
                _ => 4096,
            }));
        }
        if source == Src::Named("lanes") {
            return Ok(ArgValue::I32(32));
        }
        if source == Src::Named("vrows") {
            return Ok(ArgValue::I32(4));
        }
        // A PITCH MUST NOT BE NARROWER THAN THE ROW IT STRIDES OVER, and the
        // fixtures above hand every operand a 1024-wide rectangle. The
        // generic `8` below is a plausible extent and an impossible pitch:
        // `neox_strided` refuses it by name, which is the body doing its job
        // and the probe failing to state a case.
        if matches!(
            source,
            Src::Named("row_pitch") | Src::Named("row_stride") | Src::Named("in_width")
        ) {
            return Ok(ArgValue::I32(1024));
        }
        Ok(match ty {
            T::Buf | T::BufMut | T::Bf16s | T::Bf16sMut | T::I32s | T::I32sMut | T::U32s
            | T::U32sMut | T::U8s | T::U8sMut | T::F32s | T::F32sMut => {
                ArgValue::Buffer {
                    handle: 900,
                    writes: matches!(ty, T::BufMut | T::Bf16sMut | T::I32sMut | T::U32sMut | T::U8sMut | T::F32sMut),
                    // A RECTANGLE, because an asked table is an operand like
                    // any other now that `ArgValue` carries one variant: a
                    // body that reads `.width` off what it asked for gets zero
                    // otherwise, and zero is what every extent guard refuses.
                    rows: 7,
                    width: 1024,
                }
            }
            T::I32 => ArgValue::I32(8),
            T::U32 | T::InPacked => ArgValue::U32(8),
            T::F32 => ArgValue::F32(1.0),
            T::Usize => ArgValue::Usize(4096),
            _ => {
                return Err(Refusal::Unstated {
                    what: "a fact this generic mock does not answer",
                });
            }
        })
    }
}

/// A value of each kind, chosen so that a body that runs at all runs whole.
///
/// The extents are non-zero and small: every body in this tree refuses an
/// empty rectangle, which is correct and is also the one thing that would stop
/// this file from seeing anything. Buffer handles are their own positions, so
/// a bound list can be read back as the argument positions it came from.
///
/// A buffer-shaped position is [`ArgValue::Shaped`] now, not a plain
/// [`ArgValue::Buffer`]. `region` -- what `In`/`Out`/`InOut` call to fill
/// their own `.rows`/`.width` when they unpack -- reads a rectangle only off
/// `Shaped` and refuses every other variant, so a plain `Buffer` here would
/// make any body that reads its own operand's shape refuse before doing
/// anything else worth checking. `Const<Tensor<_>>` (the weight run) reads
/// only the handle, so the same value serves both runs; the writes flag
/// `ArgValue::Buffer` used to carry is gone with it -- writability rides the
/// MARK now (`In` reads, `Out`/`InOut` write), never the bound value, so
/// there is nothing left for this synthesised value to state either way.
fn stand_in(at: usize, ty: Ty) -> ArgValue {
    match ty {
        // THE MUTABLE HALVES ARE HERE TOO. `Ty` splits by direction because
        // it is the shader's spelling, and an `Out<Tensor<i32>>` lands on
        // `I32sMut` where its reading twin lands on `I32s` -- one operand,
        // two names, and the stand-in is the same rectangle either way.
        Ty::Buf | Ty::BufMut | Ty::Bf16s | Ty::Bf16sMut | Ty::I32s | Ty::I32sMut
        | Ty::U32s | Ty::U32sMut | Ty::U8s | Ty::U8sMut
        | Ty::F16s | Ty::F16sMut
        | Ty::F32s | Ty::F32sMut => ArgValue::Buffer {
            handle: at as u32,
            writes: matches!(
                ty,
                Ty::BufMut
                    | Ty::Bf16sMut
                    | Ty::I32sMut
                    | Ty::U32sMut
                    | Ty::U8sMut
                    | Ty::F16sMut
                    | Ty::F32sMut
            ),
            rows: 7,
            width: 1024,
        },
        Ty::I32 => ArgValue::I32(1),
        Ty::U32 | Ty::InPacked => ArgValue::U32(1),
        Ty::F32 => ArgValue::F32(1.0),
        Ty::Usize => ArgValue::Usize(1),
        // `Ty` is the union of every backend's vocabulary and this backend
        // uses thirteen of it -- measured over all 100 rows, and the same
        // thirteen metal and wgpu use. A routine that took a fourteenth could
        // not have compiled: `kernels::shader` declares an `Arg` impl for
        // these and no others, so `routine!` would have failed at its own line.
        other => panic!("no Vulkan operand has type {other:?}"),
    }
}

/// Scalar arguments that only a REAL value satisfies, by argument position.
///
/// Most scalars a body takes are extents, and any positive number is a
/// plausible one. A few are not: `layout`'s gathers take the affine group size
/// and bit width, and a body's whole job there is to refuse a point the shader
/// tree does not carry, so `1` is correctly rejected and this file would see
/// nothing.
///
/// Written out rather than guessed, and checked below for being neither stale
/// nor short. A recipe is the argument list a routine could really be called
/// with, so a routine that grows an argument shifts a position here and the
/// check that the override still lands on a scalar is what says so.
const RECIPE: &[(&str, &[(usize, i32)])] = &[
    // The affine point. Overwritten by the sweep below, and here so that the
    // position check applies to them too.
    // The point moved forward with the signature: `hidden` and the row
    // count left these lists -- the first is the result's own width and the
    // second is the fire's -- so `(group, bits)` are the last two arguments
    // rather than the last two of a longer run.
    ("embed_gather_4bit", &[(4, 32), (5, 4)]),
    ("embed_gather_mb_4bit", &[(4, 32), (5, 4)]),
    ("embed_gather_scaled_4bit", &[(5, 32), (6, 4)]),
    ("embed_gather_scaled_mb_4bit", &[(5, 32), (6, 4)]),
    // A rotation's grid IS its shape -- `neox.slang` reads the pair count and
    // the head count back out of `gl_NumWorkGroups` -- so a rotary width must
    // be even and a tensor width must be a whole number of heads. `1` is
    // neither, and correctly refused.
    // The row width and the token count left these lists with the
    // signatures: `x` is an `InOut<Tensor<bf16>>` and carries its own
    // rectangle, and the row count is the fire's, asked for. What is left is
    // the head width and the rotary width, which the statement carries.
    ("neox_decode", &[(3, 128), (4, 128)]),
    ("neox_prop_decode", &[(3, 128), (4, 128)]),
    ("neox_mb", &[(3, 128), (4, 128)]),
    ("neox_prop_mb", &[(3, 128), (4, 128)]),
    // `inv_freq` went too -- the FIRE's frequency table, asked for -- so the
    // head width sits one earlier again.
    ("neox_freqs_decode", &[(2, 128), (4, 128)]),
    ("neox_freqs_mb", &[(2, 128), (4, 128)]),
    // The fused norm+rope refuses the same shapes a rotation does and for the
    // same reasons -- but it has no scalar argument left to give a recipe FOR.
    // `axis`, `row_pitch` and the rotary width are all fields of the
    // `RmsRopeParams` struct the body forwards whole, read by index through
    // `Ctx::param` rather than taken as marks, because a `Const` mark is
    // derived onto the run by mark order and cannot name a word inside a
    // struct. `Seen::resolve` above answers a `Slot(Param, n)` generically, so
    // the shapes this recipe used to state arrive by that route instead.
    // `row_pitch` at 5, and it may not be narrower than the row.
    (
        "neox_strided",
        // The width and the row count left with the rest -- `x` carries its
        // own rectangle and the fire answers the row count -- but the PITCH
        // came back: it is `Param<4>`'s successor, the statement's own, and
        // the stand-in's 1 is narrower than any row it builds.
        &[(3, 128), (4, 128), (5, 4096)],
    ),
    // The head width, for the attention routines that carry ONE point. The
    // swept ones are below; these are here so the position check applies to
    // them too, because a head width is the argument in this family most
    // likely to move.
    ("sdpa_paged_decode_sink", &[(6, 64)]),
    ("sdpa_paged_mma", &[(5, 64)]),
    ("sdpa_paged_mma_sink", &[(6, 64)]),
    ("sdpa_paged_tiled_sink", &[(6, 64)]),
    ("sdpa_paged_tiled_strided", &[(5, 256)]),
    ("sdpa_vector_decode_sink", &[(5, 64)]),
];

/// The routines whose ENTRYPOINT is chosen by their arguments, with the
/// positions that pick it and the points the shader tree actually carries.
///
/// These are swept over every point rather than called at one, because the
/// point selects the entrypoint and an unbuilt module is not an error on this
/// backend: `vkCreateComputePipelines` faults on it, with the validation layer
/// silent. Six spellings per `layout` gather -- twenty-four in that family --
/// nine for `ssm`'s prefill scan, and seventy-two across `moe`'s three routed
/// matmuls.
///
/// A point is a LIST of values, one per swept position, and the count of
/// positions is not fixed at two: `moe`'s `affine_qmm_t_routed` picks its
/// module on FOUR numbers -- group size, bit width, and the two tile extents.
///
/// Whether the points are a product is a fact about the shader tree and not
/// about the sweep, which is why each entry names a function rather than a
/// table shape. `ssm`'s nine are the clear case AGAINST multiplying: `(32, 1)`
/// and `(4, 2)` read as obvious members of a three-by-four grid and neither is
/// compiled, so a sweep that multiplied the axes out would name six modules
/// that do not exist. `moe`'s fifty-four are the clear case FOR: the
/// `pie:instantiate` block is 3 x 2 x 3 x 3 with no gaps, and writing the
/// fifty-four out by hand would be fifty-four chances to typo a name the
/// crate would then never dispatch.
/// One swept routine: its name, the argument positions that pick the module,
/// and the points to sweep them over.
type Swept = (&'static str, &'static [usize], fn() -> Vec<Vec<i32>>);

const SWEEP: &[Swept] = &[
    ("embed_gather_4bit", &[4, 5], affine),
    ("embed_gather_mb_4bit", &[4, 5], affine),
    ("embed_gather_scaled_4bit", &[5, 6], affine),
    ("embed_gather_scaled_mb_4bit", &[5, 6], affine),
    // `gdn_core_recurrent_prefill` used to be swept here on `(lanes, vrows)`
    // at positions 9 and 10 -- both gone from the signature now that the body
    // asks for them, so there is no argument position left to override, and
    // `Seen::resolve`'s two `Source::Named` exceptions answer a single
    // compiled point (32, 4) in their place instead. See that doc comment.
    ("mxfp4_qmm_t_routed_bias", &[7, 8], tiles),
    ("qmm_t", &[5, 6, 7, 8], routed_qmm),
    ("qmm_t_bias", &[6, 7, 8, 9], routed_qmm),
    ("qmm_t_residual", &[6, 7, 8, 9], routed_qmm),
    ("qmm_t_fp16_precast", &[5, 6], tiles),
    ("qmm_t_bias_fp16_precast", &[6, 7], tiles),
    ("qmm_t_residual_fp16_precast", &[6, 7], tiles),
    ("qmm_t_splitk", &[5, 6, 7], wide_qmm),
    ("qmm_t_splitk_f32", &[5, 6, 7], wide_qmm),
    ("qmm_t_splitk_fp16_precast", &[5], row_tiles),
    ("qmm_t_splitk_fp16_precast_f32", &[5], row_tiles),
    ("qmm_t_strided", &[5, 6, 7], wide_qmm),
    ("qmm_t_strided_residual", &[6, 7, 8], wide_qmm),
    ("qmm_t_strided_fp16_precast", &[5], row_tiles),
    ("qmm_t_strided_fp16_precast_residual", &[6], row_tiles),
    ("qmv_fast", &[5, 6], affine),
    ("qmv_fast_residual", &[6, 7], affine),
    ("qmv_tail", &[5], bit_widths),
    ("qmv_tail_bias", &[6], bit_widths),
    ("qmv_wide_strided", &[5], bit_widths),
    ("qmm_t_routed", &[7, 8, 9, 10], routed_qmm),
    ("qmm_t_routed_fp16", &[7, 8], tiles),
    // ONE swept position, which the widened `Swept` makes as ordinary as four:
    // attention picks its module on the head width alone.
        // `paged_split` swept a head width and a SPLIT COUNT; the split is the
    // plan's, which the body asks for, so only the width is a position now.
    ("sdpa_paged_decode", &[5], paged_dims),
    ("sdpa_paged_tiled", &[5], paged_dims),
    ("sdpa_vector_decode", &[3], vector_dims),
    ("sdpa_vector_decode_swa", &[4], swa_dims),
];

/// The four head widths, each at one split and at eight.
///
/// The split count is a second module selector and not a mere number: above
/// one, `sdpa_paged_decode` stops dispatching `sdpa_paged_decode_bfloat16_d_*`
/// and dispatches a `_split_` pass and a `_combine_` fold instead. Sweeping
/// only the widths would leave nine of this family's modules named by nothing
/// -- which on this backend is not a compile error but a fault inside
/// `vkCreateComputePipelines`.
fn paged_split() -> Vec<Vec<i32>> {
    let mut out = Vec::new();
    for d in paged_dims() {
        for splits in [1, 8] {
            out.push(vec![d[0], splits]);
        }
    }
    out
}

/// The four head widths the paged kernels are compiled for.
///
/// `sdpa_paged_decode`'s ROW states seven points; this is four, and the
/// difference is deliberate -- see `attn::PAGED_DECODE`, whose three
/// page-shape tails are unreachable and one of which is a bare name. A sweep
/// over seven would name three modules the routine cannot spell.
fn paged_dims() -> Vec<Vec<i32>> {
    [64, 128, 256, 512].iter().map(|d| vec![*d]).collect()
}

/// The three the dense decode carries. 512 is missing because gemma-4's wide
/// layers are paged.
fn vector_dims() -> Vec<Vec<i32>> {
    [64, 128, 256].iter().map(|d| vec![*d]).collect()
}

/// The two the sliding window carries.
fn swa_dims() -> Vec<Vec<i32>> {
    [256, 512].iter().map(|d| vec![*d]).collect()
}

/// The six affine points the shader tree carries, in `affine_point` order.
fn affine() -> Vec<Vec<i32>> {
    [(32, 4), (32, 8), (64, 4), (64, 8), (128, 4), (128, 8)]
        .iter()
        .map(|(g, b)| vec![*g, *b])
        .collect()
}

/// The eighteen `(group, bits, BM)` points the wide forms carry.
///
/// `qmm_t_splitk` and `_strided` instantiate `_bn_32` alone, so the column
/// tile is not an axis here and the row tile is swept on its own.
fn wide_qmm() -> Vec<Vec<i32>> {
    let mut out = Vec::new();
    for group in [32, 64, 128] {
        for bits in [4, 8] {
            for bm in [16, 32, 64] {
                out.push(vec![group, bits, bm]);
            }
        }
    }
    out
}

/// The three row tiles the precast wide forms carry, and nothing else: their
/// codec is fixed at `gs_64 b_4` by the name and their column tile at 32.
fn row_tiles() -> Vec<Vec<i32>> {
    [16, 32, 64].iter().map(|bm| vec![*bm]).collect()
}

/// The two bit widths the matvec tail forms carry, at the one group size.
fn bit_widths() -> Vec<Vec<i32>> {
    [4, 8].iter().map(|b| vec![*b]).collect()
}

/// The nine `(BM, BN)` tiles every routed qmm is compiled for.
fn tiles() -> Vec<Vec<i32>> {
    let mut out = Vec::new();
    for bm in [16, 32, 64] {
        for bn in [16, 32, 64] {
            out.push(vec![bm, bn]);
        }
    }
    out
}

/// The fifty-four `(group, bits, BM, BN)` points the affine routed qmm carries.
fn routed_qmm() -> Vec<Vec<i32>> {
    let mut out = Vec::new();
    for group in [32, 64, 128] {
        for bits in [4, 8] {
            for tile in tiles() {
                out.push(vec![group, bits, tile[0], tile[1]]);
            }
        }
    }
    out
}

/// The argument lists one routine should be exercised with.
fn recipes(r: &Routine<Vulkan>) -> Vec<Vec<ArgValue>> {
    let mut base: Vec<ArgValue> = r
        .args
        .iter()
        .enumerate()
        .map(|(at, ty)| stand_in(at, *ty))
        .collect();

    if let Some((_, overrides)) = RECIPE.iter().find(|(n, _)| *n == r.name) {
        for (at, value) in *overrides {
            assert!(
                *at < base.len(),
                "`{}`'s RECIPE overrides argument {at}, and it takes {}. The \
                 signature moved and the recipe did not.",
                r.name,
                base.len()
            );
            assert!(
                matches!(base[*at], ArgValue::I32(_)),
                "`{}`'s RECIPE overrides argument {at}, which is {:?} and not \
                 an `i32`. The signature moved and the recipe did not.",
                r.name,
                base[*at]
            );
            base[*at] = ArgValue::I32(*value);
        }
    }

    let Some((_, at, points)) = SWEEP.iter().find(|(n, _, _)| *n == r.name) else {
        return vec![base];
    };
    for p in *at {
        assert!(
            matches!(base.get(*p), Some(ArgValue::I32(_))),
            "`{}`'s SWEEP picks its module on argument {p}, which is {:?} and \
             not an `i32`. The signature moved and the sweep did not.",
            r.name,
            base.get(*p)
        );
    }
    points()
        .into_iter()
        .map(|point| {
            assert_eq!(
                point.len(),
                at.len(),
                "`{}`'s SWEEP names {} position(s) and its points carry {}",
                r.name,
                at.len(),
                point.len()
            );
            let mut args = base.clone();
            for (p, v) in at.iter().zip(point) {
                args[*p] = ArgValue::I32(v);
            }
            args
        })
        .collect()
}

/// Run every routine against a recorder, with arguments it could really take.
fn fired() -> Vec<(&'static Routine<Vulkan>, Vec<ArgValue>, Seen)> {
    let all = kernels_vulkan::routines();
    for name in RECIPE
        .iter()
        .map(|(n, _)| n)
        .chain(SWEEP.iter().map(|(n, _, _)| n))
    {
        assert!(
            all.iter().any(|r| r.name == *name),
            "RECIPE or SWEEP names `{name}`, which is not a crossed routine. \
             A stale entry is a routine silently running on stand-ins."
        );
    }

    let mut out = Vec::new();
    for r in all {
        // One `Seen` per argument list: the comparison above is against the
        // arguments that produced the dispatch, and pooling them would compare
        // a swept gather's second point against its first point's recipe.
        for args in recipes(r) {
            let seen = Seen::default();
            (r.body)(&seen, &args).unwrap_or_else(|e| {
                panic!("`{}` refused {args:?}: {e:?}", r.name);
            });
            out.push((r, args, seen));
        }
    }
    out
}

/// A body passes the arguments its signature takes, as a SUBSEQUENCE and in
/// order.
///
/// Not "every buffer is bound" -- every buffer the recipe supplied, found
/// again among what the routine actually fired, in the same relative order it
/// was given. The weaker form was written first and missed a real thing
/// within the hour: giving `neox_freqs_decode` a seventh pushed word turned
/// nothing red, because a pushed float is not a descriptor and no other check
/// in either plane counts them. `kernels/tests/shader_backends_agree.rs`
/// compares `Routine::args` against the row and cannot see it either -- an
/// extra `.v()` in the body is not an extra argument in the signature.
///
/// What each half of the list costs when it is wrong:
///
/// A missing BUFFER is a descriptor set one entry short of what the module's
/// layout declares, which on this backend is not an error -- the slot holds
/// whatever it last held. `mlp::gptoss_swiglu` was written this way, taking a
/// `params` buffer and binding three of its four.
///
/// A missing or extra SCALAR is a push block written to a different layout
/// than the one the shader reads. `rope/neox.slang`'s block is
/// `{ float scale; float base; int head_dim; }` under one `#define` and
/// `{ float scale; int head_dim; float mscale; }` under another, so a word in
/// the wrong place is `head_dim` read as a float, not a diagnostic.
///
/// ORDER matters as much as membership, because a shader's bindings and its
/// push block are both POSITIONAL. This tree's trace order is not the
/// shader's for 2,898 of its 3,992 rectangles -- `norm/rms.slang` binds
/// `0=x, 1=w, 2=out` where the trace hands over `In(0), Out(0), Weight(0)` --
/// so a body that passes the right things in the wrong order reads its own
/// output as a weight and returns success.
///
/// The writability of each buffer rides along, and matters on its own:
/// `driver-vulkan` puts a barrier between two dispatches only when they touch
/// the same bytes and decides that from which operands may write. The
/// SIGNATURE's `Ty` -- `BufMut` and its element-typed siblings -- is the only
/// statement of it now: a mark's own `.arg()` answers `writes` from its OWN
/// Rust type (`In`/`Const` always false, `Out`/`InOut` always true) and never
/// from the bound value, so this check reads the expectation off `Ty` at each
/// recipe position and asks whether the fired flag agrees, rather than off
/// the recipe's own value -- which is `Shaped` now and carries no such flag
/// to compare (see `stand_in`).
///
/// # The BUFFERS, and only they
///
/// This used to filter the signature to `Provenance::Trace` and compare the
/// whole argument list, and both halves of that were wrong: an `Env` SCALAR
/// was a grid fact that never reached the device, but an `Env` BUFFER was the
/// staged parameter block, which reached it every time -- `split_qkv_bf16`
/// bound `packed, q, k, v, params` and only the first four were the trace's,
/// so filtering on `Provenance::Trace` dropped `params` from what the
/// signature was said to take and then reported the body inventing a fifth
/// buffer.
///
/// `Provenance` is gone along with the rest of `Env`, and what replaced it
/// settles the same question more simply: most of what was `Env` is now
/// either a `Const` parameter (still positional, still in `r.args`, still
/// part of `want` below) or a fact the body `ctx.ask`s for from INSIDE
/// itself -- never on the signature at all, so `params` and an asked
/// operand's own buffer are simply ABSENT from `r.args` and therefore from
/// `want`, even though the body still fires them to the device. So `want` is
/// no longer everything `got` contains; it is a subset that must still appear
/// in order, and `got` is allowed extra buffers `want` never named. That is
/// why the loop below walks `want` looking for each handle in `got`, rather
/// than the other way around -- the direction that would require every fired
/// buffer to trace back to a signature position, which an asked buffer never
/// will.
///
/// - An asked scalar or buffer reaching the device is ORDINARY, not a fault.
///   `gdn_core_recurrent_prefill` asks for three buffers and two scalars this
///   way, none of which are in `r.args` at all.
/// - Every synthesised scalar here is `I32(1)` or similar. A subsequence
///   check over values that cannot be told apart passes for any permutation
///   of them, so the scalar half stays out of the comparison for the same
///   reason it always was: what would make it real is distinct synthesised
///   values per position, which is a change to `fired()` and a different
///   test's argument.
///
/// The shape is read off the synthesised VALUE and not off `Ty`, because
/// `Ty::Buf` and `Ty::BufMut` are not the whole of what reaches the device
/// as a buffer -- `kv_append_paged`'s page tables arrive as `U32s`, and a
/// `Ty` list written out by hand here would be a second place to keep the
/// answer.
///
/// What the check does NOT require is that every trace argument is forwarded,
/// because slangc DELETES a global nothing reads and the deleted ones cannot
/// be passed. `attn/sdpa_paged.slang` declares `sinks` at binding 10 whether
/// or not `PIE_WITH_SINK` is set; without it the module decorates 0..=9 and
/// `encode::dispatch` -- which requires exactly
/// `declared.bindings - holes()` buffers -- refuses an eleventh. The row
/// still states `sinks`, because a row is positional and the trace still has
/// one to hand over, so the SIGNATURE keeps it and the CALL drops it.
///
/// So the invariant is a SUBSEQUENCE, in order, writability included: a body
/// may skip a signature buffer, or interleave one the signature never named
/// (an asked fact's own buffer, or the staged `params` block), and may not
/// reorder two signature buffers, invent one, or change a writability flag on
/// the way through. `every_routine_binds_the_buffers_its_module_uses_and_no_
/// others` is what says the skipping was right; this says the keeping was.
#[test]
fn a_body_passes_a_subsequence_of_the_arguments_its_signature_takes_in_order() {
    /// The handle underneath a buffer-shaped argument, recipe or fired.
    ///
    /// A recipe's own values are always [`stand_in`]'s `Shaped`, and every
    /// fired value a mark forwards is always a plain `Buffer` (`Tensor::arg`
    /// reads only `.handle`, so the two variants never compare equal to each
    /// other even when they name the same buffer) -- this is the projection
    /// that makes the two sides comparable at all.
    fn handle_of(v: &ArgValue) -> Option<u32> {
        match *v {
            ArgValue::Buffer { handle, .. } => Some(handle),
            _ => None,
        }
    }

    /// Whether the signature's `Ty` at this recipe position says the fired
    /// buffer should come back writable.
    fn expects_write(ty: Ty) -> bool {
        matches!(ty, Ty::BufMut | Ty::Bf16sMut | Ty::F32sMut)
    }

    for (r, args, seen) in fired() {
        let want: Vec<(u32, bool)> = r
            .args
            .iter()
            .zip(&args)
            .filter_map(|(ty, value)| handle_of(value).map(|h| (h, expects_write(*ty))))
            .collect();

        for (entrypoint, _, got) in seen.0.borrow().iter() {
            // A `Shaped` HANDLE IS A HANDLE, and it is the shape a bound
            // operand has now: `bind`'s `shaped` mints it so a mark can read
            // its own rectangle. Its writability is the MARK's -- `In` reads,
            // `Out`/`InOut` write -- which the direction half of `Ty` states
            // and the value no longer needs to. Matching only `Buffer` here
            // saw an empty list for every routine whose operands came through
            // the column, which is all of them.
            let got_handles: Vec<(u32, bool)> = got
                .iter()
                .filter_map(|v| match *v {
                    ArgValue::Buffer { handle, writes, .. } => Some((handle, writes)),
                    _ => None,
                })
                .collect();
            // THE TWO ROUTINES WHOSE SHADER ORDERS ITS BUFFERS DIFFERENTLY
            // FROM THE STATEMENT'S SLOTS.
            //
            // The rule below reads a signature's order as the fire's, and it
            // held while a mark could carry its slot in its type: HEAD wrote
            // `residual: InSlot<1, _>` BEFORE `half_in: InSlot<0, _>`, so the
            // declaration followed the shader's buffer table and the numbers
            // followed the statement. The four marks derive the slot from the
            // POSITION, so a signature cannot say both -- and the statement's
            // order is the one that decides what binds where.
            //
            // So on these two the declaration is `half_in` then `residual`
            // (inputs 0 and 1) and `qmm_t.slang` binds them the other way
            // round. Naming them is the same shape as `DIVERGED`: a rule with
            // an exception list is a rule, and a third routine arriving here
            // is a question rather than a waiver.
            const SHADER_ORDERS_ITS_OWN: &[&str] = &[
                "qmm_t_residual_fp16_precast",
                "qmm_t_strided_fp16_precast_residual",
            ];
            let ordered = !SHADER_ORDERS_ITS_OWN.contains(&r.name);
            let mut last: Option<usize> = None;
            for (handle, writable) in &want {
                // Not found at all: legitimately skipped, exactly as a
                // `Buffer` was always allowed to be.
                let Some(pos) = got_handles.iter().position(|(h, _)| h == handle) else {
                    continue;
                };
                if let Some(before) = last {
                    assert!(
                        pos > before || !ordered,
                        "`{}` fires `{entrypoint}` with its buffers out of \
                         order: handle {handle} lands at position {pos} \
                         among {got_handles:?}, not after the previous \
                         recipe buffer's position {before}",
                        r.name
                    );
                }
                // THE WRITABILITY CHECK STOOD HERE AND HAS NOTHING LEFT TO
                // READ. It compared the flag on the FIRED value against the
                // direction half of the signature's `Ty`, and a fired value
                // no longer carries one: a body re-emits its operand through
                // `Bind::arg`, which mints a plain handle, because writability
                // rides the MARK now -- `In` reads, `Out`/`InOut` write -- and
                // the mark is not in the value. This file's own header says
                // so twenty lines up.
                //
                // What survives is the ORDER check above, which is the half
                // that was ever load-bearing: a dropped operand binds the next
                // buffer one slot early, and that is what this test is for.
                let _ = writable;
                last = Some(pos);
            }
        }
    }
}

/// No routine ever asks for a zero-length grid.
///
/// `vkCmdDispatch(0, 1, 1)` is legal Vulkan that runs nothing and reports
/// success, over a buffer that kept whatever it held. This backend has paid
/// for that once: a shared expert's gate came back untouched and every routed
/// token was combined under `sigmoid(0)`.
#[test]
fn no_routine_dispatches_an_empty_grid() {
    for (r, _, seen) in fired() {
        for (entrypoint, lanes, _) in seen.0.borrow().iter() {
            assert!(
                !lanes.contains(&0),
                "`{}` fires `{entrypoint}` with lanes {lanes:?}",
                r.name
            );
        }
    }
}

/// Every routine names an entrypoint the shader tree actually declares.
///
/// A body composes its own entrypoint spelling, which is what lets a routine
/// serve an instantiation axis without a driver pasting suffixes on. The cost
/// is that a typo is a `&'static str` nobody checks -- and this tree has the
/// scar: `neox_freqs_mb` once named the DECODE symbol, a single-row kernel
/// over a multi-row grid, which rotated row zero, left every row after it
/// untouched and reported success. Rope is the identity at position 0, so even
/// the first row agreed.
#[test]
fn every_entrypoint_a_routine_names_exists() {
    let known: std::collections::BTreeSet<String> =
        kernels_vulkan::entrypoints().into_iter().collect();
    for (r, _, seen) in fired() {
        for (entrypoint, _, _) in seen.0.borrow().iter() {
            assert!(
                known.contains(entrypoint),
                "`{}` fires `{entrypoint}`, which no shader instantiates",
                r.name
            );
        }
    }
}

/// A routine name and a row name are the same hundred kernels, never a new one.
#[test]
fn no_routine_invents_a_kernel() {
    // A row OR a row this crate has retired: once a family crosses its rows
    // are deleted, and the retired list is the record of which hundred names
    // there ever were. Without it this test would forbid the deletion it was
    // written to survive.
    // `KERNELS.iter().map(|r| r.name).chain(..)` STOOD HERE. `KERNELS` is
    // empty -- every family crossed -- so the union is exactly the retired
    // list, and the `.chain` was joining a hundred names to nothing. The
    // retired list is the whole record of which hundred names there ever
    // were, which is what this test needs and all it ever needed once the
    // last family crossed.
    let rows: std::collections::BTreeSet<&str> =
        kernels_vulkan::retired_rows().iter().copied().collect();
    for r in kernels_vulkan::routines() {
        assert!(
            rows.contains(r.name),
            "`{}` is a routine with no row of the same name and none this \
             crate has retired. While a family is crossing, both planes hold \
             it; a routine that names something the table never did is a \
             kernel invented by the port.",
            r.name
        );
    }
}

/// Where this build put its modules, if it built any.
const SPV_DIR: Option<&str> = option_env!("PIE_KERNELS_VULKAN_SPV_DIR");

/// The binding numbers a compiled module actually decorates.
///
/// A twenty-line SPIR-V walk rather than a call into `driver-vulkan::spirv`,
/// because the thing being checked is an ABI and a check that shares its
/// reflector with the code under test can only find disagreements with itself.
/// `OpDecorate` is opcode 71 and `Binding` is decoration 33; both are fixed by
/// the specification and neither is going to drift.
fn decorated_bindings(spv: &[u8]) -> Vec<u32> {
    let words: Vec<u32> = spv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut out = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let op = words[at] & 0xffff;
        let len = (words[at] >> 16) as usize;
        assert!(len > 0, "an instruction of no words does not advance");
        if op == 71 && len >= 4 && words[at + 2] == 33 {
            out.push(words[at + 3]);
        }
        at += len;
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// A routine binds EXACTLY the buffers its module uses -- not the buffers its
/// shader source declares.
///
/// This is the check that a recorder cannot make and that
/// `every_entrypoint_a_routine_names_exists` does not: the recorder sees the
/// argument list a body passed and has no opinion about whether the module
/// wants that many. `driver-vulkan::encode::dispatch` does have one, and it is
/// `declared.bindings - declared.holes()` -- so a body that binds a buffer the
/// module dropped is `Refusal::Arity` at the first real dispatch, on a device,
/// long after the crossing looked green.
///
/// The reason a source declaration is not the answer is that slangc DELETES an
/// unread global. `attn/sdpa_paged.slang` declares `sinks` at binding 10
/// unconditionally, and without `PIE_WITH_SINK` nothing reads it, so
/// `sdpa_paged_decode_bfloat16_d_64` decorates 0..=9 and its sinked twin
/// decorates 0..=10. Two signatures, from one text, differing by a buffer.
///
/// It cuts the other way too, and that half is the more surprising one: a
/// module may keep a HOLE and still want fewer arguments than its highest
/// binding suggests. `kv_append_paged_bfloat16` decorates 0, 1, 2, 3, 10 and
/// 11 -- six buffers across a twelve-wide set -- so the six placeholders that
/// would be needed to push `w_page` to descriptor 10 are exactly the six
/// arguments that make the dispatch too long.
#[test]
fn every_routine_binds_the_buffers_its_module_uses_and_no_others() {
    let Some(dir) = SPV_DIR.map(std::path::Path::new) else {
        eprintln!("no modules: build with `--features native` and `slangc` on PATH");
        return;
    };

    let mut checked = 0usize;
    for (r, _, seen) in fired() {
        for (entrypoint, _, passed) in seen.0.borrow().iter() {
            // What the BODY handed the driver, not what its signature takes.
            // The two differ exactly when a body forgets an argument, which
            // is the failure this test exists to see, so counting the
            // signature would make it blind to half of its own subject.
            let bound = passed
                .iter()
                .filter(|a| matches!(a, ArgValue::Buffer { .. }))
                .count();
            let spv = std::fs::read(dir.join(format!("{entrypoint}.spv")))
                .unwrap_or_else(|e| panic!("`{}` fires `{entrypoint}.spv`: {e}", r.name));
            let used = decorated_bindings(&spv);
            assert_eq!(
                bound,
                used.len(),
                "`{}` binds {bound} buffers and `{entrypoint}` uses {} of them, \
                 at bindings {used:?}. The driver's arity is \
                 `declared.bindings - holes()`, so this is a refused dispatch \
                 and not a wasted descriptor.",
                r.name,
                used.len()
            );
            checked += 1;
        }
    }
    assert!(
        checked > 100,
        "only {checked} module comparisons: the sweep has stopped reaching them"
    );
}
