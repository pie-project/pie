//! THE METAL LAUNCHER SURFACE — and, in practice, the surface THREE
//! backends execute.
//!
//! The METAL launchers a lowered declaration may state — the `cuda`
//! of the second backend (`.wiki/tart/dsl.md` ②).
//!
//! # Who states this, and who runs it
//!
//! `model::shared::llama_like::forward::llama_like_metal` is the only text
//! written against it, and its plans are executed by `driver-metal`,
//! `driver-vulkan` AND `driver-wgpu` — the latter two have kernel tables
//! (`kernels_vulkan::KERNELS`, `kernels_wgpu::KERNELS`) that are
//! `kernels-metal`'s coverage row for row, so a plan naming these symbols
//! resolves in any of the three. That is why there is no `dsl::vulkan` and
//! no `dsl::wgpu`: a second copy of these 32 statements would be 32 more
//! places for the same symbol to be spelled differently, and the tables are
//! pinned equal precisely so that it never has to be.
//!
//! UNVERIFIED ON METAL (2026-08-05). Every symbol here is an MSL entrypoint
//! read off the driver's source (`crates/driver-metal/csrc/src/batch/decode_psos.cpp`'s
//! `PsoSpec` table and `model/qwen3_5/decode_step.hpp`'s `Kernel` kinds),
//! not something a running METAL deployment produced: the Metal driver
//! cannot build on the machine we have, because `xcrun --find metal` fails —
//! the shader compiler ships with full Xcode.
//! `.wiki/tart/macos.md` rung 3 is where that gets proven, by showing
//! `declared_dag.hpp`'s emitted descriptors come out unchanged.
//!
//! It is NOT unexercised, though, and the sentence here used to say
//! "nothing consumes this yet" — which stopped being true when the WebGPU
//! shell landed. `driver-wgpu`'s `checkpoint`, `serving` and `arena` suites
//! and `driver-vulkan`'s `checkpoint` suite all build their plans by calling
//! `llama_like_metal`, and `driver-wgpu` needs no adapter, no toolchain and
//! no Mac to do it. So the SHAPE of every statement below — its operands,
//! its widths, what it records — is under test on an ordinary Linux box;
//! what waits on a Mac is only whether Metal's own PSOs match.
//!
//! ONE DECISION worth stating, because it will look like an omission:
//! the quantized entrypoints are spelled by their BASE name
//! (`affine_qmv_fast`), not with the checkpoint's affine suffix
//! (`..._bfloat16_gs_64_b_4`, `AffineFormat::kernel_suffix()`). The
//! suffix is the driver's binding of a checkpoint fact, in the same
//! class as the stream and the workspace scratch — it selects no
//! different arithmetic and no different arm. What the text chooses is
//! the kernel FAMILY.

use super::*;

/// The tile the sort rounds each expert's run up to.
///
/// ONE for the MATVEC path: `qmv_routed` reads one row per thread block and
/// indexes the bank by that row's own expert, so grouping the rows buys
/// locality and nothing about a tile. At one, `moe_aligned_rows` is exactly
/// the route count and the sort is a pure permutation — no padding rows to
/// zero, no spare tiles, and `tile_expert` is one entry per row.
///
/// It was `QMM_TILE`-shaped (sixteen) by inheritance
/// from a tiled matmul this text did not launch, which padded a
/// four-token fire's sixteen routes out to two hundred and fifty-six
/// rows — sixteen times the matvec work, and every padded row read by
/// the expert projection.
///
/// It is no longer a constant because the blocked path arrived and it has to
/// be the GEMM's OWN row tile. `affine_qmm_t_routed` reads `tile_expert[t]`
/// once per tile and applies that expert's bank to all `bm` of the tile's
/// rows, so a run that does not fill a whole tile must be padded out or the
/// next expert's rows are computed against this expert's weights. That is not
/// a rounding error, it is the wrong matrix.
pub const ROUTE_BLOCK_MATVEC: u32 = 1;

fn record(
    t: &Trace,
    layer: Option<u32>,
    kernel: &str,
    weights: Vec<String>,
    state: Option<StateRef>,
    inputs: Vec<model_ir::trace::ValueId>,
    out: Option<(Shape, DType)>,
) -> Option<Val> {
    with_params(t, layer, kernel, weights, state, Vec::new(), inputs, out)
}

/// [`record`], plus the scalars the symbol's row names.
///
/// A kernel takes numbers no operand shape gives — a projection's two
/// extents, a norm's epsilon, an attention's strides. The row says which
/// slot wants which; this is where the statement supplies them, and the
/// order is the row's `Param(i)` order.
///
/// A float rides as its bits (`f32::to_bits`) and the row reads it back
/// with `ParamF32`: the channel is untyped `u32` and what each slot means
/// is the symbol's contract, which is the row.
#[allow(clippy::too_many_arguments)]
fn with_params(
    t: &Trace,
    layer: Option<u32>,
    kernel: &str,
    weights: Vec<String>,
    state: Option<StateRef>,
    params: Vec<u32>,
    inputs: Vec<model_ir::trace::ValueId>,
    out: Option<(Shape, DType)>,
) -> Option<Val> {
    let ids = t.with(layer, |b| {
        b.launch_with_params(
            kernel,
            weights,
            state,
            params,
            inputs,
            out.into_iter().collect(),
        )
    });
    ids.first().map(|&id| Val {
        t: t.clone(),
        id,
        layer,
    })
}

/// [`with_params`], plus the scalars whose value is an extent the FIRE
/// decides -- see [`model_ir::trace::OpKind::Launch::param_extents`].
#[allow(clippy::too_many_arguments)]
fn with_extents(
    t: &Trace,
    layer: Option<u32>,
    kernel: &str,
    weights: Vec<String>,
    state: Option<StateRef>,
    params: Vec<u32>,
    param_extents: Vec<(u8, Shape)>,
    inputs: Vec<model_ir::trace::ValueId>,
    out: Option<(Shape, DType)>,
) -> Option<Val> {
    let ids = t.with(layer, |b| {
        b.launch_with_extents(
            kernel,
            weights,
            state,
            params,
            param_extents,
            inputs,
            out.into_iter().collect(),
        )
    });
    ids.first().map(|&id| Val {
        t: t.clone(),
        id,
        layer,
    })
}

fn kv_state(kv: &Kv) -> Option<StateRef> {
    Some(StateRef {
        store: StateStore::KvCache,
        layer: kv.l,
    })
}

fn same_shape(v: &Val) -> (Shape, DType) {
    (v.t.inner.borrow().value_shape(v.id), DType::BF16)
}

/// The result a statement records — `None` inside a value-producing
/// region, where the enclosing construct owns it.
///
/// The same rule `seam::attn_at` follows and for the same reason: whether
/// a dispatch produces its own value is a property of the STATEMENT'S
/// POSITION, which the tape knows. A projection written plainly produces
/// its value; the same projection written as a guard's arm is a LOWERING
/// of the guard's.
///
/// Getting this wrong is not a small error. A guard arm that records its
/// own value leaves the guard's unwritten, so every statement after reads
/// the slot one before it — measured, when the projection guard first went
/// in: the KV pool came back holding q in its K pages and k in its V.
fn region_out(t: &Trace, shape: (Shape, DType)) -> Option<(Shape, DType)> {
    (!t.inner.borrow().inside_value_region()).then_some(shape)
}

/// The value a statement hands back.
///
/// `Some` outside a region, where the statement produced it. Inside one,
/// `with_params` recorded no output and there is nothing to hand back —
/// the caller has the GUARD's value and ignores this one — so the input is
/// returned as a placeholder rather than panicking on a `None` that is
/// correct.
fn or_regions(v: Option<Val>, x: &Val) -> Val {
    v.unwrap_or_else(|| x.clone())
}

/// `embed_gather.metal::embed_gather_4bit` (M=1) /
/// `embed_gather_mb_4bit` (M>1).
pub fn embed_gather(
    t: &Trace,
    weight: &str,
    hidden: u32,
    multi_batch: bool,
    repr: WeightRepr,
    point: &str,
) -> Val {
    // ALWAYS the M>1 symbol, and `multi_batch` is deliberately unread.
    //
    // `embed_gather_4bit` reads `id[0]` and writes `out[hidden]` — one row,
    // by construction, whatever grid it is handed. The class is not the
    // question: a DECODE of four requests is four rows, so a text that
    // picks by class names the single-row gather for a four-row fire and
    // three lanes get nothing. Measured against a real checkpoint: one of
    // four readout lanes held anything, and bisecting the fire put the
    // stop at statement ZERO.
    //
    // The mb variant's own comment says it "reduces to embed_gather_4bit
    // at N=1", so naming it unconditionally is not a widening — it is the
    // same kernel with the row read from the grid instead of assumed.
    let _ = multi_batch;
    let stem = "embed_gather_mb_4bit";
    with_params(
        t,
        None,
        &format!("{stem}{point}"),
        quant_table(weight, repr),
        None,
        // THE CODEC POINT, AS NUMBERS. `hidden` stood here alone and the
        // group and the bit width were `Ask<keys::QuantGroup>` and
        // `Ask<keys::QuantBits>` -- facts the driver recovered from the
        // SYMBOL. They are the checkpoint's constants, so the statement
        // carries them (`.wiki/migration.md` §3.2), and the routine reads
        // `hidden` off the result's own width rather than off this run.
        //
        // The pair is derived from the same `repr` and `point` the symbol
        // above is composed from, so the string and the numbers cannot
        // disagree: `point_of` is `affine_point`'s inverse over the two
        // fields that build it.
        point_of(repr, point),
        vec![],
        Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
    )
    .expect("embed produces the residual stream")
}

/// `rms_norm.metal::rms_single_row_bfloat16` — ONE entrypoint for
/// every norm this family states (attn_norm, mlp_norm, q_norm,
/// k_norm, final_norm; the driver fans five `Kernel` kinds onto it).
pub fn rms_norm(x: &Val, w: &NormW, row: u32, eps: f32) -> Val {
    rms_norm_gain(x, w, row, eps, 1.0)
}

/// The same, with a CONSTANT folded into the gain vector.
///
/// `rms.metal` already multiplies every channel by `p.gain`, and every caller
/// but one wants it at unity. gemma-4's ROUTER is the one: `Router.__call__`
/// is `rms_norm(x, self.scale * hidden**-0.5, eps)`, where the scalar is not
/// in the checkpoint and not in the weight -- the reference builds the
/// product at call time.
///
/// It is a GAIN and not a separate multiply because the shader has the slot,
/// and because 2816**-0.5 is 0.0188: dropping it scales the router's logits
/// by fifty-three, which leaves the top-k RANKING untouched and turns the
/// softmax over the chosen eight into very nearly one-hot. A model that
/// routes to the right experts and weights them wrong is fluent, which is
/// how it read as "close but the argmax moved".
pub fn rms_norm_gain(x: &Val, w: &NormW, row: u32, eps: f32, gain: f32) -> Val {
    let out = same_shape(x);
    with_params(
        &x.t,
        w.layer,
        "rms_single_row_bfloat16",
        vec![w.name.clone()],
        None,
        // `RmsParams`, field for field: eps, axis_size, w_stride,
        // plus_one, gain.
        //
        // `w_stride` is ONE, and the distance between this and the `row`
        // it used to say is the whole of a wrong answer. It is the stride
        // between consecutive CHANNELS of the gain vector -- `ws[w_stride
        // * i]` in the shader -- and a contiguous row's channels are one
        // apart. `rms.metal`'s own header says `w_stride=1`.
        //
        // Passing the axis made every norm read `w[2048 * i]`: it strode
        // out of the gain vector on the second channel and multiplied by
        // whatever followed it in the checkpoint. Measured against MLX at
        // position zero, channel 1 came out -0.016 where the reference
        // says +0.052 -- the wrong SIGN, from the wrong tensor, on the
        // second statement of the fire.
        //
        // `plus_one` is the `(1 + w)` reading gemma takes and this family
        // does not; the gain is unity.
        vec![
            eps.to_bits(),
            row,
            1,
            u32::from(w.variant == model_ir::trace::NormVariant::Gemma),
            gain.to_bits(),
        ],
        vec![x.id],
        Some(out),
    )
    .expect("a norm produces its value")
}

/// The tensors a quantized projection reads: the packed weight, then
/// its scales and zero point.
///
/// An affine kernel takes THREE buffers and the statements here used to
/// name one, which left the driver to derive the other two from a naming
/// convention it had to know. `dsl::matmul` already states the triplet
/// for the same reason its own doc gives — *"the driver never sees a
/// descriptor and never routes: it binds the names the statement gives it
/// and calls the symbol the statement names"* — and the Metal statements
/// now say the same thing.
/// The instantiation point an affine entrypoint is compiled at.
///
/// `quantized_qmv.metal` stamps one template over
/// `(activation dtype × group size × bit width)`, so the symbol a
/// statement names is `affine_qmv_fast_bfloat16_gs_64_b_4` and not the
/// stem. A stem does not resolve — which is the GOOD failure: the runtime
/// compiler reports it by listing what the shader does export, where a
/// WRONG point would compile and read the wrong bytes (the `_d_256`
/// defect, one axis over).
///
/// Both numbers come from the deployment's facts. Nothing here derives
/// them: g64/b8 and g128/b4 pack to identical shapes, so no tensor can be
/// asked.
/// Whether a projection this wide may name the tiled GEMM at all.
///
/// `quant/qmm_t.metal`'s header states the contract: *"the driver only
/// selects this kernel when `M % BM == 0`, `N % BN == 0` and `K % BK == 0`,
/// so every tile is full and the `load_unsafe` path is the only one
/// reachable"*. The row axis is a fire's, so a text guards it with
/// `GuardPred::TokensMultipleOf`; the COLUMN axis is the weight's, known
/// here, and nothing checked it.
///
/// qwen3.6 states `in_proj_a` and `in_proj_b` at 48 -- one scalar per value
/// head -- against a deployment tile of `bn = 32`. The second column tile
/// loaded sixteen rows past the end of the weight and the projection came
/// back NaN from row 32 of 64, differently on each fire. `qmm_grid` refuses
/// it now; this is what keeps a text from having to be refused.
#[must_use]
pub const fn gemm_fits(width: u32, tile: (u32, u32)) -> bool {
    tile.1 != 0 && width.is_multiple_of(tile.1)
}

/// The GEMM's instantiation point: [`affine_point`] plus its tile.
///
/// `affine_qmm_t` is stamped over `(group × bits × bm × bn)`, so its
/// symbol carries two more numbers than the GEMV's.
#[must_use]
pub fn affine_gemm_point(repr: WeightRepr, bits: u32, tile: (u32, u32)) -> String {
    let (bm, bn) = tile;
    format!("{}_bm_{bm}_bn_{bn}", affine_point(repr, bits))
}

/// The tile an affine GEMM point spells, as `[bm, bn]`.
///
/// [`point_of`]'s companion, and read back for its reason: a `qmm` routine
/// takes the tile as two `Const<i32>`s after its codec pair, and the caller
/// composed the spelling out of the same numbers.
#[must_use]
pub fn tile_of(point: &str) -> Vec<u32> {
    let num = |k: &str| {
        point
            .rsplit_once(k)
            .and_then(|(_, v)| v.split('_').next())
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0)
    };
    vec![num("_bm_"), num("_bn_")]
}

/// The params run an affine point's own numbers make: `[group, bits]`.
///
/// The pair every `qmm`/`qmv`/`embed_gather` routine reads as `Const<i32>`,
/// in the order their marks claim the run. It is derived from the same two
/// things [`affine_point`] composes its string out of -- the `repr`'s group
/// and the bit width -- so the entrypoint a statement names and the numbers
/// it carries are one decision written twice, not two.
///
/// The bit width is recovered from the SPELLING because that is where the
/// caller put it: `affine_point` is the only thing that writes `_b_{bits}`,
/// and every caller passes its output straight through. A point that does
/// not carry one yields zero, which the routine refuses.
#[must_use]
pub fn point_of(repr: WeightRepr, point: &str) -> Vec<u32> {
    let group = match repr {
        WeightRepr::Scaled { group, .. } => group,
        _ => 0,
    };
    let bits = point
        .rsplit_once("_b_")
        .and_then(|(_, b)| b.split('_').next())
        .and_then(|b| b.parse::<u32>().ok())
        .unwrap_or(0);
    vec![group, bits]
}

#[must_use]
pub fn affine_point(repr: WeightRepr, bits: u32) -> String {
    let group = match repr {
        WeightRepr::Scaled { group, .. } => group,
        _ => 0,
    };
    format!("_bfloat16_gs_{group}_b_{bits}")
}

/// A value's row width, from the shape the trace already carries.
///
/// A projection's INPUT extent, which no fact states and no operand
/// carries — the statement's own operand does, and this reads it. Zero for
/// a shape whose trailing dim is not a constant, which is a shape no
/// projection here has.
fn in_width(x: &Val) -> u32 {
    match x.t.inner.borrow().value_shape(x.id).0.last() {
        Some(Dim::Const(n)) => *n,
        _ => 0,
    }
}

/// A value's ROW axis, from the shape the trace already carries.
///
/// An elementwise statement's grid is `rows * width`, and until the routed
/// GEMM arrived every value in this text had `Dim::Tokens` for its rows, so
/// the three activations wrote that constant. The batched mixture breaks it:
/// its gate and up are the SORTED STACK, whose height is
/// [`Dim::MoeAlignedRoutes`] and is larger than the token count both because
/// each token has `k` routes and because each expert's run is padded to a
/// tile. An activation told `Dim::Tokens` there would cover a `k`th of its
/// own operands and leave the rest whatever the arena held.
///
/// Reading it off the operand rather than taking it as an argument keeps
/// every dense call site unchanged, which is the point: a dense gate IS
/// `[Tokens, _]` and answers the same thing it always did.
fn rows_of(v: &Val) -> Dim {
    v.t.inner
        .borrow()
        .value_shape(v.id)
        .0
        .first()
        .copied()
        .unwrap_or(Dim::Tokens)
}

fn quant_weights(w: &MatW) -> Vec<String> {
    let mut out = vec![w.name.clone()];
    out.extend(w.scale_names());
    out
}

/// The same triplet for a table the text names by STRING rather than
/// through a [`MatW`] handle — the embedding and the readout.
///
/// They take a `repr` for the same reason a projection does: the symbols
/// are `embed_gather_4bit` and `affine_qmv_fast`, both affine, both
/// reading three tensors.
fn quant_table(name: &str, repr: WeightRepr) -> Vec<String> {
    quant_weights(&MatW {
        name: name.to_string(),
        width: 0,
        layer: None,
        repr,
    })
}

/// `quantized_qmv.metal::affine_qmv_fast` — the projection GEMV,
/// M=1. The driver fans every projection kind onto it.
pub fn qmv(x: &Val, w: &MatW, point: &str) -> Val {
    let out = with_params(
        &x.t,
        w.layer,
        &format!("affine_qmv_fast{point}"),
        quant_weights(w),
        None,
        // THE CODEC POINT, AS NUMBERS. The two extents stood here — the row
        // the GEMV reads and the row it writes — and the routine reads both
        // off its own marks now (`x.width`, `y.width`), while the group and
        // the bit width it used to `Ask` for are the checkpoint's constants
        // the statement carries.
        point_of(w.repr, point),
        vec![x.id],
        region_out(
            &x.t,
            (Shape(vec![Dim::Tokens, Dim::Const(w.width)]), DType::BF16),
        ),
    );
    or_regions(out, x)
}

/// `quantized_qmv.metal::affine_qmv_fast_residual` — the same GEMV
/// with the block residual folded into its epilogue, which is what a
/// `beta_one` matmul is on this backend.
pub fn qmv_residual(x: &Val, w: &MatW, residual: &Val, point: &str) -> Val {
    let out = with_params(
        &x.t,
        w.layer,
        &format!("affine_qmv_fast_residual{point}"),
        quant_weights(w),
        None,
        point_of(w.repr, point),
        vec![x.id, residual.id],
        region_out(
            &x.t,
            (Shape(vec![Dim::Tokens, Dim::Const(w.width)]), DType::BF16),
        ),
    );
    or_regions(out, x)
}

/// `quant/qmm_t.metal::affine_qmm_t` — MLX's steel quantized
/// GEMM, the M>1 projection.
pub fn qmm(x: &Val, w: &MatW, point: &str) -> Val {
    let out = with_params(
        &x.t,
        w.layer,
        &format!("affine_qmm_t{point}"),
        quant_weights(w),
        None,
        // As `qmv`'s, plus the TILE: a `qmm` is stamped over
        // `(group × bits × bm × bn)` and takes all four as `Const<i32>`.
        [point_of(w.repr, point), tile_of(point)].concat(),
        vec![x.id],
        region_out(
            &x.t,
            (Shape(vec![Dim::Tokens, Dim::Const(w.width)]), DType::BF16),
        ),
    );
    or_regions(out, x)
}

/// `quant/qmm_t.metal::affine_qmm_t_residual`.
pub fn qmm_residual(x: &Val, w: &MatW, residual: &Val, point: &str) -> Val {
    let out = with_params(
        &x.t,
        w.layer,
        &format!("affine_qmm_t_residual{point}"),
        quant_weights(w),
        None,
        [point_of(w.repr, point), tile_of(point)].concat(),
        vec![x.id, residual.id],
        region_out(
            &x.t,
            (Shape(vec![Dim::Tokens, Dim::Const(w.width)]), DType::BF16),
        ),
    );
    or_regions(out, x)
}

/// `quant/qmm_t.metal::cast_qmm_input_strided_bfloat16_to_float16` — the
/// staging pass the precast GEMMs read.
///
/// # Why a separate statement and not a flag on the GEMM
///
/// The tile the multiply accumulates in is `half` and the checkpoint is
/// `bfloat`, so somebody has to convert. Converting inside the GEMM means
/// converting each `x` tile once per output tile — `N/BN` times, which is
/// 128 for a gate/up projection — and `qmm_t.metal` says so where it
/// explains why the ROUTED kernel does it the other way: *"the dense
/// projections take a different road to the same instruction — see
/// `qmm_t_fp16_precast_impl`, which stages the activations once in a
/// separate dispatch instead of converting each x tile N/BN times."*
///
/// # Why the strided form
///
/// The packed one takes a `count` and the count is `rows × k`, which is a
/// FIRE's number: a trace that stated it would be stating a row count it
/// cannot know. The strided form takes `k` and a row pitch and reads the
/// row count off the fire, which is the same shape every other multi-row
/// statement here has.
pub fn cast_qmm_input(x: &Val) -> Val {
    let k = in_width(x);
    let out = with_params(
        &x.t,
        x.layer,
        "cast_qmm_input_strided_bfloat16_to_float16",
        Vec::new(),
        None,
        // THE ROW PITCH, ALONE. `k` and the unread `n` slot stood before it
        // — the first is `cast_in.width`, which the operand carries, and the
        // second was never read. The pitch is `k` for a packed activation,
        // and every activation this is asked of is packed.
        vec![k],
        vec![x.id],
        region_out(&x.t, (Shape(vec![Dim::Tokens, Dim::Const(k)]), DType::F16)),
    );
    or_regions(out, x)
}

/// [`cast_qmm_input`], but recorded only when `pred` holds at fire time.
///
/// The staged half-precision activation exists for one reader,
/// `affine_qmm_t_fp16_precast`, which itself sits behind a guard because
/// `qmm_t` needs `M % BM == 0`. The cast could not go inside that guard's
/// arm -- an arm's launches bind the GUARD's output buffer and record no
/// value of their own, and the cast's value has to outlive the arm to be
/// bound as the GEMM's input -- so it was recorded unconditionally beside
/// it, and every fire whose row count misses the tile built a half-precision
/// copy nobody read.
///
/// At one token that costs nothing measurable, because the cast shares a
/// stage with the projection next to it. At a batch of eight it is 112 fires
/// and 0.50 ms a step, 6% of the step, and still dead -- `TokensMultipleOf`
/// refuses every batch a scheduler gathers, so the GEMM arm never runs. The
/// measurement is in `driver-vulkan`'s `hazards` doc.
///
/// So the cast gets a guard of ITS OWN, carrying the same predicate, whose
/// output is the staged activation. The `otherwise` region is empty on
/// purpose: when the predicate fails nothing reads this buffer, so nothing
/// needs to write it, and an empty region costs one skipped range in
/// `walk.rs` and no dispatch at all.
pub fn cast_qmm_input_when(x: &Val, pred: model_ir::trace::GuardPred) -> Val {
    let k = in_width(x);
    let (g, v) = crate::guard::guarded_value(
        &x.t,
        x.layer,
        (Shape(vec![Dim::Tokens, Dim::Const(k)]), DType::F16),
    );
    g.arm(pred, || {
        cast_qmm_input(x);
    })
    .otherwise(|| {});
    v
}

/// `quant/qmm_t.metal::affine_qmm_t_fp16_precast` — [`qmm`] over an
/// activation [`cast_qmm_input`] already staged to `half`.
///
/// The weights do not move: they stay affine-quantized with `bfloat`
/// scales, and the loader decodes them into a `half` threadgroup tile.
/// What changes is the matrix instruction. An Apple GPU below family 9 has
/// no `bfloat` matrix unit, so `simdgroup_matrix<bfloat>` is emulated and
/// `simdgroup_matrix<half>` is not — which is the whole of the win and the
/// reason the kernel's own header calls itself *"the same loop, staged to
/// HALF when the device has no bfloat matrix unit."*
///
/// `x` is the statement's first input and the RESULT comes before it in the
/// argument table, which is the one way this form's binding differs from
/// [`qmm`]'s; see `driver-metal`'s `arm::precast`.
pub fn qmm_fp16(x: &Val, w: &MatW, point: &str) -> Val {
    let out = with_params(
        &x.t,
        w.layer,
        &format!("affine_qmm_t_fp16_precast{point}"),
        quant_weights(w),
        None,
        // The precast pair takes the TILE alone: its operands are already
        // half-precision, so no codec point is read at the launch.
        tile_of(point),
        vec![x.id],
        region_out(
            &x.t,
            (Shape(vec![Dim::Tokens, Dim::Const(w.width)]), DType::BF16),
        ),
    );
    or_regions(out, x)
}

/// `quant/qmm_t.metal::affine_qmm_t_residual_fp16_precast`.
pub fn qmm_residual_fp16(x: &Val, w: &MatW, residual: &Val, point: &str) -> Val {
    let out = with_params(
        &x.t,
        w.layer,
        &format!("affine_qmm_t_residual_fp16_precast{point}"),
        quant_weights(w),
        None,
        tile_of(point),
        vec![x.id, residual.id],
        region_out(
            &x.t,
            (Shape(vec![Dim::Tokens, Dim::Const(w.width)]), DType::BF16),
        ),
    );
    or_regions(out, x)
}

/// `norm/add_bias.metal::add_bias_bfloat16` — the Qwen-2 family's q/k/v
/// projection biases, in place over the projection.
///
/// One statement, and the value it produces is the value it was given:
/// the row declares `in_place = &[(0, 0)]`, so a driver binds one
/// allocation for both. The trace still names a result, the way
/// [`residual_add`] does, because a tape whose statements did not produce
/// values could not say what the next one reads.
///
/// The width the kernel needs is the OUTPUT's row width, which the row
/// reads with `Source::Slot(Kind::OutWidth, 0)` rather than taking as a stated scalar
/// — a bias vector's length is the projection's width, and the trace
/// already said that when it sized the value.
///
/// [`residual_add`]: metal::residual_add
pub fn add_bias(x: &Val, w: &MatW) -> Val {
    let shape = same_shape(x);
    let out = record(
        &x.t,
        w.layer,
        "add_bias_bfloat16",
        vec![w.name.clone()],
        None,
        vec![x.id],
        region_out(&x.t, shape),
    );
    or_regions(out, x)
}

/// `residual_add.metal::residual_add_bfloat16` — the explicit
/// landing, for the deployments and positions where no epilogue fold
/// exists.
pub fn residual_add(x: &Val, residual: &Val) -> Val {
    let shape = same_shape(x);
    let out = record(
        &x.t,
        x.layer,
        "residual_add_bfloat16",
        vec![],
        None,
        vec![x.id, residual.id],
        region_out(&x.t, shape),
    );
    or_regions(out, x)
}

/// `rope/rope.metal::neox_decode_bfloat16` (M=1) /
/// `neox_mb_bfloat16` (M>1). One dispatch for q and k together,
/// as the plan states it (`declared_dag.hpp`'s `Kind::Rope`).
///
/// `proportional` says which LADDER, and it is a third form beside the two
/// this used to state. A geometric rotation pairs channel `i` with
/// `i + rotary/2` and takes its exponent over the rotated slice; gemma-4's
/// pairs `i` with `i + head_dim/2` and takes the exponent over the WHOLE
/// head. Those are the same rotation when the rotary covers the head and a
/// different one when it does not -- gemma-4's full-attention layers rotate
/// 128 of 512, so the channels that move are `[0,63]` and `[256,319]` and
/// not `[0,127]`, at frequencies four ladder steps apart from the ones the
/// geometric form computes.
pub fn rope(
    q: &Val,
    k: &Val,
    multi_batch: bool,
    theta: f32,
    scale: f32,
    head_dim: u32,
    rotary_dim: u32,
    table: bool,
    proportional: bool,
) -> (Val, Val) {
    (
        rope_one(
            q,
            multi_batch,
            theta,
            scale,
            head_dim,
            rotary_dim,
            table,
            proportional,
        ),
        rope_one(
            k,
            multi_batch,
            theta,
            scale,
            head_dim,
            rotary_dim,
            table,
            proportional,
        ),
    )
}

/// One tensor's rotation — which is what the kernel does.
///
/// `rope_neox_decode` takes ONE `device T* x` and rotates it in place.
/// This helper used to state a single launch carrying q and k, two inputs
/// and two results, on the strength of a comment saying the DAG spells it
/// as one `Kind::Rope`. The DAG spells one KIND and dispatches it twice;
/// the trace stated one LAUNCH, so the second tensor was never rotated.
///
/// Nothing could see it until the rows carried their operands: a statement
/// whose shape disagrees with its kernel's is invisible to every check
/// that only asks whether the symbol exists.
///
/// In place, so the result and the operand are one buffer: the row states
/// `x` as its input AND declares a result of the same shape, and
/// `in_place = &[(0, 0)]` on the routine is what says they are the same
/// allocation.
fn rope_one(
    x: &Val,
    multi_batch: bool,
    theta: f32,
    scale: f32,
    head_dim: u32,
    rotary_dim: u32,
    table: bool,
    proportional: bool,
) -> Val {
    // IN PLACE, and the statement says so the way every other in-place
    // statement in the tree does: it declares BOTH halves and the routine
    // pairs them. Every `neox` entrypoint takes ONE buffer -- `device T* x`,
    // read and written -- so the input and the result are one allocation,
    // and `kernels-metal`'s `routine!(neox_*, in_place = &[(0, 0)])` is what
    // says that.
    //
    // It used to declare the input and NO result, on a comment claiming
    // that "stating no result makes `dispatch::reorder` bind `Out(0)` to
    // the last widthed operand, which for a one-operand launch is the
    // input". `reorder` does not exist on any backend any more -- the row
    // path that needed it was retired -- and `Handles::output` reads
    // declared results flat, so `o.output(0)` in `driver-metal`'s `neox`
    // arm had no result to find and the rotation refused outright.
    // `arity_problem` said both halves of it: the routine writes one
    // pointer against a statement declaring none, and reads none against a
    // statement placing one.
    //
    // CUDA's `rope_bf16` is the same kernel shape spelled correctly and is
    // the model here: two `Out`s, no `In`s, `in_place = &[(0, 0), (1, 1)]`,
    // and a statement declaring two inputs and two results. The value
    // handed back is the RESULT's, which after the launch is the same
    // buffer the input named -- that is what the pair means.
    //
    // A deployment that RESCALES its frequency ladder cannot state a base:
    // llama-3 rescales piecewise and YaRN rescales differently, and both
    // are tables. The driver derives one at load and answers it as
    // `Source::Named(<keys::RopeFrequencies as keys::Fact>::KEY)`, so the statement's job is only to say
    // WHICH form this deployment takes.
    // Deliberately unread, like `embed_gather`'s: both stems below name
    // the M>1 form unconditionally.
    let _ = multi_batch;
    let (kernel, params) = if table {
        // The batched form exists and is the same rotation over N rows.
        // This branch used to name the decode symbol whatever the fire
        // was, while the branch below already chose — so a rescaled-ladder
        // PREFILL dispatched a single-row kernel over a multi-row grid.
        // `pos.z` is not delivered to a `uint2` thread position, so every
        // row computed row zero's index and rows one and up were never
        // rotated at all. Position zero makes rope the identity, so row
        // zero looked right and nothing said which row was wrong.
        // ALWAYS the M>1 symbol. Choosing by class fixed the PREFILL
        // and left the DECODE, because it read the class as though it
        // answered "how many rows", and it does not: a decode of four
        // requests is FOUR ROWS. `neox_freqs_decode` reads `position[0]`
        // whatever grid it is handed, so a four-lane decode rotated lane
        // zero and left three lanes unrotated -- and position zero makes
        // rope the identity, so the one lane every single-request gate
        // looks at agreed exactly.
        //
        // The mb form is the same rotation with the row read from the
        // grid instead of assumed; its operands, its `LaunchRule::Rope`
        // and its `head_param` are identical, so naming it at N=1 is not
        // a widening.
        let stem = "neox_freqs_mb_bfloat16";
        (
            stem.to_string(),
            // Scale, head width, and YaRN's `mscale` -- one for llama-3,
            // whose rescaling lives entirely in the frequencies.
            // The rotary WIDTH last, and the row says so with
            // `grid_param`. The kernel does not read it -- its operand
            // list stops before it -- but the DRIVER does, because
            // `Rule::Rope`'s grid is half this number and gemma-4 states
            // it per layer type. A statement that carries it is the
            // alternative to a fire-wide `rotary_dims` that cannot be two
            // things at once.
            vec![scale.to_bits(), head_dim, 1.0f32.to_bits(), rotary_dim],
        )
    } else {
        // ALWAYS the M>1 symbol, for the reason the table branch above
        // states: the class does not answer how many rows a fire has.
        //
        // WHICH LADDER is the other question, and for two years this text
        // could only ask one of them. `neox_prop_mb_bfloat16` has existed in
        // `rope/neox.metal` the whole time, `kernels-metal::rope` states its
        // binding order, `driver-metal`'s routine table names it and
        // `lowering::dispatch` has a test asserting it plans -- and this
        // file, the only thing that can put it in a trace, named it ZERO
        // times. So every gemma-4 rotated its full-attention heads on the
        // geometric ladder over a quarter-head: the pair was `(i, i+64)`
        // where the checkpoint means `(i, i+256)`, and the exponent divided
        // by 64 where it means 512. Nothing refused, nothing NaN'd, and the
        // model answered a prompt about the capital of France with
        // `-p--r-r-c-c--f--ter---`.
        //
        // The same shape of gap as the routed MoE GEMM and the GDN prefill
        // pair: a kernel with every seam behind it built and no text naming
        // it, on a driver no CI machine compiles.
        let stem = if proportional {
            "neox_prop_mb_bfloat16"
        } else {
            "neox_mb_bfloat16"
        };
        (
            stem.to_string(),
            // The rotation's scale, its log2 base and the head width. The
            // base is `log2(theta)` because the shader raises two to it --
            // `rope_neox_geometric_body` -- and handing it theta rotates
            // by a frequency ladder wrong from the second channel on.
            // The rotary WIDTH last -- see the table form above.
            //
            // The proportional form takes the same four in the same order:
            // it reads the head width where the geometric one reads the
            // grid, so only the arithmetic differs and not the binding.
            vec![
                scale.to_bits(),
                theta.log2().to_bits(),
                head_dim,
                rotary_dim,
            ],
        )
    };
    let shape = same_shape(x);
    let out = with_params(
        x.trace(),
        x.layer(),
        &kernel,
        vec![],
        None,
        params,
        vec![x.id],
        region_out(x.trace(), shape),
    );
    or_regions(out, x)
}

/// `norm/rms_rope.slang::rms_rope_bfloat16` — the per-head q/k norm and the
/// NEOX rotation that always follows it, in one dispatch.
///
/// # This statement replaces TWO
///
/// A caller that states this must not also state [`rms_norm`] or [`rope`] for
/// the same tensor. It is the pair, not an addition to it, and the value it
/// hands back is already normed and already rotated.
///
/// # In place, and that is a constraint on the caller
///
/// The rotation is in place, as every rotation on this backend is, so the
/// norm has to be too -- the fused kernel reads what it just wrote. The
/// separate norm is OUT of place, and one caller depends on that: a k-eq-v
/// layer takes V from the K projection and relies on `k_norm` producing a new
/// buffer, so that V still names the raw projection. Fusing there would norm
/// and rotate V as a side effect of doing it to K, silently, and the caller's
/// gate is what keeps that from happening.
///
/// # Nine params where every other norm states five
///
/// `RmsRopeParams` is `RmsParams` with `row_pitch`, `rotary`, `scale` and the
/// rope base appended, and `driver-vulkan` mints the block from the whole
/// stated run. The four extra are the rotation's, and `rotary` in particular
/// cannot be recovered from the rectangle: gemma-4 rotates a quarter of each
/// full-attention head over the same tensor width.
///
/// The base is `log2(theta)` because the shader raises two to it, which is
/// the same conversion [`rope`] makes and for the same reason -- handing it
/// theta rotates by a frequency ladder wrong from the second channel on.
///
/// There is no Metal kernel behind this symbol. The name resolves through
/// `kernels-metal`'s census so that `model-ir` can check a Vulkan text, and
/// the caller's gate is defaulted off so no Metal text can name it.
#[allow(clippy::too_many_arguments)]
pub fn rms_rope(
    x: &Val,
    w: &NormW,
    head_dim: u32,
    eps: f32,
    theta: f32,
    scale: f32,
    rotary_dim: u32,
) -> Val {
    let shape = same_shape(x);
    let out = with_params(
        &x.t,
        w.layer,
        "rms_rope_bfloat16",
        vec![w.name.clone()],
        None,
        // `RmsRopeParams`, field for field. The first five are `RmsParams`
        // and carry the same readings [`rms_norm_gain`] explains at length:
        // `w_stride` is ONE because it is the distance between consecutive
        // CHANNELS of the gain vector and a contiguous row's are one apart,
        // `plus_one` is gemma's `(1 + w)` reading which this family does not
        // take, and the gain is unity.
        vec![
            eps.to_bits(),
            head_dim,
            1,
            0,
            1.0f32.to_bits(),
            // The distance between two tokens' rows, which is the whole
            // projection and not the head: the base the kernel builds is
            // two-level, `row * row_pitch + head * axis_size`, and the head
            // count it launches is these two divided.
            in_width(x),
            rotary_dim,
            scale.to_bits(),
            theta.log2().to_bits(),
        ],
        vec![x.id],
        region_out(&x.t, shape),
    );
    or_regions(out, x)
}

/// `attn/split_qkv.metal::split_qkv_bf16`: deinterleave the packed QKV
/// projection `[rows, q_width + 2*kv_width]` into three buffers.
///
/// # Why this exists beside `dsl::split_qkv`
///
/// The generic `split_qkv` records an `OpKind::SplitQkv`, which carries
/// the two widths *in the op kind*. A driver could read them — by
/// matching on `OpKind`, which is exactly what "nothing in the driver may
/// choose a kernel" forbids: the widths would reach the kernel because the
/// driver knew what a QKV split is.
///
/// So the Metal text states the launch outright, and the widths ride the
/// channel built for them — [`OpKind::Launch::params`], whose own doc says
/// *"a scalar that has nowhere to ride is a scalar the DRIVER re-derives
/// from its config. That is the thing this arc removes."* The driver then
/// forwards `params` to every kernel that states them, knowing nothing
/// about what they mean.
///
/// [`OpKind::Launch::params`]: model_ir::trace::OpKind::Launch
pub fn split_qkv(packed: &Val, q_width: u32, kv_width: u32) -> (Val, Val, Val) {
    let rows = packed.t.inner.borrow().value_shape(packed.id).0[0];
    let out = |w: u32| (Shape(vec![rows, Dim::Const(w)]), DType::BF16);
    let ids = packed.t.with(packed.layer, |b| {
        b.launch_with_params(
            "split_qkv_bf16",
            vec![],
            None,
            vec![q_width, kv_width],
            vec![packed.id],
            vec![out(q_width), out(kv_width), out(kv_width)],
        )
    });
    let mk = |id| Val {
        t: packed.t.clone(),
        id,
        layer: packed.layer,
    };
    (mk(ids[0]), mk(ids[1]), mk(ids[2]))
}

/// `kv_append.metal::kv_append_bfloat16` (contiguous) /
/// `kv_append_paged.metal::kv_append_paged_bfloat16` (page table).
pub fn kv_append(k: &Val, v: &Val, kv: &Kv, paged: bool, head_dim: u32, kv_heads: u32) {
    let kernel = if paged {
        "kv_append_paged_bfloat16"
    } else {
        "kv_append_bfloat16"
    };
    with_params(
        &kv.t,
        Some(kv.l),
        kernel,
        vec![],
        kv_state(kv),
        // The model's two: how wide a head is and how many there are. The
        // pool's strides come from the ROW (`KvHeadStride`, `KvSeqStride`)
        // because they are the shape the driver allocated, not the shape
        // the model has.
        vec![head_dim, kv_heads],
        vec![k.id, v.id],
        None,
    );
}

/// `sdpa_vector.metal::sdpa_vector_decode_bfloat16_d_<head_dim>` (M=1) /
/// `sdpa_paged.metal::sdpa_paged_decode_bfloat16_d_<head_dim>` (M>1).
///
/// The width is the deployment's, not a literal. It used to be `_d_256`
/// unconditionally, which is wrong for every checkpoint whose heads are
/// narrower — `qwen3_0_6b`'s are 128 — and wrong in the way that does not
/// fault: a 256-wide kernel over 128-wide heads reads past the end of
/// every head and answers with whatever is there. `.wiki/driver/progress-metal.md`
/// records the same defect in the C++ llama walk, where `_d128` was a
/// literal that strode 64-wide heads past their end.
///
/// Both kernels instantiate `_d_64`, `_d_128` and `_d_256`; the paged one
/// also `_d_512`. A width neither carries has no kernel, and the symbol
/// this returns will simply not resolve — which the driver's
/// `every_symbol_the_lowering_names_has_a_row` check reports by name.
#[allow(clippy::too_many_arguments)]
pub fn sdpa(
    q: &Val,
    kv: &Kv,
    q_width: u32,
    head_dim: u32,
    paged: bool,
    gqa_factor: u32,
    kv_heads: u32,
    window: i32,
    sinks: Option<&str>,
    scale: f32,
    multi_batch: bool,
) -> Option<Val> {
    // The SINK variant is the same template at `sinks = true`, so it is
    // the same statement with one weight. A sink is a per-head learned
    // logit that joins the softmax without a value behind it — gpt-oss's,
    // and the reason `sdpa_paged_decode`'s row has carried an open slot
    // since the rows were written.
    //
    // `multi_batch` is the other axis, and it is a DIFFERENT KERNEL
    // rather than a parameter. The decode kernel gives one threadgroup
    // per (head, query row), and each of those re-reads the whole key run
    // — which is right at one row and quadratic above it. The shader's
    // own header measures the cost on the 30B checkpoint: fitting
    // `a + b*n` puts the quadratic term at 39% of prefill time at
    // n = 2048, about 527 GB/s against some 5% of fp16 peak. Bandwidth,
    // not arithmetic. The tiled kernel stages a run of keys in
    // threadgroup memory and lets 32 query rows share it, so the run is
    // read once instead of 32 times.
    //
    // Both tiled points exist wherever the decode point this replaces
    // does — `_d_{64,128,256,512}` plain and `_d_64` with sinks, which is
    // exactly the decode kernels' own reach — so that branch is total and
    // needs no fallback. A head dim the tiled kernel lacked would need
    // one, and `every_text_names_a_symbol_this_build_compiles` is what
    // would say so.
    //
    // AND A THIRD KERNEL AT ONE WIDTH. `sdpa_paged_mma` tiles the same 32
    // rows, but a simdgroup owns eight of them and multiplies 8x8
    // fragments on the matrix unit instead of owning one row and adding
    // scalars. Q Kᵀ and P V were always matmuls and the tiled kernel does
    // not issue them as one. It exists only at `_d_64`, so this is a
    // preference and not a replacement: the tiled form still serves every
    // other width, and the match falls through to it.
    //
    // Measured on this driver's own gate,
    // `attention_is_a_minority_of_a_long_prefill`, which reads the
    // quadratic coefficient out of three prefill timings (ms per token²):
    //
    //                       decode      tiled        mma
    //   Llama-3.2-1B      2.788e-4   1.000e-4   2.589e-5
    //   gpt-oss-20b       4.215e-4   1.654e-4   4.883e-5
    //
    // 3.9x under the tiled form on the dense checkpoint and 3.4x on the
    // mixture, which is the only measurement the SINK variant has. The
    // whole 2048-token fire falls from 1684 ms to 1378 ms on the 1B --
    // 1.80x against the row-by-row kernel it started at. Both variants
    // reproduce MLX exactly on both checkpoints, 24 comparisons.
    //
    // The cost of preferring it: `sdpa_paged_tiled_sink` compiles at
    // `_d_64` alone, gpt-oss is the only family with sinks and its heads
    // are 64 wide, so nothing reaches that symbol any more and the dark
    // ledger in `text_conformance.rs` says so with this measurement as
    // the reason.
    let kernel = match (paged, sinks.is_some(), multi_batch, head_dim) {
        (true, true, false, _) => format!("sdpa_paged_decode_sink_bfloat16_d_{head_dim}"),
        (true, false, false, _) => format!("sdpa_paged_decode_bfloat16_d_{head_dim}"),
        (true, true, true, 64) => "sdpa_paged_mma_sink_bfloat16_d_64".to_string(),
        (true, false, true, 64) => "sdpa_paged_mma_bfloat16_d_64".to_string(),
        (true, true, true, _) => format!("sdpa_paged_tiled_sink_bfloat16_d_{head_dim}"),
        (true, false, true, _) => format!("sdpa_paged_tiled_bfloat16_d_{head_dim}"),
        (false, _, _, _) => format!("sdpa_vector_decode_bfloat16_d_{head_dim}"),
    };
    let kernel = kernel.as_str();
    // The model's scalars, in the order both rows name them. The strides
    // and the page size are the POOL's and come from the row; the mask
    // stride is zero because this text states no custom mask.
    //
    // The softmax temperature, and the one number here a reader is most
    // likely to assume the kernel knows. It does not: it takes it, and a
    // zero makes every logit zero and every attention uniform.
    //
    // DERIVED ONLY AS A DEFAULT. `1/sqrt(head_dim)` is llama's rule, not
    // attention's: gemma-3 states `query_pre_attn_scalar` and gemma-4
    // states **1.0**, because its per-head `q_norm`/`k_norm` have already
    // divided by the thing this would divide by again. A statement that
    // derives it cannot serve a family that states it, and the derivation
    // fails SILENTLY -- attention stays a probability distribution at any
    // temperature, so the fire is finite, varied, and wrong.
    let scale = if scale > 0.0 {
        scale
    } else {
        1.0f32 / (head_dim as f32).sqrt()
    };
    with_params(
        &q.t,
        Some(kv.l),
        kernel,
        // The sink weight, when this deployment has one: a per-head
        // learned logit, and the row's `Weight(0)`.
        sinks.map(|w| vec![w.to_string()]).unwrap_or_default(),
        kv_state(kv),
        // THE RUN THE ROUTINES DECLARE, in the order their `Const` marks
        // claim it: `[n_kv_heads, scale, window, head_dim, q_heads]`.
        //
        // It was `[gqa_factor, kv_heads, scale, 0, window]` -- a run with a
        // HOLE at index 3, and a `gqa_factor` the bodies now derive from the
        // two head counts rather than being told a third time. The extents
        // moved in the other direction: `head_dim` and `q_heads` were facts
        // the driver recovered from the SYMBOL (`_d_128`) and they are the
        // checkpoint's, so the statement carries them.
        //
        // `q_heads` is `q_width / head_dim` and not a separate number: the
        // query's row is heads laid end to end, which is the same division
        // the symbol's `_d_` suffix already implies.
        vec![
            kv_heads,
            scale.to_bits(),
            window as u32,
            head_dim,
            if head_dim > 0 { q_width / head_dim } else { 0 },
        ],
        vec![q.id],
        // `region_out` and not `Some(..)`: a batched DECODE guards this
        // statement (`GuardPred::WindowOne`) and an arm's launches bind the
        // guard's output rather than recording one of their own. Outside a
        // region this is the same `Some` it always was.
        region_out(&q.t, (Shape(vec![Dim::Tokens, Dim::Const(q_width)]), DType::BF16)),
    )
}

/// `silu_mul.metal::silu_mul_bfloat16` — the SwiGLU activation over
/// the packed gate/up bank.
pub fn silu_mul(gate: &Val, up: &Val, intermediate: u32) -> Val {
    record(
        &gate.t,
        gate.layer,
        "silu_mul_bfloat16",
        vec![],
        None,
        vec![gate.id, up.id],
        Some((
            Shape(vec![rows_of(gate), Dim::Const(intermediate)]),
            DType::BF16,
        )),
    )
    .expect("the activation produces its value")
}

// ── gemma's per-layer embeddings. ──
//
// A SIDE NETWORK, and the only thing in this module that is: a second
// embedding table gathered once per step, projected, normed and joined,
// producing `[n_layers, ple_dim]` that each layer then reads its own slice
// of. Nothing llama-like has a counterpart, which is what makes gemma4 a
// family where qwen3-moe and gpt-oss were facts.
//
// Four statements before the stack and four inside it, and every symbol is
// one this backend already had — `psos_gemma4.rs` maps six of the nine
// `G4Ple*` roles onto kernels other families name. What is new is the
// WALK.

/// `layout/embed_gather.metal::embed_gather_scaled_4bit` — the embedding,
/// with gemma's `sqrt(hidden)` scale folded into the gather.
///
/// The scale is the STATEMENT's, not the kernel's: a kernel that knew it
/// would be a kernel that knew the model.
pub fn embed_gather_scaled(
    t: &Trace,
    weight: &str,
    width: u32,
    multi_batch: bool,
    repr: WeightRepr,
    point: &str,
    scale: f32,
) -> Val {
    // ALWAYS the M>1 symbol, exactly as `embed_gather` above -- and this
    // is the twin that fix did not reach. `embed_gather_scaled_4bit`
    // reads `id[0]` and writes `out[hidden]`, one row by construction,
    // so a four-lane decode gathered lane zero and left three lanes
    // holding a zeroed arena. Bisecting gemma-4-31b's decode put the
    // stop at statement ZERO and every later statement inherited it --
    // which is what made a geglu twenty statements downstream look like
    // the defect.
    //
    // The scale is the only difference between these two and their
    // unscaled twins, and it is the statement's either way.
    let _ = multi_batch;
    let stem = "embed_gather_scaled_mb_4bit";
    with_params(
        t,
        None,
        &format!("{stem}{point}"),
        quant_table(weight, repr),
        None,
        // THE RUN THE ROUTINE DECLARES, in the order its three `Const` marks
        // claim it: `[embed_scale, group, bits]`. It was `[width, scale]` --
        // a `width` at the slot the scale is read from and a scale at the
        // slot the group is, so the gather scaled by a group size and picked
        // its point from a bit count that was really a float's bits.
        //
        // `width` is not a param at all: the body reads `out.width`, which
        // the statement already gives as the rectangle below.
        {
            let mut p = vec![scale.to_bits()];
            p.extend(point_of(repr, point));
            p
        },
        vec![],
        Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
    )
    .expect("the gather produces its rows")
}

/// `layout/ple_combine.metal::ple_combine` — `(proj + token) * inv_sqrt2`.
///
/// The scale is the JOIN's rather than a deployment's: two streams
/// averaged in the root-mean-square sense, which is what `1/sqrt(2)` is.
pub fn ple_combine(proj: &Val, token: &Val, width: u32) -> Val {
    with_params(
        &proj.t,
        None,
        "ple_combine_bfloat16",
        vec![],
        None,
        // `PleCombineParams`: inv_sqrt2 then n.
        vec![std::f32::consts::FRAC_1_SQRT_2.to_bits(), width],
        vec![proj.id, token.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
    )
    .expect("the join produces its value")
}

/// `mlp/gated.metal::geglu_tanh_strided` — the activation over rows that
/// are not contiguous.
///
/// gemma's PLE reads a narrow gate out of a wide buffer, so each operand
/// states its own pitch. The plain `geglu` is this with all three equal.
/// THE SECOND WORD IS DEAD AND HAS TO STAY WRITTEN. It was a row count,
/// stated here as the literal `1` under a comment saying the real count "is
/// the fire's and rides the shape" — which nothing fills, so any body
/// bounding itself with it clamped to a single row. `kernels-metal`'s body
/// now recovers its row by dividing the flat thread id by `width`, on the
/// same reasoning `GegluParams` already took: the grid is the extent.
/// `kernels-vulkan` and `kernels-wgpu` flattened their bodies too but still
/// guard on `p.rows * p.width`, so they still see one row; the word keeps its
/// slot and its value of `1` until they stop reading it, because a `0` here
/// would make their guard reject every thread instead of all but the first
/// row's.
pub fn geglu_strided(gate: &Val, up: &Val, width: u32, gate_pitch: u32, up_pitch: u32) -> Val {
    with_params(
        &gate.t,
        gate.layer,
        "geglu_tanh_strided_bfloat16",
        vec![],
        None,
        // `GegluStridedParams`: width, the dead word, then the pitches.
        vec![width, 1, gate_pitch, up_pitch, width],
        vec![gate.id, up.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
    )
    .expect("the activation produces its value")
}

/// `norm/vector.metal::vnorm_single_row` — a norm with NO gain.
///
/// The row divided by its own RMS and nothing else. The absence of a
/// weight is the whole difference from [`rms_norm`], and it is why this is
/// its own symbol rather than a norm handed a vector of ones — which would
/// be a multiply per element to compute the identity.
pub fn vnorm(x: &Val, row: u32, eps: f32) -> Val {
    with_params(
        &x.t,
        x.layer,
        "vnorm_single_row_bfloat16",
        vec![],
        None,
        // `VNormParams`: eps then axis_size.
        vec![eps.to_bits(), row],
        vec![x.id],
        Some(same_shape(x)),
    )
    .expect("the norm produces its value")
}

/// `norm/layer_scalar.metal::layer_scalar_mul` — one number per layer.
///
/// Read from a BUFFER rather than stated, because which layer is running
/// is the fire's and not the text's.
pub fn layer_scalar(x: &Val, scalar: &str, width: u32) -> Val {
    with_params(
        &x.t,
        x.layer,
        "layer_scalar_mul_bfloat16",
        vec![scalar.to_string()],
        None,
        // `LayerScalarParams`: the hidden width.
        vec![width],
        vec![x.id],
        Some(same_shape(x)),
    )
    .expect("the scale produces its value")
}

/// `norm/rms.metal::rms_residual` — a norm with the block residual folded
/// into its epilogue, and `rms_residual_scaled` with a per-layer gain
/// beside it.
pub fn rms_norm_residual(
    x: &Val,
    w: &NormW,
    residual: &Val,
    scale: Option<&Val>,
    row: u32,
    eps: f32,
) -> Val {
    let mut ins = vec![x.id, residual.id];
    let kernel = match scale {
        Some(s) => {
            ins.push(s.id);
            "rms_residual_scaled_bfloat16"
        }
        None => "rms_residual_bfloat16",
    };
    with_params(
        &x.t,
        w.layer,
        kernel,
        vec![w.name.clone()],
        None,
        // `RmsParams`, field for field — `w_stride` is ONE, the distance
        // between consecutive CHANNELS of the gain vector. See `rms_norm`.
        vec![
            eps.to_bits(),
            row,
            1,
            u32::from(w.variant == model_ir::trace::NormVariant::Gemma),
            1.0f32.to_bits(),
        ],
        ins,
        Some(same_shape(x)),
    )
    .expect("a norm produces its value")
}

/// `attn/logit_softcap.metal::logit_softcap` — `cap * tanh(x / cap)`.
///
/// gemma's, applied to the readout so no logit runs away. A STATEMENT and
/// not a mode: a deployment without one names nothing here, rather than
/// passing a cap so large it does nothing — which would be a kernel run
/// per fire to compute the identity.
pub fn softcap(x: &Val, width: u32, cap: f32) -> Val {
    with_params(
        &x.t,
        x.layer,
        "logit_softcap_bfloat16",
        vec![],
        None,
        // `SoftcapParams`, field for field: cap then n.
        vec![cap.to_bits(), width],
        vec![x.id],
        Some((Shape(vec![Dim::Requests, Dim::Const(width)]), DType::BF16)),
    )
    .expect("the softcap produces its value")
}

/// `mlp/gated.metal::geglu_tanh` — gemma's activation.
///
/// `gelu_tanh(gate) * up`, and the gelu is the TANH approximation rather
/// than the erf one. A third symbol beside `silu_mul` and
/// `gptoss_swiglu`, and which a deployment takes is a load-time fact.
pub fn geglu(gate: &Val, up: &Val, intermediate: u32) -> Val {
    with_params(
        &gate.t,
        gate.layer,
        "geglu_tanh_bfloat16",
        vec![],
        None,
        // `GegluParams`: the element count.
        vec![intermediate],
        vec![gate.id, up.id],
        Some((
            Shape(vec![rows_of(gate), Dim::Const(intermediate)]),
            DType::BF16,
        )),
    )
    .expect("the activation produces its value")
}

/// `mlp/gated.metal::gptoss_swiglu` — gpt-oss's activation.
///
/// Not `silu_mul` with parameters. The gate is clamped ABOVE only, the
/// linear branch is clamped both ways and carries a `+1`, and dropping
/// either produces a model that runs and is wrong. So it is its own
/// symbol, and which one a deployment takes is a load-time fact.
pub fn swiglu(gate: &Val, up: &Val, intermediate: u32, limit: f32, alpha: f32) -> Val {
    with_params(
        &gate.t,
        gate.layer,
        "gptoss_swiglu_bfloat16",
        vec![],
        None,
        // `GptOssSwiGluParams`, field for field: n, limit, alpha.
        vec![intermediate, limit.to_bits(), alpha.to_bits()],
        vec![gate.id, up.id],
        Some((
            Shape(vec![rows_of(gate), Dim::Const(intermediate)]),
            DType::BF16,
        )),
    )
    .expect("the activation produces its value")
}

/// `layout/row_gather.metal::row_gather` — the sampled rows, in order.
///
/// A prefill's stream is one row per TOKEN and its readout is one
/// distribution per REQUEST, so the rows a fire samples have to be picked
/// out before the lm head runs. Which rows is `Step::sampling_indices`, a
/// fire table, so the row names it and no statement supplies it.
///
/// Absent, a prefill's readout reads row 0 and answers the FIRST token's
/// distribution — measured against MLX, and exactly right for a question
/// nobody asked.
pub fn sample_rows(x: &Val, width: u32) -> Val {
    with_params(
        &x.t,
        None,
        "row_gather_bfloat16",
        vec![],
        None,
        // `RowGatherParams`, packed: width then count. WIDTH only here --
        // how many rows to gather is the REQUEST count, a number of the
        // fire's that no text can state, and the row names it
        // (`Source::Named(<keys::RequestCount as keys::Fact>::KEY)`, `Ty::InPacked`) so the driver appends
        // it as the struct's second field.
        vec![width],
        vec![x.id],
        Some((Shape(vec![Dim::Requests, Dim::Const(width)]), DType::BF16)),
    )
    .expect("the gather produces its rows")
}

/// `quantized_qmv.metal::affine_qmv_fast` against the lm head — the
/// readout, `[Requests, vocab]` f32 like every family's.
pub fn lm_head(x: &Val, weight: &str, vocab: u32, repr: WeightRepr, point: &str) -> Val {
    with_params(
        &x.t,
        None,
        &format!("affine_qmv_fast{point}"),
        quant_table(weight, repr),
        None,
        // The codec point, as `qmv` states it: the two extents this run held
        // are the operands' own rectangles now.
        point_of(repr, point),
        vec![x.id],
        // BF16, because that is what the kernel WRITES. `affine_qmv_fast`
        // is instantiated at bfloat and its output is `device T*`; the
        // readout is not special-cased to widen.
        //
        // Stating F32 here sized the arena slot for four bytes an element
        // and the kernel filled two, so the logits region came back
        // EXACTLY half zero -- 64128 of 128256 -- with every surviving
        // value a fraction of its real magnitude. A dtype the trace states
        // and the kernel disagrees with is not a rounding difference; it
        // is a stride, and every value after the first is at the wrong
        // address.
        Some((Shape(vec![Dim::Requests, Dim::Const(vocab)]), DType::BF16)),
    )
    .expect("the readout produces the logits")
}

// ── The mixture. ──
//
// Six statements, and the reason they are six rather than one is the
// reason a mixture is interesting at all: a routed FFN's SHAPE depends on
// a value the fire computes. The router picks experts, the sort groups
// rows by the expert they picked, the gather materializes those groups
// contiguously, the matmuls run over the groups, and the combine puts the
// rows back where they started weighted by the router's confidence.
//
// Nothing here is a per-family branch. The executor walks these exactly as
// it walks a projection: symbol, row, file, rule, grid, operands. What is
// different is only that `LaunchRule::RouteRows` and `RoutedQmv` read
// `n_experts` and `experts_per_token` off the dims -- which is the same
// way `Qmv` reads `width`.

/// `moe/route.metal::router_topk` — which experts a row goes to, and how
/// much of each.
///
/// Two outputs: the expert slots and their weights. Both are read by name
/// downstream, which is why this returns the pair rather than folding them.
///
/// `norm_topk_prob` is the CHECKPOINT's, and it decides which softmax the
/// weights come out of: true takes it over the selected k, so they sum to
/// one; false takes it over ALL experts and then selects, so they sum to
/// less and scale the routed FFN's whole contribution down with them. It
/// is a parameter rather than a constant here because no DSL can know it
/// — HF states it per model, and qwen2-moe ships it false where qwen3-moe
/// ships it true.
pub fn router_topk(
    logits: &Val,
    n_experts: u32,
    experts_per_token: u32,
    per_expert_scale: Option<&MatW>,
    norm_topk_prob: bool,
) -> (Val, Val) {
    // The SCALED form takes a fifth buffer, and it used to be selected by a
    // `bool` that bound nothing: the text named `router_topk_scaled_bfloat16`
    // and handed it four operands, leaving the shader to read its per-expert
    // gain out of whatever the slot held. Nothing reached it -- every caller
    // passed `false` -- so the gap was a symbol away from firing rather than
    // a live defect. Naming the tensor instead of a flag makes the two
    // inseparable: there is no way to ask for the scaled kernel without
    // saying which weight it scales by.
    let sym = if per_expert_scale.is_some() {
        "router_topk_scaled_bfloat16"
    } else {
        "router_topk_bfloat16"
    };
    let slots = Dim::Const(experts_per_token);
    let ids = logits.t.with(logits.layer, |b| {
        b.launch_with_params(
            sym,
            per_expert_scale
                .map(|w| vec![w.name.clone()])
                .unwrap_or_default(),
            None,
            // `RouterParams`, packed: the shader takes a struct pointer,
            // so this run IS the struct and every word of it has to be
            // here. It used to be the first two, and the shader read the
            // other two out of the next dispatch's staged scalars --
            // `Params::new` sizes a packed run from the statement, and
            // the statement was two words short of what `route.metal`
            // reads. `softmax_over_all` decides the DENOMINATOR of every
            // routing weight, so a nonzero word in that position scales
            // the whole routed FFN down; `logits_pitch` strides the read.
            // Both produce weights, neither faults.
            vec![
                n_experts,
                experts_per_token,
                u32::from(!norm_topk_prob),
                // PACKED, spelled out rather than left as the shader's
                // zero-means-`n_experts`, which is what `route_gather`
                // does with its own `x_pitch` one function up. The
                // router's input is the gemm against `w.router`, whose
                // Shape is `[Tokens, n_experts]`.
                n_experts,
            ],
            vec![logits.id],
            vec![
                (Shape(vec![Dim::Tokens, slots]), DType::I32),
                (Shape(vec![Dim::Tokens, slots]), DType::BF16),
            ],
        )
    });
    let mk = |id| Val {
        t: logits.t.clone(),
        id,
        layer: logits.layer,
    };
    (mk(ids[0]), mk(ids[1]))
}

/// `moe/route.metal::route_sort` — group the rows by expert.
///
/// FOUR outputs, and a text that named fewer would leave the combine
/// reading whatever was in the buffer: the permutation, the per-row
/// expert, the per-tile expert, and the inverse the combine reads back.
///
/// # Two numbers, and neither is a constant
///
/// `n` is the count of `(token, slot)` PAIRS the router chose, and
/// `padded` is the height of the SORTED STACK those pairs land in —
/// each touched expert's run rounded up to a whole tile. Both are
/// functions of the fire, so both ride [`OpKind::Launch`](model_ir::trace::OpKind::Launch)(model_ir::trace::OpKind::Launch)'s `param_extents`
/// and the constants written beside them are zero.
///
/// They were one number, `n_experts * experts_per_token`, which is
/// neither: it is a property of the deployment. See
/// [`OpKind::Launch`](model_ir::trace::OpKind::Launch)(model_ir::trace::OpKind::Launch)'s `param_extents` for what that measured as.
pub fn route_sort(
    expert_ids: &Val,
    n_experts: u32,
    experts_per_token: u32,
    width: u32,
    block: u32,
) -> (Val, Val, Val, Val) {
    let pairs = Shape(vec![Dim::Tokens, Dim::Const(experts_per_token)]);
    let stack = Shape(vec![Dim::MoeAlignedRoutes {
        top_k: experts_per_token,
        experts: n_experts,
        block,
    }]);
    let ids = expert_ids.t.with(expert_ids.layer, |b| {
        b.launch_with_extents(
            "route_sort",
            vec![],
            None,
            // `MoeRouteParams`, packed and SHARED with the gather so the
            // sort's padding and the gather's bounds cannot disagree.
            vec![
                0,
                n_experts,
                experts_per_token,
                block,
                0,
                width,
                width,
            ],
            vec![(0, pairs.clone()), (4, stack.clone())],
            vec![expert_ids.id],
            vec![
                (stack.clone(), DType::I32),
                (stack.clone(), DType::I32),
                // One entry per TILE, sized as the whole stack: at
                // `block == 1` a tile IS a row and this is exact, and at a
                // GEMM's block it is an upper bound by a factor of `block`.
                // Over-sizing it is the safe direction -- the sort writes
                // `stack / block` entries and the GEMM reads that many --
                // and the alternative is a `Dim` that divides, which would
                // have to round the same way `moe_aligned_rows` does or
                // disagree with it at exactly the fires that pad.
                (stack.clone(), DType::I32),
                // Indexed by PAIR, not by position: `inv[i]` is where
                // pair `i` landed. `combine_sorted` reads it at
                // `token * k + slot`, which is why it is not the stack's
                // shape even when the two happen to be the same size.
                (pairs, DType::I32),
            ],
        )
    });
    let mk = |id| Val {
        t: expert_ids.t.clone(),
        id,
        layer: expert_ids.layer,
    };
    (mk(ids[0]), mk(ids[1]), mk(ids[2]), mk(ids[3]))
}

/// `moe/route.metal::route_gather` — the rows, in expert order.
///
/// One output row per sorted position, which is what the row's
/// [`KernelSig::rows_param`](model_ir::kernels::KernelSig::rows_param) tells the driver: this statement's
/// row axis is the stack, not the fire.
pub fn route_gather(
    x: &Val,
    perm: &Val,
    n_experts: u32,
    experts_per_token: u32,
    width: u32,
    block: u32,
) -> Val {
    let stack = Dim::MoeAlignedRoutes {
        top_k: experts_per_token,
        experts: n_experts,
        block,
    };
    with_extents(
        &x.t,
        x.layer,
        "route_gather",
        vec![],
        None,
        vec![
            0,
            n_experts,
            experts_per_token,
            block,
            0,
            width,
            width,
        ],
        vec![
            (0, Shape(vec![Dim::Tokens, Dim::Const(experts_per_token)])),
            (4, Shape(vec![stack])),
        ],
        vec![x.id, perm.id],
        Some((Shape(vec![stack, Dim::Const(width)]), DType::BF16)),
    )
    .expect("the gather produces its rows")
}

/// `quant/qmv.metal::affine_qmv_routed` — the expert-selecting GEMV.
///
/// `sel = row * slots_per_row + slot`, which is why the launch's row and
/// slot axes are not interchangeable and why `slots_per_row` is stated.
#[allow(clippy::too_many_arguments)]
pub fn routed_qmv(
    x: &Val,
    row_expert: &Val,
    w: &MatW,
    experts_per_token: u32,
    in_vec: u32,
    biased: bool,
    bits: u32,
) -> Val {
    // The EXPERT BANK's own format, which is not the dense projections'.
    //
    // This used to be the literal `..._bfloat16_gs_64_b_4`, both arms, and
    // that is a text choosing a quantisation instead of reading one. It
    // held for as long as the only routed checkpoint anyone ran was affine
    // at group 64.
    //
    // `mlx-community/gpt-oss-20b-MXFP4-Q4` is not: its `quantization`
    // block lists 98 tensors as `affine/64/4` and leaves the expert banks
    // OUT, so they take the top-level default -- `mxfp4`, group **32**.
    // The block has 122 entries; the 24 unaccounted for are the
    // `mlp.router` gates at 64/**8**, a third format in the same file.
    // Dequantising that as affine-64 reads every scale from the wrong
    // offset, and bf16 garbage is NaN more often than not. The fire ran,
    // bound everything, and produced 909,207 NaNs starting at the first
    // routed projection of layer 0.
    //
    // So the symbol comes from the repr the caller states, the same way
    // `qmv`'s does. A format with no routed instantiation names a symbol
    // no shader defines, which fails at pipeline construction with the
    // name in hand -- the refusal `moe.rs` already promised and was not
    // getting.
    let sym = match w.repr {
        // MXFP4's E2M1 mantissas with E8M0 block exponents. Not affine at
        // some other point -- different arithmetic -- so it is a different
        // symbol, and `quantized_qmv.metal` exports exactly one:
        // `mxfp4_qmv_routed_bias`, at group 32 and 4 bits, which is the
        // only shape MXFP4 has. There is no unbiased twin, which is why
        // this arm ignores `biased` -- and, since the name is not
        // decoration, why it must be HANDED one. See below.
        //
        // The point is spelled out because the arm KNEW it and did not
        // write it: this returned the bare symbol, and a bare symbol is
        // not an entry point. `quant/qmv.metal` exports exactly
        // `mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4`, so the fire failed
        // at pipeline construction with "exports no such entry point".
        //
        // The two numbers are literal here and derived on the affine arm
        // for a reason that is not laziness. Affine's group and bits are
        // the CHECKPOINT's -- its `quantization` block picks 64/4 or
        // 128/8 -- so `affine_point` reads them off the repr. MXFP4's are
        // the FORMAT's: E2M1 mantissas are four bits and an E8M0 block
        // exponent covers thirty-two of them, and a checkpoint claiming
        // any other pair would not be MXFP4.
        WeightRepr::Mxfp4Marlin => "mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4".to_string(),
        repr => {
            let point = affine_point(repr, bits);
            if biased {
                format!("affine_qmv_routed_bias{point}")
            } else {
                format!("affine_qmv_routed{point}")
            }
        }
    };
    let sym = sym.as_str();
    // WHETHER the chosen symbol reads an additive bias, which is not the
    // same question as `biased`: the affine pair has both instantiations
    // and honours the caller, MXFP4 has only the biased one and reads a
    // bias whatever was asked for.
    //
    // It decides the WEIGHT LIST, and that is why it has to be asked here.
    // `qmv_routed_bias`'s twelfth parameter is `const device T* bias
    // [[buffer(7)]]`, read per output row under `BIASED`, and it is a
    // different tensor from the codec's zero-point plane at `buffer(2)` --
    // one value per row against one per group. Handing the row only
    // `quant_weights` left the kernel's `bias` naming a weight the
    // statement does not have, which `driver-metal` bound as an address of
    // zero and the kernel added to every logit.
    //
    // Nothing caught it because no catalog row states `moe_mxfp4`, so the
    // one symbol that always reads a bias is the one nothing had run.
    let mut weights = quant_weights(w);
    if matches!(w.repr, WeightRepr::Mxfp4Marlin) || biased {
        weights.push(format!("{}.bias", w.name));
    }
    // `sel = tid.x * slots_per_row + tid.z` is a SORTED POSITION, so the
    // second operand is the sort's `row_expert` -- "the expert p reads,
    // for the matvec path", in `route_sort`'s own words -- and not the
    // router's `[Tokens, k]` choice. Given the latter the kernel read
    // `ids[sel]` where `sel` ranges over the stack, which agrees with the
    // routing only where the sort happened to be the identity.
    //
    // The two strides say the same thing about the INPUT. Every sorted
    // row has its own activation, at `sel * in_vec`, and `sel` arrives
    // factored as `(tid.x, tid.z)` -- so a row is `k` slots wide and a
    // slot is one. `x_slot_stride` was zero, which is the kernel's own
    // documented hazard: "reading slot 0 for every expert is not a crash
    // -- it is four copies of the first expert's activation, which
    // survives all the way to a plausible wrong token."
    //
    // `in_vec` is STATED rather than read off `x`'s trailing dim,
    // because that dim is a whole token's `k` runs end to end and this
    // number is one run. The caller knows which projection it is asking
    // for; the shape cannot say.
    with_params(
        &x.t,
        w.layer,
        sym,
        weights,
        None,
        // THE SORTED STACK'S THREE STRIDES. `in_vec` and `w.width` stood
        // first and left: they are the operands' own widths, which the marks
        // carry. What is left is the geometry the MIXTURE has and no operand
        // does -- a row is `k` slots wide and a slot is one.
        vec![in_vec, in_vec * experts_per_token, experts_per_token],
        vec![x.id, row_expert.id],
        // `k` results per token, end to end. The row axis of the MATVEC
        // is `w.width` alone and the row states so (`grid_param`); this
        // width is what the ELEMENTWISE activation between two of these
        // has to cover, and an elementwise grid is `width * rows`.
        Some((
            Shape(vec![
                Dim::Tokens,
                Dim::Const(w.width * experts_per_token.max(1)),
            ]),
            DType::BF16,
        )),
    )
    .expect("a routed projection produces its value")
}

/// `quant/qmm_t.metal::{affine,mxfp4}_qmm_t_routed` — the expert-selecting
/// GEMM, which is [`routed_qmv`]'s batched twin.
///
/// # Why this exists
///
/// A `qmv` reads one row per thread block and the whole of that row's expert
/// bank to produce it, so a fire of `n` rows reads the bank `n` times. That
/// is the right trade for a DECODE, where `n` is one route deep and a tile
/// would be fifteen sixteenths padding. It is the wrong one for a prefill,
/// and how wrong is measurable: pie's gpt-oss prefill was **flat at 166
/// tok/s from 32 tokens to 2048** — dead linear, six milliseconds a token,
/// which is what a per-row weight read looks like — against 595.6 recorded
/// for the C++ driver and 506.0 for llama.cpp on the same machine and file.
/// A batched GEMM amortises the read over its rows, so its tok/s RISES with
/// the prompt until the matrix unit saturates.
///
/// The kernels were never missing. `affine_qmm_t_routed` is in
/// `kernels-metal`'s catalog at nine tile shapes and six affine formats and
/// `mxfp4_qmm_t_routed_bias` at nine, all compiled, launch-ruled and
/// unit-tested, and `driver-metal` has had arms for all three since the
/// routine table was written. Nothing named them.
///
/// # What the caller owes
///
/// `tile_expert` — the third output of [`route_sort`], which every caller
/// already had and every caller discarded. It is one expert id per TILE, and
/// it is the whole difference between this and the matvec: the matvec asks
/// each row which bank it reads, and the GEMM asks each tile.
///
/// Which is why the sort's `block` must equal `tile.0` here. A tile whose
/// rows come from two experts gets one of their banks applied to both, and
/// that is not a rounding error — it is the wrong matrix, silently, in a
/// fire that still produces finite numbers.
///
/// # The row axis
///
/// The output is the SORTED STACK, `[MoeAlignedRoutes, w.width]`, and not
/// the matvec's `[Tokens, w.width * k]`. Those are the same buffer when
/// nothing is padded — `Tokens * k` rows of `width` either way — and stop
/// being the same the moment a block above one rounds an expert's run up.
/// `combine_sorted` reads this through the sort's `inv`, which is a sorted
/// POSITION, so it needs no telling.
#[allow(clippy::too_many_arguments)]
pub fn routed_qmm(
    rows: &Val,
    row_expert: &Val,
    tile_expert: &Val,
    w: &MatW,
    n_experts: u32,
    experts_per_token: u32,
    in_vec: u32,
    bits: u32,
    tile: (u32, u32),
    staged: bool,
) -> Val {
    let (bm, bn) = tile;
    // The bank's format picks the family exactly as it does for the matvec,
    // and for the same reason: MXFP4's E2M1 mantissas under E8M0 block
    // exponents are different arithmetic, not affine at another point.
    // MXFP4 has one instantiation and it reads a projection bias, so this
    // arm hands it one whatever was asked -- `routed_qmv` says why at
    // length.
    let sym = match w.repr {
        WeightRepr::Mxfp4Marlin => {
            format!("mxfp4_qmm_t_routed_bias_bfloat16_bm_{bm}_bn_{bn}")
        }
        // `staged` picks the HALF form of the same kernel: same buffers, same
        // grid, same `tile_expert` contract, so nothing below this line reads
        // differently. What it buys is the matrix instruction. On a device
        // with no native bfloat matrix unit a `simdgroup_matrix<bfloat>` is an
        // emulated sequence and a `<half>` is one instruction, and the
        // dequantizing loader hands the tiles over in either type for the same
        // work -- about 40% on the GEMM, which for gemma-4-26b-a4b is the
        // largest single term in a prefill.
        //
        // It is a PARAMETER and not a property of `repr` because the answer is
        // not the codec's. `affine_qmm_t_routed_fp16` is stamped at
        // `gs = 64, b = 4` alone, so the codec is necessary; it is not
        // sufficient, because a routed model's NEXT layer reads this layer's
        // output through a top-k, and a top-k is a comparison that a rounding
        // difference can reorder. llama's did, under `llama_numerics_test`,
        // which is why llama asks for false at the same codec gemma-4 asks
        // for true. The C++ driver excludes every routed projection here, but
        // for a reason that is not this one and does not apply: its precast
        // kernel has no routed form to select, and this one needs no precast
        // because it stages its own tiles.
        repr if staged => format!(
            "affine_qmm_t_routed_fp16{}",
            affine_gemm_point(repr, bits, tile)
        ),
        repr => format!("affine_qmm_t_routed{}", affine_gemm_point(repr, bits, tile)),
    };
    let mut weights = quant_weights(w);
    if matches!(w.repr, WeightRepr::Mxfp4Marlin) {
        weights.push(format!("{}.bias", w.name));
    }
    let stack = Dim::MoeAlignedRoutes {
        top_k: experts_per_token,
        experts: n_experts,
        block: bm,
    };
    with_params(
        &rows.t,
        w.layer,
        &sym,
        weights,
        None,
        // THE TILE, AND THE CODEC PAIR WHERE THE SYMBOL HAS ONE. `k` and `n`
        // stood here -- the contraction and the output width -- and both are
        // the operands' own rectangles now, which the marks carry.
        //
        // The tile is not: `_bm_32_bn_64` is a decision the compiler made
        // about this deployment's shapes, and the routine reads it as two
        // `Const<i32>`s. The affine form reads the group and the bit width
        // before them; the MXFP4 one has no codec point to read, which is
        // the same split `sym` above is chosen by.
        if matches!(w.repr, WeightRepr::Mxfp4Marlin) || staged {
            vec![bm, bn]
        } else {
            let mut run = point_of(w.repr, &affine_point(w.repr, bits));
            run.extend([bm, bn]);
            run
        },
        // `row_expert` rides the second slot the arm calls `pad` -- the
        // GEMM does not read it, and binding a real buffer there rather
        // than nothing keeps the operand list the same length as the
        // matvec's for anything that counts operands.
        vec![rows.id, row_expert.id, tile_expert.id],
        Some((Shape(vec![stack, Dim::Const(w.width)]), DType::BF16)),
    )
    .expect("a routed GEMM produces its rows")
}

/// `moe/route.metal::combine_sorted` — the rows back where they started,
/// weighted by the router.
pub fn combine_sorted(
    y: &Val,
    expert_weights: &Val,
    inv: &Val,
    experts_per_token: u32,
    width: u32,
) -> Val {
    with_params(
        &y.t,
        y.layer,
        "combine_sorted",
        vec![],
        None,
        // `ExpertCombineParams`, packed — all THREE words, for the
        // reason `router_topk` states: a short run leaves the shader
        // reading the next dispatch's scalars as its own trailing
        // fields. `out_pitch` is the elements between one output row and
        // the next, and the combine's output is its own value with Shape
        // `[Tokens, width]`, so the rows are `width` apart.
        vec![width, experts_per_token, width],
        vec![y.id, expert_weights.id, inv.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
    )
    .expect("the combine produces its rows")
}

/// `moe/route.metal::shared_expert_combine` — `routed + sigmoid(gate) *
/// shared`, the landing for a mixture that also has a dense expert.
pub fn shared_expert_combine(routed: &Val, shared: &Val, gate: &Val, width: u32) -> Val {
    with_params(
        &routed.t,
        routed.layer,
        "shared_expert_combine",
        vec![],
        None,
        vec![width],
        vec![routed.id, shared.id, gate.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
    )
    .expect("the shared landing produces its rows")
}

// ------------------------------------------------------------------ gdn ---

/// The shape a gated-DeltaNet block runs at, as `GdnCoreParams` orders it.
///
/// One struct rather than eleven arguments because the shader reads it as one
/// struct: `ssm/gdn_params.h` declares `{Dk, Dv, Hk, Hv, conv_dim, Kc, q_off,
/// k_off, v_off, eps, inv_sqrt_dk}` and `Handles::params_block` packs the
/// statement's scalars in the order they were stated. A text that stated ten
/// of them would not fail -- it would leave `inv_sqrt_dk` reading whatever
/// word followed, which is a prescale and so a wrong answer rather than a
/// fault.
#[derive(Debug, Clone, Copy)]
pub struct GdnShape {
    /// Key head width.
    pub k_dim: u32,
    /// Value head width.
    pub v_dim: u32,
    /// Key heads.
    pub k_heads: u32,
    /// Value heads.
    pub v_heads: u32,
    /// Channels the causal convolution carries: `2*Hk*Dk + Hv*Dv`.
    pub conv_dim: u32,
    /// Convolution kernel width.
    pub conv_k: u32,
    /// Where q, k and v start in the fused in-projection's row.
    pub q_off: u32,
    /// See [`Self::q_off`].
    pub k_off: u32,
    /// See [`Self::q_off`].
    pub v_off: u32,
    /// The l2 norm's epsilon.
    pub eps: f32,
}

impl GdnShape {
    /// The eleven scalars, in `GdnCoreParams` order.
    ///
    /// `inv_sqrt_dk` is COMPUTED here rather than stated: it is `Dk**-0.5`
    /// and nothing else, and a text free to state it separately is a text
    /// that can state it inconsistently with the `Dk` beside it.
    fn params(self) -> Vec<u32> {
        vec![
            self.k_dim,
            self.v_dim,
            self.k_heads,
            self.v_heads,
            self.conv_dim,
            self.conv_k,
            self.q_off,
            self.k_off,
            self.v_off,
            self.eps.to_bits(),
            (self.k_dim as f32).powf(-0.5).to_bits(),
        ]
    }
}

/// The four weights every GDN statement reads, in the order the arms ask.
///
/// `arm::gates` reads weights two and three and the conv pair is zero and
/// one, so the order here is the signature and not a convenience. The `a`
/// and `b` gates are NOT here: they are the in-projection's per-token
/// outputs and arrive as operands.
#[derive(Debug, Clone)]
pub struct GdnW {
    /// The causal convolution's kernel and bias.
    pub conv_w: String,
    /// See [`Self::conv_w`].
    pub conv_b: String,
    /// The decay's log-space rate, `f32`.
    pub a_log: String,
    /// The decay's bias.
    pub dt_bias: String,
}

impl GdnW {
    fn names(&self) -> Vec<String> {
        vec![
            self.conv_w.clone(),
            self.conv_b.clone(),
            self.a_log.clone(),
            self.dt_bias.clone(),
        ]
    }
}

fn recurrent_state(layer: u32) -> Option<StateRef> {
    Some(StateRef {
        store: StateStore::RecurrentState,
        layer,
    })
}

/// `ssm/gdn_core.metal::gdn_core_slotted_bfloat16` — the fused
/// convolution, norm, gating and recurrent step, one token per request.
///
/// SLOTTED unconditionally, and the sealed `gdn_core_bfloat16` is not stated
/// by any text here. The two symbols take the same twelve buffers; the
/// difference is whether the conv and recurrent slabs are indexed by the row
/// or by the fire's slot table, and a served fire always has requests taking
/// turns in a slab. The sealed form is what `device_gdn.rs` compares the
/// split pair against, which is a claim about the arithmetic and not a
/// serving path.
pub fn gdn_core(mixed: &Val, a: &Val, b: &Val, shape: GdnShape, w: &GdnW, layer: u32) -> Val {
    with_params(
        &mixed.t,
        Some(layer),
        "gdn_core_slotted_bfloat16",
        w.names(),
        recurrent_state(layer),
        shape.params(),
        vec![mixed.id, a.id, b.id],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(shape.v_heads * shape.v_dim)]),
            DType::BF16,
        )),
    )
    .expect("the gdn core produces its rows")
}

/// `ssm/gdn_prep.metal::gdn_prep_slotted_bfloat16` — the q/k path, computed
/// once per head instead of once per value dimension.
///
/// Three results, all `f32`: the normalized and prescaled q, the normalized
/// k, and `{decay, beta}` per value head. They are the bridge to
/// [`gdn_core_recurrent`], which recomputes none of them.
pub fn gdn_prep(
    mixed: &Val,
    a: &Val,
    b: &Val,
    shape: GdnShape,
    w: &GdnW,
    layer: u32,
) -> (Val, Val, Val) {
    let per_head = Shape(vec![Dim::Tokens, Dim::Const(shape.v_heads * shape.k_dim)]);
    let gate = Shape(vec![Dim::Tokens, Dim::Const(2 * shape.v_heads)]);
    let ids = mixed.t.with(Some(layer), |t| {
        t.launch_with_params(
            "gdn_prep_slotted_bfloat16",
            w.names(),
            recurrent_state(layer),
            shape.params(),
            vec![mixed.id, a.id, b.id],
            vec![
                (per_head.clone(), DType::F32),
                (per_head, DType::F32),
                (gate, DType::F32),
            ],
        )
    });
    let mk = |id| Val {
        t: mixed.t.clone(),
        id,
        layer: Some(layer),
    };
    (mk(ids[0]), mk(ids[1]), mk(ids[2]))
}

/// How a prefill scan divides the value dimension: `(lanes, vrows)`.
///
/// `LANES` lanes own one `dv` row, so `32/LANES` rows share a simdgroup and
/// each walks `VROWS` of them. It is an instantiation axis spelled into the
/// entrypoint's own name — `gdn_core_recurrent_prefill_bfloat16_l_32_v_2` —
/// and the shader tree carries nine of the pairs, so a text states one the
/// tree has or the driver refuses `Narrow` before it fires.
///
/// # Why this pair, measured
///
/// All nine answer the walk identically — `device_gdn.rs`'s
/// `the_prefill_scan_answers_the_decode_walked_token_by_token` holds every
/// one of them to the token-by-token decode — so the choice is only speed.
/// Swept on M1 Max at a 128-token prefill, tok/s:
///
/// |          | (4,1) | (8,1) | (8,2) | (16,1) | (16,2) | (16,4) | (32,2) | (32,4) | (32,8) |
/// |----------|-------|-------|-------|--------|--------|--------|--------|--------|--------|
/// | 35B-A3B  | 272.2 | 250.0 | 278.1 | 285.1  | 290.1  | 287.2  | **296.8** | 293.3 | 285.4 |
/// | 27B      |       |       |       |        | 90.5   |        | 91.7   | 91.1   | **92.2** |
///
/// The two shapes do not agree on a winner — A3B's best is 27B's worst — so
/// this is the pair that is at least the old default on BOTH, not the one
/// that wins either. A3B's three repeats were 296.6/296.9/296.8 against
/// 293.5/293.1/293.3, which do not overlap; the whole axis is worth 4% on
/// A3B and 2% on 27B, so a per-shape choice is not worth the second knob.
///
/// The axis is NOT where a prefill regression lives. `f63aa3021` cost A3B
/// 16% by replacing a token-parallel decode scan with this serial one, and
/// no tiling here returns any of it — the serialization is the correctness.
pub const GDN_SCAN_TILE: (u32, u32) = (32, 2);

/// `ssm/gdn_prep.metal::gdn_prep_prefill_bfloat16` — [`gdn_prep`] over a
/// whole PROMPT, plus the value channels the fused pair convolves later.
///
/// # Why a prefill needs its own pair
///
/// `gdn_core_recurrent_slotted` is a DECODE kernel. It indexes the recurrent
/// state by SLOT and runs its grid over ROWS, so a prefill fires one thread
/// per token at one state and every one of them reads it, decays it, updates
/// it and stores it — a recurrence defined over tokens, run in parallel over
/// them. It does not fault and it does not warn; it returns a different
/// answer each time the same program fires.
///
/// This pair walks the tokens inside ONE kernel, with the state in registers
/// for the duration, and there is no order for a scheduler to get wrong.
///
/// # The four results
///
/// Three more than [`gdn_prep`]'s three, in one: `pre_gate` carries the
/// `{decay, beta}` pair per value head AND the value channels, because the
/// scan's `Dk` lanes all consume the same `v` scalar and computing it here —
/// while tokens are still parallel — is the whole reason to split. So its
/// row is `2*Hv + Hv*Dv` where the slotted prep's is `2*Hv`.
///
/// Every row is packed at its own width. The kernels took one `row_pitch`
/// for all of them until qwen3-next asked for a `pre_gate` row wider than
/// its own in-projection.
pub fn gdn_prep_prefill(
    mixed: &Val,
    a: &Val,
    b: &Val,
    shape: GdnShape,
    w: &GdnW,
    layer: u32,
) -> (Val, Val, Val) {
    let per_head = Shape(vec![Dim::Tokens, Dim::Const(shape.v_heads * shape.k_dim)]);
    let gate = Shape(vec![
        Dim::Tokens,
        Dim::Const(2 * shape.v_heads + shape.v_heads * shape.v_dim),
    ]);
    let ids = mixed.t.with(Some(layer), |t| {
        t.launch_with_params(
            "gdn_prep_prefill_bfloat16",
            w.names(),
            recurrent_state(layer),
            shape.params(),
            vec![mixed.id, a.id, b.id],
            vec![
                (per_head.clone(), DType::F32),
                (per_head, DType::F32),
                (gate, DType::F32),
            ],
        )
    });
    let mk = |id| Val {
        t: mixed.t.clone(),
        id,
        layer: Some(layer),
    };
    (mk(ids[0]), mk(ids[1]), mk(ids[2]))
}

/// `ssm/gdn_prep.metal::gdn_core_recurrent_prefill_bfloat16_l_*_v_*` — the
/// whole prompt's recurrence, walked inside one kernel.
///
/// See [`gdn_prep_prefill`] for why the slotted scan cannot serve a prefill.
///
/// It does NOT take `mixed` as an operand it reads — the prep already
/// convolved v — but it is still the first input, because this entrypoint
/// declares nothing at five of its fourteen buffer slots and a Metal
/// argument table is a contiguous run. The driver binds it into the holes.
///
/// It names NO WEIGHTS, and that is the same fact stated on the other axis.
/// [`gdn_core_recurrent`], the slotted form, reads `conv_w` and `conv_b`
/// because it convolves v itself; this one does not convolve anything, so
/// its routine takes no weight parameter and `arm::gdn_core_recurrent_prefill`
/// binds none. It used to pass `w.names()` anyway -- copied from the
/// slotted statement, where the four are real -- and §6.2 read the
/// difference as a routine four pointers short of what its statement placed.
///
/// Nothing is lost by dropping them: [`gdn_prep_prefill`] runs immediately
/// before this on the same layer and names all four, so the loader resolves
/// and pins exactly what it did.
///
/// `tile` rides TWICE: spelled into the symbol, where `routine::crossed`
/// finds the stem by longest match and `plan_routine` checks the driver's
/// composed name against it, and as the statement's two scalars past
/// `GdnShape::params`'s eleven, where the arm reads them. Two spellings of
/// one choice, and the dispatch refuses if they ever disagree.
pub fn gdn_core_recurrent_prefill(
    mixed: &Val,
    pre_q: &Val,
    pre_k: &Val,
    pre_gate: &Val,
    shape: GdnShape,
    w: &GdnW,
    layer: u32,
    tile: (u32, u32),
) -> Val {
    // See the doc: the scan reads none of the four, and the prep beside it
    // names every one.
    let _ = w;
    let mut params = shape.params();
    params.push(tile.0);
    params.push(tile.1);
    let (lanes, vrows) = tile;
    with_params(
        &mixed.t,
        Some(layer),
        &format!("gdn_core_recurrent_prefill_bfloat16_l_{lanes}_v_{vrows}"),
        vec![],
        recurrent_state(layer),
        params,
        vec![mixed.id, pre_q.id, pre_k.id, pre_gate.id],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(shape.v_heads * shape.v_dim)]),
            DType::BF16,
        )),
    )
    .expect("the gdn prefill scan produces its rows")
}

/// `ssm/gdn_prep.metal::gdn_core_recurrent_slotted_bfloat16` — the scan over
/// what [`gdn_prep`] wrote.
///
/// It still takes `mixed`, and that is not a redundancy: the v channel is
/// unique per value dimension, so there is no q/k-style sharing to hoist and
/// the scan does its own convolution over v.
pub fn gdn_core_recurrent(
    mixed: &Val,
    pre_q: &Val,
    pre_k: &Val,
    pre_gate: &Val,
    shape: GdnShape,
    w: &GdnW,
    layer: u32,
) -> Val {
    with_params(
        &mixed.t,
        Some(layer),
        "gdn_core_recurrent_slotted_bfloat16",
        w.names(),
        recurrent_state(layer),
        shape.params(),
        vec![mixed.id, pre_q.id, pre_k.id, pre_gate.id],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(shape.v_heads * shape.v_dim)]),
            DType::BF16,
        )),
    )
    .expect("the gdn scan produces its rows")
}

/// `norm/gated_rms.metal::gated_rms` — `w * rms(x) * silu(z)`, over each
/// value head.
///
/// The gated-deltanet landing, and the one norm in this surface that does
/// NOT fold `1 + w`: `gated_rms.metal:11` reads the weight raw, because the
/// checkpoint's `linear_attn.norm.weight` is already the multiplier. So it
/// takes a name rather than a [`NormW`] -- there is no variant to choose,
/// and a handle offering one would be offering a choice the shader does not
/// have.
pub fn gated_rms(x: &Val, z: &Val, weight: &str, v_heads: u32, v_dim: u32, eps: f32) -> Val {
    with_params(
        &x.t,
        x.layer,
        "gated_rms_bfloat16",
        vec![weight.to_string()],
        None,
        // `GatedRmsParams`: `{eps, vd}`, both words, in that order.
        vec![eps.to_bits(), v_dim],
        vec![x.id, z.id],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(v_heads * v_dim)]),
            DType::BF16,
        )),
    )
    .expect("the gated norm produces its rows")
}

/// `attn/gate.metal::q_gate_split` — the `[query | gate]` bank cut per head.
///
/// The projection is `[rows, heads, 2, head_dim]`: each head's queries are
/// followed by that head's gate, which is why a row split at `heads *
/// head_dim` would be wrong and this is a kernel rather than two views.
///
/// Both pitches are stated because the source is twice as wide per head as
/// either result, so they are different numbers even when both are packed.
pub fn q_gate_split(qg: &Val, q_heads: u32, head_dim: u32) -> (Val, Val) {
    let width = q_heads * head_dim;
    let shape = Shape(vec![Dim::Tokens, Dim::Const(width)]);
    let ids = qg.t.with(qg.layer, |t| {
        t.launch_with_params(
            "q_gate_split_bfloat16",
            vec![],
            None,
            // `[head_dim, qg_row_stride, out_row_stride, q_heads]`, which is
            // the run the routine's four `Const` marks claim, in order. The
            // two strides were `Param<1>`/`Param<2>` at HEAD and spent a spell
            // as asks no driver answered; the head count closes the run.
            vec![head_dim, 2 * width, width, q_heads],
            vec![qg.id],
            vec![(shape.clone(), DType::BF16), (shape, DType::BF16)],
        )
    });
    let mk = |id| Val {
        t: qg.t.clone(),
        id,
        layer: qg.layer,
    };
    (mk(ids[0]), mk(ids[1]))
}

/// `attn/gate.metal::gate` — `attn *= sigmoid(gate)`, in place.
///
/// IN PLACE, which is why the result is the first operand read back and not
/// a fresh value: the shader takes the gated tensor as a mutable buffer at
/// slot zero. `arm::gate` reads it as `o.output(0)`, so the statement
/// declares its result and hands the shader the same rows it was given.
pub fn sigmoid_gate(attn: &Val, gate: &Val, width: u32) -> Val {
    with_params(
        &attn.t,
        attn.layer,
        "gate_bfloat16",
        vec![],
        None,
        vec![width],
        vec![attn.id, gate.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
    )
    .expect("the output gate produces its rows")
}
