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
/// ONE, because the only routed projection this backend states is a
/// MATVEC: `qmv_routed` reads one row per thread block and indexes the
/// bank by that row's own expert, so grouping the rows buys locality
/// and nothing about a tile. At one, `moe_aligned_rows` is exactly the
/// route count and the sort is a pure permutation — no padding rows to
/// zero, no spare tiles, and `tile_expert` is one entry per row.
///
/// It was `QMM_TILE`-shaped (sixteen) by inheritance
/// from a tiled matmul this text does not launch, which padded a
/// four-token fire's sixteen routes out to two hundred and fifty-six
/// rows — sixteen times the matvec work, and every padded row read by
/// the expert projection.
///
/// A blocked path would state its own block here, and would then need
/// an extent for `tile_expert` that divides by it.
const ROUTE_BLOCK: u32 = 1;

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
        vec![hidden],
        vec![],
        Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
    )
    .expect("embed produces the residual stream")
}

/// `rms_norm.metal::rms_single_row_bfloat16` — ONE entrypoint for
/// every norm this family states (attn_norm, mlp_norm, q_norm,
/// k_norm, final_norm; the driver fans five `Kernel` kinds onto it).
pub fn rms_norm(x: &Val, w: &NormW, row: u32, eps: f32) -> Val {
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
            1.0f32.to_bits(),
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
/// The GEMM's instantiation point: [`affine_point`] plus its tile.
///
/// `affine_qmm_t` is stamped over `(group × bits × bm × bn)`, so its
/// symbol carries two more numbers than the GEMV's.
#[must_use]
pub fn affine_gemm_point(repr: WeightRepr, bits: u32, tile: (u32, u32)) -> String {
    let (bm, bn) = tile;
    format!("{}_bm_{bm}_bn_{bn}", affine_point(repr, bits))
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
        // The GEMV's two extents: the row it reads and the row it writes.
        // A projection told its output is zero wide computes nothing and
        // reports success, which is why these are stated and not derived.
        vec![in_width(x), w.width],
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
        vec![in_width(x), w.width],
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
        // The GEMV's two extents: the row it reads and the row it writes.
        // A projection told its output is zero wide computes nothing and
        // reports success, which is why these are stated and not derived.
        vec![in_width(x), w.width],
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
        vec![in_width(x), w.width],
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
/// reads with `Source::OutWidth(0)` rather than taking as a stated scalar
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
pub fn rope(
    q: &Val,
    k: &Val,
    multi_batch: bool,
    theta: f32,
    scale: f32,
    head_dim: u32,
    rotary_dim: u32,
    table: bool,
) -> (Val, Val) {
    (
        rope_one(q, multi_batch, theta, scale, head_dim, rotary_dim, table),
        rope_one(k, multi_batch, theta, scale, head_dim, rotary_dim, table),
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
/// In place, so the result is the operand: the row states `x` as its
/// `Out(0)` and the same buffer is read and written.
fn rope_one(
    x: &Val,
    multi_batch: bool,
    theta: f32,
    scale: f32,
    head_dim: u32,
    rotary_dim: u32,
    table: bool,
) -> Val {
    // IN PLACE, and the statement has to say so. Every `neox` entrypoint
    // takes ONE buffer -- `device T* x`, read and written -- so a
    // statement that declared a separate result had the row's `Out(0)`
    // bind the RESULT's slot, which no kernel had written. The rotation
    // then read whatever the arena held there and the value everything
    // downstream wanted was never rotated at all.
    //
    // Position zero hid it completely: rope is the identity at position
    // zero (cos 0 = 1, sin 0 = 0), so rotating the wrong buffer and
    // rotating the right one agree exactly, and the first reference gate
    // was set there for an unrelated reason.
    //
    // Stating no result makes `dispatch::reorder` bind `Out(0)` to the
    // last widthed operand, which for a one-operand launch is the input --
    // the buffer the kernel actually mutates. The value handed back is the
    // input's, because after the launch that IS the rotated tensor.
    //
    // A deployment that RESCALES its frequency ladder cannot state a base:
    // llama-3 rescales piecewise and YaRN rescales differently, and both
    // are tables. The driver derives one at load and answers it as
    // `Source::RopeFrequencies`, so the statement's job is only to say
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
        let stem = "neox_mb_bfloat16";
        (
            stem.to_string(),
            // The rotation's scale, its log2 base and the head width. The
            // base is `log2(theta)` because the shader raises two to it --
            // `rope_neox_geometric_body` -- and handing it theta rotates
            // by a frequency ladder wrong from the second channel on.
            // The rotary WIDTH last -- see the table form above.
            vec![
                scale.to_bits(),
                theta.log2().to_bits(),
                head_dim,
                rotary_dim,
            ],
        )
    };
    with_params(
        x.trace(),
        x.layer(),
        &kernel,
        vec![],
        None,
        params,
        vec![x.id],
        None,
    );
    x.clone()
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
        vec![gqa_factor, kv_heads, scale.to_bits(), 0, window as u32],
        vec![q.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(q_width)]), DType::BF16)),
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
            Shape(vec![Dim::Tokens, Dim::Const(intermediate)]),
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
        vec![width, scale.to_bits()],
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
            Shape(vec![Dim::Tokens, Dim::Const(intermediate)]),
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
            Shape(vec![Dim::Tokens, Dim::Const(intermediate)]),
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
        // (`Source::RequestCount`, `Ty::InPacked`) so the driver appends
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
        vec![in_width(x), vocab],
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
    scaled: bool,
    norm_topk_prob: bool,
) -> (Val, Val) {
    let sym = if scaled {
        "router_topk_scaled_bfloat16"
    } else {
        "router_topk_bfloat16"
    };
    let slots = Dim::Const(experts_per_token);
    let ids = logits.t.with(logits.layer, |b| {
        b.launch_with_params(
            sym,
            vec![],
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
) -> (Val, Val, Val, Val) {
    let pairs = Shape(vec![Dim::Tokens, Dim::Const(experts_per_token)]);
    let stack = Shape(vec![Dim::MoeAlignedRoutes {
        top_k: experts_per_token,
        experts: n_experts,
        block: ROUTE_BLOCK,
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
                ROUTE_BLOCK,
                0,
                width,
                width,
            ],
            vec![(0, pairs.clone()), (4, stack.clone())],
            vec![expert_ids.id],
            vec![
                (stack.clone(), DType::I32),
                (stack.clone(), DType::I32),
                // One entry per TILE, and at [`ROUTE_BLOCK`] a tile is a
                // row -- so the stack's own extent is exact rather than
                // an upper bound. A blocked path would state its block
                // here and need an extent that divides by it.
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
) -> Val {
    let stack = Dim::MoeAlignedRoutes {
        top_k: experts_per_token,
        experts: n_experts,
        block: ROUTE_BLOCK,
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
            ROUTE_BLOCK,
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
        vec![
            in_vec,
            w.width,
            in_vec,
            in_vec * experts_per_token,
            experts_per_token,
        ],
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
