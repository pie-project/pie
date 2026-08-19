//! `llama_like` — the family with no structural divergence.
//!
//! Three texts over one shape: a SEMANTIC trace that names operations and
//! never kernels, and one LOWERED text per backend. The semantic arm is what
//! parity holds the other two to.

pub mod facts;

use self::facts::{
    Activation, LlamaLikeCudaFacts, LlamaLikeFacts, LlamaLikeMetalFacts, NormPlacement, QkNorm,
};
use model_dsl::{
    self as dsl, MatW, Val, add_bias, attention, cuda, matmul, rmsnorm, rope, split_qkv, swiglu,
};
use model_ir::trace::{DType, Dim, FireClass, ForwardPlan, GuardPred, RopeKind, Shape};

/// The llama_like body — SEMANTIC form: no structural divergence, one
/// trace serves every fire shape, kernel choice stays with the consumer
/// (Metal, the engine's site table, `declared_dag`).
///
/// This is its OWN text. It was until recently the `lower: None` reading
/// of a single text shared with [`llama_like_cuda`], with eight
/// `m.lowering()` tests deciding which of two programs the reader was
/// looking at. `.wiki/tart/dsl.md` ③ says a model file is written for one
/// backend, so the two readings are now two texts and neither asks "am I
/// lowered?". The goldens pin that the split changed no traced byte.
///
/// Mirrors `crates/driver-cuda/csrc/src/model/llama_like/llama_like.cpp`
/// (`llama_like_forward_paged`) op for op; the golden test pins that
/// correspondence and the comment there maps each op to the kernel(s) the
/// hand-written pass would launch.
///
/// Norm placement branches the block structure itself (the first fact to
/// do so):
///
/// * `Pre` — norm the stream into the sub-layer, accumulate the output
///   projection straight back (`matmul_add`, the `beta=1` GEMM).
/// * `Post` (olmo2) — the sub-layer reads the stream raw, its output
///   projection lands in scratch (`beta=0`), the norm applies to THAT, and
///   a separate `ResidualAdd` lands it — the hand-written post-norm walk's
///   gemm → `kernels::norm::rmsnorm_bf16` → `kernels::norm::residual_add_bf16` triplet.
pub fn llama_like(facts: &LlamaLikeFacts) -> ForwardPlan {
    dsl::trace_semantic("llama_like", &facts.shape(), |m| {
        dsl::seam(m.trace(), &dsl::seam::IN, &[], None);
        let f = facts.clone();
        let q_w = f.q_width();
        let kv_w = f.kv_width();
        let post_norm = f.norm_placement == NormPlacement::Post;

        let mut y = m.embed();

        for l in 0..f.layers {
            let w = m.layer(l);

            // Attention block: (pre-norm) -> qkv -> (q/k norms) -> rope
            // -> append -> attention -> o_proj landed on the residual.
            let x = if post_norm {
                y.clone()
            } else {
                rmsnorm(&y, &w.attn_norm)
            };

            let (q, k, v) = if f.fused_qkv {
                split_qkv(&matmul(&x, &w.qkv), q_w, kv_w)
            } else {
                (
                    matmul(&x, &w.q_proj),
                    matmul(&x, &w.k_proj),
                    matmul(&x, &w.v_proj),
                )
            };
            // Qwen-2 family qkv biases: on the raw projections, before
            // norms and rope.
            let (q, k, v) = if f.qkv_bias {
                (
                    add_bias(&q, &w.q_bias),
                    add_bias(&k, &w.k_bias),
                    add_bias(&v, &w.v_bias),
                )
            } else {
                (q, k, v)
            };
            // The q/k norm convention is the weight handle's ("the weight
            // knows"); the semantic text states norm and rope separately
            // because their kernels are 1:1.
            let (q, k) = if f.qk_norm == QkNorm::Off {
                (q, k)
            } else {
                (rmsnorm(&q, &w.q_norm), rmsnorm(&k, &w.k_norm))
            };
            let (q, k) = rope(&q, &k, f.rope);
            w.kv.append(&k, &v);
            let a = attention(&q, &w.kv, q_w);

            if post_norm {
                // Post-norm: o_proj to scratch, norm the OUTPUT, then the
                // separate residual landing (`+=` of a non-matmul records
                // the explicit ResidualAdd launch).
                y += rmsnorm(&matmul(&a, &w.o_proj), &w.attn_norm);
                let mlp = matmul(&swiglu(&matmul(&y, &w.gate_up), f.intermediate), &w.down);
                y += rmsnorm(&mlp, &w.mlp_norm);
            } else {
                // Pre-norm: `+=` of a fresh matmul IS the beta=1 fold.
                y += matmul(&a, &w.o_proj);
                let x = rmsnorm(&y, &w.mlp_norm);
                y += matmul(&swiglu(&matmul(&x, &w.gate_up), f.intermediate), &w.down);
            }
        }

        let logits = m.logits(&rmsnorm(&y, &m.final_norm()));
        dsl::seam(m.trace(), &dsl::seam::OUT, &[&logits], None);
    })
}

/// The LOWERED llama_like: the SAME text as [`llama_like`], traced with
/// the CUDA backend facts and a fire class in hand, so the class arms run
/// The all-reduce, as the pair of arms it actually is.
///
/// `NcclComm::all_reduce_bf16` asks `can_handle(bytes)` and routes to
/// the NVLink P2P kernel below the threshold and `ncclAllReduce` above
/// it. That is an `if` inside a driver method choosing between two
/// implementations, which is the shape this arc removes everywhere
/// else, and it was left standing because a collective did not look
/// like a kernel choice. It is one.
///
/// So the text states both and the fire picks: `TokensLE(n)` where `n`
/// is the threshold in ROWS ([`LlamaLikeCudaFacts::all_reduce_p2p_max_rows`],
/// converted from bytes at load, because a row is `hidden` bf16
/// elements). A deployment with no threshold — no registered P2P
/// buffers, or no custom all-reduce at all — states the NCCL arm alone,
/// which is the truth rather than a guard whose predicate never holds.
fn all_reduce(t: &model_dsl::Trace, x: &Val, hidden: u32, cuda: &LlamaLikeCudaFacts) -> Val {
    if cuda.all_reduce_p2p_max_rows == 0 {
        return cuda::all_reduce_out(x, hidden);
    }
    let shape = (Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16);
    let (g, v) = dsl::guarded_value(t, x.layer(), shape);
    // Both discards are the guard's doing: an arm's statement lands in `v`,
    // which `guarded_value` minted before either arm ran, so the `Val` a
    // builder hands back here is a second name for something already held.
    g.arm(GuardPred::TokensLE(cuda.all_reduce_p2p_max_rows), || {
        let _ = cuda::all_reduce_p2p(x, hidden);
    })
    .otherwise(|| {
        let _ = cuda::all_reduce_out(x, hidden);
    });
    v
}

/// Whether this deployment's heads and intermediate divide by `tp`.
///
/// The engine checks the same thing at load; the text checks it because
/// a shard width that does not divide is a trace whose every projection
/// is quietly wrong, and a `ForwardPlan` has no later place to notice.
fn shard_divides(f: &LlamaLikeFacts, tp: u32) -> bool {
    tp > 0
        && f.q_heads.is_multiple_of(tp)
        && f.kv_heads.is_multiple_of(tp)
        && f.intermediate.is_multiple_of(tp)
}

/// The MLP's projection-and-activation pair, in the spelling this
/// deployment's BINDING fires (2d).
///
/// `packed` is [`LlamaLikeCudaFacts::gate_up_fused`]. The loader's dense
/// join either materialised one `[2I, H]` bank or it did not, and that
/// is known at load — so the trace states one form or the other and
/// nothing downstream asks again.
///
/// It used to state the PACKED matmul either way and let the activation
/// carry a `packed` flag. That made the unfused reading a lie: the
/// executor fired two GEMMs into `ws.gate` / `ws.up`, buffers the single
/// traced value did not describe, and then cross-checked the activation
/// against the fact on every launch to catch the drift it had created.
/// Two statements say it instead.
fn mlp(x: &Val, w: &dsl::Layer, intermediate: u32, packed: bool) -> Val {
    if packed {
        cuda::swiglu(&matmul(x, &w.gate_up), intermediate)
    } else {
        cuda::swiglu_pair(
            &matmul(x, &w.gate_proj),
            &matmul(x, &w.up_proj),
            intermediate,
        )
    }
}

/// and the traced form states its kernels as raw signatures
/// ([`model_dsl::cuda`]; north-star-dsl.md). One trace per
/// [`FireClass`]; family names `llama_like.cuda.decode` / `.prefill`.
pub fn llama_like_cuda(
    facts: &LlamaLikeFacts,
    cuda: &LlamaLikeCudaFacts,
    class: FireClass,
) -> ForwardPlan {
    llama_like_cuda_text(facts, cuda, class)
}

/// The llama_like METAL text (`.wiki/tart/dsl.md` ③) — the second
/// backend's own model file, stating Metal's kernels.
///
/// ★ UNVERIFIED, AND DELIBERATELY SO (2026-08-05). Nothing has executed
/// this. The Metal driver cannot build on the machine we have —
/// `xcrun --find metal` fails, because the shader compiler ships with
/// full Xcode and only CommandLineTools is installed — so this text is
/// written against the driver's SOURCE, not against a running
/// deployment. It is boilerplate, requested as such, and it is here so
/// the shape exists to be corrected rather than invented under time
/// pressure later. `.wiki/tart/macos.md` rung 3 states the proof it
/// owes: the descriptors `crates/driver-metal/csrc/src/model/llama_like/declared_dag.hpp`
/// emits must come out unchanged.
///
/// WHAT IT IS FOR. Metal today consumes the SEMANTIC trace and chooses
/// its kernels in C++ (`decode_psos.cpp`), which is the same "the driver
/// decides" shape the CUDA side is being cured of, approached from the
/// other end. A backend with a text of its own can be read: this file
/// says what runs.
///
/// WHAT IS ALMOST CERTAINLY WRONG, so a reader does not mistake
/// plausibility for correctness:
///
/// * the M>1 (Prefill) lane states one GEMM and one paged attention, and
///   that was written down as a guess against `MultiBatchPsos`'s split-k,
///   fp16-precast, strided and bias variants. Checked 2026-08-10: it is
///   **correct for what runs**. The live `MbFeatures` on this family is
///   `{ gdn, sdpa_d256 }` — every one of those variants is `false` in every
///   live path and set true only in `psos_mb.rs`'s `all_features()` test
///   fixture, and `.wiki/driver/progress-metal.md` records them as deferred with reasons
///   ("with split-K deferred every dispatch is unsplit, which by the C++'s
///   OWN measurement makes `qmm_bn_unsplit` the right width"). `bias` is
///   gpt-oss's, and `routed` is a mixture this family does not model at all.
///   So the driver carries rungs nothing turns on; the text states the lane
///   that fires. What remains untested is the `kQmmMinBatch` gate, which the
///   text takes as the load-time fact `qmm_multi_batch` rather than deciding.
/// * ~~`sdpa_*_d_256` pins head_dim 256~~ — **fixed 2026-08-10, and it was a
///   real defect rather than a simplification.** `dsl::metal::sdpa` spelled
///   the width as a literal, so this family — whose heads are 128 wide —
///   named a 256-wide attention kernel. That does not fault: it reads past
///   the end of every head and answers with whatever is there, which is the
///   same defect `.wiki/driver/progress-metal.md` records in the C++ llama walk, where
///   `_d128` was a literal that strode 64-wide heads past their end. The
///   symbol now takes `head_dim`; a width no kernel instantiates simply does
///   not resolve, and the driver's row check reports it by name.
/// * no seams. The adapter, the two observation taps and the boundaries
///   are stated by the CUDA text and absent here, because none of the
///   machinery behind them exists on this backend yet.
/// * qk-norm and bias are stated as ordinary norms and are untested
///   against `declared_dag.hpp`'s expectations.
fn llama_like_metal_text(
    facts: &LlamaLikeFacts,
    metal: &LlamaLikeMetalFacts,
    class: FireClass,
) -> ForwardPlan {
    // The two lanes the Metal driver actually has: M=1 (the per-token
    // decode step) and M>1 (the multi-batch lane). `FireClass` is the
    // same instantiation index it is on CUDA.
    let multi_batch = class != FireClass::Decode;
    // The namespace, with the deployment's WEIGHT REPRESENTATION on it, for
    // the reason `llama_like_cuda` states at length: `facts.shape()` answers
    // `Bf16` because the semantic facts carry no backend, and every handle
    // `m.layer(l)` hands out is built from this one answer — which is why no
    // projection below spells a repr and none can spell a different one.
    let shape = dsl::ModelShape {
        proj_repr: metal.proj_repr,
        ..facts.shape()
    };
    dsl::trace_metal("llama_like", &shape, class, |m| {
        // The depth axis, and it is stated unconditionally here where the
        // CUDA text gates it on deployment facts. The gate exists there
        // because a padded deployment stages q/k at PHYSICAL width while a
        // row window addresses at LOGICAL width, so half the axis is
        // unservable. Metal has neither padding fact nor an XQA path — its
        // attention takes `head_dim` as an operand since the `_d_256` fix —
        // so both halves are free, and the argument the CUDA comment makes
        // for the narrowing half ("stopping after layer `k` addresses
        // nothing at all, because the retired ops simply do not run")
        // applies to the whole of it.
        //
        // This is the ONE statement that makes the text polymorphic on
        // depth: every layer-tagged op below becomes implicitly
        // `rows(depth > layer)`, so a fire whose rows truncate at different
        // layers lowers to rectangles that narrow rather than to one
        // rectangle per op. `driver-metal/tests/polymorphism.rs`
        // measures it.
        m.depth_window();

        let f = facts.clone();
        // The affine entrypoints are instantiated over (dtype x group x bits),
        // so every statement below names its POINT and not the stem. A stem
        // does not resolve, and the runtime compiler says so by listing what
        // the shader exports — which is the failure worth having, because a
        // WRONG point compiles and reads the wrong bytes (the `_d_256` defect,
        // one axis over).
        let point = dsl::metal::affine_point(metal.proj_repr, metal.affine_bits);
        // The GEMM carries its tile too — see `LlamaLikeMetalFacts::qmm_tile`
        // for why a tile is a load-time fact and not a fire-time one.
        let gemm_point =
            dsl::metal::affine_gemm_point(metal.proj_repr, metal.affine_bits, metal.qmm_tile);
        // The attention widths are per LAYER now and derived inside the loop:
        // gemma-4's full-attention layers are a different shape from its
        // sliding ones, so a fire-wide `q_width()` is only right for a stack
        // that states one.
        let post_norm = f.norm_placement == NormPlacement::Post;
        // gemma's four-norm block. `post_norm` stays false under it — the
        // stream is normed on the way IN as well, so the input side reads
        // exactly like `Pre` and only the output side is new.
        let sandwich = f.norm_placement == NormPlacement::Sandwich;

        // The projection this deployment takes: MLX's steel GEMM above the
        // batch gate, the GEMV below it.
        //
        // A GUARD and not a Rust `if`, because the condition is the FIRE's and
        // not the deployment's. `qmm_t.metal` has no `M` argument -- its
        // header says the driver only selects it when `M % BM == 0`, so the
        // row count lives in the grid and every tile is full -- and the
        // narrowest tile is `qmm_tile.0`. A prefill of fewer rows than that
        // cannot take the GEMM at all.
        //
        // The driver used to paper over it, and both ways were measured wrong
        // against a real checkpoint: handing the GEMM's symbol the matvec's
        // grid gave NaN, and rounding the row axis up gave a finite wrong
        // answer plus a tile's worth of overrun into the next value.
        // `Rule::Qmm` refuses now (`Ungeometric::PartialTile`), which is the
        // right answer for a driver and leaves the choice here -- where a
        // choice between two kernels belongs.
        //
        // A GUARD, not a Rust `if`, because the condition is the FIRE's.
        //
        // `qmm_t.metal` has no `M` argument -- its header says the driver only
        // selects it when `M % BM == 0`, so the row count lives in the grid
        // and every tile is full -- and the narrowest tile is `qmm_tile.0`. A
        // prefill of fewer rows cannot take the GEMM at all, and a Rust `if`
        // resolves at trace time and would leave it with nothing to run.
        // `Rule::Qmm` refuses such a fire (`Ungeometric::PartialTile`), which
        // is the right answer for a driver and leaves the choice here.
        //
        // It took both halves of a mechanism whose doc had described it for
        // longer than either half existed. `guarded_value`: *"each region's
        // launches are their lowerings, binding the same output buffer and
        // recording no SSA outputs of their own."* The arms record no output
        // (`dsl::metal`'s projections ask `inside_value_region`, as
        // `seam::attn_at` always has) and the lowering binds the guard's to
        // them (`Lowering::region_outs`). Missing either one is a silent wrong
        // answer, and both were measured on a real checkpoint: without the
        // first the KV pool held q in its K pages, without the second every
        // projection wrote zeros over its own input.
        //
        // `TokensMultipleOf(tile)` and not `TokensGT(tile - 1)`, which is
        // what this guard said for as long as it existed. The quoted
        // precondition three paragraphs up is `M % BM == 0`, and a threshold
        // does not test it: 32 rows is two whole 16-row tiles and 35 is not,
        // and both are `> 15`. So every count above the tile that the tile
        // does not divide -- fifteen in sixteen -- reached an arm no driver
        // can launch, and a real `pie run` of a 35-token prompt died on
        // `PartialTile { rows: 35, tile: 16 }` on Metal, Vulkan and wgpu
        // alike. The threshold needed no companion and neither does this: for
        // a non-zero token count, `N % tile == 0` already implies `N >= tile`.
        //
        // Stated POSITIVELY, still: the arm that runs on the common path is
        // the one that wants to be read first.
        let tile = metal.qmm_tile.0.max(1);
        // The POINT is a parameter because one tensor in this stack may not
        // share it: see `LlamaLikeMetalFacts::router_repr`. Everything else
        // -- the guard, its two arms, the shape the value takes -- is the
        // same for every projection, and threading a point rather than
        // writing a second closure is what keeps it so. A router that took
        // `qmv` unguarded wrote row zero of a three-row prefill and left the
        // rest as the arena's zeros, which reads as a model with no
        // distribution rather than as a wrong one.
        // `gemm_fits` is the COLUMN half of the tiled GEMM's contract, which
        // the guard below cannot state: `TokensMultipleOf` is about the fire
        // and `qmm_t.metal` asks for `N % BN == 0` too. See its doc.
        // The FP16 staging pass, once per ACTIVATION and not once per
        // projection.
        //
        // q, k and v are three projections of one normed activation and
        // gate/up are two of another, so a cast per projection would pay
        // five times for two conversions. The C++ driver this replaced spells
        // the same grouping as a kind test -- `llama_fp16_cast_before` is
        // true for q, o, gate and down and false for k, v and up -- and a
        // memo on the value's identity says it without a kind list, because
        // the text already knows which projections share an `x`: they were
        // handed the same one.
        //
        // Keyed on the SSA id and never cleared. Ids are unique for the whole
        // trace, so an entry cannot be stale; the map is as long as the model
        // has distinct projection sources, which is two per layer.
        //
        // OUTSIDE the guard, deliberately. A guard arm records no value of
        // its own -- `region_out` returns `None` inside one and every launch
        // in the arm binds the GUARD's output buffer -- so a cast written
        // there would write half-precision activations over the projection's
        // result. The cast has a shape of its own, so it needs a statement of
        // its own, and that statement has to be outside.
        //
        // Which means a decode pays for it. At one token the guard is false
        // and every projection takes the `qmv` arm, which reads the bf16 `x`
        // and never looks at the half-precision copy -- so a 28-layer decode
        // records 112 casts nothing reads, 20% of the step's fires.
        //
        // That is worth knowing and NOT worth fixing AT BATCH ONE, which is
        // the unusual part. `driver-vulkan` was made to skip exactly these
        // fires and the step did not move: 2.769 ms against 2.777, inside the
        // spread of either. Every one of them shares a stage with the
        // projection beside it, and a fire that shares a stage with another
        // fire is free -- see the `hazards` doc in
        // `driver-vulkan/src/device.rs` for why the currency is the stage and
        // not the fire. Giving the cast a guard of its own would buy host
        // time and arena pressure, both already small, and no device time.
        //
        // # AT BATCH EIGHT IT IS REAL AND STILL DEAD -- SO IT IS NOW GUARDED
        //
        // The paragraph above was measured at one token and is true there and
        // nowhere else. A batched decode has M>1, so it is planned on the
        // PREFILL lane -- and this cast is on that lane in earnest, 112 fires
        // a step.
        //
        // And nothing reads them there either. `affine_qmm_t_*` fires ZERO
        // times at batch eight, because the projection guard is
        // `TokensMultipleOf(32)` and no batch a scheduler gathers is a
        // multiple of 32, so every projection takes the `qmv` arm exactly as
        // it does at one token -- while the cast, being outside the guard,
        // fires anyway. At batch one the waste is free because the stage
        // hides it. At batch eight the cast is doing real work on eight rows
        // and there is no stage to hide behind.
        //
        // So the fix that was correctly declined at batch one is taken here,
        // and the shape of it is the same as the problem: the cast is emitted
        // only when the GEMM arm can actually run, which is the guard's own
        // predicate. It cannot go INSIDE that arm for the reason three
        // paragraphs up, so it gets a guard of its OWN carrying the same
        // predicate -- `dsl::metal::cast_qmm_input_when`, whose `otherwise`
        // region is empty because nothing reads the buffer when the predicate
        // fails.
        //
        // What it is worth, A/B on the shipped build, two runs each,
        // interleaved: -0.029 ms at batch one, -0.066 at two, -0.164 at four,
        // -0.107 at eight. Reproducible and outside the spread, and three to
        // five times SMALLER than the per-symbol timestamps predicted -- they
        // charge two timestamps a dispatch and so overprice 112 short fires,
        // and some of these casts still share a stage with the projection
        // beside them even at batch eight. The full correction is in
        // `device.rs`'s `hazards` doc under "where the batched step's extra
        // 5.04 ms actually goes".
        let staged: std::cell::RefCell<
            std::collections::HashMap<model_ir::trace::ValueId, Val>,
        > = std::cell::RefCell::new(std::collections::HashMap::new());
        let stage = |x: &Val| -> Val {
            if let Some(v) = staged.borrow().get(&x.key()) {
                return v.clone();
            }
            let v = dsl::metal::cast_qmm_input_when(x, GuardPred::TokensMultipleOf(tile));
            staged.borrow_mut().insert(x.key(), v.clone());
            v
        };
        // Whether the batched arm multiplies in `half`. See
        // `LlamaLikeMetalFacts::qmm_fp16_precast`.
        let precast = metal.qmm_fp16_precast;
        // `staged` is the POINT's and not the deployment's, for the same
        // reason `pt` is a parameter: one projection in this stack may not
        // share the stack's codec. The router gate is 8-bit where the
        // projections are 4 (`LlamaLikeMetalFacts::router_repr`), and
        // `affine_qmm_t_fp16_precast` is stamped at `gs = 64, b = 4` alone --
        // so a text that staged by deployment named
        // `affine_qmm_t_fp16_precast_bfloat16_gs_64_b_8_bm_32_bn_32` for the
        // gate. `builder`'s signature check refused it, which is the failure
        // worth having and the one that made this a parameter.
        let gemm_at = |x: &Val, w: &MatW, pt: &str, gpt: &str, staged: bool| {
            if !(multi_batch
                && metal.qmm_multi_batch
                && dsl::metal::gemm_fits(w.width, metal.qmm_tile))
            {
                return dsl::metal::qmv(x, w, pt);
            }
            let shape = (Shape(vec![Dim::Tokens, Dim::Const(w.width)]), DType::BF16);
            let half = staged.then(|| stage(x));
            let (g, v) = dsl::guarded_value(x.trace(), w.layer, shape);
            g.arm(GuardPred::TokensMultipleOf(tile), || match &half {
                Some(h) => {
                    dsl::metal::qmm_fp16(h, w, gpt);
                }
                None => {
                    dsl::metal::qmm(x, w, gpt);
                }
            })
            .otherwise(|| {
                dsl::metal::qmv(x, w, pt);
            });
            v
        };
        let gemm = |x: &Val, w: &MatW| gemm_at(x, w, &point, &gemm_point, precast);
        // The residual-fused twin, guarded the same way and for the same
        // reason: `affine_qmm_t_residual` is that tiling with an epilogue.
        let gemm_add = |x: &Val, w: &MatW, residual: &Val| {
            if !metal.fuse_residual_gemv {
                return dsl::metal::residual_add(&gemm(x, w), residual);
            }
            if !(multi_batch
                && metal.qmm_multi_batch
                && dsl::metal::gemm_fits(w.width, metal.qmm_tile))
            {
                return dsl::metal::qmv_residual(x, w, residual, &point);
            }
            let shape = (Shape(vec![Dim::Tokens, Dim::Const(w.width)]), DType::BF16);
            let half = precast.then(|| stage(x));
            let (g, v) = dsl::guarded_value(x.trace(), w.layer, shape);
            g.arm(GuardPred::TokensMultipleOf(tile), || match &half {
                Some(h) => {
                    dsl::metal::qmm_residual_fp16(h, w, residual, &gemm_point);
                }
                None => {
                    dsl::metal::qmm_residual(x, w, residual, &gemm_point);
                }
            })
            .otherwise(|| {
                dsl::metal::qmv_residual(x, w, residual, &point);
            });
            v
        };
        // ALWAYS paged, and the class is not the question -- the POOL's
        // layout is. `model::kv::Pool` allocates `[page, token, head, dim]`
        // for every fire this driver runs, so a decode that names the
        // contiguous `kv_append`/`sdpa_vector_decode` walks a paged pool with
        // a contiguous kernel's arithmetic: it reads real memory at every step
        // and attends to the wrong tokens.
        //
        // The two also disagree about what the scalars MEAN. The paged row
        // reads `Param(1)` as `n_kv_heads` and the contiguous row reads it as
        // `n`, the key count, so one statement cannot supply both correctly.
        // Naming one variant everywhere is what makes the statement's scalars
        // answerable at all.
        //
        // Same shape as the gather it sits beside: `multi_batch` is a fact
        // about the FIRE and this is a fact about the DRIVER's allocation.
        let _ = metal.paged_multi_batch;
        let paged = true;
        // The gated MLP. `silu_mul` takes gate and up as TWO buffers, so a
        // deployment whose loader did not join them states two projections —
        // which on this backend is every deployment, because
        // `compile_load_plan` authors with `Projections::InPlace` and the join
        // declines under it. The packed arm stays for one that does.
        assert!(
            !metal.gate_up_fused,
            "llama_like's Metal text has no packed gate\u{2016}up arm: `silu_mul` \
             takes two buffers and no Metal kernel splits a packed bank into \
             them. No deployment needs one -- `compile_load_plan` authors with \
             `Projections::InPlace` and the join declines under it -- so the \
             arm is refused at trace time rather than written untested."
        );
        // The FFN, dense or routed, and the branch is a FACT rather than a
        // family. A llama-like architecture with a mixture is still llama-like
        // -- the attention above is untouched and only the block between the
        // two norms differs -- which is the tart argument stated as code: one
        // supergraph, more polymorphism, no second text.
        //
        // The mixture is six statements because a routed FFN's SHAPE depends
        // on a value the fire computes. The router picks experts; the sort
        // groups rows by the expert they picked; the gather materialises those
        // groups contiguously; the matmuls run over the groups; the combine
        // puts the rows back weighted by the router's confidence. The executor
        // walks all six exactly as it walks a projection -- symbol, row, file,
        // rule, grid, operands -- and `RouteRows`/`RoutedQmv` read the expert
        // counts off the dims the same way `Qmv` reads `width`.
        //
        // # The width this returns, which is NOT the same on both arms
        //
        // Dense returns the activated INTERMEDIATE, and the caller owes the
        // down projection -- which it fuses into the residual add. Routed
        // returns HIDDEN, because the down projection is per-expert and
        // happens inside, before the combine puts the rows back.
        //
        // Two widths behind one name is a trap, and it sprang: two of the
        // three call sites applied `w.down` to a value that had already been
        // down-projected. That is a phantom dense matmul at the wrong width
        // under a tensor name no mixture checkpoint publishes -- so every
        // mixture was wrong, and nothing said so, because no mixture had ever
        // been held to a checkpoint. `owes_down` below is the fact the sites
        // now ask instead of assuming.
        // `router_in` is the value the ROUTER projects, which need not be the
        // one the experts read. For every routed family but gemma-4 it is the
        // same value and the two arguments are handed the same `Val`; gemma-4
        // routes off the post-attention stream and feeds its experts a
        // separately-normed copy of it, so a text with one argument here
        // could only be right for one of them.
        // WHICH activation, and it is a symbol rather than a flag:
        // gpt-oss clamps the gate above only, clamps the linear branch
        // both ways and adds one to it, and dropping either produces a
        // model that runs and is wrong.
        let activate = |gate: &Val, up: &Val, width: u32| match metal.activation {
            Activation::SiluMul => dsl::metal::silu_mul(gate, up, width),
            Activation::SwiGlu { limit, alpha } => {
                dsl::metal::swiglu(gate, up, width, limit, alpha)
            }
            Activation::Geglu => dsl::metal::geglu(gate, up, width),
        };
        // The DENSE FFN's gate, up and activation, WITHOUT the down
        // projection -- see `owes_down` below for why that is the caller's.
        //
        // Hoisted out of `gated` because a `dense_beside_moe` layer needs
        // both this and the routed path in the SAME layer, and `gated`
        // returns one or the other. It was reachable only as `gated`'s
        // `n_experts == 0` arm, which a mixture row can never take.
        let dense_ffn = |x: &Val, w: &dsl::Layer| {
            activate(&gemm(x, &w.gate_proj), &gemm(x, &w.up_proj), f.intermediate)
        };
        let gated = |x: &Val, router_in: &Val, w: &dsl::Layer| {
            if f.n_experts == 0 {
                return dense_ffn(x, w);
            }
            let k = f.experts_per_token.max(1);
            // The router's logits, biased if the checkpoint publishes one.
            // BEFORE the top-k, which is the whole point: this bias moves a
            // ranking rather than an activation, so applying it afterwards
            // -- or not at all -- picks different experts.
            //
            // AT THE GATE'S OWN POINT, which need not be the dense one. This
            // is `bank` below for a matrix of thirty-two columns, and it is
            // the same mechanism because it is the same fact: `mlx_lm`
            // publishes gpt-oss's gate at 8 bits inside a 4-bit stack. The
            // difference is what getting it wrong looks like -- a bank read
            // at the wrong format is NaN, and a GATE read at the wrong width
            // is a fluent model routing to almost the right experts.
            //
            // `qmv` directly rather than through `gemm`: the gate is
            // `[hidden, n_experts]` and no GEMM tile is that narrow, so the
            // guard `gemm` builds would have an arm that can never run.
            // The router's input, normed at its OWN scale when the
            // checkpoint publishes one. gemma-4 does: `router.scale` is a
            // `[hidden]` RMS weight and the reference folds `hidden**-0.5`
            // into it, which `rms_norm` already applies.
            let router_x = if metal.router_input_norm {
                // `hidden**-0.5` folded into the gain, which is where the
                // reference has it: `rms_norm(x, self.scale * root, eps)`.
                // The scalar is nowhere in the checkpoint.
                dsl::metal::rms_norm_gain(
                    router_in,
                    &w.router_scale,
                    f.hidden,
                    metal.rms_eps,
                    (f.hidden as f32).powf(-0.5),
                )
            } else {
                router_in.clone()
            };
            let logits = match metal.router_repr {
                None => gemm(&router_x, &w.router),
                Some(repr) => gemm_at(
                    &router_x,
                    &dsl::MatW {
                        repr,
                        ..w.router.clone()
                    },
                    &dsl::metal::affine_point(repr, metal.router_bits),
                    &dsl::metal::affine_gemm_point(repr, metal.router_bits, metal.qmm_tile),
                    crate::shared::llama_like::project::qmm_fp16_precast(
                        match repr {
                            model_dsl::WeightRepr::Scaled { group, .. } => group,
                            _ => 0,
                        },
                        metal.router_bits,
                    ),
                ),
            };
            let logits = if f.router_bias && metal.add_bias {
                dsl::metal::add_bias(&logits, &w.router_bias)
            } else {
                logits
            };
            let (ids, weights) = dsl::metal::router_topk(
                &logits,
                f.n_experts,
                k,
                metal.router_expert_scale.then_some(&w.router_expert_scale),
                metal.norm_topk_prob,
            );
            // The SORTED stack: one row per `(token, expert-slot)` route,
            // grouped so that consecutive rows read the same expert bank.
            // Its height is a function of the FIRE, which is why nothing
            // here is a count -- `dsl::metal::route_sort` states the extent
            // and the lowering resolves it.
            //
            // `row_expert` is the operand the MATVEC takes. It is the sort's
            // answer to "which expert does sorted row `p` read", and it was
            // discarded here while the projections were handed the router's
            // `[Tokens, k]` choice instead -- a list indexed by PAIR being
            // read at a SORTED position.
            //
            // `tile_expert` is the GEMM's, and was discarded outright: it
            // was `_tile_expert` for as long as this text had no batched
            // arm to hand it to.
            //
            // WHICH ARM, and the block that goes with it. A decode routes
            // one token, so `k` rows over `n_experts` banks is one live row
            // in a tile of sixteen and the matvec is right. A prefill routes
            // `tokens * k`, and reading each expert's bank once per row
            // there is what held gpt-oss's prefill flat at 166 tok/s from 32
            // tokens to 2048. The sort's block must be the GEMM's row tile
            // or a tile spans two experts and gets one of their banks.
            //
            // `moe_tile` may also be `None`, which takes the matvec in a
            // prefill too. That is a family opting out of the batched arm,
            // not a tile choice; qwen3.6 is the one that does.
            let tile = if class == FireClass::Prefill {
                metal.moe_tile
            } else {
                None
            };
            let block = tile.map_or(dsl::metal::ROUTE_BLOCK_MATVEC, |t| t.0);
            let (perm, row_expert, tile_expert, inv) =
                dsl::metal::route_sort(&ids, f.n_experts, k, f.hidden, block);
            let rows = dsl::metal::route_gather(x, &perm, f.n_experts, k, f.hidden, block);
            // The bank's OWN format, which need not be the dense one --
            // gpt-oss stores 98 tensors affine/64/4, its expert banks
            // mxfp4/32, and its 24 router gates affine/64/**8**: three
            // formats, one checkpoint. See `LlamaLikeMetalFacts::moe_repr`
            // and `router_repr`.
            let bank = |m: &dsl::MatW| match metal.moe_repr {
                Some(repr) => dsl::MatW { repr, ..m.clone() },
                None => m.clone(),
            };
            let bits = if metal.moe_repr.is_some() {
                metal.moe_bits
            } else {
                metal.affine_bits
            };
            // ONE projection, either arm. The matvec's value is
            // `[Tokens, width * k]` and the GEMM's is `[stack, width]`, and
            // those are the same bytes in the same order whenever nothing is
            // padded -- which is exactly when the matvec runs.
            let project = |x: &Val, m: &dsl::MatW, in_vec: u32| {
                if let Some(tile) = tile {
                    dsl::metal::routed_qmm(
                        x,
                        &row_expert,
                        &tile_expert,
                        &bank(m),
                        f.n_experts,
                        k,
                        in_vec,
                        bits,
                        tile,
                        metal.routed_qmm_fp16 && bits == 4,
                    )
                } else {
                    dsl::metal::routed_qmv(x, &row_expert, &bank(m), k, in_vec, false, bits)
                }
            };
            // `k * moe_intermediate` and not `moe_intermediate` FOR THE
            // MATVEC: each of its values is a whole token's `k` expert
            // results end to end, and the activation between them is
            // elementwise over all of it. Told one result's width it covered
            // a `k`th of the stack.
            //
            // The GEMM's rows are already the stack, one result each, so its
            // width is one run. The activation reads which it got off the
            // operand's own row axis -- see `dsl::metal::rows_of`.
            //
            // The projections read one run at a time either way, and say so
            // with their own `in_vec`: `hidden` into the bank,
            // `moe_intermediate` out of it.
            let h = activate(
                &project(&rows, &w.expert_gate, f.hidden),
                &project(&rows, &w.expert_up, f.hidden),
                if tile.is_some() {
                    f.moe_intermediate
                } else {
                    f.moe_intermediate * k
                },
            );
            let routed = dsl::metal::combine_sorted(
                &project(&h, &w.expert_down, f.moe_intermediate),
                &weights,
                &inv,
                k,
                f.hidden,
            );
            if f.shared_intermediate == 0 {
                return routed;
            }
            // The dense expert a mixture may also have, blended in by a
            // per-row sigmoid gate.
            let shared = activate(
                &gemm(x, &w.shared_gate),
                &gemm(x, &w.shared_up),
                f.shared_intermediate,
            );
            dsl::metal::shared_expert_combine(
                &routed,
                &gemm(&shared, &w.shared_down),
                &gemm(x, &w.shared_gate_proj),
                f.hidden,
            )
        };

        // Whether the caller still owes the down projection after `gated` --
        // see the note on its two widths.
        let owes_down = f.n_experts == 0;

        // Whether this deployment's layers run BOTH FFNs. gemma-4's mixture
        // rows do; every other routed family replaces the dense MLP with the
        // routed one, and `f.n_experts > 0` is then enough to know which ran.
        let mixture_beside_dense = metal.dense_beside_moe && f.n_experts > 0;
        // And it is written in the SANDWICH arm only, because the reference
        // that defines it is a sandwich block: the two legs are joined
        // between `post_feedforward_layernorm` and the residual, and under
        // `Pre` or `Post` there is no such position. The branch used to sit
        // after all three arms and apply to any placement, which read as
        // generality and was arithmetic no reference states -- under `Pre`
        // it appended a second routed FFN to a layer that had already run
        // one, which is what
        // `gemmas_mixture_runs_beside_the_dense_mlp_rather_than_instead_of_it`
        // was asserting when it counted two routers.
        assert!(
            !mixture_beside_dense || sandwich,
            "a mixture beside the dense MLP is a SANDWICH block's shape and \
             this row states {:?}: there is no position under it for the \
             join, so a text that ran the branch anyway would be inventing \
             one",
            f.norm_placement
        );

        // The embedding, with gemma's `sqrt(hidden)` scale folded into the
        // gather when this deployment wants one. The scale is the statement's
        // — a kernel that knew it would be a kernel that knew the model.
        //
        // Asked of `embed_scale`, not of `per_layer_emb_dim`. The two agreed
        // for as long as the only gemma this text served had per-layer
        // embeddings; gemma-4-31b has none and is still a gemma, and reading
        // the scale off the side network's width silently dropped it.
        let mut y = if metal.embed_scale > 0.0 {
            dsl::metal::embed_gather_scaled(
                m.trace(),
                "embed",
                f.hidden,
                multi_batch,
                metal.proj_repr,
                &point,
                metal.embed_scale,
            )
        } else {
            dsl::metal::embed_gather(
                m.trace(),
                "embed",
                f.hidden,
                multi_batch,
                metal.proj_repr,
                &point,
            )
        };

        // ── gemma's PLE prologue, once per step and layer-less. ──
        //
        // A SECOND embedding table gathered, projected, normed and joined into
        // `[n_layers, ple_dim]`, which each layer then reads its own slice of.
        // Four statements, before the stack, and the reason gemma4 is a family
        // where qwen3-moe and gpt-oss were fixtures: nothing llama-like has a
        // counterpart to a side network.
        let ple = (metal.per_layer_emb_dim > 0).then(|| {
            let block = f.layers * metal.per_layer_emb_dim;
            let token = dsl::metal::embed_gather_scaled(
                m.trace(),
                "ple_embed",
                block,
                multi_batch,
                metal.proj_repr,
                &point,
                1.0,
            );
            let proj = dsl::metal::qmv(
                &token,
                &dsl::MatW {
                    name: "ple_proj".to_string(),
                    width: block,
                    layer: None,
                    repr: metal.proj_repr,
                },
                &point,
            );
            let normed = dsl::metal::rms_norm(
                &proj,
                &dsl::NormW {
                    name: "ple_proj_norm".to_string(),
                    variant: f.norm_variant,
                    per_head: None,
                    layer: None,
                },
                metal.per_layer_emb_dim,
                metal.rms_eps,
            );
            dsl::metal::ple_combine(&normed, &token, block)
        });

        for l in 0..f.layers {
            let w = m.layer(l);

            let x = if post_norm {
                y.clone()
            } else {
                dsl::metal::rms_norm(&y, &w.attn_norm, f.hidden, metal.rms_eps)
            };

            // A KV-SHARED layer rotates its own Q and reads the pages its
            // source wrote: no k/v projection, no k/v norm, no append.
            //
            // Suppressing those statements is not an optimisation — it is
            // which tensors the checkpoint SHIPS. A shared layer has no
            // `k_proj` weight at all, so a text that stated one would name a
            // tensor that is not there and the load would say so.
            //
            // gemma's, and the layers that share are the LAST `kv_shared`
            // of the stack.
            let shares_kv = l >= f.layers.saturating_sub(metal.kv_shared_layers);
            // The window this layer attends over, `-1` for all of it. Bound
            // here rather than at the attention because the PROJECTIONS need
            // it: gemma4's full-attention layers are the ones that take V from
            // K, and `window < 0` is that test.
            let window = metal.window_left_at(l);
            // THIS layer's attention shape, which is not the stack's on every
            // deployment. gemma-4 states two: its full-attention layers are
            // twice as wide per head as its sliding ones and carry a quarter
            // the KV heads, and its own tensors say so -- on the 31b, layer
            // 0's `q_norm` is `[256]` and layer 5's is `[512]`.
            //
            // Keyed on `window` rather than on a second per-layer list, the
            // same way `v_from_k` and `rope_theta_at` are: "does this layer
            // attend everything" is already answered, and two lists are two
            // chances to disagree.
            //
            // Zero on `global_head_dim` means one shape for the whole stack,
            // which is every family here but gemma-4 -- so these read exactly
            // as `f.head_dim`/`f.kv_heads` did for everyone else.
            let head_dim = metal.head_dim_at(l, f.head_dim);
            let kv_heads = metal.kv_heads_at(l, f.kv_heads);
            let q_w = f.q_heads * head_dim;
            let kv_w = kv_heads * head_dim;
            // THIS layer's projections, at this layer's widths.
            //
            // `M::layer` builds every handle from `ModelShape`'s single
            // `q_width`/`kv_width`, because for every family but gemma-4 a
            // stack has ONE attention shape. gemma-4 has two, and the widths
            // are not decoration: `MatW::width` is the row length the lowered
            // qmv is dispatched over, so a full-attention layer given the
            // sliding width tells the kernel to read 4096 rows out of a
            // 2048-row `k_proj` -- 2048 rows past the end of the tensor, into
            // whatever the loader staged next.
            //
            // Measured on gemma-4-31b before this: layer 17's k_proj lowered
            // to `@151552w4096` and the fire's first NaN was at element 2048
            // of exactly that value, while the SAME layer's sdpa lowered to
            // `_d_512` and its o_proj to `w16384`. One layer disagreed with
            // itself about its own shape, because half of it asked
            // `head_dim_at` and half asked the namespace.
            //
            // Overridden here rather than taught to `ModelShape`, for the
            // reason `bank` overrides `repr` twenty lines up: the width is a
            // fact of the LAYER, the namespace is a fact of the model, and
            // the numbers this needs are the two lines above -- already
            // derived, from the one question (`window < 0`) both per-layer
            // facts are keyed on.
            let at_w = |m: &dsl::MatW, width: u32| dsl::MatW { width, ..m.clone() };
            let (q_proj, k_proj, v_proj) = (
                at_w(&w.q_proj, q_w),
                at_w(&w.k_proj, kv_w),
                at_w(&w.v_proj, kv_w),
            );
            // THREE projections, never one packed bank, and not as a
            // choice: `compile_load_plan` authors every Metal load with
            // `Projections::InPlace`, so the MLX path publishes
            // `self_attn.{q,k,v}_proj` separately and no Metal deployment
            // has a `qkv` handle at all -- `lowering::resolve` says so in
            // those words.
            //
            // This used to be a branch on a `qkv_fused` fact, and the
            // fact cost a whole checkpoint. `driver-metal/src/model/text.rs`
            // built `LlamaLikeFacts` itself and answered it from the staged
            // tensors; deleting that in favour of the catalog left the
            // ROW's answer reaching this text, and the row's answer is
            // CUDA's -- `LlamaLikeFacts::fused_qkv` says `true` on all
            // eight llama-3 rows and its own doc calls it "a *binding*
            // fact, not an architecture fact". The text asked for
            // `layer.0.qkv` and got `Unbound { symbol:
            // "affine_qmv_fast_bfloat16_gs_64_b_4", why:
            // UnknownWeight("layer.0.qkv") }` on llama-3.2-1B.
            //
            // The repair at the time was a second field that every
            // projection stated `false`; a field only ever written one way
            // is the same fact with a way to get it wrong still attached.
            let (q, k, v) = if shares_kv {
                // Q only. `k` and `v` stand for the source layer's pages,
                // which the attention reads through the pool rather than as
                // operands, so the values here are never consumed.
                let q = gemm(&x, &q_proj);
                (q.clone(), q.clone(), q)
            } else if metal.v_from_k && window < 0 {
                // A K-EQ-V layer projects K and takes V FROM it, so it ships
                // no `v_proj` tensor at all. gemma4's FULL-attention layers
                // are the ones that do — `window < 0` is that test, and it is
                // the same list `window_left` already states rather than a
                // second one that has to agree with it. The two norms then run in the
                // other order: V reads the projection K's norm is about to
                // overwrite, so V goes first.
                let q = gemm(&x, &q_proj);
                let k = gemm(&x, &k_proj);
                (q, k.clone(), k)
            } else {
                (gemm(&x, &q_proj), gemm(&x, &k_proj), gemm(&x, &v_proj))
            };
            // Qwen-2 family qkv biases: on the raw projections, before norms
            // and rope -- the semantic text's position and the CUDA text's.
            //
            // Gated on the DEPLOYMENT and not on the model alone, which is
            // the whole of what `add_bias` says: whether the biases exist is
            // `f.qkv_bias`, and it has answered `true` for Qwen-2 since the
            // row was written. This text stated no bias for anyone anyway,
            // because no Metal-side kernel added one -- so every Qwen-2
            // served through it computed its q/k/v without them. Nothing
            // downstream could tell: the biases are small and the text stays
            // fluent, which is why the gap survived until a driver compared
            // a whole distribution against a CPU oracle.
            //
            // A deployment that says `false` gets exactly the text it got
            // before. That is the conservative half of the same claim -- see
            // `LlamaLikeMetalFacts::add_bias` for why `driver-metal` still
            // says it.
            let (q, k, v) = if f.qkv_bias && metal.add_bias {
                (
                    dsl::metal::add_bias(&q, &w.q_bias),
                    dsl::metal::add_bias(&k, &w.k_bias),
                    dsl::metal::add_bias(&v, &w.v_bias),
                )
            } else {
                (q, k, v)
            };
            // The per-head q/k norm and the rotation that always follows it,
            // as ONE dispatch. Four conditions, and not one of them is a
            // preference:
            //
            // `metal.fused_qk_rope` is the deployment saying it has the
            // kernel. Only `driver-vulkan` does.
            //
            // `QkNorm::PerHead` because the fused kernel norms a HEAD -- its
            // base is `row * row_pitch + head * axis_size`. The `Global`
            // convention norms the whole row against a `[heads * head_dim]`
            // weight and is different arithmetic, not a different shape.
            //
            // `!rope_freq_table` because a rescaled ladder is a TABLE, and the
            // arm of the fused family that reads one is compiled and has no
            // routine. llama-3's piecewise rescaling and YaRN's are not bases.
            //
            // `!k_is_v` is the one that would corrupt an answer rather than
            // refuse a launch. A gemma-4 full-attention layer takes V FROM the
            // K projection and both values name the same buffer; it works
            // today only because the separate norm is OUT of place, so K's
            // normed value is a new allocation and V still names the raw
            // projection. The fused kernel is in place -- it has to be, the
            // rotation reads what the norm just wrote -- so fusing there would
            // norm AND rotate V as a side effect of doing it to K. Nothing
            // downstream would say so: V would still be a plausible tensor.
            let k_is_v = metal.v_from_k && window < 0;
            let fused_qk_rope = metal.fused_qk_rope
                && f.qk_norm == QkNorm::PerHead
                && !metal.rope_freq_table
                && !k_is_v;
            let (q, k) = if fused_qk_rope {
                let rotary = metal.rotary_dim_at(l, f.head_dim);
                let theta = metal.rope_theta_at(l);
                (
                    dsl::metal::rms_rope(
                        &q,
                        &w.q_norm,
                        head_dim,
                        metal.rms_eps,
                        theta,
                        1.0,
                        rotary,
                    ),
                    dsl::metal::rms_rope(
                        &k,
                        &w.k_norm,
                        head_dim,
                        metal.rms_eps,
                        theta,
                        1.0,
                        rotary,
                    ),
                )
            } else if f.qk_norm == QkNorm::Off {
                (q, k)
            } else {
                (
                    dsl::metal::rms_norm(&q, &w.q_norm, head_dim, metal.rms_eps),
                    dsl::metal::rms_norm(&k, &w.k_norm, head_dim, metal.rms_eps),
                )
            };
            // V's own norm, which has no weight and so no tensor to be found
            // by. See [`LlamaLikeMetalFacts::v_norm`]: gemma-4 norms V per
            // head before it reaches the pool, and on a k-eq-v layer `v` above
            // is deliberately the PROJECTION rather than `k`'s normed value.
            //
            // Before the rope and the append, because the pool stores what
            // attention will read and V is never rotated.
            let v = if metal.v_norm && !shares_kv {
                dsl::metal::vnorm(&v, head_dim, metal.rms_eps)
            } else {
                v
            };
            // One dispatch for q and k together, as `declared_dag.hpp`'s
            // `Kind::Rope` states it.
            // A deployment that RESCALES its ladder takes the table form:
            // llama-3's piecewise rescaling and YaRN's are not bases, so no
            // theta expresses them. The driver derives the table at load and
            // answers it; the text only says which form.
            let (q, k) = if fused_qk_rope {
                // Already rotated, by the statement that normed them.
                (q, k)
            } else {
                dsl::metal::rope(
                    &q,
                    &k,
                    multi_batch,
                    metal.rope_theta_at(l),
                    1.0,
                    head_dim,
                    // The rotation's EXTENT, which is not always the head.
                    // gemma-4 rotates a quarter of each full-attention head
                    // and all of each sliding one, so this is per layer like
                    // the shape above -- and it reaches the GRID rather than
                    // the kernel, through the row's `grid_param`.
                    metal.rotary_dim_at(l, f.head_dim),
                    metal.rope_freq_table,
                )
            };
            // A shared layer appends nothing: its source already did.
            if !shares_kv {
                dsl::metal::kv_append(&k, &v, &w.kv, paged, head_dim, kv_heads);
            }
            // The attention SINK this layer has, if any: a per-head learned
            // logit that joins the softmax without a value behind it.
            // gpt-oss's, and a deployment without them names none.
            let sink = metal.attn_sinks.then(|| format!("layer.{l}.attn_sinks"));
            let attend = |mb: bool| {
                dsl::metal::sdpa(
                    &q,
                    &w.kv,
                    q_w,
                    head_dim,
                    paged,
                    f.q_heads / kv_heads.max(1),
                    kv_heads,
                    window,
                    // The attention SINK this layer has, if any: a per-head
                    // learned logit that joins the softmax without a value
                    // behind it. gpt-oss's, and a deployment without them
                    // names none.
                    sink.as_deref(),
                    metal.attn_scale,
                    mb,
                )
            };
            // A BATCHED DECODE IS NOT A SMALL PREFILL, and until this guard
            // the text had no way to say so. `multi_batch` is `class !=
            // Decode`, so eight sequences each advancing by one token are
            // planned on the prefill lane and reach `sdpa_paged_tiled` -- a
            // 32-row query tile holding eight rows that belong to eight
            // DIFFERENT sequences with eight different key runs, so the tile
            // shares nothing and pays its staging for nothing. Measured at
            // 50.63 us a fire against the decode pair's 15.09 at batch eight,
            // and 644.98 at a long context.
            //
            // `GuardPred::WindowOne` is exactly the missing question -- "is
            // every row a one-token query window", which is what
            // `FireClass::Decode` meant and could not say about a fire it did
            // not classify. A mixed fire answers false and takes the tiled
            // arm, which serves a one-token row as its degenerate case, so
            // the fallback is correct rather than merely safe.
            //
            // The projections are NOT guarded this way and must not be: their
            // gate is `TokensMultipleOf`, a question about the tile, and a
            // batched decode fails it for a real reason. This one is about
            // the SHAPE OF THE QUERY WINDOW, which the batch does not change.
            let a = if multi_batch {
                let shape = (Shape(vec![Dim::Tokens, Dim::Const(q_w)]), DType::BF16);
                let (g, v) = dsl::guarded_value(q.trace(), Some(l), shape);
                g.arm(GuardPred::WindowOne, || {
                    attend(false);
                })
                .otherwise(|| {
                    attend(true);
                });
                v
            } else {
                attend(false).expect("a plain attention statement produces its value")
            };

            // The attention landing, and the bias the checkpoint may publish
            // on it. Stated once and used by all three norm arms, because
            // the arms differ in what they NORM and not in what they land.
            //
            // No bias here, and that is checked rather than assumed:
            // only the FUSED arm below adds one, and a row that both
            // publishes `o_bias` and reaches this closure is refused by
            // `NO_METAL_NORMED_LANDING_BIAS` before any of this is
            // traced. A branch used to stand here that no row in the
            // catalog could take.
            let land = |a: &Val| gemm(a, &w.o_proj);
            if post_norm {
                // NORM AND LANDING IN ONE. Every `rms_norm` in this arm is
                // read by exactly one `residual_add` and by nothing else, and
                // `rms_residual` is that pair: the threadgroup that computed
                // the row's inverse RMS still holds every element of the row,
                // so the add costs one more load and no synchronisation. What
                // it saves is a DISPATCH, and three quarters of them carry a
                // barrier -- 845 of gemma-4's 1080. See `rms_norm_residual`.
                y = dsl::metal::rms_norm_residual(
                    &land(&a),
                    &w.attn_norm,
                    &y,
                    None,
                    f.hidden,
                    metal.rms_eps,
                );
                let h = gated(&y, &y, &w);
                let ffn = if owes_down { gemm(&h, &w.down) } else { h };
                y = dsl::metal::rms_norm_residual(
                    &ffn,
                    &w.mlp_norm,
                    &y,
                    None,
                    f.hidden,
                    metal.rms_eps,
                );
            } else if sandwich {
                // gemma's FOUR norms. The stream was normed on the way IN
                // (`x`, above, from `attn_norm`); each sub-layer's output is
                // normed again on the way OUT, and only then does the residual
                // add land it.
                //
                // Why this is its own arm and not `post_norm` with more
                // weights: `post_norm` reads the stream RAW into each
                // sub-layer, and gemma does not — it norms both ways. Folding
                // them would need a text that norms the input under one flag
                // and the output under another, which is two facts pretending
                // to be one.
                //
                // Nothing here can fuse the residual into the projection
                // (`gemm_add`): a norm sits between them. That is arithmetic,
                // not a missed optimisation.
                // See the `post_norm` arm above for why the pair is one
                // statement. gemma-4 is the row the kernel was written for.
                y = dsl::metal::rms_norm_residual(
                    &land(&a),
                    &w.post_attn_norm,
                    &y,
                    None,
                    f.hidden,
                    metal.rms_eps,
                );
                if mixture_beside_dense {
                    // ── gemma-4's MIXTURE layer: two FFNs, side by side. ──
                    //
                    // `mlx_lm/models/gemma4_text.py::DecoderLayer.__call__`:
                    //
                    //     h1 = post_ffn_norm_1(mlp(pre_ffn_norm(h)))
                    //     h2 = post_ffn_norm_2(experts(pre_ffn_norm_2(h),
                    //                                 router(h)))
                    //     h  = post_ffn_norm(h1 + h2) + residual
                    //
                    // THREE values read the post-attention stream `y`: each
                    // leg's input norm and the ROUTER. The routed leg does
                    // not route off its own normed input — a text that fed
                    // the router `x2` would pick different experts and stay
                    // fluent, which is the failure mode this whole family of
                    // facts exists to make impossible to write by accident.
                    //
                    // Nothing fuses. `owes_down` is not asked: the dense
                    // leg's down projection lands in `post_mlp_norm_1`, not
                    // in the residual, so the fused `gemm_add` the dense
                    // rows take would skip a norm.
                    let x1 = dsl::metal::rms_norm(&y, &w.mlp_norm, f.hidden, metal.rms_eps);
                    let g1 = gemm(&dense_ffn(&x1, &w), &w.down);
                    let h1 = dsl::metal::rms_norm(&g1, &w.post_mlp_norm_1, f.hidden, metal.rms_eps);

                    let x2 = dsl::metal::rms_norm(&y, &w.mlp_norm_2, f.hidden, metal.rms_eps);
                    let g2 = gated(&x2, &y, &w);

                    // The JOIN, then the norm, then the residual: three adds
                    // in the reference and two statements here, because the
                    // routed leg's OUT norm is what the join reads. Norming
                    // before the join would norm two values that are meant to
                    // be normed as a sum -- that is still true, and folding
                    // `h1` into `post_mlp_norm_2`'s epilogue does not do it.
                    let joined = dsl::metal::rms_norm_residual(
                        &g2,
                        &w.post_mlp_norm_2,
                        &h1,
                        None,
                        f.hidden,
                        metal.rms_eps,
                    );
                    y = dsl::metal::rms_norm_residual(
                        &joined,
                        &w.post_mlp_norm,
                        &y,
                        None,
                        f.hidden,
                        metal.rms_eps,
                    );
                } else {
                    let x = dsl::metal::rms_norm(&y, &w.mlp_norm, f.hidden, metal.rms_eps);
                    let h = gated(&x, &x, &w);
                    let ffn = if owes_down { gemm(&h, &w.down) } else { h };
                    y = dsl::metal::rms_norm_residual(
                        &ffn,
                        &w.post_mlp_norm,
                        &y,
                        None,
                        f.hidden,
                        metal.rms_eps,
                    );
                }
            } else {
                // The one arm that FUSES the residual into the landing, so
                // the bias cannot ride the same gemm. Added afterwards, which
                // is what the CUDA text does too (`y += matmul(...)` and then
                // `add_bias`) -- three addends into one accumulator, and the
                // two backends agree on the order.
                y = gemm_add(&a, &w.o_proj, &y);
                if f.o_bias && metal.add_bias {
                    y = dsl::metal::add_bias(&y, &w.o_bias);
                }
                let x = dsl::metal::rms_norm(&y, &w.mlp_norm, f.hidden, metal.rms_eps);
                let h = gated(&x, &x, &w);
                // Dense FUSES the down projection into the residual add, which
                // is one kernel instead of two over the widest activation in
                // the block. A mixture cannot: its rows were already projected
                // and combined, so all that is left is the add.
                y = if owes_down {
                    gemm_add(&h, &w.down, &y)
                } else {
                    dsl::metal::residual_add(&h, &y)
                };
            }

            // ── gemma's per-layer tail. ──
            //
            // This layer's slice of the PLE block, gated and projected back
            // into the residual stream. Four statements, and the middle one is
            // STRIDED because the gate is a narrow read out of a wide buffer.
            //
            // A deployment with no PLE takes the per-layer SCALAR instead —
            // one number per layer, from a buffer, because which layer is
            // running is the fire's and not the text's.
            if let Some(ple) = &ple {
                let gate = dsl::metal::qmv(
                    &y,
                    &dsl::MatW {
                        name: format!("layer.{l}.ple_gate"),
                        width: metal.per_layer_emb_dim,
                        layer: Some(l),
                        repr: metal.proj_repr,
                    },
                    &point,
                );
                let h = dsl::metal::geglu_strided(
                    &gate,
                    ple,
                    metal.per_layer_emb_dim,
                    metal.per_layer_emb_dim,
                    f.layers * metal.per_layer_emb_dim,
                );
                let back = dsl::metal::qmv(
                    &h,
                    &dsl::MatW {
                        name: format!("layer.{l}.ple_out"),
                        width: f.hidden,
                        layer: Some(l),
                        repr: metal.proj_repr,
                    },
                    &point,
                );
                y = dsl::metal::rms_norm_residual(
                    &back,
                    &w.mlp_norm,
                    &y,
                    Some(&back),
                    f.hidden,
                    metal.rms_eps,
                );
            } else if metal.per_layer_scalar {
                y = dsl::metal::layer_scalar(&y, &format!("layer.{l}.scalar"), f.hidden);
            }
        }

        let normed = dsl::metal::rms_norm(&y, &m.final_norm(), f.hidden, metal.rms_eps);
        // The SAMPLED rows, before the readout. A fire's stream is one row
        // per TOKEN and its readout is one distribution per REQUEST, so
        // something has to pick, and it is `Step::sampling_indices`.
        //
        // Absent, the readout read row 0 and a two-token prefill answered the
        // FIRST token's distribution -- exactly right, for a question nobody
        // asked. A decode of one token per request is unaffected, which is why
        // the decode gate agreed with MLX over it for as long as it did.
        let sampled = dsl::metal::sample_rows(&normed, f.hidden);
        let head = if f.tied_embeddings {
            "embed"
        } else {
            "lm_head"
        };
        let logits = dsl::metal::lm_head(&sampled, head, f.vocab, metal.proj_repr, &point);
        // The readout's softcap, for a deployment that has one. Named or not
        // named -- a cap large enough to do nothing is still a kernel run per
        // fire to compute the identity.
        //
        // OUT OF PLACE: `logit_softcap` takes distinct in and out buffers, so
        // the fire's answer is this value and not the one handed to it. The
        // exit seam below therefore names the CAPPED value -- stating the
        // uncapped one put the driver's read-out one buffer behind the
        // arithmetic, which is a distribution that is right except for the
        // last thing done to it.
        let logits = if metal.logit_softcap > 0.0 {
            dsl::metal::softcap(&logits, f.vocab, metal.logit_softcap)
        } else {
            logits
        };
        // The exit boundary. Without it the read-out had no name, and the two
        // places that needed one guessed the same guess in two dialects: the
        // reference gate took the widest arena region, and the engine seam
        // took nothing at all -- it ran the fire and dropped what the fire
        // computed. A text that states its exit lets `Lowered::readout` answer
        // both, and the guess is now held against the answer
        // (`text_conformance::every_text_says_where_its_answer_lands...`).
        dsl::seam(m.trace(), &dsl::seam::OUT, &[&logits], None);
    })
}

/// Trace the llama_like METAL text for one [`FireClass`]. See
/// [`llama_like_metal_text`] for what is and is not verified about it.
pub fn llama_like_metal(
    facts: &LlamaLikeFacts,
    metal: &LlamaLikeMetalFacts,
    class: FireClass,
) -> ForwardPlan {
    llama_like_metal_text(facts, metal, class)
}

/// The llama_like CUDA text (`.wiki/tart/dsl.md` ③): computation and
/// kernel choice together, on the dsl surface, for ONE backend. The
/// semantic text is [`llama_like`] — a separate text, because a model
/// file is written for a backend and "am I lowered?" is not a question a
/// body asks. The class arms run as ordinary trace-time matches beside
/// the fact arms, and what they choose is exactly what
/// `declared_forward.cpp` chooses at fire time today — the migration
/// deletes the C++ copy of these matches, not this one.
fn llama_like_cuda_text(
    facts: &LlamaLikeFacts,
    cuda: &LlamaLikeCudaFacts,
    class: FireClass,
) -> ForwardPlan {
    // The namespace, with the deployment's WEIGHT REPRESENTATION on it
    // (1b). `facts.shape()` answers `Bf16` because the semantic facts
    // carry no backend; the CUDA facts do, and every handle `m.layer(l)`
    // hands out is built from this one answer -- which is why no
    // projection below spells a repr and none can spell a different one.
    //
    // And with its SHARD widths. Sharding needs no vocabulary: a rank's
    // trace states ITS widths, so this text divides by `tp_size` the
    // way it divides by anything else, and every projection below reads
    // as it did. `hidden` does NOT divide -- the residual stream is
    // replicated, which is why the landings are collectives.
    let tp = cuda.tp_size.max(1);
    assert!(
        shard_divides(facts, tp),
        "llama_like states a shard per rank; this deployment's heads or \
         intermediate do not divide by tp_size"
    );
    let shape = dsl::ModelShape {
        proj_repr: cuda.proj_repr,
        q_width: facts.q_width() / tp,
        kv_width: facts.kv_width() / tp,
        intermediate: facts.intermediate / tp,
        ..facts.shape()
    };
    dsl::trace_cuda("llama_like", &shape, class, |m| {
        dsl::seam(m.trace(), &dsl::seam::IN, &[], None);
        // THIS RANK's facts: the widths the shard actually computes.
        // Everything below reads them as if the model were that size,
        // which is the whole of what sharding costs a text.
        let mut f = facts.clone();
        f.q_heads /= tp;
        f.kv_heads /= tp;
        f.intermediate /= tp;
        let q_w = f.q_width();
        let kv_w = f.kv_width();
        let post_norm = f.norm_placement == NormPlacement::Post;
        // The backend facts, readable only under the class this text is
        // being traced for — the `FireClass` match, spelled as a filter
        // so the arms below read as they did when the lowering arrived
        // through the context.
        let cuda_of = |class_want: FireClass| (class == class_want).then_some(cuda);

        // STRUCTURAL S-3, stated IN THE BODY (V2 rung ②; formerly the
        // post-trace paint-over the review named): a class declares the
        // depth axis exactly where its body can honour it — the same
        // deployment gate as the mask peel's. Recording assigns each
        // layer-tagged op's role from here on.
        //
        // PREFILL states it too, since the cutover's last decline class
        // was "truncated-prefill" and this was its whole cause. What a
        // truncated prefill needs is the cheap half of the axis: every
        // row sits at the same `k`, so the window STOPS after layer `k`
        // and narrows nothing. The expensive half — a UNION fire, where
        // full-depth rows sit beside truncated ones and the tail layers
        // run over a row prefix — needs the qo/kv CSRs narrowed with
        // them, and there is no prefill analogue of
        // `depth_prefix_decode_plan`. The trace cannot tell those apart
        // (`k` is a runtime input), so it states the axis and the
        // driver's eligibility test admits only the uniform case.
        //
        // `xqa_decode` is a decode-path property and gates the Decode
        // class only. `head_dim_padded` gates NEITHER, and that is the
        // same two-halves argument one step further: a padded deployment
        // stages q/k at PHYSICAL width while a row window addresses at
        // logical width, so it cannot serve the narrowing half — but
        // stopping after layer `k` addresses nothing at all, because the
        // retired ops simply do not run. The driver holds `k`, so the
        // driver is where that split gets decided; withholding the axis
        // here refused the free half along with the costly one.
        if cuda_of(FireClass::Decode).is_some_and(|c| !c.xqa_decode)
            || cuda_of(FireClass::Prefill).is_some()
        {
            m.depth_window();
        }

        // The fused decode-QKV arm's predicate: the model-fact terms
        // here, the load-time backend terms on the facts struct — term
        // for term the hand-written `fused_decode_qkv_post`
        // (declared_forward.cpp:465-479), written where it belongs.
        // The head width the attention kernels run at, or 0 when that
        // is the logical one. The single reading of the padding fact in
        // this text: the three pads and the strip below take their
        // shapes from it, so nothing re-derives a width.
        let pad_to = if cuda.head_dim_padded {
            assert!(
                cuda.head_dim_kernel > f.head_dim,
                "a padded deployment states the width its kernels run at"
            );
            cuda.head_dim_kernel
        } else {
            0
        };

        // The width the attention SCHEDULES are planned at, which is the width
        // the dispatch runs at: a padded deployment plans on the padded one, or
        // the plan and the launch disagree about the page.
        let plan_head_dim = if pad_to == 0 { f.head_dim } else { pad_to };

        let fused_post = cuda_of(FireClass::Decode).is_some_and(|c| c.decode_fused_post)
            && f.fused_qkv
            && f.qk_norm == QkNorm::PerHead
            && f.rope == RopeKind::Standard
            // The fused epilogue has no bias step (the hand-written
            // predicate's `!use_qkv_bias` term, stated here since the
            // build gate no longer excludes bias deployments).
            && !f.qkv_bias;

        let mut y = m.embed();

        // The fire's rope table: a VALUE the fused kernel consumes, built
        // once — not the hand-written `rope_table_ready` latch, and
        // hoisted where a once-per-fire launch belongs.
        let table = (fused_post && cuda_of(FireClass::Decode).is_some_and(|c| c.rope_table))
            .then(|| cuda::rope_standard_table(m.trace(), f.head_dim));

        for l in 0..f.layers {
            let w = m.layer(l);

            // Attention block: (pre-norm) -> qkv -> (q/k norms) -> rope
            // -> append -> attention -> o_proj landed on the residual.
            let x = if post_norm {
                y.clone()
            } else {
                dsl::cuda::rmsnorm(&y, &w.attn_norm)
            };

            // The general QKV arm, produced once and called from every
            // path that takes it: packed-or-split projections, the q/k
            // norm convention ("the weight knows"), rope, and the KV
            // write (the HasWriteDesc guard when lowered). Produces q.
            let general_qkv = || {
                let (q, k, v) = if f.fused_qkv {
                    split_qkv(&matmul(&x, &w.qkv), q_w, kv_w)
                } else {
                    (
                        matmul(&x, &w.q_proj),
                        matmul(&x, &w.k_proj),
                        matmul(&x, &w.v_proj),
                    )
                };
                {
                    // The adapter value seam (§5.1): attachments land on
                    // the just-materialized RAW q/v projections, BEFORE
                    // anything consumes them — bias, norms, rope, the KV
                    // append (the hand-written apply's position;
                    // correcting after rope is different arithmetic, the
                    // bug the first live A/B caught). Rung-① lowering is
                    // the HasLora guard with an EMPTY else: a fire with
                    // no usable lanes launches nothing.
                    dsl::seam(m.trace(), &dsl::seam::ATTN_QV, &[&q, &v], Some(l));
                }
                // Qwen-2 family qkv biases: on the raw projections, after
                // the lora correction and before norms/rope — the
                // hand-written `maybe_add_bias` position (bias order vs
                // the correction matters: the adapter delta lands on the
                // base projection, not on base + bias).
                let (q, k, v) = if f.qkv_bias {
                    (
                        add_bias(&q, &w.q_bias),
                        add_bias(&k, &w.k_bias),
                        add_bias(&v, &w.v_bias),
                    )
                } else {
                    (q, k, v)
                };
                // The per-head convention with Standard rope states the
                // fused norm+rope kernel (the hand-written
                // `fuse_qk_norm_rope` branch — bf16 rounds differently
                // from the triple, so parity requires the same launch);
                // the Global and Off conventions state the separate
                // kernels, whose semantic ops are 1:1.
                let per_head_fused = f.qk_norm == QkNorm::PerHead && f.rope == RopeKind::Standard;
                let (q, k) = if per_head_fused {
                    cuda::qk_rmsnorm_rope(&q, &k, &w.q_norm, &w.k_norm)
                } else {
                    let (q, k) = if f.qk_norm == QkNorm::Off {
                        (q, k)
                    } else {
                        (
                            dsl::cuda::rmsnorm(&q, &w.q_norm),
                            dsl::cuda::rmsnorm(&k, &w.k_norm),
                        )
                    };
                    // STATED (2a). The build gate admits only Standard
                    // rope, and the executor's arm asked whether a
                    // rotary width was set to pick between two
                    // launchers -- a kernel choice from a param. This
                    // family rotates the full head, and says which
                    // kernel that is.
                    dsl::cuda::rope(&q, &k, f.q_heads, f.kv_heads, f.head_dim)
                };
                // The KV-write mechanism is a per-fire runtime input
                // (explicit descriptors when the fire steers a graph
                // replay, page-derived otherwise). Under the fused
                // deployment's mask arm this guard NESTS inside the
                // HasCustomMask guard (A1 — the walk keeps a stack).
                // 2c: the PAD STAGING, stated.
                //
                // A deployment whose attention kernels run at a wider
                // head than the checkpoint's (Phi-3-mini: 96 -> 128)
                // copies q, k and v into zero-padded buffers before the
                // KV write, and narrows the attention's output after.
                // Sixteen executor sites read a boolean and staged into
                // `ws.{q,k,v,attn_out}_padded` -- workspace fields no
                // traced value described, which is why the writes could
                // not move onto the arena and why the strip's
                // destination needed a lambda of its own.
                //
                // Three launches and their results, so the padded
                // copies are VALUES and every consumer names one.
                //
                // Only this path can be padded: the fused decode-QKV
                // arm's own fact requires `head_dim == head_dim_kernel`
                // (`cuda.decode_fused_post`), so the region form below
                // never coincides with staging -- which is the same
                // thing the executor's Peel comment said, from the
                // other side.
                let (q, k, v) = if pad_to > 0 {
                    (
                        cuda::pad_head_dim(&q, f.q_heads, pad_to),
                        cuda::pad_head_dim(&k, f.kv_heads, pad_to),
                        cuda::pad_head_dim(&v, f.kv_heads, pad_to),
                    )
                } else {
                    (q, k, v)
                };
                dsl::guard(
                    m.trace(),
                    GuardPred::HasWriteDesc,
                    || cuda::write_kv_explicit(&k, &v, &w.kv),
                    || cuda::write_kv_to_pages(&k, &v, &w.kv),
                );
                q
            };
            // The attention's own output width: the PADDED one where
            // the kernels run wide. The strip below is what brings it
            // back to `q_w`, and it is a statement rather than a
            // driver's parting copy.
            // THIS LAYER's sliding window, `-1` for none. A load-time
            // fact, so it erases into the statements below rather than
            // being re-derived from `fwd_cfg.per_layer_window_left` on
            // every dispatch -- which is what four executors did, in
            // eleven copies of the same three lines.
            let window_left = cuda.window_left_at(l);
            let attn_out_shape = (
                Shape(vec![
                    Dim::Tokens,
                    Dim::Const(if pad_to > 0 { f.q_heads * pad_to } else { q_w }),
                ]),
                DType::BF16,
            );

            let a = match class {
                // A1–A3 (the class-collapse amendment): per-fire
                // attachments are guard arms and ROW WINDOWS of the
                // shape classes, not classes. The chain per layer:
                // custom mask (the custom dispatch; the whole general
                // QKV sequence in the fused deployment) | else the ONE
                // body every unmasked fire walks — the QKV production
                // (a `Peel` in the fused deployment: fused epilogue over
                // the hook-free prefix rows, general sequence over the
                // tail, `fast_rows` the runtime split; fast_rows == N is
                // the classic all-fused fire, 0 the all-hooked one),
                // then the two HookSites (argument no-ops on an unhooked
                // fire) and the WantsAttnScore-guarded attention (the
                // score-capturing dispatch is a different launcher, and
                // whether the fire's programs read scores is a runtime
                // input). XQA has no capture variant: the body states
                // the plain XQA launch, and a score-wanting program
                // under XQA fails loudly PTIR-side (the hand-written
                // contract). Masked+hooked stays hand-written (the mask
                // arm carries no sites); the caller's gate encodes it.
                // V2 rung ②b: ONE dispatch statement for both shape
                // classes. The divergence keys on the WINDOW OPERAND'S
                // CLASS — `window_one` (every row a 1-token qo window:
                // today's Decode instantiation) vs ragged (Prefill) —
                // stated as trace-time predicates the way the fact arms
                // are. The two per-class arm bodies this replaces were
                // structurally one body already (the goldens pin the
                // collapse is byte-identical); rung ③ makes the window
                // class a PER-ROW operand and this match a region table.
                FireClass::Decode | FireClass::Prefill => {
                    let c = cuda;
                    // THE WINDOW CLASS IS A GUARD, not a class.
                    //
                    // `let window_one = class == FireClass::Decode` was
                    // the whole of what separated Decode from Prefill —
                    // one body, one boolean, and the goldens pinning the
                    // collapse as byte-identical. Directive 4.1 of
                    // `.wiki/driver/graph.md` names the destination and
                    // the retired masked/hooked classes (A1/A2) are the
                    // precedent: a delta this local belongs at op
                    // granularity, which is what a guard is.
                    //
                    // So every site below that asked the class now states
                    // `GuardPred::WindowOne` instead, and both arms lower
                    // into ONE graph. A mixed fire answers false and takes
                    // the ragged arm, which serves a one-token request as
                    // its degenerate case — the property that makes the
                    // merge sound rather than merely tidy.
                    //
                    // The `c.*` tests that remain are DEPLOYMENT
                    // constants (padded head dims, XQA, the GQA ratio
                    // that forces the prefill path), not fire facts, so
                    // they stay host-side branches.
                    // ORDER IS LOAD-BEARING: `guarded_value` OPENS the
                    // chain, and every op recorded after it counts into
                    // the first arm's region. The non-fused deployments'
                    // general QKV must therefore trace BEFORE the guard
                    // opens (the hoisted `q` below) — tracing it after
                    // put the whole QKV sequence inside the mask arm,
                    // and every unmasked fire skipped it (the phi3/
                    // mistral live-garbage regression, caught 2026-08-03
                    // by the three-model battery; the mistral lowered
                    // goldens now pin this structure). The fused-post
                    // deployment (window-one only by its predicate) is
                    // the one QKV-inside-the-arms shape.
                    let hoisted_q = (!fused_post).then(&general_qkv);
                    // NOT migrated to `regions`, deliberately. The other
                    // three sites in the tree are, and the goldens prove
                    // the surface changes no traced byte — but this one
                    // branches on `fused_post` in BOTH its arm and its
                    // rest, so moving it is a restructure rather than a
                    // rename, and the order it depends on is the order
                    // the phi3/mistral live-garbage regression was about.
                    // A restructure whose only gate is a golden it also
                    // rewrites is not gated; do this one where the
                    // three-model battery can run.
                    let (g, a) = dsl::guarded_value(m.trace(), Some(l), attn_out_shape.clone());
                    // The masked attention states its SPATIAL SPLIT as
                    // vocabulary (NS-4 landed in the IR): a Peel on the
                    // unmasked-prefix axis — the deployment's CAUSAL
                    // dispatch for this window class serves the plain
                    // prefix rows, the custom dispatch the masked
                    // suffix, the split a runtime input, UNPLANNED
                    // collapsing to tail-only full-N (the fire-level
                    // dispatch as the peel's endpoint). Padded head dims
                    // keep the fire-level word (the split's row offsets
                    // are logical-width, the padded staging is not), and
                    // XQA deployments too (the XQA fire-wide prepare is
                    // R-shaped) — both mirror the prepare gate exactly,
                    // so the trace never states a split prepare refuses
                    // to plan.
                    let masked_attention = |q: &Val| {
                        // The peeled form: the deployment's causal
                        // dispatch over the unmasked prefix, the custom
                        // one over the masked suffix.
                        let peeled = |q: &Val| {
                            dsl::by_rows(m.trace(), Some(l), None, |r| {
                                r.arm(dsl::RowPred::Unmasked, || {
                                    // The prefix states THE DEPLOYMENT'S
                                    // causal form: the planned decode
                                    // dispatch on window-one fires —
                                    // force_prefill (GQA ratio outside
                                    // the decode kernel's set) falling
                                    // back to the plan-free prefill
                                    // dispatch behind its dequant
                                    // staging — and the causal prefill
                                    // dispatch (same staging) on ragged
                                    // fires: any mix of prefill and
                                    // plain-decode requests, ragged qo.
                                    if c.force_prefill_path {
                                        cuda::dequant_only(&w.kv);
                                        cuda::attention_flashinfer_prefill(q, &w.kv, window_left, plan_head_dim, 0.0, 0.0);
                                    } else {
                                        dsl::guarded(m.trace())
                                            .arm(GuardPred::WindowOne, || {
                                                // hook×mask: the prefix decode IS
                                                // the paged decode path and the
                                                // hooked rows live in it (the
                                                // seriation puts masked rows in
                                                // the suffix, so the prefix
                                                // starts at row 0 and the request
                                                // ordinals are the unsplit ones).
                                                // So the score capture rides here
                                                // exactly as in the unsplit arm —
                                                // the hand-written body's
                                                // `if (score_capture.active())`
                                                // on this same branch.
                                                dsl::guarded(m.trace())
                                                    .arm(GuardPred::WantsAttnScore, || {
                                                        cuda::attention_flashinfer_decode_capture(
                                                            q,
                                                            &w.kv,
                                                            window_left,
                                                        plan_head_dim,
 0.0, 0.0);
                                                    })
                                                    .otherwise(|| {
                                                        cuda::attention_flashinfer_decode(
                                                            q,
                                                            &w.kv,
                                                            window_left,
                                                        plan_head_dim,
);
                                                    });
                                            })
                                            .otherwise(|| {
                                                cuda::dequant_only(&w.kv);
                                                cuda::attention_flashinfer_prefill(
                                                    q,
                                                    &w.kv,
                                                    window_left,
                                                plan_head_dim,
 0.0, 0.0);
                                            });
                                    }
                                });
                                r.rest(|| {
                                    cuda::attention_flashinfer_prefill_custom(
                                        q,
                                        &w.kv,
                                        window_left,
                                    plan_head_dim,
 0.0, 0.0);
                                });
                            });
                        };
                        if c.head_dim_padded {
                            // The split's row offsets are logical-width
                            // and the padded staging is not, so this
                            // deployment keeps the fire-level word.
                            cuda::attention_flashinfer_prefill_custom(q, &w.kv, window_left, plan_head_dim, 0.0, 0.0);
                        } else if c.xqa_decode {
                            // XQA's prepare is fire-wide (R-shaped), so a
                            // window-one fire cannot peel; a ragged one
                            // never reaches XQA and can.
                            dsl::guard(
                                m.trace(),
                                GuardPred::WindowOne,
                                || {
                                    cuda::attention_flashinfer_prefill_custom(
                                        q,
                                        &w.kv,
                                        window_left,
                                    plan_head_dim,
 0.0, 0.0);
                                },
                                || peeled(q),
                            );
                        } else {
                            peeled(q);
                        }
                    };
                    let attn_with_sites = |q: &Val| {
                        dsl::seam(q.trace(), &dsl::seam::ATTN_Q, &[q], Some(l));
                        if c.xqa_decode {
                            // XQA has no capture variant and no ragged
                            // form: the deployment states it or it does
                            // not, and a score-wanting program under XQA
                            // fails loudly PTIR-side.
                            cuda::attention_xqa_decode(q, &w.kv, window_left);
                        } else if c.force_prefill_path {
                            // The GQA ratio sits outside the decode
                            // kernel's set, so BOTH window classes take
                            // the prefill dispatch — there is nothing
                            // left for a guard to choose between.
                            cuda::dequant_only(&w.kv);
                            cuda::attention_flashinfer_prefill(q, &w.kv, window_left, plan_head_dim, 0.0, 0.0);
                        } else {
                            dsl::guarded(m.trace())
                                .arm(GuardPred::WindowOne, || {
                                    dsl::guarded(m.trace())
                                        .arm(GuardPred::WantsAttnScore, || {
                                            cuda::attention_flashinfer_decode_capture(
                                                q,
                                                &w.kv,
                                                window_left,
                                            plan_head_dim,
 0.0, 0.0);
                                        })
                                        .otherwise(|| {
                                            cuda::attention_flashinfer_decode(
                                                q,
                                                &w.kv,
                                                window_left,
                                            plan_head_dim,
);
                                        });
                                })
                                .otherwise(|| {
                                    // Ragged fires are row-uniform:
                                    // dequant, then the score-guarded
                                    // causal dispatch.
                                    cuda::dequant_only(&w.kv);
                                    dsl::guarded(m.trace())
                                        .arm(GuardPred::WantsAttnScore, || {
                                            cuda::attention_flashinfer_prefill_capture(
                                                q,
                                                &w.kv,
                                                window_left,
                                            plan_head_dim,
 0.0, 0.0);
                                        })
                                        .otherwise(|| {
                                            cuda::attention_flashinfer_prefill(
                                                q,
                                                &w.kv,
                                                window_left,
                                            plan_head_dim,
 0.0, 0.0);
                                        });
                                });
                        }
                        dsl::seam(q.trace(), &dsl::seam::ATTN_OUT, &[q], Some(l));
                    };
                    if fused_post {
                        g.arm(GuardPred::HasCustomMask, || {
                            // Masked+hooked composes here: the sites run
                            // around the custom dispatch exactly as the
                            // hand-written unconditional invokes do. The
                            // SPLIT's unmasked prefix carries the score
                            // capture (see `masked_attention`); only the
                            // masked suffix's custom dispatch has no
                            // capture variant, and a fire that is masked
                            // all the way down publishes nothing, which
                            // is the publish-gated contract.
                            let q = general_qkv();
                            dsl::seam(q.trace(), &dsl::seam::ATTN_Q, &[&q], Some(l));
                            masked_attention(&q);
                            dsl::seam(q.trace(), &dsl::seam::ATTN_OUT, &[&q], Some(l));
                        })
                        // The lora arm: the fused epilogue writes V
                        // straight to the paged cache — nothing exists to
                        // correct into — so a lora fire runs the whole
                        // general sequence (whose internal adapter seam
                        // lands the correction), full-N: the hand-written
                        // `!has_lora` predicate term, stated as an arm.
                        // Mask+lora composes in the mask arm above (its
                        // general body carries the same internal seam).
                        .arm(GuardPred::HasLora, || {
                            let q = general_qkv();
                            attn_with_sites(&q);
                        })
                        .otherwise(|| {
                            // The packed GEMM runs over every row; the
                            // Peel splits its postprocess: the fused
                            // kernel (split + norms + rope + KV write,
                            // one launch) owns the hook-free prefix, the
                            // general sequence owns the hook-visible
                            // tail — the hand-written mixed fire,
                            // launch for launch.
                            let packed = matmul(&x, &w.qkv);
                            // The outer construct is a ROW partition and
                            // the inner one a FIRE guard, nested — which
                            // `regions` allows and refuses only to flatten
                            // into one chain. Migrating the outer one
                            // leaves the nesting exactly as the text had
                            // it.
                            let q = dsl::regions(
                                m.trace(),
                                Some(l),
                                Some(attn_out_shape.clone()),
                                |r| {
                                    r.arm(dsl::Region::Rows(dsl::RowPred::HookFree), || {
                                        cuda::qkv_decode_qk_norm_rope_write_kv_region(
                                            &packed,
                                            &w.q_norm,
                                            &w.k_norm,
                                            &w.kv,
                                            table.as_ref(),
                                        );
                                    });
                                },
                                || {
                                    let (qt, kt, vt) = split_qkv(&packed, q_w, kv_w);
                                    let (_qt, kt) =
                                        cuda::qk_rmsnorm_rope(&qt, &kt, &w.q_norm, &w.k_norm);
                                    dsl::guard(
                                        m.trace(),
                                        GuardPred::HasWriteDesc,
                                        || cuda::write_kv_explicit(&kt, &vt, &w.kv),
                                        || cuda::write_kv_to_pages(&kt, &vt, &w.kv),
                                    );
                                },
                            )
                            .expect("a value-producing row partition produces its value");
                            attn_with_sites(&q);
                        });
                    } else {
                        let q = hoisted_q.as_ref().expect("hoisted for the non-fused arms");
                        g.arm(GuardPred::HasCustomMask, || {
                            // Masked+hooked (the fused arm's comment).
                            dsl::seam(q.trace(), &dsl::seam::ATTN_Q, &[q], Some(l));
                            masked_attention(q);
                            dsl::seam(q.trace(), &dsl::seam::ATTN_OUT, &[q], Some(l));
                        })
                        .otherwise(|| attn_with_sites(q));
                    }
                    a
                }
            };
            // 2c: the STRIP. The attention wrote at the kernel width;
            // `o_proj` reads at the logical one, and this is what
            // narrows it -- one statement, whose result is what every
            // consumer downstream names.
            //
            // It sits after the guard chain rather than inside its
            // arms, which is also what the executor did: the padded
            // output is one buffer whichever dispatch filled it, so the
            // narrowing is one launch and not one per arm.
            let a = if pad_to > 0 {
                cuda::strip_head_dim(&a, f.q_heads, f.head_dim)
            } else {
                a
            };
            if post_norm {
                // Post-norm: o_proj to scratch, norm the OUTPUT, then the
                // separate residual landing (`+=` of a non-matmul records
                // the explicit ResidualAdd launch).
                //
                // Under TP the projection is ROW-PARALLEL: each rank's
                // GEMM produces a partial `[N, hidden]`, so the sum
                // across ranks has to happen before the norm reads it.
                // In place, because nothing else reads the partial.
                let o = matmul(&a, &w.o_proj);
                let o = if tp > 1 {
                    all_reduce(m.trace(), &o, f.hidden, cuda)
                } else {
                    o
                };
                y += dsl::cuda::rmsnorm(&o, &w.attn_norm);
                // ② The MLP's two spellings, and the binding picks
                // which the text STATES -- not which the executor
                // reads. A packed bank is one matmul into the chunked
                // kernel; an unfused binding is TWO matmuls into the
                // pair kernel, and until 2d that second reading was a
                // one-statement lie the executor repaired by firing two
                // GEMMs into workspace buffers no value described.
                let act = mlp(&y, &w, f.intermediate, cuda.gate_up_fused);
                // The MLP's landing, same shape as the attention's
                // above: `down` is row-parallel, so its output is a
                // partial and the sum precedes the norm.
                let d_out = matmul(&act, &w.down);
                let d_out = if tp > 1 {
                    all_reduce(m.trace(), &d_out, f.hidden, cuda)
                } else {
                    d_out
                };
                y += dsl::cuda::rmsnorm(&d_out, &w.mlp_norm);
            } else if tp > 1 {
                // Pre-norm under TP. `+=` cannot fold here: the beta=1
                // GEMM would add a PARTIAL into the residual, and the
                // sum across ranks has to happen first. So the
                // projection writes fresh, the collective sums it, and
                // the residual add and the next norm are a statement of
                // their own -- which is the pair the hand-written pass
                // fires as `all_reduce_bf16_out` + `residual_add_rmsnorm`.
                //
                // The FUSED landing (`comm::all_reduce_residual_rmsnorm_bf16`,
                // one launch for all three) is what the hand pass takes
                // when `can_fuse_residual_rmsnorm(N, H, stream)` holds.
                // Not stated here, and the reason is a vocabulary gap
                // rather than a preference: that kernel has TWO effects
                // -- the stream updated in place and the normed
                // activation -- while the two-step form's SSA shape is
                // one value, and `guarded_value` carries one value per
                // chain. A guard whose arms produce a PAIR is what the
                // fused arm needs, and until it exists stating the
                // fused form would mean an arm the else could not
                // match.
                let partial = matmul(&a, &w.o_proj);
                let summed = all_reduce(m.trace(), &partial, f.hidden, cuda);
                let x = cuda::residual_add_rmsnorm(&y, &summed, &w.mlp_norm.name, f.hidden);
                let act = mlp(&x, &w, f.intermediate, cuda.gate_up_fused);
                // The MLP is COLUMN-parallel through `gate_up` and
                // row-parallel through `down`, so its output is a
                // partial too and lands the same way.
                let mlp_out = matmul(&act, &w.down);
                y += all_reduce(m.trace(), &mlp_out, f.hidden, cuda);
            } else {
                // Pre-norm: `+=` of a fresh matmul IS the beta=1 fold.
                y += matmul(&a, &w.o_proj);
                let x = dsl::cuda::rmsnorm(&y, &w.mlp_norm);
                let act = mlp(&x, &w, f.intermediate, cuda.gate_up_fused);
                y += matmul(&act, &w.down);
            }
        }

        let logits = m.logits(&dsl::cuda::rmsnorm(&y, &m.final_norm()));
        dsl::seam(m.trace(), &dsl::seam::OUT, &[&logits], None);
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_ir::trace::{Dim, OpKind};

    /// Every kernel symbol a Metal plan states, in order.
    ///
    /// A Metal text is a sequence of STATED launches -- `OpKind::Launch`
    /// carries the driver's launcher symbol -- so "which kernel does this
    /// deployment run" is a question the plan answers directly, and it is
    /// the only question these facts exist to settle.
    fn metal_kernels_at(plan: &ForwardPlan, layer: u32) -> Vec<String> {
        plan.layer_ops(layer)
            .filter_map(|op| match &op.kind {
                OpKind::Launch { kernel, .. } => Some(kernel.clone()),
                _ => None,
            })
            .collect()
    }

    fn metal_kernels(plan: &ForwardPlan) -> Vec<String> {
        metal_kernels_at(plan, 0)
    }

    fn runs(plan: &ForwardPlan, kernel: &str) -> usize {
        metal_kernels(plan).iter().filter(|k| *k == kernel).count()
    }

    /// Whether any statement in the plan names this weight.
    fn binds(plan: &ForwardPlan, weight: &str) -> bool {
        plan.ops.iter().any(|op| match &op.kind {
            OpKind::Launch { weights, .. } => weights.iter().any(|w| w == weight),
            _ => false,
        })
    }

    /// The fused per-head norm+rope replaces TWO statements, and only
    /// where all four of its conditions hold.
    ///
    /// This is the test the `fused_qk_rope` fact is stated both ways by:
    /// no shipped Metal binding turns it on -- there is no `.metal` kernel
    /// behind the symbol, only a census name so `model-ir` can check a
    /// Vulkan text -- so a fixture stating `true` would describe a Metal
    /// deployment that does not exist. The fact reaches production through
    /// `engine`'s Vulkan backend alone.
    ///
    /// Four claims, because a fusion that fires where it must not is worse
    /// than one that never fires: the wrong arm here produces a plausible
    /// tensor and no check downstream can see it.
    #[test]
    fn the_fused_qk_norm_rope_replaces_the_pair_and_only_when_it_may() {
        let f = LlamaLikeFacts::qwen3_0_6b();
        assert_eq!(
            f.qk_norm,
            QkNorm::PerHead,
            "this fixture is the one the fusion is for"
        );
        let apart = LlamaLikeMetalFacts::synthetic();
        assert!(!apart.fused_qk_rope && !apart.rope_freq_table && !apart.v_norm);
        let fused = LlamaLikeMetalFacts {
            fused_qk_rope: true,
            ..apart.clone()
        };

        let separate = llama_like_metal(&f, &apart, FireClass::Decode);
        let together = llama_like_metal(&f, &fused, FireClass::Decode);

        // The pair it replaces: two per-head norms and two rotations,
        // gone, and two fused statements in their place.
        assert_eq!(runs(&together, "rms_rope_bfloat16"), 2);
        assert_eq!(runs(&separate, "rms_rope_bfloat16"), 0);
        assert_eq!(
            runs(&separate, "neox_mb_bfloat16") - runs(&together, "neox_mb_bfloat16"),
            2,
            "the fused text still rotates separately: {:?}",
            metal_kernels(&together)
        );
        assert_eq!(
            // The same symbol the two BLOCK norms use, so this reads as a
            // difference and not as a count: a decode row is one row, and
            // `dsl::metal::rms_norm` states the single-row form for it.
            runs(&separate, "rms_single_row_bfloat16")
                - runs(&together, "rms_single_row_bfloat16"),
            2,
            "the fused text still norms separately: {:?}",
            metal_kernels(&together)
        );
        // Four statements out, two in, so the layer is two shorter. Stated
        // on the LENGTH as well as the counts, because a fusion that added
        // a statement elsewhere would satisfy every count above.
        assert_eq!(
            metal_kernels(&separate).len() - metal_kernels(&together).len(),
            2
        );

        // A GLOBAL qk-norm is different arithmetic -- the whole row against
        // a `[heads * head_dim]` weight -- and the fused base is per head.
        let mut global = f.clone();
        global.qk_norm = QkNorm::Global;
        assert_eq!(
            runs(&llama_like_metal(&global, &fused, FireClass::Decode), "rms_rope_bfloat16"),
            0
        );

        // A rescaled ladder is a TABLE, and the fused family's table arm is
        // compiled with no routine behind it.
        let table = LlamaLikeMetalFacts {
            rope_freq_table: true,
            ..fused.clone()
        };
        assert_eq!(
            runs(&llama_like_metal(&f, &table, FireClass::Decode), "rms_rope_bfloat16"),
            0
        );

        // And the one that would corrupt rather than refuse: a layer that
        // takes V FROM the K projection. The fused kernel is in place, so
        // it would norm and rotate V as a side effect of doing it to K.
        // `window < 0` is the full-attention arm, which is what `synthetic`
        // states for every layer.
        let k_is_v = LlamaLikeMetalFacts {
            v_from_k: true,
            ..fused.clone()
        };
        let plan = llama_like_metal(&f, &k_is_v, FireClass::Decode);
        assert_eq!(
            runs(&plan, "rms_rope_bfloat16"),
            0,
            "the fusion fired on a k-eq-v layer and rotated V: {:?}",
            metal_kernels(&plan)
        );
    }

    /// The three activations are three different kernels, and the fact is
    /// the only thing that picks between them.
    ///
    /// A wrong answer here is silent in the worst way: all three take a
    /// gate and an up bank of the same extents and write a tensor of the
    /// same shape, so nothing downstream can tell that the wrong curve was
    /// applied. The model generates, fluently, through the wrong
    /// non-linearity.
    #[test]
    fn each_metal_activation_names_its_own_kernel() {
        let f = LlamaLikeFacts::qwen3_0_6b();
        for (activation, kernel) in [
            (Activation::SiluMul, "silu_mul_bfloat16"),
            (
                Activation::SwiGlu {
                    limit: 7.0,
                    alpha: 1.702,
                },
                "gptoss_swiglu_bfloat16",
            ),
            (Activation::Geglu, "geglu_tanh_bfloat16"),
        ] {
            let metal = LlamaLikeMetalFacts {
                activation,
                ..LlamaLikeMetalFacts::synthetic()
            };
            let plan = llama_like_metal(&f, &metal, FireClass::Decode);
            let kernels = metal_kernels(&plan);
            assert!(
                kernels.iter().any(|k| k == kernel),
                "{activation:?} did not state `{kernel}`; it stated {kernels:?}"
            );
            // And states no OTHER activation, which is what makes the
            // three arms a choice rather than a list.
            for other in [
                "silu_mul_bfloat16",
                "gptoss_swiglu_bfloat16",
                "geglu_tanh_bfloat16",
            ] {
                if other != kernel {
                    assert!(
                        !kernels.iter().any(|k| k == other),
                        "{activation:?} also stated `{other}`"
                    );
                }
            }
        }
    }

    /// A layer that shares another's KV projects Q and NOTHING ELSE.
    ///
    /// The k and v it hands on stand for the SOURCE layer's pages, which
    /// the attention reads through the pool rather than as operands. So a
    /// shared layer that still projected its own would compute two banks
    /// per layer and write neither anywhere the attention looks -- the
    /// cost of a full KV projection for a result nothing reads.
    #[test]
    fn a_kv_sharing_metal_layer_projects_q_alone() {
        let f = LlamaLikeFacts::qwen3_0_6b();
        // The sharers are the LAST `kv_shared_layers` layers, so layer 0 is
        // never one of them and the two plans below differ only at the tail.
        let last = f.layers - 1;
        let shared = LlamaLikeMetalFacts {
            kv_shared_layers: 4,
            ..LlamaLikeMetalFacts::synthetic()
        };
        let alone = LlamaLikeMetalFacts {
            kv_shared_layers: 0,
            ..LlamaLikeMetalFacts::synthetic()
        };
        let with_sharing = llama_like_metal(&f, &shared, FireClass::Decode);
        let without = llama_like_metal(&f, &alone, FireClass::Decode);
        assert!(
            metal_kernels_at(&with_sharing, last).len() < metal_kernels_at(&without, last).len(),
            "sharing KV did not remove any launch from the last layer:\n  \
             shared: {:?}\n  own: {:?}",
            metal_kernels_at(&with_sharing, last),
            metal_kernels_at(&without, last)
        );
        // And the layers BEFORE the shared tail are untouched, which is what
        // makes `kv_shared_layers` a count and not a switch.
        assert_eq!(
            metal_kernels_at(&with_sharing, 0),
            metal_kernels_at(&without, 0)
        );
    }

    /// A mixture with a dense expert beside it blends the two, and the
    /// blend is a kernel of its own.
    ///
    /// Without it the routed output is returned as the layer's answer and
    /// the shared expert's weights are loaded, bound, and never read. The
    /// model runs at full speed and is missing a term.
    #[test]
    fn a_shared_expert_beside_a_mixture_is_blended_in() {
        let mut f = LlamaLikeFacts::qwen3_0_6b();
        f.n_experts = 4;
        f.experts_per_token = 2;
        f.shared_intermediate = 512;
        let metal = LlamaLikeMetalFacts::synthetic();
        let blended = llama_like_metal(&f, &metal, FireClass::Decode);
        assert_eq!(
            runs(&blended, "shared_expert_combine"),
            1,
            "a mixture with `shared_intermediate` did not blend its dense \
             expert: {:?}",
            metal_kernels(&blended)
        );

        // And a mixture WITHOUT one does not, so the blend is keyed on the
        // width and not merely on being a mixture.
        f.shared_intermediate = 0;
        let routed_only = llama_like_metal(&f, &metal, FireClass::Decode);
        assert_eq!(runs(&routed_only, "shared_expert_combine"), 0);
    }

    /// The attention biases are added only when the row says the weights
    /// exist AND the driver says it still adds them.
    ///
    /// Two facts, because they are two different claims: `qkv_bias` is a
    /// property of the checkpoint and `add_bias` is a property of this
    /// build's Metal path. Either one alone must not produce the launches,
    /// or a family whose checkpoint has no bias tensors would bind three
    /// weights that are not there.
    #[test]
    fn the_attention_bias_launches_need_both_facts() {
        let mut with_bias = LlamaLikeFacts::qwen3_0_6b();
        with_bias.qkv_bias = true;
        let mut without_bias = LlamaLikeFacts::qwen3_0_6b();
        without_bias.qkv_bias = false;
        let adds = LlamaLikeMetalFacts {
            add_bias: true,
            ..LlamaLikeMetalFacts::synthetic()
        };
        let does_not = LlamaLikeMetalFacts {
            add_bias: false,
            ..LlamaLikeMetalFacts::synthetic()
        };
        let both = llama_like_metal(&with_bias, &adds, FireClass::Decode);
        assert_eq!(
            runs(&both, "add_bias_bfloat16"),
            3,
            "q, k and v each take their own bias: {:?}",
            metal_kernels(&both)
        );
        for (facts, metal, why) in [
            (&with_bias, &does_not, "the driver does not add them"),
            (&without_bias, &adds, "the checkpoint has none"),
            (&without_bias, &does_not, "neither"),
        ] {
            let plan = llama_like_metal(facts, metal, FireClass::Decode);
            assert_eq!(
                runs(&plan, "add_bias_bfloat16"),
                0,
                "biases added when {why}"
            );
        }
    }

    /// All three norm arms LAND, whatever they norm.
    ///
    /// Two of the three land through a shared closure and the third
    /// fuses the projection into the residual add, so "the attention
    /// output is projected by `o_proj`" is stated three times in three
    /// spellings. A closure that stopped projecting would leave the
    /// attention output going straight into the norm at the wrong width
    /// for every `Post` and `Sandwich` row -- and nothing in this module
    /// asked, which is how the landing's deleted bias branch sat unread
    /// for as long as it did.
    #[test]
    fn every_norm_arm_lands_through_the_output_projection() {
        for placement in [
            NormPlacement::Pre,
            NormPlacement::Post,
            NormPlacement::Sandwich,
        ] {
            let mut f = LlamaLikeFacts::qwen3_0_6b();
            f.norm_placement = placement;
            let plan = llama_like_metal(&f, &LlamaLikeMetalFacts::synthetic(), FireClass::Decode);
            let lands = plan
                .ops
                .iter()
                .filter(|op| match &op.kind {
                    OpKind::Launch { weights, .. } => weights.iter().any(|w| w == "layer.0.o_proj"),
                    _ => false,
                })
                .count();
            assert_eq!(
                lands, 1,
                "{placement:?} does not project its attention output"
            );
        }
    }

    /// A mixture owes no `down` under ANY norm placement.
    ///
    /// `owes_down` is `n_experts == 0`: a dense MLP hands back the hidden
    /// width and needs the down projection, a mixture already combined
    /// its rows and does not. The text asks three times -- once per norm
    /// arm -- and only the fused arm had ever been asked it as a mixture,
    /// because every mixture this build ships norms `Pre`.
    ///
    /// Both mistakes are quiet and neither is a fault. A mixture that
    /// took the dense arm binds `layer.0.down`, a tensor a routed
    /// checkpoint does not publish, and the load fails naming a missing
    /// weight -- which reads as a corrupt download. A dense row that took
    /// the mixture arm skips the projection and feeds the intermediate
    /// width into a residual add that expects the hidden one.
    #[test]
    fn a_mixture_owes_no_down_projection_under_any_norm_placement() {
        for placement in [
            NormPlacement::Pre,
            NormPlacement::Post,
            NormPlacement::Sandwich,
        ] {
            let mut dense = LlamaLikeFacts::qwen3_0_6b();
            dense.norm_placement = placement;
            let mut routed = LlamaLikeFacts::qwen3_30b_a3b();
            routed.norm_placement = placement;
            assert!(
                dense.n_experts == 0 && routed.n_experts > 0,
                "the two cases"
            );

            for (f, owed) in [(&dense, true), (&routed, false)] {
                let plan =
                    llama_like_metal(f, &LlamaLikeMetalFacts::synthetic(), FireClass::Decode);
                let binds_down = plan.ops.iter().any(|op| match &op.kind {
                    OpKind::Launch { weights, .. } => weights.iter().any(|w| w == "layer.0.down"),
                    _ => false,
                });
                assert_eq!(
                    binds_down, owed,
                    "{placement:?} with {} experts binds layer.0.down",
                    f.n_experts
                );
            }
        }
    }

    /// gemma runs its mixture BESIDE the dense MLP, not instead of it.
    ///
    /// Every other family in this text runs one FFN or the other, so the
    /// branch is easy to read as a variant spelling of the same thing. It
    /// is not: both read the post-attention residual and both are added
    /// back, and a build that took it for a variant would drop one of the
    /// two terms the layer is defined as.
    ///
    /// This used to assert TWO routers, which is what the branch produced
    /// and not what the model is. The old text ran the routed FFN in the
    /// placement arm and then ran it AGAIN in the branch, so "beside" was
    /// measured as "routes twice" -- a count that the defect satisfies and
    /// the reference does not. What beside means is ONE router and, in the
    /// same layer, a DENSE gate/up/down, which is what this asserts now.
    #[test]
    fn gemmas_mixture_runs_beside_the_dense_mlp_rather_than_instead_of_it() {
        let mut f = LlamaLikeFacts::qwen3_0_6b();
        f.n_experts = 4;
        f.experts_per_token = 2;
        f.norm_placement = NormPlacement::Sandwich;
        let beside = LlamaLikeMetalFacts {
            dense_beside_moe: true,
            router_input_norm: true,
            router_expert_scale: true,
            ..LlamaLikeMetalFacts::synthetic()
        };
        let instead = LlamaLikeMetalFacts::synthetic();
        let two = llama_like_metal(&f, &beside, FireClass::Decode);
        let one = llama_like_metal(&f, &instead, FireClass::Decode);
        assert_eq!(
            runs(&two, "router_topk_scaled_bfloat16"),
            runs(&one, "router_topk_bfloat16"),
            "beside is not routing twice:\n  beside: {:?}",
            metal_kernels(&two)
        );
        // The DENSE leg, which the routed-instead row does not have. Its
        // down projection is bound explicitly rather than fused, because
        // `post_feedforward_layernorm_1` sits between it and the residual.
        for w in ["layer.0.gate_proj", "layer.0.up_proj", "layer.0.down"] {
            assert!(binds(&two, w), "the dense leg binds {w}");
            assert!(!binds(&one, w), "the routed-instead row does not bind {w}");
        }
        // SEVEN norms a layer, against the sandwich's four: `post_mlp_norm_1`,
        // `mlp_norm_2`, `post_mlp_norm_2` and the ROUTER's. A count, because
        // the tensors are the only thing that distinguishes them and a text
        // that reused one would still be at the right width.
        //
        // BOTH spellings, because a norm whose value is read by exactly one
        // residual add is one statement -- `rms_residual` -- and which of the
        // seven take that form is not this gate's subject. It counts norms,
        // not dispatches; the one below it counts what the fusion is for.
        let norms =
            |t: &ForwardPlan| runs(t, "rms_single_row_bfloat16") + runs(t, "rms_residual_bfloat16");
        assert_eq!(
            norms(&two) - norms(&one),
            4,
            "the two legs' extra norms and the router's:\n  beside: {:?}",
            metal_kernels(&two)
        );
        // What the fusion is for: gemma-4's three landings are three
        // statements and not six. A text that spelled them as `rms_norm`
        // then `residual_add` would still be arithmetically right and would
        // pay three more barriers a layer, which is the thing that made a
        // 128-token prefill 0.73x of the C++ shell.
        assert_eq!(
            runs(&two, "residual_add_bfloat16"),
            0,
            "a sandwich lands through the norm's epilogue:\n  beside: {:?}",
            metal_kernels(&two)
        );
        for w in [
            "layer.0.post_mlp_norm_1",
            "layer.0.mlp_norm_2",
            "layer.0.post_mlp_norm_2",
            "layer.0.router_scale",
            "layer.0.router_expert_scale",
        ] {
            assert!(binds(&two, w), "the mixture layer binds {w}");
            assert!(!binds(&one, w), "a routed-instead layer does not bind {w}");
        }
        // And the per-expert gain reaches the kernel that reads it. Naming
        // `router_topk_scaled_bfloat16` while binding four operands is the
        // shape this used to have; the weight and the symbol move together
        // now.
        assert_eq!(runs(&two, "router_topk_scaled_bfloat16"), 1);
        assert_eq!(runs(&two, "router_topk_bfloat16"), 0);
        // A ROW WITH NO EXPERTS takes no branch however the fact reads, so
        // `n_experts > 0` is load-bearing rather than a restatement.
        f.n_experts = 0;
        f.experts_per_token = 0;
        assert_eq!(
            metal_kernels(&llama_like_metal(&f, &beside, FireClass::Decode)),
            metal_kernels(&llama_like_metal(&f, &instead, FireClass::Decode))
        );
    }

    /// An unfused binding states TWO matmuls and the pair kernel; a
    /// packed one states a single matmul and the chunked kernel.
    ///
    /// The doc above `mlp` records that until 2d the second reading was a
    /// one-statement lie: the text stated the packed form either way and
    /// the executor repaired it by firing two GEMMs into workspace buffers
    /// no value described. So the thing to hold is that the two bindings
    /// state DIFFERENT texts, which is exactly what a single statement
    /// could not do.
    #[test]
    fn an_unfused_gate_up_binding_states_the_second_gemm() {
        let f = LlamaLikeFacts::qwen3_0_6b();
        let fused = LlamaLikeCudaFacts {
            gate_up_fused: true,
            ..LlamaLikeCudaFacts::qwen3_0_6b_l40s()
        };
        let unfused = LlamaLikeCudaFacts {
            gate_up_fused: false,
            ..LlamaLikeCudaFacts::qwen3_0_6b_l40s()
        };
        let matmuls = |cuda: &LlamaLikeCudaFacts| {
            llama_like_cuda(&f, cuda, FireClass::Decode)
                .layer_ops(0)
                .filter(|op| matches!(op.kind, OpKind::Matmul { .. }))
                .count()
        };
        assert_eq!(
            matmuls(&unfused),
            matmuls(&fused) + 1,
            "an unfused binding did not state the second GEMM"
        );
    }

    /// Post-norm under tensor parallel sums BOTH row-parallel projections
    /// before their norms read them.
    ///
    /// `o_proj` and `down` are row-parallel, so each rank holds a PARTIAL
    /// `[N, hidden]`. A norm reads across the hidden dimension, so norming
    /// a partial is not a rescaling of the right answer -- it is a
    /// different answer, and one no single rank can detect. The two
    /// reductions are what make the ordering right, and they exist only on
    /// this arm because the pre-norm arm lands its residual elsewhere.
    #[test]
    fn a_post_norm_layer_reduces_before_it_norms_under_tensor_parallel() {
        let mut f = LlamaLikeFacts::qwen3_0_6b();
        f.norm_placement = NormPlacement::Post;
        let sharded = LlamaLikeCudaFacts {
            tp_size: 2,
            ..LlamaLikeCudaFacts::qwen3_0_6b_l40s()
        };
        let alone = LlamaLikeCudaFacts {
            tp_size: 1,
            ..LlamaLikeCudaFacts::qwen3_0_6b_l40s()
        };
        let reductions = |cuda: &LlamaLikeCudaFacts| {
            llama_like_cuda(&f, cuda, FireClass::Decode)
                .layer_ops(0)
                .filter(|op| match &op.kind {
                    OpKind::Launch { kernel, .. } => kernel.contains("all_reduce"),
                    _ => false,
                })
                .count()
        };
        assert_eq!(
            reductions(&sharded),
            2,
            "post-norm under TP states one reduction per row-parallel \
             projection -- `o_proj` and `down`"
        );
        assert_eq!(reductions(&alone), 0);
    }

    /// The traced form of one qwen3 layer, mapped op-by-op to the kernel
    /// sequence `llama_like_forward_paged` launches on the unfused path.
    /// (The fused decode QKV kernel covers Matmul+SplitQkv+RmsnormPerHead
    /// x2+Rope+KvAppend — an emitter peephole over exactly this adjacency;
    /// see stage1-notes.md for why the trace must stay unfused.)
    ///
    /// | trace op            | hand-written kernel(s)                          |
    /// |---------------------|-------------------------------------------------|
    /// | Rmsnorm(attn_norm)  | kernels::norm::rmsnorm_bf16                              |
    /// | Matmul(qkv)         | kernels::gemm::act_x_w (qkv_proj_fused)               |
    /// | SplitQkv            | kernels::attn::split_qkv_bf16                            |
    /// | RmsnormPerHead x2 + Rope | kernels::rope::qk_rmsnorm_rope_bf16 (fused pair)    |
    /// | KvAppend            | kernels::attn::write_kv_to_pages                         |
    /// | Attention           | dispatch_attention_flashinfer_{decode,prefill}   |
    /// | Matmul(o_proj)+res  | kernels::gemm::act_x_w beta=1                         |
    /// | Rmsnorm(mlp_norm)   | kernels::norm::rmsnorm_bf16                              |
    /// | Matmul(gate_up)     | kernels::gemm::act_x_w                                |
    /// | Swiglu              | (silu-and-mul kernel)                            |
    /// | Matmul(down)+res    | kernels::gemm::act_x_w beta=1                         |
    #[test]
    fn qwen3_layer_op_sequence() {
        let plan = llama_like(&LlamaLikeFacts::qwen3_0_6b());
        let kinds: Vec<&'static str> = plan
            .layer_ops(0)
            .map(|op| match op.kind {
                OpKind::Rmsnorm { .. } => "rmsnorm",
                OpKind::Matmul {
                    beta_one: false, ..
                } => "matmul",
                OpKind::Matmul { beta_one: true, .. } => "matmul+res",
                OpKind::SplitQkv { .. } => "split_qkv",
                OpKind::RmsnormPerHead { .. } => "rmsnorm_per_head",
                OpKind::Rope { .. } => "rope",
                OpKind::KvAppend { .. } => "kv_append",
                OpKind::Attention { .. } => "attention",
                OpKind::Swiglu { .. } => "swiglu",
                _ => "other",
            })
            .collect();
        assert_eq!(
            kinds,
            [
                "rmsnorm",
                "matmul",
                "split_qkv",
                "rmsnorm_per_head",
                "rmsnorm_per_head",
                "rope",
                "kv_append",
                "attention",
                "matmul+res",
                "rmsnorm",
                "matmul",
                "swiglu",
                "matmul+res",
            ]
        );
    }

    #[test]
    fn qwen3_full_plan_shape() {
        let facts = LlamaLikeFacts::qwen3_0_6b();
        let plan = llama_like(&facts);
        // 13 ops per layer + embed + final norm + lm_head.
        assert_eq!(plan.ops.len(), 13 * facts.layers as usize + 3);
        // Weight tying: the lm head names the embedding table.
        assert!(matches!(
            &plan.ops.last().unwrap().kind,
            OpKind::LmHead { weight } if weight == "embed"
        ));
        // Logits are per-request f32 over the vocab.
        let logits = plan.ops.last().unwrap().outputs[0];
        assert_eq!(
            plan.values[logits as usize].shape.0,
            vec![Dim::Requests, Dim::Const(facts.vocab)]
        );
    }

    #[test]
    fn unfused_binding_traces_three_matmuls() {
        let facts = LlamaLikeFacts {
            fused_qkv: false,
            ..LlamaLikeFacts::qwen3_0_6b()
        };
        let plan = llama_like(&facts);
        let layer0: Vec<_> = plan.layer_ops(0).collect();
        let matmuls = layer0
            .iter()
            .filter(|op| matches!(op.kind, OpKind::Matmul { .. }))
            .count();
        // q, k, v, o_proj, gate_up, down — and no SplitQkv anywhere.
        assert_eq!(matmuls, 6);
        assert!(
            !layer0
                .iter()
                .any(|op| matches!(op.kind, OpKind::SplitQkv { .. }))
        );
    }

    /// Phi-3-mini's traced form: the qk-norm branch folds away (no
    /// RmsnormPerHead anywhere) so Rope follows the projections directly
    /// — the hand-written path's `apply_rope` with no `fuse_qk_norm_rope`
    /// kernel in sight — and the unfused binding (the dense join cannot
    /// re-fuse the contract-split q/k/v bands) traces three projection
    /// matmuls and no SplitQkv.
    #[test]
    fn phi3_layer_op_sequence() {
        let plan = llama_like(&LlamaLikeFacts::phi3_mini());
        let kinds: Vec<&'static str> = plan
            .layer_ops(0)
            .map(|op| match op.kind {
                OpKind::Rmsnorm { .. } => "rmsnorm",
                OpKind::Matmul {
                    beta_one: false, ..
                } => "matmul",
                OpKind::Matmul { beta_one: true, .. } => "matmul+res",
                OpKind::SplitQkv { .. } => "split_qkv",
                OpKind::RmsnormPerHead { .. } => "rmsnorm_per_head",
                OpKind::Rope { .. } => "rope",
                OpKind::KvAppend { .. } => "kv_append",
                OpKind::Attention { .. } => "attention",
                OpKind::Swiglu { .. } => "swiglu",
                _ => "other",
            })
            .collect();
        assert_eq!(
            kinds,
            [
                "rmsnorm",
                "matmul",
                "matmul",
                "matmul",
                "rope",
                "kv_append",
                "attention",
                "matmul+res",
                "rmsnorm",
                "matmul",
                "swiglu",
                "matmul+res",
            ]
        );
    }

    #[test]
    fn phi3_full_plan_shape() {
        let facts = LlamaLikeFacts::phi3_mini();
        let plan = llama_like(&facts);
        // 12 ops per layer (13 minus the two per-head norms and the
        // SplitQkv, plus the two extra projection matmuls) + embed +
        // final norm + lm_head.
        assert_eq!(plan.ops.len(), 12 * facts.layers as usize + 3);
        // Untied embeddings: the lm head names its own weight, not the
        // embedding table.
        assert!(matches!(
            &plan.ops.last().unwrap().kind,
            OpKind::LmHead { weight } if weight == "lm_head"
        ));
        let logits = plan.ops.last().unwrap().outputs[0];
        assert_eq!(
            plan.values[logits as usize].shape.0,
            vec![Dim::Requests, Dim::Const(facts.vocab)]
        );
    }

    /// Mistral-7B-v0.3's traced form: the fused-QKV binding keeps
    /// Matmul(qkv) + SplitQkv, but with no qk-norm the RmsnormPerHead pair
    /// between SplitQkv and Rope folds away — the one branch combination
    /// neither qwen3 (fused + qk-norm) nor phi3 (unfused + no qk-norm) had
    /// run. On this shape the executor's fused decode-QKV peephole can
    /// never fire (its predicate requires qk-norm), so SplitQkv and Rope
    /// launch as the standalone kernels.
    #[test]
    fn mistral_layer_op_sequence() {
        let plan = llama_like(&LlamaLikeFacts::mistral_7b_v03());
        let kinds: Vec<&'static str> = plan
            .layer_ops(0)
            .map(|op| match op.kind {
                OpKind::Rmsnorm { .. } => "rmsnorm",
                OpKind::Matmul {
                    beta_one: false, ..
                } => "matmul",
                OpKind::Matmul { beta_one: true, .. } => "matmul+res",
                OpKind::SplitQkv { .. } => "split_qkv",
                OpKind::RmsnormPerHead { .. } => "rmsnorm_per_head",
                OpKind::Rope { .. } => "rope",
                OpKind::KvAppend { .. } => "kv_append",
                OpKind::Attention { .. } => "attention",
                OpKind::Swiglu { .. } => "swiglu",
                _ => "other",
            })
            .collect();
        assert_eq!(
            kinds,
            [
                "rmsnorm",
                "matmul",
                "split_qkv",
                "rope",
                "kv_append",
                "attention",
                "matmul+res",
                "rmsnorm",
                "matmul",
                "swiglu",
                "matmul+res",
            ]
        );
    }

    #[test]
    fn mistral_full_plan_shape() {
        let facts = LlamaLikeFacts::mistral_7b_v03();
        let plan = llama_like(&facts);
        // 11 ops per layer (13 minus the two per-head norms) + embed +
        // final norm + lm_head.
        assert_eq!(plan.ops.len(), 11 * facts.layers as usize + 3);
        // Untied embeddings: the lm head names its own weight.
        assert!(matches!(
            &plan.ops.last().unwrap().kind,
            OpKind::LmHead { weight } if weight == "lm_head"
        ));
        let logits = plan.ops.last().unwrap().outputs[0];
        assert_eq!(
            plan.values[logits as usize].shape.0,
            vec![Dim::Requests, Dim::Const(facts.vocab)]
        );
    }

    /// OLMo-2-1B's traced form: the post-norm walk. No pre-norm before the
    /// projections — QKV reads the residual stream raw — and each
    /// sub-layer ends with the matmul(beta=0) → rmsnorm → residual_add
    /// triplet instead of one accumulate GEMM. The global qk-norm traces
    /// as plain row Rmsnorm on q and k (weight `[heads * head_dim]`, the
    /// hand-written `rmsnorm_qk` global branch), so no RmsnormPerHead
    /// appears and neither fused peephole (qk-norm+rope, decode-QKV) can
    /// ever fire — both predicates require the per-head convention.
    #[test]
    fn olmo2_layer_op_sequence() {
        let plan = llama_like(&LlamaLikeFacts::olmo2_1b());
        let kinds: Vec<&'static str> = plan
            .layer_ops(0)
            .map(|op| match op.kind {
                OpKind::Rmsnorm { .. } => "rmsnorm",
                OpKind::Matmul {
                    beta_one: false, ..
                } => "matmul",
                OpKind::Matmul { beta_one: true, .. } => "matmul+res",
                OpKind::SplitQkv { .. } => "split_qkv",
                OpKind::RmsnormPerHead { .. } => "rmsnorm_per_head",
                OpKind::Rope { .. } => "rope",
                OpKind::KvAppend { .. } => "kv_append",
                OpKind::Attention { .. } => "attention",
                OpKind::Swiglu { .. } => "swiglu",
                OpKind::ResidualAdd => "residual_add",
                _ => "other",
            })
            .collect();
        assert_eq!(
            kinds,
            [
                "matmul",  // q_proj — reads y raw: no attn pre-norm
                "matmul",  // k_proj
                "matmul",  // v_proj
                "rmsnorm", // q_norm (global: row norm over [N, Hq])
                "rmsnorm", // k_norm
                "rope",
                "kv_append",
                "attention",
                "matmul",       // o_proj, beta=0 — scratch, not the stream
                "rmsnorm",      // attn_norm on the o_proj OUTPUT
                "residual_add", // y += norm(o_proj(attn))
                "matmul",       // gate_up — reads y raw: no mlp pre-norm
                "swiglu",
                "matmul",       // down, beta=0
                "rmsnorm",      // mlp_norm on the down OUTPUT
                "residual_add", // y += norm(down(act))
            ]
        );
    }

    #[test]
    fn olmo2_full_plan_shape() {
        let facts = LlamaLikeFacts::olmo2_1b();
        let plan = llama_like(&facts);
        // 16 ops per layer + embed + final norm + lm_head.
        assert_eq!(plan.ops.len(), 16 * facts.layers as usize + 3);
        // Untied embeddings: the lm head names its own weight.
        assert!(matches!(
            &plan.ops.last().unwrap().kind,
            OpKind::LmHead { weight } if weight == "lm_head"
        ));
        let logits = plan.ops.last().unwrap().outputs[0];
        assert_eq!(
            plan.values[logits as usize].shape.0,
            vec![Dim::Requests, Dim::Const(facts.vocab)]
        );
        // No RmsnormPerHead anywhere: the global convention is a plain
        // Rmsnorm, and mistaking one for the other is different arithmetic.
        assert!(
            !plan
                .ops
                .iter()
                .any(|op| matches!(op.kind, OpKind::RmsnormPerHead { .. }))
        );
    }

    /// The global qk-norm's traced Rmsnorm ops carry the q/k projection
    /// shapes (`[Tokens, heads * head_dim]`) — one norm over the flattened
    /// heads, not `heads` norms of `head_dim` — and name the q/k norm
    /// weights.
    #[test]
    fn olmo2_global_qk_norm_is_row_rmsnorm_over_projection_width() {
        let facts = LlamaLikeFacts::olmo2_1b();
        let plan = llama_like(&facts);
        let qk_norms: Vec<_> = plan
            .layer_ops(0)
            .filter(|op| {
                matches!(&op.kind, OpKind::Rmsnorm { weight, .. }
                    if weight.ends_with("q_norm") || weight.ends_with("k_norm"))
            })
            .collect();
        assert_eq!(qk_norms.len(), 2);
        for (op, width) in qk_norms.iter().zip([facts.q_width(), facts.kv_width()]) {
            assert_eq!(
                plan.values[op.outputs[0] as usize].shape.0,
                vec![Dim::Tokens, Dim::Const(width)]
            );
        }
    }

    /// Post-norm residual dataflow: every ResidualAdd consumes the normed
    /// sub-layer output AND the residual stream it lands on, in that order
    /// (the matmul_add convention), and its input really is the Rmsnorm's
    /// output — the norm sits BETWEEN the projection and the add.
    #[test]
    fn olmo2_post_norm_residual_dataflow() {
        let plan = llama_like(&LlamaLikeFacts::olmo2_1b());
        let layer0: Vec<_> = plan.layer_ops(0).collect();
        let adds: Vec<_> = layer0
            .iter()
            .filter(|op| matches!(op.kind, OpKind::ResidualAdd))
            .collect();
        assert_eq!(adds.len(), 2);
        for add in adds {
            assert_eq!(add.inputs.len(), 2, "residual missing on {add:?}");
            let normed = add.inputs[0];
            let norm_op = layer0
                .iter()
                .find(|op| op.outputs.contains(&normed))
                .expect("producer of the add's first operand");
            assert!(
                matches!(&norm_op.kind, OpKind::Rmsnorm { weight, .. }
                    if weight.ends_with("attn_norm") || weight.ends_with("mlp_norm")),
                "post-norm add must consume a block-norm output, got {norm_op:?}"
            );
        }
        // And no beta=1 accumulate anywhere: the residual fold is illegal
        // when a norm sits between the GEMM and the stream.
        assert!(
            !plan
                .ops
                .iter()
                .any(|op| matches!(op.kind, OpKind::Matmul { beta_one: true, .. }))
        );
    }

    #[test]
    fn residual_dataflow_is_recorded() {
        let plan = llama_like(&LlamaLikeFacts::qwen3_0_6b());
        // Every accumulate consumes two values: the projection input and
        // the residual it adds into.
        for op in &plan.ops {
            if let OpKind::Matmul { beta_one: true, .. } = op.kind {
                assert_eq!(op.inputs.len(), 2, "residual missing on {op:?}");
            }
        }
    }

    /// The traced form is a stable artifact: serialize one layer and pin
    /// it. A representation change must show up as a reviewed diff here,
    /// the same discipline the loader applies to its golden plans.
    #[test]
    fn traced_form_round_trips() {
        let plan = llama_like(&LlamaLikeFacts::qwen3_0_6b());
        let json = serde_json::to_string(&plan).unwrap();
        let back: ForwardPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(plan, back);
    }
}

#[cfg(test)]
mod metal_tests {
    use self::facts::LlamaLikeMetalFacts;
    use super::*;
    use model_ir::trace::OpKind;

    /// The Metal text TRACES, and every kernel it states is declared in
    /// Metal's table.
    ///
    /// This is not a claim that the text is RIGHT — nothing has executed
    /// it, and `llama_like_metal_text`'s comment lists what is probably
    /// wrong. What it does check is the one thing that can be checked
    /// without a device: the text and the ② table agree, which is the
    /// discipline the empty table was put there to force.
    #[test]
    fn the_metal_text_states_only_declared_kernels() {
        for class in [FireClass::Decode, FireClass::Prefill] {
            // Tracing runs `kernels::check_plan` from `finish`, so an
            // undeclared symbol would have panicked before we get here.
            let plan = llama_like_metal(
                &LlamaLikeFacts::qwen3_0_6b(),
                &LlamaLikeMetalFacts::synthetic(),
                class,
            );
            assert_eq!(
                model_ir::kernels::Backend::of_family(&plan.family),
                Some(model_ir::kernels::Backend::Metal)
            );
            assert!(model_ir::kernels::check_plan(&plan).is_empty());

            let launches = plan
                .ops
                .iter()
                .filter(|op| matches!(op.kind, OpKind::Launch { .. }))
                .count();
            // EVERY op of this text is now a stated kernel. It used to be
            // "every op except the 28 `SplitQkv`s the fused binding traces":
            // the generic `split_qkv` records an `OpKind::SplitQkv`, whose two
            // widths a driver could only reach by matching on `OpKind` — which
            // is the driver knowing what a QKV split is. The Metal text states
            // the launch outright now and rides the widths on
            // `OpKind::Launch::params`, the channel built for scalars no
            // operand shape gives. So the count is exact, and that exactness
            // is the property: nothing in this text is a kind the driver has
            // to recognise.
            // Guards are not launches and never were: `OpKind::Guard` states
            // WHICH arm runs and the arms' own launches are counted above. The
            // property is unchanged — nothing in this text is a kind the
            // driver has to recognise — and a guard is the one kind the driver
            // is *supposed* to evaluate.
            let guards = plan
                .ops
                .iter()
                .filter(|op| matches!(op.kind, model_ir::trace::OpKind::Guard { .. }))
                .count();
            assert_eq!(
                launches + guards,
                plan.ops.len(),
                "every op of the metal text is a stated kernel or a guard"
            );
        }
    }

    /// The deployment facts BRANCH the text, and the branches vanish —
    /// the load-time-condition rule (`.wiki/tart/dsl.md`: "resolves once,
    /// vanishes"). A deployment without the epilogue fold states an
    /// explicit residual landing per block instead.
    #[test]
    fn the_metal_facts_resolve_at_trace_time() {
        let facts = LlamaLikeFacts::qwen3_0_6b();
        let fold = llama_like_metal(&facts, &LlamaLikeMetalFacts::synthetic(), FireClass::Decode);
        let no_fold = llama_like_metal(
            &facts,
            &LlamaLikeMetalFacts {
                fuse_residual_gemv: false,
                ..LlamaLikeMetalFacts::synthetic()
            },
            FireClass::Decode,
        );
        // By PREFIX: an affine symbol is its INSTANTIATION POINT
        // (`affine_qmv_fast_bfloat16_gs_64_b_4`), because a bare stem is not
        // an entry point any shader exports. The stems are unambiguous
        // prefixes of each other's points except for the residual pair, which
        // is why the assertions below name the residual form explicitly.
        let count = |p: &ForwardPlan, sym: &str| {
            p.ops
                .iter()
                .filter(
                    |op| matches!(&op.kind, OpKind::Launch { kernel, .. } if kernel.starts_with(sym)),
                )
                .count()
        };
        assert_eq!(count(&fold, "residual_add_bfloat16"), 0);
        // Two folds per block (o_proj and down), landed explicitly.
        assert_eq!(
            count(&no_fold, "residual_add_bfloat16"),
            2 * facts.layers as usize
        );

        // And the M>1 lane takes the GEMM where M=1 takes the GEMV.
        let mb = llama_like_metal(
            &facts,
            &LlamaLikeMetalFacts::synthetic(),
            FireClass::Prefill,
        );
        // BOTH arms of the projection guard are in the M>1 text, which is what
        // a guard is for: `qmm_t.metal` needs `M % BM == 0` and no row count
        // under `qmm_tile.0` gives it, so the text states the pair with
        // `TokensMultipleOf(tile)` and the FIRE picks. A Rust `if` would resolve
        // at trace time and leave a short prefill with nothing to run.
        assert!(
            count(&mb, "affine_qmv_fast") > 1,
            "the M>1 text carries the GEMV arm as well as the readout"
        );
        assert!(count(&mb, "affine_qmm_t_residual") > 0);
        assert!(count(&mb, "affine_qmv_fast_residual") > 0, "and its twin");
        // The attention width is the DEPLOYMENT's, not a literal. It was
        // `_d_256` unconditionally, and `qwen3_0_6b`'s heads are 128 wide — a
        // 256-wide kernel over them reads past the end of every head and
        // answers with whatever is there, which is the same defect
        // `.wiki/driver/progress-metal.md` records in the C++ llama walk. Spelling the
        // expectation from `facts.head_dim` is what stops it coming back: a
        // literal here would fail the moment a checkpoint's heads differ.
        assert_eq!(facts.head_dim, 128, "the fixture this expectation reads");
        let paged = format!("sdpa_paged_decode_bfloat16_d_{}", facts.head_dim);
        let tiled = format!("sdpa_paged_tiled_bfloat16_d_{}", facts.head_dim);
        let vector = format!("sdpa_vector_decode_bfloat16_d_{}", facts.head_dim);
        // BOTH lanes take a PAGED symbol, because the POOL is paged. This
        // expected the contiguous one at M=1 and the lane was never the
        // question: `model::kv::Pool` allocates `[page, token, head, dim]` for
        // every fire, so a decode naming `sdpa_vector_decode` walks a paged
        // pool with a contiguous kernel's arithmetic -- real memory, wrong
        // tokens, no bounds check anywhere.
        //
        // The lane picks the SHAPE. M=1 takes the per-row kernel because a
        // tile of 32 query rows is 31 rows of wasted grid at one row; a
        // multi-token fire takes the tiled one because the per-row kernel
        // re-reads the whole key run per query row, and that read is 39% of
        // prefill time at n = 2048 by the shader's own measurement.
        //
        // The two also disagree about their scalars: the paged row reads
        // `Param(1)` as `n_kv_heads` and the contiguous row reads it as `n`,
        // the key count. One statement cannot supply both, which is why the
        // paged/contiguous choice cannot be per-lane.
        //
        // BOTH ATTENTION ARMS ARE IN THE M>1 TEXT NOW, for the same reason
        // both projection arms are: the lane is not a fine enough question.
        // `multi_batch` is `class != Decode`, so a batch of eight sequences
        // each advancing by ONE token lands here -- and it is a decode in
        // every way that matters to attention, since each of the eight rows
        // is a one-token query window over its own key run. Handing it the
        // tiled kernel put eight rows in a 32-row tile that shares nothing
        // between them: 50.63 us a fire against the decode pair's 15.09, and
        // 644.98 at a long context.
        //
        // So the text states the pair under `GuardPred::WindowOne` and the
        // FIRE picks, exactly as `TokensMultipleOf` does for the projections.
        // A Rust `if` cannot: `class` is known at trace time and the window
        // shape is not.
        assert!(
            count(&mb, &tiled) > 0,
            "the M>1 text carries {tiled} for a real multi-token fire"
        );
        assert_eq!(
            count(&mb, &paged),
            facts.layers as usize,
            "and {paged} beside it, once a layer, for the batched decode"
        );
        assert!(
            count(&fold, &paged) > 0,
            "the M=1 lane must take {paged} too"
        );
        assert_eq!(
            count(&fold, &vector),
            0,
            "no contiguous attention over a paged pool"
        );
        assert_eq!(
            count(&mb, "sdpa_paged_decode_bfloat16_d_256"),
            0,
            "no 256-wide attention over 128-wide heads"
        );
    }
}
