//! `llama_like` — the family with no structural divergence.
//!
//! Three texts over one shape: a SEMANTIC trace that names operations and
//! never kernels, and one LOWERED text per backend. The semantic arm is what
//! parity holds the other two to.

pub mod emit;
pub mod facts;

use self::facts::{
    LlamaLikeCudaFacts, LlamaLikeFacts, LlamaLikeMetalFacts, NormPlacement, QkNorm,
};
use model_compiler::dsl::{
    self, add_bias, attention, cuda, matmul,
    rmsnorm, rope, split_qkv, swiglu, MatW, Val,
};
use model_compiler::trace::{
    DType, Dim, FireClass, ForwardPlan, GuardPred, RopeKind, Shape,
};

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
fn all_reduce(
    t: &model_compiler::dsl::Trace,
    x: &Val,
    hidden: u32,
    cuda: &LlamaLikeCudaFacts,
) -> Val {
    if cuda.all_reduce_p2p_max_rows == 0 {
        return cuda::all_reduce_out(x, hidden);
    }
    let shape = (
        Shape(vec![Dim::Tokens, Dim::Const(hidden)]),
        DType::BF16,
    );
    let (g, v) = dsl::guarded_value(t, x.layer(), shape);
    g.arm(GuardPred::TokensLE(cuda.all_reduce_p2p_max_rows), || {
        cuda::all_reduce_p2p(x, hidden);
    })
    .otherwise(|| {
        cuda::all_reduce_out(x, hidden);
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
        && f.q_heads % tp == 0
        && f.kv_heads % tp == 0
        && f.intermediate % tp == 0
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
fn mlp(
    x: &Val,
    w: &dsl::Layer,
    intermediate: u32,
    packed: bool,
) -> Val {
    if packed {
        cuda::swiglu(&matmul(x, &w.gate_up), intermediate, true)
    } else {
        cuda::swiglu_pair(
            &matmul(x, &w.gate_proj),
            &matmul(x, &w.up_proj),
            intermediate,
        )
    }
}

/// and the traced form states its kernels as raw signatures
/// ([`model_compiler::dsl::cuda`]; north-star-dsl.md). One trace per
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
///   fixture, and `PARITY-BATCH.md` records them as deferred with reasons
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
///   same defect `PARITY-BATCH.md` records in the C++ llama walk, where
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
        // rectangle per op. `driver-metal-new/tests/polymorphism.rs`
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
        let q_w = f.q_width();
        let kv_w = f.kv_width();
        let post_norm = f.norm_placement == NormPlacement::Post;

        // The projection arm this deployment takes, chosen once: GEMV on
        // the M=1 lane, MLX's steel GEMM above the batch gate.
        let gemm = |x: &Val, w: &MatW| {
            if multi_batch && metal.qmm_multi_batch {
                dsl::metal::qmm(x, w, &gemm_point)
            } else {
                dsl::metal::qmv(x, w, &point)
            }
        };
        // A `beta_one` matmul: the epilogue fold when the deployment has
        // it, the projection plus an explicit landing when it does not.
        let gemm_add = |x: &Val, w: &MatW, residual: &Val| {
            if metal.fuse_residual_gemv {
                if multi_batch && metal.qmm_multi_batch {
                    dsl::metal::qmm_residual(x, w, residual, &gemm_point)
                } else {
                    dsl::metal::qmv_residual(x, w, residual, &point)
                }
            } else {
                dsl::metal::residual_add(&gemm(x, w), residual)
            }
        };
        let paged = multi_batch && metal.paged_multi_batch;
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
        let gated = |x: &Val, w: &dsl::Layer| {
            dsl::metal::silu_mul(
                &gemm(x, &w.gate_proj),
                &gemm(x, &w.up_proj),
                f.intermediate,
            )
        };

        let mut y = dsl::metal::embed_gather(m.trace(), "embed", f.hidden, multi_batch, metal.proj_repr, &point);

        for l in 0..f.layers {
            let w = m.layer(l);

            let x = if post_norm {
                y.clone()
            } else {
                dsl::metal::rms_norm(&y, &w.attn_norm)
            };

            let (q, k, v) = if f.fused_qkv {
                dsl::metal::split_qkv(&gemm(&x, &w.qkv), q_w, kv_w)
            } else {
                (
                    gemm(&x, &w.q_proj),
                    gemm(&x, &w.k_proj),
                    gemm(&x, &w.v_proj),
                )
            };
            let (q, k) = if f.qk_norm == QkNorm::Off {
                (q, k)
            } else {
                (
                    dsl::metal::rms_norm(&q, &w.q_norm),
                    dsl::metal::rms_norm(&k, &w.k_norm),
                )
            };
            // One dispatch for q and k together, as `declared_dag.hpp`'s
            // `Kind::Rope` states it.
            let (q, k) = dsl::metal::rope(&q, &k, multi_batch);
            dsl::metal::kv_append(&k, &v, &w.kv, paged);
            let a = dsl::metal::sdpa(&q, &w.kv, q_w, f.head_dim, paged)
                .expect("a plain attention statement produces its value");

            if post_norm {
                let o = dsl::metal::rms_norm(&gemm(&a, &w.o_proj), &w.attn_norm);
                y = dsl::metal::residual_add(&o, &y);
                let h = gated(&y, &w);
                let d = dsl::metal::rms_norm(&gemm(&h, &w.down), &w.mlp_norm);
                y = dsl::metal::residual_add(&d, &y);
            } else {
                y = gemm_add(&a, &w.o_proj, &y);
                let x = dsl::metal::rms_norm(&y, &w.mlp_norm);
                let h = gated(&x, &w);
                y = gemm_add(&h, &w.down, &y);
            }
        }

        let normed = dsl::metal::rms_norm(&y, &m.final_norm());
        let head = if f.tied_embeddings { "embed" } else { "lm_head" };
        dsl::metal::lm_head(&normed, head, f.vocab, metal.proj_repr, &point);
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
                let per_head_fused =
                    f.qk_norm == QkNorm::PerHead && f.rope == RopeKind::Standard;
                let (q, k) = if per_head_fused {
                    cuda::qk_rmsnorm_rope(&q, &k, &w.q_norm, &w.k_norm)
                } else {
                    let (q, k) = if f.qk_norm == QkNorm::Off {
                        (q, k)
                    } else {
                        (dsl::cuda::rmsnorm(&q, &w.q_norm), dsl::cuda::rmsnorm(&k, &w.k_norm))
                    };
                    // STATED (2a). The build gate admits only Standard
                    // rope, and the executor's arm asked whether a
                    // rotary width was set to pick between two
                    // launchers -- a kernel choice from a param. This
                    // family rotates the full head, and says which
                    // kernel that is.
                    dsl::cuda::rope(&q, &k)
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
                    let window_one = class == FireClass::Decode;
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
                    let hoisted_q =
                        (!fused_post).then(|| general_qkv());
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
                    let (g, a) =
                        dsl::guarded_value(m.trace(), Some(l), attn_out_shape.clone());
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
                        if c.head_dim_padded || (window_one && c.xqa_decode) {
                            cuda::attention_flashinfer_prefill_custom(q, &w.kv, window_left);
                        } else {
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
                                    if window_one && !c.force_prefill_path {
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
                                                    q, &w.kv, window_left,
                                                );
                                            })
                                            .otherwise(|| {
                                                cuda::attention_flashinfer_decode(
                                                    q, &w.kv, window_left,
                                                );
                                            });
                                    } else {
                                        cuda::dequant_only(&w.kv);
                                        cuda::attention_flashinfer_prefill(
                                            q, &w.kv, window_left,
                                        );
                                    }
                                });
                                r.rest(|| {
                                    cuda::attention_flashinfer_prefill_custom(q, &w.kv, window_left);
                                });
                            });
                        }
                    };
                    let attn_with_sites = |q: &Val| {
                        dsl::seam(q.trace(), &dsl::seam::ATTN_Q, &[q], Some(l));
                        if !window_one {
                            // Ragged fires are row-uniform: dequant,
                            // then the score-guarded causal dispatch.
                            cuda::dequant_only(&w.kv);
                            dsl::guarded(m.trace())
                                .arm(GuardPred::WantsAttnScore, || {
                                    cuda::attention_flashinfer_prefill_capture(q, &w.kv, window_left);
                                })
                                .otherwise(|| {
                                    cuda::attention_flashinfer_prefill(q, &w.kv, window_left);
                                });
                        } else if c.xqa_decode {
                            cuda::attention_xqa_decode(q, &w.kv, window_left);
                        } else if c.force_prefill_path {
                            cuda::dequant_only(&w.kv);
                            cuda::attention_flashinfer_prefill(q, &w.kv, window_left);
                        } else {
                            dsl::guarded(m.trace())
                                .arm(GuardPred::WantsAttnScore, || {
                                    cuda::attention_flashinfer_decode_capture(q, &w.kv, window_left);
                                })
                                .otherwise(|| {
                                    cuda::attention_flashinfer_decode(q, &w.kv, window_left);
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
                FireClass::CommitAdvance | FireClass::StateOnly | FireClass::FrozenVerify => {
                    unreachable!("llama_like refuses the service classes at trace start")
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
                let x = cuda::residual_add_rmsnorm(
                    &y, &summed, &w.mlp_norm.name, f.hidden,
                );
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
    use model_compiler::trace::{Dim, OpKind};

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
                OpKind::Matmul { beta_one: false, .. } => "matmul",
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
                OpKind::Matmul { beta_one: false, .. } => "matmul",
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
                OpKind::Matmul { beta_one: false, .. } => "matmul",
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
                OpKind::Matmul { beta_one: false, .. } => "matmul",
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
                "matmul",        // q_proj — reads y raw: no attn pre-norm
                "matmul",        // k_proj
                "matmul",        // v_proj
                "rmsnorm",       // q_norm (global: row norm over [N, Hq])
                "rmsnorm",       // k_norm
                "rope",
                "kv_append",
                "attention",
                "matmul",        // o_proj, beta=0 — scratch, not the stream
                "rmsnorm",       // attn_norm on the o_proj OUTPUT
                "residual_add",  // y += norm(o_proj(attn))
                "matmul",        // gate_up — reads y raw: no mlp pre-norm
                "swiglu",
                "matmul",        // down, beta=0
                "rmsnorm",       // mlp_norm on the down OUTPUT
                "residual_add",  // y += norm(down(act))
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
    use super::*;
    use self::facts::LlamaLikeMetalFacts;
    use model_compiler::trace::OpKind;

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
                model_compiler::kernels::Backend::of_family(&plan.family),
                Some(model_compiler::kernels::Backend::Metal)
            );
            assert!(model_compiler::kernels::check_plan(&plan).is_empty());

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
            assert_eq!(
                launches,
                plan.ops.len(),
                "every op of the metal text is a stated kernel"
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
        let fold = llama_like_metal(
            &facts,
            &LlamaLikeMetalFacts::synthetic(),
            FireClass::Decode,
        );
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
        // `affine_qmv_fast` prefixes `affine_qmv_fast_residual`'s point too,
        // so the readout is the difference of the two counts.
        assert_eq!(
            count(&mb, "affine_qmv_fast") - count(&mb, "affine_qmv_fast_residual"),
            1,
            "the readout only"
        );
        assert!(count(&mb, "affine_qmm_t_residual") > 0);
        // The attention width is the DEPLOYMENT's, not a literal. It was
        // `_d_256` unconditionally, and `qwen3_0_6b`'s heads are 128 wide — a
        // 256-wide kernel over them reads past the end of every head and
        // answers with whatever is there, which is the same defect
        // `PARITY-BATCH.md` records in the C++ llama walk. Spelling the
        // expectation from `facts.head_dim` is what stops it coming back: a
        // literal here would fail the moment a checkpoint's heads differ.
        assert_eq!(facts.head_dim, 128, "the fixture this expectation reads");
        let paged = format!("sdpa_paged_decode_bfloat16_d_{}", facts.head_dim);
        let vector = format!("sdpa_vector_decode_bfloat16_d_{}", facts.head_dim);
        assert!(count(&mb, &paged) > 0, "the M>1 lane must take {paged}");
        assert!(count(&fold, &vector) > 0, "the M=1 lane must take {vector}");
        assert_eq!(
            count(&mb, "sdpa_paged_decode_bfloat16_d_256"),
            0,
            "no 256-wide attention over 128-wide heads"
        );
    }
}
