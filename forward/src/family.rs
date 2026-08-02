//! Family declarations.
//!
//! Each function here is a forward pass written as ordinary Rust over a
//! [`TraceBuilder`]; running it *is* the trace. Branches on facts execute
//! now and vanish — a deployment that binds no fused QKV traces three
//! matmuls and no split, and the traced forms differ the way two compiled
//! programs differ, not the way two runtime paths do.

use crate::facts::{
    LlamaLikeCudaFacts, LlamaLikeFacts, NormPlacement, QkNorm, Qwen35CudaFacts,
    Qwen35FullAttnFacts, Qwen35GdnFacts, Qwen35HybridFacts, Qwen35MlpKind, Qwen35MoeMlpFacts,
};
use crate::dsl::{
    self, add_bias, attention, causal_conv1d, cuda, gated_delta, gdn_prep, matmul, matmul_per_token,
    rmsnorm, rmsnorm_gated, rope, rope_partial, sigmoid_gate_add, sigmoid_gate_mul, split_gdn,
    split_q_gate, split_qkv, swiglu, topk, weighted_sum, ConvW, GdnPrepW, Kv, MatW, NormW, Rs,
    Trace, Val,
};
use crate::trace::{
    DType, Dim, FireClass, ForwardPlan, GuardPred, HookStage, NormVariant, RopeKind, Shape,
};

/// The lowering a qwen3_5 body traces under, threaded by value: the CUDA
/// backend facts and the fire class the class arms match on. `None` is
/// the semantic form.
type Qwen35Lower<'a> = Option<(&'a Qwen35CudaFacts, FireClass)>;

/// The llama_like body — SEMANTIC form: no structural divergence, one
/// trace serves every fire shape, kernel choice stays with the consumer.
/// [`llama_like_cuda`] is the same text with the class arms live.
///
/// Mirrors `driver/cuda/src/model/llama_like/llama_like.cpp`
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
///   gemm → `launch_rmsnorm_bf16` → `launch_residual_add_bf16` triplet.
pub fn llama_like(facts: &LlamaLikeFacts) -> ForwardPlan {
    llama_like_text(facts, None)
}

/// The LOWERED llama_like: the SAME text as [`llama_like`], traced with
/// the CUDA backend facts and a fire class in hand, so the class arms run
/// and the traced form states its kernels as raw signatures
/// ([`crate::dsl::cuda`]; north-star-dsl.md). One trace per
/// [`FireClass`]; family names `llama_like.cuda.decode` / `.prefill`.
pub fn llama_like_cuda(
    facts: &LlamaLikeFacts,
    cuda: &LlamaLikeCudaFacts,
    class: FireClass,
) -> ForwardPlan {
    llama_like_text(facts, Some((cuda, class)))
}

/// THE one llama_like text (north-star-dsl.md): computation and kernel
/// choice together, on the dsl surface. With `lower: None` this is the
/// semantic trace — the general arm everywhere, no kernel stated,
/// byte-identical to what `llama_like` always produced (the goldens pin
/// it). With a lowering, the class arms run as ordinary trace-time
/// matches beside the fact arms, and what they choose is exactly what
/// `declared_forward.cpp` chooses at fire time today — the migration
/// deletes the C++ copy of these matches, not this one.
fn llama_like_text(
    facts: &LlamaLikeFacts,
    lower: Option<(&LlamaLikeCudaFacts, FireClass)>,
) -> ForwardPlan {
    dsl::trace(facts, lower, |m| {
        let f = m.facts().clone();
        let q_w = f.q_width();
        let kv_w = f.kv_width();
        let post_norm = f.norm_placement == NormPlacement::Post;
        let cuda_of = |class_want: FireClass| {
            m.lowering()
                .filter(|(_, class)| *class == class_want)
                .map(|(c, _)| c)
        };

        // The fused decode-QKV arm's predicate: the model-fact terms
        // here, the load-time backend terms on the facts struct — term
        // for term the hand-written `fused_decode_qkv_post`
        // (declared_forward.cpp:465-479), written where it belongs.
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
            .then(|| cuda::rope_standard_table(m));

        for l in 0..f.layers {
            let w = m.layer(l);

            // Attention block: (pre-norm) -> qkv -> (q/k norms) -> rope
            // -> append -> attention -> o_proj landed on the residual.
            let x = if post_norm {
                y.clone()
            } else {
                rmsnorm(&y, &w.attn_norm)
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
                if m.lowering().is_some() {
                    // The §5.1 lora correction: the adapter delta lands
                    // on the just-materialized RAW q/v projections,
                    // BEFORE anything consumes them — bias, norms, rope,
                    // the KV append (the hand-written apply's position;
                    // correcting after rope is different arithmetic, the
                    // bug the first live A/B caught). A guard with an
                    // EMPTY else: a fire with no usable lanes launches
                    // nothing.
                    dsl::guard(
                        m,
                        GuardPred::HasLora,
                        || cuda::lora_qkv_correction(&q, &v, l),
                        || {},
                    );
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
                // A lowered arm with the per-head convention and Standard
                // rope states the fused norm+rope kernel (the hand-written
                // `fuse_qk_norm_rope` branch — bf16 rounds differently
                // from the triple, so parity requires the same launch);
                // the Global and Off conventions keep the semantic ops,
                // whose kernels are 1:1.
                let per_head_fused = m.lowering().is_some()
                    && f.qk_norm == QkNorm::PerHead
                    && f.rope == RopeKind::Standard;
                let (q, k) = if per_head_fused {
                    cuda::qk_rmsnorm_rope(&q, &k, &w.q_norm, &w.k_norm)
                } else {
                    let (q, k) = if f.qk_norm == QkNorm::Off {
                        (q, k)
                    } else {
                        (rmsnorm(&q, &w.q_norm), rmsnorm(&k, &w.k_norm))
                    };
                    rope(&q, &k, f.rope)
                };
                if m.lowering().is_some() {
                    // The KV-write mechanism is a per-fire runtime input
                    // (explicit descriptors when the fire steers a graph
                    // replay, page-derived otherwise). Under the fused
                    // deployment's mask arm this guard NESTS inside the
                    // HasCustomMask guard (A1 — the walk keeps a stack).
                    dsl::guard(
                        m,
                        GuardPred::HasWriteDesc,
                        || cuda::write_kv_explicit(&k, &v, &w.kv),
                        || cuda::write_kv_to_pages(&k, &v, &w.kv),
                    );
                } else {
                    w.kv.append(&k, &v);
                }
                q
            };
            let attn_out_shape = (
                Shape(vec![Dim::Tokens, Dim::Const(q_w)]),
                DType::BF16,
            );

            let a = match m.lowering() {
                None => {
                    let q = general_qkv();
                    attention(&q, &w.kv, q_w)
                }
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
                Some((c, FireClass::Decode)) => {
                    // ORDER IS LOAD-BEARING: `guarded_value` OPENS the
                    // chain, and every op recorded after it counts into
                    // the first arm's region. The non-fused deployment's
                    // general QKV must therefore trace BEFORE the guard
                    // opens (the hoisted `q` below) — tracing it after
                    // put the whole QKV sequence inside the mask arm,
                    // and every unmasked fire skipped it (the phi3/
                    // mistral live-garbage regression, caught 2026-08-03
                    // by the three-model battery; the mistral lowered
                    // goldens now pin this structure).
                    let hoisted_q =
                        (!fused_post).then(|| general_qkv());
                    let (g, a) =
                        dsl::guarded_value(m.trace(), Some(l), attn_out_shape.clone());
                    // The masked attention states its SPATIAL SPLIT as
                    // vocabulary (NS-4 landed in the IR): a Peel on the
                    // unmasked-prefix axis — the deployment's decode
                    // dispatch serves the plain prefix rows, the custom
                    // dispatch the masked suffix, the split a runtime
                    // input, UNPLANNED collapsing to tail-only full-N
                    // (the fire-level dispatch as the peel's endpoint).
                    // Padded head dims keep the fire-level word (the
                    // split's row offsets are logical-width, the padded
                    // staging is not), and XQA deployments too (the XQA
                    // fire-wide prepare is R-shaped) — both mirror the
                    // prepare gate exactly, so the trace never states a
                    // split prepare refuses to plan.
                    let masked_attention = |q: &Val| {
                        if c.xqa_decode || c.head_dim_padded {
                            cuda::attention_flashinfer_prefill_custom_region(q, &w.kv);
                        } else {
                            dsl::peel_masked(
                                m.trace(),
                                Some(l),
                                || {
                                    // The prefix states THE DEPLOYMENT'S
                                    // decode form, windowed to the plain
                                    // rows: the force_prefill fallback
                                    // (GQA ratio outside the decode
                                    // kernel's set) is the plan-free
                                    // prefill dispatch behind its
                                    // dequant staging, everyone else
                                    // the planned decode dispatch —
                                    // `attn_with_sites`' choice, minus
                                    // the sites (a planned split never
                                    // carries hooks).
                                    if c.force_prefill_path {
                                        cuda::dequant_only(&w.kv);
                                        cuda::attention_flashinfer_prefill_region(
                                            q, &w.kv,
                                        );
                                    } else {
                                        cuda::attention_flashinfer_decode_region(
                                            q, &w.kv,
                                        );
                                    }
                                },
                                || {
                                    cuda::attention_flashinfer_prefill_custom_region(
                                        q, &w.kv,
                                    )
                                },
                            );
                        }
                    };
                    let attn_with_sites = |q: &Val| {
                        dsl::hook_site(HookStage::OnAttnProj, q, l);
                        if c.xqa_decode {
                            cuda::attention_xqa_decode_region(q, &w.kv);
                        } else if c.force_prefill_path {
                            cuda::dequant_only(&w.kv);
                            cuda::attention_flashinfer_prefill_region(q, &w.kv);
                        } else {
                            dsl::guarded(m)
                                .arm(GuardPred::WantsAttnScore, || {
                                    cuda::attention_flashinfer_decode_capture(q, &w.kv)
                                })
                                .otherwise(|| {
                                    cuda::attention_flashinfer_decode_region(q, &w.kv)
                                });
                        }
                        dsl::hook_site(HookStage::OnAttn, q, l);
                    };
                    if fused_post {
                        g.arm(GuardPred::HasCustomMask, || {
                            // Masked+hooked composes here: the sites run
                            // around the custom dispatch exactly as the
                            // hand-written unconditional invokes do. No
                            // WantsAttnScore guard — the custom dispatch
                            // has no capture variant, so nothing
                            // publishes and the OnAttn sideband hands
                            // the programs a null scores pointer (the
                            // publish-gated contract).
                            let q = general_qkv();
                            dsl::hook_site(HookStage::OnAttnProj, &q, l);
                            masked_attention(&q);
                            dsl::hook_site(HookStage::OnAttn, &q, l);
                        })
                        // The lora arm: the fused epilogue writes V
                        // straight to the paged cache — nothing exists to
                        // correct into — so a lora fire runs the whole
                        // general sequence (whose internal HasLora guard
                        // lands the correction), full-N: the hand-written
                        // `!has_lora` predicate term, stated as an arm.
                        // Mask+lora composes in the mask arm above (its
                        // general body carries the same internal guard).
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
                            let q = dsl::peel(
                                m.trace(),
                                Some(l),
                                attn_out_shape.clone(),
                                || {
                                    cuda::qkv_decode_qk_norm_rope_write_kv_region(
                                        &packed,
                                        &w.q_norm,
                                        &w.k_norm,
                                        &w.kv,
                                        table.as_ref(),
                                    );
                                },
                                || {
                                    let (qt, kt, vt) = split_qkv(&packed, q_w, kv_w);
                                    let (_qt, kt) =
                                        cuda::qk_rmsnorm_rope(&qt, &kt, &w.q_norm, &w.k_norm);
                                    dsl::guard(
                                        m,
                                        GuardPred::HasWriteDesc,
                                        || cuda::write_kv_explicit(&kt, &vt, &w.kv),
                                        || cuda::write_kv_to_pages(&kt, &vt, &w.kv),
                                    );
                                },
                            );
                            attn_with_sites(&q);
                        });
                    } else {
                        let q = hoisted_q.as_ref().expect("hoisted for the non-fused arm");
                        g.arm(GuardPred::HasCustomMask, || {
                            // Masked+hooked (the fused arm's comment).
                            dsl::hook_site(HookStage::OnAttnProj, q, l);
                            masked_attention(q);
                            dsl::hook_site(HookStage::OnAttn, q, l);
                        })
                        .otherwise(|| attn_with_sites(q));
                    }
                    a
                }
                Some((c, FireClass::Prefill)) => {
                    let q = general_qkv();
                    let (g, a) =
                        dsl::guarded_value(m.trace(), Some(l), attn_out_shape.clone());
                    g.arm(GuardPred::HasCustomMask, || {
                        // The custom dispatch takes the layer view whole —
                        // no dequant staging (the hand-written custom-mask
                        // branch's contract). Masked+hooked composes: the
                        // sites bracket the dispatch (null scores at
                        // OnAttn — no capture variant publishes).
                        dsl::hook_site(HookStage::OnAttnProj, &q, l);
                        // The MIXED FIRE: the prefill-class mask arm
                        // states the same UnmaskedPrefix peel as the
                        // decode class — prefix region = the causal
                        // dispatch behind its dequant staging (any mix
                        // of prefill and plain-decode requests, ragged
                        // qo), tail = the custom dispatch over the
                        // masked 1-token suffix. UNPLANNED (prepare
                        // declined the shape, hooks/lora, disarmed
                        // gate) collapses to tail-only full-N — the
                        // fire-level custom dispatch as the peel's
                        // endpoint. Padded head dims keep the
                        // fire-level word (prepare's gate mirror).
                        if c.head_dim_padded {
                            cuda::attention_flashinfer_prefill_custom_region(
                                &q, &w.kv,
                            );
                        } else {
                            dsl::peel_masked(
                                m.trace(),
                                Some(l),
                                || {
                                    cuda::dequant_only(&w.kv);
                                    cuda::attention_flashinfer_prefill_region(
                                        &q, &w.kv,
                                    );
                                },
                                || {
                                    cuda::attention_flashinfer_prefill_custom_region(
                                        &q, &w.kv,
                                    )
                                },
                            );
                        }
                        dsl::hook_site(HookStage::OnAttn, &q, l);
                    })
                    .otherwise(|| {
                        // Prefill has no fused post, so no Peel: the body
                        // is row-uniform — sites (argument no-ops when
                        // unhooked), dequant, the score-guarded dispatch.
                        dsl::hook_site(HookStage::OnAttnProj, &q, l);
                        cuda::dequant_only(&w.kv);
                        dsl::guarded(m)
                            .arm(GuardPred::WantsAttnScore, || {
                                cuda::attention_flashinfer_prefill_capture(&q, &w.kv)
                            })
                            .otherwise(|| {
                                cuda::attention_flashinfer_prefill_region(&q, &w.kv)
                            });
                        dsl::hook_site(HookStage::OnAttn, &q, l);
                    });
                    a
                }
                Some((
                    _,
                    FireClass::CommitAdvance | FireClass::StateOnly | FireClass::FrozenVerify,
                )) => {
                    unreachable!("llama_like refuses the service classes at trace start")
                }
            };
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

        m.logits(&rmsnorm(&y, &m.final_norm()));
    })
}

/// One qwen3_5_moe MoE MLP block, traced standalone — the first `dyn`
/// fragment.
///
/// This is a FRAGMENT, not a model: the unit the future qwen3_5 declaration
/// composes per layer (`y += moe_mlp(l, rmsnorm(y, mlp_norm))`), traced
/// against layer 0 with the residual stream as a fragment parameter
/// ([`dsl::input`]). A full qwen3_5 declaration also needs the
/// hybrid GDN attention vocabulary (`causal_conv1d`, `gated_delta`, gated
/// rmsnorm, per-request recurrent state) — out of scope here, a separate
/// rung; see [`Qwen35MoeMlpFacts`].
///
/// Mirrors `qwen3_5_moe_forward.cpp::run_moe_mlp` launch for launch, in the
/// decode fast path's one-launch-per-op form (the canonical granularity;
/// the prefill path's host-routed per-expert gather/GEMM/scatter loop is a
/// LOWERING of ops 4–7, as is the CUTLASS fused pipeline — both the
/// emitter's per-fire choice):
///
/// | trace op                       | hand-written kernel(s)                      |
/// |--------------------------------|---------------------------------------------|
/// | Rmsnorm(mlp_norm)              | launch_rmsnorm_gemma_bf16                   |
/// | Matmul(router)                 | ops::gemm_act_x_wt_bf16 (router logits)     |
/// | TopK                           | launch_topk_softmax_bf16                    |
/// | Matmul(expert.{e}.gate_up, sel)| grouped GEMM (batched/aligned/CUTLASS)      |
/// | Swiglu                         | launch_chunked_swiglu_bf16 over N*k rows    |
/// | Matmul(expert.{e}.down, sel)   | grouped GEMM (batched/aligned/CUTLASS)      |
/// | WeightedSum                    | launch_token_batched_weighted_sum_bf16      |
/// | Matmul(shared_expert.gate_up)  | ops::gemm_act_x_w                           |
/// | Swiglu                         | launch_chunked_swiglu_bf16                  |
/// | Matmul(shared_expert.down)     | ops::gemm_act_x_w                           |
/// | Matmul(shared_expert_gate)     | ops::gemm_act_x_w ([Tokens, 1] logit)       |
/// | SigmoidGateAdd                 | launch_sigmoid_scalar_gate_add_bf16         |
/// | ResidualAdd                    | launch_residual_add_bf16                    |
///
/// The five shared-expert ops fold away when the facts say the checkpoint
/// has none (`shared_expert_intermediate == 0`, the qwen3_moe shape), the
/// same way llama_like's branches fold: at trace time, leaving no trace.
pub fn qwen3_5_moe_mlp_block(facts: &Qwen35MoeMlpFacts) -> ForwardPlan {
    dsl::trace_named("qwen3_5_moe_mlp_block", |t| {
        // The fragment's parameter: the residual stream entering the block.
        let y = dsl::input(t, facts.hidden);
        moe_mlp_body(0, facts, &y);
    })
}

/// The MoE MLP block's weight namespace at layer `l`: the router, the
/// `{e}`-templated expert banks, and the shared expert's three handles —
/// eager strings, [`crate::dsl::Layer`]-style, so a checkpoint without a
/// shared expert simply never reads them.
struct MoeLayerW {
    mlp_norm: NormW,
    router: MatW,
    expert_gate_up: MatW,
    expert_down: MatW,
    shared_gate_up: MatW,
    shared_down: MatW,
    shared_gate: MatW,
}

impl MoeLayerW {
    fn new(l: u32, f: &Qwen35MoeMlpFacts) -> Self {
        let w = |name: &str| format!("layer.{l}.{name}");
        let mat = |name: &str, width: u32| MatW {
            name: w(name),
            width,
            layer: Some(l),
        };
        MoeLayerW {
            mlp_norm: NormW {
                name: w("mlp_norm"),
                variant: f.norm_variant,
                per_head: None,
                layer: Some(l),
            },
            router: mat("router", f.num_experts),
            expert_gate_up: mat("expert.{e}.gate_up", 2 * f.moe_intermediate),
            expert_down: mat("expert.{e}.down", f.hidden),
            shared_gate_up: mat("shared_expert.gate_up", 2 * f.shared_expert_intermediate),
            shared_down: mat("shared_expert.down", f.hidden),
            shared_gate: mat("shared_expert_gate", 1),
        }
    }
}

/// The MoE MLP block's op emission at layer `l` — the unit
/// [`qwen3_5_moe_mlp_block`] traces standalone (at layer 0) and
/// [`qwen3_5_hybrid`] composes per layer. One body so the hybrid's MLP ops
/// ARE the fragment's, by construction rather than by parallel maintenance.
fn moe_mlp_body(l: u32, facts: &Qwen35MoeMlpFacts, y: &Val) -> Val {
    let w = MoeLayerW::new(l, facts);
    let mut y = y.clone();

    let m = rmsnorm(&y, &w.mlp_norm);

    // Routed experts: router -> topk -> grouped gate_up -> swiglu ->
    // grouped down -> per-token weighted combine.
    let logits = matmul(&m, &w.router);
    let (experts, weights) = topk(&logits, facts.top_k);
    let gate_up = matmul_per_token(&m, &w.expert_gate_up, &experts);
    let act = swiglu(&gate_up, facts.moe_intermediate);
    let down = matmul_per_token(&act, &w.expert_down, &experts);
    let routed = weighted_sum(&weights, &down);

    // Shared expert (qwen3.5/3.6-MoE: always-on dense MLP behind a
    // per-token sigmoid scalar gate; absent on qwen3_moe).
    let combined = if facts.shared_expert_intermediate > 0 {
        let inter = facts.shared_expert_intermediate;
        let act = swiglu(&matmul(&m, &w.shared_gate_up), inter);
        let shared = matmul(&act, &w.shared_down);
        let gate = matmul(&m, &w.shared_gate);
        sigmoid_gate_add(&shared, &gate, &routed)
    } else {
        routed
    };

    // Not a fresh matmul, so `+=` records the explicit ResidualAdd.
    y += combined;
    y
}

/// One qwen3_5 GDN (gated-deltanet) linear-attention block, traced
/// standalone — the second fragment, and the other layer kind of the
/// qwen3.5 hybrid.
///
/// This is a FRAGMENT, not a model: the unit the qwen3_5 declaration
/// composes on a `Linear` layer (`y += gdn(l, rmsnorm(y, attn_norm))`,
/// plan.md Part 1's `match layers[l]`), traced against layer 0 with the
/// residual stream as a fragment parameter ([`dsl::input`]),
/// exactly the MoE fragment's shape. The FULL-attention layer kind of this
/// family — not llama_like's: q_proj 2× wide with the per-head
/// `[query | gate]` split, sigmoid output gate, partial rope, Gemma-fold
/// per-head norms — is its own fragment, [`qwen3_5_full_attn_block`], and
/// [`qwen3_5_hybrid`] composes all three bodies into the full model.
///
/// Mirrors `qwen3_5_forward.cpp::linear_attn_layer_body` launch for launch
/// on the TP=1 decode fast path (the canonical granularity; the prefill
/// conv/recurrence walks, the batched slot-indirected variants, the
/// warp-tiled/cached/FLA recurrence kernels and the GQA
/// `repeat_interleave` materialization are all LOWERINGS of ops 5–7, the
/// emitter's per-fire choice — as are the verify-stash and rs-buffer
/// scatter/gather paths, which are speculative-decode services around the
/// same ops, not ops of the pass):
///
/// | trace op                | hand-written kernel(s)                          |
/// |-------------------------|--------------------------------------------------|
/// | Rmsnorm(attn_norm)      | launch_rmsnorm_gemma_bf16                        |
/// | Matmul(in_proj_qkv)     | ops::gemm_act_x_w                                |
/// | Matmul(in_proj_z)       | ops::gemm_act_x_w                                |
/// | Matmul(in_proj_a)       | ops::gemm_act_x_w                                |
/// | Matmul(in_proj_b)       | ops::gemm_act_x_w                                |
/// | CausalConv1d            | launch_causal_conv1d_update[_batched]_bf16       |
/// | GdnPrep                 | launch_qwen_gdn_post_conv_prep_bf16              |
/// | GatedDelta              | launch_recurrent_gated_delta_step_* (decode)     |
/// | RmsnormGated            | launch_rmsnorm_gated_fp32_in_bf16                |
/// | Matmul(o_proj)+res      | ops::gemm_act_x_w beta=1                         |
///
/// With the fused binding (`fused_in_proj`, `PIE_QWEN35_FUSED_GDN_PROJ`)
/// the four projections become two matmuls + two [`SplitGdn`] launches
/// (`launch_split_bf16_rows`, `launch_split_qwen_gdn_ba_bf16`) — same op
/// count, different ops, resolved at trace time like llama_like's
/// `fused_qkv`.
///
/// `CausalConv1d` and `GatedDelta` address the layer's PER-REQUEST
/// conv/recurrent state — implicit, marked by the op kinds themselves
/// ([`crate::trace::OpKind::state_ref`]); see the trace module doc's "the
/// per-request state axis" for why the state is not a traced value.
///
/// [`SplitGdn`]: crate::trace::OpKind::SplitGdn
pub fn qwen3_5_gdn_block(facts: &Qwen35GdnFacts) -> ForwardPlan {
    dsl::trace_named("qwen3_5_gdn_block", |t| {
        // The fragment's parameter: the residual stream entering the block.
        let y = dsl::input(t, facts.hidden);
        gdn_attn_body(t, 0, facts, &y, None);
    })
}

/// The GDN block's weight namespace at layer `l`: both in-projection
/// bindings (fused and unfused — eager strings, only the traced branch
/// reads its handles), the conv/prep weight pairs, and the layer's
/// per-request recurrent state.
struct GdnLayerW {
    attn_norm: NormW,
    in_proj_qkvz: MatW,
    in_proj_ba: MatW,
    in_proj_qkv: MatW,
    in_proj_z: MatW,
    in_proj_a: MatW,
    in_proj_b: MatW,
    conv: ConvW,
    prep: GdnPrepW,
    gate_norm: NormW,
    o_proj: MatW,
    rs: Rs,
}

impl GdnLayerW {
    fn new(t: &Trace, l: u32, f: &Qwen35GdnFacts) -> Self {
        let conv_dim = f.conv_dim();
        let v_dim = f.value_width();
        let w = |name: &str| format!("layer.{l}.{name}");
        let mat = |name: &str, width: u32| MatW {
            name: w(name),
            width,
            layer: Some(l),
        };
        GdnLayerW {
            attn_norm: NormW {
                name: w("attn_norm"),
                variant: f.norm_variant,
                per_head: None,
                layer: Some(l),
            },
            in_proj_qkvz: mat("in_proj_qkvz", conv_dim + v_dim),
            in_proj_ba: mat("in_proj_ba", 2 * f.value_heads),
            in_proj_qkv: mat("in_proj_qkv", conv_dim),
            in_proj_z: mat("in_proj_z", v_dim),
            in_proj_a: mat("in_proj_a", f.value_heads),
            in_proj_b: mat("in_proj_b", f.value_heads),
            conv: ConvW {
                name: w("conv"),
                kernel: f.conv_kernel,
                layer: l,
            },
            prep: GdnPrepW {
                a_log: w("a_log"),
                dt_bias: w("dt_bias"),
                layer: l,
            },
            // The gated norm's fold is Plain by construction (the op
            // carries no variant); the handle contributes name and layer.
            gate_norm: NormW {
                name: w("gate_norm"),
                variant: NormVariant::Plain,
                per_head: None,
                layer: Some(l),
            },
            o_proj: mat("o_proj", f.hidden),
            rs: Rs::at(t, l),
        }
    }
}

/// The GDN linear-attention block's op emission at layer `l` — the unit
/// [`qwen3_5_gdn_block`] traces standalone (at layer 0) and
/// [`qwen3_5_hybrid`] composes on every `Linear` layer. One body so the
/// hybrid's GDN ops ARE the fragment's, by construction.
///
/// ONLY the kernel CHOICES lower under `Some(lower)`: the conv (decode
/// update vs prefill walk) and the recurrence (the decode step's four
/// name variants; the prefill three-way behind the first value-producing
/// guard chain). Everything else — the norms, the in-projections and
/// their fused/unfused splits, `gdn_prep`, the gated norm, the o_proj
/// fold — is a 1:1-kernel semantic op and stays semantic in every form.
/// The GDN in-projections against `w`'s layer: the fused/unfused branch
/// resolves at trace time (a binding fact); operand packing mirrors the
/// driver's: qkvz = [mixed_qkv | z], ba = [b | a]. Returns
/// `(qkv, z, a, b)`. One function so the CommitAdvance pass's no-stash
/// arm ([`commit_advance_body`]) runs EXACTLY the normal body's GEMMs
/// and splits, by construction rather than by parallel maintenance.
fn gdn_in_proj(x: &Val, w: &GdnLayerW, facts: &Qwen35GdnFacts) -> (Val, Val, Val, Val) {
    if facts.fused_in_proj {
        let qkvz = matmul(x, &w.in_proj_qkvz);
        let (qkv, z) = split_gdn(&qkvz, facts.conv_dim(), facts.value_width());
        let ba = matmul(x, &w.in_proj_ba);
        let (b, a) = split_gdn(&ba, facts.value_heads, facts.value_heads);
        (qkv, z, a, b)
    } else {
        (
            matmul(x, &w.in_proj_qkv),
            matmul(x, &w.in_proj_z),
            matmul(x, &w.in_proj_a),
            matmul(x, &w.in_proj_b),
        )
    }
}

fn gdn_attn_body(
    t: &Trace,
    l: u32,
    facts: &Qwen35GdnFacts,
    y: &Val,
    lower: Qwen35Lower<'_>,
) -> Val {
    let w = GdnLayerW::new(t, l, facts);
    let mut y = y.clone();

    let x = rmsnorm(&y, &w.attn_norm);

    let (qkv, z, a, b) = gdn_in_proj(&x, &w, facts);

    // FrozenVerify: the frozen verify pass caches the cheap in-proj
    // activations for the later commit-advance replay — the stash STORE,
    // at the hand-written launch position (after the splits, before the
    // conv). Present iff the deployment configures the stash (the same
    // engine-owned fact the load consumes); everything else in this body
    // rides the Prefill arms, and write_state=false is a runtime ARG of
    // the stated kernels, not a trace difference.
    if let Some((c, FireClass::FrozenVerify)) = lower {
        if c.verify_stash {
            cuda::verify_stash_store(&qkv, &a, &b, &w.rs);
        }
    }

    // Conv → prep → recurrence: the GDN core, against the layer's
    // per-request conv/recurrent state. A class arm states the conv
    // kernel; the semantic form keeps the opaque op.
    // StateOnly is prefill-shaped throughout the backbone — it takes the
    // Prefill arm in every kernel choice here; only the model epilogue
    // differs, and that class match lives in `qwen3_5_hybrid_text`.
    // CommitAdvance never enters this body at all: it is its own pass
    // ([`commit_advance_body`]), not a variant of the layer loop.
    let qkv = match lower {
        None => causal_conv1d(&qkv, &w.conv),
        Some((_, FireClass::Decode)) => cuda::gdn_conv_update_batched(&qkv, &w.conv, &w.rs),
        Some((_, FireClass::Prefill | FireClass::StateOnly | FireClass::FrozenVerify)) => {
            cuda::gdn_conv_prefill_batched(&qkv, &w.conv, &w.rs)
        }
        Some((_, FireClass::CommitAdvance)) => {
            unreachable!("CommitAdvance traces its own pass, never the layer body")
        }
    };
    let (q, k, v, g, beta) = gdn_prep(
        &qkv,
        &a,
        &b,
        &w.prep,
        facts.key_heads,
        facts.key_head_dim,
        facts.value_heads,
        facts.value_head_dim,
    );
    // The OnAttnProj site (A4): the hand-written GDN body invokes the
    // fire's programs here observing q_pre (fp32; qwen3_5's sites are
    // OBSERVATION-only — no page-mask sink, no score capture). Lowered
    // traces only; a fire with nothing attached passes by argument.
    // (The hand-written invoke sits after the cached family's GQA
    // repeats; the repeats read q_pre and never write it, so observing
    // before the recurrence guard sees the same bytes.)
    if lower.is_some() {
        dsl::hook_site(HookStage::OnAttnProj, &q, l);
    }
    // GQA (value heads sharing fewer key heads) picks the `_gqa` decode
    // step; the prefill kernels state their own layout handling.
    let gqa = facts.value_heads != facts.key_heads;
    let core = match lower {
        None => gated_delta(&w.rs, &q, &k, &v, &g, &beta),
        Some((c, FireClass::Decode)) => {
            cuda::gdn_step_batched(&q, &k, &v, &g, &beta, &w.rs, gqa, c.state_bf16)
        }
        Some((
            c,
            FireClass::Prefill | FireClass::StateOnly | FireClass::FrozenVerify,
        )) => {
            // The prefill recurrence three-way, as the first
            // VALUE-PRODUCING guard chain (north-star-dsl.md 4b): the
            // guard's output is the recurrence core — the same
            // `[Tokens, Vh, Vd]` f32 the semantic `gated_delta`
            // produces — and each arm's launch binds that buffer,
            // recording no SSA outputs of its own. Arm order is the
            // hand-written probe order: warp-tiled (when eligible at
            // all — a fact), then the cached family (whose kernels
            // index the REPEATED head layout, so the GQA repeats
            // materialize INSIDE its arm and nowhere else — launch
            // order matches the hand-written stream order: prep,
            // [repeats], recurrence), else the batched GQA-aware FLA.
            let out_shape = (
                Shape(vec![
                    Dim::Tokens,
                    Dim::Const(facts.value_heads),
                    Dim::Const(facts.value_head_dim),
                ]),
                DType::F32,
            );
            let (mut guard, core) = dsl::guarded_value(t, Some(l), out_shape);
            if c.warp_tiled {
                guard = guard.arm(GuardPred::TokensLE(c.warp_tiled_max), || {
                    cuda::gdn_prefill_warp_tiled(&q, &k, &v, &g, &beta, &w.rs, gqa, c.state_bf16)
                });
            }
            guard
                .arm(GuardPred::TokensLE(c.cached_max), || {
                    if gqa {
                        cuda::repeat_interleave_heads(&q);
                        cuda::repeat_interleave_heads(&k);
                    }
                    cuda::gdn_prefill_cached(&q, &k, &v, &g, &beta, &w.rs, c.state_bf16)
                })
                .otherwise(|| cuda::gdn_prefill_fla(&q, &k, &v, &g, &beta, &w.rs, c.state_bf16));
            core
        }
        Some((_, FireClass::CommitAdvance)) => {
            unreachable!("CommitAdvance traces its own pass, never the layer body")
        }
    };
    // The OnAttn site: after the recurrence core, before the gated norm
    // — the hand-written invoke's position (observing q_pre again).
    if lower.is_some() {
        dsl::hook_site(HookStage::OnAttn, &q, l);
    }

    // Gated norm (z-gated, per-head, plain fold) → o_proj landed on
    // the residual (`+=` of a fresh matmul IS the beta=1 fold).
    let o = rmsnorm_gated(&core, &z, &w.gate_norm);
    y += matmul(&o, &w.o_proj);
    y
}

/// One qwen3_5 FULL-attention block, traced standalone — the third
/// fragment, and the last layer kind the qwen3.5 hybrid needed.
///
/// This is a FRAGMENT, not a model: the unit [`qwen3_5_hybrid`] composes on
/// a `Full` layer (plan.md Part 1's `match layers[l] { Full => full_attn(l,
/// x), .. }`), traced against layer 0 with the residual stream as a
/// fragment parameter ([`dsl::input`]), exactly the MoE and GDN
/// fragments' shape.
///
/// Mirrors `qwen3_5_forward.cpp::full_attn_layer_body` launch for launch on
/// the TP=1 path (the canonical granularity; decode vs prefill vs
/// small-naive attention plans, the explicit KV-write descriptor branch and
/// the TP all-reduce/residual split are all LOWERINGS the emitter picks per
/// fire), on the default (unfused) binding:
///
/// | trace op                  | hand-written kernel(s)                       |
/// |---------------------------|-----------------------------------------------|
/// | Rmsnorm(attn_norm)        | launch_rmsnorm_gemma_bf16                     |
/// | Matmul(q_proj) [2×-wide]  | ops::gemm_act_x_w → [N, 2·Hq]                 |
/// | Matmul(k_proj)            | ops::gemm_act_x_w → [N, Hk]                   |
/// | Matmul(v_proj)            | ops::gemm_act_x_w → [N, Hk]                   |
/// | SplitQGate                | launch_split_q_gate_bf16 (per-head q‖gate)    |
/// | RmsnormPerHead(q, Gemma)  | launch_rmsnorm_gemma_bf16 over N·Hq rows of d |
/// | RmsnormPerHead(k, Gemma)  | launch_rmsnorm_gemma_bf16 over N·Hkv rows of d|
/// | Rope(partial)             | launch_rope_partial_bf16 (rotary_dim chans)   |
/// | KvAppend                  | launch_write_kv_to_pages / _explicit          |
/// | Attention                 | dispatch_attention_flashinfer_{decode,prefill}|
/// | SigmoidGateMul            | launch_sigmoid_gate_inplace_bf16              |
/// | Matmul(o_proj)+res        | ops::gemm_act_x_w beta=1                      |
///
/// With the fused binding (`fused_qkv`, `PIE_QWEN35_FUSED_FULL_ATTN_QGKV`)
/// the three projections become Matmul(qgkv) + [`SplitQkv`] whose "q" leg
/// is the 2×-wide `[query | gate]` bank (`use_fused_qgkv`:
/// `launch_split_qkv_bf16(packed, qg, k, v, N, 2*Hq, Hk)`) — the
/// [`SplitQGate`] de-interleave still follows, exactly as in the
/// hand-written body. `KvAppend`/`Attention` mark the layer's KV cache
/// ([`crate::trace::StateStore::KvCache`] via
/// [`crate::trace::OpKind::state_ref`]), the same marking llama_like
/// carries.
///
/// [`SplitQkv`]: crate::trace::OpKind::SplitQkv
/// [`SplitQGate`]: crate::trace::OpKind::SplitQGate
pub fn qwen3_5_full_attn_block(facts: &Qwen35FullAttnFacts) -> ForwardPlan {
    dsl::trace_named("qwen3_5_full_attn_block", |t| {
        // The fragment's parameter: the residual stream entering the block.
        let y = dsl::input(t, facts.hidden);
        full_attn_body(t, 0, facts, &y, None);
    })
}

/// The full-attention block's weight namespace at layer `l`: both
/// projection bindings (the fused `qgkv` bank and the unfused three), the
/// per-head qk-norm handles, and the layer's KV cache. The q handles are
/// 2× wide (per-head `[query | gate]`).
struct FullAttnLayerW {
    attn_norm: NormW,
    qgkv: MatW,
    q_proj: MatW,
    k_proj: MatW,
    v_proj: MatW,
    q_norm: NormW,
    k_norm: NormW,
    o_proj: MatW,
    kv: Kv,
}

impl FullAttnLayerW {
    fn new(t: &Trace, l: u32, f: &Qwen35FullAttnFacts) -> Self {
        let q2_w = 2 * f.q_width();
        let kv_w = f.kv_width();
        let w = |name: &str| format!("layer.{l}.{name}");
        let mat = |name: &str, width: u32| MatW {
            name: w(name),
            width,
            layer: Some(l),
        };
        // Per-head convention throughout this family — the weight knows,
        // so `rmsnorm(q, &w.q_norm)` needs no variant arguments.
        let qk_norm = |name: &str| NormW {
            name: w(name),
            variant: f.norm_variant,
            per_head: Some(f.head_dim),
            layer: Some(l),
        };
        FullAttnLayerW {
            attn_norm: NormW {
                name: w("attn_norm"),
                variant: f.norm_variant,
                per_head: None,
                layer: Some(l),
            },
            qgkv: mat("qgkv", q2_w + 2 * kv_w),
            q_proj: mat("q_proj", q2_w),
            k_proj: mat("k_proj", kv_w),
            v_proj: mat("v_proj", kv_w),
            q_norm: qk_norm("q_norm"),
            k_norm: qk_norm("k_norm"),
            o_proj: mat("o_proj", f.hidden),
            kv: Kv::at(t, l),
        }
    }
}

/// The full-attention block's op emission at layer `l` — the unit
/// [`qwen3_5_full_attn_block`] traces standalone (at layer 0) and
/// [`qwen3_5_hybrid`] composes on every `Full` layer.
///
/// `KvAppend`/`Attention` carry the MODEL layer `l`. The driver's compact
/// KV slot (`Qwen3_5LayerWeights::kv_layer`, assigned `kv_slot++` over the
/// full-attention layers in `qwen3_5.cpp::bind_qwen3_5`) is storage
/// knowledge derived from the layer-kind schedule — the count of full
/// layers before `l` — not a fact of what the pass computes, so the trace
/// states the layer and the emitter derives the slot, exactly as the GDN
/// ops state `l` while the driver keys its stash on the compact
/// `linear_idx`.
///
/// ONLY the kernel CHOICES lower under `Some(lower)`: the KV write (the
/// per-fire `HasWriteDesc` guard, both arms stated — llama_like's 4a
/// form) and the attention kernel (FlashInfer decode vs the planned
/// prefill dispatch). Everything else — the norms (incl. the Gemma
/// per-head pair), the projections and splits, the partial rope, the
/// sigmoid output gate, the o_proj fold — is a 1:1-kernel semantic op
/// and stays semantic in every form.
fn full_attn_body(
    t: &Trace,
    l: u32,
    facts: &Qwen35FullAttnFacts,
    y: &Val,
    lower: Qwen35Lower<'_>,
) -> Val {
    let w = FullAttnLayerW::new(t, l, facts);
    let mut y = y.clone();

    let x = rmsnorm(&y, &w.attn_norm);

    // Projections: q is 2× wide (per-head [query | gate]). The fused
    // binding packs [2q | k | v] into one bank (`qkv_proj.fused`, joined
    // behind PIE_QWEN35_FUSED_FULL_ATTN_QGKV); the split widths mirror the
    // driver's launch_split_qkv_bf16(N, 2*Hq, Hk).
    let (qg, k, v) = if facts.fused_qkv {
        split_qkv(&matmul(&x, &w.qgkv), 2 * facts.q_width(), facts.kv_width())
    } else {
        (
            matmul(&x, &w.q_proj),
            matmul(&x, &w.k_proj),
            matmul(&x, &w.v_proj),
        )
    };
    let (q, gate) = split_q_gate(&qg, facts.q_heads, facts.head_dim);

    // Per-head q/k norms (the weight knows: Gemma fold, per-head), then
    // partial rope: only the first rotary_dim channels of each head rotate.
    let q = rmsnorm(&q, &w.q_norm);
    let k = rmsnorm(&k, &w.k_norm);
    let (q, k) = rope_partial(&q, &k, RopeKind::Standard, facts.rotary_dim);
    // The OnAttnProj site (A4): post-rope, pre-KV-write — the
    // hand-written full-attn invoke's position, observing the roped q
    // (bf16). Observation-only, like the GDN sites.
    if lower.is_some() {
        dsl::hook_site(HookStage::OnAttnProj, &q, l);
    }

    // KV write. Lowered: the mechanism is a per-fire runtime input
    // (explicit descriptors when the fire steers a graph replay,
    // page-derived otherwise) — the same HasWriteDesc guard llama_like's
    // lowered arm carries, both arms stated.
    if lower.is_some() {
        dsl::guard_on(
            t,
            GuardPred::HasWriteDesc,
            || cuda::write_kv_explicit(&k, &v, &w.kv),
            || cuda::write_kv_to_pages(&k, &v, &w.kv),
        );
    } else {
        w.kv.append(&k, &v);
    }

    // Attention (semantic: opaque, the backend owns plan choice; a class
    // arm states its kernel — qwen3_5's cache is bf16-gated, so the
    // prefill arm is the dequant-less planned dispatch), then the
    // multiply-only output gate and the o_proj accumulate (`+=` of a
    // fresh matmul IS the beta=1 fold).
    // StateOnly runs the full backbone, prefill-shaped — the Prefill arm;
    // CommitAdvance skips full-attention layers entirely and never enters
    // this body ([`commit_advance_body`]).
    let attn = match lower {
        None => attention(&q, &w.kv, facts.q_width()),
        Some((_, FireClass::Decode)) => cuda::attention_flashinfer_decode(&q, &w.kv, facts.q_width()),
        Some((_, FireClass::Prefill | FireClass::StateOnly | FireClass::FrozenVerify)) => {
            cuda::attention_flashinfer_prefill_planned(&q, &w.kv, facts.q_width())
        }
        Some((_, FireClass::CommitAdvance)) => {
            unreachable!("CommitAdvance traces its own pass, never the layer body")
        }
    };
    let gated = sigmoid_gate_mul(&attn, &gate);
    // The OnAttn site: after the output gate, before the o_proj — the
    // hand-written invoke's position (observing q).
    if lower.is_some() {
        dsl::hook_site(HookStage::OnAttn, &q, l);
    }
    y += matmul(&gated, &w.o_proj);
    y
}

/// The dense SwiGLU MLP block's op emission at layer `l`
/// (`qwen3_5_forward.cpp::qwen35_dense_mlp_block`): pre-norm → gate‖up →
/// swiglu → down landed on the residual (the beta=1 GEMM). The driver's
/// fused-vs-unfused gate/up banks are emitter dispatch on the single traced
/// `gate_up` matmul, not a fact — the same call llama_like's olmo2 comment
/// records for its unfused gate/up binding.
fn dense_mlp_body(
    l: u32,
    hidden: u32,
    intermediate: u32,
    variant: NormVariant,
    y: &Val,
) -> Val {
    let w = |name: &str| format!("layer.{l}.{name}");
    let mlp_norm = NormW {
        name: w("mlp_norm"),
        variant,
        per_head: None,
        layer: Some(l),
    };
    let gate_up = MatW {
        name: w("gate_up"),
        width: 2 * intermediate,
        layer: Some(l),
    };
    let down = MatW {
        name: w("down"),
        width: hidden,
        layer: Some(l),
    };
    let mut y = y.clone();
    let m = rmsnorm(&y, &mlp_norm);
    let act = swiglu(&matmul(&m, &gate_up), intermediate);
    y += matmul(&act, &down);
    y
}

/// The full qwen3_5 HYBRID declaration — the first whole-model trace beyond
/// llama_like, composing the three fragment bodies exactly as plan.md Part
/// 1 sketches:
///
/// ```text
/// let mut y = embed[tok];
/// for l in 0..layers {
///     y += match layers[l] {          // static match, resolved at trace time
///         Full   => full_attn(l, rmsnorm(y, attn_norm)),
///         Linear => gdn(l, rmsnorm(y, attn_norm)),
///     };
///     y += mlp(l, rmsnorm(y, mlp_norm));   // dense or MoE, per the facts
/// }
/// lm_head(rmsnorm(y, final_norm))
/// ```
///
/// The `match layers[l]` runs over [`Qwen35HybridFacts::is_full_attn`] —
/// the checkpoint's `layer_types` schedule stated as the regular interval
/// (see the facts doc for the provenance chain) — and, like every fact
/// branch, executes at trace time and vanishes: the traced form is a flat
/// op list whose layer kinds are baked in. Each layer's attention ops are
/// EXACTLY the standalone fragment's ([`qwen3_5_gdn_block`] /
/// [`qwen3_5_full_attn_block`] — one shared body each, pinned by test), so
/// everything those fragments state about lowerings, per-request state
/// marking and binding facts holds here per layer.
///
/// Mirrors `qwen3_5_forward.cpp::qwen3_5_forward_paged`'s walk: embed
/// (`launch_embed_bf16`) → per layer {pre-attn norm + attention body,
/// pre-MLP norm + MLP body} → final norm (`launch_rmsnorm_gemma_bf16`) →
/// lm_head (`gemm_act_x_w`). The compact-logit gather, the state-only and
/// commit-advance fires, MTP and the verify/rs-buffer services are
/// per-fire services around this one pass, not ops of it.
pub fn qwen3_5_hybrid(facts: &Qwen35HybridFacts) -> ForwardPlan {
    qwen3_5_hybrid_text(facts, None)
}

/// The LOWERED qwen3_5 hybrid: the SAME text as [`qwen3_5_hybrid`],
/// traced with the CUDA backend facts and a fire class in hand, so the
/// class arms run and the traced form states its kernels as raw
/// signatures ([`crate::dsl::cuda`]; north-star-dsl.md rung 4c). One
/// trace per [`FireClass`] the deployment fires; family names
/// `qwen3_5_hybrid.cuda.decode` / `.prefill` — the [`llama_like_cuda`]
/// naming, verbatim — plus the two SERVICE classes (rung 4c-iv):
/// `.state_only` (the whole backbone, prefill-shaped, minus the
/// final-norm/lm_head epilogue) and `.commit_advance` (the spec-decode
/// repair: a genuinely different pass — `commit_advance_body`).
pub fn qwen3_5_hybrid_cuda(
    facts: &Qwen35HybridFacts,
    cuda: &Qwen35CudaFacts,
    class: FireClass,
) -> ForwardPlan {
    qwen3_5_hybrid_text(facts, Some((cuda, class)))
}

/// THE one qwen3_5 hybrid text (north-star-dsl.md): computation and
/// kernel choice together. With `lower: None` this is the semantic
/// trace, byte-identical to what [`qwen3_5_hybrid`] always produced (the
/// `qwen3_5_hybrid_0_8b` golden is the gate). With a lowering, the class
/// arms inside the two attention bodies run as ordinary trace-time
/// matches; the MLP bodies take no lowering because they hold no kernel
/// choice — every op of theirs is 1:1.
fn qwen3_5_hybrid_text(facts: &Qwen35HybridFacts, lower: Qwen35Lower<'_>) -> ForwardPlan {
    let hidden = facts.hidden();
    assert_eq!(
        facts.gdn.hidden, hidden,
        "hybrid sub-facts disagree on hidden (gdn)"
    );
    if let Qwen35MlpKind::Moe(moe) = &facts.mlp {
        assert_eq!(
            moe.hidden, hidden,
            "hybrid sub-facts disagree on hidden (moe)"
        );
    }

    let family = match &lower {
        None => "qwen3_5_hybrid".to_string(),
        Some((_, class)) => format!(
            "qwen3_5_hybrid.cuda.{}",
            match class {
                FireClass::Decode => "decode",
                FireClass::Prefill => "prefill",
                FireClass::CommitAdvance => "commit_advance",
                FireClass::StateOnly => "state_only",
                FireClass::FrozenVerify => "frozen_verify",
            }
        ),
    };
    dsl::trace_named(&family, |t| {
        // CommitAdvance changes WHICH OPS RUN so radically — only the
        // linear layers' conv+prep+recurrence, no embed/attention/MLP,
        // nothing after — that it is not a variant of the walk below but
        // its own pass, stated as its own body (north-star-dsl.md 4b:
        // "a genuinely different pass, so a genuinely different trace").
        if let Some((c, FireClass::CommitAdvance)) = lower {
            commit_advance_body(t, facts, c);
            return;
        }

        let mut y = dsl::embed_with(t, "embed", hidden);

        for l in 0..facts.layers {
            let y_attn = if facts.is_full_attn(l) {
                full_attn_body(t, l, &facts.attn, &y, lower)
            } else {
                gdn_attn_body(t, l, &facts.gdn, &y, lower)
            };
            y = match &facts.mlp {
                Qwen35MlpKind::Dense { intermediate } => {
                    dense_mlp_body(l, hidden, *intermediate, facts.norm_variant, &y_attn)
                }
                Qwen35MlpKind::Moe(moe) => moe_mlp_body(l, moe, &y_attn),
            };
        }

        // The epilogue class match: StateOnly is the backbone alone —
        // the trace simply ends after the last layer, exactly the pair
        // the hand-written pass's `if (num_logit_rows < 0) return` skips
        // (final-norm rmsnorm + lm_head, nothing else).
        if matches!(lower, Some((_, FireClass::StateOnly))) {
            return;
        }
        let final_norm = NormW {
            name: "final_norm".to_string(),
            variant: facts.norm_variant,
            per_head: None,
            layer: None,
        };
        let lm_head = if facts.tied_embeddings { "embed" } else { "lm_head" };
        dsl::lm_head_at(t, &rmsnorm(&y, &final_norm), lm_head, facts.vocab);
    })
}

/// The CommitAdvance pass (north-star-dsl.md 4b, rung 4c-iv): the
/// spec-decode repair that re-advances each LINEAR layer's conv window
/// and recurrent state over the confirmed prefix. Full-attention layers,
/// every MLP, embed and the epilogue do not exist in this pass — per
/// linear layer it is exactly conv → prep → recurrence, fed either from
/// the verify hidden stash ([`Qwen35CudaFacts::verify_stash`]) or from
/// re-run in-projections.
///
/// The pass's root value is a bare [`dsl::input`] placeholder, not an
/// embed: the hand-written no-stash path leans, degenerately, on
/// whatever the workspace's `norm_x` buffer happens to hold from the
/// preceding fire, and the placeholder states that reliance honestly —
/// a value produced by no op of this trace, bound by the driver.
///
/// The pass is prefill-shaped, so the conv states the prefill walk; its
/// `commit_lens` is a runtime argument the driver binds (the FLA family
/// is the only recurrence family threading commit_lens, which is why the
/// recurrence states [`cuda::gdn_prefill_fla`] directly — no N-threshold
/// guard chain). The recurrence output is consumed by NOTHING in this
/// pass — no gated norm, no o_proj — so the FLA launch stays the
/// output-less launch it already is: the pass's product is the advanced
/// per-request state, which is not a traced value.
fn commit_advance_body(t: &Trace, facts: &Qwen35HybridFacts, cuda: &Qwen35CudaFacts) {
    let gdn = &facts.gdn;
    let x = dsl::input(t, facts.hidden());
    for l in (0..facts.layers).filter(|&l| !facts.is_full_attn(l)) {
        let w = GdnLayerW::new(t, l, gdn);
        let (qkv, a, b) = if cuda.verify_stash {
            // The stash replays the in-proj OUTPUTS, so the GEMMs and
            // splits are skipped entirely: qkv/a/b arrive via the one
            // load (z is not stashed — nothing downstream reads it).
            cuda::verify_stash_load(t, &w.rs, gdn.conv_dim(), gdn.value_heads)
        } else {
            // No stash: re-run the in-projections — the normal body's
            // fused/unfused arms verbatim ([`gdn_in_proj`]) — against
            // the input placeholder. The z leg is traced (it is part of
            // those arms) and consumed by nothing, like the recurrence
            // output below.
            let (qkv, _z, a, b) = gdn_in_proj(&x, &w, gdn);
            (qkv, a, b)
        };
        let qkv = cuda::gdn_conv_prefill_batched(&qkv, &w.conv, &w.rs);
        let (q, k, v, g, beta) = gdn_prep(
            &qkv,
            &a,
            &b,
            &w.prep,
            gdn.key_heads,
            gdn.key_head_dim,
            gdn.value_heads,
            gdn.value_head_dim,
        );
        // The hand-written commit replay passes through both hook
        // invokes (they precede its early return), so the commit trace
        // mirrors them (A4) — argument no-ops on every commit fire
        // today, stated because the contract is the body's.
        dsl::hook_site(HookStage::OnAttnProj, &q, l);
        cuda::gdn_prefill_fla(&q, &k, &v, &g, &beta, &w.rs, cuda.state_bf16);
        dsl::hook_site(HookStage::OnAttn, &q, l);
    }
    // Nothing after the loop: no final norm, no lm_head — the pass ends
    // with the last linear layer's recurrence.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{DType, Dim, NormVariant, OpKind, StateRef, StateStore};

    /// The traced form of one qwen3 layer, mapped op-by-op to the kernel
    /// sequence `llama_like_forward_paged` launches on the unfused path.
    /// (The fused decode QKV kernel covers Matmul+SplitQkv+RmsnormPerHead
    /// x2+Rope+KvAppend — an emitter peephole over exactly this adjacency;
    /// see stage1-notes.md for why the trace must stay unfused.)
    ///
    /// | trace op            | hand-written kernel(s)                          |
    /// |---------------------|-------------------------------------------------|
    /// | Rmsnorm(attn_norm)  | launch_rmsnorm_bf16                              |
    /// | Matmul(qkv)         | ops::gemm_act_x_w (qkv_proj_fused)               |
    /// | SplitQkv            | launch_split_qkv_bf16                            |
    /// | RmsnormPerHead x2 + Rope | launch_qk_rmsnorm_rope_bf16 (fused pair)    |
    /// | KvAppend            | launch_write_kv_to_pages                         |
    /// | Attention           | dispatch_attention_flashinfer_{decode,prefill}   |
    /// | Matmul(o_proj)+res  | ops::gemm_act_x_w beta=1                         |
    /// | Rmsnorm(mlp_norm)   | launch_rmsnorm_bf16                              |
    /// | Matmul(gate_up)     | ops::gemm_act_x_w                                |
    /// | Swiglu              | (silu-and-mul kernel)                            |
    /// | Matmul(down)+res    | ops::gemm_act_x_w beta=1                         |
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

    /// The MoE block fragment's op sequence, mapped launch for launch to
    /// `run_moe_mlp`'s decode fast path (the table on
    /// [`qwen3_5_moe_mlp_block`]).
    #[test]
    fn moe_block_op_sequence() {
        let plan = qwen3_5_moe_mlp_block(&Qwen35MoeMlpFacts::qwen3_5_35b_a3b());
        let kinds: Vec<&'static str> = plan
            .layer_ops(0)
            .map(|op| match &op.kind {
                OpKind::Rmsnorm { .. } => "rmsnorm",
                OpKind::Matmul { selector: Some(_), .. } => "matmul_per_token",
                OpKind::Matmul { .. } => "matmul",
                OpKind::TopK { .. } => "topk",
                OpKind::Swiglu { .. } => "swiglu",
                OpKind::WeightedSum { .. } => "weighted_sum",
                OpKind::SigmoidGateAdd => "sigmoid_gate_add",
                OpKind::ResidualAdd => "residual_add",
                _ => "other",
            })
            .collect();
        assert_eq!(
            kinds,
            [
                "rmsnorm",           // mlp_norm (gemma fold)
                "matmul",            // router logits [Tokens, E]
                "topk",              // launch_topk_softmax: idx + renormed weights
                "matmul_per_token",  // grouped gate_up over the selected experts
                "swiglu",            // chunked swiglu over [Tokens, k, Im]
                "matmul_per_token",  // grouped down
                "weighted_sum",      // [Tokens, k, H] -> [Tokens, H]
                "matmul",            // shared_expert.gate_up
                "swiglu",
                "matmul",            // shared_expert.down
                "matmul",            // shared_expert_gate: [Tokens, 1] logit
                "sigmoid_gate_add",  // routed + sigmoid(gate) * shared
                "residual_add",      // y += moe_out
            ]
        );
    }

    /// Without a shared expert (qwen3_moe: `shared_expert_intermediate` 0)
    /// the five shared ops fold away at trace time, llama_like-branch
    /// style, and the routed combine lands on the residual directly.
    #[test]
    fn moe_block_without_shared_expert_folds_the_shared_ops() {
        let facts = Qwen35MoeMlpFacts {
            shared_expert_intermediate: 0,
            norm_variant: NormVariant::Plain,
            ..Qwen35MoeMlpFacts::qwen3_5_35b_a3b()
        };
        let plan = qwen3_5_moe_mlp_block(&facts);
        assert_eq!(plan.ops.len(), 8);
        assert!(
            !plan
                .ops
                .iter()
                .any(|op| matches!(op.kind, OpKind::SigmoidGateAdd))
        );
        assert!(!plan.ops.iter().any(|op| {
            matches!(&op.kind, OpKind::Matmul { weight, .. } if weight.contains("shared"))
        }));
        // The residual add consumes the weighted sum's output directly.
        let add = plan.ops.last().unwrap();
        assert!(matches!(add.kind, OpKind::ResidualAdd));
        let combine = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::WeightedSum { .. }))
            .unwrap();
        assert_eq!(add.inputs[0], combine.outputs[0]);
    }

    /// The dyn dataflow: TopK's index output is the fragment's only
    /// dyn-marked value, both expert matmuls name it as their selector AND
    /// their last input, and their weight names are `{e}` templates.
    #[test]
    fn moe_block_selector_dataflow() {
        let plan = qwen3_5_moe_mlp_block(&Qwen35MoeMlpFacts::qwen3_5_35b_a3b());
        let topk = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::TopK { .. }))
            .unwrap();
        let idx = topk.outputs[0];
        let dyn_values: Vec<_> = plan
            .values
            .iter()
            .enumerate()
            .filter(|(_, v)| v.dyn_axis.is_some())
            .map(|(i, _)| i as u32)
            .collect();
        assert_eq!(dyn_values, vec![idx]);
        assert_eq!(plan.values[idx as usize].dtype, DType::I32);

        let grouped: Vec<_> = plan
            .ops
            .iter()
            .filter(|op| matches!(&op.kind, OpKind::Matmul { selector: Some(_), .. }))
            .collect();
        assert_eq!(grouped.len(), 2);
        for op in &grouped {
            let OpKind::Matmul { weight, selector, .. } = &op.kind else {
                unreachable!()
            };
            assert_eq!(*selector, Some(idx));
            assert_eq!(*op.inputs.last().unwrap(), idx);
            assert!(weight.contains("{e}"), "not a template: {weight}");
        }
        assert!(matches!(
            &grouped[0].kind,
            OpKind::Matmul { weight, .. } if weight == "layer.0.expert.{e}.gate_up"
        ));
        assert!(matches!(
            &grouped[1].kind,
            OpKind::Matmul { weight, .. } if weight == "layer.0.expert.{e}.down"
        ));
    }

    /// Route-expanded shapes: the grouped matmuls and the swiglu between
    /// them carry the `[Tokens, k, ...]` factored form of the driver's
    /// `[N*K, ...]` scratch, and the weighted sum collapses it back.
    #[test]
    fn moe_block_route_expanded_shapes() {
        let facts = Qwen35MoeMlpFacts::qwen3_5_35b_a3b();
        let plan = qwen3_5_moe_mlp_block(&facts);
        let k = Dim::Const(facts.top_k);
        let by_kind = |pred: fn(&OpKind) -> bool| {
            plan.ops
                .iter()
                .filter(move |op| pred(&op.kind))
                .collect::<Vec<_>>()
        };

        let grouped = by_kind(|k| matches!(k, OpKind::Matmul { selector: Some(_), .. }));
        assert_eq!(
            plan.values[grouped[0].outputs[0] as usize].shape.0,
            vec![Dim::Tokens, k, Dim::Const(2 * facts.moe_intermediate)]
        );
        assert_eq!(
            plan.values[grouped[1].outputs[0] as usize].shape.0,
            vec![Dim::Tokens, k, Dim::Const(facts.hidden)]
        );

        // The routed swiglu keeps the route dims; the shared one is the
        // ordinary dense shape.
        let swiglus = by_kind(|k| matches!(k, OpKind::Swiglu { .. }));
        assert_eq!(
            plan.values[swiglus[0].outputs[0] as usize].shape.0,
            vec![Dim::Tokens, k, Dim::Const(facts.moe_intermediate)]
        );
        assert_eq!(
            plan.values[swiglus[1].outputs[0] as usize].shape.0,
            vec![
                Dim::Tokens,
                Dim::Const(facts.shared_expert_intermediate)
            ]
        );

        let combine = by_kind(|k| matches!(k, OpKind::WeightedSum { .. }));
        assert!(
            matches!(combine[0].kind, OpKind::WeightedSum { k } if k == facts.top_k)
        );
        assert_eq!(
            plan.values[combine[0].outputs[0] as usize].shape.0,
            vec![Dim::Tokens, Dim::Const(facts.hidden)]
        );

        // The shared gate logit is the [Tokens, 1] scalar-gate GEMM.
        let gate = plan
            .ops
            .iter()
            .find(|op| {
                matches!(&op.kind, OpKind::Matmul { weight, .. }
                    if weight.ends_with("shared_expert_gate"))
            })
            .unwrap();
        assert_eq!(
            plan.values[gate.outputs[0] as usize].shape.0,
            vec![Dim::Tokens, Dim::Const(1)]
        );
    }

    /// The fragment parameter is honest dataflow: value 0 is produced by no
    /// op, read by the block's first norm, and landed on by the final
    /// residual add.
    #[test]
    fn moe_block_residual_stream_is_a_fragment_parameter() {
        let plan = qwen3_5_moe_mlp_block(&Qwen35MoeMlpFacts::qwen3_5_35b_a3b());
        assert!(!plan.ops.iter().any(|op| op.outputs.contains(&0)));
        assert!(matches!(&plan.ops[0].kind, OpKind::Rmsnorm { weight, .. }
            if weight == "layer.0.mlp_norm"));
        assert_eq!(plan.ops[0].inputs, vec![0]);
        let add = plan.ops.last().unwrap();
        assert!(matches!(add.kind, OpKind::ResidualAdd));
        assert_eq!(*add.inputs.last().unwrap(), 0);
    }

    /// The dyn vocabulary survives serde — selector fields, dyn markers,
    /// rank-3 shapes — and, per the additive rule, none of it appears in a
    /// dyn-free plan's serialization (the goldens pin that byte-for-byte;
    /// this pins the reason).
    #[test]
    fn moe_traced_form_round_trips() {
        let plan = qwen3_5_moe_mlp_block(&Qwen35MoeMlpFacts::qwen3_5_35b_a3b());
        let json = serde_json::to_string(&plan).unwrap();
        let back: ForwardPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(plan, back);

        let dense = serde_json::to_string(&llama_like(&LlamaLikeFacts::qwen3_0_6b())).unwrap();
        assert!(!dense.contains("selector"));
        assert!(!dense.contains("dyn_axis"));
    }

    /// The GDN block fragment's op sequence, mapped launch for launch to
    /// `linear_attn_layer_body`'s decode fast path (the table on
    /// [`qwen3_5_gdn_block`]), on the default (unfused) binding.
    #[test]
    fn gdn_block_op_sequence() {
        let plan = qwen3_5_gdn_block(&Qwen35GdnFacts::qwen3_5_0_8b());
        let kinds: Vec<&'static str> = plan
            .layer_ops(0)
            .map(|op| match &op.kind {
                OpKind::Rmsnorm { .. } => "rmsnorm",
                OpKind::Matmul { beta_one: false, .. } => "matmul",
                OpKind::Matmul { beta_one: true, .. } => "matmul+res",
                OpKind::SplitGdn { .. } => "split_gdn",
                OpKind::CausalConv1d { .. } => "causal_conv1d",
                OpKind::GdnPrep { .. } => "gdn_prep",
                OpKind::GatedDelta { .. } => "gated_delta",
                OpKind::RmsnormGated { .. } => "rmsnorm_gated",
                _ => "other",
            })
            .collect();
        assert_eq!(
            kinds,
            [
                "rmsnorm",       // attn_norm (gemma fold)
                "matmul",        // in_proj_qkv [Tokens, conv_dim]
                "matmul",        // in_proj_z  [Tokens, v_dim]
                "matmul",        // in_proj_a  [Tokens, Vh]
                "matmul",        // in_proj_b  [Tokens, Vh]
                "causal_conv1d", // per-request conv state, fused silu
                "gdn_prep",      // q/k/v/g/beta from qkv+a+b (+a_log, dt_bias)
                "gated_delta",   // per-request recurrent state -> core
                "rmsnorm_gated", // z-gated per-head norm, plain fold
                "matmul+res",    // o_proj, beta=1
            ]
        );
        assert_eq!(plan.ops.len(), 10);
    }

    /// The fused in-proj binding (`PIE_QWEN35_FUSED_GDN_PROJ`) trades the
    /// four projection matmuls for two matmuls + two SplitGdn launches —
    /// same count, resolved at trace time — and the ba split's outputs are
    /// (b, a) in the driver's packing order, so `a` is the split's SECOND
    /// output while gdn_prep consumes `[qkv, a, b]`.
    #[test]
    fn gdn_block_fused_binding_traces_two_splits() {
        let facts = Qwen35GdnFacts {
            fused_in_proj: true,
            ..Qwen35GdnFacts::qwen3_5_0_8b()
        };
        let plan = qwen3_5_gdn_block(&facts);
        assert_eq!(plan.ops.len(), 10);
        let matmuls = plan
            .ops
            .iter()
            .filter(|op| matches!(op.kind, OpKind::Matmul { .. }))
            .count();
        assert_eq!(matmuls, 3); // qkvz, ba, o_proj
        let splits: Vec<_> = plan
            .ops
            .iter()
            .filter(|op| matches!(op.kind, OpKind::SplitGdn { .. }))
            .collect();
        assert_eq!(splits.len(), 2);
        assert!(matches!(
            splits[0].kind,
            OpKind::SplitGdn { width0, width1 }
                if width0 == facts.conv_dim() && width1 == facts.value_width()
        ));
        assert!(matches!(
            splits[1].kind,
            OpKind::SplitGdn { width0, width1 }
                if width0 == facts.value_heads && width1 == facts.value_heads
        ));
        // gdn_prep's a operand is the ba split's SECOND output ([b | a]).
        let prep = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::GdnPrep { .. }))
            .unwrap();
        assert_eq!(prep.inputs[1], splits[1].outputs[1]); // a
        assert_eq!(prep.inputs[2], splits[1].outputs[0]); // b
    }

    /// Dataflow and shapes of the GDN core: conv is shape-preserving over
    /// the packed `[Tokens, conv_dim]`, prep emits the compact per-head
    /// rank-3 f32 forms, the recurrence keeps v's shape, and the gated
    /// norm flattens to the z gate's `[Tokens, v_dim]` bf16.
    #[test]
    fn gdn_block_core_shapes() {
        let facts = Qwen35GdnFacts::qwen3_5_0_8b();
        let plan = qwen3_5_gdn_block(&facts);
        let shape_of = |id: u32| plan.values[id as usize].shape.0.clone();
        let dtype_of = |id: u32| plan.values[id as usize].dtype;

        // 0.8B geometry sanity, against the metal driver's stated launch
        // geometry (decode_consts.cpp): 1024 -> 6144, z 1024 -> 2048.
        assert_eq!(facts.conv_dim(), 6144);
        assert_eq!(facts.value_width(), 2048);

        let conv = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::CausalConv1d { .. }))
            .unwrap();
        assert!(matches!(
            &conv.kind,
            OpKind::CausalConv1d { weight, layer: 0, kernel: 4 } if weight == "layer.0.conv"
        ));
        assert_eq!(
            shape_of(conv.outputs[0]),
            vec![Dim::Tokens, Dim::Const(facts.conv_dim())]
        );

        let prep = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::GdnPrep { .. }))
            .unwrap();
        assert!(matches!(
            &prep.kind,
            OpKind::GdnPrep { a_log, dt_bias }
                if a_log == "layer.0.a_log" && dt_bias == "layer.0.dt_bias"
        ));
        assert_eq!(prep.inputs[0], conv.outputs[0]);
        assert_eq!(prep.outputs.len(), 5);
        let kh = Dim::Const(facts.key_heads);
        let kd = Dim::Const(facts.key_head_dim);
        let vh = Dim::Const(facts.value_heads);
        let vd = Dim::Const(facts.value_head_dim);
        assert_eq!(shape_of(prep.outputs[0]), vec![Dim::Tokens, kh, kd]); // q
        assert_eq!(shape_of(prep.outputs[1]), vec![Dim::Tokens, kh, kd]); // k
        assert_eq!(shape_of(prep.outputs[2]), vec![Dim::Tokens, vh, vd]); // v
        assert_eq!(shape_of(prep.outputs[3]), vec![Dim::Tokens, vh]); // g
        assert_eq!(shape_of(prep.outputs[4]), vec![Dim::Tokens, vh]); // beta
        for &out in &prep.outputs {
            assert_eq!(dtype_of(out), DType::F32);
        }

        let delta = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::GatedDelta { .. }))
            .unwrap();
        assert_eq!(delta.inputs, prep.outputs); // [q, k, v, g, beta]
        assert_eq!(shape_of(delta.outputs[0]), vec![Dim::Tokens, vh, vd]);
        assert_eq!(dtype_of(delta.outputs[0]), DType::F32);

        // The gated norm consumes the rank-3 core and the z gate, and
        // lands the flat bf16 form the o_proj GEMM reads.
        let gated = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::RmsnormGated { .. }))
            .unwrap();
        assert_eq!(gated.inputs[0], delta.outputs[0]);
        let z = plan
            .ops
            .iter()
            .find(|op| {
                matches!(&op.kind, OpKind::Matmul { weight, .. } if weight == "layer.0.in_proj_z")
            })
            .unwrap();
        assert_eq!(gated.inputs[1], z.outputs[0]);
        assert_eq!(
            shape_of(gated.outputs[0]),
            vec![Dim::Tokens, Dim::Const(facts.value_width())]
        );
        assert_eq!(dtype_of(gated.outputs[0]), DType::BF16);

        // o_proj accumulates onto the fragment parameter (value 0).
        let o_proj = plan.ops.last().unwrap();
        assert!(matches!(
            &o_proj.kind,
            OpKind::Matmul { beta_one: true, weight, .. } if weight == "layer.0.o_proj"
        ));
        assert_eq!(o_proj.inputs, vec![gated.outputs[0], 0]);
    }

    /// The per-request state axis (§5.4), marked by vocabulary: exactly the
    /// conv and the recurrence address the RecurrentState store at the
    /// block's layer — the traced-form statement of `touches_rs_buffer` —
    /// while llama_like's KvAppend/Attention mark KvCache and the MoE
    /// fragment marks nothing.
    #[test]
    fn gdn_block_marks_the_per_request_state() {
        let plan = qwen3_5_gdn_block(&Qwen35GdnFacts::qwen3_5_0_8b());
        let marks: Vec<_> = plan
            .ops
            .iter()
            .filter_map(|op| op.kind.state_ref())
            .collect();
        assert_eq!(
            marks,
            vec![
                StateRef { store: StateStore::RecurrentState, layer: 0 },
                StateRef { store: StateStore::RecurrentState, layer: 0 },
            ]
        );

        let kv_marks: Vec<_> = llama_like(&LlamaLikeFacts::qwen3_0_6b())
            .layer_ops(3)
            .filter_map(|op| op.kind.state_ref())
            .collect();
        assert_eq!(
            kv_marks,
            vec![
                StateRef { store: StateStore::KvCache, layer: 3 },
                StateRef { store: StateStore::KvCache, layer: 3 },
            ]
        );

        let moe = qwen3_5_moe_mlp_block(&Qwen35MoeMlpFacts::qwen3_5_35b_a3b());
        assert!(moe.ops.iter().all(|op| op.kind.state_ref().is_none()));
    }

    /// The fragment parameter is honest dataflow, MoE-fragment style: value
    /// 0 is produced by no op, read first by the block norm, and landed on
    /// by the o_proj accumulate.
    #[test]
    fn gdn_block_residual_stream_is_a_fragment_parameter() {
        let plan = qwen3_5_gdn_block(&Qwen35GdnFacts::qwen3_5_0_8b());
        assert!(!plan.ops.iter().any(|op| op.outputs.contains(&0)));
        assert!(matches!(&plan.ops[0].kind, OpKind::Rmsnorm { weight, .. }
            if weight == "layer.0.attn_norm"));
        assert_eq!(plan.ops[0].inputs, vec![0]);
        assert_eq!(*plan.ops.last().unwrap().inputs.last().unwrap(), 0);
    }

    /// The full-attention block fragment's op sequence, mapped launch for
    /// launch to `full_attn_layer_body` (the table on
    /// [`qwen3_5_full_attn_block`]), on the default (unfused) binding.
    #[test]
    fn full_attn_block_op_sequence() {
        let plan = qwen3_5_full_attn_block(&Qwen35FullAttnFacts::qwen3_5_0_8b());
        let kinds: Vec<&'static str> = plan
            .layer_ops(0)
            .map(|op| match &op.kind {
                OpKind::Rmsnorm { .. } => "rmsnorm",
                OpKind::Matmul { beta_one: false, .. } => "matmul",
                OpKind::Matmul { beta_one: true, .. } => "matmul+res",
                OpKind::SplitQkv { .. } => "split_qkv",
                OpKind::SplitQGate { .. } => "split_q_gate",
                OpKind::RmsnormPerHead { .. } => "rmsnorm_per_head",
                OpKind::Rope { .. } => "rope",
                OpKind::KvAppend { .. } => "kv_append",
                OpKind::Attention { .. } => "attention",
                OpKind::SigmoidGateMul => "sigmoid_gate_mul",
                _ => "other",
            })
            .collect();
        assert_eq!(
            kinds,
            [
                "rmsnorm",          // attn_norm (gemma fold)
                "matmul",           // q_proj, 2x wide: [Tokens, 2*Hq]
                "matmul",           // k_proj [Tokens, Hk]
                "matmul",           // v_proj [Tokens, Hk]
                "split_q_gate",     // per-head [query | gate] de-interleave
                "rmsnorm_per_head", // q_norm (gemma fold)
                "rmsnorm_per_head", // k_norm
                "rope",             // partial: first rotary_dim channels
                "kv_append",
                "attention",
                "sigmoid_gate_mul", // attn_out *= sigmoid(gate)
                "matmul+res",       // o_proj, beta=1
            ]
        );
        assert_eq!(plan.ops.len(), 12);
    }

    /// The fused qgkv binding (`PIE_QWEN35_FUSED_FULL_ATTN_QGKV`) trades
    /// the three projections for Matmul(qgkv) + SplitQkv whose "q" leg is
    /// the 2×-wide `[query | gate]` bank — and the SplitQGate de-interleave
    /// still follows, consuming that leg, exactly as the hand-written
    /// `use_fused_qgkv` branch.
    #[test]
    fn full_attn_block_fused_binding_traces_qgkv_split() {
        let facts = Qwen35FullAttnFacts {
            fused_qkv: true,
            ..Qwen35FullAttnFacts::qwen3_5_0_8b()
        };
        let plan = qwen3_5_full_attn_block(&facts);
        assert_eq!(plan.ops.len(), 11);
        let matmuls = plan
            .ops
            .iter()
            .filter(|op| matches!(op.kind, OpKind::Matmul { .. }))
            .count();
        assert_eq!(matmuls, 2); // qgkv, o_proj
        let split = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::SplitQkv { .. }))
            .unwrap();
        assert!(matches!(
            split.kind,
            OpKind::SplitQkv { q_width, kv_width }
                if q_width == 2 * facts.q_width() && kv_width == facts.kv_width()
        ));
        // SplitQGate consumes the split's first (2x-wide q|gate) leg.
        let qg_split = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::SplitQGate { .. }))
            .unwrap();
        assert_eq!(qg_split.inputs, vec![split.outputs[0]]);
        // KvAppend consumes the k and v legs.
        let append = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::KvAppend { .. }))
            .unwrap();
        assert_eq!(append.inputs[1], split.outputs[2]); // v (pre-rope)
    }

    /// Dataflow and params of the gated attention: the interleaved split
    /// carries head geometry and halves the 2×-wide projection, the
    /// per-head norms fold Gemma, rope is partial at the fixture's 64
    /// channels, the output gate multiplies attention's output by the
    /// split's GATE leg, and o_proj lands the gated value on the residual.
    #[test]
    fn full_attn_block_gate_dataflow_and_shapes() {
        let facts = Qwen35FullAttnFacts::qwen3_5_0_8b();
        let plan = qwen3_5_full_attn_block(&facts);
        let shape_of = |id: u32| plan.values[id as usize].shape.0.clone();

        // 0.8B geometry sanity, against the metal driver's stated launch
        // geometry (decode_consts.cpp): q 1024 -> 4096 (2x-wide), k/v
        // 1024 -> 512, o 2048 -> 1024.
        assert_eq!(2 * facts.q_width(), 4096);
        assert_eq!(facts.kv_width(), 512);
        assert_eq!(facts.q_width(), 2048);

        let qg_split = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::SplitQGate { .. }))
            .unwrap();
        assert!(matches!(
            qg_split.kind,
            OpKind::SplitQGate { heads: 8, head_dim: 256 }
        ));
        let q_proj = &plan.ops[1];
        assert!(matches!(
            &q_proj.kind,
            OpKind::Matmul { weight, .. } if weight == "layer.0.q_proj"
        ));
        assert_eq!(qg_split.inputs, vec![q_proj.outputs[0]]);
        assert_eq!(
            shape_of(q_proj.outputs[0]),
            vec![Dim::Tokens, Dim::Const(4096)]
        );
        for &out in &qg_split.outputs {
            assert_eq!(shape_of(out), vec![Dim::Tokens, Dim::Const(2048)]);
        }

        // Per-head norms: Gemma fold, head_dim 256, on the QUERY leg.
        let per_head: Vec<_> = plan
            .ops
            .iter()
            .filter(|op| matches!(op.kind, OpKind::RmsnormPerHead { .. }))
            .collect();
        assert_eq!(per_head.len(), 2);
        assert!(matches!(
            &per_head[0].kind,
            OpKind::RmsnormPerHead { weight, head_dim: 256, variant: NormVariant::Gemma }
                if weight == "layer.0.q_norm"
        ));
        assert_eq!(per_head[0].inputs, vec![qg_split.outputs[0]]);

        // Partial rope: the fixture's 64 channels (0.25 x 256).
        let rope = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::Rope { .. }))
            .unwrap();
        assert!(matches!(
            rope.kind,
            OpKind::Rope { kind: RopeKind::Standard, partial: Some(64) }
        ));

        // The output gate: attention's output times the GATE leg — the
        // gate flows AROUND the norm/rope/attention chain, untouched.
        let attn = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::Attention { .. }))
            .unwrap();
        let gate_mul = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::SigmoidGateMul))
            .unwrap();
        assert_eq!(gate_mul.inputs, vec![attn.outputs[0], qg_split.outputs[1]]);
        assert_eq!(
            shape_of(gate_mul.outputs[0]),
            vec![Dim::Tokens, Dim::Const(2048)]
        );

        // o_proj accumulates the GATED value onto the fragment parameter.
        let o_proj = plan.ops.last().unwrap();
        assert!(matches!(
            &o_proj.kind,
            OpKind::Matmul { beta_one: true, weight, .. } if weight == "layer.0.o_proj"
        ));
        assert_eq!(o_proj.inputs, vec![gate_mul.outputs[0], 0]);

        // KvCache marking: exactly KvAppend + Attention, at the block's
        // layer — the same marks llama_like carries, none of the GDN ones.
        let marks: Vec<_> = plan
            .ops
            .iter()
            .filter_map(|op| op.kind.state_ref())
            .collect();
        assert_eq!(
            marks,
            vec![
                StateRef { store: StateStore::KvCache, layer: 0 },
                StateRef { store: StateStore::KvCache, layer: 0 },
            ]
        );
    }

    /// The fragment parameter is honest dataflow, MoE/GDN-fragment style.
    #[test]
    fn full_attn_block_residual_stream_is_a_fragment_parameter() {
        let plan = qwen3_5_full_attn_block(&Qwen35FullAttnFacts::qwen3_5_0_8b());
        assert!(!plan.ops.iter().any(|op| op.outputs.contains(&0)));
        assert!(matches!(&plan.ops[0].kind, OpKind::Rmsnorm { weight, .. }
            if weight == "layer.0.attn_norm"));
        assert_eq!(plan.ops[0].inputs, vec![0]);
        assert_eq!(*plan.ops.last().unwrap().inputs.last().unwrap(), 0);
    }

    /// Rewrite a fragment op's kind from layer 0 to layer `l`: weight names
    /// re-prefixed, state-layer params re-pointed. What "the hybrid's layer
    /// ops equal the fragment's" means, made precise.
    fn relayer(kind: &OpKind, l: u32) -> OpKind {
        let re = |w: &str| w.replacen("layer.0.", &format!("layer.{l}."), 1);
        let mut kind = kind.clone();
        match &mut kind {
            OpKind::Matmul { weight, .. }
            | OpKind::Rmsnorm { weight, .. }
            | OpKind::RmsnormPerHead { weight, .. }
            | OpKind::CausalConv1d { weight, .. }
            | OpKind::RmsnormGated { weight }
            | OpKind::AddBias { weight }
            | OpKind::Embed { weight }
            | OpKind::LmHead { weight } => *weight = re(weight),
            OpKind::GdnPrep { a_log, dt_bias } => {
                *a_log = re(a_log);
                *dt_bias = re(dt_bias);
            }
            _ => {}
        }
        match &mut kind {
            OpKind::KvAppend { layer }
            | OpKind::Attention { layer, .. }
            | OpKind::CausalConv1d { layer, .. }
            | OpKind::GatedDelta { layer } => *layer = l,
            _ => {}
        }
        kind
    }

    /// Assert the hybrid's layer-`l` ATTENTION ops are the standalone
    /// fragment's, op for op: same kinds (modulo the layer rewrite) and the
    /// same SSA dataflow under the id mapping {fragment 0 → the layer's
    /// incoming residual, fragment i → the layer's i-th fresh value}.
    fn assert_layer_head_matches_fragment(
        hybrid: &crate::trace::ForwardPlan,
        l: u32,
        fragment: &crate::trace::ForwardPlan,
    ) {
        let h_ops: Vec<_> = hybrid.layer_ops(l).collect();
        let f_ops: Vec<_> = fragment.layer_ops(0).collect();
        assert!(h_ops.len() > f_ops.len(), "layer {l} shorter than fragment");
        // Fragment value 0 is the parameter; its fresh values start at 1.
        // The hybrid's layer reads the stream as the first op's input and
        // allocates fresh values from the first op's output on.
        let y_in = h_ops[0].inputs[0];
        let base = h_ops[0].outputs[0];
        let map = |id: u32| if id == 0 { y_in } else { base + (id - 1) };
        for (f, h) in f_ops.iter().zip(&h_ops) {
            assert_eq!(h.kind, relayer(&f.kind, l), "kind at layer {l}");
            let mapped_in: Vec<u32> = f.inputs.iter().map(|&i| map(i)).collect();
            let mapped_out: Vec<u32> = f.outputs.iter().map(|&i| map(i)).collect();
            assert_eq!(h.inputs, mapped_in, "inputs of {:?} at layer {l}", f.kind);
            assert_eq!(h.outputs, mapped_out, "outputs of {:?} at layer {l}", f.kind);
        }
    }

    /// The hybrid's layer-kind schedule is the checkpoint's 3:1 pattern:
    /// full attention exactly on layers 3, 7, 11, 15, 19, 23 (interval 4,
    /// end of each block — the Metal geometry's `is_full_attn`), GDN
    /// everywhere else, and every layer carries the dense MLP.
    #[test]
    fn hybrid_layer_kind_sequence_matches_the_pattern() {
        let facts = Qwen35HybridFacts::qwen3_5_0_8b();
        let plan = qwen3_5_hybrid(&facts);
        for l in 0..facts.layers {
            let ops: Vec<_> = plan.layer_ops(l).collect();
            let full = ops
                .iter()
                .any(|op| matches!(op.kind, OpKind::Attention { .. }));
            let linear = ops
                .iter()
                .any(|op| matches!(op.kind, OpKind::GatedDelta { .. }));
            assert_eq!(full, l % 4 == 3, "layer {l} full-attention");
            assert_eq!(linear, l % 4 != 3, "layer {l} linear-attention");
            assert!(!(full && linear), "layer {l} mixes kinds");
            // The uniform dense MLP: gate_up + down on every layer.
            assert!(ops.iter().any(|op| matches!(&op.kind,
                OpKind::Matmul { weight, .. } if weight.ends_with("gate_up"))));
        }
    }

    /// The hybrid's GDN layers ARE the standalone GDN fragment, op for op
    /// and edge for edge — the shared-body refactor pinned as behaviour.
    #[test]
    fn hybrid_gdn_layers_equal_the_standalone_fragment() {
        let facts = Qwen35HybridFacts::qwen3_5_0_8b();
        let hybrid = qwen3_5_hybrid(&facts);
        let fragment = qwen3_5_gdn_block(&facts.gdn);
        for l in (0..facts.layers).filter(|&l| !facts.is_full_attn(l)) {
            assert_layer_head_matches_fragment(&hybrid, l, &fragment);
        }
    }

    /// The hybrid's full-attention layers ARE the standalone full-attention
    /// fragment, same pinning.
    #[test]
    fn hybrid_full_attn_layers_equal_the_standalone_fragment() {
        let facts = Qwen35HybridFacts::qwen3_5_0_8b();
        let hybrid = qwen3_5_hybrid(&facts);
        let fragment = qwen3_5_full_attn_block(&facts.attn);
        for l in (0..facts.layers).filter(|&l| facts.is_full_attn(l)) {
            assert_layer_head_matches_fragment(&hybrid, l, &fragment);
        }
    }

    /// The op-count formula: 18 GDN layers x (10 attn + 4 mlp) + 6 full
    /// layers x (12 attn + 4 mlp) + embed + final norm + lm_head — and the
    /// epilogue: tied lm_head over the 0.8B vocab.
    #[test]
    fn hybrid_full_plan_shape() {
        let facts = Qwen35HybridFacts::qwen3_5_0_8b();
        let plan = qwen3_5_hybrid(&facts);
        assert_eq!(plan.ops.len(), 18 * 14 + 6 * 16 + 3);
        assert!(matches!(&plan.ops[0].kind, OpKind::Embed { weight } if weight == "embed"));
        assert!(matches!(
            &plan.ops.last().unwrap().kind,
            OpKind::LmHead { weight } if weight == "embed"
        ));
        let logits = plan.ops.last().unwrap().outputs[0];
        assert_eq!(
            plan.values[logits as usize].shape.0,
            vec![Dim::Requests, Dim::Const(facts.vocab)]
        );
        // Both state stores are marked, on disjoint layer sets: the KV
        // cache exactly on the full-attention layers (twice each: append +
        // attention), the recurrent store exactly on the GDN layers
        // (twice each: conv + recurrence).
        for l in 0..facts.layers {
            let stores: Vec<_> = plan
                .layer_ops(l)
                .filter_map(|op| op.kind.state_ref())
                .collect();
            let store = if facts.is_full_attn(l) {
                StateStore::KvCache
            } else {
                StateStore::RecurrentState
            };
            assert_eq!(
                stores,
                vec![StateRef { store, layer: l }, StateRef { store, layer: l }]
            );
        }
    }

    /// A MoE-MLP hybrid composes the MoE fragment body per layer (the
    /// qwen3.5/3.6-MoE shape): every layer carries the router → topk →
    /// grouped GEMMs → combine block in place of the dense four.
    #[test]
    fn hybrid_with_moe_mlp_composes_the_moe_fragment() {
        let moe = Qwen35MoeMlpFacts {
            hidden: 1024,
            ..Qwen35MoeMlpFacts::qwen3_5_35b_a3b()
        };
        let facts = Qwen35HybridFacts {
            layers: 4,
            mlp: Qwen35MlpKind::Moe(moe),
            ..Qwen35HybridFacts::qwen3_5_0_8b()
        };
        let plan = qwen3_5_hybrid(&facts);
        // 3 GDN layers x (10 + 13) + 1 full layer x (12 + 13) + 3.
        assert_eq!(plan.ops.len(), 3 * 23 + 25 + 3);
        for l in 0..facts.layers {
            assert_eq!(
                plan.layer_ops(l)
                    .filter(|op| matches!(op.kind, OpKind::TopK { .. }))
                    .count(),
                1,
                "layer {l} routes"
            );
        }
    }

    /// The lowered GDN prefill recurrence under a GQA share (the 0.8B
    /// fixture has Kh == Vh, so the golden cannot show this): the
    /// repeat_interleave launches materialize INSIDE the cached arm only
    /// — the warp-tiled and FLA arms index the compact layout directly —
    /// and every arm binds the guard's output, which the gated norm
    /// consumes as its core. The decode class under the same share
    /// states the `_gqa` step variant.
    #[test]
    fn lowered_gdn_prefill_gqa_repeats_live_inside_the_cached_arm() {
        let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
        facts.gdn.key_heads = 8; // 16 value heads sharing 8 key heads
        let cuda = Qwen35CudaFacts::qwen3_5_0_8b_synthetic();
        let plan = qwen3_5_hybrid_cuda(&facts, &cuda, FireClass::Prefill);

        let idx = plan
            .ops
            .iter()
            .position(|op| matches!(op.kind, OpKind::Guard { .. }) && op.layer == Some(0))
            .expect("layer 0 (GDN) carries the recurrence guard");
        let OpKind::Guard { arms, else_ops } = &plan.ops[idx].kind else {
            unreachable!()
        };
        assert_eq!(arms.len(), 2);
        assert_eq!(arms[0].pred, GuardPred::TokensLE(64));
        assert_eq!(arms[0].ops, 1); // warp-tiled alone
        assert_eq!(arms[1].pred, GuardPred::TokensLE(4096));
        assert_eq!(arms[1].ops, 3); // 2 repeats + cached
        assert_eq!(*else_ops, 1); // FLA alone

        let kernels: Vec<&str> = plan.ops[idx + 1..idx + 6]
            .iter()
            .map(|op| match &op.kind {
                OpKind::Launch { kernel, .. } => kernel.as_str(),
                other => panic!("guard region holds a non-launch: {other:?}"),
            })
            .collect();
        assert_eq!(
            kernels,
            [
                "launch_chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16",
                "launch_repeat_interleave_heads_fp32",
                "launch_repeat_interleave_heads_fp32",
                "launch_chunk_gated_delta_prefill_batched_cached_state_bf16",
                "launch_chunk_gated_delta_prefill_batched_state_bf16",
            ]
        );
        // Region launches are output-less lowerings of the guard's value,
        // and that value is the core the gated norm consumes.
        for op in &plan.ops[idx + 1..idx + 6] {
            assert!(op.outputs.is_empty(), "region launch grew outputs: {op:?}");
        }
        let core = plan.ops[idx].outputs[0];
        let gated = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::RmsnormGated { .. }) && op.layer == Some(0))
            .unwrap();
        assert_eq!(gated.inputs[0], core);

        let decode = qwen3_5_hybrid_cuda(&facts, &cuda, FireClass::Decode);
        assert!(decode.ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Launch { kernel, .. }
                if kernel == "launch_recurrent_gated_delta_step_batched_gqa_state_bf16"
        )));
    }

    /// The full-attention and hybrid traced forms survive serde — the new
    /// kinds, the partial rope, the per-head Gemma variant — and, per the
    /// additive rule, none of the new vocabulary appears in any pre-hybrid
    /// plan's serialization: the seven existing goldens stay byte-identical.
    #[test]
    fn full_attn_and_hybrid_traced_forms_round_trip() {
        for plan in [
            qwen3_5_full_attn_block(&Qwen35FullAttnFacts::qwen3_5_0_8b()),
            qwen3_5_hybrid(&Qwen35HybridFacts::qwen3_5_0_8b()),
        ] {
            let json = serde_json::to_string(&plan).unwrap();
            let back: crate::trace::ForwardPlan = serde_json::from_str(&json).unwrap();
            assert_eq!(plan, back);
        }

        for old in [
            serde_json::to_string(&llama_like(&LlamaLikeFacts::qwen3_0_6b())).unwrap(),
            serde_json::to_string(&llama_like(&LlamaLikeFacts::olmo2_1b())).unwrap(),
            serde_json::to_string(&qwen3_5_moe_mlp_block(
                &Qwen35MoeMlpFacts::qwen3_5_35b_a3b(),
            ))
            .unwrap(),
            serde_json::to_string(&qwen3_5_gdn_block(&Qwen35GdnFacts::qwen3_5_0_8b())).unwrap(),
        ] {
            for token in ["SplitQGate", "SigmoidGateMul", "partial"] {
                assert!(!old.contains(token), "{token} leaked into a pre-hybrid plan");
            }
            // RmsnormPerHead's variant field is serde-skipped at its Plain
            // default, so pre-variant serializations carry no per-head
            // variant key (Rmsnorm's own always-present variant remains).
            assert!(!old.contains(r#""head_dim":128,"variant""#));
        }
    }

    /// The GDN vocabulary survives serde — new op kinds, rank-3 f32 values,
    /// two-name GdnPrep — and, per the additive rule, none of it (nor the
    /// dyn vocabulary) appears in a pre-GDN plan's serialization: the
    /// existing goldens stay byte-identical.
    #[test]
    fn gdn_traced_form_round_trips() {
        let plan = qwen3_5_gdn_block(&Qwen35GdnFacts::qwen3_5_0_8b());
        let json = serde_json::to_string(&plan).unwrap();
        let back: ForwardPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(plan, back);

        for dense in [
            serde_json::to_string(&llama_like(&LlamaLikeFacts::qwen3_0_6b())).unwrap(),
            serde_json::to_string(&qwen3_5_moe_mlp_block(
                &Qwen35MoeMlpFacts::qwen3_5_35b_a3b(),
            ))
            .unwrap(),
        ] {
            for token in ["SplitGdn", "CausalConv1d", "GdnPrep", "GatedDelta", "RmsnormGated"] {
                assert!(!dense.contains(token), "{token} leaked into a pre-GDN plan");
            }
        }
    }
}
