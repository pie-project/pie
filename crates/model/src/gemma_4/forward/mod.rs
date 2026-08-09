//! `gemma-4`.

pub mod facts;

use self::facts::{
    Gemma4CudaFacts, Gemma4Facts,
};
use model_compiler::dsl::{
    WeightRepr,
    self, matmul, MatW, NormW,
};
use model_compiler::trace::{
    FireClass, ForwardPlan, NormVariant, RopeKind,
};

// ── gemma-4 ──────────────────────────────────────────────────────────

/// One gemma-4 layer's weight namespace, named after the driver's own
/// fields (`Gemma4LayerWeights`) so the executor's binder is a straight
/// map rather than a translation.
struct Gemma4LayerW {
    attn_norm: NormW,
    post_attn_norm: NormW,
    pre_ffw_norm: NormW,
    post_ffw_norm: NormW,
    qkv: MatW,
    q_proj: MatW,
    k_proj: MatW,
    v_proj: MatW,
    o_proj: MatW,
    q_norm: NormW,
    k_norm: NormW,
    gate_up: MatW,
    gate_proj: MatW,
    up_proj: MatW,
    down: MatW,
    ple_gate: MatW,
    ple_proj: MatW,
    ple_norm: NormW,
}

impl Gemma4LayerW {
    fn new(l: u32, f: &Gemma4Facts) -> Self {
        let w = |name: &str| format!("layer.{l}.{name}");
        let d = f.head_dim_of(l);
        let mat = |name: &str, width: u32| MatW {
            name: w(name),
            width,
            layer: Some(l),
            repr: WeightRepr::Bf16,
        };
        // PLAIN, despite the family name: `gemma4.cpp` fires
        // `kernels::norm::rmsnorm_bf16` at all fourteen of its norm sites and
        // `kernels::norm::rmsnorm_gemma_bf16` at none. The `(1 + w)` fold is
        // done to the tensors at LOAD for this family, so a declaration
        // that stated Gemma would be stating a second fold.
        let norm = |name: &str| NormW {
            name: w(name),
            variant: NormVariant::Plain,
            per_head: None,
            layer: Some(l),
        };
        let head_norm = |name: &str| NormW {
            name: w(name),
            variant: NormVariant::Plain,
            per_head: Some(d),
            layer: Some(l),
        };
        Gemma4LayerW {
            attn_norm: norm("attn_norm"),
            post_attn_norm: norm("post_attn_norm"),
            pre_ffw_norm: norm("pre_ffw_norm"),
            post_ffw_norm: norm("post_ffw_norm"),
            qkv: mat("qkv", (f.q_heads + 2 * f.kv_heads) * d),
            q_proj: mat("q_proj", f.q_heads * d),
            k_proj: mat("k_proj", f.kv_heads * d),
            v_proj: mat("v_proj", f.kv_heads * d),
            o_proj: mat("o_proj", f.hidden),
            q_norm: head_norm("q_norm"),
            k_norm: head_norm("k_norm"),
            // The double-wide variant widens exactly the KV-shared
            // layers, so the width is per-layer and erases here.
            gate_up: mat("gate_up", 2 * f.intermediate_of(l)),
            gate_proj: mat("gate_proj", f.intermediate_of(l)),
            up_proj: mat("up_proj", f.intermediate_of(l)),
            down: mat("down", f.hidden),
            ple_gate: mat("ple_gate", f.ple_dim),
            ple_proj: mat("ple_proj", f.hidden),
            ple_norm: norm("ple_norm"),
        }
    }
}

/// The gemma-4 model's CUDA reading — `gemma4.cpp`'s decode path as a
/// list of stated kernels.
///
/// # The three things a reader should look for
///
/// **The input norm is missing from every layer but the first.** That is
/// not an omission: layer `l`'s PLE epilogue fires
/// `kernels::norm::rmsnorm_residual_add_scale_rmsnorm_bf16`, whose FOURTH
/// statement is layer `l+1`'s `attn_norm`. The fusion crosses the layer
/// boundary, so the declaration does too — `gemma4.cpp:1999` produces
/// it and `:1529` is the guard that skips re-computing it.
///
/// **A KV-shared layer's statements are ABSENT, not skipped.** The
/// trailing [`Gemma4Facts::kv_shared_layers`] layers project no k/v,
/// norm neither, rope no k and write no cache; their attention names
/// the SOURCE layer's cache handle. Nothing here tests a flag per fire,
/// because there is nothing per fire about it — the binding decided at
/// load, and a fact is a trace-time `match`.
///
/// **The two layer kinds differ by WIDTH, not by function.** Sliding
/// layers rope fully at `head_dim`; full layers rope partially at
/// `global_head_dim`. That is why the full layers cannot take the fused
/// packed post (its predicate reads `!partial`) and fall to the
/// separate norm/rope statements instead.
///
/// Prefill and the service classes are not stated yet: this is the
/// decode reading, and the class parameter is here so the next rung adds
/// them where llama_like's does.
pub fn gemma4_cuda(
    facts: &Gemma4Facts,
    cuda: &Gemma4CudaFacts,
    class: FireClass,
) -> ForwardPlan {
    let family = format!(
        "gemma4.cuda.{}",
        match class {
            FireClass::Decode => "decode",
            FireClass::Prefill => "prefill",
            other => panic!("gemma4 states no {other:?} class yet"),
        }
    );
    let hidden = facts.hidden;
    dsl::trace_named(&family, |t| {
        // The entry boundary. A `Put` attachment lands device embeds or
        // channel reads here, before anything reads the stream.
        dsl::seam(t, &dsl::seam::IN, &[], None);

        // ── Prologue ────────────────────────────────────────────────
        // The token embedding, scaled by sqrt(hidden).
        let mut y = dsl::cuda::scalar_mul(
            &dsl::embed_with(t, "embed", hidden),
            "sqrt_hidden",
            Some((hidden as f32).sqrt()),
        );

        // PLE: a SECOND embedding table, projected to the whole stack's
        // per-layer width, normed, scaled and relaid so each layer reads
        // a contiguous slice. Once per fire, not per layer — which is
        // the entire reason for the relay.
        let ple_total = facts.layers * facts.ple_dim;
        let table = dsl::cuda::scalar_mul(
            &dsl::embed_with(t, "embed_per_layer", ple_total),
            "sqrt_ple_dim",
            Some((facts.ple_dim as f32).sqrt()),
        );
        // The projection consumes the MAIN embedding, not the table:
        // `per_layer_proj = inputs_embeds @ ple_model_proj.T`. The table
        // is the other addend of the residual below. Reading the call
        // site is what settled it — the body had the projection eating
        // its own table, which is a plausible pipeline and not this one.
        let ple = matmul(
            &y,
            &MatW {
                name: "ple_model_proj".into(),
                width: ple_total,
                layer: None,
                repr: WeightRepr::Bf16,
            },
        );
        let scaled =
            dsl::cuda::scalar_mul(&ple, "rsqrt_hidden", Some(1.0 / (hidden as f32).sqrt()));
        let normed_ple = dsl::cuda::rmsnorm(
            &scaled,
            &NormW {
                name: "ple_model_norm".into(),
                variant: NormVariant::Plain,
                per_head: Some(facts.ple_dim),
                layer: None,
            },
        );
        // The projection lands back on the SCALED TABLE, not on nothing:
        // `(proj + table) / sqrt(2)`. The residual add was missing from
        // this prologue until an executor arm went looking for the value
        // its scale consumes and found two producers where the trace had
        // one.
        let ple = dsl::cuda::residual_add(&normed_ple, &table, ple_total);
        let ple = dsl::cuda::scalar_mul(&ple, "rsqrt_2", Some(1.0 / 2f32.sqrt()));
        let ple_table = dsl::cuda::transpose_nld_to_lnd(&ple, facts.layers, facts.ple_dim);

        // ── Layers ──────────────────────────────────────────────────
        // Layer 0 norms the stream itself; every other layer received
        // its input norm from the layer before (see the doc above).
        let mut normed = dsl::cuda::rmsnorm(&y, &Gemma4LayerW::new(0, facts).attn_norm);

        for l in 0..facts.layers {
            // THIS LAYER's sliding window, `-1` for none — a
            // load-time fact the dispatch statements carry, where four
            // executors used to re-derive it per launch.
            let window_left =
                model_compiler::facts::window_left_at(&cuda.window_left, l);
            let w = Gemma4LayerW::new(l, facts);
            let full = facts.is_full_attn(l);
            let d = facts.head_dim_of(l);
            let shared = facts.is_kv_shared(l);
            // A shared layer attends through the pages of the last
            // earlier layer of its own kind. The handle IS the sharing.
            let kv = dsl::Kv::at(t, facts.kv_source(l).unwrap_or(l));

            // The fused post writes k/v to the pages itself, so it is
            // unavailable to a layer that writes none — and to the full
            // layers, whose partial rope it does not implement.
            let fused_post = cuda.fused_qkv
                && cuda.kv_native_bf16
                && !full
                && !shared
                && class == FireClass::Decode;

            let attn_in = if fused_post {
                let packed = matmul(&normed, &w.qkv);
                dsl::cuda::qkv_packed_post(&packed, &w.q_norm, &w.k_norm, &kv, facts.q_heads * d)
            } else if shared {
                // A shared layer takes only the Q leg: no k/v
                // projection, no k/v norm, no rope on k, no write. Which
                // KERNEL rotates q still follows the layer kind, because
                // the driver reaches both by passing `num_kv_heads = 0`
                // to the launcher the un-shared layer of that kind would
                // have used — NOT by falling back to a generic rope.
                let q = matmul(&normed, &w.q_proj);
                if full {
                    let q = dsl::cuda::rmsnorm(&q, &w.q_norm);
                    dsl::cuda::rope_partial_q_only(&q, facts.global_rotary_dim)
                } else {
                    dsl::cuda::qk_rmsnorm_rope_rounded_q_only(&q, &w.q_norm)
                }
            } else {
                let (q, k, v) = if cuda.fused_qkv {
                    let packed = matmul(&normed, &w.qkv);
                    dsl::split_qkv(&packed, facts.q_heads * d, facts.kv_heads * d)
                } else {
                    (
                        matmul(&normed, &w.q_proj),
                        matmul(&normed, &w.k_proj),
                        matmul(&normed, &w.v_proj),
                    )
                };
                // The adapter seam, and its POSITION is the point: the
                // correction lands on the RAW projections. `v`'s norm is
                // the next statement, so anywhere below this is already
                // different arithmetic -- which is what
                // `seam::check_plan` refuses.
                dsl::seam(q.trace(), &dsl::seam::ATTN_QV, &[&q, &v], Some(l));
                let v = dsl::cuda::rmsnorm_no_scale(&v);
                let (q, k) = if full {
                    // Partial rope has no fused pair, so the norms are
                    // their own statements — `can_fuse_qk_norm_rope`
                    // reads `!partial`.
                    let q = dsl::cuda::rmsnorm(&q, &w.q_norm);
                    let k = dsl::cuda::rmsnorm(&k, &w.k_norm);
                    dsl::rope_partial(&q, &k, RopeKind::Standard, facts.global_rotary_dim)
                } else {
                    dsl::cuda::qk_rmsnorm_rope_rounded(&q, &k, &w.q_norm, &w.k_norm)
                };
                dsl::cuda::write_kv_to_pages(&k, &v, &kv);
                q
            };

            // The dispatch is the one place the two classes diverge, and
            // the PREFILL side diverges again per layer — on the HEAD
            // DIM, not on the layer kind. flashinfer 0.6.x refuses to
            // instantiate its TC prefill template at head_dim 512
            // ("NUM_MMA_D_QK=32"), so gemma-4's full-attention layers
            // take a naive paged kernel there while decode at 512 is
            // fine. Reading the driver's own test (`d == 512`) is what
            // this states; `is_full_attn` happens to agree on E4B and is
            // not the question asked.
            // Pre-attention observation, and where a page-mask sink
            // lands.
            dsl::seam(attn_in.trace(), &dsl::seam::ATTN_Q, &[&attn_in], Some(l));
            let a = match class {
                FireClass::Decode => dsl::cuda::attention_flashinfer_decode(&attn_in, &kv, window_left),
                FireClass::Prefill if d == 512 => {
                    dsl::cuda::attention_naive_paged(&attn_in, &kv, window_left)
                }
                FireClass::Prefill => {
                    dsl::cuda::attention_flashinfer_prefill_planless(&attn_in, &kv, window_left)
                }
                other => unreachable!("gemma4 refuses {other:?} at trace start"),
            }
            .expect("the class states its attention");
            // Post-attention observation. Whether the scores are actually
            // servable is the dispatch's business -- `attention_naive_paged`
            // above declares `lacks Scores`, and the table is where that is
            // written down.
            let attn_out = dsl::attention_landing(&a, &w.o_proj, l);

            // Post-attention norm, land on the stream, scale, and norm
            // for the MLP — four statements, one launch.
            let (landed, mlp_in) = dsl::cuda::norm_residual_scale_norm(
                &attn_out,
                &y,
                &w.post_attn_norm,
                &w.pre_ffw_norm,
                hidden,
            );
            y = landed;

            // The projection follows the BINDING, not just the
            // activation: a deployment without the packed bank (E2B —
            // the fuse is gated on E4B's exact dims) runs two gemms and
            // the PAIR activation. Stating only the fused shape made the
            // trace name a `gate_up` weight E2B never binds, which the
            // executor caught at LOAD.
            let inter = facts.intermediate_of(l);
            let act = if cuda.gate_up_fused {
                dsl::cuda::geglu_tanh(&matmul(&mlp_in, &w.gate_up), inter, true)
            } else {
                let gate = matmul(&mlp_in, &w.gate_proj);
                let up = matmul(&mlp_in, &w.up_proj);
                dsl::cuda::geglu_tanh_pair(&gate, &up, inter)
            };
            let mlp_out = matmul(&act, &w.down);
            y = dsl::cuda::norm_residual_add(&mlp_out, &y, &w.post_ffw_norm, hidden);

            // ── The PLE epilogue ────────────────────────────────────
            // Gate this layer's slice of the per-layer table into the
            // stream, then land it — and, for every layer but the last,
            // produce the NEXT layer's input norm in the same launch.
            let gate = matmul(&y, &w.ple_gate);
            // THIS LAYER's slice of the relay, as a `select` rather than
            // as an offset the executor computes. The relay is `[L,
            // Tokens, ple_dim]` -- the layer axis leads, which is the
            // whole reason the transpose above exists -- so a select at
            // `l` IS the slice, and `Buffers::assign` places it at
            // `offset(relay) + l * N * ple_dim` without being told.
            //
            // The arm used to add that offset itself, and to tell this
            // site from the MLP's by comparing the result's WIDTH
            // against `ple_dim`. Both go: the two sites now differ only
            // in which values they name.
            let slice = dsl::select(&ple_table, l);
            let gated = dsl::cuda::geglu_tanh_pair(&gate, &slice, facts.ple_dim);
            let ple_out = matmul(&gated, &w.ple_proj);
            if l + 1 < facts.layers {
                let next = Gemma4LayerW::new(l + 1, facts);
                let (landed, next_norm) = dsl::cuda::norm_residual_scale_norm(
                    &ple_out,
                    &y,
                    &w.ple_norm,
                    &next.attn_norm,
                    hidden,
                );
                y = landed;
                normed = next_norm;
            } else {
                // The last layer has no next input norm to fuse, so it
                // lands unfused and the epilogue norms for itself —
                // `gemma4.cpp`'s :2010 arm.
                y = dsl::cuda::norm_residual_add(&ple_out, &y, &w.ple_norm, hidden);
            }
        }

        // ── Epilogue ────────────────────────────────────────────────
        let normed = dsl::cuda::rmsnorm(
            &y,
            &NormW {
                name: "final_norm".into(),
                variant: NormVariant::Plain,
                per_head: None,
                layer: None,
            },
        );
        let logits = dsl::lm_head_tied(t, &normed, facts.tied_embeddings, facts.vocab);
        let logits = if facts.logit_softcap > 0.0 {
            dsl::cuda::logit_softcap(&logits, facts.vocab)
        } else {
            logits
        };
        // The exit boundary: where sampling and host-visible emits attach.
        // It sees the logits AFTER the softcap, because that is what a
        // sampler would draw from.
        dsl::seam(t, &dsl::seam::OUT, &[&logits], None);
    })
}
