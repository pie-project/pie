//! THE SEMANTIC VOCABULARY, as free functions — the backend-neutral
//! half of the surface, plus the `+=` residual landing that is the one
//! place a statement REWRITES the op it just recorded.

use super::*;

/// `x[index]` — the window of a value along its LEADING dim.
///
/// Launches nothing. The value it produces is the operand's bytes at an
/// offset, which `lower::Buffers` computes; `lower` emits no rectangle.
///
/// It exists because gemma3n's AltUp needs it and no other family does:
/// `altup_predict` produces all `k` streams, the layer body runs on ONE,
/// and in `gemma3n.cpp` that is `predictions + active * N * H` — a
/// pointer offset with no kernel behind it. Stating it as an op rather
/// than hiding it in a `Val` method is deliberate: which window the body
/// reads is a fact about the MODEL, and a reader following the dataflow
/// has to see it.
///
/// The shape drops the leading dim. Selecting from a rank-1 value is
/// refused — there would be nothing left to name.
pub fn select(x: &Val, index: u32) -> Val {
    let id = x.t.with(x.layer, |b| b.select(x.id, index));
    Val {
        t: x.t.clone(),
        id,
        layer: x.layer,
    }
}

/// `y = x @ Wᵀ`, over a weight stored however [`MatW::repr`] says.
///
/// POLYMORPHIC OVER THE REPRESENTATION, and that is the whole of the
/// quantization axis. A dense weight records the semantic
/// [`model_ir::trace::OpKind::Matmul`] — one arithmetic, one kernel per
/// backend, nothing chosen. Any other representation records a stated
/// `Launch` naming the kernel that can read it, plus the scale and
/// zero-point tensors as extra WEIGHTS.
///
/// So the driver never sees a descriptor and never routes: it binds the
/// names the statement gives it and calls the symbol the statement
/// names. `make_weight_view` and `gemm::act_x_w`'s internal dispatch
/// have nothing left to decide.
pub fn matmul(x: &Val, w: &MatW) -> Val {
    let id = match w.gemm_symbol() {
        None => x.t.with(w.layer, |b| b.matmul(x.id, &w.name, w.width)),
        Some(symbol) => {
            let mut weights = vec![w.name.clone()];
            weights.extend(w.scale_names());
            let shape = Shape(vec![Dim::Tokens, Dim::Const(w.width)]);
            let outs = x.t.with(w.layer, |b| {
                b.launch(
                    symbol,
                    weights,
                    None,
                    vec![x.id],
                    vec![(shape, DType::BF16)],
                )
            });
            outs[0]
        }
    };
    Val {
        t: x.t.clone(),
        id,
        layer: w.layer,
    }
}

/// Row RMSNorm, or per-head RMSNorm when the weight handle says so.
///
/// SEMANTIC: this is the backend-independent spelling, and it stays one
/// because a trace with no backend cannot name a CUDA symbol —
/// `check_plan` refuses it, correctly. The stated form is
/// [`cuda::rmsnorm`], which is what a `*.cuda.*` text calls.
pub fn rmsnorm(x: &Val, w: &NormW) -> Val {
    let id = x.t.with(w.layer, |b| match w.per_head {
        None => b.rmsnorm(x.id, &w.name, w.variant),
        Some(head_dim) => b.rmsnorm_per_head(x.id, &w.name, head_dim, w.variant),
    });
    Val {
        t: x.t.clone(),
        id,
        layer: w.layer,
    }
}

/// Broadcast bias add (`OpKind::AddBias`): `x[r, :] += bias`. The Qwen-2
/// family's qkv biases; the kernel is 1:1 so semantic and lowered traces
/// state the same op.
pub fn add_bias(x: &Val, w: &MatW) -> Val {
    let id = x.t.with(w.layer, |b| b.add_bias(x.id, &w.name));
    Val {
        t: x.t.clone(),
        id,
        layer: w.layer,
    }
}

pub fn split_qkv(x: &Val, q_width: u32, kv_width: u32) -> (Val, Val, Val) {
    let (q, k, v) = x.t.with(x.layer, |b| b.split_qkv(x.id, q_width, kv_width));
    let mk = |id| Val {
        t: x.t.clone(),
        id,
        layer: x.layer,
    };
    (mk(q), mk(k), mk(v))
}

pub fn rope(q: &Val, k: &Val, kind: RopeKind) -> (Val, Val) {
    let (qo, ko) = q.t.with(q.layer, |b| b.rope(q.id, k.id, kind));
    let mk = |id| Val {
        t: q.t.clone(),
        id,
        layer: q.layer,
    };
    (mk(qo), mk(ko))
}

/// THE PROLOGUE: the entry seam, then the token embedding.
///
/// Two lines, written identically by every family that has no per-layer
/// embedding table to build first. It is shared for the reason the
/// epilogue's exit seam is inside its block: the ORDER is the contract.
/// The seam is where attached programs bind their inputs, so a family
/// that embedded before seaming would hand them a value they were
/// supposed to be able to influence.
///
/// Nothing enforces that order but this function, and nothing needs to
/// once every caller is this function.
pub fn embedded_prologue(t: &Trace, hidden: u32) -> Val {
    seam(t, &seam::IN, &[], None);
    embed_with(t, "embed", hidden)
}

/// THE EPILOGUE, and the three facts that vary in it.
///
/// Five families wrote this: final norm, readout, an optional logit
/// softcap, and the exit seam. What differed was the norm's VARIANT
/// (gemma's `(1 + w)` fold or plain), whether the readout is tied, and
/// whether the deployment caps its logits — three facts, and everything
/// else identical down to the `"final_norm"` weight name.
///
/// The softcap is a LAUNCH, not a parameter: a cap large enough to do
/// nothing is still a kernel run per fire to compute the identity, so a
/// deployment without one states no statement rather than a wide one.
///
/// The exit seam is inside, because it is the boundary sampling attaches
/// to and a family that forgot it would trace a plan nothing can read
/// from. Four of the five had it as the last line; making it the block's
/// last line means a fifth cannot omit it.
pub fn logits_epilogue(
    t: &Trace,
    y: &Val,
    norm_variant: model_ir::trace::NormVariant,
    tied_embeddings: bool,
    vocab: u32,
    logit_softcap: bool,
) {
    let normed = cuda::rmsnorm(
        y,
        &NormW {
            name: "final_norm".to_string(),
            variant: norm_variant,
            per_head: None,
            layer: None,
        },
    );
    let logits = lm_head_tied(t, &normed, tied_embeddings, vocab);
    let logits = if logit_softcap {
        cuda::logit_softcap(&logits, vocab)
    } else {
        logits
    };
    seam(t, &seam::OUT, &[&logits], None);
}

/// THE ATTENTION LANDING: observe the core's output, then project it.
///
/// Nine sites across eight families, and the ORDER is the contract. The
/// `attn.out` seam is the OnAttn site — where attached programs read the
/// attention's result and where a score consumer binds — so it must see
/// the value BEFORE `o_proj` consumes it. A family that projected first
/// would seam a value nothing else can reach.
///
/// That is not locally visible: both orders trace, both lower, and the
/// difference only shows when something actually attaches. Which is why
/// it is worth being a function rather than a convention repeated nine
/// times.
///
/// Returns the projection. Whether the caller writes `y += …` or folds
/// the residual into the GEMM's beta is the family's business — gpt-oss
/// does the second — so this has no opinion about the landing.
pub fn attention_landing(a: &Val, o_proj: &MatW, layer: u32) -> Val {
    seam(a.trace(), &seam::ATTN_OUT, &[a], Some(layer));
    matmul(a, o_proj)
}

/// MLA's TWO LATENTS: the query's, normed and expanded, and the KV's.
///
/// Three families wrote the unfused form identically — glm5, kimi-k2's
/// `else` arm and kimi-k3 — four statements that never varied. `hidden`
/// appears nowhere in what they produce, which is what makes this MLA
/// rather than a wide attention.
///
/// `fused` is the BINDING fact, and it is a different kernel rather than
/// a buffer detail: a deployment whose load joined the two latents into
/// one bank norms the query half IN PLACE with a pitch, and
/// `rmsnorm_strided` reads a row stride the plain norm has no parameter
/// for. So the fork is an `Option` on the fused bank — present means the
/// join happened — rather than a bool beside a weight that may or may not
/// exist.
///
/// Returns `(q_b, kv_a, q_a_n)`. The first two are what every MLA prepare
/// takes next, fused or split; the third is the NORMED query latent,
/// which glm5's DSA indexer scores its pages from — a second consumer of
/// an intermediate, and the reason this returns it rather than keeping
/// it private.
pub fn mla_latents(
    x: &Val,
    fused: Option<&MatW>,
    q_a_proj: &MatW,
    q_a_norm: &NormW,
    q_b_proj: &MatW,
    kv_a_proj: &MatW,
    q_lora_rank: u32,
) -> (Val, Val, Val) {
    let (q_a_n, kv_a) = match fused {
        Some(bank) => {
            let qkv_a = matmul(x, bank);
            // The statement carries the NARROW extent it produces; the
            // pitch is the buffer question the kernel owns.
            let q_a_n = cuda::rmsnorm_strided(&qkv_a, &q_a_norm.name, q_lora_rank);
            (q_a_n, qkv_a)
        }
        None => {
            let q_a = matmul(x, q_a_proj);
            (cuda::rmsnorm(&q_a, q_a_norm), matmul(x, kv_a_proj))
        }
    };
    (matmul(&q_a_n, q_b_proj), kv_a, q_a_n)
}

/// The five widths an MLA layer is described by.
///
/// Three families carry a field-identical struct for these — glm5's
/// `Glm5MlaFacts`, kimi-k2's `KimiMlaFacts`, kimi-k3's `KimiK3MlaFacts`
/// — and pass them one at a time to statements that always want the same
/// four. Grouping them is what lets the block below take one argument
/// instead of four, and what makes a fifth family's copy obviously a
/// copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlaWidths {
    pub heads: u32,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    pub v_head_dim: u32,
}

/// MLA's ABSORBED attention: the three statements every latent-cache
/// family runs, in the order all three wrote them.
///
/// `q_nope` is absorbed into the latent space, attention runs THERE
/// against the compressed cache, and the result is absorbed back out to
/// the value space. Both absorptions name the whole `kv_b_proj` bank and
/// slice it themselves, which is why the bank is one weight name rather
/// than two.
///
/// What is NOT here is the PREPARE, and that is the point of where the
/// line falls. glm5 and kimi-k2 take the fused `mla_prepare`, which does
/// the rope as part of what it fuses; kimi-k3 takes the split pair,
/// because its MLA carries no rope. That is a real fact about a family,
/// so it stays at the call site — while the three statements after it,
/// which no family varies, stop being written three times.
#[must_use]
pub fn mla_absorbed_attention(
    q_nope: &Val,
    q_pe: &Val,
    kv_b_proj: &str,
    layer: u32,
    w: MlaWidths,
) -> Val {
    let q_latent = cuda::mla_absorb_q_to_latent(
        q_nope,
        kv_b_proj,
        w.heads,
        w.kv_lora_rank,
        w.v_head_dim,
        w.qk_nope_head_dim,
    );
    let attn_latent = cuda::attention_mla(&q_latent, q_pe, layer, w.heads, w.kv_lora_rank);
    cuda::mla_absorb_latent_to_v(
        &attn_latent,
        kv_b_proj,
        w.heads,
        w.v_head_dim,
        w.qk_nope_head_dim,
        w.kv_lora_rank,
    )
}

/// The DENSE GATED MLP, and which activation is a FACT.
///
/// Four families wrote this block identically — glm5, kimi-k2, kimi-k3
/// and deepseek-v4 — and the only thing that differed was one statement:
/// `swiglu`, `swiglu`, `situ`, `swiglu_clamp`. Four copies of a sequence
/// with a single varying line is exactly what `cuda.md` §5.C2 means by
/// "turn the variant forks into data": the fork is not an architecture,
/// it is a value.
///
/// The `_up` binding is load-bearing and easy to lose. The gate and up
/// projections are SEPARATE weights, and the activation kernel reads them
/// as one contiguous pair — so the up matmul must be traced (it writes the
/// second half) even though nothing names its value. Every one of the four
/// copies had it as `let _up = …`, and a fifth would have to remember.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatedAct {
    /// `kernels::mlp::swiglu_bf16` — the plain gated linear unit.
    SwiGlu,
    /// deepseek-v4's clamped variant. A DIFFERENT KERNEL, not a
    /// parameter: the clamp changes the arithmetic.
    SwiGluClamp,
    /// kimi-k3's SiTU.
    Situ,
}

/// Trace one dense gated MLP: norm-input in, contribution out.
///
/// The caller adds the result to its residual, because whether that is
/// `y += …` or a stated `residual_add` is the family's business and this
/// block does not have an opinion about it.
#[must_use]
pub fn dense_gated_mlp(
    m: &Val,
    gate_w: &MatW,
    up_w: &MatW,
    down_w: &MatW,
    intermediate: u32,
    act: GatedAct,
) -> Val {
    let gate = matmul(m, gate_w);
    // Traced for its WRITE, not its value — see the note above.
    let _up = matmul(m, up_w);
    let activated = match act {
        GatedAct::SwiGlu => cuda::swiglu(&gate, intermediate, false),
        GatedAct::SwiGluClamp => cuda::swiglu_clamp(&gate, intermediate, false),
        GatedAct::Situ => cuda::situ(&gate, intermediate, false),
    };
    matmul(&activated, down_w)
}

pub fn swiglu(x: &Val, inter: u32) -> Val {
    let id = x.t.with(x.layer, |b| b.swiglu(x.id, inter));
    Val {
        t: x.t.clone(),
        id,
        layer: x.layer,
    }
}

/// Paged attention over the layer's cache — the SEMANTIC form, opaque.
/// A lowered arm states a kernel from [`cuda`] instead.
pub fn attention(q: &Val, kv: &Kv, q_width: u32) -> Val {
    let id = q.t.with(Some(kv.l), |b| b.attention(kv.l, q.id, q_width));
    Val {
        t: q.t.clone(),
        id,
        layer: Some(kv.l),
    }
}

/// The expert-indexed matmul ([`TraceBuilder::matmul_per_token`]): `w` is
/// an `{e}`-templated bank handle and `selector` a [`topk`] index value.
/// Layer from the weight handle, like [`matmul`].
pub fn matmul_per_token(x: &Val, w: &MatW, selector: &Val) -> Val {
    let id = x.t.with(w.layer, |b| {
        b.matmul_per_token(x.id, &w.name, selector.id, w.width)
    });
    Val {
        t: x.t.clone(),
        id,
        layer: w.layer,
    }
}

/// Router top-k: `(indices, weights)`, softmaxed and renormalized
/// ([`TraceBuilder::topk`]). Weightless — layer from the logits.
pub fn topk(logits: &Val, k: u32) -> (Val, Val) {
    let (idx, w) = logits.t.with(logits.layer, |b| b.topk(logits.id, k));
    let mk = |id| Val {
        t: logits.t.clone(),
        id,
        layer: logits.layer,
    };
    (mk(idx), mk(w))
}

/// The top-k combine: collapse `x` under per-token `weights`
/// ([`TraceBuilder::weighted_sum`] — operand order weights-then-value is
/// the builder's). Layer from `weights`, the first input.
pub fn weighted_sum(weights: &Val, x: &Val) -> Val {
    let id = weights
        .t
        .with(weights.layer, |b| b.weighted_sum(weights.id, x.id));
    Val {
        t: weights.t.clone(),
        id,
        layer: weights.layer,
    }
}

/// The shared-expert landing: `base + sigmoid(gate) * x`. Layer from
/// `x`, the first input.
pub fn sigmoid_gate_add(x: &Val, gate: &Val, base: &Val) -> Val {
    let id =
        x.t.with(x.layer, |b| b.sigmoid_gate_add(x.id, gate.id, base.id));
    Val {
        t: x.t.clone(),
        id,
        layer: x.layer,
    }
}

/// The multiply-only output gate: `x * sigmoid(gate)`. Layer from `x`.
pub fn sigmoid_gate_mul(x: &Val, gate: &Val) -> Val {
    let id = x.t.with(x.layer, |b| b.sigmoid_gate_mul(x.id, gate.id));
    Val {
        t: x.t.clone(),
        id,
        layer: x.layer,
    }
}

/// The two-way GDN row split of a packed projection. Layer from `x`.
pub fn split_gdn(x: &Val, w0: u32, w1: u32) -> (Val, Val) {
    let (a, b) = x.t.with(x.layer, |b| b.split_gdn(x.id, w0, w1));
    let mk = |id| Val {
        t: x.t.clone(),
        id,
        layer: x.layer,
    };
    (mk(a), mk(b))
}

/// The interleaved per-head `[query | gate]` split of a 2×-wide gated q
/// projection. Layer from `x`.
pub fn split_q_gate(x: &Val, heads: u32, head_dim: u32) -> (Val, Val) {
    let (q, gate) = x.t.with(x.layer, |b| b.split_q_gate(x.id, heads, head_dim));
    let mk = |id| Val {
        t: x.t.clone(),
        id,
        layer: x.layer,
    };
    (mk(q), mk(gate))
}

/// The partial-rotary rope: only the first `rotary_dim` channels of each
/// head rotate. Layer from `q`, [`rope`]-style.
pub fn rope_partial(q: &Val, k: &Val, kind: RopeKind, rotary_dim: u32) -> (Val, Val) {
    let (qo, ko) =
        q.t.with(q.layer, |b| b.rope_partial(q.id, k.id, kind, rotary_dim));
    let mk = |id| Val {
        t: q.t.clone(),
        id,
        layer: q.layer,
    };
    (mk(qo), mk(ko))
}

/// Depthwise causal conv1d (+ fused SiLU) against the handle's layer and
/// that layer's per-request conv state. Layer from the weight handle.
pub fn causal_conv1d(x: &Val, w: &ConvW) -> Val {
    let id = x.t.with(Some(w.layer), |b| {
        b.causal_conv1d(w.layer, x.id, &w.name, w.kernel)
    });
    Val {
        t: x.t.clone(),
        id,
        layer: Some(w.layer),
    }
}

/// The post-conv GDN prep: `(q, k, v, g, beta)`, all f32, per-head
/// layouts from the four geometry params ([`TraceBuilder::gdn_prep`]).
/// Layer from the weight handle.
pub fn gdn_prep(
    qkv: &Val,
    a: &Val,
    b: &Val,
    w: &GdnPrepW,
    key_heads: u32,
    key_dim: u32,
    value_heads: u32,
    value_dim: u32,
) -> (Val, Val, Val, Val, Val) {
    let out = qkv.t.with(Some(w.layer), |bld| {
        bld.gdn_prep(
            qkv.id,
            a.id,
            b.id,
            &w.a_log,
            &w.dt_bias,
            key_heads,
            key_dim,
            value_heads,
            value_dim,
        )
    });
    let mk = |id| Val {
        t: qkv.t.clone(),
        id,
        layer: Some(w.layer),
    };
    (mk(out.0), mk(out.1), mk(out.2), mk(out.3), mk(out.4))
}

/// The gated-delta recurrence against the layer's per-request recurrent
/// state. Layer from the state handle, [`attention`]-style.
pub fn gated_delta(rs: &Rs, q: &Val, k: &Val, v: &Val, g: &Val, beta: &Val) -> Val {
    let id = rs.t.with(Some(rs.l), |b| {
        b.gated_delta(rs.l, q.id, k.id, v.id, g.id, beta.id)
    });
    Val {
        t: rs.t.clone(),
        id,
        layer: Some(rs.l),
    }
}

/// The gated RMSNorm landing: per-head norm of the rank-3 f32 core,
/// silu-gated by `gate`, flattened to `gate`'s bf16 shape. The op's fold
/// is Plain by construction ([`TraceBuilder::rmsnorm_gated`] takes only
/// the weight name), so the handle's `variant`/`per_head` are unread —
/// only its name and layer speak.
pub fn rmsnorm_gated(x: &Val, gate: &Val, w: &NormW) -> Val {
    let id =
        x.t.with(w.layer, |b| b.rmsnorm_gated(x.id, gate.id, &w.name));
    Val {
        t: x.t.clone(),
        id,
        layer: w.layer,
    }
}

// ── `+=`: the residual landing ─────────────────────────────────────────

impl std::ops::AddAssign<Val> for Val {
    /// `y += rhs`. If `rhs` is the op just recorded and it is a plain
    /// matmul nobody else consumed, rewrite it to the `beta_one`
    /// accumulate (the cuBLAS fold — id-neutral, so dataflow never sees
    /// the difference). Otherwise record the explicit
    /// [`OpKind::ResidualAdd`](model_ir::trace::OpKind::ResidualAdd) launch, the post-norm landing.
    fn add_assign(&mut self, rhs: Val) {
        let folded_or_added = rhs.t.with(rhs.layer, |b| {
            if b.try_fold_residual(rhs.id, self.id) {
                rhs.id
            } else {
                b.residual_add(rhs.id, self.id)
            }
        });
        self.id = folded_or_added;
        self.layer = rhs.layer;
    }
}
