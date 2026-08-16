//! Backend-neutral DSL operations plus shared statement idioms.

use super::*;

/// Leading-dim window; records a pointer offset, not a launch.
pub fn select(x: &Val, index: u32) -> Val {
    let id = x.t.with(x.layer, |b| b.select(x.id, index));
    Val {
        t: x.t.clone(),
        id,
        layer: x.layer,
    }
}

/// Dense records semantic matmul; quantized records launch plus scale/zero weights.
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

/// Semantic row/per-head RMSNorm; `w.per_head` selects convention.
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

/// Order is contract: input seam before embedding.
pub fn embedded_prologue(t: &Trace, hidden: u32) -> Val {
    seam(t, &seam::IN, &[], None);
    embed_with(t, "embed", hidden)
}

/// Order is contract: final norm, readout, optional softcap, output seam.
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

/// Order is contract: `attn.out` seam must observe before `o_proj` consumes.
pub fn attention_landing(a: &Val, o_proj: &MatW, layer: u32) -> Val {
    seam(a.trace(), &seam::ATTN_OUT, &[a], Some(layer));
    matmul(a, o_proj)
}

/// Returns `(q_b, kv_a, q_a_n)`; `q_a_n` has a second consumer.
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

/// MLA geometry widths passed together to keep positional call sites short.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlaWidths {
    pub heads: u32,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    pub v_head_dim: u32,
}

/// Absorbed attention after family-specific prepare.
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

/// Dense gated-MLP activation; variants map to different kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatedAct {
    SwiGlu,
    SwiGluClamp,
    Situ,
}

/// Trace gate, up, activation, down; caller lands the residual.
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
    let up = matmul(m, up_w);
    let activated = match act {
        GatedAct::SwiGlu => cuda::swiglu_pair(&gate, &up, intermediate),
        GatedAct::SwiGluClamp => cuda::swiglu_clamp_pair(&gate, &up, intermediate),
        GatedAct::Situ => cuda::situ_pair(&gate, &up, intermediate),
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

/// Semantic paged attention over `kv` layer cache.
pub fn attention(q: &Val, kv: &Kv, q_width: u32) -> Val {
    let id = q.t.with(Some(kv.l), |b| b.attention(kv.l, q.id, q_width));
    Val {
        t: q.t.clone(),
        id,
        layer: Some(kv.l),
    }
}

/// Expert-bank matmul; `selector` indexes `{e}` slots.
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

/// Router top-k returns `(indices, weights)`.
pub fn topk(logits: &Val, k: u32) -> (Val, Val) {
    let (idx, w) = logits.t.with(logits.layer, |b| b.topk(logits.id, k));
    let mk = |id| Val {
        t: logits.t.clone(),
        id,
        layer: logits.layer,
    };
    (mk(idx), mk(w))
}

/// Operand order is builder order: weights, then value.
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

/// Operand order: gated value, gate, base.
pub fn sigmoid_gate_add(x: &Val, gate: &Val, base: &Val) -> Val {
    let id =
        x.t.with(x.layer, |b| b.sigmoid_gate_add(x.id, gate.id, base.id));
    Val {
        t: x.t.clone(),
        id,
        layer: x.layer,
    }
}

pub fn sigmoid_gate_mul(x: &Val, gate: &Val) -> Val {
    let id = x.t.with(x.layer, |b| b.sigmoid_gate_mul(x.id, gate.id));
    Val {
        t: x.t.clone(),
        id,
        layer: x.layer,
    }
}

/// GDN split returns widths `(w0, w1)`.
pub fn split_gdn(x: &Val, w0: u32, w1: u32) -> (Val, Val) {
    let (a, b) = x.t.with(x.layer, |b| b.split_gdn(x.id, w0, w1));
    let mk = |id| Val {
        t: x.t.clone(),
        id,
        layer: x.layer,
    };
    (mk(a), mk(b))
}

/// Interleaved per-head split returns `(query, gate)`.
pub fn split_q_gate(x: &Val, heads: u32, head_dim: u32) -> (Val, Val) {
    let (q, gate) = x.t.with(x.layer, |b| b.split_q_gate(x.id, heads, head_dim));
    let mk = |id| Val {
        t: x.t.clone(),
        id,
        layer: x.layer,
    };
    (mk(q), mk(gate))
}

/// Partial rope rotates first `rotary_dim` channels per head.
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

/// Depthwise conv uses `w.layer` as tag and conv-state key.
pub fn causal_conv1d(x: &Val, w: &ConvW) -> Val {
    let id = x.t.with(Some(w.layer), |b| {
        b.causal_conv1d(w.layer, x.id, &w.name, w.bias.as_deref(), w.kernel)
    });
    Val {
        t: x.t.clone(),
        id,
        layer: Some(w.layer),
    }
}

/// Returns `(q, k, v, g, beta)`; geometry params define per-head layouts.
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

/// GDN recurrence over `rs` layer state.
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

/// Gated RMSNorm: reads only `w.name` and `w.layer`; fold is plain.
pub fn rmsnorm_gated(x: &Val, gate: &Val, w: &NormW) -> Val {
    let id =
        x.t.with(w.layer, |b| b.rmsnorm_gated(x.id, gate.id, &w.name));
    Val {
        t: x.t.clone(),
        id,
        layer: w.layer,
    }
}


impl std::ops::AddAssign<Val> for Val {
    /// Fold `y += matmul` to beta-one when possible; otherwise residual-add.
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
