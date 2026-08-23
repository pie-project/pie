//! Backend-neutral DSL operations plus shared statement idioms.
//!
//! # Every statement is a launch now
//!
//! The tier-1 fns below used to record semantic `OpKind`s and leave a table
//! in `model-compiler/lower/semantics.rs` to decide what runs. The no-ask
//! contract retires that: each fn states THE symbol — the same
//! backend-neutral name the three shader tables and CUDA's all declare
//! (`shader_backends_agree` holds them equal) — and carries every number the
//! routine's swept signature marks as `Const`, in that signature's order.
//! A backend-less description trace states the same symbols; `check_plan`
//! skips families with no backend, and the engine's site derivation matches
//! the symbol string instead of an op kind.

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

/// Dense states `gemm::act_x_w` (a driver op — cuBLAS answers it);
/// quantized records its launch plus scale/zero weights. The residual fold
/// (`+=`) swaps the symbol to `gemm::act_x_w_acc` — see `AddAssign`.
pub fn matmul(x: &Val, w: &MatW) -> Val {
    let (symbol, mut weights) = match w.gemm_symbol() {
        None => (x.t.canon("matmul"), vec![w.name.clone()]),
        Some(symbol) => (symbol.to_string(), vec![w.name.clone()]),
    };
    weights.extend(w.scale_names());
    let shape = Shape(vec![Dim::Tokens, Dim::Const(w.width)]);
    let id = x.t.with(w.layer, |b| {
        b.launch(
            &symbol,
            weights,
            None,
            vec![x.id],
            vec![(shape, DType::BF16)],
        )[0]
    });
    Val {
        t: x.t.clone(),
        id,
        layer: w.layer,
    }
}

/// Row/per-head RMSNorm. `norm::rmsnorm{,_gemma}_bf16`'s params run is
/// `[per_head_dim, eps]` — the swept signature's order; `0` is the plain
/// (whole-row) reading, and the epsilon is the handle's, which is the
/// forward's, which is the checkpoint's.
pub fn rmsnorm(x: &Val, w: &NormW) -> Val {
    let symbol = x.t.canon(if w.variant.is_plain() {
        "rmsnorm"
    } else {
        "rmsnorm.gemma"
    });
    let params = vec![w.per_head.unwrap_or(0), w.eps.to_bits()];
    let id = x.t.with(w.layer, |b| {
        let shape = b.value_shape(x.id);
        let dtype = b.value_dtype(x.id);
        b.launch_with_params(
            &symbol,
            vec![w.name.clone()],
            None,
            params,
            vec![x.id],
            vec![(shape, dtype)],
        )[0]
    });
    Val {
        t: x.t.clone(),
        id,
        layer: w.layer,
    }
}

/// `norm::add_bias`: `x[r, :] += bias`, in place.
pub fn add_bias(x: &Val, w: &MatW) -> Val {
    let symbol = x.t.canon("add_bias");
    let id = x.t.with(w.layer, |b| {
        let shape = b.value_shape(x.id);
        let dtype = b.value_dtype(x.id);
        b.launch(
            &symbol,
            vec![w.name.clone()],
            None,
            vec![x.id],
            vec![(shape, dtype)],
        )[0]
    });
    Val {
        t: x.t.clone(),
        id,
        layer: w.layer,
    }
}

/// `attn::split_qkv_bf16`: packed `[N, q + 2kv]` into three.
pub fn split_qkv(x: &Val, q_width: u32, kv_width: u32) -> (Val, Val, Val) {
    let symbol = x.t.canon("split_qkv");
    let outs = x.t.with(x.layer, |b| {
        b.launch(
            &symbol,
            vec![],
            None,
            vec![x.id],
            vec![
                (Shape(vec![Dim::Tokens, Dim::Const(q_width)]), DType::BF16),
                (Shape(vec![Dim::Tokens, Dim::Const(kv_width)]), DType::BF16),
                (Shape(vec![Dim::Tokens, Dim::Const(kv_width)]), DType::BF16),
            ],
        )
    });
    let mk = |id| Val {
        t: x.t.clone(),
        id,
        layer: x.layer,
    };
    (mk(outs[0]), mk(outs[1]), mk(outs[2]))
}

/// The numbers `rope::rope_bf16`'s swept signature marks `Const`, in its
/// order: `[num_q_heads, num_kv_heads, head_dim, theta, interleaved]`. The
/// positions stream is minted by name — `dsl::runtime` is the veneer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeShape {
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub theta: f32,
    pub interleaved: bool,
}

/// `rope::rope_bf16`, in place over q and k.
pub fn rope(q: &Val, k: &Val, kind: RopeKind, n: RopeShape) -> (Val, Val) {
    assert!(
        matches!(kind, RopeKind::Standard),
        "yarn rope is a tier-2 statement; state its own symbol"
    );
    let params = vec![
        n.num_q_heads,
        n.num_kv_heads,
        n.head_dim,
        n.theta.to_bits(),
        u32::from(n.interleaved),
    ];
    let symbol = q.t.canon("rope");
    let (qo, ko) = q.t.with(q.layer, |b| {
        let positions = b.runtime_tensor("positions", None, Shape(vec![Dim::Tokens]), DType::I32);
        let q_shape = b.value_shape(q.id);
        let k_shape = b.value_shape(k.id);
        let outs = b.launch_with_params(
            &symbol,
            vec![],
            None,
            params,
            vec![q.id, k.id, positions],
            vec![(q_shape, DType::BF16), (k_shape, DType::BF16)],
        );
        (outs[0], outs[1])
    });
    let mk = |id| Val {
        t: q.t.clone(),
        id,
        layer: q.layer,
    };
    (mk(qo), mk(ko))
}

/// `rope.partial`: params `[rotary_dim, head_dim, theta]`.
pub fn rope_partial(
    q: &Val,
    k: &Val,
    kind: RopeKind,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> (Val, Val) {
    assert!(
        matches!(kind, RopeKind::Standard),
        "yarn rope is a tier-2 statement; state its own symbol"
    );
    let params = vec![rotary_dim, head_dim, theta.to_bits()];
    let symbol = q.t.canon("rope.partial");
    let (qo, ko) = q.t.with(q.layer, |b| {
        let positions = b.runtime_tensor("positions", None, Shape(vec![Dim::Tokens]), DType::I32);
        let q_shape = b.value_shape(q.id);
        let k_shape = b.value_shape(k.id);
        let outs = b.launch_with_params(
            &symbol,
            vec![],
            None,
            params,
            vec![q.id, k.id, positions],
            vec![(q_shape, DType::BF16), (k_shape, DType::BF16)],
        );
        (outs[0], outs[1])
    });
    let mk = |id| Val {
        t: q.t.clone(),
        id,
        layer: q.layer,
    };
    (mk(qo), mk(ko))
}

/// Order is contract: input seam before embedding.
pub fn embedded_prologue(t: &Trace, hidden: u32, vocab: u32) -> Val {
    seam(t, &seam::IN, &[], None);
    embed_with(t, "embed", hidden, vocab)
}

/// Order is contract: final norm, readout, optional softcap, output seam.
pub fn logits_epilogue(
    t: &Trace,
    y: &Val,
    norm_variant: model_ir::trace::NormVariant,
    tied_embeddings: bool,
    vocab: u32,
    logit_softcap: Option<f32>,
    norm_eps: f32,
) {
    let normed = cuda::rmsnorm(
        y,
        &NormW {
            name: "final_norm".to_string(),
            variant: norm_variant,
            per_head: None,
            layer: None,
            eps: norm_eps,
        },
    );
    let logits = lm_head_tied(t, &normed, tied_embeddings, vocab);
    let logits = if let Some(cap) = logit_softcap {
        cuda::generated::logit_softcap(
            &logits,
            (Shape(vec![Dim::Requests, Dim::Const(vocab)]), DType::BF16),
            cap,
            None,
            None,
        )
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
            let q_a_n = cuda::generated::rmsnorm_strided_bf16(
                &qkv_a,
                &q_a_norm.name,
                (
                    Shape(vec![Dim::Tokens, Dim::Const(q_lora_rank)]),
                    DType::BF16,
                ),
                q_a_norm.eps,
                qkv_a.layer(),
                None,
            );
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

/// Dense gated-MLP activation; variants map to different kernels, and each
/// carries the numbers its kernel's swept signature marks `Const`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GatedAct {
    SwiGlu,
    SwiGluClamp { limit: f32 },
    Situ { beta: f32, linear_beta: f32 },
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
    // The pair kernels' results are ruled `like(gate)` on their rows, so
    // the `intermediate` restatement retired with the hand wrappers.
    let _ = intermediate;
    let activated = match act {
        GatedAct::SwiGlu => cuda::generated::swiglu(&gate, &up, gate.layer(), None),
        GatedAct::SwiGluClamp { limit } => {
            cuda::generated::swiglu_clamp(&gate, &up, limit, gate.layer(), None)
        }
        GatedAct::Situ { beta, linear_beta } => {
            cuda::generated::situ(&gate, &up, beta, linear_beta, gate.layer(), None)
        }
    };
    matmul(&activated, down_w)
}

/// Description-trace statement: no backend declares a fused packed swiglu
/// as fixed behaviour, so a lowered text states its own symbol and this
/// `canon::` form appears only in backend-less description traces.
pub fn swiglu(x: &Val, inter: u32) -> Val {
    let symbol = x.t.canon("swiglu");
    let id = x.t.with(x.layer, |b| {
        b.launch(
            &symbol,
            vec![],
            None,
            vec![x.id],
            vec![(Shape(vec![Dim::Tokens, Dim::Const(inter)]), DType::BF16)],
        )[0]
    });
    Val {
        t: x.t.clone(),
        id,
        layer: x.layer,
    }
}

/// Description-trace paged attention over `kv`'s layer cache. Lowered texts
/// state their attention kernel as their own launch; this `canon::` form is
/// the backend-less description's.
pub fn attention(q: &Val, kv: &Kv, q_width: u32) -> Val {
    let symbol = q.t.canon("attention");
    let id = q.t.with(Some(kv.l), |b| {
        b.launch(
            &symbol,
            vec![],
            Some(StateRef {
                store: StateStore::KvCache,
                layer: kv.l,
            }),
            vec![q.id],
            vec![(Shape(vec![Dim::Tokens, Dim::Const(q_width)]), DType::BF16)],
        )[0]
    });
    Val {
        t: q.t.clone(),
        id,
        layer: Some(kv.l),
    }
}

/// Expert-bank matmul: `moe::moe_grouped_gemm`, the selector as the
/// LAST input — the convention the retired `Matmul::selector` recorded.
pub fn matmul_per_token(x: &Val, w: &MatW, selector: &Val) -> Val {
    let symbol = x.t.canon("matmul_select");
    let id = x.t.with(w.layer, |b| {
        b.launch(
            &symbol,
            vec![w.name.clone()],
            None,
            vec![x.id, selector.id],
            vec![(Shape(vec![Dim::Tokens, Dim::Const(w.width)]), DType::BF16)],
        )[0]
    });
    Val {
        t: x.t.clone(),
        id,
        layer: w.layer,
    }
}

/// Router top-k: `moe::topk_softmax`, `(indices, weights)` in that
/// order, `k` on the params run.
pub fn topk(logits: &Val, k: u32) -> (Val, Val) {
    let symbol = logits.t.canon("topk.softmax");
    let outs = logits.t.with(logits.layer, |b| {
        b.launch_with_params(
            &symbol,
            vec![],
            None,
            vec![k],
            vec![logits.id],
            vec![
                (Shape(vec![Dim::Tokens, Dim::Const(k)]), DType::I32),
                (Shape(vec![Dim::Tokens, Dim::Const(k)]), DType::F32),
            ],
        )
    });
    let mk = |id| Val {
        t: logits.t.clone(),
        id,
        layer: logits.layer,
    };
    (mk(outs[0]), mk(outs[1]))
}

/// `moe::token_batched_weighted_sum`; operand order: weights, value.
pub fn weighted_sum(weights: &Val, x: &Val) -> Val {
    let symbol = weights.t.canon("weighted_sum");
    let id = weights.t.with(weights.layer, |b| {
        let shape = match b.value_shape(x.id).0.as_slice() {
            [Dim::Tokens, _, d] => Shape(vec![Dim::Tokens, *d]),
            other => Shape(other.to_vec()),
        };
        b.launch(
            &symbol,
            vec![],
            None,
            vec![weights.id, x.id],
            vec![(shape, DType::BF16)],
        )[0]
    });
    Val {
        t: weights.t.clone(),
        id,
        layer: weights.layer,
    }
}

/// `mlp::sigmoid_dot_scalar_gate_add`; operands: value, gate, base.
pub fn sigmoid_gate_add(x: &Val, gate: &Val, base: &Val) -> Val {
    let symbol = x.t.canon("sigmoid_gate_add");
    let id = x.t.with(x.layer, |b| {
        let shape = b.value_shape(base.id);
        b.launch(
            &symbol,
            vec![],
            None,
            vec![x.id, gate.id, base.id],
            vec![(shape, DType::BF16)],
        )[0]
    });
    Val {
        t: x.t.clone(),
        id,
        layer: x.layer,
    }
}

/// `mlp::sigmoid_gate_inplace_bf16`: `x *= sigmoid(gate)`.
pub fn sigmoid_gate_mul(x: &Val, gate: &Val) -> Val {
    let symbol = x.t.canon("sigmoid_gate_mul");
    let id = x.t.with(x.layer, |b| {
        let shape = b.value_shape(x.id);
        b.launch(
            &symbol,
            vec![],
            None,
            vec![x.id, gate.id],
            vec![(shape, DType::BF16)],
        )[0]
    });
    Val {
        t: x.t.clone(),
        id,
        layer: x.layer,
    }
}

/// GDN two-way split at `w0`. Description form (`canon::`); no plane
/// declares a split routine, and a lowered text states its own — CUDA's
/// `layout::split_bf16_rows` and `layout::split_qwen_gdn_ba`.
pub fn split_gdn(x: &Val, w0: u32, w1: u32) -> (Val, Val) {
    let symbol = x.t.canon("split_gdn");
    let outs = x.t.with(x.layer, |b| {
        b.launch(
            &symbol,
            vec![],
            None,
            vec![x.id],
            vec![
                (Shape(vec![Dim::Tokens, Dim::Const(w0)]), DType::BF16),
                (Shape(vec![Dim::Tokens, Dim::Const(w1)]), DType::BF16),
            ],
        )
    });
    let mk = |id| Val {
        t: x.t.clone(),
        id,
        layer: x.layer,
    };
    (mk(outs[0]), mk(outs[1]))
}

/// Interleaved per-head `[query | gate]` split: `layout::split_q_gate_bf16`,
/// `head_dim` on the params run (the swept signature's one `Const`).
pub fn split_q_gate(x: &Val, heads: u32, head_dim: u32) -> (Val, Val) {
    let w = heads * head_dim;
    let symbol = x.t.canon("split_q_gate");
    let outs = x.t.with(x.layer, |b| {
        b.launch_with_params(
            &symbol,
            vec![],
            None,
            vec![head_dim],
            vec![x.id],
            vec![
                (Shape(vec![Dim::Tokens, Dim::Const(w)]), DType::BF16),
                (Shape(vec![Dim::Tokens, Dim::Const(w)]), DType::BF16),
            ],
        )
    });
    let mk = |id| Val {
        t: x.t.clone(),
        id,
        layer: x.layer,
    };
    (mk(outs[0]), mk(outs[1]))
}

/// Depthwise causal conv with fused SiLU: `ssm::causal_conv1d_update_batched`
/// on the decode reading; the conv-state slab is the layer's runtime object.
/// Params `[conv_dim, kernel]` — the swept `c`/`k` Consts, in order.
pub fn causal_conv1d(x: &Val, w: &ConvW) -> Val {
    let mut weights = vec![w.name.clone()];
    weights.extend(w.bias.clone());
    let symbol = x.t.canon("causal_conv1d");
    let id = x.t.with(Some(w.layer), |b| {
        let shape = b.value_shape(x.id);
        let conv_dim = match shape.0.as_slice() {
            [_, Dim::Const(c)] => *c,
            _ => 0,
        };
        b.launch_with_params(
            &symbol,
            weights,
            Some(StateRef {
                store: StateStore::RecurrentState,
                layer: w.layer,
            }),
            vec![conv_dim, w.kernel],
            vec![x.id],
            vec![(shape, DType::BF16)],
        )[0]
    });
    Val {
        t: x.t.clone(),
        id,
        layer: Some(w.layer),
    }
}

/// Returns `(q, k, v, g, beta)`: `ssm::qwen_gdn_post_conv_prep_bf16`,
/// params `[key_heads, key_dim, value_heads, value_dim]` in the swept
/// signature's order.
pub fn gdn_prep(
    qkv: &Val,
    a: &Val,
    b: &Val,
    w: &GdnPrepW,
    key_heads: u32,
    key_dim: u32,
    value_heads: u32,
    value_dim: u32,
    conv_dim: u32,
) -> (Val, Val, Val, Val, Val) {
    let symbol = qkv.t.canon("gdn_prep");
    let outs = qkv.t.with(Some(w.layer), |bld| {
        bld.launch_with_params(
            &symbol,
            vec![w.a_log.clone(), w.dt_bias.clone()],
            None,
            // THE ROUTINE'S ORDER: `[k_h, v_h, k_d, v_d, conv_dim]`. The
            // catalogue coverage test caught this run stated four scalars
            // in a transposed order — slot 2 carried key_dim where the
            // routine reads v_h — which no fire could refuse: a wrong
            // NUMBER at a bound slot is silent arithmetic.
            vec![key_heads, value_heads, key_dim, value_dim, conv_dim],
            vec![qkv.id, a.id, b.id],
            vec![
                (
                    Shape(vec![Dim::Tokens, Dim::Const(key_heads * key_dim)]),
                    DType::F32,
                ),
                (
                    Shape(vec![Dim::Tokens, Dim::Const(key_heads * key_dim)]),
                    DType::F32,
                ),
                (
                    Shape(vec![Dim::Tokens, Dim::Const(value_heads * value_dim)]),
                    DType::F32,
                ),
                (
                    Shape(vec![Dim::Tokens, Dim::Const(value_heads)]),
                    DType::F32,
                ),
                (
                    Shape(vec![Dim::Tokens, Dim::Const(value_heads)]),
                    DType::F32,
                ),
            ],
        )
    });
    let mk = |id| Val {
        t: qkv.t.clone(),
        id,
        layer: Some(w.layer),
    };
    (
        mk(outs[0]),
        mk(outs[1]),
        mk(outs[2]),
        mk(outs[3]),
        mk(outs[4]),
    )
}

/// GDN recurrence over `rs`'s layer state. Description form (`canon::`);
/// a lowered text states its own recurrence launch.
pub fn gated_delta(rs: &Rs, q: &Val, k: &Val, v: &Val, g: &Val, beta: &Val) -> Val {
    let symbol = rs.t.canon("gated_delta");
    let id = rs.t.with(Some(rs.l), |b| {
        let shape = b.value_shape(v.id);
        b.launch(
            &symbol,
            vec![],
            Some(StateRef {
                store: StateStore::RecurrentState,
                layer: rs.l,
            }),
            vec![q.id, k.id, v.id, g.id, beta.id],
            vec![(shape, DType::F32)],
        )[0]
    });
    Val {
        t: rs.t.clone(),
        id,
        layer: Some(rs.l),
    }
}

/// Gated RMSNorm: `norm::rmsnorm_gated_fp32_in`, params `[eps,
/// per_head_dim]` — the swept signature's order (eps first on this one).
pub fn rmsnorm_gated(x: &Val, gate: &Val, w: &NormW) -> Val {
    let id = x.t.with(w.layer, |b| {
        let shape = b.value_shape(gate.id);
        b.launch_with_params(
            "norm::rmsnorm_gated_fp32_in",
            vec![w.name.clone()],
            None,
            vec![w.eps.to_bits(), w.per_head.unwrap_or(0)],
            vec![x.id, gate.id],
            vec![(shape, DType::BF16)],
        )[0]
    });
    Val {
        t: x.t.clone(),
        id,
        layer: w.layer,
    }
}

impl std::ops::AddAssign<Val> for Val {
    /// Fold `y += matmul` into the GEMM's beta (`gemm::act_x_w` becomes
    /// `gemm::act_x_w_acc` and takes the residual); otherwise state
    /// `norm::residual_add`.
    fn add_assign(&mut self, rhs: Val) {
        let plain = rhs.t.canon("matmul");
        let acc = rhs.t.canon("matmul.acc");
        let add = rhs.t.canon("residual_add");
        let folded_or_added = rhs.t.with(rhs.layer, |b| {
            if b.try_fold_beta(rhs.id, self.id, &plain, &acc) {
                rhs.id
            } else {
                let shape = b.value_shape(self.id);
                b.launch(
                    &add,
                    vec![],
                    None,
                    vec![rhs.id, self.id],
                    vec![(shape, DType::BF16)],
                )[0]
            }
        });
        self.id = folded_or_added;
        self.layer = rhs.layer;
    }
}
