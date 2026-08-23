pub mod facts;

use self::facts::{
    Activation, LlamaLikeCudaFacts, LlamaLikeFacts, LlamaLikeMetalFacts, NormPlacement, QkNorm,
};
use model_dsl::axes::{Bf16Ax, DtypeAxis, KvAxis, NativeKv};
use model_dsl::{
    self as dsl, MatW, Val, add_bias, attention, cuda, matmul, rmsnorm, rope, split_qkv, swiglu,
};
use model_ir::trace::{DType, Dim, FireClass, ForwardPlan, GuardPred, RopeKind, Shape};

pub fn llama_like(facts: &LlamaLikeFacts, norm_eps: f32, rope_theta: f32) -> ForwardPlan {
    dsl::trace_semantic("llama_like", &facts.shape(norm_eps), |m| {
        dsl::seam(m.trace(), &dsl::seam::IN, &[], None);
        let f = facts.clone();
        let q_w = f.q_width();
        let kv_w = f.kv_width();
        let post_norm = f.norm_placement == NormPlacement::Post;

        let mut y = m.embed();

        for l in 0..f.layers {
            let w = m.layer(l);

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

            let (q, k, v) = if f.qkv_bias {
                (
                    add_bias(&q, &w.q_bias),
                    add_bias(&k, &w.k_bias),
                    add_bias(&v, &w.v_bias),
                )
            } else {
                (q, k, v)
            };

            let (q, k) = if f.qk_norm == QkNorm::Off {
                (q, k)
            } else {
                (rmsnorm(&q, &w.q_norm), rmsnorm(&k, &w.k_norm))
            };
            let (q, k) = rope(
                &q,
                &k,
                f.rope,
                dsl::RopeShape {
                    num_q_heads: f.q_heads,
                    num_kv_heads: f.kv_heads,
                    head_dim: f.head_dim,
                    theta: rope_theta,
                    interleaved: false,
                },
            );
            w.kv.append(&k, &v);
            let a = attention(&q, &w.kv, q_w);

            if post_norm {
                y += rmsnorm(&matmul(&a, &w.o_proj), &w.attn_norm);
                let mlp = matmul(&swiglu(&matmul(&y, &w.gate_up), f.intermediate), &w.down);
                y += rmsnorm(&mlp, &w.mlp_norm);
            } else {
                y += matmul(&a, &w.o_proj);
                let x = rmsnorm(&y, &w.mlp_norm);
                y += matmul(&swiglu(&matmul(&x, &w.gate_up), f.intermediate), &w.down);
            }
        }

        let logits = m.logits(&rmsnorm(&y, &m.final_norm()));
        dsl::seam(m.trace(), &dsl::seam::OUT, &[&logits], None);
    })
}

fn all_reduce(t: &model_dsl::Trace, x: &Val, hidden: u32, cuda: &LlamaLikeCudaFacts) -> Val {
    if cuda.all_reduce_p2p_max_rows == 0 {
        return cuda::generated::all_reduce_bf16_out(x, x.layer(), None);
    }
    let shape = (Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16);
    let (g, v) = dsl::guarded_value(t, x.layer(), shape);

    g.arm(GuardPred::TokensLE(cuda.all_reduce_p2p_max_rows), || {
        let _ = cuda::all_reduce_p2p(x, hidden);
    })
    .otherwise(|| {
        let _ = cuda::generated::all_reduce_bf16_out(x, x.layer(), None);
    });
    v
}

fn shard_divides(f: &LlamaLikeFacts, tp: u32) -> bool {
    tp > 0
        && f.q_heads.is_multiple_of(tp)
        && f.kv_heads.is_multiple_of(tp)
        && f.intermediate.is_multiple_of(tp)
}

fn mlp(x: &Val, w: &dsl::Layer, packed: bool) -> Val {
    if packed {
        let gate_up = matmul(x, &w.gate_up);
        cuda::generated::chunked_swiglu(&gate_up, gate_up.layer(), None)
    } else {
        let gate = matmul(x, &w.gate_proj);
        let up = matmul(x, &w.up_proj);
        cuda::generated::swiglu(&gate, &up, gate.layer(), None)
    }
}

pub fn llama_like_cuda<A: DtypeAxis, K: KvAxis>(
    facts: &LlamaLikeFacts,
    cuda: &LlamaLikeCudaFacts,
    class: FireClass,
    norm_eps: f32,
    rope_theta: f32,
) -> ForwardPlan {
    const {
        assert!(matches!(A::DTYPE, model_ir::trace::DType::BF16));
        assert!(K::NATIVE_BF16);
    }
    llama_like_cuda_text(
        &format!("llama_like-{}", K::NAME),
        facts,
        cuda,
        class,
        norm_eps,
        rope_theta,
    )
}

pub type TraceFn = fn(&LlamaLikeFacts, &LlamaLikeCudaFacts, FireClass, f32, f32) -> ForwardPlan;

pub type ShippedA = Bf16Ax;

pub type ShippedKv = NativeKv;

pub const CATALOG: &[(&str, TraceFn)] =
    model_dsl::catalogue![("llama_like-kv-bf16", llama_like_cuda::<ShippedA, ShippedKv>),];

fn llama_like_metal_text(
    facts: &LlamaLikeFacts,
    metal: &LlamaLikeMetalFacts,
    class: FireClass,
) -> ForwardPlan {
    let multi_batch = class != FireClass::Decode;

    let shape = dsl::ModelShape {
        proj_repr: metal.proj_repr,
        ..facts.shape(metal.rms_eps)
    };
    dsl::trace_metal("llama_like", &shape, class, |m| {
        m.depth_window();

        let f = facts.clone();

        let point = dsl::metal::affine_point(metal.proj_repr, metal.affine_bits);

        let gemm_point =
            dsl::metal::affine_gemm_point(metal.proj_repr, metal.affine_bits, metal.qmm_tile);

        let post_norm = f.norm_placement == NormPlacement::Post;

        let sandwich = f.norm_placement == NormPlacement::Sandwich;

        let plus_one = |w: &dsl::NormW| u32::from(w.variant == model_ir::trace::NormVariant::Gemma);
        let norm = |x: &Val, w: &dsl::NormW, row: u32| {
            dsl::metal::generated::rms_single_row(
                x,
                &w.name,
                metal.rms_eps,
                row as i32,
                1,
                plus_one(w),
                1.0,
                w.layer,
                None,
            )
        };

        let norm_res = |x: &Val, w: &dsl::NormW, residual: &Val, row: u32| {
            dsl::metal::generated::rms_residual(
                x,
                &w.name,
                residual,
                metal.rms_eps,
                row as i32,
                1,
                plus_one(w),
                1.0,
                w.layer,
                None,
            )
        };

        let tile = metal.qmm_tile.0.max(1);

        let rows_guard = if metal.qmm_partial_rows {
            GuardPred::TokensGT(tile - 1)
        } else {
            GuardPred::TokensMultipleOf(tile)
        };

        let staged: std::cell::RefCell<std::collections::HashMap<model_ir::trace::ValueId, Val>> =
            std::cell::RefCell::new(std::collections::HashMap::new());
        let stage = |x: &Val| -> Val {
            if let Some(v) = staged.borrow().get(&x.key()) {
                return v.clone();
            }
            let v = dsl::metal::cast_qmm_input_when(x, rows_guard);
            staged.borrow_mut().insert(x.key(), v.clone());
            v
        };

        let precast = metal.qmm_fp16_precast
            && crate::shared::llama_like::project::qmm_fp16_precast(
                match metal.proj_repr {
                    model_dsl::WeightRepr::Scaled { group, .. } => group,
                    _ => 0,
                },
                metal.affine_bits,
            );

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
            g.arm(rows_guard, || match &half {
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

        let gemm_add = |x: &Val, w: &MatW, residual: &Val| {
            if !metal.fuse_residual_gemv {
                let y = gemm(x, w);
                return dsl::metal::generated::residual_add(&y, residual, y.layer(), None);
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
            g.arm(rows_guard, || match &half {
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

        let _ = metal.paged_multi_batch;
        let paged = true;

        assert!(
            !metal.gate_up_fused,
            "llama_like's Metal text has no packed gate\u{2016}up arm: `silu_mul` \
             takes two buffers and no Metal kernel splits a packed bank into \
             them. No deployment needs one -- `compile_load_plan` authors with \
             `Projections::InPlace` and the join declines under it -- so the \
             arm is refused at trace time rather than written untested."
        );

        let activate = |gate: &Val, up: &Val, width: u32| match metal.activation {
            Activation::SiluMul => dsl::metal::generated::silu_mul(gate, up, gate.layer(), None),
            Activation::SwiGlu { limit, alpha } => dsl::metal::generated::gptoss_swiglu(
                gate,
                up,
                width,
                limit,
                alpha,
                gate.layer(),
                None,
            ),
            Activation::Geglu => dsl::metal::generated::geglu_tanh(gate, up, gate.layer(), None),
        };

        let dense_ffn = |x: &Val, w: &dsl::Layer| {
            activate(&gemm(x, &w.gate_proj), &gemm(x, &w.up_proj), f.intermediate)
        };
        let gated = |x: &Val, router_in: &Val, w: &dsl::Layer| {
            if f.n_experts == 0 {
                return dense_ffn(x, w);
            }
            let k = f.experts_per_token.max(1);

            let router_x = if metal.router_input_norm {
                dsl::metal::generated::rms_single_row(
                    router_in,
                    &w.router_scale.name,
                    metal.rms_eps,
                    f.hidden as i32,
                    1,
                    plus_one(&w.router_scale),
                    (f.hidden as f32).powf(-0.5),
                    w.router_scale.layer,
                    None,
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
                dsl::metal::generated::add_bias(
                    &logits,
                    &w.router_bias.name,
                    w.router_bias.layer,
                    None,
                )
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

            let tile = if class == FireClass::Prefill {
                metal.moe_tile
            } else {
                None
            };
            let block = tile.map_or(dsl::metal::ROUTE_BLOCK_MATVEC, |t| t.0);
            let (perm, row_expert, tile_expert, inv) =
                dsl::metal::route_sort(&ids, f.n_experts, k, f.hidden, block);
            let rows = dsl::metal::route_gather(x, &perm, f.n_experts, k, f.hidden, block);

            let bank = |m: &dsl::MatW| match metal.moe_repr {
                Some(repr) => dsl::MatW { repr, ..m.clone() },
                None => m.clone(),
            };
            let bits = if metal.moe_repr.is_some() {
                metal.moe_bits
            } else {
                metal.affine_bits
            };

            let project = |x: &Val, m: &dsl::MatW, in_vec: u32| {
                if let Some(tile) = tile {
                    dsl::metal::routed_qmm(
                        x,
                        &row_expert,
                        &tile_expert,
                        &bank(m),
                        f.n_experts,
                        k,
                        bits,
                        tile,
                        metal.routed_qmm_fp16 && bits == 4,
                    )
                } else {
                    dsl::metal::routed_qmv(x, &row_expert, &bank(m), k, in_vec, false, bits)
                }
            };

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

            let shared = activate(
                &gemm(x, &w.shared_gate),
                &gemm(x, &w.shared_up),
                f.shared_intermediate,
            );
            let shared_down = gemm(&shared, &w.shared_down);
            let shared_gate = gemm(x, &w.shared_gate_proj);
            dsl::metal::generated::shared_expert_combine(
                &routed,
                &shared_down,
                &shared_gate,
                routed.layer(),
                None,
            )
        };

        let owes_down = f.n_experts == 0;

        let mixture_beside_dense = metal.dense_beside_moe && f.n_experts > 0;

        assert!(
            !mixture_beside_dense || sandwich,
            "a mixture beside the dense MLP is a SANDWICH block's shape and \
             this row states {:?}: there is no position under it for the \
             join, so a text that ran the branch anyway would be inventing \
             one",
            f.norm_placement
        );

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
            let normed = dsl::metal::generated::rms_single_row(
                &proj,
                "ple_proj_norm",
                metal.rms_eps,
                metal.per_layer_emb_dim as i32,
                1,
                u32::from(f.norm_variant == model_ir::trace::NormVariant::Gemma),
                1.0,
                None,
                None,
            );

            dsl::metal::generated::ple_combine(
                &normed,
                &token,
                std::f32::consts::FRAC_1_SQRT_2,
                None,
                None,
            )
        });

        for l in 0..f.layers {
            let w = m.layer(l);

            let x = if post_norm {
                y.clone()
            } else {
                norm(&y, &w.attn_norm, f.hidden)
            };

            let shares_kv = l >= f.layers.saturating_sub(metal.kv_shared_layers);

            let window = metal.window_left_at(l);

            let head_dim = metal.head_dim_at(l, f.head_dim);
            let kv_heads = metal.kv_heads_at(l, f.kv_heads);
            let q_w = f.q_heads * head_dim;
            let kv_w = kv_heads * head_dim;

            let at_w = |m: &dsl::MatW, width: u32| dsl::MatW { width, ..m.clone() };
            let (q_proj, k_proj, v_proj) = (
                at_w(&w.q_proj, q_w),
                at_w(&w.k_proj, kv_w),
                at_w(&w.v_proj, kv_w),
            );

            let (q, k, v) = if shares_kv {
                let q = gemm(&x, &q_proj);
                (q.clone(), q.clone(), q)
            } else if metal.v_from_k && window < 0 {
                let q = gemm(&x, &q_proj);
                let k = gemm(&x, &k_proj);
                (q, k.clone(), k)
            } else {
                (gemm(&x, &q_proj), gemm(&x, &k_proj), gemm(&x, &v_proj))
            };

            let (q, k, v) = if f.qkv_bias && metal.add_bias {
                (
                    dsl::metal::generated::add_bias(&q, &w.q_bias.name, w.q_bias.layer, None),
                    dsl::metal::generated::add_bias(&k, &w.k_bias.name, w.k_bias.layer, None),
                    dsl::metal::generated::add_bias(&v, &w.v_bias.name, w.v_bias.layer, None),
                )
            } else {
                (q, k, v)
            };

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
                (norm(&q, &w.q_norm, head_dim), norm(&k, &w.k_norm, head_dim))
            };

            let v = if metal.v_norm && !shares_kv {
                dsl::metal::generated::vnorm_single_row(
                    &v,
                    metal.rms_eps,
                    head_dim as i32,
                    v.layer(),
                    None,
                )
            } else {
                v
            };

            let (q, k) = if fused_qk_rope {
                (q, k)
            } else {
                dsl::metal::rope(
                    &q,
                    &k,
                    multi_batch,
                    metal.rope_theta_at(l),
                    1.0,
                    head_dim,
                    metal.rotary_dim_at(l, f.head_dim),
                    metal.rope_freq_table,
                    metal.rope_proportional,
                )
            };

            if !shares_kv {
                dsl::metal::kv_append(&k, &v, &w.kv, paged, head_dim, kv_heads);
            }

            let sink = metal.attn_sinks.then(|| format!("layer.{l}.attn_sinks"));
            let attend = |mb: bool| {
                dsl::metal::sdpa(
                    &q,
                    &w.kv,
                    q_w,
                    head_dim,
                    paged,
                    kv_heads,
                    window,
                    sink.as_deref(),
                    metal.attn_scale,
                    mb,
                )
            };

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

            let land = |a: &Val| gemm(a, &w.o_proj);
            if post_norm {
                y = norm_res(&land(&a), &w.attn_norm, &y, f.hidden);
                let h = gated(&y, &y, &w);
                let ffn = if owes_down { gemm(&h, &w.down) } else { h };
                y = norm_res(&ffn, &w.mlp_norm, &y, f.hidden);
            } else if sandwich {
                y = norm_res(&land(&a), &w.post_attn_norm, &y, f.hidden);
                if mixture_beside_dense {
                    let x1 = norm(&y, &w.mlp_norm, f.hidden);
                    let g1 = gemm(&dense_ffn(&x1, &w), &w.down);
                    let h1 = norm(&g1, &w.post_mlp_norm_1, f.hidden);

                    let x2 = norm(&y, &w.mlp_norm_2, f.hidden);
                    let g2 = gated(&x2, &y, &w);

                    let joined = norm_res(&g2, &w.post_mlp_norm_2, &h1, f.hidden);
                    y = norm_res(&joined, &w.post_mlp_norm, &y, f.hidden);
                } else {
                    let x = norm(&y, &w.mlp_norm, f.hidden);
                    let h = gated(&x, &x, &w);
                    let ffn = if owes_down { gemm(&h, &w.down) } else { h };
                    y = norm_res(&ffn, &w.post_mlp_norm, &y, f.hidden);
                }
            } else {
                y = gemm_add(&a, &w.o_proj, &y);
                if f.o_bias && metal.add_bias {
                    y = dsl::metal::generated::add_bias(&y, &w.o_bias.name, w.o_bias.layer, None);
                }
                let x = norm(&y, &w.mlp_norm, f.hidden);
                let h = gated(&x, &x, &w);

                y = if owes_down {
                    gemm_add(&h, &w.down, &y)
                } else {
                    dsl::metal::generated::residual_add(&h, &y, h.layer(), None)
                };
            }

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

                let h = dsl::metal::generated::geglu_tanh_strided(
                    &gate,
                    ple,
                    metal.per_layer_emb_dim,
                    1,
                    metal.per_layer_emb_dim,
                    f.layers * metal.per_layer_emb_dim,
                    metal.per_layer_emb_dim,
                    gate.layer(),
                    None,
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
                y = dsl::metal::generated::rms_residual_scaled(
                    &back,
                    &w.mlp_norm.name,
                    &y,
                    &back,
                    metal.rms_eps,
                    f.hidden as i32,
                    1,
                    plus_one(&w.mlp_norm),
                    1.0,
                    w.mlp_norm.layer,
                    None,
                );
            } else if metal.per_layer_scalar {
                y = dsl::metal::generated::layer_scalar_mul(
                    &y,
                    &format!("layer.{l}.scalar"),
                    y.layer(),
                    None,
                );
            }
        }

        let normed = norm(&y, &m.final_norm(), f.hidden);

        let sampled = dsl::metal::sample_rows(&normed, f.hidden);
        let head = if f.tied_embeddings {
            "embed"
        } else {
            "lm_head"
        };
        let logits = dsl::metal::lm_head(&sampled, head, f.vocab, metal.proj_repr, &point);

        let logits = if metal.logit_softcap > 0.0 {
            dsl::metal::generated::logit_softcap(&logits, metal.logit_softcap, logits.layer(), None)
        } else {
            logits
        };

        dsl::seam(m.trace(), &dsl::seam::OUT, &[&logits], None);
    })
}

pub fn llama_like_metal(
    facts: &LlamaLikeFacts,
    metal: &LlamaLikeMetalFacts,
    class: FireClass,
) -> ForwardPlan {
    llama_like_metal_text(facts, metal, class)
}

fn llama_like_cuda_text(
    family: &str,
    facts: &LlamaLikeFacts,
    cuda: &LlamaLikeCudaFacts,
    class: FireClass,
    norm_eps: f32,
    rope_theta: f32,
) -> ForwardPlan {
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
        ..facts.shape(norm_eps)
    };
    dsl::trace_cuda(family, &shape, class, |m| {
        dsl::seam(m.trace(), &dsl::seam::IN, &[], None);

        let rt = dsl::rt(m.trace());

        let mut f = facts.clone();
        f.q_heads /= tp;
        f.kv_heads /= tp;
        f.intermediate /= tp;
        let q_w = f.q_width();
        let kv_w = f.kv_width();
        let post_norm = f.norm_placement == NormPlacement::Post;

        let cuda_of = |class_want: FireClass| (class == class_want).then_some(cuda);

        if cuda_of(FireClass::Decode).is_some_and(|c| !c.xqa_decode)
            || cuda_of(FireClass::Prefill).is_some()
        {
            m.depth_window();
        }

        let pad_to = if cuda.head_dim_padded {
            assert!(
                cuda.head_dim_kernel > f.head_dim,
                "a padded deployment states the width its kernels run at"
            );
            cuda.head_dim_kernel
        } else {
            0
        };

        let plan_head_dim = if pad_to == 0 { f.head_dim } else { pad_to };

        let fused_post = cuda_of(FireClass::Decode).is_some_and(|c| c.decode_fused_post)
            && f.fused_qkv
            && f.qk_norm == QkNorm::PerHead
            && f.rope == RopeKind::Standard
            && !f.qkv_bias;

        let mut y = m.embed();

        let table =
            (fused_post && cuda_of(FireClass::Decode).is_some_and(|c| c.rope_table)).then(|| {
                cuda::generated::rope_standard_table(
                    (Shape(vec![Dim::Tokens, Dim::Const(f.head_dim)]), DType::F32),
                    f.head_dim as i32,
                    rope_theta,
                    &rt.positions(),
                    None,
                    None,
                )
            });

        for l in 0..f.layers {
            let w = m.layer(l);

            let x = if post_norm {
                y.clone()
            } else {
                dsl::cuda::rmsnorm(&y, &w.attn_norm)
            };

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
                    dsl::seam(m.trace(), &dsl::seam::ATTN_QV, &[&q, &v], Some(l));
                }

                let (q, k, v) = if f.qkv_bias {
                    (
                        add_bias(&q, &w.q_bias),
                        add_bias(&k, &w.k_bias),
                        add_bias(&v, &w.v_bias),
                    )
                } else {
                    (q, k, v)
                };

                let per_head_fused = f.qk_norm == QkNorm::PerHead && f.rope == RopeKind::Standard;
                let (q, k) = if per_head_fused {
                    cuda::generated::qk_rmsnorm_rope_bf16(
                        &q,
                        &k,
                        &w.q_norm.name,
                        &w.k_norm.name,
                        w.q_norm
                            .per_head
                            .expect("a per-head q norm carries its head dim")
                            as i32,
                        rope_theta,
                        w.q_norm.eps,
                        &rt.positions(),
                        q.layer(),
                        None,
                    )
                } else {
                    let (q, k) = if f.qk_norm == QkNorm::Off {
                        (q, k)
                    } else {
                        (
                            dsl::cuda::rmsnorm(&q, &w.q_norm),
                            dsl::cuda::rmsnorm(&k, &w.k_norm),
                        )
                    };

                    dsl::cuda::generated::rope_bf16(
                        &q,
                        &k,
                        f.q_heads as i32,
                        f.kv_heads as i32,
                        f.head_dim as i32,
                        rope_theta,
                        false,
                        &rt.positions(),
                        q.layer(),
                        None,
                    )
                };

                let (q, k, v) = if pad_to > 0 {
                    let padded = |heads: u32| {
                        (
                            Shape(vec![Dim::Tokens, Dim::Const(heads), Dim::Const(pad_to)]),
                            DType::BF16,
                        )
                    };
                    (
                        cuda::generated::pad_head_dim(
                            &q,
                            padded(f.q_heads),
                            f.head_dim as i32,
                            q.layer(),
                            None,
                        ),
                        cuda::generated::pad_head_dim(
                            &k,
                            padded(f.kv_heads),
                            f.head_dim as i32,
                            k.layer(),
                            None,
                        ),
                        cuda::generated::pad_head_dim(
                            &v,
                            padded(f.kv_heads),
                            f.head_dim as i32,
                            v.layer(),
                            None,
                        ),
                    )
                } else {
                    (q, k, v)
                };
                dsl::guard(
                    m.trace(),
                    GuardPred::HasWriteDesc,
                    || {
                        cuda::generated::write_kv_explicit_bf16(
                            &k,
                            &v,
                            &w.kv.cache(),
                            f.kv_heads as i32,
                            plan_head_dim as i32,
                            &rt.row_valid(),
                            Some(w.kv.l),
                            Some(w.kv.state()),
                        );
                    },
                    || cuda::write_kv_to_pages(&k, &v, &w.kv, f.kv_heads, plan_head_dim),
                );
                q
            };

            let window_left = cuda.window_left_at(l);
            let attn_out_shape = (
                Shape(vec![
                    Dim::Tokens,
                    Dim::Const(if pad_to > 0 { f.q_heads * pad_to } else { q_w }),
                ]),
                DType::BF16,
            );

            let a = match class {
                FireClass::Decode | FireClass::Prefill => {
                    let c = cuda;

                    let hoisted_q = (!fused_post).then(&general_qkv);

                    let (g, a) = dsl::guarded_value(m.trace(), Some(l), attn_out_shape.clone());

                    let masked_attention = |q: &Val| {
                        let peeled = |q: &Val| {
                            dsl::by_rows(m.trace(), Some(l), None, |r| {
                                r.arm(dsl::RowPred::Unmasked, || {

                                        if c.force_prefill_path {
                                            cuda::generated::dequant_kv_cache_layer_to_bf16_active(
                                &w.kv.cache(),
                                f.kv_heads as i32,
                                plan_head_dim as i32,
                                Some(w.kv.l),
                                Some(w.kv.state()),
                            );
                                            cuda::attention_flashinfer_prefill(
                                                &dsl::runtime::query_windows(q),
                                                &w.kv,
                                                window_left,
                                                plan_head_dim,
                                                0.0,
                                                0.0,
                                            );
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
                                                        plan_head_dim, 0.0, 0.0);
                                                    });
                                                })
                                                .otherwise(|| {
                                                    cuda::generated::dequant_kv_cache_layer_to_bf16_active(
                                                        &w.kv.cache(),
                                                        f.kv_heads as i32,
                                                        plan_head_dim as i32,
                                                        Some(w.kv.l),
                                                        Some(w.kv.state()),
                                                    );
                                                    cuda::attention_flashinfer_prefill(
                                                        &dsl::runtime::query_windows(q),
                                                        &w.kv,
                                                        window_left,
                                                        plan_head_dim,
                                                        0.0,
                                                        0.0,
                                                    );
                                                });
                                        }
                                    });
                                r.rest(|| {
                                    cuda::attention_flashinfer_prefill_custom(
                                        &dsl::runtime::query_windows(q),
                                        &w.kv,
                                        window_left,
                                        plan_head_dim,
                                        0.0,
                                        0.0,
                                    );
                                });
                            });
                        };
                        if c.head_dim_padded {
                            cuda::attention_flashinfer_prefill_custom(
                                &dsl::runtime::query_windows(q),
                                &w.kv,
                                window_left,
                                plan_head_dim,
                                0.0,
                                0.0,
                            );
                        } else if c.xqa_decode {
                            dsl::guard(
                                m.trace(),
                                GuardPred::WindowOne,
                                || {
                                    cuda::attention_flashinfer_prefill_custom(
                                        &dsl::runtime::query_windows(q),
                                        &w.kv,
                                        window_left,
                                        plan_head_dim,
                                        0.0,
                                        0.0,
                                    );
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
                            cuda::attention_xqa_decode(
                                q,
                                &w.kv,
                                window_left,
                                f.q_heads,
                                f.kv_heads,
                                plan_head_dim,
                                0.0,
                            );
                        } else if c.force_prefill_path {
                            cuda::generated::dequant_kv_cache_layer_to_bf16_active(
                                &w.kv.cache(),
                                f.kv_heads as i32,
                                plan_head_dim as i32,
                                Some(w.kv.l),
                                Some(w.kv.state()),
                            );
                            cuda::attention_flashinfer_prefill(
                                &dsl::runtime::query_windows(q),
                                &w.kv,
                                window_left,
                                plan_head_dim,
                                0.0,
                                0.0,
                            );
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
                                                0.0,
                                                0.0,
                                            );
                                        })
                                        .otherwise(|| {
                                            cuda::attention_flashinfer_decode(
                                                q,
                                                &w.kv,
                                                window_left,
                                                plan_head_dim,
                                                0.0,
                                                0.0,
                                            );
                                        });
                                })
                                .otherwise(|| {
                                    cuda::generated::dequant_kv_cache_layer_to_bf16_active(
                                        &w.kv.cache(),
                                        f.kv_heads as i32,
                                        plan_head_dim as i32,
                                        Some(w.kv.l),
                                        Some(w.kv.state()),
                                    );
                                    dsl::guarded(m.trace())
                                        .arm(GuardPred::WantsAttnScore, || {
                                            cuda::attention_flashinfer_prefill_capture(
                                                &dsl::runtime::query_windows(q),
                                                &w.kv,
                                                window_left,
                                                plan_head_dim,
                                                0.0,
                                                0.0,
                                            );
                                        })
                                        .otherwise(|| {
                                            cuda::attention_flashinfer_prefill(
                                                &dsl::runtime::query_windows(q),
                                                &w.kv,
                                                window_left,
                                                plan_head_dim,
                                                0.0,
                                                0.0,
                                            );
                                        });
                                });
                        }
                        dsl::seam(q.trace(), &dsl::seam::ATTN_OUT, &[q], Some(l));
                    };
                    if fused_post {
                        g.arm(GuardPred::HasCustomMask, || {
                            let q = general_qkv();
                            dsl::seam(q.trace(), &dsl::seam::ATTN_Q, &[&q], Some(l));
                            masked_attention(&q);
                            dsl::seam(q.trace(), &dsl::seam::ATTN_OUT, &[&q], Some(l));
                        })
                        .arm(GuardPred::HasLora, || {
                            let q = general_qkv();
                            attn_with_sites(&q);
                        })
                        .otherwise(|| {
                            let packed = matmul(&x, &w.qkv);

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
                                            f.kv_heads,
                                            rope_theta,
                                        );
                                    });
                                },
                                || {
                                    let (qt, kt, vt) = split_qkv(&packed, q_w, kv_w);
                                    let (_qt, kt) = cuda::generated::qk_rmsnorm_rope_bf16(
                                        &qt,
                                        &kt,
                                        &w.q_norm.name,
                                        &w.k_norm.name,
                                        w.q_norm
                                            .per_head
                                            .expect("a per-head q norm carries its head dim")
                                            as i32,
                                        rope_theta,
                                        w.q_norm.eps,
                                        &rt.positions(),
                                        qt.layer(),
                                        None,
                                    );
                                    dsl::guard(
                                        m.trace(),
                                        GuardPred::HasWriteDesc,
                                        || {
                                            cuda::generated::write_kv_explicit_bf16(
                                                &kt,
                                                &vt,
                                                &w.kv.cache(),
                                                f.kv_heads as i32,
                                                plan_head_dim as i32,
                                                &rt.row_valid(),
                                                Some(w.kv.l),
                                                Some(w.kv.state()),
                                            )
                                        },
                                        || {
                                            cuda::write_kv_to_pages(
                                                &kt,
                                                &vt,
                                                &w.kv,
                                                f.kv_heads,
                                                plan_head_dim,
                                            )
                                        },
                                    );
                                },
                            )
                            .expect("a value-producing row partition produces its value");
                            attn_with_sites(&q);
                        });
                    } else {
                        let q = hoisted_q.as_ref().expect("hoisted for the non-fused arms");
                        g.arm(GuardPred::HasCustomMask, || {
                            dsl::seam(q.trace(), &dsl::seam::ATTN_Q, &[q], Some(l));
                            masked_attention(q);
                            dsl::seam(q.trace(), &dsl::seam::ATTN_OUT, &[q], Some(l));
                        })
                        .otherwise(|| attn_with_sites(q));
                    }
                    a
                }
            };

            let a = if pad_to > 0 {
                cuda::generated::strip_head_dim(&a, f.head_dim as i32, a.layer(), None)
            } else {
                a
            };
            if post_norm {
                let o = matmul(&a, &w.o_proj);
                let o = if tp > 1 {
                    all_reduce(m.trace(), &o, f.hidden, cuda)
                } else {
                    o
                };
                y += dsl::cuda::rmsnorm(&o, &w.attn_norm);

                let act = mlp(&y, &w, cuda.gate_up_fused);

                let d_out = matmul(&act, &w.down);
                let d_out = if tp > 1 {
                    all_reduce(m.trace(), &d_out, f.hidden, cuda)
                } else {
                    d_out
                };
                y += dsl::cuda::rmsnorm(&d_out, &w.mlp_norm);
            } else if tp > 1 {
                let partial = matmul(&a, &w.o_proj);
                let summed = all_reduce(m.trace(), &partial, f.hidden, cuda);
                let x = cuda::generated::residual_add_rmsnorm(
                    &y,
                    &summed,
                    &w.mlp_norm.name,
                    norm_eps,
                    y.layer(),
                    None,
                );
                let act = mlp(&x, &w, cuda.gate_up_fused);

                let mlp_out = matmul(&act, &w.down);
                y += all_reduce(m.trace(), &mlp_out, f.hidden, cuda);
            } else {
                y += matmul(&a, &w.o_proj);
                let x = dsl::cuda::rmsnorm(&y, &w.mlp_norm);
                let act = mlp(&x, &w, cuda.gate_up_fused);
                y += matmul(&act, &w.down);
            }
        }

        let logits = m.logits(&dsl::cuda::rmsnorm(&y, &m.final_norm()));
        dsl::seam(m.trace(), &dsl::seam::OUT, &[&logits], None);
    })
}
