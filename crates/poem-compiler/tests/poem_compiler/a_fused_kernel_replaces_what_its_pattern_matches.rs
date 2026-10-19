use poem::ops::{attn, elemwise, layout, linear};
use poem::pattern::{Pattern, template};
use poem_compiler::fuse::fuse;
use poem_ir::{Operands, Trace};

fn trace(build: impl FnOnce(&Pattern)) -> Trace {
    template(build).trace
}

fn ops(trace: &Trace) -> Vec<&'static str> {
    trace.nodes.iter().map(|node| node.op.name()).collect()
}

fn biased(p: &Pattern) -> poem::Value {
    linear::matmul(&p.rows("act", 64), &p.weight("w", [64, 64]))
}

#[test]
fn a_matmul_and_its_bias_fold_into_one_epilogue() {
    let t = trace(|p| {
        elemwise::add_bias(&p.weight("bias", [64]), &biased(p));
    });
    assert_eq!(
        ops(&fuse(t, &["linear.matmul_bias"])),
        ["linear.matmul_bias"]
    );
}

#[test]
fn a_backend_forms_only_the_kernels_it_lists() {
    let t = trace(|p| {
        elemwise::add_bias(&p.weight("bias", [64]), &biased(p));
    });
    assert_eq!(
        ops(&fuse(t, &["elementwise.residual_add_rmsnorm"])),
        ["linear.matmul", "elementwise.add_bias"]
    );
}

#[test]
fn a_value_read_past_the_match_keeps_the_pair_apart() {
    let t = trace(|p| {
        let y = biased(p);
        elemwise::add_bias(&p.weight("bias", [64]), &y);
        elemwise::silu(&y);
    });
    assert_eq!(
        ops(&fuse(t, &["linear.matmul_bias"])),
        ["linear.matmul", "elementwise.add_bias", "elementwise.silu"]
    );
}

#[test]
fn an_op_between_the_pair_stays_ahead_of_the_fused_one() {
    let t = trace(|p| {
        let y = biased(p);
        elemwise::silu(&p.rows("other", 64));
        elemwise::add_bias(&p.weight("bias", [64]), &y);
    });
    assert_eq!(
        ops(&fuse(t, &["linear.matmul_bias"])),
        ["elementwise.silu", "linear.matmul_bias"]
    );
}

#[test]
fn an_op_that_overwrites_what_the_match_reads_keeps_it_in_place() {
    let t = trace(|p| {
        let act = p.rows("act", 64);
        let y = linear::matmul(&act, &p.weight("w", [64, 64]));
        elemwise::residual_add(&p.rows("other", 64), &act);
        elemwise::add_bias(&p.weight("bias", [64]), &y);
    });
    assert_eq!(
        ops(&fuse(t, &["linear.matmul_bias"])),
        [
            "linear.matmul",
            "elementwise.residual_add",
            "elementwise.add_bias"
        ]
    );
}

fn dense_mlp(p: &Pattern) -> (poem::Value, poem::Value) {
    let packed = linear::matmul(&p.rows("x", 64), &p.weight("gate_up", [128, 64]));
    let h = linear::mlp_swiglu(&packed, 64);
    let y = linear::matmul(&h, &p.weight("down", [64, 64]));
    (h, y)
}

#[test]
fn a_dense_mlp_folds_into_one_op() {
    let fused = fuse(trace(|p| drop(dense_mlp(p))), &["linear.mlp_swiglu"]);
    assert_eq!(ops(&fused), ["linear.mlp_swiglu"]);
    let Some(poem_ir::Operation::Fused(poem_ir::Fused::MlpSwiglu { intermediate, .. })) =
        fused.nodes.first().map(|node| &node.op)
    else {
        panic!("the op is the fused mlp");
    };
    assert_eq!(*intermediate, 64);
}

#[test]
fn a_dense_mlp_whose_intermediate_is_read_elsewhere_stays_apart() {
    // Bonsai rotates the swiglu output before `down`: `h` has another
    // reader, so the chain is not one kernel's.
    let fused = fuse(
        trace(|p| {
            let (h, _) = dense_mlp(p);
            elemwise::silu(&h);
        }),
        &["linear.mlp_swiglu"],
    );
    assert_eq!(
        ops(&fused),
        [
            "linear.matmul",
            "linear.mlp_swiglu",
            "linear.matmul",
            "elementwise.silu"
        ]
    );
}

fn qkv_write(p: &Pattern, q_eps: f32, k_eps: f32) {
    let positions = p.indices("positions");
    let (q, k, v) = layout::split_qkv(&p.rows("packed", 512), 256, 128);
    let v = elemwise::rmsnorm_no_scale(&v, 64, q_eps);
    let q = elemwise::rmsnorm_per_head(&q, &p.weight("q_norm", [64]), 64, q_eps);
    let k = elemwise::rmsnorm_per_head(&k, &p.weight("k_norm", [64]), 64, k_eps);
    let (q, k) = elemwise::rope_full(&q, &k, &positions, 64, 10_000.0, false);
    attn::kv_append(
        &k,
        &v,
        p.cache("kv"),
        &p.rows("write_page", 1),
        &p.rows("write_offset", 1),
    );
    elemwise::silu(&q);
}

#[test]
fn the_qkv_write_folds_when_its_norms_share_an_epsilon() {
    let fused = fuse(
        trace(|p| qkv_write(p, 1e-6, 1e-6)),
        &["custom_cuda.qkv_fused_qknorm_rope_vnorm_write"],
    );
    assert_eq!(
        ops(&fused),
        [
            "custom_cuda.qkv_fused_qknorm_rope_vnorm_write",
            "elementwise.silu"
        ]
    );
    let Some(poem_ir::Operation::Fused(poem_ir::Fused::QkvFusedQknormRopeVnormWrite {
        kv_heads,
        head_dim,
        rotary_dim,
        ..
    })) = fused.nodes.first().map(|node| &node.op)
    else {
        panic!("the first op is the fused write");
    };
    assert_eq!((*kv_heads, *head_dim, *rotary_dim), (2, 64, 64));
}

/// Muse Spark's form folds too: scaleless q and k norms, v as it is, and a
/// global layer's rows unturned.
#[test]
fn the_scaleless_unturned_qkv_write_folds() {
    let fused = fuse(
        trace(|p| {
            let (q, k, v) = layout::split_qkv(&p.rows("packed", 512), 256, 128);
            let q = elemwise::rmsnorm_no_scale(&q, 64, 1e-5);
            let k = elemwise::rmsnorm_no_scale(&k, 64, 1e-5);
            let (page, offset) = (p.rows("write_page", 1), p.rows("write_offset", 1));
            attn::kv_append(&k, &v, p.cache("kv"), &page, &offset);
            elemwise::silu(&q);
        }),
        &["custom_cuda.qkv_fused_qknorm_rope_vnorm_write"],
    );
    let Some(poem_ir::Operation::Fused(poem_ir::Fused::QkvFusedQknormRopeVnormWrite {
        q_norm_weight: None,
        v_norm: false,
        rotary_dim: 0,
        positions: None,
        ..
    })) = fused.nodes.first().map(|node| &node.op)
    else {
        panic!("the first op is the fused write, scaleless and unturned");
    };
}

/// MiMo's form folds too: no norms, v scaled, the first 64 entries of a
/// head turned in pairs 32 apart; its heads laid 256 wide from the packed
/// row's own widths or already so.
#[test]
fn the_unnormed_partly_turned_qkv_write_folds() {
    for padded in [false, true] {
        let fused = fuse(
            trace(|p| {
                let packed = p.rows("packed", 1024);
                let (q, k, v) = if padded {
                    layout::split_qkv_padded(&packed, [2, 1], [192, 128], 256)
                } else {
                    layout::split_qkv(&packed, 512, 256)
                };
                let v = elemwise::mul_scalar(0.707, &v);
                let positions = p.indices("positions");
                let (q, k) = elemwise::rope_partial(&q, &k, &positions, 64, 256, 1e4, None);
                let (page, offset) = (p.rows("write_page", 1), p.rows("write_offset", 1));
                attn::kv_append(&k, &v, p.cache("kv"), &page, &offset);
                elemwise::silu(&q);
            }),
            &["custom_cuda.qkv_fused_qknorm_rope_vnorm_write"],
        );
        let Some(poem_ir::Operation::Fused(poem_ir::Fused::QkvFusedQknormRopeVnormWrite {
            qk_norm: false,
            rotary_dim: 64,
            span: 32,
            v_scale,
            widths,
            kv_heads: 1,
            ..
        })) = fused.nodes.first().map(|node| &node.op)
        else {
            panic!("the first op is the fused write, unnormed and partly turned");
        };
        assert_eq!(*v_scale, 0.707);
        assert_eq!(*widths, if padded { [192, 128] } else { [256, 256] });
    }
}

#[test]
fn the_qkv_write_stays_apart_when_its_norms_do_not() {
    let apart = trace(|p| qkv_write(p, 1e-6, 1e-5));
    let fused = fuse(
        apart.clone(),
        &["custom_cuda.qkv_fused_qknorm_rope_vnorm_write"],
    );
    assert_eq!(ops(&fused), ops(&apart), "no op folds");
}

fn yarn_write(p: &Pattern, interleaved: bool) {
    let positions = p.indices("positions");
    let (q, k, v) = (p.rows("q", 512), p.rows("k", 128), p.rows("v", 128));
    let (q, k) = elemwise::rope_yarn(
        &q,
        &k,
        &positions,
        64,
        150_000.0,
        32.0,
        32.0,
        1.0,
        1.3,
        4096,
        interleaved,
    );
    attn::kv_append(
        &k,
        &v,
        p.cache("kv"),
        &p.rows("write_page", 1),
        &p.rows("write_offset", 1),
    );
    elemwise::silu(&q);
}

/// gpt-oss's YaRN and its cache write fold into one launch, carrying the
/// rotation's every parameter.
#[test]
fn a_yarn_rotation_and_its_cache_write_fold_into_one_op() {
    for interleaved in [false, true] {
        let fused = fuse(
            trace(|p| yarn_write(p, interleaved)),
            &["elementwise.rope_yarn_append"],
        );
        assert_eq!(
            ops(&fused),
            ["elementwise.rope_yarn_append", "elementwise.silu"]
        );
        let Some(poem_ir::Operation::Fused(poem_ir::Fused::RopeYarnAppend {
            head_dim,
            theta,
            factor,
            original_max_position,
            interleaved: carried,
            ..
        })) = fused.nodes.first().map(|node| &node.op)
        else {
            panic!("the first op is the fused rotation");
        };
        assert_eq!(
            (*head_dim, *theta, *factor, *original_max_position, *carried),
            (64, 150_000.0, 32.0, 4096, interleaved)
        );
    }
}

/// gpt-oss's packed q/k/v row, split, rotated and written, folds into one
/// op reading the packed row; with the split left to read elsewhere, the
/// rotation and write fold as they did.
#[test]
fn a_split_rotation_and_its_cache_write_fold_into_one_op() {
    let packed_write = |interleaved: bool, split_read: bool| {
        trace(move |p| {
            let positions = p.indices("positions");
            let packed = p.rows("packed", 768);
            let (q, k, v) = layout::split_qkv(&packed, 512, 128);
            let (qr, kr) = elemwise::rope_yarn(
                &q,
                &k,
                &positions,
                64,
                150_000.0,
                32.0,
                32.0,
                1.0,
                1.3,
                4096,
                interleaved,
            );
            attn::kv_append(
                &kr,
                &v,
                p.cache("kv"),
                &p.rows("write_page", 1),
                &p.rows("write_offset", 1),
            );
            elemwise::silu(&qr);
            if split_read {
                elemwise::silu(&v);
            }
        })
    };
    let kernels = [
        "elementwise.split_rope_yarn_append",
        "elementwise.rope_yarn_append",
    ];
    for interleaved in [false, true] {
        let fused = fuse(packed_write(interleaved, false), &kernels);
        assert_eq!(
            ops(&fused),
            ["elementwise.split_rope_yarn_append", "elementwise.silu"]
        );
        let Some(poem_ir::Operation::Fused(poem_ir::Fused::SplitRopeYarnAppend {
            q_width,
            kv_width,
            head_dim,
            theta,
            original_max_position,
            interleaved: carried,
            ..
        })) = fused.nodes.first().map(|node| &node.op)
        else {
            panic!("the first op is the fused split rotation");
        };
        assert_eq!(
            (
                *q_width,
                *kv_width,
                *head_dim,
                *theta,
                *original_max_position,
                *carried
            ),
            (512, 128, 64, 150_000.0, 4096, interleaved)
        );
    }
    let fused = fuse(packed_write(false, true), &kernels);
    assert_eq!(
        ops(&fused),
        [
            "layout.split_qkv",
            "elementwise.rope_yarn_append",
            "elementwise.silu",
            "elementwise.silu"
        ],
        "a split read past the rotation stays a split"
    );
}

/// DeepSeek-V4's windowed attention, its pool attention (selected or whole),
/// their merge and the sinks land as one op carrying the four; the lone
/// window and sinks of a pool-less layer stay `lse_sink`'s.
#[test]
fn deepseeks_window_pool_merge_and_sinks_land_as_one_op() {
    let layer = |selected: bool, pooled: bool| {
        trace(move |p| {
            let q = p.rows("q", 64 * 512);
            let cache = p.cache("kv");
            let plan = p.value(
                "plan",
                poem_ir::Ty::Struct(poem_ir::StructKind::AttnPrefillPlan),
            );
            let (o, lse) = attn::prefill_lse(&q, &plan, cache, Some(128), 512, 1, 0.04, None);
            let (o, lse) = if pooled {
                let positions = p.indices("positions");
                let requests = p.indices("requests");
                let entries = p.cache("entries");
                let (po, plse) = if selected {
                    let selection = p.rows("selection", 512);
                    attn::pool_lse_selected(
                        &q, &positions, &requests, &selection, entries, 4, 512, 64, 512, 0.04,
                    )
                } else {
                    attn::pool_lse(&q, &positions, &requests, entries, 128, 64, 512, 0.04)
                };
                attn::merge_lse(&o, &lse, &po, &plse, 64, 512)
            } else {
                (o, lse)
            };
            let out = attn::sink(&o, &lse, &p.weight("sinks", [64]), 512);
            elemwise::silu(&out);
        })
    };
    let kernels = ["attention.pool_attention", "attention.lse_sink"];
    for selected in [true, false] {
        let fused = fuse(layer(selected, true), &kernels);
        assert_eq!(
            ops(&fused),
            ["attention.pool_attention", "elementwise.silu"]
        );
        let Some(poem_ir::Operation::Fused(poem_ir::Fused::PoolAttention {
            window,
            pool,
            merge,
            head_dim,
            ..
        })) = fused.nodes.first().map(|node| &node.op)
        else {
            panic!("the first op is the pool attention");
        };
        assert_eq!(*head_dim, 512);
        assert_eq!(
            (window.name(), pool.name(), merge.name()),
            (
                "attention.prefill_lse",
                if selected {
                    "attention.pool_lse_selected"
                } else {
                    "attention.pool_lse"
                },
                "attention.merge_lse"
            )
        );
    }
    assert_eq!(
        ops(&fuse(layer(false, false), &kernels)),
        ["attention.lse_sink", "elementwise.silu"]
    );
}

/// gpt-oss's sinks fold onto the log-sum-exp attention before them, decode
/// and prefill alike, carrying the attention whole; a log-sum-exp read past
/// the sink keeps the pair apart.
#[test]
fn a_sink_folds_onto_the_attention_whose_log_sum_exp_it_reads() {
    let sinked = |decode: bool, scores: bool| {
        trace(move |p| {
            let q = p.rows("q", 512);
            let cache = p.cache("kv");
            let (o, lse) = if decode {
                let plan = p.value(
                    "plan",
                    poem_ir::Ty::Struct(poem_ir::StructKind::AttnDecodePlan),
                );
                attn::decode_lse(&q, &plan, cache, Some(128), 64, 0.125)
            } else {
                let plan = p.value(
                    "plan",
                    poem_ir::Ty::Struct(poem_ir::StructKind::AttnPrefillPlan),
                );
                attn::prefill_lse(&q, &plan, cache, None, 64, 8, 0.125, None)
            };
            let out = attn::sink(&o, &lse, &p.weight("sinks", [8]), 64);
            elemwise::silu(&out);
            if scores {
                elemwise::silu(&lse);
            }
        })
    };
    for decode in [true, false] {
        let fused = fuse(sinked(decode, false), &["attention.lse_sink"]);
        assert_eq!(ops(&fused), ["attention.lse_sink", "elementwise.silu"]);
        let Some(poem_ir::Operation::Fused(poem_ir::Fused::LseSink { attn, head_dim, .. })) =
            fused.nodes.first().map(|node| &node.op)
        else {
            panic!("the first op is the sinked attention");
        };
        assert_eq!(*head_dim, 64);
        assert_eq!(
            attn.name(),
            if decode {
                "attention.decode_lse"
            } else {
                "attention.prefill_lse"
            }
        );
        let apart = sinked(decode, true);
        assert_eq!(
            ops(&fuse(apart.clone(), &["attention.lse_sink"])),
            ops(&apart),
            "a log-sum-exp read elsewhere keeps the pair apart"
        );
    }
}

#[test]
fn a_projection_cut_into_two_halves_lands_as_one_fused_op() {
    let t = trace(|p| {
        let (left, right) = layout::split_rows(&biased(p), 40);
        elemwise::silu(&left);
        elemwise::silu(&right);
    });
    assert_eq!(
        ops(&fuse(t, &["linear.matmul_split_rows"])),
        [
            "linear.matmul_split_rows",
            "elementwise.silu",
            "elementwise.silu"
        ]
    );
}

#[test]
fn qwens_attention_prologue_lands_as_one_fused_op() {
    let t = trace(|p| {
        let d = 64;
        let (q, gate) = layout::split_q_gate(&p.rows("qg", 4 * 64), d);
        let q = elemwise::rmsnorm_per_head_plus_one(&q, &p.weight("q_norm", [64]), d, 1e-6);
        let k = elemwise::rmsnorm_per_head_plus_one(
            &p.rows("k", 64),
            &p.weight("k_norm", [64]),
            d,
            1e-6,
        );
        let positions = p.indices("positions");
        let (q, k) = elemwise::rope_partial(&q, &k, &positions, 16, d, 1e6, None);
        let cache = p.cache("cache");
        attn::kv_append(
            &k,
            &p.rows("v", 64),
            cache,
            &p.rows("wp", 1),
            &p.rows("wo", 1),
        );
        elemwise::silu(&q);
        elemwise::silu(&gate);
    });
    assert_eq!(
        ops(&fuse(t, &["elementwise.qg_qknorm_rope_append"])),
        [
            "elementwise.qg_qknorm_rope_append",
            "elementwise.silu",
            "elementwise.silu"
        ]
    );
}

#[test]
fn deepseeks_norm_and_tail_rotation_land_as_one_op() {
    let yarn = Some(poem_ir::ops::elemwise::Yarn {
        factor: 16.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        original_max_position: 65536,
    });
    let kernels = ["elementwise.norm_rope_last"];
    // q: a scale-free norm a 512-wide head, its tail turned.
    let q = trace(|p| {
        let positions = p.indices("positions");
        let y = elemwise::rmsnorm_no_scale(&p.rows("q", 64 * 512), 512, 1e-6);
        let q = elemwise::rope_partial_last_yarn(&y, &positions, 64, 512, 1e4, true, false, yarn);
        elemwise::silu(&q);
    });
    let fused = fuse(q, &kernels);
    assert_eq!(
        ops(&fused),
        ["elementwise.norm_rope_last", "elementwise.silu"]
    );
    let Some(poem_ir::Operation::Fused(poem_ir::Fused::NormRopeLast {
        weight,
        head_dim,
        interleaved,
        yarn: got,
        ..
    })) = fused.nodes.first().map(|node| &node.op)
    else {
        panic!("the first op is the norm and rotation");
    };
    assert_eq!(
        (weight.is_none(), *head_dim, *interleaved, *got),
        (true, 512, true, yarn)
    );
    // The kv plane: a scaled norm of a row one head wide.
    let plane = trace(|p| {
        let positions = p.indices("positions");
        let y = elemwise::rmsnorm(&p.rows("kv", 512), &p.weight("norm", [512]), 1e-6);
        let k = elemwise::rope_partial_last_yarn(&y, &positions, 64, 512, 1e4, true, true, None);
        elemwise::silu(&k);
    });
    assert_eq!(
        ops(&fuse(plane, &kernels)),
        ["elementwise.norm_rope_last", "elementwise.silu"]
    );
    // A row of several heads normed whole is not a head's norm.
    let wide = trace(|p| {
        let positions = p.indices("positions");
        let y = elemwise::rmsnorm(&p.rows("kv", 1024), &p.weight("norm", [1024]), 1e-6);
        let k = elemwise::rope_partial_last(&y, &positions, 64, 512, 1e4, true);
        elemwise::silu(&k);
    });
    assert_eq!(
        ops(&fuse(wide, &kernels)),
        [
            "elementwise.rmsnorm",
            "elementwise.rope_partial_last",
            "elementwise.silu"
        ]
    );
}

#[test]
fn deepseeks_kv_plane_norm_rotation_and_write_land_as_one_op() {
    let kv = trace(|p| {
        let positions = p.indices("positions");
        let y = elemwise::rmsnorm(&p.rows("kv", 512), &p.weight("norm", [512]), 1e-6);
        let k = elemwise::rope_partial_last_yarn(&y, &positions, 64, 512, 1e4, true, false, None);
        let cache = p.cache("kv_cache");
        attn::kv_append_shared(
            &k,
            cache,
            &p.rows("write_page", 1),
            &p.rows("write_offset", 1),
        );
    });
    assert_eq!(
        ops(&fuse(
            kv,
            &[
                "elementwise.norm_rope_last_append",
                "elementwise.norm_rope_last"
            ]
        )),
        ["elementwise.norm_rope_last_append"]
    );
}

/// A hyper-connection sublayer's pre-mix: the stream norm, the mix
/// projection, the combined gates and the RMS norm over their row land as
/// one op where the backend forms it, as the norm and projection plus the
/// gates and norm where it forms only those, and a normed row read past the
/// projection keeps the norm apart.
#[test]
fn a_hyper_connection_pre_mix_lands_as_one_op() {
    let sublayer = |escape: bool| {
        trace(move |p| {
            let streams = p.rows("streams", 4 * 64);
            let normed = elemwise::hc_rmsnorm_f32(&streams, 1e-6);
            let mixes = elemwise::hc_project(&normed, &p.weight("plane", [24, 256]), 4);
            let (x, post, comb) = elemwise::hc_gates(
                &mixes,
                &streams,
                &p.weight("scale", [3]),
                &p.weight("base", [24]),
                4,
                1e-6,
                2.0,
                20,
            );
            let y = elemwise::rmsnorm(&x, &p.weight("norm", [64]), 1e-6);
            elemwise::silu(&y);
            elemwise::hc_fold(&y, &streams, &post, &comb);
            if escape {
                elemwise::hc_project(&normed, &p.weight("other", [4, 256]), 4);
            }
        })
    };
    let all = [
        "elementwise.hc_pre",
        "elementwise.hc_prenorm_project",
        "elementwise.hc_gates_norm",
    ];
    assert_eq!(
        ops(&fuse(sublayer(false), &all)),
        [
            "elementwise.hc_pre",
            "elementwise.silu",
            "elementwise.hc_fold"
        ]
    );
    assert_eq!(
        ops(&fuse(sublayer(false), &all[1..])),
        [
            "elementwise.hc_prenorm_project",
            "elementwise.hc_gates_norm",
            "elementwise.silu",
            "elementwise.hc_fold"
        ]
    );
    let kept = ops(&fuse(sublayer(true), &all));
    assert!(
        kept.contains(&"elementwise.hc_rmsnorm_f32") && kept.contains(&"elementwise.hc_gates_norm"),
        "{kept:?}"
    );
}

#[test]
fn deepseeks_compressor_lands_as_one_op() {
    let chain = |read_latent: bool| {
        trace(move |p| {
            let (bpos, breq, brope) = (p.indices("bpos"), p.indices("breq"), p.indices("brope"));
            let pages = p.cache("pages");
            let ape = p.weight("ape", [4, 1024]);
            let g = attn::pool_gather(
                &bpos,
                &breq,
                pages,
                Some(&ape),
                512,
                4,
                poem_ir::Dtype::Bf16,
            );
            let latent = elemwise::rmsnorm(&g, &p.weight("norm", [512]), 1e-6);
            if read_latent {
                elemwise::silu(&latent);
            }
            let k =
                elemwise::rope_partial_last_yarn(&latent, &brope, 64, 512, 1e4, true, false, None);
            let pool = p.cache("entries");
            attn::pool_kv_append(&k, &bpos, &breq, pool, &p.rows("wp", 1), &p.rows("wo", 1));
        })
    };
    let kernels = ["attention.pool_compress", "elementwise.norm_rope_last"];
    assert_eq!(
        ops(&fuse(chain(false), &kernels)),
        ["attention.pool_compress"]
    );
    // A latent read past the match keeps the compressor apart.
    assert!(!ops(&fuse(chain(true), &kernels)).contains(&"attention.pool_compress"));
}

#[test]
fn deepseeks_q_and_kv_plane_land_as_one_op() {
    let layer = trace(|p| {
        let positions = p.indices("positions");
        let x = p.rows("x", 64);
        let q = linear::matmul(&x, &p.weight("q_up", [64 * 512, 64]));
        let q = elemwise::rmsnorm_no_scale(&q, 512, 1e-6);
        let q = elemwise::rope_partial_last_yarn(&q, &positions, 64, 512, 1e4, true, false, None);
        let plane = linear::matmul(&x, &p.weight("kv_down", [512, 64]));
        let plane = elemwise::rmsnorm(&plane, &p.weight("kv_norm", [512]), 1e-6);
        let plane =
            elemwise::rope_partial_last_yarn(&plane, &positions, 64, 512, 1e4, true, false, None);
        let cache = p.cache("kv");
        attn::kv_append_shared(&plane, cache, &p.rows("wp", 1), &p.rows("wo", 1));
        elemwise::silu(&q);
    });
    let kernels = [
        "elementwise.norm_rope_pair",
        "elementwise.norm_rope_last_append",
        "elementwise.norm_rope_last",
    ];
    assert_eq!(
        ops(&fuse(layer, &kernels)),
        [
            "linear.matmul",
            "linear.matmul",
            "elementwise.norm_rope_pair",
            "elementwise.silu"
        ]
    );
}

#[test]
fn deepseeks_attention_lands_its_output_turned_back() {
    let layer = trace(|p| {
        let q = p.rows("q", 64 * 512);
        let cache = p.cache("kv");
        let plan = p.value(
            "plan",
            poem_ir::Ty::Struct(poem_ir::StructKind::AttnPrefillPlan),
        );
        let (o, lse) = attn::prefill_lse(&q, &plan, cache, Some(128), 512, 1, 0.04, None);
        let positions = p.indices("positions");
        let requests = p.indices("requests");
        let entries = p.cache("entries");
        let (po, plse) = attn::pool_lse(&q, &positions, &requests, entries, 128, 64, 512, 0.04);
        let (o, lse) = attn::merge_lse(&o, &lse, &po, &plse, 64, 512);
        let out = attn::sink(&o, &lse, &p.weight("sinks", [64]), 512);
        let turned =
            elemwise::rope_partial_last_yarn(&out, &positions, 64, 512, 1e4, true, true, None);
        elemwise::silu(&turned);
    });
    let fused = fuse(layer, &["attention.pool_attention"]);
    assert_eq!(
        ops(&fused),
        ["attention.pool_attention", "elementwise.silu"]
    );
    let Some(poem_ir::Operation::Fused(poem_ir::Fused::PoolAttention { rotate, .. })) =
        fused.nodes.first().map(|node| &node.op)
    else {
        panic!("the first op is the pool attention");
    };
    assert!(
        matches!(
            rotate.as_deref(),
            Some(poem_ir::Elementwise::RopePartialLast { inverse: true, .. })
        ),
        "the attention holds the turn"
    );
}

#[test]
fn an_index_ranking_lands_its_queries_turn() {
    let t = trace(|p| {
        let positions = p.indices("positions");
        let q = elemwise::rope_partial_last_yarn(
            &p.rows("q", 64 * 128),
            &positions,
            64,
            128,
            1e4,
            true,
            false,
            None,
        );
        let w = p.rows("w", 64);
        let sel = attn::index_topk(
            &q,
            Some(&w),
            p.cache("keys"),
            64,
            128,
            512,
            4,
            poem_ir::IndexScore::Pooled,
        );
        elemwise::silu(&sel);
    });
    assert_eq!(
        ops(&fuse(t, &["attention.index_topk_rope"])),
        ["attention.index_topk_rope", "elementwise.silu"]
    );
}

#[test]
fn deepseeks_compressor_lands_its_state_write_too() {
    let t = trace(|p| {
        let (bpos, breq, brope) = (p.indices("bpos"), p.indices("breq"), p.indices("brope"));
        let pages = p.cache("pages");
        attn::pool_state_write(
            &p.rows("kv", 1024),
            &p.rows("score", 1024),
            pages,
            &p.rows("swp", 1),
            &p.rows("swo", 1),
            512,
            4,
        );
        let ape = p.weight("ape", [4, 1024]);
        let g = attn::pool_gather(
            &bpos,
            &breq,
            pages,
            Some(&ape),
            512,
            4,
            poem_ir::Dtype::Bf16,
        );
        let latent = elemwise::rmsnorm(&g, &p.weight("norm", [512]), 1e-6);
        let k = elemwise::rope_partial_last_yarn(&latent, &brope, 64, 512, 1e4, true, false, None);
        let pool = p.cache("entries");
        attn::pool_kv_append(&k, &bpos, &breq, pool, &p.rows("wp", 1), &p.rows("wo", 1));
    });
    let fused = fuse(t, &["attention.pool_compress"]);
    assert_eq!(ops(&fused), ["attention.pool_compress"]);
    let Some(poem_ir::Operation::Fused(poem_ir::Fused::PoolCompress { state, .. })) =
        fused.nodes.first().map(|node| &node.op)
    else {
        panic!("the op is the compressor");
    };
    assert!(state.is_some(), "the compressor holds its state write");
}

#[test]
fn every_rule_states_its_pattern_with_widths_it_can_hold() {
    // The patterns are recorded on first use; a split past its row (the
    // matmul-split rule's sentinel width once overflowed in debug builds and
    // wrapped in release) refuses here.
    assert!(poem_compiler::fuse::kernels().count() > 0);
}

#[test]
fn a_matmul_and_the_split_of_its_row_fold_into_one_op() {
    let fused = fuse(
        trace(|p| {
            let y = linear::matmul(&p.rows("act", 64), &p.weight("w", [96, 64]));
            let (left, right) = layout::split_rows(&y, 32);
            elemwise::silu(&left);
            elemwise::silu(&right);
        }),
        &["linear.matmul_split_rows"],
    );
    assert_eq!(
        ops(&fused),
        [
            "linear.matmul_split_rows",
            "elementwise.silu",
            "elementwise.silu"
        ]
    );
}

#[test]
#[should_panic(expected = "the split is past the row")]
fn a_split_past_its_row_is_refused() {
    trace(|p| drop(layout::split_rows(&p.rows("x", 64), 96)));
}

fn mhc_mixes(p: &Pattern) -> (poem::Value, poem::Value) {
    let normed = elemwise::hc_rmsnorm_f32(&p.rows("streams", 256), 1e-6);
    let mixes = elemwise::hc_project(&normed, &p.weight("dynamic", [24, 256]), 4);
    (normed, mixes)
}

#[test]
fn an_mhc_norm_read_only_by_its_projection_folds_into_it() {
    let fused = fuse(
        trace(|p| drop(mhc_mixes(p))),
        &["elementwise.hc_norm_project"],
    );
    assert_eq!(ops(&fused), ["elementwise.hc_norm_project"]);
    let Some(poem_ir::Operation::Fused(poem_ir::Fused::HcNormProject {
        eps, stream_count, ..
    })) = fused.nodes.first().map(|node| &node.op)
    else {
        panic!("the op is the fused projection");
    };
    assert_eq!((*eps, *stream_count), (1e-6, 4));
}

#[test]
fn an_mhc_norm_read_elsewhere_stays_written() {
    let fused = fuse(
        trace(|p| {
            let (normed, _) = mhc_mixes(p);
            elemwise::silu(&normed);
        }),
        &["elementwise.hc_norm_project"],
    );
    assert_eq!(
        ops(&fused),
        [
            "elementwise.hc_rmsnorm_f32",
            "elementwise.hc_project",
            "elementwise.silu"
        ]
    );
}

/// One gated-delta decode step (one key head, two value heads of 128), the
/// conv's output handed to `read` as well when it is given.
fn gdn_step(p: &Pattern, read_conv: bool) -> poem::Value {
    let qkvz = p.rows("qkvz", 6 * 128);
    let (qkv, z) = layout::split_rows(&qkvz, 4 * 128);
    let conv = attn::ssm_causal_conv1d(
        &qkv,
        &p.weight("conv", [4 * 128, 4]),
        None,
        p.cache("conv_state"),
        4,
    );
    let gates = attn::ssm_gdn_prep(
        &p.rows("ba", 4),
        &p.weight("dt_bias", [2]),
        &p.weight("a_log", [2]),
    );
    let y = attn::ssm_gated_delta(&conv, &z, &gates, p.cache("state"), 1, 2, 128, 128);
    if read_conv {
        elemwise::silu(&conv);
    }
    y
}

#[test]
fn a_gated_delta_step_folds_into_one_launch() {
    let fused = fuse(
        trace(|p| {
            elemwise::silu(&gdn_step(p, false));
        }),
        &["attention.ssm_gdn_step"],
    );
    assert_eq!(ops(&fused), ["attention.ssm_gdn_step", "elementwise.silu"]);
    let Some(poem_ir::Operation::Fused(poem_ir::Fused::GdnStep {
        width,
        conv_width,
        k_heads,
        v_heads,
        ..
    })) = fused.nodes.first().map(|node| &node.op)
    else {
        panic!("the op is the fused step");
    };
    assert_eq!((*width, *conv_width, *k_heads, *v_heads), (512, 4, 1, 2));
}

#[test]
fn a_gated_delta_step_whose_conv_is_read_elsewhere_stays_apart() {
    let fused = fuse(
        trace(|p| drop(gdn_step(p, true))),
        &["attention.ssm_gdn_step"],
    );
    assert_eq!(
        ops(&fused),
        [
            "layout.split_rows",
            "attention.ssm_causal_conv1d",
            "attention.ssm_gdn_prep",
            "attention.ssm_gated_delta",
            "elementwise.silu"
        ]
    );
}
