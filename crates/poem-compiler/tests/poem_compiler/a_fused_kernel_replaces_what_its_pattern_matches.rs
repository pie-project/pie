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
            let (o, lse) = attn::prefill_lse(&q, &plan, cache, Some(128), 512, 1, 0.04);
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
                attn::prefill_lse(&q, &plan, cache, None, 64, 8, 0.125)
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
