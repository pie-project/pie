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
