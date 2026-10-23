//! `poem_ir::canon` names a trace's values by what defines them, so a forward
//! that records the same ops in another order is the same trace, and one that
//! computes something else is not.

use poem_compiler::catalog::{one_rank, replacing, trace_of};
use poem_ir::{Dtype, Platform, canon};

const ID: &str = "llama31-8b";

fn traced(forward: &str) -> poem_ir::Trace {
    replacing(ID, "forward.poem", "forward", forward)
        .unwrap_or_else(|why| panic!("{why}"))
        .trace(ID, &one_rank(Dtype::Bf16, Dtype::Bf16), ID, Platform::Cuda)
        .unwrap_or_else(|why| panic!("{why:#}"))
}

/// Llama 3's forward with its dispatch written by hand, its attention plans
/// made after the embedding rather than before it.
const PLANS_LATER: &str = r#"
def forward(m, inputs):
    ([input_m, input_s, input_d], input_p) = inputs.partition([fact.has(fact.Mask), fact.scores(), fact.single_token()])
    mask = inputs.mask()
    positions = inputs.positions()
    scaling = rope_scaling({
        "rope_type": "llama3",
        "factor": m.rope.factor,
        "low_freq_factor": m.rope.low_freq_factor,
        "high_freq_factor": m.rope.high_freq_factor,
        "original_max_position_embeddings": m.rope.original_max_position,
    })
    y = ops.gather.embed(inputs.tokens(), m.embed, m.vocab)
    plans = struct(
        s = ops.attn.plan_prefill(input_s, m.q_heads, m.kv_heads, m.head_dim, None),
        p = ops.attn.plan_prefill(input_p, m.q_heads, m.kv_heads, m.head_dim, None),
        d = ops.attn.plan_decode(input_d, m.q_heads, m.kv_heads, m.head_dim, None),
        m = ops.attn.plan_prefill(input_m, m.q_heads, m.kv_heads, m.head_dim, None),
    )
    routes = inputs.adapter_routes()

    def block(l, w, y):
        x = ops.elemwise.rmsnorm(y, w.attn_norm, m.eps)
        o = attention_by_hand(x, inputs, m, plans, mask, positions, scaling, w)
        adapted = fact.has(fact.Adapter)
        o = ops.linear.lora_correct(x.on(adapted), w.lora_a, w.lora_b, routes, o.on(adapted))
        y = ops.elemwise.residual_add(o, y)
        x = ops.elemwise.rmsnorm(y, w.mlp_norm, m.eps)
        f = ops.linear.matmul(ops.linear.glu(ops.linear.matmul(x, w.gate_up), gating.swiglu(), m.inter), w.down)
        return ops.elemwise.residual_add(f, y)

    y = inputs.fold_layers(m.layers, y, block)
    x = ops.elemwise.rmsnorm(y, m.final_norm, m.eps)
    x = ops.gather.gather_rows(x, inputs.readout_rows())
    return ops.linear.lm_head(x, m.head)

def attention_by_hand(x, inputs, m, plans, mask, positions, scaling, w):
    pages = inputs.kv(w.kv)
    d = m.head_dim
    q, k, v = ops.gather.split_qkv(ops.linear.matmul(x, w.qkv), m.q_width, m.kv_width)
    seam.at(seam.ATTN_QV, [q, v])
    q, k = ops.elemwise.rope(q, positions, d, m.theta, rotary.full(), k = k, scaling = scaling)
    ops.attn.kv_append(k, v, pages, inputs.write_page(w.kv), inputs.write_offset(w.kv))
    seam.at(seam.ATTN_Q, [q])

    def scored(q):
        o, lse = ops.attn.prefill_lse(q, plans.s, pages, None, d, m.kv_heads, m.sm_scale)
        seam.at(seam.SCORES, [lse])
        return o

    o = switch(q, [
        (fact.has(fact.Mask), lambda q: ops.attn.masked(q, plans.m, mask, pages, None, d, m.kv_heads, True, m.sm_scale)),
        (fact.scores(), scored),
        (fact.single_token(), lambda q: ops.attn.decode(q, plans.d, pages, None, d, m.sm_scale)),
    ], otherwise = lambda q: ops.attn.prefill(q, plans.p, pages, None, d, m.kv_heads, m.sm_scale))
    seam.at(seam.ATTN_OUT, [o])
    return ops.linear.matmul(o, w.o)
"#;

#[test]
fn a_trace_is_the_same_whatever_order_it_was_recorded_in() {
    let stock = trace_of(ID, Dtype::Bf16, Dtype::Bf16, Platform::Cuda);
    assert_eq!(canon::difference(&stock, &stock), None, "a trace is itself");

    let later = traced(PLANS_LATER);
    assert_ne!(
        stock.nodes, later.nodes,
        "the plans were recorded elsewhere"
    );
    assert_eq!(
        canon::difference(&stock, &later),
        None,
        "and it is the same trace"
    );

    // The final norm at another epsilon is another computation.
    let unlike = traced(&PLANS_LATER.replace(
        "x = ops.elemwise.rmsnorm(y, m.final_norm, m.eps)",
        "x = ops.elemwise.rmsnorm(y, m.final_norm, m.eps * 2.0)",
    ));
    let why = canon::difference(&stock, &unlike).expect("a changed op is a changed trace");
    assert!(
        why.contains("nodes"),
        "named by the node that differs: {why}"
    );
}
