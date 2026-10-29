//! `ops.attn.attend` records what a model's hand-written dispatch did: the
//! same plans over the same lanes, the same kernels on the same rows. A
//! model that calls it traces to the same computation.

use poem_compiler::catalog::{one_rank, replacing, trace_of};
use poem_ir::{Dtype, Platform, canon};

/// Llama 3's forward as it was written before `attend`: its plans made up
/// front, its kinds of lane told apart by hand.
const BY_HAND: &str = r#"
def forward(m, inputs):
    ([input_m, input_d], input_p) = inputs.partition([fact.has(fact.Mask), fact.single_token()])
    plans = struct(
        m = ops.attn.plan_prefill(input_m, m.q_heads, m.kv_heads, m.head_dim, None),
        d = ops.attn.plan_decode(input_d, m.q_heads, m.kv_heads, m.head_dim, None),
        p = ops.attn.plan_prefill(input_p, m.q_heads, m.kv_heads, m.head_dim, None),
    )
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
    routes = inputs.adapter_routes()

    def block(l, w, y):
        x = ops.elemwise.rmsnorm(y, w.attn_norm, m.eps)
        o = attention_by_hand(x, inputs, m, plans, mask, positions, scaling, w)
        adapted = fact.has(fact.Adapter)
        o = ops.linear.lora_correct(x.on(adapted), w.lora_a, w.lora_b, routes, o.on(adapted))
        y = ops.elemwise.residual_add(o, y)
        x = ops.elemwise.rmsnorm(y, w.mlp_norm, m.eps)
        f = ops.linear.matmul(ops.linear.glu(ops.linear.matmul(x, w.gate), gating.swiglu(), up = ops.linear.matmul(x, w.up)), w.down)
        return ops.elemwise.residual_add(f, y)

    y = inputs.fold_layers(m.layers, y, block)
    x = ops.elemwise.rmsnorm(y, m.final_norm, m.eps)
    x = ops.gather.gather_rows(x, inputs.readout_rows())
    logits = ops.linear.lm_head(x, m.head)
    draft_self(logits)
    output("logits", logits)

def attention_by_hand(x, inputs, m, plans, mask, positions, scaling, w):
    pages = inputs.kv(w.kv)
    d = m.head_dim
    q, k, v = ops.linear.matmul(x, w.q), ops.linear.matmul(x, w.k), ops.linear.matmul(x, w.v)
    q, k = ops.elemwise.rope(q, positions, d, m.theta, rotary.full(), k = k, scaling = scaling)
    ops.attn.kv_append(k, v, pages, inputs.write_page(w.kv), inputs.write_offset(w.kv))

    o = branch(q, [
        (fact.has(fact.Mask), lambda q: ops.attn.prefill(q, plans.m, pages, None, d, m.kv_heads, m.sm_scale, mask = mask)),
        (fact.single_token(), lambda q: ops.attn.decode(q, plans.d, pages, None, d, m.sm_scale)),
    ], otherwise = lambda q: ops.attn.prefill(q, plans.p, pages, None, d, m.kv_heads, m.sm_scale))
    output("attn.scores", ops.attn.score_capture(q, inputs, w.kv, m.q_heads, m.kv_heads, d, m.sm_scale, observe = 32, kv_max = 2048))
    return ops.linear.matmul(o, w.o)
"#;

#[test]
fn attend_is_the_dispatch_a_model_wrote_by_hand() {
    for id in ["llama31-8b", "llama33-70b-mini-l4"] {
        for (w, kv) in [(Dtype::Bf16, Dtype::Bf16), (Dtype::Bf16, Dtype::E4m3)] {
            let stock = trace_of(id, w, kv, Platform::Cuda);
            let by_hand = replacing(id, "forward", BY_HAND)
                .unwrap_or_else(|why| panic!("{why}"))
                .trace(id, &one_rank(w, kv), id, Platform::Cuda)
                .unwrap_or_else(|why| panic!("{id}: {why:#}"));
            assert_eq!(
                canon::difference(&by_hand, &stock),
                None,
                "{id} at {w:?}/{kv:?}"
            );
        }
    }
}
