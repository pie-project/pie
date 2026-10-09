# The forward pieces the Qwen 3.5-family trunks share: the draft head's
# gated attention, and the routed MLP.

def draft_attn(x, inputs, plan, a, head_dim, kv_heads, rope, append = True):
    """A draft head's gated attention over `plan`, rotating q and k by
    `rope(q, k)`; `append` caches this step's k and v."""
    d = head_dim
    pages = inputs.kv(a.kv)
    write_page = inputs.write_page(a.kv)
    write_offset = inputs.write_offset(a.kv)
    q, gate = ops.layout.split_q_gate(ops.linear.matmul(x, a.qg_proj), d)
    k = ops.linear.matmul(x, a.k_proj)
    v = ops.linear.matmul(x, a.v_proj)
    q = ops.elemwise.rmsnorm_per_head_plus_one(q, a.q_norm, d, a.q_norm_eps)
    k = ops.elemwise.rmsnorm_per_head_plus_one(k, a.k_norm, d, a.k_norm_eps)
    q, k = rope(q, k)
    if append:
        ops.attn.kv_append(k, v, pages, write_page, write_offset)
    o = ops.attn.prefill(q, plan, pages, None, d, kv_heads, a.sm_scale)
    return ops.linear.matmul(ops.elemwise.gate_sigmoid_mul(o, gate), a.o_proj)

def moe(x, f):
    """The routed MLP: `f.top_k` of `f.experts`, plus the gated shared expert."""
    experts, weights = ops.linear.moe_topk_softmax(ops.linear.matmul(x, f.router), f.experts, f.top_k)

    def select(act, bank):
        if bank.dtype in [dtype.bf16, dtype.f16, dtype.f32]:
            return ops.linear.moe_matmul_select(act, bank, experts, f.top_k)
        return ops.linear.moe_matmul_select_quant(act, bank, experts, f.top_k)

    hidden = ops.linear.mlp_swiglu(select(x, f.gate_up), f.inter)
    routed = ops.linear.moe_weighted_sum(select(hidden, f.down), weights)
    shared = ops.linear.matmul(
        ops.linear.mlp_swiglu(ops.linear.matmul(x, f.shared_gate_up), f.shared_inter),
        f.shared_down,
    )
    return ops.linear.moe_sigmoid_gate_add(routed, shared, ops.linear.matmul(x, f.shared_gate))
