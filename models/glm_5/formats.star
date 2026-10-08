# How a GLM-5 checkpoint is laid out: transformers' names, each expert's
# projections a tensor of their own.

def formats(m):
    return [format("huggingface", read = lambda reads: huggingface(m, reads))]

def huggingface(m, reads):
    reads.read(m.embed, "model.embed_tokens.weight")
    reads.read(m.final_norm, "model.norm.weight")
    reads.read(m.head, "lm_head.weight")
    for l, layer in enumerate(m.layers):
        at = lambda tail: "model.layers.{}.{}".format(l, tail)
        attn = layer.attn
        index = attn.indexer
        reads.read(layer.attn_norm, at("input_layernorm.weight"))
        reads.read(layer.mlp_norm, at("post_attention_layernorm.weight"))
        reads.read(attn.q_a_proj, at("self_attn.q_a_proj.weight"))
        reads.read(attn.q_a_norm, at("self_attn.q_a_layernorm.weight"))
        reads.read(attn.q_b_proj, at("self_attn.q_b_proj.weight"))
        reads.read(attn.kv_a_proj, at("self_attn.kv_a_proj_with_mqa.weight"))
        reads.read(attn.kv_a_norm, at("self_attn.kv_a_layernorm.weight"))
        reads.read(attn.kv_b_proj, at("self_attn.kv_b_proj.weight"))
        reads.read(attn.o_proj, at("self_attn.o_proj.weight"))
        reads.read(index.q_proj, at("self_attn.indexer.wq_b.weight"))
        reads.read(index.k_proj, at("self_attn.indexer.wk.weight"))
        reads.read(index.weights_proj, at("self_attn.indexer.weights_proj.weight"))
        reads.read(index.k_norm, at("self_attn.indexer.k_norm.weight"))
        reads.read(index.k_norm_bias, at("self_attn.indexer.k_norm.bias"))
        f = layer.mlp
        if not f.routed:
            reads.read_concat(f.gate_up, [at("mlp.gate_proj.weight"), at("mlp.up_proj.weight")])
            reads.read(f.down, at("mlp.down_proj.weight"))
            continue
        reads.read(f.router, at("mlp.gate.weight"))
        expert = lambda e, leaf: src(at("mlp.experts.{}.{}.weight".format(e, leaf)))
        reads.read_expr(f.gate_up, concat(0, [
            concat(0, [expert(e, "gate_proj"), expert(e, "up_proj")])
                .transmute([1, -1, m.hidden], encoding(f.gate_up.dtype))
            for e in range(f.experts)
        ]))
        reads.read_expr(f.down, concat(0, [
            expert(e, "down_proj").transmute([1, m.hidden, -1], encoding(f.down.dtype))
            for e in range(f.experts)
        ]))
        if f.shared != None:
            reads.read_concat(f.shared.gate_up, [
                at("mlp.shared_experts.gate_proj.weight"),
                at("mlp.shared_experts.up_proj.weight"),
            ])
            reads.read(f.shared.down, at("mlp.shared_experts.down_proj.weight"))
