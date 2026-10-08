# How a Muse Glimmer checkpoint is laid out: transformers' names, the text
# model under `model.language_model.`.

TRUNK = "model.language_model."

def formats(m):
    return [format("huggingface", read = lambda reads: huggingface(m, reads))]

def huggingface(m, reads):
    at = lambda leaf: TRUNK + leaf
    reads.read(m.embed, at("embed_tokens.weight"))
    reads.read(m.final_norm, at("norm.weight"))
    reads.read(m.lm_head, "lm_head.weight")
    for l, w in enumerate(m.layers):
        n = lambda leaf: "{}layers.{}.{}".format(TRUNK, l, leaf)
        reads.read(w.attn_norm, n("input_layernorm.weight"))
        reads.read(w.post_attn_norm, n("post_attention_layernorm.weight"))
        reads.read(w.pre_ffw_norm, n("pre_feedforward_layernorm.weight"))
        reads.read(w.post_ffw_norm, n("post_feedforward_layernorm.weight"))
        reads.read_concat(w.qkv, [
            n("self_attn.q_proj.weight"),
            n("self_attn.k_proj.weight"),
            n("self_attn.v_proj.weight"),
        ])
        reads.read(w.gate, n("self_attn.gate_proj.weight"))
        reads.read(w.o_proj, n("self_attn.o_proj.weight"))
        reads.read_concat(w.gate_up, [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")])
        reads.read(w.down, n("mlp.down_proj.weight"))
