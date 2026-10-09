# How latent attention is named in a transformers checkpoint.

def named(a, at):
    """`a`'s weights, each beside the name `at` gives its transformers leaf,
    in the order a checkpoint lists them."""
    out = [
        (a.q_a_proj, at("self_attn.q_a_proj.weight")),
        (a.q_a_norm, at("self_attn.q_a_layernorm.weight")),
        (a.q_b_proj, at("self_attn.q_b_proj.weight")),
        (a.kv_a_proj, at("self_attn.kv_a_proj_with_mqa.weight")),
        (a.kv_a_norm, at("self_attn.kv_a_layernorm.weight")),
        (a.kv_b_proj, at("self_attn.kv_b_proj.weight")),
    ]
    if a.gate != None:
        out.append((a.gate, at("self_attn.g_proj.weight")))
    out.append((a.o_proj, at("self_attn.o_proj.weight")))
    return out
