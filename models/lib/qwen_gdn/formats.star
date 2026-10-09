# How a Qwen 3.5-family safetensors checkpoint spells its trunk: transformers'
# names, or mlx_lm's, which nest the trunk otherwise, store the norms less
# one, split the experts' gate and up, and lay the tower out channels-last.

load("//lib/reads/formats.star", "squeezed")

LAYOUTS = [
    struct(
        name = "transformers",
        trunk = "model.language_model.",
        head = "lm_head.weight",
        tower = "model.visual.",
        mlx = False,
    ),
    struct(
        name = "mlx_lm",
        trunk = "language_model.model.",
        head = "language_model.lm_head.weight",
        tower = "vision_tower.",
        mlx = True,
    ),
]

def safetensors_format(layout, read, states = []):
    """`layout`'s format, recognized by its embedding: `read(reads, layout)`."""
    return format(
        layout.name,
        recognizes = lambda checkpoint: has(layout.trunk + "embed_tokens.weight"),
        read = lambda reads: read(reads, layout),
        states = states,
    )

def norm_of(layout):
    """`norm(name)`: a norm as the layout stores it, read as the forward's."""
    if layout.mlx:
        return lambda name: src(name).bias(-1.0)
    return lambda name: src(name)

def read_attn(reads, a, n, norm):
    reads.read(a.qg_proj, n("self_attn.q_proj.weight"))
    reads.read(a.k_proj, n("self_attn.k_proj.weight"))
    reads.read(a.v_proj, n("self_attn.v_proj.weight"))
    reads.read(a.o_proj, n("self_attn.o_proj.weight"))
    reads.read_expr(a.q_norm, norm(n("self_attn.q_norm.weight")))
    reads.read_expr(a.k_norm, norm(n("self_attn.k_norm.weight")))

def read_gdn(reads, g, n):
    reads.read_concat(g.in_qkvz, [n("linear_attn.in_proj_qkv.weight"), n("linear_attn.in_proj_z.weight")])
    reads.read_concat(g.in_ba, [n("linear_attn.in_proj_b.weight"), n("linear_attn.in_proj_a.weight")])
    reads.read_expr(g.conv, squeezed(n("linear_attn.conv1d.weight")))
    reads.read(g.dt_bias, n("linear_attn.dt_bias"))
    reads.read(g.a_log, n("linear_attn.A_log"))
    reads.read(g.norm, n("linear_attn.norm.weight"))
    reads.read(g.out_proj, n("linear_attn.out_proj.weight"))

def read_routed(reads, f, n, layout):
    reads.read(f.router, n("mlp.gate.weight"))
    if layout.mlx:
        reads.read_concat(f.gate_up, [n("mlp.switch_mlp.gate_proj.weight"), n("mlp.switch_mlp.up_proj.weight")])
        reads.read(f.down, n("mlp.switch_mlp.down_proj.weight"))
    else:
        reads.read(f.gate_up, n("mlp.experts.gate_up_proj"))
        reads.read(f.down, n("mlp.experts.down_proj"))
    reads.read_concat(f.shared_gate_up, [n("mlp.shared_expert.gate_proj.weight"), n("mlp.shared_expert.up_proj.weight")])
    reads.read(f.shared_down, n("mlp.shared_expert.down_proj.weight"))
    reads.read(f.shared_gate, n("mlp.shared_expert_gate.weight"))
