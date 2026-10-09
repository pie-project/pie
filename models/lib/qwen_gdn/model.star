# The weights the Qwen 3.5-family trunks share: gated attention (q stored
# beside its sigmoid gate), the gated delta-net (GDN) mixer, and a routed MLP
# of experts beside a sigmoid-gated shared one. `n(s)` names a layer's weight.

def gated_attn(n, d, w, kv):
    """Gated attention over `d`'s heads, its projections at `w`, caching in `kv`."""
    dense = compute(w)
    hd = d.head_dim
    return struct(
        rotary_dim = d.rotary_dim,
        theta = d.theta,
        sm_scale = f32(1.0 / f32(sqrt(hd))),
        qg_proj = weight(n("qg_proj"), [2 * d.q_heads * hd, d.hidden], w).columns(),
        k_proj = weight(n("k_proj"), [d.kv_heads * hd, d.hidden], w).columns(heads = d.kv_heads),
        v_proj = weight(n("v_proj"), [d.kv_heads * hd, d.hidden], w).columns(heads = d.kv_heads),
        o_proj = weight(n("o_proj"), [d.hidden, d.q_heads * hd], w).rows(),
        q_norm = weight(n("q_norm"), [hd], dense),
        q_norm_eps = d.norm_eps,
        k_norm = weight(n("k_norm"), [hd], dense),
        k_norm_eps = d.norm_eps,
        kv = kv,
    )

def gdn(n, d, proj, ba, dense, conv_state, delta_state):
    """A GDN mixer: its projections at `proj`, beta/alpha's at `ba`, its
    convolution and gate parameters at `dense`."""
    k_w = d.k_heads * d.k_dim
    v_w = d.v_heads * d.v_dim
    qkv = 2 * k_w + v_w
    return struct(
        k_heads = d.k_heads,
        v_heads = d.v_heads,
        k_dim = d.k_dim,
        v_dim = d.v_dim,
        qkv_width = qkv,
        conv_kernel = d.conv_kernel,
        in_qkvz = weight(n("in_qkvz"), [qkv + v_w, d.hidden], proj).packed([k_w, k_w, v_w, v_w]),
        in_ba = weight(n("in_ba"), [2 * d.v_heads, d.hidden], ba).packed([d.v_heads, d.v_heads]),
        conv = weight(n("conv"), [qkv, d.conv_kernel], dense).packed([k_w, k_w, v_w]),
        dt_bias = weight(n("dt_bias"), [d.v_heads], dense).columns(),
        a_log = weight(n("a_log"), [d.v_heads], dtype.f32).columns(),
        norm = weight(n("gdn_norm"), [d.v_dim], dtype.f32),
        norm_eps = d.norm_eps,
        out_proj = weight(n("out_proj"), [d.hidden, v_w], proj).rows(),
        conv_state = conv_state,
        delta_state = delta_state,
    )

def routed(n, hidden, m, experts, proj, gate):
    """`m.experts` experts at `experts`, `m.top_k` routed to by a router at
    `gate`, beside a shared expert at `proj` whose gate is at `gate`."""
    return struct(
        routed = True,
        router = weight(n("router"), [m.experts, hidden], gate),
        gate_up = weight(n("experts_gate_up"), [m.experts, 2 * m.inter, hidden], experts).bank([m.inter, m.inter]),
        down = weight(n("experts_down"), [m.experts, hidden, m.inter], experts).rows(),
        shared_gate_up = weight(n("shared_gate_up"), [2 * m.shared_inter, hidden], proj).packed([m.shared_inter, m.shared_inter]),
        shared_down = weight(n("shared_down"), [hidden, m.shared_inter], proj).rows(),
        shared_gate = weight(n("shared_gate"), [1, hidden], gate),
        experts = m.experts,
        top_k = m.top_k,
        inter = m.inter,
        shared_inter = m.shared_inter,
    )
