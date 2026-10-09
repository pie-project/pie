# The Qwen3 text encoder's caches and its forward over a prompt's tokens.

def caches(te, c, kv):
    space = c.kv_space(kv)
    plane = te.kv_heads * te.head_dim
    for layer in te.layers:
        c.kv(space, layer.kv, [plane, plane], te.head_dim, heads = True)

def encode(arm, te, tap = None):
    """The encoder over `arm`'s tokens. `tap(l, y)` sees each layer's output
    rows and reads the encoder out; without one, the last layer's rows are
    its readout."""
    plan = ops.attn.plan_prefill(arm, te.q_heads, te.kv_heads, te.head_dim, None)
    ids = arm.tokens()
    positions = arm.positions()
    y = ops.layout.embed(ids, te.embed, te.vocab)
    last = len(te.layers) - 1

    def layer(l, w, y):
        pages = arm.kv(w.kv)
        x = ops.elemwise.rmsnorm(y, w.attn_norm, te.eps)
        q = ops.linear.matmul(x, w.q)
        k = ops.linear.matmul(x, w.k)
        v = ops.linear.matmul(x, w.v)
        q = ops.elemwise.rmsnorm_per_head(q, w.q_norm, te.head_dim, te.eps)
        k = ops.elemwise.rmsnorm_per_head(k, w.k_norm, te.head_dim, te.eps)
        q, k = ops.elemwise.rope_full(q, k, positions, te.head_dim, te.theta, False)
        ops.attn.kv_append(k, v, pages, arm.write_page(w.kv), arm.write_offset(w.kv))
        o = ops.attn.prefill(q, plan, pages, None, te.head_dim, te.kv_heads, te.sm_scale)
        y = ops.elemwise.residual_add(ops.linear.matmul(o, w.o), y)

        x = ops.elemwise.rmsnorm(y, w.mlp_norm, te.eps)
        f = ops.linear.matmul(ops.linear.mlp_swiglu(ops.linear.matmul(x, w.gate_up), te.inter), w.down)
        y = ops.elemwise.residual_add(f, y)
        if tap != None:
            tap(l, y)
        elif l == last:
            seam.at(seam.HIDDEN, [y])
        return y

    arm.fold_layers(te.layers, y, layer)
