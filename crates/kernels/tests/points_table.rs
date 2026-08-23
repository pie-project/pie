use kernels::points::{
    ATTENTION_POINTS, DIST_POINTS, Dtype, Dtype::*, GATE_POINTS, GEMM_POINTS, HC_POINTS,
    INDEX_POINTS, LAYOUT_POINTS, MLA_POINTS, MLP_POINTS, MOE_POINTS, Mark, Mark::*, NORM_POINTS,
    POOL_POINTS, Point, Prim::*, ROPE_POINTS, SSM_POINTS,
};
use kernels_macros::claims;

fn families() -> impl Iterator<Item = &'static Point> {
    NORM_POINTS
        .iter()
        .chain(ROPE_POINTS)
        .chain(MLP_POINTS)
        .chain(GEMM_POINTS)
        .chain(DIST_POINTS)
        .chain(MOE_POINTS)
        .chain(GATE_POINTS)
        .chain(LAYOUT_POINTS)
        .chain(SSM_POINTS)
        .chain(ATTENTION_POINTS)
        .chain(MLA_POINTS)
        .chain(INDEX_POINTS)
        .chain(POOL_POINTS)
        .chain(HC_POINTS)
}

fn point(name: &str) -> &'static Point {
    families()
        .find(|p| p.name == name)
        .expect("the family declares it")
}

fn slots(name: &str) -> Vec<(&'static str, Mark, Dtype)> {
    point(name)
        .slots
        .iter()
        .map(|s| (s.name, s.mark, s.dtype))
        .collect()
}

fn find(table: &'static [Point], name: &str) -> &'static Point {
    table
        .iter()
        .chain(MLP_POINTS)
        .find(|p| p.name == name)
        .expect("the family declares it")
}

fn marks(p: &'static Point) -> Vec<(&'static str, Mark, Dtype)> {
    p.slots.iter().map(|s| (s.name, s.mark, s.dtype)).collect()
}

#[test]
fn the_table_is_the_norm_trait() {
    assert_eq!(
        NORM_POINTS.iter().map(|p| p.name).collect::<Vec<_>>(),
        [
            "norm.rmsnorm",
            "norm.rmsnorm_per_head",
            "norm.rmsnorm_no_scale",
            "norm.rmsnorm_gated",
            "norm.rmsnorm_gated_by",
            "norm.residual_add",
            "norm.add_bias",
            "norm.mul_scalar",
            "norm.scale",
            "norm.res_blend",
        ]
    );

    // Kimi's residual blend, filed by its arithmetic and not by the `attn/`
    // directory its cuda routine lives in: the residual stream in, the
    // residual stream out, through a norm and a projection. The blocks are
    // ONE slot because a point's arity is its slot list — the text's
    // growing `&[Value]` is the open ledger item the declaration names.
    assert_eq!(
        slots("norm.res_blend"),
        [
            ("prefix", In, Generic(0)),
            ("blocks", In, Generic(0)),
            ("weight", Const, Generic(0)),
            ("eps", Scalar, Fixed(F32)),
            ("proj", Const, Generic(0)),
            ("y", Out, Generic(0)),
        ]
    );

    assert_eq!(point("norm.rmsnorm").axes, 1);
    assert_eq!(
        slots("norm.rmsnorm"),
        [
            ("x", In, Generic(0)),
            ("weight", Const, Generic(0)),
            ("eps", Scalar, Fixed(F32)),
            ("y", Out, Generic(0)),
        ]
    );

    // The mixed one: the core arrives f32, the gate and the result ride `T`.
    assert_eq!(
        slots("norm.rmsnorm_gated"),
        [
            ("x", In, Fixed(F32)),
            ("gate", In, Generic(0)),
            ("weight", Const, Fixed(F32)),
            ("eps", Scalar, Fixed(F32)),
            ("y", Out, Generic(0)),
        ]
    );

    assert_eq!(
        slots("norm.residual_add"),
        [("x", In, Generic(0)), ("y", InOut, Generic(0))]
    );

    // A host constant and a device one, at the same slot of the same shape.
    assert_eq!(
        slots("norm.mul_scalar"),
        [("s", Scalar, Fixed(F32)), ("x", InOut, Generic(0))]
    );
    assert_eq!(
        slots("norm.scale"),
        [("s", Const, Generic(0)), ("x", InOut, Generic(0))]
    );

    // Every generic slot indexes an axis its own method quantifies over.
    for p in NORM_POINTS {
        for slot in p.slots {
            if let Generic(axis) = slot.dtype {
                assert!(axis < p.axes, "{}.{}", p.name, slot.name);
            }
        }
    }
}

#[test]
fn the_table_is_the_mlp_trait() {
    assert_eq!(
        MLP_POINTS.iter().map(|p| p.name).collect::<Vec<_>>(),
        [
            "mlp.swiglu",
            "mlp.swiglu_clamp",
            "mlp.swiglu_clamp_alpha",
            "mlp.geglu_tanh",
            "mlp.geglu_tanh_packed",
            "mlp.situ",
        ]
    );

    // The packed forms read ONE row and write a row half as wide, so the
    // intermediate width is a stated scalar rather than a derived one.
    assert_eq!(
        slots("mlp.swiglu"),
        [
            ("packed", In, Generic(0)),
            ("intermediate", Scalar, Fixed(U32)),
            ("y", Out, Generic(0)),
        ]
    );
    assert_eq!(
        slots("mlp.situ"),
        [
            ("packed", In, Generic(0)),
            ("intermediate", Scalar, Fixed(U32)),
            ("beta", Scalar, Fixed(F32)),
            ("up_cap", Scalar, Fixed(F32)),
            ("y", Out, Generic(0)),
        ]
    );

    // The one unpacked point: two operands in, and no width to state.
    assert_eq!(
        slots("mlp.geglu_tanh"),
        [
            ("gate", In, Generic(0)),
            ("up", In, Generic(0)),
            ("y", Out, Generic(0)),
        ]
    );

    for p in MLP_POINTS {
        for slot in p.slots {
            if let Generic(axis) = slot.dtype {
                assert!(axis < p.axes, "{}.{}", p.name, slot.name);
            }
        }
    }
}

#[test]
fn the_table_is_the_gemm_trait() {
    assert_eq!(
        GEMM_POINTS.iter().map(|p| p.name).collect::<Vec<_>>(),
        ["gemm.matmul", "gemm.lm_head", "gemm.attention_landing"]
    );

    // One arithmetic wearing three purposes: the same three operands, at the
    // same marks, on the one axis all three ride.
    let one = [
        ("act", In, Generic(0)),
        ("w", Const, Generic(0)),
        ("y", Out, Generic(0)),
    ];
    assert_eq!(marks(find(GEMM_POINTS, "gemm.matmul")), one);
    assert_eq!(marks(find(GEMM_POINTS, "gemm.lm_head")), one);

    // Plus the layer the landing is stated at — a bare host scalar, which is
    // what wearing no mark means, and `Out` still last.
    assert_eq!(
        marks(find(GEMM_POINTS, "gemm.attention_landing")),
        [
            ("act", In, Generic(0)),
            ("w", Const, Generic(0)),
            ("layer", Scalar, Fixed(U32)),
            ("y", Out, Generic(0)),
        ]
    );

    for p in GEMM_POINTS {
        assert_eq!(p.axes, 1, "{}", p.name);
    }
}

#[test]
fn the_table_is_the_dist_trait() {
    assert_eq!(
        DIST_POINTS.iter().map(|p| p.name).collect::<Vec<_>>(),
        ["dist.all_reduce"]
    );

    // A rank enters holding a shard of the row and leaves holding the row:
    // one operand, read and written, and no result beside it.
    assert_eq!(
        marks(find(DIST_POINTS, "dist.all_reduce")),
        [("buf", InOut, Generic(0))]
    );
}

#[test]
fn the_table_is_the_moe_trait() {
    assert_eq!(
        MOE_POINTS.iter().map(|p| p.name).collect::<Vec<_>>(),
        [
            "moe.topk_softmax",
            "moe.topk_sigmoid",
            "moe.topk_sqrt_softplus",
            "moe.matmul_select",
            "moe.matmul_select_bias",
            "moe.weighted_sum",
            "moe.sigmoid_gate_add",
        ]
    );

    // A router states TWO results: the experts a row goes to, and how much
    // each one counts. The two numbers that size them are stated, because
    // an `Out` is allocated from the statement, not read for it.
    assert_eq!(
        slots("moe.topk_softmax"),
        [
            ("logits", In, Generic(0)),
            ("experts", Scalar, Fixed(U32)),
            ("top_k", Scalar, Fixed(U32)),
            ("routes", Out, Fixed(I32)),
            ("weights", Out, Fixed(F32)),
        ]
    );

    // The one router that carries a correction bias, and the only weight in
    // the family's routing half.
    assert_eq!(
        slots("moe.topk_sqrt_softplus"),
        [
            ("logits", In, Generic(0)),
            ("bias", Const, Fixed(F32)),
            ("experts", Scalar, Fixed(U32)),
            ("top_k", Scalar, Fixed(U32)),
            ("renormalize", Scalar, Fixed(Bool)),
            ("scaling", Scalar, Fixed(F32)),
            ("routes", Out, Fixed(I32)),
            ("weights", Out, Fixed(F32)),
        ]
    );

    // The selector rides `i32` on both sides of the family: the router
    // writes it, the expert-bank matmul reads it.
    assert_eq!(
        slots("moe.matmul_select"),
        [
            ("x", In, Generic(0)),
            ("bank", Const, Generic(0)),
            ("routes", In, Fixed(I32)),
            ("y", Out, Generic(0)),
        ]
    );
    assert_eq!(
        slots("moe.weighted_sum"),
        [
            ("routed", In, Generic(0)),
            ("weights", In, Fixed(F32)),
            ("y", Out, Generic(0)),
        ]
    );
}

#[test]
fn the_table_is_the_gate_trait() {
    assert_eq!(
        GATE_POINTS.iter().map(|p| p.name).collect::<Vec<_>>(),
        ["gate.sigmoid_mul"]
    );
    assert_eq!(
        slots("gate.sigmoid_mul"),
        [("x", InOut, Generic(0)), ("gate", In, Generic(0))]
    );
}

#[test]
fn the_table_is_the_layout_trait() {
    assert_eq!(
        LAYOUT_POINTS.iter().map(|p| p.name).collect::<Vec<_>>(),
        [
            "layout.embed",
            "layout.split_qkv",
            "layout.split_q_gate",
            "layout.split_rows",
            "layout.select",
        ]
    );

    // A token id is an `i32` on every plane, so the slot is spelled rather
    // than quantified; the table and the rows it yields ride the axis. The
    // result is the ids' ROWS by the TABLE's width, and neither a `Const`
    // weight nor the first `In` carries that — the shape rule this family
    // is waiting on.
    assert_eq!(point("layout.embed").axes, 1);
    assert_eq!(
        slots("layout.embed"),
        [
            ("ids", In, Fixed(I32)),
            ("table", Const, Generic(0)),
            ("y", Out, Generic(0)),
        ]
    );

    // THREE results, and the two widths that size them: an `Out` is
    // allocated from the statement, never read for it.
    assert_eq!(
        slots("layout.split_qkv"),
        [
            ("packed", In, Generic(0)),
            ("q_width", Scalar, Fixed(U32)),
            ("kv_width", Scalar, Fixed(U32)),
            ("q", Out, Generic(0)),
            ("k", Out, Generic(0)),
            ("v", Out, Generic(0)),
        ]
    );

    // The interleaved cut states a PITCH, not a width: the halves come out
    // the same width as each other.
    assert_eq!(
        slots("layout.split_q_gate"),
        [
            ("packed", In, Generic(0)),
            ("head_dim", Scalar, Fixed(U32)),
            ("q", Out, Generic(0)),
            ("gate", Out, Generic(0)),
        ]
    );
    assert_eq!(
        slots("layout.split_rows"),
        [
            ("x", In, Generic(0)),
            ("width", Scalar, Fixed(U32)),
            ("left", Out, Generic(0)),
            ("right", Out, Generic(0)),
        ]
    );

    // The measured gap: declared so gemma's layer slice has a name, claimed
    // by nothing, because a slice of a laid-out stack is a base and an
    // offset and the plane that answers it may answer with a view.
    assert_eq!(
        slots("layout.select"),
        [
            ("table", In, Generic(0)),
            ("layer", Scalar, Fixed(U32)),
            ("y", Out, Generic(0)),
        ]
    );

}

#[test]
fn the_table_is_the_ssm_trait() {
    assert_eq!(
        SSM_POINTS.iter().map(|p| p.name).collect::<Vec<_>>(),
        [
            "ssm.causal_conv1d",
            "ssm.causal_conv1d_chunked",
            "ssm.gdn_prep",
            "ssm.gated_delta",
            "ssm.gated_delta_chunked",
            "ssm.kda_step",
            "ssm.kda_chunked",
        ]
    );

    // THE CACHE SLOT. `Mark::Cache` is the third binder — the arena binds
    // `In`/`InOut`/`Out`, the Load contract binds `Const`, and the cache
    // POOL binds this one — and its dtype column is `Opaque` because the
    // payload is a view of a slab whose element type the pool chose, not a
    // rectangle of anything this method quantifies over.
    assert_eq!(
        slots("ssm.causal_conv1d"),
        [
            ("x", In, Generic(0)),
            ("weight", Const, Generic(0)),
            ("state", Cache, Opaque),
            ("conv_width", Scalar, Fixed(U32)),
            ("y", Out, Generic(0)),
        ]
    );

    // The chunked reading takes the fire's query CSR beside its rows: an
    // ORDINARY `In` at a fixed `i32`, because a boundary buffer is a device
    // rectangle the runtime stages and nothing a pool holds.
    assert_eq!(
        slots("ssm.causal_conv1d_chunked"),
        [
            ("x", In, Generic(0)),
            ("indptr", In, Fixed(I32)),
            ("weight", Const, Generic(0)),
            ("state", Cache, Opaque),
            ("conv_width", Scalar, Fixed(U32)),
            ("y", Out, Generic(0)),
        ]
    );

    // The prologue names no cache row: it reads a projection and writes the
    // gate columns, and the recurrence beside it is what touches the slab.
    assert_eq!(
        slots("ssm.gdn_prep"),
        [
            ("ba", In, Generic(0)),
            ("dt_bias", Const, Generic(0)),
            ("a_log", Const, Generic(0)),
            ("gates", Out, Fixed(F32)),
        ]
    );

    // The mixed one: the packed rows ride `T`, the decay columns and the
    // result ride f32, and the four head numbers are stated because a GQA
    // rule cannot read them off a packed row.
    assert_eq!(
        slots("ssm.gated_delta"),
        [
            ("qkv", In, Generic(0)),
            ("z", In, Generic(0)),
            ("gates", In, Fixed(F32)),
            ("state", Cache, Opaque),
            ("k_heads", Scalar, Fixed(U32)),
            ("v_heads", Scalar, Fixed(U32)),
            ("k_dim", Scalar, Fixed(U32)),
            ("v_dim", Scalar, Fixed(U32)),
            ("y", Out, Fixed(F32)),
        ]
    );

    assert_eq!(
        slots("ssm.kda_step"),
        [
            ("mixed", In, Generic(0)),
            ("f", In, Generic(0)),
            ("b", In, Generic(0)),
            ("dt_bias", Const, Generic(0)),
            ("a_log", Const, Generic(0)),
            ("state", Cache, Opaque),
            ("heads", Scalar, Fixed(U32)),
            ("head_dim", Scalar, Fixed(U32)),
            ("norm_eps", Scalar, Fixed(F32)),
            ("y", Out, Fixed(F32)),
        ]
    );

    // Every chunked point is its plain sibling plus the CSR at slot one,
    // and nothing else moves.
    for (plain, chunked) in [
        ("ssm.causal_conv1d", "ssm.causal_conv1d_chunked"),
        ("ssm.gated_delta", "ssm.gated_delta_chunked"),
        ("ssm.kda_step", "ssm.kda_chunked"),
    ] {
        let mut expected = slots(plain);
        expected.insert(1, ("indptr", In, Fixed(I32)));
        assert_eq!(slots(chunked), expected, "{chunked}");
    }

    // A cache slot is exactly one per point, and never the first: the
    // receiver of a statement is a rectangle, and a pool row is named
    // beside it.
    for p in SSM_POINTS {
        let cached = p.slots.iter().filter(|s| s.mark == Cache).count();
        assert!(cached <= 1, "{}", p.name);
        assert_ne!(p.slots[0].mark, Cache, "{}", p.name);
        for slot in p.slots {
            if let Generic(axis) = slot.dtype {
                assert!(axis < p.axes, "{}.{}", p.name, slot.name);
            }
            assert_eq!(slot.dtype == Opaque, slot.mark == Cache, "{}", p.name);
        }
    }
}

#[test]
fn the_table_is_the_attention_trait() {
    assert_eq!(
        ATTENTION_POINTS.iter().map(|p| p.name).collect::<Vec<_>>(),
        [
            "attention.decode",
            "attention.prefill",
            "attention.masked",
            "attention.decode_lse",
            "attention.prefill_lse",
            "attention.sink",
            "attention.merge_lse",
            "attention.lse_ln",
            "attention.logit_softcap",
            "attention.kv_append",
            "attention.kv_append_shared",
        ]
    );

    // The one-token reading: the page row is a `Cache` beside the query,
    // and the three numbers are stated because a packed row cannot be
    // divided by reading it. `window` is a plain `u32` — the DSL's
    // `.window()` has always recorded `int(w.unwrap_or(0))`, so ZERO IS NO
    // WINDOW and there is no `Option` on the params run to declare.
    assert_eq!(
        slots("attention.decode"),
        [
            ("q", In, Generic(0)),
            ("pages", Cache, Opaque),
            ("window", Scalar, Fixed(U32)),
            ("head_dim", Scalar, Fixed(U32)),
            ("sm_scale", Scalar, Fixed(F32)),
            ("o", Out, Generic(0)),
        ]
    );

    // The window reading takes the fire's query CSR beside its rows: an
    // ORDINARY `In` at a fixed `i32`, the `ssm.*_chunked` precedent, plus
    // the GQA key-head count no rectangle carries.
    assert_eq!(
        slots("attention.prefill"),
        [
            ("q", In, Generic(0)),
            ("indptr", In, Fixed(I32)),
            ("pages", Cache, Opaque),
            ("window", Scalar, Fixed(U32)),
            ("head_dim", Scalar, Fixed(U32)),
            ("kv_heads", Scalar, Fixed(U32)),
            ("sm_scale", Scalar, Fixed(F32)),
            ("o", Out, Generic(0)),
        ]
    );

    // An `_lse` reading is its plain sibling plus ONE `Out` at f32, and
    // nothing else moves: a second result is two `Out` slots, never a flag.
    for (plain, with_lse) in [
        ("attention.decode", "attention.decode_lse"),
        ("attention.prefill", "attention.prefill_lse"),
    ] {
        let mut expected = slots(plain);
        expected.push(("lse", Out, Fixed(F32)));
        assert_eq!(slots(with_lse), expected, "{with_lse}");
    }

    // The mixed one: the output rides `T`, the lse and the learned sink
    // ride f32 — a normaliser is accumulated, not activated.
    assert_eq!(
        slots("attention.sink"),
        [
            ("o", InOut, Generic(0)),
            ("lse", In, Fixed(F32)),
            ("sink", Const, Fixed(F32)),
            ("head_dim", Scalar, Fixed(U32)),
        ]
    );

    assert_eq!(
        slots("attention.merge_lse"),
        [
            ("o1", In, Generic(0)),
            ("lse1", In, Fixed(F32)),
            ("o2", In, Generic(0)),
            ("lse2", In, Fixed(F32)),
            ("heads", Scalar, Fixed(U32)),
            ("head_dim", Scalar, Fixed(U32)),
            ("o", Out, Generic(0)),
            ("lse", Out, Fixed(F32)),
        ]
    );

    // The only point on the floor that quantifies over nothing: an lse is
    // f32 wherever it came from, so its dispatch is a single arm.
    assert_eq!(point("attention.lse_ln").axes, 0);
    assert_eq!(slots("attention.lse_ln"), [("lse", InOut, Fixed(F32))]);

    // The two appends are EFFECTS: a statement that names a cache row and
    // leaves the fire's rows in it states no result, and no `Out` slot is
    // what that looks like. The shared form takes ONE plane where the
    // ordinary one takes a key and a value.
    assert_eq!(
        slots("attention.kv_append"),
        [
            ("k", In, Generic(0)),
            ("v", In, Generic(0)),
            ("pages", Cache, Opaque),
        ]
    );
    assert_eq!(
        slots("attention.kv_append_shared"),
        [("plane", In, Generic(0)), ("pages", Cache, Opaque)]
    );
    for p in ATTENTION_POINTS {
        if p.name.starts_with("attention.kv_append") {
            assert!(p.slots.iter().all(|s| s.mark != Out), "{}", p.name);
        }
    }

    // A cache slot is exactly one per point, and never the first: the
    // receiver of a statement is a rectangle, and a pool row is named
    // beside it.
    for p in ATTENTION_POINTS {
        let cached = p.slots.iter().filter(|s| s.mark == Cache).count();
        assert!(cached <= 1, "{}", p.name);
        assert_ne!(p.slots[0].mark, Cache, "{}", p.name);
        for slot in p.slots {
            if let Generic(axis) = slot.dtype {
                assert!(axis < p.axes, "{}.{}", p.name, slot.name);
            }
            assert_eq!(slot.dtype == Opaque, slot.mark == Cache, "{}", p.name);
        }
    }
}

#[test]
fn the_cache_mark_is_the_pooled_families_alone() {
    for p in families() {
        let pooled = ["ssm.", "attention.", "mla.", "index.", "pool."]
            .iter()
            .any(|f| p.name.starts_with(f));
        for slot in p.slots {
            if pooled {
                assert_eq!(slot.dtype == Opaque, slot.mark == Cache, "{}", p.name);
                continue;
            }
            assert_ne!(slot.mark, Cache, "{}.{}", p.name, slot.name);
            assert_ne!(slot.dtype, Opaque, "{}.{}", p.name, slot.name);
        }
    }
}

/// THE CACHE OWNERSHIP LAW, spelled as a table row. A cache write belongs to
/// the family that owns the cache, so every append here is `<family>.
/// kv_append` and the old `kv_append.<family>` prefix names nothing. The
/// three appends carry the pool they fill and nothing else geometric.
#[test]
fn an_append_belongs_to_the_family_that_owns_the_pool() {
    for (name, before) in [
        ("mla.kv_append", 2),
        ("index.kv_append", 1),
        ("pool.kv_append", 3),
    ] {
        let p = point(name);
        assert_eq!(p.axes, 1, "{name}");
        // The rows to write, then the pool row that takes them. No `Out`:
        // an append is an effect, and the statement records none.
        assert_eq!(p.slots.len(), before + 1, "{name}");
        assert_eq!(p.slots[before].mark, Cache, "{name}");
        assert_eq!(p.slots[before].dtype, Opaque, "{name}");
        for slot in &p.slots[..before] {
            assert_eq!(slot.mark, In, "{name}");
        }
        assert!(
            p.slots.iter().all(|s| s.mark != Out),
            "{name} states a result"
        );
    }
}

#[test]
fn the_table_is_the_mla_trait() {
    assert_eq!(
        MLA_POINTS.iter().map(|p| p.name).collect::<Vec<_>>(),
        [
            "mla.latents",
            "mla.latents_rope",
            "mla.split_q_b",
            "mla.absorb_q",
            "mla.absorb_q_pe",
            "mla.absorb_out",
            "mla.kv_append",
            "mla.attention_decode",
            "mla.attention_prefill",
            "mla.attention_decode_selected",
            "mla.attention_prefill_selected",
        ]
    );

    assert_eq!(
        slots("mla.latents"),
        [
            ("kv_a", In, Generic(0)),
            ("weight", Const, Generic(0)),
            ("eps", Scalar, Fixed(F32)),
            ("kv_lora_rank", Scalar, Fixed(U32)),
            ("kv_c", Out, Generic(0)),
            ("k_pe", Out, Generic(0)),
        ]
    );

    // The roped reading is the plain one plus the rotation's own three:
    // the positions at slot one, and `rope_dim`/`theta` after the cut's
    // width. Nothing else moves.
    assert_eq!(
        slots("mla.latents_rope"),
        [
            ("kv_a", In, Generic(0)),
            ("positions", In, Fixed(I32)),
            ("weight", Const, Generic(0)),
            ("eps", Scalar, Fixed(F32)),
            ("kv_lora_rank", Scalar, Fixed(U32)),
            ("rope_dim", Scalar, Fixed(U32)),
            ("theta", Scalar, Fixed(F32)),
            ("kv_c", Out, Generic(0)),
            ("k_pe", Out, Generic(0)),
        ]
    );

    // EACH ABSORB STATES BOTH HALVES OF THE BANK. The pitch a batched gemm
    // walks between heads is `(nope_dim + v_head_dim) * kv_lora_rank`, and a
    // `Const` weight carries an address with no rectangle behind it, so the
    // half the arithmetic never multiplies by is stated anyway.
    assert_eq!(
        slots("mla.absorb_q"),
        [
            ("q_nope", In, Generic(0)),
            ("kv_b", Const, Generic(0)),
            ("heads", Scalar, Fixed(U32)),
            ("kv_lora_rank", Scalar, Fixed(U32)),
            ("nope_dim", Scalar, Fixed(U32)),
            ("v_head_dim", Scalar, Fixed(U32)),
            ("q_latent", Out, Generic(0)),
        ]
    );
    for name in ["mla.absorb_q", "mla.absorb_q_pe", "mla.absorb_out"] {
        let stated: Vec<&str> = point(name)
            .slots
            .iter()
            .filter(|s| s.mark == Scalar)
            .map(|s| s.name)
            .collect();
        assert_eq!(stated.len(), 4, "{name}");
        for width in ["heads", "kv_lora_rank", "nope_dim", "v_head_dim"] {
            assert!(stated.contains(&width), "{name} does not state {width}");
        }
    }

    // The selected pair reads a BYTE MASK where the plain pair reads the
    // query's rotated half: a selection is not an activation and never
    // rides the family's axis.
    assert_eq!(
        slots("mla.attention_decode"),
        [
            ("q", In, Generic(0)),
            ("q_pe", In, Generic(0)),
            ("pages", Cache, Opaque),
            ("heads", Scalar, Fixed(U32)),
            ("kv_lora_rank", Scalar, Fixed(U32)),
            ("sm_scale", Scalar, Fixed(F32)),
            ("o", Out, Generic(0)),
        ]
    );
    assert_eq!(
        slots("mla.attention_decode_selected"),
        [
            ("q", In, Generic(0)),
            ("selection", In, Fixed(U8)),
            ("pages", Cache, Opaque),
            ("heads", Scalar, Fixed(U32)),
            ("kv_lora_rank", Scalar, Fixed(U32)),
            ("sm_scale", Scalar, Fixed(F32)),
            ("o", Out, Generic(0)),
        ]
    );

    // Every prefill point is its decode sibling plus the query CSR at slot
    // one, and nothing else moves — the `ssm` chunked relation, again.
    for (decode, prefill) in [
        ("mla.attention_decode", "mla.attention_prefill"),
        (
            "mla.attention_decode_selected",
            "mla.attention_prefill_selected",
        ),
    ] {
        let mut expected = slots(decode);
        expected.insert(1, ("indptr", In, Fixed(I32)));
        assert_eq!(slots(prefill), expected, "{prefill}");
    }

    for p in MLA_POINTS {
        assert!(p.slots.iter().filter(|s| s.mark == Cache).count() <= 1);
        for slot in p.slots {
            if let Generic(axis) = slot.dtype {
                assert!(axis < p.axes, "{}.{}", p.name, slot.name);
            }
        }
    }
}

#[test]
fn the_table_is_the_index_trait() {
    assert_eq!(
        INDEX_POINTS.iter().map(|p| p.name).collect::<Vec<_>>(),
        [
            "index.layernorm_rope",
            "index.rope",
            "index.topk",
            "index.kv_append",
        ]
    );

    // BOTH ROTATIONS ARE IN PLACE, which is what `InOut` says and what
    // leaves them with no `Out` at all — the `rope` family's reading. The
    // layernorm is the one norm in the tree with a learned BIAS beside its
    // weight, and both `Const`s ride the activation dtype.
    assert_eq!(
        slots("index.layernorm_rope"),
        [
            ("k", InOut, Generic(0)),
            ("positions", In, Fixed(I32)),
            ("weight", Const, Generic(0)),
            ("bias", Const, Generic(0)),
            ("eps", Scalar, Fixed(F32)),
            ("rope_dim", Scalar, Fixed(U32)),
            ("theta", Scalar, Fixed(F32)),
        ]
    );
    assert_eq!(
        slots("index.rope"),
        [
            ("q", InOut, Generic(0)),
            ("positions", In, Fixed(I32)),
            ("heads", Scalar, Fixed(U32)),
            ("head_dim", Scalar, Fixed(U32)),
            ("rope_dim", Scalar, Fixed(U32)),
            ("theta", Scalar, Fixed(F32)),
        ]
    );

    // THE SELECTION IS A BYTE PLANE, and the same one `mla`'s selected
    // attention reads: one family writes it, the other consumes it, and the
    // two slots agree at the table rather than by convention.
    assert_eq!(
        slots("index.topk"),
        [
            ("q", In, Generic(0)),
            ("weights", In, Generic(0)),
            ("keys", Cache, Opaque),
            ("heads", Scalar, Fixed(U32)),
            ("head_dim", Scalar, Fixed(U32)),
            ("top_k", Scalar, Fixed(U32)),
            ("selection", Out, Fixed(U8)),
        ]
    );
    let written = slots("index.topk").pop().expect("the point states one");
    let read = slots("mla.attention_decode_selected")[1];
    assert_eq!(written.2, read.2);
}

#[test]
fn the_table_is_the_pool_trait() {
    assert_eq!(
        POOL_POINTS.iter().map(|p| p.name).collect::<Vec<_>>(),
        [
            "pool.boundary_decode",
            "pool.boundary_prefill",
            "pool.gather",
            "pool.kv_append",
            "pool.attention_lse",
        ]
    );

    // THE ONE FAMILY WITH A POINT OVER NO ELEMENT AT ALL. A boundary walk
    // reads positions and writes positions, so nothing about it quantifies
    // over an activation dtype and both boundary points declare ZERO axes.
    assert_eq!(point("pool.boundary_decode").axes, 0);
    assert_eq!(point("pool.boundary_prefill").axes, 0);
    assert_eq!(
        slots("pool.boundary_decode"),
        [
            ("positions", In, Fixed(I32)),
            ("ratio", Scalar, Fixed(U32)),
            ("boundary_pos", Out, Fixed(I32)),
            ("boundary_req", Out, Fixed(I32)),
        ]
    );
    // The prefill reading is the decode one plus the query CSR at slot one.
    let mut expected = slots("pool.boundary_decode");
    expected.insert(1, ("indptr", In, Fixed(I32)));
    assert_eq!(slots("pool.boundary_prefill"), expected);

    assert_eq!(
        slots("pool.gather"),
        [
            ("boundary_pos", In, Fixed(I32)),
            ("boundary_req", In, Fixed(I32)),
            ("pages", Cache, Opaque),
            ("head_dim", Scalar, Fixed(U32)),
            ("ratio", Scalar, Fixed(U32)),
            ("entries", Out, Generic(0)),
        ]
    );

    // The lse rides f32 beside an output on the family's axis: a
    // log-sum-exp is accumulated, not activated, which is what makes the
    // merge with the full-resolution attention exact.
    assert_eq!(
        slots("pool.attention_lse"),
        [
            ("q", In, Generic(0)),
            ("positions", In, Fixed(I32)),
            ("entries", Cache, Opaque),
            ("ratio", Scalar, Fixed(U32)),
            ("heads", Scalar, Fixed(U32)),
            ("head_dim", Scalar, Fixed(U32)),
            ("sm_scale", Scalar, Fixed(F32)),
            ("o", Out, Generic(0)),
            ("lse", Out, Fixed(F32)),
        ]
    );

    // Every point of this family states the pooling ratio, which lives in
    // the checkpoint and in no operand.
    for p in POOL_POINTS {
        if p.name == "pool.kv_append" {
            continue;
        }
        assert!(
            p.slots.iter().any(|s| s.name == "ratio"),
            "{} states no ratio",
            p.name
        );
    }
}

#[test]
fn the_table_is_the_hc_trait() {
    assert_eq!(
        HC_POINTS.iter().map(|p| p.name).collect::<Vec<_>>(),
        [
            "hc.expand",
            "hc.rmsnorm_f32",
            "hc.gates",
            "hc.fold",
            "hc.collapse",
        ]
    );

    assert_eq!(
        slots("hc.expand"),
        [
            ("x", In, Generic(0)),
            ("streams", Scalar, Fixed(U32)),
            ("y", Out, Generic(0)),
        ]
    );

    // THE MIXER RIDES f32 AND THE STREAMS RIDE `T`. A mix matrix whose rows
    // and columns are driven to sum to one would be measurably
    // un-stochastic at the activation dtype, so every mix slot here — the
    // normed input, the two gate weights, the two results — is fixed at
    // f32, and only the stack and the block's own row quantify over the
    // family's axis.
    assert_eq!(
        slots("hc.gates"),
        [
            ("normed", In, Fixed(F32)),
            ("streams", In, Generic(0)),
            ("scale", Const, Fixed(F32)),
            ("base", Const, Fixed(F32)),
            ("stream_count", Scalar, Fixed(U32)),
            ("gate_eps", Scalar, Fixed(F32)),
            ("alpha", Scalar, Fixed(F32)),
            ("sinkhorn", Scalar, Fixed(U32)),
            ("x", Out, Generic(0)),
            ("post_mix", Out, Fixed(F32)),
            ("comb_mix", Out, Fixed(F32)),
        ]
    );

    // THREE RESULTS, AND THE FIRST IS THE ONE THE BLOCK CONSUMES. `x` leads
    // because that is the order every text reads the triple in; the two
    // mixes follow, and `fold` takes them back in the same order.
    assert_eq!(
        slots("hc.gates")
            .into_iter()
            .filter(|s| s.1 == Out)
            .map(|s| s.0)
            .collect::<Vec<_>>(),
        ["x", "post_mix", "comb_mix"]
    );
    assert_eq!(
        slots("hc.fold"),
        [
            ("x", In, Generic(0)),
            ("streams", In, Generic(0)),
            ("post_mix", In, Fixed(F32)),
            ("comb_mix", In, Fixed(F32)),
            ("y", Out, Generic(0)),
        ]
    );

    // The collapse is the gated sum with no Sinkhorn behind it: the same
    // two f32 gate weights, no `alpha` and no iteration count.
    assert_eq!(
        slots("hc.collapse"),
        [
            ("streams", In, Generic(0)),
            ("head_scale", Const, Fixed(F32)),
            ("head_base", Const, Fixed(F32)),
            ("stream_count", Scalar, Fixed(U32)),
            ("gate_eps", Scalar, Fixed(F32)),
            ("y", Out, Generic(0)),
        ]
    );

    // No point of this family names a pool: hyper-connections are the
    // residual's shape and nothing the driver keeps across fires.
    for p in HC_POINTS {
        for slot in p.slots {
            assert_ne!(slot.mark, Cache, "{}.{}", p.name, slot.name);
        }
    }
}

/// `Prim::U8` is a TENSOR element and never a host scalar's run: the byte
/// mask a selection writes is the only slot that carries it, and no
/// parameter anywhere is a bare `u8`.
#[test]
fn the_byte_plane_is_a_rectangle_and_not_a_run() {
    let mut carried = Vec::new();
    for p in families() {
        for slot in p.slots {
            if slot.dtype == Fixed(U8) {
                assert_ne!(slot.mark, Scalar, "{}.{}", p.name, slot.name);
                carried.push(format!("{}.{}", p.name, slot.name));
            }
        }
    }
    assert_eq!(
        carried,
        [
            "mla.attention_decode_selected.selection",
            "mla.attention_prefill_selected.selection",
            "index.topk.selection",
        ]
    );
}

trait Toy {
    fn kept(&self) -> u32 {
        0
    }

    fn given(&self) -> u32 {
        0
    }
}

struct Plate;

#[claims]
impl Toy for Plate {
    fn given(&self) -> u32 {
        1
    }
}

#[test]
fn a_claim_is_what_the_impl_overrides() {
    assert_eq!(*TOY_CLAIMS, ["toy.given"]);
    assert_eq!(Plate.given(), 1);
    assert_eq!(Plate.kept(), 0);
}
