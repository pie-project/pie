//! A MODEL SPEC THAT SHARDS AND QUANTIZES — the two axes together.
//!
//! Written as a test rather than prose because a spec that does not
//! lower is not a spec. Everything here compiles, traces, and goes
//! through `Buffers::assign`; the assertions at the bottom are what the
//! text is claiming.
//!
//! It is a LAYER, not a model: no attention, no epilogue. Attention
//! needs a kv handle and a prepare, and neither says anything about
//! sharding or quantization — the two axes show up in the projections
//! around it and in the landing after it, which is what this covers.
//!
//! ## What the deployment's facts decide
//!
//! Both axes resolve at TRACE time from facts a load already has:
//!
//!   tp_size, rank        -> every projection's width is this rank's
//!   quant per projection -> which kernel can read that weight
//!   fuse_max             -> the token bound the fused collective holds to
//!
//! Nothing reaches the driver as a descriptor to switch on. A width is
//! a width; a symbol is a symbol.

use model_compiler::lower::{Buffers, Fire, Row, lower};
use model_dsl::{self as dsl, MatW, NormW, ScaleLayout, WeightRepr, matmul, rmsnorm};
use model_ir::trace::{DType, Dim, GuardPred, NormVariant, OpKind, Shape};

/// One deployment's facts, as a load would derive them.
struct Facts {
    hidden: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    inter: u32,
    /// How many ranks this weight set is split across, and which one
    /// this trace is for. A rank states ITS widths.
    tp_size: u32,
    /// The checkpoint's storage for the attention and MLP projections.
    proj_repr: WeightRepr,
    /// Above this token count the fused collective declines, so the
    /// text states the two-step form there instead.
    fuse_max: u32,
}

fn awq_int4() -> WeightRepr {
    WeightRepr::Scaled {
        layout: ScaleLayout::PerGroup,
        group: 128,
        axis: 0,
        zero_point: true,
    }
}

fn spec(f: &Facts) -> model_ir::trace::ForwardPlan {
    let shard = |w: u32| w / f.tp_size;
    let hq = shard(f.q_heads * f.head_dim);
    let hk = shard(f.kv_heads * f.head_dim);
    let inter = shard(f.inter);

    dsl::trace_named("tp_quantized.cuda.decode", |t| {
        let l = 0u32;
        let norm = |name: &str| NormW {
            name: format!("layer.{l}.{name}"),
            variant: NormVariant::Plain,
            per_head: None,
            layer: Some(l),
        };
        // A projection handle carries BOTH facts: the width is this
        // rank's, the repr is the checkpoint's.
        let proj = |name: &str, width: u32| {
            MatW::dense(format!("layer.{l}.{name}"), width, Some(l)).with_repr(f.proj_repr)
        };

        let y = dsl::embed_with(t, "embed", f.hidden);
        let normed = rmsnorm(&y, &norm("attn_norm"));

        // SHARDED projections. Each rank computes its slice; the widths
        // are the only thing that says so.
        let _q = matmul(&normed, &proj("q_proj", hq));
        let _k = matmul(&normed, &proj("k_proj", hk));
        let _v = matmul(&normed, &proj("v_proj", hk));

        // (attention runs here on this rank's heads)

        // The output projection produces a PARTIAL sum — every rank has
        // a full-width `[tokens, hidden]` that is only its own
        // contribution. Recombining it is the collective below.
        let partial = matmul(&_q, &proj("o_proj", f.hidden));

        // THE LANDING, as a guard. `can_fuse_residual_rmsnorm(N, H)` in
        // the hand-written pass is three terms: buffer registration and
        // `hidden` are load-time and resolved into `fuse_max` already;
        // what is left is the token count, and that is a predicate.
        let mlp_norm = norm("mlp_norm");
        let mlp_in = dsl::regions(
            t,
            Some(l),
            Some((Shape(vec![Dim::Tokens, Dim::Const(f.hidden)]), DType::BF16)),
            |ctx| {
                ctx.arm(dsl::Region::Fire(GuardPred::TokensLE(f.fuse_max)), || {
                    // One launch: sum the shards, add the residual,
                    // norm. Two results — the stream in place, and
                    // the normed activation.
                    dsl::cuda::all_reduce_residual_rmsnorm(&partial, &y, &mlp_norm, f.hidden);
                });
            },
            || {
                // The two-step form the fused kernel declines to serve.
                let summed = dsl::cuda::all_reduce(&partial, f.hidden);
                let _ = dsl::cuda::residual_add_rmsnorm(&summed, &y, &mlp_norm.name, f.hidden);
            },
        )
        .expect("the guarded landing produces the normed activation");

        // The MLP shards the same way, and its down projection produces
        // another partial. A full text would land it with a second
        // guard; one is enough to show the shape.
        let gate_up = matmul(&mlp_in, &proj("gate_up", 2 * inter));
        let act = dsl::cuda::swiglu(&gate_up, inter);
        let _down = matmul(&act, &proj("down", f.hidden));
    })
}

fn facts() -> Facts {
    Facts {
        hidden: 4096,
        q_heads: 32,
        kv_heads: 8,
        head_dim: 128,
        inter: 14336,
        tp_size: 4,
        proj_repr: awq_int4(),
        fuse_max: 512,
    }
}

#[test]
fn a_sharded_quantized_layer_states_both_axes() {
    let f = facts();
    let plan = spec(&f);

    // ── QUANTIZATION: the statement names the kernel, and the scales
    // ── are operands rather than a struct the driver reaches for.
    let scaled: Vec<&OpKind> = plan
        .ops
        .iter()
        .map(|o| &o.kind)
        .filter(|k| {
            matches!(k, OpKind::Launch { kernel, .. }
                     if kernel == "gemm::act_x_wt_grouped_scaled")
        })
        .collect();
    assert_eq!(
        scaled.len(),
        6,
        "q, k, v, o, gate_up, down — and nothing else is a projection"
    );
    let OpKind::Launch { weights, .. } = scaled[0] else {
        unreachable!()
    };
    assert_eq!(
        weights,
        &vec![
            "layer.0.q_proj".to_string(),
            "layer.0.q_proj.scales".to_string(),
            "layer.0.q_proj.zeros".to_string(),
        ],
        "an AWQ weight names its scales and zero-points AS WEIGHTS"
    );
    assert!(
        !plan
            .ops
            .iter()
            .any(|o| matches!(&o.kind, OpKind::Matmul { .. })),
        "no semantic Matmul survives: every projection is quantized here, \
         so every one states its kernel"
    );

    // ── SHARDING: the widths are this rank's, and nothing else says so.
    let q = plan
        .ops
        .iter()
        .find(|o| {
            matches!(&o.kind, OpKind::Launch { weights, .. }
                           if weights[0] == "layer.0.q_proj")
        })
        .expect("q_proj is stated");
    let w = plan.values[q.outputs[0] as usize].shape.0.last().unwrap();
    assert_eq!(
        *w,
        Dim::Const(32 * 128 / 4),
        "tp_size 4 makes this rank's q width a quarter of the model's"
    );

    // ── THE LANDING: a guard, with the fused arm and the two-step else.
    let guard = plan
        .ops
        .iter()
        .find(|o| matches!(&o.kind, OpKind::Guard { .. }))
        .expect("the landing is a guard");
    let OpKind::Guard { arms, else_ops } = &guard.kind else {
        unreachable!()
    };
    assert_eq!(arms.len(), 1, "one fused arm");
    assert_eq!(arms[0].pred, GuardPred::TokensLE(512));
    assert_eq!(arms[0].ops, 1, "the fused landing is ONE launch");
    assert_eq!(
        *else_ops, 2,
        "the two-step form is a collective then a norm"
    );
    assert!(
        !guard.outputs.is_empty(),
        "the guard OWNS the normed activation — both arms bind it"
    );
}

#[test]
fn the_sharded_quantized_layer_lowers() {
    let f = facts();
    let plan = spec(&f);
    let rows = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        8
    ];
    let out = lower(&plan, &rows, Fire::default())
        .expect("a sharded, quantized layer lowers like any other");
    assert!(out.launches.len() >= 8);
    let buffers = Buffers::assign(&plan, &rows);
    assert!(buffers.bytes > 0, "and gets an arena");
}
