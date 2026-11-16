//! Routed experts lowered into the units' existing steps (GENERIC.md G1): no expert kernel.
//!
//! A layer's `moe_route` → `moe_select` (gate | up) → `glu` → `moe_select` (down) →
//! `moe_combine` is the gated sum of every expert, the unpicked ones gated by zero:
//!
//! - the router's logits through a root region: the scores as the route states them (its
//!   softmax or sigmoid, a pick bias), the top `k` a round at a time (the largest key left, the
//!   lowest expert holding it on a tie, as a route picks), the picked scores weighted (the
//!   softmax over them, or the sigmoid's renormalized and scaled): one weight an expert, zero
//!   unpicked;
//! - that gate expanded to each expert's columns by a matvec of constant ones;
//! - every expert's gate and up as one matvec each over its bank (an expert's rows a block);
//! - the activation times the expanded gate, a root region;
//! - every expert's down as one matvec over its bank (an expert's columns a block of the input).
//!
//! The sum over the picked experts is exact (an unpicked expert adds zeros); its cost is a dense
//! layer's of the experts' width spread over their PEs (skipping a zero gate's PEs is a speed
//! lever, not a numerics one).

use poem_ir::{Dtype, Linear, Operation, RouteScore, Trace, ValueId};

use kernels_ir::build::Ops;
use kernels_ir::{Binary, Op, Reduce, Region, Shape, Unary};
use poem_compiler::fuse::chain::{Chain, In, Out};

type R<T> = Result<T, String>;

/// A value a routed layer's lowering makes that no trace node does: this bit, its route's node
/// above [`KINDS`], its kind below.
pub const SYNTH: u32 = 1 << 29;
const KINDS: u32 = 8;

/// What a synthetic value is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The gate: an expert's weight, zero where unpicked (E, f32).
    Gate = 0,
    /// The gate, each expert's repeated over its intermediate columns (E x I, f32).
    Spread = 1,
    /// Every expert's gate projection (E x I, the bank's 16 bits).
    Gates = 2,
    /// Every expert's up projection (E x I).
    Ups = 3,
    /// The activation, gated (E x I).
    Acts = 4,
    /// The layer's input moved to the side the gate is on (H).
    Input = 5,
}

/// The synthetic value of `kind` of the route at `node`.
#[must_use]
pub fn synth(node: u32, kind: Kind) -> ValueId {
    ValueId(SYNTH | (node * KINDS) | kind as u32)
}

/// A routed layer's nodes and widths.
#[derive(Clone, Debug)]
pub struct Group {
    pub route: u32,
    /// Every node the lowering stands for (the route, the selects, the activation, the
    /// combine).
    pub nodes: Vec<u32>,
    pub logits: ValueId,
    pub bias: Option<ValueId>,
    /// A per-expert gain on the picked weights.
    pub scale: Option<ValueId>,
    pub experts: u32,
    pub top_k: u32,
    pub score: RouteScore,
    /// The layer's input (the selects' first `x`) and its width.
    pub x: ValueId,
    pub hidden: u32,
    /// The gate's and the up's bank, each expert's rows `stride` apart from `off` (a gate | up
    /// bank's halves: stride `2 I`), and the down bank `[E, H, I]`.
    pub gate: (ValueId, u32, u32),
    pub up: (ValueId, u32, u32),
    pub down: ValueId,
    pub inter: u32,
    pub silu: bool,
    /// The combine's output.
    pub y: ValueId,
    pub dtype: Dtype,
}

fn width(trace: &Trace, v: ValueId) -> u32 {
    match trace.values.get(v.0 as usize).map(|d| &d.ty) {
        Some(poem_ir::Ty::Tensor { shape, .. }) => match shape.last() {
            Some(poem_ir::Dim::Const(n)) => *n as u32,
            _ => 0,
        },
        _ => 0,
    }
}

fn dtype(trace: &Trace, v: ValueId) -> Dtype {
    match trace.values.get(v.0 as usize).map(|d| &d.ty) {
        Some(poem_ir::Ty::Tensor { dtype, .. }) => *dtype,
        _ => Dtype::F32,
    }
}

/// The routed layer the route at `node` starts, read off the nodes after it in `nodes`;
/// refused by what it states that the lowering does not take.
pub fn group(trace: &Trace, node: u32, nodes: &[u32]) -> R<Group> {
    let refuse = |what: &str| format!("node {node}: linear.moe_route {what}");
    let Operation::Linear(Linear::MoeRoute {
        ids,
        table,
        logits,
        bias,
        scale,
        hint,
        experts,
        top_k,
        score,
        routes,
        weights,
    }) = &trace.nodes[node as usize].op
    else {
        return Err(refuse("is no route"));
    };
    if ids.is_some() || table.is_some() || hint.is_some() {
        return Err(refuse("by ids, a table or a hint"));
    }
    match score {
        RouteScore::Softmax if bias.is_none() => {}
        RouteScore::Sigmoid { .. } => {}
        s => return Err(refuse(&format!("scored {s:?}"))),
    }
    let after = |pick: &dyn Fn(&Operation) -> bool| {
        nodes
            .iter()
            .copied()
            .find(|&n| n > node && pick(&trace.nodes[n as usize].op))
    };
    let plain = |op: &Operation, r: ValueId| {
        matches!(op, Operation::Linear(Linear::MoeSelect {
            routes, bias: None, input_scales: None, quantized: false, quant: None, ..
        }) if *routes == r)
    };
    let s1 = after(&|op| plain(op, *routes)).ok_or_else(|| refuse("selects no plain bank"))?;
    let select = |n: u32| match &trace.nodes[n as usize].op {
        Operation::Linear(Linear::MoeSelect { x, bank, y, .. }) => (*x, *bank, *y),
        _ => unreachable!(),
    };
    let (x, b1, y1) = select(s1);
    // the experts' activation over their gate | up rows (one bank, its halves), or over a gate
    // and an up selected apart (a bank each)
    let glu = after(&|op| {
        matches!(op, Operation::Linear(Linear::Glu { packed, gate, up, weights: None,
            limit: None, .. })
            if *packed == Some(y1) || (*gate == Some(y1) && up.is_some()))
    })
    .ok_or_else(|| refuse("activates its experts by no plain gated unit"))?;
    let Operation::Linear(Linear::Glu {
        packed,
        up,
        intermediate,
        act,
        y: a,
        ..
    }) = &trace.nodes[glu as usize].op
    else {
        unreachable!()
    };
    let i = *intermediate;
    let (gate_half, up_half, s_up) = if packed.is_some() {
        ((b1, 2 * i, 0), (b1, 2 * i, i), None)
    } else {
        let u = up.ok_or_else(|| refuse("activates no up"))?;
        let su = after(&|op| {
            plain(op, *routes)
                && matches!(op, Operation::Linear(Linear::MoeSelect { x: sx, y, .. })
                    if *y == u && *sx == x)
        })
        .ok_or_else(|| refuse("selects its up apart from its input"))?;
        let (_, b2, _) = select(su);
        ((b1, i, 0), (b2, i, 0), Some(su))
    };
    let silu = match act {
        poem_ir::GluAct::Swiglu => true,
        poem_ir::GluAct::GegluTanh => false,
        a => return Err(refuse(&format!("activates its experts by {a:?}"))),
    };
    let s2 = after(&|op| {
        plain(op, *routes) && matches!(op, Operation::Linear(Linear::MoeSelect { x, .. }) if x == a)
    })
    .ok_or_else(|| refuse("projects its experts down by no plain bank"))?;
    let Operation::Linear(Linear::MoeSelect {
        bank: down, y: dn, ..
    }) = &trace.nodes[s2 as usize].op
    else {
        unreachable!()
    };
    let comb = after(&|op| {
        matches!(op, Operation::Linear(Linear::MoeCombine { routed, weights: Some(w),
            shared: None, .. }) if routed == dn && w == weights)
    })
    .ok_or_else(|| refuse("combines its experts other than by its weights alone"))?;
    let Operation::Linear(Linear::MoeCombine { y, .. }) = &trace.nodes[comb as usize].op else {
        unreachable!()
    };
    let mut nodes = vec![node, s1];
    nodes.extend(s_up);
    nodes.extend([glu, s2, comb]);
    Ok(Group {
        route: node,
        nodes,
        logits: *logits,
        bias: *bias,
        scale: *scale,
        experts: *experts,
        top_k: *top_k,
        score: *score,
        x,
        hidden: width(trace, x),
        gate: gate_half,
        up: up_half,
        down: *down,
        inter: i,
        silu,
        y: *y,
        dtype: dtype(trace, *y),
    })
}

/// A synthetic value's width and dtype.
#[must_use]
pub fn ty(trace: &Trace, v: ValueId) -> Option<(u32, Dtype)> {
    if v.0 & SYNTH == 0 {
        return None;
    }
    let raw = v.0 & !SYNTH;
    let (node, kind) = (raw / KINDS, raw % KINDS);
    // (a Mamba-2's own: super::mamba)
    if let Some(t) = super::mamba::ty(trace, node, kind) {
        return Some(t);
    }
    let nodes: Vec<u32> = (0..trace.nodes.len() as u32).collect();
    let g = group(trace, node, &nodes).ok()?;
    let ei = g.experts * g.inter;
    let narrow = dtype(trace, g.gate.0);
    Some(match kind {
        0 => (g.experts, Dtype::F32),
        1 => (ei, Dtype::F32),
        2..=4 => (
            ei,
            if narrow == Dtype::F16 {
                Dtype::F16
            } else {
                Dtype::Bf16
            },
        ),
        _ => (g.hidden, dtype(trace, g.x)),
    })
}

fn input(value: ValueId, dtype: Dtype, shared: bool) -> In {
    In {
        value,
        dtype,
        shared,
        one: false,
        each: false,
        by: None,
        from: 0,
    }
}

/// The router's region: the logits (and the pick bias) in, the gate out.
pub fn router(trace: &Trace, g: &Group) -> R<Chain> {
    let mut b = Ops::new(1 + u32::from(g.bias.is_some()) + u32::from(g.scale.is_some()));
    let gain = 1 + u32::from(g.bias.is_some());
    let logits = 0;
    let sigmoid = |b: &mut Ops, t: u32| {
        let one = b.f32(1.0);
        let neg = b.push(Op::Unary(Unary::Neg, t));
        let e = b.push(Op::Unary(Unary::Exp, neg));
        let den = b.push(Op::Binary(Binary::Add, one, e));
        b.push(Op::Unary(Unary::Recip, den))
    };
    let s = match g.score {
        RouteScore::Sigmoid { .. } => sigmoid(&mut b, logits),
        _ => logits,
    };
    let key = match g.bias {
        Some(_) => b.push(Op::Binary(Binary::Add, s, 1)),
        None => s,
    };
    // the top k keys, one a round: the largest, the lowest expert holding it on a tie (as a
    // route picks), that one out of the next rounds
    let top = b.push(Op::Reduce(Reduce::Max, logits));
    let col = b.push(Op::Column);
    let col = b.push(Op::Cast(col, kernels_ir::Elem::F32));
    let (lowest, past) = (b.f32(-3.0e38), b.f32(3.0e38));
    let (one, zero) = (b.f32(1.0), b.f32(0.0));
    let (mut left, mut mask) = (key, zero);
    for _ in 0..g.top_k {
        let m = b.push(Op::Reduce(Reduce::Max, left));
        let at = b.push(Op::Binary(Binary::Eq, left, m));
        let cand = b.push(Op::Select(at, col, past));
        let first = b.push(Op::Reduce(Reduce::Min, cand));
        let hit = b.push(Op::Binary(Binary::Eq, col, first));
        mask = b.push(Op::Select(hit, one, mask));
        left = b.push(Op::Select(hit, lowest, left));
    }
    let gate = match g.score {
        RouteScore::Sigmoid {
            renormalize,
            scaling,
        } => {
            let w = b.push(Op::Binary(Binary::Mul, mask, s));
            let w = if renormalize {
                let sum = b.push(Op::Reduce(Reduce::Sum, w));
                b.push(Op::Binary(Binary::Div, w, sum))
            } else {
                w
            };
            if scaling == 1.0 {
                w
            } else {
                let k = b.f32(scaling);
                b.push(Op::Binary(Binary::Mul, w, k))
            }
        }
        // the softmax over the picked logits (the largest the reference)
        _ => {
            let d = b.push(Op::Binary(Binary::Sub, logits, top));
            let e = b.push(Op::Unary(Unary::Exp, d));
            let w = b.push(Op::Binary(Binary::Mul, mask, e));
            let sum = b.push(Op::Reduce(Reduce::Sum, w));
            b.push(Op::Binary(Binary::Div, w, sum))
        }
    };
    // the per-expert gain on the picked weights
    let gate = match g.scale {
        Some(_) => b.push(Op::Binary(Binary::Mul, gate, gain)),
        None => gate,
    };
    let mut ins = vec![input(g.logits, dtype(trace, g.logits), false)];
    let mut params = vec![Shape::Row];
    for w in [g.bias, g.scale].into_iter().flatten() {
        ins.push(input(w, dtype(trace, w), true));
        params.push(Shape::Row);
    }
    let region = Region {
        width: g.experts,
        params,
        ops: b.ops,
        results: vec![gate],
        rounds: Vec::new(),
    };
    region
        .types()
        .map_err(|e| format!("node {}: the router's region: {e:?}", g.route))?;
    Ok(Chain {
        nodes: vec![g.route],
        ins,
        outs: vec![Out {
            value: synth(g.route, Kind::Gate),
            dtype: Dtype::F32,
        }],
        region,
    })
}

/// The experts' activation, gated: `act(gates) · ups · spread`, rounded as the activation's
/// output is, then as the down projection's input.
pub fn activation(g: &Group, narrow: Dtype) -> R<Chain> {
    let mut b = Ops::new(3);
    let (gate, up, spread) = (0, 1, 2);
    let a = if g.silu {
        let one = b.f32(1.0);
        let neg = b.push(Op::Unary(Unary::Neg, gate));
        let e = b.push(Op::Unary(Unary::Exp, neg));
        let den = b.push(Op::Binary(Binary::Add, one, e));
        let s = b.push(Op::Unary(Unary::Recip, den));
        b.push(Op::Binary(Binary::Mul, gate, s))
    } else {
        // 0.5 t (1 + tanh(√(2/π) (t + 0.044715 t³)))
        let t3 = b.push(Op::Binary(Binary::Mul, gate, gate));
        let t3 = b.push(Op::Binary(Binary::Mul, t3, gate));
        let c = b.f32(0.044_715);
        let inner = b.push(Op::Fma(t3, c, gate));
        let k = b.f32(0.797_884_6);
        let inner = b.push(Op::Binary(Binary::Mul, inner, k));
        let th = b.push(Op::Unary(Unary::Tanh, inner));
        let one = b.f32(1.0);
        let th = b.push(Op::Binary(Binary::Add, th, one));
        let half = b.f32(0.5);
        let h = b.push(Op::Binary(Binary::Mul, gate, half));
        b.push(Op::Binary(Binary::Mul, h, th))
    };
    let act = b.push(Op::Binary(Binary::Mul, a, up));
    let out = b.push(Op::Binary(Binary::Mul, act, spread));
    let region = Region {
        width: g.experts * g.inter,
        params: vec![Shape::Row; 3],
        ops: b.ops,
        results: vec![out],
        rounds: vec![(act, narrow), (out, narrow)],
    };
    region
        .types()
        .map_err(|e| format!("node {}: the experts' activation: {e:?}", g.route))?;
    Ok(Chain {
        nodes: vec![g.route],
        ins: vec![
            input(synth(g.route, Kind::Gates), narrow, false),
            input(synth(g.route, Kind::Ups), narrow, false),
            input(synth(g.route, Kind::Spread), Dtype::F32, false),
        ],
        outs: vec![Out {
            value: synth(g.route, Kind::Acts),
            dtype: narrow,
        }],
        region,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernels_ir::interp::{Arg, eval};

    fn group(score: RouteScore, bias: bool, scale: bool) -> Group {
        Group {
            route: 7,
            nodes: vec![7],
            logits: ValueId(0),
            bias: bias.then_some(ValueId(1)),
            scale: scale.then_some(ValueId(2)),
            experts: 8,
            top_k: 3,
            score,
            x: ValueId(0),
            hidden: 16,
            gate: (ValueId(3), 8, 0),
            up: (ValueId(3), 8, 4),
            down: ValueId(4),
            inter: 4,
            silu: true,
            y: ValueId(5),
            dtype: Dtype::Bf16,
        }
    }

    /// The route the reference takes: the top k (the lower expert first on a tie) of the keys,
    /// their weights as the score states them, a gate an expert.
    fn reference(
        l: &[f32],
        bias: Option<&[f32]>,
        gain: Option<&[f32]>,
        score: RouteScore,
        k: usize,
    ) -> Vec<f32> {
        let s: Vec<f32> = match score {
            RouteScore::Sigmoid { .. } => l.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect(),
            _ => l.to_vec(),
        };
        let mut key: Vec<f32> = (0..l.len())
            .map(|i| s[i] + bias.map_or(0.0, |b| b[i]))
            .collect();
        let mut picks = Vec::new();
        for _ in 0..k {
            let (at, _) = key
                .iter()
                .enumerate()
                .fold((0, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
            picks.push(at);
            key[at] = f32::MIN;
        }
        let mut g = vec![0.0f32; l.len()];
        match score {
            RouteScore::Sigmoid {
                renormalize,
                scaling,
            } => {
                let sum: f32 = picks.iter().map(|&p| s[p]).sum();
                for &p in &picks {
                    g[p] = if renormalize { s[p] / sum } else { s[p] } * scaling;
                }
            }
            _ => {
                let m = picks.iter().map(|&p| l[p]).fold(f32::MIN, f32::max);
                let sum: f32 = picks.iter().map(|&p| (l[p] - m).exp()).sum();
                for &p in &picks {
                    g[p] = (l[p] - m).exp() / sum;
                }
            }
        }
        if let Some(w) = gain {
            for (x, w) in g.iter_mut().zip(w) {
                *x *= w;
            }
        }
        g
    }

    fn close(a: &[f32], b: &[f32]) -> bool {
        a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= 1e-5 * (1.0 + y.abs()))
    }

    /// The router's region picks and weighs as the route does: a softmax over the picked logits
    /// (with a per-expert gain), a biased sigmoid renormalized and scaled; a tie at the cut
    /// picks the lower expert alone.
    #[test]
    fn the_routers_region_picks_and_weighs_as_the_route() {
        // (the trace states the values' dtypes alone: any trace's)
        let trace = poem_compiler::catalog::repository()
            .parse("qwen3-0.6b-bf16-kv-bf16-l1")
            .expect("a catalog model")
            .trace(poem_ir::Platform::Cerebras);
        let l = vec![0.3, -1.2, 2.5, 0.3, 1.1, 0.3, -0.4, 0.9];
        let bias = vec![0.0, 0.5, -0.2, 0.0, 0.1, 0.0, 0.3, -0.1];
        let gain = vec![1.5, 0.5, 2.0, 1.0, 0.25, 1.0, 3.0, 1.0];
        let cases = [
            (RouteScore::Softmax, false, false),
            (RouteScore::Softmax, false, true),
            (
                RouteScore::Sigmoid {
                    renormalize: true,
                    scaling: 2.5,
                },
                true,
                false,
            ),
            (
                RouteScore::Sigmoid {
                    renormalize: false,
                    scaling: 1.0,
                },
                false,
                false,
            ),
        ];
        for (score, b, s) in cases {
            let g = group(score, b, s);
            let c = router(&trace, &g).unwrap_or_else(|e| panic!("{e}"));
            let mut args = vec![Arg::Row(l.clone())];
            if b {
                args.push(Arg::Row(bias.clone()));
            }
            if s {
                args.push(Arg::Row(gain.clone()));
            }
            let got = eval(&c.region, &args).unwrap_or_else(|e| panic!("{e}"));
            let want = reference(&l, b.then_some(&bias[..]), s.then_some(&gain[..]), score, 3);
            assert!(close(&got[0], &want), "{score:?}: {:?} vs {want:?}", got[0]);
        }
        // three logits tied at 0.3 past the top two: the lowest of them (expert 0) alone
        let g = group(RouteScore::Softmax, false, false);
        let c = router(&trace, &g).unwrap();
        let tied = vec![0.3, -1.2, 2.5, 0.3, 1.1, 0.3, -0.4, 0.3];
        let got = eval(&c.region, &[Arg::Row(tied)]).unwrap();
        let picked: Vec<usize> = (0..8).filter(|&i| got[0][i] != 0.0).collect();
        assert_eq!(picked, [0, 2, 4]);
    }

    /// A bank's views: an expert's gate or up rows a block, its down columns a block.
    #[test]
    fn a_banks_views_lay_its_experts() {
        use crate::stream::unit::View;
        // gate | up bank [E 2, 2I 4, H 3]: up row o of expert 1 is the bank's row 1·4 + 2 + o % 2
        let up = View::Rows {
            block: 2,
            stride: 4,
            off: 2,
        };
        assert_eq!(up.at(3, 1, 3, 4), Some((4 + 2 + 1) * 3 + 1));
        // down bank [E 2, H 3, I 2]: input i = e·I + j reads expert e's [H, I] at (o, j)
        let down = View::Blocks { block: 2 };
        assert_eq!(down.at(1, 3, 4, 3), Some(3 * 2 + 2 + 1));
        assert_eq!(View::Plain.at(2, 1, 5, 3), Some(11));
        assert_eq!(View::Spread { rep: 4 }.at(0, 0, 1, 4), None);
    }
}
