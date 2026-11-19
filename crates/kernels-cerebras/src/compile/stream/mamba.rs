//! Mamba-2 lowered onto the units' existing steps: its conv and scan a state column per head, its
//! gated group norm as root regions and matvecs of constant ones:
//!
//! - the scan's raw output `r = C S + D x` per head, at the head roots;
//! - `g = r silu(z)` and `g²`, a root region;
//! - each group's sum of squares, a matvec of constant ones (`View::Sum`);
//! - that sum spread back over the group's columns, a matvec of constant ones (`View::Spread`);
//! - `y = g rsqrt(sum / width + eps) w`, a root region.

use poem_ir::{Dtype, Operation, Ssm, Trace, ValueId};

use kernels_ir::build::Ops;
use kernels_ir::{Binary, Op, Region, Shape, Unary};
use poem_compiler::fuse::chain::{Chain, In, Out};

use super::moe::SYNTH;

type R<T> = Result<T, String>;

/// What a synthetic value of a Mamba-2 node is (its node's [`super::moe::synth`] numbering).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The scan's raw output `C S + D x`, a head's columns each (f32, the inner width).
    Raw = 0,
    /// `g = r silu(z)` (f32, inner).
    Gated = 1,
    /// `g²` (f32, inner).
    Squares = 2,
    /// Each group's sum of squares (f32, a group each).
    Sums = 3,
    /// The sums over each group's columns (f32, inner).
    Spread = 4,
    /// A row vector moved to the column roots (e.g. delta attention's forget logits, f32).
    Moved = 5,
}

/// The synthetic value of `kind` of the Mamba-2 at `node`.
#[must_use]
pub fn synth(node: u32, kind: Kind) -> ValueId {
    ValueId(SYNTH | (node * 8) | kind as u32)
}

/// A Mamba-2 node's shape: (heads, groups, state dim, head dim, inner width).
#[must_use]
pub fn dims(trace: &Trace, node: u32) -> Option<(u32, u32, u32, u32, u32)> {
    let Operation::Ssm(Ssm::Mamba2 {
        heads,
        groups,
        state_dim,
        z,
        ..
    }) = &trace.nodes.get(node as usize)?.op
    else {
        return None;
    };
    let inner = match trace.values.get(z.0 as usize).map(|d| &d.ty) {
        Some(poem_ir::Ty::Tensor { shape, .. }) => match shape.last() {
            Some(poem_ir::Dim::Const(n)) => *n as u32,
            _ => return None,
        },
        _ => return None,
    };
    Some((*heads, *groups, *state_dim, inner / heads.max(&1), inner))
}

/// The width and dtype of a synthetic value of a Mamba-2 node; `None` if `node` is no Mamba-2.
#[must_use]
pub fn ty(trace: &Trace, node: u32, kind: u32) -> Option<(u32, Dtype)> {
    let (_, groups, _, _, inner) = dims(trace, node)?;
    Some(match kind {
        3 => (groups, Dtype::F32),
        _ => (inner, Dtype::F32),
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

/// `g = r silu(z)` and `g²` over the inner width.
pub fn gate(node: u32, z: ValueId, z_dtype: Dtype, inner: u32) -> R<Chain> {
    let mut b = Ops::new(2);
    let (raw, zz) = (0, 1);
    let one = b.f32(1.0);
    let neg = b.push(Op::Unary(Unary::Neg, zz));
    let e = b.push(Op::Unary(Unary::Exp, neg));
    let den = b.push(Op::Binary(Binary::Add, one, e));
    let s = b.push(Op::Unary(Unary::Recip, den));
    let silu = b.push(Op::Binary(Binary::Mul, zz, s));
    let g = b.push(Op::Binary(Binary::Mul, raw, silu));
    let sq = b.push(Op::Binary(Binary::Mul, g, g));
    let region = Region {
        width: inner,
        params: vec![Shape::Row; 2],
        ops: b.ops,
        results: vec![g, sq],
        rounds: Vec::new(),
    };
    region
        .types()
        .map_err(|e| format!("node {node}: mamba2's gate: {e:?}"))?;
    Ok(Chain {
        nodes: vec![node],
        ins: vec![
            input(synth(node, Kind::Raw), Dtype::F32, false),
            input(z, z_dtype, false),
        ],
        outs: vec![
            Out {
                value: synth(node, Kind::Gated),
                dtype: Dtype::F32,
            },
            Out {
                value: synth(node, Kind::Squares),
                dtype: Dtype::F32,
            },
        ],
        region,
    })
}

/// `y = g rsqrt(spread / group + eps) w` over the inner width (`w` the norm's weight).
#[allow(clippy::too_many_arguments)]
pub fn norm(
    node: u32,
    group: u32,
    eps: f32,
    w: ValueId,
    w_dtype: Dtype,
    y: ValueId,
    y_dtype: Dtype,
    inner: u32,
) -> R<Chain> {
    let mut b = Ops::new(3);
    let (g, sp, ww) = (0, 1, 2);
    let inv = b.f32(1.0 / group as f32);
    let ep = b.f32(eps);
    let m = b.push(Op::Binary(Binary::Mul, sp, inv));
    let m = b.push(Op::Binary(Binary::Add, m, ep));
    let r = b.push(Op::Unary(Unary::Rsqrt, m));
    let n = b.push(Op::Binary(Binary::Mul, g, r));
    let out = b.push(Op::Binary(Binary::Mul, n, ww));
    let region = Region {
        width: inner,
        params: vec![Shape::Row; 3],
        ops: b.ops,
        results: vec![out],
        rounds: if y_dtype == Dtype::F32 {
            Vec::new()
        } else {
            vec![(out, y_dtype)]
        },
    };
    region
        .types()
        .map_err(|e| format!("node {node}: mamba2's norm: {e:?}"))?;
    Ok(Chain {
        nodes: vec![node],
        ins: vec![
            input(synth(node, Kind::Gated), Dtype::F32, false),
            input(synth(node, Kind::Spread), Dtype::F32, false),
            input(w, w_dtype, true),
        ],
        outs: vec![Out {
            value: y,
            dtype: y_dtype,
        }],
        region,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernels_ir::interp::{Arg, eval};

    /// The gate, the group sums (a `View::Sum` matvec's ones), their spread (a `View::Spread`'s)
    /// and the norm give Mamba-2's gated group norm `rmsnorm_group(r silu(z)) w`.
    #[test]
    fn the_gated_group_norm_is_mamba2s() {
        use crate::compile::stream::unit::View;
        let (inner, groups) = (8usize, 2usize);
        let gw = inner / groups;
        let r: Vec<f32> = (0..inner).map(|i| 0.3 * i as f32 - 1.0).collect();
        let z: Vec<f32> = (0..inner).map(|i| 0.5 - 0.2 * i as f32).collect();
        let w: Vec<f32> = (0..inner).map(|i| 1.0 + 0.1 * i as f32).collect();
        let eps = 1e-5;
        let g = gate(9, ValueId(1), Dtype::F32, inner as u32).unwrap();
        let out = eval(&g.region, &[Arg::Row(r.clone()), Arg::Row(z.clone())]).unwrap();
        let (gated, sq) = (&out[0], &out[1]);
        let sum = View::Sum { rep: gw as u32 };
        let spread = View::Spread { rep: gw as u32 };
        assert_eq!(sum.at(0, 0, inner, groups), None);
        let sums: Vec<f32> = (0..groups)
            .map(|o| (0..inner).filter(|i| i / gw == o).map(|i| sq[i]).sum())
            .collect();
        let sp: Vec<f32> = (0..inner).map(|o| sums[o / gw]).collect();
        let _ = spread;
        let n = norm(
            9,
            gw as u32,
            eps,
            ValueId(2),
            Dtype::F32,
            ValueId(3),
            Dtype::F32,
            inner as u32,
        )
        .unwrap();
        let y = eval(
            &n.region,
            &[Arg::Row(gated.clone()), Arg::Row(sp), Arg::Row(w.clone())],
        )
        .unwrap();
        for i in 0..inner {
            let silu = |x: f32| x / (1.0 + (-x).exp());
            let grp = i / gw;
            let gs: Vec<f32> = (grp * gw..(grp + 1) * gw)
                .map(|j| r[j] * silu(z[j]))
                .collect();
            let ms = gs.iter().map(|x| x * x).sum::<f32>() / gw as f32;
            let want = r[i] * silu(z[i]) / (ms + eps).sqrt() * w[i];
            assert!(
                (y[0][i] - want).abs() < 1e-4 * want.abs().max(1.0),
                "{i}: {} vs {want}",
                y[0][i]
            );
        }
    }
}
