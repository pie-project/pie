//! A run of ops named outright (`fuse::chain::lower_run`: a spatial target's root phase, GENERIC.md
//! §3.2) lowers to its region whether or not a chain forms over it: flux2-mini's rotation over
//! four position axes, its positions a row's elements (scalar parameters), and its gated MLP's
//! activation over the packed `[gate | up]` row. No chain forms over either.

use kernels_ir::Shape;
use kernels_ir::interp::{Arg, eval};
use poem_compiler::fuse::chain::{chains, lower_run};
use poem_ir::{Elemwise, Linear, Operation, Platform, Trace};

fn flux2_mini() -> Trace {
    poem_compiler::catalog::repository()
        .parse("flux2-mini-bf16-kv-bf16")
        .expect("the catalog holds flux2-mini")
        .trace(Platform::Cerebras)
}

fn first(trace: &Trace, pick: impl Fn(&Operation) -> bool) -> u32 {
    trace
        .nodes
        .iter()
        .position(|n| pick(&n.op))
        .expect("flux2-mini runs it") as u32
}

#[test]
fn a_rotation_over_position_axes_lowers_with_its_positions_as_scalars() {
    let trace = flux2_mini();
    let at = first(&trace, |op| {
        matches!(op, Operation::Elemwise(Elemwise::RopeAxes { .. }))
    });
    let Operation::Elemwise(Elemwise::RopeAxes {
        dims,
        thetas,
        head_dim,
        ..
    }) = trace.nodes[at as usize].op.clone()
    else {
        unreachable!()
    };
    let chain = lower_run(&trace, &[at]).expect("the rotation lowers");
    let reg = &chain.region;
    let axes = dims.iter().filter(|d| **d > 0).count();
    assert_eq!(reg.params.len(), 1 + axes);
    assert_eq!(reg.params[0], Shape::Row);
    assert!(reg.params[1..].iter().all(|s| *s == Shape::Scalar));
    assert!(chain.ins[1..].iter().all(|i| i.each), "{:?}", chain.ins);
    let width = reg.width as usize;
    let x: Vec<f32> = (0..width)
        .map(|i| ((i * 37 % 23) as f32 - 11.0) / 8.0)
        .collect();
    let run = |pos: &[f32]| {
        let mut args = vec![Arg::Row(x.clone())];
        args.extend(pos.iter().map(|&p| Arg::Scalar(p)));
        eval(reg, &args).expect("interp runs it").remove(0)
    };
    // at position 0 every pair turns by nothing
    let zero = run(&vec![0.0; axes]);
    assert!(
        zero.iter()
            .zip(&x)
            .all(|(a, b)| (a - kernels_ir::skeleton::round_bf16(*b)).abs()
                <= 1e-6 * b.abs().max(1.0))
    );
    // else each interleaved pair of an axis at theta^(-2w / dim) times its position
    let pos = [0.0, 3.0, 5.0, 7.0];
    let got = run(&pos[..axes]);
    let hd = head_dim as usize;
    for head in 0..width / hd {
        let mut start = 0;
        for (a, &dim) in dims.iter().enumerate().filter(|(_, d)| **d > 0) {
            for w in 0..dim as usize / 2 {
                let ang =
                    f64::from(pos[a]) * f64::from(thetas[a]).powf(-2.0 * w as f64 / f64::from(dim));
                let (lo, hi) = (head * hd + start + 2 * w, head * hd + start + 2 * w + 1);
                let (xa, xb) = (f64::from(x[lo]), f64::from(x[hi]));
                let want = [
                    xa * ang.cos() - xb * ang.sin(),
                    xb * ang.cos() + xa * ang.sin(),
                ];
                for (e, w) in [lo, hi].into_iter().zip(want) {
                    assert!(
                        (f64::from(got[e]) - w).abs() <= 1e-2 * w.abs().max(0.1),
                        "column {e}: {} vs {w}",
                        got[e]
                    );
                }
            }
            start += dim as usize;
        }
    }
    assert!(
        chains(&trace, &|_, _, _| true)
            .iter()
            .all(|c| !c.nodes.contains(&at)),
        "no chain forms over the rotation"
    );
}

#[test]
fn a_gated_mlps_activation_lowers_over_its_packed_halves() {
    let trace = flux2_mini();
    let at = first(&trace, |op| {
        matches!(op, Operation::Linear(Linear::Glu { .. }))
    });
    let chain = lower_run(&trace, &[at]).expect("the activation lowers");
    let reg = &chain.region;
    let w = reg.width as usize;
    assert_eq!(reg.params, vec![Shape::Row, Shape::Row]);
    assert_eq!((chain.ins[0].from, chain.ins[1].from), (0, reg.width));
    let g: Vec<f32> = (0..w).map(|i| (i % 13) as f32 / 4.0 - 1.5).collect();
    let u: Vec<f32> = (0..w).map(|i| (i % 7) as f32 / 2.0 - 1.0).collect();
    let got = eval(reg, &[Arg::Row(g.clone()), Arg::Row(u.clone())]).expect("interp runs it");
    for e in 0..w {
        let want = g[e] / (1.0 + (-g[e]).exp()) * u[e];
        assert!(
            (got[0][e] - want).abs() <= 1e-2 * want.abs().max(0.1),
            "column {e}"
        );
    }
    assert!(
        chains(&trace, &|_, _, _| true)
            .iter()
            .all(|c| !c.nodes.contains(&at))
    );
}
