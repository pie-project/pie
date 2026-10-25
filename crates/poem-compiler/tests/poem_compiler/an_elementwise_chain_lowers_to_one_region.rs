//! Under `compiler.synth_regions`, a run of elementwise and row ops forms a
//! chain, lowered to one region (`fuse::chain`): a function of one row
//! the row core schedules into passes over the row, and folded into one
//! `fused.region` op the plan is of. Off, nothing forms.

use kernels_ir::interp::{Arg, eval};
use kernels_ir::{Shape, schedule};
use poem::ops::{elemwise, linear};
use poem::pattern::{Pattern, template};
use poem_compiler::DeviceProfile;
use poem_compiler::fuse::chain::{Chain, chains, fold};
use poem_ir::{Fused, Operation, Trace, ValueId};

use crate::common::bake_with;

const WIDTH: u64 = 64;
const EPS: f32 = 1e-6;

fn trace(build: impl FnOnce(&Pattern)) -> Trace {
    template(build).trace
}

/// `h = y + x`, then `rmsnorm(h) * w`: the pilot's residual norm, written
/// as the two ops a model writes.
fn residual_norm(p: &Pattern) -> (poem::Value, poem::Value) {
    let h = elemwise::residual_add(&p.rows("x", WIDTH), &p.rows("y", WIDTH));
    let n = elemwise::rmsnorm(&h, &p.weight("w", [WIDTH]), EPS);
    (h, n)
}

/// `v` read past the chain, as the next layer reads the stream.
fn read_later(p: &Pattern, v: &poem::Value) {
    let _ = linear::matmul(v, &p.weight("next", [WIDTH, WIDTH]));
}

fn row(seed: u32) -> Vec<f32> {
    (0..WIDTH as u32)
        .map(|i| ((i * 37 + seed * 11) % 23) as f32 / 8.0 - 1.4)
        .collect()
}

/// `x` rounded to bf16, nearest and ties to even.
fn bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}

fn the_one(chains: &[Chain]) -> &Chain {
    assert_eq!(chains.len(), 1, "{chains:#?}");
    &chains[0]
}

#[test]
fn a_residual_add_and_its_norm_form_one_region() {
    let t = trace(|p| {
        let (h, _) = residual_norm(p);
        read_later(p, &h);
    });
    let formed = chains(&t);
    let chain = the_one(&formed);
    assert_eq!(chain.nodes, [0, 1]);
    let shared: Vec<bool> = chain.ins.iter().map(|input| input.shared).collect();
    assert_eq!(
        shared,
        [false, false, true],
        "x and y a row a row, w one row"
    );
    assert_eq!(
        chain.outs.len(),
        2,
        "the stream and the normed row both leave"
    );

    let (x, y, w) = (row(1), row(2), row(3));
    let out = eval(
        &chain.region,
        &[
            Arg::Row(x.clone()),
            Arg::Row(y.clone()),
            Arg::Row(w.clone()),
        ],
    )
    .expect("the region runs");
    // Each value an op defines is rounded to its bf16 where it is defined,
    // as the two ops store it: the stream, and the normed row.
    assert_eq!(
        chain.region.rounds,
        chain
            .region
            .results
            .iter()
            .map(|&v| (v, poem_ir::Dtype::Bf16))
            .collect::<Vec<_>>()
    );
    let h: Vec<f32> = y.iter().zip(&x).map(|(y, x)| bf16(y + x)).collect();
    assert_eq!(out[0], h, "result 0 is the stream");
    let ms = h.iter().map(|v| v * v).sum::<f32>() / WIDTH as f32;
    let r = 1.0 / (ms + EPS).sqrt();
    let normed = &out[1];
    for (e, got) in normed.iter().enumerate() {
        let want = h[e] * r * w[e];
        assert_eq!(*got, bf16(*got), "{e}: the normed row is rounded");
        assert!(
            (got - want).abs() <= want.abs().max(1.0) / 128.0,
            "{e}: {got} vs {want}"
        );
    }

    let kernel = schedule(&chain.region).expect("the row core schedules it");
    assert_eq!(kernel.passes(), 2, "the row is held across its sum");
}

#[test]
fn the_fused_residual_norm_lowers_as_its_two_ops_do() {
    let written = trace(|p| drop(residual_norm(p)));
    let fused = poem_compiler::fuse::fuse(
        written.clone(),
        &poem_compiler::fuse::rules::forming(&["fused.residual_add_rmsnorm"]),
    );
    assert_eq!(fused.nodes.len(), 1, "the rule forms the fused op");
    let lone = poem_compiler::fuse::chain::chains(&fused);
    assert!(lone.is_empty(), "a lone op keeps its hand kernel");

    // Behind an activation, the fused op joins a chain, and its region
    // computes what the two ops' region does.
    let behind = |p: &Pattern| {
        let (h, n) = residual_norm(p);
        elemwise::silu(&n);
        read_later(p, &h);
    };
    let unfused = chains(&trace(behind));
    let refused = poem_compiler::fuse::fuse(
        trace(behind),
        &poem_compiler::fuse::rules::forming(&["fused.residual_add_rmsnorm"]),
    );
    let fused = chains(&refused);
    let args: Vec<Arg> = [row(4), row(5), row(6)].into_iter().map(Arg::Row).collect();
    assert_eq!(
        eval(&the_one(&unfused).region, &args),
        eval(&the_one(&fused).region, &args)
    );
}

/// An op between a chain's ops that neither reads what the chain computes
/// nor writes what it reads runs before the region, which lands where the
/// chain's last op was.
#[test]
fn a_chain_runs_past_an_op_it_does_not_touch() {
    let t = trace(|p| {
        let h = elemwise::residual_add(&p.rows("x", WIDTH), &p.rows("y", WIDTH));
        let side = linear::matmul(&p.rows("z", WIDTH), &p.weight("side", [WIDTH, WIDTH]));
        let n = elemwise::rmsnorm(&h, &p.weight("w", [WIDTH]), EPS);
        let _ = linear::matmul(&n, &p.weight("out", [WIDTH, WIDTH]));
        read_later(p, &side);
    });
    let formed = chains(&t);
    assert_eq!(the_one(&formed).nodes, [0, 2]);
    let (folded, _) = fold(t);
    assert!(
        matches!(folded.nodes[0].op, Operation::Linear(_)),
        "the matmul runs first"
    );
    assert!(matches!(
        folded.nodes[1].op,
        Operation::Fused(Fused::Region { .. })
    ));
}

/// A layer's scale, a weight of one element, is a scalar parameter: Gemma's
/// norm, residual add and scale form one region.
#[test]
fn a_one_element_scale_is_a_scalar_parameter() {
    let t = trace(|p| {
        let n = elemwise::rmsnorm(&p.rows("x", WIDTH), &p.weight("w", [WIDTH]), EPS);
        let h = elemwise::residual_add(&n, &p.rows("y", WIDTH));
        let scaled = elemwise::scale(&p.weight("s", [1]), &h);
        read_later(p, &scaled);
    });
    let formed = chains(&t);
    let chain = the_one(&formed);
    assert_eq!(chain.nodes, [0, 1, 2]);
    let ones: Vec<bool> = chain.ins.iter().map(|input| input.one).collect();
    assert_eq!(ones, [false, false, false, true], "x, w, y and the scale");
    assert_eq!(
        chain.region.params[3],
        Shape::Scalar,
        "the scale is a scalar"
    );
    let (x, w, y) = (row(1), row(2), row(3));
    let out = eval(
        &chain.region,
        &[
            Arg::Row(x),
            Arg::Row(w),
            Arg::Row(y.clone()),
            Arg::Scalar(0.5),
        ],
    )
    .expect("the region runs");
    assert_eq!(out.len(), 1, "only the scaled stream leaves");
    let kernel = schedule(&chain.region).expect("the row core schedules it");
    assert!(
        kernel
            .phases
            .iter()
            .flat_map(|p| p.pass.iter())
            .any(|pass| pass
                .rows
                .iter()
                .any(|row| format!("{:?}", row.expr).contains("One(3)"))),
        "the scale is read as one value: {kernel:?}"
    );
}

#[test]
fn a_lone_op_keeps_its_hand_kernel() {
    let t = trace(|p| drop(elemwise::silu(&p.rows("x", WIDTH))));
    assert!(chains(&t).is_empty());
}

#[test]
fn a_value_read_past_the_chain_leaves_it() {
    let t = trace(|p| {
        let (h, n) = residual_norm(p);
        let g = elemwise::silu(&n);
        let _ = linear::matmul(&h, &p.weight("proj", [WIDTH, WIDTH]));
        let _ = linear::matmul(&g, &p.weight("down", [WIDTH, WIDTH]));
    });
    let formed = chains(&t);
    let chain = the_one(&formed);
    assert_eq!(chain.nodes, [0, 1, 2], "the matmuls end the chain");
    let leaving: Vec<poem_ir::ValueId> = chain.outs.iter().map(|out| out.value).collect();
    // The stream (a matmul reads it) and the activation leave; the normed
    // row only the activation reads stays in the registers.
    assert_eq!(leaving.len(), 2, "{leaving:?}");
    assert!(schedule(&chain.region).is_ok());
}

#[test]
fn an_op_no_region_lowers_ends_the_chain() {
    let t = trace(|p| {
        let a = elemwise::silu(&p.rows("x", WIDTH));
        let b = elemwise::mul(&a, &p.rows("g", WIDTH));
        let c = elemwise::gelu(&b, false);
        let d = elemwise::tanh(&c);
        let _ = elemwise::add(&d, &p.rows("z", WIDTH));
    });
    let formed = chains(&t);
    let spans: Vec<_> = formed.iter().map(|chain| chain.nodes.clone()).collect();
    assert_eq!(
        spans,
        [vec![0, 1], vec![3, 4]],
        "erf's gelu has no region form"
    );
}

fn qwen3() -> Trace {
    poem_compiler::catalog::deployments()
        .find(|row| row.name.starts_with("qwen3"))
        .expect("a qwen3 row")
        .trace(poem::Platform::Cuda)
}

fn synth() -> DeviceProfile {
    DeviceProfile {
        synth_regions: true,
        ..DeviceProfile::default()
    }
}

/// The `fused.region` ops of `t`: their inputs, outputs and region key.
fn regions(t: &Trace) -> Vec<(Vec<ValueId>, Vec<ValueId>, u64)> {
    t.nodes
        .iter()
        .filter_map(|node| match &node.op {
            Operation::Fused(Fused::Region { ins, outs, key, .. }) => {
                Some((ins.clone(), outs.clone(), *key))
            }
            _ => None,
        })
        .collect()
}

#[test]
fn a_folded_trace_plans_as_it_stands() {
    let t = qwen3();
    let (folded, bodies) = fold(t.clone());
    let formed = chains(&t);
    let ops = regions(&folded);
    assert!(
        !ops.is_empty(),
        "a qwen3 forward has a residual norm a layer"
    );
    let held = formed
        .iter()
        .filter(|chain| poem_compiler::fuse::chain::skeleton_holds(&t, chain, chain.region.width))
        .count();
    assert_eq!(ops.len(), held, "each chain the skeleton holds is one op");
    for (ins, outs, key) in &ops {
        let region = bodies.get(key).expect("the op's region is held");
        assert_eq!(region.key(), *key);
        assert_eq!(region.params.len(), ins.len());
        assert_eq!(region.results.len(), outs.len());
    }
    assert!(
        bodies.len() < ops.len(),
        "a layer's chain shares the next's region"
    );
    poem_ir::check(&folded).expect("the folded trace checks");
    let plan = bake_with(&folded, &synth()).expect("bakes");
    assert_eq!(
        plan.classes.node_mask.len(),
        folded.nodes.len(),
        "the plan is of the folded trace"
    );
}

/// A layer's last residual add and the next layer's input norm are one
/// chain, as `fuse.residual_add_rmsnorm` makes them one op: no residual add
/// of a qwen3 forward is left outside a region, and a region that crosses a
/// layer's end lands in one of its layers, so the folded trace's layers
/// still only climb.
#[test]
fn a_residual_add_and_the_next_layers_norm_form_one_region() {
    let t = qwen3();
    let crossing = chains(&t)
        .iter()
        .filter(|chain| {
            t.nodes[chain.nodes[0] as usize].layer
                != t.nodes[*chain.nodes.last().expect("ops") as usize].layer
        })
        .count();
    assert!(crossing > 0, "a layer's stream is normed by the next");
    let (folded, _) = fold(t.clone());
    let left = folded
        .nodes
        .iter()
        .filter(|node| {
            matches!(
                node.op,
                Operation::Elemwise(poem_ir::Elemwise::ResidualAdd { .. })
            )
        })
        .count();
    assert_eq!(left, 0, "every residual add folds with the norm after it");
    let layers: Vec<u32> = folded.nodes.iter().filter_map(|node| node.layer).collect();
    assert!(
        layers.windows(2).all(|pair| pair[0] <= pair[1]),
        "the folded trace's layers climb"
    );
}

/// The arena lands no region's output over one of its inputs: a row skeleton
/// reads its row whole before it writes any of it, but a block's row is
/// not another block's.
#[test]
fn no_region_lands_an_output_over_its_input() {
    let (folded, _) = fold(qwen3());
    let folded = &folded;
    let on = bake_with(folded, &synth()).expect("bakes");
    let fire = poem_compiler::FireRows::ceilings(&crate::common::budgets_of(folded));
    let (mut wrong, mut held) = (Vec::new(), 0usize);
    for (ins, outs, _) in regions(folded) {
        for out in &outs {
            for input in &ins {
                let (o, i) = (on.arena.root(*out), on.arena.root(*input));
                let overlap = match (on.arena.window(*out, fire), on.arena.window(*input, fire)) {
                    (Some(a), Some(b)) => {
                        held += 1;
                        a.offset < b.offset + b.bytes && b.offset < a.offset + a.bytes
                    }
                    _ => false,
                };
                if o == i || overlap {
                    wrong.push(format!("{out:?} lands over {input:?}"));
                }
            }
        }
    }
    assert!(held > 0, "a region reads a value the arena holds");
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// Every catalog deployment, on CUDA, folded: each chain the row skeleton holds
/// one op, and the trace still checks.
#[test]
fn every_folded_catalog_trace_checks() {
    let mut wrong: Vec<String> = Vec::new();
    for row in poem_compiler::catalog::deployments() {
        let t = row.trace(poem::Platform::Cuda);
        let (folded, bodies) = fold(t.clone());
        let ops = regions(&folded);
        let held = chains(&t)
            .into_iter()
            .filter(|chain| {
                schedule(&chain.region).is_ok()
                    && poem_compiler::fuse::chain::skeleton_holds(&t, chain, chain.region.width)
            })
            .count();
        if ops.len() != held {
            wrong.push(format!("{}: a chain is not folded", row.name));
        }
        if ops.iter().any(|(_, _, body)| !bodies.contains_key(body)) {
            wrong.push(format!("{}: an op names no body", row.name));
        }
        if let Err(faults) = poem_ir::check(&folded) {
            wrong.push(format!("{}: {faults:?}", row.name));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// Every catalog deployment's chains, on CUDA: each lies in its trace, after
/// the one before, and the row core schedules its region.
#[test]
fn every_catalog_chain_has_a_body() {
    let mut wrong: Vec<String> = Vec::new();
    let (mut formed, mut ops) = (0usize, 0usize);
    for row in poem_compiler::catalog::deployments() {
        let t = row.trace(poem::Platform::Cuda);
        let mut after = 0u32;
        for chain in chains(&t) {
            formed += 1;
            ops += chain.nodes.len();
            let ordered = chain.nodes.windows(2).all(|pair| pair[0] < pair[1]);
            if chain.nodes[0] < after
                || !ordered
                || *chain.nodes.last().expect("ops") as usize >= t.nodes.len()
            {
                wrong.push(format!("{}: {:?} is out of order", row.name, chain.nodes));
            }
            after = chain.nodes[0] + 1;
            if let Err(why) = schedule(&chain.region) {
                wrong.push(format!(
                    "{}: {:?} has no schedule: {why:?}",
                    row.name, chain.nodes
                ));
            }
        }
    }
    eprintln!("{formed} chains of {ops} ops");
    assert!(formed > 0, "the catalog forms chains");
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
