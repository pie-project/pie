//! Under `compiler.synth_regions`, a run of elementwise and row ops forms a
//! chain, lowered to one eta region (`fuse::chain`): a function of one row
//! the stream core writes a row frame's body for. Off, nothing forms.

use eta_compiler::eval::interp::Value;
use poem::ops::{elemwise, linear};
use poem::pattern::{Pattern, template};
use poem_compiler::DeviceProfile;
use poem_compiler::fuse::chain::{Chain, chains};
use poem_ir::Trace;

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
    assert_eq!(chain.nodes, 0..2);
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
    let out = chain
        .region
        .eval(&[
            Value::F32(x.clone()),
            Value::F32(y.clone()),
            Value::F32(w.clone()),
        ])
        .expect("the region runs");
    let h: Vec<f32> = y.iter().zip(&x).map(|(y, x)| y + x).collect();
    assert_eq!(out[0], Value::F32(h.clone()), "result 0 is the stream");
    let ms = h.iter().map(|v| v * v).sum::<f32>() / WIDTH as f32;
    let r = 1.0 / (ms + EPS).sqrt();
    let Value::F32(normed) = &out[1] else {
        panic!("an f32 row")
    };
    for (e, got) in normed.iter().enumerate() {
        let want = h[e] * r * w[e];
        assert!(
            (got - want).abs() <= 1e-5 * want.abs().max(1.0),
            "{e}: {got} vs {want}"
        );
    }

    let body = chain.region.body().expect("the stream core covers it");
    assert_eq!(
        body.passes, 2,
        "the row is held across its sum: {}",
        body.source
    );
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
    let args: Vec<Value> = [row(4), row(5), row(6)]
        .into_iter()
        .map(Value::F32)
        .collect();
    assert_eq!(
        the_one(&unfused).region.eval(&args),
        the_one(&fused).region.eval(&args)
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
    assert_eq!(chain.nodes, 0..3, "the matmuls end the chain");
    let leaving: Vec<poem_ir::ValueId> = chain.outs.iter().map(|out| out.value).collect();
    // The stream (a matmul reads it) and the activation leave; the normed
    // row only the activation reads stays in the registers.
    assert_eq!(leaving.len(), 2, "{leaving:?}");
    assert!(chain.region.body().is_ok());
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
    assert_eq!(spans, [0..2, 3..5], "erf's gelu has no eta form");
}

#[test]
fn the_compiler_forms_regions_only_when_asked() {
    let t = poem_compiler::catalog::deployments()
        .find(|row| row.name.starts_with("qwen3"))
        .expect("a qwen3 row")
        .trace(poem::Platform::Cuda);
    let off = bake_with(&t, &DeviceProfile::default()).expect("bakes");
    assert!(off.chains.is_empty(), "off by default");
    let on = bake_with(
        &t,
        &DeviceProfile {
            synth_regions: true,
            ..DeviceProfile::default()
        },
    )
    .expect("bakes");
    assert!(
        !on.chains.is_empty(),
        "a qwen3 forward has a residual norm a layer"
    );
    assert_eq!(
        poem_compiler::CompiledModel {
            chains: Vec::new(),
            ..on
        },
        off,
        "the regions change nothing else"
    );
}

/// Every catalog deployment's chains, on CUDA: each lies in its trace, after
/// the one before, and the stream core writes its body.
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
            if chain.nodes.start < after || chain.nodes.end as usize > t.nodes.len() {
                wrong.push(format!("{}: {:?} is out of order", row.name, chain.nodes));
            }
            after = chain.nodes.end;
            if let Err(why) = chain.region.body() {
                wrong.push(format!(
                    "{}: {:?} has no body: {why:?}",
                    row.name, chain.nodes
                ));
            }
        }
    }
    eprintln!("{formed} chains of {ops} ops");
    assert!(formed > 0, "the catalog forms chains");
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
