//! `survey`: every catalog text model's bf16 deployment tried through the wafer compiler, stage by
//! stage: the trace, the layer program (`spatial::layer::lower`), the shape (`target::shape`), the
//! kernels' refusals (`compose::refuse`), the plan (pinned or searched) and the program
//! (`Program::build`). Prints one line a deployment: the stage it stops at and why, and every op
//! that runs on the wafer with no spatial lowering (not just the first). A poem with a `denoise`
//! reading (a diffusion transformer) gets a second line, its stream regime's: the reading's stream
//! program and shape (`kernels_cerebras::stream::shape_of`, 64 rows a class) or the refusal.
//!
//! ```text
//! cargo run --release -p kernels-cerebras --example survey [-- <name filter>... | =<deployment>...]
//! ```
//!
//! With no `=`, every listed bf16 deployment (`ANY=1`: every listed one) whose name holds a
//! filter; `=<deployment>` that one exactly, listed or not. `DEEP=1` also plans (the pinned plan
//! or the search); `READINGS=1` with `=<deployment>` counts the ops each value of the trace's
//! `reading` switch runs (a diffusion or speech poem's sub-models). `LIST=1`: a deployment a
//! model. GENERIC.md's baseline tables are this survey's.

use std::collections::BTreeMap;
use std::time::Instant;

use poem_compiler::catalog::{Deployment, repository};
use poem_ir::{Operands, Operation, Platform, Trace};

use kernels_cerebras::target::{self, Ask};

/// The ops `spatial::layer::lower` gives a phase inside a layer (its `Body::node`).
const LAYER: &[&str] = &[
    "gather.split_qkv",
    "gather.split_q_gate",
    "gather.split_rows",
    "linear.matmul",
    "linear.moe_select",
    "attn.decode",
    "ssm.gated_delta",
    "attn.kv_append",
    "attn.sink",
    "ssm.causal_conv1d",
    "ssm.gdn_gates",
    "gather.embed",
    "gather.select",
    "elemwise.act",
    "elemwise.add_bias",
    "elemwise.gate_sigmoid_mul",
    "elemwise.residual_add",
    "elemwise.rmsnorm",
    "elemwise.rmsnorm_gated",
    "elemwise.rope",
    "linear.glu",
    "linear.moe_route",
    "linear.moe_combine",
    "linear.moe_bias_sum",
];

const OUTER: &[&str] = &[
    "attn.plan_decode",
    "elemwise.act",
    "elemwise.rmsnorm",
    "elemwise.softcap",
    "gather.embed",
    "gather.rows",
    "linear.matmul",
    "sampling.argmax",
];

/// The ops that run on the wafer (one token) with no lowering, layer ops and outer ops apart,
/// with their counts; only from the token embedding on.
fn missing(trace: &Trace) -> (BTreeMap<String, usize>, BTreeMap<String, usize>) {
    let (mut inner, mut outer) = (BTreeMap::new(), BTreeMap::new());
    let Ok(an) = poem_compiler::analysis::analyze(trace) else {
        return (inner, outer);
    };
    let first = an.layers.iter().position(Option::is_some).unwrap_or(0);
    let embed = trace
        .nodes
        .iter()
        .enumerate()
        .take(first)
        .filter(|(_, n)| {
            matches!(
                n.op,
                Operation::Gather(poem_ir::Gather::Embed { weights: None, .. })
            )
        })
        .map(|(a, _)| a)
        .next_back()
        .unwrap_or(0);
    for (at, n) in trace.nodes.iter().enumerate().skip(embed) {
        if !poem_compiler::spatial::layer::single_token(trace, &n.guard) {
            continue;
        }
        let name = n.op.name();
        match an.layers[at] {
            Some(_) if !LAYER.contains(&name) => *inner.entry(name.to_string()).or_insert(0) += 1,
            None if !OUTER.contains(&name) => *outer.entry(name.to_string()).or_insert(0) += 1,
            _ => {}
        }
    }
    (inner, outer)
}

fn short(s: &str) -> String {
    let s = s.replace('\n', " ");
    if s.len() > 300 {
        format!("{}...", &s[..300])
    } else {
        s
    }
}

/// The stream regime's line for a poem with a `denoise` reading: its program at 64 rows a class
/// (STREAM-BUILD-OK: units, K / V columns and lane bus laid and coloured, the CSL emitted; its
/// fabric and the op-derived step cycles), or where it stops (STREAM-LOWER: the stream program;
/// STREAM-SHAPE: the params; STREAM-BUILD: the units, layouts, floorplan, colours or CSL).
fn stream(d: &Deployment, trace: &Trace) {
    let reading = "denoise";
    let denoises = trace.facts.table.iter().any(|f| {
        matches!(f, poem_ir::Fact::Switch { name, value } if name == "reading" && value == reading)
    });
    if !denoises {
        return;
    }
    let name = &d.name;
    let classes = poem_compiler::spatial::layer::stream::classes(trace, reading, 64);
    let an = match poem_compiler::analysis::analyze(trace) {
        Ok(a) => a,
        Err(e) => {
            return println!(
                "{name}\treading={reading}\tANALYSIS\t{}",
                short(&e.to_string())
            );
        }
    };
    if let Err(e) = poem_compiler::spatial::layer::stream::lower(trace, &an, &classes) {
        return println!("{name}\treading={reading}\tSTREAM-LOWER\t{}", short(&e));
    }
    let s = match kernels_cerebras::stream::shape(trace, reading, &classes) {
        Ok(s) => s,
        Err(e) => return println!("{name}\treading={reading}\tSTREAM-SHAPE\t{}", short(&e)),
    };
    let what = format!(
        "classes={} units={} layer-bodies={} hidden={} heads={}x{}",
        s.classes
            .iter()
            .map(|(c, r)| format!("{c}:{r}"))
            .collect::<Vec<_>>()
            .join(","),
        s.units,
        s.layer_bodies,
        s.hidden,
        s.attn.q_heads,
        s.attn.head_dim
    );
    let built = kernels_cerebras::stream::build::program(trace, s, true).and_then(|p| {
        kernels_cerebras::emit::stream::emit(&p)?;
        Ok(p)
    });
    match built {
        Ok(p) => println!(
            "{name}\treading={reading}\tSTREAM-BUILD-OK\t{what} fabric={}x{} kv-columns={} \
             lane-words={} cycles~{:.1}M",
            p.stream.w,
            p.stream.h,
            p.stream.kv.len(),
            p.lane_words,
            kernels_cerebras::cost::stream::time(&p).cycles / 1e6
        ),
        Err(e) => println!(
            "{name}\treading={reading}\tSTREAM-BUILD\t{what} {}",
            short(&e)
        ),
    }
}

fn survey(d: &Deployment, deep: bool) {
    let t0 = Instant::now();
    let name = &d.name;
    let trace = match d.try_trace(Platform::Cerebras) {
        Ok(t) => t,
        Err(e) => return println!("{name}\tTRACE\t{}", short(&e.0)),
    };
    stream(d, &trace);
    let (inner, outer) = missing(&trace);
    let fmt = |m: &BTreeMap<String, usize>| {
        m.iter()
            .map(|(k, v)| format!("{k}x{v}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    let ops = format!(
        "missing-layer-ops=[{}] missing-outer-ops=[{}]",
        fmt(&inner),
        fmt(&outer)
    );
    let an = match poem_compiler::analysis::analyze(&trace) {
        Ok(a) => a,
        Err(e) => return println!("{name}\tANALYSIS\t{}\t{ops}", short(&e.to_string())),
    };
    let prog = poem_compiler::spatial::layer::lower(&trace, &an, &|g| {
        poem_compiler::spatial::layer::single_token(&trace, g)
    });
    let prog = match prog {
        Ok(p) => p,
        Err(e) => return println!("{name}\tLOWER\t{}\t{ops}", short(&e)),
    };
    let bodies = prog.bodies.len();
    let shape = match target::shape(&trace) {
        Ok(s) => s,
        Err(e) => return println!("{name}\tSHAPE\t{}\tbodies={bodies}\t{ops}", short(&e)),
    };
    if let Err(e) = kernels_cerebras::compose::refuse(&shape) {
        return println!("{name}\tREFUSE\t{}\tbodies={bodies}", short(&e));
    }
    let kernels = kernels_cerebras::kernels::KernelSet::of_shape(&shape).name();
    if !deep {
        return println!("{name}\tSHAPE-OK\tkernels={kernels} bodies={bodies}");
    }
    match target::plan(&trace, &Ask::default()) {
        Ok(p) => println!(
            "{name}\tPLAN-OK\tkernels={kernels} preset={} bodies={bodies} {:.1}s",
            p.preset.name,
            t0.elapsed().as_secs_f64()
        ),
        Err(e) => println!(
            "{name}\tPLAN\tkernels={kernels} {}\t{:.1}s",
            short(&e),
            t0.elapsed().as_secs_f64()
        ),
    }
}

/// Whether a node guarded by `g` runs under the switch `name` = `value` (every other fact free).
fn under(trace: &Trace, g: &poem_ir::Guard, name: &str, value: &str) -> bool {
    let (mut word, mut free) = (0u64, Vec::new());
    for bit in g.referenced_bits() {
        match trace.facts.table.get(usize::from(bit)) {
            Some(poem_ir::Fact::Switch { name: n, value: v }) if n == name => {
                if v == value {
                    word |= 1 << bit;
                }
            }
            _ => free.push(bit),
        }
    }
    (0..1u64 << free.len()).any(|a| {
        let w = free
            .iter()
            .enumerate()
            .filter(|(i, _)| a & (1 << i) != 0)
            .fold(word, |w, (_, b)| w | 1 << b);
        g.holds(w)
    })
}

/// `READINGS=1 survey =<name>`: the ops each value of the `reading` switch runs, counted.
fn readings(d: &Deployment) {
    let trace = match d.try_trace(Platform::Cerebras) {
        Ok(t) => t,
        Err(e) => return println!("{}\tTRACE\t{}", d.name, short(&e.0)),
    };
    let values: Vec<String> = trace
        .facts
        .table
        .iter()
        .filter_map(|f| match f {
            poem_ir::Fact::Switch { name, value } if name == "reading" => Some(value.clone()),
            _ => None,
        })
        .collect();
    for v in values {
        let mut ops: BTreeMap<&str, usize> = BTreeMap::new();
        for n in &trace.nodes {
            if under(&trace, &n.guard, "reading", &v) {
                *ops.entry(n.op.name()).or_insert(0) += 1;
            }
        }
        let total: usize = ops.values().sum();
        let list = ops
            .iter()
            .map(|(k, c)| format!("{k}x{c}"))
            .collect::<Vec<_>>()
            .join(",");
        println!("{}\treading={v}\tnodes={total}\t{list}", d.name);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let deep = std::env::var("DEEP").is_ok();
    let catalog = repository();
    // `=<name>`: that deployment exactly, listed or not
    let exact: Vec<&String> = args.iter().filter(|a| a.starts_with('=')).collect();
    if !exact.is_empty() {
        for n in exact {
            match catalog.parse(&n[1..]) {
                Some(d) if std::env::var("READINGS").is_ok() => readings(&d),
                Some(d) => survey(&d, deep),
                None => println!("{}\tPARSE\tno such deployment", &n[1..]),
            }
        }
        return;
    }
    let list = std::env::var("LIST").is_ok();
    let mut seen = std::collections::BTreeSet::new();
    let mut models = std::collections::BTreeSet::new();
    for d in catalog.deployments() {
        let n = &d.name;
        if list {
            if models.insert((d.package.name().to_string(), d.model.id.clone())) {
                println!("{}\t{}\t{n}", d.package.name(), d.model.id);
            }
            continue;
        }
        if !d.deploy.parts.is_empty() || d.deploy.drafter.is_some() {
            continue;
        }
        // a model's bf16 deployment, else (no bf16 one: a quantized release) its first
        let bf16 = n.ends_with("-bf16-kv-bf16");
        if !bf16 && std::env::var("ANY").is_err() {
            continue;
        }
        if !args.is_empty() && !args.iter().any(|a| n.contains(a.as_str())) {
            continue;
        }
        if !seen.insert(n.clone()) {
            continue;
        }
        survey(d, deep);
    }
}
