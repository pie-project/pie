//! Runs of repeating layers in a trace: the layer stack a fire program can
//! emit once, as a loop body, instead of once per layer.
//!
//! A program that spells every layer out carries every layer's code (each
//! Mosaic kernel once a layer), and libtpu holds a program's code on the
//! host for as long as the program is loaded: qwen3-0.6b's 28 layers cost
//! ~1.5 MB of code a layer in a prefill program, ~2.7x that in host RAM.
//!
//! Each layer's nodes are written as a signature: the ops in order, every
//! value named by its role instead of its id: a value the layer defines by
//! its place among the layer's own definitions, a weight by its type, a
//! cache by its type, a value from outside the layer that every layer reads
//! (positions, plans, masks) by its id, and the rest (what the layer before
//! hands on) by their order of first reading; an op naming its own layer
//! (a per-layer table row) names the counter. Layers whose signatures agree
//! run the same body; a run may also repeat with a period (sliding and
//! global attention alternating, a hybrid's recurrent and attention layers).

use std::collections::HashMap;
use std::fmt::Write as _;

use poem_ir::{Def, Trace, ValueId};

/// `repeats` passes of a body of `period` layers, from layer `first`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Run {
    pub first: u32,
    pub period: u32,
    pub repeats: u32,
}

impl Run {
    /// The layers the run covers.
    #[must_use]
    pub fn layers(&self) -> u32 {
        self.period * self.repeats
    }
}

/// The layer count the trace's nodes name (one past the highest).
#[must_use]
pub fn layer_count(trace: &Trace) -> u32 {
    (trace.nodes.iter())
        .filter_map(|n| n.layer)
        .max()
        .map_or(0, |l| l + 1)
}

/// Layer `layer`'s signature: what it computes, with no value's id but the
/// ones every layer reads alike.
#[must_use]
pub fn signature(trace: &Trace, layer: u32, shared: &[ValueId]) -> String {
    let nodes: Vec<_> = (trace.nodes.iter())
        .filter(|n| n.layer == Some(layer))
        .collect();
    let mut own: HashMap<u32, usize> = HashMap::new();
    for n in &nodes {
        let mut outs = Vec::new();
        poem_ir::Operands::outputs(&n.op, &mut outs);
        for v in outs {
            let next = own.len();
            own.entry(v.0).or_insert(next);
        }
    }
    // An accumulator a layer adds to (ScoreCapture's `into`) is carried
    // from the layer before; the first layer starts it (`into: None`).
    // Either way it is the layer's one accumulator, so both read `A`.
    let mut accumulators: Vec<u32> = Vec::new();
    for n in &nodes {
        let text = format!("{:?}", n.op);
        let mut rest = text.as_str();
        while let Some(at) = rest.find(ACCUMULATED) {
            let tail = &rest[at + ACCUMULATED.len()..];
            let end = tail.find(')').unwrap_or(tail.len());
            if let Ok(id) = tail[..end].parse::<u32>()
                && !own.contains_key(&id)
            {
                accumulators.push(id);
            }
            rest = &tail[end.min(tail.len())..];
        }
    }
    let mut carried: HashMap<u32, usize> = HashMap::new();
    let mut out = String::new();
    for n in &nodes {
        // An op naming its own layer (a per-layer row of a table, PLE) names
        // the loop's counter in a body: `@`.
        let text = format!("{:?}|{:?}", n.op, n.guard)
            .replace("into: None", "into: Some(A)")
            .replace(&format!("layer: {layer},"), "layer: @,")
            .replace(&format!("layer: {layer} }}"), "layer: @ }");
        let mut rest = text.as_str();
        while let Some(at) = rest.find("ValueId(") {
            out.push_str(&rest[..at]);
            let tail = &rest[at + "ValueId(".len()..];
            let end = tail.find(')').unwrap_or(tail.len());
            let id: u32 = tail[..end].parse().unwrap_or(u32::MAX);
            if accumulators.contains(&id) {
                out.push('A');
            } else {
                let _ = write!(out, "{}", role(trace, id, &own, &mut carried, shared));
            }
            rest = &tail[(end + 1).min(tail.len())..];
        }
        out.push_str(rest);
        out.push('\n');
    }
    out
}

/// How an op names the accumulator it adds to.
const ACCUMULATED: &str = "into: Some(ValueId(";

fn role(
    trace: &Trace,
    id: u32,
    own: &HashMap<u32, usize>,
    carried: &mut HashMap<u32, usize>,
    shared: &[ValueId],
) -> String {
    if let Some(at) = own.get(&id) {
        return format!("L{at}");
    }
    if shared.contains(&ValueId(id)) {
        return format!("G{id}");
    }
    let Some(decl) = trace.values.get(id as usize) else {
        return format!("?{id}");
    };
    match &decl.def {
        Def::Weight(_) => format!("W[{:?}]", decl.ty),
        Def::Cache(_) => format!("C[{:?}]", decl.ty),
        _ => {
            let next = carried.len();
            format!("X{}", carried.entry(id).or_insert(next))
        }
    }
}

/// The values two or more layers read that no layer defines and that are
/// neither weights nor caches: the fire's own inputs every layer shares.
#[must_use]
pub fn shared_inputs(trace: &Trace) -> Vec<ValueId> {
    let mut readers: HashMap<u32, std::collections::BTreeSet<u32>> = HashMap::new();
    let mut defined: HashMap<u32, u32> = HashMap::new();
    for n in &trace.nodes {
        let Some(layer) = n.layer else { continue };
        let mut outs = Vec::new();
        poem_ir::Operands::outputs(&n.op, &mut outs);
        for v in outs {
            defined.insert(v.0, layer);
        }
    }
    for n in &trace.nodes {
        let Some(layer) = n.layer else { continue };
        let mut ins = Vec::new();
        poem_ir::Operands::inputs(&n.op, &mut ins);
        for v in ins {
            readers.entry(v.0).or_default().insert(layer);
        }
    }
    let mut shared: Vec<ValueId> = (readers.into_iter())
        .filter(|(id, layers)| {
            layers.len() > 1
                && !defined.contains_key(id)
                && trace
                    .values
                    .get(*id as usize)
                    .is_some_and(|d| !matches!(d.def, Def::Weight(_) | Def::Cache(_)))
        })
        .map(|(id, _)| ValueId(id))
        .collect();
    shared.sort_unstable();
    shared
}

/// The trace's layers as runs: each the longest stretch from its first
/// layer that repeats a body of the shortest period that repeats at least
/// twice; a layer that repeats nothing is a run of one.
#[must_use]
pub fn runs(trace: &Trace) -> Vec<Run> {
    let count = layer_count(trace);
    let shared = shared_inputs(trace);
    let sigs: Vec<String> = (0..count).map(|l| signature(trace, l, &shared)).collect();
    let mut runs = Vec::new();
    let mut first = 0u32;
    while first < count {
        let mut best = Run {
            first,
            period: 1,
            repeats: 1,
        };
        for period in 1..=(count - first) / 2 {
            let mut repeats = 1;
            while first + (repeats + 1) * period <= count
                && (0..period).all(|k| {
                    sigs[(first + k) as usize] == sigs[(first + repeats * period + k) as usize]
                })
            {
                repeats += 1;
            }
            if repeats > 1 && period * repeats > best.layers() {
                best = Run {
                    first,
                    period,
                    repeats,
                };
            }
        }
        runs.push(best);
        first += best.layers();
    }
    runs
}

/// Layer `layer`'s weights (trace params), in the order its nodes first
/// read them.
#[must_use]
pub fn weights_of(trace: &Trace, layer: u32) -> Vec<u32> {
    let mut seen = Vec::new();
    for n in trace.nodes.iter().filter(|n| n.layer == Some(layer)) {
        let mut ins = Vec::new();
        poem_ir::Operands::inputs(&n.op, &mut ins);
        for v in ins {
            if let Some(Def::Weight(param)) = trace.values.get(v.0 as usize).map(|d| &d.def)
                && !seen.contains(param)
            {
                seen.push(*param);
            }
        }
    }
    seen
}

/// A run's weights as stacks: for each weight its body reads (by place in
/// the body), the params each pass of the run reads there, pass by pass. A
/// weight every pass reads alike (one param) is no stack.
#[must_use]
pub fn stacks(trace: &Trace, runs: &[Run]) -> Vec<Vec<u32>> {
    let mut stacks = Vec::new();
    for run in runs.iter().filter(|r| r.repeats > 1) {
        for k in 0..run.period {
            let passes: Vec<Vec<u32>> = (0..run.repeats)
                .map(|r| weights_of(trace, run.first + r * run.period + k))
                .collect();
            let slots = passes[0].len();
            if passes.iter().any(|p| p.len() != slots) {
                continue;
            }
            for slot in 0..slots {
                let params: Vec<u32> = passes.iter().map(|p| p[slot]).collect();
                if params.iter().all(|&p| p == params[0]) {
                    continue;
                }
                stacks.push(params);
            }
        }
    }
    stacks
}
