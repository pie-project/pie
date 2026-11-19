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

/// Where adapter bank stacks are numbered in a [`Plan`]'s steps, past every
/// weight stack.
pub const BANK_STACKS: u32 = 1 << 20;

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
    let nodes: Vec<&poem_ir::Node> = (trace.nodes.iter())
        .filter(|n| n.layer == Some(layer))
        .collect();
    nodes_signature(trace, &nodes, layer, 1, shared)
}

/// `nodes`' signature, every value by its role and every layer from `base`
/// to `base + span` named relative to `base` (a layer the nodes name of
/// their own, a per-layer table row, is the loop's counter in a body).
fn nodes_signature(
    trace: &Trace,
    nodes: &[&poem_ir::Node],
    base: u32,
    span: u32,
    shared: &[ValueId],
) -> String {
    let mut own: HashMap<u32, usize> = HashMap::new();
    for n in nodes {
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
    for n in nodes {
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
    for n in nodes {
        let mut text = format!(
            "{:?}|{:?}|{:?}",
            n.op,
            n.guard,
            n.layer.map(|l| l.wrapping_sub(base))
        )
        .replace("into: None", "into: Some(A)");
        for l in base..base + span {
            text = text
                .replace(&format!("layer: {l},"), &format!("layer: @{},", l - base))
                .replace(
                    &format!("layer: {l} }}"),
                    &format!("layer: @{} }}", l - base),
                );
        }
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

/// A run of the compiled template's regions that repeats: `repeats` blocks
/// of `regions` regions from region `start`, block `k` computing what block
/// 0 computes `k * layers` layers on. A block may straddle a layer boundary
/// (the compiler fuses a layer's last ops with the next layer's first):
/// the body is whatever repeats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Block {
    pub start: usize,
    pub regions: usize,
    pub repeats: u32,
    pub layers: u32,
    /// The lowest layer block 0's nodes name.
    pub base: u32,
}

/// A block of regions' signature: each region's lowering, mask and phase,
/// then its nodes as [`signature`] writes a layer's, a layer named relative
/// to `base`.
fn block_signature(
    trace: &Trace,
    compiled: &poem_compiler::CompiledModel,
    regions: std::ops::Range<usize>,
    base: u32,
    span: u32,
    shared: &[ValueId],
) -> String {
    let template = compiled.template();
    let nodes: Vec<&poem_ir::Node> = (regions.clone())
        .flat_map(|r| template[r].nodes.clone())
        .map(|n| &trace.nodes[n as usize])
        .collect();
    let mut out = String::new();
    for r in regions {
        let region = &template[r];
        let _ = write!(
            out,
            "[{:?} {:?} {:?} {:?} {} {}]",
            region.lowering,
            region.mask,
            region.phase,
            region.axis,
            region.nodes.len(),
            region.collective
        );
    }
    out.push('\n');
    out.push_str(&nodes_signature(trace, &nodes, base, span, shared));
    out
}

/// The lowest and highest layer `regions`' nodes name.
fn layers_of(
    trace: &Trace,
    compiled: &poem_compiler::CompiledModel,
    regions: std::ops::Range<usize>,
) -> Option<(u32, u32)> {
    let template = compiled.template();
    let mut span: Option<(u32, u32)> = None;
    for r in regions {
        for n in template[r].nodes.clone() {
            if let Some(l) = trace.nodes[n as usize].layer {
                span = Some(span.map_or((l, l), |(a, b)| (a.min(l), b.max(l))));
            }
        }
    }
    span
}

/// The template's repeating blocks: from each region, the run of blocks of
/// the size that covers the most regions, two blocks at least, each block
/// naming layers and every block the first's `layers` layers on.
#[must_use]
pub fn blocks(trace: &Trace, compiled: &poem_compiler::CompiledModel) -> Vec<Block> {
    let shared = shared_inputs(trace);
    let count = compiled.template().len();
    let mut found = Vec::new();
    let mut start = 0;
    while start < count {
        let mut best: Option<Block> = None;
        for size in 1..=(count - start) / 2 {
            let Some((base, top)) = layers_of(trace, compiled, start..start + size) else {
                continue;
            };
            let Some((next, _)) = layers_of(trace, compiled, start + size..start + 2 * size) else {
                continue;
            };
            if next <= base {
                continue;
            }
            let step = next - base;
            let span = top - base + 1;
            let first = block_signature(trace, compiled, start..start + size, base, span, &shared);
            let mut repeats = 1u32;
            while start + (repeats as usize + 1) * size <= count {
                let lo = start + repeats as usize * size;
                let b = base + repeats * step;
                if block_signature(trace, compiled, lo..lo + size, b, span, &shared) != first {
                    break;
                }
                repeats += 1;
            }
            if repeats > 1
                && best.is_none_or(|b| size * repeats as usize > b.regions * b.repeats as usize)
            {
                best = Some(Block {
                    start,
                    regions: size,
                    repeats,
                    layers: step,
                    base,
                });
            }
        }
        match best {
            Some(block) => {
                start += block.regions * block.repeats as usize;
                found.push(block);
            }
            None => start += 1,
        }
    }
    found
}

/// What the load rolls: each repeating block of regions and its weights'
/// stack passes, or why it stays spelled out.
#[derive(Debug, Clone, Default)]
pub struct Plan {
    /// Each block, with the passes a turn steps each weight stack its body
    /// reads (by stack), or why the block does not roll.
    pub blocks: Vec<(Block, Result<std::collections::BTreeMap<u32, u32>, String>)>,
    pub layers: u32,
}

impl Plan {
    /// The template's repeating blocks, each checked: every weight block
    /// `k` reads is the stack pass block 0 reads at that place, `k` passes
    /// on (`member`: param -> (stack, pass)).
    pub fn of(
        trace: &Trace,
        compiled: &poem_compiler::CompiledModel,
        member: &dyn Fn(u32) -> Option<(u32, u32)>,
        streams: &dyn Fn(u32) -> bool,
        kv: &dyn Fn(usize) -> Option<KvLaid>,
    ) -> Plan {
        let layers = layer_count(trace);
        let blocks = (blocks(trace, compiled).into_iter())
            .map(|block| {
                let weights = |k: u32| -> Vec<u32> {
                    let lo = block.start + k as usize * block.regions;
                    region_weights(trace, compiled, lo..lo + block.regions)
                };
                let first = weights(0);
                let lo = block.start;
                let reads_state = (lo..lo + block.regions)
                    .flat_map(|r| compiled.template()[r].nodes.clone())
                    .any(|n| {
                        let mut ins = Vec::new();
                        poem_ir::Operands::inputs(&trace.nodes[n as usize].op, &mut ins);
                        ins.iter().any(|v| {
                            matches!(
                                trace.values.get(v.0 as usize).map(|d| &d.def),
                                Some(Def::Cache(row))
                                    if matches!(trace.caches.get(*row as usize), Some(poem_ir::CacheRow::State { .. }))
                            )
                        })
                    });
                let mut checked: Result<(), String> = if reads_state {
                    Err("its body reads recurrent state rows, which a body does not yet address \
                         by its counter"
                        .to_string())
                } else {
                    Ok(())
                };
                let mut step: Vec<Option<u32>> = vec![None; first.len()];
                'blocks: for k in 0..block.repeats {
                    if checked.is_err() {
                        break;
                    }
                    let at = weights(k);
                    for (j, (&p0, &pk)) in first.iter().zip(&at).enumerate() {
                        if p0 == pk {
                            continue;
                        }
                        let (Some((s0, a0)), Some((sk, ak))) = (member(p0), member(pk)) else {
                            let at = if member(p0).is_none() { p0 } else { pk };
                            let lone = &trace.params[at as usize];
                            checked = Err(if streams(at) {
                                format!(
                                    "weight `{}` streams through the pool: a fire whose layers' \
                                     weights are not all resident runs each layer as its own \
                                     segment, cut before its first read",
                                    lone.name
                                )
                            } else if lone.source == poem_ir::ParamSource::Registered {
                                format!(
                                    "weight `{}` is an adapter bank, read through the pool's \
                                     adapter slots, which a body does not yet address by its \
                                     counter",
                                    lone.name
                                )
                            } else {
                                format!(
                                    "weight `{}` lands in a buffer of its own (it streams, or \
                                     lands with a partner plane or a tied copy)",
                                    lone.name
                                )
                            });
                            break 'blocks;
                        };
                        let d = ak.checked_sub(a0).filter(|d| d % k == 0).map(|d| d / k);
                        let agrees =
                            s0 == sk && d.is_some() && step[j].is_none_or(|s| Some(s) == d);
                        if !agrees {
                            checked = Err(format!(
                                "weight `{}` is not its stack's pass a whole step on from \
                                 block 0's `{}`",
                                trace.params[pk as usize].name, trace.params[p0 as usize].name
                            ));
                            break 'blocks;
                        }
                        step[j] = d;
                    }
                }
                if checked.is_ok() {
                    checked = kv_steps(trace, compiled, &block, kv);
                }
                let checked = checked.map(|()| {
                    (first.iter().zip(&step))
                        .filter_map(|(&p, step)| Some((member(p)?.0, (*step)?)))
                        .collect()
                });
                (block, checked)
            })
            .collect();
        Plan { blocks, layers }
    }

    /// The layers in blocks that roll.
    #[must_use]
    pub fn rolled(&self) -> u32 {
        (self.blocks.iter())
            .filter(|(_, c)| c.is_ok())
            .map(|(b, _)| b.layers * b.repeats)
            .sum()
    }

    /// The load line: `rolled N of M layers`, then each block that stays
    /// spelled out and why.
    #[must_use]
    pub fn line(&self) -> String {
        let mut line = format!("rolled {} of {} layers", self.rolled(), self.layers);
        for (block, checked) in &self.blocks {
            if let Err(why) = checked {
                let _ = write!(
                    line,
                    "; layers {}..{} unrolled: {why}",
                    block.base,
                    block.base + block.layers * block.repeats
                );
            }
        }
        line
    }
}

/// A kv cache row's planes as the load lays them: each plane's view (one
/// identity a view of the pool, its row order and its form) and its page
/// map's stride and offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvLaid {
    pub views: [u64; 2],
    pub maps: [(i64, i64); 2],
}

/// Whether every cache row `block`'s passes read lies where a loop can
/// address it by its counter: in block 0's row's views, at its strides, its
/// offsets a whole step on each pass.
fn kv_steps(
    trace: &Trace,
    compiled: &poem_compiler::CompiledModel,
    block: &Block,
    kv: &dyn Fn(usize) -> Option<KvLaid>,
) -> Result<(), String> {
    let template = compiled.template();
    let nodes = |k: u32| -> Vec<usize> {
        let lo = block.start + k as usize * block.regions;
        (lo..lo + block.regions)
            .flat_map(|r| template[r].nodes.clone())
            .map(|n| n as usize)
            .collect()
    };
    let passes: Vec<Vec<usize>> = (0..block.repeats).map(nodes).collect();
    let inputs = |n: usize| {
        let mut ins = Vec::new();
        poem_ir::Operands::inputs(&trace.nodes[n].op, &mut ins);
        ins
    };
    let row_of = |v: ValueId| match trace.values.get(v.0 as usize).map(|d| &d.def) {
        Some(Def::Cache(row)) => Some(*row as usize),
        _ => None,
    };
    for (j, &n0) in passes[0].iter().enumerate() {
        for (at, &v) in inputs(n0).iter().enumerate() {
            let Some(row) = row_of(v) else { continue };
            let Some(first) = kv(row) else { continue };
            let rows: Vec<Option<usize>> = (passes.iter())
                .map(|p| inputs(p[j]).get(at).copied().and_then(row_of))
                .collect();
            let second = rows[1].and_then(kv);
            for (k, r) in rows.iter().enumerate() {
                let laid = r.and_then(kv);
                let step = |plane: usize| second.map(|s| s.maps[plane].1 - first.maps[plane].1);
                let whole = laid.is_some_and(|l| {
                    l.views == first.views
                        && (0..2).all(|p| {
                            l.maps[p].0 == first.maps[p].0
                                && step(p)
                                    .is_some_and(|d| l.maps[p].1 == first.maps[p].1 + k as i64 * d)
                        })
                });
                if !whole {
                    return Err(format!(
                        "cache row {}'s pages do not lie a whole step on from row {row}'s, in \
                         its views at its strides",
                        r.map_or_else(|| "?".to_string(), |r| r.to_string())
                    ));
                }
            }
        }
    }
    Ok(())
}

/// The weights `regions`' nodes read, in the order they first read them.
fn region_weights(
    trace: &Trace,
    compiled: &poem_compiler::CompiledModel,
    regions: std::ops::Range<usize>,
) -> Vec<u32> {
    let template = compiled.template();
    let mut seen = Vec::new();
    for r in regions {
        for n in template[r].nodes.clone() {
            let mut ins = Vec::new();
            poem_ir::Operands::inputs(&trace.nodes[n as usize].op, &mut ins);
            for v in ins {
                if let Some(Def::Weight(param)) = trace.values.get(v.0 as usize).map(|d| &d.def)
                    && !seen.contains(param)
                {
                    seen.push(*param);
                }
            }
        }
    }
    seen
}
