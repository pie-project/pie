//! Independent branches that meet at one op, laid side by side so the stream
//! pass can run them at once.
//!
//! A join is an op whose inputs come from two or more chains that share no
//! op since they parted: the shared expert and the routed experts a MoE sums,
//! the query, key/value and index chains an attention reads. [`split`]
//! reorders the trace so each such chain is one contiguous run just ahead of
//! its join (the ops between that read none of them first).
//!
//! Order is kept: an op reads what it read and writes what it wrote, an op
//! touching a cache stays after every earlier op touching that cache, one
//! writing into another value's buffer after every earlier reader of it, and
//! nothing crosses a collective.

use std::collections::BTreeSet;

use poem_ir::{Def, Operands, Operation, Trace, ValueId};

/// How far back from a join its branches are looked for, in ops.
const WINDOW: usize = 256;

/// The ops each op must follow (its direct predecessors), over the whole
/// trace: producers of what it reads (through merges), earlier ops touching
/// a cache it touches, earlier readers of a buffer it writes into, and the
/// last collective.
fn predecessors(trace: &Trace) -> Vec<Vec<u32>> {
    let n = trace.nodes.len();
    let mut producer: Vec<Vec<u32>> = vec![Vec::new(); trace.values.len()];
    for (v, decl) in trace.values.iter().enumerate() {
        if let Def::Op(i) = decl.def
            && (i as usize) < n
        {
            producer[v].push(i);
        }
    }
    // A merge reads every arm's producer.
    for v in 0..trace.values.len() {
        if let Def::Merge(arms) = &trace.values[v].def {
            let mut all = Vec::new();
            for (arm, _) in arms {
                all.extend(producer[arm.0 as usize].iter().copied());
            }
            producer[v] = all;
        }
    }
    let is_cache = |v: ValueId| matches!(trace.values[v.0 as usize].def, Def::Cache(_));

    let mut preds: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut cache_touch: std::collections::BTreeMap<u32, Vec<u32>> = Default::default();
    let mut readers: Vec<Vec<u32>> = vec![Vec::new(); trace.values.len()];
    let mut last_collective: Option<u32> = None;
    let (mut ins, mut aliases) = (Vec::new(), Vec::new());
    for (j, node) in trace.nodes.iter().enumerate() {
        let j = j as u32;
        let p = &mut preds[j as usize];
        ins.clear();
        node.op.inputs(&mut ins);
        for &v in &ins {
            if is_cache(v) {
                let touched = cache_touch.entry(v.0).or_default();
                p.extend(touched.iter().copied());
                touched.push(j);
            } else {
                p.extend(producer[v.0 as usize].iter().copied().filter(|&i| i < j));
                readers[v.0 as usize].push(j);
            }
        }
        aliases.clear();
        node.op.aliases(&mut aliases);
        for &(_, into) in &aliases {
            p.extend(readers[into.0 as usize].iter().copied().filter(|&i| i < j));
            p.extend(producer[into.0 as usize].iter().copied().filter(|&i| i < j));
        }
        if let Some(c) = last_collective {
            p.push(c);
        }
        if matches!(node.op, Operation::Collective(_)) {
            // A collective follows everything before it.
            p.extend(j.saturating_sub(WINDOW as u32)..j);
            last_collective = Some(j);
        }
        p.sort_unstable();
        p.dedup();
    }
    preds
}

/// Whether a chain is worth a stream of its own: it multiplies.
fn heavy(trace: &Trace, nodes: &[u32]) -> bool {
    nodes.iter().any(|&i| {
        let name = trace.nodes[i as usize].op.name();
        name.contains("matmul")
            || name.starts_with("attention.pool_lse")
            || name.starts_with("linear.moe")
    })
}

/// Everything `from` follows within `[lo, from]`, itself included.
fn closure(preds: &[Vec<u32>], from: u32, lo: u32) -> BTreeSet<u32> {
    let mut seen = BTreeSet::new();
    let mut stack = vec![from];
    while let Some(i) = stack.pop() {
        if i < lo || !seen.insert(i) {
            continue;
        }
        stack.extend(preds[i as usize].iter().copied().filter(|&p| p >= lo));
    }
    seen
}

/// The branches meeting at `join`, and the ops between them that read none:
/// `(branches, between)`, each sorted; `None` where fewer than two chains of
/// weight meet there or an op between reads one.
fn branches_at(trace: &Trace, preds: &[Vec<u32>], join: u32) -> Option<(Vec<Vec<u32>>, Vec<u32>)> {
    let lo = join.saturating_sub(WINDOW as u32);
    let direct: Vec<u32> = preds[join as usize]
        .iter()
        .copied()
        .filter(|&p| p >= lo)
        .collect();
    if direct.len() < 2 {
        return None;
    }
    let closures: Vec<BTreeSet<u32>> = direct.iter().map(|&p| closure(preds, p, lo)).collect();
    let mut branches: Vec<Vec<u32>> = Vec::new();
    for (k, c) in closures.iter().enumerate() {
        let own: Vec<u32> = c
            .iter()
            .copied()
            .filter(|i| {
                closures
                    .iter()
                    .enumerate()
                    .all(|(q, other)| q == k || !other.contains(i))
            })
            .collect();
        // A chain of one op is left in place: it is one launch either way,
        // and a fork around it costs more than it overlaps.
        if own.len() >= 2 && heavy(trace, &own) {
            branches.push(own);
        }
    }
    // One chain may feed the join twice (two of its ops): keep each once.
    branches.sort();
    branches.dedup();
    if branches.len() < 2 {
        return None;
    }
    let in_branch: BTreeSet<u32> = branches.iter().flatten().copied().collect();
    let start = *in_branch.first()?;
    let mut between = Vec::new();
    for i in start..join {
        if in_branch.contains(&i) {
            continue;
        }
        // An op between that follows a branch op cannot move ahead of it.
        if preds[i as usize].iter().any(|p| in_branch.contains(p))
            || closure(preds, i, start)
                .iter()
                .any(|p| in_branch.contains(p))
        {
            return None;
        }
        if matches!(trace.nodes[i as usize].op, Operation::Collective(_)) {
            return None;
        }
        between.push(i);
    }
    // Branches in the order they start.
    branches.sort_by_key(|b| b[0]);
    Some((branches, between))
}

/// The trace with each join's branches laid contiguous, in the order they
/// started, just ahead of it; the ops between them that read none first.
#[must_use]
pub fn split(mut trace: Trace) -> Trace {
    let mut at = 0usize;
    // Each pass lays out the first join from `at` that needs it; the ops from
    // that join on keep their places, so the next pass starts past it.
    while let Some((join, order)) = next_layout(&trace, at) {
        trace = permute(trace, &order);
        at = join + 1;
    }
    trace
}

/// The first join at or after `at` whose branches are not yet laid out, and
/// the order that lays them.
fn next_layout(trace: &Trace, at: usize) -> Option<(usize, Vec<u32>)> {
    let preds = predecessors(trace);
    (at..trace.nodes.len()).find_map(|join| {
        let (branches, between) = branches_at(trace, &preds, join as u32)?;
        let start = branches.iter().map(|b| b[0]).min().unwrap_or(join as u32);
        let mut order: Vec<u32> = (0..start).collect();
        order.extend(between.iter().copied());
        for b in &branches {
            order.extend(b.iter().copied());
        }
        order.extend(join as u32..trace.nodes.len() as u32);
        order
            .iter()
            .enumerate()
            .any(|(k, &i)| k as u32 != i)
            .then_some((join, order))
    })
}

/// `trace` with its ops in `order` (a permutation of their indices).
fn permute(mut trace: Trace, order: &[u32]) -> Trace {
    let mut landed = vec![0u32; order.len()];
    for (k, &i) in order.iter().enumerate() {
        landed[i as usize] = k as u32;
    }
    let mut old: Vec<Option<poem_ir::Node>> = std::mem::take(&mut trace.nodes)
        .into_iter()
        .map(Some)
        .collect();
    trace.nodes = order
        .iter()
        .map(|&i| {
            old[i as usize]
                .take()
                .expect("a permutation names each op once")
        })
        .collect();
    for decl in &mut trace.values {
        if let Def::Op(i) = &mut decl.def
            && let Some(&to) = landed.get(*i as usize)
        {
            *i = to;
        }
    }
    trace
}
