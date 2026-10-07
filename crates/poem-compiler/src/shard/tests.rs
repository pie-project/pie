use std::collections::BTreeMap;

use poem_ir::{Def, Operands, Platform, Trace, ValueId};

use crate::tree;

/// `trace` with its values numbered by where the nodes first touch them, so
/// two traces that differ only in numbering compare equal.
fn canonical(trace: &Trace) -> Trace {
    let mut order: BTreeMap<ValueId, ValueId> = BTreeMap::new();
    let next = |v: ValueId, order: &mut BTreeMap<ValueId, ValueId>| {
        let n = order.len() as u32;
        order.entry(v).or_insert(ValueId(n));
    };
    let mut ids = Vec::new();
    for node in &trace.nodes {
        ids.clear();
        node.op.inputs(&mut ids);
        node.op.outputs(&mut ids);
        for v in &ids {
            next(*v, &mut order);
        }
    }
    for seam in &trace.seams {
        for v in &seam.values {
            next(*v, &mut order);
        }
    }
    for v in 0..trace.values.len() {
        next(ValueId(v as u32), &mut order);
    }
    let f = |v: ValueId| order[&v];
    let mut values = vec![None; trace.values.len()];
    for (v, decl) in trace.values.iter().enumerate() {
        let mut decl = decl.clone();
        if let Def::Merge(arms) = &mut decl.def {
            for (arm, _) in arms {
                *arm = f(*arm);
            }
        }
        values[f(ValueId(v as u32)).0 as usize] = Some(decl);
    }
    let mut out = trace.clone();
    out.values = values.into_iter().map(Option::unwrap).collect();
    for node in &mut out.nodes {
        node.op = tree::remap(&node.op, &f);
    }
    for seam in &mut out.seams {
        for v in &mut seam.values {
            *v = f(*v);
        }
    }
    out
}

fn first_difference(a: &Trace, b: &Trace) -> String {
    if a.params != b.params {
        let i = a.params.iter().zip(&b.params).position(|(x, y)| x != y);
        return format!(
            "params at {i:?}: {:?} vs {:?}",
            i.map(|i| &a.params[i]),
            i.map(|i| &b.params[i])
        );
    }
    if a.caches != b.caches {
        let i = a.caches.iter().zip(&b.caches).position(|(x, y)| x != y);
        return format!(
            "caches at {i:?}: {:?} vs {:?}",
            i.map(|i| &a.caches[i]),
            i.map(|i| &b.caches[i])
        );
    }
    if a.nodes.len() != b.nodes.len() {
        let i = a
            .nodes
            .iter()
            .zip(&b.nodes)
            .position(|(x, y)| x.op.name() != y.op.name());
        return format!(
            "{} nodes vs {}; first op differing at {i:?}: {:?} vs {:?}",
            a.nodes.len(),
            b.nodes.len(),
            i.map(|i| &a.nodes[i].op),
            i.map(|i| &b.nodes[i].op)
        );
    }
    if let Some(i) = a.nodes.iter().zip(&b.nodes).position(|(x, y)| x != y) {
        return format!("node {i}: {:?}\n   vs {:?}", a.nodes[i], b.nodes[i]);
    }
    if let Some(i) = a.values.iter().zip(&b.values).position(|(x, y)| x != y) {
        return format!("value {i}: {:?} vs {:?}", a.values[i], b.values[i]);
    }
    if a.values.len() != b.values.len() {
        return format!("{} values vs {}", a.values.len(), b.values.len());
    }
    let i = a.seams.iter().zip(&b.seams).position(|(x, y)| x != y);
    format!(
        "seam {i:?}: {:?} vs {:?} ({} vs {} seams)",
        i.map(|i| &a.seams[i]),
        i.map(|i| &b.seams[i]),
        a.seams.len(),
        b.seams.len()
    )
}

/// Hand-split traces the pass does not reproduce, and why.
const KNOWN: &[(&str, &str)] = &[
    (
        "kimik3-mini-",
        "the hand split norms the latent experts' partial sum before reducing it",
    ),
    (
        "minimax-h3-",
        "the hand split halves heads whose weights it never splits and reduces nothing",
    ),
    (
        "dsv4-base-",
        "the pool gathers entries from a kv cache the ranks split, so each rank \
         pools different heads",
    ),
];

/// Every catalog model that ships a hand-split trace splits, through this
/// pass, into exactly that trace, but for the ones `KNOWN` says differ.
#[test]
fn every_hand_split_trace_is_what_the_pass_splits() {
    let only = std::env::var("ONLY").ok();
    let mut failed = Vec::new();
    let mut checked = 0;
    for sku in models::skus().filter(|sku| sku.recipe.tp > 1) {
        if only.as_deref().is_some_and(|o| !sku.name.contains(o)) {
            continue;
        }
        let Some(base) = sku
            .name
            .rsplit_once("-tp")
            .and_then(|(base, _)| models::sku(base))
        else {
            continue;
        };
        checked += 1;
        let known = KNOWN
            .iter()
            .find(|(prefix, _)| sku.name.starts_with(prefix));
        let want = (sku.trace)(Platform::Cuda);
        let same = super::shard((base.trace)(Platform::Cuda), sku.recipe.tp).map(|got| {
            let (mut got, want) = (canonical(&got), canonical(&want));
            got.name.clone_from(&want.name);
            (got == want)
                .then_some(())
                .ok_or_else(|| first_difference(&got, &want))
        });
        match (same, known) {
            (Ok(Ok(())), None) | (Ok(Err(_)) | Err(_), Some(_)) => {}
            (Ok(Ok(())), Some((_, why))) => {
                failed.push(format!("{}: matches, though KNOWN says {why}", sku.name));
            }
            (Ok(Err(diff)), None) => failed.push(format!("{}: {diff}", sku.name)),
            (Err(e), None) => failed.push(format!("{}: {e}", sku.name)),
        }
    }
    assert!(
        failed.is_empty(),
        "{checked} checked, {} differ:\n{}",
        failed.len(),
        failed.join("\n")
    );
}
