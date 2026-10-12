//! Finding a rule's pattern in a trace over use-def edges, and replacing each
//! match with the fused op.

use std::collections::BTreeSet;

use poem::pattern::{Attr, Template};
use poem_ir::{Def, Node, Operands, Operation, Trace, ValueId};

use super::{Match, Rule};
use crate::tree::{self, Tree};

const UNCLAIMED: u32 = u32::MAX;

/// A template compiled for search: each pattern op as a tree, what each
/// pattern value may bind, and the order the ops are matched in.
pub(super) struct Shape {
    pub(super) template: Template,
    trees: Vec<Tree>,
    kinds: Vec<Kind>,
    steps: Vec<Step>,
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Any,
    Weight,
    Cache,
    Computed,
}

/// How a pattern op finds its candidates once the ones before it are bound:
/// the op that defines a bound value is the one the trace says defines it,
/// and an op that reads a bound value is one of that value's readers.
#[derive(Clone, Copy)]
enum Step {
    Anchor,
    Defines(usize, ValueId),
    Reads(usize, ValueId),
}

impl Shape {
    pub(super) fn new(template: Template) -> Shape {
        let trace = &template.trace;
        assert!(!trace.nodes.is_empty(), "a pattern holds at least one op");
        let trees = trace.nodes.iter().map(|n| tree::of(&n.op)).collect();
        let kinds = trace
            .values
            .iter()
            .map(|decl| match decl.def {
                Def::Op(UNCLAIMED) => Kind::Any,
                Def::Op(_) => Kind::Computed,
                Def::Weight(_) => Kind::Weight,
                Def::Cache(_) => Kind::Cache,
                Def::Input(_) | Def::Merge(_) => {
                    panic!("a pattern's holes are values, weights and caches")
                }
            })
            .collect();
        let mut steps = vec![Step::Anchor];
        let mut placed = vec![false; trace.nodes.len()];
        placed[0] = true;
        let mut bound = BTreeSet::new();
        operands(&trace.nodes[0].op, &mut bound);
        while steps.len() < trace.nodes.len() {
            let next = (0..trace.nodes.len())
                .filter(|&p| !placed[p])
                .find_map(|p| {
                    let mut outs = Vec::new();
                    trace.nodes[p].op.outputs(&mut outs);
                    outs.into_iter()
                        .find(|v| bound.contains(v))
                        .map(|v| Step::Defines(p, v))
                })
                .or_else(|| {
                    (0..trace.nodes.len())
                        .filter(|&p| !placed[p])
                        .find_map(|p| {
                            let mut ins = Vec::new();
                            trace.nodes[p].op.inputs(&mut ins);
                            ins.into_iter()
                                .find(|v| bound.contains(v))
                                .map(|v| Step::Reads(p, v))
                        })
                })
                .expect("a pattern's ops are connected through the values they share");
            let (Step::Defines(p, _) | Step::Reads(p, _)) = next else {
                unreachable!("only the first op anchors")
            };
            placed[p] = true;
            operands(&trace.nodes[p].op, &mut bound);
            steps.push(next);
        }
        Shape {
            template,
            trees,
            kinds,
            steps,
        }
    }

    fn attr(&self, tree: &Tree) -> Option<usize> {
        let sentinel = match tree {
            Tree::F32(bits) => Attr::F32(*bits),
            Tree::Int(n) => Attr::U32(u32::try_from(*n).ok()?),
            _ => return None,
        };
        self.template
            .attrs
            .iter()
            .position(|(_, attr)| *attr == sentinel)
    }
}

fn operands(op: &Operation, sink: &mut BTreeSet<ValueId>) {
    let mut ids = Vec::new();
    op.inputs(&mut ids);
    op.outputs(&mut ids);
    sink.extend(ids);
}

/// Who reads each value, and the seams and merges that hold one.
struct Uses {
    readers: Vec<Vec<u32>>,
    held: Vec<bool>,
}

impl Uses {
    fn of(trace: &Trace) -> Uses {
        let mut readers = vec![Vec::new(); trace.values.len()];
        let mut held = vec![false; trace.values.len()];
        let mut ins = Vec::new();
        for (i, node) in trace.nodes.iter().enumerate() {
            ins.clear();
            node.op.inputs(&mut ins);
            for v in &ins {
                readers[v.0 as usize].push(i as u32);
            }
        }
        for decl in &trace.values {
            if let Def::Merge(arms) = &decl.def {
                for (arm, _) in arms {
                    held[arm.0 as usize] = true;
                }
            }
        }
        for seam in &trace.seams {
            for v in &seam.values {
                held[v.0 as usize] = true;
            }
        }
        Uses { readers, held }
    }
}

#[derive(Clone)]
pub(super) struct Binding {
    pub(super) values: Vec<Option<ValueId>>,
    pub(super) attrs: Vec<Option<Tree>>,
    nodes: Vec<Option<usize>>,
}

/// Every match of `rule` replaced by its fused op, scanning the trace in order
/// and taking the first alternative that matches at each op.
pub(super) fn apply(mut trace: Trace, rule: &Rule) -> Trace {
    let mut uses = Uses::of(&trace);
    let mut at = 0;
    while at < trace.nodes.len() {
        let found = rule
            .shapes
            .iter()
            .find_map(|shape| matched(&trace, &uses, rule, shape, at).map(|b| (shape, b)));
        match found {
            Some((shape, binding)) => {
                let fused = (rule.replace)(&Match {
                    trace: &trace,
                    shape,
                    binding: &binding,
                });
                debug_assert_eq!(
                    fused.name(),
                    rule.kernel,
                    "a rule forms the kernel it is listed under"
                );
                trace = rewrite(trace, &binding, fused);
                uses = Uses::of(&trace);
            }
            None => at += 1,
        }
    }
    trace
}

fn matched(trace: &Trace, uses: &Uses, rule: &Rule, shape: &Shape, at: usize) -> Option<Binding> {
    let p = &shape.template.trace;
    let start = Binding {
        values: vec![None; p.values.len()],
        attrs: vec![None; shape.template.attrs.len()],
        nodes: vec![None; p.nodes.len()],
    };
    let mut found = None;
    extend(trace, uses, shape, 0, at, start, &mut |binding| {
        let ok = escapes_nothing(uses, shape, binding)
            && keeps_order(trace, binding)
            && (rule.when)(&Match {
                trace,
                shape,
                binding,
            });
        if ok {
            found = Some(binding.clone());
        }
        ok
    });
    found
}

fn extend(
    trace: &Trace,
    uses: &Uses,
    shape: &Shape,
    step: usize,
    anchor: usize,
    binding: Binding,
    done: &mut dyn FnMut(&Binding) -> bool,
) -> bool {
    let Some(&how) = shape.steps.get(step) else {
        return done(&binding);
    };
    let (p, candidates): (usize, Vec<usize>) = match how {
        Step::Anchor => (0, vec![anchor]),
        Step::Defines(p, v) => {
            let bound = binding.values[v.0 as usize].expect("a step follows a bound value");
            match trace.values[bound.0 as usize].def {
                Def::Op(i) if (i as usize) < trace.nodes.len() => (p, vec![i as usize]),
                _ => (p, Vec::new()),
            }
        }
        Step::Reads(p, v) => {
            let bound = binding.values[v.0 as usize].expect("a step follows a bound value");
            let readers = uses.readers[bound.0 as usize]
                .iter()
                .map(|&i| i as usize)
                .collect();
            (p, readers)
        }
    };
    let pattern = &shape.template.trace.nodes[p];
    for t in candidates {
        if binding.nodes.contains(&Some(t)) {
            continue;
        }
        let node = &trace.nodes[t];
        if node.op.name() != pattern.op.name() {
            continue;
        }
        if let Some(first) = binding.nodes.iter().flatten().next()
            && trace.nodes[*first].guard != node.guard
        {
            continue;
        }
        let mut next = binding.clone();
        next.nodes[p] = Some(t);
        if unify(
            trace,
            shape,
            &shape.trees[p],
            &tree::of(&node.op),
            &mut next,
        ) && extend(trace, uses, shape, step + 1, anchor, next, done)
        {
            return true;
        }
    }
    false
}

fn unify(trace: &Trace, shape: &Shape, p: &Tree, t: &Tree, b: &mut Binding) -> bool {
    if let Some(attr) = shape.attr(p) {
        return match &b.attrs[attr] {
            Some(seen) => seen == t,
            None => {
                b.attrs[attr] = Some(t.clone());
                true
            }
        };
    }
    match (p, t) {
        (Tree::Id(pv), Tree::Id(tv)) => {
            let (pv, tv) = (*pv as usize, ValueId(*tv));
            if let Some(seen) = b.values[pv] {
                return seen == tv;
            }
            let def = &trace.values[tv.0 as usize].def;
            let fits = match shape.kinds[pv] {
                Kind::Any => true,
                Kind::Weight => matches!(def, Def::Weight(_)),
                Kind::Cache => matches!(def, Def::Cache(_)),
                Kind::Computed => matches!(def, Def::Op(_)),
            };
            if fits {
                b.values[pv] = Some(tv);
            }
            fits
        }
        (Tree::Some(p), Tree::Some(t)) => unify(trace, shape, p, t, b),
        (Tree::Seq(ps), Tree::Seq(ts)) => {
            ps.len() == ts.len() && ps.iter().zip(ts).all(|(p, t)| unify(trace, shape, p, t, b))
        }
        (Tree::Struct(ps), Tree::Struct(ts)) => {
            ps.len() == ts.len()
                && ps.iter().zip(ts).all(|((pk, pv), (tk, tv))| {
                    pk == tk && (shape.template.free.contains(pk) || unify(trace, shape, pv, tv, b))
                })
        }
        (Tree::Variant(pn, p), Tree::Variant(tn, t)) => pn == tn && unify(trace, shape, p, t, b),
        (p, t) => p == t,
    }
}

/// A value the pattern computes but does not export is gone once fused, so
/// nothing outside the match may read it.
fn escapes_nothing(uses: &Uses, shape: &Shape, b: &Binding) -> bool {
    let matched: Vec<usize> = b.nodes.iter().flatten().copied().collect();
    shape.kinds.iter().enumerate().all(|(pv, kind)| {
        if *kind != Kind::Computed || shape.template.exports.contains(&ValueId(pv as u32)) {
            return true;
        }
        let Some(tv) = b.values[pv] else {
            return true;
        };
        !uses.held[tv.0 as usize]
            && uses.readers[tv.0 as usize]
                .iter()
                .all(|r| matched.contains(&(*r as usize)))
    })
}

/// The fused op lands where the last matched op was, so every op between the
/// first and the last must not depend on, or be depended on by, the ones that
/// move past it.
fn keeps_order(trace: &Trace, b: &Binding) -> bool {
    let matched: BTreeSet<usize> = b.nodes.iter().flatten().copied().collect();
    let (Some(&first), Some(&last)) = (matched.first(), matched.last()) else {
        return false;
    };
    let mut read = BTreeSet::new();
    let mut written = BTreeSet::new();
    let mut caches = BTreeSet::new();
    let mut ids = Vec::new();
    for &m in &matched {
        ids.clear();
        trace.nodes[m].op.inputs(&mut ids);
        for v in &ids {
            read.insert(*v);
            if matches!(trace.values[v.0 as usize].def, Def::Cache(_)) {
                caches.insert(*v);
            }
        }
        ids.clear();
        trace.nodes[m].op.outputs(&mut ids);
        written.extend(ids.iter().copied());
    }
    let mut aliases = Vec::new();
    (first + 1..last).filter(|i| !matched.contains(i)).all(|i| {
        let op = &trace.nodes[i].op;
        ids.clear();
        op.inputs(&mut ids);
        aliases.clear();
        op.aliases(&mut aliases);
        ids.iter()
            .all(|v| !written.contains(v) && !caches.contains(v))
            && aliases.iter().all(|(_, into)| !read.contains(into))
    })
}

fn rewrite(mut trace: Trace, b: &Binding, fused: poem_ir::Fused) -> Trace {
    let matched: BTreeSet<usize> = b.nodes.iter().flatten().copied().collect();
    let last = *matched.last().expect("a match holds at least one op");
    let guard = trace.nodes[last].guard.clone();
    let layer = matched.iter().rev().find_map(|&m| trace.nodes[m].layer);
    let mut landed = Vec::with_capacity(trace.nodes.len());
    let mut nodes = Vec::with_capacity(trace.nodes.len());
    let mut at = None;
    for (i, node) in std::mem::take(&mut trace.nodes).into_iter().enumerate() {
        if i == last {
            at = Some(nodes.len() as u32);
            nodes.push(Node {
                op: Operation::Fused(fused.clone()),
                guard: guard.clone(),
                layer,
            });
        } else if !matched.contains(&i) {
            landed.push(Some(nodes.len() as u32));
            nodes.push(node);
            continue;
        }
        landed.push(None);
    }
    let at = at.expect("the last matched op is in the trace");
    for decl in &mut trace.values {
        if let Def::Op(i) = &mut decl.def
            && *i != UNCLAIMED
        {
            *i = landed[*i as usize].unwrap_or(at);
        }
    }
    trace.nodes = nodes;
    trace
}

/// Drops the values a fused op computes inside itself and no longer writes,
/// renumbering the rest.
pub(super) fn compact(mut trace: Trace) -> Trace {
    let mut outs = Vec::new();
    let written: Vec<BTreeSet<ValueId>> = trace
        .nodes
        .iter()
        .map(|node| {
            outs.clear();
            node.op.outputs(&mut outs);
            outs.iter().copied().collect()
        })
        .collect();
    let keep: Vec<bool> = trace
        .values
        .iter()
        .enumerate()
        .map(|(v, decl)| match decl.def {
            Def::Op(i) => written
                .get(i as usize)
                .is_none_or(|outs| outs.contains(&ValueId(v as u32))),
            _ => true,
        })
        .collect();
    if keep.iter().all(|k| *k) {
        return trace;
    }
    let mut renumbered = Vec::with_capacity(keep.len());
    let mut next = 0u32;
    for k in &keep {
        renumbered.push(k.then(|| {
            next += 1;
            ValueId(next - 1)
        }));
    }
    let f = |v: ValueId| {
        renumbered[v.0 as usize].unwrap_or_else(|| panic!("value {} is gone but still read", v.0))
    };
    trace.values = std::mem::take(&mut trace.values)
        .into_iter()
        .zip(&keep)
        .filter(|(_, k)| **k)
        .map(|(mut decl, _)| {
            if let Def::Merge(arms) = &mut decl.def {
                for (arm, _) in arms {
                    *arm = f(*arm);
                }
            }
            decl
        })
        .collect();
    for node in &mut trace.nodes {
        node.op = tree::remap(&node.op, &f);
    }
    for seam in &mut trace.seams {
        for v in &mut seam.values {
            *v = f(*v);
        }
    }
    trace
}
