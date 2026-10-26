//! Whether two traces compute the same thing, whatever order they were
//! recorded in and however their values and facts are numbered.
//!
//! Each value is named by what defines it: an input by what it reads, a
//! weight by its param's name, a cache by its row's name, an op's output by
//! the op, its operands' names and the rows its guard holds for. Two traces
//! are the same when they hold the same live nodes and outputs under those
//! names, the same params, caches and facts, and every cache sees its reads
//! between the same writes. A node is live when an output or a write reads
//! what it makes, however far down; the rest compute nothing anyone sees.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};

use crate::operands::Operands;
use crate::trace::{CacheRow, Trace};
use crate::value::{Def, ValueId};

/// The first way `a` and `b` differ, or none if they compute the same thing.
#[must_use]
pub fn difference(a: &Trace, b: &Trace) -> Option<String> {
    let (mut ka, mut kb) = (Keys::of(a), Keys::of(b));
    if a.params != b.params {
        return Some(first_unlike("params", &a.params, &b.params));
    }
    if a.caches != b.caches {
        return Some(first_unlike("caches", &a.caches, &b.caches));
    }
    if a.drafter != b.drafter {
        return Some(format!("drafter {:?} vs {:?}", a.drafter, b.drafter));
    }
    let (fa, fb) = (facts(a), facts(b));
    if fa != fb {
        return Some(format!("facts {fa:?} vs {fb:?}"));
    }
    let (na, nb) = (ka.nodes(a), kb.nodes(b));
    if na != nb {
        return Some(unlike_multiset("nodes", &na, &nb));
    }
    let (sa, sb) = (ka.outputs(a), kb.outputs(b));
    if sa != sb {
        return Some(unlike_multiset("outputs", &sa, &sb));
    }
    None
}

fn first_unlike<T: std::fmt::Debug + PartialEq>(what: &str, a: &[T], b: &[T]) -> String {
    match a.iter().zip(b).position(|(x, y)| x != y) {
        Some(at) => format!("{what}[{at}]: {:?} vs {:?}", a[at], b[at]),
        None => format!("{what}: {} vs {} entries", a.len(), b.len()),
    }
}

fn unlike_multiset(what: &str, a: &BTreeMap<String, usize>, b: &BTreeMap<String, usize>) -> String {
    let only = |x: &BTreeMap<String, usize>, y: &BTreeMap<String, usize>| {
        x.iter()
            .filter(|(k, n)| y.get(*k) != Some(n))
            .map(|(k, n)| format!("{n}× {}", short(k)))
            .take(3)
            .collect::<Vec<_>>()
    };
    format!(
        "{what}: only in the first {:?}; only in the second {:?}",
        only(a, b),
        only(b, a)
    )
}

fn short(key: &str) -> &str {
    &key[..key.len().min(240)]
}

/// The facts a trace reads, by what they are rather than by bit.
fn facts(t: &Trace) -> Vec<String> {
    let mut names: Vec<String> = t.facts.table.iter().map(|f| format!("{f:?}")).collect();
    names.sort();
    names
}

struct Keys {
    /// Each value's name, by id.
    names: HashMap<ValueId, u64>,
    /// Each fact bit's name.
    bits: Vec<String>,
    /// Each fact bit's place among the facts in name order.
    order: Vec<usize>,
}

impl Keys {
    fn of(t: &Trace) -> Keys {
        let mut producer = HashMap::new();
        for (n, node) in t.nodes.iter().enumerate() {
            let mut outs = Vec::new();
            Operands::outputs(&node.op, &mut outs);
            for (k, id) in outs.into_iter().enumerate() {
                producer.insert(id, (n, k));
            }
        }
        let bits: Vec<String> = t.facts.table.iter().map(|f| format!("{f:?}")).collect();
        let mut sorted = bits.clone();
        sorted.sort();
        let order = bits
            .iter()
            .map(|b| {
                sorted
                    .iter()
                    .position(|s| s == b)
                    .expect("a fact of its own table")
            })
            .collect();
        let mut keys = Keys {
            names: HashMap::new(),
            bits,
            order,
        };
        for id in 0..t.values.len() {
            keys.name(t, &producer, ValueId(id as u32));
        }
        keys
    }

    fn name(&mut self, t: &Trace, producer: &HashMap<ValueId, (usize, usize)>, id: ValueId) -> u64 {
        if let Some(&h) = self.names.get(&id) {
            return h;
        }
        let decl = &t.values[id.0 as usize];
        let text = match &decl.def {
            Def::Input(input) => format!("in {input:?} {:?}", decl.ty),
            Def::Weight(w) => format!("weight {}", t.params[*w as usize].name),
            Def::Cache(c) => format!("cache {}", cache_name(&t.caches[*c as usize])),
            Def::Op(_) => match producer.get(&id) {
                Some(&(n, k)) => {
                    let node = self.node_text(t, producer, n);
                    format!("out {k} of {node}")
                }
                None => format!("op output {id:?} with no producer"),
            },
            Def::Merge(arms) => {
                let arms: Vec<String> = arms
                    .iter()
                    .map(|(v, g)| {
                        let v = self.name(t, producer, *v);
                        format!("{v:x} when {}", self.rows(g))
                    })
                    .collect();
                // The rows the arms imply it is over, as the class check
                // reads a merge an output holds.
                let Def::Merge(joined) = &decl.def else {
                    unreachable!("a merge's own declaration")
                };
                let conds: Vec<crate::guard::Guard> =
                    joined.iter().map(|(_, g)| g.clone()).collect();
                let over = self.rows(&crate::guard::Guard::implied(&conds));
                format!("merge {arms:?} over {over} {:?}", decl.ty)
            }
        };
        let h = hash(&text);
        self.names.insert(id, h);
        h
    }

    /// An op as its kind, its attributes and its operands' names, the
    /// outputs left unnamed (they are named by this text).
    fn node_text(
        &mut self,
        t: &Trace,
        producer: &HashMap<ValueId, (usize, usize)>,
        n: usize,
    ) -> String {
        let node = &t.nodes[n];
        let mut outs = Vec::new();
        Operands::outputs(&node.op, &mut outs);
        let text = format!("{:?}", node.op);
        let text = self.substitute(&text, |keys, id| {
            if outs.contains(&id) {
                "out".to_string()
            } else {
                format!("{:x}", keys.name(t, producer, id))
            }
        });
        format!(
            "{text} when {} layer {:?}",
            self.rows(&node.guard),
            node.layer
        )
    }

    /// Every node, the ones touching a cache with how many writes to it
    /// came before them, as a multiset of names. Reads between two writes
    /// commute; a write is a node that returns nothing or aliases the cache.
    fn nodes(&mut self, t: &Trace) -> BTreeMap<String, usize> {
        let mut producer = HashMap::new();
        for (n, node) in t.nodes.iter().enumerate() {
            let mut outs = Vec::new();
            Operands::outputs(&node.op, &mut outs);
            for (k, id) in outs.into_iter().enumerate() {
                producer.insert(id, (n, k));
            }
        }
        let live = live(t);
        let mut epochs: HashMap<u64, usize> = HashMap::new();
        let mut out = BTreeMap::new();
        for (n, &alive) in live.iter().enumerate() {
            if !alive {
                continue;
            }
            let op = &t.nodes[n].op;
            let (mut ins, mut outs, mut aliases) = (Vec::new(), Vec::new(), Vec::new());
            Operands::inputs(op, &mut ins);
            Operands::outputs(op, &mut outs);
            Operands::aliases(op, &mut aliases);
            let mut order = Vec::new();
            for id in ins {
                if matches!(t.values[id.0 as usize].def, Def::Cache(_)) {
                    let cache = self.name(t, &producer, id);
                    let writes_it = writes(t, n)
                        && (outs.is_empty() || aliases.iter().any(|(_, of)| *of == id));
                    let epoch = epochs.entry(cache).or_default();
                    order.push(format!(
                        "{cache:x}@{epoch}{}",
                        if writes_it { "w" } else { "r" }
                    ));
                    if writes_it {
                        *epoch += 1;
                    }
                }
            }
            let text = format!("{} {order:?}", self.node_text(t, &producer, n));
            *out.entry(text).or_default() += 1;
        }
        out
    }

    fn outputs(&mut self, t: &Trace) -> BTreeMap<String, usize> {
        let mut out = BTreeMap::new();
        for output in &t.outputs {
            let values: Vec<String> = output
                .values()
                .map(|v| format!("{:x}", self.names.get(&v).copied().unwrap_or_default()))
                .collect();
            *out.entry(format!(
                "{} {values:?} layer {:?}",
                output.name, output.layer
            ))
            .or_default() += 1;
        }
        out
    }

    /// The rows a guard holds for, as the assignments of the facts, taken in
    /// name order, it holds under: two guards spelled apart that hold for
    /// the same rows are one.
    fn rows(&self, guard: &crate::guard::Guard) -> String {
        let facts = self.bits.len();
        assert!(facts <= 20, "a trace over {facts} facts");
        let mut holds = Vec::new();
        for assignment in 0..1u64 << facts {
            let mut word = 0u64;
            for (bit, place) in self.order.iter().enumerate() {
                if assignment & (1 << place) != 0 {
                    word |= 1 << bit;
                }
            }
            if guard.holds(word) {
                holds.push(assignment);
            }
        }
        format!("{holds:x?}")
    }

    /// A guard's text with each fact named rather than numbered.
    fn guard(&self, text: &str) -> String {
        replace_numbered(text, "Fact(", |bit| {
            self.bits
                .get(bit as usize)
                .cloned()
                .unwrap_or_else(|| format!("unknown fact {bit}"))
        })
    }

    fn substitute(
        &mut self,
        text: &str,
        mut name: impl FnMut(&mut Keys, ValueId) -> String,
    ) -> String {
        let text = self.guard(text);
        let mut ids = Vec::new();
        replace_numbered(&text, "ValueId(", |id| {
            ids.push(id);
            String::new()
        });
        let names: Vec<String> = ids.into_iter().map(|id| name(self, ValueId(id))).collect();
        let mut names = names.into_iter();
        replace_numbered(&text, "ValueId(", |_| names.next().unwrap_or_default())
    }
}

/// Whether a write is a node: it returns nothing, or writes a cache in place.
fn writes(t: &Trace, n: usize) -> bool {
    let op = &t.nodes[n].op;
    let (mut outs, mut aliases) = (Vec::new(), Vec::new());
    Operands::outputs(op, &mut outs);
    Operands::aliases(op, &mut aliases);
    outs.is_empty()
        || aliases
            .iter()
            .any(|(_, of)| matches!(t.values[of.0 as usize].def, Def::Cache(_)))
}

/// Each node, whether an output or a write reads what it makes.
fn live(t: &Trace) -> Vec<bool> {
    let mut producer = HashMap::new();
    for (n, node) in t.nodes.iter().enumerate() {
        let mut outs = Vec::new();
        Operands::outputs(&node.op, &mut outs);
        for id in outs {
            producer.insert(id, n);
        }
    }
    let mut live = vec![false; t.nodes.len()];
    let mut wanted: Vec<ValueId> = t.outputs.iter().flat_map(crate::Output::values).collect();
    for (n, node) in t.nodes.iter().enumerate() {
        if writes(t, n) {
            live[n] = true;
            Operands::inputs(&node.op, &mut wanted);
        }
    }
    let mut seen = std::collections::HashSet::new();
    while let Some(id) = wanted.pop() {
        if !seen.insert(id) {
            continue;
        }
        if let Def::Merge(arms) = &t.values[id.0 as usize].def {
            wanted.extend(arms.iter().map(|(v, _)| *v));
        }
        if let Some(&n) = producer.get(&id)
            && !live[n]
        {
            live[n] = true;
            Operands::inputs(&t.nodes[n].op, &mut wanted);
        }
    }
    live
}

fn cache_name(row: &CacheRow) -> &str {
    match row {
        CacheRow::Kv { name, .. } | CacheRow::State { name, .. } => name,
    }
}

/// `text` with each `{prefix}{number})` replaced by what `with` makes of the
/// number.
fn replace_numbered(text: &str, prefix: &str, mut with: impl FnMut(u32) -> String) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(prefix) {
        let after = &rest[at + prefix.len()..];
        let digits = after.bytes().take_while(u8::is_ascii_digit).count();
        if digits > 0 && after.as_bytes().get(digits) == Some(&b')') {
            out.push_str(&rest[..at]);
            out.push('<');
            out.push_str(&with(after[..digits].parse().expect("digits")));
            out.push('>');
            rest = &after[digits + 1..];
        } else {
            out.push_str(&rest[..at + prefix.len()]);
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

fn hash(text: &str) -> u64 {
    let mut h = DefaultHasher::new();
    text.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canon_every_case() {
        a_numbered_token_is_replaced_and_others_kept();
    }

    fn a_numbered_token_is_replaced_and_others_kept() {
        let text = "Decode { q: ValueId(3), head_dim: 128, plan: ValueId(12) } Fact(2) ValueId(x)";
        let out = replace_numbered(text, "ValueId(", |n| format!("v{n}"));
        assert_eq!(
            out,
            "Decode { q: <v3>, head_dim: 128, plan: <v12> } Fact(2) ValueId(x)"
        );
    }
}
