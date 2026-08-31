//! The class sweep and the backward demand walk (palo design §1). One
//! utility, two callers: the model test suite, where it is authoring-time
//! feedback, and the compiler's P0+P1, where it is the accept pass and the
//! class table at once — the check IS the sweep, so neither side pays for the
//! other (decision #7).
//!
//! WHAT IT ANSWERS: for every behavior a plan can exhibit, which nodes run,
//! and which arm of each `Def::Merge` writes the rows. A merge whose arms
//! leave a hole is the bug that presents as "nondeterministic garbage tokens
//! under specific batch mixes"; two arms holding at once is two writers of one
//! row range, and which one lands is a race. Both are one sentence here.
//!
//! WHY BACKWARD, NOT FORWARD. A forward rule — "a merge's arms must cover
//! every word" — refuses legal nesting: an inner merge that covers only the
//! `masked` world is fine precisely because in the `¬masked` classes the outer
//! merge never asks it for anything. A merge arm is a CONDITIONAL read and an
//! op input an unconditional one, so the only pass that can tell the two apart
//! is one that starts from what a class must produce and walks back to what
//! that needs (decision #6).
//!
//! WHAT A CLASS IS DEMANDED FOR — the roots of the walk:
//!
//! - cache writes: the kv/state appends and the fused write op, which exist
//!   for their effect and hand nothing back that a later node reads;
//! - `Trace::seams`: a value that materializes outside the graph (design §9),
//!   and — the discovery that settled the third root — THE PLAN'S OUTPUT IS
//!   ONE OF THESE. The IR has no `Trace::outputs` field, and the trace does not
//!   need one: `model_dsl::trace_hybrid` ends with
//!   `rec.seam(seam::OUT.name, &[&logits])`, so the value a `forward` returns
//!   is recorded as the `"out"` seam row. Rooting every seam value roots the
//!   output with it, and a model that exports a second value gets the same
//!   treatment without this file learning a new name.
//!
//! Collectives are NOT rooted here, deliberately. "A collective is never
//! elided" (decision #5) is a LOWERING rule — the compiler's P3 keeps every
//! Collective-family node always-launch so the NCCL rendezvous still pairs by
//! call order — and stating it twice would mean a `dead` report that quietly
//! disagreed with itself about what demand means.

use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};

use crate::check::V;
use crate::ops::{Attention, CustomCuda};
use crate::{Guard, Def, Operands, Operation, Trace, ValueId};

/// One deduplicated behavior: the fact words that run the same nodes and
/// resolve every merge the same way, and the nodes their guards admit.
///
/// `live` is what the fact word says CAN run; the nodes a class actually runs
/// are `live` narrowed by demand, which is what [`ClassTable::node_mask`]
/// carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Class {
    /// Every fact word in this class, ascending. Never empty.
    pub words: Vec<u64>,
    /// Node indices whose `guard` holds for these words, ascending.
    pub live: Vec<u32>,
}

impl Class {
    /// The class's representative word — the smallest, and the one every
    /// message about this class is written in terms of.
    #[must_use]
    pub fn word(&self) -> u64 {
        self.words[0]
    }
}

/// A dense set of class indices. Classes are few and consecutively numbered,
/// so a bitset is the whole implementation and `node_mask` is one allocation
/// per node rather than a `Vec<u32>` per node.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct ClassSet {
    words: Vec<u64>,
}

impl ClassSet {
    /// The set of exactly these class indices, in any order.
    ///
    /// WHY A NON-COMPILER CALLER BUILDS ONE. The sweep below answers what a
    /// plan CAN run, once, at bake — that is [`ClassTable::node_mask`], and
    /// nobody outside this crate has any business writing it. The fire path
    /// asks the mirror-image question every launch and gets a set of classes
    /// back for its trouble: which classes did this batch TURN OUT to contain?
    /// That answer is the argument `model_compiler::ClassOrder::class_order`
    /// takes, so an engine composing a fire has to be able to say it in this
    /// type — otherwise the one caller the seriation exists for is the one
    /// caller that cannot knock on the door, and reaches a layer further down
    /// for the ordering instead.
    #[must_use]
    pub fn of(classes: impl IntoIterator<Item = usize>) -> ClassSet {
        let mut set = ClassSet::default();
        for class in classes {
            set.insert(class);
        }
        set
    }

    /// Add a class to the set, widening it if the class is past the end.
    ///
    /// The companion to [`of`](ClassSet::of) for a caller that discovers its
    /// classes one at a time — a fire walking its lanes, say — rather than
    /// having an iterator of them in hand.
    pub fn insert(&mut self, class: usize) {
        let (w, bit) = (class / 64, class % 64);
        if self.words.len() <= w {
            self.words.resize(w + 1, 0);
        }
        self.words[w] |= 1 << bit;
    }

    #[must_use]
    pub fn contains(&self, class: usize) -> bool {
        self.words
            .get(class / 64)
            .is_some_and(|w| w & (1 << (class % 64)) != 0)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|w| *w == 0)
    }

    /// How many classes are in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// The class indices, ascending.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.words.iter().enumerate().flat_map(|(w, &bits)| {
            (0..64)
                .filter(move |b| bits & (1 << b) != 0)
                .map(move |b| w * 64 + b)
        })
    }
}

/// What one sweep of a plan found. This is P1's whole output as well as the
/// author's report: the compiler keeps it, the test suite throws it away.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassTable {
    /// The deduplicated behaviors, in ascending order of representative word.
    pub classes: Vec<Class>,
    /// Node index → the classes that RUN it (live and demanded).
    pub node_mask: Vec<ClassSet>,
    /// The plan's `Def::Merge` values in `Trace::values` order — the row key of
    /// [`merge_arm`](ClassTable::merge_arm), since a merge is a value and there
    /// is no separate merge id space to index by.
    pub merges: Vec<ValueId>,
    /// Merge row × class → the one arm that writes here, or `None` where no
    /// class demanded the merge. Rows are parallel to
    /// [`merges`](ClassTable::merges); [`arms_of`](ClassTable::arms_of) looks one up
    /// by `ValueId`.
    pub merge_arm: Vec<Vec<Option<u8>>>,
    /// The bits the sweep ran over: `(1 << F) - 1` for the `F` this plan's
    /// guards reach, and 0 for a plan no guard splits.
    ///
    /// A fire's word comes from the model's own `Classify`, which packs every
    /// fact the model COMPUTES; the sweep enumerates only the bits some guard
    /// READS. The two agree in every catalog text today and are not required
    /// to: a model may state a fact it does not yet split on, and a word with
    /// that bit set names no class here. Masking with this before
    /// [`class_of`](ClassTable::class_of) is what keeps that from reading as "the
    /// runtime and the shell disagree about what is loaded".
    pub mask: u64,
    /// Node indices demanded in no class at all. A report, not a fault: a plan
    /// may legitimately compute something only a later revision will read, and
    /// the compiler is free to drop these — but a surprise here is usually a
    /// forgotten consumer.
    pub dead: Vec<u32>,
}

impl ClassTable {
    /// The per-class arm resolution of one merge, or `None` if that value is
    /// not a merge in this plan.
    #[must_use]
    pub fn arms_of(&self, merge: ValueId) -> Option<&[Option<u8>]> {
        let row = self.merges.binary_search(&merge).ok()?;
        Some(&self.merge_arm[row])
    }

    /// The class a fact word belongs to.
    #[must_use]
    pub fn class_of(&self, word: u64) -> Option<usize> {
        self.classes.iter().position(|c| c.words.contains(&word))
    }
}

/// A merge that does not resolve. Both kinds name the merge and the fact
/// combination, which is the whole point: uncaught, these are a garbled token
/// under one batch mix; caught, they are one line.
///
/// The validator next door owns the name `check::Fault` already, so this one
/// keeps the design's spelling inside its own module and is re-exported from
/// the crate root as `ClassFault`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fault {
    /// No arm holds for a class that demands the merge — those rows are never
    /// written, and the kernel downstream reads whatever the buffer held.
    Uncovered { merge: ValueId, word: u64 },
    /// Two arms hold at once: a nondeterministic same-rows write race. The
    /// pair named is the first two in arm order; a third would be the same
    /// bug.
    Ambiguous {
        merge: ValueId,
        word: u64,
        arms: (u8, u8),
    },
}

impl Fault {
    /// The merge this fault is about.
    #[must_use]
    pub fn merge(&self) -> ValueId {
        match self {
            Fault::Uncovered { merge, .. } | Fault::Ambiguous { merge, .. } => *merge,
        }
    }

    /// The representative fact word of the class that found it.
    #[must_use]
    pub fn word(&self) -> u64 {
        match self {
            Fault::Uncovered { word, .. } | Fault::Ambiguous { word, .. } => *word,
        }
    }

    /// The fault as a sentence with the fact word spelled at the plan's own
    /// width. `resolve_classes` returns faults without the plan attached — a
    /// `Fault` is small, comparable data — and `Display` therefore has only
    /// the word, which prints `0b1` for a plan whose classes are four bits
    /// wide and says nothing about the three bits that are off. The plan comes
    /// back in here for exactly that: the width to pad to.
    #[must_use]
    pub fn say(&self, trace: &Trace) -> String {
        let width = fact_width(trace).max(1);
        let word = format!("0b{:0width$b}", self.word());
        match self {
            Fault::Uncovered { merge, .. } => format!(
                "merge {} is demanded for fact word {word}, and no arm holds \
                 there — those rows are never written",
                V(*merge),
            ),
            Fault::Ambiguous { merge, arms, .. } => format!(
                "merge {}: arms {} and {} both hold for fact word {word} — two \
                 writers of one row range, and which one lands is a race",
                V(*merge),
                arms.0,
                arms.1,
            ),
        }
    }
}

impl Display for Fault {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Fault::Uncovered { merge, word } => write!(
                f,
                "merge {} is demanded for fact word {word:#b}, and no arm holds \
                 there — those rows are never written",
                V(*merge),
            ),
            Fault::Ambiguous { merge, word, arms } => write!(
                f,
                "merge {}: arms {} and {} both hold for fact word {word:#b} — \
                 two writers of one row range, and which one lands is a race",
                V(*merge),
                arms.0,
                arms.1,
            ),
        }
    }
}

impl std::error::Error for Fault {}

/// How many fact bits this plan actually splits on: one past the highest bit
/// any guard names, and 0 for a plan no `Guard::Fact` appears in.
///
/// THE PLAN NO LONGER DECLARES A VOCABULARY, AND DOES NOT NEED TO. A fact list
/// was a second statement of what the guards already say, free to disagree
/// with them — a name nothing splits on doubled the sweep, and a bit some
/// guard read past the end of the list was a word the sweep never visited.
/// Reading the width off the guards themselves makes both impossible, and
/// costs one walk of the conds the sweep is about to intern anyway.
///
/// The arm guards count as much as the node guards do: an arm is what a merge
/// resolves by, so a bit that only ever appears in one is still a bit two fact
/// words can be told apart by.
#[must_use]
pub fn fact_width(trace: &Trace) -> usize {
    let nodes = trace.nodes.iter().map(|node| &node.guard);
    let arms = trace
        .values
        .iter()
        .filter_map(|decl| match &decl.def {
            Def::Merge(arms) => Some(arms),
            Def::Op(_) | Def::Input(_) | Def::Weight(_) | Def::Cache(_) => None,
        })
        .flatten()
        .map(|(_, cond)| cond);
    nodes
        .chain(arms)
        .filter_map(|cond| cond.referenced_bits().last().copied())
        .max()
        .map_or(0, |top| usize::from(top) + 1)
}

/// Sweep every fact word, deduplicate the words into classes, and resolve
/// every demanded merge in every class. Collects ALL faults before returning:
/// a coverage hole is usually one authoring mistake seen from several classes
/// at once, and stopping at the first would hide the shape of it.
///
/// Assumes a plan that already passed [`check`](crate::check) — but refuses to
/// panic on one that did not, since this is also the compiler's front door: an
/// id that indexes nothing is skipped here and named there.
///
/// # Panics
///
/// If the plan's guards reach past 20 fact bits. The sweep is `2^F`, and past
/// 20 it stops being a sweep; the same ceiling `Guard::simplified` states.
#[must_use = "the classes are P1's output; dropping them re-runs the sweep"]
pub fn resolve_classes(trace: &Trace) -> Result<ClassTable, Vec<Fault>> {
    let facts = fact_width(trace);
    assert!(facts <= 20, "a plan over {facts} facts");

    // Every guard the plan states, deduplicated by structure. Two fact words
    // belong to the same class iff every one of these answers the same for
    // both — node guards say what runs, arm guards say which arm writes, and
    // BOTH matter. Deduplicating on the live node set alone would conflate two
    // words that resolve a merge differently, which is possible whenever an
    // arm is a split of a value no node guards (a runtime input, say).
    let mut guards: Vec<&Guard> = Vec::new();
    let node_guard: Vec<usize> = trace
        .nodes
        .iter()
        .map(|node| intern(&mut guards, &node.guard))
        .collect();

    let mut merges: Vec<ValueId> = Vec::new();
    let mut merge_row: Vec<Option<usize>> = vec![None; trace.values.len()];
    for (idx, decl) in trace.values.iter().enumerate() {
        let Def::Merge(arms) = &decl.def else {
            continue;
        };
        assert!(
            arms.len() <= u8::MAX as usize + 1,
            "merge v{idx} has {} arms; an arm is named by a u8",
            arms.len(),
        );
        merge_row[idx] = Some(merges.len());
        merges.push(ValueId(idx as u32));
        for (_, cond) in arms {
            intern(&mut guards, cond);
        }
    }

    // The sweep. A signature is the truth of every guard at once, so classes
    // come out at exactly the granularity the walk below can tell apart.
    let mut classes: Vec<Class> = Vec::new();
    let mut seen: HashMap<Vec<u64>, usize> = HashMap::new();
    for word in 0..1u64 << facts {
        let mut signature = vec![0u64; guards.len().div_ceil(64)];
        for (g, cond) in guards.iter().enumerate() {
            if cond.holds(word) {
                signature[g / 64] |= 1 << (g % 64);
            }
        }
        match seen.get(&signature) {
            Some(&c) => classes[c].words.push(word),
            None => {
                let live = node_guard
                    .iter()
                    .enumerate()
                    .filter(|&(_, &g)| signature[g / 64] & (1 << (g % 64)) != 0)
                    .map(|(j, _)| j as u32)
                    .collect();
                seen.insert(signature, classes.len());
                classes.push(Class {
                    words: vec![word],
                    live,
                });
            }
        }
    }

    // The backward demand walk, once per class.
    let mut node_mask = vec![ClassSet::default(); trace.nodes.len()];
    let mut merge_arm = vec![vec![None; classes.len()]; merges.len()];
    let mut faults = Vec::new();
    let mut ins: Vec<ValueId> = Vec::new();

    for (c, class) in classes.iter().enumerate() {
        let word = class.word();
        let mut walk = Walk {
            trace,
            class: c,
            node: vec![false; trace.nodes.len()],
            value: vec![false; trace.values.len()],
            stack: Vec::new(),
            ins: &mut ins,
        };

        // Roots: the effects this class owes the world.
        for &j in &class.live {
            if writes_cache(&trace.nodes[j as usize].op) {
                walk.demand(j as usize, &mut node_mask);
            }
        }
        for seam in &trace.seams {
            walk.stack.extend(seam.values.iter().copied());
        }

        while let Some(id) = walk.stack.pop() {
            let Some(decl) = trace.values.get(id.0 as usize) else {
                continue; // `check` names an out-of-range id; the walk skips it.
            };
            if walk.value[id.0 as usize] {
                continue;
            }
            walk.value[id.0 as usize] = true;
            match &decl.def {
                // Bound before the first node: the walk ends here.
                Def::Input(_) | Def::Weight(_) | Def::Cache(_) => {}
                // An op input is an UNCONDITIONAL read: every input of a
                // demanded op is demanded. Aliases add nothing — an in-place
                // pair's `in` is already among `inputs()`, which is what makes
                // the overwrite visible to a liveness pass in the first place.
                Def::Op(i) => {
                    let i = *i as usize;
                    // A producer the class does not run cannot be walked
                    // through. That hole is only ever reached through a merge,
                    // and a merge is where this pass faults.
                    if trace.nodes.get(i).is_some_and(|n| n.guard.holds(word)) {
                        walk.demand(i, &mut node_mask);
                    } else if let Some(through) = passes_through(trace, i, id) {
                        // **A GUARDED IN-PLACE OP IS A PASS-THROUGH IN THE
                        // CLASSES IT SKIPS** — palo design §8's correction
                        // class, in the one pass that had to learn it.
                        //
                        // `linear.lora_correct` writes THROUGH the value it
                        // corrects: `y_out` aliases `y`, one arena column,
                        // and a class outside the adapter window simply never
                        // runs the node and reads what the trunk already put
                        // there. That is the identity for free — no merge, no
                        // arm, no φ — and it is the whole reason the axis
                        // costs a fire with no adapter lane nothing at all.
                        //
                        // But a demand walk that stopped here would conclude
                        // that NOTHING produces `y_out` in those classes, and
                        // therefore that the trunk which produces `y` is dead
                        // in them. So the alias is followed: the demand moves
                        // from the overwrite to the value it would have
                        // overwritten, which is exactly what the fire will
                        // read.
                        //
                        // Stated over `Operands::aliases` rather than over an
                        // op list, so it is a fact about the IR and not a
                        // table somebody has to remember to add to.
                        walk.stack.push(through);
                    }
                }
                // A merge arm is a CONDITIONAL read: exactly one arm writes
                // these rows in this class, and only that one is demanded.
                Def::Merge(arms) => {
                    let mut holds = arms
                        .iter()
                        .enumerate()
                        .filter(|(_, (_, cond))| cond.holds(word))
                        .map(|(k, (arm, _))| (k as u8, *arm));
                    match (holds.next(), holds.next()) {
                        (None, _) => faults.push(Fault::Uncovered { merge: id, word }),
                        (Some((k, arm)), None) => {
                            if let Some(row) = merge_row[id.0 as usize] {
                                merge_arm[row][c] = Some(k);
                            }
                            walk.stack.push(arm);
                        }
                        (Some((a, _)), Some((b, _))) => faults.push(Fault::Ambiguous {
                            merge: id,
                            word,
                            arms: (a, b),
                        }),
                    }
                }
            }
        }
    }

    if !faults.is_empty() {
        // Merge order, then class order: the report reads as a walk down the
        // plan rather than as the order a stack happened to pop.
        faults.sort_by_key(|f| (f.merge().0, f.word()));
        return Err(faults);
    }

    let dead = node_mask
        .iter()
        .enumerate()
        .filter(|(_, mask)| mask.is_empty())
        .map(|(j, _)| j as u32)
        .collect();

    Ok(ClassTable {
        classes,
        node_mask,
        merges,
        merge_arm,
        mask: (1u64 << facts) - 1,
        dead,
    })
}

/// One class's walk: what it has already been through, and what it still owes
/// a visit. `ins` is borrowed from the caller because every class refills it
/// once per demanded node and none of them wants its own allocation.
struct Walk<'a> {
    trace: &'a Trace,
    class: usize,
    node: Vec<bool>,
    value: Vec<bool>,
    stack: Vec<ValueId>,
    ins: &'a mut Vec<ValueId>,
}

impl Walk<'_> {
    /// Demand one node: mark it run in this class, and put every input it
    /// reads on the stack.
    fn demand(&mut self, node: usize, node_mask: &mut [ClassSet]) {
        if self.node[node] {
            return;
        }
        self.node[node] = true;
        node_mask[node].insert(self.class);
        self.ins.clear();
        self.trace.nodes[node].op.inputs(self.ins);
        self.stack.extend(self.ins.iter().copied());
    }
}

/// The index of a guard in the interned list, appending it if it is new.
/// Structural equality is the right identity here: two spellings of one
/// predicate are two guards, which costs a class the sweep would otherwise
/// merge and never costs correctness.
fn intern<'a>(guards: &mut Vec<&'a Guard>, cond: &'a Guard) -> usize {
    guards.iter().position(|g| *g == cond).unwrap_or_else(|| {
        guards.push(cond);
        guards.len() - 1
    })
}

/// The value a skipped in-place node would have left standing at `id`.
///
/// `Some(input)` when node `i` declares `id` as the `out` of an
/// `Operands::aliases` pair — the SSA form of "this op overwrites its
/// operand", which the compiler folds onto one arena slot. In a class that
/// does not run the node, that slot holds the operand, so a reader of `id`
/// reads `input`.
///
/// `None` for every other output, and that is the standing rule this is the
/// one exception to: a value whose producer the class does not run is a hole,
/// and the only legal way to reach one is a merge arm.
fn passes_through(trace: &Trace, i: usize, id: ValueId) -> Option<ValueId> {
    let mut aliases = Vec::new();
    trace.nodes.get(i)?.op.aliases(&mut aliases);
    aliases
        .into_iter()
        .find(|(out, _)| *out == id)
        .map(|(_, input)| input)
}

/// Does this op write a cache — is it demanded for its effect, whatever a
/// class does with what it returns?
///
/// Hand-written and exhaustive, like the port table next door and the
/// `Operands` impls it reads: a new op variant must answer this question
/// before it compiles, because getting it wrong by default is a kv append
/// that a class silently drops.
///
/// Reading a cache is not writing one. `Attention::Decode` and its kin name
/// their pool pointer too, and rooting them would make every attention node
/// live-and-demanded everywhere, which is exactly the "schedule the classes
/// apart" answer this design exists to replace.
///
/// Only two families can touch a cache at all — the family procedure in
/// `ops.rs` sends anything that touches a sequence cache to `Attention` (rule
/// 1), and `CustomCuda` is where a backend's fusion of one goes — so the other
/// four answer as families rather than variant by variant.
fn writes_cache(op: &Operation) -> bool {
    match op {
        Operation::Attention(op) => match op {
            // The appends: kv pages, the indexer's key cache, the pooled
            // entries. All of them return nothing at all.
            Attention::KvAppend { .. }
            | Attention::KvAppendShared { .. }
            | Attention::MlaKvAppend { .. }
            | Attention::IndexKvAppend { .. }
            | Attention::PoolKvAppend { .. } => true,
            // The recurrent mixers: their sequence cache is a state slab
            // updated IN PLACE, so the fire's write is invisible in `outputs`.
            Attention::SsmCausalConv1d { .. }
            | Attention::SsmCausalConv1dChunked { .. }
            | Attention::SsmGatedDelta { .. }
            | Attention::SsmGatedDeltaChunked { .. }
            | Attention::SsmKdaStep { .. }
            | Attention::SsmKdaChunked { .. }
            // The n-gram hasher's window of token ids is a state slab too.
            | Attention::PleNgramIds { .. }
            | Attention::PleNgramIdsChunked { .. } => true,
            // Everything else reads: the plan builders, the attention kernels
            // and their epilogues, the MLA projections, the indexer's scoring,
            // the pool's boundary math and gather.
            Attention::PlanDecode { .. }
            | Attention::PlanPrefill { .. }
            | Attention::Decode { .. }
            | Attention::Prefill { .. }
            | Attention::Masked { .. }
            // The tower's attention touches no sequence cache at all — no kv
            // pool, no page table, no append. It is an `Attention` by rule 1's
            // first clause (tokens interact) and by nothing else.
            | Attention::Dense { .. }
            | Attention::DecodeLse { .. }
            | Attention::PrefillLse { .. }
            | Attention::Sink { .. }
            | Attention::MergeLse { .. }
            | Attention::LogitSoftcap { .. }
            | Attention::MlaPlan { .. }
            | Attention::MlaLatents { .. }
            | Attention::MlaLatentsRope { .. }
            | Attention::MlaSplitQB { .. }
            | Attention::MlaAbsorbQ { .. }
            | Attention::MlaAbsorbOut { .. }
            | Attention::MlaDecode { .. }
            | Attention::MlaPrefill { .. }
            | Attention::MlaDecodeSelected { .. }
            | Attention::MlaPrefillSelected { .. }
            | Attention::SsmGdnPrep { .. }
            | Attention::IndexLayernormRope { .. }
            | Attention::IndexRope { .. }
            | Attention::IndexTopk { .. }
            | Attention::PoolBoundaryDecode { .. }
            | Attention::PoolBoundaryPrefill { .. }
            | Attention::PoolGather { .. }
            | Attention::PoolLse { .. } => false,
        },
        // The one fused variant says so in its name: it lands k and v in the
        // pages on its way to returning q.
        Operation::CustomCuda(op) => match op {
            CustomCuda::QkvFusedQknormRopeVnormWrite { .. } => true,
        },
        Operation::Linear(_)
        | Operation::Elementwise(_)
        | Operation::Layout(_)
        | Operation::Collective(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{Attention, Elementwise};
    use crate::{Dim, Dtype, Node, RuntimeInput, Ty, ValueDecl};

    /// A plan built by hand, one statement per line. `model-dsl` is the
    /// authoring surface and cannot be reached from here — it depends on this
    /// crate — so these tests say in `Def` and `Guard` what a forward pass says
    /// in `split` and `Value::merge`.
    struct Build {
        trace: Trace,
        inputs: u32,
    }

    fn act() -> Ty {
        Ty::Tensor {
            shape: vec![Dim::Tokens],
            dtype: Dtype::Bf16,
        }
    }

    fn fact(bit: u8) -> Guard {
        Guard::Fact(bit)
    }

    impl Build {
        fn new() -> Build {
            Build {
                trace: Trace {
                    name: "hand-built".to_string(),
                    platform: crate::Platform::Cuda,
                    params: Vec::new(),
                    caches: vec![crate::CacheRow::State {
                        name: "state".to_string(),
                        slab: vec![1],
                        dtype: crate::Dtype::Bf16,
                    }],
                    values: Vec::new(),
                    nodes: Vec::new(),
                    seams: Vec::new(),
                },
                inputs: 0,
            }
        }

        fn value(&mut self, def: Def) -> ValueId {
            self.trace.values.push(ValueDecl { def, ty: act() });
            ValueId((self.trace.values.len() - 1) as u32)
        }

        /// A demand sink: something the engine binds, distinct per call.
        fn input(&mut self) -> ValueId {
            self.inputs += 1;
            let which = RuntimeInput::Mask {
                space: self.inputs - 1,
            };
            self.value(Def::Input(which))
        }

        fn cache(&mut self) -> ValueId {
            self.value(Def::Cache(0))
        }

        /// One guarded op over `x`, handing back its output.
        fn op(&mut self, x: ValueId, guard: Guard) -> ValueId {
            let node = self.trace.nodes.len() as u32;
            let y = self.value(Def::Op(node));
            self.trace.nodes.push(Node {
                op: Elementwise::MulScalar {
                    s: 2.0,
                    x,
                    x_out: y,
                }
                .into(),
                guard,
                layer: None,
            });
            y
        }

        /// A cache write: an effect root, and it returns nothing.
        fn append(&mut self, x: ValueId, guard: Guard) -> usize {
            let cache = self.cache();
            let page = self.input();
            let offset = self.input();
            self.trace.nodes.push(Node {
                op: Attention::KvAppendShared {
                    plane: x,
                    cache,
                    write_page: page,
                    write_offset: offset,
                }
                .into(),
                guard,
                layer: None,
            });
            self.trace.nodes.len() - 1
        }

        fn merge(&mut self, arms: &[(ValueId, Guard)]) -> ValueId {
            self.value(Def::Merge(arms.to_vec()))
        }

        /// The `"out"` seam — what a trace writes the forward's return value
        /// as, and therefore what roots the walk.
        fn out(&mut self, v: ValueId) -> &mut Build {
            self.trace.seams.push(crate::Seam {
                seam: "out".to_string(),
                values: vec![v],
                layer: None,
            });
            self
        }

        fn resolve(&self) -> Result<ClassTable, Vec<Fault>> {
            resolve_classes(&self.trace)
        }
    }

    #[test]
    fn a_split_and_its_merge_resolve_to_one_arm_per_class() {
        // The decode/prefill shape every shipped model has:
        //   let (dq, p) = q.split(&Facts::qo_one());
        //   let o = Value::merge(vec![decode(&dq), prefill(&p)]);
        let mut b = Build::new();
        let q = b.input();
        let d = b.op(q, fact(0));
        let p = b.op(q, Guard::not(fact(0)));
        let o = b.merge(&[(d, fact(0)), (p, Guard::not(fact(0)))]);
        b.out(o);

        let classes = b.resolve().expect("a covering split resolves");
        assert_eq!(classes.classes.len(), 2);
        assert_eq!(classes.classes[0].words, vec![0]);
        assert_eq!(classes.classes[1].words, vec![1]);
        // Class 0 is the ¬qo_one world: the prefill arm writes, and only the
        // prefill node runs.
        assert_eq!(classes.arms_of(o), Some([Some(1), Some(0)].as_slice()));
        assert!(classes.node_mask[0].contains(1) && !classes.node_mask[0].contains(0));
        assert!(classes.node_mask[1].contains(0) && !classes.node_mask[1].contains(1));
        assert!(classes.dead.is_empty());
    }

    #[test]
    fn a_gap_in_the_arms_is_uncovered_and_names_the_word() {
        // masked is bit 1, and the second arm forgot the ¬qo_one ∧ ¬masked
        // world entirely. Both bits are guarded on, so the word `say` spells
        // is two wide.
        let mut b = Build::new();
        let q = b.input();
        let d = b.op(q, fact(0));
        let m = b.op(q, Guard::and(Guard::not(fact(0)), fact(1)));
        let o = b.merge(&[(d, fact(0)), (m, Guard::and(Guard::not(fact(0)), fact(1)))]);
        b.out(o);

        let faults = b.resolve().expect_err("a hole is a fault");
        assert_eq!(faults, vec![Fault::Uncovered { merge: o, word: 0 }]);
        assert_eq!(
            faults[0].say(&b.trace),
            "merge v3 is demanded for fact word 0b00, and no arm holds there — \
             those rows are never written",
        );
    }

    #[test]
    fn two_arms_holding_at_once_are_ambiguous() {
        // An `Always` arm beside a guarded one: in the qo_one world both write
        // the same rows.
        let mut b = Build::new();
        let q = b.input();
        let a = b.op(q, Guard::Always);
        let d = b.op(q, fact(0));
        let o = b.merge(&[(a, Guard::Always), (d, fact(0))]);
        b.out(o);

        let faults = b.resolve().expect_err("a race is a fault");
        assert_eq!(
            faults,
            vec![Fault::Ambiguous {
                merge: o,
                word: 1,
                arms: (0, 1),
            }],
        );
    }

    #[test]
    fn an_inner_merge_covering_only_masked_is_legal() {
        // design §1, verbatim:
        //   let (m, um) = x.split(&Facts::masked());
        //   let (md, mp) = m.split(&Facts::qo_one());
        //   let o_m = Value::merge(vec![attn_md(&md), attn_mp(&mp)]);  // masked only
        //   let o   = Value::merge(vec![o_m, attn_plain(&um)]);        // everything
        // The inner merge has NO arm in the ¬masked classes, and that is fine
        // because the outer merge never asks it there. A forward coverage rule
        // would refuse this plan; the backward walk accepts it.
        let masked = fact(0);
        let qo_one = fact(1);
        let mut b = Build::new();
        let x = b.input();
        let md = Guard::and(masked.clone(), qo_one.clone());
        let mp = Guard::and(masked.clone(), Guard::not(qo_one));
        let attn_md = b.op(x, md.clone());
        let attn_mp = b.op(x, mp.clone());
        let inner = b.merge(&[(attn_md, md), (attn_mp, mp)]);
        let plain = b.op(x, Guard::not(masked.clone()));
        let outer = b.merge(&[(inner, masked.clone()), (plain, Guard::not(masked))]);
        b.out(outer);

        let classes = b.resolve().expect("legal nesting resolves");
        // Three behaviors, not four: with masked off, qo_one changes nothing
        // — no guard in the plan can tell those two words apart.
        assert_eq!(classes.classes.len(), 3);
        let plain_class = classes.class_of(0b10).expect("qo_one ∧ ¬masked");
        assert_eq!(classes.class_of(0b00), Some(plain_class));
        assert_eq!(classes.arms_of(inner).unwrap()[plain_class], None);
        assert_eq!(classes.arms_of(outer).unwrap()[plain_class], Some(1));
        // …and in the masked ∧ qo_one world both merges resolve.
        let inner_class = classes.class_of(0b11).expect("masked ∧ qo_one");
        assert_eq!(classes.arms_of(inner).unwrap()[inner_class], Some(0));
        assert_eq!(classes.arms_of(outer).unwrap()[inner_class], Some(0));
        assert!(classes.dead.is_empty());
    }

    #[test]
    fn a_merge_no_class_demands_may_have_a_gap() {
        // Both arms of `inner` are guarded by qo_one, so the ¬qo_one classes
        // have nothing to pick — and never ask, because the only consumer of
        // `inner` is a node those classes do not run. The gap is legal exactly
        // as long as nothing demands it, which is the same rule the nested
        // case above passes under, stated with one merge instead of two.
        let qo_one = fact(0);
        let masked = fact(1);
        let mut b = Build::new();
        let q = b.input();
        let hot = Guard::and(qo_one.clone(), masked.clone());
        let cold = Guard::and(qo_one.clone(), Guard::not(masked));
        let one = b.op(q, hot.clone());
        let two = b.op(q, cold.clone());
        let inner = b.merge(&[(one, hot), (two, cold)]);
        let used = b.op(inner, qo_one.clone());
        let rest = b.op(q, Guard::not(qo_one.clone()));
        let o = b.merge(&[(used, qo_one.clone()), (rest, Guard::not(qo_one))]);
        b.out(o);

        let classes = b.resolve().expect("an undemanded gap is not a fault");
        let off = classes.class_of(0b00).expect("¬qo_one");
        assert_eq!(
            classes.class_of(0b10),
            Some(off),
            "masked alone runs nothing new",
        );
        assert_eq!(classes.arms_of(inner).unwrap()[off], None);
        assert_eq!(classes.arms_of(o).unwrap()[off], Some(1));
        // …and inside qo_one the same merge resolves, one arm per class.
        let hot = classes.class_of(0b11).expect("qo_one ∧ masked");
        let cold = classes.class_of(0b01).expect("qo_one ∧ ¬masked");
        assert_eq!(classes.arms_of(inner).unwrap()[hot], Some(0));
        assert_eq!(classes.arms_of(inner).unwrap()[cold], Some(1));
        assert!(classes.dead.is_empty());
    }

    #[test]
    fn a_cache_write_is_its_own_root_and_an_unread_op_is_dead() {
        let mut b = Build::new();
        let q = b.input();
        let k = b.op(q, Guard::Always); // node 0
        let append = b.append(k, Guard::Always); // node 1: hands nothing back
        b.op(q, Guard::Always); // node 2: nobody ever reads it
        b.out(q);

        let classes = b.resolve().expect("no merges, no faults");
        assert_eq!(classes.classes.len(), 1, "nothing here is guarded");
        assert_eq!(classes.mask, 0, "and so the sweep is over no bits at all");
        // The append is demanded for its effect, and drags its input in with
        // it — nothing downstream reads either one.
        assert!(classes.node_mask[append].contains(0));
        assert!(classes.node_mask[0].contains(0));
        assert_eq!(classes.dead, vec![2]);
    }
}
