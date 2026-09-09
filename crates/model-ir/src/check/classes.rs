use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};

use crate::check::V;
use crate::ops::{Attention, CustomCuda, Layout, Spatial};
use crate::{Def, Guard, Operands, Operation, Trace, ValueId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Class {
    pub words: Vec<u64>,
    pub live: Vec<u32>,
}

impl Class {
    #[must_use]
    pub fn word(&self) -> u64 {
        self.words[0]
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct ClassSet {
    words: Vec<u64>,
}

impl ClassSet {
    #[must_use]
    pub fn of(classes: impl IntoIterator<Item = usize>) -> ClassSet {
        let mut set = ClassSet::default();
        for class in classes {
            set.insert(class);
        }
        set
    }

    pub fn insert(&mut self, class: usize) {
        let (w, bit) = (class / 64, class % 64);
        if self.words.len() <= w {
            self.words.resize(w + 1, 0);
        }
        self.words[w] |= 1 << bit;
    }

    #[must_use]
    pub fn disjoint(&self, other: &ClassSet) -> bool {
        !self.iter().any(|class| other.contains(class))
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

    #[must_use]
    pub fn len(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.words.iter().enumerate().flat_map(|(w, &bits)| {
            (0..64)
                .filter(move |b| bits & (1 << b) != 0)
                .map(move |b| w * 64 + b)
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassTable {
    pub classes: Vec<Class>,
    pub node_mask: Vec<ClassSet>,
    pub merges: Vec<ValueId>,
    pub merge_arm: Vec<Vec<Option<u8>>>,
    pub mask: u64,
    pub dead: Vec<u32>,
}

impl ClassTable {
    #[must_use]
    pub fn arms_of(&self, merge: ValueId) -> Option<&[Option<u8>]> {
        let row = self.merges.binary_search(&merge).ok()?;
        Some(&self.merge_arm[row])
    }

    #[must_use]
    pub fn class_of(&self, word: u64) -> Option<usize> {
        self.classes.iter().position(|c| c.words.contains(&word))
    }
}

fn written_in_class(trace: &Trace, id: ValueId, word: u64) -> bool {
    let Some(Def::Merge(arms)) = trace.values.get(id.0 as usize).map(|decl| &decl.def) else {
        return true;
    };
    if arms.iter().any(|(_, cond)| cond.holds(word)) {
        return true;
    }
    let conds: Vec<Guard> = arms.iter().map(|(_, cond)| cond.clone()).collect();
    let common = Guard::common(&conds);
    matches!(common, Guard::Always) || common.holds(word)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fault {
    Uncovered {
        merge: ValueId,
        word: u64,
    },
    Ambiguous {
        merge: ValueId,
        word: u64,
        arms: (u8, u8),
    },
}

impl Fault {
    #[must_use]
    pub fn merge(&self) -> ValueId {
        match self {
            Fault::Uncovered { merge, .. } | Fault::Ambiguous { merge, .. } => *merge,
        }
    }

    #[must_use]
    pub fn word(&self) -> u64 {
        match self {
            Fault::Uncovered { word, .. } | Fault::Ambiguous { word, .. } => *word,
        }
    }

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

#[must_use = "the classes are P1's output; dropping them re-runs the sweep"]
pub fn resolve_classes(trace: &Trace) -> Result<ClassTable, Vec<Fault>> {
    let facts = fact_width(trace);
    assert!(facts <= 20, "a plan over {facts} facts");

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

        for &j in &class.live {
            let op = &trace.nodes[j as usize].op;
            if writes_cache(op) || spans_classes(op) {
                walk.demand(j as usize, &mut node_mask);
            }
        }
        for seam in &trace.seams {
            walk.stack.extend(
                seam.values
                    .iter()
                    .copied()
                    .filter(|id| written_in_class(trace, *id, word)),
            );
        }

        while let Some(id) = walk.stack.pop() {
            let Some(decl) = trace.values.get(id.0 as usize) else {
                continue;
            };
            if walk.value[id.0 as usize] {
                continue;
            }
            walk.value[id.0 as usize] = true;
            match &decl.def {
                Def::Input(_) | Def::Weight(_) | Def::Cache(_) => {}
                Def::Op(i) => {
                    let i = *i as usize;
                    if trace.nodes.get(i).is_some_and(|n| n.guard.holds(word)) {
                        walk.demand(i, &mut node_mask);
                    } else if let Some(through) = passes_through(trace, i, id) {
                        walk.stack.push(through);
                    }
                }
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

struct Walk<'a> {
    trace: &'a Trace,
    class: usize,
    node: Vec<bool>,
    value: Vec<bool>,
    stack: Vec<ValueId>,
    ins: &'a mut Vec<ValueId>,
}

impl Walk<'_> {
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

fn intern<'a>(guards: &mut Vec<&'a Guard>, cond: &'a Guard) -> usize {
    guards.iter().position(|g| *g == cond).unwrap_or_else(|| {
        guards.push(cond);
        guards.len() - 1
    })
}

fn passes_through(trace: &Trace, i: usize, id: ValueId) -> Option<ValueId> {
    let mut aliases = Vec::new();
    trace.nodes.get(i)?.op.aliases(&mut aliases);
    aliases
        .into_iter()
        .find(|(out, _)| *out == id)
        .map(|(_, input)| input)
}

fn spans_classes(op: &Operation) -> bool {
    matches!(
        op,
        Operation::Attention(Attention::Ragged { .. })
            | Operation::Layout(Layout::PackRows { .. } | Layout::UnpackRows { .. })
    )
}

fn writes_cache(op: &Operation) -> bool {
    match op {
        Operation::Attention(op) => match op {
            Attention::KvAppend { .. }
            | Attention::KvAppendShared { .. }
            | Attention::MlaKvAppend { .. }
            | Attention::IndexKvAppend { .. }
            | Attention::PoolKvAppend { .. }
            | Attention::PoolStateWrite { .. } => true,
            Attention::SsmCausalConv1d { .. }
            | Attention::SsmCausalConv1dChunked { .. }
            | Attention::ShortConv { .. }
            | Attention::ShortConvChunked { .. }
            | Attention::SsmGatedDelta { .. }
            | Attention::SsmGatedDeltaChunked { .. }
            | Attention::SsmKdaStep { .. }
            | Attention::SsmKdaChunked { .. }
            | Attention::PleNgramIds { .. }
            | Attention::PleNgramIdsChunked { .. } => true,
            Attention::PlanDecode { .. }
            | Attention::PlanPrefill { .. }
            | Attention::Decode { .. }
            | Attention::Prefill { .. }
            | Attention::DecodeRel { .. }
            | Attention::PrefillRel { .. }
            | Attention::Masked { .. }
            | Attention::Dense { .. }
            | Attention::Ragged { .. }
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
            | Attention::BlockDynConv { .. }
            | Attention::SelectorWalk { .. }
            | Attention::IndexLayernormRope { .. }
            | Attention::IndexRope { .. }
            | Attention::IndexTopk { .. }
            | Attention::PoolBoundaryDecode { .. }
            | Attention::PoolBoundaryPrefill { .. }
            | Attention::PoolGather { .. }
            | Attention::PoolLse { .. }
            | Attention::PoolLseSelected { .. } => false,
        },
        Operation::CustomCuda(op) => match op {
            CustomCuda::QkvFusedQknormRopeVnormWrite { .. } => true,
        },
        Operation::Spatial(op) => match op {
            Spatial::Conv3d { cache, .. } => cache.is_some(),
            Spatial::CacheStore { .. } => true,
            Spatial::Grid { .. }
            | Spatial::GroupNorm { .. }
            | Spatial::Attention { .. }
            | Spatial::UpsampleNearest { .. }
            | Spatial::PixelShuffle { .. }
            | Spatial::PixelUnshuffle { .. }
            | Spatial::AvgDown { .. }
            | Spatial::Patchify { .. }
            | Spatial::Unpatchify { .. } => false,
        },
        Operation::Linear(_)
        | Operation::Elementwise(_)
        | Operation::Layout(_)
        | Operation::Collective(_) => false,
    }
}

impl ClassTable {
    #[must_use]
    pub fn adapter_fact(&self, corrected: &ClassSet) -> Option<u32> {
        if corrected.is_empty() {
            return None;
        }
        let domain = self.correction_domain(corrected);
        let reachable = |class: &Class| class.words.iter().any(|word| word & !domain == 0);
        let mut found = None;
        for bit in 0..u64::BITS {
            if self.mask & (1u64 << bit) == 0 {
                continue;
            }
            let decides = self.classes.iter().enumerate().all(|(at, class)| {
                if !reachable(class) {
                    return true;
                }
                let runs = corrected.contains(at);
                class
                    .words
                    .iter()
                    .all(|word| ((word >> bit) & 1 == 1) == runs)
            });
            if decides {
                if found.is_some() {
                    return None;
                }
                found = Some(bit);
            }
        }
        found
    }

    #[must_use]
    pub fn correction_domain(&self, corrected: &ClassSet) -> u64 {
        self.classes
            .iter()
            .enumerate()
            .filter(|(at, _)| corrected.contains(*at))
            .flat_map(|(_, class)| class.words.iter().copied())
            .fold(0u64, |acc, word| acc | word)
    }

    #[must_use]
    pub fn correction_reaches(&self, corrected: &ClassSet, word: u64) -> bool {
        word & self.mask & !self.correction_domain(corrected) == 0
    }

    #[must_use]
    pub fn adapted_word(&self, corrected: &ClassSet, bit: u32, word: u64) -> Option<u64> {
        if !self.correction_reaches(corrected, word) {
            return Some(word);
        }
        let adapted = word | (1u64 << bit);
        let class = self.class_of(adapted & self.mask)?;
        corrected.contains(class).then_some(adapted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{Attention, Elementwise};
    use crate::{Dim, Dtype, Node, RuntimeInput, Ty, ValueDecl};

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
                    drafter: None,
                },
                inputs: 0,
            }
        }

        fn value(&mut self, def: Def) -> ValueId {
            self.trace.values.push(ValueDecl { def, ty: act() });
            ValueId((self.trace.values.len() - 1) as u32)
        }

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
    fn classes_every_case() {
        a_split_and_its_merge_resolve_to_one_arm_per_class();
        a_gap_in_the_arms_is_uncovered_and_names_the_word();
        two_arms_holding_at_once_are_ambiguous();
        a_cache_write_is_its_own_root_and_an_unread_op_is_dead();
    }

    fn a_split_and_its_merge_resolve_to_one_arm_per_class() {
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
        assert_eq!(classes.arms_of(o), Some([Some(1), Some(0)].as_slice()));
        assert!(classes.node_mask[0].contains(1) && !classes.node_mask[0].contains(0));
        assert!(classes.node_mask[1].contains(0) && !classes.node_mask[1].contains(1));
        assert!(classes.dead.is_empty());
    }

    fn a_gap_in_the_arms_is_uncovered_and_names_the_word() {
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

    fn two_arms_holding_at_once_are_ambiguous() {
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

    fn a_cache_write_is_its_own_root_and_an_unread_op_is_dead() {
        let mut b = Build::new();
        let q = b.input();
        let k = b.op(q, Guard::Always);
        let append = b.append(k, Guard::Always);
        b.op(q, Guard::Always);
        b.out(q);

        let classes = b.resolve().expect("no merges, no faults");
        assert_eq!(classes.classes.len(), 1, "nothing here is guarded");
        assert_eq!(classes.mask, 0, "and so the sweep is over no bits at all");
        assert!(classes.node_mask[append].contains(0));
        assert!(classes.node_mask[0].contains(0));
        assert_eq!(classes.dead, vec![2]);
    }
}
