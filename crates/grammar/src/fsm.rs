use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use crate::grammar::{Expr, ExprId, Grammar, RuleId};
use anyhow::{Result, bail};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StateId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FsmEdge {
    CharRange { min: u8, max: u8, target: StateId },
    Epsilon(StateId),
    RuleRef { rule: RuleId, target: StateId },
}

#[derive(Debug, Clone)]
pub struct NfaGraph {
    edges: Vec<Vec<FsmEdge>>,
}

impl NfaGraph {
    pub fn new() -> Self {
        Self { edges: Vec::new() }
    }

    pub fn add_state(&mut self) -> StateId {
        let id = StateId(self.edges.len() as u32);
        self.edges.push(Vec::new());
        id
    }

    pub fn num_states(&self) -> usize {
        self.edges.len()
    }

    pub fn add_edge(&mut self, from: StateId, edge: FsmEdge) {
        self.edges[from.0 as usize].push(edge);
    }

    pub fn add_char_edge(&mut self, from: StateId, min: u8, max: u8, target: StateId) {
        self.add_edge(from, FsmEdge::CharRange { min, max, target });
    }

    pub fn add_epsilon(&mut self, from: StateId, target: StateId) {
        self.add_edge(from, FsmEdge::Epsilon(target));
    }

    pub fn add_rule_ref(&mut self, from: StateId, rule: RuleId, target: StateId) {
        self.add_edge(from, FsmEdge::RuleRef { rule, target });
    }

    pub fn edges(&self, state: StateId) -> &[FsmEdge] {
        &self.edges[state.0 as usize]
    }

    pub fn epsilon_closure(&self, states: &BTreeSet<StateId>) -> BTreeSet<StateId> {
        let mut closure = states.clone();
        let mut queue: VecDeque<StateId> = states.iter().copied().collect();

        while let Some(s) = queue.pop_front() {
            for edge in &self.edges[s.0 as usize] {
                if let FsmEdge::Epsilon(target) = edge
                    && closure.insert(*target)
                {
                    queue.push_back(*target);
                }
            }
        }
        closure
    }

    pub fn to_compact(&self) -> DfaTable {
        let mut all_edges = Vec::new();
        let mut state_offsets = Vec::with_capacity(self.edges.len() + 1);

        for state_edges in &self.edges {
            state_offsets.push(all_edges.len() as u32);
            let mut sorted = state_edges.clone();
            sorted.sort_by(|a, b| match (a, b) {
                (FsmEdge::CharRange { min: a_min, .. }, FsmEdge::CharRange { min: b_min, .. }) => {
                    a_min.cmp(b_min)
                }
                (FsmEdge::CharRange { .. }, _) => std::cmp::Ordering::Less,
                (_, FsmEdge::CharRange { .. }) => std::cmp::Ordering::Greater,
                _ => std::cmp::Ordering::Equal,
            });
            all_edges.extend(sorted);
        }
        state_offsets.push(all_edges.len() as u32);

        let num_states = self.edges.len();
        let mut byte_table = vec![0xFFFFu16; num_states * 256];
        for s in 0..num_states {
            let start = state_offsets[s] as usize;
            let end = state_offsets[s + 1] as usize;
            for edge in &all_edges[start..end] {
                if let FsmEdge::CharRange { min, max, target } = edge {
                    for b in *min..=*max {
                        byte_table[s * 256 + b as usize] = target.0 as u16;
                    }
                }
            }
        }

        DfaTable {
            edges: all_edges,
            state_offsets,
            byte_table,
        }
    }
}

impl Default for NfaGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct DfaTable {
    edges: Vec<FsmEdge>,
    state_offsets: Vec<u32>,
    byte_table: Vec<u16>,
}

impl DfaTable {
    pub fn num_states(&self) -> usize {
        self.state_offsets.len() - 1
    }

    pub fn edges(&self, state: StateId) -> &[FsmEdge] {
        let s = state.0 as usize;
        let start = self.state_offsets[s] as usize;
        let end = self.state_offsets[s + 1] as usize;
        &self.edges[start..end]
    }

    #[inline(always)]
    pub fn byte_table(&self) -> &[u16] {
        &self.byte_table
    }

    #[inline(always)]
    pub fn next_state(&self, from: StateId, value: u8) -> Option<StateId> {
        let target = self.byte_table[from.0 as usize * 256 + value as usize];
        if target != 0xFFFF {
            Some(StateId(target as u32))
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct Automaton<F> {
    pub fsm: F,
    pub start: StateId,
    pub ends: Vec<bool>,
}

impl Automaton<NfaGraph> {
    pub fn is_end(&self, state: StateId) -> bool {
        self.ends.get(state.0 as usize).copied().unwrap_or(false)
    }

    pub(crate) fn to_dfa_limited(&self, max_states: usize) -> Result<Automaton<NfaGraph>> {
        let mut dfa = NfaGraph::new();
        let mut dfa_ends = Vec::new();

        let mut state_map: HashMap<BTreeSet<StateId>, StateId> = HashMap::new();
        let mut worklist: VecDeque<BTreeSet<StateId>> = VecDeque::new();

        let get_or_create = |target_set: BTreeSet<StateId>,
                             ends: &Vec<bool>,
                             dfa: &mut NfaGraph,
                             dfa_ends: &mut Vec<bool>,
                             state_map: &mut HashMap<BTreeSet<StateId>, StateId>,
                             worklist: &mut VecDeque<BTreeSet<StateId>>|
         -> Result<StateId> {
            if let Some(&existing) = state_map.get(&target_set) {
                Ok(existing)
            } else {
                if dfa.num_states() >= max_states {
                    bail!("DFA state limit {} exceeded", max_states);
                }
                let new_id = dfa.add_state();
                dfa_ends.push(target_set.iter().any(|s| ends[s.0 as usize]));
                state_map.insert(target_set.clone(), new_id);
                worklist.push_back(target_set);
                Ok(new_id)
            }
        };

        let start_set = {
            let mut s = BTreeSet::new();
            s.insert(self.start);
            self.fsm.epsilon_closure(&s)
        };

        let dfa_start = dfa.add_state();
        dfa_ends.push(start_set.iter().any(|s| self.is_end(*s)));
        state_map.insert(start_set.clone(), dfa_start);
        worklist.push_back(start_set);

        while let Some(nfa_states) = worklist.pop_front() {
            let dfa_state = state_map[&nfa_states];

            let intervals = self.collect_intervals(&nfa_states);

            for (min, max, targets) in intervals {
                let target_set = self.fsm.epsilon_closure(&targets);
                if target_set.is_empty() {
                    continue;
                }
                let dfa_target = get_or_create(
                    target_set,
                    &self.ends,
                    &mut dfa,
                    &mut dfa_ends,
                    &mut state_map,
                    &mut worklist,
                )?;
                dfa.add_char_edge(dfa_state, min, max, dfa_target);
            }

            for &nfa_state in &nfa_states {
                for edge in self.fsm.edges(nfa_state) {
                    let FsmEdge::RuleRef {
                        rule,
                        target: target_nfa,
                    } = edge
                    else {
                        continue;
                    };
                    let target_set = {
                        let mut s = BTreeSet::new();
                        s.insert(*target_nfa);
                        self.fsm.epsilon_closure(&s)
                    };
                    if target_set.is_empty() {
                        continue;
                    }
                    let dfa_target = get_or_create(
                        target_set,
                        &self.ends,
                        &mut dfa,
                        &mut dfa_ends,
                        &mut state_map,
                        &mut worklist,
                    )?;
                    dfa.add_rule_ref(dfa_state, *rule, dfa_target);
                }
            }
        }

        Ok(Automaton {
            fsm: dfa,
            start: dfa_start,
            ends: dfa_ends,
        })
    }

    fn collect_intervals(
        &self,
        nfa_states: &BTreeSet<StateId>,
    ) -> Vec<(u8, u8, BTreeSet<StateId>)> {
        let mut ranges: Vec<(u8, u8, StateId)> = Vec::new();
        for &state in nfa_states {
            for edge in self.fsm.edges(state) {
                if let FsmEdge::CharRange { min, max, target } = edge {
                    ranges.push((*min, *max, *target));
                }
            }
        }

        if ranges.is_empty() {
            return Vec::new();
        }

        let mut points: BTreeSet<u16> = BTreeSet::new();
        for &(min, max, _) in &ranges {
            points.insert(min as u16);
            if (max as u16) < 255 {
                points.insert(max as u16 + 1);
            }
        }

        let points: Vec<u16> = points.into_iter().collect();
        let mut result = Vec::new();

        for (i, &start) in points.iter().enumerate() {
            let end = if i + 1 < points.len() {
                points[i + 1] - 1
            } else {
                255
            };

            let mut targets = BTreeSet::new();
            for &(min, max, target) in &ranges {
                if (min as u16) <= start && end <= (max as u16) {
                    targets.insert(target);
                }
            }

            if !targets.is_empty() {
                result.push((start as u8, end as u8, targets));
            }
        }

        let mut merged: Vec<(u8, u8, BTreeSet<StateId>)> = Vec::new();
        for (min, max, targets) in result {
            if let Some(last) = merged.last_mut()
                && last.2 == targets
                && last.1.checked_add(1) == Some(min)
            {
                last.1 = max;
                continue;
            }
            merged.push((min, max, targets));
        }

        merged
    }

    pub fn to_compact(&self) -> Automaton<DfaTable> {
        Automaton {
            fsm: self.fsm.to_compact(),
            start: self.start,
            ends: self.ends.clone(),
        }
    }
}

impl Automaton<DfaTable> {}

fn build_char_class_nfa(
    fsm: &mut NfaGraph,
    negated: bool,
    ranges: &[(u32, u32)],
    start: StateId,
    end: StateId,
) {
    let effective_ranges = if negated {
        complement_codepoint_ranges(ranges)
    } else {
        ranges.to_vec()
    };

    for &(lo, hi) in &effective_ranges {
        add_codepoint_range_nfa(fsm, lo, hi, start, end);
    }
}

fn complement_codepoint_ranges(ranges: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut sorted = ranges.to_vec();
    sorted.sort_by_key(|&(lo, _)| lo);
    let mut merged: Vec<(u32, u32)> = Vec::new();
    for (lo, hi) in sorted {
        if let Some(last) = merged.last_mut()
            && lo <= last.1 + 1
        {
            last.1 = last.1.max(hi);
            continue;
        }
        merged.push((lo, hi));
    }

    let mut complement = Vec::new();
    let mut prev_end: u32 = 0;
    for &(lo, hi) in &merged {
        if lo > prev_end {
            complement.push((prev_end, lo - 1));
        }
        prev_end = hi.saturating_add(1);
    }
    if prev_end <= 0x10FFFF {
        complement.push((prev_end, 0x10FFFF));
    }

    let mut result = Vec::new();
    for &(lo, hi) in &complement {
        if hi < 0xD800 || lo > 0xDFFF {
            result.push((lo, hi));
        } else {
            if lo < 0xD800 {
                result.push((lo, 0xD7FF));
            }
            if hi > 0xDFFF {
                result.push((0xE000, hi));
            }
        }
    }
    result
}

fn add_codepoint_range_nfa(fsm: &mut NfaGraph, lo: u32, hi: u32, start: StateId, end: StateId) {
    let ascii_lo = lo;
    let ascii_hi = hi.min(0x7F);
    if ascii_lo <= ascii_hi {
        fsm.add_char_edge(start, ascii_lo as u8, ascii_hi as u8, end);
    }

    let two_lo = lo.max(0x80);
    let two_hi = hi.min(0x7FF);
    if two_lo <= two_hi {
        add_utf8_nfa_range(fsm, two_lo, two_hi, start, end);
    }

    let three_lo = lo.max(0x800);
    let three_hi = hi.min(0xD7FF);
    if three_lo <= three_hi {
        add_utf8_nfa_range(fsm, three_lo, three_hi, start, end);
    }

    let three_lo2 = lo.max(0xE000);
    let three_hi2 = hi.min(0xFFFF);
    if three_lo2 <= three_hi2 {
        add_utf8_nfa_range(fsm, three_lo2, three_hi2, start, end);
    }

    let four_lo = lo.max(0x10000);
    let four_hi = hi.min(0x10FFFF);
    if four_lo <= four_hi {
        add_utf8_nfa_range(fsm, four_lo, four_hi, start, end);
    }
}

fn encode_codepoint_utf8(cp: u32) -> Vec<u8> {
    let c = char::from_u32(cp).expect("valid codepoint");
    let mut buf = [0u8; 4];
    let s = c.encode_utf8(&mut buf);
    s.as_bytes().to_vec()
}

fn add_utf8_nfa_range(fsm: &mut NfaGraph, lo: u32, hi: u32, start: StateId, end: StateId) {
    let lo_bytes = encode_codepoint_utf8(lo);
    let hi_bytes = encode_codepoint_utf8(hi);
    debug_assert_eq!(lo_bytes.len(), hi_bytes.len());
    add_utf8_byte_range(fsm, &lo_bytes, &hi_bytes, 0, start, end);
}

fn add_utf8_byte_range(
    fsm: &mut NfaGraph,
    lo: &[u8],
    hi: &[u8],
    depth: usize,
    start: StateId,
    end: StateId,
) {
    if depth == lo.len() - 1 {
        fsm.add_char_edge(start, lo[depth], hi[depth], end);
        return;
    }

    if lo[depth] == hi[depth] {
        let mid = fsm.add_state();
        fsm.add_char_edge(start, lo[depth], hi[depth], mid);
        add_utf8_byte_range(fsm, lo, hi, depth + 1, mid, end);
        return;
    }

    {
        let s = fsm.add_state();
        fsm.add_char_edge(start, lo[depth], lo[depth], s);
        let mut hi_full = lo.to_vec();
        hi_full[depth + 1..].fill(0xBF);
        add_utf8_byte_range(fsm, lo, &hi_full, depth + 1, s, end);
    }

    if lo[depth] < hi[depth].saturating_sub(1) {
        let s = fsm.add_state();
        fsm.add_char_edge(start, lo[depth] + 1, hi[depth] - 1, s);
        let mut lo_min = lo.to_vec();
        let mut hi_max = hi.to_vec();
        lo_min[depth + 1..].fill(0x80);
        hi_max[depth + 1..].fill(0xBF);
        add_utf8_byte_range(fsm, &lo_min, &hi_max, depth + 1, s, end);
    }

    {
        let s = fsm.add_state();
        fsm.add_char_edge(start, hi[depth], hi[depth], s);
        let mut lo_min = hi.to_vec();
        lo_min[depth + 1..].fill(0x80);
        add_utf8_byte_range(fsm, &lo_min, hi, depth + 1, s, end);
    }
}

fn is_inlineable(grammar: &Grammar, expr_id: ExprId, known: &HashSet<RuleId>) -> bool {
    match grammar.get_expr(expr_id) {
        Expr::EmptyString
        | Expr::ByteString(_)
        | Expr::CharacterClass { .. }
        | Expr::CharacterClassStar { .. } => true,
        Expr::RuleRef(rid) => known.contains(rid),
        Expr::Repeat { rule, .. } => known.contains(rule),
        Expr::Sequence(es) => es.iter().all(|e| is_inlineable(grammar, *e, known)),
        Expr::Choices(es) => es.iter().all(|e| is_inlineable(grammar, *e, known)),
    }
}

fn find_inlineable_rules(grammar: &Grammar) -> HashSet<RuleId> {
    let mut inlineable = HashSet::new();
    loop {
        let mut changed = false;
        for (i, rule) in grammar.rules().iter().enumerate() {
            let rid = RuleId(i as u32);
            if !inlineable.contains(&rid) && is_inlineable(grammar, rule.body, &inlineable) {
                inlineable.insert(rid);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    inlineable
}

fn build_expr_nfa_inlining(
    grammar: &Grammar,
    fsm: &mut NfaGraph,
    expr_id: ExprId,
    start: StateId,
    end: StateId,
    inlineable: &HashSet<RuleId>,
) {
    match grammar.get_expr(expr_id) {
        Expr::EmptyString => {
            fsm.add_epsilon(start, end);
        }

        Expr::ByteString(bytes) => {
            if bytes.is_empty() {
                fsm.add_epsilon(start, end);
                return;
            }
            let mut prev = start;
            for (i, &byte) in bytes.iter().enumerate() {
                let next = if i + 1 == bytes.len() {
                    end
                } else {
                    fsm.add_state()
                };
                fsm.add_char_edge(prev, byte, byte, next);
                prev = next;
            }
        }

        Expr::CharacterClass { negated, ranges } => {
            build_char_class_nfa(fsm, *negated, ranges, start, end);
        }

        Expr::CharacterClassStar { negated, ranges } => {
            fsm.add_epsilon(start, end);
            let effective_ranges = if *negated {
                complement_codepoint_ranges(ranges)
            } else {
                ranges.to_vec()
            };
            for &(lo, hi) in &effective_ranges {
                add_codepoint_range_nfa(fsm, lo, hi, start, start);
            }
        }

        Expr::RuleRef(rule_id) => {
            if inlineable.contains(rule_id) {
                let body = grammar.get_rule(*rule_id).body;
                build_expr_nfa_inlining(grammar, fsm, body, start, end, inlineable);
            } else {
                fsm.add_rule_ref(start, *rule_id, end);
            }
        }

        Expr::Sequence(exprs) => {
            if exprs.is_empty() {
                fsm.add_epsilon(start, end);
                return;
            }
            let mut prev = start;
            for (i, &eid) in exprs.iter().enumerate() {
                let next = if i + 1 == exprs.len() {
                    end
                } else {
                    fsm.add_state()
                };
                build_expr_nfa_inlining(grammar, fsm, eid, prev, next, inlineable);
                prev = next;
            }
        }

        Expr::Choices(exprs) => {
            for &eid in exprs {
                build_expr_nfa_inlining(grammar, fsm, eid, start, end, inlineable);
            }
        }

        Expr::Repeat { rule, min, max } => {
            let min = *min;
            let max = *max;
            let rule = *rule;

            if min == 0 && max == Some(0) {
                fsm.add_epsilon(start, end);
                return;
            }

            if inlineable.contains(&rule) {
                let body = grammar.get_rule(rule).body;
                build_inlined_repeat(grammar, fsm, body, min, max, start, end, inlineable);
            } else {
                let mut prev = start;
                for i in 0..min {
                    let next = if max == Some(min) && i + 1 == min {
                        end
                    } else {
                        fsm.add_state()
                    };
                    fsm.add_rule_ref(prev, rule, next);
                    prev = next;
                }
                if let Some(max) = max {
                    for i in min..max {
                        if prev != end {
                            fsm.add_epsilon(prev, end);
                        }
                        let next = if i + 1 == max { end } else { fsm.add_state() };
                        fsm.add_rule_ref(prev, rule, next);
                        prev = next;
                    }
                } else {
                    fsm.add_epsilon(prev, end);
                    fsm.add_rule_ref(prev, rule, prev);
                }
            }
        }
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "grammar, graph, body, bounds and endpoints are each independent \
              inputs to one wiring step; a struct would only rename them"
)]
fn build_inlined_repeat(
    grammar: &Grammar,
    fsm: &mut NfaGraph,
    body: ExprId,
    min: u32,
    max: Option<u32>,
    start: StateId,
    end: StateId,
    inlineable: &HashSet<RuleId>,
) {
    let mut prev = start;

    for i in 0..min {
        let next = if max == Some(min) && i + 1 == min {
            end
        } else {
            fsm.add_state()
        };
        build_expr_nfa_inlining(grammar, fsm, body, prev, next, inlineable);
        prev = next;
    }

    if let Some(max) = max {
        for i in min..max {
            if prev != end {
                fsm.add_epsilon(prev, end);
            }
            let next = if i + 1 == max { end } else { fsm.add_state() };
            build_expr_nfa_inlining(grammar, fsm, body, prev, next, inlineable);
            prev = next;
        }
    } else {
        fsm.add_epsilon(prev, end);
        build_expr_nfa_inlining(grammar, fsm, body, prev, prev, inlineable);
    }
}

pub fn build_rule_fsms(grammar: &Grammar) -> Vec<Automaton<NfaGraph>> {
    let inlineable = find_inlineable_rules(grammar);
    let mut result = Vec::new();

    for rule in grammar.rules() {
        let mut fsm = NfaGraph::new();
        let start = fsm.add_state();
        let end = fsm.add_state();

        build_expr_nfa_inlining(grammar, &mut fsm, rule.body, start, end, &inlineable);

        let mut ends = vec![false; fsm.num_states()];
        ends[end.0 as usize] = true;

        result.push(Automaton { fsm, start, ends });
    }

    result
}
