use std::cell::Cell;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use rustc_hash::FxHashSet;

use crate::compiled_grammar::CompiledGrammar;
use crate::fsm::{FsmEdge, StateId};
use crate::grammar::RuleId;

pub(super) const NO_PARENT: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub(super) struct StackState {
    pub(super) rule_id: u16,
    pub(super) dfa_state: u16,
    pub(super) return_level: u32,
}

impl Hash for StackState {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.packed().hash(state);
    }
}

impl StackState {
    #[inline(always)]
    fn packed(self) -> u64 {
        (self.rule_id as u64) | ((self.dfa_state as u64) << 16) | ((self.return_level as u64) << 32)
    }
}

const SMALL_DEDUP_THRESHOLD: usize = 12;

#[derive(Clone)]
pub(super) struct SmallDedup<T: Eq + Hash + Copy> {
    vec: Vec<T>,
    set: Option<FxHashSet<T>>,
}

impl<T: Eq + Hash + Copy> Default for SmallDedup<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Eq + Hash + Copy> SmallDedup<T> {
    pub(super) fn new() -> Self {
        Self {
            vec: Vec::new(),
            set: None,
        }
    }

    fn clear(&mut self) {
        self.vec.clear();
        if let Some(ref mut set) = self.set {
            set.clear();
        }
    }

    fn insert(&mut self, item: T) -> bool {
        if let Some(ref mut set) = self.set {
            if set.insert(item) {
                self.vec.push(item);
                return true;
            }
            return false;
        }

        if self.vec.contains(&item) {
            return false;
        }
        self.vec.push(item);

        if self.vec.len() >= SMALL_DEDUP_THRESHOLD {
            let mut set = FxHashSet::default();
            for &x in &self.vec {
                set.insert(x);
            }
            self.set = Some(set);
        }
        true
    }
}

enum SteadyAdvance {
    InRange,
    OutOfRange,
    NotActive,
}

#[derive(Clone)]
struct SteadyState {
    active: bool,
    ranges: Vec<(u8, u8)>,
    is_completed: bool,
    is_lazy: bool,
    count: usize,
    state_deltas: Vec<i32>,
    return_deltas: Vec<i32>,
}

impl SteadyState {
    fn new() -> Self {
        Self {
            active: false,
            ranges: Vec::new(),
            is_completed: false,
            is_lazy: false,
            count: 0,
            state_deltas: Vec::new(),
            return_deltas: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.active = false;
        self.is_lazy = false;
        self.count = 0;
        self.state_deltas.clear();
        self.return_deltas.clear();
    }

    fn try_advance(&self, byte: u8) -> SteadyAdvance {
        if !self.active {
            return SteadyAdvance::NotActive;
        }
        if self.ranges.iter().any(|&(lo, hi)| byte >= lo && byte <= hi) {
            SteadyAdvance::InRange
        } else {
            SteadyAdvance::OutOfRange
        }
    }
}

#[derive(Clone)]
pub(super) struct StackParser {
    compiled: Arc<CompiledGrammar>,
    state_arena: Vec<StackState>,
    state_offsets: Vec<usize>,
    return_arena: Vec<(u16, StackState)>,
    return_offsets: Vec<usize>,
    is_completed: Vec<bool>,
    buf_queue: Vec<StackState>,
    buf_visited: SmallDedup<StackState>,
    buf_scanable: Vec<StackState>,
    buf_return: Vec<(u16, StackState)>,
    steady: SteadyState,
    chain_terminal: Cell<Option<(u16, u16, usize, usize)>>,
}

impl StackParser {
    pub(super) fn new(compiled: Arc<CompiledGrammar>) -> Self {
        let mut parser = Self {
            compiled,
            state_arena: Vec::with_capacity(128),
            state_offsets: Vec::with_capacity(32),
            return_arena: Vec::with_capacity(64),
            return_offsets: Vec::with_capacity(32),
            is_completed: Vec::with_capacity(32),
            buf_queue: Vec::new(),
            buf_visited: SmallDedup::new(),
            buf_scanable: Vec::new(),
            buf_return: Vec::new(),
            steady: SteadyState::new(),
            chain_terminal: Cell::new(None),
        };
        parser.init();
        parser
    }

    fn init(&mut self) {
        self.state_arena.clear();
        self.state_offsets.clear();
        self.return_arena.clear();
        self.return_offsets.clear();
        self.is_completed.clear();

        let mut queue = std::mem::take(&mut self.buf_queue);
        let mut visited = std::mem::take(&mut self.buf_visited);
        let mut scanable = std::mem::take(&mut self.buf_scanable);
        let mut returns = std::mem::take(&mut self.buf_return);
        queue.clear();
        visited.clear();
        scanable.clear();
        returns.clear();
        let mut accept_stop = false;

        let root = self.compiled.grammar.root_rule();
        self.expand_rule(root, NO_PARENT, &mut queue, &mut visited);

        self.process_queue(
            &mut queue,
            &mut visited,
            &mut scanable,
            &mut returns,
            &mut accept_stop,
            &[],
        );

        self.state_offsets.push(self.state_arena.len());
        self.state_arena.extend_from_slice(&scanable);
        self.return_offsets.push(self.return_arena.len());
        self.return_arena.extend_from_slice(&returns);
        self.is_completed.push(accept_stop);

        scanable.clear();
        returns.clear();
        queue.clear();
        visited.clear();
        self.buf_scanable = scanable;
        self.buf_return = returns;
        self.buf_queue = queue;
        self.buf_visited = visited;
    }

    pub(super) fn advance(&mut self, ch: u8) -> bool {
        match self.steady.try_advance(ch) {
            SteadyAdvance::InRange => {
                if self.steady.is_lazy {
                    self.steady.count += 1;
                } else {
                    self.advance_steady_with_delta();
                }
                return true;
            }
            SteadyAdvance::OutOfRange => {
                if self.steady.is_lazy {
                    self.flush_steady();
                }
                self.steady.active = false;
            }
            SteadyAdvance::NotActive => {}
        }

        let state_start = match self.state_offsets.last() {
            Some(&start) => start,
            None => return false,
        };
        let state_end = self.state_arena.len();

        let mut queue = std::mem::take(&mut self.buf_queue);
        let mut visited = std::mem::take(&mut self.buf_visited);
        let mut scanable = std::mem::take(&mut self.buf_scanable);
        let mut returns = std::mem::take(&mut self.buf_return);
        queue.clear();
        visited.clear();
        scanable.clear();
        returns.clear();

        self.scan_states(
            &self.state_arena[state_start..state_end],
            ch,
            &mut queue,
            &mut visited,
            &mut scanable,
        );

        if queue.is_empty() && scanable.is_empty() {
            self.buf_queue = queue;
            self.buf_visited = visited;
            self.buf_scanable = scanable;
            self.buf_return = returns;
            return false;
        }

        let mut accept_stop = false;

        if !queue.is_empty() {
            self.process_queue(
                &mut queue,
                &mut visited,
                &mut scanable,
                &mut returns,
                &mut accept_stop,
                &[],
            );
        }

        let new_offset = self.state_arena.len();
        self.state_offsets.push(new_offset);
        self.state_arena.extend_from_slice(&scanable);
        let new_ret_offset = self.return_arena.len();
        self.return_offsets.push(new_ret_offset);
        self.return_arena.extend_from_slice(&returns);
        self.is_completed.push(accept_stop);

        scanable.clear();
        returns.clear();
        queue.clear();
        visited.clear();
        self.buf_scanable = scanable;
        self.buf_return = returns;
        self.buf_queue = queue;
        self.buf_visited = visited;

        self.detect_and_enter_steady_state(ch);

        true
    }

    pub(super) fn advance_bytes(&mut self, bytes: &[u8]) -> bool {
        let start = self.position();
        for &byte in bytes {
            if !self.advance(byte) {
                self.pop_last_states(self.position() - start);
                return false;
            }
        }
        true
    }

    fn advance_steady_with_delta(&mut self) {
        let prev_start = *self.state_offsets.last().unwrap();
        let state_count = self.state_arena.len() - prev_start;
        let new_start = self.state_arena.len();
        self.state_offsets.push(new_start);
        for i in 0..state_count {
            let mut s = self.state_arena[prev_start + i];
            if s.return_level != NO_PARENT {
                s.return_level =
                    (s.return_level as i64 + self.steady.state_deltas[i] as i64) as u32;
            }
            self.state_arena.push(s);
        }
        let prev_rstart = *self.return_offsets.last().unwrap();
        let ret_count = self.return_arena.len() - prev_rstart;
        let new_rstart = self.return_arena.len();
        self.return_offsets.push(new_rstart);
        for i in 0..ret_count {
            let (expected, mut parent) = self.return_arena[prev_rstart + i];
            if parent.return_level != NO_PARENT {
                parent.return_level =
                    (parent.return_level as i64 + self.steady.return_deltas[i] as i64) as u32;
            }
            self.return_arena.push((expected, parent));
        }
        self.is_completed.push(self.steady.is_completed);
        if let Some((rid, dfa, terminal, last)) = self.chain_terminal.get() {
            self.chain_terminal
                .set(Some((rid, dfa, terminal, last + 1)));
        }
    }

    fn scan_states(
        &self,
        states: &[StackState],
        ch: u8,
        queue: &mut Vec<StackState>,
        visited: &mut SmallDedup<StackState>,
        scanable: &mut Vec<StackState>,
    ) {
        for &state in states {
            let dfa = &self.compiled.rule_dfas[state.rule_id as usize];
            if let Some(next_dfa) = dfa.fsm.next_state(StateId(state.dfa_state as u32), ch) {
                let next = StackState {
                    rule_id: state.rule_id,
                    dfa_state: next_dfa.0 as u16,
                    return_level: state.return_level,
                };
                let flags = self.compiled.action(next.rule_id, next.dfa_state).flags;
                if flags.has_char_edges() && !flags.has_rule_ref() && !flags.is_accepting() {
                    if visited.insert(next) {
                        scanable.push(next);
                    }
                } else if visited.insert(next) {
                    queue.push(next);
                }
            }
        }
    }

    fn process_queue(
        &self,
        queue: &mut Vec<StackState>,
        visited: &mut SmallDedup<StackState>,
        scanable: &mut Vec<StackState>,
        returns: &mut Vec<(u16, StackState)>,
        accept_stop: &mut bool,
        extra_returns: &[(u16, StackState)],
    ) {
        let current_level = self.state_offsets.len() as u32;
        let mut completed_at_level = Vec::with_capacity(8);

        let mut idx = 0;
        while idx < queue.len() {
            let state = queue[idx];
            idx += 1;

            let action = self.compiled.action(state.rule_id, state.dfa_state);

            for &(rule_id, target) in &action.rule_refs {
                let parent_after = StackState {
                    rule_id: state.rule_id,
                    dfa_state: target,
                    return_level: state.return_level,
                };
                returns.push((rule_id, parent_after));

                let parent_action = self
                    .compiled
                    .action(parent_after.rule_id, parent_after.dfa_state);
                if parent_action.flags.is_pass_through() && parent_after.return_level != NO_PARENT {
                    let level = parent_after.return_level as usize;
                    if level < self.return_offsets.len() {
                        let rstart = self.return_offsets[level];
                        let rend = self
                            .return_offsets
                            .get(level + 1)
                            .copied()
                            .unwrap_or(self.return_arena.len());
                        for i in rstart..rend {
                            let (expected, grandparent) = self.return_arena[i];
                            if expected == parent_after.rule_id {
                                returns.push((rule_id, grandparent));
                            }
                        }
                    }
                }

                if !self.expand_rule(RuleId(rule_id as u32), current_level, queue, visited)
                    && completed_at_level.contains(&rule_id)
                    && visited.insert(parent_after)
                {
                    queue.push(parent_after);
                }
            }

            if action.flags.is_accepting() {
                if state.return_level == current_level
                    && !completed_at_level.contains(&state.rule_id)
                {
                    completed_at_level.push(state.rule_id);
                }
                self.complete(&state, queue, visited, returns, accept_stop, extra_returns);
            }

            if action.flags.has_char_edges() {
                scanable.push(state);
            }
        }
    }

    fn complete(
        &self,
        state: &StackState,
        queue: &mut Vec<StackState>,
        visited: &mut SmallDedup<StackState>,
        returns: &[(u16, StackState)],
        accept_stop: &mut bool,
        extra_returns: &[(u16, StackState)],
    ) {
        if state.return_level == NO_PARENT {
            *accept_stop = true;
            return;
        }

        let start_pos = state.return_level as usize;
        let rule_id = state.rule_id;

        if start_pos < self.return_offsets.len() {
            let rstart = self.return_offsets[start_pos];
            let rend = self
                .return_offsets
                .get(start_pos + 1)
                .copied()
                .unwrap_or(self.return_arena.len());
            for i in rstart..rend {
                let (expected_rule, parent_after) = self.return_arena[i];
                if expected_rule == rule_id {
                    if self.compiled.has_self_ref_chains
                        && parent_after.rule_id == state.rule_id
                        && parent_after.return_level != NO_PARENT
                        && (parent_after.return_level as usize) < start_pos
                        && self
                            .compiled
                            .action(parent_after.rule_id, parent_after.dfa_state)
                            .flags
                            .is_pass_through()
                    {
                        self.follow_chain_to_terminal(
                            parent_after.rule_id,
                            parent_after.dfa_state,
                            parent_after.return_level as usize,
                            queue,
                            visited,
                            accept_stop,
                        );
                        continue;
                    }
                    if visited.insert(parent_after) {
                        queue.push(parent_after);
                    }
                }
            }
        }

        let current_level = self.state_offsets.len();
        if start_pos == current_level {
            for &(expected_rule, parent_after) in returns {
                if expected_rule == rule_id && visited.insert(parent_after) {
                    queue.push(parent_after);
                }
            }
            for &(expected_rule, parent_after) in extra_returns {
                if expected_rule == rule_id && visited.insert(parent_after) {
                    queue.push(parent_after);
                }
            }
        }
    }

    fn follow_chain_to_terminal(
        &self,
        chain_rule_id: u16,
        chain_dfa_state: u16,
        start_level: usize,
        queue: &mut Vec<StackState>,
        visited: &mut SmallDedup<StackState>,
        accept_stop: &mut bool,
    ) {
        if let Some((cached_rid, cached_dfa, terminal, last_start)) = self.chain_terminal.get()
            && cached_rid == chain_rule_id
            && cached_dfa == chain_dfa_state
            && start_level == last_start + 1
            && terminal < self.return_offsets.len()
        {
            self.process_terminal_returns(chain_rule_id, terminal, queue, visited, accept_stop);
            self.chain_terminal
                .set(Some((cached_rid, cached_dfa, terminal, start_level)));
            return;
        }

        let mut level = start_level;
        loop {
            if level >= self.return_offsets.len() {
                break;
            }
            let rstart = self.return_offsets[level];
            let rend = self
                .return_offsets
                .get(level + 1)
                .copied()
                .unwrap_or(self.return_arena.len());

            let mut chain_target = None;
            for i in rstart..rend {
                let (expected, parent) = self.return_arena[i];
                if expected == chain_rule_id {
                    if parent.rule_id == chain_rule_id
                        && parent.dfa_state == chain_dfa_state
                        && parent.return_level != NO_PARENT
                        && (parent.return_level as usize) < level
                    {
                        chain_target = Some(parent.return_level as usize);
                    } else {
                        if visited.insert(parent) {
                            queue.push(parent);
                        }
                    }
                }
            }
            if let Some(next_level) = chain_target {
                level = next_level;
            } else {
                self.chain_terminal
                    .set(Some((chain_rule_id, chain_dfa_state, level, start_level)));
                break;
            }
        }
    }

    fn process_terminal_returns(
        &self,
        chain_rule_id: u16,
        terminal_level: usize,
        queue: &mut Vec<StackState>,
        visited: &mut SmallDedup<StackState>,
        _accept_stop: &mut bool,
    ) {
        let rstart = self.return_offsets[terminal_level];
        let rend = self
            .return_offsets
            .get(terminal_level + 1)
            .copied()
            .unwrap_or(self.return_arena.len());

        for i in rstart..rend {
            let (expected, parent) = self.return_arena[i];
            if expected == chain_rule_id && visited.insert(parent) {
                queue.push(parent);
            }
        }
    }

    fn expand_rule(
        &self,
        rule_id: RuleId,
        return_level: u32,
        queue: &mut Vec<StackState>,
        visited: &mut SmallDedup<StackState>,
    ) -> bool {
        let dfa = &self.compiled.rule_dfas[rule_id.0 as usize];
        let state = StackState {
            rule_id: rule_id.0 as u16,
            dfa_state: dfa.start.0 as u16,
            return_level,
        };
        if visited.insert(state) {
            queue.push(state);
            true
        } else {
            false
        }
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "the four `_buf` parameters exist so the caller can reuse \
                  allocations across a trie walk; folding them into a struct \
                  would hide the borrow that makes the reuse safe"
    )]
    pub(super) fn probe_advance_reuse(
        &self,
        current_states: &[StackState],
        extra_returns: &[(u16, StackState)],
        ch: u8,
        queue_buf: &mut Vec<StackState>,
        visited_buf: &mut SmallDedup<StackState>,
        scanable_buf: &mut Vec<StackState>,
        returns_buf: &mut Vec<(u16, StackState)>,
    ) -> bool {
        queue_buf.clear();
        visited_buf.clear();
        scanable_buf.clear();
        returns_buf.clear();

        self.scan_states(current_states, ch, queue_buf, visited_buf, scanable_buf);

        if queue_buf.is_empty() && scanable_buf.is_empty() {
            return false;
        }

        let mut accept_stop = false;
        if !queue_buf.is_empty() {
            self.process_queue(
                queue_buf,
                visited_buf,
                scanable_buf,
                returns_buf,
                &mut accept_stop,
                extra_returns,
            );
        }

        true
    }

    pub(super) fn is_completed(&self) -> bool {
        if self.steady.count > 0 {
            return self.steady.is_completed;
        }
        self.is_completed.last().copied().unwrap_or(false)
    }

    pub(super) fn write_cache_key(&self, key: &mut Vec<u64>) {
        let states = self.current_states();
        let returns = self.current_returns();
        key.clear();
        key.reserve(4 + states.len() + returns.len() * 2);
        key.extend([1, states.len() as u64]);
        key.extend(states.iter().map(|state| state.packed()));
        key.push(returns.len() as u64);
        for &(expected_rule, state) in returns {
            key.push(expected_rule as u64);
            key.push(state.packed());
        }
        key.push(self.is_completed() as u64);
    }

    pub(super) fn current_states(&self) -> &[StackState] {
        if let Some(&start) = self.state_offsets.last() {
            &self.state_arena[start..]
        } else {
            &[]
        }
    }

    pub(super) fn current_returns(&self) -> &[(u16, StackState)] {
        if let Some(&start) = self.return_offsets.last() {
            &self.return_arena[start..]
        } else {
            &[]
        }
    }

    pub(super) fn position(&self) -> usize {
        self.state_offsets.len().saturating_sub(1) + self.steady.count
    }

    pub(super) fn pop_last_states(&mut self, count: usize) {
        if count == 0 {
            return;
        }
        self.chain_terminal.set(None);
        if self.steady.count > 0 {
            let from_lazy = count.min(self.steady.count);
            self.steady.count -= from_lazy;
            if from_lazy == count {
                return;
            }
            self.steady.active = false;
            self.pop_committed(count - from_lazy);
            return;
        }
        self.steady.active = false;
        self.pop_committed(count);
    }

    fn pop_committed(&mut self, count: usize) {
        for _ in 0..count {
            if self.state_offsets.len() > 1 {
                let start = self.state_offsets.pop().unwrap();
                self.state_arena.truncate(start);
                let rstart = self.return_offsets.pop().unwrap();
                self.return_arena.truncate(rstart);
                self.is_completed.pop();
            }
        }
    }

    pub(super) fn reset(&mut self) {
        self.steady.reset();
        self.chain_terminal.set(None);
        self.init();
    }

    fn flush_steady(&mut self) {
        if self.steady.count == 0 {
            return;
        }
        let prev_start = *self.state_offsets.last().unwrap();
        let new_start = self.state_arena.len();
        self.state_arena.extend_from_within(prev_start..);
        self.state_offsets.push(new_start);

        let prev_rstart = *self.return_offsets.last().unwrap();
        let new_rstart = self.return_arena.len();
        self.return_arena.extend_from_within(prev_rstart..);
        self.return_offsets.push(new_rstart);

        self.is_completed.push(self.steady.is_completed);
        self.steady.count = 0;
    }

    fn detect_and_enter_steady_state(&mut self, ch: u8) {
        if self.state_offsets.len() < 2 {
            return;
        }

        let num = self.state_offsets.len();
        let prev_start = self.state_offsets[num - 2];
        let prev_end = self.state_offsets[num - 1];
        let curr_start = self.state_offsets[num - 1];
        let prev_states = &self.state_arena[prev_start..prev_end];
        let curr_states = &self.state_arena[curr_start..];

        if prev_states.len() != curr_states.len() {
            return;
        }

        let structurally_same = prev_states
            .iter()
            .zip(curr_states.iter())
            .all(|(a, b)| a.rule_id == b.rule_id && a.dfa_state == b.dfa_state);
        if !structurally_same {
            return;
        }

        let prev_rstart = self.return_offsets[num - 2];
        let prev_rend = self.return_offsets[num - 1];
        let curr_rstart = self.return_offsets[num - 1];
        let prev_returns = &self.return_arena[prev_rstart..prev_rend];
        let curr_returns = &self.return_arena[curr_rstart..];

        if prev_returns.len() != curr_returns.len() {
            return;
        }
        let returns_same = prev_returns.iter().zip(curr_returns.iter()).all(|(a, b)| {
            a.0 == b.0 && a.1.rule_id == b.1.rule_id && a.1.dfa_state == b.1.dfa_state
        });
        if !returns_same {
            return;
        }

        let state_deltas: Vec<i32> = prev_states
            .iter()
            .zip(curr_states.iter())
            .map(|(p, c)| {
                if p.return_level == NO_PARENT {
                    0
                } else {
                    c.return_level as i32 - p.return_level as i32
                }
            })
            .collect();

        if state_deltas.iter().any(|&d| !(0..=1).contains(&d)) {
            return;
        }

        let return_deltas: Vec<i32> = prev_returns
            .iter()
            .zip(curr_returns.iter())
            .map(|(p, c)| {
                if p.1.return_level == NO_PARENT {
                    0
                } else {
                    c.1.return_level as i32 - p.1.return_level as i32
                }
            })
            .collect();

        if return_deltas.iter().any(|&d| !(0..=1).contains(&d)) {
            return;
        }

        let all_zero =
            state_deltas.iter().all(|&d| d == 0) && return_deltas.iter().all(|&d| d == 0);

        if let Some(ranges) = self.extract_steady_ranges(curr_states, ch) {
            self.steady.ranges = ranges;
            self.steady.is_completed = *self.is_completed.last().unwrap();
            self.steady.is_lazy = all_zero;
            self.steady.state_deltas = state_deltas;
            self.steady.return_deltas = return_deltas;
            self.steady.active = true;
        }
    }

    fn extract_steady_ranges(&self, states: &[StackState], ch: u8) -> Option<Vec<(u8, u8)>> {
        let mut winner: Option<Vec<(u8, u8)>> = None;
        let mut found_direct = false;

        for state in states {
            let dfa = &self.compiled.rule_dfas[state.rule_id as usize];

            let ch_target = dfa.fsm.next_state(StateId(state.dfa_state as u32), ch);
            if ch_target.is_none() {
                continue;
            }
            found_direct = true;
            let ch_target = ch_target.unwrap();

            let edges = dfa.fsm.edges(StateId(state.dfa_state as u32));
            let ranges: Vec<(u8, u8)> = edges
                .iter()
                .filter_map(|e| {
                    if let FsmEdge::CharRange { min, max, target } = e {
                        if *target == ch_target {
                            Some((*min, *max))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();

            match &winner {
                None => winner = Some(ranges),
                Some(w) if *w == ranges => {}
                _ => return None,
            }
        }

        if found_direct { winner } else { None }
    }
}
