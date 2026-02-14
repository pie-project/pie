//! DFA-based pushdown automaton (stack parser).
//!
//! `StackParser` drives per-rule DFA transitions with a pushdown stack. States are
//! 8-byte `StackState` values (rule_id + dfa_state + return_level). The DFA encodes
//! all intra-rule transitions, so predict/complete cycles are only needed at rule
//! boundaries (RuleRef edges and accepting states).

use std::cell::Cell;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use rustc_hash::{FxHashSet, FxHasher};

use crate::structured::compiled_grammar::CompiledGrammar;
use crate::structured::fsm::{FsmEdge, StateId};
use crate::structured::grammar::RuleId;

// ---------------------------------------------------------------------------
// Stack state
// ---------------------------------------------------------------------------

pub(super) const NO_PARENT: u32 = u32::MAX;

/// DFA-based parser state: tracks position within a grammar rule via DFA state.
///
/// 8 bytes (down from 28-byte ParserState), since the DFA encodes all intra-rule
/// transitions (ByteString offsets, CharacterClass, CharacterClassStar, Repeat).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub(super) struct StackState {
    /// Which rule we're parsing.
    pub(super) rule_id: u16,
    /// Position within the rule's DFA.
    pub(super) dfa_state: u16,
    /// History level where this rule was entered (NO_PARENT for root).
    pub(super) return_level: u32,
}

impl Hash for StackState {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        // StackState is exactly 8 bytes (repr(C): u16+u16+u32) — hash as u64.
        let bits = (self.rule_id as u64)
            | ((self.dfa_state as u64) << 16)
            | ((self.return_level as u64) << 32);
        bits.hash(state);
    }
}

// ---------------------------------------------------------------------------
// SmallDedup — linear scan for small sets, FxHashSet above threshold
// ---------------------------------------------------------------------------

const SMALL_DEDUP_THRESHOLD: usize = 12;

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
        Self { vec: Vec::new(), set: None }
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

        // Linear scan for small sets
        if self.vec.iter().any(|x| *x == item) {
            return false;
        }
        self.vec.push(item);

        // Upgrade to hash set if threshold reached
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

// ---------------------------------------------------------------------------
// Steady-state acceleration
// ---------------------------------------------------------------------------

/// Result of checking whether a byte can be handled by steady-state.
enum SteadyAdvance {
    /// Byte is in range — steady-state handles it.
    InRange,
    /// Byte is out of range — exit steady-state, fall through to normal advance.
    OutOfRange,
    /// Steady-state is not active.
    NotActive,
}

/// Encapsulates the steady-state detection and acceleration fields.
///
/// When consecutive advance() calls produce structurally identical state sets
/// (same rule_id + dfa_state, only return_level differs by a uniform delta),
/// the parser enters steady-state mode. In lazy mode (zero delta), only a counter
/// is incremented. In delta mode, states are copied with adjusted return_levels.
struct SteadyState {
    active: bool,
    ranges: Vec<(u8, u8)>,
    is_completed: bool,
    /// When true, deltas are all zero and we use a lazy counter (no arena copies).
    /// When false, deltas are non-zero and we copy states per byte with delta adjustment.
    is_lazy: bool,
    /// Lazy byte count for zero-delta steady state.
    count: usize,
    /// Per-state return_level deltas for copy-with-delta steady state.
    state_deltas: Vec<i32>,
    /// Per-return parent return_level deltas for copy-with-delta steady state.
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

// ---------------------------------------------------------------------------
// Stack parser (DFA-based pushdown automaton)
// ---------------------------------------------------------------------------

/// DFA-based pushdown automaton parser.
///
/// Replaces the Earley parser with per-rule DFA transitions. States are 8-byte
/// `StackState` values (rule_id + dfa_state + return_level). The DFA encodes all
/// intra-rule transitions, so predict/complete cycles are only needed at rule
/// boundaries (RuleRef edges and accepting states).
pub(super) struct StackParser {
    compiled: Arc<CompiledGrammar>,
    /// Flat arena of scanable states across all levels.
    state_arena: Vec<StackState>,
    /// Start offset of each level in `state_arena`.
    state_offsets: Vec<usize>,
    /// Flat arena of return entries: (expected_rule_id, parent_state_after_advance).
    return_arena: Vec<(u16, StackState)>,
    /// Start offset of each level in `return_arena`.
    return_offsets: Vec<usize>,
    /// Whether the grammar can terminate at each position.
    is_completed: Vec<bool>,
    // Reusable scratch buffers
    buf_queue: Vec<StackState>,
    buf_visited: SmallDedup<StackState>,
    buf_scanable: Vec<StackState>,
    buf_return: Vec<(u16, StackState)>,
    // Steady-state acceleration
    steady: SteadyState,
    /// Cached terminal level for completion chain short-circuit.
    /// When a self-referencing rule (e.g., CharacterClassStar) completes,
    /// it creates a chain of ghost completions cascading through ALL previous
    /// return levels — O(N) per advance. This cache stores the terminal level
    /// where the chain reaches a different parent rule, enabling O(1) lookups.
    /// Format: (chain_rule_id, chain_dfa_state, terminal_level, last_walk_start).
    /// Cache hit requires start_level == last_walk_start + 1 (consecutive advance).
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

    /// Initialize the parser by expanding the root rule.
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

        // Expand root rule
        let root = self.compiled.grammar.root_rule();
        self.expand_rule(root, NO_PARENT, &mut queue, &mut visited);

        // Process queue (predict/complete)
        self.process_queue(
            &mut queue, &mut visited, &mut scanable, &mut returns,
            &mut accept_stop, &[],
        );

        // Append to arenas
        self.state_offsets.push(self.state_arena.len());
        self.state_arena.extend_from_slice(&scanable);
        self.return_offsets.push(self.return_arena.len());
        self.return_arena.extend_from_slice(&returns);
        self.is_completed.push(accept_stop);

        // Return scratch buffers
        scanable.clear();
        returns.clear();
        queue.clear();
        visited.clear();
        self.buf_scanable = scanable;
        self.buf_return = returns;
        self.buf_queue = queue;
        self.buf_visited = visited;
    }

    /// Advance the parser by one byte. Returns true if at least one state survived.
    pub(super) fn advance(&mut self, ch: u8) -> bool {
        // Fast path: steady-state
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

        // Scan phase: advance each active state by the byte via DFA
        self.scan_states(
            &self.state_arena[state_start..state_end],
            ch, &mut queue, &mut visited, &mut scanable,
        );

        if queue.is_empty() && scanable.is_empty() {
            self.buf_queue = queue;
            self.buf_visited = visited;
            self.buf_scanable = scanable;
            self.buf_return = returns;
            return false;
        }

        let mut accept_stop = false;

        // Process queue (predict/complete) — only when there are non-pure-scan states
        if !queue.is_empty() {
            self.process_queue(
                &mut queue, &mut visited, &mut scanable, &mut returns,
                &mut accept_stop, &[],
            );
        }

        // Append to arenas
        let new_offset = self.state_arena.len();
        self.state_offsets.push(new_offset);
        self.state_arena.extend_from_slice(&scanable);
        let new_ret_offset = self.return_arena.len();
        self.return_offsets.push(new_ret_offset);
        self.return_arena.extend_from_slice(&returns);
        self.is_completed.push(accept_stop);

        // Return scratch buffers
        scanable.clear();
        returns.clear();
        queue.clear();
        visited.clear();
        self.buf_scanable = scanable;
        self.buf_return = returns;
        self.buf_queue = queue;
        self.buf_visited = visited;

        // Detect steady state: check if current level is identical to previous
        self.detect_and_enter_steady_state(ch);

        true
    }

    /// Advance through all bytes in a slice. On failure, rolls back any partial
    /// progress and returns false.
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

    /// Non-zero uniform delta steady advance: copy states+returns with delta adjustment.
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
                parent.return_level = (parent.return_level as i64
                    + self.steady.return_deltas[i] as i64)
                    as u32;
            }
            self.return_arena.push((expected, parent));
        }
        self.is_completed.push(self.steady.is_completed);
        // Keep chain cache up to date
        if let Some((rid, dfa, terminal, last)) = self.chain_terminal.get() {
            self.chain_terminal.set(Some((rid, dfa, terminal, last + 1)));
        }
    }

    /// Scan phase: advance each state by one byte via DFA.
    ///
    /// States whose successors have only CharRange edges go to `scanable` (pure scan
    /// optimization). All others go to `queue` for predict/complete processing.
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

    /// Process the queue: run predict/complete until fixed point.
    ///
    /// Tracks which rules have completed at the current level so that when a
    /// nullable rule is expanded again (deduped by `visited`), the parent can
    /// be advanced immediately. This handles the Earley "forward prediction"
    /// case where a new return entry is added after the rule already completed.
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
        // Track rule_ids that completed at current_level (for nullable dedup handling).
        // Inline array avoids heap allocation for the common case (0-8 completions).
        let mut completed_at_level = [0u16; 8];
        let mut completed_count = 0usize;

        let mut idx = 0;
        while idx < queue.len() {
            let state = queue[idx];
            idx += 1;

            let action = self.compiled.action(state.rule_id, state.dfa_state);

            // Predict: expand RuleRef edges (using pre-computed rule_refs)
            for &(rule_id, target) in &action.rule_refs {
                // Record return: when `rule` completes, parent advances to `target`
                let parent_after = StackState {
                    rule_id: state.rule_id,
                    dfa_state: target,
                    return_level: state.return_level,
                };
                returns.push((rule_id, parent_after));

                // Ancestor flattening: if parent_after is pass-through (accepting,
                // no char edges, no rule_ref), it will immediately complete when the
                // child finishes. Pre-register grandparents as also waiting for this
                // child rule, so complete() finds them directly without cascading
                // through the pass-through intermediate.
                let parent_action = self.compiled.action(parent_after.rule_id, parent_after.dfa_state);
                if parent_action.flags.is_pass_through()
                    && parent_after.return_level != NO_PARENT
                {
                    let level = parent_after.return_level as usize;
                    if level < self.return_offsets.len() {
                        let rstart = self.return_offsets[level];
                        let rend = self.return_offsets
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

                // Expand the referenced rule
                if !self.expand_rule(RuleId(rule_id as u32), current_level, queue, visited) {
                    // Rule already expanded at this level. If it has already
                    // completed (nullable), advance the parent immediately.
                    if completed_at_level[..completed_count].contains(&rule_id) {
                        if visited.insert(parent_after) {
                            queue.push(parent_after);
                        }
                    }
                }
            }

            // Complete: if accepting state, advance parents
            if action.flags.is_accepting() {
                // Track completion for nullable dedup handling
                if state.return_level == current_level {
                    if !completed_at_level[..completed_count].contains(&state.rule_id) {
                        if completed_count < completed_at_level.len() {
                            completed_at_level[completed_count] = state.rule_id;
                            completed_count += 1;
                        }
                    }
                }
                self.complete(
                    &state, queue, visited, returns, accept_stop, extra_returns,
                );
            }

            // If has char edges, it's scanable
            if action.flags.has_char_edges() {
                scanable.push(state);
            }
        }
    }

    /// Complete: a rule finished, advance parent states.
    fn complete(
        &self,
        state: &StackState,
        queue: &mut Vec<StackState>,
        visited: &mut SmallDedup<StackState>,
        returns: &mut Vec<(u16, StackState)>,
        accept_stop: &mut bool,
        extra_returns: &[(u16, StackState)],
    ) {
        if state.return_level == NO_PARENT {
            *accept_stop = true;
            return;
        }

        let start_pos = state.return_level as usize;
        let rule_id = state.rule_id;

        // From stored history
        if start_pos < self.return_offsets.len() {
            let rstart = self.return_offsets[start_pos];
            let rend = self.return_offsets
                .get(start_pos + 1)
                .copied()
                .unwrap_or(self.return_arena.len());
            for i in rstart..rend {
                let (expected_rule, parent_after) = self.return_arena[i];
                if expected_rule == rule_id {
                    // Detect self-referencing completion chain (only for grammars
                    // with explicit recursive rules, not Repeat-based JSON schemas).
                    if self.compiled.has_self_ref_chains
                        && parent_after.rule_id == state.rule_id
                        && parent_after.return_level != NO_PARENT
                        && (parent_after.return_level as usize) < start_pos
                        && self.compiled.action(parent_after.rule_id, parent_after.dfa_state)
                            .flags.is_pass_through()
                    {
                        self.follow_chain_to_terminal(
                            parent_after.rule_id,
                            parent_after.dfa_state,
                            parent_after.return_level as usize,
                            queue, visited, accept_stop,
                        );
                        continue;
                    }
                    if visited.insert(parent_after) {
                        queue.push(parent_after);
                    }
                }
            }
        }

        // From current (being built) returns + extra returns
        let current_level = self.state_offsets.len();
        if start_pos == current_level {
            for i in 0..returns.len() {
                let (expected_rule, parent_after) = returns[i];
                if expected_rule == rule_id {
                    if visited.insert(parent_after) {
                        queue.push(parent_after);
                    }
                }
            }
            for &(expected_rule, parent_after) in extra_returns {
                if expected_rule == rule_id {
                    if visited.insert(parent_after) {
                        queue.push(parent_after);
                    }
                }
            }
        }
    }

    /// Follow a self-referencing completion chain to its terminal level.
    ///
    /// When a rule like `[^"\\]*` completes, it creates a chain of ghost
    /// completions that cascade through all previous return levels:
    /// `(R,D,ret=K) → (R,D,ret=K-1) → ... → terminal`. Only the terminal
    /// produces a useful parent state (e.g., the root rule).
    ///
    /// This method follows the chain in a tight loop, using a cached terminal
    /// level for O(1) amortized cost.
    fn follow_chain_to_terminal(
        &self,
        chain_rule_id: u16,
        chain_dfa_state: u16,
        start_level: usize,
        queue: &mut Vec<StackState>,
        visited: &mut SmallDedup<StackState>,
        accept_stop: &mut bool,
    ) {
        // Check cache — only valid for consecutive advances (start == last + 1).
        // This ensures the chain is growing from the same context, not a new one.
        if let Some((cached_rid, cached_dfa, terminal, last_start)) = self.chain_terminal.get() {
            if cached_rid == chain_rule_id && cached_dfa == chain_dfa_state
                && start_level == last_start + 1
                && terminal < self.return_offsets.len()
            {
                // Use cached terminal — process its returns directly.
                // Non-chain parents at start_level are handled by complete()'s normal loop.
                self.process_terminal_returns(
                    chain_rule_id, terminal, queue, visited, accept_stop,
                );
                self.chain_terminal.set(Some((cached_rid, cached_dfa, terminal, start_level)));
                return;
            }
        }

        // Walk the chain to find the terminal level.
        // At each level, process ALL non-chain parents (they may go to the
        // parent rule, e.g., object rule from pair star). The chain link is
        // followed to the next level.
        let mut level = start_level;
        loop {
            if level >= self.return_offsets.len() {
                break;
            }
            let rstart = self.return_offsets[level];
            let rend = self.return_offsets
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
                self.chain_terminal.set(Some((chain_rule_id, chain_dfa_state, level, start_level)));
                break;
            }
        }
    }

    /// Process return entries at the terminal level of a completion chain.
    /// Also collects non-chain parents from intermediate levels between
    /// start_level and the terminal (they're all the same after dedup).
    fn process_terminal_returns(
        &self,
        chain_rule_id: u16,
        terminal_level: usize,
        queue: &mut Vec<StackState>,
        visited: &mut SmallDedup<StackState>,
        _accept_stop: &mut bool,
    ) {
        let rstart = self.return_offsets[terminal_level];
        let rend = self.return_offsets
            .get(terminal_level + 1)
            .copied()
            .unwrap_or(self.return_arena.len());

        for i in rstart..rend {
            let (expected, parent) = self.return_arena[i];
            if expected == chain_rule_id {
                // Terminal parent — add to queue for normal processing.
                if visited.insert(parent) {
                    queue.push(parent);
                }
            }
        }
    }

    /// Expand a rule: add its DFA start state to the queue.
    /// Returns true if the state was new (not deduped).
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

    /// Like `advance`, but doesn't commit to the arenas. Used for trie walk probing.
    /// Probe advance for trie walk: advance states by one byte without committing
    /// to arenas. Returns true if any states survived. Results are appended to
    /// `scanable_buf` and `returns_buf`.
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

        // Scan phase
        self.scan_states(current_states, ch, queue_buf, visited_buf, scanable_buf);

        if queue_buf.is_empty() && scanable_buf.is_empty() {
            return false;
        }

        let mut accept_stop = false;
        if !queue_buf.is_empty() {
            self.process_queue(
                queue_buf, visited_buf, scanable_buf, returns_buf,
                &mut accept_stop, extra_returns,
            );
        }

        true
    }

    /// Whether the grammar can terminate at the current position.
    pub(super) fn is_completed(&self) -> bool {
        if self.steady.count > 0 {
            return self.steady.is_completed;
        }
        self.is_completed.last().copied().unwrap_or(false)
    }

    /// Hash of the current parser state for bitmask caching.
    pub(super) fn state_hash(&self) -> u64 {
        let mut hasher = FxHasher::default();
        for state in self.current_states() {
            state.hash(&mut hasher);
        }
        for &(rid, ref state) in self.current_returns() {
            rid.hash(&mut hasher);
            state.hash(&mut hasher);
        }
        self.is_completed().hash(&mut hasher);
        hasher.finish()
    }

    /// Get the current scanable states.
    pub(super) fn current_states(&self) -> &[StackState] {
        if let Some(&start) = self.state_offsets.last() {
            &self.state_arena[start..]
        } else {
            &[]
        }
    }

    /// Get the current return entries.
    pub(super) fn current_returns(&self) -> &[(u16, StackState)] {
        if let Some(&start) = self.return_offsets.last() {
            &self.return_arena[start..]
        } else {
            &[]
        }
    }

    /// Current input position (number of bytes consumed).
    pub(super) fn position(&self) -> usize {
        self.state_offsets.len().saturating_sub(1) + self.steady.count
    }

    /// Pop the last `count` input positions (rollback).
    pub(super) fn pop_last_states(&mut self, count: usize) {
        if count == 0 {
            return;
        }
        // First consume lazy steady-state bytes
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

    /// Pop `count` committed levels from arenas.
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

    /// Reset to initial state.
    pub(super) fn reset(&mut self) {
        self.steady.reset();
        self.chain_terminal.set(None);
        self.init();
    }

    /// Commit one level for pending lazy steady-state bytes.
    /// Since deltas are zero, the states are identical to the last committed level.
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

    /// Detect steady state: compare current and previous levels structurally.
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

        // Structural comparison: same rule_id and dfa_state (return_level may differ)
        let structurally_same = prev_states.iter().zip(curr_states.iter()).all(|(a, b)| {
            a.rule_id == b.rule_id && a.dfa_state == b.dfa_state
        });
        if !structurally_same {
            return;
        }

        // Also check return entries structurally
        let prev_rstart = self.return_offsets[num - 2];
        let prev_rend = self.return_offsets[num - 1];
        let curr_rstart = self.return_offsets[num - 1];
        let prev_returns = &self.return_arena[prev_rstart..prev_rend];
        let curr_returns = &self.return_arena[curr_rstart..];

        if prev_returns.len() != curr_returns.len() {
            return;
        }
        let returns_same = prev_returns.iter().zip(curr_returns.iter()).all(|(a, b)| {
            a.0 == b.0
                && a.1.rule_id == b.1.rule_id
                && a.1.dfa_state == b.1.dfa_state
        });
        if !returns_same {
            return;
        }

        // Compute return_level deltas (state deltas)
        let state_deltas: Vec<i32> = prev_states.iter().zip(curr_states.iter())
            .map(|(p, c)| {
                if p.return_level == NO_PARENT { 0 }
                else { c.return_level as i32 - p.return_level as i32 }
            })
            .collect();

        // Safety: reject deltas outside {0, 1} to avoid recursive-grammar pitfalls
        if state_deltas.iter().any(|&d| d < 0 || d > 1) {
            return;
        }

        // Compute return_level deltas (return entry deltas)
        let return_deltas: Vec<i32> = prev_returns.iter().zip(curr_returns.iter())
            .map(|(p, c)| {
                if p.1.return_level == NO_PARENT { 0 }
                else { c.1.return_level as i32 - p.1.return_level as i32 }
            })
            .collect();

        if return_deltas.iter().any(|&d| d < 0 || d > 1) {
            return;
        }

        let all_zero = state_deltas.iter().all(|&d| d == 0)
            && return_deltas.iter().all(|&d| d == 0);

        // Extract steady ranges from DFA edges
        if let Some(ranges) = self.extract_steady_ranges(curr_states, ch) {
            self.steady.ranges = ranges;
            self.steady.is_completed = *self.is_completed.last().unwrap();
            self.steady.is_lazy = all_zero;
            self.steady.state_deltas = state_deltas;
            self.steady.return_deltas = return_deltas;
            self.steady.active = true;
        }
    }

    /// Extract accepted byte ranges for steady-state optimization.
    ///
    /// Uses the "same-target-as-ch" approach: for each state that directly
    /// advances on `ch`, collect all byte ranges leading to the same DFA target.
    /// States that don't have a DFA transition for `ch` are "reborn" states
    /// (re-added by process_queue each advance) and don't constrain the ranges.
    fn extract_steady_ranges(&self, states: &[StackState], ch: u8) -> Option<Vec<(u8, u8)>> {
        let mut winner: Option<Vec<(u8, u8)>> = None;
        let mut found_direct = false;

        for state in states {
            let dfa = &self.compiled.rule_dfas[state.rule_id as usize];

            // Check if this state directly advances on ch
            let ch_target = dfa.fsm.next_state(StateId(state.dfa_state as u32), ch);
            if ch_target.is_none() {
                // Reborn state — doesn't constrain steady ranges
                continue;
            }
            found_direct = true;
            let ch_target = ch_target.unwrap();

            // Collect byte ranges leading to the SAME target as ch
            let edges = dfa.fsm.edges(StateId(state.dfa_state as u32));
            let ranges: Vec<(u8, u8)> = edges.iter().filter_map(|e| {
                if let FsmEdge::CharRange { min, max, target } = e {
                    if *target == ch_target {
                        Some((*min, *max))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }).collect();

            // All direct-advance states must agree on the ranges
            match &winner {
                None => winner = Some(ranges),
                Some(w) if *w == ranges => {}
                _ => return None,
            }
        }

        if found_direct { winner } else { None }
    }
}

impl Clone for StackParser {
    fn clone(&self) -> Self {
        // Flush lazy state into the clone so it starts fresh
        let mut cloned = Self {
            compiled: Arc::clone(&self.compiled),
            state_arena: self.state_arena.clone(),
            state_offsets: self.state_offsets.clone(),
            return_arena: self.return_arena.clone(),
            return_offsets: self.return_offsets.clone(),
            is_completed: self.is_completed.clone(),
            buf_queue: Vec::new(),
            buf_visited: SmallDedup::new(),
            buf_scanable: Vec::new(),
            buf_return: Vec::new(),
            steady: SteadyState::new(),
            chain_terminal: Cell::new(None),
        };
        // If the original has lazy steady bytes, commit one level in the clone
        if self.steady.count > 0 {
            let prev_start = *cloned.state_offsets.last().unwrap();
            let new_start = cloned.state_arena.len();
            cloned.state_arena.extend_from_within(prev_start..);
            cloned.state_offsets.push(new_start);
            let prev_rstart = *cloned.return_offsets.last().unwrap();
            let new_rstart = cloned.return_arena.len();
            cloned.return_arena.extend_from_within(prev_rstart..);
            cloned.return_offsets.push(new_rstart);
            cloned.is_completed.push(self.steady.is_completed);
        }
        cloned
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::structured::compiled_grammar::CompiledGrammar;
    use crate::model::tokenizer::Tokenizer;

    #[test]
    #[ignore] // Run with: cargo test diagnostic_per_byte_stats -- --ignored --nocapture
    fn diagnostic_per_byte_stats() {
        let json_schema = r#"{
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string"},
                "status": {"type": "string", "enum": ["active", "inactive"]},
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "state": {"type": "string"},
                        "zip": {"type": "string"}
                    },
                    "required": ["street", "city", "state", "zip"],
                    "additionalProperties": false
                },
                "scores": {
                    "type": "array",
                    "items": {"type": "integer"}
                }
            },
            "required": ["id", "name", "status", "address", "scores"],
            "additionalProperties": false
        }"#;
        let grammar = Arc::new(crate::structured::json_schema::json_schema_to_grammar(json_schema, &crate::structured::json_schema::JsonSchemaOptions::default()).unwrap());
        let vocab: Vec<String> = (0..256u16).map(|b| String::from(b as u8 as char)).collect();
        let tokenizer = Arc::new(
            Tokenizer::from_vocab(&vocab),
        );
        let compiled = Arc::new(CompiledGrammar::new(&grammar, &tokenizer));
        let mut parser = StackParser::new(compiled.clone());

        let input = r#"{"id":42,"name":"Alice","status":"active","address":{"street":"123 Main St","city":"Springfield","state":"IL","zip":"62704"},"scores":[95,87,92]}"#;

        eprintln!("\n=== PER-BYTE DIAGNOSTIC: {} bytes ===", input.len());
        eprintln!("{:<5} {:<5} {:<8} {:<8} {:<6}",
            "pos", "char", "states", "returns", "steady");

        for (pos, &byte) in input.as_bytes().iter().enumerate() {
            let prev_states = parser.state_arena.len()
                - parser.state_offsets.last().copied().unwrap_or(0);
            let was_steady = parser.steady.active;
            assert!(parser.advance(byte), "failed at pos {} char '{}'", pos, byte as char);
            let new_states = parser.state_arena.len()
                - parser.state_offsets.last().copied().unwrap_or(0);
            let new_returns = parser.return_arena.len()
                - parser.return_offsets.last().copied().unwrap_or(0);
            let is_steady = parser.steady.active;
            let steady_str = if was_steady && is_steady { "lazy" }
                else if !was_steady && is_steady { "ENTER" }
                else if was_steady && !is_steady { "EXIT" }
                else { "-" };
            eprintln!("{:<5} {:<5} {:<8} {:<8} {:<6}",
                pos, format!("'{}'", byte as char),
                format!("{}→{}", prev_states, new_states),
                format!("→{}", new_returns), steady_str);
        }
        eprintln!("\n=== ARENA SIZES ===");
        eprintln!("state_arena: {} entries, return_arena: {} entries",
            parser.state_arena.len(), parser.return_arena.len());
        eprintln!("\n=== RULE DFA SIZES ===");
        for (i, dfa) in compiled.rule_dfas.iter().enumerate() {
            let name = &compiled.grammar.rules()[i].name;
            eprintln!("  rule {} ({}): {} DFA states", i, name, dfa.fsm.num_states());
        }
    }
}
