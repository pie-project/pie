//! Lexer construction and vocabulary grouping.
//!
//! This is the half of the design that makes a CPU-free decode step possible.
//! Token admissibility factors as
//!
//! ```text
//! allowed(token) = lexer_ok(lexer_state, token)
//!                  AND parser_ok(stack_top, terminals(token))
//! ```
//!
//! The left half is finite: lexer states are finite and the vocabulary is
//! fixed, so `(lexer_state, token)` resolves at compile time. The right half
//! depends only on the *terminal sequence* a token emits, so tokens emitting
//! the same sequence and landing in the same lexer state are indistinguishable
//! to the parser and share one entry.
//!
//! Measured on real vocabularies that collapse is large: 128k-262k tokens fall
//! into roughly 50-165 groups, and the count does not grow with vocabulary
//! size. Per-step parser work becomes one ACTION lookup per group.

pub mod lexicon;
pub mod regular;

use std::collections::{BTreeSet, HashMap, VecDeque};

use gpugrammar_ir::fsm::{Automaton, FsmEdge, NfaGraph, StateId};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

/// Identifies a terminal of the context-free skeleton.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TerminalId(pub u32);

/// A lexer DFA state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LexState(pub u32);

/// Where a scan begins when no lexeme is in progress.
pub const START: LexState = LexState(0);

const NO_STATE: u32 = u32::MAX;

/// A terminal's byte-level pattern, supplied as an automaton over bytes.
pub struct Terminal {
    pub name: String,
    pub automaton: Automaton<NfaGraph>,
}

/// A deterministic scanner over the union of the declared terminals.
///
/// An accepting state carries *every* terminal that ends there, not just the
/// first. Generated grammars are lexically ambiguous by construction — a
/// declared property name `"id"` is also a generic JSON string, and a colon is
/// also a character of a string body — so collapsing to one terminal loses the
/// one the parser wanted and rejects valid input. The scan therefore returns
/// candidate terminal sequences and the parser chooses among them, which is
/// the LR viable-prefix property doing the disambiguation.
pub struct Lexer {
    /// `transitions[state * 256 + byte]`, or `NO_STATE` when the byte is dead.
    transitions: Vec<u32>,
    accepts: Vec<Vec<TerminalId>>,
    terminal_names: Vec<String>,
}

impl Lexer {
    /// The raw transition table, `transitions[state * 256 + byte]`, with
    /// [`NO_STATE`] where a byte is impossible.
    ///
    /// Precomputed masks are one way to answer "is this token allowed"; walking
    /// the token's bytes on device is the other, and it needs this. The table
    /// is `states * 256 * 4` bytes and, unlike a mask, does not scale with the
    /// vocabulary at all.
    pub fn transitions(&self) -> &[u32] {
        &self.transitions
    }

    /// Sentinel for a byte with no transition.
    pub const NO_STATE: u32 = NO_STATE;

    pub fn num_states(&self) -> usize {
        self.accepts.len()
    }

    pub fn num_terminals(&self) -> usize {
        self.terminal_names.len()
    }

    pub fn terminal_name(&self, terminal: TerminalId) -> &str {
        &self.terminal_names[terminal.0 as usize]
    }

    pub fn accepting(&self, state: LexState) -> &[TerminalId] {
        &self.accepts[state.0 as usize]
    }

    /// Terminals still reachable from `state`, including one it may already
    /// accept.
    ///
    /// A token that ends mid-lexeme emits nothing, so the parser has nothing
    /// to check and would admit it unconditionally. That is how a finished
    /// document can be followed by the start of a second one. Knowing which
    /// terminals the pending lexeme could still become turns that into a real
    /// check.
    pub fn reachable_terminals(&self, state: LexState) -> Vec<TerminalId> {
        let mut seen = vec![false; self.num_states()];
        let mut found = BTreeSet::new();
        let mut queue = VecDeque::from([state]);
        seen[state.0 as usize] = true;
        while let Some(current) = queue.pop_front() {
            found.extend(self.accepting(current).iter().copied());
            let row = current.0 as usize * 256;
            for byte in 0..256usize {
                let next = self.transitions[row + byte];
                if next != NO_STATE && !seen[next as usize] {
                    seen[next as usize] = true;
                    queue.push_back(LexState(next));
                }
            }
        }
        found.into_iter().collect()
    }

    /// [`Self::reachable_terminals`] for every state at once.
    ///
    /// The artifact needs this per group, and a breadth-first search per group
    /// is quadratic in the number of lexer states, which is what made a large
    /// lexer take minutes to emit. One fixpoint over the reverse graph gives
    /// them all: a state reaches what it accepts plus whatever its successors
    /// reach.
    pub fn reachable_terminals_all(&self) -> Vec<Vec<TerminalId>> {
        let states = self.num_states();
        let words = self.num_terminals().div_ceil(64).max(1);
        let mut reach = vec![0u64; states * words];
        for (state, accepting) in self.accepts.iter().enumerate() {
            for terminal in accepting {
                reach[state * words + terminal.0 as usize / 64] |= 1u64 << (terminal.0 % 64);
            }
        }

        let mut predecessors: Vec<Vec<u32>> = vec![Vec::new(); states];
        for state in 0..states {
            let row = state * 256;
            let mut seen: Vec<u32> = self.transitions[row..row + 256]
                .iter()
                .copied()
                .filter(|next| *next != NO_STATE)
                .collect();
            seen.sort_unstable();
            seen.dedup();
            for next in seen {
                predecessors[next as usize].push(state as u32);
            }
        }

        let mut queue: VecDeque<u32> = (0..states as u32).collect();
        let mut queued = vec![true; states];
        while let Some(state) = queue.pop_front() {
            queued[state as usize] = false;
            for index in 0..predecessors[state as usize].len() {
                let predecessor = predecessors[state as usize][index] as usize;
                let mut changed = false;
                for word in 0..words {
                    let incoming = reach[state as usize * words + word];
                    let slot = &mut reach[predecessor * words + word];
                    if *slot | incoming != *slot {
                        *slot |= incoming;
                        changed = true;
                    }
                }
                if changed && !queued[predecessor] {
                    queued[predecessor] = true;
                    queue.push_back(predecessor as u32);
                }
            }
        }

        (0..states)
            .map(|state| {
                (0..self.num_terminals())
                    .filter(|terminal| {
                        reach[state * words + terminal / 64] & (1u64 << (terminal % 64)) != 0
                    })
                    .map(|terminal| TerminalId(terminal as u32))
                    .collect()
            })
            .collect()
    }

    /// Cut transitions into states that can no longer accept anything.
    ///
    /// Subset construction leaves states from which no accepting state is
    /// reachable. They are harmless for recognition, which reads to the end and
    /// asks, but not for constrained decoding, which asks after every byte: a
    /// live transition into a dead state says a token is allowed when nothing
    /// can complete it. With a multi-terminal grammar the parser hides most of
    /// this, but 68% of real schemas are purely regular, and there the lexer is
    /// the only check there is.
    fn trim(&mut self) {
        let reachable = self.reachable_terminals_all();
        let dead: Vec<bool> = reachable
            .iter()
            .map(|terminals| terminals.is_empty())
            .collect();
        for next in &mut self.transitions {
            if *next != NO_STATE && dead[*next as usize] {
                *next = NO_STATE;
            }
        }
    }

    /// Merge states that behave identically from here on.
    ///
    /// Subset construction produces the reachable states, not the fewest. Two
    /// states are the same for our purposes when they accept the same terminals
    /// and every byte takes them to states that are themselves the same, and
    /// merging them costs nothing: the mask a state implies is a function of
    /// exactly that behaviour. It matters more here than in a normal scanner
    /// because a lexer state carries a token bitset per group - measured at
    /// 19 KiB with a 151,669-token vocabulary - so states are the unit of
    /// memory, not just of table size.
    ///
    /// Moore's algorithm: start by separating states with different accepting
    /// sets, then repeatedly split any block whose members disagree about which
    /// block a byte leads to.
    fn minimise(&mut self) {
        let states = self.num_states();
        if states <= 1 {
            return;
        }

        let mut block: Vec<u32> = {
            let mut by_accepting: FxHashMap<&[TerminalId], u32> = FxHashMap::default();
            self.accepts
                .iter()
                .map(|accepting| {
                    let next = by_accepting.len() as u32;
                    *by_accepting.entry(accepting.as_slice()).or_insert(next)
                })
                .collect()
        };

        let mut signature: Vec<u32> = vec![0; 256];
        loop {
            let mut refined: FxHashMap<Vec<u32>, u32> = FxHashMap::default();
            let mut next_block = vec![0u32; states];
            for state in 0..states {
                let row = state * 256;
                for byte in 0..256 {
                    let target = self.transitions[row + byte];
                    signature[byte] = if target == NO_STATE {
                        u32::MAX
                    } else {
                        block[target as usize]
                    };
                }
                let mut key = Vec::with_capacity(257);
                key.push(block[state]);
                key.extend_from_slice(&signature);
                let size = refined.len() as u32;
                next_block[state] = *refined.entry(key).or_insert(size);
            }
            let settled = refined.len() == block.iter().collect::<FxHashSet<_>>().len();
            block = next_block;
            if settled {
                break;
            }
        }

        // The start state has to keep index zero, so renumber from it.
        let blocks = block.iter().copied().max().map_or(0, |m| m as usize + 1);
        if blocks == states {
            return;
        }
        let mut representative: Vec<Option<u32>> = vec![None; blocks];
        let mut order: Vec<u32> = Vec::with_capacity(blocks);
        for state in 0..states {
            let id = block[state] as usize;
            if representative[id].is_none() {
                representative[id] = Some(order.len() as u32);
                order.push(state as u32);
            }
        }
        let renumber =
            |state: u32| -> u32 { representative[block[state as usize] as usize].unwrap() };

        let mut transitions = vec![NO_STATE; order.len() * 256];
        let mut accepts = Vec::with_capacity(order.len());
        for (index, &state) in order.iter().enumerate() {
            let row = state as usize * 256;
            for byte in 0..256 {
                let target = self.transitions[row + byte];
                transitions[index * 256 + byte] = if target == NO_STATE {
                    NO_STATE
                } else {
                    renumber(target)
                };
            }
            accepts.push(self.accepts[state as usize].clone());
        }
        self.transitions = transitions;
        self.accepts = accepts;
    }

    fn step(&self, state: LexState, byte: u8) -> Option<LexState> {
        let next = self.transitions[state.0 as usize * 256 + byte as usize];
        (next != NO_STATE).then_some(LexState(next))
    }

    /// True when no byte can extend this state, so a lexeme ending here is
    /// final and cannot be continued by the next token.
    fn is_dead_end(&self, state: LexState) -> bool {
        let row = state.0 as usize * 256;
        self.transitions[row..row + 256]
            .iter()
            .all(|&next| next == NO_STATE)
    }

    /// Every way this token can be read, longest match first.
    ///
    /// `None` means the token is lexically impossible here, which is what
    /// removes most of the vocabulary at a structural position.
    ///
    /// Maximal munch alone is wrong for a generated lexicon. A whole regular
    /// rule becomes one terminal, so `{` is both a complete terminal and the
    /// start of a longer one covering an entire object; a greedy scanner
    /// consumes the next byte and the `{` the parser needed can no longer be
    /// emitted. Where a lexeme ends is therefore as much a choice as which
    /// terminal it is, and both are left to the parser, which resolves them
    /// with its viable prefixes. Longest match is offered first, so it wins
    /// whenever the parser can follow it.
    pub fn scan(&self, token: &[u8], state: LexState) -> Option<Scan> {
        let mut options = Vec::new();
        let mut emitted = Vec::new();
        let mut budget = MAX_STEPS;
        self.readings(token, 0, state, &mut emitted, &mut options, &mut budget);
        (!options.is_empty()).then_some(Scan { options })
    }

    fn readings(
        &self,
        token: &[u8],
        index: usize,
        state: LexState,
        emitted: &mut Vec<TerminalId>,
        out: &mut Vec<ScanOption>,
        budget: &mut usize,
    ) {
        // A branch that settles a lexeme and finds nothing after it produces no
        // option, so capping the output does not cap the search: a state that
        // accepts several terminals branches at every byte and most branches
        // die. One schema spent minutes here on a 49-state lexer. Losing a
        // reading can only narrow a mask, never widen one, which is what makes
        // a budget the safe way to bound it.
        if *budget == 0 {
            return;
        }
        *budget -= 1;
        if out.len() >= MAX_OPTIONS {
            return;
        }
        if index == token.len() {
            // Carry the lexeme into the next token, or settle it here.
            out.push(ScanOption {
                terminals: emitted.clone(),
                next_state: state,
            });
            for terminal in self.accepting(state) {
                emitted.push(*terminal);
                out.push(ScanOption {
                    terminals: emitted.clone(),
                    next_state: START,
                });
                emitted.pop();
            }
            return;
        }

        if let Some(next) = self.step(state, token[index]) {
            self.readings(token, index + 1, next, emitted, out, budget);
        }
        // Settling restarts the scan at the same byte. It cannot loop, because
        // the start state accepts nothing: nullable terminals were removed, so
        // no lexeme is empty.
        for terminal in self.accepting(state) {
            emitted.push(*terminal);
            self.readings(token, index, START, emitted, out, budget);
            emitted.pop();
        }
    }
}

/// One way of reading a token: what it emits and where it leaves the lexer.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ScanOption {
    pub terminals: Vec<TerminalId>,
    pub next_state: LexState,
}

/// The result of scanning one token.
///
/// `options` holds the readings, longest match first. There is always at least
/// one; a reading is empty when the token ends mid-lexeme.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Scan {
    pub options: Vec<ScanOption>,
}

/// How many readings one token may carry. Real grammars stay at one or two;
/// the cap only stops a pathological fan-out.
const MAX_OPTIONS: usize = 16;

/// How many nodes one token's reading search may visit.
///
/// Generous against what a real token needs - a settle point per byte over a
/// long token is already past it - and small enough that a pathological lexer
/// costs microseconds rather than minutes.
const MAX_STEPS: usize = 4096;

/// Build a scanner by determinising the union of the terminal automata.
///
/// `Automaton::to_dfa` is not reused because it discards which terminal
/// accepted in each subset, and that tag is the whole point here.
pub fn build_lexer(terminals: Vec<Terminal>) -> Lexer {
    build_lexer_within(terminals, usize::MAX).expect("no state budget was set")
}

/// As [`build_lexer`], but abandoned once the DFA exceeds `budget` states.
///
/// A length bound has to be unrolled to be held in a DFA: `"maxLength": 2048`
/// over UTF-8 costs roughly seventy states per counted position, so a single
/// schema can ask for hundreds of thousands. Those are correct but too large
/// to emit, and the caller needs to find that out cheaply rather than after
/// determinising them.
///
/// The budget bounds work, not just the result. Subset construction costs
/// `dfa_states * 256 * subset_size`, and a large unrolled automaton makes the
/// subsets large, so bounding the state count alone still leaves a
/// determinisation that runs for minutes.
pub fn build_lexer_within(terminals: Vec<Terminal>, budget: usize) -> Option<Lexer> {
    let mut union = NfaGraph::new();
    let start = union.add_state();
    let mut ends: Vec<(StateId, TerminalId)> = Vec::new();
    let mut names = Vec::new();

    for (index, terminal) in terminals.iter().enumerate() {
        let offset = union.num_states() as u32;
        let id = TerminalId(index as u32);
        names.push(terminal.name.clone());
        for _ in 0..terminal.automaton.fsm.num_states() {
            union.add_state();
        }
        for state in 0..terminal.automaton.fsm.num_states() {
            let from = StateId(offset + state as u32);
            for edge in terminal.automaton.fsm.edges(StateId(state as u32)) {
                let shifted = match edge {
                    FsmEdge::CharRange { min, max, target } => FsmEdge::CharRange {
                        min: *min,
                        max: *max,
                        target: StateId(offset + target.0),
                    },
                    FsmEdge::Epsilon(target) => FsmEdge::Epsilon(StateId(offset + target.0)),
                    FsmEdge::RuleRef { rule, target } => FsmEdge::RuleRef {
                        rule: *rule,
                        target: StateId(offset + target.0),
                    },
                };
                union.add_edge(from, shifted);
            }
            if terminal.automaton.is_end(StateId(state as u32)) {
                ends.push((from, id));
            }
        }
        union.add_epsilon(start, StateId(offset + terminal.automaton.start.0));
    }

    let end_of: HashMap<StateId, TerminalId> = ends.into_iter().collect();
    let mut lexer = determinise(&union, start, &end_of, names, budget)?;
    lexer.trim();
    lexer.minimise();
    Some(lexer)
}

fn determinise(
    nfa: &NfaGraph,
    start: StateId,
    end_of: &HashMap<StateId, TerminalId>,
    terminal_names: Vec<String>,
    budget: usize,
) -> Option<Lexer> {
    let mut subsets: FxHashMap<BTreeSet<StateId>, u32> = FxHashMap::default();
    let mut order: Vec<BTreeSet<StateId>> = Vec::new();
    let mut queue: VecDeque<BTreeSet<StateId>> = VecDeque::new();

    let initial = nfa.epsilon_closure(&BTreeSet::from([start]));
    subsets.insert(initial.clone(), 0);
    order.push(initial.clone());
    queue.push_back(initial);

    let mut transitions: Vec<u32> = Vec::new();
    let mut work = 0usize;
    let work_budget = budget.saturating_mul(50_000);
    while let Some(subset) = queue.pop_front() {
        let row = transitions.len();
        transitions.resize(row + 256, NO_STATE);
        work = work.saturating_add(subset.len().saturating_mul(256));
        if work > work_budget {
            return None;
        }
        for byte in 0..=255u8 {
            let mut targets = BTreeSet::new();
            for &state in &subset {
                for edge in nfa.edges(state) {
                    if let FsmEdge::CharRange { min, max, target } = edge {
                        if byte >= *min && byte <= *max {
                            targets.insert(*target);
                        }
                    }
                }
            }
            if targets.is_empty() {
                continue;
            }
            let closure = nfa.epsilon_closure(&targets);
            let id = match subsets.get(&closure) {
                Some(&existing) => existing,
                None => {
                    if order.len() >= budget {
                        return None;
                    }
                    let fresh = order.len() as u32;
                    subsets.insert(closure.clone(), fresh);
                    order.push(closure.clone());
                    queue.push_back(closure);
                    fresh
                }
            };
            transitions[row + byte as usize] = id;
        }
    }

    let accepts: Vec<Vec<TerminalId>> = order
        .iter()
        .map(|subset| {
            let mut terminals: Vec<TerminalId> = subset
                .iter()
                .filter_map(|state| end_of.get(state).copied())
                .collect();
            terminals.sort_unstable();
            terminals.dedup();
            terminals
        })
        .collect();

    Some(Lexer {
        transitions,
        accepts,
        terminal_names,
    })
}

/// Tokens that are indistinguishable to the parser from one lexer state.
#[derive(Debug, Clone)]
pub struct Group {
    pub scan: Scan,
    pub tokens: Vec<u32>,
}

/// The grouping of a vocabulary, one bucket per lexer state.
#[derive(Debug, Clone)]
pub struct VocabularyGroups {
    pub per_state: Vec<Vec<Group>>,
    pub rejected: Vec<u32>,
}

impl VocabularyGroups {
    pub fn total_groups(&self) -> usize {
        self.per_state.iter().map(Vec::len).sum()
    }
}

/// Group every token of `vocabulary`, from one lexer state.
///
/// Grouping a single state is what a lazy cache needs: a mask is a pure
/// function of the state, and a real document reaches 2-44% of the states its
/// grammar can, so grouping all of them up front pays for work no request will
/// use.
pub fn group_state(lexer: &Lexer, vocabulary: &[Vec<u8>], state: LexState) -> (Vec<Group>, u32) {
    let mut rejected = 0u32;
    let mut buckets: FxHashMap<Scan, Vec<u32>> = FxHashMap::default();
    for (token_id, bytes) in vocabulary.iter().enumerate() {
        if bytes.is_empty() {
            rejected += 1;
            continue;
        }
        match lexer.scan(bytes, state) {
            Some(scan) => buckets.entry(scan).or_default().push(token_id as u32),
            None => rejected += 1,
        }
    }
    let mut groups: Vec<Group> = buckets
        .into_iter()
        .map(|(scan, tokens)| Group { scan, tokens })
        .collect();
    groups.sort_by(|a, b| b.tokens.len().cmp(&a.tokens.len()));
    (groups, rejected)
}

/// Group every token of `vocabulary`, from every lexer state.
pub fn group_vocabulary(lexer: &Lexer, vocabulary: &[Vec<u8>]) -> VocabularyGroups {
    // Every lexer state scans the whole vocabulary, and no state's answer
    // depends on another's, so this is the one stage of compilation that is
    // embarrassingly parallel - and the one that dominates it. It is also
    // where residency is paid for: this is precisely the work a host-side
    // matcher repeats at every decode step instead of doing once.
    let (per_state, rejected): (Vec<Vec<Group>>, Vec<u32>) = (0..lexer.num_states())
        .into_par_iter()
        .map(|state| {
            let from = LexState(state as u32);
            let mut refused = 0u32;
            let mut buckets: FxHashMap<Scan, Vec<u32>> = FxHashMap::default();
            for (token_id, bytes) in vocabulary.iter().enumerate() {
                if bytes.is_empty() {
                    refused += 1;
                    continue;
                }
                match lexer.scan(bytes, from) {
                    Some(scan) => buckets.entry(scan).or_default().push(token_id as u32),
                    None => refused += 1,
                }
            }
            let mut groups: Vec<Group> = buckets
                .into_iter()
                .map(|(scan, tokens)| Group { scan, tokens })
                .collect();
            groups.sort_by(|a, b| b.tokens.len().cmp(&a.tokens.len()));
            (groups, refused)
        })
        .unzip();

    VocabularyGroups {
        per_state,
        rejected,
    }
}
