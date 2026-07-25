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
use rustc_hash::FxHashMap;

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

    /// Scan one token with maximal munch, starting from `state`.
    ///
    /// `None` means the token is lexically impossible here, which is the case
    /// that removes most of the vocabulary at a structural position.
    ///
    /// A lexeme is only emitted once it cannot be extended. When a token ends
    /// on an accepting state that a further byte could still extend — a digit
    /// run, say — the terminal is withheld and the state is carried, because
    /// the next token may continue the same lexeme.
    pub fn scan(&self, token: &[u8], state: LexState) -> Option<Scan> {
        let mut current = state;
        let mut committed: Vec<&[TerminalId]> = Vec::new();
        let mut index = 0usize;
        let mut rounds = 0usize;

        while index < token.len() {
            rounds += 1;
            if rounds > token.len() * 2 + 4 {
                return None;
            }

            let mut cursor = current;
            let mut position = index;
            // A lexeme finished by an earlier token is still pending in
            // `current`; a byte that cannot extend it commits it rather than
            // failing the scan.
            let mut last_accept = match self.accepting(cursor) {
                [] => None,
                accepting => Some((index, accepting)),
            };

            while position < token.len() {
                let Some(next) = self.step(cursor, token[position]) else {
                    break;
                };
                cursor = next;
                position += 1;
                match self.accepting(cursor) {
                    [] => {}
                    accepting => last_accept = Some((position, accepting)),
                }
            }

            if position == token.len() {
                return match last_accept {
                    Some((end, accepting)) if end == position && self.is_dead_end(cursor) => {
                        committed.push(accepting);
                        Some(Scan {
                            choices: product(&committed),
                            next_state: START,
                        })
                    }
                    _ => Some(Scan {
                        choices: product(&committed),
                        next_state: cursor,
                    }),
                };
            }

            // A byte blocked the lexeme, so commit the longest match.
            let (end, accepting) = last_accept?;
            committed.push(accepting);
            index = end;
            current = START;
        }

        Some(Scan {
            choices: product(&committed),
            next_state: current,
        })
    }
}

/// The result of scanning one token.
///
/// `choices` holds the terminal sequences the token could emit, one per way of
/// resolving the lexical ambiguities it crossed. There is always at least one,
/// and it is empty when the token ends mid-lexeme. All choices share
/// `next_state`, because maximal munch fixes where the lexemes end; only which
/// terminal each one is differs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Scan {
    pub choices: Vec<Vec<TerminalId>>,
    pub next_state: LexState,
}

/// How many disambiguations one token may carry. Real grammars stay at one or
/// two; the cap only stops a pathological product.
const MAX_CHOICES: usize = 16;

/// Expand per-lexeme accepting sets into whole-token terminal sequences.
fn product(committed: &[&[TerminalId]]) -> Vec<Vec<TerminalId>> {
    let mut choices: Vec<Vec<TerminalId>> = vec![Vec::new()];
    for accepting in committed {
        let mut next = Vec::new();
        for prefix in &choices {
            for terminal in accepting.iter() {
                if next.len() == MAX_CHOICES {
                    break;
                }
                let mut extended = prefix.clone();
                extended.push(*terminal);
                next.push(extended);
            }
        }
        choices = next;
    }
    choices
}

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
    determinise(&union, start, &end_of, names, budget)
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

/// Group every token of `vocabulary`, from every lexer state.
pub fn group_vocabulary(lexer: &Lexer, vocabulary: &[Vec<u8>]) -> VocabularyGroups {
    let mut per_state = Vec::with_capacity(lexer.num_states());
    let mut rejected = vec![0u32; lexer.num_states()];

    for state in 0..lexer.num_states() {
        let from = LexState(state as u32);
        let mut buckets: FxHashMap<Scan, Vec<u32>> = FxHashMap::default();
        for (token_id, bytes) in vocabulary.iter().enumerate() {
            if bytes.is_empty() {
                rejected[state] += 1;
                continue;
            }
            match lexer.scan(bytes, from) {
                Some(scan) => buckets.entry(scan).or_default().push(token_id as u32),
                None => rejected[state] += 1,
            }
        }
        let mut groups: Vec<Group> = buckets
            .into_iter()
            .map(|(scan, tokens)| Group { scan, tokens })
            .collect();
        groups.sort_by(|a, b| b.tokens.len().cmp(&a.tokens.len()));
        per_state.push(groups);
    }

    VocabularyGroups {
        per_state,
        rejected,
    }
}
