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
/// Accepting states carry the terminal that wins there; ties break by
/// declaration order, so a caller can put keywords ahead of identifiers.
pub struct Lexer {
    /// `transitions[state * 256 + byte]`, or `NO_STATE` when the byte is dead.
    transitions: Vec<u32>,
    accepts: Vec<Option<TerminalId>>,
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

    pub fn accepting(&self, state: LexState) -> Option<TerminalId> {
        self.accepts[state.0 as usize]
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
        let mut terminals = Vec::new();
        let mut index = 0usize;

        while index < token.len() {
            let mut cursor = current;
            let mut position = index;
            let mut last_accept: Option<(usize, TerminalId)> = None;

            while position < token.len() {
                let Some(next) = self.step(cursor, token[position]) else {
                    break;
                };
                cursor = next;
                position += 1;
                if let Some(terminal) = self.accepting(cursor) {
                    last_accept = Some((position, terminal));
                }
            }

            if position == index {
                // Not a single byte could be consumed from here.
                return None;
            }

            if position == token.len() {
                return match last_accept {
                    Some((end, terminal)) if end == position && self.is_dead_end(cursor) => {
                        terminals.push(terminal);
                        Some(Scan {
                            terminals,
                            next_state: START,
                        })
                    }
                    _ => Some(Scan {
                        terminals,
                        next_state: cursor,
                    }),
                };
            }

            // A byte blocked the lexeme, so commit the longest match.
            let (end, terminal) = last_accept?;
            terminals.push(terminal);
            index = end;
            current = START;
        }

        Some(Scan {
            terminals,
            next_state: current,
        })
    }
}

/// The result of scanning one token.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Scan {
    pub terminals: Vec<TerminalId>,
    pub next_state: LexState,
}

/// Build a scanner by determinising the union of the terminal automata.
///
/// `Automaton::to_dfa` is not reused because it discards which terminal
/// accepted in each subset, and that tag is the whole point here.
pub fn build_lexer(terminals: Vec<Terminal>) -> Lexer {
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
    determinise(&union, start, &end_of, names)
}

fn determinise(
    nfa: &NfaGraph,
    start: StateId,
    end_of: &HashMap<StateId, TerminalId>,
    terminal_names: Vec<String>,
) -> Lexer {
    let mut subsets: FxHashMap<BTreeSet<StateId>, u32> = FxHashMap::default();
    let mut order: Vec<BTreeSet<StateId>> = Vec::new();
    let mut queue: VecDeque<BTreeSet<StateId>> = VecDeque::new();

    let initial = nfa.epsilon_closure(&BTreeSet::from([start]));
    subsets.insert(initial.clone(), 0);
    order.push(initial.clone());
    queue.push_back(initial);

    let mut transitions: Vec<u32> = Vec::new();
    while let Some(subset) = queue.pop_front() {
        let row = transitions.len();
        transitions.resize(row + 256, NO_STATE);
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

    let accepts = order
        .iter()
        .map(|subset| {
            subset
                .iter()
                .filter_map(|state| end_of.get(state).copied())
                .min()
        })
        .collect();

    Lexer {
        transitions,
        accepts,
        terminal_names,
    }
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
