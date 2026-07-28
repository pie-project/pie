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
        let mut scratch = ScanScratch::default();
        self.scan_into(token, state, &mut scratch)
            .then(|| scratch.take())
    }

    /// Scan into buffers the caller owns, and say whether anything survived.
    ///
    /// Grouping runs this once per (lexer state, token) - tens of millions of
    /// times for one schema, and 62% of all the time compilation takes. A
    /// `Scan` is a vector of readings each holding a vector of terminals, so
    /// producing one allocated three or four times and hashing it walked all
    /// of them. Here the readings go into two flat buffers the caller keeps,
    /// and the only allocation left is for a reading nobody has seen before.
    pub fn scan_into(&self, token: &[u8], state: LexState, scratch: &mut ScanScratch) -> bool {
        self.scan_into_with(token, state, scratch, None)
    }

    /// As `scan_into`, splicing in a token's readings from the start state
    /// rather than walking them again.
    pub fn scan_into_with(
        &self,
        token: &[u8],
        state: LexState,
        scratch: &mut ScanScratch,
        memo: Option<(&StartScans, usize)>,
    ) -> bool {
        self.readings_from(token, 0, state, scratch, memo);
        !scratch.ends.is_empty()
    }

    /// The readings of `token[index..]` from `state`, into the caller's
    /// buffers. `scan_into_with` is this at index zero.
    fn readings_from(
        &self,
        token: &[u8],
        index: usize,
        state: LexState,
        scratch: &mut ScanScratch,
        memo: Option<(&StartScans, usize)>,
    ) {
        scratch.clear();
        // Nothing to do if the first byte cannot be taken and nothing can be
        // settled here, which is most of the vocabulary at a structural
        // position. One transition lookup rather than a call and a recursion.
        if let Some(&first) = token.get(index)
            && self.step(state, first).is_none()
            && self.accepting(state).is_empty()
        {
            return;
        }
        let mut budget = MAX_STEPS;
        let mut emitted = std::mem::take(&mut scratch.emitted);
        self.readings(token, index, state, &mut emitted, scratch, &mut budget, memo);
        scratch.emitted = emitted;
    }

    fn readings(
        &self,
        token: &[u8],
        index: usize,
        state: LexState,
        emitted: &mut Vec<TerminalId>,
        out: &mut ScanScratch,
        budget: &mut usize,
        memo: Option<(&StartScans, usize)>,
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
        if out.ends.len() >= MAX_OPTIONS {
            return;
        }
        if index == token.len() {
            // Carry the lexeme into the next token, or settle it here.
            out.push(emitted, state);
            for terminal in self.accepting(state) {
                emitted.push(*terminal);
                out.push(emitted, START);
                emitted.pop();
            }
            return;
        }

        if let Some(next) = self.step(state, token[index]) {
            self.readings(token, index + 1, next, emitted, out, budget, memo);
        }
        // Settling restarts the scan at the same byte. It cannot loop, because
        // the start state accepts nothing: nullable terminals were removed, so
        // no lexeme is empty.
        for terminal in self.accepting(state) {
            emitted.push(*terminal);
            // At the first byte the restart is the whole token from the start
            // state, which is the same walk for every state that can settle.
            // Spliced from the memo when there is one.
            match memo {
                // Only at the first byte: memoising every suffix cut the walk
                // from 22 nodes a token to 2.5 and was *slower*, which is how
                // it is known that the walk was never the cost.
                Some((held, token_id)) if index == 0 => {
                    for (terminals, next_state) in held.readings_of(token_id) {
                        if out.ends.len() >= MAX_OPTIONS {
                            break;
                        }
                        let before = emitted.len();
                        emitted.extend_from_slice(terminals);
                        out.push(emitted, next_state);
                        emitted.truncate(before);
                    }
                }
                _ => self.readings(token, index, START, emitted, out, budget, memo),
            }
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

/// The vocabulary as one buffer, so scanning it does not chase pointers.
///
/// Grouping walks every token from every lexer state, and a `&[Vec<u8>]` is a
/// hundred and fifty thousand separate allocations - one cache miss per token
/// per state, before any lexer work happens at all. Flattened once it is under
/// a megabyte and stays in cache for every state that follows.
struct FlatVocabulary {
    bytes: Vec<u8>,
    ends: Vec<u32>,
}

impl FlatVocabulary {
    fn of(vocabulary: &[Vec<u8>]) -> Self {
        let mut bytes = Vec::with_capacity(vocabulary.iter().map(Vec::len).sum());
        let mut ends = Vec::with_capacity(vocabulary.len());
        for token in vocabulary {
            bytes.extend_from_slice(token);
            ends.push(bytes.len() as u32);
        }
        FlatVocabulary { bytes, ends }
    }

    fn len(&self) -> usize {
        self.ends.len()
    }

    fn get(&self, index: usize) -> &[u8] {
        let from = if index == 0 { 0 } else { self.ends[index - 1] } as usize;
        &self.bytes[from..self.ends[index] as usize]
    }
}

/// Every token read from the start state, computed once.
///
/// A state that can settle a lexeme restarts the scan at the same byte from
/// the start state, and what comes of that depends only on the bytes left -
/// not on where the restart came from. Settles nest, so the walk of one token
/// is a tree rather than a path: measured on a real lexicon it is 22 nodes for
/// a token of about five bytes, and grouping repeats it for every state.
///
/// Built once for the whole vocabulary, back to front so that a suffix can use
/// the suffixes inside it. That costs about `len` steps per entry and turns
/// each of the tens of millions of scans that follow into a forward walk with
/// splices.
#[derive(Debug, Default)]
pub struct StartScans {
    /// Where each token's readings begin and end in `ends`.
    span: Vec<(u32, u32)>,
    /// Each reading: where its terminals end in `flat`, and the state it
    /// leaves the lexer in.
    ends: Vec<(u32, LexState)>,
    flat: Vec<TerminalId>,
}

impl StartScans {
    fn build(lexer: &Lexer, vocabulary: &FlatVocabulary) -> Self {
        let mut held = StartScans::default();
        let mut scratch = ScanScratch::default();
        for index in 0..vocabulary.len() {
            let bytes = vocabulary.get(index);
            let from = held.ends.len() as u32;
            if !bytes.is_empty() {
                // No memo of its own: the start state accepts nothing, so this
                // cannot recurse into what it is building.
                lexer.readings_from(bytes, 0, START, &mut scratch, None);
                let mut previous = 0usize;
                for (end, next_state) in &scratch.ends {
                    let to = *end as usize;
                    held.flat.extend_from_slice(&scratch.flat[previous..to]);
                    held.ends.push((held.flat.len() as u32, *next_state));
                    previous = to;
                }
            }
            held.span.push((from, held.ends.len() as u32));
        }
        held
    }

    fn readings_of(&self, token_id: usize) -> impl Iterator<Item = (&[TerminalId], LexState)> {
        let (from, to) = self.span[token_id];
        // `flat` is one run after another over the whole vocabulary, so a
        // reading's terminals start where the previous one ended.
        (from..to).map(move |at| {
            let start = if at == 0 { 0 } else { self.ends[at as usize - 1].0 };
            let (end, state) = self.ends[at as usize];
            (&self.flat[start as usize..end as usize], state)
        })
    }
}

/// The readings of one token, flat, in buffers a caller reuses.
///
/// A `Scan` is a vector of vectors: building one per token allocated three or
/// four times and hashing it walked all of them, which for tens of millions of
/// scans is where compilation went. Here a reading is a run of `flat` and an
/// entry in `ends`, both reused, and the bytes of the two are the key a bucket
/// is found by - so a token that reads like one already seen allocates nothing
/// at all.
#[derive(Debug, Default)]
pub struct ScanScratch {
    /// Every reading's terminals, one run after another.
    pub flat: Vec<TerminalId>,
    /// Where each reading's run ends, and the state it leaves the lexer in.
    pub ends: Vec<(u32, LexState)>,
    /// The terminals of the reading being built, during the walk.
    pub emitted: Vec<TerminalId>,
    /// The two above as bytes, for looking a bucket up without allocating.
    pub key: Vec<u8>,
}

impl ScanScratch {
    fn clear(&mut self) {
        self.flat.clear();
        self.ends.clear();
        self.emitted.clear();
        self.key.clear();
    }

    fn push(&mut self, emitted: &[TerminalId], next_state: LexState) {
        self.flat.extend_from_slice(emitted);
        self.ends.push((self.flat.len() as u32, next_state));
    }

    /// The readings as bytes. Two tokens read the same way exactly when these
    /// agree, so it is both the hash key and the equality test.
    pub fn key(&mut self) -> &[u8] {
        self.key.clear();
        for (end, state) in &self.ends {
            self.key.extend_from_slice(&end.to_le_bytes());
            self.key.extend_from_slice(&state.0.to_le_bytes());
        }
        for terminal in &self.flat {
            self.key.extend_from_slice(&terminal.0.to_le_bytes());
        }
        &self.key
    }

    /// The readings as the public type, built only for a bucket that is new.
    pub fn take(&self) -> Scan {
        let mut options = Vec::with_capacity(self.ends.len());
        let mut from = 0usize;
        for (end, next_state) in &self.ends {
            let to = *end as usize;
            options.push(ScanOption {
                terminals: self.flat[from..to].to_vec(),
                next_state: *next_state,
            });
            from = to;
        }
        Scan { options }
    }
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
    let mut buckets: FxHashMap<Box<[u8]>, usize> = FxHashMap::default();
    let mut groups: Vec<Group> = Vec::new();
    let mut scratch = ScanScratch::default();
    for (token_id, bytes) in vocabulary.iter().enumerate() {
        if bytes.is_empty() {
            rejected += 1;
            continue;
        }
        if !lexer.scan_into(bytes, state, &mut scratch) {
            rejected += 1;
            continue;
        }
        match buckets.get(scratch.key()) {
            Some(&at) => groups[at].tokens.push(token_id as u32),
            None => {
                buckets.insert(scratch.key.clone().into_boxed_slice(), groups.len());
                groups.push(Group { scan: scratch.take(), tokens: vec![token_id as u32] });
            }
        }
    }
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
    // Built once and shared by every state. A state that can settle restarts
    // the scan from the start state, and doing that again for each of a
    // thousand states was most of what this cost.
    let flat = FlatVocabulary::of(vocabulary);
    let start_scans = StartScans::build(lexer, &flat);
    let (per_state, rejected): (Vec<Vec<Group>>, Vec<u32>) = (0..lexer.num_states())
        .into_par_iter()
        .map(|state| {
            let from = LexState(state as u32);
            let mut refused = 0u32;
            // Keyed by the readings as bytes rather than by a `Scan`, so a
            // token that reads like one already seen costs a lookup and a
            // push. Building and hashing a `Scan` per token was the cost.
            let mut buckets: FxHashMap<Box<[u8]>, usize> = FxHashMap::default();
            let mut groups: Vec<Group> = Vec::new();
            let mut scratch = ScanScratch::default();
            for token_id in 0..flat.len() {
                let bytes = flat.get(token_id);
                if bytes.is_empty() {
                    refused += 1;
                    continue;
                }
                if !lexer.scan_into_with(
                    bytes,
                    from,
                    &mut scratch,
                    Some((&start_scans, token_id)),
                ) {
                    refused += 1;
                    continue;
                }
                match buckets.get(scratch.key()) {
                    Some(&at) => groups[at].tokens.push(token_id as u32),
                    None => {
                        buckets
                            .insert(scratch.key.clone().into_boxed_slice(), groups.len());
                        groups.push(Group {
                            scan: scratch.take(),
                            tokens: vec![token_id as u32],
                        });
                    }
                }
            }
            groups.sort_by(|a, b| b.tokens.len().cmp(&a.tokens.len()));
            (groups, refused)
        })
        .unzip();

    VocabularyGroups {
        per_state,
        rejected,
    }
}
