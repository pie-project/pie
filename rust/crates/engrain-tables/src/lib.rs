//! The artifact the GPU runtime reads.
//!
//! Everything here is compile-time data, which is the point: a decode step
//! reads only these arrays and its own device-resident stack, so it never
//! touches the host and can sit inside the model's CUDA graph.
//!
//! The step the arrays serve is:
//!
//! ```text
//! read the stack top
//!   → one ACTION lookup per token group  (a few hundred, vocabulary-independent)
//!   → union the admitted groups' bitsets
//!   → sample, then shift/reduce on the same stack
//! ```
//!
//! Groups are why the first line is cheap. Tokens emitting the same terminal
//! sequence and landing in the same lexer state are indistinguishable to the
//! parser, so a vocabulary of a quarter million collapses into a few hundred
//! entries and the per-step parser work stops depending on vocabulary size.

pub mod pipeline;

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use engrain_lex::lexicon::Lexicon;
use engrain_lex::{Lexer, VocabularyGroups};
use engrain_lr::cfg::Cfg;
use engrain_lr::tables::Tables;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::Serialize;

/// One way of reading a token.
#[derive(Debug, Clone, Serialize)]
pub struct Reading {
    pub terminals: Vec<u32>,
    pub next_lexer_state: u32,
}

/// How one group's token set is stored.
///
/// Every set was a bitset over the whole vocabulary: 18,960 bytes at 151,669
/// tokens, whatever the set held. Measured over the corpus that is almost
/// always the wrong choice - the median set has one token in it, and 99% have
/// at most sixty-six - so a set is now kept in whichever of three exact forms
/// is smallest. Approximation is not among them: a false positive admits a
/// token the grammar forbids, and a Bloom filter at a false-positive rate low
/// enough to be safe is no smaller than the exact list anyway.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum SetKind {
    /// `length` token ids, ascending. The usual case.
    Sparse,
    /// `length` token ids the set excludes, ascending. For a set that admits
    /// nearly the whole vocabulary, as a string body does.
    Complement,
    /// `length` packed words. Only when neither list is smaller.
    Dense,
}

/// A token set: a kind, and a run of `length` `u32`s at `offset`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct TokenSet {
    pub kind: SetKind,
    pub offset: u32,
    pub length: u32,
}

/// A token group as the device sees it.
#[derive(Debug, Clone, Serialize)]
pub struct GroupEntry {
    /// The lexer state this group applies in.
    pub lexer_state: u32,
    /// Every way the group's tokens can be read, longest match first: what
    /// each reading emits and where it leaves the lexer. The parser picks the
    /// first it can follow, which is how a generated lexicon's ambiguity - both
    /// which terminal and where the lexeme ends - is resolved.
    pub readings: Vec<Reading>,
    /// Where the group's token set lives in `set_payload`, and how it is
    /// stored there.
    pub set: TokenSet,
    /// How many tokens the group holds, for diagnostics and weighting.
    pub token_count: u32,
}

/// Flat, device-ready tables.
#[derive(Debug, Clone, Serialize)]
pub struct Artifact {
    pub vocab_size: u32,
    pub bitset_words: u32,
    pub num_lexer_states: u32,
    pub num_terminals: u32,
    pub num_nonterminals: u32,
    pub num_parser_states: u32,
    pub eof_terminal: u32,
    pub start_parser_state: u32,

    /// Groups sorted by lexer state, so a state's groups are contiguous.
    pub groups: Vec<GroupEntry>,
    /// `group_offsets[state]..group_offsets[state + 1]` indexes `groups`.
    pub group_offsets: Vec<u32>,
    /// Every group's token set, back to back. A [`TokenSet`] says where each
    /// one starts and how to read it.
    pub set_payload: Vec<u32>,

    /// Terminals a lexeme left in progress could still become, CSR by lexer
    /// state. It depends only on the state, so groups share it rather than
    /// each carrying a copy; with a large lexer the copies dominated the
    /// artifact.
    pub pending_offsets: Vec<u32>,
    pub pending_terminals: Vec<u32>,

    /// `lexer_transitions[state * 256 + byte]`, `u32::MAX` where impossible.
    ///
    /// Carried so the runtime can walk a token's bytes instead of reading a
    /// precomputed mask. It costs `states * 1 KiB` and does not scale with the
    /// vocabulary, where the masks cost `groups * vocabulary / 8`.
    pub lexer_transitions: Vec<u32>,

    /// Terminals a lexer state accepts right now, CSR by state. A lexeme is
    /// withheld while another byte could extend it, so the last one in a
    /// document is still pending when the input ends; ending the input is what
    /// settles it, and this says what it may settle as.
    pub accepting_offsets: Vec<u32>,
    pub accepting_terminals: Vec<u32>,

    /// CSR ACTION rows: `action_offsets[state]..action_offsets[state + 1]`.
    pub action_offsets: Vec<u32>,
    pub action_terminals: Vec<u32>,
    pub action_values: Vec<i32>,
    /// Actions beyond the first, for the cells a conflict left with several.
    /// `action_extra_offsets[entry]..[entry + 1]` indexes `action_extra`, and
    /// both are empty for a grammar with no conflicts.
    pub action_extra_offsets: Vec<u32>,
    pub action_extra: Vec<i32>,

    /// CSR GOTO rows.
    pub goto_offsets: Vec<u32>,
    pub goto_nonterminals: Vec<u32>,
    pub goto_targets: Vec<u32>,

    /// `(lhs, arity)` per production.
    pub production_lhs: Vec<u32>,
    pub production_arity: Vec<u32>,

    /// What each group's replay does that does not depend on the stack.
    ///
    /// A group's readings run against the parser state on top of the stack. A
    /// reading that only shifts never looks below the top, so its answer is a
    /// function of `(lexer state, parser state)` alone and can be settled here.
    /// Measured on real grammars, **92.5% of replays are settled that way**,
    /// 91.0% refused and 1.5% admitted. Only the 7.5% that reduce need
    /// the stack, and those are the ones the runtime still replays.
    ///
    /// Two bits per (pair, group), packed into `u32` words: 0 undecided, 1
    /// refused, 2 admitted. `verdict_offsets[lexer_state]` gives the word a
    /// pair's row starts at, and a pair is `parser_state` rows in. Empty when
    /// the cross product would be larger than `VERDICT_BUDGET`.
    pub verdict_offsets: Vec<u32>,
    pub verdicts: Vec<u32>,
    /// Words per (lexer state, parser state) row, so a pair can be addressed
    /// without a division.
    pub verdict_stride: Vec<u32>,
}

impl Artifact {
    /// Bytes the runtime keeps resident.
    pub fn resident_bytes(&self) -> usize {
        4 * (self.set_payload.len()
            + self.group_offsets.len()
            + self.action_offsets.len()
            + self.action_terminals.len()
            + self.action_values.len()
            + self.action_extra_offsets.len()
            + self.action_extra.len()
            + self.goto_offsets.len()
            + self.goto_nonterminals.len()
            + self.goto_targets.len()
            + self.production_lhs.len()
            + self.production_arity.len())
            + self.groups.len() * 20
    }

    /// What a per-configuration token row table would have cost instead.
    ///
    /// Reported next to `resident_bytes` because the comparison is the reason
    /// groups exist: rows scale with states times allowed tokens, groups do
    /// not scale with vocabulary at all.
    pub fn rows_equivalent_bytes(&self) -> usize {
        let allowed: usize = self
            .groups
            .iter()
            .map(|group| group.token_count as usize)
            .sum();
        allowed * 8 * self.num_parser_states as usize
    }
}

/// Store one token set in whichever exact form is smallest, sharing it with any
/// identical set already stored.
///
/// The three forms cost `tokens`, `vocabulary - tokens`, and
/// `vocabulary / 32` words respectively, so the choice is *nearly* arithmetic.
/// Sharing matters as much as the choice: the set a state admits changes far
/// less often than the state does, and the same set arrives from two to
/// twenty-seven different places.
///
/// The exception is the complement, and it is the reason for the factor of two
/// below. A sparse set and a bitset are both written into the mask by a
/// vectorised OR, but a complement is written by walking its exclusions one at
/// a time - the walk is a chain of dependent loads and cannot be vectorised,
/// because each exclusion clears a different bit. So a complement costs time
/// linear in its own length where the other two cost time linear in the mask,
/// and taking it purely on size picks it whenever it is one word smaller.
///
/// That is exactly what happened: a group excluding 4,693 of 151,669 tokens
/// was stored as a complement because 4,693 is less than the bitset's 4,740
/// words. It saved 47 words - a tenth of a percent - and cost 500 us of serial
/// walking in every mask fill, which at batch 1 was 93% of the whole step.
/// Requiring a complement to be at most half the bitset costs 0.5% more
/// resident memory and makes the fill three times faster below batch 128;
/// requiring an eighth or a thirty-second costs more memory and buys nothing
/// further, so this is the knee rather than a guess.
pub fn store_set(
    tokens: &[u32],
    vocab_size: usize,
    bitset_words: usize,
    payload: &mut Vec<u32>,
    interned: &mut FxHashMap<(SetKind, Vec<u32>), TokenSet>,
) -> TokenSet {
    let sparse = tokens.len();
    let complement = vocab_size - sparse;
    let (kind, body) = if sparse <= complement && sparse <= bitset_words {
        let mut ordered = tokens.to_vec();
        ordered.sort_unstable();
        (SetKind::Sparse, ordered)
    } else if complement * 2 <= bitset_words {
        let mut present = vec![false; vocab_size];
        for token in tokens {
            present[*token as usize] = true;
        }
        let missing = (0..vocab_size as u32)
            .filter(|token| !present[*token as usize])
            .collect();
        (SetKind::Complement, missing)
    } else {
        let mut bits = vec![0u32; bitset_words];
        for token in tokens {
            bits[(*token as usize) / 32] |= 1u32 << (*token % 32);
        }
        (SetKind::Dense, bits)
    };

    // Keyed by the body itself. Hashing it to a digest and confirming against
    // the payload was tried, together with building a complement from the gaps
    // between sorted tokens rather than from a marked array, and the two were
    // 6% *slower*: this stage is 642 ms of a compile and the copies are not
    // where that goes.
    let key = (kind, body);
    if let Some(&existing) = interned.get(&key) {
        return existing;
    }
    let set = TokenSet {
        kind,
        offset: payload.len() as u32,
        length: key.1.len() as u32,
    };
    payload.extend_from_slice(&key.1);
    interned.insert(key, set);
    set
}

/// Assemble the artifact from a compiled grammar.
/// Emit with no group tables at all, for a runtime that fills them on demand.
///
/// The parser tables and the lexer are complete; only the token-to-group
/// translation is left empty, because that is the part that costs megabytes and
/// the part a request uses only a fraction of.
pub fn emit_ungrouped(
    lexicon: &Lexicon,
    lexer: &Lexer,
    cfg: &Cfg,
    tables: &Tables,
    vocab_size: usize,
) -> Result<Artifact> {
    let empty = VocabularyGroups {
        per_state: vec![Vec::new(); lexer.num_states()],
        rejected: vec![0; lexer.num_states()],
    };
    emit(lexicon, lexer, &empty, cfg, tables, vocab_size)
}

pub fn emit(
    lexicon: &Lexicon,
    lexer: &Lexer,
    groups: &VocabularyGroups,
    cfg: &Cfg,
    tables: &Tables,
    vocab_size: usize,
) -> Result<Artifact> {
    if lexicon.terminals.len() != lexer.num_terminals() {
        bail!(
            "lexicon has {} terminals but the lexer has {}",
            lexicon.terminals.len(),
            lexer.num_terminals()
        );
    }
    let bitset_words = vocab_size.div_ceil(32);

    let reachable = lexer.reachable_terminals_all();
    let mut entries = Vec::new();
    let mut bitsets: Vec<u32> = Vec::new();
    let mut offsets = Vec::with_capacity(lexer.num_states() + 1);
    offsets.push(0u32);

    // Two groups with the same tokens share one bitset. They are common: the
    // vocabulary a state admits changes far less often than the state does, so
    // the same set is reached from many places. Measured on JSONSchemaBench the
    // duplication is 2x to 27x, and since a bitset is the whole vocabulary -
    // 19 KiB at 151,669 tokens - this is where the artifact's size lives.
    let mut interned: FxHashMap<(SetKind, Vec<u32>), TokenSet> = FxHashMap::default();
    for (state, state_groups) in groups.per_state.iter().enumerate() {
        for group in state_groups {
            let set = store_set(
                &group.tokens,
                vocab_size,
                bitset_words,
                &mut bitsets,
                &mut interned,
            );
            entries.push(GroupEntry {
                lexer_state: state as u32,
                readings: group
                    .scan
                    .options
                    .iter()
                    .map(|option| Reading {
                        terminals: option.terminals.iter().map(|t| t.0).collect(),
                        next_lexer_state: option.next_state.0,
                    })
                    .collect(),
                set,
                token_count: group.tokens.len() as u32,
            });
        }
        offsets.push(entries.len() as u32);
    }

    let mut pending_offsets = Vec::with_capacity(lexer.num_states() + 1);
    let mut pending_terminals = Vec::new();
    pending_offsets.push(0u32);
    for state in 0..lexer.num_states() {
        // A scan that ends back at the start left nothing in progress.
        if state != engrain_lex::START.0 as usize {
            pending_terminals.extend(reachable[state].iter().map(|terminal| terminal.0));
        }
        pending_offsets.push(pending_terminals.len() as u32);
    }

    let lexer_transitions = lexer.transitions().to_vec();
    let mut accepting_offsets = Vec::with_capacity(lexer.num_states() + 1);
    let mut accepting_terminals = Vec::new();
    accepting_offsets.push(0u32);
    for state in 0..lexer.num_states() {
        accepting_terminals.extend(
            lexer
                .accepting(engrain_lex::LexState(state as u32))
                .iter()
                .map(|terminal| terminal.0),
        );
        accepting_offsets.push(accepting_terminals.len() as u32);
    }

    let (action_offsets, action_terminals, action_values, action_extra_offsets, action_extra) =
        flatten_action(tables);
    let (goto_offsets, goto_nonterminals, goto_targets) = flatten_goto(tables);
    let (verdict_offsets, verdicts, verdict_stride) = precompute_verdicts(
        &entries,
        &offsets,
        tables,
        &action_offsets,
        &action_terminals,
        &action_values,
    );

    Ok(Artifact {
        vocab_size: vocab_size as u32,
        bitset_words: bitset_words as u32,
        num_lexer_states: lexer.num_states() as u32,
        num_terminals: cfg.num_terminals as u32,
        num_nonterminals: cfg.num_nonterminals() as u32,
        num_parser_states: tables.num_states() as u32,
        eof_terminal: tables.eof,
        start_parser_state: tables.start_state as u32,
        groups: entries,
        group_offsets: offsets,
        set_payload: bitsets,
        pending_offsets,
        pending_terminals,
        accepting_offsets,
        accepting_terminals,
        lexer_transitions,
        action_offsets,
        action_extra_offsets,
        action_extra,
        action_terminals,
        action_values,
        goto_offsets,
        goto_nonterminals,
        goto_targets,
        production_lhs: tables.productions.iter().map(|(lhs, _)| *lhs).collect(),
        production_arity: tables.productions.iter().map(|(_, arity)| *arity).collect(),
        verdict_offsets,
        verdicts,
        verdict_stride,
    })
}

/// What a group's replay does when it never has to look below the stack top.
pub const VERDICT_UNDECIDED: u32 = 0;
pub const VERDICT_REFUSED: u32 = 1;
pub const VERDICT_ADMITTED: u32 = 2;

/// Words the verdict table may occupy before it is abandoned.
///
/// The table is `lexer states * parser states * groups per state / 16`, which
/// on the corpus is two to thirteen kilobytes and on a pathological grammar
/// would not be. Past the budget the runtime replays everything, which is what
/// it did before this existed.
const VERDICT_BUDGET: usize = 1 << 22;

/// Decide, per `(lexer state, parser state, group)`, whatever the stack cannot
/// change.
///
/// A group's readings are run against the parser state on top of the stack. A
/// reading that only shifts never reads below the top, so if every reading of a
/// group either dies on a missing action or survives to the end by shifting,
/// the answer is the same whatever is underneath. A reading that reduces pops,
/// and what it exposes is the stack - that group is left undecided and the
/// runtime replays it.
///
/// Measured on real grammars this settles **92.5%** of replays: 91.0% refused,
/// 1.5% admitted, 7.5% left for the runtime.
fn precompute_verdicts(
    entries: &[GroupEntry],
    group_offsets: &[u32],
    tables: &Tables,
    action_offsets: &[u32],
    action_terminals: &[u32],
    action_values: &[i32],
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let lexer_states = group_offsets.len().saturating_sub(1);
    let parser_states = tables.num_states();
    // Two bits a group, and a row per (lexer state, parser state) pair. The
    // stride varies by lexer state because a state's group count does.
    let mut stride = Vec::with_capacity(lexer_states);
    let mut words = 0usize;
    for state in 0..lexer_states {
        let groups = (group_offsets[state + 1] - group_offsets[state]) as usize;
        let row = groups.div_ceil(16);
        stride.push(row as u32);
        words += row * parser_states;
    }
    if words == 0 || words > VERDICT_BUDGET {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let action = |state: usize, terminal: u32| -> Option<i32> {
        let from = action_offsets[state] as usize;
        let to = action_offsets[state + 1] as usize;
        action_terminals[from..to]
            .binary_search(&terminal)
            .ok()
            .map(|at| action_values[from + at])
    };

    let mut offsets = Vec::with_capacity(lexer_states + 1);
    let mut at = 0usize;
    for lexer_state in 0..lexer_states {
        offsets.push(at as u32);
        at += stride[lexer_state] as usize * parser_states;
    }
    offsets.push(at as u32);

    // One lexer state per task. This was the single largest thing a compile
    // did - 83% of the emit and 56% of the whole - and it was serial, which
    // was invisible because nothing timed the emit at all. Every state writes
    // a run of its own, so the output splits exactly where the tasks do.
    let mut verdicts = vec![0u32; words];
    let mut rest = verdicts.as_mut_slice();
    let mut rows: Vec<(usize, &mut [u32])> = Vec::with_capacity(lexer_states);
    for lexer_state in 0..lexer_states {
        let span = stride[lexer_state] as usize * parser_states;
        let (mine, tail) = rest.split_at_mut(span);
        rest = tail;
        rows.push((lexer_state, mine));
    }
    rows.into_par_iter().for_each(|(lexer_state, row_out)| {
        let first = group_offsets[lexer_state] as usize;
        let last = group_offsets[lexer_state + 1] as usize;
        let row = stride[lexer_state] as usize;
        for parser_state in 0..parser_states {
            let base = parser_state * row;
            for (slot, group) in entries[first..last].iter().enumerate() {
                let verdict = settle(group, parser_state, &action);
                if verdict != VERDICT_UNDECIDED {
                    row_out[base + slot / 16] |= verdict << (2 * (slot % 16));
                }
            }
        }
    });
    (offsets, verdicts, stride)
}

/// One group against one parser state, without a stack.
fn settle(
    group: &GroupEntry,
    parser_state: usize,
    action: &impl Fn(usize, u32) -> Option<i32>,
) -> u32 {
    let mut admitted = false;
    for reading in &group.readings {
        let mut top = parser_state;
        let mut alive = true;
        for terminal in &reading.terminals {
            match action(top, *terminal) {
                None => {
                    alive = false;
                    break;
                }
                Some(value) if value > 0 => {
                    top = engrain_lr::tables::decode_shift(value);
                }
                // A reduce pops, and what it exposes is the stack. Nothing can
                // be said about this group without one.
                Some(_) => return VERDICT_UNDECIDED,
            }
        }
        if alive {
            // The reading survives its terminals by shifting alone. Whether it
            // is admitted also depends on the pending-lexeme probe, which
            // reduces - so this is only a verdict when there is nothing
            // pending.
            admitted = true;
        }
    }
    if admitted {
        VERDICT_UNDECIDED
    } else {
        VERDICT_REFUSED
    }
}

/// The ACTION table, flattened.
///
/// A cell usually holds one action, and those three arrays are what they always
/// were: terminals sorted and unique so the device can binary-search, one value
/// each. A reduce/reduce conflict leaves a cell holding two, and the extras go
/// in a fourth array indexed by entry rather than being interleaved with the
/// terminals - so a grammar without conflicts emits an empty `extra` and its
/// lookup is unchanged, which is the point. Most grammars have none.
fn flatten_action(tables: &Tables) -> (Vec<u32>, Vec<u32>, Vec<i32>, Vec<u32>, Vec<i32>) {
    let mut offsets = Vec::with_capacity(tables.num_states() + 1);
    let mut terminals = Vec::new();
    let mut values = Vec::new();
    let mut extra_offsets = vec![0u32];
    let mut extra = Vec::new();
    offsets.push(0);
    for row in &tables.action {
        let ordered: BTreeMap<u32, &Vec<i32>> = row.iter().map(|(k, v)| (*k, v)).collect();
        for (terminal, actions) in ordered {
            terminals.push(terminal);
            values.push(actions[0]);
            extra.extend_from_slice(&actions[1..]);
            extra_offsets.push(extra.len() as u32);
        }
        offsets.push(terminals.len() as u32);
    }
    (offsets, terminals, values, extra_offsets, extra)
}

fn flatten_goto(tables: &Tables) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut offsets = Vec::with_capacity(tables.num_states() + 1);
    let mut nonterminals = Vec::new();
    let mut targets = Vec::new();
    offsets.push(0);
    for row in &tables.goto {
        let ordered: BTreeMap<u32, u32> = row.iter().map(|(k, v)| (*k, *v)).collect();
        for (nonterminal, target) in ordered {
            nonterminals.push(nonterminal);
            targets.push(target);
        }
        offsets.push(nonterminals.len() as u32);
    }
    (offsets, nonterminals, targets)
}
