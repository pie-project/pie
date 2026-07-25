//! Reference runtime over the device artifact.
//!
//! The GPU kernel and this module run the same algorithm; having it in Rust
//! means the tables can be exercised, rolled back and differentially tested
//! without a GPU, and it is what the vLLM adapter drives today.
//!
//! One step is:
//!
//! ```text
//! for each group of the current lexer state
//!     for each way the group's tokens can be read
//!         replay its terminal sequence against a copy of the parser stack
//!     if any replay survives, union the group's token bitset into the mask
//! ```
//!
//! The cost is a few replays per group — a few hundred groups, independent of
//! the vocabulary — rather than one check per token.
//!
//! Scanning a generated lexicon is not deterministic. A whole regular rule
//! becomes one terminal, so `{` is both a complete terminal and the start of a
//! longer one covering an entire object, and a declared property name is also
//! a generic string. Committing to one reading and hoping is wrong: the reading
//! that survives the next token is not knowable yet. The matcher therefore
//! carries every configuration still alive - a lexer state paired with a parser
//! stack - and a token is admissible when it leaves at least one. That is the
//! same shape as a parser with a state set, kept small because the ambiguity is
//! local: it resolves within a token or two.

pub mod cache;

use std::sync::Arc;

use gpugrammar_tables::{Artifact, SetKind};

/// One live reading of the input: a lexer state and the LR stack it produced.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Config {
    lexer_state: u32,
    stack: Vec<u32>,
}

/// How many configurations to carry. Ambiguity in these grammars is local, so
/// the set collapses within a token or two; the cap only bounds the worst case.
/// Dropping configurations can only make the matcher stricter, never looser.
const MAX_CONFIGS: usize = 64;

/// A parse in progress.
#[derive(Debug, Clone)]
pub struct Matcher {
    artifact: Arc<Artifact>,
    configs: Vec<Config>,
    history: Vec<Snapshot>,
    max_rollback: usize,
    terminated: bool,
}

#[derive(Debug, Clone)]
struct Snapshot {
    configs: Vec<Config>,
    terminated: bool,
}

/// Why a token was refused, which is worth distinguishing when debugging a
/// grammar: the scanner and the parser fail for different reasons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    NotInAnyGroup,
    ParserRejected,
}

impl Matcher {
    pub fn new(artifact: Arc<Artifact>, max_rollback: usize) -> Self {
        Self {
            configs: vec![Config {
                lexer_state: 0,
                stack: vec![artifact.start_parser_state],
            }],
            artifact,
            history: Vec::new(),
            max_rollback,
            terminated: false,
        }
    }

    /// The lexer state of the first live configuration, for diagnostics.
    pub fn lexer_state(&self) -> u32 {
        self.configs.first().map_or(0, |config| config.lexer_state)
    }

    /// The parser state of the first live configuration, for diagnostics.
    pub fn parser_state(&self) -> u32 {
        self.configs
            .first()
            .and_then(|config| config.stack.last().copied())
            .unwrap_or(self.artifact.start_parser_state)
    }

    /// How many readings are still alive.
    pub fn num_configs(&self) -> usize {
        self.configs.len()
    }

    pub fn is_terminated(&self) -> bool {
        self.terminated
    }

    /// Can the input end here?
    ///
    /// A lexeme may still be in progress: the closing brace of an object is
    /// carried rather than settled, because a longer terminal starts the same
    /// way. Ending the input is what settles it, so every terminal the lexer
    /// state accepts is tried, in every live configuration.
    pub fn can_terminate(&self) -> bool {
        let eof = self.artifact.eof_terminal;
        self.configs.iter().any(|config| {
            if config.lexer_state == 0
                && replay(&self.artifact, &config.stack, &[eof], true).is_some()
            {
                return true;
            }
            self.accepting(config.lexer_state).iter().any(|terminal| {
                replay(&self.artifact, &config.stack, &[*terminal, eof], true).is_some()
            })
        })
    }

    /// The parser stack of the first live configuration.
    pub fn stack(&self) -> &[u32] {
        self.configs
            .first()
            .map_or(&[][..], |config| config.stack.as_slice())
    }

    /// Put the matcher back into a single known configuration.
    ///
    /// The lazy cache rebuilds a matcher each step as the artifact grows, so it
    /// needs to carry the parse across.
    pub fn restore(&mut self, lexer_state: u32, stack: Vec<u32>) {
        self.configs.clear();
        self.configs.push(Config { lexer_state, stack });
        self.terminated = false;
    }

    pub fn reset(&mut self) {
        self.configs.clear();
        self.configs.push(Config {
            lexer_state: 0,
            stack: vec![self.artifact.start_parser_state],
        });
        self.history.clear();
        self.terminated = false;
    }

    /// Terminals a lexer state accepts right now.
    fn accepting(&self, lexer_state: u32) -> &[u32] {
        let from = self.artifact.accepting_offsets[lexer_state as usize] as usize;
        let to = self.artifact.accepting_offsets[lexer_state as usize + 1] as usize;
        &self.artifact.accepting_terminals[from..to]
    }

    /// Terminals a lexeme left in progress could still become.
    fn pending(&self, lexer_state: u32) -> &[u32] {
        let from = self.artifact.pending_offsets[lexer_state as usize] as usize;
        let to = self.artifact.pending_offsets[lexer_state as usize + 1] as usize;
        &self.artifact.pending_terminals[from..to]
    }

    /// Groups admitted by at least one live configuration.
    ///
    /// Configurations may sit in different lexer states, so the candidate
    /// groups are the union over those states.
    pub fn admissible_groups(&self) -> Vec<usize> {
        let mut groups: Vec<usize> = Vec::new();
        for state in self.live_states() {
            let from = self.artifact.group_offsets[state as usize] as usize;
            let to = self.artifact.group_offsets[state as usize + 1] as usize;
            for index in from..to {
                if !self.successors(&self.artifact.groups[index]).is_empty() {
                    groups.push(index);
                }
            }
        }
        groups.sort_unstable();
        groups.dedup();
        groups
    }

    fn live_states(&self) -> Vec<u32> {
        let mut states: Vec<u32> = self.configs.iter().map(|c| c.lexer_state).collect();
        states.sort_unstable();
        states.dedup();
        states
    }

    /// Every configuration this group leads to, from the ones alive now.
    ///
    /// A reading survives when the parser can follow its terminals and, if it
    /// leaves a lexeme in progress, that lexeme could still become something
    /// the parser would accept.
    fn successors(&self, group: &gpugrammar_tables::GroupEntry) -> Vec<Config> {
        let mut next: Vec<Config> = Vec::new();
        for config in &self.configs {
            if config.lexer_state != group.lexer_state {
                continue;
            }
            for reading in &group.readings {
                let Some(stack) = replay(&self.artifact, &config.stack, &reading.terminals, false)
                else {
                    continue;
                };
                let pending = self.pending(reading.next_lexer_state);
                let viable = pending.is_empty()
                    || pending.iter().any(|terminal| {
                        replay(&self.artifact, &stack, &[*terminal], true).is_some()
                    });
                if !viable {
                    continue;
                }
                let candidate = Config {
                    lexer_state: reading.next_lexer_state,
                    stack,
                };
                if !next.contains(&candidate) {
                    next.push(candidate);
                    if next.len() == MAX_CONFIGS {
                        return next;
                    }
                }
            }
        }
        next
    }

    /// Union the admitted groups' token sets into `mask`.
    ///
    /// A group's set is stored in whichever exact form is smallest, so this
    /// unions a token list, a complement list or a bitset depending on the
    /// group. Only the dense case touches the whole mask.
    pub fn fill_bitmask(&self, mask: &mut [u32]) {
        let words = self.artifact.bitset_words as usize;
        mask[..words].fill(0);
        if self.terminated {
            return;
        }
        for index in self.admissible_groups() {
            let set = self.artifact.groups[index].set;
            let body = self.payload(set);
            match set.kind {
                SetKind::Sparse => {
                    for token in body {
                        mask[*token as usize / 32] |= 1u32 << (*token % 32);
                    }
                }
                SetKind::Complement => {
                    // Everything except the listed tokens, and the listed ones
                    // only if some other group admits them - so build this
                    // group's contribution separately and union it in.
                    let mut all = vec![u32::MAX; words];
                    for token in body {
                        all[*token as usize / 32] &= !(1u32 << (*token % 32));
                    }
                    Self::clear_tail(&mut all, self.artifact.vocab_size as usize);
                    for (slot, word) in mask[..words].iter_mut().enumerate() {
                        *word |= all[slot];
                    }
                }
                SetKind::Dense => {
                    for (slot, word) in mask[..words].iter_mut().enumerate() {
                        *word |= body[slot];
                    }
                }
            }
        }
    }

    /// Bits past the last token must stay zero, or a complement would set them.
    fn clear_tail(mask: &mut [u32], vocab_size: usize) {
        let spare = mask.len() * 32 - vocab_size;
        if spare > 0 {
            let last = mask.len() - 1;
            mask[last] &= u32::MAX >> spare;
        }
    }

    fn payload(&self, set: gpugrammar_tables::TokenSet) -> &[u32] {
        let from = set.offset as usize;
        &self.artifact.set_payload[from..from + set.length as usize]
    }

    /// Does this group hold `token`?
    fn contains(&self, set: gpugrammar_tables::TokenSet, token: u32) -> bool {
        let body = self.payload(set);
        match set.kind {
            SetKind::Sparse => body.binary_search(&token).is_ok(),
            SetKind::Complement => body.binary_search(&token).is_err(),
            SetKind::Dense => body[token as usize / 32] & (1u32 << (token % 32)) != 0,
        }
    }

    /// Advance by one token, given the group it belongs to.
    pub fn accept_group(&mut self, group: usize) -> Result<(), Refusal> {
        if self.history.len() == self.max_rollback && self.max_rollback > 0 {
            self.history.remove(0);
        }
        if self.max_rollback > 0 {
            self.history.push(Snapshot {
                configs: self.configs.clone(),
                terminated: self.terminated,
            });
        }
        let next = self.successors(&self.artifact.groups[group]);
        if next.is_empty() {
            if self.max_rollback > 0 {
                self.history.pop();
            }
            return Err(Refusal::ParserRejected);
        }
        self.configs = next;
        Ok(())
    }

    /// The group holding `token`, in each live lexer state.
    ///
    /// One per state: a token belongs to exactly one group of a given state,
    /// but configurations may sit in different states.
    pub fn groups_of(&self, token: u32) -> Vec<usize> {
        self.live_states()
            .into_iter()
            .filter_map(|state| {
                let from = self.artifact.group_offsets[state as usize] as usize;
                let to = self.artifact.group_offsets[state as usize + 1] as usize;
                (from..to).find(|index| self.contains(self.artifact.groups[*index].set, token))
            })
            .collect()
    }

    /// The group holding `token` in the first live state, for diagnostics.
    pub fn group_of(&self, token: u32) -> Option<usize> {
        self.groups_of(token).into_iter().next()
    }

    pub fn accept_token(&mut self, token: u32) -> Result<(), Refusal> {
        let groups = self.groups_of(token);
        if groups.is_empty() {
            return Err(Refusal::NotInAnyGroup);
        }
        if self.max_rollback > 0 {
            if self.history.len() == self.max_rollback {
                self.history.remove(0);
            }
            self.history.push(Snapshot {
                configs: self.configs.clone(),
                terminated: self.terminated,
            });
        }
        let mut next: Vec<Config> = Vec::new();
        for group in groups {
            for config in self.successors(&self.artifact.groups[group]) {
                if !next.contains(&config) {
                    next.push(config);
                }
            }
        }
        if next.is_empty() {
            if self.max_rollback > 0 {
                self.history.pop();
            }
            return Err(Refusal::ParserRejected);
        }
        self.configs = next;
        Ok(())
    }

    pub fn rollback(&mut self, tokens: usize) {
        for _ in 0..tokens {
            let Some(snapshot) = self.history.pop() else {
                break;
            };
            self.configs = snapshot.configs;
            self.terminated = snapshot.terminated;
        }
    }

    pub fn terminate(&mut self) {
        self.terminated = true;
    }

    fn action(&self, state: usize, terminal: u32) -> Option<i32> {
        action(&self.artifact, state, terminal)
    }
}

/// Run a terminal sequence against a copy of `stack`.
///
/// Returns the resulting stack, or `None` if the parser refuses. With
/// `accept_is_success` an ACCEPT action counts as surviving, which is what the
/// end-of-input check wants.
fn replay(
    artifact: &Artifact,
    from: &[u32],
    terminals: &[u32],
    accept_is_success: bool,
) -> Option<Vec<u32>> {
    let mut stack = from.to_vec();
    for terminal in terminals {
        loop {
            let top = *stack.last()? as usize;
            let value = action(artifact, top, *terminal)?;
            if value == gpugrammar_lr::tables::ACCEPT {
                return accept_is_success.then_some(stack);
            }
            if value > 0 {
                stack.push(gpugrammar_lr::tables::decode_shift(value) as u32);
                break;
            }
            let production = gpugrammar_lr::tables::decode_reduce(value);
            let lhs = artifact.production_lhs[production];
            let arity = artifact.production_arity[production] as usize;
            if stack.len() <= arity {
                return None;
            }
            stack.truncate(stack.len() - arity);
            let exposed = *stack.last()? as usize;
            let target = goto(artifact, exposed, lhs)?;
            stack.push(target);
        }
    }
    Some(stack)
}

fn action(artifact: &Artifact, state: usize, terminal: u32) -> Option<i32> {
    let from = artifact.action_offsets[state] as usize;
    let to = artifact.action_offsets[state + 1] as usize;
    artifact.action_terminals[from..to]
        .binary_search(&terminal)
        .ok()
        .map(|index| artifact.action_values[from + index])
}

fn goto(artifact: &Artifact, state: usize, nonterminal: u32) -> Option<u32> {
    let from = artifact.goto_offsets[state] as usize;
    let to = artifact.goto_offsets[state + 1] as usize;
    artifact.goto_nonterminals[from..to]
        .binary_search(&nonterminal)
        .ok()
        .map(|index| artifact.goto_targets[from + index])
}
