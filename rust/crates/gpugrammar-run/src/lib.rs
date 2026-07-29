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

/// How many configurations to carry.
///
/// Most ambiguity in these grammars is local and collapses within a token or
/// two, but order-free objects keep one configuration per subset of required
/// properties they might have completed, and nesting multiplies those.
///
/// Dropping configurations can only make the matcher stricter, never looser, so
/// this is a source of refusals rather than of wrong acceptances - every
/// document in this corpus that the parser refuses although its schema accepts
/// it is refused after the set was truncated here, and all of them at the
/// closing brace, having lost the configuration that had seen every required
/// property. Settable so that the cost of the ceiling can be measured rather
/// than assumed.
fn max_configs() -> usize {
    static LIMIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("GPUGRAMMAR_MAX_CONFIGS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(128)
    })
}

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
                && !replay_all(&self.artifact, &config.stack, &[eof], true).is_empty()
            {
                return true;
            }
            self.accepting(config.lexer_state).iter().any(|terminal| {
                !replay_all(&self.artifact, &config.stack, &[*terminal, eof], true).is_empty()
            })
        })
    }

    /// Every live configuration, as a lexer state and the stack it produced.
    pub fn configurations(&self) -> Vec<(u32, Vec<u32>)> {
        self.configs
            .iter()
            .map(|config| (config.lexer_state, config.stack.clone()))
            .collect()
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
                // Every derivation the reading admits, not the first: two of
                // them are two places the parse can be, and both stay alive.
                for stack in replay_all(&self.artifact, &config.stack, &reading.terminals, false)
                {
                    let pending = self.pending(reading.next_lexer_state);
                    let viable = pending.is_empty()
                        || pending.iter().any(|terminal| {
                            !replay_all(&self.artifact, &stack, &[*terminal], true).is_empty()
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
                        if next.len() == max_configs() {
                            return next;
                        }
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

}

/// Run a terminal sequence against a copy of `stack`.
///
/// Returns the resulting stack, or `None` if the parser refuses. With
/// `accept_is_success` an ACCEPT action counts as surviving, which is what the
/// end-of-input check wants.
/// How many derivations one replay will follow before giving up.
///
/// A reduce/reduce conflict doubles the paths, and they compound over a
/// reading's terminals, so this is bounded rather than exhaustive. Losing a
/// derivation can only make the parser stricter, never looser, so the bound is
/// a source of refusals - which is why it is generous and why exceeding it is
/// worth knowing about.
const MAX_PATHS: usize = 16;

/// Run a terminal sequence against a copy of `stack`, following every
/// derivation.
///
/// Usually there is one. The grammars a JSON Schema lowers to are ambiguous
/// where its `oneOf` branches overlap, and a mask does not need the ambiguity
/// resolved: it needs to know whether *some* derivation admits the token, and
/// what states all of them reach. So a cell holding two actions forks here
/// rather than being refused when the tables were built.
fn replay_all(
    artifact: &Artifact,
    from: &[u32],
    terminals: &[u32],
    accept_is_success: bool,
) -> Vec<Vec<u32>> {
    let mut live: Vec<Vec<u32>> = vec![from.to_vec()];
    let mut accepted: Vec<Vec<u32>> = Vec::new();
    for terminal in terminals {
        // Stacks that have not yet consumed this terminal, against those that
        // have. A reduction returns to the first; a shift moves to the second.
        let mut agenda = std::mem::take(&mut live);
        let mut settled: Vec<Vec<u32>> = Vec::new();
        let mut steps = 0usize;
        while let Some(stack) = agenda.pop() {
            steps += 1;
            if steps > MAX_PATHS * 64 {
                break;
            }
            let Some(&top) = stack.last() else {
                continue;
            };
            let Some(actions) = actions_for(artifact, top as usize, *terminal) else {
                continue;
            };
            for value in actions {
                if value == gpugrammar_lr::tables::ACCEPT {
                    if accept_is_success {
                        accepted.push(stack.clone());
                    }
                } else if value > 0 {
                    if settled.len() < MAX_PATHS {
                        let mut shifted = stack.clone();
                        shifted.push(gpugrammar_lr::tables::decode_shift(value) as u32);
                        settled.push(shifted);
                    }
                } else if agenda.len() < MAX_PATHS
                    && let Some(reduced) = reduce_once(artifact, stack.clone(), value)
                {
                    agenda.push(reduced);
                }
            }
        }
        live = settled;
        live.sort();
        live.dedup();
        if live.is_empty() {
            break;
        }
    }
    live.extend(accepted);
    live.sort();
    live.dedup();
    live
}

/// Pop a production's right-hand side and push where GOTO lands.
fn reduce_once(artifact: &Artifact, mut stack: Vec<u32>, value: i32) -> Option<Vec<u32>> {
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
    Some(stack)
}

/// Every action a cell holds, first one first.
fn actions_for(artifact: &Artifact, state: usize, terminal: u32) -> Option<Vec<i32>> {
    let from = artifact.action_offsets[state] as usize;
    let to = artifact.action_offsets[state + 1] as usize;
    let at = from
        + artifact.action_terminals[from..to]
            .binary_search(&terminal)
            .ok()?;
    let mut values = vec![artifact.action_values[at]];
    if !artifact.action_extra_offsets.is_empty() {
        let low = artifact.action_extra_offsets[at] as usize;
        let high = artifact.action_extra_offsets[at + 1] as usize;
        values.extend_from_slice(&artifact.action_extra[low..high]);
    }
    Some(values)
}

fn goto(artifact: &Artifact, state: usize, nonterminal: u32) -> Option<u32> {
    let from = artifact.goto_offsets[state] as usize;
    let to = artifact.goto_offsets[state + 1] as usize;
    artifact.goto_nonterminals[from..to]
        .binary_search(&nonterminal)
        .ok()
        .map(|index| artifact.goto_targets[from + index])
}
