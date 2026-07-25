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
//!     replay its terminal sequence against a copy of the parser stack
//!     if the replay survives, union the group's token bitset into the mask
//! ```
//!
//! The cost is one replay per group — a few hundred, independent of the
//! vocabulary — rather than one check per token.

use std::sync::Arc;

use gpugrammar_tables::Artifact;

/// A parse in progress: a lexer state plus the LR stack.
#[derive(Debug, Clone)]
pub struct Matcher {
    artifact: Arc<Artifact>,
    lexer_state: u32,
    stack: Vec<u32>,
    history: Vec<Snapshot>,
    max_rollback: usize,
    terminated: bool,
}

#[derive(Debug, Clone)]
struct Snapshot {
    lexer_state: u32,
    stack: Vec<u32>,
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
            stack: vec![artifact.start_parser_state],
            artifact,
            lexer_state: 0,
            history: Vec::new(),
            max_rollback,
            terminated: false,
        }
    }

    pub fn lexer_state(&self) -> u32 {
        self.lexer_state
    }

    pub fn parser_state(&self) -> u32 {
        *self.stack.last().expect("the stack is never empty")
    }

    pub fn is_terminated(&self) -> bool {
        self.terminated
    }

    /// Can the input end here?
    pub fn can_terminate(&self) -> bool {
        self.lexer_state == 0 && self.replay(&[self.artifact.eof_terminal], true).is_some()
    }

    pub fn reset(&mut self) {
        self.lexer_state = 0;
        self.stack.clear();
        self.stack.push(self.artifact.start_parser_state);
        self.history.clear();
        self.terminated = false;
    }

    /// Groups of the current lexer state whose terminals the parser accepts.
    pub fn admissible_groups(&self) -> Vec<usize> {
        let state = self.lexer_state as usize;
        let from = self.artifact.group_offsets[state] as usize;
        let to = self.artifact.group_offsets[state + 1] as usize;
        (from..to)
            .filter(|index| self.admits(&self.artifact.groups[*index]))
            .collect()
    }

    /// Would the parser survive this group, including whatever lexeme it
    /// leaves in progress?
    ///
    /// Checking only the emitted terminals is not enough: a token that ends
    /// mid-lexeme emits none, so it would always be admitted, and a completed
    /// document could be followed by the opening of a second one. A pending
    /// lexeme is therefore required to have at least one continuation the
    /// parser would accept.
    fn admits(&self, group: &gpugrammar_tables::GroupEntry) -> bool {
        self.follow(group).is_some()
    }

    /// The stack after the first terminal sequence the parser can follow.
    ///
    /// A token is admissible when *some* reading of it survives. Trying every
    /// reading is what makes an ambiguous lexicon workable: the parser state,
    /// not a declaration-order tie-break, decides whether `"id"` is a declared
    /// property name or an arbitrary string.
    fn follow(&self, group: &gpugrammar_tables::GroupEntry) -> Option<Vec<u32>> {
        let from = self.artifact.pending_offsets[group.next_lexer_state as usize] as usize;
        let to = self.artifact.pending_offsets[group.next_lexer_state as usize + 1] as usize;
        let pending = &self.artifact.pending_terminals[from..to];
        for choice in &group.terminal_choices {
            let Some(stack) = self.replay(choice, false) else {
                continue;
            };
            if pending.is_empty() {
                return Some(stack);
            }
            let probe = Matcher {
                artifact: self.artifact.clone(),
                lexer_state: group.next_lexer_state,
                stack: stack.clone(),
                history: Vec::new(),
                max_rollback: 0,
                terminated: false,
            };
            if pending
                .iter()
                .any(|terminal| probe.replay(&[*terminal], true).is_some())
            {
                return Some(stack);
            }
        }
        None
    }

    /// Union the admitted groups' bitsets into `mask`.
    pub fn fill_bitmask(&self, mask: &mut [u32]) {
        let words = self.artifact.bitset_words as usize;
        mask[..words].fill(0);
        if self.terminated {
            return;
        }
        for index in self.admissible_groups() {
            let offset = self.artifact.groups[index].bitset_offset as usize;
            for (slot, word) in mask[..words].iter_mut().enumerate() {
                *word |= self.artifact.group_bitsets[offset + slot];
            }
        }
    }

    /// Advance by one token, given the group it belongs to.
    pub fn accept_group(&mut self, group: usize) -> Result<(), Refusal> {
        if self.history.len() == self.max_rollback && self.max_rollback > 0 {
            self.history.remove(0);
        }
        if self.max_rollback > 0 {
            self.history.push(Snapshot {
                lexer_state: self.lexer_state,
                stack: self.stack.clone(),
                terminated: self.terminated,
            });
        }
        let entry = self.artifact.groups[group].clone();
        let Some(stack) = self.follow(&entry) else {
            return Err(Refusal::ParserRejected);
        };
        self.stack = stack;
        self.lexer_state = entry.next_lexer_state;
        Ok(())
    }

    /// Find the group a token belongs to in the current lexer state.
    pub fn group_of(&self, token: u32) -> Option<usize> {
        let state = self.lexer_state as usize;
        let from = self.artifact.group_offsets[state] as usize;
        let to = self.artifact.group_offsets[state + 1] as usize;
        let word = token as usize / 32;
        let bit = 1u32 << (token % 32);
        (from..to).find(|index| {
            let offset = self.artifact.groups[*index].bitset_offset as usize;
            self.artifact.group_bitsets[offset + word] & bit != 0
        })
    }

    pub fn accept_token(&mut self, token: u32) -> Result<(), Refusal> {
        let Some(group) = self.group_of(token) else {
            return Err(Refusal::NotInAnyGroup);
        };
        self.accept_group(group)
    }

    pub fn rollback(&mut self, tokens: usize) {
        for _ in 0..tokens {
            let Some(snapshot) = self.history.pop() else {
                break;
            };
            self.lexer_state = snapshot.lexer_state;
            self.stack = snapshot.stack;
            self.terminated = snapshot.terminated;
        }
    }

    pub fn terminate(&mut self) {
        self.terminated = true;
    }

    /// Run a terminal sequence against a copy of the stack.
    ///
    /// Returns the resulting stack, or `None` if the parser refuses. With
    /// `accept_is_success` an ACCEPT action counts as surviving, which is what
    /// the end-of-input check wants.
    fn replay(&self, terminals: &[u32], accept_is_success: bool) -> Option<Vec<u32>> {
        let mut stack = self.stack.clone();
        for terminal in terminals {
            loop {
                let top = *stack.last()? as usize;
                let action = self.action(top, *terminal)?;
                if action == gpugrammar_lr::tables::ACCEPT {
                    return accept_is_success.then_some(stack);
                }
                if action > 0 {
                    stack.push(gpugrammar_lr::tables::decode_shift(action) as u32);
                    break;
                }
                let production = gpugrammar_lr::tables::decode_reduce(action);
                let lhs = self.artifact.production_lhs[production];
                let arity = self.artifact.production_arity[production] as usize;
                if stack.len() <= arity {
                    return None;
                }
                stack.truncate(stack.len() - arity);
                let exposed = *stack.last()? as usize;
                let target = self.goto(exposed, lhs)?;
                stack.push(target);
            }
        }
        Some(stack)
    }

    fn action(&self, state: usize, terminal: u32) -> Option<i32> {
        let from = self.artifact.action_offsets[state] as usize;
        let to = self.artifact.action_offsets[state + 1] as usize;
        let slice = &self.artifact.action_terminals[from..to];
        slice
            .binary_search(&terminal)
            .ok()
            .map(|index| self.artifact.action_values[from + index])
    }

    fn goto(&self, state: usize, nonterminal: u32) -> Option<u32> {
        let from = self.artifact.goto_offsets[state] as usize;
        let to = self.artifact.goto_offsets[state + 1] as usize;
        let slice = &self.artifact.goto_nonterminals[from..to];
        slice
            .binary_search(&nonterminal)
            .ok()
            .map(|index| self.artifact.goto_targets[from + index])
    }
}
