//! Fill the token-to-group tables for the states a request actually reaches.
//!
//! An artifact holds, per lexer state, one bitset over the whole vocabulary per
//! group. That is the translation from a model's tokens to a grammar's
//! terminals, and it is where the memory goes: 19 KiB per group at a
//! 151,669-token vocabulary, hundreds of megabytes for one schema.
//!
//! Most of it is never read. A mask is a pure function of the lexer state, and
//! a real document reaches 2-44% of the states its grammar can - eleven of 440
//! in one case. Building all of them is work no request will use.
//!
//! So states are grouped on demand and kept. The cost that matters is that a
//! miss must not leave the device: filling a state is a scan over the
//! vocabulary, which is a kernel, not a host round trip. This module is the CPU
//! reference for that policy - it measures which states are reached and what
//! they cost, so the device implementation has a target to match.

use std::sync::Arc;

use gpugrammar_lex::{Group, LexState, Lexer, group_state};
use gpugrammar_tables::{Artifact, GroupEntry, Reading, SetKind, TokenSet};
use rustc_hash::FxHashMap;

/// An artifact whose group tables are filled as states are reached.
pub struct Cache {
    artifact: Artifact,
    lexer: Arc<Lexer>,
    vocabulary: Arc<Vec<Vec<u8>>>,
    /// Where a state's groups start in `artifact.groups`, once filled.
    filled: FxHashMap<u32, (u32, u32)>,
    /// Sets already stored, so two states admitting the same tokens share one
    /// copy. Duplication is 2x to 27x across real schemas.
    interned: FxHashMap<(SetKind, u64), TokenSet>,
    misses: usize,
    hits: usize,
}

impl Cache {
    /// Take an artifact emitted with no groups and fill it lazily.
    pub fn new(artifact: Artifact, lexer: Arc<Lexer>, vocabulary: Arc<Vec<Vec<u8>>>) -> Self {
        Self {
            artifact,
            lexer,
            vocabulary,
            filled: FxHashMap::default(),
            interned: FxHashMap::default(),
            misses: 0,
            hits: 0,
        }
    }

    pub fn hits(&self) -> usize {
        self.hits
    }

    pub fn misses(&self) -> usize {
        self.misses
    }

    pub fn resident_bytes(&self) -> usize {
        self.artifact.resident_bytes()
    }

    pub fn filled_states(&self) -> usize {
        self.filled.len()
    }

    /// The artifact, with every state reached so far filled in.
    pub fn artifact(&self) -> &Artifact {
        &self.artifact
    }

    /// Group `state` if it has not been grouped yet.
    pub fn ensure(&mut self, state: u32) {
        if self.filled.contains_key(&state) {
            self.hits += 1;
            return;
        }
        self.misses += 1;

        let (groups, _) = group_state(&self.lexer, &self.vocabulary, LexState(state));
        let words = self.artifact.bitset_words as usize;
        let first = self.artifact.groups.len() as u32;

        for group in &groups {
            let set = self.store(group, words);
            self.artifact.groups.push(GroupEntry {
                lexer_state: state,
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

        let last = self.artifact.groups.len() as u32;
        self.filled.insert(state, (first, last));
        self.artifact.group_offsets[state as usize] = first;
        self.artifact.group_offsets[state as usize + 1] = last;
    }

    fn store(&mut self, group: &Group, words: usize) -> TokenSet {
        gpugrammar_tables::store_set(
            &group.tokens,
            self.artifact.vocab_size as usize,
            words,
            &mut self.artifact.set_payload,
            &mut self.interned,
        )
    }
}
