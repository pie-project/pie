//! Single-DFA fast path engine.
//!
//! When the entire grammar compiles down to a single DFA (no RuleRef edges
//! in the root rule), this engine bypasses the stack parser entirely.
//! Raw `byte_table[state*256+byte]` lookups at ~2ns/byte.

use std::collections::VecDeque;

use crate::structured::bitmask::{self, set_bit};
use crate::structured::compiled_grammar::CompiledGrammar;
use crate::structured::fsm::StateId;
use crate::structured::tokenizer::TokenizerInfo;

pub(super) struct SingleDfaEngine {
    pub(super) rule_idx: usize,
    pub(super) state: u16,
    pub(super) history: VecDeque<u16>,
}

impl SingleDfaEngine {
    pub(super) fn new(rule_idx: usize, initial_state: u16) -> Self {
        Self {
            rule_idx,
            state: initial_state,
            history: VecDeque::new(),
        }
    }

    /// Advance through all bytes via byte_table lookup.
    /// On success, pushes previous state to history and updates current state.
    /// On failure, returns false without modifying state.
    pub(super) fn advance_bytes(&mut self, compiled: &CompiledGrammar, bytes: &[u8]) -> bool {
        let bt = compiled.rule_dfas[self.rule_idx].fsm.byte_table();
        let mut state = self.state as usize;
        for &byte in bytes {
            let next = bt[state * 256 + byte as usize];
            if next == 0xFFFF {
                return false;
            }
            state = next as usize;
        }
        self.history.push_back(self.state);
        self.state = state as u16;
        true
    }

    /// Whether the DFA is in an accepting state.
    pub(super) fn is_completed(&self, compiled: &CompiledGrammar) -> bool {
        compiled.rule_dfas[self.rule_idx].ends[self.state as usize]
    }

    /// Hash of current state for bitmask caching.
    pub(super) fn state_hash(&self) -> u64 {
        self.state as u64
    }

    /// Rollback num_tokens tokens by popping from history.
    pub(super) fn rollback(&mut self, num_tokens: usize) -> usize {
        let n = num_tokens.min(self.history.len());
        for _ in 0..n {
            self.state = self.history.pop_back().unwrap();
        }
        n
    }

    /// Reset to the DFA start state.
    pub(super) fn reset(&mut self, compiled: &CompiledGrammar) {
        self.state = compiled.rule_dfas[self.rule_idx].start.0 as u16;
        self.history.clear();
    }

    /// Find a deterministic prefix by walking DFA edges.
    pub(super) fn find_jump_forward(&self, compiled: &CompiledGrammar) -> String {
        let rule_dfa = &compiled.rule_dfas[self.rule_idx];
        if rule_dfa.ends[self.state as usize] {
            return String::new();
        }
        let mut result = Vec::new();
        let mut state = self.state;
        loop {
            if rule_dfa.ends[state as usize] {
                break;
            }
            let edges = rule_dfa.fsm.edges(StateId(state as u32));
            let byte = match super::deterministic_byte(edges) {
                Some(b) => b,
                None => break,
            };
            // Find the target state for this byte
            let bt = rule_dfa.fsm.byte_table();
            let next = bt[state as usize * 256 + byte as usize];
            if next == 0xFFFF {
                break;
            }
            result.push(byte);
            state = next;
        }
        String::from_utf8(result).unwrap_or_default()
    }

    /// Fill bitmask for current DFA state using pre-computed masks and trie walk.
    pub(super) fn fill_bitmask(
        &self,
        compiled: &CompiledGrammar,
        tokenizer_info: &TokenizerInfo,
        bitmask: &mut [u32],
        scratch_stack: &mut Vec<u16>,
        scratch_prefix: &mut Vec<u8>,
    ) {
        let hash = self.state_hash();
        if compiled.get_cached_bitmask(hash, bitmask) {
            return;
        }

        // DFA mask fast path
        let dfa_key = (self.rule_idx as u32, self.state as u32);
        let mut need_trie_walk = false;
        if let Some(mask) = compiled.token_masks.get(&dfa_key) {
            for (j, &word) in mask.accepted_mask.iter().enumerate() {
                if j < bitmask.len() {
                    bitmask[j] |= word;
                }
            }
            if !mask.uncertain_tokens.is_empty() {
                need_trie_walk = true;
            }
        }

        if need_trie_walk {
            self.fill_bitmask_trie_walk(
                compiled, tokenizer_info, bitmask,
                scratch_stack, scratch_prefix,
            );
        }

        compiled.cache_bitmask(hash, bitmask);
    }

    /// Trie walk for uncertain tokens using raw byte_table lookups.
    fn fill_bitmask_trie_walk(
        &self,
        compiled: &CompiledGrammar,
        tokenizer_info: &TokenizerInfo,
        bitmask: &mut [u32],
        stack: &mut Vec<u16>,
        active_prefix: &mut Vec<u8>,
    ) {
        let sorted = tokenizer_info.sorted_vocab();
        let trie_end = tokenizer_info.trie_subtree_end();
        let bt = compiled.rule_dfas[self.rule_idx].fsm.byte_table();

        stack.clear();
        stack.push(self.state);
        active_prefix.clear();

        let mut i = 0;
        while i < sorted.len() {
            let (token_id, ref token_str) = sorted[i];
            let bytes = token_str.as_bytes();

            if bytes.is_empty() || bitmask::get_bit(bitmask, token_id as usize) {
                i += 1;
                continue;
            }

            // Rewind to common prefix
            let common = super::longest_common_prefix(bytes, active_prefix);
            if common < active_prefix.len() {
                stack.truncate(common + 1);
                active_prefix.truncate(common);
            }

            // Advance through remaining bytes
            let mut dead = false;
            for &byte in &bytes[common..] {
                let state = *stack.last().unwrap() as usize;
                let next = bt[state * 256 + byte as usize];
                if next == 0xFFFF {
                    if active_prefix.is_empty() {
                        i = trie_end[i];
                    } else {
                        i += 1;
                    }
                    dead = true;
                    break;
                }
                stack.push(next);
                active_prefix.push(byte);
            }

            if !dead {
                set_bit(bitmask, token_id as usize);
                i += 1;
            }
        }
    }
}
