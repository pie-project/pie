use std::collections::VecDeque;

use crate::bitmask::{self, set_bit};
use crate::compiled_grammar::CompiledGrammar;
use tokenizer::Tokenizer;

#[derive(Clone)]
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

    pub(super) fn is_completed(&self, compiled: &CompiledGrammar) -> bool {
        compiled.rule_dfas[self.rule_idx].ends[self.state as usize]
    }

    pub(super) fn rollback(&mut self, num_tokens: usize) -> usize {
        let n = num_tokens.min(self.history.len());
        for _ in 0..n {
            self.state = self.history.pop_back().unwrap();
        }
        n
    }

    pub(super) fn reset(&mut self, compiled: &CompiledGrammar) {
        self.state = compiled.rule_dfas[self.rule_idx].start.0 as u16;
        self.history.clear();
    }

    pub(super) fn fill_bitmask(
        &self,
        compiled: &CompiledGrammar,
        tokenizer_info: &Tokenizer,
        bitmask: &mut [u32],
        scratch_stack: &mut Vec<u16>,
        scratch_prefix: &mut Vec<u8>,
        cache_key: &mut Vec<u64>,
    ) {
        cache_key.clear();
        cache_key.extend([0, self.rule_idx as u64, self.state as u64]);
        if compiled.get_cached_bitmask(cache_key, bitmask) {
            return;
        }

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
                compiled,
                tokenizer_info,
                bitmask,
                scratch_stack,
                scratch_prefix,
            );
        }

        compiled.cache_bitmask(cache_key, bitmask);
    }

    fn fill_bitmask_trie_walk(
        &self,
        compiled: &CompiledGrammar,
        tokenizer_info: &Tokenizer,
        bitmask: &mut [u32],
        stack: &mut Vec<u16>,
        active_prefix: &mut Vec<u8>,
    ) {
        let sorted = tokenizer_info.sorted_token_ids();
        let trie_end = tokenizer_info.trie_subtree_end();
        let bt = compiled.rule_dfas[self.rule_idx].fsm.byte_table();

        stack.clear();
        stack.push(self.state);
        active_prefix.clear();

        let mut i = 0;
        while i < sorted.len() {
            let token_id = sorted[i];
            let bytes = tokenizer_info
                .decoded_token_bytes(token_id)
                .expect("sorted token IDs have decoded bytes");

            if bitmask::get_bit(bitmask, token_id as usize) {
                i += 1;
                continue;
            }

            let common = super::longest_common_prefix(bytes, active_prefix);
            if common < active_prefix.len() {
                stack.truncate(common + 1);
                active_prefix.truncate(common);
            }

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
