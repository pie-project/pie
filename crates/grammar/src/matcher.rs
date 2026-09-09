mod single_dfa;
mod stack_parser;

use std::collections::VecDeque;
use std::sync::Arc;

use crate::bitmask::{self, clear_bit, set_bit};
use crate::compiled_grammar::CompiledGrammar;
use crate::fsm::{FsmEdge, StateId};
use crate::grammar::Grammar;
use tokenizer::Tokenizer;

use single_dfa::SingleDfaEngine;
use stack_parser::{SmallDedup, StackParser, StackState};

#[allow(
    clippy::large_enum_variant,
    reason = "constructed once per matcher, dispatched per byte; see above"
)]
#[derive(Clone)]
enum ParserEngine {
    SingleDfa(SingleDfaEngine),
    Stack(StackParser),
}

pub struct GrammarMatcher {
    engine: ParserEngine,
    compiled: Arc<CompiledGrammar>,
    tokenizer: Arc<Tokenizer>,
    stop_token_ids: Vec<u32>,
    token_length_history: VecDeque<usize>,
    terminated: bool,
    max_rollback_tokens: usize,
    trie_scratch: TrieWalkScratch,
    bitmask_scratch: Vec<u32>,
    bitmask_cache_key: Vec<u64>,
}

struct TrieWalkScratch {
    stack_states: Vec<StackState>,
    stack_state_offsets: Vec<usize>,
    stack_returns: Vec<(u16, StackState)>,
    stack_return_offsets: Vec<usize>,
    active_prefix: Vec<u8>,
    queue_buf: Vec<StackState>,
    visited_buf: SmallDedup<StackState>,
    scanable_buf: Vec<StackState>,
    returns_buf: Vec<(u16, StackState)>,
    dfa_stack: Vec<u16>,
    dfa_active_prefix: Vec<u8>,
}

impl TrieWalkScratch {
    fn new() -> Self {
        Self {
            stack_states: Vec::new(),
            stack_state_offsets: Vec::new(),
            stack_returns: Vec::new(),
            stack_return_offsets: Vec::new(),
            active_prefix: Vec::new(),
            queue_buf: Vec::new(),
            visited_buf: SmallDedup::new(),
            scanable_buf: Vec::new(),
            returns_buf: Vec::new(),
            dfa_stack: Vec::new(),
            dfa_active_prefix: Vec::new(),
        }
    }
}

impl GrammarMatcher {
    pub fn new(
        grammar: Arc<Grammar>,
        tokenizer_info: Arc<Tokenizer>,
        stop_token_ids: Vec<u32>,
        max_rollback_tokens: usize,
    ) -> Self {
        let compiled = Arc::new(CompiledGrammar::new(&grammar, &tokenizer_info));
        Self::with_compiled(compiled, stop_token_ids, max_rollback_tokens)
    }

    pub fn with_compiled(
        compiled: Arc<CompiledGrammar>,
        stop_token_ids: Vec<u32>,
        max_rollback_tokens: usize,
    ) -> Self {
        let tokenizer_info = compiled.tokenizer.clone();
        let parser = StackParser::new(compiled.clone());
        let engine = if compiled.is_single_dfa
            && parser.current_states().len() == 1
            && parser.current_returns().is_empty()
        {
            let rule_idx = compiled.grammar.root_rule().0 as usize;
            let initial_state = parser.current_states()[0].dfa_state;
            ParserEngine::SingleDfa(SingleDfaEngine::new(rule_idx, initial_state))
        } else {
            ParserEngine::Stack(parser)
        };
        let vocab_size = tokenizer_info.vocab_size();
        Self {
            engine,
            compiled,
            tokenizer: tokenizer_info,
            stop_token_ids,
            token_length_history: VecDeque::new(),
            terminated: false,
            max_rollback_tokens,
            trie_scratch: TrieWalkScratch::new(),
            bitmask_scratch: vec![0u32; bitmask::bitmask_size(vocab_size)],
            bitmask_cache_key: Vec::new(),
        }
    }

    pub fn accept_token(&mut self, token_id: u32) -> bool {
        if self.terminated {
            return false;
        }

        if self.stop_token_ids.contains(&token_id) {
            if self.can_terminate() {
                self.terminated = true;
                return true;
            }
            return false;
        }

        if self
            .tokenizer
            .special_token_ids()
            .binary_search(&token_id)
            .is_ok()
        {
            return false;
        }

        let Some(decoded) = self.tokenizer.decoded_token_bytes(token_id) else {
            return false;
        };

        let ok = match &mut self.engine {
            ParserEngine::SingleDfa(e) => e.advance_bytes(&self.compiled, decoded),
            ParserEngine::Stack(p) => p.advance_bytes(decoded),
        };
        if !ok {
            return false;
        }

        self.push_token_history(decoded.len());
        true
    }

    pub fn accept_string(&mut self, s: &str) -> bool {
        if self.terminated {
            return false;
        }

        let ok = match &mut self.engine {
            ParserEngine::SingleDfa(e) => e.advance_bytes(&self.compiled, s.as_bytes()),
            ParserEngine::Stack(p) => p.advance_bytes(s.as_bytes()),
        };
        if !ok {
            return false;
        }

        self.push_token_history(s.len());
        true
    }

    fn push_token_history(&mut self, len: usize) {
        self.token_length_history.push_back(len);
        while self.token_length_history.len() > self.max_rollback_tokens {
            self.token_length_history.pop_front();
            if let ParserEngine::SingleDfa(e) = &mut self.engine {
                e.history.pop_front();
            }
        }
    }

    pub fn fill_next_token_bitmask(&mut self, bitmask: &mut [u32]) {
        bitmask::clear_bitmask(bitmask);

        if self.terminated {
            return;
        }

        match &self.engine {
            ParserEngine::SingleDfa(e) => {
                e.fill_bitmask(
                    &self.compiled,
                    &self.tokenizer,
                    bitmask,
                    &mut self.trie_scratch.dfa_stack,
                    &mut self.trie_scratch.dfa_active_prefix,
                    &mut self.bitmask_cache_key,
                );
            }
            ParserEngine::Stack(_) => {
                self.fill_bitmask_stack(bitmask);
            }
        }

        if !self.stop_token_ids.is_empty() {
            let can_terminate = self.can_terminate();
            let vocab_size = self.tokenizer.vocab_size();
            for &stop_id in &self.stop_token_ids {
                if (stop_id as usize) < vocab_size {
                    if can_terminate {
                        set_bit(bitmask, stop_id as usize);
                    } else {
                        clear_bit(bitmask, stop_id as usize);
                    }
                }
            }
        }
    }

    pub fn fill_next_token_mask(&mut self) -> Vec<u32> {
        let mut scratch = std::mem::take(&mut self.bitmask_scratch);
        self.fill_next_token_bitmask(&mut scratch);
        let mask = scratch.clone();
        self.bitmask_scratch = scratch;
        mask
    }

    fn fill_bitmask_stack(&mut self, bitmask: &mut [u32]) {
        let parser = match &self.engine {
            ParserEngine::Stack(p) => p,
            _ => unreachable!(),
        };

        parser.write_cache_key(&mut self.bitmask_cache_key);
        if self
            .compiled
            .get_cached_bitmask(&self.bitmask_cache_key, bitmask)
        {
            return;
        }

        let current_states = parser.current_states();
        let mut seen_keys = [(0u32, 0u32); 16];
        let mut seen_count = 0usize;
        let mut need_trie_walk = false;

        for state in current_states {
            let dfa_key = (state.rule_id as u32, state.dfa_state as u32);
            let already_seen = seen_keys[..seen_count].contains(&dfa_key);
            if !already_seen {
                if seen_count < seen_keys.len() {
                    seen_keys[seen_count] = dfa_key;
                    seen_count += 1;
                }
                if let Some(mask) = self.compiled.token_masks.get(&dfa_key) {
                    for (j, &word) in mask.accepted_mask.iter().enumerate() {
                        if j < bitmask.len() {
                            bitmask[j] |= word;
                        }
                    }
                    if !mask.uncertain_tokens.is_empty() {
                        need_trie_walk = true;
                    }
                }
            }
        }

        if !need_trie_walk {
            self.compiled
                .cache_bitmask(&self.bitmask_cache_key, bitmask);
            return;
        }

        self.fill_bitmask_trie_walk(bitmask);

        self.compiled
            .cache_bitmask(&self.bitmask_cache_key, bitmask);
    }

    fn fill_bitmask_trie_walk(&mut self, bitmask: &mut [u32]) {
        let parser = match &self.engine {
            ParserEngine::Stack(p) => p,
            _ => unreachable!(),
        };

        let sorted = self.tokenizer.sorted_token_ids();
        let trie_end = self.tokenizer.trie_subtree_end();

        let s = &mut self.trie_scratch;
        s.stack_states.clear();
        s.stack_state_offsets.clear();
        s.stack_returns.clear();
        s.stack_return_offsets.clear();
        s.active_prefix.clear();

        s.stack_state_offsets.push(0);
        s.stack_states.extend_from_slice(parser.current_states());
        s.stack_return_offsets.push(0);
        s.stack_returns.extend_from_slice(parser.current_returns());

        let mut i = 0;
        while i < sorted.len() {
            let token_id = sorted[i];
            let bytes = self
                .tokenizer
                .decoded_token_bytes(token_id)
                .expect("sorted token IDs have decoded bytes");

            if bitmask::get_bit(bitmask, token_id as usize) {
                i += 1;
                continue;
            }

            let s = &mut self.trie_scratch;
            let common = longest_common_prefix(bytes, &s.active_prefix);
            if common < s.active_prefix.len() {
                let depth = common + 1;
                if depth < s.stack_state_offsets.len() {
                    let s_end = s.stack_state_offsets[depth];
                    s.stack_states.truncate(s_end);
                    s.stack_state_offsets.truncate(depth);
                    let r_end = s.stack_return_offsets[depth];
                    s.stack_returns.truncate(r_end);
                    s.stack_return_offsets.truncate(depth);
                }
                s.active_prefix.truncate(common);
            }

            let parser = match &self.engine {
                ParserEngine::Stack(p) => p,
                _ => unreachable!(),
            };
            let s = &mut self.trie_scratch;
            let mut dead = false;
            for &byte in &bytes[common..] {
                let s_start = *s.stack_state_offsets.last().unwrap();
                let r_start = *s.stack_return_offsets.last().unwrap();
                let states = &s.stack_states[s_start..];
                let rets = &s.stack_returns[r_start..];

                if parser.probe_advance_reuse(
                    states,
                    rets,
                    byte,
                    &mut s.queue_buf,
                    &mut s.visited_buf,
                    &mut s.scanable_buf,
                    &mut s.returns_buf,
                ) {
                    s.stack_state_offsets.push(s.stack_states.len());
                    s.stack_states.extend_from_slice(&s.scanable_buf);
                    s.stack_return_offsets.push(s.stack_returns.len());
                    s.stack_returns.extend_from_slice(&s.returns_buf);
                    s.active_prefix.push(byte);
                } else {
                    if s.active_prefix.is_empty() {
                        i = trie_end[i];
                    } else {
                        i += 1;
                    }
                    dead = true;
                    break;
                }
            }

            if !dead {
                set_bit(bitmask, token_id as usize);
                i += 1;
            }
        }
    }

    pub fn fork(&self) -> Self {
        Self {
            engine: self.engine.clone(),
            compiled: self.compiled.clone(),
            tokenizer: self.tokenizer.clone(),
            stop_token_ids: self.stop_token_ids.clone(),
            token_length_history: self.token_length_history.clone(),
            terminated: self.terminated,
            max_rollback_tokens: self.max_rollback_tokens,
            trie_scratch: TrieWalkScratch::new(),
            bitmask_scratch: vec![0u32; self.bitmask_scratch.len()],
            bitmask_cache_key: Vec::new(),
        }
    }

    pub fn rollback_capacity(&self) -> usize {
        self.token_length_history.len()
    }

    pub fn rollback(&mut self, mut num_tokens: usize) {
        if num_tokens == 0 {
            return;
        }
        if self.terminated {
            self.terminated = false;
            num_tokens -= 1;
            if num_tokens == 0 {
                return;
            }
        }

        match &mut self.engine {
            ParserEngine::SingleDfa(e) => {
                let n = e.rollback(num_tokens);
                for _ in 0..n {
                    self.token_length_history.pop_back();
                }
            }
            ParserEngine::Stack(p) => {
                let n = num_tokens.min(self.token_length_history.len());
                for _ in 0..n {
                    if let Some(len) = self.token_length_history.pop_back() {
                        p.pop_last_states(len);
                    }
                }
            }
        }
    }

    pub fn find_jump_forward_string(&mut self) -> String {
        if self.terminated {
            return String::new();
        }

        match &mut self.engine {
            ParserEngine::SingleDfa(e) => e.find_jump_forward(&self.compiled),
            ParserEngine::Stack(p) => {
                if p.is_completed() {
                    return String::new();
                }

                let mut result = Vec::new();
                let start_pos = p.position();

                loop {
                    if p.is_completed() {
                        break;
                    }

                    let states = p.current_states().to_vec();
                    if states.is_empty() {
                        break;
                    }

                    let mut next_byte: Option<u8> = None;
                    let mut conflict = false;

                    for state in &states {
                        let flags = self.compiled.action(state.rule_id, state.dfa_state).flags;

                        if flags.has_rule_ref() || flags.is_accepting() {
                            conflict = true;
                            break;
                        }

                        let dfa = &self.compiled.rule_dfas[state.rule_id as usize];
                        let edges = dfa.fsm.edges(StateId(state.dfa_state as u32));
                        let state_byte = deterministic_byte(edges);

                        if state_byte.is_none() {
                            conflict = true;
                            break;
                        }

                        match next_byte {
                            None => next_byte = state_byte,
                            Some(b) if Some(b) == state_byte => {}
                            _ => {
                                conflict = true;
                                break;
                            }
                        }
                    }

                    if conflict || next_byte.is_none() {
                        break;
                    }

                    let byte = next_byte.unwrap();
                    if !p.advance(byte) {
                        break;
                    }
                    result.push(byte);
                }

                let advanced = p.position() - start_pos;
                p.pop_last_states(advanced);

                String::from_utf8_lossy(&result).to_string()
            }
        }
    }

    pub fn is_terminated(&self) -> bool {
        self.terminated
    }

    pub fn can_terminate(&self) -> bool {
        match &self.engine {
            ParserEngine::SingleDfa(e) => e.is_completed(&self.compiled),
            ParserEngine::Stack(p) => p.is_completed(),
        }
    }

    pub fn reset(&mut self) {
        match &mut self.engine {
            ParserEngine::SingleDfa(e) => e.reset(&self.compiled),
            ParserEngine::Stack(p) => p.reset(),
        }
        self.token_length_history.clear();
        self.terminated = false;
    }
}

fn longest_common_prefix(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

fn deterministic_byte(edges: &[FsmEdge]) -> Option<u8> {
    let mut result = None;
    for edge in edges {
        if let FsmEdge::CharRange { min, max, .. } = edge {
            if min != max {
                return None;
            }
            match result {
                None => result = Some(*min),
                Some(b) if b == *min => {}
                _ => return None,
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::grammar::Grammar;

    #[test]
    fn matcher_every_case() {
        test_star_quantifier();
        test_plus_quantifier();
        test_question_quantifier();
    }

    fn test_star_quantifier() {
        let ebnf = r#"root ::= "a"*"#;
        let grammar = Arc::new(Grammar::from_ebnf(ebnf, "root").unwrap());
        let vocab: Vec<String> = vec!["a".into(), "b".into()];
        let tok = Arc::new(Tokenizer::from_vocab(&vocab));

        let mut m = GrammarMatcher::new(grammar, tok, vec![], 10);
        assert!(m.can_terminate());
        assert!(m.accept_string("a"));
        assert!(m.can_terminate());
        assert!(m.accept_string("aa"));
        assert!(m.can_terminate());
    }

    fn test_plus_quantifier() {
        let ebnf = r#"root ::= "a"+"#;
        let grammar = Arc::new(Grammar::from_ebnf(ebnf, "root").unwrap());
        let vocab: Vec<String> = vec!["a".into()];
        let tok = Arc::new(Tokenizer::from_vocab(&vocab));

        let mut m = GrammarMatcher::new(grammar, tok, vec![], 10);
        assert!(!m.can_terminate());
        assert!(m.accept_string("a"));
        assert!(m.can_terminate());
    }

    fn test_question_quantifier() {
        let ebnf = r#"root ::= "a"?"#;
        let grammar = Arc::new(Grammar::from_ebnf(ebnf, "root").unwrap());
        let vocab: Vec<String> = vec!["a".into()];
        let tok = Arc::new(Tokenizer::from_vocab(&vocab));

        let mut m = GrammarMatcher::new(grammar, tok, vec![], 10);
        assert!(m.can_terminate());
        assert!(m.accept_string("a"));
        assert!(m.can_terminate());
    }
}
