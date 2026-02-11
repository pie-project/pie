//! Grammar matcher: DFA-based pushdown automaton + token acceptance + bitmask generation.
//!
//! `GrammarMatcher` is the main runtime entry point. It:
//! - Accepts tokens/strings byte-by-byte via a DFA-based stack parser
//! - Generates token bitmasks (which tokens are valid next)
//! - Supports rollback for speculative decoding
//! - Supports jump-forward decoding (finding deterministic prefixes)

mod single_dfa;
mod stack_parser;

use std::collections::VecDeque;
use std::sync::Arc;

use crate::structured::bitmask::{self, set_bit};
use crate::structured::compiled_grammar::CompiledGrammar;
use crate::structured::fsm::{FsmEdge, StateId};
use crate::structured::grammar::Grammar;
use crate::structured::tokenizer::TokenizerInfo;

use single_dfa::SingleDfaEngine;
use stack_parser::{SmallDedup, StackParser, StackState};

// ---------------------------------------------------------------------------
// Parser Engine
// ---------------------------------------------------------------------------

/// Two-variant engine that eliminates all `if single_dfa_mode` dual-path code.
enum ParserEngine {
    /// Single-DFA fast path: raw byte_table lookups (~2ns/byte).
    SingleDfa(SingleDfaEngine),
    /// Stack parser: DFA-based pushdown automaton for multi-rule grammars.
    Stack(StackParser),
}

// ---------------------------------------------------------------------------
// Grammar Matcher
// ---------------------------------------------------------------------------

/// Grammar-guided token matcher.
///
/// Wraps a parser engine with token-level operations: accept/reject
/// tokens, generate next-token bitmasks, rollback, and jump-forward decoding.
pub struct GrammarMatcher {
    engine: ParserEngine,
    compiled: Arc<CompiledGrammar>,
    tokenizer_info: Arc<TokenizerInfo>,
    /// Token IDs that signal end of generation.
    stop_token_ids: Vec<u32>,
    /// Length of each accepted token (in bytes), for rollback.
    token_length_history: VecDeque<usize>,
    /// Whether a stop token has been accepted.
    terminated: bool,
    /// Maximum number of tokens that can be rolled back.
    max_rollback_tokens: usize,
    /// Reusable scratch buffers for trie walk (avoids per-call heap allocations).
    trie_scratch: TrieWalkScratch,
}

/// Reusable scratch buffers for the trie walk in `fill_next_token_bitmask`.
struct TrieWalkScratch {
    // Stack parser trie walk arenas
    stack_states: Vec<StackState>,
    stack_state_offsets: Vec<usize>,
    stack_returns: Vec<(u16, StackState)>,
    stack_return_offsets: Vec<usize>,
    active_prefix: Vec<u8>,
    queue_buf: Vec<StackState>,
    visited_buf: SmallDedup<StackState>,
    scanable_buf: Vec<StackState>,
    returns_buf: Vec<(u16, StackState)>,
    // Single-DFA trie walk
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
    /// Create a new grammar matcher.
    pub fn new(
        grammar: Arc<Grammar>,
        tokenizer_info: Arc<TokenizerInfo>,
        stop_token_ids: Vec<u32>,
        max_rollback_tokens: usize,
    ) -> Self {
        let compiled = Arc::new(CompiledGrammar::new(&grammar, &tokenizer_info));
        Self::with_compiled(compiled, tokenizer_info, stop_token_ids, max_rollback_tokens)
    }

    /// Create a grammar matcher from a pre-compiled grammar.
    pub fn with_compiled(
        compiled: Arc<CompiledGrammar>,
        tokenizer_info: Arc<TokenizerInfo>,
        stop_token_ids: Vec<u32>,
        max_rollback_tokens: usize,
    ) -> Self {
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
        Self {
            engine,
            compiled,
            tokenizer_info,
            stop_token_ids,
            token_length_history: VecDeque::new(),
            terminated: false,
            max_rollback_tokens,
            trie_scratch: TrieWalkScratch::new(),
        }
    }

    /// Accept a token by its ID. Returns true if the token was valid.
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

        if self.tokenizer_info.special_token_ids().contains(&token_id) {
            return false;
        }

        let decoded = match self.tokenizer_info.decode_token(token_id) {
            Some(s) => s.to_string(),
            None => return false,
        };

        if decoded.is_empty() {
            return false;
        }

        let ok = match &mut self.engine {
            ParserEngine::SingleDfa(e) => e.advance_bytes(&self.compiled, decoded.as_bytes()),
            ParserEngine::Stack(p) => p.advance_bytes(decoded.as_bytes()),
        };
        if !ok {
            return false;
        }

        self.push_token_history(decoded.len());
        true
    }

    /// Accept a string. Returns true if the entire string was valid.
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

    /// Push a token length to history and trim if needed.
    fn push_token_history(&mut self, len: usize) {
        self.token_length_history.push_back(len);
        while self.token_length_history.len() > self.max_rollback_tokens {
            self.token_length_history.pop_front();
            if let ParserEngine::SingleDfa(e) = &mut self.engine {
                e.history.pop_front();
            }
        }
    }

    /// Fill the bitmask with valid next tokens.
    ///
    /// Strategy:
    /// 1. Check runtime bitmask cache.
    /// 2. DFA mask fast path: OR pre-computed accepted masks for (rule_id, dfa_state).
    /// 3. Batch trie walk for uncertain tokens.
    pub fn fill_next_token_bitmask(&mut self, bitmask: &mut [u32]) {
        bitmask::clear_bitmask(bitmask);

        if self.terminated {
            return;
        }

        let vocab_size = self.tokenizer_info.vocab_size();

        // Allow stop tokens if grammar can terminate here
        if self.can_terminate() {
            for &stop_id in &self.stop_token_ids {
                if (stop_id as usize) < vocab_size {
                    set_bit(bitmask, stop_id as usize);
                }
            }
        }

        match &self.engine {
            ParserEngine::SingleDfa(e) => {
                e.fill_bitmask(
                    &self.compiled, &self.tokenizer_info, bitmask,
                    &mut self.trie_scratch.dfa_stack,
                    &mut self.trie_scratch.dfa_active_prefix,
                );
            }
            ParserEngine::Stack(_) => {
                self.fill_bitmask_stack(bitmask);
            }
        }
    }

    /// Fill bitmask using the stack parser (multi-rule path).
    fn fill_bitmask_stack(&mut self, bitmask: &mut [u32]) {
        let parser = match &self.engine {
            ParserEngine::Stack(p) => p,
            _ => unreachable!(),
        };

        // Check runtime bitmask cache
        let state_hash = parser.state_hash();
        if self.compiled.get_cached_bitmask(state_hash, bitmask) {
            return;
        }

        // DFA mask fast path — direct (rule_id, dfa_state) lookup
        // Inline dedup for the common case of 1-8 unique DFA states (avoids FxHashSet alloc).
        let current_states = parser.current_states();
        let mut seen_keys = [(0u32, 0u32); 16];
        let mut seen_count = 0usize;
        let mut need_trie_walk = false;

        for state in current_states {
            let dfa_key = (state.rule_id as u32, state.dfa_state as u32);
            // Inline linear dedup (typical: 1-5 unique states)
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
            self.compiled.cache_bitmask(state_hash, bitmask);
            return;
        }

        // Batch trie walk for remaining tokens
        self.fill_bitmask_trie_walk(bitmask);

        self.compiled.cache_bitmask(state_hash, bitmask);
    }

    /// Batch trie walk: process sorted vocabulary tokens with shared prefix optimization.
    ///
    /// Uses flat arena storage for the trie walk stack to avoid per-probe allocations.
    /// Scratch buffers are reused across calls via `self.trie_scratch`.
    fn fill_bitmask_trie_walk(&mut self, bitmask: &mut [u32]) {
        let parser = match &self.engine {
            ParserEngine::Stack(p) => p,
            _ => unreachable!(),
        };

        let sorted = self.tokenizer_info.sorted_vocab();
        let trie_end = self.tokenizer_info.trie_subtree_end();

        // Reuse scratch buffers (clear but keep allocated capacity)
        let s = &mut self.trie_scratch;
        s.stack_states.clear();
        s.stack_state_offsets.clear();
        s.stack_returns.clear();
        s.stack_return_offsets.clear();
        s.active_prefix.clear();

        // Push initial level (current parser state)
        s.stack_state_offsets.push(0);
        s.stack_states.extend_from_slice(parser.current_states());
        s.stack_return_offsets.push(0);
        s.stack_returns.extend_from_slice(parser.current_returns());

        let mut i = 0;
        while i < sorted.len() {
            let (token_id, ref token_str) = sorted[i];
            let bytes = token_str.as_bytes();

            if bytes.is_empty() {
                i += 1;
                continue;
            }

            // Skip tokens already accepted by DFA mask
            if bitmask::get_bit(bitmask, token_id as usize) {
                i += 1;
                continue;
            }

            // Rewind stack to common prefix
            let s = &mut self.trie_scratch;
            let common = longest_common_prefix(bytes, &s.active_prefix);
            if common < s.active_prefix.len() {
                let depth = common + 1; // keep `depth` levels (0..=common)
                // Truncate arenas to the end of the `common` level
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

            // Advance through remaining bytes
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
                    states, rets, byte,
                    &mut s.queue_buf, &mut s.visited_buf,
                    &mut s.scanable_buf, &mut s.returns_buf,
                ) {
                    // Push new level
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

    /// Rollback the last `num_tokens` accepted tokens.
    pub fn rollback(&mut self, num_tokens: usize) {
        if self.terminated {
            self.terminated = false;
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

    /// Find a deterministic prefix string that all states must accept.
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

                        // If state has RuleRef edges or is accepting, not deterministic
                        if flags.has_rule_ref() || flags.is_accepting() {
                            conflict = true;
                            break;
                        }

                        // Check DFA edges for single deterministic byte
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
                            _ => { conflict = true; break; }
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

    /// Whether the matcher has accepted a stop token.
    pub fn is_terminated(&self) -> bool {
        self.terminated
    }

    /// Whether the grammar can terminate at the current position.
    pub fn can_terminate(&self) -> bool {
        match &self.engine {
            ParserEngine::SingleDfa(e) => e.is_completed(&self.compiled),
            ParserEngine::Stack(p) => p.is_completed(),
        }
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        match &mut self.engine {
            ParserEngine::SingleDfa(e) => e.reset(&self.compiled),
            ParserEngine::Stack(p) => p.reset(),
        }
        self.token_length_history.clear();
        self.terminated = false;
    }
}

/// Longest common prefix of two byte slices.
fn longest_common_prefix(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Check if a DFA state's edges have exactly one deterministic next byte.
/// Returns `Some(byte)` if all CharRange edges point to the same single byte,
/// `None` if there's a range, no char edges, or conflicting bytes.
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structured::bitmask::bitmask_size;
    use crate::structured::grammar::Grammar;
    use crate::structured::tokenizer::VocabType;

    /// Helper to build a grammar matcher with a small test vocabulary.
    fn make_matcher(ebnf: &str, root: &str, vocab: &[&str]) -> GrammarMatcher {
        let grammar = Arc::new(Grammar::from_ebnf(ebnf, root).unwrap());
        let encoded: Vec<String> = vocab.iter().map(|s| s.to_string()).collect();
        let tokenizer = Arc::new(
            TokenizerInfo::new(&encoded, VocabType::Raw, None).unwrap(),
        );
        GrammarMatcher::new(grammar, tokenizer, vec![], 10)
    }

    /// Helper to build a matcher with explicit stop tokens.
    fn make_matcher_with_stop(
        ebnf: &str,
        root: &str,
        vocab: &[&str],
        stop_ids: Vec<u32>,
    ) -> GrammarMatcher {
        let grammar = Arc::new(Grammar::from_ebnf(ebnf, root).unwrap());
        let encoded: Vec<String> = vocab.iter().map(|s| s.to_string()).collect();
        let tokenizer = Arc::new(
            TokenizerInfo::new(&encoded, VocabType::Raw, None).unwrap(),
        );
        GrammarMatcher::new(grammar, tokenizer, stop_ids, 10)
    }

    // ---- Basic accept_string tests ----

    #[test]
    fn test_accept_simple_string() {
        let grammar = Arc::new(
            Grammar::from_ebnf(r#"root ::= "hello""#, "root").unwrap(),
        );
        let vocab: Vec<String> = vec!["hello".into()];
        let tok = Arc::new(TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap());
        let mut m = GrammarMatcher::new(grammar, tok, vec![], 10);

        assert!(m.accept_string("hello"));
        assert!(m.can_terminate());
    }

    #[test]
    fn test_reject_wrong_string() {
        let grammar = Arc::new(
            Grammar::from_ebnf(r#"root ::= "hello""#, "root").unwrap(),
        );
        let vocab: Vec<String> = vec!["hello".into()];
        let tok = Arc::new(TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap());
        let mut m = GrammarMatcher::new(grammar, tok, vec![], 10);

        assert!(!m.accept_string("world"));
        // Parser should be unchanged after rejection
        assert!(m.accept_string("hello"));
    }

    #[test]
    fn test_accept_choices() {
        let grammar = Arc::new(
            Grammar::from_ebnf(r#"root ::= "yes" | "no""#, "root").unwrap(),
        );
        let vocab: Vec<String> = vec!["yes".into(), "no".into()];
        let tok = Arc::new(TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap());

        let mut m = GrammarMatcher::new(grammar.clone(), tok.clone(), vec![], 10);
        assert!(m.accept_string("yes"));
        assert!(m.can_terminate());

        let mut m = GrammarMatcher::new(grammar, tok, vec![], 10);
        assert!(m.accept_string("no"));
        assert!(m.can_terminate());
    }

    #[test]
    fn test_accept_sequence_of_rules() {
        let ebnf = r#"
            root ::= greeting " " name
            greeting ::= "hi" | "hello"
            name ::= "alice" | "bob"
        "#;
        let grammar = Arc::new(Grammar::from_ebnf(ebnf, "root").unwrap());
        let vocab: Vec<String> = vec!["test".into()];
        let tok = Arc::new(TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap());

        let mut m = GrammarMatcher::new(grammar, tok, vec![], 10);
        assert!(m.accept_string("hi alice"));
        assert!(m.can_terminate());
    }

    #[test]
    fn test_accept_char_class() {
        let ebnf = r#"root ::= [a-z]"#;
        let grammar = Arc::new(Grammar::from_ebnf(ebnf, "root").unwrap());
        let vocab: Vec<String> = vec!["a".into()];
        let tok = Arc::new(TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap());

        let mut m = GrammarMatcher::new(grammar, tok, vec![], 10);
        assert!(m.accept_string("a"));
        assert!(m.can_terminate());
    }

    #[test]
    fn test_accept_char_class_star() {
        let ebnf = r#"root ::= [a-z]*"#;
        let grammar = Arc::new(Grammar::from_ebnf(ebnf, "root").unwrap());
        let vocab: Vec<String> = vec!["abc".into()];
        let tok = Arc::new(TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap());

        let mut m = GrammarMatcher::new(grammar, tok, vec![], 10);
        assert!(m.can_terminate()); // star allows empty
        assert!(m.accept_string("abc"));
        assert!(m.can_terminate());
    }

    #[test]
    fn test_reject_char_class() {
        let ebnf = r#"root ::= [a-z]"#;
        let grammar = Arc::new(Grammar::from_ebnf(ebnf, "root").unwrap());
        let vocab: Vec<String> = vec!["A".into()];
        let tok = Arc::new(TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap());

        let mut m = GrammarMatcher::new(grammar, tok, vec![], 10);
        assert!(!m.accept_string("A"));
    }

    // ---- Token acceptance ----

    #[test]
    fn test_accept_token() {
        let mut m = make_matcher(r#"root ::= "hello world""#, "root", &["hello", " ", "world"]);
        assert!(m.accept_token(0)); // "hello"
        assert!(m.accept_token(1)); // " "
        assert!(m.accept_token(2)); // "world"
        assert!(m.can_terminate());
    }

    #[test]
    fn test_reject_token() {
        let mut m = make_matcher(r#"root ::= "hello""#, "root", &["hello", "world"]);
        assert!(!m.accept_token(1)); // "world" should fail
        assert!(m.accept_token(0)); // "hello" should still work
    }

    // ---- Bitmask tests ----

    #[test]
    fn test_bitmask_simple() {
        let mut m = make_matcher(
            r#"root ::= "ab" | "cd""#,
            "root",
            &["ab", "cd", "ef"],
        );

        let mut bm = vec![0u32; bitmask_size(3)];
        m.fill_next_token_bitmask(&mut bm);

        assert!(bitmask::get_bit(&bm, 0)); // "ab" allowed
        assert!(bitmask::get_bit(&bm, 1)); // "cd" allowed
        assert!(!bitmask::get_bit(&bm, 2)); // "ef" not allowed
    }

    #[test]
    fn test_bitmask_after_partial() {
        let mut m = make_matcher(
            r#"root ::= "abc""#,
            "root",
            &["a", "ab", "abc", "b", "bc", "c"],
        );

        // Initially, only tokens starting with "a" should be valid
        let mut bm = vec![0u32; bitmask_size(6)];
        m.fill_next_token_bitmask(&mut bm);

        assert!(bitmask::get_bit(&bm, 0)); // "a"
        assert!(bitmask::get_bit(&bm, 1)); // "ab"
        assert!(bitmask::get_bit(&bm, 2)); // "abc"
        assert!(!bitmask::get_bit(&bm, 3)); // "b"
        assert!(!bitmask::get_bit(&bm, 4)); // "bc"
        assert!(!bitmask::get_bit(&bm, 5)); // "c"

        // After accepting "a", tokens continuing "bc" should be valid
        m.accept_token(0); // "a"
        m.fill_next_token_bitmask(&mut bm);

        assert!(!bitmask::get_bit(&bm, 0)); // "a" - would give "aa"
        assert!(!bitmask::get_bit(&bm, 1)); // "ab" - would give "aab"
        assert!(!bitmask::get_bit(&bm, 2)); // "abc" - would give "aabc"
        assert!(bitmask::get_bit(&bm, 3)); // "b" - gives "ab"
        assert!(bitmask::get_bit(&bm, 4)); // "bc" - gives "abc"
        assert!(!bitmask::get_bit(&bm, 5)); // "c" - gives "ac"
    }

    #[test]
    fn test_bitmask_with_stop_tokens() {
        let mut m = make_matcher_with_stop(
            r#"root ::= "a" | "ab""#,
            "root",
            &["a", "ab", "b", "<eos>"],
            vec![3], // <eos> is stop token
        );

        let mut bm = vec![0u32; bitmask_size(4)];
        m.fill_next_token_bitmask(&mut bm);

        assert!(bitmask::get_bit(&bm, 0)); // "a"
        assert!(bitmask::get_bit(&bm, 1)); // "ab"
        assert!(!bitmask::get_bit(&bm, 2)); // "b"
        assert!(!bitmask::get_bit(&bm, 3)); // <eos> not yet (grammar not complete)

        // After "a", grammar can terminate
        m.accept_token(0);
        m.fill_next_token_bitmask(&mut bm);

        assert!(!bitmask::get_bit(&bm, 0)); // "a" - no more input expected
        assert!(!bitmask::get_bit(&bm, 1)); // "ab" - no
        assert!(bitmask::get_bit(&bm, 2)); // "b" - completes "ab" alternative
        assert!(bitmask::get_bit(&bm, 3)); // <eos> - can stop (grammar complete from "a")
    }

    // ---- Rollback tests ----

    #[test]
    fn test_rollback() {
        let mut m = make_matcher(
            r#"root ::= "abc""#,
            "root",
            &["a", "b", "c"],
        );

        assert!(m.accept_token(0)); // "a"
        assert!(m.accept_token(1)); // "b"

        m.rollback(1); // undo "b"

        assert!(m.accept_token(1)); // "b" again
        assert!(m.accept_token(2)); // "c"
        assert!(m.can_terminate());
    }

    #[test]
    fn test_rollback_multiple() {
        let mut m = make_matcher(
            r#"root ::= "abcd""#,
            "root",
            &["a", "b", "c", "d"],
        );

        assert!(m.accept_token(0)); // "a"
        assert!(m.accept_token(1)); // "b"
        assert!(m.accept_token(2)); // "c"

        m.rollback(2); // undo "c" and "b"

        assert!(m.accept_token(1)); // "b"
        assert!(m.accept_token(2)); // "c"
        assert!(m.accept_token(3)); // "d"
        assert!(m.can_terminate());
    }

    // ---- Jump forward tests ----

    #[test]
    fn test_jump_forward_simple() {
        let mut m = make_matcher(
            r#"root ::= "hello""#,
            "root",
            &["hello"],
        );

        let jf = m.find_jump_forward_string();
        assert_eq!(jf, "hello");
    }

    #[test]
    fn test_jump_forward_partial() {
        let mut m = make_matcher(
            r#"root ::= "prefix" ("a" | "b")"#,
            "root",
            &["prefix", "a", "b"],
        );

        let jf = m.find_jump_forward_string();
        assert_eq!(jf, "prefix");
    }

    #[test]
    fn test_jump_forward_after_accept() {
        let mut m = make_matcher(
            r#"root ::= "ab" "cd""#,
            "root",
            &["ab", "cd"],
        );

        m.accept_token(0); // "ab"
        let jf = m.find_jump_forward_string();
        assert_eq!(jf, "cd");
    }

    // ---- Repetition tests ----

    #[test]
    fn test_star_quantifier() {
        let ebnf = r#"root ::= "a"*"#;
        let grammar = Arc::new(Grammar::from_ebnf(ebnf, "root").unwrap());
        let vocab: Vec<String> = vec!["a".into(), "b".into()];
        let tok = Arc::new(TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap());

        let mut m = GrammarMatcher::new(grammar, tok, vec![], 10);
        assert!(m.can_terminate()); // * allows empty
        assert!(m.accept_string("a"));
        assert!(m.can_terminate());
        assert!(m.accept_string("aa"));
        assert!(m.can_terminate());
    }

    #[test]
    fn test_plus_quantifier() {
        let ebnf = r#"root ::= "a"+"#;
        let grammar = Arc::new(Grammar::from_ebnf(ebnf, "root").unwrap());
        let vocab: Vec<String> = vec!["a".into()];
        let tok = Arc::new(TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap());

        let mut m = GrammarMatcher::new(grammar, tok, vec![], 10);
        assert!(!m.can_terminate()); // + requires at least one
        assert!(m.accept_string("a"));
        assert!(m.can_terminate());
    }

    #[test]
    fn test_question_quantifier() {
        let ebnf = r#"root ::= "a"?"#;
        let grammar = Arc::new(Grammar::from_ebnf(ebnf, "root").unwrap());
        let vocab: Vec<String> = vec!["a".into()];
        let tok = Arc::new(TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap());

        let mut m = GrammarMatcher::new(grammar, tok, vec![], 10);
        assert!(m.can_terminate()); // ? allows empty
        assert!(m.accept_string("a"));
        assert!(m.can_terminate());
    }

    // ---- Unicode tests ----

    #[test]
    fn test_unicode_char_class() {
        // [à-ÿ] = U+00E0 to U+00FF (2-byte UTF-8)
        let ebnf = r#"root ::= [\u00e0-\u00ff]"#;
        let grammar = Arc::new(Grammar::from_ebnf(ebnf, "root").unwrap());
        let vocab: Vec<String> = vec!["a".into()];
        let tok = Arc::new(TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap());

        let mut m = GrammarMatcher::new(grammar, tok, vec![], 10);
        // 'à' is U+00E0, UTF-8: 0xC3 0xA0
        assert!(m.accept_string("\u{00e0}"));
        assert!(m.can_terminate());
    }

    #[test]
    fn test_unicode_reject() {
        let ebnf = r#"root ::= [\u00e0-\u00ff]"#;
        let grammar = Arc::new(Grammar::from_ebnf(ebnf, "root").unwrap());
        let vocab: Vec<String> = vec!["a".into()];
        let tok = Arc::new(TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap());

        let mut m = GrammarMatcher::new(grammar, tok, vec![], 10);
        // 'a' is U+0061, should not match
        assert!(!m.accept_string("a"));
    }

    #[test]
    fn test_unicode_3byte() {
        // CJK character range (3-byte UTF-8)
        let ebnf = r#"root ::= [\u4e00-\u9fff]+"#;
        let grammar = Arc::new(Grammar::from_ebnf(ebnf, "root").unwrap());
        let vocab: Vec<String> = vec!["a".into()];
        let tok = Arc::new(TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap());

        let mut m = GrammarMatcher::new(grammar, tok, vec![], 10);
        assert!(m.accept_string("\u{4e00}")); // 一 (CJK first)
        assert!(m.accept_string("\u{9fff}")); // last in range
        assert!(m.can_terminate());

        m.reset();
        assert!(!m.accept_string("a")); // ASCII rejected
    }

    #[test]
    fn test_unicode_4byte() {
        // Emoji range (4-byte UTF-8)
        let ebnf = r#"root ::= [\U0001f600-\U0001f64f]"#;
        let grammar = Arc::new(Grammar::from_ebnf(ebnf, "root").unwrap());
        let vocab: Vec<String> = vec!["a".into()];
        let tok = Arc::new(TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap());

        let mut m = GrammarMatcher::new(grammar, tok, vec![], 10);
        // 😀 is U+1F600, UTF-8: F0 9F 98 80
        assert!(m.accept_string("\u{1f600}"));
        assert!(m.can_terminate());
    }

    #[test]
    fn test_unicode_mixed_byte_lengths() {
        // Range spanning 1-byte and 2-byte UTF-8
        let ebnf = r#"root ::= [\u0041-\u00ff]+"#;
        let grammar = Arc::new(Grammar::from_ebnf(ebnf, "root").unwrap());
        let vocab: Vec<String> = vec!["a".into()];
        let tok = Arc::new(TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap());

        let mut m = GrammarMatcher::new(grammar, tok, vec![], 10);
        assert!(m.accept_string("A")); // U+0041 (1-byte)
        assert!(m.accept_string("z")); // U+007A (1-byte)
        assert!(m.accept_string("\u{00e0}")); // U+00E0 (2-byte)
        assert!(m.can_terminate());

        m.reset();
        assert!(!m.accept_string("@")); // U+0040, below range
    }

    // ---- Complex grammar tests ----

    #[test]
    fn test_json_schema_smoke() {
        use crate::structured::json_schema::{json_schema_to_grammar, JsonSchemaOptions};
        let schema = r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}"#;
        let opts = JsonSchemaOptions { any_whitespace: false, ..JsonSchemaOptions::default() };
        let grammar = Arc::new(json_schema_to_grammar(schema, &opts).unwrap());

        let mut vocab_strs: Vec<String> = Vec::new();
        let structural = ["{", "}", "[", "]", ",", ":", "\"", "\\", "true", "false", "null",
            "\\n", "\\t", "\\r", "\\\\", "\\\"",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-", ".", "e", "E", "+"];
        for s in &structural { vocab_strs.push(s.to_string()); }
        for c in 32u8..=126 { let s = String::from(c as char); if !vocab_strs.contains(&s) { vocab_strs.push(s); } }
        while vocab_strs.len() < 1000 { vocab_strs.push(format!("tok_{}", vocab_strs.len())); }
        vocab_strs.truncate(1000);

        let tok = Arc::new(TokenizerInfo::new(&vocab_strs, VocabType::Raw, None).unwrap());

        let mut m = GrammarMatcher::new(grammar, tok, vec![], 10);
        assert!(m.accept_string(r#"{"name":"test","age":42}"#));
        assert!(m.can_terminate());
    }

    #[test]
    fn test_json_number() {
        let ebnf = r#"
            root ::= integer
            integer ::= [0-9]+
        "#;
        let grammar = Arc::new(Grammar::from_ebnf(ebnf, "root").unwrap());
        let vocab: Vec<String> = vec!["1".into(), "23".into(), "a".into()];
        let tok = Arc::new(TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap());

        let mut m = GrammarMatcher::new(grammar, tok, vec![], 10);
        assert!(m.accept_string("123"));
        assert!(m.can_terminate());
    }

    #[test]
    fn test_multiple_rules_with_bitmask() {
        let ebnf = r#"
            root ::= "true" | "false" | "null"
        "#;
        let mut m = make_matcher(
            ebnf,
            "root",
            &["t", "tr", "true", "f", "fa", "false", "n", "nu", "null", "x"],
        );

        let mut bm = vec![0u32; bitmask_size(10)];
        m.fill_next_token_bitmask(&mut bm);

        assert!(bitmask::get_bit(&bm, 0)); // "t"
        assert!(bitmask::get_bit(&bm, 1)); // "tr"
        assert!(bitmask::get_bit(&bm, 2)); // "true"
        assert!(bitmask::get_bit(&bm, 3)); // "f"
        assert!(bitmask::get_bit(&bm, 4)); // "fa"
        assert!(bitmask::get_bit(&bm, 5)); // "false"
        assert!(bitmask::get_bit(&bm, 6)); // "n"
        assert!(bitmask::get_bit(&bm, 7)); // "nu"
        assert!(bitmask::get_bit(&bm, 8)); // "null"
        assert!(!bitmask::get_bit(&bm, 9)); // "x"
    }

    #[test]
    fn test_nested_group_normalization() {
        let ebnf = r#"root ::= "a" ("b" | "c") "d""#;
        let grammar = Arc::new(Grammar::from_ebnf(ebnf, "root").unwrap());
        let vocab: Vec<String> = vec!["a".into()];
        let tok = Arc::new(TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap());

        let mut m = GrammarMatcher::new(grammar, tok, vec![], 10);
        assert!(m.accept_string("abd"));
        assert!(m.can_terminate());

        m.reset();
        assert!(m.accept_string("acd"));
        assert!(m.can_terminate());

        m.reset();
        assert!(!m.accept_string("aed"));
    }

    #[test]
    fn test_steady_state_string_content() {
        // Test that steady state activates for CharacterClassStar content
        let ebnf = r#"root ::= "\"" [^"\\]* "\""  "#;
        let grammar = Arc::new(Grammar::from_ebnf(ebnf, "root").unwrap());
        let vocab: Vec<String> = vec!["a".into()];
        let tok = Arc::new(TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap());

        let mut m = GrammarMatcher::new(grammar, tok, vec![], 10);
        // Parse opening quote
        assert!(m.accept_string("\""));
        // Parse string content — after 2 bytes, steady state should activate
        assert!(m.accept_string("ABCDEFGHIJ"));
        // Verify the parser correctly tracks position with lazy steady count
        assert!(m.accept_string("\""));
        assert!(m.can_terminate());
    }

    #[test]
    fn test_steady_state_rollback() {
        // Verify rollback works correctly with lazy steady-state bytes
        let ebnf = r#"root ::= "\"" [^"\\]* "\""  "#;
        let grammar = Arc::new(Grammar::from_ebnf(ebnf, "root").unwrap());
        let vocab: Vec<String> = vec!["a".into()];
        let tok = Arc::new(TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap());

        let mut m = GrammarMatcher::new(grammar, tok, vec![], 10);
        assert!(m.accept_string("\"")); // opening quote (token 0)
        assert!(m.accept_string("ABCDE")); // 5 chars in steady state (token 1)
        // Rollback the 5 chars (token 1)
        m.rollback(1);
        // After rollback, should be at position after opening quote
        assert!(m.accept_string("XYZ\""));
        assert!(m.can_terminate());
    }

    #[test]
    fn test_steady_state_json_schema() {
        // Test with actual JSON schema grammar (the primary use case)
        use crate::structured::json_schema::{json_schema_to_grammar, JsonSchemaOptions};
        let schema = r#"{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}"#;
        let opts = JsonSchemaOptions { any_whitespace: false, ..JsonSchemaOptions::default() };
        let grammar = Arc::new(json_schema_to_grammar(schema, &opts).unwrap());

        let vocab: Vec<String> = (32u8..=126).map(|c| String::from(c as char)).collect();
        let tok = Arc::new(
            TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap(),
        );

        let mut m = GrammarMatcher::new(grammar, tok, vec![], 10);
        // Parse prefix including string content with steady-state opportunity
        let long_name = "A".repeat(100);
        let input = format!(r#"{{"name":"{}"}}"#, long_name);
        assert!(m.accept_string(&input));
        assert!(m.can_terminate());
    }

    #[test]
    fn test_chain_shortcircuit_ebnf_json() {
        let ebnf = r#"
root ::= value
value ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws (pair ("," ws pair)*)? ws "}"
pair ::= ws string ws ":" ws value
array ::= "[" ws (value ("," ws value)*)? ws "]"
string ::= "\"" char* "\""
char ::= [^"\\] | "\\" escape
escape ::= "\"" | "\\" | "/" | "b" | "f" | "n" | "r" | "t" | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
number ::= integer fraction? exponent?
integer ::= "-"? ("0" | [1-9] [0-9]*)
fraction ::= "." [0-9]+
exponent ::= [eE] [+-]? [0-9]+
ws ::= [ \t\n\r]*
"#;
        let grammar = Arc::new(Grammar::from_ebnf(ebnf, "root").unwrap());
        let vocab: Vec<String> = vec!["dummy".into()];
        let tok = Arc::new(
            TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap(),
        );
        let mut m = GrammarMatcher::new(grammar, tok, vec![], 10);
        let input = r#"{"name": "John", "age": 30}"#;
        assert!(m.accept_string(input));
    }

    #[test]
    fn test_chain_shortcircuit_long_string() {
        // Verify chain short-circuit handles long strings without O(N^2) blowup
        use crate::structured::json_schema::{json_schema_to_grammar, JsonSchemaOptions};
        let schema = r#"{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}"#;
        let opts = JsonSchemaOptions { any_whitespace: false, ..JsonSchemaOptions::default() };
        let grammar = Arc::new(json_schema_to_grammar(schema, &opts).unwrap());

        let vocab: Vec<String> = (0u16..256).map(|i| String::from(i as u8 as char)).collect();
        let tok = Arc::new(
            TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap(),
        );
        let mut m = GrammarMatcher::new(grammar, tok, vec![], 10);
        // 200-char string — without chain short-circuit this would be O(N^2)
        let long_name = "A".repeat(200);
        let input = format!(r#"{{"name":"{}"}}"#, long_name);
        assert!(m.accept_string(&input));
    }
}
