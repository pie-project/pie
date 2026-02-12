//! Pre-compiled grammar with per-rule DFAs and adaptive token masks.
//!
//! `CompiledGrammar` pre-computes per-DFA-state token masks at construction time,
//! enabling O(states × V/32) `fill_next_token_bitmask` instead of O(V × bytes × states).

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::{Arc, LazyLock, Mutex, RwLock};

use lru::LruCache;
use rustc_hash::FxHasher;

use crate::structured::bitmask;
use crate::structured::fsm::{
    build_rule_fsms, DfaTable, FsmEdge, Automaton, StateId,
};
use crate::structured::grammar::Grammar;
use crate::structured::grammar::normalize::normalize_grammar;
use crate::tokenizer::Tokenizer;

// ---------------------------------------------------------------------------
// Compilation cache
// ---------------------------------------------------------------------------

/// Cache key: (grammar source string, tokenizer pointer identity).
#[derive(Hash, Eq, PartialEq)]
struct CacheKey {
    grammar_source: String,
    tokenizer_ptr: usize,
}

/// Maximum number of compiled grammars to keep in cache.
const CACHE_CAPACITY: usize = 64;

/// Global LRU cache of compiled grammars, keyed by (source, tokenizer).
static CACHE: LazyLock<Mutex<LruCache<CacheKey, Arc<CompiledGrammar>>>> =
    LazyLock::new(|| Mutex::new(LruCache::new(NonZeroUsize::new(CACHE_CAPACITY).unwrap())));

impl CompiledGrammar {
    /// Get a previously compiled grammar from cache, or compile and cache a new one.
    ///
    /// The cache is keyed by the original grammar source string and the
    /// tokenizer's `Arc` pointer identity. Two identical source strings
    /// compiled against the same tokenizer will share the same `Arc<CompiledGrammar>`.
    pub fn get_or_compile(
        source: &str,
        grammar: &Grammar,
        tokenizer: &Arc<Tokenizer>,
    ) -> Arc<Self> {
        let key = CacheKey {
            grammar_source: source.to_owned(),
            tokenizer_ptr: Arc::as_ptr(tokenizer) as usize,
        };

        let mut cache = CACHE.lock().unwrap();
        if let Some(compiled) = cache.get(&key) {
            return compiled.clone();
        }

        let compiled = Arc::new(CompiledGrammar::new(grammar, tokenizer));
        cache.put(key, compiled.clone());
        compiled
    }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Packed state flags for fast branching during advance (1 byte).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StateFlags(pub(crate) u8);

impl StateFlags {
    pub(crate) const CHAR_EDGES: u8 = 1;
    pub(crate) const RULE_REF: u8 = 2;
    pub(crate) const ACCEPTING: u8 = 4;
    /// accepting && !char_edges && !rule_ref: will immediately complete when child finishes.
    pub(crate) const PASS_THROUGH: u8 = 8;

    #[inline(always)]
    pub(crate) fn has_char_edges(self) -> bool { self.0 & Self::CHAR_EDGES != 0 }
    #[inline(always)]
    pub(crate) fn has_rule_ref(self) -> bool { self.0 & Self::RULE_REF != 0 }
    #[inline(always)]
    pub(crate) fn is_accepting(self) -> bool { self.0 & Self::ACCEPTING != 0 }
    #[inline(always)]
    pub(crate) fn is_pass_through(self) -> bool { self.0 & Self::PASS_THROUGH != 0 }
}

/// Pre-computed action for a (rule_id, dfa_state) pair.
/// Eliminates runtime DFA edge iteration in process_queue.
#[derive(Debug, Clone)]
pub(crate) struct StateAction {
    pub(crate) flags: StateFlags,
    /// Pre-extracted RuleRef edges: (predicted_rule_id, target_state_in_parent_dfa).
    pub(crate) rule_refs: Vec<(u16, u16)>,
}

/// Pre-computed token mask for a specific (rule_id, dfa_state) pair.
pub(crate) struct AdaptiveTokenMask {
    /// Bitmask of tokens that are definitely accepted from this DFA state
    /// (all bytes consumed via CharRange edges only, no rule boundaries crossed).
    pub(crate) accepted_mask: Vec<u32>,
    /// Token IDs that need runtime Earley checking (cross rule boundaries
    /// or encounter other non-deterministic situations).
    pub(crate) uncertain_tokens: Vec<u32>,
    /// Whether this DFA state has RuleRef edges. When true, the uncertain
    /// tokens include tokens that would be consumed by the referenced rules.
    /// At runtime, the Earley predict phase already expands these into
    /// leaf-level states, so the uncertain tokens from RuleRef states can
    /// be skipped (they're covered by the leaf states' masks).
    pub(crate) has_rule_ref: bool,
}

/// Token classification during pre-computation.
enum TokenClass {
    Accepted,
    Rejected,
    Uncertain,
}

/// A compiled grammar with pre-computed per-rule DFAs and token masks.
///
/// Created once per (grammar, tokenizer) pair. Shared across GrammarMatcher
/// instances via `Arc`.
pub struct CompiledGrammar {
    /// The normalized grammar.
    pub(crate) grammar: Arc<Grammar>,
    /// Per-rule DFAs (indexed by RuleId).
    pub(crate) rule_dfas: Vec<Automaton<DfaTable>>,
    /// Flat array of pre-computed state actions, indexed via `state_action_offsets`.
    pub(crate) state_actions: Vec<StateAction>,
    /// Start offset of each rule's state actions in `state_actions`.
    pub(crate) state_action_offsets: Vec<u32>,
    /// Whether any rule has self-referencing pass-through chains
    /// (needed for chain detection in complete()).
    pub(crate) has_self_ref_chains: bool,
    /// True when the root rule's DFA has no RuleRef edges at any state.
    /// When true, GrammarMatcher can bypass StackParser entirely.
    pub(crate) is_single_dfa: bool,
    /// Pre-computed token masks, keyed by (rule_id, dfa_state_id).
    pub(crate) token_masks: HashMap<(u32, u32), AdaptiveTokenMask>,
    /// Runtime bitmask cache: maps a hash of the parser state set to the
    /// fully resolved bitmask. Populated lazily during `fill_next_token_bitmask`.
    bitmask_cache: RwLock<HashMap<u64, Vec<u32>>>,
}

impl CompiledGrammar {
    /// Look up the pre-computed action for a (rule_id, dfa_state) pair.
    #[inline(always)]
    pub(crate) fn action(&self, rule_id: u16, dfa_state: u16) -> &StateAction {
        &self.state_actions[self.state_action_offsets[rule_id as usize] as usize + dfa_state as usize]
    }
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

impl CompiledGrammar {
    /// Build a compiled grammar from a grammar and tokenizer.
    ///
    /// This performs:
    /// 1. Grammar normalization
    /// 2. Per-rule NFA→DFA conversion
    /// 3. DFA state info pre-computation
    /// 4. Adaptive token mask pre-computation
    pub fn new(grammar: &Grammar, tokenizer_info: &Tokenizer) -> Self {
        let normalized = Arc::new(normalize_grammar(grammar));

        // Build per-rule NFAs and convert to DFAs
        let nfa_fsms = build_rule_fsms(&normalized);
        let rule_dfas: Vec<_> = nfa_fsms.iter()
            .map(|nfa| nfa.to_dfa().to_compact())
            .collect();

        // Pre-compute state actions (replaces old dfa_state_info)
        let (state_actions, state_action_offsets, has_self_ref_chains) =
            compute_state_actions(&rule_dfas);

        // Pre-compute token masks
        let token_masks = precompute_token_masks(
            &rule_dfas, tokenizer_info, &state_actions, &state_action_offsets,
        );

        // Detect single-DFA: root rule has no RuleRef edges at any state
        let is_single_dfa = {
            let root_id = normalized.root_rule().0 as usize;
            let root_offset = state_action_offsets[root_id] as usize;
            let root_end = state_action_offsets.get(root_id + 1).copied().unwrap_or(state_actions.len() as u32) as usize;
            state_actions[root_offset..root_end].iter().all(|a| !a.flags.has_rule_ref())
        };

        CompiledGrammar {
            grammar: normalized,
            rule_dfas,
            state_actions,
            state_action_offsets,
            has_self_ref_chains,
            is_single_dfa,
            token_masks,
            bitmask_cache: RwLock::new(HashMap::new()),
        }
    }

    /// Look up a cached bitmask by state hash. Copies directly into the output
    /// slice if found. Returns true on cache hit.
    pub(crate) fn get_cached_bitmask(&self, key: u64, bitmask: &mut [u32]) -> bool {
        let cache = self.bitmask_cache.read().unwrap_or_else(|e| e.into_inner());
        if let Some(cached) = cache.get(&key) {
            bitmask.copy_from_slice(cached);
            true
        } else {
            false
        }
    }

    /// Store a computed bitmask in the cache.
    pub(crate) fn cache_bitmask(&self, key: u64, bitmask: &[u32]) {
        let mut cache = self.bitmask_cache.write().unwrap_or_else(|e| e.into_inner());
        cache.insert(key, bitmask.to_vec());
    }
}

// ---------------------------------------------------------------------------
// DFA state info pre-computation
// ---------------------------------------------------------------------------

/// Pre-compute state actions for all (rule_id, dfa_state) pairs.
/// Returns (flat_actions, offsets_per_rule, has_self_ref_chains).
fn compute_state_actions(
    rule_dfas: &[Automaton<DfaTable>],
) -> (Vec<StateAction>, Vec<u32>, bool) {
    let mut actions = Vec::new();
    let mut offsets = Vec::with_capacity(rule_dfas.len());
    let mut has_self_ref_chains = false;

    assert!(
        rule_dfas.len() <= u16::MAX as usize,
        "too many rules ({}) — StackState.rule_id is u16 (max {})",
        rule_dfas.len(),
        u16::MAX,
    );
    for dfa in rule_dfas {
        assert!(
            dfa.fsm.num_states() <= u16::MAX as usize,
            "DFA has too many states ({}) — StackState.dfa_state is u16 (max {})",
            dfa.fsm.num_states(),
            u16::MAX,
        );
    }

    for (rule_idx, dfa) in rule_dfas.iter().enumerate() {
        offsets.push(actions.len() as u32);
        for si in 0..dfa.fsm.num_states() {
            let state = StateId(si as u32);
            let edges = dfa.fsm.edges(state);
            let has_char = edges.iter().any(|e| matches!(e, FsmEdge::CharRange { .. }));
            let has_rr = edges.iter().any(|e| matches!(e, FsmEdge::RuleRef { .. }));
            let accepting = dfa.ends.get(si).copied().unwrap_or(false);
            let pass_through = accepting && !has_char && !has_rr;

            let mut flags = 0u8;
            if has_char { flags |= StateFlags::CHAR_EDGES; }
            if has_rr { flags |= StateFlags::RULE_REF; }
            if accepting { flags |= StateFlags::ACCEPTING; }
            if pass_through { flags |= StateFlags::PASS_THROUGH; }

            // Pre-extract RuleRef edges
            let mut rule_refs = Vec::new();
            for edge in edges {
                if let FsmEdge::RuleRef { rule, target } = edge {
                    rule_refs.push((rule.0 as u16, target.0 as u16));
                    // Detect self-referencing pass-through chains
                    if rule.0 as usize == rule_idx && pass_through {
                        has_self_ref_chains = true;
                    }
                }
            }

            actions.push(StateAction {
                flags: StateFlags(flags),
                rule_refs,
            });
        }
    }

    (actions, offsets, has_self_ref_chains)
}

// ---------------------------------------------------------------------------
// Adaptive token mask pre-computation
// ---------------------------------------------------------------------------

/// Classify a token against a DFA state.
///
/// - `Accepted`: all bytes consumed via CharRange edges, staying within this rule
/// - `Rejected`: dead end with no accepting state reached and no RuleRef fallback
/// - `Uncertain`: DFA hit a dead end but an accepting state was reached earlier
///   (rule could have ended, remaining bytes need parent context), or hit a
///   RuleRef edge
///
/// Key optimization: if the DFA can CONTINUE consuming bytes after an accepting
/// state (e.g., self-looping `[^"\\]*`), we keep going instead of immediately
/// returning Uncertain. Only when the DFA actually hits a dead end do we check
/// whether an accepting state was previously seen.
fn classify_token(
    dfa: &Automaton<DfaTable>,
    start_state: StateId,
    token_bytes: &[u8],
    actions_offset: usize,
    state_actions: &[StateAction],
) -> TokenClass {
    let mut cur = start_state;
    // Track whether any accepting state was reached (including start state for * patterns)
    let mut saw_end = dfa.ends.get(start_state.0 as usize).copied().unwrap_or(false);

    for &byte in token_bytes.iter() {
        match dfa.fsm.next_state(cur, byte) {
            Some(next) => {
                cur = next;
                if dfa.ends.get(cur.0 as usize).copied().unwrap_or(false) {
                    saw_end = true;
                }
            }
            None => {
                // DFA can't consume this byte.
                // If we previously passed through an accepting state, the rule
                // could have ended there and remaining bytes go to the parent.
                if saw_end || dfa.ends.get(cur.0 as usize).copied().unwrap_or(false) {
                    return TokenClass::Uncertain;
                }
                // If RuleRef edges exist, a sub-rule might consume this byte.
                if state_actions[actions_offset + cur.0 as usize].flags.has_rule_ref() {
                    return TokenClass::Uncertain;
                }
                return TokenClass::Rejected;
            }
        }
    }

    // All bytes consumed via CharRange edges — token stays within this rule
    TokenClass::Accepted
}

/// Hash a DFA structure for deduplication.
///
/// Two DFAs with the same hash produce identical token masks.
/// Includes byte_table (all CharRange transitions), ends (accepting states),
/// and RuleRef edges (which affect uncertain token classification).
fn hash_dfa(dfa: &Automaton<DfaTable>) -> u64 {
    let mut hasher = FxHasher::default();
    dfa.start.0.hash(&mut hasher);
    dfa.ends.hash(&mut hasher);
    dfa.fsm.byte_table().hash(&mut hasher);
    // Include RuleRef edges (not captured in byte_table)
    for si in 0..dfa.fsm.num_states() {
        for edge in dfa.fsm.edges(StateId(si as u32)) {
            if let FsmEdge::RuleRef { rule, target } = edge {
                rule.0.hash(&mut hasher);
                target.0.hash(&mut hasher);
            }
        }
    }
    hasher.finish()
}

/// Pre-compute adaptive token masks for all (rule_id, dfa_state) pairs.
///
/// Deduplicates structurally identical DFAs: when two rules produce the same
/// DFA (common with JSON schema grammars where string/char rules repeat),
/// the masks are cloned instead of recomputed over the full vocabulary.
fn precompute_token_masks(
    rule_dfas: &[Automaton<DfaTable>],
    tokenizer_info: &Tokenizer,
    state_actions: &[StateAction],
    state_action_offsets: &[u32],
) -> HashMap<(u32, u32), AdaptiveTokenMask> {
    let mut masks = HashMap::new();
    let vocab_size = tokenizer_info.vocab_size();
    let bitmask_words = (vocab_size + 31) / 32;

    // Cache: DFA hash → per-state masks (keyed by state index within the DFA)
    let mut dfa_cache: HashMap<u64, Vec<AdaptiveTokenMask>> = HashMap::new();

    for (rule_idx, dfa) in rule_dfas.iter().enumerate() {
        let dfa_hash = hash_dfa(dfa);

        // Check if we already computed masks for an identical DFA
        if let Some(cached) = dfa_cache.get(&dfa_hash) {
            for (state_idx, mask) in cached.iter().enumerate() {
                masks.insert(
                    (rule_idx as u32, state_idx as u32),
                    AdaptiveTokenMask {
                        accepted_mask: mask.accepted_mask.clone(),
                        uncertain_tokens: mask.uncertain_tokens.clone(),
                        has_rule_ref: mask.has_rule_ref,
                    },
                );
            }
            continue;
        }

        // Compute masks for this DFA
        let mut rule_masks = Vec::with_capacity(dfa.fsm.num_states());

        for state_idx in 0..dfa.fsm.num_states() {
            let dfa_state = StateId(state_idx as u32);

            // Skip dead states (no outgoing edges at all)
            let edges = dfa.fsm.edges(dfa_state);
            if edges.is_empty() && !dfa.ends.get(state_idx).copied().unwrap_or(false) {
                rule_masks.push(AdaptiveTokenMask {
                    accepted_mask: vec![0u32; bitmask_words],
                    uncertain_tokens: Vec::new(),
                    has_rule_ref: false,
                });
                continue;
            }

            // Look up pre-computed flags
            let offset = state_action_offsets[rule_idx] as usize;
            let flags = state_actions[offset + state_idx].flags;
            let state_has_rule_ref = flags.has_rule_ref();
            let only_rule_ref = !flags.has_char_edges() && state_has_rule_ref;

            let mut accepted = vec![0u32; bitmask_words];
            let mut uncertain = Vec::new();

            // Use sorted vocab with trie skip for efficiency
            let sorted = tokenizer_info.sorted_vocab();
            let trie_end = tokenizer_info.trie_subtree_end();
            let mut i = 0;

            while i < sorted.len() {
                let (token_id, ref token_str) = sorted[i];
                let bytes = token_str.as_bytes();

                if bytes.is_empty() {
                    i += 1;
                    continue;
                }

                if only_rule_ref {
                    uncertain.push(token_id);
                    i += 1;
                    continue;
                }

                match classify_token(dfa, dfa_state, bytes, offset, state_actions) {
                    TokenClass::Accepted => {
                        bitmask::set_bit(&mut accepted, token_id as usize);
                        i += 1;
                    }
                    TokenClass::Rejected => {
                        if bytes.len() >= 1 && dfa.fsm.next_state(dfa_state, bytes[0]).is_none()
                            && !flags.has_rule_ref()
                        {
                            i = trie_end[i];
                        } else {
                            i += 1;
                        }
                    }
                    TokenClass::Uncertain => {
                        uncertain.push(token_id);
                        i += 1;
                    }
                }
            }

            rule_masks.push(AdaptiveTokenMask {
                accepted_mask: accepted,
                uncertain_tokens: uncertain,
                has_rule_ref: state_has_rule_ref,
            });
        }

        // Insert into result map and cache
        for (state_idx, mask) in rule_masks.iter().enumerate() {
            masks.insert(
                (rule_idx as u32, state_idx as u32),
                AdaptiveTokenMask {
                    accepted_mask: mask.accepted_mask.clone(),
                    uncertain_tokens: mask.uncertain_tokens.clone(),
                    has_rule_ref: mask.has_rule_ref,
                },
            );
        }
        dfa_cache.insert(dfa_hash, rule_masks);
    }

    masks
}
