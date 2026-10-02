use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Result, bail};
use lru::LruCache;
use rustc_hash::FxHasher;

use crate::bitmask;
use crate::compiler::GrammarLimits;
use crate::fsm::{Automaton, DfaTable, FsmEdge, StateId, build_rule_fsms};
use crate::grammar::Grammar;
use crate::grammar::normalize::normalize_grammar;
use tokenizer::Tokenizer;

const BITMASK_CACHE_BUDGET_BYTES: usize = 1024 * 1024;
const BITMASK_CACHE_MAX_ENTRIES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StateFlags(pub(crate) u8);

impl StateFlags {
    pub(crate) const CHAR_EDGES: u8 = 1;
    pub(crate) const RULE_REF: u8 = 2;
    pub(crate) const ACCEPTING: u8 = 4;
    pub(crate) const PASS_THROUGH: u8 = 8;

    #[inline(always)]
    pub(crate) fn has_char_edges(self) -> bool {
        self.0 & Self::CHAR_EDGES != 0
    }
    #[inline(always)]
    pub(crate) fn has_rule_ref(self) -> bool {
        self.0 & Self::RULE_REF != 0
    }
    #[inline(always)]
    pub(crate) fn is_accepting(self) -> bool {
        self.0 & Self::ACCEPTING != 0
    }
    #[inline(always)]
    pub(crate) fn is_pass_through(self) -> bool {
        self.0 & Self::PASS_THROUGH != 0
    }
}

#[derive(Debug, Clone)]
pub(crate) struct StateAction {
    pub(crate) flags: StateFlags,
    pub(crate) rule_refs: Vec<(u16, u16)>,
}

#[derive(Clone)]
pub(crate) struct AdaptiveTokenMask {
    pub(crate) accepted_mask: Vec<u32>,
    pub(crate) uncertain_tokens: Vec<u32>,
}

enum TokenClass {
    Accepted,
    Rejected,
    Uncertain,
}

pub struct CompiledGrammar {
    pub(crate) tokenizer: Arc<Tokenizer>,
    pub(crate) grammar: Arc<Grammar>,
    pub(crate) rule_dfas: Vec<Automaton<DfaTable>>,
    pub(crate) state_actions: Vec<StateAction>,
    pub(crate) state_action_offsets: Vec<u32>,
    pub(crate) has_self_ref_chains: bool,
    pub(crate) is_single_dfa: bool,
    pub(crate) token_masks: HashMap<(u32, u32), AdaptiveTokenMask>,
    bitmask_cache: Mutex<LruCache<Vec<u64>, Vec<u32>>>,
}

impl CompiledGrammar {
    pub fn grammar(&self) -> &Grammar {
        &self.grammar
    }

    #[inline(always)]
    pub(crate) fn action(&self, rule_id: u16, dfa_state: u16) -> &StateAction {
        &self.state_actions
            [self.state_action_offsets[rule_id as usize] as usize + dfa_state as usize]
    }
}

// `Instant::now` panics on wasm32, so the deadline is never armed there.
pub(crate) fn started_now() -> Option<Instant> {
    #[cfg(target_arch = "wasm32")]
    {
        None
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        Some(Instant::now())
    }
}

impl CompiledGrammar {
    pub fn new(grammar: &Grammar, tokenizer_info: &Arc<Tokenizer>) -> Self {
        Self::try_new(
            grammar,
            tokenizer_info,
            &GrammarLimits::default(),
            started_now(),
        )
        .expect("grammar exceeds default compilation limits")
    }

    pub(crate) fn try_new(
        grammar: &Grammar,
        tokenizer_info: &Arc<Tokenizer>,
        limits: &GrammarLimits,
        started: Option<Instant>,
    ) -> Result<Self> {
        let deadline = started.and_then(|started| started.checked_add(limits.max_compile_duration));
        check_deadline(deadline)?;
        let vocab_size = tokenizer_info.vocab_size();
        let normalized = Arc::new(normalize_grammar(grammar));

        let nfa_fsms = build_rule_fsms(&normalized);
        for nfa in &nfa_fsms {
            if nfa.fsm.num_states() > limits.max_nfa_states_per_rule {
                bail!(
                    "NFA has {} states; per-rule limit is {}",
                    nfa.fsm.num_states(),
                    limits.max_nfa_states_per_rule
                );
            }
        }
        check_deadline(deadline)?;

        let mut rule_dfas = Vec::with_capacity(nfa_fsms.len());
        let mut total_dfa_states = 0usize;
        for nfa in &nfa_fsms {
            let dfa = nfa
                .to_dfa_limited(limits.max_dfa_states_per_rule)?
                .to_compact();
            check_deadline(deadline)?;
            total_dfa_states = total_dfa_states
                .checked_add(dfa.fsm.num_states())
                .ok_or_else(|| anyhow::anyhow!("total DFA state count overflow"))?;
            if total_dfa_states > limits.max_total_dfa_states {
                bail!(
                    "grammar has {} total DFA states; limit is {}",
                    total_dfa_states,
                    limits.max_total_dfa_states
                );
            }
            rule_dfas.push(dfa);
        }
        let runtime_rules = find_runtime_rules(&normalized, &rule_dfas);
        let runtime_state_count: usize = rule_dfas
            .iter()
            .enumerate()
            .filter(|(index, _)| runtime_rules[*index])
            .map(|(_, dfa)| dfa.fsm.num_states())
            .sum();
        let mask_bytes = runtime_state_count
            .checked_mul(bitmask::bitmask_size(vocab_size))
            .and_then(|words| words.checked_mul(size_of::<u32>()))
            .ok_or_else(|| anyhow::anyhow!("token-mask memory estimate overflow"))?;
        if mask_bytes > limits.max_token_mask_bytes {
            bail!(
                "token masks require at least {} bytes; limit is {}",
                mask_bytes,
                limits.max_token_mask_bytes
            );
        }
        check_deadline(deadline)?;

        let (state_actions, state_action_offsets, has_self_ref_chains) =
            compute_state_actions(&rule_dfas);

        let token_masks = precompute_token_masks(
            &rule_dfas,
            tokenizer_info,
            &state_actions,
            &state_action_offsets,
            &runtime_rules,
            deadline,
        )?;

        let is_single_dfa = {
            let root_id = normalized.root_rule().0 as usize;
            let root_offset = state_action_offsets[root_id] as usize;
            let root_end = state_action_offsets
                .get(root_id + 1)
                .copied()
                .unwrap_or(state_actions.len() as u32) as usize;
            state_actions[root_offset..root_end]
                .iter()
                .all(|a| !a.flags.has_rule_ref())
        };

        Ok(CompiledGrammar {
            tokenizer: tokenizer_info.clone(),
            grammar: normalized,
            rule_dfas,
            state_actions,
            state_action_offsets,
            has_self_ref_chains,
            is_single_dfa,
            token_masks,
            bitmask_cache: Mutex::new(LruCache::new(bitmask_cache_capacity(vocab_size))),
        })
    }

    pub(crate) fn get_cached_bitmask(&self, key: &[u64], bitmask: &mut [u32]) -> bool {
        let mut cache = self
            .bitmask_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(cached) = cache.get(key) {
            bitmask.copy_from_slice(cached);
            true
        } else {
            false
        }
    }

    pub(crate) fn cache_bitmask(&self, key: &[u64], bitmask: &[u32]) {
        let mut cache = self
            .bitmask_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        cache.put(key.to_vec(), bitmask.to_vec());
    }
}

fn bitmask_cache_capacity(vocab_size: usize) -> NonZeroUsize {
    let mask_bytes = bitmask::bitmask_size(vocab_size)
        .saturating_mul(size_of::<u32>())
        .max(1);
    let entries = (BITMASK_CACHE_BUDGET_BYTES / mask_bytes).clamp(1, BITMASK_CACHE_MAX_ENTRIES);
    NonZeroUsize::new(entries).unwrap()
}

fn check_deadline(deadline: Option<Instant>) -> Result<()> {
    if deadline.is_some_and(|deadline| Instant::now() > deadline) {
        bail!("grammar compilation deadline exceeded");
    }
    Ok(())
}

fn compute_state_actions(rule_dfas: &[Automaton<DfaTable>]) -> (Vec<StateAction>, Vec<u32>, bool) {
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
            if has_char {
                flags |= StateFlags::CHAR_EDGES;
            }
            if has_rr {
                flags |= StateFlags::RULE_REF;
            }
            if accepting {
                flags |= StateFlags::ACCEPTING;
            }
            if pass_through {
                flags |= StateFlags::PASS_THROUGH;
            }

            let mut rule_refs = Vec::new();
            for edge in edges {
                if let FsmEdge::RuleRef { rule, target } = edge {
                    rule_refs.push((rule.0 as u16, target.0 as u16));
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

fn find_runtime_rules(grammar: &Grammar, rule_dfas: &[Automaton<DfaTable>]) -> Vec<bool> {
    let mut runtime_rules = vec![false; rule_dfas.len()];
    let root = grammar.root_rule().0 as usize;
    runtime_rules[root] = true;
    let mut queue = VecDeque::from([root]);

    while let Some(rule_index) = queue.pop_front() {
        let dfa = &rule_dfas[rule_index];
        for state_index in 0..dfa.fsm.num_states() {
            for edge in dfa.fsm.edges(StateId(state_index as u32)) {
                if let FsmEdge::RuleRef { rule, .. } = edge {
                    let referenced = rule.0 as usize;
                    if !runtime_rules[referenced] {
                        runtime_rules[referenced] = true;
                        queue.push_back(referenced);
                    }
                }
            }
        }
    }

    runtime_rules
}

fn classify_token(
    dfa: &Automaton<DfaTable>,
    start_state: StateId,
    token_bytes: &[u8],
    actions_offset: usize,
    state_actions: &[StateAction],
) -> TokenClass {
    let mut cur = start_state;
    let mut saw_end = dfa
        .ends
        .get(start_state.0 as usize)
        .copied()
        .unwrap_or(false);

    for &byte in token_bytes.iter() {
        match dfa.fsm.next_state(cur, byte) {
            Some(next) => {
                cur = next;
                if dfa.ends.get(cur.0 as usize).copied().unwrap_or(false) {
                    saw_end = true;
                }
            }
            None => {
                if saw_end || dfa.ends.get(cur.0 as usize).copied().unwrap_or(false) {
                    return TokenClass::Uncertain;
                }
                if state_actions[actions_offset + cur.0 as usize]
                    .flags
                    .has_rule_ref()
                {
                    return TokenClass::Uncertain;
                }
                return TokenClass::Rejected;
            }
        }
    }

    TokenClass::Accepted
}

fn hash_dfa(dfa: &Automaton<DfaTable>) -> u64 {
    let mut hasher = FxHasher::default();
    dfa.start.0.hash(&mut hasher);
    dfa.ends.hash(&mut hasher);
    dfa.fsm.byte_table().hash(&mut hasher);
    for si in 0..dfa.fsm.num_states() {
        dfa.fsm
            .edges(StateId(si as u32))
            .iter()
            .any(|edge| matches!(edge, FsmEdge::RuleRef { .. }))
            .hash(&mut hasher);
    }
    hasher.finish()
}

fn dfa_mask_equivalent(left: &Automaton<DfaTable>, right: &Automaton<DfaTable>) -> bool {
    left.start == right.start
        && left.ends == right.ends
        && left.fsm.byte_table() == right.fsm.byte_table()
        && (0..left.fsm.num_states()).all(|state| {
            let has_rule_ref = |dfa: &Automaton<DfaTable>| {
                dfa.fsm
                    .edges(StateId(state as u32))
                    .iter()
                    .any(|edge| matches!(edge, FsmEdge::RuleRef { .. }))
            };
            has_rule_ref(left) == has_rule_ref(right)
        })
}

fn precompute_token_masks(
    rule_dfas: &[Automaton<DfaTable>],
    tokenizer_info: &Tokenizer,
    state_actions: &[StateAction],
    state_action_offsets: &[u32],
    runtime_rules: &[bool],
    deadline: Option<Instant>,
) -> Result<HashMap<(u32, u32), AdaptiveTokenMask>> {
    let mut masks = HashMap::new();
    let vocab_size = tokenizer_info.vocab_size();
    let bitmask_words = vocab_size.div_ceil(32);

    let mut dfa_cache: HashMap<u64, Vec<(usize, Vec<AdaptiveTokenMask>)>> = HashMap::new();

    for (rule_idx, dfa) in rule_dfas.iter().enumerate() {
        check_deadline(deadline)?;
        if !runtime_rules[rule_idx] {
            continue;
        }
        let dfa_hash = hash_dfa(dfa);

        if let Some((_, cached)) = dfa_cache.get(&dfa_hash).and_then(|bucket| {
            bucket
                .iter()
                .find(|(source_rule, _)| dfa_mask_equivalent(&rule_dfas[*source_rule], dfa))
        }) {
            for (state_idx, mask) in cached.iter().enumerate() {
                masks.insert((rule_idx as u32, state_idx as u32), mask.clone());
            }
            continue;
        }

        let mut rule_masks = Vec::with_capacity(dfa.fsm.num_states());

        for state_idx in 0..dfa.fsm.num_states() {
            let dfa_state = StateId(state_idx as u32);

            let edges = dfa.fsm.edges(dfa_state);
            if edges.is_empty() && !dfa.ends.get(state_idx).copied().unwrap_or(false) {
                rule_masks.push(AdaptiveTokenMask {
                    accepted_mask: vec![0u32; bitmask_words],
                    uncertain_tokens: Vec::new(),
                });
                continue;
            }

            let offset = state_action_offsets[rule_idx] as usize;
            let flags = state_actions[offset + state_idx].flags;
            let state_has_rule_ref = flags.has_rule_ref();
            let only_rule_ref = !flags.has_char_edges() && state_has_rule_ref;

            let mut accepted = vec![0u32; bitmask_words];
            let mut uncertain = Vec::new();

            let sorted = tokenizer_info.sorted_token_ids();
            let trie_end = tokenizer_info.trie_subtree_end();
            let mut i = 0;

            while i < sorted.len() {
                let token_id = sorted[i];
                let bytes = tokenizer_info
                    .decoded_token_bytes(token_id)
                    .expect("sorted token IDs have decoded bytes");

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
                        if dfa.fsm.next_state(dfa_state, bytes[0]).is_none()
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
            });
        }

        for (state_idx, mask) in rule_masks.iter().enumerate() {
            masks.insert((rule_idx as u32, state_idx as u32), mask.clone());
        }
        dfa_cache
            .entry(dfa_hash)
            .or_default()
            .push((rule_idx, rule_masks));
    }

    Ok(masks)
}
