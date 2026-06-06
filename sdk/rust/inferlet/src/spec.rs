//! Speculator trait for speculative-decoding drafters.
//!
//! Plug a [`Speculator`] into a [`Generator`](crate::gen::Generator) via
//! [`Generator::speculator`](crate::gen::Generator::speculator) to drive
//! draft tokens off your own logic.
//!
//! For host-driven speculation (where the runtime returns next-iter draft
//! tokens via the forward-pass output's spec channel), call
//! [`Generator::system_speculation`](crate::gen::Generator::system_speculation)
//! instead — that mode is built into the Generator and does not need a
//! `Speculator` impl.

/// A speculative-decoding drafter. Each iteration the [`Generator`] asks
/// for `draft()` tokens, runs the verifier, then reports `accept()`. On
/// rejection the Generator calls `rollback()` so the speculator can
/// truncate any state it grew during drafting.
///
/// [`Generator`]: crate::gen::Generator
pub trait Speculator: Send {
    /// Produce draft tokens and their absolute positions for the next
    /// forward pass. Empty vec means "no speculation this step."
    fn draft(&mut self) -> (Vec<u32>, Vec<u32>);

    /// Called with the verifier's accepted token sequence. The first
    /// accepted token corresponds to the anchor's own next-token
    /// prediction; the rest (if any) are matched drafts.
    fn accept(&mut self, accepted: &[u32]);

    /// Roll back the last `n` drafted tokens — used when the verifier
    /// rejects the tail of the draft sequence and the speculator's own
    /// internal context needs to mirror that truncation.
    fn rollback(&mut self, n: u32) {
        let _ = n;
    }

    /// Reset the speculator to its initial state.
    fn reset(&mut self) {}
}

/// Cacheback tree drafting primitives.
///
/// This module is intentionally SDK-local and side-effect free: it implements
/// the paper-shaped drafter data structures, branch-isolation mask, acceptance
/// walk, metrics, and runtime support guard without claiming the current Pie
/// runtime can execute branch KV commit/rollback. The system-spec runtime and
/// drivers remain linear-only until they can preserve branch KV semantics.
pub mod cacheback_tree {
    use std::collections::{HashMap, VecDeque};

    /// Configuration for the Cacheback LRU n-gram tree drafter.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CachebackTreeConfig {
        /// Number of committed tokens used as the cache lookup key.
        pub leader_len: usize,
        /// Number of continuation tokens stored per cache hit.
        pub follower_len: usize,
        /// Maximum non-root draft nodes in one verifier pass.
        pub max_tree_nodes: usize,
        /// Maximum draft depth, measured in tokens from the verifier anchor.
        pub max_tree_depth: usize,
        /// Maximum number of distinct leader keys retained by the LRU cache.
        pub cache_capacity: usize,
    }

    impl CachebackTreeConfig {
        fn validate(self) -> Result<(), String> {
            if self.leader_len == 0 {
                return Err("cacheback_tree: leader_len must be > 0".to_string());
            }
            if self.follower_len == 0 {
                return Err("cacheback_tree: follower_len must be > 0".to_string());
            }
            if self.max_tree_nodes == 0 {
                return Err("cacheback_tree: max_tree_nodes must be > 0".to_string());
            }
            if self.max_tree_depth == 0 {
                return Err("cacheback_tree: max_tree_depth must be > 0".to_string());
            }
            if self.cache_capacity == 0 {
                return Err("cacheback_tree: cache_capacity must be > 0".to_string());
            }
            Ok(())
        }
    }

    /// Runtime/driver capability gate for full tree execution.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct RuntimeTreeSupport {
        /// Whether the backend can honor per-draft branch attention masks.
        pub supports_branch_attention_masks: bool,
        /// Whether accepted branch KV rows can be committed while rejected
        /// branch rows are discarded even when they are not a suffix.
        pub supports_branch_kv_commit: bool,
    }

    impl RuntimeTreeSupport {
        /// Returns a loud fallback reason when tree mode must not be reported
        /// as active for this runtime/driver combination.
        pub fn fallback_reason(&self) -> Option<String> {
            if !self.supports_branch_attention_masks {
                return Some("tree_cacheback requires branch attention masks".to_string());
            }
            if !self.supports_branch_kv_commit {
                return Some("tree_cacheback requires branch-aware KV commit/rollback".to_string());
            }
            None
        }
    }

    #[derive(Debug, Clone)]
    struct TreeNode {
        token: u32,
        parent: Option<usize>,
        depth: usize,
    }

    /// A flattened pre-order Cacheback draft tree.
    #[derive(Debug, Clone)]
    pub struct DraftTree {
        nodes: Vec<TreeNode>,
    }

    impl DraftTree {
        /// Draft tokens in verifier input order.
        pub fn tokens(&self) -> Vec<u32> {
            self.nodes.iter().map(|n| n.token).collect()
        }

        /// Parent indices in [`Self::tokens`] order. `None` means a root child.
        pub fn parents(&self) -> Vec<Option<usize>> {
            self.nodes.iter().map(|n| n.parent).collect()
        }

        /// 1-based draft depths from the verifier anchor.
        pub fn depths(&self) -> Vec<usize> {
            self.nodes.iter().map(|n| n.depth).collect()
        }

        /// First-token alternatives at the root, in newest-cache-hit order.
        pub fn root_children_tokens(&self) -> Vec<u32> {
            self.nodes
                .iter()
                .filter(|n| n.parent.is_none())
                .map(|n| n.token)
                .collect()
        }

        /// Number of leaf branches represented by this draft tree.
        pub fn leaf_count(&self) -> usize {
            self.nodes
                .iter()
                .enumerate()
                .filter(|(idx, _)| !self.nodes.iter().any(|n| n.parent == Some(*idx)))
                .count()
        }

        /// Draft-to-draft branch attention mask.
        ///
        /// `mask[row][col]` is true iff `col` is `row` itself or an ancestor of
        /// `row`. The pre-existing committed context/anchor is common to every
        /// branch and is intentionally not represented in this draft-only mask.
        pub fn branch_attention_mask(&self) -> Vec<Vec<bool>> {
            let mut mask = vec![vec![false; self.nodes.len()]; self.nodes.len()];
            for row in 0..self.nodes.len() {
                let mut cursor = Some(row);
                while let Some(idx) = cursor {
                    mask[row][idx] = true;
                    cursor = self.nodes[idx].parent;
                }
            }
            mask
        }

        /// Verify the tree from target-model predictions and select the
        /// longest accepted path while preserving target-model semantics.
        ///
        /// `root_next_token` is the target model's prediction at the verifier
        /// anchor. `node_next_tokens[i]` is the prediction after draft node `i`
        /// when that node is evaluated with branch-isolated attention.
        pub fn verify(
            &self,
            predictions: &VerificationPredictions,
        ) -> Result<VerificationOutcome, String> {
            if predictions.node_next_tokens.len() != self.nodes.len() {
                return Err(format!(
                    "cacheback_tree verify: expected {} node predictions, got {}",
                    self.nodes.len(),
                    predictions.node_next_tokens.len()
                ));
            }

            let mut accepted = vec![false; self.nodes.len()];
            for (idx, node) in self.nodes.iter().enumerate() {
                let parent_accepted = match node.parent {
                    Some(parent) => accepted[parent],
                    None => true,
                };
                if !parent_accepted {
                    continue;
                }
                let predicted = match node.parent {
                    Some(parent) => predictions.node_next_tokens[parent],
                    None => predictions.root_next_token,
                };
                accepted[idx] = predicted == node.token;
            }

            let best = accepted
                .iter()
                .enumerate()
                .filter(|(_, ok)| **ok)
                .max_by_key(|(idx, _)| (self.nodes[*idx].depth, std::cmp::Reverse(*idx)))
                .map(|(idx, _)| idx);

            let mut accepted_path_nodes = Vec::new();
            if let Some(mut idx) = best {
                loop {
                    accepted_path_nodes.push(idx);
                    match self.nodes[idx].parent {
                        Some(parent) => idx = parent,
                        None => break,
                    }
                }
                accepted_path_nodes.reverse();
            }

            let mut accepted_tokens: Vec<u32> = accepted_path_nodes
                .iter()
                .map(|idx| self.nodes[*idx].token)
                .collect();
            let bonus = match accepted_path_nodes.last() {
                Some(idx) => predictions.node_next_tokens[*idx],
                None => predictions.root_next_token,
            };
            accepted_tokens.push(bonus);

            let metrics = TreeVerificationMetrics {
                tree_nodes_drafted: self.nodes.len(),
                verified_branches: self.leaf_count(),
                accepted_path_length: accepted_path_nodes.len(),
                rejected_nodes: self.nodes.len().saturating_sub(accepted_path_nodes.len()),
                accepted_tokens_per_verifier_pass: accepted_tokens.len() as f64,
            };

            Ok(VerificationOutcome {
                accepted_tokens,
                accepted_path_nodes,
                metrics,
            })
        }
    }

    /// Target-model verifier predictions for a draft tree.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct VerificationPredictions {
        pub root_next_token: u32,
        pub node_next_tokens: Vec<u32>,
    }

    /// Structured metrics emitted for one tree verifier pass.
    #[derive(Debug, Clone, PartialEq)]
    pub struct TreeVerificationMetrics {
        pub tree_nodes_drafted: usize,
        pub verified_branches: usize,
        pub accepted_path_length: usize,
        pub rejected_nodes: usize,
        pub accepted_tokens_per_verifier_pass: f64,
    }

    /// Result of applying target-model verification to a draft tree.
    #[derive(Debug, Clone, PartialEq)]
    pub struct VerificationOutcome {
        /// Tokens to append to user-visible output: verified draft path plus
        /// the target-model bonus token from the accepted leaf (or root).
        pub accepted_tokens: Vec<u32>,
        /// Indices of draft nodes accepted along the chosen path.
        pub accepted_path_nodes: Vec<usize>,
        pub metrics: TreeVerificationMetrics,
    }

    /// Cache/rollback accounting after applying a tree verification outcome.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct TreeRollbackPlan {
        /// Number of draft tree nodes that survived verification.
        pub accepted_draft_nodes: usize,
        /// Number of draft tree nodes that must not be committed.
        pub rejected_draft_nodes: usize,
    }

    /// LRU n-gram Cacheback tree drafter.
    #[derive(Debug, Clone)]
    pub struct CachebackTreeDrafter {
        cfg: CachebackTreeConfig,
        followers_by_leader: HashMap<Vec<u32>, VecDeque<Vec<u32>>>,
        leaders_lru: VecDeque<Vec<u32>>,
    }

    impl CachebackTreeDrafter {
        pub fn new(cfg: CachebackTreeConfig) -> Result<Self, String> {
            cfg.validate()?;
            Ok(Self {
                cfg,
                followers_by_leader: HashMap::new(),
                leaders_lru: VecDeque::new(),
            })
        }

        /// Add committed target-model tokens to the LRU n-gram cache.
        pub fn observe_committed(&mut self, tokens: &[u32]) {
            let window = self.cfg.leader_len + self.cfg.follower_len;
            if tokens.len() < window {
                return;
            }
            for start in 0..=tokens.len() - window {
                let leader = tokens[start..start + self.cfg.leader_len].to_vec();
                let follower = tokens[start + self.cfg.leader_len..start + window].to_vec();
                self.insert_ngram(leader, follower);
            }
        }

        /// Apply accepted target-model tokens to the cache and return the
        /// branch rollback plan for the verifier side.
        ///
        /// `prior_context_suffix` should contain at least the leader tokens
        /// immediately preceding the verification pass so cache entries that
        /// cross the old/new boundary are learned. Rejected branches are never
        /// observed, preserving exact target-model semantics.
        pub fn accept_verified(
            &mut self,
            prior_context_suffix: &[u32],
            outcome: &VerificationOutcome,
        ) -> TreeRollbackPlan {
            let mut committed =
                Vec::with_capacity(prior_context_suffix.len() + outcome.accepted_tokens.len());
            committed.extend_from_slice(prior_context_suffix);
            committed.extend_from_slice(&outcome.accepted_tokens);
            self.observe_committed(&committed);
            self.observe_committed_with_partial_tail(&committed);

            TreeRollbackPlan {
                accepted_draft_nodes: outcome.accepted_path_nodes.len(),
                rejected_draft_nodes: outcome.metrics.rejected_nodes,
            }
        }

        /// Build a bounded draft tree from the suffix of `context`.
        pub fn draft_tree(&self, context: &[u32]) -> Result<Option<DraftTree>, String> {
            if context.len() < self.cfg.leader_len {
                return Ok(None);
            }

            let mut nodes = Vec::new();
            self.expand_from_context(context.to_vec(), None, 0, &mut nodes);
            if nodes.is_empty() {
                return Ok(None);
            }
            Ok(Some(DraftTree { nodes }))
        }

        fn observe_committed_with_partial_tail(&mut self, tokens: &[u32]) {
            if tokens.len() <= self.cfg.leader_len {
                return;
            }
            let last_leader_start = tokens.len() - self.cfg.leader_len;
            for start in 0..last_leader_start {
                let follower_start = start + self.cfg.leader_len;
                let follower_end = (follower_start + self.cfg.follower_len).min(tokens.len());
                if follower_start >= follower_end {
                    continue;
                }
                let leader = tokens[start..follower_start].to_vec();
                let follower = tokens[follower_start..follower_end].to_vec();
                self.insert_ngram(leader, follower);
            }
        }

        fn insert_ngram(&mut self, leader: Vec<u32>, follower: Vec<u32>) {
            if self.followers_by_leader.contains_key(&leader) {
                self.touch_leader(&leader);
            } else {
                while self.followers_by_leader.len() >= self.cfg.cache_capacity {
                    if let Some(oldest) = self.leaders_lru.pop_front() {
                        self.followers_by_leader.remove(&oldest);
                    } else {
                        break;
                    }
                }
                self.leaders_lru.push_back(leader.clone());
            }

            let followers = self
                .followers_by_leader
                .entry(leader)
                .or_insert_with(VecDeque::new);
            if let Some(pos) = followers.iter().position(|f| f == &follower) {
                followers.remove(pos);
            }
            followers.push_front(follower);
        }

        fn touch_leader(&mut self, leader: &[u32]) {
            if let Some(pos) = self.leaders_lru.iter().position(|k| k.as_slice() == leader) {
                let key = self.leaders_lru.remove(pos).expect("LRU position exists");
                self.leaders_lru.push_back(key);
            }
        }

        fn expand_from_context(
            &self,
            context_path: Vec<u32>,
            parent: Option<usize>,
            depth: usize,
            nodes: &mut Vec<TreeNode>,
        ) {
            if nodes.len() >= self.cfg.max_tree_nodes || depth >= self.cfg.max_tree_depth {
                return;
            }
            if context_path.len() < self.cfg.leader_len {
                return;
            }

            let leader_start = context_path.len() - self.cfg.leader_len;
            let leader = &context_path[leader_start..];
            let Some(followers) = self.followers_by_leader.get(leader) else {
                return;
            };

            for follower in followers {
                if nodes.len() >= self.cfg.max_tree_nodes {
                    break;
                }

                let mut branch_context = context_path.clone();
                let mut branch_parent = parent;
                let mut branch_depth = depth;

                for &token in follower {
                    if nodes.len() >= self.cfg.max_tree_nodes
                        || branch_depth >= self.cfg.max_tree_depth
                    {
                        break;
                    }
                    branch_depth += 1;
                    let idx = nodes.len();
                    nodes.push(TreeNode {
                        token,
                        parent: branch_parent,
                        depth: branch_depth,
                    });
                    branch_context.push(token);
                    branch_parent = Some(idx);
                }

                self.expand_from_context(branch_context, branch_parent, branch_depth, nodes);
            }
        }
    }
}

#[cfg(test)]
mod cacheback_tree_tests {
    use super::cacheback_tree::{
        CachebackTreeConfig, CachebackTreeDrafter, RuntimeTreeSupport, VerificationPredictions,
    };

    #[test]
    fn lru_ngram_cache_refreshes_and_evicts_oldest_leaders() {
        let cfg = CachebackTreeConfig {
            leader_len: 2,
            follower_len: 2,
            max_tree_nodes: 8,
            max_tree_depth: 4,
            cache_capacity: 2,
        };
        let mut drafter = CachebackTreeDrafter::new(cfg).unwrap();

        drafter.observe_committed(&[1, 2, 3, 4]);
        drafter.observe_committed(&[5, 6, 7, 8]);
        // Refresh leader [1, 2], making [5, 6] the oldest leader.
        drafter.observe_committed(&[1, 2, 9, 10]);
        drafter.observe_committed(&[11, 12, 13, 14]);

        assert!(drafter.draft_tree(&[5, 6]).unwrap().is_none());
        assert_eq!(
            drafter
                .draft_tree(&[1, 2])
                .unwrap()
                .unwrap()
                .root_children_tokens(),
            vec![9, 3]
        );
    }

    #[test]
    fn tree_construction_expands_cache_hits_until_node_or_depth_limit() {
        let cfg = CachebackTreeConfig {
            leader_len: 2,
            follower_len: 2,
            max_tree_nodes: 5,
            max_tree_depth: 3,
            cache_capacity: 16,
        };
        let mut drafter = CachebackTreeDrafter::new(cfg).unwrap();
        drafter.observe_committed(&[1, 2, 3, 4, 7, 8]);
        drafter.observe_committed(&[1, 2, 5, 6]);
        drafter.observe_committed(&[3, 4, 9, 10]);

        let tree = drafter.draft_tree(&[1, 2]).unwrap().unwrap();

        assert_eq!(tree.tokens(), vec![5, 6, 3, 4, 9]);
        assert_eq!(tree.parents(), vec![None, Some(0), None, Some(2), Some(3)]);
        assert_eq!(tree.depths(), vec![1, 2, 1, 2, 3]);
        assert_eq!(tree.leaf_count(), 2);
    }

    #[test]
    fn branch_attention_mask_allows_only_ancestors_and_self() {
        let cfg = CachebackTreeConfig {
            leader_len: 2,
            follower_len: 2,
            max_tree_nodes: 4,
            max_tree_depth: 2,
            cache_capacity: 16,
        };
        let mut drafter = CachebackTreeDrafter::new(cfg).unwrap();
        drafter.observe_committed(&[1, 2, 3, 4]);
        drafter.observe_committed(&[1, 2, 5, 6]);
        let tree = drafter.draft_tree(&[1, 2]).unwrap().unwrap();

        assert_eq!(tree.tokens(), vec![5, 6, 3, 4]);
        assert_eq!(
            tree.branch_attention_mask(),
            vec![
                vec![true, false, false, false],
                vec![true, true, false, false],
                vec![false, false, true, false],
                vec![false, false, true, true],
            ]
        );
    }

    #[test]
    fn verification_accepts_longest_model_verified_path_and_reports_metrics() {
        let cfg = CachebackTreeConfig {
            leader_len: 2,
            follower_len: 2,
            max_tree_nodes: 4,
            max_tree_depth: 2,
            cache_capacity: 16,
        };
        let mut drafter = CachebackTreeDrafter::new(cfg).unwrap();
        drafter.observe_committed(&[1, 2, 3, 4]);
        drafter.observe_committed(&[1, 2, 5, 6]);
        let tree = drafter.draft_tree(&[1, 2]).unwrap().unwrap();

        let outcome = tree
            .verify(&VerificationPredictions {
                root_next_token: 5,
                node_next_tokens: vec![6, 42, 4, 99],
            })
            .unwrap();

        assert_eq!(outcome.accepted_tokens, vec![5, 6, 42]);
        assert_eq!(outcome.accepted_path_nodes, vec![0, 1]);
        assert_eq!(outcome.metrics.tree_nodes_drafted, 4);
        assert_eq!(outcome.metrics.verified_branches, 2);
        assert_eq!(outcome.metrics.accepted_path_length, 2);
        assert_eq!(outcome.metrics.rejected_nodes, 2);
        assert_eq!(outcome.metrics.accepted_tokens_per_verifier_pass, 3.0);
    }

    #[test]
    fn runtime_support_guard_does_not_silently_claim_tree_mode() {
        let unsupported = RuntimeTreeSupport {
            supports_branch_attention_masks: false,
            supports_branch_kv_commit: true,
        }
        .fallback_reason();
        assert_eq!(
            unsupported.as_deref(),
            Some("tree_cacheback requires branch attention masks")
        );

        let unsupported = RuntimeTreeSupport {
            supports_branch_attention_masks: true,
            supports_branch_kv_commit: false,
        }
        .fallback_reason();
        assert_eq!(
            unsupported.as_deref(),
            Some("tree_cacheback requires branch-aware KV commit/rollback")
        );

        let supported = RuntimeTreeSupport {
            supports_branch_attention_masks: true,
            supports_branch_kv_commit: true,
        }
        .fallback_reason();
        assert!(supported.is_none());
    }

    #[test]
    fn accepted_path_updates_cache_and_reports_rejected_branch_rollback() {
        let cfg = CachebackTreeConfig {
            leader_len: 2,
            follower_len: 2,
            max_tree_nodes: 4,
            max_tree_depth: 2,
            cache_capacity: 16,
        };
        let mut drafter = CachebackTreeDrafter::new(cfg).unwrap();
        drafter.observe_committed(&[1, 2, 3, 4]);
        drafter.observe_committed(&[1, 2, 5, 6]);
        let tree = drafter.draft_tree(&[1, 2]).unwrap().unwrap();
        let outcome = tree
            .verify(&VerificationPredictions {
                root_next_token: 5,
                node_next_tokens: vec![6, 42, 4, 99],
            })
            .unwrap();

        let rollback = drafter.accept_verified(&[1, 2], &outcome);

        assert_eq!(rollback.accepted_draft_nodes, 2);
        assert_eq!(rollback.rejected_draft_nodes, 2);
        assert_eq!(
            drafter
                .draft_tree(&[5, 6])
                .unwrap()
                .unwrap()
                .root_children_tokens(),
            vec![42]
        );
    }
}
