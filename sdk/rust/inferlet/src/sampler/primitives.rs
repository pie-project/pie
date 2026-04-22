//! Reusable sampling primitives for WASM-side custom sampling.
//!
//! These primitives are composable building blocks for inferlets that need to
//! implement their own sampling pipeline — typically the grammar-constrained
//! path where only the WASM side knows which tokens are grammar-legal, so the
//! engine-side sampler must be bypassed.
//!
//! The canonical pipeline (matching vLLM's prob-space approximation used by
//! pieclaw's `ConstrainedSampler`) is:
//!
//! 1. Collect `(token, prob)` pairs from the engine's output.
//! 2. `apply_repetition_penalty(&mut dist, &history, penalty)` — scale down
//!    probs of tokens in the recent emission window.
//! 3. Sort `dist` descending by prob.
//! 4. `apply_top_k(&mut dist, k)` — truncate long tail.
//! 5. `apply_top_p(&mut dist, p)` — keep shortest prefix with cumulative mass
//!    meeting the nucleus threshold.
//! 6. `weighted_sample(&dist, &mut rng)` — weighted random draw.
//! 7. Push the sampled token into `history` for the next step.
//!
//! Callers handle normalization themselves (or skip it — `weighted_sample`
//! does not require sum-to-1).
//!
//! ### Why prob-space, not logit-space?
//!
//! vLLM applies `repetition_penalty` on pre-softmax logits. WASM-side custom
//! samplers only receive post-softmax probs (the true logits depend on the
//! unknown normalization constant Z), so we approximate via division in
//! prob-space. Strictly weaker than logit-space for high-confidence tokens,
//! but sufficient for the grammar path where the admissible set is already
//! narrow. Closing that gap requires engine-side penalty application.

use std::collections::{HashMap, VecDeque};

/// Bounded sliding-window tracker for recently emitted tokens.
///
/// Backed by a `VecDeque` (FIFO order + O(1) front pop) plus a `HashMap<u32,
/// u32>` ref-count mirror so `contains` runs in O(1) instead of the O(n)
/// linear scan you get from `VecDeque::contains`.
///
/// The map counts occurrences: when a token is evicted from the deque its
/// count decrements, and the map key is removed when the count hits zero. A
/// token appears in `contains` iff it currently occupies at least one slot in
/// the deque.
#[derive(Debug, Clone)]
pub struct EmittedHistory {
    deque: VecDeque<u32>,
    counts: HashMap<u32, u32>,
    max: usize,
}

impl EmittedHistory {
    /// Creates an empty history with room for at most `max` tokens.
    ///
    /// `max = 0` creates a degenerate "drop everything" history: `push` is a
    /// no-op and `contains` always returns `false`.
    pub fn new(max: usize) -> Self {
        Self {
            deque: VecDeque::with_capacity(max),
            counts: HashMap::new(),
            max,
        }
    }

    /// Append `token` to the window. If the window is full, the oldest token
    /// is evicted first.
    pub fn push(&mut self, token: u32) {
        if self.max == 0 {
            return;
        }
        if self.deque.len() == self.max {
            let evicted = self
                .deque
                .pop_front()
                .expect("deque at capacity must have a front");
            let count = self
                .counts
                .get_mut(&evicted)
                .expect("count invariant: every deque slot has a ref-count entry");
            *count -= 1;
            if *count == 0 {
                self.counts.remove(&evicted);
            }
        }
        self.deque.push_back(token);
        *self.counts.entry(token).or_insert(0) += 1;
    }

    /// O(1) membership test — true iff `token` currently occupies at least
    /// one slot in the window.
    pub fn contains(&self, token: u32) -> bool {
        self.counts.contains_key(&token)
    }

    /// Number of tokens currently stored (≤ `max`).
    pub fn len(&self) -> usize {
        self.deque.len()
    }

    /// True iff no tokens are stored.
    pub fn is_empty(&self) -> bool {
        self.deque.is_empty()
    }

    /// Configured maximum window size.
    pub fn capacity(&self) -> usize {
        self.max
    }
}

/// Divide the prob of any `(token, prob)` pair whose `token` appears in
/// `history` by `penalty`.
///
/// `penalty <= 1.0 + 1e-6` is treated as "disabled" (no-op) — this is the
/// neutral-default convention: a repetition_penalty of 1.0 means no scaling.
///
/// Renormalization is the caller's responsibility; `weighted_sample` does not
/// require a sum-to-1 distribution.
pub fn apply_repetition_penalty(dist: &mut [(u32, f32)], history: &EmittedHistory, penalty: f32) {
    if penalty <= 1.0 + 1e-6 {
        return;
    }
    if history.is_empty() {
        return;
    }
    for (token, prob) in dist.iter_mut() {
        if history.contains(*token) {
            *prob /= penalty;
        }
    }
}

/// Nucleus (top-p) truncation over an **already descending-sorted** dist.
///
/// Keeps the shortest prefix whose cumulative prob mass meets or exceeds
/// `top_p`, discarding the rest via `Vec::truncate`.
///
/// - `top_p >= 1.0` or `top_p <= 0.0` → no-op.
/// - Dist must be pre-sorted by the caller. This primitive is stateless and
///   cheap so callers can share one sort across top_k + top_p.
pub fn apply_top_p(dist: &mut Vec<(u32, f32)>, top_p: f32) {
    if top_p >= 1.0 || top_p <= 0.0 {
        return;
    }
    let total: f32 = dist.iter().map(|(_, p)| *p).sum();
    if total <= 0.0 {
        return;
    }
    let target = top_p * total;
    let mut acc = 0.0f32;
    let mut keep = dist.len();
    for (i, (_, p)) in dist.iter().enumerate() {
        acc += *p;
        if acc >= target {
            keep = i + 1;
            break;
        }
    }
    if keep < dist.len() {
        dist.truncate(keep);
    }
}

/// Top-k truncation over an **already descending-sorted** dist.
///
/// - `top_k == 0` or `top_k >= dist.len()` → no-op.
/// - Otherwise `dist.truncate(top_k)`.
pub fn apply_top_k(dist: &mut Vec<(u32, f32)>, top_k: u32) {
    if top_k == 0 {
        return;
    }
    let k = top_k as usize;
    if k >= dist.len() {
        return;
    }
    dist.truncate(k);
}

/// Weighted random draw from `(token, prob)` pairs.
///
/// Uses a cumulative-sum inverse-CDF scheme: draws a uniform `u ∈ [0, total)`
/// from `rng` (via `next_u32` normalized) and returns the first token whose
/// prefix sum reaches `u`. Does NOT require `probs` to sum to 1.
///
/// # Panics
/// Panics if `dist` is empty — an empty distribution indicates a caller bug
/// (grammar produced no admissible tokens, which the caller must handle
/// explicitly, e.g. by falling back to EOS).
///
/// # Degenerate input
/// If all probs are zero or negative, returns `dist[0].0`. The distribution
/// has collapsed but we still need to return something; callers should
/// inspect and fall back to a canonical token (EOS) before relying on this.
pub fn weighted_sample(dist: &[(u32, f32)], rng: &mut impl rand_core::RngCore) -> u32 {
    assert!(
        !dist.is_empty(),
        "weighted_sample called with empty distribution (caller bug)"
    );
    let total: f32 = dist.iter().map(|(_, p)| *p).sum();
    if !total.is_finite() || total <= 0.0 {
        return dist[0].0;
    }
    // uniform [0, 1) from u32 / 2^32
    let u01 = (rng.next_u32() as f64) / (u32::MAX as f64 + 1.0);
    let u = (u01 as f32) * total;
    let mut acc = 0.0f32;
    // Default to the last token: covers floating-point drift where `acc`
    // never quite reaches `u` due to rounding.
    let mut picked = dist[dist.len() - 1].0;
    for &(id, p) in dist {
        acc += p;
        if acc >= u {
            picked = id;
            break;
        }
    }
    picked
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_repetition_penalty_scales_history_tokens() {
        let mut dist = vec![(1u32, 0.5f32), (2u32, 0.3f32), (3u32, 0.2f32)];
        let mut h = EmittedHistory::new(256);
        h.push(1);
        apply_repetition_penalty(&mut dist, &h, 1.1);
        let new_p1 = dist.iter().find(|(t, _)| *t == 1).unwrap().1;
        assert!((new_p1 - 0.5 / 1.1).abs() < 1e-6);
        assert!((dist.iter().find(|(t, _)| *t == 2).unwrap().1 - 0.3).abs() < 1e-6);
        assert!((dist.iter().find(|(t, _)| *t == 3).unwrap().1 - 0.2).abs() < 1e-6);
    }

    #[test]
    fn apply_repetition_penalty_disabled_when_penalty_one() {
        let mut dist = vec![(1u32, 0.5f32)];
        let mut h = EmittedHistory::new(256);
        h.push(1);
        apply_repetition_penalty(&mut dist, &h, 1.0);
        assert_eq!(dist[0].1, 0.5);
    }

    #[test]
    fn apply_repetition_penalty_empty_history_is_no_op() {
        let mut dist = vec![(1u32, 0.5f32)];
        let h = EmittedHistory::new(256);
        apply_repetition_penalty(&mut dist, &h, 1.5);
        assert_eq!(dist[0].1, 0.5);
    }

    #[test]
    fn apply_top_p_truncates_at_cumulative_mass() {
        let mut dist = vec![(1u32, 0.5), (2, 0.3), (3, 0.15), (4, 0.05)];
        apply_top_p(&mut dist, 0.8);
        assert_eq!(dist.len(), 2);
    }

    #[test]
    fn apply_top_p_one_is_no_op() {
        let mut dist = vec![(1u32, 0.5), (2, 0.5)];
        apply_top_p(&mut dist, 1.0);
        assert_eq!(dist.len(), 2);
    }

    #[test]
    fn apply_top_p_keeps_at_least_one_element() {
        // First element alone can already satisfy any top_p < 1.0 when it
        // dominates the mass.
        let mut dist = vec![(1u32, 0.9), (2, 0.1)];
        apply_top_p(&mut dist, 0.5);
        assert_eq!(dist.len(), 1);
        assert_eq!(dist[0].0, 1);
    }

    #[test]
    fn apply_top_k_truncates_to_k() {
        let mut dist = vec![(1u32, 0.5), (2, 0.3), (3, 0.2)];
        apply_top_k(&mut dist, 2);
        assert_eq!(dist.len(), 2);
    }

    #[test]
    fn apply_top_k_zero_is_no_op() {
        let mut dist = vec![(1u32, 0.5), (2, 0.5)];
        apply_top_k(&mut dist, 0);
        assert_eq!(dist.len(), 2);
    }

    #[test]
    fn apply_top_k_larger_than_len_is_no_op() {
        let mut dist = vec![(1u32, 0.5), (2, 0.5)];
        apply_top_k(&mut dist, 10);
        assert_eq!(dist.len(), 2);
    }

    #[test]
    fn emitted_history_o1_contains() {
        let mut h = EmittedHistory::new(3);
        h.push(10);
        h.push(20);
        h.push(30);
        assert!(h.contains(10));
        h.push(40);
        assert!(!h.contains(10));
        assert!(h.contains(40));
    }

    #[test]
    fn emitted_history_refcount_duplicate_tokens() {
        let mut h = EmittedHistory::new(3);
        h.push(5);
        h.push(5);
        h.push(5);
        assert!(h.contains(5));
        h.push(6);
        assert!(h.contains(5));
        h.push(7);
        assert!(h.contains(5));
        h.push(8);
        assert!(!h.contains(5));
    }

    #[test]
    fn emitted_history_zero_capacity_is_sink() {
        let mut h = EmittedHistory::new(0);
        h.push(1);
        h.push(2);
        assert_eq!(h.len(), 0);
        assert!(!h.contains(1));
    }

    #[test]
    fn weighted_sample_deterministic_with_seeded_rng() {
        use rand_core::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let dist = vec![(10u32, 0.5f32), (20u32, 0.5f32)];
        let t1 = weighted_sample(&dist, &mut rng);
        let t2 = weighted_sample(&dist, &mut rng);
        assert!(t1 == 10 || t1 == 20);
        assert!(t2 == 10 || t2 == 20);
    }

    #[test]
    fn weighted_sample_single_element() {
        use rand_core::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let dist = vec![(42u32, 1.0f32)];
        assert_eq!(weighted_sample(&dist, &mut rng), 42);
    }

    #[test]
    fn weighted_sample_respects_weights_over_many_draws() {
        use rand_core::SeedableRng;
        // 90/10 split — over 10k draws, token 10 should dominate.
        let dist = vec![(10u32, 0.9f32), (20u32, 0.1f32)];
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let mut count10 = 0usize;
        let n = 10_000;
        for _ in 0..n {
            if weighted_sample(&dist, &mut rng) == 10 {
                count10 += 1;
            }
        }
        let frac = count10 as f64 / n as f64;
        assert!(
            (0.86..0.94).contains(&frac),
            "expected ~0.9 for token 10, got {}",
            frac
        );
    }

    #[test]
    fn weighted_sample_all_zero_probs_returns_first() {
        use rand_core::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let dist = vec![(7u32, 0.0f32), (8u32, 0.0f32)];
        assert_eq!(weighted_sample(&dist, &mut rng), 7);
    }

    #[test]
    #[should_panic(expected = "empty distribution")]
    fn weighted_sample_empty_panics() {
        use rand_core::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let dist: Vec<(u32, f32)> = vec![];
        weighted_sample(&dist, &mut rng);
    }

    #[test]
    fn full_pipeline_integration() {
        // Smoke-test the canonical grammar-path pipeline end-to-end.
        use rand_core::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(99);
        let mut history = EmittedHistory::new(16);
        history.push(2); // token 2 will be penalized

        let mut dist = vec![(1u32, 0.5f32), (2, 0.3), (3, 0.15), (4, 0.05)];
        apply_repetition_penalty(&mut dist, &history, 1.1);
        dist.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        apply_top_k(&mut dist, 3);
        apply_top_p(&mut dist, 0.9);
        assert!(!dist.is_empty());
        let sampled = weighted_sample(&dist, &mut rng);
        assert!([1u32, 2, 3, 4].contains(&sampled));
    }
}
