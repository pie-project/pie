//! ASAp — grammar-aligned decoding (Park et al., 2405.21047).
//!
//! Grammar-constrained decoding (GCD) masks the tokens the grammar forbids and
//! renormalises what is left. That is *locally* correct and *globally* wrong:
//! it samples
//!
//! ```text
//! q(w | u) ∝ p_LM(w | u) · [w allowed at u]
//! ```
//!
//! whereas the distribution the caller actually wants — the LM conditioned on
//! the output being grammatical — is
//!
//! ```text
//! p*(w | u) ∝ p_LM(w | u) · [w allowed at u] · EFG(u · w)
//! ```
//!
//! where `EFG(v)` is the *expected future grammaticality* of the prefix `v`:
//! the probability that the LM, continuing freely from `v`, ever completes a
//! string the grammar accepts. GCD is the special case `EFG ≡ 1`, i.e. the
//! optimistic assumption that every locally-legal token can still be finished.
//! When it cannot, GCD has already spent probability mass that the true
//! posterior never had, and the resulting likelihoods are wrong.
//!
//! ## The approximation
//!
//! ASAp keeps an over-approximation `α ≥ EFG`, initialised to 1, and tightens
//! it from sampled trajectories. Along a sampled path, with `M(u)` the LM mass
//! the grammar leaves at `u` and `w*` the token taken there,
//!
//! ```text
//! α(u) = Σ_{w allowed}  p_LM(w | u) · α(u · w)
//!      = M(u) − p_LM(w* | u) · (1 − α(u · w*))
//! ```
//!
//! because every sibling of `w*` is still unexplored and keeps `α = 1`. The
//! recurrence is evaluated backwards from the leaf, where `α = 1` if the
//! grammar accepted and `α = 0` if the path died. Every term is non-negative
//! and bounded by `M(u) ≤ 1`, so `α` is a valid probability and, crucially,
//! can only *fall* as more of the tree is explored — the monotone convergence
//! from above that the paper proves, and which this inferlet asserts.
//!
//! ## What runs where
//!
//! The device does the sampling: `softmax` → grammar mask → multiply by `α` →
//! Gumbel-max, and returns the two scalars the recurrence needs (`M` and
//! `p_LM(w*)`). The host owns the trie of explored prefixes and ships each
//! step's `α` as a sparse `(index, value)` pair list — a node can have at most
//! one explored child per completed round, so a `rounds`-wide pair list is an
//! exact representation, not a truncation.
//!
//! ## Source
//!
//! Park et al., *Grammar-Aligned Decoding* —
//! <https://arxiv.org/abs/2405.21047> (Eq. 3–4, Algorithm 1).
//!
//! Faithfulness: **Exact (equivalent form)** — the paper's `EXPAND` walks a
//! shared trie in place; this rebuilds the path each round from the stored
//! `Vec<Vec<u32>>` of approximations, which is the same recurrence with
//! different storage. See
//! `inference-time-algorithms/10-implementation-faithfulness-audit.md`.

use inferlet::chat;
use inferlet::grammar::{Grammar, Matcher};
use inferlet::mask::unpack_mask;
use inferlet::ptir::attention::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_schema")]
    schema: String,
    #[serde(default = "default_rounds")]
    rounds: usize,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_seed")]
    seed: u32,
}

#[derive(Serialize)]
struct RoundReport {
    round: usize,
    text: String,
    tokens: usize,
    terminated: bool,
    /// `α` at the root *before* this round sampled, i.e. the approximation the
    /// round actually used. Round 1 always sees 1.0 — that is plain GCD.
    root_alpha_before: f32,
    /// `α` at the root after this round's trajectory was folded in.
    root_alpha_after: f32,
    /// log of the GCD proposal probability of the sampled string.
    gcd_logprob: f32,
    /// log of the ASAp proposal probability of the same string. Equal to
    /// `gcd_logprob` in round 1, where `α ≡ 1`.
    asap_logprob: f32,
}

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    rounds: usize,
    reports: Vec<RoundReport>,
    /// `α(root)` after each round. The paper's convergence result says this
    /// sequence is non-increasing and bounded below by the true EFG.
    root_alpha_trace: Vec<f32>,
    /// Whether that monotonicity actually held.
    monotone: bool,
    /// How much mass GCD's optimism had over-counted by the final round.
    total_correction: f32,
    /// Distinct strings the refinement explored.
    distinct_outputs: usize,
}

fn default_prompt() -> String {
    "Give the profile of a fictional engineer.".into()
}

fn default_schema() -> String {
    r#"{
        "type": "object",
        "properties": {
            "name": { "type": "string", "minLength": 1, "maxLength": 12 },
            "age": { "type": "integer", "minimum": 0, "maximum": 150 }
        },
        "required": ["name", "age"],
        "additionalProperties": false
    }"#
    .into()
}

fn default_rounds() -> usize {
    6
}

fn default_max_tokens() -> usize {
    64
}

fn default_seed() -> u32 {
    0x51D0
}

/// The explored prefix trie. `alpha[node]` is the current over-approximation of
/// that node's expected future grammaticality.
struct Trie {
    alpha: Vec<f32>,
    /// `M(u)`: the LM mass the grammar leaves standing at `u`. Deterministic
    /// given the prefix, so it is recorded on first visit and reused.
    mass: Vec<f32>,
    children: HashMap<(usize, u32), usize>,
    /// `p_LM(w | u)` on each explored edge — needed because the update sums
    /// over *every* explored child, not just the one this round took.
    edge_prob: HashMap<(usize, u32), f32>,
}

impl Trie {
    fn new() -> Self {
        Self {
            alpha: vec![1.0],
            mass: vec![1.0],
            children: HashMap::new(),
            edge_prob: HashMap::new(),
        }
    }

    /// `α(u) = Σ_{w allowed} p_LM(w|u) · α(u·w)`, evaluated as the grammar-left
    /// mass minus what the explored children have been shown to lose. Every
    /// unexplored sibling still contributes its full `p_LM` because its `α` is
    /// still the optimistic 1.
    fn recompute(&mut self, node: usize) {
        let deficit: f32 = self
            .children
            .iter()
            .filter(|((parent, _), _)| *parent == node)
            .map(|(edge, &child)| self.edge_prob[edge] * (1.0 - self.alpha[child]))
            .sum();
        self.alpha[node] = (self.mass[node] - deficit).clamp(0.0, 1.0);
    }

    fn child(&mut self, node: usize, token: u32) -> usize {
        if let Some(&existing) = self.children.get(&(node, token)) {
            return existing;
        }
        let fresh = self.alpha.len();
        self.alpha.push(1.0);
        self.mass.push(1.0);
        self.children.insert((node, token), fresh);
        fresh
    }

    /// The sparse `α` overrides in effect at `node`: one entry per explored
    /// child, everything else implicitly 1.
    fn overrides(&self, node: usize) -> Vec<(u32, f32)> {
        self.children
            .iter()
            .filter(|((parent, _), _)| *parent == node)
            .map(|((_, token), &child)| (*token, self.alpha[child]))
            .collect()
    }
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if input.max_tokens == 0 {
        return Err("max_tokens must be at least 1".into());
    }
    if input.rounds == 0 || input.rounds > 16 {
        return Err("rounds must satisfy 1 <= rounds <= 16".into());
    }

    let vocab = model::output_vocab_size();
    // Exact, not truncated: a node gains at most one explored child per round.
    let pairs = input.rounds;

    let mut prompt = chat::system_user(
        "Generate only the requested JSON value, with no markdown or explanation.",
        &input.prompt,
    );
    prompt.extend(chat::cue());
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;

    let mut trie = Trie::new();
    let mut reports = Vec::with_capacity(input.rounds);
    let mut trace = Vec::with_capacity(input.rounds);
    let mut distinct: Vec<String> = Vec::new();
    let pipeline = Pipeline::new();

    for round in 0..input.rounds {
        let root_alpha_before = trie.alpha[0];
        let constraint = Matcher::new(&Grammar::from_json_schema(&input.schema)?);

        // A fresh KV state per round: ASAp resamples the whole string, and a
        // WorkingSet permanently claims the first pipeline it fires on, so all
        // rounds share this inferlet's single pipeline.
        let ws = WorkingSet::new();
        let page_size = kv_page_size();
        let max_pages = (n + input.max_tokens as u32 + 1).div_ceil(page_size).max(1);
        ws.reserve(max_pages).context("reserve KV")?;

        let prompt_tokens = Channel::from_iter(prompt.iter().map(|&t| t as i32));
        let pre_indptr = Channel::from([0u32, n]).named("pre_indptr");
        let pre_positions = Channel::from_iter(0..n).named("pre_positions");
        let pre_pages = Channel::from_iter(0..max_pages).named("pre_pages");
        let pre_page_indptr = Channel::from([0u32, n.div_ceil(page_size)]).named("pre_page_indptr");
        let pre_w_slot = Channel::from_iter((0..n).map(|p| p / page_size)).named("pre_w_slot");
        let pre_w_off = Channel::from_iter((0..n).map(|p| p % page_size)).named("pre_w_off");
        let pre_kv_len = Channel::from([n]).named("pre_kv_len");
        let pre_mask = Channel::new([vocab], dtype::bool).named("pre_mask");
        let pre_alpha_idx = Channel::new([pairs as u32], dtype::u32).named("pre_alpha_idx");
        let pre_alpha_val = Channel::new([pairs as u32], dtype::f32).named("pre_alpha_val");
        let pre_rng = Channel::from([input.seed ^ (round as u32 * 0x9E37), 0]).named("pre_rng");
        let pre_token = Channel::new([1], dtype::i32).named("pre_token");
        let pre_mass = Channel::new([1], dtype::f32).named("pre_mass");
        let pre_prob = Channel::new([1], dtype::f32).named("pre_prob");
        let pre_wmass = Channel::new([1], dtype::f32).named("pre_wmass");

        let prefill = ForwardPass::new();
        prefill.embed(&prompt_tokens, &pre_indptr)?;
        prefill.attention(
            &ws,
            KvGeometry {
                readable_pages: ..,
                writable_pages: ..,
                kv_len: &pre_kv_len,
                pages: &pre_pages,
                page_indptr: &pre_page_indptr,
                w_slot: &pre_w_slot,
                w_off: &pre_w_off,
                positions: &pre_positions,
                mask: None,
            },
        )?;
        prefill.epilogue(move || {
            let allowed = pre_mask.take();
            let idx = pre_alpha_idx.take();
            let val = pre_alpha_val.take();
            let r = pre_rng.take();
            let (token, mass, prob, wmass) =
                asap_step(intrinsics::logits(), vocab, &allowed, &idx, &val, &r);
            pre_token.put(&token);
            pre_mass.put(&mass);
            pre_prob.put(&prob);
            pre_wmass.put(&wmass);
        });

        let (idx0, val0) = sparse_alpha(&trie, 0, vocab, pairs);
        pre_mask.put(unpack_mask(&constraint.mask(), vocab));
        pre_alpha_idx.put(idx0);
        pre_alpha_val.put(val0);
        prefill.submit(&pipeline).context("ASAp prefill")?;

        let first = pre_token.take_host::<i32>().await?;
        let mut path_nodes = vec![0usize];
        let mut masses = vec![pre_mass.take_host::<f32>().await?];
        let mut probs = vec![pre_prob.take_host::<f32>().await?];
        let mut weighted_masses = vec![pre_wmass.take_host::<f32>().await?];
        // The α the step actually used for the token it took.
        let mut used_alpha = vec![alpha_of(&trie, 0, first as u32)];

        let mut generated = vec![first as u32];
        constraint
            .accept_tokens(&[first as u32])
            .context("grammar rejected the prefill token")?;
        let mut node = trie.child(0, first as u32);
        path_nodes.push(node);

        if !constraint.is_terminated() && generated.len() < input.max_tokens {
            let token_in = Channel::from([first]).named("token_in");
            let embed_indptr = Channel::from([0u32, 1]).named("embed_indptr");
            let positions = Channel::from([n]).named("positions");
            let pages = Channel::from_iter(0..max_pages).named("pages");
            let page_indptr =
                Channel::from([0u32, (n + 1).div_ceil(page_size)]).named("page_indptr");
            let w_slot = Channel::from([n / page_size]).named("w_slot");
            let w_off = Channel::from([n % page_size]).named("w_off");
            let kv_len = Channel::from([n + 1]).named("kv_len");
            let mask_ch = Channel::new([vocab], dtype::bool).named("mask");
            let alpha_idx = Channel::new([pairs as u32], dtype::u32).named("alpha_idx");
            let alpha_val = Channel::new([pairs as u32], dtype::f32).named("alpha_val");
            let rng =
                Channel::from([input.seed ^ (round as u32 * 0x9E37) ^ 0x5bd1, 0]).named("rng");
            let token_out = Channel::new([1], dtype::i32).named("token_out");
            let mass_out = Channel::new([1], dtype::f32).named("mass_out");
            let prob_out = Channel::new([1], dtype::f32).named("prob_out");
            let wmass_out = Channel::new([1], dtype::f32).named("wmass_out");

            let decode = ForwardPass::new();
            decode.embed(&token_in, &embed_indptr)?;
            decode.attention(
                &ws,
                KvGeometry {
                    readable_pages: ..,
                    writable_pages: (n / page_size)..,
                    kv_len: &kv_len,
                    pages: &pages,
                    page_indptr: &page_indptr,
                    w_slot: &w_slot,
                    w_off: &w_off,
                    positions: &positions,
                    mask: None,
                },
            )?;
            decode.epilogue(move || {
                let length = kv_len.take();
                let allowed = mask_ch.take();
                let idx = alpha_idx.take();
                let val = alpha_val.take();
                let r = rng.take();
                let (token, mass, prob, wmass) =
                    asap_step(intrinsics::logits(), vocab, &allowed, &idx, &val, &r);

                let next_length = &length + 1u32;
                let page_count = next_length.div_ceil(page_size);
                token_in.put(&token);
                kv_len.put(&next_length);
                positions.put(&length);
                w_slot.put(&length / page_size);
                w_off.put(&length % page_size);
                page_indptr.put(indptr(1, &page_count));
                rng.put(&r + iota(2));
                token_out.put(&token);
                mass_out.put(&mass);
                prob_out.put(&prob);
                wmass_out.put(&wmass);
            });

            // Strictly sequential: each step's `α` depends on the trie node the
            // previous token landed on, so nothing can be submitted ahead.
            while !constraint.is_terminated() && generated.len() < input.max_tokens {
                let (idx, val) = sparse_alpha(&trie, node, vocab, pairs);
                mask_ch.put(unpack_mask(&constraint.mask(), vocab));
                alpha_idx.put(idx);
                alpha_val.put(val);
                decode.submit(&pipeline).context("ASAp decode")?;

                let token = token_out.take_host::<i32>().await? as u32;
                masses.push(mass_out.take_host::<f32>().await?);
                probs.push(prob_out.take_host::<f32>().await?);
                weighted_masses.push(wmass_out.take_host::<f32>().await?);
                used_alpha.push(alpha_of(&trie, node, token));
                generated.push(token);
                constraint.accept_tokens(&[token]).map_err(|e| {
                    format!(
                        "round {round} step {}: grammar rejected token {token}: {e}",
                        generated.len() - 1
                    )
                })?;
                node = trie.child(node, token);
                path_nodes.push(node);
            }
        }

        // Fold the trajectory back into the trie. A path that reached an
        // accepting state contributes α = 1 at the leaf; one that ran out of
        // budget or died contributes 0, because no grammatical string was
        // reached through it.
        let terminated = constraint.is_terminated();
        let leaf = *path_nodes.last().expect("path always has a leaf");
        trie.alpha[leaf] = if terminated { 1.0 } else { 0.0 };
        for step in (0..masses.len()).rev() {
            let parent = path_nodes[step];
            trie.mass[parent] = masses[step];
            trie.edge_prob
                .insert((parent, generated[step]), probs[step]);
            trie.recompute(parent);
        }

        let gcd_logprob = probs
            .iter()
            .zip(masses.iter())
            .map(|(p, m)| (p / m.max(1e-30)).max(1e-30).ln())
            .sum::<f32>();
        // What the α-reweighted proposal actually assigned to the same string.
        // Identical to `gcd_logprob` in round 1, where every α is still 1.
        let asap_logprob = (0..probs.len())
            .map(|t| {
                ((probs[t] * used_alpha[t]) / weighted_masses[t].max(1e-30))
                    .max(1e-30)
                    .ln()
            })
            .sum::<f32>();

        let text = model::decode(&generated)?;
        if !distinct.contains(&text) {
            distinct.push(text.clone());
        }
        trace.push(trie.alpha[0]);
        reports.push(RoundReport {
            round: round + 1,
            text,
            tokens: generated.len(),
            terminated,
            root_alpha_before,
            root_alpha_after: trie.alpha[0],
            gcd_logprob,
            asap_logprob,
        });
    }
    pipeline.close();

    let monotone = trace.windows(2).all(|w| w[1] <= w[0] + 1e-6);
    if !monotone {
        return Err(format!(
            "ASAp invariant violated: alpha(root) is not non-increasing across rounds: {trace:?}"
        ));
    }

    Ok(Output {
        sampler: "asap-grammar-aligned-decoding",
        rounds: input.rounds,
        total_correction: 1.0 - trace.last().copied().unwrap_or(1.0),
        distinct_outputs: distinct.len(),
        root_alpha_trace: trace,
        monotone,
        reports,
    })
}

/// The `α` in force for one edge — the explored child's value, or the
/// optimistic 1 if this round is the first to take it.
fn alpha_of(trie: &Trie, node: usize, token: u32) -> f32 {
    trie.children
        .get(&(node, token))
        .map(|&child| trie.alpha[child])
        .unwrap_or(1.0)
}

/// The sparse `α` override buffers for `node`, padded with the out-of-range
/// index `vocab` so the padding scatters somewhere harmless.
fn sparse_alpha(trie: &Trie, node: usize, vocab: u32, pairs: usize) -> (Vec<u32>, Vec<f32>) {
    let mut idx = vec![vocab; pairs];
    let mut val = vec![1.0f32; pairs];
    for (slot, (token, alpha)) in trie.overrides(node).into_iter().take(pairs).enumerate() {
        idx[slot] = token;
        val[slot] = alpha;
    }
    (idx, val)
}

/// One ASAp sampling step.
///
/// Returns `(token, M, p_LM(token))`, where `M` is the LM mass the grammar left
/// standing. Both scalars are read off the *unmodified* LM distribution — the
/// `α` reweighting steers the sample but must not enter the recurrence, which
/// is defined over `p_LM`.
fn asap_step(
    logits: Tensor,
    vocab: u32,
    allowed: &Tensor,
    alpha_idx: &Tensor,
    alpha_val: &Tensor,
    state: &Tensor,
) -> (Tensor, Tensor, Tensor, Tensor) {
    let probabilities = softmax(logits);
    let zero = broadcast(0.0f32, [vocab]);
    let masked = select(allowed, &probabilities, &zero);
    let mass = reduce_sum(&masked);

    // `α` is scattered into a `[vocab + 1]` base so the padding index `vocab`
    // has somewhere to land; duplicate padding writes are all 1.0, so their
    // undefined ordering does not matter.
    let base = broadcast(1.0f32, [vocab + 1]);
    let alpha = gather(scatter_set(&base, alpha_idx, alpha_val), iota(vocab));

    let weighted = &masked * &alpha;
    let weighted_mass = reduce_sum(&weighted);

    // Gumbel-max, but spelled out rather than taken from `gumbel_max`, so that
    // masking is the *last* thing that happens. Folding the mask into the
    // scores handed to `gumbel_max` does not survive: with forbidden entries
    // pushed to -1e9 the fused sampler still returned them (observed
    // chosen_score = -1e9 against best_score = -0.06). `masked_argmax` applies
    // a true -inf after the noise is already in place, which is the same
    // primitive the grammar-constrained-decoding inferlet relies on.
    let floored = log(max_elem(&weighted, 1e-30f32));
    let scores = &floored + gumbel(state, [vocab]);
    let token = masked_argmax(&scores, allowed);

    let probability = scalar_gather(&masked, &token);
    (
        reshape(token, [1]),
        reshape(mass, [1]),
        reshape(probability, [1]),
        reshape(weighted_mass, [1]),
    )
}
