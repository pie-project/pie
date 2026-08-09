//! Load-time facts a declaration traces against.
//!
//! These are the `config.json`-derived values that the hand-written
//! `LlamaLikeForwardCfg` + `HfConfig` pair carries into the forward today,
//! reduced to what the *declaration* needs: everything here is resolved at
//! trace time and none of it survives into the traced form except as
//! constants and op choices.
//!
//! ## What is left here is the shared vocabulary
//!
//! [`NormPlacement`] and [`QkNorm`] are words more than one family is written
//! in. Every family's own facts moved to `crates/model`, beside that family's
//! declaration -- adding a fact to a model touches that model, and only it.

use serde::{Deserialize, Serialize};


/// Where each sub-layer's norm sits relative to the residual add.
///
/// `Pre` is the standard Llama shape: norm the residual stream *into* the
/// sub-layer, accumulate the sub-layer's projection straight back onto the
/// stream (the `beta=1` GEMM). `Post` is the OLMo-2/OLMo-3 shape: the
/// sub-layer reads the residual stream raw, the norm applies to the
/// sub-layer's OUTPUT, and only then does a separate residual add land it —
/// a genuinely different op ORDER, which is why it is a fact and not an
/// emitter choice. Mirrors the driver's `NormPlacement`
/// (`crates/driver-cuda/csrc/src/model/llama_like/llama_like.hpp`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormPlacement {
    #[default]
    Pre,
    Post,
}

/// Which q/k-norm convention the checkpoint ships, if any.
///
/// Two conventions exist in the wild (the driver's `rmsnorm_qk` dispatch,
/// `llama_like.cpp`): *per-head* (qwen3, gemma-3 — weight shape
/// `[head_dim]`, each head's channels normalised independently) and
/// *global* (OLMo-2, OLMo-3 — weight shape `[heads * head_dim]`, ONE
/// RMSNorm over the flattened projection). They are different arithmetic —
/// the global form shares one scale across heads — so the tri-state is a
/// fact, and the traced ops differ: per-head traces `RmsnormPerHead`,
/// global traces a plain row `Rmsnorm` (which is exactly what the kernel
/// launches: `kernels::norm::rmsnorm_bf16` over `[N, heads * head_dim]`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum QkNorm {
    #[default]
    Off,
    PerHead,
    Global,
}


/// The SLIDING WINDOW layer `l` attends over, `-1` for none.
///
/// One reader for a fact every family's backend carries the same way,
/// because the shape of the list is the same everywhere: empty is "no
/// window", one element is a config's single `sliding_window` broadcast
/// to every layer, and a longer list is the per-layer array an
/// alternating architecture states (gemma-2, gemma-4, OLMo-3).
///
/// It lives here rather than on any one facts struct for the reason
/// `NormPlacement` and `QkNorm` do: more than one family is written in
/// these words. Eleven executor sites across four families derived this
/// from `fwd_cfg.per_layer_window_left` -- an array no statement
/// mentioned -- and the dispatch statements carry it now.
///
/// A list SHORTER than the layer count is not an error: the last entry
/// covers the tail, which is what the drivers' fallback meant.
pub fn window_left_at(list: &[i32], l: u32) -> i32 {
    match list.len() {
        0 => -1,
        n => list[(l as usize).min(n - 1)],
    }
}

/// The ROPE THETA layer `l` rotates at.
///
/// The same shape as [`window_left_at`] and for the same reason: most
/// architectures carry one value in config, and some (gemma-4) alternate
/// it per layer between their local and global attention. A driver that
/// read the single one from config was reading the wrong one for half of
/// gemma-4's layers, which is why this is a fact and not a cfg field.
///
/// An EMPTY list is a text that states no rotation, and the accessor
/// says so by answering zero rather than by guessing a default.
pub fn rope_theta_at(list: &[f32], l: u32) -> f32 {
    match list.len() {
        0 => 0.0,
        n => list[(l as usize).min(n - 1)],
    }
}

/// Whether layer `l` runs FULL attention, on a family whose layer kinds
/// repeat with period `interval`.
///
/// THE SCHEDULE, said once. Three families wrote this predicate — gemma-4,
/// qwen3.5 and kimi-k3 — and gemma-4's own doc admits it: *"the same
/// predicate `Qwen35HybridFacts::is_full_attn` states, because the two
/// families schedule their layer kinds the same way."* A fact repeated in
/// three places is a fact three places can disagree about, and these did.
///
/// The core is identical: the LAST layer of each period is the full one,
/// which `(l + 1) % interval == 0` and `l % interval == interval - 1`
/// both spell. The disagreement was at `interval == 0`. gemma-4 and
/// qwen3.5 wrote `interval <= 1 || …`, making a zero mean EVERY layer;
/// kimi-k3 wrote `interval > 0 && …`, making it mean NONE.
///
/// Zero means NONE here, because zero is what a config with no
/// `full_attention` entry produces and "no periodic full layer" is what
/// that says. It costs those two families nothing: both derivations
/// refuse an interval of zero before any fact is built (gemma-4's
/// `regular` check requires `interval > 0`), so the value they disagreed
/// about is one neither can hold. An interval of ONE still means every
/// layer, which `(l + 1) % 1 == 0` gives without a special case.
#[must_use]
pub fn full_attn_at(interval: u32, l: u32) -> bool {
    interval > 0 && (l + 1) % interval == 0
}

/// Whether layer `l` is past the DENSE PREFIX — the second schedule
/// shape, and the one four families spell identically.
///
/// deepseek-v4, glm5, kimi-k2 and kimi-k3 each write
/// `l >= self.dense_layers` under the name `is_moe_layer`. Unlike
/// [`full_attn_at`] the four agree exactly, so this extraction changes
/// no behaviour at all; what it adds is a NAME for the shape. "A dense
/// prefix, then the mixture" is a schedule, and a fifth family that
/// wants it should find it stated rather than write the comparison a
/// fifth time.
///
/// The predicate is deliberately about the PREFIX and not about
/// mixtures: gemma3n's `is_sparse` is `l < self.sparsity_layers`, the
/// same schedule read from the other end, and naming this one
/// `is_moe_layer` here would have made that relationship invisible.
#[must_use]
pub fn after_dense_prefix(dense_layers: u32, l: u32) -> bool {
    l >= dense_layers
}

#[cfg(test)]
mod schedule {
    use super::full_attn_at;

    /// The LAST layer of each period is the full one, which is what all
    /// three families' spellings meant.
    #[test]
    fn the_last_layer_of_each_period_is_the_full_one() {
        // gemma-4 E4B: interval 6, full at 5, 11, …, 41.
        let full: Vec<u32> = (0..42).filter(|&l| full_attn_at(6, l)).collect();
        assert_eq!(full, vec![5, 11, 17, 23, 29, 35, 41]);
        // E2B: interval 5 over 35 layers, which the interval does not divide.
        let full: Vec<u32> = (0..35).filter(|&l| full_attn_at(5, l)).collect();
        assert_eq!(full, vec![4, 9, 14, 19, 24, 29, 34]);
    }

    /// An interval of ONE is every layer, with no special case: `(l + 1) %
    /// 1` is always zero. Two of the three families wrote `interval <= 1`
    /// to get this and did not need to.
    #[test]
    fn an_interval_of_one_is_every_layer() {
        assert!((0..8).all(|l| full_attn_at(1, l)));
    }

    /// The other shape: a dense prefix, then the mixture. Four families
    /// spell this identically, so unlike [`super::full_attn_at`] there was
    /// no edge to settle — what the extraction adds is the NAME.
    #[test]
    fn the_dense_prefix_runs_out_and_the_mixture_starts() {
        use super::after_dense_prefix;
        let moe: Vec<u32> = (0..8).filter(|&l| after_dense_prefix(3, l)).collect();
        assert_eq!(moe, vec![3, 4, 5, 6, 7]);
        // No prefix at all is a mixture from layer zero, which is what
        // every fully-sparse deployment states.
        assert!((0..8).all(|l| after_dense_prefix(0, l)));
    }

    /// THE EDGE THE THREE DISAGREED ABOUT. gemma-4 and qwen3.5 wrote
    /// `interval <= 1 || …`, so a zero meant EVERY layer; kimi-k3 wrote
    /// `interval > 0 && …`, so it meant NONE.
    ///
    /// None wins: zero is what a config with no `full_attention` entry
    /// produces, and "no periodic full layer" is what that says. It costs
    /// the other two nothing — their derivations refuse an interval of
    /// zero before any fact is built, so it is a value neither can hold.
    #[test]
    fn an_interval_of_zero_is_no_layer() {
        assert!((0..8).all(|l| !full_attn_at(0, l)));
    }
}
