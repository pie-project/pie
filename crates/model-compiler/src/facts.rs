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
///
/// `Sandwich` is gemma's, and it is BOTH rather than a third position: the
/// sub-layer is normed on the way in *and* its output is normed on the way
/// out, so a gemma block runs four norms where a llama block runs two. The
/// checkpoint states it plainly — gemma-4 ships `input_layernorm`,
/// `post_attention_layernorm`, `pre_feedforward_layernorm` and
/// `post_feedforward_layernorm` per layer, and a `Pre` deployment ships the
/// first two only.
///
/// The distinction is not cosmetic and it does not fail loudly. Reading a
/// gemma checkpoint as `Pre` binds `pre_feedforward_layernorm` where the
/// text asked for the post-attention one, drops both output norms, and
/// returns fluent text that is not the model's.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormPlacement {
    #[default]
    Pre,
    Post,
    Sandwich,
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

/// The MLA attention block's dims — the LATENT-CACHE geometry, said once.
///
/// glm5, kimi-k2 and kimi-k3 each carried a struct with these fields, and
/// the first two were field-identical; k3's added `output_gate` alone.
/// Three structs for one geometry is three places a family can disagree
/// about what MLA is.
///
/// `qk_nope_head_dim + qk_rope_head_dim` is the query width per head; the
/// CACHE stores the latent plus the rope half, so
/// `kv_lora_rank + qk_rope_head_dim` is its own number and not derivable
/// from the query's. That distinction is the whole reason MLA is not a
/// head count — the driver's pool geometry reads the second, and every
/// `PlannedFamily::head_dim_of` in the MLA lineage answers with it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MlaFacts {
    pub hidden: u32,
    pub heads: u32,
    pub q_lora_rank: u32,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub v_head_dim: u32,
    /// kimi-k3 gates the MLA output; the others do not. Serde-defaulted
    /// so the two families that never had the field read back unchanged.
    #[serde(default)]
    pub output_gate: bool,
}

impl MlaFacts {
    /// The per-head query width the `q_b` projection produces.
    #[must_use]
    pub const fn qk_head_dim(&self) -> u32 {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// The width `q_b_proj` writes: every head's nope+rope halves.
    #[must_use]
    pub const fn q_b_width(&self) -> u32 {
        self.heads * self.qk_head_dim()
    }

    /// The width `kv_a_proj_with_mqa` writes: the latent plus ONE shared
    /// rope half. Also one page row of the compressed cache, which is
    /// what a driver allocates per token and what every MLA family's
    /// `PlannedFamily::head_dim_of` answers with.
    ///
    /// A method rather than a stored fact, because a stored sum is a
    /// second thing to keep in step with its addends.
    #[must_use]
    pub const fn kv_a_width(&self) -> u32 {
        self.kv_lora_rank + self.qk_rope_head_dim
    }

    /// Every head's value half.
    #[must_use]
    pub const fn v_width(&self) -> u32 {
        self.heads * self.v_head_dim
    }

    /// The FUSED `q_kv_a` projection's width, for a deployment whose load
    /// joined the query's and the KV's latents into one bank.
    #[must_use]
    pub const fn q_kv_a_width(&self) -> u32 {
        self.q_lora_rank + self.kv_a_width()
    }
}

/// A ROUTED FFN's shape — the mixture, said once.
///
/// kimi-k2, kimi-k3 and nemotron-h carried this struct field-identically,
/// and glm5, deepseek-v4 and qwen3.5 carry the same four numbers plus one
/// of their own. These four ARE the mixture: how many experts, how many a
/// row goes to, how wide one is, and whether a shared expert rides beside
/// them.
///
/// The families that add a field are adding a real one — glm5's
/// `aligned_block` is its dispatch's tile, deepseek-v4's `hash_routed`
/// picks a different router — so they keep their own struct and this is
/// the part that stopped being written six times.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MoeFacts {
    /// Experts in this deployment's mixture.
    pub num_experts: u32,
    /// How many of them a row goes to — the router's k.
    pub top_k: u32,
    /// One expert's inner width.
    pub moe_intermediate: u32,
    /// The shared expert's inner width; 0 for a mixture without one,
    /// which is the plain `qwen3_moe` shape.
    pub shared_intermediate: u32,
}

impl MoeFacts {
    /// Whether a shared expert rides beside the routed ones. A predicate
    /// rather than a second field, for the reason every derived width in
    /// this module is a method: a stored answer is a second thing to keep
    /// in step.
    #[must_use]
    pub const fn has_shared_expert(&self) -> bool {
        self.shared_intermediate > 0
    }

    /// The fire's ROUTE count at `tokens` rows — tokens times k, the
    /// number the aligned dispatch's every extent is derived from.
    #[must_use]
    pub const fn routes(&self, tokens: u32) -> u32 {
        tokens * self.top_k
    }
}

/// A GQA attention block's three widths.
///
/// gemma-3n and nemotron-h carried this field-identically. It is the
/// smallest fact a family can have and still be describing attention:
/// query heads, key/value heads, and the width of one head. Everything
/// else an attention statement wants is derived from these — which is
/// what the two methods below are for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GqaFacts {
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
}

impl GqaFacts {
    /// Every query head's width — what `q_proj` writes.
    #[must_use]
    pub const fn q_width(&self) -> u32 {
        self.heads * self.head_dim
    }

    /// Every KV head's width — what `k_proj` and `v_proj` each write.
    #[must_use]
    pub const fn kv_width(&self) -> u32 {
        self.kv_heads * self.head_dim
    }

    /// Query heads per KV head. FlashInfer instantiates a fixed set of
    /// these and reports anything else by throwing, which is why the
    /// driver refuses an unservable ratio at LOAD — see
    /// `refuse_unservable_gqa`.
    #[must_use]
    pub const fn group_size(&self) -> u32 {
        if self.kv_heads == 0 { 0 } else { self.heads / self.kv_heads }
    }
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
