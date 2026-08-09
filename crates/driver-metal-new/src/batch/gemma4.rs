//! Gemma 4 decode geometry.
//!
//! Defaults are `google/gemma-4-E2B-it`'s. Three things make this
//! family's shape different, all in the SCHEDULE rather than the
//! kernels:
//!
//! * **head_dim is per attention type.** Sliding layers use `head_dim`,
//!   full layers `global_head_dim` — so are the rope base and the
//!   partial-rotary factor.
//! * **The tail of the stack shares KV.** The last
//!   `num_kv_shared_layers` layers re-attend the most recent earlier
//!   layer of the SAME attention type; they ship no k/v projections at
//!   all.
//! * **The MLP doubles over exactly that range.**
//!
//! The 26B mixture adds two more: the routed branch sits BESIDE the
//! dense MLP (both read the post-attention residual and their outputs
//! add), and full-attention layers take V from the K PROJECTION —
//! `V = v_norm(k_proj(x))` while `K = rope(k_norm(k_proj(x)))` — with a
//! KV head count of their own. Neither is an optimisation to skip:
//! both are which weights the checkpoint ships.

use crate::facts::ModelFacts;

use super::abi::Kernel;
use super::consts::KN;
use super::geometry::{AffineFormat, DecodeGeometry};
use super::geometry_facts::GeometryRefused;

/// The family's shape. Defaults are `google/gemma-4-E2B-it`'s.
#[derive(Clone, Debug, PartialEq)]
pub struct Gemma4Geometry {
    /// The model width.
    pub hidden: u32,
    /// Decoder layers.
    pub n_layers: u32,
    /// The head's vocabulary.
    pub vocab: u32,
    /// The RMS-norm epsilon.
    pub eps: f32,
    /// Gemma ties its head to the embedding.
    pub tied_embeddings: bool,
    /// Query heads.
    pub n_q_heads: u32,
    /// Key/value heads on the sliding layers (and the full ones, unless
    /// [`n_global_kv_heads`](Self::n_global_kv_heads) overrides).
    pub n_kv_heads: u32,
    /// Sliding layers' head width; full layers use
    /// [`global_head_dim`](Self::global_head_dim).
    pub head_dim: u32,
    /// The affine format most tensors are in.
    pub quant: AffineFormat,
    /// The format some tensors are in when it is not the model's —
    /// mlx_lm quantizes per tensor and the 26B's predicate has shipped
    /// two different exemption sets. `{0, 0}` means one format.
    pub ffn_quant: AffineFormat,
    /// Whether the dense `mlp.{gate,up,down}` are at
    /// [`ffn_quant`](Self::ffn_quant).
    pub alt_quant_ffn: bool,
    /// Whether `router.proj` is at [`ffn_quant`](Self::ffn_quant).
    pub alt_quant_router: bool,
    /// Full layers' head width.
    pub global_head_dim: u32,
    /// Full layers rotate this fraction of their head; sliding layers
    /// rotate all of it.
    pub full_partial_rotary: f32,
    /// Full layers' rope base.
    pub rope_theta_global: f32,
    /// Sliding layers' rope base.
    pub rope_theta_local: f32,
    /// Sliding layers attend the last this-many positions.
    pub sliding_window: u32,
    /// Dense MLP base width; doubles on the KV-shared range when
    /// [`double_wide_mlp`](Self::double_wide_mlp).
    pub intermediate: u32,
    /// See [`intermediate`](Self::intermediate).
    pub double_wide_mlp: bool,
    /// Per-layer embeddings: a second table `n_layers × this` wide. 0
    /// disables the PLE path entirely.
    pub per_layer_emb_dim: u32,
    /// The tail layers that re-attend an earlier layer's pages.
    pub num_kv_shared_layers: u32,
    /// The 26B's mixture switch — carried separately from the counts so
    /// a config that sets it without naming experts is refusable.
    pub enable_moe: bool,
    /// Routed experts.
    pub n_experts: u32,
    /// Experts each token routes to.
    pub experts_per_token: u32,
    /// One expert's width — much narrower than the dense
    /// [`intermediate`](Self::intermediate) beside it.
    pub moe_intermediate: u32,
    /// Full-attention layers take V from the K projection and ship no
    /// `v_proj` at all.
    pub attention_k_eq_v: bool,
    /// Full-attention layers' own KV head count, when
    /// [`attention_k_eq_v`](Self::attention_k_eq_v).
    pub n_global_kv_heads: u32,
    /// `out = cap · tanh(logits / cap)`; 0 disables.
    pub final_softcap: f32,
    /// One full-attention layer every this-many, counting from 1.
    pub full_attn_interval: u32,
    /// The widest fire the pools are sized for.
    pub max_tokens: u32,
    /// The most requests one fire may carry.
    pub max_requests: u32,
    /// Recurrent-state slots (none; carried for the shared plumbing).
    pub max_slots: u32,
    /// The KV page size.
    pub kv_page_size: u32,
    /// Physical pages in the paged pool.
    pub total_pages: u32,
    /// Whether the paged-KV regions exist.
    pub paged_kv_enabled: bool,
}

impl Default for Gemma4Geometry {
    fn default() -> Self {
        Gemma4Geometry {
            hidden: 1536,
            n_layers: 35,
            vocab: 262_144,
            eps: 1e-6,
            tied_embeddings: true,
            n_q_heads: 8,
            n_kv_heads: 1,
            head_dim: 256,
            quant: AffineFormat { bits: 4, group: 64 },
            ffn_quant: AffineFormat { bits: 0, group: 0 },
            alt_quant_ffn: false,
            alt_quant_router: false,
            global_head_dim: 512,
            full_partial_rotary: 0.25,
            rope_theta_global: 1.0e6,
            rope_theta_local: 1.0e4,
            sliding_window: 512,
            intermediate: 6144,
            double_wide_mlp: true,
            per_layer_emb_dim: 256,
            num_kv_shared_layers: 20,
            enable_moe: false,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            attention_k_eq_v: false,
            n_global_kv_heads: 0,
            final_softcap: 30.0,
            full_attn_interval: 5,
            max_tokens: 1,
            max_requests: 1,
            max_slots: 1,
            kv_page_size: 32,
            total_pages: 1,
            paged_kv_enabled: false,
        }
    }
}

impl Gemma4Geometry {
    /// Whether the second affine format exists at all.
    #[must_use]
    pub const fn has_alt_quant(&self) -> bool {
        self.ffn_quant.bits != 0 && self.ffn_quant.group != 0
    }

    /// Whether the routed branch exists.
    #[must_use]
    pub const fn is_moe(&self) -> bool {
        self.enable_moe && self.n_experts > 0 && self.experts_per_token > 0
    }

    /// One full-attention layer every interval-th, counting from 1 —
    /// layers 4, 9, 14, … on E2B. Verified against `layer_types`.
    #[must_use]
    pub const fn is_full_attn(&self, layer: u32) -> bool {
        self.full_attn_interval <= 1 || (layer + 1).is_multiple_of(self.full_attn_interval)
    }

    /// The complement of [`is_full_attn`](Self::is_full_attn).
    #[must_use]
    pub const fn is_sliding(&self, layer: u32) -> bool {
        !self.is_full_attn(layer)
    }

    /// Whether this layer takes V from its K projection.
    #[must_use]
    pub const fn k_is_v(&self, layer: u32) -> bool {
        self.attention_k_eq_v && self.is_full_attn(layer)
    }

    /// A layer's KV head count — genuinely per layer on the 26B: 2 on
    /// its full-attention layers against 8 sliding, so one scalar either
    /// over-allocates the cache or truncates it.
    #[must_use]
    pub const fn n_kv_heads_of(&self, layer: u32) -> u32 {
        if self.attention_k_eq_v && self.is_full_attn(layer) && self.n_global_kv_heads > 0 {
            self.n_global_kv_heads
        } else {
            self.n_kv_heads
        }
    }

    /// A layer's head width.
    #[must_use]
    pub const fn head_dim_of(&self, layer: u32) -> u32 {
        if self.is_full_attn(layer) {
            self.global_head_dim
        } else {
            self.head_dim
        }
    }

    /// A layer's rope base.
    #[must_use]
    pub fn rope_theta_of(&self, layer: u32) -> f32 {
        if self.is_full_attn(layer) {
            self.rope_theta_global
        } else {
            self.rope_theta_local
        }
    }

    /// How much of a layer's head rotates.
    #[must_use]
    pub fn rotary_dims_of(&self, layer: u32) -> u32 {
        let hd = self.head_dim_of(layer);
        if self.is_full_attn(layer) {
            (self.full_partial_rotary * hd as f32) as u32
        } else {
            hd
        }
    }

    /// The first KV-shared layer.
    #[must_use]
    pub const fn first_kv_shared(&self) -> u32 {
        self.n_layers.saturating_sub(self.num_kv_shared_layers)
    }

    /// Whether a layer re-attends another's pages.
    #[must_use]
    pub const fn is_kv_shared(&self, layer: u32) -> bool {
        self.num_kv_shared_layers > 0 && layer >= self.first_kv_shared()
    }

    /// Which layer's KV `layer` reads: itself when it owns pages, else
    /// the most recent earlier owning layer of the same attention type.
    /// `None` means the config describes a stack whose shared layers
    /// have no source — a config error, not something to paper over.
    #[must_use]
    pub fn kv_source(&self, layer: u32) -> Option<u32> {
        if !self.is_kv_shared(layer) {
            return Some(layer);
        }
        let want_sliding = self.is_sliding(layer);
        (0..self.first_kv_shared())
            .rev()
            .find(|&j| self.is_sliding(j) == want_sliding)
    }

    /// A layer's dense MLP width — double exactly where the KV is
    /// shared.
    #[must_use]
    pub const fn intermediate_of(&self, layer: u32) -> u32 {
        if self.double_wide_mlp && self.is_kv_shared(layer) {
            2 * self.intermediate
        } else {
            self.intermediate
        }
    }

    /// Layers attending the full context.
    #[must_use]
    pub fn n_full_attn(&self) -> u32 {
        (0..self.n_layers).filter(|&l| self.is_full_attn(l)).count() as u32
    }

    /// Layers that own KV pages — the only ones the KV region sizes for.
    #[must_use]
    pub fn n_kv_owning(&self) -> u32 {
        (0..self.n_layers)
            .filter(|&l| !self.is_kv_shared(l))
            .count() as u32
    }
}

/// The shared-machinery view of a gemma4 shape — what the weight walk,
/// the staging and the scratch sizing read. The KV region this view
/// would imply is WRONG for this family (uniform widths, every layer)
/// and is replaced by `stage_gemma4_kv`; `head_dim` carries the WIDER
/// width so the scratch sizing over-covers the sliding layers.
#[must_use]
pub fn gemma4_decode_geometry(g: &Gemma4Geometry) -> DecodeGeometry {
    DecodeGeometry {
        hidden: g.hidden,
        n_layers: g.n_layers,
        vocab: g.vocab,
        eps: g.eps,
        tied_embeddings: g.tied_embeddings,
        n_q_heads: g.n_q_heads,
        n_kv_heads: g.n_kv_heads,
        head_dim: g.global_head_dim.max(g.head_dim),
        quant: g.quant,
        alt_quant: g.ffn_quant,
        rotary_dims: g.head_dim,
        rope_theta: g.rope_theta_local,
        gdn_k_heads: 0,
        gdn_v_heads: 0,
        gdn_k_dim: 0,
        gdn_v_dim: 0,
        gdn_conv_k: 0,
        gdn_conv_dim: 0,
        gdn_v_total: 0,
        intermediate: if g.double_wide_mlp {
            2 * g.intermediate
        } else {
            g.intermediate
        },
        n_experts: g.n_experts,
        experts_per_token: g.experts_per_token,
        moe_intermediate: g.moe_intermediate,
        mxfp4_experts: false,
        shared_intermediate: 0,
        max_tokens: g.max_tokens,
        max_requests: g.max_requests,
        max_slots: g.max_slots,
        kv_page_size: g.kv_page_size,
        total_pages: g.total_pages,
        paged_kv_enabled: g.paged_kv_enabled,
        full_attn_interval: 1,
        ..DecodeGeometry::default()
    }
}

/// This family's K and N per matvec kind — per LAYER, because the head
/// width, the KV head count and the MLP width all move with it.
#[must_use]
pub fn gemma4_qmv_kn(kind: Kernel, g: &Gemma4Geometry, layer: Option<u32>) -> KN {
    let h = g.hidden;
    let kn = |k, n| KN { k, n };
    let l = layer.unwrap_or(0);
    match kind {
        Kernel::QmvQ => kn(h, g.n_q_heads * g.head_dim_of(l)),
        Kernel::QmvK | Kernel::QmvV => kn(h, g.n_kv_heads_of(l) * g.head_dim_of(l)),
        Kernel::QmvO => kn(g.n_q_heads * g.head_dim_of(l), h),
        Kernel::QmvGate | Kernel::QmvUp => kn(h, g.intermediate_of(l)),
        Kernel::QmvDown => kn(g.intermediate_of(l), h),
        Kernel::G4PleProjGemv => kn(h, g.n_layers * g.per_layer_emb_dim),
        Kernel::G4PleGateGemv => kn(h, g.per_layer_emb_dim),
        Kernel::G4PleProjLayerGemv => kn(g.per_layer_emb_dim, h),
        Kernel::G4Router => kn(h, g.n_experts),
        Kernel::G4ExpertGate | Kernel::G4ExpertUp => kn(h, g.moe_intermediate),
        Kernel::G4ExpertDown => kn(g.moe_intermediate, h),
        Kernel::QmvLmHead | Kernel::LmHeadUntied => kn(h, g.vocab),
        _ => kn(0, 0),
    }
}

/// Build the geometry a config describes, or report why it cannot be
/// built. Refused rather than defaulted, as with every family here.
///
/// # Errors
///
/// [`GeometryRefused`] naming the missing or inconsistent fact.
pub fn gemma4_geometry_from_facts(f: &ModelFacts) -> Result<Gemma4Geometry, GeometryRefused> {
    let refuse = |why: String| Err(GeometryRefused(format!("gemma4: {why}")));
    if f.g4_num_hidden_layers <= 0 {
        return refuse("config carried no text_config shape".to_string());
    }
    if f.g4_full_attn_interval < 0 {
        return refuse(
            "`layer_types` is not a regular interval, which the decode DAG's schedule assumes"
                .to_string(),
        );
    }
    let mut g = Gemma4Geometry {
        n_layers: f.g4_num_hidden_layers.unsigned_abs(),
        hidden: f.g4_hidden_size.unsigned_abs(),
        intermediate: f.g4_intermediate_size.unsigned_abs(),
        n_q_heads: f.g4_num_attention_heads.unsigned_abs(),
        n_kv_heads: f.g4_num_key_value_heads.unsigned_abs(),
        head_dim: f.g4_head_dim.unsigned_abs(),
        global_head_dim: if f.g4_global_head_dim > 0 {
            f.g4_global_head_dim.unsigned_abs()
        } else {
            f.g4_head_dim.unsigned_abs()
        },
        sliding_window: f.g4_sliding_window.unsigned_abs(),
        num_kv_shared_layers: f.g4_num_kv_shared_layers.unsigned_abs(),
        per_layer_emb_dim: f.g4_per_layer_emb_dim.unsigned_abs(),
        double_wide_mlp: f.g4_double_wide_mlp,
        final_softcap: f.g4_final_softcap,
        rope_theta_global: f.g4_rope_theta_full,
        rope_theta_local: f.g4_rope_theta_sliding,
        full_partial_rotary: f.g4_full_partial_rotary,
        enable_moe: f.g4_enable_moe,
        n_experts: f.g4_num_experts.unsigned_abs(),
        experts_per_token: f.g4_experts_per_token.unsigned_abs(),
        moe_intermediate: f.g4_moe_intermediate.unsigned_abs(),
        attention_k_eq_v: f.g4_attention_k_eq_v,
        n_global_kv_heads: f.g4_num_global_kv_heads.unsigned_abs(),
        ..Gemma4Geometry::default()
    };
    if f.g4_full_attn_interval > 0 {
        g.full_attn_interval = f.g4_full_attn_interval.unsigned_abs();
    }
    // Two ways a config can claim a mixture it does not describe. Both
    // refuse rather than fall back to dense: a driver that quietly ran
    // the dense half of a mixture would produce fluent, wrong text.
    if g.enable_moe && (g.n_experts == 0 || g.experts_per_token == 0) {
        return refuse(
            "`enable_moe_block` is set but the config names no experts to route between"
                .to_string(),
        );
    }
    if g.is_moe() && g.moe_intermediate == 0 {
        return refuse(format!(
            "a mixture of {} experts with no `moe_intermediate_size` — the routed projections have no width",
            g.n_experts
        ));
    }
    // A shared layer with no source is a config this driver cannot
    // schedule.
    for layer in 0..g.n_layers {
        if g.is_kv_shared(layer) && g.kv_source(layer).is_none() {
            return refuse(format!(
                "layer {layer} shares KV but no earlier layer of its attention type owns any"
            ));
        }
    }
    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e2b_facts() -> ModelFacts {
        ModelFacts {
            g4_num_hidden_layers: 35,
            g4_hidden_size: 1536,
            g4_intermediate_size: 6144,
            g4_num_attention_heads: 8,
            g4_num_key_value_heads: 1,
            g4_head_dim: 256,
            g4_global_head_dim: 512,
            g4_sliding_window: 512,
            g4_num_kv_shared_layers: 20,
            g4_per_layer_emb_dim: 256,
            g4_full_attn_interval: 5,
            g4_double_wide_mlp: true,
            g4_final_softcap: 30.0,
            g4_rope_theta_full: 1.0e6,
            g4_rope_theta_sliding: 1.0e4,
            g4_full_partial_rotary: 0.25,
            ..ModelFacts::default()
        }
    }

    #[test]
    fn the_e2b_config_lands_on_the_defaults_and_the_axes_move_per_layer() {
        let g = gemma4_geometry_from_facts(&e2b_facts()).expect("the E2B shape");
        assert_eq!(g, Gemma4Geometry::default());
        // Layer 4 is the first full layer: wide head, quarter rotary,
        // the global base.
        assert!(g.is_full_attn(4) && g.is_sliding(3));
        assert_eq!(g.head_dim_of(4), 512);
        assert_eq!(g.rotary_dims_of(4), 128, "a quarter of 512");
        assert_eq!(g.rotary_dims_of(3), 256, "sliding rotates the whole head");
        assert!((g.rope_theta_of(4) - 1.0e6).abs() < 1.0);
        // The KV-shared tail: 15 owners, 20 sharers, and every sharer
        // resolves to an earlier owner of its own attention type.
        assert_eq!(g.first_kv_shared(), 15);
        assert_eq!(g.n_kv_owning(), 15);
        assert_eq!(g.kv_source(16), Some(13), "sliding 16 reads sliding 13");
        assert_eq!(g.kv_source(19), Some(14), "full 19 reads full 14");
        assert_eq!(g.kv_source(3), Some(3), "an owner reads itself");
        // The MLP doubles exactly on the shared range.
        assert_eq!(g.intermediate_of(14), 6144);
        assert_eq!(g.intermediate_of(15), 12288);
        // Per-layer widths flow into the KN table.
        assert_eq!(gemma4_qmv_kn(Kernel::QmvQ, &g, Some(4)).n, 8 * 512);
        assert_eq!(gemma4_qmv_kn(Kernel::QmvQ, &g, Some(3)).n, 8 * 256);
        assert_eq!(gemma4_qmv_kn(Kernel::QmvGate, &g, Some(15)).n, 12288);
        assert_eq!(
            gemma4_qmv_kn(Kernel::G4PleProjGemv, &g, None).n,
            35 * 256,
            "the PLE projection covers every layer's row"
        );
    }

    #[test]
    fn the_26b_axes_diverge_where_the_checkpoint_says_so() {
        let g = Gemma4Geometry {
            n_kv_heads: 8,
            attention_k_eq_v: true,
            n_global_kv_heads: 2,
            enable_moe: true,
            n_experts: 128,
            experts_per_token: 4,
            moe_intermediate: 704,
            ..Gemma4Geometry::default()
        };
        assert!(g.is_moe());
        // Full layers: 2 KV heads and V taken from K; sliding: 8 and a
        // v_proj of their own.
        assert!(g.k_is_v(4) && !g.k_is_v(3));
        assert_eq!(g.n_kv_heads_of(4), 2);
        assert_eq!(g.n_kv_heads_of(3), 8);
        assert_eq!(gemma4_qmv_kn(Kernel::QmvK, &g, Some(4)).n, 2 * 512);
        assert_eq!(gemma4_qmv_kn(Kernel::G4ExpertGate, &g, Some(0)).n, 704);
    }

    #[test]
    fn a_mixture_the_config_does_not_describe_is_refused() {
        let mut f = e2b_facts();
        f.g4_enable_moe = true;
        assert!(
            gemma4_geometry_from_facts(&f).is_err(),
            "enable_moe with no experts named"
        );
        f.g4_num_experts = 128;
        f.g4_experts_per_token = 4;
        assert!(
            gemma4_geometry_from_facts(&f).is_err(),
            "a mixture with no expert width"
        );
        f.g4_moe_intermediate = 704;
        assert!(gemma4_geometry_from_facts(&f).is_ok());
        // An irregular layer_types pattern refuses at the interval.
        let mut f = e2b_facts();
        f.g4_full_attn_interval = -1;
        assert!(gemma4_geometry_from_facts(&f).is_err());
    }
}
