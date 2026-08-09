//! The decode geometry: the numbers a checkpoint's shape decides, in one
//! value.
//!
//! Every consumer of the batch layer sizes something from these — the heap
//! regions, the scratch slots, the launch grids, the kernel names — and the
//! C++ (`model/qwen3_5/geometry.hpp`, generic despite its path) keeps them
//! in one struct because scattering them was the defect: two numbers that
//! travel separately can be half-supplied. The stories are carried on the
//! fields they belong to.
//!
//! [`AffineFormat`] is the sharpest of them: the affine width and group are
//! one fact ("g64/b8 and g128/b4 pack to identical shapes, so the
//! checkpoint's config is the only source"), and a pipeline built for the
//! wrong pair does not fail — it reads the scales against the wrong weights
//! and returns fluent nonsense. Observed: a g64 pipeline over a g32
//! checkpoint answers token 3504, repeated. When width and group were
//! adjacent defaulted parameters, call sites passed one and let the other
//! default — twice — which compiled, bound, dispatched, and lied.

/// The affine quantization's width and group: one fact, never half of it.
///
/// [`kernel_suffix`](Self::kernel_suffix) is the trailing segment shared by
/// every quantized kernel name, spelled here once instead of at every place
/// that builds one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct AffineFormat {
    /// Bits per weight.
    pub bits: u32,
    /// Weights per scale/zero group.
    pub group: u32,
}

impl AffineFormat {
    /// The shipped body format: 4-bit, group 64.
    pub const G64_B4: AffineFormat = AffineFormat { bits: 4, group: 64 };

    /// `_bfloat16_gs_<group>_b_<bits>` — bf16 is not an axis; it is the one
    /// activation dtype this driver instantiates.
    #[must_use]
    pub fn kernel_suffix(self) -> String {
        format!("_bfloat16_gs_{}_b_{}", self.group, self.bits)
    }

    /// Whether the format names anything at all — the `alt_quant` absence
    /// test.
    #[must_use]
    pub const fn is_set(self) -> bool {
        self.bits != 0 && self.group != 0
    }
}

/// The checkpoint-decided shape of the decode step.
///
/// Field meanings and defaults follow the C++; the ones that carry a story
/// keep it.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodeGeometry {
    /// The model width.
    pub hidden: u32,
    /// Decoder layers.
    pub n_layers: u32,
    /// The vocabulary width the head projects onto.
    pub vocab: u32,
    /// The RMS-norm epsilon.
    pub eps: f32,
    /// The embedding and the head are one tensor.
    pub tied_embeddings: bool,
    /// Attention query heads.
    pub n_q_heads: u32,
    /// Attention key/value heads.
    pub n_kv_heads: u32,
    /// Per-head width.
    pub head_dim: u32,
    /// The body's affine format. See [`AffineFormat`].
    pub quant: AffineFormat,
    /// The format the ROUTING projections are in when it is not the body's.
    ///
    /// mlx_lm quantizes per tensor and spares the two that decide where a
    /// token goes: `mlp.gate` and `mlp.shared_expert_gate` are 8-bit inside
    /// a 4-bit checkpoint. Read as 4-bit they still produce finite,
    /// plausible numbers — the router's logits came out at cosine 0.84 to
    /// the reference and the shared gate's at cosine 1.0 with 0.56 of the
    /// magnitude: a mixture routing to almost the right experts and
    /// weighting them wrongly. Unset ([`AffineFormat::is_set`] false) when
    /// the body format covers everything.
    pub alt_quant: AffineFormat,
    /// How many leading dims of each head rotate.
    pub rotary_dims: u32,
    /// The rope base.
    pub rope_theta: f32,
    /// The multimodal rope section split.
    pub mrope_section: [u32; 3],
    /// GDN key heads.
    pub gdn_k_heads: u32,
    /// GDN value heads.
    pub gdn_v_heads: u32,
    /// GDN key width per head.
    pub gdn_k_dim: u32,
    /// GDN value width per head.
    pub gdn_v_dim: u32,
    /// GDN convolution taps.
    pub gdn_conv_k: u32,
    /// GDN convolution channels.
    pub gdn_conv_dim: u32,
    /// GDN value channels in total.
    pub gdn_v_total: u32,
    /// The dense FFN width.
    pub intermediate: u32,
    /// Routed experts; zero is a dense FFN. The difference between a dense
    /// and a routed decoder is these fields, not a different family.
    pub n_experts: u32,
    /// Experts each token routes to.
    pub experts_per_token: u32,
    /// Whether routing weights renormalize over the selected experts.
    /// False routes with weights from the softmax over ALL experts, which
    /// sum to less than one.
    pub norm_topk_prob: bool,
    /// One routed expert's FFN width.
    pub moe_intermediate: u32,
    /// The bank stays in the checkpoint's MXFP4 rather than being
    /// re-quantized at load. It changes what is *bound* — MXFP4 has block
    /// exponents and no zero point, so there is no `.biases` to bind —
    /// which is why a codec belongs in the geometry and not only in a
    /// kernel name. Solved from the staged tensors, never assumed.
    pub mxfp4_experts: bool,
    /// The dense FFN every routed member runs beside the bank, under a
    /// one-scalar-per-token sigmoid gate. Zero only for a routing that has
    /// none.
    pub shared_intermediate: u32,
    /// The widest fire the pools are sized for.
    pub max_tokens: u32,
    /// The most requests one fire may carry.
    pub max_requests: u32,
    /// Recurrent-state slots.
    pub max_slots: u32,
    /// The KV page size.
    pub kv_page_size: u32,
    /// Physical pages in the paged pool.
    pub total_pages: u32,
    /// Whether the paged-KV regions exist at all.
    pub paged_kv_enabled: bool,
    /// Full attention every N layers; an interval of one (or less) makes
    /// every layer qualify. Runtime rather than constant, because the
    /// interval is a property of the checkpoint and this driver is no
    /// longer built around exactly one of them.
    pub full_attn_interval: u32,
}

impl Default for DecodeGeometry {
    /// The C++ defaults: the qwen3.5 dense shape at M=1.
    fn default() -> Self {
        DecodeGeometry {
            hidden: 1024,
            n_layers: 24,
            vocab: 248_320,
            eps: 1e-6,
            tied_embeddings: true,
            n_q_heads: 8,
            n_kv_heads: 2,
            head_dim: 256,
            quant: AffineFormat::G64_B4,
            alt_quant: AffineFormat { bits: 0, group: 0 },
            rotary_dims: 64,
            rope_theta: 1e7,
            mrope_section: [11, 11, 10],
            gdn_k_heads: 16,
            gdn_v_heads: 16,
            gdn_k_dim: 128,
            gdn_v_dim: 128,
            gdn_conv_k: 4,
            gdn_conv_dim: 6144,
            gdn_v_total: 2048,
            intermediate: 3584,
            n_experts: 0,
            experts_per_token: 0,
            norm_topk_prob: true,
            moe_intermediate: 0,
            mxfp4_experts: false,
            shared_intermediate: 0,
            max_tokens: 1,
            max_requests: 1,
            max_slots: 1,
            kv_page_size: 32,
            total_pages: 1,
            paged_kv_enabled: false,
            full_attn_interval: 4,
        }
    }
}

impl DecodeGeometry {
    /// Whether `layer` uses full attention rather than the linear path.
    #[must_use]
    pub fn is_full_attn(&self, layer: u32) -> bool {
        self.full_attn_interval <= 1
            || layer % self.full_attn_interval == self.full_attn_interval - 1
    }

    /// How many layers use full attention.
    #[must_use]
    pub fn full_attn_layers(&self) -> u32 {
        (0..self.n_layers)
            .filter(|&layer| self.is_full_attn(layer))
            .count() as u32
    }

    /// One GDN slot's convolution-state stride, in bytes (fp32 state).
    #[must_use]
    pub fn gdn_conv_stride_bytes(&self) -> u64 {
        u64::from(self.gdn_conv_dim) * u64::from(self.gdn_conv_k) * 4
    }

    /// One GDN slot's recurrent-state stride, in bytes (fp32 state).
    #[must_use]
    pub fn gdn_recurrent_stride_bytes(&self) -> u64 {
        u64::from(self.gdn_v_heads) * u64::from(self.gdn_v_dim) * u64::from(self.gdn_k_dim) * 4
    }

    /// Whether the FFN is a routed mixture.
    #[must_use]
    pub const fn is_moe(&self) -> bool {
        self.n_experts > 0 && self.experts_per_token > 0
    }

    /// Whether a shared expert runs beside the bank.
    #[must_use]
    pub const fn has_shared_expert(&self) -> bool {
        self.is_moe() && self.shared_intermediate > 0
    }

    /// The width one expert's gate/up produce, or the dense width.
    #[must_use]
    pub const fn ffn_width(&self) -> u32 {
        if self.is_moe() {
            self.moe_intermediate
        } else {
            self.intermediate
        }
    }

    /// Whether the routing projections live in a second affine format.
    #[must_use]
    pub const fn has_alt_quant(&self) -> bool {
        self.alt_quant.is_set()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_kernel_suffix_is_the_one_spelling_of_the_format() {
        assert_eq!(AffineFormat::G64_B4.kernel_suffix(), "_bfloat16_gs_64_b_4");
        assert_eq!(
            AffineFormat { bits: 8, group: 32 }.kernel_suffix(),
            "_bfloat16_gs_32_b_8"
        );
    }

    #[test]
    fn an_unset_alt_quant_is_an_absence_not_a_zero_format() {
        let mut geometry = DecodeGeometry::default();
        assert!(!geometry.has_alt_quant());
        geometry.alt_quant = AffineFormat { bits: 8, group: 64 };
        assert!(geometry.has_alt_quant());
        // Half a format is not a format: the pair travels together.
        geometry.alt_quant = AffineFormat { bits: 8, group: 0 };
        assert!(!geometry.has_alt_quant());
    }

    #[test]
    fn the_full_attention_interval_places_one_layer_per_period() {
        let geometry = DecodeGeometry::default();
        // Interval 4: layers 3, 7, 11, ... — one per period, at its end.
        assert!(!geometry.is_full_attn(0));
        assert!(geometry.is_full_attn(3));
        assert!(!geometry.is_full_attn(4));
        assert!(geometry.is_full_attn(7));
        assert_eq!(geometry.full_attn_layers(), 6, "24 layers / 4");

        // An interval of one makes every layer qualify — a family with no
        // linear attention.
        let dense = DecodeGeometry {
            full_attn_interval: 1,
            ..DecodeGeometry::default()
        };
        assert_eq!(dense.full_attn_layers(), dense.n_layers);
    }

    #[test]
    fn a_mixture_is_both_numbers_or_neither() {
        let mut geometry = DecodeGeometry::default();
        assert!(!geometry.is_moe());
        assert_eq!(geometry.ffn_width(), 3584);
        geometry.n_experts = 512;
        assert!(
            !geometry.is_moe(),
            "experts without a per-token count route nothing"
        );
        geometry.experts_per_token = 10;
        geometry.moe_intermediate = 768;
        assert!(geometry.is_moe());
        assert_eq!(geometry.ffn_width(), 768);
        assert!(!geometry.has_shared_expert());
        geometry.shared_intermediate = 512;
        assert!(geometry.has_shared_expert());
    }

    #[test]
    fn the_gdn_strides_are_the_slotted_kernels_arithmetic() {
        let geometry = DecodeGeometry::default();
        assert_eq!(geometry.gdn_conv_stride_bytes(), 6144 * 4 * 4);
        assert_eq!(geometry.gdn_recurrent_stride_bytes(), 16 * 128 * 128 * 4);
    }
}
