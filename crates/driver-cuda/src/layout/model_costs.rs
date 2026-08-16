//! What a loaded checkpoint costs this device, for the memory planner.
//!
//! Every figure comes from the code that actually allocates it: the arena from
//! [`WorkspaceLayout::cpp_budget_bytes`] (the walk `allocate_full` makes), the
//! KV token cost from the fire path's page geometry. A wrong figure does not
//! crash; it moves bytes between the arena and the KV pool, so the pool ends
//! up too small (fewer contexts) or too large (OOM at the first wide prefill).

use crate::layout::workspace::{WorkspaceLayout, WorkspaceShape};
use model::deployment::Deployment;

use super::memory_planner::ModelCosts;

/// The float half of the attention workspace, as the shell allocates it.
///
/// A constant, not derived from `(n, r)`: the driver allocates a fixed
/// `32 << 20`, so a shape-dependent figure would budget for an allocation
/// nobody makes.
pub const ATTN_FLOAT_WORKSPACE_BYTES: u64 = 32 << 20;

/// One rank's view of what a checkpoint costs.
///
/// It holds a `Deployment` (the loaded row) rather than re-reading
/// `config.json`: a mis-read cost does not crash, it moves bytes between the
/// arena and the pool, so it must read the facts the trace was built from.
pub struct CheckpointCosts {
    dep: Deployment,
    tp_size: i32,
}

impl CheckpointCosts {
    /// Read the costs off a loaded model's deployment for a rank of
    /// `tp_size`.
    #[must_use]
    pub fn new(dep: &Deployment, tp_size: u32) -> Self {
        Self {
            dep: dep.clone(),
            tp_size: i32::try_from(tp_size).unwrap_or(1).max(1),
        }
    }

    /// This rank's KV head count, truncating division floored at one.
    ///
    /// Must match [`kv_geometry`](super::kv_geometry)'s flooring, or the pool
    /// the planner sizes is not the pool the fire builds.
    fn kv_heads(&self) -> u64 {
        let per_rank = i32::try_from(self.dep.shape.kv_heads).unwrap_or(0) / self.tp_size.max(1);
        u64::from(per_rank.max(1).unsigned_abs())
    }

    fn head_dim(&self) -> u64 {
        u64::from(self.dep.shape.head_dim_alloc())
    }

    fn layers(&self) -> u64 {
        u64::from(self.dep.layers)
    }

    /// This checkpoint's MLA cache shape, or `None` for ordinary attention.
    fn mla_geometry(&self) -> Option<super::mla_geometry::MlaGeometry> {
        let model::deployment::KvStyle::Mla {
            kv_lora_rank,
            qk_rope_head_dim,
        } = &self.dep.kv
        else {
            return None;
        };
        if *kv_lora_rank == 0 || *qk_rope_head_dim == 0 {
            return None;
        }
        super::mla_geometry::MlaGeometry::new(
            u32::try_from(self.layers()).unwrap_or(1).max(1),
            1,
            16,
            *kv_lora_rank,
            *qk_rope_head_dim,
            crate::dtype::DType::Bf16,
        )
        .ok()
    }

    /// The widest MLP any layer in the stack asks for.
    ///
    /// A mixture's experts can be wider than the dense `intermediate`, and the
    /// one shared workspace buffer must hold whichever is wider.
    fn max_intermediate(&self) -> i64 {
        i64::from(self.dep.shape.widest_mlp())
    }
}

impl ModelCosts for CheckpointCosts {
    /// Layers × this rank's heads × head dim × 2 bytes × (K and V).
    ///
    /// MLA (DeepSeek/Kimi/GLM5) instead caches a compressed latent plus a rope
    /// key per token: `kv_lora_rank + qk_rope_head_dim`, not
    /// `kv_heads × head_dim × 2`. On DeepSeek-V3 these differ by more than an
    /// order of magnitude, so the dense formula would oversize the pool.
    fn per_kv_token_bytes(&self) -> u64 {
        // DSv4 carries a compressor cache beside its KV, charged per token via
        // `compress_bytes_per_token`. A checkpoint that states no ratios adds
        // zero, which is every family but this one.
        let ratios: &[i32] = match &self.dep.kv {
            model::deployment::KvStyle::CompressedPlane { ratios } => ratios,
            model::deployment::KvStyle::Paged | model::deployment::KvStyle::Mla { .. } => &[],
        };
        let compress = super::compressed_plane_geometry::compress_bytes_per_token(
            ratios,
            u32::try_from(self.head_dim()).unwrap_or(0),
        );
        if let Some(mla) = self.mla_geometry() {
            return mla.bytes_per_token() + compress;
        }
        self.layers() * self.kv_heads() * self.head_dim() * 2 * 2 + compress
    }

    /// Zero: this driver keeps no Quest key envelopes, and
    /// `DriverCapabilities::has_kv_envelopes` says so.
    fn envelope_bytes_per_page(&self) -> u64 {
        0
    }

    /// The GDN conv and recurrent slabs, per slot.
    ///
    /// The conv window is bf16 and the recurrent state is fp32, charged
    /// separately rather than as one width.
    fn state_slot_bytes(&self) -> u64 {
        let Some(r) = self.dep.recurrent.as_ref() else {
            return 0;
        };
        // The strides `gdn_shape` hands the allocator, read off the row's
        // `RecurrentShape`. Slabs exist only for linear layers: a hybrid's
        // full-attention layers keep a KV cache, so charging every layer would
        // over-count by the ratio of its two layer kinds.
        let linear_layers = r.linear_layers.len() as u64;
        linear_layers * (r.conv_stride as u64 + r.state_stride as u64)
    }

    /// The forward workspace at `n` tokens, from the layout that allocates it.
    fn arena_bytes(&self, n: i32, output_rows: i32, mtp_rows: i32) -> u64 {
        let head_dim = i64::from(self.dep.shape.head_dim);
        WorkspaceLayout::new(WorkspaceShape {
            hidden_size: i64::from(self.dep.shape.hidden),
            vocab_size: i64::from(self.dep.shape.vocab),
            head_dim,
            head_dim_kernel: i64::from(self.dep.shape.head_dim_alloc()),
            max_tokens: i64::from(n).max(0),
            max_intermediate: self.max_intermediate(),
            max_hq: i64::from(self.dep.shape.q_heads) * head_dim,
            max_hk: (i64::from(self.dep.shape.kv_heads) / i64::from(self.tp_size.max(1))).max(1)
                * head_dim,
            max_output_rows: i64::from(output_rows).max(0),
            max_mtp_draft_rows: i64::from(mtp_rows).max(0),
        })
        .cpp_budget_bytes()
    }

    fn attn_float_workspace_bytes(&self, _n: i32, _r: i32) -> u64 {
        ATTN_FLOAT_WORKSPACE_BYTES
    }

    /// The eight u32 descriptor arrays a fire uploads, plus the mask.
    ///
    /// Kilobytes, but persistent (`Scratch` keeps them), so charged: a term
    /// left out is bytes handed to the KV pool.
    fn persistent_input_bytes(
        &self,
        n: i32,
        r: i32,
        max_page_refs: i32,
        max_custom_mask_bytes: i32,
    ) -> u64 {
        let n = u64::from(n.max(0).unsigned_abs());
        let r = u64::from(r.max(0).unsigned_abs());
        let pages = u64::from(max_page_refs.max(0).unsigned_abs());
        // token ids, positions, and the two write targets are per token; the
        // qo offsets, page indptr and last-page lengths are per request (plus
        // one for the CSR terminator); page indices are per page reference.
        let per_token = 4 * n;
        let per_request = 3 * (r + 1);
        (per_token + per_request + pages) * 4
            + u64::from(max_custom_mask_bytes.max(0).unsigned_abs())
    }

    /// Zero: this driver binds quantized weights as stored and never
    /// re-quantizes at runtime (`RuntimeQuant::None`).
    fn runtime_quant_scratch_bytes(&self, _n: i32) -> u64 {
        0
    }

    /// True when the row states a recurrent shape with layers to allocate.
    fn has_linear_state(&self) -> bool {
        self.dep
            .recurrent
            .as_ref()
            .is_some_and(|r| !r.linear_layers.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use model::deployment::{Geometry, KvStyle, LayerAttention, RecurrentShape};

    /// Qwen3-0.6B, as the catalog row projects it.
    fn qwen3_0_6b() -> Deployment {
        let mut d = Deployment::empty();
        d.layers = 28;
        d.norm_eps = 1e-6;
        d.shape = Geometry {
            hidden: 1024,
            q_heads: 16,
            kv_heads: 8,
            head_dim: 128,
            head_dim_kernel: 128,
            intermediate: 3072,
            // Dense: no router, so no expert count and no shared FFN.
            moe_intermediate: 0,
            experts_per_token: 0,
            shared_intermediate: 0,
            vocab: 151_936,
        };
        d.attention = (0..28)
            .map(|l| LayerAttention {
                head_dim: 128,
                kv_heads: 8,
                window: -1,
                kv_source: l,
                sm_scale: 1.0 / (128.0_f32).sqrt(),
                rope_theta: 1e6,
                rotary_dim: 0,
                q_gate: false,
            })
            .collect();
        d
    }

    /// The GDN slab strides `gdn_shape` hands the allocator, for a
    /// hybrid whose linear layers are the ones named.
    fn gdn(linear_layers: Vec<u32>) -> RecurrentShape {
        let key_width = 128 * 16;
        let value_width = 128 * 32;
        RecurrentShape {
            linear_layers,
            // The conv window is bf16 and spans the whole packed
            // in-projection; the recurrent state is fp32.
            conv_stride: 4 * (2 * key_width + value_width) * 2,
            state_stride: 32 * 128 * 128 * 4,
            state_elem: 4,
            k_h: 16,
            v_h: 32,
            k_d: 128,
            v_d: 128,
            // Mamba-2's grouping; a GDN slab has none.
            n_groups: 0,
            conv_dim: (2 * key_width + value_width) as i32,
            conv_k: 4,
        }
    }

    #[test]
    fn a_kv_token_costs_what_a_page_is_made_of() {
        let c = CheckpointCosts::new(&qwen3_0_6b(), 1);
        // 28 layers x 8 heads x 128 dim x 2 bytes x (K and V).
        assert_eq!(c.per_kv_token_bytes(), 28 * 8 * 128 * 2 * 2);
    }

    #[test]
    fn a_rank_of_two_holds_half_the_heads() {
        let whole = CheckpointCosts::new(&qwen3_0_6b(), 1).per_kv_token_bytes();
        let half = CheckpointCosts::new(&qwen3_0_6b(), 2).per_kv_token_bytes();
        assert_eq!(half * 2, whole, "the split is exact on eight heads");
    }

    #[test]
    fn a_rank_never_holds_less_than_one_head() {
        // Truncating division floored at one; `kv_geometry` floors the same.
        let c = CheckpointCosts::new(&qwen3_0_6b(), 16);
        assert_eq!(c.per_kv_token_bytes(), 28 * 1 * 128 * 2 * 2);
    }

    #[test]
    fn the_arena_grows_with_the_forward_shape() {
        let c = CheckpointCosts::new(&qwen3_0_6b(), 1);
        let small = c.arena_bytes(128, 0, 0);
        let large = c.arena_bytes(4096, 0, 0);
        assert!(small > 0, "a shape with tokens has an arena");
        assert!(large > small, "more tokens is more arena");
        assert_eq!(
            large,
            WorkspaceLayout::new(WorkspaceShape {
                hidden_size: 1024,
                vocab_size: 151_936,
                head_dim: 128,
                head_dim_kernel: 128,
                max_tokens: 4096,
                max_intermediate: 3072,
                max_hq: 16 * 128,
                max_hk: 8 * 128,
                max_output_rows: 0,
                max_mtp_draft_rows: 0,
            })
            .cpp_budget_bytes()
        );
    }

    #[test]
    fn a_mixtures_workspace_holds_its_widest_layer() {
        let mut d = qwen3_0_6b();
        d.shape.moe_intermediate = 9216;
        let c = CheckpointCosts::new(&d, 1);
        assert_eq!(c.max_intermediate(), 9216);
        assert!(
            c.arena_bytes(4096, 0, 0)
                > CheckpointCosts::new(&qwen3_0_6b(), 1).arena_bytes(4096, 0, 0)
        );
    }

    #[test]
    fn an_mla_checkpoint_is_charged_its_latent_and_not_a_dense_cache() {
        // DeepSeek-V3: 128 kv heads of 192, but the cache holds a 512 latent
        // plus a 64 rope key — ~85x smaller per token than the dense formula.
        let mut d = qwen3_0_6b();
        d.layers = 61;
        d.shape.kv_heads = 128;
        d.shape.head_dim = 192;
        d.shape.head_dim_kernel = 192;
        d.kv = KvStyle::Mla {
            kv_lora_rank: 512,
            qk_rope_head_dim: 64,
        };
        let c = CheckpointCosts::new(&d, 1);

        let latent = 61 * (512 + 64) * 2;
        assert_eq!(c.per_kv_token_bytes(), latent, "the cache is the latent");

        let dense = 61 * 128 * 192 * 2 * 2;
        assert!(
            dense > c.per_kv_token_bytes() * 80,
            "the dense formula would have charged {dense} for {latent}"
        );
    }

    #[test]
    fn a_checkpoint_without_a_latent_keeps_the_dense_formula() {
        let c = CheckpointCosts::new(&qwen3_0_6b(), 1);
        assert_eq!(c.per_kv_token_bytes(), 28 * 8 * 128 * 2 * 2);
    }

    #[test]
    fn a_latent_of_zero_width_is_not_a_latent() {
        // The loader won't build this, but falling back beats a later divide by zero.
        let mut d = qwen3_0_6b();
        d.kv = KvStyle::Mla {
            kv_lora_rank: 0,
            qk_rope_head_dim: 64,
        };
        assert_eq!(
            CheckpointCosts::new(&d, 1).per_kv_token_bytes(),
            28 * 8 * 128 * 2 * 2
        );
        d.kv = KvStyle::Mla {
            kv_lora_rank: 512,
            qk_rope_head_dim: 0,
        };
        assert_eq!(
            CheckpointCosts::new(&d, 1).per_kv_token_bytes(),
            28 * 8 * 128 * 2 * 2
        );
    }

    #[test]
    fn a_v4_checkpoint_pays_for_its_compressor_cache_too() {
        let plain = CheckpointCosts::new(&qwen3_0_6b(), 1).per_kv_token_bytes();

        let mut d = qwen3_0_6b();
        d.kv = KvStyle::CompressedPlane {
            ratios: vec![4, 4, 4],
        };
        let with_compressor = CheckpointCosts::new(&d, 1).per_kv_token_bytes();
        assert!(
            with_compressor > plain,
            "the compressor cache is resident and was charged nothing"
        );
    }

    #[test]
    fn a_dense_model_keeps_no_recurrent_state() {
        let c = CheckpointCosts::new(&qwen3_0_6b(), 1);
        assert!(!c.has_linear_state());
        assert_eq!(c.state_slot_bytes(), 0);
    }

    #[test]
    fn a_stated_recurrence_over_no_layers_is_no_recurrence() {
        let mut d = qwen3_0_6b();
        d.recurrent = Some(gdn(Vec::new()));
        let c = CheckpointCosts::new(&d, 1);
        assert!(!c.has_linear_state());
        assert_eq!(c.state_slot_bytes(), 0);
    }

    #[test]
    fn a_hybrid_charges_for_its_slabs() {
        let mut d = qwen3_0_6b();
        d.recurrent = Some(gdn(vec![0]));
        let c = CheckpointCosts::new(&d, 1);
        assert!(c.has_linear_state());
        // One linear layer of the two; conv is the packed in-projection width.
        let key_width = 128 * 16;
        let value_width = 128 * 32;
        let conv = 4 * (2 * key_width + value_width);
        let state = 32 * 128 * 128;
        assert_eq!(c.state_slot_bytes(), conv * 2 + state * 4);
    }

    #[test]
    fn only_the_linear_layers_of_a_hybrid_carry_slabs() {
        let mut d = qwen3_0_6b();
        d.recurrent = Some(gdn(vec![0, 1, 2, 3]));
        let all_linear = CheckpointCosts::new(&d, 1).state_slot_bytes();

        d.recurrent = Some(gdn(vec![0, 2]));
        let half = CheckpointCosts::new(&d, 1).state_slot_bytes();
        assert_eq!(half * 2, all_linear, "two of four layers carry slabs");
    }

    #[test]
    fn the_workspace_and_the_envelopes_are_what_this_shell_states() {
        let c = CheckpointCosts::new(&qwen3_0_6b(), 1);
        assert_eq!(
            c.attn_float_workspace_bytes(4096, 256),
            ATTN_FLOAT_WORKSPACE_BYTES
        );
        assert_eq!(c.envelope_bytes_per_page(), 0);
        assert_eq!(c.runtime_quant_scratch_bytes(4096), 0);
    }

    #[test]
    fn the_persistent_inputs_are_charged_even_though_they_are_small() {
        let c = CheckpointCosts::new(&qwen3_0_6b(), 1);
        let b = c.persistent_input_bytes(4096, 256, 1024, 0);
        assert!(b > 0, "a term left out is bytes handed to the KV pool");
        assert!(b < 1 << 20, "and it is kilobytes, not megabytes");
    }
}

/// The planner's read side of the on-disk profile cache.
///
/// A missing, unreadable, or unknown-schema cache degrades to "no measurement"
/// ([`Lookup`](super::profile_cache::Lookup)), and the planner falls back to
/// the analytic score.
pub struct DiskProfiles {
    cache: super::profile_cache::ProfileCache,
}

impl DiskProfiles {
    /// The cache at the configured directory, or wherever `cache_path`
    /// derives from the environment.
    ///
    /// # Errors
    ///
    /// No cache directory could be derived.
    pub fn discover(configured_dir: &str) -> Result<Self, super::profile_cache::StoreError> {
        Ok(Self {
            cache: super::profile_cache::ProfileCache::discover(configured_dir)?,
        })
    }
}

impl super::memory_planner::ProfileSource for DiskProfiles {
    fn lookup(&self, key: &super::profile_key::ProfileKey) -> super::memory_planner::ProfileRead {
        self.cache.lookup(key).into()
    }

    fn path(&self) -> String {
        self.cache.path().display().to_string()
    }
}
