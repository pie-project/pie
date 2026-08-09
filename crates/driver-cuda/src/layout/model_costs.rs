//! What a loaded checkpoint costs this device, for the memory planner.
//!
//! [`ModelCosts`](super::memory_planner::ModelCosts) is the planner's model of
//! the checkpoint: how many bytes a KV token takes, how big the arena grows
//! with the forward shape, what a recurrent slot costs. The planner takes it
//! as a parameter rather than computing it, which is what lets the lattice be
//! exercised without a GPU or a checkpoint — and which is why, until now, the
//! only implementation was a test's.
//!
//! This is the production one. Every figure comes from the thing that actually
//! allocates it, never from a second statement of the same layout:
//!
//! - the arena is [`WorkspaceLayout::cpp_budget_bytes`], which is the same
//!   walk `allocate_full` makes. The C++ said that layout twice and they had
//!   drifted by 503 MB on a Qwen3-32B shape; there is one list here and the
//!   planner reads it.
//! - the attention float workspace is the constant the shell allocates.
//! - the KV token cost is the page geometry the fire path builds.
//!
//! A figure this gets wrong does not fail. It moves bytes between the arena
//! and the KV pool, and the symptom is a pool that is too small (fewer
//! contexts than the card can hold) or too large (an out-of-memory at the
//! first wide prefill). That is why each one below names its source.

use model::config::schema::HfConfig;
use crate::layout::workspace::{WorkspaceLayout, WorkspaceShape};

use super::memory_planner::ModelCosts;

/// The float half of the attention workspace, as the shell allocates it.
///
/// A CONSTANT, and stated here rather than derived from `(n, r)` because that
/// is what the driver does: `AttentionWorkspace::allocate(&mut ops, 32 << 20,
/// 16 << 20, 2)`. A planner told a shape-dependent figure would be budgeting
/// for an allocation nobody makes.
pub const ATTN_FLOAT_WORKSPACE_BYTES: u64 = 32 << 20;

/// One rank's view of what a checkpoint costs.
pub struct CheckpointCosts {
    hf: HfConfig,
    tp_size: i32,
}

impl CheckpointCosts {
    /// Read the costs off a loaded model's config for a rank of `tp_size`.
    #[must_use]
    pub fn new(hf: &HfConfig, tp_size: u32) -> Self {
        Self {
            hf: hf.clone(),
            tp_size: i32::try_from(tp_size).unwrap_or(1).max(1),
        }
    }

    /// This rank's KV head count, floored at one.
    ///
    /// Truncating division, which is the same arithmetic
    /// [`kv_geometry`](super::kv_geometry) does — a rank cannot hold a
    /// fraction of a head, and the two have to agree or the pool the planner
    /// sized is not the pool the fire builds.
    fn kv_heads(&self) -> u64 {
        let per_rank = self.hf.num_key_value_heads / self.tp_size.max(1);
        u64::from(per_rank.max(1).unsigned_abs())
    }

    fn head_dim(&self) -> u64 {
        u64::from(self.hf.head_dim_kernel.max(self.hf.head_dim).unsigned_abs())
    }

    fn layers(&self) -> u64 {
        u64::from(self.hf.num_hidden_layers.unsigned_abs())
    }

    /// This checkpoint's MLA cache shape, or `None` for ordinary attention.
    ///
    /// `kv_lora_rank` is the signature: a config that states one caches a
    /// compressed latent instead of K and V. One page and one layer is enough
    /// to ask for `bytes_per_token`, which is per-token and per-layer and does
    /// not depend on the pool's size.
    fn mla_geometry(&self) -> Option<super::mla_geometry::MlaGeometry> {
        let rank = u32::try_from(self.hf.kv_lora_rank).ok().filter(|r| *r > 0)?;
        let rope = u32::try_from(self.hf.qk_rope_head_dim).ok().filter(|d| *d > 0)?;
        super::mla_geometry::MlaGeometry::new(
            u32::try_from(self.layers()).unwrap_or(1).max(1),
            1,
            16,
            rank,
            rope,
            crate::dtype::DType::Bf16,
        )
        .ok()
    }

    /// The widest MLP any layer in the stack asks for.
    ///
    /// `intermediate_size` on a uniform stack; a mixture states its expert
    /// width separately and the workspace has to hold whichever is wider,
    /// because the buffer is one and the layers share it.
    fn max_intermediate(&self) -> i64 {
        let dense = i64::from(self.hf.intermediate_size);
        let moe = i64::from(self.hf.moe_intermediate_size);
        dense.max(moe)
    }
}

impl ModelCosts for CheckpointCosts {
    /// Layers × this rank's heads × head dim × two bytes, for K and for V.
    ///
    /// The same product `capabilities_json` divides the budget by and the same
    /// one `resize_pool` allocates a page from. One geometry, three readers.
    ///
    /// # Except for MLA, which does not cache K and V at all
    ///
    /// DeepSeek/Kimi/GLM5 cache a COMPRESSED LATENT and a rope key per token:
    /// `kv_lora_rank + qk_rope_head_dim`, not `kv_heads × head_dim × 2`. On
    /// DeepSeek-V3's shape those differ by more than an order of magnitude, so
    /// charging the dense formula sizes the pool as if every token cost the
    /// uncompressed cache — and the planner hands the difference to an arena
    /// nothing will use, leaving the pool a fraction of what the card holds.
    ///
    /// `layout::mla_geometry::MlaGeometry::bytes_per_token` is that number, and
    /// its own doc says what it is for: "the number the planner multiplies by
    /// a token budget". It had no caller.
    fn per_kv_token_bytes(&self) -> u64 {
        // DSV4 CARRIES A COMPRESSOR CACHE BESIDE ITS KV, and it is charged
        // per token like the rest. `dsv4_geometry::compress_bytes_per_token`'s
        // own doc says where it belongs: "This is what the memory planner adds
        // on top of the KV cache for a V4 model." It had no caller either.
        //
        // A checkpoint that states no ratios adds zero, which is every family
        // but this one.
        let compress = super::dsv4_geometry::compress_bytes_per_token(
            &self.hf.dsv4_compress_ratios,
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
    /// The strides the fire path derives for `GdnState`: the conv window is
    /// bf16 and the recurrent state is fp32, which is why they are charged
    /// separately rather than as one width.
    fn state_slot_bytes(&self) -> u64 {
        if !self.has_linear_state() {
            return 0;
        }
        // THE SAME TWO STRIDES `gdn_shape` HANDS THE ALLOCATOR, and the same
        // layer count. Getting either wrong here does not fail: it moves bytes
        // between the arena and the KV pool, silently.
        //
        // `conv_dim` is `2 * key_width + value_width`, not one head's width --
        // the conv window spans the whole packed in-projection. And the slabs
        // exist only for LINEAR layers: a hybrid's full-attention layers keep
        // a KV cache instead, so charging every layer would over-count a
        // qwen3.5 hybrid by the ratio of its two layer kinds.
        let key_width = u64::from(self.hf.linear_key_head_dim.unsigned_abs())
            * u64::from(self.hf.linear_num_key_heads.unsigned_abs());
        let value_width = u64::from(self.hf.linear_value_head_dim.unsigned_abs())
            * u64::from(self.hf.linear_num_value_heads.unsigned_abs());
        let conv = u64::from(self.hf.linear_conv_kernel_dim.unsigned_abs())
            * (2 * key_width + value_width);
        let state = u64::from(self.hf.linear_num_value_heads.unsigned_abs())
            * u64::from(self.hf.linear_key_head_dim.unsigned_abs())
            * u64::from(self.hf.linear_value_head_dim.unsigned_abs());
        let linear_layers = if self.hf.layer_types.is_empty() {
            self.layers()
        } else {
            self.hf
                .layer_types
                .iter()
                .filter(|t| *t == "linear_attention")
                .count() as u64
        };
        // The conv window is bf16; the recurrent state is fp32 unless the
        // deployment says otherwise, which is the shell's own default.
        linear_layers * (conv * 2 + state * 4)
    }

    /// The forward workspace at `n` tokens, from the layout that allocates it.
    fn arena_bytes(&self, n: i32, output_rows: i32, mtp_rows: i32) -> u64 {
        let head_dim = i64::from(self.hf.head_dim);
        WorkspaceLayout::new(WorkspaceShape {
            hidden_size: i64::from(self.hf.hidden_size),
            vocab_size: i64::from(self.hf.vocab_size),
            head_dim,
            head_dim_kernel: i64::from(self.hf.head_dim_kernel.max(self.hf.head_dim)),
            max_tokens: i64::from(n).max(0),
            max_intermediate: self.max_intermediate(),
            max_hq: i64::from(self.hf.num_attention_heads) * head_dim,
            max_hk: i64::from(self.hf.num_key_value_heads / self.tp_size.max(1)).max(1) * head_dim,
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
    /// Small — kilobytes against an arena of gigabytes — and charged anyway,
    /// because they are persistent (`Scratch` keeps them) and a planner
    /// that leaves a term out gives those bytes to the KV pool.
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
        // token ids, positions, and the two write targets are per TOKEN; the
        // qo offsets, the page indptr and the last-page lengths are per
        // REQUEST (plus one for the CSR terminator); the page indices are per
        // page reference.
        let per_token = 4 * n;
        let per_request = 3 * (r + 1);
        (per_token + per_request + pages) * 4 + u64::from(max_custom_mask_bytes.max(0).unsigned_abs())
    }

    /// Zero: this driver binds quantized weights AS STORED and never
    /// re-quantizes at runtime, which is what `RuntimeQuant::None` in the load
    /// policy states.
    fn runtime_quant_scratch_bytes(&self, _n: i32) -> u64 {
        0
    }

    /// A layer typed `linear_attention` is the signature, the same test
    /// `capabilities_json` answers `rs_cache_required` with.
    fn has_linear_state(&self) -> bool {
        self.hf.layer_types.iter().any(|t| t == "linear_attention")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen3_0_6b() -> HfConfig {
        HfConfig {
            model_type: "qwen3".to_owned(),
            hidden_size: 1024,
            intermediate_size: 3072,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            head_dim_kernel: 128,
            vocab_size: 151_936,
            ..Default::default()
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
        // Truncating division, floored at one -- a rank cannot hold a
        // fraction of a head, and `kv_geometry` floors the same way.
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
        // The layout is the one that ALLOCATES, so the planner cannot be
        // told a smaller figure than the driver will take.
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
    fn an_mla_checkpoint_is_charged_its_latent_and_not_a_dense_cache() {
        // DeepSeek-V3's shape: 128 kv heads of 192, but the cache holds a
        // 512-wide latent plus a 64-wide rope key. Charging the dense formula
        // is not a rounding error -- it is 85x per token (5,996,544 bytes
        // against 70,272), and the pool the planner sizes from it is that much
        // too small.
        let mut hf = qwen3_0_6b();
        hf.num_hidden_layers = 61;
        hf.num_key_value_heads = 128;
        hf.head_dim = 192;
        hf.head_dim_kernel = 192;
        hf.kv_lora_rank = 512;
        hf.qk_rope_head_dim = 64;
        let c = CheckpointCosts::new(&hf, 1);

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
        // `kv_lora_rank` is the signature, and qwen3 states none.
        let c = CheckpointCosts::new(&qwen3_0_6b(), 1);
        assert_eq!(c.per_kv_token_bytes(), 28 * 8 * 128 * 2 * 2);
    }

    #[test]
    fn a_v4_checkpoint_pays_for_its_compressor_cache_too() {
        let mut hf = qwen3_0_6b();
        hf.kv_lora_rank = 512;
        hf.qk_rope_head_dim = 64;
        let plain = CheckpointCosts::new(&hf, 1).per_kv_token_bytes();

        hf.dsv4_compress_ratios = vec![4, 4, 4];
        let with_compressor = CheckpointCosts::new(&hf, 1).per_kv_token_bytes();
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
    fn a_hybrid_charges_for_its_slabs() {
        let mut hf = qwen3_0_6b();
        hf.layer_types = vec!["linear_attention".to_owned(), "full_attention".to_owned()];
        hf.linear_conv_kernel_dim = 4;
        hf.linear_key_head_dim = 128;
        hf.linear_num_key_heads = 16;
        hf.linear_value_head_dim = 128;
        hf.linear_num_value_heads = 32;
        let c = CheckpointCosts::new(&hf, 1);
        assert!(c.has_linear_state());
        // ONE linear layer of the two, and `conv_dim` is the packed
        // in-projection's width -- the same two strides `gdn_shape` hands the
        // allocator.
        let key_width = 128 * 16;
        let value_width = 128 * 32;
        let conv = 4 * (2 * key_width + value_width);
        let state = 32 * 128 * 128;
        assert_eq!(c.state_slot_bytes(), conv * 2 + state * 4);
    }

    #[test]
    fn only_the_linear_layers_of_a_hybrid_carry_slabs() {
        // A hybrid's full-attention layers keep a KV cache instead, so
        // charging every layer over-counts by the ratio of the two kinds --
        // which the planner then takes out of the KV pool.
        let mut hf = qwen3_0_6b();
        hf.linear_conv_kernel_dim = 4;
        hf.linear_key_head_dim = 128;
        hf.linear_num_key_heads = 16;
        hf.linear_value_head_dim = 128;
        hf.linear_num_value_heads = 32;

        hf.layer_types = vec!["linear_attention".to_owned(); 4];
        let all_linear = CheckpointCosts::new(&hf, 1).state_slot_bytes();

        hf.layer_types = vec![
            "linear_attention".to_owned(),
            "full_attention".to_owned(),
            "linear_attention".to_owned(),
            "full_attention".to_owned(),
        ];
        let half = CheckpointCosts::new(&hf, 1).state_slot_bytes();
        assert_eq!(half * 2, all_linear, "two of four layers carry slabs");
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
/// [`ProfileSource`](super::memory_planner::ProfileSource) is a trait so the
/// lattice can be verified without a file on disk; this is the one that reads
/// the real file. A cache that is missing, unreadable or written at a schema
/// version this build does not know degrades to "no measurement", which is
/// what [`Lookup`](super::profile_cache::Lookup) already expresses and what
/// the planner already handles: it falls back to the analytic score.
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
