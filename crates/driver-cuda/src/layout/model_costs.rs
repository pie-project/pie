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

use crate::layout::workspace::{WorkspaceLayout, WorkspaceShape};
use model::deployment::Deployment;

use super::memory_planner::ModelCosts;

/// The float half of the attention workspace, as the shell allocates it.
///
/// A CONSTANT, and stated here rather than derived from `(n, r)` because that
/// is what the driver does: `AttentionWorkspace::allocate(&mut ops, 32 << 20,
/// 16 << 20, 2)`. A planner told a shape-dependent figure would be budgeting
/// for an allocation nobody makes.
pub const ATTN_FLOAT_WORKSPACE_BYTES: u64 = 32 << 20;

/// One rank's view of what a checkpoint costs.
///
/// # Why it holds a `Deployment` and not a config
///
/// It held a cloned `HfConfig` and read forty-seven fields off it: the
/// KV heads, the head dim, the layer count, the MLA lora rank, the
/// linear-attention widths, the DSv4 compression ratios, the
/// `layer_types` array. Every one of those is a fact the loaded row
/// already states, and reading them again here made the planner a
/// SECOND reader of the checkpoint's `config.json` — deciding how big
/// the KV pool is from a document the trace was not built from.
///
/// The failure that arrangement invites is silent by construction: a
/// mis-read cost does not crash, it moves bytes between the arena and
/// the pool. Both allocations succeed and the model just serves fewer
/// tokens than the card can hold.
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

    /// This rank's KV head count, floored at one.
    ///
    /// Truncating division, which is the same arithmetic
    /// [`kv_geometry`](super::kv_geometry) does — a rank cannot hold a
    /// fraction of a head, and the two have to agree or the pool the planner
    /// sized is not the pool the fire builds.
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
    ///
    /// A `match` on the row's stated [`KvStyle`], where it used to be
    /// `kv_lora_rank > 0` on a parsed config — the same "is this MLA"
    /// question asked of a number that happened to be present rather
    /// than of the answer the row gives.
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
    /// `intermediate` on a uniform stack; a mixture states its expert
    /// width separately and the workspace has to hold whichever is wider,
    /// because the buffer is one and the layers share it.
    fn max_intermediate(&self) -> i64 {
        i64::from(self.dep.shape.widest_mlp())
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
        let ratios: &[i32] = match &self.dep.kv {
            model::deployment::KvStyle::Dsv4 { ratios } => ratios,
            model::deployment::KvStyle::Paged | model::deployment::KvStyle::Mla { .. } => &[],
        };
        let compress = super::dsv4_geometry::compress_bytes_per_token(
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
    /// The strides the fire path derives for `GdnState`: the conv window is
    /// bf16 and the recurrent state is fp32, which is why they are charged
    /// separately rather than as one width.
    fn state_slot_bytes(&self) -> u64 {
        let Some(r) = self.dep.recurrent.as_ref() else {
            return 0;
        };
        // THE SAME TWO STRIDES `gdn_shape` HANDS THE ALLOCATOR, and the
        // same layer set. Getting either wrong here does not fail: it
        // moves bytes between the arena and the KV pool, silently.
        //
        // Read off the row's `RecurrentShape` rather than recomputed
        // from six config fields. The old code rebuilt `conv_dim` as
        // `2 * key_width + value_width` here, which is the third place
        // in this workspace that product was written down; the row
        // states it once, as `conv_stride`, and the fire path allocates
        // from that same field.
        //
        // And the slabs exist only for LINEAR layers: a hybrid's
        // full-attention layers keep a KV cache instead, so charging
        // every layer would over-count a qwen3.5 hybrid by the ratio of
        // its two layer kinds.
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
        (per_token + per_request + pages) * 4
            + u64::from(max_custom_mask_bytes.max(0).unsigned_abs())
    }

    /// Zero: this driver binds quantized weights AS STORED and never
    /// re-quantizes at runtime, which is what `RuntimeQuant::None` in the load
    /// policy states.
    fn runtime_quant_scratch_bytes(&self, _n: i32) -> u64 {
        0
    }

    /// A stated recurrent shape is the signature, the same test
    /// `capabilities_json` answers `rs_cache_required` with.
    ///
    /// It used to scan `layer_types` for the string `"linear_attention"`
    /// — the config's word for it, and one this crate had to keep
    /// spelling correctly in three places. The row answers instead:
    /// `recurrent` is `Some` exactly when there are slabs to allocate.
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
    ///
    /// The fixtures below say the same things the `HfConfig` ones did;
    /// what changed is that they say them in the vocabulary the PLANNER
    /// is now given, so a test cannot state a shape the loader could
    /// never hand it. `kv_lora_rank = 512` on a `Paged` deployment used
    /// to be expressible here and was not expressible anywhere else.
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
            moe_intermediate: 0,
            vocab: 151_936,
        };
        d.attention = (0..28)
            .map(|l| LayerAttention {
                head_dim: 128,
                // The stack-wide count, repeated: this row's layers agree,
                // which is what having no per-layer count used to say.
                kv_heads: 8,
                window: -1,
                kv_source: l,
                sm_scale: 1.0 / (128.0_f32).sqrt(),
                rope_theta: 1e6,
                rotary_dim: 0,
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
    fn a_mixtures_workspace_holds_its_widest_layer() {
        // The dense width alone under-sizes a mixture whose experts are
        // wider, and the failure is silent: both allocations succeed and
        // the bytes come out of the KV pool.
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
        // DeepSeek-V3's shape: 128 kv heads of 192, but the cache holds a
        // 512-wide latent plus a 64-wide rope key. Charging the dense formula
        // is not a rounding error -- it is 85x per token (5,996,544 bytes
        // against 70,272), and the pool the planner sizes from it is that much
        // too small.
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
        // `KvStyle::Paged` is the signature, and qwen3 states that.
        let c = CheckpointCosts::new(&qwen3_0_6b(), 1);
        assert_eq!(c.per_kv_token_bytes(), 28 * 8 * 128 * 2 * 2);
    }

    #[test]
    fn a_latent_of_zero_width_is_not_a_latent() {
        // A row cannot state this and the loader will not build it, but
        // dividing by it later would be worse than falling back here.
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
        d.kv = KvStyle::Dsv4 {
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
        let mut d = qwen3_0_6b();
        d.recurrent = Some(gdn(vec![0, 1, 2, 3]));
        let all_linear = CheckpointCosts::new(&d, 1).state_slot_bytes();

        d.recurrent = Some(gdn(vec![0, 2]));
        let half = CheckpointCosts::new(&d, 1).state_slot_bytes();
        assert_eq!(half * 2, all_linear, "two of four layers carry slabs");
    }

    #[test]
    fn the_workspace_and_the_envelopes_are_what_this_shell_states() {
        // Constants, and asserted so a change to either has to be a
        // change to a test that says why.
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
