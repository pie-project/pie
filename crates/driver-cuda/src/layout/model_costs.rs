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
        u64::from(self.dep.shape.head_dim_kernel)
    }

    fn layers(&self) -> u64 {
        u64::from(self.dep.layers)
    }

    /// The widest MLP any layer in the stack asks for.
    ///
    /// A mixture's experts can be wider than the dense `intermediate`, and the
    /// one shared workspace buffer must hold whichever is wider.
    fn max_intermediate(&self) -> i64 {
        i64::from(self.dep.shape.widest_mlp)
    }
}

impl ModelCosts for CheckpointCosts {
    /// Layers × this rank's heads × head dim × 2 bytes × (K and V).
    ///
    /// ONE FORMULA, because there is one pool. The MLA and compressed-plane
    /// arms STOOD HERE — a latent `kv_lora_rank + qk_rope_head_dim` per token
    /// and DSv4's per-layer compressor ratios — and both were unreachable
    /// from this function: `Deployment::of` refuses a plan whose kv rows are
    /// not the `[2, kv_heads * head_dim]` pair before any cost is asked for,
    /// so a checkpoint that needed either arm never reached a planner. The
    /// arms come back with the pool, which is where their geometry belongs.
    fn per_kv_token_bytes(&self) -> u64 {
        self.layers() * self.kv_heads() * self.head_dim() * 2 * 2
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
            head_dim_kernel: i64::from(self.dep.shape.head_dim_kernel),
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

    use model::deployment::{Geometry, LayerAttention, RecurrentShape};

    /// A 28-layer dense tower with a paged cache, the shape every cost
    /// below is arithmetic about. Written out rather than traced: what is
    /// under test is the COST FORMULA, and a formula wants a shape it can
    /// vary one number of at a time.
    fn dense_28() -> Deployment {
        let mut d = Deployment::empty();
        d.layers = 28;
        d.shape = Geometry {
            hidden: 1024,
            q_heads: 16,
            kv_heads: 8,
            head_dim: 128,
            head_dim_kernel: 128,
            widest_mlp: 3072,
            vocab: 151_936,
        };
        d.attention = (0..28)
            .map(|l| LayerAttention {
                head_dim: 128,
                kv_source: l,
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
        let c = CheckpointCosts::new(&dense_28(), 1);
        // 28 layers x 8 heads x 128 dim x 2 bytes x (K and V).
        assert_eq!(c.per_kv_token_bytes(), 28 * 8 * 128 * 2 * 2);
    }

    #[test]
    fn a_rank_of_two_holds_half_the_heads() {
        let whole = CheckpointCosts::new(&dense_28(), 1).per_kv_token_bytes();
        let half = CheckpointCosts::new(&dense_28(), 2).per_kv_token_bytes();
        assert_eq!(half * 2, whole, "the split is exact on eight heads");
    }

    #[test]
    fn a_rank_never_holds_less_than_one_head() {
        // Truncating division floored at one; `kv_geometry` floors the same.
        let c = CheckpointCosts::new(&dense_28(), 16);
        assert_eq!(c.per_kv_token_bytes(), 28 * 1 * 128 * 2 * 2);
    }

    #[test]
    fn the_arena_grows_with_the_forward_shape() {
        let c = CheckpointCosts::new(&dense_28(), 1);
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
        let mut d = dense_28();
        d.shape.widest_mlp = 9216;
        let c = CheckpointCosts::new(&d, 1);
        assert_eq!(c.max_intermediate(), 9216);
        assert!(
            c.arena_bytes(4096, 0, 0)
                > CheckpointCosts::new(&dense_28(), 1).arena_bytes(4096, 0, 0)
        );
    }

    // FOUR TESTS STOOD HERE — an MLA latent charged instead of a dense
    // cache, a zero-width latent falling back, a DSv4 compressor cache
    // charged beside its KV, and the dense formula's own control. All four
    // exercised arms of `per_kv_token_bytes` that R3 deleted, and they were
    // deleted rather than repaired because the arms were UNREACHABLE from
    // this function: `model::deployment::Deployment::of` refuses a plan
    // whose kv rows are not the `[2, kv_heads * head_dim]` pair, so a
    // checkpoint that needed either arm never reached a planner. What the
    // tests actually proved is now
    // `model/tests/rows_are_the_traces.rs::the_pool_refusals_are_the_measured_ones`,
    // which names the SKUs by name. The dense control survives below.

    #[test]
    fn the_dense_formula_is_the_only_formula() {
        assert_eq!(
            CheckpointCosts::new(&dense_28(), 1).per_kv_token_bytes(),
            28 * 8 * 128 * 2 * 2
        );
    }

    #[test]
    fn a_dense_model_keeps_no_recurrent_state() {
        let c = CheckpointCosts::new(&dense_28(), 1);
        assert!(!c.has_linear_state());
        assert_eq!(c.state_slot_bytes(), 0);
    }

    #[test]
    fn a_stated_recurrence_over_no_layers_is_no_recurrence() {
        let mut d = dense_28();
        d.recurrent = Some(gdn(Vec::new()));
        let c = CheckpointCosts::new(&d, 1);
        assert!(!c.has_linear_state());
        assert_eq!(c.state_slot_bytes(), 0);
    }

    #[test]
    fn a_hybrid_charges_for_its_slabs() {
        let mut d = dense_28();
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
        let mut d = dense_28();
        d.recurrent = Some(gdn(vec![0, 1, 2, 3]));
        let all_linear = CheckpointCosts::new(&d, 1).state_slot_bytes();

        d.recurrent = Some(gdn(vec![0, 2]));
        let half = CheckpointCosts::new(&d, 1).state_slot_bytes();
        assert_eq!(half * 2, all_linear, "two of four layers carry slabs");
    }

    #[test]
    fn the_workspace_and_the_envelopes_are_what_this_shell_states() {
        let c = CheckpointCosts::new(&dense_28(), 1);
        assert_eq!(
            c.attn_float_workspace_bytes(4096, 256),
            ATTN_FLOAT_WORKSPACE_BYTES
        );
        assert_eq!(c.envelope_bytes_per_page(), 0);
        assert_eq!(c.runtime_quant_scratch_bytes(4096), 0);
    }

    #[test]
    fn the_persistent_inputs_are_charged_even_though_they_are_small() {
        let c = CheckpointCosts::new(&dense_28(), 1);
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
