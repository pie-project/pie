//! What a loaded checkpoint costs this device, for the memory planner.
//!
//! Every figure comes from the code that actually allocates it: the arena from
//! [`WorkspaceLayout::bytes`] (the walk `allocate_full` makes), the
//! KV token cost from the fire path's page geometry. A wrong figure does not
//! crash; it moves bytes between the arena and the KV pool, so the pool ends
//! up too small (fewer contexts) or too large (OOM at the first wide prefill).

use crate::layout::workspace::{WorkspaceLayout, WorkspaceShape};
use model::deployment::Deployment;

use super::memory_planner::ModelCosts;

/// The float half of ONE attention workspace, as the shell allocates it.
///
/// A constant, not derived from `(n, r)`: `raise_attn_plans` allocates a
/// fixed `32 << 20` whatever the fire's shape, so a shape-dependent figure
/// would budget for an allocation nobody makes.
///
/// # What this still does NOT charge, measured
///
/// A workspace is 32 MiB of float AND 16 MiB of int, both
/// `alloc_device` (`fire::attention_workspace::allocate`), plus two 16 MiB
/// pinned HOST staging slots. Only the float half has ever been in the
/// budget, so the device charge is 2/3 of the truth per workspace, on top of
/// the per-class factor `CheckpointCosts` now applies. On gemma-4-e4b's four
/// workspaces that is 128 MiB charged against 192 MiB resident.
///
/// It is left alone here because the term is NAMED for the float half and
/// three other readers spell it that way (`budget::CudaMemoryPlan`,
/// `layout::rendezvous`, `memory_planner`'s trait). Widening it is a rename,
/// not a number.
///
/// # It is the allocation's number now, not a copy of it
///
/// `fire::launch::raise_attn_plans` wrote `32 << 20` at both of its
/// `AttentionWorkspace::allocate` calls, so the budget and the allocation
/// were two literals that had to agree and nothing made them. They read this
/// constant, which is what "as the shell allocates it" above was always
/// claiming.
pub const ATTN_FLOAT_WORKSPACE_BYTES: u64 = 32 << 20;

/// The int half of ONE attention workspace, as the shell allocates it.
///
/// Also `alloc_device`, and NOT in the planner's budget — see the sibling
/// above. It is named here so the two allocation sites stop spelling it, and
/// so the gap between what is charged and what is taken has a name to be
/// closed by.
pub const ATTN_INT_WORKSPACE_BYTES: u64 = 16 << 20;

/// Plan-staging slots per attention workspace.
///
/// Each is `ATTN_INT_WORKSPACE_BYTES` of PINNED HOST memory
/// (`attention_workspace::ensure_plan_slot`), so this is host pressure, not
/// device budget. Two, because a plan update must not overwrite the slot a
/// launch is still reading.
pub const ATTN_PLAN_STAGING_SLOTS: usize = 2;

/// One element of a causal-conv window, in bytes.
///
/// `u16` unconditionally — `RecurrentStateLayout::conv_slot_stride_bytes`
/// takes no dtype and neither does the kernel that indexes the slab.
const CONV_ELEM_BYTES: u64 = 2;

/// One element of a recurrent state, in bytes.
///
/// bf16, because that is what the two allocations a shell makes force:
/// `fire::launch` and `serve::state::ensure_slots` both call
/// `RecurrentStateCache::allocate_bf16_recurrent`, whose whole body is
/// `force_recurrent_bf16`. The layout's f32 arm is reachable only from its
/// own tests, so charging 4 here would reserve for a pool nobody builds.
const RECURRENT_ELEM_BYTES: u64 = 2;

/// One rank's view of what a checkpoint costs.
///
/// It holds a `Deployment` (the loaded row) rather than re-reading
/// `config.json`: a mis-read cost does not crash, it moves bytes between the
/// arena and the pool, so it must read the facts the trace was built from.
pub struct CheckpointCosts {
    dep: Deployment,
    tp_size: i32,
    /// How many `AttentionWorkspace`s this SKU's fires hold, from
    /// `baker::Baked::attn_workspaces` — one per attention CLASS the lanes
    /// state, not one per driver. See [`ATTN_FLOAT_WORKSPACE_BYTES`].
    attn_workspaces: u64,
}

impl CheckpointCosts {
    /// Read the costs off a loaded model's deployment for a rank of
    /// `tp_size`, holding `attn_workspaces` attention workspaces.
    #[must_use]
    pub fn new(dep: &Deployment, tp_size: u32, attn_workspaces: u32) -> Self {
        Self {
            dep: dep.clone(),
            tp_size: i32::try_from(tp_size).unwrap_or(1).max(1),
            attn_workspaces: u64::from(attn_workspaces.max(1)),
        }
    }

    /// One layer's KV head count on this rank, truncating division floored at
    /// one.
    ///
    /// Must match [`kv_geometry`](super::kv_geometry)'s flooring, or the pool
    /// the planner sizes is not the pool the fire builds.
    fn kv_heads_at(&self, at: &model::deployment::LayerAttention) -> u64 {
        let per_rank = i32::try_from(at.kv_heads).unwrap_or(0) / self.tp_size.max(1);
        u64::from(per_rank.max(1).unsigned_abs())
    }

    // `layers()` STOOD HERE and had one reader, the KV byte product. That
    // product is a per-layer SUM now and the depth is not a factor in it, so
    // the accessor went with the multiplication.

    /// The widest MLP any layer in the stack asks for.
    ///
    /// A mixture's experts can be wider than the dense `intermediate`, and the
    /// one shared workspace buffer must hold whichever is wider.
    fn max_intermediate(&self) -> i64 {
        i64::from(self.dep.shape.widest_mlp)
    }
}

impl ModelCosts for CheckpointCosts {
    /// Summed over the layers that OWN pages: heads × head dim × 2 bytes ×
    /// (K and V).
    ///
    /// A SUM AND NOT A PRODUCT, because a tower may disagree with itself
    /// about both factors and about how many layers pay at all. gemma-4 is
    /// both cases at once: 35 of e4b's 42 layers read 256-wide heads and 7
    /// read 512-wide ones, and its trailing 18 project no k/v and attend
    /// through an earlier layer's pages — so `layers × one width` over-counts
    /// by 75% here and would under-count a tower whose wide layers were the
    /// many. `Deployment::attention` states each layer's own, and this bills
    /// exactly what `KvCacheLayout::plan_slot` allocates: an aliasing layer
    /// gets `LayerSlot::default()` and no tensors.
    ///
    /// The head width is the layer's OWN and not `head_dim_kernel`, for that
    /// reason — the pool reserves `head_dim_at(layer)`. The two are equal for
    /// every catalogued SKU (each states a width the round-up is identity on)
    /// and the pool is the one that decides.
    ///
    /// The MLA and compressed-plane arms STOOD HERE (a latent `kv_lora_rank
    /// + qk_rope_head_dim` per token, and DSv4's per-layer compressor
    /// ratios) and both were unreachable from this function:
    /// `Deployment::of` refuses a plan whose kv rows are not the
    /// `[2, kv_heads * head_dim]` pair before any cost is asked for, so a
    /// checkpoint that needed either arm never reached a planner. The arms
    /// come back with the pool, which is where their geometry belongs.
    /// `kv_geometry::page_bytes_per_layer` is the FORMAT-AWARE sibling of
    /// this walk and takes a `KvCacheFormat` this struct does not hold; it is
    /// the home for a quantised pool's bytes.
    fn per_kv_token_bytes(&self) -> u64 {
        self.dep
            .attention
            .iter()
            .enumerate()
            .filter(|(l, at)| at.kv_source as usize == *l)
            .map(|(_, at)| self.kv_heads_at(at) * u64::from(at.head_dim) * 2 * 2)
            .sum()
    }

    /// Zero: this driver keeps no Quest key envelopes, and
    /// `DriverCapabilities::has_kv_envelopes` says so.
    fn envelope_bytes_per_page(&self) -> u64 {
        0
    }

    /// The GDN conv and recurrent slabs, per slot.
    ///
    /// # The strides are ELEMENTS and this is a BYTE budget
    ///
    /// This summed `conv_stride + state_stride` RAW, as if the row stated
    /// bytes, and `Deployment::recurrent_of` fills both with element counts —
    /// so every recurrent SKU was charged HALF its slab. The planner then
    /// reserved that half and `serve::load` advertised
    /// `free_after_arena / 8 / slot_bytes` slots against it, which is twice
    /// as many slots as the reservation covers: the pool the fire then
    /// allocates runs past its reservation into the KV pages subtracted after
    /// it. Not a crash — the wrong kind of wrong, an over-commit that shows
    /// up as an OOM under load rather than a plan that refuses.
    ///
    /// The widths come from the code that allocates, per this module's rule:
    /// [`RecurrentStateLayout::conv_slot_stride_bytes`] is u16 always, and
    /// [`RecurrentStateLayout::recurrent_slot_stride_bytes`] is f32 or bf16 —
    /// bf16 here, because BOTH live allocations force it
    /// (`fire::launch`'s and `serve::state::ensure_slots`' calls are to
    /// `RecurrentStateCache::allocate_bf16_recurrent`). The f32 arm exists in
    /// the layout and is reached by no shell path; charging it would
    /// over-reserve by the same factor this under-reserved by.
    ///
    /// [`RecurrentStateLayout::conv_slot_stride_bytes`]: super::recurrent_layout::RecurrentStateLayout::conv_slot_stride_bytes
    /// [`RecurrentStateLayout::recurrent_slot_stride_bytes`]: super::recurrent_layout::RecurrentStateLayout::recurrent_slot_stride_bytes
    fn state_slot_bytes(&self) -> u64 {
        let Some(r) = self.dep.recurrent.as_ref() else {
            return 0;
        };
        // Slabs exist only for linear layers: a hybrid's full-attention layers
        // keep a KV cache, so charging every layer would over-count by the
        // ratio of its two layer kinds.
        let linear_layers = r.linear_layers.len() as u64;
        linear_layers
            * (r.conv_stride_elems as u64 * CONV_ELEM_BYTES
                + r.state_stride_elems as u64 * RECURRENT_ELEM_BYTES)
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
        .bytes()
    }

    /// ONE PER CLASS, WHICH IS THE WHOLE CORRECTION HERE.
    ///
    /// Still not derived from `(n, r)` — the allocation is a fixed
    /// `32 << 20` whatever the fire's shape — but there is one of it per
    /// attention class the lanes state, and this charged one per DRIVER.
    /// `baker::Baked::attn_workspaces` counts them; its doc has the gemma-4
    /// arithmetic (four workspaces, 128 MiB of float, 32 MiB budgeted).
    fn attn_float_workspace_bytes(&self, _n: i32, _r: i32) -> u64 {
        ATTN_FLOAT_WORKSPACE_BYTES * self.attn_workspaces
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
                kv_heads: 8,
                kv_source: l,
            })
            .collect();
        d
    }

    /// The two-kind tower, in the same arithmetic-only spirit: half the
    /// layers wide, a quarter of them aliasing another layer's pages.
    fn two_kinds_14() -> Deployment {
        let mut d = dense_28();
        d.layers = 14;
        d.attention = (0..14u32)
            .map(|l| LayerAttention {
                head_dim: if l % 2 == 0 { 128 } else { 256 },
                kv_heads: 8,
                kv_source: if l < 10 { l } else { l - 10 },
            })
            .collect();
        d
    }

    /// The byte budget is a SUM over owning layers, not a product.
    ///
    /// 5 even layers at 128 and 5 odd ones at 256 own pages; the trailing 4
    /// alias and cost nothing. `8 heads * 2 bytes * (K and V)` is 32 bytes a
    /// head-element, so the answer is `32 * (5*128 + 5*256)`. The uniform
    /// product this replaced would have charged `14 * 8 * 128 * 4` — every
    /// aliasing layer billed and every wide one billed narrow.
    #[test]
    fn a_two_kind_tower_is_charged_layer_by_layer() {
        let costs = CheckpointCosts::new(&two_kinds_14(), 1, 1);
        assert_eq!(costs.per_kv_token_bytes(), 32 * (5 * 128 + 5 * 256));
        assert_ne!(costs.per_kv_token_bytes(), 14 * 8 * 128 * 4);
    }

    /// The GDN slab strides `Deployment::recurrent_of` fills, for a hybrid
    /// whose linear layers are the ones named.
    ///
    /// ELEMENTS, and it is the fixture that hid the bug: it built both fields
    /// pre-multiplied (`* 2` for the conv, `* 4` for the state), so the raw
    /// sum under test came out in bytes and its assertion agreed with it. A
    /// fixture that lies the same way as the code under test proves nothing —
    /// this one now states exactly what `recurrent_of` states, `conv_k *
    /// conv_dim` and `v_h * k_d * v_d`, and the widths live only in the
    /// assertions.
    fn gdn(linear_layers: Vec<u32>) -> RecurrentShape {
        let key_width = 128 * 16;
        let value_width = 128 * 32;
        let conv_dim = 2 * key_width + value_width;
        RecurrentShape {
            linear_layers,
            conv_stride_elems: 4 * conv_dim,
            state_stride_elems: 32 * 128 * 128,
            k_h: 16,
            v_h: 32,
            k_d: 128,
            v_d: 128,
            conv_dim: conv_dim as i32,
            conv_k: 4,
        }
    }

    #[test]
    fn a_kv_token_costs_what_a_page_is_made_of() {
        let c = CheckpointCosts::new(&dense_28(), 1, 1);
        // 28 layers x 8 heads x 128 dim x 2 bytes x (K and V).
        assert_eq!(c.per_kv_token_bytes(), 28 * 8 * 128 * 2 * 2);
    }

    #[test]
    fn a_rank_of_two_holds_half_the_heads() {
        let whole = CheckpointCosts::new(&dense_28(), 1, 1).per_kv_token_bytes();
        let half = CheckpointCosts::new(&dense_28(), 2, 1).per_kv_token_bytes();
        assert_eq!(half * 2, whole, "the split is exact on eight heads");
    }

    #[test]
    fn a_rank_never_holds_less_than_one_head() {
        // Truncating division floored at one; `kv_geometry` floors the same.
        let c = CheckpointCosts::new(&dense_28(), 16, 1);
        assert_eq!(c.per_kv_token_bytes(), 28 * 1 * 128 * 2 * 2);
    }

    #[test]
    fn the_arena_grows_with_the_forward_shape() {
        let c = CheckpointCosts::new(&dense_28(), 1, 1);
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
            .bytes()
        );
    }

    #[test]
    fn a_mixtures_workspace_holds_its_widest_layer() {
        let mut d = dense_28();
        d.shape.widest_mlp = 9216;
        let c = CheckpointCosts::new(&d, 1, 1);
        assert_eq!(c.max_intermediate(), 9216);
        assert!(
            c.arena_bytes(4096, 0, 0)
                > CheckpointCosts::new(&dense_28(), 1, 1).arena_bytes(4096, 0, 0)
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
            CheckpointCosts::new(&dense_28(), 1, 1).per_kv_token_bytes(),
            28 * 8 * 128 * 2 * 2
        );
    }

    #[test]
    fn a_dense_model_keeps_no_recurrent_state() {
        let c = CheckpointCosts::new(&dense_28(), 1, 1);
        assert!(!c.has_linear_state());
        assert_eq!(c.state_slot_bytes(), 0);
    }

    #[test]
    fn a_stated_recurrence_over_no_layers_is_no_recurrence() {
        let mut d = dense_28();
        d.recurrent = Some(gdn(Vec::new()));
        let c = CheckpointCosts::new(&d, 1, 1);
        assert!(!c.has_linear_state());
        assert_eq!(c.state_slot_bytes(), 0);
    }

    /// The charge is BYTES, and the fixture's numbers are elements.
    ///
    /// `conv * 2` and `state * 2` are the widths
    /// `RecurrentStateLayout::conv_slot_stride_bytes` and
    /// `recurrent_slot_stride_bytes` use once `allocate_bf16_recurrent` has
    /// forced the state to bf16. The `assert_ne!` is the bug this replaces:
    /// the raw element sum is what the old code charged, and it is half.
    #[test]
    fn a_hybrid_charges_for_its_slabs() {
        let mut d = dense_28();
        d.recurrent = Some(gdn(vec![0]));
        let c = CheckpointCosts::new(&d, 1, 1);
        assert!(c.has_linear_state());
        // One linear layer; conv spans the packed in-projection width.
        let key_width = 128 * 16;
        let value_width = 128 * 32;
        let conv = 4 * (2 * key_width + value_width);
        let state = 32 * 128 * 128;
        assert_eq!(c.state_slot_bytes(), conv * 2 + state * 2);
        assert_ne!(
            c.state_slot_bytes(),
            conv + state,
            "the element sum is not a byte budget"
        );
    }

    /// The planner's figure IS the allocator's, held against it rather than
    /// derived from it.
    ///
    /// `CheckpointCosts` multiplies the row's element strides by a width;
    /// `RecurrentStateLayout` multiplies its own `conv_k * conv_dim` and
    /// `v_h * k_d * v_d` by the same one. Two walks to one number, and the
    /// unit bug this test was written for was exactly the two disagreeing —
    /// nothing else in the tree compared them.
    #[test]
    fn the_planners_slot_is_the_caches_slot() {
        use crate::layout::recurrent_layout::{RecurrentShape as CacheShape, RecurrentStateLayout};

        let linear = vec![0u32, 2, 4];
        let mut d = dense_28();
        d.recurrent = Some(gdn(linear.clone()));
        let r = d.recurrent.clone().expect("just set");

        // What `fire::launch` builds: one slot, no MTP row, bf16 state.
        let is_linear: Vec<bool> = (0..d.layers).map(|l| linear.contains(&l)).collect();
        let layout = RecurrentStateLayout::new(
            &is_linear,
            CacheShape {
                conv_dim: r.conv_dim.unsigned_abs(),
                conv_kernel: r.conv_k.unsigned_abs(),
                v_heads: r.v_h.unsigned_abs(),
                head_k_dim: r.k_d.unsigned_abs(),
                head_v_dim: r.v_d.unsigned_abs(),
                hidden_size: 0,
                max_slots: 1,
                recurrent_is_bf16: true,
            },
        );

        assert_eq!(
            CheckpointCosts::new(&d, 1, 1).state_slot_bytes(),
            layout.bytes_per_slot(),
        );
    }

    #[test]
    fn only_the_linear_layers_of_a_hybrid_carry_slabs() {
        let mut d = dense_28();
        d.recurrent = Some(gdn(vec![0, 1, 2, 3]));
        let all_linear = CheckpointCosts::new(&d, 1, 1).state_slot_bytes();

        d.recurrent = Some(gdn(vec![0, 2]));
        let half = CheckpointCosts::new(&d, 1, 1).state_slot_bytes();
        assert_eq!(half * 2, all_linear, "two of four layers carry slabs");
    }

    #[test]
    fn the_workspace_and_the_envelopes_are_what_this_shell_states() {
        let c = CheckpointCosts::new(&dense_28(), 1, 1);
        assert_eq!(
            c.attn_float_workspace_bytes(4096, 256),
            ATTN_FLOAT_WORKSPACE_BYTES
        );
        assert_eq!(c.envelope_bytes_per_page(), 0);
        assert_eq!(c.runtime_quant_scratch_bytes(4096), 0);
    }

    /// THE WORKSPACE IS CHARGED PER CLASS, and it was charged per driver.
    ///
    /// `raise_attn_plans` allocates one workspace per attention class the
    /// lanes state and `Scratch` holds every one of them for the driver's
    /// life. gemma-4 states four (two decode geometries, two masked), so a
    /// term that answered `ATTN_FLOAT_WORKSPACE_BYTES` whatever the SKU
    /// under-counted resident memory by 96 MiB and handed it to the KV pool.
    ///
    /// The shape arguments stay ignored, which is the OTHER half of the
    /// claim: the allocation is a fixed `32 << 20` and does not move with
    /// `(n, r)`.
    #[test]
    fn the_attention_workspace_is_charged_once_per_class_and_not_once_per_driver() {
        for classes in [1_u32, 2, 4, 7] {
            let c = CheckpointCosts::new(&dense_28(), 1, classes);
            assert_eq!(
                c.attn_float_workspace_bytes(4096, 256),
                ATTN_FLOAT_WORKSPACE_BYTES * u64::from(classes),
                "{classes} classes hold {classes} workspaces",
            );
            assert_eq!(
                c.attn_float_workspace_bytes(1, 1),
                c.attn_float_workspace_bytes(65536, 512),
                "the allocation is fixed; only the COUNT varies",
            );
        }
        // A zero count is a caller that could not measure, not a SKU with no
        // attention: charge one rather than nothing.
        assert_eq!(
            CheckpointCosts::new(&dense_28(), 1, 0).attn_float_workspace_bytes(4096, 256),
            ATTN_FLOAT_WORKSPACE_BYTES
        );
    }

    #[test]
    fn the_persistent_inputs_are_charged_even_though_they_are_small() {
        let c = CheckpointCosts::new(&dense_28(), 1, 1);
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
