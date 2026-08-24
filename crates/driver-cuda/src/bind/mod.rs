//! The fire's prepared state: the fa2 plan caches, the per-fire descriptors
//! every view is cut from, and the view arena itself.
//!
//! # What this module WAS
//!
//! "The executor's first half: binding a flat launch's operands." It held
//! `DispatchPlan` (the derived column joining a lowered launch to the routine
//! that runs it), `bind`/`resolve_arg` (an `Arg` — arena offset, named SSA
//! value, weight name — into a device address), `dispatch` (a 1500-line match
//! from a symbol to a `kernels_cuda` launcher), `run`/`run_captured` (the
//! walk, eager and captured), the `Resolver` trait both walked through, the
//! route table, the claim table and the fact vocabulary an arm asked the
//! driver questions with.
//!
//! Every one of those is deleted. A `model_compiler::program::Program` states
//! its own operands, its own results and the point each statement calls, and
//! `kernels_cuda::points_dispatch` is the one crossing from
//! a stated point to a launcher. There is no launch list to bind, no column to
//! join and no fact to ask for.
//!
//! # What is left, and why each stayed
//!
//! * [`DecodePlan`] / [`PrefillPlan`] — FlashInfer's plan caches. A schedule
//!   is planned on the host from the fire's CSRs; the lane's statements bind
//!   the pointer. Nothing about them was ever the lowering's.
//! * [`AttnCtx`] / [`GdnCtx`] — the per-fire descriptors [`views`] cuts every
//!   KV, mask, score and recurrent view from. `AttnCtx` is a THIRD of what it
//!   was: every field the legacy dispatch read as a loose operand (the plan
//!   handles, the workspaces, `q_out`, `o_out`, `lse_out_d`, the windows, the
//!   scales) is gone, and what remains is exactly what a view reads.
//! * [`views`] — the per-fire view arena, which is how a claim body gets a
//!   driver-owned object.
//! * [`abi`] — the kernel-facing records the pools and workspaces speak.
//! * [`RunRefusal`] — why a walk stopped.

/// The kernel-facing records the pools and workspaces speak.
pub mod abi;
/// The per-fire view arena: the runtime objects this driver answers, built
/// once per fire from `AttnCtx`/`GdnCtx`.
pub mod views;

use std::ffi::c_void;

/// FlashInfer's decode plan cache, owned in Rust. A raw pointer, not a `Box`:
/// [`Self::as_ptr`] is `const`, and a `*mut` keeps this `!Send`.
#[cfg(feature = "_cuda")]
#[derive(Debug)]
pub struct DecodePlan {
    cache: *mut kernels_cuda::attn::fa2::plan::DecodePlanCache,
}

#[cfg(feature = "_cuda")]
impl DecodePlan {
    /// A fresh, unplanned cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: Box::into_raw(Box::new(
                kernels_cuda::attn::fa2::plan::DecodePlanCache::default(),
            )),
        }
    }

    /// The raw handle a dispatch arm passes as the `DecodePlanCache&`.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.cache.cast()
    }

    /// Where the plan's int arrays sit inside the workspace's `int_buffer`.
    pub fn set_int_base(&mut self, bytes: usize) {
        self.get().set_int_base(bytes);
    }

    fn get(&mut self) -> &mut kernels_cuda::attn::fa2::plan::DecodePlanCache {
        // SAFETY: `cache` came from `Box::into_raw` in `new`, is never
        // reassigned, and `&mut self` proves no other reference is live.
        unsafe { &mut *self.cache }
    }

    /// Run FlashInfer's decode planner over the fire's HOST page indptr, inside
    /// the workspace's `begin_plan_update`/`end_plan_update` fence.
    ///
    /// `full_attention_variant` is a PARAMETER and was once a wrapper's
    /// hard-coded `false`. That wrapper — `plan_decode`, zero callers — is
    /// deleted, and the note at its one-time caller (`fire::launch`'s
    /// `raise_attn_plans`) records why it had to be: it "hardcodes
    /// `full_attention_variant = false`, so a stack with NO sliding window
    /// planned the windowed schedule and every decode ran the wrong kernel".
    /// gemma-4 plans TWO decode caches, its layer kinds disagreeing on head
    /// dim, and it is the reason the flag has to be visible here.
    ///
    /// # Panics
    ///
    /// If the planner declines.
    // Safe by design: the view's pointers are the workspace's own.
    #[allow(clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref)]
    pub fn plan_decode_variant(
        &mut self,
        kv_page_indptr_h: &[u32],
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        workspace: crate::bind::abi::AttentionWorkspaceView,
        stream: *mut c_void,
        enable_cuda_graph: bool,
        full_attention_variant: bool,
        window_left: i32,
    ) {
        use kernels_cuda::attn::fa2::plan as fa2;

        let _ = stream;
        // THE CARVE THE PLAN WAS RAISED IN, STAMPED ON THE CACHE.
        //
        // `DecodePlanCache`/`PrefillPlanCache` grew these two pointers when
        // the no-ask migration retired `keys::AttnWorkspaceFloat` and
        // `keys::AttnWorkspaceInt`: a launcher used to ASK the driver for the
        // carve it should accumulate split-KV partials into, and now reads it
        // off the plan it was handed. The field docs say the driver stamps
        // them when it raises the cache -- and nothing did. Every planned
        // dispatch therefore read a null workspace, and the first one to
        // dereference it refused by the name of a buffer three frames down
        // (`v is null`, out of fa2's split-KV merge, because a
        // split-KV decode folds its partials out of the float carve).
        //
        // Stamped BEFORE the planner runs, not after: `plan_decode` reads the
        // cache it is given and a plan that declines still leaves a cache a
        // later fire may look at.
        {
            let cache = self.get();
            cache.int_workspace = workspace.int_buffer;
            cache.float_workspace = workspace.float_buffer;
        }
        let num_requests =
            i32::try_from(kv_page_indptr_h.len() - 1).expect("request count fits i32");
        let device = fa2::plan_device();
        let max_grid_size = fa2::decode_max_grid_size(head_dim, num_q_heads, num_kv_heads);
        let planned = fa2::plan_decode(
            self.get(),
            kv_page_indptr_h,
            num_requests,
            num_q_heads,
            num_kv_heads,
            head_dim,
            page_size,
            kernels_cuda::attn::plan::Workspace {
                float_bytes: workspace.float_bytes,
                int_bytes: workspace.int_bytes,
            },
            &device,
            max_grid_size,
            enable_cuda_graph,
            full_attention_variant,
            // `hnd_layout`: `bind` has no HND deployment.
            false,
            window_left,
        );
        if let fa2::Planned::Declined(why) = planned {
            panic!("flashinfer decode plan: {why}");
        }
    }
}

#[cfg(feature = "_cuda")]
impl Default for DecodePlan {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "_cuda")]
impl Drop for DecodePlan {
    fn drop(&mut self) {
        // SAFETY: `cache` came from `Box::into_raw` in `new` and is dropped
        // exactly once, here.
        drop(unsafe { Box::from_raw(self.cache) });
    }
}

/// FlashInfer's prefill plan cache — [`DecodePlan`]'s twin, owned the same way.
#[cfg(feature = "_cuda")]
#[derive(Debug)]
pub struct PrefillPlan {
    cache: *mut kernels_cuda::attn::fa2::plan::PrefillPlanCache,
}

#[cfg(feature = "_cuda")]
impl PrefillPlan {
    /// A fresh, unplanned cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: Box::into_raw(Box::new(
                kernels_cuda::attn::fa2::plan::PrefillPlanCache::default(),
            )),
        }
    }

    /// The raw handle a dispatch arm passes.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.cache.cast()
    }

    /// Where the plan's int arrays sit inside the workspace's `int_buffer`.
    pub fn set_int_base(&mut self, bytes: usize) {
        self.get().set_int_base(bytes);
    }

    fn get(&mut self) -> &mut kernels_cuda::attn::fa2::plan::PrefillPlanCache {
        // SAFETY: as `DecodePlan::get`.
        unsafe { &mut *self.cache }
    }

    /// The carve, stamped, with NO plan run in it.
    ///
    /// The planless prefill leg is why this exists apart from
    /// [`Self::plan_prefill`], which stamps the same four fields on its way
    /// into the planner: `attention.prefill`'s body carves its own schedule
    /// out of this cache at fire time (`fa2::plan_own_prefill` reads the two
    /// pointers AND the two sizes), so a lane that states only that point
    /// still needs the workspace on the cache — and used to get a zeroed one,
    /// which is a planner told it has no room.
    pub fn stamp_workspace(&mut self, workspace: crate::bind::abi::AttentionWorkspaceView) {
        let cache = self.get();
        cache.int_workspace = workspace.int_buffer;
        cache.float_workspace = workspace.float_buffer;
        cache.int_workspace_bytes = workspace.int_bytes;
        cache.float_workspace_bytes = workspace.float_bytes;
    }

    /// Run FlashInfer's prefill planner over the fire's HOST CSRs, bracketed
    /// by the workspace's plan-update fence. `kv_last_page_lens_h` is accepted
    /// and **not read**: it guards the SM90 route, which this never plans.
    ///
    /// # Panics
    ///
    /// If the planner declines.
    // Safe by design: the view's pointers are the workspace's own.
    #[allow(clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref)]
    pub fn plan_prefill(
        &mut self,
        qo_indptr_h: &[u32],
        kv_page_indptr_h: &[u32],
        kv_last_page_lens_h: &[u32],
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        workspace: crate::bind::abi::AttentionWorkspaceView,
        stream: *mut c_void,
        enable_cuda_graph: bool,
        window_left: i32,
    ) {
        use kernels_cuda::attn::fa2::plan as fa2;

        // THE FIVE FLAGS, HERE, BECAUSE ONE CALLER SETS ALL FIVE THE SAME WAY.
        //
        // `plan_prefill_variant` STOOD BETWEEN THIS AND `fa2::plan_prefill`,
        // taking the flags so "a caller that needs a non-causal plan (the ViT
        // is bidirectional)" could reach them. That caller does not exist —
        // R3 deleted the tower and `serve::encode` is a refusal by name now —
        // so the wrapper had exactly one caller, this function, passing this
        // literal. Two entry points, one behaviour, and the second one's
        // stated reason retired with the tower.
        //
        // The STRUCT stays, and stays named, because its own doc argues for
        // it and the argument is still true: `fa2::plan_prefill` ends in five
        // adjacent positional `bool`s, and `causal_mask` in `hnd_layout`'s
        // slot plans a causal ViT. Building it here keeps that a compile
        // error while removing the door nobody comes through.
        let flags = PrefillPlanFlags {
            full_attention_variant: false,
            hnd_layout: false,
            // TRUE, and this driver states no other kind: every lane it fires
            // is a decoder's.
            causal_mask: true,
            custom_mask: false,
            wants_prefill_score: false,
        };

        let _ = (stream, kv_last_page_lens_h);
        // See `DecodePlan::plan_decode_variant`. The prefill cache carries the
        // two SIZES as well, because the planless leg plans against them.
        {
            let cache = self.get();
            cache.int_workspace = workspace.int_buffer;
            cache.float_workspace = workspace.float_buffer;
            cache.int_workspace_bytes = workspace.int_bytes;
            cache.float_workspace_bytes = workspace.float_bytes;
        }
        let num_requests = i32::try_from(qo_indptr_h.len() - 1).expect("request count fits i32");
        let total_tokens = i32::try_from(*qo_indptr_h.last().expect("a CSR has a last entry"))
            .expect("token count fits i32");
        let device = fa2::plan_device();
        let planned = fa2::plan_prefill(
            self.get(),
            qo_indptr_h,
            kv_page_indptr_h,
            total_tokens,
            num_requests,
            num_q_heads,
            num_kv_heads,
            head_dim,
            page_size,
            kernels_cuda::attn::plan::Workspace {
                float_bytes: workspace.float_bytes,
                int_bytes: workspace.int_bytes,
            },
            &device,
            enable_cuda_graph,
            window_left,
            flags.full_attention_variant,
            flags.hnd_layout,
            flags.causal_mask,
            flags.custom_mask,
            flags.wants_prefill_score,
        );
        if let fa2::Planned::Declined(why) = planned {
            panic!("flashinfer prefill plan: {why}");
        }
    }
}

/// The five booleans `plan_attention_flashinfer_prefill_bf16` took after its
/// numbers. Named rather than positional: `causal_mask` in `hnd_layout`'s
/// slot plans a causal ViT, and a name makes that a compile error.
#[cfg(feature = "_cuda")]
#[derive(Debug, Clone, Copy)]
pub struct PrefillPlanFlags {
    /// `FullAttention` rather than the sliding-window variant.
    pub full_attention_variant: bool,
    /// KV pages laid out `[head, page, dim]` rather than `[page, head, dim]`.
    pub hnd_layout: bool,
    /// A causal mask; **`false` is a bidirectional layer** — a ViT, not a decoder.
    pub causal_mask: bool,
    /// A caller-supplied packed mask, supplied at the dispatch.
    pub custom_mask: bool,
    /// This plan will be dispatched through a score-capturing arm.
    pub wants_prefill_score: bool,
}

#[cfg(feature = "_cuda")]
impl Default for PrefillPlan {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "_cuda")]
impl Drop for PrefillPlan {
    fn drop(&mut self) {
        // SAFETY: as `DecodePlan::drop`.
        drop(unsafe { Box::from_raw(self.cache) });
    }
}

/// The fire's KV/mask/score descriptors: what [`views`] cuts every per-layer
/// view from. Assembled once per fire.
#[cfg(feature = "_cuda")]
#[derive(Debug, Clone)]
pub struct AttnCtx {
    /// One KV pool descriptor per model layer.
    pub layers: Vec<crate::bind::abi::KvCacheLayerView>,
    /// Device page-index CSR.
    pub kv_page_indices_d: *const u32,
    /// Device page indptr.
    pub kv_page_indptr_d: *const u32,
    /// Device last-page lengths.
    pub kv_last_page_lens_d: *const u32,
    /// Device query indptr — the token-rows-per-request CSR.
    pub qo_indptr_d: *const u32,
    /// Requests in the fire (`indptr.len() - 1`).
    pub num_requests: i32,
    /// Pages the fire's CSR names.
    pub num_pages_in_batch: i32,
    /// The widest single request's page count — NOT the batch total: XQA's
    /// `maxNbPagesPerSeq` is a page-table row STRIDE. Host-computed.
    pub max_pages_per_request: i32,
    /// Per-row target page for this fire's KV append.
    pub w_page_d: *const u32,
    /// Per-row offset-in-page for the append.
    pub w_off_d: *const u32,
    /// Per-row validity for the append.
    pub row_valid_d: *const u8,
    /// The custom attention mask, one byte per `(q, kv)`. Published on every
    /// fire; the resident form is plain causal.
    pub mask_d: *const u8,
    /// See [`Self::mask_d`].
    pub mask_indptr_d: *const i32,
    /// The observed-rows CSR the `"attn.score"` view carries.
    pub score_indptr_d: *const i32,
    /// The OBSERVATION window the score sink keeps, parsed once and carried.
    pub score_window: u32,
}

/// The fire's GDN context: the per-layer conv/recurrent state slabs, the
/// request→slot indirection and the head geometry, assembled once per fire.
#[cfg(feature = "_cuda")]
#[derive(Debug, Clone)]
pub struct GdnCtx {
    /// Key heads (compact, pre-GQA-repeat).
    pub k_h: i32,
    /// Value heads.
    pub v_h: i32,
    /// Key head width.
    pub k_d: i32,
    /// Value head width.
    pub v_d: i32,
    /// Conv channels (`2*K_h*K_d + V_h*V_d`).
    pub conv_dim: i32,
    /// Conv window width (`linear_conv_kernel_dim`).
    pub conv_k: i32,
    /// mamba's B/C group count; zero on GDN. On a MAMBA fire `v_h`/`v_d`/`k_d`
    /// read as heads/head_dim/state, so `v_h·k_d·v_d` IS mamba's slab.
    pub n_groups: i32,
    /// Device base of each MODEL layer's conv-state slab (slot 0); else zero.
    pub conv_state: Vec<u64>,
    /// Elements per conv slot (`conv_k * conv_dim`).
    pub conv_stride_elems: i64,
    /// Device base of each recurrent-state slab (slot 0), in the store's dtype.
    pub recurrent_state: Vec<u64>,
    /// Elements per recurrent slot.
    pub state_stride_elems: i64,
    /// Device request→slot ids, one per request in the fire.
    pub slot_ids_d: *const i32,
    /// Whether this fire advances state. True for every class that exists.
    pub write_state: bool,
}

/// Why a fire's walk stopped.
///
/// One flat reason, where this was `RunRefusalKind::{Bind, Dispatch}`: a
/// walk has one crossing now (`baker::fire::Fire::step`), so there is no
/// second half for a refusal to have come from.
#[cfg(feature = "_cuda")]
#[derive(Debug)]
pub struct RunRefusal {
    /// Which step of the lane refused.
    pub step: usize,
    /// The point or symbol that step names.
    pub kernel: String,
    /// The refusal itself, as the shim rendered it.
    pub why: String,
}

#[cfg(feature = "_cuda")]
impl core::fmt::Display for RunRefusal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "step {} (`{}`): {}", self.step, self.kernel, self.why)
    }
}
