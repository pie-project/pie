//! The query-only fire vocabulary, and the context a bind arm reads a
//! launch through.

use core::ffi::c_void;
use core::ptr::NonNull;

use kernels::Refusal;

use super::facts::Fire;
use kernels_cuda::attn::{AttnWorkspace, KvLayer, MlaLayer, MlaPlan, Plan, Rows};
use kernels_cuda::rope::Yarn;
use kernels_cuda::ssm::{Gdn, Slab};

/// The query-only fire context a bind arm reads. Wraps the concrete
/// [`Fire`]: a `dyn Facts`' defaulted methods let a fact nobody had
/// written down refuse silently.
#[derive(Clone, Copy)]
pub struct Cx<'a> {
    fire: &'a Fire<'a>,
}

impl core::fmt::Debug for Cx<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let rows = self.fire.rows();
        f.debug_struct("Cx").field("layer", &self.fire.layer()).field("rows", &rows).finish()
    }
}

/// `let x = cx.thing()?` where the refusal names the fact.
macro_rules! query {
    ($(#[$m:meta])* $name:ident -> $ty:ty, $what:literal) => {
        $(#[$m])*
        /// # Errors
        pub fn $name(&self) -> Result<$ty, Refusal> {
            self.fire.$name().ok_or(Refusal::Unstated { what: $what })
        }
    };
    ($(#[$m:meta])* $name:ident ( $arg:ident : $at:ty ) -> $ty:ty, $what:literal) => {
        $(#[$m])*
        /// # Errors
        pub fn $name(&self, $arg: $at) -> Result<$ty, Refusal> {
            self.fire.$name($arg).ok_or(Refusal::Absent { what: $what })
        }
    };
}

impl<'a> Cx<'a> {
    /// Wrap one fire's facts.
    #[must_use]
    pub const fn new(fire: &'a Fire<'a>) -> Self {
        Self { fire }
    }

    /// The fire behind the queries, for the GENERATED binder only.
    /// `table::derived_arm` needs the operand list UNINDEXED and the
    /// input/result split; no hand arm may reach past the accessors above.
    pub(crate) const fn fire(&self) -> &'a Fire<'a> {
        self.fire
    }

    query!(
        /// Routes per token. Refuses rather than answering the zero
        /// sentinel: a fanout of zero is a gather addressing no routes.
        experts_per_token -> i32, "experts_per_token"
    );
    query!(
        arg_in(i: usize) -> *mut c_void, "an input operand"
    );
    query!(
        arg_out(i: usize) -> *mut c_void, "an output operand"
    );
    query!(
        weight(i: usize) -> *mut c_void, "a weight"
    );
    query!(
        weight_named(i: usize) -> *mut c_void, "a named weight"
    );
    query!(
        in_width(i: usize) -> i32, "an input's width"
    );
    query!(
        out_width(i: usize) -> i32, "an output's width"
    );
    query!(
        param(i: usize) -> u32, "a statement parameter"
    );
    query!(
        positions -> *const i32, "positions"
    );
    query!(
        token_ids -> *const i32, "the fire's token ids"
    );
    query!(
        vocab -> i32, "the vocabulary size"
    );
    query!(
        sampling_indices -> *const i32, "the rows a sampling gather collects"
    );
    query!(
        /// The per-layer-embedding width.
        ple_dim -> i32, "the per-layer-embedding width"
    );
    query!(
        /// The fire's device-resident peel window, `[start, count]`.
        peel_window -> NonNull<u32>, "a peel window"
    );
    query!(
        head_dim -> i32, "head_dim"
    );
    query!(
        num_q_heads -> i32, "num_q_heads"
    );
    query!(
        num_kv_heads -> i32, "num_kv_heads"
    );
    query!(
        /// The rotary base, one value for every layer. Not [`Cx::theta`].
        rope_theta -> f32, "rope_theta"
    );
    query!(
        /// The rotary base for THIS statement's layer.
        theta -> f32, "this layer's rope theta"
    );
    query!(
        rms_eps -> f32, "the rms epsilon"
    );
    query!(
        final_logit_softcap -> f32, "the final logit soft cap"
    );
    query!(
        rotary_width -> i32, "the rotary width"
    );
    query!(
        yarn -> Yarn, "the YaRN parameters"
    );
    query!(
        kv_layer -> KvLayer, "this layer's kv cache"
    );
    query!(
        /// The widest single request's page count — a page-table row stride.
        max_pages_per_request -> i32, "the widest request's page count"
    );
    query!(
        mla_layer -> MlaLayer, "this layer's latent cache"
    );
    query!(
        mla_plan -> MlaPlan, "the mla plan"
    );
    query!(
        attn_workspace -> AttnWorkspace, "the attention workspace"
    );
    query!(
        sm_scale -> f32, "the attention softmax scale"
    );
    query!(
        first_token -> i32, "the fire's write origin"
    );
    query!(
        num_pages_in_batch -> i32, "the fire's page count"
    );
    query!(
        /// Per-row target page for this fire's KV append.
        w_page_d -> *const u32, "the append's per-row target page"
    );
    query!(
        /// Per-row offset-in-page for the append. [`Cx::w_page_d`]'s pair.
        w_off_d -> *const u32, "the append's per-row page offset"
    );
    query!(
        /// The WNA16 weight's group size, in elements along K. Always
        /// `None`: the driver does not have it. See
        /// [`Fire::wna16_group_size`].
        wna16_group_size -> i32, "a WNA16 group size"
    );
    query!(
        q_out -> *mut c_void, "the fused QKV kernel's query destination"
    );
    query!(
        window_left -> i32, "the fire's sliding-window span"
    );
    query!(
        logits_soft_cap -> f32, "the attention logit soft cap"
    );
    query!(
        /// The LSE scratch the decode dispatch writes.
        lse_out -> *mut f32, "the decode's LSE scratch"
    );
    query!(
        plan -> Plan, "the attention plan"
    );
    query!(
        slab(which: Slab) -> *mut c_void, "a state slab"
    );
    query!(
        /// This fire's linear-attention shape and state addressing.
        gdn -> Gdn, "a linear-attention context"
    );
    query!(
        /// The `i`th auxiliary buffer. Always absent — see [`Fire::aux`].
        aux(i: usize) -> *mut c_void, "an auxiliary buffer"
    );
    query!(
        /// The `i`th result placement. Always absent — see [`Fire::result`].
        result(i: usize) -> *mut c_void, "a result placement"
    );
    query!(
        moe_norm_topk -> bool, "whether the router renormalises its top-k"
    );
    query!(
        moe_routed_scaling -> f32, "the router's routed scaling factor"
    );
    query!(
        in_rows(i: usize) -> i32, "an input's row count"
    );
    query!(
        glu_limit -> f32, "the clamped-GLU limit"
    );
    query!(
        glu_alpha -> f32, "the clamped-GLU alpha"
    );

    /// The PREFILL carve of the attention workspace — a second query and
    /// not an argument, so the decode carve cannot be passed by mistake.
    pub fn attn_prefill_workspace(&self) -> Result<AttnWorkspace, Refusal> {
        self.fire
            .attn_prefill_workspace()
            .ok_or(Refusal::Unstated { what: "the attention prefill workspace" })
    }

    /// Which altup stream ran through the real layer.
    #[must_use]
    pub fn altup_active(&self) -> Option<i32> {
        self.fire.altup_active()
    }

    /// A per-layer constant the model names.
    #[must_use]
    pub fn named_scale(&self, name: &str) -> Option<f32> {
        self.fire.named_scale(name)
    }

    /// A weight the statement names by SUFFIX rather than by position.
    #[must_use]
    pub fn weight_suffixed(&self, suffix: &str) -> Option<*mut c_void> {
        self.fire.weight_suffixed(suffix)
    }

    /// The statement's bias weight, or `None` when it carries none.
    #[must_use]
    pub fn weight_bias(&self) -> Option<*mut c_void> {
        self.fire.weight_bias()
    }

    /// The statement's bias weight, NULL when it states none: that null is
    /// a real deployment (`ConvW::bias == None`), not a lookup that failed.
    pub fn weight_bias_stated(&self) -> *mut c_void {
        self.fire.weight_bias().unwrap_or(core::ptr::null_mut())
    }

    /// Which rows this fire launches. Always answerable.
    #[must_use]
    pub fn rows(&self) -> Rows {
        self.fire.rows()
    }

    /// Which layer this statement belongs to. Always answerable.
    #[must_use]
    pub fn layer(&self) -> usize {
        self.fire.layer()
    }

    /// The engine's cuBLAS handle. Null IS the answer when none exists, and
    /// `Ctx::cublas()` already turns that into a refusal that names it.
    #[must_use]
    pub fn cublas(&self) -> *mut c_void {
        self.fire.cublas()
    }

    /// The statement's per-head width, or `None` for the plain kind:
    /// absence is legal here, so it must not become a `Refusal`.
    #[must_use]
    pub fn per_head_dim(&self) -> Option<i32> {
        self.fire.per_head_dim()
    }

    /// The rotation pairs adjacent elements rather than halves.
    #[must_use]
    pub fn rope_interleaved(&self) -> bool {
        self.fire.rope_interleaved()
    }

    /// The `i`th statement parameter as a float.
    pub fn param_f32(&self, i: usize) -> Result<f32, Refusal> {
        self.param(i).map(f32::from_bits)
    }

    /// The attention half of the fire, whole. One query and not a dozen:
    /// the FA2 arms read twelve [`AttnCtx`] fields, each used by one arm.
    pub fn attn_ctx(&self) -> Result<&'a super::AttnCtx, Refusal> {
        self.fire.attn.ok_or(Refusal::Unstated { what: "the fire's attention context" })
    }

    /// The op join this statement was bound under.
    ///
    /// KEPT WITHOUT A CALLER, and the caller it lost is worth naming: this
    /// said *"its one caller, [`super::attn_plan`], picks a decode plan on
    /// `window_of(..) == -1`"*. That pick is deleted -- a statement names the
    /// schedule it executes now, so `attn_plan` takes no spec and infers
    /// nothing (`.wiki/designs/design-struct.md` §7). The accessor stays
    /// because a `Cx` handing out its own join is not a thing the removal of
    /// one reader makes wrong.
    #[must_use]
    pub const fn spec(&self) -> &'a super::LaunchSpec {
        self.fire.spec
    }
}
