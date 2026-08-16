//! The query-only fire vocabulary, and the context a bind arm reads it
//! through.
//!
//! This lived in `kernels-cuda` and had exactly ONE implementor, here.
//! A backend kernels crate cannot know what a trace states -- that is the
//! driver's vocabulary -- so the trait and its wrapper live beside the
//! implementation, and the kernels crate keeps only the DATA types a routine
//! actually takes (`KvLayer`, `MlaLayer`, `KvDType`).

use core::ffi::c_void;
use core::ptr::NonNull;

use kernels::Refusal;

use super::facts::Fire;
use kernels_cuda::attn::{AttnWorkspace, KvLayer, MlaLayer, MlaPlan, Plan, Rows};
use kernels_cuda::rope::Yarn;
use kernels_cuda::ssm::{Gdn, Slab};

/// The query-only fire context a bind arm reads.
///
/// Wraps the CONCRETE [`Fire`] rather than a `dyn Facts`. There was one
/// implementor and the trait's defaults were the problem: seven of its
/// fifty-one methods answered `None` unless overridden, so an arm asking for
/// one always refused and nothing said so. With a concrete receiver, a fact
/// the driver cannot supply has to be written down where it would be
/// supplied.
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

    query!(
        /// The `i`th input operand's address.
        arg_in(i: usize) -> *mut c_void, "an input operand"
    );
    query!(
        /// The `i`th output operand's address.
        arg_out(i: usize) -> *mut c_void, "an output operand"
    );
    query!(
        /// The `i`th positional weight's address.
        weight(i: usize) -> *mut c_void, "a weight"
    );
    query!(
        /// The `i`th named weight's address.
        weight_named(i: usize) -> *mut c_void, "a named weight"
    );
    query!(
        /// The `i`th input's row width, in elements.
        in_width(i: usize) -> i32, "an input's width"
    );
    query!(
        /// The `i`th output's row width, in elements.
        out_width(i: usize) -> i32, "an output's width"
    );
    query!(
        /// The `i`th statement parameter.
        param(i: usize) -> u32, "a statement parameter"
    );
    query!(
        /// The fire's positions, one per row.
        positions -> *const i32, "positions"
    );
    query!(
        /// The fire's token ids, one per row.
        token_ids -> *const i32, "the fire's token ids"
    );
    query!(
        /// How many tokens the vocabulary holds.
        vocab -> i32, "the vocabulary size"
    );
    query!(
        /// The rows a sampling gather collects.
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
        /// Elements per attention head.
        head_dim -> i32, "head_dim"
    );
    query!(
        /// How many query heads.
        num_q_heads -> i32, "num_q_heads"
    );
    query!(
        /// How many key/value heads.
        num_kv_heads -> i32, "num_kv_heads"
    );
    query!(
        /// The rotary base, one value for every layer.
        rope_theta -> f32, "rope_theta"
    );
    query!(
        /// The rotary base for this statement's layer.
        theta -> f32, "this layer's rope theta"
    );
    query!(
        /// The RMS norm epsilon.
        rms_eps -> f32, "the rms epsilon"
    );
    query!(
        /// The logit soft cap. Absent unless the deployment states one.
        final_logit_softcap -> f32, "the final logit soft cap"
    );
    query!(
        /// How many of each head's elements rotate.
        rotary_width -> i32, "the rotary width"
    );
    query!(
        /// The checkpoint's YaRN parameters.
        yarn -> Yarn, "the YaRN parameters"
    );
    query!(
        /// This layer's paged KV cache.
        kv_layer -> KvLayer, "this layer's kv cache"
    );
    query!(
        /// This layer's LATENT cache — MLA's [`KvLayer`].
        mla_layer -> MlaLayer, "this layer's latent cache"
    );
    query!(
        /// The plan [`Prepare::MlaPlan`](kernels::Prepare) built for this fire.
        mla_plan -> MlaPlan, "the mla plan"
    );
    query!(
        /// The attention workspace this fire was given.
        attn_workspace -> AttnWorkspace, "the attention workspace"
    );
    query!(
        /// The softmax scale this fire was planned with.
        sm_scale -> f32, "the attention softmax scale"
    );
    query!(
        /// `write_kv_to_pages`' first-token scalar — the fire's write origin.
        first_token -> i32, "the fire's write origin"
    );
    query!(
        /// The pages this fire's CSR names, which the dequant staging walks.
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
        /// The WNA16 weight's group size, in elements along K.
        ///
        /// `None` until a `Facts` implementation carries it: the driver holds
        /// it on `WeightView::group_size`, and nothing hands it to a `Cx`.
        wna16_group_size -> i32, "a WNA16 group size"
    );
    query!(
        /// The destination the fused QKV kernel writes Q into.
        q_out -> *mut c_void, "the fused QKV kernel's query destination"
    );
    query!(
        /// The sliding-window span this fire attends, in rows.
        window_left -> i32, "the fire's sliding-window span"
    );
    query!(
        /// The attention logit soft cap, `0` for none.
        logits_soft_cap -> f32, "the attention logit soft cap"
    );
    query!(
        /// The LSE scratch the decode dispatch writes.
        lse_out -> *mut f32, "the decode's LSE scratch"
    );
    query!(
        /// The fire's per-request plan arrays.
        plan -> Plan, "the attention plan"
    );
    query!(
        /// One of a gated-delta-net layer's state slabs.
        slab(which: Slab) -> *mut c_void, "a state slab"
    );
    query!(
        /// This fire's linear-attention shape and state addressing.
        gdn -> Gdn, "a linear-attention context"
    );
    query!(
        /// The `i`th auxiliary buffer.
        aux(i: usize) -> *mut c_void, "an auxiliary buffer"
    );
    query!(
        /// The `i`th result placement. NOT [`Cx::arg_out`] — see
        result(i: usize) -> *mut c_void, "a result placement"
    );
    query!(
        /// Whether the router renormalises its top-k weights.
        moe_norm_topk -> bool, "whether the router renormalises its top-k"
    );
    query!(
        /// The router's routed scaling factor.
        moe_routed_scaling -> f32, "the router's routed scaling factor"
    );
    query!(
        /// The `i`th input's row count.
        in_rows(i: usize) -> i32, "an input's row count"
    );
    query!(
        /// gpt-oss's clamped-GLU ceiling, `Source::Ctx("glu_limit")`.
        glu_limit -> f32, "the clamped-GLU limit"
    );
    query!(
        /// The clamped GLU's alpha, `Source::Ctx("glu_alpha")`.
        glu_alpha -> f32, "the clamped-GLU alpha"
    );

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

    /// The engine's cuBLAS handle for this fire — see [`Fire::cublas`].
    ///
    /// Not a `query!` and not an `Option`: null IS the answer when no handle
    /// exists, and `Ctx::cublas()` already turns that into a refusal that
    /// names it. Two refusals for one absence would be one too many.
    #[must_use]
    pub fn cublas(&self) -> *mut c_void {
        self.fire.cublas()
    }

    /// The statement's per-head width, or `None` for the plain kind.
    ///
    /// NOT a `query!`: absence is a legal answer here rather than an unstated
    /// fact, so it must not become a `Refusal`. `Fire::per_head_dim`'s doc has
    /// which statement means which.
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

    /// The attention half of the fire, whole.
    ///
    /// # Why this one query is not a dozen
    ///
    /// The FlashInfer FA2 arms read twelve [`AttnCtx`] fields TOGETHER — the
    /// two plan handles and the full-attention plan beside them, the two
    /// workspaces, the score sink and its CSR, the mask pair, the host CSR
    /// mirrors and `o_out` — and each is used by exactly one arm. Twelve
    /// `query!` lines with one caller each would state the same fact twelve
    /// times without checking any of it.
    ///
    /// # And why handing it over is not §3.3's forbidden surface
    ///
    /// `bind/mod.rs`' FA2 banner argued that a `Cx` able to hand over a plan
    /// cache would be *"a cache a bind body could re-plan, mid-fire, from
    /// inside what is supposed to be a query"*. The reference here is SHARED,
    /// and [`kernels_cuda::attn::fa2::plan::plan_decode`] takes `&mut`: the
    /// re-plan the objection is about is not writable through this. What an
    /// arm does with it is destructure it into a
    /// [`kernels_cuda::attn::fa2::params::DecodePlan`], which is `Copy` and is
    /// the whole of what a launch reads out of a cache.
    ///
    /// # Errors
    ///
    /// [`Refusal::Unstated`] for a fire with no attention half.
    pub fn attn_ctx(&self) -> Result<&'a super::AttnCtx, Refusal> {
        self.fire.attn.ok_or(Refusal::Unstated { what: "the fire's attention context" })
    }

    /// The op join this statement was bound under.
    ///
    /// Reached by exactly one caller, and for one decision:
    /// [`super::attn_plan`] picks between a family's two decode plans on
    /// `window_of(spec, ..) == -1`, and that decision is written once there
    /// rather than re-derived in an arm.
    #[must_use]
    pub const fn spec(&self) -> &'a super::LaunchSpec {
        self.fire.spec
    }
}
