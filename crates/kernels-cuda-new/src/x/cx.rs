//! §3.3 — [`Cx`], the query-only fire context.
//!
//! # This is a promotion, not an invention
//!
//! Every method below already exists. `driver-cuda`'s `bind::dispatch_generated`
//! is a ten-thousand-line generated `match` whose arms are written in exactly
//! this vocabulary — `width_of(b, n_in + 0)`, `rotary_width(ctx, spec,
//! layer)`, `attn_plan(a, spec, layer, family)`, `gdn_slab(g, state,
//! "conv_state")`, `is_set(ctx.head_dim)`, `kv_view(attn, layer)` — over
//! `DispatchCtx`, `AttnCtx`, `GdnCtx`, `BoundLaunch`, `LaunchSpec` and
//! `Frame`. The API was always there. It was hidden inside generated code,
//! where no human wrote it and no human could call it.
//!
//! Naming it and exposing it is the whole of this file.
//!
//! # Why query-only is the safety argument
//!
//! A bind body is code that runs per fire, written by whoever writes the
//! kernel. The reason that is safe is that there is nothing here to
//! misbehave on: **no device API, no allocation, no stream mutation, no
//! interior mutability, no `&mut` anywhere.** A bind body can read facts and
//! call its own `fn`. It cannot free a buffer, cannot enqueue on the wrong
//! stream, and cannot leave the context different from how it found it.
//!
//! That is a stronger guarantee than the row world had, where a `Source` was
//! interpreted by generated code with full access to everything.
//!
//! # `Option` replaces `is_set`
//!
//! `DispatchCtx`'s numeric facts use zero as "not set" and the generated
//! guards call `is_set(ctx.head_dim)`. Here an unset fact is `None`, so the
//! `?` in a bind body IS the guard — and a fact that is genuinely zero
//! stops being indistinguishable from a fact nobody stated.
//!
//! # Why a trait
//!
//! §3.3 names `DispatchCtx` and friends as `Cx`'s contents. They live in
//! `driver-cuda`, which depends on this crate, so naming them here is a
//! dependency cycle. [`Facts`] is the query-only vocabulary; the driver
//! implements it over the structs it already has; [`Cx`] is the facade that
//! turns each `Option` into a named [`Refusal`] so a bind body is a chain of
//! `?`.

use core::ffi::c_void;
use core::ptr::NonNull;

use crate::x::contract::Refusal;

/// WHICH ROWS this fire is launching.
///
/// `start`/`count` are the region — `BoundLaunch::rows` — and `total` is
/// the whole fire's lane space, which a `_devwin` launch spans regardless of
/// how many rows its own region serves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rows {
    /// The first row of this region.
    pub start: i32,
    /// How many rows this region serves.
    pub count: i32,
    /// How many rows the whole fire has.
    pub total: i32,
}

/// One layer's paged KV cache, as the launchers take it.
///
/// `driver-cuda`'s `kv_view(attn, layer)`, whose result most CUDA launchers
/// take whole and a few take loose.
#[derive(Clone, Copy, Debug)]
pub struct KvLayer {
    /// The layer's key pages.
    pub k_pages: *mut c_void,
    /// The layer's value pages.
    pub v_pages: *mut c_void,
    /// Rows per page.
    pub page_size: i32,
    /// Elements per head.
    pub head_dim: i32,
    /// How many KV heads the cache holds.
    pub num_kv_heads: i32,
    /// The pages are `[head, page, dim]` rather than `[page, head, dim]`.
    pub hnd: bool,
}

/// The per-request index arrays a paged write needs.
///
/// `AttnCtx`'s device-resident plan arrays. Every one of them is a device
/// pointer the launcher takes loose, which is why no `Source` could ever
/// name them: a `Source` binds values a trace states, and these are
/// assembled by the driver between statements.
#[derive(Clone, Copy, Debug)]
pub struct Plan {
    /// Where each request's query rows begin.
    pub qo_indptr: *const u32,
    /// Which pages each request holds.
    pub kv_page_indices: *const u32,
    /// Where each request's page list begins.
    pub kv_page_indptr: *const u32,
    /// How many rows of the last page each request uses.
    pub kv_last_page_lens: *const u32,
    /// Which rows are live, or null when every row is.
    pub row_valid: *const u8,
    /// How many requests.
    pub requests: i32,
}

/// Which of a gated-delta-net layer's two state slabs.
///
/// `driver-cuda`'s `gdn_slab(g, state, field)` takes the field as a `&str`
/// and matches it against `"conv_state"` and `"recurrent_state"`. There are
/// two, there have always been two, and a `&str` that can be misspelled is
/// the row world's habit surviving into code. Two variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slab {
    /// The short convolution's ring buffer.
    Conv,
    /// The recurrent state.
    Recurrent,
}

/// The YaRN quartet, as a checkpoint states it.
///
/// The four are `DispatchCtx::yarn` plus `yarn_original_max_position`, and
/// they are the ORIGINAL YaRN parameterisation: a scale factor, two
/// dimension-space ramp bounds, and an attention temperature.
///
/// **They are not llama-3's `low_freq_factor`/`high_freq_factor`.** Those
/// are a different scheme with the same arity, and reading one for the other
/// produces a model that runs and is wrong past its training length. The
/// row world wrote that warning as prose beside an unsourced operand; here
/// it is a type with named fields, and a caller that wants llama-3's scheme
/// cannot reach these by accident.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Yarn {
    /// The context scale factor.
    pub factor: f32,
    /// The ramp's high-frequency bound, in rotations.
    pub beta_fast: f32,
    /// The ramp's low-frequency bound, in rotations.
    pub beta_slow: f32,
    /// The attention temperature.
    pub attention_factor: f32,
    /// The position count the checkpoint was trained at.
    pub original_max_position: i32,
}

impl Yarn {
    /// A checkpoint with no YaRN block.
    ///
    /// `factor: 1.0` and not `0.0`, because 1 is what "no scaling" means to
    /// every kernel that reads it: `rope.cu:367`'s guard is
    /// `yarn_factor > 1 && yarn_original_max_position > 0`, so this value
    /// takes the un-ramped branch by arithmetic rather than by a second
    /// flag. A bind uses it where an absent YaRN block is not a refusal.
    pub const NONE: Self = Self {
        factor: 1.0,
        beta_fast: 0.0,
        beta_slow: 0.0,
        attention_factor: 1.0,
        original_max_position: 0,
    };
}

/// The query-only fire vocabulary, implemented by the driver.
///
/// Every method answers `Option`, and every `None` means "nothing stated
/// this". [`Cx`] turns each into a named refusal; nothing else should call
/// this trait directly.
///
/// The default bodies answer `None`, so a driver implements only what its
/// families ask for and adding a fact is one method rather than a breaking
/// change to every impl.
pub trait Facts {
    /// The `i`th INPUT operand's address.
    fn arg_in(&self, i: usize) -> Option<*mut c_void> {
        let _ = i;
        None
    }
    /// The `i`th OUTPUT operand's address.
    fn arg_out(&self, i: usize) -> Option<*mut c_void> {
        let _ = i;
        None
    }
    /// The `i`th positional WEIGHT's address.
    fn weight(&self, i: usize) -> Option<*mut c_void> {
        let _ = i;
        None
    }
    /// The `i`th NAMED weight's address — the statement's `weights[i]`,
    /// resolved through the model's tensor map.
    fn weight_named(&self, i: usize) -> Option<*mut c_void> {
        let _ = i;
        None
    }
    /// The `i`th input's row width, in elements.
    fn in_width(&self, i: usize) -> Option<i32> {
        let _ = i;
        None
    }
    /// The `i`th output's row width, in elements.
    fn out_width(&self, i: usize) -> Option<i32> {
        let _ = i;
        None
    }
    /// Which rows this fire launches.
    fn rows(&self) -> Rows;
    /// Which layer this statement belongs to.
    fn layer(&self) -> usize;
    /// The `i`th statement parameter, as the lowering wrote it.
    fn param(&self, i: usize) -> Option<u32> {
        let _ = i;
        None
    }
    /// The fire's positions, one per row.
    fn positions(&self) -> Option<*const i32> {
        None
    }
    /// The fire's PEEL WINDOW: a device-resident `[start, count]`, or `None`
    /// when this fire has no row split.
    ///
    /// `DispatchCtx::peel_window`, whose own doc says "no text can state
    /// it" — and which is therefore the exact fact the row world had to
    /// leave unsourced. Here it is an ordinary query, and the `_devwin`
    /// kernels take its result as an ordinary parameter.
    fn peel_window(&self) -> Option<NonNull<u32>> {
        None
    }
    /// Elements per attention head.
    fn head_dim(&self) -> Option<i32> {
        None
    }
    /// How many query heads.
    fn num_q_heads(&self) -> Option<i32> {
        None
    }
    /// How many key/value heads.
    fn num_kv_heads(&self) -> Option<i32> {
        None
    }
    /// The rotary base, for a deployment that states one value for every
    /// layer.
    fn rope_theta(&self) -> Option<f32> {
        None
    }
    /// The rotary base for THIS layer.
    ///
    /// An accessor, not a field: the driver decides whether its per-layer
    /// vector falls back, filters or refuses, and the caller's claim is only
    /// that the statement's layer is the index.
    fn theta(&self) -> Option<f32> {
        None
    }
    /// The RMS norm epsilon.
    fn rms_eps(&self) -> Option<f32> {
        None
    }
    /// The rotation pairs adjacent elements rather than halves.
    fn rope_interleaved(&self) -> bool {
        false
    }
    /// How many of each head's elements rotate, for a partial rotation.
    fn rotary_width(&self) -> Option<i32> {
        None
    }
    /// The checkpoint's YaRN parameters.
    fn yarn(&self) -> Option<Yarn> {
        None
    }
    /// This layer's paged KV cache.
    fn kv_layer(&self) -> Option<KvLayer> {
        None
    }
    /// The fire's per-request plan arrays.
    fn plan(&self) -> Option<Plan> {
        None
    }
    /// One of a gated-delta-net layer's state slabs.
    fn slab(&self, which: Slab) -> Option<*mut c_void> {
        let _ = which;
        None
    }
}

/// The query-only fire context a bind body reads.
///
/// A borrow of the driver's [`Facts`] and nothing else. Its methods are
/// [`Facts`]' with the `Option` turned into a named [`Refusal`], so a bind
/// body reads as a list of the facts it needs and refuses on the first one
/// missing — which is the `?` §2.1 promises, replacing the floored-`Div`
/// apologies the `Source` grammar needed.
#[derive(Clone, Copy)]
pub struct Cx<'a> {
    facts: &'a dyn Facts,
}

impl core::fmt::Debug for Cx<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let rows = self.facts.rows();
        f.debug_struct("Cx")
            .field("layer", &self.facts.layer())
            .field("rows", &rows)
            .finish()
    }
}

/// `let x = cx.thing()?` where the refusal names the fact.
macro_rules! query {
    ($(#[$m:meta])* $name:ident -> $ty:ty, $what:literal) => {
        $(#[$m])*
        ///
        /// # Errors
        ///
        /// [`Refusal::Unstated`] when nothing states it.
        pub fn $name(&self) -> Result<$ty, Refusal> {
            self.facts.$name().ok_or(Refusal::Unstated { what: $what })
        }
    };
    ($(#[$m:meta])* $name:ident ( $arg:ident : $at:ty ) -> $ty:ty, $what:literal) => {
        $(#[$m])*
        ///
        /// # Errors
        ///
        /// [`Refusal::Absent`] when the fire did not carry it.
        pub fn $name(&self, $arg: $at) -> Result<$ty, Refusal> {
            self.facts.$name($arg).ok_or(Refusal::Absent { what: $what })
        }
    };
}

impl<'a> Cx<'a> {
    /// Wrap a driver's facts.
    ///
    /// The only constructor, and it takes a shared borrow: there is no way
    /// to build a `Cx` that can write.
    #[must_use]
    pub const fn new(facts: &'a dyn Facts) -> Self {
        Self { facts }
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
        /// The fire's per-request plan arrays.
        plan -> Plan, "the attention plan"
    );
    query!(
        /// One of a gated-delta-net layer's state slabs.
        slab(which: Slab) -> *mut c_void, "a state slab"
    );

    /// Which rows this fire launches. Always answerable.
    #[must_use]
    pub fn rows(&self) -> Rows {
        self.facts.rows()
    }

    /// Which layer this statement belongs to. Always answerable.
    #[must_use]
    pub fn layer(&self) -> usize {
        self.facts.layer()
    }

    /// The rotation pairs adjacent elements rather than halves.
    ///
    /// A `bool` and not an `Option<bool>`: `false` is what a deployment that
    /// says nothing means, and it is what every NeoX-style checkpoint means.
    #[must_use]
    pub fn rope_interleaved(&self) -> bool {
        self.facts.rope_interleaved()
    }

    /// The `i`th statement parameter as a float.
    ///
    /// The lowering writes a float parameter as its bit pattern, which is
    /// the row world's `Source::ParamF32`.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] when the statement has no such parameter.
    pub fn param_f32(&self, i: usize) -> Result<f32, Refusal> {
        self.param(i).map(f32::from_bits)
    }
}
