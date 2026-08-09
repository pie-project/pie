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

/// How a KV cache stores its elements.
///
/// `driver-cuda`'s `bind::abi::KvCacheScheme`, spelled here for [`KvLayer`]'s
/// reason — `kernels-cuda-new` cannot name a `driver-cuda` type. **The
/// discriminants are the C++ ABI's** and are asserted rather than assumed:
/// `Native = 0` is what `kv_cache_view.hpp` writes, and every other value
/// selects a dequantisation the kernel performs inline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum KvScheme {
    /// Pages hold the model's own element type; no scale tensors apply.
    Native = 0,
    /// One fp8 scale for the whole tensor.
    Fp8PerTensor = 1,
    /// One int8 scale per (token, head).
    Int8PerTokenHead = 2,
    /// One fp8 scale per (token, head).
    Fp8PerTokenHead = 3,
    /// fp4, blocked — `block_size` is the block.
    Fp4Block = 4,
}

/// The element type a KV page actually holds.
///
/// `driver-cuda`'s `dtype::DType`, and **only the variants a KV page can
/// be**. The full enum has twelve; a page is never `Int4Packed` or
/// `Mxfp4Packed` — those are weight representations — and carrying them here
/// would be a vocabulary that admits states the producer cannot reach.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum KvDType {
    /// The model's own bf16.
    Bf16 = 0,
    /// fp16.
    Fp16 = 1,
    /// int8, under a per-tensor or per-token-head scheme.
    Int8 = 3,
    /// fp8 e4m3.
    Fp8E4M3 = 7,
    /// fp8 e5m2.
    Fp8E5M2 = 8,
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
    // ---- the eleven `kv_paged` Half B needs, and why they are FIELDS ----
    //
    // `x::attn::kv_paged`'s four moved host programs use SEVENTEEN accessors
    // of `KvCacheLayerView` and this struct carried six. The other eleven
    // are here rather than as separate `Cx` queries because they are all one
    // struct's fields, read together, by bodies that already hold the whole
    // view — splitting them would make eleven queries that always succeed or
    // fail as one, which is a vocabulary that cannot express a real state.
    //
    // **Order matters and the agent stated it**: `Cx` first, bodies second.
    // Writing those bodies against seventeen loose parameters would mint a
    // signature the query immediately obsoletes, and every call site would
    // carry it.
    /// How the pages are quantised, and whether the scale tensors apply.
    pub scheme: KvScheme,
    /// What a page element actually is — the model's dtype only under
    /// [`KvScheme::Native`].
    pub storage_dtype: KvDType,
    /// The quantisation block, meaningful under [`KvScheme::Fp4Block`].
    pub block_size: i32,
    /// How many pages the layer's arena holds.
    pub num_pages: i32,
    /// Key scales, null under [`KvScheme::Native`].
    pub k_scales: *mut c_void,
    /// Value scales, likewise.
    pub v_scales: *mut c_void,
    /// The bf16 shadow of the key pages, when a dequantised copy exists.
    pub k_bf16_pages: *mut c_void,
    /// The value shadow.
    pub v_bf16_pages: *mut c_void,
    /// Per-page key minimum, for the envelope path.
    pub k_env_min: *mut u16,
    /// Per-page key maximum.
    pub k_env_max: *mut u16,
    /// **A `bool`, not the fields it derives from.**
    ///
    /// `has_envelopes()` is `!k_env_min.is_null() && !k_env_max.is_null()`
    /// and `is_native_bf16()` reads a `native_bf16` field this mirror does
    /// not carry. Both arrive answered, because a predicate handed over as
    /// its inputs is a predicate every caller re-derives — and two callers
    /// deriving it differently is exactly the class this port keeps finding.
    pub has_envelopes: bool,
    /// Storage is the model's own bf16, so no dequantisation step applies.
    ///
    /// **Not `storage_dtype == Bf16`.** `driver-cuda` carries a separate
    /// `native_bf16` flag and reads it directly; whether the two can
    /// disagree is the producer's business, and a mirror that computed one
    /// from the other would be answering a question it was not asked.
    pub is_native_bf16: bool,
}

/// One layer's LATENT cache, as MLA's launchers take it.
///
/// [`KvLayer`]'s sibling and a different shape, not a variant of it: MLA
/// splits a KV entry into a compressed latent (`ckv`) and a RoPE'd key half
/// (`kpe`), so there are two page arrays and two widths where the paged cache
/// has one of each. `driver-cuda`'s `MlaCacheLayerView` is the same five
/// fields; this is the `Cx` spelling of them, for [`KvLayer`]'s reason —
/// `kernels-cuda-new` cannot name a `driver-cuda` type.
#[derive(Clone, Copy, Debug)]
pub struct MlaLayer {
    /// The layer's compressed-latent pages.
    pub ckv_pages: *mut c_void,
    /// The layer's RoPE'd key pages.
    pub kpe_pages: *mut c_void,
    /// Rows per page.
    pub page_size: i32,
    /// The latent rank — `ckv`'s width.
    pub kv_lora_rank: i32,
    /// The RoPE'd half's head dimension — `kpe`'s width.
    pub qk_rope_head_dim: i32,
}

/// The attention workspace a fire was given.
///
/// One allocation the driver hands every attention launcher, carved into
/// per-request scratch by the plan. `driver-cuda`'s
/// `AttentionWorkspaceView`; this is the `Cx` spelling, for [`KvLayer`]'s
/// reason.
///
/// **Two families asked for this independently** — `x/attn.rs`'s MLA arm and
/// `x/xqa.rs`'s decode arm named the same two lines, `attn_workspace` and
/// `sm_scale`, in the same shape. Neither invented it: both rows carried
/// `Source::Attn(..)`, which is the row world's name for exactly this.
#[derive(Clone, Copy, Debug)]
pub struct AttnWorkspace {
    /// The `float` half.
    pub float_buffer: *mut c_void,
    /// How many bytes it holds.
    pub float_bytes: usize,
    /// The `int` half.
    pub int_buffer: *mut c_void,
    /// How many bytes it holds.
    pub int_bytes: usize,
}

/// The plan `Prepare::MlaPlan` built for this fire.
///
/// **[`crate::plan::info::MlaPlanInfo`] plus the two device arenas it
/// indexes.** The info is offsets and extents — `q_indptr_offset`,
/// `num_blks_x`, the rest — and every one of them is an index *into* an
/// arena, so handing back the info alone would be `Facts::slab()`'s defect
/// exactly: an address with no addressing. `ssm` hit that and it cost a
/// `Cx::gdn()`; this one is shaped right at the first ask.
///
/// A HANDLE, not a computation: `crate::plan::mla` already builds it.
#[derive(Clone, Copy, Debug)]
pub struct MlaPlan {
    /// The offsets and extents the scheduler computed.
    pub info: crate::plan::info::MlaPlanInfo,
    /// The `int` arena those offsets index.
    pub int_arena: *mut c_void,
    /// The `float` arena.
    pub float_arena: *mut c_void,
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
///
/// # A SLAB ADDRESS IS TWO HALVES, and this is one of them
///
/// [`Facts::slab`] hands back a layer's **base** and nothing that can index
/// it. Every kernel that takes `state_base` takes `slot_ids` and a stride in
/// the next two argument slots, because the address a request actually
/// touches is
///
/// ```text
/// base + slot_ids[r] * stride
/// ```
///
/// `ssm` is `slab()`'s first caller and found this: *"an address with no
/// addressing"*. The two-variant enum is right and the `spec.state.layer`
/// indexing is right; what was missing was never a third variant. The other
/// half is [`Gdn::slot_ids_d`] with [`Gdn::conv_stride_elems`] or
/// [`Gdn::state_stride_elems`] — **and the stride you take must match the
/// slab you took**, `Conv` with `conv_stride_elems`, `Recurrent` with
/// `state_stride_elems`, because both are `i64` and swapping them is a
/// silent stride error rather than a type error.
///
/// The next person to add a slab kind adds a stride to [`Gdn`] in the same
/// change, or repeats this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slab {
    /// The short convolution's ring buffer.
    Conv,
    /// The recurrent state.
    Recurrent,
}

/// A linear-attention fire's shape and its state addressing.
///
/// `GdnCtx`, which `Fire` already borrows, named. Eighteen of `ssm`'s
/// twenty-seven rows sourced at least one operand from `Source::Gdn(..)`,
/// which is the row world spelling one context through eleven separate
/// string keys; here it is one query returning one value, and a family that
/// wants three of the eleven pays for one lookup.
///
/// **It is one struct for two architectures, deliberately.** `GdnCtx`'s own
/// doc gives the mapping: when a fire is MAMBA rather than GDN, `v_h` is
/// `num_heads`, `v_d` is `head_dim` and `k_d` is `state_size`, so the state
/// stride `v_h·k_d·v_d` reads `heads·state·head_dim` — which IS mamba's
/// slab. The two shapes share one context because they share one arithmetic,
/// not because nobody separated them.
#[derive(Clone, Copy, Debug)]
pub struct Gdn {
    /// Key heads, compact — before any GQA repeat.
    pub k_h: i32,
    /// Value heads. Mamba's `num_heads`.
    pub v_h: i32,
    /// Key head width. Mamba's `state_size`.
    pub k_d: i32,
    /// Value head width. Mamba's `head_dim`.
    pub v_d: i32,
    /// Conv channels, `2·k_h·k_d + v_h·v_d`.
    pub conv_dim: i32,
    /// Conv window width.
    pub conv_k: i32,
    /// Mamba's B/C group count. **Zero on a GDN family**, and zero is the
    /// honest answer rather than an absence — a GDN fire has no groups, it
    /// does not fail to state how many.
    pub n_groups: i32,
    /// Elements per conv slot, `conv_k · conv_dim`. Pairs with
    /// [`Slab::Conv`].
    pub conv_stride_elems: i64,
    /// Elements per recurrent slot. Pairs with [`Slab::Recurrent`].
    pub state_stride_elems: i64,
    /// Device request→slot ids, one per request in the fire.
    ///
    /// The indexing half of every slab address — see [`Slab`].
    pub slot_ids_d: *const i32,
    /// Whether this fire advances state.
    pub write_state: bool,
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
    /// The fire's TOKEN IDS, one per row.
    ///
    /// `DispatchCtx::token_ids`, which the row world sourced as
    /// `Source::Ctx("token_ids")`. The first launch of every fire —
    /// `layout::embed_bf16` — reads it, so this is the least optional query
    /// on the trait and is still `Option`, because the trait is one shape
    /// for every family and a norm kernel has no tokens.
    fn token_ids(&self) -> Option<*const i32> {
        None
    }
    /// How many tokens the vocabulary holds.
    ///
    /// `DispatchCtx::vocab`, row-sourced as `Source::Ctx("vocab")`. An
    /// embedding gather bounds-checks against it and an LM head's output
    /// width is it.
    fn vocab(&self) -> Option<i32> {
        None
    }
    /// The rows a sampling gather collects, or `None` when the fire gathers
    /// every row.
    ///
    /// `DispatchCtx::sampling_indices`, row-sourced as
    /// `Source::SamplingIndices` — a source spelling with exactly one
    /// consumer, which is the shape §0 means by "one small declaration
    /// serves the readers that cannot call". Under fn-world it is a query
    /// and the grammar entry retires with the row.
    fn sampling_indices(&self) -> Option<*const i32> {
        None
    }
    /// The per-layer-embedding width, for a checkpoint that carries PLE
    /// tables beside its weights.
    ///
    /// `DispatchCtx::ple_dim`, row-sourced as `Source::Ctx("ple_dim")` and,
    /// for the layer count, `Div(Width(In(0)), CtxNonZero("ple_dim"))`.
    /// **`CtxNonZero` is where the row grammar already said that zero means
    /// absent**; `Option` is that statement in the type system, so a driver
    /// impl returns `None` for zero rather than handing back a divisor that
    /// faults.
    fn ple_dim(&self) -> Option<i32> {
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
    /// The logit soft cap, or `None` where the deployment states none.
    ///
    /// `Source::CtxNonZero("final_logit_softcap")` is where the row grammar
    /// already said that zero means absent, and this is that statement moved
    /// into the type system: an impl returns `None` for zero rather than
    /// handing back a cap of zero, which would scale every logit to nothing.
    ///
    /// Gemma-2, Gemma-3 and Gemma-3n state it; nothing else does. That is why
    /// `attn::logit_softcap_bf16` could not cross without this — a `none:`
    /// arm surfaces as [`crate::x::Route::Unbound`] at model **load**, so it
    /// would refuse every Gemma deployment for a symbol that fires correctly
    /// today.
    fn final_logit_softcap(&self) -> Option<f32> {
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
    /// This layer's latent cache — [`Facts::kv_layer`]'s MLA sibling.
    ///
    /// **Nothing implements this yet, and the reason is structural rather
    /// than an oversight.** `AttnCtx` carries `layers: Vec<KvCacheLayerView>`
    /// and no MLA equivalent; the views come from
    /// `pools::mla_cache::MlaCachePool::layer_view`, which no `Fire` can
    /// reach. Implementing it means `AttnCtx` carrying the MLA layer list the
    /// way it carries the paged one — a `driver-cuda` change, not a `Cx`
    /// query — and until that lands this refuses with its sentence rather
    /// than answering wrongly.
    fn mla_layer(&self) -> Option<MlaLayer> {
        None
    }
    /// The plan `Prepare::MlaPlan` built for this fire.
    ///
    /// [`Facts::mla_layer`]'s situation: `crate::plan::mla` builds it, and
    /// nothing hands the result to a `Fire`.
    fn mla_plan(&self) -> Option<MlaPlan> {
        None
    }
    /// The attention workspace this fire was given.
    fn attn_workspace(&self) -> Option<AttnWorkspace> {
        None
    }
    /// The softmax scale this fire was planned with.
    fn sm_scale(&self) -> Option<f32> {
        None
    }
    /// `write_kv_to_pages`' first-token scalar.
    fn first_token(&self) -> Option<i32> {
        None
    }
    /// The pages this fire's CSR names.
    fn num_pages_in_batch(&self) -> Option<i32> {
        None
    }
    /// Per-row target page for this fire's KV append.
    fn w_page_d(&self) -> Option<*const u32> {
        None
    }
    /// Per-row offset-in-page for the append.
    fn w_off_d(&self) -> Option<*const u32> {
        None
    }
    /// The destination the fused QKV kernel writes Q into.
    ///
    /// `None` when the fire's statement pinned no query buffer. The device
    /// does not null-test this pointer, so absence must be refused here
    /// rather than passed on.
    fn q_out(&self) -> Option<*mut c_void> {
        None
    }
    /// The sliding-window span this fire attends, in rows.
    fn window_left(&self) -> Option<i32> {
        None
    }
    /// The attention logit soft cap, `0` for none.
    fn logits_soft_cap(&self) -> Option<f32> {
        None
    }
    /// The LSE scratch the decode dispatch writes.
    fn lse_out(&self) -> Option<*mut f32> {
        None
    }
    /// The fire's per-request plan arrays.
    fn plan(&self) -> Option<Plan> {
        None
    }
    /// One of a gated-delta-net layer's state slabs.
    ///
    /// The BASE only. See [`Slab`] for the other half of the address, which
    /// is [`Facts::gdn`]'s.
    fn slab(&self, which: Slab) -> Option<*mut c_void> {
        let _ = which;
        None
    }
    /// This fire's linear-attention shape and state addressing.
    fn gdn(&self) -> Option<Gdn> {
        None
    }
    /// The `i`th AUXILIARY buffer this statement publishes or reads.
    ///
    /// `Source::Aux(i)`. Scratch the lowering placed and named by index —
    /// a chunked scan's per-chunk partials, a split-K's accumulator.
    ///
    /// **Pre-resolved.** `bind/facts.rs`'s note is the constraint: *"a
    /// `Resolver` is `&mut` and `Facts` is not."* A driver impl holds the
    /// answer already, computed at the dispatch site; it does not resolve
    /// one here. That is what keeps [`Cx`] query-only and it is why this
    /// returns a pointer rather than an index into something.
    fn aux(&self, i: usize) -> Option<*mut c_void> {
        let _ = i;
        None
    }
    /// The `i`th RESULT placement.
    ///
    /// `Source::ResultOrRegion(i)`, and **this is not [`Facts::arg_out`]**.
    /// The two read different lists: a result reads `spec.outs[i]`, an
    /// output operand reads `bound.args[n_in + i]`. They agree for a
    /// statement whose outputs are all operands and disagree for one that
    /// writes into a region the lowering placed, which is exactly the case
    /// `ResultOrRegion` was minted for. Taking the wrong one writes to a
    /// valid address that belongs to something else.
    ///
    /// Pre-resolved, for [`Facts::aux`]'s reason.
    fn result(&self, i: usize) -> Option<*mut c_void> {
        let _ = i;
        None
    }
    /// Whether the router renormalises its top-k weights.
    ///
    /// A deployment constant, not a shape: `DispatchCtx` fills it from
    /// `model.deployment.norm_topk_prob`. Four `moe` routers carried it as
    /// `Source::Ctx("moe_norm_topk")` in the row world.
    fn moe_norm_topk(&self) -> Option<bool> {
        None
    }
    /// The router's routed scaling factor.
    ///
    /// The other half of [`Facts::moe_norm_topk`]'s pair, and carried the
    /// same way.
    fn moe_routed_scaling(&self) -> Option<f32> {
        None
    }
    /// The `i`th input's row count.
    ///
    /// **An operand's own extent**, which is why it is here rather than on
    /// `Dims`: `Source::InRows(1)` is the padded rectangle's height that the
    /// aligned MoE path's gather and reorder read as `aligned_rows`, and no
    /// width, param or context field carries it. The generated dispatcher
    /// computed it per arm as `rows_of(b, i, rows)`.
    fn in_rows(&self, i: usize) -> Option<i32> {
        let _ = i;
        None
    }
    /// gpt-oss's clamped-GLU ceiling.
    fn glu_limit(&self) -> Option<f32> {
        None
    }
    /// The clamped GLU's alpha.
    fn glu_alpha(&self) -> Option<f32> {
        None
    }
    /// A weight the statement names by SUFFIX rather than by position.
    ///
    /// `Source::WeightSuffix(s)`: the driver resolves `{bank}{s}` out of its
    /// own table and the trace never states it. **Pre-resolved**, for
    /// [`Facts::aux`]'s reason — a `Resolver` is `&mut` and this trait is
    /// not, and a bind body that could consult a store could also make the
    /// store answer differently the second time.
    ///
    /// `Option` rather than a refusal because the caller knows which of its
    /// suffixes may be null and this does not; see [`Cx::weight_suffixed`].
    fn weight_suffixed(&self, suffix: &str) -> Option<*mut c_void> {
        let _ = suffix;
        None
    }
    /// The statement's bias weight, when it carries one.
    ///
    /// **Absence is legal here and is not a refusal** —
    /// `csrc/src/ssm/causal_conv1d.cuh:383` marks the parameter
    /// `// [C] nullable`, so the kernel takes a null and the host program
    /// says so. [`Cx::weight_bias`] therefore returns `Option` rather than
    /// `Result`: the only query on [`Cx`] that does, because it is the only
    /// fact whose absence the device text explicitly accepts.
    ///
    /// Pre-resolved, for [`Facts::aux`]'s reason.
    fn weight_bias(&self) -> Option<*mut c_void> {
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
        ///
        /// A HANDLE, not a computation: `crate::plan::mla` already builds it,
        /// and this query hands back what it built plus the device `int` and
        /// `float` arenas the plan indexes.
        mla_plan -> MlaPlan, "the mla plan"
    );
    query!(
        /// The attention workspace this fire was given.
        ///
        /// Asked for by two families independently — `x::attn`'s MLA arm and
        /// `x::xqa`'s decode arm — because both rows carried
        /// `Source::Attn(..)`, which is the row world's name for it.
        attn_workspace -> AttnWorkspace, "the attention workspace"
    );
    query!(
        /// The softmax scale this fire was planned with.
        ///
        /// [`Cx::attn_workspace`]'s pair, and asked for by the same two
        /// families for the same reason: `Source::Attn("sm_scale")`.
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
        ///
        /// With [`Cx::w_off_d`], the pair `write_kv_explicit_bf16` binds
        /// against. **`AttnCtx` has carried both since before fn-world** —
        /// unlike [`Cx::mla_layer`], which refuses because nothing fills it,
        /// these had a producer all along and only lacked a way to ask.
        w_page_d -> *const u32, "the append's per-row target page"
    );
    query!(
        /// Per-row offset-in-page for the append. [`Cx::w_page_d`]'s pair.
        w_off_d -> *const u32, "the append's per-row page offset"
    );
    query!(
        /// The destination the fused QKV kernel writes Q into.
        ///
        /// **The one query in this family whose `Option` contradicts its
        /// row.** `table/attn.rs` sources it plain `Source::Attn("q_out")`,
        /// which asserts presence; the producer at `fire/launch.rs:3248`
        /// ends `.unwrap_or(core::ptr::null_mut())`, which does not.
        ///
        /// The device breaks the tie: `qkv_fused.cuh:177` is
        /// `dst = q_out + …` with **no null test**, while `:182` null-tests
        /// `w_page`/`w_off` two arguments along. So a fire pinning no query
        /// buffer stored through null unconditionally, and that is not a
        /// state the row was describing — it is a state the row was wrong
        /// about.
        ///
        /// The rule, corrected: the row grammar decides **while the row and
        /// the producer agree**. When they disagree the producer is the fact
        /// and the row is a claim, because a row cannot make a pointer
        /// non-null. See `Facts::q_out`.
        q_out -> *mut c_void, "the fused QKV kernel's query destination"
    );
    query!(
        /// The sliding-window span this fire attends, in rows.
        ///
        /// **A three-tier decision, answered.** `bind::window_of` resolves it
        /// as *statement parameter → per-layer vector → fire default*, and
        /// the driver already makes that decision once per launch. Handing
        /// back the three inputs would be a predicate every caller
        /// re-derives, which is [`Cx::kv_layer`]'s `has_envelopes` argument
        /// in a second place.
        ///
        /// Wanted by `attn::split_qkv_bf16_devwin`, which is the other edge
        /// of the driver-op discriminator: it needs **no driver resource**
        /// and still cannot bind from `arg_in`/`arg_out` alone, because those
        /// are pre-windowed by `resolve_arg_windowed` and this kernel windows
        /// again from device memory.
        window_left -> i32, "the fire's sliding-window span"
    );
    query!(
        /// The attention logit soft cap, `0` for none.
        ///
        /// **Not [`Cx::final_logit_softcap`] — same word, different fact.**
        /// That one is the SAMPLER's cap on the final logits and its row
        /// grammar is `Source::CtxNonZero`; this is the ATTENTION score cap
        /// that `attention_naive_paged` takes as an operand, and its row
        /// grammar is plain `Source::Ctx`. Two facts that share a noun and
        /// disagree about zero is exactly the shape §3.2 is about.
        logits_soft_cap -> f32, "the attention logit soft cap"
    );
    query!(
        /// The LSE scratch the decode dispatch writes.
        ///
        /// **Not null-checked, and that is deliberate.** [`Cx::w_page_d`] is
        /// checked because its row grammar is `Source::AttnNonZero` — a row
        /// that TESTS absence, so a null is a state the row admits. This one
        /// is plain `Source::Attn`, a row that ASSERTS presence, and an
        /// `Option` here would invent a state the row denies.
        ///
        /// The two look identical at the `Facts` impl and are not; the row
        /// grammar is the discriminator and it is one word apart.
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
        /// [`Facts::result`].
        result(i: usize) -> *mut c_void, "a result placement"
    );
    query!(
        /// Whether the router renormalises its top-k weights.
        ///
        /// Four `moe` routers carried this as `Source::Ctx("moe_norm_topk")`
        /// and the generated dispatcher rendered it `ctx.moe_norm_topk`.
        /// `DispatchCtx` has held it since `bind/mod.rs:1179`, filled from
        /// `model.deployment.norm_topk_prob`; what was missing was a way for
        /// a fn-world body to ask.
        moe_norm_topk -> bool, "whether the router renormalises its top-k"
    );
    query!(
        /// The router's routed scaling factor.
        ///
        /// The other half of the pair above. Together they turn
        /// `topk_sigmoid`, `topk_sqrtsoftplus`, `topk_sigmoid_bias` and
        /// `hash_route_lookup` from `none:` arms into binds — and the first
        /// three fire today, so the `none:` was a stated regression rather
        /// than a dormant one.
        moe_routed_scaling -> f32, "the router's routed scaling factor"
    );
    query!(
        /// The `i`th input's row count.
        ///
        /// `Source::InRows(1)` in the row world: the padded rectangle's
        /// height, which the aligned path's gather and reorder both read as
        /// `aligned_rows`. **An operand's own extent that no width, param or
        /// context field carries** — which is why it needed a query rather
        /// than a `Dims` axis. The driver already computes it for every
        /// generated arm as `rows_of(b, i, rows)`.
        in_rows(i: usize) -> i32, "an input's row count"
    );
    query!(
        /// gpt-oss's clamped-GLU ceiling, `Source::Ctx("glu_limit")`.
        ///
        /// A deployment constant on `DispatchCtx` since `bind/mod.rs:1193`.
        glu_limit -> f32, "the clamped-GLU limit"
    );
    query!(
        /// The clamped GLU's alpha, `Source::Ctx("glu_alpha")`.
        glu_alpha -> f32, "the clamped-GLU alpha"
    );

    /// A weight the statement names by SUFFIX rather than by position.
    ///
    /// `Source::WeightSuffix(s)` in the row world: the driver resolves
    /// `{bank}{s}` out of its own table, and **the trace never states it** —
    /// `weight_names.rs:505` records that `quant::mxfp4_moe_gate_up_decode_bf16`
    /// reaches `_scales`, `_gate_bias` and `_up_bias` this way.
    ///
    /// # Why this returns `Option` and does not refuse
    ///
    /// [`Cx::weight_bias`]'s reason, and it is the caller's to make: two of
    /// the three suffixes are NULLABLE and one is not, in the same bind body.
    /// A query that refused would force the nullable pair through a
    /// `.ok().unwrap_or(null)` that reads as carelessness; returning `Option`
    /// lets `_scales` write `.ok_or(Refusal::Absent { what: "scale_ptrs" })?`
    /// and name the operand it is missing. **The refusal is more accurate
    /// when the call site makes it**, because only the call site knows which
    /// of its three suffixes the kernel may take a null for.
    ///
    /// Pre-resolved, for [`Facts::aux`]'s reason: a `Resolver` is `&mut` and
    /// `Facts` is not, so the resolve happens once at the dispatch site,
    /// before the `Cx` exists.
    #[must_use]
    pub fn weight_suffixed(&self, suffix: &str) -> Option<*mut c_void> {
        self.facts.weight_suffixed(suffix)
    }

    /// The statement's bias weight, or `None` when it carries none.
    ///
    /// **The one query that does not refuse**, and the exception is the
    /// device text's: `causal_conv1d.cuh:383` marks the parameter
    /// `// [C] nullable`. A `Result` here would make a host program write
    /// `.unwrap_or(null_mut())` at every call site, which is a refusal
    /// spelled as an escape — and a refusal that every caller escapes is
    /// worse than no refusal, because the next reader cannot tell which
    /// escapes are load-bearing.
    ///
    /// Absence is `None` and the kernel takes a null. Nothing else on
    /// [`Cx`] may copy this shape without a `// [C] nullable` beside the
    /// parameter it feeds.
    #[must_use]
    pub fn weight_bias(&self) -> Option<*mut c_void> {
        self.facts.weight_bias()
    }

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
