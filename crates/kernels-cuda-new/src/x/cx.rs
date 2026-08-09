use core::ffi::c_void;
use core::ptr::NonNull;

use crate::x::contract::Refusal;

/// WHICH ROWS this fire is launching.
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
    /// How the pages are quantised, and whether the scale tensors apply.
    pub scheme: KvScheme,
    /// What a page element actually is — the model's dtype only under
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
    pub has_envelopes: bool,
    /// Storage is the model's own bf16, so no dequantisation step applies.
    pub is_native_bf16: bool,
}

/// One layer's LATENT cache, as MLA's launchers take it.
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slab {
    /// The short convolution's ring buffer.
    Conv,
    /// The recurrent state.
    Recurrent,
}

/// A linear-attention fire's shape and its state addressing.
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
    pub n_groups: i32,
    /// Elements per conv slot, `conv_k · conv_dim`. Pairs with
    pub conv_stride_elems: i64,
    /// Elements per recurrent slot. Pairs with [`Slab::Recurrent`].
    pub state_stride_elems: i64,
    /// Device request→slot ids, one per request in the fire.
    pub slot_ids_d: *const i32,
    /// Whether this fire advances state.
    pub write_state: bool,
}

/// The YaRN quartet, as a checkpoint states it.
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
    pub const NONE: Self = Self {
        factor: 1.0,
        beta_fast: 0.0,
        beta_slow: 0.0,
        attention_factor: 1.0,
        original_max_position: 0,
    };
}

/// The query-only fire vocabulary, implemented by the driver.
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
    fn token_ids(&self) -> Option<*const i32> {
        None
    }
    /// How many tokens the vocabulary holds.
    fn vocab(&self) -> Option<i32> {
        None
    }
    /// The rows a sampling gather collects, or `None` when the fire gathers
    fn sampling_indices(&self) -> Option<*const i32> {
        None
    }
    /// The per-layer-embedding width, for a checkpoint that carries PLE
    fn ple_dim(&self) -> Option<i32> {
        None
    }
    /// The fire's PEEL WINDOW: a device-resident `[start, count]`, or `None`
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
    fn rope_theta(&self) -> Option<f32> {
        None
    }
    /// The rotary base for THIS layer.
    fn theta(&self) -> Option<f32> {
        None
    }
    /// The RMS norm epsilon.
    fn rms_eps(&self) -> Option<f32> {
        None
    }
    /// The logit soft cap, or `None` where the deployment states none.
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
    fn mla_layer(&self) -> Option<MlaLayer> {
        None
    }
    /// The plan `Prepare::MlaPlan` built for this fire.
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
    /// The WNA16 weight's group size, in elements along K.
    fn wna16_group_size(&self) -> Option<i32> {
        None
    }
    /// The destination the fused QKV kernel writes Q into.
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
    fn slab(&self, which: Slab) -> Option<*mut c_void> {
        let _ = which;
        None
    }
    /// This fire's linear-attention shape and state addressing.
    fn gdn(&self) -> Option<Gdn> {
        None
    }
    /// The `i`th AUXILIARY buffer this statement publishes or reads.
    fn aux(&self, i: usize) -> Option<*mut c_void> {
        let _ = i;
        None
    }
    /// The `i`th RESULT placement.
    fn result(&self, i: usize) -> Option<*mut c_void> {
        let _ = i;
        None
    }
    /// Whether the router renormalises its top-k weights.
    fn moe_norm_topk(&self) -> Option<bool> {
        None
    }
    /// The router's routed scaling factor.
    fn moe_routed_scaling(&self) -> Option<f32> {
        None
    }
    /// The `i`th input's row count.
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
    fn weight_suffixed(&self, suffix: &str) -> Option<*mut c_void> {
        let _ = suffix;
        None
    }
    /// The statement's bias weight, when it carries one.
    fn weight_bias(&self) -> Option<*mut c_void> {
        None
    }
}

/// The query-only fire context a bind body reads.
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
        /// # Errors
        pub fn $name(&self) -> Result<$ty, Refusal> {
            self.facts.$name().ok_or(Refusal::Unstated { what: $what })
        }
    };
    ($(#[$m:meta])* $name:ident ( $arg:ident : $at:ty ) -> $ty:ty, $what:literal) => {
        $(#[$m])*
        /// # Errors
        pub fn $name(&self, $arg: $at) -> Result<$ty, Refusal> {
            self.facts.$name($arg).ok_or(Refusal::Absent { what: $what })
        }
    };
}

impl<'a> Cx<'a> {
    /// Wrap a driver's facts.
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
        self.facts.weight_suffixed(suffix)
    }

    /// The statement's bias weight, or `None` when it carries none.
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
    #[must_use]
    pub fn rope_interleaved(&self) -> bool {
        self.facts.rope_interleaved()
    }

    /// The `i`th statement parameter as a float.
    pub fn param_f32(&self, i: usize) -> Result<f32, Refusal> {
        self.param(i).map(f32::from_bits)
    }
}
