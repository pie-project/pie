use core::ffi::c_void;

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
