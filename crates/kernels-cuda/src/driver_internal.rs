//! Launchers for a driver-side caller, not a family: no `FAMILY`, no
//! `ROUTINES` here.
//!
//! Four host programs below are named by statements; each is declared with a
//! `routine!` line in its own family module (`attn`, `layout`, `mlp`, `ssm`)
//! naming this file's `fn` by bare identifier, since `Family::symbol` needs a
//! module prefix this file doesn't have. [`add_bias_bf16`] and
//! [`rmsnorm_gated_fp32_in_bf16`] are `void*` entry points forwarding into an
//! already-declared `norm` routine.
//!
//! Every `# Safety` below also requires that the memory it names live on
//! `ctx`'s stream, which must outlive the launch.

use core::ffi::c_void;
use core::ptr::NonNull;

use crate::jit::{Ctx, Launch};
use crate::jit::Abi;
use crate::jit::abi::bf16;
use crate::{norm, quant};
use kernels::Refusal;
use kernels::keys::{self, Fact};
use kernels::routine::{Bank, Env, In, Out, Weight};

/// The pointwise launch block, matching `runtime/launch.rs`'s `BLOCK`.
const BLOCK: u32 = 256;

/// The QKV split — `attn::split_qkv_bf16`, `attn/split_packed.cuh`.
///
/// # Safety
///
/// `packed` is `[n_tokens, q_dim + 2 * kv_dim]` bf16; `q_out` is
/// `[n_tokens, q_dim]` and `k_out`/`v_out` are `[n_tokens, kv_dim]`, all
/// bf16 and all writable.
#[kernels_macros::routine]
pub fn split_qkv_bf16(
    ctx: &Ctx,
    packed: In<0, bf16>,
    q_out: Out<0, bf16>,
    k_out: Out<1, bf16>,
    v_out: Out<2, bf16>,
) -> Result<(), Refusal> {
    let q = q_out.all("the q half")?;
    let k = k_out.all("the k half")?;
    let (q_dim, kv_dim) = (q.width, k.width);
    if q_dim <= 0 && kv_dim <= 0 {
        return Err(Refusal::Empty { what: "q_dim and kv_dim" });
    }
    let n_tokens = q.rows;
    let width = q_dim.max(kv_dim).unsigned_abs();
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "attn/split_packed.cuh",
            "::pie::attn::split_qkv<::pie::bf16>",
            Launch::grid([width.div_ceil(BLOCK), n_tokens.unsigned_abs(), 1], [BLOCK, 1, 1]),
            &[
                packed.ptr.arg(),
                q_out.ptr.arg(),
                k_out.ptr.arg(),
                v_out.ptr.arg(),
                q_dim.arg(),
                kv_dim.arg(),
            ],
        )
    }
}

/// The bias add — `norm::add_bias_bf16`, `norm/add_bias.cuh`.
///
/// The geometry is [`norm::add_bias_bf16`]'s own; this is the cast and
/// nothing else, from `void*` to the `bf16*`/`Weight` the routine takes.
/// Wrapping `bias` as `Weight` here derives no column, since this module
/// declares no `Family`; it is purely the callee's parameter type.
///
/// # Safety
///
/// `out` is `[num_rows, dim]` bf16 and writable, `bias` is `[dim]` bf16.
pub fn add_bias_bf16(
    ctx: &Ctx,
    out: *mut c_void,
    bias: *const c_void,
    num_rows: i32,
    dim: i32,
) -> Result<(), Refusal> {
    // `num_rows`/`dim` stay separate params here (a `*mut c_void` entry point
    // is what non-`Ctx` driver callers have) and become one `Out` region
    // only once they cross into the routine.
    norm::add_bias::<bf16>(
        ctx,
        Out { ptr: out.cast::<bf16>(), rows: num_rows, width: dim },
        Weight { ptr: bias.cast::<bf16>() },
    )
}

/// Qwen3.5's post-convolution split — `ssm::qwen_gdn_post_conv_prep_bf16`,
/// `gated_delta_net.cu`.
///
/// # Safety
///
/// `qkv_post` is `[N, conv_dim]` bf16; `a`, `b` and `dt_bias` are bf16 over
/// `[N, V_h]`, `[N, V_h]` and `[V_h]`; `a_log` is `[V_h]` fp32; the five
/// outputs are writable for `[N, K_h, K_d]`, `[N, K_h, K_d]`,
/// `[N, V_h, V_d]`, `[N, V_h]` and `[N, V_h]`.
#[allow(clippy::too_many_arguments)]
#[kernels_macros::routine]
pub fn qwen_gdn_post_conv_prep_bf16(
    ctx: &Ctx,
    qkv_post: In<0, bf16>,
    a: In<1, bf16>,
    b: In<2, bf16>,
    // `Bank<N>` reads the positional weight run; `Weight<N>` would compile
    // and bind the same two tensors through the named-weight table instead.
    a_log: Bank<0, f32>,
    dt_bias: Bank<1, bf16>,
    q_norm_kh: Out<0, f32>,
    k_norm_kh: Out<1, f32>,
    v_fp32: Out<2, f32>,
    g_log_out: Out<3, f32>,
    beta_out: Out<4, f32>,
    k_h: Env<keys::GdnKHeads>,
    v_h: Env<keys::GdnVHeads>,
    k_d: Env<keys::GdnKDim>,
    v_d: Env<keys::GdnVDim>,
    conv_dim: Env<keys::GdnConvDim>,
) -> Result<(), Refusal> {
    /// `gated_delta_net.cu`'s `constexpr int BLOCK = 128;`.
    const PREP_BLOCK: u32 = 128;
    let n = qkv_post.all("the post-convolution qkv")?.rows;
    #[allow(clippy::cast_precision_loss)]
    let q_scale = (**k_d as f32).sqrt().recip();
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "ssm/gated_delta_net_prep.cuh",
            "::pie::ssm::qwen_gdn_qk_norm<::pie::bf16, 128>",
            Launch::grid([n.unsigned_abs(), (**k_h).unsigned_abs(), 1], [PREP_BLOCK, 1, 1]),
            &[
                qkv_post.ptr.arg(),
                q_norm_kh.ptr.arg(),
                k_norm_kh.ptr.arg(),
                k_h.arg(),
                k_d.arg(),
                conv_dim.arg(),
                q_scale.arg(),
            ],
        )?;
    }
    // SAFETY: the caller's assertion, forwarded. No barrier needed: the
    // second launch re-reads `qkv_post`, not anything the first wrote.
    unsafe {
        ctx.launch(
            "ssm/gated_delta_net_prep.cuh",
            "::pie::ssm::qwen_gdn_v_g_beta<::pie::bf16, 128>",
            Launch::grid([n.unsigned_abs(), (**v_h).unsigned_abs(), 1], [PREP_BLOCK, 1, 1]),
            &[
                qkv_post.ptr.arg(),
                a.ptr.arg(),
                b.ptr.arg(),
                a_log.ptr.arg(),
                dt_bias.ptr.arg(),
                v_fp32.ptr.arg(),
                g_log_out.ptr.arg(),
                beta_out.ptr.arg(),
                k_h.arg(),
                v_h.arg(),
                k_d.arg(),
                v_d.arg(),
                conv_dim.arg(),
            ],
        )
    }
}

/// The per-head query/gate split — `layout::split_q_gate_bf16`,
/// `layout/deinterleave.cuh`.
///
/// # Safety
///
/// `packed` is `[n, num_heads, 2 * head_dim]` bf16; `q_out` and `gate_out`
/// are `[n, num_heads, head_dim]` bf16 and writable. All live on `ctx`'s
/// stream.
#[kernels_macros::routine]
pub fn split_q_gate_bf16(
    ctx: &Ctx,
    packed: In<0, bf16>,
    q_out: Out<0, bf16>,
    gate_out: Out<1, bf16>,
    // No operand carries this: both results are `heads * head_dim` wide and
    // `packed` is twice that, so the extents fix the product, never the
    // factors; `heads` is the division below.
    head_dim: Env<keys::PerHeadDim>,
) -> Result<(), Refusal> {
    let q = q_out.all("the query half")?;
    let head_dim = **head_dim;
    if head_dim <= 0 {
        return Err(Refusal::Unstated { what: "the head pitch a q/gate split grids by" });
    }
    if q.width % head_dim != 0 {
        return Err(Refusal::Unstated { what: "a q/gate half whose width is not whole heads" });
    }
    let (n, num_heads) = (q.rows, q.width / head_dim);
    let block = if head_dim < 128 { 64 } else { 128 };
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "layout/deinterleave.cuh",
            "::pie::layout::split_q_gate<::pie::bf16>",
            Launch::grid([n.unsigned_abs(), num_heads.unsigned_abs(), 1], [block, 1, 1]),
            &[
                packed.ptr.arg(),
                q_out.ptr.arg(),
                gate_out.ptr.arg(),
                n.arg(),
                num_heads.arg(),
                head_dim.arg(),
            ],
        )
    }
}

/// That gate applied — `mlp::sigmoid_gate_inplace_bf16`, `mlp/swiglu.cuh`.
///
/// # Safety
///
/// `x` and `gate` are both `num_elements` bf16 elements; `x` is writable and
/// is read and written by the same threads.
#[kernels_macros::routine]
pub fn sigmoid_gate_inplace_bf16(
    ctx: &Ctx,
    x: Out<0, bf16>,
    gate: In<1, bf16>,
) -> Result<(), Refusal> {
    let num_elements = x.all("the gated rectangle")?.elements();
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "mlp/swiglu.cuh",
            "::pie::mlp::sigmoid_gate_inplace<::pie::bf16>",
            Launch::flat(num_elements.unsigned_abs(), BLOCK),
            &[x.ptr.arg(), gate.ptr.arg(), num_elements.arg()],
        )
    }
}

/// The gated norm with an FP32 `x` — `norm::rmsnorm_gated_fp32_in_bf16`,
/// `norm/rmsnorm.cuh`.
///
/// Same geometry as the routine at `per_head_dim = 0`; the refusals are not
/// — this reads `hidden` and `num_rows` off a layer instead of trusting the
/// routine's own row count.
///
/// # Safety
///
/// `x` is `[num_rows, hidden]` fp32; `gate` is `[num_rows, hidden]` bf16;
/// `weight` is `[hidden]` fp32; `y` is `[num_rows, hidden]` bf16 and
/// writable.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_gated_fp32_in_bf16(
    ctx: &Ctx,
    x: *const c_void,
    gate: *const c_void,
    weight: *const c_void,
    y: *mut c_void,
    num_rows: i32,
    hidden: i32,
    eps: f32,
) -> Result<(), Refusal> {
    // `weight` is wrapped as the named bank: `OpKind::RmsnormGated` only ever
    // names it through `LaunchSpec::weight`, so there is no positional slot
    // for it to be `Bank<0>` of.
    let shape = |p: *mut bf16| Out { ptr: p, rows: num_rows, width: hidden };
    norm::rmsnorm_gated_fp32_in::<bf16>(
        ctx,
        In { ptr: x.cast::<f32>(), rows: num_rows, width: hidden },
        In { ptr: gate.cast::<bf16>(), rows: num_rows, width: hidden },
        Weight { ptr: weight.cast::<f32>() },
        shape(y.cast::<bf16>()),
        0,
        keys::RmsEps::env(eps),
    )
}

// ===========================================================================
// Seven launchers for seven `__global__`s whose C++ callers were deleted
// along with the archive crate that held them. The `.cuh` device text stayed
// (`src/source.rs` carries every file under `kernels/`), so each fn below
// keeps its kernel reachable and compiling though nothing calls it: no
// statement names any of the seven and no `Family` claims them, for
// `geometry.cuh`'s reason — none has operands a `Source` could name. Each
// geometry recovers the deleted launcher's own; a refusal that is not the
// old launcher's says so.
// ===========================================================================

/// A device address as a by-value aggregate carries one — matching
/// `attn::fa2`'s `addr`: a `u64`, not a pointer, since the host never
/// dereferences it and the device pointer is 64-bit regardless of the
/// host's.
fn addr<T>(p: *const T) -> crate::jit::abi::DevicePtr {
    p as usize as u64
}

/// `kv_len[r]` out of the CSR page descriptors — `layout/geometry.cuh`.
///
/// One thread per request; the arithmetic mirrors the kernel's own host
/// formula, since a device-composed wave has no host copy of `kv_len`.
///
/// # Safety
///
/// `kv_page_indptr` addresses `num_requests + 1` live `u32`s and
/// `kv_last_page_lens` `num_requests` of them; `kv_len` is `num_requests`
/// writable `u32`s.
pub fn derive_kv_len(
    ctx: &Ctx,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    page_size: u32,
    num_requests: u32,
    kv_len: *mut u32,
) -> Result<(), Refusal> {
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "layout/geometry.cuh",
            "::pie::layout::derive_kv_len",
            Launch::flat(num_requests, BLOCK),
            &[
                kv_page_indptr.arg(),
                kv_last_page_lens.arg(),
                page_size.arg(),
                num_requests.arg(),
                kv_len.arg(),
            ],
        )
    }
}

/// A working-set slot id to its physical page-pool block —
/// `layout/geometry.cuh`.
///
/// An out-of-range slot resolves to `0xFFFFFFFF` rather than wrapping — the
/// kernel's own choice — so nothing here bounds-checks one; a corrupt slot
/// must fail visibly at the gather instead.
///
/// # Safety
///
/// `pages` addresses `count` live `u32` slot ids, `slot_to_block` `num_slots`
/// live `u32`s, and `page_indices` `count` writable ones. All live on `ctx`'s
/// stream, which must outlive the launch.
pub fn resolve_slot_to_block(
    ctx: &Ctx,
    pages: *const u32,
    slot_to_block: *const u32,
    num_slots: u32,
    count: u32,
    page_indices: *mut u32,
) -> Result<(), Refusal> {
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "layout/geometry.cuh",
            "::pie::layout::resolve_slot_to_block",
            Launch::flat(count, BLOCK),
            &[
                pages.arg(),
                slot_to_block.arg(),
                num_slots.arg(),
                count.arg(),
                page_indices.arg(),
            ],
        )
    }
}

/// A device-composed decode wave's page CSR — `layout/geometry.cuh`'s
/// `compose_envelope_csr`.
///
/// One block of `member_count` threads, once per wave: the output CSR is a
/// single shared object, and a per-row grid would have every block writing
/// its own copy of it.
///
/// `EnvelopeMember` is three `u32`s the caller writes directly — no Rust
/// mirror of the struct, for [`gather_tokens`]'s reason: two spellings of one
/// layout agree only until a field moves.
///
/// # Safety
///
/// `members` addresses `member_count` live `EnvelopeMember` records;
/// `traced_page_indptr` `2 * member_count` live `u32`s; `traced_kv_len`,
/// `traced_w_slot` and `token_ids` `member_count` each; `traced_pages` the
/// sum of the members' `page_bound`s. `kv_page_indptr` is `member_count + 1`
/// writable `u32`s, `kv_page_indices` that same sum, `kv_last_page_lens` and
/// `w_slot_out` `member_count` each, and `row_valid` `member_count` writable
/// bytes. `kills` is null or one writable `u32`.
#[allow(clippy::too_many_arguments)]
pub fn compose_envelope_csr(
    ctx: &Ctx,
    members: *const c_void,
    traced_page_indptr: *const u32,
    traced_pages: *const u32,
    traced_kv_len: *const u32,
    traced_w_slot: *const u32,
    token_ids: *const u32,
    member_count: u32,
    page_size: u32,
    kv_page_indptr: *mut u32,
    kv_page_indices: *mut u32,
    kv_last_page_lens: *mut u32,
    w_slot_out: *mut u32,
    row_valid: *mut u8,
    kills: Option<NonNull<u32>>,
) -> Result<(), Refusal> {
    // Like `graph_pad_rows`' ceiling: `<<<1, member_count>>>` is a block
    // width, and past the device's maximum this fails inside
    // `cudaGetLastError`, launch already issued.
    if member_count > MAX_BLOCK.unsigned_abs() {
        return Err(Refusal::Wide {
            what: "member_count, as one block's threads",
            at: i64::from(member_count),
            max: i64::from(MAX_BLOCK),
        });
    }
    // One `u32` per member — page count, then (post-scan) its offset into
    // the composed list. `extern __shared__` sizes nothing itself.
    let smem = member_count * 4;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "layout/geometry.cuh",
            "::pie::layout::compose_envelope_csr",
            Launch::grid([1, 1, 1], [member_count, 1, 1]).smem(smem),
            &[
                members.arg(),
                traced_page_indptr.arg(),
                traced_pages.arg(),
                traced_kv_len.arg(),
                traced_w_slot.arg(),
                token_ids.arg(),
                member_count.arg(),
                page_size.arg(),
                kv_page_indptr.arg(),
                kv_page_indices.arg(),
                kv_last_page_lens.arg(),
                w_slot_out.arg(),
                row_valid.arg(),
                kills.arg(),
            ],
        )
    }
}

/// The KV compaction copy — `layout/gather_tokens.cuh`, choosing between its
/// two kernels.
///
/// One block per `(op, layer)`; the vectorised arm needs `token_stride` and
/// `layer_stride_elems` both eight-element aligned, a runtime test no
/// `LaunchRule` can state. Pointers are `u16`, not `bf16`: both kernels are
/// pure copies that never convert to float, and the vectorised arm reads the
/// same bytes as `int4` by kernel selection alone, with no Rust `int4` type
/// to cast the pointer to.
///
/// # Safety
///
/// `k_pages` and `v_pages` address the layer's whole page pool as `u16`, and
/// every op in `ops` names spans inside it. `ops` addresses `num_ops` live
/// `GatherTokenOp` records — five `u32`s each, the layout the caller must
/// have written.
#[allow(clippy::too_many_arguments)]
pub fn gather_tokens(
    ctx: &Ctx,
    k_pages: *mut u16,
    v_pages: *mut u16,
    ops: *const c_void,
    num_ops: i32,
    num_layers: i32,
    layer_stride_elems: i64,
    page_size: i32,
    num_kv_heads: i32,
    head_dim: i32,
) -> Result<(), Refusal> {
    let token_stride = i64::from(num_kv_heads) * i64::from(head_dim);
    let page_stride = token_stride * i64::from(page_size);
    let grid = Launch::grid([num_ops.unsigned_abs(), 1, num_layers.unsigned_abs()], [BLOCK, 1, 1]);

    // Eight bf16 is one `int4`: the vectorised arm needs the token and layer
    // strides eight-aligned. `page_stride` isn't tested because it's
    // `token_stride * page_size`, so token alignment already implies it.
    if token_stride % 8 == 0 && layer_stride_elems % 8 == 0 {
        // SAFETY: the caller's assertion, forwarded.
        return unsafe {
            ctx.launch(
                "layout/gather_tokens.cuh",
                "::pie::layout::gather_i4",
                grid,
                &[
                    k_pages.cast::<c_void>().arg(),
                    v_pages.cast::<c_void>().arg(),
                    ops.arg(),
                    (token_stride / 8).arg(),
                    (page_stride / 8).arg(),
                    (layer_stride_elems / 8).arg(),
                ],
            )
        };
    }
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "layout/gather_tokens.cuh",
            "::pie::layout::gather_u16",
            grid,
            &[
                k_pages.arg(),
                v_pages.arg(),
                ops.arg(),
                token_stride.arg(),
                page_stride.arg(),
                layer_stride_elems.arg(),
            ],
        )
    }
}

/// The graph-lattice pad lanes' CSR — `layout/graph_pad.cuh`.
///
/// One block of `padding` threads, once per captured wave: each thread
/// writes a different pad lane of one shared CSR, so one block per row would
/// race many copies on the same words.
///
/// # Safety
///
/// Every pointer addresses the wave's own CSR arrays, sized for
/// `real_requests + padding` rows and `real_tokens + pad_tokens` tokens, and
/// `kv_page_indices` has `padding` free entries past `kv_page_indptr[
/// real_requests]`. `custom_mask` and `custom_mask_indptr` are both null or
/// both live.
#[allow(clippy::too_many_arguments)]
pub fn graph_pad_rows(
    ctx: &Ctx,
    qo_indptr: *mut u32,
    kv_page_indptr: *mut u32,
    kv_page_indices: *mut u32,
    kv_last_page_lens: *mut u32,
    tokens: *mut u32,
    positions: *mut u32,
    row_valid: *mut u8,
    custom_mask: Option<NonNull<u8>>,
    custom_mask_indptr: Option<NonNull<i32>>,
    real_mask_bytes: i32,
    real_requests: i32,
    real_tokens: i32,
    padding: i32,
    pad_tokens: i32,
    pad_page: u32,
) -> Result<(), Refusal> {
    // A real guard: at `pad_tokens < padding` the kernel's `base =
    // pad_tokens / padding` is zero, so lanes past `extra` get a zero-length
    // last page while still consuming one.
    if pad_tokens < padding {
        return Err(Refusal::Narrow { what: "pad_tokens, in lanes", at: i64::from(pad_tokens) });
    }
    // A block-width ceiling: past the device's 1024-thread maximum this
    // fails inside `cudaGetLastError`, launch already issued.
    if padding > MAX_BLOCK {
        return Err(Refusal::Wide {
            what: "padding, as one block's threads",
            at: i64::from(padding),
            max: i64::from(MAX_BLOCK),
        });
    }
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "layout/graph_pad.cuh",
            "::pie::layout::graph_pad_rows",
            Launch::grid([1, 1, 1], [padding.unsigned_abs(), 1, 1]),
            &[
                qo_indptr.arg(),
                kv_page_indptr.arg(),
                kv_page_indices.arg(),
                kv_last_page_lens.arg(),
                tokens.arg(),
                positions.arg(),
                row_valid.arg(),
                custom_mask.arg(),
                custom_mask_indptr.arg(),
                real_mask_bytes.arg(),
                real_requests.arg(),
                real_tokens.arg(),
                padding.arg(),
                pad_tokens.arg(),
                pad_page.arg(),
            ],
        )
    }
}

/// One block's max threads on every architecture this crate targets —
/// `threadIdx.x` is an `int`.
const MAX_BLOCK: i32 = 1024;

/// A packed gate/up bank cut in halves — `layout/split_gate_up.cuh`.
///
/// `[ceil(inter / 256), n_tokens]`, channel axis on `grid.x` — this is
/// `LaunchRule::SplitPacked`'s order, not `ElementwiseRows`'.
///
/// Not `layout::split_bf16_rows` with equal dims, though both write the same
/// bytes: that routine is one block per row, right for a head-wide row and
/// wrong for an MLP intermediate (112 blocks per token at `inter = 28672`
/// there, one block per token here).
///
/// # Safety
///
/// `packed` addresses `n_tokens * 2 * inter` live bf16 elements and `gate_out`
/// and `up_out` `n_tokens * inter` writable ones each. All live on `ctx`'s
/// stream, which must outlive the launch.
pub fn split_gate_up_bf16(
    ctx: &Ctx,
    packed: *const c_void,
    gate_out: *mut c_void,
    up_out: *mut c_void,
    n_tokens: i32,
    inter: i32,
) -> Result<(), Refusal> {
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "layout/split_gate_up.cuh",
            "::pie::layout::split_gate_up<::pie::bf16>",
            Launch::grid(
                [inter.unsigned_abs().div_ceil(BLOCK), n_tokens.unsigned_abs(), 1],
                [BLOCK, 1, 1],
            ),
            &[
                packed.cast::<bf16>().arg(),
                gate_out.cast::<bf16>().arg(),
                up_out.cast::<bf16>().arg(),
                inter.arg(),
            ],
        )
    }
}

/// MXFP4's group width — `transcode.cuh`'s `kGroup = 32`.
///
/// The two transcodes below refuse a `cols` that isn't a whole multiple of
/// it: the kernel's `groups = cols / GROUP` truncates, so a trailing partial
/// block would otherwise be silently dropped rather than mis-encoded.
const MXFP4_GROUP: i32 = 32;

/// A BF16 rectangle transcoded to MXFP4 in one pass — `transcode.cuh`'s
/// `transcode_rowmajor_kernel<kGroup, DecodeBf16, EncodeMxfp4>`.
///
/// One block per row; the `float[32]` intermediate never leaves registers.
/// Same arithmetic as the two-step this replaces, minus that step's BF16
/// scratch buffer's HBM round-trip.
///
/// # Safety
///
/// `src` addresses `rows * cols` live bf16, `packed` `rows * cols / 2`
/// writable bytes and `scales` `rows * cols / 32` writable bytes.
pub fn transcode_bf16_to_mxfp4(
    ctx: &Ctx,
    src: *const bf16,
    packed: *mut u8,
    scales: *mut u8,
    rows: i32,
    cols: i32,
) -> Result<(), Refusal> {
    if cols % MXFP4_GROUP != 0 {
        return Err(Refusal::Narrow {
            what: "cols, in whole 32-element blocks",
            at: i64::from(cols),
        });
    }
    let decode = quant::transcode::DecodeBf16 { src: addr(src), cols };
    let encode =
        quant::transcode::EncodeMxfp4 { packed: addr(packed), scales: addr(scales), cols };
    // SAFETY: the caller's assertion, forwarded; both aggregates are bound to
    // local bindings that outlive the launch call, per `jit::abi`'s `Abi::arg`.
    unsafe {
        ctx.launch(
            "quant/transcode.cuh",
            "::pie::transcode::transcode_rowmajor_kernel<\
                 ::pie::transcode::EncodeMxfp4::kGroup,::pie::transcode::DecodeBf16,::pie::transcode::EncodeMxfp4>",
            Launch::per_row(rows.unsigned_abs(), BLOCK),
            &[decode.arg(), encode.arg(), cols.arg()],
        )
    }
}

/// A block-scaled FP8 E4M3 checkpoint transcoded to MXFP4 in one pass —
/// `transcode.cuh`'s `transcode_rowmajor_kernel<kGroup, DecodeFp8E4m3PerGroup,
/// EncodeMxfp4>`.
///
/// The pair `TransformFusion::Fp8ToMxfp4` means: it saves what the two-step
/// it collapses spends, a whole BF16 copy of the tensor round-tripped
/// through HBM. No caller yet — the driver still asserts it has none.
///
/// # Safety
///
/// `src` addresses `rows * cols` live E4M3 bytes and `src_scales` the f32
/// plane they index — `ceil(rows / group_size)` rows of `scale_cols`, which
/// is `ceil(cols / group_size)`. `packed` and `scales` are `rows * cols / 2`
/// and `rows * cols / 32` writable bytes.
#[allow(clippy::too_many_arguments)]
pub fn transcode_fp8_e4m3_per_group_to_mxfp4(
    ctx: &Ctx,
    src: *const u8,
    src_scales: *const f32,
    packed: *mut u8,
    scales: *mut u8,
    rows: i32,
    cols: i32,
    group_size: i32,
) -> Result<(), Refusal> {
    if cols % MXFP4_GROUP != 0 {
        return Err(Refusal::Narrow {
            what: "cols, in whole 32-element blocks",
            at: i64::from(cols),
        });
    }
    let decode = quant::transcode::DecodeFp8E4m3PerGroup {
        src: addr(src),
        scales: addr(src_scales),
        cols,
        scale_cols: (cols + group_size - 1) / group_size,
        group_size,
    };
    let encode =
        quant::transcode::EncodeMxfp4 { packed: addr(packed), scales: addr(scales), cols };
    // SAFETY: as `transcode_bf16_to_mxfp4`'s.
    unsafe {
        ctx.launch(
            "quant/transcode.cuh",
            "::pie::transcode::transcode_rowmajor_kernel<\
                 ::pie::transcode::EncodeMxfp4::kGroup,::pie::transcode::DecodeFp8E4m3PerGroup,::pie::transcode::EncodeMxfp4>",
            Launch::per_row(rows.unsigned_abs(), BLOCK),
            &[decode.arg(), encode.arg(), cols.arg()],
        )
    }
}

// ===========================================================================
// Pins for the four columns this module now derives: with no arm to diff
// against, the derived column IS the binding, so a shifted operand slot
// would rebind a launch silently.
// ===========================================================================
const _: () = {
    // `attn::split_qkv_bf16` and `mlp::sigmoid_gate_inplace_bf16` are pinned
    // where their families' other rows are.
    let d = <split_q_gate_bf16 as kernels::Derivation>::DERIVED;
    assert!(d.len() == 4);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // `Out(1)`, not `In(1)`: the gate half is a result of the split, though
    // both halves read as `const T*` in the C++.
    assert!(matches!(d[2].source, Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    // The head pitch, carried by no operand; a fifth entry here would mean
    // something started stating a number the geometry already implies.
    assert!(kernels::source_is_named(&d[3].source, <kernels::keys::PerHeadDim as kernels::keys::Fact>::KEY));

    let d = <qwen_gdn_post_conv_prep_bf16 as kernels::Derivation>::DERIVED;
    assert!(d.len() == 15);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(d[2].source, Some(kernels::Source::Slot(kernels::Kind::In, 2))));
    // The positional bank, not `Weight<0>`/`Weight<1>`: `OpKind::GdnPrep`
    // also lands these two on `spec.weight`/`weight2`, so the named form
    // would compile and bind the same tensors from the other table.
    assert!(matches!(d[3].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 0))));
    assert!(matches!(d[4].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 1))));
    // Five results in the order `builder::gdn_prep` pushes them: the two
    // norms are `[N, K_h, K_d]`, the last three `[N, V_h, ..]` — swapping
    // within either group is a wrong shape, not a compile error.
    assert!(matches!(d[5].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[6].source, Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    assert!(matches!(d[7].source, Some(kernels::Source::Slot(kernels::Kind::Out, 2))));
    assert!(matches!(d[8].source, Some(kernels::Source::Slot(kernels::Kind::Out, 3))));
    assert!(matches!(d[9].source, Some(kernels::Source::Slot(kernels::Kind::Out, 4))));
    // `k_h`/`v_h` and `k_d`/`v_d` differ by one letter and grid different
    // launches — swapping either pair compiles clean. `n` has no entry here:
    // it comes from `qkv_post`'s own row count, not a fact.
    assert!(kernels::source_is_named(&d[10].source, <kernels::keys::GdnKHeads as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[11].source, <kernels::keys::GdnVHeads as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[12].source, <kernels::keys::GdnKDim as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[13].source, <kernels::keys::GdnVDim as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[14].source, <kernels::keys::GdnConvDim as kernels::keys::Fact>::KEY));
};
