//! Launchers written for a DRIVER-SIDE caller, and NOT a family.
//!
//! Every kernel fired here belongs to `attn`, `norm`, `layout`, `mlp` or
//! `ssm`; what this module holds is the geometry a driver-side caller wants
//! for it, in the vocabulary such a caller has — `void*` where the family's
//! own routine takes `bf16*`, and a row count read off a layer. So there is
//! no `FAMILY` and no `ROUTINES` here.
//!
//! # The sentence that stood here, and why it was wrong twice
//!
//! It read: *"these are plain `pub fn`s `driver-cuda` calls by path, no
//! statement names them, and there is nothing for a trace to resolve."*
//!
//! **Statements name four of the six.** `model-compiler`'s `semantic()`
//! lowers `OpKind::SplitQkv` (outside a peel tail) to `attn::split_qkv_bf16`,
//! `SplitQGate` to `layout::split_q_gate_bf16`, `SigmoidGateMul` to
//! `mlp::sigmoid_gate_inplace_bf16` and `GdnPrep` to
//! `ssm::qwen_gdn_post_conv_prep_bf16` — gemma-4 and the llama-like anchor
//! for the first, gemma3n and the qwen3.5 hybrid for the others. **And
//! `driver-cuda` calls none of the six by path.** The other two,
//! [`add_bias`] and [`rmsnorm_gated_fp32_in`], forward to `norm`
//! routines that were already declared, which is why they are two rather than
//! six.
//!
//! # Where the four are declared, and why not here
//!
//! In `attn`, `layout`, `mlp` and `ssm` — one `driver_bound!` line each,
//! naming the `fn` below. `Family::symbol` is the module path's FIRST SEGMENT
//! after the crate root plus the routine's name, so a `Family` declared in
//! this module would offer `driver_internal::split_qkv_bf16` and three more
//! strings no lowering emits. The namespace follows the declaration, so the
//! declaration goes to the namespace and the body stays where a driver-side
//! caller's vocabulary belongs.
//!
//! Two shapes were rejected. **Moving the `fn`s into their families** would
//! put `void*` parameter lists beside typed ones and lose the distinction
//! this module is named for — `attn::split_qkv_bf16_devwin` takes `*const
//! bf16` and is a `routine!`; its non-window twin takes `*const c_void` and
//! is not, and the difference is the point. **Declaring them with `routine!`
//! instead** would compile — every parameter here is `Arg`, `*const c_void`
//! included, through `jit::abi`'s `ptr_abi!(c_void, ..)`, and it was tried to
//! be sure — and would derive an `args`/`spelling` table describing the CAST
//! LAYER: `Ty::Buf` and `"const void*"` on the four pointers of
//! [`split_qkv_bf16`], where the `__global__` takes `const ::pie_cuda_driver::
//! kernels::pie::bf16*` and `attn`'s own `split_qkv_bf16_devwin` derives
//! exactly that. A column right about the `fn` and wrong about the launch is
//! worse than an empty one, and `KernelSig::args`' own doc reads empty as
//! UNSTATED.
//!
//! # Declared is not armed
//!
//! A `driver_bound!` line puts the symbol in [`crate::sigs`], so every
//! consumer that looks a lowered symbol up in that table now finds one. It
//! adds no arm. `driver-cuda`'s `bind/arms/` has an entry for none of the
//! four, so a fire naming one still refuses with `DispatchRefusal::NoArm`,
//! exactly as before — `driver-cuda/tests/executor_bind.rs` is where that is
//! recorded. The arm is that crate's to write.
//!
//! **The coverage rule was never the thing refusing**, and it is worth being
//! exact about that because the obvious reading is wrong. `check_plan`'s
//! *every launched symbol must be declared* walks `OpKind::Launch` and
//! nothing else; these four arrive through `semantic()`, from op kinds that
//! carry no kernel string, so the rule never looked at them — which is how
//! gemma-4 went on loading for as long as it did with all four undeclared.
//! What the declaration closes is the gap between the two paths: the day a
//! text states one outright — which is the shape the Metal builders already
//! use — the rule finds a row instead of refusing the model at load.

use core::ffi::c_void;
use core::ptr::NonNull;

use crate::jit::{Ctx, Launch};
use crate::jit::Abi;
use crate::jit::abi::bf16;
use crate::{norm, quant};
use kernels::Refusal;

/// `runtime/launch.rs` — `const BLOCK: u32 = 256;`, the block the pointwise
/// launches here take.
///
/// It is also the 256 that four of the deleted launchers recovered in the
/// section at the end of this file each spelled for themselves —
/// `geometry.cu`'s `kThreads`, `gather_tokens.cu`'s `threads`,
/// `split_gate_up.cu`'s `BLOCK` and `transcode.cu`'s `kBlock`. Naming one
/// constant rather than transcribing four is not a merge of four decisions:
/// no two of them differ, each is the block a pointwise kernel gets by
/// default, and four spellings would leave a reader checking whether any one
/// of them meant something.
const BLOCK: u32 = 256;

/// The QKV split — `attn::split_qkv_bf16`, `attn/split_packed.cuh:74`.
///
/// # Safety
///
/// `packed` is `[n_tokens, q_dim + 2 * kv_dim]` bf16; `q_out` is
/// `[n_tokens, q_dim]` and `k_out`/`v_out` are `[n_tokens, kv_dim]`, all
/// bf16 and all writable. All four live on `ctx`'s stream, which must
/// outlive the launch.
pub fn split_qkv_bf16(
    ctx: &Ctx,
    packed: *const c_void,
    q_out: *mut c_void,
    k_out: *mut c_void,
    v_out: *mut c_void,
    n_tokens: i32,
    q_dim: i32,
    kv_dim: i32,
) -> Result<(), Refusal> {
    if q_dim <= 0 && kv_dim <= 0 {
        return Err(Refusal::Empty { what: "q_dim and kv_dim" });
    }
    let width = q_dim.max(kv_dim).unsigned_abs();
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "attn/split_packed.cuh",
            "::pie::attn::split_qkv<::pie::bf16>",
            Launch::grid([width.div_ceil(BLOCK), n_tokens.unsigned_abs(), 1], [BLOCK, 1, 1]),
            &[
                packed.cast::<bf16>().arg(),
                q_out.cast::<bf16>().arg(),
                k_out.cast::<bf16>().arg(),
                v_out.cast::<bf16>().arg(),
                q_dim.arg(),
                kv_dim.arg(),
            ],
        )
    }
}

/// The bias add — `norm::add_bias_bf16`, `norm/add_bias.cuh`.
///
/// The geometry is [`norm::add_bias_bf16`]'s own, so this is the cast and
/// nothing else: the driver hands `void*` where the routine takes `bf16*`.
///
/// # Safety
///
/// `out` is `[num_rows, dim]` bf16 and writable, `bias` is `[dim]` bf16.
/// Both live on `ctx`'s stream.
pub fn add_bias_bf16(
    ctx: &Ctx,
    out: *mut c_void,
    bias: *const c_void,
    num_rows: i32,
    dim: i32,
) -> Result<(), Refusal> {
    norm::add_bias::<bf16>(ctx, out.cast::<bf16>(), bias.cast::<bf16>(), num_rows, dim)
}

/// Qwen3.5's post-convolution split — `ssm::qwen_gdn_post_conv_prep_bf16`,
/// `gated_delta_net.cu:139-168`.
///
/// # Safety
///
/// `qkv_post` is `[N, conv_dim]` bf16; `a`, `b` and `dt_bias` are bf16 over
/// `[N, V_h]`, `[N, V_h]` and `[V_h]`; `a_log` is `[V_h]` fp32; the five
/// outputs are writable for `[N, K_h, K_d]`, `[N, K_h, K_d]`,
/// `[N, V_h, V_d]`, `[N, V_h]` and `[N, V_h]`. All live on `ctx`'s stream.
#[allow(clippy::too_many_arguments)]
pub fn qwen_gdn_post_conv_prep_bf16(
    ctx: &Ctx,
    qkv_post: *const c_void,
    a: *const c_void,
    b: *const c_void,
    a_log: *const c_void,
    dt_bias: *const c_void,
    q_norm_kh: *mut f32,
    k_norm_kh: *mut f32,
    v_fp32: *mut f32,
    g_log_out: *mut f32,
    beta_out: *mut f32,
    n: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    conv_dim: i32,
) -> Result<(), Refusal> {
    /// `gated_delta_net.cu:154` — `constexpr int BLOCK = 128;`.
    const PREP_BLOCK: u32 = 128;
    #[allow(clippy::cast_precision_loss)]
    let q_scale = (k_d as f32).sqrt().recip();
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "ssm/gated_delta_net_prep.cuh",
            "::pie::ssm::qwen_gdn_qk_norm<::pie::bf16, 128>",
            Launch::grid([n.unsigned_abs(), k_h.unsigned_abs(), 1], [PREP_BLOCK, 1, 1]),
            &[
                qkv_post.arg(),
                q_norm_kh.arg(),
                k_norm_kh.arg(),
                k_h.arg(),
                k_d.arg(),
                conv_dim.arg(),
                q_scale.arg(),
            ],
        )?;
    }
    // SAFETY: the caller's assertion, forwarded. No barrier between the two
    // launches: the second reads `qkv_post` again rather than anything the
    // first wrote, so the stream's own ordering is all they need.
    unsafe {
        ctx.launch(
            "ssm/gated_delta_net_prep.cuh",
            "::pie::ssm::qwen_gdn_v_g_beta<::pie::bf16, 128>",
            Launch::grid([n.unsigned_abs(), v_h.unsigned_abs(), 1], [PREP_BLOCK, 1, 1]),
            &[
                qkv_post.arg(),
                a.arg(),
                b.arg(),
                a_log.cast::<f32>().arg(),
                dt_bias.arg(),
                v_fp32.arg(),
                g_log_out.arg(),
                beta_out.arg(),
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
/// `layout/deinterleave.cuh:130`.
///
/// # Safety
///
/// `packed` is `[n, num_heads, 2 * head_dim]` bf16; `q_out` and `gate_out`
/// are `[n, num_heads, head_dim]` bf16 and writable. All live on `ctx`'s
/// stream.
pub fn split_q_gate_bf16(
    ctx: &Ctx,
    packed: *const c_void,
    q_out: *mut c_void,
    gate_out: *mut c_void,
    n: i32,
    num_heads: i32,
    head_dim: i32,
) -> Result<(), Refusal> {
    let block = if head_dim < 128 { 64 } else { 128 };
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "layout/deinterleave.cuh",
            "::pie::layout::split_q_gate<::pie::bf16>",
            Launch::grid([n.unsigned_abs(), num_heads.unsigned_abs(), 1], [block, 1, 1]),
            &[
                packed.cast::<bf16>().arg(),
                q_out.cast::<bf16>().arg(),
                gate_out.cast::<bf16>().arg(),
                n.arg(),
                num_heads.arg(),
                head_dim.arg(),
            ],
        )
    }
}

/// That gate applied — `mlp::sigmoid_gate_inplace_bf16`, `mlp/swiglu.cuh:261`.
///
/// # Safety
///
/// `x` and `gate` are both `num_elements` bf16 elements; `x` is writable and
/// is read and written by the same threads. Both live on `ctx`'s stream.
pub fn sigmoid_gate_inplace_bf16(
    ctx: &Ctx,
    x: *mut c_void,
    gate: *const c_void,
    num_elements: i32,
) -> Result<(), Refusal> {
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "mlp/swiglu.cuh",
            "::pie::mlp::sigmoid_gate_inplace<::pie::bf16>",
            Launch::flat(num_elements.unsigned_abs(), BLOCK),
            &[x.cast::<bf16>().arg(), gate.cast::<bf16>().arg(), num_elements.arg()],
        )
    }
}

/// The gated norm with an FP32 `x` — `norm::rmsnorm_gated_fp32_in_bf16`,
/// `norm/rmsnorm.cuh:763`.
///
/// One block per row is [`norm::rmsnorm_gated_fp32_in_bf16`] at
/// `per_head_dim = 0`, so the geometry is that routine's. The two refusals
/// are not: the routine takes a row count it trusts, and the driver's
/// `hidden` and `num_rows` are read off a layer.
///
/// # Safety
///
/// `x` is `[num_rows, hidden]` fp32; `gate` is `[num_rows, hidden]` bf16;
/// `weight` is `[hidden]` fp32; `y` is `[num_rows, hidden]` bf16 and
/// writable. All live on `ctx`'s stream.
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
    norm::rmsnorm_gated_fp32_in::<bf16>(
        ctx,
        x.cast::<f32>(),
        gate.cast::<bf16>(),
        weight.cast::<f32>(),
        y.cast::<bf16>(),
        num_rows,
        hidden,
        0,
        eps,
    )
}

// ===========================================================================
// The seven launchers for seven `__global__`s that had lost theirs
//
// `kernels/layout/{geometry,gather_tokens,graph_pad,split_gate_up}.cuh` and
// `kernels/quant/transcode.cuh` hold seven kernels between them, and every
// `<<<>>>` any of them ever had was in a `.cu` beside it in the archive
// crate. Those five files are gone -- four at `2ef431d02` and
// `gather_tokens.cu` at `cd5cebd3d`, with the archive crate `kernels-cuda`
// itself following at
// `85c6c674b`, by which point it held no `.cu` at all. The device text stayed,
// because `src/source.rs` carries every file under `kernels/` whether or not anything
// names it, and three of the five headers went on describing a launcher that
// had been deleted underneath them.
//
// `every_carried_file_is_reachable` is what said so: no compile in this crate
// could arrive at any of the five. A root plus a launcher is what answers
// that. The roots are beside their families, in `layout.rs` and `quant.rs`,
// and the launchers are the seven `fn`s below.
//
// **Here and not in a `Family`, and the five headers say why in their own
// words.** `geometry.cuh`: its two kernels are *"called by the DRIVER while it
// composes a wave, not by a statement, so there is no fire whose operands a
// `Source` could name and inventing one would be a contract nothing checks."*
// That sentence is the whole test, and the other four pass it the same way --
// a gather plan the driver resolved slot ids for, a pad-lane CSR written
// during graph capture, an MLP bank split no model text names, a transcode
// whose operands are assembled from a loader plan. A `routine!` line would
// offer a trace seven symbols no trace has any way to state.
//
// **None of the seven has a caller**, and that is the expected shape rather
// than a gap: every caller they had was C++ in a crate that no longer exists,
// and what replaces each is a `driver-cuda` decision -- for the two
// transcodes, one bit in `StorageTarget::fusion_mask` that
// `driver-cuda/src/weights/plan.rs:212` currently asserts is zero. What the
// launchers buy before that day comes is that the text is COMPILED:
// `every_instantiation_compiles` hands the eight template-ids these seven
// name to NVRTC, on any box that can load `libnvrtc`.
//
// The geometry of each is the deleted launcher's, recovered from the last
// commit that held it (`2ef431d02^`, and `cd5cebd3d^` for `gather_tokens.cu`)
// rather than re-derived, so that a launch here is the launch whatever tests
// those kernels once passed were run against. Where a refusal is NOT the old
// launcher's, it says so.
// ===========================================================================

/// A device address as a by-value aggregate carries one.
///
/// `attn::fa2`'s `addr` for the same reason it gives: the aggregate holds a
/// `u64` rather than a pointer, because the host may never dereference it and
/// the device's pointer is 64-bit whatever the host's is.
fn addr<T>(p: *const T) -> crate::jit::abi::DevicePtr {
    p as usize as u64
}

/// `kv_len[r]` out of the CSR page descriptors —
/// `layout/geometry.cuh:48`.
///
/// One thread per request, and the arithmetic is bit-identical to the host
/// formula the kernel's own comment names, which is the point of it: a
/// device-composed wave has no host copy of `kv_len` to derive from.
///
/// # Safety
///
/// `kv_page_indptr` addresses `num_requests + 1` live `u32`s and
/// `kv_last_page_lens` `num_requests` of them; `kv_len` is `num_requests`
/// writable `u32`s. All live on `ctx`'s stream, which must outlive the launch.
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
/// `layout/geometry.cuh:69`.
///
/// The dictionary is uploaded per wave, so this is the indirection a
/// device-composed page list cannot do on the host. An out-of-range slot
/// resolves to `0xFFFFFFFF` rather than wrapping, which is the kernel's
/// choice and is why nothing here refuses one: a corrupt slot must fail
/// visibly at the gather, not be caught by a host bound this launcher has no
/// way to check per element.
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
/// One block of `member_count` threads, once per wave, and that shape is the
/// kernel's own reason for having no row: the output CSR is one shared object
/// and a rule that launched one block per row would have every block writing
/// its own copy of it.
///
/// The `EnvelopeMember` records are three `u32`s each, in the order
/// `geometry.cuh` declares them, and the caller writes them — there is no Rust
/// mirror of the struct for the reason the kernel's own comment gives, which
/// is [`gather_tokens`]'s: two spellings of one layout agree until a field
/// moves.
///
/// **No caller, and the kernel has never run.** That is not the state its two
/// neighbours are in by accident either; `geometry.cuh`'s header names the one
/// thing standing between all three and a caller, and it is `driver-cuda`'s
/// per-instance channel rings rather than anything about this arithmetic.
///
/// # Safety
///
/// `members` addresses `member_count` live `EnvelopeMember` records;
/// `traced_page_indptr` `2 * member_count` live `u32`s; `traced_kv_len`,
/// `traced_w_slot` and `token_ids` `member_count` each; `traced_pages` the
/// sum of the members' `page_bound`s. `kv_page_indptr` is `member_count + 1`
/// writable `u32`s, `kv_page_indices` that same sum, `kv_last_page_lens` and
/// `w_slot_out` `member_count` each, and `row_valid` `member_count` writable
/// bytes. `kills` is null or one writable `u32`. All live on `ctx`'s stream,
/// which must outlive the launch.
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
    // `graph_pad_rows`' ceiling, for its reason: `<<<1, member_count>>>` is a
    // block width, and a wave wider than the device's maximum fails inside
    // `cudaGetLastError` with the launch already issued.
    if member_count > MAX_BLOCK.unsigned_abs() {
        return Err(Refusal::Wide {
            what: "member_count, as one block's threads",
            at: i64::from(member_count),
            max: i64::from(MAX_BLOCK),
        });
    }
    // One `u32` per member, which the kernel uses first as each member's page
    // count and then, after the scan, as its offset into the composed page
    // list. Stated here because `extern __shared__` sizes nothing itself.
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

/// The KV compaction copy — `layout/gather_tokens.cuh:71` or `:95`, and the
/// choice between them is this function.
///
/// One block per `(op, layer)`, which is the third grid axis
/// `gather_tokens.cuh`'s header names as one reason no `LaunchRule` could
/// state it. The other reason is the branch below: `token_stride % 8 == 0`
/// is a test on a value read off the layer, and a `Source` cannot produce it.
///
/// The pointers are `u16` and not `bf16` for `gather_rows`' reason -- both
/// kernels are pure copies, neither ever converts to float, and a tag type
/// that promises arithmetic nobody performs is a tag type that invites it.
/// The vectorised arm's parameters are `int4*`, and nothing but the address
/// crosses, so the two arms are handed the same pointer under two readings of
/// it. The deleted launcher spelled that as a `reinterpret_cast` at the call;
/// here there is no Rust type for `int4` to cast to, so the pointer goes as
/// an opaque one and the reading is the kernel's parameter list alone.
///
/// # Safety
///
/// `k_pages` and `v_pages` address the layer's whole page pool as `u16`, and
/// every op in `ops` names spans inside it. `ops` addresses `num_ops` live
/// `GatherTokenOp` records -- five `u32`s each, `gather_tokens.cuh:58`, which
/// is the layout the caller must have written. All live on `ctx`'s stream,
/// which must outlive the launch.
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

    // Eight bf16 is one `int4`, so the vectorised arm needs every span base
    // and length to be a multiple of eight elements. The token stride carries
    // both -- an op's offsets are in tokens -- and the layer stride is the
    // only other term in a base address, which is why exactly these two are
    // tested and the page stride is not: it is `token_stride * page_size`.
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

/// The graph-lattice pad lanes' CSR — `layout/graph_pad.cuh:71`.
///
/// One block of `padding` threads, once per captured wave, and that shape is
/// the header's reason for having no row: every thread writes a different pad
/// lane of one shared CSR, so a rule that launched one block per row would
/// run the kernel once per row and race sixteen copies on the same words.
///
/// # Safety
///
/// Every pointer addresses the wave's own CSR arrays, sized for
/// `real_requests + padding` rows and `real_tokens + pad_tokens` tokens, and
/// `kv_page_indices` has `padding` free entries past `kv_page_indptr[
/// real_requests]`. `custom_mask` and `custom_mask_indptr` are both null or
/// both live. All live on `ctx`'s stream, which must outlive the launch.
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
    // The deleted launcher's second guard, and it is a real one: at
    // `pad_tokens < padding` the kernel's `base = pad_tokens / padding` is
    // zero and the lanes past `extra` take no token at all, which writes a
    // zero-length last page for a lane that still consumes one.
    if pad_tokens < padding {
        return Err(Refusal::Narrow { what: "pad_tokens, in lanes", at: i64::from(pad_tokens) });
    }
    // NOT the old launcher's, which had no ceiling: `<<<1, padding>>>` is a
    // block width, and a wave padded past the device's 1024-thread maximum
    // failed inside `cudaGetLastError` with the launch already issued.
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

/// `layout/graph_pad.cuh:87` — `threadIdx.x` is an `int`, and one block is
/// 1024 threads wide at most on every architecture this crate targets.
const MAX_BLOCK: i32 = 1024;

/// A packed gate/up bank cut in halves — `layout/split_gate_up.cuh:64`.
///
/// `[ceil(inter / 256), n_tokens]` with the CHANNEL axis on `grid.x`, which is
/// `LaunchRule::SplitPacked`'s order and not `ElementwiseRows`'. The header
/// records that the refusal which kept this kernel out of the row world had
/// named the wrong rule; what keeps it out of a `Family` is the other half of
/// that header -- nothing anywhere names this launcher, in any language.
///
/// It is not `layout::split_bf16_rows` with `left_dim == right_dim`, though
/// the two write the same bytes. That routine is one block per row, which is
/// the right shape for a row as wide as a head and the wrong one for an MLP
/// intermediate: at `inter = 28672` this grid is 112 blocks per token and
/// that one is a single block striding the whole row.
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

/// MXFP4's group width, `transcode.cuh:164` — `kGroup = 32`.
///
/// Named here because the two transcodes below refuse a `cols` that is not a
/// whole number of it, and that refusal is the host's: the kernel's `groups =
/// cols / GROUP` truncates, so a trailing partial block would be silently
/// dropped rather than mis-encoded.
const MXFP4_GROUP: i32 = 32;

/// A BF16 rectangle transcoded to MXFP4 in one pass —
/// `transcode.cuh:199` at `<kGroup, DecodeBf16, EncodeMxfp4>`.
///
/// One block per row, and the intermediate `float[32]` never leaves registers.
/// Against the two-step it replaces this is the same arithmetic -- the decode
/// rounds through BF16 deliberately -- with the BF16 scratch buffer's HBM
/// traffic removed.
///
/// # Safety
///
/// `src` addresses `rows * cols` live bf16, `packed` `rows * cols / 2`
/// writable bytes and `scales` `rows * cols / 32` writable bytes. All live on
/// `ctx`'s stream, which must outlive the launch.
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
    // SAFETY: the caller's assertion, forwarded. Both aggregates are bound by
    // reference to bindings of this frame, which outlive the launch call --
    // see `jit::abi`'s header for why `Abi::arg` takes `&self`.
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
/// `transcode.cuh:199` at `<kGroup, DecodeFp8E4m3PerGroup, EncodeMxfp4>`.
///
/// This is the pair the loader means by `TransformFusion::Fp8ToMxfp4`, and the
/// one that saves something: the two-step it collapses writes a whole BF16
/// copy of the tensor to HBM and reads it back. `model-loader`'s
/// `plan/passes/tile.rs:567` decides when it is wanted and
/// `driver-cuda/src/weights/plan.rs:212` asserts the driver has no such
/// kernel -- an assertion that is true until a caller for this appears.
///
/// # Safety
///
/// `src` addresses `rows * cols` live E4M3 bytes and `src_scales` the f32
/// plane they index -- `ceil(rows / group_size)` rows of the `scale_cols`
/// this derives, which is `ceil(cols / group_size)` and is the deleted
/// dispatch's own expression rather than a second rule. `packed` and
/// `scales` are `rows * cols / 2` and
/// `rows * cols / 32` writable bytes. All live on `ctx`'s stream, which must
/// outlive the launch.
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
    // SAFETY: the caller's assertion, forwarded. Both aggregates are bound by
    // reference to bindings of this frame, which outlive the launch call.
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
