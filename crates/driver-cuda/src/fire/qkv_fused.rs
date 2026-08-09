//! `attn/qkv_fused.cu`'s fused decode epilogue, in Rust — the whole file.
//!
//! One public launcher over FOUR kernels, and the reason this file outlived
//! every other `attn` launcher: the choice between them is a host `if` chain
//! on `head_dim` that falls THROUGH to a different kernel with a different
//! `LaunchRule`, and `Specialisation::agrees` forbids an arm that changes the
//! rule. `families/attn.rs` states the refusal in two lines — *"this row
//! states `RowsPackedHeadsNarrow` and the warp triple below states
//! `WarpPackedHeads`, which is the whole refusal"* — and adds the leg that
//! matters most: *"lifting it would not land the row, because `head_dim == 64
//! | 128 | 256` is still unspellable."*
//!
//! A host program spells it. That is the whole of what this file is.
//!
//! # The shape, exactly as the C++ had it
//!
//! ```text
//! head_dim == 64  ─┐
//! head_dim == 128 ─┼─ warp form, <<<ceil(units / 8), 256>>>
//! head_dim == 256 ─┘
//! anything else   ─── block form, <<<dim3(num_requests, q + kv), 128>>>
//! ```
//!
//! and inside every arm, a second choice:
//!
//! ```text
//! rope_table != nullptr ── USE_ROPE_TABLE = true
//! rope_table == nullptr ── USE_ROPE_TABLE = false
//! ```
//!
//! Eight instantiations, six of which are the warp form. Four of those six
//! had no device row before this change: `families/attn.rs` stated the warp
//! form at `_d128` only, and `qkv_fused.cu`'s own header named the
//! consequence — *"`head_dim == 64` and `head_dim == 256` reach a `<<<>>>`
//! below that no row names, and the block form's fallthrough has no row
//! either"*. The block form's two arms did have rows (`#rope`/`#norope`); the
//! four warp expansions are written in the same change as this file.
//!
//! # `num_requests` is an operand of the warp form and not of the block form
//!
//! Not an inconsistency — the grid's. The block form gets its request index
//! from `blockIdx.x` and needs no count; the warp form flattens
//! `(request, head)` onto one axis, recovers `r = unit / total_qk_heads` at
//! `qkv_fused.cuh:267`, and has to be told where the units stop. Both are
//! `Dims::rows` on the rule side. The two argument lists below differ in
//! exactly that one cell and in `head_dim`, which the warp form carries as a
//! TEMPLATE argument and so must not also be passed.
//!
//! # `win` is always null on this path
//!
//! `qkv_decode_qk_norm_rope_write_kv_bf16` passes `/*win=*/nullptr` at
//! `qkv_fused.cu:189`. The `_devwin` twin that passed a real window was
//! deleted in an earlier pass, so the parameter survives on the shared
//! dispatch and has exactly one caller passing exactly one value. It is kept
//! on [`qkv_decode_fused_dispatch`] rather than folded away, because the
//! kernels read it per row and a future peel-aware caller is a caller of that
//! function and not a new kernel.

use std::ffi::c_void;

use kernels_cuda_new::runtime::{ArgValue, Launch};

/// `attn::qkv_decode_qk_norm_rope_write_kv_bf16` — the table symbol.
pub const QKV_DECODE_SYMBOL: &str = "attn::qkv_decode_qk_norm_rope_write_kv_bf16";

/// `qkv_fused.cu:51` — `constexpr int WARP_BLOCK = 256;`.
const WARP_BLOCK: u32 = 256;

/// `qkv_fused.cu:105` — `constexpr int BLOCK = 128;`, the block form's width.
const BLOCK: u32 = 128;

/// Warps per block: `WARP_BLOCK / 32`.
///
/// The warp form assigns one WARP per `(request, head)` unit, so the grid is
/// `ceil(total_units / WARPS_PER_BLOCK)` and not `ceil(total_units / 256)`.
/// Getting this wrong by the factor 32 is a grid eight times too large, which
/// does not fault — the kernel bounds itself on `total_units` — so it is
/// named rather than inlined.
const WARPS_PER_BLOCK: u32 = WARP_BLOCK / 32;

/// Which form and which rope arm a launch took.
///
/// `#[must_use]` for `fire/gemv.rs`' reason, and it carries the arm because
/// the arm is the whole content of this launcher: a caller that believes it
/// got the warp form and got the block form has a performance bug with no
/// symptom.
#[must_use]
pub enum QkvDecode {
    /// The warp form, at the head width its template argument names.
    Warp {
        /// 64, 128 or 256 — the `HEAD_DIM` template argument.
        head_dim: i32,
        /// Whether the `USE_ROPE_TABLE = true` instantiation was fired.
        rope_table: bool,
    },
    /// The block form — the fallthrough, for any other head width.
    Block {
        /// Whether the `USE_ROPE_TABLE = true` instantiation was fired.
        rope_table: bool,
    },
    /// Nothing was launched.
    Declined(QkvDecline),
}

/// The one way the fused decode epilogue declines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QkvDecline {
    /// `qkv_fused.cu:50` — `num_requests == 0`.
    ///
    /// Transcribed as `== 0` and then widened to `<= 0`: the C++ tested
    /// equality, and a negative request count would have produced a negative
    /// `total_units` and a grid expression that underflows `unsigned`. The
    /// C++ could not reach it; the Rust would panic on the cast, so the guard
    /// covers it. That is the one place this port is not a transcription, and
    /// it is noted rather than silent.
    NoRequests,
}

/// The eight instantiations, by `(head_dim, rope_table)`.
///
/// `None` for a head width the warp form was not compiled for, which is the
/// fallthrough to the block form.
fn warp_symbol(head_dim: i32, rope_table: bool) -> Option<&'static str> {
    Some(match (head_dim, rope_table) {
        (64, true) => "attn::qkv_decode_qk_norm_rope_write_kv_warp_d64#rope",
        (64, false) => "attn::qkv_decode_qk_norm_rope_write_kv_warp_d64#norope",
        (128, true) => "attn::qkv_decode_qk_norm_rope_write_kv_warp_d128#rope",
        (128, false) => "attn::qkv_decode_qk_norm_rope_write_kv_warp_d128#norope",
        (256, true) => "attn::qkv_decode_qk_norm_rope_write_kv_warp_d256#rope",
        (256, false) => "attn::qkv_decode_qk_norm_rope_write_kv_warp_d256#norope",
        _ => return None,
    })
}

/// The block form's two arms.
fn block_symbol(rope_table: bool) -> &'static str {
    if rope_table {
        "attn::qkv_decode_qk_norm_rope_write_kv#rope"
    } else {
        "attn::qkv_decode_qk_norm_rope_write_kv#norope"
    }
}

/// `attn/qkv_fused.cu:31` — `qkv_decode_fused_dispatch`, the `static` one.
///
/// The whole host program: pick the form on `head_dim`, pick the arm on
/// `rope_table`, compute the grid, fire.
///
/// ```text
/// :50    if (num_requests == 0) return;
/// :51    constexpr int WARP_BLOCK = 256;
/// :52    const int total_units = num_requests * (num_q_heads + num_kv_heads);
/// :53    dim3 warp_grid((total_units + (WARP_BLOCK / 32) - 1) / (WARP_BLOCK / 32));
/// :57-58 ...write_kv_warp<(HEAD_DIM_VALUE), true ><<<warp_grid, WARP_BLOCK, 0, stream>>>
/// :70-71 ...write_kv_warp<(HEAD_DIM_VALUE), false><<<warp_grid, WARP_BLOCK, 0, stream>>>
/// :105   constexpr int BLOCK = 128;
/// :106   dim3 grid(num_requests, num_q_heads + num_kv_heads);
/// :108   ...write_kv<BLOCK, true ><<<grid, BLOCK, 0, stream>>>
/// :134   ...write_kv<BLOCK, false><<<grid, BLOCK, 0, stream>>>
/// ```
///
/// The C++ wrote the two rope arms once per head width through
/// `LAUNCH_QKV_DECODE_POST_WARP`, a `do { } while (0)` macro expanded three
/// times. The Rust picks the symbol instead, which is the same three-by-two
/// table with the duplication removed and the argument list written once.
///
/// # Panics
///
/// If the kernel table and this driver disagree; see [`super::hand::fire`].
/// A `head_dim` of 64, 128 or 256 whose instantiation is missing panics with
/// that symbol named rather than falling through to the block form — a
/// refusal is never a fallback, and the block form at those widths would be
/// correct and several times slower, which is the failure that reports
/// nothing.
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch;
/// `rope_table`, `w_page`, `w_off`, `row_valid` and `win` may be null, and
/// the kernels test each. `stream` is the caller's stream.
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
pub unsafe fn qkv_decode_fused_dispatch(
    packed: *const c_void,
    q_out: *mut c_void,
    k_pages: *mut c_void,
    v_pages: *mut c_void,
    q_weight: *const c_void,
    k_weight: *const c_void,
    positions: *const i32,
    rope_table: *const f32,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    w_page: *const u32,
    w_off: *const u32,
    row_valid: *const u8,
    win: *const u32,
    num_requests: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    page_size: i32,
    hnd_layout: bool,
    theta: f32,
    eps: f32,
    stream: *mut c_void,
) -> QkvDecode {
    // `qkv_fused.cu:50`. See `QkvDecline::NoRequests` for why `<=` and not
    // `==`.
    if num_requests <= 0 {
        return QkvDecode::Declined(QkvDecline::NoRequests);
    }
    let use_rope_table = !rope_table.is_null();

    // The eight arguments shared by both forms, in the order both kernels
    // declare them. The two forms differ only in what follows `win`.
    let head: [ArgValue; 15] = [
        ArgValue::Ptr(packed.cast_mut()),
        ArgValue::Ptr(q_out),
        ArgValue::Ptr(k_pages),
        ArgValue::Ptr(v_pages),
        ArgValue::Ptr(q_weight.cast_mut()),
        ArgValue::Ptr(k_weight.cast_mut()),
        ArgValue::Ptr(positions.cast_mut().cast()),
        ArgValue::Ptr(rope_table.cast_mut().cast()),
        ArgValue::Ptr(kv_page_indices.cast_mut().cast()),
        ArgValue::Ptr(kv_page_indptr.cast_mut().cast()),
        ArgValue::Ptr(kv_last_page_lens.cast_mut().cast()),
        ArgValue::Ptr(w_page.cast_mut().cast()),
        ArgValue::Ptr(w_off.cast_mut().cast()),
        ArgValue::Ptr(row_valid.cast_mut().cast()),
        ArgValue::Ptr(win.cast_mut().cast()),
    ];

    if let Some(symbol) = warp_symbol(head_dim, use_rope_table) {
        // `:52-53` — one WARP per `(request, head)` unit, `WARPS_PER_BLOCK`
        // units per block. `head_dim` does NOT appear in the argument list:
        // it is the template argument the symbol names.
        let total_units = num_requests.saturating_mul(num_q_heads + num_kv_heads);
        #[allow(clippy::cast_sign_loss)]
        let units = total_units.max(0) as u32;
        let grid_x = (units + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        let launch = Launch {
            grid: [grid_x.max(1), 1, 1],
            block: [WARP_BLOCK, 1, 1],
            smem: 0,
        };
        let values = [
            head[0], head[1], head[2], head[3], head[4], head[5], head[6],
            head[7], head[8], head[9], head[10], head[11], head[12], head[13],
            head[14],
            ArgValue::I32(num_requests),
            ArgValue::I32(num_q_heads),
            ArgValue::I32(num_kv_heads),
            ArgValue::I32(page_size),
            ArgValue::Bool(hnd_layout),
            ArgValue::F32(theta),
            ArgValue::F32(eps),
        ];
        super::hand::fire(symbol, launch, &values, stream);
        return QkvDecode::Warp { head_dim, rope_table: use_rope_table };
    }

    // `:105-106` — the fallthrough. One block per `(request, head)`, and the
    // kernel is TOLD `head_dim` because there is no template argument
    // carrying it.
    #[allow(clippy::cast_sign_loss)]
    let launch = Launch {
        grid: [num_requests as u32, (num_q_heads + num_kv_heads).max(1) as u32, 1],
        block: [BLOCK, 1, 1],
        smem: 0,
    };
    let values = [
        head[0], head[1], head[2], head[3], head[4], head[5], head[6],
        head[7], head[8], head[9], head[10], head[11], head[12], head[13],
        head[14],
        ArgValue::I32(num_q_heads),
        ArgValue::I32(num_kv_heads),
        ArgValue::I32(head_dim),
        ArgValue::I32(page_size),
        ArgValue::Bool(hnd_layout),
        ArgValue::F32(theta),
        ArgValue::F32(eps),
    ];
    super::hand::fire(block_symbol(use_rope_table), launch, &values, stream);
    QkvDecode::Block { rope_table: use_rope_table }
}

/// `attn/qkv_fused.cu:160` — `qkv_decode_qk_norm_rope_write_kv_bf16`.
///
/// The public launcher: [`qkv_decode_fused_dispatch`] with `win = nullptr`.
///
/// ```text
/// :183   qkv_decode_fused_dispatch(
/// :189       w_page, w_off, row_valid, /*win=*/nullptr,
/// ```
///
/// # Panics
///
/// [`qkv_decode_fused_dispatch`]'s.
///
/// # Safety
///
/// [`qkv_decode_fused_dispatch`]'s.
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
pub unsafe fn qkv_decode_qk_norm_rope_write_kv_bf16(
    packed: *const c_void,
    q_out: *mut c_void,
    k_pages: *mut c_void,
    v_pages: *mut c_void,
    q_weight: *const c_void,
    k_weight: *const c_void,
    positions: *const i32,
    rope_table: *const f32,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    w_page: *const u32,
    w_off: *const u32,
    row_valid: *const u8,
    num_requests: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    page_size: i32,
    hnd_layout: bool,
    theta: f32,
    eps: f32,
    stream: *mut c_void,
) -> QkvDecode {
    // SAFETY: the caller's contract, forwarded; `win` is null on this path.
    unsafe {
        qkv_decode_fused_dispatch(
            packed,
            q_out,
            k_pages,
            v_pages,
            q_weight,
            k_weight,
            positions,
            rope_table,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_lens,
            w_page,
            w_off,
            row_valid,
            core::ptr::null(),
            num_requests,
            num_q_heads,
            num_kv_heads,
            head_dim,
            page_size,
            hnd_layout,
            theta,
            eps,
            stream,
        )
    }
}
