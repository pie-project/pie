//! `attn/split_packed.cu`'s one surviving launcher, in Rust — and with it the
//! file.
//!
//! `split_qkv_bf16`'s launcher went in an earlier pass: its row is in
//! `device::JIT_DISPATCHED` and `LaunchRule::SplitPacked` states the grid the
//! deleted host code computed. `split_qkv_bf16_devwin` stayed, and
//! `families/attn.rs`' `SPLIT_PACKED_SIGS` wrote down exactly why:
//!
//! > **`split_qkv_devwin` has no row, and its geometry is not why.** The rule
//! > computes the same `dim3(xblocks, rows)` for it — the launcher is cited
//! > beside this one — but it computes it from the wrong numbers, twice over:
//! > `grid.y` is `n_max`, which the ahead-of-time row sources from
//! > `Ctx("rows_total")` — the FIRE's lane count — while a rule reads
//! > `Dims::rows`, the statement's rectangle. […] Its buffers are BASE
//! > pointers by contract — the window lives in device memory so a captured
//! > graph can replay across row splits without re-recording — and the JIT
//! > binder resolves `In`/`Out` THROUGH the statement's window.
//!
//! Both objections are to a `LaunchRule` and to the JIT BINDER, and neither is
//! an objection to Rust. A driver-owned [`Launch`] states `grid.y = n_max`
//! because this function is HANDED `n_max`, and `super::hand::fire` binds the
//! pointers it is given without re-windowing them. The row is takeable
//! precisely because the host is no longer a rule.
//!
//! # The double-window, restated as an invariant of this function
//!
//! The pointers this launcher passes are BASE pointers. The kernel compares
//! an ABSOLUTE `blockIdx.y` against the device window at `win_d` and indexes
//! from the base; a caller that has already offset `packed`/`q_out`/`k_out`/
//! `v_out` by the statement's row window will write the peel's tail twice
//! offset. That is the failure the `.cuh` names and this file cannot detect —
//! there is no runtime test for "this pointer has already been windowed" —
//! so it is stated here and enforced by the caller.
//!
//! # How the row closed
//!
//! `attn::split_qkv_bf16_devwin` was a `table::driver_internal` row and is now
//! a `table::attn` row, for `layout::embed_bf16`'s reason and by the same
//! rule: `driver_internal` holds *"launchers the driver fires with no DSL
//! statement"*, and `model-compiler/src/lower.rs:1503` names this symbol from
//! a statement. `execution::RUST_SERVED` is gated on `table::sig` resolving,
//! which `driver_internal` never does.

use std::ffi::c_void;

use kernels_cuda_new::runtime::{ArgValue, Launch};

/// `attn::split_qkv_bf16_devwin` — the table symbol this file serves.
pub const SPLIT_DEVWIN_SYMBOL: &str = "attn::split_qkv_bf16_devwin";

/// `attn::split_qkv_devwin` — the device row it fires.
const SPLIT_DEVWIN_DEVICE: &str = "attn::split_qkv_devwin";

/// `split_packed.cu:30` — `constexpr int BLOCK = 256;`.
const BLOCK: u32 = 256;

/// Whether the device-window split ran.
///
/// `#[must_use]` for `fire/gemv.rs`' reason.
#[must_use]
pub enum SplitPacked {
    /// The kernel was launched on the caller's stream.
    Launched,
    /// Nothing was launched, and why.
    Declined(SplitDecline),
}

/// The one way this launcher declines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitDecline {
    /// `split_packed.cu:42` — `n_max <= 0`.
    ///
    /// `n_max` is `grid.y`, so an empty lane count is an empty grid.
    NoRows,
}

/// `attn/split_packed.cu:35` — `split_qkv_bf16_devwin`.
///
/// Splits a packed `[N, q_dim + 2 * kv_dim]` activation into Q, K and V,
/// visiting only the rows a DEVICE-RESIDENT window admits — which is what
/// lets a captured graph replay across row splits without re-recording.
///
/// ```text
/// :43   const int max_dim = q_dim > kv_dim ? q_dim : kv_dim;
/// :44   const int xblocks = (max_dim + BLOCK - 1) / BLOCK;
/// :45   dim3 grid(xblocks, n_max);
/// :46   device::split_qkv_devwin<bf16><<<grid, BLOCK, 0, stream>>>(
/// ```
///
/// `grid.x` covers the WIDER of the two outputs, not the packed width:
/// `split_packed.cuh` licenses the difference — *"every loop below strides by
/// `blockDim.x * gridDim.x` and bounds itself on its own output width, so
/// extra blocks contribute nothing but a shorter loop"* — and the direction
/// matters only one way. A grid narrower than an output leaves the tail of
/// every row unwritten, so `max` and not `min`, transcribed rather than
/// re-derived.
///
/// # Panics
///
/// If the kernel table and this driver disagree; see [`super::hand::fire`].
///
/// # Safety
///
/// `packed`, the three outputs and `win_d` are device addresses the caller
/// keeps live across the launch, and `stream` is the caller's stream. The
/// four buffer pointers must be BASE pointers — see the module header's
/// double-window paragraph, which is a real precondition and not a note.
#[allow(clippy::too_many_arguments)]
pub unsafe fn split_qkv_bf16_devwin(
    packed: *const c_void,
    q_out: *mut c_void,
    k_out: *mut c_void,
    v_out: *mut c_void,
    win_d: *const u32,
    n_max: i32,
    q_dim: i32,
    kv_dim: i32,
    stream: *mut c_void,
) -> SplitPacked {
    // `split_packed.cu:42`.
    if n_max <= 0 {
        return SplitPacked::Declined(SplitDecline::NoRows);
    }
    let max_dim = if q_dim > kv_dim { q_dim } else { kv_dim };
    #[allow(clippy::cast_sign_loss)]
    let xblocks = ((max_dim.max(0) as u32) + BLOCK - 1) / BLOCK;
    #[allow(clippy::cast_sign_loss)]
    let launch = Launch {
        grid: [xblocks.max(1), n_max as u32, 1],
        block: [BLOCK, 1, 1],
        smem: 0,
    };
    let values = [
        ArgValue::Ptr(packed.cast_mut()),
        ArgValue::Ptr(q_out),
        ArgValue::Ptr(k_out),
        ArgValue::Ptr(v_out),
        ArgValue::Ptr(win_d.cast_mut().cast()),
        ArgValue::I32(q_dim),
        ArgValue::I32(kv_dim),
    ];
    super::hand::fire(SPLIT_DEVWIN_DEVICE, launch, &values, stream);
    SplitPacked::Launched
}
