//! The loader's two casts: the host half of `quant/dtype_cast.cu`, in Rust.
//!
//! Ports `kernels::quant::cast_fp32_to_bf16` and
//! `kernels::quant::scale_rows_bf16` — the file's whole surviving surface, two
//! launchers over two `<<<>>>` and no device text at all. The launcher, its
//! header (`quant/dtype_cast.hpp`) and its `CMakeLists.txt` entry are
//! DELETED; every `__global__` it fired is `kernels-cuda-new`'s
//! `quant/dtype_cast` unit, which NVRTC compiles from
//! `kernels-cuda-new/csrc/src/quant/dtype_cast.cuh`.
//!
//! # Why this file exists at all, when both rows are already routed
//!
//! Both symbols are in `kernels_cuda_new::device::JIT_DISPATCHED`, so
//! `emit_c_shim` drops their `pie_k_*` entries and every call the model
//! compiler makes reaches the JIT through the generated dispatch. That would
//! have been the whole change, except that `model-loader`'s two call sites do
//! not go through the compiler: `fire::lora` called
//! `ffi::pie_k_quant_cast_fp32_to_bf16` and `ffi::pie_k_quant_scale_rows_bf16`
//! **by hand**, and a hand-written arm is invisible to every check that reads
//! generated text. Routing the rows without moving those two calls leaves a
//! declaration with no definition — `new-horizon.md` §22.1's failure, where
//! `rust-lld`'s `--error-limit=20` reports a fifth of the undefined symbols
//! and each fix reveals twenty more.
//!
//! So this module is the loader-side seam: the two entry points `model-loader`
//! names, spelled once, firing the same two rows through the JIT.
//!
//! # Why these are rule-driven and `super::gemv` is not
//!
//! Nothing here builds a [`Launch`] by hand, because nothing needs to. Both
//! geometries are already stated as [`kernels::LaunchRule`]s on the device rows
//! in `kernels_cuda_new::families::quant`, and the rules reproduce the
//! launchers:
//!
//! * `quant::cast_fp32_to_bf16` states [`LaunchRule::Elementwise`], which is
//!   `grid = ceil(n / 256), block = 256, smem = 0`. `dtype_cast.cu:51-54`:
//!
//!   ```text
//!   const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
//!   device::cast_f32_to<device::bf16><<<blocks, BLOCK, 0, stream>>>(
//!   ```
//!
//!   with `constexpr int BLOCK = 256` at `:20`. Identical, number for number.
//!
//! * `quant::scale_rows_bf16` states [`LaunchRule::RouteRows`], which is
//!   `grid = [rows, 1, 1], block = ceil_warp(width)` capped at 1024.
//!   `dtype_cast.cu:69-72`:
//!
//!   ```text
//!   device::scale_rows<device::bf16><<<rows, BLOCK, 0, stream>>>(
//!   ```
//!
//!   Same grid; the block widths differ ON PURPOSE and the `.cu` said so
//!   beside the `<<<>>>`: *"The block width is the launcher's to pick because
//!   the kernel reads `blockDim.x`; 256 here, `ceil_warp(width)` under the
//!   rule, same answer."* The kernel's `for (c = threadIdx.x; c < width; c +=
//!   blockDim.x)` makes both exact. `families::quant`'s row carries the same
//!   paragraph, which is where it survives now that the `.cu` does not.
//!
//! [`LaunchRule::Elementwise`]: kernels::LaunchRule::Elementwise
//! [`LaunchRule::RouteRows`]: kernels::LaunchRule::RouteRows
//! [`Launch`]: kernels_cuda_new::runtime::Launch
//!
//! # The operand lists are the DEVICE rows', not the table's
//!
//! `scale_rows_bf16` takes `(buf, l, rows, width, stream)` in
//! `quant/dtype_cast.hpp` and `(buf_bf16, l_bf16, width)` on the row.
//! `rows` is gone because it IS the grid the rule computes and the
//! `__global__` never took it; the stream is gone because a stream is
//! `cuLaunchKernel`'s sixth parameter and not a member of the `void**`. The
//! two functions below keep the C++ parameter lists so their call sites did
//! not have to change, and drop the two operands at the fire.
//!
//! # The empty-extent guards are the launchers', kept
//!
//! `dtype_cast.cu:50` is `if (n == 0) return;` and `:65` is
//! `if (rows == 0 || width == 0) return;`. Both are reproduced here rather
//! than left to the rule, because [`bind::launch::eval`] answers
//! `Ungeometric::Empty` for a collapsed rectangle and `bind::jit::fire`
//! reports that as a refusal — which would turn a launcher that did nothing
//! quietly into a fire that complains. A zero-element cast is a real thing a
//! loader asks for (an adapter with no rows for this site), and it was never
//! an error.
//!
//! [`bind::launch::eval`]: crate::bind::launch
//!
//! # A drift is a panic, never a silent short launch
//!
//! `n` is a `usize` at the C++ boundary and a `u32` in
//! [`kernels_cuda_new::Dims`]. A cast that truncated would launch a grid over
//! the low 32 bits and leave the rest of the buffer holding whatever was
//! there — a wrong weight, silently, once per load. So the conversion is
//! checked and a failure panics with the count named.

use std::ffi::c_void;

use kernels_cuda_new::Dims;

use crate::bind::device::ArgValue;

/// `dst[i] = (bf16)src[i]` for `n` fp32 elements — `quant::cast_fp32_to_bf16`.
///
/// The signature is `quant/dtype_cast.hpp:18-22`'s, minus nothing: the two
/// pointers, the element count and the stream, in that order.
///
/// # Panics
///
/// If `n` does not fit in a `u32`. See the module docs.
///
/// # Safety
///
/// `src_fp32` must address `n` live fp32 elements and `dst_bf16` `n` writable
/// bf16 elements, and `stream` must be live across the launch — the same
/// obligations the caller met when this was a `pie_k_*` call handing the
/// stream to a `<<<>>>`.
pub unsafe fn cast_fp32_to_bf16(
    src_fp32: *const c_void,
    dst_bf16: *mut c_void,
    n: usize,
    stream: *mut c_void,
) {
    // `dtype_cast.cu:50`.
    if n == 0 {
        return;
    }
    let Ok(elems) = u32::try_from(n) else {
        panic!(
            "quant::cast_fp32_to_bf16: {n} elements does not fit a 32-bit launch extent; \
             a truncating cast would launch over the low 32 bits and leave the rest of \
             the destination unwritten"
        );
    };
    // `Elementwise` multiplies `rows * width`, so one row of `n` says `n`.
    let dims = Dims {
        rows: 1,
        width: elems,
        in_width: elems,
    };
    // The device row's operands, in the row's order —
    // `families::quant::DTYPE_CAST_SIGS[0]`: `src_fp32: F32s`, `dst_bf16:
    // BufMut`, `n: Usize`. `Args::bind` checks them against the signature, so
    // a drift between this list and that row is a refusal and not a shifted
    // argument.
    let values = [
        ArgValue::Ptr(src_fp32.cast_mut()),
        ArgValue::Ptr(dst_bf16),
        ArgValue::Usize(n),
    ];
    // SAFETY: the caller's obligation, above.
    unsafe {
        crate::bind::jit::fire("quant::cast_fp32_to_bf16", dims, &values, stream);
    }
}

/// `buf[r, c] *= l[c]` over a `rows x width` bf16 buffer, in place —
/// `quant::scale_rows_bf16`.
///
/// The signature is `quant/dtype_cast.hpp:30-35`'s. `rows` is still a
/// parameter here and is not an operand of the fire: it is the grid the
/// [`LaunchRule::RouteRows`] computes, and the `__global__` never took it.
///
/// [`LaunchRule::RouteRows`]: kernels::LaunchRule::RouteRows
///
/// # Panics
///
/// If `rows` or `width` is negative. The C++ took `int` and passed `rows`
/// straight into `<<<rows, ...>>>`, where a negative converted to an enormous
/// unsigned grid and the launch failed at the driver; saying it here names the
/// value instead.
///
/// # Safety
///
/// `buf_bf16` must address `rows * width` writable bf16 elements, `l_bf16`
/// `width` readable ones, and `stream` must be live across the launch.
pub unsafe fn scale_rows_bf16(
    buf_bf16: *mut c_void,
    l_bf16: *const c_void,
    rows: i32,
    width: i32,
    stream: *mut c_void,
) {
    // `dtype_cast.cu:65`. The C++ tested `== 0`; this tests `<= 0` for the
    // negative half, which the C++ did not survive either — see the panic
    // below for the half that is a bug rather than an empty extent.
    if rows == 0 || width == 0 {
        return;
    }
    assert!(
        rows > 0 && width > 0,
        "quant::scale_rows_bf16: {rows} x {width} is not an extent"
    );
    let dims = Dims {
        rows: rows.unsigned_abs(),
        width: width.unsigned_abs(),
        in_width: width.unsigned_abs(),
    };
    // `families::quant::DTYPE_CAST_SIGS`' `scale_rows_bf16` row:
    // `buf_bf16: BufMut` (declared `in_place`), `l_bf16: Buf`, `width: I32`.
    let values = [
        ArgValue::Ptr(buf_bf16),
        ArgValue::Ptr(l_bf16.cast_mut()),
        ArgValue::I32(width),
    ];
    // SAFETY: the caller's obligation, above.
    unsafe {
        crate::bind::jit::fire("quant::scale_rows_bf16", dims, &values, stream);
    }
}
