//! `layout/embed.cu`'s one launcher, in Rust — and with it the whole of
//! `kernels-cuda/csrc/src/layout/`.
//!
//! # The `VEC` choice is the file
//!
//! `layout/embed.cuh:18-25` is the argument that kept this launcher in C++
//! through four passes:
//!
//! > `embed` is NOT a row and is not templated. Its `VEC` parameter is chosen
//! > on the HOST from a run-time test the device cannot make — `hidden % 8 ==
//! > 0` and both `weight` and `y` 16-byte aligned — and the element count it
//! > launches over is `num_tokens * (vec ? hidden/8 : hidden)`, an extent
//! > that depends on the answer. No `Source` in `kernels/src/lib.rs` produces
//! > "is this pointer 16-byte aligned", and `new-horizon.md` §10.5 refuses an
//! > invented one.
//!
//! Every clause of that is still true, and none of it is an argument for
//! C++. It is an argument that the choice is a HOST PROGRAM, which is what
//! this file is. `fire::hand::aligned16` is the predicate the C++ spelled as
//! `(uintptr_t)p % 16 == 0`; it exists here already, for the same reason, in
//! `fire/kv_paged.rs`' neighbourhood.
//!
//! # The measurement, carried rather than consumed
//!
//! `layout/embed.cuh:27-31`:
//!
//! > The vectorised form is not an optimisation to drop: at decode the
//! > token-per-block form issued 24 dependent 2-byte loads from 8 blocks and
//! > ran at 8 GB/s — the row it reads is a random offset into the largest
//! > tensor in the model, so the access is a cold TLB miss whose latency only
//! > a wide grid hides.
//!
//! That is why [`embed_bf16`] takes the scalar arm rather than always taking
//! it, and why the arm is not "the slow path we could delete".
//!
//! # How the row closed
//!
//! `layout::embed_bf16` was a `table::driver_internal` row and is now a
//! `table::layout` row, which is what made `execution::RUST_SERVED` available
//! to it — `driver_internal` is deliberately outside `table::TABLES`, so
//! `table::sig` never resolved it and the takeover test would have refused.
//! The move is justified on its own terms: `model-compiler/src/lower.rs:1462`
//! lowers `Embed { .. }` to `Semantic::Kernels(&["layout::embed_bf16"])`, so
//! a statement names the symbol and `driver_internal`'s stated membership
//! rule ("launchers the driver fires with no DSL statement") had stopped
//! describing it.
//!
//! Classified `Execution::Walk` with `Control::Switch { on: "the 16-byte
//! alignment test" }` — two instantiations, one chosen per call.

use std::ffi::c_void;

use kernels_cuda_new::runtime::{ArgValue, Launch};

/// `layout::embed_bf16` — the table symbol this file serves.
pub const EMBED_SYMBOL: &str = "layout::embed_bf16";

/// `layout::embed#vec` — `embed<true>`.
const EMBED_VEC: &str = "layout::embed#vec";

/// `layout::embed#scalar` — `embed<false>`.
const EMBED_SCALAR: &str = "layout::embed#scalar";

/// `embed.cu:31` — `constexpr int BLOCK = 256;`.
const BLOCK: u32 = 256;

/// `embed.cu:35` — the vector width, in `bf16` elements.
///
/// Eight `bf16` is sixteen bytes, which is why the alignment test is 16 and
/// the divisibility test is 8. One constant, two tests.
const VEC_WIDTH: i32 = 8;

/// Whether the embedding gather ran, and which arm it took.
///
/// `#[must_use]` for `fire/gemv.rs`' reason.
#[must_use]
pub enum Embed {
    /// Launched `embed<true>` — the wide arm.
    Vectorised,
    /// Launched `embed<false>` — one lane per element.
    Scalar,
    /// Nothing was launched, and which extent was empty.
    Declined(EmbedDecline),
}

/// The two ways the gather declines. Both are `embed.cu:32`'s one `return`,
/// split so the caller learns which extent was empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbedDecline {
    /// `num_tokens <= 0` — an empty batch.
    NoTokens,
    /// `hidden <= 0` — a zero-width model, which is a bug upstream rather
    /// than a shape; the C++ returned rather than launching an empty grid and
    /// so does this.
    NoWidth,
}

/// `embed.cu:33-35` — the host test that picks `VEC`.
///
/// ```text
/// :33   const bool vec = (hidden % 8) == 0 &&
/// :34                    (reinterpret_cast<std::uintptr_t>(weight) % 16) == 0 &&
/// :35                    (reinterpret_cast<std::uintptr_t>(y) % 16) == 0;
/// ```
///
/// Transcribed with `%` kept as `%`: `hidden % 8` is an `int` remainder and
/// the two pointer tests are `& 15` in every sense that matters, which is
/// what [`super::hand::aligned16`] does.
#[must_use]
pub fn vectorisable(hidden: i32, weight: *const c_void, y: *const c_void) -> bool {
    hidden % VEC_WIDTH == 0
        && super::hand::aligned16(weight.cast())
        && super::hand::aligned16(y.cast())
}

/// `layout/embed.cu:25` — `embed_bf16`, the first launch of every fire.
///
/// Gathers one row of the vocabulary table per token. Out-of-range token ids
/// read row 0 rather than faulting — `embed.cuh:72`,
/// `(tid_raw >= 0 && tid_raw < vocab) ? tid_raw : 0` — which is the kernel's
/// decision and not this file's.
///
/// ```text
/// :36   const int per_row = vec ? hidden / 8 : hidden;
/// :37   const long long total = static_cast<long long>(num_tokens) * per_row;
/// :38   dim3 grid(static_cast<unsigned>((total + BLOCK - 1) / BLOCK));
/// :39   dim3 block(BLOCK);
/// :41   device::embed<true><<<grid, block, 0, stream>>>(...)
/// :47   device::embed<false><<<grid, block, 0, stream>>>(...)
/// ```
///
/// The `long long` is transcribed as `i64` and matters: `num_tokens * hidden`
/// for a 128k-token prefill against a 8192-wide model overflows `i32`, and
/// the C++ widened before multiplying for exactly that reason. The grid is
/// then narrowed to `u32`, which is the cast the C++ spells as
/// `static_cast<unsigned>`.
///
/// # Panics
///
/// If the kernel table and this driver disagree; see [`super::hand::fire`].
///
/// # Safety
///
/// `token_ids`, `weight` and `y` are device addresses the caller keeps live
/// across the launch, and `stream` is the caller's stream.
#[allow(clippy::too_many_arguments)]
pub unsafe fn embed_bf16(
    token_ids: *const i32,
    weight: *const c_void,
    y: *mut c_void,
    num_tokens: i32,
    hidden: i32,
    vocab: i32,
    stream: *mut c_void,
) -> Embed {
    // `embed.cu:32`, split so the caller learns which extent was empty.
    if num_tokens <= 0 {
        return Embed::Declined(EmbedDecline::NoTokens);
    }
    if hidden <= 0 {
        return Embed::Declined(EmbedDecline::NoWidth);
    }
    let vec = vectorisable(hidden, weight, y.cast_const());
    let per_row = if vec { hidden / VEC_WIDTH } else { hidden };
    let total = i64::from(num_tokens) * i64::from(per_row);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let blocks = ((total + i64::from(BLOCK) - 1) / i64::from(BLOCK)) as u32;
    let launch = Launch { grid: [blocks, 1, 1], block: [BLOCK, 1, 1], smem: 0 };
    let values = [
        ArgValue::Ptr(token_ids.cast_mut().cast()),
        ArgValue::Ptr(weight.cast_mut()),
        ArgValue::Ptr(y),
        ArgValue::I32(hidden),
        ArgValue::I32(vocab),
        ArgValue::I32(num_tokens),
        ArgValue::I32(per_row),
    ];
    if vec {
        super::hand::fire(EMBED_VEC, launch, &values, stream);
        Embed::Vectorised
    } else {
        super::hand::fire(EMBED_SCALAR, launch, &values, stream);
        Embed::Scalar
    }
}
