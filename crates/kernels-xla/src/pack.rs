//! How a quantized weight plane is stored on the device: the stored forms
//! the kernels read. The engine lands each checkpoint plane in its form
//! (engine-xla's `pack`); a kernel sees the form in its operand's `dtype`.
//!
//! # The storage contract
//!
//! - **Bank codes** (`U4g32`, `U4g64`, `U2g32/64/128`, `U8g64`, `Mxfp4`) land
//!   *one code per element*, row-major exactly as the checkpoint orders them
//!   (code `k` of row `r` at `[r, k]`), in a native narrow type the TPU
//!   converts for free inside a dot: `ui4` for 2- and 4-bit affine codes,
//!   `ui8` for 8-bit, `f4E2M1FN` for mxfp4. The plane keeps the checkpoint's
//!   packed dtype and declared width ([`codes_per_row`] turns it into the
//!   stored width). Unpacking the codes in-graph instead costs several times
//!   the matmul it feeds on TPU (the codes of a byte are a minor-axis
//!   interleave), which is what this contract removes.
//! - **Pre-scaled mxfp4.** An mxfp4 bank may instead land as its weights,
//!   `e2m1(c) · 2^(e − 127)`, one `f8E5M2` each: exact
//!   whenever every block's scale keeps its values inside e5m2's range, which
//!   the transform checks (it answers `None` otherwise and the bank lands as
//!   codes). The plane then says [`Dtype::E5m2`] with one element per
//!   weight, and the kernels read no scales: the weight converts to bf16
//!   exactly and feeds the dot directly, at twice the codes' bytes. The
//!   engine chooses it when the weights still fit.
//! - Scales and zero points land as declared (bf16; e8m0 bytes). K-quant
//!   blocks and nvfp4 stay the raw bytes of their rows.

use dtype::Dtype;

use crate::hlo::Elem;

/// The element a bank's codes land as, one per code.
#[must_use]
pub const fn code_elem(dtype: Dtype) -> Option<Elem> {
    match dtype {
        Dtype::U4g32 | Dtype::U4g64 | Dtype::U2g32 | Dtype::U2g64 | Dtype::U2g128 => Some(Elem::U4),
        Dtype::U8g64 => Some(Elem::U8),
        Dtype::Mxfp4 => Some(Elem::F4E2m1fn),
        _ => None,
    }
}

/// Bits one code occupies in the checkpoint's packed rows.
#[must_use]
pub const fn code_bits(dtype: Dtype) -> Option<u32> {
    match dtype {
        Dtype::U4g32 | Dtype::U4g64 | Dtype::Mxfp4 => Some(4),
        Dtype::U2g32 | Dtype::U2g64 | Dtype::U2g128 => Some(2),
        Dtype::U8g64 => Some(8),
        _ => None,
    }
}

/// Codes in a row a param of `dtype` declares `width` wide (an mxfp4 param
/// declares its row in bytes; the affine formats in codes).
#[must_use]
pub const fn codes_per_row(dtype: Dtype, width: u64) -> Option<u64> {
    match dtype {
        Dtype::Mxfp4 => Some(width * 2),
        Dtype::U4g32
        | Dtype::U4g64
        | Dtype::U2g32
        | Dtype::U2g64
        | Dtype::U2g128
        | Dtype::U8g64 => Some(width),
        _ => None,
    }
}
