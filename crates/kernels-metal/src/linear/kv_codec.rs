//! C2a — the low-bit KV codec's pack/unpack kernels, in isolation.
//!
//! This is the codec MATH only: it packs an f32/bf16 activation into the LOCKED
//! v1 KV format (4-bit symmetric absmax, block = 256, one fp16 scale per block,
//! no bias) and unpacks it back. It touches no cache, no allocation and no
//! forward path — it is a standalone op driven directly from a test, the same
//! way `pointwise::hadamard` is.
//!
//! The packed layout is defined by the kernels, not by a `Dtype`: each 256-block
//! is 128 bytes of nibbles (element `2*b` in the low nibble of byte `b`,
//! `2*b+1` in the high nibble; offset-binary `q + 8`) followed by one fp16 scale
//! stored little-endian, 130 bytes total.

use crate::encode::{Arg, Ctx, Fire, dtype_dispatch, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "linear/kv_codec.metal";

/// Elements per block. One whole head at head_dim 256; a fixed 256-vector
/// otherwise. The 130-byte packed layout is defined for this width.
pub const BLOCK: u32 = 256;

/// Packed bytes per block: 128 nibble bytes + one fp16 scale.
pub const PACKED_BYTES_PER_BLOCK: u64 = 130;

/// The number of whole 256-blocks in a `rows x width` rectangle, or an error if
/// the element count is not a multiple of `BLOCK` (the format has no partial
/// block — a ragged tail is unsupported and rejected here, not silently padded).
fn blocks_of(op: &'static str, rows: u32, width: u32) -> Result<u32, Error> {
    let total = u64::from(rows) * u64::from(width);
    if total == 0 {
        return Err(refuse(op, "the tensor is empty"));
    }
    if !total.is_multiple_of(u64::from(BLOCK)) {
        return Err(refuse(
            op,
            format!(
                "the tensor holds {total} elements, which is not a multiple of {BLOCK}; the v1 \
                 KV codec packs whole {BLOCK}-blocks and has no partial-block form"
            ),
        ));
    }
    u32::try_from(total / u64::from(BLOCK))
        .map_err(|_| refuse(op, format!("{total} elements is too many blocks to launch")))
}

/// Assert the packed buffer holds at least `blocks * 130` bytes.
fn fits_packed(op: &'static str, packed: Tensor, blocks: u32) -> Result<(), Error> {
    let have = u64::from(packed.rows) * u64::from(packed.width) * packed.dtype.bytes_ceil();
    let need = u64::from(blocks) * PACKED_BYTES_PER_BLOCK;
    if have < need {
        return Err(refuse(
            op,
            format!(
                "the packed buffer holds {have} bytes and {blocks} block(s) need {need} \
                 ({PACKED_BYTES_PER_BLOCK} bytes each)"
            ),
        ));
    }
    Ok(())
}

/// Pack `x` (f32 or bf16, row-major) into the v1 KV format in `packed`. `packed`
/// is a raw byte buffer of at least `blocks * 130` bytes; its declared dtype is
/// immaterial (the layout is the codec's, not a `Dtype`'s). One GPU thread packs
/// one 256-block.
pub fn pack(ctx: &Ctx<'_>, x: Tensor, packed: Tensor) -> Result<(), Error> {
    const OP: &str = "kv_codec.pack";
    let entry = dtype_dispatch!(OP, x.dtype, {
        F32 => "kv_pack_sym4_f32",
        Bf16 => "kv_pack_sym4_bf16",
    });
    let blocks = blocks_of(OP, x.rows, x.width)?;
    fits_packed(OP, packed, blocks)?;
    ctx.fire(
        Fire::at(FILE, entry).apply([blocks, 1, 1]),
        &[x.arg(), packed.arg_mut(), blocks.arg()],
    )
}

/// Unpack the v1 KV format in `packed` into `out` (f32 or bf16, row-major). `out`
/// states the element count (`rows * width`, a multiple of 256); `packed` must
/// hold at least `blocks * 130` bytes. One GPU thread unpacks one 256-block.
pub fn unpack(ctx: &Ctx<'_>, packed: Tensor, out: Tensor) -> Result<(), Error> {
    const OP: &str = "kv_codec.unpack";
    let entry = dtype_dispatch!(OP, out.dtype, {
        F32 => "kv_unpack_sym4_f32",
        Bf16 => "kv_unpack_sym4_bf16",
    });
    let blocks = blocks_of(OP, out.rows, out.width)?;
    fits_packed(OP, packed, blocks)?;
    ctx.fire(
        Fire::at(FILE, entry).apply([blocks, 1, 1]),
        &[packed.arg(), out.arg_mut(), blocks.arg()],
    )
}
