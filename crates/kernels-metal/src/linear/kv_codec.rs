//! C2a — the low-bit KV codec's pack/unpack kernels, in isolation.
//!
//! This is the codec MATH only: it packs an f32/bf16 activation into the LOCKED
//! v1 KV format (4-bit symmetric absmax, one fp16 scale per block, no bias) and
//! unpacks it back. It touches no cache, no allocation and no forward path — it
//! is a standalone op driven directly from a test, the same way
//! `pointwise::hadamard` is.
//!
//! The block is a PARAMETER — the model's full-attention `head_dim`, not a baked
//! 256 — because pie serves many models. Each block of `N` elements packs to
//! `N/2` bytes of nibbles (element `2*b` in the low nibble of byte `b`, `2*b+1`
//! in the high nibble; offset-binary `q + 8`) followed by one fp16 scale stored
//! little-endian, `N/2 + 2` bytes total. The kernels are stamped per block size
//! `N ∈ {64, 128, 256}` (mirroring the Hadamard op); the wrapper selects the
//! shader from the block.

use crate::encode::{Arg, Ctx, Fire, refuse};
use crate::error::Error;
use crate::tensor::Tensor;
use dtype::Dtype;

const FILE: &str = "linear/kv_codec.metal";

/// The block sizes stamped in the codec shader (one whole attention head each).
pub const BLOCKS: [u32; 3] = [64, 256, 128];

/// Packed bytes for one block of `block` elements: `block/2` nibble bytes + one
/// fp16 scale (2 bytes). 130 at block 256, 66 at 128, 34 at 64.
#[must_use]
pub fn packed_bytes_per_block(block: u32) -> u64 {
    u64::from(block) / 2 + 2
}

/// The pack entry point for `dtype` at block `block`, or an error if no shader is
/// stamped at that block. `f32`/`bf16` only.
fn pack_entry(op: &'static str, dtype: Dtype, block: u32) -> Result<&'static str, Error> {
    Ok(match (dtype, block) {
        (Dtype::F32, 64) => "kv_pack_sym4_f32_64",
        (Dtype::F32, 128) => "kv_pack_sym4_f32_128",
        (Dtype::F32, 256) => "kv_pack_sym4_f32_256",
        (Dtype::Bf16, 64) => "kv_pack_sym4_bf16_64",
        (Dtype::Bf16, 128) => "kv_pack_sym4_bf16_128",
        (Dtype::Bf16, 256) => "kv_pack_sym4_bf16_256",
        (Dtype::F32 | Dtype::Bf16, other) => {
            return Err(refuse(
                op,
                format!("no KV codec shader is stamped at block {other}; block ∈ {BLOCKS:?}"),
            ));
        }
        (other, _) => return Err(Error::DtypeUnsupported { op, dtype: other }),
    })
}

/// The unpack entry point for `dtype` at block `block`. `f32`/`bf16` only.
fn unpack_entry(op: &'static str, dtype: Dtype, block: u32) -> Result<&'static str, Error> {
    Ok(match (dtype, block) {
        (Dtype::F32, 64) => "kv_unpack_sym4_f32_64",
        (Dtype::F32, 128) => "kv_unpack_sym4_f32_128",
        (Dtype::F32, 256) => "kv_unpack_sym4_f32_256",
        (Dtype::Bf16, 64) => "kv_unpack_sym4_bf16_64",
        (Dtype::Bf16, 128) => "kv_unpack_sym4_bf16_128",
        (Dtype::Bf16, 256) => "kv_unpack_sym4_bf16_256",
        (Dtype::F32 | Dtype::Bf16, other) => {
            return Err(refuse(
                op,
                format!("no KV codec shader is stamped at block {other}; block ∈ {BLOCKS:?}"),
            ));
        }
        (other, _) => return Err(Error::DtypeUnsupported { op, dtype: other }),
    })
}

/// The number of whole `block`-blocks in a `rows x width` rectangle, or an error
/// if the element count is not a multiple of `block` (the format has no partial
/// block — a ragged tail is unsupported and rejected here, not silently padded).
fn blocks_of(op: &'static str, rows: u32, width: u32, block: u32) -> Result<u32, Error> {
    if block == 0 {
        return Err(refuse(op, "the codec block is zero"));
    }
    let total = u64::from(rows) * u64::from(width);
    if total == 0 {
        return Err(refuse(op, "the tensor is empty"));
    }
    if !total.is_multiple_of(u64::from(block)) {
        return Err(refuse(
            op,
            format!(
                "the tensor holds {total} elements, which is not a multiple of the block {block}; \
                 the v1 KV codec packs whole {block}-blocks and has no partial-block form"
            ),
        ));
    }
    u32::try_from(total / u64::from(block))
        .map_err(|_| refuse(op, format!("{total} elements is too many blocks to launch")))
}

/// Assert the packed buffer holds at least `blocks * (block/2 + 2)` bytes.
fn fits_packed(op: &'static str, packed: Tensor, blocks: u32, block: u32) -> Result<(), Error> {
    let have = u64::from(packed.rows) * u64::from(packed.width) * packed.dtype.bytes_ceil();
    let per = packed_bytes_per_block(block);
    let need = u64::from(blocks) * per;
    if have < need {
        return Err(refuse(
            op,
            format!(
                "the packed buffer holds {have} bytes and {blocks} block(s) need {need} \
                 ({per} bytes each)"
            ),
        ));
    }
    Ok(())
}

/// Pack `x` (f32 or bf16, row-major) into the v1 KV format in `packed`, one whole
/// `block`-element head per GPU thread. `packed` is a raw byte buffer of at least
/// `blocks * (block/2 + 2)` bytes; its declared dtype is immaterial (the layout
/// is the codec's, not a `Dtype`'s).
pub fn pack(ctx: &Ctx<'_>, x: Tensor, packed: Tensor, block: u32) -> Result<(), Error> {
    const OP: &str = "kv_codec.pack";
    let entry = pack_entry(OP, x.dtype, block)?;
    let blocks = blocks_of(OP, x.rows, x.width, block)?;
    fits_packed(OP, packed, blocks, block)?;
    ctx.fire(
        Fire::at(FILE, entry).apply([blocks, 1, 1]),
        &[x.arg(), packed.arg_mut(), blocks.arg()],
    )
}

/// Unpack the v1 KV format in `packed` into `out` (f32 or bf16, row-major). `out`
/// states the element count (`rows * width`, a multiple of `block`); `packed`
/// must hold at least `blocks * (block/2 + 2)` bytes. One thread per block.
pub fn unpack(ctx: &Ctx<'_>, packed: Tensor, out: Tensor, block: u32) -> Result<(), Error> {
    const OP: &str = "kv_codec.unpack";
    let entry = unpack_entry(OP, out.dtype, block)?;
    let blocks = blocks_of(OP, out.rows, out.width, block)?;
    fits_packed(OP, packed, blocks, block)?;
    ctx.fire(
        Fire::at(FILE, entry).apply([blocks, 1, 1]),
        &[packed.arg(), out.arg_mut(), blocks.arg()],
    )
}
