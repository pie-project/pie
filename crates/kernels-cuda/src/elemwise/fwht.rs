use crate::error::Error;

use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, symbol};
use crate::tensor::Tensor;

const FILE: &str = "elemwise/fwht.cuh";

/// The entrypoint instantiation and threads-per-block for a butterfly FWHT of a
/// given block size and C++ element type. Blocks up to 256 ride one warp (32
/// threads); 512 and 1024 ride a 256-thread block.
fn fwht_point(op: &'static str, block: u32, t: &str) -> Result<(String, u32), Error> {
    Ok(match block {
        64 | 128 | 256 => (format!("::pie::elemwise::fwht_warp<{t}, {block}>"), 32),
        512 | 1024 => (
            format!("::pie::elemwise::fwht_block<{t}, {block}, 256>"),
            256,
        ),
        other => {
            return Err(refuse(
                op,
                format!(
                    "block {other} is not a supported Hadamard size; the butterfly turns \
                     64, 128, 256, 512 and 1024-vectors"
                ),
            ));
        }
    })
}

/// A blockwise butterfly FWHT over the last dim, in place. The row width must
/// be a multiple of `block`; each contiguous `block`-vector is turned by the
/// normalized Sylvester Hadamard matrix. When `signs` is `Some`, its +-1 entries
/// multiply the activation on load, before the butterfly (the Randomized
/// Hadamard `H.S`); the sign vector repeats block-wise if wider than `block`.
pub fn hadamard(ctx: &Ctx, x: &mut Tensor, block: u32, signs: Option<Tensor>) -> Result<(), Error> {
    const OP: &str = "elementwise.hadamard";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F32 => "float" });
    let (sym, threads) = fwht_point(OP, block, t)?;
    if !x.width.is_multiple_of(block) {
        return Err(refuse(
            OP,
            format!(
                "the row width {} is not a multiple of {block}; the block Hadamard turns \
                 contiguous {block}-vectors",
                x.width
            ),
        ));
    }
    if let Some(s) = signs {
        if s.dtype != x.dtype {
            return Err(refuse(
                OP,
                format!(
                    "the sign diagonal is {:?} and the activation is {:?}; the signs ride the \
                     activation's element",
                    s.dtype, x.dtype
                ),
            ));
        }
        let signs_width = u64::from(s.rows) * u64::from(s.width);
        if !signs_width.is_multiple_of(u64::from(block)) {
            return Err(refuse(
                OP,
                format!(
                    "the sign diagonal holds {signs_width} entries, which is not a multiple of \
                     {block}; it repeats block-wise"
                ),
            ));
        }
    }
    let total = x.elements();
    let nblocks = u32::try_from(total / u64::from(block))
        .map_err(|_| refuse(OP, format!("{total} elements will not launch")))?;
    nonzero(OP, "the block count", nblocks)?;
    let signs_width = signs.map_or(0, |s| s.rows * s.width);
    let signs_arg = signs.map_or(ArgValue::ABSENT, |s| s.arg());
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&sym)).apply(Launch::per_row(nblocks, threads)),
        &[x.arg(), signs_arg, signs_width.arg()],
    )
}
