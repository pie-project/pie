use crate::encode::{Arg, Ctx, Fire, dtype_dispatch, nonzero, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "elemwise/hadamard.slang";

/// Threads per workgroup; one workgroup owns one N-block. Matches the
/// `PIE_GROUP_X` the `hadamard.slang` variants are compiled with.
const GROUP: u32 = 256;

/// A blockwise butterfly FWHT over the last dim, in place. The row width must be
/// a multiple of `block`; each contiguous `block`-vector is turned by the
/// normalized Sylvester Hadamard matrix. When `signs` is `Some`, its +/-1 entries
/// multiply the activation on load, before the butterfly (the Randomized
/// Hadamard `H . S`); the sign vector repeats block-wise if wider than `block`.
///
/// A 1:1 analog of `kernels_metal::elemwise::pointwise::hadamard`. The Metal
/// side splits into `fwht_simd` (N <= 256) and `fwht_tg` (N >= 512); here a
/// single groupshared kernel handles 64..1024 with the block size carried as a
/// push constant.
pub fn hadamard(ctx: &Ctx<'_>, x: Tensor, block: u32, signs: Option<Tensor>) -> Result<(), Error> {
    const OP: &str = "elementwise.hadamard";
    let entry = dtype_dispatch!(OP, x.dtype, {
        Bf16 => "hadamard_bf16",
        F32 => "hadamard_f32",
    });
    if !matches!(block, 64 | 128 | 256 | 512 | 1024) {
        return Err(refuse(
            OP,
            format!(
                "block {block} is not a supported Hadamard size; the butterfly turns 64, 128, \
                 256, 512 and 1024-vectors"
            ),
        ));
    }
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
    let total = u64::from(nonzero(OP, "width", x.width)?) * u64::from(nonzero(OP, "rows", x.rows)?);
    let nblocks = u32::try_from(total / u64::from(block))
        .map_err(|_| refuse(OP, format!("{total} elements will not launch")))?;
    let signs_width = signs.map_or(0, |s| s.rows * s.width);
    let signs_arg = match signs {
        Some(s) => s.arg(),
        None => ctx.absent()?,
    };
    ctx.fire(
        Fire::at(FILE, entry)
            .group([GROUP, 1, 1])
            .groups([nblocks, 1, 1]),
        &[x.arg_mut(), signs_arg, block.arg(), signs_width.arg()],
    )
}
