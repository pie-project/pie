//! The PTIR substrate's own kernels — the ones the tensor-compiler's emitted
//! MSL cannot produce because they predate a region.
//!
//! Crossed second, after [`crate::sample`], for the same two reasons: one
//! kernel, and one this backend never dispatches, so the crossing cannot
//! change what any model computes.

use kernels::Grid;
use kernels_macros::routine;
use kernels::routine::Refusal;

use crate::routine::{Asks, Bind, Ctx, Fire, In, Out, Tensor, bf16, keys};

/// The shaders this family's routines reach: `(file, entrypoint)`, one pair
/// per instantiated name.
///
/// A row's `axes` GENERATED these names and its `file` column said where they
/// live. Retiring the row moved who NAMES them, not what exists -- the shader
/// is still compiled and still dispatched -- so the pairs are stated here and
/// [`crate::entrypoints`] reads them back. The FILE rides along because Metal
/// compiles from `(path, entry name)` at run time, and `device_kernels.rs`
/// builds every one of them against a real device; a name without its file
/// would leave that sweep nothing to open. See [`crate::RETIRED`].
pub static ENTRYPOINTS: &[(&str, &str)] = &[("ptir/logits_copy.metal", "copy_logits_bf16")];

/// Threads per threadgroup on the vocabulary axis.
///
/// The kernel is a guarded elementwise copy — `if (tid.x >= p.vocab) return;`
/// — so any width is correct and this is the one every other flat elementwise
/// launch on this backend uses (`grid::elementwise_mb`). Stated here because
/// MSL declares no threadgroup and Metal has nothing to reflect it from.
const GROUP_X: u32 = 256;

/// Stage `rows` vocabulary rows, source and destination row per row.
///
/// `ptir/logits_copy.metal`. One dispatch stages every row a fire needs, with
/// `tid.y` selecting which — and that shape is the whole point of the kernel.
/// It used to copy a single row and be submitted as its own command buffer,
/// once per row, so a sixteen-request fire paid sixteen command-buffer round
/// trips per token to move sixteen vocabulary rows: about 3 ms of a 23.5 ms
/// step, scaling linearly with the batch, which is what made the sampler look
/// linear in lanes when the sampler itself is 0.5 ms of GPU.
///
/// `params` is `const device PtirLogitsCopyParams*` — an ARRAY, indexed by
/// `tid.y`, one struct per row. So it is a buffer and not a scalar run, and
/// `rows` has to reach the grid rather than the kernel: nothing in the shader
/// bounds `tid.y`, because the grid is what bounds it.
///
/// # Errors
///
/// [`Refusal::Empty`] when there is no row to stage, or no vocabulary to stage
/// it over. Either would be a dispatch over an empty grid, which runs nothing
/// and reports success — and the caller would then read a destination that
/// still holds the previous token's logits, which is a wrong answer and not a
/// missing one.
#[routine]
pub fn copy_logits_bf16(
    ctx: &Ctx<'_>,
    source: In<Tensor<bf16>>,
    destination: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    let params = ctx.params()?;
    let vocab = source.width.unsigned_abs();
    let rows = ctx.ask::<u32, keys::Rows>()?;
    if rows == 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if vocab == 0 {
        return Err(Refusal::Empty { what: "vocab" });
    }
    ctx.fire(
        Fire::at("ptir/logits_copy.metal", "copy_logits_bf16").apply(Grid::of([vocab, rows, 1], [GROUP_X, 1, 1])),
        &[source.arg(), destination.arg(), params],
    )
}
