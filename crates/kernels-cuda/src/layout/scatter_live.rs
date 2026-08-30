//! `ScatterLive`: the embed merge for a rectangle whose tail has no
//! destination (`.wiki/alto/multimodal.md` §8.6).
//!
//! **THIS FILE IS `src/layout/scatter_live.rs` AND ITS MODULE PATH IS
//! `kernels_cuda::layout_scatter_live`**, behind the door `attn_dense` opened
//! and [`crate::layout_fold`] uses next door. It is a separate entry rather
//! than a guard inside `layout::scatter_rows` for two reasons and only one of
//! them is the conflict map: `src/layout.rs` and `kernels/layout/layout.cuh`
//! are closed to this wave, AND the existing op's contract — every route names
//! a row, checked host-side before the launch — is one its consumers rely on
//! and should not be widened underneath them. Two ops, two contracts, one
//! body apart.
//!
//! # What the sentinel is for
//!
//! A compacting fold ([`crate::layout_fold::pool_rows`],
//! [`crate::layout_fold::merge_rows`]) answers `rows / side²` rows and leaves
//! the rest of the patch rectangle as whatever the arena held.
//! `RuntimeInput::PatchRoutes` is `[Dim::Patches]` — one destination per row
//! of the FULL rectangle — so those tail rows have route entries, and before
//! this op there was no legal value to put in them: the shell refuses
//! `route < 0` by name and `layout.scatter_rows` would take a negative index
//! as a write below the base of the token rectangle.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "layout/scatter_live.cuh";

const WARP: u32 = 32;

const MAX_BLOCK: u32 = 1024;

/// **THE EMBED MERGE, WITH A DROP SENTINEL**: row `i` of `src` lands at token
/// row `routes[i]` of `y`, and a NEGATIVE `routes[i]` places it nowhere.
///
/// `-1` is the value a submission writes; any negative is dropped, because a
/// kernel that distinguished `-1` from `-2` would be inventing a second
/// sentinel nobody declared.
///
/// The bound at the other end is still not this kernel's to check: an entry
/// past `y.rows` is an out-of-bounds device write that no arena faults on, and
/// the fire path validates the vector against the fire's token row count
/// before the launch (`Fault::PatchRoute`) exactly as it does for the
/// unguarded twin. What this op changes is the LOWER end of that check and
/// nothing else.
///
/// # Errors
///
/// [`Error::DtypeUnsupported`] for anything but bf16, f16 and f32; a refusal
/// for a route vector that is not `i32`, one whose length is not `src.rows`,
/// a width or element mismatch between the two rectangles, and an empty
/// source.
pub fn scatter_live_rows(
    ctx: &Ctx,
    src: Tensor,
    routes: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.scatter_live_rows";
    let unit = dtype_dispatch!(OP, src.dtype, {
        Bf16 => "::pie::bf16",
        F16 => "::pie::f16",
        F32 => "float"
    });
    if routes.dtype != Dtype::I32 {
        return Err(refuse(
            OP,
            format!(
                "the destination rows this merge is handed are {:?}, and it reads an i32 row map",
                routes.dtype
            ),
        ));
    }
    if routes.rows != src.rows {
        return Err(refuse(
            OP,
            format!(
                "{} rows to place and {} destinations named",
                src.rows, routes.rows
            ),
        ));
    }
    if y.dtype != src.dtype || y.width != src.width {
        return Err(refuse(
            OP,
            format!(
                "the token rectangle is {} x {:?} and the tower's is {} x {:?}; a row copy \
                 does not reshape",
                y.width, y.dtype, src.width, src.dtype
            ),
        ));
    }
    let rows = nonzero(OP, "rows to place", src.rows)?;
    let units = stated(OP, nonzero(OP, "the placed row's width", src.width)?)?;
    let threads = src
        .width
        .div_ceil(WARP)
        .max(1)
        .saturating_mul(WARP)
        .min(MAX_BLOCK);
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!("::pie::layout::scatter_live_rows<{unit}>")),
        )
        .apply(Launch::per_row(rows, threads)),
        &[src.arg(), y.arg(), routes.arg(), units.arg()],
    )
}
