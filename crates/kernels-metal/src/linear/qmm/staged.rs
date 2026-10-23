//! The staged tile (`linear/qmm_staged.metal`): eight rows a tile, the
//! contraction split in partials when the columns alone do not fill the
//! device; wide enough, the register-fed cooperative tile instead.

use crate::encode::{Arg, Ctx, Fire, Grid};
use crate::error::Error;
use crate::tensor::{Bank, Tensor};

use super::tiled::splitk_reduce;
use super::{Scratch, symbol};

const STAGED_FILE: &str = "linear/qmm_staged.metal";

/// The rows of one staged tile.
pub(super) const TILE: i32 = 8;

const STAGED_SPLIT_WIDE: i32 = 8;

const STAGED_SPLIT: i32 = 4;

const STAGED_DEEP: i32 = 8192;

const STAGED_FEW_TILES: i32 = 32;

/// The contraction span the register-fed tile permutes within: four quant groups.
const COOP_SPAN: i32 = 256;

/// Narrower than this the staged tile is as quick (a 1024-column projection is 32 tiles).
const COOP_LEAST_COLUMNS: i32 = 4096;

/// From this many columns the register-fed tile fills the device without a k cut.
const COOP_WHOLE_COLUMNS: i32 = 12288;

const COOP_SPLIT: i32 = 4;

#[allow(clippy::too_many_arguments)]
pub(super) fn fire(
    ctx: &Ctx<'_>,
    op: &'static str,
    (act, w, biases, y): (Tensor, Bank, Tensor, Tensor),
    scratch: &Scratch<'_>,
    (m, n, k, cols, ld): (i32, i32, i32, i32, i32),
) -> Result<(), Error> {
    let tiles = cols / 64;
    if cols >= COOP_LEAST_COLUMNS && k % COOP_SPAN == 0 {
        // Wide enough to fill the device from its columns alone, it runs whole; narrower, the
        // contraction is cut in four so a 5120-column projection is not 160 simdgroups.
        let split = if cols >= COOP_WHOLE_COLUMNS || (k / COOP_SPAN) % COOP_SPLIT != 0 {
            1
        } else {
            COOP_SPLIT
        };
        let partial = if split > 1 {
            (scratch.partials)(m.unsigned_abs() * split.unsigned_abs(), n.unsigned_abs())
        } else {
            None
        };
        if split == 1 || partial.is_some() {
            let entry = symbol(&format!("affine_qmm_coop_m{TILE}_k{split}"));
            let stamp = symbol(&format!("PIE_COOP_POINT(\"{entry}\",{split})"));
            ctx.fire(
                Fire::at(STAGED_FILE, entry).stamp(stamp).apply(Grid::of(
                    [64 * tiles.unsigned_abs(), split.unsigned_abs(), 1],
                    [64, 1, 1],
                )),
                &[
                    w.codes.arg(),
                    w.scales.arg(),
                    biases.arg(),
                    act.arg(),
                    y.arg_mut(),
                    k.arg(),
                    n.arg(),
                    match partial {
                        Some(partial) => partial.arg_mut(),
                        None => ctx.absent()?,
                    },
                    m.arg(),
                    ld.arg(),
                ],
            )?;
            return match partial {
                Some(partial) => splitk_reduce(ctx, op, y, partial, (m, n, cols), split),
                None => Ok(()),
            };
        }
    }
    let mut split = if k >= STAGED_DEEP || tiles < STAGED_FEW_TILES {
        STAGED_SPLIT_WIDE
    } else {
        STAGED_SPLIT
    };
    while split > 1 && (k / 64) % split != 0 {
        split /= 2;
    }
    let partial = if split > 1 {
        (scratch.partials)(m.unsigned_abs() * split.unsigned_abs(), n.unsigned_abs())
    } else {
        None
    };
    let split = if partial.is_some() { split } else { 1 };
    let entry = symbol(&format!("affine_qmm_staged_m{TILE}_k{split}"));
    let stamp = symbol(&format!("PIE_STAGED_POINT(\"{entry}\",{TILE},{split})"));
    ctx.fire(
        Fire::at(STAGED_FILE, entry).stamp(stamp).apply(Grid::of(
            [64 * tiles.unsigned_abs(), split.unsigned_abs(), 1],
            [64, 1, 1],
        )),
        &[
            w.codes.arg(),
            w.scales.arg(),
            biases.arg(),
            act.arg(),
            y.arg_mut(),
            k.arg(),
            n.arg(),
            match partial {
                Some(partial) => partial.arg_mut(),
                None => ctx.absent()?,
            },
            m.arg(),
            ld.arg(),
        ],
    )?;
    match partial {
        Some(partial) => splitk_reduce(ctx, op, y, partial, (m, n, cols), split),
        None => Ok(()),
    }
}
