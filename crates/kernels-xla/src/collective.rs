//! Collectives over the replicas of one program.
//!
//! The signatures are kernels-slang's (and kernels-cuda's): the world size is
//! read off the shapes (`all_gather` lands `[rows, width * world]`,
//! `reduce_scatter` keeps `[rows, width / world]`, each rank its own column
//! band, as kernels-cuda lays them), and the rank is implicit — an XLA SPMD
//! program is the same on every replica, and `stablehlo.all_gather` /
//! `all_reduce` / `reduce_scatter` over `replica_groups = [[0 .. world)]` need
//! no rank. This backend runs one replica, so a world of one is the only one
//! answered: `all_reduce` leaves its buffer as it is, the other two copy. A
//! wider world is refused until the engine compiles with `num_replicas > 1`.

use crate::error::{Error, refuse};
use crate::hlo::{Func, Val};
use crate::tensor::{Landing, Operand, Stored};

/// The replicas this backend compiles for.
const WORLD: u64 = 1;

fn world_of(op: &'static str, narrow: impl Stored, wide: impl Stored) -> Result<u64, Error> {
    if narrow.dtype() != wide.dtype() {
        return Err(refuse(
            op,
            format!(
                "the shard is {:?} and the whole {:?}; a collective does not convert",
                narrow.dtype(),
                wide.dtype()
            ),
        ));
    }
    if narrow.rows() != wide.rows()
        || narrow.width() == 0
        || !wide.width().is_multiple_of(narrow.width())
    {
        return Err(refuse(
            op,
            format!(
                "a {}x{} shard is not a column band of the {}x{} whole",
                narrow.rows(),
                narrow.width(),
                wide.rows(),
                wide.width()
            ),
        ));
    }
    let world = u64::from(wide.width() / narrow.width());
    if world != WORLD {
        return Err(refuse(
            op,
            format!("the shapes span {world} ranks, and this backend runs {WORLD}"),
        ));
    }
    Ok(world)
}

/// Sums `buf` over the replicas, in place. One replica: nothing moves.
pub fn all_reduce(_buf: impl Stored) -> Result<(), Error> {
    Ok(())
}

/// `y = x_0 ‖ x_1 ‖ …` along the row; one replica copies.
pub fn all_gather(_f: &mut Func, x: Operand, y: Landing) -> Result<Val, Error> {
    const OP: &str = "collective.all_gather";
    world_of(OP, x, y)?;
    let v = x.val;
    Ok(v)
}

/// `y` = this rank's column band of `Σ_r x_r`; one replica copies.
pub fn reduce_scatter(_f: &mut Func, x: Operand, y: Landing) -> Result<Val, Error> {
    const OP: &str = "collective.reduce_scatter";
    world_of(OP, y, x)?;
    let v = x.val;
    Ok(v)
}
