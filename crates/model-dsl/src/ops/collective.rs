//! The `Collective` family: the cross-rank reductions and gathers.

use super::*;

pub fn all_reduce(buf: &Value) -> Value {
    let r = buf.rec();
    let buf_out = r.fresh(buf.ty().clone());
    r.push(
        Collective::AllReduce {
            buf: buf.id(),
            buf_out: buf_out.id(),
        },
        &[buf],
    );
    buf_out
}

/// Concatenates each rank's `width`-shard into the full tensor.
pub fn all_gather(x: &Value, world: u32) -> Value {
    let r = x.rec();
    let y = r.fresh(tensor(x.rows(), x.width() * u64::from(world), x.dtype()));
    r.push(
        Collective::AllGather {
            x: x.id(),
            y: y.id(),
        },
        &[x],
    );
    y
}

/// Sums across ranks, leaving each rank its `width`-shard of the result.
pub fn reduce_scatter(x: &Value, world: u32) -> Value {
    let world = u64::from(world);
    assert!(
        x.width().is_multiple_of(world),
        "a width of {} does not scatter {world} ways",
        x.width(),
    );
    let r = x.rec();
    let y = r.fresh(tensor(x.rows(), x.width() / world, x.dtype()));
    r.push(
        Collective::ReduceScatter {
            x: x.id(),
            y: y.id(),
        },
        &[x],
    );
    y
}
