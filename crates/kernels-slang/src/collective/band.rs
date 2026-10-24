//! The collectives (`collective.all_reduce`, `all_gather`,
//! `reduce_scatter`) from `kernels/collective/band.slang`: a rank publishes
//! its operand into its slot of the communicator's band, meets the other
//! ranks, folds the band, and meets them again before the band is reused.

use dtype::Dtype;

use crate::encode::{Arg, ArgValue, Ctx, Fire, refuse};
use crate::error::Error;
use crate::tensor::{Comm, Tensor};

const BAND: &str = "collective/band.slang";

/// The workgroup width every instantiation is built at.
const GROUP: u32 = 256;

/// `buf = sum over ranks of buf`.
pub fn all_reduce(ctx: &Ctx<'_>, buf: Tensor) -> Result<(), Error> {
    const OP: &str = "collective.all_reduce";
    let comm = ctx.comm(OP)?;
    let Some(plan) = Plan::of(OP, comm, buf.dtype, elements(buf))? else {
        return Ok(());
    };
    plan.publish(ctx, buf)?;
    ctx.rendezvous(OP)?;
    plan.fold(ctx, "reduce", plan.n, buf)?;
    ctx.rendezvous(OP)
}

/// `y = concat over ranks of x`.
pub fn all_gather(ctx: &Ctx<'_>, x: Tensor, y: Tensor) -> Result<(), Error> {
    const OP: &str = "collective.all_gather";
    let comm = ctx.comm(OP)?;
    if x.dtype != y.dtype {
        return Err(refuse(OP, "a gather does not change the dtype"));
    }
    let Some(plan) = Plan::of(OP, comm, x.dtype, elements(x))? else {
        return Ok(());
    };
    if u64::from(plan.n) * u64::from(comm.world) != elements(y) {
        return Err(refuse(OP, "the gathered rectangle is one shard per rank"));
    }
    if x.dtype == Dtype::Bf16 && !plan.n.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!(
                "a shard of {} bf16 values: the ranks' shards land as whole words",
                plan.n
            ),
        ));
    }
    plan.publish(ctx, x)?;
    ctx.rendezvous(OP)?;
    plan.fold(ctx, "gather", plan.n, y)?;
    ctx.rendezvous(OP)
}

/// `y = this rank's shard of the sum over ranks of x`.
pub fn reduce_scatter(ctx: &Ctx<'_>, x: Tensor, y: Tensor) -> Result<(), Error> {
    const OP: &str = "collective.reduce_scatter";
    let comm = ctx.comm(OP)?;
    if x.dtype != y.dtype {
        return Err(refuse(OP, "a reduction does not change the dtype"));
    }
    let Some(plan) = Plan::of(OP, comm, x.dtype, elements(x))? else {
        return Ok(());
    };
    let shard = elements(y);
    if shard * u64::from(comm.world) != u64::from(plan.n) {
        return Err(refuse(OP, "the reduced rectangle is one shard per rank"));
    }
    let shard =
        u32::try_from(shard).map_err(|_| refuse(OP, "a shard of more than 4Gi elements"))?;
    plan.publish(ctx, x)?;
    ctx.rendezvous(OP)?;
    plan.fold(ctx, "scatter", shard, y)?;
    ctx.rendezvous(OP)
}

struct Plan {
    op: &'static str,
    comm: Comm,
    suffix: &'static str,
    /// The operand's elements.
    n: u32,
    /// A rank's slot of the band, in elements.
    slot: u32,
}

impl Plan {
    fn of(op: &'static str, comm: Comm, dtype: Dtype, n: u64) -> Result<Option<Self>, Error> {
        if n == 0 {
            return Ok(None);
        }
        let (suffix, width) = match dtype {
            Dtype::Bf16 => ("bf16", 2),
            Dtype::F32 => ("f32", 4),
            other => return Err(Error::DtypeUnsupported { op, dtype: other }),
        };
        let n =
            u32::try_from(n).map_err(|_| refuse(op, "a collective of more than 4Gi elements"))?;
        if !comm.slot_bytes.is_multiple_of(4) {
            return Err(refuse(
                op,
                format!(
                    "the band's slot is {} bytes, and a slot starts on a word",
                    comm.slot_bytes
                ),
            ));
        }
        let slot = comm.slot_bytes / width;
        if n > slot {
            return Err(refuse(
                op,
                "the operand is wider than the band this communicator was sized for",
            ));
        }
        Ok(Some(Self {
            op,
            comm,
            suffix,
            n,
            slot,
        }))
    }

    fn publish(&self, ctx: &Ctx<'_>, x: Tensor) -> Result<(), Error> {
        self.launch(
            ctx,
            "publish",
            self.n,
            x.arg(),
            ArgValue::BufferMut(self.comm.band),
        )
    }

    fn fold(&self, ctx: &Ctx<'_>, arm: &'static str, n: u32, out: Tensor) -> Result<(), Error> {
        self.launch(ctx, arm, n, ArgValue::Buffer(self.comm.band), out.arg_mut())
    }

    fn launch(
        &self,
        ctx: &Ctx<'_>,
        arm: &'static str,
        n: u32,
        in_: ArgValue,
        out: ArgValue,
    ) -> Result<(), Error> {
        let entry =
            crate::encode::instantiated(self.op, format!("collective_band_{arm}_{}", self.suffix))?;
        ctx.fire(
            Fire::at(BAND, entry).threads([n.div_ceil(2), 1, 1], [GROUP, 1, 1]),
            &[
                in_,
                out,
                n.arg(),
                self.comm.world.arg(),
                self.comm.rank.arg(),
                self.slot.arg(),
            ],
        )
    }
}

const fn elements(t: Tensor) -> u64 {
    (t.rows as u64) * (t.width as u64)
}
