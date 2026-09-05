use dtype::Dtype;

use crate::encode::{Arg, ArgValue, Ctx, Fire};
use crate::error::Error;
use crate::tensor::{Comm, Tensor};

const GROUP: u32 = 256;

pub fn all_reduce(ctx: &Ctx<'_>, buf: Tensor) -> Result<(), Error> {
    const OP: &str = "collective.all_reduce";
    let comm = ctx.comm(OP)?;
    let Some(plan) = Plan::of(OP, &comm, buf.dtype, elements(buf))? else {
        return Ok(());
    };
    plan.publish(ctx, buf)?;
    ctx.rendezvous(OP)?;
    plan.fold(ctx, "reduce", plan.n, buf)?;
    ctx.rendezvous(OP)
}

pub fn all_gather(ctx: &Ctx<'_>, x: Tensor, y: Tensor) -> Result<(), Error> {
    const OP: &str = "collective.all_gather";
    let comm = ctx.comm(OP)?;
    if x.dtype != y.dtype {
        return Err(shape(OP, "a gather does not change the dtype"));
    }
    let Some(plan) = Plan::of(OP, &comm, x.dtype, elements(x))? else {
        return Ok(());
    };
    if u64::from(plan.n) * u64::from(comm.world) != elements(y) {
        return Err(shape(OP, "the gathered rectangle is one shard per rank"));
    }
    plan.publish(ctx, x)?;
    ctx.rendezvous(OP)?;
    plan.fold(ctx, "gather", plan.n, y)?;
    ctx.rendezvous(OP)
}

pub fn reduce_scatter(ctx: &Ctx<'_>, x: Tensor, y: Tensor) -> Result<(), Error> {
    const OP: &str = "collective.reduce_scatter";
    let comm = ctx.comm(OP)?;
    if x.dtype != y.dtype {
        return Err(shape(OP, "a reduction does not change the dtype"));
    }

    let Some(plan) = Plan::of(OP, &comm, x.dtype, elements(x))? else {
        return Ok(());
    };
    let shard = elements(y);
    if shard * u64::from(comm.world) != u64::from(plan.n) {
        return Err(shape(OP, "the reduced rectangle is one shard per rank"));
    }
    let Ok(shard) = u32::try_from(shard) else {
        return Err(shape(OP, "a shard of more than 4Gi elements"));
    };
    plan.publish(ctx, x)?;
    ctx.rendezvous(OP)?;
    plan.fold(ctx, "scatter", shard, y)?;
    ctx.rendezvous(OP)
}

struct Plan {
    op: &'static str,
    comm: Comm,
    dtype: Dtype,

    suffix: &'static str,

    n: u32,

    slot: u32,
}

impl Plan {
    fn of(op: &'static str, comm: &Comm, dtype: Dtype, n: u64) -> Result<Option<Self>, Error> {
        if n == 0 {
            return Ok(None);
        }
        let (suffix, width) = match dtype {
            Dtype::Bf16 => ("bf16", 2),
            Dtype::F32 => ("f32", 4),
            other => return Err(Error::DtypeUnsupported { op, dtype: other }),
        };
        let Ok(n) = u32::try_from(n) else {
            return Err(shape(op, "a collective of more than 4Gi elements"));
        };
        let Some(slot) = comm.slot_elems(width) else {
            return Err(shape(op, "the band's stride is not whole in this dtype"));
        };
        if n > slot {
            return Err(shape(
                op,
                "the operand is wider than the band this communicator was sized for",
            ));
        }
        Ok(Some(Self {
            op,
            comm: *comm,
            dtype,
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
            self.comm.band(self.dtype).arg_mut(),
            self.n,
        )
    }

    fn fold(&self, ctx: &Ctx<'_>, arm: &'static str, lanes: u32, out: Tensor) -> Result<(), Error> {
        self.launch(
            ctx,
            arm,
            lanes,
            self.comm.band(self.dtype).arg(),
            out.arg_mut(),
            lanes,
        )
    }

    fn launch(
        &self,
        ctx: &Ctx<'_>,
        arm: &'static str,
        lanes: u32,
        in_: ArgValue,
        out: ArgValue,
        n: u32,
    ) -> Result<(), Error> {
        let entrypoint = point(arm, self.suffix).ok_or_else(|| Error::Backend {
            op: self.op,
            detail: format!("no `collective_{arm}_{}` point", self.suffix),
        })?;
        ctx.fire(
            Fire::at("collective/band.slang", entrypoint).threads([lanes, 1, 1], [GROUP, 1, 1]),
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

const fn point(arm: &str, suffix: &str) -> Option<&'static str> {
    let bf16 = matches!(suffix.as_bytes(), b"bf16");
    Some(match arm.as_bytes() {
        b"publish" if bf16 => "collective_publish_bf16",
        b"publish" => "collective_publish_f32",
        b"reduce" if bf16 => "collective_reduce_bf16",
        b"reduce" => "collective_reduce_f32",
        b"gather" if bf16 => "collective_gather_bf16",
        b"gather" => "collective_gather_f32",
        b"scatter" if bf16 => "collective_scatter_bf16",
        b"scatter" => "collective_scatter_f32",
        _ => return None,
    })
}

const fn elements(t: Tensor) -> u64 {
    (t.rows as u64) * (t.width as u64)
}

fn shape(op: &'static str, detail: &'static str) -> Error {
    Error::Backend {
        op,
        detail: detail.to_string(),
    }
}
