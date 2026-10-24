use kernels_cuda::collective::{self, Landing, Tokens};
use kernels_cuda::{Error, Tensor};
use poem_exec::{DispatchCollective, KernelError};
use poem_ir::Collective;

use crate::run::Run;

impl DispatchCollective for Run<'_> {
    fn dispatch(&mut self, op: &Collective) -> Result<(), KernelError> {
        self.collective(op).map_err(crate::error::kernel)
    }
}

fn backend(op: &'static str, detail: impl ToString) -> Error {
    Error::Backend {
        op,
        detail: detail.to_string(),
    }
}

impl Run<'_> {
    fn collective(&mut self, op: &Collective) -> Result<(), Error> {
        match op {
            Collective::AllReduce { buf, buf_out: _ } => {
                collective::all_reduce(self.ctx(), &mut self.tensor(*buf))
            }
            Collective::AllGather { x, y } => {
                collective::all_gather(self.ctx(), self.tensor(*x), &mut self.tensor(*y))
            }
            Collective::ReduceScatter { x, y } => {
                collective::reduce_scatter(self.ctx(), self.tensor(*x), &mut self.tensor(*y))
            }
            Collective::Dispatch {
                x,
                routes,
                weights,
                top_k,
                layer,
                rows,
                local,
                row_weights,
                origin,
                counts,
                ..
            } => {
                const OP: &str = "collective.dispatch";
                let ep = self
                    .ep_rank()
                    .ok_or_else(|| backend(OP, "this rank is in no expert-parallel group"))?;
                let place = ep
                    .placement(*layer)
                    .ok_or_else(|| backend(OP, format!("no placement for layer {layer}")))?;
                let x = self.tensor(*x);
                let step = self.bucket();
                if x.rows > step {
                    return Err(backend(
                        OP,
                        format!("{} rows past the step's bucket of {step}", x.rows),
                    ));
                }
                let tokens = Tokens {
                    x,
                    routes: self.tensor(*routes),
                    weights: weights.map(|w| self.tensor(w)),
                    step,
                };
                let landing: Vec<Landing> = ep
                    .wires
                    .iter()
                    .map(|w| w.landing(step, *top_k, x.width, x.dtype, weights.is_some()))
                    .collect();
                let own = landing[ep.rank as usize];
                seated(OP, own.rows, self.tensor(*rows))?;
                seated(OP, own.local, self.tensor(*local))?;
                seated(OP, own.origin, self.tensor(*origin))?;
                if let (Some(on), Some(value)) = (own.row_weights, *row_weights) {
                    seated(OP, on, self.tensor(value))?;
                }
                let mut counts = self.tensor(*counts);
                let stream = self.ctx().stream();
                fence(OP, ep, stream)?;
                collective::dispatch(
                    self.ctx(),
                    &ep.wire().ep,
                    place,
                    tokens,
                    &landing,
                    &mut counts,
                )?;
                fence(OP, ep, stream)
            }
            Collective::Combine {
                y,
                origin,
                local,
                top_k,
                routed,
                ..
            } => {
                const OP: &str = "collective.combine";
                let ep = self
                    .ep_rank()
                    .ok_or_else(|| backend(OP, "this rank is in no expert-parallel group"))?;
                let y = self.tensor(*y);
                let local = self.tensor(*local);
                let step = self.bucket();
                let landing = Landing {
                    rows: Tensor::ABSENT,
                    local,
                    row_weights: None,
                    origin: self.tensor(*origin),
                };
                let back: Vec<Tensor> = ep
                    .wires
                    .iter()
                    .map(|w| w.routed(step, *top_k, y.width, y.dtype))
                    .collect();
                seated(OP, back[ep.rank as usize], self.tensor(*routed))?;
                let stream = self.ctx().stream();
                fence(OP, ep, stream)?;
                collective::combine(self.ctx(), &ep.wire().ep, &landing, y, &back)?;
                fence(OP, ep, stream)
            }
        }
    }
}

/// A quiet point on each side of a collective while the group fences them
/// (`EpRank::fence`, as it arms).
fn fence(
    op: &'static str,
    ep: &crate::ep::EpRank,
    stream: *mut core::ffi::c_void,
) -> Result<(), Error> {
    ep.fence(stream).map_err(|why| backend(op, why))
}

/// Refuses an IR value the fire did not seat on the EP buffer its peers
/// write (`ep::Seats`): its readers would read rows nobody wrote.
fn seated(op: &'static str, on: Tensor, value: Tensor) -> Result<(), Error> {
    if value.ptr == on.ptr && value.rows <= on.rows {
        Ok(())
    } else {
        Err(backend(
            op,
            "an IR value not seated on its expert-parallel buffer",
        ))
    }
}
