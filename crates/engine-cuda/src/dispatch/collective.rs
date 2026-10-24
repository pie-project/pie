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
                let step = self.bucket().max(x.rows);
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
                let mut counts = self.tensor(*counts);
                collective::dispatch(
                    self.ctx(),
                    &ep.wire().ep,
                    place,
                    tokens,
                    &landing,
                    &mut counts,
                )?;
                // Peers wrote into this rank's fixed landing; the IR's values
                // are the arena's.
                let own = landing[ep.rank as usize];
                self.land(OP, own.rows, self.tensor(*rows))?;
                self.land(OP, own.local, self.tensor(*local))?;
                self.land(OP, own.origin, self.tensor(*origin))?;
                if let (Some(from), Some(to)) = (own.row_weights, *row_weights) {
                    self.land(OP, from, self.tensor(to))?;
                }
                Ok(())
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
                let step = local.rows / ep.wires.len() as u32;
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
                collective::combine(self.ctx(), &ep.wire().ep, &landing, y, &back)?;
                // An eager walk's `routed` is its live rows: the return holds
                // them from its first row, as the dispatch numbered them.
                self.land(OP, back[ep.rank as usize], self.tensor(*routed))
            }
        }
    }

    /// Copies the head of `from` (a fixed EP buffer) into the IR's `to`.
    fn land(&self, op: &'static str, from: Tensor, to: Tensor) -> Result<(), Error> {
        let bytes = to.elements() * to.dtype.bytes_ceil();
        if bytes > from.elements() * from.dtype.bytes_ceil() {
            return Err(backend(op, "an IR value wider than its EP buffer"));
        }
        crate::device::alloc::copy_any(
            self.ctx().stream(),
            to.ptr,
            from.ptr,
            usize::try_from(bytes).unwrap_or(usize::MAX),
        )
        .map_err(|fault| backend(op, fault))
    }
}
