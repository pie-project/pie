use kernels_xla::sampling;
use poem_exec::{DispatchSampling, KernelError};
use poem_ir::{Operands, Sampling};

use crate::run::Run;

impl DispatchSampling for Run<'_> {
    fn dispatch(&mut self, op: &Sampling) -> Result<(), KernelError> {
        self.ctx().scope(op.name());
        self.sampling(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn sampling(&mut self, op: &Sampling) -> Result<(), kernels_xla::Error> {
        match op {
            Sampling::TopK {
                x,
                k,
                values,
                indices,
            } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let o_values = self.tensor(*values);
                let o_indices = self.tensor(*indices);
                let (v0, v1) =
                    sampling::topk(l.f(), x_v, *k, o_values.landing(), o_indices.landing())?;
                l.land(o_values, v0)?;
                l.land(o_indices, v1)?;
                Ok(())
            }),
            Sampling::Argmax { xs, y } => {
                if let Some(&first) = xs.first()
                    && xs.iter().all(|&x| x == first)
                {
                    return crate::lower::lower(self.ctx(), |l| {
                        let x_v = l.read(self.tensor(first))?;
                        let o_y = self.tensor(*y);
                        let v = sampling::argmax(l.f(), x_v, 0, o_y.landing())?;
                        for column in 0..xs.len() {
                            l.land_column(
                                o_y,
                                u32::try_from(column).expect("a draft depth inside u32"),
                                v,
                            )?;
                        }
                        Ok(())
                    });
                }
                for (column, x) in xs.iter().enumerate() {
                    crate::lower::lower(self.ctx(), |l| {
                        let x_v = l.read(self.tensor(*x))?;
                        let o_y = self.tensor(*y);
                        let v = sampling::argmax(
                            l.f(),
                            x_v,
                            u32::try_from(column).expect("a draft depth inside u32"),
                            o_y.landing(),
                        )?;
                        l.land_column(
                            o_y,
                            u32::try_from(column).expect("a draft depth inside u32"),
                            v,
                        )
                    })?;
                }
                Ok(())
            }
            Sampling::SelectorWalk {
                cand,
                unary,
                hp,
                tokens,
                pred,
                succ,
                first,
                picks,
            } => {
                // The candidates are readout-rowed (a block drafter reads
                // every row back), the tokens and the projected hidden
                // token-rowed, padded past them: the GPU kernel reads the
                // first rows.
                let hp = hp.map(|hp| self.first_rows(self.tensor(hp), *cand));
                let tokens = self.first_rows(self.tensor(*tokens), *cand);
                let picks = self.tensor(*picks);
                crate::lower::lower(self.ctx(), |l| {
                    let cand = l.ragged(self.ragged(*cand))?;
                    let unary = l.read(self.tensor(*unary))?;
                    let hp = l.read_opt(hp)?;
                    let tokens = l.read(tokens)?;
                    let pred = l.read(self.tensor(*pred))?;
                    let succ = l.read(self.tensor(*succ))?;
                    let picked = l.read(picks)?;
                    let v = sampling::selector_walk::selector_walk(
                        l.f(),
                        cand,
                        unary,
                        hp,
                        tokens,
                        pred,
                        succ,
                        *first,
                        picked,
                    )?;
                    l.land(picks, v)
                })
            }
            Sampling::MarkovWalk { .. } => Err(kernels_xla::Error::Unsupported { op: op.name() }),
        }
    }
}
