//! A `fused.region`'s launch, the same on every build: the generated row
//! kernel the load made for its body over the planes of its values.

use kernels_cuda::Tensor;
use kernels_cuda::synth::Plane;
use poem_ir::ValueId;

use crate::run::Run;

impl Run<'_> {
    /// A `fused.region`: the kernel the load generated for its body over the
    /// planes of its values, a weight's one row every row reads.
    pub(super) fn region(
        &self,
        ins: &[ValueId],
        reads: &[poem_ir::RegionRead],
        width: u32,
        body: u64,
        outs: &[ValueId],
    ) -> Result<(), kernels_cuda::Error> {
        const OP: &str = "fused.region";
        let broken = |detail: String| kernels_cuda::Error::Backend { op: OP, detail };
        let forms = crate::synth::forms(self.values(), ins, reads, |v| Some(self.tensor(v).dtype))
            .map_err(broken)?;
        let kernel = self.region_kernel(body, width, &forms).ok_or_else(|| {
            broken(format!(
                "the load generated no kernel for body {body:016x} at {width}"
            ))
        })?;
        let landed: Vec<Tensor> = outs.iter().map(|&v| self.tensor(v)).collect();
        let rows = landed.first().map_or(0, |t| t.rows);
        // A row cut into its heads: each part of it one of the kernel's rows.
        let row_width = landed.first().map_or(width, |t| t.width);
        if width == 0 || !row_width.is_multiple_of(width) {
            return Err(broken(format!(
                "the region's rows are {width} wide and it lands rows {row_width} wide"
            )));
        }
        let mut planes = Vec::with_capacity(ins.len());
        for (k, (&v, form)) in ins.iter().zip(&forms).enumerate() {
            let t = self.tensor(v);
            let read = reads.get(k).copied().unwrap_or_default();
            planes.push(match form {
                kernels_cuda::synth::In::Shared(_) | kernels_cuda::synth::In::One(_) => {
                    Plane::shared(&t)
                }
                // a table's rows, each row reading the one its id picks
                kernels_cuda::synth::In::Gathered(_) => {
                    let by = read.by.and_then(|by| ins.get(by as usize)).ok_or_else(|| {
                        broken(format!("parameter {k} is picked by no input of the region"))
                    })?;
                    Plane::cols(&t, read.from).by(&self.tensor(*by))
                }
                _ if t.rows >= rows => Plane::cols(&t, read.from),
                _ => {
                    return Err(broken(format!(
                        "parameter value {} holds {} rows and the region lands {rows}",
                        v.0, t.rows
                    )));
                }
            });
        }
        let landed: Vec<Plane> = landed.iter().map(Plane::rows).collect();
        kernel.fire_parts(self.ctx(), &planes, &landed, rows, row_width / width)
    }
}
