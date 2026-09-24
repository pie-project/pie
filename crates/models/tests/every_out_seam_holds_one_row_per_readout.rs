use model_dsl::{Dim, Platform, Ty, seam};

/// The engine reads the `out` seam by readout index: a lane's rows in the
/// readback are `readout_first..readout_first + readout_count`, counted over
/// the fire's readouts, not its tokens. A tokens-shaped `out` therefore hands
/// a prefill's `Readout::Last` row 0 of the plane, the prompt's first
/// position. Every head reads its `readout_rows` before the `lm_head`, so the
/// seam carries exactly one row per readout.
#[test]
fn every_out_seam_holds_one_row_per_readout() {
    let mut faults = Vec::new();
    let mut seen = 0usize;

    for row in models::skus() {
        let trace = (row.trace)(Platform::Cuda);
        for out in trace.seams.iter().filter(|s| s.seam == seam::OUT.name) {
            for value in &out.values {
                seen += 1;
                let rows = match &trace.values[value.0 as usize].ty {
                    Ty::Tensor { shape, .. } => shape.first().copied(),
                    Ty::Struct(_) => None,
                };
                if rows != Some(Dim::Readouts) {
                    faults.push(format!(
                        "`{}` lands its `out` seam over {rows:?} rows: the engine reads \
                         that plane by readout index, so a prefill's last-token logits \
                         come from row 0; the head owes a `gather_rows` over \
                         `readout_rows` first",
                        row.name,
                    ));
                }
            }
        }
    }

    assert!(seen > 0, "no catalog SKU lands an `out` seam");
    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}
