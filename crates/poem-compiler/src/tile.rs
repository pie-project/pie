//! Physical layout: a package states the precision a weight is stored at,
//! and the backend picks its arrangement. An affine u4 projection
//! (`u4g64`) that only products read is stored in the matrix core's tiled
//! order on a platform that reads it (`u4g64tiled`): a row-major u4 plane
//! decodes through a scalar arm whose rate follows the SM clock, the tiled
//! one through the matrix core. A weight gathered by rows (an embedding)
//! stays row-major. [`tiled`] is what an import stores tiled, so the trace
//! and the artifact agree.

use poem_ir::{Def, Dtype, Linear, Operands, Operation, TILED_BAND, TILED_STEP, Trace};

/// The params of `trace` [`tile`] lays out tiled, by name.
#[must_use]
pub fn tiled(trace: &Trace) -> Vec<String> {
    tiles(trace)
        .into_iter()
        .map(|p| trace.params[p as usize].name.clone())
        .collect()
}

/// `trace` with each u4 projection its platform reads tiled stored so.
#[must_use]
pub fn tile(mut trace: Trace) -> Trace {
    for p in tiles(&trace) {
        trace.params[p as usize].dtype = Dtype::U4g64tiled;
    }
    trace
}

fn tiles(trace: &Trace) -> Vec<u32> {
    if !trace.platform.reads_placement(Dtype::U4g64tiled) {
        return Vec::new();
    }
    // How each param is read: by products alone, or by anything else.
    let mut by_products = vec![true; trace.params.len()];
    let mut read = vec![false; trace.params.len()];
    let mut ins = Vec::new();
    for node in &trace.nodes {
        ins.clear();
        node.op.inputs(&mut ins);
        let product_w = match &node.op {
            Operation::Linear(Linear::Matmul { w, quant: None, .. }) => Some(*w),
            _ => None,
        };
        for v in &ins {
            if let Def::Weight(p) = trace.values[v.0 as usize].def {
                read[p as usize] = true;
                if product_w != Some(*v) {
                    by_products[p as usize] = false;
                }
            }
        }
    }
    let band = u64::from(TILED_BAND);
    let step = u64::from(TILED_STEP);
    (0..trace.params.len() as u32)
        .filter(|&p| {
            let param = &trace.params[p as usize];
            param.dtype == Dtype::U4g64
                && read[p as usize]
                && by_products[p as usize]
                && param.experts.is_none()
                && matches!(param.shape[..], [n, k] if n % band == 0 && k % step == 0)
        })
        .collect()
}
