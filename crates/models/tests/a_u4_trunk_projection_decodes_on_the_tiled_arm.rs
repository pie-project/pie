//! A row-major u4 plane decodes through the scalar `matmul_affine` arm, whose
//! rate follows SM clock rather than bandwidth (an A100 decodes 31b at half an
//! L40S), so a u4 trunk's projections are declared tiled.

use poem_ir::{Def, Dtype, Linear, Operation, Platform};

#[test]
fn a_u4_trunk_projection_decodes_on_the_tiled_arm() {
    let sku = models::deployment("gemma4-31b-u4g64-kv-bf16").expect("the 31b u4 row ships");
    let trace = sku.trace(Platform::Cuda);
    let mut row_major = Vec::new();
    for node in &trace.nodes {
        let Operation::Linear(Linear::Matmul { w, .. }) = &node.op else {
            continue;
        };
        let Def::Weight(p) = trace.values[w.0 as usize].def else {
            continue;
        };
        let param = &trace.params[p as usize];
        if param.name.starts_with("layer.") && param.dtype != Dtype::U4g64tiled {
            row_major.push(format!("{} {:?}", param.name, param.dtype));
        }
    }
    assert!(row_major.is_empty(), "{row_major:#?}");
}
