//! Muse Spark Open's rows trace: its residual gates and expert norms are
//! the post-norm gate op, its first layer's experts stay bf16 in the FP4
//! rows, and the routed rest are NVFP4 or MXFP4.

use poem::{Elemwise, Operation, Platform};

#[test]
fn the_muse_spark_rows_bake() {
    let rows: Vec<&poem_compiler::catalog::Deployment> = poem_compiler::catalog::deployments()
        .filter(|row| row.name.starts_with("muse-spark-open"))
        .collect();
    assert!(
        rows.iter()
            .any(|row| row.name.starts_with("muse-spark-open-mini-l8")),
        "the miniature ships: {:?}",
        rows.iter().map(|row| row.name.as_str()).collect::<Vec<_>>()
    );
    for row in rows {
        let trace = row.trace(Platform::Cuda);
        let gates = trace
            .nodes
            .iter()
            .filter(|node| matches!(node.op, Operation::Elemwise(Elemwise::NormGate { .. })))
            .count();
        // Two residual gates and one expert norm a layer.
        let layers = trace
            .params
            .iter()
            .filter(|p| p.name.ends_with(".attn_gate"))
            .count();
        assert!(layers > 0, "`{}` names its layers' gates", row.name);
        assert_eq!(gates, 3 * layers, "`{}`", row.name);
        let bank = |l: usize| {
            trace
                .params
                .iter()
                .find(|p| p.name == format!("layer.{l}.experts_down"))
                .unwrap_or_else(|| panic!("`{}` has layer {l}'s down bank", row.name))
                .dtype
        };
        if row.name.contains("fp4") {
            assert_eq!(
                bank(0),
                poem::Dtype::Bf16,
                "`{}`: layer 0 keeps bf16 experts",
                row.name
            );
            assert_ne!(
                bank(1),
                poem::Dtype::Bf16,
                "`{}`: layer 1's are quantized",
                row.name
            );
        }
    }
}
