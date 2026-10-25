use poem_ir::Operation;

/// Every op the inference's tables name is an op: a row under a name no op
/// carries never applies.
#[test]
fn every_op_the_inference_names_is_an_op() {
    let named = super::UNPACKS
        .iter()
        .chain(super::AFFINE.iter())
        .chain(super::PER_HEAD.iter().map(|(op, _)| op));
    let stale: Vec<&&str> = named.filter(|name| !Operation::names_an_op(name)).collect();
    assert!(stale.is_empty(), "named by no op: {stale:?}");
}
