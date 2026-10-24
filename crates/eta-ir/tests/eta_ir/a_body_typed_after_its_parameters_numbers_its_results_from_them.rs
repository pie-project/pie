//! A body over given values (a region's parameters) types its ops' results
//! after them: the parameters are values `0..n`, the first result is `n`.

use eta_ir::infer::{BodyCtx, BodyErrorKind, body_types_after};
use eta_ir::op::Op;
use eta_ir::types::{Dtype, Shape, ValueType};

const NO_CHANNELS: BodyCtx<'static> = BodyCtx {
    channel_types: &[],
    n_names: 0,
};

#[test]
fn a_body_typed_after_its_parameters_numbers_its_results_from_them() {
    let row = ValueType::vector(8, Dtype::F32);
    let types = body_types_after(
        &[row, row],
        &[Op::Add(0, 1), Op::ReduceSum(2)],
        &NO_CHANNELS,
    )
    .expect("the sum of two rows types");
    assert_eq!(types, [row, row, row, ValueType::scalar(Dtype::F32)]);

    let read_past = body_types_after(&[row], &[Op::Exp(1)], &NO_CHANNELS)
        .expect_err("value 1 is the op's own result, not yet defined");
    assert_eq!(read_past.kind, BodyErrorKind::ValueIdOutOfRange(1));

    let mismatched = body_types_after(
        &[row, ValueType::new(Shape::vector(4), Dtype::F32)],
        &[Op::Mul(0, 1)],
        &NO_CHANNELS,
    );
    assert!(mismatched.is_err(), "rows of two widths do not multiply");
}
