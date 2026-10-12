//! A `Ptq1_0` trunk keeps each weight's logical width in the trace, which
//! placement reserves its packed rows by.

use poem::{Dtype, Platform};
use poem_compiler::catalog::trace_of;

use crate::common;

#[test]
fn a_ptq1_0_trunk_traces_and_places() {
    let trace = trace_of("qwen36-27b", Dtype::Ptq1_0, Dtype::Bf16, Platform::Metal);
    common::bake(&trace).unwrap_or_else(|why| panic!("the Ptq1_0 trunk places: {why}"));
}
