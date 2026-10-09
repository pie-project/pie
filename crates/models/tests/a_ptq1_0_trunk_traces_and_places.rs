//! A `Ptq1_0` trunk keeps each weight's logical width in the trace, which
//! placement reserves its packed rows by.

use models::star::trace_of;
use poem::{Dtype, Platform};

#[test]
fn a_ptq1_0_trunk_traces_and_places() {
    let trace = trace_of("qwen36-27b", Dtype::Ptq1_0, Dtype::Bf16, Platform::Metal);
    assert!(trace.nodes.len() > 100, "the Ptq1_0 forward traced");
}
