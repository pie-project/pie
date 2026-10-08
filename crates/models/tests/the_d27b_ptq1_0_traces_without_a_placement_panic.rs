//! M3d blocker 1 regression: a `Ptq1_0` (ternary, single-inline-plane) weight
//! traces and places without the "6144 logical / 1344 stored" panic.
//!
//! The block-packed codecs above `Ptq1_0` store their contracted axis as a padded
//! BYTE count, but `Ptq1_0` keeps its LOGICAL shape in the trace plane (like the
//! affine codes plane): the engine reserves `rows * row_bytes(k)` from the logical
//! width and the matmul contracts over the logical `k`. Before the fix, `planes()`
//! packed the axis to bytes and `record::restated` rejected the byte width as not
//! dividing the logical dim. This guards both the plain `d27b` and the Bonsai
//! `d27b_bonsai` instances tracing clean in `Ptq1_0`.

use models::qwen_3::model::Model;
use poem::{Dtype, Platform, trace_hybrid};

#[test]
fn a_ptq1_0_d27b_traces_and_places_without_panicking() {
    let plain = trace_hybrid(
        "d27b",
        &Model::d27b_undrafted(Dtype::Ptq1_0, Dtype::Bf16),
        Platform::Metal,
    );
    assert!(plain.nodes.len() > 100, "the d27b Ptq1_0 forward traced");

    let bonsai = trace_hybrid(
        "d27b-bonsai",
        &Model::d27b_bonsai(Dtype::Ptq1_0, Dtype::Bf16),
        Platform::Metal,
    );
    // The Bonsai instance adds the online-Hadamard rotation-undo at every rotated
    // site (the GDN v-heads are reordered tiled→block at import, not in the trace).
    assert!(
        bonsai.nodes.len() > plain.nodes.len(),
        "the Bonsai forward wires strictly more nodes than the plain d27b"
    );
}
