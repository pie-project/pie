//! A matmul in the GEMM regime (`gemm_pe.csl`'s `gemm`): a class's rows times a weight block off
//! the tape below the array, every PE of a row reading every other's piece along the row line.

use super::super::{GemmForm, Kernel, LineAlong, Source};

/// A GEMM.
pub struct Gemm;

/// The registry's GEMM.
pub static GEMM: Gemm = Gemm;

/// Its GEMM form: the row line, which orders the row (no PE leaves before every other has
/// reached it, but for its own tail).
static FORM: GemmForm = GemmForm {
    func: "gemm",
    what: "a GEMM",
    line: LineAlong::Row,
    masked: true,
    orders_row: |_| true,
    drift: "",
    drift_after: "",
};

impl Kernel for Gemm {
    fn name(&self) -> &'static str {
        "linear.gemm"
    }

    fn ops(&self) -> &'static [&'static str] {
        &["linear.matmul"]
    }

    fn sources(&self) -> &'static [Source] {
        &[]
    }

    fn gemm(&self) -> Option<&'static GemmForm> {
        Some(&FORM)
    }
}
