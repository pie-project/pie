//! Attention on a unit's K / V column (P7, `ops/satt.csl`, `ops/cy_satt.csl`): every row's q, k
//! and v stored down the column, each query attending what is stored (a step's rows, or a decode
//! step's cache in pages across launches).

use super::super::{Kernel, Source};

/// Attention on a K / V column.
pub struct Column;

/// The registry's attention column.
pub static COLUMN: Column = Column;

impl Kernel for Column {
    fn name(&self) -> &'static str {
        "attn.column"
    }

    fn ops(&self) -> &'static [&'static str] {
        &[
            "attn.ragged",
            "attn.prefill",
            "attn.decode",
            "attn.kv_append",
            "attn.sink",
        ]
    }

    fn sources(&self) -> &'static [Source] {
        &[
            (
                "cy_satt.csl",
                include_str!("../../../programs/wafer/ops/cy_satt.csl"),
            ),
            (
                "satt.csl",
                include_str!("../../../programs/wafer/ops/satt.csl"),
            ),
        ]
    }
}
