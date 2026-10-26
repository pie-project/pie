//! The outputs a test program's lanes read.

use eta_dsl::Dtype;
use eta_ir::registry::{Axis, OutputDecl};

pub fn decl(name: &str, axis: Axis, width: u32, readout: bool) -> OutputDecl {
    OutputDecl {
        name: name.into(),
        axis,
        width,
        dtype: Dtype::F32,
        layers: 0,
        readout,
    }
}

/// One lane reading `vocab`-wide logits on its readout rows.
pub fn logits(vocab: u32) -> Vec<Vec<OutputDecl>> {
    vec![vec![decl("logits", Axis::Row, vocab, true)]]
}
