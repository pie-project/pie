//! What a test program is bound against.

use eta_ir::container::TraceContainer;
use eta_ir::op::Op;
use eta_ir::registry::{Axis, EngineCaps, OutputDecl};
use eta_ir::validate::{BoundTrace, ValidateError, bind};

/// Logits a lane reads out, `vocab` wide.
pub fn logits_decl(vocab: u32) -> OutputDecl {
    OutputDecl {
        name: "logits".into(),
        axis: Axis::Row,
        width: vocab,
        dtype: eta_ir::Dtype::F32,
        layers: 0,
        readout: true,
    }
}

/// A loaded class's output as a program reads it.
pub fn eta_decl(decl: &poem_ir::OutputDecl) -> OutputDecl {
    OutputDecl {
        name: decl.name.clone(),
        axis: match decl.axis {
            poem_ir::InputAxis::Lane => Axis::Lane,
            poem_ir::InputAxis::Row => Axis::Row,
            poem_ir::InputAxis::Voxel => Axis::Voxel,
        },
        width: decl.width,
        dtype: match decl.dtype {
            poem_ir::Dtype::I32 => eta_ir::Dtype::I32,
            _ => eta_ir::Dtype::F32,
        },
        layers: decl.layers,
        readout: decl.readout,
    }
}

/// Every output the loaded classes write, as one lane reads them.
pub fn loaded_decls(loaded: &engine::Capabilities) -> Vec<OutputDecl> {
    let mut decls: Vec<OutputDecl> = Vec::new();
    for decl in loaded
        .classes
        .classes
        .iter()
        .flat_map(|class| &class.outputs)
    {
        if !decls.iter().any(|seen| seen.name == decl.name) {
            decls.push(eta_decl(decl));
        }
    }
    decls
}

pub fn logits(container: TraceContainer, vocab: u32) -> Result<BoundTrace, ValidateError> {
    bind(container, vec![vec![logits_decl(vocab)]], caps())
}

/// One lane per lane `container` reads outputs of, reading what the loaded classes read out.
pub fn loaded(
    container: TraceContainer,
    loaded: &engine::Capabilities,
) -> Result<BoundTrace, ValidateError> {
    let lanes = container
        .stages
        .iter()
        .flat_map(|stage| &stage.ops)
        .filter_map(|op| match *op {
            Op::Output { lane, .. } => Some(lane + 1),
            _ => None,
        })
        .max()
        .unwrap_or(1);
    let decls = loaded_decls(loaded);
    bind(container, vec![decls; lanes as usize], caps())
}

fn caps() -> EngineCaps {
    EngineCaps {
        lora: true,
        attn_page_mask: true,
    }
}
