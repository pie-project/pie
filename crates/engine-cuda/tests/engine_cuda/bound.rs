//! What a test program is bound against: the lanes of its pass, the
//! model's graph and the engine's sinks.

use eta_ir::container::TraceContainer;
use eta_ir::op::{IntrinsicId, Op};
use eta_ir::registry::{EngineCaps, Graph, Readout, ReadoutKind};
use eta_ir::validate::{BoundTrace, ValidateError, bind};

/// One lane reading `vocab`-wide logits.
pub fn logits(container: TraceContainer, vocab: u32) -> Result<BoundTrace, ValidateError> {
    let lane = Readout {
        kind: ReadoutKind::Logits,
        width: vocab,
    };
    bind(container, vec![Some(lane)], graph(), caps())
}

/// As many lanes as `container` reads intrinsics of, each reading out what
/// the loaded classes read out of the kind it reads there.
#[allow(dead_code)]
pub fn loaded(
    container: TraceContainer,
    loaded: &engine::Capabilities,
) -> Result<BoundTrace, ValidateError> {
    let kind_of = |intr: IntrinsicId| match intr {
        IntrinsicId::Velocity => Some(poem_ir::ReadoutKind::Velocity),
        IntrinsicId::Hidden => Some(poem_ir::ReadoutKind::Hidden),
        IntrinsicId::Pixels => Some(poem_ir::ReadoutKind::Pixels),
        IntrinsicId::Logits => Some(poem_ir::ReadoutKind::Logits),
        _ => None,
    };
    let mut lanes: Vec<Option<Readout>> = vec![None];
    for op in container.stages.iter().flat_map(|stage| &stage.ops) {
        let Op::IntrinsicVal { intr, lane, .. } = *op else {
            continue;
        };
        let Some(kind) = kind_of(intr) else { continue };
        let Some(read) = loaded
            .classes
            .classes
            .iter()
            .filter_map(|class| class.readout)
            .find(|readout| readout.kind == kind)
        else {
            continue;
        };
        if lanes.len() <= lane as usize {
            lanes.resize(lane as usize + 1, None);
        }
        lanes[lane as usize] = Some(Readout {
            kind: match read.kind {
                poem_ir::ReadoutKind::Logits => ReadoutKind::Logits,
                poem_ir::ReadoutKind::Velocity => ReadoutKind::Velocity,
                poem_ir::ReadoutKind::Hidden => ReadoutKind::Hidden,
                poem_ir::ReadoutKind::Pixels => ReadoutKind::Pixels,
            },
            width: read.width,
        });
    }
    bind(container, lanes, graph(), caps())
}

fn graph() -> Graph {
    Graph {
        num_layers: 2,
        mtp_logits: true,
        mtp_depth: 1,
        attn_score: true,
    }
}

fn caps() -> EngineCaps {
    EngineCaps {
        lora: true,
        attn_page_mask: true,
    }
}
