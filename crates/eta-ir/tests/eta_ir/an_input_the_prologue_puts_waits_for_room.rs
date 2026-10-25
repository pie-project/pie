use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer, decode};
use eta_ir::op::Op;
use eta_ir::registry::{EngineCaps, Graph, Phase, Stage};
use eta_ir::types::{Dtype, Literal, Shape};
use eta_ir::validate::{Direction, ReadinessEntry, bind};

/// Two guest inputs: the prologue puts the first before the forward reads
/// it, nothing touches the second.
#[test]
fn an_input_the_prologue_puts_waits_for_room() {
    let latent = ChannelDecl {
        shape: Shape::matrix(4, 8),
        dtype: ChanDType::Concrete(Dtype::F32),
        capacity: 1,
        host_role: HostRole::None,
        seeded: false,
    };
    let container = TraceContainer {
        channels: vec![latent.clone(), latent],
        stages: vec![StageProgram {
            stage: Stage::Prologue,
            ops: vec![
                Op::Const(Literal::F32(0.5)),
                Op::Broadcast {
                    value: 0,
                    shape: Shape::matrix(4, 8),
                },
                Op::ChanPut { chan: 0, value: 1 },
            ],
        }],
        inputs: vec![0, 1],
        ..TraceContainer::default()
    };
    assert_eq!(decode(&container.encode()).as_ref(), Ok(&container));

    let bound = bind(
        container,
        vec![None],
        Graph::default(),
        EngineCaps::default(),
    )
    .expect("binds");
    assert_eq!(
        bound.readiness,
        vec![
            ReadinessEntry {
                chan: 0,
                phase: Phase::Prologue,
                dir: Direction::NeedsEmpty,
            },
            ReadinessEntry {
                chan: 1,
                phase: Phase::Descriptor,
                dir: Direction::NeedsFull,
            },
        ]
    );
}
