//! A registration binds within what the device serving it states
//! (`validate::bind_within`): a guest that declares a stage whose values
//! pass an instance's state room is refused by name, at its bind, before
//! the host evaluates any of it or the device lays any of it.

use eta_ir::container::{StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{EngineCaps, Stage};
use eta_ir::types::{Literal, Shape};
use eta_ir::validate::{Budget, ValidateError, bind_within, footprints};

/// A prologue that broadcasts one float to `shape`: its values are the
/// literal's four bytes and the broadcast's.
fn broadcast(shape: Shape) -> TraceContainer {
    TraceContainer {
        stages: vec![StageProgram {
            stage: Stage::Prologue,
            ops: vec![
                Op::Const(Literal::F32(1.0)),
                Op::Broadcast { value: 0, shape },
            ],
        }],
        ..TraceContainer::default()
    }
}

#[test]
fn a_stage_past_its_budget_is_refused_by_name() {
    let room = Budget {
        room_bytes: 1 << 20,
        submit: None,
    };
    let fits = bind_within(
        broadcast(Shape::vector(1024)),
        vec![],
        EngineCaps::default(),
        room,
    )
    .expect("a 4 KiB stage binds within a MiB");
    assert_eq!(footprints(&fits).unwrap()[0].bytes, 4 + 4 * 1024);
    assert_eq!(
        fits.budget,
        Some(room),
        "the bind holds what it was held to"
    );

    // A stage of 64 Gi floats: refused by name, with both numbers.
    let wide = Shape::matrix(1 << 16, 1 << 20);
    let refused = bind_within(broadcast(wide), vec![], EngineCaps::default(), room).unwrap_err();
    assert_eq!(
        refused,
        ValidateError::StagePastRoom {
            stage: Stage::Prologue,
            bytes: 4 + 4 * (1u64 << 36),
            room: 1 << 20,
        }
    );
    assert!(refused.to_string().contains("state room"), "{refused}");

    // The exact room binds; a byte less does not.
    let exact = Budget {
        room_bytes: 4 + 4 * 1024,
        submit: None,
    };
    assert!(
        bind_within(
            broadcast(Shape::vector(1024)),
            vec![],
            EngineCaps::default(),
            exact
        )
        .is_ok()
    );
    let short = Budget {
        room_bytes: 3 + 4 * 1024,
        submit: None,
    };
    assert!(matches!(
        bind_within(
            broadcast(Shape::vector(1024)),
            vec![],
            EngineCaps::default(),
            short
        ),
        Err(ValidateError::StagePastRoom { .. })
    ));
}

/// A stage whose values' bytes pass a u64 is refused by name, not wrapped
/// into a small number that would fit.
#[test]
fn a_stage_past_a_u64_of_bytes_is_refused_by_name() {
    let past = Shape::new(&[u32::MAX, u32::MAX]).expect("its elements fit a u64");
    let refused = bind_within(
        broadcast(past),
        vec![],
        EngineCaps::default(),
        Budget {
            room_bytes: u64::MAX,
            submit: None,
        },
    )
    .unwrap_err();
    assert_eq!(
        refused,
        ValidateError::StageBytesOverflow {
            stage: Stage::Prologue
        }
    );
}
