//! A guest program runs on the device or not at all: one that reads an
//! output no fire of the load returns on the device is refused at its
//! registration, by name, and never reaches a fire to find it unbound (a
//! host interpreter once took such passes). The same program reading the
//! load's `logits` registers. Skipped without a PJRT plugin.

use engine::program::ProgramRegistration;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::Stage;
use eta_ir::types::{Dtype, Shape};

use crate::common_llm::Model;

/// An epilogue reading output `name` of `vocab` columns at lane 0 and
/// putting its largest value on a channel.
fn reads(name: &str, vocab: u32) -> TraceContainer {
    TraceContainer {
        names: vec![name.to_string()],
        channels: vec![ChannelDecl {
            shape: Shape::vector(1),
            dtype: ChanDType::Concrete(Dtype::F32),
            capacity: 1,
            host_role: HostRole::None,
            seeded: false,
        }],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::Output {
                    index: 0,
                    lane: 0,
                    shape: Shape::matrix(1, vocab),
                    dtype: Dtype::F32,
                },
                Op::ReduceMax(0),
                Op::Reshape {
                    value: 1,
                    shape: Shape::vector(1),
                },
                Op::ChanPut { chan: 0, value: 2 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

fn register(model: &mut Model, name: &str) -> Result<engine::program::ProgramId, String> {
    let bound = eta_ir::validate::bind(
        reads(name, model.vocab),
        vec![vec![eta_ir::registry::OutputDecl {
            name: name.to_string(),
            axis: eta_ir::registry::Axis::Row,
            width: model.vocab,
            dtype: Dtype::F32,
            layers: 0,
            readout: true,
        }]],
        eta_ir::registry::EngineCaps::default(),
    )
    .map_err(|e| format!("bind: {e:?}"))?;
    let stages = eta_compiler::plan::compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    engine::Programs::register(
        &mut model.engine,
        &ProgramRegistration {
            launch,
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string())
}

#[test]
fn a_guest_reading_no_plane_is_refused_at_registration() {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    let mut model = Model::load_with("qwen3-micro-text", None, 256, 7, None);
    register(&mut model, "logits").expect("a program reading the load's logits registers");
    let why = register(&mut model, "absent").expect_err("a program reading no plane is refused");
    assert!(
        why.contains("reads output `absent`") && why.contains("no fire of this load returns"),
        "refused by name: {why}"
    );
}
