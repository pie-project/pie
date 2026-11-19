//! A sink the shell runs nothing for is refused by name at registration
//! (`guest::admits`, what a registration's prepare lowers), never lowered
//! to a no-op whose program answers as if it had acted: `minference_sparse`
//! binds (eta-ir knows the name) and XLA refuses it. `lora`, which applies
//! in the forward through the attachment's adapter, still admits. Needs no
//! device.

use eta_ir::container::{StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{EngineCaps, Stage};
use eta_ir::types::Literal;

/// A prologue calling sink `name` on one constant.
fn calls(name: &str) -> TraceContainer {
    TraceContainer {
        names: vec![name.to_string()],
        channels: Vec::new(),
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Prologue,
            ops: vec![
                Op::Const(Literal::F32(0.0)),
                Op::SinkCall {
                    name: 0,
                    args: vec![0],
                },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

fn admits(name: &str) -> Result<(), String> {
    let caps = EngineCaps {
        lora: true,
        attn_page_mask: false,
    };
    let bound = eta_ir::validate::bind(calls(name), vec![Vec::new()], caps)
        .unwrap_or_else(|e| panic!("a program calling `{name}` binds: {e:?}"));
    let stages = eta_compiler::plan::compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let plan = eta_exec::adopt_launch_package(launch).expect("the package adopts");
    engine_xla::guest::admits(&plan.package).map_err(|why| why.to_string())
}

#[test]
fn a_sink_xla_does_not_run_is_refused_at_registration() {
    let why = admits("minference_sparse").expect_err("a sink XLA runs nothing for is refused");
    assert!(
        why.contains("calls sink `minference_sparse`, which this engine does not run"),
        "refused by name: {why}"
    );
    if let Err(why) = admits("lora") {
        assert!(
            !why.contains("calls sink"),
            "`lora` applies in the forward and is no unrun sink: {why}"
        );
    }
}
