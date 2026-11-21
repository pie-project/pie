//! A guest program may compile at its own registration (registration is its
//! ready), never inside a fire; what it builds there is counted
//! (`Tally::BuiltAtRegister`), and a later load whose disk cache holds those
//! programs loads them and counts none. The program is an epilogue over the
//! load's logits, a few rounds long, with a constant unique to the run
//! folded into its answer so the first registration finds nothing cached
//! (its size fixed: a program made unique by its length is one whose compile
//! is as large as the chance makes it). Skipped without a PJRT plugin.

use engine::Engine;
use engine::live::Tally;
use engine::program::ProgramRegistration;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::Stage;
use eta_ir::types::{Dtype, Literal, Shape};

use crate::common_llm::Model;

/// Rounds of sin / add the epilogue runs: a few, whatever the run.
const ROUNDS: u32 = 4;

/// An epilogue reading lane 0's logits (`vocab` wide) through `ROUNDS`
/// sin / add rounds, its largest value plus `salt` put on a channel.
fn epilogue(vocab: u32, salt: f32) -> TraceContainer {
    let mut ops = vec![Op::Output {
        index: 0,
        lane: 0,
        shape: Shape::matrix(1, vocab),
        dtype: Dtype::F32,
    }];
    let mut last = 0;
    for round in 0..ROUNDS * 2 {
        ops.push(if round % 2 == 0 {
            Op::Sin(last)
        } else {
            Op::Add(last, 0)
        });
        last += 1;
    }
    ops.push(Op::ReduceMax(last));
    ops.push(Op::Reshape {
        value: last + 1,
        shape: Shape::vector(1),
    });
    ops.push(Op::Const(Literal::F32(salt)));
    ops.push(Op::Broadcast {
        value: last + 3,
        shape: Shape::vector(1),
    });
    ops.push(Op::Add(last + 2, last + 4));
    ops.push(Op::ChanPut {
        chan: 0,
        value: last + 5,
    });
    TraceContainer {
        names: vec!["logits".to_string()],
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
            ops,
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

fn registration(model: &Model, nonce: u32) -> ProgramRegistration {
    let bound = eta_ir::validate::bind(
        epilogue(model.vocab, nonce as f32),
        vec![vec![eta_ir::registry::OutputDecl {
            name: "logits".to_string(),
            axis: eta_ir::registry::Axis::Row,
            width: model.vocab,
            dtype: Dtype::F32,
            layers: 0,
            readout: true,
        }]],
        eta_ir::registry::EngineCaps::default(),
    )
    .expect("the program binds");
    let stages = eta_compiler::plan::compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    ProgramRegistration {
        launch,
        program_hash: 0x7e61_0000_0000 | u64::from(nonce),
        ..Default::default()
    }
}

/// Registers the run's program on a fresh load: what that registration
/// built.
fn register(nonce: u32) -> u64 {
    let mut model = Model::load_with("qwen3-micro-text", None, 256, 7, None);
    let live = model.engine.live();
    let before = live.tallied(Tally::BuiltAtRegister);
    let cold = registration(&model, nonce);
    let compile = model
        .engine
        .prepare_program(&cold)
        .expect("the program prepares");
    let prepared = compile();
    model
        .engine
        .finish_program(prepared)
        .expect("the program registers");
    for other in [Tally::Compiled, Tally::BuiltInFire] {
        assert_eq!(
            live.tallied(other),
            0,
            "registering counted {other:?}: no fire program is built"
        );
    }
    live.tallied(Tally::BuiltAtRegister) - before
}

#[test]
fn a_registration_build_is_counted_and_cached() {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    // A constant no earlier run registered (the disk cache is shared).
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos())
        % (1 << 24);
    let first = register(nonce);
    assert!(
        first > 0,
        "a registration that compiled its stages counted none built at register"
    );
    let again = register(nonce);
    assert_eq!(
        again, 0,
        "a load whose disk cache holds the program's stages built {again} at register"
    );
    eprintln!("registration of nonce {nonce}: built {first} cold, {again} warm");
}
