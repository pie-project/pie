//! A lane fires on while a guest program's registration compiles cold on
//! another thread (`Engine::prepare_program`, then `finish_program` here):
//! its fires keep landing, none waiting out the compile, and the program
//! registers at the finish. XLA's form of the GPU suite's
//! `runtime::register::a_cold_registration_compiles_while_a_lane_fires`
//! (sc/np-register-offthread-3): the cold program is an epilogue reading the
//! load's logits through `ROUNDS` rounds of transcendentals, code no
//! program compiled yet. Reports the compile's time and the largest gap
//! between two fires; a compile run on the fire path would stall a fire for
//! about its whole time. Skipped without a PJRT plugin.

use std::time::{Duration, Instant};

use engine::program::ProgramRegistration;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::Stage;
use eta_ir::types::{Dtype, Shape};

use crate::common_llm::{Model, Refs, Seq, tokens};

/// Rounds of the cold program: four ops each.
const ROUNDS: u32 = 512;

/// An epilogue reading lane 0's logits (`vocab` wide), through `rounds`
/// rounds of sin, exp, cos and an add of the logits, its largest value put
/// on a channel.
fn heavy(vocab: u32, rounds: u32) -> TraceContainer {
    let mut ops = vec![Op::Output {
        index: 0,
        lane: 0,
        shape: Shape::matrix(1, vocab),
        dtype: Dtype::F32,
    }];
    let mut last = 0;
    for round in 0..rounds * 4 {
        ops.push(match round % 4 {
            0 => Op::Sin(last),
            1 => Op::Exp(last),
            2 => Op::Cos(last),
            _ => Op::Add(last, 0),
        });
        last += 1;
    }
    ops.push(Op::ReduceMax(last));
    ops.push(Op::Reshape {
        value: last + 1,
        shape: Shape::vector(1),
    });
    ops.push(Op::ChanPut {
        chan: 0,
        value: last + 2,
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

fn registration(model: &Model) -> ProgramRegistration {
    let bound = eta_ir::validate::bind(
        heavy(model.vocab, ROUNDS),
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
        program_hash: 0xc01d,
        ..Default::default()
    }
}

#[test]
fn a_cold_registration_compiles_while_a_lane_fires() {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    let mut model = Model::load_with("qwen3-micro-text", None, 256, 7, None);
    let mut refs = Refs::of(&model);
    let vocab = model.vocab;
    let mut seq = Seq::new(0);
    let prompt = seq.lane(&model, &mut refs, tokens(3, 8, vocab));
    model.fire(vec![prompt]);
    seq.held += 8;
    // A decode step at the lane's next position, fired again and again (it
    // rewrites that one position): the lane fires for as long as the
    // compile takes, its context never running out.
    let step = |model: &mut Model, refs: &mut Refs, seq: &mut Seq| {
        let lane = seq.lane(model, refs, vec![(seq.held * 7) % vocab]);
        model.fire(vec![lane]);
    };
    // The lane's own fires, warm: their usual gap.
    let mut usual = Duration::ZERO;
    for _ in 0..8 {
        let t = Instant::now();
        step(&mut model, &mut refs, &mut seq);
        usual = usual.max(t.elapsed());
    }

    let cold = registration(&model);
    let compile = engine::Programs::prepare(&mut model.engine, &cold)
        .expect("the program prepares");
    let began = Instant::now();
    let compiling = std::thread::spawn(move || {
        let prepared = compile();
        (prepared, began.elapsed())
    });
    let mut fires = 0u32;
    let mut widest = Duration::ZERO;
    while !compiling.is_finished() {
        let t = Instant::now();
        step(&mut model, &mut refs, &mut seq);
        widest = widest.max(t.elapsed());
        fires += 1;
    }
    let (prepared, took) = compiling.join().expect("the compile thread");
    let id = engine::Programs::finish(&mut model.engine, prepared)
        .expect("the cold program registers at the finish");
    eprintln!(
        "cold registration: compiled in {:.3} s off the lane; {fires} fires meanwhile, the \
         widest {:.1} ms (warm fires at most {:.1} ms); program {id:?}",
        took.as_secs_f64(),
        widest.as_secs_f64() * 1e3,
        usual.as_secs_f64() * 1e3
    );
    assert!(fires > 0, "the lane fired while the program compiled");
    assert!(
        widest < took / 2,
        "a fire waited {widest:?} of the compile's {took:?}: the compile ran on the fire path"
    );
}
