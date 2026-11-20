//! A program registered in two steps, its compile on another thread
//! (`Engine::prepare_program`, `Engine::finish_program`), is the program a
//! registration whole makes: the same package and facts, its stages'
//! dispatches lowered by the compile. A compile that refuses is refused at the finish,
//! by the program's hash.

use engine::Engine;
use engine::program::ProgramRegistration;

use super::outputs::{Outputs, epilogue_registration, loaded};

/// Prepares on this thread, compiles on another, finishes here.
fn in_two_steps(
    engine: &mut crate::backend::Engine,
    registration: &ProgramRegistration,
) -> engine::Result<engine::program::ProgramId> {
    let compile = engine.prepare_program(registration)?;
    let prepared = std::thread::spawn(compile)
        .join()
        .expect("the compile thread");
    engine.finish_program(prepared)
}

#[test]
fn a_program_prepared_off_thread_is_the_program_registered_whole() {
    let Some(mut loaded) = loaded("register", &Outputs) else {
        return;
    };
    let registration = epilogue_registration(&loaded.caps);
    let whole = loaded
        .engine
        .register_program(&registration)
        .expect("the program registers whole");
    let split = in_two_steps(&mut loaded.engine, &registration)
        .expect("the program registers in two steps");
    assert_ne!(whole, split, "each registration is its own program");

    let programs = &loaded.engine.programs;
    assert_eq!(
        programs.package(whole),
        programs.package(split),
        "the same package"
    );
    let facts = |program| {
        loaded
            .engine
            .program_facts(program)
            .expect("an instance's facts")
    };
    assert_eq!(facts(whole), facts(split), "the same facts");

    // A launch whose plans and stages disagree does not adopt: the compile
    // refuses and the finish says so, by the program's hash.
    let mut broken = registration.clone();
    broken.program_hash = 0xbad5;
    broken.launch.plans.clear();
    let refused = in_two_steps(&mut loaded.engine, &broken)
        .expect_err("a program that does not adopt is refused");
    assert!(
        refused.to_string().contains("0x000000000000bad5"),
        "the refusal names the program: {refused}"
    );
    // And a prepared program another engine's compile made is refused.
    let foreign = engine::Prepared(Box::new(0u8));
    let refused = loaded
        .engine
        .finish_program(foreign)
        .expect_err("a foreign prepared program is refused");
    assert!(
        refused.to_string().contains("did not prepare"),
        "{refused}"
    );
    let _ = std::fs::remove_dir_all(&loaded.dir);
}

/// Rounds of the cold program: enough ops that its lowering takes the lane
/// many fires.
const ROUNDS: u32 = 512;

/// The epilogue with its stacked output run through `rounds` rounds of
/// transcendentals before it is put: code no program registered yet holds,
/// so its pipelines build cold.
fn heavier_epilogue(rounds: u32) -> eta_ir::container::TraceContainer {
    use eta_ir::op::Op;
    let mut program = super::outputs::epilogue();
    let ops = &mut program.stages[0].ops;
    // The stack's put is the last op; each op below adds one value after
    // the value it puts, the epilogue's last.
    let Some(Op::ChanPut { chan: 1, value }) = ops.pop() else {
        panic!("the epilogue ends putting the stack");
    };
    let mut last = value;
    for round in 0..rounds * 4 {
        ops.push(match round % 4 {
            0 => Op::Sin(last),
            1 => Op::Exp(last),
            2 => Op::Cos(last),
            _ => Op::Add(last, 1),
        });
        last += 1;
    }
    ops.push(Op::ChanPut {
        chan: 1,
        value: last,
    });
    program
}

/// A lane fires on while a program's registration compiles cold on another
/// thread (`Engine::prepare_program`): its fires keep landing, none waiting
/// out the compile, and the program registers at the finish.
///
/// The test holds the device alone (`gpu::alone`): a model test beside it
/// (the score oracles' loads and fires) keeps the device busy for tens of
/// milliseconds, and the lane's fence wait across it reads as a stall behind
/// the compile, which is host work alone (`program::adopt`).
///
/// Known to fail on the RTX Spark (Windows NVIDIA driver): a one-time submit stall early in
/// the engine's life lands inside the compile window in about half the runs, with no pie lock
/// involved (`new-memory/FOLLOWUPS.md` F41). The measurement stays strict.
#[test]
fn a_cold_registration_compiles_while_a_lane_fires() {
    use engine::fire::{Attachment, FrameSubmission, InputFeed, Lane, Readout, Step};
    crate::gpu::alone();
    let Some(super::outputs::Bound {
        mut engine,
        instance,
        channels,
        t_input,
        word,
        dir,
        decls,
        ..
    }) = super::outputs::bound("register-fires")
    else {
        return;
    };
    let fire = |engine: &mut crate::backend::Engine| {
        let lane = Lane {
            slot: 0,
            word,
            tokens: vec![3, 11, 7],
            readout: Readout::Last,
            kv_less: true,
            inputs: vec![InputFeed {
                id: t_input,
                channel: channels[2],
            }],
            ..Lane::default()
        };
        let mut ticket = engine
            .submit(&FrameSubmission::of(Step {
                lanes: vec![lane],
                attachments: vec![Attachment {
                    lane: 0,
                    lanes: 1,
                    instance,
                    adapter_only: false,
                    scratch: ::engine::ScratchPlace::None,
                }],
                media: Vec::new(),
                voxels: Vec::new(),
            }))
            .expect("the frame fires");
        engine.settle_frame(&mut ticket).expect("the frame settles");
        for dense in [0, 1] {
            engine
                .take_channel(instance, dense)
                .expect("the channel reads")
                .expect("the channel holds a cell");
        }
        // The timestep `bound` published is an input the fire reads, not
        // takes: it stands for every fire.
    };
    fire(&mut engine);

    let cold = super::outputs::registration_of(&decls, heavier_epilogue(ROUNDS), 0xc01d);
    let compile = engine.prepare_program(&cold).expect("the program prepares");
    let began = std::time::Instant::now();
    let compiling = std::thread::spawn(move || {
        let prepared = compile();
        (prepared, began.elapsed())
    });
    let (mut fires, mut longest, mut last) = (0u32, std::time::Duration::ZERO, began);
    while !compiling.is_finished() {
        fire(&mut engine);
        fires += 1;
        longest = longest.max(last.elapsed());
        last = std::time::Instant::now();
    }
    let (prepared, built) = compiling.join().expect("the compile finishes");
    let finishing = std::time::Instant::now();
    let program = engine
        .finish_program(prepared)
        .expect("the program registers");
    let finished = finishing.elapsed();
    eprintln!(
        "{}: compiled in {built:?} off the lane, finished in {finished:?}; the lane fired \
         {fires} times meanwhile, at most {longest:?} apart",
        crate::backend::NAME
    );
    assert!(
        engine.programs.package(program).is_some(),
        "the program is registered"
    );
    assert!(
        fires >= 3 && longest < built / 2,
        "the lane stalled behind the compile: {fires} fires in {built:?}, at most {longest:?} apart"
    );
    fire(&mut engine);
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The load's prewarm of a program (`Engine::prewarm_programs`) adopts it as
/// its registration does and holds every step call its stages dispatch to a
/// pipeline the load built: a registration after it builds nothing. What
/// the prewarm lowers is keyed on the stages' code, not the program's hash:
/// the program at another channel capacity (the runtime's frame size sets
/// it) has another hash and lowers to the same dispatches. The negative
/// control: a call the load's pipeline was taken out of refuses the
/// prewarm, by the call.
#[test]
fn a_prewarm_holds_a_programs_calls_to_the_loads_pipelines() {
    use engine::live::Tally;
    use engine_gpu::program::Programs as _;
    let Some(mut loaded) = loaded("prewarm", &Outputs) else {
        return;
    };
    let decls = super::outputs::output_decls(&loaded.caps);
    let registration = epilogue_registration(&loaded.caps);
    let mut wider = super::outputs::epilogue();
    for channel in &mut wider.channels {
        channel.capacity = 5;
    }
    let wider = super::outputs::registration_of(&decls, wider, 0x0e8);
    assert_ne!(
        registration.launch, wider.launch,
        "the capacity is in the program"
    );
    let narrow = engine_gpu::program::prewarm(std::slice::from_ref(&registration))
        .expect("the program adopts");
    let wide =
        engine_gpu::program::prewarm(std::slice::from_ref(&wider)).expect("the program adopts");
    assert_ne!(narrow.hashes, wide.hashes, "another hash");
    assert_eq!(narrow.forms, wide.forms, "the same dispatches");
    assert_eq!(narrow.calls, wide.calls, "the same pipelines");

    loaded
        .engine
        .prewarm_programs(&[registration.clone(), wider.clone()])
        .expect("the prewarm finds every call built");
    let live = loaded.engine.live();
    for program in [&registration, &wider] {
        loaded
            .engine
            .register_program(program)
            .expect("the program registers");
    }
    assert_eq!(live.tallied(Tally::BuiltAtRegister), 0, "a registration built nothing");
    assert_eq!(live.tallied(Tally::BuiltInFire), 0);

    let call = *narrow.calls.first().expect("the program dispatches a call");
    loaded.engine.programs.forget_step(call);
    let refused = loaded
        .engine
        .prewarm_programs(std::slice::from_ref(&registration))
        .expect_err("a call with no pipeline from the load refuses the prewarm");
    assert!(
        refused.to_string().contains(&format!("guest step call {call}")),
        "the refusal names the call: {refused}"
    );
    let _ = std::fs::remove_dir_all(&loaded.dir);
}
