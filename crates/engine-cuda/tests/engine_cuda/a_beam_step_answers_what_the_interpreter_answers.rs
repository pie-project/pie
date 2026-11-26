//! The guest-program contract (`eta_compiler::contract`) on this engine: each
//! case's program registered, bound with its seeds, its logits seated as the
//! case states (one lane's seat, or a lane a row), fired, and every channel the host reads exactly what the interpreter puts there.

use engine::{ChannelId, ProgramRegistration};
use engine_cuda::device::{Buffer, Context};
use engine_cuda::program::launch::INTRINSIC_STORAGE_F32;
use engine_cuda::program::session::Fired;
use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::contract::{Accuracy, Binding, Case, Seat, beam_stage, split_stage};
use eta_compiler::plan::compile_bound;
use eta_exec::Extents;
use eta_ir::registry::GeometryClass;

/// What the device puts in each of `case`'s readers, its logits seated at
/// `seats`, or why the fire was refused.
fn fired(case: &Case, seats: &[Seat]) -> Result<Vec<Vec<u8>>, String> {
    fired_over(case, seats, &crate::guest::alone_span)
}

/// A guest item in a buffer of its own, every byte 0x7f: an f32 or bf16
/// cell of it reads 3.4e38, above any log-probability.
fn poisoned_span(_: engine_cuda::program::Item<'_>, bytes: u64) -> engine_cuda::Result<u64> {
    let mut buffer = Buffer::zeroed(bytes as usize)?;
    buffer.write(0, &vec![0x7f; bytes as usize])?;
    Ok(Box::leak(Box::new(buffer)).ptr())
}

/// [`fired`], the instance's items laid by `span`.
fn fired_over(
    case: &Case,
    seats: &[Seat],
    span: &dyn Fn(engine_cuda::program::Item<'_>, u64) -> engine_cuda::Result<u64>,
) -> Result<Vec<Vec<u8>>, String> {
    let context = Context::bind(0, None).expect("a device context");
    let stages = compile_bound(&case.bound);
    let backend = Backend::parse(engine_cuda::api::CODEGEN).expect("the engine's codegen");
    let registration = ProgramRegistration {
        program_hash: 0xbea0_0000 ^ u64::from(case.rows),
        emitted_kernels: emit_program(backend, &stages, &case.bound),
        emitter_version: backend.emitter_version(),
        region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(&stages),
        launch: eta_compiler::codegen::launch::build(&case.bound, &stages),
        ..Default::default()
    };
    let mut plane = crate::guest::alone_plane();
    let program = plane
        .register(&context, &registration)
        .expect("the program compiles for the device");
    let extents = Extents {
        row_count: case.rows,
        sampled_rows: case.rows,
        query_len: case.rows,
        ..Extents::default()
    };
    let count = case.bound.container.channels.len();
    let ids: Vec<ChannelId> = (1..=count as u64).map(ChannelId).collect();
    let instance = plane
        .bind(
            program,
            &case.wire_seeds(),
            extents,
            GeometryClass::Host,
            eta_ir::registry::PortMask::NONE,
            false,
            &vec![None; count],
            &ids,
            crate::guest::alone_slots(&plane, program),
            span,
            context.stream(),
        )
        .expect("the instance binds");
    let laid = case.buffer();
    let mut logits = Buffer::zeroed(laid.len() * 4).expect("a logits plane");
    let bytes: Vec<u8> = laid.iter().flat_map(|x| x.to_le_bytes()).collect();
    logits.write(0, &bytes).expect("the logits upload");
    for seat in seats {
        plane
            .bind_output(
                instance,
                "logits",
                seat.lane,
                logits.ptr() + seat.offset,
                INTRINSIC_STORAGE_F32,
                seat.width,
                seat.row_stride,
                seat.row_offset,
                match case.binding {
                    Binding::OneLane => case.rows,
                    Binding::LaneARow => 1,
                },
            )
            .expect("the logits bind");
    }
    let fired = plane
        .fire(&context, instance)
        .map_err(|why| why.to_string())?;
    assert!(
        matches!(fired, Fired::Committed),
        "{}: {fired:?}",
        case.name
    );
    let session = plane.instance_mut(instance).expect("the instance");
    Ok(case
        .readers
        .iter()
        .map(|&channel| {
            session
                .take(channel)
                .expect("a channel reads")
                .expect("the channel holds a cell")
        })
        .collect())
}

#[test]
fn a_beam_step_answers_what_the_interpreter_answers() {
    if !engine_cuda::device::present() {
        eprintln!("no device: skipping");
        return;
    }
    for case in beam_stage() {
        let want = case.expected();
        let got = fired(&case, &case.seats()).unwrap_or_else(|why| panic!("{}: {why}", case.name));
        for ((channel, got), want) in case.readers.iter().zip(&got).zip(&want) {
            assert_eq!(got, want, "{}: channel {channel}", case.name);
        }
    }
    // A seat no row of the program reaches is refused by name, not dropped:
    // lanes 0 and 3 of an output read at lane 0 are no rows 0 and 1.
    let case = &beam_stage()[0];
    let mut seats = case.seats();
    seats.push(Seat {
        lane: 3,
        ..seats[0]
    });
    let refused = fired(case, &seats).expect_err("a seat past the rows is refused");
    assert!(refused.contains("bound at lane 3"), "{refused}");
}

/// The split cases (`contract::split_stage`): a one-row region over the
/// model's row run across the lane's blocks, at a width of whole stretches
/// and one with a ragged last stretch, answers what the interpreter answers.
/// The candidates' ids byte for byte and their values within rounding; the
/// instance's scratch filled with a value above every log-probability, so
/// an element a block left unwritten, or read before another block wrote
/// it, lands among the top.
#[test]
fn a_split_row_answers_what_the_interpreter_answers() {
    if !engine_cuda::device::present() {
        eprintln!("no device: skipping");
        return;
    }
    // CUDA's `expf` and `logf` (Programming Guide, Table 13): 2 and 1 ulp.
    let cuda = Accuracy {
        exp_ulps: 2,
        log_ulps: 1,
    };
    for case in split_stage() {
        let want = case.expected();
        let got = fired_over(&case, &case.seats(), &poisoned_span)
            .unwrap_or_else(|why| panic!("{}: {why}", case.name));
        for ((&channel, got), want) in case.readers.iter().zip(&got).zip(&want) {
            case.agrees(channel, got, want, cuda, 1)
                .unwrap_or_else(|why| panic!("{why}"));
        }
    }
}

/// Binds `case` in `plane`, its logits seated as the case states: the
/// instance, and the plane its logits lie in (held as long as it fires).
fn bind_case(
    plane: &mut engine_cuda::ProgramPlane,
    context: &Context,
    case: &Case,
    hash: u64,
) -> (engine::InstanceId, Buffer) {
    let stages = compile_bound(&case.bound);
    let backend = Backend::parse(engine_cuda::api::CODEGEN).expect("the engine's codegen");
    let registration = ProgramRegistration {
        program_hash: hash,
        emitted_kernels: emit_program(backend, &stages, &case.bound),
        emitter_version: backend.emitter_version(),
        region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(&stages),
        launch: eta_compiler::codegen::launch::build(&case.bound, &stages),
        ..Default::default()
    };
    let program = plane
        .register(context, &registration)
        .expect("the program compiles for the device");
    let extents = Extents {
        row_count: case.rows,
        sampled_rows: case.rows,
        query_len: case.rows,
        ..Extents::default()
    };
    let count = case.bound.container.channels.len();
    let ids: Vec<ChannelId> = (1..=count as u64).map(ChannelId).collect();
    let instance = plane
        .bind(
            program,
            &case.wire_seeds(),
            extents,
            GeometryClass::Host,
            eta_ir::registry::PortMask::NONE,
            false,
            &vec![None; count],
            &ids,
            crate::guest::alone_slots(plane, program),
            &crate::guest::alone_span,
            context.stream(),
        )
        .expect("the instance binds");
    let laid = case.buffer();
    let mut logits = Buffer::zeroed(laid.len() * 4).expect("a logits plane");
    let bytes: Vec<u8> = laid.iter().flat_map(|x| x.to_le_bytes()).collect();
    logits.write(0, &bytes).expect("the logits upload");
    for seat in case.seats() {
        plane
            .bind_output(
                instance,
                "logits",
                seat.lane,
                logits.ptr() + seat.offset,
                INTRINSIC_STORAGE_F32,
                seat.width,
                seat.row_stride,
                seat.row_offset,
                match case.binding {
                    Binding::OneLane => case.rows,
                    Binding::LaneARow => 1,
                },
            )
            .expect("the logits bind");
    }
    (instance, logits)
}

/// Eight programs' beam steps in one wave, each a group of its own, wave
/// after wave: every split select and split row meets in the engine's lane
/// arenas, one buffer every group's lanes lay at the same offsets, so the
/// groups run one after another (none on a side stream beside another) and
/// each launch's counters start from zero. Each answers what the
/// interpreter answers, every wave.
#[test]
fn groups_splitting_in_one_wave_answer_as_each_alone() {
    use eta_ir::registry::Stage;
    if !engine_cuda::device::present() {
        eprintln!("no device: skipping");
        return;
    }
    let context = Context::bind(0, None).expect("a device context");
    // Eight programs (the one-lane steps, each registered under hashes of
    // its own, as beams of distinct prompts and lengths are): eight groups,
    // which the wave fans over its side streams unless they meet.
    let cases: Vec<Case> = (0..4)
        .flat_map(|_| {
            beam_stage()
                .into_iter()
                .filter(|case| case.binding == Binding::OneLane)
        })
        .collect();
    assert!(cases.len() >= 8, "eight programs to group apart");
    let mut plane = crate::guest::alone_plane();
    for wave in 0..6u64 {
        let bound: Vec<(engine::InstanceId, Buffer)> = cases
            .iter()
            .enumerate()
            .map(|(at, case)| bind_case(&mut plane, &context, case, 0xbe90_0000 + at as u64))
            .collect();
        for (instance, _) in &bound {
            plane
                .stage(*instance, Stage::Epilogue)
                .expect("the step stages");
        }
        plane.fly(&context).expect("the wave flies");
        plane.land(&context).expect("the wave lands");
        context.synchronize().expect("the wave settles");
        for (case, (instance, _)) in cases.iter().zip(&bound) {
            let fired = plane.settle_launched(*instance).expect("the step settles");
            assert!(
                matches!(fired, Fired::Committed),
                "{}, wave {wave}: {fired:?}",
                case.name
            );
            let session = plane.instance_mut(*instance).expect("the instance");
            for (&channel, want) in case.readers.iter().zip(case.expected()) {
                let got = session
                    .take(channel)
                    .expect("a channel reads")
                    .expect("the channel holds a cell");
                assert_eq!(got, want, "{}, wave {wave}: channel {channel}", case.name);
            }
        }
    }
}

const SAMPLER_VOCAB: u32 = 151_936;

/// A decode's sampler as the runtime builds it (a temperature, a softmax, a
/// top-p cut, Gumbel noise, an argmax): the lane sampler, whose blocks meet
/// in the engine's lane arenas from their base, its barrier words zeroed
/// before each launch.
fn sampler_step() -> eta_ir::container::TraceContainer {
    use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
    use eta_ir::op::Op;
    use eta_ir::registry::Stage;
    use eta_ir::types::{Dtype, Literal, Predicate, RngKind, Shape};
    let row = Shape::vector(SAMPLER_VOCAB);
    let ops = vec![
        Op::ChanRead(0),
        Op::ChanRead(1),
        Op::ChanRead(2),
        Op::Const(Literal::F32(0.8)),
        Op::Div(2, 3),
        Op::ReduceMax(4),
        Op::Broadcast {
            value: 5,
            shape: row,
        },
        Op::Sub(4, 6),
        Op::Exp(7),
        Op::ReduceSum(8),
        Op::Broadcast {
            value: 9,
            shape: row,
        },
        Op::Div(8, 10),
        Op::PivotThreshold {
            input: 11,
            predicate: Predicate::CummassLe(1),
        },
        Op::Const(Literal::F32(f32::NEG_INFINITY)),
        Op::Select {
            cond: 12,
            a: 4,
            b: 13,
        },
        Op::RngKeyed {
            state: 0,
            shape: row,
            kind: RngKind::Gumbel,
        },
        Op::Add(14, 15),
        Op::ReduceArgmax(16),
        Op::ChanPut { chan: 3, value: 17 },
    ];
    let seeded = |shape, dtype| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 1,
        host_role: HostRole::Writer,
        seeded: true,
    };
    TraceContainer {
        names: Vec::new(),
        channels: vec![
            seeded(Shape::vector(2), Dtype::U32),
            seeded(Shape::new(&[]).unwrap(), Dtype::F32),
            seeded(row, Dtype::F32),
            ChannelDecl {
                shape: Shape::new(&[]).unwrap(),
                dtype: ChanDType::Concrete(Dtype::I32),
                capacity: 2,
                host_role: HostRole::Reader,
                seeded: false,
            },
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

/// A beam step's split select and decodes' lane samplers in one wave, wave
/// after wave: both meet in the engine's lane arenas from their base (the
/// select's finished-blocks count and candidates, the samplers' barrier
/// words, zeroed before their launch, and exchange buffers), so the two
/// groups must not run side by side on two streams. Every beam answers what
/// the interpreter answers and every decode draws the interpreter's token.
#[test]
fn a_split_select_beside_lane_samplers_answers_as_alone() {
    use eta_compiler::eval::interp::{Instance, NoKernels, PassInputs, Value};
    use eta_ir::registry::Stage;
    if !engine_cuda::device::present() {
        eprintln!("no device: skipping");
        return;
    }
    let context = Context::bind(0, None).expect("a device context");
    let beam = beam_stage()
        .into_iter()
        .find(|case| case.binding == Binding::OneLane && case.rows == 4)
        .expect("the w4 beam step");
    let sampler = crate::bound::logits(sampler_step(), SAMPLER_VOCAB).expect("the sampler binds");
    let stages = compile_bound(&sampler);
    let backend = Backend::parse(engine_cuda::api::CODEGEN).expect("the engine's codegen");
    let mut plane = crate::guest::alone_plane();
    let sampler_program = plane
        .register(
            &context,
            &ProgramRegistration {
                program_hash: sampler.hash,
                emitted_kernels: emit_program(backend, &stages, &sampler),
                emitter_version: backend.emitter_version(),
                region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(
                    &stages,
                ),
                launch: eta_compiler::codegen::launch::build(&sampler, &stages),
                ..Default::default()
            },
        )
        .expect("the sampler compiles");
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut unit = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 / (1u64 << 53) as f64) as f32
    };
    let logits: Vec<f32> = (0..SAMPLER_VOCAB).map(|_| unit() * 16.0 - 8.0).collect();
    let seeds = vec![
        (0u32, Value::U32(vec![7, 11])),
        (1u32, Value::F32(vec![0.9])),
        (2u32, Value::F32(logits)),
    ];
    let want_token = {
        let mut reference = Instance::new(&sampler, &seeds).expect("the interpreter instantiates");
        let stepped = reference
            .step(&sampler, &PassInputs::default(), &mut NoKernels)
            .expect("the interpreter steps");
        assert!(stepped.committed);
        match reference.host_take(&sampler, 3).unwrap() {
            Value::I32(v) => v[0],
            other => panic!("expected i32, got {other:?}"),
        }
    };
    let wire: Vec<(u32, Vec<u8>)> = seeds.iter().map(|(c, v)| (*c, v.to_le_bytes())).collect();
    let count = sampler.container.channels.len();
    let ids: Vec<ChannelId> = (1..=count as u64).map(ChannelId).collect();
    let (mut beams_wrong, mut draws_wrong) = (0usize, 0usize);
    for wave in 0..12u64 {
        let (beam_instance, _held) = bind_case(&mut plane, &context, &beam, 0xbe91_0000 + wave);
        let decodes: Vec<engine::InstanceId> = (0..8)
            .map(|_| {
                plane
                    .bind(
                        sampler_program,
                        &wire,
                        Extents {
                            row_count: 1,
                            sampled_rows: 1,
                            query_len: 1,
                            ..Extents::default()
                        },
                        GeometryClass::Host,
                        eta_ir::registry::PortMask::NONE,
                        false,
                        &vec![None; count],
                        &ids,
                        crate::guest::alone_slots(&plane, sampler_program),
                        &crate::guest::alone_span,
                        context.stream(),
                    )
                    .expect("a decode binds")
            })
            .collect();
        for instance in decodes.iter().chain([&beam_instance]) {
            plane
                .stage(*instance, Stage::Epilogue)
                .expect("the step stages");
        }
        plane.fly(&context).expect("the wave flies");
        plane.land(&context).expect("the wave lands");
        context.synchronize().expect("the wave settles");
        let fired = plane
            .settle_launched(beam_instance)
            .expect("the beam settles");
        let session = plane.instance_mut(beam_instance).expect("the beam");
        let agrees = matches!(fired, Fired::Committed)
            && beam
                .readers
                .iter()
                .zip(beam.expected())
                .all(|(&channel, want)| {
                    session
                        .take(channel)
                        .ok()
                        .flatten()
                        .is_some_and(|got| got == want)
                });
        beams_wrong += usize::from(!agrees);
        for instance in decodes {
            let fired = plane.settle_launched(instance).expect("the decode settles");
            let token = plane
                .instance_mut(instance)
                .expect("the decode")
                .take(3)
                .ok()
                .flatten()
                .map(|cell| i32::from_le_bytes(cell[..4].try_into().unwrap()));
            draws_wrong +=
                usize::from(!matches!(fired, Fired::Committed) || token != Some(want_token));
        }
    }
    assert_eq!(
        (beams_wrong, draws_wrong),
        (0, 0),
        "beams and draws off the interpreter's over 12 waves, a split select beside lane samplers"
    );
}
