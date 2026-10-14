//! The decode sampler as compat builds it — logits over a temperature, a
//! softmax, a top-k / top-p cut, the cut logits masked, a Gumbel argmax —
//! timed one variant at a time (`cargo test ... a_sampler_epilogue_times --
//! --ignored --nocapture`, under a kernel trace to read each program's
//! kernel; each variant prints its program hash, the kernel's name).

use engine::ProgramRegistration;
use engine_rocm::device::{Buffer, Context};
use engine_rocm::program::Plane;
use engine_rocm::program::compile::Disk;
use engine_rocm::program::launch::INTRINSIC_STORAGE_F32;
use engine_rocm::program::session::Fired;
use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::eval::interp::Value;
use eta_compiler::plan::compile_bound;
use eta_exec::Extents;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::{IntrinsicId, Op};
use eta_ir::registry::{GeometryClass, ModelProfile, Stage};
use eta_ir::types::{Dtype, Literal, Predicate, RngKind, Shape};
use eta_ir::validate::bind;

const VOCAB: u32 = 151_936;

const STATE: u32 = 0;
const KNOB: u32 = 1;
const TOKEN: u32 = 2;

#[derive(Clone, Copy, Debug)]
enum Cut {
    None,
    Floor,
    Rank,
    Mass,
}

fn sampler(cut: Cut) -> TraceContainer {
    let row = Shape::vector(VOCAB);
    let mut ops = vec![
        Op::ChanRead(STATE),
        Op::ChanRead(KNOB),
        Op::IntrinsicVal {
            intr: IntrinsicId::Logits,
            shape: Shape::matrix(1, VOCAB),
            dtype: Dtype::F32,
        },
        Op::Reshape {
            value: 2,
            shape: row,
        },
        Op::Const(Literal::F32(0.8)),
        Op::Div(3, 4),
    ];
    let mut truncated = 5;
    if !matches!(cut, Cut::None) {
        let predicate = match cut {
            Cut::Rank => Predicate::RankLe(1),
            Cut::Mass => Predicate::CummassLe(1),
            _ => Predicate::ProbGe(1),
        };
        ops.extend([
            Op::ReduceMax(5),
            Op::Broadcast {
                value: 6,
                shape: row,
            },
            Op::Sub(5, 7),
            Op::Exp(8),
            Op::ReduceSum(9),
            Op::Broadcast {
                value: 10,
                shape: row,
            },
            Op::Div(9, 11),
            Op::PivotThreshold {
                input: 12,
                predicate,
            },
            Op::Const(Literal::F32(f32::NEG_INFINITY)),
            Op::Select {
                cond: 13,
                a: 5,
                b: 14,
            },
        ]);
        truncated = 15;
    }
    let noise = ops.len() as u32;
    ops.extend([
        Op::RngKeyed {
            state: 0,
            shape: row,
            kind: RngKind::Gumbel,
        },
        Op::Add(truncated, noise),
        Op::ReduceArgmax(noise + 1),
        Op::ChanPut {
            chan: TOKEN,
            value: noise + 2,
        },
    ]);
    let knob = match cut {
        Cut::Rank => Dtype::I32,
        _ => Dtype::F32,
    };
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
            seeded(Shape::new(&[]).unwrap(), knob),
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
    }
}

fn time(cut: Cut, fires: u32, logits: &[f32]) {
    let profile = ModelProfile {
        vocab: VOCAB,
        ..ModelProfile::dummy()
    };
    let bound = bind(sampler(cut), profile).expect("the sampler binds");
    let context = Context::bind(0, None).expect("a device context");
    let stages = compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let backend = Backend::parse("hip").expect("the hip backend");
    if let Ok(dir) = std::env::var("CUT_DUMP") {
        for kernel in emit_program(backend, &stages, &bound) {
            let path = format!("{dir}/{cut:?}-{}.hip", kernel.entry_name);
            std::fs::write(&path, &kernel.source).expect("the dump writes");
        }
    }
    let registration = ProgramRegistration {
        program_hash: bound.hash,
        emitted_kernels: emit_program(backend, &stages, &bound),
        emitter_version: backend.emitter_version(),
        region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(&stages),
        launch,
        ..Default::default()
    };
    let mut plane = Plane::new(Disk::disabled());
    let program = plane
        .register(&context, &registration)
        .expect("the sampler compiles");
    let knob = match cut {
        Cut::Rank => Value::I32(vec![40]),
        Cut::Mass => Value::F32(vec![0.9]),
        _ => Value::F32(vec![1e-6]),
    };
    let seeds = vec![
        (STATE, Value::U32(vec![7, 11]).to_le_bytes()),
        (KNOB, knob.to_le_bytes()),
    ];
    let count = bound.container.channels.len();
    let ids: Vec<u64> = (1..=count as u64).collect();
    let instance = plane
        .bind(
            program,
            &seeds,
            Extents {
                row_count: 1,
                sampled_rows: 1,
                query_len: 1,
                ..Extents::default()
            },
            GeometryClass::Host,
            &vec![None; count],
            &ids,
            context.stream(),
        )
        .expect("the sampler binds an instance");
    let mut buffer = Buffer::zeroed(logits.len() * 4).expect("a logits plane");
    let bytes: Vec<u8> = logits.iter().flat_map(|x| x.to_le_bytes()).collect();
    buffer.write(0, &bytes).expect("the plane uploads");
    plane
        .bind_intrinsic(
            instance,
            IntrinsicId::Logits,
            buffer.ptr(),
            INTRINSIC_STORAGE_F32,
            VOCAB,
            VOCAB,
            0,
        )
        .expect("the logits bind");
    let start = std::time::Instant::now();
    for _ in 0..=fires {
        let fired = plane.fire(&context, instance).expect("the fire runs");
        assert!(matches!(fired, Fired::Committed), "{fired:?}");
        plane
            .instance_mut(instance)
            .expect("the instance")
            .take(TOKEN)
            .expect("the token reads")
            .expect("a token");
    }
    eprintln!(
        "sampler {cut:?}: program {:016x}, {:.1} us a fire (wall)",
        bound.hash,
        start.elapsed().as_secs_f64() * 1e6 / f64::from(fires + 1)
    );
}

#[test]
#[ignore = "a timing, not a check"]
fn a_sampler_epilogue_times() {
    if !engine_rocm::device::present() {
        return;
    }
    let mut state = 0x2545_f491_4f6c_dd1du64;
    // Random-weight logits: a near-flat row, as a mini model's are.
    let logits: Vec<f32> = (0..VOCAB)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
        })
        .collect();
    let fires = std::env::var("CUT_FIRES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    use core::sync::atomic::Ordering;
    let streams = &eta_compiler::codegen::cuda::fused::SINGLE_ROW_STREAMS;
    // SAMPLER_STREAMS=0 times the op-by-op emission a single-row region had.
    let on = std::env::var("SAMPLER_STREAMS").map_or(true, |v| v != "0");
    streams.store(on, Ordering::Relaxed);
    eprintln!("single-row streams {}", if on { "on" } else { "off" });
    for cut in [Cut::None, Cut::Floor, Cut::Rank, Cut::Mass] {
        time(cut, fires, &logits);
    }
    streams.store(true, Ordering::Relaxed);
}
