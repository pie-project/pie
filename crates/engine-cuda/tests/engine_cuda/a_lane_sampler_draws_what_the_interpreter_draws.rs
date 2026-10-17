//! The decode sampler as compat builds it — logits over a temperature, a
//! softmax, a top-p (`CummassLe`) or top-k (`RankLe`) cut, the cut logits
//! masked, Gumbel noise, an argmax — which the compiler hands the lane
//! sampler (a lane's row split over several blocks), drawing the token the
//! host interpreter draws, over rows a trained model's and a random mini's
//! next-token distributions span and bf16 logits' runs of equal values.
//! `PIE_LANE_SAMPLER_BLOCKS` caps the blocks a lane takes (the split must not
//! move a draw).

use engine::ProgramRegistration;
use engine_cuda::device::Context;
use engine_cuda::program::Plane;
use engine_cuda::program::compile::Disk;
use engine_cuda::program::session::Fired;
use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::eval::interp::{Instance, NoKernels, PassInputs, Value};
use eta_compiler::plan::{LibraryOp, RegionKind, compile_bound};
use eta_exec::Extents;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{GeometryClass, ModelProfile, Stage};
use eta_ir::types::{Dtype, Literal, Predicate, RngKind, Shape};
use eta_ir::validate::{BoundTrace, bind};

const VOCAB: u32 = 151_936;

/// Fires of each draw (its seed, and so its token, the same each time).
const REPEATS: usize = 4;

const STATE: u32 = 0;
const KNOB: u32 = 1;
const LOGITS: u32 = 2;
const TOKEN: u32 = 3;

fn sampler(rank: bool) -> TraceContainer {
    let row = Shape::vector(VOCAB);
    let ops = vec![
        Op::ChanRead(STATE),
        Op::ChanRead(KNOB),
        Op::ChanRead(LOGITS),
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
            predicate: if rank {
                Predicate::RankLe(1)
            } else {
                Predicate::CummassLe(1)
            },
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
        Op::ChanPut {
            chan: TOKEN,
            value: 17,
        },
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
            seeded(
                Shape::new(&[]).unwrap(),
                if rank { Dtype::I32 } else { Dtype::F32 },
            ),
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
    }
}

struct Noise(u64);

impl Noise {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) as f32
    }
    fn gaussian(&mut self) -> f32 {
        let (u, v) = (self.unit().max(1e-12), self.unit());
        (-2.0 * u.ln()).sqrt() * (core::f32::consts::TAU * v).cos()
    }
}

fn bf16(x: f32) -> f32 {
    f32::from_bits((x.to_bits() + 0x8000) & 0xffff_0000)
}

fn host_token(bound: &BoundTrace, seeds: &[(u32, Value)]) -> i32 {
    let mut reference = Instance::new(bound, seeds).expect("the interpreter instantiates");
    let stepped = reference
        .step(bound, &PassInputs::default(), &mut NoKernels)
        .expect("the interpreter steps");
    assert!(stepped.committed, "the reference did not commit");
    match reference.host_take(bound, TOKEN).unwrap() {
        Value::I32(v) => v[0],
        other => panic!("expected i32, got {other:?}"),
    }
}

fn draw_all(rank: bool, cases: &[(&str, Vec<f32>, f32, [u32; 2])]) -> (usize, usize) {
    let bound = bind(
        sampler(rank),
        ModelProfile {
            vocab: VOCAB,
            ..ModelProfile::dummy()
        },
    )
    .expect("the sampler binds");
    let stages = compile_bound(&bound);
    let lane = stages.iter().any(|stage| {
        stage.fused.regions.iter().any(|region| {
            matches!(
                region.kind,
                RegionKind::Library(LibraryOp::NucleusSample | LibraryOp::RankSample)
            )
        })
    });
    assert!(lane, "the sampler chain is not handed the lane sampler");
    let context = Context::bind(0, None).expect("a device context");
    let backend = Backend::parse(engine_cuda::api::CODEGEN).expect("the engine's codegen backend");
    let registration = ProgramRegistration {
        program_hash: bound.hash,
        emitted_kernels: emit_program(backend, &stages, &bound),
        emitter_version: backend.emitter_version(),
        region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(&stages),
        launch: eta_compiler::codegen::launch::build(&bound, &stages),
        ..Default::default()
    };
    let mut plane = Plane::new(Disk::disabled());
    let program = plane
        .register(&context, &registration)
        .expect("the sampler compiles");
    let count = bound.container.channels.len();
    let ids: Vec<u64> = (1..=count as u64).collect();
    let (mut agree, mut total) = (0, 0);
    for (what, logits, knob, state) in cases {
        let knob = if rank {
            Value::I32(vec![*knob as i32])
        } else {
            Value::F32(vec![*knob])
        };
        let seeds = vec![
            (STATE, Value::U32(state.to_vec())),
            (KNOB, knob.clone()),
            (LOGITS, Value::F32(logits.clone())),
        ];
        let want = host_token(&bound, &seeds);
        let wire: Vec<(u32, Vec<u8>)> = seeds.iter().map(|(c, v)| (*c, v.to_le_bytes())).collect();
        let instance = plane
            .bind(
                program,
                &wire,
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
        // The same draw, fired again and again, lands on one token.
        let mut drawn = Vec::new();
        for _ in 0..REPEATS {
            let fired = plane.fire(&context, instance).expect("the fire runs");
            assert!(matches!(fired, Fired::Committed), "{fired:?}");
            let cell = plane
                .instance_mut(instance)
                .expect("the instance")
                .take(TOKEN)
                .expect("the token reads")
                .expect("a token");
            drawn.push(i32::from_le_bytes(cell[..4].try_into().unwrap()));
        }
        let got = drawn[0];
        assert!(
            drawn.iter().all(|&t| t == got),
            "{what}: the same draw landed on {drawn:?}"
        );
        total += 1;
        if got == want {
            agree += 1;
        } else {
            eprintln!(
                "{what} (knob {knob:?}, state {state:?}): the device drew {got}, the interpreter {want}"
            );
        }
    }
    (agree, total)
}

fn cases(rank: bool) -> Vec<(&'static str, Vec<f32>, f32, [u32; 2])> {
    let mut noise = Noise(0x5eed_1234_9876_abcd);
    let len = VOCAB as usize;
    let flat: Vec<f32> = (0..len).map(|_| noise.unit() * 2.0 - 1.0).collect();
    let trained: Vec<f32> = (0..len).map(|_| noise.gaussian() * 3.5).collect();
    let peaked: Vec<f32> = (0..len).map(|_| noise.gaussian() * 5.0).collect();
    let tied: Vec<f32> = (0..len).map(|_| bf16(noise.gaussian() * 3.0)).collect();
    let knobs: &[f32] = if rank {
        &[1.0, 40.0, 1000.0, 151_936.0]
    } else {
        &[0.9, 0.5, 0.99, 1.0]
    };
    let mut out = Vec::new();
    for (what, row) in [
        ("flat", &flat),
        ("trained", &trained),
        ("peaked", &peaked),
        ("bf16 ties", &tied),
    ] {
        for (k, &knob) in knobs.iter().enumerate() {
            for draw in 0..3u32 {
                out.push((
                    what,
                    row.clone(),
                    knob,
                    [7 + draw * 13 + k as u32, 11 + draw],
                ));
            }
        }
    }
    out
}

#[test]
fn a_lane_sampler_draws_what_the_interpreter_draws() {
    if !engine_cuda::device::present() {
        eprintln!("no device: skipping");
        return;
    }
    for rank in [false, true] {
        let (agree, total) = draw_all(rank, &cases(rank));
        eprintln!(
            "{}: {agree} of {total} draws as the interpreter",
            if rank { "top-k" } else { "top-p" }
        );
        assert_eq!(
            agree, total,
            "the lane sampler drew other tokens than the interpreter"
        );
    }
}
