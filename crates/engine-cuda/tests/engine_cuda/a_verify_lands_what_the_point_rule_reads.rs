//! A sampling verify's epilogue (`inferlet::speculative::point_rows`, as
//! compat traces it above temperature zero): each row's chance of its
//! proposal, and a Gumbel draw from the row with the proposal struck out,
//! the last row's struck of nothing. The device lands the chances the host
//! interpreter computes and draws the tokens it draws, over a vocabulary a
//! trained model's rows span.

use engine::{ChannelId, ProgramRegistration};
use engine_cuda::device::Context;
use engine_cuda::program::Plane;
use engine_cuda::program::compile::Disk;
use engine_cuda::program::session::Fired;
use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::eval::interp::{Instance, NoKernels, PassInputs, Value};
use eta_compiler::plan::compile_bound;
use eta_dsl::Traced;
use eta_dsl::prelude::*;
use eta_exec::Extents;
use eta_ir::registry::{GeometryClass, ModelProfile};
use eta_ir::validate::bind;

const ROWS: u32 = 5;
const VOCAB: u32 = 151_936;
const TEMPERATURE: f32 = 0.8;

fn leak(channel: Channel) -> &'static Channel {
    Box::leak(Box::new(channel))
}

fn traced(logits: Vec<f32>, proposals: Vec<u32>) -> Traced {
    let logits = leak(Channel::from_shaped([ROWS, VOCAB], logits).named("logits"));
    let proposals = leak(Channel::from(proposals).named("proposals"));
    let noise = leak(Channel::from([0u32, 0]).named("noise"));
    let chances = leak(Channel::new([ROWS], dtype::f32).named("chances"));
    let draws = leak(Channel::new([ROWS], dtype::i32).named("draws"));
    let mut b = Builder::new(VOCAB, 16);
    b.stage(Stage::Epilogue, move || {
        let scaled = logits.read() / TEMPERATURE;
        let (c, d) = inferlet::speculative::point_rows(
            &scaled,
            &proposals.read(),
            ROWS,
            VOCAB,
            &noise.read(),
        );
        chances.put(c);
        draws.put(d);
    });
    chances.note_host_take();
    draws.note_host_take();
    b.build().expect("the verify epilogue traces")
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

#[test]
fn a_verify_lands_what_the_point_rule_reads() {
    if !engine_cuda::device::present() {
        eprintln!("no device: skipping");
        return;
    }
    let mut noise = Noise(0x0bad_cafe_1234_5678);
    let mut logits: Vec<f32> = (0..ROWS * VOCAB).map(|_| noise.gaussian() * 3.5).collect();
    // Each proposal is a token its row likes, as a drafter's would be.
    let proposals: Vec<u32> = (0..ROWS).map(|r| 1_000 + 7_919 * r).collect();
    for (r, &x) in proposals.iter().enumerate() {
        logits[r * VOCAB as usize + x as usize] = 14.0 + r as f32;
    }
    let traced = traced(logits.clone(), proposals.clone());
    let names = traced.channel_names();
    let at = |name: &str| names.iter().position(|n| n == name).unwrap() as u32;
    let bound = bind(
        traced.container().clone(),
        ModelProfile {
            vocab: VOCAB,
            ..ModelProfile::dummy()
        },
    )
    .expect("the epilogue binds");
    let stages = compile_bound(&bound);
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
        .expect("the epilogue compiles");
    let count = bound.container.channels.len();
    let ids: Vec<ChannelId> = (1..=count as u64).map(ChannelId).collect();
    let (mut agree, mut total) = (0, 0);
    for draw in 0..8u32 {
        let seeds = vec![
            (at("logits"), Value::F32(logits.clone())),
            (at("proposals"), Value::U32(proposals.clone())),
            (at("noise"), Value::U32(vec![17 + draw, 3 * draw])),
        ];
        let mut reference = Instance::new(&bound, &seeds).expect("the interpreter instantiates");
        let stepped = reference
            .step(&bound, &PassInputs::default(), &mut NoKernels)
            .expect("the interpreter steps");
        assert!(stepped.committed, "the reference did not commit");
        let Value::F32(want_chances) = reference.host_take(&bound, at("chances")).unwrap() else {
            panic!("chances are f32");
        };
        let Value::I32(want_draws) = reference.host_take(&bound, at("draws")).unwrap() else {
            panic!("draws are i32");
        };
        let wire: Vec<(u32, Vec<u8>)> = seeds.iter().map(|(c, v)| (*c, v.to_le_bytes())).collect();
        let instance = plane
            .bind(
                program,
                &wire,
                Extents {
                    row_count: ROWS,
                    sampled_rows: ROWS,
                    query_len: ROWS,
                    ..Extents::default()
                },
                GeometryClass::Host,
                &vec![None; count],
                &ids,
                context.stream(),
            )
            .expect("the epilogue binds an instance");
        let fired = plane.fire(&context, instance).expect("the fire runs");
        assert!(matches!(fired, Fired::Committed), "{fired:?}");
        let session = plane.instance_mut(instance).expect("the instance");
        let chances: Vec<f32> = session
            .take(at("chances"))
            .expect("the chances read")
            .expect("chances")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let draws: Vec<i32> = session
            .take(at("draws"))
            .expect("the draws read")
            .expect("draws")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c))
            .collect();
        for (r, (got, want)) in chances.iter().zip(&want_chances).enumerate() {
            assert!(
                (got - want).abs() <= 1e-4 * (1.0 + want.abs()),
                "row {r}'s chance: device {got} vs interpreter {want}"
            );
        }
        for (r, &d) in draws.iter().enumerate() {
            if r + 1 < ROWS as usize {
                assert_ne!(d as u32, proposals[r], "row {r} drew its struck proposal");
            }
        }
        total += draws.len();
        agree += draws
            .iter()
            .zip(&want_draws)
            .filter(|(a, b)| a == b)
            .count();
        if draws != want_draws {
            eprintln!("draw {draw}: the device drew {draws:?}, the interpreter {want_draws:?}");
        }
    }
    assert_eq!(
        agree, total,
        "the verify drew other tokens than the interpreter"
    );
}
