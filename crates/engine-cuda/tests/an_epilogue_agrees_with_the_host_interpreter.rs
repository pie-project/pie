use engine::ProgramRegistration;
use engine_cuda::device::{Buffer, Context};
use engine_cuda::program::compile::Disk;
use engine_cuda::program::launch::INTRINSIC_STORAGE_F32;
use engine_cuda::program::session::Fired;
use engine_cuda::program::Plane;
use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::eval::interp::{Instance, NoKernels, PassInputs, Value};
use eta_compiler::plan::compile_bound;
use eta_exec::Extents;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::{IntrinsicId, Op};
use eta_ir::registry::{GeometryClass, ModelProfile, Stage};
use eta_ir::types::{Dtype, Shape};
use eta_ir::validate::bind;

const ROWS: u32 = 8;
const VOCAB: u32 = 4096;
const K: u32 = 4;

const TEMP: u32 = 0;
const INDEX: u32 = 1;
const GATHERED: u32 = 2;
const ARGMAX: u32 = 3;
const TOP_IDS: u32 = 4;
const TOP_VALUES: u32 = 5;
const MASS: u32 = 6;
const PEAK: u32 = 7;

fn seeded(shape: Shape, dtype: Dtype) -> ChannelDecl {
    ChannelDecl { shape, dtype: ChanDType::Concrete(dtype), capacity: 1, host_role: HostRole::Writer, seeded: true }
}

fn reader(shape: Shape, dtype: Dtype) -> ChannelDecl {
    ChannelDecl { shape, dtype: ChanDType::Concrete(dtype), capacity: 2, host_role: HostRole::Reader, seeded: false }
}

fn epilogue() -> TraceContainer {
    let ops = vec![
        Op::ChanRead(TEMP),
        Op::Reshape { value: 0, shape: Shape::new(&[]).unwrap() },
        Op::ChanTake(INDEX),
        Op::IntrinsicVal { intr: IntrinsicId::Logits, shape: Shape::matrix(ROWS, VOCAB), dtype: Dtype::F32 },
        Op::Div(3, 1),
        Op::ReduceMax(4),
        Op::Reshape { value: 5, shape: Shape::matrix(ROWS, 1) },
        Op::Broadcast { value: 6, shape: Shape::matrix(ROWS, VOCAB) },
        Op::Sub(4, 7),
        Op::Exp(8),
        Op::ReduceSum(9),
        Op::Reshape { value: 10, shape: Shape::matrix(ROWS, 1) },
        Op::Broadcast { value: 11, shape: Shape::matrix(ROWS, VOCAB) },
        Op::Div(9, 12),
        Op::GatherRow { src: 4, idx: 2 },
        Op::ReduceArgmax(4),
        Op::TopK { input: 4, k: K },
        Op::ReduceSum(13),
        Op::Cast { value: 15, dtype: Dtype::U32 },
        Op::GatherRow { src: 13, idx: 19 },
        Op::ChanPut { chan: PEAK, value: 20 },
        Op::ChanPut { chan: GATHERED, value: 14 },
        Op::ChanPut { chan: ARGMAX, value: 15 },
        Op::ChanPut { chan: TOP_IDS, value: 17 },
        Op::ChanPut { chan: TOP_VALUES, value: 16 },
        Op::ChanPut { chan: MASS, value: 18 },
    ];
    TraceContainer {
        names: Vec::new(),
        channels: vec![
            seeded(Shape::vector(1), Dtype::F32),
            seeded(Shape::vector(ROWS), Dtype::I32),
            reader(Shape::vector(ROWS), Dtype::F32),
            reader(Shape::vector(ROWS), Dtype::I32),
            reader(Shape::matrix(ROWS, K), Dtype::U32),
            reader(Shape::matrix(ROWS, K), Dtype::F32),
            reader(Shape::vector(ROWS), Dtype::F32),
            reader(Shape::vector(ROWS), Dtype::F32),
        ],
        ports: Vec::new(),
        stages: vec![StageProgram { stage: Stage::Epilogue, ops }],
        externs: Vec::new(),
    }
}

fn logits() -> Vec<f32> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 / (1u64 << 53) as f64) as f32
    };
    let mut plane = vec![0f32; (ROWS * VOCAB) as usize];
    for row in 0..ROWS as usize {
        for column in 0..VOCAB as usize {
            plane[row * VOCAB as usize + column] = next() * 8.0 - 4.0 + (column % 7) as f32 * 0.01;
        }
        plane[row * VOCAB as usize + (row * 97 + 5)] = 9.0 + row as f32;
    }
    plane
}

fn f32s(value: &Value) -> Vec<f32> {
    match value {
        Value::F32(v) => v.clone(),
        other => panic!("expected f32, got {other:?}"),
    }
}

fn wire_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn device_outputs(
    bound: &eta_ir::validate::BoundTrace,
    seeds: &[(u32, Value)],
    rows: u32,
    logits: Option<&[f32]>,
    channels: &[u32],
) -> Vec<Vec<u8>> {
    let context = Context::bind(0, core::ptr::null_mut()).expect("a CUDA context");
    let stages = compile_bound(bound);
    let launch = eta_compiler::codegen::launch::build(bound, &stages);
    let backend = Backend::parse("cuda").expect("the cuda backend");
    let registration = ProgramRegistration {
        program_hash: 0x5eed ^ u64::from(rows),
        emitted_kernels: emit_program(backend, &stages, bound),
        emitter_version: backend.emitter_version(),
        region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(&stages),
        launch,
        ..Default::default()
    };
    let mut plane = Plane::new(Disk::disabled());
    let program = plane.register(&context, &registration).expect("the program compiles for the device");
    let extents = Extents { row_count: rows, sampled_rows: rows, query_len: rows, ..Extents::default() };
    let wire_seeds: Vec<(u32, Vec<u8>)> = seeds.iter().map(|(c, v)| (*c, v.to_le_bytes())).collect();
    let count = bound.container.channels.len();
    let ids: Vec<u64> = (1..=count as u64).collect();
    let instance = plane
        .bind(
            program,
            &wire_seeds,
            extents,
            GeometryClass::Host,
            &vec![None; count],
            &ids,
            context.stream(),
        )
        .expect("the instance binds");
    let _plane_buffer = logits.map(|plane_values| {
        let mut buffer = Buffer::zeroed(plane_values.len() * 4).expect("a logits plane");
        let bytes: Vec<u8> = plane_values.iter().flat_map(|x| x.to_le_bytes()).collect();
        buffer.write(0, &bytes).expect("the plane uploads");
        let width = plane_values.len() as u32 / rows;
        plane
            .bind_intrinsic(instance, IntrinsicId::Logits, buffer.ptr(), INTRINSIC_STORAGE_F32, width, width, 0)
            .expect("the logits bind");
        buffer
    });
    let fired = plane.fire(&context, instance).expect("the fire runs");
    assert!(matches!(fired, Fired::Committed), "the device fire did not commit: {fired:?}");
    let session = plane.instance_mut(instance).expect("the instance");
    channels
        .iter()
        .map(|&channel| session.take(channel).expect("a channel reads").expect("the channel holds a cell"))
        .collect()
}

fn host_outputs(
    bound: &eta_ir::validate::BoundTrace,
    seeds: &[(u32, Value)],
    logits: Option<&[f32]>,
    channels: &[u32],
) -> Vec<Value> {
    let mut reference = Instance::new(bound, seeds).expect("the interpreter instantiates");
    let inputs = PassInputs { logits: logits.map(|l| Value::F32(l.to_vec())), ..Default::default() };
    let stepped = reference.step(bound, &inputs, &mut NoKernels).expect("the interpreter steps");
    assert!(stepped.committed, "the reference did not commit: {:?}", stepped.missed);
    channels.iter().map(|&c| reference.host_take(bound, c).unwrap()).collect()
}

fn close(a: &[f32], b: &[f32], tolerance: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: lengths");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tolerance * (1.0 + y.abs()), "{what}[{i}]: device {x} vs interpreter {y}");
    }
}

fn wire_bools(bytes: &[u8], count: usize) -> Vec<bool> {
    (0..count).map(|i| bytes[i / 8] >> (i % 8) & 1 == 1).collect()
}

fn bools(value: &Value) -> Vec<bool> {
    match value {
        Value::Bool(v) => v.clone(),
        other => panic!("expected bool, got {other:?}"),
    }
}

#[test]
fn an_epilogue_agrees_with_the_host_interpreter_every_case() {
    the_device_answers_what_the_interpreter_answers();
    the_acceptance_rule_agrees_with_the_interpreter();
    the_gaussian_draw_and_the_transcendentals_agree_with_the_interpreter();
}

fn the_device_answers_what_the_interpreter_answers() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let profile = ModelProfile { vocab: VOCAB, ..ModelProfile::dummy() };
    let bound = bind(epilogue(), profile).expect("the epilogue binds");
    let seeds = vec![
        (TEMP, Value::F32(vec![0.7])),
        (INDEX, Value::I32((0..ROWS as i32).map(|r| (r * 131 + 17) % VOCAB as i32).collect())),
    ];
    let plane = logits();
    let channels = [GATHERED, ARGMAX, TOP_IDS, TOP_VALUES, MASS, PEAK];
    let want = host_outputs(&bound, &seeds, Some(&plane), &channels);
    let got = device_outputs(&bound, &seeds, ROWS, Some(&plane), &channels);

    close(&wire_f32(&got[0]), &f32s(&want[0]), 1e-5, "gather_row of the scaled logits");
    assert_eq!(got[1], want[1].to_le_bytes(), "argmax per row");
    assert_eq!(got[2], want[2].to_le_bytes(), "top-{K} ids per row");
    close(&wire_f32(&got[3]), &f32s(&want[3]), 1e-5, "top-k values");
    close(&wire_f32(&got[4]), &f32s(&want[4]), 1e-4, "softmax row mass");
    close(&wire_f32(&got[4]), &vec![1.0; ROWS as usize], 1e-3, "softmax row mass against one");
    close(&wire_f32(&got[5]), &f32s(&want[5]), 1e-5, "probability at the row's argmax (a row block's gather)");
}

const ENTROPY: u32 = 0;
const SAMPLED: u32 = 1;
const NOISE: u32 = 2;
const PREVIOUS: u32 = 3;
const ACCEPT: u32 = 4;
const NEXT: u32 = 5;
const COUNT: u32 = 6;
const DONE: u32 = 7;
const ORDER: u32 = 8;

fn acceptance() -> TraceContainer {
    use eta_ir::types::Literal;
    let ops = vec![
        Op::ChanTake(ENTROPY),
        Op::Neg(0),
        Op::SortDesc(1),
        Op::Neg(2),
        Op::CumSum(4),
        Op::Sub(5, 4),
        Op::Const(Literal::F32(3.0)),
        Op::Le(6, 7),
        Op::Iota { len: ROWS },
        Op::Const(Literal::U32(0)),
        Op::Lt(9, 10),
        Op::ScatterSet { base: 11, idx: 3, vals: 8 },
        Op::ChanTake(SAMPLED),
        Op::ChanTake(NOISE),
        Op::Select { cond: 12, a: 13, b: 14 },
        Op::Cast { value: 12, dtype: Dtype::I32 },
        Op::ReduceSum(16),
        Op::Reshape { value: 17, shape: Shape::vector(1) },
        Op::ChanTake(PREVIOUS),
        Op::Eq(13, 19),
        Op::Cast { value: 20, dtype: Dtype::I32 },
        Op::ReduceSum(21),
        Op::Const(Literal::I32(ROWS as i32)),
        Op::Eq(22, 23),
        Op::ReduceSum(0),
        Op::Const(Literal::F32(ROWS as f32)),
        Op::Div(25, 26),
        Op::Const(Literal::F32(0.5)),
        Op::Lt(27, 28),
        Op::And(24, 29),
        Op::Reshape { value: 30, shape: Shape::vector(1) },
        Op::ChanPut { chan: ACCEPT, value: 12 },
        Op::ChanPut { chan: NEXT, value: 15 },
        Op::ChanPut { chan: COUNT, value: 18 },
        Op::ChanPut { chan: DONE, value: 31 },
        Op::ChanPut { chan: ORDER, value: 3 },
    ];
    TraceContainer {
        names: Vec::new(),
        channels: vec![
            seeded(Shape::vector(ROWS), Dtype::F32),
            seeded(Shape::vector(ROWS), Dtype::I32),
            seeded(Shape::vector(ROWS), Dtype::I32),
            seeded(Shape::vector(ROWS), Dtype::I32),
            reader(Shape::vector(ROWS), Dtype::Bool),
            reader(Shape::vector(ROWS), Dtype::I32),
            reader(Shape::vector(1), Dtype::I32),
            reader(Shape::vector(1), Dtype::Bool),
            reader(Shape::vector(ROWS), Dtype::U32),
        ],
        ports: Vec::new(),
        stages: vec![StageProgram { stage: Stage::Epilogue, ops }],
        externs: Vec::new(),
    }
}

fn the_acceptance_rule_agrees_with_the_interpreter() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let profile = ModelProfile { vocab: VOCAB, ..ModelProfile::dummy() };
    let bound = bind(acceptance(), profile).expect("the acceptance rule binds");
    let entropy: Vec<f32> = vec![0.9, 0.05, 2.5, 0.4, 1.7, 0.01, 0.6, 3.2];
    let sampled: Vec<i32> = (0..ROWS as i32).map(|r| 100 + r).collect();
    let noise: Vec<i32> = (0..ROWS as i32).map(|r| 900 + 7 * r).collect();
    let previous: Vec<i32> = (0..ROWS as i32).map(|r| if r == 3 { -1 } else { 100 + r }).collect();
    let seeds = vec![
        (ENTROPY, Value::F32(entropy)),
        (SAMPLED, Value::I32(sampled)),
        (NOISE, Value::I32(noise)),
        (PREVIOUS, Value::I32(previous)),
    ];
    let channels = [ACCEPT, NEXT, COUNT, DONE, ORDER];
    let want = host_outputs(&bound, &seeds, None, &channels);
    let got = device_outputs(&bound, &seeds, ROWS, None, &channels);
    assert_eq!(wire_bools(&got[0], ROWS as usize), bools(&want[0]), "accepted rows");
    assert_eq!(got[1], want[1].to_le_bytes(), "the next canvas");
    assert_eq!(got[2], want[2].to_le_bytes(), "the accepted count");
    assert_eq!(wire_bools(&got[3], 1), bools(&want[3]), "the stop flag");
    assert_eq!(got[4], want[4].to_le_bytes(), "the sort's permutation");
    let accepted = bools(&want[0]).iter().filter(|&&b| b).count();
    assert!(accepted > 0 && accepted < ROWS as usize, "the bound should split the rows, accepted {accepted}");
}

const LATENT: u32 = 0;
const STATE: u32 = 1;
const GAUSSIAN: u32 = 2;
const SINE: u32 = 3;
const COSINE: u32 = 4;
const PYTHAGORAS: u32 = 5;
const ROOT: u32 = 6;
const INVERSE_ROOT: u32 = 7;
const UNITY: u32 = 8;
const STEPPED: u32 = 9;

const WIDTH: u32 = 16;

fn sampler_arithmetic() -> TraceContainer {
    use eta_ir::types::RngKind;
    let plane = Shape::matrix(ROWS, WIDTH);
    let ops = vec![
        Op::ChanRead(STATE),
        Op::RngKeyed { state: 0, shape: plane, kind: RngKind::Normal },
        Op::ChanTake(LATENT),
        Op::Sin(2),
        Op::Cos(2),
        Op::Mul(3, 3),
        Op::Mul(4, 4),
        Op::Add(5, 6),
        Op::Sqrt(2),
        Op::Rsqrt(2),
        Op::Mul(8, 9),
        Op::Add(2, 1),
        Op::ChanPut { chan: GAUSSIAN, value: 1 },
        Op::ChanPut { chan: SINE, value: 3 },
        Op::ChanPut { chan: COSINE, value: 4 },
        Op::ChanPut { chan: PYTHAGORAS, value: 7 },
        Op::ChanPut { chan: ROOT, value: 8 },
        Op::ChanPut { chan: INVERSE_ROOT, value: 9 },
        Op::ChanPut { chan: UNITY, value: 10 },
        Op::ChanPut { chan: STEPPED, value: 11 },
    ];
    TraceContainer {
        names: Vec::new(),
        channels: vec![
            seeded(plane, Dtype::F32),
            seeded(Shape::vector(2), Dtype::U32),
            reader(plane, Dtype::F32),
            reader(plane, Dtype::F32),
            reader(plane, Dtype::F32),
            reader(plane, Dtype::F32),
            reader(plane, Dtype::F32),
            reader(plane, Dtype::F32),
            reader(plane, Dtype::F32),
            reader(plane, Dtype::F32),
        ],
        ports: Vec::new(),
        stages: vec![StageProgram { stage: Stage::Epilogue, ops }],
        externs: Vec::new(),
    }
}

fn latent() -> Vec<f32> {
    (0..(ROWS * WIDTH) as usize)
        .map(|i| 0.05 + (i % 97) as f32 * 0.11)
        .collect()
}

fn the_gaussian_draw_and_the_transcendentals_agree_with_the_interpreter() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let profile = ModelProfile { vocab: VOCAB, ..ModelProfile::dummy() };
    let bound = bind(sampler_arithmetic(), profile).expect("the sampler epilogue binds");
    let x = latent();
    let (key, counter) = (0x7ce1u32, 5u32);
    let seeds = vec![
        (LATENT, Value::F32(x.clone())),
        (STATE, Value::U32(vec![key, counter])),
    ];
    let channels = [GAUSSIAN, SINE, COSINE, PYTHAGORAS, ROOT, INVERSE_ROOT, UNITY, STEPPED];
    let want = host_outputs(&bound, &seeds, None, &channels);
    let got = device_outputs(&bound, &seeds, ROWS, None, &channels);

    close(&wire_f32(&got[0]), &f32s(&want[0]), 1e-5, "the normal draw");
    close(&wire_f32(&got[1]), &f32s(&want[1]), 1e-5, "sin");
    close(&wire_f32(&got[2]), &f32s(&want[2]), 1e-5, "cos");
    close(&wire_f32(&got[3]), &f32s(&want[3]), 1e-5, "sin^2 + cos^2");
    close(&wire_f32(&got[4]), &f32s(&want[4]), 1e-5, "sqrt");
    close(&wire_f32(&got[5]), &f32s(&want[5]), 1e-5, "rsqrt");
    close(&wire_f32(&got[6]), &f32s(&want[6]), 1e-5, "sqrt * rsqrt");
    close(&wire_f32(&got[7]), &f32s(&want[7]), 1e-5, "x + z");

    let n = (ROWS * WIDTH) as usize;
    close(&wire_f32(&got[3]), &vec![1.0; n], 1e-5, "sin^2 + cos^2 against one");
    close(&wire_f32(&got[6]), &vec![1.0; n], 1e-5, "sqrt * rsqrt against one");
    let want_root: Vec<f32> = x.iter().map(|v| v.sqrt()).collect();
    close(&wire_f32(&got[4]), &want_root, 1e-6, "sqrt against the host's");

    let seed = eta_ir::rng::keyed_seed(key, counter);
    let want_noise: Vec<f32> =
        (0..n as u32).map(|i| eta_ir::rng::hash_normal(seed, i)).collect();
    close(&wire_f32(&got[0]), &want_noise, 1e-5, "the normal draw against `rng::hash_normal`");

    let drawn = wire_f32(&got[0]);
    let mean = drawn.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
    let variance =
        drawn.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / n as f64;
    assert!(variance > 0.4, "the device drew a near-constant plane (variance {variance})");
}
