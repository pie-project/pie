//! A guest stage carrying cells reads and writes a lane's rows of the
//! pool, not the pool: compiled for one lane over a pool the size of a
//! Qwen3-0.6B load's on v6e (50 408 960 rows, 25.8 GB), gumbel_watermark's
//! step asks temporaries of its own values, not of the pool. A one-row read
//! or write of the pool once became a `[1, 128]` slice XLA laid out
//! column-major, which copied the whole pool to that layout (rematerialized,
//! four times): the stage asked 48 GB of a 31 GB chip and never ran.
//! Skipped without a PJRT plugin.

use eta_dsl::prelude::*;
use eta_exec::ExecPlan;
use eta_ir::registry::EngineCaps;

/// Qwen3's vocabulary.
const VOCAB: u32 = 151_936;
/// The pool a Qwen3-0.6B load seats on v6e, in rows.
const POOL_ROWS: i64 = 50_408_960;
/// The temporaries a lane's pass may ask: its vocabulary-wide values are
/// megabytes; a copy of the pool would be 25.8 GB.
const TEMP_BOUND: u64 = 256 << 20;

const HASH_MODULUS: u32 = 2_000_003;
const HASH_MULTIPLIER: u32 = 31;
const CONTEXT: u32 = 4;
const SECRET: u32 = 15_485_863;

fn leak(channel: Channel) -> &'static Channel {
    Box::leak(Box::new(channel))
}

/// gumbel_watermark's rolling hash of the last `CONTEXT` tokens.
fn context_counter(hist: &Tensor, hlen: &Tensor) -> Tensor {
    let last = cast(hlen, dtype::i32) - 1i32;
    let mut h = broadcast(0u32, [1]);
    for d in (0..CONTEXT).rev() {
        let idx = max_elem(&last - d as i32, 0i32);
        let tok = cast(gather(hist, &idx) + 1i32, dtype::u32);
        h = (&h * HASH_MULTIPLIER + tok) % HASH_MODULUS;
    }
    h
}

fn rng_state(key: u32, counter: &Tensor) -> Tensor {
    let is_counter = eq(iota(2), broadcast(1u32, [2]));
    select(&is_counter, broadcast(counter, [2]), broadcast(key, [2]))
}

fn score(noise: &Tensor, token: &Tensor) -> Tensor {
    let g = scalar_gather(noise, token);
    let r = exp(-exp(-&g));
    reshape(-log(&max_elem(1.0f32 - &r, 1e-7f32)), [1])
}

/// gumbel_watermark's epilogue: its rng state, history and length carried
/// in the pool from pass to pass; the token and both scores read by the
/// host.
fn gumbel_watermark() -> eta_ir::container::TraceContainer {
    let rng = leak(Channel::from([7u32, 0]));
    let hist = leak(Channel::from(vec![-1i32; 64]));
    let hlen = leak(Channel::from([3u32]));
    let tok_out = leak(Channel::new([1], dtype::i32));
    let score_out = leak(Channel::new([1], dtype::f32));
    let null_out = leak(Channel::new([1], dtype::f32));
    let mut b = Builder::new(VOCAB, 16);
    b.outputs(logits());
    b.stage(Stage::Epilogue, move || {
        let r = rng.take();
        let h = hist.take();
        let n = hlen.take();
        let logits = intrinsics::output(0, "logits");
        let counter = context_counter(&h, &n);
        let noise = gumbel(&rng_state(SECRET, &counter), [VOCAB]);
        let token = reduce_argmax(&logits + &noise);
        let null = score(
            &gumbel(&rng_state(SECRET ^ 0x9E37_79B9, &counter), [VOCAB]),
            &token,
        );
        tok_out.put(&token);
        score_out.put(&score(&noise, &token));
        null_out.put(&null);
        hist.put(&scatter_set(&h, &n, &token));
        hlen.put(&n + 1u32);
        rng.put(&r + iota(2));
    });
    for out in [tok_out, score_out, null_out] {
        out.note_host_take();
    }
    b.build().expect("the epilogue traces").container().clone()
}

fn logits() -> Vec<Vec<eta_ir::registry::OutputDecl>> {
    vec![vec![eta_ir::registry::OutputDecl {
        name: "logits".to_string(),
        axis: eta_ir::registry::Axis::Row,
        width: VOCAB,
        dtype: eta_ir::types::Dtype::F32,
        layers: 0,
        readout: true,
    }]]
}

fn plan() -> ExecPlan {
    let bound = eta_ir::validate::bind(gumbel_watermark(), logits(), EngineCaps::default())
        .expect("the program binds");
    let stages = eta_compiler::plan::compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let plan = eta_exec::adopt_launch_package(launch).expect("the package adopts");
    if let Err(why) = engine_xla::guest::admits(&plan.package) {
        panic!("the program has no device form: {why}");
    }
    plan
}

#[test]
fn a_carried_history_copies_none_of_the_pool() {
    if std::env::var_os("PIE_XLA_PLUGIN").is_none()
        && std::env::var_os("TPU_LIBRARY_PATH").is_none()
    {
        eprintln!("no PJRT plugin named: skipping");
        return;
    }
    let _lock = engine_xla::bench::lock_device();
    let device = &engine_xla::device::Device::open(None, 0).expect("the device opens");
    let plan = plan();
    let carried = engine_xla::guest::carried_channels(&plan.package);
    assert_eq!(carried.len(), 3, "rng, history and length ride the pool");
    // The 128-row plane a fire lands the logits in, lane 0 reading row 0.
    let rows = 128u32;
    let bytes = vec![0u8; (rows * VOCAB) as usize * 4];
    let buffer = device
        .upload(eta_ir::types::Dtype::F32, rows, VOCAB, &bytes)
        .expect("the plane uploads");
    let kept = engine_xla::readout::Kept {
        planes: vec![engine_xla::readout::Plane::new(
            "logits".to_string(),
            poem_ir::InputAxis::Row,
            true,
            std::sync::Arc::new(buffer),
            rows,
            VOCAB,
            0,
            eta_ir::types::Dtype::F32,
        )],
        seats: vec![vec![Some(engine_xla::readout::Seat { first: 0, count: 1 })]],
        readback: vec![None],
        pixels: (String::new(), Vec::new()),
        host: None,
    };
    let bound = engine_xla::guest::run::bound(&plan, Some(&kept), 0, 1)
        .expect("the program reads its own lane")
        .expect("the lane binds");
    let batch = engine_xla::guest::run::batch_of(&plan.package, 0, 1, Some(&kept), &bound);
    let mut stages = engine_xla::guest::run::Stages::default();
    stages.pool_rows = POOL_ROWS;
    stages.sink_row = (POOL_ROWS - 8) as u32;
    let key = *blake3::hash(b"gumbel_watermark").as_bytes();
    let staged = stages
        .get(device, &plan.package, key, 0, batch, &carried)
        .expect("the stage compiles over a 0.6B load's pool");
    let Some(program) = staged.program.as_ref() else {
        panic!("the stage puts its cells");
    };
    let temp = program.memory.temp;
    eprintln!("gumbel_watermark's step, one lane, a 25.8 GB pool: {temp} temp bytes");
    assert!(
        temp < TEMP_BOUND,
        "temporaries of the step, not of the pool: {temp} bytes (bound {TEMP_BOUND})"
    );
}
