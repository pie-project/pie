//! WIT call latency microbench.
//!
//! Runs tight loops of cheap WIT calls (no GPU work) to measure
//! the per-call cost of the WASM→host boundary, isolated from
//! scheduler / GPU latency.

use std::time::Instant;
use inferlet::{Result, model::Model, runtime};
use inferlet::pie::core::inference::{ForwardPass, Sampler as WitSampler};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_iterations")]
    iterations: u64,
}

fn default_iterations() -> u64 { 200_000 }

#[derive(Serialize)]
struct BenchResult {
    name: String,
    iterations: u64,
    total_us: u64,
    per_call_ns: u64,
}

#[derive(Serialize)]
struct Output {
    benches: Vec<BenchResult>,
}

fn measure<F: FnMut()>(name: &str, iters: u64, mut f: F) -> BenchResult {
    // Warmup
    for _ in 0..(iters / 100).max(1) {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let dur = start.elapsed();
    BenchResult {
        name: name.into(),
        iterations: iters,
        total_us: dur.as_micros() as u64,
        per_call_ns: (dur.as_nanos() as u64).checked_div(iters).unwrap_or(0),
    }
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let n = input.iterations;
    let mut benches = Vec::new();

    // 1. Pure-WASM baseline: no host calls.
    benches.push(measure("baseline (no WIT call)", n, || {
        std::hint::black_box(0u32);
    }));

    // 2. Cheapest WIT call: runtime::instance_id returns a String.
    //    Used as the "minimum WIT crossing cost" data point.
    benches.push(measure("runtime::instance_id()", n.min(50_000), || {
        let id = runtime::instance_id();
        std::hint::black_box(id);
    }));

    // For ForwardPass we need a model. Load the first available.
    let models = runtime::models();
    let model_name = models.first().ok_or("no models available")?;
    let model = Model::load(model_name)?;

    // 3. ForwardPass::new + drop. Measures full resource lifecycle:
    //    push to ResourceTable + WIT drop call.
    benches.push(measure("ForwardPass::new + drop", n.min(50_000), || {
        let pass = ForwardPass::new(&model);
        std::hint::black_box(&pass);
        drop(pass);
    }));

    // Reuse one ForwardPass for the field-setter measurements.
    let pass = ForwardPass::new(&model);

    // 4. Simplest field setter: a single bool.
    benches.push(measure("pass.output_speculative_tokens(true)", n, || {
        pass.output_speculative_tokens(true);
    }));

    // 5. Vec setter with 1-element data (decode shape).
    let tokens = vec![1u32];
    let positions = vec![0u32];
    benches.push(measure("pass.input_tokens(&[1], &[0])", n, || {
        pass.input_tokens(&tokens, &positions);
    }));

    // 6. Vec setter with 32-element data (prefill-ish shape).
    let big_tokens: Vec<u32> = (0..32).collect();
    let big_positions: Vec<u32> = (0..32).collect();
    benches.push(measure("pass.input_tokens(&[32], &[32])", n.min(50_000), || {
        pass.input_tokens(&big_tokens, &big_positions);
    }));

    // 7. Sampler setter — the one with HashMap<String, rmpv> on the host side.
    let sampler = WitSampler::Multinomial((0.0, 0));
    let idx = vec![0u32];
    benches.push(measure("pass.sampler(...)", n, || {
        pass.sampler(&idx, &sampler);
    }));

    // 8. Full setup sequence per call (new + context-omitted + 3 setters + drop).
    //    Approximates the per-iteration WIT cost in GenStep::execute.
    benches.push(measure(
        "full setup (new + 3 setters + drop)",
        n.min(50_000),
        || {
            let p = ForwardPass::new(&model);
            p.output_speculative_tokens(true);
            p.input_tokens(&tokens, &positions);
            p.sampler(&idx, &sampler);
            drop(p);
        },
    ));

    Ok(Output { benches })
}
