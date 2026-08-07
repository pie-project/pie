//! What the declared drive COSTS per fire, on the host.
//!
//! The throughput A/B (gemma-4, alternating boots) found no measurable
//! difference between the declared drive and the hand-written pass, which
//! is the expected answer: the drive fires the same kernels in the same
//! order, so its only extra work is host-side — build the row vector,
//! lower it, walk the flat list. "Too small to see end-to-end" is a
//! bound, not a number, and this file turns it into one.
//!
//! It measures the LOWERING only, which is the dominant term and the one
//! that scales with the plan: `lower()` walks every op of the traced form
//! and emits rectangles. The executor's own walk is a switch per launch
//! and the row build is a `memset`-shaped loop; both are smaller and both
//! are C++ rather than Rust, so they are outside what this can honestly
//! claim.
//!
//! `#[ignore]` — it is a measurement, not an assertion. Timings are not
//! CI material: they would flake on a loaded machine and the number that
//! matters is the ORDER, not the digits.
//!
//! ```text
//! cargo test -p pie-forward --release --test lowering_cost -- --ignored --nocapture
//! ```

use std::time::Instant;

use pie_forward::family::{gemma4_cuda, gpt_oss_cuda, llama_like_cuda, qwen3_5_hybrid_cuda};
use pie_forward::lower::{lower, Fire, Row};
use pie_forward::{
    FireClass, Gemma4CudaFacts, Gemma4Facts, GptOssCudaFacts, GptOssFacts, LlamaLikeCudaFacts,
    LlamaLikeFacts, Qwen35CudaFacts, Qwen35HybridFacts,
};

fn rows(n: usize) -> Vec<Row> {
    (0..n)
        .map(|_| Row {
            samples: true,
            ..Row::default()
        })
        .collect()
}

/// Median of `reps` timed calls, in microseconds. Median rather than mean
/// because one preempted call should not set the number.
fn median_us(plan: &pie_forward::ForwardPlan, rows: &[Row], reps: usize) -> f64 {
    // Warm the allocator and any lazily built tables first; the first call
    // is not the one being asked about.
    for _ in 0..8 {
        let _ = lower(plan, rows, Fire::default());
    }
    let mut samples: Vec<f64> = (0..reps)
        .map(|_| {
            let t0 = Instant::now();
            let out = lower(plan, rows, Fire::default());
            let dt = t0.elapsed().as_secs_f64() * 1e6;
            // Consume the result so nothing is optimized away.
            assert!(out.is_ok());
            dt
        })
        .collect();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

#[test]
#[ignore = "a measurement, not an assertion"]
fn what_lowering_costs_per_fire() {
    const REPS: usize = 200;
    let cases: Vec<(&str, pie_forward::ForwardPlan)> = vec![
        (
            "llama_like qwen3-0.6B decode",
            llama_like_cuda(
                &LlamaLikeFacts::qwen3_0_6b(),
                &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
                FireClass::Decode,
            ),
        ),
        (
            "qwen3_5 0.8B decode",
            qwen3_5_hybrid_cuda(
                &Qwen35HybridFacts::qwen3_5_0_8b(),
                &Qwen35CudaFacts::qwen3_5_0_8b_synthetic(),
                FireClass::Decode,
            ),
        ),
        (
            "gemma-4 E4B decode",
            gemma4_cuda(
                &Gemma4Facts::gemma_4_e4b(),
                &Gemma4CudaFacts::gemma_4_e4b_synthetic(),
                FireClass::Decode,
            ),
        ),
        (
            "gpt-oss 20B decode",
            gpt_oss_cuda(
                &GptOssFacts::gpt_oss_20b(),
                &GptOssCudaFacts::gpt_oss_20b_synthetic(),
                FireClass::Decode,
            ),
        ),
    ];

    println!(
        "\n{:<30} {:>5} {:>10} {:>10} {:>10}",
        "plan", "ops", "N=1 us", "N=16 us", "N=64 us"
    );
    for (name, plan) in &cases {
        let one = median_us(plan, &rows(1), REPS);
        let sixteen = median_us(plan, &rows(16), REPS);
        let sixtyfour = median_us(plan, &rows(64), REPS);
        println!(
            "{:<30} {:>5} {:>10.1} {:>10.1} {:>10.1}",
            name,
            plan.ops.len(),
            one,
            sixteen,
            sixtyfour
        );
    }
    // The scale, with its arithmetic shown rather than asserted. The
    // throughput A/B ran 16 concurrent x 128 tokens in ~2.2 s. Those
    // requests co-batch, so the wave is ~128 decode FIRES, not 2048:
    // 2.2 s / 128 = ~17 ms per fire. A ~36 us lowering is ~0.2% of that.
    //
    // Which is the answer to why the alternating-boot A/B saw nothing: the
    // effect is two orders of magnitude below the ~5% boot-to-boot spread
    // that measurement had to fight.
    println!(
        "\nScale: the throughput A/B ran 16 concurrent x 128 tokens in ~2.2 s.\n\
         Co-batched, that is ~128 decode fires => ~17 ms per fire.\n\
         A ~36 us lowering is ~0.2% of one fire — two orders of magnitude\n\
         under the ~5% boot-to-boot spread the A/B had to fight, which is\n\
         why it could not see this and should not have been expected to.\n"
    );
}
