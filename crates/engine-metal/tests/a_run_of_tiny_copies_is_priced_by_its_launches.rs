//! **WHAT A FIRE'S TINY DEVICE COPIES COST, AS A FUNCTION OF THEIR COUNT** —
//! the question behind batching `layout::copy_words`.
//!
//! The recurrent buffer's scatter/gather (`engine_metal::dispatch::rs`) fires
//! ONE `rs_copy_words` launch per lane per plane per kind per recurrent op:
//! 864 launches in a two-lane speculative fire on Qwen3.6-27B, 432 a lane,
//! and the count is linear in lanes (~3.5k at eight). Each copy is a few
//! hundred bytes. Whether collapsing them into one launch per (source,
//! destination) pair is worth the descriptor table it needs is a question
//! about the PER-LAUNCH cost of a dispatch this small, which is what this
//! measures.
//!
//! # THE ANSWER IS NO, AND THIS FILE IS WHY (M4 Pro, 2026-09-04)
//!
//! **A launch of this shape costs ~1.25 us on the device** — steady to
//! within a few percent across 108, 432, 864 and 3456 launches and across
//! three runs, once the clock is up (see the warm hold below; an unwarmed
//! first point reads 2-4x slow). Host encode adds ~0.2-0.3 us apiece.
//!
//! So the whole of a two-lane speculative fire's 864 copies is **1.08 ms of
//! a 111 ms fire**, and batching them across lanes — the only batching the
//! buffer pairs admit, since each plane binds its own handle into the
//! scratch — takes them to 432, saving **0.54 ms, or 0.5%**. At eight lanes
//! 3456 launches are 4.3 ms and batching saves 3.8 ms, ~2.9% of that fire.
//! Neither flips speculation's concurrency verdict, which needs 3.4 ms at
//! two lanes to reach break-even and does not get there from here.
//!
//! The change that was NOT made on this evidence: a `rs_copy_runs` kernel
//! reading a per-fire descriptor table of `(src_off, dst_off, words)`. It
//! needs a host-written region in `crate::inputs` (per arm, so a run-ahead
//! fire cannot overwrite a live one), that region threaded through
//! `FireBindings` into `Run`, a cursor with a flush-when-full rule, and the
//! batching itself in `crate::dispatch::rs` — real correctness surface for
//! half a percent. The dominant cost in that fire is `gated_delta_committed`
//! at ~2.8x the plain scan (+4.8 ms), which is inherent: fold-commit runs
//! the scan over replay+own rows, which is what fold-commit IS.
//!
//! ```text
//! PIE_COPY_LAUNCHES=54,108,432,864,3456 PIE_COPY_BYTES=512 PIE_COPY_STEPS=20 \
//!   cargo test -p engine-metal --release --test a_run_of_tiny_copies_is_priced_by_its_launches -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use std::time::Instant;

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::{Tensor, layout};
use model_ir::Dtype;

fn env<T: std::str::FromStr>(name: &str, fallback: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(fallback)
}

fn list(name: &str, fallback: &str) -> Vec<u32> {
    std::env::var(name)
        .unwrap_or_else(|_| fallback.to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

#[test]
fn a_run_of_copies_is_timed() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    // One recurrent row of the 27B's GDN planes is a few hundred bytes; the
    // default is that size, and the counts bracket what a fire fires today.
    let bytes: u64 = env("PIE_COPY_BYTES", 512u64);
    let counts = list("PIE_COPY_LAUNCHES", "54,108,432,864,3456");
    let steps: usize = env("PIE_COPY_STEPS", 20usize);
    let widest = u64::from(*counts.iter().max().expect("a count"));
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    eprintln!("device: {}", device.name());

    // Source and destination far enough apart that no two copies overlap.
    let src_b = Buffer::zeroed(&device, widest * bytes).expect("src");
    let dst_b = Buffer::zeroed(&device, widest * bytes).expect("dst");
    eprintln!("\n  {bytes} bytes a copy, {steps} timed command buffers a point");
    let mut one = 0.0f64;
    for &n in &counts {
        // A distinct handle per copy, as `Run::rs_copy` cuts one per end.
        let ends: Vec<(u32, u32)> = (0..n)
            .map(|i| {
                let at = u64::from(i) * bytes;
                (
                    handles.bind(&src_b, at, bytes).expect("a src handle"),
                    handles.bind(&dst_b, at, bytes).expect("a dst handle"),
                )
            })
            .collect();
        let fire = |sink: &Sink<'_>| {
            for &(s, d) in &ends {
                layout::copy_words(
                    sink,
                    Tensor::new(s, 1, 1, Dtype::U32),
                    Tensor::new(d, 1, 1, Dtype::U32),
                    bytes,
                )
                .expect("the copy");
            }
        };
        // Held busy until the GPU's clock is up: a few hundred microseconds
        // of work reads two-to-four times slow on an idle governor, which is
        // the trap `a_quantized_matmul_is_priced_by_its_rows` documents.
        {
            let warm_ms: u64 = env("PIE_COPY_WARM_MS", 300u64);
            let began = Instant::now();
            loop {
                let frame = device.frame().expect("a frame");
                fire(&Sink::new(&device, &frame, &pipelines, &handles));
                frame.commit().expect("the warm commit");
                if began.elapsed().as_millis() as u64 >= warm_ms {
                    break;
                }
            }
        }
        let began = Instant::now();
        let mut device_s = 0.0f64;
        for _ in 0..steps {
            let frame = device.frame().expect("a frame");
            fire(&Sink::new(&device, &frame, &pipelines, &handles));
            device_s += frame.commit_timed().expect("the commit");
        }
        let dev_us = device_s * 1e6 / steps as f64;
        let wall_us = began.elapsed().as_secs_f64() * 1e6 / steps as f64;
        if n == counts[0] {
            one = dev_us / f64::from(n);
        }
        eprintln!(
            "    {n:>5} launches: {dev_us:>9.1} us device ({:>5.2} us a launch)  wall {wall_us:>9.1} us ({:>5.2} us a launch, encode included)",
            dev_us / f64::from(n),
            wall_us / f64::from(n),
        );
        let _ = one;
    }
}
