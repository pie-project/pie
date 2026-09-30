//! Decode step time against batch width, for the bench. Asked for with
//! `PIE_XLA_SNAPSHOT` + `PIE_XLA_SKU` (as `a_model_speaks_on_the_device`)
//! and `PIE_XLA_BENCH=1`.

mod common;

use std::time::Instant;

use engine_xla::{Boot, DeviceBoot, Lane, Seated, Shell};
use model_compiler::Budget;
use model_dsl::{Platform, Request};

#[test]
fn decode_step_time_by_batch_width() {
    if std::env::var("PIE_XLA_BENCH").is_err() {
        eprintln!("not asked: set PIE_XLA_BENCH");
        return;
    }
    let Some(m) = common::model() else {
        eprintln!("not asked: set PIE_XLA_SNAPSHOT + PIE_XLA_SKU or PIE_XLA_ARTIFACT");
        return;
    };
    let sku = m.sku;
    let word = |query_len: u32| (sku.classify)(&Request::new(query_len, false));
    let widths: Vec<u32> = std::env::var("PIE_XLA_BENCH_WIDTHS")
        .ok()
        .map(|s| s.split(',').filter_map(|w| w.parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 8, 32, 64]);
    let most = *widths.iter().max().unwrap_or(&1);
    let context = 1024;
    let _device = engine_xla::bench::lock_device();
    let mut shell = Shell::load(Boot {
        trace: (sku.trace)(Platform::Xla),
        contract: &m.contract,
        checkpoint: &m.checkpoint,
        budget: Budget::new(most, 2048),
        page_size: 16,
        context,
        slots: most,
        // Slot-seated lanes cycle a whole ring of windowed pages each, and
        // the windowed pool holds half the kv pages past one sequence's: twice
        // the pages seats every slot's ring there.
        pages: 2 * most * context / 16,
        device: &DeviceBoot::default(),
        patches: None,
    })
    .expect("the shell loads");

    // 72 tokens: both timed loops and the traced steps (to 125 tokens) stay
    // within one bucket of pages per lane (8 pages of 16), so none of them
    // times a compile.
    let prompt: Vec<u32> = (0..72).map(|i| 1000 + i).collect();
    for &width in &widths {
        for slot in 0..width {
            shell.open(slot).expect("a slot opens");
            shell
                .fire(&[Lane {
                    slot,
                    word: word(prompt.len() as u32),
                    tokens: &prompt,
                }])
                .expect("a prefill fires");
        }
        let fed: Vec<[u32; 1]> = (0..width).map(|i| [100 + i]).collect();
        let lanes: Vec<Lane<'_>> = fed
            .iter()
            .enumerate()
            .map(|(slot, t)| Lane {
                slot: slot as u32,
                word: word(1),
                tokens: t,
            })
            .collect();
        // Warm (compile), then time.
        shell.fire(&lanes).expect("a decode fires");
        let steps = 20;
        let started = Instant::now();
        for _ in 0..steps {
            shell.fire(&lanes).expect("a decode fires");
        }
        let per = started.elapsed().as_secs_f64() / f64::from(steps);
        // The same step with the logits left on the device, as a device
        // sampler reads them: no full-vocab rows cross to the host.
        // Warmed, as above.
        let seated: Vec<Seated<'_>> = lanes.iter().copied().map(Seated::of).collect();
        for _ in 0..2 {
            shell.fire_kept(&seated).expect("a decode fires");
        }
        // Fires are enqueued, not awaited: the last one's readout, read once
        // at the end, waits for them all (its one download is spread over
        // the steps).
        let kept_steps = 22;
        let started = Instant::now();
        let mut last = None;
        for _ in 0..kept_steps {
            last = Some(shell.fire_kept(&seated).expect("a decode fires"));
        }
        if let Some(kept) = last.and_then(|fired| fired.kept) {
            kept.lane_rows(0).expect("the last readout reads");
        }
        let kept = started.elapsed().as_secs_f64() / f64::from(kept_steps);
        // `PIE_XLA_BENCH_PROFILE=<dir>`: a device trace of 8 more kept
        // steps, as `<dir>/w<width>.xplane.pb` (read it with
        // `jax.profiler.ProfileData`).
        if let Some(dir) = std::env::var_os("PIE_XLA_BENCH_PROFILE") {
            let api = shell.device().api().expect("a device").clone();
            let profiler =
                engine_xla::pjrt::profiler::Profiler::start(&api).expect("a trace starts");
            let mut last = None;
            for _ in 0..8 {
                last = Some(shell.fire_kept(&seated).expect("a decode fires"));
            }
            if let Some(kept) = last.and_then(|fired| fired.kept) {
                kept.lane_rows(0).expect("the last readout reads");
            }
            let space = profiler.finish().expect("the trace collects");
            let path = std::path::Path::new(&dir).join(format!("w{width}.xplane.pb"));
            std::fs::write(&path, space).expect("the trace writes");
        }
        eprintln!(
            "width {width:>3}: {:.2} ms/step, {:.0} tok/s; logits kept on the device {:.2} ms/step",
            per * 1e3,
            f64::from(width) / per,
            kept * 1e3
        );
    }
}
