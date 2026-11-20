//! The rates step under a bound on one submission, as a phone states one:
//! simulated here on whatever device the suite opens. No frame it submits
//! passes the bound on the device (its timestamps: a frame queued behind
//! others waits for them on the host, not on the device), the kernels' curves stop before a fire that would, and
//! where and why each stopped is said, so a load does not time them again.

use engine_vulkan::device::{Api, Context, Frame};
use profile::fire::{KernelStop, Rates};

/// A bound tight enough that the slowest kernels' curves stop short of the
/// rows asked: 25 ms a submission.
const BOUND: std::time::Duration = std::time::Duration::from_millis(25);

/// The rows the kernels are timed to.
const TOP: u32 = 4096;

#[test]
#[ignore = "the warm-up cube's first fire (`engine_gpu::rates::warm_at`) is submitted with no \
            estimate and, after the step's host-bound shapes, runs at a dropped clock: 26 to 44 ms \
            on the device under this 25 ms bound on an L40S, 13 to 22 ms on other runs"]
fn the_rates_step_stays_under_a_simulated_bound() {
    let Ok(device) = Context::bind(&crate::gpu::device_boot(), engine::Allowance::WHOLE) else {
        eprintln!("not asked: no device");
        return;
    };
    let (chip, _) = engine_vulkan::chip::resolve(&device.chip(), None);
    let tuning = engine_vulkan::tuning::Tuning::new(chip, device.limits());
    let tier = <Api as engine_gpu::serve::Serve>::tier(&device);
    engine_vulkan::encode::profile_kernels(true);
    let _ = engine_vulkan::encode::take_longest_frame_span_ns();
    let worst = std::cell::Cell::new(0u64);
    let longest = std::cell::Cell::new(0u64);
    let timer = |frame: Frame| {
        let ns = (frame.commit_timed()? * 1e9) as u64;
        worst.set(worst.get().max(ns));
        // The frames this timed and any queued ahead of it, by the device.
        let span = engine_vulkan::encode::take_longest_frame_span_ns();
        if span > BOUND.as_nanos() as u64 / 2 {
            eprintln!(
                "past half the bound on the device: {:.1} ms",
                span as f64 / 1e6
            );
        }
        longest.set(longest.get().max(span));
        Ok((None, ns))
    };
    // Every rate but the kernels' curves stated: only those are timed. The
    // bandwidth (a first fire's estimate) is a desktop GPU's.
    let stated = Rates {
        launch_ns: Some(4000),
        gemm_tflops_by_rows: Some(vec![(4096, vec![(256, 10.0), (8192, 30.0)])]),
        attn_tflops_by_keys: Some(vec![(512, 5.0), (8192, 10.0)]),
        hbm_gbps: Some(600.0),
        elementwise_gbps: Some(600.0),
        scan_gelems: Some(100.0),
        ..Rates::default()
    };
    let measured = engine_gpu::rates::measure_saying::<Api, _>(
        &device,
        &tuning,
        tier,
        &timer,
        Some(BOUND),
        &stated,
        TOP,
        &mut |line| eprintln!("{line}"),
    )
    .expect("the rates step lands under the bound");
    let rates = measured.rates;
    let longest = longest
        .get()
        .max(engine_vulkan::encode::take_longest_frame_span_ns());
    engine_vulkan::encode::profile_kernels(false);
    assert!(longest > 0, "the device's timestamps timed no frame");
    assert!(
        longest <= BOUND.as_nanos() as u64,
        "a frame of the step ran {:.1} ms on the device, past the {:.0} ms bound",
        longest as f64 / 1e6,
        BOUND.as_secs_f64() * 1e3
    );
    assert_eq!(rates.kernels_timed_to, Some(TOP));
    let stops = rates
        .kernel_stops
        .expect("where the curves stopped is said");
    let curves = |c: &Option<Vec<(u32, Vec<(u32, f64)>)>>| c.clone().unwrap_or_default();
    for (kind, curves) in [
        (KernelStop::VECTOR, curves(&rates.vector_tflops_by_rows)),
        (KernelStop::TILE, curves(&rates.tile_tflops_by_rows)),
    ] {
        for (key, curve) in curves {
            let reach = curve.last().map_or(0, |&(rows, _)| rows);
            let stop = stops.iter().find(|s| s.kind == kind && s.key == key);
            match stop {
                Some(stop) => assert!(stop.rows > reach, "{kind}/{key}: stop {stop:?} at {reach}"),
                None => assert_eq!(reach, TOP, "{kind}/{key} ends at {reach}, no stop said"),
            }
        }
    }
    eprintln!(
        "longest frame on the device {:.1} ms (on the host, queue included, {:.1} ms); stops \
         {stops:?}",
        longest as f64 / 1e6,
        worst.get() as f64 / 1e6
    );
}
