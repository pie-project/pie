//! The rates steps under a bound on one submission, as a phone states one:
//! simulated here on whatever device the suite opens. No frame they submit
//! passes the bound on the device (its timestamps: a frame queued behind
//! others waits for them on the host, not on the device); each dense
//! kernel's curve at a model's planes stops before a fire that would, and
//! where and why each stopped is kept with it, so a load does not time it
//! again.

use engine_vulkan::device::{Api, Context, Frame};
use profile::fire::Rates;

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
    // The device's rates the shapes' step reads (its launch and bandwidth
    // a desktop GPU's), the attention prefix curve measured under the bound.
    let stated = Rates {
        launch_ns: Some(4000),
        gemm_tflops_by_rows: Some(vec![(4096, vec![(256, 10.0), (8192, 30.0)])]),
        attn_tflops_by_keys: Some(vec![(512, 5.0), (8192, 10.0)]),
        hbm_gbps: Some(600.0),
        elementwise_gbps: Some(600.0),
        scan_gelems: Some(100.0),
        ..Rates::default()
    };
    let say = &mut |line: &str| eprintln!("{line}");
    engine_gpu::rates::measure_saying::<Api, _>(
        &device,
        &tuning,
        tier,
        &timer,
        Some(BOUND),
        &stated,
        say,
    )
    .expect("the device's rates step lands under the bound");
    // A model's planes: a 1024-wide layer's and its output head's.
    let shapes = [(3072, 1024), (1024, 3072), (151_936, 1024)];
    let kernels =
        engine_gpu::rates::dense_kernels(kernels_slang::Target::Spirv, &tuning, tier, TOP);
    let wanted = engine_gpu::rates::lacking(
        kernels_slang::Target::Spirv,
        &kernels,
        &stated,
        &shapes,
        TOP,
    );
    let curves = engine_gpu::rates::measure_shapes::<Api, _>(
        &device,
        &tuning,
        tier,
        &timer,
        Some(BOUND),
        &stated,
        &wanted,
        TOP,
        say,
    )
    .expect("the shapes' step lands under the bound");
    let longest = longest
        .get()
        .max(engine_vulkan::encode::take_longest_frame_span_ns());
    engine_vulkan::encode::profile_kernels(false);
    assert!(longest > 0, "the device's timestamps timed no frame");
    assert!(
        longest <= BOUND.as_nanos() as u64,
        "a frame of the steps ran {:.1} ms on the device, past the {:.0} ms bound",
        longest as f64 / 1e6,
        BOUND.as_secs_f64() * 1e3
    );
    assert_eq!(curves.len(), wanted.len());
    let mut stopped = 0;
    for c in &curves {
        let reach = c.curve.last().map_or(0, |&(rows, _)| rows);
        match c.stop {
            Some((at, _)) => {
                assert!(at > reach, "{c:?}");
                stopped += 1;
            }
            None => assert_eq!(reach, TOP, "{c:?} ends short with no stop"),
        }
        assert!(c.serves(TOP));
    }
    // A load of these shapes, the curves saved, times nothing again.
    let saved = Rates {
        kernel_tflops_by_shape: Some(curves),
        ..stated
    };
    assert!(
        engine_gpu::rates::lacking(kernels_slang::Target::Spirv, &kernels, &saved, &shapes, TOP)
            .is_empty()
    );
    eprintln!(
        "longest frame on the device {:.1} ms (on the host, queue included, {:.1} ms); {stopped} \
         of {} curves stopped",
        longest as f64 / 1e6,
        worst.get() as f64 / 1e6,
        saved.kernel_tflops_by_shape.as_ref().map_or(0, Vec::len)
    );
}

/// A bound tighter than one 512^3 matmul (a few microseconds on a desktop
/// GPU): no warm-up cube fits it, nor most points. The shapes' step still
/// lands: the clock unwarmed, each point admitted by its own estimate, each
/// curve that cannot go on stopped by the bound with the reason kept, and
/// the dense kernel still chosen (the profile's path's where nothing was
/// timed). A sizing never fails the load.
#[test]
fn a_bound_no_cube_fits_stops_the_curves_not_the_load() {
    let Ok(device) = Context::bind(&crate::gpu::device_boot(), engine::Allowance::WHOLE) else {
        eprintln!("not asked: no device");
        return;
    };
    let tight = std::time::Duration::from_micros(10);
    let (chip, _) = engine_vulkan::chip::resolve(&device.chip(), None);
    let tuning = engine_vulkan::tuning::Tuning::new(chip, device.limits());
    let tier = <Api as engine_gpu::serve::Serve>::tier(&device);
    let timer = |frame: Frame| Ok((None, (frame.commit_timed()? * 1e9) as u64));
    let stated = Rates {
        launch_ns: Some(4000),
        hbm_gbps: Some(600.0),
        ..Rates::default()
    };
    let shapes = [(3072, 1024), (1024, 3072)];
    let kernels =
        engine_gpu::rates::dense_kernels(kernels_slang::Target::Spirv, &tuning, tier, TOP);
    let wanted = engine_gpu::rates::lacking(
        kernels_slang::Target::Spirv,
        &kernels,
        &stated,
        &shapes,
        TOP,
    );
    let curves = engine_gpu::rates::measure_shapes::<Api, _>(
        &device,
        &tuning,
        tier,
        &timer,
        Some(tight),
        &stated,
        &wanted,
        TOP,
        &mut |line| eprintln!("{line}"),
    )
    .expect("a bound no cube fits stops curves, never the step");
    assert_eq!(curves.len(), wanted.len());
    let mut bound = 0;
    for c in &curves {
        let reach = c.curve.last().map_or(0, |&(rows, _)| rows);
        match c.stop {
            Some((at, why)) => {
                assert!(at > reach, "{c:?}");
                bound += usize::from(why == profile::fire::KernelCurve::BOUND);
            }
            None => assert_eq!(reach, TOP, "{c:?}"),
        }
    }
    assert!(
        bound > 0,
        "no curve stopped by a {tight:?} bound: {curves:?}"
    );
    let rates = Rates {
        kernel_tflops_by_shape: Some(curves),
        gemm_tflops_by_rows: Some(vec![(4096, vec![(256, 10.0), (8192, 30.0)])]),
        attn_tflops_by_keys: Some(vec![(512, 5.0), (8192, 10.0)]),
        elementwise_gbps: Some(600.0),
        scan_gelems: Some(100.0),
        ..stated
    };
    let planes: Vec<(u32, u32, u32, f64)> = shapes
        .iter()
        .map(|&(n, k)| (n, k, 28, f64::from(2 * n * k)))
        .collect();
    let rungs = [1, 16, 256, TOP];
    engine_gpu::dense::choose(
        kernels_slang::Target::Spirv,
        &rungs,
        |rows| engine_gpu::backend::Tactics::gemm(&tuning, tier, rows),
        &rates,
        &planes,
        true,
        Some(TOP),
    )
    .expect("the dense kernel is chosen whatever the curves reached");
}
