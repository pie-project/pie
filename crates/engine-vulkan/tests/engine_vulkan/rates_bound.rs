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
    // a desktop GPU's: a launch short of the L40S's smallest fire, so no
    // timed point lands below it), the attention prefix curve measured
    // under the bound.
    let stated = Rates {
        launch_ns: Some(1000),
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
        None,
        &stated,
        &wanted,
        None,
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
        None,
        &stated,
        &wanted,
        None,
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

/// An output head (151936 x 1024) under a bound its whole fire's estimate
/// at a row passes (10 ms: grown from the probe's first, cold fire, the
/// head's 311 MB are estimated far past it), as on the Adreno, where every
/// head curve stopped at 1 row: its fires are timed on slices of its
/// columns the bound admits, their times carried to the whole plane, so
/// the head's curves hold points instead. The admission's worst ratio is
/// pinned (`measure_shapes`'s `worst`), so the slices admitted follow the
/// slicing, not the box's clock.
#[test]
fn a_head_past_the_bound_is_timed_on_slices_of_its_columns() {
    let Ok(device) = Context::bind(&crate::gpu::device_boot(), engine::Allowance::WHOLE) else {
        eprintln!("not asked: no device");
        return;
    };
    let bound = std::time::Duration::from_millis(10);
    let (chip, _) = engine_vulkan::chip::resolve(&device.chip(), None);
    let tuning = engine_vulkan::tuning::Tuning::new(chip, device.limits());
    let tier = <Api as engine_gpu::serve::Serve>::tier(&device);
    let timer = |frame: Frame| Ok((None, (frame.commit_timed()? * 1e9) as u64));
    let stated = Rates {
        launch_ns: Some(4000),
        hbm_gbps: Some(600.0),
        ..Rates::default()
    };
    let top = 4;
    let shapes = [(151_936, 1024)];
    let kernels =
        engine_gpu::rates::dense_kernels(kernels_slang::Target::Spirv, &tuning, tier, top);
    let wanted = engine_gpu::rates::lacking(
        kernels_slang::Target::Spirv,
        &kernels,
        &stated,
        &shapes,
        top,
    );
    let curves = engine_gpu::rates::measure_shapes::<Api, _>(
        &device,
        &tuning,
        tier,
        &timer,
        Some(bound),
        Some(1.0),
        &stated,
        &wanted,
        None,
        top,
        &mut |line| eprintln!("{line}"),
    )
    .expect("the head's step lands under the bound");
    let timed = curves
        .iter()
        .filter(|c| c.curve.first().is_some_and(|&(rows, _)| rows == 1));
    assert!(
        timed.count() > 0,
        "no head curve timed at one row: {curves:?}"
    );
}

/// A settle's plane-alone weighing under a stated bound smaller than one
/// whole-plane fire (an output head's at 256 rows, tens of ms on a desktop
/// GPU, against a 4 ms bound): its fires are cut into pieces of the plane's
/// columns, each a submission of its own, and no submission's time on the
/// device passes the bound, while every kernel still gets its samples.
#[test]
fn a_plane_weighed_past_the_bound_is_cut_into_pieces_each_within_it() {
    let Ok(device) = Context::bind(&crate::gpu::device_boot(), engine::Allowance::WHOLE) else {
        eprintln!("not asked: no device");
        return;
    };
    let bound = std::time::Duration::from_millis(4);
    let (chip, _) = engine_vulkan::chip::resolve(&device.chip(), None);
    let tuning = engine_vulkan::tuning::Tuning::new(chip, device.limits());
    let tier = <Api as engine_gpu::serve::Serve>::tier(&device);
    let longest = std::cell::Cell::new(0u64);
    let timer = |frame: Frame| {
        let started = std::time::Instant::now();
        let span = frame.commit_spanned()?;
        let ns = span.unwrap_or(started.elapsed().as_nanos() as u64);
        longest.set(longest.get().max(ns));
        Ok((span, started.elapsed().as_nanos() as u64))
    };
    let (n, k, rows) = (151_936, 1024, 256);
    let kernels: Vec<kernels_slang::linear::gemm::Tactic> = [1u32, 8]
        .iter()
        .map(|&vector| kernels_slang::linear::gemm::Tactic {
            vector,
            ..engine_gpu::backend::Tactics::gemm(&tuning, tier, rows)
        })
        .collect();
    let samples = engine_gpu::rates::time_kernels::<Api, _>(
        &device,
        &tuning,
        tier,
        &timer,
        Some(bound),
        1000.0,
        &kernels,
        &[1, 1],
        (n, k),
        rows,
    )
    .expect("the weighing lands under the bound");
    assert!(
        samples.iter().all(|s| !s.is_empty()),
        "a kernel got no sample: {samples:?}"
    );
    // Carried to the whole plane, a fire is past the bound: it was cut.
    assert!(
        samples
            .iter()
            .flatten()
            .any(|&ns| ns > bound.as_nanos() as f64),
        "no fire needed cutting: {samples:?}"
    );
    assert!(
        longest.get() <= bound.as_nanos() as u64,
        "a submission ran {:.1} ms on the device, past the {:.1} ms bound",
        longest.get() as f64 / 1e6,
        bound.as_secs_f64() * 1e3
    );
}

/// A model of a smaller top rung measured first, then one of a larger top
/// sharing its shapes: the second extends each shared curve with the rows
/// it lacks and never times a stored point again, so the first model's
/// points (what its settle and its cover were weighed by) are unchanged.
#[test]
fn a_later_model_extends_a_shared_curve_and_keeps_its_points() {
    let Ok(device) = Context::bind(&crate::gpu::device_boot(), engine::Allowance::WHOLE) else {
        eprintln!("not asked: no device");
        return;
    };
    let (chip, _) = engine_vulkan::chip::resolve(&device.chip(), None);
    let tuning = engine_vulkan::tuning::Tuning::new(chip, device.limits());
    let tier = <Api as engine_gpu::serve::Serve>::tier(&device);
    let timer = |frame: Frame| {
        let started = std::time::Instant::now();
        let span = frame.commit_spanned()?;
        Ok((span.filter(|&ns| ns > 0), started.elapsed().as_nanos() as u64))
    };
    let say = &mut |_: &str| {};
    let stated = Rates {
        launch_ns: Some(1000),
        ..Rates::default()
    };
    let shapes = [(1024, 1024)];
    let measure = |stated: &Rates, top: u32| {
        let kernels =
            engine_gpu::rates::dense_kernels(kernels_slang::Target::Spirv, &tuning, tier, top);
        let wanted =
            engine_gpu::rates::lacking(kernels_slang::Target::Spirv, &kernels, stated, &shapes, top);
        let curves = engine_gpu::rates::measure_shapes::<Api, _>(
            &device, &tuning, tier, &timer, None, None, stated, &wanted, None, top, say,
        )
        .expect("the curves are timed");
        Rates {
            kernel_tflops_by_shape: Some(curves),
            ..Rates::default()
        }
        .over(stated.clone())
    };
    let first = measure(&stated, 16);
    let second = measure(&first, 64);
    for a in first.kernel_tflops_by_shape.iter().flatten().filter(|c| c.stop.is_none()) {
        let b = second
            .kernel_tflops_by_shape
            .iter()
            .flatten()
            .find(|b| b.id() == a.id())
            .expect("the shared curve stands");
        assert_eq!(
            &b.curve[..a.curve.len()],
            &a.curve[..],
            "{}: the first model's points are kept as stored",
            a.name()
        );
        assert!(
            b.curve.last().map(|p| p.0) > a.curve.last().map(|p| p.0) || b.stop.is_some(),
            "{}: extended past the first model's rows, or stopped",
            a.name()
        );
    }
}
