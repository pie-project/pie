//! The device's rates on wgpu ([`engine_gpu::rates::measure`]): the
//! matmul and the copy timed by the adapter's timestamps (the frame's span)
//! where it grants them, else by the host's clock around submit and wait.
//! A browser's load calls it (its adapter is masked, so no profile states
//! its rates); so does `pie tune wgpu --rates`.

use engine_gpu::error::Result;

use crate::device::{Api, Context, Frame};

/// The `[provenance]` entry a measurement of `device`'s rates writes beside
/// them (`engine::rates::provenance`): how wgpu measured them.
#[must_use]
pub fn provenance(device: &str) -> (&'static str, String) {
    engine::rates::provenance(
        "wgpu",
        &format!(
            "{device}; the matmul through the profile's path, the adapter's timestamps where it \
             has them; operands: engine::rates' pattern; at {}",
            engine::rates::build_commit()
        ),
    )
}

/// The bound the platform states on one GPU submission, the same the
/// native Vulkan engine states ([`engine_gpu::rates::stated_submit_bound`]):
/// wgpu submits through the same driver, whose watchdog loses the device
/// past it.
pub use engine_gpu::rates::stated_submit_bound;

/// `tuning` with every roofline constant stated
/// (`engine::rates::rates_at_load`, every engine's rule): as its layers
/// state them where they state them all; else `device`'s own, measured
/// before the load is ready, each timed frame under `max_submit` where the
/// device states one, saved to the user's profile under `profiles` (a
/// browser, its adapter masked and no profiles directory, keeps them for
/// its load alone).
pub fn at_load(
    device: &Context,
    tuning: crate::tuning::Tuning,
    profiles: Option<&std::path::Path>,
    max_submit: Option<std::time::Duration>,
    measuring: engine::rates::Measuring,
) -> Result<crate::tuning::Tuning> {
    // What the profile states stands; a load times only what it lacks.
    let measured = device.chip();
    let save = profiles.map(|profiles| engine::rates::SaveTo {
        path: crate::chip::SHELF.user_path(profiles, &measured.name),
        provenance: provenance(&measured.name),
        registry: crate::chip::registry(),
    });
    let (chip, resident) = engine_gpu::rates::resolve_at_load(
        tuning.profile().clone(),
        save,
        device.left(),
        max_submit,
        measuring,
        |max_submit, stated| measure(device, &tuning, max_submit, stated),
        || resident_for_skip(device, &tuning, max_submit),
    )?;
    device.hold_resident(resident);
    Ok(crate::tuning::Tuning::new(chip).with_workgroup_bytes(device.workgroup_bytes()))
}

/// `[rates] resident_workgroups` timed where a load skips measuring and no
/// layer states it (`engine_gpu::rates::resident_for_skip`): the
/// workgroups, and the ms it took.
fn resident_for_skip(
    device: &Context,
    tuning: &crate::tuning::Tuning,
    max_submit: Option<std::time::Duration>,
) -> Result<(f64, u128)> {
    let tier = <Api as engine_gpu::serve::Serve>::tier(device);
    let timer = timer();
    engine_gpu::rates::resident_for_skip::<Api, _>(device, tuning, tier, &timer, max_submit)
}

/// The rates of `device` that `stated` lacks (every one where it states
/// none), under the tactics `tuning` picks.
pub fn measure(
    device: &Context,
    tuning: &crate::tuning::Tuning,
    max_submit: Option<std::time::Duration>,
    stated: &profile::rates::Rates,
) -> Result<engine::rates::Measured> {
    measure_saying(device, tuning, max_submit, stated, &mut |_| {})
}

/// [`measure`], saying how the warm-up ended and what each shape reached.
pub fn measure_saying(
    device: &Context,
    tuning: &crate::tuning::Tuning,
    max_submit: Option<std::time::Duration>,
    stated: &profile::rates::Rates,
    say: &mut dyn FnMut(&str),
) -> Result<engine::rates::Measured> {
    let tier = <Api as engine_gpu::serve::Serve>::tier(device);
    let timer = timer();
    let mut measured = engine_gpu::rates::measure_saying::<Api, _>(
        device, tuning, tier, &timer, max_submit, stated, say,
    )?;
    // Stated as measured on this adapter (`engine_gpu::stored`).
    measured.rates.measured_on = Some(device.measured_on());
    Ok(measured)
}

/// `rates` saved into the user's profile under `profiles` for this device
/// (`engine::rates::save_rates`), where a load names a profile directory.
pub fn save_rates(
    device: &Context,
    profiles: Option<&std::path::Path>,
    rates: &profile::rates::Rates,
) -> Result<()> {
    let Some(profiles) = profiles else {
        return Ok(());
    };
    let measured = device.chip();
    let save = engine::rates::SaveTo {
        path: crate::chip::SHELF.user_path(profiles, &measured.name),
        provenance: provenance(&measured.name),
        registry: crate::chip::registry(),
    };
    let rates = profile::rates::Rates {
        measured_on: Some(device.measured_on()),
        ..rates.clone()
    };
    engine::rates::save_rates(&save, &rates)
        .map(drop)
        .map_err(|why| engine_gpu::Fault::Device {
            call: "rates at load",
            why,
        })
}

/// `tuning` with each dense kernel's curve at each of the model's `shapes`
/// to `top_rows` (`engine_gpu::rates::shapes_at_load`), saved to the user's
/// profile under `profiles`, each timed frame under `max_submit` where the
/// device states one.
#[allow(clippy::too_many_arguments)]
pub fn shapes_at_load(
    device: &Context,
    tuning: crate::tuning::Tuning,
    profiles: Option<&std::path::Path>,
    max_submit: Option<std::time::Duration>,
    shapes: &[(u32, u32)],
    model_bytes: u64,
    top_rows: u32,
    measuring: engine::rates::Measuring,
) -> Result<crate::tuning::Tuning> {
    let tier = <Api as engine_gpu::serve::Serve>::tier(device);
    // Each point timed by the device's two timestamps around its frame (a
    // spanned frame, written between passes, where the device enabled
    // timestamps): the host's wait swings a frame more than the kernels it
    // weighs apart.
    let timer = |frame: Frame| {
        let started = std::time::Instant::now();
        let span = frame.commit_spanned()?;
        Ok((
            span.filter(|&ns| ns > 0),
            started.elapsed().as_nanos() as u64,
        ))
    };
    let measured = device.chip();
    let save = profiles.map(|profiles| engine::rates::SaveTo {
        path: crate::chip::SHELF.user_path(profiles, &measured.name),
        provenance: provenance(&measured.name),
        registry: crate::chip::registry(),
    });
    let chip = engine_gpu::rates::shapes_at_load::<Api, _>(
        device,
        &tuning,
        tier,
        &timer,
        max_submit,
        tuning.profile().clone(),
        save,
        &device.measured_on(),
        shapes,
        Some(model_bytes),
        top_rows,
        measuring,
    );
    // What the load held of the adapter stays (the decodes it splits): a
    // load naming its kernels after its curves are timed names what a load
    // that read them names.
    Ok(tuning.with_profile(chip?))
}

/// How a rates frame is timed on wgpu: the device's span where it states
/// one, and the host's clock around submit and wait.
fn timer() -> impl Fn(Frame) -> Result<(Option<u64>, u64)> {
    |frame: Frame| {
        crate::encode::reset_kernel_profile();
        let host = frame.commit_timed()?;
        let span = crate::encode::kernel_profile()
            .into_iter()
            .find(|(name, ..)| name == "frame.span")
            .map(|(_, ns, _)| ns)
            .filter(|&ns| ns > 0);
        Ok((span, (host * 1e9) as u64))
    }
}
