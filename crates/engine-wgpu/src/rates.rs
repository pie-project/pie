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
    let chip = tuning.profile().clone();
    // What the profile states stands; a load times only what it lacks.
    let stated = chip.rates();
    let measured = device.chip();
    let save = profiles.map(|profiles| engine::rates::SaveTo {
        path: crate::chip::SHELF.user_path(profiles, &measured.name),
        provenance: provenance(&measured.name),
        registry: crate::chip::registry(),
    });
    let keys = engine_gpu::rates::keys_at(max_submit);
    let lacked = stated.missing(&keys);
    if lacked.is_empty() {
        tracing::info!(
            "rates: every [rates] key the load weighs by is stated ({}); none measured",
            keys.join(", ")
        );
    } else {
        tracing::info!(
            "rates: {} {}: no layer states them, or the user's file and the committed profile \
             store them under no matching key (the stored measurements' lines say which)",
            match measuring {
                engine::rates::Measuring::Time => "measuring",
                engine::rates::Measuring::Skip => "skipping (`[engine] measure = false`)",
            },
            lacked.join(", ")
        );
    }
    let chip = engine::rates::resolve_at_load(
        chip,
        &keys,
        save,
        Some(device.left()),
        max_submit,
        measuring,
        // A stale key's figure serves until the export round or `pie tune`
        // measures it: its step needs device buffers of its own, which
        // nothing lays after ready, so it is not timed behind serving.
        engine::rates::StaleRates::Served,
        |max_submit, stated| measure(device, &tuning, max_submit, stated).map_err(|e| e.to_string()),
    )
    .map_err(|why| engine_gpu::error::Fault::Device {
        call: "rates at load",
        why,
    })?;
    let tuning = crate::tuning::Tuning::new(chip).with_workgroup_bytes(device.workgroup_bytes());
    let resident = match (tuning.resident(), measuring) {
        (Some(resident), _) => {
            tracing::info!(
                resident,
                "the workgroups the device holds at once, as its profile states them"
            );
            resident
        }
        (None, engine::rates::Measuring::Skip) => {
            // A device fact no driver states as such: timed (milliseconds,
            // stored nowhere), never defaulted.
            let (resident, ms) = resident_for_skip(device, &tuning, max_submit)?;
            let resident = resident.round().clamp(1.0, f64::from(u32::MAX)) as u32;
            tracing::info!(
                resident,
                ms,
                "rates: `[rates] {}` timed despite measure = false (no layer states it, and the \
                 device states no count a guest step's grid can stride over): a {}-row matmul of \
                 width {} against one workgroup of it, single fires; stored nowhere",
                profile::fire::RESIDENT_WORKGROUPS,
                engine_gpu::rates::RESIDENT_ROWS,
                engine_gpu::rates::RESIDENT_SHAPE
            );
            resident
        }
        (None, engine::rates::Measuring::Time) => {
            return Err(engine_gpu::error::Fault::Device {
                call: "rates at load",
                why: "the profile states no [rates] resident_workgroups after the rates step"
                    .into(),
            });
        }
    };
    device.hold_resident(resident);
    if tuning.resident().is_none() && measuring == engine::rates::Measuring::Skip {
        let mut tuned = tuning.profile().tuned().clone();
        tuned.timed.push(profile::fire::RESIDENT_WORKGROUPS.to_string());
        return Ok(tuning.with_profile(tuning.profile().with_tuned(tuned)));
    }
    Ok(tuning)
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
    stated: &profile::fire::Rates,
) -> Result<engine::rates::Measured> {
    measure_saying(device, tuning, max_submit, stated, &mut |_| {})
}

/// [`measure`], saying how the warm-up ended and what each shape reached.
pub fn measure_saying(
    device: &Context,
    tuning: &crate::tuning::Tuning,
    max_submit: Option<std::time::Duration>,
    stated: &profile::fire::Rates,
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
    rates: &profile::fire::Rates,
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
    let rates = profile::fire::Rates {
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::stated_submit_bound;

    /// A load's rates step measures under the platform's stated bound (the
    /// Rig's admission refuses a frame estimated past it): KGSL's 2 s on
    /// Android; on Windows the timeout detection's `TdrDelay` (2 where the
    /// registry states none, none where detection is off); none elsewhere.
    #[test]
    fn a_load_measures_under_the_platforms_stated_bound() {
        let two = Some(Duration::from_secs(2));
        assert_eq!(engine_gpu::rates::tdr_bound(None, None), two);
        assert_eq!(
            engine_gpu::rates::tdr_bound(Some(3), Some(8)),
            Some(Duration::from_secs(8))
        );
        assert_eq!(engine_gpu::rates::tdr_bound(Some(0), Some(8)), None);
        let admission = engine::rates::Admission::new(two);
        assert!(admission.fits(Some(1.5e9)));
        assert!(!admission.fits(Some(2.5e9)));
        if cfg!(target_os = "android") {
            assert_eq!(stated_submit_bound(), two);
        } else if !cfg!(windows) {
            assert_eq!(stated_submit_bound(), None);
        }
    }
}
