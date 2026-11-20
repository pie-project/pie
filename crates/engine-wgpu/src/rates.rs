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
    let chip = engine::rates::rates_at_load(
        chip,
        &engine_gpu::rates::keys_at(max_submit),
        save,
        Some(device.left()),
        max_submit,
        |max_submit| measure(device, &tuning, max_submit, &stated).map_err(|e| e.to_string()),
    )
    .map_err(|why| engine_gpu::error::Fault::Device {
        call: "rates at load",
        why,
    })?;
    Ok(crate::tuning::Tuning::new(chip, device.cores())
        .with_workgroup_bytes(device.workgroup_bytes()))
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
    let timer = |frame: Frame| {
        crate::encode::reset_kernel_profile();
        let host = frame.commit_timed()?;
        let span = crate::encode::kernel_profile()
            .into_iter()
            .find(|(name, ..)| name == "frame.span")
            .map(|(_, ns, _)| ns)
            .filter(|&ns| ns > 0);
        Ok((span, (host * 1e9) as u64))
    };
    engine_gpu::rates::measure_saying::<Api, _>(
        device, tuning, tier, &timer, max_submit, stated, say,
    )
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
