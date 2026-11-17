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
    let measured = device.chip();
    let check = |text: &str| {
        crate::chip::registry()
            .parse(text, "the measured rates", profile::Strict::User)
            .map(|_| ())
            .map_err(|e| e.to_string())
    };
    let save = profiles.map(|profiles| engine::rates::SaveTo {
        path: crate::chip::SHELF.user_path(profiles, &measured.name),
        provenance: provenance(&measured.name),
        check: &check,
    });
    let chip =
        engine::rates::rates_at_load(chip, save, Some(device.left()), max_submit, |max_submit| {
            measure(device, &tuning, max_submit).map_err(|e| e.to_string())
        })
        .map_err(|why| engine_gpu::error::Fault::Device {
            call: "rates at load",
            why,
        })?;
    Ok(crate::tuning::Tuning::new(chip, device.cores()))
}

/// The rates of `device`, under the tactics `tuning` picks.
pub fn measure(
    device: &Context,
    tuning: &crate::tuning::Tuning,
    max_submit: Option<std::time::Duration>,
) -> Result<engine::rates::Measured> {
    measure_saying(device, tuning, max_submit, &mut |_| {})
}

/// [`measure`], saying how the warm-up ended.
pub fn measure_saying(
    device: &Context,
    tuning: &crate::tuning::Tuning,
    max_submit: Option<std::time::Duration>,
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
    engine_gpu::rates::measure_saying::<Api, _>(device, tuning, tier, &timer, max_submit, say)
}
