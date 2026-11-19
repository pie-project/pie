//! `pie tune wgpu --check-estimate` and `--rates`: the check's loads of a
//! model ([`engine_gpu::check::Bench`]) and the rates step alone.

use std::path::Path;
use std::time::Instant;

use crate::device::{Api, Context};

/// The check's loads of `deployment` from `artifact` on adapter 0, under
/// the user's profiles in `profiles`; a walked fire dissected by the
/// frame's per-dispatch timestamps, where the adapter writes them.
pub fn bench(
    profiles: &Path,
    artifact: &Path,
    deployment: &str,
    contract_for: engine_gpu::api::ContractFor,
) -> Result<engine_gpu::check::Bench<Api>, String> {
    engine_gpu::check::Bench::new(
        artifact,
        deployment,
        poem_ir::Platform::Wgpu,
        contract_for,
        crate::api::DeviceBoot {
            profiles: Some(profiles.to_path_buf()),
            ..crate::api::DeviceBoot::default()
        },
        engine_gpu::check::Probe {
            timing: crate::encode::profile_timing,
            reset: crate::encode::reset_kernel_profile,
            rows: crate::encode::kernel_profile,
        },
    )
}

/// `pie tune wgpu --rates`: adapter 0's `[rates]` measured, every one, and
/// its user file under `profiles` with them in place of its own.
pub fn rated(profiles: &Path, say: &mut dyn FnMut(&str)) -> Result<engine::rates::Rated, String> {
    let began = Instant::now();
    let device = Context::bind(&crate::api::DeviceBoot::default(), engine::Allowance::WHOLE)
        .map_err(|e| e.to_string())?;
    let chip = device.chip();
    let (profile, _) = crate::chip::resolve(&chip, Some(profiles));
    let tuning = crate::tuning::Tuning::new(profile, device.cores())
        .with_workgroup_bytes(device.workgroup_bytes());
    say(&format!("measuring the [rates] of {}", chip.name));
    let measured = crate::rates::measure_saying(
        &device,
        &tuning,
        crate::rates::STATED_SUBMIT_BOUND,
        &profile::fire::Rates::default(),
        say,
    )
    .map_err(|e| e.to_string())?;
    if measured.unsettled {
        say("a rate's frames stopped at the device's bound on one submission before it settled");
    }
    let existing = std::fs::read_to_string(crate::chip::SHELF.user_path(profiles, &chip.name)).ok();
    Ok(engine_gpu::rates::rated(
        &chip.name,
        existing.as_deref(),
        &measured.rates,
        &crate::rates::provenance(&chip.name),
        began.elapsed().as_secs_f64(),
    ))
}
