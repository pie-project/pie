//! `pie tune wgpu`: the `[rates]` step alone (the device's rates measured
//! on it, [`crate::rates::measure`]); wgpu has no decision search yet.

use std::path::Path;
use std::time::Instant;

use profile::Profile;

pub struct Outcome {
    pub device: String,
    /// The complete profile, ready for the user's shelf.
    pub file: String,
    pub seconds: f64,
}

/// Measure the first adapter's rates and lay them over the profile it runs.
pub fn rates(profiles: Option<&Path>, say: &mut dyn FnMut(&str)) -> Result<Outcome, String> {
    let began = Instant::now();
    let device =
        crate::device::Context::bind(&crate::api::DeviceBoot::default(), engine::Allowance::WHOLE)
            .map_err(|e| e.to_string())?;
    let chip = device.chip();
    let (base, report) = crate::chip::resolve(&chip, profiles);
    say(&format!("{}: starting from {}", chip.name, report.used));
    let tuning = crate::tuning::Tuning::new(base.clone(), device.cores());
    let r = crate::rates::measure(&device, &tuning).map_err(|e| e.to_string())?;
    for &(m, k, rate) in r
        .bf16_flops_per_ns_by_rows_and_depth
        .as_deref()
        .unwrap_or(&[])
    {
        say(&format!(
            "[rates]: matmul m {m:>5} k {k:>5}: {:6.1} TFLOP/s",
            rate / 1e3
        ));
    }
    say(&format!(
        "[rates]: attention {:.1} TFLOP/s, scan {:.2} G state elements/s, {:.0} GB/s, launch \
         {:.2} us",
        r.attn_at(16384, 128).unwrap_or(0.0) / 1e3,
        r.scan_elements_per_ns.unwrap_or(0.0),
        r.bytes_per_ns.unwrap_or(0.0),
        r.launch_ns.unwrap_or(0) as f64 / 1e3
    ));
    let tuned = engine::rates::tuned(&r);
    let seconds = began.elapsed().as_secs_f64();
    let provenance = [
        (
            "source",
            "\"pie tune wgpu --rates on this machine\"".to_string(),
        ),
        engine::rates::provenance(
            "wgpu",
            "the matmul through the profile's path, the adapter's timestamps where it has them; operands: engine::rates' pattern",
        ),
        ("from", format!("\"{}\"", report.used)),
        ("pie", format!("\"{}\"", env!("CARGO_PKG_VERSION"))),
        ("seconds", format!("{seconds:.0}")),
    ];
    let file = profile::search::assemble(
        crate::chip::registry(),
        &base,
        &chip.name,
        &chip.facts(),
        &tuned,
        &provenance,
        &[],
    )?;
    Ok(Outcome {
        device: chip.name,
        file,
        seconds,
    })
}

/// The adapter this machine would serve, by name, with its facts.
#[must_use]
pub fn probe() -> Option<(String, profile::Facts)> {
    let device =
        crate::device::Context::bind(&crate::api::DeviceBoot::default(), engine::Allowance::WHOLE)
            .ok()?;
    let chip = device.chip();
    Some((chip.name.clone(), chip.facts()))
}

/// The profile a load on the adapter named `name` runs under.
#[must_use]
pub fn resolve_facts(
    name: &str,
    facts: &profile::Facts,
    profiles: Option<&Path>,
) -> (Profile, profile::Report) {
    crate::chip::SHELF.resolve(crate::chip::registry(), name, facts, profiles)
}
