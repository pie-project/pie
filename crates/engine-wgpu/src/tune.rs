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
    // The step's device takes what its own fires lay, no load's budget.
    let device = crate::device::Context::bind(
        &crate::api::DeviceBoot::default(),
        engine::Allowance {
            budget: Some(engine::rates::step_bytes()),
        },
    )
    .map_err(|e| e.to_string())?;
    let chip = device.chip();
    let (base, report) = crate::chip::resolve(&chip, profiles);
    say(&format!("{}: starting from {}", chip.name, report.used));
    let tuning = crate::tuning::Tuning::new(base.clone(), device.cores());
    let r = crate::rates::measure_saying(&device, &tuning, say).map_err(|e| e.to_string())?;
    say(&format!(
        "[rates]: matmul {:.1} TFLOP/s at its widest rows, attention {:.1} TFLOP/s at its \
         longest keys, per-row ops {:.0} GB/s, scan {:.2} G state elements/s, {:.0} GB/s, \
         launch {:.2} us",
        r.gemm_tflops_by_rows
            .as_ref()
            .and_then(|w| w.last())
            .and_then(|(_, c)| c.last())
            .map_or(0.0, |&(_, t)| t),
        r.attn_tflops_by_keys
            .as_ref()
            .and_then(|c| c.last())
            .map_or(0.0, |&(_, t)| t),
        r.elementwise_gbps.unwrap_or(0.0),
        r.scan_gelems.unwrap_or(0.0),
        r.hbm_gbps.unwrap_or(0.0),
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
            &format!(
                "the matmul through the profile's path, the adapter's timestamps where it has \
                 them; operands: engine::rates' pattern; at {}",
                engine::rates::build_commit()
            ),
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
