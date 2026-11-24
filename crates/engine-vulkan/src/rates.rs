//! The device's rates on Vulkan ([`engine_gpu::rates::measure`]), every
//! fire timed by the host's clock around submit and wait: what
//! `pie tune vulkan --rates` states in the device's profile.

use engine_gpu::backend::Tactics as _;
use engine_gpu::error::Result;

use crate::device::{Api, Context, Frame};

/// The `[provenance]` entry a measurement of `device`'s rates writes beside
/// them (`engine::rates::provenance`): how Vulkan measured them.
#[must_use]
pub fn provenance(device: &str) -> (&'static str, String) {
    engine::rates::provenance(
        "vulkan",
        &format!(
            "{device}; the matmul through the profile's path, the host's clock around each \
             frame; operands: engine::rates' pattern; at {}",
            engine::rates::build_commit()
        ),
    )
}

/// The bound the platform states on one GPU submission
/// ([`engine_gpu::rates::stated_submit_bound`]).
pub use engine_gpu::rates::stated_submit_bound;

/// `tuning` with every roofline constant stated
/// (`engine::rates::rates_at_load`, every engine's rule): as its layers
/// state them where they state them all; else `device`'s own, measured
/// before the load is ready, each timed frame under `max_submit` where the
/// device states one, saved to the user's profile under `profiles`.
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
        // A stale key's figure serves; the engine times it again behind
        // serving (`engine_gpu::serve::Shell::measure_behind`).
        true,
        |max_submit, stated| measure(device, &tuning, max_submit, stated).map_err(|e| e.to_string()),
    )
    .map_err(|why| engine_gpu::error::Fault::Device {
        call: "rates at load",
        why,
    })?;
    let stated_resident = chip
        .rates()
        .resident_workgroups
        .map(|workgroups| workgroups.round().clamp(1.0, f64::from(u32::MAX)) as u32);
    let resident = match (stated_resident, measuring) {
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
    let chip = if stated_resident.is_none() && measuring == engine::rates::Measuring::Skip {
        let mut tuned = chip.tuned().clone();
        tuned.timed.push(profile::fire::RESIDENT_WORKGROUPS.to_string());
        chip.with_tuned(tuned)
    } else {
        chip
    };
    Ok(crate::tuning::Tuning::new(chip, device.limits()))
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
    let timer = |frame: Frame| Ok((None, (frame.commit_timed()? * 1e9) as u64));
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
    let tier = <Api as engine_gpu::serve::Serve>::tier(device);
    let timer = |frame: Frame| Ok((None, (frame.commit_timed()? * 1e9) as u64));
    tracing::debug!(
        bound_ms = ?max_submit.map(|b| b.as_secs_f64() * 1e3),
        stated = ?stated_submit_bound().map(|b| b.as_secs_f64() * 1e3),
        "rates: measuring, each submission within the bound"
    );
    engine_gpu::rates::measure::<Api, _>(device, tuning, tier, &timer, max_submit, stated)
        .map(|measured| on(device, measured))
}

/// [`measure`] behind serving: `yield_to` read before each frame, an
/// arrival stopping the step (`engine_gpu::rates::yielded`).
pub fn measure_behind(
    device: &Context,
    tuning: &crate::tuning::Tuning,
    max_submit: Option<std::time::Duration>,
    stated: &profile::fire::Rates,
    yield_to: &dyn Fn() -> bool,
) -> Result<engine::rates::Measured> {
    let tier = <Api as engine_gpu::serve::Serve>::tier(device);
    let timer = |frame: Frame| Ok((None, (frame.commit_timed()? * 1e9) as u64));
    engine_gpu::rates::measure_yielding::<Api, _>(
        device,
        tuning,
        tier,
        &timer,
        max_submit,
        stated,
        &mut |_| {},
        Some(yield_to),
    )
    .map(|measured| on(device, measured))
}

/// Each of `wanted`'s dense kernel curves to `top` rows behind serving
/// (`engine_gpu::rates::measure_shapes`), each point timed as
/// [`shapes_at_load`] times it, `yield_to` read before each frame.
pub fn curves_behind(
    device: &Context,
    tuning: &crate::tuning::Tuning,
    max_submit: Option<std::time::Duration>,
    stated: &profile::fire::Rates,
    wanted: &[(kernels_slang::linear::gemm::Tactic, u32, u32)],
    top: u32,
    yield_to: &dyn Fn() -> bool,
) -> Result<Vec<profile::fire::KernelCurve>> {
    let tier = <Api as engine_gpu::serve::Serve>::tier(device);
    let timer = |frame: Frame| {
        let started = std::time::Instant::now();
        let span = frame.commit_spanned()?;
        Ok((
            span.filter(|&ns| ns > 0),
            started.elapsed().as_nanos() as u64,
        ))
    };
    engine_gpu::rates::measure_shapes::<Api, _>(
        device,
        tuning,
        tier,
        &timer,
        max_submit,
        None,
        stated,
        wanted,
        None,
        top,
        &mut |_| {},
        Some(yield_to),
    )
}

/// `measured` stated as measured on `device` (`engine_gpu::stored`).
fn on(device: &Context, mut measured: engine::rates::Measured) -> engine::rates::Measured {
    measured.rates.measured_on = Some(device.measured_on());
    measured
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
    // spanned frame, none between its dispatches): the host's wait swings a
    // frame more than the kernels it weighs apart.
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
    Ok(crate::tuning::Tuning::new(chip?, device.limits()))
}

/// [`measure`], saying how the warm-up ended.
pub fn measure_saying(
    device: &Context,
    tuning: &crate::tuning::Tuning,
    say: &mut dyn FnMut(&str),
) -> Result<engine::rates::Measured> {
    let tier = <Api as engine_gpu::serve::Serve>::tier(device);
    let timer = |frame: Frame| Ok((None, (frame.commit_timed()? * 1e9) as u64));
    tracing::debug!(
        bound_ms = ?stated_submit_bound().map(|b| b.as_secs_f64() * 1e3),
        "rates: measuring (pie tune), each submission within the bound"
    );
    engine_gpu::rates::measure_saying::<Api, _>(
        device,
        tuning,
        tier,
        &timer,
        stated_submit_bound(),
        &profile::fire::Rates::default(),
        say,
    )
    .map(|measured| on(device, measured))
}

/// The bf16 matmul rate of `device` at `(m, n, k)`, under the tactic
/// `tuning` picks ([`engine_gpu::rates::gemm_rate`]).
pub fn gemm_rate(
    device: &Context,
    tuning: &crate::tuning::Tuning,
    mnk: (u32, u32, u32),
) -> Result<f64> {
    let tier = <Api as engine_gpu::serve::Serve>::tier(device);
    let timer = |frame: Frame| Ok((None, (frame.commit_timed()? * 1e9) as u64));
    engine_gpu::rates::gemm_rate::<Api>(device, tuning.gemm(tier, mnk.0), &timer, mnk)
}

/// The attention rate of `device` at `shape`, through the kernel a served
/// fire takes there ([`engine_gpu::rates::attention_rate`]).
pub fn attention_rate(
    device: &Context,
    tuning: &crate::tuning::Tuning,
    shape: engine::rates::AttnShape,
    launch: f64,
) -> Result<f64> {
    let tier = <Api as engine_gpu::serve::Serve>::tier(device);
    let timer = |frame: Frame| Ok((None, (frame.commit_timed()? * 1e9) as u64));
    engine_gpu::rates::attention_rate::<Api, _>(device, tuning, tier, &timer, shape, launch)
}
