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
/// ([`engine_gpu::rates::STATED_SUBMIT_BOUND`]).
pub use engine_gpu::rates::STATED_SUBMIT_BOUND;

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
        &engine_gpu::rates::KEYS,
        save,
        Some(device.left()),
        max_submit,
        |max_submit| measure(device, &tuning, max_submit, &stated).map_err(|e| e.to_string()),
    )
    .map_err(|why| engine_gpu::error::Fault::Device {
        call: "rates at load",
        why,
    })?;
    Ok(crate::tuning::Tuning::new(chip, device.limits()))
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
        stated = ?STATED_SUBMIT_BOUND.map(|b| b.as_secs_f64() * 1e3),
        "rates: measuring, each submission within the bound"
    );
    engine_gpu::rates::measure::<Api, _>(device, tuning, tier, &timer, max_submit, stated)
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
        bound_ms = ?STATED_SUBMIT_BOUND.map(|b| b.as_secs_f64() * 1e3),
        "rates: measuring (pie tune), each submission within the bound"
    );
    engine_gpu::rates::measure_saying::<Api, _>(
        device,
        tuning,
        tier,
        &timer,
        STATED_SUBMIT_BOUND,
        &profile::fire::Rates::default(),
        say,
    )
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
