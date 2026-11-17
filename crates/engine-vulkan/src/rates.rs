//! The device's rates on Vulkan ([`engine_gpu::rates::measure`]), every
//! fire timed by the host's clock around submit and wait: what
//! `pie tune vulkan --rates` states in the device's profile.

use engine_gpu::backend::Tactics as _;
use engine_gpu::error::Result;
use profile::fire::Rates;

use crate::device::{Api, Context, Frame};

/// The rates of `device`, under the tactics `tuning` picks.
pub fn measure(device: &Context, tuning: &crate::tuning::Tuning) -> Result<Rates> {
    let tier = <Api as engine_gpu::serve::Serve>::tier(device);
    let timer = |frame: Frame| Ok((None, (frame.commit_timed()? * 1e9) as u64));
    engine_gpu::rates::measure::<Api, _>(device, tuning, tier, &timer)
}

/// [`measure`], saying how the warm-up ended.
pub fn measure_saying(
    device: &Context,
    tuning: &crate::tuning::Tuning,
    say: &mut dyn FnMut(&str),
) -> Result<Rates> {
    let tier = <Api as engine_gpu::serve::Serve>::tier(device);
    let timer = |frame: Frame| Ok((None, (frame.commit_timed()? * 1e9) as u64));
    engine_gpu::rates::measure_saying::<Api, _>(device, tuning, tier, &timer, say)
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
    engine_gpu::rates::gemm_rate::<Api>(
        device,
        tuning.gemm(tier),
        &timer,
        mnk,
        engine_gpu::backend::Tactics::max_submit_ns(tuning),
    )
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
