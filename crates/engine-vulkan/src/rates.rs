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

/// The bound the platform states on one GPU submission, which the rates
/// step holds each of its frames (and its warm-up's cube) within, each at
/// its estimate times the worst ratio of a frame's time to its estimate
/// seen (`engine_gpu::rates`). On Android, KGSL's (Qualcomm's msm GPU
/// driver, `qcom/opensource/graphics-kernel`, branch
/// gfx-kernel.lnx.16.0.r15-rel, the android16 / 6.12 line):
///
/// - The one duration the driver states is `adreno_drawobj_timeout = 2000`
///   ms (adreno_dispatch.c:51, "Command batch timeout (in milliseconds)"):
///   the ringbuffer dispatcher's per-submission expiry (adreno_dispatch.c:
///   2271, `drawqueue->expires = jiffies + msecs_to_jiffies(
///   adreno_drawobj_timeout)`) and the preemption timer
///   (adreno_gen8_preempt.c:55). Its fault-detect interval,
///   `_fault_timer_interval = 200` ms (adreno_dispatch.c:54), only samples
///   the GPU's progress registers on the ringbuffer dispatcher.
/// - An Adreno 840 (gen8, `adreno_gpu_core_gen8_*`, adreno-gpulist.h) runs
///   the hardware scheduler (`adreno_gen8_hwsched_gpudev`), not that
///   dispatcher: there long-IB detection (`ft_long_ib_detect`, 1 on the
///   840) enables the GMU firmware's bail-out timer
///   (adreno_gen8_hwsched_hfi.c:2663-2667, `HFI_FEATURE_BAIL_OUT_TIMER`
///   sent with data 0: the firmware's own default), whose expiry is the
///   context fault `GMU_GPU_SW_HANG` (603, adreno_hfi.h: "Fault due to Long
///   IB timeout"). Its duration lives in the GMU firmware: no kernel
///   constant, sysfs node or property states it (`KGSL_PROP_IB_TIMEOUT`,
///   0x21, is gone from msm_kgsl.h), and the 840's sysfs states none.
/// - The hardware hang interrupt (`ft_hang_intr_status` 1) counts cycles
///   of no interface progress (`hang_detect_cycles` 0xcfffff,
///   RBBM_INTERFACE_HANG_INT_CNTL, adreno_gen8.c:1785): a stall, not a
///   long submission that makes progress.
///
/// So the bound stated is `adreno_drawobj_timeout`'s 2 s, the one duration
/// the driver states; where the GMU's bail-out is shorter, the per-submit
/// trace (`RUST_LOG=engine_gpu::rates=debug`) shows the estimate and time
/// of every submission up to the one that faults. None stated elsewhere.
#[cfg(target_os = "android")]
pub const STATED_SUBMIT_BOUND: Option<std::time::Duration> =
    Some(std::time::Duration::from_secs(2));
#[cfg(not(target_os = "android"))]
pub const STATED_SUBMIT_BOUND: Option<std::time::Duration> = None;

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
    Ok(crate::tuning::Tuning::new(chip, device.limits()))
}

/// The rates of `device`, under the tactics `tuning` picks.
pub fn measure(
    device: &Context,
    tuning: &crate::tuning::Tuning,
    max_submit: Option<std::time::Duration>,
) -> Result<engine::rates::Measured> {
    let tier = <Api as engine_gpu::serve::Serve>::tier(device);
    let timer = |frame: Frame| Ok((None, (frame.commit_timed()? * 1e9) as u64));
    tracing::debug!(
        bound_ms = ?max_submit.map(|b| b.as_secs_f64() * 1e3),
        stated = ?STATED_SUBMIT_BOUND.map(|b| b.as_secs_f64() * 1e3),
        "rates: measuring, each submission within the bound"
    );
    engine_gpu::rates::measure::<Api, _>(device, tuning, tier, &timer, max_submit)
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
