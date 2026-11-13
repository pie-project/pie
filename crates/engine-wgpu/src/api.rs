//! The runtime's engine ([`engine_gpu::api`]) bound to WebGPU: a step's
//! guest passes run where its rows land, off the submitting thread.

use std::path::PathBuf;

use engine::error::{Error, Result as EngineResult};
use engine::fire::{FireTicket, Step};
use engine::transfer::MemoryDomain;
use eta_ir::registry::PortMask;

use engine_gpu::api::{Envelope, Host, PendingStep, program_attachments, refusal};

use crate::device::Api;

#[derive(Debug, Clone, PartialEq)]
pub struct DeviceBoot {
    pub adapter_index: u32,

    pub backends: Option<String>,

    pub power_preference: String,

    pub pipeline_cache: Option<PathBuf>,

    pub device_memory: Option<u64>,

    /// `$PIE_HOME/profiles`, where a native user's adapter profiles live.
    pub profiles: Option<PathBuf>,

    /// Diagnostics: the chunk size the pool's recipe aims at, in place of
    /// the profile's (a small one cuts records into pieces).
    pub chunk_bytes: Option<u64>,
}

impl Default for DeviceBoot {
    fn default() -> DeviceBoot {
        DeviceBoot {
            adapter_index: 0,
            backends: None,
            power_preference: crate::boot::DEFAULT_POWER_PREFERENCE.to_string(),
            pipeline_cache: None,
            device_memory: None,
            profiles: None,
            chunk_bytes: None,
        }
    }
}

pub type Wgpu = engine_gpu::api::Gpu<Api>;

impl Host for Api {
    const KIND: &'static str = "wgpu";

    const DEVICE_CHANNEL_COMMIT: bool = true;

    type Programs = crate::program::Plane;

    fn domain(boot: &DeviceBoot) -> MemoryDomain {
        MemoryDomain::WgpuDevice(boot.adapter_index)
    }

    fn ports() -> PortMask {
        PortMask::DECODE_ENVELOPE
            .with(eta_ir::registry::Port::Fold)
            .with(eta_ir::registry::Port::AttnMask)
    }

    fn settles_asynchronously(_gpu: &Wgpu) -> bool {
        true
    }

    fn step(
        gpu: &mut Wgpu,
        step: &Step,
        at: engine::StepDone,
    ) -> EngineResult<(FireTicket, PendingStep)> {
        gpu.fire_step(step, at, false)
    }

    fn submitted(_gpu: &Wgpu) {}

    fn run_prologues(gpu: &mut Wgpu, step: &Step) -> EngineResult<()> {
        // Checked before a prologue moves the cursors the caller predicted.
        for attachment in &step.attachments {
            if let Some(lane) = step.lanes.get(attachment.lane as usize)
                && !lane.channels.is_empty()
                && let Some(why) = gpu
                    .programs
                    .disagreeing_ticket(attachment.instance, &lane.channels)
            {
                return Err(Error::Program(format!(
                    "this fire's channel predictions and the engine's disagree: {why}"
                )));
            }
        }
        run_prologues(gpu, step)
    }

    fn envelope(gpu: &mut Wgpu, step: &Step, envelope: &mut Envelope<Api>) -> EngineResult<()> {
        for attachment in &step.attachments {
            let lane_count = gpu
                .programs
                .lane_count(attachment.instance)
                .map_err(refusal)?;
            for lane_at in 0..lane_count {
                let lane_index = attachment.lane as usize + lane_at;
                let Some(lane) = step.lanes.get(lane_index) else {
                    return Err(Error::Invalid(format!(
                        "attachment names lane {} of a {}-lane step",
                        lane_index,
                        step.lanes.len()
                    )));
                };
                envelope.tokens[lane_index] = gpu
                    .programs
                    .envelope_tokens(attachment.instance, lane.tokens.len(), lane_at)
                    .map_err(refusal)?;
                if lane.mask.is_none() {
                    envelope.masks[lane_index] = gpu
                        .programs
                        .envelope_mask(attachment.instance, lane.tokens.len(), lane_at)
                        .map_err(refusal)?;
                }
            }
        }
        Ok(())
    }

    fn guests(gpu: &mut Wgpu, step: &Step) -> EngineResult<Option<crate::landing::Guests>> {
        deferred_guests(gpu, step)
    }
}

#[cfg(feature = "wgpu")]
fn run_prologues(gpu: &mut Wgpu, submission: &Step) -> EngineResult<()> {
    let mut prologues = program_attachments(submission)
        .filter(|attachment| gpu.programs.prologue_runs(attachment.instance))
        .peekable();
    if prologues.peek().is_none() {
        return Ok(());
    }
    let device = gpu.loaded()?.device().clone();
    for attachment in prologues {
        gpu.programs
            .fire_prologue(&device, attachment.instance)
            .map_err(refusal)?;
    }
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
fn run_prologues(gpu: &mut Wgpu, submission: &Step) -> EngineResult<()> {
    if program_attachments(submission)
        .any(|attachment| gpu.programs.prologue_runs(attachment.instance))
    {
        return Err(Error::Program(
            "this build carries no wgpu device, so a guest pass has nowhere to run".into(),
        ));
    }
    Ok(())
}

#[cfg(feature = "wgpu")]
fn deferred_guests(
    gpu: &mut Wgpu,
    submission: &Step,
) -> EngineResult<Option<crate::landing::Guests>> {
    if program_attachments(submission).next().is_none() {
        return Ok(None);
    }
    let device = gpu.loaded()?.device().clone();
    gpu.programs
        .deferred(
            &device,
            program_attachments(submission)
                .map(|attachment| (attachment.instance, attachment.lane, attachment.lanes)),
            gpu.loaded()?.output_names(),
        )
        .map(Some)
        .map_err(refusal)
}

#[cfg(not(feature = "wgpu"))]
fn deferred_guests(
    _gpu: &mut Wgpu,
    submission: &Step,
) -> EngineResult<Option<crate::landing::Guests>> {
    if program_attachments(submission).next().is_none() {
        Ok(None)
    } else {
        Err(Error::Program(
            "this build carries no wgpu device, so a guest pass has nowhere to run".into(),
        ))
    }
}
