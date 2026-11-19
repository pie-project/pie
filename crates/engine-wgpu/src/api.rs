//! The runtime's engine ([`engine_gpu::api`]) bound to WebGPU: a step's
//! guest passes run where its rows land, off the submitting thread.

use std::path::PathBuf;

use engine::error::{Error, Result as EngineResult};
use engine::fire::{FireTicket, Step};
use engine::transfer::MemoryDomain;
use eta_ir::registry::PortMask;

use engine_gpu::api::{Envelope, Host, PendingStep, program_attachments};
#[cfg(feature = "wgpu")]
use engine_gpu::api::refusal;

use crate::device::Api;

#[derive(Debug, Clone, PartialEq)]
pub struct DeviceBoot {
    pub adapter_index: u32,

    pub backends: Option<String>,

    pub power_preference: String,

    /// `$PIE_HOME/profiles`, where a native user's adapter profiles live.
    pub profiles: Option<PathBuf>,

    /// Diagnostics: the chunk size the pool's recipe aims at, in place of
    /// the profile's (a small one cuts records into pieces).
    pub chunk_bytes: Option<u64>,

    /// A test's device that binds less (as a phone's GPU does): the
    /// largest buffer the load lays, below the device's own. Test builds
    /// only (`test-limits`): no config or profile reaches it.
    #[cfg(feature = "test-limits")]
    pub max_buffer: Option<u64>,
}

impl Default for DeviceBoot {
    fn default() -> DeviceBoot {
        DeviceBoot {
            adapter_index: 0,
            backends: None,
            power_preference: crate::boot::DEFAULT_POWER_PREFERENCE.to_string(),
            profiles: None,
            chunk_bytes: None,
            #[cfg(feature = "test-limits")]
            max_buffer: None,
        }
    }
}

pub type Wgpu = engine_gpu::api::Gpu<Api>;

impl Host for Api {
    const KIND: &'static str = "wgpu";


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
        gpu.fire_step(step, at, false, None)
    }

    fn submitted(_gpu: &Wgpu) {}

    fn run_prologues(
        gpu: &mut Wgpu,
        step: &Step,
        faulted: &mut engine::StepOutcome,
    ) -> EngineResult<()> {
        run_prologues(gpu, step, faulted)
    }

    fn envelope(
        gpu: &mut Wgpu,
        step: &Step,
        envelope: &mut Envelope<Api>,
        faulted: &mut engine::StepOutcome,
    ) -> EngineResult<()> {
        'attachments: for attachment in &step.attachments {
            let lane_count = match gpu.programs.lane_count(attachment.instance) {
                Ok(count) => count,
                Err(refused) => {
                    envelope.refused(step, attachment, &refused.to_string(), faulted);
                    continue;
                }
            };
            if let Some(why) = engine_gpu::api::overreach(attachment, lane_count) {
                envelope.refused(step, attachment, &why, faulted);
                continue;
            }
            for lane_at in 0..lane_count {
                let lane_index = attachment.lane as usize + lane_at;
                let Some(lane) = step.lanes.get(lane_index) else {
                    return Err(Error::Invalid(format!(
                        "attachment names lane {} of a {}-lane step",
                        lane_index,
                        step.lanes.len()
                    )));
                };
                let resolved = gpu
                    .programs
                    .envelope_tokens(attachment.instance, lane.tokens.len(), lane_at)
                    .and_then(|tokens| {
                        let mask = match lane.mask {
                            Some(_) => None,
                            None => gpu.programs.envelope_mask(
                                attachment.instance,
                                lane.tokens.len(),
                                lane_at,
                            )?,
                        };
                        Ok((tokens, mask))
                    });
                match resolved {
                    Ok((tokens, mask)) => {
                        envelope.tokens[lane_index] = tokens;
                        if lane.mask.is_none() {
                            envelope.masks[lane_index] = mask;
                        }
                    }
                    Err(refused) => {
                        envelope.refused(step, attachment, &refused.to_string(), faulted);
                        continue 'attachments;
                    }
                }
            }
        }
        Ok(())
    }

    fn guests(gpu: &mut Wgpu, step: &Step) -> EngineResult<Option<crate::landing::Guests>> {
        deferred_guests(gpu, step)
    }
}

/// Runs `submission`'s prologues, each on its own: one that refuses (its
/// instance poisoned in an earlier step, a guard) faults its own lanes.
#[cfg(feature = "wgpu")]
fn run_prologues(
    gpu: &mut Wgpu,
    submission: &Step,
    faulted: &mut engine::StepOutcome,
) -> EngineResult<()> {
    let mut prologues = program_attachments(submission)
        .filter(|attachment| gpu.programs.prologue_runs(attachment.instance))
        .peekable();
    if prologues.peek().is_none() {
        return Ok(());
    }
    let device = gpu.loaded()?.device().clone();
    let programs = &gpu.programs;
    engine_gpu::api::each_pass(prologues, "prologue", faulted, |attachment| {
        programs.fire_prologue(&device, attachment.instance)
    });
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
fn run_prologues(
    gpu: &mut Wgpu,
    submission: &Step,
    _faulted: &mut engine::StepOutcome,
) -> EngineResult<()> {
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
