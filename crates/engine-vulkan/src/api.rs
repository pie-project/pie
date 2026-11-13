//! The runtime's engine ([`engine_gpu::api`]) bound to Vulkan: a guest pass
//! reads its lane's outputs on the device, behind the forward on the queue,
//! and a sampler may land on its own while the next forward reads its token
//! there.

use std::path::PathBuf;

#[cfg(feature = "vulkan")]
use engine::Engine;
use engine::error::{Error, Result as EngineResult};
use engine::fire::{FireTicket, Step};
use engine::transfer::MemoryDomain;
use eta_ir::registry::{PortMask, Stage};

use engine_gpu::api::{Envelope, Host, PendingStep, fault, program_attachments, refusal};

use crate::device::Api;
use engine_gpu::serve::Landed;
#[cfg(feature = "vulkan")]
use engine_gpu::serve::Shell;

use crate::serve::Seats;
#[cfg(feature = "vulkan")]
use engine_gpu::settle::Done;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct DeviceBoot {
    pub device_index: u32,

    pub validation: bool,

    pub pipeline_cache: Option<PathBuf>,

    /// `$PIE_HOME/profiles`, where a user's device profiles live.
    pub profiles: Option<PathBuf>,

    /// Diagnostics: the chunk size the pool's recipe aims at, in place of
    /// the profile's (a small one cuts records into pieces).
    pub chunk_bytes: Option<u64>,
}

pub type Vulkan = engine_gpu::api::Gpu<Api>;

impl Host for Api {
    const KIND: &'static str = "vulkan";

    const DEVICE_CHANNEL_COMMIT: bool = false;

    type Programs = crate::program::Plane;

    fn domain(boot: &DeviceBoot) -> MemoryDomain {
        MemoryDomain::VulkanDevice(boot.device_index)
    }

    fn ports() -> PortMask {
        PortMask::DECODE_ENVELOPE.with(eta_ir::registry::Port::Fold)
    }

    fn settles_asynchronously(gpu: &Vulkan) -> bool {
        gpu.shell
            .as_ref()
            .is_some_and(|shell| shell.frames_in_flight() > 1)
    }

    fn step(
        gpu: &mut Vulkan,
        step: &Step,
        at: engine::StepDone,
    ) -> EngineResult<(FireTicket, PendingStep)> {
        let defers = defers(gpu, step);
        let (ticket, mut step_pending) = gpu.fire_step(step, at, defers)?;
        let landed = step_pending.landed;
        if defers && reads_on_device(gpu, step, &landed) {
            // The sampler lands on its own and reports the step;
            // the next frame's forward reads its token on the
            // device, so nothing here waits.
            defer_attachment(gpu, step, at, &landed)?;
            let shell = gpu.loaded_mut()?;
            shell.leave_unread(&landed);
            shell.forget_seat(&landed);
            step_pending.rows = Some(Vec::new());
            return Ok((ticket, step_pending));
        }
        if program_attachments(step).next().is_some()
            && reads_on_device(gpu, step, &step_pending.landed)
        {
            // Every attachment reads its logits on the device:
            // it queues behind the forward without waiting for
            // it, so the GPU runs on into the sampler and the
            // step waits once.
            let landed = step_pending.landed;
            run_attachments(gpu, step, &landed)?;
            let shell = gpu.loaded_mut()?;
            step_pending.rows = Some(shell.rows_of(&landed).map_err(fault)?);
            shell.forget_seat(&landed);
        } else if program_attachments(step).next().is_some() {
            let shell = gpu.loaded_mut()?;
            let rows = shell.rows_of(&step_pending.landed).map_err(fault)?;
            step_pending.rows = Some(rows);
            let landed = step_pending.landed;
            run_attachments(gpu, step, &landed)?;
        }
        if defers && let Some(sink) = gpu.sink.as_ref() {
            // The forward's own report was held for the sampler.
            (sink)(at, engine::StepOutcome::Committed);
        }
        Ok((ticket, step_pending))
    }

    fn submitted(gpu: &Vulkan) {
        #[cfg(feature = "vulkan")]
        if let Some(shell) = gpu.shell.as_ref() {
            shell.device().pipeline_cache().settle();
        }
        #[cfg(not(feature = "vulkan"))]
        let _ = gpu;
    }

    /// Before the step resolves its inputs: the forward reads the cells a prologue put.
    fn run_prologues(gpu: &mut Vulkan, step: &Step) -> EngineResult<()> {
        for attachment in program_attachments(step) {
            if !gpu.programs.prologue_runs(attachment.instance) {
                continue;
            }
            gpu.programs
                .fire(
                    attachment.instance,
                    Stage::Prologue,
                    #[cfg(feature = "vulkan")]
                    gpu.shell.as_ref().map(Shell::device),
                    #[cfg(feature = "vulkan")]
                    Vec::new(),
                )
                .map_err(refusal)?;
        }
        Ok(())
    }

    fn envelope(gpu: &mut Vulkan, step: &Step, envelope: &mut Envelope<Api>) -> EngineResult<()> {
        for attachment in &step.attachments {
            let Some(lane) = step.lanes.get(attachment.lane as usize) else {
                return Err(Error::Invalid(format!(
                    "attachment names lane {} of a {}-lane step",
                    attachment.lane,
                    step.lanes.len()
                )));
            };
            #[cfg(feature = "vulkan")]
            if let Some(cell) = gpu.programs.token_feed(attachment.instance) {
                // A pass still on the queue wrote this lane's token; the
                // forward copies it in behind that pass.
                envelope.tokens[attachment.lane as usize] = Some(vec![0; lane.tokens.len()]);
                envelope.feeds[attachment.lane as usize] = Some(cell);
                continue;
            }
            envelope.tokens[attachment.lane as usize] = gpu
                .programs
                .envelope_tokens(attachment.instance, lane.tokens.len())
                .map_err(refusal)?;
        }
        Ok(())
    }

    fn guests(_gpu: &mut Vulkan, _step: &Step) -> EngineResult<Option<std::convert::Infallible>> {
        Ok(None)
    }
}

/// Whether every attachment of `step` can read its lane's outputs on
/// the device before the step lands.
fn reads_on_device(gpu: &Vulkan, step: &Step, landed: &Landed) -> bool {
    let Some(shell) = gpu.shell.as_ref() else {
        return false;
    };
    cfg!(feature = "vulkan")
        && program_attachments(step)
            .all(|attachment| shell.readout_seats(landed, attachment.lane).is_some())
}

fn run_attachments(gpu: &mut Vulkan, step: &Step, landed: &Landed) -> EngineResult<()> {
    let Vulkan {
        shell, programs, ..
    } = gpu;
    for attachment in program_attachments(step) {
        #[cfg(feature = "vulkan")]
        let (device, seats) = match shell.as_ref() {
            Some(shell) => (Some(shell.device()), lane_seats(shell, landed, attachment)),
            None => (None, Vec::new()),
        };
        programs
            .fire(
                attachment.instance,
                Stage::Epilogue,
                #[cfg(feature = "vulkan")]
                device,
                #[cfg(feature = "vulkan")]
                seats,
            )
            .map_err(refusal)?;
    }
    if let Some(shell) = shell.as_mut() {
        shell.forget_seat(landed);
    }
    Ok(())
}

/// Whether `step`'s one attachment can run behind its forward without
/// the host waiting for either.
fn defers(gpu: &Vulkan, step: &Step) -> bool {
    #[cfg(feature = "vulkan")]
    {
        gpu.sink.is_some()
            && gpu.settles_asynchronously()
            && step.attachments.len() == 1
            && !step.attachments[0].adapter_only
            && gpu.programs.defers(step.attachments[0].instance)
    }
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = (gpu, step);
        false
    }
}

fn defer_attachment(
    gpu: &mut Vulkan,
    step: &Step,
    at: engine::StepDone,
    landed: &Landed,
) -> EngineResult<()> {
    #[cfg(feature = "vulkan")]
    {
        let Vulkan {
            shell,
            programs,
            sink,
            ..
        } = gpu;
        let shell = shell
            .as_ref()
            .ok_or_else(|| Error::Load("the vulkan engine has no model loaded".into()))?;
        let attachment = &step.attachments[0];
        let seats = lane_seats(shell, landed, attachment);
        let done = sink.clone().map(|sink| Done { at, sink });
        programs
            .fire_deferred(attachment.instance, shell.device(), seats, done)
            .map_err(refusal)
    }
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = (gpu, step, at, landed);
        Err(Error::unsupported("vulkan", "a deferred guest pass"))
    }
}

/// What step `landed` read out for each lane of `attachment`, its pixels among them.
#[cfg(feature = "vulkan")]
fn lane_seats<'a>(
    shell: &'a Shell<Api>,
    landed: &Landed,
    attachment: &engine::fire::Attachment,
) -> Vec<Vec<crate::guest::run::Readout<'a>>> {
    (attachment.lane..attachment.lane + attachment.lanes.max(1))
        .map(|lane| {
            let mut seats: Vec<crate::guest::run::Readout<'a>> = shell
                .readout_seats(landed, lane)
                .unwrap_or_default()
                .into_iter()
                .map(
                    |(seat, at, width, dtype, output)| crate::guest::run::Readout {
                        output,
                        seat,
                        at,
                        width,
                        dtype,
                    },
                )
                .collect();
            if let Some((seat, at, width, output)) = shell.pixels_row(landed, lane) {
                seats.push(crate::guest::run::Readout {
                    output,
                    seat,
                    at,
                    width,
                    dtype: poem_ir::Dtype::Bf16,
                });
            }
            seats
        })
        .collect()
}
