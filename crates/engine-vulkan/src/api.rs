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

use engine_gpu::api::{Envelope, Host, PendingStep, each_pass, fault, program_attachments};
#[cfg(feature = "vulkan")]
use engine_gpu::api::{attachment_lanes, refusal};

use crate::device::Api;
use engine_gpu::serve::Landed;
#[cfg(feature = "vulkan")]
use engine_gpu::serve::Shell;

use crate::serve::Seats;
use engine_gpu::settle::{Meet, merged};
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

    /// A test's device that binds less (as a phone's GPU does): the
    /// largest buffer the load lays, below the device's own. Test builds
    /// only (`test-limits`): no config or profile reaches it.
    #[cfg(feature = "test-limits")]
    pub max_buffer: Option<u64>,
}

pub type Vulkan = engine_gpu::api::Gpu<Api>;

impl Host for Api {
    const KIND: &'static str = "vulkan";


    type Programs = crate::program::Plane;

    fn domain(boot: &DeviceBoot) -> MemoryDomain {
        MemoryDomain::VulkanDevice(boot.device_index)
    }

    fn ports() -> PortMask {
        PortMask::DEVICE_GEOMETRY
            .with(eta_ir::registry::Port::Fold)
            .with(eta_ir::registry::Port::AttnMask)
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
        let guests = program_attachments(step).next().is_some();
        // The host runs the step's guest passes behind its forward: off the
        // submit, the step lands once both are in, with the lanes they
        // faulted; inside it, the frame's settle tells them.
        let settles_off = gpu.sink.is_some() && <Api as Host>::settles_asynchronously(gpu);
        let meet = (guests && !defers && settles_off).then(Meet::new);
        let (ticket, mut step_pending) = gpu.fire_step(step, at, defers, meet.clone())?;
        let landed = step_pending.landed;
        if defers && reads_on_device(gpu, step, &landed) {
            // The sampler lands on its own and reports the step;
            // the next frame's forward reads its token on the
            // device, so nothing here waits.
            let wrote = std::mem::take(&mut step_pending.wrote);
            let faulted = std::mem::take(&mut step_pending.faulted);
            let fed = step_pending.fed.take();
            defer_attachment(gpu, step, at, &landed, wrote, faulted, fed)?;
            gpu.loaded_mut()?.forget_seat(&landed);
            step_pending.rows = Some(Vec::new());
            return Ok((ticket, step_pending));
        }
        let mut faulted = Vec::new();
        if guests && reads_on_device(gpu, step, &step_pending.landed) {
            // Every attachment reads its logits on the device:
            // it queues behind the forward without waiting for
            // it, so the GPU runs on into the sampler and the
            // step waits once.
            let landed = step_pending.landed;
            faulted = run_attachments(gpu, step, &landed);
            let shell = gpu.loaded_mut()?;
            step_pending.rows = Some(shell.rows_of(&landed).map_err(fault)?);
            shell.forget_seat(&landed);
        } else if guests {
            let shell = gpu.loaded_mut()?;
            let rows = shell.rows_of(&step_pending.landed).map_err(fault)?;
            step_pending.rows = Some(rows);
            let landed = step_pending.landed;
            faulted = run_attachments(gpu, step, &landed);
        }
        if let Some(meet) = meet {
            meet.guests_ran(faulted);
        } else if defers && let Some(sink) = gpu.sink.as_ref() {
            // The forward's own report was held for the sampler.
            let mut held = engine::StepOutcome {
                wrote: std::mem::take(&mut step_pending.wrote),
                faulted: std::mem::take(&mut step_pending.faulted),
            };
            // Its rows are read: the forward landed.
            if let Some(fed) = step_pending.fed.take() {
                held = merged(held, fed.faults());
            }
            (sink)(at, merged(held, faulted));
        } else {
            // Told with the frame's settle, inside the submit.
            let held = engine::StepOutcome {
                wrote: Vec::new(),
                faulted: std::mem::take(&mut step_pending.faulted),
            };
            step_pending.faulted = merged(held, faulted).faulted;
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

    /// Before the step resolves its inputs: the forward reads the cells a
    /// prologue put. A prologue that refuses (its instance poisoned in an
    /// earlier step, a guard) faults its own lanes; the others run.
    fn run_prologues(
        gpu: &mut Vulkan,
        step: &Step,
        faulted: &mut engine::StepOutcome,
    ) -> EngineResult<()> {
        let Vulkan {
            #[cfg(feature = "vulkan")]
            shell,
            programs,
            ..
        } = gpu;
        let prologues: Vec<&engine::fire::Attachment> = program_attachments(step)
            .filter(|attachment| programs.prologue_runs(attachment.instance))
            .collect();
        each_pass(prologues, "prologue", faulted, |attachment| {
            programs.fire(
                attachment.instance,
                Stage::Prologue,
                #[cfg(feature = "vulkan")]
                shell.as_ref().map(Shell::device),
                #[cfg(feature = "vulkan")]
                Vec::new(),
            )
        });
        Ok(())
    }

    fn envelope(
        gpu: &mut Vulkan,
        step: &Step,
        envelope: &mut Envelope<Api>,
        faulted: &mut engine::StepOutcome,
    ) -> EngineResult<()> {
        for attachment in &step.attachments {
            // Device geometry resolves every port its lanes read, the
            // shared engine's ([`engine_gpu::program::device_lanes`]).
            if gpu.programs.geometry(attachment.instance)
                == Some(eta_ir::registry::GeometryClass::DeviceGeometry)
            {
                continue;
            }
            // A refusal of this attachment's envelope (its lane count, a
            // mask, its tokens) faults its lanes; the others resolve.
            let refused: Option<String> = 'pass: {
                let lane_count = match gpu.programs.lane_count(attachment.instance) {
                    Ok(count) => count,
                    Err(refused) => break 'pass Some(refused.to_string()),
                };
                if let Some(why) = engine_gpu::api::overreach(attachment, lane_count) {
                    break 'pass Some(why);
                }
                for lane_at in 0..lane_count {
                    let lane_index = attachment.lane as usize + lane_at;
                    let Some(lane) = step.lanes.get(lane_index) else {
                        return Err(Error::Invalid(format!(
                            "attachment names lane {lane_index} of a {}-lane step",
                            step.lanes.len()
                        )));
                    };
                    #[cfg(feature = "vulkan")]
                    let mask_feed = (lane.mask.is_none() && lane.tokens.len() == 1)
                        .then(|| gpu.programs.mask_feed(attachment.instance, lane_at))
                        .flatten();
                    #[cfg(not(feature = "vulkan"))]
                    let mask_feed: Option<engine_gpu::serve::DeviceCell<Api>> = None;
                    if let Some(cell) = mask_feed {
                        // A pass still on the queue put this lane's mask row;
                        // the forward packs it in behind that pass. The
                        // placeholder states the row's width.
                        let keys = cell.bytes;
                        let admitted = u32::try_from(keys).unwrap_or(u32::MAX);
                        envelope.masks[lane_index] = Some(engine::fire::Masking::Extent(
                            engine::fire::Mask::new(vec![0, admitted], keys),
                        ));
                        envelope.mask_feeds[lane_index] = Some(cell);
                    } else if lane.mask.is_none() {
                        match gpu.programs.envelope_mask(
                            attachment.instance,
                            lane.tokens.len(),
                            lane_at,
                        ) {
                            Ok(mask) => envelope.masks[lane_index] = mask,
                            Err(refused) => break 'pass Some(refused.to_string()),
                        }
                    }
                    #[cfg(feature = "vulkan")]
                    if lane.tokens.len() == 1
                        && let Some(cell) = gpu.programs.token_feed(attachment.instance, lane_at)
                    {
                        // A pass still on the queue wrote this lane's token; the
                        // forward copies it in behind that pass.
                        envelope.tokens[lane_index] = Some(vec![0; lane.tokens.len()]);
                        envelope.feeds[lane_index] = Some(cell);
                        continue;
                    }
                    match gpu.programs.envelope_tokens(
                        attachment.instance,
                        lane.tokens.len(),
                        lane_at,
                    ) {
                        Ok(tokens) => envelope.tokens[lane_index] = tokens,
                        Err(refused) => break 'pass Some(refused.to_string()),
                    }
                }
                None
            };
            if let Some(why) = refused {
                envelope.refused(step, attachment, &why, faulted);
            }
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

/// Runs `step`'s guest epilogues on the host behind its forward, each
/// attachment's on its own: a pass that faults (its instance poisoned, a
/// guard, a stage's fault) faults its own lanes and no other.
fn run_attachments(gpu: &mut Vulkan, step: &Step, landed: &Landed) -> Vec<engine::LaneFault> {
    let Vulkan {
        shell, programs, ..
    } = gpu;
    let mut faulted = engine::StepOutcome::committed();
    each_pass(program_attachments(step), "epilogue", &mut faulted, |attachment| {
        #[cfg(feature = "vulkan")]
        let (device, seats) = match shell.as_ref() {
            Some(shell) => (Some(shell.device()), lane_seats(shell, landed, attachment)),
            None => (None, Vec::new()),
        };
        programs.fire(
            attachment.instance,
            Stage::Epilogue,
            #[cfg(feature = "vulkan")]
            device,
            #[cfg(feature = "vulkan")]
            seats,
        )
    });
    if let Some(shell) = shell.as_mut() {
        shell.forget_seat(landed);
    }
    faulted.faulted
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
    wrote: Vec<engine::KvWrites>,
    faulted: Vec<engine::LaneFault>,
    fed: Option<engine_gpu::settle::FedCheck>,
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
        let done = sink.clone().map(|sink| Done {
            at,
            sink,
            wrote,
            lanes: u32::try_from(step.lanes.len()).unwrap_or(u32::MAX),
            faulted,
            meet: None,
            fed,
        });
        programs
            .fire_deferred(
                attachment.instance,
                attachment_lanes(attachment),
                shell.device(),
                seats,
                done,
            )
            .map_err(refusal)
    }
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = (gpu, step, at, landed, wrote, faulted, fed);
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
