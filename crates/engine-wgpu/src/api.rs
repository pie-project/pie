//! The runtime's engine ([`engine_gpu::api`]) bound to WebGPU: a step's
//! guest passes run where its rows land, off the submitting thread.

use std::path::PathBuf;

use engine::error::{Error, Result as EngineResult};
use engine::fire::{FireTicket, Step};
use engine::transfer::{DeviceReach, MemoryDomain};
use eta_ir::registry::PortMask;

#[cfg(feature = "wgpu")]
use engine_gpu::api::refusal;
use engine_gpu::api::{Envelope, Host, PendingStep, program_attachments};

use crate::device::Api;

#[derive(Debug, Clone, PartialEq)]
pub struct DeviceBoot {
    pub adapter_index: u32,

    pub backends: Option<String>,

    pub power_preference: String,

    /// `$PIE_HOME/profiles`, where a native user's adapter profiles live.
    pub profiles: Option<PathBuf>,

    /// Whether the load times what no stored or committed profile states
    /// under a matching key (`[engine] measure`, `pie serve --no-measure`):
    /// skipped, each rate, curve and settle is served by the estimate, named.
    pub measure: engine::rates::Measuring,

    /// Diagnostics: the chunk size the pool's recipe aims at, in place of
    /// the profile's (a small one cuts records into pieces).
    pub chunk_bytes: Option<u64>,

    /// A test's device that binds less (as a phone's GPU does): the
    /// largest buffer the load lays, below the device's own. Test builds
    /// only (`test-limits`): no config or profile reaches it.
    #[cfg(feature = "test-limits")]
    pub max_buffer: Option<u64>,

    /// A test's device whose platform states a bound on one GPU submission
    /// (as Android's KGSL states 2 s): the bound every serving step is held
    /// to, in place of the platform's (`engine_gpu::rates::
    /// stated_submit_bound`). Test builds only (`test-limits`): no config
    /// or profile reaches it.
    #[cfg(feature = "test-limits")]
    pub submit_bound: Option<std::time::Duration>,

    /// A test's load whose harness states what the test fires: its fires
    /// laid to the rows its budgets state, in place of the fire-top rule's fire
    /// top (`engine_gpu::serve::Serve::fires_stated`). Test builds only
    /// (`test-limits`): no config or profile reaches it.
    #[cfg(feature = "test-limits")]
    pub fires_stated: bool,

    /// A test run's forms of the tab on a native device ([`TestForms`]).
    /// Test builds only (`test-limits`): no config or profile reaches it.
    #[cfg(feature = "test-limits")]
    pub test_forms: TestForms,
}

/// What a native test run takes of the tab's forms, which the tab takes
/// for its own reasons (`cfg wasm32`) and a native device does not: stated
/// by the test's harness, never read from the environment (a shell reads
/// none). Test builds only (`test-limits`).
#[cfg(feature = "test-limits")]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TestForms {
    /// The storage buffers a shader stage binds at most, below the
    /// adapter's (as SwiftShader's 10): the kernels' one-segment forms, and
    /// the tab's aliasing refusals, on a device with room.
    pub storage_buffers: Option<u32>,
    /// Where those refusals are: each overlap said, none refused, for a
    /// whole list in one run.
    pub aliasing_said: bool,
    /// Each guest step compacted by naga, as the tab's is.
    pub guest_compact: bool,
    /// The guest's pivot finish rolled, as the tab's is.
    pub guest_rolled: bool,
}

impl Default for DeviceBoot {
    fn default() -> DeviceBoot {
        DeviceBoot {
            adapter_index: 0,
            backends: None,
            power_preference: crate::boot::DEFAULT_POWER_PREFERENCE.to_string(),
            profiles: None,
            measure: engine::rates::Measuring::Time,
            chunk_bytes: None,
            #[cfg(feature = "test-limits")]
            max_buffer: None,
            #[cfg(feature = "test-limits")]
            submit_bound: None,
            #[cfg(feature = "test-limits")]
            fires_stated: false,
            #[cfg(feature = "test-limits")]
            test_forms: TestForms::default(),
        }
    }
}

pub type Wgpu = engine_gpu::api::Gpu<Api>;

impl Host for Api {
    const KIND: &'static str = "wgpu";

    type Programs = crate::program::Plane;

    fn domain(boot: &DeviceBoot) -> MemoryDomain {
        MemoryDomain::Device {
            ordinal: boot.adapter_index,
            reach: DeviceReach::Private,
        }
    }

    fn measured_on(device: &crate::device::Context) -> String {
        device.measured_on()
    }

    fn ports() -> PortMask {
        PortMask::DEVICE_GEOMETRY
            .with(eta_ir::registry::Port::Fold)
            .with(eta_ir::registry::Port::AttnMask)
            // Each beam's parent: the engine keeps the beams' page lists
            // (`engine_gpu::beam`).
            .with(eta_ir::registry::Port::Parents)
    }

    fn settles_asynchronously(_gpu: &Wgpu) -> bool {
        true
    }

    fn step(
        gpu: &mut Wgpu,
        step: &Step,
        at: engine::StepDone,
    ) -> EngineResult<(FireTicket, PendingStep)> {
        let defers = defers(gpu, step);
        let (ticket, mut step_pending) = gpu.fire_step(step, at, defers, None)?;
        if defers {
            // The passes land on their own and report the step; the next
            // frame's forward reads their tokens on the device, so nothing
            // here waits.
            let wrote = std::mem::take(&mut step_pending.wrote);
            let faulted = std::mem::take(&mut step_pending.faulted);
            let fed = step_pending.fed.take();
            let landed = step_pending.landed;
            defer_attachments(gpu, step, at, &landed, wrote, faulted, fed)?;
            step_pending.rows = Some(Vec::new());
        }
        Ok((ticket, step_pending))
    }

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
            // Device geometry resolves every port its lanes read, the
            // shared engine's ([`engine_gpu::program::device_lanes`]).
            if gpu.programs.geometry(attachment.instance)
                == Some(eta_ir::registry::GeometryClass::DeviceGeometry)
            {
                continue;
            }
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
                #[cfg(feature = "wgpu")]
                if lane.tokens.len() == 1
                    && let Some(cell) = gpu.programs.token_feed(attachment.instance, lane_at)
                {
                    // A pass still on the queue wrote this lane's token; the
                    // forward copies it in behind that pass.
                    envelope.tokens[lane_index] = Some(vec![0; lane.tokens.len()]);
                    envelope.feeds[lane_index] = Some(cell);
                    if lane.mask.is_none() {
                        match gpu.programs.envelope_mask(
                            attachment.instance,
                            lane.tokens.len(),
                            lane_at,
                        ) {
                            Ok(mask) => envelope.masks[lane_index] = mask,
                            Err(refused) => {
                                envelope.refused(step, attachment, &refused.to_string(), faulted);
                                continue 'attachments;
                            }
                        }
                    }
                    continue;
                }
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

/// Whether `step`'s attachments can run behind its forward, all in one
/// frame, without the host waiting for any of them: each reads its outputs
/// on the device and records one stage.
fn defers(gpu: &Wgpu, step: &Step) -> bool {
    #[cfg(feature = "wgpu")]
    {
        gpu.sink.is_some()
            && program_attachments(step).next().is_some()
            && program_attachments(step).all(|attachment| gpu.programs.defers(attachment.instance))
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let _ = (gpu, step);
        false
    }
}

#[allow(clippy::too_many_arguments)]
fn defer_attachments(
    gpu: &mut Wgpu,
    step: &Step,
    at: engine::StepDone,
    landed: &engine_gpu::serve::Landed,
    wrote: Vec<engine::KvWrites>,
    faulted: Vec<engine::LaneFault>,
    fed: Option<engine_gpu::settle::FedCheck>,
) -> EngineResult<()> {
    #[cfg(feature = "wgpu")]
    {
        let Wgpu {
            shell,
            programs,
            sink,
            ..
        } = gpu;
        let shell = shell
            .as_ref()
            .ok_or_else(|| Error::Load("the wgpu engine has no model loaded".into()))?;
        let passes: Vec<crate::program::DeferredPass<'_>> = program_attachments(step)
            .map(|attachment| crate::program::DeferredPass {
                instance: attachment.instance,
                lanes: engine_gpu::api::attachment_lanes(attachment),
                seats: (attachment.lane..attachment.lane + attachment.lanes.max(1))
                    .map(|lane| {
                        let mut seats: Vec<crate::guest::run::Readout<'_>> = shell
                            .flight_seats(landed, lane)
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
                                dtype: eta_ir::Dtype::Bf16,
                            });
                        }
                        seats
                    })
                    .collect(),
            })
            .collect();
        let done = sink.clone().map(|sink| engine_gpu::settle::Done {
            at,
            sink,
            wrote,
            lanes: u32::try_from(step.lanes.len()).unwrap_or(u32::MAX),
            faulted,
            meet: None,
            fed,
        });
        programs
            .fire_deferred(passes, shell.device(), done)
            .map_err(refusal)
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let _ = (gpu, step, at, landed, wrote, faulted, fed);
        Err(Error::unsupported("wgpu", "a deferred guest pass"))
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
