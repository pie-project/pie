//! The runtime's [`Engine`] over a backend's shell: loading a model,
//! submitting and settling frames, and the guest program plane around them.

use std::collections::BTreeMap;
use std::path::Path;

use checkpoint::contract::ModelContract;
use engine::Engine;
use engine::adapter::AdapterRegistration;
use engine::caps::{Capabilities, DeviceFacts, FireLimits};
use engine::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use engine::error::{Error, Result as EngineResult};
use engine::fire::{
    FireId, FireTicket, FrameId, FrameSubmission, FrameTicket, LaneReadout, Readout, Step,
};
use engine::load::{Budgets as LoadBudgets, Checkpoint, LoadFacts, LoadRequest, Loaded};
use engine::program::{BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
use engine::transfer::{KvCopy, MemoryDomain};
use eta_ir::registry::{GeometryClass, PortMask};
use poem_compiler::{Budget, DeviceProfile, PATCH_LATTICE_FLOOR, PatchLadder};
use poem_ir::Trace;

use crate::error::Fault;
use crate::program::{Place, Programs, Refusal};
use crate::serve::{Boot, Clips, Landed, Lane, Media, Seated, Serve, Shell, StepView};
use crate::settle::Done;

pub type ContractFor = fn(&Trace, &Path) -> std::result::Result<ModelContract, String>;

/// A step fired and not yet read: each lane's readout policy, and its rows
/// once the host has them.
pub struct PendingStep {
    pub readout: Vec<Readout>,

    pub landed: Landed,

    pub rows: Option<Vec<Vec<f32>>>,

    /// The fold reports of a forward whose own completion does not carry
    /// them: fired with its completion deferred, the pass that reports the
    /// step carries them; on an engine that settles inside `submit`, the
    /// frame's settle there tells them.
    pub wrote: Vec<engine::KvWrites>,

    /// The lanes whose guest passes faulted ahead of the forward (a
    /// prologue, an envelope), where the forward's own completion does not
    /// carry them, as `wrote`.
    pub faulted: Vec<engine::LaneFault>,

    /// The check of the step's device-fed token ids, where the forward's
    /// own completion does not carry it: told with the step once it landed.
    pub fed: Option<crate::settle::FedCheck>,
}

/// What a backend crate supplies beside its [`Serve`] to stand behind the
/// runtime's [`Engine`]: its name, its program plane, and how a step's guest
/// passes run around its forward.
pub trait Host: Serve {
    /// The engine's name, as the runtime and a message state it.
    const KIND: &'static str;

    type Programs: Programs<Self>;

    /// The memory domain of the device `boot` opens.
    fn domain(boot: &Self::DeviceBoot) -> MemoryDomain;

    /// The ports a load resolves on the device.
    fn ports() -> PortMask;

    /// The device as what is measured on it is keyed by
    /// (`crate::stored`): its vendor and device ids and its driver.
    fn measured_on(device: &Self::Context) -> String;

    fn settles_asynchronously(gpu: &Gpu<Self>) -> bool;

    /// Fires `step` ([`Gpu::fire_step`]) and runs what attaches to it.
    fn step(
        gpu: &mut Gpu<Self>,
        step: &Step,
        at: engine::StepDone,
    ) -> EngineResult<(FireTicket, PendingStep)>;

    /// Before `step` resolves its inputs: the passes whose cells its forward
    /// reads. A pass that refuses (its instance poisoned, a guard) faults
    /// its own attachment's lanes into `faulted`; the others' run on.
    fn run_prologues(
        gpu: &mut Gpu<Self>,
        step: &Step,
        faulted: &mut engine::StepOutcome,
    ) -> EngineResult<()>;

    /// The tokens, masks and token feeds `step`'s attachments resolve for
    /// their lanes. An attachment whose instance refuses its envelope
    /// faults its own lanes ([`Envelope::refused`]); the others resolve.
    fn envelope(
        gpu: &mut Gpu<Self>,
        step: &Step,
        envelope: &mut Envelope<Self>,
        faulted: &mut engine::StepOutcome,
    ) -> EngineResult<()>;

    /// The guest passes that land with `step`'s rows.
    fn guests(gpu: &mut Gpu<Self>, step: &Step) -> EngineResult<Option<Self::Guests>>;
}

/// What a step's attachments resolve for each of its lanes, at the lane.
pub struct Envelope<B: Serve> {
    pub tokens: Vec<Option<Vec<u32>>>,

    pub masks: Vec<Option<engine::fire::Masking>>,

    /// A device cell holding the lane's one token ([`Seated::feed`]).
    pub feeds: Vec<Option<crate::serve::DeviceCell<B>>>,
}

impl<B: Serve> Envelope<B> {
    /// `attachment`'s instance refused its envelope for `why`: its lanes
    /// fault, and the forward embeds token 0 for them, rows nothing reads.
    pub fn refused(
        &mut self,
        step: &Step,
        attachment: &engine::fire::Attachment,
        why: &str,
        faulted: &mut engine::StepOutcome,
    ) {
        let lanes = attachment_lanes(attachment);
        for lane in lanes.clone() {
            let at = lane as usize;
            if let Some(stated) = step.lanes.get(at) {
                self.tokens[at] = Some(vec![0; stated.tokens.len()]);
                self.masks[at] = None;
                self.feeds[at] = None;
            }
        }
        tracing::warn!(instance = attachment.instance.0, %why, "a guest envelope refused; its lanes fault");
        faulted.fault_lanes(lanes, Some(attachment.instance), why);
    }
}

/// Runs `pass` for each of `attachments`, each on its own: one that
/// refuses (its instance poisoned, a guard, a stage's fault) faults its own
/// attachment's lanes into `faulted` by `stage`, and the others run on.
pub fn each_pass<'a>(
    attachments: impl IntoIterator<Item = &'a engine::fire::Attachment>,
    stage: &str,
    faulted: &mut engine::StepOutcome,
    mut pass: impl FnMut(&engine::fire::Attachment) -> std::result::Result<(), Refusal>,
) {
    for attachment in attachments {
        if let Err(refused) = pass(attachment) {
            tracing::warn!(
                instance = attachment.instance.0,
                stage,
                %refused,
                "a guest pass refused; its lanes fault"
            );
            faulted.fault_lanes(
                attachment_lanes(attachment),
                Some(attachment.instance),
                &refused.to_string(),
            );
        }
    }
}

/// The rows of `trace`'s widest token table, where it embeds any.
fn token_rows(trace: &Trace) -> Option<u32> {
    trace
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            poem_ir::Operation::Gather(poem_ir::Gather::Embed { vocab, .. }) => Some(*vocab),
            _ => None,
        })
        .max()
}

/// A lane whose tokens (as its guest resolved them, or as stated) hold an
/// id at or past `vocab` (an out-of-bounds row the device would read as row
/// 0) faults by name: the lanes of the attachment it is of, or itself alone.
fn out_of_range_tokens<B: Serve>(
    step: &Step,
    envelope: &mut Envelope<B>,
    vocab: u32,
    faulted: &mut engine::StepOutcome,
) {
    for (at, lane) in step.lanes.iter().enumerate() {
        let tokens = envelope.tokens[at].as_deref().unwrap_or(&lane.tokens);
        let Some(&id) = tokens.iter().find(|&&id| id >= vocab) else {
            continue;
        };
        // A skip row (`engine::ports::SKIP_TOKEN`) is a contract id, not an
        // out-of-range one; this engine does not carry skip rows yet.
        let what = if id == engine::ports::SKIP_TOKEN {
            "a skip row (token -1), which this engine does not carry yet".to_string()
        } else {
            format!("token {id}, out of range for a vocabulary of {vocab}")
        };
        let lane_at = u32::try_from(at).unwrap_or(u32::MAX);
        match step
            .attachments
            .iter()
            .find(|attachment| attachment_lanes(attachment).contains(&lane_at))
        {
            Some(attachment) => {
                let why = format!("instance {}: {what}", attachment.instance);
                envelope.refused(step, attachment, &why, faulted);
            }
            None => {
                envelope.tokens[at] = Some(vec![0; lane.tokens.len()]);
                faulted.fault_lanes(lane_at..lane_at + 1, None, &what);
            }
        }
    }
}

/// The lanes `attachment`'s pass is of.
#[must_use]
pub fn attachment_lanes(attachment: &engine::fire::Attachment) -> std::ops::Range<u32> {
    attachment.lane..attachment.lane + attachment.lanes.max(1)
}

/// Why an attachment whose program's `embed_indptr` port describes
/// `described` lanes is refused, where that is more than it spans: its
/// ports would state the next request's tokens and mask. `None`: they
/// describe its own lanes.
#[must_use]
pub fn overreach(attachment: &engine::fire::Attachment, described: usize) -> Option<String> {
    let lanes = attachment_lanes(attachment);
    (described > lanes.len()).then(|| {
        format!(
            "instance {} is attached over lanes {}..{} and its `embed_indptr` port describes \
             {described} lane(s): a program's ports state its own lanes' geometry, never the \
             next request's",
            attachment.instance, lanes.start, lanes.end
        )
    })
}

/// The runtime's engine over a backend's device: one shell, its program
/// plane, and the frame in flight.
pub struct Gpu<B: Host> {
    pub boot: B::DeviceBoot,
    contract_for: ContractFor,
    pub shell: Option<Shell<B>>,
    caps: Option<Capabilities>,
    next_fire: u64,
    next_frame: u64,

    pub sink: Option<engine::CompletionSink>,
    /// Where a move off the device still on the queue is told landed.
    landed: Option<engine::LandingSink>,

    pending: Option<(FrameId, Vec<PendingStep>)>,

    pub programs: B::Programs,

    adapters: BTreeMap<InstanceId, crate::adapter::Binding>,

    /// A device the host handed over, adopted at the load.
    handed: Option<B::Handed>,

    /// What the engine counts while it serves.
    live: engine::live::Live,

    /// The guest pipelines built after the load as of the last count
    /// (`Programs::built_serving`).
    guest_built: u64,
    /// Whether a build on a serving fire's path was warned of: the first
    /// is, at warn; the rest are counted, and named at debug.
    warned_in_fire: bool,
    /// The rows of the loaded model's widest token table: a token id a
    /// guest resolves at or past it is out of bounds.
    vocab: Option<u32>,
    /// What the work behind serving was last said to be doing
    /// ([`Gpu::behind_state`]).
    said_behind: Option<&'static str>,
    /// Each stale rate's points completed behind serving so far, by key: a
    /// slice resumes at its next unmeasured point.
    rates_resume: std::collections::HashMap<String, crate::rates::Resume>,
    /// The stale rates one of whose points is past one decode step: their
    /// rest runs once nothing is live.
    rates_long: std::collections::HashSet<String>,
}

impl<B: Host> Gpu<B> {
    /// The pending frame's numbers, each step's rows read back from its
    /// readout seats where not read yet ([`Engine::settle_frame`] takes
    /// them from here): what frees the seats on an idle turn.
    fn take_pending_rows(&mut self) -> EngineResult<()> {
        let (Some((_, pending)), Some(shell)) = (self.pending.as_mut(), self.shell.as_mut()) else {
            return Ok(());
        };
        for step in pending.iter_mut().filter(|step| step.rows.is_none()) {
            step.rows = Some(shell.rows_of(&step.landed).map_err(fault)?);
        }
        Ok(())
    }

    /// What the work left behind serving is doing now, said once each time
    /// it changes: running, or why it does not run.
    fn behind_state(&mut self, state: &'static str) {
        if self.said_behind != Some(state) {
            self.said_behind = Some(state);
            tracing::info!(state, "behind serving: {state}");
        }
    }

    /// One slice behind serving of a `[rates]` key or a dense kernel's curve
    /// the load serves by a figure stored under a stale key
    /// (`profile::fire::Tuned::behind`): the rates step timing that key
    /// alone over the rates the load serves by, or that curve alone. Each
    /// frame of it reads whether a request arrived and stops the slice where
    /// one did (`crate::rates::yielded`), so an arrival waits at most the
    /// frame in flight; the key is timed again whole at a later idle turn.
    /// Kept only where it had the device alone at a steady clock (the
    /// device idle as it starts, no other lane submitting while it runs,
    /// the device's settled clock before it no faster than after it): stored
    /// for the next load; else discarded and timed again later. This load
    /// serves on by the stale figure.
    fn rate_behind(&mut self, behind: &engine::BehindServing<'_>) -> EngineResult<bool> {
        use crate::backend::Tactics as _;
        let Some(shell) = self.shell.as_mut() else {
            return Ok(false);
        };
        let Some(key) = shell.rates_behind() else {
            return Ok(false);
        };
        if !crate::backend::Context::idle(shell.device()) {
            self.behind_state("waiting: the device is busy");
            return Ok(false);
        }
        // A key one of whose points is past one decode step runs (at the
        // platform's bound) only once nothing is live.
        let long = self.rates_long.contains(&key);
        if long && !behind.alone {
            self.behind_state(
                "waiting: a stale rate's next point runs longer than one decode step, and runs \
                 only once no process is live",
            );
            return Ok(false);
        }
        self.behind_state("running: the stale rates and curves");
        let Some(shell) = self.shell.as_mut() else {
            return Ok(false);
        };
        let max_submit = B::submit_bound(&self.boot);
        // Under a platform's bound, each frame held to one decode step of
        // this load, until a point is found past it.
        let cap = (!long)
            .then(|| shell.decode_step_ns())
            .flatten()
            .map(|ns| std::time::Duration::from_nanos(ns as u64));
        let resume = self.rates_resume.entry(key.clone()).or_default();
        resume.begin();
        let mut stated = shell.tuning().profile().rates();
        stated.stale = None;
        let reference = shell.tuning().gemm(shell.tier(), crate::submit::PROBE.0);
        let submitted = (behind.others)();
        let began = std::time::Instant::now();
        let Some(before) =
            crate::rates::clock_warmed_until::<B>(shell.device(), reference, behind.arrived)
                .map_err(fault)?
        else {
            return Ok(true);
        };
        let timed: crate::error::Result<profile::fire::Rates> =
            match key.strip_prefix(crate::rates::CURVE_BEHIND) {
                Some(curve) => {
                    let Some(wanted) = shell.curve_behind(curve) else {
                        tracing::warn!(
                            curve,
                            "rates behind serving: the curve names no kernel of this load; not timed"
                        );
                        shell.rate_timed(&key);
                        return Ok(true);
                    };
                    stated.kernel_tflops_by_shape = stated.kernel_tflops_by_shape.map(|all| {
                        all.into_iter().filter(|c| c.name() != curve).collect()
                    });
                    B::measure_curves(
                        shell.device(),
                        shell.tuning(),
                        max_submit,
                        &stated,
                        &[wanted],
                        shell.curve_top(),
                        behind.arrived,
                    )
                    .map(|curves| profile::fire::Rates {
                        kernel_tflops_by_shape: Some(curves),
                        ..profile::fire::Rates::default()
                    })
                }
                None => {
                    stated.unstate(&key);
                    B::measure_rates(
                        shell.device(),
                        shell.tuning(),
                        max_submit,
                        &stated,
                        behind.arrived,
                        crate::rates::Behind { resume, cap },
                    )
                    .and_then(|measured| match measured.throttled {
                        None => Ok(measured.rates),
                        Some(ratio) => Err(crate::error::Fault::Device {
                            call: "rates behind serving",
                            why: format!(
                                "the device's clock {ratio:.2}x slower at its end than its \
                                 settled start"
                            ),
                        }),
                    })
                }
            };
        let ms = began.elapsed().as_secs_f64() * 1e3;
        // A slice that stopped keeps the points it completed, where it had
        // the device alone at a steady clock; else they are timed again.
        let kept = |this: &mut Self, why: &str| -> EngineResult<()> {
            let Some(shell) = this.shell.as_ref() else {
                return Ok(());
            };
            let after = crate::rates::clock_read::<B>(shell.device(), reference).map_err(fault)?;
            let moved = (behind.others)() != submitted
                || crate::rates::throttled(before, after).is_some();
            let resume = this.rates_resume.entry(key.clone()).or_default();
            if moved {
                resume.discard();
            }
            tracing::info!(
                key = %key,
                ms,
                done = resume.done(),
                moved,
                "rates behind serving: `{key}` {why}; {} points kept{}",
                resume.done(),
                if moved { " (this slice's taken back: overlapped, or its clock moved)" } else { "" }
            );
            Ok(())
        };
        let timed = match timed {
            Ok(rates) => rates,
            Err(why) if crate::rates::yielded(&why) => {
                kept(self, "yielded to an arrival, resumed later")?;
                return Ok(true);
            }
            Err(why) if crate::rates::deferred(&why) || (cap.is_some() && !long) => {
                // A point past one decode step (or a step that cannot hold
                // its frames to it): the key's rest runs once nothing is
                // live, at the platform's bound.
                self.rates_long.insert(key.clone());
                kept(self, &format!("deferred until nothing is live ({why})"))?;
                return Ok(true);
            }
            Err(why @ crate::error::Fault::Device { call: "rates behind serving", .. }) => {
                tracing::info!(key = %key, ms, "rates behind serving: `{key}` discarded ({why})");
                return Ok(true);
            }
            Err(why) => return Err(fault(why)),
        };
        let after = crate::rates::clock_read::<B>(shell.device(), reference).map_err(fault)?;
        let why = if (behind.others)() != submitted {
            Some("another lane submitted while it ran".to_string())
        } else {
            crate::rates::throttled(before, after).map(|ratio| {
                format!("the device's clock {ratio:.2}x slower after it than its settled start")
            })
        };
        if let Some(why) = why {
            tracing::info!(
                key = %key,
                ms,
                "rates behind serving: `{key}` discarded ({why}); timed again later"
            );
            if let Some(resume) = self.rates_resume.get_mut(&key) {
                resume.discard();
            }
            return Ok(true);
        }
        self.rates_resume.remove(&key);
        self.rates_long.remove(&key);
        let Some(shell) = self.shell.as_mut() else {
            return Ok(false);
        };
        if let Err(why) = B::save_rates(shell.device(), &self.boot, &timed) {
            tracing::warn!(%why, key = %key, "rates behind serving: not kept in the profile");
        }
        let left = shell.rate_timed(&key);
        tracing::info!(
            key = %key,
            ms,
            left,
            "rates behind serving: `{key}` timed and stored for the next load (this load serves \
             on by its stale figure); {left} left"
        );
        Ok(true)
    }
    /// `step` held to the platform's bound on one submission, where the
    /// load states one (`crate::submit`): the shell told to walk it a layer
    /// at a time where it is past the bound whole, and its lanes' refusal,
    /// by name, where it is past it even so (its prompt lanes', or every
    /// lane's where it carries no prompt). `None` where it is admitted.
    fn admit_step(&mut self, step: &Step, at: engine::StepDone) -> Option<Error> {
        let shell = self.shell.as_mut()?;
        // The fires landed so far move the scale this step is read at
        // (`crate::submit::Measured`).
        shell.track();
        let held = shell.submit_bound()?;
        let rows = crate::submit::StepRows::of(step);
        let admitted = held.admit(rows);
        let (whole, per_layer) = held.step_ns(rows);
        let raw = held.raw_ns(rows).0;
        tracing::debug!(
            frame = at.frame.0,
            step = at.step,
            rows = rows.rows,
            lanes = rows.lanes,
            keys = rows.keys,
            prompt = rows.prompt,
            rung = held.rung_of(rows.rows),
            estimate_ms = whole / 1e6,
            per_layer_ms = per_layer / 1e6,
            worst = held.worst,
            probe_scale = held.scale,
            measured_worst = held
                .measured_rung(rows)
                .map_or_else(|| held.measured.worst(), |rung| held.measured.worst_of(rung)),
            measured_fires = held
                .measured_rung(rows)
                .map_or_else(|| held.measured.len(), |rung| held.measured.len_of(rung)),
            stored_ratio = held.stored_ratio(),
            measured_rung = ?held.measured_rung(rows),
            error = held.error_for(rows),
            floor_factor = held.floor_factor(),
            ratio = held.ratio_for(rows),
            bound_ms = held.bound_ns / 1e6,
            split = admitted.split(),
            layers_a_piece = ?admitted.layers_a_piece(),
            refused = matches!(admitted, crate::submit::Admitted::Refused { .. }),
            "submit bound: a step admitted"
        );
        if let crate::submit::Admitted::Refused { .. } = admitted {
            let why = held.refusal(rows, &admitted);
            tracing::warn!(frame = at.frame.0, step = at.step, %why, "submit bound: a step refused");
            let prompts: Vec<u32> = (0..step.lanes.len() as u32)
                .filter(|&lane| step.lanes[lane as usize].rows() > 1)
                .collect();
            let lanes = if prompts.is_empty() {
                (0..step.lanes.len() as u32).collect()
            } else {
                prompts
            };
            return Some(Error::Lanes(
                lanes
                    .into_iter()
                    .map(|lane| engine::LaneFault {
                        lane,
                        instance: None,
                        reason: why.clone(),
                    })
                    .collect(),
            ));
        }
        shell.split_next(admitted.layers_a_piece());
        shell.estimate_next(Some(crate::submit::Estimated {
            ns: raw,
            at,
            rows: rows.rows,
            lanes: rows.lanes,
            pieces: match admitted {
                crate::submit::Admitted::Layers { submissions, .. } => submissions,
                _ => 1,
            },
            hz: None,
        }));
        None
    }

    /// Under a stated submit bound, where `step` is past it even a layer at
    /// a time: the piece to fire before it and the rest, cut at the ratio
    /// read now (`crate::submit::next_cut`). `None` where it is admitted as
    /// it is, where no bound is stated, or where it cannot be cut (then
    /// refused by name as it is admitted, the reason said here).
    fn next_piece(&mut self, step: &Step) -> Option<(Step, Step)> {
        let shell = self.shell.as_mut()?;
        shell.track();
        let held = shell.submit_bound()?;
        let rows = crate::submit::StepRows::of(step);
        match crate::submit::next_cut(held, step, &shell.trace().facts) {
            Ok(None) => None,
            Ok(Some((piece, rest, piece_rows))) => {
                tracing::debug!(
                    rows = rows.rows,
                    lanes = rows.lanes,
                    keys = rows.keys,
                    piece_rows,
                    ratio = held.ratio_for(rows),
                    "submit bound: a step past the bound a layer at a time: a piece cut by rows"
                );
                Some((piece, rest))
            }
            Err(why) => {
                tracing::warn!(
                    rows = rows.rows,
                    lanes = rows.lanes,
                    keys = rows.keys,
                    %why,
                    "submit bound: a step past the bound a layer at a time cannot be cut by rows; \
                     refused by name"
                );
                None
            }
        }
    }

    #[must_use]
    pub fn new(boot: B::DeviceBoot, contract_for: ContractFor) -> Gpu<B> {
        Gpu {
            boot,
            contract_for,
            shell: None,
            caps: None,
            next_fire: 1,
            next_frame: 1,
            sink: None,
            landed: None,
            pending: None,
            programs: B::Programs::default(),
            adapters: BTreeMap::new(),
            handed: None,
            live: engine::live::Live::armed(),
            guest_built: 0,
            warned_in_fire: false,
            vocab: None,
            said_behind: None,
            rates_resume: std::collections::HashMap::new(),
            rates_long: std::collections::HashSet::new(),
        }
    }

    /// Counts the pipelines built on a serving fire's path since the last
    /// count, by its prepare, enqueue or settle: the load built every kernel
    /// pipeline its walks named and every guest step's, so each one built
    /// since is one a fire waited on ([`engine::live::Tally::BuiltInFire`]),
    /// the first warned of. A guest pipeline built at a registration or a
    /// bind, before any fire of it, is counted there instead
    /// ([`Self::count_registered`]).
    /// And the steps a backend that keeps recordings walked to keep: a shape
    /// recorded at its first serving fire ([`engine::live::Tally::Compiled`]).
    /// And the steps walked and kept by no recording: each ran eagerly. On a
    /// backend that keeps recordings that is a fire it should have replayed
    /// (a split step, bindings a recording cannot hold:
    /// [`engine::live::Tally::Unrecorded`]); on one that keeps none it is
    /// every fire, as the backend states ([`engine::live::Tally::Eager`]).
    fn count_built(&mut self) {
        let built = self
            .shell
            .as_mut()
            .map_or_else(Vec::new, Shell::built_serving);
        // A kernel is built once, so each is named once.
        for kernel in built {
            self.live.tally(engine::live::Tally::BuiltInFire, 1);
            self.warn_in_fire(kernel);
        }
        // What a dispatch binds, built on the fire's path: the load's walk
        // built every one it reached.
        let bound = self.shell.as_mut().map_or(0, Shell::bound_serving);
        if bound > 0 {
            self.live.tally(engine::live::Tally::BoundInFire, bound);
            self.warn_in_fire("what a dispatch binds (a bind group the load's walk did not build)");
        }
        let recorded = self.shell.as_mut().map_or(0, Shell::recorded_serving);
        if recorded > 0 {
            self.live.tally(engine::live::Tally::Compiled, recorded);
        }
        // A guest pipeline built on the fire's own path: a registration or
        // a bind took its own builds' count where it landed.
        let guest = self.guest_built_since();
        if guest > 0 {
            self.live.tally(engine::live::Tally::BuiltInFire, guest);
            self.warn_in_fire("a guest step's pipeline");
        }
        // The submit-bound probes, apart from the model's fires: recorded at
        // load where the backend keeps recordings.
        let probed = self.shell.as_mut().map_or(0, Shell::probed_serving);
        if probed > 0 {
            self.live.tally(engine::live::Tally::Probed, probed);
        }
        let unkept = self.shell.as_mut().map_or(0, Shell::unkept_serving);
        if unkept > 0 {
            let tally = if B::RECORDS {
                engine::live::Tally::Unrecorded
            } else {
                engine::live::Tally::Eager
            };
            self.live.tally(tally, unkept);
        }
    }

    /// Counts the tier moves through the host that waited for the frames in
    /// flight ([`engine::live::Tally::Drained`]).
    fn count_drained(&mut self) {
        let drained = self.shell.as_mut().map_or(0, Shell::drained_serving);
        if drained > 0 {
            self.live.tally(engine::live::Tally::Drained, drained);
        }
    }

    /// The guest pipelines built since the last count, which this one takes.
    fn guest_built_since(&mut self) -> u64 {
        let guest = self.programs.built_serving();
        let since = guest.saturating_sub(self.guest_built);
        self.guest_built = guest;
        since
    }

    /// Counts the guest pipelines a registration or a bind built after the
    /// load: a program's code built at its own registration, before its
    /// first fire, which no fire waits on
    /// ([`engine::live::Tally::BuiltAtRegister`]).
    fn count_registered(&mut self) {
        let guest = self.guest_built_since();
        if guest > 0 {
            self.live.tally(engine::live::Tally::BuiltAtRegister, guest);
            tracing::debug!(
                engine = B::KIND,
                guest,
                "a registration built guest pipelines the load did not"
            );
        }
    }

    /// `result`, its refusal counted ([`engine::live::Tally::Refused`]).
    fn refused<T>(&self, result: EngineResult<T>) -> EngineResult<T> {
        if result.is_err() {
            self.live.tally(engine::live::Tally::Refused, 1);
        }
        result
    }

    /// Warns once of a build on a serving fire's path, the graph-capture
    /// rule's violation; later ones are named at debug, and all counted.
    fn warn_in_fire(&mut self, what: &str) {
        if std::mem::replace(&mut self.warned_in_fire, true) {
            tracing::debug!(engine = B::KIND, what, "a serving fire built a pipeline");
        } else {
            tracing::warn!(
                engine = B::KIND,
                what,
                "a serving fire built a pipeline the load did not (counted built_in_fire; \
                 later ones are counted, named at debug)"
            );
        }
    }

    /// An engine over a device the host already opened.
    #[must_use]
    pub fn with_device(
        boot: B::DeviceBoot,
        contract_for: ContractFor,
        handed: B::Handed,
    ) -> Gpu<B> {
        Gpu {
            handed: Some(handed),
            ..Gpu::new(boot, contract_for)
        }
    }

    #[must_use]
    pub fn boot(&self) -> &B::DeviceBoot {
        &self.boot
    }

    #[must_use]
    pub fn contract_for(&self) -> ContractFor {
        self.contract_for
    }

    pub fn shell_mut(&mut self) -> Option<&mut Shell<B>> {
        self.shell.as_mut()
    }

    #[must_use]
    pub fn capabilities(&self) -> Option<&Capabilities> {
        self.caps.as_ref()
    }

    pub fn open(&mut self, slot: u32) -> EngineResult<()> {
        self.loaded_mut()?.open(slot).map_err(fault)
    }

    pub fn loaded(&self) -> EngineResult<&Shell<B>> {
        self.shell.as_ref().ok_or_else(unloaded::<B>)
    }

    /// Where an instance of `program` bound at `extents` and its pair lay
    /// their state ([`crate::guest::state::GuestLayout`]): its stages'
    /// heaps, and the pair's score room where a read of it switches score
    /// capture on. `None`
    /// before a load or for a program the plane does not hold.
    pub fn pair_layout(
        &self,
        program: ProgramId,
        extents: &engine::BindExtents,
    ) -> EngineResult<Option<crate::guest::state::GuestLayout>> {
        let (Some(shell), Some(package)) = (self.shell.as_ref(), self.programs.package(program))
        else {
            return Ok(None);
        };
        let scores = shell.score_room(
            package
                .values
                .iter()
                .filter_map(|value| Some((value.output.as_deref()?, value.lane))),
        );
        // A beam group's state, laid for the search's depth (the bind's
        // `key_len`, its end slot), the load's kv page and its frames in
        // flight.
        let beam = crate::beam::lanes_of(package)
            .map(|lanes| {
                crate::guest::state::BeamLayout::of(
                    lanes,
                    extents.key_len,
                    shell.paging().page_size,
                    u32::try_from(shell.frames_in_flight()).unwrap_or(u32::MAX),
                )
                .map(|layout| match shell.pools().rs_rows() {
                    // On a hybrid its set's rs entries are held with it.
                    Some(rows) => layout.buffering(shell.paging().page_size, rows.tokens),
                    None => layout,
                })
            })
            .transpose()
            .map_err(|why| Error::Program(format!("program {program}'s state: {why}")))?;
        crate::guest::state::GuestLayout::of_plans(
            &package.plans,
            &crate::guest::state::extents_of(extents),
            scores,
        )
        .map(|layout| Some(layout.with_beam(beam)))
        .map_err(|why| Error::Program(format!("program {program}'s state: {why}")))
    }

    pub fn loaded_mut(&mut self) -> EngineResult<&mut Shell<B>> {
        self.shell.as_mut().ok_or_else(unloaded::<B>)
    }

    /// Tells the sink what the lanes of the frame just settled inside
    /// `submit` wrote, step by step, before `submit` returns: a fold's
    /// count is the host's from the fire's submit, so this waits on
    /// nothing. A step that wrote nothing tells nothing; `submit`'s result
    /// answers for it.
    fn tell_wrote(&mut self) {
        let (Some(sink), Some((frame, steps))) = (self.sink.as_ref(), self.pending.as_mut()) else {
            return;
        };
        for (step, pending) in steps.iter_mut().enumerate() {
            let mut outcome = engine::StepOutcome {
                beam_common: Vec::new(),
                wrote: std::mem::take(&mut pending.wrote),
                faulted: std::mem::take(&mut pending.faulted),
            };
            // The frame settled: every step's forward landed.
            if let Some(fed) = pending.fed.take() {
                outcome = crate::settle::merged(outcome, fed.faults());
                outcome.beam_common = fed.commons();
            }
            if !outcome.wrote.is_empty()
                || !outcome.faulted.is_empty()
                || !outcome.beam_common.is_empty()
            {
                let at = engine::StepDone {
                    frame: *frame,
                    step: step as u32,
                };
                (sink)(at, outcome);
            }
        }
    }
}

/// Where `shell`'s guests place their tables and cells.
fn place<B: Host>(shell: &Shell<B>) -> Place<'_, B> {
    Place {
        device: shell.device(),
        pools: shell.pools(),
        align: B::table_align(shell.device()),
        seats: shell.readout.iter().chain(&shell.pixel_out).collect(),
        max_lanes: shell.budget().max_lanes,
        planes: u32::try_from(shell.output_names().len() + shell.pixel_planes())
            .unwrap_or(u32::MAX),
        shelf: &shell.shelf,
    }
}

fn unloaded<B: Host>() -> Error {
    Error::Load(format!("the {} engine has no model loaded", B::KIND))
}

fn kv_copy_direction(src: MemoryDomain, dst: MemoryDomain) -> &'static str {
    match (src, dst) {
        (MemoryDomain::HostPinned, MemoryDomain::HostPinned) => {
            "`copy_kv` host-pinned to host-pinned, which is the caller's own memmove"
        }
        (MemoryDomain::HostPinned, _) => {
            "`copy_kv` out of host-pinned memory, which needs a swap pool this load does not \
             reserve"
        }
        (_, MemoryDomain::HostPinned) => {
            "`copy_kv` into host-pinned memory, which needs a swap pool this load does not \
             reserve"
        }
        _ => "`copy_kv` between the domains named, neither of which is this load's own device",
    }
}

#[must_use]
pub fn refusal(refusal: Refusal) -> Error {
    match refusal {
        Refusal::Closed { what, id } => Error::Closed { what, id },
        Refusal::Program(why) => Error::Program(why),
        Refusal::Claimed(refusal) => Error::Claimed(refusal),
    }
}

#[must_use]
pub fn fault(fault: Fault) -> Error {
    // A fault one lane earns refuses that lane alone; the step's other
    // lanes are the runtime's to submit again (`engine::Error::Lanes`).
    if let Some(lane) = fault.lane() {
        return Error::lane(lane, fault);
    }
    match fault {
        Fault::Deviceless | Fault::Device { .. } => Error::Device(fault.to_string()),
        Fault::TierFull(_) => Error::TierFull(fault.to_string()),

        Fault::PatchPayload { .. } => Error::Invalid(fault.to_string()),

        Fault::Bake(_)
        | Fault::Load(_)
        | Fault::Backing { .. }
        | Fault::Mapped { .. }
        | Fault::Param { .. }
        | Fault::Recipe(_)
        | Fault::Shader { .. }
        | Fault::Unbound { .. } => Error::Load(fault.to_string()),
        Fault::Ceiling { what, need, have } => Error::Impossible(format!(
            "this fire wants {need} {what} and the load reserved {have}"
        )),

        Fault::Fragmented { .. } => Error::Device(fault.to_string()),

        Fault::Unaffine { .. } | Fault::Unstructured { .. } => Error::Load(fault.to_string()),
        Fault::Straddled { .. } => Error::Load(fault.to_string()),

        Fault::Mask { .. }
        | Fault::MaskRows { .. }
        | Fault::Maskless { .. }
        | Fault::MaskUnread { .. }
        | Fault::Positions { .. } => Error::Invalid(fault.to_string()),

        Fault::Adapterless { .. } | Fault::AdapterWord { .. } => Error::Invalid(fault.to_string()),

        Fault::AdapterSlots { .. } => Error::Exhausted {
            resource: "adapter slots",
            wanted: 1,
            available: 0,
        },

        Fault::Adapter { .. } => Error::Load(fault.to_string()),
        Fault::Blob { .. } => Error::Load(fault.to_string()),
        Fault::Claimed(refusal) => Error::Claimed(refusal),
        Fault::Program { .. } => Error::Program(fault.to_string()),
        Fault::Vulkan { .. }
        | Fault::Wgpu { .. }
        | Fault::DeviceLost { .. }
        | Fault::NoDevice { .. }
        | Fault::Blocking { .. } => Error::Device(fault.to_string()),
        Fault::Fire(_) => Error::Invalid(fault.to_string()),

        Fault::Residency(_)
        | Fault::Workgroups { .. }
        | Fault::StorageBuffers { .. }
        | Fault::StageBuffers { .. } => Error::Impossible(fault.to_string()),
    }
}

/// The load's budgets: its logit readouts sized to what `trace` reads, not a
/// general figure, so a vocab-wide head carves only the rows a fire can read.
fn bake_budgets(budgets: &LoadBudgets, trace: &Trace) -> Budget {
    Budget {
        max_lanes: budgets.max_lanes,
        max_tokens: budgets.max_tokens,
        buckets: budgets.buckets.clone(),
        max_adapters: budgets.max_adapters,
        readouts_per_lane: poem_compiler::arena::readouts_per_lane(trace),
    }
}

#[must_use]
pub fn patch_ladder(trace: &Trace, budgets: &LoadBudgets) -> Option<PatchLadder> {
    const DERIVED_PATCH_CEILING: u32 = 4096;

    let declares_patches = trace.values.iter().any(|decl| {
        matches!(&decl.ty, poem_ir::Ty::Tensor { shape, .. }
            if shape.first().and_then(|dim| dim.axis()) == Some(poem_ir::RowAxis::Patches))
    });
    if !declares_patches {
        return None;
    }

    let max_patches = budgets
        .max_patches
        .unwrap_or_else(|| budgets.max_tokens.min(DERIVED_PATCH_CEILING))
        .max(PATCH_LATTICE_FLOOR);
    let mut buckets = Vec::new();
    let mut rung = PATCH_LATTICE_FLOOR;
    while rung < max_patches {
        buckets.push(rung);
        rung = rung.saturating_mul(2);
    }
    buckets.push(max_patches);
    Some(PatchLadder {
        max_images: budgets
            .max_images
            .unwrap_or(max_patches / PATCH_LATTICE_FLOOR)
            .max(1),
        max_patches,
        buckets,
    })
}

pub fn patch_bytes(
    patches: &[u8],
    element: poem_ir::Dtype,
) -> std::result::Result<Vec<u8>, &'static str> {
    match element {
        poem_ir::Dtype::U8 => Ok(patches.to_vec()),
        _ => Err("a media submission against a plan whose pixel rows are not u8 bytes"),
    }
}

pub fn voxel_bytes(
    payload: &[f32],
    element: poem_ir::Dtype,
) -> std::result::Result<Vec<u8>, &'static str> {
    match element {
        poem_ir::Dtype::Bf16 => Ok(payload
            .iter()
            .flat_map(|&v| {
                let bits = v.to_bits();
                (((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16).to_le_bytes()
            })
            .collect()),
        poem_ir::Dtype::F32 => Ok(payload.iter().flat_map(|&v| v.to_le_bytes()).collect()),
        _ => Err("a voxel payload against a plan whose voxel element is neither bf16 nor f32"),
    }
}

fn adapter_of<B: Host>(
    shell: &mut Shell<B>,
    package: &eta_compiler::codegen::launch::LaunchPackage,
    instance: InstanceId,
    seeds: &[(u32, Vec<u8>)],
) -> EngineResult<Option<crate::adapter::Binding>> {
    let Some(sink) = crate::adapter::sink_of(package).map_err(fault)? else {
        return Ok(None);
    };
    let seats = shell.bank_seats();
    let site = sink.site().map_err(fault)?;
    let mut built: Vec<(String, Vec<u8>)> = Vec::new();
    for (role, channel) in &sink.planes {
        let wire = seeds
            .iter()
            .find(|(seeded, _)| seeded == channel)
            .map(|(_, bytes)| bytes.as_slice())
            .ok_or_else(|| {
                Error::Load(format!(
                    "this program's `lora` sink reads its `{}` plane out of channel \
                     {channel} and this bind seeded nothing into it; an adapter's \
                     weights are the seed, because the fire path never reads the cell, \
                     so an unseeded plane is a correction of zero \
                     that nobody asked for",
                    role.bank()
                ))
            })?;
        built.extend(crate::adapter::planes_of(*role, site, wire, &seats).map_err(fault)?);
    }
    let planes: Vec<crate::weights::AdapterPlane<'_>> = built
        .iter()
        .map(|(bank, bytes)| crate::weights::AdapterPlane {
            bank: bank.as_str(),
            bytes,
        })
        .collect();
    shell
        .bind_adapter(crate::adapter::Source::Own {
            instance,
            planes: &planes,
        })
        .map(Some)
        .map_err(fault)
}

impl<B: Host> Engine for Gpu<B>
where
    Gpu<B>: Send,
{
    fn kind(&self) -> &'static str {
        B::KIND
    }

    fn device_facts(&self) -> Option<&DeviceFacts> {
        self.caps.as_ref().map(|caps| &caps.device)
    }

    fn host_room(&self) -> Option<engine::HostRoom> {
        crate::backend::Context::host_room(self.shell.as_ref()?.device())
    }

    fn committed_covers(&self) -> &'static [&'static str] {
        B::committed_covers()
    }

    fn measured_at_load(&self) -> Vec<String> {
        self.shell
            .as_ref()
            .map(crate::serve::Shell::measured_at_load)
            .unwrap_or_default()
    }

    fn skips_measuring(&self) -> bool {
        B::measuring(&self.boot) == engine::rates::Measuring::Skip
    }

    fn cover_fingerprint(&self) -> Option<String> {
        let shell = self.shell.as_ref()?;
        match shell.cover_fingerprint(&B::measured_on(shell.device())) {
            Ok(print) => {
                tracing::info!(
                    fingerprint = %print,
                    "the host cover's key on this engine: the device, its kernels' code, the \
                     guest emitter's version and the tuning its fires run by"
                );
                Some(print)
            }
            Err(why) => {
                tracing::warn!(%why, "the host cover is keyed by nothing here: measured this boot");
                None
            }
        }
    }

    fn live(&self) -> engine::live::Live {
        self.live.clone()
    }

    fn bind_thread(&mut self) -> EngineResult<()> {
        match self.shell.as_ref() {
            Some(shell) => shell.bind_thread().map_err(fault),
            None => Ok(()),
        }
    }

    fn load(&mut self, request: LoadRequest) -> EngineResult<Loaded> {
        if self.shell.is_some() {
            return Err(Error::Load(format!(
                "this {} engine already has a model loaded; one shell per engine",
                B::KIND
            )));
        }
        let LoadRequest {
            trace,
            checkpoint,
            budgets,
            residency,
            ordinal,
            frames_in_flight,
            steps,
        } = request;

        if ordinal > 0 {
            return Err(Error::unsupported(B::KIND, "device ordinal selection"));
        }
        self.vocab = token_rows(&trace);

        let Checkpoint::Path(path) = checkpoint else {
            return Err(Error::Load(format!(
                "the {} shell lands a checkpoint or nothing runs; `Checkpoint::None` has no \
                 weightless path here",
                B::KIND
            )));
        };
        let contract = (self.contract_for)(&trace, &path).map_err(Error::Load)?;

        let patches = patch_ladder(&trace, &budgets);
        let voxels = poem_compiler::VoxelLadder::of(
            &trace,
            budgets.max_voxels,
            budgets.max_clips,
            budgets.max_lanes,
        );
        // A served lane names its slots by ref: the only seat-addressed
        // fires of a served load are its own naming walk's, the widest a
        // cut's lanes, and those are the seats it is asked to lay.
        let slots = poem_compiler::state_slots(budgets.slots, &trace, crate::serve::naming_lanes());
        let (disk_kv_budget, disk_kv_dir, seats) = (
            residency.disk_kv_budget,
            residency.disk_kv_dir.clone(),
            residency.kv_seats.unwrap_or(budgets.max_lanes),
        );
        let budget = bake_budgets(&budgets, &trace);
        let mut shell = Shell::load(Boot {
            trace,
            contract: &contract,
            checkpoint: &path,
            budget,
            patches,

            profile: None::<DeviceProfile>,
            page_size: budgets.page_size,
            context: budgets.max_context,
            slots,

            runahead: engine::runahead::Runahead::of(frames_in_flight),
            voxels,
            residency,
            device: &self.boot,
            handed: self.handed.take(),
            check: B::fires_stated(&self.boot)
                .then_some(crate::serve::CheckRungs::To(budgets.max_tokens)),
            places_programs: true,
        })
        .map_err(fault)?;
        self.programs.load(&place(&shell)).map_err(refusal)?;
        shell.programs_placed();
        let started = std::time::Instant::now();
        let walked = shell.walk_binds().map_err(fault)?;
        if walked > 0 {
            tracing::info!(
                engine = B::KIND,
                walked,
                ms = started.elapsed().as_millis() as u64,
                "every fire shape the load serves walked bind-only: what a dispatch binds is built"
            );
        }
        // What the load itself walked is the load's: serving counts from here.
        shell.unkept_serving();
        shell.bound_serving();
        <B as crate::serve::Serve>::serving(&mut shell).map_err(fault)?;
        // The fire the shell took, which a binding may have bounded below
        // the one asked.
        let (max_lanes, max_tokens) = (shell.budget().max_lanes, shell.budget().max_tokens);
        let max_readouts = u32::try_from(poem_compiler::arena::readouts_ceiling(shell.budget()))
            .unwrap_or(u32::MAX);
        shell
            .seat_disk_kv(
                disk_kv_budget,
                disk_kv_dir.as_deref(),
                seats,
                budgets.max_context,
            )
            .map_err(fault)?;
        // The dense kernel at each rung as a stored or committed settle of
        // the load's key decided it, else the nearest settle's or the
        // curves': nothing fires before ready; what no settle decided is
        // weighed behind serving (`Engine::measure_behind`) and stored for
        // the next load. What a newly taken kernel binds is built by a
        // second bind walk, and what it walked is the load's.
        let settling = std::time::Instant::now();
        shell.settle_dense().map_err(fault)?;
        tracing::info!(
            settle_ms = settling.elapsed().as_millis() as u64,
            "load phases: the dense settle's picks (nothing fires before ready)"
        );
        shell.walk_binds().map_err(fault)?;
        shell.unkept_serving();
        shell.bound_serving();

        let trace_name = shell.trace().name.clone();
        let (weight_bytes, arena_bytes, pool_bytes, input_bytes) = shell.footprint();
        let weights_resident = !shell.weights_stream();

        let paging = shell.paging();

        let pools = shell.pool_facts().map_err(fault)?;
        let chunks = shell.chunk_facts().map_err(fault)?;
        let memory = crate::backend::Context::memory(shell.device());
        if memory.unified() {
            tracing::info!(
                device = crate::backend::Context::name(shell.device()),
                disk_chunks = chunks.disk_chunks,
                "{}",
                engine::DeviceMemory::NO_HOST_TIER
            );
        }

        let caps = Capabilities {
            device: DeviceFacts {
                backend: B::KIND.to_string(),
                domain: B::domain(&self.boot),
                sms: shell.cores(),
                unified_memory: memory.unified(),

                fp8_native: false,
                native_mxfp4_moe: false,

                storage_alignment: 256,

                storage_max_tile_bytes: shell.max_buffer(),

                codegen_backend: None,
                // The WGSL emitter compiles a program's steps from its ops and
                // has no library regions: no form is served as a library call.
                library_forms: engine::LibraryForms {
                    draw: false,
                    bf16_cut: false,
                },
            },
            pools,
            chunks,
            limits: FireLimits {
                max_lanes,
                max_tokens,
                max_page_refs: paging.pages_per_slot.saturating_mul(max_lanes),
                max_context: paging.context(),
                // Where the platform states a bound on one submission, a
                // prompt is cut to the widest laid rung whose step at the
                // context is admitted under it (`crate::submit`).
                prefill_rows: shell
                    .submit_bound()
                    .and_then(|held| held.prefill_rows(paging.context()))
                    .unwrap_or(0),
                prefill_lanes: 0,
                max_readouts,
                // A fire's staged rs table holds as many carried tokens as
                // its widest bucket's rows, every lane's summed, a decode
                // lane's included (`inputs::rs_carry_ints`).
                replayed_rows: if shell.pools().rs_rows().is_some() { max_tokens } else { 0 },
            },
            has_lora: true,
            has_attn_page_mask: false,
            draft_block: shell.trace().drafter.map_or(0, |d| d.rows),
            draft_mask_token: shell.trace().drafter.map_or(0, |d| d.mask_token),
            draft_proposals_from: shell.trace().drafter.map_or(1, |d| d.proposals_from),

            ports: B::ports(),
            geometry: GeometryClass::DeviceGeometry,

            kv_handle: None,
            media_encode: false,
            rs_verbs: shell.serves_rs_verbs(),
            // Windowed layers are ring + buffer (CONTRACT-BC 19, 21).
            windowed_state: shell.pools().rings().is_some(),
            // The prompt scans land the state as of rows inside a lane, a
            // record a snapshot in pieces (CONTRACT-BC 16).
            rs_snapshots: crate::rs::snapshots(shell.trace()),
            bidirectional_attention: false,
            adapter_attachments: false,
            facts: shell.trace().facts.clone(),
            inputs: shell.inputs(),
            classes: shell.classes(),
            // The platform's stated bound on one submission (KGSL's on
            // Android, none elsewhere) and the rates this load measured.
            // Where the load skipped measuring them, the rates its bound's
            // estimate was timed at (`crate::rates::for_bound`): what holds
            // its steps to the bound holds its guest stages to it too.
            guest_cost: {
                let rates = crate::backend::Tactics::profile(shell.tuning()).rates();
                let held = shell.submit_bound().map(|held| &held.estimate.rates);
                if let Some(c) = held
                    && (rates.launch_ns.is_none() || rates.elementwise_gbps.is_none())
                {
                    tracing::info!(
                        launch_ns = c.launch_ns,
                        elementwise_gbps = c.elementwise_bytes_per_ns,
                        "a guest stage is costed against the submit bound at the rates the \
                         bound's probe timed (no layer states them, measure = false): probed"
                    );
                }
                engine::GuestCost {
                    submit_bound_ns: shell.submit_bound().map(|held| held.bound_ns as u64),
                    launch_ns: rates
                        .launch_ns
                        .or_else(|| held.map(|c| c.launch_ns.round() as u64)),
                    elementwise_gbps: rates
                        .elementwise_gbps
                        .or_else(|| held.map(|c| c.elementwise_bytes_per_ns)),
                }
            },
            // A beam group's page list (`beam_list`), read at a search's finish.
            host_ops: engine::HostOps { beam_list: true },
            mixed_lanes: engine::MixedLanes::Any,
            frames_in_flight: engine::stated_frames(shell.frames_in_flight())?,
            steps: engine::StatedSteps::default(),
        };

        // The step compositions the runtime's builder forms over this load's
        // facts (CONTRACT-BC 24). Interim: computed once the caps are; the walk's own branch
        // (sc/gpu-arm-before-ready) consumes it before ready.
        let mut caps = caps;
        caps.steps = engine::StatedSteps::of(
            &steps,
            &engine::StepFacts::stated(&caps, &engine::ReadoutFacts::default()),
        );
        tracing::info!(
            "engine-gpu: step set: {}; its arm walk does not key on the set yet, so none \
             of the {} is checked against what it records",
            caps.steps.shapes.census(),
            caps.steps.shapes.len(),
        );

        self.shell = Some(shell);
        self.caps = Some(caps.clone());
        Ok(Loaded {
            facts: LoadFacts {
                trace_name,
                weight_bytes,

                weights_resident,
                weights_from_cache: false,
                arena_bytes,
                pool_bytes,
                input_bytes,

                pool_committed_bytes: pool_bytes,
                pool_high_water_bytes: pool_bytes,
            },
            caps,
        })
    }

    fn commit_segments(&mut self, segments: u32) -> EngineResult<()> {
        self.loaded_mut()?.commit_segments(segments).map_err(fault)
    }

    fn release_segments(&mut self, segments: u32) -> EngineResult<()> {
        self.loaded_mut()?.release_segments(segments).map_err(fault)
    }

    fn evacuate(&mut self, evacuation: &engine::chunk::Evacuation) -> EngineResult<()> {
        let Gpu {
            shell, programs, ..
        } = self;
        let shell = shell.as_mut().ok_or_else(unloaded::<B>)?;
        programs.quiesce().map_err(refusal)?;
        // All or nothing (`Engine::evacuate`): the copies recorded and the
        // table moved, nothing committed; guest tables and cells, which
        // bind their slots' chunks by place, resolved against the moved
        // table and their runs claimed; only then is a copy committed.
        let Some(staged) = shell.stage_evacuation(evacuation).map_err(fault)? else {
            return Ok(());
        };
        let rebound = match programs.resolve_rebind(shell.pools()) {
            Ok(rebound) => rebound,
            Err(refused) => {
                shell.abort_evacuation(staged);
                return Err(refusal(refused));
            }
        };
        shell.commit_evacuation(staged).map_err(fault)?;
        programs.install_rebind(rebound).map_err(refusal)
    }

    fn chunk_table(&self) -> EngineResult<engine::chunk::ChunkTable> {
        let shell = self.shell.as_ref().ok_or_else(unloaded::<B>)?;
        Ok(shell.pools().chunk_table().clone())
    }

    fn program_facts(&self, program: ProgramId) -> EngineResult<engine::program::ProgramFacts> {
        let Some(shell) = self.shell.as_ref() else {
            return Ok(engine::program::ProgramFacts::default());
        };
        self.programs.facts(program, &place(shell)).map_err(refusal)
    }

    /// What an instance of `program` bound at `extents` and its pair keep
    /// as their state ([`crate::guest::state`]): its stages' heaps, the
    /// instance's own, and, where it reads captured attention scores, the
    /// pair's lane-indexed room a fire lays its capturing lanes' rows in.
    /// Nothing of a guest lies in a fire's span: this engine states no
    /// fire term.
    fn guest_state(
        &self,
        program: ProgramId,
        extents: &engine::BindExtents,
        // The class bound: the engine half states per-class arms.
        _class: engine::ports::GeometryClass,
    ) -> EngineResult<engine::GuestState> {
        let Some(shell) = self.shell.as_ref() else {
            return Ok(engine::GuestState::default());
        };
        let Some(layout) = self.pair_layout(program, extents)? else {
            return Ok(engine::GuestState::default());
        };
        let facts = crate::guest::state::Facts::of(shell.pools()).map_err(fault)?;
        layout
            .state(&facts)
            .map_err(|why| Error::Program(format!("program {program}'s state: {why}")))
    }

    /// Every registration: a ring the device keeps in its `Channel` slots
    /// or on the host.
    fn holds_channel(&self, registration: &ChannelRegistration) -> bool {
        let _ = registration;
        true
    }

    fn ring_facts(
        &self,
        registration: &ChannelRegistration,
    ) -> EngineResult<Option<engine::program::RingFacts>> {
        let Some(shell) = self.shell.as_ref() else {
            return Ok(None);
        };
        self.programs
            .ring_facts(registration, &place(shell))
            .map_err(refusal)
    }

    fn rebind_slots(&mut self, rebind: engine::program::Rebind) -> EngineResult<()> {
        let shell = self.shell.as_ref().ok_or_else(unloaded::<B>)?;
        self.programs
            .rebind_slots(shell.pools(), &rebind)
            .map_err(refusal)
    }

    /// Returns once every copy submitted so far has landed: every frame
    /// and device copy in flight, and a slot file read's copy onto the
    /// device (`Shell::settle_copies`). Off the hot path: an aborted move
    /// waits on it.
    fn settle_copies(&mut self) -> EngineResult<()> {
        self.loaded_mut()?.settle_copies().map_err(fault)
    }

    /// `false` only for a move off the device into mapped host chunks: it
    /// rides the queue past the call, and its device slots, once back, may
    /// be written by anything; told landed from its frame's fence. Every
    /// other copy's sources may go back at once (`Shell::copy_slots_landing`
    /// states why for each kind).
    fn copy_slots_landing(
        &mut self,
        copy: &engine::chunk::SlotCopy,
        token: u64,
    ) -> EngineResult<bool> {
        let landed = self
            .landed
            .clone()
            .map(|sink| -> Box<dyn FnOnce() + Send> { Box::new(move || sink(token, engine::Landing::Landed)) });
        let copied = self
            .loaded_mut()?
            .copy_slots_landing(copy, landed)
            .map_err(fault);
        self.count_drained();
        copied
    }

    fn on_landed(&mut self, sink: engine::LandingSink) {
        self.landed = Some(sink);
    }

    fn copy_slots(&mut self, copy: &engine::chunk::SlotCopy) -> EngineResult<()> {
        let copied = self.loaded_mut()?.copy_slots(copy).map_err(fault);
        self.count_drained();
        copied
    }

    fn read_tier_slots(
        &mut self,
        slots: &[(engine::chunk::ShapeId, Vec<u32>)],
    ) -> EngineResult<Vec<u8>> {
        let read = self.loaded_mut()?.read_tier_slots(slots).map_err(fault);
        self.count_drained();
        read
    }

    fn write_tier_slots(
        &mut self,
        slots: &[(engine::chunk::ShapeId, Vec<u32>)],
        bytes: &[u8],
    ) -> EngineResult<()> {
        let written = self
            .loaded_mut()?
            .write_tier_slots(slots, bytes)
            .map_err(fault);
        self.count_drained();
        written
    }

    fn weight_residency(
        &mut self,
        residency: &engine::chunk::WeightResidency,
    ) -> EngineResult<u64> {
        // This engine holds back none of the chunks it states
        // (`ChunkFacts::held_bytes` is 0 at load): the answer repeats it.
        self.loaded_mut()?
            .weight_residency(residency)
            .map(|()| 0)
            .map_err(fault)
    }

    fn weight_uses(&mut self) -> EngineResult<engine::chunk::WeightUses> {
        self.loaded_mut()?.weight_uses().map_err(fault)
    }

    fn submit(&mut self, frame: &FrameSubmission) -> EngineResult<FrameTicket> {
        let submitted = self.submit_frame(frame);
        // A frame whose step refused its lanes by name fired the steps
        // before it alone: a refusal too.
        if submitted
            .as_ref()
            .is_ok_and(|ticket| !ticket.refused.is_empty())
        {
            self.live.tally(engine::live::Tally::Refused, 1);
        }
        self.refused(submitted)
    }

    fn register_program(&mut self, registration: &ProgramRegistration) -> EngineResult<ProgramId> {
        let registered = self.programs.register(registration).map_err(refusal);
        self.count_registered();
        self.refused(registered)
    }

    /// The runtime's standard samplers, before ready: each adopted and its
    /// stages lowered as its registration does them, and every step call
    /// their stages dispatch held to a pipeline the load built
    /// (`Programs::load`), so a request that registers one, or binds it,
    /// builds nothing. A program that does not adopt, or that dispatches a
    /// call the load could not build, fails the load by name. No stage holds
    /// code of its own (`guest::Compiled`): nothing here is keyed on a
    /// program's hash, which its channels' capacities enter.
    fn prewarm_programs(&mut self, registrations: &[ProgramRegistration]) -> EngineResult<()> {
        if self.shell.is_none() {
            return Err(Error::Program(
                "the standard samplers prewarmed before load".to_owned(),
            ));
        }
        let started = std::time::Instant::now();
        let prewarmed = crate::program::prewarm(registrations).map_err(refusal)?;
        for &call in &prewarmed.calls {
            if let Some(why) = self.programs.unbuilt(call) {
                return Err(Error::Load(format!(
                    "a standard sampler dispatches guest step call {call}, which the load did \
                     not build: {why}"
                )));
            }
        }
        tracing::info!(
            engine = B::KIND,
            programs = prewarmed.programs,
            stages = prewarmed.stages,
            calls = prewarmed.calls.len(),
            built_serving = self.programs.built_serving(),
            slowest_ms = prewarmed.slowest.as_secs_f64() * 1e3,
            ms = started.elapsed().as_secs_f64() * 1e3,
            hashes = format_args!("{:016x}", prewarmed.hashes),
            forms = format_args!("{:016x}", prewarmed.forms),
            "the standard samplers adopted before ready; every step call they dispatch has its \
             pipeline from the load"
        );
        Ok(())
    }

    /// The program adopted and its stages lowered in the returned compile,
    /// off the engine's thread (`program::adopt`); nothing of the plane's
    /// is touched until the finish.
    fn prepare_program(
        &mut self,
        registration: &ProgramRegistration,
    ) -> EngineResult<engine::ProgramCompile> {
        let registration = registration.clone();
        Ok(Box::new(move || {
            engine::Prepared(Box::new(crate::program::adopt(&registration)))
        }))
    }

    /// The adopted program's id and form held (`Programs::finish`); a
    /// compile that refused refuses here.
    fn finish_program(&mut self, prepared: engine::Prepared) -> EngineResult<ProgramId> {
        let Ok(prepared) = prepared.0.downcast::<crate::program::Prepared>() else {
            return self.refused(Err(Error::Program(format!(
                "a prepared program the {} engine did not prepare",
                B::KIND
            ))));
        };
        let finished = self.programs.finish(*prepared).map_err(refusal);
        self.count_registered();
        self.refused(finished)
    }

    fn close_program(&mut self, program: ProgramId) -> EngineResult<()> {
        self.programs.close_program(program).map_err(refusal)
    }

    fn register_channel(
        &mut self,
        registration: &ChannelRegistration,
    ) -> EngineResult<RegisteredChannel> {
        let place = self.shell.as_ref().map(place);
        self.programs
            .register_channel(registration, place.as_ref())
            .map_err(refusal)
    }

    fn bind_instance(&mut self, binding: &InstanceBinding) -> EngineResult<BoundInstance> {
        let caps = self
            .caps
            .as_ref()
            .ok_or_else(|| Error::Program("bind_instance before load".to_string()))?;
        if !caps.admits(binding.geometry) {
            return Err(Error::Program(format!(
                "this load resolves {:?} on the device, so it binds at most {:?} and not {:?}",
                caps.ports, caps.geometry, binding.geometry
            )));
        }

        let layout = self
            .pair_layout(binding.program, &binding.extents)?
            .unwrap_or_default();
        let place = self.shell.as_ref().map(place);
        let bound = self
            .programs
            .bind(binding, place.as_ref(), &layout)
            .map_err(refusal);
        // What the bind built, counted where it lands, before its first
        // fire: the program's own registration's builds.
        self.count_registered();
        let bound = bound?;

        // A lane that only classes the kernels refuse serve is refused
        // here, where its first fire would be.
        let unserved = self.shell.as_ref().and_then(|shell| {
            let unservable = &shell.named().unservable;
            if unservable.is_empty() {
                return None;
            }
            crate::naming::refuses(unservable, &shell.classes(), &self.programs.reads(bound.id))
        });
        if let Some(why) = unserved {
            let _ = self.programs.close_instance(bound.id);
            return Err(Error::Program(why));
        }

        let seeds: Vec<(u32, Vec<u8>)> = binding
            .seeds
            .iter()
            .map(|seed| (seed.channel, seed.bytes.clone()))
            .collect();
        let package = self.programs.package(binding.program).cloned();
        let landed = match (package, self.shell.as_mut()) {
            (Some(package), Some(shell)) => adapter_of(shell, &package, bound.id, &seeds),
            _ => Ok(None),
        };
        match landed {
            Ok(Some(binding)) => {
                self.adapters.insert(bound.id, binding);
            }
            Ok(None) => {}
            Err(why) => {
                let _ = self.programs.close_instance(bound.id);
                return Err(why);
            }
        }
        Ok(bound)
    }

    fn close_instance(&mut self, id: InstanceId) -> EngineResult<u64> {
        if let (Some(held), Some(shell)) = (self.adapters.remove(&id), self.shell.as_mut()) {
            shell.release_adapter(&held);
        }
        // A guest's tables lie in its slots and its scratch in a buffer of
        // its own: nothing standing in the span.
        self.programs.close_instance(id).map_err(refusal)?;
        if let Some(shell) = self.shell.as_ref() {
            shell.beam_forget(id);
        }
        Ok(0)
    }

    fn register_adapter(&mut self, registration: &AdapterRegistration) -> EngineResult<()> {
        let planes: Vec<crate::weights::AdapterPlane<'_>> = registration
            .planes
            .iter()
            .map(|plane| crate::weights::AdapterPlane {
                bank: plane.bank.as_str(),
                bytes: &plane.bytes,
            })
            .collect();
        self.loaded_mut()?
            .register_adapter(registration.id, &planes)
            .map_err(fault)
    }

    fn close_channel(&mut self, id: ChannelId) -> EngineResult<()> {
        self.programs.close_channel(id).map_err(refusal)
    }

    fn publish_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
        cell: &[u8],
    ) -> EngineResult<bool> {
        self.programs
            .publish(instance, channel, cell)
            .map_err(refusal)
    }

    fn take_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
    ) -> EngineResult<Option<Vec<u8>>> {
        self.programs.take(instance, channel).map_err(refusal)
    }

    /// Beam `beam`'s page list of `instance`'s group as its last fire left
    /// it (`BEAM-PAGE-LISTS.md` §6.6): every fire in flight lands first,
    /// then the state's last list is read from the pool.
    fn beam_list(&mut self, instance: InstanceId, beam: u32) -> EngineResult<Vec<u32>> {
        let shell = self.shell.as_mut().ok_or_else(unloaded::<B>)?;
        shell.drain().map_err(fault)?;
        let resident = self.programs.beam_state(instance).ok_or_else(|| {
            Error::Program(format!("instance {instance} binds no beam group (parents)"))
        })?;
        resident.list(shell.pools(), beam).map_err(fault)
    }

    fn settles_asynchronously(&self) -> bool {
        B::settles_asynchronously(self)
    }

    fn measures_behind(&self) -> bool {
        self.shell
            .as_ref()
            .is_some_and(|shell| shell.measures_behind() || shell.rates_behind().is_some())
    }

    fn behind_need(&self) -> Option<engine::BehindNeed> {
        self.shell.as_ref().and_then(Shell::behind_need)
    }

    fn measure_behind(&mut self, behind: &engine::BehindServing<'_>) -> EngineResult<bool> {
        // A frame whose numbers are not yet taken holds its readout seats,
        // which a settle fire would write: on an idle turn they are taken
        // now (they are due anyway; the frame's settle reads them back from
        // here), so the seats are free for what runs behind serving.
        if self.pending.is_some() {
            self.take_pending_rows()?;
        }
        let Some(shell) = self.shell.as_mut() else {
            return Ok(false);
        };
        // The settle's ranges first (decode's rows before the large ones),
        // where the runtime took the slots its next fire asks for; else,
        // and after, each rate and curve served by a stale figure (its step
        // writes no pool slot: it runs on any idle turn).
        let need = shell.behind_need();
        let seats_short = match (need, behind.seats) {
            (Some(need), Some(seats)) => seats.len() < need.lanes as usize,
            _ => false,
        };
        // A point longer than one decode step runs only with no process
        // live, so a live request never waits behind it.
        let long_held = need.is_some_and(|need| need.long) && !behind.alone;
        if !shell.measures_behind() || seats_short || long_held {
            if shell.rates_behind().is_none() {
                if long_held {
                    self.behind_state(
                        "waiting: the settle's next point runs longer than one decode step, and \
                         runs only once no process is live",
                    );
                } else if seats_short {
                    self.behind_state(
                        "waiting: the pool holds no room for one settle point's slots beyond \
                         live work (the settle runs on a later idle turn; the fallback picks \
                         serve)",
                    );
                }
            }
            return self.rate_behind(behind);
        }
        self.behind_state("running: the dense settle");
        let Some(shell) = self.shell.as_mut() else {
            return Ok(false);
        };
        let fired = shell
            .measure_behind(behind.others, behind.arrived, behind.seats)
            .map_err(fault)?;
        // What the settle decided so far, kept in the user's profile for
        // the next load of the same curves, model, rungs and build.
        if let Some(settled) = shell.take_settled() {
            let rates = profile::fire::Rates {
                dense_settled: vec![settled],
                ..profile::fire::Rates::default()
            };
            if let Err(why) = B::save_rates(shell.device(), &self.boot, &rates) {
                tracing::warn!(%why, "the dense settle was not kept in the profile");
            }
        }
        Ok(fired)
    }

    fn on_complete(&mut self, sink: engine::CompletionSink) {
        self.sink = Some(sink);
    }

    fn settle_frame(&mut self, ticket: &mut FrameTicket) -> EngineResult<()> {
        let Some((id, _)) = self.pending.as_ref() else {
            return Err(Error::Invalid(format!(
                "frame {}'s numbers are gone: nothing is pending, so either it was \
                 never submitted to this engine or a later frame has already taken \
                 its readout seats",
                ticket.id
            )));
        };
        if *id != ticket.id {
            return Err(Error::Invalid(format!(
                "frame {}'s numbers are gone: frame {id} has been submitted since, and \
                 a step's rows live in the readout seat its arm owns. Ask for a \
                 frame's readouts before submitting the next one",
                ticket.id
            )));
        }
        let (_, mut pending) = self.pending.take().expect("checked just above");
        let mut refused = None;
        match self.shell.as_mut() {
            None => {
                refused = Some(unloaded::<B>());
            }
            Some(shell) => {
                for step in &mut pending {
                    if step.rows.is_some() {
                        continue;
                    }
                    match shell.rows_of(&step.landed) {
                        Ok(rows) => step.rows = Some(rows),
                        Err(why) => {
                            refused = Some(fault(why));
                            break;
                        }
                    }
                }
            }
        }

        self.pending = Some((ticket.id, pending));
        // What the settle built, on the fire's path.
        self.count_built();
        if let Some(error) = refused {
            return Err(error);
        }
        let (_, pending) = self.pending.as_ref().expect("just put back");
        for (receipt, step) in ticket.steps.iter_mut().zip(pending) {
            receipt.readouts = readouts_of(step);
        }
        Ok(())
    }

    fn expect_fire(&mut self, submission: &Step) {
        let _ = submission;
        if let Some(shell) = self.shell.as_mut() {
            let _ = shell.reap();
        }
    }

    fn copy_kv(&mut self, copy: &KvCopy) -> EngineResult<()> {
        copy.validate()?;
        let device = self.caps.as_ref().map(|caps| caps.device.domain);
        if device.is_none_or(|device| copy.src != device || copy.dst != device) {
            return Err(Error::Unsupported {
                verb: kv_copy_direction(copy.src, copy.dst),
                engine: B::KIND,
            });
        }
        let page_size = self.loaded_mut()?.paging().page_size;
        let moves = copy.runs(page_size)?;
        self.loaded_mut()?.copy_kv(&moves).map_err(fault)
    }
}

/// A ring end that two of `step`'s fires move, if any: the contract has a
/// step carry at most one fire moving a given end of a ring, since the cell
/// each move takes is the one the ring's cursor stands at as the step is built.
fn second_mover<B: Host>(programs: &B::Programs, step: &Step) -> Option<(ChannelId, bool)> {
    let mut moved = std::collections::BTreeSet::new();
    let mut instances = std::collections::BTreeSet::new();
    for attachment in step.attachments.iter().filter(|a| !a.adapter_only) {
        if !instances.insert(attachment.instance) {
            continue;
        }
        for end in programs.moved_ends(attachment.instance) {
            if !moved.insert(end) {
                return Some(end);
            }
        }
    }
    None
}

impl<B: Host> Gpu<B>
where
    Gpu<B>: Send,
{
    /// [`Engine::submit`] as fired, before its refusal is counted.
    fn submit_frame(&mut self, frame: &FrameSubmission) -> EngineResult<FrameTicket> {
        frame.validate_for(
            self.caps
                .as_ref()
                .map_or(engine::fire::Serves::NONE, engine::fire::Serves::of),
        )?;
        // The runtime's stamp of each step's stated shape, cross-checked
        // against this load's stated set in debug builds (CONTRACT-BC 24).
        if let Some(caps) = self.caps.as_ref() {
            frame.debug_check_shapes(&caps.steps);
        }
        if let Some(shell) = self.shell.as_ref() {
            let facts = &shell.trace().facts;
            for step in &frame.steps {
                step.validate_captures(|word| facts.holds(word, poem_ir::Builtin::Scores))?;
            }
        }
        let id = FrameId(self.next_frame);
        self.next_frame = self.next_frame.wrapping_add(1);

        self.pending = None;
        // The frame before can no longer be asked for its rows.
        if let Some(shell) = self.shell.as_mut() {
            shell.open_frame();
            // Under a stated submit bound, a frame that follows an idle
            // device re-times its clock before its first step is admitted
            // (`crate::submit`); nothing runs where no bound is stated.
            shell.retime(id).map_err(fault)?;
        }

        let mut steps = Vec::with_capacity(frame.steps.len());
        tracing::debug!(
            engine = B::KIND,
            frame = id.0,
            steps = frame.steps.len(),
            "submit"
        );
        let mut pending = Vec::with_capacity(frame.steps.len());
        let mut refused = None;
        for (index, step) in frame.steps.iter().enumerate() {
            if let Some(next) = frame.steps.get(index + 1) {
                self.expect_fire(next);
            }

            let at = engine::StepDone {
                frame: id,
                step: index as u32,
            };
            // Held to the platform's bound on one submission, where it
            // states one (`crate::submit`): a step past it whole walks a
            // layer at a time; past it even so, its prompt lane is cut by
            // rows, its leading pieces fired first, each alone and held
            // to the bound in turn; a step no row of which fits, or that
            // cannot be cut, has its lanes refused by name. Nothing is
            // estimated where no bound is stated.
            let mut rest: Option<Step> = None;
            let mut piece_failed = None;
            loop {
                let current = rest.as_ref().unwrap_or(step);
                // The first piece at the ratio read now: the pieces landed
                // before move it (`crate::submit::Measured`).
                let Some((piece, after)) = self.next_piece(current) else {
                    break;
                };
                if self.admit_step(&piece, at).is_some() {
                    // Chosen to fit: refused only past what was weighed;
                    // the step is refused as a whole, by name.
                    piece_failed = self.admit_step(current, at);
                    break;
                }
                // A piece tells no completion: the step's last piece does.
                let fired = self.fire_step(&piece, at, true, None);
                if let Some(shell) = self.shell.as_mut() {
                    shell.split_next(None);
                    shell.estimate_next(None);
                }
                if let Err(error) = fired {
                    piece_failed = Some(error);
                    break;
                }
                rest = Some(after);
            }
            if let Some(error) = piece_failed {
                if error.stops_frame_at(index) {
                    refused = Some(error);
                    break;
                }
                self.pending = None;
                self.count_built();
                return Err(error);
            }
            let step = rest.as_ref().unwrap_or(step);
            if let Some(refused_lanes) = self.admit_step(step, at) {
                if refused_lanes.stops_frame_at(index) {
                    refused = Some(refused_lanes);
                    break;
                }
                self.pending = None;
                self.count_built();
                return Err(refused_lanes);
            }
            let fired = B::step(self, step, at);
            if let Some(shell) = self.shell.as_mut() {
                shell.split_next(None);
                shell.estimate_next(None);
            }
            match fired {
                Ok((ticket, step_pending)) => {
                    steps.push(ticket);
                    pending.push(step_pending);
                }
                // A step's lanes refused past the first step: the frame is
                // the steps before it (`engine::Error::Lanes`).
                Err(error) if error.stops_frame_at(index) => {
                    refused = Some(error);
                    break;
                }
                Err(error) => {
                    self.pending = None;
                    self.count_built();
                    return Err(error);
                }
            }
        }
        self.count_built();
        let mut ticket = FrameTicket::of(id, steps, refused);
        self.pending = Some((id, pending));

        if !self.settles_asynchronously() {
            self.settle_frame(&mut ticket)?;
            self.tell_wrote();
        }
        Ok(ticket)
    }
}

impl<B: Host> Gpu<B> {
    /// Fires `submission`'s forward; with `defers` its completion is left
    /// for the guest pass behind it to report. With `meet`, its landing
    /// waits there for the guest passes the host runs behind it.
    pub fn fire_step(
        &mut self,
        submission: &Step,
        at: engine::StepDone,
        defers: bool,
        meet: Option<crate::settle::Meet>,
    ) -> EngineResult<(FireTicket, PendingStep)> {
        debug_assert_eq!(
            second_mover::<B>(&self.programs, submission),
            None,
            "a step carries at most one fire moving a given end of a ring"
        );
        // The lanes whose guest passes refused ahead of the forward: they
        // fault, and the forward runs for every lane all the same.
        let mut faulted = engine::StepOutcome::committed();
        B::run_prologues(self, submission, &mut faulted)?;
        let mut staged: Vec<Vec<u8>> = Vec::new();
        if !submission.media.is_empty() {
            let Some(element) = self.loaded()?.patch_element() else {
                return Err(fault(crate::error::Fault::from(poem_exec::Error::Fire(
                    poem_exec::fire::Fault::Towerless {
                        lane: submission.media[0].lane,
                    },
                ))));
            };
            staged.reserve(submission.media.len());
            for row in &submission.media {
                staged.push(patch_bytes(&row.patches, element).map_err(|why| {
                    Error::Unsupported {
                        verb: why,
                        engine: B::KIND,
                    }
                })?);
            }
        }
        let voxel_element = self.loaded()?.voxel_element();
        if !submission.voxels.is_empty() && voxel_element.is_none() {
            return Err(Error::Invalid(
                "this step submits VAE clips and the loaded plan states no voxel row".into(),
            ));
        }
        let voxel_payloads = submission
            .voxels
            .iter()
            .map(|row| {
                voxel_bytes(&row.payload, voxel_element.unwrap_or(poem_ir::Dtype::Bf16)).map_err(
                    |why| Error::Unsupported {
                        verb: why,
                        engine: B::KIND,
                    },
                )
            })
            .collect::<EngineResult<Vec<_>>>()?;
        let clips: Vec<Clips<'_>> = submission
            .voxels
            .iter()
            .zip(&voxel_payloads)
            .map(|(row, payload)| Clips {
                lane: row.lane,
                boxes: &row.clips,
                payload,
            })
            .collect();

        let media: Vec<Media<'_>> = submission
            .media
            .iter()
            .zip(&staged)
            .map(|(row, patches)| Media {
                lane: row.lane,
                rows: &row.rows,
                patches,
                routes: &row.routes,
                positions: &row.positions,
                grids: &row.grids,
                token_positions: &row.token_positions,
            })
            .collect();

        let id = FireId(self.next_fire);
        self.next_fire = self.next_fire.wrapping_add(1);

        let lanes = submission.lanes.len();
        let mut envelope = Envelope {
            tokens: vec![None; lanes],
            masks: vec![None; lanes],
            feeds: vec![None; lanes],
        };
        B::envelope(self, submission, &mut envelope, &mut faulted)?;
        let device = self.device_lanes(submission, &mut envelope, &mut faulted);
        if let Some(vocab) = self.vocab {
            out_of_range_tokens(submission, &mut envelope, vocab, &mut faulted);
        }
        let Envelope {
            tokens: resolved,
            masks: resolved_masks,
            feeds,
        } = envelope;

        // A lane whose fold length is the device's takes the count its
        // program's `fold` port holds at the fire; the lane stays stated as
        // it came, so its record keeps its own frozen position and the
        // lane reports what the fold froze (`store::frozen`). A lane with
        // no recurrent buffer that asks a report takes it too, and reports
        // it as read.
        let mut folds: Vec<crate::store::frozen::DeviceFold> = submission
            .lanes
            .iter()
            .map(|lane| crate::store::frozen::DeviceFold {
                count: None,
                reports: lane.reports,
            })
            .collect();
        for attachment in &submission.attachments {
            let lane = attachment.lane as usize;
            let stated = &submission.lanes[lane];
            let window_counted = matches!(
                stated.window,
                Some(engine::fire::WindowFold {
                    fold: engine::fire::FoldLen::Device,
                    ..
                })
            );
            if crate::store::frozen::reads_count(&stated.rs, stated.reports) || window_counted {
                match self.programs.envelope_fold(attachment.instance) {
                    Ok(count) => folds[lane].count = count,
                    Err(refused) => faulted.fault_lanes(
                        attachment_lanes(attachment),
                        Some(attachment.instance),
                        &refused.to_string(),
                    ),
                }
            }
        }

        // Each beam group's state, layout and parents
        // (`crate::beam`): its lanes reorder their page lists on the device
        // ahead of the forward. An instance that refuses faults its own
        // lanes, which then read their staged tables (their set's pages).
        let mut beams: Vec<crate::beam::Group<B>> = Vec::new();
        for attachment in program_attachments(submission) {
            let lanes = attachment_lanes(attachment);
            let group = self
                .loaded()
                .map_err(|error| Refusal::Program(error.to_string()))
                .and_then(|shell| self.programs.beam_group(attachment.instance, shell.pools()));
            let refused = match group {
                Ok(None) => None,
                Ok(Some((state, layout, parents))) => {
                    let stated: Vec<&engine::fire::Lane> = lanes
                        .clone()
                        .filter_map(|at| submission.lanes.get(at as usize))
                        .collect();
                    match stated.first() {
                        Some(first)
                            if stated.iter().all(|lane| {
                                lane.reserve == first.reserve && lane.kv.folded == first.kv.folded
                            }) && !first.reserve.is_empty() =>
                        {
                            beams.push(crate::beam::Group {
                                instance: attachment.instance,
                                state,
                                layout,
                                parents,
                                lanes: lanes.clone().map(|at| at as usize).collect(),
                                reserve: first.reserve.clone(),
                                folded: first.kv.folded,
                                appended: self
                                    .loaded()
                                    .map_or((u32::MAX, Vec::new()), |shell| shell.beam_appended(attachment.instance)),
                            });
                            None
                        }
                        _ => Some(format!(
                            "instance {}'s beams state no working set's reserve, or not one \
                             set's: a beam group's lanes name one set and its whole kv table \
                             (BEAM-PAGE-LISTS.md §2, §6.5)",
                            attachment.instance
                        )),
                    }
                }
                Err(refused) => Some(refused.to_string()),
            };
            if let Some(why) = refused {
                faulted.fault_lanes(lanes, Some(attachment.instance), &why);
            }
        }

        let mut input_cells: Vec<Vec<(engine::fire::InputFeed, crate::serve::InputCell<B>)>> =
            vec![Vec::new(); submission.lanes.len()];
        for (at, lane) in submission.lanes.iter().enumerate() {
            if lane.inputs.is_empty() {
                continue;
            }
            // A lane feeding inputs with no attachment is a submission that
            // names no instance to read them through: the fire as stated is
            // malformed, not one lane's fault, and it is refused whole.
            let attachment = submission
                .attachments
                .iter()
                .find(|attachment| attachment_lanes(attachment).contains(&(at as u32)))
                .ok_or_else(|| {
                    Error::Invalid(format!(
                        "lane {at} feeds guest inputs and has no attachment; an input is read \
                         through the instance the lane is attached to"
                    ))
                })?;
            for feed in &lane.inputs {
                // A cell its instance refuses is that attachment's fault: its
                // lanes fault, and the seat takes zeros nothing reads.
                let cell = match self.programs.input_cell(attachment.instance, feed.channel) {
                    Ok(cell) => cell,
                    Err(refused) => {
                        faulted.fault_lanes(
                            attachment_lanes(attachment),
                            Some(attachment.instance),
                            &refused.to_string(),
                        );
                        crate::serve::InputCell::Refused
                    }
                };
                input_cells[at].push((*feed, cell));
            }
        }

        // A deferred step's passes go behind its forward from the engine's
        // thread; none runs on its landing.
        let guests = if defers {
            None
        } else {
            B::guests(self, submission)?
        };

        let mut reads: Vec<Vec<String>> = vec![Vec::new(); submission.lanes.len()];
        // Each lane's score room: its first attachment's pair's, where that
        // program reads scores.
        let mut scores: Vec<Option<crate::serve::DeviceCell<B>>> =
            (0..submission.lanes.len()).map(|_| None).collect();
        for attachment in program_attachments(submission) {
            if let Some(room) = self.programs.scores(attachment.instance) {
                let end = attachment.lane + attachment.lanes.max(1);
                for at in attachment.lane..end {
                    if let Some(lane) = scores.get_mut(at as usize)
                        && lane.is_none()
                    {
                        *lane = Some(room.clone());
                    }
                }
            }
            // A root read at lane `k` may span rows: row `r` is lane
            // `k + r`'s readout, so every lane of the attachment from `k`
            // on reads the output out.
            let end = attachment.lane + attachment.lanes.max(1);
            for (name, k) in self.programs.reads(attachment.instance) {
                for at in attachment.lane + k..end {
                    if let Some(lane) = reads.get_mut(at as usize)
                        && !lane.contains(&name)
                    {
                        lane.push(name.clone());
                    }
                }
            }
        }

        // A frame settled inside `submit` answers there: its steps'
        // completions tell the sink nothing.
        let sink = self
            .sink
            .clone()
            .filter(|_| B::settles_asynchronously(self));

        let mut lane_adapters: Vec<Option<u32>> = vec![None; submission.lanes.len()];
        if !self.adapters.is_empty() {
            for attachment in &submission.attachments {
                let Some(bound) = self.adapters.get(&attachment.instance) else {
                    continue;
                };
                if let Some(slot) = lane_adapters.get_mut(attachment.lane as usize) {
                    *slot = Some(bound.slot);
                }
            }
        }
        let vocab = self.vocab;
        let shell = self.loaded_mut()?;

        let mut words: Vec<u64> = submission.lanes.iter().map(|lane| lane.word).collect();
        for (at, slot) in lane_adapters.iter().enumerate() {
            if slot.is_none() {
                continue;
            }
            words[at] = shell.adapted_word(words[at]).ok_or_else(|| {
                Error::Invalid(format!(
                    "lane {at} is attached to an instance that bound an adapter, and \
                     this load's model text has no corrected class for its fact word \
                     {:#x}: the text declares no `linear.lora_correct` arm, or the arm's \
                     window is not decided by one fact. A lane that asked for a \
                     correction and got the base model is the one wrong answer this \
                     axis refuses to give",
                    words[at]
                ))
            })?;
        }
        // A lane names its kv pages and its record in its slot tables
        // (CONTRACT-BC 4): this engine states one shared pool, so the
        // runtime states them.
        let mut pages_of: Vec<Option<Vec<u32>>> = Vec::with_capacity(submission.lanes.len());
        let mut record_of: Vec<Option<Vec<u32>>> = Vec::with_capacity(submission.lanes.len());
        let mut run_of: Vec<Option<Vec<u32>>> = Vec::with_capacity(submission.lanes.len());
        let mut forks: Vec<(u32, u32)> = Vec::new();
        for (at, lane) in submission.lanes.iter().enumerate() {
            // A lane whose table is a working set's whole, for its program
            // to place rows through, is placed by its device-geometry ports
            // ([`crate::program::device_lanes`], each row in
            // `KvDelta::writable`), or refused already and sunk; any other
            // would land rows in pages not its own.
            let placed = device[at].as_ref().is_some_and(|placed| placed.have.is_some());
            let refused = faulted.faulted.iter().any(|fault| fault.lane as usize == at);
            if lane.list == engine::fire::ListKind::Translation && !placed && !refused {
                return Err(Error::lane(
                    u32::try_from(at).unwrap_or(u32::MAX),
                    "its kv table is a translation its program places rows through, and no \
                     device-geometry port of an attached instance places its rows",
                ));
            }
            pages_of.push(
                lane.tables
                    .iter()
                    .find(|(shape, _)| *shape == crate::store::layout::LIST)
                    .map(|(_, refs)| refs.clone()),
            );
            // A load that holds state is told every lane's record: its
            // pieces, or none (the null record, ref 0 a piece).
            let fixed = match &lane.record {
                engine::fire::LaneRecord::Pieces(pieces) => pieces
                    .iter()
                    .find(|(shape, _)| *shape == crate::store::layout::FIXED)
                    .map(|(_, pieces)| pieces.clone()),
                engine::fire::LaneRecord::Null => {
                    Some(vec![[0, 0]; shell.pools().pieces() as usize])
                }
                engine::fire::LaneRecord::Unstated if shell.pools().has_state() => {
                    return Err(Error::Invalid(format!(
                        "lane in slot {} states no record, and this load holds recurrent \
                         state: the runtime states every lane's record, or that it has none",
                        lane.slot
                    )));
                }
                engine::fire::LaneRecord::Unstated => None,
            };
            let fixed = fixed.as_ref();
            run_of.push(fixed.and_then(|pieces| {
                pieces
                    .iter()
                    .any(|&[run, cow]| run != cow)
                    .then(|| pieces.iter().map(|&[run, _]| run).collect())
            }));
            record_of.push(fixed.map(|pieces| {
                // `cow != run`: the record is copied before the fire, once
                // for every lane of a group naming the same pair.
                for &[run, cow] in pieces {
                    if run != cow && !forks.contains(&(run, cow)) {
                        forks.push((run, cow));
                    }
                }
                pieces.iter().map(|&[_, cow]| cow).collect()
            }));
        }
        shell.copy_state(&forks).map_err(fault)?;
        // Tail copies (an rs-list or kv slot a fork shares) land before any
        // lane of the fire reads or writes (CONTRACT-BC 20).
        let slot_copies: Vec<_> = submission
            .lanes
            .iter()
            .flat_map(|lane| lane.slot_copies.iter().cloned())
            .collect();
        shell.copy_slot_pairs(&slot_copies).map_err(fault)?;
        let mut attached = vec![false; submission.lanes.len()];
        for attachment in program_attachments(submission) {
            let lanes = attachment.lane..attachment.lane + attachment.lanes.max(1);
            for at in lanes {
                if let Some(lane) = attached.get_mut(at as usize) {
                    *lane = true;
                }
            }
        }
        // A refused lane still fires (token 0) to keep the composition; one
        // whose program places its rows has no run of its own the host may
        // append to, so its rows land in the null chunk, never its table.
        let sunk: Vec<bool> = submission
            .lanes
            .iter()
            .enumerate()
            .map(|(at, lane)| {
                lane.list == engine::fire::ListKind::Translation
                    && faulted.faulted.iter().any(|fault| fault.lane as usize == at)
            })
            .collect();
        let seated: Vec<Seated<'_, B>> = submission
            .lanes
            .iter()
            .enumerate()
            .map(|(at, lane)| Seated {
                lane: Lane {
                    slot: lane.slot,

                    word: words[at],
                    tokens: resolved[at].as_deref().unwrap_or(&lane.tokens),
                },
                pages: pages_of[at].as_deref().unwrap_or(&[]),
                held: match device[at].as_ref().and_then(|placed| placed.have) {
                    _ if sunk[at] => Some(0),
                    Some(have) => Some(have),
                    None => pages_of[at].is_some().then_some(lane.kv.held),
                },
                sunk: sunk[at],
                folded: lane.kv.folded,
                kv_less: lane.kv_less,

                mask: if sunk[at] {
                    None
                } else {
                    resolved_masks[at].as_ref().or(lane.mask.as_ref())
                },

                adapter: lane_adapters[at].or(lane.adapter),
                positions: device[at]
                    .as_ref()
                    .and_then(|placed| placed.positions.as_deref())
                    .unwrap_or(&lane.positions),
                slots: device[at].as_ref().and_then(|placed| placed.slots.as_deref()),

                readout: match &lane.readout {
                    Readout::Rows(rows) => Some(rows.as_slice()),
                    Readout::Last | Readout::None => None,
                },
                // A program attached to the lane reads its rows on the
                // device and hands back what it makes through its
                // channels: the host reads back the lanes no program reads.
                hosts: !matches!(lane.readout, Readout::None) && !attached[at],
                reads: &reads[at],
                scores: scores[at].as_ref(),

                rs: &lane.rs,
                rs_reset: lane.rs_reset,
                rs_snapshots: &lane.rs_snapshots,
                refs: true,
                windows: &lane.tables,
                window: lane.window,
                record: record_of[at].as_deref(),
                record_run: run_of[at].as_deref(),
                feed: feeds[at].as_ref(),
                group: lane.group,
                inputs: &input_cells[at],
            })
            .collect();

        let wrote;
        let mut faulted = faulted.faulted;
        let mut fed = None;
        let landed = {
            use engine::frame::Shell as FrameShell;
            let done = sink.filter(|_| !defers).map(|sink| Done {
                at,
                sink,
                wrote: Vec::new(),
                lanes: u32::try_from(submission.lanes.len()).unwrap_or(u32::MAX),
                faulted: std::mem::take(&mut faulted),
                meet,
                fed: None,
            });
            let mut prepared = FrameShell::prepare(
                shell,
                StepView {
                    lanes: &seated,
                    attachments: &[],
                    media: &media,
                    clips: &clips,
                    folds: &folds,
                    done,
                    guests,
                    beams: &beams,
                },
                None,
            )
            .map_err(fault)?;
            wrote = std::mem::take(&mut prepared.wrote);
            // A lane fed its token on the device is checked once the
            // forward landed, from what the forward embedded.
            if !prepared.beam_faults.is_empty() || (vocab.is_some() && !prepared.fed_lanes.is_empty())
            {
                let vocab = vocab.unwrap_or(u32::MAX);
                let lanes = prepared
                    .fed_lanes
                    .iter()
                    .map(|&(source, row)| {
                        let attached = submission
                            .attachments
                            .iter()
                            .find(|attachment| attachment_lanes(attachment).contains(&source));
                        crate::settle::FedLane {
                            row,
                            lanes: attached.map_or(source..source + 1, attachment_lanes),
                            instance: attached.map(|attachment| attachment.instance),
                        }
                    })
                    .collect();
                let beams = prepared
                    .beam_faults
                    .iter()
                    .map(|&(word, lane, instance)| crate::settle::BeamWord {
                        word,
                        lane: u32::try_from(lane).unwrap_or(u32::MAX),
                        instance,
                    })
                    .collect();
                let commons = prepared.beam_commons.clone();
                let check = crate::settle::FedCheck::with_beams(
                    <B as crate::serve::Serve>::token_row_reader(shell, prepared.arm),
                    lanes,
                    vocab,
                    shell.store_word_reader(prepared.arm),
                    beams,
                    commons,
                );
                prepared.fed_check = Some(check.clone());
                match prepared.done.as_mut() {
                    Some(done) => done.fed = Some(check),
                    None => fed = Some(check),
                }
            }
            let enqueued = FrameShell::enqueue(shell, prepared).map_err(fault)?;
            FrameShell::settle(shell, enqueued).map_err(fault)?
        };

        Ok((
            FireTicket {
                id,
                readouts: Vec::new(),
            },
            PendingStep {
                readout: submission
                    .lanes
                    .iter()
                    .map(|lane| lane.readout.clone())
                    .collect(),
                landed,
                rows: None,
                wrote,
                faulted,
                fed,
            },
        ))
    }
}

impl<B: Host> Gpu<B> {
    /// Each lane's geometry as its instance's descriptor ports resolve it
    /// ([`crate::program::device_lanes`]): device geometry's tokens,
    /// positions, extent, slots and mask, and a decode envelope's positions
    /// where its program computes them. The tokens and mask land in
    /// `envelope`; an attachment whose ports refuse faults its own lanes.
    fn device_lanes(
        &self,
        step: &Step,
        envelope: &mut Envelope<B>,
        faulted: &mut engine::StepOutcome,
    ) -> Vec<Option<crate::program::DeviceLane>> {
        let mut out: Vec<Option<crate::program::DeviceLane>> = vec![None; step.lanes.len()];
        for attachment in program_attachments(step) {
            let first = (attachment.lane as usize).min(step.lanes.len());
            let end = attachment_lanes(attachment).end as usize;
            let stated = &step.lanes[first..end.min(step.lanes.len())];
            let resolved = self
                .programs
                .descriptor_ports(attachment.instance)
                .and_then(|ports| match ports {
                    Some(mut ports) => crate::program::device_lanes(
                        &mut ports,
                        stated,
                        first,
                        attachment.instance,
                    )
                    .map(Some),
                    None => Ok(None),
                });
            match resolved {
                Ok(None) => {}
                Ok(Some(lanes)) => {
                    for (at, placed) in lanes.into_iter().enumerate() {
                        let at = first + at;
                        if out[at].is_some() {
                            envelope.refused(
                                step,
                                attachment,
                                &format!(
                                    "lane {at} is claimed by two attached instances; a lane's \
                                     descriptor ports have one author"
                                ),
                                faulted,
                            );
                            break;
                        }
                        if let Some(tokens) = &placed.tokens {
                            envelope.tokens[at] = Some(tokens.clone());
                            envelope.feeds[at] = None;
                        }
                        if placed.mask.is_some() {
                            envelope.masks[at].clone_from(&placed.mask);
                        }
                        out[at] = Some(placed);
                    }
                }
                Err(refused) => {
                    envelope.refused(step, attachment, &refused.to_string(), faulted);
                }
            }
        }
        out
    }
}

fn readouts_of(step: &PendingStep) -> Vec<LaneReadout> {
    let rows: &[Vec<f32>] = step.rows.as_deref().unwrap_or(&[]);
    step.readout
        .iter()
        .enumerate()
        .map(|(lane, policy)| {
            let count = match policy {
                Readout::None => return LaneReadout::default(),
                Readout::Last => 1,
                Readout::Rows(list) => list.len().max(1),
            };
            let values = rows.get(lane).cloned().unwrap_or_default();
            LaneReadout {
                rows: u32::try_from(count).unwrap_or(u32::MAX),
                width: u32::try_from(values.len() / count).unwrap_or(u32::MAX),
                values,
                ..LaneReadout::default()
            }
        })
        .collect()
}

/// The attachments whose program runs on this fire.
pub fn program_attachments(step: &Step) -> impl Iterator<Item = &engine::fire::Attachment> {
    step.attachments
        .iter()
        .filter(|attachment| !attachment.adapter_only)
}

#[cfg(test)]
mod tests {
    use engine::fire::Attachment;
    use engine::program::InstanceId;

    use super::each_pass;
    use crate::program::Refusal;

    /// A step's prologues run each on its own: the one whose instance was
    /// poisoned in an earlier step refuses and faults its own lanes, by
    /// name; the other three run, and fault nothing.
    #[test]
    fn a_refused_prologue_faults_its_own_lanes_alone() {
        let attachments: Vec<Attachment> = (0..4u32)
            .map(|at| Attachment {
                lane: at * 2,
                lanes: 2,
                instance: InstanceId(100 + u64::from(at)),
                adapter_only: false,
                scratch: ::engine::ScratchPlace::None,
            })
            .collect();
        let mut ran = Vec::new();
        let mut faulted = engine::StepOutcome::committed();
        each_pass(&attachments, "prologue", &mut faulted, |attachment| {
            ran.push(attachment.instance);
            if attachment.instance == InstanceId(102) {
                Err(Refusal::Program("instance 102 is poisoned".into()))
            } else {
                Ok(())
            }
        });
        assert_eq!(ran.len(), 4, "every prologue ran");
        let lanes: Vec<(u32, Option<InstanceId>)> = faulted
            .faulted
            .iter()
            .map(|fault| (fault.lane, fault.instance))
            .collect();
        assert_eq!(
            lanes,
            [(4, Some(InstanceId(102))), (5, Some(InstanceId(102)))],
            "the refusing instance's two lanes, and no other"
        );
        assert!(
            faulted
                .faulted
                .iter()
                .all(|f| f.reason.contains("poisoned"))
        );
        assert!(faulted.wrote.is_empty());
    }
}
