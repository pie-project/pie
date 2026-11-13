//! The shared shell ([`engine_gpu::serve`]) bound to Vulkan: a step's frame
//! is kept and replayed where it repeats, and lands when its fence signals.

use std::collections::BTreeMap;

use engine_gpu::serve::{Landed, Launch, Plan, Prepared, ReadSpan, Serve, Shell, rows_from};
use poem_ir::Dtype;

use crate::device::{Api, Buffer, Context, Frame, Pending, Pipelines};
use engine_gpu::error::{Fault, Result};

/// What the shell keeps of Vulkan's own: the frames kept to replay, and
/// what each harvested step read out, for readers behind it on the queue.
pub struct Shelf {
    recorder: crate::record::Recorder,

    landed_seats: BTreeMap<u64, LandedSeat>,

    /// Bumped by every step: a landed seat a later step may have rewritten
    /// is read no more.
    generation: u64,
}

struct LandedSeat {
    arm: usize,

    rows: Vec<ReadSpan>,

    generation: u64,
}

/// A step in flight: its submission, the token feed it waits behind, and
/// whether the host reads its rows back.
pub struct Flight {
    pending: Pending,

    _fed: Option<Pending>,

    read: bool,
}

impl Serve for Api {
    type DeviceBoot = crate::api::DeviceBoot;

    type Handed = std::convert::Infallible;

    type Pipelines = Pipelines;

    type Sink<'f> = crate::encode::Sink<'f>;

    type Guests = std::convert::Infallible;

    type Shelf = Shelf;

    type Flight = Flight;

    type Keep = crate::record::Key;

    const FUSIONS: &'static [poem_compiler::fuse::Rule] = &crate::FUSIONS;

    const RECORDS: bool = true;

    fn open(
        boot: &crate::api::DeviceBoot,
        allowance: engine::Allowance,
        handed: Option<std::convert::Infallible>,
    ) -> Result<(Context, crate::tuning::Tuning)> {
        if let Some(handed) = handed {
            match handed {}
        }
        let device = Context::bind(boot, allowance)?;
        let measured = device.chip();
        let (chip, report) = crate::chip::resolve(&measured, boot.profiles.as_deref());
        crate::chip::SHELF.tell_untuned(&measured.name, &report);
        let tuning = crate::tuning::Tuning::new(chip, device.limits());
        Ok((device, tuning))
    }

    fn chunk_bytes(boot: &crate::api::DeviceBoot) -> Option<u64> {
        boot.chunk_bytes
    }

    fn tier(_device: &Context) -> kernels_slang::Tier {
        // Vulkan's tactics read their own tuning, not the tier.
        kernels_slang::Tier::Subgroup
    }

    fn compiled(pipelines: &Pipelines) -> u64 {
        pipelines.compiled()
    }

    fn measures_outside(device: &Context) -> bool {
        device.enabled().memory_budget
    }

    fn builds(device: &Context, key: &str, tier: &str) -> bool {
        // A tactic fires at the baseline, the subgroup tier or, where the
        // device has them, cooperative matrices.
        let runs = match tier {
            "baseline" | "subgroup" => true,
            "coopmat" => device.enabled().coopmat,
            _ => false,
        };
        runs && kernels_slang::spirv(key).is_some()
    }

    fn build(
        device: &Context,
        pipelines: &Pipelines,
        unit: &'static str,
        key: &'static str,
    ) -> Result<()> {
        pipelines.get(device, unit, key).map(drop)
    }

    fn sink<'f>(
        device: &'f Context,
        frame: &'f Frame,
        pipelines: &'f Pipelines,
        handles: &'f engine_gpu::Handles<Api>,
    ) -> crate::encode::Sink<'f> {
        crate::encode::Sink::new(device, frame, pipelines, handles)
    }

    fn walk_frame(device: &Context, keep: bool) -> Result<Frame> {
        if keep {
            device.frame_kept()
        } else {
            device.frame()
        }
    }

    fn shelf(_device: &Context, _readout: &[Buffer], weights_stream: bool) -> Result<Shelf> {
        // Every fire records whatever is resident: a row table's misses are
        // staged by the host before the fire, a routed expert's between the
        // segments of the recording its stage cut ([`Kept::replay_cut`]).
        Ok(Shelf {
            recorder: if weights_stream {
                crate::record::Recorder::refusing("the weights plane streams")
            } else {
                crate::record::Recorder::new()
            },
            landed_seats: BTreeMap::new(),
            generation: 0,
        })
    }

    fn forget(shelf: &mut Shelf) {
        // Recordings stand across a residency: a slot moved is an entry
        // rewritten in a table they bind, not a binding. A routed expert's
        // misses are staged between a recording's segments, one cut a layer
        // with an expert in no slot, so a recording stands only while those
        // layers do (and the stage they bind with them), and only while the
        // segments it binds stand as they did.
        shelf.recorder.forget();
    }

    fn grown(_shell: &mut Shell<Api>, _arm: usize, _bytes: u64) -> Result<()> {
        Ok(())
    }

    fn plan(shell: &mut Shell<Api>, prepared: &mut Prepared<'_, Api>) -> Result<Plan<Api>> {
        // The custom-mask buffer grows: a recording would keep the one it outgrew.
        // A fire landing snapshots binds a table of its own: not recorded.
        let key = (shell.shelf.recorder.records()
            && prepared.bindings.group_masks.is_empty()
            && prepared.bindings.rs_snaps.is_none()
            && (prepared.bindings.row_ring.is_some() || shell.row_caches.whole()))
        .then(|| record_key(shell, prepared));

        let replayable = key
            .and_then(|key| shell.shelf.recorder.get(key))
            .map(|replay| (replay.layout.clone(), replay.launches));
        if let Some((rows, launches)) = replayable {
            let key = key.expect("a recording was found under a key");
            let pixels = shell.pixel_spans(prepared)?;
            shell.shelf.generation = shell.shelf.generation.wrapping_add(1);
            shell.advance(prepared);
            let seq = shell.airborne.enter();
            let counts = shell.airborne.clone();
            let done = prepared.done.take();
            let recording = shell
                .shelf
                .recorder
                .get(key)
                .expect("the recording was there a line ago");
            // The segments an expert stage cut, each landed before its stage.
            let staged = if recording.cuts.is_empty() {
                Ok(())
            } else {
                match shell.weights.host() {
                    None => Err(Fault::Unbound {
                        what: "the host copy of the weight cache's banks".to_string(),
                    }),
                    Some(host) => {
                        let planes = shell.weights.cached_planes(|at| {
                            shell.row_caches.streams(at) || shell.expert_caches.streams(at)
                        });
                        recording.cuts.iter().enumerate().try_for_each(|(at, cut)| {
                            recording.kept.replay_cut(at)?;
                            shell.expert_caches.stage_cut(host, &planes, cut)
                        })
                    }
                }
            };
            let pending = staged.and_then(|()| {
                recording
                    .kept
                    .replay(Some(Box::new(move |refused: Option<String>| {
                        counts.leave();
                        if let Some(done) = done.as_ref() {
                            let outcome = match &refused {
                                None => engine::StepOutcome::Committed,
                                Some(why) => engine::StepOutcome::Faulted(format!(
                                    "vulkan replay for frame {} step {}: {why}",
                                    done.at.frame, done.at.step
                                )),
                            };
                            (done.sink)(done.at, outcome);
                        }
                    })))
            });
            let pending = match pending {
                Ok(pending) => pending,
                Err(fault) => {
                    shell.airborne.abandon();
                    shell.handles.rewind();
                    return Err(fault);
                }
            };
            shell.handles.rewind();
            return Ok(Plan::Replayed {
                pending,
                seq,
                rows,
                launches,
                pixels,
            });
        }

        // A split step is not recorded: a recording replays as one submit.
        let split_rows = shell
            .tuning
            .profile()
            .get::<crate::chip::SubmitSplitRows>()
            .0;
        // Whole, a 512x512 VAE decode outlasts one Adreno submission and loses the device.
        let split = (split_rows > 0
            && prepared.seats.iter().map(|seat| seat.rows).sum::<u32>() >= split_rows)
            || prepared.bindings.voxels.is_some();
        let keep = key.filter(|_| !split && shell.shelf.recorder.has_room());
        Ok(Plan::Walk { keep, split })
    }

    fn submit(
        shell: &mut Shell<Api>,
        frame: Frame,
        _prepared: &mut Prepared<'_, Api>,
        launch: Launch<'_, Api>,
    ) -> Result<Pending> {
        let Launch {
            layout,
            launches,
            cuts,
            keep,
            counts,
            done,
            ..
        } = launch;
        shell.shelf.generation = shell.shelf.generation.wrapping_add(1);
        let on_done: Box<dyn Fn(Option<String>) + Send + 'static> =
            Box::new(move |refused: Option<String>| {
                counts.leave();
                if let Some(done) = done.as_ref() {
                    let outcome = match &refused {
                        None => engine::StepOutcome::Committed,
                        Some(why) => engine::StepOutcome::Faulted(format!(
                            "vulkan command buffer for frame {} step {}: {why}",
                            done.at.frame, done.at.step
                        )),
                    };
                    (done.sink)(done.at, outcome);
                }
            });
        match keep {
            Some(key) => frame.into_kept().and_then(|kept| {
                let pending = kept.replay(Some(on_done))?;
                // Each segment a flush cut must have an expert stage's host
                // step to replay after it; a flush of anything else would
                // not repeat, so such a fire is not kept.
                if kept.cuts() == cuts.len() {
                    shell.shelf.recorder.insert(
                        key,
                        crate::record::Recording::new(kept, layout.to_vec(), launches, cuts),
                    );
                }
                Ok(pending)
            }),
            None => frame.commit_async(Some(on_done as engine_gpu::backend::OnDone)),
        }
    }

    fn fly(pending: Pending, fed: Option<Pending>) -> Flight {
        Flight {
            pending,
            _fed: fed,
            read: true,
        }
    }

    fn reap(shell: &mut Shell<Api>) -> Result<()> {
        while shell
            .inflight
            .front()
            .is_some_and(|flight| flight.api.pending.landed())
        {
            Api::harvest_one(shell)?;
        }
        Ok(())
    }

    fn harvest_one(shell: &mut Shell<Api>) -> Result<()> {
        let Some(flight) = shell.inflight.pop_front() else {
            return Ok(());
        };
        if let Err(fault) = flight.api.pending.wait() {
            shell.arms.give(flight.arm);
            return Err(match fault {
                Fault::Device { call, why } => Fault::Device {
                    call,
                    why: format!("step {} of this load: {why}", flight.seq),
                },
                other => other,
            });
        }
        if !flight.api.read {
            shell.arms.give(flight.arm);
            return Ok(());
        }
        let end = flight
            .rows
            .iter()
            .filter(|span| span.host)
            .map(|span| span.at + span.bytes())
            .max()
            .unwrap_or(0);
        let mut raw = vec![0u8; end as usize];
        shell.readout[flight.arm].read(0, &mut raw)?;
        let rows = rows_from(&raw, &flight.rows);

        shell.shelf.landed_seats.insert(
            flight.seq,
            LandedSeat {
                arm: flight.arm,
                rows: flight.rows,
                generation: shell.shelf.generation,
            },
        );
        shell.land(flight.seq, flight.arm, rows);
        Ok(())
    }
}

fn record_key(shell: &Shell<Api>, p: &Prepared<'_, Api>) -> crate::record::Key {
    let mut key = crate::record::KeyBuilder::new();
    key.bytes(&p.descriptor.pack());
    key.shown(&p.arm);
    key.shown(&shell.readout[p.arm].bytes());
    // The tables run to a slot per traced value; hashing them field by
    // field rather than through their debug text keeps a step's key off
    // the CPU's critical path (1.6 ms of a token on a phone's core).
    key.hashed(&p.slots);
    key.hashed(&p.caches);
    key.shown(&p.bindings);
    key.shown(&p.windows);
    // Not the fire's demand: the pool only checks it against what is
    // committed (a commit forgets every recording), so a recording reads
    // it nowhere, and the kv refs it counts climb as lanes grow.

    for lane in p.lanes {
        key.shown(&lane.readout);
        key.shown(&lane.reads);
    }

    key.finish()
}

/// An output a lane read out: its seat, offset, element count, dtype and name.
pub type ReadSeat<'a> = (&'a Buffer, u64, u32, Dtype, &'a str);

/// What a step read out, for readers on the device behind it.
pub trait Seats {
    /// What step `landed` read out for its lane `lane`.
    fn readout_seats(&self, landed: &Landed, lane: u32) -> Option<Vec<ReadSeat<'_>>>;

    fn forget_seat(&mut self, landed: &Landed);

    /// Lets the host leave a step's rows unread: the device read them.
    fn leave_unread(&mut self, landed: &Landed);
}

impl Seats for Shell<Api> {
    fn readout_seats(&self, landed: &Landed, lane: u32) -> Option<Vec<ReadSeat<'_>>> {
        let (arm, rows) = match self.shelf.landed_seats.get(&landed.seq) {
            Some(seat) if seat.generation == self.shelf.generation => (seat.arm, &seat.rows),
            Some(_) => return None,
            // Still in flight: its arm stays taken until the harvest, and a
            // reader submitted behind it on the queue sees its readout.
            None => {
                let flight = self.inflight.iter().find(|f| f.seq == landed.seq)?;
                (flight.arm, &flight.rows)
            }
        };
        let seat = self.readout.get(arm)?;
        Some(
            rows.iter()
                .filter(|span| span.lane == lane && span.rows > 0)
                .map(|span| {
                    (
                        seat,
                        span.at,
                        span.values(),
                        span.dtype,
                        self.output_name(span.plane),
                    )
                })
                .collect(),
        )
    }

    fn forget_seat(&mut self, landed: &Landed) {
        self.shelf.landed_seats.remove(&landed.seq);
    }

    fn leave_unread(&mut self, landed: &Landed) {
        if let Some(flight) = self.inflight.iter_mut().find(|f| f.seq == landed.seq) {
            flight.api.read = false;
        }
    }
}
