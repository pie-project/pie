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

    /// Each arm's readout tail: a replayed body's reads, copied behind it.
    tails: crate::device::Tails,

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

    /// A pass keeps the cells only the device reads in `Channel` slots
    /// (`guest::session::Kept`).
    const CHANNEL_CELLS: bool = true;

    const TARGET: kernels_slang::Target = kernels_slang::Target::Spirv;

    fn table_align(device: &Context) -> u64 {
        device.table_align()
    }

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

    fn loaded(device: &Context) -> Result<()> {
        device.persist_pipelines()
    }

    fn encodes(shelf: &Shelf) -> Option<&'static str> {
        shelf.recorder.refused()
    }

    fn made(device: &Context) -> Option<&engine_gpu::made::Made> {
        Some(device.made())
    }

    fn compiled(pipelines: &Pipelines) -> u64 {
        pipelines.compiled()
    }

    fn built_since(pipelines: &Pipelines, from: u64) -> Vec<&'static str> {
        pipelines.built_since(from)
    }

    fn hold(device: &Context, unit: &'static str, key: &'static str) -> Result<()> {
        Pipelines::hold(device, unit, key)
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

    fn shelf(device: &Context, readout: &[Buffer], weights_stream: bool) -> Result<Shelf> {
        // Every fire records whatever is resident: a row table's misses are
        // staged by the host before the fire, a routed expert's between the
        // segments of the recording its stage cut ([`Kept::replay_cut`]).
        Ok(Shelf {
            recorder: if weights_stream {
                crate::record::Recorder::refusing("the weights plane streams")
            } else {
                crate::record::Recorder::new()
            },
            tails: device.tails(readout.len())?,
            landed_seats: BTreeMap::new(),
            generation: 0,
        })
    }

    /// Vulkan binds a storage buffer at any offset its descriptors state.
    fn staging_align(_device: &Context) -> Option<u64> {
        None
    }

    /// A Vulkan frame's scalars ride push constants and it stages nothing;
    /// what it takes of its own is a descriptor set a dispatch. A frame
    /// encoded each fire takes its sets from what this lays: for each set
    /// layout the load's pipelines share, the most dispatches one fire
    /// makes for each arm a fire can be in flight on.
    fn lay_frames(
        device: &Context,
        pipelines: &Pipelines,
        facts: engine_gpu::serve::FrameFacts,
    ) -> Result<()> {
        let started = std::time::Instant::now();
        let count = (facts.dispatches as usize).saturating_mul(facts.arms.max(1) as usize);
        device.lay_sets(pipelines, count)?;
        tracing::info!(
            sets = count,
            layouts = pipelines.set_layouts().len(),
            ms = started.elapsed().as_millis() as u64,
            "a frame encoded each fire takes its descriptor sets from these, laid for each set \
             layout: the most dispatches a fire makes, for each arm"
        );
        Ok(())
    }

    /// Every recording's descriptors of a pool segment are written to the
    /// segment's buffer, or its stand-in where it is not backed: a
    /// recording stands across the pool's growth and shrink.
    fn rebind(shell: &mut Shell<Api>) -> Result<()> {
        let kept: Vec<&crate::device::Kept> = shell.shelf.recorder.kept().collect();
        shell
            .device
            .rebind_segments(&shell.placed_segments(), &kept)
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

    /// A decode fire of the lattice the load armed replays the recording of
    /// its composition at its arm, its reads copied behind it. One the load
    /// armed none of (or armed over other bindings) is a bug: it is
    /// recorded at this fire, said by name and counted
    /// ([`engine::live::Tally::RecordedAfterReady`]). Every other fire (prefill, mixed and masked rows, unbounded) is
    /// encoded afresh from the pipelines, layouts and descriptor sets the
    /// load laid, and kept by nothing.
    fn plan(shell: &mut Shell<Api>, prepared: &mut Prepared<'_, Api>) -> Result<Plan<Api>> {
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
        let point = shell
            .lattice_point(prepared)
            .filter(|_| shell.shelf.recorder.records() && !split);
        let Some(point) = point else {
            return Ok(Plan::Walk { keep: None, split });
        };
        let key = composition_key(prepared);
        let binds = binding_key(shell, prepared);
        let launches = match shell.shelf.recorder.get(key) {
            Some(recording) if recording.binds == binds => recording.launches,
            held => {
                tracing::error!(
                    arm = prepared.arm,
                    class = point.class,
                    lanes = point.lanes,
                    why = if held.is_some() {
                        "the load armed it over other bindings"
                    } else {
                        "the load armed no recording of it"
                    },
                    "a decode fire records its body while the load serves"
                );
                return Ok(Plan::Walk {
                    keep: Some(key),
                    split,
                });
            }
        };
        let rows = prepared.readout_layout().to_vec();
        let pixels = shell.pixel_spans(prepared)?;
        shell.shelf.generation = shell.shelf.generation.wrapping_add(1);
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
        let tail = staged.and_then(|()| {
            shell.shelf.tails.record(
                prepared.arm,
                shell.readout_source()?,
                &shell.readout[prepared.arm],
                &regions(prepared),
            )
        });
        let pending = tail.and_then(|tail| {
            recording.kept.replay(
                Some(&tail),
                Some(Box::new(move |refused: Option<String>| {
                    counts.leave();
                    if let Some(done) = done.as_ref() {
                        let outcome = match &refused {
                            None => done.settled(),
                            Some(why) => engine::StepOutcome::Faulted(format!(
                                "vulkan replay for frame {} step {}: {why}",
                                done.at.frame, done.at.step
                            )),
                        };
                        (done.sink)(done.at, outcome);
                    }
                })),
            )
        });
        let pending = match pending {
            Ok(pending) => pending,
            Err(fault) => {
                shell.airborne.abandon();
                shell.handles.rewind();
                return Err(fault);
            }
        };
        shell.advance(prepared, &pending);
        shell.handles.rewind();
        Ok(Plan::Replayed {
            pending,
            seq,
            rows,
            launches,
            pixels,
        })
    }

    /// Records a decode fire of the lattice the load arms, submitting
    /// nothing: its body, without its reads, kept under its composition
    /// and arm with the bindings it was walked over.
    fn arm(shell: &mut Shell<Api>, prepared: &mut Prepared<'_, Api>) -> Result<bool> {
        if !shell.shelf.recorder.records() {
            return Ok(false);
        }
        // The descriptors a recording binds of a segment are noted as the
        // pool now places its segments.
        shell.device.rebind_segments(&shell.placed_segments(), &[])?;
        // A fire staging group masks or landing snapshots binds a table of
        // its own; one over a streamed row table stages its misses.
        if !prepared.bindings.group_masks.is_empty()
            || prepared.bindings.rs_snaps.is_some()
            || (prepared.bindings.row_ring.is_none() && !shell.row_caches.whole())
        {
            return Err(Fault::Unbound {
                what: format!(
                    "a recording of the decode fire of {} lane(s) at arm {}: it binds a table \
                     a recording cannot hold",
                    prepared.lanes.len(),
                    prepared.arm
                ),
            });
        }
        let key = composition_key(prepared);
        let binds = binding_key(shell, prepared);
        let (frame, launches, cuts) = shell.walk_kept(prepared)?;
        let kept = frame.into_kept()?;
        // Each segment a flush cut must have an expert stage's host step
        // to replay after it; a flush of anything else would not repeat.
        if kept.cuts() != cuts.len() {
            return Err(Fault::Unbound {
                what: format!(
                    "a recording of the decode fire of {} lane(s) at arm {}: it cut {} \
                     segment(s), {} of them an expert stage's",
                    prepared.lanes.len(),
                    prepared.arm,
                    kept.cuts(),
                    cuts.len()
                ),
            });
        }
        shell
            .shelf
            .recorder
            .insert(key, crate::record::Recording::new(kept, launches, cuts, binds));
        Ok(true)
    }

    fn submit(
        shell: &mut Shell<Api>,
        frame: Frame,
        prepared: &mut Prepared<'_, Api>,
        launch: Launch<'_, Api>,
    ) -> Result<Pending> {
        let Launch {
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
                        None => done.settled(),
                        Some(why) => engine::StepOutcome::Faulted(format!(
                            "vulkan command buffer for frame {} step {}: {why}",
                            done.at.frame, done.at.step
                        )),
                    };
                    (done.sink)(done.at, outcome);
                }
            });
        match keep {
            // A decode fire the load did not arm (a bug, counted).
            Some(key) => frame.into_kept().and_then(|kept| {
                let tail = shell.shelf.tails.record(
                    prepared.arm,
                    shell.readout_source()?,
                    &shell.readout[prepared.arm],
                    &regions(prepared),
                )?;
                let pending = kept.replay(Some(&tail), Some(on_done))?;
                if kept.cuts() == cuts.len() {
                    let binds = binding_key(shell, prepared);
                    shell
                        .shelf
                        .recorder
                        .insert(key, crate::record::Recording::new(kept, launches, cuts, binds));
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

/// A decode fire's composition and arm: what its recording is kept under.
/// A recording binds its tables at their capacity (the kv pages' indices,
/// the bindings and windows over them), and copies its reads behind it, so
/// the fire's lanes' pages and reads are no part of it.
fn composition_key(p: &Prepared<'_, Api>) -> crate::record::Key {
    let mut key = crate::record::KeyBuilder::new();
    key.bytes(&p.descriptor.pack());
    // The tables run to a slot per traced value; hashing them field by
    // field rather than through their debug text keeps a step's key off
    // the CPU's critical path (1.6 ms of a token on a phone's core).
    key.hashed(&p.slots);
    // The rung names where its carve and rooms lie in the span: a fire of
    // another rung binds other bytes under the same handles.
    key.shown(&p.rung);
    key.shown(&p.arm);
    key.finish()
}

/// What a fire binds beside its composition: the cache table, the
/// bindings and the windows. A decode fire of the lattice binds them as
/// the load's recording of its composition did, which a replay checks.
fn binding_key(shell: &Shell<Api>, p: &Prepared<'_, Api>) -> crate::record::Key {
    let _ = shell;
    let mut key = crate::record::KeyBuilder::new();
    key.hashed(&p.caches);
    key.shown(&p.bindings);
    key.shown(&p.windows);
    key.finish()
}

/// A fire's readout copies as the device copies regions.
fn regions(p: &Prepared<'_, Api>) -> Vec<(u64, u64, u64)> {
    p.readout_copies
        .iter()
        .map(|copy| (copy.from, copy.to, copy.bytes))
        .collect()
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
