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

/// A step in flight: its submission and the token feed it waits behind.
pub struct Flight {
    pending: Pending,

    _fed: Option<Pending>,
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

    const MAX_BUFFER_SOURCE: &'static str = "the least of VkPhysicalDeviceLimits::\
        maxStorageBufferRange, maxMemoryAllocationSize (Vulkan 1.1) and maxBufferSize \
        (Vulkan 1.3)";

    const RECORDS: bool = true;

    /// A pass keeps the cells only the device reads in `Channel` slots
    /// (`guest::session::Kept`).
    const CHANNEL_CELLS: bool = true;

    /// A step's guest passes are submitted behind its forward on the one
    /// queue before the step returns, so before any later fire, and every
    /// frame waits for all work submitted before it: they read the rows
    /// where the forward left them in its span.
    const READS_SPAN: bool = true;

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
        // A device whose profile lacks the roofline's constants measures
        // them here, before ready, and keeps them in the user's profile,
        // each submission within the platform's stated bound on one.
        let tuning = crate::rates::at_load(
            &device,
            tuning,
            boot.profiles.as_deref(),
            crate::rates::STATED_SUBMIT_BOUND,
        )?;
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

    const BINDS_AT_LOAD: bool = false;

    fn bind_frame(_device: &Context) -> Result<Frame> {
        Err(engine_gpu::error::Fault::Unbound {
            what: "a bind walk on Vulkan, which keeps no descriptor set per set of views".into(),
        })
    }

    fn bound(_pipelines: &Pipelines) -> u64 {
        0
    }

    fn walk_frame(device: &Context, keep: bool) -> Result<Frame> {
        if keep {
            device.frame_kept()
        } else {
            device.frame()
        }
    }

    fn shelf(
        _device: &Context,
        _pipelines: &Pipelines,
        _readout: &[Buffer],
        weights_stream: bool,
        _guest: engine_gpu::serve::GuestRoom,
    ) -> Result<Shelf> {
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
            let seq = shell.airborne.enter();
            let counts = shell.airborne.clone();
            let done = prepared.done.take();
            let fed = prepared.fed_check.clone();
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
                        // What the fire embedded is read back before its
                        // count is left: its arm may be taken again then.
                        if let Some(fed) = fed.as_ref() {
                            fed.landed(refused.is_none());
                        }
                        if let Some(done) = done.as_ref() {
                            done.tell(refused, "vulkan replay");
                        }
                        counts.leave();
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
            shell.advance(prepared, &pending);
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
            fed_check,
            ..
        } = launch;
        shell.shelf.generation = shell.shelf.generation.wrapping_add(1);
        let on_done: Box<dyn Fn(Option<String>) + Send + 'static> =
            Box::new(move |refused: Option<String>| {
                // What the fire embedded is read back before its count is
                // left: its arm may be taken again then.
                if let Some(fed) = fed_check.as_ref() {
                    fed.landed(refused.is_none());
                }
                if let Some(done) = done.as_ref() {
                    done.tell(refused, "vulkan command buffer");
                }
                counts.leave();
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
        Flight { pending, _fed: fed }
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
        let waited = flight.api.pending.wait();
        // The arm's token seat is read back at the landing: it is written
        // again only once that read is done.
        if let Some(fed) = &flight.fed_check {
            fed.wait();
        }
        if let Err(fault) = waited {
            shell.arms.give(flight.arm);
            return Err(match fault {
                Fault::Device { call, why } => Fault::Device {
                    call,
                    why: format!("step {} of this load: {why}", flight.seq),
                },
                other => other,
            });
        }
        let end = flight
            .rows
            .iter()
            .filter(|span| span.hosted)
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
    // The rung names where its carve and rooms lie in the span: a fire of
    // another rung binds other bytes under the same handles.
    key.shown(&p.rung);
    // Not how many pages the fire's tables name: a view binds its buffer
    // to the end, and no dispatch is sized by the pages: a decode's splits
    // are the shape's, each row's walker splitting its own keys, and a
    // prefill's dispatches bound no work (`Tactic::prefill_work` is 0 here).
    // So a recording replays as the fire's keys grow.
    let mut caches = p.caches.clone();
    for pool in &mut caches.0 {
        if let engine_gpu::run::CachePool::Kv(kv) = pool {
            kv.page_indices.rows = 0;
        }
    }
    key.hashed(&caches);
    let mut bindings = p.bindings.clone();
    for geometry in &mut bindings.geometry {
        if let Some(indices) = geometry.indices.as_mut() {
            indices.rows = 0;
        }
    }
    key.shown(&bindings);
    key.shown(&p.windows);
    // Not the fire's demand: the pool only checks it against what is
    // committed (a commit forgets every recording), so a recording reads
    // it nowhere, and the kv refs it counts climb as lanes grow.

    for lane in p.lanes {
        key.shown(&lane.readout);
        key.shown(&lane.hosts);
        key.shown(&lane.reads);
        // A score room is its pair's place in the pool, which an unpark
        // or an evacuation moves: a recording binds the one it was taken at.
        key.hashed(&lane.scores.map(|room| {
            (engine_gpu::backend::Buffer::slab_id(&room.buffer), room.at)
        }));
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
        // A spanned read's rows lie in the fire's span: read only while no
        // later fire was submitted to carve over them.
        let span = self.span_buffer().ok()?;
        let last = landed.seq + 1 == self.airborne.entered();
        let reads: Vec<&ReadSpan> = (rows.iter())
            .filter(|read| read.lane == lane && read.rows > 0)
            .filter(|read| read.seated || read.spanned)
            .collect();
        if !last && reads.iter().any(|read| read.spanned) {
            return None;
        }
        Some(
            reads
                .into_iter()
                .map(|read| {
                    (
                        if read.spanned { span } else { seat },
                        read.at,
                        read.values(),
                        read.dtype,
                        self.output_name(read.plane),
                    )
                })
                .collect(),
        )
    }

    fn forget_seat(&mut self, landed: &Landed) {
        self.shelf.landed_seats.remove(&landed.seq);
    }
}
