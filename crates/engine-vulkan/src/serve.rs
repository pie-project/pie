//! The shared shell ([`engine_gpu::serve`]) bound to Vulkan: a step's frame
//! is kept and replayed where it repeats, and lands when its fence signals.

use std::collections::BTreeMap;

use engine_gpu::serve::{Landed, Launch, Plan, Prepared, ReadSpan, Serve, Shell, rows_from};

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

    /// The host room every guest pass stages its host-held roots through
    /// and reads its status and puts back into
    /// ([`engine_gpu::guest::ring`]): laid here, before the pool is sized.
    #[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
    pub(crate) guest: GuestRing,
}

/// The guest plane's ring of host room, a berth a pass in flight.
pub type GuestRing = std::sync::Arc<engine_gpu::guest::ring::Ring<Buffer>>;

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

    /// The probe recorded at load into a spanned frame kept to replay.
    type Probed = crate::device::Kept;

    type Stamps = crate::device::Stamps;

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
        let (chip, report) =
            crate::chip::resolve_stored(&measured, &device.measured_on(), boot.profiles.as_deref());
        crate::chip::SHELF.tell_untuned(&measured.name, &report);
        let tuning = crate::tuning::Tuning::new(chip, device.limits());
        // A device whose profile lacks the roofline's constants measures
        // them here, before ready, and keeps them in the user's profile,
        // each submission within the platform's stated bound on one.
        let tuning = crate::rates::at_load(
            &device,
            tuning,
            boot.profiles.as_deref(),
            <Self as Serve>::submit_bound(boot),
            boot.measure,
        )?;
        Ok((device, tuning))
    }

    fn rate_shapes(
        device: &Context,
        boot: &crate::api::DeviceBoot,
        tuning: crate::tuning::Tuning,
        shapes: &[(u32, u32)],
        model_bytes: u64,
        top_rows: u32,
    ) -> Result<crate::tuning::Tuning> {
        crate::rates::shapes_at_load(
            device,
            tuning,
            boot.profiles.as_deref(),
            crate::rates::stated_submit_bound(),
            shapes,
            model_bytes,
            top_rows,
            boot.measure,
        )
    }

    fn time_kernels(
        device: &Context,
        tuning: &crate::tuning::Tuning,
        tier: kernels_slang::Tier,
        max_submit: Option<std::time::Duration>,
        launch: f64,
        kernels: &[kernels_slang::linear::gemm::Tactic],
        pieces: &[u32],
        plane: (u32, u32),
        rows: u32,
    ) -> Result<Vec<Vec<f64>>> {
        let timer = |frame: crate::device::Frame| {
            let started = std::time::Instant::now();
            let span = frame.commit_spanned()?;
            Ok((
                span.filter(|&ns| ns > 0),
                started.elapsed().as_nanos() as u64,
            ))
        };
        engine_gpu::rates::time_kernels::<Api, _>(
            device, tuning, tier, &timer, max_submit, launch, kernels, pieces, plane, rows,
        )
    }

    fn time_for_bound(
        device: &Context,
        tuning: &crate::tuning::Tuning,
        tier: kernels_slang::Tier,
        max_submit: std::time::Duration,
        at: &engine_gpu::rates::ForBound,
    ) -> Result<engine_gpu::rates::BoundTimes> {
        let timer = |frame: crate::device::Frame| {
            let started = std::time::Instant::now();
            let span = frame.commit_spanned()?;
            Ok((
                span.filter(|&ns| ns > 0),
                started.elapsed().as_nanos() as u64,
            ))
        };
        engine_gpu::rates::for_bound::<Api, _>(device, tuning, tier, &timer, max_submit, at)
    }

    fn save_rates(
        device: &Context,
        boot: &crate::api::DeviceBoot,
        rates: &profile::fire::Rates,
    ) -> Result<()> {
        crate::rates::save_rates(device, boot.profiles.as_deref(), rates)
    }

    fn measure_rates(
        device: &Context,
        tuning: &crate::tuning::Tuning,
        max_submit: Option<std::time::Duration>,
        stated: &profile::fire::Rates,
        yield_to: &dyn Fn() -> bool,
    ) -> Result<engine::rates::Measured> {
        crate::rates::measure_behind(device, tuning, max_submit, stated, yield_to)
    }

    fn measure_curves(
        device: &Context,
        tuning: &crate::tuning::Tuning,
        max_submit: Option<std::time::Duration>,
        stated: &profile::fire::Rates,
        wanted: &[(kernels_slang::linear::gemm::Tactic, u32, u32)],
        top: u32,
        yield_to: &dyn Fn() -> bool,
    ) -> Result<Vec<profile::fire::KernelCurve>> {
        crate::rates::curves_behind(device, tuning, max_submit, stated, wanted, top, yield_to)
    }

    fn chunk_bytes(boot: &crate::api::DeviceBoot) -> Option<u64> {
        boot.chunk_bytes
    }

    fn committed_covers() -> &'static [&'static str] {
        crate::chip::COVERS
    }

    fn measuring(boot: &crate::api::DeviceBoot) -> engine::rates::Measuring {
        boot.measure
    }

    fn submit_bound(boot: &crate::api::DeviceBoot) -> Option<std::time::Duration> {
        #[cfg(feature = "test-limits")]
        if let Some(bound) = boot.submit_bound {
            return Some(bound);
        }
        let _ = boot;
        crate::rates::stated_submit_bound()
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

    fn walk_frame(
        device: &Context,
        keep: bool,
        timed: Option<&crate::device::Stamps>,
    ) -> Result<Frame> {
        // A timed step is spanned into its arm's timestamps: their reset and
        // writes recorded with its work, so a kept one (kept under its arm,
        // `record_key`) writes them anew on each replay, and its device time
        // is read at its harvest (`Pending::span_ns`), before the arm's next
        // step goes out.
        match (timed, keep) {
            (Some(stamps), keep) => device.spanned_into(stamps, keep),
            (None, true) => device.frame_kept(),
            (None, false) => device.frame(),
        }
    }

    /// One pool an arm: a step's time is read at its harvest, before its
    /// arm is free again, so no later frame finds it still being read.
    fn lay_stamps(device: &Context, _pieces: u32) -> Result<crate::device::Stamps> {
        device.stamps()
    }

    fn fire_timing(device: &Context) -> Result<engine_gpu::submit::FireTiming> {
        if device.writes_timestamps() {
            Ok(engine_gpu::submit::FireTiming::Timestamps)
        } else {
            Err(Fault::Unbound {
                what: format!(
                    "a serving fire's device time on {}, whose queue family writes no timestamps \
                     (timestampValidBits 0, or timestampPeriod 0): under the platform's stated \
                     bound on one GPU submission each fire's time moves the scale the next \
                     fires are estimated at",
                    engine_gpu::backend::Context::name(device)
                ),
            })
        }
    }

    fn record_probe(
        device: &Context,
        record: &dyn Fn(&Frame) -> Result<()>,
    ) -> Result<crate::device::Kept> {
        let frame = device.spanned_kept()?;
        record(&frame)?;
        frame.into_kept_spanned()
    }

    fn replay_probe(
        _device: &Context,
        probed: &crate::device::Kept,
        _record: &dyn Fn(&Frame) -> Result<()>,
    ) -> Result<Option<u64>> {
        probed.replay_spanned()
    }

    fn shelf(
        device: &Context,
        _pipelines: &Pipelines,
        _readout: &[Buffer],
        weights_stream: bool,
        guest: engine_gpu::serve::GuestRoom,
    ) -> Result<Shelf> {
        // A guest bind writes its tables through the device's staging slab,
        // laid here where the load has not yet written through it.
        device.lay_staging()?;
        // Every guest pass stages and reads back through a berth of one
        // ring of host room, laid for the frames in flight
        // ([`engine_gpu::guest::ring::ring_bytes`]).
        let ring = engine_gpu::guest::ring::ring_bytes(guest.chunk_bytes, guest.seats);
        let guest = engine_gpu::guest::ring::Ring::new(
            Buffer::with(
                device,
                ring,
                engine_gpu::Memory::Host,
                engine_gpu::books::owners::GUEST,
            )?,
            ring,
        );
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
            guest,
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

    /// The descriptor pools of the frames serving holds at once, laid now
    /// and none made after: each arm's forward, the feed and the guest
    /// passes behind it, and the one a settle or a walk holds beside them.
    fn serving(shell: &mut Shell<Api>) -> Result<()> {
        shell
            .device()
            .seal_descriptor_pools(shell.readout.len() + 3)
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
        // A step past the platform's bound on one submission walks a layer
        // at a time by the shell's own admission (`engine_gpu::submit`);
        // whole, a 512x512 VAE decode outlasts one Adreno submission and
        // loses the device, and nothing estimates it.
        let split = prepared.bindings.voxels.is_some();
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
        // Under a stated submit bound, the step's device time from its
        // timestamps, read now its fence was waited (`Shell::timed`).
        if flight.estimate.is_some() || shell.spans_wanted() {
            let time = engine_gpu::submit::DeviceTime::default();
            match flight.api.pending.span_ns()? {
                Some(ns) => time.set(ns),
                None => time.unread(),
            }
            shell.timed(flight.estimate, time);
        }
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
        key.hashed(
            &lane
                .scores
                .map(|room| (engine_gpu::backend::Buffer::slab_id(&room.buffer), room.at)),
        );
    }

    key.finish()
}

/// An output a lane read out: its seat, offset, element count, dtype and name.
pub type ReadSeat<'a> = engine_gpu::serve::ReadSeat<'a, Api>;

/// What a step read out, for readers on the device behind it.
pub trait Seats {
    /// What step `landed` read out for its lane `lane`.
    fn readout_seats(&self, landed: &Landed, lane: u32) -> Option<Vec<ReadSeat<'_>>>;

    fn forget_seat(&mut self, landed: &Landed);
}

impl Seats for Shell<Api> {
    fn readout_seats(&self, landed: &Landed, lane: u32) -> Option<Vec<ReadSeat<'_>>> {
        match self.shelf.landed_seats.get(&landed.seq) {
            Some(seat) if seat.generation == self.shelf.generation => {
                self.seats_of(seat.arm, &seat.rows, landed, lane)
            }
            Some(_) => None,
            None => self.flight_seats(landed, lane),
        }
    }

    fn forget_seat(&mut self, landed: &Landed) {
        self.shelf.landed_seats.remove(&landed.seq);
    }
}
