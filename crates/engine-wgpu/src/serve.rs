//! The shared shell ([`engine_gpu::serve`]) bound to WebGPU: a step's rows
//! land through a readback mirror into a mailbox, never by a blocking wait
//! on the browser's thread.

use engine_gpu::serve::{Launch, Plan, Prepared, Serve, Shell};

use crate::device::{Api, Buffer, Context, Frame, Pending, Pipelines};
use engine_gpu::error::Result;

/// The token words a landing read back for its fed check.
pub(crate) type FedWords = std::sync::Arc<std::sync::Mutex<Vec<u8>>>;

/// What the shell keeps of WebGPU's own: each arm's readback mirror, the
/// mailbox landed steps post to, and the worker that lands them.
pub struct Shelf {
    mirror: Vec<Buffer>,

    /// Each arm's fed token rows, copied from its token seat by a fire whose
    /// lanes the device fed, read by its landing, and the words it read:
    /// a fed check reads them there, never behind the work queued after the
    /// fire (a device buffer's read waits for the whole queue).
    fed: Vec<(Buffer, FedWords)>,

    /// The guest plane's readback buffers, one a thread that runs guest
    /// passes, a chunk of the pool each, and its readout widen's table:
    /// reserved before the pool is sized, the plane takes them at its load.
    #[cfg(feature = "wgpu")]
    pub(crate) guest: (std::sync::Arc<crate::guest::outbox::Pool>, Buffer),

    mailbox: crate::landing::Mailbox,

    worker: crate::landing::Worker,
}

impl Serve for Api {
    type DeviceBoot = crate::api::DeviceBoot;

    type Handed = crate::device::Handed;

    type Pipelines = Pipelines;

    type Sink<'f> = crate::encode::Sink<'f>;

    type Guests = crate::landing::Guests;

    type Shelf = Shelf;

    type Flight = engine_gpu::submit::DeviceTime;

    type Keep = std::convert::Infallible;

    /// Eager by design: the probe is encoded at each fire.
    type Probed = crate::device::Stamps;

    type Stamps = crate::device::Stamps;

    const FUSIONS: &'static [poem_compiler::fuse::Rule] = &crate::FUSIONS;

    const MAX_BUFFER_SOURCE: &'static str = "the least of wgpu::Limits::max_buffer_size, \
        max_storage_buffer_binding_size and, on Vulkan, maxMemoryAllocationSize";

    const RECORDS: bool = false;

    /// A guest channel keeps a cell on the device between passes where the
    /// host does not read it (`engine_gpu::guest::defer`).
    const CHANNEL_CELLS: bool = true;

    /// A step's guest passes run once it landed (`landing`), and a later
    /// fire may be on the queue by then: they read the arm's seat.
    const READS_SPAN: bool = false;

    const TARGET: kernels_slang::Target = kernels_slang::Target::Wgsl;

    fn table_align(device: &Context) -> u64 {
        device.table_align()
    }

    fn open(
        boot: &crate::api::DeviceBoot,
        allowance: engine::Allowance,
        handed: Option<crate::device::Handed>,
    ) -> Result<(Context, crate::tuning::Tuning)> {
        let device = match handed {
            Some(handed) => Context::adopt(boot, allowance, handed)?,
            None => Context::bind(boot, allowance)?,
        };
        let measured = device.chip();
        let (chip, report) =
            crate::chip::resolve_stored(&measured, &device.measured_on(), boot.profiles.as_deref());
        crate::chip::SHELF.tell_untuned(&measured.name, &report);
        let tuning =
            crate::tuning::Tuning::new(chip).with_workgroup_bytes(device.workgroup_bytes());
        // A device whose profile lacks the roofline's constants measures
        // them here, before ready, and keeps them in the user's profile (a
        // browser masks its adapter, so no profile states its rates, and
        // keeps them for its load alone), each submission within the
        // platform's stated bound on one.
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

    fn tier(device: &Context) -> kernels_slang::Tier {
        if device.chip().subgroups_32() {
            kernels_slang::Tier::Subgroup
        } else {
            kernels_slang::Tier::Baseline
        }
    }

    /// The decodes the trace walks are held to the storage buffers the
    /// adapter binds a shader stage: the load is refused where a decode's
    /// own kernel does not bind, and splits where the split walk binds.
    #[cfg(feature = "wgpu")]
    fn admit(
        device: &Context,
        tuning: &mut crate::tuning::Tuning,
        tier: kernels_slang::Tier,
        trace: &poem_ir::Trace,
    ) -> Result<()> {
        tuning.hold_decodes(
            tier,
            &engine_gpu::scratch::decodes(trace),
            device.storage_buffers(),
        )
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

    fn walk_frame(
        device: &Context,
        _keep: bool,
        timed: Option<&crate::device::Stamps>,
    ) -> Result<Frame> {
        // A timed step writes its arm's timestamps; its device time is
        // filled once its work is done (`Context::timed_frame`) and read at
        // its landing (`collect`), before the arm's next step goes out.
        match timed {
            Some(stamps) => device.timed_frame(stamps),
            None => device.frame(),
        }
    }

    /// `pieces` sets an arm: a frame's readback is mapped from its
    /// work-done callback, which may run after its arm is free again, so
    /// the arm's next frames take a set not still being read.
    fn lay_stamps(device: &Context, pieces: u32) -> Result<crate::device::Stamps> {
        device.stamps(pieces)
    }

    fn fire_timing(device: &Context) -> Result<engine_gpu::submit::FireTiming> {
        Ok(if device.enabled().timestamps {
            engine_gpu::submit::FireTiming::Timestamps
        } else {
            engine_gpu::submit::FireTiming::SubmitToDone
        })
    }

    /// Eager by design, the probe records nothing: its timestamps alone
    /// are laid here, at load, and every probe writes them.
    fn record_probe(
        device: &Context,
        _record: &dyn Fn(&Frame) -> Result<()>,
    ) -> Result<crate::device::Stamps> {
        device.stamps(1)
    }

    fn replay_probe(
        device: &Context,
        probed: &crate::device::Stamps,
        record: &dyn Fn(&Frame) -> Result<()>,
    ) -> Result<Option<u64>> {
        // Into the timestamps laid at load: a probe lays nothing.
        let frame = device.spanned_into(probed)?;
        record(&frame)?;
        frame.commit_spanned()
    }

    const BINDS_AT_LOAD: bool = true;

    fn bind_frame(device: &Context) -> Result<Frame> {
        device.bind_frame()
    }

    fn bound(pipelines: &Pipelines) -> u64 {
        pipelines.bound()
    }

    fn shelf(
        device: &Context,
        pipelines: &Pipelines,
        readout: &[Buffer],
        _weights_stream: bool,
        guest: engine_gpu::serve::GuestRoom,
    ) -> Result<Shelf> {
        // A frame takes a scratch chunk from the ring: stocked for each frame
        // in flight (an arm a readout seat) and the one a settle or a walk
        // holds beside them, here at the load, before the pool is sized past
        // them. A shell fires frames with or without a guest program. Its
        // uniforms are the device's, made with it.
        #[cfg(feature = "wgpu")]
        device.core().stock_rings(readout.len() + 1);
        #[cfg(not(feature = "wgpu"))]
        let _ = (device, guest);
        let _ = pipelines;
        #[cfg(feature = "wgpu")]
        let threads = u32::try_from(crate::landing::guest_threads()).unwrap_or(u32::MAX);
        Ok(Shelf {
            #[cfg(feature = "wgpu")]
            // A readback buffer, a chunk each, for each thread that runs
            // guest passes and for each frame in flight whose passes land
            // behind it (each arm's seats) and the one a settle holds.
            guest: (
                crate::guest::outbox::Pool::new(
                    (0..threads as usize + guest.seats + 1)
                        .map(|_| {
                            Buffer::with(
                                device,
                                guest.chunk_bytes,
                                crate::device::Memory::Readback,
                                engine_gpu::books::owners::GUEST,
                            )
                        })
                        .collect::<Result<Vec<_>>>()?,
                ),
                Buffer::with(
                    device,
                    crate::guest::widen::Widen::table_bytes(
                        device,
                        threads,
                        u64::from(guest.max_lanes) * u64::from(guest.planes),
                    ),
                    crate::device::Memory::Device,
                    engine_gpu::books::owners::GUEST,
                )?,
            ),
            fed: readout
                .iter()
                .map(|_| {
                    Ok((
                        Buffer::with(
                            device,
                            u64::from(guest.max_tokens.max(1)) * 4,
                            crate::device::Memory::Readback,
                            engine_gpu::books::owners::READOUT,
                        )?,
                        FedWords::default(),
                    ))
                })
                .collect::<Result<Vec<_>>>()?,
            mirror: readout
                .iter()
                .map(|seat| {
                    Buffer::with(
                        device,
                        seat.bytes(),
                        crate::device::Memory::Readback,
                        engine_gpu::books::owners::READOUT,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            mailbox: std::sync::Arc::default(),
            worker: crate::landing::Worker::new(),
        })
    }

    fn forget(_shelf: &mut Shelf) {}

    fn token_row_reader(shell: &Shell<Api>, arm: usize) -> engine_gpu::settle::ReadRow {
        let words = shell
            .shelf
            .fed
            .get(arm)
            .map(|(_, words)| words.clone())
            .unwrap_or_default();
        Box::new(move |row| {
            let words = words.lock().expect("fed words lock poisoned");
            let at = row as usize * 4;
            words
                .get(at..at + 4)
                .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        })
    }

    fn grown(shell: &mut Shell<Api>, arm: usize, bytes: u64) -> Result<()> {
        shell.shelf.mirror[arm] = Buffer::with(
            &shell.device,
            bytes,
            crate::device::Memory::Readback,
            engine_gpu::books::owners::READOUT,
        )?;
        Ok(())
    }

    fn plan(_shell: &mut Shell<Api>, _prepared: &mut Prepared<'_, Api>) -> Result<Plan<Api>> {
        Ok(Plan::Walk {
            keep: None,
            split: false,
        })
    }

    fn submit(
        shell: &mut Shell<Api>,
        mut frame: Frame,
        prepared: &mut Prepared<'_, Api>,
        launch: Launch<'_, Api>,
    ) -> Result<Pending> {
        let Launch {
            seq,
            layout,
            pixels,
            counts,
            done,
            fed_check,
            ..
        } = launch;
        let arm = prepared.arm;
        // The host reads back its own spans alone: the rest of the seat is
        // for the step's programs, on the device.
        let hosted: Vec<(u64, u64, u64)> = layout
            .iter()
            .filter(|span| span.hosted && span.rows > 0)
            .map(|span| (span.at, span.at, span.bytes()))
            .collect();
        engine_gpu::backend::Frame::copy_regions(
            &mut frame,
            &shell.readout[arm],
            &shell.shelf.mirror[arm],
            &hosted,
        )?;
        // The fed lanes' token rows, for the landing to read with its rows.
        let fed = match fed_check.as_ref().map(engine_gpu::settle::FedCheck::rows) {
            Some(rows) if rows > 0 => {
                let (store, tokens) = shell.token_seat(arm);
                let (into, words) = &shell.shelf.fed[arm];
                let bytes = (u64::from(rows) * 4).min(into.bytes());
                engine_gpu::backend::Frame::copy(&mut frame, store, tokens, into, 0, bytes)?;
                Some((into.clone(), bytes, words.clone()))
            }
            _ => None,
        };
        let on_done = crate::landing::on_done(
            crate::landing::Seat {
                seq,
                arm,
                layout: layout.to_vec(),
                seat: shell.readout[arm].clone(),
                mirror: shell.shelf.mirror[arm].clone(),
                pixels: (pixels.values > 0).then(|| {
                    (
                        shell.pixel_out[arm].clone(),
                        shell.pixel_output(pixels.plane).to_string(),
                        pixels.spans.clone(),
                    )
                }),
                guests: prepared.guests.take(),
                fed,
            },
            std::sync::Arc::clone(&shell.shelf.mailbox),
            counts,
            done,
            fed_check,
            shell.shelf.worker.clone(),
        );
        let pending = frame.commit_async(None)?;
        on_done(None);
        Ok(pending)
    }

    /// What a step in flight keeps: its device time, filled once its work
    /// is done where it was walked timed.
    fn fly(pending: Pending, _fed: Option<Pending>) -> engine_gpu::submit::DeviceTime {
        pending.device_time()
    }

    fn reap(shell: &mut Shell<Api>) -> Result<()> {
        collect(shell);
        Ok(())
    }

    fn harvest_one(shell: &mut Shell<Api>) -> Result<()> {
        loop {
            collect(shell);
            let Some(flight) = shell.inflight.front() else {
                return Ok(());
            };
            tracing::debug!(seq = flight.seq, "harvest parks until the step lands");
            let mailbox = std::sync::Arc::clone(&shell.shelf.mailbox);
            crate::device::host::block_on_poll("harvest", |cx| {
                let mut post = mailbox.lock().expect("mailbox lock poisoned");
                if post.landed.is_empty() {
                    post.waker = Some(cx.waker().clone());
                    std::task::Poll::Pending
                } else {
                    std::task::Poll::Ready(())
                }
            })?;
        }
    }
}

/// Takes every step the mailbox holds.
fn collect(shell: &mut Shell<Api>) {
    let landed = std::mem::take(
        &mut shell
            .shelf
            .mailbox
            .lock()
            .expect("mailbox lock poisoned")
            .landed,
    );
    for step in landed {
        if shell
            .inflight
            .front()
            .is_some_and(|flight| flight.seq == step.seq)
            && let Some(flight) = shell.inflight.pop_front()
        {
            // Under a stated submit bound, the step's device time, read
            // without waiting once its backend filled it (`Shell::track`).
            shell.timed(flight.estimate, flight.api);
        }
        if step.failed {
            shell.arms.give(step.arm);
            continue;
        }
        shell.land(step.seq, step.arm, step.rows);
    }
}
