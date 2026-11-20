//! The shared shell ([`engine_gpu::serve`]) bound to WebGPU: a step's rows
//! land through a readback mirror into a mailbox, never by a blocking wait
//! on the browser's thread.

use engine_gpu::serve::{Launch, Plan, Prepared, Serve, Shell};

use crate::device::{Api, Buffer, Context, Frame, Pending, Pipelines};
use engine_gpu::error::Result;

/// What the shell keeps of WebGPU's own: each arm's readback mirror, the
/// mailbox landed steps post to, and the worker that lands them.
pub struct Shelf {
    mirror: Vec<Buffer>,

    /// The guest plane's readback buffers, one a thread that runs guest
    /// passes, a chunk of the pool each, and its readout widen's table:
    /// reserved before the pool is sized, the plane takes them at its load.
    #[cfg(feature = "wgpu")]
    pub(crate) guest: (Vec<Buffer>, Buffer),

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

    type Flight = ();

    type Keep = std::convert::Infallible;

    const FUSIONS: &'static [poem_compiler::fuse::Rule] = &crate::FUSIONS;

    const MAX_BUFFER_SOURCE: &'static str = "the least of wgpu::Limits::max_buffer_size, \
        max_storage_buffer_binding_size and, on Vulkan, maxMemoryAllocationSize";

    const RECORDS: bool = false;

    /// Every cell of a guest's rings lies on the host: a pass reads its
    /// takes from there and lands its puts there.
    const CHANNEL_CELLS: bool = false;

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
        let (chip, report) = crate::chip::resolve(&measured, boot.profiles.as_deref());
        crate::chip::SHELF.tell_untuned(&measured.name, &report);
        let tuning = crate::tuning::Tuning::new(chip, device.cores())
            .with_workgroup_bytes(device.workgroup_bytes());
        // A device whose profile lacks the roofline's constants measures
        // them here, before ready, and keeps them in the user's profile (a
        // browser masks its adapter, so no profile states its rates, and
        // keeps them for its load alone), each submission within the
        // platform's stated bound on one.
        let tuning = crate::rates::at_load(
            &device,
            tuning,
            boot.profiles.as_deref(),
            crate::rates::stated_submit_bound(),
        )?;
        Ok((device, tuning))
    }

    fn chunk_bytes(boot: &crate::api::DeviceBoot) -> Option<u64> {
        boot.chunk_bytes
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

    fn walk_frame(device: &Context, _keep: bool) -> Result<Frame> {
        device.frame()
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
            guest: (
                (0..threads)
                    .map(|_| {
                        Buffer::with(
                            device,
                            guest.chunk_bytes,
                            crate::device::Memory::Readback,
                            engine_gpu::books::owners::GUEST,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?,
                Buffer::with(
                    device,
                    crate::guest::widen::Widen::table_bytes(
                        device,
                        guest.seats,
                        threads,
                        u64::from(guest.max_lanes) * u64::from(guest.planes),
                    ),
                    crate::device::Memory::Device,
                    engine_gpu::books::owners::GUEST,
                )?,
            ),
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

    fn fly(pending: Pending, _fed: Option<Pending>) {
        #[cfg_attr(not(feature = "wgpu"), allow(clippy::drop_non_drop))]
        drop(pending);
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
        {
            shell.inflight.pop_front();
        }
        if step.failed {
            shell.arms.give(step.arm);
            continue;
        }
        shell.land(step.seq, step.arm, step.rows);
    }
}
