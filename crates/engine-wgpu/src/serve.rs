//! The shared shell ([`engine_gpu::serve`]) bound to WebGPU: a step's rows
//! land through a readback mirror into a mailbox, never by a blocking wait
//! on the browser's thread.

use engine_gpu::serve::{Launch, Plan, Serve};

use crate::device::{Api, Buffer, Context, Frame, Pending, Pipelines};
use crate::error::Result;

pub use engine_gpu::serve::{
    Attached, Clips, Landed, Lane, Media, PixelSpans, READOUT_ROWS_PER_LANE, ReadSpan, rows_from,
};

pub type Shell = engine_gpu::serve::Shell<Api>;
pub type Boot<'a> = engine_gpu::serve::Boot<'a, Api>;
pub type Seated<'a> = engine_gpu::serve::Seated<'a, Api>;
pub type StepView<'a> = engine_gpu::serve::StepView<'a, Api>;
pub type Prepared<'a> = engine_gpu::serve::Prepared<'a, Api>;
pub type Enqueued<'a> = engine_gpu::serve::Enqueued<'a, Api>;
pub type InputCell = engine_gpu::serve::InputCell<Api>;

/// What the shell keeps of WebGPU's own: each arm's readback mirror, the
/// mailbox landed steps post to, and the worker that lands them.
pub struct Shelf {
    mirror: Vec<Buffer>,

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

    const RECORDS: bool = false;

    fn open(
        boot: &crate::api::DeviceBoot,
        share: f64,
        handed: Option<crate::device::Handed>,
    ) -> Result<(Context, crate::tuning::Tuning)> {
        let device = match handed {
            Some(handed) => Context::adopt(boot, share, handed)?,
            None => Context::bind(boot, share)?,
        };
        let measured = device.chip();
        let (chip, report) = crate::chip::resolve(&measured, boot.profiles.as_deref());
        crate::chip::SHELF.tell_untuned(&measured.name, &report);
        let tuning = crate::tuning::Tuning::new(chip, device.cores(), device.storage_buffers());
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

    fn compiled(pipelines: &Pipelines) -> u64 {
        pipelines.compiled()
    }

    fn sink<'f>(
        device: &'f Context,
        frame: &'f Frame,
        pipelines: &'f Pipelines,
        handles: &'f crate::device::Handles,
    ) -> crate::encode::Sink<'f> {
        crate::encode::Sink::new(device, frame, pipelines, handles)
    }

    fn walk_frame(device: &Context, _keep: bool) -> Result<Frame> {
        device.frame()
    }

    fn shelf(device: &Context, readout: &[Buffer], _weights_stream: bool) -> Result<Shelf> {
        Ok(Shelf {
            mirror: readout
                .iter()
                .map(|seat| Buffer::with(device, seat.bytes(), crate::device::Memory::Readback))
                .collect::<Result<Vec<_>>>()?,
            mailbox: std::sync::Arc::default(),
            worker: crate::landing::Worker::default(),
        })
    }

    fn forget(_shelf: &mut Shelf) {}

    fn grown(shell: &mut Shell, arm: usize, bytes: u64) -> Result<()> {
        shell.shelf.mirror[arm] =
            Buffer::with(&shell.device, bytes, crate::device::Memory::Readback)?;
        Ok(())
    }

    fn plan(_shell: &mut Shell, _prepared: &mut Prepared<'_>) -> Result<Plan<Api>> {
        Ok(Plan::Walk {
            keep: None,
            split: false,
        })
    }

    fn submit(
        shell: &mut Shell,
        mut frame: Frame,
        prepared: &mut Prepared<'_>,
        launch: Launch<'_, Api>,
    ) -> Result<Pending> {
        let Launch {
            seq,
            layout,
            pixels,
            counts,
            done,
            ..
        } = launch;
        let arm = prepared.arm;
        let hosted = layout
            .iter()
            .filter(|span| span.host)
            .map(|span| span.at + span.bytes())
            .max()
            .unwrap_or(0);
        frame.copy(&shell.readout[arm], 0, &shell.shelf.mirror[arm], 0, hosted)?;
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

    fn reap(shell: &mut Shell) -> Result<()> {
        collect(shell);
        Ok(())
    }

    fn harvest_one(shell: &mut Shell) -> Result<()> {
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
fn collect(shell: &mut Shell) {
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
