//! The shared engine's traits over this crate's device (or its refusing
//! twin, where the `wgpu` feature is off).

use engine_gpu::backend::OnDone;

use super::{Api, Buffer, Context, Frame, Pending};
use engine_gpu::error::{Fault, Result};

impl engine_gpu::Backend for Api {
    const NAME: &'static str = "wgpu";

    const MAPS_HOST: bool = false;

    const COPY_GRAIN: u64 = 4;

    /// The pool is laid whole at load and never resized, as XLA's: no
    /// `yield` applies.
    const RESIZES_POOL: bool = false;

    const STORAGE: checkpoint::types::BackendKind = checkpoint::types::BackendKind::Wgpu;

    type Context = Context;

    type Buffer = Buffer;

    type Frame = Frame;

    type Pending = Pending;

    type Tuning = crate::tuning::Tuning;

    fn is_device(domain: engine::transfer::MemoryDomain) -> bool {
        matches!(
            domain,
            engine::transfer::MemoryDomain::Device {
                reach: engine::transfer::DeviceReach::Private,
                ..
            }
        )
    }

    fn stage_rows(
        staging: &Buffer,
        at: u64,
        len: u64,
        fill: impl FnOnce(usize) -> Result<()>,
    ) -> Result<()> {
        stage_rows(staging, at, len, fill)
    }

    fn copy_unretained(
        frame: &Frame,
        source: &Buffer,
        source_at: u64,
        into: &Buffer,
        into_at: u64,
        len: u64,
    ) -> Result<()> {
        copy_unretained(frame, source, source_at, into, into_at, len)
    }
}

impl engine_gpu::backend::Context<Api> for Context {
    fn frame(&self) -> Result<Frame> {
        Context::frame(self)
    }

    fn spanned(&self) -> Result<Frame> {
        Context::spanned(self)
    }

    fn idle(&self) -> bool {
        Context::idle(self)
    }

    fn max_buffer(&self) -> u64 {
        Context::max_buffer(self)
    }

    fn name(&self) -> &str {
        Context::name(self)
    }

    /// The workgroups the device holds at once, as the load measured them
    /// (`Context::resident`): every caller reads it past the load's rates
    /// step, which refuses a load whose profile still lacks it.
    fn cores(&self) -> u32 {
        Context::resident(self).unwrap_or(1)
    }

    fn resident(&self) -> Option<u32> {
        Context::resident(self)
    }

    fn capacity(&self) -> u64 {
        Context::capacity(self)
    }

    fn outside(&self) -> u64 {
        Context::outside(self)
    }

    fn host_room(&self) -> Option<engine::HostRoom> {
        Context::host_room(self)
    }

    fn memory(&self) -> engine::DeviceMemory {
        Context::whose_memory(self)
    }

    fn left(&self) -> u64 {
        Context::left(self)
    }

    fn books(&self) -> &engine_gpu::Books {
        Context::books(self)
    }

    fn bind_thread(&self) -> Result<()> {
        Context::bind_thread(self)
    }
}

impl engine_gpu::backend::Frame<Api> for Frame {
    fn copy(
        &mut self,
        source: &Buffer,
        source_at: u64,
        into: &Buffer,
        into_at: u64,
        len: u64,
    ) -> Result<()> {
        Frame::copy(self, source, source_at, into, into_at, len)
    }

    fn fill_zero(&mut self, into: &Buffer, at: u64, len: u64) -> Result<()> {
        Frame::fill_zero(self, into, at, len)
    }

    fn update(&self, into: &Buffer, at: u64, bytes: &[u8]) -> Result<()> {
        Frame::update(self, into, at, bytes)
    }

    fn dispatches(&self) -> u64 {
        Frame::dispatches(self)
    }

    fn uniform_room(&self) -> Option<(u64, u64)> {
        Some(Frame::uniform_room(self))
    }

    fn flush(&self) -> Result<()> {
        Frame::flush(self)
    }

    fn flush_settled(&self) -> Result<()> {
        Frame::flush_settled(self)
    }

    fn submit_segment(&self) -> Result<()> {
        Frame::submit_segment(self)
    }

    fn commit(self) -> Result<()> {
        Frame::commit(self)
    }

    fn commit_async(self, on_done: Option<OnDone>) -> Result<Pending> {
        Frame::commit_async(self, on_done)
    }

    fn commit_spanned(self) -> Result<Option<u64>> {
        Frame::commit_spanned(self)
    }
}

impl engine_gpu::backend::Pending for Pending {
    fn wait(&self) -> Result<()> {
        Pending::wait(self)
    }

    fn landed(&self) -> bool {
        Pending::landed(self)
    }
}

/// Lays the rows through a staging view of the queue: they land before the
/// work submitted next.
#[cfg(feature = "wgpu")]
fn stage_rows(
    staging: &Buffer,
    at: u64,
    len: u64,
    fill: impl FnOnce(usize) -> Result<()>,
) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    staging.span(at, len)?;
    let slab = staging.slab();
    let size = std::num::NonZeroU64::new(len).expect("a non-empty stage");
    let mut view = slab
        .core
        .queue
        .write_buffer_with(&slab.buffer, at, size)
        .ok_or(Fault::Device {
            call: "write_buffer_with",
            why: "no staging view".into(),
        })?;
    let base = view.slice(..).as_raw_ptr().as_ptr() as *mut u8 as usize;
    fill(base)
}

#[cfg(not(feature = "wgpu"))]
fn stage_rows(
    _staging: &Buffer,
    _at: u64,
    _len: u64,
    _fill: impl FnOnce(usize) -> Result<()>,
) -> Result<()> {
    Err(Fault::Deviceless)
}

#[cfg(feature = "wgpu")]
fn copy_unretained(
    frame: &Frame,
    source: &Buffer,
    source_at: u64,
    into: &Buffer,
    into_at: u64,
    len: u64,
) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    source.span(source_at, len)?;
    into.span(into_at, len)?;
    frame.encode(|encoder| {
        encoder.copy_buffer_to_buffer(
            &source.slab().buffer,
            source_at,
            &into.slab().buffer,
            into_at,
            Some(len),
        );
    })
}

#[cfg(not(feature = "wgpu"))]
fn copy_unretained(
    _frame: &Frame,
    _source: &Buffer,
    _source_at: u64,
    _into: &Buffer,
    _into_at: u64,
    _len: u64,
) -> Result<()> {
    Err(Fault::Deviceless)
}
