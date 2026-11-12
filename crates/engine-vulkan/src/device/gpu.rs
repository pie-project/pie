//! The shared engine's traits over this crate's device (or its refusing
//! twin, where the `vulkan` feature is off).

use engine_gpu::backend::OnDone;

use super::{Api, Buffer, Context, Frame, Pending};
use crate::error::{Fault, Result};

impl engine_gpu::Backend for Api {
    const NAME: &'static str = "Vulkan";

    const MAPS_HOST: bool = true;

    const COPY_GRAIN: u64 = 1;

    type Context = Context;

    type Buffer = Buffer;

    type Frame = Frame;

    type Pending = Pending;

    fn is_device(domain: engine::transfer::MemoryDomain) -> bool {
        matches!(domain, engine::transfer::MemoryDomain::VulkanDevice(_))
    }

    fn stage_rows(
        staging: &Buffer,
        at: u64,
        len: u64,
        fill: impl FnOnce(usize) -> Result<()>,
    ) -> Result<()> {
        let base = staging.mapped_ptr().ok_or(Fault::Device {
            call: "gather",
            why: "the staging slot is not host-mapped".into(),
        })?;
        staging.span(at, len)?;
        fill(base + usize::try_from(at).unwrap_or(usize::MAX))
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

    fn copy_regions(
        &mut self,
        source: &Buffer,
        into: &Buffer,
        regions: &[(u64, u64, u64)],
    ) -> Result<()> {
        Frame::copy_regions(self, source, into, regions)
    }

    fn fill_zero(&mut self, into: &Buffer, at: u64, len: u64) -> Result<()> {
        Frame::fill_zero(self, into, at, len)
    }

    fn flush(&self) -> Result<()> {
        Frame::flush(self)
    }

    fn flush_settled(&self) -> Result<()> {
        Frame::flush_settled(self)
    }

    fn commit(self) -> Result<()> {
        Frame::commit(self)
    }

    fn commit_async(self, on_done: Option<OnDone>) -> Result<Pending> {
        Frame::commit_async(self, on_done)
    }
}

impl engine_gpu::backend::Pending for Pending {
    fn wait(&self) -> Result<()> {
        Pending::wait(self)
    }
}

/// A copy recorded with its own barriers whose ends the frame does not
/// retain: their owner outlives every frame.
#[cfg(feature = "vulkan")]
fn copy_unretained(
    frame: &Frame,
    source: &Buffer,
    source_at: u64,
    into: &Buffer,
    into_at: u64,
    len: u64,
) -> Result<()> {
    use ash::vk;
    if len == 0 {
        return Ok(());
    }
    source.span(source_at, len)?;
    into.span(into_at, len)?;
    let (source, into) = (source.slab(), into.slab());
    let d = &frame.core().device;
    let cmd = frame.cmd();

    // SAFETY: `cmd` is the open frame's buffer, recorded only on this thread
    // (`Frame` is not `Sync`); both slabs are live buffers of its device and
    // `span` bounds the copy in each. They are not retained by the frame: the
    // seats own them for the shell's life, and the shell's context
    // drains the queue before they drop.
    #[allow(unsafe_code)]
    unsafe {
        let before = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE);
        d.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[before],
            &[],
            &[],
        );
        d.cmd_copy_buffer(
            cmd,
            source.buffer,
            into.buffer,
            &[vk::BufferCopy::default()
                .src_offset(source_at)
                .dst_offset(into_at)
                .size(len)],
        );
        let after = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        d.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[after],
            &[],
            &[],
        );
    }
    Ok(())
}

#[cfg(not(feature = "vulkan"))]
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
