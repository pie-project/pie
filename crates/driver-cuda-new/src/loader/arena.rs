//! The persistent arena of an executed `LoadPlan`, living on the DEVICE.
//!
//! `model_loader::executor::host` decides everything about a load — which file
//! extents to read, which transforms to run, where each tensor lands — and
//! bottoms out in three verbs on an [`ArenaBacking`]. This is those three
//! verbs against CUDA global memory.
//!
//! Which is the whole of the "device load plan executor" the C++ tree spent
//! `load_plan_executor.hpp` (629 lines), `weight_copy_engine.hpp` (399) and
//! the host half of `transcode_engine.hpp` on. The executor was never the
//! device-specific part; the *addressing* was.
//!
//! **Host memory stays bounded by the largest single write**, which is what a
//! 39 GB checkpoint needs: the executor reads one file extent at a time and
//! hands it here, and here it goes straight across. Nothing accumulates.

use std::borrow::Cow;

use model_loader::error::Error;
use model_loader::executor::arena::ArenaBacking;

use crate::cuda::{DeviceBuffer, OwnedStream};

/// A `LoadPlan`'s persistent arena as one CUDA allocation.
pub struct DeviceArena {
    buf: DeviceBuffer,
    stream: OwnedStream,
}

impl DeviceArena {
    /// Allocate `bytes` of device memory to execute a plan into.
    ///
    /// # Errors
    ///
    /// The device could not satisfy the allocation, or a stream to order the
    /// copies on could not be created.
    pub fn new(bytes: usize, alloc: &crate::cuda::Allocator) -> Result<Self, Error> {
        let buf = alloc.alloc(bytes).map_err(device)?;
        let stream = OwnedStream::new(0).map_err(device)?;
        Ok(Self { buf, stream })
    }

    /// The filled arena, once the plan has run.
    ///
    /// Takes `self` because a plan is executed once: handing the buffer back
    /// while the backing still exists would let a second execution write under
    /// the weights the first one published.
    ///
    /// # Errors
    ///
    /// The stream faulted while draining the writes.
    pub fn into_buffer(self) -> Result<DeviceBuffer, Error> {
        self.stream.as_ref().synchronize().map_err(device)?;
        Ok(self.buf)
    }
}

fn device(e: crate::Error) -> Error {
    Error::Contract(format!("device arena: {e:?}"))
}

impl ArenaBacking for DeviceArena {
    fn len(&self) -> usize {
        self.buf.len()
    }

    /// A device read is a STAGING COPY, and it synchronizes.
    ///
    /// Both are acceptable here and neither is on the write path: the executor
    /// reads the arena only to feed a transform whose input is a tensor it
    /// already wrote there, while it writes the arena once per file extent.
    fn read(&self, offset: usize, len: usize) -> Result<Cow<'_, [u8]>, Error> {
        let mut out = vec![0u8; len];
        self.buf
            .read_at(offset, &mut out, self.stream.as_ref())
            .map_err(device)?;
        self.stream.as_ref().synchronize().map_err(device)?;
        Ok(Cow::Owned(out))
    }

    /// Enqueued, not awaited — [`Self::into_buffer`] drains it.
    ///
    /// `bytes` is host memory the executor owns and reuses, so the copy has to
    /// be ordered against the next write anyway; leaving it on one stream does
    /// that, and a pageable source makes `cudaMemcpyAsync` synchronous in any
    /// case.
    fn write(&mut self, offset: usize, bytes: &[u8]) -> Result<(), Error> {
        let stream = self.stream.as_ref();
        self.buf.write_at(offset, bytes, stream).map_err(device)
    }

    fn fill(&mut self, offset: usize, len: usize, byte: u8) -> Result<(), Error> {
        let stream = self.stream.as_ref();
        self.buf.memset_at(offset, len, byte, stream).map_err(device)
    }
}
