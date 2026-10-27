#![allow(unsafe_code)]
//! The streams a wave's program groups launch their stages on.
//!
//! The groups of one wave are distinct programs (or one program at distinct
//! extents) over distinct instances: no group reads what another writes, so
//! nothing orders them but the wave's pull before and its land after. Each
//! launches a handful of blocks a lane, so one after another they leave the
//! device all but idle; side by side they overlap.
//!
//! A group whose stage holds a lane-sampler region stays on the main stream:
//! that region's blocks wait for one another at a barrier and so need the
//! whole grid resident at once, which two such grids in flight together
//! could deny each other. Beside any number of grids that run to completion
//! it is safe, as those free what they hold.

use crate::device::graph::Event;
use crate::error::Result;

/// The side streams a wave fans out to, at most: the device's default
/// eight hardware queues, less the main stream's.
const SIDE: usize = 15;

#[derive(Debug, Default)]
pub struct Fan {
    side: Vec<*mut core::ffi::c_void>,
    forked: Option<Event>,
    joined: Vec<Event>,
    /// The side streams the last `fork` opened, which `join` closes.
    open: usize,
}

impl Fan {
    /// The stream each group launches on: `main` for those that `meet`
    /// across blocks, the others spread over `main` and the side streams,
    /// each side stream ordered after what `main` holds so far (the wave's
    /// pull).
    pub fn fork(
        &mut self,
        meet: &[bool],
        main: *mut core::ffi::c_void,
    ) -> Result<Vec<*mut core::ffi::c_void>> {
        let free = meet.iter().filter(|meets| !**meets).count();
        let width = free.saturating_sub(1).min(SIDE);
        if width == 0 {
            return Ok(vec![main; meet.len()]);
        }
        self.reserve(width)?;
        let forked = match &self.forked {
            Some(event) => event,
            None => self.forked.insert(Event::new()?),
        };
        forked.record(main)?;
        for &side in &self.side[..width] {
            forked.wait(side)?;
        }
        self.open = width;
        // The free groups take main and the side streams in turn; main also
        // carries every group that meets across blocks.
        let mut next = 0usize;
        Ok(meet
            .iter()
            .map(|&meets| {
                if meets {
                    return main;
                }
                let at = next % (width + 1);
                next += 1;
                if at == 0 { main } else { self.side[at - 1] }
            })
            .collect())
    }

    /// `main` waits for every side stream the last `fork` opened.
    pub fn join(&mut self, main: *mut core::ffi::c_void) -> Result<()> {
        let open = core::mem::take(&mut self.open);
        for (side, joined) in self.side[..open].iter().zip(&self.joined) {
            joined.record(*side)?;
            joined.wait(main)?;
        }
        Ok(())
    }

    fn reserve(&mut self, width: usize) -> Result<()> {
        while self.side.len() < width {
            self.side.push(open_stream()?);
        }
        while self.joined.len() < width {
            self.joined.push(Event::new()?);
        }
        Ok(())
    }
}

#[cfg(feature = "device")]
fn open_stream() -> Result<*mut core::ffi::c_void> {
    use crate::device::rt;

    let mut stream: rt::cudaStream_t = core::ptr::null_mut();
    // SAFETY: a live out-parameter; the stream is the fan's, destroyed once
    // in its `Drop`. Flag 1 is `cudaStreamNonBlocking`, as the main stream's
    // companions are opened.
    unsafe {
        crate::device::ctx::check(
            "cudaStreamCreateWithFlags",
            rt::cudaStreamCreateWithFlags(&raw mut stream, 1),
        )?;
    }
    Ok(stream.cast())
}

#[cfg(not(feature = "device"))]
fn open_stream() -> Result<*mut core::ffi::c_void> {
    Err(crate::error::Fault::Runtimeless)
}

impl Drop for Fan {
    fn drop(&mut self) {
        #[cfg(feature = "device")]
        for side in self.side.drain(..) {
            // SAFETY: opened by `open_stream` and destroyed once; the shell
            // synchronizes before it tears the plane down.
            unsafe {
                let _ = crate::device::rt::cudaStreamDestroy(side.cast());
            }
        }
    }
}
