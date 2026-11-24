//! The staging ring: pinned host slots a fire's inputs are written into in
//! place, laid at load. A fire reads its slot as a host parameter and
//! copies it into the pool's slab at its start
//! (`crate::trace::Source::Stage`); nothing is uploaded per fire.
//!
//! `frames in flight + 1` slots, each the slab's rows: a slot is written
//! again only once its last fire has run on the device (the event its
//! execute returns, never a poll, whatever the fire outputs), so a fire
//! still queued never reads another's inputs. On the
//! device one slab serves every fire: fires reach it in order through the
//! donated pool, each copying its own slot at its start.

#![allow(unsafe_code)]

use crate::device::Device;
use crate::error::{Fault, Result};
use crate::inputs::ROW_WORDS;
use crate::pjrt::{Buffer, Event};

struct Slot {
    buffer: Buffer,
    at: *mut u32,
    /// Fires once its last fire has run.
    done: Option<Done>,
}

/// A run's device-complete event, shared by the slots one run reads (its
/// staging slot and its bulk slots): the first to wait takes and awaits it,
/// the others wait behind it and find it fired.
#[derive(Clone)]
pub struct Done(std::sync::Arc<std::sync::Mutex<Option<Event>>>);

impl Done {
    #[must_use]
    pub fn of(event: Event) -> Done {
        Done(std::sync::Arc::new(std::sync::Mutex::new(Some(event))))
    }

    /// Blocks until the run has completed.
    pub fn wait(&self) -> Result<()> {
        let mut event = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(event) = event.take() {
            event.wait()?;
        }
        Ok(())
    }
}

pub struct Ring {
    slots: Vec<Slot>,
    next: usize,
    rows: u32,
}

// SAFETY: a slot's pointer addresses its own buffer's pinned host memory,
// which lives as long as the ring; the ring is used from one thread at a
// time (`&mut self`).
unsafe impl Send for Ring {}

impl Ring {
    /// `slots` slots of `rows` pool rows each, in the device's pinned host
    /// memory.
    pub fn lay(device: &Device, slots: u32, rows: u32) -> Result<Ring> {
        let zeros = vec![0u8; rows as usize * ROW_WORDS * 4];
        let slots = (0..slots.max(1))
            .map(|_| {
                let buffer = device.upload_host(dtype::Dtype::U32, rows, ROW_WORDS as u32, &zeros)?;
                let at = buffer.host_pointer()?.cast::<u32>();
                Ok(Slot {
                    buffer,
                    at,
                    done: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Ring {
            slots,
            next: 0,
            rows,
        })
    }

    /// Writes `image` into the next slot, once the fire that last read it
    /// has run; answers the slot.
    pub fn write(&mut self, image: &[u32]) -> Result<usize> {
        let cap = self.rows as usize * ROW_WORDS;
        if image.len() > cap {
            return Err(Fault::Ceiling {
                what: "words of a fire's staged inputs (the slab this load sized)",
                need: image.len() as u64,
                have: cap as u64,
            });
        }
        let at = self.next;
        self.next = (self.next + 1) % self.slots.len();
        let slot = &mut self.slots[at];
        if let Some(done) = slot.done.take() {
            done.wait()?;
        }
        // SAFETY: the slot's buffer holds `cap` words at `slot.at`, and no
        // fire reads it now (its last one has run).
        unsafe { std::ptr::copy_nonoverlapping(image.as_ptr(), slot.at, image.len()) };
        Ok(at)
    }

    /// Writes `bytes` into the next slot (padded to whole words) once the
    /// program that last read it has run; answers the slot.
    pub fn write_bytes(&mut self, bytes: &[u8]) -> Result<usize> {
        let (whole, tail) = bytes.as_chunks::<4>();
        let mut words: Vec<u32> = whole.iter().map(|w| u32::from_le_bytes(*w)).collect();
        if !tail.is_empty() {
            let mut last = [0u8; 4];
            last[..tail.len()].copy_from_slice(tail);
            words.push(u32::from_le_bytes(last));
        }
        self.write(&words)
    }

    /// Slot `at`'s buffer, a fire's host parameter.
    #[must_use]
    pub fn buffer(&self, at: usize) -> &Buffer {
        &self.slots[at].buffer
    }

    /// Slot `at` is read by the run that `done` fires once it completes
    /// (`None`: one already waited for).
    pub fn read_by(&mut self, at: usize, done: Option<Done>) {
        self.slots[at].done = done;
    }

    /// Whether slot `at` waits on a run before it is written again.
    #[must_use]
    pub fn fenced(&self, at: usize) -> bool {
        self.slots[at].done.is_some()
    }

    /// Slots, and the rows each takes.
    #[must_use]
    pub fn shape(&self) -> (u32, u32) {
        (self.slots.len() as u32, self.rows)
    }
}
