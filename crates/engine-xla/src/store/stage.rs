//! The staging ring: pinned host slots a fire's inputs are written into in
//! place, laid at load. A fire reads its slot as a host parameter and
//! copies it into the pool's slab at its start
//! (`crate::trace::Source::Stage`); nothing is uploaded per fire.
//!
//! `frames in flight + 1` slots, each the slab's rows: a slot is written
//! again only once the pool its last fire returned is ready (an event, never
//! a poll), so a fire still queued never reads another's inputs. On the
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
    /// The pool its last fire returned, ready once that fire has run.
    done: Option<Event>,
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

    /// Slot `at` is read by the fire whose pool `done` fires once it ran.
    pub fn read_by(&mut self, at: usize, done: Option<Event>) {
        self.slots[at].done = done;
    }

    /// Slots, and the rows each takes.
    #[must_use]
    pub fn shape(&self) -> (u32, u32) {
        (self.slots.len() as u32, self.rows)
    }
}
