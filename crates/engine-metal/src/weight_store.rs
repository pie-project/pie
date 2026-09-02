//! **THE WEIGHT STORE, IN AS MANY BUFFERS AS IT TAKES.**
//!
//! The store used to be one `Buffer`, and one `MTLBuffer` is bounded by
//! `maxBufferLength` — 18.72 GiB on an M1 Max. A streamed dsv4 load's slab is
//! `slots` seats of every band of 43 groups, and at 56 seats that is 24.6 GB:
//! the load was refused by the buffer's ceiling while the box's memory would
//! have held it (swap did not move through 52). So the store is a VIRTUAL
//! contiguous span carved into chunks, each under the ceiling, and every
//! offset the layout hands out ([`weights::places`](crate::weights) and the
//! warm arm's own packing) stays a virtual one: a chunk is cut only at a
//! plane's boundary, so no plane straddles two buffers and a handle is still
//! one `(buffer, offset)` view.

use std::fs::File;

use crate::device::alloc::FileWriter;
use crate::device::{Buffer, Context, Handles};
use crate::error::{Fault, Result};

/// One chunk: where it starts in the virtual span, and the buffer.
#[derive(Clone, Debug)]
struct Chunk {
    start: u64,
    buffer: Buffer,
}

/// The store: chunks in ascending order over one virtual span.
#[derive(Clone, Debug)]
pub struct Store {
    chunks: Vec<Chunk>,
    bytes: u64,
}

impl Store {
    /// Reserve the store `spans` lays out — `(offset, reserved)` in ascending
    /// order, packed from zero — in chunks no larger than `ceiling`, cut only
    /// between spans.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for one span past the ceiling, and the buffer's own
    /// refusals.
    pub fn zeroed(device: &Context, spans: &[(u64, u64)], ceiling: u64) -> Result<Store> {
        let mut chunks: Vec<Chunk> = Vec::new();
        let mut start = 0u64;
        let mut held = 0u64;
        let total = spans.last().map_or(0, |&(offset, reserved)| offset + reserved);
        for &(offset, reserved) in spans {
            debug_assert_eq!(offset, start + held, "the spans are packed from zero, in order");
            if reserved > ceiling {
                return Err(Fault::Ceiling {
                    what: "bytes of one plane in one buffer",
                    need: reserved,
                    have: ceiling,
                });
            }
            if held > 0 && held + reserved > ceiling {
                chunks.push(Chunk {
                    start,
                    buffer: Buffer::zeroed(device, held)?,
                });
                start += held;
                held = 0;
            }
            held += reserved;
        }
        // The last chunk — and the whole store, when it fits one buffer or is
        // empty, which keeps the one-buffer load byte for byte what it was.
        chunks.push(Chunk {
            start,
            buffer: Buffer::zeroed(device, held)?,
        });
        Ok(Store {
            chunks,
            bytes: total,
        })
    }

    /// Which chunk holds `[offset, offset + len)`, and the offset inside it.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a span outside the store or across two chunks.
    fn locate(&self, offset: u64, len: u64) -> Result<(usize, u64)> {
        let end = offset.checked_add(len).ok_or(Fault::Ceiling {
            what: "bytes of the weight store",
            need: u64::MAX,
            have: self.bytes,
        })?;
        let at = self
            .chunks
            .partition_point(|chunk| chunk.start <= offset)
            .saturating_sub(1);
        let chunk = &self.chunks[at];
        if offset < chunk.start || end > chunk.start + chunk.buffer.bytes() {
            return Err(Fault::Ceiling {
                what: "bytes of one chunk of the weight store",
                need: end,
                have: chunk.start + chunk.buffer.bytes(),
            });
        }
        Ok((at, offset - chunk.start))
    }

    /// A handle over `[offset, offset + len)`, in the chunk that holds it.
    pub fn bind(&self, handles: &Handles, offset: u64, len: u64) -> Result<u32> {
        let (at, local) = self.locate(offset, len)?;
        handles.bind(&self.chunks[at].buffer, local, len)
    }

    /// As [`Buffer::write`], at a virtual offset.
    pub fn write(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        let (at, local) = self.locate(offset, bytes.len() as u64)?;
        self.chunks[at].buffer.write(local, bytes)
    }

    /// As [`Buffer::zero_span`], at a virtual offset.
    pub fn zero_span(&mut self, offset: u64, len: u64) -> Result<()> {
        let (at, local) = self.locate(offset, len)?;
        self.chunks[at].buffer.zero_span(local, len)
    }

    /// As [`Buffer::read`], at a virtual offset.
    pub fn read(&self, offset: u64, into: &mut [u8]) -> Result<()> {
        let (at, local) = self.locate(offset, into.len() as u64)?;
        self.chunks[at].buffer.read(local, into)
    }

    /// The jobs `(into, from, len)` — `into` virtual — grouped by chunk, each
    /// group's `into` made local: what one `pread` handle can take.
    fn group(&self, jobs: &[(u64, u64, u64)]) -> Result<Vec<(usize, Vec<(u64, u64, u64)>)>> {
        let mut groups: Vec<(usize, Vec<(u64, u64, u64)>)> = Vec::new();
        for &(into, from, len) in jobs {
            let (at, local) = self.locate(into, len)?;
            match groups.iter_mut().find(|(chunk, _)| *chunk == at) {
                Some((_, list)) => list.push((local, from, len)),
                None => groups.push((at, vec![(local, from, len)])),
            }
        }
        Ok(groups)
    }

    /// As [`Buffer::write_from_file`], over virtual offsets — one call per
    /// chunk the jobs touch.
    pub fn write_from_file(
        &mut self,
        file: &File,
        jobs: &[(u64, u64, u64)],
        threads: usize,
    ) -> Result<()> {
        for (at, local) in self.group(jobs)? {
            self.chunks[at].buffer.write_from_file(file, &local, threads)?;
        }
        Ok(())
    }

    /// As [`Buffer::file_writer`], over virtual offsets: one `Send` handle
    /// per chunk the jobs touch, each with its own local job list.
    pub fn file_writers(
        &mut self,
        jobs: &[(u64, u64, u64)],
    ) -> Result<Vec<(FileWriter, Vec<(u64, u64, u64)>)>> {
        let mut out = Vec::new();
        for (at, local) in self.group(jobs)? {
            let writer = self.chunks[at].buffer.file_writer(&local)?;
            out.push((writer, local));
        }
        Ok(out)
    }

    /// The virtual span's length — what the store reserves in all.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// How many buffers the store is carved into.
    #[must_use]
    pub fn chunks(&self) -> usize {
        self.chunks.len()
    }
}
