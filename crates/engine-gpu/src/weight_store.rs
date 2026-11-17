use std::fs::File;

use crate::backend::{Backend, Buffer as _, Memory};
use crate::error::{Fault, Result};
use crate::handles::Handles;

type Job = (u64, u64, u64);

type ChunkJobs = (usize, Vec<Job>);

#[derive(Debug)]
struct Chunk<B: Backend> {
    start: u64,
    buffer: B::Buffer,
}

impl<B: Backend> Clone for Chunk<B> {
    fn clone(&self) -> Self {
        Chunk {
            start: self.start,
            buffer: self.buffer.clone(),
        }
    }
}

#[derive(Debug)]
pub struct Store<B: Backend> {
    chunks: Vec<Chunk<B>>,
    bytes: u64,
}

impl<B: Backend> Clone for Store<B> {
    fn clone(&self) -> Self {
        Store {
            chunks: self.chunks.clone(),
            bytes: self.bytes,
        }
    }
}

/// One plane's place in the store as [`Store::with`] lays it: `(offset,
/// reserved bytes, row bytes)`. A plane past the ceiling is laid in pieces
/// of whole rows where it states its row bytes, and refused where not.
pub type Span = (u64, u64, Option<u64>);

impl<B: Backend> Store<B> {
    pub fn zeroed(device: &B::Context, spans: &[(u64, u64)], ceiling: u64) -> Result<Store<B>> {
        let spans: Vec<Span> = spans.iter().map(|&(at, bytes)| (at, bytes, None)).collect();
        Store::with(device, &spans, ceiling, Memory::Device)
    }

    pub fn with(
        device: &B::Context,
        spans: &[Span],
        ceiling: u64,
        kind: Memory,
    ) -> Result<Store<B>> {
        let mut chunks: Vec<Chunk<B>> = Vec::new();
        let mut start = 0u64;
        let mut held = 0u64;
        let total = spans
            .last()
            .map_or(0, |&(offset, reserved, _)| offset + reserved);
        for &(offset, reserved, row) in spans {
            debug_assert_eq!(
                offset,
                start + held,
                "the spans are packed from zero, in order"
            );
            if reserved > ceiling {
                let rows = row
                    .filter(|&row| row > 0)
                    .map_or(0, |row| ceiling / row);
                // Pieces of whole rows, an even count where more than one
                // fits (a WGSL kernel lands bf16 columns in pairs).
                let rows = if rows > 1 { rows & !1 } else { rows };
                let (Some(row), true) = (row, rows > 0) else {
                    return Err(Fault::Ceiling {
                        what: "bytes of one plane in one buffer",
                        need: reserved,
                        have: ceiling,
                    });
                };
                if held > 0 {
                    chunks.push(Chunk {
                        start,
                        buffer: B::Buffer::with(device, held, kind, crate::books::owners::WEIGHTS)?,
                    });
                    start += held;
                    held = 0;
                }
                // Each piece a buffer of its own, whole rows; the plane's
                // padding lies past its last row.
                let piece = rows * row;
                let mut left = reserved;
                while left > 0 {
                    let bytes = piece.min(left);
                    chunks.push(Chunk {
                        start,
                        buffer: B::Buffer::with(device, bytes, kind, crate::books::owners::WEIGHTS)?,
                    });
                    start += bytes;
                    left -= bytes;
                }
                continue;
            }
            if held > 0 && held + reserved > ceiling {
                chunks.push(Chunk {
                    start,
                    buffer: B::Buffer::with(device, held, kind, crate::books::owners::WEIGHTS)?,
                });
                start += held;
                held = 0;
            }
            held += reserved;
        }

        if held > 0 || chunks.is_empty() {
            chunks.push(Chunk {
                start,
                buffer: B::Buffer::with(device, held, kind, crate::books::owners::WEIGHTS)?,
            });
        }
        Ok(Store {
            chunks,
            bytes: total,
        })
    }

    /// The pieces the plane of `bytes` at `offset` lies in, each a whole
    /// chunk's run of it: `(chunk, offset in it, bytes)`. One for a plane
    /// in one chunk.
    fn runs(&self, offset: u64, bytes: u64) -> Result<Vec<(usize, u64, u64)>> {
        let mut out = Vec::new();
        let mut at = offset;
        let end = offset.saturating_add(bytes);
        while at < end {
            let chunk = self
                .chunks
                .partition_point(|chunk| chunk.start <= at)
                .saturating_sub(1);
            let c = &self.chunks[chunk];
            let chunk_end = c.start + c.buffer.bytes();
            if at < c.start || at >= chunk_end {
                return Err(Fault::Ceiling {
                    what: "bytes of the weight store",
                    need: end,
                    have: self.bytes,
                });
            }
            let len = (end - at).min(chunk_end - at);
            out.push((chunk, at - c.start, len));
            at += len;
        }
        Ok(out)
    }

    /// Binds the plane of `bytes` at `offset`, `row` bytes a row, piece by
    /// piece: each piece's handle and its first row and rows.
    pub fn bind_pieces(
        &self,
        handles: &Handles<B>,
        offset: u64,
        bytes: u64,
        row: u64,
    ) -> Result<Vec<(u32, u32, u32)>> {
        let mut out = Vec::new();
        let mut first = 0u64;
        for (chunk, local, len) in self.runs(offset, bytes)? {
            if len % row.max(1) != 0 {
                return Err(Fault::Ceiling {
                    what: "bytes of a piece that are whole rows",
                    need: len,
                    have: len / row.max(1) * row,
                });
            }
            let rows = len / row.max(1);
            let handle = handles.bind(&self.chunks[chunk].buffer, local, len)?;
            out.push((
                handle,
                u32::try_from(first).unwrap_or(u32::MAX),
                u32::try_from(rows).unwrap_or(u32::MAX),
            ));
            first += rows;
        }
        Ok(out)
    }

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

    pub fn bind(&self, handles: &Handles<B>, offset: u64, len: u64) -> Result<u32> {
        let (at, local) = self.locate(offset, len)?;
        handles.bind(&self.chunks[at].buffer, local, len)
    }

    pub fn write(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        let mut done = 0usize;
        for (at, local, len) in self.runs(offset, bytes.len() as u64)? {
            let len = usize::try_from(len).unwrap_or(usize::MAX);
            self.chunks[at]
                .buffer
                .write(local, &bytes[done..done + len])?;
            done += len;
        }
        Ok(())
    }

    pub fn zero_span(&mut self, offset: u64, len: u64) -> Result<()> {
        for (at, local, len) in self.runs(offset, len)? {
            self.chunks[at].buffer.zero_span(local, len)?;
        }
        Ok(())
    }

    pub fn read(&self, offset: u64, into: &mut [u8]) -> Result<()> {
        let mut done = 0usize;
        for (at, local, len) in self.runs(offset, into.len() as u64)? {
            let len = usize::try_from(len).unwrap_or(usize::MAX);
            self.chunks[at]
                .buffer
                .read(local, &mut into[done..done + len])?;
            done += len;
        }
        Ok(())
    }

    fn group(&self, jobs: &[Job]) -> Result<Vec<ChunkJobs>> {
        let mut groups: Vec<ChunkJobs> = Vec::new();
        for &(into, from, len) in jobs {
            let mut from = from;
            for (at, local, len) in self.runs(into, len)? {
                match groups.iter_mut().find(|(chunk, _)| *chunk == at) {
                    Some((_, list)) => list.push((local, from, len)),
                    None => groups.push((at, vec![(local, from, len)])),
                }
                from += len;
            }
        }
        Ok(groups)
    }

    pub fn write_from_file(&mut self, file: &File, jobs: &[Job], threads: usize) -> Result<()> {
        for (at, local) in self.group(jobs)? {
            self.chunks[at]
                .buffer
                .write_from_file(file, &local, threads)?;
        }
        Ok(())
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    #[must_use]
    pub fn chunks(&self) -> usize {
        self.chunks.len()
    }
}
