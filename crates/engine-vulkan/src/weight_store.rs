use std::fs::File;

use crate::device::alloc::{FileWriter, Memory};
use crate::device::{Buffer, Context, Handles};
use crate::error::{Fault, Result};

type Job = (u64, u64, u64);

type ChunkJobs = (usize, Vec<Job>);

#[derive(Clone, Debug)]
struct Chunk {
    start: u64,
    buffer: Buffer,
}

#[derive(Clone, Debug)]
pub struct Store {
    chunks: Vec<Chunk>,
    bytes: u64,
}

impl Store {
    pub fn zeroed(device: &Context, spans: &[(u64, u64)], ceiling: u64) -> Result<Store> {
        Store::with(device, spans, ceiling, Memory::Device)
    }

    pub fn with(
        device: &Context,
        spans: &[(u64, u64)],
        ceiling: u64,
        kind: Memory,
    ) -> Result<Store> {
        let mut chunks: Vec<Chunk> = Vec::new();
        let mut start = 0u64;
        let mut held = 0u64;
        let total = spans
            .last()
            .map_or(0, |&(offset, reserved)| offset + reserved);
        for &(offset, reserved) in spans {
            debug_assert_eq!(
                offset,
                start + held,
                "the spans are packed from zero, in order"
            );
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
                    buffer: Buffer::with(device, held, kind)?,
                });
                start += held;
                held = 0;
            }
            held += reserved;
        }

        chunks.push(Chunk {
            start,
            buffer: Buffer::with(device, held, kind)?,
        });
        Ok(Store {
            chunks,
            bytes: total,
        })
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

    pub fn bind(&self, handles: &Handles, offset: u64, len: u64) -> Result<u32> {
        let (at, local) = self.locate(offset, len)?;
        handles.bind(&self.chunks[at].buffer, local, len)
    }

    pub fn write(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        let (at, local) = self.locate(offset, bytes.len() as u64)?;
        self.chunks[at].buffer.write(local, bytes)
    }

    pub fn zero_span(&mut self, offset: u64, len: u64) -> Result<()> {
        let (at, local) = self.locate(offset, len)?;
        self.chunks[at].buffer.zero_span(local, len)
    }

    pub fn read(&self, offset: u64, into: &mut [u8]) -> Result<()> {
        let (at, local) = self.locate(offset, into.len() as u64)?;
        self.chunks[at].buffer.read(local, into)
    }

    fn group(&self, jobs: &[Job]) -> Result<Vec<ChunkJobs>> {
        let mut groups: Vec<ChunkJobs> = Vec::new();
        for &(into, from, len) in jobs {
            let (at, local) = self.locate(into, len)?;
            match groups.iter_mut().find(|(chunk, _)| *chunk == at) {
                Some((_, list)) => list.push((local, from, len)),
                None => groups.push((at, vec![(local, from, len)])),
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

    pub fn file_writers(&mut self, jobs: &[Job]) -> Result<Vec<(FileWriter, Vec<Job>)>> {
        let mut out = Vec::new();
        for (at, local) in self.group(jobs)? {
            let writer = self.chunks[at].buffer.file_writer(&local)?;
            out.push((writer, local));
        }
        Ok(out)
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
