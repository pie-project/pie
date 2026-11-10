//! The state pool: `total_chunks` chunks of `chunk_bytes` in up to
//! [`kernels_slang::SEGMENTS`] segments, each a buffer of its own no larger
//! than the device binds (SPEC 11: Vulkan and WebGPU bind no buffer past
//! their limit, so the pool is cut where Metal and CUDA map one range).
//! Physical chunk `p` lies in segment `p / segment_chunks` at `(p %
//! segment_chunks) * chunk_bytes`; a chunk never straddles two. Kernels read
//! the segments as one virtual space (`kernels/prim/pool.slang`): virtual
//! byte `p * chunk_bytes` is chunk `p`'s first.
//!
//! Segments come and go from the top, a whole buffer at a time and never on
//! a fire's path ([`Pool::commit_segments`], [`Pool::release_segments`]).
//! The runtime names chunks logically; the pool places each on a physical
//! chunk through its [`ChunkTable`], so emptying the top segment
//! ([`Pool::evacuate`]) moves bytes and never renames a held slot.

use engine::chunk::{ChunkTable, Evacuation, Growth};
use kernels_slang::{SEGMENTS, Segments};

use crate::device::ctx::Frame;
use crate::device::{Buffer, Context, Handles};
use crate::error::{Fault, Result};

#[derive(Debug)]
pub struct Pool {
    segments: Vec<Option<Buffer>>,
    chunk_bytes: u64,
    total_chunks: u32,
    segment_chunks: u32,
    /// Segments backed, the first ones.
    committed: u32,
    table: ChunkTable,
}

/// The chunks a segment holds: a quarter of the pool (the kernels bind four
/// segments), never more than one binding spans, one at least.
#[must_use]
pub fn segment_chunks(chunk_bytes: u64, total_chunks: u32, binding: u64) -> u32 {
    let quarter = total_chunks.div_ceil(SEGMENTS as u32).max(1);
    let fits = u32::try_from(binding / chunk_bytes.max(1)).unwrap_or(u32::MAX);
    quarter.min(fits).max(1)
}

impl Pool {
    /// A pool of `total_chunks` chunks of `chunk_bytes`, segments of
    /// `segment_chunks`, nothing committed.
    pub fn reserve(chunk_bytes: u64, total_chunks: u32, segment_chunks: u32) -> Result<Pool> {
        if chunk_bytes == 0 || total_chunks == 0 || segment_chunks == 0 {
            return Err(Fault::Ceiling {
                what: "chunks of the state pool",
                need: 1,
                have: 0,
            });
        }
        let segments = total_chunks.div_ceil(segment_chunks);
        if segments as usize > SEGMENTS {
            return Err(Fault::Ceiling {
                what: "segments a kernel binds of the state pool",
                need: u64::from(segments),
                have: SEGMENTS as u64,
            });
        }
        // The kernels index the pool in u32 words, a bf16 at a time.
        let bytes = chunk_bytes * u64::from(segments * segment_chunks);
        if bytes / 2 > u64::from(u32::MAX) {
            return Err(Fault::Ceiling {
                what: "bytes of the state pool (a kernel indexes it a bf16 at a time)",
                need: bytes,
                have: 2 * u64::from(u32::MAX),
            });
        }
        Ok(Pool {
            segments: (0..segments).map(|_| None).collect(),
            chunk_bytes,
            total_chunks,
            segment_chunks,
            committed: 0,
            table: ChunkTable::identity(total_chunks),
        })
    }

    #[must_use]
    pub fn chunk_bytes(&self) -> u64 {
        self.chunk_bytes
    }

    #[must_use]
    pub fn total_chunks(&self) -> u32 {
        self.total_chunks
    }

    #[must_use]
    pub fn segment_chunks(&self) -> u32 {
        self.segment_chunks
    }

    /// The segments the pool's range holds.
    #[must_use]
    pub fn segments(&self) -> u32 {
        self.segments.len() as u32
    }

    #[must_use]
    pub fn committed_segments(&self) -> u32 {
        self.committed
    }

    /// The chunks with memory behind them: the first ones, a whole number
    /// of segments (the last one short where the pool ends).
    #[must_use]
    pub fn committed_chunks(&self) -> u32 {
        self.committed
            .saturating_mul(self.segment_chunks)
            .min(self.total_chunks)
    }

    #[must_use]
    pub fn growth(&self) -> Growth {
        Growth::Bottom
    }

    /// The bytes one segment's buffer holds: its chunks (the last one
    /// short where the pool ends).
    fn segment_bytes(&self, segment: u32) -> u64 {
        let first = segment * self.segment_chunks;
        let chunks = self.segment_chunks.min(self.total_chunks - first);
        u64::from(chunks) * self.chunk_bytes
    }

    /// The pool's whole range.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.chunk_bytes * u64::from(self.total_chunks)
    }

    /// The bytes with memory behind them now.
    #[must_use]
    pub fn committed_bytes(&self) -> u64 {
        self.segments.iter().flatten().map(Buffer::bytes).sum()
    }

    /// The physical chunk logical chunk `logical` sits on.
    pub fn physical(&self, logical: u32) -> Result<u32> {
        self.table.physical(logical).ok_or(Fault::Ceiling {
            what: "chunks of the state pool",
            need: u64::from(logical) + 1,
            have: u64::from(self.total_chunks),
        })
    }

    /// Refuses a physical chunk with no memory behind it.
    pub fn admit(&self, physical: u32) -> Result<()> {
        let have = self.committed_chunks();
        if physical >= have {
            return Err(Fault::Ceiling {
                what: "committed chunks of the state pool",
                need: u64::from(physical) + 1,
                have: u64::from(have),
            });
        }
        Ok(())
    }

    /// The segment buffer virtual byte `at` lies in and its offset there;
    /// refuses an uncommitted one.
    pub fn locate(&self, at: u64) -> Result<(&Buffer, u64)> {
        let span = u64::from(self.segment_chunks) * self.chunk_bytes;
        let segment = at / span;
        let buffer = self
            .segments
            .get(usize::try_from(segment).unwrap_or(usize::MAX))
            .and_then(Option::as_ref)
            .ok_or(Fault::Ceiling {
                what: "committed segments of the state pool",
                need: segment + 1,
                have: u64::from(self.committed),
            })?;
        Ok((buffer, at % span))
    }

    /// The pool as kernels bind it: every committed segment, the rest
    /// absent.
    pub fn bind(&self, handles: &Handles) -> Result<Segments> {
        let mut bufs = [kernels_slang::ABSENT; SEGMENTS];
        for (at, segment) in self.segments.iter().enumerate() {
            if let Some(buffer) = segment {
                bufs[at] = handles.bind(buffer, 0, buffer.bytes())?;
            }
        }
        let words = u64::from(self.segment_chunks) * self.chunk_bytes / 4;
        Ok(Segments {
            bufs,
            words: u32::try_from(words).unwrap_or(u32::MAX),
        })
    }

    /// Backs the first `segments` segments, each zeroed; what is backed
    /// already stands, and fewer than are backed is no change.
    pub fn commit_segments(&mut self, device: &Context, segments: u32) -> Result<()> {
        let max = self.segments();
        if segments > max {
            return Err(Fault::Ceiling {
                what: "segments of the state pool",
                need: u64::from(segments),
                have: u64::from(max),
            });
        }
        for segment in self.committed..segments {
            let buffer = Buffer::zeroed(device, self.segment_bytes(segment))?;
            self.segments[segment as usize] = Some(buffer);
            self.committed = segment + 1;
        }
        Ok(())
    }

    /// Gives back every segment at and past `segments`. The caller has
    /// drained every frame, and the runtime evacuated their chunks first.
    pub fn release_segments(&mut self, segments: u32) {
        for segment in segments..self.committed {
            self.segments[segment as usize] = None;
        }
        self.committed = self.committed.min(segments);
    }

    /// Copies each moved chunk onto its new physical chunk in `frame`, then
    /// takes the moves into the table: the caller commits `frame` before any
    /// later fire is encoded, so the copies land first.
    /// Returns each move as `(from, to)` physical chunks, for what lies
    /// beside the pool indexed by physical chunk.
    pub fn evacuate(
        &mut self,
        frame: &mut Frame,
        evacuation: &Evacuation,
    ) -> Result<Vec<(u32, u32)>> {
        let mut moved = Vec::with_capacity(evacuation.moves.len());
        let mut table = self.table.clone();
        table.apply(evacuation).map_err(|why| Fault::Program {
            at: "store::pool::evacuate",
            why: why.to_string(),
        })?;
        for &(logical, physical) in &evacuation.moves {
            let from = self.physical(logical)?;
            self.admit(from)?;
            self.admit(physical)?;
            if from == physical {
                continue;
            }
            moved.push((from, physical));
            self.copy(
                frame,
                u64::from(from) * self.chunk_bytes,
                u64::from(physical) * self.chunk_bytes,
                self.chunk_bytes,
            )?;
        }
        self.table = table;
        Ok(moved)
    }

    /// Copies `bytes` from virtual byte `from` to `to` in `frame`; neither
    /// run crosses a chunk.
    pub fn copy(&self, frame: &mut Frame, from: u64, to: u64, bytes: u64) -> Result<()> {
        let (src, src_at) = self.locate(from)?;
        let (dst, dst_at) = self.locate(to)?;
        frame.copy(src, src_at, dst, dst_at, bytes)
    }

    /// `into.len()` bytes from virtual byte `at`, read on the host.
    pub fn read(&self, at: u64, into: &mut [u8]) -> Result<()> {
        let (buffer, local) = self.locate(at)?;
        buffer.read(local, into)
    }

    /// Writes `from` at virtual byte `at` through the queue, after every
    /// submitted frame.
    pub fn write(&mut self, at: u64, from: &[u8]) -> Result<()> {
        let span = u64::from(self.segment_chunks) * self.chunk_bytes;
        let segment = usize::try_from(at / span).unwrap_or(usize::MAX);
        let buffer = self
            .segments
            .get_mut(segment)
            .and_then(Option::as_mut)
            .ok_or(Fault::Ceiling {
                what: "committed segments of the state pool",
                need: segment as u64 + 1,
                have: u64::from(self.committed),
            })?;
        buffer.write(at % span, from)
    }

    /// Zeroes `len` bytes at virtual byte `at`. The caller has drained every
    /// frame that touches them.
    pub fn zero(&mut self, at: u64, len: u64) -> Result<()> {
        let span = u64::from(self.segment_chunks) * self.chunk_bytes;
        let segment = usize::try_from(at / span).unwrap_or(usize::MAX);
        let buffer = self
            .segments
            .get_mut(segment)
            .and_then(Option::as_mut)
            .ok_or(Fault::Ceiling {
                what: "committed segments of the state pool",
                need: segment as u64 + 1,
                have: u64::from(self.committed),
            })?;
        buffer.zero_span(at % span, len)
    }
}

#[cfg(all(test, feature = "wgpu"))]
mod tests {
    use super::*;

    #[test]
    fn a_quarter_of_the_pool_a_segment_never_past_a_binding() {
        assert_eq!(segment_chunks(1 << 20, 100, 1 << 30), 25);
        assert_eq!(segment_chunks(1 << 20, 1000, 128 << 20), 128);
        assert_eq!(segment_chunks(1 << 20, 3, 1 << 30), 1);
    }

    #[test]
    fn a_pool_commits_evacuates_and_releases_segments() {
        if !crate::device::present() {
            eprintln!("no wgpu device on this machine; skipping");
            return;
        }
        let device = Context::bind(
            &crate::DeviceBoot::default(),
            engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
        )
        .expect("a wgpu device");
        let chunk = 4096u64;
        let mut pool = Pool::reserve(chunk, 8, 2).expect("a pool");
        assert_eq!(pool.segments(), 4);
        pool.commit_segments(&device, 4).expect("committed");
        let pattern: Vec<u8> = (0..chunk).map(|at| (at % 253) as u8).collect();
        let mut staged = Buffer::host(&device, chunk).expect("a host buffer");
        staged.write(0, &pattern).expect("staged");
        let (seg, at) = pool.locate(7 * chunk).expect("chunk 7");
        let mut frame = device.frame().expect("a frame");
        frame.copy(&staged, 0, seg, at, chunk).expect("uploaded");
        frame.commit().expect("committed");
        let mut frame = device.frame().expect("a frame");
        pool.evacuate(
            &mut frame,
            &Evacuation {
                moves: vec![(7, 1)],
            },
        )
        .expect("moved");
        frame.commit().expect("committed");
        assert_eq!(pool.physical(7).unwrap(), 1);
        pool.release_segments(2);
        assert_eq!(pool.committed_chunks(), 4);
        assert!(pool.locate(6 * chunk).is_err());
        let mut back = vec![0u8; chunk as usize];
        let (seg, at) = pool.locate(chunk).expect("chunk 1");
        let mut frame = device.frame().expect("a frame");
        frame.copy(seg, at, &staged, 0, chunk).expect("downloaded");
        frame.commit().expect("committed");
        staged.read(0, &mut back).expect("read");
        assert_eq!(back, pattern);
    }
}
