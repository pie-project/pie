//! The state pool: `total_chunks` chunks of `chunk_bytes` in up to
//! [`kernels_slang::SEGMENTS`] segments, each a buffer of its own no larger
//! than the device binds (SPEC 11: Vulkan and WebGPU bind no buffer past
//! their limit, so the pool is cut where Metal and CUDA map one range).
//! Physical chunk `p` lies in segment `p / segment_chunks` at `(p %
//! segment_chunks) * chunk_bytes`; a chunk never straddles two. Kernels read
//! the segments as one virtual space (`kernels/prim/pool.slang`): virtual
//! byte `p * chunk_bytes` is chunk `p`'s first.
//!
//! The top `span` chunks are the activation span ([`crate::arena`]): they
//! lie in the last segment, never across two, and every segment holding
//! one of them is backed from the load on (`Growth::TwoEnded`). Below
//! them, segments come and go as a prefix, a whole buffer at a time and
//! never on a fire's path ([`Pool::commit_segments`],
//! [`Pool::release_segments`]). A rung lower than the largest lends the
//! span under its carve to state; raising it moves that state out first
//! ([`Pool::evacuate`]).
//! The runtime names chunks logically; the pool places each on a physical
//! chunk through its [`ChunkTable`], so emptying the top segment
//! ([`Pool::evacuate`]) moves bytes and never renames a held slot.

use engine::chunk::{ChunkTable, Evacuation, Growth};
use kernels_slang::{SEGMENTS, Segments};

use crate::backend::{Backend, Buffer as _, Frame as _};
use crate::error::{Fault, Result};
use crate::handles::Handles;

#[derive(Debug)]
pub struct Pool<B: Backend> {
    segments: Vec<Option<B::Buffer>>,
    chunk_bytes: u64,
    total_chunks: u32,
    segment_chunks: u32,
    /// The activation span's chunks, the top ones.
    span: u32,
    /// The first segment holding a chunk of the span (the segment count
    /// where there is none): it and every one above are backed for good.
    top: u32,
    table: ChunkTable,
}

/// The chunks a segment holds for a pool of `total_chunks` whose top `span`
/// lie in its last segment: the pool cut into as many segments as it may
/// be (the kernels bind [`SEGMENTS`]), so commits stay fine, with each
/// segment within one binding and the last at least the span. Cut into `n`
/// segments of `ceil(total / n)`, the last holds at least `total / n -
/// (n - 1)`, so `n <= total / (span + SEGMENTS - 1)` keeps the span whole.
/// `None`: no count of segments does both (the load is refused).
#[must_use]
pub fn segment_chunks(chunk_bytes: u64, total_chunks: u32, binding: u64, span: u32) -> Option<u32> {
    let fits = u32::try_from(binding / chunk_bytes.max(1))
        .unwrap_or(u32::MAX)
        .max(1);
    let most = if span == 0 {
        SEGMENTS as u32
    } else {
        (total_chunks / (span + SEGMENTS as u32 - 1)).min(SEGMENTS as u32)
    };
    let least = total_chunks.div_ceil(fits);
    (most > 0 && least <= most).then(|| total_chunks.div_ceil(most))
}

impl<B: Backend> Pool<B> {
    /// A pool of `total_chunks` chunks of `chunk_bytes`, segments of
    /// `segment_chunks`, the top `span` the activation's; nothing committed.
    pub fn reserve(
        chunk_bytes: u64,
        total_chunks: u32,
        segment_chunks: u32,
        [lent, span]: [u32; 2],
    ) -> Result<Pool<B>> {
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
        // The kernels index state in u32 words, a bf16 at a time: every
        // chunk under the smallest rung's activation (`lent` chunks); what
        // lies past it is the fire's alone, bound by offset.
        let bytes = chunk_bytes * u64::from(total_chunks.saturating_sub(lent));
        if bytes / 2 > u64::from(u32::MAX) {
            return Err(Fault::Ceiling {
                what: "bytes of the state pool (a kernel indexes it a bf16 at a time)",
                need: bytes,
                have: 2 * u64::from(u32::MAX),
            });
        }
        if span >= total_chunks
            || (span > 0 && (total_chunks - span) / segment_chunks + 1 != segments)
        {
            return Err(Fault::Ceiling {
                what: "chunks of the activation span in the pool's last segment",
                need: u64::from(span),
                have: u64::from(total_chunks - (segments - 1) * segment_chunks),
            });
        }
        Ok(Pool {
            segments: (0..segments).map(|_| None).collect(),
            chunk_bytes,
            total_chunks,
            segment_chunks,
            span,
            top: if span == 0 {
                segments
            } else {
                (total_chunks - span) / segment_chunks
            },
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

    /// The segments backed from the first on, without a gap.
    #[must_use]
    pub fn committed_segments(&self) -> u32 {
        self.segments
            .iter()
            .take_while(|segment| segment.is_some())
            .count() as u32
    }

    /// The activation span's chunks.
    #[must_use]
    pub fn span(&self) -> u32 {
        self.span
    }

    #[must_use]
    pub fn growth(&self) -> Growth {
        if self.span > 0 {
            Growth::TwoEnded
        } else {
            Growth::Bottom
        }
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
        self.segments.iter().flatten().map(B::Buffer::bytes).sum()
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
        let segment = physical / self.segment_chunks;
        let backed = physical < self.total_chunks
            && self
                .segments
                .get(segment as usize)
                .is_some_and(Option::is_some);
        if !backed {
            return Err(Fault::Ceiling {
                what: "committed segments of the state pool",
                need: u64::from(segment) + 1,
                have: u64::from(self.committed_segments()),
            });
        }
        Ok(())
    }

    /// The span's buffer, the pool's last segment, and the byte in it the
    /// span (the pool) ends at; `None` until that segment is backed.
    #[must_use]
    pub fn span_place(&self) -> Option<(&B::Buffer, u64)> {
        let last = self.segments() - 1;
        self.segments[last as usize]
            .as_ref()
            .map(|buffer| (buffer, self.segment_bytes(last)))
    }

    /// The segment buffer virtual byte `at` lies in and its offset there;
    /// refuses an uncommitted one.
    pub fn locate(&self, at: u64) -> Result<(&B::Buffer, u64)> {
        let span = u64::from(self.segment_chunks) * self.chunk_bytes;
        let segment = at / span;
        let buffer = self
            .segments
            .get(usize::try_from(segment).unwrap_or(usize::MAX))
            .and_then(Option::as_ref)
            .ok_or(Fault::Ceiling {
                what: "committed segments of the state pool",
                need: segment + 1,
                have: u64::from(self.committed_segments()),
            })?;
        Ok((buffer, at % span))
    }

    /// The pool as kernels bind it: every committed segment, the rest
    /// absent.
    pub fn bind(&self, handles: &Handles<B>) -> Result<Segments> {
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

    /// Backs the first `segments` segments and every one holding the span,
    /// each zeroed; what is backed already stands. The span's bytes are
    /// entered as the fire's in the device's books.
    pub fn commit_segments(&mut self, device: &B::Context, segments: u32) -> Result<()> {
        use crate::backend::Context as _;
        let max = self.segments();
        if segments > max {
            return Err(Fault::Ceiling {
                what: "segments of the state pool",
                need: u64::from(segments),
                have: u64::from(max),
            });
        }
        for segment in (0..segments).chain(self.top..max) {
            if self.segments[segment as usize].is_some() {
                continue;
            }
            let buffer = B::Buffer::zeroed(
                device,
                self.segment_bytes(segment),
                crate::books::owners::POOL,
            )?;
            self.segments[segment as usize] = Some(buffer);
            if segment == max - 1 && self.span > 0 {
                device.books().relabel(
                    crate::books::owners::POOL,
                    crate::books::owners::SPAN,
                    u64::from(self.span) * self.chunk_bytes,
                );
            }
        }
        Ok(())
    }

    /// Gives back every segment at and past `segments` below the span's.
    /// The caller has drained every frame, and the runtime evacuated their
    /// chunks first.
    pub fn release_segments(&mut self, segments: u32) {
        for segment in segments..self.top {
            self.segments[segment as usize] = None;
        }
    }

    /// Copies each moved chunk onto its new physical chunk in `frame`, then
    /// takes the moves into the table: the caller commits `frame` before any
    /// later fire is encoded, so the copies land first.
    /// Returns each move as `(from, to)` physical chunks, for what lies
    /// beside the pool indexed by physical chunk.
    pub fn evacuate(
        &mut self,
        frame: &mut B::Frame,
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

    /// Writes `from` at virtual byte `at` through the queue, after every
    /// submitted frame.
    pub fn write(&mut self, at: u64, from: &[u8]) -> Result<()> {
        let span = u64::from(self.segment_chunks) * self.chunk_bytes;
        let segment = usize::try_from(at / span).unwrap_or(usize::MAX);
        let committed = self.committed_segments();
        let buffer = self
            .segments
            .get_mut(segment)
            .and_then(Option::as_mut)
            .ok_or(Fault::Ceiling {
                what: "committed segments of the state pool",
                need: segment as u64 + 1,
                have: u64::from(committed),
            })?;
        buffer.write(at % span, from)
    }

    /// Copies `bytes` from virtual byte `from` to `to` in `frame`; neither
    /// run crosses a chunk.
    pub fn copy(&self, frame: &mut B::Frame, from: u64, to: u64, bytes: u64) -> Result<()> {
        let (src, src_at) = self.locate(from)?;
        let (dst, dst_at) = self.locate(to)?;
        frame.copy(src, src_at, dst, dst_at, bytes)
    }

    /// `into.len()` bytes from virtual byte `at`, read on the host.
    pub fn read(&self, at: u64, into: &mut [u8]) -> Result<()> {
        let (buffer, local) = self.locate(at)?;
        buffer.read(local, into)
    }

    /// Zeroes `len` bytes at virtual byte `at`. The caller has drained every
    /// frame that touches them.
    pub fn zero(&mut self, at: u64, len: u64) -> Result<()> {
        let span = u64::from(self.segment_chunks) * self.chunk_bytes;
        let segment = usize::try_from(at / span).unwrap_or(usize::MAX);
        let committed = self.committed_segments();
        let buffer = self
            .segments
            .get_mut(segment)
            .and_then(Option::as_mut)
            .ok_or(Fault::Ceiling {
                what: "committed segments of the state pool",
                need: segment as u64 + 1,
                have: u64::from(committed),
            })?;
        buffer.zero_span(at % span, len)
    }
}

#[cfg(test)]
mod tests {
    use super::segment_chunks;

    /// The last segment holds the span whole, no segment passes a binding,
    /// and the kernels bind four at most.
    #[test]
    fn the_last_segment_holds_the_span() {
        let chunk = 4 << 20;
        for (total, binding, span) in [
            (2256, 4u64 << 30, 40),
            (3016, 4095 << 20, 760),
            (2257, 4 << 30, 563),
            (1001, 4 << 30, 250),
            (64, 1 << 30, 0),
        ] {
            let segment =
                segment_chunks(chunk, total, binding, span).expect("a cut holds the span");
            let count = total.div_ceil(segment);
            assert!(
                count <= super::SEGMENTS as u32,
                "{total} {span}: {count} segments"
            );
            assert!(u64::from(segment) * chunk <= binding);
            assert!(total - (count - 1) * segment >= span, "{total} {span}");
        }
        // Refused, not cut smaller: more segments than bind, or a span past
        // the widest last segment.
        assert_eq!(segment_chunks(chunk, 64, 64 << 20, 16), None);
        assert_eq!(segment_chunks(chunk, 64, 1 << 30, 63), None);
    }
}
