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
    /// The u32 words of one whole segment, as kernels are told it.
    segment_words: u32,
    /// The activation span's chunks, the top ones.
    span: u32,
    /// The first segment holding a chunk of the span (the segment count
    /// where there is none): it and every one above are backed for good.
    top: u32,
    table: ChunkTable,
}

/// The chunks one binding holds, at least one.
fn binding_chunks(chunk_bytes: u64, binding: u64) -> u32 {
    u32::try_from(binding / chunk_bytes.max(1))
        .unwrap_or(u32::MAX)
        .max(1)
}

/// The most segments a pool of `total_chunks` is cut into with its top
/// `span` whole in the last ([`segment_chunks`]).
fn spanned_segments(total_chunks: u32, span: u32) -> u32 {
    if span == 0 {
        SEGMENTS as u32
    } else {
        (total_chunks / (span + SEGMENTS as u32 - 1)).min(SEGMENTS as u32)
    }
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
    let fits = binding_chunks(chunk_bytes, binding);
    let most = spanned_segments(total_chunks, span);
    let least = total_chunks.div_ceil(fits);
    (most > 0 && least <= most).then(|| total_chunks.div_ceil(most))
}

/// A pool's size: the room it is cut from, its chunks, the chunks of each
/// segment, and the chunks the kernels index state in (a whole count of
/// segments' worth).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sized {
    pub room: u64,
    pub total: u32,
    pub segment_chunks: u32,
    pub words: u32,
}

/// The pool `left` bytes hold (or `kv_budget` of state beside the span,
/// within `left`) in chunks of `chunk_bytes`: `least_chunks` of state (one
/// working set) beside a top rung's `act` chunks, bound in at most
/// [`SEGMENTS`] segments of no more than `binding` bytes, with the span in
/// the last.
pub fn sized(
    left: u64,
    kv_budget: Option<u64>,
    chunk_bytes: u64,
    act: u32,
    least_chunks: u32,
    binding: u64,
) -> Result<Sized> {
    let chunk = chunk_bytes;
    let room = kv_budget.map_or(left, |budget| {
        budget.saturating_add(u64::from(act) * chunk).min(left)
    });
    let total = engine::chunk::fit(room, chunk, act, least_chunks)
        .map_err(|short| Fault::Residency(short.to_string()))?;
    let binds =
        u32::try_from((binding / chunk).saturating_mul(SEGMENTS as u64)).unwrap_or(u32::MAX);
    // Whole segments of a quarter of the pool, all four within the
    // bf16-element index the kernels address the pool by.
    let words = u32::try_from(2 * u64::from(u32::MAX) / chunk).unwrap_or(u32::MAX)
        / SEGMENTS as u32
        * SEGMENTS as u32;
    // State lies within that index; the span past it is the fire's
    // alone (bound by offset, never indexed), at every rung.
    let total = total.min(binds).min(words.saturating_add(act));
    // As many segments as hold the span whole in the last, each within one
    // binding: chunks past what those bind are not laid. A span near a
    // binding cuts the pool into few segments, and a pool between whole
    // counts of them binds only the lower count's.
    let total = total.min(
        spanned_segments(total, act).saturating_mul(binding_chunks(chunk, binding)),
    );
    let span_short = || {
        Fault::Residency(format!(
            "one working set takes {least_chunks} chunks of {chunk} bytes beside a fire's \
             {act}, and this device binds four segments of {} MiB",
            binding >> 20
        ))
    };
    let segment_chunks = segment_chunks(chunk, total, binding, act).ok_or_else(span_short)?;
    if total < least_chunks + act {
        return Err(span_short());
    }
    Ok(Sized {
        room,
        total,
        segment_chunks,
        words,
    })
}

/// Why a rung is no rung of this device: one of its reservations (the span,
/// an arm's readout seat, pixel plane or inputs store) passes one binding.
pub const PAST_ONE_BINDING: &str = "past one binding";

/// Why a rung is no rung of this device: its span and one working set at
/// the context lie in no pool the device's room leaves, cut into segments
/// with the span whole in the last.
pub const PAST_THE_POOL: &str = "its span and one working set past the pool the room leaves";

/// A top rung as the pool weighs it: its fires' `tokens`, its span's `act`
/// chunks, the bytes the device leaves a pool of a load cut to it, and the
/// widest reservation one of its fires binds whole.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rung {
    pub tokens: u32,
    pub act: u32,
    pub left: u64,
    pub widest: u64,
}

/// Why `rung` cannot be a load's top: a reservation past one `binding`
/// ([`PAST_ONE_BINDING`]), or its span and one working set of
/// `least_chunks` past the pool its room gives (or `kv_budget` of state
/// beside the span), cut into segments of at most `binding` bytes with the
/// span whole in the last ([`sized`], [`PAST_THE_POOL`]). `None`: it can.
#[must_use]
pub fn infeasible(
    rung: &Rung,
    kv_budget: Option<u64>,
    chunk_bytes: u64,
    least_chunks: u32,
    binding: u64,
) -> Option<&'static str> {
    if rung.widest > binding {
        Some(PAST_ONE_BINDING)
    } else if sized(rung.left, kv_budget, chunk_bytes, rung.act, least_chunks, binding).is_err() {
        Some(PAST_THE_POOL)
    } else {
        None
    }
}

/// The widest of `rungs` that can be a load's top ([`infeasible`]).
/// `None`: none can.
#[must_use]
pub fn widest_rung(
    rungs: &[Rung],
    kv_budget: Option<u64>,
    chunk_bytes: u64,
    least_chunks: u32,
    binding: u64,
) -> Option<u32> {
    rungs
        .iter()
        .filter(|rung| infeasible(rung, kv_budget, chunk_bytes, least_chunks, binding).is_none())
        .map(|rung| rung.tokens)
        .max()
}

/// The u32 words of a segment of `segment_chunks` chunks of `chunk_bytes`,
/// which every kernel reading the pool is told: a segment too long for the
/// count is refused at the load, never stated short.
fn segment_words(chunk_bytes: u64, segment_chunks: u32) -> Result<u32> {
    let words = u64::from(segment_chunks) * chunk_bytes / 4;
    u32::try_from(words).map_err(|_| Fault::Ceiling {
        what: "words of one segment of the state pool (a kernel is told a segment in u32 words)",
        need: words,
        have: u64::from(u32::MAX),
    })
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
            segment_words: segment_words(chunk_bytes, segment_chunks)?,
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

    /// The buffers of the segments backed, in order.
    #[must_use]
    pub fn segment_buffers(&self) -> Vec<&B::Buffer> {
        self.segments.iter().flatten().collect()
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
        if !B::RESIZES_POOL {
            Growth::Fixed
        } else if self.span > 0 {
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
        Ok(Segments {
            bufs,
            words: self.segment_words,
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

    /// Records each moved chunk's copy onto its new physical chunk in
    /// `frame`, then takes the moves into the table and hands back the
    /// table as it stood. Nothing is committed: the caller resolves what
    /// the moves leave to repoint against the new table, then commits
    /// `frame` before any later fire is encoded (the copies land first), or
    /// puts the old table back ([`Pool::restore`]) and drops `frame`, its
    /// copies unsubmitted. Refused with the table as it stood.
    pub fn evacuate(&mut self, frame: &mut B::Frame, evacuation: &Evacuation) -> Result<ChunkTable> {
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
            self.copy(
                frame,
                u64::from(from) * self.chunk_bytes,
                u64::from(physical) * self.chunk_bytes,
                self.chunk_bytes,
            )?;
        }
        Ok(std::mem::replace(&mut self.table, table))
    }

    /// Puts back the table an evacuation replaced ([`Pool::evacuate`]),
    /// its copies dropped unsubmitted.
    pub fn restore(&mut self, table: ChunkTable) {
        self.table = table;
    }

    /// The chunk table as it stands.
    #[must_use]
    pub fn table(&self) -> &ChunkTable {
        &self.table
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

    /// Zeroes `len` bytes at virtual byte `at` in `frame`; the run crosses
    /// no segment.
    pub fn zero(&self, frame: &mut B::Frame, at: u64, len: u64) -> Result<()> {
        let (buffer, local) = self.locate(at)?;
        frame.fill_zero(buffer, local, len)
    }
}

#[cfg(test)]
mod tests {
    use super::{PAST_ONE_BINDING, PAST_THE_POOL, Rung, segment_chunks, segment_words};
    use crate::error::Fault;

    /// A segment's length in words is stated whole or the load is refused:
    /// 16 GiB is one word past what a u32 counts.
    #[test]
    fn a_segment_past_u32_words_is_refused() {
        let chunk = 4 << 20;
        assert_eq!(segment_words(chunk, 1).ok(), Some(1 << 20));
        assert_eq!(segment_words(chunk, 4095).ok(), Some(4095 << 20));
        let refused = segment_words(chunk, 4096).expect_err("2^32 words");
        assert!(
            matches!(
                refused,
                Fault::Ceiling { need, have, .. } if need == 1 << 32 && have == u64::from(u32::MAX)
            ),
            "{refused}"
        );
    }

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

    /// Qwen3-0.6B on an Adreno phone: 3803136-byte chunks, a 1 GiB largest
    /// buffer (282 chunks), one working set of 129, and the room its
    /// capacity leaves past the weights about 400 chunks. A pool of one
    /// binding holds no top over 153 chunks beside the working set, and two
    /// segments each holding the span need it at most 197 (400 / 2 - 3): the
    /// 10240 and 4096 rungs (266, 230) fit neither, and the top is the widest
    /// below them. With the room of a desktop card every rung fits.
    #[test]
    fn the_top_rung_is_the_widest_whose_span_and_working_set_fit_the_pool() {
        const GIB: u64 = 1 << 30;
        let (chunk, binding, least) = (3_803_136u64, GIB, 129u32);
        let left = 400 * chunk;
        let at = |left: u64| {
            [(1024, 120), (2048, 180), (4096, 230), (10240, 266)].map(|(tokens, act)| Rung {
                tokens,
                act,
                left,
                widest: u64::from(act) * chunk,
            })
        };
        let rungs = at(left);
        assert_eq!(super::widest_rung(&rungs, None, chunk, least, binding), Some(2048));
        for rung in &rungs[2..] {
            assert_eq!(
                super::infeasible(rung, None, chunk, least, binding),
                Some(PAST_THE_POOL),
                "the {} rung's span of {} chunks fits no pool of this room",
                rung.tokens,
                rung.act
            );
        }
        // Short of even the least rung's: no top.
        assert_eq!(super::widest_rung(&at(200 * chunk), None, chunk, least, binding), None);
        // A desktop card's room holds the widest.
        assert_eq!(super::widest_rung(&at(8000 * chunk), None, chunk, least, binding), Some(10240));
        // A reservation past one binding is no rung, whatever the room.
        let mut seat = at(8000 * chunk);
        seat[3].widest = binding + 1;
        assert_eq!(super::infeasible(&seat[3], None, chunk, least, binding), Some(PAST_ONE_BINDING));
        assert_eq!(super::widest_rung(&seat, None, chunk, least, binding), Some(4096));
    }

    /// Qwen3.5-0.8B on wgpu at a 12 GiB budget: 2304000-byte chunks, a
    /// binding of 2047 MiB (931 chunks), a top rung of 922 chunks and one
    /// working set of 455. Every segment must hold the span, so a pool of
    /// 3500 chunks binds three segments (2793 chunks), not four: it lays
    /// those, where it was refused for the four its chunks would take.
    #[test]
    fn a_pool_lays_the_chunks_its_segments_bind() {
        const MIB: u64 = 1 << 20;
        let (chunk, binding, act, least) = (2_304_000u64, 2047 * MIB, 922u32, 455u32);
        assert_eq!(super::segment_chunks(chunk, 3500, binding, act), None);
        let pool = super::sized(3500 * chunk, None, chunk, act, least, binding)
            .expect("the chunks three segments bind");
        assert_eq!(pool.total, 3 * 931);
        assert!(u64::from(pool.segment_chunks) * chunk <= binding);
        assert!(pool.total - 2 * pool.segment_chunks >= act, "the span lies in the last");
        // A pool short of one segment holding the span is still refused.
        assert!(super::sized(900 * chunk, None, chunk, act, least, binding).is_err());
    }

    /// Qwen3-8B on a 32 GiB card binding 4095 MiB (14.5 GiB of weights):
    /// its widest rungs' spans pass one binding, so the top is the widest
    /// that lies in one, and the load boots; at the top asked it is refused.
    #[test]
    fn a_top_rung_past_one_binding_is_no_rung() {
        const MIB: u64 = 1 << 20;
        let chunk: u64 = 2_425_856;
        let binding = 4095 * MIB;
        // What the fit leaves the pool beside the weights, and one working
        // set at the declared context.
        let (left, least) = (18_390_605_824, 257);
        // `(rung tokens, span bytes)` as the load lays them, asked 10240.
        let spans: [(u32, u64); 15] = [
            (1, 9_880_832),
            (2, 19_728_384),
            (4, 39_390_720),
            (8, 78_715_392),
            (16, 157_364_736),
            (32, 314_663_424),
            (64, 629_260_800),
            (128, 1_258_455_552),
            (256, 2_516_845_056),
            (512, 5_033_624_064),
            (1024, 5_235_343_872),
            (2048, 5_638_783_488),
            (4096, 5_978_915_328),
            (8192, 6_348_014_080),
            (10240, 6_532_563_456),
        ];
        let act = |top: u32| {
            let bytes = spans.iter().find(|&&(tokens, _)| tokens == top).unwrap().1;
            u32::try_from(bytes.div_ceil(chunk)).unwrap()
        };
        let top = crate::arena::bound_top(&spans, binding).expect("a rung lies in one binding");
        assert_eq!(top, 256);
        let pool = super::sized(left, None, chunk, act(top), least, binding)
            .expect("the bounded load boots");
        assert!(u64::from(pool.segment_chunks) * chunk <= binding);
        assert!(pool.total >= least + act(top));
        // The top asked is refused as the load refused it.
        let refused = super::sized(left, None, chunk, act(10240), least, binding)
            .expect_err("a span past one binding");
        assert_eq!(
            refused.to_string(),
            Fault::Residency(
                "one working set takes 257 chunks of 2425856 bytes beside a fire's 2693, and \
                 this device binds four segments of 4095 MiB"
                    .to_string()
            )
            .to_string()
        );
        // No rung at all lies in a binding narrower than the least span.
        assert_eq!(crate::arena::bound_top(&spans, 8 * MIB), None);
    }
}
