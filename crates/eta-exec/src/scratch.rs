use super::extent::ValueDesc;

mod lifetime;

pub use lifetime::{
    Launch, Launches, describe_launched, lay_out, lay_out_chunked, read_only_unlaunched,
    region_rows, value_lifetimes,
};

pub const ALIGN: u64 = 256;

/// A lane's first bytes, no value's: where a value that takes no bytes is
/// stated to lie.
pub const DUMMY_BYTES: u64 = ALIGN;

/// What an engine built to fill test slots writes over [`DUMMY_BYTES`]
/// before a launch,
/// so a kernel that reads a value stated to take no bytes answers wrongly
/// where a test can see it: one byte repeated, which a device's byte fill
/// writes; a float above any logit, which decides a maximum, an argmax or a
/// sum that reads it; an index past any table; a state no draw starts from.
pub const POISON: u32 = 0x7f7f_7f7f;

pub const MAX_BYTES: u64 = (4 << 30) - ALIGN;

const TEMPORARIES_PER_ELEMENT: u64 = 4;

/// The WGSL runtime's one-row select keeps its state just below the
/// temporary area, out of reach of the sort ladder that reuses all of it.
pub const SELECT_RESERVE: u64 = eta_compiler::codegen::wgsl::SELECT_WORDS as u64 * 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TooLarge {
    Bound {
        bytes: u64,
        limit: u64,
    },

    Overflow,

    /// An item larger than a piece: no chunk holds it whole, and no value
    /// crosses a chunk ([`lay_out_chunked`]).
    PastChunk {
        item: Item,
        bytes: u64,
        chunk: u64,
    },
}

/// What a scratch layout places: a value of the stage, or the temporary
/// area with the reserve below it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Item {
    Value(u32),
    Temporary,
}

impl core::fmt::Display for TooLarge {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Bound { bytes, limit } => {
                write!(
                    f,
                    "a scratch of {bytes} bytes, past the {limit} a lane addresses"
                )
            }
            Self::Overflow => write!(f, "a scratch past a u64"),
            Self::PastChunk {
                item: Item::Value(value),
                bytes,
                chunk,
            } => write!(
                f,
                "value {value} takes {bytes} bytes, past a {chunk}-byte chunk, and no value \
                 crosses a chunk"
            ),
            Self::PastChunk {
                item: Item::Temporary,
                bytes,
                chunk,
            } => write!(
                f,
                "the temporary area and the reserve below it take {bytes} bytes, past a \
                 {chunk}-byte chunk, and they lie in one"
            ),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub values: Vec<u64>,

    pub temporary: u64,

    pub temporary_bytes: u64,

    pub total: u64,
}

fn align_up(value: u64) -> Option<u64> {
    value.checked_next_multiple_of(ALIGN)
}

pub fn layout(descriptors: &[ValueDesc]) -> Result<Layout, TooLarge> {
    let mut values = Vec::with_capacity(descriptors.len());
    let mut at = DUMMY_BYTES;

    for descriptor in descriptors {
        at = align_up(at).ok_or(TooLarge::Overflow)?;
        values.push(at);
        let span = align_up(descriptor.device_bytes()).ok_or(TooLarge::Overflow)?;
        at = at.checked_add(span).ok_or(TooLarge::Overflow)?;
    }

    let temporary = at
        .checked_add(SELECT_RESERVE)
        .and_then(align_up)
        .ok_or(TooLarge::Overflow)?;
    let temporary_bytes = whole_value_temporaries(descriptors)
        .and_then(align_up)
        .ok_or(TooLarge::Overflow)?;
    let total = temporary
        .checked_add(temporary_bytes)
        .ok_or(TooLarge::Overflow)?;

    if total > MAX_BYTES {
        return Err(TooLarge::Bound {
            bytes: total,
            limit: MAX_BYTES,
        });
    }
    Ok(Layout {
        values,
        temporary,
        temporary_bytes,
        total,
    })
}

/// The temporaries a launch that sorts or reduces a whole value takes:
/// [`TEMPORARIES_PER_ELEMENT`] words an element of the widest of
/// `descriptors`. `None` past a `u64`.
#[must_use]
pub fn whole_value_temporaries(descriptors: &[ValueDesc]) -> Option<u64> {
    descriptors
        .iter()
        .map(|descriptor| u64::from(descriptor.len))
        .fold(1, u64::max)
        .checked_mul(size_of::<u32>() as u64)
        .and_then(|bytes| bytes.checked_mul(TEMPORARIES_PER_ELEMENT))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lifetime {
    pub def: u32,
    pub last: u32,
    pub dead: bool,
    pub reusable: bool,
    pub launch_def: u32,
    pub launch_last: u32,
    pub class_def: u64,
    pub class_last: u64,
}

impl Lifetime {
    pub const SEQUENTIAL: u64 = u64::MAX;
}

/// Where an item lies in a chunked scratch: byte `offset` of piece
/// `piece`. An engine addresses it as the piece's base (its chunk's first
/// byte, from the instance's base table) plus `offset`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Placed {
    pub piece: u32,
    pub offset: u64,
}

/// A stage's scratch cut into pieces of at most a chunk
/// ([`lay_out_chunked`]): no item crosses a piece, and two values whose
/// lifetimes overlap share no bytes. Piece 0 opens with the
/// [`DUMMY_BYTES`] a value that takes no bytes is stated to lie at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkLayout {
    /// Each value's place; a dead value's is piece 0, byte 0.
    pub values: Vec<Placed>,
    /// The temporary area's first byte: the reserve lies right below it,
    /// in the same piece.
    pub temporary: Placed,
    /// The bytes the temporary area holds.
    pub temporary_bytes: u64,
    /// Each piece's bytes in use, from its first: none past the chunk.
    pub ends: Vec<u64>,
}

impl ChunkLayout {
    /// The pieces the scratch takes: chunks, or one run where it fits one.
    #[must_use]
    pub fn pieces(&self) -> u32 {
        u32::try_from(self.ends.len()).unwrap_or(u32::MAX)
    }

    /// The bytes an instance holds for it in chunks of `chunk` bytes: the
    /// one piece's bytes where it takes one (a run inside a chunk), else
    /// whole chunks, one a piece.
    #[must_use]
    pub fn held_bytes(&self, chunk: u64) -> u64 {
        match self.ends[..] {
            [only] => only,
            _ => u64::from(self.pieces()).saturating_mul(chunk),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Vacant {
    piece: u32,
    offset: u64,
    span: u64,
    launch: u32,
    class: u64,
    /// One dead value's own bytes, from its offset: neither cut nor joined
    /// to another's.
    whole: bool,
}

/// Bytes for a value written in `launch` of rows `class`: a vacancy a
/// launch before it, or one its own launch left of the same class. Within a
/// launch across blocks, block `b` owns row `b` of every value of a class,
/// so a value takes there only a dead one's own bytes, at its offset: any
/// other place would lay a block's row over another block's, which may
/// still be reading it.
fn take(free: &mut Vec<Vacant>, span: u64, launch: u32, class: u64) -> Option<Placed> {
    let best = free
        .iter()
        .enumerate()
        .filter(|(_, block)| {
            block.span >= span
                && (block.launch < launch
                    || (class != 0
                        && block.class == class
                        && (class == Lifetime::SEQUENTIAL || block.whole)))
        })
        .min_by_key(|(_, block)| block.span)
        .map(|(index, _)| index)?;
    let Vacant {
        piece,
        offset,
        span: width,
        ..
    } = free[best];
    if width == span {
        free.remove(best);
    } else {
        free[best].offset = offset + span;
        free[best].span = width - span;
        free[best].whole = false;
    }
    Some(Placed { piece, offset })
}

fn release(free: &mut Vec<Vacant>, vacant: Vacant) {
    let at =
        free.partition_point(|block| (block.piece, block.offset) < (vacant.piece, vacant.offset));
    free.insert(at, vacant);
    let joins = |a: &Vacant, b: &Vacant| {
        a.piece == b.piece
            && a.offset + a.span == b.offset
            && a.launch == b.launch
            && a.class == b.class
    };
    if at + 1 < free.len() && joins(&free[at], &free[at + 1]) {
        free[at].span += free[at + 1].span;
        free[at].whole = false;
        free.remove(at + 1);
    }
    if at > 0 && joins(&free[at - 1], &free[at]) {
        free[at - 1].span += free[at].span;
        free[at - 1].whole = false;
        free.remove(at);
    }
}

/// The first piece whose free tail holds `span` bytes from an aligned
/// byte `lead` bytes past its end, opening a new one where none does:
/// `Some(None)` where a fresh piece holds no such span either, `None` past
/// a `u64`.
fn bump(ends: &mut Vec<u64>, lead: u64, span: u64, limit: u64) -> Option<Option<(u32, u64)>> {
    let place = |end: u64| -> Option<Option<u64>> {
        let at = align_up(end.checked_add(lead)?)?;
        Some(
            at.checked_add(span)
                .filter(|&stop| stop <= limit)
                .map(|_| at),
        )
    };
    for (piece, end) in ends.iter_mut().enumerate() {
        if let Some(at) = place(*end)? {
            *end = at + span;
            return Some(Some((u32::try_from(piece).ok()?, at)));
        }
    }
    let Some(at) = place(0)? else {
        return Some(None);
    };
    ends.push(at + span);
    Some(Some((u32::try_from(ends.len() - 1).ok()?, at)))
}

/// `descriptors` laid by `lifetimes` in pieces of at most `limit` bytes: a
/// value takes the bytes of one whose last reader ran before it is
/// written, in that one's piece, else a free tail of the first piece with
/// room, else a new piece. `reserve` bytes lie right below the temporary
/// area, which holds `temporary` bytes, both in one piece. An item past
/// `limit` is refused by name ([`TooLarge::PastChunk`]).
fn place(
    descriptors: &[ValueDesc],
    lifetimes: &[Lifetime],
    reserve: u64,
    temporary: u64,
    limit: u64,
) -> Result<ChunkLayout, TooLarge> {
    let count = descriptors.len();
    let mut by_def: Vec<usize> = (0..count).collect();
    by_def.sort_by_key(|&i| {
        (
            lifetimes[i].def,
            core::cmp::Reverse(descriptors[i].device_bytes()),
            i,
        )
    });
    let mut by_last: Vec<usize> = (0..count).collect();
    by_last.sort_by_key(|&i| (lifetimes[i].last, i));
    let mut expired = 0usize;

    let mut free: Vec<Vacant> = Vec::new();
    let mut values = vec![Placed::default(); count];
    let mut spans = vec![0u64; count];
    let mut ends = vec![DUMMY_BYTES];
    let mut step: Option<u32> = None;

    for &i in &by_def {
        let life = lifetimes[i];
        if step != Some(life.def) {
            while expired < count && lifetimes[by_last[expired]].last < life.def {
                let dead = by_last[expired];
                expired += 1;
                if spans[dead] > 0 {
                    release(
                        &mut free,
                        Vacant {
                            piece: values[dead].piece,
                            offset: values[dead].offset,
                            span: spans[dead],
                            launch: lifetimes[dead].launch_last,
                            class: lifetimes[dead].class_last,
                            whole: true,
                        },
                    );
                }
            }
            step = Some(life.def);
        }
        if life.dead {
            continue;
        }
        let bytes = descriptors[i].device_bytes();
        let span = align_up(bytes).ok_or(TooLarge::Overflow)?;
        let past = TooLarge::PastChunk {
            item: Item::Value(u32::try_from(i).unwrap_or(u32::MAX)),
            bytes,
            chunk: limit,
        };
        if span > limit {
            return Err(past);
        }
        let taken = if life.reusable {
            take(&mut free, span, life.launch_def, life.class_def)
        } else {
            None
        };
        values[i] = match taken {
            Some(placed) => placed,
            None => {
                let (piece, offset) = bump(&mut ends, 0, span, limit)
                    .ok_or(TooLarge::Overflow)?
                    .ok_or(past)?;
                Placed { piece, offset }
            }
        };
        spans[i] = span;
    }

    let temporary_bytes = align_up(temporary).ok_or(TooLarge::Overflow)?;
    let (piece, offset) = bump(&mut ends, reserve, temporary_bytes, limit)
        .ok_or(TooLarge::Overflow)?
        .ok_or(TooLarge::PastChunk {
            item: Item::Temporary,
            bytes: reserve.saturating_add(temporary_bytes),
            chunk: limit,
        })?;
    Ok(ChunkLayout {
        values,
        temporary: Placed { piece, offset },
        temporary_bytes,
        ends,
    })
}

/// Every value live from the first launch to the last, none taking
/// another's bytes: what a layout falls back to where its lifetimes are
/// not stated.
fn each_its_own(count: usize) -> Vec<Lifetime> {
    vec![
        Lifetime {
            def: 0,
            last: u32::MAX,
            dead: false,
            reusable: false,
            launch_def: 0,
            launch_last: u32::MAX,
            class_def: Lifetime::SEQUENTIAL,
            class_last: Lifetime::SEQUENTIAL,
        };
        count
    ]
}

/// `descriptors` laid by `lifetimes` in chunks of `chunk` bytes, the
/// layout every backend's guest scratch takes ([`ChunkLayout`]): no value
/// crosses a chunk, values packed into chunk-sized pieces, each addressed
/// as its piece's base plus its offset. A value (or the temporary area
/// with the reserve below it) larger than a chunk is refused by name. Where
/// `lifetimes` are not the values', each value holds its own bytes.
pub fn layout_chunked(
    descriptors: &[ValueDesc],
    lifetimes: &[Lifetime],
    reserve: u64,
    temporary: u64,
    chunk: u64,
) -> Result<ChunkLayout, TooLarge> {
    let own;
    let lifetimes = if lifetimes.len() == descriptors.len() {
        lifetimes
    } else {
        own = each_its_own(descriptors.len());
        &own
    };
    place(
        descriptors,
        lifetimes,
        reserve,
        temporary,
        chunk.min(MAX_BYTES),
    )
}

/// `descriptors` laid by `lifetimes`: a value takes the bytes of one whose
/// last reader ran before it is written. `reserve` bytes lie clear below
/// the temporary area, which holds `temporary` bytes: what the kernels keep
/// there, as the engine that runs them states it. One piece, unbounded but
/// for [`MAX_BYTES`]: [`layout_chunked`] with no chunk.
pub fn layout_reusing(
    descriptors: &[ValueDesc],
    lifetimes: &[Lifetime],
    reserve: u64,
    temporary: u64,
) -> Result<Layout, TooLarge> {
    if lifetimes.len() != descriptors.len() {
        return layout(descriptors);
    }
    let laid = place(descriptors, lifetimes, reserve, temporary, u64::MAX)?;
    let total = laid
        .temporary
        .offset
        .checked_add(laid.temporary_bytes)
        .ok_or(TooLarge::Overflow)?;
    if total > MAX_BYTES {
        return Err(TooLarge::Bound {
            bytes: total,
            limit: MAX_BYTES,
        });
    }
    Ok(Layout {
        values: laid.values.iter().map(|placed| placed.offset).collect(),
        temporary: laid.temporary.offset,
        temporary_bytes: laid.temporary_bytes,
        total,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use eta_ir::types::MAX_RANK;

    fn desc(len: u32) -> ValueDesc {
        ValueDesc {
            len,
            rows: 1,
            last: len,
            rank: 1,
            dtype: 0,
            dims: [0; MAX_RANK],
        }
    }

    fn life(def: u32, last: u32) -> Lifetime {
        Lifetime {
            dead: false,
            def,
            last,
            reusable: true,
            launch_def: def,
            launch_last: last,
            class_def: Lifetime::SEQUENTIAL,
            class_last: Lifetime::SEQUENTIAL,
        }
    }

    fn rowed(def: u32, last: u32, launch: u32, class: u64) -> Lifetime {
        Lifetime {
            dead: false,
            def,
            last,
            reusable: true,
            launch_def: launch,
            launch_last: launch,
            class_def: class,
            class_last: class,
        }
    }

    #[test]
    fn scratch_every_case() {
        a_value_dead_before_the_next_region_hands_its_slot_on();
        a_value_last_read_where_another_is_defined_does_not_share();
        within_a_many_block_launch_only_one_class_shares();
        a_result_that_may_go_unwritten_keeps_a_fresh_slot();
        a_wider_taker_gets_a_coalesced_pair_of_slots();
        the_temporary_area_holds_what_is_stated();
        a_reserve_lies_below_the_temporary_arena();
        mismatched_lifetimes_fall_back_to_the_naive_layout();
        a_dead_value_takes_no_slot();
        rows_of_one_class_share_only_at_one_offset();
    }

    /// Within a launch across blocks, block `b` owns row `b` of every value
    /// of a row class: a value that takes another's bytes there lies at the
    /// other's offset, so no block's row lands on another block's. Values
    /// that share across launches may lie anywhere over each other.
    fn rows_of_one_class_share_only_at_one_offset() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = |bound: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % bound
        };
        for _ in 0..20_000 {
            let count = 2 + next(10) as usize;
            let mut descriptors = Vec::new();
            let mut lifetimes = Vec::new();
            for _ in 0..count {
                let rows = 1 + next(4) as u32;
                let width = [64u32, 64, 64, 96][next(4) as usize];
                descriptors.push(ValueDesc {
                    len: rows * width,
                    rows,
                    last: width,
                    rank: 2,
                    dtype: 0,
                    dims: [0; MAX_RANK],
                });
                let def = next(8) as u32;
                let last = def + next(4) as u32;
                // Steps 0..3 one launch across blocks, 4.. another.
                let launch = |step: u32| u32::from(step >= 4);
                let class = if next(5) == 0 { 0 } else { (1u64 << 40) | u64::from(width) * 4 };
                lifetimes.push(Lifetime {
                    dead: false,
                    def,
                    last,
                    reusable: next(6) != 0,
                    launch_def: launch(def),
                    launch_last: launch(last),
                    class_def: class,
                    class_last: class,
                });
            }
            let laid = layout_reusing(&descriptors, &lifetimes, 0, 0).unwrap();
            let span = |i: usize| align_up(descriptors[i].device_bytes()).unwrap();
            for i in 0..count {
                for j in 0..count {
                    let (a, b) = (lifetimes[i], lifetimes[j]);
                    let (oi, oj) = (laid.values[i], laid.values[j]);
                    let overlap = oi < oj + span(j) && oj < oi + span(i);
                    if i == j || !overlap {
                        continue;
                    }
                    assert!(a.last < b.def || b.last < a.def, "{i} and {j} live together");
                    let (early, late) = if a.last < b.def { (a, b) } else { (b, a) };
                    if early.launch_last >= late.launch_def {
                        assert_eq!(oi, oj, "{i} and {j} share a launch's rows at two offsets");
                    }
                }
            }
        }
    }

    /// A chunk of 64 KiB: a few [`desc`]`(4096)` values a piece.
    const CHUNK: u64 = 64 << 10;

    /// Every piece holds its items whole within `chunk`, none in piece 0's
    /// dummy bytes, and two values share a byte only where one is last read
    /// before the other is written; the temporary area and the reserve
    /// below it share none with a value.
    fn holds(
        laid: &ChunkLayout,
        descriptors: &[ValueDesc],
        lifetimes: &[Lifetime],
        reserve: u64,
        chunk: u64,
    ) {
        assert_eq!(laid.values.len(), descriptors.len());
        assert!(!laid.ends.is_empty());
        assert!(laid.ends.iter().all(|&end| end <= chunk), "{:?}", laid.ends);
        let span = |i: usize| {
            let at = laid.values[i];
            (
                at.piece,
                at.offset,
                at.offset + descriptors[i].device_bytes(),
            )
        };
        let live: Vec<usize> = (0..descriptors.len())
            .filter(|&i| !lifetimes[i].dead)
            .collect();
        for &i in &live {
            let (piece, start, end) = span(i);
            assert_eq!(start % ALIGN, 0);
            let top = laid.ends[piece as usize];
            assert!(
                end <= top,
                "value {i} ends at {end}, past its piece's {top}"
            );
            if piece == 0 {
                assert!(start >= DUMMY_BYTES, "value {i} in the dummy bytes");
            }
        }
        for (x, &a) in live.iter().enumerate() {
            for &b in &live[x + 1..] {
                let (pa, sa, ea) = span(a);
                let (pb, sb, eb) = span(b);
                if pa != pb || ea <= sb || eb <= sa {
                    continue;
                }
                let (la, lb) = (lifetimes[a], lifetimes[b]);
                assert!(
                    la.last < lb.def || lb.last < la.def,
                    "values {a} {la:?} and {b} {lb:?} share bytes and their lifetimes overlap"
                );
            }
        }
        let temporary = laid.temporary;
        assert!(temporary.offset >= reserve);
        let (low, high) = (
            temporary.offset - reserve,
            temporary.offset + laid.temporary_bytes,
        );
        assert!(high <= laid.ends[temporary.piece as usize]);
        if temporary.piece == 0 {
            assert!(low >= DUMMY_BYTES);
        }
        for &i in &live {
            let (piece, start, end) = span(i);
            assert!(
                piece != temporary.piece || end <= low || high <= start,
                "value {i} lies in the temporary area"
            );
        }
    }

    #[test]
    fn chunked_every_case() {
        no_value_crosses_a_piece();
        overlapping_lifetimes_take_disjoint_bytes();
        a_value_past_a_chunk_is_refused_by_name();
        the_temporary_area_past_a_chunk_is_refused_by_name();
        one_unbounded_piece_is_the_reusing_layout();
        an_instance_holds_one_run_or_whole_chunks();
        a_freed_slot_is_taken_in_its_own_piece();
    }

    fn no_value_crosses_a_piece() {
        // Eight values of 16 KiB alive together: three in the first 64 KiB
        // piece beside the dummy bytes, four in the next, which has none,
        // and one in a third.
        let descriptors = [desc(4096); 8];
        let lifetimes = [life(0, 7); 8];
        let laid = layout_chunked(&descriptors, &lifetimes, 0, 0, CHUNK).expect("fits");
        holds(&laid, &descriptors, &lifetimes, 0, CHUNK);
        assert_eq!(laid.pieces(), 3);
        let pieces: Vec<u32> = laid.values.iter().map(|placed| placed.piece).collect();
        assert_eq!(pieces, [0, 0, 0, 1, 1, 1, 1, 2], "{laid:?}");
        assert_eq!(
            laid.values[3].offset, 0,
            "a piece past the first has no dummy bytes"
        );
        assert_eq!(laid.ends, [DUMMY_BYTES + 3 * 16384, CHUNK, 16384]);
    }

    fn overlapping_lifetimes_take_disjoint_bytes() {
        let descriptors = [desc(4096), desc(4096), desc(4096), desc(8192)];
        let lifetimes = [life(0, 1), life(1, 2), life(2, 3), life(3, 3)];
        let laid = layout_chunked(&descriptors, &lifetimes, 0, 0, CHUNK).expect("fits");
        holds(&laid, &descriptors, &lifetimes, 0, CHUNK);
        assert_eq!(
            laid.values[2], laid.values[0],
            "the first was read last a step before"
        );
        assert_ne!(laid.values[1], laid.values[0]);
    }

    fn a_value_past_a_chunk_is_refused_by_name() {
        let descriptors = [desc(1024), desc(16385)];
        assert_eq!(
            layout_chunked(&descriptors, &[life(0, 1), life(1, 1)], 0, 0, CHUNK),
            Err(TooLarge::PastChunk {
                item: Item::Value(1),
                bytes: 16385 * 4,
                chunk: CHUNK,
            })
        );
        let refused = layout_chunked(&descriptors, &[], 0, 0, CHUNK).unwrap_err();
        assert!(
            refused
                .to_string()
                .starts_with("value 1 takes 65540 bytes, past a 65536-byte chunk"),
            "{refused}"
        );
        // Exactly a chunk fits a piece of its own.
        let whole = [desc(1024), desc(16384)];
        let laid = layout_chunked(&whole, &[life(0, 1), life(1, 1)], 0, 0, CHUNK).expect("fits");
        assert_eq!(
            laid.values[1],
            Placed {
                piece: 1,
                offset: 0
            }
        );
        assert_eq!(laid.ends[1], CHUNK);
    }

    fn the_temporary_area_past_a_chunk_is_refused_by_name() {
        let descriptors = [desc(1024)];
        assert_eq!(
            layout_chunked(&descriptors, &[life(0, 0)], 4096, CHUNK - 4096 + 1, CHUNK),
            Err(TooLarge::PastChunk {
                item: Item::Temporary,
                bytes: 4096 + align_up(CHUNK - 4096 + 1).unwrap(),
                chunk: CHUNK,
            })
        );
        let laid = layout_chunked(&descriptors, &[life(0, 0)], 4096, CHUNK - 4096, CHUNK)
            .expect("the reserve and the area fill a piece of their own");
        assert_eq!(
            laid.temporary,
            Placed {
                piece: 1,
                offset: 4096
            }
        );
        holds(&laid, &descriptors, &[life(0, 0)], 4096, CHUNK);
    }

    fn one_unbounded_piece_is_the_reusing_layout() {
        let descriptors = [desc(1024), desc(1024), desc(2048), desc(512)];
        let lifetimes = [life(0, 0), life(0, 0), life(1, 2), life(2, 2)];
        let laid =
            layout_chunked(&descriptors, &lifetimes, SELECT_RESERVE, 777, MAX_BYTES).expect("fits");
        let reused = layout_reusing(&descriptors, &lifetimes, SELECT_RESERVE, 777).expect("fits");
        assert_eq!(laid.pieces(), 1);
        assert_eq!(
            laid.values
                .iter()
                .map(|placed| placed.offset)
                .collect::<Vec<_>>(),
            reused.values
        );
        assert_eq!(laid.temporary.offset, reused.temporary);
        assert_eq!(laid.ends[0], reused.total);
        assert_eq!(laid.held_bytes(CHUNK), reused.total);
    }

    fn an_instance_holds_one_run_or_whole_chunks() {
        let one = layout_chunked(&[desc(1024)], &[life(0, 0)], 0, 0, CHUNK).expect("fits");
        assert_eq!(one.held_bytes(CHUNK), DUMMY_BYTES + 4096, "the tighter run");
        let three = layout_chunked(&[desc(4096); 8], &[life(0, 0); 8], 0, 0, CHUNK).expect("fits");
        assert_eq!(
            three.held_bytes(CHUNK),
            3 * CHUNK,
            "whole chunks, one a piece"
        );
    }

    fn a_freed_slot_is_taken_in_its_own_piece() {
        // Piece 0 full of values alive to the end but the first; piece 1
        // opened; the first's bytes freed and taken back in piece 0.
        let descriptors = [desc(4096), desc(4096), desc(4096), desc(4096), desc(4096)];
        let lifetimes = [life(0, 0), life(0, 3), life(0, 3), life(0, 3), life(1, 3)];
        let laid = layout_chunked(&descriptors, &lifetimes, 0, 0, CHUNK).expect("fits");
        holds(&laid, &descriptors, &lifetimes, 0, CHUNK);
        assert_eq!(laid.values[3].piece, 1);
        assert_eq!(laid.values[4], laid.values[0]);
        assert_eq!(laid.pieces(), 2);
    }

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n.max(1)
        }
    }

    /// Random stages: widths across and past a chunk, lifetimes over a few
    /// launches of one or many blocks, results that may go unwritten, dead
    /// values, reserves and temporary areas. Each laid one holds
    /// ([`holds`]), or is refused naming an item that is past the chunk.
    #[test]
    fn chunked_adversarial_random_stages() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let (mut laid_count, mut refused_count, mut many) = (0, 0, 0);
        for _ in 0..4000 {
            let chunk = (1 + rng.below(64)) * 1024;
            let count = 1 + rng.below(24) as usize;
            let steps = 1 + rng.below(12) as u32;
            let per_launch = 1 + rng.below(3) as u32;
            let classes = [Lifetime::SEQUENTIAL, 0, 7, 9];
            let descriptors: Vec<ValueDesc> = (0..count)
                .map(|_| {
                    let words = match rng.below(8) {
                        0 => (chunk / 4).saturating_sub(rng.below(80)),
                        1 => chunk / 4 + 1 + rng.below(64),
                        _ => 1 + rng.below(chunk / 16),
                    };
                    desc(u32::try_from(words).expect("small"))
                })
                .collect();
            let lifetimes: Vec<Lifetime> = (0..count)
                .map(|_| {
                    let def = rng.below(u64::from(steps)) as u32;
                    let last = def + rng.below(u64::from(steps - def)) as u32;
                    Lifetime {
                        def,
                        last,
                        dead: rng.below(10) == 0,
                        reusable: rng.below(5) != 0,
                        launch_def: def / per_launch,
                        launch_last: last / per_launch,
                        class_def: classes[rng.below(4) as usize],
                        class_last: classes[rng.below(4) as usize],
                    }
                })
                .collect();
            let reserve = rng.below(4) * 256;
            let temporary = rng.below(chunk / 2);
            match layout_chunked(&descriptors, &lifetimes, reserve, temporary, chunk) {
                Ok(laid) => {
                    holds(&laid, &descriptors, &lifetimes, reserve, chunk);
                    laid_count += 1;
                    many += usize::from(laid.pieces() > 1);
                }
                Err(TooLarge::PastChunk {
                    item: Item::Value(value),
                    bytes,
                    chunk: named,
                }) => {
                    let value = value as usize;
                    assert_eq!(named, chunk);
                    assert_eq!(bytes, descriptors[value].device_bytes());
                    assert!(
                        align_up(bytes).unwrap() > chunk,
                        "value {value} fits a chunk"
                    );
                    assert!(!lifetimes[value].dead);
                    refused_count += 1;
                }
                Err(TooLarge::PastChunk {
                    item: Item::Temporary,
                    bytes,
                    ..
                }) => {
                    assert!(align_up(reserve).unwrap() + align_up(temporary).unwrap() > chunk);
                    assert_eq!(bytes, reserve + align_up(temporary).unwrap());
                    refused_count += 1;
                }
                Err(other) => panic!("{other:?}"),
            }
        }
        assert!(
            laid_count > 1000 && refused_count > 100 && many > 500,
            "{laid_count} {refused_count} {many}"
        );
    }

    fn a_value_dead_before_the_next_region_hands_its_slot_on() {
        let descriptors = [desc(1024), desc(1024), desc(1024)];
        let lifetimes = [life(0, 1), life(1, 2), life(2, 2)];
        let reused = layout_reusing(&descriptors, &lifetimes, 0, 0).unwrap();
        let naive = layout(&descriptors).unwrap();
        assert_eq!(reused.values[2], reused.values[0]);
        assert_ne!(reused.values[1], reused.values[0]);
        assert!(reused.total < naive.total);
    }

    fn a_value_last_read_where_another_is_defined_does_not_share() {
        let descriptors = [desc(1024), desc(1024)];
        let lifetimes = [life(0, 1), life(1, 1)];
        let reused = layout_reusing(&descriptors, &lifetimes, 0, 0).unwrap();
        assert_ne!(reused.values[0], reused.values[1]);
    }

    fn within_a_many_block_launch_only_one_class_shares() {
        let descriptors = [desc(1024), desc(1024), desc(1024), desc(1024), desc(1024)];
        let lifetimes = [
            rowed(0, 0, 0, 7),
            rowed(1, 1, 0, 7),
            rowed(2, 2, 0, 3),
            rowed(3, 3, 0, 0),
            rowed(4, 4, 1, 0),
        ];
        let reused = layout_reusing(&descriptors, &lifetimes, 0, 0).unwrap();
        assert_eq!(
            reused.values[1], reused.values[0],
            "same class, same launch"
        );
        assert_ne!(reused.values[2], reused.values[0], "another class");
        assert_ne!(reused.values[2], reused.values[1]);
        assert_ne!(
            reused.values[3], reused.values[0],
            "a whole value shares with nothing"
        );
        assert_eq!(
            reused.values[4], reused.values[0],
            "the next launch may take it"
        );
    }

    fn a_result_that_may_go_unwritten_keeps_a_fresh_slot() {
        let descriptors = [desc(1024), desc(1024)];
        let lifetimes = [
            life(0, 0),
            Lifetime {
                reusable: false,
                ..life(1, 1)
            },
        ];
        let reused = layout_reusing(&descriptors, &lifetimes, 0, 0).unwrap();
        assert_ne!(reused.values[0], reused.values[1]);
    }

    fn a_wider_taker_gets_a_coalesced_pair_of_slots() {
        let descriptors = [desc(1024), desc(1024), desc(2048)];
        let lifetimes = [life(0, 0), life(0, 0), life(1, 1)];
        let reused = layout_reusing(&descriptors, &lifetimes, 0, 0).unwrap();
        assert_eq!(reused.values[2], reused.values[0].min(reused.values[1]));
    }

    fn the_temporary_area_holds_what_is_stated() {
        let wide = ValueDesc {
            len: 4 * 1024,
            rows: 4,
            last: 1024,
            rank: 2,
            dtype: 0,
            dims: [0; MAX_RANK],
        };
        let none = layout_reusing(&[wide], &[life(0, 0)], 0, 0).unwrap();
        assert_eq!(none.temporary_bytes, 0, "nothing stated, nothing held");
        assert_eq!(none.total, none.temporary);
        let stated = layout_reusing(&[wide], &[life(0, 0)], 0, 1000).unwrap();
        assert_eq!(stated.values, none.values);
        assert_eq!(stated.temporary, none.temporary);
        assert_eq!(stated.temporary_bytes, align_up(1000).unwrap());
        assert_eq!(
            layout(&[wide]).unwrap().temporary_bytes,
            4 * 1024 * 4 * TEMPORARIES_PER_ELEMENT,
            "each value its own bytes holds the widest value's words"
        );
        assert_eq!(
            layout_reusing(&[wide], &[life(0, 0)], 0, u64::MAX),
            Err(TooLarge::Overflow)
        );
    }

    fn a_reserve_lies_below_the_temporary_arena() {
        let bare = layout_reusing(&[desc(1024)], &[life(0, 0)], 0, 0).unwrap();
        let held = layout_reusing(&[desc(1024)], &[life(0, 0)], SELECT_RESERVE, 0).unwrap();
        assert_eq!(held.values, bare.values);
        assert_eq!(
            held.temporary,
            align_up(bare.temporary + SELECT_RESERVE).unwrap()
        );
        assert_eq!(held.temporary_bytes, bare.temporary_bytes);
        assert_eq!(
            held.temporary,
            layout(&[desc(1024)]).unwrap().temporary,
            "what each value its own bytes holds clear"
        );
    }

    fn mismatched_lifetimes_fall_back_to_the_naive_layout() {
        let descriptors = [desc(1024), desc(1024)];
        assert_eq!(
            layout_reusing(&descriptors, &[life(0, 0)], 0, 0).unwrap(),
            layout(&descriptors).unwrap()
        );
    }

    fn a_dead_value_takes_no_slot() {
        let descriptors = [desc(1024), desc(1024), desc(1024)];
        let mut lifetimes = [life(0, 2), life(1, 2), life(2, 2)];
        let with = layout_reusing(&descriptors, &lifetimes, 0, 0).expect("fits");
        lifetimes[1].dead = true;
        let without = layout_reusing(&descriptors, &lifetimes, 0, 0).expect("fits");
        assert_eq!(
            without.values[1], 0,
            "a dead value's offset is the dummy region's"
        );
        assert_eq!(
            without.total + align_up(4096).unwrap(),
            with.total,
            "the dead value's slot is the whole difference"
        );
        assert_eq!(
            without.values[2],
            without.values[0] + align_up(4096).unwrap()
        );
    }
}
