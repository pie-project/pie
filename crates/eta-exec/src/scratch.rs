use super::extent::ValueDesc;
pub use eta_compiler::codegen::temporary::TemporaryNeed;

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

    /// An item larger than a segment's chunks: a piece is a run of
    /// adjacent chunks inside one segment, no run is longer than a segment,
    /// and no item crosses a piece ([`layout_chunked`]).
    PastSegment {
        item: Item,
        bytes: u64,
        chunk: u64,
        segment: u32,
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
            Self::PastSegment {
                item: Item::Value(value),
                bytes,
                chunk,
                segment,
            } => write!(
                f,
                "value {value} takes {bytes} bytes, past a segment of {segment} {chunk}-byte \
                 chunks, and a value lies whole in one run of a segment's chunks"
            ),
            Self::PastSegment {
                item: Item::Temporary,
                bytes,
                chunk,
                segment,
            } => write!(
                f,
                "the temporary area and the reserve below it take {bytes} bytes, past a \
                 segment of {segment} {chunk}-byte chunks, and they lie whole in one run of \
                 a segment's chunks"
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

/// A stage's temporary area: the widest of the whole-value rule
/// ([`whole_value_temporaries`]) and what each region's emitter publishes
/// it takes (`needs`, [`TemporaryNeed`]), evaluated against `descriptors`.
/// An emitter that publishes none passes none. Refused by name where a
/// need names a value the stage does not hold, or the bytes pass a `u64`.
pub fn stage_temporaries(
    descriptors: &[ValueDesc],
    needs: &[TemporaryNeed],
) -> Result<u64, TemporaryRefused> {
    let whole = whole_value_temporaries(descriptors).ok_or(TemporaryRefused::Overflow)?;
    needs.iter().try_fold(whole, |widest, need| {
        let rows = match need.per_row {
            Some((value, bytes)) => u64::from(
                descriptors
                    .get(value as usize)
                    .ok_or(TemporaryRefused::NoValue { value })?
                    .rows,
            )
            .checked_mul(bytes)
            .ok_or(TemporaryRefused::Overflow)?,
            None => 0,
        };
        let below = if need.above_whole { whole } else { 0 };
        let bytes = need
            .fixed
            .checked_add(rows)
            .and_then(|bytes| bytes.checked_add(below))
            .ok_or(TemporaryRefused::Overflow)?;
        Ok(widest.max(bytes))
    })
}

/// Why [`stage_temporaries`] refuses a stage's temporary area.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TemporaryRefused {
    /// A region's need is a row count of a value the stage does not hold.
    NoValue { value: u32 },
    /// The bytes pass a `u64`.
    Overflow,
}

impl core::fmt::Display for TemporaryRefused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoValue { value } => write!(
                f,
                "a region's temporary need counts the rows of value {value}, which the stage does not hold"
            ),
            Self::Overflow => write!(f, "a stage's temporary area passes a u64"),
        }
    }
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
/// `piece`. An engine addresses it as the piece's base (its run's first
/// chunk's first byte, from the instance's base table) plus `offset`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Placed {
    pub piece: u32,
    pub offset: u64,
}

/// What a chunked scratch is laid in ([`layout_chunked`]): chunks of
/// `chunk` bytes, which lie at adjacent addresses only inside one segment
/// of `segment` chunks, and the least chunks a piece takes, `run` (1: a
/// chunk; more where a program lays all its stages in one piece size).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pieces {
    pub chunk: u64,
    pub segment: u32,
    pub run: u32,
}

impl Pieces {
    /// Pieces of at least one chunk of `chunk` bytes, in segments of
    /// `segment` chunks.
    #[must_use]
    pub const fn new(chunk: u64, segment: u32) -> Self {
        Self {
            chunk,
            segment,
            run: 1,
        }
    }

    /// The same, each piece at least `run` chunks.
    #[must_use]
    pub const fn at_least(self, run: u32) -> Self {
        Self { run, ..self }
    }
}

/// A stage's scratch cut into pieces of `run` adjacent chunks each
/// ([`lay_out_chunked`]): no item crosses a piece, and two values whose
/// lifetimes overlap share no bytes. Piece 0 opens with the
/// [`DUMMY_BYTES`] a value that takes no bytes is stated to lie at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkLayout {
    /// The chunks a piece takes: a run of adjacent chunks inside one
    /// segment, what the largest item asks ([`Pieces::run`] at least).
    pub run: u32,
    /// A chunk's bytes.
    pub chunk: u64,
    /// Each value's place; a dead value's is piece 0, byte 0.
    pub values: Vec<Placed>,
    /// The temporary area's first byte: the reserve lies right below it,
    /// in the same piece.
    pub temporary: Placed,
    /// The bytes the temporary area holds.
    pub temporary_bytes: u64,
    /// Each piece's bytes in use, from its first: none past
    /// [`Self::piece_bytes`].
    pub ends: Vec<u64>,
}

impl ChunkLayout {
    /// The pieces the scratch takes: chunks, or one run where it fits one.
    #[must_use]
    pub fn pieces(&self) -> u32 {
        u32::try_from(self.ends.len()).unwrap_or(u32::MAX)
    }

    /// A piece's bytes: its run of chunks.
    #[must_use]
    pub fn piece_bytes(&self) -> u64 {
        u64::from(self.run).saturating_mul(self.chunk)
    }

    /// The bytes an instance holds for it: the one piece's bytes where it
    /// takes one inside a chunk (a run of slots), else whole runs of
    /// chunks, one a piece.
    #[must_use]
    pub fn held_bytes(&self) -> u64 {
        match self.ends[..] {
            [only] if only <= self.chunk => only,
            _ => u64::from(self.pieces()).saturating_mul(self.piece_bytes()),
        }
    }
}

/// Whether value `later`'s bytes may lie over value `earlier`'s, given the
/// two lifetimes, or `None` where they may lie anywhere over each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Share {
    /// Anywhere over each other: one is last read a launch before the
    /// other is written, or a step before it in a launch of one block.
    Anywhere,
    /// Only at one offset: within a launch across blocks, block `b` owns
    /// row `b` of every value of a class, so a value that takes another's
    /// bytes there lies at the other's offset, or a block's row would land
    /// on another block's, which may still be reading it.
    AtItsOffset,
    /// Nowhere: their lifetimes overlap, the later may go unwritten in
    /// part (it keeps bytes no value before it held), or they are rows of
    /// two classes in one launch.
    Apart,
}

fn share(a: Lifetime, b: Lifetime) -> Share {
    let (early, late) = if a.last < b.def {
        (a, b)
    } else if b.last < a.def {
        (b, a)
    } else {
        return Share::Apart;
    };
    if !late.reusable {
        Share::Apart
    } else if early.launch_last < late.launch_def {
        Share::Anywhere
    } else if late.class_def == 0 || early.class_last != late.class_def {
        Share::Apart
    } else if late.class_def == Lifetime::SEQUENTIAL {
        Share::Anywhere
    } else {
        Share::AtItsOffset
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

/// The chunks of `chunk` bytes a piece of `descriptors`' scratch takes:
/// enough for the largest value alive in it, and for the temporary area
/// of `temporary` bytes with `reserve` bytes below it, at least `least`.
/// Past `segment` chunks, the item that asks it is refused by name
/// ([`TooLarge::PastSegment`]).
fn run_for(
    descriptors: &[ValueDesc],
    lifetimes: &[Lifetime],
    reserve: u64,
    temporary: u64,
    pieces: Pieces,
) -> Result<u32, TooLarge> {
    let chunk = pieces.chunk.clamp(1, MAX_BYTES);
    let past = |item, bytes| TooLarge::PastSegment {
        item,
        bytes,
        chunk,
        segment: pieces.segment,
    };
    let runs = |span: u64| span.div_ceil(chunk);
    let mut run = u64::from(pieces.run.max(1));
    for (i, descriptor) in descriptors.iter().enumerate() {
        if lifetimes[i].dead {
            continue;
        }
        let bytes = descriptor.device_bytes();
        let span = runs(align_up(bytes).ok_or(TooLarge::Overflow)?);
        if span > u64::from(pieces.segment) {
            return Err(past(
                Item::Value(u32::try_from(i).unwrap_or(u32::MAX)),
                bytes,
            ));
        }
        run = run.max(span);
    }
    let area = align_up(reserve)
        .zip(align_up(temporary))
        .and_then(|(reserve, temporary)| reserve.checked_add(temporary))
        .ok_or(TooLarge::Overflow)?;
    let span = runs(area);
    if span > u64::from(pieces.segment) {
        return Err(past(
            Item::Temporary,
            reserve.saturating_add(align_up(temporary).unwrap_or(u64::MAX)),
        ));
    }
    u32::try_from(run.max(span)).map_err(|_| TooLarge::Overflow)
}

/// `descriptors` laid by `lifetimes` in pieces of at most `limit` bytes,
/// largest first: each value at the lowest offset of the first piece where
/// it shares bytes only with values it may share them with ([`share`]),
/// else in a new piece. Two values share bytes only where one is last read
/// before the other is written, so the bytes laid are what the values live
/// together at once take, short of the gaps that sizes leave. `reserve`
/// bytes lie right below the temporary area, which holds `temporary` bytes,
/// both in one piece past every value. The caller has sized `limit` for
/// every item ([`run_for`]).
fn place(
    descriptors: &[ValueDesc],
    lifetimes: &[Lifetime],
    reserve: u64,
    temporary: u64,
    limit: u64,
) -> Result<ChunkLayout, TooLarge> {
    let count = descriptors.len();
    let mut spans = vec![0u64; count];
    for (i, descriptor) in descriptors.iter().enumerate() {
        spans[i] = align_up(descriptor.device_bytes()).ok_or(TooLarge::Overflow)?;
    }
    let mut order: Vec<usize> = (0..count).filter(|&i| !lifetimes[i].dead).collect();
    order.sort_by_key(|&i| (core::cmp::Reverse(spans[i]), lifetimes[i].def, i));

    let mut values = vec![Placed::default(); count];
    // Each piece's values, as laid.
    let mut laid: Vec<Vec<usize>> = Vec::new();
    let mut ends = vec![DUMMY_BYTES];
    let mut near: Vec<(u64, u64, bool)> = Vec::new();
    let mut candidates: Vec<u64> = Vec::new();
    for &i in &order {
        let span = spans[i];
        let mut found = None;
        // The pieces laid so far, then a new one (piece 0 is open from the
        // start, with the dummy bytes).
        for piece in 0..=laid.len().max(1) {
            let base = if piece == 0 { DUMMY_BYTES } else { 0 };
            near.clear();
            candidates.clear();
            candidates.push(base);
            for &j in laid.get(piece).map_or(&[][..], Vec::as_slice) {
                let at = values[j].offset;
                let one = match share(lifetimes[i], lifetimes[j]) {
                    Share::Anywhere => continue,
                    Share::AtItsOffset => true,
                    Share::Apart => false,
                };
                near.push((at, at + spans[j], one));
                candidates.push(at + spans[j]);
                if one {
                    candidates.push(at);
                }
            }
            candidates.sort_unstable();
            candidates.dedup();
            let fits = |at: u64| {
                near.iter().all(|&(start, end, one)| {
                    end <= at || at + span <= start || (one && start == at)
                })
            };
            if let Some(&at) = candidates
                .iter()
                .filter(|&&at| at >= base && at.checked_add(span).is_some_and(|stop| stop <= limit))
                .find(|&&at| fits(at))
            {
                found = Some((piece, at));
                break;
            }
        }
        let Some((piece, offset)) = found else {
            return Err(TooLarge::Bound {
                bytes: descriptors[i].device_bytes(),
                limit,
            });
        };
        while laid.len() <= piece {
            laid.push(Vec::new());
        }
        laid[piece].push(i);
        while ends.len() <= piece {
            ends.push(0);
        }
        ends[piece] = ends[piece].max(offset + span);
        values[i] = Placed {
            piece: u32::try_from(piece).map_err(|_| TooLarge::Overflow)?,
            offset,
        };
    }

    let temporary_bytes = align_up(temporary).ok_or(TooLarge::Overflow)?;
    let (piece, offset) = bump(&mut ends, reserve, temporary_bytes, limit)
        .ok_or(TooLarge::Overflow)?
        .ok_or(TooLarge::Bound {
            bytes: reserve.saturating_add(temporary_bytes),
            limit,
        })?;
    Ok(ChunkLayout {
        run: 1,
        chunk: limit,
        values,
        temporary: Placed { piece, offset },
        temporary_bytes,
        ends,
    })
}

/// The most bytes `descriptors`' values alive together take at any step of
/// `lifetimes` (each value's bytes aligned as a layout lays them): the
/// least any layout of their values holds.
#[must_use]
pub fn most_alive(descriptors: &[ValueDesc], lifetimes: &[Lifetime]) -> u64 {
    let mut edges: Vec<(u32, bool, u64)> = descriptors
        .iter()
        .zip(lifetimes)
        .filter(|(_, life)| !life.dead)
        .flat_map(|(descriptor, life)| {
            let span = align_up(descriptor.device_bytes()).unwrap_or(u64::MAX);
            // A value is alive from its step through its last: born at
            // `def`, gone after `last`; at one step births count first.
            [
                (life.def, false, span),
                (life.last.saturating_add(1), true, span),
            ]
        })
        .collect();
    edges.sort_unstable_by_key(|&(step, gone, _)| (step, !gone));
    let (mut alive, mut most) = (0u64, 0u64);
    for (_, gone, span) in edges {
        if gone {
            alive -= span;
        } else {
            alive = alive.saturating_add(span);
            most = most.max(alive);
        }
    }
    most
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

/// `descriptors` laid by `lifetimes` in `pieces`, the layout every
/// backend's guest scratch takes ([`ChunkLayout`]): a piece is a run of
/// `run` adjacent chunks inside one segment, `run` the least that holds
/// the largest value alive, and the temporary area with the reserve below
/// it, whole ([`Pieces::run`] at least); no item crosses a piece, values
/// packed into pieces, each addressed as its piece's base plus its offset.
/// An item past a segment's chunks is refused by name
/// ([`TooLarge::PastSegment`]). Where `lifetimes` are not the values',
/// each value holds its own bytes.
pub fn layout_chunked(
    descriptors: &[ValueDesc],
    lifetimes: &[Lifetime],
    reserve: u64,
    temporary: u64,
    pieces: Pieces,
) -> Result<ChunkLayout, TooLarge> {
    let own;
    let lifetimes = if lifetimes.len() == descriptors.len() {
        lifetimes
    } else {
        own = each_its_own(descriptors.len());
        &own
    };
    let run = run_for(descriptors, lifetimes, reserve, temporary, pieces)?;
    let chunk = pieces.chunk.clamp(1, MAX_BYTES);
    let limit = u64::from(run)
        .checked_mul(chunk)
        .ok_or(TooLarge::Overflow)?
        .min(MAX_BYTES);
    let laid = place(descriptors, lifetimes, reserve, temporary, limit)?;
    Ok(ChunkLayout { run, chunk, ..laid })
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

    /// The stage's temporary area is the widest of the whole-value rule and
    /// what each region's emitter publishes: a short vector's select keeps
    /// its fixed state below the rule's bytes, a top-k takes so much a row
    /// of its input, an emitter that publishes none takes the rule alone,
    /// and a need naming a value the stage lacks, or past a u64, is refused
    /// by name.
    #[test]
    fn a_stage_s_temporaries_take_the_widest_published_need() {
        let short = [desc(16)];
        let whole = whole_value_temporaries(&short).unwrap();
        assert_eq!(whole, 16 * 16);
        assert_eq!(stage_temporaries(&short, &[]), Ok(whole), "none published: the rule");
        let select = TemporaryNeed {
            fixed: 51_200,
            per_row: None,
            above_whole: true,
        };
        assert_eq!(stage_temporaries(&short, &[select]), Ok(51_200 + whole));
        let mut rows = desc(16);
        rows.rows = 4;
        let topk = TemporaryNeed {
            fixed: 15,
            per_row: Some((0, 14_360)),
            above_whole: false,
        };
        assert_eq!(stage_temporaries(&[rows], &[topk]), Ok(15 + 4 * 14_360));
        assert_eq!(
            stage_temporaries(&short, &[select, topk]),
            Ok((51_200 + whole).max(15 + 14_360)),
            "the widest region's"
        );
        // A wide value's rule covers a small need.
        let wide = [desc(1 << 20)];
        assert_eq!(
            stage_temporaries(&wide, &[topk]),
            Ok(whole_value_temporaries(&wide).unwrap())
        );
        let stray = TemporaryNeed {
            per_row: Some((3, 8)),
            ..TemporaryNeed::default()
        };
        assert_eq!(
            stage_temporaries(&short, &[stray]),
            Err(TemporaryRefused::NoValue { value: 3 })
        );
        let past = TemporaryNeed {
            fixed: u64::MAX,
            above_whole: true,
            ..TemporaryNeed::default()
        };
        assert_eq!(stage_temporaries(&short, &[past]), Err(TemporaryRefused::Overflow));
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
        small_values_leave_room_for_the_wide_ones_that_follow();
        rows_of_one_class_share_only_at_one_offset(None);
        // In pieces of a chunk, a few values a piece: the same rule within
        // each piece, and none past it.
        rows_of_one_class_share_only_at_one_offset(Some(2048));
        rows_of_one_class_share_only_at_one_offset(Some(4096));
    }

    /// Within a launch across blocks, block `b` owns row `b` of every value
    /// of a row class: a value that takes another's bytes there lies at the
    /// other's offset, so no block's row lands on another block's. Values
    /// that share across launches may lie anywhere over each other. With
    /// `chunk`, laid in pieces of it ([`layout_chunked`]): two values share
    /// bytes only in one piece, and the rule holds there.
    fn rows_of_one_class_share_only_at_one_offset(chunk: Option<u64>) {
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
                let class = if next(5) == 0 {
                    0
                } else {
                    (1u64 << 40) | u64::from(width) * 4
                };
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
            let places: Vec<Placed> = match chunk {
                None => layout_reusing(&descriptors, &lifetimes, 0, 0)
                    .unwrap()
                    .values
                    .into_iter()
                    .map(|offset| Placed { piece: 0, offset })
                    .collect(),
                Some(chunk) => {
                    let laid =
                        layout_chunked(&descriptors, &lifetimes, 0, 0, Pieces::new(chunk, 1))
                            .unwrap();
                    holds(&laid, &descriptors, &lifetimes, 0);
                    laid.values
                }
            };
            let span = |i: usize| align_up(descriptors[i].device_bytes()).unwrap();
            for i in 0..count {
                for j in 0..count {
                    let (a, b) = (lifetimes[i], lifetimes[j]);
                    let (pi, pj) = (places[i], places[j]);
                    let (oi, oj) = (pi.offset, pj.offset);
                    let overlap = pi.piece == pj.piece && oi < oj + span(j) && oj < oi + span(i);
                    if i == j || !overlap {
                        continue;
                    }
                    assert!(
                        a.last < b.def || b.last < a.def,
                        "{i} and {j} live together"
                    );
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

    /// Chunks of [`CHUNK`] in segments of 8.
    const PIECES: Pieces = Pieces::new(CHUNK, 8);

    /// Chunks of [`CHUNK`] in segments of one: a piece is one chunk.
    const ONE: Pieces = Pieces::new(CHUNK, 1);

    /// Every piece holds its items whole within its run of chunks, none in
    /// piece 0's dummy bytes, and two values share a byte only where one is
    /// last read before the other is written; the temporary area and the
    /// reserve below it share none with a value.
    fn holds(laid: &ChunkLayout, descriptors: &[ValueDesc], lifetimes: &[Lifetime], reserve: u64) {
        assert_eq!(laid.values.len(), descriptors.len());
        assert!(!laid.ends.is_empty());
        let piece = laid.piece_bytes();
        assert!(laid.ends.iter().all(|&end| end <= piece), "{:?}", laid.ends);
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

    #[test]
    fn runs_every_case() {
        a_value_past_a_chunk_lies_in_a_run_of_chunks();
        the_temporary_area_past_a_chunk_lies_in_a_run_of_chunks();
        a_value_past_a_segment_is_refused_by_name();
        a_stated_run_lays_every_piece_that_long();
        run_one_is_the_chunked_layout();
    }

    fn no_value_crosses_a_piece() {
        // Eight values of 16 KiB alive together: three in the first 64 KiB
        // piece beside the dummy bytes, four in the next, which has none,
        // and one in a third.
        let descriptors = [desc(4096); 8];
        let lifetimes = [life(0, 7); 8];
        let laid = layout_chunked(&descriptors, &lifetimes, 0, 0, PIECES).expect("fits");
        holds(&laid, &descriptors, &lifetimes, 0);
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
        let laid = layout_chunked(&descriptors, &lifetimes, 0, 0, PIECES).expect("fits");
        holds(&laid, &descriptors, &lifetimes, 0);
        assert_ne!(laid.values[1], laid.values[0]);
        assert_eq!(
            laid.ends,
            [DUMMY_BYTES + 16384 + 32768],
            "the most alive at once, the last two: {laid:?}"
        );
    }

    /// A segment of one chunk: a value past a chunk is past the segment.
    fn a_value_past_a_chunk_is_refused_by_name() {
        let descriptors = [desc(1024), desc(16385)];
        assert_eq!(
            layout_chunked(&descriptors, &[life(0, 1), life(1, 1)], 0, 0, ONE),
            Err(TooLarge::PastSegment {
                item: Item::Value(1),
                bytes: 16385 * 4,
                chunk: CHUNK,
                segment: 1,
            })
        );
        let refused = layout_chunked(&descriptors, &[], 0, 0, ONE).unwrap_err();
        assert!(
            refused
                .to_string()
                .starts_with("value 1 takes 65540 bytes, past a segment of 1 65536-byte chunks"),
            "{refused}"
        );
        // Exactly a chunk fits a piece of its own.
        let whole = [desc(1024), desc(16384)];
        let laid = layout_chunked(&whole, &[life(0, 1), life(1, 1)], 0, 0, ONE).expect("fits");
        assert_eq!(laid.run, 1);
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
            layout_chunked(&descriptors, &[life(0, 0)], 4096, CHUNK - 4096 + 1, ONE),
            Err(TooLarge::PastSegment {
                item: Item::Temporary,
                bytes: 4096 + align_up(CHUNK - 4096 + 1).unwrap(),
                chunk: CHUNK,
                segment: 1,
            })
        );
        let laid = layout_chunked(&descriptors, &[life(0, 0)], 4096, CHUNK - 4096, ONE)
            .expect("the reserve and the area fill a piece of their own");
        assert_eq!(laid.run, 1);
        assert_eq!(
            laid.temporary,
            Placed {
                piece: 1,
                offset: 4096
            }
        );
        holds(&laid, &descriptors, &[life(0, 0)], 4096);
    }

    fn one_unbounded_piece_is_the_reusing_layout() {
        let descriptors = [desc(1024), desc(1024), desc(2048), desc(512)];
        let lifetimes = [life(0, 0), life(0, 0), life(1, 2), life(2, 2)];
        let laid = layout_chunked(
            &descriptors,
            &lifetimes,
            SELECT_RESERVE,
            777,
            Pieces::new(MAX_BYTES, 1),
        )
        .expect("fits");
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
        assert_eq!(laid.held_bytes(), reused.total);
    }

    fn an_instance_holds_one_run_or_whole_chunks() {
        let one = layout_chunked(&[desc(1024)], &[life(0, 0)], 0, 0, PIECES).expect("fits");
        assert_eq!(one.held_bytes(), DUMMY_BYTES + 4096, "the tighter run");
        let three = layout_chunked(&[desc(4096); 8], &[life(0, 0); 8], 0, 0, PIECES).expect("fits");
        assert_eq!(three.held_bytes(), 3 * CHUNK, "whole chunks, one a piece");
    }

    fn a_freed_slot_is_taken_in_its_own_piece() {
        // Piece 0 full of values alive to the end but the first; piece 1
        // opened; the first's bytes freed and taken back in piece 0.
        let descriptors = [desc(4096), desc(4096), desc(4096), desc(4096), desc(4096)];
        let lifetimes = [life(0, 0), life(0, 3), life(0, 3), life(0, 3), life(1, 3)];
        let laid = layout_chunked(&descriptors, &lifetimes, 0, 0, PIECES).expect("fits");
        holds(&laid, &descriptors, &lifetimes, 0);
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

    /// The chunks of `chunk` bytes the largest live value of
    /// `descriptors`, or the temporary area with its reserve, takes.
    fn widest(
        descriptors: &[ValueDesc],
        lifetimes: &[Lifetime],
        reserve: u64,
        temporary: u64,
        chunk: u64,
    ) -> u64 {
        descriptors
            .iter()
            .zip(lifetimes)
            .filter(|(_, life)| !life.dead)
            .map(|(descriptor, _)| align_up(descriptor.device_bytes()).unwrap())
            .chain([align_up(reserve).unwrap() + align_up(temporary).unwrap()])
            .map(|bytes| bytes.div_ceil(chunk))
            .fold(1, u64::max)
    }

    /// Random stages: widths across and past a chunk, lifetimes over a few
    /// launches of one or many blocks, results that may go unwritten, dead
    /// values, reserves and temporary areas, segments of one to three
    /// chunks. Each laid one holds ([`holds`]) in pieces of the least run
    /// that holds its widest item, or is refused naming an item that is
    /// past the segment.
    #[test]
    fn chunked_adversarial_random_stages() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let (mut laid_count, mut refused_count, mut many, mut runs) = (0, 0, 0, 0);
        for _ in 0..4000 {
            let chunk = (1 + rng.below(64)) * 1024;
            let segment = 1 + rng.below(3) as u32;
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
            let pieces = Pieces::new(chunk, segment);
            let wide = widest(&descriptors, &lifetimes, reserve, temporary, chunk);
            let room = u64::from(segment) * chunk;
            match layout_chunked(&descriptors, &lifetimes, reserve, temporary, pieces) {
                Ok(laid) => {
                    holds(&laid, &descriptors, &lifetimes, reserve);
                    assert_eq!(u64::from(laid.run), wide, "the least run that holds it");
                    assert_eq!(laid.chunk, chunk);
                    laid_count += 1;
                    many += usize::from(laid.pieces() > 1);
                    runs += usize::from(laid.run > 1);
                }
                Err(TooLarge::PastSegment {
                    item: Item::Value(value),
                    bytes,
                    chunk: named,
                    segment: of,
                }) => {
                    let value = value as usize;
                    assert_eq!((named, of), (chunk, segment));
                    assert_eq!(bytes, descriptors[value].device_bytes());
                    assert!(
                        align_up(bytes).unwrap() > room,
                        "value {value} fits a segment"
                    );
                    assert!(!lifetimes[value].dead);
                    refused_count += 1;
                }
                Err(TooLarge::PastSegment {
                    item: Item::Temporary,
                    bytes,
                    ..
                }) => {
                    assert!(align_up(reserve).unwrap() + align_up(temporary).unwrap() > room);
                    assert_eq!(bytes, reserve + align_up(temporary).unwrap());
                    refused_count += 1;
                }
                Err(other) => panic!("{other:?}"),
            }
        }
        assert!(
            laid_count > 1000 && refused_count > 100 && many > 500 && runs > 200,
            "{laid_count} {refused_count} {many} {runs}"
        );
    }

    /// A value of 2.5 chunks lies whole in a piece of a run of 3 adjacent
    /// chunks, at its piece's base plus its offset; the smaller values pack
    /// beside it in the same run.
    fn a_value_past_a_chunk_lies_in_a_run_of_chunks() {
        let big = u32::try_from(5 * CHUNK / 2 / 4).unwrap();
        let descriptors = [desc(1024), desc(big), desc(4096)];
        let lifetimes = [life(0, 2), life(1, 2), life(2, 2)];
        let laid = layout_chunked(&descriptors, &lifetimes, 0, 0, PIECES).expect("fits");
        holds(&laid, &descriptors, &lifetimes, 0);
        assert_eq!(laid.run, 3, "ceil(2.5 chunks)");
        assert_eq!(laid.piece_bytes(), 3 * CHUNK);
        assert_eq!(laid.pieces(), 1, "all three fit one run: {laid:?}");
        assert_eq!(laid.held_bytes(), 3 * CHUNK, "a whole run of chunks");
        // Exactly two chunks: a run of two.
        let two = [desc(u32::try_from(2 * CHUNK / 4).unwrap())];
        let laid = layout_chunked(&two, &[life(0, 0)], 0, 0, PIECES).expect("fits");
        assert_eq!(laid.run, 2);
        assert_eq!(
            laid.values[0],
            Placed {
                piece: 1,
                offset: 0
            }
        );
        assert_eq!(
            laid.held_bytes(),
            2 * 2 * CHUNK,
            "two runs, piece 0 the dummy's"
        );
    }

    /// The temporary area with the reserve below it lies whole in one run.
    fn the_temporary_area_past_a_chunk_lies_in_a_run_of_chunks() {
        let descriptors = [desc(1024)];
        let laid = layout_chunked(&descriptors, &[life(0, 0)], 4096, CHUNK, PIECES).expect("fits");
        holds(&laid, &descriptors, &[life(0, 0)], 4096);
        assert_eq!(laid.run, 2);
        assert_eq!(
            laid.temporary,
            Placed {
                piece: 0,
                offset: align_up(DUMMY_BYTES + 4096 + 4096).unwrap(),
            },
            "beside the value, in piece 0's run"
        );
    }

    /// A value past a segment's chunks is refused by name; one segment
    /// long, it lies in a run of all of them.
    fn a_value_past_a_segment_is_refused_by_name() {
        let past = u32::try_from(8 * CHUNK / 4 + 1).unwrap();
        let descriptors = [desc(1024), desc(past)];
        let refused =
            layout_chunked(&descriptors, &[life(0, 1), life(1, 1)], 0, 0, PIECES).unwrap_err();
        assert_eq!(
            refused,
            TooLarge::PastSegment {
                item: Item::Value(1),
                bytes: u64::from(past) * 4,
                chunk: CHUNK,
                segment: 8,
            }
        );
        assert!(
            refused
                .to_string()
                .contains("past a segment of 8 65536-byte chunks"),
            "{refused}"
        );
        let whole = [desc(1024), desc(past - 1)];
        let laid = layout_chunked(&whole, &[life(0, 1), life(1, 1)], 0, 0, PIECES).expect("fits");
        assert_eq!(laid.run, 8);
        assert_eq!(
            laid.values[1],
            Placed {
                piece: 1,
                offset: 0
            }
        );
        assert!(matches!(
            layout_chunked(&[desc(1024)], &[life(0, 0)], 0, 8 * CHUNK + 1, PIECES),
            Err(TooLarge::PastSegment {
                item: Item::Temporary,
                ..
            })
        ));
    }

    /// A program lays all its stages in one piece size: a stated run is
    /// kept where its items need less.
    fn a_stated_run_lays_every_piece_that_long() {
        let descriptors = [desc(4096); 8];
        let lifetimes = [life(0, 7); 8];
        let laid =
            layout_chunked(&descriptors, &lifetimes, 0, 0, PIECES.at_least(3)).expect("fits");
        holds(&laid, &descriptors, &lifetimes, 0);
        assert_eq!(laid.run, 3);
        assert_eq!(
            laid.pieces(),
            1,
            "8 x 16 KiB and the dummy in one 192 KiB run"
        );
        let big = u32::try_from(5 * CHUNK / 2 / 4).unwrap();
        let laid =
            layout_chunked(&[desc(big)], &[life(0, 0)], 0, 0, PIECES.at_least(2)).expect("fits");
        assert_eq!(laid.run, 3, "a stated run is a least, not a most");
    }

    /// Where every item fits a chunk, the run is 1 and the layout is the
    /// one pieces of a chunk give: a segment's length changes nothing.
    fn run_one_is_the_chunked_layout() {
        let descriptors = [desc(4096), desc(1024), desc(4096), desc(8192), desc(16384)];
        let lifetimes = [life(0, 1), life(0, 3), life(2, 3), life(3, 4), life(4, 4)];
        let one = layout_chunked(&descriptors, &lifetimes, 1024, 4096, ONE).expect("fits");
        let eight = layout_chunked(&descriptors, &lifetimes, 1024, 4096, PIECES).expect("fits");
        assert_eq!(one.run, 1);
        assert_eq!(one, eight);
    }

    /// A sampler's shape: a wide value dead, then a scalar written, then
    /// two wide values alive with the scalar. The scalar does not cut the
    /// dead one's bytes so that a wide value no longer fits them: the
    /// bytes laid are the most alive at once.
    fn small_values_leave_room_for_the_wide_ones_that_follow() {
        let descriptors = [desc(1024), desc(1), desc(1024), desc(1024)];
        let lifetimes = [life(0, 0), life(1, 2), life(2, 2), life(2, 2)];
        let laid = layout_reusing(&descriptors, &lifetimes, 0, 0).unwrap();
        let most = most_alive(&descriptors, &lifetimes);
        assert_eq!(most, 2 * 4096 + ALIGN);
        assert_eq!(laid.total, DUMMY_BYTES + most, "{laid:?}");
        assert!(
            laid.values[2] == laid.values[0] || laid.values[3] == laid.values[0],
            "a wide value takes the dead one's bytes: {laid:?}"
        );
    }

    /// Random stages laid in one piece: two values whose lifetimes overlap
    /// share no byte, the bytes laid hold the most alive at once, and two
    /// values with disjoint lifetimes do share where nothing else is laid.
    #[test]
    fn disjoint_lifetimes_share_and_overlapping_ones_do_not() {
        let disjoint = [desc(1024), desc(1024)];
        let apart = layout_reusing(&disjoint, &[life(0, 0), life(1, 1)], 0, 0).unwrap();
        assert_eq!(apart.values[0], apart.values[1], "disjoint: one slot");
        let together = layout_reusing(&disjoint, &[life(0, 1), life(1, 1)], 0, 0).unwrap();
        assert_ne!(together.values[0], together.values[1], "overlapping: two");

        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        for _ in 0..4000 {
            let count = 1 + rng.below(24) as usize;
            let steps = 1 + rng.below(12) as u32;
            let descriptors: Vec<ValueDesc> = (0..count)
                .map(|_| desc(1 + rng.below(4096) as u32))
                .collect();
            let lifetimes: Vec<Lifetime> = (0..count)
                .map(|_| {
                    let def = rng.below(u64::from(steps)) as u32;
                    let last = def + rng.below(u64::from(steps - def)) as u32;
                    Lifetime {
                        dead: rng.below(10) == 0,
                        ..life(def, last)
                    }
                })
                .collect();
            let laid =
                layout_chunked(&descriptors, &lifetimes, 0, 0, Pieces::new(MAX_BYTES, 1)).unwrap();
            holds(&laid, &descriptors, &lifetimes, 0);
            assert!(laid.ends[0] >= DUMMY_BYTES + most_alive(&descriptors, &lifetimes));
        }
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
