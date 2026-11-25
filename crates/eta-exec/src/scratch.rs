use super::extent::ValueDesc;
pub use eta_compiler::codegen::temporary::TemporaryNeed;

mod lifetime;

pub use lifetime::{
    Keeps, Launch, Launches, describe_launched, lay_out, lay_out_chunked, launch_areas,
    read_only_unlaunched, region_rows, value_lifetimes,
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
    /// The area launch `n` of those that keep one takes ([`Area`]).
    Area(u32),
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
                item: Item::Area(area),
                bytes,
                chunk,
                segment,
            } => write!(
                f,
                "launch area {area} and its reserve take {bytes} bytes, past a segment of \
                 {segment} {chunk}-byte chunks, and they lie whole in one run of a segment's \
                 chunks"
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

    /// Each launch's own temporary area ([`Area`]), in the order stated:
    /// its first byte, its reserve right below.
    pub areas: Vec<u64>,

    pub total: u64,
}

/// A temporary area one launch keeps for itself ([`Keeps`]): alive over
/// its steps, `reserve` bytes clear below its `bytes`. It shares bytes
/// with a value as another value would, by their lifetimes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Area {
    pub life: Lifetime,
    pub reserve: u64,
    pub bytes: u64,
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
        areas: Vec::new(),
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
    /// Each launch's own area's first byte ([`Area`]), in the order stated.
    pub areas: Vec<Placed>,
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
    areas: &[Area],
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
    for (at, area) in areas.iter().enumerate() {
        let bytes = align_up(area.reserve)
            .zip(align_up(area.bytes))
            .and_then(|(reserve, bytes)| reserve.checked_add(bytes))
            .ok_or(TooLarge::Overflow)?;
        let span = runs(bytes);
        if span > u64::from(pieces.segment) {
            return Err(past(
                Item::Area(u32::try_from(at).unwrap_or(u32::MAX)),
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
/// largest first, with each of `areas` an item of its own: each at the
/// lowest offset of the first piece where it shares bytes only with items
/// it may share them with ([`share`]), else in a new piece. Two items
/// share bytes only where one is last read before the other is written,
/// so the bytes laid are what the items alive together at once take, short
/// of the gaps that sizes leave. `reserve` bytes lie right below the
/// stage's temporary area, which holds `temporary` bytes, both in one
/// piece past every item: the area alive across the stage. The caller has
/// sized `limit` for every item ([`run_for`]).
fn place(
    descriptors: &[ValueDesc],
    lifetimes: &[Lifetime],
    reserve: u64,
    temporary: u64,
    areas: &[Area],
    limit: u64,
) -> Result<ChunkLayout, TooLarge> {
    let count = descriptors.len();
    let mut spans = Vec::with_capacity(count + areas.len());
    for descriptor in descriptors {
        spans.push(align_up(descriptor.device_bytes()).ok_or(TooLarge::Overflow)?);
    }
    // An area's item: its reserve, aligned, then its bytes.
    let mut leads = Vec::with_capacity(areas.len());
    for area in areas {
        let lead = align_up(area.reserve).ok_or(TooLarge::Overflow)?;
        let span = align_up(area.bytes)
            .and_then(|bytes| bytes.checked_add(lead))
            .ok_or(TooLarge::Overflow)?;
        leads.push(lead);
        spans.push(span);
    }
    let life_of = |i: usize| -> Lifetime {
        if i < count {
            lifetimes[i]
        } else {
            areas[i - count].life
        }
    };
    let mut order: Vec<usize> = (0..count)
        .filter(|&i| !lifetimes[i].dead)
        .chain(count..count + areas.len())
        .collect();
    order.sort_by_key(|&i| (core::cmp::Reverse(spans[i]), life_of(i).def, i));

    let mut starts = vec![0u64; spans.len()];
    let mut pieces = vec![0u32; spans.len()];
    // Each piece's items, as laid.
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
                let at = starts[j];
                let one = match share(life_of(i), life_of(j)) {
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
                bytes: if i < count {
                    descriptors[i].device_bytes()
                } else {
                    span
                },
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
        starts[i] = offset;
        pieces[i] = u32::try_from(piece).map_err(|_| TooLarge::Overflow)?;
    }
    let placed = |i: usize, lead: u64| Placed {
        piece: pieces[i],
        offset: starts[i] + lead,
    };
    let values = (0..count)
        .map(|i| {
            if lifetimes[i].dead {
                Placed::default()
            } else {
                placed(i, 0)
            }
        })
        .collect();
    let areas = leads
        .iter()
        .enumerate()
        .map(|(at, &lead)| placed(count + at, lead))
        .collect();

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
        areas,
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
    chunked_with(descriptors, lifetimes, reserve, temporary, &[], pieces)
}

/// [`layout_chunked`] with each of `areas` laid as an item of its own
/// ([`place`]).
pub(crate) fn chunked_with(
    descriptors: &[ValueDesc],
    lifetimes: &[Lifetime],
    reserve: u64,
    temporary: u64,
    areas: &[Area],
    pieces: Pieces,
) -> Result<ChunkLayout, TooLarge> {
    let own;
    let lifetimes = if lifetimes.len() == descriptors.len() {
        lifetimes
    } else {
        own = each_its_own(descriptors.len());
        &own
    };
    let run = run_for(descriptors, lifetimes, reserve, temporary, areas, pieces)?;
    let chunk = pieces.chunk.clamp(1, MAX_BYTES);
    let limit = u64::from(run)
        .checked_mul(chunk)
        .ok_or(TooLarge::Overflow)?
        .min(MAX_BYTES);
    let laid = place(descriptors, lifetimes, reserve, temporary, areas, limit)?;
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
    reusing_with(descriptors, lifetimes, reserve, temporary, &[])
}

/// [`layout_reusing`] with each of `areas` laid as an item of its own
/// ([`place`]). Where `lifetimes` are not the values', each value holds
/// its own bytes and each area its own past them.
pub(crate) fn reusing_with(
    descriptors: &[ValueDesc],
    lifetimes: &[Lifetime],
    reserve: u64,
    temporary: u64,
    areas: &[Area],
) -> Result<Layout, TooLarge> {
    if lifetimes.len() != descriptors.len() {
        return layout_with_areas(descriptors, areas);
    }
    let laid = place(descriptors, lifetimes, reserve, temporary, areas, u64::MAX)?;
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
        areas: laid.areas.iter().map(|placed| placed.offset).collect(),
        total,
    })
}

/// Each value its own bytes ([`layout`]), and each of `areas` its own
/// past them, its reserve below it.
fn layout_with_areas(descriptors: &[ValueDesc], areas: &[Area]) -> Result<Layout, TooLarge> {
    let mut laid = layout(descriptors)?;
    for area in areas {
        let at = laid
            .total
            .checked_add(area.reserve)
            .and_then(align_up)
            .ok_or(TooLarge::Overflow)?;
        laid.total = align_up(area.bytes)
            .and_then(|bytes| at.checked_add(bytes))
            .ok_or(TooLarge::Overflow)?;
        laid.areas.push(at);
    }
    if laid.total > MAX_BYTES {
        return Err(TooLarge::Bound {
            bytes: laid.total,
            limit: MAX_BYTES,
        });
    }
    Ok(laid)
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

    /// Within a launch across blocks, block `b` owns row `b` of every value
    /// of a row class: a value that takes another's bytes there lies at the
    /// other's offset, so no block's row lands on another block's. Values
    /// that share across launches may lie anywhere over each other. With
    /// `chunk`, laid in pieces of it ([`layout_chunked`]): two values share
    /// bytes only in one piece, and the rule holds there.
    #[test]
    fn rows_of_one_class_share_only_at_one_offset() {
        for chunk in [None, Some(2048), Some(4096)] {
            rows_of_one_class_share_only_at_one_offset_in(chunk);
        }
    }

    fn rows_of_one_class_share_only_at_one_offset_in(chunk: Option<u64>) {
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

    fn area(def: u32, last: u32, reserve: u64, bytes: u64) -> Area {
        Area {
            life: life(def, last),
            reserve,
            bytes,
        }
    }

    /// Random values and areas: no value alive with an area shares a byte
    /// of it or of its reserve, nor two areas alive together, nor two
    /// values alive together, and every item lies inside the scratch.
    #[test]
    fn no_item_alive_with_another_shares_its_bytes() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |bound: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % bound
        };
        let overlap = |a: Lifetime, b: Lifetime| a.last >= b.def && b.last >= a.def;
        for round in 0..20_000 {
            let count = 1 + next(10) as usize;
            let mut descriptors = Vec::new();
            let mut lifetimes = Vec::new();
            for _ in 0..count {
                descriptors.push(desc(1 + next(4096) as u32));
                let def = next(8) as u32;
                let mut value = life(def, def + next(4) as u32);
                value.reusable = next(6) != 0;
                value.dead = next(10) == 0;
                lifetimes.push(value);
            }
            let areas: Vec<Area> = (0..next(4))
                .map(|_| {
                    let def = next(8) as u32;
                    let reserve = [0, SELECT_RESERVE, 100][next(3) as usize];
                    area(def, def, reserve, next(40_000))
                })
                .collect();
            let laid = if round % 2 == 0 {
                reusing_with(&descriptors, &lifetimes, 0, 0, &areas).unwrap()
            } else {
                let chunked =
                    chunked_with(&descriptors, &lifetimes, 0, 0, &areas, PIECES).unwrap();
                // One piece's worth of items is all the check below reads.
                if chunked.pieces() != 1 {
                    continue;
                }
                Layout {
                    values: chunked.values.iter().map(|placed| placed.offset).collect(),
                    temporary: chunked.temporary.offset,
                    temporary_bytes: chunked.temporary_bytes,
                    areas: chunked.areas.iter().map(|placed| placed.offset).collect(),
                    total: chunked.ends[0],
                }
            };
            let mut items: Vec<(u64, u64, Lifetime, bool)> = (0..count)
                .filter(|&i| !lifetimes[i].dead)
                .map(|i| {
                    let at = laid.values[i];
                    (at, at + descriptors[i].device_bytes(), lifetimes[i], lifetimes[i].reusable)
                })
                .collect();
            items.extend(areas.iter().zip(&laid.areas).map(|(area, &first)| {
                (first - area.reserve, first + area.bytes, area.life, true)
            }));
            for (x, &(start, end, life, ours)) in items.iter().enumerate() {
                assert!(start >= DUMMY_BYTES && end <= laid.total, "{laid:?}");
                for &(s, e, them, theirs) in &items[x + 1..] {
                    // A value that may go unwritten in part takes no bytes
                    // any item held before it.
                    let fresh = (!theirs && them.def > life.last) || (!ours && life.def > them.last);
                    if overlap(life, them) || fresh {
                        assert!(end <= s || e <= start, "{items:?}");
                    }
                }
            }
        }
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

}
