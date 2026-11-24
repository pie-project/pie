use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::channel::ChannelId;
use crate::chunk::{ChunkFacts, ShapeId, SlotKind};
use crate::error::{Error, Result};
use crate::program::{InstanceId, ScratchPlace};

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct FireId(pub u64);

impl std::fmt::Display for FireId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct FrameId(pub u64);

impl std::fmt::Display for FrameId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// One row's mask, run-length: runs alternate hidden and admitted columns,
/// hidden first. Column `c` of a lane's mask is slot `kv.folded + c` of its
/// working set ([`Lane::mask`]); `total` is how many columns it states, at
/// least the lane's extent (its buffer, then its own rows).
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Mask {
    pub runs: Vec<u32>,
    pub total: u64,
}

impl Mask {
    #[must_use]
    pub fn new(runs: Vec<u32>, total: u64) -> Mask {
        Mask { runs, total }
    }

    #[must_use]
    pub fn len(&self) -> u64 {
        self.total
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    #[must_use]
    pub fn words(&self) -> usize {
        usize::try_from(self.total.div_ceil(32)).unwrap_or(usize::MAX)
    }

    /// The mask without its first `columns` columns: what a piece whose
    /// boundary stands `columns` slots further on reads of the slots past
    /// it.
    #[must_use]
    pub fn after(&self, columns: u64) -> Mask {
        let columns = columns.min(self.total);
        let mut runs = Vec::new();
        let mut at = columns;
        for span in self.admitted(columns..self.total) {
            runs.push(u32::try_from(span.start - at).unwrap_or(u32::MAX));
            runs.push(u32::try_from(span.end - span.start).unwrap_or(u32::MAX));
            at = span.end;
        }
        Mask {
            runs,
            total: self.total - columns,
        }
    }

    pub fn admitted(&self, range: Range<u64>) -> impl Iterator<Item = Range<u64>> + '_ {
        let mut at = 0u64;
        self.runs
            .iter()
            .enumerate()
            .filter_map(move |(index, &run)| {
                let span = at..at.saturating_add(u64::from(run)).min(self.total);
                at = span.end;
                (index % 2 == 1).then_some(span)
            })
            .map(move |span| span.start.max(range.start)..span.end.min(range.end))
            .filter(|span| !span.is_empty())
    }

    pub fn expand_into(&self, dst: &mut [u32]) {
        let total = usize::try_from(self.total).unwrap_or(usize::MAX);
        let mut at = 0usize;
        for (index, &run) in self.runs.iter().enumerate() {
            let end = at.saturating_add(run as usize).min(total);
            if index % 2 == 1 {
                for bit in at..end {
                    if let Some(word) = dst.get_mut(bit / 32) {
                        *word |= 1 << (bit % 32);
                    }
                }
            }
            if end == total {
                break;
            }
            at = end;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Masking {
    Extent(Mask),
    Rows(Vec<Mask>),
}

impl Masking {
    #[must_use]
    pub fn masks(&self) -> &[Mask] {
        match self {
            Masking::Extent(mask) => std::slice::from_ref(mask),
            Masking::Rows(rows) => rows,
        }
    }

    #[must_use]
    pub fn stated_rows(&self) -> Option<usize> {
        match self {
            Masking::Extent(_) => None,
            Masking::Rows(rows) => Some(rows.len()),
        }
    }
}

/// The buffered slots a lane's recurrent layers replay, as slots: every one
/// of `folded..first` for a lane alone on its set ([`Lineage::Set`]), the
/// ones its mask admits for a lane sharing its set ([`Lineage::Admitted`]).
/// A lane's own rows follow, in order, whatever its mask says: the mask is
/// attention's. One run serves all of a lane's rows, so where the mask
/// names the lineage every row must admit the same buffered slots.
/// `folded` is the lane's boundary, the origin of its mask's columns
/// ([`Lane::mask`]): the buffered slots are columns `0..first - folded`.
pub fn buffered_reads(
    masking: Option<&Masking>,
    folded: u32,
    first: u32,
    lineage: Lineage,
) -> Result<Vec<Range<u32>>> {
    let whole: Vec<Range<u32>> = (folded < first)
        .then_some(folded..first)
        .into_iter()
        .collect();
    let (Lineage::Admitted, Some(masking)) = (lineage, masking) else {
        return Ok(whole);
    };
    let buffer = u64::from(first.saturating_sub(folded));
    let read = |mask: &Mask| -> Vec<Range<u32>> {
        mask.admitted(0..buffer)
            .map(|span| (span.start as u32 + folded)..(span.end as u32 + folded))
            .collect()
    };
    let masks = masking.masks();
    let Some(reads) = masks.first().map(read) else {
        return Ok(whole);
    };
    for (row, mask) in masks.iter().enumerate() {
        if read(mask) != reads {
            return Err(Error::Invalid(format!(
                "row {row} admits other buffered slots than the lane's first row; its \
                 recurrent layers replay one run of them for all its rows"
            )));
        }
    }
    Ok(reads)
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvDelta {
    pub held: u32,
    /// The set's committed boundary, in the set's own slot numbering. Every
    /// slot below it is committed: the lane's rows read `[0, folded)` under
    /// the model's own bounds (causal, window, sink) and no mask bit. The
    /// lane's mask speaks of the slots from `folded` on and of nothing else.
    /// `folded <= held`: a lane writes no committed slot.
    ///
    /// It is the boundary as the runtime knows it when it stages the fire.
    /// While a device-counted fold is in flight it is the boundary before
    /// that fold; the slots the fold freezes are then read through the mask
    /// like any buffered slot. A lane that carries [`RsVerb::Buffer`] states
    /// the same `folded` there.
    #[serde(default)]
    pub folded: u32,
    /// On a lane whose program places its rows ([`ListKind::Translation`]):
    /// one past the last slot a row of it may land at, what the runtime
    /// made the lane's own before the fire. Its rows land in
    /// [`KvDelta::writable`], `[folded, write_end)`; every page of the
    /// translation outside it may be read by another working set or the
    /// prefix cache. An engine refuses the lane by name for a row its
    /// program places outside it ([`misplaced_write`]), never the frame,
    /// and never writes the row. Not read on a lane whose rows the host
    /// places.
    #[serde(default)]
    pub write_end: u32,
    /// One past the slots of the set the runtime knows written as it
    /// stages the fire: by the set's own fires, or in the prefix it shares.
    /// A page holds what its last owner left until its set writes it, so
    /// no row reads a slot at or past it that the fire does not write
    /// first, without a gap ([`crate::extent`]). Where the device places
    /// the set's rows it is the end as of the fires that settled.
    #[serde(default)]
    pub readable: u32,
    /// The working set the lane's rows land in, one token for the set's
    /// life, so an engine keeps a set's written end across its fires
    /// ([`crate::extent::Written`]). 0 names no set: a lane the runtime
    /// stages for a set always names it; a lane naming none (an engine's
    /// own fire, or a test's) is not held to `readable`.
    #[serde(default)]
    pub set: u64,
}

impl KvDelta {
    /// The slots a lane whose program places its rows may write.
    #[must_use]
    pub fn writable(&self) -> Range<u32> {
        self.folded..self.write_end
    }
}

/// Why a lane whose program places its rows at `slots` is refused, where
/// one lands outside `writable` ([`KvDelta::writable`]): the first such
/// slot, named. `None`: every row lands in what the lane may write.
#[must_use]
pub fn misplaced_write(writable: Range<u32>, slots: &[u32]) -> Option<String> {
    let &slot = slots.iter().find(|slot| !writable.contains(slot))?;
    Some(format!(
        "its program places a row at kv slot {slot}, outside the slots it may write \
         ({}..{}): below is its frozen prefix, which other working sets may read, and past \
         is not its own",
        writable.start, writable.end
    ))
}

/// A lane's recurrent-state record, as the runtime states it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum LaneRecord {
    /// Not stated: the load holds no recurrent state. An engine whose load
    /// holds state refuses a lane that states none.
    #[default]
    Unstated,
    /// The lane carries no state: it reads and writes the null record (ref
    /// 0 a piece).
    Null,
    /// The lane's slot refs of each fixed shape, one `[run, cow]` a piece,
    /// in piece order. `cow == run` writes the record in place (the normal
    /// case); `cow != run` has the engine copy `run` onto `cow` on the
    /// stream before the fire, which then reads and writes `cow` (a fork, or
    /// a snapshot restored, within the fire).
    Pieces(Vec<(ShapeId, Vec<[u32; 2]>)>),
}

impl LaneRecord {
    /// The pieces a record names; none for [`LaneRecord::Null`] and
    /// [`LaneRecord::Unstated`].
    #[must_use]
    pub fn pieces(&self) -> &[(ShapeId, Vec<[u32; 2]>)] {
        match self {
            LaneRecord::Pieces(pieces) => pieces,
            LaneRecord::Null | LaneRecord::Unstated => &[],
        }
    }
}

/// What a lane's kv list table holds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ListKind {
    /// The lane's kv pages in token order: entry `i` holds its rows
    /// `[i * T, (i + 1) * T)`.
    #[default]
    Pages,
    /// The working set's pages its program addresses by page index (a
    /// device-geometry instance's lane): entry `i` is the page index `i`
    /// names, the rows' order the program's.
    Translation,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Readout {
    #[default]
    Last,
    Rows(Vec<u32>),
    None,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lane {
    pub slot: u32,
    pub word: u64,
    pub tokens: Vec<u32>,
    pub positions: Vec<u32>,
    pub kv: KvDelta,
    /// What the lane's rows read of the slots from `kv.folded` on. Column
    /// `c` of a mask is slot `kv.folded + c`. The extent is `end -
    /// kv.folded` columns, `end` one past the lane's last row: the buffer
    /// the fire found, then the lane's own rows in row order. A set bit
    /// admits the slot, a clear bit hides it. A mask's `total` is at least
    /// the extent; columns past it are not read. A plane is `rows x (buffer
    /// + rows)` bits and no more.
    ///
    /// Among the lane's own rows the order is the row order: row `q` reads
    /// own rows `0..=q` its bits admit and no later row, whatever its bits
    /// or its stated position say. A wide lane's rows read every own row
    /// their bits admit. Bits of own rows above the diagonal are not read.
    ///
    /// Lanes of one set in one fire read one buffer, as the fire found it,
    /// and their own rows: a slot a sibling lane writes in the same fire is
    /// hidden by the guest's mask.
    ///
    /// The mask is attention's: a lane alone on its set replays every
    /// buffered slot of its set; lanes sharing a set replay the slots their
    /// mask admits, the ancestry they state ([`Lineage`]).
    ///
    /// `None`: every row reads every slot before its own and its own.
    /// `Masking::Extent`: one mask for every row. `Masking::Rows`: one per
    /// row, as many as the lane's rows. A kv-less lane's mask is over its
    /// attention group's rows, as before.
    ///
    /// The mask is the lane's data, never its class: a lane stating one
    /// lands on the class it would land on stating none (`attn.attend`
    /// reads no `fact.Mask`), and shares a fire, and a body, with lanes
    /// stating none; each reads its own entry of the fire's masks
    /// (`RuntimeInput::Mask`). An engine whose kernels do not read a lane's
    /// mask where its class attends refuses the lane by name, never serves
    /// it unmasked.
    pub mask: Option<Masking>,
    pub adapter: Option<u32>,
    #[serde(default)]
    pub rs: RsVerb,
    #[serde(default)]
    pub rs_reset: RsReset,
    /// Rows, ascending inside the lane, whose recurrent state the fire lands
    /// in slots of their own as it runs.
    #[serde(default)]
    pub rs_snapshots: Vec<RsSnapshot>,
    /// The lane's replay as the runtime stated it when it chose the fire's
    /// rung ([`LaneReplay`]): what an engine's submit check compares its own
    /// count with ([`replay_agrees`]). `None` on a lane carrying no
    /// recurrent verb.
    #[serde(default)]
    pub replay_stated: Option<LaneReplay>,
    pub readout: Readout,
    /// Unique to its pass in a fire, so two passes' lanes never attend each other.
    #[serde(default)]
    pub group: Option<u32>,
    /// How many more fires the frame this lane's fire belongs to sends
    /// after it (`None`: not part of a frame). An engine that runs a device
    /// token loop past its rows stops where the frame does.
    #[serde(default)]
    pub fires_after: Option<u32>,
    #[serde(default)]
    pub inputs: Vec<InputFeed>,
    /// The lane takes no seat; the state it carries, if any, is the record its fixed table names.
    #[serde(default)]
    pub kv_less: bool,
    /// An engine that cannot report says nothing, and the step is plainly
    /// committed.
    #[serde(default)]
    pub reports: u64,
    /// Pages of this lane's to read out as the fire computes them; only to
    /// an engine that taps ([`crate::Engine::tap_groups`]).
    #[serde(default)]
    pub tap: Option<Tap>,
    /// The lane's slot refs of the kv list, of the window buffer
    /// ([`crate::SlotRole::WindowBuffer`]) and of each window list, in token
    /// order, by the shapes of the engine's [`crate::chunk::ChunkFacts`]
    /// (pinned shapes' refs where it pins its pools); a list the lane holds
    /// no slot of is left out. The window buffer's table starts at
    /// [`WindowFold::base`]: entry `i` holds rows `[base + i * T, base + (i
    /// + 1) * T)` (`T` its shape's tokens), every entry a real slot
    /// (CONTRACT-BC 19, 21).
    ///
    /// Window lists' tables: removed by the windowed-state change; do not
    /// use in new code. A window list's table is in its own token order,
    /// entry `i` its rows `[i * T_w, (i + 1) * T_w)` (`T_w` its shape's
    /// tokens), a slot trimmed behind its window ref 0.
    #[serde(default)]
    pub tables: Vec<(ShapeId, Vec<u32>)>,
    /// What the kv list's table in `tables` holds: the rows' pages, or a
    /// translation its program's page indexes address.
    #[serde(default)]
    pub list: ListKind,
    /// A lane stating its own page list (`state-geometry.pages`): its
    /// working set's whole kv list table (slot refs), whose pages past the
    /// frozen full ones its list's entries name. Empty for any other lane.
    #[serde(default)]
    pub reserve: Vec<u32>,
    /// The lane's recurrent-state record. The runtime states one for every
    /// lane of a load that holds recurrent state, a record or none. On a
    /// load with windowed layers served as ring + buffer it holds their
    /// rings too ([`crate::ChunkFacts::windowed`]), and `cow != run` is also
    /// the copy-on-first-fold of a record another owner shares (CONTRACT-BC
    /// 19): the whole record, rings and rs state together, since the fold
    /// changes both.
    #[serde(default)]
    pub record: LaneRecord,
    /// How the lane's windowed layers stand as ring + buffer and how far
    /// this fire folds them (CONTRACT-BC 19, 21). Stated on every lane of a
    /// load whose engine states [`crate::ChunkFacts::windowed`]; `None`
    /// elsewhere, and on a kv-less lane, which carries no attention rows.
    #[serde(default)]
    pub window: Option<WindowFold>,
    /// Whole-slot copies `[src, dst]` (logical slot refs) of each list
    /// shape, made on the stream before the fire runs, after the fires
    /// before it: the rs list, where a lane writes buffered rows into a
    /// slot it shared with a fork (CONTRACT-BC 18; its
    /// `RsVerb::Buffer::pages` already name `dst`), and the window buffer,
    /// where a lane writes rows into a partly filled slot it shares with a
    /// fork or a prefix-cache entry (CONTRACT-BC 21; its table already
    /// names `dst`). A window list's copies (where a lane writes rows into
    /// a window-list slot it shares): removed by the windowed-state change;
    /// do not use in new code. Every lane's copies of
    /// a fire land before any lane of it reads or writes; no two name the
    /// same `dst`.
    #[serde(default)]
    pub slot_copies: Vec<(ShapeId, Vec<[u32; 2]>)>,
}

/// Pool pages `pages[i]` read into host kv slots `slots[i]` as the fire
/// writes them, a group of their planes at a time (a layer's), each group
/// told to the engine's tap sink under `tag` as it lands.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tap {
    pub tag: u64,
    pub pages: Vec<u32>,
    pub slots: Vec<u32>,
}

/// The channel a lane feeds guest input `id` from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InputFeed {
    pub id: u32,
    pub channel: ChannelId,
}

impl Lane {
    #[must_use]
    pub fn decode(slot: u32, word: u64, token: u32, held: u32) -> Lane {
        Lane {
            slot,
            word,
            tokens: vec![token],
            kv: KvDelta {
                held,
                ..KvDelta::default()
            },
            ..Lane::default()
        }
    }

    #[must_use]
    pub fn rows(&self) -> u32 {
        u32::try_from(self.tokens.len()).unwrap_or(u32::MAX)
    }

    pub fn validate(&self) -> Result<()> {
        self.validate_for(Serves::NONE)
    }

    pub fn validate_for(&self, serves: Serves) -> Result<()> {
        if !self.positions.is_empty() && self.positions.len() != self.tokens.len() {
            return Err(Error::Invalid(format!(
                "lane in slot {} has {} positions for {} tokens",
                self.slot,
                self.positions.len(),
                self.tokens.len()
            )));
        }
        if let Readout::Rows(rows) = &self.readout
            && let Some(&row) = rows.iter().find(|&&r| r >= self.rows())
        {
            return Err(Error::Invalid(format!(
                "lane in slot {} reads row {row} of the {} it has",
                self.slot,
                self.rows()
            )));
        }
        // A kv-less lane's mask is over its attention group's rows.
        if let Some(masking) = &self.mask
            && !self.kv_less
            && let Some(stated) = masking.stated_rows()
            && stated != self.tokens.len()
        {
            return Err(Error::Invalid(format!(
                "lane in slot {} states {stated} per-row masks for the {} rows it carries",
                self.slot,
                self.rows()
            )));
        }
        if !matches!(self.rs, RsVerb::Fold) && !serves.rs_verbs {
            return Err(Error::unsupported("engine", RS_VERBS_WITHOUT_DEVICE_HALF));
        }
        if !self.rs_snapshots.is_empty() {
            if self.rs_snapshots.len() > serves.rs_snapshots as usize {
                return Err(Error::Invalid(format!(
                    "lane in slot {} snapshots its recurrent state at {} rows, past the {} \
                     this engine lands in one fire",
                    self.slot,
                    self.rs_snapshots.len(),
                    serves.rs_snapshots
                )));
            }
            let ascending = self
                .rs_snapshots
                .iter()
                .try_fold(0, |at, snap| (at < snap.row).then_some(snap.row))
                .is_some_and(|last| last < self.rows());
            if !ascending || !matches!(self.rs, RsVerb::Fold) {
                return Err(Error::Invalid(format!(
                    "lane in slot {} snapshots its recurrent state at rows {:?} of its {}: a \
                     snapshot is an interior row, ascending, of a lane that folds every row",
                    self.slot,
                    self.rs_snapshots
                        .iter()
                        .map(|snap| snap.row)
                        .collect::<Vec<_>>(),
                    self.rows()
                )));
            }
        }
        self.validate_window(serves)?;
        self.validate_chunk_refs(serves)
    }

    /// A lane's windowed layers (CONTRACT-BC 21): stated only to an engine
    /// that serves them as ring + buffer; the buffer starts at or below the
    /// lane's folded index after the fire; on a hybrid the rs fold is the
    /// same fold.
    fn validate_window(&self, serves: Serves) -> Result<()> {
        let Some(window) = &self.window else {
            return Ok(());
        };
        if !serves.windowed_state {
            return Err(Error::unsupported("engine", WINDOW_WITHOUT_RING));
        }
        let after = match window.fold {
            FoldLen::Host(rows) => self.kv.folded.saturating_add(rows),
            FoldLen::Device => self.kv.folded,
        };
        if window.base > after {
            return Err(Error::Invalid(format!(
                "lane in slot {} starts its window buffer at row {}, past its folded \
                 index after the fire {after}: the rows below it are in the ring",
                self.slot, window.base
            )));
        }
        // Rows earlier fires buffered (the set holds rows past its folded
        // index before this fire's) stay named for the fire's reads: the
        // table starts at or below the folded index before the fire (with
        // `base` on a slot's first row, at most it rounded down).
        if self.kv.held > self.kv.folded && window.base > self.kv.folded {
            return Err(Error::Invalid(format!(
                "lane in slot {} carries rows {}..{} earlier fires buffered and starts its \
                 window buffer at row {}, past them: they stay named until the fold frees \
                 them",
                self.slot, self.kv.folded, self.kv.held, window.base
            )));
        }
        if let RsVerb::Buffer {
            folded,
            fold,
            lineage,
            ..
        } = &self.rs
            && (*folded != self.kv.folded || *fold != window.fold || *lineage != window.lineage)
        {
            return Err(Error::Invalid(format!(
                "lane in slot {} folds its recurrent layers from {folded} by {fold:?} \
                 ({lineage:?}) and its windowed layers from {} by {:?} ({:?}): a lane \
                 folds once, one folded index for both",
                self.slot, self.kv.folded, window.fold, window.lineage
            )));
        }
        Ok(())
    }

    fn validate_chunk_refs(&self, serves: Serves) -> Result<()> {
        if self.tables.is_empty()
            && matches!(self.record, LaneRecord::Unstated)
            && self.slot_copies.is_empty()
            && self.window.is_none()
        {
            return Ok(());
        }
        if !serves.chunks {
            return Err(Error::unsupported("engine", CHUNK_REFS_WITHOUT_CHUNKS));
        }
        let shapes = self
            .tables
            .iter()
            .map(|(shape, _)| *shape)
            .chain(self.record.pieces().iter().map(|(shape, _)| *shape));
        let mut seen: Vec<ShapeId> = Vec::new();
        for shape in shapes {
            if seen.contains(&shape) {
                return Err(Error::Invalid(format!(
                    "lane in slot {} names shape {shape} twice in its slot tables",
                    self.slot
                )));
            }
            seen.push(shape);
        }
        if let Some((shape, _)) = self
            .record
            .pieces()
            .iter()
            .find(|(_, pieces)| pieces.is_empty())
        {
            return Err(Error::Invalid(format!(
                "lane in slot {} names fixed shape {shape} with no piece",
                self.slot
            )));
        }
        Ok(())
    }

    /// Checks the lane's slot refs against the engine's chunks: each shape
    /// stated, of the kind its table says, each piece of a fixed shape
    /// named, every ref inside the pool (or the shape's pinned run) and off
    /// the reserved chunks, but a window list's null ref 0 in its table; a
    /// lane's windowed layers stated exactly where the engine states them,
    /// its window buffer's base on a slot's first row.
    pub fn validate_chunks(&self, facts: &ChunkFacts) -> Result<()> {
        match (&self.window, facts.window_buffer()) {
            (Some(_), None) => {
                return Err(Error::Invalid(format!(
                    "lane in slot {} states windowed layers the engine does not state",
                    self.slot
                )));
            }
            (None, Some(_)) if !self.kv_less => {
                return Err(Error::Invalid(format!(
                    "lane in slot {} states no window fold on a load with windowed layers",
                    self.slot
                )));
            }
            (Some(window), Some(shape)) => {
                if let SlotKind::List { tokens, .. } = shape.kind
                    && !window.base.is_multiple_of(tokens)
                {
                    return Err(Error::Invalid(format!(
                        "lane in slot {} starts its window buffer at row {}, inside a slot \
                         of {tokens} rows",
                        self.slot, window.base
                    )));
                }
            }
            _ => {}
        }
        for (id, pieces) in self.record.pieces() {
            if let Some(SlotKind::Fixed { pieces: stated }) = facts.shape(*id).map(|s| s.kind)
                && pieces.len() != stated as usize
            {
                return Err(Error::Invalid(format!(
                    "lane in slot {} names {} piece(s) of fixed shape {id}, which has {stated}",
                    self.slot,
                    pieces.len()
                )));
            }
        }
        let fixed: Vec<(ShapeId, Vec<u32>)> = self
            .record
            .pieces()
            .iter()
            .map(|(shape, pieces)| (*shape, pieces.iter().flatten().copied().collect()))
            .collect();
        let copies: Vec<(ShapeId, Vec<u32>)> = self
            .slot_copies
            .iter()
            .map(|(shape, pairs)| (*shape, pairs.iter().flatten().copied().collect()))
            .collect();
        // Each shape's refs, whether they are fixed pieces, and whether a
        // null ref 0 may stand among them (a window list's table).
        let refs = self
            .tables
            .iter()
            .map(|(shape, refs)| (*shape, refs.as_slice(), false, true))
            .chain(
                copies
                    .iter()
                    .map(|(shape, refs)| (*shape, refs.as_slice(), false, false)),
            )
            .chain(
                fixed
                    .iter()
                    .map(|(shape, refs)| (*shape, refs.as_slice(), true, false)),
            );
        for (id, refs, fixed, table) in refs {
            let Some(shape) = facts.shape(id) else {
                return Err(Error::Invalid(format!(
                    "lane in slot {} names shape {id}, which the engine does not state",
                    self.slot
                )));
            };
            let trims = table && shape.kind.window().is_some();
            if fixed == shape.kind.per_token() {
                return Err(Error::Invalid(format!(
                    "lane in slot {} names {:?} shape {id} as {}",
                    self.slot,
                    shape.kind,
                    if fixed { "fixed" } else { "a table" }
                )));
            }
            let pool = shape.pinned.map_or(
                crate::chunk::ChunkRun {
                    first: 0,
                    chunks: facts.total_chunks,
                },
                |pinned| pinned.device,
            );
            for &slot in refs {
                if trims && slot == 0 {
                    continue;
                }
                let chunk = shape.chunk_of(slot);
                if !pool.contains(chunk) || facts.is_reserved(chunk) {
                    return Err(Error::Invalid(format!(
                        "lane in slot {} names slot {slot} of shape {id}, in chunk {chunk}, \
                         which holds none of its slots",
                        self.slot
                    )));
                }
            }
        }
        Ok(())
    }
}

const CHUNK_REFS_WITHOUT_CHUNKS: &str =
    "Lane::tables and Lane::fixed against an engine that states no chunks";

const WINDOW_WITHOUT_RING: &str =
    "Lane::window: this engine does not serve windowed layers as ring + buffer";

const RS_VERBS_WITHOUT_DEVICE_HALF: &str =
    "Lane::rs beyond RsVerb::Fold: this engine has no device half for it";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Serves {
    pub rs_verbs: bool,
    pub adapter_attachments: bool,
    pub rs_snapshots: u32,
    /// The engine has loaded: it states [`crate::chunk::ChunkFacts`] and
    /// reads a lane's slot tables.
    pub chunks: bool,
    /// The engine serves windowed layers as ring + buffer
    /// ([`crate::Capabilities::windowed_state`]): it reads [`Lane::window`].
    pub windowed_state: bool,
}

impl Serves {
    pub const NONE: Serves = Serves {
        rs_verbs: false,
        adapter_attachments: false,
        rs_snapshots: 0,
        chunks: false,
        windowed_state: false,
    };

    #[must_use]
    pub fn of(caps: &crate::Capabilities) -> Serves {
        Serves {
            rs_verbs: caps.rs_verbs,
            adapter_attachments: caps.adapter_attachments,
            rs_snapshots: caps.rs_snapshots,
            chunks: true,
            windowed_state: caps.windowed_state,
        }
    }
}

/// The state after the lane's first `row` rows, landed in a record of its
/// own: state slot `slot`, or, to an engine that states chunks, the fixed
/// shape's slot ref of the record's first piece. `pieces` names every piece
/// of that record, in piece order, where it takes more than one; empty, the
/// record is `slot` alone. `fixed` names the record as [`Lane::fixed`]
/// names a lane's, every fixed shape's slot refs, a piece each (an engine
/// whose record spans two fixed shapes reads it; CONTRACT-BC 16).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RsSnapshot {
    pub row: u32,
    pub slot: u32,
    #[serde(default)]
    pub pieces: Vec<u32>,
    #[serde(default)]
    pub fixed: Vec<(ShapeId, Vec<u32>)>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RsReset {
    #[default]
    Inferred,
    Fresh,
    Held,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RsVerb {
    #[default]
    Fold,
    /// Replays the buffered slots `folded..first` its lineage names, then
    /// its own rows in order, then freezes `fold` slots past the row the
    /// record stands at into the state.
    Buffer {
        /// The buffer's pages: token `t` of page `p` holds slot
        /// `base + p * page + t`.
        pages: Vec<u32>,
        base: u32,
        /// How far the record is folded, as the runtime knew it when it
        /// staged the fire. Exact for a set whose folds are the host's
        /// ([`FoldLen::Host`]): the runtime counts those as it stages them.
        /// For a set whose fold length is the device's
        /// ([`FoldLen::Device`]) it is a lower bound: such a fold's count
        /// reaches the runtime only when its fire settles
        /// ([`crate::KvWrites::fold`]), and the set's later fires are
        /// staged before that. The record is the engine's: it replays from
        /// where its record stands, `max(folded, the row its last fold left
        /// the record at)`, and freezes `fold` slots past that row. The
        /// engine keeps that row with the record across every copy of the
        /// record's slot it makes on the runtime's behalf (a fork's copy);
        /// it need not survive a move between tiers or an export of the
        /// record's bytes: the runtime does not fork, checkpoint, export or
        /// move between tiers a set with such a fire on its way until that
        /// fire settles, so wherever the record leaves the engine's table
        /// the stated number is exact again.
        folded: u32,
        fold: FoldLen,
        /// Which buffered slots the lane's recurrence replays: a lane alone
        /// on its set replays every buffered slot of its set; lanes sharing
        /// a set replay the slots their mask admits, the ancestry they state.
        lineage: Lineage,
    },
}

/// A lane's windowed (sliding-window attention) layers as state + buffer,
/// as a recurrent layer is (CONTRACT-BC 19, 21). The lane's folded index is
/// [`KvDelta::folded`], the one index its recurrent layers fold by too
/// ([`RsVerb::Buffer`] states the same `folded`, `fold` and `lineage`; the
/// windowed-state integration lifts one lane-level fold out of it, so rs
/// and window read one field, and until then [`Lane::validate`] holds the
/// two equal):
///
/// - The ring: rows below `folded`, the last W of each windowed layer, in
///   the lane's record ([`Lane::record`], [`crate::ChunkFacts::windowed`]).
/// - The buffer: the rows still unfolded after the fire, in the window
///   buffer's slots the lane's table names ([`crate::SlotRole::WindowBuffer`],
///   [`Lane::tables`]), entry `i` rows `[base + i * T, base + (i + 1) * T)`,
///   every entry a real slot. Rows the fire itself folds take no buffer
///   slot: their K/V live in the fire's own activation while the fire reads
///   them, and only the ring receives them at the fold. A fire's activation
///   holds its own folded rows' K/V for its reads (an engine sizes its
///   activation for it; the runtime cuts a fire whose activation cannot, as
///   chunked prefill), so a long prefill holds at most about one fire's
///   tail of slots.
/// - Attention: row `p` of a windowed layer of window W reads keys
///   `[max(0, p - W + 1), p]`: those below `folded` from the ring, those
///   from `folded` on from the buffer rows its lineage names
///   ([`buffered_reads`]) and its own rows, under one mask, one softmax over
///   the union.
/// - The fold: advancing `folded` from f to f' writes rows `[f, f')`
///   (buffered, or the fire's own) into the ring at `row mod W`, past rows dropping out, in the fold step
///   the recurrent layers fold in. Lanes of a group naming one record fold
///   it once (CONTRACT-BC 12, 15).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WindowFold {
    /// The row entry 0 of the lane's window-buffer table holds, on a slot's
    /// first row. A lane carrying rows earlier fires buffered (its first row
    /// past `folded`): `folded`, the index before the fire, rounded down to
    /// `T`; those rows stay named for this fire's reads, and the fold frees
    /// their slots after. A lane carrying none: the folded index after the
    /// fire (`folded` plus a host-counted fold; `folded` for a
    /// device-counted one), rounded down to `T`, since rows the fire
    /// computes and folds itself take no slot.
    pub base: u32,
    /// How far this fire advances `folded`, the rows it folds into the
    /// rings: [`FoldLen`] as [`RsVerb::Buffer`]'s `fold` reads it (a
    /// device-counted fold's count reaches the runtime when the fire
    /// settles, and the engine keeps the row its rings stand at with the
    /// record, as for the recurrent state).
    pub fold: FoldLen,
    /// Which buffered rows the lane's windowed layers read, as its
    /// recurrent layers replay them: a lane never reads a sibling's
    /// divergent rows.
    pub lineage: Lineage,
}

/// Which of its set's buffered slots a lane's recurrent layers replay
/// ([`RsVerb::Buffer`]). The mask is attention's; a recurrence reads a row
/// of its lineage in order, and its own rows always.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Lineage {
    /// The lane is alone on its set in the fire: every buffered slot.
    Set,
    /// The lane shares its set with other lanes of the fire: the slots its
    /// mask admits, the ancestry it states.
    Admitted,
}

/// What one activation rung's replay scratch holds, as the engine states it
/// for that rung ([`crate::chunk::ChunkFacts::act_replay`]), from its own
/// reserve formula: every engine whose rungs replay states it, no default.
/// A fire's rows are counted by [`ReplayRows`]; it runs at a rung that holds
/// them ([`rung_holds`]), the smallest such rung the runtime finds, and one
/// no rung holds is refused by name before anything is submitted. An
/// engine's own check at submit calls the same [`rung_holds`] against the
/// rung it stages the fire in.
///
/// - `own_max`: the fire's own rows, every row its lanes compute in it
///   (`None`: not binding).
/// - `carried_max`: its carried rows counted as a rung counts them
///   ([`lane_carried`]: a decode lane's none, a wider lane's all), summed
///   (`None`: not binding).
/// - `scratch`: its rs scratch's bytes ([`ScratchBytes`]), its scratch rows
///   ([`rs_scratch_rows`]) over its lanes.
///
/// Device geometry: a lane's carried rows are the buffered slots between its
/// state and its first row that its mask admits (`folded..first`,
/// [`buffered_reads`]). Where the device says where a lane's rows land, no
/// count exists when the runtime asks: it states a bound in its place, the
/// rows that can stand unfolded in the lane's working set when the fire runs.
/// The engine's check where the count resolves is a fault for a wrong count
/// only. An engine that takes such fires reports where each lane's rows
/// ended when the fire settles ([`crate::KvWrites::end`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RungReplay {
    pub own_max: Option<u32>,
    pub carried_max: Option<u32>,
    pub scratch: ScratchBytes,
}

/// A rung's rs scratch as bytes, the engine's own reserve formula: a fire
/// of `rows` scratch rows over `lanes` lanes (each at least one) takes, for
/// each scratch slot, `rows * slot_bytes` (at least 4) rounded up to `pad`,
/// and `lanes * lane_bytes` (at least 4) rounded up to `pad`, the whole at
/// least one `pad` (Metal's `Seat::scratch_bytes` to the byte), and fits
/// where that is at most `room_bytes`. An engine laying no rs scratch
/// beyond its rung's rows (its replay inside the rung's own rooms, bounded
/// by `own_max` and `carried_max`) states no slot: no bytes bind.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScratchBytes {
    pub slot_bytes: Vec<u64>,
    pub lane_bytes: u64,
    pub room_bytes: u64,
    pub pad: u64,
}

impl ScratchBytes {
    /// The bytes a fire of `rows` scratch rows over `lanes` lanes takes.
    #[must_use]
    pub fn bytes(&self, rows: u32, lanes: u32) -> u64 {
        if self.slot_bytes.is_empty() {
            return 0;
        }
        let pad = self.pad.max(1);
        let up = |bytes: u64| bytes.max(4).div_ceil(pad).saturating_mul(pad);
        let rows = u64::from(rows.max(1));
        let lanes = u64::from(lanes.max(1));
        let slots = self
            .slot_bytes
            .iter()
            .fold(0u64, |sum, &slot| sum.saturating_add(up(rows.saturating_mul(slot))));
        slots
            .saturating_add(up(lanes.saturating_mul(self.lane_bytes)))
            .max(pad)
    }

    /// The bytes binding, where any do.
    #[must_use]
    pub fn binds(&self) -> bool {
        !self.slot_bytes.is_empty()
    }
}

/// A fire's rows as a rung's replay scratch counts them ([`RungReplay`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayRows {
    /// Every row the fire's lanes compute in it.
    pub own: u32,
    /// Its lanes' carried rows, each counted by [`lane_carried`], summed.
    pub carried: u32,
    /// Its rs scratch rows ([`rs_scratch_rows`]).
    pub scratch: u32,
    /// Its lanes.
    pub lanes: u32,
}

impl ReplayRows {
    /// The rows of a fire whose lanes compute `(own, carried)` rows each,
    /// `carried` every buffered row a lane replays ahead of its own (a
    /// decode lane's included), its rs scratch `compact` or not (the fire's,
    /// [`scratch_compact`]).
    #[must_use]
    pub fn of_lanes(lanes: impl IntoIterator<Item = (u32, u32)> + Clone, compact: bool) -> ReplayRows {
        let own = lanes
            .clone()
            .into_iter()
            .fold(0u32, |sum, (own, _)| sum.saturating_add(own));
        ReplayRows {
            own,
            carried: lanes
                .clone()
                .into_iter()
                .fold(0u32, |sum, (own, carried)| {
                    sum.saturating_add(lane_carried(own, carried))
                }),
            scratch: rs_scratch_rows(own, lanes.clone(), compact),
            lanes: u32::try_from(lanes.into_iter().count()).unwrap_or(u32::MAX),
        }
    }

    /// The same counts from a fire's sums: `own` rows, `carried` counted by
    /// [`lane_carried`] lane by lane, `replayed` every lane's carried rows
    /// (a decode lane's included), over `lanes` lanes, its rs scratch
    /// `compact` or not (the fire's, [`scratch_compact`]).
    #[must_use]
    pub fn of_sums(own: u32, carried: u32, replayed: u32, lanes: u32, compact: bool) -> ReplayRows {
        ReplayRows {
            own,
            carried,
            scratch: own.saturating_add(if compact { carried } else { replayed }),
            lanes,
        }
    }
}

/// The rs scratch rows a fire of `own` rows takes, its lanes carrying
/// `(lane own, carried)` rows each: its own rows, and each lane's carried
/// rows as the fire's scratch lays them: where it is `compact`
/// ([`scratch_compact`]) only those a rung counts ([`lane_carried`]: a
/// decode lane's none), else every one, a decode lane's included.
#[must_use]
pub fn rs_scratch_rows(own: u32, lanes: impl IntoIterator<Item = (u32, u32)>, compact: bool) -> u32 {
    lanes.into_iter().fold(own, |sum, (lane_own, carried)| {
        sum.saturating_add(if compact {
            lane_carried(lane_own, carried)
        } else {
            carried
        })
    })
}

/// Which of a rung's stated bounds a fire passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplayBound {
    Own,
    Carried,
    Scratch,
}

/// A fire's count past one of its rung's bounds: which, the fire's count of
/// it and the rung's (rows, or the scratch's bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayPast {
    pub bound: ReplayBound,
    pub rows: u64,
    pub max: u64,
}

impl std::fmt::Display for ReplayPast {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let what = match self.bound {
            ReplayBound::Own => "own rows",
            ReplayBound::Carried => "carried recurrent rows (a decode lane's counting none)",
            ReplayBound::Scratch => {
                "bytes of rs scratch (its own rows and its lanes' carried rows, over its lanes)"
            }
        };
        write!(f, "{} {what}, past the {} the rung's replay scratch holds", self.rows, self.max)
    }
}

/// Whether `rung` holds a fire of `rows`: own, carried and scratch bytes in
/// that order, the first it passes named. What the runtime picks a fire's
/// rung by and an engine checks a fire at submit by.
pub fn rung_holds(rung: &RungReplay, rows: ReplayRows) -> std::result::Result<(), ReplayPast> {
    for (bound, count, max) in [
        (ReplayBound::Own, rows.own, rung.own_max),
        (ReplayBound::Carried, rows.carried, rung.carried_max),
    ] {
        if let Some(max) = max
            && count > max
        {
            return Err(ReplayPast {
                bound,
                rows: u64::from(count),
                max: u64::from(max),
            });
        }
    }
    let bytes = rung.scratch.bytes(rows.scratch, rows.lanes);
    if rung.scratch.binds() && bytes > rung.scratch.room_bytes {
        return Err(ReplayPast {
            bound: ReplayBound::Scratch,
            rows: bytes,
            max: rung.scratch.room_bytes,
        });
    }
    Ok(())
}

/// Whether a lane's buffered rows are one stretch its fire's rs page table
/// names ([`scratch_compact`]): the buffered rows it replays
/// ([`buffered_reads`] of its mask, its folded index, its first row and its
/// lineage, as [`RsVerb::Buffer`] states them) are one run at most, ending
/// where its own rows start, so they and the rows it lands in the buffer
/// are one adjacent stretch of the same pages. A lane whose admitted slots
/// are not one run (a beam's ancestry among its siblings') is not; one
/// whose mask cannot be read is not either.
#[must_use]
pub fn lane_one_run(masking: Option<&Masking>, folded: u32, first: u32, lineage: Lineage) -> bool {
    match buffered_reads(masking, folded, first, lineage) {
        Ok(runs) => match runs.as_slice() {
            [] => true,
            [run] => run.end == first,
            _ => false,
        },
        Err(_) => false,
    }
}

/// A lane's replay as the runtime states it ([`Lane::replay_stated`]): the
/// buffered rows it replays ahead of its own (its lineage's,
/// [`buffered_reads`], a decode lane's included) and whether they are one
/// stretch ([`lane_one_run`]), from the verb it sends.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaneReplay {
    pub replayed: u32,
    pub one_run: bool,
}

impl LaneReplay {
    /// A lane of `verb` whose first row is `first`, its mask `masking`.
    #[must_use]
    pub fn of(verb: &RsVerb, masking: Option<&Masking>, first: u32) -> Option<LaneReplay> {
        let RsVerb::Buffer {
            folded, lineage, ..
        } = verb
        else {
            return None;
        };
        let reads = buffered_reads(masking, *folded, first, *lineage).ok();
        Some(LaneReplay {
            replayed: reads
                .as_ref()
                .map_or(first.saturating_sub(*folded), |runs| {
                    runs.iter().map(|run| run.len() as u32).sum()
                }),
            one_run: lane_one_run(masking, *folded, first, *lineage),
        })
    }
}

/// The debug cross-check at an engine's door: the fire's rows and
/// compactness as its lanes state the runtime counted them
/// ([`Lane::replay_stated`]) against the engine's own (`own`,
/// `own_compact`), at the rung it stages the fire in. A disagreement in
/// compactness, or in whether `rung` holds the fire, is a contract bug,
/// named with both inputs. `layout_compact`, `per_fire` and `snapshots` are
/// [`scratch_compact`]'s; `lanes` each lane's own rows and its stated replay.
/// A fire none of whose lanes states one is not compared. Engines call it
/// in debug builds ([`replay_check_on`]).
pub fn replay_agrees(
    lanes: impl IntoIterator<Item = (u32, Option<LaneReplay>)>,
    rung: &RungReplay,
    own: ReplayRows,
    own_compact: bool,
    layout_compact: bool,
    per_fire: bool,
    snapshots: bool,
) -> std::result::Result<(), String> {
    let lanes: Vec<(u32, Option<LaneReplay>)> = lanes.into_iter().collect();
    if lanes.iter().all(|(_, stated)| stated.is_none()) {
        return Ok(());
    }
    let stated: Vec<(u32, LaneReplay)> = lanes
        .iter()
        .map(|&(rows, stated)| (rows, stated.unwrap_or_default()))
        .collect();
    let compact = scratch_compact(
        layout_compact,
        per_fire,
        stated.iter().all(|(_, lane)| lane.one_run),
        snapshots,
    );
    let rows = ReplayRows::of_lanes(
        stated.iter().map(|&(own, lane)| (own, lane.replayed)),
        compact,
    );
    let held = rung_holds(rung, rows).is_ok();
    let own_held = rung_holds(rung, own).is_ok();
    // The runtime may count more than the engine lays (a mask the device
    // resolves, a bound in place of a count): only its counting less, a
    // fire it priced compact the engine lays wide, or one it found held
    // the engine refuses, is the bug.
    if (compact && !own_compact) || (held && !own_held) {
        return Err(format!(
            "the runtime's count of this fire's replay and the engine's disagree (a contract \
             bug): stated {rows:?} compact {compact} (held {held}), the engine's {own:?} compact \
             {own_compact} (held {own_held}), at {rung:?}"
        ));
    }
    Ok(())
}

/// Whether the contract's cross-checks run: debug builds, or
/// `PIE_CHECK_REPLAY` set. It gates both of them: engines running
/// [`replay_agrees`] at their door, and the step builder checking every step
/// it forms against the load's stated set ([`crate::shapes::check_on`]). It
/// changes no behavior but to name a contract bug.
#[must_use]
pub fn replay_check_on() -> bool {
    cfg!(debug_assertions) || std::env::var_os("PIE_CHECK_REPLAY").is_some()
}

/// Whether a fire's rs scratch is compact, laying only the carried rows a
/// rung counts ([`lane_carried`]: a decode lane's none) rather than every
/// lane's: the one function both the runtime (choosing the fire's rung) and
/// the engine (checking it at submit) decide it by, from what both know. A
/// fire is compact where the load's rs layout is (`layout_compact`,
/// `ChunkFacts::rs_compact`), and, on a load whose compactness is a fire's
/// (`per_fire`, `ChunkFacts::rs_scratch_per_fire`: Metal's, its rs page
/// table read per fire), every lane's buffered rows are one stretch
/// (`every_lane_one_run`, each lane by [`lane_one_run`]) and no lane lands
/// an rs snapshot (`any_lane_snapshots`: a lane's `Lane::rs_snapshots`, an
/// auto-checkpoint's included). On a load whose compactness is not a
/// fire's (CUDA: a decoding lane's carried rows never take its scratch),
/// the layout alone decides. No engine decides compactness otherwise.
#[must_use]
pub fn scratch_compact(
    layout_compact: bool,
    per_fire: bool,
    every_lane_one_run: bool,
    any_lane_snapshots: bool,
) -> bool {
    layout_compact && (!per_fire || (every_lane_one_run && !any_lane_snapshots))
}


/// The carried rows a lane of `own` rows counts toward a rung's
/// `carried_max` ([`RungReplay`]): a
/// decode lane's (one own row, a one-token class's) none, since its decode
/// step replays them in the step's own kernels, a lane's buffered rows a
/// kernel input and no rows of the fire (at most the run-ahead window of
/// them, which the set's reservation holds); a wider lane's all of them,
/// which it replays through the rung's replay scratch.
#[must_use]
pub fn lane_carried(own: u32, carried: u32) -> u32 {
    if own == 1 { 0 } else { carried }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FoldLen {
    Host(u32),
    Device,
}

/// A pass's program on a fire: the prologue before input staging, the epilogue after the forward.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attachment {
    /// The first of the pass's lanes, `lanes` of them in a row.
    pub lane: u32,
    pub lanes: u32,
    pub instance: InstanceId,
    /// Every fire but the last of a cut pass: it carries the adapter and runs no stage.
    pub adapter_only: bool,
    /// The scratch arm the runtime took for this fire ([`crate::GuestState::scratch`]):
    /// `None` only where the instance's scratch is 0.
    #[serde(default)]
    pub scratch: ScratchPlace,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StepMedia {
    pub lane: u32,
    pub rows: Vec<u32>,
    /// Each span's patches, `3 · patch²` RGB bytes a row.
    pub patches: Vec<u8>,
    pub routes: Vec<i32>,
    pub positions: Vec<i32>,
    /// Each span's `(t, rows, cols)` patch grid.
    pub grids: Vec<i32>,
    pub token_positions: Vec<i32>,
}

impl StepMedia {
    #[must_use]
    pub fn payload_rows(&self) -> u32 {
        self.rows.iter().copied().fold(0u32, u32::saturating_add)
    }

    pub fn validate(&self, lane_rows: u32) -> Result<()> {
        let rows = self.payload_rows();
        if self.rows.is_empty() {
            return Err(Error::Invalid(format!(
                "lane {} carries a media row naming no spans; a lane with no span \
                 constructs no media row at all",
                self.lane
            )));
        }
        if rows == 0 {
            return Err(Error::Invalid(format!(
                "lane {}'s spans occupy no payload rows, and a span the tower reads \
                 nothing of has nothing to scatter",
                self.lane
            )));
        }
        for (what, have, owed) in [
            ("routes", self.routes.len(), rows as usize),
            ("grid positions", self.positions.len(), 3 * rows as usize),
        ] {
            if have != owed {
                return Err(Error::Invalid(format!(
                    "lane {}'s media carries {have} {what} for {rows} payload rows, \
                     and {owed} are owed",
                    self.lane
                )));
            }
        }
        if self.grids.len() != 3 * self.rows.len() {
            return Err(Error::Invalid(format!(
                "lane {}'s media carries {} grid entries for {} spans; each span states \
                 its `(t, rows, cols)` patch grid",
                self.lane,
                self.grids.len(),
                self.rows.len()
            )));
        }
        if !self.token_positions.is_empty() && self.token_positions.len() != 3 * lane_rows as usize
        {
            return Err(Error::Invalid(format!(
                "lane {}'s trunk rotation stream carries {} entries for {lane_rows} \
                 token rows; it is empty (scalar `(p, p, p)`) or three per row",
                self.lane,
                self.token_positions.len()
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StepVoxels {
    pub lane: u32,
    pub clips: Vec<[u32; 3]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub payload: Vec<f32>,
}

impl StepVoxels {
    #[must_use]
    pub fn voxels(&self) -> u64 {
        self.clips
            .iter()
            .map(|[t, h, w]| u64::from(*t) * u64::from(*h) * u64::from(*w))
            .sum()
    }

    pub fn validate(&self) -> Result<()> {
        if self.clips.is_empty() {
            return Err(Error::Invalid(format!(
                "lane {} carries a voxel row naming no clips; a lane with no clip \
                 constructs no voxel row at all",
                self.lane
            )));
        }
        if self.clips.iter().any(|b| b.contains(&0)) {
            return Err(Error::Invalid(format!(
                "lane {} submits a clip with a zero side",
                self.lane
            )));
        }
        if !self.payload.is_empty() && !(self.payload.len() as u64).is_multiple_of(self.voxels()) {
            return Err(Error::Invalid(format!(
                "lane {} submits {} values for {} voxels, which is not a whole row per voxel",
                self.lane,
                self.payload.len(),
                self.voxels()
            )));
        }
        Ok(())
    }
}

/// One step of a fire. A step carries at most one fire moving a given end of
/// a ring (its takes, or its puts): the engine gives each move the cell the
/// ring's cursor stands at as it builds the step, and reads no order among the
/// fires of one step.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub lanes: Vec<Lane>,
    pub attachments: Vec<Attachment>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub media: Vec<StepMedia>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub voxels: Vec<StepVoxels>,
    /// The stated shape the step is of (CONTRACT-BC 24), set by the runtime
    /// as it forms the step: always one of the load's stated set. `None`
    /// only on a fire built by hand (a test, a load stating no set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<crate::shapes::FireShape>,
}

impl Step {
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.lanes.iter().map(Lane::rows).sum()
    }

    pub fn validate(&self) -> Result<()> {
        self.validate_for(Serves::NONE)
    }

    pub fn validate_for(&self, serves: Serves) -> Result<()> {
        if self.lanes.is_empty() {
            return Err(Error::Invalid("a fire carries no lanes".into()));
        }
        for (index, lane) in self.lanes.iter().enumerate() {
            lane.validate_for(serves)?;
            // A kv-less lane takes no seat: its slot names nothing.
            if !lane.kv_less
                && self.lanes[..index]
                    .iter()
                    .any(|l| !l.kv_less && l.slot == lane.slot)
            {
                return Err(Error::Invalid(format!(
                    "slot {} appears twice in one fire, at lane {index}",
                    lane.slot
                )));
            }
        }
        let lanes = u32::try_from(self.lanes.len()).unwrap_or(u32::MAX);
        for (index, attachment) in self.attachments.iter().enumerate() {
            if attachment.adapter_only && !serves.adapter_attachments {
                return Err(Error::Invalid(format!(
                    "attachment {index} is for an adapter alone, which this engine does not serve"
                )));
            }
            if attachment.lane >= lanes {
                return Err(Error::Invalid(format!(
                    "attachment names lane {} of the {lanes} this fire has",
                    attachment.lane
                )));
            }
            if self.attachments[..index]
                .iter()
                .any(|earlier| earlier.instance == attachment.instance)
            {
                return Err(Error::Invalid(format!(
                    "instance {} is attached twice to one fire, at attachment {index}; a \
                     program runs once a fire",
                    attachment.instance
                )));
            }
        }
        for (index, media) in self.media.iter().enumerate() {
            let Some(lane) = self.lanes.get(media.lane as usize) else {
                return Err(Error::Invalid(format!(
                    "media row {index} names lane {} of the {lanes} this fire has",
                    media.lane
                )));
            };
            if self.media[..index]
                .iter()
                .any(|earlier| earlier.lane == media.lane)
            {
                return Err(Error::Invalid(format!(
                    "lane {} carries two media rows, at media row {index}; a lane's \
                     spans are one concatenation with one payload order",
                    media.lane
                )));
            }
            media.validate(lane.rows())?;
        }
        for (index, voxels) in self.voxels.iter().enumerate() {
            if self.lanes.get(voxels.lane as usize).is_none() {
                return Err(Error::Invalid(format!(
                    "voxel row {index} names lane {} of the {lanes} this fire has",
                    voxels.lane
                )));
            }
            if self.voxels[..index]
                .iter()
                .any(|earlier| earlier.lane == voxels.lane)
            {
                return Err(Error::Invalid(format!(
                    "lane {} carries two voxel rows, at voxel row {index}; a lane's \
                     clips are one concatenation with one payload order",
                    voxels.lane
                )));
            }
            voxels.validate()?;
        }
        Ok(())
    }
}

impl Step {
    /// Refuses a lane whose word `captures` its attention scores where no
    /// program runs at it on this fire (none attached, or one attached for
    /// its adapter alone): the scores it would capture have no reader, which
    /// only a runtime stamping the switch on the wrong fire asks for.
    pub fn validate_captures(&self, captures: impl Fn(u64) -> bool) -> Result<()> {
        for (at, lane) in self.lanes.iter().enumerate() {
            if !captures(lane.word) {
                continue;
            }
            let at = u32::try_from(at).unwrap_or(u32::MAX);
            let read = self.attachments.iter().any(|attachment| {
                !attachment.adapter_only
                    && (attachment.lane..attachment.lane + attachment.lanes.max(1)).contains(&at)
            });
            if !read {
                return Err(Error::Invalid(format!(
                    "lane {at} captures its attention scores, and no instance whose program \
                     reads them is attached at it on this fire: only the fire that runs a \
                     pass's program captures them"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FrameSubmission {
    pub steps: Vec<Step>,
}

impl FrameSubmission {
    #[must_use]
    pub fn of(step: Step) -> FrameSubmission {
        FrameSubmission { steps: vec![step] }
    }

    #[must_use]
    pub fn rows(&self) -> u32 {
        self.steps.iter().map(Step::rows).sum()
    }

    pub fn validate(&self) -> Result<()> {
        self.validate_for(Serves::NONE)
    }

    pub fn validate_for(&self, serves: Serves) -> Result<()> {
        if self.steps.is_empty() {
            return Err(Error::Invalid("a frame carries no steps".into()));
        }
        for step in &self.steps {
            step.validate_for(serves)?;
        }
        Ok(())
    }

    /// Debug builds: each step's carried shape is the one the load's stated
    /// set (`stated`) holds its lanes in ([`crate::StepShapes::agrees`]), an
    /// engine's cheap cross-check of the runtime's stamp; a step carrying
    /// none (built by hand) is not checked. Release builds check nothing.
    pub fn debug_check_shapes(&self, stated: &crate::StatedSteps) {
        if !cfg!(debug_assertions) {
            return;
        }
        for (at, step) in self.steps.iter().enumerate() {
            if let Some(shape) = step.shape
                && let Err(why) = stated.shapes.agrees(shape, &step.lanes)
            {
                panic!("step {at} of a frame: {why}");
            }
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FrameTicket {
    pub id: FrameId,
    pub steps: Vec<FireTicket>,
    /// The lanes step `steps.len()` of the submission refused for their
    /// own sake ([`Error::Lanes`]), where one did: the frame fired the
    /// steps before it alone, and it and every step after it never ran.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub refused: Vec<crate::LaneFault>,
}

impl FrameTicket {
    /// Frame `id`, as its fired `steps`, stopped short where a step
    /// refused lanes (`refusal`, an [`Error::Lanes`]) or whole.
    #[must_use]
    pub fn of(id: FrameId, steps: Vec<FireTicket>, refusal: Option<Error>) -> FrameTicket {
        FrameTicket {
            id,
            steps,
            refused: match refusal {
                Some(Error::Lanes(lanes)) => lanes,
                _ => Vec::new(),
            },
        }
    }
}

/// The rows a lane read back of `output`, the last row or voxel output its
/// class writes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LaneReadout {
    #[serde(default)]
    pub output: String,
    pub rows: u32,
    pub width: u32,
    pub values: Vec<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub clips: Vec<[u32; 3]>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FireTicket {
    pub id: FireId,
    pub readouts: Vec<LaneReadout>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaEncode {
    pub image_grids: Vec<u32>,
    pub image_pixels: Vec<u8>,
    pub image_pixel_indptr: Vec<u32>,
    pub image_patch_positions: Vec<u32>,
    pub image_anchor_rows: Vec<u32>,
    pub audio_features: Vec<u8>,
    pub audio_feature_indptr: Vec<u32>,
    pub audio_anchor_rows: Vec<u32>,
    pub output_rows: Vec<u8>,
    pub output_row_indptr: Vec<u32>,
}

impl MediaEncode {
    pub fn validate(&self) -> Result<()> {
        const F32: usize = size_of::<f32>();
        const U16: usize = size_of::<u16>();
        let bad = |why: String| Err(Error::Invalid(why));

        let images = self.image_anchor_rows.len();
        let clips = self.audio_anchor_rows.len();
        if images + clips == 0 {
            return bad("an encode carries no image and no audio anchor".into());
        }
        if self.output_row_indptr.len() != images + clips + 1 {
            return bad(format!(
                "output_row_indptr has {} entries for {images} images and {clips} clips",
                self.output_row_indptr.len()
            ));
        }
        if self.output_rows.is_empty() || !self.output_rows.len().is_multiple_of(U16) {
            return bad(format!(
                "output_rows is {} bytes, which is empty or not a whole number of u16",
                self.output_rows.len()
            ));
        }

        if images == 0 {
            if !self.image_grids.is_empty()
                || !self.image_pixels.is_empty()
                || !self.image_pixel_indptr.is_empty()
                || !self.image_patch_positions.is_empty()
            {
                return bad("an image payload arrived with no image anchor to attach it to".into());
            }
        } else {
            if self.image_grids.len() != images.saturating_mul(3) {
                return bad(format!(
                    "image_grids has {} entries for {images} images",
                    self.image_grids.len()
                ));
            }
            if self.image_pixel_indptr.len() != images + 1 {
                return bad(format!(
                    "image_pixel_indptr has {} entries for {images} images",
                    self.image_pixel_indptr.len()
                ));
            }
            if self.image_pixels.is_empty() || !self.image_pixels.len().is_multiple_of(F32) {
                return bad(format!(
                    "image_pixels is {} bytes, which is empty or not a whole number of f32",
                    self.image_pixels.len()
                ));
            }
            if self.image_patch_positions.is_empty()
                || !self.image_patch_positions.len().is_multiple_of(2)
            {
                return bad(format!(
                    "image_patch_positions has {} entries, which is empty or not a whole number \
                     of pairs",
                    self.image_patch_positions.len()
                ));
            }
            partition(
                &self.image_pixel_indptr,
                "image_pixel_indptr",
                self.image_pixels.len(),
                F32,
                false,
            )?;
        }

        if clips == 0 {
            if !self.audio_features.is_empty() || !self.audio_feature_indptr.is_empty() {
                return bad("an audio payload arrived with no audio anchor to attach it to".into());
            }
        } else {
            if self.audio_feature_indptr.len() != clips + 1 {
                return bad(format!(
                    "audio_feature_indptr has {} entries for {clips} clips",
                    self.audio_feature_indptr.len()
                ));
            }
            if self.audio_features.is_empty() || !self.audio_features.len().is_multiple_of(F32) {
                return bad(format!(
                    "audio_features is {} bytes, which is empty or not a whole number of f32",
                    self.audio_features.len()
                ));
            }
            partition(
                &self.audio_feature_indptr,
                "audio_feature_indptr",
                self.audio_features.len(),
                F32,
                true,
            )?;
        }
        Ok(())
    }
}

fn partition(indptr: &[u32], name: &str, bytes: usize, align: usize, strict: bool) -> Result<()> {
    if indptr.first().copied() != Some(0) {
        return Err(Error::Invalid(format!(
            "{name} starts at {:?}, not 0",
            indptr.first()
        )));
    }
    if indptr.last().copied() != u32::try_from(bytes).ok() {
        return Err(Error::Invalid(format!(
            "{name} ends at {:?}, not the {bytes} bytes it partitions",
            indptr.last()
        )));
    }
    for w in indptr.windows(2) {
        let ordered = if strict { w[0] < w[1] } else { w[0] <= w[1] };
        if !ordered {
            return Err(Error::Invalid(format!(
                "{name} segment {}..{} is empty or inverted",
                w[0], w[1]
            )));
        }
        if !(w[0] as usize).is_multiple_of(align) || !(w[1] as usize).is_multiple_of(align) {
            return Err(Error::Invalid(format!(
                "{name} segment {}..{} is not {align}-byte aligned",
                w[0], w[1]
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    /// Metal's `Seat::scratch_bytes`, as engine-metal lays a rung's rs
    /// scratch: each slot's rows and the lanes' work term at least 4 bytes
    /// rounded to 256, rows and lanes each at least one, the whole at least
    /// 256.
    fn metal_scratch_bytes(slots: &[u64], work_elems: u64, rows_ext: u32, lanes: u32) -> u64 {
        let rows = u64::from(rows_ext.max(1));
        let mut at = 0u64;
        for &slot in slots {
            at += (rows * slot).max(4).next_multiple_of(256);
        }
        at += (u64::from(lanes.max(1)) * work_elems * 4).max(4).next_multiple_of(256);
        at.max(256)
    }

    #[test]
    fn scratch_bytes_is_metals_reserve_to_the_byte() {
        for (slots, work_elems) in [
            (vec![2u64, 4096, 12_288, 300], 2048u64),
            (vec![1], 0),
            (vec![513, 3], 1),
        ] {
            let model = ScratchBytes {
                slot_bytes: slots.clone(),
                lane_bytes: work_elems * 4,
                room_bytes: 0,
                pad: 256,
            };
            for rows in [0u32, 1, 2, 7, 255, 1024, 2048] {
                for lanes in [0u32, 1, 3, 8, 64] {
                    assert_eq!(
                        model.bytes(rows, lanes),
                        metal_scratch_bytes(&slots, work_elems, rows, lanes),
                        "slots {slots:?}, work {work_elems}, rows {rows}, lanes {lanes}"
                    );
                }
            }
        }
        // A zero term costs one pad, not nothing.
        let one = ScratchBytes { slot_bytes: vec![1], lane_bytes: 0, room_bytes: 0, pad: 256 };
        assert_eq!(one.bytes(0, 0), 512);
    }

    use super::*;

    fn mask(kept: &[bool]) -> Mask {
        let mut runs = vec![0u32];
        let mut keeping = false;
        for &bit in kept {
            if bit != keeping {
                runs.push(0);
                keeping = bit;
            }
            *runs.last_mut().unwrap() += 1;
        }
        Mask::new(runs, kept.len() as u64)
    }

    #[test]
    fn a_lane_folds_its_windowed_and_recurrent_layers_once() {
        let serves = Serves {
            windowed_state: true,
            rs_verbs: true,
            chunks: true,
            ..Serves::NONE
        };
        let fold = WindowFold {
            base: 16,
            fold: FoldLen::Host(3),
            lineage: Lineage::Set,
        };
        let mut lane = Lane::decode(0, 0, 1, 40);
        lane.kv.folded = 20;
        lane.window = Some(fold);
        assert!(lane.validate_for(serves).is_ok());
        let refused = lane.validate_for(Serves {
            windowed_state: false,
            ..serves
        });
        assert!(
            refused.is_err(),
            "an engine serving no rings takes no window fold"
        );
        lane.window = Some(WindowFold { base: 32, ..fold });
        let why = lane.validate_for(serves).unwrap_err().to_string();
        assert!(why.contains("past its folded index"), "{why}");
        // Rows 20..40 earlier fires buffered stay named while this fire
        // folds them: the table may not start past 20.
        lane.window = Some(WindowFold {
            base: 32,
            fold: FoldLen::Host(20),
            lineage: Lineage::Set,
        });
        let why = lane.validate_for(serves).unwrap_err().to_string();
        assert!(why.contains("earlier fires buffered"), "{why}");

        // A lane folding every row it computes (no earlier-buffered rows,
        // an empty table): the table starts at the folded index after the
        // fire, rounded down, past the one before it.
        let mut prefill = Lane::decode(0, 0, 1, 20);
        prefill.tokens = vec![1; 20];
        prefill.kv.folded = 20;
        prefill.window = Some(WindowFold {
            base: 32,
            fold: FoldLen::Host(20),
            lineage: Lineage::Set,
        });
        assert!(prefill.validate_for(serves).is_ok());
        // Past the folded index after the fire (40): refused.
        prefill.window = Some(WindowFold {
            base: 48,
            fold: FoldLen::Host(20),
            lineage: Lineage::Set,
        });
        let why = prefill.validate_for(serves).unwrap_err().to_string();
        assert!(
            why.contains("past its folded index after the fire 40"),
            "{why}"
        );
        prefill.window = Some(WindowFold {
            base: 32,
            fold: FoldLen::Device,
            lineage: Lineage::Set,
        });
        assert!(
            prefill.validate_for(serves).is_err(),
            "a device-counted fold's base is the folded index the runtime knows"
        );
        lane.window = Some(fold);
        lane.rs = RsVerb::Buffer {
            pages: Vec::new(),
            base: 16,
            folded: 20,
            fold: FoldLen::Host(3),
            lineage: Lineage::Set,
        };
        assert!(lane.validate_for(serves).is_ok());
        lane.rs = RsVerb::Buffer {
            pages: Vec::new(),
            base: 16,
            folded: 20,
            fold: FoldLen::Host(2),
            lineage: Lineage::Set,
        };
        let why = lane.validate_for(serves).unwrap_err().to_string();
        assert!(why.contains("a lane folds once"), "{why}");
    }

    #[test]
    fn a_capturing_lane_needs_a_reader_attached() {
        const SCORES: u64 = 16;
        let mut step = Step {
            lanes: vec![Lane::decode(0, 0, 1, 0), Lane::decode(1, SCORES, 2, 0)],
            ..Step::default()
        };
        let captures = |word: u64| word & SCORES != 0;
        let refused = step.validate_captures(captures).unwrap_err().to_string();
        assert!(refused.contains("lane 1 captures"), "{refused}");
        step.attachments.push(Attachment {
            lane: 1,
            lanes: 1,
            instance: InstanceId(7),
            adapter_only: true,
            scratch: ScratchPlace::None,
        });
        assert!(
            step.validate_captures(captures).is_err(),
            "an adapter alone reads no scores"
        );
        step.attachments[0].adapter_only = false;
        step.validate_captures(captures).unwrap();
    }

    #[test]
    fn a_lane_replays_its_lineage_whatever_its_mask_hides_from_attention() {
        // Slots 0..2 are folded; the lane's one row is at slot 6. Its mask's
        // columns start at the boundary: column 0 is slot 2.
        let ancestry = mask(&[false, true, false, true, true]);
        let one = Masking::Rows(vec![ancestry.clone()]);
        // Alone on its set: every buffered slot, the mask unread.
        assert_eq!(
            buffered_reads(Some(&one), 2, 6, Lineage::Set).unwrap(),
            vec![2..6]
        );
        assert_eq!(
            buffered_reads(None, 2, 6, Lineage::Set).unwrap(),
            vec![2..6]
        );
        // Sharing a set: the ancestry its mask states, as slots.
        let reads = buffered_reads(Some(&one), 2, 6, Lineage::Admitted);
        assert_eq!(reads.unwrap(), vec![3..4, 5..6]);
        assert_eq!(
            buffered_reads(None, 2, 6, Lineage::Admitted).unwrap(),
            vec![2..6]
        );

        // A row hiding its own lane's earlier row reads it in its recurrence:
        // the own rows run in order. Two rows admitting other buffered slots
        // than each other cannot share one run.
        let blind = mask(&[false, true, false, false, true]);
        let rows = Masking::Rows(vec![ancestry.clone(), blind]);
        assert_eq!(
            buffered_reads(Some(&rows), 2, 5, Lineage::Admitted).unwrap(),
            vec![3..4]
        );
        let other = mask(&[true, false, false, true, true]);
        let rows = Masking::Rows(vec![ancestry, other]);
        assert!(buffered_reads(Some(&rows), 2, 5, Lineage::Admitted).is_err());
        assert_eq!(
            buffered_reads(Some(&rows), 2, 5, Lineage::Set).unwrap(),
            vec![2..5]
        );
    }

    #[test]
    fn a_mask_after_its_first_columns_reads_the_same_slots() {
        let bits = [true, false, false, true, true, false, true];
        let whole = mask(&bits);
        for columns in 0..=bits.len() {
            let after = whole.after(columns as u64);
            assert_eq!(after.total, (bits.len() - columns) as u64);
            let mut kept = vec![false; bits.len() - columns];
            for span in after.admitted(0..after.total) {
                for column in span {
                    kept[column as usize] = true;
                }
            }
            assert_eq!(kept, bits[columns..], "after {columns}");
        }
    }

    #[test]
    fn kv_less_lanes_take_no_seat() {
        // A CFG step: four unseated float lanes beside a decode in seat 0.
        let float = || Lane {
            tokens: vec![0; 4],
            kv_less: true,
            ..Lane::default()
        };
        let mut step = Step {
            lanes: vec![float(), float(), float(), float()],
            ..Step::default()
        };
        step.lanes.push(Lane::decode(0, 0, 7, 3));
        assert!(step.validate().is_ok(), "{:?}", step.validate());
        step.lanes.push(Lane::decode(0, 0, 8, 4));
        assert!(step.validate().is_err(), "two seated lanes share seat 0");
    }
}
