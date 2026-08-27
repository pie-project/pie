//! Lane words in, the window table out (palo design §5, step 2).
//!
//! ```text
//! fire (R=5):        lane0 prefill(7 rows)  lane1 prefill(3)  lane2..4 decode(1 each)
//! rows (seriated):   [··············· 10 ···············|········ 3 ········]
//!
//! embed, norm, qkv     ───────────── all 13 rows, one kernel ─────────────   Always
//! attention.prefill    ────── window [0,10) ──────┐                          ¬qo_one
//! attention.decode                                └──── window [10,13) ────  qo_one
//! o_proj, mlp          ───────────── all 13 rows, one kernel ─────────────   Always
//! ```
//!
//! That diagram is this file. Every lane arrives with a `word` — the fact
//! bits `Classify::of` computed for its request — and a row count; every word
//! belongs to exactly one CLASS of the artifact, since the class sweep is
//! total over the `2^F` words the plan's own guards reach and a lane's word is
//! masked down to those bits first; **rows are seriated by class**, so a
//! class's rows are one contiguous interval and a node that runs in one class
//! runs over one `[offset, offset + rows)`. Prefix sums are the whole
//! computation.
//!
//! # Why seriate at all
//!
//! Because a windowed kernel takes a pointer and an extent, not a row list. A
//! consumer that runs in a SET of classes gets one launch iff that set is an
//! interval of the class order — the Consecutive-Ones Property, which is what
//! P4's layout pass solves, exactly, over the whole plan at once (design §3).
//! This file does not re-solve it: it takes P4's baked ordering, drops the
//! classes this fire has no lanes in, and adds up counts. A mask that still
//! straddles a gap — because the class between its members is present too, and
//! the consumer was one P4 could not seat — simply becomes more than one
//! launch, and [`WindowTable::segments`] is what says how many.
//!
//! # What this is NOT allowed to be
//!
//! Slow. This runs every fire, on the host, in front of a launch that costs
//! tens of microseconds. So: one pass over the lanes to count, one pass per
//! present class to seriate, no allocation per lane, and the word -> class
//! lookup memoized across the fire — a batch of 256 lanes carries at most a
//! handful of distinct words, and `Classes::class_of` is a scan over the
//! sweep's dedup'd word lists rather than a hash.

use model_compiler::{ArenaMap, Baked, Budgets, Window};
use model_ir::{ClassSet, ValueId};

use crate::fire::Fault;
use crate::{Error, Result};

/// One request inside a fire, as the engine submits it.
///
/// THE TWO NUMBERS ARE THE WHOLE SUBMISSION as far as composition is
/// concerned. `word` decides WHICH windows this lane is in — it is the one
/// genuinely new field of the palo submission contract (decision #18),
/// computed engine-side by the model's own `Classify::of` — and `rows` decides
/// HOW MUCH of them it occupies. Everything else about a request (its pages,
/// its adapter, its tokens) rides in buffers the driver already binds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lane {
    /// The lane's fact bits. `Cond::Fact(bit)` indexes them.
    pub word: u64,
    /// How many token rows this lane contributes — 1 for a decode step, the
    /// prompt length for a prefill, `1 + drafts` for speculative rows.
    pub rows: u32,
}

impl Lane {
    /// A lane of `rows` rows whose facts are `word`.
    #[must_use]
    pub fn new(word: u64, rows: u32) -> Lane {
        Lane { word, rows }
    }
}

/// One class's place in the seriated fire.
///
/// Rows and lanes are counted separately because the IR has both symbols:
/// `Dim::Tokens` columns are indexed by the row offset and `Dim::Lanes`
/// columns — the geometry vectors, the indptrs — by the lane offset, and in a
/// mixed fire the two are different numbers. Collapsing them is the arithmetic
/// the rewrite got wrong (see `model_compiler::arena`'s note on the per-row
/// pitch).
///
/// A class with no lanes in this fire is the zero window, and that is not a
/// special case anywhere: `rows == 0` is what an empty window IS, and it is
/// what a kernel reads and returns on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ClassWindow {
    /// First token row of this class's interval.
    pub row_offset: u32,
    /// How many token rows it has. Zero for a class no lane is in.
    pub rows: u32,
    /// First lane of this class's interval.
    pub lane_offset: u32,
    /// How many lanes it has.
    pub lanes: u32,
}

/// A contiguous run of token rows — what one windowed launch covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowSpan {
    /// First row.
    pub offset: u32,
    /// How many.
    pub rows: u32,
}

/// The one row-and-lane interval a class mask covers in a fire.
///
/// **THE WINDOW A NODE ACTUALLY RUNS OVER.** Design §0's diagram gives every
/// guarded node an interval `[offset, offset + rows)` and the shells resolve
/// their operands inside it; the rows are what a `Dim::Tokens` column is cut
/// at and the lanes are what a `Dim::Lanes` one is, and in a mixed fire those
/// are two different numbers ([`ClassWindow`] says why). Both halves come off
/// the same prefix sums, which is why they are answered together rather than
/// by two calls that could disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MaskSpan {
    /// First token row.
    pub row_offset: u32,
    /// How many token rows.
    pub rows: u32,
    /// First lane.
    pub lane_offset: u32,
    /// How many lanes.
    pub lanes: u32,
}

/// The window table: one [`ClassWindow`] per class of the artifact, indexed by
/// class.
///
/// ONE IMPLEMENTATION OF THE ARITHMETIC, TWO CARRIERS. A [`Composition`] holds
/// this table and so does a [`FireDescriptor`](crate::fire::FireDescriptor) —
/// the second is the first's byte form — and both answer the same question the
/// same way, because the question ("what rows does this mask cover") is asked
/// by the walk against whichever one is in hand.
///
/// Indexed by class POSITION, which is what a `ClassSet` iterates and what
/// `Classes::node_mask` was built against. A table of the wrong width does not
/// fail to find a class; it finds the wrong one, which is why
/// [`walk()`](fn@crate::fire::walk) checks the width before it walks.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WindowTable {
    classes: Vec<ClassWindow>,
}

impl WindowTable {
    /// A table of one window per class, in class order.
    #[must_use]
    pub fn new(classes: Vec<ClassWindow>) -> WindowTable {
        WindowTable { classes }
    }

    /// How many classes it covers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.classes.len()
    }

    /// Does it cover no classes at all? True only for a factless artifact that
    /// somehow swept nothing — a real plan always has at least one class.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.classes.is_empty()
    }

    /// One class's window, or the zero window for a class this table does not
    /// have.
    #[must_use]
    pub fn class(&self, class: usize) -> ClassWindow {
        self.classes.get(class).copied().unwrap_or_default()
    }

    /// The windows, in class order.
    #[must_use]
    pub fn as_slice(&self) -> &[ClassWindow] {
        &self.classes
    }

    /// How many token rows a node with this class mask runs over.
    ///
    /// **THE ZERO-ROW QUESTION, AND THE ONLY ONE THE WALK ASKS.** A region's
    /// `mask` is the classes that run it, and this fire's rows for those
    /// classes are what the region's kernels read as their count. Zero means
    /// the composition does not include this behavior at all: eager mode skips
    /// the dispatch, recorded mode launches a kernel that returns immediately
    /// (decision #3). Neither is a branch on the plan — it is the same number
    /// read two ways.
    #[must_use]
    pub fn rows_of(&self, mask: &ClassSet) -> u32 {
        mask.iter().map(|c| self.class(c).rows).sum()
    }

    /// How many lanes a node with this class mask runs over — the same
    /// question for a `Dim::Lanes` column.
    #[must_use]
    pub fn lanes_of(&self, mask: &ClassSet) -> u32 {
        mask.iter().map(|c| self.class(c).lanes).sum()
    }

    /// The one interval this mask covers, rows and lanes together.
    ///
    /// **THE SERIATION'S PROMISE, CASHED.** P4 chose a global class order that
    /// makes every structural consumer's class set consecutive (design §3), so
    /// a mask's present classes stand as ONE run of rows and one run of lanes
    /// — which is exactly what a windowed kernel takes: a pointer and an
    /// extent. `Ok(None)` is the empty window (a composition without this
    /// behavior in it), and `Err(runs)` is the mask P4 could not seat, naming
    /// how many launches it would take instead. A caller that gets `Err` from
    /// a plan whose [`FallbackTable`](model_compiler::FallbackTable) is empty
    /// has a bake-integrity failure, not a slow path.
    ///
    /// Contiguity is checked against the classes NOT in the mask: a mask is
    /// one run iff no absent class's rows stand inside the span its own
    /// classes bound. Zero-row classes are invisible to that test, because a
    /// class no lane is in occupies no rows to straddle — which is what makes
    /// an all-decode fire's prefill mask an empty window rather than a
    /// fragmented one.
    ///
    /// # Errors
    ///
    /// The number of runs the mask covers, when that is more than one.
    pub fn span(&self, mask: &ClassSet) -> core::result::Result<Option<MaskSpan>, usize> {
        let mut span = MaskSpan {
            row_offset: u32::MAX,
            rows: 0,
            lane_offset: u32::MAX,
            lanes: 0,
        };
        for class in mask.iter() {
            let window = self.class(class);
            if window.rows == 0 {
                continue;
            }
            span.row_offset = span.row_offset.min(window.row_offset);
            span.rows += window.rows;
            span.lane_offset = span.lane_offset.min(window.lane_offset);
            span.lanes += window.lanes;
        }
        if span.rows == 0 {
            return Ok(None);
        }
        let straddles = self
            .classes
            .iter()
            .enumerate()
            .filter(|(class, window)| window.rows > 0 && !mask.contains(*class))
            .any(|(_, window)| {
                window.row_offset >= span.row_offset
                    && window.row_offset < span.row_offset + span.rows
            });
        if straddles {
            return Err(self.segments(mask).len());
        }
        Ok(Some(span))
    }

    /// The maximal contiguous row runs this mask covers, ascending.
    ///
    /// ONE SPAN IS THE ANSWER P4 IS FOR. A mask whose classes are an interval
    /// of the fire's class order comes back as a single [`RowSpan`] and its
    /// consumer is one launch over pointer+extent; a mask that straddles a
    /// class it does not contain comes back as two, and the consumer is two
    /// launches — or a `Fallback` row, once P4 starts making promises and
    /// owing them. Empty classes are dropped rather than splitting a run,
    /// since a zero-row class occupies no rows to straddle.
    #[must_use]
    pub fn segments(&self, mask: &ClassSet) -> Vec<RowSpan> {
        let mut spans: Vec<RowSpan> = mask
            .iter()
            .map(|c| self.class(c))
            .filter(|w| w.rows > 0)
            .map(|w| RowSpan {
                offset: w.row_offset,
                rows: w.rows,
            })
            .collect();
        spans.sort_unstable_by_key(|s| s.offset);

        let mut merged: Vec<RowSpan> = Vec::with_capacity(spans.len());
        for span in spans {
            match merged.last_mut() {
                Some(open) if open.offset + open.rows == span.offset => open.rows += span.rows,
                _ => merged.push(span),
            }
        }
        merged
    }
}

/// One lane, placed.
///
/// `source` IS THE POINT OF THE RECORD. The engine submitted lane 4; the fire
/// runs it third, because seriation groups the classes. Every per-lane buffer
/// the driver binds — pages, positions, adapter ids — is written in submission
/// order, so the descriptor has to carry the permutation or the device reads
/// somebody else's geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaneRow {
    /// This lane's index in the submitted slice.
    pub source: u32,
    /// Its fact word, carried through so the device never re-derives it.
    pub word: u64,
    /// The class the word resolved to.
    pub class: u32,
    /// Its first token row in the seriated fire.
    pub row_offset: u32,
    /// How many rows it contributes.
    pub rows: u32,
}

/// One fire's composition: which windows have rows, where, and in what order.
///
/// WHAT IT IS NOT is a schedule. The script is baked (`Baked::template`) and
/// the same one runs every fire; this is the DATA that script reads. Which is
/// why a composition can be built, thrown away and built again 5000 times a
/// second without anything being recompiled or recaptured (design §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Composition {
    lanes: Vec<LaneRow>,
    classes: WindowTable,
    order: Vec<u32>,
    rows: u32,
    bucket: u32,
}

impl Composition {
    /// The lanes, in fire order: grouped by class, submission order kept
    /// inside each class.
    #[must_use]
    pub fn lanes(&self) -> &[LaneRow] {
        &self.lanes
    }

    /// How many lanes this fire carries.
    #[must_use]
    pub fn lane_count(&self) -> u32 {
        self.lanes.len() as u32
    }

    /// The window table, indexed by class.
    #[must_use]
    pub fn classes(&self) -> &WindowTable {
        &self.classes
    }

    /// The classes this fire has lanes in, in the order their rows stand.
    ///
    /// THE COMPOSITION, LITERALLY: design §0's word for "which windows are
    /// non-empty in a fire" is exactly this list.
    #[must_use]
    pub fn present(&self) -> &[u32] {
        &self.order
    }

    /// Total token rows.
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// The shape bucket these rows round up to — which recorded graph this
    /// fire launches (design §5). Equal to [`rows`](Composition::rows) when
    /// the budget lists no lattice.
    #[must_use]
    pub fn bucket(&self) -> u32 {
        self.bucket
    }

    /// Where a value's rectangle sits in THIS fire.
    ///
    /// The offset is static — decided once by P7, the same in every bucket —
    /// and only the length moves, which is the whole reason a composition
    /// never triggers a recapture. This is the one call that puts the two
    /// halves together, and it is a delegation because the arithmetic belongs
    /// to the carve that chose the offsets.
    #[must_use]
    pub fn value_window(&self, arena: &ArenaMap, value: ValueId) -> Option<Window> {
        arena.window(value, u64::from(self.rows), u64::from(self.lane_count()))
    }
}

/// Compose one fire: lane words to classes, classes to an order, counts to
/// prefix sums.
///
/// # Errors
///
/// [`Fault::UnknownWord`] for a lane no class of this artifact admits,
/// [`Fault::EmptyLane`] for a lane of no rows, [`Fault::TooManyLanes`] /
/// [`Fault::TooManyRows`] for a fire past the ceilings the arena was cut at,
/// and [`Fault::NoBucket`] for a fire above the whole bucket lattice. All five
/// are the same kind of thing: a batch this artifact cannot describe, refused
/// before a byte is written rather than launched into columns that are too
/// short for it.
pub fn compose(baked: &Baked, budgets: &Budgets, lanes: &[Lane]) -> Result<Composition> {
    if lanes.len() > budgets.max_lanes as usize {
        return Err(Fault::TooManyLanes {
            lanes: lanes.len(),
            max: budgets.max_lanes,
        }
        .into());
    }

    // Pass one: every lane's class, and what each class adds up to. The memo
    // is a `Vec` and the lookup is linear because a fire's DISTINCT words are
    // few — two in the diagram above, a handful in the worst catalog plan —
    // and a hash of a `u64` costs more than the scan it would replace.
    let count = baked.classes.classes.len();
    let mut memo: Vec<(u64, u32)> = Vec::new();
    let mut of_lane: Vec<u32> = Vec::with_capacity(lanes.len());
    let mut tally: Vec<(u64, u32)> = vec![(0, 0); count];
    let mut rows: u64 = 0;

    for (i, lane) in lanes.iter().enumerate() {
        let i = i as u32;
        if lane.rows == 0 {
            return Err(Fault::EmptyLane { lane: i }.into());
        }
        // THE WORD IS MASKED TO THE BITS THE SWEEP RAN OVER. A lane's word
        // comes from the model's `Classify`, which packs every fact the model
        // COMPUTES; `Classes::mask` is the bits some guard READS. A model may
        // state a fact it does not split on — and then a lane that sets it
        // carries a word no class enumerates, which would surface as
        // `UnknownWord`: "the engine and the shell disagree about what is
        // loaded", said about two halves that agree perfectly.
        let word = lane.word & baked.classes.mask;
        let class = match memo.iter().find(|(seen, _)| *seen == word) {
            Some((_, class)) => *class,
            None => {
                let class = baked.classes.class_of(word).ok_or(Fault::UnknownWord {
                    lane: i,
                    word: lane.word,
                })? as u32;
                memo.push((word, class));
                class
            }
        };
        // `class_of` answers a position in `classes.classes`, so this index is
        // in range by construction; the tally is sized from the same list.
        tally[class as usize].0 += u64::from(lane.rows);
        tally[class as usize].1 += 1;
        rows += u64::from(lane.rows);
        of_lane.push(class);
    }

    if rows > u64::from(budgets.max_tokens) {
        return Err(Fault::TooManyRows {
            rows,
            max: budgets.max_tokens,
        }
        .into());
    }
    let rows = rows as u32;

    // The classes this fire has lanes in, in the order their rows will stand.
    //
    // **ONE BAKED ANSWER, FILTERED, AND NOT A TABLE WITH A ROW PER
    // COMPOSITION.** There are `2^classes` compositions and one C1P instance;
    // a sub-order of an ordering that makes a set consecutive still makes that
    // set consecutive, so dropping the absent classes out of the global order
    // is the whole per-fire computation (design §3). P4 owns that filter —
    // this asks it rather than re-deriving it from the tree, because a second
    // reading of one answer is a second answer waiting to disagree.
    //
    // `prev` IS LAST FIRE'S ORDER, AND THIS PASSES `None` BECAUSE THERE IS
    // NOTHING TO PASS YET. v1's `class_order` ignores the argument; when the
    // stability pick lands, the order this fire chose is what the next one
    // should be handed, and that is a change to what `compose` is GIVEN — a
    // parameter — rather than to what it asks P4.
    //
    // A class is a `u8` on the far side of this call: the ceiling the whole
    // fire seam is spelled at, from the PQ-tree's leaves down to the
    // descriptor, and a plan with more behaviours than that is one P4 declines
    // to seriate and P8 refuses to fire. So the widening here is a conversion,
    // never a truncation, and a fire that could ever reach one has already
    // been turned away upstream.
    let present = ClassSet::of((0..count).filter(|&class| tally[class].1 > 0));
    let order: Vec<u32> = baked
        .order
        .class_order(&present, None)
        .into_iter()
        .map(u32::from)
        .collect();

    // Pass two: prefix sums in that order. This is the window table.
    let mut classes = vec![ClassWindow::default(); count];
    let (mut row_at, mut lane_at) = (0u32, 0u32);
    for &class in &order {
        let (class_rows, class_lanes) = tally[class as usize];
        classes[class as usize] = ClassWindow {
            row_offset: row_at,
            rows: class_rows as u32,
            lane_offset: lane_at,
            lanes: class_lanes,
        };
        row_at += class_rows as u32;
        lane_at += class_lanes;
    }

    // Pass three: place the lanes. Class by class in fire order, and INSIDE a
    // class in submission order — that inner stability is not decoration. The
    // engine's order is the order its per-lane buffers are written in, and a
    // fire that reshuffles lanes it had no reason to reshuffle churns every
    // pointer the previous fire had warm.
    let mut seriated: Vec<LaneRow> = Vec::with_capacity(lanes.len());
    for &class in &order {
        let mut row_at = classes[class as usize].row_offset;
        for (i, lane) in lanes.iter().enumerate() {
            if of_lane[i] != class {
                continue;
            }
            seriated.push(LaneRow {
                source: i as u32,
                word: lane.word,
                class,
                row_offset: row_at,
                rows: lane.rows,
            });
            row_at += lane.rows;
        }
    }

    Ok(Composition {
        lanes: seriated,
        classes: WindowTable::new(classes),
        order,
        rows,
        bucket: bucket_of(budgets, rows)?,
    })
}

/// The smallest bucket that holds these rows.
///
/// AN EMPTY LATTICE IS NOT AN ERROR: a budget that lists no buckets is a
/// deployment that has not chosen a shape lattice, and the honest bucket for a
/// fire of `rows` rows is then `rows` itself — one graph per size, which is
/// what a golden-path eager walk wants and what a test builds with
/// `Budgets::new`.
fn bucket_of(budgets: &Budgets, rows: u32) -> Result<u32> {
    if budgets.buckets.is_empty() {
        return Ok(rows);
    }
    budgets
        .buckets
        .iter()
        .copied()
        .find(|bucket| *bucket >= rows)
        .ok_or_else(|| {
            Error::Fire(Fault::NoBucket {
                rows,
                top: budgets.buckets.last().copied().unwrap_or(0),
            })
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fire::fixture::{Build, fact};
    use crate::{Error, fire::Fault};
    use model_compiler::{DeviceProfile, compile};
    use model_ir::{Cond, ValueId};

    /// A deployment small enough to state in an assert: 8 lanes, 64 rows, no
    /// bucket lattice.
    fn budgets() -> Budgets {
        Budgets::new(8, 64)
    }

    /// The same, said from inside a test that shadowed the name with a
    /// lattice of its own.
    fn budgets_without_a_lattice() -> Budgets {
        Budgets::new(8, 64)
    }

    /// Design §0's diagram, as a plan: a shared producer, an attention pair
    /// split on `qo_one`, a merge, a shared consumer. The two facts of the
    /// whole design in five nodes.
    ///
    /// Returns the plan's output value, because the arena question
    /// ([`Composition::value_window`]) is about a value and there is no other
    /// way to name one from out here.
    fn diagram() -> (Build, ValueId) {
        let mut b = Build::new();
        let x = b.input(8);
        let plan = b.prepare(Cond::Always); // node 0 — prepare, every class
        let q = b.op(x, 4, Cond::Always); // node 1 — every class
        let d = b.decode(q, plan, fact(0)); // node 2 — the decode window
        let p = b.op(q, 4, Cond::not(fact(0))); // node 3 — the prefill window
        let o = b.merge(&[(d, fact(0)), (p, Cond::not(fact(0)))], 4);
        let y = b.op(o, 4, Cond::Always); // node 4 — every class again
        b.out(y);
        (b, y)
    }

    /// The nodes of [`diagram`], by the name the design gives them.
    const SHARED: u32 = 1;
    const DECODE: u32 = 2;
    const PREFILL: u32 = 3;

    /// The rows one node runs over — its region's mask, against this fire.
    fn rows_of(baked: &Baked, fire: &Composition, node: u32) -> u32 {
        let region = baked
            .template()
            .iter()
            .find(|r| r.nodes.contains(&node))
            .expect("the regions tile the node list");
        fire.classes().rows_of(&region.mask)
    }

    #[test]
    fn the_thirteen_row_diagram_windows_the_way_the_design_draws_it() {
        // fire (R=5): lane0 prefill(7) lane1 prefill(3) lane2..4 decode(1 each)
        let (b, _) = diagram();
        let baked = compile(&b.plan, &budgets(), &DeviceProfile::default()).expect("bakes");
        let lanes = [
            Lane::new(0, 7),
            Lane::new(0, 3),
            Lane::new(1, 1),
            Lane::new(1, 1),
            Lane::new(1, 1),
        ];
        let fire = compose(&baked, &budgets(), &lanes).expect("composes");

        assert_eq!(fire.rows(), 13);
        assert_eq!(fire.lane_count(), 5);
        assert_eq!(fire.present().len(), 2);

        let prefill = baked.classes.class_of(0).expect("word 0 is a class");
        let decode = baked.classes.class_of(1).expect("word 1 is a class");
        let p = fire.classes().class(prefill);
        let d = fire.classes().class(decode);
        assert_eq!((p.rows, p.lanes), (10, 2), "two prefill lanes, ten rows");
        assert_eq!((d.rows, d.lanes), (3, 3), "three decode lanes, three rows");

        // The two windows TILE the fire — no gap, no overlap — in whichever of
        // the two orders P4's ordering seated them.
        let mut spans = [(p.row_offset, p.rows), (d.row_offset, d.rows)];
        spans.sort_unstable();
        assert!(
            spans == [(0, 10), (10, 3)] || spans == [(0, 3), (3, 10)],
            "the windows do not tile [0, 13): {spans:?}",
        );

        // And this is the diagram's actual claim: one kernel over all 13 rows
        // for the shared ops, one over each window for the split pair.
        assert_eq!(rows_of(&baked, &fire, SHARED), 13);
        assert_eq!(rows_of(&baked, &fire, DECODE), 3);
        assert_eq!(rows_of(&baked, &fire, PREFILL), 10);
    }

    #[test]
    fn an_all_decode_fire_leaves_the_prefill_window_empty() {
        let (b, _) = diagram();
        let baked = compile(&b.plan, &budgets(), &DeviceProfile::default()).expect("bakes");
        let lanes = [Lane::new(1, 1), Lane::new(1, 1), Lane::new(1, 1)];
        let fire = compose(&baked, &budgets(), &lanes).expect("composes");

        assert_eq!(fire.rows(), 3);
        assert_eq!(fire.present().len(), 1, "one class has lanes");
        // An empty window is a zero, not an absence: the class still has a
        // window, and it has no rows.
        assert_eq!(rows_of(&baked, &fire, PREFILL), 0);
        assert_eq!(rows_of(&baked, &fire, DECODE), 3);
        assert_eq!(rows_of(&baked, &fire, SHARED), 3);
        // The decode class starts at row 0 — an absent class occupies nothing.
        let decode = baked.classes.class_of(1).expect("word 1 is a class");
        assert_eq!(fire.classes().class(decode).row_offset, 0);
    }

    #[test]
    fn lanes_keep_submission_order_inside_a_class() {
        let (b, _) = diagram();
        let baked = compile(&b.plan, &budgets(), &DeviceProfile::default()).expect("bakes");
        // Interleaved on submission: prefill, decode, prefill, decode.
        let lanes = [
            Lane::new(0, 5),
            Lane::new(1, 1),
            Lane::new(0, 2),
            Lane::new(1, 1),
        ];
        let fire = compose(&baked, &budgets(), &lanes).expect("composes");

        // Grouped by class, and the sources inside each group ascend: lane 2
        // never runs before lane 0.
        let mut rows = 0;
        for window in fire.lanes() {
            assert_eq!(window.row_offset, rows, "the lanes tile the fire's rows");
            rows += window.rows;
        }
        assert_eq!(rows, 9);

        let sources: Vec<Vec<u32>> = fire
            .present()
            .iter()
            .map(|&class| {
                fire.lanes()
                    .iter()
                    .filter(|lane| lane.class == class)
                    .map(|lane| lane.source)
                    .collect()
            })
            .collect();
        for group in &sources {
            assert!(
                group.windows(2).all(|pair| pair[0] < pair[1]),
                "a class reshuffled the lanes inside it: {group:?}",
            );
        }
    }

    #[test]
    fn an_absent_class_does_not_split_a_run() {
        // Two facts, so four classes, and a node guarded on one fact alone
        // runs in two of them — which may or may not be adjacent under the
        // order P4 seated. What is NOT allowed to depend on that: a fire
        // whose only lanes are those two classes puts them side by side, so
        // the consumer is one launch.
        let mut b = Build::new();
        let x = b.input(4);
        let q = b.op(x, 4, Cond::Always);
        let one = b.op(q, 4, fact(0));
        let other = b.op(q, 4, Cond::not(fact(0)));
        let o = b.merge(&[(one, fact(0)), (other, Cond::not(fact(0)))], 4);
        let m = b.op(o, 4, fact(1));
        let um = b.op(o, 4, Cond::not(fact(1)));
        let out = b.merge(&[(m, fact(1)), (um, Cond::not(fact(1)))], 4);
        b.out(out);
        let baked = compile(&b.plan, &budgets(), &DeviceProfile::default()).expect("bakes");
        assert_eq!(baked.classes.classes.len(), 4);

        // `one` is node 1, guarded `qo_one`: it runs in the two classes whose
        // word has bit 0 set, words 0b01 and 0b11.
        let region = baked
            .template()
            .iter()
            .find(|r| r.nodes.contains(&1))
            .expect("the regions tile the node list");

        let all = [
            Lane::new(0b00, 1),
            Lane::new(0b01, 1),
            Lane::new(0b10, 1),
            Lane::new(0b11, 1),
        ];
        let fire = compose(&baked, &budgets(), &all).expect("composes");
        let spans = fire.classes().segments(&region.mask);
        assert_eq!(fire.classes().rows_of(&region.mask), 2);
        assert_eq!(
            spans.iter().map(|s| s.rows).sum::<u32>(),
            2,
            "the segments carry every row of the window",
        );
        assert!(spans.len() <= 2 && !spans.is_empty());
        assert!(
            spans
                .windows(2)
                .all(|p| p[0].offset + p[0].rows < p[1].offset),
            "maximal runs would have been merged: {spans:?}",
        );

        let only = [Lane::new(0b01, 2), Lane::new(0b11, 3)];
        let fire = compose(&baked, &budgets(), &only).expect("composes");
        let spans = fire.classes().segments(&region.mask);
        assert_eq!(
            spans,
            vec![RowSpan { offset: 0, rows: 5 }],
            "with the classes between them absent, the window is one launch",
        );
    }

    #[test]
    fn a_bit_no_guard_reads_is_masked_off_rather_than_refused() {
        let (b, _) = diagram();
        let baked = compile(&b.plan, &budgets(), &DeviceProfile::default()).expect("bakes");
        // One guarded bit, so the sweep covers words 0 and 1. A lane carrying
        // bit 1 as well is a model computing a fact this plan does not split
        // on — which is the same behaviour as the same lane without it, and
        // not two halves disagreeing about what is loaded.
        let plain = compose(&baked, &budgets(), &[Lane::new(0b01, 1)]).expect("composes");
        let extra = compose(&baked, &budgets(), &[Lane::new(0b11, 1)]).expect("composes");
        assert_eq!(rows_of(&baked, &plain, DECODE), 1);
        assert_eq!(rows_of(&baked, &extra, DECODE), 1);
        assert_eq!(rows_of(&baked, &extra, PREFILL), 0);
    }

    #[test]
    fn a_lane_of_no_rows_is_refused_before_it_takes_a_seat() {
        let (b, _) = diagram();
        let baked = compile(&b.plan, &budgets(), &DeviceProfile::default()).expect("bakes");
        assert_eq!(
            compose(&baked, &budgets(), &[Lane::new(1, 1), Lane::new(0, 0)]),
            Err(Error::Fire(Fault::EmptyLane { lane: 1 })),
        );
    }

    #[test]
    fn the_ceilings_are_the_budget_s_and_the_refusal_carries_them() {
        let (b, _) = diagram();
        let baked = compile(&b.plan, &budgets(), &DeviceProfile::default()).expect("bakes");

        let crowd: Vec<Lane> = (0..9).map(|_| Lane::new(1, 1)).collect();
        assert_eq!(
            compose(&baked, &budgets(), &crowd),
            Err(Error::Fire(Fault::TooManyLanes { lanes: 9, max: 8 })),
        );

        let long = [Lane::new(0, 40), Lane::new(0, 40)];
        assert_eq!(
            compose(&baked, &budgets(), &long),
            Err(Error::Fire(Fault::TooManyRows { rows: 80, max: 64 })),
        );
    }

    #[test]
    fn a_fire_rounds_up_to_a_bucket_and_one_above_them_all_is_refused() {
        let (b, _) = diagram();
        let mut budgets = budgets();
        budgets.buckets = vec![1, 4, 16];
        let baked = compile(&b.plan, &budgets, &DeviceProfile::default()).expect("bakes");

        let fire = compose(&baked, &budgets, &[Lane::new(0, 5)]).expect("composes");
        assert_eq!((fire.rows(), fire.bucket()), (5, 16));
        let fire = compose(&baked, &budgets, &[Lane::new(1, 1)]).expect("composes");
        assert_eq!((fire.rows(), fire.bucket()), (1, 1));

        // 17 rows round up to nothing, so there is no graph to launch them in.
        assert_eq!(
            compose(&baked, &budgets, &[Lane::new(0, 17)]),
            Err(Error::Fire(Fault::NoBucket { rows: 17, top: 16 })),
        );

        // A budget with no lattice is a deployment that has not chosen one,
        // and then the bucket is the fire's own size.
        let open = super::tests::budgets_without_a_lattice();
        let baked = compile(&b.plan, &open, &DeviceProfile::default()).expect("bakes");
        let fire = compose(&baked, &open, &[Lane::new(0, 5)]).expect("composes");
        assert_eq!(fire.bucket(), 5);
    }

    #[test]
    fn a_value_s_offset_is_static_and_only_its_length_moves() {
        let (b, y) = diagram();
        let baked = compile(&b.plan, &budgets(), &DeviceProfile::default()).expect("bakes");

        let small = compose(&baked, &budgets(), &[Lane::new(1, 1)]).expect("composes");
        let big =
            compose(&baked, &budgets(), &[Lane::new(0, 7), Lane::new(1, 1)]).expect("composes");

        let one = small
            .value_window(&baked.arena, y)
            .expect("y is in the arena");
        let eight = big
            .value_window(&baked.arena, y)
            .expect("y is in the arena");
        assert_eq!(
            one.offset, eight.offset,
            "the offset is decided once, at bake"
        );
        // Four bf16 columns: 2 bytes an element, four to a row.
        assert_eq!(one.bytes, 4 * 2);
        assert_eq!(eight.bytes, 8 * 4 * 2);
        assert_eq!(
            eight,
            baked
                .arena
                .window(y, u64::from(big.rows()), u64::from(big.lane_count()))
                .expect("the same question, asked of the carve directly"),
        );
    }

    /// Three classes standing in row order, so a mask can be asked to
    /// straddle the middle one.
    fn three() -> WindowTable {
        WindowTable::new(vec![
            ClassWindow {
                row_offset: 0,
                rows: 4,
                lane_offset: 0,
                lanes: 1,
            },
            ClassWindow {
                row_offset: 4,
                rows: 2,
                lane_offset: 1,
                lanes: 2,
            },
            ClassWindow {
                row_offset: 6,
                rows: 3,
                lane_offset: 3,
                lanes: 3,
            },
        ])
    }

    #[test]
    fn a_mask_that_is_an_interval_of_the_order_is_one_span() {
        let span = three()
            .span(&ClassSet::of([0, 1]))
            .expect("consecutive")
            .expect("non-empty");
        assert_eq!(
            span,
            MaskSpan {
                row_offset: 0,
                rows: 6,
                lane_offset: 0,
                lanes: 3
            },
            "rows and lanes both, because the IR has both symbols"
        );
    }

    #[test]
    fn a_mask_that_straddles_a_present_class_is_refused_and_says_how_many_runs() {
        // {0, 2} with class 1 standing between them: two launches, and no
        // fallback row to take them — P4's promise, found broken.
        assert_eq!(three().span(&ClassSet::of([0, 2])), Err(2));
    }

    #[test]
    fn an_absent_class_between_two_present_ones_does_not_split_them() {
        // The same mask, in a fire class 1 has no lanes in. A zero-row class
        // occupies no rows to straddle, so {0, 2} is one run — which is what
        // makes an all-decode fire's masks windows rather than refusals.
        let table = WindowTable::new(vec![
            ClassWindow {
                row_offset: 0,
                rows: 4,
                lane_offset: 0,
                lanes: 1,
            },
            ClassWindow::default(),
            ClassWindow {
                row_offset: 4,
                rows: 3,
                lane_offset: 1,
                lanes: 3,
            },
        ]);
        let span = table
            .span(&ClassSet::of([0, 2]))
            .expect("consecutive")
            .expect("non-empty");
        assert_eq!((span.row_offset, span.rows), (0, 7));
        assert_eq!(table.span(&ClassSet::of([1])), Ok(None), "the empty window");
    }

    #[test]
    fn a_fire_of_no_lanes_is_a_fire_with_every_window_empty() {
        // Not an error: a rank whose batch is empty still has collectives to
        // join (decision #5), and `walk` is where that is spelled.
        let (b, _) = diagram();
        let baked = compile(&b.plan, &budgets(), &DeviceProfile::default()).expect("bakes");
        let fire = compose(&baked, &budgets(), &[]).expect("composes");
        assert_eq!(fire.rows(), 0);
        assert_eq!(fire.lane_count(), 0);
        assert!(fire.present().is_empty());
        assert_eq!(rows_of(&baked, &fire, SHARED), 0);
    }
}
