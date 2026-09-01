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
//! launch, and [`WindowTable::spans`] is the list of them: one interval per
//! launch, which is what `Fallback::Split { r }` asks for and what
//! `model_exec::fire::walk` loops over.
//!
//! # What this is NOT allowed to be
//!
//! Slow. This runs every fire, on the host, in front of a launch that costs
//! tens of microseconds. So: one pass over the lanes to count, one pass per
//! present class to seriate, no allocation per lane, and the word -> class
//! lookup memoized across the fire — a batch of 256 lanes carries at most a
//! handful of distinct words, and `ClassTable::class_of` is a scan over the
//! sweep's dedup'd word lists rather than a hash.

use model_compiler::{ArenaMap, Budget, Budgets, ClassOrder, CompiledModel, Extent, PatchLadder};
use model_ir::{ClassSet, PerAxis, RowAxis, ValueId};

use crate::fire::Fault;
use crate::{Error, Result};

/// One request inside a fire, as the runtime submits it.
///
/// THE TWO NUMBERS ARE THE WHOLE SUBMISSION as far as composition is
/// concerned. `word` decides WHICH windows this lane is in — it is the one
/// genuinely new field of the palo submission contract (decision #18),
/// computed runtime-side by the model's own `Classify::of` — and `rows` decides
/// HOW MUCH of them it occupies. Everything else about a request (its pages,
/// its adapter, its tokens) rides in buffers the engine already binds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lane {
    /// The lane's fact bits. `Guard::Fact(bit)` indexes them.
    pub word: u64,
    /// How many token rows this lane contributes — 1 for a decode step, the
    /// prompt length for a prefill, `1 + drafts` for speculative rows.
    pub rows: u32,
    /// How many IMAGES this lane submitted — its share of the patch axis's
    /// lane space (multimodal §5.1).
    ///
    /// **ZERO FOR EVERY TEXT LANE, AND THAT IS THE WHOLE OF WHAT MAKES THE
    /// TWO AXES TWO.** A lane is one request of the token rectangle always
    /// and one-or-none-or-three requests of the patch rectangle, so the
    /// merged prefix sum the token axis is composed under — rows and lanes
    /// break at the same places — is exactly what does not hold over here.
    pub images: u32,
    /// How many PATCH rows those images total, concatenated.
    pub patches: u32,
}

impl Lane {
    /// A lane of `rows` token rows whose facts are `word`, carrying no image.
    ///
    /// THE TEXT LANE, AND IT STATES ZERO RATHER THAN DEFAULTING IT: a request
    /// that submitted no image HAS no images and no patch rows, which is a
    /// true answer and not an absent one.
    #[must_use]
    pub fn new(word: u64, rows: u32) -> Lane {
        Lane {
            word,
            rows,
            images: 0,
            patches: 0,
        }
    }

    /// The same lane, carrying `images` images of `patches` patch rows in
    /// total.
    #[must_use]
    pub fn with_images(word: u64, rows: u32, images: u32, patches: u32) -> Lane {
        Lane {
            word,
            rows,
            images,
            patches,
        }
    }

    /// **WHAT THIS REQUEST CONTRIBUTES TO ONE ROW SPACE** — `(rows, lanes)`,
    /// where "lanes" is that rectangle's own request count.
    ///
    /// **THE ONE PLACE THE SUBMISSION'S FOUR NUMBERS MEET THE AXIS INDEX.**
    /// A `Lane` names its counts because the SUBMISSION does — a runtime
    /// writes `rows` and `images` from two different facts about the request
    /// — and every pass downstream wants them per axis instead. So the pair
    /// is read here, once, and `seriate_axis` does the same arithmetic over
    /// both without knowing which it is doing.
    ///
    /// **A TOKEN LANE IS ALWAYS ONE LANE AND A PATCH LANE IS ZERO OR MORE
    /// IMAGES**, which is the whole asymmetry of the two axes stated as two
    /// expressions rather than as two loops: `1` is what makes the token
    /// axis's merged prefix sum work (a request is a request), and
    /// `self.images` is what makes the patch axis's not (a request may be no
    /// images at all).
    #[must_use]
    pub fn on(self, axis: RowAxis) -> (u32, u32) {
        match axis {
            RowAxis::Tokens => (self.rows, 1),
            RowAxis::Patches => (self.patches, self.images),
        }
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
///
/// **ONE STRUCT, TWO AXES** (multimodal §5.1). The patch axis's window is the
/// same four numbers about a different rectangle — its rows are PATCH rows and
/// its lanes are IMAGES — so [`Composition::patch_classes`] is a second
/// [`WindowTable`] of these and not a second struct. Every question the walk
/// asks a window table (`rows_of`, `spans`, `span`) is the same question over
/// there, which is what a second struct would have made two implementations
/// of.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ClassWindow {
    /// First row of this class's interval — token rows in the token table,
    /// patch rows in the patch one.
    pub row_offset: u32,
    /// How many rows it has. Zero for a class no lane is in.
    pub rows: u32,
    /// First lane of this class's interval — requests in the token table,
    /// images in the patch one.
    pub lane_offset: u32,
    /// How many lanes it has.
    pub lanes: u32,
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
/// `ClassTable::node_mask` was built against. A table of the wrong width does not
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
    /// how many launches it would take instead.
    ///
    /// **`Err` IS NOT THE END OF THE FIRE ANY MORE.** It is the question "can
    /// this consumer be ONE launch", which is what a caller that has nowhere
    /// to put a second one still needs to ask; the caller that can put one
    /// there asks [`spans`](WindowTable::spans) instead and gets every run.
    /// What an `Err` means about the ARTIFACT is read off
    /// [`FallbackTable`](model_compiler::FallbackTable): a fragmented mask
    /// with a row there is P4 saying "I could not seat this one, here is what
    /// to do instead", and a fragmented mask with no row is a bake-integrity
    /// failure, because P4 promised this mask consecutive and the fire found
    /// it broken.
    ///
    /// # Errors
    ///
    /// The number of runs the mask covers, when that is more than one.
    pub fn span(&self, mask: &ClassSet) -> core::result::Result<Option<MaskSpan>, usize> {
        let runs = self.spans(mask);
        match runs.len() {
            0 => Ok(None),
            1 => Ok(Some(runs[0])),
            more => Err(more),
        }
    }

    /// **EVERY interval this mask covers**, ascending — one per launch.
    ///
    /// THE SLOW PATH'S WHOLE ARITHMETIC. `Fallback::Split { r }` says a
    /// consumer P4 could not seat runs `r` times, once per maximal interval of
    /// its class set, and this is that list: the empty vector for a window no
    /// lane is in, one entry for the consecutive case P4 exists to produce,
    /// and `r` entries for the one it could not. A caller that walks it
    /// dispatches the region's nodes once per entry, each over its own
    /// pointer and extent, and the union of the entries is exactly the rows
    /// [`rows_of`](WindowTable::rows_of) counts — which is the invariant that
    /// makes the split a slow path rather than a different answer.
    ///
    /// Contiguity is decided by the offsets themselves: two of the mask's
    /// classes are one run iff the second's rows begin where the first's end,
    /// so a class the mask does NOT contain standing between them breaks the
    /// run and a zero-row class does not. Zero-row classes are invisible
    /// throughout — a class no lane is in occupies no rows to straddle, which
    /// is what makes an all-decode fire's prefill mask an empty window rather
    /// than a fragmented one, and what makes a fire that happens to carry no
    /// lane in the class between two runs get ONE launch where the baked
    /// order promises none.
    ///
    /// Rows and lanes break at the same places and are therefore merged
    /// together rather than by two passes that could disagree: a class with
    /// rows has lanes (a lane contributes at least one row) and a class with
    /// lanes has rows, so the two prefix sums have their gaps at exactly the
    /// same classes.
    #[must_use]
    pub fn spans(&self, mask: &ClassSet) -> Vec<MaskSpan> {
        let mut out = Vec::new();
        self.spans_into(mask, &mut out);
        out
    }

    /// [`spans`](WindowTable::spans), into a buffer the caller keeps.
    ///
    /// THE FIRE PATH'S FORM OF IT. `walk` asks this question once per region
    /// of the template — hundreds of times per fire, twice for a shell that
    /// captures — and a `Vec` per region is hundreds of allocations in front
    /// of a launch that costs tens of microseconds. One buffer, reused down
    /// the template, is the same answer with none of them.
    pub fn spans_into(&self, mask: &ClassSet, out: &mut Vec<MaskSpan>) {
        out.clear();
        for class in mask.iter() {
            let window = self.class(class);
            if window.rows == 0 {
                continue;
            }
            out.push(MaskSpan {
                row_offset: window.row_offset,
                rows: window.rows,
                lane_offset: window.lane_offset,
                lanes: window.lanes,
            });
        }
        out.sort_unstable_by_key(|span| span.row_offset);

        // Merge in place: `open` is how many runs are settled, and the entry
        // under it is the one still growing.
        let mut open = 0usize;
        for read in 0..out.len() {
            let span = out[read];
            let grows = open > 0 && {
                let last = out[open - 1];
                last.row_offset + last.rows == span.row_offset
            };
            if grows {
                out[open - 1].rows += span.rows;
                out[open - 1].lanes += span.lanes;
            } else {
                out[open] = span;
                open += 1;
            }
        }
        out.truncate(open);
    }

}

/// One lane, placed.
///
/// `source` IS THE POINT OF THE RECORD. The runtime submitted lane 4; the fire
/// runs it third, because seriation groups the classes. Every per-lane buffer
/// the engine binds — pages, positions, adapter ids — is written in submission
/// order, so the descriptor has to carry the permutation or the device reads
/// somebody else's geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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
    /// Its first PATCH row in the second seriation, which is a different
    /// order over the same classes.
    ///
    /// **A SECOND OFFSET AND NOT A SECOND RECORD**, because the permutation
    /// is the same one: the runtime wrote this lane's images in submission
    /// order beside its tokens, so the vector that has to be reordered is the
    /// same vector, and one lane record carrying both offsets is what keeps
    /// the two from being read against each other's `source`.
    pub patch_offset: u32,
    /// How many patch rows it contributes.
    pub patches: u32,
    /// Its first IMAGE in the patch seriation — where its run of the
    /// `images + 1` indptr begins.
    pub image_offset: u32,
    /// How many images it contributes.
    pub images: u32,
}

/// One fire's composition: which windows have rows, where, and in what order.
///
/// WHAT IT IS NOT is a schedule. The script is baked (`CompiledModel::template`) and
/// the same one runs every fire; this is the DATA that script reads. Which is
/// why a composition can be built, thrown away and built again 5000 times a
/// second without anything being recompiled or recaptured (design §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Composition {
    lanes: Vec<LaneRow>,
    /// **ONE [`AxisComposition`] PER ROW SPACE, ADDRESSED BY THE AXIS.**
    ///
    /// This used to be five fields and their five `patch_`-prefixed mirrors,
    /// and the accessors below are what is left of them — thin forwards, so
    /// that a consumer holding `patch_classes()` holds the same table it
    /// always did. What the array buys is that the two entries are now ONE
    /// derivation: `seriate_axis` fills each of them and no line of this
    /// file states the patch axis's arithmetic a second time.
    ///
    /// A text-only fire's patch entry is the ZERO SERIATION — one zero
    /// window per class of the artifact, an empty order, no rows, no lanes
    /// and the zero rung. Zero windows and not NO windows: every reader
    /// indexes the table by class, so an absent table would be a special case
    /// at each of them, and the G4 invariant is precisely that there is none.
    axes: PerAxis<AxisComposition>,
}

/// **ONE ROW SPACE'S WHOLE COMPOSITION** — its window table, the order its
/// classes stand in, its two totals and the rung they round up to.
///
/// **THE FIVE NUMBERS A SERIATION PRODUCES, AND EXACTLY THOSE.** They were
/// five fields of [`Composition`] and five more beside them with `patch_` in
/// front, kept in step by hand: a `patch_classes` for every `classes`, a
/// `patch_bucket` for every `bucket`. The pairing was the bug surface — two
/// fields that must agree about one rectangle and nothing making them — and
/// this is the pairing collapsed into one struct that the axis indexes.
///
/// **`rows` AND `lanes` ARE THAT AXIS'S OWN, WHICH IS WHY THEY ARE NOT
/// NAMED FOR EITHER.** On the token axis `rows` are token rows and `lanes`
/// are requests; on the patch axis `rows` are patch rows and `lanes` are
/// IMAGES. The arithmetic that produced them does not know the difference —
/// [`Lane::on`] is the one place it is stated — and neither does anything
/// that reads them back through the axis.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AxisComposition {
    /// One [`ClassWindow`] per class of the artifact, indexed by class — this
    /// rectangle's seriation.
    pub classes: WindowTable,
    /// The classes this fire has rows in on this axis, in the order their
    /// rows stand.
    pub order: Vec<u32>,
    /// This rectangle's total rows.
    pub rows: u32,
    /// This rectangle's total lanes — requests on the token axis, images on
    /// the patch one.
    pub lanes: u32,
    /// The rung `rows` rounds up to, out of this axis's own lattice — which
    /// recorded graph this fire's unit launches.
    pub bucket: u32,
}

impl Composition {
    /// The lanes, in fire order: grouped by class, submission order kept
    /// inside each class.
    #[must_use]
    pub fn lanes(&self) -> &[LaneRow] {
        &self.lanes
    }

    /// **THIS FIRE'S COMPOSITION ON ONE ROW AXIS** — the table, the order and
    /// the three counts, in one piece.
    ///
    /// **THE ONE READ, AND EVERY ACCESSOR BELOW IS IT WITH AN AXIS SPELLED.**
    /// A caller that already holds a region's axis (the walk, the window
    /// table, the shells' arming passes) asks this and is done; a caller that
    /// means the token rectangle in particular says so by name, which is what
    /// keeps every pre-campaign call site in the tree spelled the way it was.
    #[must_use]
    pub fn axis(&self, axis: RowAxis) -> &AxisComposition {
        &self.axes[axis]
    }

    /// This fire's window table on one axis — [`axis`](Composition::axis)'s
    /// table half, which is what the window pass and the descriptor take.
    #[must_use]
    pub fn table(&self, axis: RowAxis) -> &WindowTable {
        &self.axes[axis].classes
    }

    /// How many lanes this fire carries.
    #[must_use]
    pub fn lane_count(&self) -> u32 {
        self.axes[RowAxis::Tokens].lanes
    }

    /// The window table, indexed by class.
    #[must_use]
    pub fn classes(&self) -> &WindowTable {
        self.table(RowAxis::Tokens)
    }

    /// The classes this fire has lanes in, in the order their rows stand.
    ///
    /// THE COMPOSITION, LITERALLY: design §0's word for "which windows are
    /// non-empty in a fire" is exactly this list.
    #[must_use]
    pub fn present(&self) -> &[u32] {
        &self.axes[RowAxis::Tokens].order
    }

    /// Total token rows.
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.axes[RowAxis::Tokens].rows
    }

    /// The shape bucket these rows round up to — which recorded graph this
    /// fire launches (design §5). Equal to [`rows`](Composition::rows) when
    /// the budget lists no lattice.
    #[must_use]
    pub fn bucket(&self) -> u32 {
        self.axes[RowAxis::Tokens].bucket
    }

    /// How many PATCH rows this fire carries — the second row axis's count,
    /// out of the second seriation.
    ///
    /// Zero for a text-only fire and for every fire of a text-only artifact,
    /// and TRUE rather than defaulted in both cases: a fire whose lanes
    /// submitted no image has no patch rows.
    #[must_use]
    pub fn patch_rows(&self) -> u32 {
        self.axes[RowAxis::Patches].rows
    }

    /// How many IMAGES this fire carries — the patch axis's lane count.
    #[must_use]
    pub fn images(&self) -> u32 {
        self.axes[RowAxis::Patches].lanes
    }

    /// The PATCH window table, indexed by class.
    ///
    /// **THE SECOND SERIATION'S OWN ANSWER, IN THE FIRST'S VOCABULARY.** Its
    /// rows are patch rows and its lanes are images, and the classes are the
    /// same classes — but their ORDER is not: the patch axis carries its own
    /// [`ClassOrder`](model_compiler::compiled::AxisPlan), so a class standing
    /// third in the token rectangle may stand first in the patch one. That is
    /// the whole content of "patches do not break where tokens do".
    ///
    /// Every class's window is zero for an artifact with no patch axis, which
    /// is what makes a text-only fire's second table free rather than absent.
    #[must_use]
    pub fn patch_classes(&self) -> &WindowTable {
        self.table(RowAxis::Patches)
    }

    /// The classes this fire has IMAGES in, in the order their patch rows
    /// stand.
    #[must_use]
    pub fn patch_present(&self) -> &[u32] {
        &self.axes[RowAxis::Patches].order
    }

    /// The PATCH rung these patch rows round up to — which tower exec this
    /// fire launches. Equal to [`patch_rows`](Composition::patch_rows) when
    /// the ladder lists no rungs, exactly as the token bucket is.
    #[must_use]
    pub fn patch_bucket(&self) -> u32 {
        self.axes[RowAxis::Patches].bucket
    }

    /// This fire's four counts, as the carve's arithmetic takes them.
    ///
    /// **THE ONE PLACE THE AXIS INDEX GOES BACK TO NAMES**, because
    /// `FireRows` is the compiler's record and its four fields are the four
    /// SYMBOLS the arena sizes (`Dim::Tokens`, `Dim::Lanes`, `Dim::Patches`,
    /// `Dim::Images`) rather than two axes' worth of a pair. Read off the
    /// array so that the two rectangles' counts cannot part from the tables
    /// they were summed with.
    #[must_use]
    pub fn fire_rows(&self) -> model_compiler::FireRows {
        let tokens = &self.axes[RowAxis::Tokens];
        let patches = &self.axes[RowAxis::Patches];
        model_compiler::FireRows {
            tokens: u64::from(tokens.rows),
            lanes: u64::from(tokens.lanes),
            patches: u64::from(patches.rows),
            images: u64::from(patches.lanes),
        }
    }

    /// Where a value's rectangle sits in THIS fire.
    ///
    /// The offset is static — decided once by P7, the same in every bucket —
    /// and only the length moves, which is the whole reason a composition
    /// never triggers a recapture. This is the one call that puts the two
    /// halves together, and it is a delegation because the arithmetic belongs
    /// to the carve that chose the offsets.
    #[must_use]
    pub fn value_window(&self, arena: &ArenaMap, value: ValueId) -> Option<Extent> {
        arena.window(value, self.fire_rows())
    }
}

/// Compose one fire ON THE TOKEN AXIS — the door every text-only deployment
/// uses, and the one every pre-campaign caller in the tree holds.
///
/// [`compose_axes`] is this same seriation told about a second row axis, and
/// this is exactly it with no patch ladder — which is the relationship
/// `model_compiler::compile` has to `compile_axes`, kept on purpose so the
/// two halves of the campaign read the same way.
///
/// # Errors
///
/// [`Fault::UnknownWord`] for a lane no class of this artifact admits,
/// [`Fault::EmptyLane`] for a lane of no rows, [`Fault::TooManyLanes`] /
/// [`Fault::TooManyRows`] for a fire past the ceilings the arena was cut at,
/// and [`Fault::NoBucket`] for a fire above the whole bucket lattice. All five
/// are the same kind of thing: a batch this artifact cannot describe, refused
/// before a byte is written rather than launched into columns that are too
/// short for it. Plus [`Fault::Towerless`] for a lane that submitted images
/// through this door, which admits none.
pub fn compose(compiled: &CompiledModel, budget: &Budget, lanes: &[Lane]) -> Result<Composition> {
    seriate(compiled, budget, None, lanes)
}

/// Compose one fire over EVERY ROW AXIS the deployment admits.
///
/// **TWO SERIATIONS, NOT ONE WIDENED** (multimodal §5.1). The token pass is
/// unchanged, byte for byte; the patch pass runs beside it over the same
/// classes in a DIFFERENT order — the artifact's own `ClassOrder` for the
/// patch axis — with images where the lanes were. A fire whose lanes carry no
/// image gets a patch table of zero windows and a patch row count of zero,
/// which costs it one pass over a `Vec` of `(0, 0)` and nothing else.
///
/// # Errors
///
/// Everything [`compose`] refuses, plus the second axis's own three:
/// [`Fault::Towerless`] for images against an artifact with no patch axis
/// (M-1e refusal ii), [`Fault::PatchGeometry`] for a lane whose image count
/// and patch payload disagree (refusal i, host half), and
/// [`Fault::TooManyPatches`] / [`Fault::TooManyImages`] /
/// [`Fault::NoPatchBucket`] for a fire past the patch ceilings (refusal iii).
pub fn compose_axes(
    compiled: &CompiledModel,
    budgets: &Budgets,
    lanes: &[Lane],
) -> Result<Composition> {
    seriate(
        compiled,
        &budgets.tokens,
        budgets.patches.as_ref(),
        lanes,
    )
}

fn seriate(
    compiled: &CompiledModel,
    budget: &Budget,
    ladder: Option<&PatchLadder>,
    lanes: &[Lane],
) -> Result<Composition> {
    if lanes.len() > budget.max_lanes as usize {
        return Err(Fault::TooManyLanes {
            lanes: lanes.len(),
            max: budget.max_lanes,
        }
        .into());
    }

    // Pass one: every lane's class, and what each class adds up to. The memo
    // is a `Vec` and the lookup is linear because a fire's DISTINCT words are
    // few — two in the diagram above, a handful in the worst catalog plan —
    // and a hash of a `u64` costs more than the scan it would replace.
    let count = compiled.classes.classes.len();
    let mut memo: Vec<(u64, u32)> = Vec::new();
    let mut of_lane: Vec<u32> = Vec::with_capacity(lanes.len());
    // **THE TALLIES, ONE SET PER ROW AXIS** — `(rows, lanes)` per class, on
    // each rectangle. The patch entry stays all-zero for a text-only fire,
    // and the passes below read that as "no class is present on this axis"
    // without a branch, exactly as the hand-kept second vector did.
    let mut tally: PerAxis<Vec<(u64, u32)>> = PerAxis::from_fn(|_| vec![(0, 0); count]);
    // Each rectangle's `(rows, lanes)` total, in `u64` because the ceilings
    // below are what turn a sum into a `u32`.
    let mut totals: PerAxis<(u64, u64)> = PerAxis::from_fn(|_| (0, 0));
    // Whether this ARTIFACT has anywhere for a patch row to go. Read off the
    // bake rather than off the budget: a deployment may admit a patch ladder
    // for a text-only model (and pay nothing for it, which is the G4
    // invariant), so what decides refusal (ii) is whether the PLAN states the
    // axis.
    let towered = compiled.order_for(RowAxis::Patches).is_some();

    for (i, lane) in lanes.iter().enumerate() {
        let i = i as u32;
        if lane.rows == 0 {
            return Err(Fault::EmptyLane { lane: i }.into());
        }
        // **REFUSAL (i), HOST HALF.** An image is at least one patch row and
        // a patch row belongs to some image, so a lane that states one number
        // without the other has a geometry and a payload that were written by
        // two different beliefs about what it carries.
        if (lane.images == 0) != (lane.patches == 0) {
            return Err(Fault::PatchGeometry {
                lane: i,
                images: lane.images,
                patches: lane.patches,
            }
            .into());
        }
        // **REFUSAL (ii).** Images against a text with no patch axis — no
        // tower to run them through, no exec to launch, no rectangle to land
        // them in. Refused rather than dropped, because dropping them answers
        // the caller's image with the continuation of their text.
        if lane.images > 0 && !(towered && ladder.is_some()) {
            return Err(Fault::Towerless { lane: i }.into());
        }
        // THE WORD IS MASKED TO THE BITS THE SWEEP RAN OVER. A lane's word
        // comes from the model's `Classify`, which packs every fact the model
        // COMPUTES; `ClassTable::mask` is the bits some guard READS. A model may
        // state a fact it does not split on — and then a lane that sets it
        // carries a word no class enumerates, which would surface as
        // `UnknownWord`: "the runtime and the shell disagree about what is
        // loaded", said about two halves that agree perfectly.
        let word = lane.word & compiled.classes.mask;
        let class = match memo.iter().find(|(seen, _)| *seen == word) {
            Some((_, class)) => *class,
            None => {
                let class = compiled.classes.class_of(word).ok_or(Fault::UnknownWord {
                    lane: i,
                    word: lane.word,
                })? as u32;
                memo.push((word, class));
                class
            }
        };
        // `class_of` answers a position in `classes.classes`, so this index is
        // in range by construction; the tally is sized from the same list.
        //
        // **AND THE TWO SUMS ARE ONE SUM SAID PER AXIS.** A class's rows are
        // the rows the lanes that resolved to it contribute on that
        // rectangle, and its "lanes" are their requests of it — [`Lane::on`]
        // is where a token lane's `1` and a patch lane's image count part
        // company, which is where the two axes stop agreeing and the only
        // place this file says so.
        for axis in RowAxis::ALL {
            let (rows, images) = lane.on(axis);
            tally[axis][class as usize].0 += u64::from(rows);
            tally[axis][class as usize].1 += images;
            totals[axis].0 += u64::from(rows);
            totals[axis].1 += u64::from(images);
        }
        of_lane.push(class);
    }

    let rows = totals[RowAxis::Tokens].0;
    if rows > u64::from(budget.max_tokens) {
        return Err(Fault::TooManyRows {
            rows,
            max: budget.max_tokens,
        }
        .into());
    }
    let rows = rows as u32;

    // **REFUSAL (iii): THE PATCH CEILINGS, ON THEIR OWN TERMS.** Both are
    // checked even when the ladder is absent, because `patches` is then zero
    // and the comparison is free — and stating them unconditionally is what
    // keeps "a text-only fire is refused by the same code that admits it"
    // true rather than a claim about a branch not taken.
    let (patches, images) = totals[RowAxis::Patches];
    let (max_patches, max_images) = ladder.map_or((0, 0), |l| (l.max_patches, l.max_images));
    if patches > u64::from(max_patches) {
        return Err(Fault::TooManyPatches {
            patches,
            max: max_patches,
        }
        .into());
    }
    if images > u64::from(max_images) {
        return Err(Fault::TooManyImages {
            images,
            max: max_images,
        }
        .into());
    }
    let patches = patches as u32;

    // The rungs, one per axis, taken BEFORE the seriations for the reason
    // they were taken inside the struct literal that used to stand at the
    // bottom of this function: nothing between here and there can fail, and
    // the token refusal is owed before the patch one either way.
    let buckets = PerAxis::new([bucket_of(budget, rows)?, patch_bucket_of(ladder, patches)?]);

    // Where each SUBMITTED lane's rows land ON EACH AXIS, computed in that
    // axis's own order and read back in the token one. A side table rather
    // than a lane list per axis: the permutation is one permutation, and two
    // lists indexed by two orders is how a `source` comes to be read against
    // the wrong record.
    let mut placed: Vec<PerAxis<(u32, u32)>> = vec![PerAxis::from_fn(|_| (0, 0)); lanes.len()];

    // Both rectangles' totals, narrowed — every one of the four has just been
    // admitted by the ceiling it is owed to, which is the only thing that
    // makes the cast a conversion rather than a truncation. The token
    // entry's lane count IS `lanes.len()`, because a request is one request
    // of the token rectangle ([`Lane::on`]); reading it out of the same
    // tally the patch entry comes out of is what keeps the pair one
    // derivation (`Lane::on`).
    let checked: PerAxis<(u32, u32)> =
        PerAxis::from_fn(|axis| (totals[axis].0 as u32, totals[axis].1 as u32));

    // **PASS TWO, ONCE PER ROW AXIS.** The prefix sums and the lane
    // placement, over that axis's own baked order and its own tallies —
    // `seriate_axis` is the whole of it and this loop is the whole of what
    // "the patch axis costs the token axis nothing" means now: the second
    // entry is the first function called again, not a second copy of it.
    let axes = PerAxis::from_fn(|axis| {
        seriate_axis(
            compiled.order_for(axis),
            axis,
            &tally[axis],
            lanes,
            &of_lane,
            checked[axis],
            buckets[axis],
            &mut placed,
        )
    });

    // Pass three: the lane records, assembled. Class by class in the TOKEN
    // fire order, and INSIDE a class in submission order — that inner
    // stability is not decoration. The runtime's order is the order its
    // per-lane buffers are written in, and a fire that reshuffles lanes it had
    // no reason to reshuffle churns every pointer the previous fire had warm.
    //
    // **AND EVERY OFFSET IN A RECORD IS READ OUT OF `placed` RATHER THAN
    // RE-ACCUMULATED HERE**, which is what makes the token axis's placement
    // the same arithmetic as the patch axis's instead of a second one that
    // happens to agree.
    let mut seriated: Vec<LaneRow> = Vec::with_capacity(lanes.len());
    for &class in &axes[RowAxis::Tokens].order {
        for (i, lane) in lanes.iter().enumerate() {
            if of_lane[i] != class {
                continue;
            }
            let token = placed[i][RowAxis::Tokens];
            let patch = placed[i][RowAxis::Patches];
            seriated.push(LaneRow {
                source: i as u32,
                word: lane.word,
                class,
                row_offset: token.0,
                rows: lane.rows,
                patch_offset: patch.0,
                patches: lane.patches,
                image_offset: patch.1,
                images: lane.images,
            });
        }
    }

    Ok(Composition {
        lanes: seriated,
        axes,
    })
}

/// **ONE ROW AXIS'S SERIATION** — the prefix sums that place its classes and
/// the walk that places its lanes inside them, over the order this artifact
/// baked for THAT axis.
///
/// **ONE FUNCTION AND NOT TWO LOOPS, WHICH IS THE WHOLE OF WAVE B HERE.**
/// The token pass and the patch pass were two blocks of this arithmetic, and
/// they had already drifted into two SHAPES: the token pass summed its
/// classes and then placed its lanes in a third pass over the same order,
/// while the patch pass placed its lanes INSIDE the sum. Neither shape was
/// wrong; two of them is what is wrong, because the day one grows a clause
/// the other does not is a day the two rectangles' offsets stop describing
/// one submission.
///
/// The patch pass's "extra step" was exactly that inlining plus one clause —
/// **it skips a lane with no lanes of its own** — and the clause survives
/// here as a parameter of the arithmetic rather than as a second body: a lane
/// contributing zero requests to this rectangle is not placed in it, which on
/// the token axis never fires (a lane is one request always, and a lane of no
/// rows is [`Fault::EmptyLane`] one pass earlier) and on the patch axis is
/// every text lane in the batch.
///
/// **AN ABSENT PLAN IS THE ZERO SERIATION AND NOT A BRANCH AT THE CALLER.**
/// An artifact that states no patch axis has no order for it, and what that
/// answers is a table of zero windows, an empty present-set and the totals it
/// was handed — which are themselves zero, because a lane that carried an
/// image against such an artifact was refused as [`Fault::Towerless`] before
/// this was reached.
#[allow(clippy::too_many_arguments)]
fn seriate_axis(
    plan: Option<&ClassOrder>,
    axis: RowAxis,
    tally: &[(u64, u32)],
    lanes: &[Lane],
    of_lane: &[u32],
    totals: (u32, u32),
    bucket: u32,
    placed: &mut [PerAxis<(u32, u32)>],
) -> AxisComposition {
    let count = tally.len();
    let mut classes = vec![ClassWindow::default(); count];
    let (rows, lane_total) = totals;
    let Some(plan) = plan else {
        return AxisComposition {
            classes: WindowTable::new(classes),
            order: Vec::new(),
            rows,
            lanes: lane_total,
            bucket,
        };
    };

    // The classes this fire has rows in on this axis, in the order their rows
    // will stand.
    //
    // **ONE BAKED ANSWER, FILTERED, AND NOT A TABLE WITH A ROW PER
    // COMPOSITION.** There are `2^classes` compositions and one C1P instance;
    // a sub-order of an ordering that makes a set consecutive still makes that
    // set consecutive, so dropping the absent classes out of the global order
    // is the whole per-fire computation (design §3). P4 owns that filter —
    // this asks it rather than re-deriving it from the tree, because a second
    // reading of one answer is a second answer waiting to disagree. And P4
    // solved the instance ONCE PER AXIS, because the consumers it has to seat
    // are different consumers, which is why the plan is an argument here.
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
    let order: Vec<u32> = plan
        .class_order(&present, None)
        .into_iter()
        .map(u32::from)
        .collect();

    let (mut row_at, mut lane_at) = (0u32, 0u32);
    for &class in &order {
        let (class_rows, class_lanes) = tally[class as usize];
        classes[class as usize] = ClassWindow {
            row_offset: row_at,
            rows: class_rows as u32,
            lane_offset: lane_at,
            lanes: class_lanes,
        };
        // Inside a class, submission order — the same inner stability the
        // fire order keeps between classes, and for the same reason: the
        // runtime wrote this lane's rows in the order it submitted them.
        let (mut lane_row_at, mut lane_lane_at) = (row_at, lane_at);
        for (i, lane) in lanes.iter().enumerate() {
            if of_lane[i] != class {
                continue;
            }
            let (lane_rows, lane_lanes) = lane.on(axis);
            // The header's one clause: a request that is no request of THIS
            // rectangle is not placed in it, and its offsets stay the zero
            // they were built at.
            if lane_lanes == 0 {
                continue;
            }
            placed[i][axis] = (lane_row_at, lane_lane_at);
            lane_row_at += lane_rows;
            lane_lane_at += lane_lanes;
        }
        row_at += class_rows as u32;
        lane_at += class_lanes;
    }

    AxisComposition {
        classes: WindowTable::new(classes),
        order,
        rows,
        lanes: lane_total,
        bucket,
    }
}

/// **THE SMALLEST PATCH RUNG THAT HOLDS THESE PATCH ROWS** — [`bucket_of`]'s
/// question on the second row axis, out of the lattice that axis declared
/// (`PatchLadder::buckets`).
///
/// THREE CLAUSES, AND THEY ARE `bucket_of`'S THREE WITH ONE IN FRONT:
///
/// * **no ladder at all** is an artifact with no patch axis, and `0` is the
///   only honest rung for a row space that does not exist;
/// * **no patch rows** is the clause the token axis has no counterpart for. A
///   fire always has token rows — a lane of none is [`Fault::EmptyLane`] — but
///   a fire of a towered artifact may perfectly well carry no image, and an
///   axis-empty fire simply does not launch that unit's exec (multimodal §1).
///   Rounding its zero up to the lattice floor would name a rung for a tower
///   that does not run, so zero rows is the zero rung;
/// * **past the top rung** is [`Fault::NoPatchBucket`], for the reason
///   `bucket_of` refuses past the budget's top: the lattice is what the load
///   carved against, and a fire above it has no ceiling to be admitted under.
///
/// Everything below that is [`rung_of`], as it is over there — one derivation
/// with two doors, so the two axes cannot come to round differently. An empty
/// `buckets` vector falls through the same `_` arm and quantizes nothing.
fn patch_bucket_of(ladder: Option<&PatchLadder>, patches: u32) -> Result<u32> {
    let Some(ladder) = ladder else {
        return Ok(0);
    };
    if patches == 0 {
        return Ok(0);
    }
    match ladder.buckets.last().copied() {
        Some(top) if patches > top => Err(Error::Fire(Fault::NoPatchBucket { patches, top })),
        _ => Ok(rung_of(&ladder.buckets, patches)),
    }
}

/// The smallest bucket that holds these rows.
///
/// AN EMPTY LATTICE IS NOT AN ERROR: a budget that lists no buckets is a
/// deployment that has not chosen a shape lattice, and the honest bucket for a
/// fire of `rows` rows is then `rows` itself — one graph per size, which is
/// what a golden-path eager walk wants and what a test builds with
/// `Budget::new`.
///
/// PAST THE TOP RUNG IS [`Fault::NoBucket`], and it is the fire's TOTAL that
/// is being judged: the budget's top is what the load carved every ceiling
/// against, so a fire above it is refused rather than quantized to something
/// nothing reserved for. The rounding itself is [`rung_of`].
fn bucket_of(budget: &Budget, rows: u32) -> Result<u32> {
    match budget.buckets.last().copied() {
        Some(top) if rows > top => Err(Error::Fire(Fault::NoBucket { rows, top })),
        _ => Ok(rung_of(&budget.buckets, rows)),
    }
}

/// **THE SMALLEST RUNG OF THIS LATTICE THAT HOLDS `rows`, FOR A COUNT THAT IS
/// PART OF A FIRE THE CEILING ABOVE HAS ALREADY ADMITTED.**
///
/// [`bucket_of`]'s arithmetic without [`bucket_of`]'s refusal, and it is one
/// reading rather than two: the refusal is about the FIRE's total, which is
/// the only count a budget's top rung is a ceiling on. Every count this
/// answers for is a part of that total — one CLASS's rows, out of the same
/// seriation the total was summed over — so it is bounded by a number already
/// checked, and an empty lattice answers `rows` here for the reason it
/// answers `rows` there: a deployment that chose no lattice quantizes
/// nothing.
///
/// **AND THE SAME LATTICE, DELIBERATELY.** The rungs quantize arithmetic
/// drift wherever rows are COUNTED — that is what
/// [`Composition::bucket`](Composition::bucket) means and it is not a claim
/// about the fire's total in particular — so a second, coarser ladder built
/// beside this one would be a second answer to the same question.
///
/// **THE CALLERS ARE [`bucket_of`] AND [`patch_bucket_of`], AND THAT IS THE
/// WHOLE LIST.** This doc used to name `engine_cuda::record::Ladder` as the
/// reader, and it was true when a body key's rung was this function over a
/// class's LIVE rows. It is not true since the canon: `Ladder::rung` answers
/// from the key's own coordinates — the bucket, and whether the class is a
/// decode class — because a number the fire MEASURED inside the key the fire
/// is looked up by is what made four decode rows and seven decode rows two
/// captures. So `Ladder` deliberately does not read this, and the one place a
/// count still meets the lattice is a fire's own TOTAL, on either axis.
#[must_use]
pub fn rung_of(buckets: &[u32], rows: u32) -> u32 {
    buckets
        .iter()
        .copied()
        .find(|rung| *rung >= rows)
        .unwrap_or(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fire::fixture::{Build, fact};
    use crate::{Error, fire::Fault};
    use model_compiler::{DeviceProfile, compile};
    use model_ir::{Guard, ValueId};

    /// A deployment small enough to state in an assert: 8 lanes, 64 rows, no
    /// bucket lattice.
    fn budget() -> Budget {
        Budget::new(8, 64)
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
        let plan = b.prepare(Guard::Always); // node 0 — prepare, every class
        let q = b.op(x, 4, Guard::Always); // node 1 — every class
        let d = b.decode(q, plan, fact(0)); // node 2 — the decode window
        let p = b.op(q, 4, Guard::not(fact(0))); // node 3 — the prefill window
        let o = b.merge(&[(d, fact(0)), (p, Guard::not(fact(0)))], 4);
        let y = b.op(o, 4, Guard::Always); // node 4 — every class again
        b.out(y);
        (b, y)
    }

    /// The nodes of [`diagram`], by the name the design gives them.
    const SHARED: u32 = 1;
    const DECODE: u32 = 2;
    const PREFILL: u32 = 3;

    /// The rows one node runs over — its region's mask, against this fire.
    fn rows_of(compiled: &CompiledModel, fire: &Composition, node: u32) -> u32 {
        let region = compiled
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
        let compiled = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");
        let lanes = [
            Lane::new(0, 7),
            Lane::new(0, 3),
            Lane::new(1, 1),
            Lane::new(1, 1),
            Lane::new(1, 1),
        ];
        let fire = compose(&compiled, &budget(), &lanes).expect("composes");

        assert_eq!(fire.rows(), 13);
        assert_eq!(fire.lane_count(), 5);
        assert_eq!(fire.present().len(), 2);

        let prefill = compiled.classes.class_of(0).expect("word 0 is a class");
        let decode = compiled.classes.class_of(1).expect("word 1 is a class");
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
        assert_eq!(rows_of(&compiled, &fire, SHARED), 13);
        assert_eq!(rows_of(&compiled, &fire, DECODE), 3);
        assert_eq!(rows_of(&compiled, &fire, PREFILL), 10);
    }

    #[test]
    fn an_all_decode_fire_leaves_the_prefill_window_empty() {
        let (b, _) = diagram();
        let compiled = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");
        let lanes = [Lane::new(1, 1), Lane::new(1, 1), Lane::new(1, 1)];
        let fire = compose(&compiled, &budget(), &lanes).expect("composes");

        assert_eq!(fire.rows(), 3);
        assert_eq!(fire.present().len(), 1, "one class has lanes");
        // An empty window is a zero, not an absence: the class still has a
        // window, and it has no rows.
        assert_eq!(rows_of(&compiled, &fire, PREFILL), 0);
        assert_eq!(rows_of(&compiled, &fire, DECODE), 3);
        assert_eq!(rows_of(&compiled, &fire, SHARED), 3);
        // The decode class starts at row 0 — an absent class occupies nothing.
        let decode = compiled.classes.class_of(1).expect("word 1 is a class");
        assert_eq!(fire.classes().class(decode).row_offset, 0);
    }

    #[test]
    fn lanes_keep_submission_order_inside_a_class() {
        let (b, _) = diagram();
        let compiled = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");
        // Interleaved on submission: prefill, decode, prefill, decode.
        let lanes = [
            Lane::new(0, 5),
            Lane::new(1, 1),
            Lane::new(0, 2),
            Lane::new(1, 1),
        ];
        let fire = compose(&compiled, &budget(), &lanes).expect("composes");

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
        let q = b.op(x, 4, Guard::Always);
        let one = b.op(q, 4, fact(0));
        let other = b.op(q, 4, Guard::not(fact(0)));
        let o = b.merge(&[(one, fact(0)), (other, Guard::not(fact(0)))], 4);
        let m = b.op(o, 4, fact(1));
        let um = b.op(o, 4, Guard::not(fact(1)));
        let out = b.merge(&[(m, fact(1)), (um, Guard::not(fact(1)))], 4);
        b.out(out);
        let compiled = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");
        assert_eq!(compiled.classes.classes.len(), 4);

        // `one` is node 1, guarded `qo_one`: it runs in the two classes whose
        // word has bit 0 set, words 0b01 and 0b11.
        let region = compiled
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
        let fire = compose(&compiled, &budget(), &all).expect("composes");
        let spans = fire.classes().spans(&region.mask);
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
                .all(|p| p[0].row_offset + p[0].rows < p[1].row_offset),
            "maximal runs would have been merged: {spans:?}",
        );

        let only = [Lane::new(0b01, 2), Lane::new(0b11, 3)];
        let fire = compose(&compiled, &budget(), &only).expect("composes");
        let spans = fire.classes().spans(&region.mask);
        assert_eq!(
            spans.iter().map(|s| (s.row_offset, s.rows)).collect::<Vec<_>>(),
            vec![(0, 5)],
            "with the classes between them absent, the window is one launch",
        );
    }

    #[test]
    fn a_bit_no_guard_reads_is_masked_off_rather_than_refused() {
        let (b, _) = diagram();
        let compiled = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");
        // One guarded bit, so the sweep covers words 0 and 1. A lane carrying
        // bit 1 as well is a model computing a fact this plan does not split
        // on — which is the same behaviour as the same lane without it, and
        // not two halves disagreeing about what is loaded.
        let plain = compose(&compiled, &budget(), &[Lane::new(0b01, 1)]).expect("composes");
        let extra = compose(&compiled, &budget(), &[Lane::new(0b11, 1)]).expect("composes");
        assert_eq!(rows_of(&compiled, &plain, DECODE), 1);
        assert_eq!(rows_of(&compiled, &extra, DECODE), 1);
        assert_eq!(rows_of(&compiled, &extra, PREFILL), 0);
    }

    #[test]
    fn a_lane_of_no_rows_is_refused_before_it_takes_a_seat() {
        let (b, _) = diagram();
        let compiled = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");
        assert_eq!(
            compose(&compiled, &budget(), &[Lane::new(1, 1), Lane::new(0, 0)]),
            Err(Error::Fire(Fault::EmptyLane { lane: 1 })),
        );
    }

    #[test]
    fn the_ceilings_are_the_budget_s_and_the_refusal_carries_them() {
        let (b, _) = diagram();
        let compiled = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");

        let crowd: Vec<Lane> = (0..9).map(|_| Lane::new(1, 1)).collect();
        assert_eq!(
            compose(&compiled, &budget(), &crowd),
            Err(Error::Fire(Fault::TooManyLanes { lanes: 9, max: 8 })),
        );

        let long = [Lane::new(0, 40), Lane::new(0, 40)];
        assert_eq!(
            compose(&compiled, &budget(), &long),
            Err(Error::Fire(Fault::TooManyRows { rows: 80, max: 64 })),
        );
    }

    #[test]
    fn a_fire_rounds_up_to_a_bucket_and_one_above_them_all_is_refused() {
        let (b, _) = diagram();
        let mut budget = budget();
        budget.buckets = vec![1, 4, 16];
        let compiled = compile(&b.trace, &budget, &DeviceProfile::default()).expect("bakes");

        let fire = compose(&compiled, &budget, &[Lane::new(0, 5)]).expect("composes");
        assert_eq!((fire.rows(), fire.bucket()), (5, 16));
        let fire = compose(&compiled, &budget, &[Lane::new(1, 1)]).expect("composes");
        assert_eq!((fire.rows(), fire.bucket()), (1, 1));

        // 17 rows round up to nothing, so there is no graph to launch them in.
        assert_eq!(
            compose(&compiled, &budget, &[Lane::new(0, 17)]),
            Err(Error::Fire(Fault::NoBucket { rows: 17, top: 16 })),
        );

        // A budget with no lattice is a deployment that has not chosen one,
        // and then the bucket is the fire's own size.
        // `budget` is shadowed above by this test's lattice, so the plain
        // one is named through the module.
        let open = super::tests::budget();
        let compiled = compile(&b.trace, &open, &DeviceProfile::default()).expect("bakes");
        let fire = compose(&compiled, &open, &[Lane::new(0, 5)]).expect("composes");
        assert_eq!(fire.bucket(), 5);
    }

    #[test]
    fn a_value_s_offset_is_static_and_only_its_length_moves() {
        let (b, y) = diagram();
        let compiled = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");

        let small = compose(&compiled, &budget(), &[Lane::new(1, 1)]).expect("composes");
        let big =
            compose(&compiled, &budget(), &[Lane::new(0, 7), Lane::new(1, 1)]).expect("composes");

        let one = small
            .value_window(&compiled.arena, y)
            .expect("y is in the arena");
        let eight = big
            .value_window(&compiled.arena, y)
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
            compiled
                .arena
                .window(y, big.fire_rows())
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
        let compiled = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");
        let fire = compose(&compiled, &budget(), &[]).expect("composes");
        assert_eq!(fire.rows(), 0);
        assert_eq!(fire.lane_count(), 0);
        assert!(fire.present().is_empty());
        assert_eq!(rows_of(&compiled, &fire, SHARED), 0);
    }

    /// **G4, ON THE COMPOSITION'S OWN SURFACE.** A text-only fire's every
    /// public answer is the answer a hand-built expectation states — the
    /// token half unmoved by the axis array, the patch half the zero
    /// rectangle rather than an absent one.
    ///
    /// The point is not that the numbers are these numbers; the diagram test
    /// above already says that. It is that the accessors that used to read
    /// ten named fields and now read a `PerAxis` of two records answer
    /// IDENTICALLY, field for field, including the ones a text-only fire is
    /// only entitled to because the second entry is a real zero.
    #[test]
    fn a_text_only_compositions_public_answers_are_the_hand_built_ones() {
        let (b, _) = diagram();
        let compiled = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");
        let lanes = [Lane::new(0, 7), Lane::new(0, 3), Lane::new(1, 1)];
        let fire = compose(&compiled, &budget(), &lanes).expect("composes");

        // The token half, stated by hand off the submission.
        assert_eq!(fire.rows(), 11, "7 + 3 + 1");
        assert_eq!(fire.lane_count(), 3);
        assert_eq!(fire.bucket(), 11, "no lattice, so the bucket is the count");
        assert_eq!(fire.classes().len(), compiled.classes.classes.len());
        assert_eq!(fire.present().len(), 2, "one prefill class, one decode");
        assert_eq!(
            fire.classes().rows_of(&ClassSet::of(fire.present().iter().map(|&c| c as usize))),
            11,
            "the present classes tile the fire",
        );
        assert_eq!(fire.lanes().len(), 3);
        assert_eq!(
            fire.lanes().iter().map(|lane| lane.rows).sum::<u32>(),
            11,
        );

        // The patch half: the ZERO rectangle, entry for entry.
        assert_eq!(fire.patch_rows(), 0);
        assert_eq!(fire.images(), 0);
        assert_eq!(fire.patch_bucket(), 0, "no ladder, no tower exec");
        assert!(fire.patch_present().is_empty());
        assert_eq!(
            fire.patch_classes().as_slice(),
            vec![ClassWindow::default(); compiled.classes.classes.len()],
            "every class's patch window is the zero window",
        );
        for lane in fire.lanes() {
            assert_eq!(
                (lane.patch_offset, lane.patches, lane.image_offset, lane.images),
                (0, 0, 0, 0),
                "a text lane is placed on no patch rectangle",
            );
        }

        // And the two doors onto the array agree with the two named halves,
        // which is what makes the forwards forwards.
        assert_eq!(fire.axis(RowAxis::Tokens).classes, *fire.classes());
        assert_eq!(fire.axis(RowAxis::Tokens).rows, fire.rows());
        assert_eq!(fire.axis(RowAxis::Tokens).lanes, fire.lane_count());
        assert_eq!(fire.axis(RowAxis::Tokens).bucket, fire.bucket());
        assert_eq!(fire.table(RowAxis::Patches), fire.patch_classes());
        // **AND THE PATCH ENTRY IS THE ZERO SERIATION, NOT AN ABSENT ONE** —
        // a table of `count` zero windows rather than a table of none, which
        // is what makes a text-only fire's second axis free rather than a
        // special case at every reader.
        let patches = fire.axis(RowAxis::Patches);
        assert_eq!((patches.rows, patches.lanes, patches.bucket), (0, 0, 0));
        assert!(patches.order.is_empty());
        assert_eq!(patches.classes.len(), compiled.classes.classes.len());

        // The counts the carve takes are the same four numbers.
        let rows = fire.fire_rows();
        assert_eq!((rows.tokens, rows.lanes, rows.patches, rows.images), (11, 3, 0, 0));
    }
}
