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

use model_compiler::{ArenaMap, Budget, Budgets, CompiledModel, Extent, PatchLadder};
use model_ir::{ClassSet, RowAxis, ValueId};

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

    /// The maximal contiguous row runs this mask covers, ascending.
    ///
    /// ONE SPAN IS THE ANSWER P4 IS FOR. A mask whose classes are an interval
    /// of the fire's class order comes back as a single [`RowSpan`] and its
    /// consumer is one launch over pointer+extent; a mask that straddles a
    /// class it does not contain comes back as two, and the consumer is two
    /// launches — `Fallback::Split { r }`, which is the row P4 wrote for it.
    ///
    /// The row half of [`spans`](WindowTable::spans), and derived from it
    /// rather than computed again: a second implementation of "where does
    /// this window break" is a second answer waiting to disagree with the one
    /// the launches are cut at.
    #[must_use]
    pub fn segments(&self, mask: &ClassSet) -> Vec<RowSpan> {
        self.spans(mask)
            .into_iter()
            .map(|span| RowSpan {
                offset: span.row_offset,
                rows: span.rows,
            })
            .collect()
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
    classes: WindowTable,
    order: Vec<u32>,
    rows: u32,
    bucket: u32,
    patch_classes: WindowTable,
    patch_order: Vec<u32>,
    patch_rows: u32,
    images: u32,
    patch_bucket: u32,
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

    /// How many PATCH rows this fire carries — the second row axis's count,
    /// out of the second seriation.
    ///
    /// Zero for a text-only fire and for every fire of a text-only artifact,
    /// and TRUE rather than defaulted in both cases: a fire whose lanes
    /// submitted no image has no patch rows.
    #[must_use]
    pub fn patch_rows(&self) -> u32 {
        self.patch_rows
    }

    /// How many IMAGES this fire carries — the patch axis's lane count.
    #[must_use]
    pub fn images(&self) -> u32 {
        self.images
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
        &self.patch_classes
    }

    /// The classes this fire has IMAGES in, in the order their patch rows
    /// stand.
    #[must_use]
    pub fn patch_present(&self) -> &[u32] {
        &self.patch_order
    }

    /// The PATCH rung these patch rows round up to — which tower exec this
    /// fire launches. Equal to [`patch_rows`](Composition::patch_rows) when
    /// the ladder lists no rungs, exactly as the token bucket is.
    #[must_use]
    pub fn patch_bucket(&self) -> u32 {
        self.patch_bucket
    }

    /// This fire's four counts, as the carve's arithmetic takes them.
    #[must_use]
    pub fn fire_rows(&self) -> model_compiler::FireRows {
        model_compiler::FireRows {
            tokens: u64::from(self.rows),
            lanes: u64::from(self.lane_count()),
            patches: u64::from(self.patch_rows),
            images: u64::from(self.images),
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
    let mut tally: Vec<(u64, u32)> = vec![(0, 0); count];
    let mut rows: u64 = 0;
    // The second axis's tallies, in the first's shape: patch rows and images
    // per class. Both stay all-zero for a text-only fire, and the passes
    // below read that as "no class is present on this axis" without a branch.
    let mut patch_tally: Vec<(u64, u32)> = vec![(0, 0); count];
    let mut patches: u64 = 0;
    let mut images: u64 = 0;
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
        tally[class as usize].0 += u64::from(lane.rows);
        tally[class as usize].1 += 1;
        rows += u64::from(lane.rows);
        // The same two sums on the other axis. A class's patch rows are the
        // patch rows of the lanes that resolved to it, and its "lanes" are
        // their images — which is where the two axes stop agreeing, because a
        // class with lanes may have no images at all.
        patch_tally[class as usize].0 += u64::from(lane.patches);
        patch_tally[class as usize].1 += lane.images;
        patches += u64::from(lane.patches);
        images += u64::from(lane.images);
        of_lane.push(class);
    }

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
    let images = images as u32;

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
    let order: Vec<u32> = compiled
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

    // **THE SECOND SERIATION**, and it is the first one again over a
    // different order. The classes are the same classes; which of them are
    // PRESENT is a different question (a class with lanes may carry no
    // image), and the order they stand in is the artifact's own patch
    // `ClassOrder` rather than the token one — P4 solved the C1P instance
    // twice, once per axis, because the consumers it has to seat are
    // different consumers.
    let mut patch_classes = vec![ClassWindow::default(); count];
    let mut patch_order: Vec<u32> = Vec::new();
    // Where each SUBMITTED lane's images land, computed in the patch order
    // and read back in the token one. A side table rather than a second lane
    // list: the permutation is one permutation, and two lists indexed by two
    // orders is how a `source` comes to be read against the wrong record.
    let mut placed_patch: Vec<(u32, u32)> = vec![(0, 0); lanes.len()];
    if let Some(patch_plan) = compiled.order_for(RowAxis::Patches) {
        let present = ClassSet::of((0..count).filter(|&class| patch_tally[class].1 > 0));
        patch_order = patch_plan
            .class_order(&present, None)
            .into_iter()
            .map(u32::from)
            .collect();
        let (mut patch_at, mut image_at) = (0u32, 0u32);
        for &class in &patch_order {
            let (class_patches, class_images) = patch_tally[class as usize];
            patch_classes[class as usize] = ClassWindow {
                row_offset: patch_at,
                rows: class_patches as u32,
                lane_offset: image_at,
                lanes: class_images,
            };
            // Inside a class, submission order — the same inner stability the
            // token pass keeps, and for the same reason: the runtime wrote
            // this lane's images in the order it submitted them.
            let mut lane_patch_at = patch_at;
            let mut lane_image_at = image_at;
            for (i, lane) in lanes.iter().enumerate() {
                if of_lane[i] != class || lane.images == 0 {
                    continue;
                }
                placed_patch[i] = (lane_patch_at, lane_image_at);
                lane_patch_at += lane.patches;
                lane_image_at += lane.images;
            }
            patch_at += class_patches as u32;
            image_at += class_images;
        }
    }

    // Pass three: place the lanes. Class by class in fire order, and INSIDE a
    // class in submission order — that inner stability is not decoration. The
    // runtime's order is the order its per-lane buffers are written in, and a
    // fire that reshuffles lanes it had no reason to reshuffle churns every
    // pointer the previous fire had warm.
    let mut seriated: Vec<LaneRow> = Vec::with_capacity(lanes.len());
    for &class in &order {
        let mut row_at = classes[class as usize].row_offset;
        for (i, lane) in lanes.iter().enumerate() {
            if of_lane[i] != class {
                continue;
            }
            let (patch_offset, image_offset) = placed_patch[i];
            seriated.push(LaneRow {
                source: i as u32,
                word: lane.word,
                class,
                row_offset: row_at,
                rows: lane.rows,
                patch_offset,
                patches: lane.patches,
                image_offset,
                images: lane.images,
            });
            row_at += lane.rows;
        }
    }

    Ok(Composition {
        lanes: seriated,
        classes: WindowTable::new(classes),
        order,
        rows,
        bucket: bucket_of(budget, rows)?,
        patch_classes: WindowTable::new(patch_classes),
        patch_order,
        patch_rows: patches,
        images,
        patch_bucket: patch_bucket_of(ladder, patches)?,
    })
}

/// The smallest bucket that holds these rows.
///
/// AN EMPTY LATTICE IS NOT AN ERROR: a budget that lists no buckets is a
/// deployment that has not chosen a shape lattice, and the honest bucket for a
/// fire of `rows` rows is then `rows` itself — one graph per size, which is
/// what a golden-path eager walk wants and what a test builds with
/// `Budget::new`.
/// The smallest PATCH rung that holds these patch rows.
///
/// The token ladder's rule, one axis over and with its own vector: an empty
/// ladder is a deployment that chose no patch lattice, and a fire with no
/// patch rows rounds to zero rung and launches no tower exec at all — which
/// is what "an axis-empty fire simply does not launch that unit's exec"
/// (multimodal §1) means arithmetically.
fn patch_bucket_of(ladder: Option<&PatchLadder>, patches: u32) -> Result<u32> {
    let Some(ladder) = ladder else {
        return Ok(0);
    };
    if patches == 0 || ladder.buckets.is_empty() {
        return Ok(patches);
    }
    ladder
        .buckets
        .iter()
        .copied()
        .find(|rung| *rung >= patches)
        .ok_or_else(|| {
            Error::Fire(Fault::NoPatchBucket {
                patches,
                top: ladder.buckets.last().copied().unwrap_or(0),
            })
        })
}

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
/// about the fire's total in particular — so a per-class ladder built on a
/// second, coarser ladder would be a second answer to the same question. The
/// caller that reads this is `engine_cuda::record::Ladder`, which puts one
/// rung per present class into the body key.
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

    /// The same, said from inside a test that shadowed the name with a
    /// lattice of its own.
    fn budgets_without_a_lattice() -> Budget {
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
        let fire = compose(&compiled, &budget(), &only).expect("composes");
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
        let open = super::tests::budgets_without_a_lattice();
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
}
