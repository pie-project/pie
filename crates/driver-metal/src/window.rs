//! The fire's windows: which rows and which lanes each region of the baked
//! template actually runs over, and the cursor that tells a [`Run`] which one
//! it is inside.
//!
//! **THIS IS DESIGN §0's DIAGRAM, RESOLVED.** A fire seriates its lanes by
//! class, so a node guarded on `qo_one` stands over one interval of rows and a
//! node guarded on `¬qo_one` over the interval beside it; the shared nodes
//! stand over both. Every table a [`Run`] resolves through is indexed by
//! ABSOLUTE fire row (or absolute fire lane) — the arena carve gives one
//! column per value at `Dim::Tokens` rows, and design §0's merge is exactly
//! "the arms write disjoint row ranges of it" — so a node's operands are that
//! node's window's SLICE of those columns, and nothing has to be re-carved.
//!
//! ```text
//! fire:            [ prefill lane 0 : 7 rows | prefill lane 1 : 3 | decode l2 ]
//! arena column x:  [·············· 11 rows, one rectangle ··················]
//! embed/norm/qkv    ─────────── window (0,11) lanes (0,3) ────────────
//! attention.prefill ──── window (0,10) lanes (0,2) ────┐
//! attention.decode                                     └── (10,1) (2,1) ──
//! ```
//!
//! # Why per REGION and not per value
//!
//! The obvious reading — give every value its own span, from the classes its
//! defining node runs in — is not enough, and the reason is in the IR. A
//! `split` does not mint a value: `Value::split` REFINES a value's cond and
//! hands back the same `ValueId` (`model_dsl::record`), so the `q` a decode op
//! reads and the `q` a prefill op reads are one id with one rectangle. The
//! window belongs to the READER, not to the value: the decode node takes rows
//! `[10,11)` of `q` and the prefill node takes `[0,10)` of the same `q`, and
//! only the node knows which. So the resolution is `value column ∩ this
//! node's window`, and the node's window is its region's — P2 coalesces
//! exactly the nodes whose class mask is equal, which is to say exactly the
//! nodes that share a window.
//!
//! Values still land where a per-value reading would put them, because the
//! two agree wherever the value has one reader: an arm of a merge is
//! written by its own guarded node and therefore over its own window, and the
//! merge column it is aliased onto is read by an unguarded consumer over the
//! union — which is the whole fire. That is design §0's zero-instruction φ,
//! and it falls out rather than being arranged.
//!
//! # A cut is a HANDLE here, not an address
//!
//! The one place this file's arithmetic meets the plane it runs on. A
//! `kernels_cuda::Tensor` carries a device address, so the sibling shell
//! seats a window's boundary vector by writing `base + offset` into a
//! `Tensor` and is done — the number IS the location. Metal binds a BUFFER
//! and an OFFSET, so [`Windows::bind`] mints a row in
//! [`Handles`](crate::device::Handles) per window and seats the row's index.
//! Same vector, same rebasing, same one staged copy; what changes is that
//! seating a window can now FAIL (a boundary vector that would leave the
//! reservation is refused where it is minted, not where a shader reads it),
//! which is why `bind` answers a `Result` and its twin answers nothing.
//!
//! # How a `Run` learns which region it is in
//!
//! `Dispatch::exec` takes a `&Node` and the walk's signature is fixed
//! (decision #11: one walk, generic over `Dispatch` × `Sink`). But the walk
//! announces every region to the SINK, in order, before dispatching its nodes
//! — and the sink is the shell's. So [`Cursor`] is this shell's `Sink`: it
//! counts regions into a `Cell` the `Run` also holds a shared reference to.
//! No signature moves, and the one piece of state involved is a `u32` written
//! once per region.
//!
//! [`Run`]: crate::run::Run

use std::cell::Cell;

use driver::fire::{EventId, MaskSpan, Sink, WindowTable};
use kernels_metal::Tensor;
use model_compiler::{Baked, Lowering, Region};
use model_ir::{Attention, Def, Dtype, Operation, Plan};

use crate::device::handles::NIL;
use crate::device::Handles;
use crate::error::{Fault, Result};

/// One window, and the qo boundaries that go with it.
///
/// The span is the arithmetic (rows and lanes, both, because the IR has both
/// symbols); the two indptrs are the one thing a window cannot slice, because
/// a ragged view's boundaries are OFFSETS INTO the rectangle they cut and a
/// sub-rectangle starts at zero. So each window carries its own rebased copy —
/// `[lanes + 1]` entries, the first of them 0 — device-side for the launches
/// and host-side for the plan builders that walk the contents (the duality
/// [`CacheGeometry`](crate::run::CacheGeometry)'s seats and their host twins
/// state per cache space).
#[derive(Debug, Clone)]
pub struct Window {
    /// The rows and lanes this window covers, in fire coordinates.
    pub span: MaskSpan,
    /// `[lanes + 1]`: the window's qo boundaries, rebased to start at 0.
    pub indptr_host: Vec<i32>,
    /// The same vector, staged. Carries [`NIL`] — the absent handle, never
    /// the first row — until [`Windows::bind`] has minted its view; on the
    /// CUDA plane the same seat holds a null ADDRESS, and 0 is a perfectly
    /// good handle here.
    pub indptr: Tensor,
}

/// Every region's window, deduplicated.
///
/// Deduplicated because a plan has hundreds of regions and at most a handful
/// of distinct windows — one per contiguous run of the class order — and the
/// rebased boundary vectors are staged one per DISTINCT window, in a single
/// copy, rather than one per region.
#[derive(Debug, Clone, Default)]
pub struct Windows {
    windows: Vec<Window>,
    /// Region index → position in [`windows`](Windows::windows).
    of_region: Vec<u32>,
}

impl Windows {
    /// The windows of one fire: every region of the template resolved against
    /// this composition's class table.
    ///
    /// # Errors
    ///
    /// [`Fault::Fragmented`] for a region whose classes are not consecutive in
    /// the fire's class order — a promise P4 made and this fire found broken,
    /// which is a bake-integrity failure rather than a slow path (the catalog
    /// bakes an empty `FallbackTable` today).
    pub fn of(baked: &Baked, classes: &WindowTable, indptr_host: &[i32]) -> Result<Windows> {
        let mut windows: Vec<Window> = Vec::new();
        let mut of_region: Vec<u32> = Vec::with_capacity(baked.template().len());

        for (at, region) in baked.template().iter().enumerate() {
            // An empty mask (a region no class demands) answers the zero
            // window, which is the same answer a composition without this
            // behavior gives — and the walk skips both for the same reason.
            let span = classes
                .span(&region.mask)
                .map_err(|runs| Fault::Fragmented {
                    region: at as u32,
                    runs,
                })?
                .unwrap_or_default();
            let found = windows.iter().position(|held| held.span == span);
            let index = match found {
                Some(index) => index,
                None => {
                    windows.push(Window {
                        span,
                        indptr_host: rebase(indptr_host, span),
                        indptr: Tensor::new(NIL, 0, 1, Dtype::I32),
                    });
                    windows.len() - 1
                }
            };
            of_region.push(index as u32);
        }

        Ok(Windows {
            windows,
            of_region,
        })
    }

    /// How many distinct windows this fire has.
    #[must_use]
    pub fn len(&self) -> usize {
        self.windows.len()
    }

    /// Does it hold none? Only for a template with no regions at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    /// Every window's rebased boundaries, end to end — what the shell writes
    /// in one copy.
    #[must_use]
    pub fn packed(&self) -> Vec<i32> {
        self.windows
            .iter()
            .flat_map(|window| window.indptr_host.iter().copied())
            .collect()
    }

    /// Seat the staged boundaries: `base` is where
    /// [`packed`](Windows::packed) landed inside `buffer`.
    ///
    /// One handle per distinct window, minted in the order `packed` wrote
    /// them, which is the order this table holds them in — so the `n`th
    /// window's view starts `n` boundary vectors into the copy. The CUDA
    /// sibling adds those same byte counts to a device address and cannot
    /// fail; here each view is a row in the handle table and is bounds-checked
    /// against the reservation as it is minted.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] when a window's boundaries would leave `buffer` —
    /// a staging reservation too small for what `packed` produced — or when
    /// the handle table is full.
    pub fn bind(&mut self, handles: &Handles, packed: u32) -> Result<()> {
        let mut at = 0u64;
        for window in &mut self.windows {
            let rows = window.indptr_host.len() as u32;
            let bytes = u64::from(rows) * 4;
            window.indptr =
                Tensor::new(handles.cut(packed, at, bytes)?, rows, 1, Dtype::I32);
            at += bytes;
        }
        Ok(())
    }

    /// One region's window.
    ///
    /// A region index this table does not hold is an integrity failure of the
    /// shell — the cursor counts the same template the table was built from —
    /// so it panics with a sentence rather than dressing up as a window.
    #[must_use]
    pub fn at(&self, region: u32) -> &Window {
        self.of_region
            .get(region as usize)
            .and_then(|index| self.windows.get(*index as usize))
            .unwrap_or_else(|| {
                panic!(
                    "region {region} has no window, and this fire resolved {} of them \
                     over a template of {}",
                    self.windows.len(),
                    self.of_region.len()
                )
            })
    }
}

/// **THE BAKE-TIME HALF OF THE WINDOW ARGUMENT**: no attention schedule may
/// be built over more classes than the node consuming it runs in.
///
/// A schedule is not a row-shaped table that slices — it is a carving. How
/// many requests it batches, where each request's query rows start, how its
/// work items split the kv, and how much of the grant it padded to are all
/// fixed when [`plan_prefill`](kernels_metal::attn::plan_prefill) walks
/// the window it was dispatched in. The consumers then hand it their OWN
/// rebased qo boundaries ([`Window::indptr`]), and a consumer standing in a
/// narrower window hands it a vector that ends before its work items do.
/// Nothing faults: the reads land in whatever follows a `[lanes + 1]` vector
/// in the input reservation, and the answer is wrong logits. The handle
/// table narrows the blast radius — a cut is bounds-checked against its
/// buffer — and does not close it, because the vector that follows is inside
/// the same reservation and is therefore a legal read.
///
/// It is a property of the BAKE, not of a fire — region masks are static —
/// so it is asked once at load, where the sentence can name the model text
/// that has to change. What produces it is one plan value shared by arms in
/// different classes: the compiler narrows a prepare node by demand to the
/// union of the classes reading its struct (design build log 7), which is
/// the right answer for a shared value and the wrong SHAPE for two windowed
/// readers. gemma's text is the standing instance — `plan_p` feeds both
/// `attention.prefill` and `attention.masked`, so its region carries the two
/// classes and each arm carries one.
///
/// Equality rather than containment, deliberately. A schedule built over
/// FEWER classes than its reader is the same failure from the other side
/// (the reader's later requests index past the schedule's batch), and a
/// consumer that is not windowed at all does not consume a plan.
///
/// # Errors
///
/// [`Fault::Straddled`], naming the value, the consuming node, and the two
/// class sets.
pub fn no_schedule_straddles_its_readers(plan: &Plan, baked: &Baked) -> Result<()> {
    // Which region each node stands in, and therefore which classes it runs.
    let mut region_of: Vec<usize> = vec![0; plan.nodes.len()];
    for (at, region) in baked.template().iter().enumerate() {
        for node in region.nodes.clone() {
            if let Some(slot) = region_of.get_mut(node as usize) {
                *slot = at;
            }
        }
    }
    let mask_of = |node: usize| &baked.template()[region_of[node]].mask;

    for (at, node) in plan.nodes.iter().enumerate() {
        let Operation::Attention(op) = &node.op else {
            continue;
        };
        // Only the launches, never the builders: a builder DEFINES the
        // schedule and so stands in the window it is carved at by
        // construction.
        let consumed = match op {
            Attention::Decode { plan, .. }
            | Attention::DecodeLse { plan, .. }
            | Attention::Prefill { plan, .. }
            | Attention::PrefillLse { plan, .. }
            | Attention::Masked { plan, .. } => *plan,
            _ => continue,
        };
        let Some(Def::Op(built_by)) = plan.values.get(consumed.0 as usize).map(|v| &v.def) else {
            continue;
        };
        let planned = mask_of(*built_by as usize);
        let reader = mask_of(at);
        if planned != reader {
            return Err(Fault::Straddled {
                value: consumed.0,
                node: at as u32,
                planned: format!("{:?}", planned.iter().collect::<Vec<_>>()),
                consumed: format!("{:?}", reader.iter().collect::<Vec<_>>()),
            });
        }
    }
    Ok(())
}

/// The window's qo boundaries, rebased so the first is 0.
fn rebase(indptr: &[i32], span: MaskSpan) -> Vec<i32> {
    let first = span.lane_offset as usize;
    let last = first + span.lanes as usize;
    let Some(cut) = indptr.get(first..=last) else {
        return vec![0];
    };
    let base = cut[0];
    cut.iter().map(|bound| bound - base).collect()
}

/// This shell's [`Sink`]: the region counter a [`Run`](crate::run::Run) reads
/// its window out of.
///
/// **THE EAGER CURSOR RECORDS NOTHING, LIKE `EagerSink`, AND CARRIES ONE
/// NUMBER.** The walk calls [`region_begin`](Sink::region_begin) for every
/// region of the template in order — including the ones this composition has
/// no rows for, which is what makes the count an index rather than a guess —
/// and a `Run` holding a shared reference to the same `Cell` then resolves
/// each operand at that region's window.
///
/// **THERE IS NO RECORDING CURSOR HERE, AND NO STREAM SWITCH — DESIGN §6's
/// "no record.rs".** The CUDA sibling carries a second constructor
/// (`Cursor::across`) that writes a per-region stream into a cell, records an
/// event at each fork and waits one at each join, because a captured graph
/// has to CARRY the DAG's parallelism: the capture is replayed later, so the
/// structure must be in it. This shell is eager from end to end — one command
/// buffer, encoded in walk order — so the DAG's serialization IS the schedule
/// (`driver::fire::EagerSink`'s doc argues why that is correct rather than
/// merely safe: the walk emits a topological order, and a topological order
/// of a dependency DAG is a legal execution of it). [`Sink::fork`] and
/// [`Sink::join`] are therefore no-ops that name their event and drop it, and
/// the whole `Lanes` bundle — side streams, event handles, the stream cell —
/// has no counterpart on this plane and is not ported.
///
/// What is lost is the OVERLAP, not the correctness: two independent regions
/// that a CUDA capture would run on two streams are encoded one after the
/// other here. That is a performance ceiling this shell has not needed to
/// lift, and the place to lift it is a second command buffer per fork arm
/// with an `MTLEvent` between them — a change to this type, not to the walk.
#[derive(Debug)]
pub struct Cursor<'a> {
    at: u32,
    region: &'a Cell<u32>,
}

impl<'a> Cursor<'a> {
    /// A cursor writing into `region`, counting from the template's first.
    #[must_use]
    pub fn new(region: &'a Cell<u32>) -> Cursor<'a> {
        region.set(0);
        Cursor { at: 0, region }
    }

    /// What the device refused during the walk, if anything.
    ///
    /// **NOTHING CAN, AND THE METHOD IS KEPT SAYING SO.** The CUDA twin
    /// exists because a `cudaEventRecord` inside a `Sink` method has nowhere
    /// to return an error to, so the first failure is held here and drained
    /// afterwards by the code that knows a half-recorded capture must not be
    /// instantiated. This cursor makes no device call at all — it writes a
    /// `u32` into a cell — so there is nothing to hold and nothing to drain.
    /// The signature stays because the CALLER's shape is the seam: the fire
    /// path settles its cursor before it commits, and a future cursor that
    /// does encode across events (see the type's doc) would fill this in
    /// without moving a line above it.
    ///
    /// # Errors
    ///
    /// None today, by construction. The `Result` is the seam, not a claim.
    #[allow(clippy::unnecessary_wraps, reason = "the seam: see the item doc")]
    pub fn settle(self) -> Result<()> {
        Ok(())
    }
}

impl Sink for Cursor<'_> {
    fn region_begin(&mut self, _region: &Region) {
        self.region.set(self.at);
        self.at += 1;
    }
    fn region_end(&mut self, _region: &Region) {}
    fn cond_begin(&mut self, _lowering: &Lowering) {}
    fn cond_arm(&mut self, _arm: u8) {}
    fn cond_end(&mut self) {}
    /// Nothing to record: an eager encode has already ordered this region
    /// against everything before it (see the type doc).
    fn fork(&mut self, _event: EventId) {}
    /// Nothing to wait on, for the same reason `fork` records nothing.
    fn join(&mut self, _event: EventId) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use driver::fire::ClassWindow;
    use model_compiler::Phase;
    use model_ir::ClassSet;

    /// The design's own diagram: 10 prefill rows over 2 lanes, then 3 decode
    /// rows over 3 lanes.
    fn table() -> WindowTable {
        WindowTable::new(vec![
            ClassWindow {
                row_offset: 0,
                rows: 10,
                lane_offset: 0,
                lanes: 2,
            },
            ClassWindow {
                row_offset: 10,
                rows: 3,
                lane_offset: 2,
                lanes: 3,
            },
        ])
    }

    #[test]
    fn a_mask_over_both_classes_is_the_whole_fire() {
        let span = table()
            .span(&ClassSet::of([0, 1]))
            .expect("consecutive")
            .expect("non-empty");
        assert_eq!(span.row_offset, 0);
        assert_eq!(span.rows, 13);
        assert_eq!(span.lane_offset, 0);
        assert_eq!(span.lanes, 5);
    }

    #[test]
    fn one_class_is_its_own_interval() {
        let span = table()
            .span(&ClassSet::of([1]))
            .expect("consecutive")
            .expect("non-empty");
        assert_eq!((span.row_offset, span.rows), (10, 3));
        assert_eq!((span.lane_offset, span.lanes), (2, 3));
    }

    #[test]
    fn the_boundaries_are_rebased_to_the_window_s_own_zero() {
        // qo boundaries of the whole fire: two prefills then three decodes.
        let indptr = [0, 7, 10, 11, 12, 13];
        let decode = table()
            .span(&ClassSet::of([1]))
            .expect("consecutive")
            .expect("non-empty");
        assert_eq!(rebase(&indptr, decode), vec![0, 1, 2, 3]);
        let prefill = table()
            .span(&ClassSet::of([0]))
            .expect("consecutive")
            .expect("non-empty");
        assert_eq!(rebase(&indptr, prefill), vec![0, 7, 10]);
    }

    /// The cursor is a counter and nothing else: every region the walk
    /// announces lands in the cell the `Run` reads, in template order.
    #[test]
    fn the_cursor_counts_regions_into_the_cell() {
        let cell = Cell::new(7);
        let mut cursor = Cursor::new(&cell);
        assert_eq!(cell.get(), 0, "a fresh cursor rebases the cell");
        let region = Region {
            nodes: 0..0,
            mask: ClassSet::of([0]),
            phase: Phase::Capture,
            lowering: Lowering::AlwaysLaunch,
            stream: 0,
            wait: Vec::new(),
            open: None,
            close: None,
            sm_hint: None,
            collective: false,
        };
        cursor.region_begin(&region);
        assert_eq!(cell.get(), 0, "the first region is index 0");
        cursor.region_end(&region);
        cursor.region_begin(&region);
        assert_eq!(cell.get(), 1);
        // The fork points are announced and dropped — an eager encode has
        // already ordered them.
        cursor.fork(EventId(0));
        cursor.join(EventId(0));
        assert_eq!(cell.get(), 1, "an event point is not a region");
        cursor.settle().expect("an eager cursor never holds a fault");
    }
}
