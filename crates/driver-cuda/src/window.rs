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
//! # When a region's window is NOT one interval
//!
//! P4 makes as many windowed consumers consecutive as one row order can, and
//! writes a `Fallback` row for each one it could not (design §3). A region
//! with such a row covers SEVERAL row intervals in a fire that carries the
//! classes between its own, and the answer this shell serves is
//! `Fallback::Split { r }`: the region holds `r` windows rather than one, the
//! walk dispatches its nodes once per window, and each launch takes its own
//! pointer, its own extent and — the part that is easy to get silently wrong
//! — its own rebased qo boundaries. A ragged view's `indptr` is offsets INTO
//! the rectangle it cuts, so the second run's must start at 0 again over the
//! second run's lanes; sharing the first run's would hand the launch a vector
//! that describes somebody else's requests.
//!
//! ```text
//! classes in fire order:  [ 4 : 3 rows | 0 : 5 rows | 5 : 2 rows ]
//! mask {4,5,6,7}:          ──run 0──                 ──run 1──
//! qo indptr, fire-wide:   [0, 1, 3, 4, 6, 9, 10, 12]
//!            run 0:       [0, 1, 3]        run 1: [0, 2]
//! ```
//!
//! [`Fault::Fragmented`] survives, narrowed to what it always meant: a
//! fragmented window the artifact owes NO fallback row for, which is P4
//! having promised this mask consecutive and the fire finding it broken.
//!
//! # How a `Run` learns which region it is in
//!
//! `Dispatch::exec` takes a `&Node` and the walk's signature is fixed
//! (decision #11: one walk, generic over `Dispatch` × `Sink`). But the walk
//! announces every region to the SINK, in order, before dispatching its nodes
//! — and the sink is the shell's. So [`Cursor`] is this shell's `Sink`: it
//! counts regions into an [`At`] the `Run` also holds a shared reference to,
//! and writes the run index beside it for the same reason and at the same
//! instant. No signature moves, and the state involved is two `u32`s.
//!
//! [`Run`]: crate::run::Run

use std::cell::Cell;

use driver::fire::{EventId, MaskSpan, Sink, WindowTable, fallback};
use crate::device::graph::Event;
use kernels_cuda::Tensor;
use model_compiler::{Baked, Lowering, Region};
use model_ir::{Attention, Def, Dtype, Operation, Plan};

use crate::error::{Fault, Result};

/// One window, and the qo boundaries that go with it.
///
/// The span is the arithmetic (rows and lanes, both, because the IR has both
/// symbols); the two indptrs are the one thing a window cannot slice, because
/// a ragged view's boundaries are OFFSETS INTO the rectangle they cut and a
/// sub-rectangle starts at zero. So each window carries its own rebased copy —
/// `[lanes + 1]` entries, the first of them 0 — device-side for the launches
/// and host-side for the plan builders that walk the contents (the duality
/// [`CachePlanning`](crate::run::CachePlanning) states per cache space).
#[derive(Debug, Clone)]
pub struct Window {
    /// The rows and lanes this window covers, in fire coordinates.
    pub span: MaskSpan,
    /// `[lanes + 1]`: the window's qo boundaries, rebased to start at 0.
    pub indptr_host: Vec<i32>,
    /// The same vector, staged. `Tensor::new(0, 0, 0, ..)` until
    /// [`Windows::bind`] has been given the staging base.
    pub indptr: Tensor,
}

/// Every region's windows, deduplicated.
///
/// Deduplicated because a plan has hundreds of regions and at most a handful
/// of distinct windows — one per contiguous run of the class order — and the
/// rebased boundary vectors are staged one per DISTINCT window, in a single
/// copy, rather than one per region.
///
/// **A REGION HAS A LIST AND NOT A WINDOW**, because P4's fallback is a list:
/// a consumer it could not seat runs once per maximal interval of its class
/// set, and the interval is what a launch is cut at. One entry is the case P4
/// exists to produce and is what every region of every SKU the catalog seats
/// has; the empty window is one entry too, so that a region with no rows is
/// resolvable rather than special.
#[derive(Debug, Clone, Default)]
pub struct Windows {
    windows: Vec<Window>,
    /// Every region's runs end to end, as positions in
    /// [`windows`](Windows::windows) — region `r`'s are
    /// `runs[of_region[r].0 .. of_region[r].0 + of_region[r].1]`.
    runs: Vec<u32>,
    /// Region index → `(where its runs start, how many)`.
    of_region: Vec<(u32, u32)>,
}

impl Windows {
    /// The windows of one fire: every region of the template resolved against
    /// this composition's class table, one per interval its mask covers.
    ///
    /// # Errors
    ///
    /// [`Fault::Fragmented`] for a region whose classes are not consecutive in
    /// the fire's class order AND which the artifact owes no `Fallback` row —
    /// a promise P4 made and this fire found broken, which is a bake-integrity
    /// failure rather than a slow path. A region P4 DID write a row for is the
    /// slow path, and is served here as `Fallback::Split { r }` at every
    /// bucket; `driver::fire::fallback` states what that costs against the
    /// `Fallback::Copy` the table asks for below the crossover, and why this
    /// shell cannot yet serve it.
    pub fn of(baked: &Baked, classes: &WindowTable, indptr_host: &[i32]) -> Result<Windows> {
        let mut windows: Vec<Window> = Vec::new();
        let mut runs: Vec<u32> = Vec::with_capacity(baked.template().len());
        let mut of_region: Vec<(u32, u32)> = Vec::with_capacity(baked.template().len());
        let mut spans: Vec<MaskSpan> = Vec::new();

        for (at, region) in baked.template().iter().enumerate() {
            classes.spans_into(&region.mask, &mut spans);
            if spans.len() > 1 {
                // The two integrity questions, asked of the artifact
                // rather than of the fire. Did P4 PROMISE this window
                // consecutive — a capture region it seated and wrote no
                // fallback row for? And is this fire's run count within the
                // one the shipped order breaks the mask into? A fire's order
                // is that order with the absent classes dropped, and dropping
                // a class can only close a gap, so neither can happen to a
                // `Baked` and a `WindowTable` built from each other.
                let bound = fallback::bound(baked, &region.mask);
                if fallback::promised(baked, region) || spans.len() > bound as usize {
                    return Err(Fault::Fragmented {
                        region: at as u32,
                        runs: spans.len(),
                        promised: fallback::promised(baked, region).then_some(bound),
                    });
                }
            }
            // An empty mask (a region no class demands) answers the zero
            // window, which is the same answer a composition without this
            // behavior gives — and the walk skips both for the same reason.
            if spans.is_empty() {
                spans.push(MaskSpan::default());
            }

            of_region.push((runs.len() as u32, spans.len() as u32));
            for &span in &spans {
                let found = windows.iter().position(|held| held.span == span);
                let index = match found {
                    Some(index) => index,
                    None => {
                        windows.push(Window {
                            span,
                            indptr_host: rebase(indptr_host, span),
                            indptr: Tensor::new(0, 0, 1, Dtype::I32),
                        });
                        windows.len() - 1
                    }
                };
                runs.push(index as u32);
            }
        }

        Ok(Windows {
            windows,
            runs,
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

    /// Every window's rebased boundaries, end to end — what the shell stages
    /// in one copy.
    #[must_use]
    pub fn packed(&self) -> Vec<i32> {
        self.windows
            .iter()
            .flat_map(|window| window.indptr_host.iter().copied())
            .collect()
    }

    /// Seat the staged boundaries: `base` is where [`packed`](Windows::packed)
    /// landed on the device.
    pub fn bind(&mut self, base: u64) {
        let mut at = base;
        for window in &mut self.windows {
            let rows = window.indptr_host.len() as u32;
            window.indptr = Tensor::new(at, rows, 1, Dtype::I32);
            at += u64::from(rows) * 4;
        }
    }

    /// How many launches a region costs in this fire — `1` for a window P4
    /// seated, `r` for one it could not, and `1` for an empty window.
    ///
    /// THE SAME NUMBER `driver::fire::walk` LOOPS ON, and it is the same
    /// number because both read it off the same class table: the walk asks
    /// `WindowTable::spans_into` and this asked it once per region when the
    /// table was built. A disagreement would show up as
    /// [`at`](Windows::at)'s panic rather than as a wrong window.
    #[must_use]
    pub fn runs(&self, region: u32) -> u32 {
        self.of_region.get(region as usize).map_or(0, |held| held.1)
    }

    /// The most launches any region of this fire costs — what a per-run table
    /// is sized at.
    #[must_use]
    pub fn max_runs(&self) -> u32 {
        self.of_region
            .iter()
            .map(|&(_, runs)| runs)
            .max()
            .unwrap_or(1)
            .max(1)
    }

    /// One region's window, for one run of it.
    ///
    /// A region index this table does not hold, or a run past the ones it cut
    /// for that region, is an integrity failure of the shell — the cursor
    /// counts the same template the table was built from, and the walk loops
    /// over the same span list — so it panics with a sentence rather than
    /// dressing up as a window.
    #[must_use]
    pub fn at(&self, region: u32, run: u32) -> &Window {
        self.of_region
            .get(region as usize)
            .filter(|&&(_, runs)| run < runs)
            .and_then(|&(start, _)| self.runs.get((start + run) as usize))
            .and_then(|index| self.windows.get(*index as usize))
            .unwrap_or_else(|| {
                panic!(
                    "region {region} has no run {run}; this fire cut it into {} \
                     over a template of {}",
                    self.runs(region),
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
/// fixed when [`plan_prefill`](kernels_cuda::attn::plan::plan_prefill) walks
/// the window it was dispatched in. The consumers then hand it their OWN
/// rebased qo boundaries ([`Window::indptr`]), and a consumer standing in a
/// narrower window hands it a vector that ends before its work items do.
/// Nothing faults: the reads land in whatever follows a `[lanes + 1]` vector
/// in the staging store, and the answer is wrong logits.
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

/// Where the walk is: which region of the template, and which run of that
/// region's window.
///
/// **TWO NUMBERS, ONE OBJECT, BECAUSE THEY ARE READ TOGETHER.** A `Run`
/// resolves every operand at `windows.at(region, run)`, and a pair that could
/// be handed in separately is a pair that could be handed in from two
/// different walks. The [`Cursor`] writes both — the region before the
/// region's first node, the run before each launch of it — and the `Run`
/// holds a shared reference to the same object; that is the whole mechanism,
/// and it is a `Cell` rather than a `&mut` because `walk` takes the sink and
/// the dispatch as two separate borrows.
#[derive(Debug, Default)]
pub struct At {
    /// The region index, in `Baked::template` order.
    pub region: Cell<u32>,
    /// Which run of that region's window: `0` always, and `0..r` for a region
    /// P4 could not seat.
    pub run: Cell<u32>,
}

impl At {
    /// A cursor position at the top of the template.
    #[must_use]
    pub fn new() -> At {
        At::default()
    }
}

/// The stream handles and events a [`Cursor`] switches between — P6's half of
/// the sink.
///
/// **HANDED IN, NEVER OWNED.** The streams and the events are the context's,
/// opened once at load (`Context::open_lanes`); what this bundle adds is the
/// one cell the [`Run`](crate::run::Run) reads to know which of them the
/// launch it is about to make belongs on. Same mechanism as the region cell
/// beside it, for the same reason: the walk takes two `&mut` and the sink and
/// the dispatch cannot be one object.
#[derive(Debug, Clone, Copy)]
pub struct Lanes<'a> {
    /// The side streams, in stream order: `side[0]` is stream 1. The main
    /// stream is not here — a region on stream 0 is the ordinary case and
    /// needs no lookup.
    pub side: &'a [*mut core::ffi::c_void],
    /// The main stream, which is what an event on stream 0 is recorded on.
    pub main: *mut core::ffi::c_void,
    /// One event per `EventId`, in id order.
    pub events: &'a [Event],
    /// Which stream the walk is on now.
    pub at: &'a Cell<u32>,
}

/// This shell's [`Sink`]: the region counter a [`Run`](crate::run::Run) reads
/// its window out of, and — when the artifact forked — the stream switch and
/// the event points.
///
/// **THE EAGER CURSOR RECORDS NOTHING, LIKE `EagerSink`, AND CARRIES ONE
/// NUMBER.** The walk calls [`region_begin`](Sink::region_begin) for every
/// region of the template in order — including the ones this composition has
/// no rows for, which is what makes the count an index rather than a guess —
/// and a `Run` holding a shared reference to the same `Cell` then resolves
/// each operand at that region's window.
///
/// **[`Cursor::across`] IS THE RECORDING ONE, AND IT IS THE ONLY PLACE A
/// STREAM SWITCH HAPPENS.** A cursor built with [`Cursor::new`] leaves the
/// stream cell at zero forever, which is what makes the eager pass the
/// SERIALIZATION of P6's DAG (`driver::fire::EagerSink`'s doc argues why that
/// is correct rather than merely safe). A cursor built with `across` writes
/// each region's stream into the cell, waits the events the region waits on
/// and records the ones it records — the fork/join pattern
/// `.wiki/tart/evidence/green_contexts.md` Finding 3 measured, in the order
/// `driver::fire::walk` emits it.
///
/// A device call inside a `Sink` method has nowhere to return an error to, so
/// the first one is kept and [`Cursor::settle`] is where the caller asks. That
/// is not a swallowed error: a failed `cudaEventRecord` leaves the capture in
/// a state the caller must not instantiate, and the caller is the code that
/// knows it.
#[derive(Debug)]
pub struct Cursor<'a> {
    at: u32,
    place: &'a At,
    lanes: Option<Lanes<'a>>,
    /// Is this walk being WRITTEN DOWN?
    ///
    /// **NOT THE SAME QUESTION AS "does it have side streams".** A plan with
    /// no fork group captures through a cursor with no [`Lanes`], and it is
    /// still a capture — so the two are separate fields even though today's
    /// artifacts usually set both. What reads it is
    /// [`cond_begin`](Sink::cond_begin), where the difference between the two
    /// modes is the difference between ignoring a conditional (correct) and
    /// recording its body unconditionally (silently wrong).
    recording: bool,
    fault: Option<Fault>,
}

impl<'a> Cursor<'a> {
    /// A cursor writing into `place`, on the main stream from end to end.
    #[must_use]
    pub fn new(place: &'a At) -> Cursor<'a> {
        place.region.set(0);
        place.run.set(0);
        Cursor {
            at: 0,
            place,
            lanes: None,
            recording: false,
            fault: None,
        }
    }

    /// The same cursor, told that what it is walking is being recorded.
    ///
    /// The one thing a capture pass must say about itself that a stream
    /// assignment does not already say — see [`Cursor::recording`].
    #[must_use]
    pub fn writing(self) -> Cursor<'a> {
        Cursor {
            recording: true,
            ..self
        }
    }

    /// The same, plus P6: switch streams at every region boundary and put the
    /// baked event points on the device.
    #[must_use]
    pub fn across(place: &'a At, lanes: Lanes<'a>) -> Cursor<'a> {
        place.region.set(0);
        place.run.set(0);
        lanes.at.set(0);
        Cursor {
            at: 0,
            place,
            lanes: Some(lanes),
            recording: false,
            fault: None,
        }
    }

    /// What the device refused during the walk, if anything.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] from a `cudaEventRecord` or a `cudaStreamWaitEvent`,
    /// or [`Fault::Unbound`] for a template naming a stream or an event this
    /// load never opened — which is a `Baked` and a `Context` that were not
    /// set up from each other.
    pub fn settle(self) -> Result<()> {
        match self.fault {
            Some(fault) => Err(fault),
            None => Ok(()),
        }
    }

    /// The stream the current region is on, or the fault for a region naming
    /// one this load did not open.
    fn stream(&self, lanes: &Lanes<'a>) -> core::result::Result<*mut core::ffi::c_void, Fault> {
        match lanes.at.get() {
            0 => Ok(lanes.main),
            n => lanes
                .side
                .get(n as usize - 1)
                .copied()
                .ok_or_else(|| Fault::Unbound {
                    what: format!(
                        "region {} on stream {n}, and this load opened {}",
                        self.at.saturating_sub(1),
                        lanes.side.len(),
                    ),
                }),
        }
    }

    /// Record or wait one event on the current stream. `record` chooses which.
    fn event(&mut self, id: EventId, record: bool) {
        let Some(lanes) = self.lanes else {
            return;
        };
        // The first fault wins: a later call on a stream whose earlier event
        // failed says nothing new, and the caller wants the one that started
        // it.
        if self.fault.is_some() {
            return;
        }
        let outcome = self.stream(&lanes).and_then(|stream| {
            let Some(event) = lanes.events.get(id.0 as usize) else {
                return Err(Fault::Unbound {
                    what: format!(
                        "event {}, and this load created {}",
                        id.0,
                        lanes.events.len(),
                    ),
                });
            };
            if record {
                event.record(stream)
            } else {
                event.wait(stream)
            }
        });
        if let Err(fault) = outcome {
            self.fault = Some(fault);
        }
    }
}

impl Sink for Cursor<'_> {
    fn region_begin(&mut self, region: &Region) {
        self.place.region.set(self.at);
        self.place.run.set(0);
        self.at += 1;
        // The stream switch, and it is the whole of it: everything the `Run`
        // resolves afterwards fires on whatever this names, until the next
        // region says otherwise.
        if let Some(lanes) = self.lanes {
            lanes.at.set(region.stream);
        }
    }
    fn region_end(&mut self, _region: &Region) {}

    /// **THE SPLIT'S ONE PIECE OF STATE.** A region P4 could not seat runs
    /// once per interval of its class set, and every operand the `Run`
    /// resolves after this call is cut at THAT interval — its rows, its lanes,
    /// its rebased qo boundaries. A cursor that ignored this would hand every
    /// run the first one's window, which is not a fault: it is the first
    /// interval's rows computed `r` times and the rest never computed at all.
    fn run(&mut self, run: u32, _runs: u32) {
        self.place.run.set(run);
    }

    /// **THE EAGER CURSOR IGNORES IT AND THE RECORDING ONE REFUSES IT.**
    ///
    /// Ignoring is correct for an eager pass and it is not a shortcut: the
    /// walk's zero-row rule decides exactly what a conditional decides, at the
    /// same instant, so a fire that walks a conditional region eagerly runs
    /// the same nodes over the same rows (design §4 — conditionals are the
    /// optimization, zero-row always-launch is the semantics). That is what
    /// `driver::fire::EagerSink` says too, and why the two agree.
    ///
    /// A CAPTURE CANNOT IGNORE IT. The graph outlives the fire that recorded
    /// it, so a body recorded outside its conditional node is a body that runs
    /// under every composition the exec is replayed for — and it would
    /// compute. So the recording cursor answers [`Fault::Unlowered`], which
    /// names the region and says what is missing; see that variant for the two
    /// things this shell would need, neither of which is the cudarc binding.
    fn cond_begin(&mut self, lowering: &Lowering) {
        if !self.recording || self.fault.is_some() {
            return;
        }
        self.fault = Some(Fault::Unlowered {
            region: self.at.saturating_sub(1),
            lowering: format!("{lowering:?}"),
        });
    }
    fn cond_arm(&mut self, _arm: u8) {}
    fn cond_end(&mut self) {}
    fn fork(&mut self, event: EventId) {
        self.event(event, true);
    }
    fn join(&mut self, event: EventId) {
        self.event(event, false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use driver::fire::{ClassWindow, WindowTable};
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

    /// A region shaped like the one P3 picks: windowed, in the capture phase,
    /// and behind a conditional node.
    fn conditional() -> Region {
        Region {
            nodes: 0..26,
            mask: ClassSet::of([0]),
            phase: model_compiler::Phase::Capture,
            lowering: Lowering::If,
            stream: 0,
            wait: Vec::new(),
            open: None,
            close: None,
            sm_hint: None,
            collective: false,
        }
    }

    #[test]
    fn an_eager_cursor_ignores_a_conditional_and_a_recording_one_refuses_it() {
        let cell = At::new();
        let mut eager = Cursor::new(&cell);
        let region = conditional();
        eager.region_begin(&region);
        eager.cond_begin(&region.lowering);
        eager.cond_end();
        eager.region_end(&region);
        // Correct, and not a shortcut: the walk's zero-row rule decides what
        // the conditional decides, so an eager pass runs the same nodes over
        // the same rows (design §4).
        eager.settle().expect("an eager walk ignores the bracket");

        let cell = At::new();
        let mut recording = Cursor::new(&cell).writing();
        recording.region_begin(&region);
        recording.cond_begin(&region.lowering);
        let fault = recording
            .settle()
            .expect_err("a capture may not record a body outside its node");
        assert!(matches!(fault, Fault::Unlowered { region: 0, .. }), "{fault}");
        assert!(fault.to_string().contains("conditional nodes"));
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

    /// **THE EASIEST THING IN THE SPLIT TO GET SILENTLY WRONG.** A window's qo
    /// boundaries are offsets INTO the rectangle it cuts, so every run of a
    /// fragmented window needs its OWN vector, rebased to its own zero over
    /// its own lanes. Handing run 1 the vector rebased for run 0 does not
    /// fault: the schedule's work items index a boundary list that describes
    /// somebody else's requests, and the answer is wrong logits for every lane
    /// past the first interval.
    #[test]
    fn each_run_of_a_fragmented_window_rebases_its_own_boundaries() {
        // Three classes, and the middle one is not in the mask: 2 prefill
        // lanes of 3 rows, 1 lane of 5, 2 lanes of 4.
        let table = WindowTable::new(vec![
            ClassWindow {
                row_offset: 0,
                rows: 3,
                lane_offset: 0,
                lanes: 2,
            },
            ClassWindow {
                row_offset: 3,
                rows: 5,
                lane_offset: 2,
                lanes: 1,
            },
            ClassWindow {
                row_offset: 8,
                rows: 4,
                lane_offset: 3,
                lanes: 2,
            },
        ]);
        let mask = ClassSet::of([0, 2]);
        assert_eq!(table.span(&mask), Err(2), "class 1's rows stand between");

        let spans = table.spans(&mask);
        assert_eq!(spans.len(), 2);
        assert_eq!((spans[0].row_offset, spans[0].rows), (0, 3));
        assert_eq!((spans[1].row_offset, spans[1].rows), (8, 4));

        // The fire's boundaries, over all five lanes.
        let indptr = [0, 1, 3, 8, 10, 12];
        assert_eq!(rebase(&indptr, spans[0]), vec![0, 1, 3]);
        assert_eq!(
            rebase(&indptr, spans[1]),
            vec![0, 2, 4],
            "the second run starts at ITS zero, not the fire's",
        );
    }
}
