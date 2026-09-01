//! The artifact (design §2). What one plan bakes down to, once, for a
//! deployment's whole life.
//!
//! WHY THIS IS NOT A SCHEDULE PER COMPOSITION. Which windows a fire has rows
//! for is RUNTIME DATA — it is the batch the runtime happened to assemble —
//! and there are 2^K of them. So the artifact is the one capture script plus
//! the tables that parameterize it: the regions in program order, the arena's
//! static offsets, the class table the descriptor's counts are indexed by.
//! Composition is absorbed by zero-row always-launch (decision #3), not by
//! picking a different script.
//!
//! BACKEND-NEUTRAL, and that is a load-bearing property rather than good
//! manners: the CUDA shell records this into a graph, the Metal shell and the
//! golden path walk the same regions eagerly, and "captured is eager by
//! construction" (decision #11) is only true if there is exactly one thing to
//! walk. Nothing here names a device, a stream handle or a kernel symbol.
//!
//! # What is not here yet
//!
//! Design §2 lists four more fields, and each belongs to a pass that has not
//! landed. They are ABSENT rather than empty, because an empty
//! `DescriptorAbi` would be a claim about a byte layout nobody has written:
//!
//! - `descriptor: DescriptorAbi` — P8, and it is the ONE mutable channel into
//!   a recorded graph (§5). Its shape follows from the region table below
//!   plus P6's events, so it is written after both.
//! - `prepare: Vec<PrepareOp>` — P8. [`Phase::Prepare`] already marks which
//!   nodes it will carry; what a prepare op writes into which descriptor slot
//!   is the descriptor's question.
//! - `params` / `caches` / `seams` — P8's static tables. Today a caller holds
//!   the `Trace` beside the `CompiledModel` and reads them there; the tables exist so
//!   that a shell need not, and copying them out before anything reads them
//!   would be a second spelling of `Trace::params` with nothing keeping the two
//!   in step.
//! - `identity: CacheIdentity` — plan hash x compiler version. A hash of a
//!   plan is a hash of its serialization, and this crate deliberately depends
//!   on no serializer.

use std::ops::Range;

use model_ir::{ClassSet, ClassTable, RowAxis, ValueId};

use crate::arena::{ArenaMap, Concurrency};
use crate::pq::PqTree;
use crate::stream::StreamPlan;

/// A recorded synchronization point: what a fork signals and a join waits on.
///
/// **P6'S CURRENCY, AND THE TWO VERBS IT DECOMPOSES INTO.** The capture
/// pattern green contexts Finding 3 measured is "record an event on one
/// stream, wait on it from another"; a fork and a join are that same pair
/// pointed in opposite directions. So the artifact carries the two halves
/// separately — [`Region::open`] and [`Region::close`] record,
/// [`Region::wait`] waits — and the
/// fork/join of design §6 is what they compose into:
///
/// ```text
/// fork   signal on the main stream  +  wait on the side stream
/// join   signal on the side stream  +  wait on the main stream
/// ```
///
/// The ids are dense from zero and numbered in emission order, so a shell
/// creates `CompiledModel::streams.events` of them once, at load, and indexes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EventId(pub u32);

/// One plan, baked.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledModel {
    /// P1's output, whole.
    ///
    /// DESIGN §2 SPELLS THIS `Vec<Class>` AND THIS KEEPS THE WHOLE
    /// [`ClassTable`]. `merge_arm` is what resolves a phi at the fire — which arm
    /// wrote a class's rows — and `node_mask` is what a region's mask was
    /// folded from; both are already in hand, and re-deriving either means
    /// running the 2^F sweep a second time. `classes.classes` is still the
    /// design's field, one dot further in.
    pub classes: ClassTable,
    /// P2's output: maximal runs of adjacent nodes that run in the same
    /// classes and belong to the same phase. Program order, and the record
    /// script walks them front to back.
    pub regions: Vec<Region>,
    /// P4's output on the TOKEN axis: the global class order, as the whole
    /// feasible set.
    ///
    /// **ONE ORDER PER AXIS, AND THIS IS THE FIRST ONE'S** (multimodal §5.1 —
    /// "P7's one row order" reads "one per axis"). It keeps the design's
    /// spelling because the token rectangle is the axis every plan has and
    /// every pre-campaign caller means; [`patches`](CompiledModel::patches)
    /// carries the second, and [`order_for`](CompiledModel::order_for) is how
    /// a caller that knows about axes asks without knowing which field.
    pub order: ClassOrder,
    /// P4's other output on the token axis: what the consumers that order
    /// could not seat do instead, per bucket range — ranges into
    /// `Budget::buckets`.
    pub fallback: FallbackTable,
    /// P7's output: one slot per plan value, static offsets, rows symbolic in
    /// the bucket.
    pub arena: ArenaMap,
    /// P6's seam: which regions the engine may have in flight at once, and
    /// what the arena was carved against — two values may share bytes only if
    /// no instant holds them both, and "instant" is exactly what this relation
    /// widens once regions run beside each other.
    /// [`Concurrency::sequential`] is what a plan with no fork group gets, and
    /// it is what one stream means.
    pub concurrency: Concurrency,
    /// P6's other output: how many streams and how many events the template
    /// names, and the region pairs [`concurrency`](CompiledModel::concurrency) was
    /// built from. All zero-ish — one stream, no events, no pairs — for a plan
    /// with nothing to overlap, and for every plan when the profile's
    /// `side_streams` is 0.
    pub streams: StreamPlan,
    /// The capture units, in exec order — one row axis each (multimodal §1).
    ///
    /// `[RowAxis::Tokens]` for every plan that states one row space, which is
    /// every pre-campaign SKU and the M1 invariant a test pins. A plan with a
    /// vision tower carries two, the tower's first, and the fire launches one
    /// exec per entry chained on one stream: `prepare(all) → capture(tower) →
    /// capture(trunk)`, the embed handoff riding stream order with no host in
    /// it (Article 2).
    ///
    /// LADDERS ARE ONE-DIMENSIONAL PER UNIT. Six token rungs and six patch
    /// rungs are twelve graphs, not thirty-six — see
    /// [`fold_refused`](CompiledModel::fold_refused) for what that property
    /// currently costs.
    pub units: Vec<RowAxis>,
    /// Which capture unit each region is recorded into — an index into
    /// [`units`](CompiledModel::units), parallel to
    /// [`regions`](CompiledModel::regions) (multimodal §1).
    ///
    /// **DERIVED FROM THE ROW AXIS OF THE ROWS A REGION WRITES**, never
    /// declared — see [`crate::unit`] for the derivation and for what it
    /// refuses. All zero on every plan that states one row space, which is
    /// every pre-campaign SKU.
    ///
    /// PARALLEL RATHER THAN A FIELD ON [`Region`], for the reason
    /// `ClassTable::node_mask` is parallel to `Trace::nodes`: the answer is a
    /// whole-script one — which unit a region lands in depends on where every
    /// other region stands — so it is produced by one pass over the table and
    /// read by index, and `Region` keeps describing only what a run of nodes
    /// is. [`unit_of`](CompiledModel::unit_of) is the read.
    pub units_of: Vec<u32>,
    /// P4's answers on the PATCH axis, or `None` for a plan that states no
    /// patch row.
    ///
    /// **A SECOND SERIATION AND NOT AN EXTENSION OF THE FIRST**, because the
    /// invariant the first is composed under does not reach here: `compose`'s
    /// merged prefix sum has rows and lanes breaking at the same places, and a
    /// lane of a class may carry zero images or three. So the patch axis gets
    /// its own class order over its own lane space — IMAGES — and its own
    /// fallback rows, indexed into `PatchLadder::buckets` rather than into
    /// `Budget::buckets`.
    pub patches: Option<AxisPlan>,
    /// **THE FOLD STANDS DOWN, BY NAME** (multimodal §5.3).
    ///
    /// `true` exactly when this artifact has more than one capture unit. The
    /// graph-fold plane is structurally one graph per bucket per key, so a
    /// fire launching two execs has two bucket numbers and no single graph to
    /// arm; there is no correct fold to build, and the engine serves the KEYED
    /// path for the life of the load rather than folding something that would
    /// be wrong. Said out loud rather than discovered: "6 + 6, not 6 × 6" is a
    /// property OF per-unit keys, so deferring the fold defers the property,
    /// and a fire-level key carrying both bucket numbers would be the product.
    pub fold_refused: bool,
}

/// One row axis's baked answers — the pair P4 produces, per axis.
///
/// The token axis's pair is spelled out on [`CompiledModel`] itself
/// ([`order`](CompiledModel::order), [`fallback`](CompiledModel::fallback)),
/// because it is the axis every plan has and moving it into a table would
/// rename a field every caller in the tree reads. This is what a SECOND axis
/// arrives as, and [`CompiledModel::order_for`] is the spelling that does not
/// care which of the two it is.
#[derive(Debug, Clone, PartialEq)]
pub struct AxisPlan {
    /// Which row space these answers are about.
    pub axis: RowAxis,
    /// The class order a fire seriates this axis's rows by.
    pub order: ClassOrder,
    /// What the consumers that order could not seat do instead. The bucket
    /// ranges index THIS AXIS'S ladder — `PatchLadder::buckets` for
    /// [`RowAxis::Patches`] — which is the per-unit table multimodal §5.5
    /// asks for in place of an untagged index into one vector.
    pub fallback: FallbackTable,
}

impl CompiledModel {
    /// The capture script: the regions, in the order a recorder emits them.
    ///
    /// A METHOD AND NOT A FIELD, and P6 is where that stopped being an
    /// accident and became the answer. The script is still the region table
    /// walked front to back — what P6 added is not a second order but two
    /// fields ON a region: which stream it is recorded into, and which events
    /// it waits on and signals. A fork does not reorder the walk; it says
    /// where the walk's next launch lands. So a stored `CaptureTemplate` would
    /// still be a `Vec<Region>` copied beside a `Vec<Region>` with nothing to
    /// keep the two in step.
    ///
    /// **P3 DID NOT CHANGE IT EITHER, AND THAT WAS THE DESIGN QUESTION.** A
    /// SWITCH has arms and arms could have nested, which would have made the
    /// script a tree. It does not: a group's arms are CONSECUTIVE regions and
    /// each one says which arm of which merge it is
    /// ([`Lowering::Switch`]), so a recorder brackets a group by reading the
    /// members it is already walking past. Nesting is v1's flat one level
    /// (design's open items), and a nested conditional is what would need the
    /// second table.
    #[must_use]
    pub fn template(&self) -> &[Region] {
        &self.regions
    }

    /// Which capture unit this region is recorded into. `0` for a region index
    /// past the table, which is the one-unit answer and the only honest one
    /// for a region that is not there.
    #[must_use]
    pub fn unit_of(&self, region: usize) -> u32 {
        self.units_of.get(region).copied().unwrap_or(0)
    }

    /// **WHICH ROW SPACE THIS REGION'S OWN WINDOW IS A SPAN OF** — its capture
    /// unit's axis, which is the only thing a region's rows can be counted in.
    ///
    /// A METHOD BECAUSE FOUR READERS ASK IT AND MUST ALL PICK THE SAME WAY:
    /// `model_exec::fire::walk` cuts the region's spans with it,
    /// `model_exec::fire::fallback` resolves P4's menu with it, and both
    /// shells' window tables record it beside the windows they cut. It was
    /// written out four times as `units[unit_of(r)]` with a `PRIMARY` default,
    /// which is one expression that could drift in four places; it is one
    /// expression now.
    ///
    /// [`RowAxis::PRIMARY`] for a region past the table or a unit past
    /// [`units`](CompiledModel::units) — the one-unit answer, and the only
    /// honest one for a region that is not there, exactly as
    /// [`unit_of`](CompiledModel::unit_of) reads.
    #[must_use]
    pub fn axis_of(&self, region: usize) -> RowAxis {
        self.units
            .get(self.unit_of(region) as usize)
            .copied()
            .unwrap_or(RowAxis::PRIMARY)
    }

    /// The class order this axis's rows are seriated by, or `None` for an axis
    /// this plan does not state.
    #[must_use]
    pub fn order_for(&self, axis: RowAxis) -> Option<&ClassOrder> {
        match axis {
            RowAxis::Tokens => Some(&self.order),
            RowAxis::Patches => self.patches.as_ref().map(|plan| &plan.order),
        }
    }

    /// The fallback answers on this axis, or `None` for an axis this plan does
    /// not state. The rows' bucket ranges index the axis's OWN ladder.
    #[must_use]
    pub fn fallback_for(&self, axis: RowAxis) -> Option<&FallbackTable> {
        match axis {
            RowAxis::Tokens => Some(&self.fallback),
            RowAxis::Patches => self.patches.as_ref().map(|plan| &plan.fallback),
        }
    }

    /// The capture regions of one unit, as the half-open range of the record
    /// script an exec is recorded from.
    ///
    /// ONE RANGE AND NOT A LIST, which is exactly what [`crate::unit`]'s
    /// refusal buys: a unit whose regions were scattered down the script would
    /// have no such range, and the plan that produced one is refused at the
    /// bake rather than recorded as several execs of one name.
    #[must_use]
    pub fn unit_script(&self, unit: u32) -> Option<core::ops::Range<u32>> {
        let mut span: Option<core::ops::Range<u32>> = None;
        for (r, region) in self.regions.iter().enumerate() {
            if region.phase != Phase::Capture || self.unit_of(r) != unit {
                continue;
            }
            let r = r as u32;
            span = Some(match span {
                None => r..r + 1,
                Some(held) => held.start..r + 1,
            });
        }
        span
    }
}

/// Which half of a fire a node runs in (P5, absorbing the rewrite's
/// `kernels::exec::phases`).
///
/// THE RULE IS READ OFF THE TYPE AND NOT OFF A LIST OF OP NAMES (menlo §16):
/// a node is `Prepare` iff it defines a `Ty::Struct` — a host-owned plan
/// object, opaque to the IR, outside the arena. Those are the flashinfer-style
/// plan builds, and they are hoisted out of the graph into descriptor slots
/// (§5) precisely because they are host work that a captured graph cannot
/// contain. Everything else is `Capture`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    /// Runs before the launch, on the host, writing descriptor slots.
    Prepare,
    /// Runs inside the immutable graph.
    Capture,
}

/// How a region enters the graph (P3, design §4).
///
/// **CONDITIONALS ARE AN OPTIMIZATION, NOT THE SEMANTICS.** Every windowed
/// kernel is always in the graph and reads its row count from the descriptor;
/// an empty window returns immediately, at about a microsecond. That is what
/// makes composition runtime data instead of a recapture, and it is why
/// [`AlwaysLaunch`](Lowering::AlwaysLaunch) is the default and the fallback
/// and the answer on every region of every catalog plan today.
///
/// P3 reads [`DeviceProfile`](crate::DeviceProfile) and chooses one of the
/// other two for a region fat enough to amortize its evaluation point — layer
/// granularity or coarser — subject to the one rule that is not an
/// optimization: **a region whose [`Region::collective`] is set must stay
/// always-launch** (decision #5). A collective inside a skipped body
/// deadlocks the ranks that did not skip, or — worse — silently mispairs with
/// a later collective, because NCCL matches by call order. On today's catalog
/// it chooses `AlwaysLaunch` everywhere and [`crate::lowering`] says with what
/// arithmetic.
///
/// **AN EAGER SINK MAY IGNORE ALL OF THIS AND STILL BE CORRECT**, and a
/// RECORDING one may not. The walk's zero-row rule already decides what a
/// conditional decides — the composition simply does not include the behavior
/// — so `EagerSink`'s no-op `cond_begin` is the right implementation and not a
/// stub. A sink that RECORDS has to bracket the body, because a graph outlives
/// the fire that recorded it and the count it was recorded against is gone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lowering {
    /// In the graph unconditionally; the kernel reads `desc.count[region]` and
    /// returns on zero.
    AlwaysLaunch,
    /// One arm of a `cudaGraphCondTypeSwitch` over a merge's arms, exactly one
    /// taken.
    ///
    /// **THE GROUP IS FREE AND THE ACTIVATION IS NOT** (see
    /// [`crate::lowering`]). Decision #4's exclusivity is a fact about a LANE
    /// — `resolve_classes` proves no lane demands two arms — and a switch
    /// node's predicate is a fact about a FIRE, which is a batch of lanes of
    /// different classes. P3 constructs this only where at most one arm can be
    /// live in any composition the budgets admit.
    Switch {
        /// The merged value whose arms are the bodies.
        merge: ValueId,
        /// Which arm this region is, in `Def::Merge` order — the value the
        /// dispatch kernel writes when this region's class is the live one.
        arm: u8,
        /// How many arms the group has. `arm + 1 == arms` is the last one,
        /// which is where a recorder closes the switch: the region table is
        /// FLAT, so each member says where it stands in its group rather than
        /// a second table saying it for them.
        arms: u8,
    },
    /// One body behind a device-side predicate — an independent-presence axis
    /// with no exclusive sibling to switch against.
    ///
    /// **IT CARRIES NOTHING, AND THE PREDICATE IS [`Region::mask`].** Design
    /// §4 spells this `If { cond }`; a `Guard` here would be the authoring-level
    /// guard, and what the device evaluates is
    /// `descriptor.rows_of(region.mask) > 0` — the same number always-launch
    /// reads, asked one node earlier. Carrying the guard beside the mask would
    /// be two spellings of one window, free to disagree, which is the collapse
    /// build log 8 deleted `FireBindings::facts` over.
    If,
}

/// A maximal run of adjacent nodes that run in the same classes, in the same
/// phase (P2).
///
/// ADJACENT, AND THE PROGRAM ORDER IS KEPT. Reordering nodes to make runs
/// longer is a scheduling problem, correctness-neutral, and deferred (design's
/// open items). What that costs is measurable and small: a windowed op splits
/// its neighborhood into three regions where a reordering pass might have
/// managed one, and each extra boundary is a launch, not a recapture.
#[derive(Debug, Clone, PartialEq)]
pub struct Region {
    /// The half-open run of `Trace::nodes` this region covers.
    pub nodes: Range<u32>,
    /// The classes that run these nodes. Every node in the run has this exact
    /// mask — that equality is what defines the run — so a fire's row count
    /// for the region is one number, which is what the descriptor carries.
    ///
    /// An EMPTY mask is a region no class demands. `resolve_classes` reports
    /// those as `ClassTable::dead`; they stay here so that node indices keep
    /// meaning what they mean, and a shipped plan has none
    /// (`model/tests/every_class_resolves_every_merge.rs` says so).
    pub mask: ClassSet,
    /// Host prepare work, or graph body.
    pub phase: Phase,
    /// How it enters the graph (P3). [`Lowering::AlwaysLaunch`] on every
    /// region of every catalog plan at today's profile — see
    /// [`crate::lowering`] for the two gates that say so.
    pub lowering: Lowering,
    /// Which stream it is recorded into (P6). `0` is the main stream, which
    /// is where a region stays unless [`crate::stream`] found it a partner
    /// worth overlapping with.
    pub stream: u32,
    /// Events this region's stream waits on BEFORE its first node.
    ///
    /// Two things arrive here and they are the two halves of design §6's
    /// pair: a side stream's ENTRY wait (the one event the group's main arm
    /// opened with) and the main stream's REJOIN (one per arm that left it).
    /// Empty for every region of a plan P6 found nothing in.
    pub wait: Vec<EventId>,
    /// The event recorded on this region's stream BEFORE its first node, and
    /// after [`wait`](Region::wait) — the fork point.
    ///
    /// **AT THE START, AND THAT IS THE WHOLE REASON IT IS ITS OWN FIELD.** A
    /// group's arms must be ordered after everything that ran before the
    /// group and NOT after the group's own main arm, which is the instant
    /// between this region's waits and its first launch. Recording it at the
    /// end of the region BEFORE instead would be the same instant only when
    /// that region is on the main stream — and it is not, whenever two fork
    /// groups stand back to back, which is exactly what a transformer layer
    /// does (the qkv pair, then the attention arms).
    pub open: Option<EventId>,
    /// The event recorded on this region's stream AFTER its last node — the
    /// join point. Set on the arms that left the main stream; the region
    /// after the group waits on every one of them.
    pub close: Option<EventId>,
    /// What fraction of the device this region wants, if the profile has an
    /// opinion.
    ///
    /// `Some` exactly on the members of a fork group, proportional to their
    /// estimated cost and rounded to the granularity green contexts Finding 1
    /// measured (multiples of 2 SMs, minimum 4). **NOTHING READS IT IN v1**:
    /// the SM partition is baked at capture (Finding 5), so a variant
    /// multiplies bodies, and decision #14 ships the hint rather than the
    /// partition.
    pub sm_hint: Option<u32>,
    /// This run contains a `Collective`-family node.
    ///
    /// P0 RECORDS IT AND P3 IS BOUND BY IT. A collective takes any `guard` it
    /// likes — a rank's guard is its own business — but a region carrying one
    /// may never become a conditional body, because a zero-count collective
    /// still has to join the rendezvous. See [`Lowering`].
    pub collective: bool,
}

/// The global lane order a fire seriates its rows by (P4's answer).
///
/// A windowed structural consumer runs as one kernel over pointer+extent iff
/// its class set is an INTERVAL of the lane ordering — the Consecutive-Ones
/// Property, exact and linear via PQ-trees. One arena and one global row order
/// per fire means one global C1P instance (decision #8), and the tree ships
/// WHOLE — not one witness ordering — so the fire path can pick the feasible
/// order closest to the previous fire, which is what keeps pointers from
/// churning. See [`crate::layout`] for the instance and the greedy.
///
/// ADDITIVE, AND [`Identity`](ClassOrder::Identity) IS STILL A TRUE ANSWER.
/// It is what a plan gets when P4 declines to seriate it — more classes than a
/// `u8` names, or none at all — and it is correct for every plan, since a
/// consumer whose classes are not an interval simply runs as more than one
/// launch. It is no longer what `compile` returns for an ordinary plan.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ClassOrder {
    /// Lanes fire in the order the runtime assembled them: classes ascending.
    #[default]
    Identity,
    /// Every ordering under which the seated consumers are intervals, with the
    /// tree's canonical frontier as v1's pick.
    Seriated(PqTree),
}

impl ClassOrder {
    /// The order the classes PRESENT in one fire are seriated in.
    ///
    /// THE ONE THING THE FIRE PATH ASKS. A fire has rows for some subset of
    /// the plan's classes — which subset is the batch the runtime happened to
    /// assemble, and there are `2^K` of them, which is why this is a filter
    /// over one baked answer rather than a table with a row per composition.
    /// Absent classes simply drop out: a sub-order of an ordering that makes a
    /// set consecutive still makes it consecutive.
    ///
    /// A CLASS IS A BYTE HERE AND EVERYWHERE DOWNSTREAM, so this answers
    /// usefully only for a plan the descriptor could name the classes of.
    /// P4 declines to seriate one past that ceiling (see
    /// [`crate::layout`]), and P8 is where it becomes a refusal.
    ///
    /// `prev` IS LAST FIRE'S ORDER, AND v1 IGNORES IT. The refinement it is
    /// there for is design §3's stability pick — of all the orderings the tree
    /// admits, prefer the one closest to the one the graph's pointers are
    /// already bound to, since re-binding a kernel node costs ~0.11 us per
    /// node and doing it for no reason costs that times every node. The
    /// parameter is here NOW so that landing the pick is a body change: this
    /// signature is what an engine's fire path compiles against, and it is not
    /// going to move under it.
    #[must_use]
    pub fn class_order(&self, present: &ClassSet, prev: Option<&[u8]>) -> Vec<u8> {
        // v1's whole use of last fire's order. See above.
        let _ = prev;
        match self {
            ClassOrder::Identity => present.iter().map(|class| class as u8).collect(),
            ClassOrder::Seriated(tree) => tree
                .frontier()
                .iter()
                .copied()
                .filter(|&class| present.contains(class as usize))
                .collect(),
        }
    }

    /// The feasible set itself, when P4 solved one.
    ///
    /// [`PqTree::admits`] is the question the stability pick needs and
    /// [`class_order`](ClassOrder::class_order) cannot answer: "is the order
    /// I am already bound to still legal?". `None` is [`Identity`](ClassOrder::Identity),
    /// which promises nothing and therefore admits everything.
    #[must_use]
    pub fn tree(&self) -> Option<&PqTree> {
        match self {
            ClassOrder::Identity => None,
            ClassOrder::Seriated(tree) => Some(tree),
        }
    }
}

/// Per-node, per-bucket answers for the structural consumers a lane order
/// cannot make consecutive (P4's other seam).
///
/// BUCKET-DEPENDENT, WHICH IS WHY IT IS A TABLE AND NOT A FIELD ON THE NODE.
/// The measured crossover: at decode scale (M=64) a copy costs 1.07x the ideal
/// and a split 1.82x, and the two converge at prefill scale. One answer per
/// node would be the wrong answer at one end of the lattice.
///
/// EMPTY IS THE GOOD CASE, AND THE CATALOG STOPPED BEING IT. It was true while
/// every window a model text stated nested inside another — `masked` inside
/// everything, `qo_one` inside `masked` — because a laminar family is always
/// C1P. `captures_scores` (palo C4) does not nest: it CROSSES `qo_one`, and
/// over the fourteen-point bucket lattice the four qwen3.5 texts each owe 12
/// or 20 rows for it — two per `Attention::PrefillLse` node, a
/// [`Fallback::Copy`] below the crossover and a [`Fallback::Split`] above —
/// while qwen3.6-27b owes 84 over the 42 nodes of its MTP head.
///
/// A row here names a consumer the C1P instance could not seat — see
/// [`crate::layout`] for which one gets withdrawn and why — and
/// `model_exec::fire::fallback` is what reads it, at the bucket the fire
/// landed in and through the AXIS whose table this is
/// ([`CompiledModel::fallback_for`]; a plan has one menu per row space, and
/// the reader takes the axis rather than assuming the token one):
/// [`Fallback::Split`] is what every shell serves, and [`Fallback::Copy`] is
/// served by a shell that publishes a row gather and says so
/// (`fallback::Serve`) — `engine-cuda` does; `engine-metal` does not, and
/// splits.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FallbackTable {
    /// One row per (node, bucket range) that needs one.
    pub rows: Vec<FallbackRow>,
}

/// One row of the [`FallbackTable`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FallbackRow {
    /// Index into `Trace::nodes`.
    pub node: u32,
    /// The half-open range of `Budget::buckets` indices this answer covers.
    /// A deployment that declared no bucket lattice has ONE implicit bucket at
    /// `Budget::max_tokens`, and a row over it reads `0..1`.
    pub buckets: Range<u32>,
    /// What to do instead.
    pub fallback: Fallback,
}

/// What a structural consumer does when its classes are not an interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fallback {
    /// Read the rows where they lie — legal when the consumer can take a
    /// stride or an index list without paying for it.
    View,
    /// Run the kernel `r` times, once per maximal interval.
    Split {
        /// How many launches the class set breaks into, under the order the
        /// artifact ships. `1` is the free case — the consumer's classes came
        /// out contiguous even though nothing promised they would — and it is
        /// still written down, because the promise is what a fallback row
        /// records and this consumer has none.
        r: u32,
    },
    /// Gather the rows into one contiguous block, run once, scatter back —
    /// the same mechanism the weight-varied ops use, which is why adapters
    /// never enter the layout constraint at all (decision #9).
    Grouped,
    /// Copy the rows contiguous and run once. Cheapest at decode scale.
    Copy,
}
