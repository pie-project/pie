//! The artifact: what one plan bakes down to, once, for a deployment's whole
//! life — backend-neutral regions, arena offsets, and class table.

use std::ops::Range;

use model_ir::{ClassSet, ClassTable, RowAxis, ValueId};

use crate::arena::{ArenaMap, Concurrency};
use crate::pq::PqTree;
use crate::stream::StreamPlan;

/// A recorded synchronization point: what a fork signals and a join waits on.
/// [`Region::open`]/[`Region::close`] record, [`Region::wait`] waits:
///
/// ```text
/// fork   signal on the main stream  +  wait on the side stream
/// join   signal on the side stream  +  wait on the main stream
/// ```
///
/// Ids are dense from zero, numbered in emission order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EventId(pub u32);

/// One plan, baked.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledModel {
    /// The whole class table: `merge_arm` resolves a phi at the fire, and
    /// `node_mask` is what a region's mask was folded from.
    pub classes: ClassTable,
    /// Maximal runs of adjacent nodes that run in the same classes and
    /// belong to the same phase. Program order; the record script walks
    /// them front to back.
    pub regions: Vec<Region>,
    /// The global class order on the token axis, as the whole feasible set.
    /// [`patches`](CompiledModel::patches) carries the second axis;
    /// [`order_for`](CompiledModel::order_for) reads either by axis.
    pub order: ClassOrder,
    /// What the consumers that `order` could not seat do instead, per
    /// bucket range — ranges into `Budget::buckets`.
    pub fallback: FallbackTable,
    /// One slot per plan value: static offsets, rows symbolic in the bucket.
    pub arena: ArenaMap,
    /// Which regions the engine may have in flight at once, and what the
    /// arena was carved against. [`Concurrency::sequential`] is what a plan
    /// with no fork group gets.
    pub concurrency: Concurrency,
    /// How many streams and events the template names, and the region pairs
    /// [`concurrency`](CompiledModel::concurrency) was built from. All
    /// zero-ish for a plan with nothing to overlap.
    pub streams: StreamPlan,
    /// The capture units, in exec order — one row axis each.
    ///
    /// `[RowAxis::Tokens]` for a plan with one row space. A plan with a
    /// vision tower carries two, and the fire launches one exec per entry
    /// chained on one stream: `prepare(all) -> capture(tower) ->
    /// capture(trunk)`.
    pub units: Vec<RowAxis>,
    /// Which capture unit each region is recorded into — an index into
    /// [`units`](CompiledModel::units), parallel to
    /// [`regions`](CompiledModel::regions). Derived from the row axis of the
    /// rows a region writes, never declared; all zero on a single-row-space
    /// plan. [`unit_of`](CompiledModel::unit_of) is the read.
    pub units_of: Vec<u32>,
    /// The patch axis's own answers, or `None` for a plan with no patch row.
    /// A second seriation, not an extension of the token one: the patch axis
    /// gets its own class order over its own lane space (images) and its
    /// own fallback rows, indexed into `PatchLadder::buckets`.
    pub patches: Option<AxisPlan>,
    /// `true` exactly when this artifact has more than one capture unit. A
    /// fire launching two execs has two bucket numbers and no single graph
    /// to fold, so the engine serves the keyed path for the life of the load
    /// instead.
    pub fold_refused: bool,
}

/// One row axis's baked answers.
///
/// The token axis's pair is spelled out on [`CompiledModel`] itself
/// ([`order`](CompiledModel::order), [`fallback`](CompiledModel::fallback));
/// this is what a second axis arrives as, and
/// [`CompiledModel::order_for`] reads either without caring which.
#[derive(Debug, Clone, PartialEq)]
pub struct AxisPlan {
    /// Which row space these answers are about.
    pub axis: RowAxis,
    /// The class order a fire seriates this axis's rows by.
    pub order: ClassOrder,
    /// What the consumers that order could not seat do instead. The bucket
    /// ranges index this axis's own ladder — `PatchLadder::buckets` for
    /// [`RowAxis::Patches`].
    pub fallback: FallbackTable,
}

impl CompiledModel {
    /// The capture script: the regions, in the order a recorder emits them.
    /// A method, not a field: a fork does not reorder the walk, it only says
    /// where the walk's next launch lands, so the region table itself
    /// carries the stream and event fields.
    #[must_use]
    pub fn template(&self) -> &[Region] {
        &self.regions
    }

    /// Which capture unit this region is recorded into. `0` for a region
    /// index past the table — the one-unit answer.
    #[must_use]
    pub fn unit_of(&self, region: usize) -> u32 {
        self.units_of[region]
    }

    /// Which row space this region's own window is a span of: its capture
    /// unit's axis. A method because several readers ask it and must all
    /// pick the same way (`model_exec::fire::walk`/`fallback`, both shells'
    /// window tables).
    ///
    /// [`RowAxis::PRIMARY`] for a region past the table or a unit past
    /// [`units`](CompiledModel::units).
    #[must_use]
    pub fn axis_of(&self, region: usize) -> RowAxis {
        self.units[self.unit_of(region) as usize]
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
    /// script an exec is recorded from. One range, not a list: a unit whose
    /// regions were scattered down the script is refused at the bake
    /// ([`crate::unit`]).
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

/// Which half of a fire a node runs in.
///
/// A node is `Prepare` iff it defines a `Ty::Struct` — a host-owned plan
/// object, opaque to the IR, outside the arena — hoisted out of the graph
/// into descriptor slots because a captured graph cannot contain host work.
/// Everything else is `Capture`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    /// Runs before the launch, on the host, writing descriptor slots.
    Prepare,
    /// Runs inside the immutable graph.
    Capture,
}

/// How a region enters the graph. Conditionals are an optimization, not the
/// semantics: every windowed kernel is always in the graph and reads its row
/// count from the descriptor, so [`AlwaysLaunch`](Lowering::AlwaysLaunch) is
/// the default. A region whose [`Region::collective`] is set must stay
/// always-launch, since a skipped collective deadlocks or mispairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lowering {
    /// In the graph unconditionally; the kernel reads `desc.count[region]` and
    /// returns on zero.
    AlwaysLaunch,
    /// One arm of a `cudaGraphCondTypeSwitch` over a merge's arms, exactly
    /// one taken. Exclusivity is a fact about a lane (`resolve_classes`
    /// proves no lane demands two arms); this is constructed only where at
    /// most one arm can be live in any composition the budgets admit.
    Switch {
        /// The merged value whose arms are the bodies.
        merge: ValueId,
        /// Which arm this region is, in `Def::Merge` order.
        arm: u8,
        /// How many arms the group has. `arm + 1 == arms` is the last one,
        /// where a recorder closes the switch.
        arms: u8,
    },
    /// One body behind a device-side predicate — an independent-presence
    /// axis with no exclusive sibling to switch against. Carries nothing;
    /// the predicate is [`Region::mask`] (`descriptor.rows_of(region.mask) >
    /// 0`), the same number always-launch reads.
    If,
}

/// A maximal run of adjacent nodes that run in the same classes, in the same
/// phase. Program order is kept; reordering to make runs longer is deferred
/// (correctness-neutral, small cost: an extra boundary is a launch, not a
/// recapture).
#[derive(Debug, Clone, PartialEq)]
pub struct Region {
    /// The half-open run of `Trace::nodes` this region covers.
    pub nodes: Range<u32>,
    /// The classes that run these nodes. Every node in the run has this
    /// exact mask, so a fire's row count for the region is one number.
    ///
    /// An empty mask is a region no class demands (`ClassTable::dead`); it
    /// stays here so node indices keep meaning what they mean.
    pub mask: ClassSet,
    /// Host prepare work, or graph body.
    pub phase: Phase,
    /// The row axis its launches count rows on; `None` for a run of nodes
    /// that state none.
    pub axis: Option<RowAxis>,
    /// How it enters the graph. [`Lowering::AlwaysLaunch`] on every region
    /// of every catalog plan at today's profile.
    pub lowering: Lowering,
    /// Which stream it is recorded into. `0` is the main stream, where a
    /// region stays unless [`crate::stream`] found it a partner to overlap.
    pub stream: u32,
    /// Events this region's stream waits on before its first node: a side
    /// stream's entry wait, and the main stream's rejoin. Empty for a plan
    /// with no fork group.
    pub wait: Vec<EventId>,
    /// The event recorded on this region's stream before its first node,
    /// after [`wait`](Region::wait) — the fork point. Its own field because
    /// a group's arms must be ordered after the group's own main arm, not
    /// after everything before it — not the same instant unless the region
    /// is on the main stream.
    pub open: Option<EventId>,
    /// The event recorded on this region's stream after its last node — the
    /// join point. Set on the arms that left the main stream; the region
    /// after the group waits on every one of them.
    pub close: Option<EventId>,
    /// This run contains a `Collective`-family node. A region carrying one
    /// may never become a conditional body: a zero-count collective still
    /// has to join the rendezvous. See [`Lowering`].
    pub collective: bool,
}

/// The global lane order a fire seriates its rows by. A windowed structural
/// consumer runs as one kernel over pointer+extent iff its class set is an
/// interval of this ordering (the Consecutive-Ones Property, via PQ-trees;
/// see [`crate::layout`]). [`Identity`](ClassOrder::Identity) is what a plan
/// gets when seriation is declined; still correct, since a non-interval
/// consumer just runs as more than one launch.
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
    /// The order the classes present in one fire are seriated in.
    ///
    /// The one thing the fire path asks. A fire has rows for some subset of
    /// the plan's classes, so this filters one baked answer rather than
    /// tabling a row per composition; absent classes simply drop out.
    ///
    /// A class is a byte here and downstream, so this only answers usefully
    /// for a plan the descriptor could name the classes of.
    ///
    /// `prev` is last fire's order; v1 ignores it (reserved for a future
    /// stability pick that prefers the ordering closest to what the graph's
    /// pointers are already bound to, since re-binding a node has a cost).
    #[must_use]
    pub fn class_order(&self, present: &ClassSet) -> Vec<u8> {
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

    /// The feasible set itself, when one was solved.
    ///
    /// [`PqTree::admits`] answers "is the order I am already bound to still
    /// legal?", which [`class_order`](ClassOrder::class_order) cannot.
    /// `None` is [`Identity`](ClassOrder::Identity), which admits everything.
    #[must_use]
    pub fn tree(&self) -> Option<&PqTree> {
        match self {
            ClassOrder::Identity => None,
            ClassOrder::Seriated(tree) => Some(tree),
        }
    }
}

/// Per-node, per-bucket answers for the structural consumers a lane order
/// cannot make consecutive. Bucket-dependent (a copy and a split have
/// different relative costs at decode vs. prefill scale), read by
/// `model_exec::fire::fallback` at the bucket the fire landed in.
/// [`Fallback::Split`] is what every shell serves; [`Fallback::Copy`] needs
/// a shell that publishes a row gather (`engine-cuda` does, `engine-metal`
/// splits instead).
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
    /// A deployment with no bucket lattice has one implicit bucket at
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
        /// artifact ships. `1` is the free case, still written down since
        /// this consumer has no other record of the promise.
        r: u32,
    },
    /// Gather the rows into one contiguous block, run once, scatter back —
    /// the same mechanism the weight-varied ops use.
    Grouped,
    /// Copy the rows contiguous and run once. Cheapest at decode scale.
    Copy,
}

impl Region {
    /// Launches on the capture stream: capture phase, not a collective, at
    /// least one class.
    #[must_use]
    pub fn launches(&self) -> bool {
        self.phase == Phase::Capture && !self.collective && !self.mask.is_empty()
    }

    /// Launches over a proper subset of the plan's `all` classes.
    #[must_use]
    pub fn windowed(&self, all: usize) -> bool {
        self.launches() && self.mask.len() < all
    }
}
