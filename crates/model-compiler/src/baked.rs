//! The artifact (design §2). What one plan bakes down to, once, for a
//! deployment's whole life.
//!
//! WHY THIS IS NOT A SCHEDULE PER COMPOSITION. Which windows a fire has rows
//! for is RUNTIME DATA — it is the batch the engine happened to assemble —
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
//!   the `Plan` beside the `Baked` and reads them there; the tables exist so
//!   that a shell need not, and copying them out before anything reads them
//!   would be a second spelling of `Plan::params` with nothing keeping the two
//!   in step.
//! - `identity: CacheIdentity` — plan hash x compiler version. A hash of a
//!   plan is a hash of its serialization, and this crate deliberately depends
//!   on no serializer.

use std::ops::Range;

use model_ir::{ClassSet, Classes, Cond, ValueId};

use crate::arena::{ArenaMap, Concurrency};
use crate::layout::PqTree;

/// One plan, baked.
#[derive(Debug, Clone, PartialEq)]
pub struct Baked {
    /// P1's output, whole.
    ///
    /// DESIGN §2 SPELLS THIS `Vec<Class>` AND THIS KEEPS THE WHOLE
    /// [`Classes`]. `merge_arm` is what resolves a phi at the fire — which arm
    /// wrote a class's rows — and `node_mask` is what a region's mask was
    /// folded from; both are already in hand, and re-deriving either means
    /// running the 2^F sweep a second time. `classes.classes` is still the
    /// design's field, one dot further in.
    pub classes: Classes,
    /// P2's output: maximal runs of adjacent nodes that run in the same
    /// classes and belong to the same phase. Program order, and the record
    /// script walks them front to back.
    pub regions: Vec<Region>,
    /// P4's output: the global class order, as the whole feasible set.
    pub order: LayoutOrder,
    /// P4's other output: what the consumers that order could not seat do
    /// instead, per bucket range.
    pub fallback: FallbackTable,
    /// P7's output: one slot per plan value, static offsets, rows symbolic in
    /// the bucket.
    pub arena: ArenaMap,
    /// P6's seam: which regions the driver may have in flight at once.
    /// [`Concurrency::sequential`] today, which is what one stream means, and
    /// what the arena was carved against — two values may share bytes only if
    /// no instant holds them both, and "instant" is exactly what this relation
    /// widens once regions can run beside each other.
    pub concurrency: Concurrency,
}

impl Baked {
    /// The capture script: the regions, in the order a recorder emits them.
    ///
    /// A METHOD AND NOT A FIELD, and the reason is that in v1 the script IS
    /// the region table. Every region lowers as [`Lowering::AlwaysLaunch`],
    /// there are no conditional bodies to nest and no event nodes to interleave
    /// (one stream), so a stored `CaptureTemplate` would be a `Vec<Region>`
    /// copied beside a `Vec<Region>` with nothing to keep the two in step.
    ///
    /// When P3 and P6 land, the script stops being a straight walk — a SWITCH
    /// has arms, a fork has a join — and this becomes the field design §2
    /// names. The signature is what changes then; the callers are already
    /// asking the right question.
    #[must_use]
    pub fn template(&self) -> &[Region] {
        &self.regions
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
/// makes composition runtime data instead of a recapture, and it is why v1
/// constructs [`AlwaysLaunch`](Lowering::AlwaysLaunch) for every region and
/// nothing else.
///
/// P3 will read [`DeviceProfile`](crate::DeviceProfile) and choose one of the
/// other two for a region fat enough to amortize its evaluation point — layer
/// granularity or coarser — subject to the one rule that is not an
/// optimization: **a region whose [`Region::collective`] is set must stay
/// always-launch** (decision #5). A collective inside a skipped body
/// deadlocks the ranks that did not skip, or — worse — silently mispairs with
/// a later collective, because NCCL matches by call order.
#[derive(Debug, Clone, PartialEq)]
pub enum Lowering {
    /// In the graph unconditionally; the kernel reads `desc.count[region]` and
    /// returns on zero.
    AlwaysLaunch,
    /// One body per arm of a merge, exactly one taken. The arms ARE the
    /// exclusive variant set — that exclusivity is already in the IR, which is
    /// what makes `Def::Merge` the SWITCH group for free (decision #4).
    Switch {
        /// The merged value whose arms are the bodies.
        merge: ValueId,
    },
    /// One body behind a device-side mask bit — an independent-presence axis
    /// with no exclusive sibling to switch against.
    If {
        /// The predicate the dispatch kernel evaluates against the descriptor.
        cond: Cond,
    },
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
    /// The half-open run of `Plan::nodes` this region covers.
    pub nodes: Range<u32>,
    /// The classes that run these nodes. Every node in the run has this exact
    /// mask — that equality is what defines the run — so a fire's row count
    /// for the region is one number, which is what the descriptor carries.
    ///
    /// An EMPTY mask is a region no class demands. `resolve_classes` reports
    /// those as `Classes::dead`; they stay here so that node indices keep
    /// meaning what they mean, and a shipped plan has none
    /// (`model/tests/every_class_resolves_every_merge.rs` says so).
    pub mask: ClassSet,
    /// Host prepare work, or graph body.
    pub phase: Phase,
    /// How it enters the graph. Always [`Lowering::AlwaysLaunch`] today.
    pub lowering: Lowering,
    /// Which stream it is recorded on. Always 0 today — P6 is the pass that
    /// forks a dep DAG over capture regions into more than one.
    pub stream: u32,
    /// What fraction of the device this region wants, if the profile has an
    /// opinion. `None` today: SM partition is capture-baked, so a variant
    /// multiplies bodies, and v1 ships the hint rather than the partition
    /// (decision #14).
    pub sm_hint: Option<u32>,
    /// This run contains a `Collective`-family node.
    ///
    /// P0 RECORDS IT AND P3 IS BOUND BY IT. A collective takes any `cond` it
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
/// ADDITIVE, AND [`Identity`](LayoutOrder::Identity) IS STILL A TRUE ANSWER.
/// It is what a plan gets when P4 declines to seriate it — more classes than a
/// `u8` names, or none at all — and it is correct for every plan, since a
/// consumer whose classes are not an interval simply runs as more than one
/// launch. It is no longer what `compile` returns for an ordinary plan.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum LayoutOrder {
    /// Lanes fire in the order the engine assembled them: classes ascending.
    #[default]
    Identity,
    /// Every ordering under which the seated consumers are intervals, with the
    /// tree's canonical frontier as v1's pick.
    Seriated(PqTree),
}

impl LayoutOrder {
    /// The order the classes PRESENT in one fire are seriated in.
    ///
    /// THE ONE THING THE FIRE PATH ASKS. A fire has rows for some subset of
    /// the plan's classes — which subset is the batch the engine happened to
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
    /// signature is what a driver's fire path compiles against, and it is not
    /// going to move under it.
    #[must_use]
    pub fn class_order(&self, present: &ClassSet, prev: Option<&[u8]>) -> Vec<u8> {
        // v1's whole use of last fire's order. See above.
        let _ = prev;
        match self {
            LayoutOrder::Identity => present.iter().map(|class| class as u8).collect(),
            LayoutOrder::Seriated(tree) => tree
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
    /// [`class_order`](LayoutOrder::class_order) cannot answer: "is the order
    /// I am already bound to still legal?". `None` is [`Identity`](LayoutOrder::Identity),
    /// which promises nothing and therefore admits everything.
    #[must_use]
    pub fn tree(&self) -> Option<&PqTree> {
        match self {
            LayoutOrder::Identity => None,
            LayoutOrder::Seriated(tree) => Some(tree),
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
/// EMPTY IS THE GOOD CASE, and it is what the whole catalog bakes to today:
/// every consumer's class set is an interval of the order P4 found, so nobody
/// is owed anything. A row here names a consumer the C1P instance could not
/// seat — see [`crate::layout`] for which one gets withdrawn and why.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FallbackTable {
    /// One row per (node, bucket range) that needs one.
    pub rows: Vec<FallbackRow>,
}

/// One row of the [`FallbackTable`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FallbackRow {
    /// Index into `Plan::nodes`.
    pub node: u32,
    /// The half-open range of `Budgets::buckets` indices this answer covers.
    /// A deployment that declared no bucket lattice has ONE implicit bucket at
    /// `Budgets::max_tokens`, and a row over it reads `0..1`.
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
