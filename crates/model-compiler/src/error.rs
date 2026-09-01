//! Why a plan was refused. One enum, and the doctrine behind it is the
//! rewrite's: **no silent fallback**. A load that cannot be baked is a load
//! that does not happen, named at the door, rather than a graph that is
//! quietly missing a kernel and produces numbers anyway.

use model_ir::{ClassFault, Trace, ValueId};

/// The reason `compile` would not bake this plan.
///
/// EVERY VARIANT NAMES A PLACE. A refusal a reader cannot act on is a crash
/// with better manners, so each one carries the value or the number that
/// caused it; [`say`](Error::say) spells the class faults' fact words at the
/// plan's own width, which is the half a `Display` without the plan cannot do.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    /// More facts than the class sweep is a sweep over. `resolve_classes`
    /// asserts this ceiling; the compiler is a front door and answers it as a
    /// refusal instead, because a panic in a load path takes the process with
    /// it.
    #[error(
        "the plan guards on {facts} fact bits; the class sweep is 2^F \
         and stops being a sweep past 20"
    )]
    TooManyFacts {
        /// How many fact bits the plan's guards reach. The ceiling is 20.
        facts: usize,
    },
    /// A budget that cannot describe a fire: no lanes, no tokens, fewer rows
    /// than lanes (a lane carries at least one row), or a bucket lattice that
    /// does not ascend or reaches past the ceiling.
    #[error("the budgets {what}")]
    Budget {
        /// The field, and what is wrong with it.
        what: &'static str,
    },
    /// A deployment that wants more adapter banks than the model text seats.
    ///
    /// **CAPACITY IS A SHAPE, AND A BUDGET IS NOT AN ADMISSION CAP**
    /// (design §8, decision 17). The bank's first axis is how many adapters
    /// the plan reserved room for, `Budget::max_adapters` is how many the
    /// deployment intends to register, and the two have to agree at the LOAD:
    /// discovering the disagreement at the two-hundredth registration would
    /// make the capacity exactly the admission cap decision 17 refuses to
    /// build. Both numbers are in the refusal so a reader knows which one to
    /// change.
    #[error("{}", adapter_capacity(*.asked, *.seated))]
    AdapterCapacity {
        /// What `Budget::max_adapters` asked for.
        asked: u32,
        /// The smallest capacity any bank of this plan declares — an id must
        /// fit every site it will be written into. `0` when the plan declares
        /// no bank at all.
        seated: u64,
    },
    /// A device profile that describes no device.
    #[error("the device profile {what}")]
    Profile {
        /// The field, and what is wrong with it.
        what: &'static str,
    },
    /// The class sweep found merges that do not resolve — a hole no arm
    /// writes, or two arms writing one row range. ALL of them, because a
    /// coverage hole is usually one authoring mistake seen from several
    /// classes at once.
    #[error("{}", class_faults(.0))]
    Classes(Vec<ClassFault>),
    /// A value the arena cannot cut a rectangle out of.
    #[error("v{} has no arena rectangle: {}", .value.0, unrectangled(.why))]
    Unrectangled {
        /// The value whose declared type could not be sized.
        value: ValueId,
        /// Which of the two ways it could not.
        why: Unrectangled,
    },
    /// Two values the IR says occupy ONE column, declared at different sizes.
    /// The IR states the sharing — a merge's arms write disjoint windows of
    /// the merged column, an in-place result IS the operand it overwrites —
    /// so a size disagreement is not something the carve may paper over: one
    /// of the two writes past the other's end. `check` names the merge half of
    /// this first (`Fault::MergeArmTy` — an arm's type must BE the merge's);
    /// this is the front door's own reading, on a plan that may not have been
    /// through the validator.
    #[error(
        "v{} must share v{}'s column — {} — and the two are \
         declared at different sizes",
        .shares.0,
        .holds.0,
        rule(.kind)
    )]
    Mismatch {
        /// Which sharing rule was being applied.
        kind: Share,
        /// The value that owns the column.
        holds: ValueId,
        /// The value that was to share it.
        shares: ValueId,
    },
    /// A `Struct` value — an attention SCHEDULE — built in one window and
    /// read in another.
    ///
    /// **THE AUTHORING NET FOR BUILD LOG 20's SECOND BLOCKER.** A schedule is
    /// not a row-shaped table that slices; it is a carving. How many requests
    /// it batches, where each one's query rows begin, how its work items split
    /// the kv and how much of its grant it padded to are all fixed when the
    /// builder walks the window it was dispatched in. Demand then narrows the
    /// prepare node to the UNION of the classes reading it (build log 7),
    /// which is the right answer for a shared tensor and the wrong SHAPE for
    /// two windowed readers: an arm standing in a narrower window hands the
    /// schedule its OWN rebased boundaries, and every work item past the first
    /// request indexes a `qo_indptr` that has already ended. Nothing faults on
    /// the device — the reads land in whatever follows a `[lanes + 1]` vector
    /// — and the answer is wrong logits.
    ///
    /// So it is refused at the BAKE, where the sentence can still name the
    /// model text that has to change: mint a second plan value for the second
    /// reader. Equality rather than containment, deliberately — a schedule
    /// built over FEWER classes than its reader is the same failure from the
    /// other side.
    #[error(
        "v{} is an attention schedule carved over classes {planned:?} and read \
         by node {node}, which runs in classes {consumed:?}. A schedule is a \
         carving, not a table that slices: the reader hands it boundaries \
         rebased to ITS window and the work items index past their end. The \
         model text mints a second plan value for the second reader",
        .value.0
    )]
    Straddled {
        /// The schedule value.
        value: ValueId,
        /// The node that reads it from another window.
        node: u32,
        /// The classes the schedule was carved over.
        planned: Vec<usize>,
        /// The classes the reader runs in.
        consumed: Vec<usize>,
    },
    /// A prepare node that reads an activation a capture node computed.
    ///
    /// **THE PRECONDITION OF P5's HOIST, ASKED OUT LOUD** (design §5). Prepare
    /// work is host work — a `std::vector`, a work estimate, a pageable upload
    /// — and host work inside `cudaStreamBeginCapture` is either refused by
    /// the engine or, worse, recorded as nothing. So the prepare half of the
    /// template runs BEFORE the capture half, whole, and `region::hoist` moves
    /// it there; `engine::fire::walk`'s rule 3 is the same claim asked of the
    /// artifact.
    ///
    /// The move is sound exactly while nothing in the prepare half needs a
    /// number the graph has not computed yet. A `Ty::Struct` definer is
    /// supposed to be a plan build over CACHE GEOMETRY and RUNTIME INPUTS —
    /// the indptr, the page indices, the last-page length — every one of which
    /// the engine binds before the fire begins, and every one of which is a
    /// `Def::Input`. One that reads an activation instead is a plan build that
    /// cannot be hoisted and a capture that cannot contain it: there is no
    /// instant that is both after the activation and before the graph.
    ///
    /// Refused rather than half-hoisted, because the alternative is a template
    /// whose two halves each look fine and whose composition reads a slot
    /// before it was written. The model text's answer is to compute the number
    /// the plan build wants on the host, as a runtime input, rather than in
    /// the graph.
    #[error(
        "prepare node {node} reads v{}, which capture node {produced_by} computes. \
         A prepare op is host work that a captured graph cannot contain, so P5 \
         hoists the whole prepare half in front of the capture half — and there \
         is no instant that is both after an activation and before the graph. \
         The model text computes that number as a runtime input instead",
        .value.0
    )]
    HoistBlocked {
        /// The prepare node — the `Ty::Struct` definer.
        node: u32,
        /// The value it reads that a capture node produces.
        value: ValueId,
        /// The capture node that produces it.
        produced_by: u32,
    },
    /// A plan that states a row axis the budgets size no ceiling for.
    ///
    /// **EVERY SYMBOLIC DIM IS SIZED IN THE BUDGET AND NOWHERE ELSE**, so a
    /// `Dim::Patches` against a `Budget` with no `PatchLadder` is a rectangle
    /// with no height. The alternative is worse than a refusal and quieter: a
    /// carve at zero rows places every tower rectangle at the same offset,
    /// nothing clashes because nothing is live, and the graph computes over
    /// somebody else's bytes. So it is named at the door, with the field the
    /// deployment has to fill in.
    #[error(
        "the plan states {} rows and the budgets size no {} ceiling — a deployment that serves \
         this model declares the axis's ladder",
        .axis.name(),
        .axis.name()
    )]
    Unsized {
        /// The row axis the plan states.
        axis: model_ir::RowAxis,
    },
    /// Capture units that alternate down the record script.
    ///
    /// **A CAPTURE UNIT IS AN EXEC, AND AN EXEC IS ONE CONTIGUOUS STRETCH.**
    /// The fire launches one exec per unit, chained on one stream; a unit
    /// whose regions resume after another unit's have run is not one exec but
    /// several, and picking an order that would make it one is a scheduling
    /// decision this compiler has no dependence graph to make (the same reason
    /// `coalesce` keeps program order). The shape that works is the shape a
    /// declared tower already has: the tower stated before the trunk that
    /// reads its output.
    #[error(
        "the {} capture unit (unit {unit}) resumes at nodes {}..{} after another unit has run. A \
         unit is one exec and an exec is one contiguous stretch of the script; the model text \
         states the tower before the trunk that reads it",
        .axis.name(),
        .nodes.start,
        .nodes.end
    )]
    UnitsInterleave {
        /// The axis whose unit resumes.
        axis: model_ir::RowAxis,
        /// Its index into `CompiledModel::units`.
        unit: u32,
        /// The node range of the region that resumes it.
        nodes: core::ops::Range<u32>,
    },
}

/// Which of the IR's two column-sharing rules a [`Error::Mismatch`] is
/// about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Share {
    /// `Operands::aliases` — the result an op writes through its operand.
    InPlace,
    /// `Def::Merge` — an arm writing its window of the merged column.
    MergeArm,
}

/// The two ways a declared type is not a rectangle of the arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unrectangled {
    /// A symbolic `Dim` past the leading one. The row algebra is ONE symbol
    /// wide on purpose (`arena::RowExpr`): a value is `rows x width`, its rows
    /// scale with the fire and its width does not. A shape that varies in two
    /// directions at once has no static offset — and it cannot reach a
    /// validated plan either, since `check` faults it as a `SymbolicAxis`.
    SymbolicWidth,
    /// A packed element — `Mxfp4`'s 32 codes to 16 bytes, `Fp4`'s half — has
    /// no per-element byte size to multiply a width by. These name WEIGHT
    /// planes and kv-page quant schemes, neither of which is an arena
    /// rectangle, so reaching here means an op declared an activation in a
    /// storage element.
    PackedElement,
}

impl Error {
    /// The refusal as a sentence, with class faults' fact words spelled at the
    /// plan's own width.
    ///
    /// `resolve_classes` hands back faults with no plan attached — a
    /// `ClassFault` is small, comparable data — and a bare `0b1` does not say
    /// which of the plan's other bits are off, so the plan comes back in here.
    /// Every other variant says the same thing `Display` does.
    #[must_use]
    pub fn say(&self, trace: &Trace) -> String {
        match self {
            Error::Classes(faults) => faults
                .iter()
                .map(|fault| fault.say(trace))
                .collect::<Vec<_>>()
                .join("\n"),
            other => other.to_string(),
        }
    }
}

/// [`Error::AdapterCapacity`]'s sentence — two of them, because `seated == 0`
/// is a different authoring mistake (no bank at all) than a bank that is too
/// narrow, and one format string cannot branch.
fn adapter_capacity(asked: u32, seated: u64) -> String {
    if seated == 0 {
        format!(
            "the budgets ask to register {asked} adapters and this plan \
             declares no bank at all: a bank is a weight the model text \
             marked `registered`, and capacity is its leading axis"
        )
    } else {
        format!(
            "the budgets ask to register {asked} adapters and the narrowest \
             bank of this plan seats {seated}; capacity is a shape the \
             model text declares, so one of the two numbers has to move \
             before the load and not at a registration"
        )
    }
}

/// [`Error::Classes`]'s sentence: the count, then every fault on its own line.
fn class_faults(faults: &[ClassFault]) -> String {
    let mut text = format!("{} merges do not resolve:", faults.len());
    for fault in faults {
        text.push_str(&format!("\n  {fault}"));
    }
    text
}

/// [`Error::Unrectangled`]'s reason clause.
fn unrectangled(why: &Unrectangled) -> &'static str {
    match why {
        Unrectangled::SymbolicWidth => {
            "its shape is symbolic past the leading dim, and the \
             row algebra is one symbol wide"
        }
        Unrectangled::PackedElement => {
            "its element is a packed storage plane with no \
             per-element byte size"
        }
    }
}

/// [`Error::Mismatch`]'s sharing-rule clause.
fn rule(kind: &Share) -> &'static str {
    match kind {
        Share::InPlace => "the op writes through it in place",
        Share::MergeArm => "it is an arm of that merge",
    }
}
