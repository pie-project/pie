//! Why a plan was refused. One enum, and the doctrine behind it is the
//! rewrite's: **no silent fallback**. A load that cannot be baked is a load
//! that does not happen, named at the door, rather than a graph that is
//! quietly missing a kernel and produces numbers anyway.

use model_ir::{ClassFault, Trace, ValueId};

/// The reason `compile` would not bake this plan. Every variant names a
/// place: each carries the value or number that caused it; [`say`](Error::say)
/// spells the class faults' fact words at the plan's own width.
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
    /// More classes than a class order can name (`crate::MAX_CLASSES`).
    #[error("the plan's guards realize {classes} classes and a class order names at most {max}", max = crate::MAX_CLASSES)]
    TooManyClasses { classes: usize },
    /// An op whose outputs stand on two row axes.
    #[error("node {node} writes outputs on two row axes, and a region counts rows on one")]
    TwoAxes { node: u32 },
    /// A class table whose node mask is not parallel to the trace.
    #[error("the class table masks {masks} nodes and the trace has {nodes}")]
    MaskLength { masks: usize, nodes: usize },
    /// An in-place alias whose operand is not an arena rectangle: the op
    /// would write through the operand and readers of the result would read
    /// an arena slot nothing wrote.
    #[error("v{} is written in place through v{}, which is not an arena rectangle", .shares.0, .holds.0)]
    AliasOutside { holds: ValueId, shares: ValueId },
    /// A budget that cannot describe a fire: no lanes, no tokens, fewer rows
    /// than lanes (a lane carries at least one row), or a bucket lattice that
    /// does not ascend or reaches past the ceiling.
    #[error("the budgets {what}")]
    Budget {
        /// The field, and what is wrong with it.
        what: &'static str,
    },
    /// A deployment that wants more adapter banks than the model text
    /// seats. Capacity is a shape, not an admission cap: the bank's first
    /// axis is how many adapters the plan reserved room for, and
    /// `Budget::max_adapters` is how many the deployment intends to
    /// register — the two must agree at load, not discovered later at a
    /// registration.
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
    /// writes, or two arms writing one row range. All of them, since a
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
    /// Two values the IR says occupy one column, declared at different
    /// sizes: one of the two writes past the other's end. `check` names
    /// the merge half of this first (`Fault::MergeArmTy`); this is the
    /// front door's own reading, on a plan that may not have been through
    /// the validator.
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
    /// A `Struct` value — an attention schedule — built in one window and
    /// read in another. A schedule is not a row-shaped table that slices;
    /// it is a carving fixed when the builder walks its own dispatch
    /// window, so a reader standing in a narrower window would index past
    /// its own boundaries and get wrong logits with nothing faulting on
    /// the device. Refused at the bake, where the sentence can still name
    /// the model text that has to change: mint a second plan value for
    /// the second reader.
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
    /// Prepare work is host work, and host work inside a captured graph is
    /// either refused by the engine or recorded as nothing — so the whole
    /// prepare half runs before the capture half (`region::hoist`), which
    /// is sound only while nothing in it needs a number the graph hasn't
    /// computed yet. A `Ty::Struct` definer is supposed to build its plan
    /// over cache geometry and runtime inputs, all `Def::Input`; one that
    /// reads an activation instead can't be hoisted, since there's no
    /// instant both after the activation and before the graph. The model
    /// text's answer is to compute that number as a runtime input instead.
    #[error(
        "prepare node {node} reads v{}, which capture node {produced_by} computes. \
         A prepare op is host work that a captured graph cannot contain, so the \
         hoist pass runs the whole prepare half in front of the capture half — \
         and there is no instant that is both after an activation and before \
         the graph. The model text computes that number as a runtime input \
         instead",
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
    /// Every symbolic dim is sized in the budget and nowhere else, so a
    /// `Dim::Patches` against a `Budget` with no `PatchLadder` is a
    /// rectangle with no height. Refused rather than carved at zero rows,
    /// which would place every tower rectangle at the same offset and let
    /// the graph compute over somebody else's bytes.
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
    /// Capture units that alternate down the record script. A capture unit
    /// is an exec, and an exec is one contiguous stretch: a unit whose
    /// regions resume after another unit's have run is not one exec but
    /// several, and reordering to fix it is a scheduling decision this
    /// compiler has no dependence graph to make. The shape that works is
    /// the shape a declared tower already has: stated before the trunk
    /// that reads its output.
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
    /// A symbolic `Dim` past the leading one. The row algebra is one
    /// symbol wide on purpose: a value is `rows x width`, its rows scale
    /// with the fire and its width does not, so a shape varying in two
    /// directions has no static offset.
    SymbolicWidth,
    /// A packed element (e.g. `Mxfp4`'s 32 codes to 16 bytes) has no
    /// per-element byte size to multiply a width by; these name weight
    /// planes and kv-page quant schemes, never an arena rectangle.
    PackedElement,
    /// A rectangle whose byte count overflows `u64`.
    Oversize,
}

impl Error {
    /// The refusal as a sentence, with class faults' fact words spelled at
    /// the plan's own width. `resolve_classes` hands back faults with no
    /// plan attached, so the plan comes back in here. Every other variant
    /// says the same thing `Display` does.
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
        Unrectangled::Oversize => "its byte count overflows u64",
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
