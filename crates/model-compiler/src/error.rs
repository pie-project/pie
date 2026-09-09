use model_ir::{ClassFault, Trace, ValueId};

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    #[error(
        "the plan guards on {facts} fact bits; the class sweep is 2^F \
         and stops being a sweep past 20"
    )]
    TooManyFacts { facts: usize },
    #[error("the plan's guards realize {classes} classes and a class order names at most {max}", max = crate::MAX_CLASSES)]
    TooManyClasses { classes: usize },
    #[error("node {node} writes outputs on two row axes, and a region counts rows on one")]
    TwoAxes { node: u32 },
    #[error("the class table masks {masks} nodes and the trace has {nodes}")]
    MaskLength { masks: usize, nodes: usize },
    #[error("v{} is written in place through v{}, which is not an arena rectangle", .shares.0, .holds.0)]
    AliasOutside { holds: ValueId, shares: ValueId },
    #[error("the budgets {what}")]
    Budget { what: &'static str },
    #[error("{}", adapter_capacity(*.asked, *.seated))]
    AdapterCapacity { asked: u32, seated: u64 },
    #[error("the device profile {what}")]
    Profile { what: &'static str },
    #[error("{}", class_faults(.0))]
    Classes(Vec<ClassFault>),
    #[error("v{} has no arena rectangle: {}", .value.0, unrectangled(.why))]
    Unrectangled { value: ValueId, why: Unrectangled },
    #[error(
        "v{} must share v{}'s column — {} — and the two are \
         declared at different sizes",
        .shares.0,
        .holds.0,
        rule(.kind)
    )]
    Mismatch {
        kind: Share,
        holds: ValueId,
        shares: ValueId,
    },
    #[error(
        "v{} is an attention schedule carved over classes {planned:?} and read \
         by node {node}, which runs in classes {consumed:?}. A schedule is a \
         carving, not a table that slices: the reader hands it boundaries \
         rebased to ITS window and the work items index past their end. The \
         model text mints a second plan value for the second reader",
        .value.0
    )]
    Straddled {
        value: ValueId,
        node: u32,
        planned: Vec<usize>,
        consumed: Vec<usize>,
    },
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
        node: u32,
        value: ValueId,
        produced_by: u32,
    },
    #[error(
        "the plan states {} rows and the budgets size no {} ceiling — a deployment that serves \
         this model declares the axis's ladder",
        .axis.name(),
        .axis.name()
    )]
    Unsized { axis: model_ir::RowAxis },
    #[error(
        "the {} capture unit (unit {unit}) resumes at nodes {}..{} after another unit has run. A \
         unit is one exec and an exec is one contiguous stretch of the script; the model text \
         states the tower before the trunk that reads it",
        .axis.name(),
        .nodes.start,
        .nodes.end
    )]
    UnitsInterleave {
        axis: model_ir::RowAxis,
        unit: u32,
        nodes: core::ops::Range<u32>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Share {
    InPlace,
    MergeArm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unrectangled {
    SymbolicWidth,
    PackedElement,
    Oversize,
}

impl Error {
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

fn class_faults(faults: &[ClassFault]) -> String {
    let mut text = format!("{} merges do not resolve:", faults.len());
    for fault in faults {
        text.push_str(&format!("\n  {fault}"));
    }
    text
}

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

fn rule(kind: &Share) -> &'static str {
    match kind {
        Share::InPlace => "the op writes through it in place",
        Share::MergeArm => "it is an arm of that merge",
    }
}
