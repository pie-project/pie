//! Why a plan was refused. One enum, and the doctrine behind it is the
//! rewrite's: **no silent fallback**. A load that cannot be baked is a load
//! that does not happen, named at the door, rather than a graph that is
//! quietly missing a kernel and produces numbers anyway.

use std::fmt::{self, Display, Formatter};

use model_ir::{ClassFault, Plan, ValueId};

/// The reason `compile` would not bake this plan.
///
/// EVERY VARIANT NAMES A PLACE. A refusal a reader cannot act on is a crash
/// with better manners, so each one carries the value or the number that
/// caused it; [`say`](Refusal::say) spells the class faults' fact words at the
/// plan's own width, which is the half a `Display` without the plan cannot do.
#[derive(Debug, Clone, PartialEq)]
pub enum Refusal {
    /// More facts than the class sweep is a sweep over. `resolve_classes`
    /// asserts this ceiling; the compiler is a front door and answers it as a
    /// refusal instead, because a panic in a load path takes the process with
    /// it.
    TooManyFacts {
        /// How many fact bits the plan's guards reach. The ceiling is 20.
        facts: usize,
    },
    /// A budget that cannot describe a fire: no lanes, no tokens, fewer rows
    /// than lanes (a lane carries at least one row), or a bucket lattice that
    /// does not ascend or reaches past the ceiling.
    Budget {
        /// The field, and what is wrong with it.
        what: &'static str,
    },
    /// A device profile that describes no device.
    Profile {
        /// The field, and what is wrong with it.
        what: &'static str,
    },
    /// The class sweep found merges that do not resolve — a hole no arm
    /// writes, or two arms writing one row range. ALL of them, because a
    /// coverage hole is usually one authoring mistake seen from several
    /// classes at once.
    Classes(Vec<ClassFault>),
    /// A value the arena cannot cut a rectangle out of.
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
    Mismatch {
        /// Which sharing rule was being applied.
        kind: Share,
        /// The value that owns the column.
        holds: ValueId,
        /// The value that was to share it.
        shares: ValueId,
    },
}

/// Which of the IR's two column-sharing rules a [`Refusal::Mismatch`] is
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

impl Refusal {
    /// The refusal as a sentence, with class faults' fact words spelled at the
    /// plan's own width.
    ///
    /// `resolve_classes` hands back faults with no plan attached — a
    /// `ClassFault` is small, comparable data — and a bare `0b1` does not say
    /// which of the plan's other bits are off, so the plan comes back in here.
    /// Every other variant says the same thing `Display` does.
    #[must_use]
    pub fn say(&self, plan: &Plan) -> String {
        match self {
            Refusal::Classes(faults) => faults
                .iter()
                .map(|fault| fault.say(plan))
                .collect::<Vec<_>>()
                .join("\n"),
            other => other.to_string(),
        }
    }
}

impl Display for Refusal {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Refusal::TooManyFacts { facts } => write!(
                f,
                "the plan guards on {facts} fact bits; the class sweep is 2^F \
                 and stops being a sweep past 20",
            ),
            Refusal::Budget { what } => write!(f, "the budgets {what}"),
            Refusal::Profile { what } => write!(f, "the device profile {what}"),
            Refusal::Classes(faults) => {
                write!(f, "{} merges do not resolve:", faults.len())?;
                for fault in faults {
                    write!(f, "\n  {fault}")?;
                }
                Ok(())
            }
            Refusal::Unrectangled { value, why } => {
                let why = match why {
                    Unrectangled::SymbolicWidth => {
                        "its shape is symbolic past the leading dim, and the \
                         row algebra is one symbol wide"
                    }
                    Unrectangled::PackedElement => {
                        "its element is a packed storage plane with no \
                         per-element byte size"
                    }
                };
                write!(f, "v{} has no arena rectangle: {why}", value.0)
            }
            Refusal::Mismatch {
                kind,
                holds,
                shares,
            } => {
                let rule = match kind {
                    Share::InPlace => "the op writes through it in place",
                    Share::MergeArm => "it is an arm of that merge",
                };
                write!(
                    f,
                    "v{} must share v{}'s column — {rule} — and the two are \
                     declared at different sizes",
                    shares.0, holds.0,
                )
            }
        }
    }
}

impl std::error::Error for Refusal {}
