pub mod fit;

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Axis {
    pub name: String,
    pub step: Vec<(i32, i32)>,
}

impl Axis {
    #[must_use]
    pub fn new(name: impl Into<String>, step: Vec<(i32, i32)>) -> Axis {
        Axis {
            name: name.into(),
            step,
        }
    }
}

impl fmt::Display for Axis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recipe {
    pub konst: i128,
    pub rows: Vec<i128>,
    pub lanes: Vec<i128>,
}

impl Recipe {
    #[must_use]
    pub fn at(&self, classes: &[(u32, u32)]) -> i128 {
        let mut sum = self.konst;
        for (c, (rows, lanes)) in classes.iter().enumerate() {
            sum += self.rows.get(c).copied().unwrap_or(0) * i128::from(*rows);
            sum += self.lanes.get(c).copied().unwrap_or(0) * i128::from(*lanes);
        }
        sum
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SlotId(pub u32);

impl fmt::Display for SlotId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "slot[{}]", self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum At {
    Entry,
    Grid(u8),
    Block(u8),
    Shared,
    Arg { at: u16, word: u16 },
    Shape,
}

impl fmt::Display for At {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            At::Entry => f.write_str("entry"),
            At::Grid(axis) => write!(f, "grid.{axis}"),
            At::Block(axis) => write!(f, "block.{axis}"),
            At::Shared => f.write_str("shared"),
            At::Arg { at, word: 0 } => write!(f, "arg[{at}]"),
            At::Arg { at, word } => write!(f, "arg[{at}].w{word}"),
            At::Shape => f.write_str("shape"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Law {
    Const(i128),
    Affine {
        base: i128,
        slope: Vec<i128>,
    },
    Ceil {
        mul: i128,
        alpha: i128,
        beta: i128,
        div: i128,
    },
    Slot(SlotId),
}

impl Law {
    #[must_use]
    pub fn at(&self, coords: &[i128], rows: i128) -> Option<i128> {
        self.at_in(coords, rows, &[])
    }

    #[must_use]
    pub fn at_in(&self, coords: &[i128], rows: i128, slots: &[i128]) -> Option<i128> {
        match self {
            Law::Const(v) => Some(*v),
            Law::Affine { base, slope } => Some(
                slope
                    .iter()
                    .zip(coords)
                    .fold(*base, |sum, (b, x)| sum + b * x),
            ),
            Law::Ceil {
                mul,
                alpha,
                beta,
                div,
            } => {
                let numerator = alpha * rows + beta;
                Some(
                    mul * numerator.div_euclid(*div)
                        + mul * i128::from(numerator.rem_euclid(*div) != 0),
                )
            }
            Law::Slot(SlotId(id)) => slots.get(*id as usize).copied(),
        }
    }

    #[must_use]
    pub fn varies(&self) -> bool {
        !matches!(self, Law::Const(_))
    }

    #[must_use]
    pub fn reads(&self) -> Vec<usize> {
        match self {
            Law::Const(_) | Law::Ceil { .. } | Law::Slot(_) => Vec::new(),
            Law::Affine { slope, .. } => slope
                .iter()
                .enumerate()
                .filter_map(|(k, b)| (*b != 0).then_some(k))
                .collect(),
        }
    }

    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Law::Const(_) => "const",
            Law::Affine { .. } => "affine",
            Law::Ceil { .. } => "ceil",
            Law::Slot(_) => "slot",
        }
    }
}

impl fmt::Display for Law {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Law::Const(v) => write!(f, "{v}"),
            Law::Affine { base, slope } => {
                write!(f, "{base}")?;
                for (k, b) in slope.iter().enumerate() {
                    if *b != 0 {
                        write!(f, " + {b}·x{k}")?;
                    }
                }
                Ok(())
            }
            Law::Ceil {
                mul,
                alpha,
                beta,
                div,
            } => {
                if *mul == 1 {
                    write!(f, "ceil(({alpha}·rows + {beta}) / {div})")
                } else {
                    write!(f, "{mul}·ceil(({alpha}·rows + {beta}) / {div})")
                }
            }
            Law::Slot(id) => write!(f, "{id}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Component {
    pub node: u32,
    pub at: At,
    pub law: Law,
}

impl Component {
    #[must_use]
    pub fn new(node: u32, at: At, law: Law) -> Component {
        Component { node, at, law }
    }

    #[must_use]
    pub fn patch(&self, coords: &[i128], rows: i128) -> Option<Patch> {
        self.patch_in(coords, rows, &[])
    }

    #[must_use]
    pub fn patch_in(&self, coords: &[i128], rows: i128, slots: &[i128]) -> Option<Patch> {
        Some(Patch {
            node: self.node,
            at: self.at,
            value: self.law.at_in(coords, rows, slots)?,
        })
    }
}

impl fmt::Display for Component {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}.{} = {}", self.node, self.at, self.law)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Patch {
    pub node: u32,
    pub at: At,
    pub value: i128,
}

impl fmt::Display for Patch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}.{} := {}", self.node, self.at, self.value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Refuse {
    Opaque,
    Ambiguous,
    Unaffine,
    Unstructured,
    Unwritable,
}

impl Refuse {
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Refuse::Opaque => "opaque",
            Refuse::Ambiguous => "ambiguous",
            Refuse::Unaffine => "unaffine",
            Refuse::Unstructured => "unstructured",
            Refuse::Unwritable => "unwritable",
        }
    }
}

impl fmt::Display for Refuse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.kind())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    pub reason: Refuse,
    pub why: String,
}

impl Refusal {
    #[must_use]
    pub fn new(reason: Refuse, why: impl Into<String>) -> Refusal {
        Refusal {
            reason,
            why: why.into(),
        }
    }
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.why)
    }
}
