//! Shared CUDA/Metal vocabulary for what moves in a recorded launch as the
//! composition changes: the fitted [`Law`] forms and the [`Refuse`] reasons
//! a component cannot be stated as one.

pub mod fit;

use std::fmt;

/// One direction the composition can actually be moved along (not a raw
/// descriptor coordinate — `(rows, lanes)` per class are not always
/// independently reachable).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Axis {
    /// What one step of this direction does, in words.
    pub name: String,
    /// What one step does to each class's `(rows, lanes)`, for the reader
    /// and for [`Recipe`], which inverts exactly this.
    pub step: Vec<(i32, i32)>,
}

impl Axis {
    /// A direction, named.
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

/// Turns a fire's class table into the coordinate a law is written against:
/// `coord[k] = konst[k] + sum_c(rows[k][c]*classes[c].rows + lanes[k][c]*classes[c].lanes)`.
/// [`fit::invert`] solves this over the integers; a non-integral inverse is
/// refused rather than rounded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recipe {
    /// The value at an empty class table.
    pub konst: i128,
    /// One coefficient per class, over that class's rows.
    pub rows: Vec<i128>,
    /// One coefficient per class, over that class's lanes.
    pub lanes: Vec<i128>,
}

impl Recipe {
    /// This direction's coordinate at a class table.
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

/// A descriptor slot a [`Law::Slot`] reads its number out of. Opaque: which
/// table the slot indexes is the frame plane's business, this type just names
/// one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SlotId(pub u32);

impl fmt::Display for SlotId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "slot[{}]", self.0)
    }
}

/// Where in a recorded launch a law lives (Metal's `Grid` is total threads,
/// CUDA's is blocks — units belong to the recorder).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum At {
    /// The entrypoint itself — an arm switch, in the census's language.
    Entry,
    /// A grid axis, `0..3`.
    Grid(u8),
    /// A block (Metal: threadgroup) axis, `0..3`.
    Block(u8),
    /// Dynamic shared memory, in bytes.
    Shared,
    /// The `word`-th aligned eight-byte word of argument `at`. A scalar or
    /// pointer is one word; a by-value block is as many as it is wide.
    Arg {
        /// Which argument, by position in the launch's ABI block.
        at: u16,
        /// Which eight-byte word inside it.
        word: u16,
    },
    /// The argument block's own shape moved (different count/offset/width
    /// for the same entrypoint) — not a value a rebind can carry.
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

/// What one component of one recorded launch is, as a function of the
/// composition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Law {
    /// The same number in every composition and every size.
    Const(i128),
    /// `base + Σ slope[k] · coord[k]` over the probe basis. The slopes that
    /// are zero are kept, so a law's shape says which directions it reads
    /// without a second table.
    Affine {
        /// The value at the origin of the fit's coordinates.
        base: i128,
        /// One slope per direction, in the recorder's own [`Axis`] order.
        slope: Vec<i128>,
    },
    /// `mul · ⌈(α·rows + β) / div⌉`, where `rows` is the window's own row
    /// count rather than a probe coordinate (the tiling law, e.g.
    /// `rows.div_ceil(SDPA_TILE)`).
    Ceil {
        /// The scale outside the ceiling: `TILE_GROUP[1]` and its kin.
        mul: i128,
        /// The numerator's slope over the window's rows.
        alpha: i128,
        /// The numerator's offset.
        beta: i128,
        /// The tile.
        div: i128,
    },
    /// Not a function of the coordinates: read per fire out of the named
    /// descriptor slot. Never produced by fitting.
    Slot(SlotId),
}

impl Law {
    /// This law at one point. `None` for [`Law::Slot`] — use [`Law::at_in`]
    /// with the fire's descriptor slots in hand.
    #[must_use]
    pub fn at(&self, coords: &[i128], rows: i128) -> Option<i128> {
        self.at_in(coords, rows, &[])
    }

    /// This law at one point, with the fire's descriptor slots to read a
    /// [`Law::Slot`] out of.
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

    /// Whether the number moves at all.
    #[must_use]
    pub fn varies(&self) -> bool {
        !matches!(self, Law::Const(_))
    }

    /// Which axes this law reads (a slot law reads none).
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

    /// How this law names itself in a census.
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

/// One component of one recorded launch: which launch, which of its numbers,
/// and what that number is. `node` is the recorder's own canonical order and
/// means nothing outside it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Component {
    /// Which recorded launch, in the recorder's canonical order.
    pub node: u32,
    /// Which of its numbers.
    pub at: At,
    /// What that number is, as a function of the composition.
    pub law: Law,
}

impl Component {
    /// A component, stated.
    #[must_use]
    pub fn new(node: u32, at: At, law: Law) -> Component {
        Component { node, at, law }
    }

    /// This component restated for one fire — the [`Patch`] a rebind writes.
    /// `None` exactly where [`Law::at`] is.
    #[must_use]
    pub fn patch(&self, coords: &[i128], rows: i128) -> Option<Patch> {
        self.patch_in(coords, rows, &[])
    }

    /// This component restated for one fire, with the fire's descriptor slots
    /// in hand.
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

/// One component of one launch, restated for one fire: the concrete number.
/// Purely derived from a [`Component`]'s law — no handle, no device anything.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Patch {
    /// Which recorded launch, in the recorder's canonical order.
    pub node: u32,
    /// Which of its numbers.
    pub at: At,
    /// What to write there.
    pub value: i128,
}

impl fmt::Display for Patch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}.{} := {}", self.node, self.at, self.value)
    }
}

/// Why a recorder cannot state a law for a component. Reasons are named,
/// tallied and printed rather than folded into a silent fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Refuse {
    /// The component could not be read, so it cannot be written either.
    Opaque,
    /// The two recordings cannot be aligned truthfully: some launch of one is
    /// only a guess about which launch of the other it is.
    Ambiguous,
    /// The component moved and no law in the language predicts it.
    Unaffine,
    /// The two recordings are not one template at all — a different number of
    /// launches, or a different argument arity or kind at one.
    Unstructured,
    /// The recorder could read the components and state the laws, but the
    /// device refused the restatement itself.
    Unwritable,
}

impl Refuse {
    /// How this refusal names itself in a tally.
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

/// One refusal: the reason, and a human-readable sentence saying which
/// component and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    /// Which refusal this is.
    pub reason: Refuse,
    /// What to tell the reader.
    pub why: String,
}

impl Refusal {
    /// A refusal, spelled out.
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
