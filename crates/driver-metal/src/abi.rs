//! `DescriptorAbi`, discovered rather than declared (design §2's P8 field,
//! `.wiki/palo/icb.md` §3).
//!
//! Design §2 lists `descriptor: DescriptorAbi` as "the ONE mutable channel
//! into a recorded graph" and never says what is in it, because on the CUDA
//! plane nothing could say: `kernels-cuda` builds its argument bytes inside
//! `ctx.fire`, so which argument is an extent is the entry's private
//! knowledge and build log 10 keyed an exec per `(rows, lanes)` vector
//! instead. `kernels_metal::Encode::fire` hands the shell every argument as
//! a value and every grid axis as a number, so the table is derivable — and
//! this module derives it.
//!
//! # The component language, and why it has exactly three forms
//!
//! ```text
//! Const  v                                   encoded once, never rewritten
//! Affine v = base + Σ slope[k] · coord[k]     the windowed cut, the extent
//! Ceil   v = mul · ⌈(α·rows + β) / div⌉       the TILING law
//! ```
//!
//! The third one is this wave's addition and the shape of it is not a
//! generalisation for its own sake. Build log 30 derived the table for
//! qwen35-d0.8b and six components of 5579 refused: `sdpa_paged_tiled`'s
//! second grid axis, which `kernels_metal::attn::tiled_grid` writes
//! `rows.div_ceil(SDPA_TILE)`. Two more appear the moment the 127
//! arm-switching slots are fitted rather than skipped — `linear::gemm`'s tile
//! arm dispatches `div_ceil(rows, TILE_M) · TILE_GROUP[1]` row tiles. Both
//! are a ceiling over the WINDOW'S ROWS, scaled, and nothing in the catalog
//! is a ceiling over anything else. So the law reads the window's rows rather
//! than the coordinates, which costs one more law per slot ([`SlotAbi::rows`])
//! and buys a form a reader can check against the shader source in one
//! glance.
//!
//! **A component matching neither law is still `Unaffine` by name.** The
//! refusal did not get weaker; it got a second thing to try first.
//!
//! # The method, and why it is honest rather than clever
//!
//! Walk the same template many times against synthetic descriptors,
//! [record](crate::record) each walk, and read the differences:
//!
//! ```text
//! probe    : a base composition, and one LADDER per direction —
//!            base + 1·e_k, base + 2·e_k, ... base + L·e_k
//! check    : a composition no probe visited, held out of every fit
//!
//! for each slot, for each grid axis and each argument:
//!   equal across every sample of the arm   → a CONSTANT, encoded once
//!   moved, and a line fits every sample     → Affine
//!   moved, and a scaled ceiling over the
//!     window's rows fits every sample       → Ceil
//!   moved, and neither                      → Fault::Unaffine { slot, at }
//! ```
//!
//! **Why a ladder and not two points.** A fit through two points always fits.
//! Build log 30's bug was subtler than that: a grid axis written
//! `rows.div_ceil(32)` is FLAT across every step small enough to stay inside
//! one tile, so a two-point probe called it a constant and would have encoded
//! the wrong grid forever. A ladder that crosses a tile boundary is the
//! smallest thing that can see a staircase at all, and it is also what
//! brackets an arm switch to the row it happens at ([`Pick::Rows`]).
//!
//! # Arms: a slot is one ENTRY and not one PIPELINE
//!
//! `kernels_metal::linear::gemm::act_x_wt` takes `dense_gemv_t_bfloat16`
//! below `TILE_M = 32` rows and `dense_gemm_t_bfloat16_bm_32_bn_32` at or
//! above it. 127 of qwen35-d0.8b's 465 slots do that. Build log 30 named them
//! and left them out of the fit; here they are fitted ONE TABLE PER ARM, and
//! which arm runs is [`Pick::Rows`] — a threshold on the window's rows,
//! bracketed to the row by the ladder that crosses it. That is the form the
//! rebind shader can evaluate, which is the whole reason it is a threshold
//! and not a lookup.
//!
//! # The coordinates, and how a live fire finds them
//!
//! The fit is written in a basis of REACHABLE DIRECTIONS ([`Axis`]) rather
//! than in the descriptor's own `(rows, lanes)` per class, because a decode
//! class's word says one token per lane and no batch can move its rows
//! without moving its lanes. That basis is what a probe harness can step
//! along; it is not what a fire carries. So the derivation also inverts it:
//! [`Recipe`] is one linear functional per direction over the class table's
//! own numbers, solved exactly at load and verified at every probe, and
//! [`DescriptorAbi::coords_of`] is what turns a live composition into the
//! coordinates every law is written in. The rebind shader evaluates the same
//! recipe over the same packed bytes.
//!
//! # What the derivation does NOT cover, stated
//!
//! The walk skips a zero-row region's nodes (`driver::fire::walk` rule 1), so
//! a composition with an empty window produces FEWER slots than one without.
//! Every probe point here therefore holds every class, and the derived table
//! is the FULL composition's — which is the point rather than a limitation:
//! design §5's "all compositions live inside it" means the artifact holds
//! every launch and a fire turns the absent ones off. What turns one off is
//! [`SlotAbi::rows`] evaluating to zero, and the ICB is what acts on it.

use std::collections::{BTreeMap, BTreeSet};

use crate::error::{Fault, Result};
use crate::record::{Arg, Point, Recording, Slot};

/// The largest divisor a tiling law is searched for.
///
/// A tile is a compile-time constant of a `kernels-metal` entry and the ones
/// that exist are 32 (`SDPA_TILE`, `TILE_M`, `TILE_N`) and 128
/// (`VECTOR_GROUP`). The ceiling is stated so the search is bounded and so a
/// tile past it is a named miss rather than a silent one.
pub const MAX_TILE: i128 = 512;

/// The largest multiplier on the window's rows a tiling law's numerator is
/// searched for.
pub const MAX_NUMERATOR_SCALE: i128 = 32;

/// One direction the composition can actually be moved along.
///
/// **NOT A COORDINATE OF THE DESCRIPTOR, A DIRECTION IN IT.** The descriptor
/// holds `(rows, lanes)` per class and the two are not always independently
/// reachable — a decode class's word says one token per lane, so every batch
/// that adds a decode row adds a decode lane. A basis of directions the
/// harness can genuinely step along is what a pair of axes would have had to
/// pretend, and the name is what makes the law readable: a slope of 12288 on
/// "a prefill token" means twelve kilobytes of arena per token.
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

impl std::fmt::Display for Axis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name)
    }
}

/// One direction, read back out of a class table.
///
/// **THE INVERSE OF THE BASIS, AND IT IS WHY THE SHADER NEEDS NO WALK.** The
/// laws are written in the probe basis; a fire carries a class table. This is
/// the one linear functional that turns the second into the first:
///
/// ```text
/// coord[k] = konst[k] + Σ_c ( rows[k][c]·classes[c].rows
///                           + lanes[k][c]·classes[c].lanes )
/// ```
///
/// It is SOLVED, not stated: [`derive`] picks a square subsystem of the step
/// matrix, inverts it exactly over the integers, and then verifies the
/// resulting recipe against every probe's own coordinates. A basis whose
/// inverse is not integral is refused by name rather than rounded.
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

/// Where in a slot a law lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum At {
    /// A grid axis: total threads, `0..3`.
    Lane(u8),
    /// A threadgroup axis, `0..3`.
    Group(u8),
    /// An argument index — the byte offset of a buffer binding, or the value
    /// of a scalar.
    Arg(u16),
}

impl std::fmt::Display for At {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            At::Lane(axis) => write!(f, "grid axis {axis}"),
            At::Group(axis) => write!(f, "threadgroup axis {axis}"),
            At::Arg(index) => write!(f, "argument {index}"),
        }
    }
}

/// What one component of one slot is, as a function of the descriptor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Law {
    /// The same number in every composition and every size: encoded into the
    /// ICB once, at load, and never rewritten.
    Const(i128),
    /// `base + Σ slope[k] · coord[k]` over the probe basis. The slopes that
    /// are zero are kept, so a law's shape says which directions it reads
    /// without a second table.
    Affine {
        /// The value at the origin of the fit's coordinates.
        base: i128,
        /// One slope per descriptor axis, in [`DescriptorAbi::axes`] order.
        slope: Vec<i128>,
    },
    /// `mul · ⌈(α·rows + β) / div⌉`, where `rows` is the window's own row
    /// count ([`SlotAbi::rows`]).
    ///
    /// **THE TILING LAW, AND IT READS ROWS RATHER THAN COORDINATES.** Every
    /// instance of it in the catalog is a `div_ceil` an entry writes over the
    /// extent it was handed — `attn::tiled_grid`'s `rows.div_ceil(SDPA_TILE)`
    /// and `linear::gemm::tile_grid`'s `div_ceil(rows, TILE_M) · 2` — so the
    /// numerator is affine in one number and the form says which.
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
}

impl Law {
    /// This law at one point of the descriptor's space, with the window's
    /// rows the tiling form divides.
    #[must_use]
    pub fn at(&self, coords: &[i128], rows: i128) -> i128 {
        match self {
            Law::Const(v) => *v,
            Law::Affine { base, slope } => slope
                .iter()
                .zip(coords)
                .fold(*base, |sum, (b, x)| sum + b * x),
            Law::Ceil {
                mul,
                alpha,
                beta,
                div,
            } => {
                let numerator = alpha * rows + beta;
                mul * numerator.div_euclid(*div)
                    + mul * i128::from(numerator.rem_euclid(*div) != 0)
            }
        }
    }

    /// Whether the number moves at all.
    #[must_use]
    pub fn varies(&self) -> bool {
        !matches!(self, Law::Const(_))
    }

    /// Which axes this law reads. A tiling law reads whatever the window's
    /// rows read, which is [`SlotAbi::rows`]'s business and not this one's.
    #[must_use]
    pub fn reads(&self) -> Vec<usize> {
        match self {
            Law::Const(_) | Law::Ceil { .. } => Vec::new(),
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
        }
    }
}

impl std::fmt::Display for Law {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
        }
    }
}

/// One arm of one slot: a shader point, the skeleton encoded for it, and the
/// components that move within it.
#[derive(Clone, Debug)]
pub struct Arm {
    /// The shader point this arm is.
    pub point: Point,
    /// A recording of this slot at this arm, verbatim: what to encode.
    pub skeleton: Slot,
    /// Every component that moves within the arm, and its law. Ordered, so
    /// two derivations of one artifact compare.
    pub laws: Vec<(At, Law)>,
}

/// How a slot picks its arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    /// One arm, always. What 338 of qwen35-d0.8b's 465 slots are.
    Only,
    /// Arm 0 below `at` rows of the window, arm 1 at or above it.
    ///
    /// **THE ARM SWITCH IS A THRESHOLD AND NOT A TABLE**, which is what lets
    /// a shader evaluate it in two instructions. The number is BRACKETED
    /// rather than assumed: the derivation refuses unless some ladder holds
    /// the two consecutive row counts the switch happens between, so `at` is
    /// the row it happens at and not an interval it happens inside.
    Rows {
        /// The first row count that takes arm 1.
        at: u32,
    },
}

/// One dispatch's law table: the skeleton the ICB is encoded from, the
/// components a fire rewrites, and — for a slot whose entry picks its arm off
/// the window — one of each per arm.
#[derive(Clone, Debug)]
pub struct SlotAbi {
    /// The template region this dispatch stood in.
    pub region: u32,
    /// Which run of that region's window.
    pub run: u32,
    /// The window's own row count, as a law.
    ///
    /// **THE ONE LAW THAT IS NOT A COMPONENT OF THE DISPATCH.** It decides
    /// three things nothing else can: whether the slot runs at all (a zero
    /// window is `walk`'s rule 1, and the ICB's `reset`), which arm it takes
    /// ([`Pick::Rows`]), and what a tiling law divides ([`Law::Ceil`]).
    /// Fitted like any other component and refused like any other component —
    /// it is `Const` or `Affine`, never `Ceil`.
    pub rows: Law,
    /// Every arm this slot was seen at, in first-sighting order.
    pub arms: Vec<Arm>,
    /// Which of them runs.
    pub pick: Pick,
}

impl SlotAbi {
    /// The arm this slot takes at a window of `rows` rows.
    #[must_use]
    pub fn arm(&self, rows: i128) -> &Arm {
        match self.pick {
            Pick::Only => &self.arms[0],
            Pick::Rows { at } => {
                let which = usize::from(rows >= i128::from(at));
                &self.arms[which.min(self.arms.len() - 1)]
            }
        }
    }

    /// The base arm's shader point — what a census names the slot by.
    #[must_use]
    pub fn point(&self) -> Point {
        self.arms[0].point
    }

    /// The base arm's skeleton.
    #[must_use]
    pub fn skeleton(&self) -> &Slot {
        &self.arms[0].skeleton
    }

    /// How many components this slot rewrites per fire, at its widest arm.
    #[must_use]
    pub fn rewrites(&self) -> usize {
        self.arms.iter().map(|arm| arm.laws.len()).max().unwrap_or(0)
    }

    /// Whether this slot's shader point moves with the composition.
    #[must_use]
    pub fn armed(&self) -> bool {
        self.arms.len() > 1
    }
}

/// The table `.wiki/palo/icb.md` §3 calls the binding recipe and design §2
/// calls `DescriptorAbi`: slot → which arguments and which grid axes move,
/// how, and which shader they move under.
#[derive(Clone, Debug)]
pub struct DescriptorAbi {
    /// The descriptor's own directions, in the order every `slope` is written
    /// in.
    pub axes: Vec<Axis>,
    /// One entry per dispatch, in walk order.
    pub slots: Vec<SlotAbi>,
    /// The point the skeletons were recorded at, so a reader can undo the
    /// fit.
    pub origin: Vec<i128>,
    /// The class table at that point — what [`Recipe`] measures from.
    pub origin_classes: Vec<(u32, u32)>,
    /// One per direction: how to read that direction's coordinate out of a
    /// live class table.
    pub recipe: Vec<Recipe>,
    /// The box the probes covered, per direction — `(min, max)` coordinate.
    /// Not a fence: a law extrapolates by construction and the gates check
    /// that it does. It is what a reader needs to know how far the evidence
    /// reaches.
    pub probed: Vec<(i128, i128)>,
}

impl DescriptorAbi {
    /// How many dispatches.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether the artifact dispatches nothing at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// How many components across the whole table move with the descriptor.
    #[must_use]
    pub fn affine(&self) -> usize {
        self.slots.iter().map(SlotAbi::rewrites).sum()
    }

    /// How many components are constant — the ones the ICB encodes once.
    #[must_use]
    pub fn constants(&self) -> usize {
        self.slots
            .iter()
            .map(|slot| components(slot.skeleton()) - slot.arms[0].laws.len())
            .sum()
    }

    /// The slots that rewrite nothing at all: encoded once, never touched.
    #[must_use]
    pub fn frozen(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| !s.armed() && s.arms[0].laws.is_empty())
            .count()
    }

    /// How many slots pick their shader off the window.
    #[must_use]
    pub fn armed(&self) -> usize {
        self.slots.iter().filter(|s| s.armed()).count()
    }

    /// How many components of each law kind the table holds.
    #[must_use]
    pub fn by_kind(&self) -> BTreeMap<&'static str, usize> {
        let mut per = BTreeMap::new();
        for slot in &self.slots {
            for arm in &slot.arms {
                for (_, law) in &arm.laws {
                    *per.entry(law.kind()).or_default() += 1;
                }
            }
        }
        per
    }

    /// The census, per shader point: how many slots and how many rewrites.
    #[must_use]
    pub fn census(&self) -> Vec<(Point, usize, usize)> {
        let mut per: BTreeMap<Point, (usize, usize)> = BTreeMap::new();
        for slot in &self.slots {
            for arm in &slot.arms {
                let row = per.entry(arm.point).or_default();
                row.0 += 1;
                row.1 += arm.laws.len();
            }
        }
        per.into_iter()
            .map(|(point, (slots, laws))| (point, slots, laws))
            .collect()
    }

    /// Where a live composition sits in the basis every law is written in.
    ///
    /// **THIS IS THE FUNCTION THE REBIND SHADER IS A TRANSLATION OF.** The
    /// host form is here so a test can diff the two, and the device form
    /// reads the same coefficients out of the same packed bytes.
    #[must_use]
    pub fn coords_of(&self, classes: &[(u32, u32)]) -> Vec<i128> {
        self.recipe.iter().map(|row| row.at(classes)).collect()
    }

    /// One slot's window rows at a composition.
    #[must_use]
    pub fn rows_at(&self, slot: usize, coords: &[i128]) -> Option<i128> {
        Some(self.slots.get(slot)?.rows.at(coords, 0))
    }

    /// Re-derive one slot's whole argument list at a point of the
    /// descriptor's space — what `icb::rebind` computes on the device, in
    /// host arithmetic, so a test can diff the two.
    ///
    /// `None` for a slot index the table does not hold. A slot whose window
    /// is empty at these coordinates answers `None` too, because the walk
    /// would not have dispatched it at all.
    #[must_use]
    pub fn slot_at(&self, slot: usize, coords: &[i128]) -> Option<Slot> {
        let abi = self.slots.get(slot)?;
        let rows = abi.rows.at(coords, 0);
        if rows <= 0 {
            return None;
        }
        let arm = abi.arm(rows);
        let mut built = arm.skeleton.clone();
        built.window_rows = rows as u32;
        for (at, law) in &arm.laws {
            let value = law.at(coords, rows);
            match *at {
                At::Lane(axis) => built.lanes[axis as usize] = value as u32,
                At::Group(axis) => built.group[axis as usize] = value as u32,
                At::Arg(index) => {
                    let arg = &mut built.args[index as usize];
                    *arg = match *arg {
                        Arg::Buffer { slab, mutable, .. } => Arg::Buffer {
                            slab,
                            offset: value as u64,
                            mutable,
                        },
                        Arg::I32(_) => Arg::I32(value as i32),
                        Arg::U32(_) => Arg::U32(value as u32),
                        Arg::Usize(_) => Arg::Usize(value as u64),
                        other => other,
                    };
                }
            }
        }
        Some(built)
    }
}

/// How many fittable components one slot has: three grid axes, three
/// threadgroup axes, one per argument that carries a number.
fn components(slot: &Slot) -> usize {
    6 + slot.args.iter().filter(|a| a.scalar().is_some()).count()
}

/// Every component of one slot that a law could be written for, paired with
/// its value in this recording.
fn read(slot: &Slot) -> Vec<(At, Option<i128>)> {
    let mut out = Vec::with_capacity(components(slot));
    for axis in 0..3u8 {
        out.push((At::Lane(axis), Some(i128::from(slot.lanes[axis as usize]))));
    }
    for axis in 0..3u8 {
        out.push((At::Group(axis), Some(i128::from(slot.group[axis as usize]))));
    }
    for (index, arg) in slot.args.iter().enumerate() {
        out.push((At::Arg(index as u16), arg.scalar()));
    }
    out
}

/// One base composition and the ladders that step away from it.
///
/// **A LADDER, NOT A BUMP, AND THAT IS THE WHOLE OF WHAT THE THIRD LAW
/// NEEDED.** `ladders[k]` holds the walks at `base + 1·e_k`, `base + 2·e_k`,
/// … so a component that is a staircase in direction `k` is SEEN as a
/// staircase rather than as a constant that jumped once. The first rung is
/// what the affine slope is read from; the rest is what a tiling law's
/// divisor is solved against and what brackets an arm switch to the row it
/// happens at.
#[derive(Clone, Debug)]
pub struct Probe {
    /// This probe's origin.
    pub base: Recording,
    /// One ladder per direction, in [`DescriptorAbi::axes`] order.
    pub ladders: Vec<Vec<Recording>>,
}

/// Everything the derivation is fitted from and the one point it is verified
/// at.
///
/// **ONE PROBE PER ARM REGION.** A slot whose entry picks its arm off the
/// window is two law tables, and each of them has to be fitted from samples
/// that are actually IN it — a ladder that never reaches 32 rows says nothing
/// about the tile arm. So the harness supplies a base per region of the
/// composition space it means to serve, each with its own ladders, and the
/// fit refuses an arm no probe steps every direction inside.
#[derive(Clone, Debug)]
pub struct Probes {
    /// The probes. The FIRST one's base is the origin of the coordinates and
    /// the skeleton every un-armed slot is encoded from.
    pub probes: Vec<Probe>,
    /// A composition held out of every fit — the third point, in the
    /// derivation's own sense.
    pub check: Recording,
}

impl Probes {
    /// Every recording, in one list: the bases, every rung, and the check.
    fn every(&self) -> Vec<&Recording> {
        let mut out = Vec::new();
        for probe in &self.probes {
            out.push(&probe.base);
            for ladder in &probe.ladders {
                out.extend(ladder.iter());
            }
        }
        out.push(&self.check);
        out
    }

    /// How many walks this is.
    #[must_use]
    pub fn walks(&self) -> usize {
        self.every().len()
    }
}

/// The whole derivation's answer: the table that fitted, and every component
/// that did not.
///
/// **A REFUSAL LIST IS A DELIVERABLE**, which is why this exists beside
/// [`derive`]. `.wiki/palo/icb.md` §7 step 2's gate is "the derived table over
/// the whole catalog, with `Fault::Unaffine` naming anything that does not fit
/// — the census is the deliverable even if it is a refusal list", and a
/// fail-fast derivation can only ever name the first one. So the survey fits
/// every component it can, keeps the ones that did not as faults, and leaves
/// the caller to decide whether a table with a tail is a table.
#[derive(Debug)]
pub struct Survey {
    /// The components that fitted, as a table.
    pub abi: DescriptorAbi,
    /// The components that did not, each naming its slot and its place.
    pub unaffine: Vec<Fault>,
    /// The slots that are not one shader point at all, reported rather than
    /// refused: they are fitted one table per arm and picked between by
    /// [`Pick::Rows`].
    pub armed: Vec<Armed>,
}

/// One slot whose shader point is a function of the composition.
#[derive(Clone, Debug)]
pub struct Armed {
    /// The slot, in walk order.
    pub slot: u32,
    /// Every point seen at that slot, with the composition that produced it.
    pub points: Vec<(Point, Vec<(u32, u32)>)>,
    /// The window row count the switch was bracketed to, when it was.
    pub at: Option<u32>,
}

impl std::fmt::Display for Armed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "slot {} is ", self.slot)?;
        for (at, (point, classes)) in self.points.iter().enumerate() {
            if at > 0 {
                f.write_str(", and ")?;
            }
            write!(f, "{point} at {classes:?}")?;
        }
        match self.at {
            Some(rows) => write!(f, "; the switch is at {rows} window rows"),
            None => f.write_str("; the switch is not bracketed"),
        }
    }
}

/// Derive the table, refusing on the first component that does not fit.
///
/// # Errors
///
/// [`Fault::Unstructured`] when two probes did not walk the same template,
/// [`Fault::Unaffine`] for the first component no law in the language
/// predicts.
pub fn derive(axes: &[Axis], probes: &Probes) -> Result<DescriptorAbi> {
    let surveyed = survey(axes, probes)?;
    match surveyed.unaffine.into_iter().next() {
        Some(fault) => Err(fault),
        None => Ok(surveyed.abi),
    }
}

/// Derive the table, keeping every component that does not fit.
///
/// # Errors
///
/// [`Fault::Unstructured`] alone: two probes that did not walk the same
/// template have no table to survey at all, and neither does a basis whose
/// inverse is not integral, so those stay fatal.
#[allow(clippy::too_many_lines)]
pub fn survey(axes: &[Axis], probes: &Probes) -> Result<Survey> {
    let Some(first) = probes.probes.first() else {
        return Err(Fault::Unstructured {
            slot: 0,
            why: "no probe at all: the fit needs a base composition".to_string(),
        });
    };
    for (at, probe) in probes.probes.iter().enumerate() {
        if probe.ladders.len() != axes.len() {
            return Err(Fault::Unstructured {
                slot: 0,
                why: format!(
                    "probe {at} carries {} ladders and the basis has {} directions",
                    probe.ladders.len(),
                    axes.len()
                ),
            });
        }
        // A rung must step exactly its own direction, or the slope it yields
        // is the sum of two derivatives wearing one name.
        for (k, ladder) in probe.ladders.iter().enumerate() {
            for (rung, walk) in ladder.iter().enumerate() {
                let moved: Vec<usize> = walk
                    .coords
                    .iter()
                    .zip(&probe.base.coords)
                    .enumerate()
                    .filter_map(|(at, (there, here))| (there != here).then_some(at))
                    .collect();
                if moved != vec![k] {
                    return Err(Fault::Unstructured {
                        slot: 0,
                        why: format!(
                            "rung {rung} of probe {at}'s ladder along `{}` was supposed to \
                             step that direction alone and stepped {moved:?}",
                            axes[k]
                        ),
                    });
                }
            }
        }
    }

    let origin = first.base.coords.clone();
    if origin.len() != axes.len() {
        return Err(Fault::Unstructured {
            slot: 0,
            why: format!(
                "the base point has {} coordinates and the basis has {} directions",
                origin.len(),
                axes.len()
            ),
        });
    }

    let every = probes.every();
    for walk in &every {
        if walk.coords.len() != axes.len() {
            return Err(Fault::Unstructured {
                slot: 0,
                why: format!(
                    "a walk at {:?} carries {} coordinates and the basis has {}",
                    walk.classes,
                    walk.coords.len(),
                    axes.len()
                ),
            });
        }
    }

    let armed = structure(&every)?;
    let recipe = invert(axes, &first.base.classes, &origin, &every)?;
    let probed: Vec<(i128, i128)> = (0..axes.len())
        .map(|k| {
            let mut lo = i128::MAX;
            let mut hi = i128::MIN;
            for walk in &every {
                lo = lo.min(walk.coords[k]);
                hi = hi.max(walk.coords[k]);
            }
            (lo, hi)
        })
        .collect();

    // Every sample, as `(coords, window rows, the slot at that index)`.
    let samples: Vec<&Recording> = every.clone();

    let mut unaffine = Vec::new();
    let mut slots = Vec::with_capacity(first.base.slots.len());
    let mut armed_out: Vec<Armed> = Vec::new();
    for index in 0..first.base.slots.len() {
        let (abi, bracket) = fit_slot(axes, &samples, index, &mut unaffine);
        if abi.armed()
            && let Some(entry) = armed.iter().find(|entry| entry.slot == index as u32)
        {
            let mut entry = entry.clone();
            entry.at = bracket;
            armed_out.push(entry);
        }
        slots.push(abi);
    }

    Ok(Survey {
        abi: DescriptorAbi {
            axes: axes.to_vec(),
            slots,
            origin,
            origin_classes: first.base.classes.clone(),
            recipe,
            probed,
        },
        unaffine,
        armed: armed_out,
    })
}

/// One slot: its window-rows law, its arms, and the components of each.
fn fit_slot(
    axes: &[Axis],
    samples: &[&Recording],
    index: usize,
    unaffine: &mut Vec<Fault>,
) -> (SlotAbi, Option<u32>) {
    let here = samples[0].slots[index].clone();
    // 1. The window's rows, over EVERY sample: the one law that is not a
    //    component of the dispatch and the one every other law leans on.
    let rows_points: Vec<(Vec<i128>, i128)> = samples
        .iter()
        .map(|walk| {
            (
                walk.coords.clone(),
                i128::from(walk.slots[index].window_rows),
            )
        })
        .collect();
    let rows = match affine(axes, &rows_points) {
        Ok(law) => law,
        Err(why) => {
            unaffine.push(Fault::Unaffine {
                slot: index as u32,
                point: here.point.to_string(),
                at: "the window's own row count".to_string(),
                why,
            });
            Law::Const(i128::from(here.window_rows))
        }
    };

    // 2. The arms, in first-sighting order.
    let mut points: Vec<Point> = Vec::new();
    for walk in samples {
        let point = walk.slots[index].point;
        if !points.contains(&point) {
            points.push(point);
        }
    }

    // 3. The threshold, bracketed on the window's rows.
    let (pick, bracket) = if points.len() < 2 {
        (Pick::Only, None)
    } else {
        match bracket(samples, index, &points) {
            Ok((at, ordered)) => {
                points = ordered;
                (Pick::Rows { at }, Some(at))
            }
            Err(why) => {
                unaffine.push(Fault::Unaffine {
                    slot: index as u32,
                    point: points
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(" <-> "),
                    at: "which shader the slot is".to_string(),
                    why,
                });
                (Pick::Only, None)
            }
        }
    };

    // 4. One law table per arm, fitted from the samples that are IN it.
    let mut arms = Vec::with_capacity(points.len());
    for point in &points {
        let mine: Vec<&&Recording> = samples
            .iter()
            .filter(|walk| walk.slots[index].point == *point)
            .collect();
        let skeleton = mine[0].slots[index].clone();
        let shape = read(&skeleton);
        let mut laws = Vec::new();
        for (component, (at, value)) in shape.iter().enumerate() {
            if value.is_none() {
                continue;
            }
            let observed: Vec<(Vec<i128>, i128, i128)> = mine
                .iter()
                .filter_map(|walk| {
                    let slot = &walk.slots[index];
                    read(slot)[component].1.map(|v| {
                        (
                            walk.coords.clone(),
                            i128::from(slot.window_rows),
                            v,
                        )
                    })
                })
                .collect();
            if observed.len() != mine.len() {
                unaffine.push(Fault::Unaffine {
                    slot: index as u32,
                    point: point.to_string(),
                    at: at.to_string(),
                    why: "carries a number at one sample of this arm and none at another"
                        .to_string(),
                });
                continue;
            }
            match fit(axes, &observed) {
                Ok(None) => {}
                Ok(Some(law)) => laws.push((*at, law)),
                Err(why) => {
                    unaffine.push(Fault::Unaffine {
                        slot: index as u32,
                        point: point.to_string(),
                        at: at.to_string(),
                        why,
                    });
                }
            }
        }
        arms.push(Arm {
            point: *point,
            skeleton,
            laws,
        });
    }

    (
        SlotAbi {
            region: here.region,
            run: here.run,
            rows,
            arms,
            pick,
        },
        bracket,
    )
}

/// The row count an arm switch happens at, and the arms in threshold order.
///
/// **BRACKETED OR REFUSED.** The switch is a threshold on the window's rows
/// only if the two arms' row counts do not interleave, and it is a KNOWN
/// threshold only if some ladder holds the two consecutive counts it happens
/// between. Anything else names the interval it could not close.
fn bracket(
    samples: &[&Recording],
    index: usize,
    points: &[Point],
) -> std::result::Result<(u32, Vec<Point>), String> {
    if points.len() > 2 {
        return Err(format!(
            "{} arms, and a threshold picks between two",
            points.len()
        ));
    }
    let mut span: BTreeMap<Point, (u32, u32)> = BTreeMap::new();
    for walk in samples {
        let slot = &walk.slots[index];
        let row = span
            .entry(slot.point)
            .or_insert((slot.window_rows, slot.window_rows));
        row.0 = row.0.min(slot.window_rows);
        row.1 = row.1.max(slot.window_rows);
    }
    let mut ordered: Vec<Point> = points.to_vec();
    ordered.sort_by_key(|point| span[point].0);
    let low = span[&ordered[0]];
    let high = span[&ordered[1]];
    if low.1 >= high.0 {
        return Err(format!(
            "{} runs over {}..={} window rows and {} over {}..={} — they interleave, so \
             the arm is not a threshold on the rows",
            ordered[0], low.0, low.1, ordered[1], high.0, high.1
        ));
    }
    if low.1 + 1 != high.0 {
        return Err(format!(
            "the switch is somewhere in {}..={} window rows and no ladder holds the two \
             consecutive counts it happens between",
            low.1 + 1,
            high.0
        ));
    }
    Ok((high.0, ordered))
}

/// One component of one arm: a constant, a law, or a refusal.
///
/// `observed` is `(coords, window rows, value)` at every sample of the arm.
fn fit(
    axes: &[Axis],
    observed: &[(Vec<i128>, i128, i128)],
) -> std::result::Result<Option<Law>, String> {
    let first = observed[0].2;
    // **A CONSTANT IS A CLAIM AND IT GETS CHECKED**, over every sample of the
    // arm rather than over the ones a bump happened to move. Build log 30's
    // bug was exactly here: a grid axis written `rows.div_ceil(32)` is flat
    // across every step small enough to stay inside one tile, and a fit that
    // only verified the components it had already decided were variable would
    // call it a constant and encode the wrong grid into the ICB once, at
    // load.
    if observed.iter().all(|(_, _, v)| *v == first) {
        return Ok(None);
    }
    let points: Vec<(Vec<i128>, i128)> = observed
        .iter()
        .map(|(coords, _, value)| (coords.clone(), *value))
        .collect();
    match affine(axes, &points) {
        Ok(law) => Ok(Some(law)),
        Err(affine_why) => match ceiling(observed) {
            Some(law) => Ok(Some(law)),
            None => Err(format!(
                "{affine_why}; and no tiling law \
                 `mul·ceil((α·rows + β)/div)` with div ≤ {MAX_TILE} and α ≤ \
                 {MAX_NUMERATOR_SCALE} fits it either — the samples catch the staircase \
                 crossing {} time(s), and two is what pins the period",
                crossings(observed)
            )),
        },
    }
}

/// `base + Σ slope·coord`, fitted exactly and verified at every sample.
fn affine(axes: &[Axis], points: &[(Vec<i128>, i128)]) -> std::result::Result<Law, String> {
    let (here, value) = &points[0];
    let mut slope = vec![0i128; axes.len()];
    for k in 0..axes.len() {
        // A pair of samples that differ ONLY in direction k. A probe's base
        // and its own ladder are exactly that; anything else the harness
        // supplies is a bonus.
        let mut seen: Option<i128> = None;
        for (a, (xa, va)) in points.iter().enumerate() {
            for (xb, vb) in points.iter().skip(a + 1) {
                let moved: Vec<usize> = xa
                    .iter()
                    .zip(xb)
                    .enumerate()
                    .filter_map(|(at, (p, q))| (p != q).then_some(at))
                    .collect();
                if moved != vec![k] {
                    continue;
                }
                let run = xb[k] - xa[k];
                let rise = vb - va;
                if rise % run != 0 {
                    return Err(format!(
                        "stepping `{}` by {run} moved it by {rise}, which is not a whole \
                         multiple",
                        axes[k]
                    ));
                }
                let b = rise / run;
                match seen {
                    None => seen = Some(b),
                    Some(had) if had == b => {}
                    Some(had) => {
                        return Err(format!(
                            "`{}` moves it by {had} per step at one place and {b} at another",
                            axes[k]
                        ));
                    }
                }
            }
        }
        match seen {
            Some(b) => slope[k] = b,
            None => {
                return Err(format!(
                    "no two samples of this arm differ only in `{}`, so its slope is \
                     unwitnessed",
                    axes[k]
                ));
            }
        }
    }
    // The value at the ZERO of the coordinates, not at this arm's first
    // sample: a law has to be evaluable anywhere, including at the
    // compositions the probes could not visit (an empty class is one).
    let base = value - slope.iter().zip(here).map(|(b, x)| b * x).sum::<i128>();
    let law = Law::Affine { base, slope };
    for (coords, want) in points {
        let got = law.at(coords, 0);
        if got != *want {
            return Err(format!(
                "the line fitted from the ladders predicts {got} at {coords:?} and the walk \
                 produced {want}"
            ));
        }
    }
    Ok(law)
}

/// `mul·⌈(α·rows + β)/div⌉`, solved by interval arithmetic over the samples.
///
/// **THE SEARCH IS BOUNDED AND THE ANSWER IS CANONICAL.** For a candidate
/// `(mul, α, div)` the offset β is not searched at all: every sample says
/// `div·(w−1) < α·rows + β ≤ div·w`, so β lies in one half-open interval and
/// the intersection over the samples is one interval or empty. The smallest
/// `mul`, then the smallest `div`, then the smallest `α` that leaves a
/// non-empty interval wins — smallest because `⌈n/32⌉` and `⌈2n/64⌉` are the
/// same function and one of them is the one a reader can check against
/// `SDPA_TILE`.
fn ceiling(observed: &[(Vec<i128>, i128, i128)]) -> Option<Law> {
    if crossings(observed) < 2 {
        return None;
    }
    let mut common: i128 = 0;
    for (_, _, v) in observed {
        common = gcd(common, v.abs());
    }
    if common == 0 {
        return None;
    }
    let mut scales: Vec<i128> = (1..=common).filter(|m| common % m == 0).collect();
    scales.sort_unstable();
    for mul in scales {
        for div in 2..=MAX_TILE {
            for alpha in 1..=MAX_NUMERATOR_SCALE {
                let mut lo = i128::MIN;
                let mut hi = i128::MAX;
                let mut fits = true;
                for (_, rows, value) in observed {
                    if value % mul != 0 {
                        fits = false;
                        break;
                    }
                    let w = value / mul;
                    lo = lo.max(div * (w - 1) - alpha * rows);
                    hi = hi.min(div * w - alpha * rows);
                }
                if !fits || lo >= hi {
                    continue;
                }
                let law = Law::Ceil {
                    mul,
                    alpha,
                    beta: hi,
                    div,
                };
                if observed
                    .iter()
                    .all(|(coords, rows, value)| law.at(coords, *rows) == *value)
                {
                    return Some(law);
                }
            }
        }
    }
    None
}

/// How many times the samples catch the staircase in the act: a pair of
/// window row counts `r` and `r+1` whose values differ.
///
/// **TWO, OR THE PERIOD IS NOT PINNED.** One crossing says only that the
/// value stepped somewhere between two rows, and a ceiling with any divisor
/// wide enough to hold the sampled range explains that — over rows 16..47,
/// `⌈(r−15)/17⌉` is `⌈r/32⌉` exactly, and the search would answer 17 because
/// 17 is smaller. Two crossings say how far apart the steps are, which is the
/// divisor. This is the fitter's third-point discipline, in the form the
/// tiling law needs it: sample points straddling a multiple of the tile, and
/// then straddling the next one.
fn crossings(observed: &[(Vec<i128>, i128, i128)]) -> usize {
    let mut by_rows: BTreeMap<i128, i128> = BTreeMap::new();
    for (_, rows, value) in observed {
        by_rows.insert(*rows, *value);
    }
    by_rows
        .iter()
        .filter(|(rows, value)| {
            by_rows
                .get(&(**rows + 1))
                .is_some_and(|next| next != *value)
        })
        .count()
}

fn gcd(a: i128, b: i128) -> i128 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Invert the basis: one linear functional per direction over the class
/// table's own numbers, solved exactly and verified at every probe.
///
/// **THIS IS WHAT MAKES THE TABLE READABLE BY SOMETHING THAT DID NOT WALK.**
/// The laws are written in a basis of reachable directions; a fire carries a
/// class table. `step` says what one unit of each direction does to that
/// table, which is a `2·classes × directions` matrix; a square subsystem of
/// it that inverts over the integers is the recipe. There may be several and
/// they agree — the verification over every probe is what says so.
fn invert(
    axes: &[Axis],
    origin_classes: &[(u32, u32)],
    origin: &[i128],
    every: &[&Recording],
) -> Result<Vec<Recipe>> {
    let classes = origin_classes.len();
    let k = axes.len();
    for axis in axes {
        if axis.step.len() != classes {
            return Err(Fault::Unstructured {
                slot: 0,
                why: format!(
                    "direction `{axis}` states a step over {} classes and the base \
                     composition has {classes}",
                    axis.step.len()
                ),
            });
        }
    }
    // The full `2·classes × k` step matrix, row `2c` = class c's rows, row
    // `2c+1` = its lanes.
    let column = |row: usize, axis: usize| -> i128 {
        let (rows, lanes) = axes[axis].step[row / 2];
        i128::from(if row % 2 == 0 { rows } else { lanes })
    };
    let width = 2 * classes;
    // Every choice of `k` rows, in index order, until one inverts.
    let mut pivots: Vec<usize> = (0..k).collect();
    loop {
        if let Some(recipe) = try_pivots(&pivots, k, classes, origin_classes, origin, &column) {
            // The recipe is a claim about every probe, not only about the
            // rows it was solved from.
            for walk in every {
                let got: Vec<i128> = recipe.iter().map(|row| row.at(&walk.classes)).collect();
                if got != walk.coords {
                    return Err(Fault::Unstructured {
                        slot: 0,
                        why: format!(
                            "the basis inverts to a recipe that reads {got:?} out of the \
                             class table {:?}, and the harness placed that walk at {:?} — \
                             the directions are not independent in the descriptor's own \
                             numbers",
                            walk.classes, walk.coords
                        ),
                    });
                }
            }
            return Ok(recipe);
        }
        // Next combination of `k` rows out of `width`.
        let mut at = k;
        loop {
            if at == 0 {
                return Err(Fault::Unstructured {
                    slot: 0,
                    why: format!(
                        "no {k} of the class table's {width} numbers invert this basis over \
                         the integers, so a fire's composition cannot be read back into the \
                         coordinates the laws are written in"
                    ),
                });
            }
            at -= 1;
            if pivots[at] < width - (k - at) {
                pivots[at] += 1;
                for next in at + 1..k {
                    pivots[next] = pivots[next - 1] + 1;
                }
                break;
            }
        }
    }
}

/// One choice of pivot rows, inverted by Cramer's rule over the integers.
fn try_pivots(
    pivots: &[usize],
    k: usize,
    classes: usize,
    origin_classes: &[(u32, u32)],
    origin: &[i128],
    column: &dyn Fn(usize, usize) -> i128,
) -> Option<Vec<Recipe>> {
    let a: Vec<Vec<i128>> = pivots
        .iter()
        .map(|row| (0..k).map(|axis| column(*row, axis)).collect())
        .collect();
    let det = determinant(&a);
    if det == 0 {
        return None;
    }
    // Row `k` of A⁻¹: solve `xᵀ·A = e_k`, i.e. `Aᵀ·x = e_k`, by Cramer.
    let mut recipe = Vec::with_capacity(k);
    for axis in 0..k {
        let mut coefficients = vec![0i128; k];
        for pivot in 0..k {
            let mut m = transpose(&a);
            for row in 0..k {
                m[row][pivot] = i128::from(row == axis);
            }
            let numerator = determinant(&m);
            if numerator % det != 0 {
                return None;
            }
            coefficients[pivot] = numerator / det;
        }
        let mut rows = vec![0i128; classes];
        let mut lanes = vec![0i128; classes];
        let mut konst = origin[axis];
        for (pivot, weight) in pivots.iter().zip(&coefficients) {
            let class = pivot / 2;
            let (r, l) = origin_classes[class];
            if pivot % 2 == 0 {
                rows[class] += *weight;
                konst -= *weight * i128::from(r);
            } else {
                lanes[class] += *weight;
                konst -= *weight * i128::from(l);
            }
        }
        recipe.push(Recipe {
            konst,
            rows,
            lanes,
        });
    }
    Some(recipe)
}

fn transpose(a: &[Vec<i128>]) -> Vec<Vec<i128>> {
    let n = a.len();
    (0..n)
        .map(|row| (0..n).map(|col| a[col][row]).collect())
        .collect()
}

/// Laplace expansion. The matrices here are `k × k` with `k` the number of
/// probe directions — three today, and a basis with more than a handful of
/// directions is a harness that has lost the plot.
fn determinant(a: &[Vec<i128>]) -> i128 {
    let n = a.len();
    match n {
        0 => 1,
        1 => a[0][0],
        2 => a[0][0] * a[1][1] - a[0][1] * a[1][0],
        _ => {
            let mut sum = 0;
            for col in 0..n {
                let minor: Vec<Vec<i128>> = a[1..]
                    .iter()
                    .map(|row| {
                        row.iter()
                            .enumerate()
                            .filter_map(|(at, v)| (at != col).then_some(*v))
                            .collect()
                    })
                    .collect();
                let sign = if col % 2 == 0 { 1 } else { -1 };
                sum += sign * a[0][col] * determinant(&minor);
            }
            sum
        }
    }
}

/// Every probe walked the same template, slot for slot.
///
/// This is the claim `.wiki/palo/icb.md` §6 rests on, checked before anything
/// is fitted. Two things can go wrong and they are not the same thing:
///
/// - **A different NUMBER of slots, or a different argument arity or kind at
///   one.** No single indirect command buffer serves both compositions and
///   the exec key has not collapsed — refused.
/// - **The same slot at a different shader POINT.** A kernel entry that picks
///   its arm off the window (`linear`'s gemv/gemm split is the live one) hands
///   one slot two pipelines. That is survivable — a compute ICB slot's
///   pipeline state is rebindable from the GPU, measured — so it is REPORTED
///   rather than refused, fitted one table per arm, and picked between by a
///   threshold on the window's rows.
fn structure(every: &[&Recording]) -> Result<Vec<Armed>> {
    let first = every[0];
    let mut armed: Vec<Armed> = Vec::new();
    for other in &every[1..] {
        if other.slots.len() != first.slots.len() {
            return Err(Fault::Unstructured {
                slot: 0,
                why: format!(
                    "the probe at {:?} walks {} dispatches and the base walks {} — one \
                     composition's launches are not the other's",
                    other.classes,
                    other.slots.len(),
                    first.slots.len()
                ),
            });
        }
        for (index, (here, there)) in first.slots.iter().zip(&other.slots).enumerate() {
            if here.point != there.point {
                let entry = match armed.iter_mut().find(|e| e.slot == index as u32) {
                    Some(entry) => entry,
                    None => {
                        armed.push(Armed {
                            slot: index as u32,
                            points: vec![(here.point, first.classes.clone())],
                            at: None,
                        });
                        armed.last_mut().expect("just pushed")
                    }
                };
                if !entry.points.iter().any(|(point, _)| *point == there.point) {
                    entry.points.push((there.point, other.classes.clone()));
                }
                continue;
            }
            if here.args.len() != there.args.len() {
                return Err(Fault::Unstructured {
                    slot: index as u32,
                    why: format!(
                        "{} binds {} arguments at the base point and {} at {:?}",
                        here.point,
                        here.args.len(),
                        there.args.len(),
                        other.classes
                    ),
                });
            }
            for (argument, (a, b)) in here.args.iter().zip(&there.args).enumerate() {
                if a.shape() != b.shape() {
                    return Err(Fault::Unstructured {
                        slot: index as u32,
                        why: format!(
                            "{}'s argument {argument} is {} at the base point and {} at \
                             {:?} — an ICB slot binds one reservation and one kind",
                            here.point,
                            a.kind(),
                            b.kind(),
                            other.classes
                        ),
                    });
                }
            }
        }
    }
    // A slot with two arms may bind different arguments on each; the check
    // above skips those pairs deliberately, and the per-arm fit is what
    // states each arm's own list.
    let seen: BTreeSet<u32> = armed.iter().map(|entry| entry.slot).collect();
    debug_assert_eq!(seen.len(), armed.len(), "one entry per armed slot");
    Ok(armed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axes() -> Vec<Axis> {
        vec![
            Axis::new("a decode lane", vec![(1, 1), (0, 0)]),
            Axis::new("a prefill lane of 8 tokens", vec![(0, 0), (8, 1)]),
            Axis::new("one more prefill token", vec![(0, 0), (1, 0)]),
        ]
    }

    /// The basis the live harness uses, inverted: two classes, three
    /// directions, and the answer is the one a reader can check by hand.
    #[test]
    fn the_probe_basis_inverts_into_a_reading_of_the_class_table() {
        let origin_classes = vec![(2u32, 2u32), (16, 2)];
        let walk = Recording {
            slots: Vec::new(),
            classes: origin_classes.clone(),
            coords: vec![0, 0, 0],
        };
        let recipe = invert(&axes(), &origin_classes, &[0, 0, 0], &[&walk]).expect("inverts");
        // d = decode rows − 2; p = prefill lanes − 2; t = prefill rows − 8·prefill lanes.
        assert_eq!(recipe[0].at(&[(5, 5), (16, 2)]), 3);
        assert_eq!(recipe[1].at(&[(2, 2), (24, 3)]), 1);
        assert_eq!(recipe[2].at(&[(2, 2), (17, 2)]), 1);
        // AND THE ALL-DECODE COMPOSITION, which is the one the probes cannot
        // visit and the ICB has to serve: every prefill number is zero.
        assert_eq!(recipe[1].at(&[(4, 4), (0, 0)]), -2);
        assert_eq!(recipe[2].at(&[(4, 4), (0, 0)]), 0);
    }

    /// `⌈rows/32⌉` is refused as affine and fitted as a tiling law, and the
    /// answer is the constant a reader can find in `kernels_metal::attn`.
    #[test]
    fn a_div_ceil_grid_axis_fits_the_tiling_law_and_names_the_tile() {
        let observed: Vec<(Vec<i128>, i128, i128)> = (16..80)
            .map(|rows: i128| {
                (
                    vec![0, 0, rows - 16],
                    rows,
                    rows.div_euclid(32) + i128::from(rows % 32 != 0),
                )
            })
            .collect();
        let law = ceiling(&observed).expect("the tiling law fits");
        assert_eq!(
            law,
            Law::Ceil {
                mul: 1,
                alpha: 1,
                beta: 0,
                div: 32
            }
        );
        for rows in 1..200i128 {
            let want = rows.div_euclid(32) + i128::from(rows % 32 != 0);
            assert_eq!(law.at(&[], rows), want, "at {rows} rows");
        }
    }

    /// `2·⌈rows/32⌉` — `linear::gemm`'s tile arm — needs the multiplier, and
    /// the fit finds it rather than being told.
    #[test]
    fn a_scaled_tiling_law_finds_its_multiplier() {
        let observed: Vec<(Vec<i128>, i128, i128)> = (32..112)
            .map(|rows: i128| {
                (
                    vec![0, 0, rows],
                    rows,
                    2 * (rows.div_euclid(32) + i128::from(rows % 32 != 0)),
                )
            })
            .collect();
        let law = ceiling(&observed).expect("the scaled tiling law fits");
        assert_eq!(
            law,
            Law::Ceil {
                mul: 2,
                alpha: 1,
                beta: 0,
                div: 32
            }
        );
    }

    /// A staircase no ceiling explains is still a refusal.
    #[test]
    fn a_component_that_is_neither_law_is_still_refused() {
        let observed: Vec<(Vec<i128>, i128, i128)> = (1..40)
            .map(|rows: i128| (vec![rows], rows, rows * rows))
            .collect();
        assert_eq!(ceiling(&observed), None);
    }
}
