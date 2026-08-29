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
//! # The component language lives in [`engine::law`]
//!
//! **THE PROSE THAT WAS HERE MOVED WITH THE TYPES** (seat wave B-law). Why
//! the language has exactly three fitted forms, why the third one is a
//! ceiling over the window's rows and not a generalisation, and why a probe
//! is a LADDER rather than a bump, are all facts about the law language
//! rather than about Metal — and the CUDA plane's `device::map` names the
//! same components and refuses with the same reasons — so they are stated
//! once, in the neutral crate, and this module is what supplies the samples:
//!
//! ```text
//! Const  v                                   encoded once, never rewritten
//! Affine v = base + Σ slope[k] · coord[k]     the windowed cut, the extent
//! Ceil   v = mul · ⌈(α·rows + β) / div⌉       the TILING law
//! ```
//!
//! What this module does is walk the same template many times against
//! synthetic descriptors, [record](crate::record) each walk, and hand the
//! differences to [`engine::law::fit`]:
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
//! **A component matching neither law is still `Unaffine` by name.** The
//! refusal did not get weaker; it got a second thing to try first.
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
//! The walk skips a zero-row region's nodes (`engine::fire::walk` rule 1), so
//! a composition with an empty window produces FEWER slots than one without.
//! Every probe point here therefore holds every class, and the derived table
//! is the FULL composition's — which is the point rather than a limitation:
//! design §5's "all compositions live inside it" means the artifact holds
//! every launch and a fire turns the absent ones off. What turns one off is
//! [`SlotAbi::rows`] evaluating to zero, and the ICB is what acts on it.

use std::collections::{BTreeMap, BTreeSet};

use engine::law::fit;

use crate::error::{Fault, Result};
use crate::record::{Arg, Point, Recording, Slot};

// ─────────────────────────────────────────────────────────────────────────
// The vocabulary — [`engine::law`], not this module's
//
// **THE LANGUAGE MOVED AND THE MODULE DOC ABOVE WENT WITH IT** (seat wave
// B-law). `Law`, the basis it is written in (`Axis`), the inverse that reads
// a class table back into that basis (`Recipe`), the place a law lives
// (`At`) and the two search ceilings are the vocabulary the CUDA plane
// speaks too — its `device::map::diff` names the same components and refuses
// with the same reasons — so they live in the neutral crate and each shell
// supplies only its recorder. What is left in this file IS the Metal
// recorder: a `Recording` is an ICB encode capture, and everything below
// turns a set of them into a table `icb::rebind` can lower.
//
// Two places changed their spelling and are NOT re-exported under the old
// one, because a name that means two things is what this wave exists to
// stop: `At::Lane` is `At::Grid` and `At::Group` is `At::Block`. A CUDA node
// carries both numbers too, and "group" only reads as the threadgroup from
// one of the two planes.
pub use engine::law::fit::{MAX_NUMERATOR_SCALE, MAX_TILE};
pub use engine::law::{At, Axis, Law, Recipe};

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
        self.slots.get(slot)?.rows.at(coords, 0)
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
        let rows = abi.rows.at(coords, 0)?;
        if rows <= 0 {
            return None;
        }
        let arm = abi.arm(rows);
        let mut built = arm.skeleton.clone();
        built.window_rows = rows as u32;
        for (at, law) in &arm.laws {
            let value = law.at(coords, rows)?;
            match *at {
                At::Grid(axis) => built.lanes[axis as usize] = value as u32,
                At::Block(axis) => built.group[axis as usize] = value as u32,
                // The recorder enumerates exactly the grid axes, the
                // threadgroup axes and the arguments ([`read`]), so a Metal
                // law table holds no entry, shared-memory or shape component
                // and there is nothing here to restate.
                At::Entry | At::Shared | At::Shape => {}
                At::Arg { at: index, .. } => {
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
        out.push((At::Grid(axis), Some(i128::from(slot.lanes[axis as usize]))));
    }
    for axis in 0..3u8 {
        out.push((At::Block(axis), Some(i128::from(slot.group[axis as usize]))));
    }
    for (index, arg) in slot.args.iter().enumerate() {
        out.push((
            At::Arg {
                at: index as u16,
                word: 0,
            },
            arg.scalar(),
        ));
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
    // The neutral inverter verifies its answer against every probe, so it
    // is handed the pairs and not the recordings: a `(class table, the point
    // the harness placed that walk at)` is all the check reads.
    let sites: Vec<fit::Site<'_>> = every
        .iter()
        .map(|walk| (walk.classes.as_slice(), walk.coords.as_slice()))
        .collect();
    let recipe = fit::invert(axes, &first.base.classes, &origin, &sites).map_err(|refusal| {
        Fault::Unstructured {
            slot: 0,
            why: refusal.why,
        }
    })?;
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
    let rows = match fit::affine(axes, &rows_points) {
        Ok(law) => law,
        Err(refusal) => {
            unaffine.push(Fault::Unaffine {
                slot: index as u32,
                point: here.point.to_string(),
                at: "the window's own row count".to_string(),
                why: refusal.why,
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
            match fit::component(axes, &observed) {
                Ok(None) => {}
                Ok(Some(law)) => laws.push((*at, law)),
                Err(refusal) => {
                    unaffine.push(Fault::Unaffine {
                        slot: index as u32,
                        point: point.to_string(),
                        at: at.to_string(),
                        why: refusal.why,
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
