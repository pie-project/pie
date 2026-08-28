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
//! # The method, and why it is honest rather than clever
//!
//! Walk the same template several times against synthetic descriptors,
//! [record](crate::record) each walk, and read the differences:
//!
//! ```text
//! base    : every class at (rows, lanes) = (r, l)
//! bump k  : base with ONE axis moved                 (2·classes of them)
//! check   : every axis moved, none of them like a bump
//!
//! for each slot, for each grid axis and each argument:
//!   equal in base and every bump      → a CONSTANT, encoded once
//!   moved                             → slope per axis from the bumps,
//!                                       then PREDICTED at `check`
//!   moved and `check` disagrees       → Fault::Unaffine { slot, at }
//! ```
//!
//! Three properties carry it. **It is the same walk** — `driver::fire::walk`
//! over the same [`Run`](crate::Run), so there is no model of what an entry
//! does anywhere in this file. **It refuses rather than guesses** — a
//! quantity that is not affine in the descriptor is named and the derivation
//! fails, which is build log 13's `schedule_shape` hash used constructively.
//! And **the axes are the old exec key**: build log 10 keyed on the per-class
//! `(rows, lanes)` vector, and that vector is exactly the coordinate system
//! the fit is written in. A component whose law is `Const` does not depend on
//! the key; if every component of every slot is `Const` or `Affine`, the key
//! is not needed at all and one recording serves every composition.
//!
//! # What the derivation does NOT cover, stated
//!
//! The walk skips a zero-row region's nodes (`driver::fire::walk` rule 1), so
//! a composition with an empty window produces FEWER slots than one without.
//! Every probe point here therefore holds every class, and the derived table
//! is the FULL composition's — which is the point rather than a limitation:
//! design §5's "all compositions live inside it" means the artifact holds
//! every launch and a fire turns the absent ones off. What turns one off is
//! its own question and it belongs to the ICB, not to this table.

use std::collections::BTreeMap;

use crate::error::{Fault, Result};
use crate::record::{Arg, Point, Recording, Slot};

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
    /// What one step does to each class's `(rows, lanes)`, for the reader.
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
    /// `base + Σ slope[k] · axis[k]` over the descriptor's own numbers. The
    /// slopes that are zero are kept, so a law's shape says which axes it
    /// reads without a second table.
    Affine {
        /// The value at the origin of the fit's coordinates.
        base: i128,
        /// One slope per descriptor axis, in [`DescriptorAbi::axes`] order.
        slope: Vec<i128>,
    },
}

impl Law {
    /// This law at one point of the descriptor's space.
    #[must_use]
    pub fn at(&self, axes: &[i128]) -> i128 {
        match self {
            Law::Const(v) => *v,
            Law::Affine { base, slope } => slope
                .iter()
                .zip(axes)
                .fold(*base, |sum, (b, x)| sum + b * x),
        }
    }

    /// Whether the number moves at all.
    #[must_use]
    pub fn varies(&self) -> bool {
        matches!(self, Law::Affine { .. })
    }

    /// Which axes this law reads.
    #[must_use]
    pub fn reads(&self) -> Vec<usize> {
        match self {
            Law::Const(_) => Vec::new(),
            Law::Affine { slope, .. } => slope
                .iter()
                .enumerate()
                .filter_map(|(k, b)| (*b != 0).then_some(k))
                .collect(),
        }
    }
}

/// One dispatch's law table: the skeleton the ICB is encoded from, and the
/// components a fire rewrites.
#[derive(Clone, Debug)]
pub struct SlotAbi {
    /// The shader point — never a variable; a slot is one entry.
    pub point: Point,
    /// The template region this dispatch stood in.
    pub region: u32,
    /// Which run of that region's window.
    pub run: u32,
    /// The base recording's slot, verbatim: what to encode at load.
    pub skeleton: Slot,
    /// Every component that moves, and its law. Ordered, so two derivations
    /// of one artifact compare.
    pub laws: Vec<(At, Law)>,
}

impl SlotAbi {
    /// How many components this slot rewrites per fire.
    #[must_use]
    pub fn rewrites(&self) -> usize {
        self.laws.len()
    }
}

/// The table `.wiki/palo/icb.md` §3 calls the binding recipe and design §2
/// calls `DescriptorAbi`: slot → which arguments and which grid axes are
/// affine in the descriptor.
#[derive(Clone, Debug)]
pub struct DescriptorAbi {
    /// The descriptor's own numbers, in the order every `slope` is written
    /// in.
    pub axes: Vec<Axis>,
    /// One entry per dispatch, in walk order.
    pub slots: Vec<SlotAbi>,
    /// The point the skeletons were recorded at, so a reader can undo the
    /// fit: `skeleton == law.at(origin)`.
    pub origin: Vec<i128>,
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
            .map(|slot| components(&slot.skeleton) - slot.rewrites())
            .sum()
    }

    /// The slots that rewrite nothing at all: encoded once, never touched.
    #[must_use]
    pub fn frozen(&self) -> usize {
        self.slots.iter().filter(|s| s.laws.is_empty()).count()
    }

    /// The census, per shader point: how many slots and how many rewrites.
    #[must_use]
    pub fn census(&self) -> Vec<(Point, usize, usize)> {
        let mut per: BTreeMap<Point, (usize, usize)> = BTreeMap::new();
        for slot in &self.slots {
            let row = per.entry(slot.point).or_default();
            row.0 += 1;
            row.1 += slot.rewrites();
        }
        per.into_iter()
            .map(|(point, (slots, laws))| (point, slots, laws))
            .collect()
    }

    /// Re-derive one slot's whole argument list at a point of the
    /// descriptor's space — what `icb::rebind` computes on the device, in
    /// host arithmetic, so a test can diff the two.
    ///
    /// # Errors
    ///
    /// None; a law that could not be fitted is not in the table.
    #[must_use]
    pub fn slot_at(&self, slot: usize, axes: &[i128]) -> Option<Slot> {
        let abi = self.slots.get(slot)?;
        let mut built = abi.skeleton.clone();
        for (at, law) in &abi.laws {
            let value = law.at(axes);
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
    /// The slots that are not one shader point at all: a kernel entry that
    /// picks its arm off the window's size hands the same slot two different
    /// pipelines at two compositions.
    ///
    /// **THIS IS THE BREAK `.wiki/palo/icb.md` §8's first bullet did not
    /// anticipate.** It reads "if a slot's pipeline cannot be rebound from
    /// the GPU, every slot's pipeline is fixed at load — which is fine,
    /// because the slot map is static and a slot is one entry". The slot map
    /// IS static; what is not static is the entry's own arm. So the question
    /// stops being whether a pipeline CAN be rebound and becomes whether it
    /// MUST be, and these are the slots that say it must.
    pub armed: Vec<Armed>,
}

/// One slot whose shader point is a function of the composition.
#[derive(Clone, Debug)]
pub struct Armed {
    /// The slot, in walk order.
    pub slot: u32,
    /// Every point seen at that slot, with the composition that produced it.
    pub points: Vec<(Point, Vec<(u32, u32)>)>,
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
        Ok(())
    }
}

/// Derive the table, refusing on the first component that does not fit.
///
/// `base` is the origin of the fit, `bumps[k]` is `base` with the `k`th
/// direction stepped and nothing else, and `check` is a point none of the
/// bumps predicted — the verification, not the fit. Every recording carries
/// its own coordinates ([`Recording::at`]).
///
/// # Errors
///
/// [`Fault::Unstructured`] when two probes did not walk the same template,
/// [`Fault::Unaffine`] for the first component no affine law predicts.
pub fn derive(
    axes: &[Axis],
    base: &Recording,
    bumps: &[Recording],
    check: &Recording,
) -> Result<DescriptorAbi> {
    let surveyed = survey(axes, base, bumps, check)?;
    if let Some(armed) = surveyed.armed.first() {
        return Err(Fault::Unstructured {
            slot: armed.slot,
            why: armed.to_string(),
        });
    }
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
/// template have no table to survey at all, so that one is still fatal.
pub fn survey(
    axes: &[Axis],
    base: &Recording,
    bumps: &[Recording],
    check: &Recording,
) -> Result<Survey> {
    if bumps.len() != axes.len() {
        return Err(Fault::Unstructured {
            slot: 0,
            why: format!(
                "the fit needs one bump per direction: {} directions, {} bumps",
                axes.len(),
                bumps.len()
            ),
        });
    }
    let origin = base.coords.clone();
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
    // A bump must step exactly its own direction, or the slope it yields is
    // the sum of two derivatives wearing one name.
    for (k, bump) in bumps.iter().enumerate() {
        let moved: Vec<usize> = bump
            .coords
            .iter()
            .zip(&origin)
            .enumerate()
            .filter_map(|(at, (there, here))| (there != here).then_some(at))
            .collect();
        if moved != vec![k] {
            return Err(Fault::Unstructured {
                slot: 0,
                why: format!(
                    "bump {k} was supposed to step `{}` alone and stepped {moved:?}",
                    axes[k]
                ),
            });
        }
    }

    let every: Vec<&Recording> = std::iter::once(base)
        .chain(bumps.iter())
        .chain(std::iter::once(check))
        .collect();
    let armed = structure(&every)?;
    let switching: std::collections::BTreeSet<u32> =
        armed.iter().map(|entry| entry.slot).collect();

    let mut unaffine = Vec::new();
    let mut slots = Vec::with_capacity(base.slots.len());
    for (index, skeleton) in base.slots.iter().enumerate() {
        // A slot that is two shader points is not one law table; it is named
        // in `armed` and left out of the fit rather than fitted against
        // another entry's argument list.
        if switching.contains(&(index as u32)) {
            slots.push(SlotAbi {
                point: skeleton.point,
                region: skeleton.region,
                run: skeleton.run,
                skeleton: skeleton.clone(),
                laws: Vec::new(),
            });
            continue;
        }
        let here = read(skeleton);
        let mut laws = Vec::new();
        for (component, (at, value)) in here.iter().enumerate() {
            let Some(value) = *value else { continue };
            match fit(axes, &origin, bumps, check, index, component, *at, value) {
                Ok(None) => {}
                Ok(Some(law)) => laws.push((*at, law)),
                Err(fault) => unaffine.push(fault),
            }
        }
        slots.push(SlotAbi {
            point: skeleton.point,
            region: skeleton.region,
            run: skeleton.run,
            skeleton: skeleton.clone(),
            laws,
        });
    }

    Ok(Survey {
        abi: DescriptorAbi {
            axes: axes.to_vec(),
            slots,
            origin,
        },
        unaffine,
        armed,
    })
}

/// One component of one slot: a constant (`None`), a law, or a refusal.
#[allow(clippy::too_many_arguments)]
fn fit(
    axes: &[Axis],
    origin: &[i128],
    bumps: &[Recording],
    check: &Recording,
    index: usize,
    component: usize,
    at: At,
    value: i128,
) -> Result<Option<Law>> {
    let point = bumps
        .first()
        .map_or_else(String::new, |bump| bump.slots[index].point.to_string());
    let mut slope = vec![0i128; axes.len()];
    let mut moves = false;
    for (k, bump) in bumps.iter().enumerate() {
        let there = read(&bump.slots[index])[component]
            .1
            .ok_or_else(|| Fault::Unstructured {
                slot: index as u32,
                why: format!("{at} carries a number at the base point and none along `{}`", axes[k]),
            })?;
        if there == value {
            continue;
        }
        moves = true;
        let run = bump.coords[k] - origin[k];
        let rise = there - value;
        if run == 0 || rise % run != 0 {
            return Err(Fault::Unaffine {
                slot: index as u32,
                point,
                at: at.to_string(),
                why: format!(
                    "stepping `{}` by {run} moved it by {rise}, which is not a whole multiple",
                    axes[k]
                ),
            });
        }
        slope[k] = rise / run;
    }
    if !moves {
        // **A CONSTANT IS A CLAIM AND IT GETS CHECKED.** A component that did
        // not move under any single step is not thereby constant: a grid axis
        // written `rows.div_ceil(TILE)` is flat across every step small enough
        // to stay inside one tile and jumps at the check point, and a fit that
        // only verified the components it had already decided were variable
        // would call that a constant and encode the wrong grid into the ICB
        // once, at load. This is the line that catches it.
        let wanted = read(&check.slots[index])[component].1.unwrap_or(i128::MIN);
        if wanted != value {
            return Err(Fault::Unaffine {
                slot: index as u32,
                point,
                at: at.to_string(),
                why: format!(
                    "it is {value} at the base point and under every single step, and \
                     {wanted} at the check point — a step this basis cannot see, which is \
                     what a tiling `div_ceil` looks like from here"
                ),
            });
        }
        return Ok(None);
    }
    // The value at the ORIGIN of the coordinates, not at the base probe: a
    // law has to be evaluable anywhere.
    let base = value - slope.iter().zip(origin).map(|(b, x)| b * x).sum::<i128>();
    let law = Law::Affine { base, slope };
    // The third point: a fit through two points always fits, and this is the
    // only line that can say it was the right shape.
    let wanted = read(&check.slots[index])[component].1.unwrap_or(i128::MIN);
    let got = law.at(&check.coords);
    if got != wanted {
        return Err(Fault::Unaffine {
            slot: index as u32,
            point,
            at: at.to_string(),
            why: format!(
                "the law fitted from the bumps predicts {got} at the check point and the walk \
                 produced {wanted}"
            ),
        });
    }
    Ok(Some(law))
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
///   rather than refused, and the report is the census.
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
    Ok(armed)
}
