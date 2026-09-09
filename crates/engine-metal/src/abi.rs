use std::collections::{BTreeMap, BTreeSet};

use model_exec::law::fit;

use crate::error::{Fault, Result};
use crate::record::{Arg, Point, Recording, Slot};

pub use model_exec::law::fit::{MAX_NUMERATOR_SCALE, MAX_TILE};
pub use model_exec::law::{At, Axis, Law, Recipe};

#[derive(Clone, Debug)]
pub struct Arm {
    pub point: Point,
    pub skeleton: Slot,
    pub laws: Vec<(At, Law)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    Only,
    Rows { at: u32 },
}

#[derive(Clone, Debug)]
pub struct SlotAbi {
    pub region: u32,
    pub run: u32,
    pub rows: Law,
    pub arms: Vec<Arm>,
    pub pick: Pick,
}

impl SlotAbi {
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

    #[must_use]
    pub fn point(&self) -> Point {
        self.arms[0].point
    }

    #[must_use]
    pub fn skeleton(&self) -> &Slot {
        &self.arms[0].skeleton
    }

    #[must_use]
    pub fn rewrites(&self) -> usize {
        self.arms
            .iter()
            .map(|arm| arm.laws.len())
            .max()
            .unwrap_or(0)
    }

    #[must_use]
    pub fn armed(&self) -> bool {
        self.arms.len() > 1
    }
}

#[derive(Clone, Debug)]
pub struct DescriptorAbi {
    pub axes: Vec<Axis>,
    pub slots: Vec<SlotAbi>,
    pub origin: Vec<i128>,
    pub origin_classes: Vec<(u32, u32)>,
    pub recipe: Vec<Recipe>,
    pub probed: Vec<(i128, i128)>,
}

impl DescriptorAbi {
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    #[must_use]
    pub fn affine(&self) -> usize {
        self.slots.iter().map(SlotAbi::rewrites).sum()
    }

    #[must_use]
    pub fn constants(&self) -> usize {
        self.slots
            .iter()
            .map(|slot| components(slot.skeleton()) - slot.arms[0].laws.len())
            .sum()
    }

    #[must_use]
    pub fn frozen(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| !s.armed() && s.arms[0].laws.is_empty())
            .count()
    }

    #[must_use]
    pub fn armed(&self) -> usize {
        self.slots.iter().filter(|s| s.armed()).count()
    }

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

    #[must_use]
    pub fn coords_of(&self, classes: &[(u32, u32)]) -> Vec<i128> {
        self.recipe.iter().map(|row| row.at(classes)).collect()
    }

    #[must_use]
    pub fn rows_at(&self, slot: usize, coords: &[i128]) -> Option<i128> {
        self.slots.get(slot)?.rows.at(coords, 0)
    }

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

fn components(slot: &Slot) -> usize {
    6 + slot.args.iter().filter(|a| a.scalar().is_some()).count()
}

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

#[derive(Clone, Debug)]
pub struct Probe {
    pub base: Recording,
    pub ladders: Vec<Vec<Recording>>,
}

#[derive(Clone, Debug)]
pub struct Probes {
    pub probes: Vec<Probe>,
    pub check: Recording,
}

impl Probes {
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

    #[must_use]
    pub fn walks(&self) -> usize {
        self.every().len()
    }
}

#[derive(Debug)]
pub struct Survey {
    pub abi: DescriptorAbi,
    pub unaffine: Vec<Fault>,
    pub armed: Vec<Armed>,
}

#[derive(Clone, Debug)]
pub struct Armed {
    pub slot: u32,
    pub points: Vec<(Point, Vec<(u32, u32)>)>,
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

pub fn derive(axes: &[Axis], probes: &Probes) -> Result<DescriptorAbi> {
    let surveyed = survey(axes, probes)?;
    match surveyed.unaffine.into_iter().next() {
        Some(fault) => Err(fault),
        None => Ok(surveyed.abi),
    }
}

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

fn fit_slot(
    axes: &[Axis],
    samples: &[&Recording],
    index: usize,
    unaffine: &mut Vec<Fault>,
) -> (SlotAbi, Option<u32>) {
    let here = samples[0].slots[index].clone();
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

    let mut points: Vec<Point> = Vec::new();
    for walk in samples {
        let point = walk.slots[index].point;
        if !points.contains(&point) {
            points.push(point);
        }
    }

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
                    read(slot)[component]
                        .1
                        .map(|v| (walk.coords.clone(), i128::from(slot.window_rows), v))
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
    let seen: BTreeSet<u32> = armed.iter().map(|entry| entry.slot).collect();
    debug_assert_eq!(seen.len(), armed.len(), "one entry per armed slot");
    Ok(armed)
}
