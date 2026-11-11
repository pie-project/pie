//! The search: a model's partitions ranked by what they cost ([`crate::cost`]). From each of the
//! serving family's presets (the one made for the model first), it moves the preset's keys one or
//! two at a time over a ladder of values, keeps the moves that plan, serve the positions asked for
//! and leave every image its stack room, and walks to the fastest predicted step. The best few are
//! built ([`Program::build`]: the fabric and its colours) before one is returned.

use std::collections::{BTreeMap, BTreeSet};

use crate::build::STACK_ROOM;
use crate::cost::memory::{self, Anchor};
use crate::cost::time;
use crate::family::{self, Family};
use crate::plan::{Plan, Preset, Shape};
use crate::program::Program;
use crate::target::Planned;

type R<T> = Result<T, String>;

/// The bytes the memory model may be off by: kept free past the stack room.
pub const MARGIN: f64 = 512.0;

/// Keys that are not the partition's (the contract's levels and loop, the head's scale).
const FIXED: &[&str] = &["CTX", "LOOP", "NLV", "HSCALE", "RLV", "RBITS", "CAP", "WL"];

/// A partition the search kept: its plan, its predicted step (ms) and least stack room (B).
#[derive(Clone, Debug)]
pub struct Scored {
    pub planned: Planned,
    pub ms: f64,
    pub room: f64,
}

/// What a search is asked: the positions to serve (else the made-for preset's, else 4096) and
/// overrides set over every preset.
#[derive(Clone, Debug, Default)]
pub struct Asked {
    pub positions: Option<u32>,
    pub set: String,
}

/// The candidates `shape` searched to, fastest first (at most `keep`), each built.
pub fn search(shape: &Shape, ask: &Asked, keep: usize) -> R<Vec<Scored>> {
    let fam = family::of(family::serving(shape));
    fam.runs(shape)?;
    let sig = shape.signature();
    let mut seeds = fam.presets();
    seeds.sort_by_key(|p| p.made_for != Some(sig));
    let seeds: Vec<Preset> = seeds
        .into_iter()
        .filter_map(|p| p.set(&ask.set).ok())
        .collect();
    // the made-for preset's plan: the positions it serves and the measured layer cost
    let reference = seeds
        .first()
        .filter(|p| p.made_for == Some(sig))
        .and_then(|p| fam.plan(shape, p).ok());
    let positions = ask
        .positions
        .or(reference.as_ref().map(|p| p.contract.positions))
        .unwrap_or(4096);
    let layer = time::layer(shape, reference.as_ref());
    let pinned: Vec<String> = ask
        .set
        .split(',')
        .filter_map(|kv| kv.split_once('=').map(|(k, _)| k.to_string()))
        .collect();
    let s = Search {
        pinned,
        fam: fam.as_ref(),
        shape,
        positions,
        layer,
        anchor: memory::anchor(shape),
    };
    let mut found: BTreeMap<String, Scored> = BTreeMap::new();
    for seed in seeds {
        if let Some(best) = s.walk(seed) {
            found.insert(key(&best.planned.preset), best);
        }
    }
    let mut all: Vec<Scored> = found.into_values().collect();
    all.sort_by(|a, b| a.ms.total_cmp(&b.ms));
    let mut out = Vec::new();
    let mut why = Vec::new();
    for c in all {
        match Program::build(family::of(c.planned.plan.family), c.planned.plan.clone()) {
            Ok(_) => out.push(c),
            Err(e) => why.push(format!("{}: {e}", c.planned.preset.name)),
        }
        if out.len() == keep {
            break;
        }
    }
    if out.is_empty() {
        return Err(format!(
            "the search found no partition that builds{}",
            if why.is_empty() {
                String::new()
            } else {
                format!(": {}", why.join("; "))
            }
        ));
    }
    Ok(out)
}

struct Search<'a> {
    /// The keys the ask set: not moved.
    pinned: Vec<String>,
    fam: &'a dyn Family,
    shape: &'a Shape,
    positions: u32,
    layer: f64,
    anchor: &'static Anchor,
}

impl Search<'_> {
    /// `p`'s plan, step and room, if it plans, serves the positions and fits.
    fn score(&self, p: &Preset) -> Option<Scored> {
        let mut plan = self.fam.plan(self.shape, p).ok()?;
        if plan.contract.positions < self.positions {
            return None;
        }
        plan.contract.positions = self.positions;
        // the head's rows (or its band's columns) hold every vocabulary row
        let h = &plan.partition.head;
        if u64::from(h.rows.max(h.cols)) * u64::from(h.vocab_rows) < u64::from(self.shape.vocab) {
            return None;
        }
        let room = memory::room(&plan, self.anchor);
        if room < f64::from(STACK_ROOM) + MARGIN {
            return None;
        }
        Some(Scored {
            ms: time::step(&plan, self.layer),
            room,
            planned: Planned {
                plan,
                preset: p.clone(),
            },
        })
    }

    /// From `seed`, the fastest partition single and paired key moves reach (`None`: none
    /// plans).
    fn walk(&self, seed: Preset) -> Option<Scored> {
        let keys: Vec<String> = seed
            .keys
            .keys()
            .filter(|k| !FIXED.contains(&k.as_str()) && !self.pinned.contains(k))
            .cloned()
            .collect();
        let mut best = self.score(&seed).filter(|s| builds(&s.planned.plan));
        let mut at = seed;
        if best.is_none() {
            let (p, s) = self.repair(&at, &keys)?;
            at = p;
            best = Some(s);
        }
        let mut seen: BTreeSet<String> = BTreeSet::new();
        loop {
            let mut next: Option<(Preset, Scored)> = None;
            let better =
                |s: &Scored, cur: &Option<Scored>| cur.as_ref().is_none_or(|b| s.ms < b.ms - 1e-9);
            let mut consider = |p: Preset, next: &mut Option<(Preset, Scored)>| {
                if !seen.insert(key(&p)) {
                    return;
                }
                if let Some(s) = self.score(&p) {
                    let cur = next.as_ref().map(|(_, s)| s.clone()).or(best.clone());
                    if better(&s, &cur) && builds(&s.planned.plan) {
                        *next = Some((p, s));
                    }
                }
            };
            for a in &keys {
                for va in ladder(at.keys[a]) {
                    let pa = with(&at, a, va);
                    consider(pa.clone(), &mut next);
                    // a move another key must follow (the positions, the units' grid)
                    if self.fam.plan(self.shape, &pa).is_err() || self.score(&pa).is_none() {
                        for b in keys.iter().filter(|b| *b > a) {
                            for vb in ladder(at.keys[b]) {
                                consider(with(&pa, b, vb), &mut next);
                            }
                        }
                    }
                }
            }
            match next {
                Some((p, s)) => {
                    at = p;
                    best = Some(s);
                }
                None => break,
            }
        }
        best.map(|mut b| {
            b.planned.preset.name = format!("search({})", b.planned.preset.name);
            b
        })
    }
}

impl Search<'_> {
    /// A seed that does not plan (or fit), moved one key at a time to one that does: each move
    /// that gets the plan past the check it failed (its refusal changes) is followed, a few deep.
    fn repair(&self, seed: &Preset, keys: &[String]) -> Option<(Preset, Scored)> {
        let why = |p: &Preset| match self.fam.plan(self.shape, p) {
            Ok(_) => "fit".to_string(),
            Err(e) => e,
        };
        let mut frontier = vec![(seed.clone(), why(seed))];
        let mut seen: BTreeSet<String> = BTreeSet::new();
        for _ in 0..6 {
            let mut next = Vec::new();
            let mut found: Option<(Preset, Scored)> = None;
            for (p, err) in &frontier {
                for k in keys {
                    for v in ladder(p.keys[k]) {
                        let q = with(p, k, v);
                        if let Some(s) = self.score(&q) {
                            if found.as_ref().is_none_or(|(_, b)| s.ms < b.ms)
                                && builds(&s.planned.plan)
                            {
                                found = Some((q, s));
                            }
                            continue;
                        }
                        let e = why(&q);
                        if e != *err && seen.insert(e.clone()) {
                            next.push((q, e));
                        }
                    }
                }
            }
            if found.is_some() {
                return found;
            }
            if next.is_empty() {
                return None;
            }
            frontier = next;
        }
        None
    }
}

/// Whether `plan` builds (its blocks compose, its fabric fits the wafer, its routes and colours
/// hold).
fn builds(plan: &Plan) -> bool {
    Program::build(family::of(plan.family), plan.clone()).is_ok()
}

/// `p` with `k` set to `v`.
fn with(p: &Preset, k: &str, v: f64) -> Preset {
    let mut q = p.clone();
    q.keys.insert(k.to_string(), v);
    q
}

/// The values a key moves to from `v`: its neighbours, ratios of it, and small counts.
fn ladder(v: f64) -> Vec<f64> {
    let mut out = BTreeSet::new();
    let v = v.round() as i64;
    for d in [-4, -2, -1, 1, 2, 4] {
        out.insert(v + d);
    }
    for (n, d) in [
        (1, 2),
        (2, 3),
        (3, 4),
        (7, 8),
        (9, 8),
        (5, 4),
        (4, 3),
        (3, 2),
        (2, 1),
    ] {
        out.insert((v * n + d / 2) / d);
    }
    if v <= 16 {
        out.extend(1..=16);
    }
    out.into_iter()
        .filter(|&x| (1..=4096).contains(&x) && x != v)
        .map(|x| x as f64)
        .collect()
}

/// A preset's keys as text (what tells two candidates apart).
fn key(p: &Preset) -> String {
    p.keys
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// `plan`'s keys that differ from `preset`'s, as `K=V` (for reports).
#[must_use]
pub fn moved(from: &Preset, to: &Preset) -> String {
    to.keys
        .iter()
        .filter(|(k, v)| from.keys.get(*k) != Some(v))
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(",")
}

impl Scored {
    /// The plan.
    #[must_use]
    pub fn plan(&self) -> &Plan {
        &self.planned.plan
    }
}
