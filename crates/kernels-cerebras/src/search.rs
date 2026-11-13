//! The search: a model's partitions ranked by what they cost ([`crate::cost`]). From each of its
//! kernel set's pinned plans ([`crate::pins`]: the one made for the model first), it moves the keys one or
//! two at a time over a ladder of values, keeps the moves that plan, serve the positions asked for
//! and leave every image its stack room, and walks to the fastest predicted step. A candidate
//! ranks ahead of the made-for preset only where it beats it by the time model's resolution
//! ([`RESOLUTION`]). The best few are built ([`Program::build`]: the fabric and its colours) before one is returned.

use std::collections::{BTreeMap, BTreeSet};

use crate::build::STACK_ROOM;
use crate::cost::memory::{self, Anchor};
use crate::cost::time;
use crate::floorplan;
use crate::plan::{Plan, Preset, Shape};
use crate::program::Program;
use crate::target::Planned;

type R<T> = Result<T, String>;

/// The bytes the memory model may be off by: kept free past the stack room.
pub const MARGIN: f64 = 512.0;

/// The refusals the repair follows a step.
const BEAM: usize = 96;

/// The least gain (a fraction of the made-for plan's step) a candidate must predict to rank
/// ahead of that plan: the moves under it are within the time model's error (gpt-oss G2=8, 0.4%
/// on the chip, is the one it gets the wrong way: 0.8% of its step).
pub const RESOLUTION: f64 = 0.003;

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
    let set = crate::kernels::KernelSet::of_shape(shape);
    crate::compose::refuse(shape)?;
    let sig = shape.signature();
    let mut seeds = crate::pins::of(set);
    seeds.sort_by_key(|p| p.made_for != Some(sig));
    let seeds: Vec<Preset> = seeds
        .into_iter()
        .filter_map(|p| p.set(&ask.set).ok())
        .collect();
    // the made-for preset's plan: the positions it serves and the measured layer cost
    let reference = seeds
        .first()
        .filter(|p| p.made_for == Some(sig))
        .and_then(|p| floorplan::plan(shape, p).ok());
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
        shape,
        positions,
        layer,
        anchor: memory::anchor(shape),
    };
    // the made-for preset as it plans: what a candidate must beat by the resolution
    let made = seeds
        .first()
        .filter(|p| p.made_for == Some(sig))
        .and_then(|p| s.score(p))
        .filter(|c| builds(&c.planned.plan));
    let mut found: BTreeMap<String, Scored> = BTreeMap::new();
    for seed in seeds {
        if let Some(best) = s.walk(seed) {
            found.insert(key(&best.planned.preset), best);
        }
    }
    let mut all: Vec<Scored> = found.into_values().collect();
    all.sort_by(|a, b| a.ms.total_cmp(&b.ms));
    if let Some(mut m) = made {
        all.retain(|c| key(&c.planned.preset) != key(&m.planned.preset));
        let bar = m.ms * (1.0 - RESOLUTION);
        let at = all.iter().position(|c| c.ms > bar).unwrap_or(all.len());
        m.planned.preset.name = format!("search({})", m.planned.preset.name);
        all.insert(at, m);
    }
    let mut out = Vec::new();
    let mut why = Vec::new();
    for c in all {
        match Program::build(c.planned.plan.clone()) {
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
    shape: &'a Shape,
    positions: u32,
    layer: f64,
    anchor: &'static Anchor,
}

impl Search<'_> {
    /// `p`'s plan, step and room, if it plans, serves the positions and fits.
    fn score(&self, p: &Preset) -> Option<Scored> {
        let mut plan = floorplan::plan(self.shape, p).ok()?;
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
                    if floorplan::plan(self.shape, &pa).is_err() || self.score(&pa).is_none() {
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
    /// How far a plan that plans is from serving: (positions short, the memory short of the
    /// room the search keeps, the vocabulary rows short); all zero where it fits.
    fn short(&self, plan: &Plan) -> (u64, i64, u64) {
        let h = &plan.partition.head;
        let held = u64::from(h.rows.max(h.cols)) * u64::from(h.vocab_rows);
        let room = memory::room(plan, self.anchor);
        (
            u64::from(self.positions.saturating_sub(plan.contract.positions)),
            (f64::from(STACK_ROOM) + MARGIN - room).max(0.0) as i64,
            u64::from(self.shape.vocab).saturating_sub(held),
        )
    }

    /// A seed that does not plan (or fit), moved one key at a time to one that does: each move
    /// that gets the plan past the check it failed (its refusal names another check) is followed, a
    /// few deep
    /// (the first [`BEAM`] a step);
    /// from the partitions that plan but do not fit, the moves that bring them nearest a fit
    /// ([`Search::short`]) are followed.
    fn repair(&self, seed: &Preset, keys: &[String]) -> Option<(Preset, Scored)> {
        let why = |p: &Preset| match floorplan::plan(self.shape, p) {
            Ok(_) => "fit".to_string(),
            Err(e) => e,
        };
        // the check a refusal names: its text past the numbers
        let check = |e: &str| e.replace(|c: char| c.is_ascii_digit(), "");
        let mut frontier = vec![(seed.clone(), why(seed))];
        let mut seen: BTreeSet<String> = BTreeSet::new();
        let mut planned: Vec<Preset> = Vec::new();
        for _ in 0..10 {
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
                        if e == "fit" {
                            planned.push(q);
                        } else if check(&e) != check(err) && seen.insert(e.clone()) {
                            next.push((q, e));
                        }
                    }
                }
            }
            if found.is_some() {
                return found;
            }
            if next.is_empty() {
                break;
            }
            // (a beam: the first refusals reached, each a different check's or value's)
            next.truncate(BEAM);
            frontier = next;
        }
        self.climb(planned, keys)
    }

    /// From partitions that plan but do not fit, the single key moves that keep them planning
    /// and bring them nearest a fit, a step at a time, until one fits and builds.
    fn climb(&self, mut from: Vec<Preset>, keys: &[String]) -> Option<(Preset, Scored)> {
        let near = |p: &Preset| {
            floorplan::plan(self.shape, p)
                .ok()
                .map(|pl| self.short(&pl))
        };
        from.sort_by_key(|p| near(p));
        from.dedup_by_key(|p| key(p));
        let mut seen: BTreeSet<String> = BTreeSet::new();
        for start in from.into_iter().take(4) {
            let mut at = start;
            for _ in 0..40 {
                if let Some(s) = self.score(&at).filter(|s| builds(&s.planned.plan)) {
                    return Some((at, s));
                }
                let cur = near(&at)?;
                let best = keys
                    .iter()
                    .flat_map(|k| ladder(at.keys[k]).into_iter().map(move |v| (k, v)))
                    .map(|(k, v)| with(&at, k, v))
                    .filter(|q| !seen.contains(&key(q)))
                    .filter_map(|q| near(&q).map(|n| (n, q)))
                    .filter(|(n, _)| *n < cur)
                    .min_by_key(|(n, _)| *n);
                match best {
                    Some((_, q)) => {
                        seen.insert(key(&q));
                        at = q;
                    }
                    None => break,
                }
            }
        }
        None
    }
}

/// Whether `plan` builds (its blocks compose, its fabric fits the wafer, its routes and colours
/// hold).
fn builds(plan: &Plan) -> bool {
    Program::build(plan.clone()).is_ok()
}

/// `p` with `k` set to `v`.
fn with(p: &Preset, k: &str, v: f64) -> Preset {
    let mut q = p.clone();
    q.keys.insert(k.to_string(), v);
    q
}

/// The values a key moves to from `v`: its neighbours, ratios of it, small counts and the powers of
/// two around it.
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
    // the powers of two around it (the dims a hidden of a power of two splits into)
    if v > 0 {
        let lo = 1i64 << (63 - v.leading_zeros());
        out.extend([lo, lo * 2]);
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
