//! The search: a model's partitions ranked by what they cost ([`crate::compile::cost`]). From the layer
//! graph's fastest candidates ([`crate::compile::cold`]) it moves the keys one or two at a time, keeps
//! the moves that plan, serve and fit, and walks to the fastest predicted step; the best few are
//! built before one is returned.

use std::collections::{BTreeMap, BTreeSet};

use crate::compile::build::STACK_ROOM;
use crate::compile::cost::memory::{self, Anchor};
use crate::compile::cost::time;
use crate::compile::floorplan;
use crate::compile::plan::{Plan, Preset, Shape};
use crate::compile::program::Program;
use crate::compile::target::Planned;

type R<T> = Result<T, String>;

/// The bytes the memory model may be off by: kept free past the stack room.
pub const MARGIN: f64 = 512.0;

/// The refusals the repair follows a step.
const BEAM: usize = 96;

/// A cold search's starting points, and at most this many of one unit count.
const COLD_SEEDS: usize = 8;
const COLD_EACH: usize = 2;

/// Keys that are not the partition's (the contract's levels and loop, the head's scale).
const FIXED: &[&str] = &["CTX", "LOOP", "NLV", "HSCALE", "RLV", "RBITS", "CAP", "WL"];

/// A partition the search kept: its plan and its predicted step (ms).
#[derive(Clone, Debug)]
pub struct Scored {
    pub planned: Planned,
    pub ms: f64,
}

/// What a search is asked: the positions to serve (else the most it finds up to 4096) and overrides set over every
/// preset.
#[derive(Clone, Debug, Default)]
pub struct Asked {
    pub positions: Option<u32>,
    pub set: String,
}

/// The positions a search serves where none are asked: the most of these a partition holds.
const POSITIONS: u32 = 4096;
const POSITIONS_STEP: u32 = 512;

/// The candidates `shape` searched to, fastest first (at most `keep`), each built: at the
/// positions asked, else at the most up to [`POSITIONS`] (a [`POSITIONS_STEP`] less at a time)
/// some partition serves.
pub fn search(shape: &Shape, ask: &Asked, keep: usize) -> R<Vec<Scored>> {
    if let Some(n) = ask.positions {
        return search_at(shape, ask, n, keep);
    }
    let first = search_at(shape, ask, POSITIONS, keep);
    let mut n = POSITIONS;
    while first.is_err() && n > POSITIONS_STEP {
        n -= POSITIONS_STEP;
        if let Ok(found) = search_at(shape, ask, n, keep) {
            return Ok(found);
        }
    }
    first
}

fn search_at(shape: &Shape, ask: &Asked, positions: u32, keep: usize) -> R<Vec<Scored>> {
    crate::compile::compose::refuse(shape)?;
    let layer = time::kinds_layer(shape);
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
    // the starting points: the layer graph's candidates that plan and fit ([`crate::compile::cold`]),
    // the fastest few (at most [`COLD_EACH`] of a unit count)
    let mut c: Vec<Scored> = crate::compile::cold::candidates(shape, positions)
        .into_iter()
        .filter_map(|p| p.set(&ask.set).ok())
        .filter_map(|p| s.score(&p))
        .collect();
    c.sort_by(|a, b| a.ms.total_cmp(&b.ms));
    let mut seeds: Vec<Preset> = Vec::new();
    let mut units: BTreeMap<u32, usize> = BTreeMap::new();
    for x in c {
        let n = units.entry(x.planned.plan.partition.units).or_default();
        if *n < COLD_EACH && builds(&x.planned.plan) {
            *n += 1;
            seeds.push(x.planned.preset);
        }
        if seeds.len() >= COLD_SEEDS {
            break;
        }
    }
    let mut found: BTreeMap<String, Scored> = BTreeMap::new();
    let mut keep_best = |best: Scored| {
        found
            .entry(format!(
                "{}|{}",
                key(&best.planned.preset),
                best.planned.preset.levers
            ))
            .and_modify(|c| {
                if best.ms < c.ms {
                    *c = best.clone();
                }
            })
            .or_insert(best);
    };
    for seed in seeds {
        if let Some(best) = s.walk(seed) {
            keep_best(best);
        }
    }
    let mut all: Vec<Scored> = found.into_values().collect();
    all.sort_by(|a, b| a.ms.total_cmp(&b.ms));
    let mut out = Vec::new();
    let mut why = Vec::new();
    for c in all {
        // a partition builds where its program does and every PE's words do (a head's turns
        // where its roles take them: what the emitter reads, checked here, not after the pick)
        match Program::build(c.planned.plan.clone())
            .and_then(|p| crate::compile::emit::rcfg::all(&p).map(|_| ()))
        {
            Ok(()) => out.push(c),
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

    /// A seed that does not plan (or fit), moved one key at a time to one that does: moves that
    /// get past the failed check are followed a few deep ([`BEAM`] a step); partitions that plan
    /// but do not fit follow the moves that bring them nearest a fit ([`Search::short`]).
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
