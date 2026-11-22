//! The simulated-annealing placer: a stream program's units and K / V columns placed anywhere
//! on the fabric, priced by the link term ([`super::cost`]) plus penalties for overlap and the
//! fabric's bounds (the program's own price reads no position). Moves translate a block, swap
//! two units, or relocate a K / V column beside its block. The winner is routed
//! ([`super::route`]) and kept only if its routed price is no more than the search's placement's.
//! Seeded and budgeted by moves a block, so deterministic; off unless asked ([`Anneal`]).

use super::{BlockKind, Problem, Solution};
use crate::compile::floor::{Coord, Rect};
use crate::compile::floorplan::stream::Stream;
use crate::compile::stream::build::Program;

type R<T> = Result<T, String>;

/// What the annealer is asked: off (the default), or moves a block (the budget is that times the
/// program's blocks), a seed, and whether units move (else the K / V columns alone).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Anneal {
    pub per_block: u64,
    pub seed: u64,
    pub units: bool,
}

impl Anneal {
    /// The annealer off: a program placed as the search placed it.
    #[must_use]
    pub fn off() -> Anneal {
        Anneal::default()
    }

    #[must_use]
    pub fn is_on(&self) -> bool {
        self.per_block > 0
    }
}

/// The penalty a PE of overlap or past the fabric (cycles).
const PENALTY: f64 = 1.0e6;
/// A block's pull toward the origin, cycles a PE (ties broken toward a compact placement).
const PULL: f64 = 1e-3;
/// Placements tried through the router at most (the best estimates found, best first).
const TRIES: usize = 8;
/// A run's safety stop (seconds): past it the run is abandoned and the search's program kept.
/// Below it a budget's result is the same on any machine.
const SAFETY: f64 = 600.0;

/// What a run did.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Report {
    /// The program's own price (cycles), placement-free.
    pub price: f64,
    /// The link term's estimate at the search's placement and at the best found.
    pub seed_estimate: f64,
    pub best_estimate: f64,
    /// The link term routed (cycles), at the search's placement and the one kept.
    pub seed_links: f64,
    pub kept_links: f64,
    pub moves: u64,
    pub accepted: u64,
    pub seconds: f64,
    /// The placement kept is not the search's.
    pub changed: bool,
    /// Why the best estimate was not kept (a route or lay that failed, a routed price no
    /// better), if it was not.
    pub refused: Option<String>,
    /// The kept placement is one the band layout cannot state (a unit off its band's row, out
    /// of path order, or a K / V column apart from its unit).
    pub beyond_bands: bool,
}

/// `p` placed by the annealer as `a` asks: the program and what the run did, or `None` where
/// it is off.
pub fn maybe(p: &Program, a: &Anneal) -> R<Option<(Program, Report)>> {
    if !a.is_on() {
        return Ok(None);
    }
    let blocks = (p.stream.units.len() + p.stream.kv.len()) as u64;
    let (q, r) = stream(p, a.per_block.saturating_mul(blocks), a.seed, a.units)?;
    eprintln!(
        "anneal: links {:.0} -> {:.0} cycles (the price {:.0}), {} moves in {:.1} s{}",
        r.seed_links,
        r.kept_links,
        r.price,
        r.moves,
        r.seconds,
        r.refused.as_ref().map_or(String::new(), |w| format!(
            "; the search's placement kept: {w}"
        ))
    );
    Ok(Some((q, r)))
}

impl Report {
    /// A run that kept the search's placement.
    fn off(price: f64, est: f64, moves: u64, seconds: f64) -> Report {
        Report {
            price,
            seed_estimate: est,
            best_estimate: est,
            seed_links: 0.0,
            kept_links: 0.0,
            moves,
            accepted: 0,
            seconds,
            changed: false,
            refused: None,
            beyond_bands: false,
        }
    }
}

/// One term of the energy: the cycles a step its blocks' distance costs.
#[derive(Clone, Copy, Debug)]
enum Term {
    /// The row stream from unit block `a` to `b` (their CX roots, the same row), once a step.
    Gap { a: usize, b: usize, wires: u32 },
    /// A unit's head root (column `col` of block `u`) to a K / V column's root (block `k`),
    /// there and back, once a step (`place_kv_hop` a hop).
    Trip {
        u: usize,
        col: u32,
        /// The head root's offset below the block's origin.
        down: u32,
        k: usize,
    },
}

/// The annealer's view of a stream: blocks' sizes, terms, origins.
struct Model {
    /// Every block's size (a unit's roots and weight PEs; a K / V column's root and PEs).
    size: Vec<(u32, u32)>,
    /// A unit's helpers below its column roots: their rectangle's offset from the block's
    /// origin and its size.
    skirt: Vec<Option<(u32, u32, u32, u32)>>,
    fixed: Vec<bool>,
    terms: Vec<Term>,
    /// The terms a block takes part in.
    of: Vec<Vec<usize>>,
    /// The units' blocks (swappable when of one size).
    units: Vec<usize>,
    /// Each attending block's K / V columns' blocks and its units' blocks (they move together).
    groups: Vec<(Vec<usize>, Vec<usize>)>,
    w: u32,
    h: u32,
}

impl Model {
    fn term(&self, t: Term, at: &[Coord]) -> f64 {
        let hop = crate::tuning::get().stream_chain_hop;
        match t {
            Term::Gap { a, b, wires } => {
                // the gap's longest wire; a next unit not right after the last on its rows turns
                // the whole bundle (a staircase), priced at four hops a wire
                let (dx, dy) = (at[a].x.abs_diff(at[b].x), at[a].y.abs_diff(at[b].y));
                let adjacent = at[a].y == at[b].y && at[b].x == at[a].x + self.size[a].0;
                let turn = if adjacent {
                    0.0
                } else {
                    4.0 * f64::from(wires)
                };
                hop * (f64::from(dx + dy) + turn)
            }
            Term::Trip { u, col, down, k } => {
                let (cx, cy) = (at[u].x + 1 + col, at[u].y + down);
                let d = cx.abs_diff(at[k].x) + cy.abs_diff(at[k].y);
                crate::tuning::get().place_kv_hop * 2.0 * f64::from(d)
            }
        }
    }

    fn rect(&self, i: usize, at: &[Coord]) -> Rect {
        Rect::new(at[i].x, at[i].y, self.size[i].0, self.size[i].1)
    }

    /// Block `i`'s rectangles: its own, and its helpers'.
    fn rects(&self, i: usize, at: &[Coord]) -> [Option<Rect>; 2] {
        [
            Some(self.rect(i, at)),
            self.skirt[i].map(|(dx, dy, w, h)| Rect::new(at[i].x + dx, at[i].y + dy, w, h)),
        ]
    }

    fn past(&self, i: usize, at: &[Coord]) -> f64 {
        let mut past = 0u32;
        for r in self.rects(i, at).into_iter().flatten() {
            past +=
                (r.x + r.w).saturating_sub(self.w) * r.h + (r.y + r.h).saturating_sub(self.h) * r.w;
        }
        PENALTY * f64::from(past)
    }

    fn overlap(&self, i: usize, at: &[Coord], skip: &[usize]) -> f64 {
        let mine = self.rects(i, at);
        let mut over = 0u64;
        for j in 0..at.len() {
            if j == i || skip.contains(&j) {
                continue;
            }
            for q in self.rects(j, at).into_iter().flatten() {
                for r in mine.iter().flatten() {
                    let ox = (r.x + r.w).min(q.x + q.w).saturating_sub(r.x.max(q.x));
                    let oy = (r.y + r.h).min(q.y + q.h).saturating_sub(r.y.max(q.y));
                    over += u64::from(ox) * u64::from(oy);
                }
            }
        }
        PENALTY * over as f64
    }

    /// A block's pull toward the fabric's origin (a compact placement among equals).
    fn pull(&self, i: usize, at: &[Coord]) -> f64 {
        PULL * f64::from(at[i].x + at[i].y)
    }

    /// The energy of `at` (the link estimate and the penalties).
    fn energy(&self, at: &[Coord]) -> f64 {
        let links: f64 = self.terms.iter().map(|&t| self.term(t, at)).sum::<f64>()
            + (0..at.len()).map(|i| self.pull(i, at)).sum::<f64>();
        let past: f64 = (0..at.len()).map(|i| self.past(i, at)).sum();
        // (each overlap counted from both sides: halved)
        let over: f64 = (0..at.len()).map(|i| self.overlap(i, at, &[])).sum::<f64>() / 2.0;
        links + past + over
    }

    /// Block `i`'s overlap at `at` with every other block (`skip` aside), found through `idx`:
    /// the same sum as [`Model::overlap`]'s, in time a block's neighbourhood.
    fn overlap_near(
        &self,
        i: usize,
        at: &[Coord],
        skip: &[usize],
        moved: &[usize],
        idx: &mut Index,
    ) -> f64 {
        let mine = self.rects(i, at);
        let mut over = 0u64;
        let mut add = |j: usize| {
            for q in self.rects(j, at).into_iter().flatten() {
                for r in mine.iter().flatten() {
                    let ox = (r.x + r.w).min(q.x + q.w).saturating_sub(r.x.max(q.x));
                    let oy = (r.y + r.h).min(q.y + q.h).saturating_sub(r.y.max(q.y));
                    over += u64::from(ox) * u64::from(oy);
                }
            }
        };
        for j in idx.near(&self.cells(i, at, idx.c)) {
            if j != i && !skip.contains(&j) && !moved.contains(&j) {
                add(j);
            }
        }
        for &j in moved {
            if j != i && !skip.contains(&j) {
                add(j);
            }
        }
        PENALTY * over as f64
    }

    /// The index cells (`c` PEs a side) block `i`'s rectangles at `at` touch.
    fn cells(&self, i: usize, at: &[Coord], c: u32) -> Vec<(u32, u32)> {
        let mut v = Vec::new();
        for r in self.rects(i, at).into_iter().flatten() {
            if r.w == 0 || r.h == 0 {
                continue;
            }
            for cy in r.y / c..=(r.y + r.h - 1) / c {
                for cx in r.x / c..=(r.x + r.w - 1) / c {
                    v.push((cx, cy));
                }
            }
        }
        v.sort_unstable();
        v.dedup();
        v
    }

    /// The energy's change if blocks `moved` take their new origins in `at` from `was` (`idx`
    /// the blocks at `was`).
    fn delta(&self, moved: &[usize], was: &[Coord], at: &[Coord], idx: &mut Index) -> f64 {
        let mut ts: Vec<usize> = moved
            .iter()
            .flat_map(|&i| self.of[i].iter().copied())
            .collect();
        ts.sort_unstable();
        ts.dedup();
        let mut d = 0.0;
        for &t in &ts {
            d += self.term(self.terms[t], at) - self.term(self.terms[t], was);
        }
        // a moved block's clashes (a pair of moved blocks' overlap counted once)
        for (n, &i) in moved.iter().enumerate() {
            // (a pair of moved blocks counted at the first of them)
            let skip = &moved[..n];
            d += self.past(i, at) - self.past(i, was);
            d += self.overlap_near(i, at, skip, moved, idx)
                - self.overlap_near(i, was, skip, moved, idx);
            d += self.pull(i, at) - self.pull(i, was);
        }
        d
    }
}

/// The blocks by the coarse cells (`c` PEs a side) their rectangles touch: a moved block's
/// overlap read from its neighbours alone.
struct Index {
    c: u32,
    cells: std::collections::HashMap<(u32, u32), Vec<usize>>,
    seen: Vec<u64>,
    epoch: u64,
}

impl Index {
    /// Cells of `c` PEs a side over the blocks at `at`.
    fn new(model: &Model, at: &[Coord], c: u32) -> Index {
        let mut x = Index {
            c,
            cells: std::collections::HashMap::new(),
            seen: vec![0; at.len()],
            epoch: 0,
        };
        for i in 0..at.len() {
            x.insert(model, i, at);
        }
        x
    }

    fn insert(&mut self, model: &Model, i: usize, at: &[Coord]) {
        for k in model.cells(i, at, self.c) {
            self.cells.entry(k).or_default().push(i);
        }
    }

    fn remove(&mut self, model: &Model, i: usize, at: &[Coord]) {
        for k in model.cells(i, at, self.c) {
            if let Some(v) = self.cells.get_mut(&k)
                && let Some(n) = v.iter().position(|&j| j == i)
            {
                v.swap_remove(n);
            }
        }
    }

    /// Block `i` from its origin in `was` to the one in `at`.
    fn relocate(&mut self, model: &Model, i: usize, was: &[Coord], at: &[Coord]) {
        self.remove(model, i, was);
        self.insert(model, i, at);
    }

    /// The blocks in `cells`, each once.
    fn near(&mut self, cells: &[(u32, u32)]) -> Vec<usize> {
        self.epoch += 1;
        let mut v = Vec::new();
        for k in cells {
            for &j in self.cells.get(k).map_or(&[][..], |v| &v[..]) {
                if self.seen[j] != self.epoch {
                    self.seen[j] = self.epoch;
                    v.push(j);
                }
            }
        }
        v
    }
}

/// The index's cell side (PEs).
const CELL: u32 = 32;

/// A seeded xorshift generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: u64) -> u64 {
        if n == 0 { 0 } else { self.next() % n }
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// `p`'s units and K / V columns annealed (`moves` moves from `seed`): the program kept (the
/// search's where nothing better routes) and the run's report.
pub fn stream(p: &Program, moves: u64, seed: u64, units: bool) -> R<(Program, Report)> {
    let t0 = std::time::Instant::now();
    let s = &p.stream;
    let w0 = s.wiring();
    let pr = super::stream::problem(s, &w0)?;
    let b = crate::compile::floorplan::stream::Bounds::system();
    let model = model(p, &pr, b, units)?;
    let start: Vec<Coord> = Solution::pinned(&pr).at;
    let price = crate::compile::cost::stream::time(p).cycles;
    let e0 = model.energy(&start);
    let (mut at, mut e) = (start.clone(), e0);
    let mut idx = Index::new(&model, &at, CELL);
    let (mut best, mut be) = (start.clone(), e0);
    let mut rng = Rng(seed.max(1).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    // the starting temperature: the mean change of a few random moves
    // (units move one at a time; K / V columns by their attending block)
    let free: Vec<usize> = model
        .units
        .iter()
        .copied()
        .filter(|&i| !model.fixed[i])
        .collect();
    if free.is_empty() && model.groups.is_empty() {
        return Err("anneal: no block to move".into());
    }
    // (moves that clash say nothing of the links' scale: left out)
    let (mut probe, mut n) = (0.0, 0u32);
    for _ in 0..256 {
        // (a unit where units move, else an attending block's K / V columns)
        let moved: Vec<usize> = if free.is_empty() {
            model.groups[rng.below(model.groups.len() as u64) as usize]
                .0
                .clone()
        } else {
            vec![free[rng.below(free.len() as u64) as usize]]
        };
        let to = jitter(&mut rng, at[moved[0]], 16, &model);
        let dd = (
            i64::from(to.x) - i64::from(at[moved[0]].x),
            i64::from(to.y) - i64::from(at[moved[0]].y),
        );
        let mut next = at.clone();
        if moved.iter().any(|&k| super::shift(at[k], dd).is_err()) {
            continue;
        }
        for &k in &moved {
            next[k] = super::shift(at[k], dd).unwrap_or(at[k]);
        }
        let d = model.delta(&moved, &at, &next, &mut idx).abs();
        if d < PENALTY / 2.0 {
            probe += d;
            n += 1;
        }
    }
    let t_start = if n > 0 { probe / f64::from(n) } else { 0.0 }
        .max(10.0 * crate::tuning::get().stream_chain_hop);
    let mut accepted = 0u64;
    let mut made = 0u64;
    // the bests found on the way (the router tries the best first, then the ones before it)
    let mut found: Vec<(f64, Vec<Coord>)> = Vec::new();
    for step in 0..moves {
        if step % 4096 == 0 && t0.elapsed().as_secs_f64() > SAFETY {
            let mut r = Report::off(price, e0, made, t0.elapsed().as_secs_f64());
            r.refused = Some(format!(
                "past the {SAFETY:.0} s safety stop: the search's placement kept"
            ));
            return Ok((p.clone(), r));
        }
        made += 1;
        let frac = step as f64 / moves as f64;
        let temp = t_start * (1e-4f64).powf(frac);
        let reach = ((1.0 - frac) * 256.0).max(1.0) as u32;
        let mut next = at.clone();
        let kind = rng.below(10);
        let moved: Vec<usize> = if kind < 6 && !free.is_empty() {
            let i = free[rng.below(free.len() as u64) as usize];
            next[i] = jitter(&mut rng, at[i], reach, &model);
            vec![i]
        } else if kind < 7 && !free.is_empty() {
            // a swap of two units of one footprint
            let i = model.units[rng.below(model.units.len() as u64) as usize];
            let same: Vec<usize> = model
                .units
                .iter()
                .copied()
                .filter(|&j| {
                    j != i
                        && model.size[j] == model.size[i]
                        && model.skirt[j] == model.skirt[i]
                        && !model.fixed[j]
                })
                .collect();
            if model.fixed[i] || same.is_empty() {
                continue;
            }
            let j = same[rng.below(same.len() as u64) as usize];
            next.swap(i, j);
            vec![i, j]
        } else {
            // an attending block's K / V columns together (their links a staircase by head):
            // moved a step, or next to one of the block's units
            if model.groups.is_empty() {
                continue;
            }
            let (all, us) = &model.groups[rng.below(model.groups.len() as u64) as usize];
            // (the block's columns together, else one of them alone)
            let one;
            let ks: &Vec<usize> = if rng.below(2) == 0 {
                all
            } else {
                one = vec![all[rng.below(all.len() as u64) as usize]];
                &one
            };
            let lead = ks[0];
            let to = if rng.below(2) == 0 {
                jitter(&mut rng, at[lead], reach, &model)
            } else {
                let u = us[rng.below(us.len() as u64) as usize];
                let (uw, uh) = model.size[u];
                match rng.below(3) {
                    0 => Coord::new(at[u].x + rng.below(u64::from(uw)) as u32, at[u].y + uh),
                    1 => Coord::new(at[u].x + uw, at[u].y + rng.below(u64::from(uh)) as u32),
                    _ => Coord::new(
                        at[u].x.saturating_sub(model.size[lead].0),
                        at[u].y + rng.below(u64::from(uh)) as u32,
                    ),
                }
            };
            let (dx, dy) = (
                i64::from(to.x) - i64::from(at[lead].x),
                i64::from(to.y) - i64::from(at[lead].y),
            );
            let mut ok = true;
            for &k in ks {
                match super::shift(at[k], (dx, dy)) {
                    Ok(c) => next[k] = c,
                    Err(_) => ok = false,
                }
            }
            if !ok {
                continue;
            }
            ks.clone()
        };
        let d = model.delta(&moved, &at, &next, &mut idx);
        if d <= 0.0 || rng.unit() < (-d / temp).exp() {
            for &k in &moved {
                idx.relocate(&model, k, &at, &next);
            }
            at = next;
            e += d;
            accepted += 1;
            if e < be - 1e-9 {
                be = e;
                best = at.clone();
                found.push((be, best.clone()));
                if found.len() > 4 * TRIES {
                    found.remove(0);
                }
            }
        }
    }
    let seed_links = super::cost::stream(p)?.cycles;
    let mut report = Report {
        price,
        seed_estimate: e0,
        best_estimate: be,
        seed_links,
        kept_links: seed_links,
        moves: made,
        accepted,
        seconds: 0.0,
        changed: false,
        refused: None,
        beyond_bands: false,
    };
    // (with no block pinned, a placement and its translates cost alike: each candidate moved
    // to the search's corner)
    if !model.fixed.iter().any(|&f| f) {
        let (x0, y0) = (
            start.iter().map(|c| c.x).min().unwrap_or(0),
            start.iter().map(|c| c.y).min().unwrap_or(0),
        );
        for (_, c) in &mut found {
            let (mx, my) = (
                c.iter().map(|c| c.x).min().unwrap_or(0),
                c.iter().map(|c| c.y).min().unwrap_or(0),
            );
            for p in c.iter_mut() {
                *p = Coord::new(p.x - mx + x0, p.y - my + y0);
            }
        }
    }
    let _ = &best;
    // the candidates, best first, each a placement distinct from the search's
    found.reverse();
    found.dedup_by(|a, b| a.1 == b.1);
    let mut kept = None;
    let mut why = Vec::new();
    for (est, cand) in found
        .iter()
        .filter(|(e, c)| *e < e0 - 1e-9 && *c != start)
        .take(TRIES)
    {
        // laid as the emitter lays it: a placement whose blocks share a PE is refused here
        match placed(p, &pr, cand, b)
            .and_then(|q| crate::compile::emit::stream::emit(&q).map(|_| q))
        {
            Ok(q) => {
                let l = super::cost::stream(&q)?.cycles;
                if l <= seed_links {
                    report.best_estimate = *est;
                    report.kept_links = l;
                    report.changed = true;
                    report.beyond_bands = beyond_bands(&q.stream);
                    kept = Some(q);
                    break;
                }
                why.push(format!("routed {l:.0} > the search's {seed_links:.0}"));
            }
            Err(e) => why.push(format!(
                "route / lay: {}",
                e.chars().take(120).collect::<String>()
            )),
        }
    }
    if kept.is_none() {
        report.refused = Some(if why.is_empty() {
            "no placement estimated cheaper".into()
        } else {
            why.join("; ")
        });
    }
    report.seconds = t0.elapsed().as_secs_f64();
    Ok((kept.unwrap_or_else(|| p.clone()), report))
}

/// A block's origin moved by up to `reach` each way, on the fabric.
fn jitter(rng: &mut Rng, c: Coord, reach: u32, m: &Model) -> Coord {
    let r = i64::from(reach);
    let dx = rng.below(2 * reach as u64 + 1) as i64 - r;
    let dy = rng.below(2 * reach as u64 + 1) as i64 - r;
    let x = (i64::from(c.x) + dx).clamp(0, i64::from(m.w));
    let y = (i64::from(c.y) + dy).clamp(1, i64::from(m.h));
    Coord::new(x as u32, y as u32)
}

/// The annealer's model of `p` over `pr`'s blocks within bounds `b`.
fn model(
    p: &Program,
    pr: &Problem,
    b: crate::compile::floorplan::stream::Bounds,
    units_free: bool,
) -> R<Model> {
    let s = &p.stream;
    let n = pr.blocks.len();
    let mut size = Vec::with_capacity(n);
    let mut skirt = Vec::with_capacity(n);
    let mut fixed = vec![false; n];
    let mut unit_block = vec![usize::MAX; s.units.len()];
    let mut kv_block = vec![usize::MAX; s.kv.len()];
    for (i, blk) in pr.blocks.iter().enumerate() {
        match blk.kind {
            BlockKind::StreamUnit(u) => {
                let su = &s.units[u];
                // (its helpers below its column roots in its footprint)
                size.push((blk.rect.w, blk.rect.h));
                // (pitch 1: the rest columns after the heads; else spread among them, the
                // K / V roots below the helpers' rows)
                let heads = su.heads();
                skirt.push((su.helpers > 0).then(|| {
                    if su.pitch <= 1 {
                        (1 + heads, blk.rect.h, su.cols() - heads, su.helpers)
                    } else {
                        (1, blk.rect.h, su.cols(), su.helpers)
                    }
                }));
                unit_block[u] = i;
                // the lane prologue's unit: the lane bus runs from it along row 0
                fixed[i] = !units_free || s.lanes.as_ref().is_some_and(|l| l.unit == u);
            }
            BlockKind::KvColumn(k) => {
                size.push((blk.rect.w, blk.rect.h));
                skirt.push(None);
                kv_block[k] = i;
            }
            _ => return Err(format!("anneal: a {:?} block in a stream", blk.kind)),
        }
    }
    let mut terms = Vec::new();
    for pair in s.path.windows(2) {
        terms.push(Term::Gap {
            a: unit_block[pair[0]],
            b: unit_block[pair[1]],
            wires: s.py,
        });
    }
    for (k, kc) in s.kv.iter().enumerate() {
        for &u in &kc.units {
            terms.push(Term::Trip {
                u: unit_block[u],
                col: s.units[u].col(kc.head),
                down: s.units[u].rows(),
                k: kv_block[k],
            });
        }
    }
    let mut of = vec![Vec::new(); n];
    for (t, term) in terms.iter().enumerate() {
        match *term {
            Term::Gap { a, b, .. } => {
                of[a].push(t);
                of[b].push(t);
            }
            Term::Trip { u, k, .. } => {
                of[u].push(t);
                of[k].push(t);
            }
        }
    }
    let units = unit_block
        .iter()
        .copied()
        .filter(|&i| i != usize::MAX)
        .collect();
    let mut groups: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
    for (k, kc) in s.kv.iter().enumerate() {
        let us: Vec<usize> = kc.units.iter().map(|&u| unit_block[u]).collect();
        match groups.iter_mut().find(|(_, g)| *g == us) {
            Some((ks, _)) => ks.push(kv_block[k]),
            None => groups.push((vec![kv_block[k]], us)),
        }
    }
    Ok(Model {
        groups,
        size,
        skirt,
        fixed,
        terms,
        of,
        units,
        w: b.w,
        h: b.h,
    })
}

/// `p` with its blocks at `at` (origins over `pr`'s blocks): every K / V column explicit, the
/// row stream's wires and the K / V links routed between their ends.
fn placed(
    p: &Program,
    pr: &Problem,
    at: &[Coord],
    b: crate::compile::floorplan::stream::Bounds,
) -> R<Program> {
    let s = &p.stream;
    let mut sol = Solution::pinned(pr);
    sol.at = at.to_vec();
    let (mut w, mut h) = (0u32, 0u32);
    for (i, blk) in pr.blocks.iter().enumerate() {
        let r = sol.rect(pr, i);
        let extra = match blk.kind {
            BlockKind::StreamUnit(u) => s.units[u].helpers,
            _ => 0,
        };
        w = w.max(r.x + r.w);
        h = h.max(r.y + r.h + extra);
    }
    if w > b.w || h > b.h {
        return Err(format!(
            "anneal: {w} x {h} past the {} x {} fabric",
            b.w, b.h
        ));
    }
    sol.w = w.max(s.w.min(b.w));
    sol.h = h.max(1);
    let mut moved = super::stream::stream(pr, s, &sol)?;
    // a K / V column whose block's units moved apart: at its root as placed, its links routed
    for k in 0..moved.kv.len() {
        let us = &moved.kv[k].units;
        let d = |u: usize| {
            (
                i64::from(moved.units[u].x) - i64::from(s.units[u].x),
                i64::from(moved.units[u].y) - i64::from(s.units[u].y),
            )
        };
        if moved.kv[k].at.is_none() && us.iter().any(|&u| d(u) != d(us[0])) {
            let root = moved.kv[k].root(&moved);
            moved.kv[k] = moved.kv[k].at(root);
        }
    }
    // (the row stream's gaps and the K / V links laid by the bundle router, wherever the
    // blocks sit)
    super::stream::relay(p, moved)
}

/// Whether `s`'s placement is beyond the band layout: a unit off its band's top, out of path
/// order left to right in a band, or a K / V column apart from its block's first unit.
fn beyond_bands(s: &Stream) -> bool {
    let off_order = s.path.windows(2).any(|w| {
        let (a, b) = (&s.units[w[0]], &s.units[w[1]]);
        a.y == b.y && b.x < a.x
    });
    // (a column takes a root of its own only where it moved apart from its first unit)
    let kv_apart = s.kv.iter().any(|k| k.at.is_some());
    off_order
        || kv_apart
        || s.units
            .windows(2)
            .any(|w| w[0].y != w[1].y && w[0].x == 0 && w[1].x == 0)
}

/// Co-resident readings (blocks with no nets, each a choice of [`super::pack::Shape`]s) annealed
/// within `w` x `h` from [`super::pack::pack_shapes`]'s packing: moves translate a reading or
/// change its shape. Returns each reading's shape, the placement and the seed's and kept
/// energies; the seed's where nothing is cheaper.
pub fn pack(
    alts: &[Vec<super::pack::Shape>],
    w: u32,
    h: u32,
    moves: u64,
    seed: u64,
) -> Result<(Vec<usize>, Solution, f64, f64), super::pack::Miss> {
    let (pick0, sol0) = super::pack::pack_shapes(alts, w, h)?;
    let n = alts.len();
    let energy = |pick: &[usize], at: &[Coord]| -> f64 {
        let mut e: f64 = pick
            .iter()
            .enumerate()
            .map(|(i, &k)| alts[i][k].price)
            .sum();
        let rect = |i: usize| {
            let sh = alts[i][pick[i]];
            Rect::new(at[i].x, at[i].y, sh.w, sh.h)
        };
        let (mut bw, mut bh) = (0u32, 0u32);
        for i in 0..n {
            let r = rect(i);
            bw = bw.max(r.x + r.w);
            bh = bh.max(r.y + r.h);
            e += PENALTY
                * f64::from(
                    (r.x + r.w).saturating_sub(w) * r.h + (r.y + r.h).saturating_sub(h) * r.w,
                );
            for j in i + 1..n {
                let q = rect(j);
                let ox = (r.x + r.w).min(q.x + q.w).saturating_sub(r.x.max(q.x));
                let oy = (r.y + r.h).min(q.y + q.h).saturating_sub(r.y.max(q.y));
                e += PENALTY * f64::from(ox) * f64::from(oy);
            }
        }
        e + PULL * f64::from(bw) * f64::from(bh)
    };
    let e0 = energy(&pick0, &sol0.at);
    let (mut pick, mut at, mut e) = (pick0.clone(), sol0.at.clone(), e0);
    let (mut bp, mut ba, mut be) = (pick.clone(), at.clone(), e);
    let mut rng = Rng(seed.max(1).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let t_start = 1.0e3;
    for step in 0..moves {
        let temp = t_start * (1e-6f64).powf(step as f64 / moves.max(1) as f64);
        let i = rng.below(n as u64) as usize;
        let (mut np, mut na) = (pick.clone(), at.clone());
        if alts[i].len() > 1 && rng.below(4) == 0 {
            np[i] = rng.below(alts[i].len() as u64) as usize;
        } else {
            let reach = 64u32;
            let dx = rng.below(2 * u64::from(reach) + 1) as i64 - i64::from(reach);
            let dy = rng.below(2 * u64::from(reach) + 1) as i64 - i64::from(reach);
            na[i] = Coord::new(
                (i64::from(at[i].x) + dx).clamp(0, i64::from(w)) as u32,
                (i64::from(at[i].y) + dy).clamp(0, i64::from(h)) as u32,
            );
        }
        let ne = energy(&np, &na);
        if ne <= e || rng.unit() < (-(ne - e) / temp).exp() {
            (pick, at, e) = (np, na, ne);
            if e < be - 1e-9 {
                (bp, ba, be) = (pick.clone(), at.clone(), e);
            }
        }
    }
    if be >= e0 - 1e-9 {
        return Ok((pick0, sol0, e0, e0));
    }
    let (bw, bh) = (0..n).fold((0, 0), |(x, y), i| {
        let sh = alts[i][bp[i]];
        (x.max(ba[i].x + sh.w), y.max(ba[i].y + sh.h))
    });
    Ok((
        bp,
        Solution {
            w: bw,
            h: bh,
            at: ba,
        },
        e0,
        be,
    ))
}

#[cfg(test)]
mod tests {
    use crate::compile::stream::build::Program;

    fn decode_units(model: &str) -> Program {
        let t = poem_compiler::catalog::repository()
            .parse(model)
            .unwrap_or_else(|| panic!("{model}"))
            .trace(poem_ir::Platform::Cerebras);
        let classes = poem_compiler::spatial::layer::stream::decode_classes(&t, 1);
        let mut s = crate::compile::stream::shape(&t, "decode", &classes).unwrap();
        s.cache = Some(128);
        crate::compile::stream::build::program_py(
            &t,
            s,
            true,
            None,
            &[],
            None,
            None,
            None,
            false,
            crate::compile::floorplan::stream::Bounds::system(),
        )
        .unwrap()
    }

    /// The index's change of energy is the whole energy's, every kind of move: a unit,
    /// a swap, an attending block's K / V columns.
    #[test]
    fn the_indexed_delta_is_the_energys_change() {
        let p = decode_units("qwen3-0.6b-bf16-kv-bf16-l2");
        let s = &p.stream;
        let pr = super::super::stream::problem(s, &s.wiring()).unwrap();
        let model = super::model(
            &p,
            &pr,
            crate::compile::floorplan::stream::Bounds::system(),
            true,
        )
        .unwrap();
        let mut at = super::Solution::pinned(&pr).at;
        let mut idx = super::Index::new(&model, &at, super::CELL);
        let mut rng = super::Rng(12345);
        for step in 0..400 {
            let mut next = at.clone();
            let moved: Vec<usize> = match step % 3 {
                0 => {
                    let i = model.units[rng.below(model.units.len() as u64) as usize];
                    if model.fixed[i] {
                        continue;
                    }
                    next[i] = super::jitter(&mut rng, at[i], 64, &model);
                    vec![i]
                }
                1 => {
                    let (i, j) = (
                        model.units[rng.below(model.units.len() as u64) as usize],
                        model.units[rng.below(model.units.len() as u64) as usize],
                    );
                    if i == j || model.fixed[i] || model.fixed[j] {
                        continue;
                    }
                    next.swap(i, j);
                    vec![i, j]
                }
                _ => {
                    let ks = &model.groups[rng.below(model.groups.len() as u64) as usize].0;
                    let d = (rng.below(41) as i64 - 20, rng.below(41) as i64 - 20);
                    if ks.iter().any(|&k| super::super::shift(at[k], d).is_err()) {
                        continue;
                    }
                    for &k in ks {
                        next[k] = super::super::shift(at[k], d).unwrap();
                    }
                    ks.clone()
                }
            };
            let d = model.delta(&moved, &at, &next, &mut idx);
            let full = model.energy(&next) - model.energy(&at);
            assert!(
                (d - full).abs() <= 1e-6 * full.abs().max(1.0),
                "step {step}: {d} vs {full}"
            );
            // (accept every other one: the index followed)
            if step % 2 == 0 {
                for &k in &moved {
                    idx.relocate(&model, k, &at, &next);
                }
                at = next;
            }
        }
    }

    /// qwen3-sim's decode units annealed: the same seed gives the same placement, the kept one is
    /// priced no more than the search's and passes the check with no new kind of warning; the
    /// search's placement, kept, is the program byte for byte.
    #[test]
    fn the_annealer_keeps_a_cheaper_placement_that_checks() {
        let p = decode_units("qwen3-sim-bf16-kv-bf16");
        let (a, ra) = super::stream(&p, 200_000, 7, false).unwrap();
        let (b, rb) = super::stream(&p, 200_000, 7, false).unwrap();
        assert_eq!(a.stream, b.stream, "deterministic");
        assert_eq!(ra.kept_links, rb.kept_links);
        assert!(ra.kept_links <= ra.seed_links, "{ra:?}");
        assert!(ra.changed, "{ra:?}");
        let base = crate::compile::emit::stream::emit(&p).unwrap();
        let base = crate::compile::check::placed(&base.placed, &base.files, None);
        let e = crate::compile::emit::stream::emit(&a).unwrap();
        let rep = crate::compile::check::placed(&e.placed, &e.files, None);
        assert!(rep.errors.is_empty(), "{:?}", rep.errors);
        let kind = |w: &String| w.replace(|c: char| c.is_ascii_digit(), "");
        let had: std::collections::BTreeSet<String> = base.warnings.iter().map(kind).collect();
        for w in &rep.warnings {
            assert!(had.contains(&kind(w)), "a new warning: {w}");
        }
        eprintln!("qwen3-sim: {ra:?}; {}", rep.line());
        // no moves: the search's program, its files byte for byte
        let (c, rc) = super::stream(&p, 0, 7, false).unwrap();
        assert!(!rc.changed);
        let (x, y) = (
            crate::compile::emit::stream::emit(&p).unwrap(),
            crate::compile::emit::stream::emit(&c).unwrap(),
        );
        assert_eq!(x.files, y.files);
        // off unless asked
        assert!(super::maybe(&p, &super::Anneal::off()).unwrap().is_none());
        // asked: a budget of moves a block, the same run on any machine
        let ask = super::Anneal {
            per_block: 100,
            seed: 7,
            units: false,
        };
        let (m, rm) = super::maybe(&p, &ask).unwrap().unwrap();
        let blocks = (p.stream.units.len() + p.stream.kv.len()) as u64;
        assert_eq!(rm.moves, 100 * blocks);
        let (n, _) = super::maybe(&p, &ask).unwrap().unwrap();
        assert_eq!(
            crate::compile::emit::stream::emit(&m).unwrap().files,
            crate::compile::emit::stream::emit(&n).unwrap().files
        );
    }

    /// Co-resident readings (klein's text, denoise and VAE at 64 rows) annealed from
    /// `pack_shapes`'s packing: within the system's fabric, apart, no dearer.
    #[test]
    fn co_resident_readings_anneal_within_the_fabric() {
        let sh = |w, h| vec![super::super::pack::Shape { w, h, price: 0.0 }];
        let alts = [sh(256, 877), sh(384, 1014), sh(130, 148)];
        let (_, s, e0, e) = super::pack(&alts, 755, 1170, 50_000, 3).unwrap();
        assert!(e <= e0);
        assert!(s.w <= 755 && s.h <= 1170, "{s:?}");
        eprintln!(
            "klein co-resident: {} x {}, energy {e0:.1} -> {e:.1}",
            s.w, s.h
        );
    }
}
