//! The router: a block states a link between blocks it does not lay itself as a [`Path`] of
//! waypoints (consecutive points not adjacent: its sender's ramp, any PEs it must pass, its
//! receiver's ramp); [`fill`] routes each gap over the fabric by negotiated congestion
//! (PathFinder): every wire a shortest path, a PE's use past what it holds priced up a round at
//! a time (present and history) until no PE carries one class twice or more routes than it
//! holds. A program with no such path is untouched.
//!
//! What a PE holds: a class at most once (one route a colour), and past the links already laid
//! there at most its kernel's route table (a weight PE's 16, a control or head PE's 12, a
//! relay's 4). An idle PE a wire crosses runs the relay kernel (route-only) once routed.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap};

use crate::floor::{Coord, Floor, Rect};
use crate::role::{Kind, PeRole, Place};
use crate::wiring::{Class, Link, Wiring};

type R<T> = Result<T, String>;

/// Rounds of negotiation at most.
const ROUNDS: usize = 40;
/// A step's cost, a turn's, an idle PE's (it takes a relay kernel).
const STEP: u64 = 1000;
const TURN: u64 = 1;
const IDLE: u64 = 3000;
/// The search window's margin past the wire's ends (widened where nothing is found).
const MARGIN: u32 = 24;

/// Routes a PE of `role` holds at most (its kernel's route table).
fn slots(role: &PeRole) -> u32 {
    if role.is_idle() || role.kind == Kind::RELAY {
        4
    } else if role.kind.name.starts_with('w') {
        16
    } else {
        12
    }
}

/// A wire to route: its link, the gap's two waypoints' indices.
struct Wire {
    link: usize,
    class: Class,
    from: Coord,
    to: Coord,
}

/// Routes every gap of `w`'s paths (see the module doc); the idle PEs a wire crosses become
/// relays (`floor`'s relay rectangles, one a PE).
pub fn fill(floor: &mut Floor, w: &mut Wiring) -> R<()> {
    let mut wires: Vec<Wire> = Vec::new();
    for (li, l) in w.links.iter().enumerate() {
        if let Link::Path(p) = l {
            for s in p.pts.windows(2) {
                if s[0].dir_to(s[1]).is_none() {
                    wires.push(Wire {
                        link: li,
                        class: p.class,
                        from: s[0],
                        to: s[1],
                    });
                }
            }
        }
    }
    if wires.is_empty() {
        return Ok(());
    }
    let (fw, fh) = (floor.w, floor.h);
    let idx = |c: Coord| (c.y * fw + c.x) as usize;
    // what the laid links hold a PE: the classes they route there
    let laid = Wiring {
        links: w
            .links
            .iter()
            .filter(|l| {
                !matches!(l, Link::Path(p) if p.pts.windows(2).any(|s| s[0].dir_to(s[1]).is_none()))
            })
            .cloned()
            .collect(),
        notes: Vec::new(),
    };
    let foot = crate::fabric::Palette::footprint(floor, &laid)?;
    let fixed: Vec<Vec<Class>> = foot.iter().map(|(_, v)| v.clone()).collect();
    let cap: Vec<u32> = floor
        .roles
        .iter()
        .map(|(c, r)| slots(r).saturating_sub(fixed[idx(c)].len() as u32))
        .collect();
    let mut hist = vec![0u64; (fw * fh) as usize];
    let mut routes: Vec<Vec<Coord>> = vec![Vec::new(); wires.len()];
    let mut pres = 1u64;
    for round in 0..ROUNDS {
        // present use a PE: wires through it, and of each class
        let mut used: HashMap<usize, (u32, BTreeMap<Class, u32>)> = HashMap::new();
        for (i, wi) in wires.iter().enumerate() {
            let r = route_one(floor, wi, &fixed, &cap, &hist, &used, pres)?;
            for &c in &r {
                let e = used.entry(idx(c)).or_default();
                e.0 += 1;
                *e.1.entry(wi.class).or_default() += 1;
            }
            routes[i] = r;
        }
        // overuse: a class twice, or past the PE's room
        let mut over = 0u64;
        for (&p, (n, by)) in &used {
            let twice: u32 = by.values().map(|&k| k.saturating_sub(1)).sum();
            let past = n.saturating_sub(cap[p]);
            let o = u64::from(twice + past);
            if o > 0 {
                hist[p] += o * STEP;
                over += o;
            }
        }
        if over == 0 {
            break;
        }
        if round + 1 == ROUNDS {
            return Err(format!(
                "route: {over} PE uses past what they hold after {ROUNDS} rounds ({} wires)",
                wires.len()
            ));
        }
        pres = pres * 3 / 2 + 1;
    }
    // the paths with their gaps filled
    let mut by_link: BTreeMap<usize, Vec<(Coord, Coord, Vec<Coord>)>> = BTreeMap::new();
    for (wi, r) in wires.iter().zip(routes) {
        by_link
            .entry(wi.link)
            .or_default()
            .push((wi.from, wi.to, r));
    }
    for (li, rs) in by_link {
        let Link::Path(p) = &mut w.links[li] else {
            unreachable!("a wire is a path's")
        };
        let mut pts = vec![p.pts[0]];
        for s in p.pts.windows(2) {
            match rs.iter().find(|(a, b, _)| *a == s[0] && *b == s[1]) {
                Some((_, _, r)) => pts.extend(&r[1..]),
                None => pts.push(s[1]),
            }
        }
        for &c in &pts {
            if floor.roles.get(c).is_idle() {
                *floor.roles.get_mut(c) = PeRole::new(Kind::RELAY, Place::None);
                floor.relays.push(Rect::new(c.x, c.y, 1, 1));
            }
        }
        p.pts = pts;
    }
    Ok(())
}

/// The cheapest path of `w` (its ends included), within a window around its ends.
fn route_one(
    floor: &Floor,
    w: &Wire,
    fixed: &[Vec<Class>],
    cap: &[u32],
    hist: &[u64],
    used: &HashMap<usize, (u32, BTreeMap<Class, u32>)>,
    pres: u64,
) -> R<Vec<Coord>> {
    let (fw, fh) = (floor.w, floor.h);
    let mut margin = MARGIN;
    loop {
        let x0 = w.from.x.min(w.to.x).saturating_sub(margin);
        let y0 = w.from.y.min(w.to.y).saturating_sub(margin);
        let x1 = (w.from.x.max(w.to.x) + margin).min(fw - 1);
        let y1 = (w.from.y.max(w.to.y) + margin).min(fh - 1);
        if let Some(r) = astar(floor, w, fixed, cap, hist, used, pres, (x0, y0, x1, y1)) {
            return Ok(r);
        }
        if x0 == 0 && y0 == 0 && x1 == fw - 1 && y1 == fh - 1 {
            return Err(format!(
                "route: no way for {:?} from ({}, {}) to ({}, {})",
                w.class, w.from.x, w.from.y, w.to.x, w.to.y
            ));
        }
        margin *= 4;
    }
}

#[allow(clippy::too_many_arguments)]
fn astar(
    floor: &Floor,
    w: &Wire,
    fixed: &[Vec<Class>],
    cap: &[u32],
    hist: &[u64],
    used: &HashMap<usize, (u32, BTreeMap<Class, u32>)>,
    pres: u64,
    (x0, y0, x1, y1): (u32, u32, u32, u32),
) -> Option<Vec<Coord>> {
    let fw = floor.w;
    let (ww, wh) = (x1 - x0 + 1, y1 - y0 + 1);
    let local = |c: Coord| ((c.y - y0) * ww + (c.x - x0)) as usize;
    let n = (ww * wh) as usize;
    let mut g = vec![u64::MAX; n];
    // the step into a PE: 0 none (the start), 1..=4 from W, E, N, S
    let mut from = vec![0u8; n];
    let h = |c: Coord| STEP * u64::from(c.x.abs_diff(w.to.x) + c.y.abs_diff(w.to.y));
    let mut q = BinaryHeap::new();
    g[local(w.from)] = 0;
    q.push(Reverse((h(w.from), 0u64, w.from.y, w.from.x)));
    while let Some(Reverse((_, gc, y, x))) = q.pop() {
        let c = Coord::new(x, y);
        let li = local(c);
        if gc > g[li] {
            continue;
        }
        if c == w.to {
            let mut path = vec![c];
            let mut at = c;
            while at != w.from {
                at = match from[local(at)] {
                    1 => Coord::new(at.x - 1, at.y),
                    2 => Coord::new(at.x + 1, at.y),
                    3 => Coord::new(at.x, at.y - 1),
                    _ => Coord::new(at.x, at.y + 1),
                };
                path.push(at);
            }
            path.reverse();
            return Some(path);
        }
        let came = from[li];
        let steps = [
            (x < x1, Coord::new(x.wrapping_add(1), y), 1u8),
            (x > x0, Coord::new(x.wrapping_sub(1), y), 2),
            (y < y1, Coord::new(x, y.wrapping_add(1)), 3),
            (y > y0, Coord::new(x, y.wrapping_sub(1)), 4),
        ];
        for (ok, d, dir) in steps {
            if !ok {
                continue;
            }
            let p = (d.y * fw + d.x) as usize;
            let end = d == w.to;
            // a PE that routes the class already is no way through (the ends are the wire's)
            if !end && fixed[p].contains(&w.class) {
                continue;
            }
            let mut cost = STEP + hist[p];
            if came != 0 && came != dir {
                cost += TURN;
            }
            if floor.roles.get(d).is_idle() {
                cost += IDLE;
            }
            if let Some((k, by)) = used.get(&p) {
                let twice = by.get(&w.class).copied().unwrap_or(0);
                let past = (k + 1).saturating_sub(cap[p]);
                cost += pres * STEP * u64::from(twice + past);
            } else if cap[p] == 0 {
                cost += pres * STEP;
            }
            let nl = local(d);
            let ng = gc + cost;
            if ng < g[nl] {
                g[nl] = ng;
                from[nl] = dir;
                q.push(Reverse((ng + h(d), ng, d.y, d.x)));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use crate::program::Program;
    use crate::target::{Ask, plan_shape};
    use crate::wiring::Link;

    /// The tiny gpt-oss's units stacked in one column above its band (unit 0 on the band, the
    /// last on top, where the two-column layout cannot put them): the last unit's x is routed
    /// down past unit 0 into the band, every gap filled, and the program passes the check.
    #[test]
    fn a_band_fed_from_above_is_routed_and_checks() {
        let path = format!(
            "{}/tools/wafer/baseline/shapes/tiny_gptoss.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let s: crate::plan::Shape =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let plan = plan_shape(s, &Ask::preset("tiny-gptoss")).unwrap().plan;
        let p = &plan.partition;
        let uh = p.rows + 1;
        let origins: Vec<(u32, u32)> = (0..p.units).map(|u| (0, (p.units - 1 - u) * uh)).collect();
        let prog = Program::build_at(plan.clone(), origins).unwrap_or_else(|e| panic!("{e}"));
        let band = prog.floor.head.rect;
        assert!(prog.floor.units.last().unwrap().rect.y + uh < band.y);
        let mut down = 0;
        for l in &prog.wiring.links {
            if let Link::Path(p) = l {
                assert!(p.pts.windows(2).all(|s| s[0].dir_to(s[1]).is_some()));
                if p.class == crate::block::band::BAND_VERT {
                    assert!(p.pts.len() as u32 > uh, "{:?}", p.pts);
                    down += 1;
                }
            }
        }
        assert_eq!(down, p.rows);
        let levers = crate::emit::params::Levers::of(&plan.levers, "").unwrap();
        let dir = std::env::temp_dir().join(format!("route-band-{}", std::process::id()));
        crate::check::keep();
        crate::build::emit(&prog, &levers, None, &dir, &crate::build::EMITS).unwrap();
        let rep = crate::check::last().expect("a check");
        let _ = std::fs::remove_dir_all(&dir);
        eprintln!("{}", rep.line());
        rep.ok().unwrap_or_else(|e| panic!("{e}"));
    }
}
