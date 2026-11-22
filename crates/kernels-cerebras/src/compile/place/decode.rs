//! A decode program as a placement problem: its units, gap slots, head and storage columns are
//! the blocks, its handoffs and buses the nets. [`floor`] moves a floor's blocks to a
//! [`Solution`].

use super::{BlockKind, NetKind, Problem, Solution};
use crate::compile::block::{band, head, head_groups, ple, shared_kv, token_loop, units};
use crate::compile::floor::{Coord, Floor, Grid};
use crate::compile::role::PeRole;
use crate::compile::wiring::{Link, Wiring};

type R<T> = Result<T, String>;

/// A link's net kind, by its classes.
fn kind(l: &Link) -> NetKind {
    let cs = l.classes();
    let any = |of: &[crate::compile::wiring::Class]| cs.iter().any(|c| of.contains(c));
    if any(&[
        units::HANDOFF_OUT,
        units::HANDOFF_IN,
        units::HANDOFF_VERT[0],
        units::HANDOFF_VERT[1],
    ]) {
        NetKind::Handoff
    } else if any(&[
        units::HANDOFF_ACROSS,
        head::HEAD_RELAY,
        head_groups::HEAD_GROUP[0],
        head_groups::HEAD_GROUP[1],
        band::BAND_VERT,
    ]) {
        // (an across path joins two units or the last unit and an east head)
        NetKind::HeadInput
    } else if any(&[
        token_loop::EMB_UP,
        token_loop::EMB_DOWN,
        token_loop::EMB_SPLIT,
        token_loop::EMB_WEST,
        band::BAND_V1,
        band::BAND_W2,
    ]) {
        NetKind::TokenReturn
    } else if any(&[shared_kv::LINE]) {
        NetKind::KvLine
    } else if any(&[ple::TOKEN]) || cs.iter().any(|c| c.name.starts_with("PB")) {
        NetKind::SideBus
    } else {
        NetKind::Other
    }
}

/// `floor`'s blocks and `w`'s links sorted onto them as nets.
pub fn problem(floor: &Floor, w: &Wiring) -> R<Problem> {
    let mut rects = Vec::new();
    for u in &floor.units {
        rects.push((BlockKind::Unit(u.index), u.rect));
    }
    for r in &floor.relays {
        rects.push((BlockKind::Relay, *r));
    }
    rects.push((BlockKind::Head, floor.head.rect));
    for (i, s) in floor.stores.iter().enumerate() {
        rects.push((BlockKind::Store(i), s.rect));
    }
    rects.retain(|(_, r)| !r.is_empty());
    let p = Problem::of(floor.w, floor.h, rects, w, &kind)?;
    // every PE that runs a kernel is a block's
    for (c, role) in floor.roles.iter() {
        if !role.is_idle() && !p.blocks.iter().any(|b| b.rect.contains(c)) {
            return Err(format!(
                "place: ({}, {}) {} is in no block",
                c.x, c.y, role.kind.name
            ));
        }
    }
    Ok(p)
}

/// `floor` with its blocks moved where `s` puts them, the fabric resized and every other PE idle.
pub fn floor(p: &Problem, floor: &Floor, s: &Solution) -> R<Floor> {
    for i in 0..p.blocks.len() {
        let r = s.rect(p, i);
        if r.x + r.w > s.w || r.y + r.h > s.h {
            return Err(format!(
                "place: {:?} at ({}, {}) past the {} x {} fabric",
                p.blocks[i].kind, r.x, r.y, s.w, s.h
            ));
        }
        for j in 0..i {
            let q = s.rect(p, j);
            if r.x < q.x + q.w && q.x < r.x + r.w && r.y < q.y + q.h && q.y < r.y + r.h {
                return Err(format!(
                    "place: {:?} overlaps {:?}",
                    p.blocks[i].kind, p.blocks[j].kind
                ));
            }
        }
    }
    let mut roles: Grid<PeRole> = Grid::filled(s.w, s.h, PeRole::IDLE);
    for (i, b) in p.blocks.iter().enumerate() {
        let d = s.delta(p, i);
        for y in b.rect.y..b.rect.y + b.rect.h {
            for x in b.rect.x..b.rect.x + b.rect.w {
                let c = Coord::new(x, y);
                *roles.get_mut(super::shift(c, d)?) = *floor.roles.get(c);
            }
        }
    }
    let at = |k: &BlockKind| -> R<usize> {
        p.blocks
            .iter()
            .position(|b| b.kind == *k)
            .ok_or_else(|| format!("place: no block {k:?}"))
    };
    let mut out = floor.clone();
    out.w = s.w;
    out.h = s.h;
    out.roles = roles;
    for u in &mut out.units {
        u.rect = s.rect(p, at(&BlockKind::Unit(u.index))?);
    }
    let mut relays = p
        .blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| b.kind == BlockKind::Relay);
    for r in &mut out.relays {
        if r.is_empty() {
            continue;
        }
        let (i, _) = relays.next().ok_or("place: a relay slot lost")?;
        *r = s.rect(p, i);
    }
    if !floor.head.rect.is_empty() {
        let i = at(&BlockKind::Head)?;
        let d = s.delta(p, i);
        out.head.rect = s.rect(p, i);
        out.head.root = super::shift(floor.head.root, d)?;
        out.head.ctl = match floor.head.ctl {
            crate::compile::floor::CtlLine::Column(x) => {
                crate::compile::floor::CtlLine::Column(super::shift(Coord::new(x, 0), (d.0, 0))?.x)
            }
            crate::compile::floor::CtlLine::Row(y) => {
                crate::compile::floor::CtlLine::Row(super::shift(Coord::new(0, y), (0, d.1))?.y)
            }
        };
    }
    for (k, st) in out.stores.iter_mut().enumerate() {
        if !st.rect.is_empty() {
            st.rect = s.rect(p, at(&BlockKind::Store(k))?);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::Solution;
    use crate::compile::build::{EMITS, emit};
    use crate::compile::emit::params::Levers;
    use crate::compile::plan::Plan;
    use crate::compile::program::Program;

    /// Every file `plan` emits, built as it is and placed by the placer pinned to it.
    fn same(plan: &Plan, name: &str) {
        let levers = Levers::of(&plan.levers, "").unwrap();
        let dir = std::env::temp_dir().join(format!("place-ident-{}-{name}", std::process::id()));
        let emitted = |p: &Program, sub: &str| {
            let out = dir.join(sub);
            emit(p, &levers, None, &out, &EMITS).unwrap_or_else(|e| panic!("{name}: {e}"));
            let mut files = std::collections::BTreeMap::new();
            let mut stack = vec![out.clone()];
            while let Some(d) = stack.pop() {
                for e in std::fs::read_dir(&d).unwrap() {
                    let path = e.unwrap().path();
                    if path.is_dir() {
                        stack.push(path);
                    } else {
                        let rel = path.strip_prefix(&out).unwrap().to_path_buf();
                        files.insert(rel, std::fs::read(&path).unwrap());
                    }
                }
            }
            files
        };
        let a = Program::build(plan.clone()).unwrap_or_else(|e| panic!("{name}: {e}"));
        let b = Program::build_placed(plan.clone(), &|p| Ok(Solution::pinned(p)))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let (fa, fb) = (emitted(&a, "a"), emitted(&b, "b"));
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(
            fa.keys().collect::<Vec<_>>(),
            fb.keys().collect::<Vec<_>>(),
            "{name}"
        );
        for (k, v) in &fa {
            assert!(fb[k] == *v, "{name}: {} differs", k.display());
        }
        eprintln!("{name}: {} files identical", fa.len());
    }

    /// The placer pinned to a decode program's own placement reproduces it byte for byte, for
    /// every shipping config's measured plan.
    #[test]
    fn the_pinned_placement_is_the_program() {
        std::thread::scope(|sc| {
            for model in crate::compile::target::SHIPPING {
                sc.spawn(move || {
                    let p = crate::compile::target::searched(model).plan;
                    same(&p, model);
                });
            }
        });
    }
}
