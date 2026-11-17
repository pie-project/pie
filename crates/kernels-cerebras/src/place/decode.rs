//! A decode program as a placement problem: its units, gap slots, head and storage columns are
//! the blocks; the handoffs, the head's input, the embedding's return, the shared cache's line
//! and the side input's buses are the nets ([`super::Problem`]). [`floor`] moves a floor's
//! blocks to a [`Solution`].

use super::{BlockKind, NetKind, Problem, Solution};
use crate::block::{band, head, head_groups, ple, shared_kv, token_loop, units};
use crate::floor::{Coord, Floor, Grid};
use crate::role::PeRole;
use crate::wiring::{Link, Wiring};

type R<T> = Result<T, String>;

/// A link's net kind, by its classes.
fn kind(l: &Link) -> NetKind {
    let cs = l.classes();
    let any = |of: &[crate::wiring::Class]| cs.iter().any(|c| of.contains(c));
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

/// `floor`'s blocks (before any block settles PEs onto the links: every PE that runs a kernel
/// is in one) and `w`'s links sorted onto them.
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

/// `floor` with its blocks where `s` puts them: each block's PEs and rectangle moved, the
/// fabric `s`'s size, every other PE idle.
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
            crate::floor::CtlLine::Column(x) => {
                crate::floor::CtlLine::Column(super::shift(Coord::new(x, 0), (d.0, 0))?.x)
            }
            crate::floor::CtlLine::Row(y) => {
                crate::floor::CtlLine::Row(super::shift(Coord::new(0, y), (0, d.1))?.y)
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
    use crate::build::{EMITS, emit};
    use crate::emit::params::Levers;
    use crate::plan::Plan;
    use crate::program::Program;
    use crate::target::{Ask, plan, plan_shape};

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

    /// The placer pinned to a decode program's own placement reproduces it byte for byte: every
    /// shipping config's measured plan and every kernel set's tiny (each file `build` emits).
    #[test]
    fn the_pinned_placement_is_the_program() {
        std::thread::scope(|sc| {
            for (model, preset) in [
                ("qwen35-d0.8b-bf16-kv-bf16", "0.8b"),
                ("qwen35-d4b-bf16-kv-bf16", "4b"),
                ("qwen35-d9b-bf16-kv-bf16", "9b"),
                ("qwen3-8b-bf16-kv-bf16", "q3-8b"),
                ("llama31-8b-bf16-kv-bf16", "l3-8b"),
                ("gemma4-e2b-bf16-kv-bf16", "e2b"),
                ("gemma4-e4b-bf16-kv-bf16", "e4b"),
                ("gemma4-12b-bf16-kv-bf16", "g12"),
                ("gptoss-20b-bf16-mxfp4-kv-bf16", "full"),
            ] {
                sc.spawn(move || {
                    let t = poem_compiler::catalog::repository()
                        .parse(model)
                        .unwrap()
                        .trace(poem_ir::Platform::Cerebras);
                    let p = plan(&t, &Ask::preset(preset)).unwrap().plan;
                    same(&p, preset);
                });
            }
            for (file, preset) in [
                ("tiny_q35_t08", "tiny-t08"),
                ("tiny_q35_t4", "tiny-t4"),
                ("tiny_q3", "tiny-q3"),
                ("tiny_q3_gap", "tiny-q3-gap"),
                ("tiny_llama", "tiny-q3"),
                ("tiny_gemma", "tiny-gemma"),
                ("tiny_g12", "tiny-g12"),
                ("tiny_gptoss", "tiny-gptoss"),
            ] {
                sc.spawn(move || {
                    let path = format!(
                        "{}/tools/wafer/baseline/shapes/{file}.json",
                        env!("CARGO_MANIFEST_DIR")
                    );
                    let s: crate::plan::Shape =
                        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
                    same(&plan_shape(s, &Ask::preset(preset)).unwrap().plan, file);
                });
            }
        });
    }

    /// Every block of a decode program is in the problem, and its links sort onto blocks and
    /// nets: the handoffs and the head's input are nets, a unit's chains its own.
    #[test]
    fn a_decode_program_is_a_problem() {
        let t = poem_compiler::catalog::repository()
            .parse("qwen35-d0.8b-bf16-kv-bf16")
            .unwrap()
            .trace(poem_ir::Platform::Cerebras);
        let p = plan(&t, &Ask::preset("0.8b")).unwrap().plan;
        let prog = Program::build(p).unwrap();
        // (before settle: the 0.8B settles nothing)
        let pr = super::problem(&prog.floor, &prog.wiring).unwrap();
        let units = pr
            .blocks
            .iter()
            .filter(|b| matches!(b.kind, super::BlockKind::Unit(_)))
            .count();
        assert_eq!(units, prog.floor.units.len());
        let kinds: std::collections::BTreeSet<_> = pr.nets.iter().map(|n| n.kind).collect();
        assert!(kinds.contains(&super::NetKind::Handoff), "{kinds:?}");
        assert!(kinds.contains(&super::NetKind::HeadInput), "{kinds:?}");
        assert!(kinds.contains(&super::NetKind::TokenReturn), "{kinds:?}");
        assert!(
            pr.blocks
                .iter()
                .all(|b| !b.links.is_empty() || b.kind == super::BlockKind::Relay)
        );
        eprintln!(
            "0.8B: {} blocks, {} nets ({:?})",
            pr.blocks.len(),
            pr.nets.len(),
            kinds
        );
    }

    /// Every block moved alike (the program shifted on a larger fabric) builds: the blocks'
    /// PEs, links, notes and settled PEs follow, and the fabric lays and validates.
    #[test]
    fn a_shifted_placement_builds() {
        for (file, preset) in [
            ("tiny_q35_t08", "tiny-t08"),
            ("tiny_q3_gap", "tiny-q3-gap"),
            ("tiny_gemma", "tiny-gemma"),
            ("tiny_g12", "tiny-g12"),
            ("tiny_gptoss", "tiny-gptoss"),
        ] {
            let path = format!(
                "{}/tools/wafer/baseline/shapes/{file}.json",
                env!("CARGO_MANIFEST_DIR")
            );
            let s: crate::plan::Shape =
                serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            let plan = plan_shape(s, &Ask::preset(preset)).unwrap().plan;
            let a = Program::build(plan.clone()).unwrap();
            let b = Program::build_placed(plan, &|p| {
                let mut s = Solution::pinned(p);
                s.w += 3;
                s.h += 2;
                for c in &mut s.at {
                    c.x += 3;
                    c.y += 2;
                }
                Ok(s)
            })
            .unwrap_or_else(|e| panic!("{file}: {e}"));
            assert_eq!((b.floor.w, b.floor.h), (a.floor.w + 3, a.floor.h + 2));
            for (c, r) in a.floor.roles.iter() {
                let d = crate::floor::Coord::new(c.x + 3, c.y + 2);
                assert_eq!(b.floor.roles.get(d), r, "{file}: {c:?}");
            }
            assert_eq!(a.wiring.links.len(), b.wiring.links.len());
            for (la, lb) in a.wiring.links.iter().zip(&b.wiring.links) {
                assert_eq!(&super::super::translate(la, (3, 2)).unwrap(), lb, "{file}");
            }
            assert_eq!(a.palette.map, b.palette.map, "{file}");
        }
    }
}
