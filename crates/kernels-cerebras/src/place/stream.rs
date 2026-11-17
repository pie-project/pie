//! A stream (or grid) program as a placement problem: each unit's roots and weight PEs, and
//! each K / V column, are blocks; the row stream between units, a K / V column's links to its
//! block's units, the lane gather and bus are the nets ([`super::Problem`]). A unit's helpers
//! below its column roots are free PEs its helper chains move with it.
//!
//! [`stream`] moves a [`Stream`]'s units to a [`Solution`]; [`relay`] re-lays a built program
//! on a moved stream (its palette and fabric; the units' code, the host's arrays and the
//! schedule do not read where a unit sits).

use super::{BlockKind, NetKind, Problem, Solution};
use crate::fabric::{Fabric, Palette};
use crate::floor::{Coord, Rect};
use crate::floorplan::stream::{KvColumn, Stream};
use crate::stream::build::Program;
use crate::wiring::{Link, Wiring};

type R<T> = Result<T, String>;

/// A unit's block: its CX column, weight PEs and CY row (from its band row's next row).
#[must_use]
pub fn unit_rect(s: &Stream, u: usize) -> Rect {
    let su = &s.units[u];
    Rect::new(su.x, su.y + 1, su.cols() + 1, su.rows() + 1)
}

/// A K / V column's block: its root and its PEs.
#[must_use]
pub fn kv_rect(s: &Stream, k: &KvColumn) -> Rect {
    let r = k.root(s);
    let (mut x1, mut y1) = (r.x, r.y);
    for i in 0..k.pya {
        let c = k.pe(s, i);
        x1 = x1.max(c.x);
        y1 = y1.max(c.y);
    }
    Rect::new(r.x, r.y, x1 - r.x + 1, y1 - r.y + 1)
}

fn kind(l: &Link) -> NetKind {
    let name = l.classes().first().map_or("", |c| c.name);
    if name.starts_with("SP") || name.starts_with("SW") {
        NetKind::Handoff
    } else if name.starts_with("KI") || name.starts_with("KO") {
        NetKind::KvLine
    } else if name.starts_with("LB") || name.starts_with("LG") {
        NetKind::SideBus
    } else {
        NetKind::Other
    }
}

/// `s`'s blocks and `w`'s links sorted onto them.
pub fn problem(s: &Stream, w: &Wiring) -> R<Problem> {
    let mut rects: Vec<(BlockKind, Rect)> = (0..s.units.len())
        .map(|u| (BlockKind::StreamUnit(u), unit_rect(s, u)))
        .collect();
    for (i, k) in s.kv.iter().enumerate() {
        rects.push((BlockKind::KvColumn(i), kv_rect(s, k)));
    }
    Problem::of(s.w, s.h, rects, w, &kind)
}

/// `s` with its units where `sol` puts them, on `sol`'s fabric. A K / V column sits where its
/// block's first unit puts it (an origin of its own wants the router: refused here), the lane
/// bus's receivers move with their units.
pub fn stream(p: &Problem, s: &Stream, sol: &Solution) -> R<Stream> {
    let mut out = s.clone();
    out.w = sol.w;
    out.h = sol.h;
    let mut moved = Vec::with_capacity(s.units.len());
    for (i, b) in p.blocks.iter().enumerate() {
        if let BlockKind::StreamUnit(u) = b.kind {
            let d = sol.delta(p, i);
            // (the block starts a row below the unit's band row)
            let at = super::shift(Coord::new(s.units[u].x, s.units[u].y), d)?;
            out.units[u].x = at.x;
            out.units[u].y = at.y;
            moved.push((u, d));
        }
    }
    let delta = |u: usize| moved.iter().find(|(v, _)| *v == u).map(|(_, d)| *d);
    for (i, b) in p.blocks.iter().enumerate() {
        if let BlockKind::KvColumn(k) = b.kind {
            let first = s.kv[k].units[0];
            if Some(sol.delta(p, i)) != delta(first) {
                return Err(format!(
                    "place: K / V column {k} moved apart from its first unit wants the router"
                ));
            }
        }
    }
    if let Some(lb) = &mut out.lanes {
        for c in &mut lb.to {
            let u = (0..s.units.len())
                .find(|&u| unit_rect(s, u).contains(*c))
                .ok_or_else(|| format!("place: a lane receiver at {c:?} in no unit"))?;
            *c = super::shift(*c, delta(u).unwrap_or((0, 0)))?;
        }
    }
    Ok(out)
}

/// `prog` re-laid on `s`: its links, palette and fabric.
pub fn relay(prog: &Program, s: Stream) -> R<Program> {
    let wiring = s.wiring();
    let floor = s.floor();
    let palette = Palette::allocate(Vec::new(), &floor, &wiring)?;
    let fabric = Fabric::lay(&floor, &wiring, &palette)?;
    fabric.validate(&floor)?;
    Ok(Program {
        trace: prog.trace.clone(),
        shape: prog.shape.clone(),
        units: prog.units.clone(),
        kv: prog.kv.clone(),
        stream: s,
        palette,
        fabric,
        class_rows: prog.class_rows.clone(),
        lane_words: prog.lane_words,
        lane_at: prog.lane_at.clone(),
        round16: prog.round16,
    })
}

#[cfg(test)]
mod tests {
    use super::super::Solution;
    use crate::stream::build::Program;

    fn trace(name: &str) -> poem_ir::Trace {
        poem_compiler::catalog::repository()
            .parse(name)
            .unwrap_or_else(|| panic!("{name} parses"))
            .trace(poem_ir::Platform::Cerebras)
    }

    /// `p` placed by the placer pinned to it, emitted: every file the same; its links sorted
    /// into blocks and nets with the row stream between units a net.
    fn same(p: &Program, name: &str) {
        let pr =
            super::problem(&p.stream, &p.stream.wiring()).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            pr.nets.iter().any(|n| n.kind == super::NetKind::Handoff) || p.stream.path.len() < 2,
            "{name}: no row stream net"
        );
        let s = super::stream(&pr, &p.stream, &Solution::pinned(&pr)).unwrap();
        assert_eq!(
            pr.apply(&p.stream.wiring(), &Solution::pinned(&pr))
                .unwrap(),
            s.wiring()
        );
        let q = super::relay(p, s).unwrap_or_else(|e| panic!("{name}: {e}"));
        let (a, b) = (
            crate::emit::stream::emit(p).unwrap(),
            crate::emit::stream::emit(&q).unwrap(),
        );
        assert_eq!(a.files.len(), b.files.len(), "{name}");
        for ((fa, ta), (fb, tb)) in a.files.iter().zip(&b.files) {
            assert_eq!(fa, fb, "{name}");
            assert!(ta == tb, "{name}: {fa} differs");
        }
        assert!(a.arrays == b.arrays, "{name}: host arrays differ");
        eprintln!(
            "{name}: {} blocks, {} nets, {} files identical",
            pr.blocks.len(),
            pr.nets.len(),
            a.files.len()
        );
    }

    /// The placer pinned to a stream program's own placement reproduces it: flux2-mini's
    /// denoise (one band, and three bands in a fabric 38 wide) and Qwen3-0.6B's decode units.
    #[test]
    fn the_pinned_placement_is_the_stream_program() {
        let t = trace("flux2-mini-bf16-kv-bf16");
        let s = crate::stream::shape_of(&t, "denoise", 64).unwrap();
        same(
            &crate::stream::build::program(&t, s, true).unwrap(),
            "flux2-mini denoise",
        );
        let s = crate::stream::shape_of(&t, "denoise", 16).unwrap();
        let p = crate::stream::build::program_py(
            &t,
            s,
            true,
            None,
            &[],
            None,
            None,
            None,
            false,
            crate::floorplan::stream::Bounds::system().narrowed(Some(38)),
        )
        .unwrap();
        same(&p, "flux2-mini denoise in 38 columns");
        // (as `stream::compile_decode` builds it: a row a step, 64 positions)
        let t = trace("qwen3-0.6b-bf16-kv-bf16");
        let classes = poem_compiler::spatial::layer::stream::decode_classes(&t, 1);
        let mut s = crate::stream::shape(&t, "decode", &classes).unwrap();
        s.cache = Some(64);
        let p = crate::stream::build::program_py(
            &t,
            s,
            true,
            None,
            &[],
            None,
            None,
            None,
            false,
            crate::floorplan::stream::Bounds::system(),
        )
        .unwrap();
        same(&p, "qwen3-0.6b units");
    }
}
