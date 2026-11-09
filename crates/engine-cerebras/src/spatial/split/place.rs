//! The role-split program's placement: every PE's `rcfg` and its tiles,
//! copy for copy as the Python host reference it was checked against sent
//! them (the same rectangles in the same order: 569 copies over the 9B's
//! 643 x 1153 PEs, sha256 each; `examples/split_place.rs`).

use super::SplitSpec;
use super::geo::Geo;
use crate::sdk::Rect;
use crate::spatial::device::Device;

type R<T> = std::result::Result<T, String>;

fn rect(x: u32, y: u32, w: u32, h: u32) -> Rect {
    Rect {
        x: x as i32,
        y: y as i32,
        w: w as i32,
        h: h as i32,
    }
}

/// A copy of `words` (n a PE) into `sym` over the rectangle `[x, y, w, h]`
/// (none if empty).
pub(super) fn put(
    d: &mut dyn Device,
    sym: &str,
    [x, y, w, h]: [u32; 4],
    words: &[u32],
    b16: bool,
) -> R<()> {
    if w * h == 0 {
        return Ok(());
    }
    let n = words.len() / (w * h) as usize;
    if b16 {
        d.put16(sym, rect(x, y, w, h), n, words)
    } else {
        d.put(sym, rect(x, y, w, h), n, words)
    }
}

/// rcfg rows for a rectangle (row-major, one PE's words after another).
pub(super) fn cfgs(rows: impl Iterator<Item = R<Vec<u32>>>) -> R<Vec<u32>> {
    let mut out = Vec::new();
    let mut width = None;
    for w in rows {
        let w = w?;
        if *width.get_or_insert(w.len()) != w.len() {
            return Err("rcfg words of unequal length in one copy".into());
        }
        out.extend(w);
    }
    Ok(out)
}

pub fn place(d: &mut dyn Device, g: &Geo, spec: &SplitSpec) -> R<()> {
    let (px, py, xc, yc) = (g.p.px, g.p.py, g.xc, g.yc);
    let quads = [
        ((0, yc), (0, xc)),
        ((0, yc), (xc, px)),
        ((yc, py), (0, xc)),
        ((yc, py), (xc, px)),
    ];
    let gam = spec.npy("gam")?;
    let cy_syms: Vec<(&str, super::npy::Npy)> = ["cw", "gpar", "gnw", "qkw", "rot"]
        .into_iter()
        .filter(|s| spec.has(s))
        .map(|s| spec.npy(s).map(|n| (s, n)))
        .collect::<R<_>>()?;
    for u in 0..g.units() {
        let t0 = std::time::Instant::now();
        let (x0u, y0u) = g.unit_origin(u);
        let wu = spec.npy(&format!("w{u}"))?;
        for &((r0, r1), (c0, c1)) in &quads {
            let (x0, y0) = (x0u + g.wl(c0, xc), y0u + g.wl(r0, yc));
            let rc = cfgs((r0..r1).flat_map(|r| {
                (c0..c1).map(move |c| g.rcfg_weight(x0u + g.wl(c, xc), y0u + g.wl(r, yc), r))
            }))?;
            put(d, "rcfg", [x0, y0, c1 - c0, r1 - r0], &rc, false)?;
            let mut rb = r0;
            while rb < r1 {
                let re = (rb + 16).min(r1);
                let blk = wu.block(None, (rb as usize, re as usize), (c0 as usize, c1 as usize))?;
                put(
                    d,
                    "w",
                    [x0, y0u + g.wl(rb, yc), c1 - c0, re - rb],
                    &blk,
                    true,
                )?;
                rb = re;
            }
        }
        for (r0, r1) in [(0, yc), (yc, py)] {
            let y0 = y0u + g.wl(r0, yc);
            let rc = cfgs((r0..r1).map(|r| g.rcfg_cx(x0u + xc, y0u + g.wl(r, yc), u == 0)))?;
            put(d, "rcfg", [x0u + xc, y0, 1, r1 - r0], &rc, false)?;
            let n = gam.shape[2];
            let mut v = Vec::new();
            gam.words(
                u * gam.stride0() + r0 as usize * n,
                (r1 - r0) as usize * n,
                &mut v,
            )?;
            put(d, "gam", [x0u + xc, y0, 1, r1 - r0], &v, true)?;
        }
        for (c0, c1) in [(0, xc), (xc, px)] {
            let x0 = x0u + g.wl(c0, xc);
            let rc = cfgs((c0..c1).map(|c| g.rcfg_cy(x0u + g.wl(c, xc), y0u + yc)))?;
            put(d, "rcfg", [x0, y0u + yc, c1 - c0, 1], &rc, false)?;
            for (sym, a) in &cy_syms {
                // [NU][PX][n] or [PX][n]
                let (base, n) = if a.shape.len() == 3 {
                    (u * a.stride0(), a.shape[2])
                } else {
                    (0, a.shape[1])
                };
                let mut v = Vec::new();
                a.words(base + c0 as usize * n, (c1 - c0) as usize * n, &mut v)?;
                put(d, sym, [x0, y0u + yc, c1 - c0, 1], &v, false)?;
            }
        }
        tracing::info!(
            unit = u,
            s = t0.elapsed().as_secs_f64(),
            "split: unit placed"
        );
    }
    if let Some((gx, gy)) = g.p.gap {
        let (x0, y0) = (gx * g.uw, gy * g.uh);
        let rc =
            cfgs((y0..y0 + g.uh).flat_map(|y| (x0..x0 + g.uw).map(move |x| g.rcfg_relay(x, y))))?;
        put(d, "rcfg", [x0, y0, g.uw, g.uh], &rc, false)?;
    }
    // the head
    let (hx0, hxc, hc, hr) = (g.hx0, g.hxc, g.p.hc, g.p.hr);
    let hw = spec.npy("hw")?;
    let hg = spec.npy("hg")?;
    let halves: Vec<(u32, u32)> = [(0, hxc), (hxc, hc)]
        .into_iter()
        .filter(|(a, b)| b > a)
        .collect();
    for &(c0, c1) in &halves {
        let x0 = g.hx(c0);
        let mut rb = 0;
        while rb < hr {
            let re = (rb + 64).min(hr);
            let rc = cfgs((rb..re).flat_map(|y| (c0..c1).map(move |c| g.rcfg_head(g.hx(c), y))))?;
            put(d, "rcfg", [x0, rb, c1 - c0, re - rb], &rc, false)?;
            let blk = hw.block(None, (rb as usize, re as usize), (c0 as usize, c1 as usize))?;
            put(d, "hw", [x0, rb, c1 - c0, re - rb], &blk, true)?;
            let n = hg.shape[1];
            let mut one = Vec::new();
            hg.words(c0 as usize * n, (c1 - c0) as usize * n, &mut one)?;
            let all: Vec<u32> = (rb..re).flat_map(|_| one.iter().copied()).collect();
            put(d, "hg", [x0, rb, c1 - c0, re - rb], &all, false)?;
            rb = re;
        }
    }
    if g.p.wl {
        let he = spec.npy("hemb")?;
        for &(c0, c1) in &halves {
            let mut rb = 0;
            while rb < hr {
                let re = (rb + 64).min(hr);
                let blk = he.block(None, (rb as usize, re as usize), (c0 as usize, c1 as usize))?;
                put(d, "hemb", [g.hx(c0), rb, c1 - c0, re - rb], &blk, true)?;
                rb = re;
            }
        }
    }
    let vr = spec.int("VR").ok_or("split meta: no VR")?;
    let rc = cfgs((0..hr).map(|y| g.rcfg_hctl(hx0 + hxc, y, vr)))?;
    put(d, "rcfg", [hx0 + hxc, 0, 1, hr], &rc, false)?;
    Ok(())
}
