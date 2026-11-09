//! Gemma 4 12B's role-split placement (`tools/split/g12/run_g12.py`'s copies): the units (weights of the
//! attention / plain rows, rcfg, CX gammas, CY q / k norm weights and RoPE tables), the head (hw0 / hw1, hg) and its
//! control column.

use super::SplitSpec;
use super::geo::g12::G12Geo;
use super::place::{cfgs, put};
use crate::spatial::device::Device;

type R<T> = std::result::Result<T, String>;

/// (r0, r1, attention rows?) runs of rows of one kind on one side of the CY row.
pub fn rowblocks(py: u32, yc: u32, ra: u32) -> Vec<(u32, u32, bool)> {
    let mut out = Vec::new();
    for (a, b) in [(0, yc), (yc, py)] {
        for (k0, k1, att) in [(0, ra, true), (ra, py, false)] {
            let (lo, hi) = (a.max(k0), b.min(k1));
            if lo < hi {
                out.push((lo, hi, att));
            }
        }
    }
    out
}

pub fn place(d: &mut dyn Device, gg: &G12Geo, spec: &SplitSpec) -> R<()> {
    let g = &gg.g;
    let (px, py, xc, yc, ra) = (g.p.px, g.p.py, g.xc, g.yc, gg.ra);
    let last = g.units() - 1;
    let (gama, gamw) = (spec.npy("gama")?, spec.npy("gamw")?);
    let (qkw, rot) = (spec.npy("qkw")?, spec.npy("rot")?);
    for u in 0..g.units() {
        let t0 = std::time::Instant::now();
        let (x0u, y0u) = g.unit_origin(u);
        let (wa, ww) = (spec.npy(&format!("w{u}a"))?, spec.npy(&format!("w{u}w"))?);
        for (r0, r1, att) in rowblocks(py, yc, ra) {
            let (wu, rb0) = if att { (&wa, 0) } else { (&ww, ra) };
            for (c0, c1) in [(0, xc), (xc, px)] {
                let (x0, y0) = (x0u + g.wl(c0, xc), y0u + g.wl(r0, yc));
                let rc = cfgs((r0..r1).flat_map(|r| {
                    (c0..c1).map(move |c| {
                        gg.rcfg_weight(x0u + g.wl(c, xc), y0u + g.wl(r, yc), r, c, u == last)
                    })
                }))?;
                put(d, "rcfg", [x0, y0, c1 - c0, r1 - r0], &rc, false)?;
                let mut rb = r0;
                while rb < r1 {
                    let re = (rb + 16).min(r1);
                    let blk = wu.block(
                        None,
                        ((rb - rb0) as usize, (re - rb0) as usize),
                        (c0 as usize, c1 as usize),
                    )?;
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
            let y0 = y0u + g.wl(r0, yc);
            let rc =
                cfgs((r0..r1).map(|r| gg.rcfg_cx(x0u + xc, y0u + g.wl(r, yc), u == 0, u == last)))?;
            put(d, "rcfg", [x0u + xc, y0, 1, r1 - r0], &rc, false)?;
            let gam = if att { &gama } else { &gamw };
            let n = gam.shape[2];
            let mut v = Vec::new();
            gam.words(
                u * gam.stride0() + (r0 - rb0) as usize * n,
                (r1 - r0) as usize * n,
                &mut v,
            )?;
            put(d, "gam", [x0u + xc, y0, 1, r1 - r0], &v, false)?;
        }
        for (c0, c1) in [(0, xc), (xc, px)] {
            let x0 = x0u + g.wl(c0, xc);
            let rc = cfgs((c0..c1).map(|c| gg.rcfg_cy(x0u + g.wl(c, xc), y0u + yc, u == last)))?;
            put(d, "rcfg", [x0, y0u + yc, c1 - c0, 1], &rc, false)?;
            let mut v = Vec::new();
            let n = qkw.shape[2];
            qkw.words(
                u * qkw.stride0() + c0 as usize * n,
                (c1 - c0) as usize * n,
                &mut v,
            )?;
            put(d, "qkw", [x0, y0u + yc, c1 - c0, 1], &v, false)?;
            let mut v = Vec::new();
            let n = rot.shape[1];
            rot.words(c0 as usize * n, (c1 - c0) as usize * n, &mut v)?;
            put(d, "rot", [x0, y0u + yc, c1 - c0, 1], &v, false)?;
        }
        tracing::info!(
            unit = u,
            s = t0.elapsed().as_secs_f64(),
            "split g12: unit placed"
        );
    }
    // the head
    let (hx0, hxc, hc, hr) = (g.hx0, g.hxc, g.p.hc, g.p.hr);
    let hg = spec.npy("hg")?;
    // the head tiles in its two halves' files (one is over the 2 GiB a staged file may take)
    for (i, (c0, c1)) in [(0, hxc), (hxc, hc)].into_iter().enumerate() {
        if c1 <= c0 {
            continue;
        }
        let hw = spec.npy(&format!("hw{i}"))?;
        let x0 = g.hx(c0);
        let mut rb = 0;
        while rb < hr {
            let re = (rb + 64).min(hr);
            let rc = cfgs((rb..re).flat_map(|y| (c0..c1).map(move |c| gg.rcfg_head(g.hx(c), y))))?;
            put(d, "rcfg", [x0, rb, c1 - c0, re - rb], &rc, false)?;
            let blk = hw.block(None, (rb as usize, re as usize), (0, (c1 - c0) as usize))?;
            put(d, "hw", [x0, rb, c1 - c0, re - rb], &blk, true)?;
            let n = hg.shape[1];
            let mut one = Vec::new();
            hg.words(c0 as usize * n, (c1 - c0) as usize * n, &mut one)?;
            let all: Vec<u32> = (rb..re).flat_map(|_| one.iter().copied()).collect();
            put(d, "hg", [x0, rb, c1 - c0, re - rb], &all, false)?;
            rb = re;
        }
    }
    let vr = spec.int("VR").ok_or("split meta: no VR")?;
    let rc = cfgs((0..hr).map(|y| gg.rcfg_hctl(hx0 + hxc, y, vr)))?;
    put(d, "rcfg", [hx0 + hxc, 0, 1, hr], &rc, false)?;
    Ok(())
}
