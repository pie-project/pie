//! Gemma 4's role-split placement (`scratch/split/host/run_gemma.py`'s copies): the units (weights,
//! rcfg, CX gammas, CY q / k norm weights and RoPE tables), the PLE storage tiles and the route-only
//! PEs, the head (hw, hg, hemb) and its control column.

use super::SplitSpec;
use super::geo::gemma::GemmaGeo;
use super::place::{cfgs, put};
use crate::spatial::device::Device;

type R<T> = std::result::Result<T, String>;

pub fn place(d: &mut dyn Device, gg: &GemmaGeo, spec: &SplitSpec) -> R<()> {
    let g = &gg.g;
    let (px, py, xc, yc) = (g.p.px, g.p.py, g.xc, g.yc);
    let last = g.units() - 1;
    let quads = [
        ((0, yc), (0, xc)),
        ((0, yc), (xc, px)),
        ((yc, py), (0, xc)),
        ((yc, py), (xc, px)),
    ];
    let (gam, qkw, rot) = (spec.npy("gam")?, spec.npy("qkw")?, spec.npy("rot")?);
    for u in 0..g.units() {
        let t0 = std::time::Instant::now();
        let (x0u, y0u) = g.unit_origin(u);
        let wu = spec.npy(&format!("w{u}"))?;
        for &((r0, r1), (c0, c1)) in &quads {
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
                let blk = wu.block(None, (rb as usize, re as usize), (c0 as usize, c1 as usize))?;
                put(d, "w", [x0, y0u + g.wl(rb, yc), c1 - c0, re - rb], &blk, true)?;
                rb = re;
            }
        }
        for (r0, r1) in [(0, yc), (yc, py)] {
            let y0 = y0u + g.wl(r0, yc);
            let rc =
                cfgs((r0..r1).map(|r| gg.rcfg_cx(x0u + xc, y0u + g.wl(r, yc), u == 0, u == last)))?;
            put(d, "rcfg", [x0u + xc, y0, 1, r1 - r0], &rc, false)?;
            let n = gam.shape[2];
            let mut v = Vec::new();
            gam.words(
                u * gam.stride0() + r0 as usize * n,
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
            "split gemma: unit placed"
        );
    }
    // PLE storage (tiles padded to NTW words) and the route-only PEs
    let cols = spec
        .meta
        .get("cols")
        .and_then(|v| v.as_array())
        .ok_or("split gemma meta: no cols")?;
    let ntw = cols
        .iter()
        .map(|c| c["nt"].as_u64().unwrap_or(0) * c["lv"].as_u64().unwrap_or(0))
        .max()
        .unwrap_or(0) as usize;
    let t0 = std::time::Instant::now();
    for col in cols {
        let x0c = col["x"].as_u64().ok_or("cols: x")? as u32;
        let y0 = col["srows"][0].as_u64().ok_or("cols: srows")? as u32;
        for f in col["files"].as_array().ok_or("cols: files")? {
            let arr = spec.npy(f["name"].as_str().ok_or("cols: file name")?)?;
            let (fr0, fr1) = (
                f["r0"].as_u64().unwrap() as u32,
                f["r1"].as_u64().unwrap() as u32,
            );
            let inner = arr.shape[2];
            for (c0, c1) in [(0, xc), (xc, px)] {
                let xs = x0c + g.wl(c0, xc);
                let mut rb = fr0;
                while rb < fr1 {
                    let re = (rb + 16).min(fr1);
                    let rc = cfgs((rb..re).flat_map(|r| {
                        (c0..c1).map(move |c| gg.rcfg_ple(x0c + g.wl(c, xc), y0 + r))
                    }))?;
                    put(d, "rcfg", [xs, y0 + rb, c1 - c0, re - rb], &rc, false)?;
                    let blk = arr.block(
                        None,
                        ((rb - fr0) as usize, (re - fr0) as usize),
                        (c0 as usize, c1 as usize),
                    )?;
                    let mut pad = Vec::with_capacity(blk.len() / inner * ntw);
                    for chunk in blk.chunks_exact(inner) {
                        pad.extend_from_slice(chunk);
                        pad.resize(pad.len() + ntw - inner, 0);
                    }
                    put(d, "ptab", [xs, y0 + rb, c1 - c0, re - rb], &pad, true)?;
                    rb = re;
                }
            }
        }
    }
    let mut passp = gg.pass_pes();
    for col in &gg.cols {
        for y in col.srows.0..col.srows.1 {
            passp.push((col.x + xc, y));
        }
    }
    let mut rows: std::collections::BTreeMap<u32, Vec<u32>> = std::collections::BTreeMap::new();
    for (x, y) in passp {
        rows.entry(y).or_default().push(x);
    }
    for (y, mut xs) in rows {
        xs.sort_unstable();
        let mut a0 = xs[0];
        for i in 1..=xs.len() {
            if i == xs.len() || xs[i] != xs[i - 1] + 1 {
                let b = xs[i - 1] + 1;
                let rc = cfgs((a0..b).map(|x| gg.rcfg_ple(x, y)))?;
                put(d, "rcfg", [a0, y, b - a0, 1], &rc, false)?;
                if i < xs.len() {
                    a0 = xs[i];
                }
            }
        }
    }
    tracing::info!(
        s = t0.elapsed().as_secs_f64(),
        "split gemma: PLE storage placed"
    );
    // the head
    let (hx0, hxc, hc, hr) = (g.hx0, g.hxc, g.p.hc, g.p.hr);
    let (hw, hg, he) = (spec.npy("hw")?, spec.npy("hg")?, spec.npy("hemb")?);
    for (c0, c1) in [(0, hxc), (hxc, hc)] {
        if c1 <= c0 {
            continue;
        }
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
            let blk = he.block(None, (rb as usize, re as usize), (c0 as usize, c1 as usize))?;
            put(d, "hemb", [x0, rb, c1 - c0, re - rb], &blk, true)?;
            rb = re;
        }
    }
    let vr = spec.int("VR").ok_or("split meta: no VR")?;
    let rc = cfgs((0..hr).map(|y| gg.rcfg_hctl(hx0 + hxc, y, vr)))?;
    put(d, "rcfg", [hx0 + hxc, 0, 1, hr], &rc, false)?;
    Ok(())
}
