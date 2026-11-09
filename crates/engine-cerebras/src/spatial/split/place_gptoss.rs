//! gpt-oss's role-split placement (`tools/split/gptoss/run_gptoss.py`'s, copy for copy and in its order):
//! every PE's `rcfg` and its tiles from a `prep_gptoss.py` prep.

use super::SplitSpec;
use super::geo::gptoss::GptossGeo;
use super::npy::Npy;
use crate::sdk::Rect;
use crate::spatial::device::Device;

type R<T> = std::result::Result<T, String>;

fn put(d: &mut dyn Device, sym: &str, [x, y, w, h]: [u32; 4], words: &[u32], b16: bool) -> R<()> {
    if w * h == 0 {
        return Ok(());
    }
    let n = words.len() / (w * h) as usize;
    let r = Rect { x: x as i32, y: y as i32, w: w as i32, h: h as i32 };
    if b16 { d.put16(sym, r, n, words) } else { d.put(sym, r, n, words) }
}

fn cfgs(rows: impl Iterator<Item = R<Vec<u32>>>) -> R<Vec<u32>> {
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

/// Rows r0..r1 of a 2-D [rows][n] array.
fn rows(a: &Npy, r0: u32, r1: u32) -> R<Vec<u32>> {
    let n = a.shape[1];
    let mut v = Vec::new();
    a.words(r0 as usize * n, (r1 - r0) as usize * n, &mut v)?;
    Ok(v)
}

pub fn place(d: &mut dyn Device, g: &GptossGeo, spec: &SplitSpec) -> R<()> {
    let (ax, mx, py, xc, yc) = (g.ax, g.mx, g.g.p.py, g.g.xc, g.g.yc);
    let rb_ = 16;
    let rot = spec.npy("rot")?;
    for u in 0..g.g.units() {
        let t0 = std::time::Instant::now();
        let (x0, y0u) = g.g.unit_origin(u);
        let (wa, wx, wr) = (spec.npy(&format!("wa{u}"))?, spec.npy(&format!("wx{u}"))?, spec.npy(&format!("wr{u}"))?);
        for (r0, r1) in [(0, yc), (yc, py)] {
            let mut rb = r0;
            while rb < r1 {
                let re = (rb + rb_).min(r1);
                let y0 = y0u + g.g.wl(rb, yc);
                let rc = cfgs((rb..re).flat_map(|r| (0..ax).map(move |c| g.rcfg_w(x0 + c, y0u + g.g.wl(r, yc), r, 0))))?;
                put(d, "rcfg", [x0, y0, ax, re - rb], &rc, false)?;
                put(d, "w", [x0, y0, ax, re - rb], &wa.block(None, (rb as usize, re as usize), (0, ax as usize))?, true)?;
                let rc = cfgs((rb..re).flat_map(|r| {
                    (0..mx).map(move |c| g.rcfg_w(x0 + xc + 1 + c, y0u + g.g.wl(r, yc), r, u32::from(c == 0)))
                }))?;
                put(d, "rcfg", [x0 + xc + 1, y0, mx, re - rb], &rc, false)?;
                put(d, "wx", [x0 + xc + 1, y0, mx, re - rb], &wx.block(None, (rb as usize, re as usize), (0, mx as usize))?, true)?;
                put(d, "wr", [x0 + xc + 1, y0, mx, re - rb], &wr.block(None, (rb as usize, re as usize), (0, mx as usize))?, false)?;
                rb = re;
            }
            let y0 = y0u + g.g.wl(r0, yc);
            let rc = cfgs((r0..r1).map(|r| g.rcfg_cx(x0 + xc, y0u + g.g.wl(r, yc), u == 0)))?;
            put(d, "rcfg", [x0 + xc, y0, 1, r1 - r0], &rc, false)?;
            for s in ["gam", "bo", "bd"] {
                put(d, s, [x0 + xc, y0, 1, r1 - r0], &rows(&spec.npy(&format!("{s}{u}"))?, r0, r1)?, false)?;
            }
        }
        let y = y0u + yc;
        put(d, "rcfg", [x0, y, ax, 1], &cfgs((0..ax).map(|c| g.rcfg_cy(x0 + c, y)))?, false)?;
        for s in ["qb", "snk"] {
            put(d, s, [x0, y, ax, 1], &rows(&spec.npy(&format!("{s}{u}"))?, 0, ax)?, false)?;
        }
        put(d, "rot", [x0, y, ax, 1], &rows(&rot, 0, ax)?, false)?;
        put(d, "rcfg", [x0 + xc + 1, y, mx, 1], &cfgs((0..mx).map(|c| g.rcfg_cy(x0 + xc + 1 + c, y)))?, false)?;
        for s in ["rb", "gb", "ps"] {
            put(d, s, [x0 + xc + 1, y, mx, 1], &rows(&spec.npy(&format!("{s}{u}"))?, 0, mx)?, false)?;
        }
        tracing::info!(unit = u, s = t0.elapsed().as_secs_f64(), "split gpt-oss: unit placed");
    }
    // the head band
    let (hw, hg, he) = (spec.npy("hw")?, spec.npy("hg")?, spec.npy("hemb")?);
    let (hxc, hvc) = (g.hxc, g.hvc);
    let cb_ = 64;
    for (i0, i1) in [(0, hxc), (hxc, py)] {
        let mut ib = i0;
        while ib < i1 {
            let ie = (ib + rb_).min(i1);
            let y0 = g.by(ib);
            let mut cb = 0;
            while cb < hvc {
                let ce = (cb + cb_).min(hvc);
                let rc = cfgs((ib..ie).flat_map(|i| (cb..ce).map(move |j| g.rcfg_head(j, g.by(i)))))?;
                let r = [cb, y0, ce - cb, ie - ib];
                put(d, "rcfg", r, &rc, false)?;
                put(d, "hw", r, &hw.block(None, (ib as usize, ie as usize), (cb as usize, ce as usize))?, true)?;
                put(d, "hemb", r, &he.block(None, (ib as usize, ie as usize), (cb as usize, ce as usize))?, true)?;
                let mut all = Vec::new();
                for i in ib..ie {
                    let one = rows(&hg, i, i + 1)?;
                    for _ in cb..ce {
                        all.extend_from_slice(&one);
                    }
                }
                put(d, "hg", r, &all, false)?;
                cb = ce;
            }
            ib = ie;
        }
    }
    let vr = spec.int("VR").ok_or("split meta: no VR")?;
    let yc_ = g.hb0 + hxc;
    put(d, "rcfg", [0, yc_, hvc, 1], &cfgs((0..hvc).map(|j| g.rcfg_hctl(j, yc_, vr)))?, false)?;
    Ok(())
}
