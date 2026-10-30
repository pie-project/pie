//! The serve2d wafer program (`payload/serve2d` in the bring-up tree): the
//! decoder layers as qwen2d grids, head2d east of them (`HEADMODE` 2), behind
//! serve28s5's request API. The Rust form of `serve_lib2d.py`.
//!
//! Grids of `PX` x `PY` PEs, `GW` across and `GH` down, run serpentine
//! (strip `sy` westward from the east end when even, eastward when odd);
//! layer 0 is the top-right grid. A grid's row `py` holds hidden chunk `py`
//! (`Hc` dims); column `px` holds column slice `px` of the projections (a
//! kv group's `Dc`-dim slice of `Gs`, the MLP's `Ic` rows). The head is one
//! head2d block of `C = PY` chunk columns (plus `VMC` guest VM columns after
//! its root column) and `HH` region rows of `Vr` vocabulary rows, its
//! feeder row `F = HH / 2` between them. Every PE takes its place at `init`.
//!
//! A request is one launch, `compute(steps, start_pos)`: the forced tokens
//! go to `in_tokens` at the final root, `out_tokens` (u32) comes back from
//! it. The logits are the root column's `y` (the vocabulary rows, the
//! feeder row skipped) times `yscale`, the final RMSNorm's 1/rms of the step.

use super::ckpt::{Ckpt, Tensor};
use super::paged::SYMBOLS as SYM;
use super::wafer::{SAMPLE, Sink};
use crate::sdk::{DataType, Order, Rect, Runtime};

type R<T> = std::result::Result<T, String>;

/// The program's shape, from the compiler's `out.json` params.
#[derive(Debug, Clone)]
pub struct Layout2d {
    pub px: usize,
    pub py: usize,
    pub hc: usize,
    pub hidden: usize,
    pub dc: usize,
    pub gs: usize,
    pub qh: usize,
    pub ic: usize,
    pub tc: usize,
    pub gw: usize,
    pub gh: usize,
    pub hh: usize,
    pub vr: usize,
    pub gx: usize,
    pub vmc: usize,
    pub vocab: usize,
    pub cap: usize,
    /// `GRID` 1: the batched, paged build (Fork 8's qwen2d_paged grids):
    /// `B` lanes a launch over a pool of `NB` blocks a row of `PS`-slot
    /// pages, at most `E` of a lane's pages a row. `None`: one sequence.
    pub paged: Option<super::paged::Shape>,
    /// `serve2d_pipe`: up to `MMAX` micro-batches of B lanes go round the
    /// ring of grids and the head a launch (`mtab` at the head's feeders,
    /// no `tab`). `None`: one batch a launch.
    pub mmax: Option<usize>,
    /// `KVCP` 0: the build has no page-copy kernels; the host copies pages
    /// by reading, patching and writing back each row's `kc`/`vc`.
    pub host_copy: bool,
    /// The eta VM on the head's VM column (a GRID 0 build with guest4h3's
    /// VM, its limits among the params): sampled iterations take its token.
    pub vm: Option<super::guest4::wafer::VmLayout>,
    /// serve2d_pipe's `HLM`: the most virtual lanes whose logits rows the
    /// head keeps.
    pub hlm: Option<usize>,
    /// `HLBN`: the most lanes whose logits rows the head keeps, where the
    /// build caps it below HLM (serve8b_region at B 4 keeps one).
    pub hlbn: Option<usize>,
    /// Words of a virtual lane's VM control (`vmlane`): 2 (gmode, forced
    /// prefix) on guest4hs, 3 on guest4hz (and the offset to the lane's
    /// turn's last row), 4 on guest4h2s (and the lane's VM set: 0, set A,
    /// the only one the engine drives yet).
    pub vmlane_words: usize,
    /// Set A's VM lanes (`vm_lanes`; 0 or absent: every head lane): head
    /// lanes past them are set B's turns (vmlane word 3).
    pub vm_set_a_lanes: Option<usize>,
    /// `KVST`: KV staging slots a grid PE (`kstage`/`vstage`, `page_stage`
    /// / `page_unstage`; scratch/fork4/kv-staging-interface.md); 0: the
    /// host reads and patches whole kc/vc rows.
    pub kvst: usize,
    /// serve8b ([`super::serve8b`]): bf16 grids in strips that skip a slot
    /// (or serve8b_region's grids and regions), a head of `PY / 2` chunk
    /// columns, the embedding strip east of it.
    pub s8b: Option<super::serve8b::S8b>,
}

fn rect(x: usize, y: usize, w: usize, h: usize) -> Rect {
    Rect {
        x: x as i32,
        y: y as i32,
        w: w as i32,
        h: h as i32,
    }
}

impl Layout2d {
    /// The serve2d shape when `get` names one (`HEADMODE` 2, f32 weights),
    /// or serve8b's (`GS` among the params).
    pub fn of(get: impl Fn(&str) -> Option<usize>) -> Option<R<Layout2d>> {
        if get("GS").is_some() || get("RPX").is_some() {
            return Some(super::serve8b::layout(&get));
        }
        let mode = get("HEADMODE")?;
        let p = |k: &str| -> R<usize> { get(k).ok_or_else(|| format!("no param {k}")) };
        let lay = (|| -> R<Layout2d> {
            if mode != 2 {
                return Err(format!(
                    "HEADMODE {mode}: the engine serves serve2d with head2d (HEADMODE 2)"
                ));
            }
            if p("W16")? != 0 || p("KV16")? != 0 || p("WF32")? != 1 {
                return Err("the engine places f32 weights and KV (W16 0, KV16 0, WF32 1)".into());
            }
            let lay = Layout2d {
                px: p("PX")?,
                py: p("PY")?,
                hc: p("Hc")?,
                hidden: p("HR")?,
                dc: p("Dc")?,
                gs: p("Gs")?,
                qh: p("Qh")?,
                ic: p("Ic")?,
                tc: p("Tc")?,
                gw: p("GW")?,
                gh: p("GH")?,
                hh: p("HH")?,
                vr: p("Vr")?,
                gx: p("Gx")?,
                // Every VM column the head steps over: set A's VMC and a
                // second VM set's VMB (guest4h2s).
                vmc: p("VMC")? + get("VMB").unwrap_or(0),
                vocab: p("vocab")?,
                cap: p("CAP")?,
                paged: None,
                mmax: None,
                host_copy: get("KVCP").unwrap_or(1) == 0,
                vm: None,
                hlm: get("HLM"),
                hlbn: get("HLBN"),
                vmlane_words: 2,
                vm_set_a_lanes: get("vm_lanes").filter(|&n| n > 0),
                kvst: get("KVST").unwrap_or(0),
                s8b: None,
            };
            let mut lay = lay;
            if get("GRID").unwrap_or(0) == 1 {
                if lay.px != lay.py || lay.qh != 2 || lay.hidden != lay.py * lay.hc {
                    return Err(format!(
                        "a batched serve2d build wants square grids, Qh 2 and no hidden \
                         padding: {lay:?}"
                    ));
                }
                lay.paged = Some(super::paged::Shape {
                    lanes: p("B")?,
                    rows: lay.py,
                    page_size: p("PS")?,
                    blocks: p("NB")?,
                    entries: p("E")?,
                });
                lay.mmax = get("MMAX").filter(|&m| m > 0);
            } else if get("MMAX").is_some() || lay.host_copy {
                return Err("MMAX and KVCP 0 are batched (GRID 1) builds' params".into());
            }
            // The eta VM: a one-sequence build, or serve2d_pipe's (MMAX 1 is
            // plain batching), whose lanes the VM time-multiplexes.
            if lay.paged.is_none() || lay.mmax.is_some() {
                lay.vm = super::guest4::wafer::VmLayout::serve2d(lay.head_x(), &get);
            }
            if lay.py * lay.hc < lay.hidden || lay.gs == 0 || lay.gx == 0 || (lay.vmc > 1 && lay.vm.is_none()) {
                return Err(format!("serve2d params do not make a layout: {lay:?}"));
            }
            Ok(lay)
        })();
        Some(lay)
    }

    pub fn layers(&self) -> usize {
        use super::serve8b::S8b;
        match self.s8b {
            Some(S8b::Strips { full }) => full * self.gw + (self.gh - full) * (self.gw - 1),
            Some(S8b::Units(u)) => u.layers(),
            None => self.gw * self.gh,
        }
    }

    /// The head's chunk columns (C; the VM columns not counted).
    pub fn chunk_cols(&self) -> usize {
        if self.s8b.is_some() { self.py / 2 } else { self.py }
    }

    /// The grids' area as rectangles that skip serve8b's free slots
    /// (serve8b_region: each unit's).
    pub fn grid_areas(&self) -> Vec<Rect> {
        use super::serve8b::S8b;
        match self.s8b {
            Some(S8b::Units(u)) => u.units(self).iter().map(|u| rect(u.x0, u.y0, u.k.px, u.k.py)).collect(),
            Some(S8b::Strips { full }) => {
                let mut a = vec![rect(0, 0, self.gw * self.px, full * self.py)];
                if self.gh > full {
                    a.push(rect(self.px, full * self.py, (self.gw - 1) * self.px, (self.gh - full) * self.py));
                }
                a
            }
            None => vec![rect(0, 0, self.gw * self.px, self.gh * self.py)],
        }
    }

    /// Head dim (a kv group's `Gs` slices of `Dc`).
    pub fn head_dim(&self) -> usize {
        self.gs * self.dc
    }

    pub fn positions(&self) -> usize {
        match &self.paged {
            // A lane's slots: E pages on every row.
            Some(shape) => shape.max_page_refs() * shape.page_size,
            None => self.py * self.tc,
        }
    }

    /// Lanes one launch carries: B, or MMAX micro-batches of B.
    pub fn launch_lanes(&self) -> usize {
        self.paged.map_or(1, |s| s.lanes * self.mmax.unwrap_or(1))
    }

    /// Lanes whose last logits rows the head keeps (`HLB`: the first
    /// micro-batch's first four).
    pub fn logit_lanes(&self) -> usize {
        self.paged.map_or(1, |s| {
            let kept = match self.mmax {
                // serve2d_pipe: HLB = min(MMAX·B, HLM) virtual lanes (HLM 4
                // before the build had it, then only micro-batch 0's B).
                Some(m) => self.hlm.map_or(s.lanes.min(4), |h| (m * s.lanes).min(h)),
                None => s.lanes.min(4),
            };
            // A launch takes host-sampled lanes up to this; more go in the
            // next launch.
            self.hlbn.map_or(kept, |n| kept.min(n.max(1)))
        })
    }

    /// The feeder row.
    pub fn feeder(&self) -> usize {
        self.hh / 2
    }

    /// The head's first column.
    pub fn head_x(&self) -> usize {
        match self.s8b {
            // serve8b_region: the regions' columns, then the grid column.
            Some(super::serve8b::S8b::Units(u)) => u.rc * u.rpx + self.px,
            _ => self.gw * self.px,
        }
    }

    /// The root column inside the head block.
    pub(crate) fn r2x(&self) -> usize {
        let ngx = self.chunk_cols() / self.gx;
        (ngx / 2) * self.gx + self.gx / 2
    }

    /// The final root (fabric): `in_tokens`, `out_tokens`, the root column.
    pub fn root(&self) -> (usize, usize) {
        (self.head_x() + self.r2x(), self.feeder())
    }

    /// Layer `l`'s grid's top-left PE (serve8b_region: its unit's).
    pub fn origin(&self, l: usize) -> (usize, usize) {
        use super::serve8b::S8b;
        let (sy, k, sxmin) = match self.s8b {
            Some(S8b::Units(u)) => {
                let units = u.units(self);
                let unit = units.iter().find(|v| v.layers.contains(&l)).unwrap_or(&units[0]);
                return (unit.x0, unit.y0);
            }
            // serve8b: strips past the full ones skip their westmost slot.
            Some(S8b::Strips { full }) if l >= full * self.gw => {
                let m = l - full * self.gw;
                (full + m / (self.gw - 1), m % (self.gw - 1), 1)
            }
            _ => (l / self.gw, l % self.gw, 0),
        };
        let sx = if sy % 2 == 0 { self.gw - 1 - k } else { sxmin + k };
        (sx * self.px, sy * self.py)
    }
}

fn expect(t: &Tensor, name: &str, shape: &[usize]) -> R<()> {
    if t.shape == shape {
        Ok(())
    } else {
        Err(format!("{name} is {:?}; the program wants {shape:?}", t.shape))
    }
}

/// `n`-word blocks of a `PY` x `PX` rectangle of PEs (row-major by PE), with
/// one leading zero word a PE where `pad` (the tile's bank offset).
fn put(
    sink: &mut dyn Sink,
    symbol: &str,
    data: &[f32],
    at: (usize, usize),
    w: usize,
    h: usize,
    pad: bool,
) -> R<()> {
    let n = data.len() / (w * h);
    let words: Vec<u32> = if pad {
        data.chunks_exact(n)
            .flat_map(|pe| std::iter::once(0).chain(pe.iter().map(|x| x.to_bits())))
            .collect()
    } else {
        data.iter().map(|x| x.to_bits()).collect()
    };
    sink.h2d(symbol, rect(at.0, at.1, w, h), n + usize::from(pad), &words)
}

/// Every weight of `ckpt` (and zeroed KV) for `lay`, as `serve_lib2d.py`
/// lays them out.
#[allow(clippy::too_many_lines)]
pub fn place(lay: &Layout2d, ckpt: &Ckpt, rope_theta: f64, sink: &mut dyn Sink) -> R<()> {
    if lay.s8b.is_some() {
        return super::serve8b::place(lay, ckpt, rope_theta, sink);
    }
    let (px_n, py_n, hc, dc, gs, qh, ic, tc) =
        (lay.px, lay.py, lay.hc, lay.dc, lay.gs, lay.qh, lay.ic, lay.tc);
    let d = lay.head_dim();
    let nkv = px_n / gs;
    let hid = lay.hidden;
    let qdim = nkv * qh * d;
    let inter = px_n * ic;
    let inv_freq: Vec<f32> = (0..d / 2)
        .map(|i| rope_theta.powf(-((2 * i) as f64) / d as f64) as f32)
        .collect();
    let sqrt_d = (d as f64).sqrt() as f32;
    let get = |name: &str, shape: &[usize]| -> R<Tensor> {
        let t = ckpt.get(name)?;
        expect(&t, name, shape)?;
        Ok(t)
    };
    // Row `r` of a (rows, hid) matrix at padded hidden column `col` (zero
    // past `hid`).
    let at = |t: &Tensor, cols: usize, r: usize, col: usize| -> f32 {
        if col < cols { t.data[r * cols + col] } else { 0.0 }
    };
    let pes = py_n * px_n;
    if lay.paged.is_some() {
        for l in 0..lay.layers() {
            place_paged_layer(lay, ckpt, &inv_freq, l, sink)?;
        }
        return head(lay, ckpt, sink);
    }
    for l in 0..lay.layers() {
        let p = format!("layers.{l}.");
        let wq = get(&format!("{p}q"), &[qdim, hid])?;
        let wk = get(&format!("{p}k"), &[nkv * d, hid])?;
        let wv = get(&format!("{p}v"), &[nkv * d, hid])?;
        let wo = get(&format!("{p}o"), &[hid, qdim])?;
        let wg = get(&format!("{p}gate"), &[inter, hid])?;
        let wu = get(&format!("{p}up"), &[inter, hid])?;
        let wd = get(&format!("{p}down"), &[hid, inter])?;
        let qn = get(&format!("{p}q_norm"), &[d])?;
        let kn = get(&format!("{p}k_norm"), &[d])?;
        let qkv_rows = (qh + 2) * dc;
        let mut wqkv = vec![0f32; pes * hc * qkv_rows];
        let mut qkn = vec![0f32; pes * 2 * dc];
        let mut freq = vec![0f32; pes * dc];
        let mut wob = vec![0f32; pes * qh * dc * hc];
        let mut wgu = vec![0f32; pes * hc * 2 * ic];
        let mut wdb = vec![0f32; pes * ic * hc];
        for px in 0..px_n {
            let (g, j) = (px / gs, px % gs);
            for py in 0..py_n {
                let pe = py * px_n + px;
                for c in 0..hc {
                    let col = py * hc + c;
                    // by_row(concat(gate rows, up rows)): [c][r].
                    for r in 0..ic {
                        let row = px * ic + r;
                        wgu[(pe * hc + c) * 2 * ic + r] = at(&wg, hid, row, col);
                        wgu[(pe * hc + c) * 2 * ic + ic + r] = at(&wu, hid, row, col);
                        // wd[:, px*Ic..]: (hp, Ic) as [Ic][Hc].
                        wdb[(pe * ic + r) * hc + c] =
                            if col < hid { wd.data[col * inter + row] } else { 0.0 };
                    }
                }
            }
            if g >= nkv {
                continue;
            }
            let lo: Vec<usize> = (j * dc / 2..(j + 1) * dc / 2).collect();
            let dl: Vec<usize> = lo.iter().copied().chain(lo.iter().map(|&i| d / 2 + i)).collect();
            let qrows: Vec<usize> = (0..qh)
                .flat_map(|i| dl.iter().map(move |&dd| (g * qh + i) * d + dd))
                .collect();
            let kvr: Vec<usize> = dl.iter().map(|&dd| g * d + dd).collect();
            for py in 0..py_n {
                let pe = py * px_n + px;
                for c in 0..hc {
                    let col = py * hc + c;
                    let rows = qrows
                        .iter()
                        .map(|&r| at(&wq, hid, r, col))
                        .chain(kvr.iter().map(|&r| at(&wk, hid, r, col)))
                        .chain(kvr.iter().map(|&r| at(&wv, hid, r, col)));
                    for (r, x) in rows.enumerate() {
                        wqkv[(pe * hc + c) * qkv_rows + r] = x;
                    }
                }
                for (i, &dd) in dl.iter().enumerate() {
                    qkn[pe * 2 * dc + i] = qn.data[dd] / sqrt_d;
                    qkn[pe * 2 * dc + dc + i] = kn.data[dd];
                }
                // The step's rotation: cos and sin of each pair's frequency.
                for (i, &f) in lo.iter().enumerate() {
                    let fr = f64::from(inv_freq[f]);
                    freq[pe * dc + i] = fr.cos() as f32;
                    freq[pe * dc + dc / 2 + i] = fr.sin() as f32;
                }
                for (q, &qrow) in qrows.iter().enumerate() {
                    for c in 0..hc {
                        let col = py * hc + c;
                        wob[(pe * qh * dc + q) * hc + c] =
                            if col < hid { wo.data[col * qdim + qrow] } else { 0.0 };
                    }
                }
            }
        }
        let origin = lay.origin(l);
        for (sym, arr, pad) in [
            ("wqkv", &wqkv, true),
            ("qkn_w", &qkn, false),
            ("inv_freq", &freq, false),
            ("wo", &wob, true),
            ("wgu", &wgu, true),
            ("wd", &wdb, true),
        ] {
            put(sink, sym, arr, origin, px_n, py_n, pad)?;
        }
        let zeros_kv = vec![0f32; pes * (dc * tc + 1)];
        put(sink, "kc", &zeros_kv, origin, px_n, py_n, false)?;
        put(sink, "vc", &zeros_kv, origin, px_n, py_n, false)?;
        for (sym, name) in [
            ("gamma1", "attn_norm"),
            ("gamma2", "mlp_norm"),
        ] {
            let v = get(&format!("{p}{name}"), &[hid])?;
            let col: Vec<f32> = (0..pes)
                .flat_map(|pe| {
                    let py = pe / px_n;
                    (0..hc).map(move |c| py * hc + c)
                })
                .map(|i| v.data.get(i).copied().unwrap_or(0.0))
                .collect();
            put(sink, sym, &col, origin, px_n, py_n, false)?;
        }
        tracing::info!(layer = l, "spatial wafer: serve2d layer placed");
    }
    head(lay, ckpt, sink)
}

/// One layer grid of the batched build, as `serve_lib2d._layer_paged` (=
/// `qwen2d_paged/run.py`) lays it out: per column its kv head's dims `dl`,
/// the two query heads and k split in halves (`wqa`: dims [0, Dc/2), `wqv`:
/// the rest), v, the q/k norm weights (q's carrying 1/sqrt(head_dim)), the
/// RoPE tables cos/sin(p·f) for p = 256a, 16b, c, o, the MLP, the empty pool.
#[allow(clippy::too_many_lines)]
fn place_paged_layer(
    lay: &Layout2d,
    ckpt: &Ckpt,
    inv_freq: &[f32],
    l: usize,
    sink: &mut dyn Sink,
) -> R<()> {
    let shape = lay.paged.expect("a batched build");
    let (p_n, hc, dc, gs, ic) = (lay.py, lay.hc, lay.dc, lay.gs, lay.ic);
    let d = lay.head_dim();
    let dh = dc / 2;
    let nkv = p_n / gs;
    let hid = lay.hidden;
    let qdim = nkv * 2 * d;
    let inter = p_n * ic;
    let pes = p_n * p_n;
    let p = format!("layers.{l}.");
    let get = |name: &str, shape: &[usize]| -> R<Tensor> {
        let t = ckpt.get(name)?;
        expect(&t, name, shape)?;
        Ok(t)
    };
    let wq = get(&format!("{p}q"), &[qdim, hid])?;
    let wk = get(&format!("{p}k"), &[nkv * d, hid])?;
    let wv = get(&format!("{p}v"), &[nkv * d, hid])?;
    let wo = get(&format!("{p}o"), &[hid, qdim])?;
    let wg = get(&format!("{p}gate"), &[inter, hid])?;
    let wu = get(&format!("{p}up"), &[inter, hid])?;
    let wd = get(&format!("{p}down"), &[hid, inter])?;
    let qn = get(&format!("{p}q_norm"), &[d])?;
    let kn = get(&format!("{p}k_norm"), &[d])?;
    let sqrt_d = (d as f64).sqrt() as f32;
    let mut wqa = vec![0f32; pes * hc * 3 * dh];
    let mut wqv = vec![0f32; pes * hc * 3 * dh];
    let mut wvb = vec![0f32; pes * hc * dc];
    let mut qkn = vec![0f32; pes * 2 * 3 * dh];
    let mut rot_c = vec![0f32; pes * 3 * 16 * dh];
    let mut rot_s = vec![0f32; pes * 3 * 16 * dh];
    let mut wob = vec![0f32; pes * 2 * dc * hc];
    let mut wgu = vec![0f32; pes * hc * 2 * ic];
    let mut wdb = vec![0f32; pes * ic * hc];
    for px in 0..p_n {
        let (g, j) = (px / gs, px % gs);
        let lo: Vec<usize> = (j * dc / 2..(j + 1) * dc / 2).collect();
        let dl: Vec<usize> = lo.iter().copied().chain(lo.iter().map(|&i| d / 2 + i)).collect();
        let qa: Vec<usize> = dl.iter().map(|&dd| 2 * g * d + dd).collect();
        let qb: Vec<usize> = dl.iter().map(|&dd| (2 * g + 1) * d + dd).collect();
        let kvr: Vec<usize> = dl.iter().map(|&dd| g * d + dd).collect();
        // The three heads' rows (q a, q b, k) and their norm weights.
        let heads: [(&Tensor, &[usize]); 3] = [(&wq, &qa), (&wq, &qb), (&wk, &kvr)];
        let norm = |h: usize, i: usize| -> f32 {
            if h < 2 { qn.data[dl[i]] / sqrt_d } else { kn.data[dl[i]] }
        };
        for py in 0..p_n {
            let pe = py * p_n + px;
            for c in 0..hc {
                let col = py * hc + c;
                for (h, (t, rows)) in heads.iter().enumerate() {
                    for k in 0..dh {
                        // wqa/wqv: [Hc][3][DH], the first and second halves of dl.
                        wqa[((pe * hc + c) * 3 + h) * dh + k] = t.data[rows[k] * hid + col];
                        wqv[((pe * hc + c) * 3 + h) * dh + k] = t.data[rows[dh + k] * hid + col];
                    }
                }
                for (k, &r) in kvr.iter().enumerate() {
                    wvb[(pe * hc + c) * dc + k] = wv.data[r * hid + col];
                }
                for i in 0..ic {
                    let row = px * ic + i;
                    wgu[(pe * hc + c) * 2 * ic + i] = wg.data[row * hid + col];
                    wgu[(pe * hc + c) * 2 * ic + ic + i] = wu.data[row * hid + col];
                    wdb[(pe * ic + i) * hc + c] = wd.data[col * inter + row];
                }
                for (q, &r) in qa.iter().chain(&qb).enumerate() {
                    wob[(pe * 2 * dc + q) * hc + c] = wo.data[col * qdim + r];
                }
            }
            for h in 0..3 {
                for k in 0..dh {
                    qkn[((pe * 2) * 3 + h) * dh + k] = norm(h, k);
                    qkn[((pe * 2 + 1) * 3 + h) * dh + k] = norm(h, dh + k);
                }
            }
            for (lv, unit) in [256.0f64, 16.0, 1.0].into_iter().enumerate() {
                for a in 0..16 {
                    for (k, &f) in lo.iter().enumerate() {
                        let ang = a as f64 * unit * f64::from(inv_freq[f]);
                        rot_c[((pe * 3 + lv) * 16 + a) * dh + k] = ang.cos() as f32;
                        rot_s[((pe * 3 + lv) * 16 + a) * dh + k] = ang.sin() as f32;
                    }
                }
            }
        }
    }
    let origin = lay.origin(l);
    for (sym, arr, pad) in [
        ("wqa", &wqa, true),
        ("wqv", &wqv, true),
        ("wv", &wvb, true),
        ("qkn_w", &qkn, false),
        ("rot_c", &rot_c, false),
        ("rot_s", &rot_s, false),
        ("wo", &wob, true),
        ("wgu", &wgu, true),
        ("wd", &wdb, true),
    ] {
        put(sink, sym, arr, origin, p_n, p_n, pad)?;
    }
    let pool = vec![0f32; pes * (shape.blocks * dc * shape.page_size + 1)];
    put(sink, "kc", &pool, origin, p_n, p_n, false)?;
    put(sink, "vc", &pool, origin, p_n, p_n, false)?;
    for (sym, name) in [
        ("gamma1", "attn_norm"),
        ("gamma2", "mlp_norm"),
    ] {
        let v = get(&format!("{p}{name}"), &[hid])?;
        let col: Vec<f32> = (0..pes)
            .flat_map(|pe| {
                let py = pe / p_n;
                v.data[py * hc..(py + 1) * hc].iter().copied()
            })
            .collect();
        put(sink, sym, &col, origin, p_n, p_n, false)?;
    }
    tracing::info!(layer = l, "spatial wafer: batched serve2d layer placed");
    Ok(())
}

/// head2d's f32 tiles (one block): region PE (c, ry) holds dims c*Hc.. of
/// vocabulary rows ry*Vr.., dim-major after one leading pad; the guest VM
/// column (if any) after the root column gets zeros; the feeder row holds
/// the final norm's gamma.
fn head(lay: &Layout2d, ckpt: &Ckpt, sink: &mut dyn Sink) -> R<()> {
    let (hh, vr, hc, f) = (lay.hh, lay.vr, lay.hc, lay.feeder());
    let emb = ckpt.get("embed")?;
    if emb.shape.len() != 2 || emb.shape[1] != lay.hidden {
        return Err(format!("embed is {:?}", emb.shape));
    }
    let hid = lay.hidden;
    let have = emb.shape[0].min(lay.vocab);
    let n = hc * vr + 1;
    let r2x = lay.r2x();
    // Fabric column q of the block -> chunk column (None: the VM column).
    let vmc = lay.vmc;
    let chunk = |q: usize| -> Option<usize> {
        if q > r2x && q <= r2x + vmc {
            None
        } else if q > r2x + vmc {
            Some(q - vmc)
        } else {
            Some(q)
        }
    };
    // The fabric column runs the head's tiles go to: on a VM build not its
    // VM column (the VM's program exports 1-word `wt`/`gam` dummies a tile
    // would run over), so the chunk columns either side of it.
    let segments = head_runs(lay);
    for qs in &segments {
        check_head_columns(lay, "wt", qs.clone())?;
    }
    let tile_rows = |rows: std::ops::Range<usize>, qs: std::ops::Range<usize>| -> Vec<f32> {
        let w = qs.len();
        let mut t = vec![0f32; rows.len() * w * n];
        for (yi, ry) in rows.enumerate() {
            for (qi, q) in qs.clone().enumerate() {
                let Some(c) = chunk(q) else { continue };
                let base = (yi * w + qi) * n + 1;
                for h in 0..hc {
                    let dim = c * hc + h;
                    if dim >= hid {
                        continue;
                    }
                    for v in 0..vr {
                        let tok = ry * vr + v;
                        if tok < have {
                            t[base + h * vr + v] = emb.data[tok * hid + dim];
                        }
                    }
                }
            }
        }
        t
    };
    let words = |t: &[f32]| -> Vec<u32> { t.iter().map(|x| x.to_bits()).collect() };
    let hx = lay.head_x();
    // The regions above and below the feeder row, a band at a time.
    for qs in &segments {
        for (lo, hi, y0) in [(0, f, 0), (f, hh, f + 1)] {
            let mut y = lo;
            while y < hi {
                let rows = (hi - y).min(64);
                let t = tile_rows(y..y + rows, qs.clone());
                sink.h2d("wt", rect(hx + qs.start, y0 + (y - lo), qs.len(), rows), n, &words(&t))?;
                y += rows;
            }
        }
    }
    let norm = ckpt.get("final_norm")?;
    expect(&norm, "final_norm", &[hid])?;
    for qs in &segments {
        let gam: Vec<f32> = qs
            .clone()
            .flat_map(|q| {
                let c = chunk(q);
                (0..hc).map(move |h| c.map(|c| c * hc + h))
            })
            .map(|dim| dim.and_then(|d| norm.data.get(d).copied()).unwrap_or(0.0))
            .collect();
        sink.h2d("gam", rect(hx + qs.start, f, qs.len(), 1), hc, &words(&gam))?;
    }
    Ok(())
}

/// One launch: the forced tokens, `steps` head iterations from `start`;
/// the argmax after each step, and the last step's logits when asked.
pub fn run(
    lay: &Layout2d,
    rt: &mut Runtime,
    forced: &[u32],
    steps: usize,
    start: usize,
    want_logits: bool,
) -> R<(Vec<u32>, Option<Vec<f32>>)> {
    if forced.is_empty() || steps + 1 > lay.cap || start + steps > lay.positions() {
        return Err(format!(
            "a launch of {steps} steps at position {start} does not fit the program \
             ({} iterations, {} positions; a launch forces its first token)",
            lay.cap,
            lay.positions()
        ));
    }
    let e = |e: crate::sdk::Error| e.to_string();
    let (rx, ry) = lay.root();
    let one = rect(rx, ry, 1, 1);
    let mut inn = vec![SAMPLE; lay.cap];
    inn[..forced.len()].copy_from_slice(forced);
    rt.memcpy_h2d_nonblock(SYM.in_tokens, inn, one, lay.cap, Order::RowMajor, DataType::Bits32)
        .map_err(e)?;
    rt.launch_nonblock(SYM.compute, &[steps as u32, start as u32])
        .map_err(e)?;
    let mut out = vec![0u32; lay.cap];
    let read = rt.memcpy_d2h(SYM.out_tokens, &mut out, one, lay.cap, Order::RowMajor, DataType::Bits32);
    rt.wait_pending();
    read.map_err(e)?;
    let y = if want_logits {
        if rt.rpc().get(SYM.yscale).is_err() {
            return Err(
                "this serve2d build exports no `yscale` (the final norm's 1/rms): its `y` \
                 is not the logits; it serves greedy (token-loop) epilogues only"
                    .into(),
            );
        }
        let mut scale = [0u32; 1];
        rt.memcpy_d2h(SYM.yscale, &mut scale, rect(rx, 0, 1, 1), 1, Order::RowMajor, DataType::Bits32)
            .map_err(e)?;
        let scale = f32::from_bits(scale[0]);
        let (f, hh, vr) = (lay.feeder(), lay.hh, lay.vr);
        let mut a = vec![0u32; f * vr];
        rt.memcpy_d2h(SYM.logits, &mut a, rect(rx, 0, 1, f), vr, Order::RowMajor, DataType::Bits32)
            .map_err(e)?;
        let mut b = vec![0u32; (hh - f) * vr];
        rt.memcpy_d2h(SYM.logits, &mut b, rect(rx, f + 1, 1, hh - f), vr, Order::RowMajor, DataType::Bits32)
            .map_err(e)?;
        Some(
            a.into_iter()
                .chain(b)
                .take(lay.vocab)
                .map(|w| f32::from_bits(w) * scale)
                .collect(),
        )
    } else {
        None
    };
    Ok((out[..steps].to_vec(), y))
}

/// One lane of a batched launch: its first pass's RoPE position and KV
/// write slot, its pages (covering every slot the launch writes or it
/// sees), the slots its first pass attends (`None`: every slot up to its
/// write slot), and the tokens forced at passes 0, 1, ... (at least one; a
/// later pass takes the lane's own argmax).
#[derive(Debug, Clone)]
pub struct BatchLane {
    pub pos: u32,
    pub slot: u32,
    pub pages: Vec<u32>,
    pub visible: Option<Vec<u32>>,
    pub forced: Vec<u32>,
}

/// One launch of `steps` passes over `lanes` (at most
/// [`Layout2d::launch_lanes`]; on a pipelined build each B of them is a
/// micro-batch, and a short batch repeats its first lane, the repeats'
/// answers dropped): every lane's argmax after each pass, and with
/// `want_logits` the last pass's logits rows (`hlogits` x `yscale`) of the
/// first [`Layout2d::logit_lanes`] lanes (the head keeps no more).
pub fn fire(
    lay: &Layout2d,
    rt: &mut Runtime,
    lanes: &[BatchLane],
    steps: usize,
    want_logits: bool,
    vm_written: &mut Vec<u32>,
) -> R<(Vec<Vec<u32>>, Vec<Vec<f32>>)> {
    fire_lanes(lay, rt, lanes, steps, want_logits, None, vm_written)
}

/// [`fire`] on a VM build, `vm_lanes[v]` 1 for each virtual lane the VM
/// samples (`None`: none; the head's argmax every lane).
pub fn fire_lanes(
    lay: &Layout2d,
    rt: &mut Runtime,
    lanes: &[BatchLane],
    steps: usize,
    want_logits: bool,
    vm_lanes: Option<&[u32]>,
    vm_written: &mut Vec<u32>,
) -> R<(Vec<Vec<u32>>, Vec<Vec<f32>>)> {
    let shape = lay.paged.ok_or("a batched launch on a one-sequence build")?;
    let b_n = shape.lanes;
    let most = lay.launch_lanes();
    if lanes.is_empty() || lanes.len() > most || steps == 0 || steps + 1 > lay.cap {
        return Err(format!(
            "{} lane(s) of {steps} pass(es) for a build of {most} lanes and {} iterations",
            lanes.len(),
            lay.cap
        ));
    }
    if let Some(lane) = lanes.iter().find(|l| l.forced.is_empty() || l.forced.len() > lay.cap) {
        return Err(format!("a lane forces {} token(s)", lane.forced.len()));
    }
    // A table for pool rows `rows` (serve8b_region's units differ; a page id
    // is still one of the pool's `shape.pages()`).
    let table_rows = |batch: &[BatchLane], rows: usize| -> R<Vec<u32>> {
        if let Some(g) = batch.iter().flat_map(|l| &l.pages).find(|&&g| g as usize >= shape.pages()) {
            return Err(format!("a lane names page {g} of a pool of {}", shape.pages()));
        }
        let table_lanes: Vec<super::paged::Lane> = batch
            .iter()
            .map(|l| super::paged::Lane {
                pos: l.pos,
                slot: l.slot,
                pages: l.pages.clone(),
                visible: l.visible.clone(),
            })
            .collect();
        super::paged::table(&super::paged::Shape { rows, ..shape }, &table_lanes, steps as u32)
    };
    let table_of = |batch: &[BatchLane]| table_rows(batch, shape.rows);
    let e = |e: crate::sdk::Error| e.to_string();
    // Micro-batches (one on a build without a ring).
    let mbs: Vec<&[BatchLane]> = lanes.chunks(b_n).collect();
    let slots = lay.mmax.unwrap_or(1);
    match lay.mmax {
        None => {
            // Every fabric row of the grids: its grid row's table row (a
            // serve8b_region unit's: its own rows').
            let units = matches!(lay.s8b, Some(super::serve8b::S8b::Units(_)));
            let mut tabs: Vec<(usize, Vec<u32>)> = Vec::new();
            for grids in lay.grid_areas() {
                let rows = if units { grids.h as usize } else { lay.py };
                if !tabs.iter().any(|t| t.0 == rows) {
                    tabs.push((rows, table_rows(lanes, rows)?));
                }
                let tab = &tabs.iter().find(|t| t.0 == rows).expect("built above").1;
                let all: Vec<u32> = (0..grids.h as usize / rows).flat_map(|_| tab.iter().copied()).collect();
                rt.memcpy_h2d_rowbcast(SYM.table, &all, grids, shape.row_words(), DataType::Bits32)
                    .map_err(e)?;
            }
        }
        Some(_) => {
            let mtab = pipe_tables(lay, &mbs.iter().map(|mb| table_of(mb)).collect::<R<Vec<_>>>()?);
            let n = slots * shape.row_words();
            head_h2d(lay, rt, SYM.ring_table, &mtab, lay.feeder(), 1, n)?;
            if lay.vm.is_some() {
                // The VM's per-lane control, every launch: (gmode_l, the
                // lane's forced prefix[, the offset to its turn's last row])
                // a virtual lane m·B + b.
                // guest4hz adds the offset to the lane's turn's last row: 0, a
                // turn being one row.
                let w = lay.vmlane_words;
                let mut control = vec![0u32; slots * b_n * w];
                for (m, mb) in mbs.iter().enumerate() {
                    for b in 0..b_n {
                        let lane = mb.get(b).unwrap_or(&mb[0]);
                        let v = m * b_n + b;
                        control[v * w] = vm_lanes.and_then(|g| g.get(v).copied()).unwrap_or(0);
                        control[v * w + 1] = lane.forced.len() as u32;
                        // guest4h2s: a head lane past set A's VM lanes is a
                        // set B turn, so set A counts only its own.
                        if w >= 4 && lay.vm_set_a_lanes.is_some_and(|a| v >= a) {
                            control[v * w + 3] = 1;
                        }
                    }
                }
                // Every head PE holds it (MB words of copy a launch at full
                // size): written only when it changes (`vm_written`, the
                // control last written; a launch with the VM off, all
                // zeros, writes it once).
                // Only the root column reads it (its region PEs put each row's
                // ctl, its feeder decides): one column, not the head's
                // width.
                if *vm_written != control {
                    let all: Vec<u32> = (0..=lay.hh).flat_map(|_| control.iter().copied()).collect();
                    check_head_columns(lay, SYM.vm_lanes, lay.r2x()..lay.r2x() + 1)?;
                    let col = rect(lay.head_x() + lay.r2x(), 0, 1, lay.hh + 1);
                    rt.memcpy_h2d(SYM.vm_lanes, &all, col, control.len(), Order::RowMajor, DataType::Bits32).map_err(e)?;
                    *vm_written = control;
                }
            }
        }
    }
    let mut inn = vec![SAMPLE; slots * b_n * lay.cap];
    for (m, mb) in mbs.iter().enumerate() {
        for b in 0..b_n {
            let lane = mb.get(b).unwrap_or(&mb[0]);
            let at = (m * b_n + b) * lay.cap;
            inn[at..at + lane.forced.len()].copy_from_slice(&lane.forced);
        }
    }
    let (rx, ry) = lay.root();
    let one = rect(rx, ry, 1, 1);
    let words = inn.len();
    rt.memcpy_h2d_nonblock(SYM.in_tokens, inn, one, words, Order::RowMajor, DataType::Bits32)
        .map_err(e)?;
    // The batched build ignores the second argument; the ring bootstraps
    // that many micro-batches.
    let second = if lay.mmax.is_some() { mbs.len() as u32 } else { 0 };
    rt.launch_nonblock(SYM.compute, &[steps as u32, second]).map_err(e)?;
    let mut out = vec![0u32; words];
    let read = rt.memcpy_d2h(SYM.out_tokens, &mut out, one, words, Order::RowMajor, DataType::Bits32);
    rt.wait_pending();
    read.map_err(e)?;
    // Lane i is micro-batch i / B's lane i % B: [i][CAP] either way.
    let tokens: Vec<Vec<u32>> = (0..lanes.len())
        .map(|i| out[i * lay.cap..i * lay.cap + steps].to_vec())
        .collect();
    let logits = if want_logits {
        logits_rows(lay, rt, lanes.len())?
    } else {
        Vec::new()
    };
    Ok((tokens, logits))
}

/// The head's chunk columns as runs of head columns (fabric x - head_x):
/// a VM build's VM columns R2X+1 .. R2X+VMC run the VM's program, where the
/// head's names (`mtab`, `vmlane`, `wt`, `gam`, ...) are one-word dummies an
/// n-word write would run over, so the runs either side of them.
pub fn head_runs(lay: &Layout2d) -> Vec<std::ops::Range<usize>> {
    let cw = lay.chunk_cols() + lay.vmc;
    // The VM columns (or, VM 0, their stand-ins) hold head symbols as
    // one-word dummies: head writes go either side of them.
    let runs = if lay.vmc > 0 {
        vec![0..lay.r2x() + 1, lay.r2x() + 1 + lay.vmc..cw]
    } else {
        vec![0..cw]
    };
    runs.into_iter().filter(|r| !r.is_empty()).collect()
}

/// Refuses a write of head-only `symbol` whose rectangle (head columns
/// `cols`) covers a VM column: the bug class that overran the VM PEs (head
/// weights, `mtab`) on the simulator.
pub fn check_head_columns(lay: &Layout2d, symbol: &str, cols: std::ops::Range<usize>) -> R<()> {
    if lay.vmc > 0 {
        let vm = lay.r2x() + 1..lay.r2x() + 1 + lay.vmc;
        if cols.start < vm.end && vm.start < cols.end {
            return Err(format!(
                "`{symbol}` is a head symbol: head columns {cols:?} cover the VM columns {vm:?}, where it is a \
                 one-word dummy"
            ));
        }
    }
    Ok(())
}

/// `data` (`per_col` words for each head column c of rows `y0..y0+h`, laid
/// out row by row over all C + VMC head columns, n words a PE) written to
/// the head's chunk columns only ([`head_runs`]).
fn head_h2d(lay: &Layout2d, rt: &mut Runtime, symbol: &str, data: &[u32], y0: usize, h: usize, n: usize) -> R<()> {
    let cw = lay.chunk_cols() + lay.vmc;
    for cols in head_runs(lay) {
        check_head_columns(lay, symbol, cols.clone())?;
        let words: Vec<u32> = (0..h)
            .flat_map(|row| data[(row * cw + cols.start) * n..(row * cw + cols.end) * n].iter().copied())
            .collect();
        rt.memcpy_h2d(symbol, &words, rect(lay.head_x() + cols.start, y0, cols.len(), h), n, Order::RowMajor, DataType::Bits32)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// The ring's `mtab` over the head's feeder row: feeder of chunk j (column
/// j, the VM column after the root zero) holds per micro-batch slot m
/// row j of micro-batch m's table (`tables[m]`, unused slots zero).
fn pipe_tables(lay: &Layout2d, tables: &[Vec<u32>]) -> Vec<u32> {
    let w = lay.paged.map_or(0, |s| s.row_words());
    let slots = lay.mmax.unwrap_or(1);
    let mut mtab = Vec::with_capacity((lay.py + lay.vmc) * slots * w);
    for c in 0..lay.py {
        for m in 0..slots {
            match tables.get(m) {
                Some(t) => mtab.extend_from_slice(&t[c * w..(c + 1) * w]),
                None => mtab.extend(std::iter::repeat_n(0, w)),
            }
        }
        if c == lay.r2x() {
            mtab.extend(std::iter::repeat_n(0, lay.vmc * slots * w));
        }
    }
    mtab
}

/// The last pass's logits rows of lanes `0..lanes` (at most HLB): the root
/// column's `hlogits` (region row r, lane b at words `(r * HLB + b) * Vr`,
/// rows in vocab order) x `yscale[b]`.
pub fn logits_rows(lay: &Layout2d, rt: &mut Runtime, lanes: usize) -> R<Vec<Vec<f32>>> {
    let e = |e: crate::sdk::Error| e.to_string();
    let (rx, _) = lay.root();
    let (f, hh, vr, hlb) = (lay.feeder(), lay.hh, lay.vr, lay.logit_lanes());
    let mut a = vec![0u32; f * hlb * vr];
    rt.memcpy_d2h(SYM.logit_lanes, &mut a, rect(rx, 0, 1, f), hlb * vr, Order::RowMajor, DataType::Bits32)
        .map_err(e)?;
    let mut c = vec![0u32; (hh - f) * hlb * vr];
    rt.memcpy_d2h(SYM.logit_lanes, &mut c, rect(rx, f + 1, 1, hh - f), hlb * vr, Order::RowMajor, DataType::Bits32)
        .map_err(e)?;
    let mut scale = vec![0u32; hlb];
    rt.memcpy_d2h(SYM.yscale, &mut scale, rect(rx, 0, 1, 1), hlb, Order::RowMajor, DataType::Bits32)
        .map_err(e)?;
    let rows: Vec<u32> = a.into_iter().chain(c).collect();
    Ok((0..lanes.min(hlb))
        .map(|b| {
            let ys = f32::from_bits(scale[b]);
            (0..hh)
                .flat_map(|r| rows[(r * hlb + b) * vr..(r * hlb + b + 1) * vr].iter())
                .take(lay.vocab)
                .map(|&w| f32::from_bits(w) * ys)
                .collect()
        })
        .collect())
}

/// `copy_kv`'s whole pages on every layer's pool: two launches a pair, or
/// on a `KVCP` 0 build the host's read-patch-write of each grid's rows.
pub fn copy_pages(lay: &Layout2d, rt: &mut Runtime, pairs: &[(u32, u32)]) -> R<()> {
    let Some(shape) = lay.paged else {
        return Err("page copies on a one-sequence build".into());
    };
    if let Some(super::serve8b::S8b::Units(u)) = lay.s8b {
        return super::serve8b_region::copy_pages(lay, &u, rt, pairs);
    }
    if !lay.host_copy {
        if lay.s8b.is_some() {
            return super::serve8b::copy_pages(lay, rt, pairs);
        }
        return super::paged::copy_pages(rt, &super::paged::Symbols::default(), pairs);
    }
    super::paged::copy_pages_host(rt, &host_pool(lay, shape), pairs)
}

/// The pool as the host reads and writes its pages: every layer's grid.
fn host_pool(lay: &Layout2d, shape: super::paged::Shape) -> super::paged::HostPool {
    super::paged::HostPool {
        grids: (0..lay.layers()).map(|l| lay.origin(l)).collect(),
        cols: lay.px,
        shape,
        cells: lay.dc * shape.page_size,
        data_type: if lay.s8b.is_some() { DataType::Bits16 } else { DataType::Bits32 },
    }
}

/// The host pool of a serve2d build whose pages the host reads and writes
/// (the host tier's suspended pages); `None` for serve8b's (its own
/// layout) and a one-sequence build.
fn page_pool(lay: &Layout2d) -> R<super::paged::HostPool> {
    let shape = lay.paged.ok_or("host pages on a one-sequence build")?;
    if lay.s8b.is_some() {
        return Err("host pages on a serve8b build (serve8b's page_out / page_in)".into());
    }
    Ok(host_pool(lay, shape))
}

/// Words of one page's host copy, if the host can keep this build's pages
/// (serve8b's: [`super::serve8b::page_words`]).
pub fn page_words(lay: &Layout2d) -> Option<usize> {
    if lay.s8b.is_some() {
        return super::serve8b::page_words(lay).ok();
    }
    page_pool(lay).ok().map(|pool| super::paged::page_words(&pool))
}

/// Page `page`'s cells off the device: through the staging slots on a build
/// with them ([`super::paged::read_page_staged`]), else whole rows
/// ([`super::paged::read_page_host`]); serve8b's [`super::serve8b::page_out`].
pub fn read_page(lay: &Layout2d, rt: &mut Runtime, page: u32) -> R<Vec<u32>> {
    if lay.s8b.is_some() {
        return super::serve8b::page_out(lay, rt, page);
    }
    let pool = page_pool(lay)?;
    if lay.kvst > 0 {
        return super::paged::read_page_staged(rt, &pool, lay.kvst, page);
    }
    super::paged::read_page_host(rt, &pool, page)
}

/// Several pages' cells off the device: through the staging slots on a
/// build with them, a round of pages a memcpy per grid and symbol
/// ([`super::paged::read_pages_staged`]); otherwise a page at a time.
pub fn read_pages(lay: &Layout2d, rt: &mut Runtime, pages: &[u32]) -> R<Vec<Vec<u32>>> {
    if lay.s8b.is_some() {
        return super::serve8b::page_out_many(lay, rt, pages);
    }
    if lay.kvst > 0 {
        return super::paged::read_pages_staged(rt, &page_pool(lay)?, lay.kvst, pages);
    }
    pages.iter().map(|&p| read_page(lay, rt, p)).collect()
}

/// [`read_pages`]' reverse: host copies back into their device pages.
pub fn write_pages(lay: &Layout2d, rt: &mut Runtime, pages: &[(u32, Vec<u32>)]) -> R<()> {
    if lay.s8b.is_some() {
        return super::serve8b::page_in_many(lay, rt, pages);
    }
    if lay.kvst > 0 {
        return super::paged::write_pages_staged(rt, &page_pool(lay)?, lay.kvst, pages);
    }
    pages.iter().try_for_each(|(p, w)| write_page(lay, rt, *p, w))
}

/// A page's host copy back into page `page`.
pub fn write_page(lay: &Layout2d, rt: &mut Runtime, page: u32, words: &[u32]) -> R<()> {
    if lay.s8b.is_some() {
        return super::serve8b::page_in(lay, rt, page, words);
    }
    let pool = page_pool(lay)?;
    if lay.kvst > 0 {
        return super::paged::write_page_staged(rt, &pool, lay.kvst, page, words);
    }
    super::paged::write_page_host(rt, &pool, page, words)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fork 5's simulator pipe build (`hb2`): 8 feeders, the VM column
    /// after the root (column 6), MMAX 3 slots of B 2 lanes.
    fn pipe() -> Layout2d {
        let params = [
            ("PX", 8), ("PY", 8), ("Hc", 16), ("HR", 128), ("Dc", 8), ("Gs", 4), ("Qh", 2),
            ("Ic", 24), ("Tc", 1), ("W16", 0), ("KV16", 0), ("WF32", 1), ("GRID", 1),
            ("HEADMODE", 2), ("B", 2), ("PS", 4), ("NB", 4), ("E", 2), ("MMAX", 3), ("GW", 2),
            ("GH", 4), ("HH", 32), ("Vr", 32), ("Gx", 4), ("VMC", 1), ("vocab", 1024), ("CAP", 4),
        ];
        Layout2d::of(|k| params.iter().find(|p| p.0 == k).map(|p| p.1))
            .expect("serve2d params")
            .expect("a layout")
    }

    #[test]
    fn hlbn_caps_the_lanes_a_launch_samples_on_the_host() {
        assert_eq!(pipe().logit_lanes(), 2);
        let lay = Layout2d { hlbn: Some(1), ..pipe() };
        assert_eq!(lay.logit_lanes(), 1);
        assert_eq!(Layout2d { hlbn: Some(8), ..pipe() }.logit_lanes(), 2);
    }

    #[test]
    fn head_writes_stay_off_stand_in_vm_columns_with_the_vm_off() {
        // VMC 1, VM 0: the VM column is a stand-in, and head symbols there
        // are still one-word dummies (0108's wrong tokens on the host).
        let params = [
            ("PX", 4), ("PY", 4), ("Hc", 16), ("HR", 64), ("Dc", 8), ("Gs", 4), ("Qh", 2),
            ("Ic", 16), ("Tc", 1), ("W16", 0), ("KV16", 0), ("WF32", 1), ("GRID", 1),
            ("HEADMODE", 2), ("B", 2), ("PS", 16), ("NB", 2), ("E", 2), ("MMAX", 2), ("GW", 2),
            ("GH", 4), ("HH", 16), ("Vr", 32), ("Gx", 4), ("VMC", 1), ("vocab", 250), ("CAP", 8),
        ];
        let lay = Layout2d::of(|k| params.iter().find(|p| p.0 == k).map(|p| p.1)).unwrap().unwrap();
        assert!(lay.vm.is_none());
        assert!(check_head_columns(&lay, "mtab", 0..lay.py + lay.vmc).unwrap_err().contains("VM columns"));
        assert_eq!(head_runs(&lay), vec![0..3, 4..5]);
    }

    #[test]
    fn head_writes_stay_off_the_vm_columns() {
        let params = [
            ("PX", 4), ("PY", 4), ("Hc", 16), ("HR", 64), ("Dc", 8), ("Gs", 4), ("Qh", 2),
            ("Ic", 16), ("Tc", 1), ("W16", 0), ("KV16", 0), ("WF32", 1), ("GRID", 1),
            ("HEADMODE", 2), ("B", 2), ("PS", 16), ("NB", 2), ("E", 2), ("MMAX", 2), ("GW", 2),
            ("GH", 4), ("HH", 16), ("Vr", 32), ("Gx", 4), ("VMC", 1), ("vocab", 250), ("CAP", 8),
            ("VM", 1), ("max_ops", 48), ("nreg", 6), ("pool_words", 64), ("feat", 99),
        ];
        let lay = Layout2d::of(|k| params.iter().find(|p| p.0 == k).map(|p| p.1)).unwrap().unwrap();
        assert!(lay.vm.is_some());
        // The old `mtab` rectangle, every head column: it covers the VM column 3.
        assert!(check_head_columns(&lay, "mtab", 0..lay.py + lay.vmc).unwrap_err().contains("VM columns"));
        // The chunk columns either side of it.
        assert_eq!(head_runs(&lay), vec![0..3, 4..5]);
        assert!(head_runs(&lay).into_iter().all(|r| check_head_columns(&lay, "mtab", r).is_ok()));
    }

    #[test]
    fn ring_lanes_and_feeder_tables() {
        let lay = pipe();
        assert_eq!((lay.launch_lanes(), lay.logit_lanes(), lay.r2x()), (6, 2, 6));
        let w = lay.paged.unwrap().row_words();
        // Micro-batch m's table: word i of row r is 1000 m + 100 r + i.
        let tables: Vec<Vec<u32>> = (0..2)
            .map(|m| (0..8 * w).map(|k| (1000 * m + 100 * (k / w) + k % w) as u32).collect())
            .collect();
        let mtab = pipe_tables(&lay, &tables);
        let pe = 3 * w;
        assert_eq!(mtab.len(), 9 * pe);
        let at = |col: usize, m: usize, i: usize| mtab[col * pe + m * w + i];
        assert_eq!((at(0, 0, 0), at(0, 1, 5), at(0, 2, 5)), (0, 1005, 0));
        assert_eq!((at(6, 1, 1), at(7, 0, 3), at(7, 1, 0)), (1601, 0, 0));
        assert_eq!((at(8, 0, 2), at(8, 1, 2)), (702, 1702));
    }
}
