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
    /// The serve2d shape when `get` names one (`HEADMODE` 2, f32 weights).
    pub fn of(get: impl Fn(&str) -> Option<usize>) -> Option<R<Layout2d>> {
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
                vmc: p("VMC")?,
                vocab: p("vocab")?,
                cap: p("CAP")?,
            };
            if lay.py * lay.hc < lay.hidden || lay.gs == 0 || lay.gx == 0 || lay.vmc > 1 {
                return Err(format!("serve2d params do not make a layout: {lay:?}"));
            }
            Ok(lay)
        })();
        Some(lay)
    }

    pub fn layers(&self) -> usize {
        self.gw * self.gh
    }

    /// Head dim (a kv group's `Gs` slices of `Dc`).
    pub fn head_dim(&self) -> usize {
        self.gs * self.dc
    }

    pub fn positions(&self) -> usize {
        self.py * self.tc
    }

    /// The feeder row.
    pub fn feeder(&self) -> usize {
        self.hh / 2
    }

    /// The head's first column.
    pub fn head_x(&self) -> usize {
        self.gw * self.px
    }

    /// The root column inside the head block.
    fn r2x(&self) -> usize {
        let ngx = self.py / self.gx;
        (ngx / 2) * self.gx + self.gx / 2
    }

    /// The final root (fabric): `in_tokens`, `out_tokens`, the root column.
    pub fn root(&self) -> (usize, usize) {
        (self.head_x() + self.r2x(), self.feeder())
    }

    /// Layer `l`'s grid's top-left PE.
    pub fn origin(&self, l: usize) -> (usize, usize) {
        let (sy, k) = (l / self.gw, l % self.gw);
        let sx = if sy % 2 == 0 { self.gw - 1 - k } else { k };
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
    for l in 0..lay.layers() {
        let p = format!("model.layers.{l}.");
        let wq = get(&format!("{p}self_attn.q_proj.weight"), &[qdim, hid])?;
        let wk = get(&format!("{p}self_attn.k_proj.weight"), &[nkv * d, hid])?;
        let wv = get(&format!("{p}self_attn.v_proj.weight"), &[nkv * d, hid])?;
        let wo = get(&format!("{p}self_attn.o_proj.weight"), &[hid, qdim])?;
        let wg = get(&format!("{p}mlp.gate_proj.weight"), &[inter, hid])?;
        let wu = get(&format!("{p}mlp.up_proj.weight"), &[inter, hid])?;
        let wd = get(&format!("{p}mlp.down_proj.weight"), &[hid, inter])?;
        let qn = get(&format!("{p}self_attn.q_norm.weight"), &[d])?;
        let kn = get(&format!("{p}self_attn.k_norm.weight"), &[d])?;
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
            ("gamma1", "input_layernorm.weight"),
            ("gamma2", "post_attention_layernorm.weight"),
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

/// head2d's f32 tiles (one block): region PE (c, ry) holds dims c*Hc.. of
/// vocabulary rows ry*Vr.., dim-major after one leading pad; the guest VM
/// column (if any) after the root column gets zeros; the feeder row holds
/// the final norm's gamma.
fn head(lay: &Layout2d, ckpt: &Ckpt, sink: &mut dyn Sink) -> R<()> {
    let (hh, vr, c_n, hc, f) = (lay.hh, lay.vr, lay.py, lay.hc, lay.feeder());
    let emb = ckpt.get("model.embed_tokens.weight")?;
    if emb.shape.len() != 2 || emb.shape[1] != lay.hidden {
        return Err(format!("model.embed_tokens.weight is {:?}", emb.shape));
    }
    let hid = lay.hidden;
    let have = emb.shape[0].min(lay.vocab);
    let cw = c_n + lay.vmc;
    let n = hc * vr + 1;
    let r2x = lay.r2x();
    // Fabric column q of the block -> chunk column (None: the VM column).
    let chunk = |q: usize| -> Option<usize> {
        if lay.vmc == 1 && q == r2x + 1 {
            None
        } else if lay.vmc == 1 && q > r2x + 1 {
            Some(q - 1)
        } else {
            Some(q)
        }
    };
    let tile_rows = |rows: std::ops::Range<usize>| -> Vec<f32> {
        let mut t = vec![0f32; rows.len() * cw * n];
        for (yi, ry) in rows.enumerate() {
            for q in 0..cw {
                let Some(c) = chunk(q) else { continue };
                let base = (yi * cw + q) * n + 1;
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
    for (lo, hi, y0) in [(0, f, 0), (f, hh, f + 1)] {
        let mut y = lo;
        while y < hi {
            let rows = (hi - y).min(64);
            let t = tile_rows(y..y + rows);
            sink.h2d("wt", rect(hx, y0 + (y - lo), cw, rows), n, &words(&t))?;
            y += rows;
        }
    }
    let norm = ckpt.get("model.norm.weight")?;
    expect(&norm, "model.norm.weight", &[hid])?;
    let gam: Vec<f32> = (0..cw)
        .flat_map(|q| {
            let c = chunk(q);
            (0..hc).map(move |h| c.map(|c| c * hc + h))
        })
        .map(|dim| dim.and_then(|d| norm.data.get(d).copied()).unwrap_or(0.0))
        .collect();
    sink.h2d("gam", rect(hx, f, cw, 1), hc, &words(&gam))
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
    rt.memcpy_h2d_nonblock("in_tokens", inn, one, lay.cap, Order::RowMajor, DataType::Bits32)
        .map_err(e)?;
    rt.launch_nonblock("compute", &[steps as u32, start as u32])
        .map_err(e)?;
    let mut out = vec![0u32; lay.cap];
    let read = rt.memcpy_d2h("out_tokens", &mut out, one, lay.cap, Order::RowMajor, DataType::Bits32);
    rt.wait_pending();
    read.map_err(e)?;
    let y = if want_logits {
        if rt.rpc().get("yscale").is_err() {
            return Err(
                "this serve2d build exports no `yscale` (the final norm's 1/rms): its `y` \
                 is not the logits; it serves greedy (token-loop) epilogues only"
                    .into(),
            );
        }
        let mut scale = [0u32; 1];
        rt.memcpy_d2h("yscale", &mut scale, rect(rx, 0, 1, 1), 1, Order::RowMajor, DataType::Bits32)
            .map_err(e)?;
        let scale = f32::from_bits(scale[0]);
        let (f, hh, vr) = (lay.feeder(), lay.hh, lay.vr);
        let mut a = vec![0u32; f * vr];
        rt.memcpy_d2h("y", &mut a, rect(rx, 0, 1, f), vr, Order::RowMajor, DataType::Bits32)
            .map_err(e)?;
        let mut b = vec![0u32; (hh - f) * vr];
        rt.memcpy_d2h("y", &mut b, rect(rx, f + 1, 1, hh - f), vr, Order::RowMajor, DataType::Bits32)
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
