//! The serve2d wafer program (`spatial/csl/serve2d`) and its host side: the
//! decoder layers as qwen2d grids, head2d east of them (`HEADMODE` 2), behind
//! a request API of tokens in and argmaxes out.
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
use super::program::ProgramSpec;
use super::wafer::{SAMPLE, inv_freq, rect};
use crate::sdk::{DataType, Order};
use crate::spatial::device::Device;

type R<T> = std::result::Result<T, String>;

/// `n`-word blocks of a `PY` x `PX` rectangle of PEs (row-major by PE), with
/// one leading zero word a PE where `pad` (the tile's bank offset).
fn put(
    dev: &mut dyn Device,
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
    dev.put(symbol, rect(at.0, at.1, w, h), n + usize::from(pad), &words)
}

/// Every layer grid's weights (and zeroed KV) of `ckpt` for `lay`, laid
/// out as the qwen2d kernels read them.
#[allow(clippy::too_many_lines)]
pub(crate) fn place_layers(
    lay: &ProgramSpec,
    ckpt: &Ckpt,
    rope_theta: f64,
    dev: &mut dyn Device,
) -> R<()> {
    let (px_n, py_n, hc, dc, gs, qh, ic, tc) = (
        lay.px(),
        lay.py(),
        lay.hc(),
        lay.dc(),
        lay.gs(),
        lay.qh(),
        lay.ic(),
        lay.tc(),
    );
    let d = lay.head_dim();
    let nkv = px_n / gs;
    let hid = lay.hidden();
    let qdim = nkv * qh * d;
    let inter = px_n * ic;
    let inv_freq = inv_freq(rope_theta, d);
    let sqrt_d = (d as f64).sqrt() as f32;
    let get = |name: &str, shape: &[usize]| -> R<Tensor> {
        let t = ckpt.get(name)?;
        t.expect_shape(name, shape)?;
        Ok(t)
    };
    // Row `r` of a (rows, hid) matrix at padded hidden column `col` (zero
    // past `hid`).
    let at = |t: &Tensor, cols: usize, r: usize, col: usize| -> f32 {
        if col < cols {
            t.data[r * cols + col]
        } else {
            0.0
        }
    };
    let pes = py_n * px_n;
    if lay.paged().is_some() {
        for l in 0..lay.layers() {
            place_paged_layer(lay, ckpt, &inv_freq, l, dev)?;
        }
        return Ok(());
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
                        wdb[(pe * ic + r) * hc + c] = if col < hid {
                            wd.data[col * inter + row]
                        } else {
                            0.0
                        };
                    }
                }
            }
            if g >= nkv {
                continue;
            }
            let lo: Vec<usize> = (j * dc / 2..(j + 1) * dc / 2).collect();
            let dl: Vec<usize> = lo
                .iter()
                .copied()
                .chain(lo.iter().map(|&i| d / 2 + i))
                .collect();
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
                        wob[(pe * qh * dc + q) * hc + c] = if col < hid {
                            wo.data[col * qdim + qrow]
                        } else {
                            0.0
                        };
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
            put(dev, sym, arr, origin, px_n, py_n, pad)?;
        }
        let zeros_kv = vec![0f32; pes * (dc * tc + 1)];
        put(dev, "kc", &zeros_kv, origin, px_n, py_n, false)?;
        put(dev, "vc", &zeros_kv, origin, px_n, py_n, false)?;
        for (sym, name) in [("gamma1", "attn_norm"), ("gamma2", "mlp_norm")] {
            let v = get(&format!("{p}{name}"), &[hid])?;
            let col: Vec<f32> = (0..pes)
                .flat_map(|pe| {
                    let py = pe / px_n;
                    (0..hc).map(move |c| py * hc + c)
                })
                .map(|i| v.data.get(i).copied().unwrap_or(0.0))
                .collect();
            put(dev, sym, &col, origin, px_n, py_n, false)?;
        }
        tracing::info!(layer = l, "spatial wafer: serve2d layer placed");
    }
    Ok(())
}

/// One layer grid of the batched build, as qwen2d_paged reads it: per
/// column its kv head's dims `dl`,
/// the two query heads and k split in halves (`wqa`: dims [0, Dc/2), `wqv`:
/// the rest), v, the q/k norm weights (q's carrying 1/sqrt(head_dim)), the
/// RoPE tables cos/sin(p·f) for p = 256a, 16b, c, o, the MLP, the empty pool.
#[allow(clippy::too_many_lines)]
fn place_paged_layer(
    lay: &ProgramSpec,
    ckpt: &Ckpt,
    inv_freq: &[f32],
    l: usize,
    dev: &mut dyn Device,
) -> R<()> {
    let shape = lay.paged().expect("a batched build");
    let (p_n, hc, dc, gs, ic) = (lay.py(), lay.hc(), lay.dc(), lay.gs(), lay.ic());
    let d = lay.head_dim();
    let dh = dc / 2;
    let nkv = p_n / gs;
    let hid = lay.hidden();
    let qdim = nkv * 2 * d;
    let inter = p_n * ic;
    let pes = p_n * p_n;
    let p = format!("layers.{l}.");
    let get = |name: &str, shape: &[usize]| -> R<Tensor> {
        let t = ckpt.get(name)?;
        t.expect_shape(name, shape)?;
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
        let dl: Vec<usize> = lo
            .iter()
            .copied()
            .chain(lo.iter().map(|&i| d / 2 + i))
            .collect();
        let qa: Vec<usize> = dl.iter().map(|&dd| 2 * g * d + dd).collect();
        let qb: Vec<usize> = dl.iter().map(|&dd| (2 * g + 1) * d + dd).collect();
        let kvr: Vec<usize> = dl.iter().map(|&dd| g * d + dd).collect();
        // The three heads' rows (q a, q b, k) and their norm weights.
        let heads: [(&Tensor, &[usize]); 3] = [(&wq, &qa), (&wq, &qb), (&wk, &kvr)];
        let norm = |h: usize, i: usize| -> f32 {
            if h < 2 {
                qn.data[dl[i]] / sqrt_d
            } else {
                kn.data[dl[i]]
            }
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
        put(dev, sym, arr, origin, p_n, p_n, pad)?;
    }
    let pool = vec![0f32; pes * (shape.blocks * dc * shape.page_size + 1)];
    put(dev, "kc", &pool, origin, p_n, p_n, false)?;
    put(dev, "vc", &pool, origin, p_n, p_n, false)?;
    for (sym, name) in [("gamma1", "attn_norm"), ("gamma2", "mlp_norm")] {
        let v = get(&format!("{p}{name}"), &[hid])?;
        let col: Vec<f32> = (0..pes)
            .flat_map(|pe| {
                let py = pe / p_n;
                v.data[py * hc..(py + 1) * hc].iter().copied()
            })
            .collect();
        put(dev, sym, &col, origin, p_n, p_n, false)?;
    }
    tracing::info!(layer = l, "spatial wafer: batched serve2d layer placed");
    Ok(())
}

/// head2d's f32 tiles (one block): region PE (c, ry) holds dims c*Hc.. of
/// vocabulary rows ry*Vr.., dim-major after one leading pad; the guest VM
/// column (if any) after the root column gets zeros; the feeder row holds
/// the final norm's gamma.
pub(crate) fn place_head(lay: &ProgramSpec, ckpt: &Ckpt, dev: &mut dyn Device) -> R<()> {
    let (hh, vr, hc, f) = (lay.hh(), lay.vr(), lay.hc(), lay.feeder());
    let emb = ckpt.get("embed")?;
    if emb.shape.len() != 2 || emb.shape[1] != lay.hidden() {
        return Err(format!("embed is {:?}", emb.shape));
    }
    let hid = lay.hidden();
    let have = emb.shape[0].min(lay.vocab());
    let n = hc * vr + 1;
    let r2x = lay.r2x();
    // Fabric column q of the block -> chunk column (None: the VM column).
    let vmc = lay.vmc();
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
                dev.put(
                    "wt",
                    rect(hx + qs.start, y0 + (y - lo), qs.len(), rows),
                    n,
                    &words(&t),
                )?;
                y += rows;
            }
        }
    }
    let norm = ckpt.get("final_norm")?;
    norm.expect_shape("final_norm", &[hid])?;
    for qs in &segments {
        let gam: Vec<f32> = qs
            .clone()
            .flat_map(|q| {
                let c = chunk(q);
                (0..hc).map(move |h| c.map(|c| c * hc + h))
            })
            .map(|dim| dim.and_then(|d| norm.data.get(d).copied()).unwrap_or(0.0))
            .collect();
        dev.put("gam", rect(hx + qs.start, f, qs.len(), 1), hc, &words(&gam))?;
    }
    Ok(())
}

/// One launch: the forced tokens, `steps` head iterations from `start`;
/// the argmax after each step, and the last step's logits when asked.
pub fn run(
    lay: &ProgramSpec,
    rt: &mut dyn Device,
    forced: &[u32],
    steps: usize,
    start: usize,
    want_logits: bool,
) -> R<(Vec<u32>, Option<Vec<f32>>)> {
    if forced.is_empty() || steps + 1 > lay.cap() || start + steps > lay.positions() {
        return Err(format!(
            "a launch of {steps} steps at position {start} does not fit the program \
             ({} iterations, {} positions; a launch forces its first token)",
            lay.cap(),
            lay.positions()
        ));
    }
    let e = |e: crate::sdk::Error| e.to_string();
    let (rx, ry) = lay.root();
    let one = rect(rx, ry, 1, 1);
    let mut inn = vec![SAMPLE; lay.cap()];
    inn[..forced.len()].copy_from_slice(forced);
    rt.memcpy_h2d_nonblock(
        SYM.in_tokens,
        inn,
        one,
        lay.cap(),
        Order::RowMajor,
        DataType::Bits32,
    )
    .map_err(e)?;
    rt.launch_nonblock(SYM.compute, &[steps as u32, start as u32])
        .map_err(e)?;
    let mut out = vec![0u32; lay.cap()];
    let read = rt.memcpy_d2h(
        SYM.out_tokens,
        &mut out,
        one,
        lay.cap(),
        Order::RowMajor,
        DataType::Bits32,
    );
    rt.wait_pending();
    read.map_err(e)?;
    let y = if want_logits {
        if !rt.has_symbol(SYM.yscale) {
            return Err(
                "this serve2d build exports no `yscale` (the final norm's 1/rms): its `y` \
                 is not the logits; it serves greedy (token-loop) epilogues only"
                    .into(),
            );
        }
        let mut scale = [0u32; 1];
        rt.memcpy_d2h(
            SYM.yscale,
            &mut scale,
            rect(rx, 0, 1, 1),
            1,
            Order::RowMajor,
            DataType::Bits32,
        )
        .map_err(e)?;
        let scale = f32::from_bits(scale[0]);
        let (f, hh, vr) = (lay.feeder(), lay.hh(), lay.vr());
        let mut a = vec![0u32; f * vr];
        rt.memcpy_d2h(
            SYM.logits,
            &mut a,
            rect(rx, 0, 1, f),
            vr,
            Order::RowMajor,
            DataType::Bits32,
        )
        .map_err(e)?;
        let mut b = vec![0u32; (hh - f) * vr];
        rt.memcpy_d2h(
            SYM.logits,
            &mut b,
            rect(rx, f + 1, 1, hh - f),
            vr,
            Order::RowMajor,
            DataType::Bits32,
        )
        .map_err(e)?;
        Some(
            a.into_iter()
                .chain(b)
                .take(lay.vocab())
                .map(|w| f32::from_bits(w) * scale)
                .collect(),
        )
    } else {
        None
    };
    Ok((out[..steps].to_vec(), y))
}

/// A batched launch's answer: each lane's argmax a step, and the logits
/// rows (when asked for).
pub type BatchOut = (Vec<Vec<u32>>, Vec<Vec<f32>>);

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
/// [`ProgramSpec::launch_lanes`]; on a pipelined build each B of them is a
/// micro-batch, and a short batch repeats its first lane, the repeats'
/// answers dropped): every lane's argmax after each pass, and with
/// `want_logits` the last pass's logits rows (`hlogits` x `yscale`) of the
/// first [`ProgramSpec::logit_lanes`] lanes (the head keeps no more).
pub fn fire(
    lay: &ProgramSpec,
    rt: &mut dyn Device,
    lanes: &[BatchLane],
    steps: usize,
    want_logits: bool,
    vm_written: &mut Vec<u32>,
) -> R<BatchOut> {
    fire_lanes(lay, rt, lanes, steps, want_logits, None, vm_written)
}

/// A VM launch's control a virtual lane (head lane m·B + b): whether the VM
/// samples it (`gmode_l`) and whose turn it is (vmlane word 3: 0 set A, 1
/// set B). A set counts its own turns, so each lap must give it exactly its
/// lanes.
pub struct VmTurns<'a> {
    pub sampled: &'a [u32],
    pub sets: &'a [u32],
}

/// [`fire`] on a VM build with `vm`'s turns (`None`: the head's argmax every
/// lane).
pub fn fire_lanes(
    lay: &ProgramSpec,
    rt: &mut dyn Device,
    lanes: &[BatchLane],
    steps: usize,
    want_logits: bool,
    vm: Option<&VmTurns>,
    vm_written: &mut Vec<u32>,
) -> R<BatchOut> {
    let shape = lay
        .paged()
        .ok_or("a batched launch on a one-sequence build")?;
    let b_n = shape.lanes;
    let most = lay.launch_lanes();
    if lanes.is_empty() || lanes.len() > most || steps == 0 || steps + 1 > lay.cap() {
        return Err(format!(
            "{} lane(s) of {steps} pass(es) for a build of {most} lanes and {} iterations",
            lanes.len(),
            lay.cap()
        ));
    }
    if let Some(lane) = lanes
        .iter()
        .find(|l| l.forced.is_empty() || l.forced.len() > lay.cap())
    {
        return Err(format!("a lane forces {} token(s)", lane.forced.len()));
    }
    // A table for pool rows `rows` (serve8b_region's units differ; a page id
    // is still one of the pool's `shape.pages()`).
    let table_rows = |batch: &[BatchLane], rows: usize| -> R<Vec<u32>> {
        if let Some(g) = batch
            .iter()
            .flat_map(|l| &l.pages)
            .find(|&&g| g as usize >= shape.pages())
        {
            return Err(format!(
                "a lane names page {g} of a pool of {}",
                shape.pages()
            ));
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
        super::paged::table(
            &super::paged::Shape { rows, ..shape },
            &table_lanes,
            steps as u32,
        )
    };
    // Micro-batches (one on a build without a ring).
    let mbs: Vec<&[BatchLane]> = lanes.chunks(b_n).collect();
    let slots = lay.mmax().unwrap_or(1);
    let e = |e: crate::sdk::Error| e.to_string();
    lay.stage.write_tables(lay, rt, &mbs, &table_rows)?;
    lay.sampler.write_turns(lay, rt, &mbs, vm, vm_written)?;
    let mut inn = vec![SAMPLE; slots * b_n * lay.cap()];
    for (m, mb) in mbs.iter().enumerate() {
        for b in 0..b_n {
            let lane = mb.get(b).unwrap_or(&mb[0]);
            let at = (m * b_n + b) * lay.cap();
            inn[at..at + lane.forced.len()].copy_from_slice(&lane.forced);
        }
    }
    let (rx, ry) = lay.root();
    let one = rect(rx, ry, 1, 1);
    let words = inn.len();
    rt.memcpy_h2d_nonblock(
        SYM.in_tokens,
        inn,
        one,
        words,
        Order::RowMajor,
        DataType::Bits32,
    )
    .map_err(e)?;
    // DEFER: every lane of the launch on the VM, two micro-batches or more.
    let all_vm = vm.is_some_and(|t| (0..mbs.len() * b_n).all(|v| t.sampled.get(v) == Some(&1)));
    let args = lay.compute_args(steps, mbs.len(), all_vm);
    rt.launch_nonblock(SYM.compute, &args).map_err(e)?;
    let mut out = vec![0u32; words];
    let read = rt.memcpy_d2h(
        SYM.out_tokens,
        &mut out,
        one,
        words,
        Order::RowMajor,
        DataType::Bits32,
    );
    rt.wait_pending();
    read.map_err(e)?;
    lay.stage.check_launch(lay, rt)?;
    // Lane i is micro-batch i / B's lane i % B: [i][CAP] either way.
    let tokens: Vec<Vec<u32>> = (0..lanes.len())
        .map(|i| out[i * lay.cap()..i * lay.cap() + steps].to_vec())
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
pub fn head_runs(lay: &ProgramSpec) -> Vec<std::ops::Range<usize>> {
    let cw = lay.chunk_cols() + lay.vmc();
    // The VM columns (or, VM 0, their stand-ins) hold head symbols as
    // one-word dummies: head writes go either side of them.
    #[allow(
        clippy::single_range_in_vec_init,
        reason = "runs of columns, not the column numbers"
    )]
    let runs = if lay.vmc() > 0 {
        vec![0..lay.r2x() + 1, lay.r2x() + 1 + lay.vmc()..cw]
    } else {
        vec![0..cw]
    };
    runs.into_iter().filter(|r| !r.is_empty()).collect()
}

/// Refuses a write of head-only `symbol` whose rectangle (head columns
/// `cols`) covers a VM column: the bug class that overran the VM PEs (head
/// weights, `mtab`) on the simulator.
pub fn check_head_columns(lay: &ProgramSpec, symbol: &str, cols: std::ops::Range<usize>) -> R<()> {
    if lay.vmc() > 0 {
        let vm = lay.r2x() + 1..lay.r2x() + 1 + lay.vmc();
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
pub(crate) fn head_h2d(
    lay: &ProgramSpec,
    rt: &mut dyn Device,
    symbol: &str,
    data: &[u32],
    y0: usize,
    h: usize,
    n: usize,
) -> R<()> {
    let cw = lay.chunk_cols() + lay.vmc();
    for cols in head_runs(lay) {
        check_head_columns(lay, symbol, cols.clone())?;
        let words: Vec<u32> = (0..h)
            .flat_map(|row| {
                data[(row * cw + cols.start) * n..(row * cw + cols.end) * n]
                    .iter()
                    .copied()
            })
            .collect();
        rt.memcpy_h2d(
            symbol,
            &words,
            rect(lay.head_x() + cols.start, y0, cols.len(), h),
            n,
            Order::RowMajor,
            DataType::Bits32,
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// The ring's `mtab` over the head's feeder row: feeder of chunk j (column
/// j, the VM column after the root zero) holds per micro-batch slot m
/// row j of micro-batch m's table (`tables[m]`, unused slots zero).
pub(crate) fn pipe_tables(lay: &ProgramSpec, tables: &[Vec<u32>]) -> Vec<u32> {
    let w = lay.paged().map_or(0, |s| s.row_words());
    let slots = lay.mmax().unwrap_or(1);
    let mut mtab = Vec::with_capacity((lay.py() + lay.vmc()) * slots * w);
    for c in 0..lay.py() {
        for m in 0..slots {
            match tables.get(m) {
                Some(t) => mtab.extend_from_slice(&t[c * w..(c + 1) * w]),
                None => mtab.extend(std::iter::repeat_n(0, w)),
            }
        }
        if c == lay.r2x() {
            mtab.extend(std::iter::repeat_n(0, lay.vmc() * slots * w));
        }
    }
    mtab
}

/// The last pass's logits rows of lanes `0..lanes` (at most HLB): the root
/// column's `hlogits` (region row r, lane b at words `(r * HLB + b) * Vr`,
/// rows in vocab order) x `yscale[b]`.
pub fn logits_rows(lay: &ProgramSpec, rt: &mut dyn Device, lanes: usize) -> R<Vec<Vec<f32>>> {
    let e = |e: crate::sdk::Error| e.to_string();
    let (rx, _) = lay.root();
    let (f, hh, vr, hlb) = (lay.feeder(), lay.hh(), lay.vr(), lay.logit_lanes());
    let mut a = vec![0u32; f * hlb * vr];
    rt.memcpy_d2h(
        SYM.logit_lanes,
        &mut a,
        rect(rx, 0, 1, f),
        hlb * vr,
        Order::RowMajor,
        DataType::Bits32,
    )
    .map_err(e)?;
    let mut c = vec![0u32; (hh - f) * hlb * vr];
    rt.memcpy_d2h(
        SYM.logit_lanes,
        &mut c,
        rect(rx, f + 1, 1, hh - f),
        hlb * vr,
        Order::RowMajor,
        DataType::Bits32,
    )
    .map_err(e)?;
    let mut scale = vec![0u32; hlb];
    rt.memcpy_d2h(
        SYM.yscale,
        &mut scale,
        rect(rx, 0, 1, 1),
        hlb,
        Order::RowMajor,
        DataType::Bits32,
    )
    .map_err(e)?;
    let rows: Vec<u32> = a.into_iter().chain(c).collect();
    Ok((0..lanes.min(hlb))
        .map(|b| {
            let ys = f32::from_bits(scale[b]);
            (0..hh)
                .flat_map(|r| rows[(r * hlb + b) * vr..(r * hlb + b + 1) * vr].iter())
                .take(lay.vocab())
                .map(|&w| f32::from_bits(w) * ys)
                .collect()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simulator serve2d_pipe build: 8 feeders, the VM column
    /// after the root (column 6), MMAX 3 slots of B 2 lanes.
    fn pipe() -> ProgramSpec {
        let params = [
            ("PX", 8),
            ("PY", 8),
            ("Hc", 16),
            ("HR", 128),
            ("Dc", 8),
            ("Gs", 4),
            ("Qh", 2),
            ("Ic", 24),
            ("Tc", 1),
            ("W16", 0),
            ("KV16", 0),
            ("WF32", 1),
            ("GRID", 1),
            ("HEADMODE", 2),
            ("B", 2),
            ("PS", 4),
            ("NB", 4),
            ("E", 2),
            ("MMAX", 3),
            ("GW", 2),
            ("GH", 4),
            ("HH", 32),
            ("Vr", 32),
            ("Gx", 4),
            ("VMC", 1),
            ("vocab", 1024),
            ("CAP", 4),
        ];
        ProgramSpec::of_pairs(&params).expect("a layout")
    }

    #[test]
    fn hlbn_caps_the_lanes_a_launch_samples_on_the_host() {
        assert_eq!(pipe().logit_lanes(), 2);
        let mut lay = pipe();
        lay.head.hlbn = Some(1);
        assert_eq!(lay.logit_lanes(), 1);
        lay.head.hlbn = Some(8);
        assert_eq!(lay.logit_lanes(), 2);
    }

    #[test]
    fn head_writes_stay_off_stand_in_vm_columns_with_the_vm_off() {
        // VMC 1, VM 0: the VM column is a stand-in, and head symbols there
        // are still one-word dummies (an n-word write there gave wrong
        // tokens on hardware).
        let params = [
            ("PX", 4),
            ("PY", 4),
            ("Hc", 16),
            ("HR", 64),
            ("Dc", 8),
            ("Gs", 4),
            ("Qh", 2),
            ("Ic", 16),
            ("Tc", 1),
            ("W16", 0),
            ("KV16", 0),
            ("WF32", 1),
            ("GRID", 1),
            ("HEADMODE", 2),
            ("B", 2),
            ("PS", 16),
            ("NB", 2),
            ("E", 2),
            ("MMAX", 2),
            ("GW", 2),
            ("GH", 4),
            ("HH", 16),
            ("Vr", 32),
            ("Gx", 4),
            ("VMC", 1),
            ("vocab", 250),
            ("CAP", 8),
        ];
        let lay = ProgramSpec::of_pairs(&params).unwrap();
        assert!(lay.vm().is_none());
        assert!(
            check_head_columns(&lay, "mtab", 0..lay.py() + lay.vmc())
                .unwrap_err()
                .contains("VM columns")
        );
        assert_eq!(head_runs(&lay), vec![0..3, 4..5]);
    }

    #[test]
    fn head_writes_stay_off_the_vm_columns() {
        let params = [
            ("PX", 4),
            ("PY", 4),
            ("Hc", 16),
            ("HR", 64),
            ("Dc", 8),
            ("Gs", 4),
            ("Qh", 2),
            ("Ic", 16),
            ("Tc", 1),
            ("W16", 0),
            ("KV16", 0),
            ("WF32", 1),
            ("GRID", 1),
            ("HEADMODE", 2),
            ("B", 2),
            ("PS", 16),
            ("NB", 2),
            ("E", 2),
            ("MMAX", 2),
            ("GW", 2),
            ("GH", 4),
            ("HH", 16),
            ("Vr", 32),
            ("Gx", 4),
            ("VMC", 1),
            ("vocab", 250),
            ("CAP", 8),
            ("VM", 1),
            ("max_ops", 48),
            ("nreg", 6),
            ("pool_words", 64),
            ("feat", 99),
        ];
        let lay = ProgramSpec::of_pairs(&params).unwrap();
        assert!(lay.vm().is_some());
        // The old `mtab` rectangle, every head column: it covers the VM column 3.
        assert!(
            check_head_columns(&lay, "mtab", 0..lay.py() + lay.vmc())
                .unwrap_err()
                .contains("VM columns")
        );
        // The chunk columns either side of it.
        assert_eq!(head_runs(&lay), vec![0..3, 4..5]);
        assert!(
            head_runs(&lay)
                .into_iter()
                .all(|r| check_head_columns(&lay, "mtab", r).is_ok())
        );
    }

    #[test]
    fn concurrent_sets_and_deferred_tokens_read_the_layout_s_names() {
        // A two-set build's out.json params: VMCONC and DEFER as the layout
        // names them (the VM image's own `vm_conc` is not the layout's).
        let base = [
            ("PX", 4),
            ("PY", 4),
            ("Hc", 16),
            ("HR", 64),
            ("Dc", 8),
            ("Gs", 4),
            ("Qh", 2),
            ("Ic", 16),
            ("Tc", 1),
            ("W16", 0),
            ("KV16", 0),
            ("WF32", 1),
            ("GRID", 1),
            ("HEADMODE", 2),
            ("B", 2),
            ("PS", 16),
            ("NB", 2),
            ("E", 2),
            ("MMAX", 2),
            ("GW", 2),
            ("GH", 4),
            ("HH", 16),
            ("Vr", 32),
            ("Gx", 4),
            ("VMC", 1),
            ("vocab", 250),
            ("CAP", 8),
            ("VM", 1),
            ("max_ops", 48),
            ("nreg", 6),
            ("pool_words", 64),
            ("feat", 99),
            ("VMB", 1),
            ("b_max_ops", 64),
            ("b_nreg", 6),
            ("b_pool_words", 64),
        ];
        let lay = |extra: &[(&str, usize)]| {
            let params: Vec<(&str, usize)> = base.iter().chain(extra).copied().collect();
            ProgramSpec::of_pairs(&params).unwrap()
        };
        let on = lay(&[("VMCONC", 1), ("DEFER", 1)]);
        assert!(on.vm_b().is_some() && on.vm_conc() && on.vm_defer());
        let off = lay(&[("vm_conc", 1)]);
        assert!(!off.vm_conc() && !off.vm_defer());
    }

    #[test]
    fn ring_lanes_and_feeder_tables() {
        let lay = pipe();
        assert_eq!(
            (lay.launch_lanes(), lay.logit_lanes(), lay.r2x()),
            (6, 2, 6)
        );
        let w = lay.paged().unwrap().row_words();
        // Micro-batch m's table: word i of row r is 1000 m + 100 r + i.
        let tables: Vec<Vec<u32>> = (0..2)
            .map(|m| {
                (0..8 * w)
                    .map(|k| (1000 * m + 100 * (k / w) + k % w) as u32)
                    .collect()
            })
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
