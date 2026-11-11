//! The serve_q35 wafer program's host side (`spatial/csl/serve_q35`):
//! serve2d's program around Qwen3.5's two layer units, layer `l`'s grid an
//! attention unit where bit `l` of `KINDS` is set, else a Gated DeltaNet
//! ([`super::program::Hybrid`]). Everything but the layers is serve2d's:
//! the head (tied, its norm `1 + w`), the paged pool (on the attention
//! grids only), the launch.
//!
//! Both units split the hidden state as serve2d's grids (row `py` holds
//! hidden chunk `py`, `Hc` dims) and the MLP likewise (`Ic` rows a column).
//! A Gated DeltaNet column `c` of k head `h = c / GK` (slice `s = c % GK`)
//! holds `DK` dims of the head's q and k, `DV` of each of its `VH` v heads'
//! v and z, and each v head's b and a; PE (`py`, `c`) holds row `k = py` of
//! each v head's `k_dim x v_dim` state over column `c`'s v dims. An
//! attention column holds `Dc` dims of a kv head (`Gs` columns a head), its
//! `Qh` query heads' and their gates', RoPE's turned pairs first (a pair's
//! two dims in one column), the unturned dims after them with a turn of 0.
//!
//! The norms scale by `1 + w` (the host adds the one); q's carries
//! 1/sqrt(head_dim).

use super::ckpt::{Ckpt, Tensor};
use super::program::{Hybrid, ProgramSpec};
use super::wafer::rect;
use crate::spatial::device::Device;

type R<T> = std::result::Result<T, String>;

/// The symbols a tile of which has one leading pad word a PE (its bank
/// offset), as qwen2d's.
const PADDED: &[&str] = &[
    "wp", "wo", "wgu", "wd", "wqa", "wqv", "wv", "wg", "kc", "vc",
];

/// `data` (row-major by PE over a `p` x `p` grid at `at`) to `symbol`.
fn put(dev: &mut dyn Device, symbol: &str, data: &[f32], at: (usize, usize), p: usize) -> R<()> {
    let n = data.len() / (p * p);
    let pad = PADDED.contains(&symbol);
    let words: Vec<u32> = if pad {
        data.chunks_exact(n)
            .flat_map(|pe| std::iter::once(0).chain(pe.iter().map(|x| x.to_bits())))
            .collect()
    } else {
        data.iter().map(|x| x.to_bits()).collect()
    };
    dev.put(symbol, rect(at.0, at.1, p, p), n + usize::from(pad), &words)
}

/// The grids' dims as the placement reads them.
struct Dims {
    p: usize,
    hc: usize,
    ic: usize,
    hid: usize,
    inter: usize,
}

fn tensor(ckpt: &Ckpt, name: &str, shape: &[usize]) -> R<Tensor> {
    let t = ckpt.get(name)?;
    t.expect_shape(name, shape)?;
    Ok(t)
}

/// Every layer grid of `ckpt` for `lay`: a Gated DeltaNet's or an
/// attention unit's tiles by the layer's kind, each with its MLP; the
/// attention grids' pools empty.
pub(crate) fn place_layers(
    lay: &ProgramSpec,
    ckpt: &Ckpt,
    rope_theta: f64,
    dev: &mut dyn Device,
) -> R<()> {
    let h = lay
        .hybrid()
        .ok_or("serve_q35's placement on a program without its layer mix")?;
    let p = lay.py();
    let d = Dims {
        p,
        hc: lay.hc(),
        ic: lay.ic(),
        hid: lay.hidden(),
        inter: p * lay.ic(),
    };
    for l in 0..lay.layers() {
        let mut tiles: Vec<(&'static str, Vec<f32>)> = if h.attends(l) {
            attn_tiles(lay, &h, &d, ckpt, l, rope_theta)?
        } else {
            gdn_tiles(&h, &d, ckpt, l)?
        };
        tiles.extend(mlp_tiles(&d, ckpt, l)?);
        let origin = lay.origin(l);
        for (sym, data) in &tiles {
            put(dev, sym, data, origin, p)?;
        }
        tracing::info!(
            layer = l,
            attends = h.attends(l),
            "spatial wafer: serve_q35 layer placed"
        );
    }
    Ok(())
}

/// The MLP's tiles and the two norms' (`1 + w`): `wgu` [Hc][gate Ic, up
/// Ic], `wd` [Ic][Hc], `gamma1`/`gamma2` [Hc].
fn mlp_tiles(d: &Dims, ckpt: &Ckpt, l: usize) -> R<Vec<(&'static str, Vec<f32>)>> {
    let (p, hc, ic, hid, inter) = (d.p, d.hc, d.ic, d.hid, d.inter);
    let n = |s: &str| format!("layers.{l}.{s}");
    let wg = tensor(ckpt, &n("mlp.gate"), &[inter, hid])?;
    let wu = tensor(ckpt, &n("mlp.up"), &[inter, hid])?;
    let wd = tensor(ckpt, &n("mlp.down"), &[hid, inter])?;
    let pes = p * p;
    let mut wgu = vec![0f32; pes * hc * 2 * ic];
    let mut wdb = vec![0f32; pes * ic * hc];
    for py in 0..p {
        for c in 0..p {
            let pe = py * p + c;
            for j in 0..hc {
                let col = py * hc + j;
                for r in 0..ic {
                    let row = c * ic + r;
                    wgu[(pe * hc + j) * 2 * ic + r] = wg.data[row * hid + col];
                    wgu[(pe * hc + j) * 2 * ic + ic + r] = wu.data[row * hid + col];
                    wdb[(pe * ic + r) * hc + j] = wd.data[col * inter + row];
                }
            }
        }
    }
    let mut out = vec![("wgu", wgu), ("wd", wdb)];
    for (sym, name) in [("gamma1", "mixer_norm"), ("gamma2", "mlp_norm")] {
        let w = tensor(ckpt, &n(name), &[hid])?;
        let tile: Vec<f32> = (0..pes)
            .flat_map(|pe| {
                let py = pe / p;
                (0..hc).map(move |j| py * hc + j)
            })
            .map(|i| 1.0 + w.data[i])
            .collect();
        out.push((sym, tile));
    }
    Ok(out)
}

/// A Gated DeltaNet layer's tiles: `wp` [Hc][q DK, k DK, v VH·DV, z VH·DV,
/// b VH, a VH], the conv `cw` [CW][q, k, v], `gab` [-exp(A_log) VH,
/// dt_bias VH], the gated norm's `nw` [VH·DV] and `wo` [VH·DV][Hc].
fn gdn_tiles(h: &Hybrid, d: &Dims, ckpt: &Ckpt, l: usize) -> R<Vec<(&'static str, Vec<f32>)>> {
    let (p, hc, hid) = (d.p, d.hc, d.hid);
    let (gk, dk, vh, dv, cw) = (h.gk, h.dk, h.vh, h.dv, h.cw);
    // k_dim is the grid's height; a k head has GK columns.
    let (kd, vd) = (gk * dk, gk * dv);
    if kd != p {
        return Err(format!(
            "serve_q35: k_dim {kd} is not the grid's height {p}"
        ));
    }
    let k_heads = p / gk;
    let v_heads = k_heads * vh;
    let key = k_heads * kd;
    let vw = v_heads * vd;
    let qkv = 2 * key + vw;
    let n = |s: &str| format!("layers.{l}.gdn.{s}");
    let in_qkv = tensor(ckpt, &n("in_qkv"), &[qkv, hid])?;
    let in_z = tensor(ckpt, &n("in_z"), &[vw, hid])?;
    let in_b = tensor(ckpt, &n("in_b"), &[v_heads, hid])?;
    let in_a = tensor(ckpt, &n("in_a"), &[v_heads, hid])?;
    let conv = ckpt.get(&n("conv"))?;
    if conv.data.len() != qkv * cw {
        return Err(format!("{}: {:?}, not {qkv} x {cw}", n("conv"), conv.shape));
    }
    let a_log = tensor(ckpt, &n("a_log"), &[v_heads])?;
    let dt_bias = tensor(ckpt, &n("dt_bias"), &[v_heads])?;
    let norm = tensor(ckpt, &n("norm"), &[vd])?;
    let out_proj = tensor(ckpt, &n("out_proj"), &[hid, vw])?;
    let nv = vh * dv;
    let nch = 2 * dk + nv;
    let npj = nch + nv + 2 * vh;
    let pes = p * p;
    let mut wp = vec![0f32; pes * hc * npj];
    let mut cwt = vec![0f32; pes * cw * nch];
    let mut gab = vec![0f32; pes * 2 * vh];
    let mut nw = vec![0f32; pes * nv];
    let mut wo = vec![0f32; pes * nv * hc];
    for c in 0..p {
        let (head, s) = (c / gk, c % gk);
        let qr: Vec<usize> = (0..dk).map(|i| head * kd + s * dk + i).collect();
        let kr: Vec<usize> = qr.iter().map(|r| key + r).collect();
        let jvs: Vec<usize> = (0..vh).map(|t| head * vh + t).collect();
        // A v head's dims in this column: its value dims, z's, out_proj's.
        let vdims: Vec<usize> = jvs
            .iter()
            .flat_map(|&jv| (0..dv).map(move |i| jv * vd + s * dv + i))
            .collect();
        let vr: Vec<usize> = vdims.iter().map(|r| 2 * key + r).collect();
        let conv_rows: Vec<usize> = qr.iter().chain(&kr).chain(&vr).copied().collect();
        // The projection's rows: (tensor, row).
        let rows: Vec<(&Tensor, usize)> = conv_rows
            .iter()
            .map(|&r| (&in_qkv, r))
            .chain(vdims.iter().map(|&r| (&in_z, r)))
            .chain(jvs.iter().map(|&r| (&in_b, r)))
            .chain(jvs.iter().map(|&r| (&in_a, r)))
            .collect();
        for py in 0..p {
            let pe = py * p + c;
            for j in 0..hc {
                let col = py * hc + j;
                for (k, (t, r)) in rows.iter().enumerate() {
                    wp[(pe * hc + j) * npj + k] = t.data[r * hid + col];
                }
                for (i, &r) in vdims.iter().enumerate() {
                    wo[(pe * nv + i) * hc + j] = out_proj.data[col * vw + r];
                }
            }
            for t in 0..cw {
                for (ch, &r) in conv_rows.iter().enumerate() {
                    cwt[(pe * cw + t) * nch + ch] = conv.data[r * cw + t];
                }
            }
            for (t, &jv) in jvs.iter().enumerate() {
                gab[pe * 2 * vh + t] = -(f64::from(a_log.data[jv]).exp() as f32);
                gab[pe * 2 * vh + vh + t] = dt_bias.data[jv];
            }
            for t in 0..vh {
                for i in 0..dv {
                    nw[pe * nv + t * dv + i] = norm.data[s * dv + i];
                }
            }
        }
    }
    Ok(vec![
        ("wp", wp),
        ("cw", cwt),
        ("gab", gab),
        ("nw", nw),
        ("wo", wo),
    ])
}

/// A gated-attention layer's tiles, as qwen35_attn reads them: per column
/// its kv head's dims `dl` (`Dc / 2` turned pairs, or unturned pairs), the
/// `Qh` query heads and k split in halves (`wqa`: the pairs' first dims,
/// `wqv`: their second), v, the query heads' gates `wg`, the q/k norm
/// weights (`1 + w`, q's carrying 1/sqrt(head_dim)), RoPE's tables
/// cos/sin(p·f) for p = 256a, 16b, c, o, and the empty pool.
#[allow(clippy::too_many_lines)]
fn attn_tiles(
    lay: &ProgramSpec,
    h: &Hybrid,
    d: &Dims,
    ckpt: &Ckpt,
    l: usize,
    theta: f64,
) -> R<Vec<(&'static str, Vec<f32>)>> {
    let shape = lay.paged().ok_or("serve_q35 is a paged build")?;
    let (p, hc, hid) = (d.p, d.hc, d.hid);
    let (dc, gs, qh) = (lay.dc(), lay.gs(), lay.qh());
    let hd = gs * dc;
    let dh = dc / 2;
    let kh = p / gs;
    let heads = kh * qh;
    let rot = h.rot;
    if rot > hd || !rot.is_multiple_of(2) || !(hd - rot).is_multiple_of(2) {
        return Err(format!("serve_q35: RoPE turns {rot} of {hd} dims"));
    }
    let n = |s: &str| format!("layers.{l}.attn.{s}");
    let qg = tensor(ckpt, &n("qg_proj"), &[2 * heads * hd, hid])?;
    let wk = tensor(ckpt, &n("k_proj"), &[kh * hd, hid])?;
    let wv = tensor(ckpt, &n("v_proj"), &[kh * hd, hid])?;
    let wo = tensor(ckpt, &n("o_proj"), &[hid, heads * hd])?;
    let qn = tensor(ckpt, &n("q_norm"), &[hd])?;
    let kn = tensor(ckpt, &n("k_norm"), &[hd])?;
    let sqrt_d = (hd as f64).sqrt() as f32;
    // A head's dim pairs: RoPE's (i, i + rot/2) at frequency i, then the
    // unturned dims in pairs.
    let pairs: Vec<(usize, usize, f64)> = (0..rot / 2)
        .map(|i| {
            let f = 1.0 / theta.powf(2.0 * i as f64 / rot as f64);
            (i, i + rot / 2, f)
        })
        .chain((0..(hd - rot) / 2).map(|i| (rot + 2 * i, rot + 2 * i + 1, 0.0)))
        .collect();
    // q's row r of head hq and its gate's in qg (each head: hd q rows, then
    // hd gate rows).
    let q_row = |hq: usize, dd: usize| hq * 2 * hd + dd;
    let g_row = |hq: usize, dd: usize| hq * 2 * hd + hd + dd;
    let nh1 = qh + 1;
    let pes = p * p;
    let mut wqa = vec![0f32; pes * hc * nh1 * dh];
    let mut wqv = vec![0f32; pes * hc * nh1 * dh];
    let mut wvb = vec![0f32; pes * hc * dc];
    let mut wg = vec![0f32; pes * hc * qh * dc];
    let mut qkn = vec![0f32; pes * 2 * nh1 * dh];
    let mut rot_c = vec![0f32; pes * 3 * 16 * dh];
    let mut rot_s = vec![0f32; pes * 3 * 16 * dh];
    let mut wob = vec![0f32; pes * qh * dc * hc];
    for c in 0..p {
        let (g, j) = (c / gs, c % gs);
        let prs = &pairs[j * dh..(j + 1) * dh];
        let dl: Vec<usize> = prs
            .iter()
            .map(|t| t.0)
            .chain(prs.iter().map(|t| t.1))
            .collect();
        // The rows each of the column's heads reads: Qh query heads, then k.
        let head_rows = |hi: usize| -> Vec<(&Tensor, usize)> {
            if hi < qh {
                dl.iter().map(|&dd| (&qg, q_row(qh * g + hi, dd))).collect()
            } else {
                dl.iter().map(|&dd| (&wk, g * hd + dd)).collect()
            }
        };
        let rows: Vec<Vec<(&Tensor, usize)>> = (0..nh1).map(head_rows).collect();
        let norm = |hi: usize, i: usize| -> f32 {
            if hi < qh {
                (1.0 + qn.data[dl[i]]) / sqrt_d
            } else {
                1.0 + kn.data[dl[i]]
            }
        };
        // o_proj's columns of the column's query heads.
        let o_cols: Vec<usize> = (0..qh)
            .flat_map(|hi| dl.iter().map(move |&dd| (qh * g + hi) * hd + dd))
            .collect();
        for py in 0..p {
            let pe = py * p + c;
            for jj in 0..hc {
                let col = py * hc + jj;
                for (hi, hr) in rows.iter().enumerate() {
                    for k in 0..dh {
                        let (t, r) = hr[k];
                        wqa[((pe * hc + jj) * nh1 + hi) * dh + k] = t.data[r * hid + col];
                        let (t, r) = hr[dh + k];
                        wqv[((pe * hc + jj) * nh1 + hi) * dh + k] = t.data[r * hid + col];
                    }
                }
                for (k, &dd) in dl.iter().enumerate() {
                    wvb[(pe * hc + jj) * dc + k] = wv.data[(g * hd + dd) * hid + col];
                    for hi in 0..qh {
                        wg[((pe * hc + jj) * qh + hi) * dc + k] =
                            qg.data[g_row(qh * g + hi, dd) * hid + col];
                    }
                }
                for (i, &oc) in o_cols.iter().enumerate() {
                    wob[(pe * qh * dc + i) * hc + jj] = wo.data[col * heads * hd + oc];
                }
            }
            for hi in 0..nh1 {
                for k in 0..dh {
                    qkn[((pe * 2) * nh1 + hi) * dh + k] = norm(hi, k);
                    qkn[((pe * 2 + 1) * nh1 + hi) * dh + k] = norm(hi, dh + k);
                }
            }
            for (lv, unit) in [256.0f64, 16.0, 1.0].into_iter().enumerate() {
                for a in 0..16 {
                    for (k, t) in prs.iter().enumerate() {
                        let ang = a as f64 * unit * t.2;
                        rot_c[((pe * 3 + lv) * 16 + a) * dh + k] = ang.cos() as f32;
                        rot_s[((pe * 3 + lv) * 16 + a) * dh + k] = ang.sin() as f32;
                    }
                }
            }
        }
    }
    let pool = vec![0f32; pes * shape.blocks * dc * shape.page_size];
    Ok(vec![
        ("wqa", wqa),
        ("wqv", wqv),
        ("wv", wvb),
        ("wg", wg),
        ("qkn_w", qkn),
        ("rot_c", rot_c),
        ("rot_s", rot_s),
        ("wo", wob),
        ("kc", pool.clone()),
        ("vc", pool),
    ])
}

#[cfg(test)]
mod tests {
    use crate::spatial::codegen::Design;
    use crate::spatial::device::Recorder;
    use crate::spatial::program::ProgramSpec;
    use crate::spatial::serve2d::{BatchLane, StateSeat};
    use crate::spatial::wafer::{WaferModel, Weights};

    /// The 0.8B plan's program on a recorder.
    fn model() -> (WaferModel, Recorder) {
        let c = poem_compiler::catalog::repository();
        let t = c
            .parse("qwen35-d0.8b-bf16-kv-bf16")
            .expect("qwen35")
            .trace(poem_ir::Platform::Cerebras);
        let s = crate::spatial::model::ModelSpec::read(&t).expect("reads");
        let p = crate::spatial::plan_q35::Q35Plan::derive(
            &s,
            crate::spatial::plan::WaferDims::CS3,
            crate::spatial::plan::Goals::lanes(1),
        )
        .expect("a plan");
        let params: Vec<(String, String)> = p
            .params()
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let spec = ProgramSpec::from_params(Design::ServeQ35, &params).expect("a program");
        let rec = Recorder::default();
        let m = WaferModel::on_device(Box::new(rec.clone()), spec, Weights::Skip, None, 0)
            .expect("the load");
        rec.take();
        (m, rec)
    }

    fn lane(key: u32, fresh: bool, slot: u32, forced: Vec<u32>) -> BatchLane {
        BatchLane {
            pos: slot,
            slot,
            pages: vec![0, 1, 2, 3],
            visible: None,
            forced,
            state: Some(StateSeat { key, fresh }),
        }
    }

    /// (resets, computes with their steps) a recorder saw.
    fn launches(rec: &Recorder) -> (usize, Vec<String>) {
        let calls = rec.take();
        let resets = calls
            .iter()
            .filter(|c| c.starts_with("launch reset"))
            .count();
        let computes = calls
            .iter()
            .filter(|c| c.contains("compute"))
            .cloned()
            .collect();
        (resets, computes)
    }

    /// A fresh lane zeroes slot 0's state; its continuation runs on it as
    /// it is; another state's lane replays that state's tokens from zero
    /// first (and so does the first lane coming back), a lane behind what
    /// its state folded (a loop run past it) replays its tokens before it;
    /// a copy's lane replays its source's tokens.
    #[test]
    fn a_lane_runs_on_its_own_recurrent_state() {
        let (mut m, rec) = model();
        m.fire_batch(&[lane(7, true, 0, vec![1, 2, 3])], 3, false)
            .expect("prefill");
        assert_eq!(launches(&rec).0, 1);
        // Decode at slot 3, 4 iterations: folds 4 more tokens.
        m.fire_batch(&[lane(7, false, 3, vec![9])], 4, false)
            .expect("decode");
        let (resets, computes) = launches(&rec);
        assert_eq!((resets, computes.len()), (0, 1));
        // Another state starts; then the first comes back at slot 7.
        m.fire_batch(&[lane(8, true, 0, vec![5])], 1, false)
            .expect("other");
        launches(&rec);
        m.fire_batch(&[lane(7, false, 7, vec![4])], 1, false)
            .expect("back");
        let (resets, computes) = launches(&rec);
        assert_eq!((resets, computes.len()), (1, 2), "{computes:?}");
        // A loop ran past slot 5 and was dropped: the state replays 5 tokens.
        m.fire_batch(&[lane(7, false, 5, vec![4])], 1, false)
            .expect("behind");
        let (resets, computes) = launches(&rec);
        assert_eq!((resets, computes.len()), (1, 2));
        // A lane past what its state knows is refused; a fresh one at a slot
        // past 0 too.
        assert!(
            m.fire_batch(&[lane(7, false, 9, vec![4])], 1, false)
                .is_err()
        );
        assert!(
            m.fire_batch(&[lane(9, true, 3, vec![4])], 1, false)
                .is_err()
        );
        // A copy's lane brings back its source's tokens.
        m.copy_state(7, 10).expect("copy");
        rec.take();
        m.fire_batch(&[lane(10, false, 6, vec![4])], 1, false)
            .expect("copy's lane");
        let (resets, computes) = launches(&rec);
        assert_eq!((resets, computes.len()), (1, 2));
        // A scratch launch leaves the state nobody's.
        m.fire_batch(
            &[BatchLane {
                state: None,
                ..lane(0, false, 0, vec![0])
            }],
            1,
            false,
        )
        .expect("scratch");
        rec.take();
        m.fire_batch(&[lane(10, false, 7, vec![4])], 1, false)
            .expect("after scratch");
        assert_eq!(launches(&rec).0, 1);
    }
}
