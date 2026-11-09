//! The serve_g4 wafer program's host side (`spatial/csl/serve_g4`): Gemma 4
//! dense as gemma4_layer grids in serve2d's chain ([`super::program::Gemma`]),
//! the f32 head2d over the tied embedding (serve2d's head, its norm as is),
//! the per-layer embedding (PLE) tables in the region east of the head, a
//! paged pool on every grid (a shared-tail layer's fed by the handoff), the
//! launch serve2d's.
//!
//! The placement is the bare-metal host's (`payload/serve_g4/g4host.py`),
//! tile for tile, from a Hugging Face directory by Gemma 4's own names:
//!
//! - A grid splits the hidden state as serve2d's (row `py`: hidden chunk
//!   `py`, `Hc` dims) and the MLP likewise (`Ic` rows a column). Column `c`
//!   of kv head `c / Gs` holds `Dc` dims of the head, RoPE's turned pairs
//!   first (a sliding head turns every pair `(i, i + d/2)`; a global one the
//!   first `rotary / 2`), the unturned after them paired with a turn of 0,
//!   its `Qh` query heads' and the kv head's (zero on a shared-tail layer)
//!   in the bf16 `wqa`/`wqv` halves, their norm weights as they are (the
//!   softmax scale is 1).
//! - bf16 symbols (`W16`) take u16 elements after one leading pad; `kc` and
//!   `vc` an f32 pad; a shared-tail grid holds no `wv`.
//! - The PLE projections: `wpm` the model projection's rows of the layer
//!   (applied to the raw embedding row x0, as the checkpoint's: the grids'
//!   x0 is unscaled), `wpg` the input gate, `wpp` the projection back,
//!   `pnw` the projection norm / sqrt(2); the five norms, the layer scalar.
//! - Band `sy` of the PLE region (rows `sy P..`) holds strip `sy`'s layers'
//!   tables: vocabulary row `t` whole in PE `t / RPP` (column `% NPC`, band
//!   row `/ NPC`), slot `t % RPP`, the strip's `GW x PLE` bf16 two a word
//!   (element `2i` low), raw (the grids scale the row).

use super::ckpt::{Ckpt, Tensor};
use super::program::{Gemma, ProgramSpec};
use super::wafer::rect;
use crate::spatial::device::Device;

type R<T> = std::result::Result<T, String>;

/// gemma4_layer's bf16 (u16) symbols.
const W16: &[&str] = &["wqa", "wqv", "wv", "wo", "wgu", "wd", "wpm", "wpg", "wpp"];
/// Its f32 symbols with one leading pad word a PE.
const PADDED: &[&str] = &["kc", "vc"];

/// f32 to bf16 bits (round to nearest even; exact for bf16 values).
fn bf16(x: f32) -> u32 {
    u32::from(super::ckpt::bf16_bits(x))
}

/// `data` (row-major by PE over the `p` x `p` grid at `at`, `n` a PE) to
/// `symbol`, as g4host's `put`.
fn put(dev: &mut dyn Device, symbol: &str, data: &[f32], at: (usize, usize), p: usize) -> R<()> {
    let n = data.len() / (p * p);
    let r = rect(at.0, at.1, p, p);
    if W16.contains(&symbol) {
        let words: Vec<u32> = data
            .chunks_exact(n)
            .flat_map(|pe| std::iter::once(0).chain(pe.iter().map(|&x| bf16(x))))
            .collect();
        dev.put16(symbol, r, n + 1, &words)
    } else {
        let pad = PADDED.contains(&symbol);
        let words: Vec<u32> = data
            .chunks_exact(n)
            .flat_map(|pe| {
                (pad.then_some(0))
                    .into_iter()
                    .chain(pe.iter().map(|x| x.to_bits()))
            })
            .collect();
        dev.put(symbol, r, n + usize::from(pad), &words)
    }
}

fn tensor(ckpt: &Ckpt, leaf: &str, shape: &[usize]) -> R<Tensor> {
    let t = ckpt.hf(leaf)?;
    t.expect_shape(leaf, shape)?;
    Ok(t)
}

/// RoPE as the checkpoint states it: the sliding layers' theta, the global
/// ones' and the fraction of a global head they turn.
#[derive(Debug, Clone, Copy)]
struct Rope {
    theta_s: f64,
    theta_g: f64,
    partial_g: f64,
}

fn rope(ckpt: &Ckpt) -> R<Rope> {
    let c = ckpt.config();
    let text = if c["text_config"].is_object() {
        &c["text_config"]
    } else {
        c
    };
    let rp = &text["rope_parameters"];
    let theta = |kind: &str| {
        rp[kind]["rope_theta"]
            .as_f64()
            .ok_or_else(|| format!("the checkpoint config states no {kind} rope_theta"))
    };
    Ok(Rope {
        theta_s: theta("sliding_attention")?,
        theta_g: theta("full_attention")?,
        partial_g: rp["full_attention"]["partial_rotary_factor"]
            .as_f64()
            .unwrap_or(1.0),
    })
}

/// A layer's reading: its dims a column, head dim and the column pairs
/// `(a, b, inverse frequency)`, turned first.
fn reading(
    lay: &ProgramSpec,
    g: &Gemma,
    rp: Rope,
    l: usize,
) -> (usize, usize, Vec<(usize, usize, f32)>) {
    let dc = g.dc(l);
    let d = lay.gs() * dc;
    let (rot, theta) = if g.global(l) {
        ((rp.partial_g * d as f64).round() as usize, rp.theta_g)
    } else {
        (d, rp.theta_s)
    };
    // numpy: (theta ** (-arange(0, rot, 2) / d)).astype(f32), in f64.
    let inv = |i: usize| theta.powf(-((2 * i) as f64) / d as f64) as f32;
    let mut pairs: Vec<(usize, usize, f32)> =
        (0..rot / 2).map(|i| (i, i + d / 2, inv(i))).collect();
    let used: std::collections::HashSet<usize> = pairs.iter().flat_map(|t| [t.0, t.1]).collect();
    let rest: Vec<usize> = (0..d).filter(|k| !used.contains(k)).collect();
    pairs.extend(rest.as_chunks::<2>().0.iter().map(|&[a, b]| (a, b, 0.0)));
    (dc, d, pairs)
}

/// Every layer grid and the PLE region of `ckpt` for `lay`.
pub(crate) fn place(lay: &ProgramSpec, ckpt: &Ckpt, dev: &mut dyn Device) -> R<()> {
    let g = lay
        .gemma()
        .ok_or("serve_g4's placement on a program without Gemma's layers")?;
    let rp = rope(ckpt)?;
    for l in 0..lay.layers() {
        let tiles = layer_tiles(lay, &g, rp, ckpt, l)?;
        let origin = lay.origin(l);
        for (sym, data) in &tiles {
            put(dev, sym, data, origin, lay.px())?;
        }
        tracing::info!(layer = l, "spatial wafer: serve_g4 layer placed");
    }
    place_ple(lay, &g, ckpt, dev)?;
    tracing::info!("spatial wafer: serve_g4 PLE tables placed");
    Ok(())
}

/// Layer `l`'s tiles (g4host's `place_layer`), each row-major by PE.
fn layer_tiles(
    lay: &ProgramSpec,
    g: &Gemma,
    rp: Rope,
    ckpt: &Ckpt,
    l: usize,
) -> R<Vec<(&'static str, Vec<f32>)>> {
    let (p, hc, gs, qh) = (lay.px(), lay.hc(), lay.gs(), lay.qh());
    let hid = lay.hidden();
    let shape = lay.paged().ok_or("serve_g4 without its pool")?;
    let (dc, d, pairs) = reading(lay, g, rp, l);
    let (dh, nh1) = (dc / 2, qh + 1);
    let kh = p / gs;
    let own = g.source(l) == l;
    let ic = g.ic(l);
    let inter = p * ic;
    let ple = p * g.pc;
    let pc = g.pc;
    let n = |s: &str| format!("layers.{l}.{s}");
    let wq = tensor(ckpt, &n("self_attn.q_proj.weight"), &[kh * qh * d, hid])?;
    let wo = tensor(ckpt, &n("self_attn.o_proj.weight"), &[hid, kh * qh * d])?;
    let qn = tensor(ckpt, &n("self_attn.q_norm.weight"), &[d])?;
    let (wk, wv, kn) = if own {
        (
            Some(tensor(ckpt, &n("self_attn.k_proj.weight"), &[kh * d, hid])?),
            Some(tensor(ckpt, &n("self_attn.v_proj.weight"), &[kh * d, hid])?),
            Some(tensor(ckpt, &n("self_attn.k_norm.weight"), &[d])?),
        )
    } else {
        (None, None, None)
    };
    let wg = tensor(ckpt, &n("mlp.gate_proj.weight"), &[inter, hid])?;
    let wu = tensor(ckpt, &n("mlp.up_proj.weight"), &[inter, hid])?;
    let wd = tensor(ckpt, &n("mlp.down_proj.weight"), &[hid, inter])?;
    let proj = tensor(
        ckpt,
        "per_layer_model_projection.weight",
        &[g.nl * ple, hid],
    )?;
    let plg = tensor(ckpt, &n("per_layer_input_gate.weight"), &[ple, hid])?;
    let plp = tensor(ckpt, &n("per_layer_projection.weight"), &[hid, ple])?;
    let pn = tensor(ckpt, "per_layer_projection_norm.weight", &[ple])?;
    let pes = p * p;
    let z = |k: usize| vec![0f32; pes * k];
    let mut wqa = z(hc * nh1 * dh);
    let mut wqv = z(hc * nh1 * dh);
    let mut wvb = z(hc * dc);
    let mut qkn = z(2 * nh1 * dh);
    let mut rot_c = z(3 * 16 * dh);
    let mut rot_s = z(3 * 16 * dh);
    let mut wob = z(qh * dc * hc);
    let mut wgu = z(hc * 2 * ic);
    let mut wdb = z(ic * hc);
    let mut wpm = z(hc * pc);
    let mut wpg = z(hc * pc);
    let mut wpp = z(pc * hc);
    let mut pnw = z(pc);
    let sqrt2 = 2f64.sqrt() as f32;
    for col in 0..p {
        let (kv, j) = (col / gs, col % gs);
        let prs = &pairs[j * dh..(j + 1) * dh];
        let dl: Vec<usize> = prs
            .iter()
            .map(|t| t.0)
            .chain(prs.iter().map(|t| t.1))
            .collect();
        // Query head h's row of its e-th dim; the kv head's.
        let qrow = |h: usize, e: usize| (qh * kv + h) * d + dl[e];
        let krow = |e: usize| kv * d + dl[e];
        for py in 0..p {
            let pe = py * p + col;
            for jj in 0..hc {
                let x = py * hc + jj;
                for h in 0..nh1 {
                    for k in 0..dh {
                        let (a, b) = if h < qh {
                            (
                                wq.data[qrow(h, k) * hid + x],
                                wq.data[qrow(h, dh + k) * hid + x],
                            )
                        } else if let Some(wk) = &wk {
                            (wk.data[krow(k) * hid + x], wk.data[krow(dh + k) * hid + x])
                        } else {
                            (0.0, 0.0)
                        };
                        wqa[((pe * hc + jj) * nh1 + h) * dh + k] = a;
                        wqv[((pe * hc + jj) * nh1 + h) * dh + k] = b;
                    }
                }
                if let Some(wv) = &wv {
                    for e in 0..dc {
                        wvb[(pe * hc + jj) * dc + e] = wv.data[krow(e) * hid + x];
                    }
                }
                // wo: [Qh Dc][Hc], query head h's dims dl in order.
                for h in 0..qh {
                    for e in 0..dc {
                        wob[(pe * qh * dc + h * dc + e) * hc + jj] =
                            wo.data[x * (kh * qh * d) + qrow(h, e)];
                    }
                }
                for r in 0..ic {
                    let m = col * ic + r;
                    wgu[(pe * hc + jj) * 2 * ic + r] = wg.data[m * hid + x];
                    wgu[(pe * hc + jj) * 2 * ic + ic + r] = wu.data[m * hid + x];
                    wdb[(pe * ic + r) * hc + jj] = wd.data[x * inter + m];
                }
                for r in 0..pc {
                    let m = col * pc + r;
                    wpm[(pe * hc + jj) * pc + r] = proj.data[(l * ple + m) * hid + x];
                    wpg[(pe * hc + jj) * pc + r] = plg.data[m * hid + x];
                    wpp[(pe * pc + r) * hc + jj] = plp.data[x * ple + m];
                }
            }
            for h in 0..nh1 {
                for k in 0..dh {
                    let w = |e: usize| {
                        if h < qh {
                            qn.data[dl[e]]
                        } else {
                            kn.as_ref().map_or(0.0, |kn| kn.data[dl[e]])
                        }
                    };
                    qkn[((pe * 2) * nh1 + h) * dh + k] = w(k);
                    qkn[((pe * 2 + 1) * nh1 + h) * dh + k] = w(dh + k);
                }
            }
            for (lv, unit) in [256.0f64, 16.0, 1.0].into_iter().enumerate() {
                for a in 0..16 {
                    for (k, t) in prs.iter().enumerate() {
                        let ang = a as f64 * unit * f64::from(t.2);
                        rot_c[((pe * 3 + lv) * 16 + a) * dh + k] = ang.cos() as f32;
                        rot_s[((pe * 3 + lv) * 16 + a) * dh + k] = ang.sin() as f32;
                    }
                }
            }
            for r in 0..pc {
                // numpy: pn / f32(sqrt(2)).
                pnw[pe * pc + r] = pn.data[col * pc + r] / sqrt2;
            }
        }
    }
    let mut out = vec![
        ("wqa", wqa),
        ("wqv", wqv),
        ("qkn_w", qkn),
        ("rot_c", rot_c),
        ("rot_s", rot_s),
        ("wo", wob),
        ("wgu", wgu),
        ("wd", wdb),
        ("wpm", wpm),
        ("wpg", wpg),
        ("wpp", wpp),
        ("pnw", pnw),
    ];
    if own {
        out.insert(2, ("wv", wvb));
    }
    for (sym, name) in [
        ("gamma1", "input_layernorm"),
        ("gamma2", "pre_feedforward_layernorm"),
        ("gamma3", "post_attention_layernorm"),
        ("gamma4", "post_feedforward_layernorm"),
        ("gamma5", "post_per_layer_input_norm"),
    ] {
        let w = tensor(ckpt, &n(&format!("{name}.weight")), &[hid])?;
        let tile: Vec<f32> = (0..pes)
            .flat_map(|pe| {
                let py = pe / p;
                w.data[py * hc..(py + 1) * hc].to_vec()
            })
            .collect();
        out.push((sym, tile));
    }
    let scal = tensor(ckpt, &n("layer_scalar"), &[1])?;
    out.push(("scal", vec![scal.data[0]; pes]));
    let pool = vec![0f32; pes * shape.blocks * dc * shape.page_size];
    out.push(("kc", pool.clone()));
    out.push(("vc", pool));
    Ok(out)
}

/// The PLE region (g4host's `place_ple`): band `sy`'s `ptab`, one pass over
/// the table's rows for every band.
fn place_ple(lay: &ProgramSpec, g: &Gemma, ckpt: &Ckpt, dev: &mut dyn Device) -> R<()> {
    let (p, gw) = (lay.px(), lay.gw());
    let ple = p * g.pc;
    if g.pc == 0 {
        return Ok(());
    }
    let nww = g.row_words(gw, p);
    let (npc, rpp) = (g.npc, g.rpp);
    let vocab = lay.vocab();
    let x0 = lay.head_x() + lay.py() + lay.vmc();
    let leaf = "embed_tokens_per_layer.weight";
    let bands = g.nl.div_ceil(gw);
    let per_pe = rpp * nww;
    let mut words: Vec<Vec<u32>> = (0..bands).map(|_| vec![0u32; p * npc * per_pe]).collect();
    let chunk = 2048;
    let mut lo = 0;
    while lo < vocab {
        let hi = (lo + chunk).min(vocab);
        let (shape, bits) = ckpt.hf_rows_bits(leaf, lo, hi)?;
        if shape != [vocab, g.nl * ple] {
            return Err(format!(
                "{leaf} is {shape:?}; the program wants [{vocab}, {}]",
                g.nl * ple
            ));
        }
        let width = g.nl * ple;
        for t in lo..hi {
            let src = &bits[(t - lo) * width..(t - lo + 1) * width];
            // Vocabulary row t: PE t / RPP of the band, slot t % RPP.
            let at = (t / rpp) * per_pe + (t % rpp) * nww;
            for (sy, band) in words.iter_mut().enumerate() {
                for k in 0..gw {
                    let l = sy * gw + k;
                    if l >= g.nl {
                        continue;
                    }
                    let row = &src[l * ple..(l + 1) * ple];
                    for i in 0..ple / 2 {
                        band[at + (k * ple) / 2 + i] =
                            u32::from(row[2 * i]) | (u32::from(row[2 * i + 1]) << 16);
                    }
                }
            }
        }
        lo = hi;
    }
    for (sy, band) in words.iter().enumerate() {
        dev.put("ptab", rect(x0, sy * p, npc, p), per_pe, band)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::spatial::ckpt::Ckpt;
    use crate::spatial::codegen::Design;
    use crate::spatial::device::Recorder;
    use crate::spatial::program::{ProgramSpec, parse_params};

    /// `G4_CMP=<dir>`: the placement of `<dir>/ckpt` (a Hugging Face
    /// directory) for `<dir>/small.params` dumped to `<dir>/rs.bin`, every
    /// PE's words of every symbol, to set against a reference placement's.
    #[test]
    #[ignore]
    fn dump_placement() {
        let dir = std::path::PathBuf::from(std::env::var("G4_CMP").expect("G4_CMP"));
        let params = parse_params(
            std::fs::read_to_string(dir.join("small.params"))
                .unwrap()
                .trim(),
        );
        let spec = ProgramSpec::from_params(Design::ServeG4, &params).expect("a serve_g4 program");
        let ckpt = Ckpt::open(&dir.join("ckpt")).expect("the checkpoint");
        // G4_FORGET: a full-size run that keeps nothing (shapes and time).
        let rec = if std::env::var("G4_FORGET").is_ok() {
            Recorder::default().forgetful()
        } else {
            Recorder::default()
        };
        let mut dev = rec.clone();
        super::place(&spec, &ckpt, &mut dev).expect("the layers");
        crate::spatial::serve2d::place_head(&spec, &ckpt, &mut dev).expect("the head");
        let mut out = Vec::new();
        for ((sym, x, y), words) in rec.dump() {
            out.extend((sym.len() as u32).to_le_bytes());
            out.extend(sym.as_bytes());
            out.extend(x.to_le_bytes());
            out.extend(y.to_le_bytes());
            out.extend((words.len() as u32).to_le_bytes());
            for w in words {
                out.extend(w.to_le_bytes());
            }
        }
        std::fs::write(dir.join("rs.bin"), out).unwrap();
    }
}
