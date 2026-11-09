//! Qwen3.5-9B's checkpoint as the role-split program's tiles (the prep:
//! `<prefix>w<u>.npy`, `gam`, `cw`, `gpar`, `gnw`, `qkw`, `rot`, `hw`, `hg`,
//! `hemb`, `embed`, `meta.json`): 8 units of 256 x 256 weight PEs, 4 layers
//! each (GDN, GDN, GDN, attention), the head 128 columns x 1153 rows (216
//! vocabulary rows x 32 dims a PE). Weights as IEEE f16 bits, rounded to
//! nearest even from the checkpoint's bf16. Element for element the Python
//! prep's (`tools/split/prep_q35.py` at the 9B's dims, which also covers
//! other Qwen3.5 sizes and geometries).

use std::io::Write;
use std::path::Path;

use std::collections::HashMap;
use std::os::unix::fs::FileExt;

/// A Hugging Face safetensors checkpoint (sharded or not), read by tensor
/// name as stored.
pub struct Ckpt {
    files: Vec<std::fs::File>,
    entries: HashMap<String, (usize, String, Vec<usize>, u64, u64)>,
}

impl Ckpt {
    pub fn open(dir: &Path) -> R<Ckpt> {
        let index = dir.join("model.safetensors.index.json");
        let names: Vec<String> = if index.exists() {
            let v: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&index).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
            let mut f: Vec<String> = v["weight_map"]
                .as_object()
                .ok_or("no weight_map")?
                .values()
                .filter_map(|x| x.as_str().map(String::from))
                .collect();
            f.sort();
            f.dedup();
            f
        } else {
            vec!["model.safetensors".into()]
        };
        let (mut files, mut entries) = (Vec::new(), HashMap::new());
        for (i, n) in names.iter().enumerate() {
            let path = dir.join(n);
            let f = std::fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let mut b = [0u8; 8];
            f.read_exact_at(&mut b, 0).map_err(|e| e.to_string())?;
            let hl = u64::from_le_bytes(b);
            let mut h = vec![0u8; hl as usize];
            f.read_exact_at(&mut h, 8).map_err(|e| e.to_string())?;
            let v: serde_json::Value = serde_json::from_slice(&h).map_err(|e| e.to_string())?;
            for (k, e) in v.as_object().ok_or("bad header")? {
                if k == "__metadata__" {
                    continue;
                }
                let dt = e["dtype"].as_str().unwrap_or("").to_string();
                let shape = e["shape"]
                    .as_array()
                    .ok_or("no shape")?
                    .iter()
                    .map(|x| x.as_u64().unwrap_or(0) as usize)
                    .collect();
                let o = e["data_offsets"].as_array().ok_or("no offsets")?;
                entries.insert(
                    k.clone(),
                    (
                        i,
                        dt,
                        shape,
                        8 + hl + o[0].as_u64().unwrap_or(0),
                        8 + hl + o[1].as_u64().unwrap_or(0),
                    ),
                );
            }
            files.push(f);
        }
        Ok(Ckpt { files, entries })
    }

    /// A bf16 tensor's bits and shape (f32 tensors rounded to bf16 would
    /// lose bits: refused).
    pub fn bits(&self, name: &str) -> R<(Vec<usize>, Vec<u16>)> {
        let (i, dt, shape, a, b) = self
            .entries
            .get(name)
            .ok_or_else(|| format!("the checkpoint has no {name}"))?;
        if dt != "BF16" {
            return Err(format!("{name} is {dt}; the prep reads bf16"));
        }
        let mut raw = vec![0u8; (b - a) as usize];
        self.files[*i]
            .read_exact_at(&mut raw, *a)
            .map_err(|e| e.to_string())?;
        Ok((
            shape.clone(),
            raw.as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes(*c))
                .collect(),
        ))
    }

    /// A tensor as f32 (bf16 widened, f32 as is).
    pub fn f32s(&self, name: &str) -> R<Vec<f32>> {
        let (i, dt, _, a, b) = self
            .entries
            .get(name)
            .ok_or_else(|| format!("the checkpoint has no {name}"))?;
        let mut raw = vec![0u8; (b - a) as usize];
        self.files[*i]
            .read_exact_at(&mut raw, *a)
            .map_err(|e| e.to_string())?;
        match dt.as_str() {
            "BF16" => Ok(raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| bf(u16::from_le_bytes(*c)))
                .collect()),
            "F32" => Ok(raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()),
            d => Err(format!("{name} is {d}")),
        }
    }
}

type R<T> = std::result::Result<T, String>;

pub const PX: usize = 256;
pub const PY: usize = 256;
const HCC: usize = 16;
const IC: usize = 48;
pub const NU: usize = 8;
pub const HC: usize = 128;
pub const HR: usize = 1153;
pub const VR: usize = 216;
pub const HDC: usize = 32;
const H: usize = 4096;
const P: &str = "model.language_model.";

/// f32 to IEEE f16 bits, round to nearest even (subnormals kept, overflow
/// to infinity): numpy's `astype(float16)`.
pub fn f16_of(x: f32) -> u16 {
    let u = x.to_bits();
    let sign = ((u >> 16) & 0x8000) as u16;
    let a = u & 0x7fff_ffff;
    if a >= 0x7f80_0000 {
        return sign | 0x7c00 | if a > 0x7f80_0000 { 0x200 } else { 0 };
    }
    let e = (a >> 23) as i32 - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if a < 0x0080_0000 {
        // f32 subnormals are far below f16's least subnormal.
        return sign;
    }
    let m = (a & 0x7f_ffff) | 0x80_0000;
    let sh = 13 + (1 - e).max(0) as u32;
    if sh > 25 {
        return sign;
    }
    let (q, r) = (m >> sh, m & ((1 << sh) - 1));
    let half = 1u32 << (sh - 1);
    let q = q + u32::from(r > half || (r == half && q & 1 == 1));
    let bits = if e >= 1 {
        ((e as u32 - 1) << 10) + q
    } else {
        q
    };
    sign | bits as u16
}

fn bf(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// A 2-D bf16 matrix, its rows.
struct Mat {
    cols: usize,
    bits: Vec<u16>,
}

impl Mat {
    fn of(ckpt: &Ckpt, name: &str) -> R<Mat> {
        let (shape, bits) = ckpt.bits(name)?;
        let cols = *shape.last().unwrap();
        Ok(Mat { cols, bits })
    }
    fn at(&self, r: usize, c: usize) -> u16 {
        self.bits[r * self.cols + c]
    }
    fn f16(&self, r: usize, c: usize) -> u16 {
        f16_of(bf(self.at(r, c)))
    }
}

/// Rows stacked (np.concatenate along axis 0).
struct Stack<'a>(Vec<&'a Mat>);

impl Stack<'_> {
    fn f16(&self, mut r: usize, c: usize) -> u16 {
        for m in &self.0 {
            let n = m.bits.len() / m.cols;
            if r < n {
                return m.f16(r, c);
            }
            r -= n;
        }
        panic!("row past the stack");
    }
}

fn vecf(ckpt: &Ckpt, name: &str) -> R<Vec<f32>> {
    ckpt.f32s(name)
}

/// The tiles of one projection: `idx[c]` the output rows (in) or input
/// columns (out) of column c.
fn tile_in(blob: &mut [Vec<u16>], w: &Stack, idx: &[Vec<usize>]) {
    // (r, c, i, n) = W[idx[c][n]][r Hc + i]
    for r in 0..PY {
        for c in 0..PX {
            let out = &mut blob[r * PX + c];
            for i in 0..HCC {
                for &row in &idx[c] {
                    out.push(w.f16(row, r * HCC + i));
                }
            }
        }
    }
}

fn tile_out(blob: &mut [Vec<u16>], w: &Mat, idx: &[Vec<usize>]) {
    // (r, c, n, i) = W[r Hc + i][idx[c][n]]
    for r in 0..PY {
        for c in 0..PX {
            let out = &mut blob[r * PX + c];
            for &col in &idx[c] {
                for i in 0..HCC {
                    out.push(w.f16(r * HCC + i, col));
                }
            }
        }
    }
}

fn perm(p: usize) -> [usize; 4] {
    if p < 16 {
        [2 * p, 2 * p + 32, 2 * p + 1, 2 * p + 33]
    } else {
        let b = 64 + (p - 16) * 4;
        [b, b + 1, b + 2, b + 3]
    }
}

/// The prep's arrays, by name: (shape, words: u16 bits or f32 bits, 16-bit).
pub struct Arrays {
    pub items: Vec<(String, Vec<usize>, Vec<u32>, bool)>,
}

/// Unit `u`'s weight tiles `[PY][PX][LW]` (f16 bits).
pub fn unit(ckpt: &Ckpt, u: usize) -> R<Vec<u16>> {
    let cs: Vec<usize> = (0..PX).collect();
    let qidx: Vec<Vec<usize>> = cs
        .iter()
        .map(|&c| (0..8).map(|k| (c / 16) * 128 + (c % 16) * 8 + k).collect())
        .collect();
    let vidx: Vec<Vec<usize>> = cs
        .iter()
        .map(|&c| (0..16).map(|k| (c / 8) * 128 + (c % 8) * 16 + k).collect())
        .collect();
    let pd: Vec<[usize; 4]> = cs.iter().map(|&c| perm(c % 64)).collect();
    let mut blob: Vec<Vec<u16>> = (0..PX * PY)
        .map(|_| Vec::with_capacity(3 * 3360 + 3200))
        .collect();
    for l in 4 * u..4 * u + 4 {
        let lp = format!("{P}layers.{l}.");
        if l % 4 < 3 {
            let a = format!("{lp}linear_attn.");
            let (qkv, z, b, aa) = (
                Mat::of(ckpt, &format!("{a}in_proj_qkv.weight"))?,
                Mat::of(ckpt, &format!("{a}in_proj_z.weight"))?,
                Mat::of(ckpt, &format!("{a}in_proj_b.weight"))?,
                Mat::of(ckpt, &format!("{a}in_proj_a.weight"))?,
            );
            let win = Stack(vec![&qkv, &z, &b, &aa]);
            let idx: Vec<Vec<usize>> = cs
                .iter()
                .map(|&c| {
                    let mut v = qidx[c].clone();
                    v.extend(qidx[c].iter().map(|x| 2048 + x));
                    v.extend(vidx[c].iter().map(|x| 4096 + x));
                    v.extend(vidx[c].iter().map(|x| 8192 + x));
                    v.push(12288 + c / 8);
                    v.push(12320 + c / 8);
                    v
                })
                .collect();
            tile_in(&mut blob, &win, &idx);
            tile_out(
                &mut blob,
                &Mat::of(ckpt, &format!("{a}out_proj.weight"))?,
                &vidx,
            );
        } else {
            let a = format!("{lp}self_attn.");
            let (q, k, v) = (
                Mat::of(ckpt, &format!("{a}q_proj.weight"))?,
                Mat::of(ckpt, &format!("{a}k_proj.weight"))?,
                Mat::of(ckpt, &format!("{a}v_proj.weight"))?,
            );
            let win = Stack(vec![&q, &k, &v]);
            let idx: Vec<Vec<usize>> = cs
                .iter()
                .map(|&c| {
                    let kvh = c / 64;
                    let mut out = Vec::with_capacity(40);
                    for h in 0..4 {
                        out.extend(pd[c].iter().map(|d| (4 * kvh + h) * 512 + d));
                    }
                    for h in 0..4 {
                        out.extend(pd[c].iter().map(|d| (4 * kvh + h) * 512 + 256 + d));
                    }
                    out.extend(pd[c].iter().map(|d| 8192 + kvh * 256 + d));
                    out.extend(pd[c].iter().map(|d| 9216 + kvh * 256 + d));
                    out
                })
                .collect();
            tile_in(&mut blob, &win, &idx);
            let oidx: Vec<Vec<usize>> = cs
                .iter()
                .map(|&c| {
                    (0..4)
                        .flat_map(|h| pd[c].iter().map(move |d| (4 * (c / 64) + h) * 256 + d))
                        .collect()
                })
                .collect();
            tile_out(
                &mut blob,
                &Mat::of(ckpt, &format!("{a}o_proj.weight"))?,
                &oidx,
            );
        }
        let m = format!("{lp}mlp.");
        let (g, up) = (
            Mat::of(ckpt, &format!("{m}gate_proj.weight"))?,
            Mat::of(ckpt, &format!("{m}up_proj.weight"))?,
        );
        let gi: Vec<Vec<usize>> = cs
            .iter()
            .map(|&c| (0..IC).map(|k| c * IC + k).collect())
            .collect();
        let gu: Vec<Vec<usize>> = gi
            .iter()
            .map(|v| {
                v.iter()
                    .copied()
                    .chain(v.iter().map(|x| 12288 + x))
                    .collect()
            })
            .collect();
        tile_in(&mut blob, &Stack(vec![&g, &up]), &gu);
        tile_out(
            &mut blob,
            &Mat::of(ckpt, &format!("{m}down_proj.weight"))?,
            &gi,
        );
    }
    let lw = blob[0].len();
    if lw != 3 * 3360 + 3200 || blob.iter().any(|b| b.len() != lw) {
        return Err(format!("unit {u}: tiles of {lw} words"));
    }
    Ok(blob.concat())
}

/// The control rows' and columns' tables and the head's.
pub fn small(ckpt: &Ckpt) -> R<Arrays> {
    let mut items = Vec::new();
    let mut gam = vec![0u32; NU * PY * 128];
    let mut cw = vec![0f32; NU * PX * 384];
    let mut gpar = vec![0f32; NU * PX * 6];
    let mut gnw = vec![0f32; NU * PX * 48];
    let mut qkw = vec![0f32; NU * PX * 20];
    for u in 0..NU {
        let mut g = 0;
        for l in 4 * u..4 * u + 4 {
            let lp = format!("{P}layers.{l}.");
            let ln1 = vecf(ckpt, &format!("{lp}input_layernorm.weight"))?;
            let ln2 = vecf(ckpt, &format!("{lp}post_attention_layernorm.weight"))?;
            for r in 0..PY {
                for i in 0..HCC {
                    let at = (u * PY + r) * 128 + (l % 4) * 32;
                    gam[at + i] = u32::from(f16_of(1.0 + ln1[r * HCC + i]));
                    gam[at + HCC + i] = u32::from(f16_of(1.0 + ln2[r * HCC + i]));
                }
            }
            if l % 4 < 3 {
                let a = format!("{lp}linear_attn.");
                let (sh, conv) = ckpt.bits(&format!("{a}conv1d.weight"))?;
                let taps = sh[2];
                let al = vecf(ckpt, &format!("{a}A_log"))?;
                let dt = vecf(ckpt, &format!("{a}dt_bias"))?;
                let nw = vecf(ckpt, &format!("{a}norm.weight"))?;
                for c in 0..PX {
                    let (kh, pk, vh, pv) = (c / 16, c % 16, c / 8, c % 8);
                    let ch: Vec<usize> = (0..8)
                        .map(|k| kh * 128 + pk * 8 + k)
                        .chain((0..8).map(|k| 2048 + kh * 128 + pk * 8 + k))
                        .chain((0..16).map(|k| 4096 + vh * 128 + pv * 16 + k))
                        .collect();
                    for t in 0..4 {
                        for (j, &chn) in ch.iter().enumerate() {
                            cw[(u * PX + c) * 384 + g * 128 + t * 32 + j] =
                                bf(conv[chn * taps + t]);
                        }
                    }
                    gpar[(u * PX + c) * 6 + 2 * g] = al[vh];
                    gpar[(u * PX + c) * 6 + 2 * g + 1] = dt[vh];
                    for k in 0..16 {
                        gnw[(u * PX + c) * 48 + g * 16 + k] = nw[pv * 16 + k];
                    }
                }
                g += 1;
            } else {
                let a = format!("{lp}self_attn.");
                let qn = vecf(ckpt, &format!("{a}q_norm.weight"))?;
                let kn = vecf(ckpt, &format!("{a}k_norm.weight"))?;
                for c in 0..PX {
                    let pd = perm(c % 64);
                    for h in 0..4 {
                        for d in 0..4 {
                            qkw[(u * PX + c) * 20 + h * 4 + d] = (1.0 + qn[pd[d]]) / 16.0;
                        }
                    }
                    for d in 0..4 {
                        qkw[(u * PX + c) * 20 + 16 + d] = 1.0 + kn[pd[d]];
                    }
                }
            }
        }
    }
    let f = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<u32>>();
    items.push(("gam".into(), vec![NU, PY, 128], gam, true));
    items.push(("cw".into(), vec![NU, PX, 384], f(cw), false));
    items.push(("gpar".into(), vec![NU, PX, 6], f(gpar), false));
    items.push(("gnw".into(), vec![NU, PX, 48], f(gnw), false));
    items.push(("qkw".into(), vec![NU, PX, 20], f(qkw), false));
    let theta: f64 = 1.0e7;
    let inv: Vec<f64> = (0..32)
        .map(|k| 1.0 / theta.powf((2 * k) as f64 / 64.0))
        .collect();
    // RoPE tables: 5 levels of 8 positions (32,768 >= the context NE 64 x PY 256)
    const RLV: usize = 5;
    let mut rot = vec![0f32; PX * RLV * 32];
    for p in 0..PX {
        for j in 0..2 {
            let fr = if p % 64 < 16 {
                inv[2 * (p % 64) + j]
            } else {
                0.0
            };
            for lv in 0..RLV {
                for d in 0..8 {
                    let ang = (d * 8usize.pow((RLV - 1 - lv) as u32)) as f64 * fr;
                    let at = p * RLV * 32 + ((lv * 8 + d) * 2) * 2;
                    rot[at + j] = ang.cos() as f32;
                    rot[at + 2 + j] = ang.sin() as f32;
                }
            }
        }
    }
    items.push(("rot".into(), vec![PX, RLV * 32], f(rot), false));
    let nw = vecf(ckpt, &format!("{P}norm.weight"))?;
    items.push((
        "hg".into(),
        vec![HC, HDC],
        f(nw.iter().map(|x| 1.0 + x).collect()),
        false,
    ));
    Ok(Arrays { items })
}

/// The head's weight tiles `[HR][HC][HDc VR]` (f16 bits: (R, c, d VR + j)
/// = lm_head[R VR + j][c HDc + d]) and the embedding's `[HR][HC][VR HDc]`
/// (bf16 bits: (R, c, j HDc + d) = embed[R VR + j][c HDc + d]). The head
/// holds HR VR rows, more than the vocabulary: the lm_head's rows past it are
/// copies of its last row, so their logits tie with it and the argmax (the
/// first maximum) never picks a padded id; zero rows would win whenever every
/// real logit is negative. The embedding's padded rows stay 0 (never sent).
pub fn head(ckpt: &Ckpt, name: &str, f16: bool) -> R<Vec<u16>> {
    let m = Mat::of(ckpt, name)?;
    let v = m.bits.len() / m.cols;
    let mut out = vec![0u16; HR * HC * HDC * VR];
    for r in 0..HR {
        for j in 0..VR {
            let mut row = r * VR + j;
            if row >= v {
                if !f16 || v == 0 {
                    continue;
                }
                row = v - 1;
            }
            for c in 0..HC {
                for d in 0..HDC {
                    let b = m.at(row, c * HDC + d);
                    let at = (r * HC + c) * HDC * VR;
                    if f16 {
                        out[at + d * VR + j] = f16_of(bf(b));
                    } else {
                        out[at + j * HDC + d] = b;
                    }
                }
            }
        }
    }
    Ok(out)
}

/// A `.npy` file as numpy writes it (format 1.0, the header padded to 64).
pub fn write_npy(path: &Path, shape: &[usize], words: &[u32], b16: bool) -> R<()> {
    let sh = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!(
            "({})",
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let descr = if b16 { "<u2" } else { "<f4" };
    let mut h = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {sh}, }}");
    let total = (10 + h.len() + 1).div_ceil(64) * 64;
    while 10 + h.len() + 1 < total {
        h.push(' ');
    }
    h.push('\n');
    let mut f = std::io::BufWriter::new(
        std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?,
    );
    let mut head = b"\x93NUMPY\x01\x00".to_vec();
    head.extend((h.len() as u16).to_le_bytes());
    head.extend(h.as_bytes());
    f.write_all(&head).map_err(|e| e.to_string())?;
    if b16 {
        let b: Vec<u8> = words
            .iter()
            .flat_map(|&w| (w as u16).to_le_bytes())
            .collect();
        f.write_all(&b).map_err(|e| e.to_string())?;
    } else {
        let b: Vec<u8> = words.iter().flat_map(|&w| w.to_le_bytes()).collect();
        f.write_all(&b).map_err(|e| e.to_string())?;
    }
    f.flush().map_err(|e| e.to_string())
}

/// The whole prep into `dir` (`prefix` before every name).
pub fn write(ckpt: &Ckpt, dir: &Path, prefix: &str) -> R<()> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let p = |n: &str| dir.join(format!("{prefix}{n}"));
    for u in 0..NU {
        let w = unit(ckpt, u)?;
        let words: Vec<u32> = w.into_iter().map(u32::from).collect();
        write_npy(
            &p(&format!("w{u}.npy")),
            &[PY, PX, 3 * 3360 + 3200],
            &words,
            true,
        )?;
        tracing::info!(unit = u, "split prep: unit written");
    }
    for (n, shape, words, b16) in small(ckpt)?.items {
        write_npy(&p(&format!("{n}.npy")), &shape, &words, b16)?;
    }
    let hw = head(ckpt, "lm_head.weight", true)?;
    write_npy(
        &p("hw.npy"),
        &[HR, HC, HDC * VR],
        &hw.into_iter().map(u32::from).collect::<Vec<_>>(),
        true,
    )?;
    let he = head(ckpt, &format!("{P}embed_tokens.weight"), false)?;
    write_npy(
        &p("hemb.npy"),
        &[HR, HC, VR * HDC],
        &he.into_iter().map(u32::from).collect::<Vec<_>>(),
        true,
    )?;
    let meta = serde_json::json!({"PX": 256, "PY": 256, "UNX": 2, "UNY": 4, "HC": 128, "HR": 1153, "VR": 216, "Hc": 16,
        "SWA": 64, "SWG": 16, "SWV": 8, "DKG": 8, "DKH": 128, "has_att": true, "has_gdn": true, "K": 4, "KM": 8,
        "Ic": 48, "NQA": 40, "NQG": 50, "Qh": 4, "Dc": 4, "DK": 8, "DV": 16, "NE": 64, "HD": 256, "NROT": 2,
        "HK": 128, "HV": 128, "H": H, "RLV": 5, "RBITS": 3});
    std::fs::write(p("meta.json"), meta.to_string()).map_err(|e| e.to_string())
}
