//! moe_mxfp4's host placement: one routed MoE layer's per-PE tiles, as the
//! bare-metal host builds them (`payload/moe_mxfp4/moe_pack.py`, word for
//! word).
//!
//! A grid of `px` x `py` PEs: row `py` holds hidden dims `[Hc py, Hc py +
//! Hc)`, column `px` intermediate dims `[Ic px, Ic px + Ic)` of gate and of
//! up. Each PE stores, for every expert, its gate/up tile (Hc inputs x 2 Ic
//! outputs) and its down tile (Ic inputs x Hc outputs) as three planes of
//! ceil(L / 3) input rows of MXFP4 codes, three a u16 (plane p's code w in
//! word w: sign at bit 15 - 3p, e1 e0 m at 8 - 3p .. 6 - 3p), then a word of
//! two scale bases (gate/up low byte, down high byte: the f32 exponent field
//! scale + 126 - MXK, less the offsets) and each tile's 2R scale offsets
//! (segment 0's R, then segment 1's: the inputs in the tile's first and
//! second 32-block) in four nibble planes. A tile whose offsets span more
//! than 15 is refused.
//!
//! Inputs follow the checkpoint with gate/up de-interleaved: gate rows, then
//! up rows.

use super::model::{Refused, refuse};

/// Inputs prescaled by 2^MXK on the device (moe_mxfp4's `MXK`).
pub const MXK: i32 = 100;

/// One layer's experts, router and biases (row-major).
pub struct MoeLayer<'a> {
    pub experts: usize,
    pub hidden: usize,
    pub inter: usize,
    /// `[experts][2 inter][hidden_p]` fp4 codes 0..15 (gate rows, then up).
    pub gu_codes: &'a [u8],
    pub hidden_p: usize,
    /// `[experts][2 inter][hidden_p / 32]` E8M0.
    pub gu_scales: &'a [u8],
    /// `[experts][2 inter]`.
    pub gu_bias: &'a [f32],
    /// `[experts][hidden][inter_p]`.
    pub dn_codes: &'a [u8],
    pub inter_p: usize,
    /// `[experts][hidden][inter_p / 32]`.
    pub dn_scales: &'a [u8],
    /// `[experts][hidden]`.
    pub dn_bias: &'a [f32],
    /// `[experts][hidden]` (bf16 values).
    pub router: &'a [f32],
    /// `[experts]`.
    pub router_bias: &'a [f32],
}

/// The grid's split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeGrid {
    pub px: usize,
    pub py: usize,
    pub hc: usize,
    pub ic: usize,
}

impl MoeGrid {
    pub fn g2(&self) -> usize {
        2 * self.ic
    }
    fn gn3(&self) -> usize {
        self.hc.div_ceil(3) * self.g2()
    }
    fn dn3(&self) -> usize {
        self.ic.div_ceil(3) * self.hc
    }
    fn n4g(&self) -> usize {
        (2 * self.g2()).div_ceil(4)
    }
    fn n4d(&self) -> usize {
        (2 * self.hc).div_ceil(4)
    }
    /// u16 words an expert (moe_mxfp4's `XW`).
    pub fn xw(&self) -> usize {
        self.gn3() + self.dn3() + 1 + self.n4g() + self.n4d()
    }
    /// `mx`'s u32 words a PE.
    pub fn mx_words(&self, experts: usize) -> usize {
        (experts * self.xw()).div_ceil(2)
    }
    /// Gate/up bias slots a PE (experts `py + k PY`), down bias slots
    /// (experts `px + k PX`).
    pub fn ner(&self, experts: usize) -> usize {
        experts.div_ceil(self.py)
    }
    pub fn nec(&self, experts: usize) -> usize {
        experts.div_ceil(self.px)
    }
}

/// Every PE's symbols, `[py][px][..]`.
#[derive(Debug, Clone, PartialEq)]
pub struct MoeTiles {
    /// Expert storage, u16 pairs (element `2i` low).
    pub mx: Vec<u32>,
    /// Router tile `[Hc][NE]` (bf16 values).
    pub rw: Vec<f32>,
    /// Router bias `[NE]`.
    pub rb: Vec<f32>,
    /// Gate/up biases `[NER][G2]`.
    pub gub: Vec<f32>,
    /// Down biases `[NEC][Hc]`.
    pub dnb: Vec<f32>,
    /// Inputs in the first 32-block: gate/up, down.
    pub mcfg: Vec<u32>,
}

/// A tile `[L][R]` of codes into three planes of u16 words.
pub fn plane_pack(codes: &[u16], l: usize, r: usize) -> Vec<u16> {
    let n3 = l.div_ceil(3) * r;
    let mut w = vec![0u16; n3];
    for p in 0..3 {
        for (k, wk) in w.iter_mut().enumerate() {
            let c = codes.get(p * n3 + k).copied().unwrap_or(0);
            *wk |= ((c & 8) << (12 - 3 * p)) | ((c & 7) << (6 - 3 * p));
        }
    }
    w
}

/// Values 0..15 into ceil(n / 4) words: value `k n4 + w` in nibble k of word w.
pub fn nibble_pack(v: &[u16]) -> Vec<u16> {
    let n4 = v.len().div_ceil(4);
    let mut w = vec![0u16; n4];
    for (i, &x) in v.iter().enumerate() {
        w[i % n4] |= x << (4 * (i / n4));
    }
    w
}

fn fieldb(sc: u16) -> Result<i32, Refused> {
    let v = sc as i32 + 126 - MXK;
    if v <= 0 || v >= 255 {
        return refuse(format!("MXFP4 scale {sc}: exponent field {v} out of range"));
    }
    Ok(v)
}

/// A tile's exponent fields (segment 0's, then segment 1's or segment 0's
/// again) as a base and offsets in 0..15.
fn offsets(seg0: &[i32], seg1: &[i32], two: bool) -> Result<(u16, Vec<u16>), Refused> {
    let f: Vec<i32> = seg0
        .iter()
        .chain(if two { seg1 } else { seg0 })
        .copied()
        .collect();
    let base = *f.iter().min().unwrap();
    let off: Vec<u16> = f.iter().map(|v| (v - base) as u16).collect();
    if let Some(m) = off.iter().max().filter(|m| **m > 15) {
        return refuse(format!("MXFP4 tile's scales span {m} (> 15)"));
    }
    Ok((base as u16, off))
}

/// The placement (moe_pack.pack_moe).
pub fn pack_moe(m: &MoeLayer, g: MoeGrid) -> Result<MoeTiles, Refused> {
    let (ne, hr, inter) = (m.experts, m.hidden, m.inter);
    let (px_n, py_n, hc, ic) = (g.px, g.py, g.hc, g.ic);
    let (g2, gn3, dn3, n4g, n4d, xw) = (g.g2(), g.gn3(), g.dn3(), g.n4g(), g.n4d(), g.xw());
    let (ner, nec) = (g.ner(ne), g.nec(ne));
    let (hp, ip) = (py_n * hc, px_n * ic);
    if hp < hr || ip < inter {
        return refuse(format!(
            "MoE grid {px_n} x {py_n} (Hc {hc}, Ic {ic}) under {hr} x {inter}"
        ));
    }
    let (nbh, nbi) = (hp.div_ceil(32), ip.div_ceil(32));
    let (kg, kd) = (m.hidden_p / 32, m.inter_p / 32);
    // The padded views: gate/up row `r` of the [gate IP | up IP] layout, hidden `j`.
    let gu_row = |r: usize| -> Option<usize> {
        if r < inter {
            Some(r)
        } else if (ip..ip + inter).contains(&r) {
            Some(inter + r - ip)
        } else {
            None
        }
    };
    let gu_code = |e: usize, r: usize, j: usize| -> u16 {
        match gu_row(r) {
            Some(q) if j < hr => m.gu_codes[(e * 2 * inter + q) * m.hidden_p + j] as u16,
            _ => 0,
        }
    };
    let gu_scale = |e: usize, r: usize, b: usize| -> u16 {
        match gu_row(r) {
            Some(q) if b < kg => m.gu_scales[(e * 2 * inter + q) * kg + b] as u16,
            _ => 127,
        }
    };
    let dn_code = |e: usize, h: usize, i: usize| -> u16 {
        if h < hr && i < inter {
            m.dn_codes[(e * hr + h) * m.inter_p + i] as u16
        } else {
            0
        }
    };
    let dn_scale = |e: usize, h: usize, b: usize| -> u16 {
        if h < hr && b < kd {
            m.dn_scales[(e * hr + h) * kd + b] as u16
        } else {
            127
        }
    };
    let words = ne * xw + (ne * xw) % 2;
    let pes = px_n * py_n;
    let mut mx16 = vec![0u16; pes * words];
    let mut t = MoeTiles {
        mx: Vec::new(),
        rw: vec![0.0; pes * hc * ne],
        rb: Vec::with_capacity(pes * ne),
        gub: vec![0.0; pes * ner * g2],
        dnb: vec![0.0; pes * nec * hc],
        mcfg: vec![0; pes * 2],
    };
    for py in 0..py_n {
        let h0 = hc * py;
        let gs = hc.min(32 - h0 % 32);
        for px in 0..px_n {
            let pe = py * px_n + px;
            let i0 = ic * px;
            let ds = ic.min(32 - i0 % 32);
            t.mcfg[2 * pe] = gs as u32;
            t.mcfg[2 * pe + 1] = ds as u32;
            let rows: Vec<usize> = (i0..i0 + ic).chain(ip + i0..ip + i0 + ic).collect();
            let mx = &mut mx16[pe * words..(pe + 1) * words];
            for e in 0..ne {
                let mut o = e * xw;
                // gate/up [i (hidden)][j (output)]
                let tile: Vec<u16> = (0..hc)
                    .flat_map(|i| rows.iter().map(move |&r| (i, r)))
                    .map(|(i, r)| gu_code(e, r, h0 + i))
                    .collect();
                mx[o..o + gn3].copy_from_slice(&plane_pack(&tile, hc, g2));
                o += gn3;
                // down [i (inter)][j (hidden)]
                let tile: Vec<u16> = (0..ic)
                    .flat_map(|i| (0..hc).map(move |j| (i, j)))
                    .map(|(i, j)| dn_code(e, h0 + j, i0 + i))
                    .collect();
                mx[o..o + dn3].copy_from_slice(&plane_pack(&tile, ic, hc));
                o += dn3;
                let b0 = h0 / 32;
                let s0 = rows
                    .iter()
                    .map(|&r| fieldb(gu_scale(e, r, b0)))
                    .collect::<Result<Vec<_>, _>>()?;
                let s1 = rows
                    .iter()
                    .map(|&r| fieldb(gu_scale(e, r, (b0 + 1).min(nbh - 1))))
                    .collect::<Result<Vec<_>, _>>()?;
                let (gb, go) = offsets(&s0, &s1, gs < hc)?;
                let b0 = i0 / 32;
                let s0 = (h0..h0 + hc)
                    .map(|h| fieldb(dn_scale(e, h, b0)))
                    .collect::<Result<Vec<_>, _>>()?;
                let s1 = (h0..h0 + hc)
                    .map(|h| fieldb(dn_scale(e, h, (b0 + 1).min(nbi - 1))))
                    .collect::<Result<Vec<_>, _>>()?;
                let (db, dof) = offsets(&s0, &s1, ds < ic)?;
                mx[o] = gb | (db << 8);
                mx[o + 1..o + 1 + n4g].copy_from_slice(&nibble_pack(&go));
                mx[o + 1 + n4g..o + 1 + n4g + n4d].copy_from_slice(&nibble_pack(&dof));
            }
            for i in 0..hc {
                for e in 0..ne {
                    let h = h0 + i;
                    t.rw[(pe * hc + i) * ne + e] = if h < hr { m.router[e * hr + h] } else { 0.0 };
                }
            }
            t.rb.extend_from_slice(&m.router_bias[..ne]);
            for k2 in 0..ner {
                let e = py + k2 * py_n;
                if e < ne {
                    for (j, &r) in rows.iter().enumerate() {
                        t.gub[(pe * ner + k2) * g2 + j] =
                            gu_row(r).map_or(0.0, |q| m.gu_bias[e * 2 * inter + q]);
                    }
                }
            }
            for k2 in 0..nec {
                let e = px + k2 * px_n;
                if e < ne {
                    for i in 0..hc {
                        let h = h0 + i;
                        t.dnb[(pe * nec + k2) * hc + i] =
                            if h < hr { m.dn_bias[e * hr + h] } else { 0.0 };
                    }
                }
            }
        }
    }
    t.mx = mx16
        .chunks(2)
        .map(|w| w[0] as u32 | (w[1] as u32) << 16)
        .collect();
    Ok(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fnv(bytes: impl Iterator<Item = u8>) -> u64 {
        bytes.fold(0xcbf29ce484222325u64, |h, b| {
            (h ^ b as u64).wrapping_mul(0x100000001b3)
        })
    }

    /// The same small layer as `payload/moe_mxfp4/moe_pack_fixture.py`, whose
    /// numpy placement hashes to the pinned values.
    #[test]
    fn the_placement_matches_moe_pack() {
        let (ne, hr, inter, hrp, ipd) = (5usize, 40usize, 20usize, 64usize, 32usize);
        let gu_codes: Vec<u8> = (0..ne * 2 * inter * hrp)
            .map(|k| {
                let (e, r, c) = (k / (2 * inter * hrp), k / hrp % (2 * inter), k % hrp);
                ((e * 7 + r * 3 + c * 5) % 16) as u8
            })
            .collect();
        let gu_scales: Vec<u8> = (0..ne * 2 * inter * 2)
            .map(|k| {
                let (e, r, b) = (k / (4 * inter), k / 2 % (2 * inter), k % 2);
                (118 + (e + r + b) % 10) as u8
            })
            .collect();
        let gu_bias: Vec<f32> = (0..ne * 2 * inter).map(|k| k as f32 * 0.01 - 1.0).collect();
        let dn_codes: Vec<u8> = (0..ne * hr * ipd)
            .map(|k| {
                let (e, r, c) = (k / (hr * ipd), k / ipd % hr, k % ipd);
                ((e * 3 + r * 5 + c * 7) % 16) as u8
            })
            .collect();
        let dn_scales: Vec<u8> = (0..ne * hr)
            .map(|k| (120 + (2 * (k / hr) + k % hr) % 7) as u8)
            .collect();
        let dn_bias: Vec<f32> = (0..ne * hr).map(|k| k as f32 * 0.02 - 2.0).collect();
        let router: Vec<f32> = (0..ne * hr)
            .map(|k| (((k / hr) * 13 + (k % hr) * 7) % 17) as f32 * 0.125 - 1.0)
            .collect();
        let router_bias: Vec<f32> = (0..ne).map(|e| e as f32 * 0.5 - 1.0).collect();
        let m = MoeLayer {
            experts: ne,
            hidden: hr,
            inter,
            gu_codes: &gu_codes,
            hidden_p: hrp,
            gu_scales: &gu_scales,
            gu_bias: &gu_bias,
            dn_codes: &dn_codes,
            inter_p: ipd,
            dn_scales: &dn_scales,
            dn_bias: &dn_bias,
            router: &router,
            router_bias: &router_bias,
        };
        let g = MoeGrid {
            px: 3,
            py: 3,
            hc: 14,
            ic: 7,
        };
        let t = pack_moe(&m, g).unwrap();
        let u = |v: &[u32]| fnv(v.iter().flat_map(|x| x.to_le_bytes()));
        let f = |v: &[f32]| fnv(v.iter().flat_map(|x| x.to_le_bytes()));
        assert_eq!(t.mx.len(), 9 * g.mx_words(ne));
        let got = [
            u(&t.mx),
            f(&t.rw),
            f(&t.rb),
            f(&t.gub),
            f(&t.dnb),
            u(&t.mcfg),
        ];
        let want = PINNED;
        assert_eq!(got, want, "{got:x?}");
    }

    const PINNED: [u64; 6] = [
        0x371267960a1ac9,
        0x71cea07000948d7b,
        0xcad0c4ef5c8df615,
        0x4700b3bbf622391b,
        0x6a73756d922128c2,
        0xe36da692aa5f3c96,
    ];

    #[test]
    fn a_tile_whose_scales_span_over_15_is_refused() {
        assert!(offsets(&[100, 116], &[], false).is_err());
        assert_eq!(
            offsets(&[100, 115], &[], false).unwrap(),
            (100, vec![0, 15, 0, 15])
        );
    }

    #[test]
    fn nibbles_and_planes_pack_as_moe_pack() {
        assert_eq!(nibble_pack(&[1, 2, 3, 4, 5]), vec![0x0531, 0x0042]);
        // L 4 x R 1: plane 0 rows 0..2, plane 1 rows 2..4.
        assert_eq!(
            plane_pack(&[0xf, 0x1, 0x8, 0x7], 4, 1),
            vec![(1 << 15) | (7 << 6) | (1 << 12), 0x40 | (7 << 3)]
        );
    }
}
