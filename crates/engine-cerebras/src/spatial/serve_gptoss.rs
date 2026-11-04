//! The serve_gptoss wafer program (`spatial/csl/serve_gptoss`): gpt-oss's
//! layers as units in slots, and its host side. Per layer an attention unit
//! (gptoss_attn, `PX` x `AY`, `AHc` dims a row) over a MoE unit (gptoss_moe,
//! `PX` x `MY`, `MHc` dims a row) in a slot of `GW` x `GH`; the head and the
//! embedding strip east of the slots.
//!
//! Layer order ([`Slots::units`]): slot row 0 from c = GW - 1 west to 0,
//! then the rows below serpentine by columns (column 0 down, 1 up, ...).
//! The handoffs are serve8b_region's pieces ([`Tables`]): attention -> MoE
//! is re-chunk A down the slot; MoE -> the next attention unit chains a dest
//! row's pieces in a MoE column and legs them down, up, east or west to the
//! row's assembling PE, which multicasts the row. This is the bare host's
//! `payload/serve_gptoss/gptoss_host.Tables`, word for word.

use super::serve8b::Kernel;
use super::serve8b_region::{Kind, Piece, Tables, Unit};

type R<T> = std::result::Result<T, String>;

const FWD: u32 = 0xffff;
const D_RAMP: u32 = 0;
const D_E: u32 = 3;
/// Row multicasts by the receiving unit's parity.
const C_R: [u32; 2] = [11, 12];
/// paths.csl: S (the last unit's chunk triples -> head), E (strip -> layer 0).
const C_S: u32 = 20;
const C_E: u32 = 13;
/// MoE -> attention by the dest layer's parity: legs, hops.
const M_LEG: [u32; 2] = [4, 17];
const M_HOP: [[u32; 2]; 2] = [[14, 15], [18, 19]];

/// serve_gptoss's slots and unit shapes (its layout params).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slots {
    pub px: usize,
    pub ay: usize,
    pub ahc: usize,
    pub my: usize,
    pub mhc: usize,
    pub gw: usize,
    pub gh: usize,
    pub nl: usize,
}

impl Slots {
    /// The slots (c, d) in layer order.
    pub fn order(&self) -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = (0..self.gw).rev().map(|c| (c, 0)).collect();
        for c in 0..self.gw {
            if c % 2 == 0 {
                out.extend((1..self.gh).map(|d| (c, d)));
            } else {
                out.extend((1..self.gh).rev().map(|d| (c, d)));
            }
        }
        out.truncate(self.nl);
        out
    }

    /// The units in layer order: layer l's attention unit (idx 2 l), then its
    /// MoE unit (2 l + 1).
    pub fn units(&self) -> Vec<Unit> {
        let sy = self.ay + self.my;
        let kernel = |py, hc| Kernel {
            px: self.px,
            py,
            hc,
            dc: 0,
            gs: 0,
            ic: 0,
        };
        let mut out = Vec::new();
        for (l, (c, d)) in self.order().into_iter().enumerate() {
            for (kind, y0, k) in [
                (Kind::Attn, d * sy, kernel(self.ay, self.ahc)),
                (Kind::Moe, d * sy + self.ay, kernel(self.my, self.mhc)),
            ] {
                out.push(Unit {
                    kind,
                    idx: out.len(),
                    x0: c * self.px,
                    y0,
                    k,
                    layers: l..l + 1,
                });
            }
        }
        out
    }
}

/// Every unit PE's handoff table (`rtab`, [`Tables::table`]): `rtr` routes
/// and `rti` pieces in a PE.
pub fn tables(s: &Slots, lanes: usize, rtr: usize, rti: usize) -> R<Tables> {
    let units = s.units();
    let mut t = Tables::empty(units.clone(), lanes, rtr);
    t.rti = rti;
    let (u0, ul) = (&units[0], &units[units.len() - 1]);
    // E: layer 0's row py from the east.
    for py in 0..u0.k.py {
        t.row_mcast(C_E, u0, py, u0.x0 + u0.k.px - 1, D_E)?;
        for x in u0.x0..u0.x0 + u0.k.px {
            t.recv(x, u0.y0 + py, C_E, 0, u0.k.hc as u32)?;
        }
    }
    // S: the last unit's rows 3k..3k+2 chained down its east column, east
    // from row 3k + 2.
    let x = ul.x0 + ul.k.px - 1;
    let hop = M_HOP[0];
    let hc = ul.k.hc as u32;
    for k in 0..ul.k.py / 3 {
        for i in 0..3 {
            let y = ul.y0 + 3 * k + i;
            let color = if i == 2 { C_S } else { hop[i % 2] };
            if i > 0 {
                t.set_fwd_in(x, y, hop[(i - 1) % 2])?;
                t.send(x, y, color, FWD, t.lanes * hc * i as u32)?;
            }
            t.send(x, y, color, 0, hc)?;
            if i < 2 {
                t.leg(color, &[(x, y), (x, y + 1)])?;
            } else {
                t.route(x, y, C_S, D_RAMP, D_E, D_E)?;
            }
        }
    }
    for w in units.windows(2) {
        handoff(&mut t, &w[0], &w[1])?;
    }
    Ok(t)
}

fn handoff(t: &mut Tables, a: &Unit, b: &Unit) -> R<()> {
    let r = C_R[b.idx % 2];
    if a.kind == Kind::Attn {
        if b.x0 != a.x0 || b.y0 != a.y0 + a.k.py {
            return Err(format!("unit {}: its MoE unit is not below it", a.idx));
        }
        return t.rechunk_a(a, b, r);
    }
    let p = b.layers.start % 2;
    let (leg, hop) = (M_LEG[p], M_HOP[p]);
    for g in 0..b.k.py {
        let pieces = Tables::pieces(a, b, g);
        if pieces.is_empty() {
            continue;
        }
        let srcs: Vec<Piece> = pieces
            .iter()
            .map(|&(row, so, d, n)| (a.y0 + row, so, d, n))
            .collect();
        let (top, bottom) = (srcs[0].0, srcs[srcs.len() - 1].0);
        let yg = b.y0 + g;
        let dest = if b.x0 == a.x0 && b.y0 == a.y0 + a.k.py {
            let col = a.x0 + g;
            let dest = (col, yg);
            t.chain(col, &srcs, hop, leg, &[(col, bottom), dest], dest, true)?;
            dest
        } else if b.x0 == a.x0 && b.y0 < a.y0 {
            let col = a.x0 + g;
            let dest = (col, yg);
            t.chain(col, &srcs, hop, leg, &[(col, top), dest], dest, false)?;
            dest
        } else if b.x0 == a.x0 + a.k.px && b.y0 + b.k.py == a.y0 {
            let col = a.x0 + a.k.px - b.k.py + g;
            let dest = (b.x0, yg);
            t.chain(col, &srcs, hop, leg, &[(col, top), (col, yg), dest], dest, false)?;
            dest
        } else if b.x0 + a.k.px == a.x0 && b.y0 + b.k.py == a.y0 {
            let col = a.x0 + b.k.py - 1 - g;
            let dest = (b.x0 + b.k.px - 1, yg);
            t.chain(col, &srcs, hop, leg, &[(col, top), (col, yg), dest], dest, false)?;
            dest
        } else {
            return Err(format!(
                "no handoff from unit {} at {:?} to unit {} at {:?}",
                a.idx,
                (a.x0, a.y0),
                b.idx,
                (b.x0, b.y0)
            ));
        };
        t.set_relay(dest.0, dest.1, r)?;
        t.row_mcast(r, b, g, dest.0, D_RAMP)?;
        for x in b.x0..b.x0 + b.k.px {
            if x != dest.0 {
                t.recv(x, yg, r, 0, b.k.hc as u32)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fnv(words: impl Iterator<Item = u32>) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for w in words {
            for b in w.to_le_bytes() {
                h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
            }
        }
        h
    }

    fn hash(s: &Slots) -> (usize, u64) {
        let t = tables(s, 1, 6, 6).expect("tables");
        let mut all = Vec::new();
        for u in s.units() {
            all.extend(t.table(&u).expect("a unit's table"));
        }
        (all.len(), fnv(all.into_iter()))
    }

    /// The tables equal the bare host's (gptoss_host.Tables, FNV-1a 64 of
    /// every unit's rtab words in unit order), at the simulator's mini shape
    /// and at full size (the program CS-3 ran in 0447).
    #[test]
    fn tables_match_the_bare_host() {
        let mini = Slots {
            px: 8,
            ay: 5,
            ahc: 13,
            my: 18,
            mhc: 4,
            gw: 3,
            gh: 3,
            nl: 9,
        };
        assert_eq!(hash(&mini), (96048, 0x9b14c126a5c119fd));
        let full = Slots {
            px: 128,
            ay: 42,
            ahc: 69,
            my: 192,
            mhc: 15,
            gw: 5,
            gh: 5,
            nl: 24,
        };
        assert_eq!(hash(&full), (41693184, 0xdf3663bd368add1b));
    }
}
