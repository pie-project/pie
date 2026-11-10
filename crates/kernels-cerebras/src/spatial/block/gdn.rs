//! Qwen3.5's gated delta net: each GDN segment of a weight row (one key head's columns) shares
//! the (k, q) key dims its PEs hold, one multicast a row from the column that holds the row's
//! key dim. The weight kernel reads its part as a source word ([`Gdn::source`]).

use crate::spatial::floor::{Coord, Floor};
use crate::spatial::plan::{LayerKind, Plan};
use crate::spatial::wiring::{Class, Multicast, Slot, Wiring};

use super::Block;

type R<T> = Result<T, String>;

/// The (k, q) multicast along a GDN segment.
pub const GDN_KQ: Class = Class::new("SG");
/// The GDN multicast a weight PE sends or receives.
pub const GDN: Slot = Slot::segment("gdn");

pub const COLOURS: &[(Class, u8)] = &[(GDN_KQ, 10)];

/// The source word's codes (`weight_q35.csl`'s `gsrc`): a one-PE segment's own key dim, the
/// multicast's source (its key dim in the low bits), a receiver, a PE with no GDN row.
pub const SOURCE_OWN: u32 = 0xC000;
pub const SOURCE_SEND: u32 = 0x8000;
pub const SOURCE_RECV: u32 = 0;
pub const SOURCE_NONE: u32 = 0x4000;

/// The GDN block of a plan with GDN layers.
pub struct Gdn {
    /// Columns a GDN segment (`SWG`).
    pub width: u32,
    /// Weight columns a unit.
    pub cols: u32,
    /// Rows holding a key dim (`DKH`: the key head dim) and key dims a column (`DKG`).
    pub key_rows: u32,
    pub dims_per_col: u32,
}

impl Gdn {
    /// The plan's GDN block (`None`: no GDN layer).
    pub fn of(plan: &Plan) -> Option<Gdn> {
        let any = plan
            .shape
            .layers
            .iter()
            .any(|k| matches!(k, LayerKind::Gdn));
        let g = plan.shape.gdn.filter(|_| any)?;
        let width = plan.partition.segments.gdn;
        Some(Gdn {
            width,
            cols: plan.partition.cols,
            key_rows: g.key_dim,
            dims_per_col: g.key_dim / width,
        })
    }

    /// Weight row `r`, column `c`'s source word.
    pub fn source(&self, r: u32, c: u32) -> u32 {
        if r >= self.key_rows {
            return SOURCE_NONE;
        }
        let s0 = c / self.width * self.width;
        let len = self.width.min(self.cols - s0);
        let d = r % self.dims_per_col;
        if len == 1 {
            SOURCE_OWN | d
        } else if c - s0 == r / self.dims_per_col {
            SOURCE_SEND | d
        } else {
            SOURCE_RECV
        }
    }
}

impl Block for Gdn {
    fn wire(&self, _plan: &Plan, floor: &Floor, w: &mut Wiring) -> R<()> {
        for u in &floor.units {
            for r in 0..u.rows().min(self.key_rows) {
                let pts: Vec<Coord> = (0..u.cols())
                    .map(|c| Coord::new(u.col_x(c), u.row_y(r)))
                    .collect();
                for s in pts.chunks(self.width as usize).filter(|s| s.len() > 1) {
                    let src = (r / self.dims_per_col) as usize;
                    if src >= s.len() {
                        return Err(format!("gdn: row {r}'s source {src} past its segment"));
                    }
                    w.multicast(Multicast {
                        pts: s.to_vec(),
                        src,
                        class: GDN_KQ,
                        slot: GDN,
                    });
                }
            }
        }
        Ok(())
    }

    fn colours(&self) -> Vec<(Class, u8)> {
        COLOURS.to_vec()
    }
}
