//! Segments: short reduction chains inside a unit row, each rooted at its middle and
//! multicasting back (no loop-back). The attention scores' segments run along every weight row
//! (one KV head's columns a segment); the o-norm's along the CY row (one value head's columns).

use crate::spatial::floor::{Coord, Floor};
use crate::spatial::plan::Plan;
use crate::spatial::wiring::{Class, Slot, Wiring, segments};

use super::Block;

type R<T> = Result<T, String>;

/// Weight rows' scores segments: level-1 halves, multicast.
pub const SCORE_L1: [Class; 2] = [Class::new("SA"), Class::new("SB")];
pub const SCORE_CAST: Class = Class::new("SM");
/// The CY row's o-norm segments.
pub const ONORM_L1: [Class; 2] = [Class::new("CS"), Class::new("CS2")];
pub const ONORM_CAST: Class = Class::new("CM");

/// The scores segment a weight PE is in.
pub const SEGMENT: Slot = Slot::segment("segment");
/// The o-norm segment a CY PE is in.
pub const ONORM: Slot = Slot::segment("onorm");

pub const COLOURS: &[(Class, u8)] = &[
    (SCORE_L1[0], 8),
    (SCORE_L1[1], 13),
    (SCORE_CAST, 9),
    (ONORM_L1[0], 11),
    (ONORM_L1[1], 17),
    (ONORM_CAST, 12),
];

/// Every weight row of every unit cut into scores segments of `width` columns (single-PE
/// segments too).
pub struct Scores {
    pub width: u32,
}

impl Block for Scores {
    fn wire(&self, _plan: &Plan, floor: &Floor, w: &mut Wiring) -> R<()> {
        for u in &floor.units {
            for r in 0..u.rows() {
                let pts: Vec<Coord> = (0..u.cols())
                    .map(|c| Coord::new(u.col_x(c), u.row_y(r)))
                    .collect();
                segments(&pts, self.width, true, (SCORE_L1, SCORE_CAST, SEGMENT), w)?;
            }
        }
        Ok(())
    }

    fn colours(&self) -> Vec<(Class, u8)> {
        COLOURS.to_vec()
    }
}

/// Every unit's CY row cut into o-norm segments of `width` columns (single-PE segments left
/// out: their PE norms alone).
pub struct Onorm {
    pub width: u32,
}

impl Block for Onorm {
    fn wire(&self, _plan: &Plan, floor: &Floor, w: &mut Wiring) -> R<()> {
        for u in &floor.units {
            let pts: Vec<Coord> = (0..u.cols())
                .map(|c| Coord::new(u.col_x(c), u.cy_y()))
                .collect();
            segments(&pts, self.width, false, (ONORM_L1, ONORM_CAST, ONORM), w)?;
        }
        Ok(())
    }

    fn colours(&self) -> Vec<(Class, u8)> {
        COLOURS.to_vec()
    }
}
