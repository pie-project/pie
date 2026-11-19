//! Attention over an image's voxels in the tiled regime (`tiled_pe.csl`'s `attend`): each key
//! sent towards the hub along its row and column, every query's product, exponent and sum
//! where the key comes back off the broadcast.

use super::super::{HubSend, Kernel, Source, TiledForm};
use crate::cost::stream::DSD_ELEM;
use crate::cost::tiled::{CALL, KEY, KEY_HOP, SCALAR, WORD};
use crate::stream::tiled::{TOp, TileProgram};

/// A voxel attention.
pub struct Voxel;

/// The registry's voxel attention.
pub static VOXEL: Voxel = Voxel;

/// Op `i`'s cycles: a key forwarded along its row and column, received off the broadcast (two
/// 16-bit values a wavelet), widened; each query's product, halving sum, exponent,
/// accumulation; every key's.
fn price(p: &TileProgram, i: usize) -> f64 {
    let TOp::Attend { q, .. } = &p.ops[i] else {
        return 0.0;
    };
    let v = &p.vals[q];
    let l = p.lay_of(*q);
    let c = f64::from(v.width);
    let keys = f64::from(v.grid[1] * v.grid[2]);
    let per = KEY
        + f64::from(p.gx + p.gy) * KEY_HOP
        + c * WORD
        + c * WORD
        + 2.0 * c * DSD_ELEM
        + f64::from(l.px())
            * (3.0 * c * DSD_ELEM + c.log2().ceil() * CALL + 3.0 * CALL + 4.0 * SCALAR);
    keys * per
}

/// Its tiled form: a PE sends a key's words (`[k | v]` 16-bit, a word two) before its receive.
static TILED: TiledForm = TiledForm {
    what: "attend",
    price,
    hub: Some(HubSend {
        func: "attend",
        what: "an attention's key",
        words: |p: &TileProgram, op: &TOp| match op {
            TOp::Attend { q, .. } => p.vals[q].width,
            _ => 0,
        },
    }),
};

impl Kernel for Voxel {
    fn name(&self) -> &'static str {
        "attn.voxel"
    }

    fn ops(&self) -> &'static [&'static str] {
        &["attn.voxel"]
    }

    fn sources(&self) -> &'static [Source] {
        &[]
    }

    fn tiled(&self) -> Option<&'static TiledForm> {
        Some(&TILED)
    }
}
