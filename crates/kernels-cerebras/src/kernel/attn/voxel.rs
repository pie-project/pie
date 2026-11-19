//! Attention over an image's voxels in the tiled regime (`tiled_pe.csl`'s `attend`): each key
//! sent towards the hub along its row and column, every query's product, exponent and sum
//! where the key comes back off the broadcast.

use super::super::{Kernel, Source, TiledForm};
use crate::stream::tiled::{TOp, TileProgram};

/// A voxel attention.
pub struct Voxel;

/// The registry's voxel attention.
pub static VOXEL: Voxel = Voxel;

/// Its tiled form: a PE sends a key's words (`[k | v]` 16-bit, a word two) before its receive.
static TILED: TiledForm = TiledForm {
    func: "attend",
    what: "an attention's key",
    words: |p: &TileProgram, op: &TOp| match op {
        TOp::Attend { q, .. } => p.vals[q].width,
        _ => 0,
    },
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
