//! A group norm in the tiled regime (`tiled_pe.csl`'s `norm`): each group's sums all-reduced
//! over the image's PEs through the hub, the result back down its broadcast tree.

use super::super::{Kernel, Source, TiledForm};
use crate::stream::tiled::{TOp, TileProgram};

/// A group norm.
pub struct GroupNorm;

/// The registry's group norm.
pub static GROUP_NORM: GroupNorm = GroupNorm;

/// Its tiled form: its two all-reduces send a word a group towards the hub before their
/// receive.
static TILED: TiledForm = TiledForm {
    func: "allreduce",
    what: "a norm's all-reduce",
    words: |_: &TileProgram, op: &TOp| match op {
        TOp::Norm { groups, .. } => *groups,
        _ => 0,
    },
};

impl Kernel for GroupNorm {
    fn name(&self) -> &'static str {
        "conv.group_norm"
    }

    fn ops(&self) -> &'static [&'static str] {
        &["conv.group_norm"]
    }

    fn sources(&self) -> &'static [Source] {
        &[]
    }

    fn tiled(&self) -> Option<&'static TiledForm> {
        Some(&TILED)
    }
}
