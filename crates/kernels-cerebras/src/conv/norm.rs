//! A group norm in the tiled regime (`tiled_pe.csl`'s `norm`): each group's sums all-reduced
//! over the image's PEs through the hub, the result back down its broadcast tree.
use crate::compile::cost::tiled::{all_reduce, region};
use crate::compile::stream::tiled::{TOp, TileProgram};
use crate::kernel::{HubSend, Kernel, Source, TiledForm};
use crate::tuning;

/// A group norm.
pub struct GroupNorm;

/// The registry's group norm.
pub static GROUP_NORM: GroupNorm = GroupNorm;

/// Op `i`'s cycles: each PE's sums, the two all-reduces, the scale and shift streamed in, its
/// region.
fn price(p: &TileProgram, i: usize) -> f64 {
    let TOp::Norm { x, groups, reg, .. } = &p.ops[i] else {
        return 0.0;
    };
    let l = p.lay_of(*x);
    let c = f64::from(p.vals[x].width);
    let px = f64::from(l.px());
    let local =
        px * c * 4.0 * tuning::get().stream_dsd_elem + 2.0 * c * tuning::get().tiled_scalar / 4.0;
    let ar = 2.0 * all_reduce(p, *groups);
    let recv = 2.0 * c * tuning::get().tiled_stream_word + c * tuning::get().tiled_scalar;
    local + ar + recv + region(p, *reg, l.px())
}

/// Its tiled form: its two all-reduces send a word a group towards the hub before their
/// receive.
static TILED: TiledForm = TiledForm {
    what: "norm",
    price,
    hub: Some(HubSend {
        func: "allreduce",
        what: "a norm's all-reduce",
        words: |_: &TileProgram, op: &TOp| match op {
            TOp::Norm { groups, .. } => *groups,
            _ => 0,
        },
    }),
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
