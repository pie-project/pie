//! An upsampling in the tiled regime (`tiled_pe.csl`'s `up`): a nearest upsampling, or a pixel
//! shuffle's, each PE's pixels copied to the finer grid and moved across the PEs a level apart.
use crate::compile::stream::tiled::{TOp, TileProgram};
use crate::kernel::{Kernel, Source, TiledForm};
use crate::tuning;

/// An upsampling.
pub struct Up;

/// The registry's upsampling.
pub static UP: Up = Up;

/// Op `i`'s cycles: the copy, the move across PEs.
fn price(p: &TileProgram, i: usize) -> f64 {
    let TOp::Up { x, y, axis, .. } = &p.ops[i] else {
        return 0.0;
    };
    let (lx, ly) = (p.lay_of(*x), p.lay_of(*y));
    let w = f64::from(p.vals[x].width);
    let copy = f64::from(ly.px()) * w * tuning::get().stream_dsd_elem;
    let ls = if *axis == 0 { lx.lsy } else { lx.lsx };
    let mv = if ls > 0 {
        f64::from(lx.px()) * w * tuning::get().tiled_word
            + f64::from(1u32 << (ls - 1)) * tuning::get().tiled_hop
            + tuning::get().tiled_call
    } else {
        0.0
    };
    copy + mv
}

static TILED: TiledForm = TiledForm {
    what: "up",
    price,
    hub: None,
};

impl Kernel for Up {
    fn name(&self) -> &'static str {
        "conv.upsample"
    }

    fn ops(&self) -> &'static [&'static str] {
        &["conv.upsample", "conv.pixel_shuffle"]
    }

    fn sources(&self) -> &'static [Source] {
        &[]
    }

    fn tiled(&self) -> Option<&'static TiledForm> {
        Some(&TILED)
    }
}
