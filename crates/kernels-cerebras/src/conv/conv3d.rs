//! A convolution in the tiled regime (`tiled_pe.csl`'s `conv`): its weights streamed in blocks
//! off the tape, each PE's pixels multiplied a block at a time, a 3 x 3 conv's input staged a
//! chunk of channels at a time with its halo exchanged.
use crate::compile::stream::tiled::{TOp, TileProgram};
use crate::kernel::{Kernel, Source, TiledForm};
use crate::tuning;

/// A convolution.
pub struct Conv3d;

/// The registry's convolution.
pub static CONV3D: Conv3d = Conv3d;

/// Op `i`'s cycles: the weights' stream, the products, the input's staging, the bias and the
/// rounding.
fn price(p: &TileProgram, i: usize) -> f64 {
    let TOp::Conv {
        x,
        y,
        k,
        cin,
        cout,
        bias,
        ..
    } = &p.ops[i]
    else {
        return 0.0;
    };
    let l = p.lay_of(*y);
    let px = f64::from(l.px());
    let taps = f64::from(k * k);
    let (cin_, cout_) = (f64::from(*cin), f64::from(*cout));
    let weights = taps * cin_ * cout_;
    let kb = f64::from(p.kb[i].max(1));
    let blocks = taps * cin_ / kb;
    let recv = weights / 2.0 * tuning::get().tiled_stream_word + blocks * tuning::get().tiled_call;
    let mac = px * weights * tuning::get().stream_p1 + blocks * px * tuning::get().tiled_call;
    let lx = p.lay_of(*x);
    // the input staged a chunk of kc channels at a time, its halo exchanged
    let kc = f64::from(p.kc[i].max(1));
    let chunks = cin_ / kc;
    let stage = if *k == 3 {
        let (th, tw) = (f64::from(lx.th), f64::from(lx.tw));
        let fill = (th + 2.0) * (tw + 2.0) * kc * tuning::get().stream_dsd_elem;
        // two 16-bit values a wavelet
        let ew = 4.0 * (th * kc / 2.0 * tuning::get().tiled_word + th * tuning::get().tiled_call)
            + 4.0 * f64::from(1u32 << lx.lsx) * tuning::get().tiled_hop;
        let ns = 4.0
            * ((tw + 2.0) * kc / 2.0 * tuning::get().tiled_word + tuning::get().tiled_call)
            + 4.0 * f64::from(1u32 << lx.lsy) * tuning::get().tiled_hop;
        chunks * (fill + ew + ns)
    } else {
        px * cin_ * tuning::get().stream_dsd_elem
    };
    let tail = if bias.is_some() {
        cout_ * tuning::get().tiled_stream_word + px * cout_ * tuning::get().stream_dsd_elem
    } else {
        0.0
    } + if p.vals[y].round {
        2.0 * px * cout_ * tuning::get().stream_dsd_elem
    } else {
        0.0
    };
    recv + mac + stage + tail
}

static TILED: TiledForm = TiledForm {
    what: "conv",
    price,
    hub: None,
};

impl Kernel for Conv3d {
    fn name(&self) -> &'static str {
        "conv.conv3d"
    }

    fn ops(&self) -> &'static [&'static str] {
        &["conv.conv3d"]
    }

    fn sources(&self) -> &'static [Source] {
        &[]
    }

    fn tiled(&self) -> Option<&'static TiledForm> {
        Some(&TILED)
    }
}
