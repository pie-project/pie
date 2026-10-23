mod conv3d;
mod group_norm;
mod resample;
mod rule;

use crate::gpu::{Fires, Gpu, Out, close};

/// Run `slang` on the outputs `fresh` makes and compare each landed output
/// with its reference: the conv group has no hand-written WGSL kernel to set
/// beside it.
pub(crate) fn slang_alone(
    gpu: &mut Gpu,
    what: &str,
    fresh: &dyn Fn(&mut Gpu) -> Vec<Out>,
    want: &[Vec<f64>],
    slang: Fires<'_>,
) {
    let outs = fresh(gpu);
    slang(gpu, &outs);
    for ((buf, n, f32), want) in outs.iter().zip(want) {
        let got = if *f32 {
            gpu.down_f32(*buf, *n)
        } else {
            gpu.down(*buf, *n)
        };
        close(&got, want, &format!("slang {what}"));
    }
}
