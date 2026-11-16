//! The device's rates on wgpu ([`engine_gpu::rates::measure`]): the
//! matmul and the copy timed by the adapter's timestamps (the frame's span)
//! where it grants them, else by the host's clock around submit and wait.
//! A browser's load calls it (its adapter is masked, so no profile states
//! its rates); so does `pie tune wgpu --rates`.

use engine_gpu::error::Result;
use profile::fire::Rates;

use crate::device::{Api, Context, Frame};

/// The rates of `device`, under the tactics `tuning` picks.
pub fn measure(device: &Context, tuning: &crate::tuning::Tuning) -> Result<Rates> {
    measure_saying(device, tuning, &mut |_| {})
}

/// [`measure`], saying how the warm-up ended.
pub fn measure_saying(
    device: &Context,
    tuning: &crate::tuning::Tuning,
    say: &mut dyn FnMut(&str),
) -> Result<Rates> {
    let tier = <Api as engine_gpu::serve::Serve>::tier(device);
    let timer = |frame: Frame| {
        crate::encode::reset_kernel_profile();
        let host = frame.commit_timed()?;
        let span = crate::encode::kernel_profile()
            .into_iter()
            .find(|(name, ..)| name == "frame.span")
            .map(|(_, ns, _)| ns)
            .filter(|&ns| ns > 0);
        Ok((span, (host * 1e9) as u64))
    };
    engine_gpu::rates::measure_saying::<Api, _>(device, tuning, tier, &timer, say)
}
