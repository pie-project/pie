//! A device's rates measured on the device itself: a few micro-fires on
//! synthetic buffers, before any model is compiled. One function every
//! backend calls: a browser's load (its adapter is masked, so no profile
//! states its rates) and `pie tune --rates` natively.
//!
//! Three fires, each timed [`ROUNDS`] times and the median kept (a first,
//! which builds its pipeline, is not):
//! - a dense bf16 matmul of [`GEMM`] rows, columns and depth, through the
//!   backend's own matmul path, for flops a nanosecond;
//! - a buffer copy of [`COPY_BYTES`], for bytes a nanosecond;
//! - the narrowest matmul alone in a frame, for what a fire's launch costs
//!   beside its work.
//!
//! The backend times a frame ([`Timer`]): by its adapter's timestamps where
//! it has them, else by the host's clock around submit and wait. The launch
//! is always the host's (it is the submit and the wait).

use kernels_slang::Tensor;
use poem_ir::Dtype;
use profile::fire::Rates;

use crate::backend::{Buffer as _, Context as _, Frame as _};
use crate::error::{Fault, Result};
use crate::handles::Handles;
use crate::serve::Serve;

/// The matmul's rows, columns and depth. At 4096 a side its arithmetic
/// intensity is 4096 / 3 ≈ 1365 flops a byte of its operands, past the
/// balance point of any device it runs on (a few hundred flops a byte at
/// most), so it is bound by compute, not memory; and its 137 GFLOP keep the
/// device busy for milliseconds even at hundreds of TFLOP/s, past launch
/// noise. Its three planes take 32 MiB each, within the binding WebGPU
/// grants by default (128 MiB).
pub const GEMM: u32 = 4096;

/// The copy's bytes, at most: 256 MiB read and written is past every last
/// level cache a GPU carries (tens of MiB to ~100 MiB), so it moves device
/// memory, and it takes long enough to dwarf a dispatch. Bounded by the
/// largest buffer the device allocates.
pub const COPY_BYTES: u64 = 256 << 20;

/// The matmul a launch is timed with: its work is nothing beside a launch.
pub const LAUNCH: u32 = 16;

/// Timed fires of each kind; the median is kept.
pub const ROUNDS: usize = 5;

/// How a backend times a committed frame: `(adapter ns where it has
/// timestamps, host ns around the submit and the wait)`.
pub type Timer<'a, B> =
    &'a dyn Fn(<B as crate::backend::Backend>::Frame) -> Result<(Option<u64>, u64)>;

/// The rates of `device`, its matmuls under `tactic`.
pub fn measure<B: Serve>(
    device: &B::Context,
    tactic: kernels_slang::linear::gemm::Tactic,
    timer: Timer<'_, B>,
) -> Result<Rates> {
    let owner = crate::books::owners::TUNING;
    let handles = Handles::<B>::new();
    let pipelines = B::Pipelines::default();
    let plane = u64::from(GEMM) * u64::from(GEMM) * 2;
    let act = B::Buffer::zeroed(device, plane, owner)?;
    let w = B::Buffer::zeroed(device, plane, owner)?;
    let y = B::Buffer::zeroed(device, plane, owner)?;
    let ha = handles.bind(&act, 0, act.bytes())?;
    let hw = handles.bind(&w, 0, w.bytes())?;
    let hy = handles.bind(&y, 0, y.bytes())?;

    // A frame of one matmul of `side`, then `then` on the frame, timed.
    let frame = |side: u32, then: &dyn Fn(&mut B::Frame) -> Result<()>| {
        let mut f = device.frame()?;
        {
            let sink = B::sink(device, &f, &pipelines, &handles);
            kernels_slang::linear::gemm::matmul(
                &sink,
                tactic,
                Tensor::new(ha, side, side, Dtype::Bf16),
                Tensor::new(hw, side, side, Dtype::Bf16),
                Tensor::new(hy, side, side, Dtype::Bf16),
            )
            .map_err(|e| Fault::Device {
                call: "rates: matmul",
                why: e.to_string(),
            })?;
        }
        then(&mut f)?;
        timer(f)
    };
    let median = |mut v: Vec<u64>| {
        v.sort_unstable();
        v[v.len() / 2].max(1)
    };
    // The median by the adapter where it timed every round, else the host.
    let timed = |side: u32, then: &dyn Fn(&mut B::Frame) -> Result<()>| {
        frame(side, then)?;
        let mut adapter = Vec::with_capacity(ROUNDS);
        let mut host = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            let (span, wall) = frame(side, then)?;
            adapter.push(span);
            host.push(wall);
        }
        Ok::<(u64, bool), Fault>(match adapter.into_iter().collect::<Option<Vec<u64>>>() {
            Some(adapter) => (median(adapter), true),
            None => (median(host), false),
        })
    };
    let nothing = |_: &mut B::Frame| Ok(());

    // The launch is the host's: the submit, the dispatch, the wait.
    frame(LAUNCH, &nothing)?;
    let launch = median(
        (0..ROUNDS)
            .map(|_| frame(LAUNCH, &nothing).map(|(_, wall)| wall))
            .collect::<Result<Vec<u64>>>()?,
    );
    let (gemm_ns, _) = timed(GEMM, &nothing)?;

    let copy = COPY_BYTES.min(device.max_buffer()) & !3;
    let from = B::Buffer::zeroed(device, copy, owner)?;
    let into = B::Buffer::zeroed(device, copy, owner)?;
    // The copy rides after the narrowest matmul, whose time is taken off.
    let (with_copy, by_adapter) = timed(LAUNCH, &|f| f.copy(&from, 0, &into, 0, copy))?;
    let alone = if by_adapter {
        timed(LAUNCH, &nothing)?.0
    } else {
        launch
    };
    let copy_ns = with_copy.saturating_sub(alone).max(1);

    let flops = 2.0 * f64::from(GEMM).powi(3);
    Ok(Rates {
        bf16_flops_per_ns: Some(flops / gemm_ns as f64),
        // Every byte copied is read and written.
        bytes_per_ns: Some(2.0 * copy as f64 / copy_ns as f64),
        launch_ns: Some(launch),
    })
}
