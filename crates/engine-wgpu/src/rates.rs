//! A device's rates measured on the device itself: a few micro-fires on
//! synthetic buffers, before any model is compiled. The browser's adapter
//! is masked, so no profile can state its rates (`[rates]`); its load
//! measures them here, for that load alone, and writes them nowhere.
//!
//! Three fires, each timed a few times and the median kept (the first,
//! which builds its pipeline, is not):
//! - a dense bf16 matmul of [`GEMM`] rows, columns and depth, for flops a
//!   nanosecond;
//! - a buffer copy of [`COPY_BYTES`], for bytes a nanosecond;
//! - the narrowest matmul alone in a frame, for what a fire's launch costs
//!   beside its work.
//!
//! The matmul and the copy are timed by the adapter's timestamps where it
//! grants them, else by the host's clock around submit and wait; the launch
//! is always the host's (it is the submit and wait).
//!
//! Its shape is the Vulkan kernel bench's (`engine_vulkan::tune::qmm`), so
//! `pie tune` can state the same rates natively from this function.

use engine_gpu::backend::Tactics as _;
use engine_gpu::error::{Fault, Result};
use engine_gpu::{Handles, Owner};
use kernels_slang::Tensor;
use poem_ir::Dtype;
use profile::fire::Rates;

use crate::device::{Api, Buffer, Context, Pipelines};
use crate::encode::Sink;

/// The matmul's rows, columns and depth. At 4096 a side its arithmetic
/// intensity is 4096 / 3 ≈ 1365 flops a byte of its operands, past the
/// balance point of any device it runs on (a few hundred flops a byte at
/// most), so it is bound by compute, not memory; and its 137 GFLOP keep the
/// device busy for milliseconds even at hundreds of TFLOP/s, past launch
/// noise. Its three planes take 32 MiB each, within the binding WebGPU
/// grants by default (128 MiB).
const GEMM: u32 = 4096;

/// The copy's bytes, at most: 256 MiB read and written is past every last
/// level cache a GPU carries (tens of MiB to ~100 MiB), so it moves device
/// memory, and takes long enough to dwarf a dispatch. Bounded by the
/// largest buffer the device allocates.
const COPY_BYTES: u64 = 256 << 20;

/// The matmul a launch is timed with: its work is nothing beside a launch.
const LAUNCH: u32 = 16;

/// Timed fires of each kind; the median is kept.
const ROUNDS: usize = 5;

/// What the micro-fires measure.
const OWNER: Owner = engine_gpu::books::owners::TUNING;

/// The rates of `device`, under the matmul tactic `tuning` picks.
pub fn measure(device: &Context, tuning: &crate::tuning::Tuning) -> Result<Rates> {
    let tier = <Api as engine_gpu::serve::Serve>::tier(device);
    let tactic = tuning.gemm(tier);
    let handles = Handles::<Api>::new();
    let pipelines = Pipelines::new();
    let plane = u64::from(GEMM) * u64::from(GEMM) * 2;
    let bind = |buffer: &Buffer| handles.bind(buffer, 0, buffer.bytes());
    let (act, w, y) = (
        Buffer::zeroed(device, plane, OWNER)?,
        Buffer::zeroed(device, plane, OWNER)?,
        Buffer::zeroed(device, plane, OWNER)?,
    );
    let (ha, hw, hy) = (bind(&act)?, bind(&w)?, bind(&y)?);
    let gemm = |sink: &Sink<'_>, side: u32| {
        kernels_slang::linear::gemm::matmul(
            sink,
            tactic,
            Tensor::new(ha, side, side, Dtype::Bf16),
            Tensor::new(hw, side, side, Dtype::Bf16),
            Tensor::new(hy, side, side, Dtype::Bf16),
        )
        .map_err(|e| Fault::Device {
            call: "rates: matmul",
            why: e.to_string(),
        })
    };

    // A frame of one matmul of `side`, then `then` on the frame: its time
    // by the adapter's span where granted, and the host's.
    let frame = |side: u32, then: &dyn Fn(&mut crate::device::Frame) -> Result<()>| {
        let mut f = device.frame()?;
        {
            let sink = Sink::new(device, &f, &pipelines, &handles);
            gemm(&sink, side)?;
        }
        then(&mut f)?;
        crate::encode::reset_kernel_profile();
        let host = f.commit_timed()?;
        let span = crate::encode::kernel_profile()
            .into_iter()
            .find(|(name, ..)| name == "frame.span")
            .map(|(_, ns, _)| ns);
        Ok::<(u64, u64), Fault>((span.unwrap_or(0), (host * 1e9) as u64))
    };
    let median = |mut v: Vec<u64>| {
        v.sort_unstable();
        v[v.len() / 2].max(1)
    };
    let timed = |side: u32, then: &dyn Fn(&mut crate::device::Frame) -> Result<()>| {
        frame(side, then)?;
        let mut adapter = Vec::new();
        let mut host = Vec::new();
        for _ in 0..ROUNDS {
            let (span, wall) = frame(side, then)?;
            adapter.push(span);
            host.push(wall);
        }
        let granted = adapter.iter().all(|&ns| ns > 0);
        Ok::<(u64, bool), Fault>((median(if granted { adapter } else { host }), granted))
    };
    let nothing = |_: &mut crate::device::Frame| Ok(());

    // The launch is the host's: submit, the dispatch, the wait.
    frame(LAUNCH, &nothing)?;
    let launch = median(
        (0..ROUNDS)
            .map(|_| frame(LAUNCH, &nothing).map(|(_, wall)| wall))
            .collect::<Result<Vec<u64>>>()?,
    );
    let (gemm_ns, _) = timed(GEMM, &nothing)?;

    let copy = COPY_BYTES.min(device.max_buffer()).max(4) & !3;
    let (from, into) = (
        Buffer::zeroed(device, copy, OWNER)?,
        Buffer::zeroed(device, copy, OWNER)?,
    );
    // The copy rides after the narrowest matmul, whose time is taken off.
    let (with_copy, granted) = timed(LAUNCH, &|f| f.copy(&from, 0, &into, 0, copy))?;
    let (alone, _) = timed(LAUNCH, &nothing)?;
    let copy_ns = if granted {
        with_copy.saturating_sub(alone)
    } else {
        with_copy.saturating_sub(launch)
    }
    .max(1);

    let flops = 2.0 * f64::from(GEMM).powi(3);
    Ok(Rates {
        bf16_flops_per_ns: Some(flops / gemm_ns as f64),
        // Every byte copied is read and written.
        bytes_per_ns: Some(2.0 * copy as f64 / copy_ns as f64),
        launch_ns: Some(launch),
    })
}
