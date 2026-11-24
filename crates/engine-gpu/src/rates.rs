//! A device's rates measured on the device itself: micro-fires on synthetic
//! buffers, before any model is compiled, at the shapes `engine::rates`
//! states for every backend. One function every slang backend calls: a
//! browser's load (its adapter is masked, so no profile states its rates)
//! and `pie tune --rates` natively.
//!
//! Each kind of work is timed inside a frame: `n` of it back to back
//! against one (`n` doubled until a fire's time settles,
//! `engine::rates::timed_frames`), the median of [`ROUNDS`] after a warm frame (which builds
//! its pipelines), so the frame's one submit and wait cancel and what is
//! left is the work's own time beside a dispatch's:
//! - the backend's own bf16 matmul `m x GEMM_WIDTH x GEMM_WIDTH` at each of
//!   `engine::rates::GEMM_ROWS`, each fire reading the next of
//!   `engine::rates::weights_at` weight planes (past the last-level cache),
//!   for `gemm_tflops_by_rows`;
//! - its prefill attention at `engine::rates::ATTN_SHAPE`'s heads over each
//!   of `engine::rates::ATTN_KEYS` keys at head width 128 (64 and 256 at the
//!   longest keys, said beside it), through the kernel a served fire selects
//!   at that shape, for `attn_tflops_by_keys`;
//! - its rmsnorm then its swiglu at `engine::rates::ELEMENTWISE_SHAPE`, each
//!   fire the next of `engine::rates::elementwise_sets` sets, for
//!   `elementwise_gbps`;
//! - its chunked gated-delta scan, the causal conv before it, at
//!   `engine::rates::SCAN_SHAPE`, for `scan_gelems`;
//! - a buffer copy of [`COPY_BYTES`], for `hbm_gbps`;
//! - a chain of dependent narrowest matmuls in one frame (each reading the
//!   one before's output) against one matmul of the chain's rows, at each
//!   length of `engine::rates::LAUNCH_CHAINS`, for what a dispatch adds to
//!   its work (`engine::rates::launch_from`).
//!
//! Once a model's shapes are known, each dense kernel the slang backends
//! choose between at each of its planes, at each power of two of rows to
//! the load's most (`kernel_rows`), for `kernel_tflops_by_shape`
//! ([`shapes_at_load`]): what a rung's dense kernel is chosen by, the
//! kernels' occupancy at the model's own planes in their time.
//!
//! Every float operand a rate reads holds `engine::rates::operand_pattern`
//! repeated (outputs, index tables and the scan's gates as laid), as every
//! backend's do.
//!
//! The backend times a frame ([`Timer`]): by its adapter's timestamps where
//! it has them, else by the host's clock around submit and wait. The
//! repeated fires are the host's clock's difference of two frames either
//! way: one submit each, so neither the clock nor the adapter's span start
//! biases it.
//!
//! Under a bound the device states on one submission, no fire goes out
//! without an estimate (`engine::rates::Admission` refuses one by name):
//! the probe (`crate::submit::Probe`, the smallest shape on the matmul
//! curve) is laid and timed first, and every shape's first fire after it is
//! estimated from the fires measured before ([`grown`]). Every frame the
//! rig lays follows the device idle (each waits for its own submissions,
//! and nothing else submits while a rates step runs), and a dropped clock
//! (an L40S idles at 210 MHz) runs a fire past an estimate taken loaded: so
//! each frame re-times the clock first, the probe fired alone and timed on
//! the device, and every estimate is read at its time over the probe's
//! first (measurements are kept at that first clock).

use engine::rates::{SCAN_SHAPE, Touched, attn_flops, gemm_flops, scan_elements, weights_at};
use kernels_slang::{KvPool, Tensor};
use poem_ir::Dtype;
use profile::fire::Rates;

use crate::backend::{Buffer as _, Context as _, Frame as _, Tactics};
use crate::error::{Fault, Result};
use crate::handles::Handles;
use crate::serve::Serve;

/// The copy's bytes, at most: 256 MiB read and written is past every last
/// level cache a GPU carries (tens of MiB to ~100 MiB), so it moves device
/// memory, and it takes long enough to dwarf a dispatch. Bounded by the
/// largest buffer the device allocates.
pub const COPY_BYTES: u64 = 256 << 20;

/// The bound the platform states on one GPU submission, which the rates
/// step holds each of its frames (and its warm-up's cube) within, each at
/// its estimate times the worst ratio of a frame's time to its estimate
/// seen (`engine_gpu::rates`). On Android, KGSL's (Qualcomm's msm GPU
/// driver, `qcom/opensource/graphics-kernel`, branch
/// gfx-kernel.lnx.16.0.r15-rel, the android16 / 6.12 line):
///
/// - The one duration the driver states is `adreno_drawobj_timeout = 2000`
///   ms (adreno_dispatch.c:51, "Command batch timeout (in milliseconds)"):
///   the ringbuffer dispatcher's per-submission expiry (adreno_dispatch.c:
///   2271, `drawqueue->expires = jiffies + msecs_to_jiffies(
///   adreno_drawobj_timeout)`) and the preemption timer
///   (adreno_gen8_preempt.c:55). Its fault-detect interval,
///   `_fault_timer_interval = 200` ms (adreno_dispatch.c:54), only samples
///   the GPU's progress registers on the ringbuffer dispatcher.
/// - An Adreno 840 (gen8, `adreno_gpu_core_gen8_*`, adreno-gpulist.h) runs
///   the hardware scheduler (`adreno_gen8_hwsched_gpudev`), not that
///   dispatcher: there long-IB detection (`ft_long_ib_detect`, 1 on the
///   840) enables the GMU firmware's bail-out timer
///   (adreno_gen8_hwsched_hfi.c:2663-2667, `HFI_FEATURE_BAIL_OUT_TIMER`
///   sent with data 0: the firmware's own default), whose expiry is the
///   context fault `GMU_GPU_SW_HANG` (603, adreno_hfi.h: "Fault due to Long
///   IB timeout"). Its duration lives in the GMU firmware: no kernel
///   constant, sysfs node or property states it (`KGSL_PROP_IB_TIMEOUT`,
///   0x21, is gone from msm_kgsl.h), and the 840's sysfs states none.
/// - The hardware hang interrupt (`ft_hang_intr_status` 1) counts cycles
///   of no interface progress (`hang_detect_cycles` 0xcfffff,
///   RBBM_INTERFACE_HANG_INT_CNTL, adreno_gen8.c:1785): a stall, not a
///   long submission that makes progress.
///
/// So the bound stated on Android is `adreno_drawobj_timeout`'s 2 s, the
/// one duration the driver states; where the GMU's bail-out is shorter, the per-submit
/// trace (`RUST_LOG=engine_gpu::rates=debug`) shows the estimate and time
/// of every submission up to the one that faults.
///
/// On Windows the bound is the display driver's timeout detection (TDR): a
/// submission past `TdrDelay` seconds resets the GPU and loses the device.
/// The platform states it in the registry (`HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers`):
/// `TdrDelay`, 2 when absent (Windows' own
/// default), and none where `TdrLevel` is 0 (detection off). None stated
/// elsewhere.
///
/// Read once, at the first load. Every engine that submits through the
/// platform's GPU driver (Vulkan, wgpu) states this one.
#[must_use]
pub fn stated_submit_bound() -> Option<std::time::Duration> {
    static STATED: std::sync::OnceLock<Option<std::time::Duration>> = std::sync::OnceLock::new();
    *STATED.get_or_init(platform_submit_bound)
}

#[cfg(target_os = "android")]
fn platform_submit_bound() -> Option<std::time::Duration> {
    Some(std::time::Duration::from_secs(2))
}

#[cfg(windows)]
fn platform_submit_bound() -> Option<std::time::Duration> {
    tdr_bound(tdr_value("TdrLevel"), tdr_value("TdrDelay"))
}

#[cfg(not(any(target_os = "android", windows)))]
fn platform_submit_bound() -> Option<std::time::Duration> {
    None
}

/// The bound Windows' timeout detection states: none where `TdrLevel` is 0
/// (detection off), else `TdrDelay` seconds, 2 where the registry states
/// none.
#[must_use]
pub fn tdr_bound(level: Option<u32>, delay: Option<u32>) -> Option<std::time::Duration> {
    if level == Some(0) {
        return None;
    }
    Some(std::time::Duration::from_secs(u64::from(
        delay.unwrap_or(2),
    )))
}

/// A `GraphicsDrivers` DWORD the registry states, or none.
#[cfg(windows)]
#[allow(unsafe_code)]
fn tdr_value(name: &str) -> Option<u32> {
    use windows_sys::Win32::System::Registry::{
        HKEY_LOCAL_MACHINE, RRF_RT_REG_DWORD, RegGetValueW,
    };
    let wide = |text: &str| {
        text.encode_utf16()
            .chain(std::iter::once(0))
            .collect::<Vec<u16>>()
    };
    let key = wide(r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers");
    let value = wide(name);
    let mut data: u32 = 0;
    let mut size = std::mem::size_of::<u32>() as u32;
    // SAFETY: the key and value names are NUL-terminated UTF-16 that outlive
    // the call, and `data` holds the `size` bytes a DWORD takes.
    let status = unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            key.as_ptr(),
            value.as_ptr(),
            RRF_RT_REG_DWORD,
            std::ptr::null_mut(),
            (&raw mut data).cast(),
            &raw mut size,
        )
    };
    (status == 0).then_some(data)
}

/// The `[rates]` keys the slang backends' estimate weighs a fire by (of
/// `profile::fire::KEYS`): what a load whose profile lacks one measures.
/// `few_row_tflops` and `workgroups_per_core` are not weighed here: no
/// choice on these backends reads them, so a load measures neither. The
/// dense kernels' curves at the model's planes, which a rung's dense kernel
/// is chosen by, are timed once the model's shapes are known
/// ([`shapes_at_load`]).
pub const KEYS: [&str; 6] = [
    "launch_ns",
    "gemm_tflops_by_rows",
    "attn_tflops_by_keys",
    "hbm_gbps",
    "elementwise_gbps",
    "scan_gelems",
];

/// The `[rates]` keys a load measures where its profile lacks one: the
/// estimate's [`KEYS`]; where the platform states a bound on one
/// submission (`max_submit`) the margin a serving step is held to it at
/// (`profile::fire::SUBMIT_WORST_RATIO`, `crate::submit`); and the
/// workgroups the device holds at once, which a guest step's grid strides
/// over and, where the device states no count of its compute units, what
/// sizes a fire's work to it (`profile::fire::RESIDENT_WORKGROUPS`).
#[must_use]
pub fn keys_at(max_submit: Option<std::time::Duration>) -> Vec<&'static str> {
    let mut keys = KEYS.to_vec();
    if max_submit.is_some() {
        keys.push(profile::fire::SUBMIT_WORST_RATIO);
        keys.push(profile::fire::ATTN_BY_ROWS_KEYS);
    }
    keys.push(profile::fire::RESIDENT_WORKGROUPS);
    keys
}

/// The rows of a prompt step the prefixed attention curve is timed at
/// (`profile::fire::Rates::attn_tflops_by_rows_keys`): the prompt rungs a
/// slang load on a bound-stating device lays its top at (wgpu's 512, native
/// Vulkan's 1024, and the narrower a phone's room leaves).
pub const PREFIX_ROWS: [u32; 4] = [128, 256, 512, 1024];

/// The rows a load's rungs lie at: every power of two from one row to
/// `top`, then `top`, the most rows a fire of the load may bring.
fn kernel_rows(top: u32) -> Vec<u32> {
    let top = top.max(1);
    let mut rows = engine::memory::rungs(1, top);
    if rows.last() != Some(&top) {
        rows.push(top);
    }
    rows
}

/// The rows each dense kernel is timed at at a shape
/// (`profile::fire::Rates::kernel_tflops_by_shape`): every power of four
/// from one row to `top`, then `top`, so no kernel is weighed past the rows
/// it was timed at; a rung between two is read between them, in log rows
/// (`crate::dense`).
fn timed_rows(top: u32) -> Vec<u32> {
    let top = top.max(1);
    let mut rows: Vec<u32> = std::iter::successors(Some(1u32), |&r| r.checked_mul(4))
        .take_while(|&r| r < top)
        .collect();
    rows.push(top);
    rows
}

/// The dense kernels a slang load chooses between at `tier` for fires of up
/// to `top` rows (`crate::dense`): each vector kernel, and each tiled kernel
/// the profile's path takes at some of those rows (a device whose path takes
/// the 32x32 tile below some rows and the cooperative-matrix tile above
/// chooses among both).
#[must_use]
pub fn dense_kernels<T: Tactics + ?Sized>(
    target: kernels_slang::Target,
    tactics: &T,
    tier: kernels_slang::Tier,
    top: u32,
) -> Vec<kernels_slang::linear::gemm::Tactic> {
    let mut all: Vec<kernels_slang::linear::gemm::Tactic> =
        kernels_slang::linear::gemm::VECTOR_ROWS
            .iter()
            .map(|&rows| kernels_slang::linear::gemm::Tactic {
                vector: rows,
                ..tactics.gemm(tier, 1)
            })
            .collect();
    for m in kernel_rows(top) {
        for tactic in crate::dense::tiled_kernels(target, tactics.gemm(tier, m)) {
            if !all.contains(&tactic) {
                all.push(tactic);
            }
        }
    }
    all
}

/// The curve `tactic`'s kernel at an `n x k` plane is measured and read as
/// (`profile::fire::KernelCurve`'s kind and key): a vector kernel by the rows
/// a workgroup carries, a tiled one by its row tile at the plane.
#[must_use]
pub fn curve_key(
    target: kernels_slang::Target,
    tactic: kernels_slang::linear::gemm::Tactic,
    n: u32,
    k: u32,
) -> (u32, u32) {
    if tactic.vector > 0 {
        (profile::fire::KernelCurve::VECTOR, tactic.vector)
    } else {
        let tile = kernels_slang::linear::gemm::tile_for(target, tactic, n, k).row_tile();
        (profile::fire::KernelCurve::TILE, tile.max(1))
    }
}

/// Of `kernels` at each of `shapes` (`(n, k)`), those whose curve `stated`
/// does not serve to `top` rows (`profile::fire::KernelCurve::serves`): what
/// a load measures once its model's shapes are known ([`measure_shapes`]).
/// Two kernels read as one curve at a shape are measured once.
#[must_use]
pub fn lacking(
    target: kernels_slang::Target,
    kernels: &[kernels_slang::linear::gemm::Tactic],
    stated: &Rates,
    shapes: &[(u32, u32)],
    top: u32,
) -> Vec<(kernels_slang::linear::gemm::Tactic, u32, u32)> {
    let mut out: Vec<(kernels_slang::linear::gemm::Tactic, u32, u32)> = Vec::new();
    let mut seen: Vec<(u32, u32, u32, u32)> = Vec::new();
    for &(n, k) in shapes {
        for &tactic in kernels {
            let (kind, key) = curve_key(target, tactic, n, k);
            let id = (kind, key, n, k);
            if seen.contains(&id) {
                continue;
            }
            seen.push(id);
            let served = stated
                .kernel_tflops_by_shape
                .iter()
                .flatten()
                .any(|c| c.id() == id && c.serves(top));
            if !served {
                out.push((tactic, n, k));
            }
        }
    }
    out
}

/// The weight planes a kernel's curve at an `n x k` plane rotates through:
/// `engine::rates::weights_at`'s (past the last-level cache), but one where
/// the plane alone passes every last-level cache served (a third of
/// `engine::rates::WEIGHT_BYTES`): read cold every fire by itself, as an
/// output head is. Never more than the model's weights hold of the plane
/// (`model`, its weight bytes, where stated): a served fire's plane is read
/// again after every other weight the model reads, so a model whose
/// weights sit in a cache reads its planes warm in serving too (the
/// rotation past the cache would lay millions of a 1 x 64 plane).
fn shape_planes(n: u32, k: u32, model: Option<u64>) -> usize {
    let plane = 2 * n as usize * k as usize;
    let planes = if plane > engine::rates::WEIGHT_BYTES / 3 {
        1
    } else {
        weights_at(n, k)
    };
    model.map_or(planes, |bytes| {
        let held = usize::try_from(bytes.div_ceil(plane.max(1) as u64)).unwrap_or(usize::MAX);
        planes.min(held.max(1))
    })
}

/// Whether a rates matmul at `m x n x k` may be laid: no operand past the
/// device's `largest` buffer, and the set (its activation, the weight planes
/// it rotates through, [`shape_planes`]; its result; bf16) within the room
/// the load admitted the rates step to lay (`engine::rates::step_bytes`).
fn operands_fit(m: u32, n: u32, k: u32, model: Option<u64>, largest: u64) -> bool {
    let (m64, n64, k64) = (u64::from(m), u64::from(n), u64::from(k));
    let (act, plane, y) = (2 * m64 * k64, 2 * n64 * k64, 2 * m64 * n64);
    let set = act + shape_planes(n, k, model) as u64 * plane + y;
    act.max(plane).max(y) <= largest && set <= engine::rates::step_bytes()
}

/// The timings a point of a kernel's curve at a shape takes, interleaved
/// with the other kernels' at the shape: its median the point, their spread
/// what a choice between two kernels may call a difference.
pub(crate) const REPEATS: usize = 3;

/// A curve point from its `samples`: the fastest (timing noise only ever
/// slows a fire) and their spread, `(max - min) / fastest`.
pub(crate) fn best_spread(samples: &[f64]) -> (f64, f64) {
    let best = samples.iter().copied().fold(f64::INFINITY, f64::min);
    let most = samples.iter().copied().fold(0.0, f64::max);
    (best, (most - best) / best.max(f64::MIN_POSITIVE))
}

/// The median of `samples` and their spread, `(max - min) / median`.
pub(crate) fn median_spread(samples: &[f64]) -> (f64, f64) {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    let spread = (sorted[sorted.len() - 1] - sorted[0]) / median.max(f64::MIN_POSITIVE);
    (median, spread)
}

/// The device time a point of a kernel's curve at a shape fills with its
/// fires' frame (`Rig::budgeted`), timed by the device's timestamps where it
/// has them (the host's wait, to 2.4 ms a frame on wgpu, then out of it):
/// enough fires that the dispatches' own spread averages out, few enough
/// that a curve's points take milliseconds each.
const POINT_NS: f64 = 4e6;

/// The matmul a launch is timed with: its work is nothing beside a launch.
pub const LAUNCH: u32 = 16;

/// The offset every view a rates step binds into a shared buffer starts at
/// a multiple of: every backend's storage-binding alignment at most.
const VIEW_ALIGN: u64 = 256;

/// Timed frames of each kind; the median is kept.
pub const ROUNDS: usize = 5;

/// The most single-fire frames a queued rates frame keeps in flight
/// (`Rig::queue`): as many as the frames of all but the tiniest fires hold
/// (a frame of 64 or fewer waits once, at its end, as an unqueued frame
/// does: every wait between is a host round trip, which on wgpu ran 0.3 to
/// 2.4 ms and read as the device's in a tiny fire's ratio), while a frame of
/// hundreds of tiny fires holds no more command buffers than that at once.
const QUEUED: usize = 64;

/// The kv page the attention's pool is laid in.
const PAGE: u32 = 16;

/// The most q / kv sets one attention point lays: each set binds its own q
/// and k and v buffers, and the step's bindings are a device's to bound
/// (a few-key point would lay thousands, its reads warm at any count).
const ATTN_SETS_MOST: usize = 256;

/// How a backend times a committed frame: `(adapter ns where it has
/// timestamps, host ns around the submit and the wait)`.
pub type Timer<'a, B> =
    &'a dyn Fn(<B as crate::backend::Backend>::Frame) -> Result<(Option<u64>, u64)>;

/// The kinds of work a rates step's fires do: a first fire's estimate is
/// grown from measured fires of its own kind ([`grown`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Matmul,
    /// A prompt's attention to itself (square).
    Attention,
    /// A prompt step's rows against a longer prefix: several times slower a
    /// flop than a square attention, never estimated from one.
    Prefixed,
    /// A copy or per-row ops: bytes moved.
    Memory,
    Scan,
}

/// What one fire asks of the device: its kind, its flops and the bytes it
/// moves at least.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Work {
    pub kind: Kind,
    pub flops: f64,
    pub bytes: f64,
}

impl Work {
    /// A bf16 matmul of `m x n x k`: its flops, its operands read and its
    /// output written.
    #[must_use]
    pub fn matmul(m: u32, n: u32, k: u32) -> Work {
        let (m, n, k) = (f64::from(m), f64::from(n), f64::from(k));
        Work {
            kind: Kind::Matmul,
            flops: 2.0 * m * n * k,
            bytes: 2.0 * (m * k + n * k + m * n),
        }
    }

    /// How many times `other`'s work this is, by the larger of its flops'
    /// and its bytes' ratios (a dimension `other` does none of is not
    /// weighed).
    fn times(&self, other: &Work) -> f64 {
        let ratio = |a: f64, b: f64| if b > 0.0 { a / b } else { 0.0 };
        ratio(self.flops, other.flops).max(ratio(self.bytes, other.bytes))
    }

    /// Whether this is no more work than `other` in either dimension.
    fn within(&self, other: &Work) -> bool {
        self.flops <= other.flops && self.bytes <= other.bytes
    }
}

/// A fire a rates step measured: its work and its time, ns, at the clock
/// the step's first probe read (its time over the clock's ratio then).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fired {
    pub work: Work,
    pub ns: f64,
}

/// A new shape's first fire, estimated before it is submitted from fires
/// measured before it, ns at the clock the probe first read: from the
/// largest fire of its kind it is no less work than, at that fire's rate
/// (a larger shape is never read faster than a smaller one ran); else from
/// the least fire of its kind it is no more work than, at that fire's time
/// (a smaller shape runs no longer); else, the first of its kind, from the
/// probe (`probe`: the smallest shape on the matmul curve, measured first)
/// at its rate, and never under its time. `None` where nothing is measured.
#[must_use]
pub fn grown(measured: &[Fired], probe: Option<Fired>, work: Work) -> Option<f64> {
    let order = |a: &&Fired, b: &&Fired| {
        a.work
            .flops
            .total_cmp(&b.work.flops)
            .then(a.work.bytes.total_cmp(&b.work.bytes))
    };
    let kind = || measured.iter().filter(|f| f.work.kind == work.kind);
    if let Some(below) = kind().filter(|f| f.work.within(&work)).max_by(order) {
        return Some(below.ns * work.times(&below.work).max(1.0));
    }
    if let Some(above) = kind().filter(|f| work.within(&f.work)).min_by(order) {
        return Some(above.ns);
    }
    probe.map(|p| p.ns * work.times(&p.work).max(1.0))
}

fn fault(call: &'static str) -> impl Fn(kernels_slang::Error) -> Fault {
    move |e| Fault::Device {
        call,
        why: e.to_string(),
    }
}

/// The device's buffers a micro-fire reads and writes, bound once.
struct Rig<'a, B: Serve> {
    device: &'a B::Context,
    handles: Handles<B>,
    pipelines: B::Pipelines,
    held: Vec<B::Buffer>,
    /// `engine::rates::operand_pattern`, made once.
    pattern: Option<Vec<u8>>,
    /// The device's bound on one submission, where it states one: the only
    /// cap on a timed frame (`engine::rates::timed_frames`).
    max_submit: Option<std::time::Duration>,
    /// A frame stopped at `max_submit` before it settled.
    unsettled: std::cell::Cell<bool>,
    /// The fire the frames now submitted run, as the trace names it.
    step: std::cell::RefCell<String>,
    /// That fire's last measured ns a fire (or the estimate its step was
    /// opened with, [`grown`]), at the clock the first probe read: what the
    /// next frame of it is estimated from, at the clock's ratio now.
    per_fire: std::cell::Cell<Option<f64>>,
    /// The fires of the frame `per_fire` was measured over (`u32::MAX` for
    /// an estimate grown from other shapes, whose error is theirs and moves
    /// no admission).
    basis: std::cell::Cell<u32>,
    /// The current step's fire, as its measurements are kept.
    work: std::cell::Cell<Option<Work>>,
    /// Every fire measured so far ([`grown`]'s).
    measured: std::cell::RefCell<Vec<Fired>>,
    /// Under the device's bound on one submission, the probe the device's
    /// clock is re-timed by before a frame that follows idle
    /// ([`crate::submit::Probe`]), the clock it read (its curve time the
    /// probe's first time: the clock every measurement is kept at), and
    /// the probe's first measurement.
    probe: Option<(
        crate::submit::Probe<B>,
        std::cell::Cell<crate::submit::Clock>,
        Fired,
    )>,
    /// The device was re-timed since the last frame landed.
    retimed: std::cell::Cell<bool>,
    /// How frames are admitted under `max_submit`
    /// (`engine::rates::Admission`): the worst ratio of a frame's time to
    /// its estimate over frames timed once the warm-up settled the clock,
    /// carried up from a frame of no more fires (one carried down holds the
    /// frame's own submit, and only overestimates).
    admission: std::cell::Cell<engine::rates::Admission>,
    /// The admission's worst ratio held where the caller gave it
    /// ([`measure_shapes`]'s `worst`): no landing moves it.
    worst_held: std::cell::Cell<bool>,
    /// The step whose fire, at one a frame, was past the bound: refused by
    /// name, and the step drops that shape ([`dropped`]).
    past_bound: std::cell::RefCell<Option<String>>,
    /// The ratio of the slowest fire to the fastest the warm-up's rotations
    /// measured at its cube: how far this device's clock swings from fire to
    /// fire, which a settled frame's admission carries as its least worst
    /// ratio (the Adreno 840 swung ~10% inside one settled step).
    swing: std::cell::Cell<f64>,
    /// The current step's fire's bytes of a frame's parameter region and
    /// the region's bytes, as its first frame took them, where the backend
    /// bounds that region ([`crate::backend::Frame::uniform_room`]): a frame
    /// of the step holds no more fires than fit it.
    uniforms: std::cell::Cell<Option<(u64, u64)>>,
    /// Whether the backend bounds a frame's parameter region (the first
    /// frame says).
    bounds_uniforms: std::cell::Cell<bool>,
    /// Each fire of a frame its own submission, queued behind the one before
    /// and the queue timed whole by the host's clock: so where the device
    /// states a bound on one submission (but for the launch's chain, whose
    /// dispatches are one submission by what it measures). No submission
    /// then holds more than one fire's work, whatever the margin is yet
    /// known to be, and the device stays loaded across them.
    queue: std::cell::Cell<bool>,
    /// The worst ratio of a warm-up rotation's time to the rotation before
    /// it (its estimate: the slowest fire it had measured), over the
    /// warm-up's rotations and the sustained ones after every rate: the
    /// device's own drift between like frames of a large fire, what the
    /// margin a serving step is held to carries (`[rates]
    /// submit_worst_ratio`). The micro-shapes' frames are left out: theirs
    /// is a submission's host latency against a tiny fire's work (a single
    /// fire's frame ran 0.3 to 2.4 ms on an RTX under wgpu) and the
    /// estimate's scaling between shapes, not the device.
    drift: std::cell::Cell<f64>,
    /// Every kernel the rig's frames fired since [`Rig::take_fired`] last
    /// took them: what a measurement's stored key names
    /// (`crate::stored`).
    fired: std::cell::RefCell<std::collections::BTreeSet<crate::naming::Kernel>>,
    /// Each point timed by single fires alone ([`for_bound`]): its sets
    /// touched and one fire untimed, then the median of [`ROUNDS`] fires,
    /// each a frame of its own, by the device's span where it writes one;
    /// no frame lengthened or warmed until its rate settles.
    quick: bool,
    /// The most weight planes a matmul's fires rotate through, where a
    /// reading needs none read cold ([`resident_for_skip`]: a compute-bound
    /// matmul against one workgroup of it, its operands' place in the cache
    /// moving neither); `None`, every reading's [`weights_at`].
    planes_most: Option<usize>,
}

/// A sink that fires through `inner` and keeps each kernel it fires: a
/// measurement's kernels, as it fired them.
struct Recording<'s, 'c> {
    inner: &'s kernels_slang::Ctx<'c>,
    fired: &'s std::cell::RefCell<std::collections::BTreeSet<crate::naming::Kernel>>,
}

impl kernels_slang::Encode for Recording<'_, '_> {
    fn fire(
        &self,
        fire: kernels_slang::Fire,
        args: &[kernels_slang::ArgValue],
    ) -> std::result::Result<(), kernels_slang::Error> {
        self.fired
            .borrow_mut()
            .insert(crate::naming::Kernel::of(fire)?);
        self.inner.fire(fire, args)
    }

    fn absent(&self) -> std::result::Result<kernels_slang::ArgValue, kernels_slang::Error> {
        self.inner.absent()
    }

    fn target(&self) -> kernels_slang::Target {
        self.inner.target()
    }

    fn comm(
        &self,
        op: &'static str,
    ) -> std::result::Result<kernels_slang::Comm, kernels_slang::Error> {
        self.inner.comm(op)
    }

    fn rendezvous(&self, op: &'static str) -> std::result::Result<(), kernels_slang::Error> {
        self.inner.rendezvous(op)
    }

    fn naming(&self) -> Option<&dyn kernels_slang::Naming> {
        self.inner.naming()
    }
}

impl<'a, B: Serve> Rig<'a, B> {
    /// The kernels fired since the last take.
    fn take_fired(&self) -> std::collections::BTreeSet<crate::naming::Kernel> {
        std::mem::take(&mut *self.fired.borrow_mut())
    }

    fn new(device: &'a B::Context, max_submit: Option<std::time::Duration>) -> Self {
        Rig {
            device,
            handles: Handles::<B>::new(),
            pipelines: B::Pipelines::default(),
            held: Vec::new(),
            pattern: None,
            max_submit,
            unsettled: std::cell::Cell::new(false),
            step: std::cell::RefCell::new(String::new()),
            per_fire: std::cell::Cell::new(None),
            basis: std::cell::Cell::new(0),
            work: std::cell::Cell::new(None),
            measured: std::cell::RefCell::new(Vec::new()),
            probe: None,
            retimed: std::cell::Cell::new(false),
            admission: std::cell::Cell::new(engine::rates::Admission::new(max_submit)),
            worst_held: std::cell::Cell::new(false),
            past_bound: std::cell::RefCell::new(None),
            swing: std::cell::Cell::new(1.0),
            uniforms: std::cell::Cell::new(None),
            bounds_uniforms: std::cell::Cell::new(true),
            queue: std::cell::Cell::new(max_submit.is_some()),
            drift: std::cell::Cell::new(1.0),
            fired: std::cell::RefCell::new(std::collections::BTreeSet::new()),
            quick: false,
            planes_most: None,
        }
    }

    /// The frames from here run `step`'s fire of `work`: its first frame
    /// estimated from the fires measured before ([`grown`]), the next from
    /// their own times as they land.
    fn step(&self, step: impl Into<String>, work: Work) {
        *self.step.borrow_mut() = step.into();
        let probe = self.probe.as_ref().map(|(_, _, first)| *first);
        self.per_fire
            .set(grown(&self.measured.borrow(), probe, work));
        self.basis.set(u32::MAX);
        self.work.set(Some(work));
        self.uniforms.set(None);
    }

    /// The frames from here go on with the same fire under the name `step`,
    /// estimated from what it measured.
    fn restep(&self, step: impl Into<String>) {
        *self.step.borrow_mut() = step.into();
    }

    /// Under the device's bound on one submission, the probe laid and fired
    /// first: the smallest shape on the matmul curve, the one fire the step
    /// submits before anything is measured (its time is what every later
    /// fire's estimate grows from, [`grown`]), and the clock every
    /// measurement is kept at.
    fn lay_probe(&mut self, tactic: kernels_slang::linear::gemm::Tactic) -> Result<()> {
        if self.max_submit.is_none() {
            return Ok(());
        }
        let probe = crate::submit::Probe::<B>::new(self.device, tactic)?;
        let first = probe.fire(self.device)?;
        let (m, n, k) = crate::submit::PROBE;
        let fired = Fired {
            work: Work::matmul(m, n, k),
            ns: first,
        };
        tracing::debug!(probe_us = first / 1e3, "rates: the probe's first time");
        self.measured.borrow_mut().push(fired);
        self.probe = Some((
            probe,
            std::cell::Cell::new(crate::submit::Clock::new(first)),
            fired,
        ));
        self.retimed.set(true);
        Ok(())
    }

    /// The clock's ratio now: the last probe's time over its first (1 with
    /// no bound stated).
    fn ratio(&self) -> f64 {
        self.probe
            .as_ref()
            .map_or(1.0, |(_, clock, _)| clock.get().ratio)
    }

    /// The current step's fire's estimate now, ns: at the clock's ratio.
    fn per_fire_now(&self) -> Option<f64> {
        self.per_fire.get().map(|p| p * self.ratio())
    }

    /// Under the device's bound on one submission, the device's clock
    /// re-timed where the next frame follows idle: the probe fired alone
    /// and timed, its time over its first the ratio every estimate is read
    /// at until the next. The rig's device is idle before every frame it
    /// lays: each frame waits for every submission it made before it
    /// returns, and nothing else submits on the device while a rates step
    /// runs (it runs before the load is ready). A queued frame's
    /// submissions go back to back and are re-timed once, before the
    /// first. The probe is admitted by its own estimate.
    fn retime(&self) -> Result<()> {
        let Some((probe, clock, _)) = self.probe.as_ref() else {
            return Ok(());
        };
        if self.retimed.get() {
            return Ok(());
        }
        let estimate = clock.get().probe_ns();
        if !self.admission.get().fits(Some(estimate)) {
            *self.past_bound.borrow_mut() = Some(self.step.borrow().clone());
            return Err(Fault::Device {
                call: "rates",
                why: format!(
                    "the probe ({:?} matmul) is estimated at {:.1} ms at the clock's last ratio \
                     x{:.2}, past the device's bound on one submission: not submitted",
                    crate::submit::PROBE,
                    estimate / 1e6,
                    clock.get().ratio
                ),
            });
        }
        let took = probe.fire(self.device)?;
        let mut read = clock.get();
        read.probed(took);
        clock.set(read);
        self.retimed.set(true);
        tracing::debug!(
            step = %self.step.borrow(),
            took_us = took / 1e3,
            ratio = read.ratio,
            "rates: the device idled; re-timed"
        );
        Ok(())
    }

    /// The most fires of the current step a frame's parameter region holds:
    /// one until a frame of the step has said what a fire takes of it, none
    /// past it where the backend bounds no such region.
    fn uniform_fires(&self) -> u32 {
        if !self.bounds_uniforms.get() {
            return u32::MAX;
        }
        match self.uniforms.get() {
            Some((per, room)) => u32::try_from(room / per.max(1)).unwrap_or(u32::MAX).max(1),
            None => 1,
        }
    }

    /// The bound one frame is admitted under: `max_submit` over the worst
    /// ratio of a settled frame's time to its estimate seen so far.
    fn admitting(&self) -> Option<std::time::Duration> {
        let a = self.admission.get();
        a.bound.map(|cap| cap.div_f64(a.worst))
    }

    /// The fires a frame of `count` of the current step holds: halved until
    /// it fits the bound; a single fire past it refused by name.
    fn fitting(&self, count: u32) -> Result<u32> {
        self.retime()?;
        let a = self.admission.get();
        let per_fire = self.per_fire_now();
        if self.queue.get() {
            // Each fire its own submission: as many as asked where one fits.
            return if a.fits(per_fire) {
                Ok(count.max(1))
            } else {
                Err(self.refuse(per_fire, 1))
            };
        }
        a.fires(count.min(self.uniform_fires()), per_fire)
            .ok_or_else(|| self.refuse(per_fire, 1))
    }

    /// The by-name refusal of a frame of `count` fires estimated at
    /// `per_fire` ns a fire (or not at all): marked, so its step drops the
    /// shape.
    fn refuse(&self, per_fire: Option<f64>, count: u32) -> Fault {
        let step = self.step.borrow().clone();
        let a = self.admission.get();
        *self.past_bound.borrow_mut() = Some(step.clone());
        let Some(per_fire) = per_fire else {
            return Fault::Device {
                call: "rates",
                why: format!(
                    "{step}: a frame of {count} fires has no estimate, and the device states a \
                     bound of {:.0} ms on one submission: not submitted",
                    a.bound.map_or(0.0, |b| b.as_secs_f64() * 1e3)
                ),
            };
        };
        let estimate = per_fire * f64::from(count);
        Fault::Device {
            call: "rates",
            why: format!(
                "{step}: a frame of {count} fires estimated at {:.1} ms ({:.1} ms at the worst \
                 ratio of a settled frame's time to its estimate, x{:.2}) is past the device's \
                 bound of {:.0} ms on one submission: not submitted",
                estimate / 1e6,
                estimate * a.worst / 1e6,
                a.worst,
                a.bound.map_or(0.0, |b| b.as_secs_f64() * 1e3)
            ),
        }
    }

    /// Every buffer laid so far, let go (their handles with them).
    fn release(&mut self) {
        self.handles = Handles::<B>::new();
        self.held.clear();
    }

    /// A buffer of `rows x width` of `dtype` holding
    /// `engine::rates::operand_pattern` repeated, bound: what every float
    /// operand a rate reads holds (not zeros, which some kernels run faster
    /// over than over a served fire's values).
    fn operand(&mut self, rows: u32, width: u32, dtype: Dtype) -> Result<Tensor> {
        let bytes = (u64::from(rows) * u64::from(width) * dtype.bits()).div_ceil(8);
        let mut buf = B::Buffer::zeroed(self.device, bytes.max(4), crate::books::owners::TUNING)?;
        let pattern = self
            .pattern
            .get_or_insert_with(engine::rates::operand_pattern);
        let mut at = 0u64;
        while at < bytes {
            let n = (pattern.len() as u64).min(bytes - at);
            buf.write(at, &pattern[..n as usize])?;
            at += n;
        }
        let handle = self.handles.bind(&buf, 0, buf.bytes())?;
        self.held.push(buf);
        Ok(Tensor::new(handle, rows, width, dtype))
    }

    /// `bytes` laid over as few buffers as the device's largest holds,
    /// each a whole number of `stride`s, held: filled with
    /// `engine::rates::operand_pattern` where `filled`, else zeroed.
    fn slab(&mut self, bytes: u64, stride: u64, filled: bool) -> Result<Vec<B::Buffer>> {
        let largest = (self.device.max_buffer() / stride).max(1) * stride;
        let mut out = Vec::new();
        let mut left = bytes.max(stride);
        while left > 0 {
            let size = left.min(largest);
            let mut buf = B::Buffer::zeroed(self.device, size, crate::books::owners::TUNING)?;
            if filled {
                let pattern = self
                    .pattern
                    .get_or_insert_with(engine::rates::operand_pattern);
                let mut at = 0u64;
                while at < size {
                    let n = (pattern.len() as u64).min(size - at);
                    buf.write(at, &pattern[..n as usize])?;
                    at += n;
                }
            }
            self.held.push(buf.clone());
            out.push(buf);
            left -= size;
        }
        Ok(out)
    }

    /// Up to `count` views of `rows x width` of `dtype` into `bufs`, one a
    /// stride of [`VIEW_ALIGN`]-aligned bytes, bound: as a served fire reads
    /// its weights from views into a few large segments, not from buffers
    /// of their own. As many as the buffers hold, one at least.
    fn views(
        &mut self,
        bufs: &[B::Buffer],
        count: usize,
        rows: u32,
        width: u32,
        dtype: Dtype,
    ) -> Result<Vec<Tensor>> {
        let bytes = (u64::from(rows) * u64::from(width) * dtype.bits())
            .div_ceil(8)
            .max(4);
        let stride = bytes.next_multiple_of(VIEW_ALIGN);
        let mut out = Vec::with_capacity(count);
        'bufs: for buf in bufs {
            let mut at = 0u64;
            while at + bytes <= buf.bytes() {
                if out.len() == count {
                    break 'bufs;
                }
                let handle = self.handles.bind(buf, at, bytes)?;
                out.push(Tensor::new(handle, rows, width, dtype));
                at += stride;
            }
        }
        if out.is_empty() {
            return Err(Fault::Ceiling {
                what: "bytes of a rates operand's views",
                need: bytes,
                have: bufs.iter().map(|b| b.bytes()).max().unwrap_or(0),
            });
        }
        Ok(out)
    }

    /// A zeroed buffer of `rows x width` of `dtype`, bound.
    fn zeros(&mut self, rows: u32, width: u32, dtype: Dtype) -> Result<Tensor> {
        let bytes = (u64::from(rows) * u64::from(width) * dtype.bits()).div_ceil(8);
        let buf = B::Buffer::zeroed(self.device, bytes.max(4), crate::books::owners::TUNING)?;
        let handle = self.handles.bind(&buf, 0, buf.bytes())?;
        self.held.push(buf);
        Ok(Tensor::new(handle, rows, width, dtype))
    }

    /// A buffer holding `words` (each four bytes), bound, read as `dtype`.
    fn words(&mut self, words: &[u32], rows: u32, width: u32, dtype: Dtype) -> Result<Tensor> {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut buf = B::Buffer::zeroed(
            self.device,
            bytes.len().max(4) as u64,
            crate::books::owners::TUNING,
        )?;
        buf.write(0, &bytes)?;
        let handle = self.handles.bind(&buf, 0, buf.bytes())?;
        self.held.push(buf);
        Ok(Tensor::new(handle, rows, width, dtype))
    }

    /// A frame of `count` of `fire`, then `then` on the frame, timed.
    fn frame(
        &self,
        timer: Timer<'_, B>,
        count: u32,
        fire: &dyn Fn(&kernels_slang::Ctx<'_>) -> Result<()>,
        then: &dyn Fn(&mut B::Frame) -> Result<()>,
    ) -> Result<(Option<u64>, u64)> {
        // A frame follows an idle device: re-timed first, under a bound.
        self.retime()?;
        let step = self.step.borrow().clone();
        let ratio = self.ratio();
        let per_fire = self.per_fire_now();
        let estimate = per_fire.map(|p| p * f64::from(count));
        let worst = self.admission.get().worst;
        let ms = |ns: f64| ns / 1e6;
        tracing::debug!(
            step = %step,
            fires = count,
            estimate_ms = ?estimate.map(ms),
            admitted_ms = ?estimate.map(|e| ms(e * worst)),
            worst,
            ratio,
            bound_ms = ?self.max_submit.map(|b| b.as_secs_f64() * 1e3),
            "submitting"
        );
        // A submission whose estimate, at the worst error seen, passes the
        // device's bound on one submission is refused by name, never
        // submitted: a submission past it loses the device. Queued, each
        // submission is one fire.
        let queued = self.queue.get() && count > 1;
        let submission = if queued { per_fire } else { estimate };
        if !self.admission.get().fits(submission) {
            let refused = if queued { 1 } else { count };
            return Err(self.refuse(per_fire, refused));
        }
        // Submitted: the device is no longer idle as the probe found it.
        self.retimed.set(false);
        let timed = if queued {
            // Each fire its own frame, all but the last committed behind
            // one another unawaited; the last, `then` on it, timed; the
            // queue's time the host's from the first commit to the last
            // landing (the adapter's span would time the last frame alone).
            let started = std::time::Instant::now();
            let mut pending = std::collections::VecDeque::with_capacity(QUEUED);
            for _ in 1..count {
                // No more than [`QUEUED`] frames in flight: the oldest
                // waited for before the next is laid.
                if pending.len() >= QUEUED
                    && let Some(oldest) = pending.pop_front()
                {
                    crate::backend::Pending::wait(&oldest)?;
                }
                let f = self.device.frame()?;
                {
                    let sink = B::sink(self.device, &f, &self.pipelines, &self.handles);
                    fire(&Recording {
                        inner: &sink,
                        fired: &self.fired,
                    })?;
                }
                pending.push_back(crate::backend::Frame::commit_async(f, None)?);
            }
            let mut f = self.device.spanned()?;
            {
                let sink = B::sink(self.device, &f, &self.pipelines, &self.handles);
                fire(&Recording {
                    inner: &sink,
                    fired: &self.fired,
                })?;
            }
            then(&mut f)?;
            timer(f)?;
            for p in &pending {
                crate::backend::Pending::wait(p)?;
            }
            (None, started.elapsed().as_nanos() as u64)
        } else {
            // A frame holds no more fires than its parameter region does
            // (`Rig::fitting` sizes frames so): one past it would not enqueue.
            if self.uniforms.get().is_some() && count > self.uniform_fires() {
                return Err(Fault::Device {
                    call: "rates",
                    why: format!(
                        "{step}: a frame of {count} fires passes the {} its parameter region holds",
                        self.uniform_fires()
                    ),
                });
            }
            // Spanned: the device's two timestamps around the frame's work,
            // none between its dispatches (a timestamp a dispatch serialises
            // a tiler's work and ran a phone's one-row fires ~10x long).
            let mut f = self.device.spanned()?;
            {
                let sink = B::sink(self.device, &f, &self.pipelines, &self.handles);
                let sink = Recording {
                    inner: &sink,
                    fired: &self.fired,
                };
                for _ in 0..count {
                    fire(&sink)?;
                }
            }
            match crate::backend::Frame::uniform_room(&f) {
                Some((taken, room)) => {
                    let per = taken.div_ceil(u64::from(count.max(1)));
                    let held = self.uniforms.get().map_or(0, |(p, _)| p);
                    self.uniforms.set(Some((per.max(held), room)));
                }
                None => self.bounds_uniforms.set(false),
            }
            then(&mut f)?;
            timer(f)?
        };
        let took = timed.0.unwrap_or(timed.1) as f64;
        if !self.worst_held.get() {
            let mut admission = self.admission.get();
            admission.landed(estimate, took, count >= self.basis.get());
            self.admission.set(admission);
        }
        // Kept at the clock the first probe read.
        let per_fire = took / f64::from(count.max(1)) / ratio;
        self.per_fire.set(Some(per_fire));
        self.basis.set(count);
        if let Some(work) = self.work.get() {
            self.measured
                .borrow_mut()
                .push(Fired { work, ns: per_fire });
        }
        tracing::debug!(
            step = %step,
            fires = count,
            took_ms = ms(took),
            estimate_ms = ?estimate.map(ms),
            worst = self.admission.get().worst,
            settled = self.admission.get().settled,
            "landed"
        );
        Ok(timed)
    }

    /// One `fire`'s time inside a frame: a frame of `n` of it against one,
    /// `n` doubled until a fire's time settles, then such frames until their
    /// rate stops moving (`engine::rates::warmed_frames`, every engine's
    /// rule; the device's `max_submit` the only cap), the host's clock around
    /// each frame's one submit.
    /// The current step's fire's ns at a point of a kernel's curve at a
    /// shape (`measure_shapes`): its sets touched once (`touched`, shared by
    /// every point over them), one warm fire, one fire timed, then one frame
    /// of as many fires as the step's estimate says fill [`POINT_NS`] of the
    /// device (as many as fit the bound), the fire's time the two frames'
    /// difference over the fires between (the device's span where the
    /// backend times one, else the host's clock). A point takes
    /// [`REPEATS`] of these, interleaved with the other kernels' at its shape
    /// (`measure_shapes`).
    fn budgeted(
        &self,
        timer: Timer<'_, B>,
        touched: &Touched,
        sets: usize,
        fire: &dyn Fn(&kernels_slang::Ctx<'_>) -> Result<()>,
    ) -> Result<u64> {
        let nothing = |_: &mut B::Frame| Ok(());
        let fitted = |n: u32| -> Result<f64> {
            let m = self.fitting(n)?;
            self.frame(timer, m, fire, &nothing)
                .map(|(_, w)| w as f64 * f64::from(n) / f64::from(m))
        };
        engine::rates::touch_once(
            touched,
            self.admitting(),
            u32::try_from(sets).unwrap_or(u32::MAX),
            &mut |n| fitted(n),
        )?;
        // A fire past the point's budget alone (an output head's at hundreds
        // of rows: a second a fire on a phone) is its own point: no warming
        // frame before it and no frame of more fires after it, whose one
        // frame's fixed cost it dwarfs.
        let long = |ns: Option<f64>| ns.is_some_and(|ns| ns >= POINT_NS);
        if !long(self.per_fire_now()) {
            self.fitting(1)?;
            self.frame(timer, 1, fire, &nothing)?;
        }
        // Each frame twice, the faster kept, by the device's span where the
        // backend times one, else the host's clock (whose wait only ever adds
        // to a frame's: it ran 0.3 to 2.4 ms on wgpu).
        let timed = |count: u32| -> Result<u64> {
            let (span, host) = self.frame(timer, count, fire, &nothing)?;
            Ok(span.unwrap_or(host))
        };
        let one = timed(1)?;
        if long(self.per_fire_now()) {
            return Ok(one.max(1));
        }
        let estimate = self.per_fire_now().unwrap_or(POINT_NS).max(1.0);
        let wanted = (POINT_NS / estimate).ceil().clamp(2.0, f64::from(u32::MAX)) as u32;
        let count = self.fitting(wanted)?;
        if count < 2 {
            return Ok(one.max(1));
        }
        let many = timed(count)?;
        Ok((many.saturating_sub(one) / u64::from(count - 1)).max(1))
    }

    fn each(
        &self,
        timer: Timer<'_, B>,
        touched: &Touched,
        sets: usize,
        fire: &dyn Fn(&kernels_slang::Ctx<'_>) -> Result<()>,
    ) -> Result<u64> {
        let nothing = |_: &mut B::Frame| Ok(());
        let wall = |count: u32| -> Result<u64> {
            let walls = (0..ROUNDS)
                .map(|_| self.frame(timer, count, fire, &nothing).map(|(_, w)| w))
                .collect::<Result<Vec<u64>>>()?;
            Ok(median(walls))
        };
        // A frame asked of `n` fires holds as many as fit the bound
        // (halved until they do, `Rig::fitting`), its time carried to `n`
        // at the rate it ran.
        let fitted = |n: u32| -> Result<f64> {
            let m = self.fitting(n)?;
            self.frame(timer, m, fire, &nothing)
                .map(|(_, w)| w as f64 * f64::from(n) / f64::from(m))
        };
        // Its sets touched once, by the first point over them.
        engine::rates::touch_once(
            touched,
            self.admitting(),
            u32::try_from(sets).unwrap_or(u32::MAX),
            &mut |n| fitted(n),
        )?;
        self.frame(timer, 1, fire, &nothing)?;
        if self.quick {
            let fires = (0..ROUNDS)
                .map(|_| {
                    self.frame(timer, 1, fire, &nothing)
                        .map(|(span, wall)| span.unwrap_or(wall))
                })
                .collect::<Result<Vec<u64>>>()?;
            return Ok(median(fires).max(1));
        }
        let frames = engine::rates::warmed_frames(self.admitting(), None, fitted, |why| {
            Fault::Device { call: "rates", why }
        })?;
        if frames.capped {
            self.unsettled.set(true);
        }
        let count = frames.fires;
        self.fitting(1)?;
        let one = wall(1)?;
        // The frame of many, halved until it fits the bound as the settled
        // device's frames have run against their estimates.
        let count = self.fitting(count)?;
        if count < 2 {
            return Ok(one);
        }
        self.frame(timer, count, fire, &nothing)?;
        let count = self.fitting(count)?;
        if count < 2 {
            return Ok(one);
        }
        let many = wall(count)?;
        Ok((many.saturating_sub(one) / u64::from(count - 1)).max(1))
    }
}

/// The call a rate's reading names where it is impossible: a frame timed
/// below its own dispatches' measured launches ([`engine::rates::rate_from`]).
/// No measurement, and no fault of the device: the sample is repeated
/// ([`repeated`]), and where every repeat is so, a curve's point is left
/// unmeasured ([`dropped`]) and a single rate refused by name.
const IMPLAUSIBLE: &str = "rates: a reading below its launches";

/// `measure` repeated while its reading is impossible ([`IMPLAUSIBLE`]): the
/// same sample, up to [`REPEATS`] times; the last reading kept.
fn repeated<T>(mut measure: impl FnMut() -> Result<T>) -> Result<T> {
    let mut last = measure();
    for _ in 1..REPEATS {
        match &last {
            Err(Fault::Device { call, .. }) if *call == IMPLAUSIBLE => last = measure(),
            _ => break,
        }
    }
    last
}

/// `measured` where its step measured; `None` where a single fire of its
/// shape was past the device's bound on one submission: that frame was
/// refused by name (never submitted), and the step drops the shape, as the
/// warm-up's walk drops a size, not the whole load.
fn dropped<B: Serve, T>(
    rig: &Rig<'_, B>,
    measured: Result<T>,
    say: &mut dyn FnMut(&str),
) -> Result<Option<T>> {
    match measured {
        Ok(v) => Ok(Some(v)),
        Err(Fault::Device { call, why }) if call == IMPLAUSIBLE => {
            tracing::info!(why = %why, "rates: every repeat of a point read below its launches; the point is left unmeasured");
            say(&format!("[rates]: unmeasured ({why}, every repeat)"));
            Ok(None)
        }
        Err(e) => match rig.past_bound.borrow_mut().take() {
            Some(step) => {
                tracing::debug!(step = %step, why = %e, "dropped: past the bound");
                say(&format!("[rates]: {step}: dropped ({e})"));
                Ok(None)
            }
            None => Err(e),
        },
    }
}

fn median(mut v: Vec<u64>) -> u64 {
    v.sort_unstable();
    v[v.len() / 2].max(1)
}

/// Each of `kernels` timed at `rows` of one `n x k` plane, [`REPEATS`]
/// rounds interleaved (each kernel's fire in turn): every fire's time, ns, a
/// kernel, carried to the whole plane. Each fire is a submission of its own
/// over `n / pieces[i]` of the plane's columns (the pieces the caller cut
/// so each fits the device's bound on one submission at its least clock),
/// reading the next plane of a rotation past the last-level cache and
/// writing one output, as a curve's point does. A piece the rig's own
/// admission still refuses is halved until admitted; a sample read below
/// its launches, or refused at every piece, is left out. What a dense
/// settle weighs a range by where it times no whole served step.
#[allow(clippy::too_many_arguments)]
pub fn time_kernels<B: Serve, T: Tactics + ?Sized>(
    device: &B::Context,
    tactics: &T,
    tier: kernels_slang::Tier,
    timer: Timer<'_, B>,
    max_submit: Option<std::time::Duration>,
    launch: f64,
    kernels: &[kernels_slang::linear::gemm::Tactic],
    pieces: &[u32],
    (n, k): (u32, u32),
    rows: u32,
) -> Result<Vec<Vec<f64>>> {
    let mut rig = Rig::<B>::new(device, max_submit);
    rig.lay_probe(tactics.gemm(tier, crate::submit::PROBE.0))?;
    let largest = rig.device.max_buffer();
    if !operands_fit(rows, n, k, None, largest) {
        return Ok(vec![Vec::new(); kernels.len()]);
    }
    let act = rig.operand(rows, k, Dtype::Bf16)?;
    let planes = shape_planes(n, k, None);
    let plane = (2 * u64::from(n) * u64::from(k)).next_multiple_of(VIEW_ALIGN);
    let slab = rig.slab(planes as u64 * plane, plane, true)?;
    let weights = rig.views(&slab, planes, n, k, Dtype::Bf16)?;
    let y = (2 * u64::from(rows) * u64::from(n)).next_multiple_of(VIEW_ALIGN);
    let ys = rig.slab(y, y, false)?;
    let nothing = |_: &mut B::Frame| Ok(());
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); kernels.len()];
    let mut split: Vec<u32> = pieces.iter().map(|&p| p.max(1)).collect();
    split.resize(kernels.len(), 1);
    let mut next = 0usize;
    // Under a bound, each kernel's pieces from its own narrowest one: one
    // fire over 64 of the plane's columns timed first, its device time
    // carried by the columns, the pieces as wide as that keeps within the
    // bound (no estimate grown from another kernel's fires stands for it).
    // A kernel whose narrowest piece passes the bound is not weighed.
    let mut skipped = vec![false; kernels.len()];
    if let Some(bound) = max_submit {
        let bound = bound.as_nanos() as f64;
        for (i, &tactic) in kernels.iter().enumerate() {
            let cols = 64u32.min(n);
            let w = Tensor {
                rows: cols,
                ..weights[0]
            };
            let out = rig.views(&ys, 1, rows, cols, Dtype::Bf16)?[0];
            let a = Tensor { rows, ..act };
            rig.step(
                format!("matmul {rows}x{cols}x{k}"),
                Work::matmul(rows, cols, k),
            );
            let fire = |sink: &kernels_slang::Ctx<'_>| {
                kernels_slang::linear::gemm::matmul(sink, tactic, a, w, out)
                    .map_err(fault("rates: matmul"))
            };
            let timed = rig.frame(timer, 1, &fire, &nothing);
            rig.past_bound.borrow_mut().take();
            let Ok((span, host)) = timed else {
                skipped[i] = true;
                continue;
            };
            let narrow = span.unwrap_or(host) as f64;
            if narrow > bound {
                skipped[i] = true;
                continue;
            }
            let per_piece = (bound / narrow).floor().max(1.0) * f64::from(cols);
            let fit = (f64::from(n) / per_piece).ceil() as u32;
            split[i] = split[i].max(fit.max(1));
        }
    }
    for _ in 0..REPEATS {
        for (i, &tactic) in kernels.iter().enumerate() {
            if skipped[i] {
                continue;
            }
            loop {
                let cols = n.div_ceil(split[i]).next_multiple_of(64).min(n);
                let w = Tensor {
                    rows: cols,
                    ..weights[next % weights.len()]
                };
                let out = rig.views(&ys, 1, rows, cols, Dtype::Bf16)?[0];
                let a = Tensor { rows, ..act };
                rig.step(
                    format!("matmul {rows}x{cols}x{k}"),
                    Work::matmul(rows, cols, k),
                );
                // One fire, one submission.
                let fire = |sink: &kernels_slang::Ctx<'_>| {
                    kernels_slang::linear::gemm::matmul(sink, tactic, a, w, out)
                        .map_err(fault("rates: matmul"))
                };
                let timed = rig.frame(timer, 1, &fire, &nothing);
                let past_bound = rig.past_bound.borrow_mut().take().is_some();
                match timed {
                    Ok((span, host)) => {
                        next += 1;
                        let ns = span.unwrap_or(host) as f64;
                        if let Ok(rate) = engine::rates::rate_from(
                            "matmul",
                            ns,
                            engine::rates::GEMM_DISPATCHES,
                            launch,
                            gemm_flops(rows, cols, k),
                        ) {
                            let ns = gemm_flops(rows, cols, k) / rate;
                            samples[i].push(ns * f64::from(n) / f64::from(cols) + launch);
                        }
                        break;
                    }
                    Err(_) if past_bound && cols > 64 => split[i] *= 2,
                    Err(e) if !past_bound => {
                        if !matches!(&e, Fault::Ceiling { .. }) {
                            return Err(e);
                        }
                        break;
                    }
                    _ => break,
                }
            }
        }
    }
    Ok(samples)
}

/// The version of [`measure_shapes`]' method (each dense kernel's curve's
/// stored key, `profile::fire::KernelCurve::by`): bumped only when how a
/// curve is measured changes.
pub const CURVE_METHOD: u32 = 1;

/// Each of `wanted` (`(kernel, n, k)`, [`lacking`]) timed at the load's
/// rows to `top` (`profile::fire::Rates::kernel_tflops_by_shape`): a matmul
/// of each row count over an `n x k` plane, each fire reading the next plane
/// past the last-level cache, its flops over its time, every weight pass it
/// makes in it. Each fire is admitted by the rig's estimate (grown from the
/// fires it measured, the probe's first under a stated bound, as every
/// rate's fire is), and a curve stops where the device's bound on one
/// submission refuses a point or its operands would not fit
/// (`curve_of`); where and why is recorded with it, so no load times it
/// again. `stated` holds the device's rates, the launch and the bandwidth
/// among them (a load measures these first, `engine::rates::rates_at_load`).
///
/// `model` states the model's weight bytes: no curve rotates through more
/// of a plane than they hold ([`shape_planes`]); `None` rotates past the
/// cache.
///
/// `worst` holds the admission's worst ratio of a fire's time to its
/// estimate where given (a test pins it so a point's admission does not
/// follow the box's clock); a load passes `None`, the ratio its fires
/// measure as they land.
pub fn measure_shapes<B: Serve, T: Tactics + ?Sized>(
    device: &B::Context,
    tactics: &T,
    tier: kernels_slang::Tier,
    timer: Timer<'_, B>,
    max_submit: Option<std::time::Duration>,
    worst: Option<f64>,
    stated: &Rates,
    wanted: &[(kernels_slang::linear::gemm::Tactic, u32, u32)],
    model: Option<u64>,
    top: u32,
    say: &mut dyn FnMut(&str),
) -> Result<Vec<profile::fire::KernelCurve>> {
    if wanted.is_empty() {
        return Ok(Vec::new());
    }
    let Some(launch_ns) = stated.launch_ns else {
        return Err(Fault::Device {
            call: "rates",
            why: "the kernels' curves at the model's shapes are timed on the device's launch, \
                  which the profile does not state"
                .to_owned(),
        });
    };
    let launch = launch_ns as f64;
    let mut rig = Rig::<B>::new(device, max_submit);
    // Under a stated bound, the probe first: every estimate grows from it.
    rig.lay_probe(tactics.gemm(tier, crate::submit::PROBE.0))?;
    // The clock warmed by the warm-up's cube where one fits the bound at the
    // worst ratio measured; where none does, the curves are timed unwarmed,
    // each point admitted by its own estimate (one past the bound stops its
    // curve, `KernelCurve::BOUND`): a sizing never fails the load.
    match warm(&mut rig, &|rows| tactics.gemm(tier, rows), timer) {
        Ok((side, warmed, _)) => say(&format!(
            "[rates]: warm-up: {} rotations of the {side}^3 matmul, settled at {:.1} TFLOP/s",
            warmed.rotations,
            warmed.rate / 1e3
        )),
        Err(why) if max_submit.is_some() => {
            rig.past_bound.borrow_mut().take();
            rig.release();
            // The clock warmed on the probe instead (one small matmul the
            // bound admitted as the step began), fired until its time
            // settles; the curves timed unwarmed only where there is none.
            let warmed = match rig.probe.as_ref() {
                Some((probe, _, _)) => {
                    let device = rig.device;
                    let settled = engine::rates::settle(|_| probe.fire(device).map(Some))?;
                    rig.retimed.set(false);
                    (!settled.capped).then_some(settled)
                }
                None => None,
            };
            match warmed {
                Some(settled) => {
                    let mut admission = rig.admission.get();
                    admission.settled = true;
                    rig.admission.set(admission);
                    tracing::info!(
                        fires = settled.steps,
                        probe_us = settled.last / 1e3,
                        "rates: the warm-up's cube is past the bound; warmed on the probe"
                    );
                }
                None => tracing::info!(
                    why = %why,
                    "rates: no warm-up cube fits the bound; the kernels' curves are timed \
                     unwarmed, each point admitted by its own estimate"
                ),
            }
        }
        Err(why) => return Err(why),
    }
    // The worst ratio of a fire's time to its estimate the points are
    // admitted at: the one measured as they land, or `worst` where given
    // (held there, every landing aside).
    if let Some(worst) = worst {
        let mut admission = rig.admission.get();
        admission.worst = worst.max(1.0);
        rig.admission.set(admission);
        rig.worst_held.set(true);
    }
    let rows = timed_rows(top);
    let largest = rig.device.max_buffer();
    // `(n, k, [(rows, ns a fire)])` of every tiled kernel's curve the
    // profile states: what a vector kernel is held beaten against beside
    // those timed here.
    let stated_tiled: Vec<(u32, u32, Vec<(u32, f64)>)> = stated
        .kernel_tflops_by_shape
        .iter()
        .flatten()
        .filter(|c| c.kind == profile::fire::KernelCurve::TILE)
        .map(|c| {
            let ns = c
                .curve
                .iter()
                .map(|&(m, t)| (m, gemm_flops(m, c.n, c.k) / (t * 1e3) + launch))
                .collect();
            (c.n, c.k, ns)
        })
        .collect();
    let mut shapes: Vec<(u32, u32)> = Vec::new();
    for &(_, n, k) in wanted {
        if !shapes.contains(&(n, k)) {
            shapes.push((n, k));
        }
    }
    let mut curves = Vec::with_capacity(wanted.len());
    for (n, k) in shapes {
        // One kernel a shape's curve, all timed over one set of operands
        // (they read a plane's shape, not its kernel): laid once at the most
        // of the rows they fit, each point a view of its rows, the weight
        // planes read once untimed for every point over them.
        struct Timed {
            tactic: kernels_slang::linear::gemm::Tactic,
            kind: u32,
            key: u32,
            name: String,
            points: Vec<(u32, f64, f64)>,
            stop: Option<(u32, u32)>,
            /// The parts of the plane's columns a fire covers one of: 1, else
            /// the slice the bound admitted, its time carried to all.
            split: u32,
            /// The points a stored curve of the kernel at the shape already
            /// states (another load's, to fewer rows), and their spreads,
            /// kept as stored: a later load extends a curve with the rows
            /// it lacks and never times a stored point again, so what a
            /// settle or a cover weighed by them stays.
            kept: Vec<(u32, f64)>,
            kept_spread: Vec<(u32, f64)>,
            /// The most rows `kept` reaches: rows past it alone are timed.
            past: u32,
        }
        let mut timed: Vec<Timed> = wanted
            .iter()
            .filter(|&&(_, wn, wk)| (wn, wk) == (n, k))
            .map(|&(tactic, ..)| {
                let (kind, key) = curve_key(B::TARGET, tactic, n, k);
                let name = match kind {
                    profile::fire::KernelCurve::VECTOR => {
                        format!("vector matmul ({key} rows a workgroup)")
                    }
                    _ => format!("{key}-row tile matmul"),
                };
                let stored = stated
                    .kernel_tflops_by_shape
                    .iter()
                    .flatten()
                    .find(|c| c.id() == (kind, key, n, k) && c.stop.is_none());
                let kept = stored.map(|c| c.curve.clone()).unwrap_or_default();
                Timed {
                    tactic,
                    kind,
                    key,
                    name,
                    points: Vec::new(),
                    stop: None,
                    split: 1,
                    past: kept.last().map_or(0, |&(m, _)| m),
                    kept,
                    kept_spread: stored.map(|c| c.spread.clone()).unwrap_or_default(),
                }
            })
            .collect();
        // A slice of the plane is a whole number of the widest column tile
        // among its kernels (so of every kernel's), and no narrower.
        let floor = timed
            .iter()
            .map(|t| kernels_slang::linear::gemm::tile_for(B::TARGET, t.tactic, n, k).col_tile())
            .max()
            .unwrap_or(1);
        let slice = |split: u32| n.div_ceil(floor).div_ceil(split) * floor;
        let fits = |m: u32| operands_fit(m, n, k, model, largest);
        let laid = match rows.iter().rev().copied().find(|&m| fits(m)) {
            Some(most) => {
                let act = rig.operand(most, k, Dtype::Bf16)?;
                // The planes as views into a few large buffers, the way a
                // served fire reads its weights from the pool's segments.
                let planes = shape_planes(n, k, model);
                let plane = (2 * u64::from(n) * u64::from(k)).next_multiple_of(VIEW_ALIGN);
                let slab = rig.slab(planes as u64 * plane, plane, true)?;
                let weights = rig.views(&slab, planes, n, k, Dtype::Bf16)?;
                // The outputs: room for one at the most rows, each fire at
                // fewer writing its own view of it (`ys`), as a served
                // step's matmuls write their own outputs.
                let y = (2 * u64::from(most) * u64::from(n)).next_multiple_of(VIEW_ALIGN);
                let ys = rig.slab(y, y, false)?;
                Some((act, weights, ys))
            }
            None => None,
        };
        let touched = Touched::default();
        for (i, &m) in rows.iter().enumerate() {
            if timed.iter().all(|t| t.stop.is_some()) {
                break;
            }
            // A kernel whose stored curve reaches these rows keeps its
            // points: only the rows past them are timed.
            let live: Vec<usize> = (0..timed.len())
                .filter(|&t| timed[t].stop.is_none() && m > timed[t].past)
                .collect();
            if live.is_empty() {
                continue;
            }
            let Some((act, weights, ys)) = laid.as_ref().filter(|_| fits(m)) else {
                for &t in &live {
                    timed[t].stop = Some((m, profile::fire::KernelCurve::BUFFER));
                }
                break;
            };
            // [`REPEATS`] rounds over every live kernel, interleaved so the
            // device's clock drifts over each alike; a kernel the bound
            // refuses stops there.
            let mut samples: Vec<Vec<f64>> = vec![Vec::new(); timed.len()];
            // One interleaved round over every live kernel. A fire the bound
            // refuses is timed on a slice of the plane's columns instead
            // (halved until admitted, no narrower than the widest kernel's
            // column tile):
            // its work and its reads are the columns', so its time carried
            // to them all is the kernel's at the plane. Refused at every
            // slice, its curve stops there.
            let round = |rig: &mut Rig<'_, B>,
                             timed: &mut Vec<Timed>,
                             samples: &mut Vec<Vec<f64>>,
                             m: u32,
                             live: &[usize],
                             say: &mut dyn FnMut(&str)|
             -> Result<()> {
                for &t in live {
                    // A slice's time, carried, while a wider one is tried.
                    let mut held: Option<f64> = None;
                    while timed[t].stop.is_none() {
                        let cols = slice(timed[t].split).min(n);
                        let plane: Vec<Tensor> = weights
                            .iter()
                            .map(|w| Tensor { rows: cols, ..*w })
                            .collect();
                        // One output every fire writes, as each matmul of a
                        // served step waits on the one before it: fires to
                        // outputs of their own ran side by side (a 64-row
                        // tile's one-row fire, latency-bound, read 15x fast),
                        // which no served step's matmuls do.
                        let outs = rig.views(ys, 1, m, cols, Dtype::Bf16)?;
                        let rate = matmul_on(
                            rig,
                            timed[t].tactic,
                            timer,
                            (Tensor { rows: m, ..*act }, &plane, &outs),
                            Some(&touched),
                            launch,
                        );
                        // A frame the device's room refuses (a ceiling, not a
                        // fault of the device) is no measurement either:
                        // its sample is dropped, never the step.
                        let rate = match rate {
                            Err(Fault::Ceiling { what, need, have }) => Ok(Err(format!(
                                "the frame needs {need} {what}, past the {have} the device holds"
                            ))),
                            rate => rate,
                        };
                        match dropped(rig, rate, say)? {
                            // Timed on a slice, the next fire tries one twice
                            // as wide, its estimate grown from this one: the
                            // widest slice the bound admits is timed, the
                            // fewest fixed costs carried to the plane.
                            // A fire timed below its launches is no
                            // measurement: the sample is dropped, never the
                            // step (a narrower slice's held time stands).
                            Some(Err(why)) => {
                                tracing::debug!(
                                    kernel = timed[t].name.as_str(),
                                    rows = m,
                                    n,
                                    k,
                                    why = %why,
                                    "rates point: a sample dropped"
                                );
                                say(&format!(
                                    "[rates]: {}: a sample dropped ({why})",
                                    timed[t].name
                                ));
                                samples[t].extend(held);
                                break;
                            }
                            Some(Ok(rate)) => {
                                let ns = gemm_flops(m, cols, k) / rate;
                                let carried = ns * f64::from(n) / f64::from(cols) + launch;
                                if timed[t].split > 1 {
                                    held = Some(carried);
                                    timed[t].split /= 2;
                                    continue;
                                }
                                samples[t].push(carried);
                                break;
                            }
                            None if held.is_some() => {
                                timed[t].split *= 2;
                                samples[t].extend(held);
                                break;
                            }
                            None if cols > floor => timed[t].split *= 2,
                            None => {
                                tracing::info!(
                                    kernel = timed[t].name.as_str(),
                                    rows = m,
                                    n,
                                    k,
                                    "rates point: past the bound, the curve stops"
                                );
                                timed[t].stop = Some((m, profile::fire::KernelCurve::BOUND));
                            }
                        }
                    }
                }
                Ok(())
            };
            // A point is the fastest of at least [`REPEATS`] samples, so one
            // noisy sample is never the point: a kernel some of whose
            // samples were dropped is timed in further rounds, as many again
            // at most.
            let short = |timed: &[_], samples: &[Vec<f64>]| {
                live.iter().any(|&t| {
                    let c: &Timed = &timed[t];
                    c.stop.is_none() && samples[t].len() < REPEATS
                })
            };
            let mut rounds = 0;
            while rounds < REPEATS || (rounds < 2 * REPEATS && short(&timed, &samples)) {
                round(&mut rig, &mut timed, &mut samples, m, &live, &mut *say)?;
                rounds += 1;
            }
            for &t in &live {
                if timed[t].stop.is_some() {
                    continue;
                }
                if samples[t].len() < REPEATS {
                    // Too few samples kept to tell a point from one noisy
                    // sample: no measurement here, the curve stops (a curve
                    // of no point and no stop would not read back).
                    tracing::info!(
                        kernel = timed[t].name.as_str(),
                        rows = m,
                        n,
                        k,
                        kept = samples[t].len(),
                        "rates point: fewer samples kept than a point's {REPEATS}, the curve stops"
                    );
                    timed[t].stop = Some((m, profile::fire::KernelCurve::UNTIMED));
                    continue;
                }
                let (ns, spread) = best_spread(&samples[t]);
                tracing::info!(
                    kernel = timed[t].name.as_str(),
                    rows = m,
                    n,
                    k,
                    measured_ms = ns / 1e6,
                    spread,
                    "rates point"
                );
                say(&format!(
                    "[rates]: {} {m}x{n}x{k}: {:.3} ms a fire (spread {:.1}%)",
                    timed[t].name,
                    ns / 1e6,
                    spread * 100.0
                ));
                timed[t].points.push((m, ns, spread));
            }
            // A point slower a flop than both its neighbours is timed again
            // ([`REPEATS`] rounds more, its fastest kept): timing noise only
            // ever slows a fire, so a dip is re-timed, never overridden; one
            // still slow after it is the kernel's own, and stands.
            for &t in &live {
                let Some(&[(m0, ns0, _), (m1, ns1, _), (m2, ns2, _)]) = timed[t]
                    .points
                    .len()
                    .checked_sub(3)
                    .map(|at| &timed[t].points[at..])
                    .and_then(|last| <&[_; 3]>::try_from(last).ok())
                else {
                    continue;
                };
                let rate = |m: u32, ns: f64| gemm_flops(m, n, k) / ns.max(1.0);
                if m2 != m || !(rate(m1, ns1) < rate(m0, ns0) && rate(m1, ns1) < rate(m2, ns2)) {
                    continue;
                }
                let stop = timed[t].stop;
                let mut again: Vec<Vec<f64>> = vec![Vec::new(); timed.len()];
                for _ in 0..REPEATS {
                    round(&mut rig, &mut timed, &mut again, m1, &[t], &mut *say)?;
                }
                timed[t].stop = stop;
                let best = again[t].iter().copied().fold(ns1, f64::min);
                tracing::info!(
                    kernel = timed[t].name.as_str(),
                    rows = m1,
                    n,
                    k,
                    was_ms = ns1 / 1e6,
                    now_ms = best / 1e6,
                    "rates point: slower a flop than both neighbours, timed again"
                );
                let at = timed[t].points.len() - 2;
                timed[t].points[at].1 = best;
            }
            // A vector kernel slower than twice the fastest tiled kernel at
            // these rows stops: a tiled kernel's time at twice the rows is no
            // more than twice its time (it could run as two fires), and a
            // vector kernel's time never falls with more rows, so it is
            // beaten over the next doubling at least; past that its weight
            // passes (one a workgroup's rows) only grow with the rows.
            let fastest_tiled = timed
                .iter()
                .filter(|c| c.kind == profile::fire::KernelCurve::TILE)
                .filter_map(|c| c.points.last().filter(|p| p.0 == m).map(|p| p.1))
                .chain(
                    stated_tiled
                        .iter()
                        .filter(|(tn, tk, _)| (*tn, *tk) == (n, k))
                        .filter_map(|(_, _, curve)| {
                            let inside = curve.first().is_some_and(|&(at, _)| at <= m)
                                && curve.last().is_some_and(|&(at, _)| at >= m);
                            inside
                                .then(|| profile::fire::interpolate(curve, f64::from(m)))
                                .flatten()
                        }),
                )
                .reduce(f64::min);
            if let (Some(tiled), Some(&next)) = (fastest_tiled, rows.get(i + 1)) {
                for c in timed
                    .iter_mut()
                    .filter(|c| c.tactic.vector > 0 && c.stop.is_none())
                {
                    if c.points
                        .last()
                        .is_some_and(|p| p.0 == m && p.1 > 2.0 * tiled)
                    {
                        c.stop = Some((next, profile::fire::KernelCurve::BEATEN));
                    }
                }
            }
        }
        drop(laid);
        rig.release();
        for c in timed {
            if let Some((at, why)) = c.stop {
                tracing::info!(
                    kernel = c.name.as_str(),
                    n,
                    k,
                    rows = at,
                    why,
                    "rates curve stops short of the load's rows"
                );
            }
            // Keyed by its method and its kernel's code (`crate::stored`).
            let kernel = kernels_slang::linear::gemm::entry(B::TARGET, c.tactic, n, k)
                .and_then(crate::naming::Kernel::of)
                .map_err(fault("rates: a curve's kernel"))?;
            let by = crate::stored::by(B::TARGET, profile::fire::KERNEL_CURVES, [&kernel])?;
            tracing::info!(
                "rates: curve {}/{} at {n}x{k} measured, stored under its key: {}",
                c.kind,
                c.key,
                by.said()
            );
            curves.push(profile::fire::KernelCurve {
                by,
                kind: c.kind,
                key: c.key,
                n,
                k,
                curve: c
                    .kept
                    .iter()
                    .copied()
                    .chain(
                        c.points
                            .iter()
                            .map(|&(m, ns, _)| (m, gemm_flops(m, n, k) / (ns - launch).max(1.0) / 1e3)),
                    )
                    .collect(),
                stop: c.stop,
                spread: c
                    .kept_spread
                    .iter()
                    .copied()
                    .chain(c.points.iter().map(|&(m, _, spread)| (m, spread)))
                    .collect(),
            });
        }
    }
    Ok(curves)
}

/// `chip` with each dense kernel's curve at each of `shapes` to `top` rows
/// (`profile::fire::Rates::kernel_tflops_by_shape`): the curves it lacks
/// ([`lacking`]) measured on `device` ([`measure_shapes`]), joined to those
/// it states, and saved to the user's profile `save` names, as a load saves
/// its rates (`engine::rates::save_rates`). Nothing is timed where it lacks
/// none.
#[allow(clippy::too_many_arguments)]
pub fn shapes_at_load<B: Serve, T: Tactics + ?Sized>(
    device: &B::Context,
    tactics: &T,
    tier: kernels_slang::Tier,
    timer: Timer<'_, B>,
    max_submit: Option<std::time::Duration>,
    chip: profile::Profile,
    save: Option<engine::rates::SaveTo<'_>>,
    on: &str,
    shapes: &[(u32, u32)],
    model: Option<u64>,
    top: u32,
    measuring: engine::rates::Measuring,
) -> Result<profile::Profile> {
    let stated = chip.rates();
    let kernels = dense_kernels(B::TARGET, tactics, tier, top);
    let wanted = lacking(B::TARGET, &kernels, &stated, shapes, top);
    let name = |&(tactic, n, k): &(kernels_slang::linear::gemm::Tactic, u32, u32)| {
        let (kind, key) = curve_key(B::TARGET, tactic, n, k);
        format!("{kind}/{key} at {n}x{k}")
    };
    // A curve stored under a stale key (a kernel's code or the method
    // changed) serves by its old figure, a fair estimate, until the export
    // round or `pie tune` measures it: nothing is timed before ready that a
    // figure exists for. Only a curve no layer states at all is timed here.
    let stale_curves = stated
        .stale
        .as_ref()
        .and_then(|stale| stale.kernel_tflops_by_shape.clone())
        .unwrap_or_default();
    let (chip, stated, wanted) = if stale_curves.is_empty() || wanted.is_empty() {
        (chip, stated, wanted)
    } else {
        let with = stated.clone().over(Rates {
            kernel_tflops_by_shape: Some(stale_curves),
            ..Rates::default()
        });
        let still = lacking(B::TARGET, &kernels, &with, shapes, top);
        let served: Vec<String> = wanted
            .iter()
            .filter(|w| !still.contains(w))
            .map(name)
            .collect();
        if served.is_empty() {
            (chip, stated, wanted)
        } else {
            let mut tuned = chip.tuned().clone();
            for curve in &served {
                tracing::info!(
                    "rates: gemm curve {curve}: served by the curve stored under its stale key \
                     (not timed again by this load: the export round or `pie tune` measures it)"
                );
            }
            tuned.rates = with.clone();
            (chip.with_tuned(tuned), with, still)
        }
    };
    if wanted.is_empty() {
        tracing::info!(
            shapes = shapes.len(),
            top,
            "rates: every dense kernel's curve at the model's planes is stored or committed under \
             a matching key; none timed"
        );
        return Ok(chip);
    }
    // Skipped: nothing timed or stored; each curve's kernel is weighed by
    // the curves stated where any is (`crate::dense::choose_unrated`), each
    // rung's kernel estimated from the work's shape where none is
    // (`crate::dense::by_shape`).
    if measuring == engine::rates::Measuring::Skip {
        let served = if stated.kernel_tflops_by_shape.is_some() {
            "the dense choice weighs the kernels whose curves are stated, and keeps the rung \
             below's kernel where none reaches"
        } else {
            "no curve is stated, so each rung's kernel is estimated from the work's shape (the \
             weight it reads and the rows it computes; `dense matmul` lines)"
        };
        let mut tuned = chip.tuned().clone();
        for want in &wanted {
            let curve = name(want);
            tracing::info!(
                "rates: skipped: gemm curve {curve} to {top} rows, served by the estimate: {served}"
            );
            tuned.skipped.push(format!("curve {curve}"));
        }
        return Ok(chip.with_tuned(tuned));
    }
    tracing::info!(
        curves = wanted.len(),
        shapes = shapes.len(),
        top,
        "rates: timing the dense kernels at the model's planes, each stored under no matching \
         key or short of {top} rows: {}",
        wanted.iter().map(name).collect::<Vec<_>>().join(", ")
    );
    let reference = tactics.gemm(tier, crate::submit::PROBE.0);
    let clock_before = clock_warmed::<B>(device, reference)?;
    let curves = measure_shapes::<B, T>(
        device,
        tactics,
        tier,
        timer,
        max_submit,
        None,
        &stated,
        &wanted,
        model,
        top,
        &mut |_| {},
    )?;
    // The user's file gains what this load measured alone: a curve taken
    // from a committed profile stays the committed profile's.
    let fresh = Rates {
        kernel_tflops_by_shape: Some(curves.clone()),
        measured_on: Some(on.to_string()),
        ..Rates::default()
    };
    let measured = Rates {
        kernel_tflops_by_shape: Some(curves),
        ..Rates::default()
    }
    .over(stated.clone());
    let joined = fresh;
    // Curves timed on a device whose clock sagged during them serve this load
    // alone, never stored (a hot or unramped device's curves would steer
    // every later load's picks).
    let throttle = throttled(clock_before, clock_read::<B>(device, reference)?);
    let saved_to = match throttle {
        None => save
            .map(|save| engine::rates::save_rates(&save, &joined))
            .transpose()
            .map_err(|why| Fault::Device {
                call: "rates at load",
                why,
            })?,
        Some(ratio) => {
            tracing::warn!(
                ratio,
                "rates: the dense curves were timed with the device {ratio:.2}x slower at their end than \
                 at their warmed start: served this load, not stored"
            );
            None
        }
    };
    let mut tuned = chip.tuned().clone();
    tuned.rates = measured;
    tuned.measured_this_load = true;
    tuned.timed.push(match throttle {
        None => format!("dense curves ({})", wanted.len()),
        Some(ratio) => format!(
            "dense curves ({}; device {ratio:.2}x slower at their end than at their start: not stored)",
            wanted.len()
        ),
    });
    if saved_to.is_some() {
        tuned.saved_to = saved_to;
    }
    Ok(chip.with_tuned(tuned))
}

/// What `pie tune <backend> --rates` writes for `device`
/// (`engine::rates::Rated`): its user file `existing` (none where there is
/// none yet) with its `[rates]` replaced by `measured` and `provenance` set,
/// as a load saves what it measured (`engine::rates::saved_text`), and the
/// block alone.
#[must_use]
pub fn rated(
    device: &str,
    existing: Option<&str>,
    measured: &Rates,
    provenance: &(&'static str, String),
    seconds: f64,
) -> engine::rates::Rated {
    let file =
        engine::rates::saved_text(existing.map(without_rates).as_deref(), measured, provenance);
    let block = format!(
        "{}\n[provenance]\n{} = {}\n",
        engine::rates::tuned(measured).trim_end(),
        provenance.0,
        provenance.1
    );
    engine::rates::Rated {
        device: device.to_owned(),
        file,
        block,
        seconds,
    }
}

/// `text` with its `[rates]` table taken out: the header and every line to
/// the next table's header (one at the line's start; a rate's array rows
/// are indented).
fn without_rates(text: &str) -> String {
    let header = |line: &str| line.starts_with('[') && line.trim_end().ends_with(']');
    let mut out = String::new();
    let mut in_rates = false;
    for line in text.lines() {
        if header(line) {
            in_rates = line.trim_end() == "[rates]";
        }
        if !in_rates {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// The rates of `device`, its kernels under the tactics `tactics` picks at
/// `tier`, no timed frame past `max_submit` where the device states one
/// (`engine::rates::timed_frames`).
pub fn measure<B: Serve, T: Tactics + ?Sized>(
    device: &B::Context,
    tactics: &T,
    tier: kernels_slang::Tier,
    timer: Timer<'_, B>,
    max_submit: Option<std::time::Duration>,
    stated: &Rates,
) -> Result<engine::rates::Measured> {
    measure_saying::<B, T>(
        device,
        tactics,
        tier,
        timer,
        max_submit,
        stated,
        &mut |_| {},
    )
}

/// Which of [`KEYS`] a step measures over `stated`: those it lacks, and
/// the launch and copy together where either is lacked (one timing gives
/// both).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Wanted {
    launch: bool,
    gemm: bool,
    attn: bool,
    elementwise: bool,
    scan: bool,
    resident: bool,
}

impl Wanted {
    fn of(stated: &Rates, keys: &[&'static str]) -> Wanted {
        let lacked = stated.missing(keys);
        let lacks = |key: &str| lacked.contains(&key);
        Wanted {
            resident: lacks(profile::fire::RESIDENT_WORKGROUPS),
            launch: lacks("launch_ns") || lacks("hbm_gbps"),
            gemm: lacks("gemm_tflops_by_rows"),
            attn: lacks("attn_tflops_by_keys"),
            elementwise: lacks("elementwise_gbps"),
            scan: lacks("scan_gelems"),
        }
    }
}

/// [`measure`], saying how the warm-up ended and what each shape reached.
pub fn measure_saying<B: Serve, T: Tactics + ?Sized>(
    device: &B::Context,
    tactics: &T,
    tier: kernels_slang::Tier,
    timer: Timer<'_, B>,
    max_submit: Option<std::time::Duration>,
    stated: &Rates,
    say: &mut dyn FnMut(&str),
) -> Result<engine::rates::Measured> {
    // Only what `stated` lacks is timed; what it states stands in for the
    // rates the timed ones read (the launch, the bandwidth, the curve's peak).
    // The keys a load asks for at this bound ([`keys_at`]), those `stated`
    // lacks.
    let want = Wanted::of(stated, &keys_at(max_submit));
    // The device's clock before the step: read again after it, the step's
    // rates are this load's alone where the after read is slower than the
    // warmed before read.
    let reference = tactics.gemm(tier, crate::submit::PROBE.0);
    let clock_before = clock_warmed::<B>(device, reference)?;
    let mut rig = Rig::<B>::new(device, max_submit);
    // Under a stated bound, the probe first: every estimate grows from it.
    rig.lay_probe(tactics.gemm(tier, crate::submit::PROBE.0))?;
    // The launch first: every rate below takes its dispatches' launches off.
    let (bytes_per_ns, launch_ns) = match (want.launch, stated.hbm_gbps, stated.launch_ns) {
        (false, Some(bytes), Some(launch)) => (bytes, launch),
        _ => repeated(|| {
            let measured = copy_and_launch(&mut rig, tactics.gemm(tier, LAUNCH), timer, say);
            rig.release();
            measured
        })?,
    };
    // Each measurement's kernels, as its frames fired them: what its stored
    // key names (`crate::stored`); every kernel the step fired keys the
    // margin it measures over the whole step.
    let launch_fired = rig.take_fired();
    let mut step_fired = launch_fired.clone();
    let launch = launch_ns as f64;
    let (side, warmed, walk_worst) = warm(&mut rig, &|rows| tactics.gemm(tier, rows), timer)?;
    step_fired.extend(rig.take_fired());
    say(&format!(
        "[rates]: warm-up: {} rotations of the {side}^3 matmul, settled at {:.1} TFLOP/s",
        warmed.rotations,
        warmed.rate / 1e3
    ));
    // The matmul curve at each width of `engine::rates::GEMM_WIDTHS` (n = k)
    // and each row count of `engine::rates::GEMM_ROWS`, through the kernel this
    // backend picks at that shape (`kernels_slang::linear::gemm::tile_for`);
    // the rows are multiples of every kernel's row tile, so the estimate's
    // padding to it (`row_tile`) lies between them.
    let mut gemm = Vec::new();
    for w in engine::rates::GEMM_WIDTHS.into_iter().filter(|_| want.gemm) {
        let mut curve = Vec::new();
        for m in engine::rates::GEMM_ROWS {
            let rate = repeated(|| {
                let rate = matmul(&mut rig, tactics.gemm(tier, m), timer, (m, w, w), launch);
                rig.release();
                rate
            });
            let Some(rate) = dropped(&rig, rate, say)? else {
                continue;
            };
            say(&format!(
                "[rates]: matmul {m}x{w}x{w}: {:.1} TFLOP/s",
                rate / 1e3
            ));
            curve.push((m, rate));
        }
        // A width whose every point was past the bound states none.
        if !curve.is_empty() {
            gemm.push((w, curve));
        }
    }
    let gemm_fired = rig.take_fired();
    step_fired.extend(gemm_fired.iter().copied());
    // `resident_workgroups`: the workgroups the device holds at once, from
    // the matmul curve's peak (its kernel compute-bound there) and one
    // workgroup of the same kernel over the same depth.
    let resident = if want.resident {
        let w = engine::rates::GEMM_WIDTH;
        let curve = gemm
            .iter()
            .find(|(width, _)| *width == w)
            .map(|(_, curve)| curve.clone())
            .or_else(|| {
                stated
                    .gemm_tflops_by_rows
                    .iter()
                    .flatten()
                    .find(|(width, _)| *width == w)
                    .map(|(_, curve)| curve.iter().map(|&(m, t)| (m, t * 1e3)).collect())
            })
            .ok_or_else(|| Fault::Device {
                call: "rates",
                why: format!(
                    "no matmul curve at width {w} to measure the resident workgroups against"
                ),
            })?;
        let measured = resident_workgroups(&mut rig, tactics, tier, timer, &curve, w, launch)?;
        rig.release();
        say(&format!(
            "[rates]: resident workgroups: {measured:.1} (one workgroup of the matmul's peak kernel against its full rate)"
        ));
        Some(measured)
    } else {
        None
    };
    let resident_fired = rig.take_fired();
    step_fired.extend(resident_fired.iter().copied());
    // The attention curve at its own keys: from the narrowest whose shape
    // is compute-bound on this device (its matmul peak over its bandwidth),
    // whatever rows the matmul curve reached.
    let stated_gemm: Vec<(u32, Vec<(u32, f64)>)> = stated
        .gemm_tflops_by_rows
        .iter()
        .flatten()
        .map(|(w, curve)| (*w, curve.iter().map(|&(m, t)| (m, t * 1e3)).collect()))
        .collect();
    let peak = if want.gemm { &gemm } else { &stated_gemm }
        .iter()
        .flat_map(|(_, curve)| curve.iter().map(|&(_, rate)| rate))
        .fold(0.0, f64::max);
    let mut attn: Vec<(u32, f64)> = Vec::new();
    let attn_keys = if want.attn {
        engine::rates::attn_keys(peak / bytes_per_ns.max(f64::MIN_POSITIVE))
    } else {
        Vec::new()
    };
    for keys in attn_keys {
        let shape = engine::rates::ATTN_SHAPE.at(keys);
        let rate = repeated(|| {
            let rate = attention(
                &mut rig,
                tactics,
                tier,
                timer,
                shape,
                launch,
                Kind::Attention,
            );
            rig.release();
            rate
        });
        let Some(rate) = dropped(&rig, rate, say)? else {
            continue;
        };
        say(&format!(
            "[rates]: attention {keys} keys d 128: {:.1} TFLOP/s",
            rate / 1e3
        ));
        attn.push((keys, rate));
    }
    // Not one square point fit the bound: the narrowest square halved until
    // one fire does (a rate a flop, which a smaller square of the same heads
    // reads); a 16-key square past it refuses by name.
    if want.attn && attn.is_empty() {
        let mut keys = engine::rates::ATTN_KEYS[0];
        loop {
            keys /= 2;
            let shape = engine::rates::ATTN_SHAPE.at(keys);
            let rate = repeated(|| {
                let rate = attention(
                    &mut rig,
                    tactics,
                    tier,
                    timer,
                    shape,
                    launch,
                    Kind::Attention,
                );
                rig.release();
                rate
            });
            if let Some(rate) = dropped(&rig, rate, say)? {
                say(&format!(
                    "[rates]: attention {keys} keys d 128: {:.1} TFLOP/s (the most keys whose \
                     fire fits the bound)",
                    rate / 1e3
                ));
                attn.push((keys, rate));
                break;
            }
            if keys <= 16 {
                return Err(Fault::Device {
                    call: "rates",
                    why: format!(
                        "a {keys}-key attention is past the device's bound on one submission: \
                         no attention can run under it"
                    ),
                });
            }
        }
    }
    let longest = attn
        .last()
        .map_or(engine::rates::ATTN_KEYS[0], |&(keys, _)| keys);
    for d in engine::rates::ATTN_DIMS
        .into_iter()
        .filter(|&d| want.attn && d != 128)
    {
        let shape = engine::rates::ATTN_SHAPE.at(longest).with_dim(d);
        let rate = repeated(|| {
            let rate = attention(
                &mut rig,
                tactics,
                tier,
                timer,
                shape,
                launch,
                Kind::Attention,
            );
            rig.release();
            rate
        });
        let Some(rate) = dropped(&rig, rate, say)? else {
            continue;
        };
        say(&format!(
            "[rates]: attention {longest} keys d {d}: {:.1} TFLOP/s",
            rate / 1e3
        ));
    }
    let attn_fired = rig.take_fired();
    step_fired.extend(attn_fired.iter().copied());
    // Under a stated submit bound, the attention of a prompt step's rows
    // against a longer prefix, at each of [`PREFIX_ROWS`] against each of
    // `engine::rates::ATTN_KEYS` past them: the shapes the steps that near
    // the bound fire, which the square curve reads up to twice as fast as
    // they run (`profile::fire::Rates::attn_tflops_by_rows_keys`). A point
    // past the bound is dropped by name, as every rate's is.
    //
    // A point's first fire is estimated from the prefixed points measured
    // before it ([`grown`] over their kind): the largest it is no less work
    // than (the point before it at these rows, or these keys at the rows
    // before: fewer rows run slower, so it is estimated long), the first
    // point from the probe. Never from the square curve: a few rows against
    // a prefix run several times slower than a square of the keys (128 rows
    // ran at an eighth of it on an RTX). An estimate grown from another
    // shape moves no admission's worst ratio: its error is that shape's,
    // not the device's.
    // A row's curve stops at the first point the bound refuses, as a
    // kernel's does (`curve_of`): more keys or more rows take no less.
    let mut prefixed: Vec<(u32, Vec<(u32, f64)>)> = Vec::new();
    if max_submit.is_some() && stated.attn_tflops_by_rows_keys.is_none() {
        for rows in PREFIX_ROWS {
            let mut curve: Vec<(u32, f64)> = Vec::new();
            for keys in engine::rates::ATTN_KEYS.into_iter().filter(|&k| k > rows) {
                let shape = engine::rates::AttnShape {
                    rows,
                    keys,
                    ..engine::rates::ATTN_SHAPE
                };
                let rate = repeated(|| {
                    let rate = attention(
                        &mut rig,
                        tactics,
                        tier,
                        timer,
                        shape,
                        launch,
                        Kind::Prefixed,
                    );
                    rig.release();
                    rate
                });
                let Some(rate) = dropped(&rig, rate, say)? else {
                    break;
                };
                say(&format!(
                    "[rates]: attention {rows} rows x {keys} keys d 128: {:.2} TFLOP/s",
                    rate / 1e3
                ));
                curve.push((keys, rate / 1e3));
            }
            if !curve.is_empty() {
                prefixed.push((rows, curve));
            }
        }
        // Not one point fit the bound: the fewest keys past the narrowest
        // rows, at fewer rows (halved) until one fire fits; a step of one
        // row against them past the bound refuses by name (no prompt step
        // could run under it).
        if prefixed.is_empty() {
            let keys = engine::rates::ATTN_KEYS[0];
            let mut rows = PREFIX_ROWS[0];
            loop {
                rows = (rows / 2).max(1);
                let shape = engine::rates::AttnShape {
                    rows,
                    keys,
                    ..engine::rates::ATTN_SHAPE
                };
                let rate = repeated(|| {
                    let rate = attention(
                        &mut rig,
                        tactics,
                        tier,
                        timer,
                        shape,
                        launch,
                        Kind::Prefixed,
                    );
                    rig.release();
                    rate
                });
                if let Some(rate) = dropped(&rig, rate, say)? {
                    say(&format!(
                        "[rates]: attention {rows} rows x {keys} keys d 128: {:.2} TFLOP/s (the \
                         fewest rows whose fire fits the bound)",
                        rate / 1e3
                    ));
                    prefixed.push((rows, vec![(keys, rate / 1e3)]));
                    break;
                }
                if rows == 1 {
                    return Err(Fault::Device {
                        call: "rates",
                        why: format!(
                            "one row's attention against {keys} keys is past the device's bound \
                             on one submission: no prompt step can run under it"
                        ),
                    });
                }
            }
        }
    }
    let prefixed_fired = rig.take_fired();
    step_fired.extend(prefixed_fired.iter().copied());
    let elementwise = if want.elementwise {
        let gbps = repeated(|| {
            let gbps = elementwise(&mut rig, timer, launch);
            rig.release();
            gbps
        })?;
        say(&format!("[rates]: per-row ops: {gbps:.0} GB/s"));
        Some(gbps)
    } else {
        None
    };
    let elementwise_fired = rig.take_fired();
    step_fired.extend(elementwise_fired.iter().copied());
    let scan = if want.scan {
        Some(scan_within(&mut rig, tactics, timer, launch, say)?)
    } else {
        None
    };
    // Under a stated submit bound, the clock the device settles at under
    // sustained load: the warm-up's cube rotated again once every rate is
    // timed (the device loaded for the whole step, past the drop a phone's
    // clock takes some seconds in), its settled rate against the warm-up's.
    // The margin a serving step is held to carries the drop as measured
    // (an Adreno 840 halved its clock ~10 s into load). Where the platform
    // states the GPU's clock table, a serving step is held at its floor
    // (`crate::clock`), which covers any drop: the margin is the estimate's
    // error alone, and the sustained drop is not measured.
    let scan_fired = rig.take_fired();
    step_fired.extend(scan_fired.iter().copied());
    let clock_table = crate::clock::ClockRange::read().ok();
    if let (Some(_), Some(range)) = (max_submit, &clock_table) {
        say(&format!(
            "[rates]: sustained: not measured; a serving step is held at the floor of the GPU's \
             clock table {range} (x{:.2})",
            range.floor_factor()
        ));
    }
    let sustained = if max_submit.is_some() && clock_table.is_none() {
        // Rotated as the warm-up rotated it: each fire its own submission,
        // admitted at the worst ratio the warm-up's walk measured (not at
        // the step's own worst since, which its micro-shapes' host latency
        // inflates and would refuse the cube that fit then: x4.59 refused an
        // 87 ms fire under a forced 300 ms bound on an RTX).
        let step_admission = rig.admission.get();
        rig.admission.set(engine::rates::Admission {
            settled: true,
            worst: walk_worst.max(1.0),
            ..step_admission
        });
        let rotated = warm_at(
            &mut rig,
            side,
            &|rows| tactics.gemm(tier, rows),
            timer,
            walk_worst,
        );
        rig.admission.set(step_admission);
        match rotated? {
            Some(hot) => {
                let drop = (warmed.rate / hot.rate.max(f64::MIN_POSITIVE)).max(1.0);
                say(&format!(
                    "[rates]: sustained: the {side}^3 matmul settled at {:.2} TFLOP/s after the \
                     step, {:.2} at its start: x{drop:.2}",
                    hot.rate / 1e3,
                    warmed.rate / 1e3
                ));
                drop
            }
            None => {
                // The cube's fire no longer fits the bound at the walk's
                // ratio once loaded: not rotated whole, the drop is read off
                // the fires it did run (the slowest against the fastest).
                let drop = rig.swing.get().max(1.0);
                tracing::warn!(
                    side,
                    drop,
                    "rates: sustained: the warm-up's cube no longer fits the bound once loaded; \
                     its drop is the swing of the fires it ran"
                );
                drop
            }
        }
    } else {
        1.0
    };
    step_fired.extend(rig.take_fired());
    let tflops = |curve: Vec<(u32, f64)>| -> Vec<(u32, f64)> {
        curve.into_iter().map(|(x, r)| (x, r / 1e3)).collect()
    };
    let mut rates = Rates {
        landing: None,
        load_width: None,
        stale: None,
        launch_ns: want.launch.then_some(launch_ns),
        // A curve with no point measured is stated as none: the load is
        // refused by its key (`rates_at_load`), never weighed by nothing.
        gemm_tflops_by_rows: (!gemm.is_empty()).then(|| {
            gemm.into_iter()
                .map(|(w, curve)| (w, tflops(curve)))
                .collect()
        }),
        attn_tflops_by_keys: (!attn.is_empty()).then(|| tflops(attn)),
        hbm_gbps: want.launch.then_some(bytes_per_ns),
        cached_gbps: None,
        elementwise_gbps: elementwise,
        scan_gelems: scan,
        few_row_tflops: None,
        product_ns: None,
        workgroups_per_core: None,
        kernel_tflops_by_shape: None,
        dense_settled: Vec::new(),
        resident_workgroups: resident,
        // How much slower the device ran than its own measurements said:
        // like frames' drift, the clock's swing and its sustained drop
        // (`Rig::drift`, `sustained`): the margin a serving step's estimate
        // is held to the bound at (`crate::submit`), measured where the
        // platform states one.
        submit_worst_ratio: max_submit
            .map(|_| rig.drift.get().max(rig.swing.get()).max(sustained).max(1.0)),
        attn_tflops_by_rows_keys: (!prefixed.is_empty()).then_some(prefixed),
        measured_on: None,
        measured_by: None,
    };
    // Each rate measured, keyed by its method and the kernels it fired
    // (`crate::stored`); the device it was measured on is the backend's to
    // state.
    let mut keyed = Vec::new();
    for (key, fired) in [
        ("launch_ns", &launch_fired),
        ("hbm_gbps", &launch_fired),
        ("gemm_tflops_by_rows", &gemm_fired),
        ("attn_tflops_by_keys", &attn_fired),
        ("elementwise_gbps", &elementwise_fired),
        ("scan_gelems", &scan_fired),
        (profile::fire::RESIDENT_WORKGROUPS, &resident_fired),
        (profile::fire::ATTN_BY_ROWS_KEYS, &prefixed_fired),
        (profile::fire::SUBMIT_WORST_RATIO, &step_fired),
    ] {
        if rates.states(key) {
            let by = crate::stored::by(B::TARGET, key, fired)?;
            tracing::info!("rates: measured `{key}`, stored under its key: {}", by.said());
            keyed.push((key.to_string(), by));
        }
    }
    rates.measured_by = (!keyed.is_empty()).then_some(keyed);
    // A rate that settled at nothing is refused by name, never saved or
    // divided by.
    if let Some((key, why)) = engine::rates::unmeasured(&rates) {
        return Err(Fault::Device {
            call: "rates",
            why: format!("measured `[rates] {key}` {why}"),
        });
    }
    Ok(engine::rates::Measured {
        rates,
        unsettled: rig.unsettled.get(),
        throttled: {
            drop(rig);
            throttled(clock_before, clock_read::<B>(device, reference)?)
        },
    })
}

/// The version of [`elementwise`]'s method (`[rates] elementwise_gbps`'s
/// stored key, `crate::stored`): bumped only when how it measures changes.
pub const ELEMENTWISE_METHOD: u32 = 1;

/// The per-row ops' bandwidth: the backend's rmsnorm, then its swiglu, at
/// `engine::rates::ELEMENTWISE_SHAPE`, each fire the next of
/// `engine::rates::elementwise_sets` sets (at most [`ATTN_SETS_MOST`]),
/// their bytes over their time.
fn elementwise<B: Serve>(rig: &mut Rig<'_, B>, timer: Timer<'_, B>, launch: f64) -> Result<f64> {
    let (rows, width) = engine::rates::ELEMENTWISE_SHAPE;
    let weight = rig.operand(1, width, Dtype::Bf16)?;
    let sets = (0..engine::rates::elementwise_sets().min(ATTN_SETS_MOST))
        .map(|_| {
            Ok((
                rig.operand(rows, width, Dtype::Bf16)?,
                rig.zeros(rows, width, Dtype::Bf16)?,
                rig.operand(rows, 2 * width, Dtype::Bf16)?,
                rig.zeros(rows, width, Dtype::Bf16)?,
            ))
        })
        .collect::<Result<Vec<(Tensor, Tensor, Tensor, Tensor)>>>()?;
    let next = std::cell::Cell::new(0usize);
    rig.step(
        format!("per-row ops {}x{}", rows, width),
        Work {
            kind: Kind::Memory,
            flops: 0.0,
            bytes: engine::rates::elementwise_bytes(),
        },
    );
    let ns = rig.each(timer, &Touched::default(), sets.len(), &|sink| {
        let (x, normed, packed, y) = sets[next.get() % sets.len()];
        next.set(next.get() + 1);
        kernels_slang::elemwise::norm::rmsnorm(sink, x, weight, 1e-6, normed)
            .map_err(fault("rates: rmsnorm"))?;
        kernels_slang::linear::glu::swiglu(sink, packed, width, y).map_err(fault("rates: swiglu"))
    })?;
    engine::rates::rate_from(
        "per-row ops",
        ns as f64,
        2,
        launch,
        engine::rates::elementwise_bytes(),
    )
    .map_err(|why| Fault::Device {
        call: IMPLAUSIBLE,
        why,
    })
}

/// The version of the method `[rates] submit_worst_ratio` is measured by
/// (the warm-up's rotations, [`warm`], the step's drift and the sustained
/// rotation after it; its stored key, `crate::stored`): bumped only when
/// how it measures changes.
pub const SUBMIT_METHOD: u32 = 1;

/// The device warmed ([`engine::rates::warm_up`]): rotations of a cube
/// matmul over its planes, each one frame timed by the host's clock around
/// its one submit, until the rate stops falling. The cube's side is
/// [`engine::rates::warm_side`]'s: walked up by doubling, a real fire timed
/// at each size, while the next one's fire, estimated by the growth the
/// last two measured at the worst ratio of a time to its estimate seen, fits
/// the device's bound on a submission (none stated:
/// [`engine::rates::WARM_SHAPE`]'s). Each rotation's frames hold as many
/// fires as their estimate (the last fire's time, at the worst ratio seen)
/// fits under the bound; where not one does (a clock sinking as it warms),
/// the cube halves and the warm-up goes on at it.
fn warm<B: Serve>(
    rig: &mut Rig<'_, B>,
    tactic_at: &dyn Fn(u32) -> kernels_slang::linear::gemm::Tactic,
    timer: Timer<'_, B>,
) -> Result<(u32, engine::rates::Warmed, f64)> {
    // The cube's side, walked up by real fires under the device's bound on
    // one submission (`engine::rates::warm_side`): each size's second fire
    // timed (its first builds and touches).
    let max_submit = rig.max_submit;
    let (mut side, worst) = engine::rates::warm_side(max_submit, 1.0, |p, estimate| {
        let act = rig.operand(p, p, Dtype::Bf16)?;
        let w = rig.operand(p, p, Dtype::Bf16)?;
        let y = rig.zeros(p, p, Dtype::Bf16)?;
        let tactic = tactic_at(p);
        let fire = |sink: &kernels_slang::Ctx<'_>| {
            kernels_slang::linear::gemm::matmul(sink, tactic, act, w, y)
                .map_err(fault("rates: matmul"))
        };
        let nothing = |_: &mut B::Frame| Ok(());
        // The walk admitted this size under its own estimate and worst
        // ratio (`warm_side`); its frames carry that estimate to the trace.
        tracing::debug!(
            side = p,
            walk_estimate_ms = ?estimate.map(|e| e / 1e6),
            "warm-up walk: a size admitted"
        );
        // Its first fire estimated from the sizes before it (the first size
        // from the probe, [`grown`]); one past the bound is refused by name
        // unsubmitted and the walk stops at the size before.
        rig.step(
            format!("warm-up walk {p}^3 (first fire)"),
            Work::matmul(p, p, p),
        );
        let timed = rig.frame(timer, 1, &fire, &nothing).and_then(|_| {
            rig.restep(format!("warm-up walk {p}^3"));
            rig.frame(timer, 1, &fire, &nothing)
        });
        rig.release();
        match timed {
            Ok((_, ns)) => Ok::<Option<f64>, Fault>(Some(ns as f64)),
            Err(e) => match rig.past_bound.borrow_mut().take() {
                Some(step) => {
                    tracing::debug!(step = %step, why = %e, "warm-up walk: past the bound");
                    Ok(None)
                }
                None => Err(e),
            },
        }
    })?;
    tracing::debug!(side, walk_worst = worst, "warm-up walk: the cube");
    loop {
        match warm_at(rig, side, tactic_at, timer, worst)? {
            Some(warmed) => {
                // The clock settled (the rotations agreed, `settle`'s rule):
                // from here every frame's time against its estimate counts.
                let mut admission = rig.admission.get();
                admission.settled = true;
                // The clock's swing the rotations measured: a settled frame
                // is never admitted nearer the bound than that.
                admission.worst = admission.worst.max(rig.swing.get());
                rig.admission.set(admission);
                return Ok((side, warmed, worst));
            }
            None if side > engine::rates::WARM_PROBE / 2 => side /= 2,
            None => {
                return Err(Fault::Device {
                    call: "rates",
                    why: format!(
                        "warm-up: not one {side}^3 matmul is estimated within the device's bound \
                         on one submission at the worst ratio its rotations' times ran against \
                         their estimates"
                    ),
                });
            }
        }
    }
}

/// [`warm`]'s rotations at a cube of `side`, or `None` where a fire of it no
/// longer fits the device's bound at the worst ratio seen.
fn warm_at<B: Serve>(
    rig: &mut Rig<'_, B>,
    side: u32,
    tactic_at: &dyn Fn(u32) -> kernels_slang::linear::gemm::Tactic,
    timer: Timer<'_, B>,
    walk_worst: f64,
) -> Result<Option<engine::rates::Warmed>> {
    let (m, n, k) = (side, side, side);
    let act = rig.operand(m, k, Dtype::Bf16)?;
    let planes = rig.planes_most.map_or(weights_at(n, k), |most| weights_at(n, k).min(most));
    let weights = (0..planes)
        .map(|_| rig.operand(n, k, Dtype::Bf16))
        .collect::<Result<Vec<Tensor>>>()?;
    let y = rig.zeros(m, n, Dtype::Bf16)?;
    let tactic = tactic_at(m);
    let next = std::cell::Cell::new(0usize);
    let fire = |sink: &kernels_slang::Ctx<'_>| {
        let w = weights[next.get() % weights.len()];
        next.set(next.get() + 1);
        kernels_slang::linear::gemm::matmul(sink, tactic, act, w, y).map_err(fault("rates: matmul"))
    };
    let planes = u32::try_from(weights.len()).unwrap_or(u32::MAX);
    let nothing = |_: &mut B::Frame| Ok(());
    // The first fire estimated from the fires before it ([`grown`]): the
    // walk's own of this cube, at the clock the probe reads now (the
    // sustained rotation's after every rate, the clock dropped by however
    // long the host-bound shapes before it left the device idle). Past the
    // bound it is never submitted: the cube halves ([`warm`]).
    rig.step(
        format!("warm-up {side}^3 (first fire)"),
        Work::matmul(m, n, k),
    );
    if let Err(e) = rig.frame(timer, 1, &fire, &nothing) {
        rig.release();
        return match rig.past_bound.borrow_mut().take() {
            Some(step) => {
                tracing::debug!(step = %step, why = %e, "warm-up: past the bound");
                Ok(None)
            }
            None => Err(e),
        };
    }
    rig.restep(format!("warm-up {side}^3 rotation"));
    // The fires the next frame holds: as many as fit under the device's
    // bound at the last fire's time and the worst ratio the rotations' own
    // frames ran against their estimates (the clock sinking as it warms),
    // all of a rotation's where it states none; 0 where not one fits.
    // An unsettled clock swings (the Adreno 840's fire ran 182 to 318 ms
    // within seconds), so a frame is sized from the slowest fire this
    // rotation has measured, not the last, at the worst ratio the walk
    // measured a fire's time against its estimate: both measurements, no
    // chosen margin. Sized from the last fire at a ratio of 1, a frame
    // estimated at 1824 ms took 2001 ms and the next lost the device.
    // Each frame follows the device idle and is re-timed first: the slowest
    // fire is kept at the clock the first probe read and read at the
    // clock's ratio now.
    let rotations = std::cell::Cell::new(engine::rates::Admission {
        settled: true,
        worst: walk_worst.max(1.0),
        ..engine::rates::Admission::new(rig.max_submit)
    });
    let slowest = std::cell::Cell::new(rig.per_fire.get());
    let per_fire = || match (slowest.get(), rig.per_fire.get()) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    };
    // The fires as timed, the clock's own swing and drift in them (what the
    // margin a serving step is held to carries, unscaled).
    let raw = |ns: f64| ns * rig.ratio();
    let slowest_raw = std::cell::Cell::new(rig.per_fire.get().map(raw));
    let fastest_raw = std::cell::Cell::new(None::<f64>);
    // Queued, each fire is its own submission: the whole rotation where one
    // fire fits, none where it does not.
    let fits = |left: u32| -> Result<u32> {
        rig.retime()?;
        let now = per_fire().map(|p| p * rig.ratio());
        let bound = rotations.get().bound;
        let worst = rotations.get().worst;
        Ok(if rig.queue.get() {
            rotation_fires(bound, now, worst, 1) * left
        } else {
            rotation_fires(bound, now, worst, left)
        })
    };
    let shrunk = std::cell::Cell::new(false);
    let warmed = engine::rates::warm_up(|| {
        let mut wall = 0u64;
        let mut left = planes;
        while left > 0 {
            let count = fits(left)?;
            if count == 0 {
                // Not one fire fits: the rotations stop here, and the cube
                // halves ([`warm`]).
                shrunk.set(true);
                return Err(Fault::Device {
                    call: "rates",
                    why: format!("warm-up: a {side}^3 fire no longer fits the bound"),
                });
            }
            let estimate = per_fire().map(|p| p * rig.ratio() * f64::from(count));
            let estimate_raw = slowest_raw.get().map(|p| p * f64::from(count));
            let took = rig.frame(timer, count, &fire, &nothing)?.1;
            slowest.set(per_fire());
            let this = took as f64 / f64::from(count.max(1));
            slowest_raw.set(Some(slowest_raw.get().map_or(this, |s: f64| s.max(this))));
            fastest_raw.set(Some(fastest_raw.get().map_or(this, |f: f64| f.min(this))));
            if let (Some(fast), Some(slow)) = (fastest_raw.get(), slowest_raw.get()) {
                rig.swing.set(rig.swing.get().max(slow / fast.max(1.0)));
            }
            let mut r = rotations.get();
            r.landed(estimate, took as f64, true);
            rotations.set(r);
            if let Some(e) = estimate_raw.filter(|&e| e > 0.0) {
                rig.drift.set(rig.drift.get().max(took as f64 / e));
            }
            wall += took;
            left -= count;
        }
        Ok::<f64, Fault>(gemm_flops(m, n, k) * f64::from(planes) / wall.max(1) as f64)
    });
    rig.release();
    match warmed {
        Err(_) if shrunk.get() => Ok(None),
        warmed => warmed.map(Some),
    }
}

/// The version of [`matmul`]'s method as `[rates] gemm_tflops_by_rows`
/// measures it (its stored key, `crate::stored`): bumped only when how it
/// measures changes.
pub const GEMM_METHOD: u32 = 1;

/// The matmul rate at `(m, n, k)`, each fire reading the next of its
/// weight planes, past the last-level cache.
fn matmul<B: Serve>(
    rig: &mut Rig<'_, B>,
    tactic: kernels_slang::linear::gemm::Tactic,
    timer: Timer<'_, B>,
    (m, n, k): (u32, u32, u32),
    launch: f64,
) -> Result<f64> {
    let act = rig.operand(m, k, Dtype::Bf16)?;
    let planes = rig.planes_most.map_or(weights_at(n, k), |most| weights_at(n, k).min(most));
    let weights = (0..planes)
        .map(|_| rig.operand(n, k, Dtype::Bf16))
        .collect::<Result<Vec<Tensor>>>()?;
    let y = rig.zeros(m, n, Dtype::Bf16)?;
    matmul_on(rig, tactic, timer, (act, &weights, &[y]), None, launch)?.map_err(|why| {
        Fault::Device {
            call: IMPLAUSIBLE,
            why,
        }
    })
}

/// [`matmul`] over operands laid already: `act` (its rows the matmul's),
/// the weight planes its fires rotate through, `y`; `touched` the planes'
/// one untimed read, shared by every point over them. A fire timed below
/// its launches comes back as the inner error: no measurement, though the
/// device did nothing wrong.
fn matmul_on<B: Serve>(
    rig: &mut Rig<'_, B>,
    tactic: kernels_slang::linear::gemm::Tactic,
    timer: Timer<'_, B>,
    (act, weights, ys): (Tensor, &[Tensor], &[Tensor]),
    touched: Option<&Touched>,
    launch: f64,
) -> Result<std::result::Result<f64, String>> {
    let (m, n, k) = (act.rows, ys[0].width, act.width);
    let next = std::cell::Cell::new(0usize);
    rig.step(format!("matmul {m}x{n}x{k}"), Work::matmul(m, n, k));
    let fire = |sink: &kernels_slang::Ctx<'_>| {
        let at = next.get();
        next.set(at + 1);
        let (w, y) = (weights[at % weights.len()], ys[at % ys.len()]);
        kernels_slang::linear::gemm::matmul(sink, tactic, act, w, y).map_err(fault("rates: matmul"))
    };
    // A curve's point (its planes' touch shared) to a small device budget;
    // a rate of the device's own, settled over repeated rounds.
    let ns = match touched {
        Some(touched) => rig.budgeted(timer, touched, weights.len(), &fire)?,
        None => rig.each(timer, &Touched::default(), weights.len(), &fire)?,
    };
    Ok(engine::rates::rate_from(
        &format!("matmul {m}x{n}x{k}"),
        ns as f64,
        engine::rates::GEMM_DISPATCHES,
        launch,
        gemm_flops(m, n, k),
    ))
}

/// The version of [`resident_workgroups`]'s method (its stored key,
/// `crate::stored`): bumped only when how it measures changes.
pub const RESIDENT_METHOD: u32 = 1;

/// The workgroups the device holds at once: the matmul `curve`'s peak
/// rows (its kernel compute-bound there, at width `w`), the same kernel
/// shrunk to one workgroup over the same depth, its time against the same
/// work at the peak's rate. One workgroup runs at `1 / S` of the kernel's
/// full rate (`engine::rates::workgroups_per_core`, the device as one).
fn resident_workgroups<B: Serve, T: Tactics + ?Sized>(
    rig: &mut Rig<'_, B>,
    tactics: &T,
    tier: kernels_slang::Tier,
    timer: Timer<'_, B>,
    curve: &[(u32, f64)],
    w: u32,
    launch: f64,
) -> Result<f64> {
    let refused = |why: String| Fault::Device { call: "rates", why };
    let &(peak_rows, peak) = curve
        .iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .ok_or_else(|| refused("the matmul curve has no point".into()))?;
    let tactic = tactics.gemm(tier, peak_rows);
    let groups = |m: u32, n: u32| {
        kernels_slang::linear::gemm::workgroups(B::TARGET, tactic, m, n, w)
            .map_err(fault("rates: resident workgroups"))
    };
    let (mut m, mut n) = (peak_rows, w);
    while groups(m, n)? > 1 {
        let (wide, tall) = (groups(m, n.div_ceil(2))?, groups(m.div_ceil(2), n)?);
        if n > 1 && wide < groups(m, n)? {
            n = n.div_ceil(2);
        } else if m > 1 && tall < groups(m, n)? {
            m = m.div_ceil(2);
        } else {
            return Err(refused(format!(
                "the {peak_rows}-row matmul's kernel launches {} workgroups at {m} x {n} x {w}, and no narrower shape of it launches fewer",
                groups(m, n)?
            )));
        }
    }
    let one = matmul(rig, tactic, timer, (m, n, w), launch)?;
    let flops = gemm_flops(m, n, w);
    engine::rates::workgroups_per_core(1, 1, flops / one, flops / peak).ok_or_else(|| {
        refused(format!(
            "one workgroup of the {peak_rows}-row matmul ({m} x {n} x {w}) ran at {one} against {peak} at full width: no count of workgroups is made of it"
        ))
    })
}

/// The version of [`attention`]'s method (`[rates] attn_tflops_by_keys`'
/// and `attn_tflops_by_rows_keys`' stored key, `crate::stored`): bumped
/// only when how it measures changes.
pub const ATTN_METHOD: u32 = 1;

/// The prefill attention's rate at `s`: one sequence of `s.rows` rows
/// attending itself, causal, its kv laid in pages of [`PAGE`], through the
/// kernel a served fire takes at that shape (the tactic at its head width,
/// tiled or walked by [`Tactics::sdpa_tile_rows_per_request`]).
fn attention<B: Serve, T: Tactics + ?Sized>(
    rig: &mut Rig<'_, B>,
    tactics: &T,
    tier: kernels_slang::Tier,
    timer: Timer<'_, B>,
    s: engine::rates::AttnShape,
    launch: f64,
    kind: Kind,
) -> Result<f64> {
    let tactic = tactics.sdpa(tier, s.head_dim);
    // Tiled where a request's rows reach the profile's tile rows, as a
    // served prefill takes it (one request here).
    let tiled = s.rows >= tactics.sdpa_tile_rows_per_request();
    let kv_width = s.kv_heads * s.head_dim;
    let o = rig.zeros(s.rows, s.q_heads * s.head_dim, Dtype::Bf16)?;
    let pages = s.keys.div_ceil(PAGE);
    // The sequence's pages scattered over the pool as a served one's are.
    let indices: Vec<u32> = engine::rates::attn_page_order(pages);
    let page_indices = rig.words(&indices, pages, 1, Dtype::U32)?;
    let page_indptr = rig.words(&[0, pages], 2, 1, Dtype::U32)?;
    // Each fire reads the next of these q / k / v sets, past the last-level
    // cache, as a served fire reads each layer's kv cold.
    let most = rig.planes_most.map_or(ATTN_SETS_MOST, |most| most.min(ATTN_SETS_MOST));
    let sets = (0..engine::rates::attn_sets_at(s, PAGE).min(most))
        .map(|_| {
            let q = rig.operand(s.rows, s.q_heads * s.head_dim, Dtype::Bf16)?;
            let keys = rig.operand(s.keys, kv_width, Dtype::Bf16)?;
            let values = rig.operand(s.keys, kv_width, Dtype::Bf16)?;
            let pool = KvPool::planes(
                keys,
                values,
                page_indices,
                page_indptr,
                PAGE as i32,
                u64::from(kv_width),
                u64::from(s.head_dim),
            );
            Ok((q, pool))
        })
        .collect::<Result<Vec<(Tensor, KvPool)>>>()?;
    let next = std::cell::Cell::new(0usize);
    // The rows' own positions, all of request 0, no row masked: the last
    // `rows` of the `keys`, each attending every key held before it and
    // its own rows up to itself (a square shape attends itself).
    let positions: Vec<u32> = (s.keys.saturating_sub(s.rows)..s.keys).collect();
    let tables = kernels_slang::attn::sdpa::Tables {
        positions: rig.words(&positions, s.rows, 1, Dtype::I32)?,
        request_of_token: rig.zeros(s.rows, 1, Dtype::I32)?,
        mask: rig.zeros(1, 4, Dtype::U8)?,
        mask_words: rig.words(&crate::mask::none(s.rows as usize), s.rows, 2, Dtype::U32)?,
        row0: 0,
        rows: s.rows,
    };
    let scale = 1.0 / (s.head_dim as f32).sqrt();
    // A new shape's first fire is estimated before it is submitted (grown
    // from the shapes of its kind measured before, by the work), so one past
    // the device's bound is refused by name unsubmitted, never run to find
    // out.
    rig.step(
        format!("attention {}x{} d {}", s.rows, s.keys, s.head_dim),
        Work {
            kind,
            flops: attn_flops(s),
            bytes: engine::rates::attn_bytes(s),
        },
    );
    let ns = rig.each(timer, &Touched::default(), sets.len(), &|sink| {
        let (q, pool) = &sets[next.get() % sets.len()];
        let (q, pool) = (*q, pool);
        next.set(next.get() + 1);
        if tiled {
            kernels_slang::attn::sdpa::prefill(
                sink, tactic, q, &tables, pool, None, s.head_dim, scale, o, None,
            )
        } else {
            kernels_slang::attn::sdpa::decode(
                sink, tactic, q, &tables, pool, None, s.head_dim, scale, o, None, None,
            )
        }
        .map_err(fault("rates: attention"))
    })?;
    engine::rates::rate_from(
        &format!("attention {}x{} d {}", s.rows, s.keys, s.head_dim),
        ns as f64,
        engine::rates::ATTN_DISPATCHES,
        launch,
        attn_flops(s),
    )
    .map_err(|why| Fault::Device {
        call: IMPLAUSIBLE,
        why,
    })
}

/// The version of [`scan`]'s method (`[rates] scan_gelems`' stored key,
/// `crate::stored`): bumped only when how it measures changes.
pub const SCAN_METHOD: u32 = 1;

/// The recurrent rate at [`SCAN_SHAPE`]: the causal conv over the scan's
/// `qkv` channels, then the chunked gated-delta scan, one sequence on one
/// state slot.
fn scan<B: Serve>(
    rig: &mut Rig<'_, B>,
    tactic: kernels_slang::ssm::gdn_recurrent::Tactic,
    timer: Timer<'_, B>,
    s: engine::rates::ScanShape,
    launch: f64,
) -> Result<f64> {
    let channels = s.channels();
    let qkv = rig.zeros(s.tokens, channels, Dtype::Bf16)?;
    let indptr = rig.words(&[0, s.tokens], 2, 1, Dtype::I32)?;
    let weight = rig.operand(channels, s.conv_width, Dtype::Bf16)?;
    let slots = rig.zeros(s.tokens, 1, Dtype::U32)?;
    let gates = rig.zeros(s.tokens, 2 * s.v_heads, Dtype::F32)?;
    let y = rig.zeros(s.tokens, s.v_heads * s.v_dim, Dtype::F32)?;
    // Each fire reads the next of these input rows and states, past the
    // last-level cache, as a served layer's.
    let sets = (0..engine::rates::scan_sets_at(s))
        .map(|_| {
            let raw = rig.operand(s.tokens, channels, Dtype::Bf16)?;
            let conv_bank = rig.operand(1, s.conv_width * channels, Dtype::F32)?;
            let scan_bank = rig.operand(1, s.v_heads * s.k_dim * s.v_dim, Dtype::F32)?;
            Ok((
                raw,
                kernels_slang::ssm::StatePool::bank(conv_bank, slots),
                kernels_slang::ssm::StatePool::bank(scan_bank, slots),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let csr = kernels_slang::ssm::Csr {
        table: indptr,
        at: 0,
        lanes: indptr.rows.saturating_sub(1),
    };
    let next = std::cell::Cell::new(0usize);
    let rows = f64::from(s.tokens) * f64::from(channels) * 2.0;
    let state = 4.0 * f64::from(s.conv_width * channels + s.v_heads * s.k_dim * s.v_dim);
    let out = f64::from(s.tokens) * f64::from(s.v_heads * s.v_dim) * 4.0;
    rig.step(
        format!("scan {} tokens", s.tokens),
        Work {
            kind: Kind::Scan,
            flops: 0.0,
            bytes: 2.0 * rows + state + out,
        },
    );
    let ns = rig.each(timer, &Touched::default(), sets.len(), &|sink| {
        let (raw, conv_state, scan_state) = &sets[next.get() % sets.len()];
        let raw = *raw;
        next.set(next.get() + 1);
        kernels_slang::ssm::conv1d::conv1d_chunked(
            sink,
            raw,
            csr,
            weight,
            conv_state,
            s.conv_width,
            1,
            qkv,
            0,
        )
        .map_err(fault("rates: conv"))?;
        kernels_slang::ssm::gdn_recurrent::gated_delta_chunked(
            sink,
            tactic,
            qkv,
            csr,
            gates,
            scan_state,
            s.k_heads,
            s.v_heads,
            s.k_dim,
            s.v_dim,
            y,
            kernels_slang::ssm::RowSpan::whole(qkv),
        )
        .map_err(fault("rates: scan"))
    })?;
    engine::rates::rate_from(
        &format!("scan of {} tokens", s.tokens),
        ns as f64,
        engine::rates::SCAN_DISPATCHES,
        launch,
        scan_elements(s),
    )
    .map_err(|why| Fault::Device {
        call: IMPLAUSIBLE,
        why,
    })
}

/// The scan's rate, timed at [`SCAN_SHAPE`] where one fire of it fits the
/// device's bound on one submission, else at the most tokens that do (halved
/// from it, refused unsubmitted by its estimate each time it does not): the
/// rate is elements a ns, which a shorter sequence of the same heads measures
/// alike, so a slow device's scan is sized to its bound, never refused.
fn scan_within<B: Serve, T: Tactics + ?Sized>(
    rig: &mut Rig<'_, B>,
    tactics: &T,
    timer: Timer<'_, B>,
    launch: f64,
    say: &mut dyn FnMut(&str),
) -> Result<f64> {
    let mut shape = SCAN_SHAPE;
    loop {
        let rate = repeated(|| {
            let gelems = scan(rig, tactics.recurrent(), timer, shape, launch);
            rig.release();
            gelems
        });
        let past = rig.past_bound.borrow_mut().take().is_some();
        match rate {
            Err(_) if past && shape.tokens > 1 => {
                let tokens = shape.tokens / 2;
                say(&format!(
                    "[rates]: scan of {} tokens is past the bound: timed at {tokens}",
                    shape.tokens
                ));
                tracing::info!(
                    tokens = shape.tokens,
                    next = tokens,
                    "rates: the scan's fire is past the device's bound on one submission: timed \
                     at half its tokens (its rate a ns is the same work's)"
                );
                shape.tokens = tokens;
            }
            rate => return rate,
        }
    }
}

/// The version of [`copy_and_launch`]'s method (`[rates] launch_ns`' and
/// `hbm_gbps`' stored key, `crate::stored`): bumped only when how it
/// measures changes.
pub const LAUNCH_METHOD: u32 = 1;

/// `(bytes a ns, a dispatch's ns)`: a copy of device memory, and the
/// narrowest matmul back to back in one frame against one alone.
fn copy_and_launch<B: Serve>(
    rig: &mut Rig<'_, B>,
    tactic: kernels_slang::linear::gemm::Tactic,
    timer: Timer<'_, B>,
    say: &mut dyn FnMut(&str),
) -> Result<(f64, u64)> {
    let act = rig.zeros(LAUNCH, LAUNCH, Dtype::Bf16)?;
    let w = rig.zeros(LAUNCH, LAUNCH, Dtype::Bf16)?;
    let y = rig.zeros(LAUNCH, LAUNCH, Dtype::Bf16)?;
    let tiny = |sink: &kernels_slang::Ctx<'_>| {
        kernels_slang::linear::gemm::matmul(sink, tactic, act, w, y).map_err(fault("rates: matmul"))
    };
    let nothing = |_: &mut B::Frame| Ok(());
    // The launch is what a dispatch adds inside one submission: its chains
    // are one submission each (their tiny fires far inside any bound).
    let queued = rig.queue.replace(false);
    let tiny_work = Work::matmul(LAUNCH, LAUNCH, LAUNCH);
    rig.step(format!("launch: one {LAUNCH}^3 matmul"), tiny_work);
    rig.frame(timer, 1, &tiny, &nothing)?;
    let single = median(
        (0..ROUNDS)
            .map(|_| rig.frame(timer, 1, &tiny, &nothing).map(|(_, w)| w))
            .collect::<Result<Vec<u64>>>()?,
    );
    // The launch: a chain of `n` dependent tiny matmuls in one frame (each
    // reads the one before's output) against one matmul of `n` times the
    // rows, at each chain length of the lattice.
    // Each chain the device's bound admits (a chain is one submission): a
    // chain past it is skipped by name, the shorter ones read; none at all
    // refuses by name.
    let mut gaps = Vec::with_capacity(engine::rates::LAUNCH_CHAINS.len());
    for n in engine::rates::LAUNCH_CHAINS {
        match launch_gap(rig, tactic, timer, (act, w, y), n) {
            Ok(gap) => gaps.push((n, gap)),
            Err(why) if rig.past_bound.borrow_mut().take().is_some() && !gaps.is_empty() => {
                say(&format!("[rates]: launch chain of {n} past the bound, skipped ({why})"));
            }
            Err(why) => return Err(why),
        }
    }
    say(&format!(
        "[rates]: launch: {}",
        gaps.iter()
            .map(|(n, ns)| format!("{ns:.0} ns a dispatch over a chain of {n}"))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    // The longest chain's: its gap the least moved by the frame's submit.
    let launch = gaps.last().map_or(0.0, |&(_, ns)| ns).round() as u64;
    rig.queue.set(queued);
    copy_after(rig, timer, &tiny, single, launch).map(|bytes| (bytes, launch))
}

/// A dispatch's launch, ns: a chain of `n` dependent [`LAUNCH`]-cubed
/// matmuls under `tactic` in one frame (each reading the one before's
/// output, over `act`, `w` and `y`) against one matmul of `n` times the
/// rows (`engine::rates::launch_from`).
fn launch_gap<B: Serve>(
    rig: &mut Rig<'_, B>,
    tactic: kernels_slang::linear::gemm::Tactic,
    timer: Timer<'_, B>,
    (act, w, y): (Tensor, Tensor, Tensor),
    n: u32,
) -> Result<f64> {
    let nothing = |_: &mut B::Frame| Ok(());
    let tiny_work = Work::matmul(LAUNCH, LAUNCH, LAUNCH);
    let wide = rig.zeros(LAUNCH * n, LAUNCH, Dtype::Bf16)?;
    let wide_y = rig.zeros(LAUNCH * n, LAUNCH, Dtype::Bf16)?;
    let next = std::cell::Cell::new(0u32);
    let chained = |sink: &kernels_slang::Ctx<'_>| {
        let (from, into) = if next.get() % 2 == 0 {
            (act, y)
        } else {
            (y, act)
        };
        next.set(next.get() + 1);
        kernels_slang::linear::gemm::matmul(sink, tactic, from, w, into)
            .map_err(fault("rates: matmul"))
    };
    let one = |sink: &kernels_slang::Ctx<'_>| {
        kernels_slang::linear::gemm::matmul(sink, tactic, wide, w, wide_y)
            .map_err(fault("rates: matmul"))
    };
    let wall = |count: u32, fire: &dyn Fn(&kernels_slang::Ctx<'_>) -> Result<()>| {
        rig.frame(timer, count, fire, &nothing)?;
        Ok::<u64, Fault>(median(
            (0..ROUNDS)
                .map(|_| rig.frame(timer, count, fire, &nothing).map(|(_, w)| w))
                .collect::<Result<Vec<u64>>>()?,
        ))
    };
    rig.step(
        format!("launch: a chain of {n} {LAUNCH}^3 matmuls"),
        tiny_work,
    );
    let chain = wall(n, &chained)?;
    rig.step(
        format!("launch: one {}x{LAUNCH}x{LAUNCH} matmul", LAUNCH * n),
        Work::matmul(LAUNCH * n, LAUNCH, LAUNCH),
    );
    let whole = wall(1, &one)?;
    Ok(engine::rates::launch_from(chain as f64, whole as f64, n))
}

/// The copy's bytes a ns: device memory copied after the narrowest matmul
/// (`tiny`, alone `single` ns), whose time is taken off.
fn copy_after<B: Serve>(
    rig: &mut Rig<'_, B>,
    timer: Timer<'_, B>,
    tiny: &dyn Fn(&kernels_slang::Ctx<'_>) -> Result<()>,
    single: u64,
    launch: u64,
) -> Result<f64> {
    let nothing = |_: &mut B::Frame| Ok(());
    let tiny_work = Work::matmul(LAUNCH, LAUNCH, LAUNCH);
    let device = rig.device;
    let copy = COPY_BYTES.min(device.max_buffer()) & !3;
    let from = B::Buffer::zeroed(device, copy, crate::books::owners::TUNING)?;
    let into = B::Buffer::zeroed(device, copy, crate::books::owners::TUNING)?;
    // The copy rides after the narrowest matmul, whose time is taken off;
    // by the adapter where it timed every round, else the host.
    let timed = |then: &dyn Fn(&mut B::Frame) -> Result<()>| -> Result<(u64, bool)> {
        rig.frame(timer, 1, tiny, then)?;
        let (mut adapter, mut host) = (Vec::new(), Vec::new());
        for _ in 0..ROUNDS {
            let (span, wall) = rig.frame(timer, 1, tiny, then)?;
            adapter.push(span);
            host.push(wall);
        }
        Ok(match adapter.into_iter().collect::<Option<Vec<u64>>>() {
            Some(adapter) => (median(adapter), true),
            None => (median(host), false),
        })
    };
    // A memory-bound op: its own frames warm the device until their rate
    // stops moving (`engine::rates::warm_up`'s rule) before it is timed.
    rig.step(
        format!("copy: {} MiB after one {LAUNCH}^3 matmul", copy >> 20),
        Work {
            kind: Kind::Memory,
            flops: tiny_work.flops,
            bytes: tiny_work.bytes + 2.0 * copy as f64,
        },
    );
    engine::rates::warm_up(|| {
        let (span, wall) = rig.frame(timer, 1, tiny, &|f| f.copy(&from, 0, &into, 0, copy))?;
        Ok::<f64, Fault>(1.0 / (span.unwrap_or(wall) as f64).max(1.0))
    })?;
    let (with_copy, by_adapter) = timed(&|f| f.copy(&from, 0, &into, 0, copy))?;
    let alone = if by_adapter {
        rig.step(format!("copy: one {LAUNCH}^3 matmul alone"), tiny_work);
        timed(&nothing)?.0
    } else {
        single
    };
    let copy_ns = with_copy.saturating_sub(alone).max(1);
    // Every byte copied is read and written.
    engine::rates::rate_from(
        "copy",
        copy_ns as f64,
        engine::rates::COPY_DISPATCHES,
        launch as f64,
        2.0 * copy as f64,
    )
    .map_err(|why| Fault::Device {
        call: IMPLAUSIBLE,
        why,
    })
}

/// The attention rate of `device` at `shape` through the kernel a served
/// fire takes there ([`attention`]'s one micro-fire): a check of the
/// estimate times it beside a served fire's attention at the same shape.
pub fn attention_rate<B: Serve, T: Tactics + ?Sized>(
    device: &B::Context,
    tactics: &T,
    tier: kernels_slang::Tier,
    timer: Timer<'_, B>,
    shape: engine::rates::AttnShape,
    launch: f64,
) -> Result<f64> {
    let mut rig = Rig::<B>::new(device, None);
    attention(
        &mut rig,
        tactics,
        tier,
        timer,
        shape,
        launch,
        Kind::Attention,
    )
}

/// The bf16 matmul rate of `device` at `(m, n, k)` under `tactic`: a check
/// of the estimate times the model's own planes with it.
pub fn gemm_rate<B: Serve>(
    device: &B::Context,
    tactic: kernels_slang::linear::gemm::Tactic,
    timer: Timer<'_, B>,
    (m, n, k): (u32, u32, u32),
) -> Result<f64> {
    let mut rig = Rig::<B>::new(device, None);
    let act = rig.operand(m, k, Dtype::Bf16)?;
    let w = rig.operand(n, k, Dtype::Bf16)?;
    let y = rig.zeros(m, n, Dtype::Bf16)?;
    rig.step(format!("check matmul {m}x{n}x{k}"), Work::matmul(m, n, k));
    let ns = rig.each(timer, &Touched::default(), 1, &|sink| {
        kernels_slang::linear::gemm::matmul(sink, tactic, act, w, y).map_err(fault("rates: matmul"))
    })?;
    Ok(2.0 * f64::from(m) * f64::from(n) * f64::from(k) / ns as f64)
}

/// What a load that skips measuring (`engine::rates::Measuring::Skip`) times
/// where the platform states a bound on one submission and no layer states
/// the roofline a serving step is held to it by (`crate::submit`): a
/// correctness guard, not a profile. The few shapes the bound's estimate
/// reads at the widest rung the load serves, each timed by single fires
/// ([`for_bound`]); nothing is stored.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ForBound {
    /// The widest rung the load serves: the most rows a step brings.
    pub rows: u32,
    /// The most keys a step's longest lane attends: the load's context.
    pub keys: u32,
    /// The model's widest dense plane of a layer, `(n, k)`: what a step's
    /// weight reads and matmuls are timed over.
    pub plane: (u32, u32),
    /// The dense kernel the load serves at `rows` over `plane`.
    pub wide: kernels_slang::linear::gemm::Tactic,
    /// The dense kernel the load serves at one row over `plane`.
    pub narrow: kernels_slang::linear::gemm::Tactic,
    /// The model's width: the matmul curve's (`[rates]
    /// gemm_tflops_by_rows`).
    pub width: u32,
    /// The model's widest head; `None` where it attends nothing.
    pub head_dim: Option<u32>,
    /// Whether the model carries recurrent state (a scan a layer).
    pub recurrent: bool,
}

/// What [`for_bound`] timed: each at a single fire, its launch in it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundTimes {
    /// A dispatch's launch, ns: a short chain against one fire of its work.
    pub launch_ns: f64,
    /// The weight bytes a ns a one-row matmul over the plane read, its
    /// launch in its time.
    pub read_bytes_per_ns: f64,
    /// The matmul's flops a ns at the widest rung's rows over the plane.
    pub matmul_flops_per_ns: f64,
    /// The attention's flops a ns, the widest rung's rows against the
    /// context's keys; `None` where the model attends nothing.
    pub attention_flops_per_ns: Option<f64>,
    /// The scan's elements a ns; `None` where the model carries no state.
    pub scan_elements_per_ns: Option<f64>,
}

impl BoundTimes {
    /// The `[rates]` the bound's estimate reads (`engine::load::estimate`),
    /// as `times` timed them at `at`: the launch, the weight read as the
    /// bandwidth (the per-row ops' bytes too, which nothing timed apart),
    /// the matmul curve one point at the widest rung's rows (a step of
    /// fewer rows read at it: a matmul runs no faster a flop at fewer
    /// rows), the attention at the context's keys (square and against a
    /// prefix alike, one point), the scan where the model carries state.
    /// Kept in the bound alone: nothing else reads or stores them.
    #[must_use]
    pub fn rates(&self, at: &ForBound) -> Rates {
        let tflops = |flops_per_ns: f64| flops_per_ns / 1e3;
        let attention = self.attention_flops_per_ns.map(tflops);
        Rates {
            launch_ns: Some(self.launch_ns.max(0.0).round() as u64),
            hbm_gbps: Some(self.read_bytes_per_ns),
            elementwise_gbps: Some(self.read_bytes_per_ns),
            gemm_tflops_by_rows: Some(vec![(
                at.width.max(1),
                vec![(at.rows.max(1), tflops(self.matmul_flops_per_ns))],
            )]),
            attn_tflops_by_keys: attention.map(|t| vec![(at.keys.max(at.rows).max(1), t)]),
            attn_tflops_by_rows_keys: attention
                .map(|t| vec![(at.rows.max(1), vec![(at.keys.max(at.rows).max(1), t)])]),
            scan_gelems: self.scan_elements_per_ns,
            ..Rates::default()
        }
    }
}

/// Times what a stated submit bound's estimate reads where a load skips
/// measuring and no layer states the roofline ([`ForBound`]): a short
/// launch chain, a one-row matmul over the model's widest layer plane (the
/// weight read), the same at the widest rung's rows, the attention of those
/// rows against the context's keys, and the scan where the model carries
/// state; each its sets touched once and one fire untimed, then the median
/// of [`ROUNDS`] single fires (no frame warmed until it settles), every
/// fire admitted under `max_submit` by its estimate, grown from the probe
/// as a rates step's are. Each time holds its launch, and the bound's
/// estimate counts a launch beside it again: the estimate only lengthens.
pub fn for_bound<B: Serve, T: Tactics + ?Sized>(
    device: &B::Context,
    tactics: &T,
    tier: kernels_slang::Tier,
    timer: Timer<'_, B>,
    max_submit: std::time::Duration,
    at: &ForBound,
) -> Result<BoundTimes> {
    let mut rig = Rig::<B>::new(device, Some(max_submit));
    rig.quick = true;
    rig.lay_probe(tactics.gemm(tier, crate::submit::PROBE.0))?;
    let launch_ns = {
        let act = rig.zeros(LAUNCH, LAUNCH, Dtype::Bf16)?;
        let w = rig.zeros(LAUNCH, LAUNCH, Dtype::Bf16)?;
        let y = rig.zeros(LAUNCH, LAUNCH, Dtype::Bf16)?;
        let queued = rig.queue.replace(false);
        let launch = launch_gap(
            &mut rig,
            tactics.gemm(tier, LAUNCH),
            timer,
            (act, w, y),
            engine::rates::LAUNCH_CHAINS[0],
        );
        rig.queue.set(queued);
        rig.release();
        launch?.max(0.0)
    };
    let (n, k) = at.plane;
    // A one-row matmul reads its plane once: two bytes an element, two
    // flops an element, so its bytes a ns are its flops a ns.
    let narrow = matmul(&mut rig, at.narrow, timer, (1, n, k), 0.0);
    rig.release();
    let read_bytes_per_ns = narrow?;
    // The bound's estimate takes each dispatch's longer of its reads at the
    // bandwidth (the one-row read above, cold) and its flops at the rates
    // below: those need no cold read, so each rotates through two sets, not
    // past the cache (most of the probe's time on a phone was laying and
    // touching the cold rotation, 768 MiB a point).
    rig.planes_most = Some(2);
    let matmul_flops_per_ns = if at.rows > 1 {
        let wide = matmul(&mut rig, at.wide, timer, (at.rows, n, k), 0.0);
        rig.release();
        wide?
    } else {
        read_bytes_per_ns
    };
    let attention_flops_per_ns = match at.head_dim {
        Some(head_dim) => {
            let shape = engine::rates::AttnShape {
                rows: at.rows.max(1),
                keys: at.keys.max(at.rows).max(1),
                head_dim,
                ..engine::rates::ATTN_SHAPE
            };
            let rate = attention(&mut rig, tactics, tier, timer, shape, 0.0, Kind::Prefixed);
            rig.release();
            Some(rate?)
        }
        None => None,
    };
    let scan_elements_per_ns = if at.recurrent {
        let rate = scan_within(&mut rig, tactics, timer, 0.0, &mut |_| {});
        rig.release();
        Some(rate?)
    } else {
        None
    };
    Ok(BoundTimes {
        launch_ns,
        read_bytes_per_ns,
        matmul_flops_per_ns,
        attention_flops_per_ns,
        scan_elements_per_ns,
    })
}

/// The width (and depth) of the matmul the device's resident workgroups are
/// read against where a load skips measuring ([`resident_for_skip`]): the
/// rates step's widest ([`engine::rates::GEMM_WIDTH`]), whose one-workgroup
/// fire is long enough that its launch is a small share of it (at 1024 a
/// phone's one-workgroup fire was launch-bound, and the count read ~2x the
/// rates step's), at [`RESIDENT_ROWS`] rows.
pub const RESIDENT_SHAPE: u32 = engine::rates::GEMM_WIDTH;

/// The rows of the full-width matmul [`resident_for_skip`] reads the kernel's
/// rate at: a thousand rows fill every device served many times over.
pub const RESIDENT_ROWS: u32 = 1024;

/// `[rates] resident_workgroups` where a load skips measuring
/// (`engine::rates::Measuring::Skip`) and no layer states it: a device fact
/// no driver states as such (Vulkan's compute units are an extension's,
/// WebGPU states none), so it is timed, never defaulted: a short launch
/// chain, the profile path's matmul at [`RESIDENT_SHAPE`] (rows, columns
/// and depth), and the same kernel shrunk to one workgroup over the same
/// depth ([`resident_workgroups`], the rates step's own reading), each by
/// single fires ([`ROUNDS`] of them, the median), every fire admitted under
/// `max_submit` where the platform states one. Milliseconds; stored
/// nowhere. The workgroups, and the ms it took.
pub fn resident_for_skip<B: Serve, T: Tactics + ?Sized>(
    device: &B::Context,
    tactics: &T,
    tier: kernels_slang::Tier,
    timer: Timer<'_, B>,
    max_submit: Option<std::time::Duration>,
) -> Result<(f64, u128)> {
    let started = std::time::Instant::now();
    let mut rig = Rig::<B>::new(device, max_submit);
    rig.quick = true;
    // Both matmuls compute-bound at their depth: a plane read warm or cold
    // moves neither, so each rotates through two, not past the cache.
    rig.planes_most = Some(2);
    rig.lay_probe(tactics.gemm(tier, crate::submit::PROBE.0))?;
    let launch = {
        let act = rig.zeros(LAUNCH, LAUNCH, Dtype::Bf16)?;
        let w = rig.zeros(LAUNCH, LAUNCH, Dtype::Bf16)?;
        let y = rig.zeros(LAUNCH, LAUNCH, Dtype::Bf16)?;
        let queued = rig.queue.replace(false);
        let launch = launch_gap(
            &mut rig,
            tactics.gemm(tier, LAUNCH),
            timer,
            (act, w, y),
            engine::rates::LAUNCH_CHAINS[0],
        );
        rig.queue.set(queued);
        rig.release();
        launch?.max(0.0)
    };
    let w = RESIDENT_SHAPE;
    let peak = matmul(
        &mut rig,
        tactics.gemm(tier, RESIDENT_ROWS),
        timer,
        (RESIDENT_ROWS, w, w),
        launch,
    );
    rig.release();
    let curve = [(RESIDENT_ROWS, peak?)];
    let resident = resident_workgroups(&mut rig, tactics, tier, timer, &curve, w, launch);
    rig.release();
    Ok((resident?, started.elapsed().as_millis()))
}

/// The device's clock read off its reference kernel ([`crate::submit::Probe`],
/// the smallest shape on the matmul curve): `engine::rates`' shared reading.
pub use engine::rates::{ClockRead, clock_moved, throttled};

/// `device`'s clock now ([`ClockRead`]), its reference kernel under `tactic`:
/// one read, as a phase's after read is taken.
pub fn clock_read<B: Serve>(
    device: &B::Context,
    tactic: kernels_slang::linear::gemm::Tactic,
) -> Result<ClockRead> {
    let probe = crate::submit::Probe::<B>::new(device, tactic)?;
    read_clock(device, &probe)
}

fn read_clock<B: Serve>(device: &B::Context, probe: &crate::submit::Probe<B>) -> Result<ClockRead> {
    let fires = (0..ROUNDS)
        .map(|_| probe.fire(device))
        .collect::<Result<Vec<f64>>>()?;
    let fastest = fires.iter().copied().fold(f64::INFINITY, f64::min);
    let slowest = fires.iter().copied().fold(0.0, f64::max);
    Ok(ClockRead {
        fastest_ns: fastest,
        spread: (slowest - fastest) / fastest.max(f64::MIN_POSITIVE),
    })
}

/// `device`'s clock as a phase starts ([`warmed`]): its reference kernel
/// read until the clock settles, so a GPU that scales its clock with its
/// load is read at the clock the phase runs at: neither off an idle DVFS
/// clock still ramping as work arrives (a phone's), nor off a brief boost
/// above the clock it sustains (a desktop's).
pub fn clock_warmed<B: Serve>(
    device: &B::Context,
    tactic: kernels_slang::linear::gemm::Tactic,
) -> Result<ClockRead> {
    let probe = crate::submit::Probe::<B>::new(device, tactic)?;
    warmed(|| read_clock(device, &probe))
}

/// [`clock_warmed`] on a probe laid once (`probe`, laid at load):
/// nothing is laid, compiled or recorded per read, and `stop` is read
/// before each of the reference's fires, so an arrival waits at most one
/// fire of it.
pub fn clock_warmed_on<B: Serve>(
    device: &B::Context,
    probe: &crate::submit::Probe<B>,
    stop: &dyn Fn() -> bool,
) -> Result<Option<ClockRead>> {
    let stopped = std::cell::Cell::new(false);
    let read = engine::rates::warmed_until::<Fault>(
        || match read_clock_until(device, probe, stop)? {
            Some(read) => Ok(read),
            None => {
                stopped.set(true);
                // Read as settled so the warm-up ends; dropped below.
                Ok(ClockRead {
                    fastest_ns: 1.0,
                    spread: f64::INFINITY,
                })
            }
        },
        &|| stopped.get() || stop(),
    )?;
    Ok(read.filter(|_| !stopped.get()))
}

/// One read of the reference on `probe` ([`clock_read`]'s), `stop` read
/// before each fire: `None` where it stopped.
pub fn read_clock_until<B: Serve>(
    device: &B::Context,
    probe: &crate::submit::Probe<B>,
    stop: &dyn Fn() -> bool,
) -> Result<Option<ClockRead>> {
    let mut fires = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        if stop() {
            return Ok(None);
        }
        fires.push(probe.fire(device)?);
    }
    let fastest = fires.iter().copied().fold(f64::INFINITY, f64::min);
    let slowest = fires.iter().copied().fold(0.0, f64::max);
    Ok(Some(ClockRead {
        fastest_ns: fastest,
        spread: (slowest - fastest) / fastest.max(f64::MIN_POSITIVE),
    }))
}


/// `engine::rates::warmed` over this crate's faults.
pub fn warmed(read: impl FnMut() -> Result<ClockRead>) -> Result<ClockRead> {
    engine::rates::warmed(read)
}


/// The fires a warm-up rotation's next frame holds of `left`: as many as fit
/// under `bound` at `per_fire` ns a fire (the slowest the rotation measured)
/// times `worst` (at least the walk's measured worst ratio); 0 where not one
/// fits or nothing estimates a fire under a bound; all of them where no
/// bound is stated.
fn rotation_fires(
    bound: Option<std::time::Duration>,
    per_fire: Option<f64>,
    worst: f64,
    left: u32,
) -> u32 {
    match (bound, per_fire) {
        (Some(cap), Some(per)) => {
            ((cap.as_nanos() as f64 / (per.max(1.0) * worst.max(1.0))).floor() as u32).min(left)
        }
        (Some(_), None) => 0,
        (None, _) => left,
    }
}

#[cfg(test)]
mod curve_tests {
    use super::*;

    /// A point is its repeats' median, and their spread its width over it.
    #[test]
    fn a_point_is_its_repeats_median_with_their_spread() {
        assert_eq!(median_spread(&[3.0, 1.0, 2.0]), (2.0, 1.0));
        assert_eq!(median_spread(&[5.0, 5.0, 5.0]), (5.0, 0.0));
    }

    /// A curve point is its fastest sample (noise only slows a fire), and
    /// their spread its width over it.
    #[test]
    fn a_curve_point_is_its_fastest_sample() {
        assert_eq!(best_spread(&[3.0, 1.0, 2.0]), (1.0, 2.0));
        assert_eq!(best_spread(&[5.0, 5.0, 5.0]), (5.0, 0.0));
    }

    /// The rows a shape's curves are timed at: every power of four, then the
    /// load's most.
    #[test]
    fn a_shape_is_timed_at_every_power_of_four_to_the_top() {
        assert_eq!(timed_rows(10240), [1, 4, 16, 64, 256, 1024, 4096, 10240]);
        assert_eq!(timed_rows(4096), [1, 4, 16, 64, 256, 1024, 4096]);
        assert_eq!(timed_rows(1), [1]);
    }

    /// A load times, at each of its shapes, each kernel whose curve the
    /// profile lacks or holds short of the load's rows with no recorded stop;
    /// a curve that reaches them, or stopped for a reason it records, is not
    /// timed again; nor is a kernel at a shape twice.
    #[test]
    fn a_stopped_curve_is_kept_and_a_short_one_measured_again() {
        let target = kernels_slang::Target::Spirv;
        let vector = |rows| kernels_slang::linear::gemm::Tactic {
            tier: kernels_slang::Tier::Subgroup,
            vector: rows,
            wide: false,
        };
        let kernels = [vector(1), vector(4), vector(4)];
        let curve = |key, n, k, to, stop| profile::fire::KernelCurve {
            by: profile::fire::MeasuredBy::default(),
            kind: profile::fire::KernelCurve::VECTOR,
            key,
            n,
            k,
            curve: vec![(1, 0.5), (to, 4.0)],
            stop,
            spread: Vec::new(),
        };
        let rates = profile::fire::Rates {
            kernel_tflops_by_shape: Some(vec![
                // Reaches 8192 rows.
                curve(1, 1024, 1024, 8192, None),
                // Stopped by the bound at 2048: final.
                curve(
                    4,
                    1024,
                    1024,
                    1024,
                    Some((2048, profile::fire::KernelCurve::BOUND)),
                ),
                // Short of 8192 with no stop: timed again.
                curve(1, 2048, 1024, 4096, None),
            ]),
            ..Default::default()
        };
        let shapes = [(1024, 1024), (2048, 1024)];
        let ids = |wanted: Vec<(kernels_slang::linear::gemm::Tactic, u32, u32)>| {
            wanted
                .into_iter()
                .map(|(t, n, k)| (t.vector, n, k))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            ids(lacking(target, &kernels, &rates, &shapes, 8192)),
            [(1, 2048, 1024), (4, 2048, 1024)]
        );
        // A load of fewer rows lacks only the shape never timed.
        assert_eq!(
            ids(lacking(target, &kernels, &rates, &shapes, 4096)),
            [(4, 2048, 1024)]
        );
        // What a load saves keeps where the curves stopped and their points'
        // spreads, and reads back.
        let mut rates = rates;
        if let Some(curves) = rates.kernel_tflops_by_shape.as_mut() {
            curves[0].spread = vec![(1, 0.0), (8192, 0.05)];
        }
        let text = format!("api = 1\n{}", rates.toml());
        let read = profile::Registry::new("test", &[], &[])
            .load_user(&text, "test")
            .unwrap()
            .rates();
        assert_eq!(read.kernel_tflops_by_shape, rates.kernel_tflops_by_shape);
    }

    /// Stored curves sifted by their keys (`crate::stored`): under the same
    /// kernels (a pie version bump) the load times none; one kernel's code
    /// changed, it times that kernel's curves alone; on another device,
    /// every one.
    #[test]
    fn a_curve_is_timed_again_only_where_its_kernel_changed() {
        struct Build {
            on: &'static str,
            code: [(&'static str, u32); 2],
        }
        impl profile::fire::Now for Build {
            fn on(&self) -> String {
                self.on.to_string()
            }
            fn method(&self, what: &str) -> Option<u32> {
                crate::stored::method(what)
            }
            fn code(&self, kernel: &str) -> Option<u32> {
                self.code
                    .iter()
                    .find(|(k, _)| *k == kernel)
                    .map(|&(_, c)| c)
            }
        }
        let target = kernels_slang::Target::Spirv;
        let vector = kernels_slang::linear::gemm::Tactic {
            tier: kernels_slang::Tier::Subgroup,
            vector: 4,
            wide: false,
        };
        let tiled = kernels_slang::linear::gemm::Tactic {
            vector: 0,
            ..vector
        };
        let kernels = [vector, tiled];
        let build = Build {
            on: "vulkan 10de:2c31 driver 0x1",
            code: [("gemv", 1), ("tiled", 2)],
        };
        let curve = |kind, key, n, kernel: &str| profile::fire::KernelCurve {
            by: profile::fire::MeasuredBy {
                method: CURVE_METHOD,
                kernels: vec![(
                    kernel.to_string(),
                    profile::fire::Now::code(&build, kernel).unwrap(),
                )],
            },
            kind,
            key,
            n,
            k: 1024,
            curve: vec![(1, 0.5), (4096, 4.0)],
            stop: None,
            spread: Vec::new(),
        };
        let stored = profile::fire::Rates {
            kernel_tflops_by_shape: Some(vec![
                curve(profile::fire::KernelCurve::VECTOR, 4, 1024, "gemv"),
                curve(profile::fire::KernelCurve::TILE, 32, 1024, "tiled"),
                curve(profile::fire::KernelCurve::VECTOR, 4, 2048, "gemv"),
                curve(profile::fire::KernelCurve::TILE, 32, 2048, "tiled"),
            ]),
            measured_on: Some(build.on.to_string()),
            ..Default::default()
        };
        let shapes = [(1024, 1024), (2048, 1024)];
        let wanted = |now: &Build| {
            let mut rates = stored.clone();
            rates.sift(now);
            lacking(target, &kernels, &rates, &shapes, 4096)
                .into_iter()
                .map(|(t, n, _)| (t.vector, n))
                .collect::<Vec<_>>()
        };
        assert_eq!(wanted(&build), [], "the same kernels: none timed");
        let changed = Build {
            code: [("gemv", 1), ("tiled", 3)],
            ..build
        };
        assert_eq!(
            wanted(&changed),
            [(0, 1024), (0, 2048)],
            "the tiled kernel's alone"
        );
        let other = Build {
            on: "vulkan 10de:2c32 driver 0x1",
            ..changed
        };
        assert_eq!(wanted(&other).len(), 4, "another device: every one");
    }
}

#[cfg(test)]
mod wanted_tests {
    use super::*;

    /// A profile lacking one key times that key alone (the launch and the
    /// copy together); one stating none times every one.
    #[test]
    fn a_step_times_only_what_the_profile_lacks() {
        let all = Rates {
            stale: None,
            landing: None,
            load_width: None,
            launch_ns: Some(3000),
            gemm_tflops_by_rows: Some(vec![(4096, vec![(256, 10.0)])]),
            attn_tflops_by_keys: Some(vec![(512, 5.0)]),
            hbm_gbps: Some(650.0),
            cached_gbps: None,
            elementwise_gbps: Some(700.0),
            scan_gelems: Some(100.0),
            few_row_tflops: None,
            workgroups_per_core: None,
            submit_worst_ratio: None,
            resident_workgroups: None,
            attn_tflops_by_rows_keys: None,
            product_ns: None,
            kernel_tflops_by_shape: None,
            dense_settled: Vec::new(),
            measured_on: None,
            measured_by: None,
        };
        let none = Wanted {
            launch: false,
            gemm: false,
            attn: false,
            elementwise: false,
            scan: false,
            resident: false,
        };
        assert_eq!(Wanted::of(&all, &KEYS), none);
        let few = Rates {
            scan_gelems: None,
            ..all.clone()
        };
        assert_eq!(Wanted::of(&few, &KEYS), Wanted { scan: true, ..none });
        let copy = Rates {
            hbm_gbps: None,
            cached_gbps: None,
            ..all.clone()
        };
        assert_eq!(
            Wanted::of(&copy, &KEYS),
            Wanted {
                launch: true,
                ..none
            }
        );
        let every = Wanted::of(&Rates::default(), &KEYS);
        assert!(
            every.launch
                && every.gemm
                && every.attn
                && every.elementwise
                && every.scan
                && !every.resident
        );
        // The workgroups the device holds are measured where the profile
        // lacks them, and only then.
        let keys = keys_at(None);
        assert!(Wanted::of(&all, &keys).resident);
        assert_eq!(
            Wanted::of(
                &Rates {
                    resident_workgroups: Some(48.0),
                    ..all.clone()
                },
                &keys
            ),
            none
        );
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;

    const BOUND: std::time::Duration = std::time::Duration::from_millis(2000);

    /// The Adreno 840's trace: a fire swinging 182 → 318 ms. Sized from the
    /// slowest fire at the walk's worst ratio, a frame stays under the bound
    /// even when every fire runs at the slowest seen; sized from the last
    /// (fastest) fire at 1, it would not.
    #[test]
    fn a_rotation_frame_sized_from_the_slowest_fire_at_the_worst_ratio_stays_under_the_bound() {
        let (fast, slow, walk_worst) = (182e6, 318e6, 1.57);
        let fires = rotation_fires(Some(BOUND), Some(slow), walk_worst, 64);
        assert!(fires >= 1);
        assert!(f64::from(fires) * slow * walk_worst <= BOUND.as_nanos() as f64);
        assert!(f64::from(fires) * slow <= BOUND.as_nanos() as f64);
        // The old sizing: the last fire's time at a ratio of 1.
        let old = rotation_fires(Some(BOUND), Some(fast), 1.0, 64);
        assert!(
            f64::from(old) * slow > BOUND.as_nanos() as f64,
            "the old sizing passes the bound at the slowest fire"
        );
    }

    /// Not one fire fits: the rotation stops (the cube halves), never
    /// submits a frame past the bound.
    #[test]
    fn a_fire_past_the_bound_at_the_worst_ratio_fits_none() {
        assert_eq!(rotation_fires(Some(BOUND), Some(1500e6), 1.4, 8), 0);
        // No bound stated: the whole rotation.
        assert_eq!(rotation_fires(None, Some(1500e6), 1.4, 8), 8);
        // Nothing estimates a fire under a bound: none.
        assert_eq!(rotation_fires(Some(BOUND), None, 1.4, 8), 0);
    }

    /// A new shape's first fire is grown from the fires of its kind measured
    /// before it: d256 after d64 at the same keys is estimated at the work
    /// ratio, so on a device where d64 took 1.29 s the d256 fire is
    /// estimated past the 2 s bound and refused before it is submitted. A
    /// smaller shape is estimated at a larger one's time; the first of its
    /// kind at the probe's rate, never under the probe's time; and nothing
    /// at all with nothing measured.
    #[test]
    fn a_first_fire_is_grown_from_the_measured_fires_before_it() {
        let attn = |s: engine::rates::AttnShape| Work {
            kind: Kind::Attention,
            flops: attn_flops(s),
            bytes: engine::rates::attn_bytes(s),
        };
        let d64 = attn(engine::rates::ATTN_SHAPE.at(8192).with_dim(64));
        let d256 = attn(engine::rates::ATTN_SHAPE.at(8192).with_dim(256));
        let took = 1.29e9;
        let measured = [Fired {
            work: d64,
            ns: took,
        }];
        let estimate = grown(&measured, None, d256).unwrap();
        assert!((estimate / took - d256.times(&d64)).abs() < 1e-9);
        let admission = engine::rates::Admission {
            settled: true,
            ..engine::rates::Admission::new(Some(BOUND))
        };
        assert!(
            !admission.fits(Some(estimate)),
            "the d256 first fire is refused unsubmitted"
        );
        assert!(
            admission.fits(Some(took)),
            "the d64 fire it was estimated from fits"
        );
        // A smaller one: no longer than the larger measured.
        let d32 = attn(engine::rates::ATTN_SHAPE.at(512));
        assert_eq!(grown(&measured, None, d32), Some(took));
        // The first of its kind: the probe's rate, never under its time.
        let (m, n, k) = crate::submit::PROBE;
        let probe = Fired {
            work: Work::matmul(m, n, k),
            ns: 2e6,
        };
        let copy = Work {
            kind: Kind::Memory,
            flops: 0.0,
            bytes: 100.0 * probe.work.bytes,
        };
        assert_eq!(grown(&[probe], Some(probe), copy), Some(200e6));
        let tiny = Work::matmul(16, 16, 16);
        assert_eq!(grown(&[probe], Some(probe), tiny), Some(2e6));
        // The cube the walk measured, its first fire after: grown from it.
        let cube = Work::matmul(2048, 2048, 2048);
        let walked = Fired {
            work: cube,
            ns: 40e6,
        };
        assert_eq!(grown(&[probe, walked], Some(probe), cube), Some(40e6));
        assert_eq!(
            grown(
                &[probe, walked],
                Some(probe),
                Work::matmul(4096, 4096, 4096)
            ),
            Some(320e6)
        );
        // Nothing measured: no estimate, and nothing fits a bound unestimated.
        assert_eq!(grown(&[], None, cube), None);
        assert!(!admission.fits(grown(&[], None, cube)));
    }
}

#[cfg(test)]
mod bound_tests {
    use super::*;

    /// A phone's order of times for a 28-layer, 4096-wide attending model:
    /// the bound's rates read what was timed, and its estimate holds a
    /// long prompt step past a 2 s bound to layers (or rows) while a
    /// decode goes whole.
    #[test]
    fn the_times_taken_for_the_bound_state_its_estimate() {
        let at = ForBound {
            rows: 512,
            keys: 4096,
            plane: (4096, 4096),
            wide: kernels_slang::linear::gemm::Tactic {
                tier: kernels_slang::Tier::Baseline,
                vector: 0,
                wide: true,
            },
            narrow: kernels_slang::linear::gemm::Tactic {
                tier: kernels_slang::Tier::Baseline,
                vector: 1,
                wide: true,
            },
            width: 4096,
            head_dim: Some(128),
            recurrent: false,
        };
        let times = BoundTimes {
            launch_ns: 10_000.0,
            read_bytes_per_ns: 60.0,
            matmul_flops_per_ns: 500.0,
            attention_flops_per_ns: Some(250.0),
            scan_elements_per_ns: None,
        };
        let rates = times.rates(&at);
        assert_eq!(rates.launch_ns, Some(10_000));
        assert_eq!(rates.hbm_gbps, Some(60.0));
        assert_eq!(rates.elementwise_gbps, Some(60.0));
        assert_eq!(rates.gemm_tflops_by_rows, Some(vec![(4096, vec![(512, 0.5)])]));
        assert_eq!(rates.attn_tflops_by_keys, Some(vec![(4096, 0.25)]));
        assert_eq!(
            rates.attn_tflops_by_rows_keys,
            Some(vec![(512, vec![(4096, 0.25)])])
        );
        assert_eq!(rates.submit_worst_ratio, None, "the ratio is the load's to state");
        let counts = engine::load::Counts {
            weight_bytes: 1 << 30,
            expert_bytes: 0,
            experts: None,
            gemm_shapes: vec![engine::load::GemmShape {
                n: 4096,
                k: 4096,
                flops_per_row: 2 * 4096 * 4096 * 28,
                planes: 28,
                bytes: 1 << 30,
                routed: false,
            }],
            head_bytes: 0,
            attn_flops_per_row_key: 4 * 4096 * 28,
            kv_bytes_per_key: 1 << 14,
            attn_head_dim: 128,
            readout_flops: 0,
            readout_depth: 4096,
            state_elements_per_row: 0,
            elementwise_ops: vec![0],
            attn_dispatches: 28,
            scan_dispatches: 0,
            width: 4096,
        };
        let held = crate::submit::SubmitBound {
            bound_ns: 2e9,
            worst: 1.0,
            estimate: engine::load::estimate(&counts, &rates).expect("the times state it"),
            launches: 200,
            layers: 28,
            fire_lanes: 64,
            rungs: vec![1, 64, 512],
            prefixed: rates
                .attn_tflops_by_rows_keys
                .clone()
                .unwrap()
                .into_iter()
                .map(|(rows, c)| (rows, c.into_iter().map(|(k, t)| (k, t * 1e3)).collect()))
                .collect(),
            scale: 1.0,
            measured: crate::submit::Measured::default(),
            clock: None,
        };
        let step = |rows: u32, keys: u32| crate::submit::StepRows {
            rows,
            lanes: 1,
            keys,
            prompt: rows > 1,
        };
        assert!(
            matches!(held.admit(step(1, 4096)), crate::submit::Admitted::Whole { .. }),
            "{:?}",
            held.admit(step(1, 4096))
        );
        assert!(
            !matches!(held.admit(step(512, 4096)), crate::submit::Admitted::Whole { .. }),
            "a 512-row step at 4096 keys is past 2 s whole: {:?}",
            held.step_ns(step(512, 4096))
        );
    }
}

#[cfg(test)]
mod clock_tests {
    use super::*;

    fn read(fastest_ns: f64, spread: f64) -> ClockRead {
        ClockRead { fastest_ns, spread }
    }

    /// A phase is throttled where its after read is slower than its warmed
    /// before read by more than the reads' own spread.
    #[test]
    fn a_phase_whose_clock_sags_is_throttled() {
        assert_eq!(throttled(read(101.0, 0.05), read(103.0, 0.05)), None, "inside the noise");
        let hot = throttled(read(100.0, 0.02), read(160.0, 0.02)).expect("throttled");
        assert!((hot - 1.6).abs() < 1e-9, "{hot}");
        assert_eq!(throttled(read(100.0, 0.02), read(91.0, 0.02)), None, "a faster after read");
    }

    /// A DVFS GPU idling at a low clock (an idle Adreno's 422 of 1300 MHz)
    /// ramps as the reference runs: the warmed read is the ramped clock, so
    /// a phase at that clock is stored (not judged throttled off the idle
    /// read).
    #[test]
    fn a_clock_ramping_from_idle_is_read_warmed_and_stores() {
        let mut reads = [308.0, 210.0, 140.0, 101.0, 100.0].into_iter();
        let before = warmed(|| Ok(read(reads.next().expect("a read"), 0.02))).unwrap();
        assert!((before.fastest_ns - 100.5).abs() < 1e-9, "{before:?}");
        assert_eq!(throttled(before, read(100.4, 0.02)), None, "stored");
    }

    /// A GPU that boosts briefly, then holds a sustained clock 1.4x lower
    /// (a desktop's, the Spark's): the warmed read is the sustained clock,
    /// so a phase at it is stored (not judged throttled off the boost).
    #[test]
    fn a_boost_then_a_sustained_clock_is_read_settled_and_stores() {
        let mut reads = [100.0, 139.0, 140.0].into_iter();
        let before = warmed(|| Ok(read(reads.next().expect("a read"), 0.02))).unwrap();
        assert!((before.fastest_ns - 139.5).abs() < 1e-9, "{before:?}");
        assert_eq!(throttled(before, read(141.0, 0.02)), None, "stored");
    }

    /// A Spark-style clock (a boost 1.4x faster than the clock it
    /// sustains, falling back mid-run) under a settle's points: each point
    /// is a kernel's 1000 ns at the sustained clock, its time scaled by the
    /// clock it ran at; a point whose own before and after reads straddle
    /// the transition is discarded. The points kept at the sustained clock
    /// spread by the kernel's own noise (1.3%), where every point together
    /// spreads 30%.
    #[test]
    fn a_point_across_a_clock_transition_is_discarded() {
        // The clock over 12 points: boosted, falling mid-point, sustained.
        let clock = |t: usize| if t < 4 { 1.4 } else { 1.0 };
        let noise = [1.0, 1.004, 0.998, 1.006, 1.002, 0.999, 1.01, 1.003, 1.0, 0.997, 1.005, 1.001];
        let (mut all, mut kept) = (Vec::new(), Vec::new());
        for point in 0..12 {
            // Read before at the point's start, after at its end (the
            // transition falls inside point 3).
            let before = read(100.0 / clock(point), 0.01);
            let after = read(100.0 / clock(point + 1), 0.01);
            let ns = 1000.0 * noise[point] / ((clock(point) + clock(point + 1)) / 2.0);
            all.push(ns);
            if clock_moved(before, after).is_none() {
                kept.push(ns);
            }
        }
        let (_, spread_all) = median_spread(&all);
        assert_eq!(kept.len(), 11, "only the point across the transition is dropped");
        assert!(spread_all > 0.25, "every point together: {spread_all}");
        // The boosted points agree among themselves too, but the kept set
        // spans both clocks: a settle's options are fired interleaved, so
        // the transition is the one point it drops; the rest each ran at
        // one clock (its before and after agreeing) and the rounds' median
        // reads the sustained clock.
        let sustained: Vec<f64> = kept.iter().copied().filter(|&ns| ns > 900.0).collect();
        let (_, spread_sustained) = median_spread(&sustained);
        assert!(spread_sustained < 0.015, "the sustained points: {spread_sustained}");
    }

    /// A clock that sags during the phase, after the warmed read: refused.
    #[test]
    fn a_clock_sagging_during_the_phase_is_refused() {
        let mut reads = [100.0, 100.5].into_iter();
        let before = warmed(|| Ok(read(reads.next().expect("a read"), 0.02))).unwrap();
        assert!(throttled(before, read(181.0, 0.02)).is_some(), "refused");
    }
}
