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
//! - each of its vector matmul kernels (by the rows a workgroup carries) at
//!   each of [`VECTOR_ROWS_TIMED`] over the widest plane, for
//!   `vector_tflops_by_rows`, and each tiled kernel at each of [`TILE_ROWS`],
//!   for `tile_tflops_by_rows`;
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
//! Every float operand a rate reads holds `engine::rates::operand_pattern`
//! repeated (outputs, index tables and the scan's gates as laid), as every
//! backend's do.
//!
//! The backend times a frame ([`Timer`]): by its adapter's timestamps where
//! it has them, else by the host's clock around submit and wait. The
//! repeated fires are the host's clock's difference of two frames either
//! way: one submit each, so neither the clock nor the adapter's span start
//! biases it.

use engine::rates::{SCAN_SHAPE, Touched, attn_flops, gemm_flops, scan_elements, weights_at};
use kernels_slang::{KvPool, RaggedTensor, Tensor};
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
/// of every submission up to the one that faults. None stated elsewhere.
///
/// Every engine that submits through the platform's GPU driver (Vulkan,
/// wgpu) states this one.
pub const STATED_SUBMIT_BOUND: Option<std::time::Duration> =
    stated_submit_bound(cfg!(target_os = "android"));

/// [`STATED_SUBMIT_BOUND`] on a platform that is Android or not: the
/// driver's 2 s there, none elsewhere.
#[must_use]
pub const fn stated_submit_bound(android: bool) -> Option<std::time::Duration> {
    if android {
        Some(std::time::Duration::from_secs(2))
    } else {
        None
    }
}

/// The `[rates]` keys the slang backends' estimate weighs a fire by (of
/// `profile::fire::KEYS`): what a load whose profile lacks one measures.
/// `vector_tflops_by_rows` and `tile_tflops_by_rows` are each vector and
/// tiled matmul kernel's rate by rows, which the dense matmul's kernel is
/// chosen by (`crate::dense`). `few_row_tflops` and `workgroups_per_core`
/// are not weighed here: no choice on these backends reads them, so a load
/// measures neither.
pub const KEYS: [&str; 8] = [
    "launch_ns",
    "gemm_tflops_by_rows",
    "attn_tflops_by_keys",
    "hbm_gbps",
    "elementwise_gbps",
    "scan_gelems",
    "tile_tflops_by_rows",
    "vector_tflops_by_rows",
];

/// The rows each tiled kernel is timed at for `tile_tflops_by_rows`: from a
/// decode's few to the matmul curve's first.
const TILE_ROWS: [u32; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];

/// The rows each vector kernel is timed at for `vector_tflops_by_rows`: a
/// decode's one to the matmul curve's first.
const VECTOR_ROWS_TIMED: [u32; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];

/// The matmul a launch is timed with: its work is nothing beside a launch.
pub const LAUNCH: u32 = 16;

/// Timed frames of each kind; the median is kept.
pub const ROUNDS: usize = 5;

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
    /// opened with): what the next frame of it is estimated from.
    per_fire: std::cell::Cell<Option<f64>>,
    /// The fires of the frame `per_fire` was measured over (0 for a step's
    /// stated estimate).
    basis: std::cell::Cell<u32>,
    /// How frames are admitted under `max_submit`
    /// (`engine::rates::Admission`): the worst ratio of a frame's time to
    /// its estimate over frames timed once the warm-up settled the clock,
    /// carried up from a frame of no more fires (one carried down holds the
    /// frame's own submit, and only overestimates).
    admission: std::cell::Cell<engine::rates::Admission>,
    /// The step whose fire, at one a frame, was past the bound: refused by
    /// name, and the step drops that shape ([`dropped`]).
    past_bound: std::cell::RefCell<Option<String>>,
    /// The ratio of the slowest fire to the fastest the warm-up's rotations
    /// measured at its cube: how far this device's clock swings from fire to
    /// fire, which a settled frame's admission carries as its least worst
    /// ratio (the Adreno 840 swung ~10% inside one settled step).
    swing: std::cell::Cell<f64>,
}

impl<'a, B: Serve> Rig<'a, B> {
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
            admission: std::cell::Cell::new(engine::rates::Admission::new(max_submit)),
            past_bound: std::cell::RefCell::new(None),
            swing: std::cell::Cell::new(1.0),
        }
    }

    /// The frames from here run `step`'s fire: estimated from `per_fire`
    /// ns a fire where the caller can say, else from their own times as
    /// they land.
    fn step(&self, step: impl Into<String>, per_fire: Option<f64>) {
        *self.step.borrow_mut() = step.into();
        self.per_fire.set(per_fire);
        self.basis.set(0);
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
        let a = self.admission.get();
        a.fires(count, self.per_fire.get())
            .ok_or_else(|| self.refuse(self.per_fire.get().unwrap_or(0.0), 1))
    }

    /// The by-name refusal of a frame of `count` fires estimated at
    /// `per_fire` ns a fire: marked, so its step drops the shape.
    fn refuse(&self, per_fire: f64, count: u32) -> Fault {
        let step = self.step.borrow().clone();
        let a = self.admission.get();
        let estimate = per_fire * f64::from(count);
        *self.past_bound.borrow_mut() = Some(step.clone());
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
        let step = self.step.borrow().clone();
        let estimate = self.per_fire.get().map(|p| p * f64::from(count));
        let worst = self.admission.get().worst;
        let ms = |ns: f64| ns / 1e6;
        tracing::debug!(
            step = %step,
            fires = count,
            estimate_ms = ?estimate.map(ms),
            admitted_ms = ?estimate.map(|e| ms(e * worst)),
            worst,
            bound_ms = ?self.max_submit.map(|b| b.as_secs_f64() * 1e3),
            "submitting"
        );
        // A frame whose estimate, at the worst error seen, passes the
        // device's bound on one submission is refused by name, never
        // submitted: a submission past it loses the device.
        if !self.admission.get().fits(estimate) {
            return Err(self.refuse(self.per_fire.get().unwrap_or(0.0), count));
        }
        let mut f = self.device.frame()?;
        {
            let sink = B::sink(self.device, &f, &self.pipelines, &self.handles);
            for _ in 0..count {
                fire(&sink)?;
            }
        }
        then(&mut f)?;
        let timed = timer(f)?;
        let took = timed.0.unwrap_or(timed.1) as f64;
        let mut admission = self.admission.get();
        admission.landed(estimate, took, count >= self.basis.get());
        self.admission.set(admission);
        self.per_fire.set(Some(took / f64::from(count.max(1))));
        self.basis.set(count);
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
    vectors: bool,
    tiles: bool,
    attn: bool,
    elementwise: bool,
    scan: bool,
}

impl Wanted {
    fn of(stated: &Rates) -> Wanted {
        let lacked = stated.missing(&KEYS);
        let lacks = |key: &str| lacked.contains(&key);
        Wanted {
            launch: lacks("launch_ns") || lacks("hbm_gbps"),
            gemm: lacks("gemm_tflops_by_rows"),
            vectors: lacks("vector_tflops_by_rows"),
            tiles: lacks("tile_tflops_by_rows"),
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
    let want = Wanted::of(stated);
    let mut rig = Rig::<B>::new(device, max_submit);
    // The launch first: every rate below takes its dispatches' launches off.
    let (bytes_per_ns, launch_ns) = match (want.launch, stated.hbm_gbps, stated.launch_ns) {
        (false, Some(bytes), Some(launch)) => (bytes, launch),
        _ => {
            let measured = copy_and_launch(&mut rig, tactics.gemm(tier, LAUNCH), timer, say)?;
            rig.release();
            measured
        }
    };
    let launch = launch_ns as f64;
    let (side, warmed) = warm(&mut rig, &|rows| tactics.gemm(tier, rows), timer)?;
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
            let rate = matmul(&mut rig, tactics.gemm(tier, m), timer, (m, w, w), launch);
            rig.release();
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
    // `vector_tflops_by_rows`: each vector kernel, by the rows a workgroup
    // carries, at `VECTOR_ROWS_TIMED` of the widest plane, each fire reading
    // the next plane past the last-level cache: its flops over its time,
    // every weight pass it makes in it.
    let w = engine::rates::GEMM_WIDTH;
    let mut vector_curves: Vec<(u32, Vec<(u32, f64)>)> = Vec::new();
    let carried = if want.vectors {
        kernels_slang::linear::gemm::VECTOR_ROWS.to_vec()
    } else {
        Vec::new()
    };
    for rows in carried {
        let vector = kernels_slang::linear::gemm::Tactic {
            vector: rows,
            ..tactics.gemm(tier, 1)
        };
        let mut curve = Vec::new();
        for m in VECTOR_ROWS_TIMED {
            let rate = matmul(&mut rig, vector, timer, (m, w, w), launch);
            rig.release();
            let Some(rate) = dropped(&rig, rate, say)? else {
                continue;
            };
            say(&format!(
                "[rates]: vector matmul ({rows} rows a workgroup) {m}x{w}x{w}: {:.1} TFLOP/s",
                rate / 1e3
            ));
            curve.push((m, rate / 1e3));
        }
        if !curve.is_empty() {
            vector_curves.push((rows, curve));
        }
    }
    // `tile_tflops_by_rows`: each tiled kernel the backend can take, at
    // `TILE_ROWS` of the widest plane, its flops at the rows it lands over
    // its time, keyed by its row tile.
    let mut tile_curves: Vec<(u32, Vec<(u32, f64)>)> = Vec::new();
    // Every tiled kernel the profile's path reaches at some timed rows (a
    // device whose path takes the 32x32 tile below some rows and the
    // cooperative-matrix tile above times both).
    let mut tiled_kernels: Vec<kernels_slang::linear::gemm::Tactic> = Vec::new();
    for m in TILE_ROWS.into_iter().filter(|_| want.tiles) {
        for tactic in crate::dense::tiled_kernels(B::TARGET, tactics.gemm(tier, m)) {
            if !tiled_kernels.contains(&tactic) {
                tiled_kernels.push(tactic);
            }
        }
    }
    for tactic in tiled_kernels {
        let tile = kernels_slang::linear::gemm::tile_for(B::TARGET, tactic, w, w).row_tile();
        let mut curve = Vec::new();
        for m in TILE_ROWS {
            let rate = matmul(&mut rig, tactic, timer, (m, w, w), launch);
            rig.release();
            let Some(rate) = dropped(&rig, rate, say)? else {
                continue;
            };
            say(&format!(
                "[rates]: {tile}-row tile matmul {m}x{w}x{w}: {:.1} TFLOP/s",
                rate / 1e3
            ));
            curve.push((m, rate / 1e3));
        }
        if !curve.is_empty() && tile_curves.iter().all(|(at, _)| *at != tile) {
            tile_curves.push((tile, curve));
        }
    }
    tile_curves.sort_by_key(|(tile, _)| *tile);
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
    // A shape's first fire, estimated by its work at the last shape's rate.
    let at_rate =
        |shape: engine::rates::AttnShape, rate: Option<f64>| first_fire_ns(shape, rate, launch);
    let attn_keys = if want.attn {
        engine::rates::attn_keys(peak / bytes_per_ns.max(f64::MIN_POSITIVE))
    } else {
        Vec::new()
    };
    for keys in attn_keys {
        let shape = engine::rates::ATTN_SHAPE.at(keys);
        let estimate = at_rate(shape, attn.last().map(|&(_, rate)| rate));
        let rate = attention(&mut rig, tactics, tier, timer, shape, launch, estimate);
        rig.release();
        let Some(rate) = dropped(&rig, rate, say)? else {
            continue;
        };
        say(&format!(
            "[rates]: attention {keys} keys d 128: {:.1} TFLOP/s",
            rate / 1e3
        ));
        attn.push((keys, rate));
    }
    let longest = attn
        .last()
        .map_or(engine::rates::ATTN_KEYS[0], |&(keys, _)| keys);
    for d in engine::rates::ATTN_DIMS
        .into_iter()
        .filter(|&d| want.attn && d != 128)
    {
        let shape = engine::rates::ATTN_SHAPE.at(longest).with_dim(d);
        let estimate = at_rate(shape, attn.last().map(|&(_, rate)| rate));
        let rate = attention(&mut rig, tactics, tier, timer, shape, launch, estimate);
        rig.release();
        let Some(rate) = dropped(&rig, rate, say)? else {
            continue;
        };
        say(&format!(
            "[rates]: attention {longest} keys d {d}: {:.1} TFLOP/s",
            rate / 1e3
        ));
    }
    let elementwise = if want.elementwise {
        let gbps = elementwise(&mut rig, timer, launch)?;
        rig.release();
        say(&format!("[rates]: per-row ops: {gbps:.0} GB/s"));
        Some(gbps)
    } else {
        None
    };
    let scan = if want.scan {
        let gelems = scan(&mut rig, tactics.recurrent(), timer, SCAN_SHAPE, launch)?;
        rig.release();
        Some(gelems)
    } else {
        None
    };
    let tflops = |curve: Vec<(u32, f64)>| -> Vec<(u32, f64)> {
        curve.into_iter().map(|(x, r)| (x, r / 1e3)).collect()
    };
    let rates = Rates {
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
        tile_tflops_by_rows: (!tile_curves.is_empty()).then_some(tile_curves),
        vector_tflops_by_rows: (!vector_curves.is_empty()).then_some(vector_curves),
        workgroups_per_core: None,
    };
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
    })
}

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
    rig.step(format!("per-row ops {}x{}", rows, width), None);
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
    .map_err(|why| Fault::Device { call: "rates", why })
}

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
) -> Result<(u32, engine::rates::Warmed)> {
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
        rig.step(format!("warm-up walk {p}^3 (first fire)"), None);
        rig.frame(timer, 1, &fire, &nothing)?;
        rig.step(format!("warm-up walk {p}^3"), estimate);
        let (_, ns) = rig.frame(timer, 1, &fire, &nothing)?;
        rig.release();
        Ok::<f64, Fault>(ns as f64)
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
                return Ok((side, warmed));
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
    let weights = (0..weights_at(n, k))
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
    rig.step(format!("warm-up {side}^3 (first fire)"), None);
    rig.frame(timer, 1, &fire, &nothing)?;
    rig.step(format!("warm-up {side}^3 rotation"), rig.per_fire.get());
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
    let rotations = std::cell::Cell::new(engine::rates::Admission {
        settled: true,
        worst: walk_worst.max(1.0),
        ..engine::rates::Admission::new(rig.max_submit)
    });
    let slowest = std::cell::Cell::new(rig.per_fire.get());
    let fastest = std::cell::Cell::new(None::<f64>);
    let per_fire = || match (slowest.get(), rig.per_fire.get()) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    };
    let fits = |left: u32| {
        rotation_fires(
            rotations.get().bound,
            per_fire(),
            rotations.get().worst,
            left,
        )
    };
    let shrunk = std::cell::Cell::new(false);
    let warmed = engine::rates::warm_up(|| {
        let mut wall = 0u64;
        let mut left = planes;
        while left > 0 {
            let count = fits(left);
            if count == 0 {
                // Not one fire fits: the rotations stop here, and the cube
                // halves ([`warm`]).
                shrunk.set(true);
                return Err(Fault::Device {
                    call: "rates",
                    why: format!("warm-up: a {side}^3 fire no longer fits the bound"),
                });
            }
            let estimate = per_fire().map(|p| p * f64::from(count));
            let took = rig.frame(timer, count, &fire, &nothing)?.1;
            slowest.set(per_fire());
            let this = took as f64 / f64::from(count.max(1));
            fastest.set(Some(fastest.get().map_or(this, |f: f64| f.min(this))));
            if let (Some(fast), Some(slow)) = (fastest.get(), slowest.get()) {
                rig.swing.set(rig.swing.get().max(slow / fast.max(1.0)));
            }
            let mut r = rotations.get();
            r.landed(estimate, took as f64, true);
            rotations.set(r);
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
    let weights = (0..weights_at(n, k))
        .map(|_| rig.operand(n, k, Dtype::Bf16))
        .collect::<Result<Vec<Tensor>>>()?;
    let y = rig.zeros(m, n, Dtype::Bf16)?;
    let next = std::cell::Cell::new(0usize);
    rig.step(format!("matmul {m}x{n}x{k}"), None);
    let ns = rig.each(timer, &Touched::default(), weights.len(), &|sink| {
        let w = weights[next.get() % weights.len()];
        next.set(next.get() + 1);
        kernels_slang::linear::gemm::matmul(sink, tactic, act, w, y).map_err(fault("rates: matmul"))
    })?;
    engine::rates::rate_from(
        &format!("matmul {m}x{n}x{k}"),
        ns as f64,
        engine::rates::GEMM_DISPATCHES,
        launch,
        gemm_flops(m, n, k),
    )
    .map_err(|why| Fault::Device { call: "rates", why })
}

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
    estimate: Option<f64>,
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
    let sets = (0..engine::rates::attn_sets_at(s, PAGE).min(ATTN_SETS_MOST))
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
    // The rows' own positions, all of request 0, no row masked.
    let positions: Vec<u32> = (0..s.rows).collect();
    let tables = kernels_slang::attn::sdpa::Tables {
        positions: rig.words(&positions, s.rows, 1, Dtype::I32)?,
        request_of_token: rig.zeros(s.rows, 1, Dtype::I32)?,
        mask: rig.zeros(1, 4, Dtype::U8)?,
        mask_words: rig.words(&crate::mask::none(s.rows as usize), s.rows, 2, Dtype::U32)?,
    };
    let scale = 1.0 / (s.head_dim as f32).sqrt();
    // A new shape's first fire is estimated before it is submitted (from
    // the rate the step's last shape measured, by the work), so one past the
    // device's bound is refused by name unsubmitted, never run to find out.
    rig.step(
        format!("attention {}x{} d {}", s.rows, s.keys, s.head_dim),
        estimate,
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
    .map_err(|why| Fault::Device { call: "rates", why })
}

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
    let next = std::cell::Cell::new(0usize);
    rig.step(format!("scan {} tokens", s.tokens), None);
    let ns = rig.each(timer, &Touched::default(), sets.len(), &|sink| {
        let (raw, conv_state, scan_state) = &sets[next.get() % sets.len()];
        let raw = *raw;
        next.set(next.get() + 1);
        kernels_slang::ssm::conv1d::conv1d_chunked(
            sink,
            RaggedTensor { data: raw, indptr },
            weight,
            conv_state,
            s.conv_width,
            1,
            qkv,
        )
        .map_err(fault("rates: conv"))?;
        kernels_slang::ssm::gdn_recurrent::gated_delta_chunked(
            sink,
            tactic,
            RaggedTensor { data: qkv, indptr },
            gates,
            scan_state,
            s.k_heads,
            s.v_heads,
            s.k_dim,
            s.v_dim,
            y,
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
    .map_err(|why| Fault::Device { call: "rates", why })
}

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
    rig.step(format!("launch: one {LAUNCH}^3 matmul"), None);
    rig.frame(timer, 1, &tiny, &nothing)?;
    let single = median(
        (0..ROUNDS)
            .map(|_| rig.frame(timer, 1, &tiny, &nothing).map(|(_, w)| w))
            .collect::<Result<Vec<u64>>>()?,
    );
    // The launch: a chain of `n` dependent tiny matmuls in one frame (each
    // reads the one before's output) against one matmul of `n` times the
    // rows, at each chain length of the lattice.
    let mut gaps = Vec::with_capacity(engine::rates::LAUNCH_CHAINS.len());
    for n in engine::rates::LAUNCH_CHAINS {
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
        rig.step(format!("launch: a chain of {n} {LAUNCH}^3 matmuls"), None);
        let chain = wall(n, &chained)?;
        rig.step(
            format!("launch: one {}x{LAUNCH}x{LAUNCH} matmul", LAUNCH * n),
            None,
        );
        let whole = wall(1, &one)?;
        gaps.push((n, engine::rates::launch_from(chain as f64, whole as f64, n)));
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

    let device = rig.device;
    let copy = COPY_BYTES.min(device.max_buffer()) & !3;
    let from = B::Buffer::zeroed(device, copy, crate::books::owners::TUNING)?;
    let into = B::Buffer::zeroed(device, copy, crate::books::owners::TUNING)?;
    // The copy rides after the narrowest matmul, whose time is taken off;
    // by the adapter where it timed every round, else the host.
    let timed = |then: &dyn Fn(&mut B::Frame) -> Result<()>| -> Result<(u64, bool)> {
        rig.frame(timer, 1, &tiny, then)?;
        let (mut adapter, mut host) = (Vec::new(), Vec::new());
        for _ in 0..ROUNDS {
            let (span, wall) = rig.frame(timer, 1, &tiny, then)?;
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
        None,
    );
    engine::rates::warm_up(|| {
        let (span, wall) = rig.frame(timer, 1, &tiny, &|f| f.copy(&from, 0, &into, 0, copy))?;
        Ok::<f64, Fault>(1.0 / (span.unwrap_or(wall) as f64).max(1.0))
    })?;
    let (with_copy, by_adapter) = timed(&|f| f.copy(&from, 0, &into, 0, copy))?;
    let alone = if by_adapter {
        rig.step(format!("copy: one {LAUNCH}^3 matmul alone"), None);
        timed(&nothing)?.0
    } else {
        single
    };
    let copy_ns = with_copy.saturating_sub(alone).max(1);
    // Every byte copied is read and written.
    // Every byte copied is read and written.
    let bytes = engine::rates::rate_from(
        "copy",
        copy_ns as f64,
        engine::rates::COPY_DISPATCHES,
        launch as f64,
        2.0 * copy as f64,
    )
    .map_err(|why| Fault::Device { call: "rates", why })?;
    Ok((bytes, launch))
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
    attention(&mut rig, tactics, tier, timer, shape, launch, None)
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
    rig.step(format!("check matmul {m}x{n}x{k}"), None);
    let ns = rig.each(timer, &Touched::default(), 1, &|sink| {
        kernels_slang::linear::gemm::matmul(sink, tactic, act, w, y).map_err(fault("rates: matmul"))
    })?;
    Ok(2.0 * f64::from(m) * f64::from(n) * f64::from(k) / ns as f64)
}

/// The fires a warm-up rotation's next frame holds of `left`: as many as fit
/// under `bound` at `per_fire` ns a fire (the slowest the rotation measured)
/// times `worst` (at least the walk's measured worst ratio); 0 where not one
/// fits; all of them where no bound is stated or nothing is measured yet.
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
        _ => left,
    }
}

/// A new attention shape's first fire, estimated before it is submitted: its
/// work at the rate the step's last shape measured, plus a launch; none
/// where no shape has been measured yet.
fn first_fire_ns(shape: engine::rates::AttnShape, rate: Option<f64>, launch: f64) -> Option<f64> {
    rate.map(|rate| attn_flops(shape) / rate.max(f64::MIN_POSITIVE) + launch)
}

#[cfg(test)]
mod wanted_tests {
    use super::*;

    /// A profile lacking one key times that key alone (the launch and the
    /// copy together); one stating none times every one.
    #[test]
    fn a_step_times_only_what_the_profile_lacks() {
        let all = Rates {
            launch_ns: Some(3000),
            gemm_tflops_by_rows: Some(vec![(4096, vec![(256, 10.0)])]),
            attn_tflops_by_keys: Some(vec![(512, 5.0)]),
            hbm_gbps: Some(650.0),
            cached_gbps: None,
            elementwise_gbps: Some(700.0),
            scan_gelems: Some(100.0),
            few_row_tflops: None,
            workgroups_per_core: None,
            tile_tflops_by_rows: Some(vec![(32, vec![(4, 1.0)])]),
            vector_tflops_by_rows: Some(vec![(4, vec![(1, 0.5)])]),
        };
        let none = Wanted {
            launch: false,
            gemm: false,
            vectors: false,
            tiles: false,
            attn: false,
            elementwise: false,
            scan: false,
        };
        assert_eq!(Wanted::of(&all), none);
        let few = Rates {
            vector_tflops_by_rows: None,
            ..all.clone()
        };
        assert_eq!(
            Wanted::of(&few),
            Wanted {
                vectors: true,
                ..none
            }
        );
        let copy = Rates {
            hbm_gbps: None,
            cached_gbps: None,
            ..all.clone()
        };
        assert_eq!(
            Wanted::of(&copy),
            Wanted {
                launch: true,
                ..none
            }
        );
        let every = Wanted::of(&Rates::default());
        assert!(
            every.launch
                && every.gemm
                && every.vectors
                && every.tiles
                && every.attn
                && every.elementwise
                && every.scan
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
        // Nothing measured yet: the whole rotation (the walk admitted it).
        assert_eq!(rotation_fires(Some(BOUND), None, 1.4, 8), 8);
    }

    /// A new shape's first fire is estimated from the last shape's rate by
    /// its work: d256 after d64 at the same keys is estimated at about the
    /// work ratio, so on a device where d64 took 1.29 s the d256 fire is
    /// estimated past the 2 s bound and is refused before it is submitted.
    #[test]
    fn a_new_attention_shape_is_estimated_by_its_work_before_its_first_fire() {
        let d64 = engine::rates::ATTN_SHAPE.at(8192).with_dim(64);
        let d256 = engine::rates::ATTN_SHAPE.at(8192).with_dim(256);
        assert_eq!(first_fire_ns(d256, None, 0.0), None);
        let took = 1.29e9;
        let rate = attn_flops(d64) / took;
        let estimate = first_fire_ns(d256, Some(rate), 0.0).unwrap();
        let ratio = attn_flops(d256) / attn_flops(d64);
        assert!((estimate / took - ratio).abs() < 1e-9);
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
    }
}
