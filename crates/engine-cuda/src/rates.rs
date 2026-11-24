//! The device's rates measured on the device, through the kernels serving
//! dispatches (`pie tune cuda --rates`, and on ROCm the same steps over
//! HIP's): the roofline constants the top rung's estimate is weighed by.
//!
//! Each rate is a burst of its kernel launched back to back on the stream,
//! `k` launches, `k` doubled until a launch's time settles
//! (`engine::rates::timed_frames`, every engine's rule), as serving runs a
//! fire's: a lone launch lets the device idle between timings, and its
//! clock state, not the kernel, sets the number.
//! The burst is timed by events on the stream, the median of [`ROUNDS`]
//! bursts ([`LAUNCH_ROUNDS`] for the launch) after a warm one kept, and
//! divided by `k`.
//!
//! Every shape is `engine::rates`', and its counts turn the times into
//! rates, so the CUDA and ROCm rates mean what every backend's do:
//! - `launch_ns`: a chain of dependent near-empty kernels captured in one
//!   graph against one kernel of the chain's whole work, at each length of
//!   `engine::rates::LAUNCH_CHAINS` (`engine::rates::launch_from`): what a
//!   dispatch adds to its work inside a recorded body.
//! - `gemm_tflops_by_rows`: the matmul `m x w x w` at each of
//!   `engine::rates::GEMM_WIDTHS` and `engine::rates::GEMM_ROWS` through `linear::gemm` (cuBLASLt's first
//!   heuristic algorithm on CUDA, hipBLASLt's on ROCm), after the warm-up;
//!   its launches rotate through `engine::rates::weights_at` weight planes,
//!   past every last-level cache.
//! - `attn_tflops_by_keys`: the paged prefill attention at [`ATTN_SHAPE`]'s
//!   heads over each of `engine::rates::ATTN_KEYS` keys, causal, in pages of
//!   [`PAGE`], head width 128 (64 and 256 timed at the longest keys and said
//!   beside it); its launches rotate through `engine::rates::attn_sets_at`
//!   sets of q rows and kv pages.
//! - `hbm_gbps`: a [`COPY_BYTES`] device-to-device copy, read plus write.
//! - `cached_gbps` (ROCm): the reads the last-level caches serve, as a
//!   normed skinny product's blocks read its rows: [`CACHED_BLOCKS`] blocks
//!   each reading one [`CACHED_SET_BYTES`] set once, block b the set b %
//!   [`CACHED_SETS`] (the sets together past a compute unit's first-level
//!   cache, within the last-level ones), their bytes over their time.
//! - `elementwise_gbps`: the rmsnorm and the swiglu as a fire runs them at
//!   `engine::rates::ELEMENTWISE_SHAPE`, rotating through
//!   `engine::rates::elementwise_sets` sets.
//! - `scan_gelems`: the causal conv and the chunked gated delta scan at
//!   [`SCAN_SHAPE`].
//! - `few_row_tflops`: the plain product of [`FEW_ROWS`] (the few-rows
//!   kernel's most rows: the rows skeleton's 16 on CUDA, the skinny
//!   kernel's 8 on ROCm) over one `engine::rates::GEMM_WIDTH`-square weight
//!   plane through `linear::gemm::matmul`, the kernel serving picks there,
//!   its flops (its kernel pads no row) over its time. The plane is not
//!   rotated: after the first fire its 32 MiB is read from the last-level
//!   cache, so the weight reads no longer bind it and the time is the
//!   kernel's compute.
//! - `workgroups_per_core`: derived, not timed: the blocks of that same
//!   kernel, at that shape, one multiprocessor (compute unit) holds at
//!   once, as the device's occupancy calculator states it for the kernel's
//!   block size, registers and shared memory
//!   (`cuOccupancyMaxActiveBlocksPerMultiprocessor` on CUDA,
//!   `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor` on ROCm;
//!   `linear::gemm::few_rows_resident_blocks`). Nothing multiplies it: the
//!   device's full-rate workgroups are it times the multiprocessors
//!   (`engine::roofline::Constants::full_rate_workgroups`).
//!
//! Every float operand the matmul, attention and scan read is
//! `engine::rates::operand_pattern` repeated, not zeros: some devices run a
//! compute-bound kernel faster on zeros than on the real data serving runs.
//! Index and page tables, gates, outputs and workspace stay as laid.

#[cfg_attr(not(feature = "device"), allow(unused_imports))]
use engine::rates::{ATTN_SHAPE, SCAN_SHAPE, Touched, weights_at};
use profile::fire::Rates;

/// The `[rates]` keys this backend's estimate weighs a fire by (of
/// `profile::fire::KEYS`, every one):
/// what a load whose profile lacks one measures.
#[cfg(not(feature = "rocm"))]
pub const KEYS: [&str; 8] = [
    "launch_ns",
    "gemm_tflops_by_rows",
    "attn_tflops_by_keys",
    "hbm_gbps",
    "elementwise_gbps",
    "scan_gelems",
    "few_row_tflops",
    "workgroups_per_core",
];

/// On ROCm also `cached_gbps`, the cached-read rate a normed skinny
/// product's redundant reads are priced at (`kernels_rocm::tuning`).
#[cfg(feature = "rocm")]
pub const KEYS: [&str; 9] = [
    "launch_ns",
    "gemm_tflops_by_rows",
    "attn_tflops_by_keys",
    "hbm_gbps",
    "cached_gbps",
    "elementwise_gbps",
    "scan_gelems",
    "few_row_tflops",
    "workgroups_per_core",
];

/// The `[provenance]` entry a measurement of `device`'s rates writes beside
/// them (`engine::rates::provenance`): how this backend measured them.
#[must_use]
pub fn provenance(device: &str) -> (&'static str, String) {
    let how = format!(
        "{device}; the matmul through {}'s first heuristic algorithm, each rate a burst on the \
         stream doubled until it settles, timed by events, the median of 5 (the launch of 15); \
         the few-rows rate on {FEW_ROWS} rows over one cached {w}-square plane; \
         workgroups_per_core the few-rows kernel's resident blocks a multiprocessor by the \
         occupancy calculator; operands: engine::rates' pattern; at {}",
        if cfg!(feature = "rocm") {
            "hipBLASLt"
        } else {
            "cuBLASLt"
        },
        engine::rates::build_commit(),
        w = engine::rates::GEMM_WIDTH,
    );
    engine::rates::provenance("cuda", &how)
}

/// `chip` with every roofline constant stated
/// ([`engine::rates::rates_at_load`], every engine's rule): as its layers
/// state them where they state them all; else the device's own, measured
/// here at the start of the load, before any pool or buffer of it is laid
/// (the step `pie tune cuda --rates` runs, seconds long, each timed
/// submission a burst settled by `engine::rates::timed_frames`: CUDA states
/// no bound on one submission), the keys it lacked saved to the user's
/// profile of `device` under `profiles` with their provenance so later
/// loads read them. The step's buffers come of `room`, what the load's
/// budget leaves on the device: where it holds less than the step lays at
/// most (`engine::rates::step_bytes`) the load is refused by name, never
/// left to wait for memory.
#[allow(clippy::too_many_arguments)]
pub fn ensure(
    chip: profile::Profile,
    ordinal: i32,
    device: &str,
    keys: &crate::stored::Keys,
    profiles: Option<&std::path::Path>,
    cache: Option<&std::path::Path>,
    room: u64,
    measuring: engine::rates::Measuring,
) -> Result<profile::Profile, String> {
    ensure_by(chip, device, keys, profiles, room, measuring, |chip| {
        measure(ordinal, chip, cache).map(|m| m.rates)
    })
}

/// [`ensure`] with the measurement `measure`. What it measures is keyed by
/// `keys` (the device and what measured each rate) before it is saved, so a
/// later load takes it where the keys still match
/// ([`engine::rates::resolve_at_load`]); under [`engine::rates::Measuring::Skip`]
/// nothing is timed and each rate no layer states is served by the
/// estimate, by name.
fn ensure_by(
    chip: profile::Profile,
    device: &str,
    keys: &crate::stored::Keys,
    profiles: Option<&std::path::Path>,
    room: u64,
    measuring: engine::rates::Measuring,
    measure: impl FnOnce(&profile::Profile) -> Result<Rates, String>,
) -> Result<profile::Profile, String> {
    let save = profiles.map(|profiles| engine::rates::SaveTo {
        path: crate::chip::SHELF.user_path(profiles, device),
        provenance: provenance(device),
        registry: crate::chip::registry(),
    });
    let measure_chip = chip.clone();
    engine::rates::resolve_at_load(
        chip,
        &KEYS,
        save,
        Some(room),
        None,
        measuring,
        |_max_submit| {
            Ok(engine::rates::Measured {
                rates: keys.stamp(measure(&measure_chip)?),
                unsettled: false,
                throttled: None,
            })
        },
    )
}

/// What a load measured of its decode products (`Shell::routes_by_body`):
/// each shape under each kernel, the median ns of the decode step whose
/// body serves it so, and the interquartile spread of the rounds.
pub type ProductTimes = Vec<(kernels_cuda::linear::routes::Shape, u32, f64, f64)>;

/// `measured` saved in device `device`'s user profile under `profiles`
/// (`[rates] product_ns`, joining `held`, what the load's profile states
/// under a matching key: each newly measured entry in place of one of the
/// same key), keyed by `keys` (the device, and what measured the table) so
/// a later load takes it where they still match; the path, or `None` where
/// the load names no profile directory.
pub fn save_product_times(
    profiles: Option<&std::path::Path>,
    device: &str,
    keys: &crate::stored::Keys,
    held: &[profile::fire::ProductTime],
    measured: &ProductTimes,
) -> Result<Option<String>, String> {
    let Some(profiles) = profiles else {
        return Ok(None);
    };
    let fresh = kernels_cuda::linear::routes::product_times(measured);
    let mut joined: Vec<profile::fire::ProductTime> = held
        .iter()
        .filter(|old| fresh.iter().all(|new| new.key() != old.key()))
        .cloned()
        .collect();
    joined.extend(fresh);
    let rates = keys.stamp(Rates {
        product_ns: Some(joined),
        ..Rates::default()
    });
    let save = engine::rates::SaveTo {
        path: crate::chip::SHELF.user_path(profiles, device),
        provenance: (
            "product_ns",
            format!(
                "\"measured in place at load, {}\"",
                engine::rates::build_commit()
            ),
        ),
        registry: crate::chip::registry(),
    };
    engine::rates::save_rates(&save, &rates)?;
    Ok(Some(save.path.display().to_string()))
}

/// How this machine lands weights (`measured`, `[rates] landing`), keyed,
/// saved in the device's user profile under `profiles`: a later load reads
/// it there and measures none of it.
pub fn save_landing(
    profiles: Option<&std::path::Path>,
    device: &str,
    keys: &crate::stored::Keys,
    measured: profile::fire::Landing,
) -> Result<Option<String>, String> {
    let Some(profiles) = profiles else {
        return Ok(None);
    };
    let rates = keys.stamp(Rates {
        landing: Some(measured),
        ..Rates::default()
    });
    let save = engine::rates::SaveTo {
        path: crate::chip::SHELF.user_path(profiles, device),
        provenance: (
            profile::fire::LANDING,
            format!(
                "\"measured in place at load, {}\"",
                engine::rates::build_commit()
            ),
        ),
        registry: crate::chip::registry(),
    };
    engine::rates::save_rates(&save, &rates)?;
    Ok(Some(save.path.display().to_string()))
}

/// The rows `few_row_tflops` is timed at: the most the backend's few-rows
/// kernel takes (`linear::gemm::FEW_ROWS`).
pub const FEW_ROWS: u32 = kernels_cuda::linear::gemm::FEW_ROWS;

/// The kv cache's page, as a served load pages it.
pub const PAGE: u32 = 16;

/// The copy's bytes: past every last-level cache, so it moves device memory.
pub const COPY_BYTES: usize = 256 << 20;

/// The cached-read probe's set: a few rows of a model's width, bf16.
pub const CACHED_SET_BYTES: usize = 16 << 10;
/// Its sets: together past a compute unit's first-level cache.
pub const CACHED_SETS: usize = 64;
/// Its blocks: enough that their reads, not the launch, set the time.
pub const CACHED_BLOCKS: u32 = 16384;

/// Timed bursts of each kind; the median is kept.
pub const ROUNDS: usize = 5;

/// Timed bursts of the launch's graph: a dispatch's cost moves more burst to
/// burst than a kernel's work (an MI300X VF's 1610 to 1719 ns over 5).
pub const LAUNCH_ROUNDS: usize = 15;

/// What a rates step measured: the rates, and the attention's rate at the
/// longest keys at each head width of `engine::rates::ATTN_DIMS` (TFLOP/s),
/// said beside them.
pub struct Measured {
    pub rates: Rates,
    pub attn_by_dim: Vec<(u32, f64)>,
}

/// The rates of device `ordinal` under `chip`'s tuning, the kernels' disk
/// cache under `cache` (`$PIE_HOME/cache`).
pub fn measure(
    ordinal: i32,
    chip: &profile::Profile,
    cache: Option<&std::path::Path>,
) -> Result<Measured, String> {
    kernels_cuda::disk::install(cache);
    #[cfg(feature = "device")]
    {
        let measured = on_device::measure(ordinal, chip);
        // The step's buffers (laid apart from the device heap, whose chunks
        // the process keeps) went to the idle store as they dropped; the
        // load reads the card's room next (the top rung's fit), so they go
        // back to the driver first. Kept, an L40S's first boot fit beside
        // 7.5 GiB the step no longer held (4.6 GiB idle, 2.5 GiB of heap
        // chunks): at a 40960 context every rung lacked room for one
        // sequence, while the next boot, reading the saved rates, served.
        crate::device::held::release_idle();
        measured
    }
    #[cfg(not(feature = "device"))]
    {
        let _ = (ordinal, chip);
        Err(crate::error::Fault::Runtimeless.to_string())
    }
}

#[cfg(feature = "device")]
#[allow(unsafe_code)]
mod on_device {
    use core::ffi::c_void;
    use core::sync::atomic::{AtomicUsize, Ordering};

    use kernels_cuda::attn::plan::{Live, Shape, Workspace};
    use kernels_cuda::{Ctx, KvPool, RaggedTensor, RecurrentPool, Slabs, Tensor};
    use poem_ir::Dtype;

    use super::*;
    use crate::device::{Context, Graph, rt};

    /// The device bytes the step's buffers hold now, and the most they held.
    static LIVE: AtomicUsize = AtomicUsize::new(0);
    static PEAK: AtomicUsize = AtomicUsize::new(0);

    /// A device buffer the step holds, counted in [`LIVE`] and [`PEAK`]
    /// while it lives.
    struct Buffer(crate::device::Buffer);

    impl Buffer {
        fn zeroed(bytes: usize) -> crate::error::Result<Buffer> {
            let b = crate::device::Buffer::zeroed_apart(bytes)?;
            let live = LIVE.fetch_add(b.bytes(), Ordering::Relaxed) + b.bytes();
            PEAK.fetch_max(live, Ordering::Relaxed);
            Ok(Buffer(b))
        }
    }

    impl core::ops::Deref for Buffer {
        type Target = crate::device::Buffer;
        fn deref(&self) -> &crate::device::Buffer {
            &self.0
        }
    }

    impl core::ops::DerefMut for Buffer {
        fn deref_mut(&mut self) -> &mut crate::device::Buffer {
            &mut self.0
        }
    }

    impl Drop for Buffer {
        fn drop(&mut self) {
            LIVE.fetch_sub(self.0.bytes(), Ordering::Relaxed);
        }
    }

    type Kernel = kernels_cuda::Error;

    fn say<E: std::fmt::Display>(what: &'static str) -> impl Fn(E) -> String {
        move |e| format!("rates: {what}: {e}")
    }

    /// Two events on a stream, timing the work between them.
    struct Clock {
        start: rt::cudaEvent_t,
        stop: rt::cudaEvent_t,
        stream: *mut c_void,
    }

    impl Clock {
        fn new(stream: *mut c_void) -> Result<Clock, String> {
            let mut start: rt::cudaEvent_t = core::ptr::null_mut();
            let mut stop: rt::cudaEvent_t = core::ptr::null_mut();
            // SAFETY: live out-parameters; this thread bound the device.
            unsafe {
                crate::device::ctx::check("cudaEventCreate", rt::cudaEventCreate(&raw mut start))
                    .map_err(say("an event"))?;
                crate::device::ctx::check("cudaEventCreate", rt::cudaEventCreate(&raw mut stop))
                    .map_err(say("an event"))?;
            }
            Ok(Clock {
                start,
                stop,
                stream,
            })
        }

        /// The ns `work` takes on the stream.
        fn time(&self, work: &mut dyn FnMut() -> Result<(), String>) -> Result<f64, String> {
            let mut ms = 0f32;
            // SAFETY: the events are this clock's, the stream the context's.
            unsafe {
                crate::device::ctx::check(
                    "cudaEventRecord",
                    rt::cudaEventRecord(self.start, self.stream.cast()),
                )
                .map_err(say("an event"))?;
            }
            work()?;
            // SAFETY: as above; the stop lands before the elapsed time is read.
            unsafe {
                crate::device::ctx::check(
                    "cudaEventRecord",
                    rt::cudaEventRecord(self.stop, self.stream.cast()),
                )
                .map_err(say("an event"))?;
                crate::device::ctx::check(
                    "cudaEventSynchronize",
                    rt::cudaEventSynchronize(self.stop),
                )
                .map_err(say("an event"))?;
                crate::device::ctx::check(
                    "cudaEventElapsedTime",
                    rt::cudaEventElapsedTime(&raw mut ms, self.start, self.stop),
                )
                .map_err(say("an event"))?;
            }
            Ok(f64::from(ms) * 1e6)
        }

        /// The ns one launch of `once(i)` takes, `i` the launch's place in
        /// the measurement (one counter across every burst of it,
        /// `engine::rates::Rotation`, so a burst reads the next operand
        /// sets, never the first again): a burst of `k` launches, `k`
        /// doubled until a launch's time settles, then such bursts until
        /// their rate stops moving (`engine::rates::warmed_frames`; CUDA
        /// states no bound on one submission), the median of `rounds` after
        /// a warm burst, over its launches.
        fn burst(
            &self,
            rounds: usize,
            touched: &Touched,
            sets: usize,
            once: &mut dyn FnMut(usize) -> Result<(), String>,
        ) -> Result<f64, String> {
            let turn = engine::rates::Rotation::default();
            let mut fire = |k: usize| (0..k).try_for_each(|_| once(turn.next(usize::MAX)));
            // Its sets touched once, by the first burst over them.
            engine::rates::touch_once(
                touched,
                None,
                u32::try_from(sets).unwrap_or(u32::MAX),
                &mut |k| self.time(&mut || fire(k as usize)),
            )?;
            self.time(&mut || fire(1))?;
            let k = engine::rates::warmed_frames(
                None,
                None,
                |k| self.time(&mut || fire(k as usize)),
                |why| why,
            )?
            .fires as usize;
            self.time(&mut || fire(k))?;
            let mut times = (0..rounds)
                .map(|_| self.time(&mut || fire(k)))
                .collect::<Result<Vec<f64>, String>>()?;
            times.sort_by(f64::total_cmp);
            Ok(times[rounds / 2] / k as f64)
        }
    }

    impl Drop for Clock {
        fn drop(&mut self) {
            // SAFETY: created by `new`, destroyed once, after the last sync.
            unsafe {
                let _ = rt::cudaEventDestroy(self.start);
                let _ = rt::cudaEventDestroy(self.stop);
            }
        }
    }

    /// Lays the scratch slabs `launch` takes on its context (a size-only
    /// walk of it, as a load lays its fires'), in the one buffer `held`
    /// keeps: zeroed again where it is wide enough, else replaced by a wider
    /// one. Every fire before has landed (each timing waits on its stop
    /// event) and the launch's graph is gone, so nothing still reads the
    /// slabs laid before; one buffer a point would hold the grid's 1425.
    fn lay(
        held: &mut Vec<Buffer>,
        launch: impl FnOnce() -> Result<(), Kernel>,
    ) -> Result<(), String> {
        crate::sizing::begin();
        let _ = launch();
        let placed = crate::sizing::end().placed();
        if placed.is_empty() {
            return Ok(());
        }
        let bytes = Slabs::placed_bytes(&placed);
        match held.last_mut() {
            Some(buffer) if buffer.bytes() >= bytes => {
                buffer.zero_span(0, bytes).map_err(say("the slabs"))?;
            }
            _ => {
                held.clear();
                held.push(Buffer::zeroed(bytes).map_err(say("the slabs"))?);
            }
        }
        let buffer = held.last().expect("laid just above");
        // SAFETY: `buffer` is zeroed device memory of at least the bytes
        // these slabs take, kept in `held` past every launch that takes them.
        unsafe { Slabs::place(buffer.ptr() as *mut c_void, &placed) };
        Ok(())
    }

    fn zeroed(bytes: usize) -> Result<Buffer, String> {
        Buffer::zeroed(bytes).map_err(say("an operand"))
    }

    /// `engine::rates::operand_pattern` on the device, uploaded once and
    /// copied device to device to fill each operand.
    struct Random {
        pattern: Buffer,
        stream: *mut c_void,
    }

    impl Random {
        fn new(stream: *mut c_void) -> Result<Random, String> {
            let bytes = engine::rates::operand_pattern();
            let mut pattern = zeroed(bytes.len())?;
            pattern
                .write(0, &bytes)
                .map_err(say("the operand pattern"))?;
            Ok(Random { pattern, stream })
        }

        /// `bytes` of device memory, the pattern repeated over them, ordered
        /// on the stream before any launch that reads it.
        fn operand(&self, bytes: usize) -> Result<Buffer, String> {
            let b = zeroed(bytes)?;
            let mut at = 0;
            while at < bytes {
                let n = self.pattern.bytes().min(bytes - at);
                crate::device::copy_d2d(self.stream, b.ptr() + at as u64, self.pattern.ptr(), n)
                    .map_err(say("an operand"))?;
                at += n;
            }
            Ok(b)
        }
    }

    fn column(values: &[i32]) -> Result<Buffer, String> {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut b = zeroed(bytes.len())?;
        b.write(0, &bytes).map_err(say("an operand"))?;
        Ok(b)
    }

    pub(super) fn measure(ordinal: i32, chip: &profile::Profile) -> Result<Measured, String> {
        let mut device = Context::bind(ordinal, None).map_err(say("the device"))?;
        device.set_tuning(kernels_cuda::tuning::Tuning::of(chip));
        let stream = device.stream();
        let clock = Clock::new(stream)?;
        let mut held = Vec::new();
        let random = Random::new(stream)?;
        // The launch first: every rate below takes its dispatches' launches
        // off (engine::rates::rate_from).
        let launch = launch(device.ctx(), stream, &clock, &mut held)?;
        let gap = launch as f64;
        let warmed = warm(device.ctx(), &clock, &random, &mut held)?;
        eprintln!(
            "[rates]: warm-up: {} rotations of the {}^3 matmul, settled at {:.1} TFLOP/s",
            warmed.rotations,
            engine::rates::WARM_SHAPE.0,
            warmed.rate / 1e3
        );
        let mut gemm = Vec::new();
        for w in engine::rates::GEMM_WIDTHS {
            let mut planes = Planes::lay(&random, (w, w))?;
            let mut curve = Vec::new();
            for m in engine::rates::GEMM_ROWS {
                let rate = planes.at(device.ctx(), &clock, &random, &mut held, m, gap)?;
                curve.push((m, rate));
            }
            gemm.push((w, curve));
        }
        let mut attn = Vec::new();
        for keys in engine::rates::ATTN_KEYS {
            let rate = attention(&device, &clock, &random, &mut held, keys, 128, gap)?;
            attn.push((keys, rate));
        }
        let longest = engine::rates::ATTN_KEYS[engine::rates::ATTN_KEYS.len() - 1];
        let mut attn_by_dim = Vec::new();
        for d in engine::rates::ATTN_DIMS {
            let rate = match attn.last() {
                Some(&(_, rate)) if d == 128 => rate,
                _ => attention(&device, &clock, &random, &mut held, longest, d, gap)?,
            };
            attn_by_dim.push((d, rate / 1e3));
        }
        let elementwise = elementwise(device.ctx(), &clock, &random, &mut held, gap)?;
        let scan = scan(device.ctx(), &clock, &random, &mut held, SCAN_SHAPE, gap)?;
        let bytes = copy(stream, &clock, gap)?;
        #[cfg(feature = "rocm")]
        let cached = Some(cached(device.ctx(), &clock, gap)?);
        #[cfg(not(feature = "rocm"))]
        let cached: Option<f64> = None;
        let (few_row, resident) = few_rows(device.ctx(), &clock, &random, &mut held, gap)?;
        eprintln!(
            "[rates]: the few-rows kernel at {FEW_ROWS} x {w} x {w}: {:.1} TFLOP/s, {resident} \
             blocks a multiprocessor",
            few_row / 1e3,
            w = engine::rates::GEMM_WIDTH
        );
        device.synchronize().map_err(say("the device"))?;
        eprintln!(
            "[rates]: the step held at most {} MiB of device buffers ({} MiB of slabs at its end)",
            PEAK.load(Ordering::Relaxed) >> 20,
            held.iter().map(|b| b.bytes()).sum::<usize>() >> 20
        );
        // A tenth of a TFLOP/s, a GB/s, a G element a second.
        let tenth = |v: f64| (v * 10.0).round() / 10.0;
        let tflops = |curve: Vec<(u32, f64)>| {
            curve
                .into_iter()
                .map(|(x, r)| (x, tenth(r / 1e3)))
                .collect()
        };
        Ok(Measured {
            rates: Rates {
                landing: None,
                launch_ns: Some(launch),
                gemm_tflops_by_rows: Some(
                    gemm.into_iter()
                        .map(|(w, curve)| (w, tflops(curve)))
                        .collect(),
                ),
                attn_tflops_by_keys: Some(tflops(attn)),
                hbm_gbps: Some(tenth(bytes)),
                cached_gbps: cached.map(tenth),
                elementwise_gbps: Some(tenth(elementwise)),
                scan_gelems: Some(tenth(scan)),
                few_row_tflops: Some(tenth(few_row / 1e3)),
                workgroups_per_core: Some(f64::from(resident)),
                submit_worst_ratio: None,
                resident_workgroups: None,
                attn_tflops_by_rows_keys: None,
                product_ns: None,
                kernel_tflops_by_shape: None,
                dense_settled: Vec::new(),
                measured_on: None,
                measured_by: None,
            },
            attn_by_dim,
        })
    }

    /// The per-row ops' bandwidth: the rmsnorm, then the swiglu, at
    /// `engine::rates::ELEMENTWISE_SHAPE`, each fire the next of
    /// `engine::rates::elementwise_sets` sets, their bytes over their time.
    fn elementwise(
        ctx: &Ctx,
        clock: &Clock,
        random: &Random,
        held: &mut Vec<Buffer>,
        gap: f64,
    ) -> Result<f64, String> {
        let (rows, width) = engine::rates::ELEMENTWISE_SHAPE;
        let sets = engine::rates::elementwise_sets();
        let row = rows as usize * width as usize * 2;
        let x = random.operand(sets * row)?;
        let normed = zeroed(sets * row)?;
        let packed = random.operand(sets * 2 * row)?;
        let y = zeroed(sets * row)?;
        let weight = random.operand(width as usize * 2)?;
        let fire = |i: usize| {
            let at = |b: &Buffer, bytes: usize| b.ptr() + ((i % sets) * bytes) as u64;
            kernels_cuda::elemwise::norm::rmsnorm(
                ctx,
                Tensor::new(at(&x, row), rows, width, Dtype::Bf16),
                Tensor::new(weight.ptr(), 1, width, Dtype::Bf16),
                1e-6,
                &mut Tensor::new(at(&normed, row), rows, width, Dtype::Bf16),
            )?;
            kernels_cuda::linear::glu::swiglu(
                ctx,
                Tensor::new(at(&packed, 2 * row), rows, 2 * width, Dtype::Bf16),
                width,
                1,
                &mut Tensor::new(at(&y, row), rows, width, Dtype::Bf16),
            )
        };
        lay(held, || fire(0))?;
        let ns = clock.burst(ROUNDS, &Touched::default(), sets, &mut |i| {
            fire(i).map_err(say("the per-row ops"))
        })?;
        engine::rates::rate_from(
            "per-row ops",
            ns,
            2,
            gap,
            engine::rates::elementwise_bytes(),
        )
    }

    /// The device warmed (engine::rates::warm_up): bursts of the widest
    /// matmul over its planes until its rate stops falling.
    fn warm(
        ctx: &Ctx,
        clock: &Clock,
        random: &Random,
        held: &mut Vec<Buffer>,
    ) -> Result<engine::rates::Warmed, String> {
        let (m, n, k) = engine::rates::WARM_SHAPE;
        let act = random.operand(m as usize * k as usize * 2)?;
        let y = zeroed(m as usize * n as usize * 2)?;
        let planes = weights_at(n, k);
        let weights = (0..planes)
            .map(|_| random.operand(n as usize * k as usize * 2))
            .collect::<Result<Vec<Buffer>, String>>()?;
        let fire = |w: usize| {
            kernels_cuda::linear::gemm::matmul(
                ctx,
                Tensor::new(act.ptr(), m, k, Dtype::Bf16),
                Tensor::new(weights[w % planes].ptr(), n, k, Dtype::Bf16),
                &mut Tensor::new(y.ptr(), m, n, Dtype::Bf16),
            )
        };
        lay(held, || fire(0))?;
        engine::rates::warm_up(|| {
            let ns = clock
                .time(&mut || (0..planes).try_for_each(|w| fire(w).map_err(say("the matmul"))))?;
            Ok(engine::rates::gemm_flops(m, n, k) * planes as f64 / ns.max(1.0))
        })
    }

    /// One `(n, k)` shape's weight rotation laid (`engine::rates::weights_at`
    /// planes of `WEIGHT_BYTES`), its activations and product laid as wide as
    /// the widest row asked of it.
    struct Planes {
        n: u32,
        k: u32,
        weights: Vec<Buffer>,
        /// The planes touched once, by the first matmul over them.
        touched: Touched,
        act: Option<Buffer>,
        y: Option<Buffer>,
        rows: u32,
    }

    impl Planes {
        fn lay(random: &Random, (n, k): (u32, u32)) -> Result<Planes, String> {
            let weights = (0..weights_at(n, k))
                .map(|_| random.operand(n as usize * k as usize * 2))
                .collect::<Result<Vec<Buffer>, String>>()?;
            Ok(Planes {
                n,
                k,
                weights,
                touched: Touched::default(),
                act: None,
                y: None,
                rows: 0,
            })
        }

        /// The matmul `m x n x k`'s rate, its launches rotating through the
        /// planes.
        fn at(
            &mut self,
            ctx: &Ctx,
            clock: &Clock,
            random: &Random,
            held: &mut Vec<Buffer>,
            m: u32,
            gap: f64,
        ) -> Result<f64, String> {
            let (n, k) = (self.n, self.k);
            if m > self.rows {
                self.act = None;
                self.y = None;
                self.act = Some(random.operand(m as usize * k as usize * 2)?);
                self.y = Some(zeroed(m as usize * n as usize * 2)?);
                self.rows = m;
            }
            let (act, y) = (
                self.act.as_ref().expect("laid"),
                self.y.as_ref().expect("laid"),
            );
            let planes = self.weights.len();
            let weights = &self.weights;
            let fire = |w: usize| {
                kernels_cuda::linear::gemm::matmul(
                    ctx,
                    Tensor::new(act.ptr(), m, k, Dtype::Bf16),
                    Tensor::new(weights[w % planes].ptr(), n, k, Dtype::Bf16),
                    &mut Tensor::new(y.ptr(), m, n, Dtype::Bf16),
                )
            };
            lay(held, || fire(0))?;
            let ns = clock.burst(ROUNDS, &self.touched, planes, &mut |i| {
                fire(i).map_err(say("the matmul"))
            })?;
            engine::rates::rate_from(
                &format!("matmul {m}x{n}x{k}"),
                ns,
                engine::rates::GEMM_DISPATCHES,
                gap,
                engine::rates::gemm_flops(m, n, k),
            )
        }
    }

    /// The attention's rate at `keys_n` keys and width `head_dim`.
    fn attention(
        device: &Context,
        clock: &Clock,
        random: &Random,
        held: &mut Vec<Buffer>,
        keys_n: u32,
        head_dim: u32,
        gap: f64,
    ) -> Result<f64, String> {
        let ctx = device.ctx();
        let at = ATTN_SHAPE.at(keys_n).with_dim(head_dim);
        // One sequence attending its own keys, written before the attention.
        let (rows, d) = (at.rows, at.head_dim);
        let (q_heads, kv_heads) = (at.q_heads, at.kv_heads);
        let pages = rows.div_ceil(PAGE);
        let kv_width = kv_heads * d;
        // Sets of q rows and kv pages past every last-level cache
        // (`engine::rates::ATTN_SET_BYTES`, as `attn_sets_at` counts them but
        // of the pages as laid, whole), each fire the next set's, so each
        // reads its q and kv from device memory as a served layer's do; the
        // sets' pages share one pool and their page tables one column.
        let q_bytes = (rows * q_heads * d) as usize * 2;
        let set_cells = (pages * PAGE) as usize;
        let sets = engine::rates::attn_sets_at(at, PAGE);
        let cells = sets * set_cells;
        let keys = random.operand(cells * kv_width as usize * 2)?;
        let values = random.operand(cells * kv_width as usize * 2)?;
        let q = random.operand(sets * q_bytes)?;
        let o = zeroed(q_bytes)?;
        // Each set's pages scattered over its span of the pool as a served
        // sequence's are.
        let order = engine::rates::attn_page_order(pages);
        let table: Vec<i32> = (0..sets as u32)
            .flat_map(|set| order.iter().map(move |&p| (set * pages + p) as i32))
            .collect();
        let indices = column(&table)?;
        let kv_indptr = [0, pages as i32];
        let indptr = column(&kv_indptr)?;
        let last = column(&[(rows - (pages - 1) * PAGE) as i32])?;
        let qo = [0, rows as i32];
        let qo_at = column(&qo)?;
        let int = zeroed(16 << 20)?;
        let float = zeroed(256 << 20)?;
        let i32s = |b: &Buffer, n: u32| Tensor::new(b.ptr(), n, 1, Dtype::I32);
        let plane = |b: &Buffer| Tensor::new(b.ptr(), cells as u32, kv_width, Dtype::Bf16);
        let absent = Tensor::new(0, 0, 0, Dtype::Bf16);
        let pool = |set: usize| {
            KvPool {
                keys: plane(&keys),
                values: plane(&values),
                bf16_keys: absent,
                bf16_values: absent,
                key_scales: Tensor::ABSENT,
                value_scales: Tensor::ABSENT,
                page_indices: Tensor::new(
                    indices.ptr() + (set * pages as usize * 4) as u64,
                    pages,
                    1,
                    Dtype::I32,
                ),
                page_indptr: i32s(&indptr, 2),
                last_page_lens: i32s(&last, 1),
                row_valid: Tensor::ABSENT,
                env_min: absent,
                env_max: absent,
                has_envelopes: false,
                page_size: PAGE as i32,
                seq_stride: i64::from(kv_width),
                head_stride: i64::from(d),
                layout: 0,
                scheme_byte: 0,
                block_size: 0,
                max_pages_per_request: pages as i32,
                pages_in_batch: pages as i32,
                live: [0, 0],
                page_stride: [0, 0],
            }
            .with_packed_stride()
        };
        let shape = Shape {
            num_requests: 1,
            lane_offset: 0,
            num_q_heads: q_heads,
            num_kv_heads: kv_heads,
            head_dim: d,
            page_size: PAGE,
            hnd_layout: false,
        };
        let live = Live {
            requests: 1,
            lane_offset: 0,
            row_offset: 0,
            rows,
        };
        let workspace = Workspace {
            int_ptr: int.ptr(),
            int_bytes: int.bytes(),
            float_ptr: float.ptr(),
            float_bytes: float.bytes(),
        };
        let plan = kernels_cuda::attn::plan::plan_prefill(
            &qo,
            &kv_indptr,
            &[rows as i32],
            rows,
            shape,
            live,
            None,
            true,
            false,
            None,
            &device.device(),
            device.toggles(),
            workspace,
        )
        .map_err(say("the attention's plan"))?;
        plan.stage(ctx).map_err(say("the attention's plan"))?;
        let pools: Vec<KvPool> = (0..sets).map(pool).collect();
        let fire = |i: usize| {
            let set = i % sets;
            kernels_cuda::attn::prefill(
                ctx,
                RaggedTensor {
                    data: Tensor::new(
                        q.ptr() + (set * q_bytes) as u64,
                        rows,
                        q_heads * d,
                        Dtype::Bf16,
                    ),
                    indptr: i32s(&qo_at, 2),
                },
                &plan,
                &pools[set],
                None,
                d,
                kv_heads,
                1.0 / (d as f32).sqrt(),
                &mut Tensor::new(o.ptr(), rows, q_heads * d, Dtype::Bf16),
            )
        };
        lay(held, || fire(0))?;
        let ns = clock.burst(ROUNDS, &Touched::default(), sets, &mut |i| {
            fire(i).map_err(say("the attention"))
        })?;
        engine::rates::rate_from(
            &format!("attention {}x{} d {}", at.rows, at.keys, at.head_dim),
            ns,
            engine::rates::ATTN_DISPATCHES,
            gap,
            engine::rates::attn_flops(at),
        )
    }

    fn scan(
        ctx: &Ctx,
        clock: &Clock,
        random: &Random,
        held: &mut Vec<Buffer>,
        shape: engine::rates::ScanShape,
        gap: f64,
    ) -> Result<f64, String> {
        let (tokens, conv_width) = (shape.tokens, shape.conv_width);
        let channels = shape.channels();
        let conv_stride = conv_width * channels;
        let state_stride = shape.v_heads * shape.k_dim * shape.v_dim;
        // `scan_sets_at` sets of input rows and conv and scan state, each
        // fire the next set's (its rows at their offset, its state the slot
        // of its index), so each reads them from device memory as a served
        // layer's do.
        let sets = engine::rates::scan_sets_at(shape);
        let x_bytes = (tokens * channels) as usize * 2;
        let x = random.operand(sets * x_bytes)?;
        let conv_y = zeroed(x_bytes)?;
        let weight = random.operand((channels * conv_width) as usize * 2)?;
        let conv_state = random.operand(sets * conv_stride as usize * 2)?;
        let state = random.operand(sets * state_stride as usize * 2)?;
        let gates = zeroed((tokens * 2 * shape.v_heads) as usize * 4)?;
        let y = zeroed((tokens * shape.v_heads * shape.v_dim) as usize * 4)?;
        let slots = (0..sets)
            .map(|set| column(&[set as i32]))
            .collect::<Result<Vec<Buffer>, String>>()?;
        let indptr = column(&[0, tokens as i32])?;
        let ragged = |at: u64| RaggedTensor {
            data: Tensor::new(at, tokens, channels, Dtype::Bf16),
            indptr: Tensor::new(indptr.ptr(), 2, 1, Dtype::I32),
        };
        let pool = |set: usize| RecurrentPool {
            slab: Tensor::new(state.ptr(), sets as u32, state_stride, Dtype::Bf16),
            slot_ids: Tensor::new(slots[set].ptr(), 1, 1, Dtype::I32),
            slot_stride_elems: i64::from(state_stride),
            conv_slab: Tensor::new(conv_state.ptr(), sets as u32, conv_stride, Dtype::Bf16),
            conv_stride: i64::from(conv_stride),
            write_state: true,
            write_state_mask: Tensor::ABSENT,
            commit_len: Tensor::ABSENT,
            begin_at: Tensor::ABSENT,
            fused_decay: false,
            segments: kernels_cuda::Segments::ONE,
            snapshots: kernels_cuda::Snapshots::NONE,
        };
        // The scan lands bf16 where the device's tuning narrows it, as a
        // served load's trace does (`narrow::gdn_out_bf16`).
        #[cfg(not(feature = "rocm"))]
        let out = if ctx.tuning().ssm_gdn_out_bf16() {
            Dtype::Bf16
        } else {
            Dtype::F32
        };
        #[cfg(feature = "rocm")]
        let out = Dtype::F32;
        let pools: Vec<RecurrentPool> = (0..sets).map(pool).collect();
        let fire = |i: usize| {
            let set = i % sets;
            kernels_cuda::ssm::causal_conv1d_chunked(
                ctx,
                ragged(x.ptr() + (set * x_bytes) as u64),
                Tensor::new(weight.ptr(), channels, conv_width, Dtype::Bf16),
                None,
                &pools[set],
                conv_width,
                1,
                &mut Tensor::new(conv_y.ptr(), tokens, channels, Dtype::Bf16),
            )?;
            kernels_cuda::ssm::gated_delta_chunked(
                ctx,
                ragged(conv_y.ptr()),
                Tensor::ABSENT,
                Tensor::new(gates.ptr(), tokens, 2 * shape.v_heads, Dtype::F32),
                &pools[set],
                shape.k_heads,
                shape.v_heads,
                shape.k_dim,
                shape.v_dim,
                &mut Tensor::new(y.ptr(), tokens, shape.v_heads * shape.v_dim, out),
            )
        };
        lay(held, || fire(0))?;
        let ns = clock.burst(ROUNDS, &Touched::default(), sets, &mut |i| {
            fire(i).map_err(say("the scan"))
        })?;
        engine::rates::rate_from(
            &format!("scan of {} tokens", shape.tokens),
            ns,
            engine::rates::SCAN_DISPATCHES,
            gap,
            engine::rates::scan_elements(shape),
        )
    }

    /// The few-rows kernel's rate (flops a ns) and the blocks of it a
    /// multiprocessor holds at once: the plain product of [`FEW_ROWS`] over
    /// one `GEMM_WIDTH`-square plane, the same plane every fire (its reads
    /// from the last-level cache after the first, so its compute binds);
    /// the occupancy is the device's calculator's for that kernel.
    fn few_rows(
        ctx: &Ctx,
        clock: &Clock,
        random: &Random,
        held: &mut Vec<Buffer>,
        gap: f64,
    ) -> Result<(f64, u32), String> {
        let (m, w) = (FEW_ROWS, engine::rates::GEMM_WIDTH);
        let act = random.operand(m as usize * w as usize * 2)?;
        let weight = random.operand(w as usize * w as usize * 2)?;
        let y = zeroed(m as usize * w as usize * 2)?;
        let fire = || {
            kernels_cuda::linear::gemm::matmul(
                ctx,
                Tensor::new(act.ptr(), m, w, Dtype::Bf16),
                Tensor::new(weight.ptr(), w, w, Dtype::Bf16),
                &mut Tensor::new(y.ptr(), m, w, Dtype::Bf16),
            )
        };
        lay(held, fire)?;
        // Its one plane is its own set, touched by its first burst.
        let ns = clock.burst(ROUNDS, &Touched::default(), 1, &mut |_| {
            fire().map_err(say("the few-rows matmul"))
        })?;
        let rate = engine::rates::rate_from(
            &format!("few-rows matmul {m}x{w}x{w}"),
            ns,
            engine::rates::GEMM_DISPATCHES,
            gap,
            engine::rates::gemm_flops(m, w, w),
        )?;
        let resident = kernels_cuda::linear::gemm::few_rows_resident_blocks(ctx, m, w, w)
            .map_err(say("the few-rows kernel's occupancy"))?;
        Ok((rate, resident))
    }

    #[cfg(feature = "rocm")]
    fn cached(ctx: &Ctx, clock: &Clock, gap: f64) -> Result<f64, String> {
        let sets = zeroed(CACHED_SET_BYTES * CACHED_SETS)?;
        let sink = zeroed(CACHED_BLOCKS as usize * 4)?;
        let packs = (CACHED_SET_BYTES / 16) as u32;
        let ns = clock.burst(ROUNDS, &Touched::default(), 1, &mut |_| {
            kernels_cuda::linear::cached_read_probe(
                ctx,
                sets.ptr(),
                packs,
                CACHED_SETS as u32,
                CACHED_BLOCKS,
                sink.ptr(),
            )
            .map_err(say("the cached-read probe"))
        })?;
        engine::rates::rate_from(
            "cached reads",
            ns,
            engine::rates::COPY_DISPATCHES,
            gap,
            CACHED_BLOCKS as f64 * CACHED_SET_BYTES as f64,
        )
    }

    fn copy(stream: *mut c_void, clock: &Clock, gap: f64) -> Result<f64, String> {
        let from = zeroed(COPY_BYTES)?;
        let into = zeroed(COPY_BYTES)?;
        let ns = clock.burst(ROUNDS, &Touched::default(), 1, &mut |_| {
            crate::device::copy_d2d(stream, into.ptr(), from.ptr(), COPY_BYTES)
                .map_err(say("the copy"))
        })?;
        // Every byte copied is read and written.
        engine::rates::rate_from(
            "copy",
            ns,
            engine::rates::COPY_DISPATCHES,
            gap,
            2.0 * COPY_BYTES as f64,
        )
    }

    fn launch(
        ctx: &Ctx,
        stream: *mut c_void,
        clock: &Clock,
        held: &mut Vec<Buffer>,
    ) -> Result<u64, String> {
        // A chain of `n` dependent residual adds captured in one graph
        // (each adds into the buffer the one before wrote, the two
        // alternating), against one add over the chain's whole length, at
        // each length of engine::rates::LAUNCH_CHAINS.
        let most = *engine::rates::LAUNCH_CHAINS.last().expect("chains") as usize;
        let (x, y) = (zeroed(256)?, zeroed(256)?);
        let (wide_x, wide_y) = (zeroed(16 * most)?, zeroed(16 * most)?);
        let add = |from: &Buffer, into: &Buffer, width: u32| {
            kernels_cuda::elemwise::pointwise::residual_add(
                ctx,
                Tensor::new(from.ptr(), 1, width, Dtype::Bf16),
                &mut Tensor::new(into.ptr(), 1, width, Dtype::Bf16),
            )
        };
        lay(held, || add(&x, &y, 8))?;
        add(&x, &y, 8).map_err(say("the launch"))?;
        let timed = |work: &dyn Fn() -> Result<(), crate::error::Fault>| {
            let graph = Graph::capture(stream, work).map_err(say("the launch's graph"))?;
            let exec = graph
                .instantiate(stream)
                .map_err(say("the launch's graph"))?;
            clock.burst(LAUNCH_ROUNDS, &Touched::default(), 1, &mut |_| {
                exec.launch(stream).map_err(say("the launch's graph"))
            })
        };
        let unsupported = |e: kernels_cuda::Error| crate::error::Fault::Unsupported(e.to_string());
        let mut gaps = Vec::new();
        for n in engine::rates::LAUNCH_CHAINS {
            let chain = timed(&|| {
                for i in 0..n {
                    let (from, into) = if i % 2 == 0 { (&x, &y) } else { (&y, &x) };
                    add(from, into, 8).map_err(unsupported)?;
                }
                Ok(())
            })?;
            let one = timed(&|| add(&wide_x, &wide_y, 8 * n).map_err(unsupported))?;
            gaps.push((n, engine::rates::launch_from(chain, one, n)));
        }
        eprintln!(
            "[rates]: launch: {}",
            gaps.iter()
                .map(|(n, ns)| format!("{ns:.0} ns a dispatch over a chain of {n}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        Ok(gaps.last().map_or(0.0, |&(_, ns)| ns).round() as u64)
    }
}

#[cfg(test)]
mod attention_tests {
    use super::PAGE;

    #[test]
    fn the_attention_sets_and_keys_are_the_shared_rule_at_cudas_page() {
        // CUDA path unchanged: 63876a684 sized its sets in place, the q rows
        // and the kv pages as laid whole against ATTN_SET_BYTES, at least 2;
        // the shared function at its PAGE gives the same.
        for (keys, d) in [(1, 64), (17, 128), (2234, 128), (4467, 256)] {
            let at = engine::rates::ATTN_SHAPE.at(keys).with_dim(d);
            let pages = at.rows.div_ceil(PAGE);
            let q_bytes = (at.rows * at.q_heads * at.head_dim) as usize * 2;
            let set_cells = (pages * PAGE) as usize;
            let kv_width = (at.kv_heads * at.head_dim) as usize;
            let inline = engine::rates::ATTN_SET_BYTES
                .div_ceil(q_bytes + 2 * set_cells * kv_width * 2)
                .max(2);
            assert_eq!(
                engine::rates::attn_sets_at(at, PAGE),
                inline,
                "{keys} keys, d {d}"
            );
        }
    }
}

#[cfg(test)]
mod ensure_tests {
    use super::*;

    fn measured() -> Rates {
        Rates {
            landing: None,
            load_width: None,
            launch_ns: Some(900),
            gemm_tflops_by_rows: Some(vec![(4096, vec![(256, 100.0), (8192, 200.0)])]),
            attn_tflops_by_keys: Some(vec![(512, 50.0), (8192, 300.0)]),
            hbm_gbps: Some(600.0),
            // ROCm's rates step measures it; CUDA's estimate weighs none.
            cached_gbps: cfg!(feature = "rocm").then_some(28800.0),
            elementwise_gbps: Some(500.0),
            scan_gelems: Some(2000.0),
            few_row_tflops: Some(20.0),
            workgroups_per_core: Some(4.0),
            submit_worst_ratio: None,
            resident_workgroups: None,
            attn_tflops_by_rows_keys: None,
            product_ns: None,
            kernel_tflops_by_shape: None,
            dense_settled: Vec::new(),
            measured_on: None,
            measured_by: None,
        }
    }

    #[test]
    fn a_load_measures_once_and_the_next_reads_what_it_kept() {
        let home = std::env::temp_dir().join(format!("pie-ensure-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        let device = "Roofline Test Card";
        let facts = crate::chip::facts(8, 9, 142);
        let keys = crate::stored::Keys::of(0, device);
        let room = engine::rates::step_bytes();
        let (chip, _) = crate::chip::resolve_keyed(device, &facts, Some(&home), &keys);
        assert!(
            !chip.rates().complete(&KEYS),
            "no file states this card's rates"
        );
        let mut times = 0;
        let first = ensure_by(
            chip,
            device,
            &keys,
            Some(&home),
            room,
            engine::rates::Measuring::Time,
            |_| {
                times += 1;
                Ok(measured())
            },
        )
        .unwrap();
        // What it measured is kept keyed: the device and what measured each.
        assert_eq!((times, first.rates()), (1, keys.stamp(measured())));
        assert!(first.tuned().measured_this_load);
        // The second load resolves the user's profile the first wrote, and
        // measures nothing.
        let (chip, _) = crate::chip::resolve_keyed(device, &facts, Some(&home), &keys);
        let second = ensure_by(
            chip,
            device,
            &keys,
            Some(&home),
            room,
            engine::rates::Measuring::Time,
            |_| {
                times += 1;
                Ok(measured())
            },
        )
        .unwrap();
        std::fs::remove_dir_all(&home).unwrap();
        assert_eq!((times, second.rates()), (1, keys.stamp(measured())));
        assert!(!second.tuned().measured_this_load);
    }

    #[test]
    fn a_budget_short_of_the_step_is_refused_by_name_and_measures_nothing() {
        let device = "Roofline Test Card";
        let facts = crate::chip::facts(8, 9, 142);
        let keys = crate::stored::Keys::of(0, device);
        let (chip, _) = crate::chip::resolve_keyed(device, &facts, None, &keys);
        let short = engine::rates::step_bytes() - 1;
        let err = ensure_by(
            chip,
            device,
            &keys,
            None,
            short,
            engine::rates::Measuring::Time,
            |_| panic!("nothing is measured past a short budget"),
        )
        .unwrap_err();
        assert!(err.contains("measuring them needs"), "{err}");
    }
}

#[cfg(all(test, feature = "cuda"))]
mod kept_routes_tests {
    use kernels_cuda::linear::routes::{Kind, Routes, Shape, code};

    /// Two loads across a version bump: the first measures every candidate
    /// of its decode products and keeps the times in the device's profile;
    /// the second, from another pie build, reads that profile and measures
    /// nothing. A changed kernel is measured again.
    #[test]
    fn a_profile_kept_across_a_version_bump_measures_nothing_again() {
        let dir = std::env::temp_dir().join(format!("pie-kept-routes-{}", std::process::id()));
        let platform = "sm_89|driver 13000|cublaslt 130000";
        let candidates: Vec<u32> = ["skinny jit/a/r1", "cublaslt 7.0.1.1", "cublaslt 7.0.2.1"]
            .iter()
            .map(|fingerprint| code(platform, fingerprint))
            .collect();
        let shapes = [
            Shape { kind: Kind::Plain, rows: 1536, k: 1536, m: 64 },
            Shape { kind: Kind::Plain, rows: 12288, k: 1536, m: 32 },
        ];

        // Load one: an empty profile notes every shape wanted.
        let first = Routes::default();
        for shape in shapes {
            first.pick(shape, 0x1000, &candidates);
        }
        let wanted = first.wanted();
        assert_eq!(wanted.len(), shapes.len());
        let measured: Vec<_> = wanted
            .iter()
            .flat_map(|(shape, noted)| {
                noted.candidates.iter().map(|&kernel| (*shape, kernel, 1000.0, 10.0))
            })
            .collect();
        let keys = crate::stored::Keys::of(0, "test-device");
        let saved = super::save_product_times(Some(&dir), "test-device", &keys, &[], &measured)
            .expect("the times save")
            .expect("a profile directory was named");

        // Load two, another pie build: the profile as saved, read back.
        let text = std::fs::read_to_string(&saved).expect("the saved profile reads");
        let bumped = text.replace(&engine::rates::build_commit(), "another-build");
        let file = crate::chip::registry()
            .parse(&bumped, "the kept routes", profile::Strict::User)
            .expect("the kept routes parse");
        let second = Routes::of(file.tuned.rates.product_ns.as_deref().unwrap_or_default());
        for shape in shapes {
            second.pick(shape, 0x1000, &candidates);
        }
        assert!(second.wanted().is_empty(), "a kept profile measures nothing across a version bump");

        // A kernel whose code changed is another candidate: wanted again.
        let changed = [code(platform, "skinny jit/a/r2"), candidates[1], candidates[2]];
        second.pick(shapes[0], 0x1000, &changed);
        assert_eq!(second.wanted().len(), 1);
        let _ = std::fs::remove_dir_all(dir);
    }
}
