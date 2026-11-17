//! The device's rates measured on the device, through the kernels serving
//! dispatches (`pie tune cuda --rates`, and on ROCm the same steps over
//! HIP's): the roofline constants the top rung's estimate is weighed by.
//!
//! Each rate is a burst of its kernel launched back to back on the stream,
//! about [`BURST_NS`] (20 ms) long (`k` launches, `k` sized off one warm
//! launch), as serving runs a fire's: a lone launch lets the device idle
//! between timings, and its clock state, not the kernel, sets the number.
//! The burst is timed by events on the stream, the median of [`ROUNDS`]
//! bursts ([`LAUNCH_ROUNDS`] for the launch) after a warm one kept, and
//! divided by `k`.
//!
//! Every shape is `engine::rates`', and its counts turn the times into
//! rates, each work as the device runs it (`engine::roofline::Machine`:
//! its waves over the SMs), so the constants read back through the
//! roofline give the times measured:
//! - `launch_ns`: a chain of dependent near-empty kernels captured in one
//!   graph against one kernel of the chain's whole work, at each length of
//!   `engine::rates::LAUNCH_CHAINS` (`engine::rates::launch_from`): what a
//!   dispatch adds to its work inside a recorded body.
//! - `gemm_tflops`: the matmul at each of `engine::rates::GEMM_SHAPES`
//!   through `linear::gemm` (cuBLASLt's first heuristic algorithm on CUDA,
//!   hipBLASLt's on ROCm), after the warm-up; its launches rotate through
//!   `engine::rates::weights_at(n, k)` weight planes, past every last-level
//!   cache. The fastest.
//! - `attn_tflops`: the paged prefill attention at [`ATTN_SHAPE`]'s heads
//!   over `engine::rates::ATTN_KEYS` keys, causal, in pages of [`PAGE`], at
//!   head width 128 (64 and 256 timed and said beside it); its launches
//!   rotate through `engine::rates::attn_sets_at` sets of q rows and kv
//!   pages.
//! - `hbm_gbps`: a [`COPY_BYTES`] device-to-device copy, read plus write.
//! - `scan_gelems`: the causal conv and the chunked gated delta scan at
//!   [`SCAN_SHAPE`].
//!
//! Every float operand the matmul, attention and scan read is
//! `engine::rates::operand_pattern` repeated, not zeros: some devices run a
//! compute-bound kernel faster on zeros than on the real data serving runs.
//! Index and page tables, gates, outputs and workspace stay as laid.

#[cfg_attr(not(feature = "device"), allow(unused_imports))]
use engine::rates::{ATTN_SHAPE, SCAN_SHAPE, weights_at};
use profile::fire::Rates;

/// The `[provenance]` entry a measurement of `device`'s rates writes beside
/// them (`engine::rates::provenance`): how this backend measured them.
#[must_use]
pub fn provenance(device: &str) -> (&'static str, String) {
    let how = format!(
        "{device}; the matmul through {}'s first heuristic algorithm, each rate a 20 ms burst on \
         the stream timed by events, the median of 5 (the launch of 15), its work its waves over \
         the device's SMs; operands: engine::rates' pattern; at {}",
        if cfg!(feature = "rocm") {
            "hipBLASLt"
        } else {
            "cuBLASLt"
        },
        engine::rates::build_commit()
    );
    engine::rates::provenance("cuda", &how)
}

/// `chip` with every roofline constant stated: as it is where its layers
/// state them all; else the device's own measured here, before the load is
/// ready (the step `pie tune cuda --rates` runs, seconds long, each timed
/// submission a burst of about [`BURST_NS`]), each constant a layer states
/// kept over the measured one, the whole written to the user's profile of
/// `device` under `profiles` (with its provenance) so later loads read it,
/// and marked measured this load for the load's line.
pub fn ensure(
    chip: profile::Profile,
    ordinal: i32,
    device: &str,
    facts: &profile::Facts,
    profiles: Option<&std::path::Path>,
    cache: Option<&std::path::Path>,
) -> Result<profile::Profile, String> {
    let stated = chip.rates();
    if stated.launch_ns.is_some()
        && stated.gemm_tflops.is_some()
        && stated.attn_tflops.is_some()
        && stated.hbm_gbps.is_some()
        && stated.scan_gelems.is_some()
    {
        return Ok(chip);
    }
    let began = std::time::Instant::now();
    let measured = measure(ordinal, &chip, cache)?.rates;
    let rates = stated.over(measured);
    eprintln!(
        "engine-cuda: {device}'s profile states no roofline rates; measured them in {:.1} s",
        began.elapsed().as_secs_f64()
    );
    if let Some(profiles) = profiles {
        let file = profile::search::assemble(
            crate::chip::registry(),
            &chip,
            device,
            facts,
            &engine::rates::tuned(&rates),
            std::slice::from_ref(&provenance(device)),
            &[crate::chip::ARCH],
        )?;
        let path = crate::chip::SHELF.user_path(profiles, device);
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        std::fs::write(&path, file).map_err(|e| format!("{}: {e}", path.display()))?;
        eprintln!("engine-cuda: the rates are written to {}", path.display());
    }
    let mut tuned = chip.tuned().clone();
    tuned.rates = rates;
    tuned.measured_this_load = true;
    Ok(chip.with_tuned(tuned))
}

/// The kv cache's page, as a served load pages it.
pub const PAGE: u32 = 16;

/// The copy's bytes: past every last-level cache, so it moves device memory.
pub const COPY_BYTES: usize = 256 << 20;

/// A burst's length.
pub const BURST_NS: f64 = 20e6;

/// Timed bursts of each kind; the median is kept.
pub const ROUNDS: usize = 5;

/// Timed bursts of the launch's graph: a dispatch's cost moves more burst to
/// burst than a kernel's work (an MI300X VF's 1610 to 1719 ns over 5).
pub const LAUNCH_ROUNDS: usize = 15;

/// What a rates step measured: the constants, and the attention's rate at
/// each head width of `engine::rates::ATTN_DIMS` (TFLOP/s), said beside them.
pub struct Measured {
    pub rates: Rates,
    pub attn_by_dim: Vec<(u32, f64)>,
    pub gemm_by_shape: Vec<((u32, u32, u32), f64)>,
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
        on_device::measure(ordinal, chip)
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
            let b = crate::device::Buffer::zeroed(bytes)?;
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
        /// its burst: a burst of about [`BURST_NS`] sized off a warm launch,
        /// the median of `rounds` after a warm burst, over its launches.
        fn burst(
            &self,
            rounds: usize,
            once: &mut dyn FnMut(usize) -> Result<(), String>,
        ) -> Result<f64, String> {
            self.time(&mut || once(0))?;
            let one = self.time(&mut || once(0))?.max(1.0);
            let k = ((BURST_NS / one).ceil() as usize).clamp(1, 4096);
            let mut run = || (0..k).try_for_each(&mut *once);
            self.time(&mut run)?;
            let mut times = (0..rounds)
                .map(|_| self.time(&mut run))
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
        let machine = crate::top::machine(chip, device.device().num_sm);
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
        let mut gemm_by_shape = Vec::new();
        for (m, n, k) in engine::rates::GEMM_SHAPES {
            let mut planes = Planes::lay(&random, (n, k))?;
            let rate = planes.at(device.ctx(), &clock, &random, &mut held, &machine, m, gap)?;
            gemm_by_shape.push(((m, n, k), rate));
        }
        let gemm = gemm_by_shape
            .iter()
            .map(|&(_, rate)| rate)
            .fold(0.0, f64::max);
        let mut attn_by_dim = Vec::new();
        for d in engine::rates::ATTN_DIMS {
            let rate = attention(
                &device,
                &clock,
                &random,
                &mut held,
                &machine,
                engine::rates::ATTN_KEYS,
                d,
                gap,
            )?;
            attn_by_dim.push((d, rate));
        }
        let attn = attn_by_dim
            .iter()
            .find(|&&(d, _)| d == 128)
            .map_or(0.0, |&(_, rate)| rate);
        let scan = scan(device.ctx(), &clock, &random, &mut held, SCAN_SHAPE, gap)?;
        let bytes = copy(stream, &clock, gap)?;
        device.synchronize().map_err(say("the device"))?;
        eprintln!(
            "[rates]: the step held at most {} MiB of device buffers ({} MiB of slabs at its end)",
            PEAK.load(Ordering::Relaxed) >> 20,
            held.iter().map(|b| b.bytes()).sum::<usize>() >> 20
        );
        // Three significant decimals of the TFLOP/s, a tenth of the rest.
        let tenth = |v: f64| (v * 10.0).round() / 10.0;
        Ok(Measured {
            rates: Rates {
                launch_ns: Some(launch),
                gemm_tflops: Some(tenth(gemm / 1e3)),
                attn_tflops: Some(tenth(attn / 1e3)),
                hbm_gbps: Some(tenth(bytes)),
                scan_gelems: Some(tenth(scan)),
            },
            attn_by_dim: attn_by_dim.iter().map(|&(d, r)| (d, r / 1e3)).collect(),
            gemm_by_shape: gemm_by_shape.iter().map(|&(s, r)| (s, r / 1e3)).collect(),
        })
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
                act: None,
                y: None,
                rows: 0,
            })
        }

        /// The matmul `m x n x k`'s rate, its launches rotating through the
        /// planes: its waves' flops on `machine` over its time.
        fn at(
            &mut self,
            ctx: &Ctx,
            clock: &Clock,
            random: &Random,
            held: &mut Vec<Buffer>,
            machine: &engine::roofline::Machine,
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
            let ns = clock.burst(ROUNDS, &mut |i| fire(i).map_err(say("the matmul")))?;
            engine::rates::rate_from(
                &format!("matmul {m}x{n}x{k}"),
                ns,
                engine::rates::GEMM_DISPATCHES,
                gap,
                machine.gemm_flops(m, n, k),
            )
        }
    }

    /// The attention's rate at `keys_n` keys and width `head_dim`: its
    /// flops, times its work-groups' wave factor on `machine`, over its time.
    fn attention(
        device: &Context,
        clock: &Clock,
        random: &Random,
        held: &mut Vec<Buffer>,
        machine: &engine::roofline::Machine,
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
        let ns = clock.burst(ROUNDS, &mut |i| fire(i).map_err(say("the attention")))?;
        engine::rates::rate_from(
            &format!("attention {}x{} d {}", at.rows, at.keys, at.head_dim),
            ns,
            engine::rates::ATTN_DISPATCHES,
            gap,
            engine::rates::attn_flops(at) * machine.attn_wave_factor(1, at.rows, at.q_heads),
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
        let out = if ctx.tuning().ssm_gdn_out_bf16 {
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
        let ns = clock.burst(ROUNDS, &mut |i| fire(i).map_err(say("the scan")))?;
        engine::rates::rate_from(
            &format!("scan of {} tokens", shape.tokens),
            ns,
            engine::rates::SCAN_DISPATCHES,
            gap,
            engine::rates::scan_elements(shape),
        )
    }

    fn copy(stream: *mut c_void, clock: &Clock, gap: f64) -> Result<f64, String> {
        let from = zeroed(COPY_BYTES)?;
        let into = zeroed(COPY_BYTES)?;
        let ns = clock.burst(ROUNDS, &mut |_| {
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
            clock.burst(LAUNCH_ROUNDS, &mut |_| {
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
