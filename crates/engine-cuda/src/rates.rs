//! The device's rates measured on the device, through the kernels serving
//! dispatches (`pie tune cuda --rates`, and on ROCm the same steps over
//! HIP's): what the top rung's estimate is weighed by.
//!
//! Each rate is a burst of its kernel launched back to back on the stream,
//! about [`BURST_NS`] (20 ms) long (`k` launches, `k` sized off one warm launch), as
//! serving runs a fire's: a lone launch lets the device idle between
//! timings, and its clock state, not the kernel, sets the number. The burst
//! is timed by events on the stream, the median of [`ROUNDS`] bursts ([`LAUNCH_ROUNDS`] for the launch) after a
//! warm one kept, and divided by `k`.
//!
//! Every shape is `engine::rates`' lattice, and its counts turn the times
//! into rates, so the CUDA and ROCm rates mean what every backend's do:
//! - `bf16_flops_per_ns_by_rows_n_k`: the matmul `m x n x k` at each row
//!   count of the load's candidate rungs ([`lattice_rows`]) and each column count and depth of [`DEPTHS`]
//!   through `linear::gemm` (cuBLAS(Lt) on CUDA,
//!   hipBLASLt/rocBLAS on ROCm) under the GEMM tactics serving runs; its
//!   launches rotate through `engine::rates::weights_at(n, k)` weight planes, `engine::rates::WEIGHT_BYTES`
//!   of them, several times what any last-level cache holds, so each reads
//!   its weight from device memory as a layer's does.
//! - `attn_flops_per_ns_by_keys_and_dim`: the paged prefill attention at
//!   [`ATTN_SHAPE`]'s heads over each of [`ATTN_KEYS`] (a sequence of as many
//!   rows attending itself) and each head width of `ATTN_DIMS`, causal, in
//!   pages of [`PAGE`]; its launches rotate through
//!   `engine::rates::attn_sets_at` sets of q rows and kv pages, so each reads
//!   them from device memory as a served layer's does.
//! - `scan_elements_per_ns_by_tokens`: at each of `engine::rates::SCAN_TOKENS`,
//!   the causal conv and the chunked gated delta
//!   scan at [`SCAN_SHAPE`].
//! - `bytes_per_ns`: a [`COPY_BYTES`] device-to-device copy, read plus write.
//! - `launch_ns`: a chain of dependent near-empty kernels captured in one
//!   graph against one kernel of the chain's whole work, at each length of
//!   `engine::rates::LAUNCH_CHAINS` (`engine::rates::launch_from`): what a
//!   dispatch adds to its work inside a recorded body, as serving replays
//!   them.
//!
//! Every float operand the matmul, attention and scan read (activations,
//! weight planes, the kv pages, the scan's input, conv weight and states) is
//! `engine::rates::operand_pattern` repeated, not zeros: some devices run a
//! compute-bound kernel faster on zeros than on the real data serving runs.
//! Index and page tables, gates, outputs and workspace stay as laid.

#[cfg_attr(not(feature = "device"), allow(unused_imports))]
use engine::rates::{ATTN_DIMS, ATTN_KEYS, ATTN_SHAPE, DEPTHS, SCAN_SHAPE, weights_at};
use profile::fire::Rates;

/// The rows the matmul grid is timed at under `chip`: the candidates a load
/// weighs its top rung among (`serve::load`'s `budget_at(asked).buckets`),
/// the chip's lattice ([`crate::api::lattice`] at its `engine.lattice_steps`,
/// `prefill_lattice_from` and `_steps`) up to the tokens a load is asked
/// when nothing states them: the chip's `engine.forward_tokens`, at least
/// [`crate::api::DEFAULT_FORWARD_TOKENS`]. A load asked other tokens (a
/// model the walk fit narrows, a stated `max_forward_tokens`) lays a top
/// this grid does not state, refused by name.
#[must_use]
pub fn lattice_rows(chip: &profile::Profile) -> Vec<u32> {
    let engine = crate::chip::Engine::of(chip);
    let top = match engine.forward_tokens {
        0 => crate::api::DEFAULT_FORWARD_TOKENS,
        tokens => tokens.max(crate::api::DEFAULT_FORWARD_TOKENS),
    };
    crate::api::lattice(
        Vec::new(),
        top,
        engine.lattice_steps,
        engine.prefill_lattice_from,
        engine.prefill_lattice_steps,
    )
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

/// The rates of device `ordinal` under `chip`'s tuning, its matmuls under
/// the GEMM tactics serving picks from `cache` (the engine's cache root,
/// `$PIE_HOME/cache`, whose `gemm-algos` table a served load reads): a
/// shape it holds runs its pick, and one it does not is timed once and kept
/// there, so every later measurement and every load runs that same pick.
pub fn measure(
    ordinal: i32,
    chip: &profile::Profile,
    cache: Option<&std::path::Path>,
) -> Result<Rates, String> {
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
    /// walk of it, as a load lays its fires'), in memory `held` keeps.
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
        let buffer = Buffer::zeroed(Slabs::placed_bytes(&placed)).map_err(say("the slabs"))?;
        // SAFETY: `buffer` is zeroed device memory of the bytes these slabs
        // take, kept in `held` past every launch that takes them.
        unsafe { Slabs::place(buffer.ptr() as *mut c_void, &placed) };
        held.push(buffer);
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

    pub(super) fn measure(ordinal: i32, chip: &profile::Profile) -> Result<Rates, String> {
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
        let rows = lattice_rows(chip);
        let grid = gemm(device.ctx(), &clock, &random, &mut held, &rows, gap)?;
        let mut attn = Vec::with_capacity(ATTN_KEYS.len() * ATTN_DIMS.len());
        for d in ATTN_DIMS {
            for keys in ATTN_KEYS {
                attn.push((
                    keys,
                    d,
                    attention(&device, &clock, &random, &mut held, keys, d, gap)?,
                ));
            }
        }
        let mut scan_curve = Vec::with_capacity(engine::rates::SCAN_TOKENS.len());
        for tokens in engine::rates::SCAN_TOKENS {
            scan_curve.push((
                tokens,
                scan(
                    device.ctx(),
                    &clock,
                    &random,
                    &mut held,
                    SCAN_SHAPE.at(tokens),
                    gap,
                )?,
            ));
        }
        let bytes = copy(stream, &clock, gap)?;
        device.synchronize().map_err(say("the device"))?;
        eprintln!(
            "[rates]: the step held at most {} MiB of device buffers ({} MiB of slabs at its end)",
            PEAK.load(Ordering::Relaxed) >> 20,
            held.iter().map(|b| b.bytes()).sum::<usize>() >> 20
        );
        Ok(Rates {
            bf16_flops_per_ns_by_rows_n_k: Some(grid),
            attn_flops_per_ns_by_keys_and_dim: Some(attn),
            scan_elements_per_ns_by_tokens: Some(scan_curve),
            bytes_per_ns: Some(bytes),
            launch_ns: Some(launch),
            ..Rates::default()
        })
    }

    fn gemm(
        ctx: &Ctx,
        clock: &Clock,
        random: &Random,
        held: &mut Vec<Buffer>,
        rows: &[u32],
        gap: f64,
    ) -> Result<Vec<(u32, u32, u32, f64)>, String> {
        // The shapes in engine::rates' spread order, so a drift left after
        // the warm-up spreads over the grid.
        let shapes: Vec<(u32, u32)> = DEPTHS
            .iter()
            .flat_map(|&k| DEPTHS.iter().map(move |&n| (n, k)))
            .collect();
        let mut grid = Vec::with_capacity(rows.len() * shapes.len());
        for at in engine::rates::spread_order(shapes.len()) {
            let (n, k) = shapes[at];
            grid.extend(gemm_at(ctx, clock, random, held, rows, n, k, gap)?);
        }
        grid.sort_by_key(|&(m, n, k, _)| (k, n, m));
        Ok(grid)
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

    /// The matmul `m x n x k` at every row count of the lattice.
    fn gemm_at(
        ctx: &Ctx,
        clock: &Clock,
        random: &Random,
        held: &mut Vec<Buffer>,
        rows: &[u32],
        n: u32,
        k: u32,
        gap: f64,
    ) -> Result<Vec<(u32, u32, u32, f64)>, String> {
        let most = rows.iter().copied().max().expect("rows") as usize;
        let act = random.operand(most * k as usize * 2)?;
        let y = zeroed(most * n as usize * 2)?;
        let planes = weights_at(n, k);
        let weights = (0..planes)
            .map(|_| random.operand(n as usize * k as usize * 2))
            .collect::<Result<Vec<Buffer>, String>>()?;
        let mut curve = Vec::with_capacity(rows.len());
        for row in engine::rates::spread_order(rows.len()) {
            let m = rows[row];
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
            curve.push((
                m,
                n,
                k,
                engine::rates::rate_from(
                    &format!("matmul {m}x{n}x{k}"),
                    ns,
                    engine::rates::GEMM_DISPATCHES,
                    gap,
                    engine::rates::gemm_flops(m, n, k),
                )?,
            ));
        }
        Ok(curve)
    }

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
        // `attn_sets_at` sets of q rows and kv pages, each fire the next
        // set's, so each reads its q and kv from device memory as a served
        // layer's do; the sets' pages share one pool.
        let sets = engine::rates::attn_sets_at(at);
        let cells = sets * (pages * PAGE) as usize;
        let q_bytes = (rows * q_heads * d) as usize * 2;
        let keys = random.operand(cells * kv_width as usize * 2)?;
        let values = random.operand(cells * kv_width as usize * 2)?;
        let q = random.operand(sets * q_bytes)?;
        let o = zeroed(q_bytes)?;
        // Each set's pages scattered over its span of the pool as a served
        // sequence's are.
        let order = engine::rates::attn_page_order(pages);
        let indices = (0..sets)
            .map(|set| {
                let first = set as u32 * pages;
                column(
                    &order
                        .iter()
                        .map(|&p| (first + p) as i32)
                        .collect::<Vec<i32>>(),
                )
            })
            .collect::<Result<Vec<Buffer>, String>>()?;
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
                page_indices: i32s(&indices[set], pages),
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
mod lattice_tests {
    #[test]
    fn the_grid_is_timed_at_the_candidates_a_load_weighs() {
        // The rows a load with nothing stated weighs (serve::load's
        // budget_at(asked).buckets): the chip's lattice to its asked tokens.
        let chip = crate::chip::profile("");
        let engine = crate::chip::Engine::of(&chip);
        let asked = match engine.forward_tokens {
            0 => crate::api::DEFAULT_FORWARD_TOKENS,
            tokens => tokens.max(crate::api::DEFAULT_FORWARD_TOKENS),
        };
        let weighed = crate::api::lattice(
            Vec::new(),
            asked,
            engine.lattice_steps,
            engine.prefill_lattice_from,
            engine.prefill_lattice_steps,
        );
        assert_eq!(super::lattice_rows(&chip), weighed);
        assert_eq!(weighed.first(), Some(&1), "a readout's one row");
    }
}
