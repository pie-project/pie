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
//! - `bf16_flops_per_ns_by_rows_and_depth`: the matmul `m x k x k` at each
//!   row count of [`LATTICE_ROWS`] and depth of [`DEPTHS`] through `linear::gemm` (cuBLAS(Lt) on CUDA,
//!   hipBLASLt/rocBLAS on ROCm) under the GEMM tactics serving runs; its
//!   launches rotate through `engine::rates::weights_at` weight planes, `engine::rates::WEIGHT_BYTES`
//!   of them, several times what any last-level cache holds, so each reads
//!   its weight from device memory as a layer's does.
//! - `attn_flops_per_ns_by_keys`: the paged prefill attention at
//!   [`ATTN_SHAPE`]'s heads over each of [`ATTN_KEYS`] (a sequence of as many
//!   rows attending itself), causal, in pages of [`PAGE`].
//! - `scan_elements_per_ns`: the causal conv and the chunked gated delta
//!   scan at [`SCAN_SHAPE`].
//! - `bytes_per_ns`: a [`COPY_BYTES`] device-to-device copy, read plus write.
//! - `launch_ns`: [`LAUNCHES`] near-empty kernels captured in one graph, the
//!   graph's time over [`LAUNCHES`]: a dispatch's cost inside a recorded
//!   body, as serving replays them.
//!
//! Every float operand the matmul, attention and scan read (activations,
//! weight planes, the kv pages, the scan's input, conv weight and states) is
//! `engine::rates::operand_pattern` repeated, not zeros: some devices run a
//! compute-bound kernel faster on zeros than on the real data serving runs.
//! Index and page tables, gates, outputs and workspace stay as laid.

#[cfg_attr(not(feature = "device"), allow(unused_imports))]
use engine::rates::{ATTN_KEYS, ATTN_SHAPE, DEPTHS, LATTICE_ROWS, SCAN_SHAPE, weights_at};
use profile::fire::Rates;

/// The kv cache's page, as a served load pages it.
pub const PAGE: u32 = 16;

/// The copy's bytes: past every last-level cache, so it moves device memory.
pub const COPY_BYTES: usize = 256 << 20;

/// The kernels the launch's graph captures.
pub const LAUNCHES: u32 = 1000;

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

    use kernels_cuda::attn::plan::{Live, Shape, Workspace};
    use kernels_cuda::{Ctx, KvPool, RaggedTensor, RecurrentPool, Slabs, Tensor};
    use poem_ir::Dtype;

    use super::*;
    use crate::device::{Buffer, Context, Graph, rt};

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
        let grid = gemm(device.ctx(), &clock, &random, &mut held)?;
        let mut attn = Vec::with_capacity(ATTN_KEYS.len());
        for keys in ATTN_KEYS {
            attn.push((keys, attention(&device, &clock, &random, &mut held, keys)?));
        }
        let scan = scan(device.ctx(), &clock, &random, &mut held)?;
        let bytes = copy(stream, &clock)?;
        let launch = launch(device.ctx(), stream, &clock, &mut held)?;
        device.synchronize().map_err(say("the device"))?;
        Ok(Rates {
            bf16_flops_per_ns_by_rows_and_depth: Some(grid),
            attn_flops_per_ns_by_keys: Some(attn),
            scan_elements_per_ns: Some(scan),
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
    ) -> Result<Vec<(u32, u32, f64)>, String> {
        let mut grid = Vec::with_capacity(LATTICE_ROWS.len() * DEPTHS.len());
        for &k in &DEPTHS {
            grid.extend(gemm_at(ctx, clock, random, held, k)?);
        }
        Ok(grid)
    }

    /// The matmul `m x k x k` at every row count of the lattice.
    fn gemm_at(
        ctx: &Ctx,
        clock: &Clock,
        random: &Random,
        held: &mut Vec<Buffer>,
        k: u32,
    ) -> Result<Vec<(u32, u32, f64)>, String> {
        let n = k;
        let most = *LATTICE_ROWS.last().expect("rows") as usize;
        let act = random.operand(most * k as usize * 2)?;
        let y = zeroed(most * n as usize * 2)?;
        let planes = weights_at(k);
        let weights = (0..planes)
            .map(|_| random.operand(n as usize * k as usize * 2))
            .collect::<Result<Vec<Buffer>, String>>()?;
        let mut curve = Vec::with_capacity(LATTICE_ROWS.len());
        for &m in &LATTICE_ROWS {
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
            curve.push((m, k, engine::rates::gemm_flops(m, k) / ns));
        }
        Ok(curve)
    }

    fn attention(
        device: &Context,
        clock: &Clock,
        random: &Random,
        held: &mut Vec<Buffer>,
        keys_n: u32,
    ) -> Result<f64, String> {
        let ctx = device.ctx();
        let at = ATTN_SHAPE.at(keys_n);
        // One sequence attending its own keys, written before the attention.
        let (rows, d) = (at.rows, at.head_dim);
        let (q_heads, kv_heads) = (at.q_heads, at.kv_heads);
        let pages = rows.div_ceil(PAGE);
        let kv_width = kv_heads * d;
        let cells = (pages * PAGE) as usize;
        let keys = random.operand(cells * kv_width as usize * 2)?;
        let values = random.operand(cells * kv_width as usize * 2)?;
        let q = random.operand((rows * q_heads * d) as usize * 2)?;
        let o = zeroed((rows * q_heads * d) as usize * 2)?;
        // The sequence's pages scattered over the pool as a served one's are.
        let page_ids: Vec<i32> = engine::rates::attn_page_order(pages)
            .into_iter()
            .map(|p| p as i32)
            .collect();
        let indices = column(&page_ids)?;
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
        let pool = KvPool {
            keys: plane(&keys),
            values: plane(&values),
            bf16_keys: absent,
            bf16_values: absent,
            key_scales: Tensor::ABSENT,
            value_scales: Tensor::ABSENT,
            page_indices: i32s(&indices, pages),
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
        .with_packed_stride();
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
        let fire = || {
            kernels_cuda::attn::prefill(
                ctx,
                RaggedTensor {
                    data: Tensor::new(q.ptr(), rows, q_heads * d, Dtype::Bf16),
                    indptr: i32s(&qo_at, 2),
                },
                &plan,
                &pool,
                None,
                d,
                kv_heads,
                1.0 / (d as f32).sqrt(),
                &mut Tensor::new(o.ptr(), rows, q_heads * d, Dtype::Bf16),
            )
        };
        lay(held, fire)?;
        let ns = clock.burst(ROUNDS, &mut |_| fire().map_err(say("the attention")))?;
        Ok(engine::rates::attn_flops(at) / ns)
    }

    fn scan(
        ctx: &Ctx,
        clock: &Clock,
        random: &Random,
        held: &mut Vec<Buffer>,
    ) -> Result<f64, String> {
        let shape = SCAN_SHAPE;
        let (tokens, conv_width) = (shape.tokens, shape.conv_width);
        let channels = shape.channels();
        let conv_stride = conv_width * channels;
        let state_stride = shape.v_heads * shape.k_dim * shape.v_dim;
        let x = random.operand((tokens * channels) as usize * 2)?;
        let conv_y = zeroed((tokens * channels) as usize * 2)?;
        let weight = random.operand((channels * conv_width) as usize * 2)?;
        let conv_state = random.operand(conv_stride as usize * 2)?;
        let state = random.operand(state_stride as usize * 2)?;
        let gates = zeroed((tokens * 2 * shape.v_heads) as usize * 4)?;
        let y = zeroed((tokens * shape.v_heads * shape.v_dim) as usize * 4)?;
        let slot = column(&[0])?;
        let indptr = column(&[0, tokens as i32])?;
        let ragged = |data: &Buffer| RaggedTensor {
            data: Tensor::new(data.ptr(), tokens, channels, Dtype::Bf16),
            indptr: Tensor::new(indptr.ptr(), 2, 1, Dtype::I32),
        };
        let pool = RecurrentPool {
            slab: Tensor::new(state.ptr(), 1, state_stride, Dtype::Bf16),
            slot_ids: Tensor::new(slot.ptr(), 1, 1, Dtype::I32),
            slot_stride_elems: i64::from(state_stride),
            conv_slab: Tensor::new(conv_state.ptr(), 1, conv_stride, Dtype::Bf16),
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
        let fire = || {
            kernels_cuda::ssm::causal_conv1d_chunked(
                ctx,
                ragged(&x),
                Tensor::new(weight.ptr(), channels, conv_width, Dtype::Bf16),
                None,
                &pool,
                conv_width,
                1,
                &mut Tensor::new(conv_y.ptr(), tokens, channels, Dtype::Bf16),
            )?;
            kernels_cuda::ssm::gated_delta_chunked(
                ctx,
                ragged(&conv_y),
                Tensor::ABSENT,
                Tensor::new(gates.ptr(), tokens, 2 * shape.v_heads, Dtype::F32),
                &pool,
                shape.k_heads,
                shape.v_heads,
                shape.k_dim,
                shape.v_dim,
                &mut Tensor::new(y.ptr(), tokens, shape.v_heads * shape.v_dim, out),
            )
        };
        lay(held, fire)?;
        let ns = clock.burst(ROUNDS, &mut |_| fire().map_err(say("the scan")))?;
        Ok(engine::rates::scan_elements(shape) / ns)
    }

    fn copy(stream: *mut c_void, clock: &Clock) -> Result<f64, String> {
        let from = zeroed(COPY_BYTES)?;
        let into = zeroed(COPY_BYTES)?;
        let ns = clock.burst(ROUNDS, &mut |_| {
            crate::device::copy_d2d(stream, into.ptr(), from.ptr(), COPY_BYTES)
                .map_err(say("the copy"))
        })?;
        // Every byte copied is read and written.
        Ok(2.0 * COPY_BYTES as f64 / ns)
    }

    fn launch(
        ctx: &Ctx,
        stream: *mut c_void,
        clock: &Clock,
        held: &mut Vec<Buffer>,
    ) -> Result<u64, String> {
        let x = zeroed(256)?;
        let y = zeroed(256)?;
        let fire = || {
            kernels_cuda::elemwise::pointwise::residual_add(
                ctx,
                Tensor::new(x.ptr(), 1, 8, Dtype::Bf16),
                &mut Tensor::new(y.ptr(), 1, 8, Dtype::Bf16),
            )
        };
        lay(held, fire)?;
        fire().map_err(say("the launch"))?;
        let graph = Graph::capture(stream, || {
            for _ in 0..LAUNCHES {
                fire().map_err(|e| crate::error::Fault::Unsupported(e.to_string()))?;
            }
            Ok(())
        })
        .map_err(say("the launch's graph"))?;
        let exec = graph
            .instantiate(stream)
            .map_err(say("the launch's graph"))?;
        let ns = clock.burst(LAUNCH_ROUNDS, &mut |_| {
            exec.launch(stream).map_err(say("the launch's graph"))
        })?;
        Ok((ns / f64::from(LAUNCHES)).round() as u64)
    }
}
