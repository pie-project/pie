//! `pie tune metal --rates`: the device's rates as pie's Metal kernels reach
//! them, at the shapes every backend times ([`engine::rates`]), each through
//! the kernel serving dispatches under the device's own tuning:
//!
//! * the matmul curve: `linear::gemm::matmul` at each of
//!   [`engine::rates::LATTICE_ROWS`], each repeat over the next of
//!   engine::rates::weights_at's distinct weights so every one streams from memory, as a fire's do;
//! * attention: `attn::arbiter::prefill` at [`engine::rates::ATTN_SHAPE`],
//!   causal, over a bf16 pool of the 16-token pages serving builds, laid
//!   in engine::rates::attn_page_order's scattered order;
//! * the scan: `ssm::causal_conv1d_chunked` then `ssm::gated_delta_chunked`
//!   at [`engine::rates::SCAN_SHAPE`];
//! * a 256 MiB blit between two buffers (read and write counted), and a
//!   dispatch's launch (the smallest matmul back to back in one frame).
//!
//! Each kernel is timed by the GPU (`Frame::commit_timed`) back to back
//! with itself, as many to a frame as fill 20 ms, the median of five such
//! frames after a warm one, over the count: serving runs hundreds of
//! dispatches a fire, and a frame of one short dispatch times the clock
//! state the device idled into, not the kernel. The counts that turn the
//! times into rates are [`engine::rates`]'.
//!
//! Every float operand the matmul, attention and scan read (activations,
//! weight planes, q and the kv pages, the scan's input, conv weight and
//! states) is `engine::rates::operand_pattern` repeated, not zeros: some
//! devices run a compute-bound kernel faster on zeros than on the real data
//! serving runs, and every backend times the same bytes. Index and page
//! tables, gates, outputs and the launch probe stay as laid.

use engine::rates::{
    ATTN_DIMS, ATTN_KEYS, ATTN_SHAPE, AttnShape, DEPTHS, LATTICE_ROWS, SCAN_SHAPE,
};
use kernels_metal::linear::gemm;
use kernels_metal::tensor::{RaggedTensor, RecurrentPool};
use kernels_metal::{KvPool, PrefillPlan, StateMap, Tensor, Tuning, attn, ssm};
use poem_ir::Dtype;
use profile::fire::Rates;

use crate::device::{Buffer, Context, Handles, Pipelines};
use crate::encode::Sink;

/// How long a timed frame runs: long enough that the device reaches and
/// holds its serving clock through it. A narrow kernel's frame of 5 ms read
/// the m = 16 matmul 26% apart run to run on an M1 Max (45% on an M5 Pro);
/// 20 ms reads it within 0.25%.
const FRAME_SECONDS: f64 = 0.020;

/// The most of one measurement a timed frame repeats.
const MAX_REPEATS: u32 = 512;

fn median(mut times: Vec<f64>) -> f64 {
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

/// Deterministic bf16 values in `[-scale, scale)`.
/// `bytes` of `engine::rates::operand_pattern`, repeated.
fn operand(pattern: &[u8], bytes: usize) -> Vec<u8> {
    pattern.iter().copied().cycle().take(bytes).collect()
}

fn words<T: Copy>(v: &[T], to: impl Fn(T) -> [u8; 4]) -> Vec<u8> {
    v.iter().flat_map(|&x| to(x)).collect()
}

struct Bench<'a> {
    device: &'a Context,
    handles: Handles,
    pipelines: Pipelines,
    tuning: &'a Tuning,
    /// `engine::rates::operand_pattern`, made once.
    pattern: Vec<u8>,
}

impl Bench<'_> {
    /// A buffer of `bytes` of the operand pattern.
    fn operand(&self, bytes: u64) -> Result<Buffer, String> {
        self.buffer(&operand(&self.pattern, usize::try_from(bytes).map_err(|e| e.to_string())?))
    }

    fn buffer(&self, bytes: &[u8]) -> Result<Buffer, String> {
        let mut b =
            Buffer::zeroed(self.device, bytes.len().max(4) as u64).map_err(|e| e.to_string())?;
        b.write(0, bytes).map_err(|e| e.to_string())?;
        Ok(b)
    }

    fn zeroed(&self, bytes: u64) -> Result<Buffer, String> {
        Buffer::zeroed(self.device, bytes.max(4)).map_err(|e| e.to_string())
    }

    fn tensor(&self, b: &Buffer, rows: u32, width: u32, dtype: Dtype) -> Result<Tensor, String> {
        let handle = self
            .handles
            .bind(b, 0, b.bytes())
            .map_err(|e| e.to_string())?;
        Ok(Tensor::new(handle, rows, width, dtype))
    }

    /// GPU seconds of one frame holding `k` of what `encode` fills.
    fn frame_of(
        &self,
        encode: &dyn Fn(&Sink<'_>, u32) -> Result<(), kernels_metal::Error>,
        k: u32,
    ) -> Result<f64, String> {
        let frame = self.device.frame().map_err(|e| e.to_string())?;
        let sink = Sink::new(self.device, &frame, &self.pipelines, &self.handles);
        for at in 0..k {
            encode(&sink, at).map_err(|e| e.to_string())?;
        }
        frame.commit_timed().map_err(|e| e.to_string())
    }

    /// The GPU seconds one of what `encode` fills takes as serving runs it,
    /// back to back with others: each frame holds `k` of them, `k` enough
    /// for a frame of [`FRAME_SECONDS`], and the median of five such frames
    /// after a warm one, over `k`. A frame of one short dispatch times the
    /// clock state the device idled into between frames, not the kernel
    /// (bimodal on an M5 Pro at m = 64..2048 of the matmul ladder).
    fn timed(
        &self,
        encode: &dyn Fn(&Sink<'_>, u32) -> Result<(), kernels_metal::Error>,
    ) -> Result<f64, String> {
        let once = self.frame_of(encode, 1)?.max(1e-7);
        let k = ((FRAME_SECONDS / once).ceil() as u32).clamp(1, MAX_REPEATS);
        self.frame_of(encode, k)?;
        let mut times = Vec::with_capacity(5);
        for _ in 0..5 {
            times.push(self.frame_of(encode, k)? / f64::from(k));
        }
        Ok(median(times))
    }

    /// An `m × k × k` matmul's rate, each repeat over the next of `weights`
    /// (`k × k` each).
    fn matmul_flops(&self, m: u32, n: u32, k: u32, weights: &[Tensor]) -> Result<f64, String> {
        let (a, y) = (
            self.operand(u64::from(m) * u64::from(k) * 2)?,
            self.zeroed(u64::from(m) * u64::from(n) * 2)?,
        );
        let (at, yt) = (
            self.tensor(&a, m, k, Dtype::Bf16)?,
            self.tensor(&y, m, n, Dtype::Bf16)?,
        );
        // Each repeat reads the next weight: serving streams a layer's from
        // memory, and one weight repeated would sit in the system cache,
        // by a share that changes run to run (2x at m = 16 on an M5 Pro).
        let seconds = self.timed(&|sink, at_k| {
            gemm::matmul(
                sink,
                self.tuning,
                at,
                weights[at_k as usize % weights.len()],
                yt,
            )
        })?;
        Ok(engine::rates::gemm_flops(m, n, k) / (seconds * 1e9))
    }

    fn attention_flops(&self, shape: AttnShape) -> Result<f64, String> {
        let (rows, keys, q_heads, kv_heads, d) = (
            shape.rows,
            shape.keys,
            shape.q_heads,
            shape.kv_heads,
            shape.head_dim,
        );
        const PAGE: u32 = 16;
        let pages = keys.div_ceil(PAGE);
        let slots = pages * PAGE;
        let kv = (slots * kv_heads * d) as usize;
        let keys = self.operand(kv as u64 * 2)?;
        let values = self.operand(kv as u64 * 2)?;
        let q = self.operand(u64::from(rows * q_heads * d) * 2)?;
        let out = self.zeroed(u64::from(rows * q_heads * d) * 2)?;
        let positions: Vec<u32> = (0..rows).collect();
        // The pool pages scattered as a served pool lays a sequence's, the
        // same order every backend times (engine::rates::attn_page_order).
        let page_ids: Vec<u32> = engine::rates::attn_page_order(pages);
        let le = u32::to_le_bytes;
        let page_indices = self.buffer(&words(&page_ids, le))?;
        let page_indptr = self.buffer(&words(&[0, pages], le))?;
        let positions = self.buffer(&words(&positions, le))?;
        let owner = self.buffer(&words(&vec![0u32; rows as usize], le))?;
        let q_indptr = self.buffer(&words(&[0, rows], le))?;
        let mask = self.buffer(&[1u8])?;
        let masked = self.buffer(&vec![0u8; rows as usize])?;
        let pool = KvPool {
            keys: self.tensor(&keys, slots, kv_heads * d, Dtype::Bf16)?,
            values: self.tensor(&values, slots, kv_heads * d, Dtype::Bf16)?,
            page_indices: self.tensor(&page_indices, pages, 1, Dtype::U32)?,
            page_indptr: self.tensor(&page_indptr, 2, 1, Dtype::U32)?,
            page_size: PAGE as i32,
            seq_stride: u64::from(kv_heads * d),
            head_stride: u64::from(d),
            q8: false,
            page_stride: [0; 2],
        }
        .paged_densely();
        let plan = PrefillPlan {
            positions: self.tensor(&positions, rows, 1, Dtype::I32)?,
            last_page_len: self.tensor(&positions, rows, 1, Dtype::I32)?,
            request_of_token: self.tensor(&owner, rows, 1, Dtype::I32)?,
            mask: self.tensor(&mask, 1, 1, Dtype::U8)?,
            mask_enabled: self.tensor(&masked, rows, 1, Dtype::U8)?,
            mask_stride: 0,
        };
        let q = RaggedTensor {
            data: self.tensor(&q, rows, q_heads * d, Dtype::Bf16)?,
            indptr: self.tensor(&q_indptr, 2, 1, Dtype::I32)?,
        };
        let o = self.tensor(&out, rows, q_heads * d, Dtype::Bf16)?;
        let scale = 1.0 / (d as f32).sqrt();
        let seconds = self.timed(&|sink, _| {
            attn::arbiter::prefill(
                sink,
                q,
                &plan,
                &pool,
                None,
                d,
                kv_heads,
                scale,
                o,
                1,
                self.tuning,
            )
        })?;
        Ok(engine::rates::attn_flops(shape) / (seconds * 1e9))
    }

    fn scan_elements(&self) -> Result<f64, String> {
        const TOKENS: u32 = SCAN_SHAPE.tokens;
        const K_HEADS: u32 = SCAN_SHAPE.k_heads;
        const HEADS: u32 = SCAN_SHAPE.v_heads;
        const DK: u32 = SCAN_SHAPE.k_dim;
        const DV: u32 = SCAN_SHAPE.v_dim;
        const CONV: u32 = SCAN_SHAPE.conv_width;
        let channels = SCAN_SHAPE.channels();
        let bank = u64::from(HEADS * DV * DK);
        let qkv_in = self.operand(u64::from(TOKENS * channels) * 2)?;
        let qkv = self.zeroed(u64::from(TOKENS * channels) * 2)?;
        let weight = self.operand(u64::from(channels * CONV) * 2)?;
        let mut gates = Vec::with_capacity((TOKENS * 2 * HEADS) as usize);
        for t in 0..TOKENS {
            gates.extend((0..HEADS).map(|h| -0.05 - 0.001 * ((t + h) % 7) as f32));
            gates.extend((0..HEADS).map(|h| 0.5 + 0.01 * ((t * 3 + h) % 11) as f32));
        }
        let gates = self.buffer(&words(&gates, f32::to_le_bytes))?;
        let state = self.operand(bank * 4)?;
        let conv_state = self.operand(u64::from(CONV * channels) * 2)?;
        let new_conv_state = self.zeroed(u64::from(CONV * channels) * 2)?;
        let y = self.zeroed(u64::from(TOKENS * HEADS * DV) * 4)?;
        let slot = self.buffer(&0u32.to_le_bytes())?;
        let indptr = self.buffer(&words(&[0i32, TOKENS as i32], i32::to_le_bytes))?;
        let map = self.buffer(&StateMap::whole(bank).bytes())?;
        let segments = self.tensor(&map, 1, (StateMap::HEAD + 1) as u32, Dtype::U64)?;
        let pool = RecurrentPool {
            state: self.tensor(&state, HEADS * DV, DK, Dtype::F32)?,
            slots: self.tensor(&slot, 1, 1, Dtype::U32)?,
            conv_state: self.tensor(&conv_state, 1, CONV * channels, Dtype::Bf16)?,
            new_conv_state: self.tensor(&new_conv_state, 1, CONV * channels, Dtype::Bf16)?,
            segments,
        };
        let indptr = self.tensor(&indptr, 2, 1, Dtype::I32)?;
        let x = RaggedTensor {
            data: self.tensor(&qkv_in, TOKENS, channels, Dtype::Bf16)?,
            indptr,
        };
        let convolved = self.tensor(&qkv, TOKENS, channels, Dtype::Bf16)?;
        let weight = self.tensor(&weight, channels, CONV, Dtype::Bf16)?;
        let gates = self.tensor(&gates, TOKENS, 2 * HEADS, Dtype::F32)?;
        let y = self.tensor(&y, TOKENS, HEADS * DV, Dtype::F32)?;
        let seconds = self.timed(&|sink, _| {
            ssm::causal_conv1d_chunked(sink, x, weight, &pool, CONV, 1, convolved)?;
            ssm::gated_delta_chunked(
                sink,
                self.tuning,
                RaggedTensor {
                    data: convolved,
                    indptr,
                },
                convolved,
                gates,
                &pool,
                K_HEADS,
                HEADS,
                DK,
                DV,
                y,
            )
        })?;
        Ok(engine::rates::scan_elements(SCAN_SHAPE) / (seconds * 1e9))
    }

    fn bytes_per_ns(&self) -> Result<f64, String> {
        let bytes = 256u64 << 20;
        let (from, into) = (self.zeroed(bytes)?, self.zeroed(bytes)?);
        let mut times = Vec::with_capacity(5);
        for at in 0..6 {
            let mut frame = self.device.frame().map_err(|e| e.to_string())?;
            frame
                .copy_span(&from, 0, &into, 0, bytes)
                .map_err(|e| e.to_string())?;
            let seconds = frame.commit_timed().map_err(|e| e.to_string())?;
            if at > 0 {
                times.push(seconds);
            }
        }
        Ok(2.0 * bytes as f64 / (median(times) * 1e9))
    }

    fn launch_ns(&self) -> Result<u64, String> {
        let (a, w, y) = (
            self.zeroed(8 * 32 * 2)?,
            self.zeroed(32 * 32 * 2)?,
            self.zeroed(8 * 32 * 2)?,
        );
        let (at, wt, yt) = (
            self.tensor(&a, 8, 32, Dtype::Bf16)?,
            self.tensor(&w, 32, 32, Dtype::Bf16)?,
            self.tensor(&y, 8, 32, Dtype::Bf16)?,
        );
        let seconds = self.timed(&|sink, _| gemm::matmul(sink, self.tuning, at, wt, yt))?;
        Ok((seconds * 1e9).round() as u64)
    }
}

/// `device`'s rates under the kernel tuning of `profile` (the profile a
/// load on it runs), saying each as it lands.
pub fn measure(
    device: &Context,
    profile: profile::Profile,
    say: &mut dyn FnMut(&str),
) -> Result<Rates, String> {
    let tuning = Tuning::new(profile, false, device.cores()).with_family(device.family());
    let bench = Bench {
        device,
        handles: Handles::new(),
        pipelines: Pipelines::new(),
        tuning: &tuning,
        pattern: engine::rates::operand_pattern(),
    };
    // Warm the device until its rate stops falling (engine::rates::warm_up):
    // rotations of the widest matmul over its planes.
    {
        let (m, n, k) = engine::rates::WARM_SHAPE;
        let noise = bf16_noise((n * k) as usize, 5, 0.05);
        let weights = (0..engine::rates::weights_at(n, k))
            .map(|_| {
                let w = bench.buffer(&noise)?;
                Ok((bench.tensor(&w, n, k, Dtype::Bf16)?, w))
            })
            .collect::<Result<Vec<(Tensor, Buffer)>, String>>()?;
        let tensors: Vec<Tensor> = weights.iter().map(|(t, _)| *t).collect();
        let warmed = engine::rates::warm_up(|| bench.matmul_flops(m, n, k, &tensors))?;
        say(&format!(
            "  warm-up: {} rotations of the {m}^3 matmul, settled at {:.2} TFLOP/s",
            warmed.rotations,
            warmed.rate / 1e3
        ));
    }
    let mut grid = Vec::with_capacity(LATTICE_ROWS.len() * DEPTHS.len() * DEPTHS.len());
    // The shapes, and each shape's rows, in engine::rates' spread order.
    let shapes: Vec<(u32, u32)> = DEPTHS
        .iter()
        .flat_map(|&k| DEPTHS.iter().map(move |&n| (k, n)))
        .collect();
    for (k, n) in engine::rates::spread_order(shapes.len())
        .into_iter()
        .map(|at| shapes[at])
    {
        // The planes every backend rotates through (engine::rates::weights_at:
        // 768 MiB, at least 12), each its own memory so each repeat streams
        // its weight as a layer's matmul does; one plane of the operand
        // pattern fills them.
        let plane = operand(&bench.pattern, (n * k) as usize * 2);
        let weights = (0..engine::rates::weights_at(n, k))
            .map(|_| {
                let w = bench.buffer(&plane)?;
                Ok((bench.tensor(&w, n, k, Dtype::Bf16)?, w))
            })
            .collect::<Result<Vec<(Tensor, Buffer)>, String>>()?;
        let tensors: Vec<Tensor> = weights.iter().map(|(t, _)| *t).collect();
        for m in engine::rates::spread_order(LATTICE_ROWS.len())
            .into_iter()
            .map(|row| LATTICE_ROWS[row])
        {
            let rate = bench.matmul_flops(m, n, k, &tensors)?;
            say(&format!(
                "  bf16 matmul {m} x {n} x {k}: {:.2} TFLOP/s",
                rate / 1e3
            ));
            grid.push((m, n, k, rate));
        }
    }
    grid.sort_by_key(|&(m, n, k, _)| (k, n, m));
    let mut keys_curve = Vec::with_capacity(ATTN_KEYS.len() * ATTN_DIMS.len());
    for (d, keys) in ATTN_DIMS
        .iter()
        .flat_map(|&d| ATTN_KEYS.iter().map(move |&keys| (d, keys)))
    {
        let rate = bench.attention_flops(ATTN_SHAPE.at(keys).with_dim(d))?;
        say(&format!(
            "  prefill attention, {keys} rows over {keys} keys, d {d}: {:.2} TFLOP/s",
            rate / 1e3
        ));
        keys_curve.push((keys, d, rate));
    }
    let scan = bench.scan_elements()?;
    say(&format!(
        "  gated-delta scan and its conv: {scan:.1} state elements a ns"
    ));
    let bytes = bench.bytes_per_ns()?;
    say(&format!("  256 MiB blit: {bytes:.1} GB/s read and write"));
    let launch = bench.launch_ns()?;
    say(&format!("  launch: {launch} ns a dispatch"));
    Ok(Rates {
        bf16_flops_per_ns_by_rows_n_k: Some(grid),
        attn_flops_per_ns_by_keys_and_dim: Some(keys_curve),
        scan_elements_per_ns: Some(scan),
        bytes_per_ns: Some(bytes),
        launch_ns: Some(launch),
    })
}
