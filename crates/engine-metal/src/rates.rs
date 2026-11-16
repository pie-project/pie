//! `pie tune metal --rates`: the device's rates as pie's Metal kernels reach
//! them, for a profile's `[rates]`. Every rate is the median of five fires
//! timed by the GPU (`Frame::commit_timed`) after a warm one, each through
//! the kernel serving dispatches under the device's own tuning, and none
//! depends on a model:
//!
//! * `bf16_flops_per_ns_by_rows`: a bf16 matmul of `m × 4096 × 4096`
//!   (`linear::gemm::matmul`) at `m` = 16, 32, …, 8192, each repeat over
//!   the next of 8 distinct weights so every one streams from memory;
//! * `attn_flops_per_ns`: a causal prefill (`attn::arbiter::prefill`) of
//!   2048 query rows over 2048 keys, 16 query and 8 kv heads of 128, over a
//!   bf16 pool of the 16-token pages serving builds; its flops are
//!   `4 · rows · keys · q_heads · d / 2` (causal: half the pairs);
//! * `scan_elements_per_ns`: a gated-delta chunked scan
//!   (`ssm::gated_delta_chunked`) and the width-4 causal conv before it
//!   (`ssm::causal_conv1d_chunked`), 2048 tokens, 16 heads, `d_k = d_v =
//!   128`; its elements are `tokens · heads · d_k · d_v`;
//! * `bytes_per_ns`: a 256 MiB blit between two buffers, read and write
//!   counted;
//! * `launch_ns`: a dispatch's launch, the smallest matmul back to back in
//!   one frame.
//!
//! Each kernel is timed back to back with itself, as many to a frame as
//! fill 20 ms, and the frame's time divided: serving runs hundreds of
//! dispatches a fire, and a frame of one short dispatch times the clock
//! state the device idled into, not the kernel.

use std::path::Path;

use kernels_metal::linear::gemm;
use kernels_metal::tensor::{RaggedTensor, RecurrentPool};
use kernels_metal::{KvPool, PrefillPlan, StateMap, Tensor, Tuning, attn, ssm};
use poem_ir::Dtype;

use crate::device::{Buffer, Context, Handles, Pipelines};
use crate::encode::Sink;

/// The rows the matmul ladder is timed at: `m × 4096 × 4096`.
pub const ROWS: [u32; 10] = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];

/// The matmul's width and contraction.
const WIDTH: u32 = 4096;

/// How long a timed frame runs: long enough that the device reaches and
/// holds its serving clock through it. A narrow kernel's frame of 5 ms read
/// the m = 16 matmul 26% apart run to run on an M1 Max (45% on an M5 Pro);
/// 20 ms reads it within 0.25%.
const FRAME_SECONDS: f64 = 0.020;

/// The most of one measurement a timed frame repeats.
const MAX_REPEATS: u32 = 512;

/// The distinct `4096 × 4096` bf16 weights (32 MiB each) the matmul ladder
/// rotates over: 256 MiB, well past any Apple GPU's system-level cache.
const WEIGHTS: u64 = 8;

/// What one `pie tune metal --rates` measured.
#[derive(Debug, Clone, PartialEq)]
pub struct Measured {
    pub device: String,
    /// `(m, bf16 flops a ns)` at each of [`ROWS`].
    pub flops_by_rows: Vec<(u32, f64)>,
    pub attn_flops_per_ns: f64,
    pub scan_elements_per_ns: f64,
    pub bytes_per_ns: f64,
    pub launch_ns: u64,
}

impl Measured {
    /// The profile's `[rates]` block.
    #[must_use]
    pub fn block(&self) -> String {
        let ladder: Vec<String> = self
            .flops_by_rows
            .iter()
            .map(|(m, r)| format!("[{m}, {r:.1}]"))
            .collect();
        format!(
            "[rates]\nbf16_flops_per_ns_by_rows = [{}]\nattn_flops_per_ns = {:.1}\n\
             scan_elements_per_ns = {:.1}\nbytes_per_ns = {:.1}\nlaunch_ns = {}\n",
            ladder.join(", "),
            self.attn_flops_per_ns,
            self.scan_elements_per_ns,
            self.bytes_per_ns,
            self.launch_ns
        )
    }
}

fn median(mut times: Vec<f64>) -> f64 {
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

/// Deterministic bf16 values in `[-scale, scale)`.
fn bf16_noise(n: usize, seed: u64, scale: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 2);
    for at in 0..n as u64 {
        let mut x = at.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ seed;
        x ^= x >> 33;
        x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        let v = ((x >> 40) as f32 / 16_777_216.0 - 0.5) * 2.0 * scale;
        out.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
    }
    out
}

fn words<T: Copy>(v: &[T], to: impl Fn(T) -> [u8; 4]) -> Vec<u8> {
    v.iter().flat_map(|&x| to(x)).collect()
}

struct Bench<'a> {
    device: &'a Context,
    handles: Handles,
    pipelines: Pipelines,
    tuning: &'a Tuning,
}

impl Bench<'_> {
    fn buffer(&self, bytes: &[u8]) -> Result<Buffer, String> {
        let mut b = Buffer::zeroed(self.device, bytes.len().max(4) as u64).map_err(|e| e.to_string())?;
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

    /// `weights` the [`WEIGHTS`] distinct weight buffers a matmul ladder
    /// rotates over.
    fn matmul_flops(&self, m: u32, weights: &[Tensor]) -> Result<f64, String> {
        let (a, y) = (
            self.zeroed(u64::from(m) * u64::from(WIDTH) * 2)?,
            self.zeroed(u64::from(m) * u64::from(WIDTH) * 2)?,
        );
        let (at, yt) = (
            self.tensor(&a, m, WIDTH, Dtype::Bf16)?,
            self.tensor(&y, m, WIDTH, Dtype::Bf16)?,
        );
        // Each repeat reads the next weight: serving streams a layer's from
        // memory, and one weight repeated would sit in the system cache,
        // by a share that changes run to run (2x at m = 16 on an M5 Pro).
        let seconds = self.timed(&|sink, at_k| {
            gemm::matmul(sink, self.tuning, at, weights[at_k as usize % weights.len()], yt)
        })?;
        Ok(2.0 * f64::from(m) * f64::from(WIDTH) * f64::from(WIDTH) / (seconds * 1e9))
    }

    fn attention_flops(&self) -> Result<f64, String> {
        const ROWS: u32 = 2048;
        const KEYS: u32 = 2048;
        const Q_HEADS: u32 = 16;
        const KV_HEADS: u32 = 8;
        const D: u32 = 128;
        const PAGE: u32 = 16;
        let pages = KEYS.div_ceil(PAGE);
        let slots = pages * PAGE;
        let kv = (slots * KV_HEADS * D) as usize;
        let keys = self.buffer(&bf16_noise(kv, 11, 1.0))?;
        let values = self.buffer(&bf16_noise(kv, 13, 1.0))?;
        let q = self.buffer(&bf16_noise((ROWS * Q_HEADS * D) as usize, 17, 1.0))?;
        let out = self.zeroed(u64::from(ROWS * Q_HEADS * D) * 2)?;
        let positions: Vec<u32> = (0..ROWS).collect();
        let page_ids: Vec<u32> = (0..pages).collect();
        let le = u32::to_le_bytes;
        let page_indices = self.buffer(&words(&page_ids, le))?;
        let page_indptr = self.buffer(&words(&[0, pages], le))?;
        let positions = self.buffer(&words(&positions, le))?;
        let owner = self.buffer(&words(&vec![0u32; ROWS as usize], le))?;
        let q_indptr = self.buffer(&words(&[0, ROWS], le))?;
        let mask = self.buffer(&[1u8])?;
        let masked = self.buffer(&vec![0u8; ROWS as usize])?;
        let pool = KvPool {
            keys: self.tensor(&keys, slots, KV_HEADS * D, Dtype::Bf16)?,
            values: self.tensor(&values, slots, KV_HEADS * D, Dtype::Bf16)?,
            page_indices: self.tensor(&page_indices, pages, 1, Dtype::U32)?,
            page_indptr: self.tensor(&page_indptr, 2, 1, Dtype::U32)?,
            page_size: PAGE as i32,
            seq_stride: u64::from(KV_HEADS * D),
            head_stride: u64::from(D),
            q8: false,
            page_stride: [0; 2],
        }
        .paged_densely();
        let plan = PrefillPlan {
            positions: self.tensor(&positions, ROWS, 1, Dtype::I32)?,
            last_page_len: self.tensor(&positions, ROWS, 1, Dtype::I32)?,
            request_of_token: self.tensor(&owner, ROWS, 1, Dtype::I32)?,
            mask: self.tensor(&mask, 1, 1, Dtype::U8)?,
            mask_enabled: self.tensor(&masked, ROWS, 1, Dtype::U8)?,
            mask_stride: 0,
        };
        let q = RaggedTensor {
            data: self.tensor(&q, ROWS, Q_HEADS * D, Dtype::Bf16)?,
            indptr: self.tensor(&q_indptr, 2, 1, Dtype::I32)?,
        };
        let o = self.tensor(&out, ROWS, Q_HEADS * D, Dtype::Bf16)?;
        let scale = 1.0 / (D as f32).sqrt();
        let seconds = self.timed(&|sink, _| {
            attn::arbiter::prefill(sink, q, &plan, &pool, None, D, KV_HEADS, scale, o, 1, self.tuning)
        })?;
        let flops = 4.0 * f64::from(ROWS) * f64::from(KEYS) * f64::from(Q_HEADS) * f64::from(D) / 2.0;
        Ok(flops / (seconds * 1e9))
    }

    fn scan_elements(&self) -> Result<f64, String> {
        const TOKENS: u32 = 2048;
        const HEADS: u32 = 16;
        const DK: u32 = 128;
        const DV: u32 = 128;
        const CONV: u32 = 4;
        let channels = 2 * HEADS * DK + HEADS * DV;
        let bank = u64::from(HEADS * DV * DK);
        let qkv_in = self.buffer(&bf16_noise((TOKENS * channels) as usize, 23, 0.5))?;
        let qkv = self.zeroed(u64::from(TOKENS * channels) * 2)?;
        let weight = self.buffer(&bf16_noise((channels * CONV) as usize, 29, 0.5))?;
        let mut gates = Vec::with_capacity((TOKENS * 2 * HEADS) as usize);
        for t in 0..TOKENS {
            gates.extend((0..HEADS).map(|h| -0.05 - 0.001 * ((t + h) % 7) as f32));
            gates.extend((0..HEADS).map(|h| 0.5 + 0.01 * ((t * 3 + h) % 11) as f32));
        }
        let gates = self.buffer(&words(&gates, f32::to_le_bytes))?;
        let state = self.zeroed(bank * 4)?;
        let conv_state = self.zeroed(u64::from(CONV * channels) * 2)?;
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
                HEADS,
                HEADS,
                DK,
                DV,
                y,
            )
        })?;
        let elements = f64::from(TOKENS) * f64::from(HEADS) * f64::from(DK) * f64::from(DV);
        Ok(elements / (seconds * 1e9))
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

/// Measures this device's rates under its resolved profile (the user's
/// shelf in `profiles`, else the shipped file), saying each as it lands.
pub fn measure(profiles: Option<&Path>, say: &mut dyn FnMut(&str)) -> Result<Measured, String> {
    let device = Context::bind().map_err(|e| e.to_string())?;
    let name = device.name().to_string();
    let facts = crate::chip::measured(&device, crate::chip::perf_cores());
    let (profile, report) = crate::chip::resolve(&name, &facts, profiles);
    say(&format!("{name}: measuring the rates under {}", report.used));
    let tuning = Tuning::new(profile, false, device.cores()).with_family(device.family());
    let bench = Bench {
        device: &device,
        handles: Handles::new(),
        pipelines: Pipelines::new(),
        tuning: &tuning,
    };
    let weights = (0..WEIGHTS)
        .map(|seed| {
            let w = bench.buffer(&bf16_noise((WIDTH * WIDTH) as usize, 7 + seed, 0.05))?;
            Ok((bench.tensor(&w, WIDTH, WIDTH, Dtype::Bf16)?, w))
        })
        .collect::<Result<Vec<(Tensor, Buffer)>, String>>()?;
    let weight_tensors: Vec<Tensor> = weights.iter().map(|(t, _)| *t).collect();
    let mut flops_by_rows = Vec::with_capacity(ROWS.len());
    for m in ROWS {
        let rate = bench.matmul_flops(m, &weight_tensors)?;
        say(&format!("  bf16 matmul {m} x {WIDTH} x {WIDTH}: {:.2} TFLOP/s", rate / 1e3));
        flops_by_rows.push((m, rate));
    }
    let attn_flops_per_ns = bench.attention_flops()?;
    say(&format!(
        "  causal prefill 2048 x 2048, 16/8 heads of 128: {:.2} TFLOP/s",
        attn_flops_per_ns / 1e3
    ));
    let scan_elements_per_ns = bench.scan_elements()?;
    say(&format!(
        "  gated-delta scan + conv, 2048 tokens, 16 heads of 128x128: {:.1} G elements/s",
        scan_elements_per_ns
    ));
    let bytes_per_ns = bench.bytes_per_ns()?;
    say(&format!("  256 MiB blit: {bytes_per_ns:.1} GB/s read+write"));
    let launch_ns = bench.launch_ns()?;
    say(&format!("  launch: {launch_ns} ns a dispatch"));
    Ok(Measured {
        device: name,
        flops_by_rows,
        attn_flops_per_ns,
        scan_elements_per_ns,
        bytes_per_ns,
        launch_ns,
    })
}
