//! `pie tune metal --rates`: the device's rates as pie's Metal kernels reach
//! them, at the shapes every backend times ([`engine::rates`]), each through
//! the kernel serving dispatches under the device's own tuning:
//!
//! * the matmul curve: `linear::gemm::matmul` at each of
//!   the rung lattice ([`lattice_rows`]), each repeat over the next of
//!   engine::rates::weights_at's distinct weights so every one streams from memory, as a fire's do;
//! * attention: `attn::arbiter::prefill` at [`engine::rates::ATTN_SHAPE`],
//!   causal, over a bf16 pool of the 16-token pages serving builds, laid
//!   in engine::rates::attn_page_order's scattered order, each repeat the
//!   next of engine::rates::attn_sets_at's q / kv sets so each reads cold;
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

use engine::rates::{AttnShape, DEPTHS, SCAN_SHAPE};
use kernels_metal::linear::gemm;
use kernels_metal::tensor::{RaggedTensor, RecurrentPool};
use kernels_metal::{KvPool, PrefillPlan, StateMap, Tensor, Tuning, attn, ssm};
use poem_ir::Dtype;
use profile::fire::Rates;

/// The widest fire this backend's grid is timed to: the top of its rung
/// lattice, the tokens a served Metal load is asked ([`crate::FIRE`]).
pub const ROWS_TOP: u32 = crate::FIRE.1;

/// The rows a Metal load fires, and the matmul grid is timed at: Metal's
/// one rung lattice, every power of two of rows from 1 to [`ROWS_TOP`] and
/// that top. A load lays its rungs at these points below its top
/// (`serve::activation_rungs`), and a top it narrows to is one of them.
#[must_use]
pub fn lattice_rows() -> Vec<u32> {
    let mut rows: Vec<u32> = (0..=ROWS_TOP.ilog2()).map(|i| 1 << i).collect();
    if rows.last() != Some(&ROWS_TOP) {
        rows.push(ROWS_TOP);
    }
    rows
}

/// The lattice point below `tokens` a load narrows its fires to (to fit
/// its arena scratch beside the resident weights): the next rung down, so
/// every row it fires stays a point of [`lattice_rows`], and no narrower
/// than the first point holding a token for each of [`crate::FIRE`]'s
/// lanes, so one fire still carries every lane's decode.
#[must_use]
pub fn narrower(tokens: u32) -> u32 {
    let rows = lattice_rows();
    let floor = rows
        .iter()
        .copied()
        .find(|&rows| rows >= crate::FIRE.0)
        .unwrap_or(ROWS_TOP);
    rows.into_iter()
        .rev()
        .find(|&rows| rows < tokens)
        .unwrap_or(1)
        .max(floor)
}

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
        self.buffer(&operand(
            &self.pattern,
            usize::try_from(bytes).map_err(|e| e.to_string())?,
        ))
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
    fn matmul_flops(
        &self,
        m: u32,
        n: u32,
        k: u32,
        weights: &[Tensor],
        gap: f64,
    ) -> Result<f64, String> {
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
        engine::rates::rate_from(
            &format!("matmul {m}x{n}x{k}"),
            seconds * 1e9,
            engine::rates::GEMM_DISPATCHES,
            gap,
            engine::rates::gemm_flops(m, n, k),
        )
    }

    fn attention_flops(&self, shape: AttnShape, gap: f64) -> Result<f64, String> {
        let (rows, keys, q_heads, kv_heads, d) = (
            shape.rows,
            shape.keys,
            shape.q_heads,
            shape.kv_heads,
            shape.head_dim,
        );
        const PAGE: u32 = 16;
        let pages = keys.div_ceil(PAGE);
        // The sets each timed fire takes the next of (engine::rates::
        // attn_sets_at): a served fire reads each layer's q and kv cold, and
        // one set repeated would sit in the system cache. Every set's pages
        // lie in one pool plane, scattered over it as a served pool lays a
        // sequence's (engine::rates::attn_page_order), each set its own q.
        // Each set binds its own q and page-table buffers: at most
        // ATTN_SETS_MOST of them (a few-key point would lay thousands).
        const ATTN_SETS_MOST: usize = 256;
        let sets = u32::try_from(engine::rates::attn_sets_at(shape, PAGE).min(ATTN_SETS_MOST))
            .map_err(|e| e.to_string())?;
        let all = pages * sets;
        let slots = all * PAGE;
        let kv = u64::from(slots) * u64::from(kv_heads * d) * 2;
        let keys = self.operand(kv)?;
        let values = self.operand(kv)?;
        let out = self.zeroed(u64::from(rows * q_heads * d) * 2)?;
        let positions: Vec<u32> = (0..rows).collect();
        let order: Vec<u32> = engine::rates::attn_page_order(all);
        let le = u32::to_le_bytes;
        let mut held = Vec::with_capacity(2 * sets as usize);
        let mut sets_of = Vec::with_capacity(sets as usize);
        for set in order.chunks_exact(pages as usize) {
            let q = self.operand(u64::from(rows * q_heads * d) * 2)?;
            let ids = self.buffer(&words(set, le))?;
            sets_of.push((
                self.tensor(&q, rows, q_heads * d, Dtype::Bf16)?,
                self.tensor(&ids, pages, 1, Dtype::U32)?,
            ));
            held.extend([q, ids]);
        }
        let page_indptr = self.buffer(&words(&[0, pages], le))?;
        let positions = self.buffer(&words(&positions, le))?;
        let owner = self.buffer(&words(&vec![0u32; rows as usize], le))?;
        let q_indptr = self.buffer(&words(&[0, rows], le))?;
        let mask = self.buffer(&[1u8])?;
        let masked = self.buffer(&vec![0u8; rows as usize])?;
        let pool = KvPool {
            keys: self.tensor(&keys, slots, kv_heads * d, Dtype::Bf16)?,
            values: self.tensor(&values, slots, kv_heads * d, Dtype::Bf16)?,
            page_indices: sets_of[0].1,
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
        let indptr = self.tensor(&q_indptr, 2, 1, Dtype::I32)?;
        let o = self.tensor(&out, rows, q_heads * d, Dtype::Bf16)?;
        let scale = 1.0 / (d as f32).sqrt();
        let seconds = self.timed(&|sink, at| {
            let (q, page_indices) = sets_of[at as usize % sets_of.len()];
            attn::arbiter::prefill(
                sink,
                RaggedTensor { data: q, indptr },
                &plan,
                &KvPool {
                    page_indices,
                    ..pool
                },
                None,
                d,
                kv_heads,
                scale,
                o,
                1,
                self.tuning,
            )
        })?;
        drop(held);
        engine::rates::rate_from(
            &format!(
                "attention {}x{} d {}",
                shape.rows, shape.keys, shape.head_dim
            ),
            seconds * 1e9,
            engine::rates::ATTN_DISPATCHES,
            gap,
            engine::rates::attn_flops(shape),
        )
    }

    fn scan_elements(&self, tokens: u32, gap: f64) -> Result<f64, String> {
        #[allow(non_snake_case)]
        let TOKENS: u32 = tokens;
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
        engine::rates::rate_from(
            &format!("scan of {tokens} tokens"),
            seconds * 1e9,
            engine::rates::SCAN_DISPATCHES,
            gap,
            engine::rates::scan_elements(SCAN_SHAPE.at(tokens)),
        )
    }

    fn bytes_per_ns(&self, gap: f64) -> Result<f64, String> {
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
        // Every byte copied is read and written.
        engine::rates::rate_from(
            "copy",
            median(times) * 1e9,
            engine::rates::COPY_DISPATCHES,
            gap,
            2.0 * bytes as f64,
        )
    }

    /// What a dependent dispatch adds to its work in one encoder
    /// (engine::rates::launch_from): a chain of tiny matmuls, each reading
    /// the one before's output, against one matmul of `n` times the rows,
    /// at each length of engine::rates::LAUNCH_CHAINS; the longest chain's.
    fn launch_ns(&self) -> Result<(u64, Vec<(u32, f64)>), String> {
        let (a, w, y) = (
            self.zeroed(32 * 32 * 2)?,
            self.zeroed(32 * 32 * 2)?,
            self.zeroed(32 * 32 * 2)?,
        );
        let (at, wt, yt) = (
            self.tensor(&a, 32, 32, Dtype::Bf16)?,
            self.tensor(&w, 32, 32, Dtype::Bf16)?,
            self.tensor(&y, 32, 32, Dtype::Bf16)?,
        );
        // One dispatch of the chain, timed back to back: each reads the
        // output of the one before.
        let link = self.timed(&|sink, i| {
            let (from, into) = if i % 2 == 0 { (at, yt) } else { (yt, at) };
            gemm::matmul(sink, self.tuning, from, wt, into)
        })?;
        let mut gaps = Vec::new();
        for n in engine::rates::LAUNCH_CHAINS {
            let (wa, wy) = (
                self.zeroed(u64::from(32 * n) * 32 * 2)?,
                self.zeroed(u64::from(32 * n) * 32 * 2)?,
            );
            let (wat, wyt) = (
                self.tensor(&wa, 32 * n, 32, Dtype::Bf16)?,
                self.tensor(&wy, 32 * n, 32, Dtype::Bf16)?,
            );
            let one = self.timed(&|sink, _| gemm::matmul(sink, self.tuning, wat, wt, wyt))?;
            gaps.push((
                n,
                engine::rates::launch_from(link * 1e9 * f64::from(n), one * 1e9, n),
            ));
        }
        let launch = gaps.last().map_or(0.0, |&(_, ns)| ns).round() as u64;
        Ok((launch, gaps))
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
    let mut bench = Bench {
        device,
        handles: Handles::new(),
        pipelines: Pipelines::new(),
        tuning: &tuning,
        pattern: engine::rates::operand_pattern(),
    };
    // The launch first: every rate below takes its dispatches' launches off
    // (engine::rates::rate_from).
    let (launch, gaps) = bench.launch_ns()?;
    say(&format!(
        "  launch: {}",
        gaps.iter()
            .map(|(n, ns)| format!("{ns:.0} ns a dispatch over a chain of {n}"))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    let gap = launch as f64;
    // Warm the device until its rate stops falling (engine::rates::warm_up):
    // rotations of the widest matmul over its planes.
    {
        let (m, n, k) = engine::rates::WARM_SHAPE;
        let plane = operand(&bench.pattern, (n * k) as usize * 2);
        let weights = (0..engine::rates::weights_at(n, k))
            .map(|_| {
                let w = bench.buffer(&plane)?;
                Ok((bench.tensor(&w, n, k, Dtype::Bf16)?, w))
            })
            .collect::<Result<Vec<(Tensor, Buffer)>, String>>()?;
        let tensors: Vec<Tensor> = weights.iter().map(|(t, _)| *t).collect();
        let warmed = engine::rates::warm_up(|| bench.matmul_flops(m, n, k, &tensors, 0.0))?;
        say(&format!(
            "  warm-up: {} rotations of the {m}^3 matmul, settled at {:.2} TFLOP/s",
            warmed.rotations,
            warmed.rate / 1e3
        ));
    }
    // The shapes, walked by engine::rates::saturate (the rule every
    // engine's rates step takes) up to the device's saturation point on the
    // lattice, at most FIRE.1: the fire a served Metal load is asked, the
    // ceiling the worker's request sets, not a limit of the device.
    let lattice = lattice_rows();
    let shapes: Vec<(u32, u32)> = DEPTHS
        .iter()
        .flat_map(|&k| DEPTHS.iter().map(move |&n| (n, k)))
        .collect();
    // A shape's planes, laid at the start of each run of it: the planes
    // every backend rotates through (engine::rates::weights_at: 768 MiB, at
    // least 12), each its own memory so each repeat streams its weight as a
    // layer's matmul does; one plane of the operand pattern fills them.
    let mut laid: Option<(Vec<Tensor>, Vec<Buffer>)> = None;
    let (rates, top) = engine::rates::saturate(&lattice, shapes.len(), |at, row, fresh| {
        let (n, k) = shapes[at];
        if fresh {
            // The handles let the last shape's planes go (a grid of every
            // (n, k) holds no more than one shape's).
            laid = None;
            bench.handles = Handles::new();
            let plane = operand(&bench.pattern, (n * k) as usize * 2);
            let weights = (0..engine::rates::weights_at(n, k))
                .map(|_| {
                    let w = bench.buffer(&plane)?;
                    Ok((bench.tensor(&w, n, k, Dtype::Bf16)?, w))
                })
                .collect::<Result<Vec<(Tensor, Buffer)>, String>>()?;
            laid = Some(weights.into_iter().unzip());
        }
        let (tensors, _) = laid.as_ref().expect("laid");
        let m = lattice[row];
        let rate = bench.matmul_flops(m, n, k, tensors, gap)?;
        say(&format!(
            "  bf16 matmul {m} x {n} x {k}: {:.4} TFLOP/s",
            rate / 1e3
        ));
        Ok::<_, String>((rate, engine::rates::gemm_cost_per_row(m, n, k, rate, gap)))
    })?;
    drop(laid);
    say(&format!(
        "  the grid saturates at {} rows of a lattice to {}: past it no (n, k) shape's cost a \
         row fell more than {:.0}% below its least through {} (at it the most {:.1}%)",
        top.rows,
        lattice.last().copied().unwrap_or(0),
        100.0 * engine::rates::AGREEMENT,
        top.through,
        100.0 * top.best_gain,
    ));
    // The grid ends at the saturation point: a load's rungs run to it.
    let mut grid = Vec::with_capacity((top.top + 1) * shapes.len());
    for (at, &(n, k)) in shapes.iter().enumerate() {
        for (i, &m) in lattice.iter().enumerate().take(top.top + 1) {
            grid.push((m, n, k, rates[at][i]));
        }
    }
    grid.sort_by_key(|&(m, n, k, _)| (k, n, m));
    // The attention at the grid's rows to its saturation point
    // (engine::rates::attention_at_rows, every engine's rule).
    let keys_curve = engine::rates::attention_at_rows(&lattice, top.rows, |shape| {
        bench.handles = Handles::new();
        let rate = bench.attention_flops(shape, gap)?;
        say(&format!(
            "  prefill attention, {0} rows over {0} keys, d {1}: {2:.4} TFLOP/s",
            shape.keys,
            shape.head_dim,
            rate / 1e3
        ));
        Ok::<_, String>(rate)
    })?;
    bench.handles = Handles::new();
    let mut scan_curve = Vec::with_capacity(engine::rates::SCAN_TOKENS.len());
    for tokens in engine::rates::SCAN_TOKENS {
        scan_curve.push((tokens, bench.scan_elements(tokens, gap)?));
    }
    let scan = scan_curve.iter().map(|&(_, r)| r).fold(0.0, f64::max);
    say(&format!(
        "  gated-delta scan and its conv: {scan:.1} state elements a ns"
    ));
    let bytes = bench.bytes_per_ns(gap)?;
    say(&format!("  256 MiB blit: {bytes:.1} GB/s read and write"));
    Ok(Rates {
        bf16_flops_per_ns_by_rows_n_k: Some(grid),
        attn_flops_per_ns_by_keys_and_dim: Some(keys_curve),
        scan_elements_per_ns_by_tokens: Some(scan_curve),
        bytes_per_ns: Some(bytes),
        launch_ns: Some(launch),
    })
}

#[cfg(test)]
mod lattice_tests {
    use super::*;

    #[test]
    fn the_lattice_is_the_powers_of_two_up_to_the_fire_and_the_fire() {
        let rows = lattice_rows();
        assert_eq!(rows.first(), Some(&1));
        assert_eq!(rows[rows.len() - 2..], [8192, crate::FIRE.1]);
        assert!(rows[..rows.len() - 1].iter().all(|r| r.is_power_of_two()));
    }

    #[test]
    fn a_narrowed_top_is_the_next_lattice_point_down() {
        assert_eq!(narrower(10240), 8192);
        assert_eq!(narrower(8192), 4096);
        assert_eq!(narrower(1024), crate::FIRE.0);
        assert_eq!(
            narrower(crate::FIRE.0),
            crate::FIRE.0,
            "no narrower than a token a lane"
        );
        assert!(lattice_rows().contains(&narrower(10240)));
    }
}
