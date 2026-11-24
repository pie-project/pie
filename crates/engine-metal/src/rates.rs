//! `pie tune metal --rates`: the device's rates as pie's Metal kernels reach
//! them, at the shapes every backend times ([`engine::rates`]), each through
//! the kernel serving dispatches under the device's own tuning:
//!
//! * the matmul curves: `linear::gemm::matmul`, `m x w x w` at each of
//!   [`engine::rates::GEMM_ROWS`] for each width `w` of
//!   [`engine::rates::GEMM_WIDTHS`], each repeat over
//!   the next of engine::rates::weights_at's distinct weights so every one
//!   streams from memory, as a fire's do;
//! * the attention curve: `attn::arbiter::prefill` at
//!   [`engine::rates::ATTN_SHAPE`] over each of [`engine::rates::ATTN_KEYS`]
//!   keys at d 128 (64 and 256 at the longest keys, said beside it),
//!   causal, over a bf16 pool of the 16-token pages serving builds, laid
//!   in engine::rates::attn_page_order's scattered order, each repeat the
//!   next of engine::rates::attn_sets_at's q / kv sets so each reads cold;
//! * the per-row ops: `elemwise::norm::rmsnorm` then `linear::mlp::swiglu`
//!   at [`engine::rates::ELEMENTWISE_SHAPE`], each repeat the next of
//!   engine::rates::elementwise_sets' sets;
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

use engine::rates::{AttnShape, SCAN_SHAPE, Touched};
use kernels_metal::linear::gemm;
use kernels_metal::tensor::{RaggedTensor, RecurrentPool};
use kernels_metal::{KvPool, PrefillPlan, StateMap, Tensor, Tuning, attn, ssm};
use poem_ir::Dtype;
use profile::fire::Rates;


/// The rates this backend's estimate weighs a fire by, of
/// `profile::fire::KEYS` (every one): what its load measures where its
/// chip's profile states none.
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

/// The widest fire this backend's lattice reaches: the tokens a served
/// Metal load is asked ([`crate::FIRE`]).
pub const ROWS_TOP: u32 = crate::FIRE.1;

/// The rows a Metal load fires: Metal's one rung lattice, every power of two of rows from 1 to [`ROWS_TOP`] and
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

    /// GPU seconds of one frame holding `k` of what `encode` fills, each
    /// told its place in the measurement (`turn`, one counter across every
    /// frame of it: a frame reads the next operand sets, never the first
    /// ones again).
    fn frame_of(
        &self,
        turn: &engine::rates::Rotation,
        encode: &dyn Fn(&Sink<'_>, u32) -> Result<(), kernels_metal::Error>,
        k: u32,
    ) -> Result<f64, String> {
        let frame = self.device.frame().map_err(|e| e.to_string())?;
        let sink = Sink::new(self.device, &frame, &self.pipelines, &self.handles);
        for _ in 0..k {
            let at = turn.next(u32::MAX as usize) as u32;
            encode(&sink, at).map_err(|e| e.to_string())?;
        }
        frame.commit_timed().map_err(|e| e.to_string())
    }

    /// The workgroups one of what `encode` fills launches, from an encode the
    /// device never runs (its frame is dropped uncommitted).
    fn groups_of(
        &self,
        encode: &dyn Fn(&Sink<'_>, u32) -> Result<(), kernels_metal::Error>,
    ) -> Result<u64, String> {
        let frame = self.device.frame().map_err(|e| e.to_string())?;
        let sink = Sink::new(self.device, &frame, &self.pipelines, &self.handles);
        encode(&sink, 0).map_err(|e| e.to_string())?;
        Ok(sink.groups())
    }

    /// The workgroups an `m x n x k` matmul launches on the kernel serving
    /// picks there.
    fn matmul_groups(&self, m: u32, n: u32, k: u32, weight: Tensor) -> Result<u64, String> {
        let (a, y) = (
            self.operand(u64::from(m) * u64::from(k) * 2)?,
            self.zeroed(u64::from(m) * u64::from(n) * 2)?,
        );
        let (at, yt) = (
            self.tensor(&a, m, k, Dtype::Bf16)?,
            self.tensor(&y, m, n, Dtype::Bf16)?,
        );
        self.groups_of(&|sink, _| gemm::matmul(sink, self.tuning, at, weight, yt))
    }

    /// The GPU seconds one of what `encode` fills takes as serving runs it,
    /// back to back with others: each frame holds `k` of them, `k` doubled
    /// until a fire's time settles, then such frames until their rate stops
    /// moving (`engine::rates::warmed_frames`, every engine's rule), and the
    /// median of five such frames after a warm one, over `k`. A frame of one short dispatch times the clock state the
    /// device idled into between frames, not the kernel (bimodal on an M5
    /// Pro at m = 64..2048 of the matmul ladder; 5 ms frames read the m = 16
    /// matmul 26% apart run to run on an M1 Max, 20 ms within 0.25%).
    /// `sets` the operand sets `encode` rotates through, touched once by
    /// the first point over them (`engine::rates::touch_once`; `touched`
    /// held beside them by whoever laid them).
    fn timed(
        &self,
        touched: &Touched,
        sets: usize,
        encode: &dyn Fn(&Sink<'_>, u32) -> Result<(), kernels_metal::Error>,
    ) -> Result<f64, String> {
        let turn = engine::rates::Rotation::default();
        engine::rates::touch_once(
            touched,
            None,
            u32::try_from(sets).unwrap_or(u32::MAX),
            &mut |k| self.frame_of(&turn, encode, k),
        )?;
        self.frame_of(&turn, encode, 1)?;
        let k = engine::rates::warmed_frames(
            None,
            None,
            |k| self.frame_of(&turn, encode, k).map(|s| s * 1e9),
            |why| why,
        )?
        .fires;
        self.frame_of(&turn, encode, k)?;
        let mut times = Vec::with_capacity(5);
        for _ in 0..5 {
            times.push(self.frame_of(&turn, encode, k)? / f64::from(k));
        }
        Ok(median(times))
    }

    /// An `m × k × k` matmul's rate, each repeat over the next of `weights`
    /// (`k × k` each), `touched` held beside them.
    fn matmul_flops(
        &self,
        m: u32,
        n: u32,
        k: u32,
        weights: &[Tensor],
        touched: &Touched,
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
        let seconds = self.timed(touched, weights.len(), &|sink, at_k| {
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

    /// The per-row ops' bandwidth: the rmsnorm then the swiglu at
    /// `engine::rates::ELEMENTWISE_SHAPE`, each repeat the next of
    /// `engine::rates::elementwise_sets` sets, their bytes over their time.
    fn elementwise_bytes(&self, gap: f64) -> Result<f64, String> {
        let (rows, width) = engine::rates::ELEMENTWISE_SHAPE;
        let row = u64::from(rows) * u64::from(width) * 2;
        let weight = self.operand(u64::from(width) * 2)?;
        let wt = self.tensor(&weight, 1, width, Dtype::Bf16)?;
        let mut held = Vec::new();
        let mut sets = Vec::new();
        for _ in 0..engine::rates::elementwise_sets() {
            let (x, normed, packed, y) = (
                self.operand(row)?,
                self.zeroed(row)?,
                self.operand(2 * row)?,
                self.zeroed(row)?,
            );
            sets.push((
                self.tensor(&x, rows, width, Dtype::Bf16)?,
                self.tensor(&normed, rows, width, Dtype::Bf16)?,
                self.tensor(&packed, rows, 2 * width, Dtype::Bf16)?,
                self.tensor(&y, rows, width, Dtype::Bf16)?,
            ));
            held.extend([x, normed, packed, y]);
        }
        let seconds = self.timed(&Touched::default(), sets.len(), &|sink, at| {
            let (x, normed, packed, y) = sets[at as usize % sets.len()];
            kernels_metal::elemwise::norm::rmsnorm(sink, x, wt, 1e-6, normed)?;
            kernels_metal::linear::mlp::swiglu(sink, packed, width, y)
        })?;
        drop(held);
        engine::rates::rate_from(
            "per-row ops",
            seconds * 1e9,
            2,
            gap,
            engine::rates::elementwise_bytes(),
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
        let seconds = self.timed(&Touched::default(), sets_of.len(), &|sink, at| {
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
        let seconds = self.timed(&Touched::default(), 1, &|sink, _| {
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
        let blit = || -> Result<f64, String> {
            let mut frame = self.device.frame().map_err(|e| e.to_string())?;
            frame
                .copy_span(&from, 0, &into, 0, bytes)
                .map_err(|e| e.to_string())?;
            frame.commit_timed().map_err(|e| e.to_string())
        };
        // A memory-bound op: its own blits warm the GPU until their rate
        // stops moving (`engine::rates::warm_up`'s rule) before it is timed.
        engine::rates::warm_up(|| blit().map(|s| 1.0 / s.max(1e-12)))?;
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
        let link = self.timed(&Touched::default(), 2, &|sink, i| {
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
            let one = self.timed(&Touched::default(), 1, &|sink, _| {
                gemm::matmul(sink, self.tuning, wat, wt, wyt)
            })?;
            gaps.push((
                n,
                engine::rates::launch_from(link * 1e9 * f64::from(n), one * 1e9, n),
            ));
        }
        let launch = gaps.last().map_or(0.0, |&(_, ns)| ns).round() as u64;
        Ok((launch, gaps))
    }
}

/// The `[provenance]` entry a measurement of `device`'s rates writes beside
/// them (`engine::rates::provenance`): how Metal measured them.
#[must_use]
pub fn provenance(device: &str) -> (&'static str, String) {
    engine::rates::provenance(
        "metal",
        &format!(
            "{device}; each kernel through the dispatch serving runs, back to back in frames \
             doubled until a fire's time settles, timed by the GPU, the matmul over engine::rates::weights_at planes, attention \
             over engine::rates::attn_sets_at sets; operands: engine::rates' pattern; at {}",
            engine::rates::build_commit()
        ),
    )
}

/// `chip` with every roofline constant stated
/// (`engine::rates::rates_at_load`, every engine's rule): as its layers
/// state them where they state them all; else `device`'s own, measured
/// before the load is ready, saved to the user's profile under `profiles`.
/// Metal states no bound on one submission yet, so its frames settle
/// uncapped (`engine::rates::timed_frames`).
pub fn at_load(
    device: &Context,
    chip: profile::Profile,
    _facts: &profile::Facts,
    profiles: Option<&std::path::Path>,
    max_submit: Option<std::time::Duration>,
    measuring: engine::rates::Measuring,
) -> Result<profile::Profile, String> {
    let name = device.name().to_string();
    let save = profiles.map(|profiles| engine::rates::SaveTo {
        path: crate::chip::SHELF.user_path(profiles, &name),
        provenance: provenance(&name),
        registry: crate::chip::registry(),
    });
    let measure_chip = chip.clone();
    // Nothing is timed behind serving on Metal yet: a stale key is
    // measured before ready.
    engine::rates::resolve_at_load(chip, &KEYS, save, None, max_submit, measuring, engine::rates::StaleRates::Measured, |_max_submit, _| {
        Ok(engine::rates::Measured {
            rates: measure(device, measure_chip, &mut |_| {})?,
            unsettled: false,
            throttled: None,
        })
    })
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
    // Each rate's key names the kernels its measurement fired
    // (`crate::stored::by`), recorded phase by phase.
    let _ = crate::encode::record_fired(true);
    let (launch, gaps) = bench.launch_ns()?;
    let fired_launch = crate::encode::record_fired(true);
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
        let touched = Touched::default();
        let warmed =
            engine::rates::warm_up(|| bench.matmul_flops(m, n, k, &tensors, &touched, 0.0))?;
        say(&format!(
            "  warm-up: {} rotations of the {m}^3 matmul, settled at {:.2} TFLOP/s",
            warmed.rotations,
            warmed.rate / 1e3
        ));
    }
    // The matmul curves: at each width w of engine::rates::GEMM_WIDTHS,
    // m x w x w at each of engine::rates::GEMM_ROWS, over its planes
    // (engine::rates::weights_at: 768 MiB, at least 12, so each repeat
    // streams its weight as a layer's matmul does).
    let mut gemm = Vec::new();
    let mut few_row = None;
    let mut per_core = None;
    for w in engine::rates::GEMM_WIDTHS {
        bench.handles = Handles::new();
        let plane = operand(&bench.pattern, (w * w) as usize * 2);
        let weights = (0..engine::rates::weights_at(w, w))
            .map(|_| {
                let b = bench.buffer(&plane)?;
                Ok((bench.tensor(&b, w, w, Dtype::Bf16)?, b))
            })
            .collect::<Result<Vec<(Tensor, Buffer)>, String>>()?;
        let tensors: Vec<Tensor> = weights.iter().map(|(t, _)| *t).collect();
        // The width's planes, touched once by its first point.
        let touched = Touched::default();
        let mut curve = Vec::new();
        for m in engine::rates::GEMM_ROWS {
            let rate = bench.matmul_flops(m, w, w, &tensors, &touched, gap)?;
            say(&format!(
                "  bf16 matmul {m} x {w} x {w}: {:.4} TFLOP/s",
                rate / 1e3
            ));
            curve.push((m, rate / 1e3));
        }
        if w == engine::rates::GEMM_WIDTH {
            // `few_row_tflops`: the matmul at decode-shaped rows on the kernel
            // serving picks there. Its time is its launch and the longer of
            // its weight reads and its flops at that rate
            // (`roofline::Constants::few_row_ns`; Metal pads no rows), so
            // flops over time reaches the rate where the weights no longer
            // bind it and stays under it everywhere: the rate is the most
            // any of the rows reads.
            let mut most = 0f64;
            for m in FEW_ROWS {
                let rate = bench.matmul_flops(m, w, w, &tensors, &touched, gap)?;
                say(&format!(
                    "  bf16 matmul {m} x {w} x {w} (few rows): {:.4} TFLOP/s",
                    rate / 1e3
                ));
                most = most.max(rate);
            }
            few_row = Some(most / 1e3);
            // `workgroups_per_core`: the curve's first rows (its kernel at a
            // compute-bound depth, `k = w`) over as few columns as launch at
            // most one workgroup a core, timed against the same work at the
            // rate the curve states there (`engine::rates::workgroups_per_core`).
            let cores = device
                .cores()
                .ok_or("the device states no core count, so no workgroups a core")?;
            let m = engine::rates::GEMM_ROWS[0];
            let full = curve
                .iter()
                .find(|&&(rows, _)| rows == m)
                .map(|&(_, tflops)| tflops * 1e3)
                .ok_or("the curve has no rate at its first rows")?;
            let mut n = w;
            let mut groups = bench.matmul_groups(m, n, w, tensors[0])?;
            while groups > u64::from(cores) && n > 1 {
                n /= 2;
                groups = bench.matmul_groups(m, n, w, tensors[0])?;
            }
            if groups == 0 || groups > u64::from(cores) {
                return Err(format!(
                    "a {m} x {n} x {w} matmul launches {groups} workgroups over {cores} cores: \
                     no shape of it launches one a core or fewer"
                ));
            }
            let narrow: Vec<Tensor> = weights
                .iter()
                .map(|(_, b)| bench.tensor(b, n, w, Dtype::Bf16))
                .collect::<Result<_, String>>()?;
            let rate = bench.matmul_flops(m, n, w, &narrow, &touched, gap)?;
            let flops = engine::rates::gemm_flops(m, n, w);
            let groups = u32::try_from(groups).map_err(|e| e.to_string())?;
            per_core = engine::rates::workgroups_per_core(groups, cores, flops / rate, flops / full);
            say(&format!(
                "  bf16 matmul {m} x {n} x {w}, {groups} workgroups over {cores} cores: \
                 {:.4} TFLOP/s against {:.4} at full width, {:.2} workgroups a core",
                rate / 1e3,
                full / 1e3,
                per_core.unwrap_or(f64::NAN)
            ));
        }
        gemm.push((w, curve));
    }
    // The attention curve at d 128; the other widths at the longest keys.
    let fired_gemm = crate::encode::record_fired(true);
    let mut attn = Vec::new();
    for keys in engine::rates::ATTN_KEYS {
        bench.handles = Handles::new();
        let rate = bench.attention_flops(engine::rates::ATTN_SHAPE.at(keys), gap)?;
        say(&format!(
            "  prefill attention, {keys} rows over {keys} keys, d 128: {:.4} TFLOP/s",
            rate / 1e3
        ));
        attn.push((keys, rate / 1e3));
    }
    let longest = engine::rates::ATTN_KEYS[engine::rates::ATTN_KEYS.len() - 1];
    for d in engine::rates::ATTN_DIMS.into_iter().filter(|&d| d != 128) {
        bench.handles = Handles::new();
        let rate = bench.attention_flops(engine::rates::ATTN_SHAPE.at(longest).with_dim(d), gap)?;
        say(&format!(
            "  prefill attention, {longest} rows over {longest} keys, d {d}: {:.4} TFLOP/s",
            rate / 1e3
        ));
    }
    let fired_attn = crate::encode::record_fired(true);
    bench.handles = Handles::new();
    let elementwise = bench.elementwise_bytes(gap)?;
    let fired_elementwise = crate::encode::record_fired(true);
    say(&format!("  rmsnorm and swiglu: {elementwise:.1} GB/s"));
    bench.handles = Handles::new();
    let scan = bench.scan_elements(SCAN_SHAPE.tokens, gap)?;
    let fired_scan = crate::encode::record_fired(true);
    say(&format!(
        "  gated-delta scan and its conv: {scan:.1} state elements a ns"
    ));
    let bytes = bench.bytes_per_ns(gap)?;
    let fired_blit = crate::encode::record_fired(false);
    say(&format!("  256 MiB blit: {bytes:.1} GB/s read and write"));
    Ok(Rates {
        landing: None,
        load_width: None,
        stale: None,
        launch_ns: Some(launch),
        gemm_tflops_by_rows: Some(gemm),
        attn_tflops_by_keys: Some(attn),
        hbm_gbps: Some(bytes),
        cached_gbps: None,
        elementwise_gbps: Some(elementwise),
        scan_gelems: Some(scan),
        few_row_tflops: few_row,
        workgroups_per_core: per_core,
        submit_worst_ratio: None,
        resident_workgroups: None,
        attn_tflops_by_rows_keys: None,
        product_ns: None,
        kernel_tflops_by_shape: None,
        dense_settled: Vec::new(),
        measured_on: Some(crate::stored::measured_on(device.name())),
        measured_by: Some(
            [
                ("launch_ns", &fired_launch),
                ("gemm_tflops_by_rows", &fired_gemm),
                ("few_row_tflops", &fired_gemm),
                ("workgroups_per_core", &fired_gemm),
                ("attn_tflops_by_keys", &fired_attn),
                ("elementwise_gbps", &fired_elementwise),
                ("scan_gelems", &fired_scan),
                ("hbm_gbps", &fired_blit),
            ]
            .into_iter()
            .map(|(what, fired)| (what.to_string(), crate::stored::by(what, fired)))
            .collect(),
        ),
    })
}

/// The decode-shaped rows `few_row_tflops` is timed at: 4 to 256, the rows
/// below the matmul curve's first.
const FEW_ROWS: [u32; 7] = [4, 8, 16, 32, 64, 128, 256];

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
