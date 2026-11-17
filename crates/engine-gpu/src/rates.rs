//! A device's rates measured on the device itself: micro-fires on synthetic
//! buffers, before any model is compiled, at the shapes `engine::rates`
//! states for every backend. One function every slang backend calls: a
//! browser's load (its adapter is masked, so no profile states its rates)
//! and `pie tune --rates` natively.
//!
//! Each kind of work is timed inside a frame: [`REPEATS`] of it back to back
//! against one, the median of [`ROUNDS`] after a warm frame (which builds
//! its pipelines), so the frame's one submit and wait cancel and what is
//! left is the work's own time beside a dispatch's:
//! - the backend's own bf16 matmul at each row count of
//!   the backend's rung lattice (`Serve::rung_lattice`, the rows its loads
//!   lay fires at) and each column count and depth of
//!   `engine::rates::DEPTHS`, `m x n x k`, each fire reading the next of
//!   `engine::rates::weights_at(n, k)`
//!   weight planes (past the last-level cache), for the matmul grid;
//! - its prefill attention at `engine::rates::ATTN_SHAPE`'s heads over each
//!   `engine::rates::ATTN_KEYS` and head width of `engine::rates::ATTN_DIMS`,
//!   through the kernel a served fire selects at that shape, for the
//!   attention grid;
//! - its chunked gated-delta scan, the causal conv before it, at
//!   `engine::rates::SCAN_SHAPE`, for the scan rate;
//! - a buffer copy of [`COPY_BYTES`], for bytes a nanosecond;
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

use engine::rates::{
    ATTN_DIMS, ATTN_KEYS, ATTN_SHAPE, DEPTHS, SCAN_SHAPE, attn_flops, gemm_flops, scan_elements,
    weights_at,
};
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

/// The matmul a launch is timed with: its work is nothing beside a launch.
pub const LAUNCH: u32 = 16;

/// The fires a timed frame carries back to back, against a frame of one.
pub const REPEATS: u32 = 16;

/// Timed frames of each kind; the median is kept.
pub const ROUNDS: usize = 5;

/// The kv page the attention's pool is laid in.
const PAGE: u32 = 16;

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
}

impl<'a, B: Serve> Rig<'a, B> {
    fn new(device: &'a B::Context) -> Self {
        Rig {
            device,
            handles: Handles::<B>::new(),
            pipelines: B::Pipelines::default(),
            held: Vec::new(),
            pattern: None,
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
        let mut f = self.device.frame()?;
        {
            let sink = B::sink(self.device, &f, &self.pipelines, &self.handles);
            for _ in 0..count {
                fire(&sink)?;
            }
        }
        then(&mut f)?;
        timer(f)
    }

    /// One `fire`'s time inside a frame: [`REPEATS`] of it against one, the
    /// host's clock around each frame's one submit.
    fn each(
        &self,
        timer: Timer<'_, B>,
        fire: &dyn Fn(&kernels_slang::Ctx<'_>) -> Result<()>,
    ) -> Result<u64> {
        let nothing = |_: &mut B::Frame| Ok(());
        let wall = |count: u32| -> Result<u64> {
            let walls = (0..ROUNDS)
                .map(|_| self.frame(timer, count, fire, &nothing).map(|(_, w)| w))
                .collect::<Result<Vec<u64>>>()?;
            Ok(median(walls))
        };
        self.frame(timer, 1, fire, &nothing)?;
        self.frame(timer, REPEATS, fire, &nothing)?;
        let (one, many) = (wall(1)?, wall(REPEATS)?);
        Ok((many.saturating_sub(one) / u64::from(REPEATS - 1)).max(1))
    }
}

fn median(mut v: Vec<u64>) -> u64 {
    v.sort_unstable();
    v[v.len() / 2].max(1)
}

/// The rates of `device`, its kernels under the tactics `tactics` picks at
/// `tier`.
pub fn measure<B: Serve, T: Tactics + ?Sized>(
    device: &B::Context,
    tactics: &T,
    tier: kernels_slang::Tier,
    timer: Timer<'_, B>,
) -> Result<Rates> {
    measure_saying::<B, T>(device, tactics, tier, timer, &mut |_| {})
}

/// [`measure`], saying how the warm-up ended.
pub fn measure_saying<B: Serve, T: Tactics + ?Sized>(
    device: &B::Context,
    tactics: &T,
    tier: kernels_slang::Tier,
    timer: Timer<'_, B>,
    say: &mut dyn FnMut(&str),
) -> Result<Rates> {
    let mut rig = Rig::<B>::new(device);
    // The launch first: every rate below takes its dispatches' launches off.
    let (bytes_per_ns, launch_ns) = copy_and_launch(&mut rig, tactics.gemm(tier), timer, say)?;
    rig.release();
    let launch = launch_ns as f64;
    let warmed = warm(&mut rig, tactics.gemm(tier), timer)?;
    say(&format!(
        "[rates]: warm-up: {} rotations of the {}^3 matmul, settled at {:.1} TFLOP/s",
        warmed.rotations,
        engine::rates::WARM_SHAPE.0,
        warmed.rate / 1e3
    ));
    let rows = B::rung_lattice();
    let (grid, top) = grid(&mut rig, tactics.gemm(tier), timer, launch, &rows)?;
    say(&format!(
        "[rates]: the grid saturates at {} rows of a lattice to {}: past it no (n, k) shape's \
         cost a row fell more than {:.0}% below its least through {} (at it the most {:.1}%)",
        top.rows,
        rows.last().copied().unwrap_or(0),
        100.0 * engine::rates::AGREEMENT,
        top.through,
        100.0 * top.best_gain,
    ));
    let mut attn = Vec::with_capacity(ATTN_KEYS.len() * ATTN_DIMS.len());
    for d in ATTN_DIMS {
        for keys in ATTN_KEYS {
            let shape = ATTN_SHAPE.at(keys).with_dim(d);
            attn.push((
                keys,
                d,
                attention(&mut rig, tactics, tier, timer, shape, launch)?,
            ));
            rig.release();
        }
    }
    let mut scan_curve = Vec::with_capacity(engine::rates::SCAN_TOKENS.len());
    for tokens in engine::rates::SCAN_TOKENS {
        let shape = SCAN_SHAPE.at(tokens);
        scan_curve.push((
            tokens,
            scan(&mut rig, tactics.recurrent(), timer, shape, launch)?,
        ));
        rig.release();
    }
    Ok(Rates {
        bf16_flops_per_ns_by_rows_n_k: Some(grid),
        attn_flops_per_ns_by_keys_and_dim: Some(attn),
        scan_elements_per_ns_by_tokens: Some(scan_curve),
        bytes_per_ns: Some(bytes_per_ns),
        launch_ns: Some(launch_ns),
    })
}

/// The device warmed ([`engine::rates::warm_up`]): rotations of
/// [`engine::rates::WARM_SHAPE`] over its planes, each one frame timed by
/// the host's clock around its one submit, until the rate stops falling.
fn warm<B: Serve>(
    rig: &mut Rig<'_, B>,
    tactic: kernels_slang::linear::gemm::Tactic,
    timer: Timer<'_, B>,
) -> Result<engine::rates::Warmed> {
    let (m, n, k) = engine::rates::WARM_SHAPE;
    let act = rig.operand(m, k, Dtype::Bf16)?;
    let weights = (0..weights_at(n, k))
        .map(|_| rig.operand(n, k, Dtype::Bf16))
        .collect::<Result<Vec<Tensor>>>()?;
    let y = rig.zeros(m, n, Dtype::Bf16)?;
    let next = std::cell::Cell::new(0usize);
    let fire = |sink: &kernels_slang::Ctx<'_>| {
        let w = weights[next.get() % weights.len()];
        next.set(next.get() + 1);
        kernels_slang::linear::gemm::matmul(sink, tactic, act, w, y).map_err(fault("rates: matmul"))
    };
    let planes = u32::try_from(weights.len()).unwrap_or(u32::MAX);
    let nothing = |_: &mut B::Frame| Ok(());
    let warmed = engine::rates::warm_up(|| {
        let (_, wall) = rig.frame(timer, planes, &fire, &nothing)?;
        Ok::<f64, Fault>(gemm_flops(m, n, k) * f64::from(planes) / wall.max(1) as f64)
    })?;
    rig.release();
    Ok(warmed)
}

/// The matmul rate at each row count, column count and depth of the
/// lattice: `(m, n, k, flops a ns)`, ordered by depth, then columns, then
/// rows. The shapes are timed in [`engine::rates::spread_order`], and each
/// shape's rows in it too, so a drift left after the warm-up spreads over
/// the grid.
fn grid<B: Serve>(
    rig: &mut Rig<'_, B>,
    tactic: kernels_slang::linear::gemm::Tactic,
    timer: Timer<'_, B>,
    launch: f64,
    lattice: &[u32],
) -> Result<(Vec<(u32, u32, u32, f64)>, engine::rates::Saturation)> {
    let widest = lattice.iter().copied().max().unwrap_or(1);
    let shapes: Vec<(u32, u32)> = DEPTHS
        .iter()
        .flat_map(|&k| DEPTHS.iter().map(move |&n| (n, k)))
        .collect();
    // A shape's operands, laid at the start of each run of it: the widest
    // row's activation and output, and the weight planes each fire reads
    // the next of, past the last-level cache.
    let mut laid: Option<(Tensor, Vec<Tensor>, Tensor)> = None;
    let next = std::cell::Cell::new(0usize);
    let (rates, top) = engine::rates::saturate(lattice, shapes.len(), |at, row, fresh| {
        let (n, k) = shapes[at];
        if fresh {
            // The last shape's planes go before this shape's are laid.
            laid = None;
            rig.release();
            let act = rig.operand(widest, k, Dtype::Bf16)?;
            let weights = (0..weights_at(n, k))
                .map(|_| rig.operand(n, k, Dtype::Bf16))
                .collect::<Result<Vec<Tensor>>>()?;
            let y = rig.zeros(widest, n, Dtype::Bf16)?;
            laid = Some((act, weights, y));
        }
        let (act, weights, y) = laid.as_ref().expect("laid");
        let m = lattice[row];
        let ns = rig.each(timer, &|sink| {
            let w = weights[next.get() % weights.len()];
            next.set(next.get() + 1);
            kernels_slang::linear::gemm::matmul(
                sink,
                tactic,
                Tensor { rows: m, ..*act },
                w,
                Tensor { rows: m, ..*y },
            )
            .map_err(fault("rates: matmul"))
        })?;
        let rate = engine::rates::rate_from(
            &format!("matmul {m}x{n}x{k}"),
            ns as f64,
            engine::rates::GEMM_DISPATCHES,
            launch,
            gemm_flops(m, n, k),
        )
        .map_err(|why| Fault::Device { call: "rates", why })?;
        Ok::<_, Fault>((
            rate,
            engine::rates::gemm_cost_per_row(m, n, k, rate, launch),
        ))
    })?;
    rig.release();
    // The grid ends at the saturation point: a load's rungs run to it.
    let mut out = Vec::with_capacity((top.top + 1) * shapes.len());
    for (at, &(n, k)) in shapes.iter().enumerate() {
        for (i, &m) in lattice.iter().enumerate().take(top.top + 1) {
            out.push((m, n, k, rates[at][i]));
        }
    }
    out.sort_by_key(|&(m, n, k, _)| (k, n, m));
    Ok((out, top))
}

/// The prefill attention's rate at `s`: one sequence of `s.rows` rows
/// attending itself, causal, its kv laid in pages of [`PAGE`], through the
/// kernel a served fire takes at that shape (the tactic at its head width,
/// tiled or walked by [`Tactics::sdpa_tiles`]).
fn attention<B: Serve, T: Tactics + ?Sized>(
    rig: &mut Rig<'_, B>,
    tactics: &T,
    tier: kernels_slang::Tier,
    timer: Timer<'_, B>,
    s: engine::rates::AttnShape,
    launch: f64,
) -> Result<f64> {
    let tactic = tactics.sdpa(tier, s.head_dim);
    let tiled = tactics.sdpa_tiles(s.rows);
    let kv_width = s.kv_heads * s.head_dim;
    let o = rig.zeros(s.rows, s.q_heads * s.head_dim, Dtype::Bf16)?;
    let pages = s.keys.div_ceil(PAGE);
    // The sequence's pages scattered over the pool as a served one's are.
    let indices: Vec<u32> = engine::rates::attn_page_order(pages);
    let page_indices = rig.words(&indices, pages, 1, Dtype::U32)?;
    let page_indptr = rig.words(&[0, pages], 2, 1, Dtype::U32)?;
    // Each fire reads the next of these q / k / v sets, past the last-level
    // cache, as a served fire reads each layer's kv cold.
    let sets = (0..engine::rates::attn_sets_at(s))
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
    const MASK_STRIDE: u32 = 4;
    let tables = kernels_slang::attn::sdpa::Tables {
        positions: rig.words(&positions, s.rows, 1, Dtype::I32)?,
        request_of_token: rig.zeros(s.rows, 1, Dtype::I32)?,
        mask: rig.zeros(s.rows, MASK_STRIDE, Dtype::U8)?,
        mask_enabled: rig.zeros(s.rows, 1, Dtype::U8)?,
        mask_stride: MASK_STRIDE,
    };
    let scale = 1.0 / (s.head_dim as f32).sqrt();
    let ns = rig.each(timer, &|sink| {
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
    let ns = rig.each(timer, &|sink| {
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
        let chain = wall(n, &chained)?;
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
    let (with_copy, by_adapter) = timed(&|f| f.copy(&from, 0, &into, 0, copy))?;
    let alone = if by_adapter {
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
    let mut rig = Rig::<B>::new(device);
    attention(&mut rig, tactics, tier, timer, shape, launch)
}

/// The bf16 matmul rate of `device` at `(m, n, k)` under `tactic`: a check
/// of the estimate times the model's own planes with it.
pub fn gemm_rate<B: Serve>(
    device: &B::Context,
    tactic: kernels_slang::linear::gemm::Tactic,
    timer: Timer<'_, B>,
    (m, n, k): (u32, u32, u32),
) -> Result<f64> {
    let mut rig = Rig::<B>::new(device);
    let act = rig.operand(m, k, Dtype::Bf16)?;
    let w = rig.operand(n, k, Dtype::Bf16)?;
    let y = rig.zeros(m, n, Dtype::Bf16)?;
    let ns = rig.each(timer, &|sink| {
        kernels_slang::linear::gemm::matmul(sink, tactic, act, w, y).map_err(fault("rates: matmul"))
    })?;
    Ok(2.0 * f64::from(m) * f64::from(n) * f64::from(k) / ns as f64)
}
