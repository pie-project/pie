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
//!   `engine::rates::LATTICE_ROWS` and each depth of `engine::rates::DEPTHS`,
//!   `m x k x k`, for the matmul grid;
//! - its prefill attention at `engine::rates::ATTN_SHAPE`'s heads over each
//!   `engine::rates::ATTN_KEYS`, for the attention curve;
//! - its chunked gated-delta scan, the causal conv before it, at
//!   `engine::rates::SCAN_SHAPE`, for the scan rate;
//! - a buffer copy of [`COPY_BYTES`], for bytes a nanosecond;
//! - [`LAUNCHES`] of the narrowest matmul back to back in one frame, against
//!   one alone, for what each dispatch inside a frame costs beside its work:
//!   `(t(LAUNCHES) − t(1)) / (LAUNCHES − 1)`.
//!
//! The backend times a frame ([`Timer`]): by its adapter's timestamps where
//! it has them, else by the host's clock around submit and wait. The
//! repeated fires are the host's clock's difference of two frames either
//! way: one submit each, so neither the clock nor the adapter's span start
//! biases it.

use engine::rates::{
    ATTN_KEYS, ATTN_SHAPE, DEPTHS, LATTICE_ROWS, SCAN_SHAPE, attn_flops, gemm_flops, scan_elements,
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

/// The launches one frame of [`LAUNCH`] matmuls carries: enough that the
/// frame's submit and wait (tens of microseconds) amortise to well under a
/// dispatch's cost (fractions of a microsecond to a few).
pub const LAUNCHES: u32 = 1024;

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
}

impl<'a, B: Serve> Rig<'a, B> {
    fn new(device: &'a B::Context) -> Self {
        Rig {
            device,
            handles: Handles::<B>::new(),
            pipelines: B::Pipelines::default(),
            held: Vec::new(),
        }
    }

    /// Every buffer laid so far, let go (their handles with them).
    fn release(&mut self) {
        self.handles = Handles::<B>::new();
        self.held.clear();
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
    let mut rig = Rig::<B>::new(device);
    let grid = grid(&mut rig, tactics.gemm(tier), timer)?;
    let sdpa = tactics.sdpa(tier, ATTN_SHAPE.head_dim);
    let mut attn = Vec::with_capacity(ATTN_KEYS.len());
    for keys in ATTN_KEYS {
        attn.push((keys, attention(&mut rig, sdpa, timer, keys)?));
        rig.release();
    }
    let scan = scan(&mut rig, tactics.recurrent(), timer)?;
    let (bytes_per_ns, launch_ns) = copy_and_launch(&mut rig, tactics.gemm(tier), timer)?;
    Ok(Rates {
        bf16_flops_per_ns_by_rows_and_depth: Some(grid),
        attn_flops_per_ns_by_keys: Some(attn),
        scan_elements_per_ns: Some(scan),
        bytes_per_ns: Some(bytes_per_ns),
        launch_ns: Some(launch_ns),
    })
}

/// The matmul rate at each row count and depth of the lattice: `(m, k,
/// flops a ns)`, `m x k x k`, by depth then rows.
fn grid<B: Serve>(
    rig: &mut Rig<'_, B>,
    tactic: kernels_slang::linear::gemm::Tactic,
    timer: Timer<'_, B>,
) -> Result<Vec<(u32, u32, f64)>> {
    let widest = LATTICE_ROWS[LATTICE_ROWS.len() - 1];
    let mut out = Vec::with_capacity(LATTICE_ROWS.len() * DEPTHS.len());
    for k in DEPTHS {
        let act = rig.zeros(widest, k, Dtype::Bf16)?;
        let w = rig.zeros(k, k, Dtype::Bf16)?;
        let y = rig.zeros(widest, k, Dtype::Bf16)?;
        for m in LATTICE_ROWS {
            let ns = rig.each(timer, &|sink| {
                kernels_slang::linear::gemm::matmul(
                    sink,
                    tactic,
                    Tensor { rows: m, ..act },
                    w,
                    Tensor { rows: m, ..y },
                )
                .map_err(fault("rates: matmul"))
            })?;
            out.push((m, k, gemm_flops(m, k) / ns as f64));
        }
        // This depth's planes go before the next depth's are laid.
        rig.release();
    }
    Ok(out)
}

/// The prefill attention's rate at [`ATTN_SHAPE`]'s heads over `keys`: one
/// sequence of `keys` rows attending itself, causal, its kv laid in pages of
/// [`PAGE`].
fn attention<B: Serve>(
    rig: &mut Rig<'_, B>,
    tactic: kernels_slang::attn::sdpa::Tactic,
    timer: Timer<'_, B>,
    keys: u32,
) -> Result<f64> {
    let s = ATTN_SHAPE.at(keys);
    let kv_width = s.kv_heads * s.head_dim;
    let q = rig.zeros(s.rows, s.q_heads * s.head_dim, Dtype::Bf16)?;
    let o = rig.zeros(s.rows, s.q_heads * s.head_dim, Dtype::Bf16)?;
    let keys = rig.zeros(s.keys, kv_width, Dtype::Bf16)?;
    let values = rig.zeros(s.keys, kv_width, Dtype::Bf16)?;
    let pages = s.keys.div_ceil(PAGE);
    let indices: Vec<u32> = (0..pages).collect();
    let page_indices = rig.words(&indices, pages, 1, Dtype::U32)?;
    let page_indptr = rig.words(&[0, pages], 2, 1, Dtype::U32)?;
    let pool = KvPool::planes(
        keys,
        values,
        page_indices,
        page_indptr,
        PAGE as i32,
        u64::from(kv_width),
        u64::from(s.head_dim),
    );
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
        kernels_slang::attn::sdpa::prefill(
            sink, tactic, q, &tables, &pool, None, s.head_dim, scale, o, None,
        )
        .map_err(fault("rates: attention"))
    })?;
    Ok(attn_flops(s) / ns as f64)
}

/// The recurrent rate at [`SCAN_SHAPE`]: the causal conv over the scan's
/// `qkv` channels, then the chunked gated-delta scan, one sequence on one
/// state slot.
fn scan<B: Serve>(
    rig: &mut Rig<'_, B>,
    tactic: kernels_slang::ssm::gdn_recurrent::Tactic,
    timer: Timer<'_, B>,
) -> Result<f64> {
    let s = SCAN_SHAPE;
    let channels = s.channels();
    let raw = rig.zeros(s.tokens, channels, Dtype::Bf16)?;
    let qkv = rig.zeros(s.tokens, channels, Dtype::Bf16)?;
    let indptr = rig.words(&[0, s.tokens], 2, 1, Dtype::I32)?;
    let weight = rig.zeros(channels, s.conv_width, Dtype::Bf16)?;
    let slots = rig.zeros(s.tokens, 1, Dtype::U32)?;
    let conv_bank = rig.zeros(1, s.conv_width * channels, Dtype::F32)?;
    let scan_bank = rig.zeros(1, s.v_heads * s.k_dim * s.v_dim, Dtype::F32)?;
    let gates = rig.zeros(s.tokens, 2 * s.v_heads, Dtype::F32)?;
    let y = rig.zeros(s.tokens, s.v_heads * s.v_dim, Dtype::F32)?;
    let conv_state = kernels_slang::ssm::StatePool::bank(conv_bank, slots);
    let scan_state = kernels_slang::ssm::StatePool::bank(scan_bank, slots);
    let ns = rig.each(timer, &|sink| {
        kernels_slang::ssm::conv1d::conv1d_chunked(
            sink,
            RaggedTensor { data: raw, indptr },
            weight,
            &conv_state,
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
            &scan_state,
            s.k_heads,
            s.v_heads,
            s.k_dim,
            s.v_dim,
            y,
        )
        .map_err(fault("rates: scan"))
    })?;
    Ok(scan_elements(s) / ns as f64)
}

/// `(bytes a ns, a dispatch's ns)`: a copy of device memory, and the
/// narrowest matmul back to back in one frame against one alone.
fn copy_and_launch<B: Serve>(
    rig: &mut Rig<'_, B>,
    tactic: kernels_slang::linear::gemm::Tactic,
    timer: Timer<'_, B>,
) -> Result<(f64, u64)> {
    let act = rig.zeros(LAUNCH, LAUNCH, Dtype::Bf16)?;
    let w = rig.zeros(LAUNCH, LAUNCH, Dtype::Bf16)?;
    let y = rig.zeros(LAUNCH, LAUNCH, Dtype::Bf16)?;
    let tiny = |sink: &kernels_slang::Ctx<'_>| {
        kernels_slang::linear::gemm::matmul(sink, tactic, act, w, y).map_err(fault("rates: matmul"))
    };
    let nothing = |_: &mut B::Frame| Ok(());
    rig.frame(timer, 1, &tiny, &nothing)?;
    rig.frame(timer, LAUNCHES, &tiny, &nothing)?;
    let wall = |count: u32| -> Result<u64> {
        Ok(median(
            (0..ROUNDS)
                .map(|_| rig.frame(timer, count, &tiny, &nothing).map(|(_, w)| w))
                .collect::<Result<Vec<u64>>>()?,
        ))
    };
    let single = wall(1)?;
    let launch = wall(LAUNCHES)?.saturating_sub(single) / u64::from(LAUNCHES - 1);

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
    Ok((2.0 * copy as f64 / copy_ns as f64, launch))
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
    let act = rig.zeros(m, k, Dtype::Bf16)?;
    let w = rig.zeros(n, k, Dtype::Bf16)?;
    let y = rig.zeros(m, n, Dtype::Bf16)?;
    let ns = rig.each(timer, &|sink| {
        kernels_slang::linear::gemm::matmul(sink, tactic, act, w, y).map_err(fault("rates: matmul"))
    })?;
    Ok(2.0 * f64::from(m) * f64::from(n) * f64::from(k) / ns as f64)
}
