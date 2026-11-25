//! What every backend's `pie tune <backend> --rates` (or a load on a
//! device whose profile lacks them) times, and how it counts the work: a few
//! shapes, model-independent, whose times give the device's rates
//! (`profile::rates::Rates`), so a profile's `[rates]` mean the same on every
//! backend and the estimate (`crate::load::FireEstimate`, over
//! `crate::roofline`) reads them the same way. Each backend times these
//! shapes through its own kernels and its own clock; the counts here turn
//! its times into rates, and [`tuned`] and [`provenance`] write them. Every
//! timed submission is a burst of one kernel sized off a first single
//! dispatch's time, so none runs long.
//!
//! - `launch_ns`: what a dispatch adds to its work in a chain of dependent
//!   ones submitted as serving submits them, at each `n` of
//!   [`LAUNCH_CHAINS`] ([`launch_from`]).
//! - `gemm_tflops_by_rows`: a bf16 matmul of each of [`GEMM_WIDTHS`]
//!   columns and depth at each of [`GEMM_ROWS`], [`gemm_flops`] over its
//!   time.
//! - `attn_tflops_by_keys`: the backend's prefill attention at
//!   [`ATTN_SHAPE`]'s heads over each of [`ATTN_KEYS`] keys (one sequence
//!   attending itself, its pages laid in [`attn_page_order`]),
//!   [`attn_flops`] over its time.
//! - `hbm_gbps`: a copy of device memory, every byte read and written.
//! - `elementwise_gbps`: the per-row ops a fire runs between its matmuls, a
//!   norm and the gated activation at [`ELEMENTWISE_SHAPE`], their bytes
//!   ([`elementwise_bytes`]) over their time together.
//! - `scan_gelems`: the prefill gated-delta scan and the causal conv before
//!   it at [`SCAN_SHAPE`], [`scan_elements`] over the two together's time.
//! - `few_row_tflops`: a bf16 matmul at decode-shaped rows, 4 to 256, on
//!   the kernel serving picks at those rows (below [`GEMM_ROWS`]' first, so
//!   the curve never weighs it), its flops over its time once its weight
//!   reads no longer bind it. Measured on every backend.
//! - `kernel_curves`: each dense matmul kernel the slang backends
//!   choose between (the vector kernels by the rows a workgroup carries,
//!   the tiled by their row tile) at each of a model's dense shapes, by
//!   rows, its flops over its time, its weight passes in it; measured once
//!   the model's shapes are known, not one of the estimate's keys.
//! - `workgroups_per_core`: the workgroups a compute unit holds at once
//!   for a kernel's full rate. Derived where the device states its
//!   occupancy (CUDA, ROCm: the resident blocks a multiprocessor);
//!   measured where its API hides it (Vulkan, wgpu, Metal): one workgroup
//!   a head of a kernel at its compute-bound shape, its time against the
//!   same work at the rate its full curve states ([`workgroups_per_core`]).
//!
//! Each backend's estimate weighs a fire by the keys it names
//! ([`rates_at_load`]'s `keys`, of `profile::rates::KEYS`); a rate measured
//! or stated at or below zero, or not finite, is refused by its key
//! ([`unmeasured`]), never saved or divided by.

/// The bytes the rates step lays on the device at once, at most: the
/// largest of a matmul's rotation and operands, the attention's cold sets,
/// the per-row ops' sets, the scan's cold sets and the copy's two buffers.
/// What the step's device binds, so it needs no load budget.
#[must_use]
pub fn step_bytes() -> u64 {
    let bf16 = |elements: u64| elements * 2;
    let (wm, wn, wk) = WARM_SHAPE;
    let warm = bf16(u64::from(wm) * u64::from(wk))
        + weights_at(wn, wk) as u64 * bf16(u64::from(wn) * u64::from(wk))
        + bf16(u64::from(wm) * u64::from(wn));
    let w = u64::from(GEMM_WIDTH);
    let widest = u64::from(GEMM_ROWS[GEMM_ROWS.len() - 1]);
    let gemm = bf16(widest * w)
        + weights_at(GEMM_WIDTH, GEMM_WIDTH) as u64 * bf16(w * w)
        + bf16(widest * w);
    let attn = {
        let s = ATTN_SHAPE.at(ATTN_KEYS[ATTN_KEYS.len() - 1]);
        let q = u64::from(s.rows) * u64::from(s.q_heads * s.head_dim);
        let kv = 2 * u64::from(s.keys.div_ceil(16) * 16) * u64::from(s.kv_heads * s.head_dim);
        ATTN_SET_BYTES as u64 + bf16(q + kv) + bf16(q)
    };
    let scan = {
        let s = SCAN_SHAPE;
        let rows = u64::from(s.tokens) * u64::from(s.channels());
        let state = u64::from(s.conv_width * s.channels() + s.v_heads * s.k_dim * s.v_dim);
        scan_sets_at(s) as u64 * (bf16(rows) + 4 * state) + bf16(rows) * 2
    };
    let elementwise = (elementwise_sets() * elementwise_set_bytes()) as u64;
    let copy = 2 * (256u64 << 20);
    warm.max(gemm)
        .max(attn)
        .max(scan)
        .max(elementwise)
        .max(copy)
}

/// Timed bursts of each kind a rates step measures: the median kept, an
/// odd count so it is one burst's.
pub const ROUNDS: usize = 5;

/// The columns and depth the matmul curves are timed at: the catalog's
/// hidden widths from a small model's 1024 to an 8B-class model's 4096 (a
/// plane is read at its square width between them, clamped past them).
pub const GEMM_WIDTHS: [u32; 3] = [1024, 2048, 4096];

/// The widest of [`GEMM_WIDTHS`].
pub const GEMM_WIDTH: u32 = GEMM_WIDTHS[GEMM_WIDTHS.len() - 1];

/// The rows the matmul curve is timed at: half octaves (steps of √2,
/// rounded) from 256 to 8192 rows, the reach of a fire's prompt. An
/// octave-spaced curve can sit in a library's dips between octaves (an
/// MI300X's hipBLASLt at width 4096: 523 TFLOP/s at 2048 rows, 599 at
/// 2896, 556 at 4096), so the shapes between them would be priced from
/// its dips.
pub const GEMM_ROWS: [u32; 11] = half_octaves(256);

/// `N` rows at steps of √2 from `from`, rounded to the nearest row: the
/// even steps `from` doubled, the odd ones those times √2.
const fn half_octaves<const N: usize>(from: u32) -> [u32; N] {
    let mut rows = [0; N];
    let mut i = 0;
    while i < N {
        let octave = from << (i / 2);
        rows[i] = if i % 2 == 0 {
            octave
        } else {
            (octave as f64 * std::f64::consts::SQRT_2 + 0.5) as u32
        };
        i += 1;
    }
    rows
}

/// The keys (and rows) the attention curve is timed at.
pub const ATTN_KEYS: [u32; 5] = [512, 1024, 2048, 4096, 8192];

/// `workgroups_per_core` from one timed dispatch: `workgroups` of a kernel
/// over `cores` compute units took `ns`, and the same work at the kernel's
/// full rate (its work over the rate its full curve states there) takes
/// `full_ns`. The dispatch ran at `workgroups / S` of its rate, so the
/// device fills at `S = workgroups · ns / full_ns`, over `cores`. `None`
/// where a time or the cores are nothing: no rate is made of them.
#[must_use]
pub fn workgroups_per_core(workgroups: u32, cores: u32, ns: f64, full_ns: f64) -> Option<f64> {
    let per = f64::from(workgroups) * ns / full_ns / f64::from(cores);
    (per.is_finite() && per > 0.0).then_some(per)
}

/// The per-row ops' shape: `(rows, width)`, a 2048-token fire of a
/// 4096-wide model; the norm reads and writes rows of `width`, the gated
/// activation reads `2 · width` and writes `width`.
pub const ELEMENTWISE_SHAPE: (u32, u32) = (2048, 4096);

/// The bytes the per-row ops move at [`ELEMENTWISE_SHAPE`]: the norm's row in
/// and out, then the gated activation's two halves in and its row out.
#[must_use]
pub fn elementwise_bytes() -> f64 {
    let (rows, width) = ELEMENTWISE_SHAPE;
    let row = 2.0 * f64::from(rows) * f64::from(width);
    2.0 * row + 3.0 * row
}

/// One set of the per-row ops' operands (bf16): the norm's in and out, the
/// activation's packed halves and its out.
#[must_use]
pub fn elementwise_set_bytes() -> usize {
    let (rows, width) = ELEMENTWISE_SHAPE;
    rows as usize * width as usize * 2 * 5
}

/// The per-row ops' sets a timed burst rotates through: [`ATTN_SET_BYTES`]
/// of them, at least 2, so each reads its rows from device memory.
#[must_use]
pub fn elementwise_sets() -> usize {
    ATTN_SET_BYTES.div_ceil(elementwise_set_bytes()).max(2)
}

/// The depths of the dense layers served, from a small model's 1024
/// (Qwen3-0.6B's hidden) through 4096 (Qwen3-8B's) and 8192 (a 30B-class
/// hidden) to 16384.
pub const DEPTHS: [u32; 5] = [1024, 2048, 4096, 8192, 16384];

/// The flops of one matmul of `rows`, `n` columns and depth `k`: two a
/// multiply-add.
#[must_use]
pub fn gemm_flops(rows: u32, n: u32, k: u32) -> f64 {
    2.0 * f64::from(rows) * f64::from(n) * f64::from(k)
}

/// The bytes of weight planes a matmul's timed fires rotate through at
/// least: three times the largest last-level cache of a device served (an
/// MI300X's 256 MiB Infinity Cache; an L40S holds 96 MiB, an RTX PRO 4500
/// 64 MiB). Twelve 2 MiB planes of depth 1024 sit in an L40S's L2 whole, and
/// its rates there moved 4% run to run.
pub const WEIGHT_BYTES: usize = 768 << 20;

/// The weight planes the timed matmuls of `n` columns at depth `k` rotate
/// through: [`WEIGHT_BYTES`] of `n x k` bf16 planes, and never fewer than
/// 12, so each reads its weight from device memory as a layer's does.
#[must_use]
pub fn weights_at(n: u32, k: u32) -> usize {
    let plane = n as usize * k as usize * 2;
    WEIGHT_BYTES.div_ceil(plane.max(1)).max(12)
}

/// The seed of [`operand_pattern`]'s xorshift.
pub const OPERAND_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// The bytes of [`operand_pattern`]: a backend uploads them once and copies
/// them device to device over and over to fill a larger operand (the
/// [`WEIGHT_BYTES`] rotation, depth-16384 planes), so laying random operands
/// costs one upload.
pub const OPERAND_BYTES: usize = 16 << 20;

/// What every float operand a rate times is filled with: [`OPERAND_BYTES`]
/// of little-endian bf16 values uniform in [-1, 1), from a xorshift64 seeded
/// [`OPERAND_SEED`], the same on every backend and every run. Not zeros:
/// some devices run a compute-bound kernel faster on zeros than on real data
/// (an MI300X's matmul and attention 15-25% faster, its power and clock
/// following the bits toggled), and serving runs real data.
#[must_use]
pub fn operand_pattern() -> Vec<u8> {
    let mut x = OPERAND_SEED;
    let mut bytes = Vec::with_capacity(OPERAND_BYTES);
    while bytes.len() < OPERAND_BYTES {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        // 24 random bits to [-1, 1) as an f32, its bf16 the top half
        // (truncation only shrinks a magnitude, so -1 <= v < 1 holds).
        let v = (x >> 40) as f32 / (1u32 << 23) as f32 - 1.0;
        bytes.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
    }
    bytes
}

/// The dispatches one timed fire of each micro-fire issues: what
/// [`rate_from`] takes their launches off for. A matmul, an attention and a
/// copy are one each; a scan is its conv and its scan.
pub const GEMM_DISPATCHES: u32 = 1;
pub const ATTN_DISPATCHES: u32 = 1;
pub const SCAN_DISPATCHES: u32 = 2;
pub const COPY_DISPATCHES: u32 = 1;

/// The tolerance two measurements of one thing agree within: two runs of a
/// rates step, a check's estimate against the measured best (what
/// [`rate_from`] allows a burst to fall below its launches before it calls
/// the two a measurement fault), and two candidates of `pie tune`'s search
/// (the lowest of those within it of the fastest is taken). Fixed, not
/// drawn from a run's own spread, so two tunes of one device take the same
/// of a near tie and write the same profile; 5% is the run-to-run spread of
/// the rates steps on the noisiest device measured (Adreno 840: a scan's
/// rate repeats within 5-10% in a session).
pub const AGREEMENT: f64 = 0.05;

/// What a backend's rates step (`pie tune --rates`) produced.
#[derive(Debug, Clone, PartialEq)]
pub struct Rated {
    pub device: String,
    /// The complete profile, its `[rates]` the ones measured.
    pub file: String,
    /// The `[rates]` block and its `[provenance]` line alone.
    pub block: String,
    pub seconds: f64,
}

/// The rate a timed fire comes to once its dispatches' launches are off:
/// `work / (burst_ns − dispatches · launch_ns)`, `burst_ns` one fire's time
/// (its `dispatches` back to back), `work` its flops, bytes or elements.
/// The estimate adds a launch a dispatch to the work at this rate
/// (`crate::load::FireEstimate`), so a rate holding its launches would count
/// them twice. A fire no longer than its launches is gap-bound: its work
/// unmeasurable at that size, stated as an infinite rate (no work, the
/// dispatch priced at its launch alone). One shorter than its launches by
/// more than [`AGREEMENT`] means the launch probe and the burst disagree,
/// and `what` is refused by name.
pub fn rate_from(
    what: &str,
    burst_ns: f64,
    dispatches: u32,
    launch_ns: f64,
    work: f64,
) -> Result<f64, String> {
    let launches = f64::from(dispatches) * launch_ns;
    let left = burst_ns - launches;
    if left > 0.0 {
        return Ok(work / left);
    }
    if burst_ns < launches * (1.0 - AGREEMENT) {
        return Err(format!(
            "{what}: a fire of {dispatches} dispatches took {burst_ns:.0} ns, below their \
             launches ({launch_ns:.0} ns each) by more than {:.0}%: the launch probe and the \
             burst disagree, so neither is a measurement",
            AGREEMENT * 100.0
        ));
    }
    Ok(f64::INFINITY)
}

/// The chain lengths the launch probe runs: a few dozen dispatches (a
/// layer's) and a thousand (a fire's), whose gaps should agree.
pub const LAUNCH_CHAINS: [u32; 2] = [64, 1024];

/// A dispatch's launch from a chain of `n` dependent kernels taking
/// `chain_ns` against one kernel doing the chain's whole work in `one_ns`:
/// what each dispatch adds, `(chain − one) / n`, never below nothing.
#[must_use]
pub fn launch_from(chain_ns: f64, one_ns: f64, n: u32) -> f64 {
    ((chain_ns - one_ns) / f64::from(n.max(1))).max(0.0)
}

/// The first rate of `rates` stated with no measurement behind it: a value
/// at or below zero, or not finite (a timing that settled at nothing), by
/// its key and where. A launch of 0 ns is a measurement (a launch the work
/// hides); nothing divides by it. Such a rate is refused by name wherever
/// rates are read ([`rates_at_load`], `crate::roofline::Constants::of`):
/// nothing divides by it.
#[must_use]
pub fn unmeasured(rates: &profile::rates::Rates) -> Option<(&'static str, String)> {
    let bad = |v: f64| !(v.is_finite() && v > 0.0);
    let point = |key: &'static str, unit: &str, curve: &[(u32, f64)]| {
        curve
            .iter()
            .find(|&&(_, v)| bad(v))
            .map(|&(at, v)| (key, format!("at {at} {unit} as {v}")))
    };
    if let Some(widths) = &rates.gemm_tflops_by_rows {
        for (w, curve) in widths {
            if let Some((key, at)) = point("gemm_tflops_by_rows", "rows", curve) {
                return Some((key, format!("{at} (width {w})")));
            }
        }
    }
    if let Some(by_rows) = &rates.attn_tflops_by_rows_keys {
        for (rows, curve) in by_rows {
            if let Some((key, at)) = point(profile::rates::ATTN_BY_ROWS_KEYS, "keys", curve) {
                return Some((key, format!("{at} ({rows} rows)")));
            }
        }
    }
    if let Some(curves) = &rates.kernel_curves {
        for c in curves {
            if let Some((key, at)) = point(profile::rates::KERNEL_CURVES, "rows", &c.curve) {
                return Some((
                    key,
                    format!("{at} (kernel {}/{} at {}x{})", c.kind, c.key, c.n, c.k),
                ));
            }
        }
    }
    if let Some(found) = rates
        .attn_tflops_by_keys
        .as_deref()
        .and_then(|curve| point("attn_tflops_by_keys", "keys", curve))
    {
        return Some(found);
    }
    for (key, v) in [
        ("hbm_gbps", rates.hbm_gbps),
        ("elementwise_gbps", rates.elementwise_gbps),
        ("scan_gelems", rates.scan_gelems),
        ("few_row_tflops", rates.few_row_tflops),
        ("workgroups_per_core", rates.workgroups_per_core),
        (profile::rates::SUBMIT_WORST_RATIO, rates.submit_worst_ratio),
    ] {
        if let Some(v) = v.filter(|&v| bad(v)) {
            return Some((key, format!("as {v}")));
        }
    }
    None
}

/// How a rates step admits its frames under the device's bound on one
/// submission (`max_submit`, where it states one), every engine's rule:
/// a frame is estimated from its fire's last measured time, and admitted
/// where that estimate times [`Admission::worst`] fits the bound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Admission {
    pub bound: Option<std::time::Duration>,
    /// Whether the device's clock has settled: the warm-up's rotations
    /// agreed ([`warm_up`], by [`settle`]'s rule). Frames before it measure
    /// the clock's ramp, not an estimate's error.
    pub settled: bool,
    /// The largest ratio of a settled frame's time to its estimate seen,
    /// never below 1: a measurement, over every settled frame carrying its
    /// estimate up, no decay and no window.
    pub worst: f64,
}

impl Admission {
    #[must_use]
    pub fn new(bound: Option<std::time::Duration>) -> Admission {
        Admission {
            bound,
            settled: false,
            worst: 1.0,
        }
    }

    /// Whether a frame estimated at `estimate_ns` fits the bound at the
    /// worst ratio seen: always with no bound, never with a bound and no
    /// estimate (a fire with no estimate is never submitted under one).
    #[must_use]
    pub fn fits(&self, estimate_ns: Option<f64>) -> bool {
        match (self.bound, estimate_ns) {
            (Some(cap), Some(e)) => e * self.worst <= cap.as_nanos() as f64,
            (Some(_), None) => false,
            (None, _) => true,
        }
    }

    /// The fires a frame of `count` holds at `per_fire_ns` a fire: `count`
    /// halved until the frame fits; `None` where not even one fire does
    /// (the shape is past the bound, refused by name).
    #[must_use]
    pub fn fires(&self, count: u32, per_fire_ns: Option<f64>) -> Option<u32> {
        let mut n = count.max(1);
        loop {
            if self.fits(per_fire_ns.map(|p| p * f64::from(n))) {
                return Some(n);
            }
            if n == 1 {
                return None;
            }
            n /= 2;
        }
    }

    /// A frame landed: `took_ns` against its `estimate_ns`, carried up
    /// from a frame of no more fires (`upward`). Only a settled device's
    /// frames move the worst ratio.
    pub fn landed(&mut self, estimate_ns: Option<f64>, took_ns: f64, upward: bool) {
        if let Some(e) = estimate_ns.filter(|&e| e > 0.0)
            && self.settled
            && upward
        {
            self.worst = self.worst.max(took_ns / e);
        }
    }
}

/// The side of the cube the warm-up's size walk ([`warm_side`]) times
/// first, and the one after it: the walk's first growth is theirs.
pub const WARM_PROBE: u32 = 1024;

/// The side of the warm-up's cube matmul (`side x side x side`), walked up
/// by doubling with a real fire timed at each size (`time(side, estimate)`,
/// ns; `estimate` the walk's own for that fire, none for the two probes;
/// `None` where the caller's own estimate of the fire is past the bound and
/// it was never submitted: the walk stops at the size before):
/// [`WARM_PROBE`] / 2 and [`WARM_PROBE`] first, then the next size only
/// while its fire, estimated from the last one timed by the growth the last
/// two measured (their time ratio, carried forward: not the flops' ratio),
/// times `worst`, fits `max_submit`; to [`WARM_SHAPE`]'s at most, which it
/// is with no bound stated (nothing is walked then). `worst` starts at the
/// caller's worst ratio of a measured time over its estimate (1 for none)
/// and rises to each walked fire's measured time over the walk's estimate
/// of it where that is more: the estimate's own error, measured, stands
/// under the bound; no factor is tuned. Returns the side and the worst
/// ratio seen. A phone whose GPU faults a submission past its watchdog
/// warms on a cube whose fire fits it.
pub fn warm_side<E>(
    max_submit: Option<std::time::Duration>,
    mut worst: f64,
    mut time: impl FnMut(u32, Option<f64>) -> Result<Option<f64>, E>,
) -> Result<(u32, f64), E> {
    let Some(cap) = max_submit else {
        return Ok((WARM_SHAPE.0, worst));
    };
    let bound = cap.as_nanos() as f64;
    let Some(mut before) = time(WARM_PROBE / 2, None)? else {
        return Ok((WARM_PROBE / 2, worst));
    };
    let Some(mut last) = time(WARM_PROBE, None)? else {
        return Ok((WARM_PROBE / 2, worst));
    };
    let mut side = WARM_PROBE;
    if last * worst > bound {
        return Ok((WARM_PROBE / 2, worst));
    }
    while side < WARM_SHAPE.0 {
        let growth = last / before.max(f64::MIN_POSITIVE);
        let estimate = last * growth;
        if estimate * worst > bound {
            break;
        }
        let Some(next) = time(side * 2, Some(estimate))? else {
            break;
        };
        worst = worst.max(next / estimate.max(f64::MIN_POSITIVE));
        if next * worst > bound {
            break;
        }
        (before, last, side) = (last, next, side * 2);
    }
    Ok((side, worst))
}

/// The bytes an attention of `shape` moves at least: its queries read and
/// its output written (each row a query element a head), and its keys' and
/// values' cells read once.
#[must_use]
pub fn attn_bytes(shape: AttnShape) -> f64 {
    let q = 2.0 * f64::from(shape.rows) * f64::from(shape.q_heads * shape.head_dim) * 2.0;
    let kv = 2.0 * f64::from(shape.keys) * f64::from(shape.kv_heads * shape.head_dim) * 2.0;
    q + kv
}

/// The keys of [`ATTN_KEYS`] the attention curve is timed at on a device
/// whose ridge is `ridge` (its matmul peak, flops a ns, over its bandwidth,
/// bytes a ns): from the narrowest whose shape ([`ATTN_SHAPE`] at it) is
/// compute-bound there (its flops a byte at least the ridge) on, so the
/// rate is the kernel's compute, not its reads; the widest where none is.
/// The matmul curve's points play no part.
#[must_use]
pub fn attn_keys(ridge: f64) -> Vec<u32> {
    let bound = |keys: u32| {
        let s = ATTN_SHAPE.at(keys);
        attn_flops(s) / attn_bytes(s) >= ridge
    };
    let from = ATTN_KEYS
        .iter()
        .position(|&k| bound(k))
        .unwrap_or(ATTN_KEYS.len() - 1);
    ATTN_KEYS[from..].to_vec()
}

/// The matmul every backend's rates step warms the device with before it
/// times anything ([`warm_up`]): the widest the grid times, `m x n x k`.
pub const WARM_SHAPE: (u32, u32, u32) = (8192, 8192, 8192);

/// How a warm-up ended: the rotations of [`WARM_SHAPE`] over its
/// [`weights_at`] planes it ran, and the rate (flops a ns) of the last.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Warmed {
    pub rotations: u32,
    pub rate: f64,
}

/// Warm the device until its clock stops falling: run `rotation` (one
/// rotation of [`WARM_SHAPE`] back to back over its [`weights_at`] planes,
/// returning its rate) while each rotation's rate is below the one before
/// it (the clock still sinking under its power cap), and stop at the first
/// rotation that is not slower than its predecessor (the descent over).
/// Every backend's rates step calls it first, so its timings start on a
/// settled clock rather than a cold one.
pub fn warm_up<E>(mut rotation: impl FnMut() -> Result<f64, E>) -> Result<Warmed, E> {
    // The rule every measurement's frames settle by ([`settle`]), over the
    // rotation's rate: until it stops moving by AGREEMENT two steps in a
    // row. A memory-bound measurement warms by it over its own op (its rate
    // rising through the DVFS levels to the sustained one serving sees).
    let mut rate = 0.0;
    let settled = settle(|_| {
        rate = rotation()?;
        Ok(Some(rate))
    })?;
    Ok(Warmed {
        rotations: settled.steps,
        rate,
    })
}

/// The longest a [`settle`] runs, in steps: a doubling's frames past it
/// carry more fires than a `u32` counts.
pub const SETTLE_STEPS: u32 = 31;

/// How a [`settle`] ended: its steps, the last value, and whether a step it
/// needed was not taken (`capped`: the measurement's bound on one frame
/// stopped it before two steps agreed).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settled {
    pub steps: u32,
    pub last: f64,
    pub capped: bool,
}

/// The one rule a rates step's measurements settle by: take `step(i)` (a
/// rotation's rate while the clock moves, a fire's time in frames of
/// doubling length, `i` from 0) until two successive steps each move it by
/// less than [`AGREEMENT`], in either direction (three agreeing points: in
/// a doubling, spanning 4x), or `step` declines to take another (`None`:
/// capped). Two short frames caught in one state of a bimodal clock agree
/// with each other, but not with the frame twice as long that leaves it
/// (an M1's attention at 512 keys read 2.32 to 4.60 TFLOP/s run to run
/// under a rule of two agreeing points, 4.1 to 4.4 under this one). The
/// warm-up ([`warm_up`]), a memory-bound measurement's warming
/// ([`warm_up`] over its own op) and every measurement's frame length
/// ([`timed_frames`]) take it.
pub fn settle<E>(mut step: impl FnMut(u32) -> Result<Option<f64>, E>) -> Result<Settled, E> {
    let Some(mut last) = step(0)? else {
        return Ok(Settled {
            steps: 0,
            last: f64::NAN,
            capped: true,
        });
    };
    let mut agreed = 0;
    for i in 1..=SETTLE_STEPS {
        let Some(next) = step(i)? else {
            return Ok(Settled {
                steps: i,
                last,
                capped: true,
            });
        };
        // Settled at the second step in a row that moves it by less than
        // AGREEMENT either way.
        agreed = if (next - last).abs() < AGREEMENT * last.abs() {
            agreed + 1
        } else {
            0
        };
        last = next;
        if agreed == 2 {
            return Ok(Settled {
                steps: i + 1,
                last,
                capped: false,
            });
        }
    }
    Ok(Settled {
        steps: SETTLE_STEPS + 1,
        last,
        capped: false,
    })
}

/// Where a measurement's fires are in their rotation of operand sets: one
/// counter carried across every frame of the measurement, so a short frame
/// reads the next sets, never the first ones again (a frame restarting at
/// set 0 re-reads warm weights).
#[derive(Debug, Default)]
pub struct Rotation(std::cell::Cell<u64>);

impl Rotation {
    /// The next fire's set of `sets`.
    #[must_use]
    pub fn next(&self, sets: usize) -> usize {
        let at = self.0.get();
        self.0.set(at + 1);
        (at % sets.max(1) as u64) as usize
    }
}

/// How long a measurement's timed frame runs: its fires, the time one fire
/// took in a frame of them, and whether the device's bound on one
/// submission stopped the doubling before it settled.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frames {
    pub fires: u32,
    pub per_fire_ns: f64,
    pub capped: bool,
}

/// A measurement's frame length, the one rule for every engine: frames of
/// 1, 2, 4, ... fires (`frame(fires)` runs one and returns its ns), doubled
/// until the time a fire stops moving by more than [`AGREEMENT`]
/// ([`settle`]): the clock held through the frame and its submission
/// amortized. `max_submit` (the device's bound on one submission, where it
/// states one) is the only cap: no frame past it is laid, and `estimate_ns`
/// (one fire's, where the caller can say) past it is refused by name
/// without running.
pub fn timed_frames<E>(
    max_submit: Option<std::time::Duration>,
    estimate_ns: Option<f64>,
    mut frame: impl FnMut(u32) -> Result<f64, E>,
    refuse: impl FnOnce(String) -> E,
) -> Result<Frames, E> {
    let cap = max_submit.map(|d| d.as_nanos() as f64);
    if let (Some(cap), Some(est)) = (cap, estimate_ns)
        && est > cap
    {
        return Err(refuse(format!(
            "one fire of about {:.1} ms is past the device's bound of {:.1} ms on one \
             submission (`max_submit_ms`): no frame of it is laid",
            est / 1e6,
            cap / 1e6
        )));
    }
    let mut fires = 1u32;
    let mut per = f64::NAN;
    let settled = settle(|i| {
        let count = 1u32 << i.min(31);
        if i > 0
            && let Some(cap) = cap
            && per * f64::from(count) > cap
        {
            return Ok(None);
        }
        per = frame(count)? / f64::from(count);
        fires = count;
        Ok(Some(per))
    })?;
    Ok(Frames {
        fires,
        per_fire_ns: settled.last,
        capped: settled.capped,
    })
}

/// Whether the operand sets one owner lays (a width's weight planes, an
/// attention's q / kv sets, ...) have been touched on the device yet: held
/// beside the sets, by whatever lays them, and set by [`touch_once`].
#[derive(Debug, Default)]
pub struct Touched(std::cell::Cell<bool>);

impl Touched {
    #[must_use]
    pub fn get(&self) -> bool {
        self.0.get()
    }
}

/// A set is touched once, by the first point that rotates it: the first
/// time a point is timed over the `sets` operand sets `touched` stands
/// for, every one of them is read once on the device, untimed
/// ([`touch_sets`]), before the point's frames ([`warmed_frames`]); a
/// later point over the same sets reads them as they are. A set the
/// device has never read costs its first fire far more than a warm one (on
/// an M5 Pro a 256 x 1024 x 1024 matmul over 384 fresh 2 MiB planes ran
/// 112 us a fire against 25 us once they had been read, so the point read
/// about 5 or about 30 TFLOP/s run to run by whether its few timed fires
/// fell on planes read before). Touched once, not before every point: a
/// rotation before each point is that much more load on the device, and
/// on the M5 Pro it read the widest matmuls 2-5% under the first set's
/// rates, heat the measurement had added.
pub fn touch_once<E>(
    touched: &Touched,
    max_submit: Option<std::time::Duration>,
    sets: u32,
    frame: &mut impl FnMut(u32) -> Result<f64, E>,
) -> Result<(), E> {
    if touched.get() {
        return Ok(());
    }
    touch_sets(max_submit, sets, frame)?;
    touched.0.set(true);
    Ok(())
}

/// Every one of the `sets` operand sets a point's fires rotate through,
/// read once on the device and untimed: `sets` fires in all, in one frame,
/// or in frames of as many as fit under `max_submit` where the device
/// states a bound on one submission.
pub fn touch_sets<E>(
    max_submit: Option<std::time::Duration>,
    sets: u32,
    frame: &mut impl FnMut(u32) -> Result<f64, E>,
) -> Result<(), E> {
    let mut left = sets.max(1);
    let one = frame(1)?;
    left -= 1;
    let per = max_submit.map_or(left, |cap| {
        ((cap.as_nanos() as f64 / one.max(1.0)) as u32).max(1)
    });
    while left > 0 {
        let n = per.min(left);
        frame(n)?;
        left -= n;
    }
    Ok(())
}

/// A timed point's frames, the one rule for every point every engine
/// times: their length settled by [`timed_frames`], then frames of that
/// length run until their rate stops moving ([`warm_up`]'s rule), so the
/// point is timed at the clock it holds under its own load, never in a
/// state an earlier point or an idle gap left behind. Before this every
/// point but the memory-bound ones went untimed by it, and on an M5 Pro a
/// few-microsecond matmul read about 5 or about 34 TFLOP/s run to run.
/// (Its sets touched first, once: [`touch_once`].)
pub fn warmed_frames<E>(
    max_submit: Option<std::time::Duration>,
    estimate_ns: Option<f64>,
    mut frame: impl FnMut(u32) -> Result<f64, E>,
    refuse: impl FnOnce(String) -> E,
) -> Result<Frames, E> {
    let frames = timed_frames(max_submit, estimate_ns, &mut frame, refuse)?;
    let k = frames.fires;
    warm_up(|| frame(k).map(|ns| f64::from(k) / ns.max(f64::MIN_POSITIVE)))?;
    Ok(frames)
}

/// What a backend's rates step measured, and whether a frame it needed
/// stopped at the device's bound on one submission before it settled.
#[derive(Debug, Clone, PartialEq)]
pub struct Measured {
    pub rates: profile::rates::Rates,
    pub unsettled: bool,
    /// The device ran slower at the step's end than at its warmed start
    /// (`Some(ratio)`, its reference kernel's time over its fastest seen):
    /// the rates are this load's alone, never stored, and said so.
    pub throttled: Option<f64>,
}

/// The order a rates step times `len` points in: a fixed stride through
/// them ([`attn_page_order`]'s), not ascending, so a drift left after the
/// warm-up spreads over the grid rather than biasing its widest rows.
#[must_use]
pub fn spread_order(len: usize) -> Vec<usize> {
    attn_page_order(u32::try_from(len).unwrap_or(u32::MAX))
        .into_iter()
        .map(|i| i as usize)
        .collect()
}

/// The bytes of q, k and v sets an attention's timed fires rotate through
/// at least: [`WEIGHT_BYTES`]' bound, past every last-level cache, since a
/// served fire reads each layer's kv (and its q) cold.
pub const ATTN_SET_BYTES: usize = WEIGHT_BYTES;

/// The q / k / v sets the timed attentions of `shape` rotate through, its
/// kv laid in pages of `page` keys: [`ATTN_SET_BYTES`] of the shape's q rows
/// and of k and v over the pages as laid, whole (bf16), and never fewer
/// than 2. A few-key point's page is laid whole, so its sets are counted by
/// it (as engine-cuda sizes them).
#[must_use]
pub fn attn_sets_at(shape: AttnShape, page: u32) -> usize {
    let q_bytes = shape.rows as usize * (shape.q_heads * shape.head_dim) as usize * 2;
    let cells = (shape.keys.div_ceil(page.max(1)) * page.max(1)) as usize;
    let kv_bytes = 2 * cells * (shape.kv_heads * shape.head_dim) as usize * 2;
    ATTN_SET_BYTES.div_ceil((q_bytes + kv_bytes).max(1)).max(2)
}

/// A prefill attention the attention rate is timed at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttnShape {
    /// One sequence's query rows, every one a new token.
    pub rows: u32,
    /// The keys those rows attend: the sequence itself (the rows' own, kv
    /// written before the attention), causal.
    pub keys: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
}

/// One sequence of 2048 tokens attending itself, causal, at the head shape
/// of the models served (Qwen3's 16 query heads over 8 kv heads at 128 a
/// head; grouped as most models group): long enough that the tiles of every
/// backend's prefill kernel fill, short enough to fit a browser's binding.
pub const ATTN_SHAPE: AttnShape = AttnShape {
    rows: 2048,
    keys: 2048,
    q_heads: 16,
    kv_heads: 8,
    head_dim: 128,
};

/// The step the attention's page table walks its pool by: page `i` of the
/// sequence is pool page `i · ATTN_PAGE_STRIDE mod pages`
/// ([`attn_page_order`]). A served fire reads pages the pool laid scattered,
/// not 0, 1, 2, ... in order, and a prefill streams them slower so; the
/// curve is timed over the same scatter on every backend. A prime past any
/// page count the lattice lays (16384 keys in pages of 16 are 1024), so it
/// is coprime with every one and the order a permutation.
pub const ATTN_PAGE_STRIDE: u32 = 7919;

/// The pool page each of `pages` sequence pages lies in, in sequence order:
/// a fixed permutation of `0..pages` (no RNG), the same on every backend.
#[must_use]
pub fn attn_page_order(pages: u32) -> Vec<u32> {
    let n = u64::from(pages.max(1));
    // A stride coprime with the count, so the walk visits every page once.
    let gcd = |mut a: u64, mut b: u64| {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    // Past two pages, a stride of neither 0 nor 1 (mod the count): the walk
    // never keeps the pages in order.
    let mut stride = u64::from(ATTN_PAGE_STRIDE) % n;
    while n > 2 && (stride <= 1 || gcd(stride, n) != 1) {
        stride = (stride + 1) % n;
    }
    let stride = stride.max(1);
    (0..u64::from(pages))
        .map(|i| u32::try_from(i * stride % n).unwrap_or(0))
        .collect()
}

/// The head widths of the models served (64 for the small and older ones,
/// 128 for most, 256 for Qwen3.5's and Gemma's global layers). The curve is
/// 128's; the step times the others at its longest keys and says them
/// beside it, so a width whose rate departs from it shows.
pub const ATTN_DIMS: [u32; 3] = [64, 128, 256];

impl AttnShape {
    /// This shape's heads, each `head_dim` wide.
    #[must_use]
    pub fn with_dim(self, head_dim: u32) -> AttnShape {
        AttnShape { head_dim, ..self }
    }

    /// This shape's heads over one sequence of `keys` rows attending itself.
    #[must_use]
    pub fn at(self, keys: u32) -> AttnShape {
        AttnShape {
            rows: keys,
            keys,
            ..self
        }
    }
}

/// The flops the estimate counts for an attention of `shape`: four a query
/// element a key (the scores' product and the values'), over every row and
/// key of the rectangle, as the estimate's attention term counts a fire's
/// (`tokens · keys · attn_flops_per_row_key`). What a causal kernel skips
/// of the rectangle is in the rate it reaches, so the term reads it back.
#[must_use]
pub fn attn_flops(shape: AttnShape) -> f64 {
    4.0 * f64::from(shape.rows)
        * f64::from(shape.keys)
        * f64::from(shape.q_heads)
        * f64::from(shape.head_dim)
}

/// A prefill scan the recurrent rate is timed at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanShape {
    /// One sequence's new tokens.
    pub tokens: u32,
    pub k_heads: u32,
    pub v_heads: u32,
    pub k_dim: u32,
    pub v_dim: u32,
    /// The causal conv's taps over the scan's `qkv` channels.
    pub conv_width: u32,
}

/// The scan sets the timed scans of `shape` rotate through: [`ATTN_SET_BYTES`]
/// of the shape's input rows (bf16) and its conv and scan state (f32), and
/// never fewer than 2, so each reads its state cold as a served layer's.
#[must_use]
pub fn scan_sets_at(shape: ScanShape) -> usize {
    let rows = shape.tokens as usize * shape.channels() as usize * 2;
    let state = 4
        * (shape.conv_width as usize * shape.channels() as usize
            + (shape.v_heads * shape.k_dim * shape.v_dim) as usize);
    ATTN_SET_BYTES.div_ceil((rows + state).max(1)).max(2)
}

impl ScanShape {
    /// This shape's heads over a prompt of `tokens`.
    #[must_use]
    pub fn at(self, tokens: u32) -> ScanShape {
        ScanShape { tokens, ..self }
    }

    /// The conv's channels: the scan's `qkv` row, its keys and queries and
    /// its values.
    #[must_use]
    pub fn channels(self) -> u32 {
        2 * self.k_heads * self.k_dim + self.v_heads * self.v_dim
    }
}

/// One sequence of 2048 tokens through a gated-delta layer at the shape of
/// the hybrids served (Qwen3.5's linear attention: 16 key and 16 value
/// heads of 128, a conv of 4 taps): the state a head is 128 x 128, the size
/// every backend's register form is built for.
pub const SCAN_SHAPE: ScanShape = ScanShape {
    tokens: 2048,
    k_heads: 16,
    v_heads: 16,
    k_dim: 128,
    v_dim: 128,
    conv_width: 4,
};

/// The state elements the estimate counts for a scan of `shape`: each token
/// steps every value head's `k_dim x v_dim` state, as the estimate's
/// recurrent term counts a fire's (`tokens · state_elements_per_row`). The
/// conv before it is in the rate, not the count.
#[must_use]
pub fn scan_elements(shape: ScanShape) -> f64 {
    f64::from(shape.tokens)
        * f64::from(shape.v_heads)
        * f64::from(shape.k_dim)
        * f64::from(shape.v_dim)
}

/// The `[rates]` block a `pie tune <backend> --rates` writes, as profile
/// TOML: every rate it measured.
#[must_use]
pub fn tuned(rates: &profile::rates::Rates) -> String {
    rates.toml()
}

/// The `[provenance]` entry beside it: `rates = "pie tune <backend> --rates
/// on <date> (<version>): <how>"`, which the load's notice reads back as how
/// the rates were measured. `how` says what the backend's own path was (its
/// matmul kernel, its clock).
#[must_use]
pub fn provenance(backend: &str, how: &str) -> (&'static str, String) {
    let (rows, width) = ELEMENTWISE_SHAPE;
    let text = format!(
        "pie tune {backend} --rates on {} (pie {}): {how}; at engine::rates' shapes (matmul m x \
         w x w, w {:?}, m {:?}; attention keys x keys, keys {:?}, {}/{} heads of \
         128; per-row ops {rows} x {width}; scan {} tokens, {}/{} heads of {}x{}, conv {})",
        today(),
        env!("CARGO_PKG_VERSION"),
        GEMM_WIDTHS,
        GEMM_ROWS,
        ATTN_KEYS,
        ATTN_SHAPE.q_heads,
        ATTN_SHAPE.kv_heads,
        SCAN_SHAPE.tokens,
        SCAN_SHAPE.k_heads,
        SCAN_SHAPE.v_heads,
        SCAN_SHAPE.k_dim,
        SCAN_SHAPE.v_dim,
        SCAN_SHAPE.conv_width,
    );
    ("rates", format!("\"{}\"", text.replace('"', "'")))
}

/// Where a load saves the rates it measured: the user's profile of the
/// device (`profile::Shelf::user_path`), with the backend's `[provenance]
/// rates` entry naming how, and its check that the file still reads (the
/// backend's registry, strictly as a user's file).
pub struct SaveTo<'a> {
    pub path: std::path::PathBuf,
    pub provenance: (&'static str, String),
    /// The backend's registry: what the saved file is read against, and
    /// what of it a load no longer reads ([`profile::Registry::prune`]).
    pub registry: &'a profile::Registry,
}

/// The user's file of a device, `existing` (none where there is none yet),
/// with its `[rates]` table written whole: the rates the file's own table
/// states where it reads (every key one this pie reads), under `measured`
/// (only the keys a load measured); a table stating a key this pie does not
/// read (a schema it outlived) is replaced, none of it kept. `provenance`
/// is set in its `[provenance]`; nothing else of the file changes, and
/// nothing of the profile it resolved to is copied into it. The user's file
/// is the loads' own measured output: no key is spliced into a table it
/// might already hold.
#[must_use]
pub fn saved_text(
    existing: Option<&str>,
    measured: &profile::rates::Rates,
    provenance: &(&'static str, String),
) -> String {
    let entry = format!("{} = {}", provenance.0, provenance.1);
    let mut lines: Vec<String> = existing
        .unwrap_or("api = 1\n")
        .lines()
        .map(str::to_string)
        .collect();
    let header =
        |lines: &[String], name: &str| lines.iter().position(|l| l.trim() == format!("[{name}]"));
    // The table's lines after `at`, to the next table: a header starts its
    // line, where a rate's array rows (`  [1024, [...]]`) are indented.
    let table_end = |lines: &[String], at: usize| {
        let mut end = lines[at + 1..]
            .iter()
            .position(|l| l.starts_with('['))
            .map_or(lines.len(), |i| at + 1 + i);
        while end > at + 1 && lines[end - 1].trim().is_empty() {
            end -= 1;
        }
        end
    };
    // The old table out, whole, the new one in its place; what of the old
    // reads stays under the measured.
    let at = header(&lines, "rates");
    let mut kept = profile::rates::Rates::default();
    if let Some(at) = at {
        let end = table_end(&lines, at);
        let table = format!("api = 1\n{}\n", lines[at..end].join("\n"));
        if let Ok(file) = profile::Registry::new("rates", &[], &[]).parse(
            &table,
            "the user's [rates]",
            profile::Strict::User,
        ) {
            kept = file.tuned.rates;
        }
        lines.drain(at..end);
    }
    // Keyed measurements ([`profile::rates::Rates::measured_on`]) are kept
    // only beside those of the same device, each keyed: what another device
    // measured, or a rate stored without a key, is no measurement of this
    // one.
    if let Some(on) = &measured.measured_on {
        if kept.measured_on.as_ref() != Some(on) {
            kept = profile::rates::Rates::default();
        }
        for key in profile::rates::KEYED {
            if kept.states(key) && kept.key_of(key).is_none() {
                kept.unstate(key);
            }
        }
    }
    let table: Vec<String> = measured
        .clone()
        .over(kept)
        .toml()
        .lines()
        .map(str::to_string)
        .collect();
    match at {
        Some(at) => {
            lines.splice(at..at, table);
        }
        None => {
            lines.push(String::new());
            lines.extend(table);
        }
    }
    match header(&lines, "provenance") {
        Some(at) => {
            let end = table_end(&lines, at);
            let key = format!("{} ", provenance.0);
            let stated = lines[at + 1..end].iter().position(|l| {
                l.trim_start().starts_with(&key)
                    || l.trim_start().starts_with(&format!("{}=", provenance.0))
            });
            match stated {
                Some(i) => lines[at + 1 + i] = entry,
                None => lines.insert(end, entry),
            }
        }
        None => {
            lines.push(String::new());
            lines.push("[provenance]".to_string());
            lines.push(entry);
        }
    }
    let mut text = lines.join("\n");
    text.push('\n');
    text
}

/// The roofline a load weighs its fires by, one rule for every engine: the
/// rates `chip`'s layers state where they state every one of `keys` (those
/// the backend's estimate weighs a fire by, of `profile::rates::KEYS`) (a checked-in
/// profile's over a saved one, `profile::Registry::resolve`); else, before
/// the load is ready, the device's own, measured by `measure` (the backend's
/// `pie tune --rates` step, seconds long, its frames sized by
/// [`timed_frames`] under the device's `max_submit` where it states one),
/// each rate a layer states kept over the measured one, and what was
/// measured saved to `save` (with its provenance) so later loads read it
/// and measure nothing. The returned profile's rates are marked measured
/// this load, where they were saved and whether a frame stopped at the
/// bound unsettled, and the load line says so (`engine::load::line_of`).
/// The step runs at the start of the load, its buffers of `room` (what the
/// load's budget leaves the device, where the engine can say): where that
/// holds less than it lays at most ([`step_bytes`]) the load is refused,
/// as is a rate the step settled at nothing ([`unmeasured`]), before
/// anything is saved.
pub fn rates_at_load(
    chip: profile::Profile,
    keys: &[&'static str],
    save: Option<SaveTo<'_>>,
    room: Option<u64>,
    max_submit: Option<std::time::Duration>,
    measure: impl FnOnce(Option<std::time::Duration>) -> Result<Measured, String>,
) -> Result<profile::Profile, String> {
    resolve_at_load(
        chip,
        keys,
        save,
        room,
        max_submit,
        Measuring::Time,
        StaleRates::Measured,
        |max_submit, _| measure(max_submit),
    )
}

/// What an engine does with a rate (or curve) a layer states of its device
/// under a stale key (a kernel's code or the measuring method changed): a
/// statement of the engine's, by platform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StaleRates {
    /// Served by the stale figure, and timed again behind serving.
    TimedBehind,
    /// Served by the stale figure, not timed again by this load (the next
    /// export round refreshes it).
    Served,
    /// Measured again before ready, as a key no layer states.
    Measured,
}

/// Whether a load times what no layer states (`[engine] measure`, `pie
/// serve --no-measure`): [`Measuring::Time`] measures each rate, curve and
/// settle no layer states under a matching key, before the load is ready,
/// and stores it; [`Measuring::Skip`] times nothing: each is served by the
/// engine's estimate, named where it is, and nothing is stored for it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Measuring {
    #[default]
    Time,
    Skip,
}

impl Measuring {
    /// `[engine] measure`'s value: `true` times, `false` skips.
    #[must_use]
    pub fn of(measure: bool) -> Measuring {
        if measure {
            Measuring::Time
        } else {
            Measuring::Skip
        }
    }
}

/// [`rates_at_load`] under `measuring`: where it skips, nothing is timed or
/// saved, each key of `keys` no layer states is said, by name, to be served
/// by the estimate, and kept as skipped in the profile
/// (`profile::rates::Tuned::skipped`), so each choice that reads it takes
/// the engine's estimate and names it. A checked-in profile stating a rate
/// this pie does not read is refused either way: it is a build's fault, not
/// a measurement.
pub fn resolve_at_load(
    chip: profile::Profile,
    keys: &[&'static str],
    save: Option<SaveTo<'_>>,
    room: Option<u64>,
    max_submit: Option<std::time::Duration>,
    measuring: Measuring,
    stale_rates: StaleRates,
    measure: impl FnOnce(Option<std::time::Duration>, &profile::rates::Rates) -> Result<Measured, String>,
) -> Result<profile::Profile, String> {
    // A checked-in file this device takes states a rate this pie does not
    // read: it was refused whole, and the load is refused by it.
    if let Some(stale) = &chip.tuned().refused {
        return Err(format!(
            "{stale}: a checked-in profile states a rate this pie does not read, and is refused \
             whole; [rates] holds {:?}",
            profile::rates::KEYS
        ));
    }
    let stated = chip.rates();
    if stated.complete(keys) {
        return Ok(chip);
    }
    let lacked = stated.missing(keys);
    // A key a layer states of this device under a stale key (a kernel's
    // code or the method changed) is served by that figure, a fair
    // estimate, and timed again behind serving
    // (`profile::rates::Tuned::behind`): a load never measures before ready
    // what it holds a figure for. Only a key no layer states at all is
    // measured here. What the engine does with such a key is its own
    // statement ([`StaleRates`]).
    let stale = stated
        .stale
        .as_deref()
        .cloned()
        .filter(|_| stale_rates != StaleRates::Measured || measuring == Measuring::Skip)
        .unwrap_or_default();
    let estimated: Vec<&'static str> =
        lacked.iter().copied().filter(|key| stale.states(key)).collect();
    let (chip, stated, lacked) = if estimated.is_empty() {
        (chip, stated, lacked)
    } else {
        for key in &estimated {
            tracing::info!(
                "rates: `[rates] {key}`: served by the figure stored under its stale key (the \
                 stored measurements' lines say why it is stale){}",
                match (measuring, stale_rates) {
                    (Measuring::Time, StaleRates::TimedBehind) => {
                        "; timed again behind serving, stored for the next load"
                    }
                    (Measuring::Time, _) => {
                        " (this engine does not time it behind serving here: the next export \
                         round refreshes it)"
                    }
                    (Measuring::Skip, _) => " (`[engine] measure = false`: not timed again)",
                }
            );
        }
        let stated = stated.over(stale.only(&estimated));
        let mut tuned = chip.tuned().clone();
        tuned.rates = stated.clone();
        if measuring == Measuring::Time && stale_rates == StaleRates::TimedBehind {
            tuned
                .behind
                .extend(estimated.iter().map(|key| (*key).to_string()));
        }
        let chip = chip.with_tuned(tuned);
        let lacked = stated.missing(keys);
        if lacked.is_empty() {
            return Ok(chip);
        }
        (chip, stated, lacked)
    };
    if measuring == Measuring::Skip {
        for key in &lacked {
            tracing::info!(
                "rates: skipped: `[rates] {key}`: nothing timed (`[engine] measure = false`) and no \
                 layer states it under a matching key; served by the estimate, which names what \
                 it takes in its place"
            );
        }
        let mut tuned = chip.tuned().clone();
        tuned.skipped.extend(lacked.iter().map(|key| (*key).to_string()));
        return Ok(chip.with_tuned(tuned));
    }
    // The step lays its buffers before any of the load's: where the load's
    // budget leaves the device less than it lays at most, the load is
    // refused by name, never left to wait for memory.
    let need = step_bytes();
    if let Some(room) = room
        && need > room
    {
        return Err(format!(
            "the profile states no roofline rates {lacked:?}, and measuring them needs {} MiB of \
             device memory where the load's budget leaves {} MiB",
            need >> 20,
            room >> 20
        ));
    }
    let Measured {
        rates: measured,
        unsettled,
        throttled,
    } = measure(max_submit, &stated)?;
    if let Some((key, why)) = unmeasured(&measured) {
        return Err(format!(
            "the device's rates step measured `[rates] {key}` {why}: nothing weighs a fire by \
             it, and no rate is divided by"
        ));
    }
    let rates = stated.over(measured);
    if !rates.complete(keys) {
        return Err(format!(
            "the device's rates step measured no {:?}: nothing weighs a fire by them",
            rates.missing(keys)
        ));
    }
    // The user's file is this load's own output (`save_rates`): rates
    // timed on a device whose clock sagged during them serve this load alone.
    let saved_to = match throttled {
        None => save
            .map(|save| save_rates(&save, &rates.only(&lacked)))
            .transpose()?,
        Some(ratio) => {
            tracing::warn!(
                ratio,
                "rates: timed with the device {ratio:.2}x slower at the step's end than at its \
                 warmed start (its reference kernel's fastest): served this load, not stored"
            );
            None
        }
    };
    let mut tuned = chip.tuned().clone();
    tuned.rates = rates;
    tuned.measured_this_load = true;
    tuned.timed.push(match throttled {
        None => format!("rates ({})", lacked.join(", ")),
        Some(ratio) => format!(
            "rates ({}; device {ratio:.2}x slower at the end than the start: not stored)",
            lacked.join(", ")
        ),
    });
    tuned.saved_to = saved_to;
    tuned.frame_cut = unsettled;
    Ok(chip.with_tuned(tuned))
}

/// `measured` written into the user's file `save` names, as a load saves
/// what it measured: the file keeps every section and key this pie still
/// reads (a person's own decisions), drops each it no longer reads (said, by
/// name), gains `measured` (each key it states replacing the file's) and how
/// it was measured. The file must read back; its path, said.
pub fn save_rates(save: &SaveTo<'_>, measured: &profile::rates::Rates) -> Result<String, String> {
    let existing = std::fs::read_to_string(&save.path).ok().map(|text| {
        let (kept, dropped) = save.registry.prune(&text);
        for key in dropped {
            tracing::warn!(
                profile = %save.path.display(),
                key = %key,
                "user profile: dropped a key this pie no longer reads"
            );
        }
        kept
    });
    let text = saved_text(existing.as_deref(), measured, &save.provenance);
    let check = save
        .registry
        .parse(&text, "the measured rates", profile::Strict::User);
    check.map_err(|why| {
        format!(
            "{}: the measured rates would not read back: {why}",
            save.path.display()
        )
    })?;
    if let Some(dir) = save.path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    std::fs::write(&save.path, text).map_err(|e| format!("{}: {e}", save.path.display()))?;
    Ok(save.path.display().to_string())
}

/// The commit this pie was built from, as the build stated it in
/// `PIE_COMMIT`, else `commit unknown`: never the source tree's `HEAD` at
/// tune time, which may have moved on since the build. A rates step's
/// provenance names it.
#[must_use]
pub fn build_commit() -> String {
    option_env!("PIE_COMMIT")
        .map_or_else(|| "commit unknown".to_string(), |c| format!("commit {c}"))
}

/// Today's date in UTC, `YYYY-MM-DD`.
fn today() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let (y, m, d) = civil(i64::try_from(secs / 86_400).unwrap_or(0));
    format!("{y:04}-{m:02}-{d:02}")
}

/// The proleptic Gregorian date `days` after 1970-01-01 (Hinnant's
/// `civil_from_days`).
fn civil(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}


/// A device's clock read off a reference kernel (a backend's own: engine-gpu's
/// smallest shape on the matmul curve, CUDA's point's own body): the fastest
/// of a few single fires and their spread, `(slowest - fastest) / fastest`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClockRead {
    pub fastest_ns: f64,
    pub spread: f64,
}

/// Reads of `read` taken until two in a row agree within their own spread
/// (either way: a ramp still rising and a boost still falling both move by
/// more): the settled clock, the median of the settled pair, at the wider
/// spread. No process-wide fastest is kept: the reference is the clock
/// measured right around the phase. A clock settles (a ramp has a top, a
/// boost a sustained floor), so the reads end.
pub fn warmed<E>(read: impl FnMut() -> Result<ClockRead, E>) -> Result<ClockRead, E> {
    warmed_until(read, &|| false).map(|read| read.expect("never stopped"))
}

/// [`warmed`] behind serving, `stop` read before each read (a request
/// arrived): `None` where it stopped, so an arrival waits at most one read.
pub fn warmed_until<E>(
    mut read: impl FnMut() -> Result<ClockRead, E>,
    stop: &dyn Fn() -> bool,
) -> Result<Option<ClockRead>, E> {
    if stop() {
        return Ok(None);
    }
    let mut last = read()?;
    loop {
        if stop() {
            return Ok(None);
        }
        let next = read()?;
        let noise = last.spread.max(next.spread);
        let low = last.fastest_ns.min(next.fastest_ns);
        let high = last.fastest_ns.max(next.fastest_ns);
        if high <= low * (1.0 + noise) {
            return Ok(Some(ClockRead {
                fastest_ns: (low + high) / 2.0,
                spread: noise,
            }));
        }
        last = next;
    }
}

/// Whether a point's clock moved under it: its own warmed `before` read
/// and its `after` read differ, either way, by more than their own spread
/// (a boost ending, a ramp still rising, a sag): `Some(after over
/// before)`. A point timed across a clock transition measures the
/// transition, not the kernel, and is discarded.
#[must_use]
pub fn clock_moved(before: ClockRead, after: ClockRead) -> Option<f64> {
    let noise = before.spread.max(after.spread);
    let ratio = after.fastest_ns / before.fastest_ns.max(f64::MIN_POSITIVE);
    (ratio > 1.0 + noise || ratio < 1.0 / (1.0 + noise)).then_some(ratio)
}

/// Whether a measuring phase ran throttled: its reference kernel read
/// `before` it (warmed, a backend's warmed read, [`warmed`]) and `after` it. Throttling builds
/// during a phase: throttled where the after read is slower than the
/// before by more than the reads' own spread (the reference's noise, no
/// margin chosen): `Some(after over before)`.
#[must_use]
pub fn throttled(before: ClockRead, after: ClockRead) -> Option<f64> {
    let noise = before.spread.max(after.spread);
    let ratio = after.fastest_ns / before.fastest_ns.max(f64::MIN_POSITIVE);
    (ratio > 1.0 + noise).then_some(ratio)
}

#[cfg(test)]
mod tests {
    /// Points whose fires are slow on any operand set the device has not
    /// read yet, their rotation longer than the frames their doublings
    /// settle in: the first point over the sets touches every one of them,
    /// so it is timed over touched sets only, at their rate, and a second
    /// point over the same sets touches none again. (On an M5 Pro a 256 x
    /// 1024 x 1024 matmul over 384 fresh planes read about 5 or about 30
    /// TFLOP/s run to run, by whether its timed fires fell on planes read
    /// before.)
    #[test]
    fn a_set_is_touched_once_by_the_first_point_that_rotates_it() {
        const SETS: usize = 384;
        // A fire on a set read before takes 1 ns, its first one 20 ns.
        let rig =
            |touched: &std::cell::RefCell<Vec<bool>>, turn: &super::Rotation, k: u32| -> f64 {
                (0..k)
                    .map(|_| {
                        let at = turn.next(SETS);
                        if std::mem::replace(&mut touched.borrow_mut()[at], true) {
                            1.0
                        } else {
                            20.0
                        }
                    })
                    .sum()
            };
        let touched = std::cell::RefCell::new(vec![false; SETS]);
        let owner = super::Touched::default();
        let fires = std::cell::Cell::new(0u32);
        let turn = super::Rotation::default();
        let frame = |k: u32| -> Result<f64, String> {
            fires.set(fires.get() + k);
            Ok(rig(&touched, &turn, k))
        };
        let mut first = frame;
        super::touch_once(&owner, None, SETS as u32, &mut first).expect("it touches");
        assert!(touched.borrow().iter().all(|&t| t), "every set was touched");
        assert!(owner.get(), "the owner holds that its sets are touched");
        let frames = super::warmed_frames(None, None, frame, |why| why).expect("it settles");
        assert_eq!(
            frames.per_fire_ns, 1.0,
            "the point settles at the touched sets' rate"
        );
        let next = frame(frames.fires).expect("a frame") / f64::from(frames.fires);
        assert_eq!(next, 1.0, "the timed frames read touched sets only");
        // A second point over the same sets: no touch again.
        fires.set(0);
        let mut second = frame;
        super::touch_once(&owner, None, SETS as u32, &mut second).expect("it touches");
        assert_eq!(
            fires.get(),
            0,
            "a later point over the same sets touches none again"
        );
        // Untouched (the doublings alone), the point times first touches:
        // the case this rule closes.
        let fresh = std::cell::RefCell::new(vec![false; SETS]);
        let turn = super::Rotation::default();
        let frame = |k: u32| -> Result<f64, String> { Ok(rig(&fresh, &turn, k)) };
        let untouched =
            super::timed_frames(None, None, frame, |why: String| why).expect("it settles");
        assert_eq!(
            untouched.per_fire_ns, 20.0,
            "untouched, the doublings time first touches"
        );
    }

    /// A point that starts in a slow clock state and leaves it only under
    /// its own load: its frames' length settles while it is slow, and the
    /// point is timed only once frames of that length run at the fast rate,
    /// whatever the doublings saw. (On an M5 Pro a short matmul read about
    /// 5 or about 34 TFLOP/s run to run: timed without this, by which state
    /// the gap before it left.)
    #[test]
    fn a_point_is_timed_only_once_its_own_load_has_settled_its_clock() {
        let calls = std::cell::Cell::new(0u32);
        // Slow (10 ns a fire) for the frames the doublings settle in, then
        // fast (1 ns a fire) once its own load has lifted the clock.
        const SLOW: u32 = 3;
        let frame = |k: u32| -> Result<f64, String> {
            calls.set(calls.get() + 1);
            let per = if calls.get() <= SLOW { 10.0 } else { 1.0 };
            Ok(per * f64::from(k))
        };
        let settled =
            super::timed_frames(None, None, frame, |why: String| why).expect("it settles");
        assert_eq!(
            settled.per_fire_ns, 10.0,
            "the doublings alone settle in the slow state"
        );
        calls.set(0);
        let frames = super::warmed_frames(None, None, frame, |why| why).expect("it settles");
        let settled_at = calls.get();
        assert!(
            settled_at > SLOW + 1,
            "the warm-up ran past the slow state ({settled_at} frames)"
        );
        // The next frame of that length reads the fast state.
        let next = frame(frames.fires).expect("a frame") / f64::from(frames.fires);
        assert_eq!(next, 1.0, "the point is timed at the clock its load holds");
    }

    use super::*;

    /// A profile of the toy registry stating `rates`.
    fn chip(rates: profile::rates::Rates) -> profile::Profile {
        let p = profile::Registry::new("toy", &[], &[]).defaults();
        let mut tuned = p.tuned().clone();
        tuned.rates = rates;
        p.with_tuned(tuned)
    }

    fn all() -> profile::rates::Rates {
        profile::rates::Rates {
            launch_ns: Some(900),
            gemm_tflops_by_rows: Some(vec![(4096, vec![(256, 100.0), (8192, 200.0)])]),
            attn_tflops_by_keys: Some(vec![(512, 50.0), (8192, 150.0)]),
            hbm_gbps: Some(700.0),
            cached_gbps: Some(20000.0),
            elementwise_gbps: Some(500.0),
            scan_gelems: Some(2000.0),
            few_row_tflops: Some(4.0),
            workgroups_per_core: Some(3.3),
            ..profile::rates::Rates::default()
        }
    }

    /// This build on a device, as a keyed load checks a stored measurement
    /// against (`profile::rates::Now`): each kernel's code, each method one.
    struct Build {
        on: &'static str,
        code: Vec<(&'static str, u32)>,
    }

    impl profile::rates::Now for Build {
        fn on(&self) -> String {
            self.on.to_string()
        }
        fn method(&self, _: &str) -> Option<u32> {
            Some(1)
        }
        fn code(&self, kernel: &str) -> Option<u32> {
            self.code
                .iter()
                .find(|(k, _)| *k == kernel)
                .map(|&(_, c)| c)
        }
    }

    /// What a rates step measures under `build`, keyed: `gemm_tflops_by_rows`
    /// fires the `gemm` kernel, every other rate the `copy` kernel.
    fn keyed(build: &Build) -> profile::rates::Rates {
        use profile::rates::Now as _;
        let by = |kernel: &str| profile::rates::MeasuredBy {
            method: 1,
            kernels: vec![(kernel.to_string(), build.code(kernel).unwrap())],
        };
        profile::rates::Rates {
            measured_on: Some(build.on()),
            measured_by: Some(
                profile::rates::KEYS
                    .iter()
                    .map(|&key| {
                        let kernel = if key == "gemm_tflops_by_rows" {
                            "gemm"
                        } else {
                            "copy"
                        };
                        (key.to_string(), by(kernel))
                    })
                    .collect(),
            ),
            ..all()
        }
    }

    /// A load under `measure = false` measures nothing and saves nothing:
    /// what a layer states serves, each key it lacked is kept as skipped.
    #[test]
    fn a_skipping_load_measures_nothing_and_names_each_skip() {
        let stated = all().only(&["launch_ns", "hbm_gbps"]);
        let loaded = resolve_at_load(
            chip(stated.clone()),
            &profile::rates::KEYS,
            None,
            None,
            None,
            Measuring::Skip,
            StaleRates::Measured,
            |_, _| -> Result<_, String> { panic!("measured under skip") },
        )
        .unwrap();
        assert!(!loaded.tuned().measured_this_load);
        assert_eq!(loaded.rates(), stated);
        assert_eq!(
            loaded.tuned().skipped.len(),
            profile::rates::KEYS.len() - 2,
            "{:?}",
            loaded.tuned().skipped
        );
    }

    /// Loads of one home under keyed measurements: a second load of another
    /// pie version (another provenance, the same kernels) measures nothing;
    /// a load whose `gemm` kernel changed measures that rate alone; a load
    /// on another device measures every one.
    #[test]
    fn keyed_loads_measure_only_what_their_keys_no_longer_match() {
        static SHELF: profile::Shelf = profile::Shelf {
            backend: "toy",
            seeded: &[],
            derived: None,
        };
        let registry = profile::Registry::new("toy", &[], &[]);
        let facts = profile::Facts::of(&registry);
        let dir = std::env::temp_dir().join(format!("pie-rates-keyed-{}", std::process::id()));
        let path = SHELF.user_path(&dir, "Card X");
        let save = |version: &str| SaveTo {
            path: path.clone(),
            provenance: ("rates", format!("\"pie {version}: a test\"")),
            registry: &registry,
        };
        let build = Build {
            on: "toy 10de:2c31 driver 1",
            code: vec![("gemm", 7), ("copy", 9)],
        };
        // Each load: the keys it measured (none where it measured nothing).
        let load = |build: &Build, version: &str| -> Vec<&'static str> {
            let (chip, _) = SHELF.resolve_keyed(&registry, "Card X", &facts, Some(&dir), build);
            let lacked = chip.rates().missing(&profile::rates::KEYS);
            rates_at_load(
                chip,
                &profile::rates::KEYS,
                Some(save(version)),
                None,
                None,
                |_| {
                    Ok(Measured {
                        rates: keyed(build).only(&lacked),
                        unsettled: false,
                        throttled: None,
                    })
                },
            )
            .unwrap();
            lacked
        };
        assert_eq!(
            load(&build, "0.6.0"),
            profile::rates::KEYS,
            "a first load measures all"
        );
        assert_eq!(
            load(&build, "0.7.0"),
            Vec::<&str>::new(),
            "a version bump measures nothing"
        );
        let changed = Build {
            code: vec![("gemm", 8), ("copy", 9)],
            ..build
        };
        assert_eq!(
            load(&changed, "0.7.1"),
            ["gemm_tflops_by_rows"],
            "a changed kernel measures its rates alone"
        );
        let other = Build {
            on: "toy 10de:2c32 driver 1",
            ..changed
        };
        assert_eq!(
            load(&other, "0.7.2"),
            profile::rates::KEYS,
            "another device measures all"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A checked-in file stating a rate this pie does not read is refused
    /// whole, and the load is refused by its name and the key.
    #[test]
    fn a_checked_in_file_with_a_stale_rate_refuses_the_load_by_name() {
        static SHELF: profile::Shelf = profile::Shelf {
            backend: "toy",
            seeded: &[(
                "derived/card-x.toml",
                "api = 1\n[target]\ndevice = \"Card X\"\n[rates]\nbytes_per_ns = 650.0\n",
            )],
            derived: Some(|_, _| "api = 1\n".to_string()),
        };
        let registry = profile::Registry::new("toy", &[], &[]);
        let facts = profile::Facts::of(&registry);
        let (chip, _) = SHELF.resolve(&registry, "Card X", &facts, None);
        let err = rates_at_load(chip, &profile::rates::KEYS, None, None, None, |_| {
            Ok(Measured {
                rates: all(),
                unsettled: false,
                throttled: None,
            })
        })
        .unwrap_err();
        assert!(err.contains("derived/card-x.toml"), "{err}");
        assert!(err.contains("`[rates] bytes_per_ns`"), "{err}");
    }

    #[test]
    fn a_budget_short_of_the_step_is_refused_by_name_and_measures_nothing() {
        let short = step_bytes() - 1;
        let err = rates_at_load(
            chip(profile::rates::Rates::default()),
            &profile::rates::KEYS,
            None,
            Some(short),
            None,
            |_| -> Result<_, String> { panic!("measured past a short budget") },
        )
        .unwrap_err();
        assert!(err.contains("measuring them needs"), "{err}");
    }

    #[test]
    fn a_rate_that_settles_at_nothing_is_refused_by_name_and_never_saved() {
        let dir = std::env::temp_dir().join(format!("pie-rates-zero-{}", std::process::id()));
        let path = dir.join("vulkan/user/card-x.toml");
        let registry = profile::Registry::new("toy", &[], &[]);
        for (zero, says) in [
            (
                profile::rates::Rates {
                    workgroups_per_core: Some(0.0),
                    ..all()
                },
                "`[rates] workgroups_per_core` as 0",
            ),
            (
                profile::rates::Rates {
                    attn_tflops_by_keys: Some(vec![(512, 0.0), (8192, 150.0)]),
                    ..all()
                },
                "`[rates] attn_tflops_by_keys` at 512 keys as 0",
            ),
        ] {
            let save = SaveTo {
                path: path.clone(),
                provenance: ("rates", "\"measured at load\"".to_string()),
                registry: &registry,
            };
            let err = rates_at_load(
                chip(profile::rates::Rates::default()),
                &profile::rates::KEYS,
                Some(save),
                None,
                None,
                |_| {
                    Ok(Measured {
                        rates: zero,
                        unsettled: false,
                        throttled: None,
                    })
                },
            )
            .unwrap_err();
            assert!(err.contains(says), "{err}");
            assert!(!path.exists(), "a refused rate was saved");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

}
