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
/// than 2. (engine-cuda's sizing, 63876a684: a few-key point's page is
/// laid whole, so its sets are counted by it.)
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

    /// The touch keeps each of its frames under the device's bound on one
    /// submission, and still covers every set once.
    #[test]
    fn the_touch_keeps_under_the_submission_bound() {
        let fires = std::cell::Cell::new(0u32);
        let longest = std::cell::Cell::new(0u32);
        let mut frame = |k: u32| -> Result<f64, String> {
            fires.set(fires.get() + k);
            longest.set(longest.get().max(k));
            Ok(f64::from(k) * 1000.0)
        };
        super::touch_sets(Some(std::time::Duration::from_micros(10)), 95, &mut frame)
            .expect("it touches");
        assert_eq!(fires.get(), 95, "every set touched once");
        assert!(
            longest.get() <= 10,
            "no frame past the bound ({} fires)",
            longest.get()
        );
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
            stale: None,
            landing: None,
            launch_ns: Some(900),
            gemm_tflops_by_rows: Some(vec![(4096, vec![(256, 100.0), (8192, 200.0)])]),
            attn_tflops_by_keys: Some(vec![(512, 50.0), (8192, 150.0)]),
            hbm_gbps: Some(700.0),
            cached_gbps: Some(20000.0),
            elementwise_gbps: Some(500.0),
            scan_gelems: Some(2000.0),
            few_row_tflops: Some(4.0),
            workgroups_per_core: Some(3.3),
            submit_worst_ratio: None,
            load_width: None,
            resident_workgroups: None,
            attn_tflops_by_rows_keys: None,
            product_ns: None,
            kernel_curves: None,
            dense_settled: Vec::new(),
            measured_on: None,
            measured_by: None,
        }
    }

    #[test]
    fn a_load_without_the_constants_measures_saves_and_says_so() {
        let dir = std::env::temp_dir().join(format!("pie-rates-at-load-{}", std::process::id()));
        let path = dir.join("vulkan/user/card-x.toml");
        let registry = profile::Registry::new("toy", &[], &[]);
        // A checked-in launch, nothing else: the rest measured, the launch
        // kept, each submission under the budget asked.
        let stated = profile::rates::Rates {
            launch_ns: Some(1234),
            ..profile::rates::Rates::default()
        };
        let mut asked = None;
        let p = rates_at_load(
            chip(stated),
            &profile::rates::KEYS,
            Some(SaveTo {
                path: path.clone(),
                provenance: provenance("toy", "a test"),
                registry: &registry,
            }),
            None,
            Some(std::time::Duration::from_millis(5)),
            |max_submit| {
                asked = max_submit;
                Ok(Measured {
                    rates: all(),
                    unsettled: true,
                    throttled: None,
                })
            },
        )
        .unwrap();
        assert_eq!(asked, Some(std::time::Duration::from_millis(5)));
        assert_eq!(p.rates().launch_ns, Some(1234), "the stated value kept");
        assert_eq!(p.rates().hbm_gbps, Some(700.0));
        assert!(p.tuned().measured_this_load);
        assert_eq!(p.tuned().saved_to.as_deref(), Some(path.to_str().unwrap()));
        // The file holds what was measured and how, not the stated launch.
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(
            text.contains("gemm_tflops_by_rows = [\n  [4096, [[256, 100.0], [8192, 200.0]]],\n]"),
            "{text}"
        );
        assert!(!text.contains("launch_ns"), "{text}");
        assert!(
            text.contains("[provenance]\nrates = \"pie tune toy --rates on "),
            "{text}"
        );
        // The line says it was measured here and where it went.
        let line = crate::load::measured_line(p.tuned()).unwrap();
        assert!(line.starts_with("measured the roofline at load"), "{line}");
        // A frame the 5 ms bound stopped before it settled: said.
        assert!(p.tuned().frame_cut);
        assert!(
            line.contains("stopped at the device's bound on one submission"),
            "{line}"
        );
        assert!(
            line.ends_with(&format!("saved to {}", path.display())),
            "{line}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_next_load_reads_what_the_first_saved_and_measures_nothing() {
        static SHELF: profile::Shelf = profile::Shelf {
            backend: "toy",
            seeded: &[],
            derived: None,
        };
        let registry = profile::Registry::new("toy", &[], &[]);
        let facts = profile::Facts::of(&registry);
        let dir = std::env::temp_dir().join(format!("pie-rates-next-{}", std::process::id()));
        let (first, _) = SHELF.resolve(&registry, "Card X", &facts, Some(&dir));
        assert!(!first.rates().complete(&profile::rates::KEYS));
        let measured = rates_at_load(
            first,
            &profile::rates::KEYS,
            Some(SaveTo {
                path: SHELF.user_path(&dir, "Card X"),
                provenance: provenance("toy", "a test"),
                registry: &registry,
            }),
            None,
            None,
            |_| {
                Ok(Measured {
                    rates: all(),
                    unsettled: false,
                    throttled: None,
                })
            },
        )
        .unwrap();
        assert!(measured.tuned().measured_this_load);
        // The next load resolves the saved file: every constant stated, so
        // it measures nothing and its line names no measurement.
        let (second, _) = SHELF.resolve(&registry, "Card X", &facts, Some(&dir));
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(second.rates(), all());
        let second = rates_at_load(
            second,
            &profile::rates::KEYS,
            None,
            None,
            None,
            |_| -> Result<_, String> { panic!("measured what the first load saved") },
        )
        .unwrap();
        assert!(!second.tuned().measured_this_load);
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

    /// A shelf of the toy backend carrying `committed`, a committed
    /// measured profile of `Card X` (none where `None`).
    fn shelf_with(committed: Option<String>) -> &'static profile::Shelf {
        let seeded: Vec<(&'static str, &'static str)> = committed
            .map(|text| {
                let name: &'static str = Box::leak(profile::measured_name("Card X").into_boxed_str());
                (name, &*Box::leak(text.into_boxed_str()))
            })
            .into_iter()
            .collect();
        Box::leak(Box::new(profile::Shelf {
            backend: "toy",
            seeded: Box::leak(seeded.into_boxed_slice()),
            derived: None,
        }))
    }

    /// The user's file a load under `build` saves, as `pie tune export`
    /// reads it: every rate measured, each keyed.
    fn stored_text(build: &Build) -> String {
        saved_text(None, &keyed(build), &provenance("toy", "a test"))
    }

    /// One load of `Card X` from `shelf` with the user's files under `dir`:
    /// the keys it measured (none where it measured nothing), each measured
    /// under `build`, and the profile and report it resolved.
    fn load_from(
        shelf: &profile::Shelf,
        dir: &std::path::Path,
        build: &Build,
        measuring: Measuring,
    ) -> (Vec<&'static str>, profile::Profile, profile::Report) {
        load_behind(shelf, dir, build, measuring, StaleRates::Measured)
    }

    /// [`load_from`] by an engine that does `stale_rates` with a stale key.
    fn load_behind(
        shelf: &profile::Shelf,
        dir: &std::path::Path,
        build: &Build,
        measuring: Measuring,
        stale_rates: StaleRates,
    ) -> (Vec<&'static str>, profile::Profile, profile::Report) {
        let registry = profile::Registry::new("toy", &[], &[]);
        let facts = profile::Facts::of(&registry);
        let (chip, report) = shelf.resolve_keyed(&registry, "Card X", &facts, Some(dir), build);
        let measured = std::cell::Cell::new(Vec::new());
        let loaded = resolve_at_load(
            chip,
            &profile::rates::KEYS,
            Some(SaveTo {
                path: shelf.user_path(dir, "Card X"),
                provenance: provenance("toy", "a test"),
                registry: &registry,
            }),
            None,
            None,
            measuring,
            stale_rates,
            |_, stated| {
                let lacked = stated.missing(&profile::rates::KEYS);
                measured.set(lacked.clone());
                Ok(Measured {
                    rates: keyed(build).only(&lacked),
                    unsettled: false,
                    throttled: None,
                })
            },
        )
        .unwrap();
        (measured.take(), loaded, report)
    }

    /// A kernel whose code changed since the committed profile was
    /// exported, on an engine that times behind serving: nothing is
    /// measured before ready; the stale figure serves, and the key is left
    /// to time behind serving.
    #[test]
    fn a_changed_kernel_is_served_by_its_stale_figure_and_timed_behind() {
        let exported = Build {
            on: "toy 10de:2c31 driver 1",
            code: vec![("gemm", 7), ("copy", 9)],
        };
        let registry = profile::Registry::new("toy", &[], &[]);
        let committed =
            profile::export(&registry, "Card X", &stored_text(&exported), "user.toml").unwrap();
        let shelf = shelf_with(Some(committed));
        let dir = home("committed-stale-behind");
        let now = Build {
            code: vec![("gemm", 8), ("copy", 9)],
            ..exported
        };
        let (measured, loaded, report) = load_behind(shelf, &dir, &now, Measuring::Time, StaleRates::TimedBehind);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(measured, Vec::<&str>::new(), "nothing before ready: {:#?}", report.stored);
        assert_eq!(loaded.rates().gemm_tflops_by_rows, all().gemm_tflops_by_rows);
        assert_eq!(loaded.tuned().behind, ["gemm_tflops_by_rows"]);
        assert!(!loaded.tuned().measured_this_load);
    }

    fn home(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("pie-rates-{name}-{}", std::process::id()))
    }

    /// A clean home whose device pie carries a committed measured profile
    /// of, under matching keys: nothing is measured or saved, every entry
    /// taken from the committed profile and said so, by its key.
    #[test]
    fn a_committed_entry_whose_key_matches_is_taken_and_nothing_measured() {
        let build = Build {
            on: "toy 10de:2c31 driver 1",
            code: vec![("gemm", 7), ("copy", 9)],
        };
        let registry = profile::Registry::new("toy", &[], &[]);
        let committed =
            profile::export(&registry, "Card X", &stored_text(&build), "user.toml").unwrap();
        let shelf = shelf_with(Some(committed));
        let dir = home("committed-taken");
        let (measured, loaded, report) = load_from(shelf, &dir, &build, Measuring::Time);
        assert_eq!(measured, Vec::<&str>::new(), "{:#?}", report.stored);
        assert_eq!(loaded.rates().launch_ns, all().launch_ns);
        assert!(!loaded.tuned().measured_this_load);
        assert!(!shelf.user_path(&dir, "Card X").exists(), "nothing is saved");
        assert!(
            report.stored.iter().any(|l| l.starts_with(
                "`gemm_tflops_by_rows`: committed (committed measured profile \
                 card-x.measured.toml), its key matches: method v1, kernels [gemm 0x00000007]"
            )),
            "{:#?}",
            report.stored
        );
        assert_eq!(
            report
                .layers
                .last()
                .cloned()
                .unwrap_or(profile::Source::Defaults),
            profile::Source::Committed("card-x.measured.toml".into())
        );
    }

    /// A kernel whose code changed since the committed profile was
    /// exported: its entries alone are measured (and saved to the user's
    /// file), every other taken from the committed profile; the next load
    /// takes the measured entry from the user's file and measures nothing.
    #[test]
    fn a_changed_kernel_measures_only_its_committed_entries() {
        let exported = Build {
            on: "toy 10de:2c31 driver 1",
            code: vec![("gemm", 7), ("copy", 9)],
        };
        let registry = profile::Registry::new("toy", &[], &[]);
        let committed =
            profile::export(&registry, "Card X", &stored_text(&exported), "user.toml").unwrap();
        let shelf = shelf_with(Some(committed));
        let dir = home("committed-changed");
        let now = Build {
            code: vec![("gemm", 8), ("copy", 9)],
            ..exported
        };
        let (measured, _, report) = load_from(shelf, &dir, &now, Measuring::Time);
        assert_eq!(measured, ["gemm_tflops_by_rows"], "{:#?}", report.stored);
        assert!(
            report.stored.iter().any(|l| l.contains(
                "committed `gemm_tflops_by_rows`: not taken as measured, kernel `gemm`'s code changed"
            )),
            "{:#?}",
            report.stored
        );
        let (again, loaded, report) = load_from(shelf, &dir, &now, Measuring::Time);
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(again, Vec::<&str>::new(), "{:#?}", report.stored);
        assert!(
            report
                .stored
                .iter()
                .any(|l| l.starts_with("`gemm_tflops_by_rows`: stored (user profile")),
            "{:#?}",
            report.stored
        );
        assert!(
            report
                .stored
                .iter()
                .any(|l| l.starts_with("`launch_ns`: committed")),
            "{:#?}",
            report.stored
        );
        assert_eq!(loaded.rates().complete(&profile::rates::KEYS), true);
        // Another driver: the committed profile is none of this device's.
        let other = Build {
            on: "toy 10de:2c31 driver 2",
            ..now
        };
        let dir = home("committed-driver");
        let (measured, _, _) = load_from(shelf, &dir, &other, Measuring::Time);
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(measured, profile::rates::KEYS);
    }

    /// The user's stored entry under a matching key is taken over the
    /// committed one; where the user's key no longer matches, the committed
    /// entry whose key does is taken.
    #[test]
    fn a_users_entry_is_taken_over_a_committed_one() {
        let build = Build {
            on: "toy 10de:2c31 driver 1",
            code: vec![("gemm", 7), ("copy", 9)],
        };
        let registry = profile::Registry::new("toy", &[], &[]);
        let committed = profile::export(
            &registry,
            "Card X",
            &saved_text(
                None,
                &profile::rates::Rates {
                    launch_ns: Some(2222),
                    ..keyed(&build)
                },
                &provenance("toy", "a test"),
            ),
            "user.toml",
        )
        .unwrap();
        let shelf = shelf_with(Some(committed));
        let dir = home("user-over-committed");
        let path = shelf.user_path(&dir, "Card X");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut user = profile::rates::Rates {
            launch_ns: Some(1111),
            ..keyed(&build)
        }
        .only(&["launch_ns"]);
        user.measured_on = Some("toy 10de:2c31 driver 1".into());
        std::fs::write(&path, saved_text(None, &user, &provenance("toy", "a test"))).unwrap();
        let (measured, loaded, report) = load_from(shelf, &dir, &build, Measuring::Time);
        assert_eq!(measured, Vec::<&str>::new());
        assert_eq!(loaded.rates().launch_ns, Some(1111), "{:#?}", report.stored);
        assert!(
            report
                .stored
                .iter()
                .any(|l| l.starts_with("`launch_ns`: stored (user profile")),
            "{:#?}",
            report.stored
        );
        // The user's launch keyed by a kernel whose code changed since: the
        // committed launch, whose key matches, is taken.
        user.measured_by = Some(vec![(
            "launch_ns".into(),
            profile::rates::MeasuredBy {
                method: 1,
                kernels: vec![("copy".into(), 8)],
            },
        )]);
        std::fs::write(&path, saved_text(None, &user, &provenance("toy", "a test"))).unwrap();
        let (measured, loaded, report) = load_from(shelf, &dir, &build, Measuring::Time);
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(measured, Vec::<&str>::new());
        assert_eq!(loaded.rates().launch_ns, Some(2222), "{:#?}", report.stored);
        assert!(
            report
                .stored
                .iter()
                .any(|l| l.contains("stored `launch_ns`: not taken as measured, kernel `copy`'s code changed")),
            "{:#?}",
            report.stored
        );
    }

    /// `pie tune export` of a user's file reads back to the entries it
    /// stores, the same text whatever order they were stored in, with no
    /// date or path; an entry stored without a key is refused by name.
    #[test]
    fn an_export_reads_back_the_entries_it_stores() {
        let build = Build {
            on: "toy 10de:2c31 driver 1",
            code: vec![("gemm", 7), ("copy", 9)],
        };
        let registry = profile::Registry::new("toy", &[], &[]);
        let mut rates = keyed(&build);
        rates.kernel_curves = Some(vec![
            profile::curves::KernelCurve {
                kind: profile::curves::KernelCurve::TILE,
                key: 64,
                n: 1024,
                k: 1024,
                curve: vec![(1, 0.5), (64, 9.0)],
                stop: None,
                spread: vec![(1, 0.01), (64, 0.02)],
                by: profile::rates::MeasuredBy {
                    method: 1,
                    kernels: vec![("gemm".into(), 7)],
                },
            },
            profile::curves::KernelCurve {
                kind: profile::curves::KernelCurve::VECTOR,
                key: 4,
                n: 1024,
                k: 1024,
                curve: vec![(1, 0.3)],
                stop: Some((16, profile::curves::KernelCurve::BEATEN)),
                spread: vec![(1, 0.01)],
                by: profile::rates::MeasuredBy {
                    method: 1,
                    kernels: vec![("gemm".into(), 7)],
                },
            },
        ]);
        rates.dense_settled = vec![
            profile::curves::DenseSettled {
                method: 1,
                of: 5,
                code: 6,
                ranges: vec![[1, 1, 0, 0, 0, 0]],
                planes: vec![(1024, 1024)],
            },
            profile::curves::DenseSettled {
                method: 1,
                of: 9,
                code: 6,
                ranges: vec![[1, 1, 1, 4, 0, 1]],
                planes: vec![(1024, 1024)],
            },
        ];
        let user = saved_text(None, &rates, &provenance("toy", "a test"));
        let text = profile::export(&registry, "Card X", &user, "user.toml").unwrap();
        let read = registry
            .parse(&text, "card-x.measured.toml", profile::Strict::Measured)
            .unwrap();
        let stored = registry
            .parse(&user, "user.toml", profile::Strict::User)
            .unwrap();
        assert_eq!(read.tuned.rates, stored.tuned.rates.clone().ordered());
        // Stored in another order, the same text.
        let mut shuffled = rates.clone();
        shuffled.measured_by.as_mut().unwrap().reverse();
        shuffled.kernel_curves.as_mut().unwrap().reverse();
        let again = saved_text(None, &shuffled, &provenance("toy", "another day"));
        assert_eq!(
            profile::export(&registry, "Card X", &again, "elsewhere.toml").unwrap(),
            text
        );
        assert!(!text.contains(&today()), "{text}");
        assert!(!text.contains("user.toml"), "{text}");
        // A decision file is refused the measured profile's keys; a
        // measured profile is refused a decision.
        assert!(registry
            .parse(&text, "x.toml", profile::Strict::CheckedIn)
            .is_err());
        // An entry stored without a key.
        let mut unkeyed = rates.clone();
        unkeyed
            .measured_by
            .as_mut()
            .unwrap()
            .retain(|(rate, _)| rate != "hbm_gbps");
        let refused = profile::export(
            &registry,
            "Card X",
            &saved_text(None, &unkeyed, &provenance("toy", "a test")),
            "user.toml",
        )
        .unwrap_err();
        assert!(
            refused.contains("`rates.hbm_gbps` is stored without a key"),
            "{refused}"
        );
        let mut nowhere = rates;
        nowhere.measured_on = None;
        let refused = profile::export(
            &registry,
            "Card X",
            &saved_text(None, &nowhere, &provenance("toy", "a test")),
            "user.toml",
        )
        .unwrap_err();
        assert!(refused.contains("names no device"), "{refused}");
    }

    /// A load under `measure = false` with no profile measures nothing,
    /// saves nothing and loads, each key it lacked kept as skipped, by name.
    #[test]
    fn a_skipping_load_measures_nothing_and_names_each_skip() {
        let build = Build {
            on: "toy 10de:2c31 driver 1",
            code: vec![("gemm", 7), ("copy", 9)],
        };
        let shelf = shelf_with(None);
        let dir = home("skip");
        let (measured, loaded, report) = load_from(shelf, &dir, &build, Measuring::Skip);
        assert_eq!(measured, Vec::<&str>::new());
        assert!(!shelf.user_path(&dir, "Card X").exists(), "nothing is saved");
        assert_eq!(loaded.tuned().skipped, profile::rates::KEYS, "{report:?}");
        assert!(!loaded.tuned().measured_this_load);
        assert_eq!(loaded.rates(), profile::rates::Rates::default());
        // A committed profile still serves what it states under skip.
        let registry = profile::Registry::new("toy", &[], &[]);
        let committed = profile::export(
            &registry,
            "Card X",
            &saved_text(
                None,
                &keyed(&build).only(&["launch_ns", "hbm_gbps"]),
                &provenance("toy", "a test"),
            ),
            "user.toml",
        )
        .unwrap();
        let (measured, loaded, _) =
            load_from(shelf_with(Some(committed)), &dir, &build, Measuring::Skip);
        assert_eq!(measured, Vec::<&str>::new());
        assert_eq!(loaded.rates().launch_ns, all().launch_ns);
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
            let (chip, report) =
                SHELF.resolve_keyed(&registry, "Card X", &facts, Some(&dir), build);
            assert!(
                !report.stored.is_empty(),
                "every keyed load says what it found"
            );
            let lacked = chip.rates().missing(&profile::rates::KEYS);
            let measures = std::cell::Cell::new(0);
            rates_at_load(
                chip,
                &profile::rates::KEYS,
                Some(save(version)),
                None,
                None,
                |_| {
                    measures.set(measures.get() + 1);
                    Ok(Measured {
                        rates: keyed(build).only(&lacked),
                        unsettled: false,
                        throttled: None,
                    })
                },
            )
            .unwrap();
            assert_eq!(measures.get(), usize::from(!lacked.is_empty()));
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
        assert_eq!(load(&changed, "0.7.2"), Vec::<&str>::new());
        let other = Build {
            on: "toy 10de:2c32 driver 1",
            ..changed
        };
        assert_eq!(
            load(&other, "0.7.2"),
            profile::rates::KEYS,
            "another device measures all"
        );
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(
            text.contains("measured_on = \"toy 10de:2c32 driver 1\""),
            "{text}"
        );
        assert!(
            !text.contains("2c31"),
            "nothing of the other device stays: {text}"
        );
    }

    /// A saved profile lacking one key a newer pie reads: the load asks for
    /// that key alone and the file keeps every other rate as it was.
    #[test]
    fn a_profile_lacking_one_key_gains_it_and_keeps_the_rest() {
        static SHELF: profile::Shelf = profile::Shelf {
            backend: "toy",
            seeded: &[],
            derived: None,
        };
        let registry = profile::Registry::new("toy", &[], &[]);
        let facts = profile::Facts::of(&registry);
        let dir = std::env::temp_dir().join(format!("pie-rates-one-key-{}", std::process::id()));
        let path = SHELF.user_path(&dir, "Card X");
        let save = || SaveTo {
            path: path.clone(),
            provenance: provenance("toy", "a test"),
            registry: &registry,
        };
        let older = profile::rates::Rates {
            few_row_tflops: None,
            ..all()
        };
        // An older pie, which read every key but this one, saved the file.
        let older_keys: Vec<&'static str> = profile::rates::KEYS
            .iter()
            .copied()
            .filter(|&key| key != "few_row_tflops")
            .collect();
        let (first, _) = SHELF.resolve(&registry, "Card X", &facts, Some(&dir));
        rates_at_load(first, &older_keys, Some(save()), None, None, |_| {
            Ok(Measured {
                rates: older.clone(),
                unsettled: false,
                throttled: None,
            })
        })
        .unwrap();
        let before = std::fs::read_to_string(&path).unwrap();
        let (second, _) = SHELF.resolve(&registry, "Card X", &facts, Some(&dir));
        assert_eq!(
            second.rates().missing(&profile::rates::KEYS),
            ["few_row_tflops"]
        );
        let lacked = second.rates().missing(&profile::rates::KEYS);
        let loaded = rates_at_load(
            second,
            &profile::rates::KEYS,
            Some(save()),
            None,
            None,
            |_| {
                // A backend times only what the profile lacks.
                Ok(Measured {
                    rates: all().only(&lacked),
                    unsettled: false,
                    throttled: None,
                })
            },
        )
        .unwrap();
        let after = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(loaded.rates(), all());
        let rates_of = |text: &str| -> Vec<String> {
            text.lines()
                .skip_while(|line| *line != "[rates]")
                .take_while(|line| !line.starts_with("[provenance]"))
                .filter(|line| !line.starts_with("few_row_tflops"))
                .map(str::to_owned)
                .collect()
        };
        assert_eq!(rates_of(&before), rates_of(&after));
        assert!(after.contains("few_row_tflops = 4.0"), "{after}");
    }

    #[test]
    fn a_saved_file_gains_the_measured_keys_and_their_provenance_alone() {
        let measured = profile::rates::Rates {
            hbm_gbps: Some(700.0),
            cached_gbps: None,
            ..profile::rates::Rates::default()
        };
        let how = ("rates", "\"measured at load\"".to_string());
        // No file yet: the keys and how, nothing else.
        assert_eq!(
            saved_text(None, &measured, &how),
            "api = 1\n\n[rates]\nhbm_gbps = 700.0\n\n[provenance]\nrates = \"measured at load\"\n"
        );
        // A user's file: its decisions kept, its [rates] written whole (the
        // key it stated kept under the measured one), its provenance's rates
        // entry replaced, the rest left.
        let existing = "api = 1\n[engine]\ngraphs = \"on\"\n\n[rates]\nlaunch_ns = 900\n\n[provenance]\nsource = \"me\"\nrates = \"old\"\n";
        assert_eq!(
            saved_text(Some(existing), &measured, &how),
            "api = 1\n[engine]\ngraphs = \"on\"\n\n[rates]\nlaunch_ns = 900\nhbm_gbps = 700.0\n\n[provenance]\nsource = \"me\"\nrates = \"measured at load\"\n"
        );
        // A rate stated as array rows: the table read whole and rewritten.
        let rows =
            "api = 1\n\n[rates]\nattn_tflops_by_keys = [\n  [512, 4.0],\n  [1024, 5.0],\n]\n";
        assert_eq!(
            saved_text(Some(rows), &measured, &how),
            "api = 1\n\n[rates]\nattn_tflops_by_keys = [[512, 4.0], [1024, 5.0]]\nhbm_gbps = 700.0\n\n[provenance]\nrates = \"measured at load\"\n"
        );
        // A table of a schema this pie outlived (Nov 16's keys): replaced
        // whole, none of it kept, so no key is written twice.
        let old = "api = 1\n[engine]\ngraphs = \"on\"\n\n[rates]\nbf16_flops_per_ns_by_rows_n_k = [[16, 1024, 1024, 1.0]]\nbytes_per_ns = 1.0\nlaunch_ns = 900\n\n[provenance]\nrates = \"old\"\n";
        let all = profile::rates::Rates {
            launch_ns: Some(1200),
            ..measured.clone()
        };
        assert_eq!(
            saved_text(Some(old), &all, &how),
            "api = 1\n[engine]\ngraphs = \"on\"\n\n[rates]\nlaunch_ns = 1200\nhbm_gbps = 700.0\n\n[provenance]\nrates = \"measured at load\"\n"
        );
    }

    /// The Spark's user file of Nov 16 (the old schema's [rates] keys and a
    /// decision this pie removed) is refused at resolve, so the load
    /// measures every key; the file is rewritten as the load's own output
    /// (the removed decision dropped and said, the old table replaced
    /// whole, what reads kept), and the next boot reads it clean and
    /// measures nothing.
    #[test]
    fn an_old_schema_user_file_is_measured_rewritten_whole_and_read_next() {
        static SHELF: profile::Shelf = profile::Shelf {
            backend: "toy",
            seeded: &[],
            derived: None,
        };
        let registry = profile::Registry::new("toy", &[], &[]);
        let facts = profile::Facts::of(&registry);
        let dir = std::env::temp_dir().join(format!("pie-rates-old-{}", std::process::id()));
        let path = SHELF.user_path(&dir, "Card X");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            &path,
            "api = 1\n\n[engine]\ngraphs = \"on\"\n\n[rates]\n\
             bf16_flops_per_ns_by_rows_n_k = [[16, 1024, 1024, 1.0]]\nbytes_per_ns = 650.0\n\
             launch_ns = 900\n\n[provenance]\nsource = \"me\"\nrates = \"Nov 16\"\n",
        )
        .unwrap();
        let (first, report) = SHELF.resolve(&registry, "Card X", &facts, Some(&dir));
        assert!(!report.rejected.is_empty(), "the old file is refused");
        assert!(!first.rates().complete(&profile::rates::KEYS));
        let measured = rates_at_load(
            first,
            &profile::rates::KEYS,
            Some(SaveTo {
                path: path.clone(),
                provenance: provenance("toy", "a test"),
                registry: &registry,
            }),
            None,
            None,
            |_| {
                Ok(Measured {
                    rates: all(),
                    unsettled: false,
                    throttled: None,
                })
            },
        )
        .unwrap();
        assert!(measured.tuned().measured_this_load);
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.matches("launch_ns").count(), 1, "{text}");
        assert!(!text.contains("bytes_per_ns"), "{text}");
        // The decision this pie no longer reads dropped; what reads kept.
        assert!(!text.contains("graphs"), "{text}");
        assert!(text.contains("source = \"me\""), "{text}");
        let (second, report) = SHELF.resolve(&registry, "Card X", &facts, Some(&dir));
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(report.rejected.is_empty(), "{:?}", report.rejected);
        assert_eq!(second.rates(), all());
        rates_at_load(
            second,
            &profile::rates::KEYS,
            None,
            None,
            None,
            |_| -> Result<_, String> { panic!("measured what the first load saved") },
        )
        .unwrap();
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

    /// A cold frame before the clock settles moves nothing: the 840's walk
    /// at 2048^3 took 23.4 ms against 12.3 ms estimated while its clock
    /// ramped, and held every admission after it 1.9x high.
    #[test]
    fn a_cold_frame_does_not_inflate_later_admissions() {
        let mut a = Admission::new(Some(std::time::Duration::from_secs(2)));
        a.landed(Some(12.33e6), 23.43e6, true);
        assert_eq!(a.worst, 1.0);
        // The attention's frame of 4 at 274 ms each fits at 1096 ms.
        assert_eq!(a.fires(4, Some(274.1e6)), Some(4));
        // Settled: its frames count, the largest ratio kept.
        a.settled = true;
        a.landed(Some(100e6), 104e6, true);
        a.landed(Some(100e6), 101e6, true);
        assert!((a.worst - 1.04).abs() < 1e-12);
        // A frame carried down (from more fires to fewer) holds its own
        // submit: it moves nothing.
        a.landed(Some(1e6), 3e6, false);
        assert!((a.worst - 1.04).abs() < 1e-12);
    }

    /// A frame that does not fit halves its fires until it does; only a
    /// single fire past the bound is refused.
    #[test]
    fn a_frame_that_does_not_fit_halves_its_fires() {
        let mut a = Admission::new(Some(std::time::Duration::from_secs(2)));
        a.settled = true;
        a.worst = 1.9;
        // 4 fires of 274 ms: 2083 ms at 1.9x; 2 fit (1042 ms).
        assert_eq!(a.fires(4, Some(274.1e6)), Some(2));
        assert_eq!(a.fires(16, Some(100e6)), Some(8));
        // One fire past the bound: refused.
        assert_eq!(a.fires(4, Some(1.2e9)), None);
        // No estimate under a bound: not one fire; no bound: the frame as
        // asked.
        assert_eq!(a.fires(4, None), None);
        assert_eq!(Admission::new(None).fires(4, Some(1e12)), Some(4));
        assert_eq!(Admission::new(None).fires(4, None), Some(4));
    }

    /// The warm-up's cube walks up while its next fire, estimated from the
    /// last by the growth the last two measured, fits the device's bound on
    /// one submission: the Adreno 840's 8192^3 ran ~1.57 s a submit against
    /// KGSL's ~2 s watchdog.
    #[test]
    fn the_warm_up_cube_walks_up_while_its_next_fire_fits_the_bound() {
        let kgsl = Some(std::time::Duration::from_secs(2));
        // Each doubling 8x the time, 1024^3 at 3 ms: 8192^3 at ~1.54 s fits.
        let cubic = |side: u32| 3e6 * (f64::from(side) / 1024.0).powi(3);
        let mut timed = Vec::new();
        let (side, worst) = warm_side(kgsl, 1.0, |s, _| {
            timed.push(s);
            Ok::<Option<f64>, ()>(Some(cubic(s)))
        })
        .unwrap();
        assert_eq!((side, worst), (8192, 1.0));
        assert_eq!(timed, vec![512, 1024, 2048, 4096, 8192]);
        // A device slowing past the flops' growth (each doubling 10x, as a
        // clock sinking under its power cap): 4096^3 at 300 ms, the next
        // estimated at 3 s, past the bound: it is never fired.
        let sinking = |side: u32| 3e6 * 10f64.powf((f64::from(side) / 1024.0).log2());
        let mut timed = Vec::new();
        let (side, _) = warm_side(kgsl, 1.0, |s, _| {
            timed.push(s);
            Ok::<Option<f64>, ()>(Some(sinking(s)))
        })
        .unwrap();
        assert_eq!(side, 4096);
        assert_eq!(timed, vec![512, 1024, 2048, 4096]);
        // No bound stated: the widest, nothing timed.
        let side = warm_side(None, 1.0, |_, _| -> Result<Option<f64>, ()> {
            panic!("timed with no bound")
        });
        assert_eq!(side, Ok((WARM_SHAPE.0, 1.0)));
        // The probe itself past the bound: the cube under it.
        assert_eq!(
            warm_side(kgsl, 1.0, |_, _| Ok::<Option<f64>, ()>(Some(3e9))),
            Ok((WARM_PROBE / 2, 1.0))
        );
        // The walk's own estimate low by 1.3x at 4096 (its fire 1.3x the
        // growth's): the worst ratio seen is 1.3, and 8192, estimated from
        // 4096's growth, stands past a 1.5 s bound under it: never fired.
        let mut timed = Vec::new();
        let lagging = |side: u32| match side {
            512 => 0.29e6,
            1024 => 2.34e6,
            2048 => 18.75e6,
            _ => 150e6 * 1.3,
        };
        let (side, worst) = warm_side(Some(std::time::Duration::from_millis(1500)), 1.0, |s, _| {
            timed.push(s);
            Ok::<Option<f64>, ()>(Some(lagging(s)))
        })
        .unwrap();
        assert_eq!(side, 4096);
        assert!((worst - 1.3).abs() < 0.01, "{worst}");
        assert_eq!(timed, vec![512, 1024, 2048, 4096]);
        // A size the caller's own estimate holds past the bound is never
        // submitted (`None`): the walk stops at the size before it.
        let mut timed = Vec::new();
        let (side, _) = warm_side(kgsl, 1.0, |s, _| {
            timed.push(s);
            Ok::<Option<f64>, ()>((s < 2048).then(|| cubic(s)))
        })
        .unwrap();
        assert_eq!(side, 1024);
        assert_eq!(timed, vec![512, 1024, 2048]);
        // The first probe never submitted: the cube under it.
        assert_eq!(
            warm_side(kgsl, 1.0, |_, _| Ok::<Option<f64>, ()>(None)),
            Ok((WARM_PROBE / 2, 1.0))
        );
    }

    /// The attention curve's keys are where its shape is compute-bound on
    /// the device (its flops a byte past the ridge), from the device's own
    /// matmul peak and bandwidth: never where a matmul lattice stopped.
    #[test]
    fn the_attention_keys_are_compute_bound_and_their_own() {
        let intensity = |keys: u32| {
            let s = ATTN_SHAPE.at(keys);
            attn_flops(s) / attn_bytes(s)
        };
        // A ridge every point clears: the whole curve.
        assert_eq!(attn_keys(1.0), ATTN_KEYS.to_vec());
        // A ridge past 512 keys' intensity: from the first past it.
        let ridge = (intensity(512) + intensity(1024)) / 2.0;
        assert_eq!(attn_keys(ridge), ATTN_KEYS[1..].to_vec());
        // None compute-bound: the widest alone, never nothing.
        assert_eq!(attn_keys(f64::MAX), vec![ATTN_KEYS[ATTN_KEYS.len() - 1]]);
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
    fn a_load_whose_layers_state_every_constant_measures_nothing() {
        let p = rates_at_load(
            chip(all()),
            &profile::rates::KEYS,
            None,
            None,
            None,
            |_| -> Result<_, String> { panic!("measured a profile stating every constant") },
        )
        .unwrap();
        assert_eq!(p.rates(), all());
        assert!(!p.tuned().measured_this_load);
        assert_eq!(crate::load::measured_line(p.tuned()), None);
    }

    /// A backend whose estimate weighs fewer keys measures nothing for the
    /// rest: a profile stating its six is complete for it, not for one
    /// weighing all eight.
    #[test]
    fn a_load_measures_the_keys_its_backend_weighs_and_no_others() {
        let six = all().only(&profile::rates::KEYS[..6]);
        let p = rates_at_load(
            chip(six.clone()),
            &profile::rates::KEYS[..6],
            None,
            None,
            None,
            |_| -> Result<_, String> { panic!("measured keys the backend does not weigh") },
        )
        .unwrap();
        assert_eq!(p.rates(), six);
        let err = rates_at_load(
            chip(six.clone()),
            &profile::rates::KEYS,
            None,
            None,
            None,
            |_| {
                Ok(Measured {
                    rates: six.clone(),
                    unsettled: false,
                    throttled: None,
                })
            },
        )
        .unwrap_err();
        assert!(
            err.contains(
                "measured no [\"scan_gelems\", \"few_row_tflops\", \"workgroups_per_core\"]"
            ),
            "{err}"
        );
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
                    few_row_tflops: Some(f64::NAN),
                    ..all()
                },
                "`[rates] few_row_tflops` as NaN",
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
        // Stated in a profile, it is refused by the same key: the roofline
        // divides by none of it.
        let stated = profile::rates::Rates {
            attn_tflops_by_keys: Some(vec![(512, 0.0)]),
            ..all()
        };
        assert_eq!(
            crate::roofline::Constants::of(&stated).err(),
            Some("attn_tflops_by_keys")
        );
        let nan = profile::rates::Rates {
            hbm_gbps: Some(f64::NAN),
            ..all()
        };
        assert_eq!(crate::roofline::Constants::of(&nan).err(), Some("hbm_gbps"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The RTX PRO 4500's tiled attention at one workgroup a head: 264
    /// workgroups fill its 80 units, 3.3 a unit; a dispatch of 32 ran 8.25x
    /// the time the full curve states.
    #[test]
    fn the_workgroups_a_core_holds_come_from_one_dispatch_against_the_full_rate() {
        let per = workgroups_per_core(32, 80, 8.25e6, 1e6).unwrap();
        assert!((per - 3.3).abs() < 1e-9, "{per}");
        assert_eq!(workgroups_per_core(32, 80, 0.0, 1e6), None);
        assert_eq!(workgroups_per_core(32, 80, 1e6, 0.0), None);
        assert_eq!(workgroups_per_core(32, 0, 1e6, 1e6), None);
    }

    #[test]
    fn a_frame_doubles_until_a_fire_settles_and_no_further_than_the_cap() {
        let ms = |n: u64| std::time::Duration::from_millis(n);
        // A fire of 1 ms behind a 300 us submission: frames of 1, 2, 4, ...
        // until two agree within AGREEMENT (the submission amortized).
        let mut laid = Vec::new();
        let f = timed_frames(
            None,
            None,
            |n| {
                laid.push(n);
                Ok::<_, String>(300e3 + 1e6 * f64::from(n))
            },
            |why| why,
        )
        .unwrap();
        assert_eq!(laid, [1, 2, 4, 8, 16]);
        assert_eq!((f.fires, f.capped), (16, false));
        assert!((f.per_fire_ns - 1.01875e6).abs() < 1.0);
        // A 3 ms bound stops the doubling at 2 fires, unsettled.
        let f = timed_frames(
            Some(ms(3)),
            None,
            |n| Ok::<_, String>(300e3 + 1e6 * f64::from(n)),
            |w| w,
        )
        .unwrap();
        assert_eq!((f.fires, f.capped), (2, true));
        // One fire estimated past the bound: refused, no frame laid.
        let e = timed_frames(
            Some(ms(5)),
            Some(8e6),
            |_| -> Result<f64, String> { panic!("laid a frame") },
            |w| w,
        )
        .unwrap_err();
        assert!(e.contains("past the device's bound of 5.0 ms"), "{e}");
    }

    #[test]
    fn the_warm_up_settles_by_the_same_rule() {
        // Rates falling 20%, 10%, then 2% and 1%: settled at the fifth.
        let mut rates = [100.0, 80.0, 72.0, 70.6, 70.0].into_iter();
        let w = warm_up(|| Ok::<_, ()>(rates.next().unwrap())).unwrap();
        assert_eq!((w.rotations, w.rate), (5, 70.0));
    }

    #[test]
    fn a_memory_bound_rate_warms_until_it_stops_rising() {
        // An M1's per-row ops reading the DVFS levels 146, 175, 217, 262 GB/s
        // as their own op warms the clock: timed at the sustained 262.
        let mut rates = [146.0, 175.0, 217.0, 262.0, 263.0, 262.5, 300.0].into_iter();
        let w = warm_up(|| Ok::<_, ()>(rates.next().unwrap())).unwrap();
        assert_eq!((w.rotations, w.rate), (6, 262.5));
    }

    #[test]
    fn a_short_frame_in_one_clock_state_does_not_settle_the_measurement() {
        // A bimodal clock: frames under 4 ms (four fires of 1 ms) run in the
        // slow state, 2 ms a fire, and agree with each other; frames past it
        // reach the sustained 1 ms. Two agreeing points would stop at the
        // slow state; three agreeing points span 4x and find the sustained.
        let mut laid = Vec::new();
        let f = timed_frames(
            None,
            None,
            |n| {
                laid.push(n);
                let per = match n {
                    1 => 2.0e6,
                    2 => 1.98e6,
                    4 => 1.3e6,
                    _ => 1.0e6,
                };
                Ok::<_, String>(per * f64::from(n))
            },
            |w| w,
        )
        .unwrap();
        assert_eq!(laid, [1, 2, 4, 8, 16, 32]);
        assert_eq!((f.fires, f.per_fire_ns), (32, 1.0e6));
    }

    #[test]
    fn the_rotation_carries_across_frames() {
        // Three frames of two fires over five sets: 0 1 | 2 3 | 4 0, never
        // 0 1 | 0 1 | 0 1.
        let turn = Rotation::default();
        let frames: Vec<Vec<usize>> = (0..3)
            .map(|_| (0..2).map(|_| turn.next(5)).collect())
            .collect();
        assert_eq!(frames, [vec![0, 1], vec![2, 3], vec![4, 0]]);
    }

    #[test]
    fn the_lattice_ascends_and_the_counts_are_the_estimates() {
        assert!(step_bytes() >= 1 << 30, "an 8192^2 rotation");
        assert!(GEMM_ROWS.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(
            GEMM_ROWS,
            [256, 362, 512, 724, 1024, 1448, 2048, 2896, 4096, 5793, 8192]
        );
        assert!(ATTN_KEYS.windows(2).all(|w| w[0] < w[1]));
        assert!(elementwise_sets() >= 2);
        assert!(DEPTHS.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(weights_at(1024, 1024), 384);
        assert_eq!(weights_at(16384, 1024), 24);
        assert_eq!(weights_at(8192, 8192), 12);
        assert_eq!(weights_at(16384, 16384), 12);
        assert_eq!(DEPTHS.last(), Some(&16384));
        assert_eq!(gemm_flops(16, 2048, 1024), 2.0 * 16.0 * 2048.0 * 1024.0);
        // 4 · 2048 · 2048 · 16 · 128.
        assert_eq!(attn_flops(ATTN_SHAPE), 4.0 * 2048.0 * 2048.0 * 2048.0);
        // 2048 tokens · 16 heads · 128 · 128.
        assert_eq!(scan_elements(SCAN_SHAPE), 2048.0 * 16.0 * 16384.0);
        assert_eq!(SCAN_SHAPE.channels(), 6144);
        assert_eq!(SCAN_SHAPE.at(512).tokens, 512);
        assert!(scan_sets_at(SCAN_SHAPE.at(512)) >= 2);
        assert!(scan_sets_at(SCAN_SHAPE.at(512)) > scan_sets_at(SCAN_SHAPE.at(16384)));
        for pages in [1, 2, 7, 32, 1024, 7919, 15838] {
            let mut order = attn_page_order(pages);
            let in_order = order.windows(2).all(|w| w[0] < w[1]);
            assert!(pages <= 2 || !in_order, "{pages} pages scattered");
            order.sort_unstable();
            assert_eq!(
                order,
                (0..pages).collect::<Vec<u32>>(),
                "a permutation of {pages}"
            );
        }
        assert_eq!(&attn_page_order(32)[..4], &[0, 15, 30, 13]);
        assert_eq!(ATTN_SHAPE.at(512).rows, 512);
        assert!(ATTN_DIMS.windows(2).all(|w| w[0] < w[1]));
        // 512 keys at d 64: 1 MiB of q, 1 MiB of k and v; 16384 at 256: 256
        // MiB, three sets past 768 MiB.
        // engine-cuda's sizing (63876a684), the pages laid whole: 512 keys
        // at d 64, q 1 MiB and kv 1 MiB, 384 sets of 768 MiB; at 16384 and
        // d 256 three; one key lays its whole page of 16.
        assert_eq!(attn_sets_at(ATTN_SHAPE.at(512).with_dim(64), 16), 384);
        assert_eq!(attn_sets_at(ATTN_SHAPE.at(16384).with_dim(256), 16), 3);
        assert_eq!(
            attn_sets_at(ATTN_SHAPE.at(1).with_dim(128), 16),
            ATTN_SET_BYTES.div_ceil(16 * 128 * 2 + 2 * 16 * 8 * 128 * 2)
        );
        assert_eq!(ATTN_SHAPE.at(512).with_dim(256).head_dim, 256);
        assert_eq!(
            attn_flops(ATTN_SHAPE.at(512).with_dim(256)),
            2.0 * attn_flops(ATTN_SHAPE.at(512).with_dim(128))
        );
        assert_eq!(attn_flops(ATTN_SHAPE.at(512)), 4.0 * 512.0 * 512.0 * 2048.0);
    }

    #[test]
    fn the_operand_pattern_is_fixed_bf16_in_range_and_varied() {
        let a = operand_pattern();
        assert_eq!(a.len(), OPERAND_BYTES);
        assert_eq!(a, operand_pattern());
        let values: Vec<f32> = a
            .chunks_exact(2)
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
            .collect();
        assert!(values.iter().all(|v| (-1.0..1.0).contains(v)));
        assert!(values.iter().any(|&v| v != values[0]));
        let mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / values.len() as f64;
        assert!(mean.abs() < 0.01, "{mean}");
    }

    #[test]
    fn a_rate_holds_no_launches() {
        // A large fire: 1 ms over a 2 us launch, its rate barely moved.
        let rate = rate_from("big", 1_000_000.0, 1, 2000.0, 1e9).unwrap();
        assert!((rate - 1e9 / 998_000.0).abs() < 1e-9);
        // A small one the gap dominates: 3 us of a 2 us launch is 1 us of
        // work, not 3.
        assert_eq!(rate_from("small", 3000.0, 1, 2000.0, 1000.0).unwrap(), 1.0);
        // A scan's two dispatches both come off.
        assert_eq!(rate_from("scan", 5000.0, 2, 2000.0, 1000.0).unwrap(), 1.0);
        // No longer than its launch (within the agreement): gap-bound.
        assert_eq!(
            rate_from("tiny", 1950.0, 1, 2000.0, 10.0).unwrap(),
            f64::INFINITY
        );
        // Far below it: the two disagree, refused by name.
        let err = rate_from("matmul 16x1024x1024", 1000.0, 1, 2000.0, 10.0).unwrap_err();
        assert!(
            err.starts_with("matmul 16x1024x1024: a fire of 1 dispatches"),
            "{err}"
        );
    }

    #[test]
    fn the_launch_is_what_each_dependent_dispatch_adds() {
        // 64 dispatches over 1 ms of work in 1.112 ms: 1.75 us each.
        assert!((launch_from(1_112_000.0, 1_000_000.0, 64) - 1750.0).abs() < 1e-6);
        // A chain no slower than its one kernel: no gap.
        assert_eq!(launch_from(990.0, 1000.0, 64), 0.0);
        assert!(LAUNCH_CHAINS.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn the_warm_up_runs_until_its_rate_stops_falling() {
        // A clock sinking 100, 90, 84, then 82 and 82.5: five rotations, to
        // the second step in a row that moves it by less than AGREEMENT
        // (the stop rule every measurement settles by; it was the first
        // rotation not slower than the one before).
        let mut rates = [100.0, 90.0, 84.0, 82.0, 82.5, 80.0].into_iter();
        let warmed = warm_up(|| Ok::<f64, ()>(rates.next().unwrap())).unwrap();
        assert_eq!((warmed.rotations, warmed.rate), (5, 82.5));
        // A warm device: three rotations.
        let mut rates = [100.0, 100.0, 100.0].into_iter();
        assert_eq!(
            warm_up(|| Ok::<f64, ()>(rates.next().unwrap()))
                .unwrap()
                .rotations,
            3
        );
    }

    #[test]
    fn the_points_are_timed_in_a_spread_order() {
        let order = spread_order(250);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..250).collect::<Vec<usize>>());
        assert!(!order.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn the_date_is_the_civil_one() {
        assert_eq!(civil(0), (1970, 1, 1));
        assert_eq!(civil(20_408), (2025, 11, 16));
        assert_eq!(civil(-1), (1969, 12, 31));
    }

    #[test]
    fn the_writer_writes_what_the_profile_reads() {
        let rates = profile::rates::Rates {
            stale: None,
            landing: None,
            launch_ns: Some(3500),
            gemm_tflops_by_rows: Some(vec![(4096, vec![(256, 90.0), (8192, 181.5)])]),
            attn_tflops_by_keys: Some(vec![(512, 60.0), (8192, 120.0)]),
            hbm_gbps: Some(800.0),
            cached_gbps: None,
            elementwise_gbps: Some(500.0),
            scan_gelems: Some(7.0),
            few_row_tflops: Some(4.5),
            workgroups_per_core: Some(3.3),
            submit_worst_ratio: None,
            load_width: None,
            resident_workgroups: None,
            attn_tflops_by_rows_keys: None,
            product_ns: None,
            kernel_curves: None,
            dense_settled: Vec::new(),
            measured_on: None,
            measured_by: None,
        };
        assert_eq!(tuned(&rates), rates.toml());
        let (key, text) = provenance("vulkan", "the matmul through Coopmat");
        assert_eq!(key, "rates");
        assert!(text.starts_with("\"pie tune vulkan --rates on "), "{text}");
        assert!(
            text.contains("the matmul through Coopmat; at engine::rates' shapes"),
            "{text}"
        );
    }
}
