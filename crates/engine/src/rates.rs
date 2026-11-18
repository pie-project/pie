//! What every backend's `pie tune <backend> --rates` (or a load on a
//! device whose profile lacks them) times, and how it counts the work: a few
//! shapes, model-independent, whose times give the device's rates
//! (`profile::fire::Rates`), so a profile's `[rates]` mean the same on every
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

/// The rows the matmul curve is timed at: the lattice's octaves a fire's
/// prompt reaches, 256 to 8192.
pub const GEMM_ROWS: [u32; 6] = [256, 512, 1024, 2048, 4096, 8192];

/// The keys (and rows) the attention curve is timed at.
pub const ATTN_KEYS: [u32; 5] = [512, 1024, 2048, 4096, 8192];

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

/// The tolerance two runs of a rates step agree within, and a check's
/// estimate within the measured best: what [`rate_from`] allows a burst to
/// fall below its launches before it calls the two a measurement fault.
/// Defined once, in the profile crate, whose search breaks ties by it.
pub use profile::AGREEMENT;

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

/// A timed point's frames, the one rule for every point every engine
/// times: their length settled by [`timed_frames`], then frames of that
/// length run until their rate stops moving ([`warm_up`]'s rule), so the
/// point is timed at the clock it holds under its own load, never in a
/// state an earlier point or an idle gap left behind. Before this every
/// point but the memory-bound ones went untimed by it, and on an M5 Pro a
/// few-microsecond matmul read about 5 or about 34 TFLOP/s run to run.
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
    pub rates: profile::fire::Rates,
    pub unsettled: bool,
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
pub fn tuned(rates: &profile::fire::Rates) -> String {
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
    pub check: &'a dyn Fn(&str) -> Result<(), String>,
}

/// The user's file of a device, `existing` (none where there is none yet),
/// with `measured` (only the keys a load measured) added to its `[rates]`
/// and `provenance` set in its `[provenance]`: nothing else of the file
/// changes, and nothing of the profile it resolved to is copied into it.
#[must_use]
pub fn saved_text(
    existing: Option<&str>,
    measured: &profile::fire::Rates,
    provenance: &(&'static str, String),
) -> String {
    let keys: Vec<String> = measured
        .toml()
        .lines()
        .skip(1)
        .map(str::to_string)
        .collect();
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
    match header(&lines, "rates") {
        Some(at) => {
            let end = table_end(&lines, at);
            lines.splice(end..end, keys);
        }
        None => {
            lines.push(String::new());
            lines.push("[rates]".to_string());
            lines.extend(keys);
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
/// rates `chip`'s layers state where they state every one (a checked-in
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
/// holds less than it lays at most ([`step_bytes`]) the load is refused.
pub fn rates_at_load(
    chip: profile::Profile,
    save: Option<SaveTo<'_>>,
    room: Option<u64>,
    max_submit: Option<std::time::Duration>,
    measure: impl FnOnce(Option<std::time::Duration>) -> Result<Measured, String>,
) -> Result<profile::Profile, String> {
    let stated = chip.rates();
    if stated.missing().is_empty() {
        return Ok(chip);
    }
    let lacked = stated.missing();
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
    } = measure(max_submit)?;
    let rates = stated.over(measured);
    if !rates.missing().is_empty() {
        return Err(format!(
            "the device's rates step measured no {:?}: nothing weighs a fire by them",
            rates.missing()
        ));
    }
    // The user's file gains what this load measured, the keys no layer
    // stated, and how: nothing else.
    let saved_to = match save {
        Some(save) => {
            let existing = std::fs::read_to_string(&save.path).ok();
            let text = saved_text(existing.as_deref(), &rates.only(&lacked), &save.provenance);
            (save.check)(&text).map_err(|why| {
                format!(
                    "{}: the measured rates would not read back: {why}",
                    save.path.display()
                )
            })?;
            if let Some(dir) = save.path.parent() {
                std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            }
            std::fs::write(&save.path, text)
                .map_err(|e| format!("{}: {e}", save.path.display()))?;
            Some(save.path.display().to_string())
        }
        None => None,
    };
    let mut tuned = chip.tuned().clone();
    tuned.rates = rates;
    tuned.measured_this_load = true;
    tuned.saved_to = saved_to;
    tuned.frame_cut = unsettled;
    Ok(chip.with_tuned(tuned))
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
        let settled = super::timed_frames(None, None, frame, |why: String| why).expect("it settles");
        assert_eq!(settled.per_fire_ns, 10.0, "the doublings alone settle in the slow state");
        calls.set(0);
        let frames = super::warmed_frames(None, None, frame, |why| why).expect("it settles");
        let settled_at = calls.get();
        assert!(settled_at > SLOW + 1, "the warm-up ran past the slow state ({settled_at} frames)");
        // The next frame of that length reads the fast state.
        let next = frame(frames.fires).expect("a frame") / f64::from(frames.fires);
        assert_eq!(next, 1.0, "the point is timed at the clock its load holds");
    }

    use super::*;

    /// A profile of the toy registry stating `rates`.
    fn chip(rates: profile::fire::Rates) -> profile::Profile {
        let p = profile::Registry::new("toy", &[], &[]).defaults();
        let mut tuned = p.tuned().clone();
        tuned.rates = rates;
        p.with_tuned(tuned)
    }

    fn all() -> profile::fire::Rates {
        profile::fire::Rates {
            launch_ns: Some(900),
            gemm_tflops_by_rows: Some(vec![(4096, vec![(256, 100.0), (8192, 200.0)])]),
            attn_tflops_by_keys: Some(vec![(512, 50.0), (8192, 150.0)]),
            hbm_gbps: Some(700.0),
            elementwise_gbps: Some(500.0),
            scan_gelems: Some(2000.0),
        }
    }

    #[test]
    fn a_load_without_the_constants_measures_saves_and_says_so() {
        let dir = std::env::temp_dir().join(format!("pie-rates-at-load-{}", std::process::id()));
        let path = dir.join("vulkan/user/card-x.toml");
        let check = |_: &str| Ok(());
        // A checked-in launch, nothing else: the rest measured, the launch
        // kept, each submission under the budget asked.
        let stated = profile::fire::Rates {
            launch_ns: Some(1234),
            ..profile::fire::Rates::default()
        };
        let mut asked = None;
        let p = rates_at_load(
            chip(stated),
            Some(SaveTo {
                path: path.clone(),
                provenance: provenance("toy", "a test"),
                check: &check,
            }),
            None,
            Some(std::time::Duration::from_millis(5)),
            |max_submit| {
                asked = max_submit;
                Ok(Measured {
                    rates: all(),
                    unsettled: true,
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
        assert!(!first.rates().missing().is_empty());
        let check = |text: &str| {
            registry
                .parse(text, "saved", profile::Strict::User)
                .map(|_| ())
                .map_err(|e| e.to_string())
        };
        let measured = rates_at_load(
            first,
            Some(SaveTo {
                path: SHELF.user_path(&dir, "Card X"),
                provenance: provenance("toy", "a test"),
                check: &check,
            }),
            None,
            None,
            |_| {
                Ok(Measured {
                    rates: all(),
                    unsettled: false,
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
        let second = rates_at_load(second, None, None, None, |_| -> Result<_, String> {
            panic!("measured what the first load saved")
        })
        .unwrap();
        assert!(!second.tuned().measured_this_load);
    }

    #[test]
    fn a_saved_file_gains_the_measured_keys_and_their_provenance_alone() {
        let measured = profile::fire::Rates {
            hbm_gbps: Some(700.0),
            ..profile::fire::Rates::default()
        };
        let how = ("rates", "\"measured at load\"".to_string());
        // No file yet: the keys and how, nothing else.
        assert_eq!(
            saved_text(None, &measured, &how),
            "api = 1\n\n[rates]\nhbm_gbps = 700.0\n\n[provenance]\nrates = \"measured at load\"\n"
        );
        // A user's file: its decisions kept, its [rates] gaining the key,
        // its provenance's rates entry replaced, the rest left.
        let existing = "api = 1\n[engine]\ngraphs = \"on\"\n\n[rates]\nlaunch_ns = 900\n\n[provenance]\nsource = \"me\"\nrates = \"old\"\n";
        assert_eq!(
            saved_text(Some(existing), &measured, &how),
            "api = 1\n[engine]\ngraphs = \"on\"\n\n[rates]\nlaunch_ns = 900\nhbm_gbps = 700.0\n\n[provenance]\nsource = \"me\"\nrates = \"measured at load\"\n"
        );
        // A rate stated as array rows: the key lands after its last row.
        let rows =
            "api = 1\n\n[rates]\nattn_tflops_by_keys = [\n  [512, 4.0],\n  [1024, 5.0],\n]\n";
        assert_eq!(
            saved_text(Some(rows), &measured, &how),
            "api = 1\n\n[rates]\nattn_tflops_by_keys = [\n  [512, 4.0],\n  [1024, 5.0],\n]\nhbm_gbps = 700.0\n\n[provenance]\nrates = \"measured at load\"\n"
        );
    }

    #[test]
    fn a_budget_short_of_the_step_is_refused_by_name_and_measures_nothing() {
        let short = step_bytes() - 1;
        let err = rates_at_load(
            chip(profile::fire::Rates::default()),
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
        let p = rates_at_load(chip(all()), None, None, None, |_| -> Result<_, String> {
            panic!("measured a profile stating every constant")
        })
        .unwrap();
        assert_eq!(p.rates(), all());
        assert!(!p.tuned().measured_this_load);
        assert_eq!(crate::load::measured_line(p.tuned()), None);
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
        let rates = profile::fire::Rates {
            launch_ns: Some(3500),
            gemm_tflops_by_rows: Some(vec![(4096, vec![(256, 90.0), (8192, 181.5)])]),
            attn_tflops_by_keys: Some(vec![(512, 60.0), (8192, 120.0)]),
            hbm_gbps: Some(800.0),
            elementwise_gbps: Some(500.0),
            scan_gelems: Some(7.0),
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
