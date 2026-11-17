//! What every backend's `pie tune <backend> --rates` times, and how it
//! counts the work: one lattice of shapes, model-independent, so a profile's
//! `[rates]` (`profile::fire::Rates`) mean the same on every backend and
//! the estimate (`crate::load::FireEstimate`) reads them the same way. Each
//! backend times these shapes through its own kernels and its own clock;
//! the counts here turn its times into rates, and [`tuned`] and
//! [`provenance`] write them.
//!
//! - `bf16_flops_per_ns_by_rows_n_k`: a bf16 matmul of `m` rows, `n`
//!   columns and depth `k` at each `m` of the engine's own rung lattice
//!   (the rows its loads lay fires at, which its rates step times exactly;
//!   no lattice is engine::rates') and each `n` and
//!   each `k` of [`DEPTHS`] (a serving projection is seldom square: a gate
//!   and up leg runs wide and shallow, a down leg narrow and deep),
//!   [`gemm_flops`] a matmul over its time.
//! - `attn_flops_per_ns_by_keys_and_dim`: the backend's prefill attention
//!   at [`ATTN_SHAPE`]'s heads, one sequence of `keys` rows attending itself
//!   at each `keys` of [`ATTN_KEYS`] and head width `d` of [`ATTN_DIMS`]
//!   ([`AttnShape::at`], [`AttnShape::with_dim`]), its pages laid in the pool
//!   in [`attn_page_order`], through the attention kernel serving selects at
//!   that shape, [`attn_flops`] over its time.
//! - `scan_elements_per_ns_by_tokens`: at each `tokens` of [`SCAN_TOKENS`]
//!   ([`ScanShape::at`]), over [`scan_sets_at`] cold sets, the backend's
//!   prefill gated-delta scan and the
//!   causal conv before it at [`SCAN_SHAPE`], [`scan_elements`] over the two
//!   together's time.
//! - `bytes_per_ns`: a copy of device memory, every byte read and written.
//! - `launch_ns`: what a dispatch adds to its work in a chain of dependent
//!   ones submitted as serving submits them (a captured graph, one command
//!   buffer, one encoder): a chain of `n` tiny kernels, each reading the one
//!   before's output, against one kernel doing the chain's whole work, at
//!   each `n` of [`LAUNCH_CHAINS`] ([`launch_from`]). Where a backend
//!   overlaps its launches the gap comes out small; where it serializes
//!   them, it is what each adds.

/// The bytes the rates step lays on the device at once, at most, for a
/// largest fire of `top` rows: the largest of the warm-up's planes and
/// operands, one grid shape's rotation and operands, one attention point's
/// cold sets, one scan point's cold sets, and the copy's two buffers. What
/// the step's device binds, so it needs no load budget.
#[must_use]
pub fn step_bytes(top: u32) -> u64 {
    let bf16 = |elements: u64| elements * 2;
    let (wm, wn, wk) = WARM_SHAPE;
    let warm = bf16(u64::from(wm) * u64::from(wk))
        + weights_at(wn, wk) as u64 * bf16(u64::from(wn) * u64::from(wk))
        + bf16(u64::from(wm) * u64::from(wn));
    let rows = u64::from(top.max(1));
    let grid = DEPTHS
        .iter()
        .flat_map(|&k| DEPTHS.iter().map(move |&n| (n, k)))
        .map(|(n, k)| {
            bf16(rows * u64::from(k))
                + weights_at(n, k) as u64 * bf16(u64::from(n) * u64::from(k))
                + bf16(rows * u64::from(n))
        })
        .max()
        .unwrap_or(0);
    let attn = ATTN_DIMS
        .iter()
        .flat_map(|&d| {
            ATTN_KEYS
                .iter()
                .map(move |&keys| ATTN_SHAPE.at(keys).with_dim(d))
        })
        .map(|s| {
            let q = u64::from(s.rows) * u64::from(s.q_heads * s.head_dim);
            let kv = 2 * u64::from(s.keys) * u64::from(s.kv_heads * s.head_dim);
            attn_sets_at(s) as u64 * bf16(q + kv) + bf16(q)
        })
        .max()
        .unwrap_or(0);
    let scan = SCAN_TOKENS
        .iter()
        .map(|&t| {
            let s = SCAN_SHAPE.at(t);
            let rows = u64::from(s.tokens) * u64::from(s.channels());
            let state = u64::from(s.conv_width * s.channels() + s.v_heads * s.k_dim * s.v_dim);
            scan_sets_at(s) as u64 * (bf16(rows) + 4 * state) + bf16(rows) * 2
        })
        .max()
        .unwrap_or(0);
    let copy = 2 * (256u64 << 20);
    warm.max(grid).max(attn).max(scan).max(copy)
}

/// The depths the matmul grid is measured at (its columns the same): the
/// contractions of the dense layers served, from a small model's 1024
/// (Qwen3-0.6B's hidden) through 4096 (Qwen3-8B's) and 8192 (a 30B-class
/// hidden) to 16384, past the down projections of the 8B and Llama-class
/// MLPs (12288 and 14336 deep), so none is priced at a clamped point; a
/// device's rate at the same rows falls with the depth (a narrower
/// contraction feeds fewer tiles).
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
    let mut last = rotation()?;
    let mut rotations = 1;
    loop {
        let rate = rotation()?;
        rotations += 1;
        if rate >= last {
            return Ok(Warmed { rotations, rate });
        }
        last = rate;
    }
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

/// The q / k / v sets the timed attentions of `shape` rotate through:
/// [`ATTN_SET_BYTES`] of the shape's q rows and k and v keys (bf16), and
/// never fewer than 2.
#[must_use]
pub fn attn_sets_at(shape: AttnShape) -> usize {
    let q = shape.rows as usize * (shape.q_heads * shape.head_dim) as usize;
    let kv = 2 * shape.keys as usize * (shape.kv_heads * shape.head_dim) as usize;
    ATTN_SET_BYTES.div_ceil(((q + kv) * 2).max(1)).max(2)
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

/// The keys the attention curve is measured at (each a sequence of as many
/// rows attending itself): powers of two from 512 (a short prompt's piece)
/// to 16384 (past the widest rung a load lays), since a paged prefill's
/// rate a flop grows with the keys its tiles stream.
pub const ATTN_KEYS: [u32; 6] = [512, 1024, 2048, 4096, 8192, 16384];

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

/// The head widths the attention grid is measured at: the widths of the
/// models served (64 for the small and older ones, 128 for most, 256 for
/// Qwen3.5's and Gemma's global layers), since a kernel's rate a flop moves
/// with the width its tiles hold.
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

/// The prompt lengths the scan curve is measured at: the attention's keys
/// (a scan's rate a token grows with the chunk it walks, as a paged
/// attention's with its keys).
pub const SCAN_TOKENS: [u32; 6] = ATTN_KEYS;

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
    let text = format!(
        "pie tune {backend} --rates on {} (pie {}): {how}; at engine::rates' lattice (matmul \
         m x n x k, m the rung lattice, n and k 1024..16384; attention keys x keys, keys 512..16384, {}/{} \
         heads of 64..256; scan {} tokens, {}/{} heads of {}x{}, conv {})",
        today(),
        env!("CARGO_PKG_VERSION"),
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
    use super::*;

    #[test]
    fn the_lattice_ascends_and_the_counts_are_the_estimates() {
        assert!(step_bytes(10240) >= 6 << 30, "a 16384^2 rotation");
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
        assert!(ATTN_KEYS.windows(2).all(|w| w[0] < w[1]));
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
        assert_eq!(attn_sets_at(ATTN_SHAPE.at(512).with_dim(64)), 384);
        assert_eq!(attn_sets_at(ATTN_SHAPE.at(16384).with_dim(256)), 3);
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
        // A clock sinking 100, 96, 93, 92 then holding: four rotations to
        // the first not slower than the one before.
        let mut rates = [100.0, 96.0, 93.0, 92.0, 92.5, 80.0].into_iter();
        let warmed = warm_up(|| Ok::<f64, ()>(rates.next().unwrap())).unwrap();
        assert_eq!((warmed.rotations, warmed.rate), (5, 92.5));
        // A warm device: two rotations.
        let mut rates = [100.0, 100.0].into_iter();
        assert_eq!(
            warm_up(|| Ok::<f64, ()>(rates.next().unwrap()))
                .unwrap()
                .rotations,
            2
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
            bf16_flops_per_ns_by_rows_n_k: Some(vec![
                (16, 2048, 1024, 100.0),
                (8192, 2048, 1024, 900.5),
            ]),
            attn_flops_per_ns_by_keys_and_dim: Some(vec![(512, 128, 40.0), (16384, 128, 50.0)]),
            scan_elements_per_ns_by_tokens: Some(vec![(2048, 7.0)]),
            bytes_per_ns: Some(800.0),
            launch_ns: Some(3500),
        };
        assert_eq!(
            tuned(&rates),
            "[rates]\nbf16_flops_per_ns_by_rows_n_k = [\n  [16, 2048, 1024, 100.0],\n  [8192, 2048, \
             1024, 900.5],\n]\n\
             attn_flops_per_ns_by_keys_and_dim = [\n  [512, 128, 40.0],\n  [16384, 128, 50.0],\n]\nscan_elements_per_ns_by_tokens = [[2048, 7.0]]\nbytes_per_ns = 800.0\n\
             launch_ns = 3500\n"
        );
        let (key, text) = provenance("vulkan", "the matmul through Coopmat");
        assert_eq!(key, "rates");
        assert!(text.starts_with("\"pie tune vulkan --rates on "), "{text}");
        assert!(
            text.contains("the matmul through Coopmat; at engine::rates' lattice"),
            "{text}"
        );
    }
}
