//! What every backend's `pie tune <backend> --rates` times, and how it
//! counts the work: one lattice of shapes, model-independent, so a profile's
//! `[rates]` (`profile::fire::Rates`) mean the same on every backend and
//! the estimate (`crate::load::FireEstimate`) reads them the same way. Each
//! backend times these shapes through its own kernels and its own clock;
//! the counts here turn its times into rates, and [`tuned`] and
//! [`provenance`] write them.
//!
//! - `bf16_flops_per_ns_by_rows`: a bf16 matmul `m x` [`GEMM_COLUMNS`] `x`
//!   [`GEMM_DEPTH`] at each `m` of [`LATTICE_ROWS`], [`gemm_flops`] a
//!   matmul over its time.
//! - `attn_flops_per_ns`: the backend's prefill attention at [`ATTN_SHAPE`],
//!   [`attn_flops`] over its time.
//! - `scan_elements_per_ns`: the backend's prefill gated-delta scan and the
//!   causal conv before it at [`SCAN_SHAPE`], [`scan_elements`] over the two
//!   together's time.
//! - `bytes_per_ns` (a copy of device memory, every byte read and written)
//!   and `launch_ns` (a dispatch's own cost inside a recorded frame) as
//!   before.

/// The rows the matmul curve is measured at: powers of two from 16 (a
/// decode fire's few lanes, one register or coopmat tile) to 8192 (past the
/// widest rung a load lays on any device today, where every device's
/// matmul is compute bound).
pub const LATTICE_ROWS: [u32; 10] = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];

/// The matmul's columns: a dense layer's width at the scale of the models
/// served (Qwen3-8B's hidden is 4096; the 0.6B's projections run 1024 to
/// 3072, the 8B's MLP 12288), wide enough that every column tile of every
/// backend's kernel fills, so the curve tells the rows' effect alone.
pub const GEMM_COLUMNS: u32 = 4096;

/// The matmul's depth, for the same reason as [`GEMM_COLUMNS`]: its
/// operands (32 MiB each at 4096 rows) fit the binding WebGPU grants by
/// default (128 MiB) at every row of [`LATTICE_ROWS`].
pub const GEMM_DEPTH: u32 = 4096;

/// The flops of one matmul of `rows` at [`GEMM_COLUMNS`] by [`GEMM_DEPTH`]:
/// two a multiply-add.
#[must_use]
pub fn gemm_flops(rows: u32) -> f64 {
    2.0 * f64::from(rows) * f64::from(GEMM_COLUMNS) * f64::from(GEMM_DEPTH)
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

impl ScanShape {
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
         m x {GEMM_COLUMNS} x {GEMM_DEPTH}, m 16..8192; attention {}x{} keys, {}/{} heads of \
         {}; scan {} tokens, {}/{} heads of {}x{}, conv {})",
        today(),
        env!("CARGO_PKG_VERSION"),
        ATTN_SHAPE.rows,
        ATTN_SHAPE.keys,
        ATTN_SHAPE.q_heads,
        ATTN_SHAPE.kv_heads,
        ATTN_SHAPE.head_dim,
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
        assert!(LATTICE_ROWS.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(gemm_flops(16), 2.0 * 16.0 * 4096.0 * 4096.0);
        // 4 · 2048 · 2048 · 16 · 128.
        assert_eq!(attn_flops(ATTN_SHAPE), 4.0 * 2048.0 * 2048.0 * 2048.0);
        // 2048 tokens · 16 heads · 128 · 128.
        assert_eq!(scan_elements(SCAN_SHAPE), 2048.0 * 16.0 * 16384.0);
        assert_eq!(SCAN_SHAPE.channels(), 6144);
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
            bf16_flops_per_ns_by_rows: Some(vec![(16, 100.0), (8192, 900.5)]),
            attn_flops_per_ns: Some(50.0),
            scan_elements_per_ns: Some(7.0),
            bytes_per_ns: Some(800.0),
            launch_ns: Some(3500),
        };
        assert_eq!(
            tuned(&rates),
            "[rates]\nbf16_flops_per_ns_by_rows = [[16, 100.0], [8192, 900.5]]\n\
             attn_flops_per_ns = 50.0\nscan_elements_per_ns = 7.0\nbytes_per_ns = 800.0\n\
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
