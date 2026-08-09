//! Differential proof that [`driver_cuda::store::dtoa`] is
//! `nlohmann::json`'s double formatter.
//!
//! # What is pinned, and why it is the C++'s hash
//!
//! The golden below is the FNV-1a 64 of the output of
//! `tests/oracle/dtoa/oracle.cpp`, which formats 200,041 doubles through
//! `nlohmann::json::dump()`. This test regenerates the identical corpus and
//! formats it through the Rust port. If the hashes match, the two agree on
//! every one of those values, byte for byte.
//!
//! The number is the **C++'s**, never the Rust's. A golden captured from the
//! implementation under test can be re-blessed by the next change to that
//! implementation, which is the one thing a parity test must not permit.
//!
//! Run `tests/oracle/dtoa/run.sh` to regenerate it.
//!
//! # Why bother at all
//!
//! Rust's own `{}` already round-trips, so this is not about correctness of
//! the value. It is about the planner profile cache being one file that two
//! implementations read-merge-rewrite. Grisu2 and shortest-correctly-rounded
//! disagree on roughly 0.07% of realistic values, so a port that used `{}`
//! would silently rewrite entries it never touched.

use driver_cuda::store::dtoa::write_f64;
use std::fmt::Write as _;

/// FNV-1a 64 of `tests/oracle/dtoa/oracle.cpp`'s output.
const GOLDEN_FNV1A64: u64 = 0xcd8e_0481_bfbd_4f18;
/// Byte count of the same, so a truncated sweep cannot collide its way to green.
const GOLDEN_BYTES: usize = 7_563_702;
/// Row count of the same.
const GOLDEN_ROWS: usize = 200_041;

/// Hand-written because `DefaultHasher` is explicitly not stable across Rust
/// releases, and a golden that moves with the toolchain pins nothing.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Knuth's MMIX LCG, matching `oracle.cpp`'s `next` exactly.
fn next(x: &mut u64) -> u64 {
    *x = x
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *x
}

fn emit(out: &mut String, v: f64) {
    let _ = write!(out, "{:016x}\t", v.to_bits());
    write_f64(out, v);
    out.push('\n');
}

/// Regenerate the oracle's corpus. Every value and its order must match
/// `oracle.cpp`, or the hashes cannot be compared.
#[expect(
    clippy::excessive_precision,
    clippy::inconsistent_digit_grouping,
    reason = "transcribed character for character from oracle.cpp, and the \
              \"excess\" digits are precisely what this file exists to prove \
              Grisu2 emits"
)]
fn corpus() -> String {
    let mut out = String::with_capacity(8 << 20);

    // Exactly the 41 values `oracle.cpp` lists, in its order. Written the way
    // the C++ writes them, so the two lists can be diffed by eye; clippy's
    // grouping and precision lints are waived for that reason.
    let edges: [f64; 41] = [
        0.0, -0.0, 1.0, -1.0, 0.5, 1.5, 0.1, 100.0, 12345.0,
        1e-4, 1e-5, 1e-6, 1e-7, 1e14, 1e15, 1e16, 1e17, 1e21, 1e-21,
        1e100, 1e-100, 1.0 / 3.0, 123_456_789.123_456, 3.0e30, 2.5e-30,
        5e-324, 1.797_693_134_862_315_7e308, 2.225_073_858_507_201_4e-308,
        999_999_999_999_999.0, 1_000_000_000_000_000.0, 0.000_099_99, 0.0001,
        2.0, 4.0, 1024.0, 4_503_599_627_370_496.0, 2.0e-308,
        46934.815_584_012_416, 72972.677_071_267_06, 27453.918_300_648_482,
        3.411_036_675_017_818_7e-295,
    ];
    for v in edges {
        emit(&mut out, v);
    }

    // Milliseconds and tokens/second: the range these fields actually hold.
    let mut x: u64 = 88_172_645_463_325_252;
    for _ in 0..100_000 {
        let r = next(&mut x);
        #[expect(
            clippy::cast_precision_loss,
            reason = "53 bits into an f64 mantissa is exact; this mirrors the C++ cast"
        )]
        let unit = (r >> 11) as f64 / 9_007_199_254_740_992.0;
        emit(&mut out, unit * 100_000.0);
    }

    // Arbitrary finite bit patterns.
    let mut y: u64 = 1_234_567_890_123_456_789;
    for _ in 0..100_000 {
        let v = loop {
            let v = f64::from_bits(next(&mut y));
            if v.is_finite() {
                break v;
            }
        };
        emit(&mut out, v);
    }
    out
}

#[test]
fn the_rust_formatter_reproduces_nlohmanns_output_exactly() {
    let text = corpus();
    let bytes = text.as_bytes();

    // Shape first: a hash mismatch on its own cannot tell a wrong digit from
    // a corpus that stopped early, and the two want different investigations.
    assert_eq!(
        bytes.iter().filter(|&&b| b == b'\n').count(),
        GOLDEN_ROWS,
        "corpus generation diverged from oracle.cpp before formatting was reached"
    );
    assert_eq!(bytes.len(), GOLDEN_BYTES, "total output length differs");
    assert_eq!(
        fnv1a64(bytes),
        GOLDEN_FNV1A64,
        "formatted output differs from nlohmann; run tests/oracle/dtoa/run.sh \
         with DTOA_ORACLE_OUT set and diff against it"
    );
}

#[test]
#[expect(clippy::excessive_precision, reason = "the extra digits are the subject")]
fn the_golden_actually_discriminates() {
    // A pin nothing can break is not a pin. Perturb one value the way a
    // plausible port mistake would -- using Rust's shortest representation
    // instead of Grisu2's digits -- and require the hash to notice.
    let mut text = corpus();
    let native = format!("{}", 46_934.815_584_012_416_f64);
    assert_eq!(
        native, "46934.81558401242",
        "Rust's own formatter no longer differs here; the premise of \
         store::dtoa needs rechecking"
    );
    text = text.replacen("46934.815584012416", &native, 1);
    assert_ne!(fnv1a64(text.as_bytes()), GOLDEN_FNV1A64);
}
