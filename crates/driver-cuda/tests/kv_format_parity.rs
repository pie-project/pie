//! Differential parity between [`KvCacheFormat`] and the C++ it replaces.
//!
//! This is the migration protocol from `.wiki/plan/model-in-rust.md` §8
//! applied to `store/kv_cache_format.{hpp,cpp}`: build the C++ as an oracle,
//! sweep both implementations over the same grid, and require the outputs to
//! be byte-identical before the C++ is allowed to go.
//!
//! # Reproducing the proof
//!
//! `kv_cache_format.cpp` depends only on `tensor.hpp` and
//! `kernels/kv_cache_view.hpp`, both header-only and both free of CUDA, so the
//! real translation unit builds with a plain host compiler:
//!
//! ```text
//! cat > oracle.cpp <<'EOF'
//! #include "store/kv_cache_format.hpp"
//! #include <cstdio>
//! // ... same three loops as `dump()` below, with printf ...
//! EOF
//!
//! g++ -std=c++20 \
//!     -Icrates/driver-cuda/csrc/src \
//!     -Icrates/kernels-cuda/csrc/src \
//!     oracle.cpp crates/driver-cuda/csrc/src/store/kv_cache_format.cpp -o oracle
//!
//! ./oracle > cpp.txt
//! cargo test -p driver-cuda --test kv_format_parity -- --ignored --nocapture
//! ```
//!
//! That was run, and the two outputs matched exactly: 11,498 rows, 434,348
//! bytes, no differing line. [`GOLDEN_FNV1A64`] is the hash of the C++ side of
//! that run, so the proof survives the C++ being deleted -- which is the whole
//! point of taking it before rather than after.

use std::fmt::Write as _;

use driver_cuda::store::KvCacheFormat;

/// FNV-1a 64 of the **C++** oracle's stdout over [`dump`]'s grid.
///
/// Not a hash of the Rust output. The distinction is the entire value of this
/// constant: it is a fact about the program being replaced, recorded while
/// that program still existed, and a Rust change that alters behaviour cannot
/// quietly re-bless it.
const GOLDEN_FNV1A64: u64 = 0x93ad_f974_5c95_e256;

/// Byte count of that same output, as a second, independent witness.
///
/// A hash mismatch says "something moved"; a length that still matches says
/// the change was a value rather than a row appearing or vanishing. Cheap, and
/// it makes the first triage step free.
const GOLDEN_BYTES: usize = 434_348;

/// FNV-1a, 64-bit.
///
/// Hand-written rather than pulled in: `DefaultHasher` is explicitly not
/// stable across Rust releases, so a golden pinned to it would rot on a
/// toolchain bump, and a dependency for sixteen lines of arithmetic in one
/// test is a poor trade.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

/// Every spelling the C++ parser is swept with.
///
/// The nine catalogue aliases, four case variants, the empty string (which
/// means `auto`), and three rejects -- so the sweep covers the accept path,
/// the case-folding, the empty-string default and the error path in one grid.
const ALIASES: &[&str] = &[
    "auto",
    "bf16",
    "bfloat16",
    "fp8_e4m3",
    "fp8_e5m2",
    "int8_per_token_head",
    "fp8_per_token_head",
    "fp4_e2m1",
    "nvfp4",
    "AUTO",
    "BF16",
    "Fp8_E4M3",
    "NVFP4",
    "",
    "fp6",
    "int4",
    "garbage",
];

/// Page sizes swept. `0` for the degenerate page, `17` and `1` to catch a
/// rounding that only shows up off a power of two.
const PAGES: &[u32] = &[0, 1, 2, 16, 17, 32, 64, 128, 256];

/// KV head counts swept.
const HEADS: &[u32] = &[0, 1, 2, 8, 16, 64, 128];

/// Head dims swept. Deliberately dense around the two divisors that matter:
/// `2` for FP4's nibble packing (so 1, 15, 17, 63, 127, 129 are all odd) and
/// `16` for the blocked scale layout (so 15, 17 straddle one boundary and 63,
/// 129 straddle others). `576` is DeepSeek's fused latent width.
const DIMS: &[u32] = &[0, 1, 15, 16, 17, 63, 64, 127, 128, 129, 256, 576, 1024];

/// The Rust side of the sweep, in the oracle's exact output format.
fn dump() -> String {
    let mut out = String::with_capacity(GOLDEN_BYTES);
    let _ = writeln!(out, "VALID_VALUES\t{}", KvCacheFormat::valid_names());
    for &alias in ALIASES {
        let parsed = KvCacheFormat::from_name(alias);
        let _ = writeln!(out, "PARSE\t{}\t{}", alias, u8::from(parsed.is_ok()));
        let Ok(f) = parsed else { continue };
        let _ = writeln!(
            out,
            "FORMAT\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            alias,
            f.name(),
            f.scheme() as u8,
            f.scale_layout() as u8,
            f.storage_dtype().tag(),
            f.block_size(),
            u8::from(f.is_native_bf16()),
            u8::from(f.has_side_scales()),
        );
        for &p in PAGES {
            for &h in HEADS {
                for &d in DIMS {
                    let _ = writeln!(
                        out,
                        "SIZE\t{alias}\t{p}\t{h}\t{d}\t{}\t{}\t{}\t{}",
                        f.storage_head_dim(d),
                        f.kv_bytes_per_page(p, h, d),
                        f.scale_bytes_per_page(p, h, d),
                        f.total_bytes_per_page(p, h, d),
                    );
                }
            }
        }
    }
    out
}

#[test]
fn the_rust_port_reproduces_the_cpp_oracle_byte_for_byte() {
    let produced = dump();
    assert_eq!(
        produced.len(),
        GOLDEN_BYTES,
        "the sweep changed shape: a row appeared or vanished, so compare the \
         grids before looking at any value"
    );
    assert_eq!(
        fnv1a64(produced.as_bytes()),
        GOLDEN_FNV1A64,
        "the sweep still has {GOLDEN_BYTES} bytes but different content, so a \
         computed value moved. Rebuild the C++ oracle (see this file's module \
         docs) and diff the two dumps to find which row"
    );
}

#[test]
fn the_grid_is_large_enough_to_be_worth_hashing() {
    // A hash over an accidentally-empty sweep passes forever. Guards the
    // guard: if a constant above is emptied, this fails before the hash does.
    assert_eq!(ALIASES.len(), 17);
    assert_eq!(PAGES.len() * HEADS.len() * DIMS.len(), 819);
    let size_rows = dump().lines().filter(|l| l.starts_with("SIZE\t")).count();
    assert_eq!(size_rows, 14 * 819, "14 of the 17 aliases parse");
}

/// Print the Rust sweep for a manual diff against the C++ oracle.
///
/// `cargo test -p driver-cuda --test kv_format_parity -- --ignored --nocapture`
///
/// Ignored by default because its output is 434 KB and its value is entirely
/// in being piped somewhere.
#[test]
#[ignore = "diagnostic: prints 434 KB for diffing against the C++ oracle"]
fn print_sweep_for_manual_diff() {
    print!("{}", dump());
}
