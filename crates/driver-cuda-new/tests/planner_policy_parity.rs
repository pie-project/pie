//! Differential proof that [`crate::store::planner_policy`] reproduces the
//! original C++ policy layer exactly.
//!
//! This follows the protocol established by `kv_format_parity.rs`: build the
//! real C++ as an oracle, sweep both implementations over the same grid,
//! require byte-identical output, and pin the **C++ side's** hash so a later
//! Rust change cannot quietly re-bless itself.
//!
//! # What makes this port's oracle different
//!
//! `kv_cache_format.cpp` could be compiled directly -- its include chain
//! happens to contain no CUDA. `memory_planner.cpp` cannot: it pulls in
//! `cuda_runtime.h`, every model header, and the toml++ config layer. So the
//! translation unit is not oracle-able as a whole.
//!
//! But the policy layer inside it is. It is an anonymous namespace at the top
//! of the file that reads four config fields and two `cudaDeviceProp` ints and
//! touches nothing else. The oracle **extracts that namespace verbatim from
//! the real source with `awk`** and compiles it against a stub
//! `cuda_runtime.h` and stub `Config`/`HfConfig`. Extraction rather than
//! copying is the point: a copied oracle silently stops testing the thing it
//! was written for the first time someone edits the original.
//!
//! # Reproducing the oracle
//!
//! ```text
//! mkdir -p /tmp/mpdiff/stub && cd /tmp/mpdiff
//! SRC=crates/driver-cuda/csrc/src/store/memory_planner.cpp
//!
//! # cudaDeviceProp carries only the two fields the policy layer reads.
//! cat > stub/cuda_runtime.h <<'EOF'
//! #pragma once
//! struct cudaDeviceProp { int major = 0; int multiProcessorCount = 0; };
//! EOF
//!
//! # The policy layer, lifted out of the real file.
//! awk '/^namespace \{$/{f=1} f{print} /^\}  \/\/ namespace$/{if(f){exit}}' \
//!     "$SRC" > anon.inc
//!
//! # derive_kv_page_size is defined outside the anonymous namespace, so it
//! # needs a second pass or the oracle fails to link.
//! awk '/^int derive_kv_page_size\(const Config& cfg,/{f=1} f{print; if($0=="}") exit}' \
//!     "$SRC" > public.inc
//!
//! g++ -std=c++20 -I stub -I crates/driver-cuda/csrc/src -I . oracle.cpp -o oracle
//! ./oracle > cpp.txt
//! ```
//!
//! `oracle.cpp` then declares the stub `Config`/`HfConfig` and `#include`s the
//! two `.inc` files inside `namespace pie_cuda_driver`, and its `main` is the
//! grid below transcribed into C++.
//!
//! # Doubles are compared as bits, not as text
//!
//! `log2_ratio` and `target_saturation_score` return `double`. Comparing them
//! through formatted text would be comparing two formatters: C's `%.17g` emits
//! 17 significant digits, while Rust's `{}` emits the shortest string that
//! round-trips, so `0.63092975357145753` and `0.6309297535714575` are the same
//! `f64` printed by two rules. Both sides therefore emit the raw IEEE-754 bit
//! pattern as hex. That is an exact comparison, and it is also strictly
//! stronger -- a one-ulp difference in `log2` would survive text rounding at
//! some precisions but cannot survive this.

use driver_cuda_new::store::planner_policy::{
    MIN_KV_TOKENS_FLOOR, align_up, clamp_pow2_nearest, decode_target, decode_target_for_profile,
    kv_page_size, kv_page_size_candidates, kv_page_size_for_profile, log2_ratio, policy_profiles,
    prefill_candidate_cap, prefill_target, prefill_target_for_profile, target_saturation_score,
    uniq_clip_desc,
};
use std::fmt::Write as _;

/// FNV-1a 64 of the **C++ oracle's** stdout.
///
/// Hand-written below rather than taken from `DefaultHasher`, whose output is
/// explicitly not stable across Rust releases and would turn a toolchain
/// upgrade into a parity failure.
const GOLDEN_FNV1A64: u64 = 0xe90e_40fa_c5fa_aa1b;

/// Byte count of the same output. Redundant with the hash, but it turns "some
/// byte differs" into "the output got shorter", which is the difference
/// between a five-minute and a fifty-minute diagnosis.
const GOLDEN_BYTES: usize = 360_724;

/// Row count, as a guard on the grid itself: if someone narrows a sweep the
/// hash also changes, and this says which way.
const GOLDEN_ROWS: usize = 14_519;

// The grid. These are the oracle's `main` verbatim.
//
// `weird` and `""` are in `PROFILES` on purpose. Every predicate in this layer
// is a string equality with an `else` fallthrough, so an unrecognised profile
// is not rejected -- it silently takes the "not latency, not throughput"
// branch of each one. That behaviour is load-bearing for anyone who typos a
// profile in config, and it is exactly the kind of thing a rewrite drops by
// pattern-matching on an enum with an error arm.
const PROFILES: [&str; 7] = ["auto", "latency", "balanced", "throughput", "capacity", "weird", ""];
const TPS: [i32; 7] = [0, 1, 2, 3, 4, 8, 16];
const SMS: [i32; 12] = [0, 1, 16, 32, 60, 78, 80, 108, 114, 132, 148, 256];
const MAJORS: [i32; 7] = [7, 8, 9, 10, 11, 12, 13];
const PINNED: [u32; 5] = [0, 1, 16, 32, 64];

fn dtoh(x: f64) -> String {
    format!("{:016x}", x.to_bits())
}

fn render() -> String {
    let mut o = String::with_capacity(GOLDEN_BYTES + 4096);

    let mut v = -4;
    while v <= 4200 {
        let _ = writeln!(o, "CLAMP\t{v}\t{}", clamp_pow2_nearest(v, 64, 2048));
        v += 7;
    }
    let mut v = -4;
    while v <= 20000 {
        let _ = writeln!(o, "CLAMP2\t{v}\t{}", clamp_pow2_nearest(v, 512, 8192));
        v += 137;
    }

    for p in PROFILES {
        let fam = policy_profiles(p);
        let _ = write!(o, "FAMILIES\t{p}\t{}", fam.len());
        for f in &fam {
            let _ = write!(o, "\t{f}");
        }
        o.push('\n');
        for tp in TPS {
            let _ = writeln!(o, "PAGEPROF\t{p}\t{tp}\t{}", kv_page_size_for_profile(p, tp));
        }
    }

    for p in PROFILES {
        for tp in TPS {
            for pin in PINNED {
                let xs = kv_page_size_candidates(pin, p, tp);
                let _ = write!(o, "PAGECAND\t{p}\t{tp}\t{pin}\t{}", xs.len());
                for x in &xs {
                    let _ = write!(o, "\t{x}");
                }
                o.push('\n');
                let _ = writeln!(o, "PAGEDERIVE\t{p}\t{tp}\t{pin}\t{}", kv_page_size(p, tp));
            }
        }
    }

    for p in PROFILES {
        for tp in TPS {
            for sm in SMS {
                for mj in MAJORS {
                    let _ = writeln!(
                        o,
                        "DECODE\t{p}\t{tp}\t{sm}\t{mj}\t{}\t{}",
                        decode_target_for_profile(p, sm),
                        decode_target(p, sm)
                    );
                    let _ = writeln!(
                        o,
                        "PREFILL\t{p}\t{tp}\t{sm}\t{mj}\t{}\t{}",
                        prefill_target_for_profile(p, sm, mj, tp),
                        prefill_target(p, sm, mj, tp)
                    );
                    let _ = writeln!(o, "PCAP\t{mj}\t{}", prefill_candidate_cap(mj));
                }
            }
        }
    }

    let mut val = -3;
    while val <= 4096 {
        for tgt in [1, 2, 64, 256, 2048, 8192] {
            let _ = writeln!(o, "LOG2R\t{val}\t{tgt}\t{}", dtoh(log2_ratio(val, tgt)));
            let _ = writeln!(o, "SAT\t{val}\t{tgt}\t{}", dtoh(target_saturation_score(val, tgt)));
        }
        val += 61;
    }

    let lists: [&[i32]; 6] = [
        &[],
        &[1],
        &[5, 5, 5],
        &[1, 2, 3, 4],
        &[4000, 2048, 1024, 1024, 1, 0, -7],
        &[8192, 16384, 300],
    ];
    for base in lists {
        for cap in [1, 16, 512, 8192, 16384] {
            let xs = uniq_clip_desc(base, cap);
            let _ = write!(o, "UNIQ\t{cap}\t{}", xs.len());
            for x in &xs {
                let _ = write!(o, "\t{x}");
            }
            o.push('\n');
        }
    }

    let _ = writeln!(o, "FLOOR\t{MIN_KV_TOKENS_FLOOR}");
    for n in [0u64, 1, 4095, 4096, 4097, 1_048_576] {
        for a in [1u64, 16, 256, 4096, 2_097_152] {
            let _ = writeln!(o, "ALIGN\t{n}\t{a}\t{}", align_up(n, a));
        }
    }
    o
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[test]
fn rust_policy_layer_is_byte_identical_to_the_cpp_original() {
    let out = render();
    assert_eq!(
        out.len(),
        GOLDEN_BYTES,
        "output length drifted from the C++ oracle ({} vs {GOLDEN_BYTES} bytes)",
        out.len()
    );
    assert_eq!(
        fnv1a64(out.as_bytes()),
        GOLDEN_FNV1A64,
        "output differs from the C++ oracle. Rebuild it per this file's module \
         docs and diff, but do NOT update the constant to match Rust -- the \
         golden is the C++'s output, and re-blessing it deletes the proof."
    );
}

#[test]
fn the_grid_still_covers_what_it_claims_to() {
    let out = render();
    let rows = out.lines().count();
    assert_eq!(rows, GOLDEN_ROWS, "the sweep changed size; the hash above is for {GOLDEN_ROWS}");

    let count = |k: &str| out.lines().filter(|l| l.starts_with(&format!("{k}\t"))).count();
    for kind in [
        "CLAMP", "CLAMP2", "FAMILIES", "PAGEPROF", "PAGECAND", "PAGEDERIVE", "DECODE", "PREFILL",
        "PCAP", "LOG2R", "SAT", "UNIQ", "FLOOR", "ALIGN",
    ] {
        assert!(count(kind) > 0, "no {kind} rows: a whole function stopped being exercised");
    }
    assert_eq!(count("DECODE"), PROFILES.len() * TPS.len() * SMS.len() * MAJORS.len());
    assert_eq!(count("PAGECAND"), PROFILES.len() * TPS.len() * PINNED.len());
}

/// The double columns are the only place a libm difference could hide, so
/// check directly that they are populated with real values rather than a
/// column of zeros or NaNs that would still hash consistently.
#[test]
fn the_float_columns_carry_distinct_finite_values() {
    let out = render();
    let bits = |k: &str| -> Vec<u64> {
        out.lines()
            .filter(|l| l.starts_with(&format!("{k}\t")))
            .map(|l| u64::from_str_radix(l.rsplit('\t').next().unwrap(), 16).unwrap())
            .collect()
    };
    for kind in ["LOG2R", "SAT"] {
        let vs = bits(kind);
        assert!(vs.len() > 100, "{kind} barely sampled");
        assert!(vs.iter().all(|&b| f64::from_bits(b).is_finite()), "{kind} produced a non-finite");
        let mut uniq = vs.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert!(uniq.len() > 50, "{kind} is nearly constant across the sweep ({})", uniq.len());
    }
}

/// Ignored by default; run with `--ignored --nocapture` to regenerate the
/// Rust side for a byte diff against `cpp.txt`.
#[test]
#[ignore = "diagnostic dump, not an assertion"]
fn dump() {
    print!("{}", render());
}
