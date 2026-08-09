//! Differential proof that [`driver_cuda::layout::kv_geometry`] reproduces
//! the C++ KV page-geometry functions exactly.
//!
//! Same protocol as `kv_format_parity.rs` and `planner_policy_parity.rs`:
//! build the real C++ as an oracle, sweep both over one grid, require
//! byte-identical output, pin the **C++ side's** hash.
//!
//! # The oracle
//!
//! `kv_cache.cpp` as a whole is not compilable off-GPU -- it pulls in
//! `cuda_check.hpp`, `elastic.hpp`, and the model config layer. The three free
//! functions at the bottom of it are, though: they read three `HfConfig` ints
//! and call into `kv_cache_format.cpp`, which builds with plain `g++`. So the
//! oracle **extracts those three functions verbatim with `awk`** and links
//! them against the real format implementation, exactly as the planner-policy
//! oracle does with the anonymous namespace.
//!
//! ```text
//! mkdir -p /tmp/kvgeo && cd /tmp/kvgeo
//! SRC=crates/driver-cuda/csrc/src
//!
//! # The three functions, lifted from the real file. Stop after the third
//! # closing brace at column 0.
//! awk '/^std::size_t kv_cache_device_bytes_per_page\(/{f=1}
//!      f{print; if($0=="}"){n++; if(n==3) exit}}' \
//!     "$SRC/store/kv_cache.cpp" > geo.inc
//!
//! g++ -std=c++20 -I "$SRC" -I crates/kernels-cuda/csrc/src -I . \
//!     oracle.cpp "$SRC/store/kv_cache_format.cpp" -o oracle
//! ./oracle > cpp.txt
//! ```
//!
//! `oracle.cpp` declares a stub `HfConfig` carrying only
//! `num_key_value_heads`, `num_hidden_layers`, and `head_dim_kernel` -- the
//! entire config surface these functions read -- and `#include`s `geo.inc`
//! inside `namespace pie_cuda_driver`.
//!
//! # Why the `PERLAYER` rows matter most
//!
//! `kv_page_bytes_per_layer` carries a fixed production bug in its comments:
//! charging Gemma-4's wide full-attention layers the flat
//! `num_key_value_heads` overcounted them up to 4x, and the planner then
//! rejected every candidate layout with "no viable forward/KV layout fits
//! budget". The six `pattern` values sweep each override slice independently
//! and together, so a port that drops the `per_layer_num_kv_heads` path -- the
//! fix itself -- cannot pass.

use driver_cuda::layout::KvCacheFormat;
use driver_cuda::layout::kv_geometry::{
    LayerShapes, device_bytes_per_page, page_bytes_homogeneous, page_bytes_per_layer,
};
use std::fmt::Write as _;

/// FNV-1a 64 of the **C++ oracle's** stdout.
const GOLDEN_FNV1A64: u64 = 0xaaaf_474b_88b4_8f96;
/// Byte count of the same output.
const GOLDEN_BYTES: usize = 1_025_256;
/// Row count, as a guard on the grid rather than on the values.
const GOLDEN_ROWS: usize = 26_622;

/// Only the aliases that parse. An unparseable one throws in the C++, so the
/// oracle cannot sweep it; `kv_format_parity.rs` covers the rejection paths.
const NAMES: [&str; 9] = [
    "auto",
    "bf16",
    "bfloat16",
    "fp8_e4m3",
    "fp8_e5m2",
    "int8_per_token_head",
    "fp8_per_token_head",
    "fp4_e2m1",
    "nvfp4",
];

const PAGES: [u32; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];
const HEADS: [u32; 7] = [1, 2, 3, 4, 8, 16, 64];
const DIMS: [u32; 12] = [1, 2, 3, 16, 32, 64, 72, 96, 128, 192, 256, 576];

const TPS: [i32; 5] = [0, 1, 2, 4, 8];
const LAYERS: [u32; 6] = [1, 2, 28, 32, 48, 80];
const HDIMS: [u32; 5] = [16, 64, 128, 192, 576];

// The per-layer sweep is narrower on each axis because it multiplies by six
// shape patterns; the wide coverage of the arithmetic itself is PAGE's job.
const P_TPS: [i32; 4] = [0, 1, 2, 8];
const P_LAYERS: [u32; 4] = [1, 2, 32, 80];
const P_HEADS: [u32; 4] = [1, 2, 8, 64];
const P_HDIMS: [u32; 3] = [16, 128, 576];

/// The C++ prints each override vector as `len:sum` rather than in full: the
/// vectors are derived from `(layers, head_dim, kv_heads, pattern)`, so
/// echoing them whole would multiply the transcript without adding coverage.
/// Length and sum still catch a port that builds different inputs.
fn vec_fp(v: &[u32]) -> String {
    format!(
        "{}:{}",
        v.len(),
        v.iter().map(|&x| u64::from(x)).sum::<u64>()
    )
}

/// The six shape patterns, built exactly as the oracle builds them.
///
/// Slices are always sized to `layers` precisely. The C++ indexes them
/// unguarded, so a short vector would be undefined behaviour rather than a
/// test case, and the sweep must not generate one.
fn pattern(p: u32, layers: u32, head_dim: u32, kv_heads: u32) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut phd = Vec::new();
    let mut pkv = Vec::new();
    let mut src = Vec::new();
    if matches!(p, 1 | 4 | 5) {
        phd = (0..layers)
            .map(|i| {
                if i % 2 == 1 {
                    head_dim
                } else {
                    (head_dim / 2).max(1)
                }
            })
            .collect();
    }
    if matches!(p, 2 | 4 | 5) {
        pkv = (0..layers)
            .map(|i| if i % 4 == 0 { kv_heads * 4 } else { kv_heads })
            .collect();
    }
    if matches!(p, 3 | 5) {
        src = (0..layers)
            .map(|i| if i % 2 == 1 { i - 1 } else { i })
            .collect();
    }
    (phd, pkv, src)
}

fn render() -> String {
    let mut o = String::with_capacity(GOLDEN_BYTES + 4096);

    for name in NAMES {
        let f = KvCacheFormat::from_name(name).expect("alias parses");
        for p in PAGES {
            for h in HEADS {
                for d in DIMS {
                    let _ = writeln!(
                        o,
                        "PAGE\t{name}\t{p}\t{h}\t{d}\t{}",
                        device_bytes_per_page(&f, p, h, d)
                    );
                }
            }
        }
    }

    for name in NAMES {
        let f = KvCacheFormat::from_name(name).expect("alias parses");
        for tp in TPS {
            for l in LAYERS {
                for kvh in HEADS {
                    for hd in HDIMS {
                        let _ = writeln!(
                            o,
                            "HOMO\t{name}\t{tp}\t{l}\t{kvh}\t{hd}\t{}",
                            page_bytes_homogeneous(l, kvh, hd, tp, &f)
                        );
                    }
                }
            }
        }
    }

    for name in NAMES {
        let f = KvCacheFormat::from_name(name).expect("alias parses");
        for tp in P_TPS {
            for l in P_LAYERS {
                for kvh in P_HEADS {
                    for hd in P_HDIMS {
                        for p in 0..6 {
                            let (phd, pkv, src) = pattern(p, l, hd, kvh);
                            let shapes = LayerShapes {
                                head_dim: &phd,
                                num_kv_heads: &pkv,
                                source_layer: &src,
                            };
                            let _ = writeln!(
                                o,
                                "PERLAYER\t{name}\t{tp}\t{l}\t{kvh}\t{hd}\t{p}\t{}\t{}\t{}\t{}",
                                vec_fp(&phd),
                                vec_fp(&pkv),
                                vec_fp(&src),
                                page_bytes_per_layer(l, kvh, hd, &shapes, tp, &f)
                            );
                        }
                    }
                }
            }
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
fn rust_geometry_is_byte_identical_to_the_cpp_original() {
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
    assert_eq!(out.lines().count(), GOLDEN_ROWS);
    let count = |k: &str| {
        out.lines()
            .filter(|l| l.starts_with(&format!("{k}\t")))
            .count()
    };
    assert_eq!(
        count("PAGE"),
        NAMES.len() * PAGES.len() * HEADS.len() * DIMS.len()
    );
    assert_eq!(
        count("HOMO"),
        NAMES.len() * TPS.len() * LAYERS.len() * HEADS.len() * HDIMS.len()
    );
    assert_eq!(
        count("PERLAYER"),
        NAMES.len() * P_TPS.len() * P_LAYERS.len() * P_HEADS.len() * P_HDIMS.len() * 6
    );
}

/// The sweep would still hash consistently if every quantised format silently
/// returned its unquantised size, so check the axis the port exists to get
/// right actually varies.
#[test]
fn the_sweep_actually_distinguishes_formats_and_shapes() {
    let out = render();
    let page: Vec<&str> = out.lines().filter(|l| l.starts_with("PAGE\t")).collect();
    let mut sizes: Vec<&str> = page
        .iter()
        .map(|l| l.rsplit('\t').next().unwrap())
        .collect();
    sizes.sort_unstable();
    sizes.dedup();
    assert!(
        sizes.len() > 200,
        "PAGE is nearly constant across the sweep ({})",
        sizes.len()
    );

    // A quantised format must not cost the same as bf16 at the same shape.
    let at = |n: &str| -> &str {
        out.lines()
            .find(|l| l.starts_with(&format!("PAGE\t{n}\t16\t8\t128\t")))
            .unwrap()
            .rsplit('\t')
            .next()
            .unwrap()
    };
    assert_ne!(at("bf16"), at("fp8_e4m3"));
    assert_ne!(at("fp8_e4m3"), at("nvfp4"));

    // And the six per-layer patterns must not collapse into one answer.
    let mut per: Vec<&str> = out
        .lines()
        .filter(|l| l.starts_with("PERLAYER\tbf16\t1\t80\t8\t128\t"))
        .map(|l| l.rsplit('\t').next().unwrap())
        .collect();
    let total = per.len();
    per.sort_unstable();
    per.dedup();
    assert_eq!(total, 6);
    assert!(
        per.len() >= 5,
        "the shape patterns are not being distinguished ({})",
        per.len()
    );
}

/// Ignored by default; run with `--ignored --nocapture` to regenerate the Rust
/// side for a byte diff against `cpp.txt`.
#[test]
#[ignore = "diagnostic dump, not an assertion"]
fn dump() {
    print!("{}", render());
}
