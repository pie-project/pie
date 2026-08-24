//! Byte-for-byte parity with the C++ `SwapPool` allocation paths.
//!
//! The oracle in `tests/oracle/caches/` compiles the real
//! `store/mla_cache.cpp`, `store/dsv4_compress_cache.cpp` and
//! `store/swap_pool.cpp`, replaces only `DeviceTensor::allocate` and
//! `<cuda_runtime.h>` with recorders, and prints exactly what memory each
//! path asks for and in what order. This test reproduces the same sweep
//! against the ports and requires the transcripts to be equal.
//!
//! TWO OF THE FOUR SECTIONS ARE RETIRED. `MlaCache::allocate` and
//! `DsV4CompressCache::allocate` swept `pools::mla_cache` and
//! `pools::compressed_plane_cache`, both deleted for want of a production
//! reader — `Deployment::of` refuses every MLA/latent SKU by name, and the
//! `model_costs` arms that sized these pools went with the legacy walk.
//! Their rows are NOT re-blessed out of the golden, which would delete the
//! proof for the half that survives: they were the transcript's PREFIX, and
//! FNV-1a chains, so [`RETIRED_PREFIX_FNV1A64`] carries the hash state they
//! left and the surviving sweep still ends at the C++'s own
//! [`GOLDEN_FNV1A64`].
//!
//! `tests/oracle/caches/run.sh` can no longer be run — its inputs were deleted, see `oracle_census.rs`. It is kept as the description of how this golden was taken, which is read but not re-derived. It once regenerated [`GOLDEN_FNV1A64`]. The
//! pinned value is the **C++'s** hash, never this file's: a golden taken from
//! the port would only prove the port agrees with itself.
//!
//! The copy half of `swap_pool.cpp` is covered by `store_parity.rs`; what is
//! new here is the two constructors, which that sweep does not reach.

#![cfg(feature = "_cuda")]

use std::fmt::Write as _;

use driver_cuda::dtype::DType;
use driver_cuda::layout::{KvCacheFormat, KvCacheScaleLayout, KvCacheScheme};
use driver_cuda::pools::kv_cache::KvCacheLayout;
use driver_cuda::pools::swap_pool::SwapPoolLayout;

/// FNV-1a 64 of the C++ oracle's WHOLE transcript, all four sections.
///
/// Hand-written rather than `DefaultHasher`, whose output is explicitly not
/// stable across Rust releases.
const GOLDEN_FNV1A64: u64 = 0x7a6f372d76ac1876;

/// The hash state the two retired sections left, and the rows they wrote.
///
/// Not a second golden and not a re-blessing: the C++ transcript began with
/// `MlaCache` and `DsV4CompressCache`, so hashing the surviving two sections
/// FROM this state reaches [`GOLDEN_FNV1A64`] exactly, and the swap half is
/// still pinned to the C++ and not to itself. It was taken by chaining the
/// full transcript this file rendered while all four sections were green.
const RETIRED_PREFIX_FNV1A64: u64 = 0x457f_98ee_96c1_717d;
const RETIRED_PREFIX_ROWS: usize = 52;

/// Rows the whole transcript must contain, so a truncated sweep cannot pass
/// by accident. [`RETIRED_PREFIX_ROWS`] of them are no longer rendered.
const GOLDEN_ROWS: usize = 82;

const SEP: char = '\u{1f}';

/// FNV-1a 64 continued from `h`, so a transcript can be hashed in pieces.
fn fnv1a64_from(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn row(out: &mut String, id: &str, fields: &[String]) {
    out.push_str(id);
    for f in fields {
        out.push(SEP);
        out.push_str(f);
    }
    out.push('\n');
}

// ---------------------------------------------------------------------------
// 3. SwapPool::allocate
// ---------------------------------------------------------------------------

struct SwapCase {
    label: &'static str,
    layers: i32,
    pages: i32,
    page_size: i32,
    kv_heads: i32,
    head_dim: i32,
    dtype: DType,
}

const SWAP_CASES: &[SwapCase] = &[
    SwapCase {
        label: "tiny",
        layers: 1,
        pages: 1,
        page_size: 1,
        kv_heads: 1,
        head_dim: 1,
        dtype: DType::Bf16,
    },
    SwapCase {
        label: "llama8b",
        layers: 32,
        pages: 64,
        page_size: 16,
        kv_heads: 8,
        head_dim: 128,
        dtype: DType::Bf16,
    },
    SwapCase {
        label: "fp16",
        layers: 4,
        pages: 8,
        page_size: 16,
        kv_heads: 4,
        head_dim: 128,
        dtype: DType::Fp16,
    },
    SwapCase {
        label: "fp8",
        layers: 4,
        pages: 8,
        page_size: 16,
        kv_heads: 4,
        head_dim: 128,
        dtype: DType::Fp8E4M3,
    },
    SwapCase {
        label: "fp32",
        layers: 4,
        pages: 8,
        page_size: 16,
        kv_heads: 4,
        head_dim: 128,
        dtype: DType::Fp32,
    },
    SwapCase {
        label: "int8",
        layers: 2,
        pages: 4,
        page_size: 8,
        kv_heads: 2,
        head_dim: 64,
        dtype: DType::Int8,
    },
    SwapCase {
        label: "pages0",
        layers: 8,
        pages: 0,
        page_size: 16,
        kv_heads: 4,
        head_dim: 128,
        dtype: DType::Bf16,
    },
    SwapCase {
        label: "pages-neg",
        layers: 8,
        pages: -4,
        page_size: 16,
        kv_heads: 4,
        head_dim: 128,
        dtype: DType::Bf16,
    },
    SwapCase {
        label: "layers0",
        layers: 0,
        pages: 8,
        page_size: 16,
        kv_heads: 4,
        head_dim: 128,
        dtype: DType::Bf16,
    },
    SwapCase {
        label: "layers-neg",
        layers: -2,
        pages: 8,
        page_size: 16,
        kv_heads: 4,
        head_dim: 128,
        dtype: DType::Bf16,
    },
    SwapCase {
        label: "both0",
        layers: 0,
        pages: 0,
        page_size: 16,
        kv_heads: 4,
        head_dim: 128,
        dtype: DType::Bf16,
    },
    SwapCase {
        label: "psize0",
        layers: 4,
        pages: 8,
        page_size: 0,
        kv_heads: 4,
        head_dim: 128,
        dtype: DType::Bf16,
    },
    SwapCase {
        label: "kvh0",
        layers: 4,
        pages: 8,
        page_size: 16,
        kv_heads: 0,
        head_dim: 128,
        dtype: DType::Bf16,
    },
    SwapCase {
        label: "hd0",
        layers: 4,
        pages: 8,
        page_size: 16,
        kv_heads: 4,
        head_dim: 0,
        dtype: DType::Bf16,
    },
    SwapCase {
        label: "psize-neg",
        layers: 4,
        pages: 8,
        page_size: -16,
        kv_heads: 4,
        head_dim: 128,
        dtype: DType::Bf16,
    },
    SwapCase {
        label: "kvh-neg",
        layers: 4,
        pages: 8,
        page_size: 16,
        kv_heads: -4,
        head_dim: 128,
        dtype: DType::Bf16,
    },
    SwapCase {
        label: "hd-neg",
        layers: 4,
        pages: 8,
        page_size: 16,
        kv_heads: 4,
        head_dim: -128,
        dtype: DType::Bf16,
    },
    SwapCase {
        label: "hd-neg1",
        layers: 1,
        pages: 1,
        page_size: 1,
        kv_heads: 1,
        head_dim: -1,
        dtype: DType::Bf16,
    },
];

/// The recorder's transcript of the operations a pool's construction issues.
fn pool_ops(p: &SwapPoolLayout) -> Vec<String> {
    let mut ops = Vec::new();
    if p.streams().evict {
        ops.push("stream_create s0 flags=1".to_owned());
    }
    if p.streams().restore {
        ops.push("stream_create s1 flags=1".to_owned());
    }
    for (i, b) in p.buffers().iter().enumerate() {
        ops.push(format!("mallochost host{i} bytes={}", b.nbytes));
    }
    ops
}

fn render_swap_uniform(out: &mut String) {
    for c in SWAP_CASES {
        let p = SwapPoolLayout::uniform(
            c.layers,
            c.pages,
            c.page_size,
            c.kv_heads,
            c.head_dim,
            c.dtype,
        );
        let s = p.streams();
        row(
            &mut *out,
            &format!("swap/uniform/{}", c.label),
            &[
                "ok".to_owned(),
                format!("ops={}", pool_ops(&p).join(",")),
                format!("layers={}", p.num_layers()),
                format!("pages={}", p.num_pages()),
                format!("bpp={}", p.bytes_per_page()),
                format!(
                    "streams={}{}",
                    if s.evict { "y" } else { "n" },
                    if s.restore { "y" } else { "n" }
                ),
                // Two `cudaStreamCreateWithFlags` calls always yield two
                // distinct handles, so this can only differ if one was never
                // created.
                format!("distinct={}", if s.evict && s.restore { "y" } else { "n" }),
            ],
        );
    }
}

// ---------------------------------------------------------------------------
// 4. SwapPool::allocate_for_cache
// ---------------------------------------------------------------------------

struct CacheCase {
    label: &'static str,
    scheme: KvCacheScheme,
    scale: KvCacheScaleLayout,
    storage: DType,
    block: u32,
    layers: i32,
    pages: i32,
    page_size: i32,
    kv_heads: i32,
    head_dim: i32,
    host_pages: i32,
}

const CACHE_CASES: &[CacheCase] = &[
    CacheCase {
        label: "bf16",
        scheme: KvCacheScheme::Native,
        scale: KvCacheScaleLayout::None,
        storage: DType::Bf16,
        block: 0,
        layers: 4,
        pages: 8,
        page_size: 16,
        kv_heads: 4,
        head_dim: 128,
        host_pages: 6,
    },
    CacheCase {
        label: "fp16",
        scheme: KvCacheScheme::Native,
        scale: KvCacheScaleLayout::None,
        storage: DType::Fp16,
        block: 0,
        layers: 2,
        pages: 8,
        page_size: 16,
        kv_heads: 4,
        head_dim: 128,
        host_pages: 3,
    },
    CacheCase {
        label: "fp8pt",
        scheme: KvCacheScheme::Fp8PerTensor,
        scale: KvCacheScaleLayout::None,
        storage: DType::Fp8E4M3,
        block: 0,
        layers: 3,
        pages: 8,
        page_size: 16,
        kv_heads: 4,
        head_dim: 128,
        host_pages: 5,
    },
    CacheCase {
        label: "fp8pth",
        scheme: KvCacheScheme::Fp8PerTokenHead,
        scale: KvCacheScaleLayout::PerTokenHead,
        storage: DType::Fp8E4M3,
        block: 0,
        layers: 3,
        pages: 8,
        page_size: 16,
        kv_heads: 4,
        head_dim: 128,
        host_pages: 5,
    },
    CacheCase {
        label: "int8pth",
        scheme: KvCacheScheme::Int8PerTokenHead,
        scale: KvCacheScaleLayout::PerTokenHead,
        storage: DType::Int8,
        block: 0,
        layers: 2,
        pages: 8,
        page_size: 16,
        kv_heads: 4,
        head_dim: 128,
        host_pages: 4,
    },
    CacheCase {
        label: "fp4b16",
        scheme: KvCacheScheme::Fp4Block,
        scale: KvCacheScaleLayout::PerTokenHeadBlock,
        storage: DType::Fp8E4M3,
        block: 16,
        layers: 2,
        pages: 8,
        page_size: 16,
        kv_heads: 4,
        head_dim: 128,
        host_pages: 4,
    },
    CacheCase {
        label: "fp4b32",
        scheme: KvCacheScheme::Fp4Block,
        scale: KvCacheScaleLayout::PerTokenHeadBlock,
        storage: DType::Fp8E4M3,
        block: 32,
        layers: 2,
        pages: 8,
        page_size: 16,
        kv_heads: 4,
        head_dim: 128,
        host_pages: 4,
    },
    CacheCase {
        label: "host0",
        scheme: KvCacheScheme::Native,
        scale: KvCacheScaleLayout::None,
        storage: DType::Bf16,
        block: 0,
        layers: 4,
        pages: 8,
        page_size: 16,
        kv_heads: 4,
        head_dim: 128,
        host_pages: 0,
    },
    CacheCase {
        label: "host-neg",
        scheme: KvCacheScheme::Native,
        scale: KvCacheScaleLayout::None,
        storage: DType::Bf16,
        block: 0,
        layers: 4,
        pages: 8,
        page_size: 16,
        kv_heads: 4,
        head_dim: 128,
        host_pages: -3,
    },
    CacheCase {
        label: "nolayers",
        scheme: KvCacheScheme::Native,
        scale: KvCacheScaleLayout::None,
        storage: DType::Bf16,
        block: 0,
        layers: 0,
        pages: 8,
        page_size: 16,
        kv_heads: 4,
        head_dim: 128,
        host_pages: 4,
    },
    CacheCase {
        label: "devpages0",
        scheme: KvCacheScheme::Native,
        scale: KvCacheScaleLayout::None,
        storage: DType::Bf16,
        block: 0,
        layers: 2,
        pages: 0,
        page_size: 16,
        kv_heads: 4,
        head_dim: 128,
        host_pages: 4,
    },
    CacheCase {
        label: "big",
        scheme: KvCacheScheme::Native,
        scale: KvCacheScaleLayout::None,
        storage: DType::Bf16,
        block: 0,
        layers: 8,
        pages: 32,
        page_size: 32,
        kv_heads: 8,
        head_dim: 64,
        host_pages: 12,
    },
];

fn render_swap_for_cache(out: &mut String) {
    for c in CACHE_CASES {
        let fmt = KvCacheFormat::from_parts(c.label, c.scheme, c.scale, c.storage, c.block);
        let cache = KvCacheLayout::plan(
            c.layers,
            c.pages,
            c.page_size,
            c.kv_heads,
            c.head_dim,
            fmt,
            false,
        )
        .expect("device cache plans");

        let dev_allocs = cache
            .slots()
            .iter()
            .map(|s| {
                [&s.k, &s.v, &s.k_scale, &s.v_scale, &s.k_bf16, &s.v_bf16]
                    .into_iter()
                    .flatten()
                    .count()
            })
            .sum::<usize>();

        let dev: Vec<Vec<u64>> = (0..c.layers)
            .map(|l| cache.page_buffers(l).into_iter().map(|(_, b)| b).collect())
            .collect();

        let p = SwapPoolLayout::for_cache(&dev, c.host_pages, c.page_size, c.kv_heads, c.head_dim);
        let s = p.streams();
        let mut ops = vec!["--".to_owned()];
        ops.extend(pool_ops(&p));

        let devbufs = dev
            .iter()
            .enumerate()
            .map(|(l, widths)| {
                let mut r = format!("{l}:");
                for w in widths {
                    let _ = write!(r, "{w}/");
                }
                r
            })
            .collect::<Vec<_>>();

        row(
            &mut *out,
            &format!("swap/cache/{}", c.label),
            &[
                "ok".to_owned(),
                format!("devallocs={dev_allocs}"),
                format!("ops={}", ops.join(",")),
                format!("layers={}", p.num_layers()),
                format!("pages={}", p.num_pages()),
                format!("bpp={}", p.bytes_per_page()),
                format!(
                    "streams={}{}",
                    if s.evict { "y" } else { "n" },
                    if s.restore { "y" } else { "n" }
                ),
                format!("dev={}", devbufs.join(",")),
            ],
        );
    }
}

#[test]
fn matches_the_cpp_cache_allocation_transcript() {
    let mut o = String::new();
    render_swap_uniform(&mut o);
    render_swap_for_cache(&mut o);

    if let Ok(path) = std::env::var("CACHES_RUST_OUT") {
        std::fs::write(&path, &o).expect("write transcript");
    }

    assert_eq!(
        o.lines().count() + RETIRED_PREFIX_ROWS,
        GOLDEN_ROWS,
        "row count drifted"
    );
    assert_eq!(
        fnv1a64_from(RETIRED_PREFIX_FNV1A64, o.as_bytes()),
        GOLDEN_FNV1A64,
        "transcript diverged from the C++; set CACHES_RUST_OUT and \
         CACHES_ORACLE_OUT and diff them"
    );
}
