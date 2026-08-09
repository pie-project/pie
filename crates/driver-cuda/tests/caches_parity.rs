//! Byte-for-byte parity with the C++ `MlaCache`, `DsV4CompressCache` and
//! `SwapPool` allocation paths.
//!
//! The oracle in `tests/oracle/caches/` compiles the real
//! `store/mla_cache.cpp`, `store/dsv4_compress_cache.cpp` and
//! `store/swap_pool.cpp`, replaces only `DeviceTensor::allocate` and
//! `<cuda_runtime.h>` with recorders, and prints exactly what memory each
//! path asks for and in what order. This test reproduces the same sweep
//! against the ports and requires the transcripts to be equal.
//!
//! Run `tests/oracle/caches/run.sh` to regenerate [`GOLDEN_FNV1A64`]. The
//! pinned value is the **C++'s** hash, never this file's: a golden taken from
//! the port would only prove the port agrees with itself.
//!
//! The copy half of `swap_pool.cpp` is covered by `store_parity.rs`; what is
//! new here is the two constructors, which that sweep does not reach.

#![cfg(feature = "_cuda")]

use std::fmt::Write as _;

use driver_cuda::dtype::DType;
use driver_cuda::store::dsv4_compress_cache::DsV4CompressLayout;
use driver_cuda::store::dsv4_geometry::compress_bytes_per_token;
use driver_cuda::store::kv_cache::KvCacheLayout;
use driver_cuda::store::mla_cache::MlaCacheLayout;
use driver_cuda::store::swap_pool::SwapPoolLayout;
use driver_cuda::store::{KvCacheFormat, KvCacheScaleLayout, KvCacheScheme};
use driver_cuda::tensor::TensorSpec;

/// FNV-1a 64 of the C++ oracle's transcript.
///
/// Hand-written rather than `DefaultHasher`, whose output is explicitly not
/// stable across Rust releases.
const GOLDEN_FNV1A64: u64 = 0x7a6f372d76ac1876;

/// Rows the transcript must contain, so a truncated sweep cannot pass by
/// accident.
const GOLDEN_ROWS: usize = 82;

const SEP: char = '\u{1f}';

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn spec_row(s: &TensorSpec) -> String {
    let dims = s
        .shape()
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!("{}[{dims}]={}", s.dtype().name(), s.nbytes())
}

fn row(out: &mut String, id: &str, fields: &[String]) {
    out.push_str(id);
    for f in fields {
        out.push(SEP);
        out.push_str(f);
    }
    out.push('\n');
}

/// Reproduces the recorder's device-buffer naming.
///
/// The recorder hands out `dev0`, `dev1`, ... in allocation order and **only
/// for allocations with bytes** -- a zero-byte request returns null, exactly
/// as the real allocator does, and consumes no ordinal. So the name of a
/// tensor depends on how many non-empty tensors preceded it, which is what
/// makes the naming a check on the allocation order rather than a label.
#[derive(Default)]
struct DevNames {
    next: usize,
}

impl DevNames {
    fn take(&mut self, spec: &TensorSpec) -> String {
        if spec.nbytes() == 0 {
            return "null".to_owned();
        }
        let n = self.next;
        self.next += 1;
        format!("dev{n}+0")
    }
}

// ---------------------------------------------------------------------------
// 1. MlaCache::allocate
// ---------------------------------------------------------------------------

struct MlaCase {
    label: &'static str,
    layers: i32,
    pages: i32,
    page_size: i32,
    lora: i32,
    rope: i32,
    dtype: DType,
}

const MLA_CASES: &[MlaCase] = &[
    MlaCase { label: "tiny", layers: 1, pages: 1, page_size: 1, lora: 1, rope: 1, dtype: DType::Bf16 },
    MlaCase { label: "ds3", layers: 61, pages: 512, page_size: 16, lora: 512, rope: 64, dtype: DType::Bf16 },
    MlaCase { label: "ds3-fp16", layers: 61, pages: 512, page_size: 16, lora: 512, rope: 64, dtype: DType::Fp16 },
    MlaCase { label: "kimi", layers: 27, pages: 128, page_size: 64, lora: 576, rope: 64, dtype: DType::Bf16 },
    MlaCase { label: "lopsided", layers: 3, pages: 7, page_size: 5, lora: 1, rope: 4096, dtype: DType::Bf16 },
    MlaCase { label: "one-layer", layers: 1, pages: 4096, page_size: 1, lora: 512, rope: 64, dtype: DType::Fp16 },
    MlaCase { label: "bad/layers0", layers: 0, pages: 8, page_size: 8, lora: 8, rope: 8, dtype: DType::Bf16 },
    MlaCase { label: "bad/pages0", layers: 8, pages: 0, page_size: 8, lora: 8, rope: 8, dtype: DType::Bf16 },
    MlaCase { label: "bad/psize0", layers: 8, pages: 8, page_size: 0, lora: 8, rope: 8, dtype: DType::Bf16 },
    MlaCase { label: "bad/lora0", layers: 8, pages: 8, page_size: 8, lora: 0, rope: 8, dtype: DType::Bf16 },
    MlaCase { label: "bad/rope0", layers: 8, pages: 8, page_size: 8, lora: 8, rope: 0, dtype: DType::Bf16 },
    MlaCase { label: "bad/layers-1", layers: -1, pages: 8, page_size: 8, lora: 8, rope: 8, dtype: DType::Bf16 },
    MlaCase { label: "bad/pages-1", layers: 8, pages: -1, page_size: 8, lora: 8, rope: 8, dtype: DType::Bf16 },
    MlaCase { label: "bad/psize-1", layers: 8, pages: 8, page_size: -1, lora: 8, rope: 8, dtype: DType::Bf16 },
    MlaCase { label: "bad/lora-1", layers: 8, pages: 8, page_size: 8, lora: -1, rope: 8, dtype: DType::Bf16 },
    MlaCase { label: "bad/rope-1", layers: 8, pages: 8, page_size: 8, lora: 8, rope: -1, dtype: DType::Bf16 },
    MlaCase { label: "bad/two", layers: 0, pages: 0, page_size: 8, lora: 8, rope: 8, dtype: DType::Bf16 },
    MlaCase { label: "bad/all", layers: 0, pages: 0, page_size: 0, lora: 0, rope: 0, dtype: DType::Bf16 },
    MlaCase { label: "bad/fp32", layers: 8, pages: 8, page_size: 8, lora: 8, rope: 8, dtype: DType::Fp32 },
    MlaCase { label: "bad/int8", layers: 8, pages: 8, page_size: 8, lora: 8, rope: 8, dtype: DType::Int8 },
    MlaCase { label: "bad/fp8", layers: 8, pages: 8, page_size: 8, lora: 8, rope: 8, dtype: DType::Fp8E4M3 },
    MlaCase { label: "bad/fp8e5", layers: 8, pages: 8, page_size: 8, lora: 8, rope: 8, dtype: DType::Fp8E5M2 },
    MlaCase { label: "bad/u8", layers: 8, pages: 8, page_size: 8, lora: 8, rope: 8, dtype: DType::Uint8 },
    MlaCase { label: "bad/i32", layers: 8, pages: 8, page_size: 8, lora: 8, rope: 8, dtype: DType::Int32 },
    MlaCase { label: "bad/i64", layers: 8, pages: 8, page_size: 8, lora: 8, rope: 8, dtype: DType::Int64 },
    MlaCase { label: "bad/int4", layers: 8, pages: 8, page_size: 8, lora: 8, rope: 8, dtype: DType::Int4Packed },
    MlaCase { label: "bad/mxfp4", layers: 8, pages: 8, page_size: 8, lora: 8, rope: 8, dtype: DType::Mxfp4Packed },
    MlaCase { label: "bad/order", layers: 0, pages: 8, page_size: 8, lora: 8, rope: 8, dtype: DType::Fp32 },
];

fn render_mla(out: &mut String) {
    for c in MLA_CASES {
        let id = format!("mla/{}", c.label);
        match MlaCacheLayout::plan(c.layers, c.pages, c.page_size, c.lora, c.rope, c.dtype) {
            Err(e) => row(&mut *out, &id, &["throw".to_owned(), e.to_string(), "allocs=".to_owned()]),
            Ok(l) => {
                let mut names = DevNames::default();
                let mut allocs = Vec::new();
                let mut addrs = Vec::new();
                for (_, _, spec) in l.allocation_order() {
                    allocs.push(spec_row(spec));
                    addrs.push(names.take(spec));
                }
                let views = [0, c.layers / 2, c.layers - 1]
                    .iter()
                    .map(|&li| {
                        let v = l.layer_view(li as u32).expect("in range");
                        format!(
                            "L{}:p{}:s{}:r{}:q{}:ckv{}:kpe{}",
                            v.layer,
                            v.num_pages,
                            v.page_size,
                            v.kv_lora_rank,
                            v.qk_rope_head_dim,
                            addrs[(li * 2) as usize],
                            addrs[(li * 2 + 1) as usize],
                        )
                    })
                    .collect::<Vec<_>>();
                let pages = [0, c.layers - 1]
                    .iter()
                    .flat_map(|&li| {
                        l.page_buffers()
                            .into_iter()
                            .enumerate()
                            .map(|(b, pb)| {
                                format!("{}/{}", addrs[(li * 2) as usize + b], pb.page_bytes)
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                row(
                    &mut *out,
                    &id,
                    &[
                        "ok".to_owned(),
                        format!("allocs={}", allocs.join(",")),
                        format!("views={}", views.join(",")),
                        format!("pages={}", pages.join(",")),
                    ],
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 2. DsV4CompressCache::allocate
// ---------------------------------------------------------------------------

struct DsCase {
    label: &'static str,
    ratios: &'static [i32],
    layers: i32,
    head_dim: i32,
    pages: i32,
    page_size: i32,
    fail_at: i32,
}

const DS_CASES: &[DsCase] = &[
    DsCase { label: "none", ratios: &[], layers: 8, head_dim: 128, pages: 16, page_size: 16, fail_at: -1 },
    DsCase { label: "all2", ratios: &[2, 2, 2, 2], layers: 4, head_dim: 128, pages: 16, page_size: 16, fail_at: -1 },
    DsCase { label: "all4", ratios: &[4, 4, 4, 4], layers: 4, head_dim: 128, pages: 16, page_size: 16, fail_at: -1 },
    DsCase { label: "mixed", ratios: &[0, 2, 4, 8, 0, 16], layers: 6, head_dim: 64, pages: 8, page_size: 32, fail_at: -1 },
    DsCase { label: "negatives", ratios: &[-1, 2, -4, 4], layers: 4, head_dim: 64, pages: 8, page_size: 32, fail_at: -1 },
    DsCase { label: "all-zero-ratios", ratios: &[0, 0], layers: 2, head_dim: 64, pages: 8, page_size: 16, fail_at: -1 },
    DsCase { label: "all-neg-ratios", ratios: &[-1, -2], layers: 2, head_dim: 64, pages: 8, page_size: 16, fail_at: -1 },
    DsCase { label: "short-ratios", ratios: &[4], layers: 6, head_dim: 64, pages: 8, page_size: 16, fail_at: -1 },
    DsCase { label: "long-ratios", ratios: &[4, 4, 4, 4, 4, 4], layers: 2, head_dim: 64, pages: 8, page_size: 16, fail_at: -1 },
    DsCase { label: "zero-layers", ratios: &[4, 4], layers: 0, head_dim: 64, pages: 8, page_size: 16, fail_at: -1 },
    DsCase { label: "neg-layers", ratios: &[4, 4], layers: -3, head_dim: 64, pages: 8, page_size: 16, fail_at: -1 },
    DsCase { label: "hd0", ratios: &[4, 2], layers: 2, head_dim: 0, pages: 8, page_size: 16, fail_at: -1 },
    DsCase { label: "hd-neg", ratios: &[4, 2], layers: 2, head_dim: -8, pages: 8, page_size: 16, fail_at: -1 },
    DsCase { label: "pages0", ratios: &[4, 4], layers: 2, head_dim: 64, pages: 0, page_size: 16, fail_at: -1 },
    DsCase { label: "psize0", ratios: &[4, 4], layers: 2, head_dim: 64, pages: 8, page_size: 0, fail_at: -1 },
    DsCase { label: "pages-neg", ratios: &[4, 4], layers: 2, head_dim: 64, pages: -1, page_size: 16, fail_at: -1 },
    DsCase { label: "psize-neg", ratios: &[4, 4], layers: 2, head_dim: 64, pages: 8, page_size: -1, fail_at: -1 },
    DsCase { label: "big", ratios: &[2, 4, 8], layers: 3, head_dim: 192, pages: 64, page_size: 64, fail_at: -1 },
    DsCase { label: "fail0", ratios: &[2, 2], layers: 2, head_dim: 64, pages: 8, page_size: 16, fail_at: 0 },
    DsCase { label: "fail1", ratios: &[2, 2], layers: 2, head_dim: 64, pages: 8, page_size: 16, fail_at: 1 },
    DsCase { label: "fail2", ratios: &[2, 2], layers: 2, head_dim: 64, pages: 8, page_size: 16, fail_at: 2 },
    DsCase { label: "fail3", ratios: &[2, 2], layers: 2, head_dim: 64, pages: 8, page_size: 16, fail_at: 3 },
    DsCase { label: "fail4", ratios: &[2, 2], layers: 2, head_dim: 64, pages: 8, page_size: 16, fail_at: 4 },
    DsCase { label: "fail-never", ratios: &[2, 2], layers: 2, head_dim: 64, pages: 8, page_size: 16, fail_at: 99 },
];

fn render_dsv4(out: &mut String) {
    for c in DS_CASES {
        let id = format!("dsv4/{}", c.label);
        let layout =
            match DsV4CompressLayout::plan(c.ratios, c.layers, c.head_dim, c.pages, c.page_size) {
                Err(e) => {
                    // The oracle prints `what()` for a `std::runtime_error`
                    // and the fixed tag `length_error` for the `resize` throw,
                    // whose text is a libstdc++ artifact rather than a
                    // contract. Which of the two applies is decided by the
                    // library, via the call name it attached to the error --
                    // this only reads that back.
                    let text = if e.call() == "dsv4_compress_cache" {
                        "length_error".to_owned()
                    } else {
                        e.to_string()
                    };
                    row(&mut *out, &id, &["throw".to_owned(), text, "allocs=".to_owned()]);
                    continue;
                }
                Ok(l) => l,
            };

        let mut names = DevNames::default();
        let mut allocs = Vec::new();
        let mut addr = std::collections::HashMap::new();
        for (li, name, spec) in layout.allocation_order() {
            allocs.push(spec_row(spec));
            addr.insert((li, name), names.take(spec));
        }

        // The zeroing pass, replayed through the library's own control flow.
        // The `break`-on-failure is the library's business, not this test's:
        // the closure only reports whether the memset "succeeded", exactly as
        // the recorder does, and never decides what happens next.
        let mut ops: Vec<String> = Vec::new();
        let mut seen = 0i32;
        layout.zero_pass(|li, name, nbytes| {
            let n = seen;
            seen += 1;
            let ok = c.fail_at < 0 || n != c.fail_at;
            ops.push(format!(
                "memset {} val=0 len={nbytes} -> {}",
                addr[&(li, name)],
                if ok { "ok" } else { "FAIL" }
            ));
            if !ok {
                // The C++ clears the sticky error before moving on, so the
                // next unrelated CUDA call does not inherit it.
                ops.push("getlasterror -> 1".to_owned());
            }
            ok
        });

        let layers = (0..c.layers)
            .map(|li| {
                let li = li as usize;
                format!(
                    "{li}:{}:{}",
                    if layout.has_layer(li) { "y" } else { "n" },
                    if layout.has_layer(li) {
                        layout.state_width(li).to_string()
                    } else {
                        "-".to_owned()
                    }
                )
            })
            .collect::<Vec<_>>();

        row(
            &mut *out,
            &id,
            &[
                "ok".to_owned(),
                format!("allocs={}", allocs.join(",")),
                format!("ops={}", ops.join(",")),
                format!("psize={}", layout.page_size()),
                format!("empty={}", if layout.is_empty() { "y" } else { "n" }),
                format!("layers={}", layers.join(",")),
                format!(
                    "bpt={}",
                    compress_bytes_per_token(c.ratios, c.head_dim.unsigned_abs())
                ),
            ],
        );
    }
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
    SwapCase { label: "tiny", layers: 1, pages: 1, page_size: 1, kv_heads: 1, head_dim: 1, dtype: DType::Bf16 },
    SwapCase { label: "llama8b", layers: 32, pages: 64, page_size: 16, kv_heads: 8, head_dim: 128, dtype: DType::Bf16 },
    SwapCase { label: "fp16", layers: 4, pages: 8, page_size: 16, kv_heads: 4, head_dim: 128, dtype: DType::Fp16 },
    SwapCase { label: "fp8", layers: 4, pages: 8, page_size: 16, kv_heads: 4, head_dim: 128, dtype: DType::Fp8E4M3 },
    SwapCase { label: "fp32", layers: 4, pages: 8, page_size: 16, kv_heads: 4, head_dim: 128, dtype: DType::Fp32 },
    SwapCase { label: "int8", layers: 2, pages: 4, page_size: 8, kv_heads: 2, head_dim: 64, dtype: DType::Int8 },
    SwapCase { label: "pages0", layers: 8, pages: 0, page_size: 16, kv_heads: 4, head_dim: 128, dtype: DType::Bf16 },
    SwapCase { label: "pages-neg", layers: 8, pages: -4, page_size: 16, kv_heads: 4, head_dim: 128, dtype: DType::Bf16 },
    SwapCase { label: "layers0", layers: 0, pages: 8, page_size: 16, kv_heads: 4, head_dim: 128, dtype: DType::Bf16 },
    SwapCase { label: "layers-neg", layers: -2, pages: 8, page_size: 16, kv_heads: 4, head_dim: 128, dtype: DType::Bf16 },
    SwapCase { label: "both0", layers: 0, pages: 0, page_size: 16, kv_heads: 4, head_dim: 128, dtype: DType::Bf16 },
    SwapCase { label: "psize0", layers: 4, pages: 8, page_size: 0, kv_heads: 4, head_dim: 128, dtype: DType::Bf16 },
    SwapCase { label: "kvh0", layers: 4, pages: 8, page_size: 16, kv_heads: 0, head_dim: 128, dtype: DType::Bf16 },
    SwapCase { label: "hd0", layers: 4, pages: 8, page_size: 16, kv_heads: 4, head_dim: 0, dtype: DType::Bf16 },
    SwapCase { label: "psize-neg", layers: 4, pages: 8, page_size: -16, kv_heads: 4, head_dim: 128, dtype: DType::Bf16 },
    SwapCase { label: "kvh-neg", layers: 4, pages: 8, page_size: 16, kv_heads: -4, head_dim: 128, dtype: DType::Bf16 },
    SwapCase { label: "hd-neg", layers: 4, pages: 8, page_size: 16, kv_heads: 4, head_dim: -128, dtype: DType::Bf16 },
    SwapCase { label: "hd-neg1", layers: 1, pages: 1, page_size: 1, kv_heads: 1, head_dim: -1, dtype: DType::Bf16 },
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
    CacheCase { label: "bf16", scheme: KvCacheScheme::Native, scale: KvCacheScaleLayout::None, storage: DType::Bf16, block: 0, layers: 4, pages: 8, page_size: 16, kv_heads: 4, head_dim: 128, host_pages: 6 },
    CacheCase { label: "fp16", scheme: KvCacheScheme::Native, scale: KvCacheScaleLayout::None, storage: DType::Fp16, block: 0, layers: 2, pages: 8, page_size: 16, kv_heads: 4, head_dim: 128, host_pages: 3 },
    CacheCase { label: "fp8pt", scheme: KvCacheScheme::Fp8PerTensor, scale: KvCacheScaleLayout::None, storage: DType::Fp8E4M3, block: 0, layers: 3, pages: 8, page_size: 16, kv_heads: 4, head_dim: 128, host_pages: 5 },
    CacheCase { label: "fp8pth", scheme: KvCacheScheme::Fp8PerTokenHead, scale: KvCacheScaleLayout::PerTokenHead, storage: DType::Fp8E4M3, block: 0, layers: 3, pages: 8, page_size: 16, kv_heads: 4, head_dim: 128, host_pages: 5 },
    CacheCase { label: "int8pth", scheme: KvCacheScheme::Int8PerTokenHead, scale: KvCacheScaleLayout::PerTokenHead, storage: DType::Int8, block: 0, layers: 2, pages: 8, page_size: 16, kv_heads: 4, head_dim: 128, host_pages: 4 },
    CacheCase { label: "fp4b16", scheme: KvCacheScheme::Fp4Block, scale: KvCacheScaleLayout::PerTokenHeadBlock, storage: DType::Fp8E4M3, block: 16, layers: 2, pages: 8, page_size: 16, kv_heads: 4, head_dim: 128, host_pages: 4 },
    CacheCase { label: "fp4b32", scheme: KvCacheScheme::Fp4Block, scale: KvCacheScaleLayout::PerTokenHeadBlock, storage: DType::Fp8E4M3, block: 32, layers: 2, pages: 8, page_size: 16, kv_heads: 4, head_dim: 128, host_pages: 4 },
    CacheCase { label: "host0", scheme: KvCacheScheme::Native, scale: KvCacheScaleLayout::None, storage: DType::Bf16, block: 0, layers: 4, pages: 8, page_size: 16, kv_heads: 4, head_dim: 128, host_pages: 0 },
    CacheCase { label: "host-neg", scheme: KvCacheScheme::Native, scale: KvCacheScaleLayout::None, storage: DType::Bf16, block: 0, layers: 4, pages: 8, page_size: 16, kv_heads: 4, head_dim: 128, host_pages: -3 },
    CacheCase { label: "nolayers", scheme: KvCacheScheme::Native, scale: KvCacheScaleLayout::None, storage: DType::Bf16, block: 0, layers: 0, pages: 8, page_size: 16, kv_heads: 4, head_dim: 128, host_pages: 4 },
    CacheCase { label: "devpages0", scheme: KvCacheScheme::Native, scale: KvCacheScaleLayout::None, storage: DType::Bf16, block: 0, layers: 2, pages: 0, page_size: 16, kv_heads: 4, head_dim: 128, host_pages: 4 },
    CacheCase { label: "big", scheme: KvCacheScheme::Native, scale: KvCacheScaleLayout::None, storage: DType::Bf16, block: 0, layers: 8, pages: 32, page_size: 32, kv_heads: 8, head_dim: 64, host_pages: 12 },
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
    render_mla(&mut o);
    render_dsv4(&mut o);
    render_swap_uniform(&mut o);
    render_swap_for_cache(&mut o);

    if let Ok(path) = std::env::var("CACHES_RUST_OUT") {
        std::fs::write(&path, &o).expect("write transcript");
    }

    assert_eq!(o.lines().count(), GOLDEN_ROWS, "row count drifted");
    assert_eq!(
        fnv1a64(o.as_bytes()),
        GOLDEN_FNV1A64,
        "transcript diverged from the C++; set CACHES_RUST_OUT and \
         CACHES_ORACLE_OUT and diff them"
    );
}
