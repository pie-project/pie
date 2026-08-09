//! Byte-for-byte parity with the C++ `KvCache` allocation path.
//!
//! The oracle in `tests/oracle/kv_cache/` compiles the real
//! `store/kv_cache.cpp` and `store/kv_cache_format.cpp`, replaces only
//! `DeviceTensor::allocate` with a recorder, and prints the exact sequence of
//! tensor allocations each configuration makes. This test reproduces the same
//! sweep against [`KvCacheLayout`] and requires the transcripts to be equal.
//!
//! Run `tests/oracle/kv_cache/run.sh` to regenerate [`GOLDEN_FNV1A64`]. The
//! pinned value is the **C++'s** hash, never this file's: a golden taken from
//! the port would only prove the port agrees with itself.

#![cfg(feature = "_cuda")]

use std::fmt::Write as _;

use driver_cuda::dtype::DType;
use driver_cuda::layout::kv_geometry::{self, LayerShapes};
use driver_cuda::layout::{KvCacheFormat, KvCacheScaleLayout, KvCacheScheme};
use driver_cuda::pools::kv_cache::{self, KvCacheLayout, LayerSlot, PerLayer};
use driver_cuda::tensor::TensorSpec;

/// FNV-1a 64 of the C++ oracle's transcript.
///
/// Hand-written rather than `DefaultHasher`, whose output is explicitly not
/// stable across Rust releases.
const GOLDEN_FNV1A64: u64 = 0x5fc3b41ce3750198;

/// Rows the transcript must contain, so a truncated sweep cannot pass by
/// accident.
const GOLDEN_ROWS: usize = 4242;

const SEP: char = '\u{1f}';

struct FormatCase {
    label: &'static str,
    scheme: KvCacheScheme,
    scale: KvCacheScaleLayout,
    storage: DType,
    block: u32,
}

const FORMATS: &[FormatCase] = &[
    FormatCase {
        label: "bf16",
        scheme: KvCacheScheme::Native,
        scale: KvCacheScaleLayout::None,
        storage: DType::Bf16,
        block: 0,
    },
    FormatCase {
        label: "fp16",
        scheme: KvCacheScheme::Native,
        scale: KvCacheScaleLayout::None,
        storage: DType::Fp16,
        block: 0,
    },
    FormatCase {
        label: "fp8pt",
        scheme: KvCacheScheme::Fp8PerTensor,
        scale: KvCacheScaleLayout::None,
        storage: DType::Fp8E4M3,
        block: 0,
    },
    FormatCase {
        label: "fp8pth",
        scheme: KvCacheScheme::Fp8PerTokenHead,
        scale: KvCacheScaleLayout::PerTokenHead,
        storage: DType::Fp8E4M3,
        block: 0,
    },
    FormatCase {
        label: "int8pth",
        scheme: KvCacheScheme::Int8PerTokenHead,
        scale: KvCacheScaleLayout::PerTokenHead,
        storage: DType::Int8,
        block: 0,
    },
    FormatCase {
        label: "fp4b16",
        scheme: KvCacheScheme::Fp4Block,
        scale: KvCacheScaleLayout::PerTokenHeadBlock,
        storage: DType::Fp8E4M3,
        block: 16,
    },
    FormatCase {
        label: "fp4b32",
        scheme: KvCacheScheme::Fp4Block,
        scale: KvCacheScaleLayout::PerTokenHeadBlock,
        storage: DType::Fp8E4M3,
        block: 32,
    },
    FormatCase {
        label: "fp4b0",
        scheme: KvCacheScheme::Fp4Block,
        scale: KvCacheScaleLayout::PerTokenHeadBlock,
        storage: DType::Fp8E4M3,
        block: 0,
    },
    FormatCase {
        label: "bf16scaled",
        scheme: KvCacheScheme::Native,
        scale: KvCacheScaleLayout::PerTokenHead,
        storage: DType::Bf16,
        block: 0,
    },
];

const SHARES: &[(&str, &[i32])] = &[
    ("none", &[]),
    ("self", &[0, 1, 2, 3, 4, 5]),
    ("all-to-0", &[0, 0, 0, 0, 0, 0]),
    ("pairs", &[0, 0, 2, 2, 4, 4]),
    ("gemma-5in6", &[0, 0, 0, 0, 0, 5]),
    ("last-source", &[5, 5, 5, 5, 5, 5]),
    ("forward-ref", &[1, 1, 3, 3, 5, 5]),
];

fn make_format(f: &FormatCase) -> KvCacheFormat {
    KvCacheFormat::from_parts(f.label, f.scheme, f.scale, f.storage, f.block)
}

/// The recorder's rendering of one `DeviceTensor::allocate` call.
fn spec_row(s: &TensorSpec) -> String {
    let dims = s
        .shape()
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!("{}[{dims}]={}", s.dtype().name(), s.nbytes())
}

/// Replay the allocation order the C++ makes for one slot.
///
/// The order is load-bearing and is not the field order of [`LayerSlot`]: the
/// C++ pushes K then V for the storage tier, then the scale pair, then the
/// mirror pair, and the envelope tier only after every slot is built.
fn slot_rows(slot: &LayerSlot, out: &mut Vec<String>) {
    for spec in [
        &slot.k,
        &slot.v,
        &slot.k_scale,
        &slot.v_scale,
        &slot.k_bf16,
        &slot.v_bf16,
    ]
    .into_iter()
    .flatten()
    {
        {
            out.push(spec_row(spec));
        }
    }
}

fn alloc_log(layout: &KvCacheLayout) -> Vec<String> {
    let mut out = Vec::new();
    for slot in layout.slots() {
        slot_rows(slot, &mut out);
    }
    if layout.envelopes_enabled() {
        // `allocate_envelopes_` swaps the allocator binding to the default for
        // its whole body and restores it on scope exit, because the KV arena
        // is elastic and an envelope must not live in uncommitted VA.
        out.push("bind(default)".to_owned());
        for slot in layout.slots() {
            for spec in [&slot.k_env_min, &slot.k_env_max].into_iter().flatten() {
                {
                    out.push(spec_row(spec));
                }
            }
        }
        out.push("bind(default)".to_owned());
    }
    out
}

fn seed_log(layout: &KvCacheLayout) -> Vec<String> {
    let mut out = Vec::new();
    if !layout.envelopes_enabled() {
        return out;
    }
    for (i, slot) in layout.slots().iter().enumerate() {
        if !slot.k_env_min.as_ref().is_some_and(|t| !t.is_empty()) {
            continue;
        }
        let layer = i32::try_from(i).unwrap_or(0);
        out.push(format!(
            "seed(min=p,max=p,pages={},kvh={},hd={})",
            layout.num_pages(),
            layout.num_kv_heads_at(layer),
            layout.head_dim_at(layer)
        ));
    }
    out
}

/// Mirror of the oracle's `readback`, which reports what the constructed cache
/// says about itself rather than what it allocated.
fn readback(layout: &KvCacheLayout) -> Vec<String> {
    let f = layout.format();
    let native = f.is_native_bf16();
    (0..layout.num_layers())
        .map(|i| {
            let src = layout.resolve(i);
            let slot = &layout.slots()[usize::try_from(src).unwrap_or(0)];
            // A zero-byte allocation returns a NULL pointer, so a cache with
            // no pages reports every tier as absent even though the shape was
            // computed and the tensor exists. `is_empty` is the port of the
            // C++'s `ptr_ == nullptr`, which is what the readback tests.
            let live = |o: &Option<TensorSpec>| o.as_ref().is_some_and(|s| !s.is_empty());
            let k_attn = if native {
                if live(&slot.k) { "same" } else { "null" }
            } else if live(&slot.k_bf16) {
                "mirror"
            } else {
                "null"
            };
            let v_attn = if native {
                if live(&slot.v) { "same" } else { "null" }
            } else if live(&slot.v_bf16) {
                "mirror"
            } else {
                "null"
            };
            let p = |o: &Option<TensorSpec>| if live(o) { "p" } else { "null" };
            let bufs = layout.page_buffers(i);
            let mut r = format!(
                "L{i} src={src} hd={} kvh={} pages={} psz={} bs={} hnd=0 nat={} \
                 kattn={k_attn} vattn={v_attn} kscale={} vscale={} env={} shared=1 bufs={}",
                layout.head_dim_at(src),
                layout.num_kv_heads_at(src),
                layout.num_pages(),
                layout.page_size(),
                f.block_size(),
                i32::from(native),
                p(&slot.k_scale),
                p(&slot.v_scale),
                p(&slot.k_env_min),
                bufs.len(),
            );
            for (_, bytes) in bufs {
                let _ = write!(r, "/{bytes}");
            }
            r
        })
        .collect()
}

struct Sweep {
    out: String,
}

impl Sweep {
    fn new() -> Self {
        Self {
            out: String::from("# kv_cache oracle v1\n"),
        }
    }

    fn emit(&mut self, id: &str, body: &str) {
        let _ = writeln!(self.out, "{id}|{body}");
    }

    fn report(&mut self, id: &str, r: Result<KvCacheLayout, driver_cuda::Error>) {
        match r {
            Ok(l) => {
                let body = format!(
                    "OK|{}|{}|{}",
                    alloc_log(&l).join(&SEP.to_string()),
                    seed_log(&l).join(&SEP.to_string()),
                    readback(&l).join(&SEP.to_string())
                );
                self.emit(id, &body);
            }
            Err(e) => {
                // A throw aborts before any allocation is recorded here, and
                // the C++ likewise reports whatever the log held at the point
                // of the throw -- which is empty, because all three length
                // checks run before the first `DeviceTensor::allocate`.
                self.emit(id, &format!("FAILED|{}|", cxx_message(&e)));
            }
        }
    }
}

/// The `what()` string the C++ exception carries.
///
/// `Error::Invalid` renders as `"{call}: {reason}"`, which is exactly the
/// C++'s `"kv_cache: per_layer_head_dim size mismatch"` when `call` is the
/// prefix the C++ throws with. Rendering the whole thing rather than stripping
/// a prefix is what keeps the prefix itself under test.
fn cxx_message(e: &driver_cuda::Error) -> String {
    e.to_string()
}

#[allow(clippy::too_many_arguments)]
fn homogeneous(
    sw: &mut Sweep,
    id: &str,
    l: i32,
    p: i32,
    ps: i32,
    kvh: i32,
    hd: i32,
    f: &FormatCase,
    env: bool,
) {
    let r = KvCacheLayout::plan(l, p, ps, kvh, hd, make_format(f), env);
    sw.report(id, r);
}

#[test]
fn the_kv_cache_allocation_matches_the_cpp_byte_for_byte() {
    let mut sw = Sweep::new();

    for f in FORMATS {
        for layers in [1, 2, 8] {
            for pages in [0, 1, 64, 4096] {
                for page_size in [1, 16, 32] {
                    for kv_heads in [1, 4, 8] {
                        for head_dim in [64, 128, 576] {
                            let id = format!(
                                "homo/{}/{layers}/{pages}/{page_size}/{kv_heads}/{head_dim}",
                                f.label
                            );
                            homogeneous(
                                &mut sw, &id, layers, pages, page_size, kv_heads, head_dim, f,
                                false,
                            );
                        }
                    }
                }
            }
        }
    }

    for f in FORMATS {
        for head_dim in [1, 2, 15, 17, 31, 33, 63, 65, 127, 129, 191] {
            let id = format!("odd/{}/{head_dim}", f.label);
            homogeneous(&mut sw, &id, 2, 32, 16, 4, head_dim, f, false);
        }
    }

    for d in [
        DType::Bf16,
        DType::Fp16,
        DType::Fp8E4M3,
        DType::Int8,
        DType::Fp32,
    ] {
        let fmt = KvCacheFormat::for_storage_dtype(d);
        let id = format!("dtype/{}", d as u8);
        match KvCacheLayout::plan(2, 64, 16, 4, 128, fmt, false) {
            Ok(l) => {
                let body = format!(
                    "OK|{}|{}|{}",
                    alloc_log(&l).join(&SEP.to_string()),
                    fmt.name(),
                    readback(&l).join(&SEP.to_string())
                );
                sw.emit(&id, &body);
            }
            Err(e) => sw.emit(&id, &format!("FAILED|{}", cxx_message(&e))),
        }
    }

    for f in FORMATS {
        for (share_label, src) in SHARES {
            for variant in 0..4 {
                let (hd, kvh): (Vec<i32>, Vec<i32>) = match variant {
                    0 => (vec![], vec![]),
                    1 => (vec![64, 128, 64, 128, 64, 128], vec![]),
                    2 => (vec![], vec![1, 2, 4, 8, 4, 2]),
                    _ => (vec![576, 64, 128, 256, 128, 64], vec![1, 8, 4, 2, 4, 8]),
                };
                let per = PerLayer {
                    head_dim: hd,
                    kv_source_layer: (*src).to_vec(),
                    num_kv_heads: kvh,
                };
                let id = format!("perlayer/{}/{share_label}/v{variant}", f.label);
                let r = KvCacheLayout::plan_per_layer(6, 128, 16, 4, per, make_format(f), false);
                sw.report(&id, r);
            }
        }
    }

    {
        let f = &FORMATS[0];
        for (id, layers, hd, src, kvh) in [
            ("bad/hd-short", 4, vec![64, 64], vec![], vec![]),
            ("bad/hd-long", 4, vec![64, 64, 64, 64, 64], vec![], vec![]),
            ("bad/src-short", 4, vec![], vec![0, 0], vec![]),
            ("bad/kvh-short", 4, vec![], vec![], vec![1, 2]),
            (
                "bad/hd-ok-kvh-short",
                4,
                vec![64, 64, 64, 64],
                vec![],
                vec![1, 2],
            ),
            ("bad/hd-and-src-short", 4, vec![64, 64], vec![0, 0], vec![]),
            ("bad/src-and-kvh-short", 4, vec![], vec![0, 0], vec![1, 2]),
            ("bad/all-three-short", 4, vec![64], vec![0], vec![1]),
            ("bad/zero-layers-with-vectors", 0, vec![64], vec![], vec![]),
            ("bad/all-empty-zero-layers", 0, vec![], vec![], vec![]),
        ] {
            let per = PerLayer {
                head_dim: hd,
                kv_source_layer: src,
                num_kv_heads: kvh,
            };
            let r = KvCacheLayout::plan_per_layer(layers, 32, 16, 4, per, make_format(f), false);
            sw.report(id, r);
        }

        for (id, hd) in [
            ("scalar-hd/empty", vec![]),
            ("scalar-hd/first-is-576", vec![576, 64, 64]),
        ] {
            let per = PerLayer {
                head_dim: hd,
                ..PerLayer::default()
            };
            let r = KvCacheLayout::plan_per_layer(3, 32, 16, 4, per, make_format(f), false);
            sw.report(id, r);
        }
    }

    for f in FORMATS {
        let fmt = make_format(f);
        for page_size in [1_u32, 16, 32] {
            for kv_heads in [1_u32, 4, 8] {
                for head_dim in [64_u32, 128, 576] {
                    let v = kv_geometry::device_bytes_per_page(&fmt, page_size, kv_heads, head_dim);
                    sw.emit(
                        &format!("bytes/{}/{page_size}/{kv_heads}/{head_dim}", f.label),
                        &v.to_string(),
                    );
                }
            }
        }
        for tp in [0_i32, 1, 2, 3, 8, 16] {
            sw.emit(
                &format!("homobytes/{}/{tp}", f.label),
                &kv_geometry::page_bytes_homogeneous(6, 8, 128, tp, &fmt).to_string(),
            );
            for (share_label, src) in SHARES {
                let src_u: Vec<u32> = src.iter().map(|&v| v.unsigned_abs()).collect();
                let a = kv_geometry::page_bytes_per_layer(
                    6,
                    8,
                    128,
                    &LayerShapes {
                        head_dim: &[],
                        num_kv_heads: &[],
                        source_layer: &src_u,
                    },
                    tp,
                    &fmt,
                );
                let b = kv_geometry::page_bytes_per_layer(
                    6,
                    8,
                    128,
                    &LayerShapes {
                        head_dim: &[64, 128, 64, 128, 64, 128],
                        num_kv_heads: &[],
                        source_layer: &src_u,
                    },
                    tp,
                    &fmt,
                );
                let c = kv_geometry::page_bytes_per_layer(
                    6,
                    8,
                    128,
                    &LayerShapes {
                        head_dim: &[],
                        num_kv_heads: &[1, 2, 4, 8, 4, 2],
                        source_layer: &src_u,
                    },
                    tp,
                    &fmt,
                );
                sw.emit(
                    &format!("layerbytes/{}/{tp}/{share_label}", f.label),
                    &format!("{a}/{b}/{c}"),
                );
            }
        }
    }

    for f in FORMATS {
        for layers in [1, 3] {
            for pages in [0, 64] {
                for kv_heads in [1, 4] {
                    for head_dim in [64, 576] {
                        let id = format!("env/{}/{layers}/{pages}/{kv_heads}/{head_dim}", f.label);
                        homogeneous(&mut sw, &id, layers, pages, 16, kv_heads, head_dim, f, true);
                    }
                }
            }
        }
        for (share_label, src) in SHARES {
            let per = PerLayer {
                head_dim: vec![576, 64, 128, 256, 128, 64],
                kv_source_layer: (*src).to_vec(),
                num_kv_heads: vec![1, 8, 4, 2, 4, 8],
            };
            sw.report(
                &format!("envshare/{}/{share_label}", f.label),
                KvCacheLayout::plan_per_layer(6, 64, 16, 4, per, make_format(f), true),
            );
            let per = PerLayer {
                kv_source_layer: (*src).to_vec(),
                ..PerLayer::default()
            };
            sw.report(
                &format!("envshare/{}/{share_label}/scalar", f.label),
                KvCacheLayout::plan_per_layer(6, 64, 16, 4, per, make_format(f), true),
            );
        }
    }

    // SAFETY-adjacent: this test process is the only reader of the variable,
    // and the sweep is single-threaded, so setting it here cannot race another
    // test. The switch is read on every call in both languages.
    for v in [
        "1", "true", "on", "0", "yes", "TRUE", "On", "", "false", "2",
    ] {
        unsafe { std::env::set_var("PIE_CUDA_KV_ENVELOPES", v) };
        let label = if v.is_empty() { "<empty>" } else { v };
        sw.emit(
            &format!("envswitch/{label}"),
            &i32::from(kv_cache::envelopes_requested()).to_string(),
        );
    }
    unsafe { std::env::remove_var("PIE_CUDA_KV_ENVELOPES") };
    sw.emit(
        "envswitch/<unset>",
        &i32::from(kv_cache::envelopes_requested()).to_string(),
    );

    // `Error::Invalid` joins with ": " but the C++ message uses "; " -- the
    // separator belongs to the sentence, not to the error type, so it is
    // restored here rather than by bending the Display impl.
    sw.emit(
        "enable-late",
        &format!(
            "THREW|{}",
            kv_cache::enable_envelopes_late_error()
                .to_string()
                .replacen(": ", "; ", 1)
        ),
    );

    if let Ok(path) = std::env::var("KV_RUST_OUT") {
        std::fs::write(path, &sw.out).expect("write transcript");
    }

    let rows = sw.out.lines().count();
    assert_eq!(rows, GOLDEN_ROWS, "row count drifted from the oracle's");
    assert_eq!(
        fnv1a64(sw.out.as_bytes()),
        GOLDEN_FNV1A64,
        "transcript diverged from the C++; set KV_RUST_OUT and diff against \
         tests/oracle/kv_cache/run.sh's KV_ORACLE_OUT"
    );
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}
