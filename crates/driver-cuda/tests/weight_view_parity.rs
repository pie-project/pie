//! ABI and behavioural parity with the C++ `WeightView`.
//!
//! The oracle in `tests/oracle/weight_view/` compiles the real
//! `kernels-cuda/csrc/src/weight_view.hpp`, `quant_meta.hpp` and `tensor.cpp`
//! and prints two things: the struct's ABI as `offsetof` sees it, and every
//! field of every view its five factories produce. This test reproduces both.
//!
//! Run `tests/oracle/weight_view/run.sh` to regenerate [`GOLDEN_FNV1A64`].
//!
//! # Why the ABI half exists
//!
//! Everything else ported so far lives on the Rust side of the crate boundary
//! and only has to *behave* like its original. `WeightView` is passed to
//! kernel launchers **by value**, so it has to be *laid out* like it too. A
//! mirror wrong by four bytes has the GEMM read `scale_data` out of the
//! padding after `dtype` and dereference it — there is no symptom short of a
//! fault at an address that has nothing to do with this file.
//!
//! Rust's `repr(C)` follows the same layout rules as C++'s standard layout, so
//! matching field order and types is *expected* to produce a matching struct.
//! This test is what turns that expectation into something that has been
//! checked, on this compiler, for this target.

use std::ffi::c_void;
use std::mem::offset_of;
use std::fmt::Write as _;

use driver_cuda::dtype::DType;
use driver_cuda::gpu::weights::weight_view::{
    QuantKind, QuantMeta, TensorRef, WeightView, make_weight_view,
};

/// FNV-1a 64 of the C++ oracle's transcript.
const GOLDEN_FNV1A64: u64 = 0x3faf_717c_f13b_df10;

/// Rows the transcript must contain.
const GOLDEN_ROWS: usize = 33;

const SEP: char = '\u{1f}';

/// The oracle's slab base. Pointers are reported as `+N` from it, so the
/// golden is about the factories rather than about an allocator.
const BASE: usize = 0x1_0000;

fn slab(offset: usize) -> *const c_void {
    (BASE + offset) as *const c_void
}

fn at(p: *const c_void) -> String {
    if p.is_null() {
        "null".to_owned()
    } else {
        format!("+{}", p as usize - BASE)
    }
}

/// The oracle's `tensor(offset, dtype, shape)`, which goes through
/// `DeviceTensor::view` and therefore derives `numel` and `nbytes` from the
/// shape. Reproduced here so the transcript's sizes are derived the same way
/// rather than written down twice.
fn tensor(offset: usize, dtype: DType, shape: &[i64]) -> TensorRef {
    let numel = shape.iter().product::<i64>() as usize;
    TensorRef::new(slab(offset), dtype, numel * dtype_bytes(dtype), numel)
}

/// `pie_cuda_driver::dtype_bytes`, for the storage types this test uses.
const fn dtype_bytes(d: DType) -> usize {
    match d {
        DType::Fp32 | DType::Int32 => 4,
        DType::Int64 => 8,
        DType::Bf16 | DType::Fp16 => 2,
        _ => 1,
    }
}

/// Script 1 — the ABI, as `offsetof` reports it.
fn script_abi(out: &mut String) {
    writeln!(out, "abi{SEP}sizeof{SEP}{}", size_of::<WeightView>()).unwrap();
    writeln!(out, "abi{SEP}alignof{SEP}{}", align_of::<WeightView>()).unwrap();
    // C++ reports `is_standard_layout` and `is_trivially_copyable`; a
    // `#[repr(C)]` struct of `Copy` fields is both by construction, and the
    // `Copy` bound below is what makes the second one true rather than
    // asserted.
    fn assert_copy<T: Copy>() {}
    assert_copy::<WeightView>();
    writeln!(out, "abi{SEP}standard_layout{SEP}1").unwrap();
    writeln!(out, "abi{SEP}trivially_copyable{SEP}1").unwrap();

    let fields: [(&str, usize, usize); 10] = [
        ("data", offset_of!(WeightView, data), size_of::<*const c_void>()),
        ("dtype", offset_of!(WeightView, dtype), size_of::<DType>()),
        ("nbytes", offset_of!(WeightView, nbytes), size_of::<usize>()),
        (
            "scale_data",
            offset_of!(WeightView, scale_data),
            size_of::<*const c_void>(),
        ),
        (
            "scale_dtype",
            offset_of!(WeightView, scale_dtype),
            size_of::<DType>(),
        ),
        (
            "scale_numel",
            offset_of!(WeightView, scale_numel),
            size_of::<usize>(),
        ),
        (
            "quant_kind",
            offset_of!(WeightView, quant_kind),
            size_of::<QuantKind>(),
        ),
        (
            "zero_point_data",
            offset_of!(WeightView, zero_point_data),
            size_of::<*const c_void>(),
        ),
        ("group_size", offset_of!(WeightView, group_size), size_of::<i32>()),
        (
            "channel_axis",
            offset_of!(WeightView, channel_axis),
            size_of::<i32>(),
        ),
    ];
    for (name, offset, size) in fields {
        writeln!(out, "abi{SEP}offsetof{SEP}{name}{SEP}{offset}{SEP}{size}").unwrap();
    }

    // The C++ prints 1 for a signed underlying type. `DType` is
    // `enum class : std::uint8_t`, `QuantMeta::Kind` has no fixed underlying
    // type and gets `int`.
    writeln!(out, "abi{SEP}enum{SEP}DType{SEP}{}{SEP}0", size_of::<DType>()).unwrap();
    writeln!(
        out,
        "abi{SEP}enum{SEP}QuantKind{SEP}{}{SEP}1",
        size_of::<QuantKind>()
    )
    .unwrap();
    for (k, kind) in [
        QuantKind::PerTensor,
        QuantKind::PerChannel,
        QuantKind::PerGroup,
    ]
    .into_iter()
    .enumerate()
    {
        writeln!(out, "abi{SEP}quant_kind_value{SEP}{k}{SEP}{}", kind as i32).unwrap();
    }
}

fn dump(out: &mut String, label: &str, v: &WeightView) {
    writeln!(
        out,
        "view{SEP}{label}{SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}",
        at(v.data),
        v.dtype as i32,
        v.nbytes,
        at(v.scale_data),
        v.scale_dtype as i32,
        v.scale_numel,
        v.quant_kind as i32,
        at(v.zero_point_data),
        v.group_size,
        v.channel_axis,
    )
    .unwrap();
}

/// Script 2 — every field of every factory.
fn script_factories(out: &mut String) {
    dump(out, "default", &WeightView::default());

    let bf16 = tensor(0, DType::Bf16, &[128, 64]);
    dump(out, "implicit_bf16", &WeightView::plain(bf16));

    let fp8 = tensor(4096, DType::Fp8E4M3, &[128, 64]);
    dump(out, "implicit_fp8", &WeightView::plain(fp8));

    dump(out, "raw_bf16", &WeightView::raw(slab(8192), DType::Bf16));
    dump(
        out,
        "raw_mxfp4",
        &WeightView::raw(slab(8192), DType::Mxfp4Packed),
    );
    dump(
        out,
        "raw_null",
        &WeightView::raw(std::ptr::null(), DType::Fp32),
    );

    let scale_f32 = tensor(12288, DType::Fp32, &[128]);
    let zp = tensor(16384, DType::Int8, &[128]);

    let per_tensor = QuantMeta {
        kind: QuantKind::PerTensor,
        scale: Some(scale_f32),
        ..QuantMeta::default()
    };
    dump(
        out,
        "quantized_per_tensor",
        &WeightView::quantized(fp8, &per_tensor),
    );

    let per_channel = QuantMeta {
        kind: QuantKind::PerChannel,
        scale: Some(scale_f32),
        channel_axis: 1,
        ..QuantMeta::default()
    };
    dump(
        out,
        "quantized_per_channel",
        &WeightView::quantized(fp8, &per_channel),
    );

    let per_group = QuantMeta {
        kind: QuantKind::PerGroup,
        scale: Some(scale_f32),
        zero_point: Some(zp),
        group_size: 128,
        channel_axis: 0,
    };
    dump(
        out,
        "quantized_per_group",
        &WeightView::quantized(fp8, &per_group),
    );

    let no_scale = QuantMeta {
        kind: QuantKind::PerChannel,
        ..QuantMeta::default()
    };
    dump(
        out,
        "quantized_no_scale",
        &WeightView::quantized(fp8, &no_scale),
    );

    let mx_w = tensor(20480, DType::Uint8, &[128, 32]);
    let mx_s = tensor(24576, DType::Uint8, &[128, 2]);
    dump(out, "mxfp4_marlin", &WeightView::mxfp4_marlin(mx_w, mx_s));
}

/// Script 3 — `make_weight_view`, the dispatch the generated bodies call.
fn script_make_weight_view(out: &mut String) {
    let w = tensor(0, DType::Bf16, &[256, 64]);
    let scale = tensor(12288, DType::Fp32, &[256]);

    dump(out, "make_unquantized", &make_weight_view(w, None));

    let meta = QuantMeta {
        kind: QuantKind::PerChannel,
        scale: Some(scale),
        channel_axis: 0,
        ..QuantMeta::default()
    };
    dump(out, "make_quantized", &make_weight_view(w, Some(&meta)));

    // A default QuantMeta cannot show which discriminator `make_weight_view`
    // uses -- PerTensor with a zero group_size produces the same bytes down
    // either branch. This one carries a kind and a group_size the quantized
    // branch copies through and the unquantized branch would zero, so the
    // transcript can tell `meta.has_value()` from `meta->scale != nullptr`.
    let empty = QuantMeta {
        kind: QuantKind::PerGroup,
        group_size: 64,
        channel_axis: 1,
        ..QuantMeta::default()
    };
    dump(
        out,
        "make_engaged_but_empty",
        &make_weight_view(w, Some(&empty)),
    );
}

fn transcript() -> String {
    let mut out = String::new();
    script_abi(&mut out);
    script_factories(&mut out);
    script_make_weight_view(&mut out);
    out
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
fn the_rust_weight_view_reproduces_the_cpp_transcript() {
    let t = transcript();
    assert_eq!(
        t.lines().count(),
        GOLDEN_ROWS,
        "transcript row count drifted from the C++ oracle"
    );
    assert_eq!(
        fnv1a64(t.as_bytes()),
        GOLDEN_FNV1A64,
        "transcript differs from the C++ oracle; \
         run tests/oracle/weight_view/run.sh with WV_ORACLE_OUT set to diff them"
    );
}

/// The ABI, asserted directly as well as hashed.
///
/// A hash says "something moved"; this says which byte. The numbers are the
/// C++'s `offsetof` output, transcribed from the oracle transcript.
#[test]
fn every_field_lands_where_the_cpp_launchers_expect_it() {
    assert_eq!(size_of::<WeightView>(), 72, "struct size");
    assert_eq!(align_of::<WeightView>(), 8, "struct alignment");
    for (name, actual, expected) in [
        ("data", offset_of!(WeightView, data), 0),
        ("dtype", offset_of!(WeightView, dtype), 8),
        ("nbytes", offset_of!(WeightView, nbytes), 16),
        ("scale_data", offset_of!(WeightView, scale_data), 24),
        ("scale_dtype", offset_of!(WeightView, scale_dtype), 32),
        ("scale_numel", offset_of!(WeightView, scale_numel), 40),
        ("quant_kind", offset_of!(WeightView, quant_kind), 48),
        ("zero_point_data", offset_of!(WeightView, zero_point_data), 56),
        ("group_size", offset_of!(WeightView, group_size), 64),
        ("channel_axis", offset_of!(WeightView, channel_axis), 68),
    ] {
        assert_eq!(actual, expected, "{name} is at the wrong offset");
    }
}

/// The dispatcher's discriminator is `scale_data`, and only `scale_data`.
///
/// Stated on its own because three fields look like they might be it —
/// `dtype`, `quant_kind`, and the presence of a `QuantMeta` — and a port that
/// picked any of the other three would still pass a layout check.
#[test]
fn only_a_null_scale_pointer_selects_the_bf16_path() {
    let w = tensor(0, DType::Fp8E4M3, &[64, 64]);
    let s = tensor(4096, DType::Fp32, &[64]);

    assert!(WeightView::plain(w).is_bf16_path());
    assert!(WeightView::raw(slab(0), DType::Mxfp4Packed).is_bf16_path());

    let with_kind_only = QuantMeta {
        kind: QuantKind::PerGroup,
        group_size: 128,
        ..QuantMeta::default()
    };
    assert!(
        WeightView::quantized(w, &with_kind_only).is_bf16_path(),
        "a quant kind without scales does not select the quantized path"
    );
    assert!(
        WeightView::quantized(w, &with_kind_only).would_silently_degrade(),
        "and that is exactly the condition worth being able to name"
    );

    let real = QuantMeta {
        kind: QuantKind::PerChannel,
        scale: Some(s),
        ..QuantMeta::default()
    };
    assert!(!WeightView::quantized(w, &real).is_bf16_path());
    assert!(!WeightView::mxfp4_marlin(w, s).is_bf16_path());
}
