//! The three operand kinds added for `quant`, and the proof that they are
//! CHECKS rather than names.
//!
//! # What is on trial
//!
//! `Ty::Bf16s`, `Ty::F16s` and `Ty::I8sMut` were added so seven `quant` rows
//! could state operands the vocabulary had no word for: a `const bf16*` that
//! stays `bf16` while the row's `elem` moves, a `const f16*` that stays
//! `f16` for the same reason, and a signed byte store beside an unsigned one.
//! Before them those rows said `Buf`/`U8sMut`, and
//! [`abi::emit_device_typecheck`] spelled `Buf` from the row's `elem` — so
//! `quant::bf16_to_fp16`, whose `elem` is `device::f16`, emitted
//! `const f16*` for a parameter the `__global__` declares `const bf16*`.
//!
//! A new `Ty` that merely RENDERS is worth nothing here. The property the
//! seven rows are being brought inside is the one
//! `emit_device_typecheck` states in its own header — *"a function-pointer
//! initialisation admits NO parameter conversions"* — and a variant that
//! spells something both `const bf16*` and `const f16*` would convert to has
//! removed that property while appearing to extend it. So this file does not
//! assert that the new kinds work; it compiles them, both ways round.
//!
//! # The shape of the proof
//!
//! For each new kind: the row AS STATED is emitted through
//! [`abi::emit_device_typecheck`] and handed to `nvcc`, and the SAME row with
//! exactly one operand's `Ty` swapped for the wrong one is emitted and handed
//! to the same `nvcc` with the same flags. The first must compile and the
//! second must not. A mutant is pinned to its original by
//! [`differs_in_exactly_one_operand`], so a row that is later edited cannot
//! leave a mutant silently testing something else — or nothing.
//!
//! # `-Xcompiler=-iquote` and not `-I`, and it matters MOST here
//!
//! `csrc/shim` holds fourteen headers that shadow real toolkit ones,
//! `cuda_fp16.h` among them, and that one opens with
//! `using __half = device::f16`. Under
//! `-I` a `.cuh` reaching `<cuda_fp16.h>` finds the SHIM even under nvcc, and
//! `new-horizon.md` §21.10 records the measurement: the same source under the
//! two spellings both compiled cleanly and exported DIFFERENT mangled symbols
//! — `bf16_to_narrow<__half>` against `bf16_to_narrow<device::f16>`, 31%
//! apart in object size. This file's subject is a `bf16`/`f16` confusion, so
//! compiling it under a flag that manufactures one would be a negative
//! control that proves the opposite of what it claims. `nvcc -iquote` is
//! rejected outright — which is why a probe reaches for `-I` and gets a quiet
//! wrong answer — so the quote path is handed to the host compiler:
//! `-Xcompiler=-iquote,<dir>`.
//!
//! Two directories since `csrc/` was cut by role: `csrc/src` for the kernel
//! headers and `csrc/shim` for the impersonating ones. Both are passed, and
//! dropping the second would not fail — it would resolve `<cuda_fp16.h>` and
//! the quoted `"cuda_fp16.h"` in `pie_fp8.cuh` against the real toolkit and
//! quietly re-manufacture the confusion this file exists to catch.
//!
//! # What runs where
//!
//! The rendering assertions are pure string work over the tables and run on
//! any machine. The compiles need `nvcc` and say so, once, when it is absent.

use std::path::PathBuf;
use std::process::Command;

use kernels::{Source, Ty, kernel, operands};
use kernels_cuda_new::device::DeviceKernel;
use kernels_cuda_new::{KernelSig, LaunchRule, abi, unit};

// ---------------------------------------------------------------------------
// The renderings, on every machine
// ---------------------------------------------------------------------------

/// Every emitter that switches on [`Ty`] has a spelling for each new kind.
///
/// Five places, and the count is the point: the C shim's parameter, the Rust
/// binding's argument, the dispatch's cast, the device typecheck's
/// function-pointer parameter, and the `ArgValue` variant the JIT arm binds
/// through. A kind that renders in four of them is a kind that fails at the
/// fifth — at link, or at launch, whichever the missing one feeds.
///
/// The C++ and Rust spellings are pinned as literals rather than compared to
/// each other, because the whole hazard is that two DIFFERENT C++ types have
/// the same width: `Ty::Bf16s` and `Ty::F16s` are both `*const u16` in Rust
/// and must not be the same string in C++.
#[test]
fn every_new_kind_renders_in_every_emitter() {
    assert_eq!(Ty::Bf16s.cpp(), "const ::pie_cuda_driver::kernels::device::bf16*");
    assert_eq!(Ty::F16s.cpp(), "const ::pie_cuda_driver::kernels::device::f16*");
    assert_eq!(Ty::I8sMut.cpp(), "::std::int8_t*");

    assert_ne!(
        Ty::Bf16s.cpp(),
        Ty::F16s.cpp(),
        "the two sixteen-bit formats collapsed to one C++ spelling, which is \
         exactly the state `pie_device.cuh` made them structs to prevent"
    );
    assert_ne!(Ty::I8sMut.cpp(), Ty::U8sMut.cpp(), "a signed store spelled unsigned");

    assert_eq!(Ty::Bf16s.rust(), "*const u16");
    assert_eq!(Ty::F16s.rust(), "*const u16");
    assert_eq!(Ty::I8sMut.rust(), "*mut i8");

    // None of the three names a `#[repr(C)]` mirror, so a row using them stays
    // in the PORTABLE binding subset the loader generates against.
    for ty in [Ty::Bf16s, Ty::F16s, Ty::I8sMut] {
        assert!(!ty.needs_mirror(), "{ty:?} claims a mirror it has no struct for");
    }

    // The C shim and the Rust binding, from a table that uses all three.
    let shim = abi::emit_c_shim(&[PROBE_TABLE], &["quant/quant_bf16_to_fp8.cuh"], &[])
        .expect("the probe table has no colliding entry point");
    assert!(
        shim.contains("const ::pie_cuda_driver::kernels::device::bf16* w"),
        "emit_c_shim did not spell Bf16s:\n{shim}"
    );
    assert!(
        shim.contains("const ::pie_cuda_driver::kernels::device::f16* src"),
        "emit_c_shim did not spell F16s:\n{shim}"
    );
    assert!(shim.contains("::std::int8_t* out"), "emit_c_shim did not spell I8sMut:\n{shim}");

    let bindings = abi::emit_rust_bindings(&[PROBE_TABLE]);
    assert!(bindings.contains("w: *const u16,"), "emit_rust_bindings dropped Bf16s:\n{bindings}");
    assert!(bindings.contains("src: *const u16,"), "emit_rust_bindings dropped F16s:\n{bindings}");
    assert!(bindings.contains("out: *mut i8,"), "emit_rust_bindings dropped I8sMut:\n{bindings}");

    // The same rows must survive `emit_rust_bindings_portable`, which is the
    // subset the model loader declares against.
    let portable = abi::emit_rust_bindings_portable(&[PROBE_TABLE]);
    assert!(portable.contains("w: *const u16,"), "the portable subset dropped Bf16s");
    assert!(portable.contains("out: *mut i8,"), "the portable subset dropped I8sMut");

    // The dispatch's casts. Every operand of the probe row is sourced, so the
    // row gets a branch and each pointer arrives with its cast applied.
    let dispatch = abi::emit_rust_dispatch(&[PROBE_TABLE], &[]);
    assert!(
        dispatch.contains("(b.args[0].ptr).cast_const().cast::<u16>()"),
        "emit_rust_dispatch did not cast Bf16s:\n{dispatch}"
    );
    assert!(
        dispatch.contains("(b.args[n_in + 0].ptr).cast::<i8>()"),
        "emit_rust_dispatch did not cast I8sMut:\n{dispatch}"
    );

    // And the JIT arm, which is the OTHER branch of the same emitter: a row
    // NVRTC compiles has no shim to call, so its arguments cross as
    // `ArgValue`s rather than as a cast argument list. Both halves have to
    // know the kind — `arg_value_variant` returning `None` would silently
    // drop the row's arm, and `runtime::args` refusing the kind would fail
    // the fire — which is why the same probe row is emitted twice.
    //
    // THE ADDRESS TYPE IS PART OF THE CLAIM. `ArgValue::Ptr` holds a
    // `*mut c_void`, and every routed row until `layout::gather_bf16_rows`
    // bound its pointers from `b.args[..].ptr`, which already is one — so
    // the emitter got away with passing the expression through untouched and
    // the first row reading a driver context field (`ctx.sampling_indices:
    // *const i32`) produced `types differ in mutability` IN A GENERATED
    // FILE. The cast is asserted here rather than left to whichever row
    // happens to be routed, because that is exactly what went wrong: the
    // property held by accident for three months and nothing said so.
    let jit = abi::emit_rust_dispatch(&[PROBE_TABLE], &[&PROBE_JIT]);
    for (kind, expr) in [
        ("Bf16s", "crate::bind::device::ArgValue::Ptr((b.args[0].ptr) as *mut ::core::ffi::c_void)"),
        ("F16s", "crate::bind::device::ArgValue::Ptr((b.args[1].ptr) as *mut ::core::ffi::c_void)"),
        (
            "I8sMut",
            "crate::bind::device::ArgValue::Ptr((b.args[n_in + 0].ptr) as *mut ::core::ffi::c_void)",
        ),
    ] {
        assert!(jit.contains(expr), "the JIT arm did not bind {kind} as a pointer:\n{jit}");
    }
    // A `Ptr` THAT IS NOT AN ADDRESS is the failure the three pins above
    // would miss if a fourth kind arrived: they name three operands, and a
    // new one binds without them. This says the property about ALL of them.
    assert!(
        !jit.lines().any(|line| line.contains("ArgValue::Ptr(")
            && !line.contains("as *mut ::core::ffi::c_void")),
        "a `Ptr` operand reached the JIT arm without the address cast, so its \
         Rust type is whatever the source expression happened to be:\n{jit}"
    );

    // The device typecheck's function-pointer parameters, from a REAL row.
    // `elem` here is `device::f16` and the emitted `bf16` therefore cannot
    // have come from it -- which is the whole difference `Ty::Bf16s` makes.
    let row = device_row("quant::bf16_to_fp16");
    let tu = abi::emit_device_typecheck(&[DeviceKernel {
        sig: row.sig,
        template_path: row.template_path,
        elem: row.elem,
    }])
    .expect("the real row emits");
    assert_eq!(row.elem, "device::f16", "the row's element type moved; this test's premise did");
    assert!(
        tu.contains("const ::pie_cuda_driver::kernels::device::bf16*"),
        "emit_device_typecheck did not spell Bf16s:\n{tu}"
    );
}

/// The seven retyped rows still marshal, and marshal as POINTERS.
///
/// `runtime::args` refuses a kind it has no `ArgValue` for, so a new `Ty` that
/// the binder does not know is a row that emits, compiles, and then fails at
/// the fire. This is the cheap half of that check — the kinds are pointers —
/// and `emit.rs`'s `every_kind_the_binder_marshals_crosses_the_same_way` is
/// the half that ties the three separate copies of the list together.
#[test]
fn the_seven_rows_state_only_kinds_the_binder_takes() {
    for symbol in SEVEN {
        let row = device_row(symbol);
        for o in row.sig.operands {
            assert!(
                abi::emit_device_typecheck(&[DeviceKernel {
                    sig: row.sig,
                    template_path: row.template_path,
                    elem: row.elem,
                }])
                .is_ok(),
                "`{symbol}` does not emit a typecheck"
            );
            assert!(
                !matches!(o.ty, Ty::Stream | Ty::CublasHandle),
                "`{symbol}`'s `{}` is not a kernel argument",
                o.name
            );
        }
    }
}

/// Each of the seven states the kind its `__global__` declares.
///
/// The list is the CLAIM, written out so that a row quietly reverting to
/// `Buf` is a failure here rather than a `const f16*` in a generated file
/// nobody reads.
#[test]
fn the_seven_rows_are_typed() {
    let expected: &[(&str, &str, Ty)] = &[
        ("quant::quant_bf16_to_fp8_e4m3", "w", Ty::Bf16s),
        // `w`, and it was `w_bf16` until `quant` crossed into fn-world. The
        // two `quantize_*_per_channel` symbols are two instantiations of ONE
        // `fn quant_per_channel`, so they share ONE parameter list and cannot
        // spell the same parameter two ways. The CLAIM this line makes is
        // about the TYPE; the name is how the claim is addressed.
        ("quant::quantize_bf16_to_fp8_e4m3_per_channel", "w", Ty::Bf16s),
        ("quant::quantize_bf16_to_int8_per_channel", "w", Ty::Bf16s),
        ("quant::quantize_bf16_to_int8_per_channel", "out", Ty::I8sMut),
        ("quant::cast_bf16_to_fp8_e4m3_per_channel", "w", Ty::Bf16s),
        ("quant::cast_bf16_to_int8_per_channel", "w", Ty::Bf16s),
        ("quant::cast_bf16_to_int8_per_channel", "out", Ty::I8sMut),
        ("quant::cast_f16_to_bf16", "src", Ty::F16s),
        ("quant::bf16_to_fp16", "in_bf16", Ty::Bf16s),
    ];
    for (symbol, operand, ty) in expected {
        let row = device_row(symbol);
        let o = row
            .sig
            .operands
            .iter()
            .find(|o| o.name == *operand)
            .unwrap_or_else(|| panic!("`{symbol}` has no operand `{operand}`"));
        assert_eq!(o.ty, *ty, "`{symbol}`'s `{operand}` is stated {:?}", o.ty);
    }
}

// ---------------------------------------------------------------------------
// The negative controls, where nvcc is
// ---------------------------------------------------------------------------

/// `quant::bf16_to_fp16` compiles as stated and does NOT compile mis-typed.
///
/// This is the row the whole change exists for. Its `__global__` is
/// `bf16_to_narrow<T>(const bf16* in, T* out, long long n)` and the row's `T`
/// is `device::f16`, so `in_bf16: Buf` — what it said before — emitted
/// `const f16*`. Two mutants, because there are two ways to be wrong: `Buf`
/// (the historical spelling, which reads the format off `elem`) and `F16s`
/// (the new vocabulary used incorrectly). Both must be refused.
#[test]
fn bf16_where_the_kernel_takes_bf16() {
    let Some(nvcc) = nvcc() else { return skip("nvcc is not on PATH or in /usr/local/cuda") };
    let row = device_row("quant::bf16_to_fp16");

    accepts(&nvcc, "bf16_to_fp16.stated", row.sig, row.template_path, row.elem);

    differs_in_exactly_one_operand(row.sig, &BF16_TO_FP16_AS_F16, "in_bf16");
    let bad = refuses(&nvcc, "bf16_to_fp16.f16s", &BF16_TO_FP16_AS_F16, row.template_path, row.elem);
    assert!(
        bad.contains("bf16_to_narrow"),
        "the diagnostic does not name the template it refused:\n{bad}"
    );

    differs_in_exactly_one_operand(row.sig, &BF16_TO_FP16_AS_BUF, "in_bf16");
    refuses(&nvcc, "bf16_to_fp16.buf", &BF16_TO_FP16_AS_BUF, row.template_path, row.elem);
}

/// `quant::cast_f16_to_bf16` compiles as stated and does NOT compile with
/// `src` spelled `Bf16s`.
///
/// The mirror image of the row above: `cast_f16_to<T>` fixes the SOURCE at
/// `const f16*` and templates the destination, and this row's `elem` is
/// `device::bf16`. So the two mutants are the same two mistakes with the
/// formats exchanged, and `Buf` is again the historical spelling.
#[test]
fn f16_where_the_kernel_takes_f16() {
    let Some(nvcc) = nvcc() else { return skip("nvcc is not on PATH or in /usr/local/cuda") };
    let row = device_row("quant::cast_f16_to_bf16");

    accepts(&nvcc, "cast_f16_to_bf16.stated", row.sig, row.template_path, row.elem);

    differs_in_exactly_one_operand(row.sig, &CAST_F16_TO_BF16_AS_BF16, "src");
    let bad =
        refuses(&nvcc, "cast_f16_to_bf16.bf16s", &CAST_F16_TO_BF16_AS_BF16, row.template_path, row.elem);
    assert!(
        bad.contains("cast_f16_to"),
        "the diagnostic does not name the template it refused:\n{bad}"
    );

    differs_in_exactly_one_operand(row.sig, &CAST_F16_TO_BF16_AS_BUF, "src");
    refuses(&nvcc, "cast_f16_to_bf16.buf", &CAST_F16_TO_BF16_AS_BUF, row.template_path, row.elem);
}

/// `quant::cast_bf16_to_int8_per_channel` compiles as stated and does NOT
/// compile with `out` spelled `U8sMut`.
///
/// The int8 half of the change, and the reason it is not decoration: `u8` and
/// `i8` are the same WIDTH, so nothing about a launch would have gone wrong
/// and nothing about a fire would have reported it. The kernel writes
/// `int8_sym::store`, whose destination is `i8*`, and a row that said
/// unsigned was a row outside the check.
#[test]
fn int8_where_the_kernel_stores_int8() {
    let Some(nvcc) = nvcc() else { return skip("nvcc is not on PATH or in /usr/local/cuda") };
    let row = device_row("quant::cast_bf16_to_int8_per_channel");

    accepts(&nvcc, "cast_bf16_to_int8.stated", row.sig, row.template_path, row.elem);

    differs_in_exactly_one_operand(row.sig, &CAST_BF16_TO_INT8_AS_U8, "out");
    let bad =
        refuses(&nvcc, "cast_bf16_to_int8.u8", &CAST_BF16_TO_INT8_AS_U8, row.template_path, row.elem);
    assert!(
        bad.contains("cast_per_channel"),
        "the diagnostic does not name the template it refused:\n{bad}"
    );
}

/// Every `quant` row that uses one of the three new kinds, in ONE translation
/// unit.
///
/// The per-row tests above prove the sharpness; this proves the COVERAGE —
/// all seven are now inside `emit_device_typecheck`'s net at once, which is
/// the state the change was made to reach, and a single compile is what says
/// so about the set rather than about a sample of it.
#[test]
fn all_seven_typecheck_together() {
    let Some(nvcc) = nvcc() else { return skip("nvcc is not on PATH or in /usr/local/cuda") };
    let rows: Vec<DeviceKernel> = SEVEN
        .iter()
        .map(|s| {
            let r = device_row(s);
            DeviceKernel { sig: r.sig, template_path: r.template_path, elem: r.elem }
        })
        .collect();
    let tu = abi::emit_device_typecheck(&rows).expect("the seven rows emit");
    assert_eq!(
        tu.matches("const ::pie_cuda_driver::kernels::device::bf16*").count(),
        6,
        "six `Bf16s` operands across the seven rows:\n{tu}"
    );
    assert_eq!(tu.matches("::std::int8_t*").count(), 2, "two `I8sMut` operands:\n{tu}");
    assert_eq!(
        tu.matches("const ::pie_cuda_driver::kernels::device::f16*").count(),
        1,
        "one `F16s` operand:\n{tu}"
    );
    if let Err(e) = compile(&nvcc, "all_seven", &tu) {
        panic!("the seven rows do not typecheck together:\n{e}\n\n{tu}");
    }
}

// ---------------------------------------------------------------------------
// The mutants
// ---------------------------------------------------------------------------

// Every mutant below states `Source::Unbound` and `LaunchRule::Unstated`
// because `quant` is fn-world now and that is what a `unit!` row IS: the
// operand list is the host `fn`'s parameter list, and the rule the row used
// to carry is a `Launch` the `fn` builds. `differs_in_exactly_one_operand`
// requires name, source and nullability to match the real row exactly, so
// these three fields are not decoration -- they are what makes a mutant a
// control rather than a second, drifting copy. The one thing that still
// differs, and the only thing this file is about, is one operand's `ty`.

/// `quant::bf16_to_fp16` with `in_bf16` spelled `F16s` — the new vocabulary,
/// used wrong.
static BF16_TO_FP16_AS_F16: KernelSig = kernel!(bf16_to_fp16 "quant::bf16_to_fp16",
    file = Some("quant/dequant_wna16.cuh"),
    launch = LaunchRule::Unstated,
    operands = operands![
        in_bf16: F16s <- Source::Unbound,
        out: BufMut <- Source::Unbound,
        n: I64 <- Source::Unbound,
    ]);

/// `quant::bf16_to_fp16` as it read BEFORE this change: `Buf`, which
/// `device_cpp_ty` spells from the row's own `elem` — `device::f16`.
static BF16_TO_FP16_AS_BUF: KernelSig = kernel!(bf16_to_fp16 "quant::bf16_to_fp16",
    file = Some("quant/dequant_wna16.cuh"),
    launch = LaunchRule::Unstated,
    operands = operands![
        in_bf16: Buf <- Source::Unbound,
        out: BufMut <- Source::Unbound,
        n: I64 <- Source::Unbound,
    ]);

/// `quant::cast_f16_to_bf16` with `src` spelled `Bf16s`.
static CAST_F16_TO_BF16_AS_BF16: KernelSig = kernel!(cast_f16_to_bf16 "quant::cast_f16_to_bf16",
    file = Some("quant/dtype_cast.cuh"),
    launch = LaunchRule::Unstated,
    operands = operands![
        src: Bf16s <- Source::Unbound,
        dst: BufMut <- Source::Unbound,
        n: Usize <- Source::Unbound,
    ]);

/// `quant::cast_f16_to_bf16` as it read before: `Buf`, spelled from `elem`,
/// which here is `device::bf16`.
static CAST_F16_TO_BF16_AS_BUF: KernelSig = kernel!(cast_f16_to_bf16 "quant::cast_f16_to_bf16",
    file = Some("quant/dtype_cast.cuh"),
    launch = LaunchRule::Unstated,
    operands = operands![
        src: Buf <- Source::Unbound,
        dst: BufMut <- Source::Unbound,
        n: Usize <- Source::Unbound,
    ]);

/// `quant::cast_bf16_to_int8_per_channel` as it read before: an UNSIGNED
/// store for `int8_sym::store`'s `i8*`.
static CAST_BF16_TO_INT8_AS_U8: KernelSig =
    kernel!(cast_bf16_to_int8_per_channel "quant::cast_bf16_to_int8_per_channel",
        file = Some("quant/quant_bf16_to_fp8.cuh"),
        launch = LaunchRule::Unstated,
        operands = operands![
            w: Bf16s <- Source::Unbound,
            out: U8sMut <- Source::Unbound,
            scale_inv: F32s <- Source::Unbound,
            cols: I32 <- Source::Unbound,
        ]);

/// A synthetic table that exercises all three kinds through the emitters that
/// only ever see AHEAD-OF-TIME rows.
///
/// Synthetic on purpose. The real `table::KERNELS` twins of these seven rows
/// describe the HOST LAUNCHER, whose parameters are `void*` because that is
/// what crosses a C ABI — so they say `Buf` correctly and would never
/// exercise a device kind. This table says what an AOT row using the new
/// kinds WOULD emit, which is the question `emit_c_shim` and
/// `emit_rust_bindings` are being asked.
static PROBE_SIGS: [KernelSig; 1] = [kernel!(ty_probe "probe::ty_probe",
    file = Some("quant/quant_bf16_to_fp8.cuh"),
    launch = LaunchRule::Elementwise,
    operands = operands![
        w: Bf16s <- Source::In(0),
        src: F16s <- Source::In(1),
        out: I8sMut <- Source::Out(0),
        cols: I32 <- Source::InWidth(0),
    ])];

/// The probe table, as the emitters take it.
static PROBE_TABLE: &[KernelSig] = &PROBE_SIGS;

/// The same probe row seen as a JIT'd one, so `emit_rust_dispatch` takes its
/// OTHER branch and the three kinds have to cross as `ArgValue`s.
///
/// The template it names does not exist and does not need to: nothing
/// compiles this row's C++, and `emit_rust_dispatch` reads a device row for
/// its symbol and its operands alone.
static PROBE_JIT: DeviceKernel = DeviceKernel {
    sig: &PROBE_SIGS[0],
    template_path: "probe::device::ty_probe",
    elem: "device::bf16",
};

/// The seven rows this change brought inside the check.
const SEVEN: &[&str] = &[
    "quant::quant_bf16_to_fp8_e4m3",
    "quant::quantize_bf16_to_fp8_e4m3_per_channel",
    "quant::quantize_bf16_to_int8_per_channel",
    "quant::cast_bf16_to_fp8_e4m3_per_channel",
    "quant::cast_bf16_to_int8_per_channel",
    "quant::cast_f16_to_bf16",
    "quant::bf16_to_fp16",
];

// ---------------------------------------------------------------------------
// harness
// ---------------------------------------------------------------------------

/// The device row for a symbol, or a panic naming it.
fn device_row(symbol: &str) -> &'static DeviceKernel {
    unit::rows()
        .find(|r| r.sig.symbol == symbol)
        .unwrap_or_else(|| panic!("no device row states `{symbol}`"))
}

/// A mutant is the original with ONE operand's type changed, and nothing else.
///
/// Without this a mutant is a second copy of a row, free to drift from the one
/// it claims to be testing — and a negative control that fails to compile for
/// a reason other than the one it names is not a control at all. Arity, order,
/// names and sources must match; exactly one `ty` must not, and it must be the
/// operand named.
fn differs_in_exactly_one_operand(real: &KernelSig, mutant: &KernelSig, operand: &str) {
    assert_eq!(real.symbol, mutant.symbol, "the mutant is a different row");
    assert_eq!(real.file, mutant.file, "the mutant names a different header");
    assert_eq!(
        real.operands.len(),
        mutant.operands.len(),
        "`{}`: the mutant's arity drifted from the row's",
        real.symbol
    );
    let mut changed = Vec::new();
    for (a, b) in real.operands.iter().zip(mutant.operands) {
        assert_eq!(a.name, b.name, "`{}`: the mutant reordered or renamed operands", real.symbol);
        assert_eq!(a.source, b.source, "`{}`: the mutant re-sourced `{}`", real.symbol, a.name);
        assert_eq!(a.nullable, b.nullable, "`{}`: the mutant re-nulled `{}`", real.symbol, a.name);
        if a.ty != b.ty {
            changed.push(a.name);
        }
    }
    assert_eq!(
        changed,
        vec![operand],
        "`{}`: a mutant must differ in exactly `{operand}` and differs in {changed:?}",
        real.symbol
    );
}

/// Emit `sig` as a device row and require `nvcc` to ACCEPT it.
fn accepts(nvcc: &PathBuf, tag: &str, sig: &'static KernelSig, path: &'static str, elem: &'static str) {
    let tu = abi::emit_device_typecheck(&[DeviceKernel { sig, template_path: path, elem }])
        .unwrap_or_else(|e| panic!("`{}` does not emit: {e}", sig.symbol));
    if let Err(e) = compile(nvcc, tag, &tu) {
        panic!("`{}` as stated does not compile:\n{e}\n\n{tu}", sig.symbol);
    }
}

/// Emit `sig` as a device row and require `nvcc` to REFUSE it, returning what
/// it said.
///
/// The refusal must be the TYPE one. `nvcc` fails a translation unit for many
/// reasons — a missing header, a bad flag — and a control that accepted any of
/// them would pass with the check removed, which is the failure mode this
/// whole file exists to rule out. So the diagnostic is required to be about
/// the initialisation not finding an instantiation of the template.
fn refuses(
    nvcc: &PathBuf,
    tag: &str,
    sig: &'static KernelSig,
    path: &'static str,
    elem: &'static str,
) -> String {
    let tu = abi::emit_device_typecheck(&[DeviceKernel { sig, template_path: path, elem }])
        .unwrap_or_else(|e| panic!("the mutant of `{}` does not emit: {e}", sig.symbol));
    let Err(e) = compile(nvcc, tag, &tu) else {
        panic!(
            "the mistyped `{}` COMPILED -- the typecheck no longer distinguishes \
             the operand types this test exists for:\n\n{tu}",
            sig.symbol
        );
    };
    assert!(
        e.contains("no instance of function template")
            || e.contains("cannot be initialized with")
            || e.contains("incompatible"),
        "the mutant of `{}` failed, but not on the operand type:\n{e}",
        sig.symbol
    );
    e
}

/// Hand one generated translation unit to `nvcc`.
///
/// `-Xcompiler=-iquote,<csrc/src>` and `<csrc/shim>`, and no `-I` anywhere:
/// see this file's header. `-fatbin` rather than `-c`, so the instantiation is actually
/// CODE-GENERATED and the check is not merely a parse.
fn compile(nvcc: &PathBuf, tag: &str, tu: &str) -> Result<(), String> {
    let dir = scratch();
    // A recycled pid would leave someone else's `.cu` and `.fatbin` here, and
    // a control that read a stale object would be reporting a scheduling
    // accident as a type property. Cleared ONCE per process, not per compile:
    // the tests in this file run concurrently and share the directory.
    static CLEAR: std::sync::Once = std::sync::Once::new();
    CLEAR.call_once(|| {
        let _ = std::fs::remove_dir_all(&dir);
    });
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let src = dir.join(format!("{tag}.cu"));
    let obj = dir.join(format!("{tag}.fatbin"));
    std::fs::write(&src, tu).map_err(|e| format!("cannot write {}: {e}", src.display()))?;

    let include = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("csrc/src");
    // The impersonating headers, since `csrc/` was cut by role. A second
    // `-iquote` and not an `-I`: `csrc/src/pie_fp8.cuh` and `pie_half2.cuh`
    // reach `cuda_fp16.h` by quoted include, and this whole file exists to
    // catch a `__half` that is not `device::f16` — a resolver that fell
    // through to the toolkit header here would break the control rather than
    // the code, which is the worse of the two.
    let shim = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("csrc/shim");
    let out = Command::new(nvcc)
        .args(["-std=c++20", "-arch=sm_89", "-fatbin"])
        .arg(format!("-Xcompiler=-iquote,{}", include.display()))
        .arg(format!("-Xcompiler=-iquote,{}", shim.display()))
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .output()
        .map_err(|e| format!("could not run {}: {e}", nvcc.display()))?;
    if out.status.success() {
        return Ok(());
    }
    Err(String::from_utf8_lossy(&out.stderr).into_owned())
}

/// This process's scratch directory, under `target/`.
///
/// Per-process, for the reason `tests/plan.rs` records: two concurrent runs
/// at the same path write each other's files, and a control that failed
/// because another process was mid-write would be a false positive for
/// exactly the property being proved.
fn scratch() -> PathBuf {
    PathBuf::from(env!("OUT_DIR")).join(format!("typecheck-types-{}", std::process::id()))
}

/// `nvcc`, wherever it is.
fn nvcc() -> Option<PathBuf> {
    for candidate in ["nvcc", "/usr/local/cuda/bin/nvcc"] {
        let path = PathBuf::from(candidate);
        if Command::new(&path).arg("--version").output().is_ok_and(|o| o.status.success()) {
            return Some(path);
        }
    }
    None
}

/// Say why a compile did not happen, on the real stderr.
fn skip(why: &str) {
    eprintln!("SKIPPED: {why}");
}
