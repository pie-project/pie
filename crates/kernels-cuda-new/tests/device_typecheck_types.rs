//! The operand kinds that name a NUMERIC FORMAT, and the proof that they are
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
//! `Ty::Bf16sMut` and `Ty::F16sMut` closed the same gap on the WRITTEN side,
//! where a bf16 destination had to say `Ty::BufMut` — `void*`, a type every
//! object pointer converts to. See
//! `the_written_sixteen_bit_kinds_are_the_declarations_own_spelling`, which
//! is the pure-Rust half of that proof; the compiling half is
//! `tests/units.rs`'s `a_written_bf16_is_asserted_as_bf16_by_the_jit`, under
//! NVRTC.
//!
//! A new `Ty` that merely RENDERS is worth nothing here. The property the
//! seven rows are being brought inside is the one `emit_device_typecheck`
//! states in its own header — type identity admits NO parameter conversions —
//! and a variant that spells something both `const bf16*` and `const f16*`
//! would convert to has removed that property while appearing to extend it.
//! So this file does not assert that the new kinds work; it compiles them,
//! both ways round.
//!
//! # And one kind that is not a numeric format
//!
//! `Ty::StructuredMasks` is here for a different failure and is kept in this
//! file because the apparatus is the same one. It is an aggregate — `struct
//! StructuredMaskParams`, three `u32`s — and its `cpp()` named
//! `attn::StructuredMaskParams`, a namespace that belonged to
//! `attn/pack_dense_mask.hpp` and did not outlive it. The struct survives one
//! namespace deeper, in `attn::device`.
//!
//! Nothing failed, and that is the point of including it: `abi::self_describing`
//! declined the kind, so the one row that carries it never handed the
//! spelling to a compiler. A name that resolves and a name nobody asked about
//! are the same green — which is the same shape as an operand that is
//! asserted correctly and one that is not asserted at all, and it is the
//! shape this whole file exists to refuse.
//!
//! # THIS FILE USED TO SPAWN `nvcc` ON THIS TREE'S OWN KERNELS
//!
//! It does not any more. Every CUDA kernel in this tree is compiled by NVRTC
//! and nvcc is forbidden; these four compiles predated that rule, and this is
//! the change that removes them. What matters about how it was done is that
//! it is a PORT and not a deletion.
//!
//! The qualifier in that heading is exact and was derived, not inherited.
//! `Command::new(nvcc)` appears five more times under `crates/`: three
//! offline probes under `examples/`, which are run by hand and are not part
//! of any build, and **two tests** — `plan.rs`'s `build_harness` and
//! `flashinfer_decode.rs`'s `build_reference`. Those two are not candidates
//! for this port and porting them would break them. They compile FlashInfer's
//! UPSTREAM headers out of `target/` into a standalone binary whose numbers
//! are then differenced against the JIT's, and `build_reference`'s own doc
//! says why the compiler has to be a different one: a reference built by the
//! machinery under test is the JIT compared with itself. nvcc's independence
//! from NVRTC is the instrument there, where here it was an accident of
//! history. Neither compiles anything under `csrc/`.
//!
//! The deletion was on the table. `tests/units.rs`'s
//! `a_drifted_row_is_refused_by_the_compiler_the_jit_uses` and
//! `a_written_bf16_is_asserted_as_bf16_by_the_jit` are control-and-mutant
//! proofs of the same instrument through NVRTC, so the obvious reading is
//! that this file's subject is already covered elsewhere. It is not, and the
//! difference was derived rather than assumed:
//!
//! * **[`abi::Elem::Resolve`] has no other compiling consumer anywhere.**
//!   `Unit::typecheck` asks for [`abi::Elem::Opaque`], which DECLINES a
//!   `Ty::Buf` operand instead of resolving it, so the two `Buf` mutants
//!   below — the HISTORICAL spelling, the one that reads a format off the
//!   row's `elem` and gets it wrong — cannot exist in an appendix at all.
//!   They are the mutants that say why the seven rows were retyped.
//! * **`tanh_inplace<T>` templates its destination; `bf16_to_narrow<T>` and
//!   `cast_f16_to<T>` FIX one end.** `units.rs`'s pair proves a drifted
//!   operand is caught where the type moves with `elem`; these prove it where
//!   the type is independent of `elem`, which is the case `Elem::Resolve`
//!   gets wrong. A port that dropped them would narrow the check to the shape
//!   that happens to be covered.
//! * **`I8sMut` against `U8sMut` is covered nowhere else**, and it is the
//!   same width — no launch and no fire would ever have said so.
//!
//! # What the port CHANGED, deliberately
//!
//! **Each mutant now names the lines it moves.** The nvcc harness's claim was
//! *"this compiles, that does not"*, and a compile failure is a weak
//! observation: a translation unit fails for many reasons and only one of
//! them is the operand. So the emitted control and the emitted mutant are
//! diffed LINE BY LINE, in pure Rust, on every machine — the lines that
//! appear must all name the type the mutant claims, the lines that vanish
//! must all name the type the row states, and nothing else may move. See
//! `every_mutant_moves_exactly_its_own_operands_lines`, which is the half of
//! this file that needs no compiler and is strictly stronger than what it
//! replaced.
//!
//! **The refusal names the ROW, not the template.** Under nvcc the check was
//! a function-pointer initialisation and the diagnostic read *"no instance of
//! function template … bf16_to_narrow matches the required type"*. It is a
//! `static_assert` now — see `jit` for why the form had to change — and the
//! message is the emitter's own, which names the symbol, the operand index
//! and the operand. That is what a person greps, so the assertion asks for it.
//!
//! # What the port did NOT change
//!
//! The rows, the mutants, the pinning, and the population. Every mutant below
//! is the same `KernelSig` it was under nvcc, still tied to its original by
//! [`differs_in_exactly_one_operand`], and all seven rows still typecheck
//! together in one translation unit.
//!
//! # `-Xcompiler=-iquote` is gone, and the hazard it guarded is gone with it
//!
//! `csrc/shim` holds fourteen headers that shadow real toolkit ones,
//! `cuda_fp16.h` among them, and that one opens with
//! `using __half = device::f16`. Under nvcc `-I` a `.cuh` reaching
//! `<cuda_fp16.h>` finds the SHIM, and `new-horizon.md` §21.10 records the
//! measurement: the same source under the two spellings both compiled
//! cleanly and exported DIFFERENT mangled symbols —
//! `bf16_to_narrow<__half>` against `bf16_to_narrow<device::f16>`, 31% apart
//! in object size. This file's subject is a `bf16`/`f16` confusion, so
//! compiling it under a flag that manufactured one would have been a negative
//! control proving the opposite of what it claimed. Hence the awkwardness:
//! `nvcc -iquote` is rejected outright, so the quote path had to be smuggled
//! to the host compiler as `-Xcompiler=-iquote,<dir>`, twice, once per role
//! directory, with dropping either being silent.
//!
//! **Under NVRTC there is no include path to get wrong.** An `#include` is
//! matched against `includeNames[]` — [`source::DEVICE_HEADERS`], the same
//! array every unit compiles against — and a name that is not in it is
//! *"could not open source file"* rather than a toolkit header answering
//! quietly. The set is `csrc/shim` and `csrc/src`, in the binary, walked by
//! `build.rs`. So the resolution the flags were arranging is now the
//! mechanism, and there is no second way to spell it.
//!
//! Two references OUTSIDE this file describe what it used to be and are now
//! stale. Neither is in this change's scope, and both are reported rather
//! than edited: `csrc/shim/README.md` lists this file's `compile()` as the
//! fourth of "four sites that name both directories" for `-iquote`, and
//! there are three now; `x/quant.rs:55` and `:1056` quote the nvcc
//! diagnostic *"no instance of function template … matches the required
//! type"* as what this file produces, and it produces a `static_assert`
//! message.
//!
//! # What runs where
//!
//! The rendering assertions, the mutant line-diffs, the coverage comparison,
//! the refusal witnesses and the header-text claims are pure string work over
//! the tables and the `.cuh` sources, and run on any machine. The six
//! compiles are in `jit`, behind the `_cuda` feature, and say so when NVRTC
//! is not loadable.
//!
//! # And two kinds no row may state
//!
//! `Ty::Stream` and `Ty::CublasHandle` are the inverse of everything else
//! here: the rest of this file proves a spelling is RIGHT, and
//! `a_row_that_states_a_handle_is_refused_and_says_which_handle` proves a row
//! stating one is refused, at both `Elem` sites, with a sentence naming which
//! handle it found. `Ty::Dtype` is carried alongside as the control that
//! keeps the claim two variants wide.

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
/// the fire. This is the cheap half of that check — the kinds are pointers.
///
/// The other half — tying the three separate copies of the "is this a
/// pointer" list together (`abi::arg_value_variant`, `abi::cast_for` and
/// `runtime::args::is_pointer`) — used to be `emit.rs`'s
/// `every_kind_the_binder_marshals_crosses_the_same_way`. **There is no
/// `tests/emit.rs`**, and the citation had outlived the file: a
/// cross-reference to a test that does not exist reads exactly like coverage
/// while being none, which is this file's own subject applied to its
/// comments. Nothing has replaced it; the three lists are kept in step by
/// hand and that is a real gap, recorded here rather than implied to be
/// covered.
///
/// The `!matches!` below is not the enforcement, and after the widening of
/// [`abi::device_typecheck`]'s handle refusal it is not the only check
/// either. It stays because its population is different: this walks the SEVEN
/// retyped rows and asserts a property of them, and
/// `no_row_states_a_handle_and_the_guard_would_catch_one_that_did` walks
/// every row there is. `Ty::Stream` and `Ty::CublasHandle` share the `|` here
/// honestly — neither is a kernel argument — which is the one question the
/// two kinds do answer the same way.
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

/// A row that states either handle kind is REFUSED, by name, at both sites.
///
/// # The word existed so a row could be told not to say it, and nothing told it
///
/// Measured over the workspace at `86a1925ef`: `Ty::CublasHandle` has zero
/// writers in every grammar — no `operands![…]`, no `impl Abi`, no
/// `scalar_abi!`/`ptr_abi!` — in every backend. Eleven rows carried one
/// once; `execution`'s `RUST_SERVED` doc says why none may again (*"a handle
/// is the SERVICE's, not the statement's"*). So the variant's whole job is
/// to be a thing a row must not state, and until this test it had no
/// enforcement at all: [`kernels::Ty::Stream`] was refused by name and its
/// near-neighbour was not.
///
/// # What a handle row got before, which was wrong TWICE and differently
///
/// The two sites disagreed, and neither said "this row is unported":
///
/// * [`abi::Elem::Opaque`] — `self_describing` declined it and the row was
///   SKIPPED, with *"whose C++ type is NOT in the tag — `Ty::cpp` spells it
///   `cublasHandle_t`, and in this population that is a projection rather
///   than the parameter"*. It is a projection of nothing. `cublasHandle_t`
///   is `<cublas_v2.h>`'s, a header NVRTC is never given.
/// * [`abi::Elem::Resolve`] — `operand_types` consults `self_describing` for
///   NOTHING under `Resolve`; only `Buf`/`BufMut` read `elem` and every other
///   kind is spelled straight from `Ty::cpp`. So the handle was ASSERTED,
///   emitting `static_assert(… , cublasHandle_t>)` into a translation unit
///   where that name does not exist — a hard NVRTC error naming a type,
///   about a row.
///
/// That is `Ty::StructuredMasks`'s shape again (`86a1925ef`): a kind one row
/// away from a compile error, kept quiet by a decline at the site nobody was
/// compiling. Both are now the same refusal, before either branch.
///
/// # `OUT_AS_DTYPE` is what makes this a measurement
///
/// A red mutant proves the guard fires; it does not prove the guard is
/// narrow. `Ty::Dtype` is declined by the same `self_describing` for a
/// neighbouring reason and must still SKIP under `Opaque` and still be
/// ASSERTED under `Resolve` — unchanged in both. The claim is two variants
/// wide and this is the half that says "and not by more".
#[test]
fn a_row_that_states_a_handle_is_refused_and_says_which_handle() {
    let real = device_row("quant::bf16_to_fp16");
    for m in [&OUT_AS_CUBLAS_HANDLE, &OUT_AS_STREAM, &OUT_AS_DTYPE] {
        differs_in_exactly_one_operand(real.sig, m, "out");
    }

    // The control for the emitter itself: the row as it stands emits.
    assert!(
        abi::emit_device_typecheck(&[DeviceKernel {
            sig: real.sig,
            template_path: real.template_path,
            elem: real.elem,
        }])
        .is_ok(),
        "`quant::bf16_to_fp16` does not emit, so nothing below is about handles"
    );

    // `Site` held fixed, `Elem` the variable — the discipline
    // `resolving_asserts_positions_the_appendix_cannot` uses, so that a
    // difference between the two rows below is a difference in `Elem`.
    let both = |sig: &'static KernelSig| {
        let row = DeviceKernel { sig, template_path: real.template_path, elem: real.elem };
        let of = |elem| abi::device_typecheck(&[&row], abi::Site::Standalone, elem);
        (of(abi::Elem::Resolve), of(abi::Elem::Opaque))
    };

    let mut by_kind: Vec<(&str, Vec<String>)> = Vec::new();
    for (sig, kind, clue) in [
        (&OUT_AS_CUBLAS_HANDLE, "CublasHandle", "cuBLAS handle"),
        (&OUT_AS_STREAM, "Stream", "stream"),
    ] {
        let (resolve, opaque) = both(sig);
        let mut seen = Vec::new();
        for (elem, got) in [("Resolve", resolve), ("Opaque", opaque)] {
            let why = got.err().unwrap_or_else(|| {
                panic!(
                    "a row stating `{kind}` was accepted under `Elem::{elem}` -- \
                     the refusal is at one site and the other still emits it"
                )
            });
            // Named, so the reader is sent to the ROW. A refusal that said
            // only "unsupported kind" would be satisfied by any red and
            // would leave the next reader looking at the emitter.
            assert!(
                why.contains("`quant::bf16_to_fp16`") && why.contains("takes `out` as an operand"),
                "the `{kind}` refusal under `Elem::{elem}` names neither the row \
                 nor the operand: {why}"
            );
            assert!(
                why.contains(clue),
                "the `{kind}` refusal under `Elem::{elem}` does not say which handle \
                 it found: {why}"
            );
            // The DELTA, stated as the disappearance of the old sentence.
            // Without this the test passes on any error, including the one
            // the kind used to produce -- which is the observation this file
            // exists to refuse.
            assert!(
                !why.contains("NOT in the tag") && !why.contains("projection"),
                "the `{kind}` refusal under `Elem::{elem}` is still the DECLINE, \
                 which diagnoses the tag for a row that is unported: {why}"
            );
            seen.push(why);
        }
        // BEFORE the `elem` branch, and the equality is how that is checked
        // rather than asserted in prose. The old behaviour differed BY SITE --
        // skipped under `Opaque`, asserted-and-unresolvable under `Resolve` --
        // so a refusal that still varied would be the same defect moved.
        assert_eq!(
            seen[0], seen[1],
            "`{kind}` is refused differently at the two sites, so the guard is \
             inside the `Elem` branch rather than before it"
        );
        by_kind.push((kind, seen));
    }

    // One guard, two sentences. A single message covering both would pass
    // every assertion above and tell a reader nothing about which kind the
    // emitter found -- and the two are not interchangeable: a stream is
    // statable in fn-world on purpose (`x::abi` carries a marker `impl Abi`
    // for it) and a cuBLAS handle is statable nowhere but a row.
    assert_ne!(
        by_kind[0].1[0], by_kind[1].1[0],
        "`{}` and `{}` are refused with ONE sentence, so the guard reports that \
         it found a handle rather than which",
        by_kind[0].0, by_kind[1].0
    );

    // AND NOT BY MORE. `Ty::Dtype` is declined by the same `self_describing`
    // and both of its sites must read exactly as they did.
    let (resolve, opaque) = both(&OUT_AS_DTYPE);
    let resolve = resolve.expect("`Ty::Dtype` is not a handle and must still emit under `Resolve`");
    assert!(
        resolve.text.contains(&format!("{}>", Ty::Dtype.cpp())),
        "`Ty::Dtype` stopped being asserted under `Elem::Resolve`, so the widening \
         reached a kind it does not name:\n{}",
        resolve.text
    );
    let opaque = opaque.expect("`Ty::Dtype` is not a handle and must still SKIP under `Opaque`");
    assert!(
        opaque.skipped.iter().any(|(sym, why)| *sym == "quant::bf16_to_fp16"
            && why.contains("NOT in the tag")
            && why.contains("out")),
        "`Ty::Dtype` is no longer DECLINED under `Elem::Opaque`, so the widening \
         turned declines into refusals: {:?}",
        opaque.skipped
    );
}

/// No row states either handle kind, and that is the doctrine rather than luck.
///
/// The population is [`unit::rows()`] — every row the JIT compiles, not one
/// list. A row acquiring a handle operand is then a VISIBLE event: it fails
/// here with a sentence, before it reaches
/// [`a_row_that_states_a_handle_is_refused_and_says_which_handle`]'s guard,
/// which would refuse its whole unit.
///
/// The two tests are not redundant. This one measures the population and can
/// only ever observe zero; that one constructs a writer and measures the
/// refusal. **A guard over an empty population and a guard that does not work
/// are the same green** — which is why neither is left to stand alone.
#[test]
fn no_row_states_a_handle_and_the_guard_would_catch_one_that_did() {
    let mut stated = Vec::new();
    let mut rows = 0usize;
    let mut positions = 0usize;
    for row in unit::rows() {
        rows += 1;
        for o in row.sig.operands {
            positions += 1;
            if matches!(o.ty, Ty::Stream | Ty::CublasHandle) {
                stated.push(format!("{}'s `{}` is a `{:?}`", row.sig.symbol, o.name, o.ty));
            }
        }
    }
    // The instrument proves it can see before it reports a zero. A `for` over
    // an empty iterator and a `for` over the whole table finding nothing both
    // leave `stated` empty, and only one of them is a measurement. The
    // relation is `abi`'s own population floor rather than a second number
    // invented here, so the two cannot drift apart.
    assert!(
        rows >= 200 && positions >= rows,
        "only {rows} rows and {positions} operand positions are in view, so a \
         zero below is a claim about the walk rather than about the rows"
    );
    assert!(
        stated.is_empty(),
        "a row states a handle operand, and `execution`'s `RUST_SERVED` doc is \
         why it must not -- a handle belongs to the SERVICE that issues the \
         launch, not to the statement. Move the row to `Execution::Service` \
         and drop the operand; do not relax the guard in `abi::device_typecheck`: \
         {stated:?}"
    );
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

/// The WRITTEN halves of the two sixteen-bit formats, and the tie that stops
/// them being decoration.
///
/// # What is on trial
///
/// [`Ty::Bf16sMut`] and [`Ty::F16sMut`]. They close the asymmetry the three
/// kinds above left: `Bf16s` and `F16s` name a bfloat16 or half operand a
/// kernel READS, and until now nothing named one it WRITES. Such an operand
/// had to say [`Ty::BufMut`], whose `cpp()` is `void*` — a type every object
/// pointer converts to, so an assertion against it is one that holds for
/// every possible kernel. `abi::self_describing` declines it for that reason.
///
/// # The tie is the point, and it is now a tie to something real
///
/// This test carried a TRIPWIRE, and the sentence it was written around was:
/// *"nothing in the tree produces the new kinds yet — `x::abi`'s
/// `ptr_abi!(bf16, …)` still tags `*mut bf16` [`Ty::BufMut`], and a `Ty`
/// variant that nothing produces and nothing compiles is a decoration."*
/// That was true when it was written and it is history now. The two tags
/// moved; the assertion below is the tripwire INVERTED, so the same line that
/// used to announce the change now defends it.
///
/// Keeping the record rather than deleting it, because the reason the
/// variants were correct-and-unused is the reason they were worth having: the
/// declaration side was never the problem. `ptr_abi!` has ALWAYS declared
/// `Abi::CPP` for `*mut bf16` as
/// `::pie_cuda_driver::kernels::device::bf16*`, beside a comment saying the
/// spellings are `Ty::cpp()`'s *"so a row and a declaration describing the
/// same parameter produce the same typecheck line"* — a sentence that was
/// **false of these two parameters for as long as the tag said `BufMut`**,
/// because `Ty::BufMut.cpp()` is `void*`. The first two assertions below are
/// that sentence, checked. Edit either side alone and this is red.
///
/// # What moved, measured rather than asserted from the message
///
/// The tripwire's own text said *"134 rows' destinations"* and a floor
/// comment said 183 positions. Both quantities were named correctly — rows
/// and positions are different things and it distinguished them — and both
/// VALUES were wrong, having been derived by an extractor that stopped at the
/// first `]` of an operand list. Re-derived at `d737aad29` over `unit!`'s
/// declarations with each row's own `where [T = …]` substituted:
///
/// | | |
/// |---|---|
/// | fn-world rows / operand positions | 266 / 2207 |
/// | rows carrying a written 16-bit destination | **172** |
/// | positions | **269** = 252 bf16 + 17 f16 |
/// | rows fully checked, before → after | 61 → 226 |
/// | positions asserted, before → after | 1843 (83%) → 2112 (95%) |
///
/// `the_written_sixteen_bit_positions_are_two_hundred_and_sixty_nine`
/// re-derives 269 from `unit::rows()` at run time, so this table is a note
/// and not the check.
///
/// `tests/units.rs`'s `a_written_bf16_is_asserted_as_bf16_by_the_jit` is the
/// other half: it compiles the real `norm::tanh_bf16` under the new kind
/// through NVRTC and requires [`Ty::F16sMut`] on the same operand to be
/// refused. This one runs on any machine.
#[test]
fn the_written_sixteen_bit_kinds_are_the_declarations_own_spelling() {
    use kernels_cuda_new::x::abi::Abi;
    use std::ptr::NonNull;

    assert_eq!(
        Ty::Bf16sMut.cpp(),
        <*mut kernels_cuda_new::x::abi::bf16 as Abi>::CPP,
        "`Ty::Bf16sMut` and `x::abi`'s declaration of `*mut bf16` no longer spell \
         the same C++ type, so a row and a fn-world parameter describing one \
         kernel argument would generate two different assertions"
    );
    assert_eq!(
        Ty::F16sMut.cpp(),
        <*mut kernels_cuda_new::x::abi::f16 as Abi>::CPP,
        "`Ty::F16sMut` and `x::abi`'s declaration of `*mut f16` no longer agree"
    );

    // THE TRIPWIRE, INVERTED. It read `assert_eq!(…::TY, Ty::BufMut)` with a
    // message saying what to do when it moved. It has moved, so the same
    // equality now says what must not move back -- and a revert to `BufMut`
    // is not a compile error anywhere, because `void*` accepts every pointer
    // the declarations could hand it. That is exactly why it needs a line.
    assert_eq!(
        <*mut kernels_cuda_new::x::abi::bf16 as Abi>::TY,
        Ty::Bf16sMut,
        "`*mut bf16` is tagged {:?} rather than `Ty::Bf16sMut`. If that is \
         `Ty::BufMut` the widening has been reverted and 252 destination \
         positions stopped being asserted SILENTLY -- `void*` is a type every \
         object pointer converts to, so nothing downstream fails",
        <*mut kernels_cuda_new::x::abi::bf16 as Abi>::TY
    );
    assert_eq!(
        <*mut kernels_cuda_new::x::abi::f16 as Abi>::TY,
        Ty::F16sMut,
        "`*mut f16` is tagged {:?} rather than `Ty::F16sMut`",
        <*mut kernels_cuda_new::x::abi::f16 as Abi>::TY
    );

    // AND THE NULLABLE SPELLING, which is a SECOND impl and not a synonym.
    // `ptr_abi!` maps `Option<NonNull<T>>` to the MUT tag, and 13 of the 269
    // positions (7 bf16, 6 f16) arrive that way. A hand-edit that moved only
    // the `*mut T` arm would leave those thirteen at `void*` and every
    // assertion above would still pass.
    assert_eq!(
        <Option<NonNull<kernels_cuda_new::x::abi::bf16>> as Abi>::TY,
        Ty::Bf16sMut,
        "the nullable spelling of a bf16 destination is not the written kind"
    );
    assert_eq!(
        <Option<NonNull<kernels_cuda_new::x::abi::f16>> as Abi>::TY,
        Ty::F16sMut,
        "the nullable spelling of an f16 destination is not the written kind"
    );
    // And the READ halves are unmoved. `ptr_abi!` takes both tags in one
    // invocation, so a mistake in it is as likely to land on the const side.
    assert_eq!(<*const kernels_cuda_new::x::abi::bf16 as Abi>::TY, Ty::Bf16s);
    assert_eq!(<*const kernels_cuda_new::x::abi::f16 as Abi>::TY, Ty::F16s);

    // WHAT THE KIND BUYS, stated as the difference it makes rather than as a
    // rendering. `void*` is not a wrong spelling of a bf16 destination -- it
    // is a spelling that cannot be wrong, which is why it is declined.
    assert_eq!(Ty::BufMut.cpp(), "void*");
    assert!(!abi::self_describing(Ty::BufMut), "`void*` became a checkable type");
    assert!(abi::self_describing(Ty::Bf16sMut), "the written kind is not asserted");
    assert!(abi::self_describing(Ty::F16sMut), "the written kind is not asserted");

    assert_ne!(
        Ty::Bf16sMut.cpp(),
        Ty::F16sMut.cpp(),
        "the two written formats collapsed to one C++ spelling, which is the state \
         `pie_device.cuh` made them structs to prevent -- and they are the same \
         WIDTH, so nothing else in the tree would notice"
    );
    assert_eq!(Ty::Bf16sMut.rust(), "*mut u16");
    assert_eq!(Ty::F16sMut.rust(), "*mut u16");
    for ty in [Ty::Bf16sMut, Ty::F16sMut] {
        assert!(!ty.needs_mirror(), "{ty:?} claims a mirror it has no struct for");
    }

    // AND THE EMITTER WRITES THEM, on the REAL row. The subject is
    // `norm::tanh_bf16`, whose `__global__` is
    // `tanh_inplace<T>(T* __restrict__ x, int n)`. This used to be a
    // synthetic copy of that row stating `x: Bf16sMut`, because the tag
    // prevented the real one from saying it; the copy is deleted and this
    // reads `unit::rows()`.
    let real = device_row("norm::tanh_bf16");
    assert_eq!(
        real.sig.operands[0].ty,
        Ty::Bf16sMut,
        "`norm::tanh_bf16`'s `x` is stated {:?} -- the real row does not carry the \
         written kind, so every assertion above is about a declaration nothing uses",
        real.sig.operands[0].ty
    );
    let stated = emit_one(real.sig, real);
    assert!(
        stated.contains("::pie_cuda_driver::kernels::device::bf16*>"),
        "the destination is not asserted as `bf16*`:\n{stated}"
    );
    assert!(
        !stated.contains("::pie_cuda_driver::kernels::device::f16*>"),
        "the bf16 row asserts an f16 destination:\n{stated}"
    );

    // A DRIFTED OPERAND MOVES ITS OWN LINES AND NOTHING ELSE -- through the
    // shared `MUTANTS` machinery rather than a hand-rolled diff, which is
    // what the real row bought. `moves_only_its_own_lines` adds the check the
    // hand-rolled one never made: that the lines which moved index operand
    // `::at<0>`, so the emitter is sensitive to WHERE the tag is and not only
    // to the tag.
    let m = MUTANTS
        .iter()
        .find(|m| m.tag == "tanh_bf16.f16s")
        .expect("the destination mutant left `MUTANTS`, so nothing line-diffs it");
    moves_only_its_own_lines(m, real);
}

/// **269 operand positions across 172 rows, and every one of them asserted.**
///
/// # Why an exact count and not a floor
///
/// Every coverage assertion in this tree is a `>=` floor, on the argument
/// that a row added to a unit must not fail a test about the instrument. That
/// argument is right and it has a cost: **a floor cannot see an
/// improvement.** `src/abi.rs`'s `checked >= 40` was green before the two
/// tags moved, green after, and would have been green had the change done
/// nothing at all — so no existing assertion in the tree can distinguish this
/// change from its absence, which is the property this whole file exists to
/// refuse.
///
/// So the delta is measured exactly, and BOTH halves of it:
///
/// * **not fewer** — 269 positions carry a written sixteen-bit kind;
/// * **not more** — every one of them is `self_describing`, and the unit's
///   `asserted` count equals the number of `self_describing` operands it
///   holds, recomputed here from the operand list rather than read back from
///   the same function that produced it.
///
/// The second is the half that catches a change which widened too far. A tag
/// move that had also switched, say, `Ty::Buf` to a resolved spelling would
/// raise `asserted` and satisfy any floor; here it fails, naming the unit.
///
/// It also carries the tree-wide floor `src/abi.rs`'s
/// `the_real_population_reaches_the_fp8_assertion` states as `checked >= 40`
/// — a number measured when nothing produced the written kinds and now five
/// times below the truth. See the tail of this test for why the raise lives
/// here and why it is `>= 226` rather than an equality.
///
/// # The number is a measurement and says where it was taken
///
/// 252 bf16 + 17 f16 = 269, over 172 distinct symbols, six of which carry
/// both formats. Measured at `d737aad29` and re-derived unchanged at
/// `a91cadcec`, which rewrote 51% of `src/`'s comment volume — comments do
/// not carry rows, and the four counts came out identical.
/// **A row added to a `unit!` will fail this test, and that is intended** —
/// the message says so and says what to do. An equality that nobody may
/// update is a floor with extra steps; an equality that must be updated
/// deliberately is a record of the population at a commit.
#[test]
fn the_written_sixteen_bit_positions_are_two_hundred_and_sixty_nine() {
    const BF16: usize = 252;
    const F16: usize = 17;
    const ROWS: usize = 172;

    let mut bf16 = 0usize;
    let mut f16 = 0usize;
    let mut symbols: Vec<&str> = Vec::new();
    let mut carrying_units = 0usize;
    let mut checked = 0usize;
    let mut row_count = 0usize;
    let mut asserted = 0usize;
    let mut positions = 0usize;

    for u in unit::UNITS {
        let rows: Vec<_> = u.rows.iter().collect();
        let tc = u
            .typecheck(&rows)
            .unwrap_or_else(|why| panic!("unit `{}` refuses its own rows: {why}", u.name));
        checked += tc.checked;
        row_count += rows.len();
        asserted += tc.asserted;
        positions += tc.positions;

        // THE "NOT BY MORE" HALF. `Unit::typecheck` emits at
        // `Elem::Opaque`, where a kind is spelled if and only if
        // `self_describing` takes it, so the count is derivable from the
        // operand list alone -- and deriving it HERE, from `Ty`, is what
        // makes this a check rather than the emitter agreeing with itself.
        let want: usize = rows
            .iter()
            .map(|r| r.sig.operands.iter().filter(|o| abi::self_describing(o.ty)).count())
            .sum();
        assert_eq!(
            tc.asserted, want,
            "unit `{}` asserts {} of its {} positions and `self_describing` accepts \
             {want} of them. The emitter and the predicate disagree, so one of the \
             two widened without the other",
            u.name, tc.asserted, tc.positions
        );

        let mut mine = 0usize;
        for r in &rows {
            let mut row_has = false;
            for o in r.sig.operands {
                match o.ty {
                    Ty::Bf16sMut => bf16 += 1,
                    Ty::F16sMut => f16 += 1,
                    _ => continue,
                }
                row_has = true;
                mine += 1;
                // AND THE POSITION IS IN THE TEXT. A kind counted in the
                // row table and skipped by the emitter would satisfy every
                // count above -- an unasserted position is what this file
                // was written about.
                assert!(
                    tc.text.contains(&format!("{}>", o.ty.cpp())),
                    "`{}`'s `{}` is stated {:?} and unit `{}`'s typecheck never \
                     spells `{}`",
                    r.sig.symbol,
                    o.name,
                    o.ty,
                    u.name,
                    o.ty.cpp()
                );
            }
            if row_has {
                symbols.push(r.sig.symbol);
            }
        }
        if mine > 0 {
            carrying_units += 1;
        }
    }

    symbols.sort_unstable();
    symbols.dedup();

    assert_eq!(
        (bf16, f16, bf16 + f16),
        (BF16, F16, BF16 + F16),
        "the written sixteen-bit population is {bf16} bf16 + {f16} f16 = {} and was \
         {BF16} + {F16} = {} at `d737aad29`. If a `unit!` row was added or removed \
         this is the number to update, deliberately -- it is here because every \
         other coverage assertion in the tree is a floor, and a floor cannot see \
         an improvement or a partial revert",
        bf16 + f16,
        BF16 + F16
    );
    assert_eq!(
        symbols.len(),
        ROWS,
        "{} rows carry a written sixteen-bit destination and {ROWS} did. Rows and \
         POSITIONS are different quantities -- the tripwire this replaces named \
         both correctly and valued both wrongly",
        symbols.len()
    );
    assert!(
        carrying_units > 1,
        "every written sixteen-bit destination in the tree is in ONE unit, so this \
         test's population is a single compilation and not the tree's"
    );

    // AND THE TREE-WIDE FLOOR, raised here rather than in `src/abi.rs`.
    //
    // `abi.rs`'s `the_real_population_reaches_the_fp8_assertion` holds
    // `checked >= 40`, measured when nothing produced the written kinds, and a
    // floor five times below the truth is a floor that cannot go red. It is
    // raised HERE because the exact number belongs beside the measurement that
    // produced it, and because `>= 226` is derivable and a tree-wide EQUALITY
    // is not: `unit::UNITS` mixes 266 fn-world rows with the row-grammar units
    // of `families::*` and `x::xqa`, and only the first were counted
    // statically. So `checked` is at least the 226 measured within `unit!`
    // alone, and `positions` at least the 2207 those rows carry.
    assert!(
        checked >= 226,
        "{checked} of {row_count} rows are FULLY checked. 226 were measured within \
         `unit!`'s rows alone at `d737aad29`, and the row-grammar units can only \
         add -- so anything below 226 is coverage going backwards, which is the \
         only direction worth a failure"
    );
    assert!(
        positions >= 2207,
        "{positions} operand positions tree-wide, against 2207 in the fn-world rows \
         alone. The population shrank"
    );
    // THE DELTA, as a subtraction rather than as a constant. `asserted` minus
    // the 269 above is what this tree asserted with both tags at
    // `Ty::BufMut` -- 83% of positions rather than 95% -- and stating it this
    // way means no second number has to be maintained.
    assert!(
        asserted > bf16 + f16,
        "the whole tree asserts {asserted} positions and {} of them are the written \
         sixteen-bit destinations, so this change is the only thing asserting \
         anything and the measurement is of itself",
        bf16 + f16
    );
}

// ---------------------------------------------------------------------------
// The negative controls, in pure Rust
// ---------------------------------------------------------------------------

/// **A mutant moves exactly the lines of the operand it drifts, and no
/// others.**
///
/// # Why this test exists at all
///
/// The nvcc harness this file used to carry asked one question of each
/// mutant: *did the translation unit fail to build?* That is a weak
/// observation, and §79's rule 3 says why — a "must not compile" test is
/// satisfied by its own setup failing, so a red that would have been red
/// anyway proves nothing about the check under test. The nvcc form leaned on
/// the DIAGNOSTIC's wording to close the gap, which ties the proof to a
/// compiler's phrasing.
///
/// This closes it in the emitted text instead, where it is a fact rather than
/// a message. For each mutant:
///
/// * the lines that VANISH from the control must every one of them name the
///   C++ type the real row asserts for that operand;
/// * the lines that APPEAR in the mutant must every one of them name the C++
///   type the mutant asserts;
/// * both sets must be non-empty — a mutant that changed nothing in the text
///   is two names for one kind, which is precisely the state `Ty::Bf16s` and
///   `Ty::F16s` were added to leave.
///
/// So a compile failure downstream cannot be incidental: the two texts differ
/// in these lines and nothing else, and the control compiles.
///
/// # `Ty::Buf` IS the wrong named kind, and that is stated as an equality
///
/// The two `Buf` mutants are the HISTORICAL spelling — the state these rows
/// were in before the three kinds existed — and under [`abi::Elem::Resolve`]
/// a `Buf` is spelled `const {head of elem}*`. For `quant::bf16_to_fp16`,
/// whose `elem` is `device::f16`, that is character-for-character
/// [`Ty::F16s`]'s `cpp()`. The bug and the mistake produce the same text,
/// which is not a coincidence to be discovered by reading two panics: it is
/// the whole reason the retyping was necessary, so it is asserted.
///
/// It also means the `Buf` mutant and the wrong-kind mutant compile to the
/// same TU. Both are still compiled in `jit`, because they arrive there down
/// DIFFERENT emitter paths — `device_cpp_ty`'s `elem` resolution against
/// `Ty::cpp` — and a change that broke one silently would leave the other
/// green.
#[test]
fn every_mutant_moves_exactly_its_own_operands_lines() {
    for m in MUTANTS {
        let real = device_row(m.real);
        moves_only_its_own_lines(m, real);
    }
}

/// The whole of the line-diff claim for one mutant, so that the machine that
/// COMPILES it makes the same claim the machine that cannot does.
///
/// Called from `every_mutant_moves_exactly_its_own_operands_lines`, which
/// runs anywhere, and from `jit::run`, which runs where NVRTC is. One
/// function rather than a comment saying the other test covers it: a test
/// selected out by a `--test` filter covers nothing, and "the refusal must be
/// about this operand" is the claim the compile is being asked to support.
fn moves_only_its_own_lines(m: &Mutant, real: &'static DeviceKernel) {
    differs_in_exactly_one_operand(real.sig, m.sig, m.operand);

    let stated = emit_one(real.sig, real);
    let drifted = emit_one(m.sig, real);
    assert_ne!(
        stated, drifted,
        "`{}`: the mutant emits the SAME text as the row it mutates, so the two \
         kinds it is meant to tell apart are one kind",
        m.tag
    );

    let gone = only_in(&stated, &drifted);
    let came = only_in(&drifted, &stated);
    assert!(
        !gone.is_empty() && !came.is_empty(),
        "`{}`: one side of the diff is empty, so an assertion was added or removed \
         rather than RETYPED -- and a mutant with no assertion at its position \
         compiles for the same reason an empty file does",
        m.tag
    );
    assert!(
        gone.iter().all(|l| l.contains(m.stated_cpp)),
        "`{}`: dropping the mutant's tag moved lines that are not about `{}`: {gone:?}",
        m.tag,
        m.operand
    );
    assert!(
        came.iter().all(|l| l.contains(m.drifted_cpp)),
        "`{}`: the mutant added lines that do not assert `{}`: {came:?}",
        m.tag,
        m.drifted_cpp
    );
    // And the assertion that moved is the one at this operand's POSITION. An
    // emitter that asserted the right types in the wrong order would satisfy
    // everything above -- `pie_pick` indexes the parameter pack, so the index
    // is the claim.
    let n = real
        .sig
        .operands
        .iter()
        .position(|o| o.name == m.operand)
        .expect("`differs_in_exactly_one_operand` found it");
    assert!(
        gone.iter().any(|l| l.contains(&format!("::at<{n}>")))
            && came.iter().any(|l| l.contains(&format!("::at<{n}>"))),
        "`{}`: the lines that moved do not index operand {n}, so the emitter is \
         sensitive to the tag and not to WHERE it is:\n{gone:?}\n{came:?}",
        m.tag
    );
}

/// The two `Buf` mutants spell what the row's `elem` says, and that is the
/// wrong format — as an equality rather than as a story.
///
/// See `every_mutant_moves_exactly_its_own_operands_lines`. This is the same
/// claim isolated, because it is the one that says why
/// [`abi::Elem::Resolve`] must keep a consumer: nothing else in the tree
/// compiles a `Ty::Buf` resolved against an element type, so nothing else can
/// notice if the resolution stops happening.
#[test]
fn the_historical_buf_spelling_is_the_wrong_named_kind() {
    for (symbol, wrong) in
        [("quant::bf16_to_fp16", Ty::F16s), ("quant::cast_f16_to_bf16", Ty::Bf16s)]
    {
        let row = device_row(symbol);
        let head = row.elem.split(',').next().unwrap_or(row.elem).trim();
        assert_eq!(
            format!("const ::pie_cuda_driver::kernels::{head}*"),
            wrong.cpp(),
            "`{symbol}` states `elem = {}`, and `Ty::Buf` resolved against it no longer \
             produces `{wrong:?}`'s spelling. That equality is the DEFECT these rows \
             were retyped to escape -- if it has stopped holding, the `Buf` mutants \
             below are testing something else",
            row.elem
        );
    }
}

/// [`abi::Elem::Resolve`] asserts operand positions [`abi::Elem::Opaque`]
/// cannot, and the difference is the reason both modes exist.
///
/// `Elem::Opaque` declines `Ty::Buf`/`Ty::BufMut` because in the mixed
/// fn-world population the tag is a projection of something wider — a
/// `*mut bf16` and a `*mut c_void` carry the same one. `Elem::Resolve` spells
/// them from the row's `elem` because in this population it is the contract.
/// That is not a preference, it is a COUNT, and a count is what keeps a
/// "port" from quietly narrowing a check.
///
/// # `Site` is held fixed and `Elem` is the variable
///
/// Both emissions below are [`abi::Site::Standalone`]. The JIT's appendix is
/// `Site::Appendix` AND `Elem::Opaque` together, so comparing the pair
/// against the pair would leave two things different and a change in the
/// numbers attributable to either. The claim is about `Elem`.
///
/// # Why the strict inequality is measured over the MUTANTS
///
/// Over the seven real rows the gap is two positions today, and both of them
/// are `Ty::BufMut` operands that `x::abi`'s `ptr_abi!(bf16, …)` and
/// `ptr_abi!(f16, …)` are one line each away from retagging
/// [`kernels::Ty::Bf16sMut`] and [`kernels::Ty::F16sMut`] — which is a change
/// this file ARGUES FOR. Pinning `>` to those two would make a correct change
/// red for the wrong reason, and a test that fights an improvement is deleted
/// rather than read.
///
/// So the strict half is measured over the two `Buf` mutants, whose
/// `Ty::Buf` is written in THIS file and cannot drift with `ptr_abi!`. They
/// are also the honest subject: `Ty::Buf` against a resolved `elem` is
/// exactly what `Elem::Resolve` is for and exactly what nothing else in the
/// tree compiles.
#[test]
fn resolving_asserts_positions_the_appendix_cannot() {
    // The seven, where the claim is that `Resolve` is COMPLETE and never
    // narrower -- both stable across the `ptr_abi!` change.
    let seven: Vec<&DeviceKernel> = SEVEN.iter().map(|s| device_row(s)).collect();
    let resolved = abi::device_typecheck(&seven, abi::Site::Standalone, abi::Elem::Resolve)
        .expect("the seven rows emit");
    let opaque = abi::device_typecheck(&seven, abi::Site::Standalone, abi::Elem::Opaque)
        .expect("the seven rows emit");
    assert_eq!(
        resolved.positions, opaque.positions,
        "the two modes disagree about how many operands the seven rows have, which is \
         not something either of them decides"
    );
    assert_eq!(
        resolved.checked,
        SEVEN.len(),
        "not every one of the seven is FULLY checked under `Elem::Resolve`, so the \
         compiles in `jit` prove less than they read as; skipped: {:?}",
        resolved.skipped
    );
    assert!(
        resolved.asserted >= opaque.asserted,
        "`Elem::Resolve` asserts FEWER of the seven rows' positions than `Elem::Opaque` \
         ({} against {}) -- resolving is meant to be the wider mode",
        resolved.asserted,
        opaque.asserted
    );

    // The two mutants that carry the buffer kinds, where the claim is strict
    // and permanent.
    for m in MUTANTS.iter().filter(|m| m.tag.ends_with(".buf")) {
        let row = device_row(m.real);
        let one = [DeviceKernel { sig: m.sig, template_path: row.template_path, elem: row.elem }];
        let by_ref: Vec<&DeviceKernel> = one.iter().collect();
        let resolved = abi::device_typecheck(&by_ref, abi::Site::Standalone, abi::Elem::Resolve)
            .expect("a mutant emits");
        let opaque = abi::device_typecheck(&by_ref, abi::Site::Standalone, abi::Elem::Opaque)
            .expect("a mutant emits");
        // The expected gap is COUNTED off the mutant rather than written
        // down: one position per buffer-kind operand, because those are the
        // only two tags the two modes treat differently.
        let buffers =
            m.sig.operands.iter().filter(|o| matches!(o.ty, Ty::Buf | Ty::BufMut)).count();
        assert!(
            buffers > 0,
            "`{}` is a buffer-kind mutant and no longer states one, so it cannot be \
             the thing that proves `Elem::Resolve` resolves anything",
            m.tag
        );
        assert_eq!(
            resolved.asserted,
            opaque.asserted + buffers,
            "`{}` holds {buffers} buffer-kind operands, so `Elem::Resolve` must assert \
             exactly that many positions more than `Elem::Opaque` -- {} against {}. If \
             the gap has closed, `Elem::Resolve` has stopped resolving and this file's \
             two `.buf` compiles emit the same text as its two named-kind ones",
            m.tag,
            resolved.asserted,
            opaque.asserted
        );
        assert_eq!(
            opaque.checked, 0,
            "`{}`'s buffer kinds are now self-describing, so `Elem::Opaque` could carry \
             this row and `Elem::Resolve` has lost the only thing it alone can do",
            m.tag
        );
    }
}

/// `Ty::StructuredMasks` is carried by exactly one position, and it is
/// asserted.
///
/// # What this is the witness for
///
/// `Ty::cpp()` spelled this kind
/// `::pie_cuda_driver::kernels::attn::StructuredMaskParams*` — namespace
/// `attn`, which was `attn/pack_dense_mask.hpp`'s. That header is deleted and
/// one definition survives, `attn::device::StructuredMaskParams` at
/// `csrc/src/attn/pack_dense_mask.cuh:136`. **A namespace outliving its
/// header is a type that still exists and can no longer be named.**
///
/// It broke nothing, and that is the whole difficulty: `abi::self_describing`
/// declined the kind, so `attn::pack_structured_mask`'s `masks` position was
/// never emitted and the spelling was never handed to a compiler. The decline
/// was load-bearing in the wrong direction — it was the thing keeping the
/// defect quiet.
///
/// # Exactly one, and the count is the claim
///
/// The widening can admit only as many positions as the tree carries, so the
/// honest form of *"the assertion count went up by one"* is *"there is one
/// position of this kind, and here it is"*. That is derived from
/// [`unit::rows()`] below rather than read off the change — a list built from
/// the claim would return the claim.
///
/// The row's other six operands were all self-describing already, so this
/// single position also moves the ROW from `skipped` into `checked`. Both
/// numbers are asserted, because a widening that added an assertion without
/// completing the row would look identical in a coverage summary.
#[test]
fn the_structured_mask_kind_is_carried_by_exactly_one_position() {
    const SYMBOL: &str = "attn::pack_structured_mask";

    // THE POPULATION, from the tables and not from the change.
    let carriers: Vec<(&str, usize)> = unit::rows()
        .flat_map(|k| {
            k.sig
                .operands
                .iter()
                .enumerate()
                .filter(|(_, o)| o.ty == Ty::StructuredMasks)
                .map(move |(n, _)| (k.sig.symbol, n))
        })
        .collect();
    assert_eq!(
        carriers,
        vec![(SYMBOL, 4)],
        "`Ty::StructuredMasks` is no longer carried by exactly `{SYMBOL}`'s operand 4. \
         If a row was ADDED this test's arithmetic below is stale; if the row was \
         removed the kind has no writer at all and belongs in the classification of \
         writerless kinds rather than in `self_describing`'s accepted set"
    );

    let row = device_row(SYMBOL);
    let tags: Vec<(&str, Ty)> = row.sig.operands.iter().map(|o| (o.name, o.ty)).collect();
    assert_eq!(
        tags,
        vec![
            ("positions", Ty::U32s),
            ("klen", Ty::U32s),
            ("qo_indptr", Ty::U32s),
            ("mask_indptr", Ty::I32s),
            ("masks", Ty::StructuredMasks),
            ("packed", Ty::U8sMut),
            ("b", Ty::I32),
        ],
        "`{SYMBOL}`'s operand list moved. The six siblings being self-describing is \
         what makes `masks` alone the difference between a skipped row and a checked \
         one, and this test's `checked == 1` rests on it"
    );
    for o in row.sig.operands {
        assert!(
            abi::self_describing(o.ty),
            "`{SYMBOL}`'s `{}` is a `{:?}`, which `self_describing` declines -- so this \
             row is skipped for a reason other than the one this test is about",
            o.name,
            o.ty
        );
    }

    // THE DELTA. Seven of seven, and the row whole.
    let checked = abi::device_typecheck(&[row], abi::Site::Appendix, abi::Elem::Opaque)
        .unwrap_or_else(|why| panic!("`{SYMBOL}` refuses the emitter: {why}"));
    assert_eq!(
        (checked.checked, checked.asserted, checked.positions),
        (1, 7, 7),
        "`{SYMBOL}` is not fully checked at all seven positions:\n{:?}",
        checked.skipped
    );
    // Only the FIRST of an operand's two lines carries the closing `>`, so
    // anchoring on the bare spelling would count the message line too.
    assert!(
        checked.text.contains(&format!("{}>", Ty::StructuredMasks.cpp())),
        "the `masks` position is not asserted as `{}`:\n{}",
        Ty::StructuredMasks.cpp(),
        checked.text
    );
    assert!(
        !checked.text.contains("kernels::attn::StructuredMaskParams"),
        "the emitted text still names the header-era `attn::StructuredMaskParams`, \
         which no declaration in the tree provides:\n{}",
        checked.text
    );

    // THE MUTANT is registered in `MUTANTS`, so the line-diff and the compile
    // are made over the same entry by the same function. Asserted here too
    // because the walkers are the wrong place to notice an ABSENCE: `jit::run`
    // refuses a row with no mutant, and `jit::run` is where NVRTC is.
    assert!(
        MUTANTS.iter().any(|m| m.real == SYMBOL && m.operand == "masks"),
        "no mutant retypes `{SYMBOL}`'s `masks`, so nothing establishes that the \
         position this test counts is a position the compiler can tell apart -- and a \
         TU with no assertion at operand 4 compiles exactly like one that checks it"
    );
}

/// The namespace `Ty::cpp()` names is the namespace the header opens.
///
/// The defect stated against the C++ text rather than inferred from a compile:
/// `pack_dense_mask.cuh` opens `pie_cuda_driver::kernels::attn::device` and
/// declares `struct StructuredMaskParams` inside it, and `__global__ void
/// pack_structured_mask` takes a `const StructuredMaskParams*`. All three are
/// read out of the file here, so a header that moves the struct to another
/// namespace fails with its own message rather than as an NVRTC log.
///
/// `include_str!` and not a compile on purpose, and for `jit`'s benefit as
/// much as this test's: a deleted or moved header is a BUILD failure here,
/// which is the one diagnosis a `#include` inside a generated string cannot
/// give.
#[test]
fn the_structured_mask_struct_is_declared_where_cpp_says_it_is() {
    let cuh = include_str!("../csrc/src/attn/pack_dense_mask.cuh");
    assert!(
        cuh.contains("namespace pie_cuda_driver::kernels::attn::device {"),
        "`attn/pack_dense_mask.cuh` does not open \
         `pie_cuda_driver::kernels::attn::device`, so every claim below is about the \
         wrong file"
    );
    assert!(
        cuh.contains("struct StructuredMaskParams {"),
        "`attn/pack_dense_mask.cuh` no longer declares `StructuredMaskParams`, so \
         `Ty::StructuredMasks` names nothing again -- which is the state this kind was \
         in until its `cpp()` was corrected"
    );
    assert!(
        cuh.contains("const StructuredMaskParams* __restrict__ masks"),
        "no `__global__` in `attn/pack_dense_mask.cuh` takes a \
         `const StructuredMaskParams*`, so the kind is a vocabulary word the kernels \
         do not say"
    );
    assert_eq!(
        Ty::StructuredMasks.cpp(),
        "const ::pie_cuda_driver::kernels::attn::device::StructuredMaskParams*",
        "`Ty::cpp()` names a namespace other than the `attn::device` this header opens. \
         It said `::pie_cuda_driver::kernels::attn::StructuredMaskParams*` until this \
         kind acquired a consumer -- `attn` was `attn/pack_dense_mask.hpp`'s, that \
         header is deleted, and a namespace outliving its header is a type that still \
         exists and can no longer be named"
    );
}

// ---------------------------------------------------------------------------
// The negative controls, where the compiler is
// ---------------------------------------------------------------------------

/// The six compiles, through **NVRTC** — the compiler the JIT itself uses.
///
/// # Why this is a module and not six `#[cfg]` attributes
///
/// The rest of this file is string work over static tables and runs on a
/// laptop with no CUDA anything. These six need `libnvrtc`, so they need the
/// `_cuda` feature, so they need `cudarc` in scope — and putting the import
/// at file scope would take the pure-Rust half of the proof down with it on
/// every machine that does not select a CUDA version. The split is the
/// feature boundary, drawn once.
///
/// # What is compiled, and why it is not `Unit::source`
///
/// [`abi::emit_device_typecheck`] — [`abi::Site::Standalone`] and
/// [`abi::Elem::Resolve`] — produces a translation unit that `#include`s the
/// headers holding the templates it checks. That is a shape the JIT never
/// compiles: `Unit::source` APPENDS its assertions to a root that already
/// holds the templates, because a second `#include` of the root under the
/// header set's name for it would define every `__global__` twice and
/// `#pragma once` cannot collapse one text met under two names.
///
/// So this is the only place a standalone typecheck TU meets a compiler, and
/// it works for the reason the appendix does not: the `#include`s here name
/// OTHER files. `#include "quant/dequant_wna16.cuh"` is matched against
/// `includeNames[]` — literally, no path search — and
/// [`source::DEVICE_HEADERS`] is the array, the same one every unit resolves
/// against. Its own `#include "pie_device.cuh"` and `<cuda_fp16.h>` resolve
/// there too, which is how the shim keeps answering for the toolkit's names
/// without a single flag saying so.
///
/// # The assertion form, and why the diagnostic changed
///
/// It was `[[maybe_unused]] void (*const check_x)(…) = &instantiation;`. At
/// namespace scope under `--device-as-default-execution-space` — which three
/// units pass and their roots are rejected sixteen times without — that is a
/// `__device__` variable, and `&__global__` is not a constant initialiser for
/// one. The instrument would have been red on every FlashInfer unit for a
/// reason no row could fix. `static_assert(::std::is_same_v<…>)` is the same
/// comparison in an unevaluated operand.
///
/// The consequence for these tests is that the refusal now names the ROW
/// rather than the template: the message is the emitter's own,
/// `"quant::bf16_to_fp16: operand 0 \`in_bf16\` is not \`…\`"`. That is a
/// better string to demand — a symbol is greppable and a template name is
/// shared by every instantiation of it.
#[cfg(feature = "_cuda")]
mod jit {
    use std::ffi::{CStr, CString};

    use cudarc::nvrtc::sys as nv;
    use kernels::Ty;
    use kernels_cuda_new::device::DeviceKernel;
    use kernels_cuda_new::runtime::{cache, nvrtc};
    use kernels_cuda_new::{abi, source};

    use super::{MUTANTS, SEVEN, device_row, emit_one, moves_only_its_own_lines, only_in};

    /// `quant::bf16_to_fp16` compiles as stated and does NOT compile
    /// mis-typed.
    ///
    /// This is the row the whole change exists for. Its `__global__` is
    /// `bf16_to_narrow<T>(const bf16* in, T* out, long long n)` and the row's
    /// `T` is `device::f16`, so `in_bf16: Buf` — what it said before —
    /// emitted `const f16*`. Two mutants, because there are two ways to be
    /// wrong: `Buf` (the historical spelling, which reads the format off
    /// `elem`) and `F16s` (the new vocabulary used incorrectly). Both must be
    /// refused.
    #[test]
    fn bf16_where_the_kernel_takes_bf16() {
        run("quant::bf16_to_fp16");
    }

    /// `quant::cast_f16_to_bf16` compiles as stated and does NOT compile with
    /// `src` spelled `Bf16s` or left as `Buf`.
    ///
    /// The mirror image of the row above: `cast_f16_to<T>` fixes the SOURCE
    /// at `const f16*` and templates the destination, and this row's `elem`
    /// is `device::bf16`. So the two mutants are the same two mistakes with
    /// the formats exchanged.
    ///
    /// Fixing one end is what makes this pair irreplaceable by
    /// `tests/units.rs`'s `norm::tanh_*` proof, whose `tanh_inplace<T>`
    /// templates the only pointer it takes. A checker that simply spelled
    /// every buffer from `elem` would pass there and fail here, and here is
    /// where the tree's rows actually live.
    #[test]
    fn f16_where_the_kernel_takes_f16() {
        run("quant::cast_f16_to_bf16");
    }

    /// `quant::cast_bf16_to_int8_per_channel` compiles as stated and does NOT
    /// compile with `out` spelled `U8sMut`.
    ///
    /// The int8 half, and the reason it is not decoration: `u8` and `i8` are
    /// the same WIDTH, so nothing about a launch would have gone wrong and
    /// nothing about a fire would have reported it. The kernel writes
    /// `int8_sym::store`, whose destination is `i8*`, and a row that said
    /// unsigned was a row outside the check. Nothing else in this tree
    /// compiles that distinction.
    #[test]
    fn int8_where_the_kernel_stores_int8() {
        run("quant::cast_bf16_to_int8_per_channel");
    }

    /// Every `quant` row that uses one of the three new kinds, in ONE
    /// translation unit.
    ///
    /// The per-row tests prove the sharpness; this proves the COVERAGE — all
    /// seven inside `emit_device_typecheck`'s net at once, which is the state
    /// the change was made to reach, and a single compile is what says so
    /// about the set rather than about a sample of it.
    ///
    /// The spelling counts come first and are exact. A translation unit with
    /// no assertions in it compiles exactly like one that checks everything,
    /// so a green compile below is worth only as much as the count above it.
    #[test]
    fn all_seven_typecheck_together() {
        let Some(arch) = arch() else { return skip("NVRTC is not loadable") };
        let rows: Vec<DeviceKernel> = SEVEN
            .iter()
            .map(|s| {
                let r = device_row(s);
                DeviceKernel { sig: r.sig, template_path: r.template_path, elem: r.elem }
            })
            .collect();
        let tu = abi::emit_device_typecheck(&rows).expect("the seven rows emit");
        // The trailing `>` is load-bearing: the emitter writes one
        // `is_same_v<check_X::at<n>, TYPE>` per operand AND names the same
        // type in that assertion's message, so a bare substring counts each
        // operand twice. `TYPE>` closes the `is_same_v` and appears once.
        assert_eq!(
            tu.matches("const ::pie_cuda_driver::kernels::device::bf16*>").count(),
            6,
            "six `Bf16s` operands across the seven rows:\n{tu}"
        );
        assert_eq!(tu.matches("::std::int8_t*>").count(), 2, "two `I8sMut` operands:\n{tu}");
        assert_eq!(
            tu.matches("const ::pie_cuda_driver::kernels::device::f16*>").count(),
            1,
            "one `F16s` operand:\n{tu}"
        );
        if let Err(e) = compile(arch, &tu) {
            panic!("the seven rows do not typecheck together:\n{e}\n\n{tu}");
        }
    }

    /// The widened position, through NVRTC, on the row that carries it.
    ///
    /// `attn::pack_structured_mask`'s `masks` operand is the one position in
    /// the tree that `emit_device_typecheck` could not assert: `Ty::cpp()`
    /// named `attn::StructuredMaskParams`, a namespace whose header is
    /// deleted, and `abi::self_describing` declined the kind rather than emit
    /// a spelling that resolves to nothing.
    ///
    /// The control is what makes that a fact instead of a story. The
    /// corrected spelling, `attn::device::StructuredMaskParams`, is handed to
    /// NVRTC against the header that declares it — so the compile is a
    /// statement that the type is REACHABLE under the JIT's own header set,
    /// which is the only compiler that will ever see it. The mutant is then
    /// the statement that the position discriminates.
    ///
    /// The old spelling is compiled by the sibling below rather than here:
    /// `Ty::cpp()` is the only thing that ever produced it and it no longer
    /// does, so putting the literal back into a mutant sig would be
    /// compiling this test's own string through the emitter. The sibling
    /// compiles it as what it is — a name — and
    /// `the_structured_mask_struct_is_declared_where_cpp_says_it_is` makes
    /// the same claim against the header text at build time.
    #[test]
    fn the_structured_mask_operand_is_asserted_and_discriminates() {
        run("attn::pack_structured_mask");
    }

    /// The 269 widened DESTINATION positions, through NVRTC, on one of them.
    ///
    /// `norm::tanh_bf16`'s `x` is `*mut bf16`, which `x::abi` tagged
    /// [`Ty::BufMut`] until the two `ptr_abi!` lines moved. `Ty::BufMut.cpp()`
    /// is `void*`, and `abi::self_describing` declines it — not because the
    /// spelling is wrong but because **every object pointer converts to
    /// `void*`, so the assertion holds for every possible kernel**. That is
    /// the same subject as the sibling above, arrived at from the opposite
    /// direction: `Ty::StructuredMasks` named a type NVRTC could not resolve,
    /// and `Ty::BufMut` named one it could never reject.
    ///
    /// So the control is the load-bearing half here. It says the new tag's
    /// `::pie_cuda_driver::kernels::device::bf16*` is a type that RESOLVES
    /// under this unit's header set and matches what `tanh_inplace<bf16>`
    /// actually takes — which is the claim 269 positions across 172 rows now
    /// rest on, and which nothing compiled while the tag was `BufMut`. The
    /// mutant is then the statement that the position discriminates, and
    /// `norm::tanh_f16` is the same `__global__` one instantiation over, so
    /// the refusal is about this row and not about the template.
    #[test]
    fn a_written_bf16_destination_is_asserted_and_discriminates() {
        run("norm::tanh_bf16");
    }

    /// The header-era namespace does not resolve, and the corrected one does.
    ///
    /// # The sharpest form this claim has
    ///
    /// Two translation units differing in ONE line: an alias for
    /// `const attn::StructuredMaskParams*` and an alias for
    /// `const attn::device::StructuredMaskParams*`. Same header, same
    /// options, same everything else. So the refusal is a fact about the
    /// name and cannot be a fact about the file.
    ///
    /// This is the assertion `Ty::StructuredMasks` needed and never had. It
    /// is worth stating separately from the row compile because the two fail
    /// for different reasons and a reader must be able to tell them apart: a
    /// red row compile means the ROW drifted from the `__global__`, and a red
    /// line here means the TYPE moved out from under the tag.
    ///
    /// # Which side is the emitter's
    ///
    /// The corrected alias is built FROM `Ty::StructuredMasks.cpp()`, so this
    /// compiles the emitter's output and not a copy of it. The historical
    /// alias is a literal, and has to be: nothing in the tree produces that
    /// string any more, which is precisely the change being witnessed.
    ///
    /// Before that change, the row compile above was red for THIS reason —
    /// `Elem::Resolve` has always spelled a `Ty::StructuredMasks` operand
    /// with `Ty::cpp()`, so the dead namespace would have reached NVRTC the
    /// moment the row did. It never did, because `self_describing` declined
    /// the kind under `Elem::Opaque` and no test ever handed the row to
    /// `Elem::Resolve`.
    #[test]
    fn the_header_era_namespace_does_not_resolve_and_the_corrected_one_does() {
        let Some(arch) = arch() else { return skip("NVRTC is not loadable") };
        const HISTORICAL: &str = "const ::pie_cuda_driver::kernels::attn::StructuredMaskParams*";
        let corrected = Ty::StructuredMasks.cpp();
        assert_ne!(
            corrected, HISTORICAL,
            "`Ty::cpp()` spells the header-era namespace again, so the two sides of \
             this compile are one side"
        );
        assert!(
            !corrected.contains("attn::StructuredMaskParams"),
            "the corrected spelling CONTAINS the historical one, so a compiler that \
             accepted one would say nothing about the other: `{corrected}`"
        );

        let tu = |alias: &str| {
            format!(
                "// A name, and nothing else. If this file fails to compile it is \
                 because\n// the name does not resolve.\n\
                 #include \"attn/pack_dense_mask.cuh\"\n\
                 using pie_masks_probe = {alias};\n"
            )
        };
        let (good, bad) = (tu(corrected), tu(HISTORICAL));
        // The claim, stated before the compiler is asked: ONE line differs.
        assert_eq!(
            (only_in(&good, &bad).len(), only_in(&bad, &good).len()),
            (1, 1),
            "the two probes differ in more than the alias, so a refusal below could \
             be about something else:\n{good}\n---\n{bad}"
        );

        if let Err(e) = compile(arch, &good) {
            panic!(
                "`Ty::StructuredMasks.cpp()` names no type NVRTC can find, which is \
                 the exact defect this spelling was corrected to fix:\n{e}\n\n{good}"
            );
        }
        let refused = compile(arch, &bad).expect_err(
            "`attn::StructuredMaskParams` resolves, so the header-era namespace is \
             back and `Ty::cpp()` had no defect to correct -- or this probe is \
             compiling something other than the name",
        );
        assert!(
            refused.contains("StructuredMaskParams"),
            "the historical spelling was refused for a reason that does not name the \
             type, so this is a generic compile failure rather than the one claimed. \
             The two probes differ in one line and the other one compiled, so a \
             refusal about the FILE would have taken both:\n{refused}"
        );
    }

    /// One row's control, then every mutant of it.
    ///
    /// The control is FIRST and its failure is worded as such: until the row
    /// as stated compiles, every red the mutants could report would be the
    /// instrument's rather than a row's, and a negative control that fails
    /// for its own reasons is the decoy this whole file is here to rule out.
    fn run(symbol: &'static str) {
        let Some(arch) = arch() else { return skip("NVRTC is not loadable") };
        let row = device_row(symbol);

        let stated = emit_one(row.sig, row);
        if let Err(e) = compile(arch, &stated) {
            panic!("`{symbol}` as stated does not compile:\n{e}\n\n{stated}");
        }

        let mine: Vec<_> = MUTANTS.iter().filter(|m| m.real == symbol).collect();
        assert!(
            !mine.is_empty(),
            "`{symbol}` has no mutant, so this test compiles a control and asserts \
             nothing about what the control would have caught"
        );
        for m in mine {
            // The claim this compile is about, stated before the compiler is
            // asked: `tu` differs from `stated` in exactly this operand's two
            // assertion lines. `stated` compiled immediately above, so a
            // refusal below can only be those lines -- which is what turns
            // "it did not compile" into evidence.
            moves_only_its_own_lines(m, row);
            let tu = emit_one(m.sig, row);
            let Err(e) = compile(arch, &tu) else {
                panic!(
                    "the mistyped `{}` COMPILED -- the typecheck no longer \
                     distinguishes `{}` from `{}` at `{}`, and those are the operand \
                     types this test exists for:\n\n{tu}",
                    m.tag, m.stated_cpp, m.drifted_cpp, m.operand
                );
            };
            // NOT merely "it failed". The row SYMBOL, because that is what
            // the emitter puts in the assertion's message and it is unique;
            // deliberately not the operand name, which is `out` on one of
            // these and would match half of any diagnostic ever written.
            assert!(
                e.contains(m.real)
                    || e.contains("static assert")
                    || e.contains("static_assert"),
                "`{}` failed, but not on the assertion this test is about -- a compile \
                 that breaks for any other reason satisfies a `must not compile` test \
                 while proving nothing:\n{e}",
                m.tag
            );
        }
    }

    /// Hand one generated translation unit to NVRTC, against the header set
    /// the binary carries.
    ///
    /// No scratch directory, no file, no subprocess and no recycled-pid
    /// story: `nvrtcCreateProgram` takes the source as a pointer and the
    /// headers as two arrays, so two concurrent tests cannot read each
    /// other's artefacts because there are none. That is the whole of what
    /// the nvcc harness's `scratch()` and its `Once` were arranging.
    fn compile(arch: &str, tu: &str) -> Result<(), String> {
        let (texts, names) = source::as_nvrtc_arrays(source::DEVICE_HEADERS)?;
        let text_ptrs: Vec<*const i8> = texts.iter().map(|c| c.as_ptr()).collect();
        let name_ptrs: Vec<*const i8> = names.iter().map(|c| c.as_ptr()).collect();
        let src = CString::new(tu).map_err(|_| "the emitted TU contains a NUL".to_string())?;
        let name = c"device_typecheck.cu".to_owned();

        let mut program: nv::nvrtcProgram = std::ptr::null_mut();
        // SAFETY: every pointer outlives the call, and the arrays are the
        // length passed with them.
        let code = unsafe {
            nv::nvrtcCreateProgram(
                &raw mut program,
                src.as_ptr(),
                name.as_ptr(),
                i32::try_from(text_ptrs.len()).unwrap(),
                text_ptrs.as_ptr(),
                name_ptrs.as_ptr(),
            )
        };
        if code != nv::nvrtcResult::NVRTC_SUCCESS {
            return Err(format!("nvrtcCreateProgram: {code:?}"));
        }

        // `runtime::nvrtc::options`' list, spelled here because that function
        // is private and takes a `Unit`. The float flags do not affect a
        // translation unit made of `static_assert`s, and they are passed
        // anyway: a TU compiled under different flags from the ones the JIT
        // uses is a TU answering a slightly different question, and this file
        // exists because that kind of gap is invisible.
        let options: Vec<CString> = vec![
            CString::new(format!("--gpu-architecture={arch}")).expect("no NUL in an arch"),
            c"-std=c++17".to_owned(),
            c"--fmad=false".to_owned(),
            c"--prec-div=true".to_owned(),
            c"--prec-sqrt=true".to_owned(),
        ];
        let option_ptrs: Vec<*const i8> = options.iter().map(|c| c.as_ptr()).collect();

        // SAFETY: `program` is live; the options outlive the call.
        let code = unsafe {
            nv::nvrtcCompileProgram(
                program,
                i32::try_from(option_ptrs.len()).unwrap(),
                option_ptrs.as_ptr(),
            )
        };
        let log = {
            let mut size = 0;
            // SAFETY: `program` is live and `size` is a live out-parameter.
            unsafe { nv::nvrtcGetProgramLogSize(program, &raw mut size) };
            let mut buffer = vec![0u8; size.max(1)];
            // SAFETY: the buffer is the size NVRTC just asked for.
            unsafe { nv::nvrtcGetProgramLog(program, buffer.as_mut_ptr().cast()) };
            CStr::from_bytes_until_nul(&buffer)
                .map_or_else(|_| String::new(), |s| s.to_string_lossy().into_owned())
        };
        // SAFETY: destroyed exactly once, after the log has been copied out.
        unsafe { nv::nvrtcDestroyProgram(&raw mut program) };

        if code == nv::nvrtcResult::NVRTC_SUCCESS {
            return Ok(());
        }
        // A failure with an empty log would be reported as a pass by any
        // `contains` below, so it is turned into a sentence rather than left
        // as one.
        Err(if log.trim().is_empty() {
            format!("nvrtcCompileProgram: {code:?}, and NVRTC offered no log")
        } else {
            log
        })
    }

    /// The architecture these six compile for.
    ///
    /// The device's own when there is one, because a `__global__`'s parameter
    /// list can sit behind `#if __CUDA_ARCH__` and a typecheck under the
    /// wrong arch would then be checking a signature this box never sees.
    ///
    /// **`sm_89` when there is not**, which is what the nvcc harness passed
    /// and is deliberate rather than a fallback of convenience: NVRTC does
    /// not need a device to compile, and gating on one would run these tests
    /// on strictly FEWER machines than the nvcc they replace — a build box
    /// with a toolkit and no GPU used to run them and would stop. A port that
    /// narrows where a check runs is the same defect as a port that narrows
    /// what it checks.
    ///
    /// `None` only when NVRTC itself cannot answer its version, which is the
    /// one condition under which nothing here can run.
    fn arch() -> Option<&'static str> {
        nvrtc::version().ok()?;
        Some(cache::arch().unwrap_or("sm_89"))
    }

    /// Say why a compile did not happen, on the real stderr.
    fn skip(why: &str) {
        eprintln!("SKIPPED device_typecheck_types::jit: {why}");
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

/// `norm::tanh_bf16` claiming its SIBLING's destination format.
///
/// # This is the one of the pair that survived
///
/// There were two synthetic rows here. The other, `TANH_WRITES_BF16`, stated
/// `x: Bf16sMut` — "the same row as `ptr_abi!` would state it with the
/// written kind" — and it was a SCAFFOLD, standing in for a row the tag
/// prevented. `x::abi` now tags `*mut bf16` [`Ty::Bf16sMut`], so
/// `unit::rows()`'s own `norm::tanh_bf16` states exactly that text and the
/// scaffold became a second population of one: the shape where a fixture
/// outlives the gap it stood in for and can drift from the thing it copied
/// without anything noticing. It is deleted; `device_row("norm::tanh_bf16")`
/// is what its users read now.
///
/// This one is not a scaffold and could not be deleted with it. **A mutant is
/// by definition not a real row** — no row states this, and if one did, this
/// file would be asserting that a correct row is refused. It stays synthetic,
/// and it moved into `MUTANTS`, which is a promotion: it now gets
/// `moves_only_its_own_lines`'s position check (`::at<0>`) and `jit`'s
/// control-then-mutant compile, neither of which the hand-written diff it
/// used to live in performed.
///
/// `norm::tanh_f16` is the same `__global__` at `T = device::f16`, so `f16*`
/// is a type this template really has — one instantiation over. That is what
/// makes the refusal a fact about this row rather than about the C++.
static TANH_BF16_CLAIMS_F16: KernelSig = kernel!(tanh_inplace "norm::tanh_bf16",
    file = Some("norm/altup_aux.cuh"),
    launch = LaunchRule::Unstated,
    operands = operands![
        x: F16sMut <- Source::Unbound,
        n: I32 <- Source::Unbound,
    ]);

/// `attn::pack_structured_mask` with its `masks` operand claiming a `u32*`.
///
/// # Why this row is here at all
///
/// `Ty::StructuredMasks` is the one kind in the tree whose `cpp()` named a
/// namespace that no longer exists: `attn::StructuredMaskParams`, where
/// `attn` was `attn/pack_dense_mask.hpp`'s. The header is deleted and the
/// struct survives one namespace deeper, in `attn::device`. Nothing failed,
/// because `abi::self_describing` declined the kind and the spelling was
/// therefore never handed to a compiler — the decline was what kept it quiet.
///
/// So this mutant is the witness for a WIDENING rather than for a retyping:
/// the other five are about telling two real types apart, and this one is
/// about a position that was not asserted at all until the spelling was
/// corrected. A position that is not asserted and a position that is asserted
/// correctly produce the same green, which is why the red is written down.
///
/// # `Ty::U32s` and not something absurd
///
/// Three of this row's seven operands really ARE `const u32*`, so the mutant
/// claims a type the row legitimately has three positions over. It cannot
/// pass by the claimed type being unspellable, and it cannot fail for a
/// reason that is not about position 4.
static MASKS_AS_U32S: KernelSig = kernel!(pack_structured_mask "attn::pack_structured_mask",
    file = Some("attn/pack_dense_mask.cuh"),
    launch = LaunchRule::Unstated,
    operands = operands![
        positions: U32s <- Source::Unbound,
        klen: U32s <- Source::Unbound,
        qo_indptr: U32s <- Source::Unbound,
        mask_indptr: I32s <- Source::Unbound,
        masks: U32s <- Source::Unbound,
        packed: U8sMut <- Source::Unbound,
        b: I32 <- Source::Unbound,
    ]);

/// `quant::bf16_to_fp16` with its destination stated as a cuBLAS handle.
///
/// `out` and not `n` on purpose. `Ty::BufMut` is `void*` and
/// `Ty::CublasHandle` is `*mut c_void` at the ABI — `Ty::rust()` gives them
/// the same string — so this is the mistake a row author could actually make,
/// a handle written into the pointer slot it is indistinguishable from once
/// marshalled. A mutant that claimed something absurd would be refused for
/// being absurd.
static OUT_AS_CUBLAS_HANDLE: KernelSig = kernel!(bf16_to_fp16 "quant::bf16_to_fp16",
    file = Some("quant/dequant_wna16.cuh"),
    launch = LaunchRule::Unstated,
    operands = operands![
        in_bf16: Bf16s <- Source::Unbound,
        out: CublasHandle <- Source::Unbound,
        n: I64 <- Source::Unbound,
    ]);

/// The same row stating a stream in the same slot.
///
/// The control for the control: `Ty::Stream` has been refused by name for as
/// long as the guard has existed, so this side must stay red however the
/// handle side moves. It is here so the two messages can be compared — a
/// guard that refused both with ONE sentence would satisfy every red and say
/// nothing about which kind it found.
static OUT_AS_STREAM: KernelSig = kernel!(bf16_to_fp16 "quant::bf16_to_fp16",
    file = Some("quant/dequant_wna16.cuh"),
    launch = LaunchRule::Unstated,
    operands = operands![
        in_bf16: Bf16s <- Source::Unbound,
        out: Stream <- Source::Unbound,
        n: I64 <- Source::Unbound,
    ]);

/// The same row stating a `Ty::Dtype` in the same slot — the CONTROL.
///
/// Not a handle, and declined by `self_describing` for a different reason:
/// `::pie_cuda_driver::DType` is a host enum a `__global__` does not take.
/// It is here so the widening can be measured as EXACTLY two variants wide.
/// Without it, "the handle mutants are refused" is equally consistent with
/// "every kind `self_describing` declines is now refused", which would have
/// turned eleven declines into build failures and is not the change.
static OUT_AS_DTYPE: KernelSig = kernel!(bf16_to_fp16 "quant::bf16_to_fp16",
    file = Some("quant/dequant_wna16.cuh"),
    launch = LaunchRule::Unstated,
    operands = operands![
        in_bf16: Bf16s <- Source::Unbound,
        out: Dtype <- Source::Unbound,
        n: I64 <- Source::Unbound,
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

/// One row's typecheck, emitted standalone with its buffer kinds resolved.
///
/// `sig` may be the row's own or a mutant of it; `row` supplies the template
/// path and the element type, so a mutant is emitted against exactly the
/// instantiation its original names. That is the second half of what makes
/// these controls: [`differs_in_exactly_one_operand`] pins the operand list,
/// and this pins everything else the emitter reads.
fn emit_one(sig: &'static KernelSig, row: &'static DeviceKernel) -> String {
    abi::emit_device_typecheck(&[DeviceKernel {
        sig,
        template_path: row.template_path,
        elem: row.elem,
    }])
    .unwrap_or_else(|e| panic!("`{}` does not emit: {e}", sig.symbol))
}

/// The lines of `a` that do not appear anywhere in `b`.
///
/// Set difference on whole lines rather than a positional diff, deliberately.
/// A positional one would report every line after an insertion as changed,
/// which is the observation this file must not accept: *"the texts differ"*
/// is what a mutant proves by existing, and *"they differ HERE and nowhere
/// else"* is what makes the difference a check.
fn only_in<'a>(a: &'a str, b: &str) -> Vec<&'a str> {
    a.lines().filter(|line| !b.lines().any(|other| other == *line)).collect()
}

/// One mutant, and the two C++ spellings that make it one.
///
/// The spellings are LITERALS rather than `Ty::cpp()` calls, for the reason
/// stated at `every_new_kind_renders_in_every_emitter`: the hazard is that
/// two different C++ types have the same width, and a test that derived both
/// sides from the same function would agree with itself while both were
/// wrong.
struct Mutant {
    /// A tag for panics, matching the `.cu` name the nvcc harness used to
    /// write so that a search of the history lands on the same case.
    tag: &'static str,
    /// The real row this mutates. `MUTANTS` is grouped by it.
    real: &'static str,
    /// The mutant's signature.
    sig: &'static KernelSig,
    /// The one operand whose `Ty` differs.
    operand: &'static str,
    /// What the REAL row asserts at that position.
    stated_cpp: &'static str,
    /// What the MUTANT asserts there instead.
    drifted_cpp: &'static str,
}

/// Every mutant in this file, over the four rows that carry the kinds at
/// issue.
///
/// Three of the four are about telling two REAL types apart — the two 16-bit
/// formats, and the sign of a byte. The fourth, `attn::pack_structured_mask`,
/// is about a position that was not asserted at all, which is the harder case
/// to see: two wrong types differ in the emitted text, and an absent
/// assertion differs from a correct one in nothing.
///
/// ONE table for both halves of the proof. The pure-Rust
/// `every_mutant_moves_exactly_its_own_operands_lines` walks it and `jit`'s
/// `run` walks it, so the population that is line-diffed and the population
/// that is compiled cannot drift apart — which is exactly the failure the
/// nvcc version was open to, with each test naming its own mutants inline.
const MUTANTS: &[Mutant] = &[
    // `bf16_to_narrow<T>(const bf16* in, T* out, long long n)`: the SOURCE is
    // fixed at bf16 for every `T`, and this row's `T` is `device::f16`.
    Mutant {
        tag: "bf16_to_fp16.f16s",
        real: "quant::bf16_to_fp16",
        sig: &BF16_TO_FP16_AS_F16,
        operand: "in_bf16",
        stated_cpp: "const ::pie_cuda_driver::kernels::device::bf16*",
        drifted_cpp: "const ::pie_cuda_driver::kernels::device::f16*",
    },
    Mutant {
        tag: "bf16_to_fp16.buf",
        real: "quant::bf16_to_fp16",
        sig: &BF16_TO_FP16_AS_BUF,
        operand: "in_bf16",
        stated_cpp: "const ::pie_cuda_driver::kernels::device::bf16*",
        // Resolved from `elem`, not from the tag -- and identical to the line
        // above. See `the_historical_buf_spelling_is_the_wrong_named_kind`.
        drifted_cpp: "const ::pie_cuda_driver::kernels::device::f16*",
    },
    // `cast_f16_to<T>`: the mirror, fixing the source at f16 while this row's
    // `elem` is `device::bf16`.
    Mutant {
        tag: "cast_f16_to_bf16.bf16s",
        real: "quant::cast_f16_to_bf16",
        sig: &CAST_F16_TO_BF16_AS_BF16,
        operand: "src",
        stated_cpp: "const ::pie_cuda_driver::kernels::device::f16*",
        drifted_cpp: "const ::pie_cuda_driver::kernels::device::bf16*",
    },
    Mutant {
        tag: "cast_f16_to_bf16.buf",
        real: "quant::cast_f16_to_bf16",
        sig: &CAST_F16_TO_BF16_AS_BUF,
        operand: "src",
        stated_cpp: "const ::pie_cuda_driver::kernels::device::f16*",
        drifted_cpp: "const ::pie_cuda_driver::kernels::device::bf16*",
    },
    // The sign confusion, at one width. `::std::int8_t*` is not a substring
    // of `::std::uint8_t*` -- the qualification is what keeps the two
    // `contains` checks from both passing on the same line.
    Mutant {
        tag: "cast_bf16_to_int8.u8",
        real: "quant::cast_bf16_to_int8_per_channel",
        sig: &CAST_BF16_TO_INT8_AS_U8,
        operand: "out",
        stated_cpp: "::std::int8_t*",
        drifted_cpp: "::std::uint8_t*",
    },
    // The fourth row, and the only one whose subject is a WIDENING. See
    // `MASKS_AS_U32S`: this position was not asserted at all until
    // `Ty::StructuredMasks.cpp()` stopped naming a dead namespace, and an
    // unasserted position is indistinguishable from a correct one.
    Mutant {
        tag: "pack_structured_mask.u32s",
        real: "attn::pack_structured_mask",
        sig: &MASKS_AS_U32S,
        operand: "masks",
        stated_cpp: "const ::pie_cuda_driver::kernels::attn::device::StructuredMaskParams*",
        drifted_cpp: "const ::std::uint32_t*",
    },
    // The sixth, and the only one whose operand is a DESTINATION. Same
    // subject as `pack_structured_mask.u32s` -- a position that was not
    // asserted at all -- but at the other end of the parameter and for the
    // other reason: `Ty::StructuredMasks` was declined for naming a dead
    // namespace, and this was declined for naming `void*`, which is not a
    // wrong type but a type that cannot be wrong.
    //
    // Both spellings are QUALIFIED to the last segment, and here that is
    // load-bearing rather than tidy: `device::bf16*` and `device::f16*` are
    // the same WIDTH, so nothing downstream would notice the swap, and the
    // unqualified `f16*` is a substring of `bf16*`. `gone` would then satisfy
    // the `drifted_cpp` test and the diff would agree with itself.
    Mutant {
        tag: "tanh_bf16.f16s",
        real: "norm::tanh_bf16",
        sig: &TANH_BF16_CLAIMS_F16,
        operand: "x",
        stated_cpp: "::pie_cuda_driver::kernels::device::bf16*",
        drifted_cpp: "::pie_cuda_driver::kernels::device::f16*",
    },
];
