//! The launch ABI, generated from the table.
//!
//! A launcher here is an authored C++ function in a nested namespace, with no
//! `extern "C"` anywhere and default arguments on several of them. None of
//! that is callable from outside C++, and the reason is not that it should be
//! — it is that the SIGNATURE was never written down. `KernelSig` carried
//! everything about a kernel except how to call it, and the argument list
//! lived only in the header and, informally, in the model bodies that call it.
//!
//! [`KernelSig::operands`] is that list. Given it, this module emits both
//! halves of a flat ABI:
//!
//! * [`emit_c_shim`] — one `extern "C"` function per row, which includes the
//!   real header and forwards to the real launcher;
//! * [`emit_rust_bindings`] — the matching `unsafe extern "C"` block.
//!
//! ## Why generating it is the proof
//!
//! The shim is not a translation layer that has to be trusted. It CALLS the
//! function it describes, with the header in scope, so C++ overload
//! resolution decides whether the row is right — arity, order, constness and
//! width, all at once, all as compile errors. A hand-written shim would need
//! a golden and a mutation suite to reach a weaker version of the same
//! claim; this needs neither, because a wrong row does not build.
//!
//! It also settles the crate's own invariant, that every symbol in the table
//! resolves to exactly one declaration and every declaration has exactly one
//! row. The first half becomes a tautology here. The second half is still a
//! count, and it is what the pilot found: `rope.hpp` declared twelve
//! launchers against ten rows.
//!
//! ## Scope
//!
//! Rows with an empty operand list are SKIPPED, not emitted as nullary. The
//! table is being filled a family at a time and an unstated row must not be
//! mistaken for a stated one — see [`KernelSig::operands`].
//!
//! ## Where this module lives, and why it followed the rows
//!
//! [`crate::table`] and [`crate::device`], which is where it is now — and the
//! second half of that sentence used to say the opposite. This file was
//! `kernels_cuda::abi` for as long as the archive owned the rows, and it
//! stayed behind when they moved (§19) on the ground that what it emits is
//! the AHEAD-OF-TIME half: an `extern "C"` that forwards into a launcher nvcc
//! compiled, a Rust declaration of that symbol, a typecheck translation unit,
//! and the dispatch arms `driver-cuda` generates against both.
//!
//! That is true of the OUTPUT and was never true of the INPUT. This is a pure
//! function from rows to text: it opens no `.cu`, calls no nvcc, links
//! nothing, and reads exactly the two statics this crate authors. Living in
//! the archive's crate meant a build script that wanted a generated dispatch
//! had to depend on a crate that also builds CMake, and `driver-cuda`'s did —
//! which is one of the three edges §21.5 counted and the only one that had
//! nothing to do with an archive. So the emitter is here, on the LAYER its
//! inputs are on: no feature, no `cudarc`, no toolkit. `kernels-cuda` keeps a
//! re-export so its own shim generator and `driver-cuda`'s tests spell it
//! unchanged.
//!
//! ## A sibling of [`crate::emit`], not a submodule of it
//!
//! They read the same rows to opposite ends, and merging them would produce a
//! generator that has to be told which build it is serving on every call.
//! [`crate::emit`] writes typed Rust over `runtime::fire` — the JIT's direct
//! surface, for rows NVRTC compiles. This writes C++ and `extern "C"` for
//! rows an archive already holds. Two generators over one table is what keeps
//! a symbol's contract single; a hierarchy between them would claim one is a
//! detail of the other, and neither is.
//!
//! ## What this file may name
//!
//! Anything this crate has, which is a rule that got SHORTER in the move. In
//! `kernels-cuda` this file was `#[path]`-included by a build script, so it
//! could name extern crates only — `crate::` inside an included module
//! resolves against the script, not the library. Nothing includes it here, so
//! [`crate::device`] and [`crate::table`] are spelled the way every other
//! module in this crate spells them.
//!
//! Note that the generated TEXT still says `crate::bind::device::ArgValue`
//! and `crate::bind::jit::fire`. Those are strings, and they resolve where
//! the text lands: inside `driver-cuda`.

use crate::device::DeviceKernel;
use kernels::KernelSig;

/// The prefix every generated entry point carries.
///
/// One underscore, not two: `__` anywhere in an identifier is reserved to the
/// implementation in C++, so the obvious `pie_k_rope__rope_bf16` spelling for
/// a `::`-joined path is not ours to use. [`entry_name`] therefore joins with
/// a single `_` and [`emit_c_shim`] rejects a collision rather than letting
/// two rows quietly share an entry point.
pub const PREFIX: &str = "pie_k_";

/// The `extern "C"` name for a row: the prefix, then the symbol with its
/// namespace separators flattened.
pub fn entry_name(symbol: &str) -> String {
    format!("{PREFIX}{}", symbol.replace("::", "_"))
}

/// The rows this module can emit — those that have stated their operands.
fn stated(table: &[&'static [KernelSig]]) -> Vec<&'static KernelSig> {
    table
        .iter()
        .flat_map(|t| t.iter())
        .filter(|k| !k.operands.is_empty())
        .collect()
}

/// The C++ namespace a symbol names, under this crate's root.
fn cpp_path(symbol: &str) -> String {
    format!("::pie_cuda_driver::kernels::{symbol}")
}

/// Emit the `extern "C"` forwarding shims for every stated row in `tables`.
///
/// `includes` are the headers the launchers are declared in, relative to
/// `csrc/src`. They are included by the generated TU, which is what makes
/// compiling it a proof of the rows.
///
/// Errors if two rows would produce the same entry point.
pub fn emit_c_shim(
    tables: &[&'static [KernelSig]],
    includes: &[&str],
    jit: &[&'static crate::device::DeviceKernel],
) -> Result<String, String> {
    // A ROW COMPILED AT RUN TIME HAS NO ENTRY HERE. NEITHER HAS ONE THE
    // DRIVER RUNS ITSELF.
    //
    // The shim exists to let Rust reach a host launcher through a C symbol.
    // A row that NVRTC compiles and `bind::jit::fire` launches has no host
    // launcher to reach, so an entry for it would forward to a `.cu` function
    // that is not there -- which is the link error that used to make deleting
    // one impossible. Skipping it here is what lets the `.cu` go.
    //
    // `crate::execution::RUST_SERVED` is the same mechanism for the other
    // kind of body: a row whose C++ was one `cublasGemmEx` and whose
    // arguments `driver-cuda::bind::service` now assembles in Rust. Skipping
    // it here is what lets the `.cpp` go, and `gemm/gemm.cpp` -- 2,470 lines
    // holding six migrated rows hostage, with not one `__global__` in it --
    // is why the door was needed.
    let rows: Vec<&'static KernelSig> = stated(tables)
        .into_iter()
        .filter(|k| !jit.iter().any(|d| d.sig.symbol == k.symbol))
        .filter(|k| !crate::execution::RUST_SERVED.contains(&k.symbol))
        .collect();
    let mut seen: Vec<(String, &str)> = Vec::new();
    let mut out = String::new();

    out.push_str(
        "// GENERATED by kernels_cuda_new::abi::emit_c_shim -- do not edit.\n\
         //\n\
         // Each body forwards to the launcher its row describes, through a\n\
         // function pointer of the row's exact type. A function-pointer\n\
         // initialisation admits NO parameter conversions, which is why it is\n\
         // written this way rather than as a direct call: a direct call is\n\
         // checked by overload resolution, and overload resolution accepts\n\
         // `void*` where the callee takes `const void*`. That is the one\n\
         // direction a shim can be wrong in and still compile -- a row\n\
         // claiming a read-only operand is written -- and it is exactly the\n\
         // fact the table exists to carry.\n\n\
         #include <cstddef>\n\
         #include <cstdint>\n\
         #include <cstdio>\n\
         #include <cstdlib>\n\
         #include <exception>\n\
         #include <cuda_runtime.h>\n\n\
         // Every body is wrapped, and the reason is worth stating because the\n\
         // wrapper looks like belt-and-braces and is not. A launcher reports a\n\
         // capability it lacks by THROWING -- FlashInfer's dispatch macros end\n\
         // in `throw std::invalid_argument(...)` for an unsupported head dim or\n\
         // GQA group size. An exception crossing the C ABI boundary is\n\
         // undefined behaviour, and what it does in practice is unwind through\n\
         // Rust frames until the runtime aborts -- SIGABRT, no message, and a\n\
         // backtrace pointing at whatever destructor ran last. Twice that cost\n\
         // a debugger session to learn one sentence the launcher had already\n\
         // written down.\n\
         //\n\
         // So the process still dies, because these signatures have nowhere to\n\
         // put a failure -- but it dies SAYING WHY.\n\
         #define PIE_K_GUARD(expr)                                             \\\n\
             try { expr; } catch (const ::std::exception& e) {                 \\\n\
                 ::std::fprintf(stderr, \"[pie_k] %s: %s\\n\", __func__, e.what()); \\\n\
                 ::std::fflush(stderr);                                        \\\n\
                 ::std::abort();                                               \\\n\
             } catch (...) {                                                   \\\n\
                 ::std::fprintf(stderr, \"[pie_k] %s: unknown C++ exception\\n\", \\\n\
                                __func__);                                     \\\n\
                 ::std::fflush(stderr);                                        \\\n\
                 ::std::abort();                                               \\\n\
             }\n\n",
    );
    for inc in includes {
        out.push_str(&format!("#include \"{inc}\"\n"));
    }
    out.push('\n');

    for k in rows {
        let entry = entry_name(k.symbol);
        if let Some((_, other)) = seen.iter().find(|(e, _)| *e == entry) {
            return Err(format!(
                "entry point `{entry}` is claimed by both `{}` and `{other}`",
                k.symbol
            ));
        }
        seen.push((entry.clone(), k.symbol));

        let params = k
            .operands
            .iter()
            .map(|o| format!("    {} {}", o.ty.cpp(), o.name))
            .collect::<Vec<_>>()
            .join(",\n");
        let types = k
            .operands
            .iter()
            .map(|o| o.ty.cpp())
            .collect::<Vec<_>>()
            .join(", ");
        let args = k
            .operands
            .iter()
            .map(|o| o.name)
            .collect::<Vec<_>>()
            .join(", ");

        // A launcher that ANSWERS something keeps its answer: the shim
        // declares the forwarding pointer with the callee's real return
        // type, so a row that said `void` about a `bool` launcher is a
        // conversion C++ refuses rather than a value quietly dropped.
        let ret = if k.returns.is_empty() { "void" } else { k.returns };
        // The guard cannot wrap a `return` (the value would escape the
        // try block), so an answering launcher assigns first.
        let forward = if k.returns.is_empty() {
            format!("    PIE_K_GUARD(fwd({args}))")
        } else {
            format!(
                "    {ret} pie_k_answer{};\n    PIE_K_GUARD(pie_k_answer = fwd({args}))\n    return pie_k_answer;",
                ""
            )
        };
        out.push_str(&format!(
            "extern \"C\" {ret} {entry}(\n{params}) {{\n    \
             static {ret} (*const fwd)({types}) = &{};\n\
             {forward}\n}}\n\n",
            cpp_path(k.symbol),
        ));
    }
    Ok(out)
}

/// Emit the Rust `extern "C"` block matching [`emit_c_shim`].
///
/// Both sides read the same rows, so they cannot disagree with each other;
/// what the shim adds is that they cannot disagree with the C++ either.
pub fn emit_rust_bindings(tables: &[&'static [KernelSig]]) -> String {
    bindings(tables, "emit_rust_bindings", false)
}

/// [`emit_rust_bindings`] restricted to rows ANY crate can declare.
///
/// A row is portable when no operand [`needs_mirror`](kernels::Ty::needs_mirror)
/// — that is, when every argument crosses as a primitive or a raw pointer, so
/// the generated block compiles wherever it is placed. A row naming
/// `KvCacheLayerView` or `HopperPrefillPlan` does not: its declaration only
/// means anything in a module holding that `#[repr(C)]` mirror, which is the
/// shell's.
///
/// The split exists because the mirrors are not the only reason to call a
/// kernel. The loader's `Encode`/`Cast`/`Scale` reach
/// `quant::quantize_bf16_to_*`, `quant::cast_fp32_to_bf16` and
/// `quant::scale_rows_bf16` — rows of plain pointers and `int`s — and it has
/// no business owning an attention workspace layout to say so. Emitting the
/// portable subset here, beside the rows, is what lets it call them from a
/// generated declaration instead of a hand-written one: a row whose signature
/// changes changes this text, and the call site stops compiling.
pub fn emit_rust_bindings_portable(tables: &[&'static [KernelSig]]) -> String {
    bindings(tables, "emit_rust_bindings_portable", true)
}

fn bindings(tables: &[&'static [KernelSig]], by: &str, portable_only: bool) -> String {
    let mut out = format!(
        "// GENERATED by kernels_cuda_new::abi::{by} -- do not edit.\n\
         //\n\
         // The other half of the same rows the C++ shim is generated from.\n\
         // A `nullable` operand is documented, not typed: making it an\n\
         // `Option<NonNull<_>>` here would change the calling convention of\n\
         // an `extern \"C\"` declaration for a fact the callee checks itself.\n\n\
         unsafe extern \"C\" {{\n"
    );
    for k in stated(tables) {
        if portable_only && k.operands.iter().any(|o| o.ty.needs_mirror()) {
            continue;
        }
        // A ROW THE DRIVER SERVES HAS NO ENTRY POINT TO DECLARE.
        //
        // `emit_c_shim` skipped it, so the archive does not define
        // `pie_k_<entry>`, and a declaration of a symbol nothing defines
        // compiles and fails at LINK -- which is the afternoon
        // `driver-cuda/build.rs`'s "a declaration with no definition is only
        // legitimate for a routed row" check exists to prevent. A routed row
        // is exempt there because `bind::jit::fire` is its path; a served row
        // is not exempt and must not be, so the declaration goes instead.
        if crate::execution::RUST_SERVED.contains(&k.symbol) {
            continue;
        }
        out.push_str(&format!("    /// `{}`\n", k.symbol));
        out.push_str(&format!("    pub unsafe fn {}(\n", entry_name(k.symbol)));
        for o in k.operands {
            let note = if o.nullable { "  // may be null" } else { "" };
            out.push_str(&format!("        {}: {},{note}\n", o.name, o.ty.rust()));
        }
        // The Rust side keeps the answer too. `bool` is the only
        // non-void return in the table and it is one byte on both
        // sides, which the layout suite already pins for `Ty::Bool`.
        if k.returns.is_empty() {
            out.push_str("    );\n");
        } else {
            let rust_ret = match k.returns {
                "bool" => "bool",
                other => panic!(
                    "abi: no Rust spelling for the return type `{other}`"
                ),
            };
            out.push_str(&format!("    ) -> {rust_ret};\n"));
        }
    }
    out.push_str("}\n");
    out
}

/// Emit the translation unit that PROVES the rows -- and, in doing so,
/// INSTANTIATES them.
///
/// # One file, two jobs, and why that is not a coincidence
///
/// Taking the address of a function template specialisation is what forces
/// the compiler to instantiate it. So a file that checks every row by taking
/// the address of the instantiation it names is, by construction, a file that
/// emits exactly those kernels into the fatbin -- and the offline build needs
/// no second list. `kernels.def`'s explicit-instantiation macros, its
/// `#include` lists and the CMake regex that reads them all answer the
/// question *"which instantiations exist?"*, which is the question these rows
/// already answer.
///
/// # What replaces the shim's proof
///
/// [`emit_c_shim`] is a proof because it calls the launcher through a
/// function pointer of the row's exact type: an initialisation admits no
/// parameter conversions, so a row that got an operand's constness or width
/// wrong does not build. Tier A deletes the launcher, and the same technique
/// survives it -- a `__global__`'s address is an ordinary host function
/// pointer (that is what `cudaLaunchKernel` receives), so the initialisations
/// below check each row against the kernel with exactly the strictness the
/// shim had.
///
/// What it checks is now MORE than the shim could. A launcher took `void*`
/// because the C ABI was the only thing crossing; a template takes
/// `typename E::storage*`, so a row that named the wrong numeric format is a
/// pointer conversion C++ refuses rather than a reinterpretation it accepts.
///
/// Three things per row are checked at once, and each can be wrong on its
/// own: the template path (no such template), the element type (no such type,
/// or one with no `Elem` specialisation), and the operand list (arity, order,
/// constness, width -- and now the numeric FORMAT, because the two formats
/// are distinct structs rather than two names for `unsigned short`).
///
/// # Errors
///
/// If two rows claim one instantiation, if a row states a return type, or if
/// it still carries a stream operand -- a `__global__` returns void and takes
/// no stream, so either is a row that has not been ported.
pub fn emit_device_typecheck(rows: &[DeviceKernel]) -> Result<String, String> {
    let mut out = String::new();
    out.push_str(
        "// GENERATED by kernels_cuda_new::abi::emit_device_typecheck -- do not edit.\n\
         //\n\
         // One function pointer per row, initialised with the address of the\n\
         // instantiation that row names. Two things follow, and the file\n\
         // exists for both:\n\
         //\n\
         //   * TAKING the address instantiates the template, so this file is\n\
         //     what puts these kernels in the fatbin. The instantiation set\n\
         //     is the ROWS, and there is no second list to keep in step.\n\
         //   * A function-pointer initialisation admits NO parameter\n\
         //     conversions, so a row whose operand list has drifted from the\n\
         //     `__global__` -- in arity, order, constness or width -- does\n\
         //     not compile. That is the property `emit_c_shim` bought with a\n\
         //     forwarding call, kept after the call is gone.\n\
         //\n\
         // The runtime path does not read this file: it hands the same\n\
         // template header to NVRTC and names the same instantiations through\n\
         // `nvrtcAddNameExpression`. This is the OFFLINE half, and its job is\n\
         // to fail the build when a row is wrong rather than the fire.\n\n",
    );

    let mut files: Vec<&str> = Vec::new();
    for k in rows {
        let file = k
            .sig
            .file
            .ok_or_else(|| format!("`{}` states no file to find its template in", k.sig.symbol))?;
        if !files.contains(&file) {
            files.push(file);
        }
    }
    // THE FIXED WIDTHS, and the one place this TU differs from a `.cuh`.
    //
    // `Ty::cpp` spells a byte `::std::uint8_t*` and a signed byte
    // `::std::int8_t*`, because it is SHARED with `emit_c_shim` and a C ABI
    // is what that emitter writes. `pie_device.cuh` deliberately spells the
    // same widths without `<cstdint>` -- "as the COMPILER's own types",
    // because NVRTC has no standard library to take them from -- so a
    // translation unit that includes only `.cuh` files has `device::u8` and
    // no `::std::uint8_t`, and a row with a byte operand failed to compile
    // for a reason that had nothing to do with the row.
    //
    // Measured, on the seven `quant` rows this check was extended over:
    // `namespace "std" has no member "uint8_t"`, five times, in a file whose
    // whole job is to report a DRIFTED ROW. `::std::size_t` resolved in the
    // same file only because nvcc force-includes `cuda_runtime.h`; relying on
    // that is the same accident waiting on a different width, so both are
    // asked for by name here.
    //
    // This is nvcc's TU and never NVRTC's -- the runtime path hands the
    // header to NVRTC and names its instantiations, and does not read this
    // file -- so the standard library IS available, and neither header is
    // shadowed by the five shims in `csrc/src` that make `-I` the wrong flag
    // for this compile.
    out.push_str(
        "// The fixed-width spellings `Ty::cpp` shares with the C shim. See\n\
         // `emit_device_typecheck`: this file is nvcc's, and nvcc has these.\n\
         #include <cstddef>\n#include <cstdint>\n\n",
    );
    out.push_str("// The templates. Nothing else is included, because nothing else is.\n");
    for f in &files {
        out.push_str(&format!("#include \"{f}\"\n"));
    }
    out.push('\n');
    // THE ONE OPERAND WHOSE WIDTH IS NOT STATED BY ITS DECLARATION.
    //
    // `Ty::KvScheme` and `Ty::KvDType` are `enum class … : ::std::uint8_t`,
    // so their crossing is a byte because the C++ says so.
    // `__nv_fp8_interpretation_t` is an UNSCOPED enum with no fixed
    // underlying type (`cuda_fp8.h:185-188`), so its width is the
    // implementation's choice — four bytes everywhere this repo builds, and
    // nowhere promised. `emit::crossing` marshals it as `Crossing::U32`, and
    // a cell one byte wide against a four-byte parameter does not fail: it
    // mis-marshals every argument AFTER it, which is a wrong answer with no
    // diagnostic anywhere.
    //
    // So it is asserted, here, in the only TU that has both the enum and the
    // rows that name it — and only when a row does, so that a tree with no
    // fp8 operand does not acquire a dependency on `<cuda_fp8.h>` through a
    // generated file. The header arrives through the templates above:
    // `attn/kv_paged.cuh` and `attn/attention_naive_paged.cuh` include it.
    if rows.iter().any(|k| k.sig.operands.iter().any(|o| o.ty == kernels::Ty::Fp8Kind)) {
        out.push_str(
            "// `Ty::Fp8Kind` crosses as four bytes. See `kernels::Ty::Fp8Kind`.\n\
             static_assert(sizeof(::__nv_fp8_interpretation_t) == 4,\n    \
             \"__nv_fp8_interpretation_t is not four bytes: `emit::crossing` \
             marshals `Ty::Fp8Kind` as `Crossing::U32`, and a narrower enum \
             shifts every argument after it\");\n\n",
        );
    }
    out.push_str("namespace {\n\n");

    let mut seen: Vec<(String, &str)> = Vec::new();
    for k in rows {
        let inst = k.instantiation();
        if let Some((_, other)) = seen.iter().find(|(i, _)| *i == inst) {
            return Err(format!(
                "instantiation `{inst}` is claimed by both `{}` and `{other}`",
                k.sig.symbol
            ));
        }
        seen.push((inst.clone(), k.sig.symbol));

        if !k.sig.returns.is_empty() {
            return Err(format!(
                "`{}` states a return type (`{}`), and a `__global__` returns void",
                k.sig.symbol, k.sig.returns
            ));
        }
        if let Some(o) = k.sig.operands.iter().find(|o| o.ty == kernels::Ty::Stream) {
            return Err(format!(
                "`{}` takes `{}` as an operand, and a stream is a launch \
                 argument rather than a kernel's",
                k.sig.symbol, o.name
            ));
        }
        // The ELEMENT type, which is the first template argument and only the
        // first. A row may spell a whole argument list — `"device::bf16, 256"`
        // for a `template <class T, int BLOCK>` kernel — because
        // `instantiation()` pastes the string between angle brackets and
        // NVRTC parses C++ rather than a name. This does not: it builds a
        // POINTER type, and `const ::pie_cuda_driver::kernels::device::bf16,
        // 256*` is not one.
        //
        // The consequence of not splitting was silent and is the reason this
        // is written out: eight multi-argument rows generated a translation
        // unit that could not be C++, so `emit_device_typecheck` — the whole
        // point of which is that a drifted row is a BUILD error rather than a
        // failed fire — quietly stopped covering exactly the rows most likely
        // to drift.
        //
        // TAKING THE HEAD IS AN APPROXIMATION, and an earlier draft of this
        // comment claimed otherwise — that a trailing non-type argument has no
        // bearing on an operand's type, so the head "is the whole of what a
        // storage type can be". That holds only while the element type is the
        // FIRST template parameter, and the tree already disagrees: four
        // spellings lead with a non-type — `device::i32(256)` and
        // `device::i32(128)` (a functional cast, so a value), a bare `true`
        // and a bare `8`, and `device::true_type::value` (a static member).
        // Pasted into a pointer declarator those yield
        // `const ::pie_cuda_driver::kernels::device::i32(256)*`, which nvcc
        // rejects inside the GENERATED file — an error naming a line no one
        // wrote, about a row it does not name. So they are refused HERE, by
        // name, with the row that caused it. Refusing is not a gap in the
        // check: a row whose storage type is not its head cannot be spelled by
        // this emitter at all, and saying so is the honest result.
        let element = k.elem.split(',').next().unwrap_or(k.elem).trim();
        if element.is_empty() {
            return Err(format!(
                "`{}` states an element type of `{}`, whose first template \
                 argument is empty -- a row's `elem` may carry a list, but its \
                 head is the storage type every buffer operand is spelled in",
                k.sig.symbol, k.elem
            ));
        }
        if element.starts_with(|c: char| c.is_ascii_digit())
            || element == "true"
            || element == "false"
            || element.contains('(')
            || element.ends_with("::value")
        {
            return Err(format!(
                "`{}` states an element type of `{}`, whose head `{element}` is \
                 a VALUE rather than a type -- this emitter spells every buffer \
                 operand as a pointer to the head of `elem`, and a non-type \
                 there becomes a pointer declarator nvcc rejects in the \
                 generated file rather than in the row. A kernel whose storage \
                 type is not its first template argument needs an operand type \
                 that carries its own element; it cannot be checked from `elem` \
                 alone",
                k.sig.symbol, k.elem
            ));
        }
        let storage = format!("::pie_cuda_driver::kernels::{element}");
        let types = k
            .sig
            .operands
            .iter()
            .map(|o| device_cpp_ty(o.ty, &storage))
            .collect::<Vec<_>>()
            .join(", ");
        // `::` first so a scoped symbol keeps ONE underscore per separator,
        // then everything else C++ will not take in an identifier. A
        // specialisation suffix is the live case — `rmsnorm_strided_bf16#vec8`
        // emitted `check_..._bf16#vec8`, and a `#` in a declarator is a parse
        // error in the generated file, which is the one place a reader cannot
        // act on it.
        let checker = format!(
            "check_{}",
            k.sig
                .symbol
                .replace("::", "_")
                .replace(|c: char| !c.is_ascii_alphanumeric() && c != '_', "_")
        );
        out.push_str(&format!("// `{}`\n", k.sig.symbol));
        out.push_str(&format!(
            "[[maybe_unused]] void (*const {checker})({types}) = &{inst};\n\n"
        ));
    }
    out.push_str("}  // namespace\n");
    Ok(out)
}

/// How a device row's operand is spelled in C++, given the element type its
/// instantiation names.
///
/// The buffer kinds resolve to the ELEMENT type rather than to `void*`, which
/// is the difference between this and [`kernels::Ty::cpp`]. A launcher took
/// `void*` because the C ABI was all that crossed; a template takes
/// `typename E::storage*`, and spelling it is what makes a row that named the
/// wrong numeric format a compile error.
///
/// Two kinds and only two read `storage`, and the rest is not an oversight.
/// A kernel may FIX one operand's format while templating another —
/// `bf16_to_narrow<T>` takes `const bf16*` for every `T`, `cast_f16_to<T>`
/// takes `const f16*` — and for those the row's own `elem` is the wrong
/// answer. [`kernels::Ty::Bf16s`] and [`kernels::Ty::F16s`] name the format
/// outright and therefore come through `cpp()` unchanged, exactly as
/// [`kernels::Ty::F32s`] and [`kernels::Ty::U8s`] already did for the fixed
/// ends a `float` or a byte could name.
fn device_cpp_ty(ty: kernels::Ty, storage: &str) -> String {
    match ty {
        kernels::Ty::Buf => format!("const {storage}*"),
        kernels::Ty::BufMut => format!("{storage}*"),
        other => other.cpp().to_string(),
    }
}


/// One record whose Rust mirror claims to have the C++ type's layout.
///
/// The offsets are the RUST side's, read off the mirror with `offset_of!`.
/// Nothing here is trusted: they are baked into a generated C++ file as
/// `static_assert`s and checked where both layouts are known. Neither side
/// states a number a human wrote down, so there is nothing to keep in sync.
pub struct Record {
    /// The fully-qualified C++ type.
    pub cpp: &'static str,
    /// `size_of` the Rust mirror.
    pub size: usize,
    /// `align_of` the Rust mirror.
    pub align: usize,
    /// Every field, in declaration order, with the Rust mirror's offset.
    pub fields: Vec<(&'static str, usize)>,
}

/// Build a [`Record`] from a `#[repr(C)]` mirror and the C++ type it claims.
///
/// The field list is written once, and both the offsets and the member-count
/// check are derived from it.
///
/// `$crate` is THIS crate now, so the expansion names
/// `kernels_cuda_new::abi::Record`. `kernels-cuda` re-exports the macro under
/// the name its callers already spell — `driver-cuda`'s `launch_abi` suite
/// writes `kernels_cuda::record!` in five places — and the expansion resolves
/// there because every one of those crates already depends on this one.
#[macro_export]
macro_rules! record {
    ($rust:ty => $cpp:literal { $($field:ident),* $(,)? }) => {
        $crate::abi::Record {
            cpp: $cpp,
            size: ::core::mem::size_of::<$rust>(),
            align: ::core::mem::align_of::<$rust>(),
            fields: ::std::vec![
                $((stringify!($field), ::core::mem::offset_of!($rust, $field))),*
            ],
        }
    };
}

/// Emit a translation unit that asserts each mirror's layout against the C++.
///
/// Three claims per record, and it takes all three to be exhaustive:
///
/// * `sizeof` and `alignof` — the shape of the whole;
/// * every field's `offsetof` — that each member is where the mirror puts it;
/// * a structured binding over the whole record — that there are no OTHER
///   members. This one is not decoration. A field added to the end of the C++
///   would land in the tail padding an aligned record already has, leave
///   `sizeof` unchanged, and disturb no existing offset, so the first two
///   claims would both still hold while the mirror silently stopped
///   describing the type. A structured binding names members positionally and
///   the count must match exactly, which is the only standard C++20 way to
///   say "and that is all of them".
pub fn emit_layout_assertions(records: &[Record], includes: &[&str]) -> String {
    let mut out = String::from(
        "// GENERATED by kernels_cuda_new::abi::emit_layout_assertions -- do not edit.\n\
         //\n\
         // The offsets are what the RUST mirrors compute for themselves. They\n\
         // are checked here because this is the only place both layouts are\n\
         // known; a mirror that disagrees does not compile.\n\n\
         #include <cstddef>\n\
         #include <cstdint>\n\n",
    );
    for inc in includes {
        out.push_str(&format!("#include \"{inc}\"\n"));
    }
    out.push('\n');

    for r in records {
        let tag = r.cpp.rsplit("::").next().unwrap_or(r.cpp);
        out.push_str(&format!(
            "static_assert(sizeof({0}) == {1},\n    \
             \"{0}: the mirror is {1} bytes\");\n\
             static_assert(alignof({0}) == {2},\n    \
             \"{0}: the mirror aligns to {2}\");\n",
            r.cpp, r.size, r.align
        ));
        for (name, off) in &r.fields {
            out.push_str(&format!(
                "static_assert(offsetof({0}, {1}) == {2},\n    \
                 \"{0}::{1} is not at {2}\");\n",
                r.cpp, name, off
            ));
        }
        let binds = r
            .fields
            .iter()
            .map(|(n, _)| format!("f_{n}"))
            .collect::<Vec<_>>();
        out.push_str(&format!(
            "\n// Exactly {} members, and no more.\nnamespace {{\ninline void \
             abi_member_count_{tag}() {{\n    {} v{{}};\n    auto& [{}] = v;\n{}}}\n}}\n\n",
            binds.len(),
            r.cpp,
            binds.join(", "),
            binds
                .iter()
                .map(|b| format!("    (void){b};\n"))
                .collect::<String>(),
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernels::Ty;

    /// The two emitters must describe the same operands in the same order, or
    /// the C++ side proves a signature the Rust side does not call.
    #[test]
    fn both_halves_are_emitted_from_the_same_rows() {
        // `crate::table::rope::KERNELS` STOOD HERE. `rope` crossed into
        // fn-world (`.wiki/kernel-x/northstar.md` §5 step 3) and its
        // contracts state no `operands`, so `stated()` returns nothing for
        // it and this test would assert over an empty set — which passes
        // and proves nothing. `norm` is the nearest table that still has
        // ahead-of-time rows.
        //
        // The `RUST_SERVED` filter is `emit_c_shim`'s own, hoisted here
        // because the table it walks now has rows on that list: a row served
        // by a Rust host program has no shim entry BY DESIGN, and asserting
        // one exists would assert the opposite of what the emitter is for.
        let tables: &[&'static [KernelSig]] = &[crate::table::norm::KERNELS];
        let c = emit_c_shim(tables, &["norm/rmsnorm.hpp"], &[]).expect("no collisions");
        let rs = emit_rust_bindings(tables);
        for k in stated(tables)
            .into_iter()
            .filter(|k| !crate::execution::RUST_SERVED.contains(&k.symbol))
        {
            let entry = entry_name(k.symbol);
            assert!(c.contains(&entry), "{entry} missing from the shim");
            assert!(rs.contains(&entry), "{entry} missing from the bindings");
            for o in k.operands {
                assert!(c.contains(o.name), "{entry}: {} missing", o.name);
            }
        }
    }

    /// An unstated row is skipped rather than emitted as a nullary call.
    #[test]
    fn rows_that_have_not_stated_their_operands_are_not_emitted() {
        static UNSTATED: &[KernelSig] = &[kernels::kernel!(nothing "fam::not_yet_stated")];
        let c = emit_c_shim(&[UNSTATED], &[], &crate::device::jit_dispatched()).expect("no collisions");
        assert!(!c.contains("not_yet_stated"));
        assert!(!emit_rust_bindings(&[UNSTATED]).contains("not_yet_stated"));
    }

    /// `::` flattens to a single underscore -- `__` is reserved in C++.
    #[test]
    fn entry_points_avoid_the_reserved_double_underscore() {
        let name = entry_name("rope::rope_bf16");
        assert_eq!(name, "pie_k_rope_rope_bf16");
        assert!(!name.contains("__"));
    }

    /// Which makes a collision possible in principle, so it is refused.
    #[test]
    fn two_rows_may_not_claim_one_entry_point() {
        static CLASH: &[KernelSig] = &[
            kernels::kernel!(a "rope::a_b", operands = kernels::operands![stream: Stream]),
            kernels::kernel!(b "rope_a::b", operands = kernels::operands![stream: Stream]),
        ];
        let err = emit_c_shim(&[CLASH], &[], &crate::device::jit_dispatched()).expect_err("a collision is an error");
        assert!(err.contains("pie_k_rope_a_b"), "{err}");
    }

    /// `bool` is one byte in C++ and `int` is four. A binding that renders a
    /// flag as an `int` mis-lays every operand after it, and no compiler on
    /// either side would say so -- which is why `Bool` is its own kind.
    #[test]
    fn a_flag_stays_one_byte_wide_on_both_sides() {
        assert_eq!(Ty::Bool.cpp(), "bool");
        assert_eq!(Ty::Bool.rust(), "bool");
        assert_ne!(Ty::Bool.rust(), Ty::I32.rust());
    }
}

// ── THE GENERATED DISPATCH ─────────────────────────────────────────
//
// [`emit_c_shim`] proved a row describes its launcher. This is what the
// proof was FOR: given the types and the sources, the call itself is
// derivable, and the arm nobody has to write is the arm nobody can
// write wrong.
//
// What comes out is still a `switch`. C++ has no way to call a function
// with a dynamically built argument list without libffi or a trampoline
// per signature, and neither is worth the trade — so the switch stays
// and stops being HAND-written, which is the part that mattered. Adding
// a kernel becomes a row plus its sources; the dispatch regenerates.
//
// A row with any [`Source::Unbound`] operand is SKIPPED, on the same
// rule an unstated row is: a partial binding is not a binding, and a
// generator that filled the gaps with guesses would be the hand-written
// arm again with worse provenance.

use kernels::{Lit, Source};

// ── THE DISPATCH, GENERATED ───────────────────────────────────────────
//
// A C++ twin of this stood here and is gone with the driver it served.
// It emitted a `switch` into `model/declared/execute.hpp`; that tree is
// reference now, not running code, and a generator with no consumer is a
// second definition of the truth waiting to drift from the first.
//
// What it leaves behind is the SHAPE, which was right: one branch per row
// that states where its arguments come from, keyed by the symbol, with
// the arity the sources ask for as part of the match. A row that has not
// stated its sources is absent and belongs to a hand-written arm until it
// does — which is a row's work, not the driver's.
//
// The driver calls the FLAT entry point (`pie_k_*`), not the launcher —
// it is not C++ and cannot be, which is what `entry_name` and the
// compiled shim exist for. The shim earns its place twice over: it is
// the only way across, and compiling it is what PROVES a row, since C++
// overload resolution decides arity, order, constness and width all at
// once. Take it away and the rows become unchecked claims.
//
// `ins`/`outs`/`aux` are not three spans here. A lowered `Launch` hands
// the binder ONE run in stated order (inputs, outputs, then the weights
// the statement names), so a generated branch slices it, and the counts
// come off the op join.

/// An element count, spelled the width the row declares — or `None`,
/// which declines the whole row to a hand-written arm.
fn elem_count(e: &str, ty: kernels::Ty) -> Option<String> {
    Some(match ty {
        kernels::Ty::Usize => e.to_string(),
        kernels::Ty::I32 => format!("i32::try_from({e}).unwrap_or(0)"),
        kernels::Ty::U32 => format!("u32::try_from({e}).unwrap_or(0)"),
        kernels::Ty::I64 => format!("i64::try_from({e}).unwrap_or(0)"),
        // A count in any other width is a row this generator does not
        // understand, and the honest answer to that is the hand-written
        // arm — same rule as an unrecognised literal.
        _ => return None,
    })
}

/// The Rust expression that binds one operand.
///
/// `n_in` names the local holding the input count, so the slicing reads
/// the same in every branch. The casts mirror [`bind_expr`]'s and exist
/// for its reason: a slot is opaque and the entry point's parameter is
/// the row's declared width, so a substitution is a compile error rather
/// than a stride bug.
/// Every leaf a source reaches, for the guard builder.
///
/// THE GUARDS ARE A PRESENCE QUESTION and a nested source's leaves are
/// as present-or-not as a top-level one's: `Mul(&Gdn(..), &Rows)` needs
/// `gdn.is_some()` exactly as `Gdn(..)` does. Matching flatly would emit
/// a branch calling `g_of`'s unwrap on a fire that carries no GDN
/// context, which is the same shape of bug as an arity arm forgotten in
/// a `match` — silent, and only on the families that reach it.
///
/// `Or` IS NOT DESCENDED. Its whole point is that one of its branches
/// may be absent, so a guard demanding either would defeat it; the `Or`
/// arm guards the composed expression instead.
fn for_each_leaf(s: &Source, f: &mut impl FnMut(&Source)) {
    match *s {
        Source::Width(ref a) | Source::Isqrt(ref a) => for_each_leaf(a, f),
        Source::Mul(ref a, ref b)
        | Source::Sub(ref a, ref b)
        | Source::Div(ref a, ref b)
        | Source::Ne(ref a, ref b) => {
            for_each_leaf(a, f);
            for_each_leaf(b, f);
        }
        // THE PROBE COUNTS TOO. It is asked, not demanded, so no
        // positive guard comes of it — but the `per_head_dim` refusal is
        // a NEGATIVE one ("no row mentions it, so require its absence"),
        // and a probe the walk skipped would leave that refusal standing
        // over the very row written to lift it.
        Source::IfPresent(ref p, ref a, ref b) => {
            for_each_leaf(p, f);
            for_each_leaf(a, f);
            for_each_leaf(b, f);
        }
        Source::Or(..) => f(s),
        _ => f(s),
    }
}

/// Whether a kernel MENTIONS a leaf anywhere, `Or` branches included.
///
/// The counterpart to [`any_leaf`], and the difference is the direction
/// of the guard. A POSITIVE guard demands ("this row reads the GDN
/// context, so require one"), and demanding a branch of an `Or` would
/// defeat the `Or`. A NEGATIVE guard refuses ("no row states `aux`, so
/// require the statement carries none"), and there the `Or` branch is
/// exactly what lifts the refusal -- `Or(&In(1), &Aux(0))` is a row
/// saying it can take the join's value, and a walk that skipped the
/// branch would leave the blanket refusal standing over it.
fn mentions(k: &kernels::KernelSig, mut pred: impl FnMut(&Source) -> bool) -> bool {
    fn walk(s: &Source, f: &mut impl FnMut(&Source) -> bool) -> bool {
        if f(s) {
            return true;
        }
        match *s {
            Source::Width(ref a) | Source::Isqrt(ref a) => walk(a, f),
            Source::Mul(ref a, ref b)
            | Source::Div(ref a, ref b)
            | Source::Ne(ref a, ref b)
            | Source::Sub(ref a, ref b)
            | Source::Or(ref a, ref b) => walk(a, f) || walk(b, f),
            Source::IfPresent(ref p, ref a, ref b) => walk(p, f) || walk(a, f) || walk(b, f),
            _ => false,
        }
    }
    k.operands.iter().any(|o| walk(&o.source, &mut pred))
}

/// Whether any of a kernel's sources reaches a leaf the predicate likes.
fn any_leaf(k: &kernels::KernelSig, mut pred: impl FnMut(&Source) -> bool) -> bool {
    let mut hit = false;
    for o in k.operands {
        for_each_leaf(&o.source, &mut |s| hit |= pred(s));
    }
    hit
}

/// The presence test for the first SLOT a source reaches.
///
/// `Or`'s left is not always a bare slot -- `rope_partial`'s kv head
/// count is "the second result's width over the head dim, or zero", and
/// what decides is still whether that second result is there. So the
/// question is asked of the tree rather than of one leaf.
///
/// THE STATEMENT'S ARITY, not the flat run's length. The run is
/// `[in.., out.., weight..]` laid end to end, so `b.args.get(1)` on a
/// one-input statement answers with its OUTPUT -- a row asking "the
/// second input, or the join's" would bind the result as an operand and
/// launch into it.
fn slot_presence(s: &Source) -> Option<String> {
    match *s {
        Source::In(i) => Some(format!("n_in > {i}")),
        Source::Out(i) => Some(format!("n_out > {i}")),
        Source::Weight(i) => Some(format!("b.args.len() > n_in + n_out + {i}")),
        Source::Param(i) | Source::ParamF32(i) => Some(format!("spec.params.len() > {i}")),
        Source::Width(ref a) | Source::Isqrt(ref a) => slot_presence(a),
        Source::Mul(ref a, ref b)
        | Source::Sub(ref a, ref b)
        | Source::Div(ref a, ref b)
        | Source::Ne(ref a, ref b) => slot_presence(a).or_else(|| slot_presence(b)),
        _ => None,
    }
}

/// The [`ArgValue`] variant a `Ty` crosses as, for the JIT dispatch arm.
///
/// `None` for the kinds a JIT'd row does not take: a stream is
/// `cuLaunchKernel`'s sixth parameter and a cuBLAS handle is not a kernel
/// argument at all, so a row naming either has no JIT arm to emit and stays
/// on the shim.
///
/// [`ArgValue`]: driver-cuda's `bind::device::ArgValue`
fn arg_value_variant(ty: kernels::Ty) -> Option<&'static str> {
    Some(match ty {
        kernels::Ty::I32 => "crate::bind::device::ArgValue::I32",
        kernels::Ty::U32 => "crate::bind::device::ArgValue::U32",
        kernels::Ty::F32 => "crate::bind::device::ArgValue::F32",
        kernels::Ty::Usize => "crate::bind::device::ArgValue::Usize",
        // THE TWO THE DRIVER'S ENUM DID NOT HAVE, and which the catch-all
        // below was answering `Ptr` for.
        //
        // `Ty::I64` is every batched SSM row's `slot_stride_elems` and
        // `Ty::Bool` is `moe_norm_topk`, `write_state`, `hnd_layout`. Both
        // are host scalars and neither is an address, so the old `_ => Ptr`
        // was not a missing variant — it was a WRONG one, and the two halves
        // failed differently: `bool as *mut c_void` is a compile error in a
        // generated file, while `i64 as *mut c_void` compiles, launches, and
        // is refused by `Args::bind` at fire time as "declared I64 and was
        // bound a pointer" — a refusal, once per launch, on a device.
        //
        // 26 hosted rows carry one of the two. They are marshalled here
        // because `runtime::args::ArgValue` has had both for some time; what
        // was missing was this driver's twin, which is now added beside them.
        kernels::Ty::I64 => "crate::bind::device::ArgValue::I64",
        kernels::Ty::Bool => "crate::bind::device::ArgValue::Bool",
        // The two by-value enums cross as ONE driver variant, for the reason
        // `runtime::ArgValue::U8` states: a kind says how a value is
        // marshalled, a `Ty` says what it means, and the swap the two `Ty`s
        // exist to catch is caught by `emit_device_typecheck`'s
        // function-pointer initialisation rather than by a marshalling kind.
        kernels::Ty::KvScheme | kernels::Ty::KvDType => "crate::bind::device::ArgValue::U8",
        kernels::Ty::Fp8Kind => "crate::bind::device::ArgValue::U32",
        // AND THE POINTERS, LISTED. This was `_ => Ptr`, which is the same
        // sentence as "everything I have not thought about is an address" —
        // and the two scalars above are what that cost. The list is
        // `runtime::args::is_pointer`'s, which is the side that CHECKS it:
        // a row whose operand is pointer-shaped there and not here loses its
        // arm (loud, at the build gate), and one that is pointer-shaped here
        // and not there is refused by `Args::bind` (loud, at the fire). The
        // duplication is paid because layer 2 may not depend on the runtime.
        kernels::Ty::Buf
        | kernels::Ty::BufMut
        | kernels::Ty::I32s
        | kernels::Ty::I32sMut
        | kernels::Ty::I64s
        | kernels::Ty::U32s
        | kernels::Ty::U32sMut
        | kernels::Ty::U8s
        | kernels::Ty::U8sMut
        | kernels::Ty::U16s
        | kernels::Ty::U16sMut
        | kernels::Ty::I8s
        | kernels::Ty::I8sMut
        | kernels::Ty::Bf16s
        | kernels::Ty::F16s
        | kernels::Ty::F32s
        | kernels::Ty::F32sMut
        | kernels::Ty::BufArray
        | kernels::Ty::BufArrayMut
        | kernels::Ty::BufArrayOut
        | kernels::Ty::BufArrayOutMut
        | kernels::Ty::U8Array
        | kernels::Ty::I32Array => "crate::bind::device::ArgValue::Ptr",
        // EVERYTHING ELSE IS A ROW THAT IS NOT READY, and saying so is the
        // point: a `Stream` belongs to the launch rather than to the argument
        // list, and a by-value view or a `Dtype` enum has no marshalling here
        // at all. `None` skips the JIT branch, `routed_rows_have_an_arm`
        // refuses the symbol at build time, and the row is left where it is.
        _ => return None,
    })
}

fn rust_bind_expr(op: &kernels::Operand) -> Option<String> {
    let e = rust_bind_expr_of(&op.source, op.ty)?;
    // `Lit::Null` and the element counts produce their final form
    // already — they are the two places a source's expression depends on
    // the operand's declared type rather than only on the source.
    if matches!(op.source, Source::Lit(Lit::Null) | Source::OutElements(_) | Source::InElements(_))
    {
        return Some(e);
    }
    cast_for(&e, op.ty)
}

/// Whether a source reads a DRIVER-DECLARED field, so its Rust type is
/// whatever that struct says rather than this grammar's `i32`.
///
/// Every other integer leaf already presents as `i32` by construction:
/// `width_of`/`rows_of` return one, `Param` narrows to one, `PerHeadDim`
/// casts to one. These read a field, and the driver's fields are `u32`
/// where a count cannot be negative and `i32` where a sentinel needs the
/// sign — `ple_dim` is `u32`, `head_dim` is `i32`, and a row naming
/// either says only `I32`.
const fn reads_a_declared_field(source: &Source) -> bool {
    matches!(
        *source,
        Source::Ctx(_)
            | Source::CtxNonZero(_)
            | Source::CtxByLayer(_)
            | Source::Gdn(_)
            | Source::Attn(_)
            | Source::AttnNonZero(_)
            | Source::KvLayerField(_)
            | Source::RequestCount
    )
}

/// One ARITHMETIC operand, in this grammar's integer type.
///
/// [`cast_for`] already states the rule — "a row declares `I32` and gets
/// an i32, whatever the width of the thing it named" — but it runs ONCE,
/// on the whole operand, so a field reached through a `Div` was composed
/// before the rule applied. `width_of(b, 0) / ctx.ple_dim` is `i32 /
/// u32` and does not compile; the row is not wrong, the seam was.
///
/// Narrowing HERE rather than at the leaf keeps the redundant wrap off
/// every top-level field read, where `cast_for` already does this.
///
/// `i32::try_from` and not `as`: a value too wide for the row's declared
/// type is a fact worth saturating on, and `as` would wrap a divisor to
/// something arbitrary. It compiles for an `i32` field too — `TryFrom`
/// is reflexive through `From`, with `Infallible` as the error — so one
/// spelling covers both widths without the emitter knowing which it met.
fn rust_arith_of(source: &Source, ty: kernels::Ty) -> Option<String> {
    let e = rust_bind_expr_of(source, ty)?;
    // ONLY where the grammar's type is an integer. Arithmetic over
    // extents and counts is all this table has, but a float leaf under a
    // `Mul` would be a real row and narrowing it would be silent damage,
    // so the float and pointer types keep the child untouched and fail
    // loudly at the seam instead.
    let integral = matches!(
        ty,
        kernels::Ty::I32
            | kernels::Ty::U32
            | kernels::Ty::I64
            | kernels::Ty::Usize
            | kernels::Ty::Bool
    );
    if integral && reads_a_declared_field(source) {
        return cast_for(&e, kernels::Ty::I32);
    }
    Some(e)
}

/// One SOURCE's expression, without the operand's cast.
///
/// Split from [`rust_bind_expr`] so the grammar's arms can recurse: a
/// `Div` asks its two children for their expressions and composes, and
/// neither child has an operand type of its own.
fn rust_bind_expr_of(source: &Source, ty: kernels::Ty) -> Option<String> {
    let e = match *source {
        Source::Unbound => return None,
        Source::In(i) => format!("b.args[{i}].ptr"),
        Source::Out(i) => format!("b.args[n_in + {i}].ptr"),
        Source::Weight(i) => format!("b.args[n_in + n_out + {i}].ptr"),
        // Resolved ONCE before the match, so the guard below can test it
        // and so a branch does not re-look-up a name per launch.
        Source::WeightNamed => "w_named".to_string(),
        Source::WeightNamed2 => "w_named2".to_string(),
        Source::Param(i) => format!("i32::try_from(spec.params[{i}]).unwrap_or(0)"),
        Source::ParamF32(i) => format!("f32::from_bits(spec.params[{i}])"),
        // Rows times a param — the MoE aligned path's route count, and
        // the one product that is neither an operand's extent nor a
        // load-time number.
        Source::RoutesOfParam(i) => format!(
            "rows.saturating_mul(i32::try_from(spec.params[{i}]).unwrap_or(0))"
        ),
        Source::Rows => "rows".to_string(),
        Source::OutRows(i) => format!("rows_of(b, n_in + {i}, rows)"),
        Source::InRows(i) => format!("rows_of(b, {i}, rows)"),
        Source::OutWidth(i) => format!("width_of(b, n_in + {i})"),
        Source::InWidth(i) => format!("width_of(b, {i})"),
        // An ACCESSOR, not a field: the driver decides whether its
        // per-layer vector falls back, filters or refuses, and the
        // generator's claim is only that the statement's layer is the
        // index.
        Source::CtxByLayer(f) => format!("ctx.{f}(b.layers.start as usize)"),
        // An element COUNT is a `usize` here and the row decides how wide
        // the launcher wants it — some spell `std::size_t`, some `int`.
        // The C++ emitter can cast unconditionally because C++ narrows
        // silently; this one has to ask the row, which is the better of
        // the two behaviours to have been forced into.
        Source::OutElements(i) => {
            elem_count(&format!("elems_of(b, n_in + {i}, rows)"), ty)?
        }
        Source::InElements(i) => {
            elem_count(&format!("elems_of(b, {i}, rows)"), ty)?
        }
        // A DIM is the plan's, not the binder's: an arg carries its row
        // width and nothing about the shape behind it. The join could
        // carry it, and until it does these rows stay hand-written —
        // which is the same rule `Source::Unbound` gets, for the same
        // reason (a partial binding is not a binding).
        Source::InDim(..) | Source::OutDim(..) => return None,
        // The fire's positions, which are TOKEN-ROWED like a value: a
        // rectangle starting partway in would rotate its own rows against
        // another rectangle's positions. Safe under the whole-fire guard
        // below, same as `In`/`Out`.
        Source::Positions => "ctx.positions".to_string(),
        // The enclosing guard's value, when a REGION launch declares no
        // result of its own. `LaunchSpec::outs` is where the join put it
        // -- which it has since `DispatchPlan` learned to map a region op
        // back to its owning guard, and the note that used to sit here
        // ("the Rust join does not carry the guard's value yet") outlived
        // that by some months. Guarded on the join HAVING one, below.
        Source::ResultOrRegion(i) => format!(
            "join_out(spec, {i}, frame, resolver).map_or(core::ptr::null_mut(), |a| a.ptr)"
        ),
        // The join's FOREIGN values -- nemotron's cross-statement mamba
        // wiring. Guarded on the join having collected one, below.
        Source::Aux(i) => format!(
            "join_aux(spec, {i}, frame, resolver).map_or(core::ptr::null_mut(), |a| a.ptr)"
        ),
        // THE BF16 MIRRORS, which is the native alias. Most CUDA
        // launchers take a `KvCacheLayerView` whole, and for a long time
        // this emitter refused the pair on that ground — but the refusal
        // was on the SOURCE when the fact it was really about is the
        // OPERAND'S TYPE. `dispatch_attention_flashinfer_prefill_bf16`
        // takes the two pointers loose, and a row saying so is exactly
        // right; a row saying so where the parameter is a view is what
        // the refusal is for, and that one is a type error the generated
        // file gets caught on.
        Source::KvKeys => format!(
            "kv_view(attn, b.layers.start as usize).k_bf16_pages"
        ),
        Source::KvValues => format!(
            "kv_view(attn, b.layers.start as usize).v_bf16_pages"
        ),
        // The fire's own tables, likewise a Metal spelling: CUDA's launchers
        // take them through a plan cache or a `KvCacheLayerView` rather than
        // as loose pointers.
        Source::TokenIds
        | Source::RequestOfToken
        | Source::KvPageIndices
        | Source::KvPageIndptr
        | Source::AttentionMask
        | Source::AttentionMaskEnabled
        | Source::KvHeadStride
        | Source::KvSeqStride
        | Source::KvPageSize
        | Source::KvWritePage
        | Source::KvWriteOffset
        | Source::RopeFrequencies => return None,
        // NOT in the decline above, and the difference is the launcher's
        // signature rather than the source's kind. The epilogue's gather
        // takes `const int32_t* row_indices` LOOSE — it is the one CUDA
        // launcher that wants a fire table as a bare pointer — so the
        // driver holds the pair on the ctx exactly as it holds
        // `peel_window`, and the row can say so.
        Source::SamplingIndices => "ctx.sampling_indices".to_string(),
        Source::RequestCount => "ctx.sampled_rows".to_string(),
        // BARE HERE, and narrowed one layer up — which is a fact about
        // `rust_arith_of` rather than a constraint on the driver. This arm
        // still cannot know the field's type (the same ignorance `is_set`
        // exists for, below), so it emits the read and nothing else;
        // `rust_arith_of` wraps the result in `i32::try_from(..)
        // .unwrap_or(i32::MAX)` when the GRAMMAR's type is integral, and
        // leaves float and pointer leaves untouched.
        //
        // THIS USED TO BE AN INVARIANT AND IS NOW HISTORY. Because the read
        // was concatenated bare with `width_of(b, i)`, which is always
        // `i32`, the arithmetic had to typecheck against whatever the driver
        // declared — so every integer `DispatchCtx` field a table divided by
        // had to BE `i32`, and nothing in this file could enforce it.
        // Thirteen divisions were emitted, eleven by `head_dim`, one by
        // `altup_streams`, one by `ple_dim`; when `ple_dim` was widened to
        // `u32` to keep a field-shorthand initialiser compiling, the third
        // stopped building while the other twelve stayed green. One red line
        // in a generated file no one reads, from an edit two crates away
        // that never mentioned a kernel — which is what argued the seam into
        // existence, and why the measurement is kept here rather than the
        // rule.
        //
        // The rule is retired, not merely satisfied: any integer field type
        // works now, and `ple_dim` is still `u32`. Reading this as live would
        // be worse than reading nothing — it would send someone to widen a
        // field back in `driver-cuda` to fix a build that is not broken. The
        // narrowing is a CHECKED `try_from` rather than an `as` cast, so a
        // value above `i32::MAX` saturates where a cast would have wrapped,
        // and it is applied by type rather than blanket: `i32::try_from` of
        // an `f32` does not exist, so `rope_theta` and `final_logit_softcap`
        // would have failed to build had it been applied to everything. They
        // appear bare in all seven of their uses, which is the check that
        // this is type-aware rather than lucky.
        Source::Ctx(f) | Source::CtxNonZero(f) => format!("ctx.{f}"),
        // `g_of` is the driver's unwrap, and the guard below is what
        // makes it total: a branch binding a GDN field is emitted with
        // `gdn.is_some()` in its guard, so the only way here is through
        // that test.
        Source::Gdn(f) => format!("g_of(gdn).{f}"),
        // The layer's slab base. Guarded above, so the lookup is total.
        Source::GdnSlab(f) => format!(
            "gdn_slab(gdn, spec.state, \"{f}\").unwrap_or(core::ptr::null_mut())"
        ),
        // Null when the checkpoint ships none, which is a fact about the
        // checkpoint and not drift. See the variant's own doc.
        Source::WeightSuffix(suffix) => format!(
            "spec.weight.as_deref()\n            .and_then(|n| resolver.weight(&format!(\"{{n}}{suffix}\")))\n            .unwrap_or(core::ptr::null())"
        ),
        // Both total by the same construction `g_of` is: the guard below
        // proves the context is there, and `kv_view` also proves the
        // layer is in range.
        Source::Attn(f) | Source::AttnNonZero(f) => format!("a_of(attn).{f}"),
        // THE DRIVER'S RULE STAYS THE DRIVER'S. `window_of` reads the
        // statement's param, then the per-layer vector, then the fire's
        // default; `attn_plan` picks the full-attention plan when the
        // layer's window says FULL. Both are one call here because a row
        // may not spell either.
        Source::AttnWindow => {
            "window_of(spec, a_of(attn), u32::from(b.layers.start))".to_string()
        }
        Source::AttnPlan(family) => {
            format!("attn_plan(a_of(attn), spec, u32::from(b.layers.start), \"{family}\")")
        }
        Source::KvLayerView => "kv_view(attn, b.layers.start as usize)".to_string(),
        Source::KvLayerField(f) => {
            format!("kv_view(attn, b.layers.start as usize).{f}")
        }
        // A NULL is returned fully typed and skips the cast step below:
        // that step turns a slot into the row's pointee, and a null has
        // no slot to turn. `null_mut().cast::<i32>()` leaves the original
        // `T` for an inference with nothing to work from, so the row's
        // own declared type produces the pointer directly.
        Source::Lit(Lit::Null) => {
            let rust = ty.rust();
            return Some(match rust.strip_prefix("*mut ") {
                Some(p) => format!("core::ptr::null_mut::<{p}>()"),
                None => format!("core::ptr::null::<{}>()", rust.strip_prefix("*const ")?),
            });
        }
        // The rest are values, and spelling a value is not parsing one.
        // A short-lived version of this function had a miniature C++
        // literal parser here, because the row held `"1.702f"` — which
        // is what a vocabulary looks like when it speaks one consumer's
        // language. The row holds `Lit::F32(1.702)` now and there is
        // nothing to parse.
        // ── The grammar, recursively ──
        //
        // Each arm asks its children for THEIR expression and composes.
        // Nothing here knows which leaves exist, which is the property
        // that lets a new row compose without an emitter edit.
        // WIDTH IS A PROJECTION ON A SLOT, so it is only meaningful over
        // `In`/`Out`. Anything else is a row asking for the width of a
        // number, and declining is better than inventing one.
        Source::Width(s) => match *s {
            Source::In(i) => format!("width_of(b, {i})"),
            Source::Out(i) => format!("width_of(b, n_in + {i})"),
            _ => return None,
        },
        Source::Mul(a, c) => {
            format!("({}) * ({})", rust_arith_of(a, ty)?, rust_arith_of(c, ty)?)
        }
        Source::Sub(a, c) => format!(
            "({}).saturating_sub({})",
            rust_arith_of(a, ty)?,
            rust_arith_of(c, ty)?
        ),
        Source::Div(a, c) => {
            format!("({}) / ({}).max(1)", rust_arith_of(a, ty)?, rust_arith_of(c, ty)?)
        }
        Source::Isqrt(s) => format!("isqrt_exact_i32({})", rust_arith_of(s, ty)?),
        Source::Ne(a, c) => {
            format!("({}) != ({})", rust_arith_of(a, ty)?, rust_arith_of(c, ty)?)
        }
        // PRESENCE is a per-leaf question, so the probe is asked as a
        // pointer-or-option rather than evaluated: `Or` on two pointer
        // leaves takes the first non-null, and on a slot takes it when
        // the run is long enough.
        Source::Or(a, c) => {
            let present = slot_presence(a)?;
            let left = rust_bind_expr_of(a, ty)?;
            let fallback = rust_bind_expr_of(c, ty)?;
            // `as *mut _` ON THE FALLBACK, and only for a pointer. Both
            // arms of the `if` must agree; an arg's pointer is always
            // `*mut c_void` while the fallback's mutability varies
            // (`w_named` const, `o_out` mut, `lse_out_d` a `*mut f32`),
            // so coercing the other side is the one rule that holds for
            // every pair. A scalar `Or` needs none and would not compile
            // with one.
            // A NULL FALLBACK IS UNTYPED. `Lit::Null` normally spells
            // its own pointee out of the operand's declared type, but
            // here the other arm is an arg's `*mut c_void` and the two
            // must agree — so the inference goes the other way and the
            // operand's cast lands on top as it does for any source.
            let fallback = if matches!(*c, Source::Lit(Lit::Null)) {
                "core::ptr::null_mut()".to_string()
            } else if ty.rust().starts_with('*') {
                format!("{fallback} as *mut _")
            } else {
                fallback
            };
            format!("if {present} {{ {left} }} else {{ {fallback} }}")
        }
        Source::IfPresent(probe, then, other) => match *probe {
            Source::PerHeadDim => format!(
                "if spec.per_head_dim.is_some() {{ {} }} else {{ {} }}",
                rust_bind_expr_of(then, ty)?,
                rust_bind_expr_of(other, ty)?
            ),
            // Only `PerHeadDim` is a probe today. Another would need its
            // own presence test, and guessing one would be worse than
            // declining the row.
            _ => return None,
        },
        Source::PerHeadDim => "spec.per_head_dim.map_or(0, |d| d as i32)".to_string(),
        Source::NamedScale => "named_scale(ctx, spec).unwrap_or(0.0)".to_string(),
        Source::LayerScale => "layer_scale(ctx, spec)".to_string(),
        Source::Beta => "if spec.beta_one { 1.0f32 } else { 0.0f32 }".to_string(),
        Source::RotaryWidth => {
            "rotary_width(ctx, spec, b.layers.start as usize).unwrap_or(0)".to_string()
        }
        Source::Lit(Lit::Bool(v)) => v.to_string(),
        Source::Lit(Lit::F32(v)) => format!("{v}f32"),
        Source::Lit(Lit::I32(v)) => format!("{v}i32"),
    };
    Some(e)
}

/// The cast an operand's TYPE puts on its source expression.
fn cast_for(e: &str, ty: kernels::Ty) -> Option<String> {
    let e = e.to_string();
    Some(match ty {
        kernels::Ty::Buf => format!("({e}).cast_const()"),
        kernels::Ty::BufMut => e,
        kernels::Ty::I32s => format!("({e}).cast_const().cast::<i32>()"),
        kernels::Ty::U32s => format!("({e}).cast_const().cast::<u32>()"),
        kernels::Ty::U8s => format!("({e}).cast_const().cast::<u8>()"),
        kernels::Ty::I64s => format!("({e}).cast_const().cast::<i64>()"),
        kernels::Ty::F32s => format!("({e}).cast_const().cast::<f32>()"),
        kernels::Ty::I32sMut => format!("({e}).cast::<i32>()"),
        kernels::Ty::U32sMut => format!("({e}).cast::<u32>()"),
        kernels::Ty::U8sMut => format!("({e}).cast::<u8>()"),
        kernels::Ty::F32sMut => format!("({e}).cast::<f32>()"),
        kernels::Ty::U16s => format!("({e}).cast_const().cast::<u16>()"),
        kernels::Ty::U16sMut => format!("({e}).cast::<u16>()"),
        kernels::Ty::I8s => format!("({e}).cast_const().cast::<i8>()"),
        kernels::Ty::I8sMut => format!("({e}).cast::<i8>()"),
        // The two sixteen-bit formats cross as their WIDTH, which is what
        // `Ty::rust` spells them as. The format is checked in the C++, by
        // `emit_device_typecheck`, against the instantiation -- a cast here
        // that named a Rust type would be naming one this crate does not have.
        kernels::Ty::Bf16s | kernels::Ty::F16s => {
            format!("({e}).cast_const().cast::<u16>()")
        }
        // An ARRAY OF DEVICE POINTERS: the bank of per-expert addresses
        // the routed GEMVs index. The bind is the same as a scalar
        // pointer's — the arg holds the bank's address — and only the
        // pointee type differs, which is a cast the row already states.
        // Spelled out rather than left to `_` because a `*mut c_void`
        // reaching a `*const *const i32` parameter is a compile error,
        // and the whole reason these casts are here is that the error is
        // better than a stride bug.
        kernels::Ty::BufArray => format!("({e}).cast_const().cast::<*const ::core::ffi::c_void>()"),
        kernels::Ty::BufArrayMut => format!("({e}).cast_const().cast::<*mut ::core::ffi::c_void>()"),
        kernels::Ty::BufArrayOut => format!("({e}).cast::<*const ::core::ffi::c_void>()"),
        kernels::Ty::BufArrayOutMut => format!("({e}).cast::<*mut ::core::ffi::c_void>()"),
        kernels::Ty::U8Array => format!("({e}).cast_const().cast::<*const u8>()"),
        kernels::Ty::I32Array => format!("({e}).cast_const().cast::<*const i32>()"),
        // A ROW DECLARES `I32` AND GETS AN i32, whatever the width of
        // the thing it named. Driver fields are `u32` where a count
        // cannot be negative and `i32` where a sentinel needs the sign,
        // and a row should not have to know which -- `score_window` is
        // `u32`, `window_left` is `i32`, and both are `I32` operands.
        //
        // `i32::MAX` rather than `0` on overflow, because every `I32`
        // operand here is a count or a window and a value too large for
        // an i32 means "effectively unbounded" in both readings. Zero
        // would mean the opposite. The hand arm this replaces chose the
        // same, for the same reason.
        kernels::Ty::I32 => format!("i32::try_from({e}).unwrap_or(i32::MAX)"),
        // A ROW DECLARES AN ENUM AND GETS ITS BYTE. The driver-side source
        // is a `#[repr(u8)]` mirror (`KvCacheScheme`, `DType`), and `as u8`
        // on a fieldless `#[repr(u8)]` enum is its discriminant — the same
        // number the C++ enumerator has, because both are the mirror pair
        // `launch_abi` checks. Spelled out rather than left to `_` because a
        // `KvCacheScheme` reaching a `u8` parameter is a compile error, and
        // that error is the whole reason these casts exist.
        //
        // **This arm is the one an hour of this session was spent on.** A
        // variant that renders in the C shim, the Rust binding, the device
        // type-check and `ArgValue` and NOT here emits a dispatch that does
        // not compile; a variant that renders here and not in one of those
        // fails at link or at launch. Five renderings, all five or none.
        kernels::Ty::KvScheme | kernels::Ty::KvDType => format!("({e}) as u8"),
        // The fp8 interpretation is the sixth rendering of the rule above and
        // was added in one edit across all of them: `Ty::cpp`, `Ty::rust`,
        // `emit::crossing`, `runtime::args`, `bind::device` and here.
        kernels::Ty::Fp8Kind => format!("({e}) as u32"),
        _ => e,
    })
}

/// One `match` arm per row whose operands are fully sourced, for the Rust
/// driver's `dispatch`.
///
/// Emitted as the body of a function returning `bool` — `true` when a
/// branch ran. A symbol with no generated branch, or one whose guard the
/// statement does not satisfy, returns `false` and the caller falls
/// through to the hand-written arm that knows the other spelling. That
/// fallthrough is the whole reason the guards are here rather than in the
/// arm: a generated branch must decline loudly, never guess.
///
/// # The rows in `jit` are the exception, and the exception has no `false`
///
/// A row named there has no shim entry — [`emit_c_shim`] skips it — so there
/// is no hand arm behind it and a fallthrough would be diagnosed as an unknown
/// kernel, which is a lie about what went wrong. Its arm therefore ends in
/// `true` unconditionally, and the function it calls returns `()`: the value
/// that would have to be `false` for the fire to be misrouted does not exist
/// to be written.
pub fn emit_rust_dispatch(
    tables: &[&'static [KernelSig]],
    jit: &[&'static crate::device::DeviceKernel],
) -> String {
    emit_dispatch(tables, jit, Arms::Whole)
}

/// The SAME arms, JIT ONLY — the half a parity harness needs before a row is
/// routed.
///
/// # Why this exists, and why it is the same function underneath
///
/// Routing a row DELETES its shim entry ([`emit_c_shim`] skips what
/// [`crate::device::JIT_DISPATCHED`] names), so the moment a row is routed
/// there is no way left for Rust to call its ahead-of-time launcher. The
/// comparison that would have proved the flip safe becomes unbuildable the
/// instant the flip lands, which is why nothing had ever run it: the AOT arm
/// and the JIT arm cannot both exist for one symbol **in one dispatcher**.
///
/// They can exist in two. This emits a second `match` over the same tables in
/// which every row with a device twin takes the JIT branch and every other row
/// is ABSENT — no `pie_k_*` call, so the probe cannot silently measure the AOT
/// path twice, which is the one way a parity harness can certify nothing while
/// passing. The production dispatcher is unchanged and still holds the AOT arm
/// for an unrouted row, so one binary can fire one statement both ways.
///
/// **It is not a fallback and cannot become one.** Nothing chains the two:
/// `dispatch` never calls this, the probe never calls `dispatch`, and a symbol
/// this declines gets `false` rather than a second attempt. The driver
/// compiles it behind its own feature, off in every shipping build.
///
/// The arm text is produced by the identical code path — same guard, same
/// staging, same `jit_dims` call, same operand expressions — because it is the
/// same loop with one `continue` added. That is what makes the harness's claim
/// worth anything: what it fires is what routing will emit, and
/// `driver-cuda/build.rs` asserts the two files agree arm for arm on every
/// routed symbol rather than leaving that to this paragraph.
#[must_use]
pub fn emit_rust_dispatch_probe(
    tables: &[&'static [KernelSig]],
    jit: &[&'static crate::device::DeviceKernel],
) -> String {
    emit_dispatch(tables, jit, Arms::JitOnly)
}

/// Which arms [`emit_dispatch`] writes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Arms {
    /// The dispatcher: a JIT arm for a routed row, the `pie_k_*` arm for
    /// every other stated row.
    Whole,
    /// The probe: the JIT arm and nothing else.
    JitOnly,
}

fn emit_dispatch(
    tables: &[&'static [KernelSig]],
    jit: &[&'static crate::device::DeviceKernel],
    arms: Arms,
) -> String {
    let mut out = String::from(match arms {
        Arms::Whole => "// GENERATED by `kernels_cuda_new::abi::emit_rust_dispatch` — DO NOT EDIT.\n",
        // Named apart because the two files are the same shape and only one
        // of them is the dispatcher: a reader who opens the wrong one and
        // finds a JIT arm for an unrouted row would read it as a routing
        // that never happened.
        Arms::JitOnly => {
            "// GENERATED by `kernels_cuda_new::abi::emit_rust_dispatch_probe` — DO NOT EDIT.\n\
             //\n\
             // THE PROBE, not the dispatcher: every row with a device twin\n\
             // takes its JIT arm here whether or not it is routed, and a row\n\
             // without one is absent rather than sent to `pie_k_*`. Nothing\n\
             // in a shipping build includes this file.\n"
        }
    });
    out.push_str(
        "//\n\
         // One branch per kernel! row that states both its operand types\n\
         // and where each argument comes from. A row missing either is\n\
         // absent here and belongs to a hand-written arm until it states\n\
         // them — which is a row's work, not the driver's.\n\
         //\n\
         // Keyed by the SYMBOL, like the C++ twin: the statement carries\n\
         // it, so there is no derived handle for the two to disagree on.\n\
         //\n\
         // A whole `match` EXPRESSION, not a run of arms: `include!` takes\n\
         // an expression or an item and never a pattern, so a file of bare\n\
         // arms has nowhere to be included from.\n\
         match b.kernel {\n",
    );
    for k in stated(tables) {
        let binds: Option<Vec<String>> =
            k.operands.iter().map(rust_bind_expr).collect();
        let Some(binds) = binds else { continue };

        // THE GUARD COMES FROM THE ROW THAT WILL RUN.
        //
        // A JIT'd symbol has two rows: the AOT one, whose operands are what
        // the shim's C function takes, and the device one, whose operands are
        // what `cuLaunchKernel` takes. They differ by the facts a launch does
        // not pass as arguments -- the stream, and every extent a
        // `LaunchRule` recovers -- and those differences reach the GUARD,
        // because an operand is what says which results and inputs a
        // statement must have.
        //
        // Reading the AOT row's arity for a fire that will run the device row
        // is how a branch comes to decline a launch it could have served, and
        // the fallthrough for a switched row is `UnknownKernel`. So `k` names
        // the symbol and `g` states the shape.
        let device = jit.iter().find(|d| d.sig.symbol == k.symbol);
        // THE PROBE HAS NO OTHER ARM. A row with no device twin is left out
        // of the probe entirely rather than given its `pie_k_*` arm: a
        // harness that fired the AOT launcher on both sides would compare a
        // buffer with itself and report parity for a row that has no JIT
        // path at all. `false` from the probe says "no JIT arm", which is a
        // gate 2 failure and is what the harness must see.
        if arms == Arms::JitOnly && device.is_none() {
            continue;
        }
        let g: &'static KernelSig = device.map_or(k, |d| d.sig);

        // THE ARITY THE SOURCES ASK FOR, as part of the match — the C++
        // twin's reasoning verbatim. A branch that indexes `outs[1]` must
        // not run on a statement with one result; here the cost of not
        // checking is worse than a wrong answer, because the flat run
        // means an over-index reads the NEXT operand rather than faulting.
        let (mut need_in, mut need_out, mut need_w, mut need_ps) = (0u8, 0u8, 0u8, 0u8);
        // WALKED, not matched. The arity a row asks for is the deepest
        // index its sources reach, and reaching it by walking means a
        // new combinator cannot forget to contribute — which is what a
        // hand-maintained `match` over combination variants did, and how
        // a branch comes to decline silently.
        fn reach(s: &Source, i: &mut u8, o: &mut u8, w: &mut u8, p: &mut u8) {
            match *s {
                Source::In(x)
                | Source::InRows(x)
                | Source::InElements(x)
                | Source::InWidth(x) => *i = (*i).max(x + 1),
                Source::Out(x)
                | Source::OutRows(x)
                | Source::OutWidth(x)
                | Source::OutElements(x) => *o = (*o).max(x + 1),
                Source::Weight(x) => *w = (*w).max(x + 1),
                Source::Param(x) | Source::ParamF32(x) | Source::RoutesOfParam(x) => {
                    *p = (*p).max(x + 1)
                }
                // A slot reached through `Or` is OPTIONAL by
                // construction — that is what `Or` means — so the left
                // branch must not raise the arity. Only the fallback's
                // reach counts, and a fallback that is also a slot would
                // be a row saying "either of two slots", which no row
                // says.
                Source::Or(_, ref b) => reach(b, i, o, w, p),
                Source::Width(ref a) | Source::Isqrt(ref a) => reach(a, i, o, w, p),
                Source::Mul(ref a, ref b)
                | Source::Sub(ref a, ref b)
                | Source::Div(ref a, ref b)
                | Source::Ne(ref a, ref b) => {
                    reach(a, i, o, w, p);
                    reach(b, i, o, w, p);
                }
                Source::IfPresent(ref q, ref a, ref b) => {
                    reach(q, i, o, w, p);
                    reach(a, i, o, w, p);
                    reach(b, i, o, w, p);
                }
                _ => {}
            }
        }
        for o in g.operands {
            reach(&o.source, &mut need_in, &mut need_out, &mut need_w, &mut need_ps);
        }
        // AN IN-PLACE ROW IS STAGED, and the staging is the row's too.
        //
        // `in_place = &[(0, 0)]` says result 0 aliases operand 0 — a fact
        // about the KERNEL, which reads and writes one buffer. The
        // lowering honours it where it can, and where it cannot (the
        // operand is live elsewhere) it assigns distinct buffers and
        // SOMETHING has to copy before the launch. That something used to
        // be a hand-written arm, and the row was skipped here for exactly
        // that reason.
        //
        // It is a convention, not a decision: `stage_d2d` is a no-op when
        // the two already alias, so emitting it is right in both the
        // honoured and the un-honoured case, and the driver owns what a
        // copy costs. Which is what makes this generatable at all —
        // nothing about WHICH buffers alias is knowable here, and nothing
        // about it needs to be.
        //
        // THE PAIR IS ARITY-DEPENDENT, and the runtime test is `lower.rs`'s
        // verbatim: *"a pair outside this statement's arity is not an
        // error: one symbol serves a q-only site and a q/k pair, and the
        // row states the widest form."* `mlp::chunked_swiglu_bf16` is that
        // case — `swiglu_aligned` states the block-major staging buffer as
        // its second operand and the activation must land on it, while
        // plain `swiglu` states one operand and there is nothing to alias.
        // So the indices are tested and not asserted, which is also why
        // they must NOT raise the arity guard: a row that demanded its
        // widest form would decline the narrow site outright.
        //
        // This is what the qwen3_5 A/B caught and gemma-4's did not:
        // `norm::residual_add_bf16` is in every layer of both, and the
        // difference was only whether that fire's buffer assignment
        // happened to alias. A bug whose reproduction depends on an
        // allocator is one to make structurally impossible, which is what
        // staging every in-place row from the row does.
        let mut stage = String::new();
        for (out, inp) in k.in_place {
            stage.push_str(&format!(
                "    if n_in > {inp} && n_out > {out} {{\n        \
                 stage_d2d(ctx, &b.rows, b.args[n_in + {out}], b.args[{inp}]);\n    }}\n"
            ));
        }
        // THE RECTANGLE, and the guard that is no longer here.
        //
        // `In`/`Out`/`Positions` used to bind a value's BASE, which is the
        // whole of the value only when the rectangle is the whole fire. A
        // peel splits a layer body in two and the second region starts at
        // `win_start`, so a launch over it must address rows `[win_start,
        // win_start + rows)`; at the base it wrote the PREFIX region's
        // rows, both regions wrote the same range, and the later one won.
        //
        // So every generated branch carried ` if b.rows.start == 0` and a
        // windowed rectangle fell through to a hand arm. That was the
        // honest answer and it was also the wrong shape: the hand arms had
        // the SAME bug and no guard, so the fallthrough was not a safety
        // net, it was a hiding place. Nothing noticed until a fire finally
        // peeled.
        //
        // The binder carries the window now (`resolve_arg_windowed`), and
        // it applies to the op join's placements as well as to the stated
        // args — one place, so a launch cannot read at the window and
        // write at the base. The C++ twin answered the same question by
        // putting the row in its context (`ArmCtx::row`); this answers it
        // one layer lower, where both consumers already meet.
        let mut guard = String::new();
        // A ZERO BOUND IS NOT A CLAUSE. `n_out >= 0` on an unsigned is
        // always true and rustc says so, thirty-six times, in a file
        // nobody edits — which is how a real warning in generated code
        // would go unread.
        let mut first = true;
        // ADDITION on the right, not subtraction on the left:
        // `b.args.len() - n_in - n_out` underflows a `usize` on a
        // statement shorter than its arity, which is a panic where a
        // declined branch is wanted.
        for clause in [
            (need_in > 0).then(|| format!("n_in >= {need_in}")),
            (need_out > 0).then(|| format!("n_out >= {need_out}")),
            (need_w > 0).then(|| format!("b.args.len() >= n_in + n_out + {need_w}")),
        ]
        .into_iter()
        .flatten()
        {
            guard.push_str(if first { " if " } else { " && " });
            first = false;
            guard.push_str(&clause);
        }
        if first {
            guard.push_str(" if true");
        }
        if need_ps > 0 {
            guard.push_str(&format!(" && spec.params.len() >= {need_ps}"));
        }
        // A NAME THE STORE LACKS IS DRIFT, not absence, and the right
        // answer is the hand arm's `UnknownWeight` rather than a null
        // bound into a kernel. So the branch declines and says nothing;
        // the fallthrough is what reports.
        for _o in g.operands {
        }
        if any_leaf(g, |s| *s == Source::WeightNamed2) {
            guard.push_str(" && !w_named2.is_null()");
        }
        if any_leaf(g, |s| *s == Source::WeightNamed) {
            guard.push_str(" && !w_named.is_null()");
        }
        // A FIRE WITH NO RECURRENT LAYERS CARRIES NO GDN CONTEXT, so a
        // row reading one declines rather than reading a default. This
        // is what makes `g_of`'s unwrap total.
        if any_leaf(g, |s| {
            matches!(
                *s,
                Source::Gdn(_) | Source::GdnSlab(_)
            )
        }) {
            guard.push_str(" && gdn.is_some()");
        }
        // THREE WAYS A SLAB IS ABSENT, and this tests all of them: the
        // fire may carry no GDN context (above), the op may state no
        // layer, and the context may hold no slab at that layer. The hand
        // arms spelled the last two as `state_layer()?` and `slab(..)?`.
        for o in g.operands {
            if let Source::GdnSlab(f) = o.source {
                guard.push_str(&format!(" && gdn_slab(gdn, spec.state, \"{f}\").is_some()"));
            }
        }
        // A FIRE WITH NO ATTENTION CARRIES NO ATTENTION CONTEXT, and a
        // statement may name a layer the fire holds no cache for. Both
        // are tested here so `a_of` and `kv_view` are total.
        if any_leaf(g, |s| {
            matches!(
                *s,
                Source::Attn(_)
                    | Source::AttnNonZero(_)
                    | Source::AttnWindow
                    | Source::AttnPlan(_)
            )
        }) {
            guard.push_str(" && attn.is_some()");
        }
        // A NULL PLAN IS A FIRE THAT RAISED NONE -- a pure-prefill fire
        // has no decode plan and vice versa, and the hand arms tested it
        // by hand (`if a.prefill_plan.is_null() { return Err(..) }`).
        // After `attn.is_some()` for the same short-circuit reason.
        for o in g.operands {
            if let Source::AttnPlan(family) = o.source {
                guard.push_str(&format!(
                    " && !attn_plan(a_of(attn), spec, u32::from(b.layers.start), \"{family}\").is_null()"
                ));
            }
        }
        // AFTER `attn.is_some()`, and for the reason the `AttnNonZero`
        // loop below is: `&&` short-circuits left to right, so a clause
        // that reaches `a_of(attn)` must sit behind the test that makes
        // it total. This loop was emitted BEFORE it, and an `Or` whose
        // fallback is an attention field — `Or(&Out(0), &Attn("o_out"))`
        // is five of the flashinfer rows — put `a_of(attn).o_out` ahead
        // of `attn.is_some()`. `a_of` is an `expect`, so that is a panic
        // on a fire with no attention, reachable through `bind::dispatch`
        // with `None`.
        //
        // Moved rather than made null-safe: the order is the rule the
        // other three loops already follow, and one exception to a rule
        // three places keep is how the fourth place gets written.
        for o in g.operands {
        // THE WHOLE `Or`, not either branch. Guarding a branch would
        // defeat the point — the row says "either", and what it needs
        // is that the composition lands on something. So: the left is
        // PRESENT, or the right RESOLVES.
        if let Source::Or(a, b) = o.source {
            let Some(present) = slot_presence(a) else { continue };
            let resolves = match *b {
                Source::NamedScale => "named_scale(ctx, spec).is_some()".to_string(),
                // A NULL FALLBACK RESOLVES. `Or(&Out(1), &Lit(Null))`
                // is a row saying "the second result, or nothing" --
                // the launcher reads the null as "there is no k" --
                // so demanding it be non-null refuses the very form
                // the row was written to serve.
                Source::Lit(Lit::Null) => "true".to_string(),
                // A POINTER FALLBACK MAY BE NULL — a fire that
                // published no output leaves `o_out` null, and a
                // launch into it is a segfault with no CUDA error.
                // A scalar fallback is always a value.
                _ if o.ty.rust().starts_with('*') => {
                    match rust_bind_expr_of(b, o.ty) {
                        Some(e) => format!("!({e}).is_null()"),
                        None => continue,
                    }
                }
                _ => "true".to_string(),
            };
            guard.push_str(&format!(" && ({present} || {resolves})"));
        }
        }

        // AFTER `attn.is_some()`, and that order is load-bearing: `&&`
        // short-circuits left to right, so `a_of` is only reached once
        // the context is known to be there.
        for o in g.operands {
            if let Source::AttnNonZero(f) = o.source {
                guard.push_str(&format!(" && is_set(a_of(attn).{f})"));
            }
        }
        // The join may hold no such output — a statement outside any
        // value-producing region, or one whose guard produces nothing.
        // The hand arms spelled it `out_slot(i, resolver)?`.
        for o in g.operands {
            match o.source {
                Source::ResultOrRegion(i) => {
                    guard.push_str(&format!(" && join_out(spec, {i}, frame, resolver).is_some()"));
                }
                Source::Aux(i) => {
                    guard.push_str(&format!(" && join_aux(spec, {i}, frame, resolver).is_some()"));
                }
                _ => {}
            }
        }
        // A LAUNCH CARRYING FOREIGN VALUES IS ONLY THIS ROW'S IF THE ROW
        // SAYS SO.
        //
        // `dispatch_generated` used to turn away every launch with a
        // non-empty `aux` before the match, and its reason was right at
        // the time: aux values are "operands the trace does not state at
        // all", so a generated branch binding only the args would drop
        // one silently.
        //
        // `Source::Aux` is a row stating exactly that, so the blanket
        // refusal became the thing keeping the feature from working —
        // the branch emitted, read correctly, and was unreachable. What
        // replaces it is per row: a row that states no `Aux` declines a
        // launch that carries one, which is the same answer the blanket
        // check gave, and a row that states one is asked whether the
        // value it names is actually there.
        if !mentions(k, |s| matches!(*s, Source::Aux(_))) {
            guard.push_str(" && spec.aux.is_empty()");
        }
        // THE SAME MOVE `aux` MADE. A statement carrying `per_head_dim`
        // has a reading a flat row cannot express, so the check was
        // ahead of the match; `IfPresent(&PerHeadDim, ..)` is a row
        // expressing it, and the refusal is per row now.
        if !mentions(k, |s| *s == Source::PerHeadDim) {
            guard.push_str(" && spec.per_head_dim.is_none()");
        }
        // A STATEMENT THAT STATES NO ROTARY WIDTH AND A FIRE THAT CARRIES
        // NO TABLE is the hand arm's `NoArm`, one layer earlier.
        if any_leaf(g, |s| *s == Source::RotaryWidth) {
            guard.push_str(" && rotary_width(ctx, spec, b.layers.start as usize).is_some()");
        }
        if any_leaf(g, |s| {
            matches!(
                *s,
                Source::KvLayerView
                    | Source::KvKeys
                    | Source::KvValues
                    | Source::KvLayerField(_)
            )
        }) {
            guard.push_str(" && has_kv_layer(attn, b.layers.start as usize)");
        }
        // A field a family zeroes to say "not mine" — and a divisor,
        // which is the same test for a different reason: a width divided
        // by an unset field is not a smaller answer, it is a meaningless
        // one, and the two arms this replaced refused explicitly on it.
        //
        // DEDUPED, because two operands may divide by the same field —
        // `qk_rmsnorm_rope` reads a q head count and a kv head count off
        // one `head_dim` — and `is_set(x) && is_set(x)` is a guard that
        // makes a reader look for the difference.
        //
        // AND WALKED, not matched at the top level. `CtxNonZero` EXISTS
        // to be a divisor, so it is almost always nested inside a `Div`
        // — and an `if let` on `o.source` sees `Div` and moves on. Seven
        // arms divided by a context field with no `is_set` beside it,
        // which is exactly the failure this variant was introduced to
        // prevent: `rope::qk_rmsnorm_rope_bf16` among them, and the
        // variant's own doc says what that costs — "it would have rotated
        // half of gemma-4 by nothing, silently".
        //
        // The sharpest pair was inside one module. `norm::mean_streams`
        // states `CtxNonZero("altup_streams")` at top level and got its
        // guard; `norm::altup_predict` states it under a `Div` and did
        // not. Two rows reading one field, one refusing and one not.
        let mut guarded: Vec<&str> = Vec::new();
        let mut fields: Vec<&'static str> = Vec::new();
        for o in g.operands {
            for_each_leaf(&o.source, &mut |s| {
                if let Source::CtxNonZero(f) = *s {
                    fields.push(f);
                }
            });
        }
        for f in fields {
            {
                if guarded.contains(&f) {
                    continue;
                }
                guarded.push(f);
                // `is_set` rather than `!= 0`: the emitter does not know
                // the field's TYPE, and Rust will not compare an `f32` to
                // an integer literal. The driver implements it for the
                // kinds a context field can be, which puts the one thing
                // the generator cannot know on the side that knows it.
                guard.push_str(&format!(" && is_set(ctx.{f})"));
            }
        }

        // BOTH NAMES, not one instead of the other. `lowered_as` is an
        // ALIAS: the symbol is this row's identity everywhere else (the
        // shim, the audit, the ABI), and both spellings reach a lowering
        // -- `gemm::act_x_wt_bf16` is what a text naming the CUDA symbol
        // directly produces, `gemm::act_x_w` what the portable operation
        // does. One branch answers to both.
        let pattern = match k.lowered_as {
            Some(also) => format!("\"{}\" | \"{also}\"", k.symbol),
            None => format!("\"{}\"", k.symbol),
        };
        // A ROW WITH A DEVICE KERNEL GOES THE OTHER WAY.
        //
        // Same pattern, same guard, same staging -- everything about WHICH
        // launches match and what has to be copied first is the row's, and
        // the row has not changed. What changes is the last statement: a
        // `pie_k_*` call becomes `ArgValue`s and a fire through the module
        // NVRTC compiled, which is what lets the `.cu` launcher and the shim
        // entry behind it be deleted.
        //
        // The row consulted for the OPERANDS is the JIT one, because the two
        // differ by exactly the operands a `cuLaunchKernel` does not take:
        // the stream is its sixth parameter, outside the `void**`
        // (`new-horizon.md` §4.2).
        if let Some(device) = device {
            let Some(values): Option<Vec<String>> = device
                .sig
                .operands
                .iter()
                .map(|o| {
                    let variant = arg_value_variant(o.ty)?;
                    // A POINTER CROSSES AS AN ADDRESS, not as a typed
                    // pointer. `rust_bind_expr` appends the operand's C cast
                    // -- `.cast_const()` for a `Buf` -- which is what the
                    // shim's signature wants and what `ArgValue::Ptr` will
                    // not take. Constness is checked by `Args::bind` against
                    // the row instead, which is a better place for it: the
                    // row is where the claim lives.
                    let expr = if variant.ends_with("::Ptr") {
                        rust_bind_expr_of(&o.source, o.ty)?
                    } else {
                        rust_bind_expr(o)?
                    };
                    // AND THEN CAST TO THE ADDRESS TYPE, because "crosses as
                    // an address" was true of the expressions this had seen
                    // and not of the ones it had not.
                    //
                    // Every routed row until now bound its pointers from
                    // `b.args[..].ptr`, which is already `*mut c_void`, so
                    // the paragraph above described the code correctly by
                    // accident. The first row reading a DRIVER CONTEXT field
                    // instead -- `layout::gather_bf16_rows`, whose `I32s`
                    // operand resolves to `ctx.sampling_indices: *const i32`
                    // -- produced `ArgValue::Ptr(*const i32)` and a `types
                    // differ in mutability` error in a generated file.
                    //
                    // A cast rather than a wider `ArgValue`: the variant
                    // means "this operand crosses as an address", and an
                    // address has one type. Constness is not lost, it is
                    // checked somewhere better -- `Args::bind` tests it
                    // against the ROW, which is where the claim that an
                    // operand is read-only actually lives, and which catches
                    // the mismatch a `*const`-preserving variant would only
                    // catch when the two happened to meet.
                    let expr = if variant.ends_with("::Ptr") {
                        format!("({expr}) as *mut ::core::ffi::c_void")
                    } else {
                        expr
                    };
                    Some(format!("{variant}({expr})"))
                })
                .collect()
            else {
                continue;
            };
            // Built line by line rather than as one continued literal: a
            // `\`-continued string in Rust keeps the continuation's own
            // indentation, which is invisible in the emitter and ragged in
            // the emitted file.
            //
            // THE RECTANGLE IS EMITTED, THE GEOMETRY IS NOT. `jit_dims` takes
            // the two widths this row reads off its own operands and fills the
            // other six axes -- heads, head width, rotary channels, experts --
            // from the fire's context, because those are facts about the FIRE
            // and not about the statement. Emitting them here would mean this
            // generator naming `DispatchCtx` fields, which is a driver's
            // struct layout leaking into the crate that describes kernels; the
            // driver knows where it keeps its own geometry and answers in one
            // place instead of in three hundred arms.
            let mut call = vec![
                "    unsafe { crate::bind::jit::fire(".to_string(),
                format!("        \"{}\",", device.sig.symbol),
                "        jit_dims(".to_string(),
                "            b, spec, ctx, attn, rows,".to_string(),
                "            width_of(b, n_in + 0),".to_string(),
                "            width_of(b, 0),".to_string(),
                "        ),".to_string(),
                "        &[".to_string(),
            ];
            call.extend(values.iter().map(|v| format!("            {v},")));
            call.push("        ],".to_string());
            call.push("        ctx.stream,".to_string());
            // NO `bool` TO READ. `fire` returns `()` and the arm answers
            // `true` unconditionally, which is the "a refusal is not a
            // fallback" rule made structural rather than remembered: this row
            // has no shim entry, so there is no arm to fall through TO, and a
            // `false` here would report a refused fire as an unknown kernel.
            call.push("    ) };".to_string());
            call.push("    true".to_string());
            out.push_str(&format!(
                "{pattern}{guard} => {{\n{stage}{}\n}}\n",
                call.join("\n")
            ));
            continue;
        }
        // THE THIRD ARM: A ROW THE DRIVER EXECUTES ITSELF.
        //
        // Same `pattern`, same `guard`, same `stage`, same `binds` — built by
        // the same code, from the same operand list, a dozen lines above.
        // Only the callee's path differs, and that is the whole design: the
        // model compiler must not be able to tell whether a symbol is cuBLAS
        // or a JIT'd kernel, and the one thing that decides which arm a row
        // gets is a list of strings no lowering reads.
        //
        // `ctx` leads because the SERVICE carries what the service needs. The
        // rows these arms serve used to state `handle: CublasHandle <-
        // Source::Ctx("cublas")` as their first operand, and that spelling
        // put one backend's library type in a vocabulary two backends share
        // — zero Metal rows name it. The handle did not disappear; it moved
        // to where it was always coming from.
        //
        // `true` unconditionally, for `bind::jit::fire`'s reason: the row has
        // no shim entry, so there is no hand arm to fall through to, and a
        // `false` here would report a refusal as an unknown kernel.
        if crate::execution::RUST_SERVED.contains(&k.symbol) {
            let served = entry_name(k.symbol);
            let served = served.strip_prefix(PREFIX).unwrap_or(&served);
            out.push_str(&format!(
                "{pattern}{guard} => {{\n{stage}    unsafe {{ crate::bind::service::{served}(\n        ctx,\n        {},\n    ) }};\n    true\n}}\n",
                binds.join(",\n        "),
            ));
            continue;
        }
        out.push_str(&format!(
            "{}{} => {{\n{}    unsafe {{ {}(\n        {},\n    ) }};\n    true\n}}\n",
            pattern,
            guard,
            stage,
            format!("crate::bind::abi::ffi::{}", entry_name(k.symbol)),
            binds.join(",\n        "),
        ));
    }
    // The fallthrough IS the answer for every symbol with no branch, and
    // it is spelled here rather than at the include site so that what
    // lands on disk is a complete expression — which `include!` requires,
    // and which also means the file can be read on its own.
    out.push_str("_ => false,\n}\n");
    out
}

#[cfg(test)]
mod stated_rows {
    /// EVERY fully-stated row emits a branch, and the failure names the
    /// operand that declined.
    ///
    /// This exists because a row that silently fails to emit is
    /// indistinguishable from one whose branch never fires, and the
    /// difference cost two sessions. `Source::Unbound` skips the WHOLE
    /// row by design — a partially-stated row is not a row — so a row
    /// sourced everywhere but `stream` reads as complete and vanishes.
    ///
    /// Over every table rather than one row, because the next one to
    /// vanish will be a different one.

    /// EVERY DIRECT INDEX IS COVERED BY A GUARD IT CANNOT OUTRUN.
    ///
    /// The generated body writes `b.args[0]`, `b.args[n_in + 1]`,
    /// `b.args[n_in + n_out + 0]` and so on, and the branch's guard is
    /// what makes each of those in range. Two independent computations
    /// produce them — `reach` walks the sources for the arity, and
    /// `rust_bind_expr_of` walks them again for the expression — so
    /// nothing but this test says they agree.
    ///
    /// If they ever disagree the failure is a PANIC inside the executor,
    /// on a family nobody in the corpus lowers, at the moment a customer
    /// runs it. That is the one failure mode the generator was supposed
    /// to remove: a hand-written arm counted its own arguments and this
    /// is what replaced the counting.
    ///
    /// The walk is scope-aware because an `Or` guards its own access
    /// locally (`if n_out > 1 { b.args[n_in + 1] }`) and the in-place
    /// staging guards a pair (`if n_in > 0 && n_out > 0`). An index
    /// inside either is safe however weak the branch guard is, so a
    /// checker that ignored nesting would report the whole table.

    /// A CLAUSE THAT UNWRAPS SITS BEHIND THE TEST THAT MAKES IT TOTAL.
    ///
    /// `a_of` is an `expect`, so every guard clause reaching it must come
    /// after `attn.is_some()` — `&&` short-circuits left to right and
    /// that is the only thing making the unwrap safe. Three of the four
    /// loops that emit such clauses were placed after it deliberately,
    /// with a comment saying the order is load-bearing. The `Or` loop was
    /// placed before, and put `a_of(attn).o_out` ahead of the test on
    /// five flashinfer rows.
    ///
    /// One exception to a rule three places keep is how the fourth place
    /// gets written, so the rule is a test now.
    #[test]
    fn nothing_unwraps_the_attention_context_before_testing_it() {
        let text = super::emit_rust_dispatch(crate::table::TABLES, &crate::device::jit_dispatched());
        let mut early = Vec::new();
        for arm in text.split("\n\"").skip(1) {
            let Some(end) = arm.find(" => {") else { continue };
            let guard = &arm[..end];
            let (unwrap, test) = (guard.find("a_of(attn)"), guard.find("attn.is_some()"));
            if let Some(u) = unwrap {
                if test.is_none_or(|t| u < t) {
                    early.push(guard.split('"').next().unwrap_or(guard).to_string());
                }
            }
        }
        assert!(
            early.is_empty(),
            "these guards call `a_of(attn)` before `attn.is_some()`, which is \
             a panic on a fire with no attention:\n  {}",
            early.join("\n  ")
        );
    }

    /// A DIVISION BY A CONTEXT FIELD CARRIES THAT FIELD'S `is_set`.
    ///
    /// `CtxNonZero` exists to be a divisor — a family zeroes the field to
    /// say "this launch is not mine" — so it is almost always nested
    /// inside a `Div`, and the guard loop that collected it matched only
    /// the TOP-LEVEL source. Seven arms divided by a context field with
    /// no refusal beside it, which is the regression the variant's own
    /// doc describes: gemma-4's per-layer theta, where "it would have
    /// rotated half of gemma-4 by nothing, silently".
    #[test]
    fn a_division_by_a_context_field_is_guarded_on_it() {
        let text = super::emit_rust_dispatch(crate::table::TABLES, &crate::device::jit_dispatched());
        let mut naked = Vec::new();
        for arm in text.split("\n\"").skip(1) {
            let Some(end) = arm.find(" => {") else { continue };
            let (guard, body) = arm.split_at(end);
            let bytes = body.as_bytes();
            for (i, _) in body.match_indices(").max(1)") {
                // The divisor is the group `.max(1)` is called on, found
                // BY BALANCING rather than by the nearest `(ctx.`. It is
                // spelled `(i32::try_from(ctx.ple_dim).unwrap_or(..))`
                // once the field needs narrowing, and a scan that
                // stopped at the first `)` read the wrapper as the whole
                // divisor and skipped the field inside it — which is how
                // this test went quiet the day the narrowing landed. It
                // still passed. It just no longer looked.
                let mut depth = 0i32;
                let mut open = None;
                for j in (0..=i).rev() {
                    match bytes[j] {
                        b')' => depth += 1,
                        b'(' => {
                            depth -= 1;
                            if depth == 0 {
                                open = Some(j);
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                let Some(open) = open else { continue };
                let divisor = &body[open + 1..i];
                // EVERY field in the divisor, not the last one: a
                // divisor may compose two.
                for (k, _) in divisor.match_indices("ctx.") {
                    let field: String = divisor[k + "ctx.".len()..]
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                        .collect();
                    if field.is_empty() {
                        continue;
                    }
                    if !guard.contains(&format!("is_set(ctx.{field})")) {
                        naked.push(format!(
                            "{}: divides by ctx.{field} with no is_set",
                            guard.split('"').next().unwrap_or(guard).trim()
                        ));
                    }
                }
            }
        }
        assert!(
            naked.is_empty(),
            "a family that zeroed one of these fields would divide by one \
             and launch instead of declining:\n  {}",
            naked.join("\n  ")
        );
    }

    /// THE TEST ABOVE CAN SEE THROUGH THE WRAPPER.
    ///
    /// Its scan is textual, so nothing but this says it still finds a
    /// field the narrowing hid. A divisor spelled the wrapped way must
    /// still be reported when its guard is missing — the assertion is on
    /// the SCANNER, with a body written here rather than emitted.
    #[test]
    fn the_divisor_scan_finds_a_field_inside_its_narrowing() {
        // Shaped exactly like an emitted arm: guard, ` => {`, body.
        let arm = "\nlayout::x\" if n_in >= 1 => {\n  \
            i32::try_from((width_of(b, 0)) / (i32::try_from(ctx.ple_dim)\
            .unwrap_or(i32::MAX)).max(1)).unwrap_or(i32::MAX),\n}";
        let (guard, body) = arm.split_at(arm.find(" => {").expect("shaped like an arm"));
        let bytes = body.as_bytes();
        let mut found = Vec::new();
        for (i, _) in body.match_indices(").max(1)") {
            let mut depth = 0i32;
            let mut open = None;
            for j in (0..=i).rev() {
                match bytes[j] {
                    b')' => depth += 1,
                    b'(' => {
                        depth -= 1;
                        if depth == 0 {
                            open = Some(j);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let Some(open) = open else { continue };
            let divisor = &body[open + 1..i];
            for (k, _) in divisor.match_indices("ctx.") {
                let field: String = divisor[k + "ctx.".len()..]
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect();
                if !field.is_empty() {
                    found.push(field);
                }
            }
        }
        assert_eq!(found, vec!["ple_dim".to_string()], "the scan lost the wrapped field");
        assert!(!guard.contains("is_set(ctx.ple_dim)"), "the fixture must be the unguarded shape");
    }

    /// A DECLARED FIELD REACHED THROUGH AN OPERATOR IS NARROWED FIRST.
    ///
    /// `cast_for` states the rule a row relies on — "a row declares
    /// `I32` and gets an i32, whatever the width of the thing it named"
    /// — but it runs once, on the whole operand. A field reached through
    /// a `Div` was composed before the rule applied, so
    /// `Div(Width(In(0)), CtxNonZero("ple_dim"))` emitted `i32 / u32`
    /// and `driver-cuda` did not compile with `abi` at all. The row was
    /// correct; the seam was not.
    ///
    /// Over every table, because the next `u32` field a row divides by
    /// will be a different one, and because the driver decides those
    /// widths in a crate this one cannot see.
    #[test]
    fn a_declared_field_under_an_operator_is_narrowed_first() {
        use kernels::Source;

        /// Arithmetic operands, and the conditionals that can hold one.
        ///
        /// Enumerated rather than derived: a grammar node added later
        /// under-covers here instead of failing, which is the safe
        /// direction for a test whose job is to catch a shape.
        fn operands_of(s: &'static Source, out: &mut Vec<&'static Source>) {
            match *s {
                Source::Mul(a, c) | Source::Sub(a, c) | Source::Div(a, c) | Source::Ne(a, c) => {
                    out.push(a);
                    out.push(c);
                    operands_of(a, out);
                    operands_of(c, out);
                }
                Source::Isqrt(a) => {
                    out.push(a);
                    operands_of(a, out);
                }
                Source::Or(a, c) => {
                    operands_of(a, out);
                    operands_of(c, out);
                }
                Source::IfPresent(p, t, o) => {
                    operands_of(p, out);
                    operands_of(t, out);
                    operands_of(o, out);
                }
                Source::Width(a) => operands_of(a, out),
                _ => {}
            }
        }

        let mut bare = Vec::new();
        for table in crate::table::TABLES {
            for row in *table {
                for op in row.operands {
                    let mut kids = Vec::new();
                    operands_of(&op.source, &mut kids);
                    for kid in kids {
                        if !super::reads_a_declared_field(kid) {
                            continue;
                        }
                        let Some(narrowed) = super::rust_arith_of(kid, op.ty) else { continue };
                        if !narrowed.contains("i32::try_from(") {
                            bare.push(format!("{} operand `{}`: {narrowed}", row.symbol, op.name));
                        }
                    }
                }
            }
        }
        assert!(
            bare.is_empty(),
            "these compose a driver-declared field into arithmetic without \
             narrowing it to the grammar's i32, so the generated file will \
             not compile the day the driver declares one of them `u32`:\n  {}",
            bare.join("\n  ")
        );
    }

    #[test]
    fn every_index_the_body_writes_is_inside_its_guard() {
        let text = super::emit_rust_dispatch(crate::table::TABLES, &crate::device::jit_dispatched());
        let mut unguarded: Vec<String> = Vec::new();

        for arm in text.split("\n\"").skip(1) {
            let Some(head_end) = arm.find(" => {") else { continue };
            let (head, rest) = arm.split_at(head_end);
            let guard = head;
            let body = &rest[" => {".len()..];

            // What the branch's own guard promises.
            let promise = |needle: &str| -> usize {
                guard
                    .match_indices(needle)
                    .filter_map(|(i, _)| {
                        guard[i + needle.len()..]
                            .split(|c: char| !c.is_ascii_digit())
                            .next()
                            .and_then(|d| d.parse::<usize>().ok())
                    })
                    .max()
                    .unwrap_or(0)
            };
            let base = [
                promise("n_in >= "),
                promise("n_out >= "),
                promise("b.args.len() >= n_in + n_out + "),
            ];

            // Walk the body, tracking the `if` conditions in scope.
            let bytes: Vec<char> = body.chars().collect();
            let (mut stack, mut cur) = (Vec::new(), base);
            let mut at: Vec<[usize; 3]> = Vec::with_capacity(bytes.len());
            for (i, &c) in bytes.iter().enumerate() {
                if c == '{' {
                    let from = i.saturating_sub(300);
                    let seg: String = bytes[from..i].iter().collect();
                    let cond = seg.rsplit("if ").next().unwrap_or("").to_string();
                    let mut add = cur;
                    for (k, needle) in
                        [(0usize, "n_in > "), (1, "n_out > "), (2, "b.args.len() > n_in + n_out + ")]
                    {
                        for (j, _) in cond.match_indices(needle) {
                            if let Some(d) = cond[j + needle.len()..]
                                .split(|c: char| !c.is_ascii_digit())
                                .next()
                                .and_then(|d| d.parse::<usize>().ok())
                            {
                                add[k] = add[k].max(d + 1);
                            }
                        }
                    }
                    stack.push(cur);
                    cur = add;
                } else if c == '}' {
                    cur = stack.pop().unwrap_or(base);
                }
                at.push(cur);
            }

            let body_s: String = bytes.iter().collect();
            for (i, _) in body_s.match_indices("b.args[") {
                let Some(end) = body_s[i..].find(']') else { continue };
                let idx = body_s[i + "b.args[".len()..i + end].trim().to_string();
                let scope = at.get(body_s[..i].chars().count()).copied().unwrap_or(base);
                let mut want = |k: usize, n: usize| {
                    if n >= scope[k] {
                        unguarded.push(format!(
                            "{}: b.args[{idx}] needs {} > {n}, guard gives {}",
                            guard.split('"').next().unwrap_or(guard).trim(),
                            ["n_in", "n_out", "len-n_in-n_out"][k],
                            scope[k]
                        ));
                    }
                };
                if let Ok(n) = idx.parse::<usize>() {
                    want(0, n);
                } else if let Some(k) = idx.strip_prefix("n_in + n_out + ") {
                    if let Ok(n) = k.trim().parse::<usize>() {
                        want(2, n);
                    }
                } else if let Some(k) = idx.strip_prefix("n_in + ") {
                    if let Ok(n) = k.trim().parse::<usize>() {
                        want(1, n);
                    }
                }
            }
        }

        assert!(
            unguarded.is_empty(),
            "these generated indexes can run past the argument list:\n  {}",
            unguarded.join("\n  ")
        );
    }

    #[test]
    fn a_fully_stated_row_emits_a_branch() {
        let tables: &[&[kernels::KernelSig]] = &[
            crate::table::attn::KERNELS,
            crate::table::gemm::KERNELS,
            crate::table::layout::KERNELS,
            crate::table::mlp::KERNELS,
            crate::table::moe::KERNELS,
            crate::table::norm::KERNELS,
            crate::table::quant::KERNELS,
            // `crate::table::rope::KERNELS` stood here; `rope` states no
            // operands now and would contribute no row to this walk.
            crate::table::ssm::KERNELS,
        ];
        let mut missing = Vec::new();
        for table in tables {
            let text = super::emit_rust_dispatch(&[table], &crate::device::jit_dispatched());
            for k in table.iter().filter(|k| !k.operands.is_empty()) {
                let declining: Vec<String> = k
                    .operands
                    .iter()
                    .filter(|o| super::rust_bind_expr(o).is_none())
                    .map(|o| format!("{:?}/{:?}", o.source, o.ty))
                    .collect();
                // A row with a declining operand is EXPECTED not to
                // emit — that is the skip working. What this catches is
                // a row whose every operand binds and which still does
                // not appear.
                if declining.is_empty() && !text.contains(k.symbol) {
                    missing.push(k.symbol.to_string());
                }
            }
        }
        assert!(
            missing.is_empty(),
            "every operand of these rows binds and none emitted a branch: {missing:?}"
        );
    }

    /// A row whose `elem` is a template ARGUMENT LIST still produces C++.
    ///
    /// # The silent failure this pins
    ///
    /// `DeviceKernel::instantiation` pastes `elem` between angle brackets, so
    /// a row for `template <class T, int BLOCK>` may spell
    /// `"device::bf16, 256"` — measured under NVRTC 13.0, and eight rows in
    /// the tree do it today. This emitter builds a POINTER type from the same
    /// string, and `const ::pie_cuda_driver::kernels::device::bf16, 256*` is
    /// not one.
    ///
    /// Nothing caught that, because the generated file is only compiled when a
    /// build wires this emitter in — so the effect was that the eight rows
    /// most likely to drift were the eight this check silently stopped
    /// covering. The head of the list is the storage type WHEN the element
    /// type is the first template parameter; when it is not, the head is a
    /// value and the row is refused rather than mis-spelled — see
    /// `an_elem_whose_head_is_a_value_is_refused`.
    #[test]
    fn a_multi_argument_elem_still_spells_a_pointer() {
        use super::emit_device_typecheck;
        use crate::device::DeviceKernel;
        let rows: Vec<DeviceKernel> = crate::device::ALTUP_AUX
            .iter()
            .take(1)
            .map(|k| DeviceKernel {
                sig: k.sig,
                template_path: k.template_path,
                elem: "device::bf16, 256",
            })
            .collect();

        let text = emit_device_typecheck(&rows).expect("a listed elem is emitted");
        assert!(
            text.contains("const ::pie_cuda_driver::kernels::device::bf16*"),
            "the operand type took the whole list rather than its head:\n{text}"
        );
        assert!(
            !text.contains("bf16, 256*"),
            "a comma reached a pointer declarator, which is not C++:\n{text}"
        );
        // The INSTANTIATION keeps the whole list -- that is the half that has
        // to name the kernel, and it is why the two cannot share a string.
        assert!(
            text.contains("device::bf16, 256>"),
            "the instantiation lost its second argument:\n{text}"
        );
    }

    /// An `elem` whose head is empty is refused rather than emitted.
    #[test]
    fn an_elem_with_no_head_is_refused() {
        use super::emit_device_typecheck;
        use crate::device::DeviceKernel;
        let rows: Vec<DeviceKernel> = crate::device::ALTUP_AUX
            .iter()
            .take(1)
            .map(|k| DeviceKernel { sig: k.sig, template_path: k.template_path, elem: ", 256" })
            .collect();
        let why = emit_device_typecheck(&rows).expect_err("an empty head is refused");
        assert!(why.contains("empty"), "{why}");
    }

    /// An `elem` whose head is a VALUE is refused by name, not pasted into a
    /// declarator.
    ///
    /// The four spellings below are the ones the tree actually carries, and
    /// every one leads a template argument list with something that is not a
    /// type: `device::i32(256)` and `device::i32(128)` are functional casts,
    /// `device::true_type::value` is a static data member, and `true` and `8`
    /// are literals. Pasted into `const ::…::{head}*` they produce a
    /// declarator nvcc rejects INSIDE THE GENERATED FILE — a diagnostic
    /// naming a line no one wrote and not naming the row that caused it,
    /// which is the failure mode this emitter exists to avoid. The row is
    /// named here instead.
    #[test]
    fn an_elem_whose_head_is_a_value_is_refused() {
        use super::emit_device_typecheck;
        use crate::device::DeviceKernel;
        for elem in [
            "device::i32(256), false, false",
            "device::i32(128)",
            "device::true_type::value",
            "true, 8",
            "8",
        ] {
            let rows: Vec<DeviceKernel> = crate::device::ALTUP_AUX
                .iter()
                .take(1)
                .map(|k| DeviceKernel { sig: k.sig, template_path: k.template_path, elem })
                .collect();
            let why = emit_device_typecheck(&rows)
                .expect_err(&format!("`{elem}` leads with a value and must be refused"));
            assert!(
                why.contains("VALUE rather than a type"),
                "`{elem}` was refused for the wrong reason: {why}"
            );
        }
    }

    /// A specialisation-suffixed symbol yields a C++ IDENTIFIER.
    ///
    /// `check_{symbol}` replaced `::` and nothing else, so
    /// `rmsnorm_strided_bf16#vec8` — a spelling the tree already carries —
    /// emitted `check_rmsnorm_strided_bf16#vec8`, and a `#` in a declarator is
    /// a parse error. The generated file is the one place a reader cannot fix
    /// what they find, so the sanitisation belongs here rather than in a
    /// convention about how a row may be named.
    #[test]
    fn a_specialisation_suffix_is_not_in_the_checker_name() {
        use super::emit_device_typecheck;
        use crate::device::DeviceKernel;
        let rows: Vec<DeviceKernel> = crate::device::ALTUP_AUX
            .iter()
            .take(1)
            .map(|k| DeviceKernel { sig: k.sig, template_path: k.template_path, elem: k.elem })
            .collect();
        let text = emit_device_typecheck(&rows).expect("the row is emitted");
        let checker = text
            .lines()
            .find_map(|l| l.split("void (*const ").nth(1)?.split(')').next())
            .expect("a checker is declared");
        assert!(
            checker.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
            "`{checker}` is not a C++ identifier"
        );
        assert!(
            !checker.contains("::"),
            "a scope separator survived into `{checker}`"
        );
    }
}
