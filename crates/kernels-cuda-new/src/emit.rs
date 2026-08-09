//! The direct-call surface, generated from the rows — this crate's
//! replacement for `emit_c_shim`.
//!
//! # What is being replaced, and by what
//!
//! `kernels-cuda`'s `abi::emit_c_shim` emits one `extern "C" pie_k_<symbol>`
//! per row, forwarding into a C++ host launcher with the real header in
//! scope, and `abi::emit_rust_bindings_portable` emits the matching
//! `unsafe extern "C"` block. That pair is the whole reason `model-loader`
//! can call four kernels by name today — it writes
//! `kernels_cuda::ffi::pie_k_quant_quantize_bf16_to_fp8_e4m3_per_channel(…)`,
//! and the ROW is what decides whether that call compiles.
//!
//! Under the JIT neither half exists, and not because something better was
//! found: the thing they connected is gone. There is no host launcher to
//! forward to, no archive to link, and no `extern "C"` symbol for a caller to
//! declare. What a fire needs is a row, a compiled module, and
//! `runtime::fire`. So this module is the SAME generator over the SAME rows,
//! emitting a Rust function where the other emitted a C forwarder — and the
//! property it exists to keep is the property the shim was worth having for:
//! **a row that changes its operands changes both call sites — the direct one
//! and the dynamic one — or fails to compile.**
//!
//! # Why not a hand-written wrapper over `fire`
//!
//! Because `fire` takes a `&[ArgValue]`, and a list is not a signature. A
//! wrapper written by hand would state each row's operands a second time, in
//! a second file, and the two would agree until the day a row changed. What a
//! reader would get then is not a compile error but a REFUSAL, at run time,
//! from inside a launch: `driver-cuda`'s `bind::device` header names that
//! trade in its own words — the value check "is a runtime check where the
//! shim's was a compile-time one", and it "is bought back by generation: the
//! caller does not write the list, the row does." This module is the direct
//! caller's half of that sentence. `model-loader` gets a typed function whose
//! arity and kinds are the row's, and a row it no longer matches is a build
//! failure at the call site rather than a `tracing` line under a model that
//! answers wrongly.
//!
//! # A row the binder refuses is ABSENT, not wrong
//!
//! `Args::bind` marshals a fixed set of [`Ty`]s and refuses the rest —
//! `Ty::Stream` and the other handles because a stream is `cuLaunchKernel`'s
//! sixth parameter rather than an argument, the struct and enum-class kinds
//! (`Ty::MoeActivation`, `Ty::Dtype`, `Ty::CustomAllReduce`) because a
//! `void**` cell is eight bytes and they are not. A generated function that
//! bound one of those would compile here and be refused there: a runtime
//! failure where a build-time absence was wanted. So such a row emits a
//! COMMENT naming the symbol, the operand and the type, and no function —
//! which is `emit_rust_bindings_portable`'s rule ("a row which cannot be
//! declared is absent rather than wrong") and which is what makes the absence
//! something a diff shows.
//!
//! The set is the binder's and moves when it does. [`Ty::Bool`] and
//! [`Ty::I64`] were both on the refused side and are not any more, for
//! reasons `runtime::args` records: a `bool` parameter is one byte in the
//! cubin's metadata and the driver copies one byte, and refusing a
//! `long long` made every batched SSM kernel unfireable because
//! `slot_stride_elems` is an element count into a multi-gigabyte arena. Six
//! rows were absent from this file for as long as its copy of that list
//! lagged — four `I64`, two `Bool` — which is the cheap direction of the
//! drift and still six kernels nobody could call by name.
//!
//! # What this file may name
//!
//! `build.rs` includes it with `#[path]`, because a build script cannot
//! depend on the crate it builds — so every module this one reads is included
//! there beside it, under the library's own names. `crate::unit`,
//! `crate::device` and `crate::families` therefore resolve in both crates:
//! in the library against `src/lib.rs`, in the script against `build.rs`,
//! which is its own crate root. That is `kernels-cuda/build.rs`'s arrangement
//! exactly, down to the `#[allow]` each included module carries, and it is
//! what the alternative reading of the same constraint cost: while this file
//! read only what it could reach WITHOUT a sibling —
//! `kernels_cuda::norm_device`'s two statics — the façade covered ten rows of
//! a hundred and thirty-five, and the other hundred and twenty-five had no
//! typed entry point at all.
//!
//! One module is not included there and must not be named here:
//! `crate::source`, whose header set arrives through
//! `include!(concat!(env!("OUT_DIR"), …))` and so cannot exist while the
//! script that writes it is being compiled. `build.rs` stubs it for the
//! tables' sake; nothing in this file asks for a source, because a row is not
//! one.
//!
//! The GENERATED text says `crate::runtime::…` freely: that is a string here
//! and a path only where it lands.

use kernels::{KernelSig, Ty};
use crate::device::DeviceKernel;

/// Every row this crate can compile, in unit order.
///
/// [`crate::unit::rows`], which is the concatenation of every unit's rows in
/// [`crate::unit::UNITS`]' order — so this is not a list of rows that agrees
/// with the units, it IS the units' rows. That is the only reason the two
/// cannot drift: a row added to a family is a row this emitter sees, and a
/// row this emitter sees is a row some unit compiles. The test at the bottom
/// asserts both directions, and it can, because both directions are true.
///
/// It read `kernels_cuda::norm_device::ENTRIES` and `ELEMENTWISE` instead
/// until the families were pulled into `build.rs` — the two statics that were
/// every row in the crate when the pilot shipped and were ten of a hundred
/// and thirty-five afterwards. Those ten are still here, and reached the way
/// every other row is: `norm`'s two units name them as their `rows`.
#[must_use]
pub fn all_rows() -> Vec<&'static DeviceKernel> {
    crate::unit::rows().collect()
}

/// One typed Rust function per row, as the text the `api` module includes.
///
/// Each emitted item takes the row's operands in the row's order, then `dims`
/// and `stream`, and forwards to `runtime::fire` with an `ArgValue` per
/// operand. A row is emitted only when every operand marshals: one naming a
/// [`Ty`] the binder refuses gets a comment naming the first such operand and
/// its type instead, and so does a row that states no operands at all, which
/// is how [`KernelSig::operands`] spells "not written yet".
///
/// # Determinism
///
/// The same rows produce byte-identical text: the walk is over the slice in
/// order, every decision is a `match` on a row's own data, and nothing here
/// hashes, sorts or iterates a map. A build script writing a file that
/// differed run to run would rebuild every dependent of this crate for no
/// reason, and would make a real diff in `api.rs` unreadable.
///
/// # Collisions need no check here
///
/// `kernels_cuda::abi::emit_c_shim` errors when two rows claim
/// one entry point, because a C symbol is a link-time global. The flatten
/// below is not injective either — `a::b_c` and `a_b::c` land on one name —
/// but the consequence is a duplicate `fn` in one module, which rustc reports
/// at the include site, by name, with both spans. The name IS the flattened
/// symbol and each function carries its symbol verbatim in a doc line, so the
/// rows are recoverable from the error. A second check here would only move
/// the same failure one build phase earlier.
#[must_use]
pub fn emit_rust_api(rows: &[&'static DeviceKernel]) -> String {
    let mut out = String::from(BANNER);
    for row in rows {
        out.push_str(&one(row));
    }
    out
}

/// The header the generated file opens with.
///
/// The same four facts every emitter in this workspace states — that the file
/// is generated, by what, from which rows, and that editing it reaches
/// nothing — plus the one a reader of `api.rs` cannot get anywhere else: what
/// this file is instead of.
const BANNER: &str = "// GENERATED by `kernels_cuda_new::emit::emit_rust_api` -- do not edit.\n\
     //\n\
     // One `pub unsafe fn` per row of `crate::unit::UNITS`, in that table's\n\
     // order -- the same rows every unit compiles and `crate::runtime::fire`\n\
     // resolves a symbol against. The build script writes this into OUT_DIR\n\
     // and `src/lib.rs` includes it as the `api` module, so an edit here is\n\
     // overwritten by the next build and reaches nothing before it.\n\
     //\n\
     // This replaces `kernels_cuda::abi::emit_c_shim`. Ahead of time, a\n\
     // direct caller reached a kernel through a generated `extern \"C\"`\n\
     // `pie_k_*` that forwarded into a C++ launcher; under the JIT there is\n\
     // no launcher and no C symbol, so the same generator reads the same rows\n\
     // and emits a Rust function. A row that changes its operands therefore\n\
     // changes this text AND the dynamic path's, or does not build.\n\
     //\n\
     // A row whose operands the binder would refuse is ABSENT, with a comment\n\
     // naming the operand and the type where its function would be: binding a\n\
     // kind `Args::bind` rejects would be a refusal at launch where an\n\
     // absence at build was wanted, and the comment is what puts that absence\n\
     // in the diff.\n\n";

/// The doc block every generated function carries under its headline.
///
/// Constant because it does not vary by row: what a caller must guarantee is
/// the same three things every time, and they are `fire`'s own obligations
/// restated where the caller can read them. A per-row rendering of the same
/// sentence would only invite one of them to drift.
const SAFETY_DOC: &str = "///\n\
     /// Generated from the row. Operands in the order the row declares them; the\n\
     /// launch geometry is the row's `LaunchRule` applied to `dims`.\n\
     ///\n\
     /// # Safety\n\
     ///\n\
     /// Every pointer must be a device address valid for this launch, `stream`\n\
     /// must be live for its duration, and `dims` must describe the extent the\n\
     /// buffers actually cover — the row's rule turns it into a grid, and a grid\n\
     /// wider than the allocation is an out-of-bounds write no hardware reports.\n";

/// How one operand crosses into a launch: the type a caller passes, and the
/// `ArgValue` the value becomes.
///
/// The variants are `ArgValue`'s, split once more on pointer CONSTNESS —
/// which `ArgValue` does not carry, because a device address is eight bytes
/// however it was declared. Constness survives here because it is the one
/// half of a pointer's type this boundary can honestly state: the pointee's
/// width and format are checked where they are knowable, against the
/// instantiation the row names, and a `*const f32` parameter would advertise
/// a check no launch performs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Crossing {
    /// A device address the kernel only reads.
    PtrConst,
    /// A device address the kernel may write through.
    PtrMut,
    /// A 32-bit signed scalar.
    I32,
    /// A 32-bit unsigned scalar.
    U32,
    /// A 32-bit float scalar.
    F32,
    /// A pointer-width unsigned scalar.
    Usize,
    /// A 64-bit signed scalar — a `long long` in the headers, and never a
    /// widened [`Crossing::I32`]: the values that need it are strides into
    /// arenas past 2^31 elements, where a truncation addresses another
    /// request's state rather than failing.
    I64,
    /// A one-byte host flag.
    Bool,
    /// A one-byte host ENUMERATOR — [`Ty::KvScheme`] and [`Ty::KvDType`].
    ///
    /// The Rust side of the boundary carries the byte and not a mirror: the
    /// C++ states the underlying type (`enum class … : ::std::uint8_t`), so
    /// `Ty::needs_mirror` is false and a caller in any crate can spell the
    /// argument. Which enumerator it means is checked in the C++, against the
    /// instantiation the row names — the same trade [`Crossing::PtrConst`]
    /// makes for a pointee's format.
    U8,
}

impl Crossing {
    /// The Rust type the generated parameter carries.
    ///
    /// A scalar is spelled as the type its `ArgValue` variant holds — `i32`,
    /// not `::core::ffi::c_int` as `Ty::rust` says. The two are the same type
    /// on every target this crate runs on, and the difference in what they
    /// MEAN is the point: `Ty::rust`'s spelling belongs to a declaration whose
    /// other side is a C compiler, and there is no C here. What the caller's
    /// value has to satisfy is `ArgValue::I32`.
    const fn rust(self) -> &'static str {
        match self {
            Crossing::PtrConst => "*const ::core::ffi::c_void",
            Crossing::PtrMut => "*mut ::core::ffi::c_void",
            Crossing::I32 => "i32",
            Crossing::U32 => "u32",
            Crossing::F32 => "f32",
            Crossing::Usize => "usize",
            Crossing::I64 => "i64",
            Crossing::Bool => "bool",
            Crossing::U8 => "u8",
        }
    }

    /// The `ArgValue` construction that binds a parameter named `name`.
    ///
    /// A read-only pointer is cast at the call rather than declared mutable:
    /// `ArgValue::Ptr` takes `*mut c_void` and the cast is where the row's
    /// claim about constness stops being expressible. Declaring the parameter
    /// `*mut` instead would push that cast onto the CALLER, which is the one
    /// place it means something — a caller holding a read-only mapping would
    /// have had to write it and would have had nothing to write it from.
    fn bind(self, name: &str) -> String {
        match self {
            Crossing::PtrConst => {
                format!("crate::runtime::ArgValue::Ptr({name} as *mut ::core::ffi::c_void)")
            }
            Crossing::PtrMut => format!("crate::runtime::ArgValue::Ptr({name})"),
            Crossing::I32 => format!("crate::runtime::ArgValue::I32({name})"),
            Crossing::U32 => format!("crate::runtime::ArgValue::U32({name})"),
            Crossing::F32 => format!("crate::runtime::ArgValue::F32({name})"),
            Crossing::Usize => format!("crate::runtime::ArgValue::Usize({name})"),
            Crossing::I64 => format!("crate::runtime::ArgValue::I64({name})"),
            Crossing::Bool => format!("crate::runtime::ArgValue::Bool({name})"),
            Crossing::U8 => format!("crate::runtime::ArgValue::U8({name})"),
        }
    }
}

/// How a [`Ty`] crosses, or nothing when `Args::bind` refuses it.
///
/// **This match is a copy of a decision made in
/// `crate::runtime::args`, and it must agree with it exactly.** That file's
/// `is_pointer` names the kinds that bind as `ArgValue::Ptr`; its
/// `Args::bind` names the six scalars — `I32`, `U32`, `F32`, `Usize`, `I64`,
/// `Bool` — and turns everything else into `ArgError::Unsupported`. A kind
/// this function admitted and that one did not would be a generated function
/// whose every call is refused at launch: the failure the whole emitter
/// exists to convert into a missing symbol.
///
/// The copy exists because it cannot be a reference. `crate::runtime` is
/// behind `_cuda` and needs `cudarc`, and this module is compiled into
/// `build.rs`, which has neither — so the list is restated and the test below
/// checks it, under the feature, against the real binder.
///
/// The two ways to drift are not symmetric, which is what makes the copy
/// tolerable. A kind added THERE and missing HERE makes a row absent from
/// `api.rs`: fail-safe, visible in a diff, and a compile error at whatever
/// wanted to call it — and measured, since that is exactly what happened when
/// `Ty::I64` and `Ty::Bool` joined the binder and six rows quietly stopped
/// having entry points. The reverse would be the runtime refusal.
///
/// The const/mut split is not `Args::bind`'s — it has none, a pointer is a
/// pointer there — but it is not invented here either: it is the outermost
/// mutability of [`Ty::rust`]'s own spelling, which is what the AOT bindings
/// declared and what the C++ header says.
const fn crossing(ty: Ty) -> Option<Crossing> {
    Some(match ty {
        Ty::Buf
        | Ty::I32s
        | Ty::I64s
        | Ty::U32s
        | Ty::U8s
        | Ty::U16s
        | Ty::I8s
        | Ty::Bf16s
        | Ty::F16s
        | Ty::F32s
        | Ty::BufArray
        | Ty::BufArrayMut
        | Ty::U8Array
        | Ty::I32Array => Crossing::PtrConst,
        Ty::BufMut
        | Ty::I32sMut
        | Ty::U32sMut
        | Ty::U8sMut
        | Ty::U16sMut
        | Ty::I8sMut
        | Ty::F32sMut
        | Ty::BufArrayOut
        | Ty::BufArrayOutMut => Crossing::PtrMut,
        Ty::I32 => Crossing::I32,
        Ty::U32 => Crossing::U32,
        Ty::F32 => Crossing::F32,
        Ty::Usize => Crossing::Usize,
        Ty::I64 => Crossing::I64,
        Ty::Bool => Crossing::Bool,
        Ty::KvScheme | Ty::KvDType => Crossing::U8,
        // A four-byte enumerator, so it rides the SAME crossing as any other
        // 32-bit scalar. There is no `Crossing::U32Enum` and there must not
        // be: a crossing says how many bytes go in the cell, a `Ty` says what
        // they mean, and `emit_device_typecheck`'s function-pointer
        // initialisation is what refuses a row that meant something else.
        Ty::Fp8Kind => Crossing::U32,
        // `Args::bind`'s own catch-all, spelled the same way. A kind that
        // arrives in `kernels` and is handled by neither is refused by both,
        // which is the agreement that matters.
        _ => return None,
    })
}

/// The function name a symbol flattens to.
///
/// The namespace prefix is KEPT. Two families may name a kernel the same
/// thing — `norm::tanh_bf16` has no more right to `tanh_bf16` than any other
/// family's would — and a flattened symbol is unique exactly because a symbol
/// is. `emit_c_shim` flattens the same way and for the same reason; what it
/// adds is a `pie_k_` prefix, which a Rust module path already provides.
fn fn_name(symbol: &str) -> String {
    identifier(&symbol.replace("::", "_"))
}

/// A parameter name, from the row author's spelling.
///
/// The row's own name, because it is what makes a call site readable against
/// the header and against the kernel — `reference`, `target_rms_out`, `eps`.
/// A keyword gets a trailing underscore rather than a rename: `type_` is
/// still the row's word, and a generated `arg2` would not be.
///
/// A SHOUTY name is kept too. Three rows call an extent `N`, `B` or `H`
/// because the `__global__` does, and lowercasing them here would be the one
/// rename this function refuses on the line above — a caller matching a call
/// against a kernel signature that says `int N` should find `N`. What that
/// costs is a `non_snake_case` warning per binding, which [`one`] silences on
/// the function it generated.
fn param_name(name: &str) -> String {
    let ident = identifier(name);
    if KEYWORDS.contains(&ident.as_str()) {
        format!("{ident}_")
    } else {
        ident
    }
}

/// Anything to a valid Rust identifier: non-alphanumerics become underscores,
/// and a leading digit gets one in front of it.
fn identifier(from: &str) -> String {
    let mut out: String = from
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect();
    if out.starts_with(|c: char| c.is_ascii_digit()) {
        out.insert(0, '_');
    }
    out
}

/// The words a parameter may not be called.
///
/// Every strict and reserved keyword through edition 2024, not the subset the
/// current rows would trip over: a row is written by whoever ports a family,
/// and the first one to name an operand `in` should get a compiling file
/// rather than a puzzle.
const KEYWORDS: &[&str] = &[
    "Self", "abstract", "as", "async", "await", "become", "box", "break", "const", "continue",
    "crate", "do", "dyn", "else", "enum", "extern", "false", "final", "fn", "for", "gen", "if",
    "impl", "in", "let", "loop", "macro", "match", "mod", "move", "mut", "override", "priv", "pub",
    "ref", "return", "self", "static", "struct", "super", "trait", "true", "try", "type", "typeof",
    "unsafe", "unsized", "use", "virtual", "where", "while", "yield",
];

/// The unit a row is fired through, as a doc line names it: the row's file
/// without its extension, which is what a [`Unit`](crate::unit::Unit) is
/// called.
///
/// Derived rather than looked up. [`crate::unit::unit_of`] is reachable —
/// [`all_rows`] calls into that module — and would answer the same string,
/// because [`Unit::name`](crate::unit::Unit::name) is its root's path without
/// the extension and `unit.rs`'s file/unit agreement test is what makes that
/// true for every row. So the lookup would be a scan of every unit's rows to
/// re-derive a stem this already has, and the day the two answers differ is
/// the day that test fails, which is a better place to learn it than a doc
/// line.
fn unit_name(file: Option<&'static str>) -> Option<&'static str> {
    file.map(|f| f.rsplit_once('.').map_or(f, |(stem, _)| stem))
}

/// One row's text: a function, or the comment that stands where it would
/// have been.
fn one(row: &'static DeviceKernel) -> String {
    let sig: &'static KernelSig = row.sig;
    // An empty operand list means the row has NOT BEEN WRITTEN -- see
    // `KernelSig::operands` -- and `emit_c_shim` skips such a row for exactly
    // that reason. Emitting a nullary launch here would also produce a `let
    // values = []` whose element type nothing determines, so the honest
    // answer and the compilable one are the same answer.
    if sig.operands.is_empty() {
        return format!(
            "// {}: not emitted -- the row states no operands. For most rows that is a row \
             that has not been written; for `crate::families::fa2`'s 460 it is the third way \
             a row loses its shim entry, and deliberate: the `__global__` takes ONE by-value \
             params struct, `Ty` has no variant for it, and a `Ty::Blob` would type-check \
             every wrong struct as readily as the right one. Those fire through \
             `KernelModule::fire_raw`.\n\n",
            sig.symbol
        );
    }

    let mut params = String::new();
    let mut values = String::new();
    let mut shouty = false;
    for operand in sig.operands {
        let Some(crossing) = crossing(operand.ty) else {
            return format!(
                "// {}: not emitted -- operand `{}` is `Ty::{:?}`, which `Args::bind` refuses.\n\n",
                sig.symbol, operand.name, operand.ty
            );
        };
        let name = param_name(operand.name);
        shouty |= name.contains(|c: char| c.is_ascii_uppercase());
        // A `nullable` operand is DOCUMENTED, not typed, for the reason
        // `emit_rust_bindings` gives: an `Option<NonNull<_>>` would change
        // what the caller passes to express a fact the KERNEL checks, and the
        // row is where that fact already lives.
        let note = if operand.nullable { "  // may be null" } else { "" };
        params.push_str(&format!("    {name}: {},{note}\n", crossing.rust()));
        values.push_str(&format!("        {},\n", crossing.bind(&name)));
    }

    let mut out = match unit_name(sig.file) {
        Some(unit) => format!("/// `{}` — fired through `{unit}`.\n", sig.symbol),
        None => format!("/// `{}` — fired through the unit that holds its row.\n", sig.symbol),
    };
    out.push_str(&instantiation_doc(row));
    out.push_str(SAFETY_DOC);
    // A row may spell an operand `N`, `B` or `H`, because the kernel does.
    // The row's word is kept -- see `param_name` -- so the lint is silenced
    // where the odd name is, per function and never module-wide: an
    // `#![allow]` over the whole file would also cover the next generated
    // item that deserved the warning. `kernels-cuda`'s `ffi.rs` needs none of
    // this, and not because its rows are tamer: a parameter of an
    // `unsafe extern "C"` declaration binds nothing, so `non_snake_case`
    // never looks at it. Here it is a real binding in a real body.
    if shouty {
        out.push_str("#[allow(non_snake_case)]\n");
    }
    out.push_str(&format!("pub unsafe fn {}(\n", fn_name(sig.symbol)));
    out.push_str(&params);
    // LAST, ALWAYS, AND IN THIS ORDER. Neither is an operand -- a stream is
    // `cuLaunchKernel`'s sixth parameter and `dims` is what the row's
    // `LaunchRule` reads -- so putting them after the row's own list keeps
    // every generated signature's prefix equal to the row.
    out.push_str("    dims: crate::runtime::Dims,\n");
    out.push_str("    stream: crate::runtime::Stream<'_>,\n");
    out.push_str(") -> ::core::result::Result<(), crate::runtime::Error> {\n");
    out.push_str("    let values = [\n");
    out.push_str(&values);
    out.push_str("    ];\n");
    out.push_str("    // SAFETY: the caller's obligations above are exactly `fire`'s.\n");
    out.push_str(&format!(
        "    unsafe {{ crate::runtime::fire(\"{}\", dims, &values, stream) }}\n",
        sig.symbol
    ));
    out.push_str("}\n\n");
    out
}

/// The doc line naming the C++ this function reaches, quoted whole.
///
/// # Why a generated function says which instantiation it fires
///
/// Because its parameters cannot. [`Crossing::rust`] spells every buffer
/// `*const c_void` or `*mut c_void` on purpose — a launch checks eight bytes
/// and nothing about a pointee — so the element type, which is the one thing
/// a caller has to get right and gets no help with, appears nowhere in the
/// signature. It is in the row, as `template_path` and `elem`, and this line
/// is where a caller reads it.
///
/// What the line states is the INSTANTIATION, and not one word about any
/// individual operand. The distinction was academic until an offline
/// `nvcc` typecheck of all 164 rows — one translation unit per unit,
/// `void (*const chk)(<operand types>) = &<instantiation>;`, which admits no
/// parameter conversions — found four units whose `__global__` takes buffers
/// of TWO element types: `mlp::gpt_oss_glu_bf16` and
/// `norm::rmsnorm_strided_bf16#vec8` both write `bf16* y` beside `f16*
/// y_fp16`, and `quant`'s pair store through `Fmt::store`. A row cannot say
/// so — [`Ty::Buf`] and `BufMut` take their element from the row's single
/// `elem`, and no `Ty` carries one of its own — so this line would be
/// LYING if it claimed every buffer were `elem`. It does not: it says which
/// template was instantiated, which is true of all 164, and the operand that
/// differs says so in its own name, because [`param_name`] keeps the row's
/// spelling and the row author called it `y_fp16`.
///
/// # Why the string is quoted and never split
///
/// [`DeviceKernel::elem`] may be an argument LIST rather than a type —
/// `"device::bf16, 256"` for a `template <class T, int BLOCK>` — and
/// [`crate::device::args`] records what the two slots measured under NVRTC
/// 13.0: slot 1 is prefixed with `::pie_cuda_driver::kernels::` and need not
/// be a type at all (`device::kBlock256` and `device::false_type::value` both
/// resolve), while slots 2 and after take bare literals at global scope. So
/// the head of an `elem` is the storage type for `rmsnorm<T, BLOCK>` and is a
/// `bool` VALUE for `rotate<kWriteKv, kHnd>`, and nothing in the string says
/// which.
///
/// `kernels_cuda::abi::emit_device_typecheck` had to split it — it builds a
/// C++ parameter type out of the head — and its first version pasted the
/// whole list into `const ::pie_cuda_driver::kernels::device::bf16, 256*`,
/// which is not C++; it takes the head deliberately now, and says why. This
/// emitter has no such need: a doc line is prose, so the row's two strings
/// are pasted verbatim and there is no reading of `elem` here to get wrong.
///
/// # The one row shape that is not `path<elem>`
///
/// A row whose kernel has NO template parameter list states
/// [`DeviceKernel::PLAIN`], and its name IS its path — see that constant for
/// the measurement. Pasting the sentinel between angle brackets would print
/// `Fires \`attn::device::write_mla<(no template arguments)>\``, which is a
/// doc line that says the opposite of what the row says, on a generated
/// function nobody can edit to fix. So the branch is here, and it is the same
/// branch [`DeviceKernel::instantiation`] makes, read off the same predicate.
fn instantiation_doc(row: &'static DeviceKernel) -> String {
    if row.is_plain() {
        return format!("///\n/// Fires `{}`, which takes no template arguments.\n", row.template_path);
    }
    format!("///\n/// Fires `{}<{}>`.\n", row.template_path, row.elem)
}

#[cfg(test)]
mod tests {
    use super::{Crossing, all_rows, crossing, emit_rust_api, fn_name};
    use kernels::{KernelSig, LaunchRule, Ty, kernel, operands};
    use crate::device::DeviceKernel;

    /// The name of every function in `text`, in emission order.
    ///
    /// Anchored on the line start, because the banner talks ABOUT
    /// `pub unsafe fn` and a scan that matched prose would count it.
    fn emitted_names(text: &str) -> Vec<&str> {
        text.match_indices("\npub unsafe fn ")
            .map(|(at, marker)| {
                let rest = &text[at + marker.len()..];
                &rest[..rest.find('(').expect("a function is followed by its parameter list")]
            })
            .collect()
    }

    /// Whether every operand of `sig` marshals, which is what makes a row
    /// emittable.
    fn emittable(sig: &KernelSig) -> bool {
        !sig.operands.is_empty() && sig.operands.iter().all(|o| crossing(o.ty).is_some())
    }

    /// The same rows produce the same bytes.
    ///
    /// A build script's output is an input to every dependent crate's
    /// compile, so text that varied run to run would rebuild the world on a
    /// no-op build and would drown a real change in `api.rs` in noise. Cheap
    /// to state, and it is the property a `HashMap` in this file would take
    /// away silently.
    #[test]
    fn the_same_rows_emit_the_same_text() {
        let rows = all_rows();
        assert_eq!(emit_rust_api(&rows), emit_rust_api(&rows));
    }

    /// Every emitted name is an identifier, and no two rows share one.
    ///
    /// A collision would be a duplicate `fn` in one module — caught by rustc,
    /// but only after `api.rs` is written and only in a file nobody edits. A
    /// non-identifier would be the same class of failure with a worse
    /// message, and it is what a symbol carrying anything but `[A-Za-z0-9_:]`
    /// would produce.
    #[test]
    fn every_emitted_name_is_an_identifier_and_names_one_row() {
        let text = emit_rust_api(&all_rows());
        let mut seen: Vec<&str> = Vec::new();
        for name in emitted_names(&text) {
            assert!(!name.is_empty(), "an empty function name was emitted");
            assert!(
                name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
                "`{name}` is not a Rust identifier"
            );
            assert!(
                !name.starts_with(|c: char| c.is_ascii_digit()),
                "`{name}` starts with a digit"
            );
            assert!(!seen.contains(&name), "`{name}` is emitted twice");
            seen.push(name);
        }
    }

    /// A row whose operands all marshal emits exactly one function, and
    /// nothing else emits any.
    ///
    /// Both halves matter. A row silently failing to emit is the defect
    /// `abi.rs` records costing two sessions — indistinguishable, from the
    /// call site, from a row that was never written — and a row emitting
    /// twice would be the collision above with the same name for two
    /// different operand lists.
    #[test]
    fn a_row_that_marshals_is_emitted_exactly_once() {
        let rows = all_rows();
        let text = emit_rust_api(&rows);
        let mut expected = 0;
        for row in &rows {
            if !emittable(row.sig) {
                continue;
            }
            expected += 1;
            let name = fn_name(row.sig.symbol);
            assert_eq!(
                text.matches(&format!("pub unsafe fn {name}(")).count(),
                1,
                "`{}` marshals and did not emit exactly one function",
                row.sig.symbol
            );
        }
        assert_eq!(emitted_names(&text).len(), expected);
    }

    /// `norm::scalar_mul_bf16` binds a pointer, a float and a count, in the
    /// row's order.
    ///
    /// The ORDER is the whole test. `Args::bind` checks kinds against the row
    /// position by position, so a generator that emitted the right three
    /// values in the wrong order would be refused at launch — and a row whose
    /// operands were all pointers would not even be refused, it would run
    /// with the arguments swapped. This is the row the switch was thrown on
    /// (`new-horizon.md` §10.8), which makes it the one worth pinning.
    #[test]
    fn scalar_mul_binds_its_operands_in_the_rows_order() {
        let text = emit_rust_api(&all_rows());
        let at = text
            .find("pub unsafe fn norm_scalar_mul_bf16(")
            .expect("the row is in `ELEMENTWISE`");
        let body = &text[at..];
        let mut from = 0;
        for wanted in [
            "crate::runtime::ArgValue::Ptr(x)",
            "crate::runtime::ArgValue::F32(s)",
            "crate::runtime::ArgValue::Usize(n)",
        ] {
            let found = body[from..]
                .find(wanted)
                .unwrap_or_else(|| panic!("`{wanted}` is missing or out of order in:\n{body}"));
            from += found + wanted.len();
        }
    }

    /// No function name carries a namespace separator.
    ///
    /// `norm::scalar_mul_bf16` is a symbol and `norm_scalar_mul_bf16` is an
    /// identifier; a name that kept the `::` would parse as a path and the
    /// generated file would not compile at all.
    #[test]
    fn a_flattened_symbol_carries_no_namespace_separator() {
        let text = emit_rust_api(&all_rows());
        for name in emitted_names(&text) {
            assert!(!name.contains(':'), "`{name}` still carries a namespace separator");
        }
    }

    /// The kinds `Args::bind` marshals cross, as the same kind and with the
    /// constness the row already declares.
    ///
    /// The pointer list is copied from `is_pointer` in
    /// `crate::runtime::args`, which is the source of truth, and the copy is
    /// the point: a kind that binder takes and this emitter refuses costs an
    /// absent function, while the reverse costs a refusal on every launch of
    /// a row that looked callable. This states the list from memory of the
    /// binder; the test after it asks the binder itself, and only under the
    /// feature that compiles it.
    #[test]
    fn every_kind_the_binder_marshals_crosses_the_same_way() {
        const POINTERS: &[Ty] = &[
            Ty::Buf,
            Ty::BufMut,
            Ty::I32s,
            Ty::I32sMut,
            Ty::I64s,
            Ty::U32s,
            Ty::U32sMut,
            Ty::U8s,
            Ty::U8sMut,
            Ty::U16s,
            Ty::U16sMut,
            Ty::I8s,
            Ty::I8sMut,
            Ty::Bf16s,
            Ty::F16s,
            Ty::F32s,
            Ty::F32sMut,
            Ty::BufArray,
            Ty::BufArrayMut,
            Ty::BufArrayOut,
            Ty::BufArrayOutMut,
            Ty::U8Array,
            Ty::I32Array,
        ];
        for &ty in POINTERS {
            let crossing = crossing(ty)
                .unwrap_or_else(|| panic!("`Args::bind` binds {ty:?} and this emitter refuses it"));
            assert!(
                matches!(crossing, Crossing::PtrConst | Crossing::PtrMut),
                "{ty:?} binds as a pointer and crosses as {crossing:?}"
            );
            // The row's own spelling decides which: `Ty::rust` is what the
            // ahead-of-time bindings declared, so a caller ported from one to
            // the other passes the same pointer without adding a cast.
            let outer = if ty.rust().starts_with("*const") { "*const" } else { "*mut" };
            assert!(
                crossing.rust().starts_with(outer),
                "{ty:?} is `{}` and crosses as `{}`",
                ty.rust(),
                crossing.rust()
            );
        }
        for (ty, want) in [
            (Ty::I32, Crossing::I32),
            (Ty::U32, Crossing::U32),
            (Ty::F32, Crossing::F32),
            (Ty::Usize, Crossing::Usize),
            (Ty::I64, Crossing::I64),
            (Ty::Bool, Crossing::Bool),
            // Both by-value enums cross as ONE kind, deliberately: `Crossing`
            // says how a value is marshalled and `Ty` says what it means. The
            // swap the two `Ty`s exist to catch is caught in the C++, by
            // `abi::emit_device_typecheck`'s function-pointer initialisation.
            (Ty::KvScheme, Crossing::U8),
            (Ty::KvDType, Crossing::U8),
        ] {
            assert_eq!(crossing(ty), Some(want), "{ty:?} crosses as the wrong scalar");
        }
    }

    /// The copy above, asked of the binder rather than of memory.
    ///
    /// Every kind this emitter crosses is bound by `Args::bind` — checked by
    /// BINDING one, against a synthetic single-operand row, with the value
    /// the generated call would construct. That is the expensive direction:
    /// a kind admitted here and refused there is a function whose every call
    /// fails at launch, and nothing else in this file can see it, because the
    /// binder lives behind `_cuda` and this module is also compiled into
    /// `build.rs`.
    ///
    /// It cannot check the cheap direction. `Ty` has no iterator, and a list
    /// of "every kind the binder takes" written here would be the same copy
    /// again with an extra step. What that direction costs is an absent
    /// function — it cost six of them when `I64` and `Bool` joined the binder
    /// — and the coverage test above is what makes that visible.
    #[cfg(feature = "_cuda")]
    #[test]
    fn every_kind_this_emitter_crosses_is_one_the_binder_binds() {
        use crate::runtime::{ArgValue, Args};

        static SIGS: [KernelSig; 10] = [
            kernel!(one "test::buf", operands = operands![a: Buf]),
            kernel!(one "test::buf_mut", operands = operands![a: BufMut]),
            kernel!(one "test::i32", operands = operands![a: I32]),
            kernel!(one "test::u32", operands = operands![a: U32]),
            kernel!(one "test::f32", operands = operands![a: F32]),
            kernel!(one "test::usize", operands = operands![a: Usize]),
            kernel!(one "test::i64", operands = operands![a: I64]),
            kernel!(one "test::bool", operands = operands![a: Bool]),
            kernel!(one "test::kv_scheme", operands = operands![a: KvScheme]),
            kernel!(one "test::kv_dtype", operands = operands![a: KvDType]),
        ];
        let values = [
            ArgValue::Ptr(core::ptr::null_mut()),
            ArgValue::Ptr(core::ptr::null_mut()),
            ArgValue::I32(0),
            ArgValue::U32(0),
            ArgValue::F32(0.0),
            ArgValue::Usize(0),
            ArgValue::I64(0),
            ArgValue::Bool(false),
            ArgValue::U8(0),
            ArgValue::U8(0),
        ];
        for (sig, value) in SIGS.iter().zip(values) {
            let ty = sig.operands[0].ty;
            assert!(crossing(ty).is_some(), "{ty:?} is bindable and this emitter refuses it");
            assert!(
                Args::bind(sig, &[value]).is_ok(),
                "{ty:?} crosses in `api.rs` and `Args::bind` refuses it"
            );
        }
    }

    /// A row carrying a kind the binder refuses is absent, and the comment
    /// says which operand made it so.
    ///
    /// `Ty::Stream`, which is the refusal with a meaning: a stream is
    /// `cuLaunchKernel`'s sixth parameter and a row that still lists one has
    /// not been ported, so the comment says "unported" in the only place a
    /// caller would look.
    ///
    /// Synthetic, because no row this crate compiles carries one — every
    /// device row is pointers and scalars, which is what a `__global__`
    /// takes. It was written against `Ty::Bool` and had to move when the
    /// binder learned to marshal a one-byte flag: a refusal test pinned to a
    /// kind that stops being refused stops testing anything, and it passes
    /// while it does.
    #[test]
    fn a_row_the_binder_would_refuse_is_absent_and_says_why() {
        static SIG: KernelSig = kernel!(refused "norm::refused_for_the_test",
            file = Some("norm/elementwise.cuh"),
            launch = LaunchRule::Elementwise,
            operands = operands![x: BufMut, stream: Stream]);
        static ROW: DeviceKernel = DeviceKernel {
            sig: &SIG,
            template_path: "norm::device::refused_for_the_test",
            elem: "device::bf16",
        };

        let text = emit_rust_api(&[&ROW]);
        assert!(
            emitted_names(&text).is_empty(),
            "a row with a `Ty::Stream` operand emitted a function:\n{text}"
        );
        assert!(
            text.contains(
                "// norm::refused_for_the_test: not emitted -- operand `stream` is \
                 `Ty::Stream`, which `Args::bind` refuses."
            ),
            "the refusal does not name the operand and the type:\n{text}"
        );
    }

    /// A row that has not stated its operands is absent too, and for the
    /// other reason.
    ///
    /// `KernelSig::operands` is empty when a row has not been written, not
    /// when a kernel takes nothing — a `__global__` with no parameters is not
    /// a thing any family has — so emitting a nullary function would invent a
    /// contract out of a blank.
    #[test]
    fn an_unstated_row_is_absent() {
        static SIG: KernelSig = kernel!(unstated "norm::unstated_for_the_test",
            file = Some("norm/elementwise.cuh"),
            launch = LaunchRule::Elementwise);
        static ROW: DeviceKernel = DeviceKernel {
            sig: &SIG,
            template_path: "norm::device::unstated_for_the_test",
            elem: "device::bf16",
        };

        let text = emit_rust_api(&[&ROW]);
        assert!(emitted_names(&text).is_empty(), "an unstated row emitted a function:\n{text}");
        assert!(text.contains("norm::unstated_for_the_test: not emitted"), "{text}");
    }

    /// The rows this emitter reads are exactly the rows the units compile.
    ///
    /// # Why equality, again
    ///
    /// It asserted equality while every row in the crate lived in
    /// `kernels_cuda::norm_device`; it was weakened to one direction when the
    /// families began declaring their own rows in `src/families/*.rs`, on the
    /// reasoning that this file is `#[path]`-included by `build.rs` and
    /// therefore cannot see a row that lives in this crate. The reasoning was
    /// right about the mechanism and wrong about the conclusion — a build
    /// script can include the families too, which `kernels-cuda/build.rs` had
    /// been doing with fourteen modules the whole time — and the weakened
    /// check is what let the façade sit at ten rows of a hundred and
    /// thirty-five without a test going red.
    ///
    /// Both directions are worth their line, and they fail differently. A row
    /// this emitter emits and no unit compiles is a `pub unsafe fn` whose
    /// every call is an unknown symbol at fire time — a call into nothing. A
    /// row some unit compiles and this emitter does not see is the defect
    /// above: a kernel reachable only by string, with the typed façade
    /// silently not covering it, which is exactly what a façade is for.
    ///
    /// [`all_rows`] is `crate::unit::rows()`, so equality holds by
    /// construction today and the test is here for the day something reads a
    /// second table again.
    #[test]
    fn the_rows_this_emitter_reads_are_the_rows_the_units_compile() {
        let hosted: Vec<&str> = crate::unit::UNITS
            .iter()
            .flat_map(|unit| unit.rows.iter().map(|row| row.sig.symbol))
            .collect();
        let read: Vec<&str> = all_rows().iter().map(|row| row.sig.symbol).collect();
        for symbol in &read {
            assert!(
                hosted.contains(symbol),
                "`{symbol}` gets a generated entry point and no unit compiles it"
            );
        }
        for symbol in &hosted {
            assert!(
                read.contains(symbol),
                "`{symbol}` is a row some unit compiles and this emitter never sees it"
            );
        }
        assert_eq!(read, hosted, "the emitter's order is the units' order");
    }

    /// The façade covers every row, not a subset of them.
    ///
    /// The count is derived, never written down: a literal would be edited to
    /// agree with whatever the emitter did next, which is how the ten-row
    /// façade survived the migration. What is asserted is the shape of the
    /// answer — a row is a function or a comment that names why it is not,
    /// and the two add up to the table — plus the one number a reader wants
    /// from a coverage test: that the functions are not a handful.
    #[test]
    fn every_row_is_a_function_or_a_stated_refusal() {
        let rows = all_rows();
        let text = emit_rust_api(&rows);
        let emitted = emitted_names(&text).len();
        let refused = rows.iter().filter(|row| !emittable(row.sig)).count();
        assert_eq!(
            emitted + refused,
            rows.len(),
            "{} rows, {emitted} functions and {refused} refusals",
            rows.len()
        );
        for row in &rows {
            if emittable(row.sig) {
                continue;
            }
            assert!(
                text.contains(&format!("// {}: not emitted", row.sig.symbol)),
                "`{}` is neither emitted nor accounted for",
                row.sig.symbol
            );
        }
        assert!(
            emitted > rows.len() / 2,
            "{emitted} of {} rows reach the typed surface, which is a façade nobody \
             can rely on",
            rows.len()
        );
    }

    /// A generated function names the instantiation it fires, with the row's
    /// `elem` quoted whole.
    ///
    /// The multi-argument case is the one that matters and it is why the
    /// assertion is on a `<class T, int BLOCK>` row: `elem` is
    /// `"device::bf16, 256"` there, and the defect this pins is the one
    /// `abi::emit_device_typecheck` shipped — a generator that pasted an
    /// argument list where a single argument was expected, producing
    /// `const ::pie_cuda_driver::kernels::device::bf16, 256*`. Nothing here
    /// splits `elem`, so the check is that the whole of it survives into the
    /// text between the angle brackets.
    #[test]
    fn a_generated_function_names_its_instantiation() {
        let text = emit_rust_api(&all_rows());
        for row in all_rows() {
            if !emittable(row.sig) {
                continue;
            }
            let wanted = if row.is_plain() {
                format!("/// Fires `{}`, which takes no template arguments.", row.template_path)
            } else {
                format!("/// Fires `{}<{}>`.", row.template_path, row.elem)
            };
            assert!(
                text.contains(&wanted),
                "`{}` does not name its instantiation as `{wanted}`",
                row.sig.symbol
            );
        }
        assert!(
            all_rows().iter().any(|row| row.elem.contains(',')),
            "no row states an argument list, so this test proves nothing -- see \
             `crate::device::args`"
        );
        // And the other end of the same claim: a plain row exists, so the
        // branch above is exercised rather than merely written.
        assert!(
            all_rows().iter().any(|row| row.is_plain()),
            "no row names a plain `__global__`, so the plain branch proves nothing \
             -- see `crate::device::DeviceKernel::PLAIN`"
        );
        assert!(
            !text.contains(DeviceKernel::PLAIN),
            "the sentinel reached the generated text, which means some line pasted \
             `elem` without asking `is_plain` first"
        );
    }

    /// A function with a capitalised parameter carries the `allow`, and one
    /// without does not.
    ///
    /// The lint is warn-by-default and this crate's library compiles clean,
    /// so three rows spelling an extent `N`, `B` or `H` would otherwise put
    /// three warnings in a file nobody can edit to fix — the state where a
    /// reader learns to skip the crate's warnings, which is how a real one
    /// gets missed. The second half of the assertion is the one that keeps
    /// this honest: a blanket `allow` on every generated item would pass the
    /// first half and would hide the next lint that mattered.
    #[test]
    fn a_shouty_parameter_is_allowed_where_it_is_and_nowhere_else() {
        static SHOUTY_SIG: KernelSig = kernel!(shouty "norm::shouty_for_the_test",
            file = Some("norm/elementwise.cuh"),
            launch = LaunchRule::Elementwise,
            operands = operands![x: BufMut, N: I32]);
        static QUIET_SIG: KernelSig = kernel!(quiet "norm::quiet_for_the_test",
            file = Some("norm/elementwise.cuh"),
            launch = LaunchRule::Elementwise,
            operands = operands![x: BufMut, n: I32]);
        static SHOUTY: DeviceKernel = DeviceKernel {
            sig: &SHOUTY_SIG,
            template_path: "norm::device::t",
            elem: "device::bf16",
        };
        static QUIET: DeviceKernel = DeviceKernel {
            sig: &QUIET_SIG,
            template_path: "norm::device::t",
            elem: "device::bf16",
        };

        let text = emit_rust_api(&[&SHOUTY, &QUIET]);
        let at = text.find("pub unsafe fn norm_shouty_for_the_test(").expect("emitted");
        assert!(
            text[..at].ends_with("#[allow(non_snake_case)]\n"),
            "a capitalised parameter did not get the allow:\n{text}"
        );
        assert_eq!(
            text.matches("#[allow(non_snake_case)]").count(),
            1,
            "the allow reached a function whose parameters are all snake case:\n{text}"
        );
    }
}
