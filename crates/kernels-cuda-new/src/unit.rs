//! A unit: one NVRTC compile, and the rows that say what to instantiate out
//! of it.
//!
//! # Why a unit is a thing at all
//!
//! §6.4 of `new-horizon.md` fixes the granularity of run-time compilation:
//! **one compile per unit, many name expressions per compile.** Both
//! neighbours of that are worse. A compile per kernel pays NVRTC's fixed cost
//! — parse, template instantiation, ptxas — once per row, on a table that has
//! hundreds; a compile of the whole tree makes every edit invalidate every
//! kernel, which is the disk cache's problem rather than the compiler's but is
//! a problem all the same. A unit is the middle: a file's worth of templates,
//! compiled together because they share a preamble and change together.
//!
//! # A unit is DATA, and that is the layer boundary
//!
//! This module names units and nothing else — no `nvrtcCreateProgram`, no
//! cubin, no module. That is not tidiness: it is what lets `model-compiler`,
//! `build.rs` and an offline cache builder read the unit list on a machine
//! with no CUDA at all. The compile lives in [`crate::runtime`], behind the
//! feature, and takes a [`Unit`] the way it takes an architecture.
//!
//! The shape this replaces is worth naming, because it is the one a second
//! unit always tempts: `driver-cuda`'s Tier A had exactly one family, so
//! `Family` was a unit struct whose associated functions reached for a `SOURCE`
//! const and an `ENTRIES` table through `use` — which made the second family a
//! COPY of the first's methods with different constants in them rather than a
//! row in a table. `UNITS` is that table, and adding a family is adding a line
//! to it.
//!
//! # The three fields were not all of it
//!
//! This header used to say that a unit needing a fact not listed on [`Unit`]
//! was a signal the fact belonged in [`DEVICE_HEADERS`], which every unit
//! shares. Two facts turned out not to be shareable, and three separate
//! audits of this crate found them independently, which is the evidence they
//! are real:
//!
//! * **which compiler may compile a unit.** `moe_grouped_gemm_tile.cuh` is
//!   finished, measured and exact, and needs NVRTC 13.3 for `cuda::tiles`;
//!   this box loads 13.0. Nothing on `Unit` could say so, so the file is
//!   carried as text and declared as no unit at all.
//! * **which header set it resolves against.** A unit compiling upstream
//!   FlashInfer needs [`crate::source::UPSTREAM`], which [`DEVICE_HEADERS`]
//!   deliberately excludes because NVRTC copies every byte of every header it
//!   is handed and a `norm` kernel has no business paying for an attention
//!   library.
//!
//! [`Unit::options`] is the wrong hook for either, and its own doc says why:
//! an option is a flag passed to ONE NVRTC, and *"needs 13.3"* is a statement
//! about WHICH NVRTC. Spelling a floor as an option would be refused by 13.0
//! as an unknown flag — a diagnostic about a flag rather than about a
//! compiler, which is the wrong failure.
//!
//! So [`Toolchain`] and [`Headers`] are the two facts, [`Demands`] is the
//! pair, and [`Unit::demands`] is how a unit states them. Both are in the
//! cache key for the reason `program::cache` already gives.

use std::fmt;

use crate::device::DeviceKernel;
use crate::source::{self, ALL_HEADERS, DEVICE_HEADERS, Header};

/// One compilable unit: a root source, and the instantiations wanted out of
/// it.
///
/// Four fields and two more facts, which are read through [`Unit::demands`]
/// rather than spelled here — see [`DEMANDS`] for the whole of why, and for
/// the two-line edit that turns them back into fields.
#[derive(Clone, Copy)]
pub struct Unit {
    /// The unit's name, which is its root's path under `csrc/src` without the
    /// extension.
    ///
    /// Derived rather than chosen, so that `KernelSig::file`, the `#include`
    /// spelling and the unit's own name are one string with an extension on
    /// or off it. The alternative — a made-up `pie_norm_device.cu`, which is
    /// what `driver-cuda` carries — means a row and the unit that compiles it
    /// agree on nothing a test can compare.
    pub name: &'static str,
    /// The root source, handed to `nvrtcCreateProgram`.
    ///
    /// Carried in the binary by [`include_str!`], via [`roots`], which is what
    /// makes a moved file a compile error here rather than a missing one at
    /// run time.
    pub root: &'static str,
    /// The instantiations this compile is asked for.
    ///
    /// A row states a template path and an element type; the compile turns
    /// each into a `nvrtcAddNameExpression` and back into a mangled symbol.
    /// A row in the wrong unit therefore fails to compile — the template is
    /// not in that root — which is what the file/unit agreement test below
    /// establishes without a GPU.
    pub rows: &'static [DeviceKernel],
    /// NVRTC options this unit needs and the others must not have.
    ///
    /// Empty for everything authored here, and that emptiness is the point:
    /// the shared options in `runtime::nvrtc::options` are a CONTRACT — the
    /// float flags are in the cache key because they decide the arithmetic —
    /// and a flag that only one unit needs must not silently become everyone's.
    ///
    /// The case that made this a field rather than a constant is
    /// `--device-as-default-execution-space`. Internalised upstream source
    /// needs it: FlashInfer's headers are full of functions with no
    /// `__device__`
    /// annotation, which nvcc forgives inside a `.cu` and NVRTC does not, and
    /// without the flag `decode.cuh` is rejected at the first one. Turning it
    /// on globally would be the wrong fix twice over — it would silently
    /// compile OUR unannotated host helpers onto the device rather than
    /// reporting them (the `yarn_original_ramp_bounds` defect, which the flag
    /// would have hidden instead of surfaced), and it would change the
    /// meaning of every existing unit's source without changing a line of it.
    ///
    /// In the cache key, necessarily: a cubin compiled with a different option
    /// set is a different cubin, and `program::cache`'s header records in the
    /// past tense what happens when a key spans less than what produced the
    /// entry.
    pub options: &'static [&'static str],
}

/// The oldest NVRTC that may compile a unit, as NVRTC reports itself:
/// `nvrtcVersion` fills a major and a minor, and 13.3 is `(13, 3)`.
///
/// # Why a version and not a flag
///
/// `Unit::options` is the neighbouring hook and it is the wrong one. An option
/// is text handed to ONE NVRTC; a floor is a statement about WHICH NVRTC. Put
/// `-std=c++20 --enable-tile` on a unit and ask 13.0 to compile it and the
/// answer is *"unknown option"* — a diagnostic about a flag, when the fact is
/// that this machine's compiler does not have the feature the flag turns on.
/// A floor is checked before `nvrtcCreateProgram` and answers with both
/// numbers, so the reader learns what to install rather than what to delete.
///
/// # Why it is not a `u32` of `major * 1000 + minor * 10`
///
/// That is CUDA's own `CUDART_VERSION` spelling and it would make the ordering
/// free. It would also make every message read `13030`, and the message is
/// half of what this type is for: *"skipped, needs 13.3, have 13.0"* is a
/// sentence an operator can act on. The ordering is derived instead — `major`
/// is declared before `minor`, so `Ord` compares them in that order, which is
/// exactly version order for the two-field form NVRTC hands back.
///
/// The patch level is deliberately absent. `nvrtcVersion` does not report one
/// — 13.0.88 answers `(13, 0)` — so a floor that named one could never be
/// checked, and a field that cannot be checked is a comment with a type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Toolchain {
    /// NVRTC's major version: the `13` of 13.3.
    pub major: u32,
    /// NVRTC's minor version: the `3` of 13.3.
    pub minor: u32,
}

impl Toolchain {
    /// No floor at all — every unit authored here, today.
    ///
    /// The default rather than an `Option<Toolchain>`, so that the check in
    /// `runtime::nvrtc` has one shape rather than two. It also earns a real
    /// property: a unit that states no floor never calls `nvrtcVersion`, so
    /// the compile path of a crate that carries no floored unit makes no new
    /// FFI call at all, and the no-GPU tests that assert a refusal without
    /// `libnvrtc.so` on the machine keep working.
    pub const ANY: Self = Self { major: 0, minor: 0 };

    /// A floor of `major.minor`.
    #[must_use]
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    /// Whether this is [`Toolchain::ANY`] — nothing to check, nothing to ask.
    #[must_use]
    pub const fn is_any(self) -> bool {
        self.major == 0 && self.minor == 0
    }

    /// Whether an NVRTC reporting `have` may compile a unit whose floor is
    /// `self`.
    ///
    /// Spelled out rather than `have >= self`, because `PartialOrd` is not
    /// callable in a `const fn` and this is the comparison the whole type
    /// exists for: it belongs where it can be read, not derived at a call
    /// site. Inclusive at the floor — *"needs 13.3"* means 13.3 compiles it.
    #[must_use]
    pub const fn met_by(self, have: Self) -> bool {
        have.major > self.major || (have.major == self.major && have.minor >= self.minor)
    }
}

impl fmt::Display for Toolchain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_any() {
            f.write_str("any")
        } else {
            write!(f, "{}.{}", self.major, self.minor)
        }
    }
}

/// Which carried header set a unit's `#include`s resolve against.
///
/// A named CHOICE and not a `&'static [Header]` field, for two reasons that
/// both bite:
///
/// * a set is 2.7 MB of text and a key must name it in a few bytes. The choice
///   has a [`Headers::tag`]; a slice would have to be digested to be named,
///   which the key already does for the CONTENT — and the content is not the
///   same claim as the choice. See [`Unit::cache_key_under`].
/// * the sets are the crate's, not a unit's. `source`'s whole argument is that
///   there is one carried library and one upstream closure; a unit that could
///   name an arbitrary set could name a private header, which is the state
///   that module exists to prevent.
///
/// An angle include is NOT answered here and cannot be. `<crt/cuda_tile.h>` is
/// the compiler's own bundled header — `new-horizon.md` §20.6 records that
/// NVRTC 13.3 exports `nvrtcInstallBundledHeaders` / `nvrtcGetBundledHeadersInfo`
/// to make one resolvable at all — so a unit that
/// spells one is making a claim on the TOOLCHAIN, and [`Toolchain`] is where
/// that claim goes. The two fields are separate because those are two facts;
/// `moe_grouped_gemm_tile` happens to need both.
///
/// The choice also does not reach the OTHER thing that decides what a name
/// resolves to: the include-path spelling an offline compile of these same
/// headers uses. `source`'s header explains what `-I` instead of `-iquote`
/// silently does to `<cuda_fp16.h>`, with the two objects measured. Nothing
/// here can state that, and a second set does not make it statable.
///
/// # Where this is going, and what it is not yet
///
/// Two arms is a DICHOTOMY, and it existed for exactly one reason: FlashInfer
/// arrived wholesale as somebody else's tree, so the only question a unit was
/// ever asked was *do you want the vendored library or not*. That is a
/// question about provenance.
///
/// `csrc/` is being re-cut so that a directory answers *what is this text
/// for* instead — [`crate::source::SHIM`] is the first cut and landed with
/// this comment. When `csrc/device/` and `csrc/attn/` follow, the useful
/// question becomes *which roles does this unit need*: an `attn` unit takes
/// `device` + `attn`, a `norm` unit takes `device` + `norm`, and neither
/// carries the other's bytes. Two arms becomes a per-unit SUBSET, and the
/// 1.2 MB that a `norm` kernel pays for today because it is in one bag
/// stops being one bag.
///
/// It is not that yet, and this comment is not pretending otherwise. Both
/// arms below gained [`crate::source::SHIM`] and neither lost anything: the
/// sets are the same files with the same names, regrouped.
///
/// # The provenance question outlived the provenance directory
///
/// The paragraph above used to end *"a subset needs the device stdlib
/// separated from the attention algorithms inside `csrc/vendor/flashinfer/`,
/// and separating them rewrites upstream's `#include` lines"*. Internalising
/// moved that tree to `csrc/src/attn/flashinfer/` and `csrc/src/attn/xqa/`
/// **without rewriting one upstream byte** — the subtree kept its shape, so
/// its fifty-eight relative directives kept resolving. `csrc/vendor` is gone
/// and the transform in `tests/upstream_manifest.rs` is still the identity.
///
/// So the split is now a PREFIX and not a directory: [`crate::source::LIBRARY`]
/// is `csrc/src` minus `attn/flashinfer/` and `attn/xqa/`, and
/// [`crate::source::UPSTREAM`] is exactly those two. That is a weaker thing
/// than a directory — a rule stated in `carried.rs` rather than a fact a
/// walker can see — and it buys the property that made it worth it: the
/// closure can be re-patched from upstream by replacing a subtree.
///
/// The dichotomy survives because the REASON survives. It is still 1.2 MB, a
/// `norm` kernel still has no business copying it through
/// `nvrtcCreateProgram`, and the arm still names who wrote the text. The
/// subset this comment wants is still not built; what changed is that the
/// separation it asks for is now a change to one `const` in `carried.rs`
/// rather than a change to somebody else's `#include` lines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Headers {
    /// [`DEVICE_HEADERS`]: `csrc/shim` and `csrc/src` minus the internalised
    /// subtrees — the impersonating headers, the prelude, and every kernel
    /// header authored here. What every unit authored here compiles against.
    Library,
    /// [`ALL_HEADERS`]: the above plus `csrc/src/attn/flashinfer` and
    /// `csrc/src/attn/xqa`, the patched FlashInfer and XQA closure — 1.2 MB,
    /// and NVRTC copies every byte of it, which is why it goes only to the
    /// units that ask.
    LibraryAndUpstream,
}

impl Headers {
    /// The set itself, as `nvrtcCreateProgram` will be handed it.
    #[must_use]
    pub const fn set(self) -> &'static [Header] {
        match self {
            Headers::Library => DEVICE_HEADERS,
            Headers::LibraryAndUpstream => ALL_HEADERS,
        }
    }

    /// The choice's name in a cache key.
    ///
    /// Short and stable: it is written into every key, and a key is compared
    /// as bytes.
    ///
    /// `"lib+vendor"` until the closure was internalised. Renaming it
    /// INVALIDATES the cached cubins of the four units that carry it, which
    /// is a cold recompile and not a correctness problem — and it is the
    /// honest outcome, because those four units' text moved. A tag that had
    /// stayed `"lib+vendor"` would have kept the old cubins addressable under
    /// a name for a directory that no longer exists.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Headers::Library => "lib",
            Headers::LibraryAndUpstream => "lib+upstream",
        }
    }
}

/// What a unit demands of the machine that compiles it, and of the text it is
/// compiled against.
///
/// One struct rather than two loose parameters, because the two travel
/// together everywhere they go: a compile takes both, a cache key spans both,
/// and the gate reports on both. It is `Copy` and two words wide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Demands {
    /// The oldest NVRTC that may compile this unit.
    pub floor: Toolchain,
    /// The header set its `#include`s resolve against.
    pub headers: Headers,
}

impl Demands {
    /// What a unit demands when it says nothing: any compiler, the library
    /// set.
    ///
    /// Which is what all 44 units declared today demand, and the reason this
    /// change moves no cubin: `Toolchain::ANY` skips the version query
    /// entirely and `Headers::Library` is the set `compile` already passed.
    pub const DEFAULT: Self = Self { floor: Toolchain::ANY, headers: Headers::Library };
}

/// The units that demand something other than [`Demands::DEFAULT`].
///
/// **Three entries, two of them prefixes, and the first covers 56 units** —
/// see the prefix rule below.
///
/// It was empty until the FA2 lattice landed, and that emptiness was a
/// measurement rather than a placeholder: every unit declared before it
/// compiles under the NVRTC this crate loads and resolves against the library
/// set. `crate::families::fa2` is the first that does not. Its 56 units
/// `#include "attn/flashinfer/attention/decode.cuh"` and
/// `"attn/flashinfer/attention/prefill.cuh"`, which reach the whole 1.2 MB
/// patched upstream closure, so they demand [`Headers::LibraryAndUpstream`].
///
/// `crate::families::cascade` is the second. Its one
/// unit `#include`s `"attn/flashinfer/attention/cascade.cuh"` — the OTHER
/// half of FA2's split path, the fold that turns `tmp_v`/`tmp_s` into `o` —
/// which reaches `cp_async.cuh`, `math.cuh`, `utils.cuh` and `state.cuh`, all
/// upstream's. `crate::x::xqa` is the third and is the entry the XQA
/// lattice's own doc predicted: five units on one root that reaches
/// `csrc/src/attn/xqa/`'s fifteen-file closure. Every other unit still
/// demands nothing.
///
/// The spellings above were `<flashinfer/…>` and `<xqa/…>` until
/// internalisation, and they are quoted now for a reason worth stating here
/// because this is the doc a reader checks them against:
/// [`crate::source::quoted_includes`] parses only `"…"`, so an angled
/// spelling is invisible to [`crate::source::reachable`] and to
/// `every_device_include_resolves`. Quoting these four roots is what puts
/// them under the gate that proves the closure still resolves.
///
/// The one file that would ALSO be here — `moe/moe_grouped_gemm_tile` — is
/// deliberately not a unit yet, and adding it here without adding the unit
/// would be a demand about nothing. The long note below is about that unit and
/// is unchanged.
///
/// # A key ending in `*` is a PREFIX, and why that is not vocabulary growth
///
/// `crate::families::fa2`'s unit names are macro-generated from four axes:
/// `attn/fa2_decode_hd128_g4`, `attn/fa2_prefill_hd512_q32_kv1`, 56 of them.
/// Listing every one here would be a table that has to be edited whenever the
/// lattice's arithmetic selects a different point — which is a table that
/// silently stops covering a unit, i.e. the exact failure the `const`
/// assertion below exists to prevent, reintroduced one level up.
///
/// So one entry states the family: `"attn/fa2_*"`. The `*` is a suffix on the
/// KEY and nothing else — there is no glob, no character class and no second
/// wildcard position — because the only thing being expressed is *"a family's
/// units demand what the family demands"*, and a family is a name prefix in
/// this crate by construction ([`crate::families`]'s header says so).
///
/// The `const` assertion is kept honest across the change: a prefix entry must
/// match **at least one** unit, so a renamed family is still a compile error.
///
/// # Why a table beside [`UNITS`] and not two fields on [`Unit`]
///
/// Two fields is what this wants to be, and the day it can be, it should. The
/// obstacle is not taste: `Unit` is constructed by 45 struct literals in
/// `src/families/*.rs` and a handful more in tests and examples, every one of
/// them spelling all four fields, so a fifth is a compile error in forty-five
/// declarations across thirteen files that this change does not own. A seam
/// that cannot be landed is worth less than a seam one indirection away from
/// where it belongs.
///
/// (The FA2 lattice is the counter-example that makes the point sharper rather
/// than weaker: its 56 units are ONE macro, so a fifth field would cost it one
/// line. It is the other 45 that keep the table here.)
///
/// So the statement is here, next to the concatenation that builds [`UNITS`],
/// and nothing outside this file reads it: every caller goes through
/// [`Unit::demands`], [`Unit::floor`] or [`Unit::header_set`]. Turning it into
/// fields later moves this table's contents into the literals and deletes
/// three functions' bodies; no call site changes.
///
/// # The one way a name table can lie, and the half of it that is checked
///
/// A demand naming a unit that does not exist is a **compile error**, by the
/// `const` assertion below — so a rename cannot silently strand a floor. The
/// other direction cannot be checked by any mechanism, table or field: a unit
/// that needs a floor and states none is indistinguishable from one that needs
/// nothing. It fails safe, which is the reason to be calm about it — NVRTC
/// rejects the source it cannot compile, loudly, in `tests/units.rs`, which is
/// exactly the failure this whole seam exists to convert into a skip once the
/// demand is stated.
///
/// # Where "fails safe" does not hold, which is the unit this table is for
///
/// The paragraph above is true of every unit declared today and false of
/// `moe/moe_grouped_gemm_tile`, so stating a floor for it is **necessary and
/// not sufficient**. Measured on this box, with NVRTC 13.3.33 and a bf16 tile
/// `mma`:
///
/// ```text
///   nvrtcCompileProgram      rc = 0
///   nvrtcGetCUBIN            47,560 bytes, .note.nv.tkinfo and NO .text
///   cuModuleLoadData         SUCCESS
///   cuModuleGetFunction      CUDA_ERROR_NOT_FOUND
/// ```
///
/// A tile kernel does not compile to SASS. NVRTC emits **Tile IR** and
/// something downstream must assemble it: either a driver new enough to do it
/// at `cuModuleLoadData` — this box's 580.159.03 is not, and loads the image
/// without assembling, which is why the module has no entry point — or
/// `tileiras`, run over `nvrtcGetTileIR`'s output before the cubin is cached.
///
/// So a floored tile unit under a 13.3 NVRTC compiles clean, caches a cubin,
/// loads it, and fails at the FIRST LAUNCH rather than at the compile. That is
/// the one shape this crate's gates cannot see, and it is the shape the one
/// unit this table was built for has.
///
/// What closes it is a step, not a field: `nvrtcGetTileIR` →
/// `tileiras --host-arch --host-os -arch=sm_NN` → the cubin that is cached.
/// **`tileiras` requires `CUDA_ROOT` in its environment and does not say so**
/// — without it every input fails with a bare `error: failed to compile Tile
/// IR program`, including nvcc's own `.tilebc`. With it the whole path runs
/// and the result is exact; end-to-end cold cost is 0.62-0.71 s, of which
/// `tileiras` is 0.18 s, which is what `program::cache` is for.
///
/// `tileiras` ships in `nvidia-cuda-tileiras` as a 95 MB binary with no
/// library form, so this is a subprocess and a packaging decision rather than
/// another `dlopen`. `.wiki/driver/new-horizon.md` §23.18 has the bisect that
/// found `CUDA_ROOT` and the eight-step transcript.
const DEMANDS: &[(&str, Demands)] = &[
    (
        "attn/fa2_*",
        Demands { floor: Toolchain::ANY, headers: Headers::LibraryAndUpstream },
    ),
    // The second entry, and the first that is not a family prefix.
    //
    // `crate::families::cascade`'s one unit `#include`s
    // `"attn/flashinfer/attention/cascade.cuh"`, which reaches `cp_async.cuh`,
    // `math.cuh`, `utils.cuh` and `state.cuh` — upstream's, every one of
    // them, and unreachable from `Headers::Library`. It is the same demand
    // `attn/fa2_*` makes and it is stated separately rather than widened into
    // a `*` that would cover both, because the two are different families and
    // a shared wildcard would be a key that stops naming what it covers.
    //
    // An EXACT key, so the `const` assertion below checks it the strongest
    // way it can: one unit, one name, and a rename is a compile error rather
    // than a prefix that silently matches nothing else.
    (
        "cascade/merge_states",
        Demands { floor: Toolchain::ANY, headers: Headers::LibraryAndUpstream },
    ),
    // The third, and the second family prefix.
    //
    // `crate::x::xqa`'s five enrolled units all compile ONE root,
    // `csrc/src/attn/attention_xqa_mha.cuh`, whose two includes are
    // `<cuda_bf16.h>` and `"attn/xqa/mha.cuh"` — and the second reaches the
    // fifteen-file, 275 KB closure at `csrc/src/attn/xqa/`. Every byte of it
    // is upstream's, none of it is reachable from `Headers::Library`, and
    // `families::attn::XQA_LATTICE`'s own doc names this entry as the one
    // the table would gain: *"Every member needs
    // `crate::unit::Headers::LibraryAndVendor`, and that is the one entry
    // the currently-empty `DEMANDS` table would gain."* The table was not
    // empty by the time it was written, which changes the sentence and not
    // the demand — and the variant it names is `LibraryAndUpstream` now,
    // which is a stale quotation in a file this task does not own and is
    // reported rather than edited.
    //
    // That root's THIRD include is `<attn/xqa/mha_sm90.cuh>`, angled, inside
    // `#if USE_SM90_MHA`. It is the one carried file that names a `csrc/`
    // path and is not in any set, and the bracket is how it says so.
    //
    // A PREFIX and not five exact keys, for `attn/fa2_*`'s reason with one
    // addition: the sixth member of the lattice is deliberately NOT enrolled
    // (see `XQA_LATTICE`'s last entry, "NOT READY"), and a prefix is the one
    // spelling that is already correct on the day it is. Five exact keys
    // would have to be edited to six, and the `const` assertion below cannot
    // see a key that was never added.
    (
        "attn/attention_xqa_mha_*",
        Demands { floor: Toolchain::ANY, headers: Headers::LibraryAndUpstream },
    ),
    // The fourth, and the second exact key.
    //
    // `crate::x::attn::mla_fa2`'s one unit compiles
    // `csrc/src/attn/attention_mla_fa2.cuh`, whose two upstream includes are
    // `"attn/flashinfer/attention/mla.cuh"` and
    // `"attn/flashinfer/attention/mla_params.cuh"`,
    // and the first of those includes `prefill.cuh` at its line 33 — so the
    // closure is `attn/fa2_*`'s, reaching `cascade.cuh`, `scheduler.cuh`,
    // `permuted_smem.cuh`, `fastdiv.cuh` and the rest. None of it is
    // reachable from `Headers::Library`.
    //
    // EXACT and not folded into a widened `attn/*`, for `cascade/merge_states`'
    // reason: one unit, one name, and a rename is a compile error rather than
    // a prefix that silently matches nothing else. It is deliberately NOT
    // merged with `attn/fa2_*` either, even though the two demand the same
    // set — they are different families with different roots, and a shared
    // key would stop naming what it covers.
    (
        "attn/attention_mla_fa2",
        Demands { floor: Toolchain::ANY, headers: Headers::LibraryAndUpstream },
    ),
];

/// Whether a [`DEMANDS`] key applies to a unit name.
///
/// Exact, unless the key ends in `*`, in which case the rest of it is a
/// prefix. See [`DEMANDS`] for why the one wildcard exists and why it is a
/// suffix on the key rather than a pattern language.
const fn demand_covers(key: &str, unit: &str) -> bool {
    let key_bytes = key.as_bytes();
    if key_bytes.is_empty() {
        return false;
    }
    if key_bytes[key_bytes.len() - 1] != b'*' {
        return str_eq(key, unit);
    }
    let prefix = key_bytes.len() - 1;
    let unit_bytes = unit.as_bytes();
    if unit_bytes.len() < prefix {
        return false;
    }
    let mut i = 0;
    while i < prefix {
        if key_bytes[i] != unit_bytes[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Every demand names a unit that exists.
///
/// Evaluated at compile time by the assertion below it. Reading
/// [`concat_families`] rather than [`UNITS`] because a `const` may not read a
/// `static`, and the two are the same array by construction.
const fn every_demand_names_a_unit() -> bool {
    let units = concat_families();
    let mut d = 0;
    while d < DEMANDS.len() {
        let mut found = false;
        let mut u = 0;
        while u < units.len() {
            if demand_covers(DEMANDS[d].0, units[u].name) {
                found = true;
            }
            u += 1;
        }
        if !found {
            return false;
        }
        d += 1;
    }
    true
}

const _: () = assert!(
    every_demand_names_a_unit(),
    "a demand names a unit that is not in UNITS -- a floor or a header set \
     keyed on a name nothing answers to is a demand that never applies"
);

/// `==` on `&str`, in a `const fn`.
///
/// `str::eq` is not const, and the alternative to eight lines here is a table
/// checked at run time by a test — which is a weaker guarantee for the same
/// information.
const fn str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

impl Unit {
    /// What this unit demands of the compiler and of the header set.
    ///
    /// [`Demands::DEFAULT`] for a unit that states nothing, which is every
    /// unit except `crate::families::fa2`'s 56. A linear scan of [`DEMANDS`],
    /// which is one entry long: this runs once per compile, not once per fire.
    #[must_use]
    pub fn demands(&self) -> Demands {
        DEMANDS
            .iter()
            .find(|(key, _)| demand_covers(key, self.name))
            .map_or(Demands::DEFAULT, |(_, demands)| *demands)
    }

    /// The oldest NVRTC that may compile this unit.
    ///
    /// Read by `runtime::nvrtc::compile_with` BEFORE the program is created,
    /// and by `tests/units.rs` to decide between compiling a unit and
    /// declining it by name. Those two must agree, which is why both ask the
    /// unit rather than each holding a list.
    #[must_use]
    pub fn floor(&self) -> Toolchain {
        self.demands().floor
    }

    /// Which set this unit's `#include`s resolve against.
    #[must_use]
    pub fn headers(&self) -> Headers {
        self.demands().headers
    }

    /// That set's text, as a compile is handed it.
    ///
    /// The thing `nvrtc::compile` passes and the thing
    /// `every_include_reachable_from_a_unit_resolves` resolves against — one
    /// function, so a unit cannot be checked against one set and compiled
    /// against another.
    #[must_use]
    pub fn header_set(&self) -> &'static [Header] {
        self.headers().set()
    }

    /// The row this unit holds for `symbol`, if it holds one.
    ///
    /// `&'static` because the rows are, and because the caller is a launch
    /// path that reads the row's `KernelSig` after the lookup has gone out of
    /// scope.
    #[must_use]
    pub fn row(&self, symbol: &str) -> Option<&'static DeviceKernel> {
        self.rows.iter().find(|row| row.sig.symbol == symbol)
    }

    /// Whether this unit is the one that would compile `symbol`.
    #[must_use]
    pub fn hosts(&self, symbol: &str) -> bool {
        self.row(symbol).is_some()
    }

    /// Every row's instantiation, in the table's order.
    ///
    /// Order is load-bearing: a compile adds one name expression per element
    /// here and asks `nvrtcGetLoweredName` for each afterwards, so the
    /// mangled names come back POSITIONALLY paired with these rows. It is also
    /// what the cache key folds, for the reason [`Unit::cache_key`] gives.
    #[must_use]
    pub fn instantiations(&self) -> Vec<String> {
        self.rows.iter().map(DeviceKernel::instantiation).collect()
    }

    /// The key a compiled unit may be cached under, for `arch` and the
    /// header set this unit asked for.
    ///
    /// **Everything that can change the cubin is in it, and that is the whole
    /// specification.** `driver-cuda/src/program/cache.rs` records what the
    /// alternative costs, in the past tense: a cubin keyed on less than what
    /// produced it is served after the thing it was not keyed on changes —
    /// kernel edits appeared to do nothing, and the model answered fluently
    /// out of the previous cubin. `driver-metal/src/program/cache.rs` keys its
    /// pipelines on the RESOLVED text for the same reason.
    ///
    /// The resolved text is not one file. NVRTC resolves
    /// `#include "pie_device.cuh"` against [`Unit::header_set`], so an edit to
    /// a header changes what compiles while leaving [`Unit::root`]
    /// byte-identical — exactly the shape of a stale-cache bug, and the reason
    /// [`source::digest`] exists.
    ///
    /// Eight components, each because it can move on its own: the unit name,
    /// the architecture (a cubin is per-`sm_XY`), the float contract, the
    /// unit's own options, its toolchain floor, its header-set CHOICE, a
    /// fingerprint of the root text, the header digest, and the instantiation
    /// list — a row added to a unit is a symbol the old cubin does not
    /// contain, which a key over the text alone cannot see.
    ///
    /// # Why the floor and the choice are in it, given that the digest is
    ///
    /// They are not the same claim, and the difference is what a stale-cache
    /// bug is made of.
    ///
    /// * The digest is the BYTES a compile was handed. The choice is which set
    ///   the unit ASKED for, which is what `nvrtc::compile` reads to decide
    ///   what to hand it. Two units whose sets happen to agree byte for byte
    ///   today would key the same on the digest alone, and stop agreeing the
    ///   moment an upstream header lands — after the cubin was written.
    /// * The floor is not in the text at all. It decides whether this machine
    ///   may produce the cubin, and a unit whose floor moved from `any` to
    ///   13.3 is a unit whose old entry was produced by a compiler it now
    ///   refuses.
    ///
    /// # What is NOT in it, and what an on-disk cache must add
    ///
    /// The NVRTC version this process actually loaded. The floor is a property
    /// of the UNIT and belongs here; the loaded version is a property of the
    /// MACHINE, and layer 2 may not depend on `cudarc` and cannot ask for it —
    /// which is the same reason [`FLOAT_CONTRACT`] is spelled twice. It does
    /// not matter for the in-process cache, where the loaded NVRTC cannot
    /// change under a running program. It matters completely for a cache on
    /// disk: two NVRTC versions compile one source to different machine code,
    /// so anything persisting these keys must fold
    /// `driver-cuda/src/program/compile.rs::version`'s answer in beside them.
    ///
    /// # The float flags are spelled twice, deliberately
    ///
    /// [`FLOAT_CONTRACT`] restates what `runtime::nvrtc` passes NVRTC, in a
    /// layer that may not depend on `cudarc` and therefore cannot ask. That
    /// duplication is the price of the layering and it is paid on purpose:
    /// `--fmad=false` is what keeps a reduction's last bit, so a cubin built
    /// under different arithmetic is a different ANSWER and must not be served
    /// for this key. The flags there are written as a literal list, checked by
    /// a literal test, so changing one is an edit that prompts changing this.
    #[must_use]
    pub fn cache_key(&self, arch: &str) -> String {
        self.cache_key_with(arch, self.header_set())
    }

    /// [`Unit::cache_key`] over an arbitrary header set, so a test can show
    /// that changing a header changes the key.
    #[must_use]
    pub fn cache_key_with(&self, arch: &str, headers: &[Header]) -> String {
        self.cache_key_under(arch, headers, self.demands())
    }

    /// [`Unit::cache_key`] over an arbitrary header set AND an arbitrary
    /// demand.
    ///
    /// The seam that makes "the key spans the floor and the choice" a checked
    /// claim rather than a stated one. It was written when [`DEMANDS`] was
    /// empty, so without a way to hand a demand in the two new terms could
    /// have been dropped from the format string with every test still
    /// passing; the table has two entries now and neither varies a FLOOR, so
    /// half of that hole is still open and this is still the only thing that
    /// closes it. It is the same seam `runtime::nvrtc::compile_under` is, for
    /// the same reason and about the same two facts.
    #[must_use]
    pub fn cache_key_under(&self, arch: &str, headers: &[Header], demands: Demands) -> String {
        let mut wanted = String::new();
        for instantiation in self.instantiations() {
            wanted.push_str(&instantiation);
            // The separator digest uses, for the reason digest uses it: two
            // row sets must not concatenate into one byte stream.
            wanted.push('\0');
        }
        format!(
            "jit/{}/{arch}/{FLOAT_CONTRACT}/{}/nvrtc>={}/{}/r{:016x}/h{:016x}/n{:016x}",
            self.name,
            self.options.join(","),
            demands.floor,
            demands.headers.tag(),
            source::fnv1a64(self.root.as_bytes()),
            source::digest(headers),
            source::fnv1a64(wanted.as_bytes()),
        )
    }
}

/// The float flags every compile in this crate is invoked with, as one string.
///
/// Layer 2's copy of `runtime::nvrtc`'s option list, which is where the flags
/// actually are. They are in a cache key because they decide the arithmetic —
/// §6.5 of `new-horizon.md` calls them a contract — and they are HERE because
/// a key that only layer 3 can compute is a key an offline cache builder
/// cannot check its entries against.
const FLOAT_CONTRACT: &str = "fmad=false,prec-div=true,prec-sqrt=true";

/// Every unit this crate can compile, in [`crate::families::ALL`]'s order.
///
/// The concatenation of the per-family lists. Order is not semantic — a unit's
/// position is its slot in the module cache and nothing depends on which slot
/// it gets — but it is stable, so a diff that migrates one family touches one
/// module and one line.
///
/// A `static` and not a function, because `unit_of` hands out `&'static Unit`
/// and `runtime::cache` indexes a fixed array of `OnceLock`s by position. A
/// `OnceLock<Vec<Unit>>` would give the same references and would put an
/// initialisation check on the launch path to express a fact that is known at
/// compile time.
pub static UNITS: &[Unit] = &concat_families();

/// `[&[Unit]] -> [Unit]` at compile time.
///
/// `Unit` is `Copy` — four `&'static` fields — which is what makes filling by
/// index legal in a const fn. `kernels-cuda/src/lib.rs` builds `KERNELS` the
/// same way for the same reason: the result must stay a `&'static [Unit]` for
/// every reader that already takes one, and neither `concat` nor iterator
/// chaining is const.
const fn concat_families() -> [Unit; UNIT_COUNT] {
    let mut out = [EMPTY_UNIT; UNIT_COUNT];
    let mut w = 0;
    let mut f = 0;
    while f < crate::families::ALL.len() {
        let family = crate::families::ALL[f];
        let mut i = 0;
        while i < family.len() {
            out[w] = family[i];
            w += 1;
            i += 1;
        }
        f += 1;
    }
    out
}

/// How many units every family declares, together.
const UNIT_COUNT: usize = count_units();

const fn count_units() -> usize {
    let mut n = 0;
    let mut f = 0;
    while f < crate::families::ALL.len() {
        n += crate::families::ALL[f].len();
        f += 1;
    }
    n
}

/// A slot to fill, never a unit anything can fire: it names no source and
/// holds no rows, so `unit_of` cannot return it and a compile of it would be
/// refused for having no instantiations.
const EMPTY_UNIT: Unit = Unit { name: "", root: "", rows: &[], options: &[] };

pub fn unit_of(symbol: &str) -> Option<(usize, &'static Unit)> {
    UNITS
        .iter()
        .enumerate()
        .find(|(_, unit)| unit.hosts(symbol))
}

/// Every row of every unit, in [`UNITS`]' order.
///
/// The JIT's whole table, for a caller that wants rows rather than compiles:
/// the emitter writing one arm per row, and a test asserting that a symbol
/// `model-compiler` can state is a symbol some unit will compile.
pub fn rows() -> impl Iterator<Item = &'static DeviceKernel> {
    UNITS.iter().flat_map(|unit| unit.rows.iter())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A row's `KernelSig::file` names the file its template is in, and a
    /// unit's root IS a file — so the two must agree, or the row is asking a
    /// compile for a template that compile has never seen.
    ///
    /// The failure this prevents is not hypothetical: a unit is a line in
    /// [`UNITS`] and a row is a line in a table, and nothing but this test
    /// stops the second `norm` unit from being handed the first's rows. On a
    /// GPU it surfaces as `NoLoweredName` at first fire; here it is a unit
    /// test on any machine.
    #[test]
    fn every_row_is_in_the_unit_its_file_names() {
        for unit in UNITS {
            let expected = format!("{}.cuh", unit.name);
            for row in unit.rows {
                assert_eq!(
                    row.sig.file,
                    Some(expected.as_str()),
                    "{} is in unit `{}`, whose root is `{expected}`",
                    row.sig.symbol,
                    unit.name
                );
            }
        }
    }

    /// [`unit_of`] is a function: one symbol, at most one unit.
    ///
    /// Two units claiming a symbol makes the answer depend on table order,
    /// which is to say a fire would launch whichever instantiation was
    /// declared first — and a test exercising the other one would still pass.
    #[test]
    fn no_symbol_is_hosted_by_two_units() {
        for row in rows() {
            let hosts: Vec<&str> = UNITS
                .iter()
                .filter(|unit| unit.hosts(row.sig.symbol))
                .map(|unit| unit.name)
                .collect();
            assert_eq!(hosts.len(), 1, "{} is hosted by {hosts:?}", row.sig.symbol);
            let (index, unit) = unit_of(row.sig.symbol).expect("a row's own unit");
            assert_eq!(unit.name, hosts[0]);
            assert_eq!(UNITS[index].name, hosts[0], "the index addresses the unit");
            assert_eq!(
                unit.row(row.sig.symbol).map(|found| found.sig.symbol),
                Some(row.sig.symbol)
            );
        }
    }

    /// The key spans everything that can change the cubin, checked one term
    /// at a time — because a key that ignores a term is indistinguishable
    /// from a correct one until the term moves.
    ///
    /// The unit under test was `families::norm::ALTUP_AUX` until §5 step 5
    /// took `norm` into fn-world. It is `x::norm::altup_aux::ALTUP_AUX` now
    /// — the same root, the same rows, the same key, emitted by a `unit!`
    /// invocation instead of written by hand. That is the point of naming a
    /// REAL unit here: the assertions below are about `cache_key`, and they
    /// hold across the move because nothing about the unit changed.
    #[test]
    fn the_cache_key_moves_when_anything_it_spans_does() {
        let unit = crate::x::norm::altup_aux::ALTUP_AUX;
        let base = unit.cache_key("sm_90");
        assert_eq!(base, unit.cache_key("sm_90"), "and is stable");
        assert_ne!(base, unit.cache_key("sm_80"), "a cubin is per-architecture");

        // The set as it is, with one header's text changed and nothing else,
        // built from the real one so a header added later is still covered.
        let mut edited = DEVICE_HEADERS.to_vec();
        edited[0].text = "// not what it was";
        assert_ne!(
            base,
            unit.cache_key_with("sm_90", &edited),
            "an edited header changes what compiles and leaves the root alone"
        );

        let fewer = Unit { rows: &unit.rows[..1], ..unit };
        assert_ne!(
            base,
            fewer.cache_key("sm_90"),
            "a row set is a symbol set, and an old cubin does not hold a new one"
        );

        let retitled = Unit { name: "norm/elsewhere", ..unit };
        assert_ne!(base, retitled.cache_key("sm_90"));

        assert!(
            base.contains(FLOAT_CONTRACT),
            "the arithmetic a cubin was built under is part of what it answers"
        );

        assert_ne!(
            crate::x::norm::altup_aux::ALTUP_AUX.cache_key("sm_90"),
            crate::x::norm::elementwise::ELEMENTWISE.cache_key("sm_90"),
            "two units, two roots, two keys"
        );
    }

    /// The two new terms are in the key, checked one at a time and through
    /// the seam that can vary them — because no entry in [`DEMANDS`] states a
    /// floor other than [`Toolchain::ANY`], so a key that dropped that term
    /// would pass every other test in this file.
    #[test]
    fn the_cache_key_moves_when_the_floor_or_the_header_choice_does() {
        let unit = crate::x::norm::altup_aux::ALTUP_AUX;
        let base = unit.cache_key("sm_90");
        assert_eq!(
            base,
            unit.cache_key_under("sm_90", DEVICE_HEADERS, Demands::DEFAULT),
            "the default demand is what every unit gets today, so it is the same key"
        );

        let floored = Demands { floor: Toolchain::new(13, 3), ..Demands::DEFAULT };
        assert_ne!(
            base,
            unit.cache_key_under("sm_90", DEVICE_HEADERS, floored),
            "a unit whose floor moved is a unit whose old cubin came from a \
             compiler it now refuses"
        );

        // The SAME text handed in, and only the choice different: this is the
        // term the header digest cannot express, because the digest is the
        // bytes and the choice is what the compile will go and fetch.
        let upstream = Demands { headers: Headers::LibraryAndUpstream, ..Demands::DEFAULT };
        assert_ne!(
            base,
            unit.cache_key_under("sm_90", DEVICE_HEADERS, upstream),
            "the header-set CHOICE is spanned, not merely the bytes it resolved to"
        );

        assert!(base.contains("nvrtc>=any"), "a unit with no floor says so in its key: {base}");
        assert!(base.contains("/lib/"), "and names the set it asked for: {base}");
        assert!(
            unit.cache_key_under("sm_90", DEVICE_HEADERS, floored).contains("nvrtc>=13.3"),
            "and a floored one says which"
        );
    }

    /// A floor orders by major and then minor, and is inclusive at the floor.
    ///
    /// Written out rather than trusting the derive, because the derive is
    /// correct only while `major` is declared before `minor` — a field
    /// reorder is a silent semantic change, and 13.0 vs 9.13 is exactly where
    /// it would show.
    #[test]
    fn a_floor_is_met_by_that_version_and_by_everything_after_it() {
        let needs = Toolchain::new(13, 3);
        assert!(needs.met_by(Toolchain::new(13, 3)), "the floor itself compiles it");
        assert!(needs.met_by(Toolchain::new(13, 4)));
        assert!(needs.met_by(Toolchain::new(14, 0)));
        assert!(!needs.met_by(Toolchain::new(13, 0)), "this box");
        assert!(!needs.met_by(Toolchain::new(12, 9)), "a bigger minor is not a bigger version");
        assert!(!needs.met_by(Toolchain::ANY), "and `any` is not a version at all");

        assert!(Toolchain::ANY.met_by(Toolchain::new(13, 0)), "no floor is met by anything");
        assert!(Toolchain::ANY.is_any());
        assert!(!Toolchain::new(13, 0).is_any());

        assert!(Toolchain::new(13, 0) < Toolchain::new(13, 3));
        assert!(Toolchain::new(9, 13) < Toolchain::new(13, 0));
        assert_eq!(Toolchain::new(13, 3).to_string(), "13.3");
        assert_eq!(Toolchain::ANY.to_string(), "any", "so a report reads as a sentence");
    }

    /// A unit states what it demands, and the table and the units agree.
    ///
    /// This asserted `DEMANDS.is_empty()` and that every unit demanded the
    /// default, which was true until the FA2 lattice landed and is not the
    /// property worth checking anyway. The property is that the two spellings
    /// of one fact agree: a unit demands the default **iff** no key covers it.
    /// That holds however many entries the table grows, and it fails for the
    /// one mistake this test can see — a key that matches a unit it was not
    /// written for, which a prefix entry makes possible.
    #[test]
    fn a_units_demand_is_exactly_what_the_table_says_about_it() {
        for unit in UNITS {
            let covered = DEMANDS.iter().any(|(key, _)| demand_covers(key, unit.name));
            if covered {
                assert_ne!(
                    unit.demands(),
                    Demands::DEFAULT,
                    "`{}` is covered by a key that states the default, which is a \
                     row saying nothing",
                    unit.name
                );
                continue;
            }
            assert_eq!(
                unit.demands(),
                Demands::DEFAULT,
                "`{}` states a demand no key covers",
                unit.name
            );
            assert!(unit.floor().is_any());
            assert_eq!(unit.headers(), Headers::Library);
            // Content, not pointer: `DEVICE_HEADERS` is a `const`, so each
            // use of it may be a separate copy of the same bytes.
            assert_eq!(unit.header_set(), DEVICE_HEADERS);
        }
    }

    /// The four demands that reach the upstream closure, and only those.
    ///
    /// Named rather than counted. A count is a number to bump; a name is a
    /// claim that fails when a fifth root starts `#include`ing
    /// `"attn/flashinfer/…"` without saying so — which is the failure
    /// [`DEMANDS`]' own doc calls the one direction no mechanism can check,
    /// caught here for the cases where it is checkable because the answer is
    /// written down.
    ///
    /// # This was two names against a four-row table, and it was wrong
    ///
    /// It read `the_vendored_closure_is_demanded_by_the_two_families_that_
    /// reach_it` and its predicate was `attn/fa2_*` OR `cascade/merge_states`
    /// — the table's first two rows, written when they were the only two.
    /// [`DEMANDS`] has had four rows since `crate::x::xqa`'s five units and
    /// `attn/attention_mla_fa2` enrolled, and `families::ALL` carries both,
    /// so six units answered `LibraryAndUpstream` to a predicate that said
    /// `false`. A test named after its own denominator goes stale the moment
    /// the denominator moves, and this one advertised the staleness in its
    /// name for the whole time it was failing.
    ///
    /// Found while renaming the variant, which is the only reason this
    /// assertion was read at all. It is not caused by internalisation and it
    /// is fixed here because the rename could not be applied without deciding
    /// what the line should say.
    ///
    /// The predicate now mirrors [`DEMANDS`] row for row, in its order, with
    /// `*` spelled as `starts_with` and an exact key as `==` — so a fifth row
    /// is a fifth clause and a reader can diff the two lists by eye.
    #[test]
    fn the_upstream_closure_is_demanded_by_exactly_the_roots_that_reach_it() {
        for unit in UNITS {
            let wants_upstream = unit.name.starts_with("attn/fa2_")
                || unit.name == crate::families::cascade::MERGE_STATES.name
                || unit.name.starts_with("attn/attention_xqa_mha_")
                || unit.name == "attn/attention_mla_fa2";
            assert_eq!(
                unit.headers() == Headers::LibraryAndUpstream,
                wants_upstream,
                "`{}`",
                unit.name
            );
        }
    }

    /// A demand is resolved by name, and the resolution is the identity every
    /// caller depends on.
    #[test]
    fn a_demand_is_resolved_by_the_units_own_name() {
        // An unknown name resolves to the default, which is the answer that
        // must never be a floor someone else stated.
        let unit = crate::x::norm::altup_aux::ALTUP_AUX;
        assert_eq!(Unit { name: "norm/nowhere", ..unit }.demands(), Demands::DEFAULT);

        // The two sets are two sets, and the choice reaches them.
        assert_eq!(Headers::Library.set(), DEVICE_HEADERS);
        assert_eq!(Headers::LibraryAndUpstream.set(), ALL_HEADERS);
        assert!(
            Headers::LibraryAndUpstream.set().len() > Headers::Library.set().len(),
            "the upstream closure is carried on top of the library, not instead of it"
        );
        assert_ne!(Headers::Library.tag(), Headers::LibraryAndUpstream.tag());
    }

    /// [`rows`] is the concatenation and nothing else — no unit skipped, no
    /// row visited twice, and the order [`UNITS`] states.
    ///
    /// The expectation is derived from [`crate::families::ALL`] rather than
    /// written out. An earlier version listed the pilot's two units by name,
    /// which was fine while two was the number and broke on the first family
    /// that migrated — for every family at once, since they all flow through
    /// the same concatenation. A test that has to be edited whenever the thing
    /// it checks grows is a test that will be edited to agree with a bug.
    #[test]
    fn rows_is_every_units_rows_in_order() {
        let flattened: Vec<&str> = rows().map(|row| row.sig.symbol).collect();
        let expected: Vec<&str> = crate::families::ALL
            .iter()
            .flat_map(|family| family.iter())
            .flat_map(|unit| unit.rows.iter())
            .map(|row| row.sig.symbol)
            .collect();
        assert_eq!(flattened, expected);
        assert_eq!(flattened.len(), UNITS.iter().map(|unit| unit.rows.len()).sum::<usize>());
    }

    /// An instantiation per row, in the row order the lowered names come back
    /// in.
    #[test]
    fn the_instantiations_are_the_rows() {
        for unit in UNITS {
            let asked = unit.instantiations();
            assert_eq!(asked.len(), unit.rows.len());
            for (at, row) in unit.rows.iter().enumerate() {
                assert_eq!(asked[at], row.instantiation());
            }
        }
    }
}
