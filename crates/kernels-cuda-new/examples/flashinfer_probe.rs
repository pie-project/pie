//! Does FlashInfer's decode path compile under NVRTC at all?
//!
//! # The question this answers
//!
//! Every other measurement in this crate has been about a header: does NVRTC
//! resolve `<cuda_fp8.h>` by itself (no), can `cooperative_groups.h` be shimmed
//! (yes), what does libcu++ cost if carried whole (13.7 MB). None of them
//! touched the thing the whole migration is FOR. `kernels-cuda` gets its
//! attention from FlashInfer, an ahead-of-time build compiles it with `nvcc -I`
//! against a directory tree, and nobody had ever handed that tree to NVRTC —
//! which has no directories at all, only [a header set carried in the
//! binary][crate::source].
//!
//! `.wiki/driver/new-horizon.md` §13.6 put a number on the unknown: walking the
//! includes of the fifteen FlashInfer headers `kernels-cuda/csrc/src` names
//! reaches **28 files and 17,981 lines**, and those 28 files make **31
//! directives that leave the tree** — of which, probed against an empty header
//! set, **0 were answered**. Not `<type_traits>`, not `<cstdint>`, not
//! `cuda_runtime.h`. The plan said: guard the host-only 23, shim the 4 device
//! headers, shim the 4 CCCL entries, and then find out.
//!
//! This is the finding out. It is the first compile of FlashInfer by NVRTC on
//! this machine, and its last line is the answer §13.6 asked for.
//!
//! # What it takes to ask the question
//!
//! Three things, and the report names each so a refusal can be attributed:
//!
//! * **The internalised closure** — `csrc/src/attn/flashinfer/`, 28 files
//!   copied byte-for-byte from FlashInfer v0.6.15 and modified only by
//!   `#ifndef __CUDACC_RTC__` guards, each marked `// PIE:` (see `NOTICE` and
//!   `csrc/src/attn/flashinfer/MODIFICATIONS`). It lived at `csrc/vendor/`
//!   until `csrc/` was narrowed to device text only; the subtree moved
//!   intact, so not one upstream byte changed and the manifest's four numeric
//!   columns are the same numbers they were.
//! * **Six carried headers of our own** — `cstdint`, `type_traits`, `cuda.h`,
//!   `cuda_runtime.h`, `bit`, `boost/math/ccmath/fabs.hpp`. Each exists because
//!   guarding its include was measured and REFUSED: the names in it reach
//!   device code. They are in `csrc/shim/` beside the other headers that
//!   answer for a compiler this one is not, and
//!   each says in its own banner what it replaces and why.
//! * **The device and CCCL headers**, which are somebody else's shims and are
//!   reported in two configurations below.
//!
//! The first two used to be a sixty-entry `include_str!` array in this file,
//! and its own comment called that a debt. `build.rs` now walks `csrc/` and
//! generates the set, so the probe reads [`source::UPSTREAM`],
//! [`source::DEVICE_HEADERS`] and [`source::ALL_HEADERS`] — **the same objects the
//! library hands NVRTC.** That is not tidiness: a probe whose header set is
//! narrower than the library's measures a configuration nothing runs, which is
//! precisely what `tests/prelude_parity.rs` caught when a sibling header became
//! a forwarder.
//!
//! The set carries every SPELLING, not every file, and that distinction was a
//! finding of the run before this one. NVRTC matches an `#include` literally —
//! FlashInfer reaches its own files as `../cp_async.cuh` and `mask.cuh` while
//! our sources reach them as `attn/flashinfer/cp_async.cuh` — so a set that
//! named each file once carried `attn/flashinfer/cp_async.cuh` present and
//! unreachable, and stopped at decode's first relative directive. This example derived the
//! missing spellings itself for one run; `build.rs` now reads each carried file
//! and registers it under every spelling that resolves to it, and refuses to
//! build when one spelling would reach two files. 34 upstream files, 66
//! entries. The probe takes the set as given, which is the point of a probe.
//!
//! [`source::UPSTREAM`]: kernels_cuda_new::source::UPSTREAM
//! [`source::DEVICE_HEADERS`]: kernels_cuda_new::source::DEVICE_HEADERS
//! [`source::ALL_HEADERS`]: kernels_cuda_new::source::ALL_HEADERS
//!
//! # Two configurations, because they answer two different questions
//!
//! **The crutch** runs first, and it is the one that answers §13.6: NVIDIA's
//! real `cuda_fp16.h`/`cuda_bf16.h`/`cuda_fp8.h`/`cuda_fp4.h` out of
//! `$CUDA_HOME`, and real CCCL for `cuda/*`. If FlashInfer refuses to compile
//! against NVIDIA's own headers, the source is the problem; if it compiles,
//! the source is acceptable and everything after is about the shims. Mixing the
//! two questions would produce one result that answers neither.
//!
//! **The shims** run second, with `csrc/shim/`'s hand-written replacements for
//! whatever exists at the moment this runs — the set is discovered rather than
//! assumed, because those files are being written by other hands and a probe
//! that hard-coded them would report on a snapshot. What has no shim yet falls
//! back to the crutch, and the report says which.
//!
//! Keeping the crutch after the shims reached parity is deliberate. It is the
//! only reason two cubins exist to hold against each other, and holding them is
//! what turned "28 of 28 acceptable" into a measured +178 `SHF.L.U32` on the
//! bf16 widening path — with bit-identical output, which is the other half of
//! that sentence and could not have been said without a baseline. A
//! configuration nothing ships earns its run time when it is the thing the
//! shipped one is measured against.
//!
//! # The crutch is a crutch, and it is marked as one
//!
//! Both configurations read headers from `$CUDA_HOME` **at run time**. That is
//! precisely what [`source::Header`] makes impossible for the library: its two
//! fields are `&'static str`, so a header that is not in the binary is not a
//! header as far as this crate is concerned. A probe is allowed to cheat where
//! a library is not, because a probe's output is a number and a library's
//! output is a running kernel. Nothing here is a proposal to read `$CUDA_HOME`
//! from the library.
//!
//! [`source::Header`]: kernels_cuda_new::source::Header
//!
//! ```text
//! cargo run -p kernels-cuda-new --features cuda-13 --example flashinfer_probe
//! ```

#[cfg(not(feature = "_cuda"))]
fn main() {
    // The gate is here rather than in `Cargo.toml`'s `[[example]]`, which this
    // file does not own: layers 1 and 2 build with no CUDA at all, and a probe
    // whose whole subject is what NVRTC will accept must not be what drags a
    // toolkit into a build that asked for the table.
    println!(
        "flashinfer_probe needs layer 3: cargo run -p kernels-cuda-new --features cuda-13 \
         --example flashinfer_probe"
    );
}

#[cfg(feature = "_cuda")]
fn main() {
    probe::run();
}

#[cfg(feature = "_cuda")]
mod probe {
    use std::borrow::Cow;
    use std::collections::{HashMap, HashSet, VecDeque};
    use std::ffi::{CStr, CString};
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use cudarc::nvrtc::sys as nv;
    use kernels_cuda_new::source::{self, Header};

    /// The internalised closure, and PIE's headers beside it, as the LIBRARY
    /// carries them.
    ///
    /// This was sixty `include_str!` entries written out by hand, and its own
    /// comment called that a debt. `build.rs` now walks `csrc/` and emits
    /// [`source::UPSTREAM`] — one entry per file under `csrc/src/attn/flashinfer`
    /// and `csrc/src/attn/xqa`, named by the path an `#include` would spell — so the twenty-eight FlashInfer files and
    /// the six headers PIE carries beside them are in this binary because they
    /// are on disk, not because a list in an example remembered them.
    ///
    /// Reading the generated set rather than a copy of it is the point and not a
    /// tidy-up: **a probe whose header set is narrower than the library's
    /// measures a configuration nothing runs.** `tests/prelude_parity.rs` caught
    /// precisely that when a sibling header became a forwarder and a hand-written
    /// list went on carrying the file it used to be.
    ///
    /// The split here is by prefix because the NAME is: `attn/flashinfer/…` is
    /// somebody else's tree — the closure this probe reports on — and
    /// `attn/xqa/…` is somebody else's OTHER tree, which this probe does not
    /// report on and which the filter therefore excludes.
    ///
    /// The prefix was already written and already correct, so this is a note
    /// and not a bug — but the note is worth having, because the numbers say
    /// something about the alias pass that is easy to get wrong.
    /// **Measured: `source::UPSTREAM` is 87 entries; 28 begin
    /// `attn/flashinfer/`, 15 begin `attn/xqa/`, and the other 44 begin with
    /// NEITHER.** Those 44 are the second spellings — `mask.cuh`,
    /// `../cp_async.cuh`, `defines.h` — and an alias is registered under the
    /// literal directive text, which has no tree prefix in it. So `closure()`
    /// yields the 28 FILES and none of their aliases, which is what its
    /// callers want (`audit` checks files against the manifest, and
    /// `compile_closure` compiles each file once) and is exactly what it
    /// yielded before the move, when the same filter said `flashinfer/`.
    ///
    /// # The name collision the alias pass could have had, and did not
    ///
    /// Both trees carry a `utils.cuh` and an `mma.cuh`. A bare `"utils.cuh"`
    /// from either tree's root would resolve `beside` the includer and
    /// register one spelling against two different files, which `collect`
    /// asserts against — the build would stop. It does not, because XQA
    /// already spells those five directives `"attn/xqa/utils.cuh"` and
    /// `"attn/xqa/mma.cuh"`, the five `// PIE:` markers the manifest records
    /// for exactly this reason. **Checked across every quoted include under
    /// `attn/xqa/`: fourteen distinct spellings, five prefixed, and not one
    /// of the other nine names a file FlashInfer also has.**
    fn closure() -> impl Iterator<Item = &'static Header> {
        source::UPSTREAM.iter().filter(|h| h.name.starts_with("attn/flashinfer/"))
    }

    /// The impersonating headers the closure resolves its non-FlashInfer
    /// names against.
    ///
    /// The headers PIE carries so that a guard did not have to lie.
    ///
    /// `cstdint`, `type_traits`, `cuda.h`, `cuda_runtime.h`, `bit`,
    /// `boost/math/ccmath/fabs.hpp`. Every one of them was a guard first, and
    /// every one of those guards was measured and refused.
    /// `#ifndef __CUDACC_RTC__` around `#include <cstdint>` compiles and then
    /// deletes `uint32_t` from 2,512 device declarations; around
    /// `<cuda_runtime.h>` it deletes `ushort`, which `math.cuh`'s
    /// `ex2.approx.f16` wrapper is written in; around `<cuda.h>` it silently
    /// unsets `CUDA_VERSION` and the fp4 vector types disappear with no
    /// diagnostic at all. A host header whose names reach device code is
    /// CARRIED, under the exact spelling the directive uses. That rule is the
    /// difference between 35 guards and roughly seventy.
    ///
    /// They used to sit at the root of the vendor tree, and this function used
    /// to find them by *not* being FlashInfer's — a definition by exclusion,
    /// which is what a provenance-shaped tree forces. Since `csrc/` was cut by
    /// ROLE they are in `csrc/shim/` beside the eight PIE wrote for its own
    /// text, and the same set is now named positively: [`source::SHIM`] IS the
    /// impersonation layer, so this is a filter for the ones this closure
    /// actually reaches rather than a subtraction.
    ///
    /// [`source::UPSTREAM`] therefore holds nothing of ours: two upstream
    /// closures and the second spellings their own relative directives
    /// register, and no third thing.
    fn carried() -> impl Iterator<Item = &'static Header> {
        // The six that arrived because FlashInfer asked, as opposed to the
        // eight in `SHIMMABLE` that PIE's own text asks for. Both sets are in
        // `csrc/shim/` now; the split here is about who needs them, which is
        // what this probe reports on.
        const UPSTREAM_SHIMS: &[&str] =
            &["bit", "boost/math/ccmath/fabs.hpp", "cstdint", "cuda.h", "cuda_runtime.h",
              "type_traits"];
        source::SHIM.iter().filter(|h| UPSTREAM_SHIMS.contains(&h.name))
    }

    /// One of `csrc/shim`'s headers, by the name an `#include` spells.
    ///
    /// Looked up in [`source::DEVICE_HEADERS`] rather than `include_str!`-ed, for the same
    /// reason the closure is: those files are other hands' work, and a probe that
    /// carried its own copy would report on the copy.
    fn shim(name: &str) -> Option<&'static Header> {
        source::DEVICE_HEADERS.iter().find(|h| h.name == name)
    }

    /// The names the closure reaches outside itself and outside the standard
    /// library, in the order a report should mention them.
    ///
    /// These eight are what `csrc/shim/` is expected to answer with hand-written
    /// shims. The probe looks for each one there at run time and falls back to the
    /// crutch, so the report is a statement about the moment it ran.
    const SHIMMABLE: &[&str] = &[
        "cuda_fp16.h",
        "cuda_bf16.h",
        "cuda_fp8.h",
        "cuda_fp4.h",
        "cooperative_groups.h",
        "cuda/std/limits",
        "cuda/cmath",
        "cuda/pipeline",
    ];

    /// One entry of a header set: the spelling, and the text behind it.
    ///
    /// Deliberately NOT [`source::Header`], whose fields are `&'static str`. That
    /// type says "carried in the binary" in the type system, and half of what this
    /// probe assembles is read off a disk at run time — so it would have to be
    /// leaked to fit, and leaking it would hide the distinction the report exists
    /// to draw.
    ///
    /// [`source::Header`]: kernels_cuda_new::source::Header
    struct Entry {
        /// What an `#include` must spell to reach this text.
        name: String,
        /// The text: `Cow::Borrowed` when it came from the binary,
        /// `Cow::Owned` when it came from a file — which is the crutch, visible.
        text: Cow<'static, str>,
    }

    /// What one configuration produced, kept for the closing summary.
    ///
    /// Both configurations must be reported side by side or the result is
    /// misleading: the crutch answers whether the SOURCE compiles, the shims
    /// answer whether the carried headers are complete, and printing only the last one run
    /// would silently substitute the second question for the first.
    struct Outcome {
        /// The configuration's name.
        what: &'static str,
        /// How many of the closure's files NVRTC accepted.
        acceptable: usize,
        /// Out of how many.
        of: usize,
        /// Header-set entries, every spelling counted.
        entries: usize,
        /// Header-set bytes.
        bytes: usize,
        /// The cubin, when the decode kernel instantiated.
        ///
        /// The IMAGE and not its size, because the two configurations compile
        /// one source with one instantiation and differ only in what answers
        /// four `#include`s — so if the shims are bit-exact replacements the
        /// bytes are the same bytes, and the comparison at the end of the run is
        /// the cheapest test of that claim there is. A size would answer
        /// "roughly the same amount of code"; this answers "the same code".
        cubin: Option<Vec<u8>>,
    }

    /// A header set, plus the provenance the report has to state.
    struct Config {
        /// What the report calls it.
        what: &'static str,
        /// The set itself, deduplicated by name — a repeated name makes
        /// `nvrtcCreateProgram` return `NVRTC_ERROR_INVALID_INPUT` before a single
        /// line is compiled, which cost an hour the first time it happened.
        entries: Vec<Entry>,
        /// Shim names taken from `csrc/shim/`.
        shimmed: Vec<String>,
        /// Names answered by a file read out of `$CUDA_HOME` or the CCCL in the
        /// build tree.
        crutched: Vec<String>,
        /// Names nothing answered. A refusal that mentions one of these is a
        /// missing shim, not a broken source.
        missing: Vec<String>,
        /// Entries taken from the library's own [`source::DEVICE_HEADERS`] — the
        /// prelude `csrc/shim`'s headers are written against, and the one part of the
        /// set that the shipped crate already carries verbatim.
        ///
        /// [`source::DEVICE_HEADERS`]: kernels_cuda_new::source::DEVICE_HEADERS
        library: Vec<String>,
    }

    /// `csrc/src/attn/flashinfer/MODIFICATIONS`, carried so it can be checked against the
    /// text it describes rather than believed.
    const MODIFICATIONS: &str = include_str!("../csrc/src/attn/flashinfer/MODIFICATIONS");

    /// Everything above, in the order the report prints it.
    pub fn run() {
        let arch = kernels_cuda_new::runtime::cache::arch().unwrap_or("sm_89");
        let cuda = cuda_home();

        println!("FlashInfer under NVRTC -- does the decode path compile at all?\n");
        println!("NVRTC version:  {}", version());
        println!("architecture:   {arch}");
        println!(
            "internalised:   {} closure files, {} lines, {} bytes",
            closure().count(),
            lines(),
            bytes()
        );
        // Entries against FILES, because they stopped being the same number when
        // the generator learned that NVRTC matches an include literally. The
        // group now spans both internalised trees: 43 files (28 FlashInfer, 15
        // XQA) under 87 names, the extra 44 being the relative spellings each
        // tree reaches its own siblings by. A probe that printed only one of
        // these two numbers would be hiding the mechanism that makes the tree
        // compile.
        let files: HashSet<&str> = source::UPSTREAM.iter().map(|h| h.text).collect();
        println!(
            "UPSTREAM:       {} entries, {} files -- {} closure, {} second spellings",
            source::UPSTREAM.len(),
            files.len(),
            closure().count(),
            source::UPSTREAM.len() - files.len()
        );
        println!(
            "SHIM:           {} headers, {} of them here because FlashInfer asks",
            source::SHIM.len(),
            carried().count()
        );
        println!("$CUDA_HOME:     {}", cuda.display());
        println!(
            "CCCL:           {}",
            cccl_root().map_or_else(|| "not found".to_string(), |p| p.display().to_string())
        );

        audit();

        let mut summary: Vec<Outcome> = Vec::new();
        for config in [crutch_config(&cuda), shim_config(&cuda)] {
            println!("\n{}", "-".repeat(78));
            println!("Configuration: {}", config.what);
            let bytes = config.entries.iter().map(|e| e.text.len()).sum::<usize>();
            println!("  header set:  {} entries, {bytes} bytes", config.entries.len());
            report_provenance("  shims:      ", &config.shimmed);
            report_provenance("  crutch:     ", &config.crutched);
            report_provenance("  missing:    ", &config.missing);
            report_provenance("  library:    ", &config.library);
            println!();

            let (acceptable, of) = compile_closure(&config, arch);
            let cubin = instantiate(&config, arch);
            summary.push(Outcome {
                what: config.what,
                acceptable,
                of,
                entries: config.entries.len(),
                bytes,
                cubin,
            });
        }

        println!("\n{}", "=".repeat(78));
        println!(
            "NVRTC {} on {arch}, against the {} vendored files -- {} lines, {} bytes of\n\
             somebody else's CUDA compiled with no include path on disk:",
            version(),
            closure().count(),
            lines(),
            bytes()
        );
        for run in &summary {
            println!("\n  {}", run.what);
            println!("      {} of {} closure files acceptable", run.acceptable, run.of);
            println!("      header set {} entries, {} bytes", run.entries, run.bytes);
            match &run.cubin {
                Some(image) => println!("      decode instantiates -- {} byte cubin", image.len()),
                None => println!("      decode does NOT instantiate"),
            }
        }
        compare(&summary);
    }

    /// Hold the two cubins against each other, which is the question the second
    /// configuration exists to answer.
    ///
    /// One source, one name expression, one architecture, one option list: the
    /// configurations differ in nothing but which text answers `cuda_fp16.h`,
    /// `cuda_bf16.h`, `cuda_fp8.h`, `cuda_fp4.h` and the three `cuda/*` doors. A
    /// shim that is a bit-exact replacement therefore produces a bit-exact cubin,
    /// and this is where that is checked rather than assumed — the alternative is
    /// a 28-of-28 that reads like a pass and a kernel that computes something
    /// else. A difference is not automatically a defect; it is the one result
    /// worth stopping on, because the shims' whole claim is sameness.
    ///
    /// **What a DIFFERENT verdict has meant, measured.** The shims produce a
    /// 32,496 byte cubin against the crutch's 125,760 — and almost none of that
    /// gap is the kernel. `cuobjdump -res-usage` gives both `REG:64 STACK:0
    /// SHARED:0 LOCAL:0 CONSTANT[0]:576`; the module's common section is
    /// `GLOBAL:1552 CONSTANT[4]:6360` against `GLOBAL:21 CONSTANT[4]:96`, and its
    /// symbol table 798 entries against 15. That is NVIDIA's headers' dead weight,
    /// not code this kernel runs. The SASS does differ, by 1,520 instructions
    /// against 1,368, and the delta is one thing: **+178 `SHF.L.U32`, −96 `PRMT`,
    /// +48 `MOV`**, which is `pie_device.cuh` widening bf16 with
    /// `__int_as_float(raw << 16)` one value at a time where `__bfloat1622float2`
    /// permutes a pair. Everything else moves by ones and twos — ptxas
    /// if-converting one `LDG.E.CONSTANT` that the crutch guards with `BSSY`/
    /// `BSYNC`, which is a scheduling consequence of the instruction mix, not a
    /// difference in what is computed.
    ///
    /// **More instructions, and the same numbers.** Both cubins were loaded and
    /// the instantiated kernel launched from each — 48 launches per configuration
    /// over identical input bytes, varying q/k/v and the KV length — and all
    /// 24,576 bf16 output lanes and 192 `lse` floats came back **bit-identical**.
    /// That is the spot check; the universal statement is
    /// `examples/halftype_parity.rs`, which holds every fp16 and bf16 function the
    /// shims define against NVIDIA's own over 39,847,842 inputs per path,
    /// including `__bfloat162float` and `__bfloat1622float2` across all 65,536
    /// bf16 patterns, and reports 43 of 43 rows bit-identical. The widening path
    /// is a different sequence of instructions computing the same function.
    ///
    /// **So compare the images, and when they differ read the resource usage
    /// before the size** — a size difference is mostly a statement about the
    /// header set, and only the SASS is a statement about the kernel.
    fn compare(summary: &[Outcome]) {
        let images: Vec<(&str, &Vec<u8>)> =
            summary.iter().filter_map(|o| o.cubin.as_ref().map(|c| (o.what, c))).collect();
        let [(_, left), (_, right)] = images[..] else {
            let refused: Vec<&str> =
                summary.iter().filter(|o| o.cubin.is_none()).map(|o| o.what).collect();
            println!("\n  no comparison -- no cubin from:");
            for what in refused {
                println!("      {what}");
            }
            return;
        };
        println!("\n  the same instantiation, both configurations:");
        if left == right {
            println!("      IDENTICAL -- {} bytes, byte for byte", left.len());
            println!("      the shims are not merely acceptable to NVRTC, they compile to");
            println!("      the same SASS NVIDIA's own headers do for this kernel");
        } else {
            println!("      DIFFERENT -- {} bytes against {} bytes", left.len(), right.len());
            println!("      one source, one instantiation, one architecture: the difference is");
            println!("      the shims. Read `cuobjdump -res-usage` before the size -- a gap of");
            println!("      this shape has been NVIDIA's headers' unused globals rather than");
            println!("      code this kernel runs, and the last time it was taken apart the two");
            println!("      cubins ran to bit-identical output. See `compare`'s doc comment.");
        }
    }

    /// Check `csrc/src/attn/flashinfer/MODIFICATIONS` against the text this binary carries.
    ///
    /// A hand-written inventory of somebody else's tree is a lie waiting for the
    /// next guard to be added, so it is not trusted here: the file is carried
    /// alongside the sources it describes and its two checkable columns -- guards
    /// and file length -- are recomputed from those sources every run. The third
    /// column, added lines, needs the upstream file to recompute and is left to
    /// the tool that generated it.
    ///
    /// A guard is counted as a `#ifndef __CUDACC_RTC__` at column 0, which is the
    /// form every one of them takes; markers are counted separately because a
    /// guard without one would be an unexplained edit, which is the thing the
    /// convention exists to make impossible.
    fn audit() {
        let mut guards = 0;
        let mut markers = 0;
        let mut wrong = Vec::new();

        for file in closure() {
            let name = file.name.trim_start_matches("attn/flashinfer/");
            let counted = file.text.matches("\n#ifndef __CUDACC_RTC__").count();
            let lines = file.text.lines().count();
            guards += counted;
            markers += file.text.matches("// PIE:").count();

            let listed = MODIFICATIONS.lines().find_map(|line| {
                let mut cols = line.split_whitespace();
                if cols.next()? != name {
                    return None;
                }
                let listed_guards = cols.next()?.parse::<usize>().ok()?;
                let listed_lines = cols.nth(1)?.parse::<usize>().ok()?;
                Some((listed_guards, listed_lines))
            });
            match listed {
                Some((g, l)) if g == counted && l == lines => {}
                Some(_) => wrong.push(format!("{name}: text says {counted} guards / {lines} lines")),
                None => wrong.push(format!("{name}: not listed")),
            }
        }

        // A repeated name in the set the LIBRARY ships is not a probe problem: it
        // is `NVRTC_ERROR_INVALID_INPUT` out of `nvrtcCreateProgram`, before a
        // line is compiled and with no indication of which name was doubled. The
        // generator refuses to emit one spelling for two files, which covers a
        // collision WITHIN a tree; this is the other half -- three trees walked
        // separately and joined by a const fn, where nothing upstream can see
        // that `csrc/shim/cuda_fp16.h` and a second `cuda_fp16.h` would be
        // two entries with one name.
        let mut names = HashSet::new();
        let doubled: Vec<&str> =
            source::ALL_HEADERS.iter().map(|h| h.name).filter(|n| !names.insert(*n)).collect();
        if doubled.is_empty() {
            println!(
                "ALL_HEADERS:    {} entries, {} bytes, no name carried twice",
                source::ALL_HEADERS.len(),
                source::ALL_HEADERS.iter().map(|h| h.text.len()).sum::<usize>()
            );
        } else {
            println!("ALL_HEADERS:    CARRIES A NAME TWICE -- {}", doubled.join(", "));
        }

        if wrong.is_empty() {
            println!(
                "MODIFICATIONS:  agrees with the carried text -- {guards} guards, \
                 {markers} markers, {} files",
                closure().count()
            );
        } else {
            println!("MODIFICATIONS:  DISAGREES with the carried text");
            for line in &wrong {
                println!("    {line}");
            }
        }
    }

    /// Compile every file in the closure as its own translation unit.
    ///
    /// Separately, and each with a `__global__` after the include, so that an OK
    /// means "this header is acceptable on its own" rather than "the file that
    /// included it happened to have defined what it needed first". A trailing
    /// kernel because NVRTC will happily accept a translation unit that declares
    /// nothing at all, and that would be a measurement of nothing.
    ///
    /// Returns how many were accepted, and out of how many.
    fn compile_closure(config: &Config, arch: &str) -> (usize, usize) {
        // The order §13.6 asked for: the leaves first, then `decode.cuh` -- the
        // target -- then the two stretch goals, then everything else the walk
        // reached. Reading top to bottom, the first REFUSED is the one that
        // matters; the rest are usually the same refusal seen through an include.
        const ORDER: &[&str] = &[
            "attn/flashinfer/page.cuh",
            "attn/flashinfer/pos_enc.cuh",
            "attn/flashinfer/layout.cuh",
            "attn/flashinfer/utils.cuh",
            "attn/flashinfer/attention/state.cuh",
            "attn/flashinfer/attention/variants.cuh",
            "attn/flashinfer/attention/mask.cuh",
            "attn/flashinfer/fastdiv.cuh",
            "attn/flashinfer/attention/decode.cuh",
            "attn/flashinfer/attention/prefill.cuh",
            "attn/flashinfer/attention/mla.cuh",
        ];

        let mut names: Vec<&str> = ORDER.to_vec();
        names.extend(closure().map(|h| h.name).filter(|n| !ORDER.contains(n)));

        let mut ok = 0;
        for name in &names {
            let source = format!("#include \"{name}\"\n__global__ void probe() {{}}\n");
            let marker = if *name == "attn/flashinfer/attention/decode.cuh" { " <-- the target" } else { "" };
            match compile(&source, config, arch, None) {
                Ok(built) => {
                    ok += 1;
                    println!("  OK       {:<46} {:7.0} ms{marker}", name, built.millis);
                }
                Err(log) => println!("  REFUSED  {:<46} {}", name, first_line(&log)),
            }
        }
        (ok, names.len())
    }

    /// The real test: name a kernel, and see whether a cubin comes back.
    ///
    /// A header that preprocesses proves the includes resolve. It does not prove
    /// the templates instantiate — every `__global__` in `decode.cuh` is a
    /// template, so a compile that names none of them type-checks their bodies
    /// only as far as a dependent name allows, which is barely at all. The
    /// instantiation is where `vec_t<half, 8>`'s unions, the `cp.async` inline PTX,
    /// the `__shfl_xor_sync` reductions and the `__grid_constant__` parameter are
    /// really compiled.
    ///
    /// The instantiation named here is not invented: it is the one
    /// `kernels-cuda/csrc/src/attn/attention_flashinfer_common.cuh` already fires
    /// through `BatchDecodeWithPagedKVCacheDispatched<HEAD_DIM=128, kNone,
    /// DefaultAttention<false, true, false, false>, BatchDecodeParams<bf16, bf16,
    /// bf16, int32_t>>`, with the launcher's own arithmetic worked through for
    /// head_dim 128 and a GQA group of 4: `vec_size = max(16/2, 128/32) = 8`,
    /// `bdx = 128/8 = 16`, `bdy = GROUP_SIZE = 4`, `num_threads = max(128, 64) =
    /// 128`, `bdz = 128/64 = 2`, `tile_size_per_bdx = 1`, and `NUM_STAGES_SMEM = 2`
    /// because `DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM` takes the `>= 8`
    /// branch on sm_89.
    ///
    /// Returns the cubin, or `None` if NVRTC refused.
    fn instantiate(config: &Config, arch: &str) -> Option<Vec<u8>> {
        const EXPR: &str = concat!(
            "flashinfer::BatchDecodeWithPagedKVCacheKernel<",
            "flashinfer::PosEncodingMode::kNone, 2, 1, 8, 16, 4, 2, ",
            "flashinfer::DefaultAttention<false, true, false, false>, ",
            "flashinfer::BatchDecodeParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int32_t>>"
        );
        const SOURCE: &str = concat!(
            "#include \"attn/flashinfer/attention/decode.cuh\"\n",
            "#include \"attn/flashinfer/attention/default_decode_params.cuh\"\n",
            "#include \"attn/flashinfer/attention/variants.cuh\"\n"
        );

        println!("\n  BatchDecodeWithPagedKVCacheKernel<kNone, 2, 1, 8, 16, 4, 2,");
        println!("      DefaultAttention<false, true, false, false>,");
        println!("      BatchDecodeParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int32_t>>");
        match compile(SOURCE, config, arch, Some(EXPR)) {
            Ok(built) => {
                println!("  INSTANTIATED                                     {:7.0} ms", built.millis);
                println!("      lowered name  {}", built.lowered.as_deref().unwrap_or("(none)"));
                println!("      cubin         {} bytes", built.cubin.len());
                Some(built.cubin)
            }
            Err(log) => {
                println!("  REFUSED");
                // From the first error, and then the lines after it -- not the
                // lines matching "error". NVRTC's "detected during:" chain names
                // the instantiation that reached the bad expression, and for a
                // template this deep that chain IS the diagnosis: the same error
                // text means one thing under `cast_from` at vec_size 8 and
                // another under a scalar store.
                let first = log.lines().position(|l| l.contains("error")).unwrap_or(0);
                for line in log.lines().skip(first).take(12) {
                    println!("      {}", line.trim_end());
                }
                None
            }
        }
    }

    /// The configuration that answered §13.6: NVIDIA's own device headers and
    /// real CCCL behind the vendored closure.
    ///
    /// If FlashInfer compiles here, the SOURCE is acceptable to NVRTC and every
    /// later refusal belongs to a shim. That separation is the only reason this
    /// configuration exists — nothing about it is shippable, since it reads a
    /// toolkit off the build machine at run time.
    ///
    /// [`source::UPSTREAM`] and not [`source::ALL_HEADERS`], deliberately: the
    /// crutch must not see `csrc/shim`, or it would be measuring the shims it
    /// exists to be independent of. The single exception is
    /// `cooperative_groups.h`, which is the shim in BOTH columns — NVIDIA's is a
    /// door onto a forty-file `cooperative_groups/details/` tree, and carrying it
    /// would have this probe reporting on CG's headers instead of FlashInfer's.
    /// `examples/cg_probe.rs` proved that one already.
    ///
    /// [`source::ALL_HEADERS`]: kernels_cuda_new::source::ALL_HEADERS
    fn crutch_config(cuda: &Path) -> Config {
        let mut entries: Vec<Entry> =
            source::UPSTREAM.iter().map(|h| entry(h.name, Cow::Borrowed(h.text))).collect();
        let cg = shim(COOPERATIVE_GROUPS).expect("`csrc/shim` carries the CG shim");
        entries.push(entry(cg.name, Cow::Borrowed(cg.text)));

        let mut crutched = Vec::new();
        let mut missing = Vec::new();

        let device = walk(&cuda.join("include"), &["cuda_fp16.h", "cuda_bf16.h", "cuda_fp8.h", "cuda_fp4.h"]);
        // CCCL's own `cuda/std/limits` comes with the walk, and must: its internals
        // include each other by paths our 4 KB shim knows nothing about.
        let roots: Vec<PathBuf> = cccl_root().into_iter().chain([cuda.join("include")]).collect();
        let cccl = walk_many(&roots, &["cuda/cmath", "cuda/pipeline", "cuda/std/limits"]);

        for name in SHIMMABLE {
            if *name == COOPERATIVE_GROUPS {
                continue;
            }
            if device.contains_key(*name) || cccl.contains_key(*name) {
                crutched.push((*name).to_string());
            } else {
                missing.push((*name).to_string());
            }
        }
        for (name, text) in device.into_iter().chain(cccl) {
            entries.push(entry(&name, Cow::Owned(text)));
        }

        Config {
            what: "the crutch -- NVIDIA's real device headers and CCCL, from disk",
            entries: dedup(entries),
            shimmed: vec![cg.name.to_string()],
            crutched,
            missing,
            library: Vec::new(),
        }
    }

    /// The configuration that will ship: the crate's own header set, exactly as
    /// the library assembles it.
    ///
    /// [`source::ALL_HEADERS`] verbatim — prelude, shims, upstream — because that is
    /// what a FlashInfer [`Unit`] would resolve its includes against, and a probe
    /// that assembled its own approximation of it would answer a question about
    /// the approximation. This example carried a hand-written sixty-entry list
    /// until `build.rs` learned to walk `csrc/`; the list is gone and the set is
    /// now the same object the crate ships.
    ///
    /// The crutch still fills anything `csrc/shim` has no shim for, and the report
    /// names it. CCCL is deliberately excluded from that fallback even when a
    /// `cuda/*` name goes unanswered: mixing was measured, and with our 4 KB
    /// `cuda/std/limits` in the set CCCL's `__bit/popcount.h` finds an incomplete
    /// `numeric_limits<uint64_t>` and buries the real answer under twenty template
    /// errors. A missing header says "no shim yet" in one line; a half-mixed one
    /// says nothing in forty.
    ///
    /// [`source::ALL_HEADERS`]: kernels_cuda_new::source::ALL_HEADERS
    /// [`Unit`]: kernels_cuda_new::unit::Unit
    fn shim_config(cuda: &Path) -> Config {
        let mut entries: Vec<Entry> =
            source::ALL_HEADERS.iter().map(|h| entry(h.name, Cow::Borrowed(h.text))).collect();

        let mut shimmed = Vec::new();
        let mut wanted = Vec::new();
        for name in SHIMMABLE {
            if shim(name).is_some() {
                shimmed.push((*name).to_string());
            } else {
                wanted.push(*name);
            }
        }
        // Only the device headers fall back. A missing `cuda/*` shim stays missing.
        let fallback: Vec<&str> = wanted.iter().copied().filter(|n| !n.starts_with("cuda/")).collect();
        for (name, text) in walk(&cuda.join("include"), &fallback) {
            entries.push(entry(&name, Cow::Owned(text)));
        }
        let mut crutched: Vec<String> = fallback.iter().map(|n| (*n).to_string()).collect();
        crutched.sort();

        // What the crate carries that FlashInfer never asks for by name: the
        // prelude every unit gets, and the shims' own supporting files.
        // `csrc/shim/cuda_bf16.h` opens with `#include "pie_device.cuh"` -- the
        // shims are written against the crate's header set rather than against
        // nothing -- so a set that carried only the doors stops on a shim's first
        // line. Measured, and it is why this configuration is `ALL_HEADERS` and
        // not a filtered copy of it.
        let library: Vec<String> = source::DEVICE_HEADERS
            .iter()
            .map(|h| h.name.to_string())
            .filter(|name| !SHIMMABLE.contains(&name.as_str()))
            .collect();

        shimmed.sort();
        Config {
            what: "the shims -- source::ALL_HEADERS, the set the library would ship",
            entries: dedup(entries),
            shimmed,
            crutched,
            missing: wanted.iter().filter(|n| n.starts_with("cuda/")).map(|n| (*n).to_string()).collect(),
            library,
        }
    }

    /// `cooperative_groups.h`, named once because two configurations reach for
    /// the same shim.
    const COOPERATIVE_GROUPS: &str = "cooperative_groups.h";

    /// One entry, spelled and filled.
    fn entry(name: &str, text: Cow<'static, str>) -> Entry {
        Entry { name: name.to_string(), text }
    }

    /// First spelling wins, and the rest are dropped.
    ///
    /// Not a tidy-up: two entries with one name is `NVRTC_ERROR_INVALID_INPUT` out
    /// of `nvrtcCreateProgram`, with no indication of which name was doubled. The
    /// order matters too — a shim must win over the crutch that would otherwise
    /// answer the same include, which is why the shims are pushed first.
    ///
    /// [`source::ALL_HEADERS`] no longer needs this — the generator refuses to
    /// emit one spelling for two files — but the crutch does: its device and CCCL
    /// entries come from a walk of two directory trees, and the same header is
    /// reachable from both.
    ///
    /// [`source::ALL_HEADERS`]: kernels_cuda_new::source::ALL_HEADERS
    fn dedup(entries: Vec<Entry>) -> Vec<Entry> {
        let mut seen = HashSet::new();
        entries.into_iter().filter(|e| seen.insert(e.name.clone())).collect()
    }

    /// What a compile produced, when it produced anything.
    struct Built {
        /// Wall time inside `nvrtcCompileProgram`.
        millis: f64,
        /// The mangled symbol NVRTC assigned to the name expression, if one was
        /// asked for. This is what `cuModuleGetFunction` would be handed.
        lowered: Option<String>,
        /// The cubin, empty when none was requested.
        cubin: Vec<u8>,
    }

    /// Compile `source` against `config`'s header set, optionally instantiating
    /// `expr`.
    ///
    /// The options are `src/runtime/nvrtc.rs`'s, with one addition that is itself a
    /// finding: **`--device-as-default-execution-space`**. Without it NVRTC treats
    /// an unannotated function as host code and refuses it — *"a function
    /// explicitly marked as a `__host__` function is not allowed in JIT mode"* for
    /// the annotated ones, and *"considered a host function... Consider using
    /// -default-device"* for the rest — which would have meant guarding several
    /// hundred perfectly good `constexpr` helpers all through FlashInfer. The
    /// crate's `options()` does not pass this flag globally and should not: on our
    /// own sources it would compile an unannotated HOST helper onto the device
    /// silently instead of reporting it. [`Unit::options`] is the mechanism — a
    /// per-unit list appended after the shared contract and spanned by the cache
    /// key — and a real FlashInfer unit is where this flag belongs.
    ///
    /// [`Unit::options`]: kernels_cuda_new::unit::Unit
    fn compile(source: &str, config: &Config, arch: &str, expr: Option<&str>) -> Result<Built, String> {
        let src = CString::new(source).map_err(|_| "a NUL in the probe source")?;
        let name = c"flashinfer_probe.cu";

        let texts: Vec<CString> = config
            .entries
            .iter()
            .map(|e| CString::new(e.text.as_ref()).map_err(|_| format!("NUL in {}", e.name)))
            .collect::<Result<_, _>>()?;
        let names: Vec<CString> = config
            .entries
            .iter()
            .map(|e| CString::new(e.name.as_str()).expect("no NUL in a header name"))
            .collect();
        let text_ptrs: Vec<_> = texts.iter().map(|t| t.as_ptr()).collect();
        let name_ptrs: Vec<_> = names.iter().map(|n| n.as_ptr()).collect();

        let mut program: nv::nvrtcProgram = std::ptr::null_mut();
        // SAFETY: every string outlives the call and the two arrays are the same
        // length, which is the whole of `nvrtcCreateProgram`'s contract. The header
        // set is an in-memory filesystem; NVRTC opens nothing.
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

        let expression = expr.map(|e| CString::new(e).expect("no NUL in a name expression"));
        if let Some(expression) = &expression {
            // SAFETY: the program is live and the string outlives the call.
            let code = unsafe { nv::nvrtcAddNameExpression(program, expression.as_ptr()) };
            if code != nv::nvrtcResult::NVRTC_SUCCESS {
                return Err(format!("nvrtcAddNameExpression: {code:?}"));
            }
        }

        let gpu = CString::new(format!("--gpu-architecture={arch}")).unwrap();
        let options = [
            gpu.as_ptr(),
            c"-std=c++17".as_ptr(),
            c"--fmad=false".as_ptr(),
            c"--prec-div=true".as_ptr(),
            c"--prec-sqrt=true".as_ptr(),
            c"--device-as-default-execution-space".as_ptr(),
        ];

        let started = Instant::now();
        // SAFETY: the program is live and the options outlive the call.
        let code = unsafe {
            nv::nvrtcCompileProgram(program, i32::try_from(options.len()).unwrap(), options.as_ptr())
        };
        let millis = started.elapsed().as_secs_f64() * 1e3;
        let log = program_log(program);

        if code != nv::nvrtcResult::NVRTC_SUCCESS {
            // SAFETY: destroyed exactly once, and not used after.
            unsafe { nv::nvrtcDestroyProgram(&raw mut program) };
            return Err(log);
        }

        let mut lowered = None;
        if let Some(expression) = &expression {
            let mut out: *const std::ffi::c_char = std::ptr::null();
            // SAFETY: the program compiled with this expression registered, so
            // NVRTC owns a string for it; it stays valid until the program is
            // destroyed, which is why it is copied here.
            let code = unsafe { nv::nvrtcGetLoweredName(program, expression.as_ptr(), &raw mut out) };
            if code == nv::nvrtcResult::NVRTC_SUCCESS && !out.is_null() {
                // SAFETY: NVRTC returned a NUL-terminated string it still owns.
                lowered = Some(unsafe { CStr::from_ptr(out) }.to_string_lossy().into_owned());
            }
        }

        let mut cubin = Vec::new();
        if expression.is_some() {
            let mut size = 0;
            // SAFETY: the program compiled, so a cubin exists; `size` is live.
            unsafe { nv::nvrtcGetCUBINSize(program, &raw mut size) };
            cubin = vec![0u8; size.max(1)];
            // SAFETY: the buffer is exactly the size NVRTC just asked for.
            unsafe { nv::nvrtcGetCUBIN(program, cubin.as_mut_ptr().cast()) };
            cubin.truncate(size);
        }

        // SAFETY: destroyed exactly once, and not used after.
        unsafe { nv::nvrtcDestroyProgram(&raw mut program) };
        Ok(Built { millis, lowered, cubin })
    }

    /// Read `roots[0]`'s include graph starting from `seeds`, following every
    /// `#include` that resolves inside them.
    ///
    /// A crutch needs a closure, not a file: `cuda_fp16.h` is a door onto
    /// `cuda_fp16.hpp`, `vector_types.h` and `device_types.h`, and NVRTC will not
    /// find any of them either. Resolution is the C rule as far as it matters here
    /// — beside the includer first, then at the root — and a directive that
    /// resolves nowhere is left alone, because it is either NVRTC's own preamble or
    /// the next thing this probe is meant to report.
    fn walk(root: &Path, seeds: &[&str]) -> HashMap<String, String> {
        walk_many(std::slice::from_ref(&root.to_path_buf()), seeds)
    }

    /// [`walk`], across several roots — CCCL lives beside the toolkit, not in it.
    fn walk_many(roots: &[PathBuf], seeds: &[&str]) -> HashMap<String, String> {
        let mut out: HashMap<String, String> = HashMap::new();
        let mut queue: VecDeque<String> = seeds.iter().map(|s| (*s).to_string()).collect();

        while let Some(name) = queue.pop_front() {
            if out.contains_key(&name) {
                continue;
            }
            let Some(text) = roots.iter().find_map(|r| std::fs::read_to_string(r.join(&name)).ok())
            else {
                continue;
            };
            let dir = Path::new(&name).parent().map(Path::to_path_buf).unwrap_or_default();
            for include in includes(&text) {
                let beside = normalize(&dir.join(&include));
                if roots.iter().any(|r| r.join(&beside).is_file()) {
                    queue.push_back(beside);
                } else if roots.iter().any(|r| r.join(&include).is_file()) {
                    queue.push_back(include);
                }
            }
            out.insert(name, text);
        }
        out
    }

    /// Every `#include` in `text`, quoted or angled, in source order.
    ///
    /// Deliberately not a preprocessor: a directive inside an `#if` that will never
    /// be taken is followed anyway. That over-collects, which for a crutch costs
    /// bytes; under-collecting would cost a compile.
    fn includes(text: &str) -> Vec<String> {
        text.lines()
            .filter_map(|line| {
                let line = line.trim_start();
                let rest = line.strip_prefix('#')?.trim_start().strip_prefix("include")?.trim_start();
                let (open, close) = match rest.chars().next()? {
                    '"' => ('"', '"'),
                    '<' => ('<', '>'),
                    _ => return None,
                };
                let rest = rest.strip_prefix(open)?;
                rest.split(close).next().map(str::to_string)
            })
            .collect()
    }

    /// Collapse `a/../b` to `b`, which the C rule leaves for the filesystem and a
    /// header set has no filesystem to leave it to.
    fn normalize(path: &Path) -> String {
        let mut parts: Vec<&str> = Vec::new();
        for part in path.iter().filter_map(|p| p.to_str()) {
            match part {
                "." => {}
                ".." => {
                    parts.pop();
                }
                other => parts.push(other),
            }
        }
        parts.join("/")
    }

    /// Where the crutch comes from. `$CUDA_HOME`, then the conventional path.
    fn cuda_home() -> PathBuf {
        std::env::var_os("CUDA_HOME")
            .or_else(|| std::env::var_os("CUDA_PATH"))
            .map_or_else(|| PathBuf::from("/usr/local/cuda"), PathBuf::from)
    }

    /// CCCL 3.3.2, as FlashInfer vendors it, inside `kernels-cuda`'s CMake build
    /// tree.
    ///
    /// Found rather than configured, and found there rather than in `$CUDA_HOME`
    /// for a measured reason: the toolkit ships CCCL 3.0.0, which has no
    /// `cuda::fast_mod_div` — the one name `fastdiv.cuh` reaches `<cuda/cmath>`
    /// for, and therefore the one name the whole decode path needs. FlashInfer
    /// pins its own copy ahead of the toolkit's precisely because of differences
    /// like that.
    fn cccl_root() -> Option<PathBuf> {
        let builds = manifest().join("../../target/debug/build");
        let mut found: Vec<PathBuf> = std::fs::read_dir(builds)
            .ok()?
            .filter_map(Result::ok)
            .map(|e| {
                e.path().join("out/kernels-cuda/build/_deps/flashinfer-src/3rdparty/cccl/libcudacxx/include")
            })
            .filter(|p| p.is_dir())
            .collect();
        found.sort();
        found.pop()
    }

    /// This crate's root, so the probe can find `csrc/src/` without assuming a
    /// working directory.
    fn manifest() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    /// Print a provenance list, or say it is empty — an empty crutch line is the
    /// result this crate is working toward and should be visible when it arrives.
    fn report_provenance(label: &str, names: &[String]) {
        if names.is_empty() {
            println!("{label} (none)");
        } else {
            println!("{label} {}", names.join(", "));
        }
    }

    /// Total lines of vendored FlashInfer, for the report's first paragraph.
    fn lines() -> usize {
        closure().map(|h| h.text.lines().count()).sum()
    }

    /// Total bytes of vendored FlashInfer, counted once per file rather than once
    /// per spelling.
    fn bytes() -> usize {
        closure().map(|h| h.text.len()).sum()
    }

    /// `libnvrtc`'s own version, which is what every measurement here is about.
    fn version() -> String {
        let (mut major, mut minor) = (0, 0);
        // SAFETY: both are live out-parameters for the call's duration.
        let code = unsafe { nv::nvrtcVersion(&raw mut major, &raw mut minor) };
        if code == nv::nvrtcResult::NVRTC_SUCCESS {
            format!("{major}.{minor}")
        } else {
            format!("unavailable ({code:?})")
        }
    }

    /// The compile log, whatever NVRTC put in it.
    fn program_log(program: nv::nvrtcProgram) -> String {
        let mut size = 0;
        // SAFETY: `program` is live and `size` is a live out-parameter.
        unsafe { nv::nvrtcGetProgramLogSize(program, &raw mut size) };
        let mut log = vec![0u8; size.max(1)];
        // SAFETY: the buffer is `size` bytes, which is what NVRTC just asked for.
        unsafe { nv::nvrtcGetProgramLog(program, log.as_mut_ptr().cast()) };
        CStr::from_bytes_until_nul(&log).map_or_else(|_| String::new(), |s| s.to_string_lossy().into_owned())
    }

    /// The first line of a diagnosis that says something, so a refusal fits on a
    /// row of the report.
    fn first_line(log: &str) -> String {
        log.lines()
            .find(|line| line.contains("error") || line.contains("catastrophic"))
            .unwrap_or_else(|| log.lines().next().unwrap_or("(no log)"))
            .trim()
            .to_string()
    }
}
