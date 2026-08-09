//! Build `csrc/` into `libpie_kernels_cuda.a`, and publish what the shell
//! needs to compile against it.
//!
//! Two jobs, and the second is the interesting one. Building the archive is
//! ordinary. Publishing is not: FlashInfer is fetched and PATCHED by this
//! crate's CMake — three source patches over a CPM cache shared by every
//! build directory on the machine — so `driver-cuda` must not fetch it a
//! second time. Its `batch/workspace.cu` calls FlashInfer's scheduler
//! directly, so it needs those headers on its include path, and it gets them
//! from here through cargo's `links` metadata rather than by running the
//! fetch again.
//!
//! Everything published below comes out of `pie_kernels_cuda_paths.txt`,
//! which the CMake writes at generate time. Nothing here re-derives a path
//! CMake already resolved.
//!
//! Without the `native` feature this script does nothing at all, which is the
//! point of the feature: `model-compiler` depends on this crate for the
//! signature table it re-exports and must never pay nvcc to read it.
//!
//! # The launch shim
//!
//! `native` also generates `shim.cpp` — one `extern "C" pie_k_*` per stated
//! row, forwarding to the real launcher with its header in scope — and
//! compiles it into `libpie_launch_shim.a`.
//!
//! It lives here rather than in a consumer because the shim is the only thing
//! that DEFINES those symbols, and a symbol may have one definition. While
//! `driver-cuda` was the only caller it could own the shim; it is not, and a
//! second generator would make the two mutually exclusive in a binary rather
//! than merely redundant. The Rust `unsafe extern "C"` bindings are a
//! different matter and stay with their callers: those are DECLARATIONS, any
//! number of crates may state them, and `driver-cuda`'s name its own
//! `#[repr(C)]` mirrors.
//!
//! It is BUILT here and LINKED by whoever calls it, which is the same split
//! `libpie_kernels_cuda.a` already has and not a second arrangement to
//! remember. `shim()`'s doc has the measurement that forced it.
//!
//! Everything the shim compiles against is this crate's already — `csrc/src`'s
//! per-family headers, and nothing out of tree since §47 deleted the vendored
//! Marlin wrapper — which is the same argument in the other direction.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Table only. Silently, and on purpose: this is the path `model-compiler`
    // takes on every build, and a build script that printed something here
    // would print it on every build.
    //
    // The gate is `#[cfg]` rather than a runtime `CARGO_FEATURE_NATIVE` check
    // because the `cmake` crate is an OPTIONAL build-dependency: without the
    // feature it is not in the graph at all, and a runtime check would still
    // leave `cmake::Config` in the token stream for rustc to fail on.
    #[cfg(feature = "native")]
    native::build();
}

// The table's crate, and the emitter that reads it.
//
// THERE IS NO `#[path]` INCLUDE HERE ANY MORE, and the absence is the end of
// a constraint rather than a tidy-up. A build script cannot depend on the
// crate it builds, so every module this script needed from `src/` had to be
// pulled in as a source file — fourteen of them, under the rule "every one
// imports `kernels` and nothing else, because a module that reached for a
// sibling or for `crate::` would drag it in too."
//
// A build script MAY depend on any crate that is not the one it builds. The
// tables moved to `kernels-cuda-new` and thirteen of those includes became
// one `[build-dependencies]` line; `src/abi.rs` followed the rows it reads
// and took the fourteenth with it. So the rule is not "one file, one
// exception" any more — it is nothing at all, and `native::tables()` below
// reads `kernels_cuda_new::table::TABLES` rather than restating the family
// list, which is the failure this removes: a family added in one place and
// forgotten in the other emitted no shim entry and failed at LINK time.
//
// What the include cost while it lasted is worth recording, because it is the
// argument for not reaching for the trick again. `abi.rs`'s `#[cfg(test)]`
// module named `crate::attn::KERNELS` and compiled only because a build
// script is never built with `--test` — a silent exemption that looked like
// an invariant. An `extern crate` path has one meaning in the library and in
// the script; an included module has two, and only one of them is checked.

#[cfg(feature = "native")]
mod native {
    use std::path::{Path, PathBuf};

    /// Every family table.
    ///
    /// `kernels_cuda_new::table::TABLES` is the same list `KERNELS`
    /// concatenates, read rather than restated — a second hand-written copy
    /// of it is the shape that goes stale in silence, since a family added
    /// there and forgotten here emits no `extern "C"` and fails at link time
    /// in whichever binary happened to state one of its symbols first.
    ///
    /// `driver_internal` USED to be appended, because it was deliberately
    /// NOT in that list: its rows had no DSL statement, which changed which
    /// invariant they answered to and not whether they needed an entry
    /// point. **They no longer need one.** §5 step 5 made them `fn`s in
    /// `x::driver_internal` with no `contract!`, so nothing generates a
    /// `pie_k_*` for them and nothing calls one — the six callers are direct
    /// Rust calls. There is no second table left to append.
    fn tables() -> Vec<&'static [kernels::KernelSig]> {
        kernels_cuda_new::table::TABLES.to_vec()
    }

    fn csrc_src() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("csrc/src")
    }

    /// The headers the shim compiles against: every family directory's
    /// `.hpp`s. It used to say "plus the vendored Marlin wrapper (`moe`'s one
    /// out-of-tree row)" — that row and both vendored trees are deleted (§47),
    /// and the loop below never named a `third_party` path in the first place,
    /// so the list is now exactly the family directories that still exist.
    /// The union of exactly the per-family lists the `launch_abi` tests
    /// prove — each family's set compiles alone there, and the shim is where
    /// they have to compile TOGETHER.
    ///
    /// # A MISSING DIRECTORY IS A FINISHED FAMILY, NOT A BROKEN TREE
    ///
    /// This loop used to `panic!` on `read_dir`, and by the time it was
    /// changed FIVE of the twelve names below no longer existed: `mlp` and
    /// `sample` had gone before this round, and `norm`, `quant` and `ssm`
    /// went with it — every launcher in each ported to `driver-cuda/src/fire`
    /// and every `__global__` already NVRTC's out of `kernels-cuda-new`. The
    /// panic was the right shape when a family disappearing meant something
    /// had gone wrong; it is the wrong shape now that a family disappearing
    /// is the migration finishing, and the end state of this crate is that
    /// **all twelve are gone and this function returns an empty vector**.
    ///
    /// The name list is kept rather than replaced by a `read_dir` over
    /// `csrc/src` for the reason §21 gives about textual gates: an explicit
    /// list says what it looked at, so a family that comes BACK under a new
    /// name is a header nothing includes rather than a silent addition to the
    /// shim's translation unit. Nothing here covers the other direction —
    /// a `.hpp` on disk in a directory not named — and nothing needs to:
    /// `tests/sources.rs` walks the tree itself.
    fn includes() -> Vec<String> {
        let mut out = Vec::new();
        // `comm` joined when the fused all-reduce landing got a row. It is
        // the one directory no per-family `launch_abi` case covers, so
        // nothing proved its headers alone and the shim was the first
        // thing to ask for them — which is the failure mode the doc above
        // describes: a family's set compiling alone is not the shim
        // compiling, and here there was no alone-case either.
        //
        // `csrc/src/comm/` IS NOW GONE — `custom_all_reduce.{cu,hpp}` and
        // `custom_all_reduce_stub.cpp` are deleted, the whole thing is
        // `driver-cuda/src/fire/all_reduce.rs`, and both rows are
        // `execution::RUST_SERVED` so the shim asks for nothing. The name
        // stays in this list for the reason the doc above gives: the
        // `read_dir` below already returns nothing for a missing directory,
        // and an explicit list says what it LOOKED at. Six of the twelve are
        // in that state now.
        for dir in [
            "attn", "rope", "norm", "mlp", "gemm", "moe", "ssm", "quant", "layout", "sample",
            "vision", "comm",
        ] {
            let Ok(entries) = std::fs::read_dir(csrc_src().join(dir)) else {
                continue;
            };
            let mut hs: Vec<String> = entries
                .filter_map(|e| {
                    let n = e.ok()?.file_name().into_string().ok()?;
                    n.ends_with(".hpp").then(|| format!("{dir}/{n}"))
                })
                .collect();
            hs.sort();
            out.extend(hs);
        }
        out
    }

    fn cuda_home() -> PathBuf {
        std::env::var_os("CUDA_HOME")
            .or_else(|| std::env::var_os("CUDA_PATH"))
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"))
    }

    /// Generate `shim.cpp` and compile it into `libpie_launch_shim.a`.
    ///
    /// Compiling it is the proof, not a translation to be trusted: each
    /// `extern "C"` body CALLS the launcher with the real header in scope, so
    /// C++ overload resolution decides whether the row is right — arity,
    /// order, constness and width, all at once, all as compile errors.
    ///
    /// # This crate BUILDS the archive and does not LINK it
    ///
    /// `cargo_metadata(false)` below, and the `cargo:launch_shim=` line after
    /// it, are the whole of that sentence. What they replace was
    /// `cc::Build`'s default `cargo:rustc-link-lib=static=pie_launch_shim`,
    /// and the doc that used to sit here said it was *"emitted FIRST so that
    /// `-lpie_launch_shim` precedes every other `-l` this crate or its
    /// dependents state … cargo emits a dependency's directives ahead of its
    /// dependent's."*
    ///
    /// **Cargo does not do that.** A build script's `-l` is passed to the
    /// rustc invocation for its OWN package's lib and to no other; only its
    /// `-L` reaches a dependent's command line. Transitive propagation of a
    /// `-l` is rustc's, through crate metadata, and rustc can only carry an
    /// upstream crate's native libraries into a link if it LOADED that crate.
    ///
    /// `static=` compounds it. The default modifier is `+bundle`, which means
    /// rustc does not re-emit a `-l` at all — it copies the archive's members
    /// INTO `libkernels_cuda.rlib` (`ar t` shows `…-shim.o`, `nm -g` shows
    /// all 212 `pie_k_*` as ` T ` there). So the definitions travel on the
    /// rlib, and the rlib travels only if the crate is loaded.
    ///
    /// Since §19 and §21.8 moved every table, header and emitter out of here,
    /// **no crate in the workspace names `kernels_cuda::` any more.** Cargo
    /// still passes `--extern kernels_cuda=…rlib` to `driver-cuda`'s tests;
    /// rustc never loads it, so of the 118 rlibs on that link line
    /// `libkernels_cuda-*.rlib` is not one, and every `pie_k_*` the generated
    /// dispatch calls is undefined — 112 of them, seen as seven only because
    /// `rust-lld` stops at `--error-limit=20`.
    ///
    /// The control is `pie_kernels_cuda`, which uses the *same* `static=` and
    /// works: `driver-cuda/build.rs` emits it, `driver_cuda` IS loaded, and
    /// its rlib carries the CMake objects. So the rule is not about the
    /// modifier. **The crate that references a symbol is the crate that must
    /// name the archive defining it**, and for `pie_k_*` that is `driver-cuda`
    /// — which already states exactly this rule for `libpie_kernels_cuda.a`
    /// twenty lines below, and now states it for the shim too, in the order a
    /// once-scanned archive needs.
    ///
    /// This crate keeps OWNING the shim, for the reason the module header
    /// gives: it is the only definition, and it must be compiled where the
    /// launchers it forwards into are. Building is not linking.
    fn shim(out_dir: &Path) {
        println!("cargo:rerun-if-env-changed=CUDA_HOME");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        let cuda_include = cuda_home().join("include");
        if !cuda_include.join("cuda_runtime.h").is_file() {
            panic!(
                "kernels-cuda's `native` feature needs the CUDA toolkit headers, \
                 and {cuda_include:?} has no cuda_runtime.h. Set \
                 $CUDA_HOME/$CUDA_PATH or install the toolkit — or drop `native` \
                 for the signature table alone."
            );
        }

        let includes = includes();
        let include_refs: Vec<&str> = includes.iter().map(String::as_str).collect();
        // The rows NVRTC compiles get no shim entry: there is no host
        // launcher for one to forward to, and emitting it is what kept the
        // `.cu` alive.
        let jit: Vec<&'static kernels_cuda_new::device::DeviceKernel> =
            kernels_cuda_new::device::jit_dispatched();
        let text = kernels_cuda_new::abi::emit_c_shim(&tables(), &include_refs, &jit)
            .expect("two rows may not claim one entry point");
        let shim_path = out_dir.join("shim.cpp");
        std::fs::write(&shim_path, text).expect("write shim.cpp");

        // The portable half of the Rust side, included by `kernels_cuda::ffi`.
        //
        // Only rows whose operands all cross as primitives, because this
        // block is placed in THIS crate and this crate holds no `#[repr(C)]`
        // mirror — an attention-workspace row would name a type that is not
        // here. The shell generates the full set for itself, where the
        // mirrors are; these are the rows a caller needs no layout to make,
        // which is what the loader's quantize/cast/scale are.
        std::fs::write(
            out_dir.join("ffi.rs"),
            kernels_cuda_new::abi::emit_rust_bindings_portable(&tables()),
        )
        .expect("write ffi.rs");

        // `cc` would otherwise print `cargo:rustc-link-lib=static=…` (and
        // `stdc++`, and an `-L` to OUT_DIR) against THIS crate. See the doc
        // above for why that is the one place they cannot be printed. What
        // is lost with them is `cc`'s own `rerun-if-env-changed` set, so the
        // two entries that actually change this compilation are restated.
        for var in ["CXX", "CXXFLAGS"] {
            println!("cargo:rerun-if-env-changed={var}");
        }
        cc::Build::new()
            .cpp(true)
            .std("c++20")
            .include(csrc_src())
            .include(&cuda_include)
            .file(&shim_path)
            .cargo_metadata(false)
            .compile("pie_launch_shim");

        // THE HANDOFF FOR THE SHIM, read as `DEP_PIE_KERNELS_CUDA_LAUNCH_SHIM`
        // by any crate with a direct dependency on this one — the same
        // mechanism `build()`'s `lib`/`include`/FlashInfer keys use, and for
        // the same reason: a consumer that has to name an archive needs its
        // directory stated rather than inherited. `driver-cuda`'s `-L` for
        // this path did arrive implicitly (cargo does propagate a
        // dependency's `-L`), and relying on that is the other half of the
        // mechanism this defect hid inside. It is explicit now.
        println!("cargo:launch_shim={}", out_dir.display());
    }

    pub fn build() {
        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        if target_os != "linux" {
            panic!(
                "kernels-cuda's `native` feature is Linux-only (got target_os={target_os:?}). \
             The signature table builds everywhere; only the CUDA archive does not."
            );
        }

        let csrc = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("csrc");
        for dir in ["CMakeLists.txt", "cmake", "src", "third_party"] {
            println!("cargo:rerun-if-changed=csrc/{dir}");
        }
        // The device headers, which are not under `csrc/` any more: the
        // `.cuh` files moved to `kernels-cuda-new/csrc/src` and the CMake
        // reaches them with `-iquote`. Cargo cannot see that, and the failure
        // it causes is the quiet one -- an edited kernel template, an archive
        // that is not rebuilt because nothing this script watches changed, and
        // a `.a` whose kernels are one revision behind the `.cuh` the JIT
        // compiles from the same bytes.
        //
        // Two directories since `csrc/` was cut by role, and the second is
        // watched for a sharper version of the same reason: `csrc/shim` holds
        // the headers that answer `<cuda_fp16.h>` and friends, so an edit
        // there changes what `__half` MEANS in every translation unit the
        // archive compiles. An archive built against the previous meaning
        // links against the new one without complaint.
        println!("cargo:rerun-if-changed=../kernels-cuda-new/csrc/src");
        println!("cargo:rerun-if-changed=../kernels-cuda-new/csrc/shim");

        let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
        shim(&out_dir);

        for var in [
            "CUDACXX",
            "CMAKE_CUDA_COMPILER",
            "CMAKE_CUDA_ARCHITECTURES",
            "PIE_COMPILER_LAUNCHER",
            // `PIE_CUDA_BUILD_MARLIN` and `PIE_CUDA_BUILD_MARLIN_MOE` were
            // here. Both options are gone from `csrc/CMakeLists.txt` with the
            // two vendored trees they gated (§47), so a rerun keyed on either
            // is a rerun keyed on a variable no CMake line reads.
            "CPM_SOURCE_CACHE",
        ] {
            println!("cargo:rerun-if-env-changed={var}");
        }

        let mut cfg = cmake::Config::new(&csrc);
        cfg.out_dir(PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("kernels-cuda"));
        // `ptir/tier0.cuh` includes `rng_contract.generated.h`: the bit mapping
        // tier-0 must reproduce exactly, because tier-1 emits the same RNG
        // through NVRTC and the two are diffed. That is a CONTRACT, not driver
        // machinery -- the same shape as `kernels.def` being read by both C++
        // and CMake -- so it is declared here rather than reached for by a
        // relative path, exactly as `driver-cuda/build.rs` declares it.
        let ptir_include = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("tensor-compiler/include");
        cfg.build_target("pie_kernels_cuda")
            .define("BUILD_SHARED_LIBS", "OFF")
            .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
            .define("PIE_PTIR_INCLUDE_DIR", &ptir_include);

        let build_dir = cfg.build().join("build");
        emit_link_search_paths(&build_dir);

        // Deliberately NO `cargo:rustc-link-lib=pie_kernels_cuda` here. A static
        // archive must come after everything that references it on the link
        // line, and cargo emits a dependency's link directives BEFORE its
        // dependent's -- so declaring it here would put `-lpie_kernels_cuda`
        // ahead of `-lpie_driver_cuda_lib`, which is exactly backwards for a
        // shell that calls kernels. `driver-cuda`'s build.rs names both, in
        // order. The search paths above do propagate and are the half that
        // belongs here.
        //
        // AND DELIBERATELY NO `-lpie_launch_shim` EITHER, which is this
        // round's correction rather than a restatement. The line above used
        // to say the shim's directive WAS emitted here and that "before every
        // dependent is exactly where a dependency's directives land". Cargo
        // does not land them there at all -- see `shim()`'s doc for the
        // measurement -- and the shim was consequently linked by nothing.
        // Both archives are named by the crate that calls into them, in the
        // order a once-scanned archive needs.

        // --- the handoff -------------------------------------------------------
        //
        // Read as DEP_PIE_KERNELS_CUDA_<KEY> by any crate with a direct
        // dependency on this one.
        //
        // # This published nine keys and one is read
        //
        // The list was `lib`, `include`, `cccl`, `flashinfer`, `marlin`,
        // `marlin_moe`, `has_marlin`, `has_marlin_moe`, `mamba_sm90` — and
        // every one of the eight now gone was read by **nobody**, in any
        // crate, in any feature combination. `LAUNCH_SHIM` (published by
        // `shim()`, above) is the only key with a consumer:
        // `driver-cuda/build.rs:573`.
        //
        // What makes this worth a paragraph rather than a quiet deletion is
        // the `unwrap_or_else` below. A missing key **panics the build** with
        // a message insisting that *"the export block and this list have to
        // name the same keys"* — so eight dead keys were not merely carried,
        // they were ENFORCED, and CMake was obliged to keep computing eight
        // paths so that a `println!` could publish them into silence. A
        // check that fails loudly for an unread value is the most convincing
        // kind of dead code there is: it looks like a contract.
        //
        // `lib` went with the other eight, and it is worth one more line
        // because it is the key that LOOKED load-bearing. CMake does place
        // `libpie_kernels_cuda.a` under `lib/`, so publishing that path reads
        // as the archive's own address -- but the search paths do not come
        // from it. `emit_link_search_paths` walks `build_dir` and prints a
        // `rustc-link-search` for every directory holding an archive,
        // precisely because the vendored object libraries land wherever
        // their `add_library` put them. `driver-cuda/build.rs` says the same
        // thing from the other side: *"Search paths come from `kernels-cuda`'s
        // own build script"* -- which is true, and is not this.
        //
        // So the whole block goes, and with it the `read_paths` call. Nothing
        // in the tree observes `pie_kernels_cuda_paths.txt` any more; CMake
        // may go on writing it.
        //
        // # ...and then two of them came back, with a reader
        //
        // `flashinfer` and `cccl`, and the sentence above is why they are
        // worth re-deriving rather than reinstating: they now have a consumer
        // and it is named. `new-horizon.md` §44 moved
        // `attn/attention_flashinfer.cu` to `driver-cuda/csrc/attn/` -- its
        // two score-capture dispatches are host WALKS, a `switch` over
        // `src/kernels.def` head dims, and a host walk belongs to the driver
        // -- and that translation unit includes
        // `attn/attention_flashinfer_common.cuh`, which includes eleven
        // FlashInfer headers and reaches CCCL through them.
        //
        // It reads them from HERE rather than fetching its own, because
        // FlashInfer is fetched and PATCHED by `csrc/CMakeLists.txt`: a
        // second `CPMAddPackage` on the cargo side would be a second clone, a
        // second patch pass over the same shared CPM cache, and a second
        // chance to disagree -- the generator's own words, one file over.
        // Two copies of a header set that agree today are two copies that
        // drift, which is §21.7's measurement wearing a different hat.
        //
        // Not the VENDORED tree in `kernels-cuda-new/csrc/vendor/flashinfer`
        // either, which is the same library at v0.6.15 and would have been
        // the shorter path. That directory also holds PIE-written files named
        // `cuda.h`, `cuda_runtime.h`, `cstdint`, `type_traits` and `bit` --
        // shims that exist so NVRTC has something to answer with -- and an
        // `-I` at it is searched before the system directories, so the host
        // compiler's `<cstdint>` would resolve to a device shim. That is the
        // exact hazard `csrc/CMakeLists.txt` argues `-iquote` for at length.
        read_flashinfer_paths(&build_dir);

        println!(
            "cargo:rustc-env=PIE_KERNELS_CUDA_BUILD_DIR={}",
            build_dir.display()
        );
    }

    /// Republish the two include-path keys `driver-cuda` compiles against.
    ///
    /// `DEP_PIE_KERNELS_CUDA_FLASHINFER` and `DEP_PIE_KERNELS_CUDA_CCCL`,
    /// each a `:`-joined directory list, straight out of the
    /// `pie_kernels_cuda_paths.txt` CMake generates.
    ///
    /// A hard panic on a missing file or a missing key, in the shape
    /// `shim()`'s `expect` uses: the consumer is a `cc::Build` whose archive
    /// is NAMED unconditionally on `driver-cuda`'s link line, so a silently
    /// absent include path is not a missing optimisation -- it is
    /// `attention_flashinfer_common.cuh` failing to find
    /// `<flashinfer/attention/prefill.cuh>`, hundreds of lines into a
    /// template, in another crate's build output.
    fn read_flashinfer_paths(build_dir: &Path) {
        let file = build_dir.join("pie_kernels_cuda_paths.txt");
        println!("cargo:rerun-if-changed={}", file.display());
        let text = std::fs::read_to_string(&file).unwrap_or_else(|e| {
            panic!(
                "{file:?} does not read ({e}). CMake writes it with `file(GENERATE)` at the \
                 bottom of `csrc/CMakeLists.txt`, and `driver-cuda`'s `pie_attn_flashinfer` \
                 target needs the `flashinfer=` and `cccl=` lines out of it to compile \
                 `attn/attention_flashinfer_common.cuh` at all."
            )
        });
        for key in ["flashinfer", "cccl"] {
            let value = text
                .lines()
                .find_map(|line| line.strip_prefix(&format!("{key}=")))
                .unwrap_or_else(|| {
                    panic!(
                        "{file:?} has no `{key}=` line. The `file(GENERATE)` block in \
                         `csrc/CMakeLists.txt` and this list have to name the same keys."
                    )
                });
            println!("cargo:{key}={value}");
        }
    }

    /// Every directory under `build_dir` holding at least one static archive.
    /// CMake places `pie_kernels_cuda` under `lib/`, but the vendored object
    /// libraries land wherever their `add_library` put them, and a build-type
    /// subdirectory appears on multi-config generators.
    fn emit_link_search_paths(build_dir: &Path) {
        let mut dirs = std::collections::HashSet::new();
        walk(build_dir, &mut dirs);
        for d in &dirs {
            println!("cargo:rustc-link-search=native={}", d.display());
        }
    }

    fn walk(dir: &Path, out: &mut std::collections::HashSet<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let mut has_archive = false;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "a") {
                has_archive = true;
            }
        }
        if has_archive {
            out.insert(dir.to_path_buf());
        }
    }
}
