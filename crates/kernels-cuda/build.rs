//! Generate and compile the launch shim. There is no archive any more.
//!
//! # What this script used to be, and what deleting it measured
//!
//! It had two jobs: build `csrc/` into `libpie_kernels_cuda.a` through CMake,
//! and publish the include paths a consumer needed to compile against it —
//! FlashInfer's especially, because this crate's CMake FETCHED and PATCHED
//! FlashInfer (three source patches over a CPM cache shared by every build
//! directory on the machine) and a second fetch on the cargo side would have
//! been a second clone, a second patch pass and a second chance to disagree.
//!
//! Both jobs are gone, and the second went first: `driver-cuda`'s
//! `pie_attn_flashinfer` target was deleted when its dispatches became rows,
//! which left the two published keys with no reader. The first went with
//! `csrc/CMakeLists.txt`, and the measurement is in `build()`:
//! `add_library(pie_kernels_cuda STATIC …)` had **no sources of its own** —
//! its whole content was the CUTLASS MoE object library — and `csrc/src` now
//! holds zero `.cu` and zero `.cpp`. There was nothing left to compile.
//!
//! **That was the last nvcc invocation in this workspace.** Not the last
//! host C++ compile: `shim()` below is a `cc::Build`, and it is next.
//!
//! Without the `native` feature this script still does nothing at all, which
//! remains the point of the feature: `model-compiler` depends on this crate
//! for the signature table it re-exports, and must not pay a C++ compiler to
//! read a table.
//!
//! # The launch shim
//!
//! `native` generates `shim.cpp` — one `extern "C" pie_k_*` per stated row
//! that is NOT JIT-dispatched, forwarding to the real launcher with its
//! header in scope — and compiles it into `libpie_launch_shim.a`.
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
//! It is BUILT here and LINKED by whoever calls it. That used to be described
//! as "the same split `libpie_kernels_cuda.a` already has"; the archive is
//! gone, so the split is no longer a convention with two instances but a rule
//! with one, and `shim()`'s doc has the measurement that forced it.
//!
//! Everything the shim compiles against is this crate's already — `csrc/src`'s
//! per-family headers, and nothing out of tree since §47 deleted the vendored
//! Marlin wrapper — which is the same argument in the other direction, and is
//! why `csrc/src` outlived the CMake that used to name every file in it.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Table only. Silently, and on purpose: this is the path `model-compiler`
    // takes on every build, and a build script that printed something here
    // would print it on every build.
    //
    // The gate is `#[cfg]` rather than a runtime `CARGO_FEATURE_NATIVE` check
    // because `cc` is an OPTIONAL build-dependency: without the feature it is
    // not in the graph at all, and a runtime check would still leave
    // `cc::Build` in the token stream for rustc to fail on. The argument was
    // first written for `cmake::Config` and survives it unchanged, because it
    // was never about which crate — it is about optional build-dependencies
    // and token streams.
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
    /// `pie_k_*` for them and nothing calls one. There is no second table
    /// left to append.
    ///
    /// # This paragraph used to end *"— the six callers are direct Rust
    /// calls"*, and there are none
    ///
    /// Measured over every `.rs` under `crates/` — `src/`, `tests/`,
    /// `examples/`, `benches/` and `build.rs` — with comments stripped and
    /// string literals preserved, the token `driver_internal` occurs three
    /// times: `x/mod.rs`'s `pub mod driver_internal;`, and one mention each
    /// in `driver-cuda/tests/launch_abi.rs` (`bridge_smoke.rs` was the
    /// second and is deleted with `bind::abi::ffi`). **Not one
    /// `src/` outside the declaration.** The six `pub unsafe fn`s have no
    /// caller of any kind, direct or generated.
    ///
    /// `x/driver_internal.rs`'s own header states the same absent set from
    /// the other end — *"Launchers the DRIVER reaches for directly"* — and
    /// `driver-cuda` reaches none of them; where it wants that shape it has
    /// written its own, e.g. `fire/split_packed.rs::split_qkv_bf16_devwin`
    /// beside `x::driver_internal::split_qkv_bf16`. Two files one crate
    /// apart agreeing about a caller set that is empty is why the sentence
    /// survived: each read as corroboration of the other.
    ///
    /// **Nothing is deleted on this measurement.** It is stated here and
    /// not acted on because the module is `x/**` and not this crate's, and
    /// because "no caller today" is the premise of a decision, not the
    /// decision — the same restraint `x/attn.rs` applies to
    /// `write_kv_at_positions`, carried with no caller precisely so that a
    /// transcription does not silently drop a kernel.
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
        // and an explicit list says what it LOOKED at.
        //
        // HOW MANY ARE IN THAT STATE — measured, because the sentence here
        // used to say "Six of the twelve" and that number was written once
        // and never re-run:
        //
        //   present, contributing .hpp   attn (5), rope (1), vision (5)
        //   present, contributing none   layout (0 headers, empty dir)
        //   MISSING                      norm, mlp, gemm, moe, ssm, quant,
        //                                sample, comm
        //
        // **Eight of the twelve are gone and nine contribute nothing**, so
        // this list looks at twelve names to find eleven headers under
        // three of them. The count is stated as a derivation rather than a
        // number so that the next reader can re-run it: it is `ls
        // csrc/src/<name>/*.hpp` for each name in the array below, and
        // nothing else.
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
    /// The control WAS `pie_kernels_cuda`, which used the *same* `static=`
    /// and worked: `driver-cuda/build.rs` emitted it, `driver_cuda` IS
    /// loaded, and its rlib carried the CMake objects. So the rule is not
    /// about the modifier. **The crate that references a symbol is the crate
    /// that must name the archive defining it**, and for `pie_k_*` that is
    /// `driver-cuda`.
    ///
    /// The control is gone — the archive was deleted with the CMake that
    /// built it, and `driver-cuda/build.rs`'s `-lpie_kernels_cuda` with it —
    /// so the measurement it settled is recorded rather than reproducible.
    /// That is worth saying plainly: this doc's claim now rests on a
    /// comparison a reader cannot re-run, and the reason to keep it is that
    /// it explains a `cargo_metadata(false)` that would otherwise look like
    /// an oversight. The shim is the only archive left and the only one that
    /// needs the rule.
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

        // `csrc/src` alone. There were four entries here — `CMakeLists.txt`,
        // `cmake`, `src`, `third_party` — and three of them name directories
        // that no longer exist. What is left is watched because the shim
        // `#include`s it: `shim()` passes `csrc_src()` to `cc::Build`, and an
        // edited `.hpp` that did not trigger a rebuild is a shim compiled
        // against the previous host interface.
        println!("cargo:rerun-if-changed=csrc/src");
        // The device headers, which are not under `csrc/` at all: the `.cuh`
        // files live in `kernels-cuda-new/csrc/src`, which compiles them
        // through NVRTC. Cargo cannot see the edge, and the failure it causes
        // is the quiet one -- an edited kernel template and a shim that is not
        // regenerated because nothing this script watches changed.
        //
        // Both directories are still watched, and `csrc/shim`'s reason is now
        // the SHARPER one rather than the archive's: it holds the headers that
        // answer `<cuda_fp16.h>` and friends, so an edit there changes what
        // `__half` MEANS in every translation unit NVRTC compiles, and
        // `emit_c_shim`'s `jit_dispatched()` list is derived from rows whose
        // instantiation strings name those types.
        println!("cargo:rerun-if-changed=../kernels-cuda-new/csrc/src");
        println!("cargo:rerun-if-changed=../kernels-cuda-new/csrc/shim");

        let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
        shim(&out_dir);

        // ---------------------------------------------------------------
        // THE CMAKE HALF STOOD HERE, AND IT IS DELETED. THAT IS NVCC LEAVING
        // THE WORKSPACE.
        //
        // It ran `cmake::Config::new(csrc).build_target("pie_kernels_cuda")`,
        // walked the build tree for `-L` paths, and republished two include
        // keys. All three go together, and the reason is a measurement rather
        // than a preference:
        //
        //   `add_library(pie_kernels_cuda STATIC …)`   0 live source lines
        //   its only content: `target_sources(… $<TARGET_OBJECTS:
        //                      pie_flashinfer_cutlass_moe>)`
        //   `csrc/src/**.cu` on disk                   0
        //   `csrc/src/**.cpp` on disk                  0
        //
        // The archive was not a target that survived the CUTLASS MoE object
        // library and needed retiring in its own pass — it was EMPTY the
        // moment that library went, and an `add_library` with no sources is a
        // CMake error rather than a small target. So `csrc/CMakeLists.txt` was
        // deleted whole: both `add_library` calls, the `CPMAddPackage` that
        // cloned and patched FlashInfer, the `_SM90_`/`_SM100_` source lists,
        // the 45 generated instantiation units, and `enable_language(CUDA)`,
        // which is the line that made this workspace look for nvcc.
        //
        // # The three things it published, and where each went
        //
        // `rustc-link-search` came from `emit_link_search_paths`, which
        // walked the CMake build tree for directories holding an archive.
        // There is no build tree. The one archive this script still produces
        // is the shim's, and `shim()` prints its own `-L` — deliberately, and
        // its doc has the measurement for why a `-l` may not go with it.
        //
        // `DEP_PIE_KERNELS_CUDA_{FLASHINFER,CCCL}` came from
        // `read_flashinfer_paths`, reading `pie_kernels_cuda_paths.txt` out of
        // a `file(GENERATE)` at the bottom of the CMakeLists. Their consumer
        // was `driver-cuda`'s `pie_attn_flashinfer` `cc::Build`, whose two
        // `expect`s were deleted with the target itself; a sweep for
        // `DEP_PIE_KERNELS_CUDA_FLASHINFER` and its CCCL sibling now finds
        // comments and this one. What makes them worth a paragraph is the
        // shape of what was left behind: the reader `panic!`ed on a missing
        // key, insisting that *"the export block and this list have to name
        // the same keys"* — so CMake was obliged to keep computing two paths
        // so that a `println!` could publish them into silence, and the check
        // that enforced it looked exactly like a contract. That is the most
        // convincing kind of dead code there is, and it is the second time
        // this same block has produced one: eight earlier keys went the same
        // way, for the same reason, guarded by the same `panic!`.
        //
        // `PIE_KERNELS_CUDA_BUILD_DIR` was a `rustc-env` with no reader in
        // any crate, in any feature combination.
        //
        // # What is left, and when it goes
        //
        // `shim()` above, which is a HOST C++ compile (`cc`, not `cmake`) of
        // a `shim.cpp` this script GENERATES into `OUT_DIR` — there is no
        // checked-in `.cpp` anywhere in the workspace. It is the forwarding
        // layer the generated dispatch calls, and it dies with `bridge`, on
        // the schedule `ROW_TABLES` emptying sets. Two different lines, and
        // they are worth keeping apart: **this deletion ends nvcc; `attn`'s
        // ends the host C++ compile.**
    }

    // `read_flashinfer_paths`, `emit_link_search_paths` and `walk` STOOD
    // HERE, and are DELETED with the CMake half that was their only caller.
    // `build()`'s comment above has the account of what each published and
    // why nothing reads it. `walk` was a plain directory recursion over the
    // CMake build tree; it has no second use, and a helper kept for a use it
    // does not have is how a build script grows a second source of truth.
}
