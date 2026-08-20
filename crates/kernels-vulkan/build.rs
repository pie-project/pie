//! Publish `kernels/` to the Vulkan shell, and — under `native` — compile every
//! declared variant to SPIR-V.
//!
//! ## Why there IS a compile step here, where the Metal sibling has none
//!
//! Metal compiles its shaders at RUN time: the driver hands `.metal` source and
//! an entry name to the Metal compiler and gets a pipeline state object back,
//! so `kernels-metal`'s build script copies one generated file and stops.
//! Vulkan has no such door. `vkCreateShaderModule` takes SPIR-V words, and
//! nothing in the loader turns Slang into them — so the Slang -> SPIR-V hop is a
//! BUILD-time obligation, and this is where it is paid.
//!
//! That is also what makes the entrypoint set mechanical here in a way it is
//! not on Metal. A Slang compute shader has exactly one entry point and it is
//! always called `main`; what distinguishes `rms_single_row_bfloat16` from
//! `rms_strided_row_bfloat16` is the `-D` set the module was compiled with and
//! the FILE the resulting words are written to. So an entrypoint is a SPIR-V
//! artifact name, one `.spv` per (shader, define set) — which is exactly what
//! llama.cpp's `vulkan-shaders-gen` does, and for the same reason.
//!
//! ## Where the variant list lives
//!
//! In the shader, beside the body it stamps, as a `// pie:instantiate` line:
//!
//! ```glsl
//! // pie:instantiate rms_single_row_bfloat16  T=bf16 N_READS=4
//! ```
//!
//! This is the same decision `.wiki/kernel-x/metal-refactor.md` §2 records for
//! the Metal tree — the macro that stamps the instantiations sits next to the
//! template, so a reader checking coverage reads one file — with the one
//! difference the shading language forces: a `#define` matrix cannot be expanded by the
//! preprocessor into differently-NAMED entry points, so the matrix is a
//! directive a build reads rather than a macro a compiler expands.
//!
//! `scripts/vulkan-kernel-audit.py` reads the same lines to print the census
//! and to run `slangc` over every variant, and `tests/entrypoints.rs` reads
//! them again to pin the table's product against them. Three readers, one
//! source of truth.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

// The tier vocabulary, read straight out of the library's source. A build
// script cannot depend on the crate it builds, so a script that needs
// something the crate declares has to include the source instead; the archive
// crate `kernels-cuda` (deleted at `85c6c674b`) had the same problem with its
// tables and solved it the same way, fourteen modules over. The point is that
// the names a module is STAMPED with here and the names a driver LOOKS UP
// through `kernels_vulkan::Capability` are one definition.
#[path = "src/capability.rs"]
// The build stamps module names; it does not resolve device features, so the
// half of the vocabulary a DRIVER uses is unreachable from here.
#[allow(dead_code)]
mod capability;
use capability::Capability;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // `#[path]`-included below to derive every module's file name. Without
    // this, editing a tier tag or the naming rule leaves the previously
    // compiled `.spv` set in place under the old names -- which is not a build
    // error but a module that cannot be found at runtime, or worse one found
    // under a name that now means something else.
    println!("cargo:rerun-if-changed=src/capability.rs");
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-changed=include");

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let kernels = root.join("kernels");
    let include = root.join("include");

    // Two keys, two DIRECTORIES, and they are different roles — the split
    // `kernels-metal`'s build script draws, kept because the roles are the
    // same ones.
    //
    // `include` is the host include path: `include/pie/kernels/` holds the
    // launch shapes and the name grammar the shell's own C++/Rust reads.
    //
    // `kernels_dir` is the Slang tree. A shell that ships prebuilt SPIR-V never
    // needs it; a tool that re-derives the variant set does.
    println!("cargo:include={}", include.display());
    println!("cargo:kernels_dir={}", kernels.display());

    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));

    // THE CENSUS IS WRITTEN WITHOUT `native`, and that is the point of doing it
    // here rather than after the early return below. Reading the variant set is
    // a PARSE -- a `// pie:instantiate` line is the declaration -- and only
    // COMPILING one needs `slangc`. So the portable half of this crate can
    // answer "what entrypoints are there" on a box with no shader toolchain,
    // which is what `kernels_vulkan::entrypoints()` is and what
    // `driver-vulkan`'s `Shell::admit` asks before it opens a device.
    //
    // It replaces a hand-written `RETIRED` list of the same names in `lib.rs`.
    // That list was the fourth reader of these directives -- after this script,
    // `scripts/vulkan-kernel-audit.py` and `tests/entrypoints.rs` -- and the
    // only one that restated them instead of reading them.
    let variants = collect(&kernels);
    assert!(
        !variants.is_empty(),
        "no `// pie:instantiate` directive under {} — the shader tree cannot \
         be empty and the directive is how a variant is declared",
        kernels.display()
    );
    emit_census(&out, &variants);

    // The SOURCES, for the same reason and under the same condition as the
    // census: reading them is not compiling them. `runtime` is what turns them
    // into SPIR-V, and it needs the text in the binary rather than a path into
    // a `target/` tree — the deployment argument `emit_table` makes below,
    // applied one step earlier.
    emit_sources(&out, &kernels);

    if std::env::var_os("CARGO_FEATURE_NATIVE").is_none() {
        // The table is written EMPTY rather than not written, because
        // `src/module.rs` includes it unconditionally. A file that exists only
        // under a feature would make the library fail to COMPILE without
        // `native`, and the whole point of the portable half is that
        // `model-ir` can read the signature table without owning a shader
        // toolchain. Empty is also the honest answer: no `native`, no modules.
        emit_table(&out, &BTreeMap::new(), Path::new(""));
        return;
    }

    let spv = out.join("spv");
    std::fs::create_dir_all(&spv).expect("create the SPIR-V output directory");

    let slangc = slangc();

    for ((entrypoint, _), variant) in &variants {
        compile_slang(&slangc, &kernels, &spv, entrypoint, variant);
    }

    emit_table(&out, &variants, &spv);

    println!("cargo:spv_dir={}", spv.display());

    // `cargo:spv_dir` reaches DEPENDENTS, as `DEP_PIE_KERNELS_VULKAN_SPV_DIR`.
    // It does not reach this crate's OWN tests and examples, which is where the
    // GPU harness lives -- a test cannot read the link metadata of the crate it
    // is testing. So the same path is also handed over as a rustc env, which
    // `option_env!` picks up, and the `option_` is load-bearing: without
    // `native` there are no modules and the harness has to say so rather than
    // fail to compile.
    println!(
        "cargo::rustc-env=PIE_KERNELS_VULKAN_SPV_DIR={}",
        spv.display()
    );
}

/// Write `census.rs`: every entrypoint the tree declares, sorted.
///
/// The NAME and nothing else. `modules.rs` beside it is keyed by file STEM and
/// exists only under `native`; this is keyed by entrypoint and exists always,
/// because the two answer different questions — "what words do I have" against
/// "what can this backend be asked for".
///
/// A tier is dropped into the name it is a tier OF: `@fp16` declares another
/// compile of an entrypoint that already exists at baseline, which
/// [`collect`]'s own baseline assertion guarantees, so the pair key collapses
/// to one name here.
fn emit_census(out: &Path, variants: &BTreeMap<(String, Capability), Variant>) {
    let mut names: Vec<&str> = variants.keys().map(|(name, _)| name.as_str()).collect();
    names.sort_unstable();
    names.dedup();

    let mut generated = String::from(
        "// Generated by build.rs from the tree's `// pie:instantiate` lines.\n\
         pub static CENSUS: &[&str] = &[\n",
    );
    for name in names {
        generated.push_str(&format!("    {name:?},\n"));
    }
    generated.push_str("];\n");

    std::fs::write(out.join("census.rs"), generated).expect("OUT_DIR is writable");
}

/// Write `sources.rs`: every `.slang` in the tree, keyed by tree-relative path.
///
/// # Why the whole tree and not the files something fires
///
/// A `.slang` root `#include`s others — `common/bf16.slang`, `common/affine.slang`
/// — and the runtime compiler resolves those itself, through the same
/// `search_paths` the build passes `-I`. So the set a fire needs is not the set
/// of files a body NAMES: it is those plus their include closure, transitively.
///
/// This walks the tree instead of computing that closure, because the closure
/// is the thing that goes stale. A new `#include` in a shader would be a build
/// that works here and a fire that refuses at run time with a missing file — a
/// failure a long way from the line that caused it. The tree is 392 KB.
fn emit_sources(out: &Path, kernels: &Path) {
    let mut files = Vec::new();
    walk_slang(kernels, kernels, &mut files);
    files.sort();
    assert!(
        !files.is_empty(),
        "no `.slang` source under {} — `runtime` compiles from these",
        kernels.display()
    );

    let mut generated = String::from(
        "// Generated by build.rs by walking the `.slang` tree.\n\
         pub static SOURCES: &[(&str, &str)] = &[\n",
    );
    for rel in &files {
        generated.push_str(&format!(
            "    ({rel:?}, include_str!(concat!({:?}, \"/\", {rel:?}))),\n",
            kernels.display().to_string()
        ));
    }
    generated.push_str("];\n");

    std::fs::write(out.join("sources.rs"), generated).expect("OUT_DIR is writable");
}

/// Every `.slang` under `dir`, as paths relative to `root`, with `/` separators.
fn walk_slang(root: &Path, dir: &Path, out: &mut Vec<String>) {
    let entries =
        std::fs::read_dir(dir).unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("a readable directory entry").path();
        if path.is_dir() {
            walk_slang(root, &path, out);
        } else if path.extension().is_some_and(|e| e == "slang") {
            let rel = path
                .strip_prefix(root)
                .expect("walked from `root`")
                .to_string_lossy()
                .replace('\\', "/");
            out.push(rel);
        }
    }
}

/// Write `modules.rs`: every compiled module, in the rlib.
///
/// # Why the words go in the binary and not beside it
///
/// The directory above is real and stays published, because a tool that wants
/// to disassemble a module should not have to extract one from an archive.
/// What it cannot be is how a SERVER finds its kernels. `OUT_DIR` is a path
/// inside a `target/` tree — it names a build, not a deployment — so a `pie`
/// that resolved its kernels there worked exactly until the machine that ran
/// it was not the machine that built it, and failed at the first fire rather
/// than at boot. Carrying 666 files beside the binary instead is the same
/// deployment with more moving parts: a release becomes an archive, `pie init`
/// has to write a path into a config, and every one of those is a way for the
/// words to go missing after the build proved they were there.
///
/// `kernels-wgpu` already refuses the same handoff and says so in
/// `src/source.rs` — naming THIS crate's directory as the shape worth not
/// copying. It is cheaper here than the argument needs: the whole compiled
/// tree is 5.5 MB, which is under a tenth of what `pie` already links.
///
/// The key is the FILE STEM (`foo`, `foo.coopmat`), not the entrypoint, because
/// that is the vocabulary `driver-vulkan`'s `Modules` seam already looks a
/// module up under. Embedding is meant to change where the words come from and
/// nothing about what they are called.
fn emit_table(out: &Path, variants: &BTreeMap<(String, Capability), Variant>, spv: &Path) {
    // Sorted by STEM rather than taken in the map's `(entrypoint, tier)` order,
    // because the library binary-searches this. The two orders agree for every
    // name Slang will accept as an entrypoint -- both are identifiers -- and
    // relying on that would be resting a lookup on a fact about a shader
    // language, one file away from anything that says so.
    let mut rows: Vec<(String, PathBuf)> = variants
        .iter()
        .map(|((entrypoint, _), variant)| {
            let file = variant.tier.module(entrypoint);
            let stem = file
                .strip_suffix(".spv")
                .expect("`Capability::module` names a `.spv`")
                .to_string();
            (stem, spv.join(&file))
        })
        .collect();
    rows.sort();

    let mut generated = String::from(
        "// Generated by build.rs. Every compiled SPIR-V module, keyed by file stem.\n\
         pub static MODULES: &[(&str, &[u8])] = &[\n",
    );
    for (stem, path) in &rows {
        // A generated Rust literal must not carry a Windows separator, and
        // `include_bytes!` takes the path as written.
        let path = path.to_string_lossy().replace('\\', "/");
        generated.push_str(&format!("    ({stem:?}, include_bytes!({path:?})),\n"));
    }
    generated.push_str("];\n");

    std::fs::write(out.join("modules.rs"), generated).expect("OUT_DIR is writable");
}

/// One `(shader, tier, define set)` triple — one SPIR-V module.
struct Variant {
    /// The `.slang` this is stamped from, relative to `kernels/`.
    file: PathBuf,
    /// Which optional device features the module is allowed to use.
    tier: Capability,
    /// The `-D` set, in the order the directive spells it.
    defines: Vec<(String, String)>,
}

/// Every variant the tree declares, keyed by `(entrypoint, tier)`.
///
/// The key is the artifact name, and it is a PAIR rather than a name because a
/// tier is an additional module for an entrypoint that already exists (see
/// [`Capability`]). Two directives claiming one entrypoint AT ONE TIER would
/// silently have the second overwrite the first's `.spv` — the Vulkan spelling
/// of the duplicate `host_name` the Metal tree's
/// `no_two_rows_claim_the_same_entrypoint` exists to catch. Here the collision
/// is caught while it is still a source fact.
fn collect(kernels: &Path) -> BTreeMap<(String, Capability), Variant> {
    let mut out: BTreeMap<(String, Capability), Variant> = BTreeMap::new();
    let mut files = Vec::new();
    walk(kernels, &mut files);
    files.sort();

    for path in files {
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let rel = path
            .strip_prefix(kernels)
            .expect("walk yields paths under kernels/")
            .to_path_buf();

        for (lineno, line) in text.lines().enumerate() {
            let Some(rest) = directive(line) else {
                continue;
            };
            let mut words = rest.split_whitespace().peekable();
            let entrypoint = words
                .next()
                .unwrap_or_else(|| {
                    panic!(
                        "{}:{}: a `pie:instantiate` names an entrypoint first",
                        rel.display(),
                        lineno + 1
                    )
                })
                .to_string();

            // The tier is optional and, when present, immediately follows the
            // name. Absent means baseline, so the 480 directives written before
            // tiers existed keep their meaning without being touched.
            let tier = match words.peek().and_then(|w| w.strip_prefix('@')) {
                Some(tag) => {
                    let cap = Capability::from_tag(tag).unwrap_or_else(|| {
                        panic!(
                            "{}:{}: `@{tag}` is not a capability tier; expected \
                             one of baseline, fp16, coopmat",
                            rel.display(),
                            lineno + 1
                        )
                    });
                    words.next();
                    cap
                }
                None => Capability::Baseline,
            };

            let defines = words
                .map(|w| {
                    let (k, v) = w.split_once('=').unwrap_or_else(|| {
                        panic!(
                            "{}:{}: `{w}` is not a `KEY=VALUE` define",
                            rel.display(),
                            lineno + 1
                        )
                    });
                    (k.to_string(), v.to_string())
                })
                .collect();

            if let Some(prior) = out.insert(
                (entrypoint.clone(), tier),
                Variant {
                    file: rel.clone(),
                    tier,
                    defines,
                },
            ) {
                panic!(
                    "`{entrypoint}` is instantiated twice at tier `{}`: {} and {}",
                    tier.tag(),
                    prior.file.display(),
                    rel.display()
                );
            }
        }
    }

    // A tier is a faster answer to a question the baseline already answers. A
    // tiered module with no baseline beside it would be an entrypoint that
    // exists only on some devices, which is precisely what the tier mechanism
    // is built to prevent -- catch it here, where the file and line are known,
    // rather than as a missing pipeline on a user's laptop.
    for (entrypoint, tier) in out.keys() {
        assert!(
            *tier == Capability::Baseline
                || out.contains_key(&(entrypoint.clone(), Capability::Baseline)),
            "`{entrypoint}` is instantiated at tier `{}` with no baseline; every \
             entrypoint must resolve on a device with no optional features",
            tier.tag()
        );
    }
    out
}

/// Where `slangc` is, for the shaders that have been ported to Slang.
///
/// The build asks the environment first so a machine with an unusual toolchain
/// does not have to patch this file. Failing to FIND it is a build error and
/// not a warning: `native` means "produce the modules", and a `native` build
/// that quietly produced none would hand the shell an empty pipeline cache and
/// let the failure surface at model load, one layer away from its cause.
///
/// Slang is not in Ubuntu and is not part of the Vulkan SDK's default install,
/// so unlike the `slangc` this tree used to call there is no version a runner
/// is likely to have by accident. What this tree is measured against is the release binary,
/// **2026.14.1**, from `shader-slang/slang`; CI fetches exactly that.
fn slangc() -> PathBuf {
    println!("cargo:rerun-if-env-changed=PIE_SLANGC");
    std::env::var_os("PIE_SLANGC").map_or_else(|| PathBuf::from("slangc"), PathBuf::from)
}

/// The directive's payload, if this line is one.
///
/// Leading whitespace is allowed and the marker is anchored to the start of the
/// COMMENT, so a `pie:instantiate` mentioned inside prose does not become a
/// build instruction.
fn directive(line: &str) -> Option<&str> {
    let line = line.trim_start();
    let rest = line.strip_prefix("//")?.trim_start();
    rest.strip_prefix("pie:instantiate").map(str::trim)
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read the directory {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("a readable directory entry").path();
        if path.is_dir() {
            walk(&path, out);
        } else if path.extension().is_some_and(|e| e == "slang") {
            out.push(path);
        }
    }
}

/// One Slang module.
///
/// The flags are all forced, and each is worth naming because a reader will
/// otherwise assume they are arbitrary -- several of them exist only because
/// this tree was once GLSL compiled by `glslc`, and the SPIR-V it emitted is
/// what every reflection and every test here was written against.
///
/// `-fvk-use-entrypoint-name` is the important one. Slang names the SPIR-V
/// entry point after the Slang function by default only for some targets; this
/// tree's shell looks a module up by its ENTRYPOINT, and the reflection in
/// `spirv.rs` reads that name out of the binary. Forcing it means a Slang
/// module is the same object to everything downstream that the GLSL module it
/// replaced was.
///
/// There is deliberately NO `-profile`. Passing `-profile glsl_450` is the
/// obvious thing to write for a tree ported out of `#version 450` GLSL and it is
/// wrong: it puts Slang into the SPIR-V 1.0 dialect, where a storage buffer is
/// a `BufferBlock` in the `Uniform` storage class rather than a `Block` in
/// `StorageBuffer`. Both load, which is why this was nearly missed -- the
/// difference is visible only by disassembling. Without the flag Slang emits
/// the modern form, which is what `glslc --target-env=vulkan1.3` emitted and
/// therefore what every reflection and every test in this tree was written
/// against.
///
/// `-allow-glsl` is on for one construct: the scalar parameter BLOCK. The
/// driver finds a launch's parameter block by its SIZE -- see
/// `binding::params`, and the two kernels that put theirs at a binding with
/// operands after it -- so a module has to declare "a storage buffer holding
/// exactly these bytes". Slang's own `StructuredBuffer<T>` cannot say that: it
/// is a runtime array, whose extent is the descriptor's business and not the
/// module's, so reflection correctly reports no size and the launch is refused
/// with "2 scalars stated, room for 0". A GLSL-style `buffer` block is the
/// only construct in either language that states a fixed extent, Slang
/// supports it as a first-class input, and it produces byte-for-byte the same
/// block `slangc` did. Everything else in these shaders is ordinary Slang.
///
/// `-matrix-layout-column-major` is off, because nothing here uses a matrix
/// type in an interface block; stating it anyway would be a claim about the
/// layout of something that does not exist.
///
/// The optimisation split is the same as `slangc`'s and rests on the same
/// borrowed finding (ggml #15344): the cooperative-matrix tier is built
/// unoptimised. Slang runs spirv-opt through the same library `slangc` does, so
/// the miscompile is the same miscompile.
fn compile_slang(slangc: &Path, kernels: &Path, spv: &Path, entrypoint: &str, variant: &Variant) {
    let src = kernels.join(&variant.file);
    let dst = spv.join(variant.tier.module(entrypoint));

    let mut cmd = Command::new(slangc);
    cmd.arg("-target")
        .arg("spirv")
        .arg("-stage")
        .arg("compute")
        .arg("-entry")
        .arg("main")
        .arg("-fvk-use-entrypoint-name")
        .arg("-emit-spirv-directly")
        .arg("-allow-glsl")
        .arg("-I")
        .arg(kernels)
        .arg(format!("-DPIE_ENTRYPOINT={entrypoint}"));
    if variant.tier == Capability::Coopmat {
        cmd.arg("-O0");
    } else {
        cmd.arg("-O2");
    }
    for (k, v) in &variant.defines {
        cmd.arg(format!("-D{k}={v}"));
    }
    cmd.arg("-o").arg(&dst).arg(&src);

    let status = cmd.status().unwrap_or_else(|e| {
        panic!(
            "cannot run `{}` (set PIE_SLANGC, or put slangc on PATH — see \
             `slangc()` for the release this tree is measured against): {e}",
            slangc.display()
        )
    });
    assert!(
        status.success(),
        "slangc failed for `{entrypoint}` at tier `{}` ({}): {status}",
        variant.tier.tag(),
        variant.file.display()
    );
}
