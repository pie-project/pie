//! Publish `kernels/` to the Vulkan shell, and — under `native` — compile every
//! declared variant to SPIR-V.
//!
//! ## Why there IS a compile step here, where the Metal sibling has none
//!
//! Metal compiles its shaders at RUN time: the driver hands `.metal` source and
//! an entry name to the Metal compiler and gets a pipeline state object back,
//! so `kernels-metal`'s build script copies one generated file and stops.
//! Vulkan has no such door. `vkCreateShaderModule` takes SPIR-V words, and
//! nothing in the loader turns GLSL into them — so the GLSL -> SPIR-V hop is a
//! BUILD-time obligation, and this is where it is paid.
//!
//! That is also what makes the entrypoint set mechanical here in a way it is
//! not on Metal. A GLSL compute shader has exactly one entry point and it is
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
//! This is the same decision `.wiki/kernel-metal-refactor.md` §2 records for
//! the Metal tree — the macro that stamps the instantiations sits next to the
//! template, so a reader checking coverage reads one file — with the one
//! difference GLSL forces: a `#define` matrix cannot be expanded by the
//! preprocessor into differently-NAMED entry points, so the matrix is a
//! directive a build reads rather than a macro a compiler expands.
//!
//! `scripts/vulkan-kernel-audit.py` reads the same lines to write
//! `entrypoints.generated.txt`, and `tests/entrypoints.rs` pins that against
//! the table's product. Three readers, one source of truth.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

// The tier vocabulary, read straight out of the library's source. A build
// script cannot depend on the crate it builds, and `kernels-cuda`'s build
// script has the same problem with its tables and solves it the same way. The
// point is that the names a module is STAMPED with here and the names a driver
// LOOKS UP through `kernels_vulkan::Capability` are one definition.
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
    // `kernels_dir` is the GLSL tree. A shell that ships prebuilt SPIR-V never
    // needs it; a tool that re-derives the variant set does.
    println!("cargo:include={}", include.display());
    println!("cargo:kernels_dir={}", kernels.display());

    if std::env::var_os("CARGO_FEATURE_NATIVE").is_none() {
        return;
    }

    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    let spv = out.join("spv");
    std::fs::create_dir_all(&spv).expect("create the SPIR-V output directory");

    let glslc = glslc();
    let variants = collect(&kernels);
    assert!(
        !variants.is_empty(),
        "no `// pie:instantiate` directive under {} — the shader tree cannot \
         be empty and the directive is how a variant is declared",
        kernels.display()
    );

    for ((entrypoint, _), variant) in &variants {
        compile(&glslc, &kernels, &spv, entrypoint, variant);
    }

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

/// One `(shader, tier, define set)` triple — one SPIR-V module.
struct Variant {
    /// The `.comp` this is stamped from, relative to `kernels/`.
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

/// The SPIR-V compiler.
///
/// `PIE_GLSLC` first, so a build on a box whose `glslc` is not on `PATH` — the
/// Vulkan SDK installs to a versioned prefix — can say where it is rather than
/// having to shim it. Failing to FIND it is a build error and not a warning:
/// `native` means "produce the modules", and a `native` build that quietly
/// produced none would hand the shell an empty pipeline cache and let the
/// failure surface at model load, one layer away from its cause.
fn glslc() -> PathBuf {
    println!("cargo:rerun-if-env-changed=PIE_GLSLC");
    std::env::var_os("PIE_GLSLC").map_or_else(|| PathBuf::from("glslc"), PathBuf::from)
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
        } else if path.extension().is_some_and(|e| e == "comp") {
            out.push(path);
        }
    }
}

fn compile(glslc: &Path, kernels: &Path, spv: &Path, entrypoint: &str, variant: &Variant) {
    let src = kernels.join(&variant.file);
    let dst = spv.join(variant.tier.module(entrypoint));

    let mut cmd = Command::new(glslc);
    cmd.arg("-fshader-stage=compute")
        // 1.3 is what the subgroup and 16-bit-storage extensions the tree uses
        // are core or promoted in, and what llama.cpp's Vulkan backend targets
        // for everything but its cooperative-matrix path.
        .arg("--target-env=vulkan1.3");

    // spirv-opt is skipped for the cooperative-matrix tier, and this is a
    // borrowed finding rather than caution: llama.cpp's `vulkan-shaders-gen`
    // disables optimization for its coopmat and bf16 shaders because spirv-opt
    // miscompiles them (ggml #15344). The baseline tier keeps `-O` -- its bf16
    // is plain `uint16_t` storage with integer shifts, which is not the pattern
    // that bug is about.
    if variant.tier != Capability::Coopmat {
        cmd.arg("-O");
    }

    cmd
        // The tree's own headers (`common/*.glsl`, `*_params.glsl`) resolve
        // relative to the shader root, so an include reads the same in every
        // family.
        .arg("-I")
        .arg(kernels)
        // The name a module knows itself by. It is what the shell looks a
        // pipeline up under, so it has to be the entrypoint and not the file.
        .arg(format!("-DPIE_ENTRYPOINT={entrypoint}"));
    for (k, v) in &variant.defines {
        cmd.arg(format!("-D{k}={v}"));
    }
    cmd.arg("-o").arg(&dst).arg(&src);

    let status = cmd.status().unwrap_or_else(|e| {
        panic!(
            "cannot run `{}` (set PIE_GLSLC, or install the Vulkan SDK / \
             shaderc): {e}",
            glslc.display()
        )
    });
    assert!(
        status.success(),
        "glslc failed for `{entrypoint}` at tier `{}` ({}): {status}",
        variant.tier.tag(),
        variant.file.display()
    );
}
