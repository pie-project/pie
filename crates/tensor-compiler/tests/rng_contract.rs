//! The RNG constants live in exactly one place.
//!
//! Every backend has to reproduce PTIR's noise bit for bit, so each of them
//! needs the same magic numbers. These tests check that the numbers reach the
//! generated sources by projection from `tensor_ir::rng::RNG_FORMULA` rather than
//! by transcription: a transcribed constant keeps compiling after the formula
//! moves, and the divergence only shows up as a replay that no longer matches
//! its original.

use std::fs;
use std::path::{Path, PathBuf};

use tensor_compiler::codegen::rng::{cuda_device_functions, generate_msl_preamble};
use tensor_ir::rng::{RNG_FORMULA, UNIFORM_MAX, hash_uniform, keyed_seed};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn check_or_regenerate(path: &Path, expected: &str) {
    if std::env::var("PTIR_REGEN").is_ok() {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, expected).unwrap();
        return;
    }
    let actual = fs::read_to_string(path).unwrap_or_else(|error| {
        panic!(
            "{} missing ({error}); regenerate with PTIR_REGEN=1 cargo test -p pie-compiler-tests --test rng_contract",
            path.display()
        )
    });
    assert_eq!(
        actual,
        expected,
        "{} is stale; regenerate with PTIR_REGEN=1 cargo test -p pie-compiler-tests --test rng_contract",
        path.display()
    );
}

/// The one generated artifact left, and why there is only one.
///
/// `rng_contract.generated.h` stood beside it: the same contract in C, for the
/// C++ drivers to `#include`. They were deleted, every backend is Rust now,
/// and the CUDA device text is spliced straight out of
/// [`cuda_device_functions`] by the emitter rather than compiled from a header
/// — NVRTC is called with zero headers and zero include names, so no include
/// path could have reached one. MSL is the exception that keeps this test:
/// Metal's runtime shader compiler resolves `#include "..."` against the
/// including file's directory, so the preamble has to exist as a FILE.
#[test]
fn generated_rng_artifacts_are_uptodate() {
    let root = repo_root();
    let msl = generate_msl_preamble();
    check_or_regenerate(
        &root.join("crates/tensor-compiler/include/ptir_rng.generated.metal"),
        &msl,
    );
}

#[test]
fn keyed_rng_byte_parity_vectors() {
    let cases = [
        (0x0000_0000, 0x0000_0000, 0),
        (0x0000_0001, 0x0000_0000, 0),
        (0x0000_04d2, 0x0000_0000, 0),
        (0x0000_04d2, 0x0000_0000, 1),
        (0xffff_ffff, 0xffff_ffff, 31),
        (0x1234_5678, 0x9abc_def0, 7),
    ];
    let expected_hashes: [u64; 6] = [
        0x0000_0000_0000_0000,
        0x5237_5cd7_3dbe_d523,
        0x2db5_6ca5_bfd5_b704,
        0x2db5_6ca5_bfd5_b704,
        0x78a9_666a_39c1_a1b5,
        0x3b88_23c5_eac7_f534,
    ];
    let expected_uniform_bits: [u32; 6] = [
        0x3f37_0fb2,
        0x3f60_2672,
        0x3ebc_c971,
        0x3e68_2006,
        0x3f49_0c40,
        0x3d21_10a8,
    ];

    for (index, &(key, counter, lane)) in cases.iter().enumerate() {
        let seed = keyed_seed(key, counter);
        assert_eq!(seed.to_le_bytes(), expected_hashes[index].to_le_bytes());
        assert_eq!(
            hash_uniform(seed, lane).to_bits().to_le_bytes(),
            expected_uniform_bits[index].to_le_bytes()
        );
    }
}

/// The uniform must stay strictly inside `[0, 1)`.
///
/// `(bits + 0.5) / 2^24` is mathematically below 1 for every representable
/// `bits`, but at `bits = 2^24 - 1` the quotient lands exactly halfway between
/// `0x1.fffffep-1` and `1.0` and round-to-even snaps it to `1.0`. Consumers
/// read the draw as `gumbel = -ln(-ln(u))`, which is `+inf` at `u = 1`, and
/// `+inf` wins every `argmax` it is added to — one uniformly random token per
/// `2^24 / vocab` decode steps. The clamp is what keeps that from happening,
/// so this test walks all `2^24` mantissa values rather than sampling.
#[test]
fn uniform_never_reaches_one() {
    let denominator = (1u32 << RNG_FORMULA.uniform_mantissa_bits) as f32;
    let mut unclamped_hits = 0u32;

    for bits in 0..(1u32 << RNG_FORMULA.uniform_mantissa_bits) {
        let raw = (bits as f32 + RNG_FORMULA.uniform_midpoint) * (1.0 / denominator);
        if raw >= 1.0 {
            unclamped_hits += 1;
        }
        let clamped = if raw < UNIFORM_MAX { raw } else { UNIFORM_MAX };
        assert!(
            (0.0..1.0).contains(&clamped),
            "bits {bits} left the half-open unit interval: {clamped}"
        );
        assert!(
            (-(-clamped.ln()).ln()).is_finite(),
            "bits {bits} produced a non-finite gumbel"
        );
    }

    assert_eq!(
        unclamped_hits, 1,
        "expected exactly one of 2^24 draws to round up to 1.0 without the clamp"
    );
    const _: () = assert!(UNIFORM_MAX < 1.0);
    assert_eq!(UNIFORM_MAX.to_bits(), 0x3f7f_ffff);
}

/// Both generated backends must carry the clamp, or the device diverges from
/// the host on the one draw that matters.
#[test]
fn generated_backends_clamp_the_uniform() {
    for (backend, source) in [
        ("cuda", cuda_device_functions()),
        ("metal", generate_msl_preamble()),
    ] {
        assert!(
            source.contains("0.99999994f"),
            "{backend} backend lost the uniform clamp"
        );
    }
}

fn visit_sources(root: &Path, relative: &Path, files: &mut Vec<PathBuf>) {
    let path = root.join(relative);
    for entry in fs::read_dir(path).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        let child_relative = relative.join(&name);
        let file_type = entry.file_type().unwrap();
        if file_type.is_dir() {
            if matches!(
                name.to_str(),
                Some(
                    ".git"
                        | ".claude"
                        | "target"
                        | "build"
                        | "_deps"
                        | "node_modules"
                        | "third_party"
                        | "vendor"
                )
            ) {
                continue;
            }
            visit_sources(root, &child_relative, files);
            continue;
        }
        let entry_path = entry.path();
        let Some(extension) = entry_path.extension().and_then(|value| value.to_str()) else {
            continue;
        };
        if matches!(
            extension,
            "rs" | "c" | "cc" | "cpp" | "cu" | "cuh" | "h" | "hpp" | "metal"
        ) {
            files.push(child_relative);
        }
    }
}

/// The one allowlisted path that is generated rather than committed.
///
/// `tensor-compiler` owns the preamble's text and commits it once, under
/// `include/`. Metal's runtime shader compiler resolves a `#include "..."`
/// against the including file's directory and nothing else, so a shader that
/// lives in the runtime shader directory can only reach the preamble if a copy
/// sits there too: `kernels-metal/build.rs` writes that copy on a `native`
/// build and `.gitignore` keeps it from becoming a second committed source.
///
/// It is therefore an owner — the contract's own text, at the second place the
/// contract requires it — and the only entry [`allowlisted_paths_still_exist`]
/// exempts, because a checkout that has not built it is not a broken checkout.
const STAGED_RNG_PREAMBLE: &str = "crates/kernels-metal/kernels/ptir/ptir_rng.generated.metal";

/// The four allowlists [`rng_magic_is_owned_by_the_contract`] reads, named so
/// that [`allowlisted_paths_still_exist`] can check the same lists rather than
/// a copy of them.
struct Allowlists {
    owners: &'static [&'static str],
    stride: &'static [&'static str],
    mask: &'static [&'static str],
    shift: &'static [&'static str],
}

impl Allowlists {
    fn lists(&self) -> [(&'static str, &'static [&'static str]); 4] {
        [
            ("owners", self.owners),
            ("unrelated_stride_users", self.stride),
            ("unrelated_mask_users", self.mask),
            ("unrelated_shift_users", self.shift),
        ]
    }
}

fn path_in(list: &[&str], relative: &Path) -> bool {
    list.iter().any(|entry| Path::new(entry) == relative)
}

fn allowlists() -> Allowlists {
    Allowlists {
        // Repo-wide on purpose: the point is that nobody outside the contract
        // — a driver, a kernel, the engine — re-types these constants.
        owners: &[
            "crates/tensor-ir/src/rng.rs",
            // `crates/tensor-compiler/include/rng_contract.generated.h` was the
            // third owner: the contract in C, for the C++ drivers to
            // `#include`. It was deleted with them -- and this entry in the
            // same commit, which is what [`allowlisted_paths_still_exist`] is
            // for.
            "crates/tensor-compiler/include/ptir_rng.generated.metal",
            // See `STAGED_RNG_PREAMBLE`: the same generated text, at the second
            // place the contract requires it.
            STAGED_RNG_PREAMBLE,
        ],
        stride: &[
            "crates/gateway/src/route.rs",
            "crates/engine/src/inferlet/linker.rs",
            // splitmix64 id generation: the canonical splitmix increment
            // happens to be the same golden-ratio word; not a keyed-RNG
            // transcription.
            "crates/engine/src/pipeline/offload.rs",
            // The staged lora table's fingerprint, a splitmix mixer over what
            // a captured lora body bakes. Same word, same reason, and it is
            // here rather than deduplicated because it has to agree with the
            // capture it fingerprints, branch for branch.
            //
            // Two C++ siblings used to sit beside it — `llama_like.cpp`'s
            // NS-3 spatial-split plan fingerprint and `dispatch.cu`'s
            // stage-hook `hash_combine` — and a Rust `model/llama_like.rs`
            // carrying the NS-3 layout key. All three went with the C++
            // driver and the module that had no caller.
            "crates/driver-cuda/src/fire/lora.rs",
            // boost-style `hash_combine` for the GEMM autotune cache key; the
            // golden-ratio word again, and nothing to do with the PTIR stream.
            // These have moved twice — `driver-cuda/ops/` to `kernels-cuda/ops/`
            // when the kernel crate split, that `kernels-cuda` being the
            // ahead-of-time archive crate deleted at `85c6c674b`, then into the
            // family layout — and both times `allowlisted_paths_still_exist`
            // named the stale entry instead of letting the guard fail somewhere
            // unrelated.
            //
            // THREE TIMES NOW, and the third is a language change rather than
            // a move: `kernels-cuda/csrc/src/gemm/gemm.cpp` is deleted, with
            // `tuning_cache.hpp` and `cache_root.hpp` — its only includer's
            // only include. It held zero `__global__` and zero `<<<>>>` — a
            // host program, not a kernel file — and its dense autotuner,
            // `tuning_hash` and all, is Rust now, in the entry below. Same
            // word, same `tuning_cache` lineage, same non-transcription.
            //
            // The CUTLASS MoE tactic cache is the same format again
            // (`moe/flashinfer_moe.cu` was a host program too), and both
            // autotuners came to rest in the same module when the families
            // crossed. ONE entry covers what was briefly two `driver-cuda`
            // ones, which is the move this list is built to survive.
            "crates/kernels-cuda/src/gemm/dense.rs",
            // SEEDS, not streams. The four below fill TEST DATA with a
            // seeded generator, and the golden-ratio word is the obvious
            // constant to seed one with -- it is in splitmix64, in boost's
            // `hash_combine`, and in half the fixtures ever written. None of
            // them reproduces the PTIR stream, which is what this guard is
            // for; the scanner cannot tell a seed from a transcription, which
            // is what this list is for.
            //
            // A per-request weight that breaks ties in an argmax plateau: a
            // multiplicative hash with no period this side of 2^64, so
            // distinct rows get distinct weights and a maximum is a maximum.
            "crates/driver-vulkan/tests/device.rs",
            // The GEMM service parity pair. The seed has to be the SAME word
            // on both sides -- a Rust harness and a C++ oracle comparing
            // element for element -- so it cannot be shared through a helper
            // either crate could call, and the `.cu` half cannot call the
            // Rust one at all.
            "crates/driver-cuda/tests/gemm_service_parity.rs",
            "crates/driver-cuda/tests/oracle/gemm_service/oracle.cu",
            "crates/driver-cuda/tests/oracle/gemm_service/bias_fold.cu",
            // A fifth entry, `crates/kernels-cuda/examples/fp8_pipeline_probe.rs`,
            // held a splitmix state in an example that generated its own
            // inputs. It went with `kernels-cuda/examples/`, and this entry
            // went in the same commit -- which is what
            // [`allowlisted_paths_still_exist`] is for, and what every move
            // and deletion in that test's own doc failed to do.
        ],
        mask: &["crates/grammar/src/brle.rs"],
        // The float conversion's shift is the weakest needle here: any 64-bit
        // -> float reduction that wants 24 mantissa bits writes it. A murmur3
        // finalizer in a driver test is not a PTIR stream.
        //
        // (Spelled in fragments at the use site like every other constant, and
        // for the reason this guard exists: writing it out in a comment makes
        // THIS file a transcription, which is exactly what the first draft of
        // this comment did and what the guard then reported.)
        //
        // Empty AGAIN, and left in place rather than deleted for the reason
        // it was left in place the first time: an empty list is the statement
        // that the exemption is unused, which is worth more than the absence
        // of a field.
        //
        // Twice now the one entry has been a test that went away rather than
        // an exemption that was withdrawn -- first a Metal numerics test with
        // the C++ driver, now `kernels-cuda`'s `fa4_fires`, whose whole
        // `tests/` directory is gone. That is the pattern this field's
        // companion gate is for: an allowlist keyed by path cannot tell "the
        // file stopped needing the exemption" from "the file stopped
        // existing", and only one of those is progress. Nothing outside the
        // contract writes this shift today either way.
        shift: &[],
    }
}

/// An allowlist keyed by path rots silently, and this is what stops it.
///
/// Every entry above says "the magic in THIS file is not a transcription". When
/// the file moves, the entry does not follow it: the entry becomes dead, the
/// file reappears at a new path the guard has never heard of, and
/// [`rng_magic_is_owned_by_the_contract`] fails somewhere far from the move
/// that caused it. That happened three times over one refactor — `ops/gemm.cpp`
/// and `ops/tuning_cache.hpp` moved to `kernels-cuda` (the archive crate,
/// deleted at `85c6c674b`), `program_identity.hpp`
/// was deleted outright, and the staged MSL preamble moved to `kernels-metal` —
/// and each was found by the failure rather than by the move.
///
/// So the list is checked against the tree it describes. A rename now fails
/// HERE, naming the entry, in the commit that renamed it.
#[test]
fn allowlisted_paths_still_exist() {
    let root = repo_root();
    for (name, list) in allowlists().lists() {
        for entry in list {
            // The one generated entry: absent until a `native` build stages it,
            // and a checkout that has not built is not a broken checkout.
            if *entry == STAGED_RNG_PREAMBLE {
                continue;
            }
            assert!(
                root.join(entry).exists(),
                "`{name}` allowlists {entry}, which no longer exists. Point the \
                 entry at the file's new home, or drop it if the file is gone."
            );
        }
    }
}

/// The constants, spelled in fragments so that THIS file does not become the
/// transcription the guard hunts. Three of them excuse a path in an allowlist,
/// so they are named; the other three are the stream's own and may appear
/// nowhere but an owner.
struct Needles {
    stride: String,
    mask: String,
    shift: String,
    all: Vec<String>,
}

fn needles() -> Needles {
    let stride = ["9e37", "79b9", "7f4a", "7c15"].concat();
    let mask = ["a5a5", "a5a5"].concat();
    let shift = [">>", "40"].concat();
    let all = vec![
        ["3c79", "ac49", "2ba7", "b653"].concat(),
        ["1c69", "b3f7", "4ac4", "ae35"].concat(),
        stride.clone(),
        mask.clone(),
        shift.clone(),
        ["16777216", ".0"].concat(),
    ];
    Needles {
        stride,
        mask,
        shift,
        all,
    }
}

fn normalize(text: &str) -> String {
    text.chars()
        .filter(|character| *character != '_' && !character.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

/// The other way an allowlist rots, and the one
/// [`allowlisted_paths_still_exist`] cannot see.
///
/// An entry says "the magic in this file is not a transcription". When the
/// file stops carrying the magic — the fingerprint is deleted, the helper is
/// shared, the hash changes — the entry survives, because the path is still
/// there. What is left is a standing exemption for a file that no longer
/// needs one, and the next transcription written into it passes.
///
/// That is not hypothetical either: this suite's stride list carried four
/// entries whose files had been deleted with the C++ CUDA driver, and one
/// that had merely MOVED. Only the move was reported, and only because the
/// magic reappeared at a path the guard had never heard of. Had the file been
/// deleted instead, nothing would have said so.
#[test]
fn allowlist_entries_still_carry_what_they_excuse() {
    let root = repo_root();
    let needles = needles();
    let allow = allowlists();
    for (name, list, needle) in [
        ("unrelated_stride_users", allow.stride, &needles.stride),
        ("unrelated_mask_users", allow.mask, &needles.mask),
        ("unrelated_shift_users", allow.shift, &needles.shift),
    ] {
        for entry in list {
            let contents =
                fs::read_to_string(root.join(entry)).unwrap_or_else(|e| panic!("{entry}: {e}"));
            assert!(
                normalize(&contents).contains(needle),
                "`{name}` excuses {entry}, which no longer contains the constant \
                 it was excused for. The exemption outlived its reason: drop the \
                 entry, or it will silently permit the next transcription written \
                 into that file."
            );
        }
    }
}

#[test]
fn rng_magic_is_owned_by_the_contract() {
    if std::env::var("PTIR_REGEN").is_ok() {
        return;
    }
    let root = repo_root();
    let mut files = Vec::new();
    visit_sources(&root, Path::new(""), &mut files);

    let allow = allowlists();
    let owner = |relative: &Path| path_in(allow.owners, relative);
    let needles = needles();

    // Collected rather than panicked on the first hit. A move that lands the
    // magic in two files reports one, gets repointed, and reports the other
    // on the next run — and the second report reads like a new regression
    // rather than the rest of the same one.
    let mut found = Vec::new();
    for relative in files {
        let contents = fs::read_to_string(root.join(&relative)).unwrap();
        let normalized = normalize(&contents);
        for needle in &needles.all {
            if !normalized.contains(needle) || owner(&relative) {
                continue;
            }
            if needle == &needles.stride && path_in(allow.stride, &relative) {
                continue;
            }
            if needle == &needles.mask && path_in(allow.mask, &relative) {
                continue;
            }
            if needle == &needles.shift && path_in(allow.shift, &relative) {
                continue;
            }
            found.push(format!("`{needle}` in {}", relative.display()));
        }
    }
    assert!(
        found.is_empty(),
        "PTIR RNG magic is independently transcribed in {} place(s):\n  {}",
        found.len(),
        found.join("\n  ")
    );
}

/// The emitted CUDA runtime carries the device RNG text, and gets it from
/// [`cuda_device_functions`].
///
/// This checked something narrower until the C++ drivers were deleted: a
/// checked-in `rng_contract.generated.h` embedded the same text in a
/// `PTIR_RNG_CUDA_PREAMBLE` raw-string literal, and the test found the literal
/// by its delimiters and compared it against what the emitter splices. The
/// header is gone with its last includer, so what is left is the half that was
/// always about the emitter: a runtime template that silently stopped
/// splicing the preamble would emit sources whose `ptir_rng_*` calls do not
/// resolve, and NVRTC — called with no include path — has nowhere to find them.
#[test]
fn emitted_cuda_runtime_carries_the_rng_preamble() {
    let spliced = tensor_compiler::codegen::cuda::singleton_runtime_source();
    let functions = cuda_device_functions();
    assert!(
        spliced.contains(&functions),
        "the emitted CUDA runtime does not contain the RNG device functions"
    );
    for name in [
        "ptir_rng_splitmix64",
        "ptir_rng_seed_eff",
        "ptir_rng_stream_salt",
        "ptir_rng_seed_eff_stream",
        "ptir_rng_keyed_seed",
        "ptir_rng_hash_uniform",
    ] {
        assert!(
            functions.contains(name),
            "the CUDA projection no longer defines {name}"
        );
    }
}
