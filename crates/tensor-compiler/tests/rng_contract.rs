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

use tensor_compiler::codegen::rng::{generate_cuda_header, generate_msl_preamble};
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

#[test]
fn generated_rng_artifacts_are_uptodate() {
    let root = repo_root();
    let cuda = generate_cuda_header();
    let msl = generate_msl_preamble();
    check_or_regenerate(
        &root.join("crates/tensor-compiler/include/rng_contract.generated.h"),
        &cuda,
    );
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
        ("cuda", generate_cuda_header()),
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
            "crates/tensor-compiler/include/rng_contract.generated.h",
            "crates/tensor-compiler/include/ptir_rng.generated.metal",
            // See `STAGED_RNG_PREAMBLE`: the same generated text, at the second
            // place the contract requires it.
            STAGED_RNG_PREAMBLE,
        ],
        stride: &[
            "crates/driver-cuda/csrc/src/batch/forward_graph.hpp",
            "crates/driver-cuda/csrc/src/loader/weight_store_codec.hpp",
            "crates/gateway/src/route.rs",
            "crates/engine/src/inferlet/linker.rs",
            // splitmix64 id generation: the canonical splitmix increment
            // happens to be the same golden-ratio word; not a keyed-RNG
            // transcription.
            "crates/engine/src/pipeline/offload.rs",
            // Same word, same reason: two splitmix mixers fingerprinting what a
            // captured graph body bakes (the NS-3 spatial-split plan, and the
            // staged lora table). Added by `f4a63579b` / `d7df4b575` without a
            // row here, which is why this guard only started reporting them
            // once the workspace test sweep could reach this suite again.
            "crates/driver-cuda/csrc/src/model/llama_like/llama_like.cpp",
            // The RUST port of the staged lora table's fingerprint,
            // transcribed branch for branch with its splitmix finalizer.
            // Same word, same reason, and it is here rather than
            // deduplicated because a port that reached for a shared helper
            // would no longer be a transcription of the function it has to
            // agree with.
            //
            // `model/llama_like.rs` used to sit beside it, carrying the
            // NS-3 spatial-split layout key. That module is gone (2026-08-10,
            // `cuda.md` §3.4.1): nothing in the driver called it, so the
            // layout key it transcribed had no caller either.
            "crates/driver-cuda-new/src/model/lora.rs",
            // boost-style `hash_combine` for the GEMM autotune cache key; the
            // golden-ratio word again, and nothing to do with the PTIR stream.
            // These have moved twice — `driver-cuda/ops/` to `kernels-cuda/ops/`
            // when the kernel crate split, then into the family layout — and
            // both times `allowlisted_paths_still_exist` named the stale entry
            // instead of letting the guard fail somewhere unrelated.
            "crates/kernels-cuda/csrc/src/gemm/gemm.cpp",
            "crates/kernels-cuda/csrc/src/tuning_cache.hpp",
            // And again, for the stage-hook fingerprint's `hash_combine`.
            "crates/driver-cuda/csrc/src/pipeline/dispatch.cu",
        ],
        mask: &[
            "crates/driver-cuda/csrc/tests/ptir_tier0_test.cu",
            "crates/grammar/src/brle.rs",
        ],
        // The float conversion's shift is the weakest needle here: any 64-bit
        // -> float reduction that wants 24 mantissa bits writes it. A murmur3
        // finalizer in a driver test is not a PTIR stream.
        //
        // (Spelled in fragments at the use site like every other constant, and
        // for the reason this guard exists: writing it out in a comment makes
        // THIS file a transcription, which is exactly what the first draft of
        // this comment did and what the guard then reported.)
        shift: &["crates/driver-metal/csrc/tests/llama_numerics_test.cpp"],
    }
}

/// An allowlist keyed by path rots silently, and this is what stops it.
///
/// Every entry above says "the magic in THIS file is not a transcription". When
/// the file moves, the entry does not follow it: the entry becomes dead, the
/// file reappears at a new path the guard has never heard of, and
/// [`rng_magic_is_owned_by_the_contract`] fails somewhere far from the move
/// that caused it. That happened three times over one refactor — `ops/gemm.cpp`
/// and `ops/tuning_cache.hpp` moved to `kernels-cuda`, `program_identity.hpp`
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

    let stride = ["9e37", "79b9", "7f4a", "7c15"].concat();
    let ambient_mask = ["a5a5", "a5a5"].concat();
    let float_shift = [">>", "40"].concat();
    let magic = [
        ["3c79", "ac49", "2ba7", "b653"].concat(),
        ["1c69", "b3f7", "4ac4", "ae35"].concat(),
        stride.clone(),
        ambient_mask.clone(),
        float_shift.clone(),
        ["16777216", ".0"].concat(),
    ];

    for relative in files {
        let contents = fs::read_to_string(root.join(&relative)).unwrap();
        let normalized: String = contents
            .chars()
            .filter(|character| *character != '_' && !character.is_whitespace())
            .flat_map(char::to_lowercase)
            .collect();
        for needle in &magic {
            if !normalized.contains(needle) || owner(&relative) {
                continue;
            }
            if needle == &stride && path_in(allow.stride, &relative) {
                continue;
            }
            if needle == &ambient_mask && path_in(allow.mask, &relative) {
                continue;
            }
            if needle == &float_shift && path_in(allow.shift, &relative) {
                continue;
            }
            panic!(
                "PTIR RNG magic `{needle}` is independently transcribed in {}",
                relative.display()
            );
        }
    }
}

/// The emitted CUDA source and the checked-in C++ header carry the same device
/// RNG text, and both must keep coming from `render_cuda_functions`.
///
/// Both callers reach `render_cuda_functions` directly; neither recovers the
/// preamble by generating the header and searching for the raw literal's
/// delimiters, which would make the emitter fail on a whitespace edit to a
/// header. This test is what ties the two outputs together instead: the
/// header's embedded literal must be exactly what the emitter splices in.
#[test]
fn cuda_preamble_matches_the_header_literal() {
    let header = generate_cuda_header();
    let open = "inline constexpr char PTIR_RNG_CUDA_PREAMBLE[] = R\"PTIR_RNG_CUDA(";
    let close = ")PTIR_RNG_CUDA\";";
    let start = header.find(open).expect("header defines the preamble") + open.len();
    let end = header[start..].find(close).expect("literal terminates") + start;

    let spliced = tensor_compiler::codegen::cuda::singleton_runtime_source();
    let embedded = &header[start..end];
    assert!(
        spliced.contains(embedded),
        "the emitted CUDA runtime does not contain the header's preamble literal"
    );
    assert_eq!(
        embedded,
        format!("\n{}", tensor_compiler::codegen::rng::cuda_device_functions()),
        "the header literal and the emitter's preamble have drifted"
    );
}
