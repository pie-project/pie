use std::fs;
use std::path::{Path, PathBuf};

use pie_ptir::rng::{generate_cuda_header, generate_msl_preamble, hash_uniform, keyed_seed};

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
            "{} missing ({error}); regenerate with PTIR_REGEN=1 cargo test -p pie-ptir --test rng_contract",
            path.display()
        )
    });
    assert_eq!(
        actual,
        expected,
        "{} is stale; regenerate with PTIR_REGEN=1 cargo test -p pie-ptir --test rng_contract",
        path.display()
    );
}

#[test]
fn generated_rng_artifacts_are_uptodate() {
    let root = repo_root();
    check_or_regenerate(
        &root.join("driver/common/include/pie_native/ptir/rng_contract.generated.h"),
        &generate_cuda_header(),
    );
    check_or_regenerate(
        &root.join("driver/metal/src/kernels/ptir_rng.generated.metal"),
        &generate_msl_preamble(),
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

#[test]
fn rng_magic_is_owned_by_the_contract() {
    if std::env::var("PTIR_REGEN").is_ok() {
        return;
    }
    let root = repo_root();
    let mut files = Vec::new();
    visit_sources(&root, Path::new(""), &mut files);

    let owners = [
        Path::new("interface/ptir/src/rng.rs"),
        Path::new("driver/common/include/pie_native/ptir/rng_contract.generated.h"),
        Path::new("driver/metal/src/kernels/ptir_rng.generated.metal"),
    ];
    let unrelated_stride_users = [
        Path::new("driver/cuda/src/batch/forward_graph.hpp"),
        Path::new("driver/cuda/src/loader/weight_store_codec.hpp"),
        Path::new("driver/cuda/src/pipeline/program_identity.hpp"),
        Path::new("gateway/src/route.rs"),
        Path::new("runtime/engine/src/inferlet/linker.rs"),
    ];
    let unrelated_mask_users = [
        Path::new("driver/cuda/tests/ptir_tier0_test.cu"),
        Path::new("runtime/grammar/src/brle.rs"),
    ];
    let stride = ["9e37", "79b9", "7f4a", "7c15"].concat();
    let ambient_mask = ["a5a5", "a5a5"].concat();
    let magic = [
        ["3c79", "ac49", "2ba7", "b653"].concat(),
        ["1c69", "b3f7", "4ac4", "ae35"].concat(),
        stride.clone(),
        ambient_mask.clone(),
        [">>", "40"].concat(),
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
            if !normalized.contains(needle) || owners.contains(&relative.as_path()) {
                continue;
            }
            if needle == &stride && unrelated_stride_users.contains(&relative.as_path()) {
                continue;
            }
            if needle == &ambient_mask && unrelated_mask_users.contains(&relative.as_path()) {
                continue;
            }
            panic!(
                "PTIR RNG magic `{needle}` is independently transcribed in {}",
                relative.display()
            );
        }
    }
}
