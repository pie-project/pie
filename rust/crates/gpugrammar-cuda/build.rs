//! Compile the kernels to a fatbin and hand it to `lib.rs` to embed.
//!
//! A fatbin rather than a PTX string because PTX is compiled by the driver on
//! first load, which is the JIT this rewrite exists partly to remove - vLLM
//! reports the Triton one as a latency spike during inference. A fatbin holds
//! real SASS for every architecture named below, plus PTX for the newest so a
//! future card still runs.
//!
//! Nothing here needs a CUDA toolkit at *install* time; this runs when the
//! crate is built, and the wheel carries the result.

use std::path::{Path, PathBuf};
use std::process::Command;

/// SASS for these, so no driver compilation on load.
///
/// A100 is 80 and is what everything in this project has been measured on;
/// 86/89 are the consumer and L40 parts a user is most likely to have; 90 is
/// H100 and 100 is B200. Each costs about 7 KB of fatbin.
const ARCHITECTURES: &[&str] = &["80", "86", "89", "90", "100"];

/// And PTX for this one, as the forward-compatible tail. The driver will
/// compile it for anything newer, which is slow but is better than not running.
const NEWEST: &str = "120";

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=GPUGRAMMAR_SKIP_CUDA");

    let kernels = Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
    let mut sources: Vec<PathBuf> = std::fs::read_dir(&kernels)
        .expect("the kernels directory should exist")
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().is_some_and(|e| e == "cu"))
        .collect();
    // Sorted, so the fatbin is the same bytes from the same sources - a build
    // that differs only in readdir order is a build nobody can compare.
    sources.sort();
    for source in &sources {
        println!("cargo:rerun-if-changed={}", source.display());
    }
    for header in std::fs::read_dir(&kernels).into_iter().flatten().flatten() {
        if header.path().extension().is_some_and(|e| e == "cuh") {
            println!("cargo:rerun-if-changed={}", header.path().display());
        }
    }

    let out = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    let fatbin = out.join("gpugrammar.fatbin");

    // A crate that cannot find nvcc still has to build, or every contributor
    // without a CUDA toolkit is blocked from touching the Rust front end.
    // `available()` then reports false and the CUDA backend is simply absent.
    if std::env::var_os("GPUGRAMMAR_SKIP_CUDA").is_some() {
        std::fs::write(&fatbin, []).expect("writing an empty fatbin should work");
        return;
    }
    let Some(nvcc) = find_nvcc() else {
        println!("cargo:warning=nvcc not found; building without the CUDA backend");
        std::fs::write(&fatbin, []).expect("writing an empty fatbin should work");
        return;
    };

    let mut command = Command::new(&nvcc);
    command.arg("-fatbin").arg("-O3").arg("--std=c++17");
    // Line info costs nothing at run time and is what makes a
    // `compute-sanitizer` report name a line rather than an address.
    command.arg("-lineinfo");
    for arch in ARCHITECTURES {
        command.arg("-gencode").arg(format!("arch=compute_{arch},code=sm_{arch}"));
    }
    command.arg("-gencode").arg(format!("arch=compute_{NEWEST},code=compute_{NEWEST}"));
    command.arg("-I").arg(&kernels);
    for source in &sources {
        command.arg(source);
    }
    command.arg("-o").arg(&fatbin);

    let output = command.output().expect("nvcc should be runnable once found");
    if !output.status.success() {
        panic!(
            "nvcc failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    // Deliberately no `cargo:rustc-link-lib=cuda`. The driver is opened with
    // `dlopen` at run time instead - see the note in `lib.rs`. Linking it here
    // would make the wheel depend on a library manylinux does not allow, and
    // would stop this crate building on a machine with no driver stub.
}

fn find_nvcc() -> Option<PathBuf> {
    for variable in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(home) = std::env::var_os(variable) {
            let candidate = PathBuf::from(home).join("bin").join("nvcc");
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    for prefix in ["/usr/local/cuda", "/opt/cuda", "/usr"] {
        let candidate = PathBuf::from(prefix).join("bin").join("nvcc");
        if candidate.exists() {
            return Some(candidate);
        }
    }
    let found = Command::new("which").arg("nvcc").output().ok()?;
    found.status.success().then(|| {
        PathBuf::from(String::from_utf8_lossy(&found.stdout).trim().to_string())
    })
}
