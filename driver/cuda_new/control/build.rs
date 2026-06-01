//! Build `libpie_cuda_device` (the C++/CUDA device library) via CMake and
//! link it into this crate, plus the CUDA toolkit libs the device lib
//! references. Mirrors the cuda branch of `server/build.rs`; at cutover
//! that logic moves here and `server/build.rs::build_cuda()` just links
//! this crate's archive.

use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=../device/CMakeLists.txt");
    println!("cargo:rerun-if-changed=../device/src");
    println!("cargo:rerun-if-changed=../device/include/pie_cuda_device.h");
    println!("cargo:rerun-if-env-changed=CMAKE_CUDA_ARCHITECTURES");
    println!("cargo:rerun-if-env-changed=CUDACXX");

    let mut cfg = cmake::Config::new("../device");
    cfg.build_target("pie_cuda_device")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON");

    // nvcc discovery — same probing as server/build.rs.
    if std::env::var_os("CUDACXX").is_none()
        && std::env::var_os("CMAKE_CUDA_COMPILER").is_none()
    {
        for c in ["/usr/local/cuda/bin/nvcc", "/opt/cuda/bin/nvcc"] {
            if Path::new(c).exists() {
                cfg.define("CMAKE_CUDA_COMPILER", c);
                break;
            }
        }
    }
    if let Ok(arch) = std::env::var("CMAKE_CUDA_ARCHITECTURES") {
        cfg.define("CMAKE_CUDA_ARCHITECTURES", arch);
    }

    // FlashInfer (header-only) for the perf attention path. Persist the CPM
    // clone so build-hash churn doesn't re-fetch; allow an explicit pre-fetched
    // source dir to skip the network entirely.
    println!("cargo:rerun-if-env-changed=CPM_SOURCE_CACHE");
    println!("cargo:rerun-if-env-changed=PIE_FLASHINFER_SOURCE_DIR");
    let cpm_cache = std::env::var("CPM_SOURCE_CACHE")
        .unwrap_or_else(|_| "/root/.cache/pie_cpm".to_string());
    cfg.define("CPM_SOURCE_CACHE", &cpm_cache);
    if let Ok(fi) = std::env::var("PIE_FLASHINFER_SOURCE_DIR") {
        cfg.define("PIE_FLASHINFER_SOURCE_DIR", fi);
    }

    let dst = cfg.build();
    println!("cargo:rustc-link-search=native={}", dst.join("build").display());
    println!("cargo:rustc-link-lib=static=pie_cuda_device");

    // CUDA toolkit libs the device lib references.
    let cuda_home = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let lib64 = PathBuf::from(&cuda_home).join("lib64");
    if lib64.is_dir() {
        println!("cargo:rustc-link-search=native={}", lib64.display());
    }
    for lib in ["cudart", "cublas"] {
        println!("cargo:rustc-link-lib={lib}");
    }
    println!("cargo:rustc-link-lib=stdc++");
}
