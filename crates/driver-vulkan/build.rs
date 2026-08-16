//! Hands the module directory down to the tests: `kernels-vulkan`'s `cargo:spv_dir` (via
//! `links = "pie_kernels_vulkan"`) reaches this build script as
//! `DEP_PIE_KERNELS_VULKAN_SPV_DIR` but not the crate's own code, so it is re-emitted as a
//! rustc env for `option_env!`. `option_` is load-bearing: without `kernels-vulkan/native`
//! there are no modules, so dependents must be able to say so rather than fail to compile.

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-env-changed=DEP_PIE_KERNELS_VULKAN_SPV_DIR");
    if let Ok(dir) = std::env::var("DEP_PIE_KERNELS_VULKAN_SPV_DIR") {
        println!("cargo::rustc-env=PIE_KERNELS_VULKAN_SPV_DIR={dir}");
    }
}
