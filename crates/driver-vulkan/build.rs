//! Hand the module directory down to the tests.
//!
//! `kernels-vulkan` declares `links = "pie_kernels_vulkan"` and prints
//! `cargo:spv_dir`, which cargo delivers to a DEPENDENT's build script as
//! `DEP_PIE_KERNELS_VULKAN_SPV_DIR`. It does not deliver it to the dependent's
//! own code, so this re-emits it as a rustc env for `option_env!` to pick up.
//!
//! `option_` is load-bearing: without `kernels-vulkan/native` there are no
//! modules at all, and a test that cross-checks against them has to be able to
//! SAY it has none rather than fail to compile.

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-env-changed=DEP_PIE_KERNELS_VULKAN_SPV_DIR");
    if let Ok(dir) = std::env::var("DEP_PIE_KERNELS_VULKAN_SPV_DIR") {
        println!("cargo::rustc-env=PIE_KERNELS_VULKAN_SPV_DIR={dir}");
    }
}
