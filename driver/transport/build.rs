//! Build script — only does work under `--features nixl`.
//!
//! Under the `nixl` feature it generates the NIXL C-API bindings from the
//! vendored `engines/nixl/wrapper.h` and links the wheel's precompiled
//! `libnixl_capi.so` from `$NIXL_PREFIX/lib`. The header is the single file
//! taken from NIXL source (Apache-2.0); everything else comes from the
//! `pip download nixl-cu12` wheel assembled into `$NIXL_PREFIX` (see the crate
//! README). Without the feature this is a no-op, so the default build needs no
//! NIXL, no `bindgen`, and no native library.

fn main() {
    #[cfg(feature = "nixl")]
    nixl::generate();
}

#[cfg(feature = "nixl")]
mod nixl {
    use std::path::PathBuf;

    pub fn generate() {
        println!("cargo:rerun-if-changed=src/engines/nixl/wrapper.h");
        println!("cargo:rerun-if-env-changed=NIXL_PREFIX");

        let prefix = std::env::var("NIXL_PREFIX").expect(
            "NIXL_PREFIX must point at an assembled NIXL install \
             (lib/ with libnixl_capi.so + deps) when building --features nixl. \
             See transport/README.md for the `pip download nixl-cu12` recipe.",
        );

        // Link the precompiled C-API shim; rpath so the .so chain resolves at
        // run time without an install step.
        println!("cargo:rustc-link-search=native={prefix}/lib");
        println!("cargo:rustc-link-lib=dylib=nixl_capi");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{prefix}/lib");

        let bindings = bindgen::Builder::default()
            .header("src/engines/nixl/wrapper.h")
            .allowlist_item("nixl_capi.*")
            .generate()
            .expect("failed to generate NIXL bindings from wrapper.h");

        let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out.join("nixl_ffi.rs"))
            .expect("failed to write nixl_ffi.rs");
    }
}
