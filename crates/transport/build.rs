fn main() {
    #[cfg(feature = "nixl")]
    nixl::generate();
}

#[cfg(feature = "nixl")]
mod nixl {
    use std::path::PathBuf;

    pub fn generate() {
        println!("cargo:rerun-if-changed=src/backends/nixl/wrapper.h");
        println!("cargo:rerun-if-env-changed=NIXL_PREFIX");

        let prefix = std::env::var("NIXL_PREFIX").expect(
            "NIXL_PREFIX must point at an assembled NIXL install \
             (lib/ with libnixl_capi.so + deps) when building --features nixl.",
        );

        println!("cargo:rustc-link-search=native={prefix}/lib");
        println!("cargo:rustc-link-lib=dylib=nixl_capi");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{prefix}/lib");

        let bindings = bindgen::Builder::default()
            .header("src/backends/nixl/wrapper.h")
            .allowlist_item("nixl_capi.*")
            .generate()
            .expect("failed to generate NIXL bindings from wrapper.h");

        let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out.join("nixl_ffi.rs"))
            .expect("failed to write nixl_ffi.rs");
    }
}
