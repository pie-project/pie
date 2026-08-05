use std::env;
use std::path::PathBuf;

fn main() {
    // Hand the committed, cbindgen-generated header directory to downstream
    // build scripts. `driver/{cuda,metal}` compile against it, and
    // `worker/build.rs` reads `DEP_PIE_LOADER_INCLUDE` to find it.
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let include = manifest_dir.join("include");
    println!(
        "cargo:rerun-if-changed={}",
        include.join("pie_loader.h").display()
    );
    println!("cargo:include={}", include.display());
    println!("cargo:rerun-if-changed=src");
}
