use std::path::PathBuf;

fn main() {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let include = crate_dir.join("include");
    println!(
        "cargo:rerun-if-changed={}",
        include.join("pie_driver_abi.h").display()
    );
    println!("cargo:include={}", include.display());
}
