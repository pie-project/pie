use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let config = manifest_dir.join("cbindgen.toml");
    let out = manifest_dir.join("include").join("weight_loader.h");

    cbindgen::Builder::new()
        .with_crate(&manifest_dir)
        .with_config(cbindgen::Config::from_file(config).unwrap())
        .generate()
        .expect("failed to generate weight_loader.h")
        .write_to_file(out);

    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}
