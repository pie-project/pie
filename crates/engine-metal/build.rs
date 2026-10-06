fn main() {
    let apple = std::env::var("CARGO_CFG_TARGET_VENDOR").is_ok_and(|v| v == "apple");
    if !apple {
        return;
    }
    println!("cargo:rerun-if-changed=native/coreml.m");
    cc::Build::new()
        .file("native/coreml.m")
        .flag("-fobjc-arc")
        .flag("-fmodules")
        .compile("pie_coreml");
    println!("cargo:rustc-link-lib=framework=CoreML");
    println!("cargo:rustc-link-lib=framework=Foundation");
}
