fn main() {
    let apple = std::env::var("CARGO_CFG_TARGET_VENDOR").is_ok_and(|v| v == "apple");
    let mac_host = std::env::var("HOST").is_ok_and(|host| host.contains("apple"));
    if !apple || !mac_host {
        return;
    }
    println!("cargo:rerun-if-changed=native/coreml.m");
    println!("cargo:rerun-if-changed=native/private.m");
    cc::Build::new()
        .file("native/coreml.m")
        .file("native/private.m")
        .flag("-fobjc-arc")
        .flag("-fmodules")
        .compile("pie_coreml");
    println!("cargo:rustc-link-lib=framework=CoreML");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=IOSurface");
    println!("cargo:rustc-link-lib=framework=Metal");
}
