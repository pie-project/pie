//! Compiles the Neural Engine bridge, `native/ane.m`, on an Apple host. Every
//! other target gets no bridge and no `ane` module.

fn main() {
    println!("cargo:rerun-if-changed=native/ane.m");
    let apple = std::env::var("CARGO_CFG_TARGET_VENDOR").is_ok_and(|v| v == "apple");
    let mac_host = std::env::var("HOST").is_ok_and(|host| host.contains("apple"));
    if !apple || !mac_host {
        return;
    }
    cc::Build::new()
        .file("native/ane.m")
        .flag("-fobjc-arc")
        .flag("-fmodules")
        .compile("pie_ane");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=IOSurface");
}
