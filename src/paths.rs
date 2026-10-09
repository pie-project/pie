use std::path::PathBuf;

pub fn pie_home() -> PathBuf {
    if let Ok(dir) = std::env::var("PIE_HOME")
        && !dir.trim().is_empty()
    {
        return PathBuf::from(dir);
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".pie")
}

pub fn pie_home_file(name: &str) -> PathBuf {
    pie_home().join(name)
}

/// Where installed inferlets live: `<name>/<version>.{wasm,py,js}` beside
/// `<version>.toml`.
pub fn inferlets_dir() -> PathBuf {
    pie_home().join("inferlets")
}

/// Where the language components live: `<language>.wasm`, one per
/// language script inferlets can be written in.
pub fn languages_dir() -> PathBuf {
    pie_home().join("languages")
}

/// Where compiled wasm is cached across boots, keyed by wasmtime on the
/// bytes and its own version: the language components and installed
/// inferlets compile once per machine, not once per boot.
pub fn compile_cache_dir() -> PathBuf {
    pie_home().join("cache").join("wasmtime")
}
