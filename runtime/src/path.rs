use std::path::PathBuf;

/// Resolves the Pie home directory ($PIE_HOME or ~/.pie).
pub fn get_pie_home() -> PathBuf {
    if let Ok(pie_home) = std::env::var("PIE_HOME") {
        if !pie_home.trim().is_empty() {
            return PathBuf::from(pie_home);
        }
    }
    dirs::home_dir()
        .expect("could not determine home directory")
        .join(".pie")
}

/// Resolves the py-runtime directory ($PIE_HOME/py-runtime or ~/.pie/py-runtime).
pub fn get_py_runtime_dir() -> PathBuf {
    get_pie_home().join("py-runtime")
}
