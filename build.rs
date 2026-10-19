//! The commit `pie` was built from, for the header of the traces it writes.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

fn main() {
    println!("cargo:rerun-if-env-changed=PIE_COMMIT");
    if let Some(dir) = git(&["rev-parse", "--absolute-git-dir"]) {
        println!("cargo:rerun-if-changed={dir}/HEAD");
        println!("cargo:rerun-if-changed={dir}/logs/HEAD");
    }
    if std::env::var_os("PIE_COMMIT").is_none()
        && let Some(commit) = git(&["rev-parse", "--short=9", "HEAD"])
    {
        println!("cargo:rustc-env=PIE_COMMIT={commit}");
    }
}
