//! The fingerprint of the code that shapes what a deployment's bodies hold
//! (`serve::measured`): this crate's sources and the kernel crate it builds
//! over (its Rust and its kernels' sources). A record of what the bodies
//! took is kept across pie versions that change none of them, and measured
//! again by one that changes any; nothing of the build's version, commit or
//! file times keys it. Test-only code shapes no body: the items under
//! `#[cfg(test)]` / `#[cfg(all(test, ...))]` are left out of the hash
//! ([`strip_tests`]), as are `tests/` and `benches/` (outside the roots).

use std::path::{Path, PathBuf};

fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            files(&path, out);
        } else {
            out.push(path);
        }
    }
}

/// The end (exclusive) of the item starting at `at`: through its block's
/// closing brace, or its `;`, whichever ends it at depth zero; strings,
/// chars and comments skipped.
fn item_end(src: &[u8], mut at: usize) -> usize {
    let mut paren = 0i32;
    let mut braces = 0i32;
    let mut opened = false;
    while at < src.len() {
        let c = src[at];
        match c {
            b'/' if src.get(at + 1) == Some(&b'/') => {
                while at < src.len() && src[at] != b'\n' {
                    at += 1;
                }
                continue;
            }
            b'/' if src.get(at + 1) == Some(&b'*') => {
                at += 2;
                while at + 1 < src.len() && !(src[at] == b'*' && src[at + 1] == b'/') {
                    at += 1;
                }
                at += 2;
                continue;
            }
            b'r' if matches!(src.get(at + 1), Some(b'"' | b'#'))
                && (at == 0 || !(src[at - 1].is_ascii_alphanumeric() || src[at - 1] == b'_')) =>
            {
                let mut hashes = 0;
                let mut k = at + 1;
                while src.get(k) == Some(&b'#') {
                    hashes += 1;
                    k += 1;
                }
                if src.get(k) == Some(&b'"') {
                    k += 1;
                    'raw: while k < src.len() {
                        if src[k] == b'"' && (0..hashes).all(|h| src.get(k + 1 + h) == Some(&b'#'))
                        {
                            k += 1 + hashes;
                            break 'raw;
                        }
                        k += 1;
                    }
                    at = k;
                    continue;
                }
            }
            b'"' => {
                at += 1;
                while at < src.len() && src[at] != b'"' {
                    if src[at] == b'\\' {
                        at += 1;
                    }
                    at += 1;
                }
                at += 1;
                continue;
            }
            b'\'' => {
                // A char literal ('x', '\n', '\u{..}'), not a lifetime.
                let close = if src.get(at + 1) == Some(&b'\\') {
                    src[at + 2..]
                        .iter()
                        .position(|&b| b == b'\'')
                        .map(|p| at + 2 + p)
                } else if src.get(at + 2) == Some(&b'\'') {
                    Some(at + 2)
                } else {
                    None
                };
                if let Some(close) = close {
                    at = close + 1;
                    continue;
                }
            }
            b'(' | b'[' => paren += 1,
            b')' | b']' => paren -= 1,
            b'{' => {
                braces += 1;
                opened = true;
            }
            b'}' => {
                braces -= 1;
                if opened && braces == 0 && paren == 0 {
                    return at + 1;
                }
            }
            b';' if !opened && paren == 0 && braces == 0 => return at + 1,
            _ => {}
        }
        at += 1;
    }
    src.len()
}

/// `src` without its test-only items: each item an attribute
/// `#[cfg(test)]` or `#[cfg(all(test, ...))]` gates, with its other
/// attributes and doc comments.
fn strip_tests(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut at = 0;
    while at < bytes.len() {
        let rest = &src[at..];
        let line_start = at == 0 || bytes[at - 1] == b'\n';
        let trimmed = rest.trim_start_matches([' ', '\t']);
        if line_start
            && (trimmed.starts_with("#[cfg(test)]") || trimmed.starts_with("#[cfg(all(test,"))
        {
            // Past the attribute's own closing bracket, then the item.
            let open = at + (rest.len() - trimmed.len()) + 1;
            let mut depth = 0i32;
            let mut k = open;
            while k < bytes.len() {
                match bytes[k] {
                    b'[' => depth += 1,
                    b']' => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    _ => {}
                }
                k += 1;
            }
            at = item_end(bytes, k + 1);
            continue;
        }
        let next = rest.find('\n').map_or(src.len(), |n| at + n + 1);
        out.push_str(&src[at..next]);
        at = next;
    }
    out
}

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets it"));
    let kernels = manifest
        .join("..")
        .join(if std::env::var_os("CARGO_FEATURE_ROCM").is_some() {
            "kernels-rocm"
        } else {
            "kernels-cuda"
        });
    let roots = [
        manifest.join("src"),
        kernels.join("src"),
        kernels.join("kernels"),
        kernels.join("shim"),
    ];
    let mut paths = Vec::new();
    for root in &roots {
        println!("cargo:rerun-if-changed={}", root.display());
        files(root, &mut paths);
    }
    let base = manifest.join("..");
    let mut named: Vec<(String, PathBuf)> = paths
        .into_iter()
        .map(|path| {
            let name = path
                .strip_prefix(&base)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            (name, path)
        })
        .collect();
    named.sort();
    // FNV-1a over each file's name and bytes, in name order.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for (name, path) in &named {
        let bytes = std::fs::read(path).unwrap_or_default();
        let kept = if name.ends_with(".rs") {
            String::from_utf8(bytes)
                .map(|text| strip_tests(&text).into_bytes())
                .unwrap_or_default()
        } else {
            bytes
        };
        eat(name.as_bytes());
        eat(&[0]);
        eat(&kept);
        eat(&[0]);
    }
    std::fs::write(
        PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets it")).join("bodies_build"),
        format!("{hash:016x}"),
    )
    .expect("OUT_DIR is writable");
    println!("cargo:rustc-env=PIE_BODIES_BUILD={hash:016x}");

    // The committed bodies records (`profiles/bodies/<key>`, `pie tune
    // export --engine cuda`), carried by key: a load whose cache holds no
    // record of its key takes the committed one.
    let committed = manifest.join("profiles").join("bodies");
    println!("cargo:rerun-if-changed={}", committed.display());
    let mut records: Vec<(String, PathBuf)> = std::fs::read_dir(&committed)
        .map(|entries| {
            entries
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| path.is_file())
                .filter_map(|path| {
                    let name = path.file_name()?.to_str()?.to_string();
                    Some((name, path))
                })
                .collect()
        })
        .unwrap_or_default();
    records.sort();
    let mut table = String::from(
        "/// The committed bodies records, by key.\npub(crate) const COMMITTED: &[(&str, &str)] = &[\n",
    );
    for (name, path) in &records {
        table.push_str(&format!(
            "    ({name:?}, include_str!({:?})),\n",
            path.display().to_string()
        ));
    }
    table.push_str("];\n");
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets it"));
    std::fs::write(out.join("committed_bodies.rs"), table).expect("OUT_DIR is writable");
}
