use std::ffi::CString;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub name: &'static str,
    pub text: &'static str,
}

#[allow(clippy::all)]
mod generated {
    use super::Header;
    include!(concat!(env!("OUT_DIR"), "/headers.rs"));
}

pub use generated::{LIBRARY, SHIM, UPSTREAM};

const SHIMMED: [Header; SHIM.len() + LIBRARY.len()] =
    join::<{ SHIM.len() + LIBRARY.len() }>(SHIM, LIBRARY);

pub const DEVICE_HEADERS: &[Header] = &SHIMMED;

pub const ALL_HEADERS: &[Header] =
    &join::<{ SHIM.len() + LIBRARY.len() + UPSTREAM.len() }>(&SHIMMED, UPSTREAM);

const fn join<const N: usize>(left: &[Header], right: &[Header]) -> [Header; N] {
    let mut out = [Header { name: "", text: "" }; N];
    let mut w = 0;
    let mut i = 0;
    while i < left.len() {
        out[w] = left[i];
        w += 1;
        i += 1;
    }
    let mut j = 0;
    while j < right.len() {
        out[w] = right[j];
        w += 1;
        j += 1;
    }
    out
}

#[must_use]
pub const fn carried(name: &'static str) -> &'static str {
    let mut i = 0;
    while i < LIBRARY.len() {
        if str_eq(LIBRARY[i].name, name) {
            return LIBRARY[i].text;
        }
        i += 1;
    }
    panic!("no file under `kernels/` is carried under that name")
}

#[must_use]
pub fn text_of(name: &str) -> Option<&'static str> {
    LIBRARY
        .iter()
        .find(|header| header.name == name)
        .map(|header| header.text)
}

pub(crate) const fn str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

pub fn as_nvrtc_arrays(headers: &[Header]) -> Result<(Vec<CString>, Vec<CString>), String> {
    let mut texts = Vec::with_capacity(headers.len());
    let mut names = Vec::with_capacity(headers.len());
    for header in headers {
        texts.push(
            CString::new(header.text)
                .map_err(|_| format!("header `{}` contains a NUL", header.name))?,
        );
        names.push(
            CString::new(header.name)
                .map_err(|_| format!("header name `{}` contains a NUL", header.name))?,
        );
    }
    Ok((texts, names))
}

pub fn reachable(from: &str, root: &str, headers: &[Header]) -> Result<Vec<&'static str>, String> {
    let mut seen: Vec<&'static str> = Vec::new();
    let mut queue: Vec<(&str, &str)> = vec![(from, root)];
    while let Some((at, text)) = queue.pop() {
        for included in quoted_includes(text) {
            let Some(header) = headers.iter().find(|h| h.name == included) else {
                return Err(format!(
                    "`{from}` reaches `{included}` from `{at}`, and the header set it \
                     compiles against does not carry it -- NVRTC resolves against the \
                     set and nothing else, so this compiles nowhere"
                ));
            };
            if !seen.contains(&header.name) {
                seen.push(header.name);
                queue.push((header.name, header.text));
            }
        }
    }
    Ok(seen)
}

#[must_use]
pub fn digest(headers: &[Header]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for header in headers {
        hash = fold(hash, header.name.as_bytes());
        hash = fold(hash, &[0]);
        hash = fold(hash, header.text.as_bytes());
        hash = fold(hash, &[0]);
    }
    hash
}

#[must_use]
pub fn quoted_includes(source: &str) -> Vec<&str> {
    source
        .lines()
        .filter_map(|line| {
            let rest = line.strip_prefix("#include")?;
            let rest = rest.strip_prefix(|c: char| c == ' ' || c == '\t')?;
            rest.trim_start().strip_prefix('"')?.split('"').next()
        })
        .collect()
}

pub(crate) fn fnv1a64(bytes: &[u8]) -> u64 {
    fold(FNV_OFFSET_BASIS, bytes)
}

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

fn fold(mut hash: u64, bytes: &[u8]) -> u64 {
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}
