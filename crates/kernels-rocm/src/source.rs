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

pub use generated::LIBRARY;

#[must_use]
pub fn text_of(name: &str) -> Option<&'static str> {
    LIBRARY
        .iter()
        .find(|header| header.name == name)
        .map(|header| header.text)
}

pub fn as_rtc_arrays(headers: &[Header]) -> Result<(Vec<CString>, Vec<CString>), String> {
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
