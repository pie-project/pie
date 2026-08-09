//! The live pair: Rust's enumerators against the device's, compared directly.
//!
//! # Why this file exists, and what it replaces
//!
//! `KvCacheScheme` and `DType` are each spelled three times:
//!
//! | # | copy | where |
//! |---|---|---|
//! | 1 | Rust | [`driver_cuda::bind::abi::KvCacheScheme`], `driver_cuda::dtype::DType` |
//! | 2 | host C++ | `kernels-cuda/csrc/src/attn/kv_cache_view.hpp`, `tensor.hpp` |
//! | 3 | device | `kernels-cuda-new/csrc/src/attn/attention_naive_paged.cuh` |
//!
//! **The driver fills a view from (1) and NVRTC's kernel switches on (3), so
//! (1) → (3) is the live pair.** For most of this migration the only
//! mechanical comparison in the tree was `attention_naive_paged.cu`'s
//! `static_assert`s, and those compare (2) ↔ (3) — a pair that stopped
//! mattering when the archive's host launchers went, because no `.cu` left in
//! the tree reads a `KvCacheScheme` out of a view and acts on it.
//!
//! That file said so itself, and said the right thing about it: *"this file
//! guards a leg that has quietly stopped mattering while the leg that matters
//! is unguarded. That is an argument for REPLACING the check, and it is
//! emphatically not an argument for deleting it now."* This is the
//! replacement. With it landed, (2) is free to go.
//!
//! # Why it is a text scan and not a `static_assert`
//!
//! The obvious home is `emit_device_typecheck`, and it is the wrong one: that
//! machinery exists so a **row's declaration** can be checked against a
//! `__global__`, and it runs inside `kernels-cuda-new`. Asking it to reach for
//! `driver-cuda`'s Rust would point the dependency backwards — `driver-cuda`
//! depends on `kernels-cuda-new`, not the other way round. So the comparison
//! belongs on this side, and on this side the device text is a `.cuh` we read
//! rather than a type we can name. `kernels-cuda/tests/sources.rs` established
//! the idiom.
//!
//! # What this cannot catch, stated so nobody assumes otherwise
//!
//! A text scan sees what is written, not what is compiled. If the `.cuh`
//! guarded an enumerator behind `#if`, this would compare a branch that never
//! reaches a kernel. Neither enum is so guarded today, and the check below
//! that the file holds exactly one definition of each is what would notice a
//! second spelling appearing.

use std::path::{Path, PathBuf};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .parent()
        .expect("repo root")
        .to_path_buf()
}

fn read(rel: &str) -> String {
    let p = repo().join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// `Name = 7` pairs from a braced block, in source order.
///
/// Takes the text between the first `{` after `head` and its matching `}`,
/// which is enough because neither an `enum class` body nor a `#[repr(u8)]
/// enum` body nests braces.
fn enumerators(src: &str, head: &str) -> Vec<(String, u8)> {
    let at = src
        .find(head)
        .unwrap_or_else(|| panic!("`{head}` is not in this file any more"));
    let rest = &src[at..];
    let open = rest.find('{').expect("an enum body");
    let close = rest[open..].find('}').expect("an enum body's end") + open;
    let body = &rest[open + 1..close];

    let mut out = Vec::new();
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with("///") {
            continue;
        }
        let Some((name, value)) = line.split_once('=') else {
            continue;
        };
        let name = name.trim();
        if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            continue;
        }
        let value = value.trim().trim_end_matches(',').trim();
        let Ok(value) = value.parse::<u8>() else {
            continue;
        };
        out.push((name.to_string(), value));
    }
    assert!(!out.is_empty(), "`{head}` parsed to no enumerators");
    out
}

/// `Fp8E4M3`, `FP8_E4M3` and `Fp8PerTensor` all fold to the same word.
///
/// The two sides use different conventions on purpose — Rust's is
/// `UpperCamel`, the device text's is upstream's `SHOUTY_SNAKE` for dtypes —
/// so a comparison on the literal spelling would fail on every line and prove
/// nothing. Folding case and underscores away leaves exactly what a mirror is
/// supposed to preserve: which name sits at which number.
fn fold(name: &str) -> String {
    name.chars()
        .filter(|c| *c != '_')
        .flat_map(char::to_lowercase)
        .collect()
}

#[track_caller]
fn mirrors(what: &str, rust: &[(String, u8)], device: &[(String, u8)]) {
    assert_eq!(
        rust.len(),
        device.len(),
        "{what}: the Rust spelling has {} enumerators and the device's has {}. \
         A partial mirror is a renumbering waiting to happen.",
        rust.len(),
        device.len()
    );
    for (i, ((rn, rv), (dn, dv))) in rust.iter().zip(device).enumerate() {
        assert_eq!(
            rv, dv,
            "{what}: position {i} is `{rn} = {rv}` in Rust and `{dn} = {dv}` on \
             the device. The driver fills a view from the Rust and the kernel \
             switches on the device's, so this is a wrong answer, not a build \
             error."
        );
        assert_eq!(
            fold(rn),
            fold(dn),
            "{what}: {rv} is `{rn}` in Rust and `{dn}` on the device. The \
             numbers agree, so nothing would fault -- the kernel would take \
             the arm for a different scheme."
        );
    }
}

const DEVICE: &str = "crates/kernels-cuda-new/csrc/src/attn/attention_naive_paged.cuh";

/// The KV cache scheme, Rust against device.
///
/// `bind::abi::KvCacheScheme`'s own doc says the numbers "are the C++ enum's
/// and are load-bearing", which is a claim about copy (2). This asserts the
/// thing that claim was standing in for.
#[test]
fn the_kv_cache_scheme_mirrors_the_device() {
    let rust = enumerators(&read("crates/driver-cuda/src/bind/abi.rs"), "pub enum KvCacheScheme");
    let device = enumerators(&read(DEVICE), "enum class KvScheme");
    mirrors("KvCacheScheme", &rust, &device);
}

/// The tensor dtype, Rust against device.
///
/// The device mirror's own header states the rule this checks: *"Only `BF16`,
/// `FP8_E4M3` and `FP8_E5M2` are read here, but every enumerator is mirrored
/// and asserted: a partial mirror is a renumbering waiting to happen."*
#[test]
fn the_dtype_mirrors_the_device() {
    let rust = enumerators(&read("crates/driver-cuda/src/dtype.rs"), "pub enum DType");
    let device = enumerators(&read(DEVICE), "enum class KvDType");
    mirrors("DType", &rust, &device);
}

/// Neither device enum is spelled twice in its own file.
///
/// `enumerators` reads the FIRST match, so a second definition — a `#if`
/// branch, a copy left behind by an edit — would be compared against nothing
/// and would reach a kernel unchecked. This is the assumption that makes the
/// two tests above a check rather than a sample.
#[test]
fn the_device_spells_each_enum_once() {
    let device = read(DEVICE);
    for head in ["enum class KvScheme", "enum class KvDType"] {
        assert_eq!(
            device.matches(head).count(),
            1,
            "`{head}` appears more than once in {DEVICE}; the mirror tests read \
             the first and would not see the others."
        );
    }
}
