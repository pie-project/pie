//! The device headers, carried in the binary, that an `#include` resolves
//! against.
//!
//! # The rule this file is
//!
//! > **No include path on disk. Includes resolve against a header set carried
//! > in the binary, or they do not resolve at all.**
//!
//! Tier A got its first authored kernel header compiled by NVRTC by giving it
//! nothing to include: `nvrtcCreateProgram` was called with zero headers and
//! zero include names, so `altup_aux.cuh` spelled its own bf16 conversions
//! rather than reaching for `cuda_bf16.h`. That was the right price for one
//! file. It is the wrong price for a family of them — the second family to
//! widen a bf16 restates the arithmetic, and
//! `driver-metal/src/layout/shader.rs` records where that ends: *"restating
//! it is what the 4-bit codecs did before this existed."*
//!
//! Metal solved it by SPLICING — reading the includer, finding the directive,
//! pasting the text. CUDA does not have to. `nvrtcCreateProgram` takes
//! `headers[]` and `includeNames[]`, an in-memory virtual filesystem, so the
//! compiler resolves the directive itself against text this module hands it.
//! That is strictly better than splicing and the reasons are worth stating,
//! because they are why this is not a port of the Metal module:
//!
//! * **The authored source keeps its `#include` lines**, so it stays
//!   nvcc-compilable — which is what lets `abi::emit_device_typecheck` go on
//!   turning a drifted row into a build error instead of a failed fire.
//! * **Include guards work.** A diamond — two headers including a third — is
//!   one definition, because `#pragma once` is evaluated by a preprocessor
//!   rather than approximated by a `HashSet` of paths the splicer has seen.
//! * **Nothing is read from disk.** [`include_str!`] puts every byte in the
//!   binary at build time, so a header cannot be missing, cannot be stale
//!   relative to the rows it was built with, and cannot be found on a machine
//!   that has a CUDA toolkit and not found on one that does not.
//!
//! That last one is the crown jewel this whole design is arranged around.
//! `driver-cuda` builds and runs with no CUDA toolkit; buying includes with
//! `-I /usr/local/cuda/include` would trade that for a convenience, and would
//! also make the compiled result depend on a host fact the cache key cannot
//! see.
//!
//! # What is deliberately NOT here
//!
//! `cuda_fp16.h`, `cuda_bf16.h`, `cuda_fp8.h`, `cuda_fp4.h`,
//! `cooperative_groups.h`. NVRTC does not bundle the CUDA device headers
//! before 13.3, so including one means **vendoring** it — which is a
//! redistribution decision with a `NOTICE` entry and a pinned device ABI
//! behind it, not something a refactor gets to decide on the way past. Until
//! that is settled, [`pie_device.cuh`] keeps writing its conversions out.
//!
//! The mechanism does not care which headers it carries. The day those are
//! vendored they are entries in [`DEVICE_HEADERS`] and nothing else changes —
//! which is the argument for building the set before the decision rather than
//! after it.
//!
//! [`pie_device.cuh`]: DEVICE_HEADERS
//!
//! # Names are relative to `csrc/src`
//!
//! One spelling, two resolvers. `#include "pie_device.cuh"` is what
//! NVRTC matches against [`Header::name`] and what nvcc finds under the
//! `-I csrc/src` the offline typecheck already passes. The same string is
//! `KernelSig::file`, so a row, a `#include` and an entry here cannot drift
//! into three spellings of one file.

use std::ffi::CString;

/// One header, and the name an `#include` spells to reach it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    /// What `#include "…"` must say, relative to `csrc/src`.
    pub name: &'static str,
    /// The text, carried in the binary by [`include_str!`].
    pub text: &'static str,
}

/// Every header an NVRTC compile in this crate may resolve an include
/// against.
///
/// A header that is not here does not exist as far as a compile is
/// concerned — which is the property `every_authored_include_resolves` below
/// checks, so that adding a `#include` and forgetting the entry is a test
/// failure on any machine rather than an NVRTC error on a GPU.
///
/// The paths are relative to THIS file, so a header that moves is a compile
/// error here rather than a missing file at run time.
pub const DEVICE_HEADERS: &[Header] = &[
    // The prelude, which every other entry here includes. First because it is
    // the one header with no dependencies of its own; the order is otherwise
    // immaterial, since NVRTC resolves by name rather than by position.
    Header {
        name: "pie_device.cuh",
        text: include_str!("../../../kernels-cuda-new/csrc/src/pie_device.cuh"),
    },
    // The shared device headers §7's Stage D names. Each exists because two
    // kernels would otherwise restate it: `rope_device.cuh`'s own comment
    // says *"a second copy of these three lines is a bit-exactness bug
    // waiting to happen"*, which is the whole argument for the set.
    Header {
        name: "rope_device.cuh",
        text: include_str!("../../../kernels-cuda-new/csrc/src/rope_device.cuh"),
    },
    Header {
        name: "kv_paged_addr.cuh",
        text: include_str!("../../../kernels-cuda-new/csrc/src/kv_paged_addr.cuh"),
    },
];

/// The compilable unit sources, carried the same way their headers are.
///
/// A unit's root is not in [`DEVICE_HEADERS`] — nothing `#include`s it, it is
/// what `nvrtcCreateProgram` is handed as THE source — but it comes from the
/// same place and moves for the same reason, so it is spelled here rather
/// than beside the table that names its rows.
pub mod sources {
    /// `norm`'s six AltUp auxiliary templates.
    pub const NORM_ALTUP_AUX: &str =
        include_str!("../../../kernels-cuda-new/csrc/src/norm/altup_aux.cuh");
    /// `norm`'s pointwise pair: `residual_add` and `scalar_mul`.
    pub const NORM_ELEMENTWISE: &str =
        include_str!("../../../kernels-cuda-new/csrc/src/norm/elementwise.cuh");
}

impl Header {
    /// The set the driver ships, as a slice.
    ///
    /// A function rather than naming [`DEVICE_HEADERS`] at each call site, so
    /// that a caller wanting "the real set" and a caller wanting a doctored
    /// one read the same and differ in one word.
    #[must_use]
    pub fn device_headers() -> &'static [Header] {
        DEVICE_HEADERS
    }
}

/// The header set as the two parallel arrays `nvrtcCreateProgram` wants.
///
/// Returned as owned `CString`s because NVRTC copies neither: the pointers
/// must outlive the call, and the caller holding the vectors is what makes
/// that visible rather than a comment.
///
/// # Errors
///
/// A header name or its text contains a NUL, which no source can.
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

/// FNV-1a 64 over every header's name and text, in table order.
///
/// **The cache key must span this.** `driver-cuda/src/program/cache.rs`
/// records the lesson in the past tense — a cubin keyed on less than what
/// produced it is served after the thing it was not keyed on changes — and
/// `driver-metal/src/program/cache.rs` keys on the RESOLVED text for the same
/// reason. With NVRTC resolving includes, the resolved text is the root plus
/// every header that can reach it, and this is the half that is not the root.
///
/// Content, not modification time: a rebuild that changes nothing must keep
/// the cache warm, and a header edited back to its old bytes must hit the
/// entry it had before.
#[must_use]
pub fn digest(headers: &[Header]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for header in headers {
        eat(header.name.as_bytes());
        // A separator, so `("ab", "c")` and `("a", "bc")` are different sets.
        eat(&[0]);
        eat(header.text.as_bytes());
        eat(&[0]);
    }
    hash
}

/// Every quoted `#include` in `source`, in order of appearance.
///
/// Column zero only, matching the Metal splicer's rule and for its reason: a
/// directive is a directive only at the start of a line, so the same
/// characters inside a string literal or a comment are left alone. Angle
/// includes are not reported — those name a compiler's own headers, and this
/// set does not carry any.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The check that makes the set authoritative: every `#include` in
    /// anything NVRTC will be handed names a header the set carries.
    ///
    /// Without this, adding a `#include` and forgetting the entry is an NVRTC
    /// error on a machine with a GPU — the slowest possible place to find out
    /// — and this file's whole claim is that the binary is self-sufficient.
    #[test]
    fn every_authored_include_resolves_against_the_set() {
        let mut sources: Vec<(&str, &str)> =
            vec![("norm/altup_aux.cuh", super::super::nvrtc::SOURCE)];
        sources.extend(DEVICE_HEADERS.iter().map(|h| (h.name, h.text)));

        for (name, text) in sources {
            for included in quoted_includes(text) {
                assert!(
                    DEVICE_HEADERS.iter().any(|h| h.name == included),
                    "{name} includes `{included}`, which the header set does not \
                     carry -- NVRTC resolves against the set and nothing else, so \
                     this compiles nowhere"
                );
            }
        }
    }

    /// The set is what a compile is keyed on, so two different sets must not
    /// key the same. A digest that ignored the text would serve a stale cubin
    /// after a header edit — which is the exact failure
    /// `program::cache`'s header records having made once.
    #[test]
    fn the_digest_moves_when_any_header_does() {
        let base = digest(DEVICE_HEADERS);
        assert_eq!(base, digest(DEVICE_HEADERS), "and is stable");

        let edited = [Header {
            name: DEVICE_HEADERS[0].name,
            text: "// not what it was",
        }];
        assert_ne!(base, digest(&edited), "text is in the key");

        let renamed = [Header {
            name: "norm/somewhere_else.cuh",
            text: DEVICE_HEADERS[0].text,
        }];
        assert_ne!(base, digest(&renamed), "and so is the name it resolves by");

        // The separator earns its keep: without it these two sets are one
        // byte stream and one digest.
        let split = [
            Header {
                name: "a",
                text: "bc",
            },
            Header {
                name: "d",
                text: "e",
            },
        ];
        let joined = [
            Header {
                name: "ab",
                text: "c",
            },
            Header {
                name: "d",
                text: "e",
            },
        ];
        assert_ne!(digest(&split), digest(&joined));
    }

    #[test]
    fn only_column_zero_quoted_includes_are_directives() {
        let source = "\
#include \"a.cuh\"
  #include \"indented.cuh\"
#include <cuda_bf16.h>
const char* s = \"#include \\\"in_a_string.cuh\\\"\";
#include\t\"tabbed.cuh\"
";
        assert_eq!(quoted_includes(source), vec!["a.cuh", "tabbed.cuh"]);
    }

    /// The arrays handed to NVRTC are the table, in order and complete.
    #[test]
    fn the_nvrtc_arrays_are_the_table() {
        let (texts, names) = as_nvrtc_arrays(DEVICE_HEADERS).expect("no NULs in a source");
        assert_eq!(texts.len(), DEVICE_HEADERS.len());
        assert_eq!(names.len(), DEVICE_HEADERS.len());
        for (at, header) in DEVICE_HEADERS.iter().enumerate() {
            assert_eq!(names[at].to_str().unwrap(), header.name);
            assert_eq!(texts[at].to_str().unwrap(), header.text);
        }
    }
}
