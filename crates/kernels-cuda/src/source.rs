use std::ffi::CString;

/// One header, and the name an `#include` spells to reach it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    /// What `#include "…"` must say, relative to `kernels/`.
    pub name: &'static str,
    /// The text, carried in the binary by [`include_str!`].
    pub text: &'static str,
}

// ── The three lists, WALKED ────────────────────────────────────────────────
//
// `build.rs` writes them from `kernels/` and `shim/`, and this is the whole of
// what it needed to know:
//
// * **which list** a file lands in is its DIRECTORY -- `shim/` to [`SHIM`],
//   `kernels/flashinfer` and `kernels/xqa` to [`UPSTREAM`], the rest of
//   `kernels/` to [`LIBRARY`];
// * **what it is called** is its path minus that root;
// * **and what else it is called** is scanned out of the `#include` directives
//   that reach it, because NVRTC matches `includeNames[]` against the literal
//   string in the directive and resolves nothing. A file two directives spell
//   two ways needs an entry per spelling.
//
// # What the 187 hand-written lines cost, and what they bought
//
// They were maintained under a rule a person had to remember -- *"adding a
// file under `kernels/` or `shim/` means adding a line to one of these"* --
// and 174 of the 187 were the file's own path written twice. The other 13 were
// the `../` spellings the upstream trees use to reach their siblings, which
// `build.rs` finds by reading the directives that write them.
//
// The rule was checked from one side only. [`quoted_includes`] and
// `every_device_include_resolves` walked the set's own includes and failed on
// the first one nothing carried -- so a MISSING entry was caught, and the doc
// was upfront that *"a file nothing includes yet"* was not. That gap is closed
// by construction now: a file is carried because it EXISTS.
//
// # This is where the set was before `384aaeed0`
//
// A build script generated it, and the script's own header argued the case
// against checking the output in: *"a stale file still COMPILES. Generated
// into `OUT_DIR`, it cannot be stale by construction."* The lists were checked
// in anyway when `kernels-cuda-new` was folded in, for one less compile stage.
// See `Cargo.toml` for the whole of that argument, which reads the same way
// now as it did then.
//
// `#[allow(clippy::all)]` on the include: the generated rows are one long line
// each and no lint has anything to tell a table.
#[allow(clippy::all)]
mod generated {
    use super::Header;
    include!(concat!(env!("OUT_DIR"), "/headers.rs"));
}

pub use generated::{LIBRARY, SHIM, UPSTREAM};

/// What every compile in this crate resolves an `#include` against, unless the
/// unit asks for [`UPSTREAM`] as well.
const SHIMMED: [Header; SHIM.len() + LIBRARY.len()] =
    join::<{ SHIM.len() + LIBRARY.len() }>(SHIM, LIBRARY);

/// See [`SHIMMED`].
pub const DEVICE_HEADERS: &[Header] = &SHIMMED;

/// [`DEVICE_HEADERS`] plus [`UPSTREAM`] — what a unit compiling upstream
/// attention resolves against.
///
/// Not the default, and the reason is bytes: `nvrtcCreateProgram` copies every
/// header it is given, so a `norm` compile that carried FlashInfer would pay
/// for it on every launch that missed the cache.
pub const ALL_HEADERS: &[Header] =
    &join::<{ SHIM.len() + LIBRARY.len() + UPSTREAM.len() }>(&SHIMMED, UPSTREAM);

/// `[T] ++ [U] -> [T; N]` at compile time.
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

/// The text [`LIBRARY`] carries under `name`, by the spelling an `#include`
/// reaches it with.
///
/// # Why a root does not `include_str!` its own file
///
/// [`LIBRARY`] is the whole of `kernels/` as the binary carries it, and
/// [`DEVICE_HEADERS`] hands all of it to every compile. A root that also wrote
/// `include_str!("../kernels/layout/slot_ops.cuh")` was stating a second time
/// what this list already holds, in a second spelling — a `../` path whose
/// depth is a fact about where the *Rust* file sits, so moving `layout.rs` to
/// `layout/mod.rs` broke every root declared in it. Naming the carried file is
/// the whole of what a root has to say.
///
/// # A name nothing answers to does not compile
///
/// That is the point of the `const fn` and the [`panic!`]: `include_str!`
/// failed at compile time on a path that did not exist, and giving that up for
/// a runtime lookup would move a typo from `cargo check` to the first fire of
/// that kernel on a GPU. Const evaluation refuses instead, and the diagnostic
/// points at the declaration, which is where the misspelled name is written.
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

/// [`carried`], as a value rather than a refusal to compile.
///
/// The launch path names its file at run time, so a miss there has to be
/// answerable — see [`crate::jit::Root::of`].
#[must_use]
pub fn text_of(name: &str) -> Option<&'static str> {
    LIBRARY.iter().find(|header| header.name == name).map(|header| header.text)
}

/// `a == b`, in a `const` context.
///
/// `str::eq` is not `const`, and the comparison has to happen during const
/// evaluation or [`carried`] cannot refuse at compile time.
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

/// The header set as the two parallel arrays `nvrtcCreateProgram` wants,
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

/// Every carried header reachable from `root`, or the first `#include` the set
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

/// FNV-1a 64 over every header's name and text, in table order.
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

/// Every quoted `#include` in `source`, in order of appearance.
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

/// FNV-1a 64 over bytes, the fold [`digest`] is built out of.
pub(crate) fn fnv1a64(bytes: &[u8]) -> u64 {
    fold(FNV_OFFSET_BASIS, bytes)
}

/// The algorithm's offset basis, and the same one
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// One more chunk into a running FNV-1a, so a digest over several fields is
fn fold(mut hash: u64, bytes: &[u8]) -> u64 {
    /// The algorithm's prime.
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::Headers;

    /// The set a unit chooses decides what resolves — checked with a header
    #[test]
    fn a_units_header_choice_decides_what_resolves() {
        let upstream = UPSTREAM
            .iter()
            .find(|v| !Headers::Library.set().iter().any(|l| l.name == v.name))
            .expect("the upstream closure carries a header the library does not");
        let root = format!("#include \"{}\"\n", upstream.name);

        let refused = reachable("a/unit", &root, Headers::Library.set())
            .expect_err("the library set does not carry an upstream header");
        assert!(refused.contains(upstream.name), "a refusal names the include: {refused}");

        let resolved = reachable("a/unit", &root, Headers::LibraryAndUpstream.set())
            .expect("the upstream set carries it");
        assert!(resolved.contains(&upstream.name), "and reports what it reached: {resolved:?}");
    }

    /// Reachability is transitive and reports the header it came THROUGH,
    #[test]
    fn an_include_two_headers_deep_is_reached_and_blamed_precisely() {
        let carried = [
            Header { name: "one.cuh", text: "#include \"two.cuh\"\n" },
            Header { name: "two.cuh", text: "#include \"three.cuh\"\n" },
        ];
        let root = "#include \"one.cuh\"\n";

        let why = reachable("a/unit", root, &carried).expect_err("`three.cuh` is not carried");
        assert!(why.contains("`three.cuh`"), "the include that is missing: {why}");
        assert!(why.contains("from `two.cuh`"), "and the header that reaches it: {why}");
        assert!(why.contains("`a/unit`"), "and the unit that would not compile: {why}");

        let full = [carried[0], carried[1], Header { name: "three.cuh", text: "" }];
        assert_eq!(
            reachable("a/unit", root, &full).expect("all three are carried"),
            ["one.cuh", "two.cuh", "three.cuh"]
        );
    }

    /// Every header the crate carries is self-consistent.
    #[test]
    fn every_device_include_resolves() {
        for header in ALL_HEADERS {
            for included in quoted_includes(header.text) {
                assert!(
                    ALL_HEADERS.iter().any(|h| h.name == included),
                    "`{}` includes `{included}`, which the set does not carry",
                    header.name
                );
            }
        }
    }

    /// The set is what a compile is keyed on, so two different sets must not
    #[test]
    fn the_digest_moves_when_any_header_does() {
        let base = digest(DEVICE_HEADERS);
        assert_eq!(base, digest(DEVICE_HEADERS), "and is stable");

        let edited = [Header { name: DEVICE_HEADERS[0].name, text: "// not what it was" }];
        assert_ne!(base, digest(&edited), "text is in the key");

        let renamed = [Header { name: "norm/somewhere_else.cuh", text: DEVICE_HEADERS[0].text }];
        assert_ne!(base, digest(&renamed), "and so is the name it resolves by");

        let split = [Header { name: "a", text: "bc" }, Header { name: "d", text: "e" }];
        let joined = [Header { name: "ab", text: "c" }, Header { name: "d", text: "e" }];
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
