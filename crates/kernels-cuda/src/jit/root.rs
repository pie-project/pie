use core::fmt;

use crate::source::{self, ALL_HEADERS, DEVICE_HEADERS, Header};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Toolchain {
    pub major: u32,
    pub minor: u32,
}

impl Toolchain {
    pub const ANY: Self = Self { major: 0, minor: 0 };

    #[must_use]
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    #[must_use]
    pub const fn is_any(self) -> bool {
        self.major == 0 && self.minor == 0
    }

    #[must_use]
    pub const fn met_by(self, have: Self) -> bool {
        have.major > self.major || (have.major == self.major && have.minor >= self.minor)
    }
}

impl fmt::Display for Toolchain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_any() {
            f.write_str("any")
        } else {
            write!(f, "{}.{}", self.major, self.minor)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Headers {
    Library,
    LibraryAndUpstream,
}

impl Headers {
    #[must_use]
    pub const fn set(self) -> &'static [Header] {
        match self {
            Headers::Library => DEVICE_HEADERS,
            Headers::LibraryAndUpstream => ALL_HEADERS,
        }
    }

    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Headers::Library => "lib",
            Headers::LibraryAndUpstream => "lib+upstream",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Root {
    pub name: &'static str,
    pub text: &'static str,
    pub file: &'static str,
    pub options: &'static [&'static str],
    pub headers: Headers,
    pub floor: Toolchain,
}

const CONFIGURED: &[(&str, &[&str], Headers, Toolchain)] = &[
    (
        "attn/attention_mla_fa2.cuh",
        &[
            "--device-as-default-execution-space",
            "--relocatable-device-code=true",
        ],
        Headers::LibraryAndUpstream,
        Toolchain::ANY,
    ),
    (
        "cascade/merge_states.cuh",
        &["--device-as-default-execution-space"],
        Headers::LibraryAndUpstream,
        Toolchain::ANY,
    ),
    (
        "comm/all_reduce.cuh",
        &[],
        Headers::LibraryAndUpstream,
        Toolchain::ANY,
    ),
    (
        "attn/fa2.cuh",
        &["--device-as-default-execution-space"],
        Headers::LibraryAndUpstream,
        Toolchain::ANY,
    ),
    (
        "attn/fa4.cuh",
        &["--device-as-default-execution-space"],
        Headers::LibraryAndUpstream,
        Toolchain::ANY,
    ),
    (
        "attn/attention_xqa_mha.cuh",
        &[],
        Headers::LibraryAndUpstream,
        Toolchain::ANY,
    ),
    (
        "tile/alternatives.cuh",
        &[],
        Headers::Library,
        Toolchain::new(13, 3),
    ),
    (
        "sample/argmax_tile.cuh",
        &[],
        Headers::Library,
        Toolchain::new(13, 3),
    ),
    (
        "layout/gather_rows_tile.cuh",
        &[],
        Headers::Library,
        Toolchain::new(13, 3),
    ),
    (
        "quant/dequant_wna16_tile.cuh",
        &[],
        Headers::Library,
        Toolchain::new(13, 3),
    ),
    (
        "quant/wna16_gemv_tile.cuh",
        &[],
        Headers::Library,
        Toolchain::new(13, 3),
    ),
];

const fn configured_for(file: &str) -> (&'static [&'static str], Headers, Toolchain) {
    let mut i = 0;
    while i < CONFIGURED.len() {
        let (name, options, headers, floor) = CONFIGURED[i];
        if source::str_eq(name, file) {
            return (options, headers, floor);
        }
        i += 1;
    }
    (&[], Headers::Library, Toolchain::ANY)
}

impl Root {
    #[must_use]
    pub const fn new(file: &'static str) -> Self {
        let (options, headers, floor) = configured_for(file);
        Self {
            name: strip_cuh(file),
            text: source::carried(file),
            file,
            options,
            headers,
            floor,
        }
    }

    #[must_use]
    pub fn of(file: &'static str) -> Option<Self> {
        let text = source::text_of(file)?;
        let name = file.strip_suffix(".cuh")?;
        let (options, headers, floor) = configured_for(file);
        Some(Self {
            name,
            text,
            file,
            options,
            headers,
            floor,
        })
    }

    #[must_use]
    pub const fn variant(name: &'static str, file: &'static str) -> Self {
        let (options, headers, floor) = configured_for(file);
        Self {
            name,
            text: source::carried(file),
            file,
            options,
            headers,
            floor,
        }
    }

    #[must_use]
    pub const fn options(mut self, options: &'static [&'static str]) -> Self {
        self.options = options;
        self
    }

    #[must_use]
    pub const fn upstream(mut self) -> Self {
        self.headers = Headers::LibraryAndUpstream;
        self
    }

    #[must_use]
    pub const fn since(mut self, major: u32, minor: u32) -> Self {
        self.floor = Toolchain::new(major, minor);
        self
    }

    #[must_use]
    pub fn header_set(&self) -> &'static [Header] {
        self.headers.set()
    }

    #[must_use]
    pub fn needs_device_runtime(&self) -> bool {
        self.options
            .iter()
            .any(|o| *o == "--relocatable-device-code=true" || *o == "-dc" || *o == "--device-c")
    }

    /// The disk-cache key for one instantiation of this root.
    ///
    /// # Why the expensive half is memoized
    ///
    /// The key states everything that could change what NVRTC produces: the
    /// root's own source, every header it may include, the arch, the options,
    /// the float contract and the toolchain floor. Two of those — the source
    /// digest and the header-set digest — are hashes over the whole text, and
    /// the header set for an attention root is the entire device library.
    ///
    /// This is called ONCE PER LAUNCH, from [`crate::jit::cache::resolve`], and
    /// the fire that produced this comment is 535 launches. Measured on a
    /// Qwen3-0.6B decode: **2.3 ms of host time per ordinary launch and 14 ms
    /// per attention launch**, all of it re-hashing text that had not changed,
    /// against a `cudaLaunchKernel` of about five microseconds. The GPU sat at
    /// 0 % while the host hashed.
    ///
    /// Both digests are over `&'static` data, so within one process they are
    /// constants. The prefix is computed once per `(root, arch)` and leaked, so
    /// the per-launch cost falls to hashing the instantiation string — which is
    /// short, and is the only part that actually varies per launch.
    ///
    /// Editing a root's source still changes its key, because editing it
    /// changes the text the next process compiles in; nothing here caches
    /// across processes, and the disk cache is keyed by the result.
    #[must_use]
    pub fn key(&self, instantiation: &str, arch: &str) -> String {
        format!(
            "{}/i{:016x}",
            self.key_prefix(arch),
            source::fnv1a64(instantiation.as_bytes()),
        )
    }

    /// Everything in [`Self::key`] except the instantiation, computed once per
    /// `(root, arch)`.
    ///
    /// The memo is keyed by the root's identity as the key itself sees it —
    /// name, options, floor, header tag, arch — plus the ADDRESS of the static
    /// source text, so two roots that agree on all of the cheap fields but are
    /// backed by different text cannot share an answer.
    fn key_prefix(&self, arch: &str) -> &'static str {
        use std::collections::HashMap;
        use std::sync::{Mutex, OnceLock};

        static PREFIXES: OnceLock<Mutex<HashMap<String, &'static str>>> = OnceLock::new();
        let identity = format!(
            "{}\u{1f}{arch}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{:p}",
            self.name,
            self.options.join(","),
            self.floor,
            self.headers.tag(),
            self.text.as_ptr(),
        );
        let cache = PREFIXES.get_or_init(|| Mutex::new(HashMap::new()));
        let mut cache = cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(found) = cache.get(&identity) {
            return found;
        }
        let prefix: &'static str = Box::leak(
            format!(
                "jit/{}/{arch}/{FLOAT_CONTRACT}/{}/nvrtc>={}/{}/r{:016x}/h{:016x}",
                self.name,
                self.options.join(","),
                self.floor,
                self.headers.tag(),
                source::fnv1a64(self.text.as_bytes()),
                source::digest(self.header_set()),
            )
            .into_boxed_str(),
        );
        cache.insert(identity, prefix);
        prefix
    }
}

const FLOAT_CONTRACT: &str = "fmad=false,prec-div=true,prec-sqrt=true";

const fn strip_cuh(file: &'static str) -> &'static str {
    let bytes = file.as_bytes();
    let Some(stem) = bytes.len().checked_sub(4) else {
        panic!("a root's file is a `.cuh` under `kernels/`")
    };
    let (name, extension) = file.split_at(stem);
    if !matches!(extension.as_bytes(), b".cuh") {
        panic!("a root's file is a `.cuh` under `kernels/`")
    }
    name
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two roots the memo has to keep apart: different files, so different
    /// source text, header sets and options.
    fn two_roots() -> (Root, Root) {
        (
            Root::new("graph/supergraph.cuh"),
            Root::new("norm/rmsnorm.cuh"),
        )
    }

    /// The memo is an optimisation, so the key it hands back has to be the key
    /// the unmemoized formatting would have produced — every field, in order.
    /// Written out longhand rather than by calling the private helper, so a
    /// field dropped from the prefix fails here instead of agreeing with
    /// itself.
    #[test]
    fn the_memoized_key_states_everything_it_used_to() {
        let root = Root::new("graph/supergraph.cuh");
        let want = format!(
            "jit/{}/sm_89/{FLOAT_CONTRACT}/{}/nvrtc>={}/{}/r{:016x}/h{:016x}/i{:016x}",
            root.name,
            root.options.join(","),
            root.floor,
            root.headers.tag(),
            source::fnv1a64(root.text.as_bytes()),
            source::digest(root.header_set()),
            source::fnv1a64(b"::pie::graph::probe"),
        );
        assert_eq!(root.key("::pie::graph::probe", "sm_89"), want);
    }

    /// A memo keyed too loosely would answer for the wrong root. The prefix is
    /// shared across instantiations BY DESIGN, so these three axes are the ones
    /// that must still separate: the root, the arch, and the instantiation.
    #[test]
    fn the_memo_does_not_merge_two_roots_two_arches_or_two_instantiations() {
        let (a, b) = two_roots();
        let inst = "::pie::probe";
        assert_ne!(
            a.key(inst, "sm_89"),
            b.key(inst, "sm_89"),
            "two different roots"
        );
        assert_ne!(
            a.key(inst, "sm_89"),
            a.key(inst, "sm_90"),
            "one root, two arches"
        );
        assert_ne!(
            a.key("::pie::probe", "sm_89"),
            a.key("::pie::other", "sm_89"),
            "one root, two instantiations"
        );
    }

    /// The property the fix depends on: asking twice is asking once. A memo
    /// that rebuilt the prefix per call would still pass the equality tests
    /// above while costing what it cost before, so this one asserts the answer
    /// is the SAME ALLOCATION, which only a hit can produce.
    #[test]
    fn the_prefix_is_computed_once_per_root_and_arch() {
        let root = Root::new("graph/supergraph.cuh");
        let first = root.key_prefix("sm_89");
        let second = root.key_prefix("sm_89");
        assert_eq!(
            first.as_ptr(),
            second.as_ptr(),
            "the second ask rebuilt the prefix instead of finding it"
        );
    }
}
