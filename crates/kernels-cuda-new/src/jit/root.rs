use core::fmt;

use crate::source::{self, ALL_HEADERS, DEVICE_HEADERS, Header};

/// The oldest NVRTC that may compile a root, as NVRTC reports itself:
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Toolchain {
    /// NVRTC's major version: the `13` of 13.3.
    pub major: u32,
    /// NVRTC's minor version: the `3` of 13.3.
    pub minor: u32,
}

impl Toolchain {
    /// No floor at all — every unit authored here, today.
    pub const ANY: Self = Self { major: 0, minor: 0 };

    /// A floor of `major.minor`.
    #[must_use]
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    /// Whether this is [`Toolchain::ANY`] — nothing to check, nothing to ask.
    #[must_use]
    pub const fn is_any(self) -> bool {
        self.major == 0 && self.minor == 0
    }

    /// Whether an NVRTC reporting `have` may compile a root whose floor is
    #[must_use]
    pub const fn met_by(self, have: Self) -> bool {
        have.major > self.major || (have.major == self.major && have.minor >= self.minor)
    }
}

impl fmt::Display for Toolchain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_any() { f.write_str("any") } else { write!(f, "{}.{}", self.major, self.minor) }
    }
}

/// Which carried header set a root's `#include`s resolve against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Headers {
    /// [`DEVICE_HEADERS`]: `csrc/shim` and `csrc/src` minus the internalised
    Library,
    /// [`ALL_HEADERS`]: the above plus `csrc/src/attn/flashinfer` and
    LibraryAndUpstream,
}

impl Headers {
    /// The set itself, as `nvrtcCreateProgram` will be handed it.
    #[must_use]
    pub const fn set(self) -> &'static [Header] {
        match self {
            Headers::Library => DEVICE_HEADERS,
            Headers::LibraryAndUpstream => ALL_HEADERS,
        }
    }

    /// The choice's name in a cache key.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Headers::Library => "lib",
            Headers::LibraryAndUpstream => "lib+upstream",
        }
    }
}

/// One compilable root: the device text a routine's symbol comes out of.
///
/// A root is not a root. A unit carried a ROW LIST — the instantiations to ask
/// for — because the whole set had to be enumerated before anything could be
/// compiled. Compilation is per symbol now, so the enumeration has no reader
/// and the root carries only what a compile of it needs.
#[derive(Clone, Copy, Debug)]
pub struct Root {
    /// The root's name: its path under `csrc/src` without the extension.
    pub name: &'static str,
    /// The text, handed to `nvrtcCreateProgram`.
    pub text: &'static str,
    /// The path a diagnostic names, relative to `csrc/src`.
    pub file: &'static str,
    /// NVRTC options this root needs and the others must not have.
    ///
    /// Part of the cache key, and not decoratively: XQA compiles ONE root five
    /// ways by `-D` alone, and `--device-as-default-execution-space` changes
    /// what compiles at all. Neither is visible in a symbol string.
    pub options: &'static [&'static str],
    /// Which carried header set its `#include`s resolve against.
    pub headers: Headers,
    /// The oldest NVRTC that may compile it.
    pub floor: Toolchain,
}

impl Root {
    /// A root that asks for nothing beyond the library headers.
    #[must_use]
    pub const fn new(name: &'static str, text: &'static str, file: &'static str) -> Self {
        Self { name, text, file, options: &[], headers: Headers::Library, floor: Toolchain::ANY }
    }

    /// The same root, with NVRTC options.
    #[must_use]
    pub const fn options(mut self, options: &'static [&'static str]) -> Self {
        self.options = options;
        self
    }

    /// The same root, resolving `#include`s against the upstream closure too.
    #[must_use]
    pub const fn upstream(mut self) -> Self {
        self.headers = Headers::LibraryAndUpstream;
        self
    }

    /// The same root, refusing an NVRTC older than `major.minor`.
    #[must_use]
    pub const fn since(mut self, major: u32, minor: u32) -> Self {
        self.floor = Toolchain::new(major, minor);
        self
    }

    /// The header set a compile of this root is handed.
    #[must_use]
    pub fn header_set(&self) -> &'static [Header] {
        self.headers.set()
    }

    /// Whether this root's cubin must be device-linked before it will load.
    ///
    /// Stated as an option rather than a field, because the two are one
    /// decision: relocatable device code is what leaves the extern unresolved
    /// for a link step, and asking for it without the link produces the
    /// `ptxas` fatal this exists to avoid.
    #[must_use]
    pub fn needs_device_runtime(&self) -> bool {
        self.options
            .iter()
            .any(|o| *o == "--relocatable-device-code=true" || *o == "-dc" || *o == "--device-c")
    }

    /// The key one INSTANTIATION of this root is cached under.
    ///
    /// Every term is something that changes the cubin. Dropping any one of
    /// them makes a stale image loadable:
    ///
    /// - the instantiation, because that is what is being compiled;
    /// - the options, because one root compiles five ways under `-D` alone;
    /// - the architecture, because a cubin is per-architecture;
    /// - the root text and the header bytes, because either is the source;
    /// - the float contract, because it is the arithmetic the answer is in;
    /// - the floor, because a raised floor means the old cubin came from a
    ///   compiler this root now refuses.
    #[must_use]
    pub fn key(&self, instantiation: &str, arch: &str) -> String {
        format!(
            "jit/{}/{arch}/{FLOAT_CONTRACT}/{}/nvrtc>={}/{}/r{:016x}/h{:016x}/i{:016x}",
            self.name,
            self.options.join(","),
            self.floor,
            self.headers.tag(),
            source::fnv1a64(self.text.as_bytes()),
            source::digest(self.header_set()),
            source::fnv1a64(instantiation.as_bytes()),
        )
    }
}

/// The float flags every compile in this crate is invoked with, as one string.
const FLOAT_CONTRACT: &str = "fmad=false,prec-div=true,prec-sqrt=true";

#[cfg(test)]
mod tests {
    use super::Root;
    use crate::source::DEVICE_HEADERS;

    const TEXT: &str = "__global__ void nothing() {}";
    const R: Root = Root::new("test/root", TEXT, "test/root.cuh");

    /// Every term the key spans moves it, checked one at a time.
    #[test]
    fn the_key_moves_when_anything_it_spans_does() {
        let base = R.key("a::b<1>", "sm_90");
        assert_eq!(base, R.key("a::b<1>", "sm_90"), "and is stable");
        assert_ne!(base, R.key("a::b<2>", "sm_90"), "a different instantiation");
        assert_ne!(base, R.key("a::b<1>", "sm_80"), "a cubin is per-architecture");
        assert_ne!(base, R.options(&["-DX=1"]).key("a::b<1>", "sm_90"), "an option");
        assert_ne!(base, R.upstream().key("a::b<1>", "sm_90"), "a header set");
        assert_ne!(base, R.since(13, 3).key("a::b<1>", "sm_90"), "a raised floor");
        assert_ne!(
            base,
            Root::new("test/root", "__global__ void nothing() { ; }", "test/root.cuh")
                .key("a::b<1>", "sm_90"),
            "an edited root"
        );
        assert_ne!(base, Root::new("test/elsewhere", TEXT, "x.cuh").key("a::b<1>", "sm_90"));
        assert!(base.contains("fmad=false"), "the arithmetic it was built under");
    }

    /// The header BYTES are spanned, not merely the choice of set.
    #[test]
    fn an_edited_header_moves_the_key() {
        let base = R.key("a::b<1>", "sm_90");
        let mut edited = DEVICE_HEADERS.to_vec();
        edited[0].text = "// not what it was";
        let digest = crate::source::digest(&edited);
        assert!(
            !base.contains(&format!("h{digest:016x}")),
            "an edited header changes what compiles and leaves the root alone"
        );
    }

    /// Device linking is read off the options, so the two cannot disagree.
    #[test]
    fn relocatable_device_code_is_what_asks_for_the_link() {
        assert!(!R.needs_device_runtime());
        assert!(R.options(&["--relocatable-device-code=true"]).needs_device_runtime());
        assert!(R.options(&["-dc"]).needs_device_runtime());
    }
}
