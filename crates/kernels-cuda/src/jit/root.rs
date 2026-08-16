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
    /// [`DEVICE_HEADERS`]: `shim/` and `kernels/` minus the internalised
    Library,
    /// [`ALL_HEADERS`]: the above plus `kernels/flashinfer` and
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
    /// The root's name: its path under `kernels/` without the extension.
    pub name: &'static str,
    /// The text, handed to `nvrtcCreateProgram`.
    pub text: &'static str,
    /// The path a diagnostic names, relative to `kernels/`.
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

/// The carried files whose compile is not the default, and what each asks for.
///
/// # Why this is a table and not three builder calls at the declaration
///
/// Options, header set and floor are properties of the FILE — of what the C++
/// says — and not of the site that fires it. `cascade/merge_states.cuh` needs
/// `--device-as-default-execution-space` because it is upstream's header with
/// three `using` declarations on top, and that is true of every compile of it
/// from anywhere. Measured across every declaration in this crate, **no file
/// was ever declared under two different option sets**, which is what makes a
/// per-file table an answer rather than a guess.
///
/// Stating it here is what lets a launch name a file and nothing else. While
/// the options lived on the declaration, a `Root` could only be assembled
/// where someone had written the chain out, so every launch had to reach a
/// `static ROOT` by path; a fire that dropped a `.upstream()` would compile
/// under the wrong header set and fail at NVRTC rather than at review.
const CONFIGURED: &[(&str, &[&str], Headers, Toolchain)] = &[
    // Two `#include`s reach `attn/flashinfer/attention/mla{,_params}.cuh`,
    // which the library set does not answer, and `grid.sync()` leaves
    // `cudaCGGetIntrinsicHandle` extern for the `cuLink` step to close.
    (
        "attn/attention_mla_fa2.cuh",
        &["--device-as-default-execution-space", "--relocatable-device-code=true"],
        Headers::LibraryAndUpstream,
        Toolchain::ANY,
    ),
    // Upstream's header with three `using` declarations on top: without the
    // option NVRTC parses `cascade.cuh`'s helpers as host functions.
    (
        "cascade/merge_states.cuh",
        &["--device-as-default-execution-space"],
        Headers::LibraryAndUpstream,
        Toolchain::ANY,
    ),
    // Two `#include`s reach `attn/flashinfer/comm/`, which the library set
    // does not answer. NO `--device-as-default-execution-space`, and that is
    // measured rather than assumed: every surviving function in both comm
    // headers carries `__device__`, `__global__` or `DINLINE`, because
    // removing their host halves is exactly what left that true. rc=0 without
    // the flag at sm_89, sm_90a, sm_100a and sm_120a, 32 template-ids lowered
    // at each; sm_80 and sm_86 refuse on `shim/cuda_fp8.h`'s own `#error`,
    // which every root under this header set hits.
    (
        "comm/all_reduce.cuh",
        &[],
        Headers::LibraryAndUpstream,
        Toolchain::ANY,
    ),
    // The FA2 lattice's one file. Its 56 points differ in NAME alone, so what
    // they compile under belongs here with every other per-file fact.
    (
        "attn/fa2.cuh",
        &["--device-as-default-execution-space"],
        Headers::LibraryAndUpstream,
        Toolchain::ANY,
    ),
    // XQA's one file. The header set is the file's; the `-D` set is NOT, and
    // is the one thing in this crate that genuinely varies per point -- five
    // members compile this text five ways -- so `xqa` states its own with
    // `.options()` and only the closure is answered here.
    (
        "attn/attention_xqa_mha.cuh",
        &[],
        Headers::LibraryAndUpstream,
        Toolchain::ANY,
    ),
    // `cuda::std::tile` is 13.3 and later. The five below hold no kernel this
    // crate fires; they are compiled by `every_instantiation_compiles` alone,
    // and the floor is what keeps that from failing on an older NVRTC.
    ("tile/alternatives.cuh", &[], Headers::Library, Toolchain::new(13, 3)),
    ("sample/argmax_tile.cuh", &[], Headers::Library, Toolchain::new(13, 3)),
    ("layout/gather_rows_tile.cuh", &[], Headers::Library, Toolchain::new(13, 3)),
    ("quant/dequant_wna16_tile.cuh", &[], Headers::Library, Toolchain::new(13, 3)),
    ("quant/wna16_gemv_tile.cuh", &[], Headers::Library, Toolchain::new(13, 3)),
];

/// What [`CONFIGURED`] says about `file`, or the defaults.
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
    /// The root compiled out of the carried file `file` names.
    ///
    /// One argument, because one is all there is. The text is
    /// [`source::carried`]'s, the name is the same string without its
    /// extension, and the options, header set and floor are [`CONFIGURED`]'s —
    /// every one of them a fact about the FILE, so none of them has to be
    /// restated where a kernel is fired.
    ///
    /// A name no carried file answers to fails const evaluation at this call,
    /// which is the guarantee the `include_str!` this replaced used to give.
    #[must_use]
    pub const fn new(file: &'static str) -> Self {
        let (options, headers, floor) = configured_for(file);
        Self { name: strip_cuh(file), text: source::carried(file), file, options, headers, floor }
    }

    /// The same, resolved at RUN time, for a file named by a launch.
    ///
    /// [`Root::new`] panics on a name nothing carries, which is what makes it
    /// a compile-time check where the name is a literal in a `static`. A
    /// launch names its file as an ordinary argument, so the same miss has to
    /// be a value: a kernel that cannot be found must refuse, not abort the
    /// process it was fired from.
    #[must_use]
    pub fn of(file: &'static str) -> Option<Self> {
        let text = source::text_of(file)?;
        let name = file.strip_suffix(".cuh")?;
        let (options, headers, floor) = configured_for(file);
        Some(Self { name, text, file, options, headers, floor })
    }

    /// A root whose name is not its file's.
    ///
    /// For the two lattices only. FA2 compiles `attn/fa2.cuh` at 56 points and
    /// XQA `attn/attention_xqa_mha.cuh` at five, each under its own `-D` set,
    /// and the cache key spans the name — so the points have to be *named*
    /// apart (`attn/fa2_decode_hd128_g4`) while sharing one carried file.
    /// [`Root::new`] cannot express that, and widening it to two arguments for
    /// three call sites would put the redundancy back on every other one.
    #[must_use]
    pub const fn variant(name: &'static str, file: &'static str) -> Self {
        let (options, headers, floor) = configured_for(file);
        Self { name, text: source::carried(file), file, options, headers, floor }
    }

    /// A root assembled from parts, for this file's tests alone.
    ///
    /// [`Root::key`] must be shown to span the root TEXT, and no constructor
    /// above can vary text under a fixed name — the text is derived from the
    /// name, which is the property being relied on everywhere else. Production
    /// has three ways to make a root and this is not one of them.
    #[cfg(test)]
    const fn from_parts(name: &'static str, text: &'static str, file: &'static str) -> Self {
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

/// `file` without its `.cuh`, which is what a root is NAMED.
///
/// A root's name reaches a diagnostic and the cache key, and it has always
/// been the path under `kernels/` with the extension dropped. Deriving it here
/// rather than asking for it again is what keeps one key stable across this
/// change: `Root::new("layout/slot_ops.cuh")` names itself `layout/slot_ops`,
/// exactly as the three-argument form was written by hand.
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
    use super::Root;
    use crate::source::DEVICE_HEADERS;

    const TEXT: &str = "__global__ void nothing() {}";
    const R: Root = Root::from_parts("test/root", TEXT, "test/root.cuh");

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
            Root::from_parts("test/root", "__global__ void nothing() { ; }", "test/root.cuh")
                .key("a::b<1>", "sm_90"),
            "an edited root"
        );
        assert_ne!(base, Root::from_parts("test/elsewhere", TEXT, "x.cuh").key("a::b<1>", "sm_90"));
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
