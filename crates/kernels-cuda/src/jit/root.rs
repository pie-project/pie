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
    /// Compiled for the device's arch-specific target (`sm_100a`), not the
    /// plain one: tcgen05, TMEM and the block-scaled MMAs exist only there,
    /// and a cubin built for it loads on that exact chip alone.
    pub specific: bool,
}

const CONFIGURED: &[(&str, &[&str], Headers, Toolchain)] = &[
    (
        "attn/mla.cuh",
        &["--device-as-default-execution-space"],
        Headers::LibraryAndUpstream,
        Toolchain::ANY,
    ),
    (
        "collective/all_reduce.cuh",
        &[],
        Headers::LibraryAndUpstream,
        Toolchain::ANY,
    ),
    (
        "attn/attention.cuh",
        &["--device-as-default-execution-space"],
        Headers::LibraryAndUpstream,
        Toolchain::ANY,
    ),
];

/// The units compiled for the arch-specific target, each with the NVRTC
/// that first spells everything it asks of it (`.block16` is 12.9's).
const SPECIFIC: &[(&str, Toolchain)] = &[
    ("linear/tc5.cuh", Toolchain::new(12, 9)),
    ("attn/tc5_fmha.cuh", Toolchain::new(12, 9)),
    ("attn/gdn_tc5.cuh", Toolchain::new(12, 9)),
];

const fn specific_for(file: &str) -> Option<Toolchain> {
    let mut i = 0;
    while i < SPECIFIC.len() {
        let (name, floor) = SPECIFIC[i];
        if source::str_eq(name, file) {
            return Some(floor);
        }
        i += 1;
    }
    None
}

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
    pub fn of(file: &'static str) -> Option<Self> {
        let text = source::text_of(file)?;
        let name = file.strip_suffix(".cuh")?;
        let (options, headers, floor) = configured_for(file);
        let (specific, floor) = specific_for(file).map_or((false, floor), |floor| (true, floor));
        Some(Self {
            name,
            text,
            file,
            options,
            headers,
            floor,
            specific,
        })
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
    pub const fn arch_specific(mut self) -> Self {
        self.specific = true;
        self
    }

    /// The target this unit compiles for on a device whose plain target is
    /// `arch`: that one, or its arch-specific twin when the unit asks for
    /// it. `None` when the unit asks and the device has no twin.
    #[must_use]
    pub fn target(&self, arch: &str) -> Option<String> {
        if self.specific {
            specific_of(arch)
        } else {
            Some(arch.to_owned())
        }
    }

    #[must_use]
    pub fn header_set(&self) -> &'static [Header] {
        self.headers.set()
    }

    #[must_use]
    pub fn key(&self, instantiation: &str, arch: &str) -> String {
        format!(
            "{}/i{:016x}",
            self.key_prefix(arch),
            source::fnv1a64(instantiation.as_bytes()),
        )
    }

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

/// The arch-specific twin of a plain `sm_XY` target, for the chips that
/// have one: Hopper's `sm_90a`, the datacenter Blackwells' `sm_100a` and
/// `sm_103a`, Thor's `sm_110a` and the consumer Blackwells' `sm_120a` and
/// `sm_121a`. An `a` cubin loads on that compute capability alone, so the
/// twin is never a neighbour's.
#[must_use]
pub fn specific_of(arch: &str) -> Option<String> {
    let digits = arch.strip_prefix("sm_")?;
    matches!(digits, "90" | "100" | "101" | "103" | "110" | "120" | "121")
        .then(|| format!("sm_{digits}a"))
}

/// What a device's arch-specific target admits, read off its plain target
/// and the NVRTC that would compile for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Features {
    /// `tcgen05.*` over TMEM with `kind::f16`, `f8f6f4`, `mxf8f6f4` and
    /// `mxf4nvf4`: the datacenter Blackwells (10.x) and Thor (11.0).
    pub tcgen05: bool,
    /// Warp-level `mma.sync` with `kind::mxf8f6f4`/`mxf4nvf4.block_scale`:
    /// the consumer Blackwells (12.x), which have no TMEM.
    pub warp_block_scale: bool,
}

impl Features {
    /// The first NVRTC to spell `.block16`/`.block32`. An older one admits
    /// nothing here, since no unit that asks would compile.
    pub const NVRTC: Toolchain = Toolchain::new(12, 9);

    #[must_use]
    pub fn of(arch: &str, nvrtc: Toolchain) -> Self {
        let twin = specific_of(arch).filter(|_| Self::NVRTC.met_by(nvrtc));
        let twin = twin.as_deref().unwrap_or("");
        Self {
            tcgen05: matches!(twin, "sm_100a" | "sm_101a" | "sm_103a" | "sm_110a"),
            warp_block_scale: matches!(twin, "sm_120a" | "sm_121a"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_arch_specific_twin_is_named_for_its_own_chip_only() {
        assert_eq!(specific_of("sm_100").as_deref(), Some("sm_100a"));
        assert_eq!(specific_of("sm_103").as_deref(), Some("sm_103a"));
        assert_eq!(specific_of("sm_120").as_deref(), Some("sm_120a"));
        assert_eq!(specific_of("sm_90").as_deref(), Some("sm_90a"));
        for none in ["sm_80", "sm_89", "sm_100a", "compute_100", "sm_1x0"] {
            assert_eq!(specific_of(none), None, "{none} has no twin");
        }
    }

    #[test]
    fn tcgen05_is_the_datacenter_blackwells_and_warp_block_scale_the_consumer_ones() {
        let nvrtc = Toolchain::new(13, 0);
        let none = Features {
            tcgen05: false,
            warp_block_scale: false,
        };
        assert!(Features::of("sm_100", nvrtc).tcgen05);
        assert!(Features::of("sm_103", nvrtc).tcgen05);
        assert!(!Features::of("sm_100", nvrtc).warp_block_scale);
        assert!(!Features::of("sm_120", nvrtc).tcgen05);
        assert!(Features::of("sm_120", nvrtc).warp_block_scale);
        assert_eq!(Features::of("sm_90", nvrtc), none);
        assert_eq!(Features::of("sm_89", nvrtc), none);
        assert_eq!(
            Features::of("sm_100", Toolchain::new(12, 8)),
            none,
            "an NVRTC that cannot spell block16 admits none of it"
        );
    }

    #[test]
    fn an_arch_specific_unit_keys_apart_from_the_plain_one() {
        let plain = Root::of("linear/skinny.cuh").expect("a carried unit");
        let specific = plain.arch_specific();
        let a = plain.key("x", &plain.target("sm_100").expect("plain"));
        let b = specific.key("x", &specific.target("sm_100").expect("twin"));
        assert_ne!(a, b, "an sm_100a cubin must not answer for an sm_100 one");
        assert!(b.contains("/sm_100a/"), "{b}");
        assert_eq!(specific.target("sm_89"), None);
    }
}
