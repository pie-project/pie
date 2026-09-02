//! A compilation root: one carried `.cuh`, the options it compiles under,
//! the header set it resolves against, and the cache key all of that folds
//! into. Most units take the defaults; the exceptions are configured by
//! name.

use core::fmt;

use crate::source::{self, ALL_HEADERS, DEVICE_HEADERS, Header};

/// The NVRTC floor a unit states, `0.0` meaning any.
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

/// Which header closure a unit compiles against: the plane's own text, or
/// that plus the internalised upstream (FlashInfer/XQA) tree.
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

/// The units that do not take the defaults. The rows live here rather than
/// with the entries that fire them, so the configuration travels with the
/// file name.
const CONFIGURED: &[(&str, &[&str], Headers, Toolchain)] = &[
    (
        "attn/mla.cuh",
        &[
            "--device-as-default-execution-space",
            "--relocatable-device-code=true",
        ],
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
    /// The carried unit with this name, or `None` — a [`Fire`](crate::jit::Fire)
    /// naming a file the binary does not carry is refused, not conjured.
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

    /// The disk/memory cache key: everything that can change the cubin.
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
