use std::fmt;

use crate::abi;
use crate::device::DeviceKernel;
use crate::source::{self, ALL_HEADERS, DEVICE_HEADERS, Header};

/// One compilable unit: a root source, and the instantiations wanted out of
#[derive(Clone, Copy)]
pub struct Unit {
    /// The unit's name, which is its root's path under `csrc/src` without the
    pub name: &'static str,
    /// The root source, handed to `nvrtcCreateProgram`.
    pub root: &'static str,
    /// The instantiations this compile is asked for.
    pub rows: &'static [DeviceKernel],
    /// NVRTC options this unit needs and the others must not have.
    pub options: &'static [&'static str],
}

/// The oldest NVRTC that may compile a unit, as NVRTC reports itself:
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

    /// Whether an NVRTC reporting `have` may compile a unit whose floor is
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

/// Which carried header set a unit's `#include`s resolve against.
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

/// What a unit demands of the machine that compiles it, and of the text it is
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Demands {
    /// The oldest NVRTC that may compile this unit.
    pub floor: Toolchain,
    /// The header set its `#include`s resolve against.
    pub headers: Headers,
}

impl Demands {
    /// What a unit demands when it says nothing: any compiler, the library
    pub const DEFAULT: Self = Self { floor: Toolchain::ANY, headers: Headers::Library };
}

/// The units that demand something other than [`Demands::DEFAULT`].
const DEMANDS: &[(&str, Demands)] = &[
    (
        "attn/fa2_*",
        Demands { floor: Toolchain::ANY, headers: Headers::LibraryAndUpstream },
    ),
    (
        "cascade/merge_states",
        Demands { floor: Toolchain::ANY, headers: Headers::LibraryAndUpstream },
    ),
    (
        "attn/attention_xqa_mha_*",
        Demands { floor: Toolchain::ANY, headers: Headers::LibraryAndUpstream },
    ),
    (
        "attn/attention_mla_fa2",
        Demands { floor: Toolchain::ANY, headers: Headers::LibraryAndUpstream },
    ),
];

/// Whether a [`DEMANDS`] key applies to a unit name.
const fn demand_covers(key: &str, unit: &str) -> bool {
    let key_bytes = key.as_bytes();
    if key_bytes.is_empty() {
        return false;
    }
    if key_bytes[key_bytes.len() - 1] != b'*' {
        return str_eq(key, unit);
    }
    let prefix = key_bytes.len() - 1;
    let unit_bytes = unit.as_bytes();
    if unit_bytes.len() < prefix {
        return false;
    }
    let mut i = 0;
    while i < prefix {
        if key_bytes[i] != unit_bytes[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Every demand names a unit that exists.
const fn every_demand_names_a_unit() -> bool {
    let units = concat_families();
    let mut d = 0;
    while d < DEMANDS.len() {
        let mut found = false;
        let mut u = 0;
        while u < units.len() {
            if demand_covers(DEMANDS[d].0, units[u].name) {
                found = true;
            }
            u += 1;
        }
        if !found {
            return false;
        }
        d += 1;
    }
    true
}

const _: () = assert!(
    every_demand_names_a_unit(),
    "a demand names a unit that is not in UNITS -- a floor or a header set \
     keyed on a name nothing answers to is a demand that never applies"
);

/// `==` on `&str`, in a `const fn`.
const fn str_eq(a: &str, b: &str) -> bool {
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

impl Unit {
    /// What this unit demands of the compiler and of the header set.
    #[must_use]
    pub fn demands(&self) -> Demands {
        DEMANDS
            .iter()
            .find(|(key, _)| demand_covers(key, self.name))
            .map_or(Demands::DEFAULT, |(_, demands)| *demands)
    }

    /// The oldest NVRTC that may compile this unit.
    #[must_use]
    pub fn floor(&self) -> Toolchain {
        self.demands().floor
    }

    /// Which set this unit's `#include`s resolve against.
    #[must_use]
    pub fn headers(&self) -> Headers {
        self.demands().headers
    }

    /// That set's text, as a compile is handed it.
    #[must_use]
    pub fn header_set(&self) -> &'static [Header] {
        self.headers().set()
    }

    /// The row this unit holds for `symbol`, if it holds one.
    #[must_use]
    pub fn row(&self, symbol: &str) -> Option<&'static DeviceKernel> {
        self.rows.iter().find(|row| row.sig.symbol == symbol)
    }

    /// Whether this unit is the one that would compile `symbol`.
    #[must_use]
    pub fn hosts(&self, symbol: &str) -> bool {
        self.row(symbol).is_some()
    }

    /// Every row's instantiation, in the table's order.
    #[must_use]
    pub fn instantiations(&self) -> Vec<String> {
        self.rows.iter().map(DeviceKernel::instantiation).collect()
    }

    /// The assertions that prove `rows` against the templates in this unit's
    pub fn typecheck(&self, rows: &[&DeviceKernel]) -> Result<abi::DeviceTypecheck, String> {
        abi::device_typecheck(rows, abi::Site::Appendix, abi::Elem::Opaque)
    }

    /// What `nvrtcCreateProgram` is handed: this unit's root, with the
    pub fn source(&self, rows: &[&DeviceKernel]) -> Result<String, String> {
        let tail = self.typecheck(rows)?;
        Ok(format!("{}\n\n{}", self.root, tail.text))
    }

    /// [`Unit::source`] over every row this unit holds, for a caller that
    fn compiled_text(&self) -> String {
        let rows: Vec<&DeviceKernel> = self.rows.iter().collect();
        self.source(&rows)
            .unwrap_or_else(|why| format!("{}\n\n// TYPECHECK REFUSED: {why}\n", self.root))
    }

    /// The key a compiled unit may be cached under, for `arch` and the
    #[must_use]
    pub fn cache_key(&self, arch: &str) -> String {
        self.cache_key_with(arch, self.header_set())
    }

    /// [`Unit::cache_key`] over an arbitrary header set, so a test can show
    #[must_use]
    pub fn cache_key_with(&self, arch: &str, headers: &[Header]) -> String {
        self.cache_key_under(arch, headers, self.demands())
    }

    /// [`Unit::cache_key`] over an arbitrary header set AND an arbitrary
    #[must_use]
    pub fn cache_key_under(&self, arch: &str, headers: &[Header], demands: Demands) -> String {
        let mut wanted = String::new();
        for instantiation in self.instantiations() {
            wanted.push_str(&instantiation);
            wanted.push('\0');
        }
        format!(
            "jit/{}/{arch}/{FLOAT_CONTRACT}/{}/nvrtc>={}/{}/r{:016x}/h{:016x}/n{:016x}",
            self.name,
            self.options.join(","),
            demands.floor,
            demands.headers.tag(),
            source::fnv1a64(self.compiled_text().as_bytes()),
            source::digest(headers),
            source::fnv1a64(wanted.as_bytes()),
        )
    }
}

/// The float flags every compile in this crate is invoked with, as one string.
const FLOAT_CONTRACT: &str = "fmad=false,prec-div=true,prec-sqrt=true";

/// Every unit this crate can compile, in [`crate::families::ALL`]'s order.
pub static UNITS: &[Unit] = &concat_families();

/// `[&[Unit]] -> [Unit]` at compile time.
const fn concat_families() -> [Unit; UNIT_COUNT] {
    let mut out = [EMPTY_UNIT; UNIT_COUNT];
    let mut w = 0;
    let mut f = 0;
    while f < crate::families::ALL.len() {
        let family = crate::families::ALL[f];
        let mut i = 0;
        while i < family.len() {
            out[w] = family[i];
            w += 1;
            i += 1;
        }
        f += 1;
    }
    out
}

/// How many units every family declares, together.
const UNIT_COUNT: usize = count_units();

const fn count_units() -> usize {
    let mut n = 0;
    let mut f = 0;
    while f < crate::families::ALL.len() {
        n += crate::families::ALL[f].len();
        f += 1;
    }
    n
}

/// A slot to fill, never a unit anything can fire: it names no source and
const EMPTY_UNIT: Unit = Unit { name: "", root: "", rows: &[], options: &[] };

pub fn unit_of(symbol: &str) -> Option<(usize, &'static Unit)> {
    UNITS
        .iter()
        .enumerate()
        .find(|(_, unit)| unit.hosts(symbol))
}

/// Every row of every unit, in [`UNITS`]' order.
pub fn rows() -> impl Iterator<Item = &'static DeviceKernel> {
    UNITS.iter().flat_map(|unit| unit.rows.iter())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row of a unit names ONE file, because a unit compiles one root.
    ///
    /// It used to assert `row.sig.file == "{unit.name}.cuh"`, which holds only
    /// where a unit and its file are 1:1. A LATTICE breaks that and is not a
    /// defect: `attn/attention_xqa_mha_gqa2_p32` and its siblings all compile
    /// `attn/attention_xqa_mha.cuh` at different geometries, as the FA2 decode
    /// and prefill units compile `attn/fa2.cuh`. What must hold is that a
    /// unit's rows agree with each other.
    #[test]
    fn every_row_is_in_the_unit_its_file_names() {
        for unit in UNITS {
            let mut files = unit.rows.iter().map(|row| row.sig.file);
            let Some(first) = files.next() else { continue };
            for file in files {
                assert_eq!(
                    file, first,
                    "unit `{}` compiles one root, and its rows name two files",
                    unit.name
                );
            }
        }
    }

    /// [`unit_of`] is a function: one symbol, at most one unit.
    #[test]
    fn no_symbol_is_hosted_by_two_units() {
        for row in rows() {
            let hosts: Vec<&str> = UNITS
                .iter()
                .filter(|unit| unit.hosts(row.sig.symbol))
                .map(|unit| unit.name)
                .collect();
            assert_eq!(hosts.len(), 1, "{} is hosted by {hosts:?}", row.sig.symbol);
            let (index, unit) = unit_of(row.sig.symbol).expect("a row's own unit");
            assert_eq!(unit.name, hosts[0]);
            assert_eq!(UNITS[index].name, hosts[0], "the index addresses the unit");
            assert_eq!(
                unit.row(row.sig.symbol).map(|found| found.sig.symbol),
                Some(row.sig.symbol)
            );
        }
    }

    /// The key spans everything that can change the cubin, checked one term
    #[test]
    fn the_cache_key_moves_when_anything_it_spans_does() {
        let unit = crate::x::norm::altup_aux::ALTUP_AUX;
        let base = unit.cache_key("sm_90");
        assert_eq!(base, unit.cache_key("sm_90"), "and is stable");
        assert_ne!(base, unit.cache_key("sm_80"), "a cubin is per-architecture");

        let mut edited = DEVICE_HEADERS.to_vec();
        edited[0].text = "// not what it was";
        assert_ne!(
            base,
            unit.cache_key_with("sm_90", &edited),
            "an edited header changes what compiles and leaves the root alone"
        );

        let fewer = Unit { rows: &unit.rows[..1], ..unit };
        assert_ne!(
            base,
            fewer.cache_key("sm_90"),
            "a row set is a symbol set, and an old cubin does not hold a new one"
        );

        let retitled = Unit { name: "norm/elsewhere", ..unit };
        assert_ne!(base, retitled.cache_key("sm_90"));

        assert!(
            base.contains(FLOAT_CONTRACT),
            "the arithmetic a cubin was built under is part of what it answers"
        );

        assert_ne!(
            crate::x::norm::altup_aux::ALTUP_AUX.cache_key("sm_90"),
            crate::x::norm::elementwise::ELEMENTWISE.cache_key("sm_90"),
            "two units, two roots, two keys"
        );
    }

    /// The two new terms are in the key, checked one at a time and through
    #[test]
    fn the_cache_key_moves_when_the_floor_or_the_header_choice_does() {
        let unit = crate::x::norm::altup_aux::ALTUP_AUX;
        let base = unit.cache_key("sm_90");
        assert_eq!(
            base,
            unit.cache_key_under("sm_90", DEVICE_HEADERS, Demands::DEFAULT),
            "the default demand is what every unit gets today, so it is the same key"
        );

        let floored = Demands { floor: Toolchain::new(13, 3), ..Demands::DEFAULT };
        assert_ne!(
            base,
            unit.cache_key_under("sm_90", DEVICE_HEADERS, floored),
            "a unit whose floor moved is a unit whose old cubin came from a \
             compiler it now refuses"
        );

        let upstream = Demands { headers: Headers::LibraryAndUpstream, ..Demands::DEFAULT };
        assert_ne!(
            base,
            unit.cache_key_under("sm_90", DEVICE_HEADERS, upstream),
            "the header-set CHOICE is spanned, not merely the bytes it resolved to"
        );

        assert!(base.contains("nvrtc>=any"), "a unit with no floor says so in its key: {base}");
        assert!(base.contains("/lib/"), "and names the set it asked for: {base}");
        assert!(
            unit.cache_key_under("sm_90", DEVICE_HEADERS, floored).contains("nvrtc>=13.3"),
            "and a floored one says which"
        );
    }

    /// A floor orders by major and then minor, and is inclusive at the floor.
    #[test]
    fn a_floor_is_met_by_that_version_and_by_everything_after_it() {
        let needs = Toolchain::new(13, 3);
        assert!(needs.met_by(Toolchain::new(13, 3)), "the floor itself compiles it");
        assert!(needs.met_by(Toolchain::new(13, 4)));
        assert!(needs.met_by(Toolchain::new(14, 0)));
        assert!(!needs.met_by(Toolchain::new(13, 0)), "this box");
        assert!(!needs.met_by(Toolchain::new(12, 9)), "a bigger minor is not a bigger version");
        assert!(!needs.met_by(Toolchain::ANY), "and `any` is not a version at all");

        assert!(Toolchain::ANY.met_by(Toolchain::new(13, 0)), "no floor is met by anything");
        assert!(Toolchain::ANY.is_any());
        assert!(!Toolchain::new(13, 0).is_any());

        assert!(Toolchain::new(13, 0) < Toolchain::new(13, 3));
        assert!(Toolchain::new(9, 13) < Toolchain::new(13, 0));
        assert_eq!(Toolchain::new(13, 3).to_string(), "13.3");
        assert_eq!(Toolchain::ANY.to_string(), "any", "so a report reads as a sentence");
    }

    /// A unit states what it demands, and the table and the units agree.
    #[test]
    fn a_units_demand_is_exactly_what_the_table_says_about_it() {
        for unit in UNITS {
            let covered = DEMANDS.iter().any(|(key, _)| demand_covers(key, unit.name));
            if covered {
                assert_ne!(
                    unit.demands(),
                    Demands::DEFAULT,
                    "`{}` is covered by a key that states the default, which is a \
                     row saying nothing",
                    unit.name
                );
                continue;
            }
            assert_eq!(
                unit.demands(),
                Demands::DEFAULT,
                "`{}` states a demand no key covers",
                unit.name
            );
            assert!(unit.floor().is_any());
            assert_eq!(unit.headers(), Headers::Library);
            assert_eq!(unit.header_set(), DEVICE_HEADERS);
        }
    }

    /// The four demands that reach the upstream closure, and only those.
    #[test]
    fn the_upstream_closure_is_demanded_by_exactly_the_roots_that_reach_it() {
        for unit in UNITS {
            let wants_upstream = unit.name.starts_with("attn/fa2_")
                || unit.name == crate::families::cascade::MERGE_STATES.name
                || unit.name.starts_with("attn/attention_xqa_mha_")
                || unit.name == "attn/attention_mla_fa2";
            assert_eq!(
                unit.headers() == Headers::LibraryAndUpstream,
                wants_upstream,
                "`{}`",
                unit.name
            );
        }
    }

    /// A demand is resolved by name, and the resolution is the identity every
    #[test]
    fn a_demand_is_resolved_by_the_units_own_name() {
        let unit = crate::x::norm::altup_aux::ALTUP_AUX;
        assert_eq!(Unit { name: "norm/nowhere", ..unit }.demands(), Demands::DEFAULT);

        assert_eq!(Headers::Library.set(), DEVICE_HEADERS);
        assert_eq!(Headers::LibraryAndUpstream.set(), ALL_HEADERS);
        assert!(
            Headers::LibraryAndUpstream.set().len() > Headers::Library.set().len(),
            "the upstream closure is carried on top of the library, not instead of it"
        );
        assert_ne!(Headers::Library.tag(), Headers::LibraryAndUpstream.tag());
    }

    /// [`rows`] is the concatenation and nothing else — no unit skipped, no
    #[test]
    fn rows_is_every_units_rows_in_order() {
        let flattened: Vec<&str> = rows().map(|row| row.sig.symbol).collect();
        let expected: Vec<&str> = crate::families::ALL
            .iter()
            .flat_map(|family| family.iter())
            .flat_map(|unit| unit.rows.iter())
            .map(|row| row.sig.symbol)
            .collect();
        assert_eq!(flattened, expected);
        assert_eq!(flattened.len(), UNITS.iter().map(|unit| unit.rows.len()).sum::<usize>());
    }

    /// An instantiation per row, in the row order the lowered names come back
    #[test]
    fn the_instantiations_are_the_rows() {
        for unit in UNITS {
            let asked = unit.instantiations();
            assert_eq!(asked.len(), unit.rows.len());
            for (at, row) in unit.rows.iter().enumerate() {
                assert_eq!(asked[at], row.instantiation());
            }
        }
    }
}
