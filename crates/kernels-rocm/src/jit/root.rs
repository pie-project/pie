use crate::source::{self, LIBRARY};

/// The options every unit compiles under, after `--offload-arch`. No
/// contraction of a*b+c into an fma: a kernel rounds as its reference does,
/// and a fused path says so in its own text with `fmaf`.
/// A unit is also a header other units include, so it carries `#pragma
/// once`, which clang flags in the file it compiles first.
pub const OPTIONS: [&str; 4] = [
    "-std=c++17",
    "-O3",
    "-ffp-contract=off",
    "-Wno-pragma-once-outside-header",
];

/// A unit hiprtc compiles: one `.hip` file under `kernels/`, with every
/// other carried file in scope as a header.
#[derive(Clone, Copy, Debug)]
pub struct Root {
    pub name: &'static str,
    pub text: &'static str,
    pub file: &'static str,
}

impl Root {
    #[must_use]
    pub fn of(file: &'static str) -> Option<Self> {
        let text = source::text_of(file)?;
        let name = file.strip_suffix(".hip")?;
        Some(Self { name, text, file })
    }

    /// The cache key of one instantiation of this unit for `arch`: it moves
    /// when the unit, any carried header, or the instantiation does.
    #[must_use]
    pub fn key(&self, instantiation: &str, arch: &str) -> String {
        format!(
            "jit/{}/{arch}/{}/r{:016x}/h{:016x}/i{:016x}",
            self.name,
            OPTIONS.join(","),
            source::fnv1a64(self.text.as_bytes()),
            source::digest(LIBRARY),
            source::fnv1a64(instantiation.as_bytes()),
        )
    }
}
