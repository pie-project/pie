#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Capability {
    Baseline,
    Fp16,
    Subgroup,
    /// F16xF16->F32 cooperative matrix, which on Metal is
    /// `simdgroup_matrix<T, 8, 8>` and in wgpu is
    /// `EXPERIMENTAL_COOPERATIVE_MATRIX` behind an `unsafe` token.
    ///
    /// ABOVE `Subgroup` DELIBERATELY, and it is the only tier whose feature
    /// carries a "there may be UB-containing bugs in these apis" contract. It
    /// is still a tier and not a switch, which is what makes that acceptable:
    /// a tier is a promise about a BODY with a baseline fallback beneath it,
    /// so a symbol with no `@matrix` instantiation is unaffected and a core
    /// WebGPU adapter never reaches this arm at all. `tests/cooperative.rs`
    /// measured it at 2.4x on an M4 Pro's projections shape with every
    /// spot-checked output bit-exact against an f32 CPU dot.
    ///
    /// It requires `SHADER_F16` as well, because the operand type is f16 and
    /// a module that declares a cooperative matrix must `enable f16`.
    Matrix,
}

impl Capability {
    pub const PREFERENCE: [Self; 4] = [Self::Matrix, Self::Subgroup, Self::Fp16, Self::Baseline];

    pub const ALL: [Self; 4] = [Self::Baseline, Self::Fp16, Self::Subgroup, Self::Matrix];

    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Fp16 => "fp16",
            Self::Subgroup => "subgroup",
            Self::Matrix => "matrix",
        }
    }

    #[must_use]
    pub fn from_tag(tag: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.tag() == tag)
    }

    #[must_use]
    pub fn variant(self, entrypoint: &str) -> String {
        match self {
            Self::Baseline => entrypoint.to_owned(),
            other => format!("{entrypoint}.{}", other.tag()),
        }
    }

    #[must_use]
    pub const fn requires(self) -> &'static [&'static str] {
        match self {
            Self::Baseline => &[],
            Self::Fp16 => &["SHADER_F16"],
            Self::Subgroup => &["SUBGROUP"],
            Self::Matrix => &["EXPERIMENTAL_COOPERATIVE_MATRIX", "SHADER_F16"],
        }
    }

    #[must_use]
    pub const fn defines(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Self::Baseline => &[],
            Self::Fp16 => &[("PIE_FP16", "1")],
            Self::Subgroup => &[("PIE_SUBGROUP", "1")],
            Self::Matrix => &[("PIE_MATRIX", "1")],
        }
    }
}
