#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum Capability {
    #[default]
    Baseline,
    Fp16,
    Subgroup,
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
