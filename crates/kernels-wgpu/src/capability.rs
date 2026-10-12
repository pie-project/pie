#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum Capability {
    #[default]
    Baseline,
    Subgroup,
}

impl Capability {
    pub const PREFERENCE: [Self; 2] = [Self::Subgroup, Self::Baseline];

    pub const ALL: [Self; 2] = [Self::Baseline, Self::Subgroup];

    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Subgroup => "subgroup",
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
            Self::Subgroup => &["SUBGROUP"],
        }
    }

    #[must_use]
    pub const fn defines(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Self::Baseline => &[],
            Self::Subgroup => &[("PIE_SUBGROUP", "1")],
        }
    }
}
