#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Capability {
    Baseline,
    Fp16,
    Coopmat,
}

impl Capability {
    pub const PREFERENCE: [Self; 3] = [Self::Coopmat, Self::Fp16, Self::Baseline];

    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Fp16 => "fp16",
            Self::Coopmat => "coopmat",
        }
    }

    #[must_use]
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "baseline" => Some(Self::Baseline),
            "fp16" => Some(Self::Fp16),
            "coopmat" => Some(Self::Coopmat),
            _ => None,
        }
    }

    #[must_use]
    pub fn module(self, entrypoint: &str) -> String {
        match self {
            Self::Baseline => format!("{entrypoint}.spv"),
            other => format!("{entrypoint}.{}.spv", other.tag()),
        }
    }

    #[must_use]
    pub const fn requires(self) -> &'static [&'static str] {
        match self {
            Self::Baseline => &[],
            Self::Fp16 => &["shaderFloat16"],
            Self::Coopmat => &[
                "cooperativeMatrix",
                "shaderFloat16",
                "vulkanMemoryModel",
                "vulkanMemoryModelDeviceScope",
            ],
        }
    }
}
