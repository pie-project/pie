pub const CONFIG_OBJECT: &str = "model/config";

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Encoding {
    pub method: String,

    pub bits: u32,

    pub group_size: u32,
}

impl Encoding {
    #[must_use]
    pub fn dense() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn is_none(&self) -> bool {
        self.method.is_empty()
    }

    #[must_use]
    pub fn is_mxfp4(&self) -> bool {
        self.method.eq_ignore_ascii_case("mxfp4")
    }

    #[cfg(feature = "contract")]
    pub fn from_config_json(text: &str) -> Result<Self, serde_json::Error> {
        let root: serde_json::Value = serde_json::from_str(text)?;
        Ok(Self::from_config_value(&root))
    }

    #[cfg(feature = "contract")]
    #[must_use]
    pub fn from_config_value(root: &serde_json::Value) -> Self {
        let text = root.get("text_config");
        let block = [
            text.and_then(|t| t.get("quantization_config")),
            root.get("quantization_config"),
            text.and_then(|t| t.get("quantization")),
            root.get("quantization"),
        ]
        .into_iter()
        .flatten()
        .find(|v| v.is_object());
        let Some(q) = block else {
            return Self::dense();
        };
        let u32_of = |key: &str| {
            q.get(key)
                .and_then(serde_json::Value::as_u64)
                .and_then(|n| u32::try_from(n).ok())
                .unwrap_or(0)
        };
        Self {
            method: q
                .get("quant_method")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string(),
            bits: u32_of("bits"),
            group_size: u32_of("group_size"),
        }
    }
}
