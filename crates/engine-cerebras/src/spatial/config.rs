//! The `[cerebras.spatial]` table.

/// The `[cerebras.spatial]` table. An unknown key or a mistyped value is an
/// error, not a silently different engine.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpatialBoot {
    /// An already compiled program's output directory.
    pub program: Option<String>,
    /// Who runs translatable epilogues: `"off"`, `"check"` or `"wafer"`
    /// (`Spatial::guest_mode`).
    pub guest: Option<String>,
    /// The most steps a greedy lane's token loop runs past its row (0: off;
    /// loops start at 8 and double while a sequence uses them up).
    pub lookahead: Option<usize>,
    /// A Hugging Face directory to read the weights from instead of the
    /// load's checkpoint.
    pub ckpt: Option<String>,
    /// The guest placement's forecast ([`super::guest4::cost::Placement`]): the
    /// model's token in cycles (default: from its parameter count), how
    /// much of it a table's `pre` may take and `main` may add.
    pub token_cycles: Option<u64>,
    #[serde(default, deserialize_with = "number")]
    pub pre_margin: Option<f64>,
    #[serde(default, deserialize_with = "number")]
    pub main_fraction: Option<f64>,
    /// `[cerebras.spatial.serve]`: with no `program`, PIE builds the plan's program for the model itself
    /// ([`super::build::ServeConfig::from_toml`]; compiled once, then cached).
    pub serve: Option<toml::Table>,
}

impl SpatialBoot {
    /// The table, if the config has one: `None` selects the cerebras engine.
    pub fn of(doc: &toml::Table) -> Result<Option<SpatialBoot>, String> {
        let Some(t) = doc
            .get("cerebras")
            .and_then(toml::Value::as_table)
            .and_then(|c| c.get("spatial"))
        else {
            return Ok(None);
        };
        let boot: SpatialBoot = t
            .clone()
            .try_into()
            .map_err(|e| format!("[cerebras.spatial]: {e}"))?;
        if boot.program.is_none() && boot.serve.is_none() {
            return Err("[cerebras.spatial] names no wafer program (`program` or `serve`)".into());
        }
        Ok(Some(boot))
    }
}

/// A float that may be written as an integer (`pre_margin = 1`).
use serde::Deserialize;

fn number<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<f64>, D::Error> {
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum Number {
        Float(f64),
        Int(i64),
    }
    Ok(Option::<Number>::deserialize(d)?.map(|n| match n {
        Number::Float(f) => f,
        Number::Int(i) => i as f64,
    }))
}

#[cfg(test)]
mod tests {
    use super::SpatialBoot;

    fn of(text: &str) -> Result<Option<SpatialBoot>, String> {
        SpatialBoot::of(&text.parse().expect("toml"))
    }

    #[test]
    fn no_table_is_the_cerebras_engine() {
        assert!(of("[cerebras]\n").expect("parses").is_none());
    }

    #[test]
    fn the_table_reads_its_keys() {
        let b = of("[cerebras.spatial]\nprogram = \"p\"\nlookahead = 8\npre_margin = 1\nmain_fraction = 0.5\n")
            .expect("parses")
            .expect("a table");
        assert_eq!(b.program.as_deref(), Some("p"));
        assert_eq!(b.lookahead, Some(8));
        assert_eq!((b.pre_margin, b.main_fraction), (Some(1.0), Some(0.5)));
    }

    #[test]
    fn unknown_keys_and_no_program_are_refused() {
        assert!(of("[cerebras.spatial]\nprogram = \"p\"\nparams = \"x\"\n").is_err());
        assert!(of("[cerebras.spatial]\nlookahead = \"8\"\n").is_err());
        assert!(of("[cerebras.spatial]\nguest = \"off\"\n").is_err());
    }
}
