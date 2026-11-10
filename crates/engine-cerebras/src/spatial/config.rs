//! The `[cerebras.spatial]` table.

/// The `[cerebras.spatial]` table. An unknown key or a mistyped value is an
/// error, not a silently different engine.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpatialBoot {
    /// The compiled wafer program's output directory (its `artifact/`).
    pub program: String,
    /// The most steps a greedy lane's token loop runs past its row (0: off;
    /// loops start at 8 and double while a sequence uses them up).
    pub lookahead: Option<usize>,
    /// Whether frames settle asynchronously: each step lands on the
    /// completion sink once its epilogue ran. `false` (the default): each
    /// frame runs whole inside its submit.
    #[serde(rename = "async")]
    pub asynchronous: Option<bool>,
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
        t.clone()
            .try_into()
            .map(Some)
            .map_err(|e| format!("[cerebras.spatial]: {e}"))
    }
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
        let b = of("[cerebras.spatial]\nprogram = \"p\"\nlookahead = 8\n")
            .expect("parses")
            .expect("a table");
        assert_eq!(b.program, "p");
        assert_eq!(b.lookahead, Some(8));
        assert_eq!(b.asynchronous, None);
        let b = of("[cerebras.spatial]\nprogram = \"p\"\nasync = false\n")
            .expect("parses")
            .expect("a table");
        assert_eq!(b.asynchronous, Some(false));
    }

    #[test]
    fn unknown_keys_and_no_program_are_refused() {
        assert!(of("[cerebras.spatial]\nprogram = \"p\"\nserve = {}\n").is_err());
        assert!(of("[cerebras.spatial]\nlookahead = 8\n").is_err());
        assert!(of("[cerebras.spatial]\nprogram = \"p\"\nlookahead = \"8\"\n").is_err());
    }
}
