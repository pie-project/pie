use crate::star::package::Package;

pub const ADOPT: &str = "adopt";

const ADOPTION: &str = "ADOPTION";

pub struct Adoption {
    pub name: String,
    pub files: Vec<(String, String)>,
    pub rebinds: Vec<String>,
    pub block: String,
    pub package: Package,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    pub family: String,
    pub reason: String,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "`{}` refuses it: {}", self.family, self.reason)
    }
}

impl std::error::Error for Refusal {}

impl std::fmt::Debug for Adoption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Adoption")
            .field("name", &self.name)
            .field("rebinds", &self.rebinds)
            .finish_non_exhaustive()
    }
}

impl Package {
    #[must_use]
    pub fn adopts(&self) -> bool {
        self.defining_file(ADOPT).ok().flatten().is_some() && !self.is_adoption()
    }

    #[must_use]
    pub fn is_adoption(&self) -> bool {
        self.defining_file(ADOPTION).ok().flatten().is_some()
    }

    pub fn adopt(
        &self,
        name: &str,
        id: &str,
        config: &serde_json::Value,
        tokenizer: &serde_json::Value,
        origin: &str,
    ) -> Result<Adoption, Refusal> {
        if self.is_adoption() {
            return Err(self.refusal("it is itself an adoption; adopt from its family"));
        }
        let file = self
            .defining_file(ADOPT)
            .map_err(|why| self.refusal(format!("{why:#}")))?
            .ok_or_else(|| {
                self.refusal(format!(
                    "it states no `{ADOPT}` and lists its models by hand"
                ))
            })?;
        if !valid_name(name) {
            return Err(self.refusal(format!(
                "`{name}` is no package name: lowercase letters, digits and `_`"
            )));
        }
        let config = &lifted(config);
        let call = format!(
            "{ADOPTION} = {ADOPT}({}, {}, {})\n",
            literal(&serde_json::Value::String(id.to_string()), 0),
            literal(config, 0),
            literal(tokenizer, 1)
        );
        let probe = self.with_appended(self.name(), &file, &file, &call)?;
        let adopted = probe
            .entry(ADOPTION)
            .map_err(|why| self.refusal(format!("{why:#}")))?
            .ok_or_else(|| self.refusal(format!("its `{ADOPT}` bound nothing")))?;
        let rebinds: Vec<String> = adopted
            .value()
            .dir_attr()
            .into_iter()
            .filter(|field| !field.starts_with('_'))
            .collect();
        if rebinds.is_empty() {
            return Err(self.refusal(format!(
                "its `{ADOPT}` returned {}, not a struct of the globals it rebinds",
                adopted.value().get_type()
            )));
        }
        if let Some(stranger) = rebinds.iter().find(|field| !self.binds(&file, field)) {
            return Err(self.refusal(format!(
                "its `{ADOPT}` rebinds `{stranger}`, which `{file}` does not bind"
            )));
        }
        let mut block = format!(
            "\n# ---- Adopted by `pie model adopt` from {origin} ----\n\
             # The family's text above is as it shipped; its `{ADOPT}` read the\n\
             # checkpoint's configuration below, and what it returned rebinds\n\
             # {}.\n",
            rebinds.join(", ")
        );
        block.push_str(&call);
        for field in &rebinds {
            block.push_str(&format!("{field} = {ADOPTION}.{field}\n"));
        }
        if self.binds(&file, "PUBLISHED") && !rebinds.iter().any(|f| f == "PUBLISHED") {
            block.push_str("PUBLISHED = []\n");
        }
        let own: Vec<(String, String)> = self
            .files()
            .filter(|(f, _)| !f.starts_with("//"))
            .map(|(f, s)| (f.to_string(), s.to_string()))
            .collect();
        let single = own.len() == 1;
        let renamed = |f: &str| {
            if single {
                format!("{name}.poem")
            } else {
                f.to_string()
            }
        };
        let package = self.with_appended(name, &file, &renamed(&file), &block)?;
        let header = format!(
            "# `{name}`: the model `{id}`, adopted by `pie model adopt` from {origin}\n\
             # onto the `{}` package (digest {}). Delete this file to forget it.\n\n",
            self.name(),
            &self.digest()[..12]
        );
        let files = own
            .iter()
            .map(|(f, text)| {
                let text = if *f == file {
                    format!("{header}{text}{block}")
                } else {
                    text.clone()
                };
                (renamed(f), text)
            })
            .collect();
        Ok(Adoption {
            name: name.to_string(),
            files,
            rebinds,
            block,
            package,
        })
    }

    fn with_appended(
        &self,
        name: &str,
        file: &str,
        as_file: &str,
        block: &str,
    ) -> Result<Package, Refusal> {
        let files: Vec<(String, String)> = self
            .files()
            .map(|(f, text)| {
                if f == file {
                    (as_file.to_string(), format!("{text}\n{block}"))
                } else {
                    (f.to_string(), text.to_string())
                }
            })
            .collect();
        let files: Vec<(&str, &str)> = files
            .iter()
            .map(|(f, s)| (f.as_str(), s.as_str()))
            .collect();
        Package::new(name, &files).map_err(|why| self.refusal(failed(&why)))
    }

    fn refusal(&self, reason: impl Into<String>) -> Refusal {
        Refusal {
            family: self.name().to_string(),
            reason: reason.into(),
        }
    }
}

/// `config` with transformers 5's `rope_parameters` also stated where a hook
/// reads it, at the top and in a `text_config`: its `rope_theta` and
/// `partial_rotary_factor` as themselves, and a scaling other than the
/// default as `rope_scaling`. A key the configuration states is kept.
fn lifted(config: &serde_json::Value) -> serde_json::Value {
    let mut config = config.clone();
    lift(&mut config);
    if let Some(text) = config.get_mut("text_config") {
        lift(text);
    }
    config
}

fn lift(config: &mut serde_json::Value) {
    let Some(map) = config.as_object_mut() else {
        return;
    };
    let Some(params) = map
        .get("rope_parameters")
        .and_then(serde_json::Value::as_object)
        .cloned()
    else {
        return;
    };
    for key in ["rope_theta", "partial_rotary_factor"] {
        if let (false, Some(value)) = (map.contains_key(key), params.get(key)) {
            map.insert(key.to_string(), value.clone());
        }
    }
    let kind = params
        .get("rope_type")
        .or_else(|| params.get("type"))
        .and_then(serde_json::Value::as_str);
    if map
        .get("rope_scaling")
        .is_none_or(serde_json::Value::is_null)
        && kind.is_some_and(|kind| kind != "default")
    {
        let mut scaling = params;
        scaling.remove("rope_theta");
        scaling.remove("partial_rotary_factor");
        map.insert(
            "rope_scaling".to_string(),
            serde_json::Value::Object(scaling),
        );
    }
}

fn failed(why: &anyhow::Error) -> String {
    let said = format!("{why:#}");
    said.lines()
        .find_map(|line| line.trim().strip_prefix("error: fail: "))
        .or_else(|| {
            said.lines()
                .find_map(|line| line.split_once("fail: ").map(|(_, why)| why))
        })
        .map_or_else(|| said.clone(), str::to_string)
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name != "lib"
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        && !name.starts_with(|c: char| c.is_ascii_digit())
}

fn literal(value: &serde_json::Value, depth: usize) -> String {
    use serde_json::Value;
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => quoted(s),
        Value::Array(items) => format!(
            "[{}]",
            items
                .iter()
                .map(|v| literal(v, depth + 1))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(map) if map.is_empty() => "{}".to_string(),
        Value::Object(map) if depth > 0 => format!(
            "{{{}}}",
            map.iter()
                .map(|(k, v)| format!("{}: {}", quoted(k), literal(v, depth + 1)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(map) => {
            let mut out = "{\n".to_string();
            for (k, v) in map {
                out.push_str(&format!("    {}: {},\n", quoted(k), literal(v, depth + 1)));
            }
            out.push('}');
            out
        }
    }
}

fn quoted(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const FAMILY: &str = r#"
MODELS = [model("a-1", template = template("raw"), tokenizer = tokenizer(), arch = "a", layers = 2, vocab = 8)]
DEPLOYMENTS = [deployment("a-1", weights = dtype.bf16, kv = dtype.bf16)]
DIMS = {"a-1": struct(layers = 2)}

def adopt(id, config, special):
    if config.get("rope_scaling") != None:
        fail("this family turns no scaled rope")
    return struct(
        MODELS = [model(id, template = template("raw"), tokenizer = tokenizer(), arch = "a", layers = config["num_hidden_layers"], vocab = config["vocab_size"])],
        DEPLOYMENTS = [deployment(id, weights = dtype.bf16, kv = dtype.bf16)],
        DIMS = {id: struct(layers = config["num_hidden_layers"])},
    )
"#;

    fn family() -> Package {
        Package::new("fam", &[("fam.poem", FAMILY)]).expect("the family loads")
    }

    #[test]
    fn an_adoption_rebinds_what_the_hook_returns() {
        let config =
            serde_json::json!({"num_hidden_layers": 6, "vocab_size": 32, "rope_scaling": null});
        let adopted = family()
            .adopt("fam_b", "b-1", &config, &serde_json::json!({}), "org/b-1")
            .expect("the family adopts it");
        assert_eq!(adopted.rebinds, ["DEPLOYMENTS", "DIMS", "MODELS"]);
        let ids: Vec<&str> = adopted
            .package
            .manifest()
            .models
            .iter()
            .map(|m| m.id.as_str())
            .collect();
        assert_eq!(ids, ["b-1"]);
        assert_eq!(adopted.package.manifest().models[0].layers, 6);
        assert_eq!(adopted.files.len(), 1);
        assert_eq!(adopted.files[0].0, "fam_b.poem");
        assert!(adopted.files[0].1.contains("\"num_hidden_layers\": 6,"));
        let reloaded = Package::new("fam_b", &[("fam_b.poem", adopted.files[0].1.as_str())])
            .expect("the written package loads as written");
        assert_eq!(reloaded.manifest().models[0].id, "b-1");
    }

    #[test]
    fn a_hook_that_fails_refuses_by_its_reason() {
        let config = serde_json::json!({"num_hidden_layers": 6, "vocab_size": 32, "rope_scaling": {"type": "yarn"}});
        let why = family()
            .adopt("fam_b", "b-1", &config, &serde_json::json!({}), "org/b-1")
            .expect_err("the hook refuses it")
            .to_string();
        assert!(why.contains("this family turns no scaled rope"), "{why}");
    }

    #[test]
    fn a_family_listing_its_models_by_hand_adopts_nothing() {
        let by_hand = Package::new("hand", &[("hand.poem", "MODELS = []\nDEPLOYMENTS = []\n")])
            .expect("it loads");
        assert!(!by_hand.adopts());
        assert!(
            by_hand
                .adopt(
                    "x",
                    "x",
                    &serde_json::json!({}),
                    &serde_json::json!({}),
                    "x"
                )
                .is_err()
        );
    }

    #[test]
    fn a_transformers_5_configuration_states_its_rope_where_a_hook_reads_it() {
        let config = serde_json::json!({
            "rope_parameters": {"rope_type": "llama3", "rope_theta": 500000.0, "factor": 8.0},
            "text_config": {"rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0}},
        });
        let lifted = lifted(&config);
        assert_eq!(lifted["rope_theta"], 500000.0);
        assert_eq!(lifted["rope_scaling"]["rope_type"], "llama3");
        assert_eq!(lifted["rope_scaling"]["factor"], 8.0);
        assert!(lifted["rope_scaling"].get("rope_theta").is_none());
        assert_eq!(lifted["text_config"]["rope_theta"], 1000000.0);
        assert!(lifted["text_config"].get("rope_scaling").is_none());
        let stated =
            serde_json::json!({"rope_theta": 10000.0, "rope_parameters": {"rope_theta": 1.0}});
        assert_eq!(lifted_theta(&stated), 10000.0);
    }

    fn lifted_theta(config: &serde_json::Value) -> f64 {
        lifted(config)["rope_theta"].as_f64().expect("a theta")
    }

    #[test]
    fn a_configuration_is_written_as_starlark() {
        let config = serde_json::json!({"a": [1, 2.5, true, null], "b": {"c": "q\"x"}});
        assert_eq!(
            literal(&config, 0),
            "{\n    \"a\": [1, 2.5, True, None],\n    \"b\": {\"c\": \"q\\\"x\"},\n}"
        );
    }
}
