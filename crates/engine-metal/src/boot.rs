use crate::api::{ContractFor, DeviceBoot, Metal};

pub fn open(config_bytes: &[u8], contract_for: ContractFor) -> Result<Metal, String> {
    let doc: toml::Table = std::str::from_utf8(config_bytes)
        .map_err(|error| format!("the metal boot config is not utf-8: {error}"))?
        .parse()
        .map_err(|error| format!("the metal boot config is not TOML: {error}"))?;
    let profiles = profiles(&doc)?;
    crate::diag::publish(&diagnostics(&doc)?);
    Ok(Metal::new(
        DeviceBoot {
            adapter_dir: adapter_dir(&doc),
            profiles,
        },
        contract_for,
    ))
}

fn adapter_dir(doc: &toml::Table) -> Option<std::path::PathBuf> {
    doc.get("model")
        .and_then(toml::Value::as_table)
        .and_then(|model| model.get("adapter_dir"))
        .and_then(toml::Value::as_str)
        .map(str::trim)
        .filter(|dir| !dir.is_empty())
        .map(std::path::PathBuf::from)
}

fn diagnostics(doc: &toml::Table) -> Result<crate::diag::Diagnostics, String> {
    let Some(words) = doc
        .get("metal")
        .and_then(toml::Value::as_table)
        .and_then(|metal| metal.get("diagnostics"))
    else {
        return Ok(crate::diag::Diagnostics::default());
    };
    let words = words.as_str().ok_or_else(|| {
        format!("[metal] diagnostics is a comma-separated word list, not {words}")
    })?;
    words
        .parse::<crate::diag::Diagnostics>()
        .map_err(|error| format!("[metal] diagnostics: {error}"))
}

fn profiles(doc: &toml::Table) -> Result<Option<std::path::PathBuf>, String> {
    let Some(metal) = doc.get("metal").and_then(toml::Value::as_table) else {
        return Ok(None);
    };
    match metal.get("profiles") {
        None => Ok(None),
        Some(toml::Value::String(dir)) if dir.trim().is_empty() => Ok(None),
        Some(toml::Value::String(dir)) => Ok(Some(std::path::PathBuf::from(dir))),
        Some(other) => Err(format!("[metal] profiles is a directory path, not {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nothing(
        _trace: &poem_ir::Trace,
        _path: &std::path::Path,
    ) -> Result<checkpoint::contract::ModelContract, String> {
        Err("this door never loads".to_string())
    }

    #[test]
    fn boot_every_case() {
        a_boot_document_that_says_nothing_about_this_engine_still_opens();
        a_boot_document_that_is_not_toml_is_refused_at_the_door();
        the_diagnostics_word_list_is_read_or_refused_by_name();
        the_shared_adapter_directory_is_read_and_an_empty_one_is_off();
        the_profiles_directory_is_read_or_refused_by_name();
    }

    fn a_boot_document_that_says_nothing_about_this_engine_still_opens() {
        assert!(open(b"", nothing).is_ok());
        assert!(open(b"[model]\nid = \"qwen35-d0.8b\"\n", nothing).is_ok());
    }

    fn a_boot_document_that_is_not_toml_is_refused_at_the_door() {
        assert!(open(b"this is not = = toml", nothing).is_err());
    }

    fn the_diagnostics_word_list_is_read_or_refused_by_name() {
        let of = |src: &str| super::diagnostics(&src.parse::<toml::Table>().unwrap());
        assert_eq!(
            of("").expect("silence is no diagnostics"),
            crate::diag::Diagnostics::default()
        );
        let stated = of("[metal]\ndiagnostics = \"tier-trace,kernel-profile=2\"\n")
            .expect("two words this shell speaks");
        assert!(stated.tier_trace);
        assert_eq!(stated.kernel_profile, crate::diag::Profile::Shaped);

        let why = of("[metal]\ndiagnostics = \"teir-trace\"\n").expect_err("a misspelling");
        assert!(why.contains("teir-trace"), "the refusal names it: {why}");
        let why = of("[metal]\ndiagnostics = 3\n").expect_err("a number is not a word list");
        assert!(why.contains("word list"), "{why}");
        assert!(open(b"[metal]\ndiagnostics = \"teir-trace\"\n", nothing).is_err());
    }

    fn the_shared_adapter_directory_is_read_and_an_empty_one_is_off() {
        let read = |text: &str| adapter_dir(&text.parse::<toml::Table>().expect("valid TOML"));
        assert_eq!(
            read("[model]\nadapter_dir = \"/srv/pie/shared\""),
            Some(std::path::PathBuf::from("/srv/pie/shared"))
        );
        assert_eq!(read("[model]\nadapter_dir = \"  \""), None, "empty is off");
        assert_eq!(read("[model]"), None, "and so is absent");
        assert_eq!(read(""), None);
    }

    fn the_profiles_directory_is_read_or_refused_by_name() {
        let of = |src: &str| super::profiles(&src.parse::<toml::Table>().unwrap());
        assert_eq!(of(""), Ok(None));
        assert_eq!(
            of("[metal]\nprofiles = \"/home/p/.pie/profiles\"\n"),
            Ok(Some("/home/p/.pie/profiles".into()))
        );
        assert!(of("[metal]\nprofiles = 3\n").is_err());
    }
}
