//! The built-in inferlets: what serves pie's OpenAI / Anthropic / Gemini
//! routes, compiled from `inferlets/*` (every one whose Pie.toml says
//! `tier = "builtin"`) and embedded, so a pie binary serves those routes
//! with nothing installed beside it and the inferlets always match the host
//! interface they were built against.
//!
//! `build.rs` does the compiling; this crate only holds the bytes.

/// One embedded inferlet: its manifest as TOML and its component.
#[derive(Clone, Copy, Debug)]
pub struct Builtin {
    pub name: &'static str,
    pub version: &'static str,
    pub manifest: &'static str,
    pub component: &'static [u8],
}

include!(concat!(env!("OUT_DIR"), "/builtins.rs"));

pub fn all() -> &'static [Builtin] {
    ALL
}

pub fn find(name: &str) -> Option<&'static Builtin> {
    ALL.iter().find(|b| b.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_builtin_is_a_named_component_with_a_builtin_manifest() {
        let mut names: Vec<&str> = ALL.iter().map(|b| b.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), ALL.len(), "duplicate names");
        for b in ALL {
            assert!(!b.version.is_empty(), "{}: no version", b.name);
            assert!(
                b.component.starts_with(b"\0asm"),
                "{}: component is not wasm",
                b.name
            );
            assert!(
                b.manifest.contains("tier = \"builtin\""),
                "{}: not builtin",
                b.name
            );
            assert!(
                b.manifest.contains(&format!("name = \"{}\"", b.name)),
                "{}: manifest names something else",
                b.name
            );
        }
        assert_eq!(find("compat-openai").map(|b| b.name), Some("compat-openai"));
        assert!(find("nobody").is_none());
    }
}
