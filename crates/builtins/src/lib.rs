//! The built-in inferlets: what serves pie's OpenAI / Anthropic / Gemini
//! interface they were built against.
//!
//! `build.rs` does the compiling; this crate only holds the bytes.

use std::borrow::Cow;

#[derive(Clone, Copy, Debug)]
pub struct Builtin {
    pub name: &'static str,
    pub version: &'static str,
    pub component: &'static [u8],
}

include!(concat!(env!("OUT_DIR"), "/builtins.rs"));

pub fn all() -> &'static [Builtin] {
    ALL
}

pub fn find(name: &str) -> Option<&'static Builtin> {
    ALL.iter().find(|b| b.name == name)
}

pub fn policies() -> &'static [Builtin] {
    POLICIES
}

pub fn policy(name: &str) -> Option<&'static Builtin> {
    POLICIES.iter().find(|b| b.name == name)
}

pub fn policy_component(spec: &str) -> std::io::Result<Cow<'static, [u8]>> {
    if let Some(builtin) = policy(spec) {
        return Ok(Cow::Borrowed(builtin.component));
    }
    std::fs::read(spec).map(Cow::Owned).map_err(|e| {
        let why = format!("`{spec}` is neither a built-in policy nor a readable file: {e}");
        std::io::Error::new(e.kind(), why)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_builtin_is_a_named_component() {
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
        }
        assert_eq!(find("compat-openai").map(|b| b.name), Some("compat-openai"));
        assert!(find("nobody").is_none());
    }
}
