//! What a program is called: the hash of its bytes, which is the program,
//! and the name it was installed under, which only finds it.

use std::sync::LazyLock;

use anyhow::{Context, Result, anyhow, bail};

use super::language::Language;

/// The blake3 hash of a program's bytes, as 64 hex digits. Two uploads of
/// the same bytes are one program, wherever and by whomever they came.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProgramHash(String);

impl ProgramHash {
    /// The digits a name shows it under: enough to tell builds apart, and a
    /// prefix [`ProgramRef`] accepts.
    pub const SHORT: usize = 12;

    pub fn of(bytes: &[u8]) -> Self {
        Self(blake3::hash(bytes).to_hex().to_string())
    }

    /// 64 lowercase hex digits, as [`ProgramHash::as_str`] gives them.
    pub fn parse(hex: &str) -> Option<Self> {
        (hex.len() == 64 && is_hex(hex)).then(|| Self(hex.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn short(&self) -> &str {
        &self.0[..Self::SHORT]
    }
}

impl std::fmt::Display for ProgramHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A program as one launch runs it: its bytes, by hash, and the name it was
/// found under. Displays as `name@<short hash>`, which launches it again.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Program {
    pub name: String,
    pub hash: ProgramHash,
}

impl Program {
    /// `name@<full hash>`: what an install answers, naming this build for
    /// good.
    pub fn pinned(&self) -> String {
        format!("{}@{}", self.name, self.hash)
    }
}

impl std::fmt::Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", self.name, self.hash.short())
    }
}

/// How a client names a program to run: `name`, whatever that name last
/// installed, or `name@<hash prefix>`, that one build.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgramRef {
    pub name: String,
    pub hash: Option<String>,
}

impl ProgramRef {
    /// The fewest hex digits a reference may pin a build by.
    pub const MIN_PREFIX: usize = 8;

    pub fn parse(s: &str) -> Result<Self> {
        let (name, hash) = match s.split_once('@') {
            Some((name, hash)) => (name, Some(hash)),
            None => (s, None),
        };
        check_name(name).with_context(|| format!("{s:?} names no program"))?;
        if let Some(hash) = hash
            && (!(Self::MIN_PREFIX..=64).contains(&hash.len()) || !is_hex(hash))
        {
            bail!(
                "{s:?} names no program: after `@` comes the build's hash, {} to 64 \
                 lowercase hex digits",
                Self::MIN_PREFIX
            );
        }
        Ok(Self {
            name: name.to_string(),
            hash: hash.map(str::to_string),
        })
    }

    pub fn matches(&self, program: &Program) -> bool {
        self.name == program.name
            && self
                .hash
                .as_ref()
                .is_none_or(|prefix| program.hash.as_str().starts_with(prefix.as_str()))
    }
}

impl std::fmt::Display for ProgramRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.hash {
            Some(hash) => write!(f, "{}@{hash}", self.name),
            None => f.write_str(&self.name),
        }
    }
}

/// Whose names a name is among. The system's (the operator's installs, and
/// the binary's built-ins) every tenant reads; a tenant's, only it.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Namespace {
    System,
    Tenant(String),
}

impl std::fmt::Display for Namespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::System => f.write_str("system"),
            Self::Tenant(tenant) => write!(f, "tenant {tenant}"),
        }
    }
}

/// Lowercase hex digits, as a hash is written.
fn is_hex(s: &str) -> bool {
    s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// A program name: a letter or digit, then letters, digits, `-` and `_`.
pub fn check_name(name: &str) -> Result<()> {
    static RE: LazyLock<fancy_regex::Regex> =
        LazyLock::new(|| fancy_regex::Regex::new(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$").unwrap());
    if RE.is_match(name).unwrap_or(false) {
        Ok(())
    } else {
        Err(anyhow!(
            "{name:?} is not a program name: a letter or digit, then letters, digits, `-` and `_`"
        ))
    }
}

/// The custom section `#[inferlet::main]` writes the package's name into.
pub const NAME_SECTION: &str = "pie.name";

/// What an upload is, from its bytes and the file they came as.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    pub program: Program,
    pub language: Option<Language>,
}

/// A `.wasm` file is a component, named by the package it was built from
/// (its [`NAME_SECTION`]) or else its file; a `.py` or `.js` file is a
/// script in that language, named by its file. `_` in a file's name reads
/// as `-`.
pub fn identify(file: &str, bytes: &[u8]) -> Result<Identity> {
    let path = std::path::Path::new(file);
    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let language = match extension {
        "wasm" => None,
        other => Some(Language::from_extension(other).ok_or_else(|| {
            anyhow!(
                "{file}: not a program pie can install; a program is a `.wasm` component or a \
                 `.py` / `.js` script"
            )
        })?),
    };
    let name = match language.is_none().then(|| name_section(bytes)).flatten() {
        Some(name) => name,
        None => path
            .file_stem()
            .and_then(|s| s.to_str())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| anyhow!("{file}: no file name to name the program by"))?
            .replace('_', "-"),
    };
    check_name(&name).with_context(|| format!("{file} cannot name a program"))?;
    Ok(Identity {
        program: Program {
            name,
            hash: ProgramHash::of(bytes),
        },
        language,
    })
}

pub fn name_section(bytes: &[u8]) -> Option<String> {
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        let Ok(wasmparser::Payload::CustomSection(section)) = payload else {
            continue;
        };
        if section.name() == NAME_SECTION {
            return std::str::from_utf8(section.data())
                .ok()
                .map(|s| s.trim().to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn component_named(name: &str) -> Vec<u8> {
        let section = NAME_SECTION.as_bytes();
        let mut payload = vec![section.len() as u8];
        payload.extend_from_slice(section);
        payload.extend_from_slice(name.as_bytes());
        let mut bytes = b"\0asm\x01\0\0\0".to_vec();
        bytes.push(0);
        bytes.push(payload.len() as u8);
        bytes.extend_from_slice(&payload);
        bytes
    }

    #[test]
    fn identity_every_case() {
        a_script_is_named_by_its_file_and_keyed_by_its_bytes();
        a_component_is_named_by_the_package_it_carries();
        a_reference_names_a_program_or_pins_one_build();
    }

    fn a_script_is_named_by_its_file_and_keyed_by_its_bytes() {
        let source = b"async def main(input): return input\n";
        let identity = identify("beam_search.py", source).unwrap();
        assert_eq!(identity.program.name, "beam-search");
        assert_eq!(identity.program.hash, ProgramHash::of(source));
        assert_eq!(identity.language, Some(Language::Python));
        let edited = identify("beam_search.py", b"async def main(i): return i\n").unwrap();
        assert_ne!(edited.program.hash, identity.program.hash);
        assert!(identify("notes.txt", source).is_err());
    }

    fn a_component_is_named_by_the_package_it_carries() {
        let bytes = component_named("text-completion");
        let identity = identify("whatever.wasm", &bytes).unwrap();
        assert_eq!(identity.program.name, "text-completion");
        assert_eq!(identity.language, None);
        let bare = identify("probe_x.wasm", b"\0asm\x01\0\0\0").unwrap();
        assert_eq!(bare.program.name, "probe-x");
    }

    fn a_reference_names_a_program_or_pins_one_build() {
        let program = Program {
            name: "pico".to_string(),
            hash: ProgramHash::of(b"pico"),
        };
        let shown = program.to_string();
        assert_eq!(shown.len(), "pico@".len() + ProgramHash::SHORT);
        for spec in ["pico", shown.as_str(), program.pinned().as_str()] {
            assert!(ProgramRef::parse(spec).unwrap().matches(&program), "{spec}");
        }
        assert!(
            !ProgramRef::parse("pico@00000000")
                .unwrap()
                .matches(&program)
        );
        assert!(!ProgramRef::parse("pica").unwrap().matches(&program));
        for bad in ["pico@0.1.0", "pico@abc", "pico@ABCDEF12", "-x", ""] {
            assert!(ProgramRef::parse(bad).is_err(), "{bad}");
        }
    }
}
