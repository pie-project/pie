//! What a load of a backend that keeps recordings armed before ready: the
//! lattice shapes whose walks recorded a family, and the rows its allowance
//! stated eager from. A later load that finds it records only those shapes
//! and measures no allowance ([`crate::serve::Shell::record_bodies`]); the
//! lattice's other shapes fill the families those recorded at their first
//! fire, recording nothing.
//!
//! Each record is its own entry, keyed on what shapes the bodies: the
//! backend's version of the layout a recording's commands take and the code
//! of the kernels the load built ([`crate::serve::Serve::armed_build`]), the
//! device, the model's trace, the lattice, the rungs and the dense kernels
//! the load settled. A key that differs in any part names another
//! file, so a stale entry is measured again alone. A load reads the user's
//! record (`<cache>/vulkan-bodies/<key>`), else the build's committed one of
//! the same key (`pie tune export` copies the user's into the engine's
//! `profiles/bodies/`). Text:
//!
//! ```text
//! pie-armed 1
//! build <hash>
//! eager <rows, or ->
//! shapes <index>,<index>,...
//! ```

use std::path::{Path, PathBuf};

const HEADER: &str = "pie-armed 1";

/// What a load armed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Armed {
    /// The lattice's shapes, by their place in it, whose walks recorded a
    /// family.
    pub shapes: Vec<usize>,
    /// The first stated eager: past the bodies' allowance. A shape index
    /// where the load states its steps, else the fewest rows of a fire.
    pub eager: Option<u32>,
}

/// The record's file in `dir` for a load whose bodies `parts` shape under
/// build `build`.
#[must_use]
pub fn path(dir: &Path, parts: &str, build: u64) -> PathBuf {
    let mut key = build.to_le_bytes().to_vec();
    key.extend_from_slice(parts.as_bytes());
    dir.join(format!("{:016x}", hash(&key)))
}

/// FNV-1a over `bytes`: a key for files, not a defence.
fn hash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The record at `path`, else, where the home keeps no entry of the key,
/// the committed record of the same key (`committed`, by file name); `None`
/// where the one read is not whole and of `build` (a home's entry written
/// stale passes over the committed one: a load that armed from it left a
/// fire unarmed).
#[must_use]
pub fn read(path: &Path, build: u64, committed: &[(&str, &str)]) -> Option<Armed> {
    match std::fs::read_to_string(path) {
        Ok(text) => parse(&text, build),
        Err(_) => {
            let name = path.file_name()?.to_str()?;
            let (_, text) = committed.iter().find(|(key, _)| *key == name)?;
            parse(text, build)
        }
    }
}

fn parse(text: &str, build: u64) -> Option<Armed> {
    let mut lines = text.lines();
    if lines.next()? != HEADER {
        return None;
    }
    let built = lines.next()?.strip_prefix("build ")?;
    if u64::from_str_radix(built.trim(), 16).ok()? != build {
        return None;
    }
    let eager = lines.next()?.strip_prefix("eager ")?;
    let eager = match eager.trim() {
        "-" => None,
        rows => Some(rows.parse().ok()?),
    };
    let shapes = lines.next()?.strip_prefix("shapes")?.trim();
    let shapes = if shapes.is_empty() {
        Vec::new()
    } else {
        shapes
            .split(',')
            .map(|at| at.parse().ok())
            .collect::<Option<Vec<usize>>>()?
    };
    Some(Armed { shapes, eager })
}

fn render(armed: &Armed, build: u64) -> String {
    let shapes: Vec<String> = armed.shapes.iter().map(usize::to_string).collect();
    format!(
        "{HEADER}\nbuild {build:016x}\neager {}\nshapes {}\n",
        armed
            .eager
            .map_or_else(|| "-".to_string(), |rows| rows.to_string()),
        shapes.join(",")
    )
}

/// Keeps `armed` at `path` under `build`, written whole or not at all.
pub fn write(path: &Path, armed: &Armed, build: u64) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let partial = path.with_extension("partial");
    std::fs::write(&partial, render(armed, build))?;
    std::fs::rename(partial, path)
}

/// Copies every whole record in `from` into `to` (a backend's
/// `profiles/bodies/`), each under its key: what a build carries for a
/// clean home on a known device. The keys copied.
pub fn export(from: &Path, to: &Path) -> std::io::Result<Vec<String>> {
    let mut copied = Vec::new();
    let Ok(entries) = std::fs::read_dir(from) else {
        return Ok(copied);
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()).map(str::to_string) else {
            continue;
        };
        if path.extension().is_some() {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let build = text
            .lines()
            .nth(1)
            .and_then(|line| line.strip_prefix("build "))
            .and_then(|hex| u64::from_str_radix(hex.trim(), 16).ok());
        if build.and_then(|build| parse(&text, build)).is_none() {
            continue;
        }
        std::fs::create_dir_all(to)?;
        std::fs::write(to.join(&name), text)?;
        copied.push(name);
    }
    copied.sort();
    Ok(copied)
}

#[cfg(test)]
mod tests {
    use super::{Armed, parse, render};

    #[test]
    fn a_record_reads_back_as_written_and_only_under_its_build() {
        for armed in [
            Armed {
                shapes: vec![0, 3, 17],
                eager: Some(512),
            },
            Armed {
                shapes: Vec::new(),
                eager: None,
            },
        ] {
            assert_eq!(parse(&render(&armed, 7), 7), Some(armed.clone()));
            assert_eq!(parse(&render(&armed, 7), 8), None, "another build's record");
        }
        assert_eq!(parse("pie-armed 0\nbuild 7\neager -\nshapes 1\n", 7), None);
        assert_eq!(parse("pie-armed 1\nbuild 7\neager x\nshapes 1\n", 7), None, "torn");
    }

}
