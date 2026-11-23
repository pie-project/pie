//! What a deployment's graph bodies took the last time it armed whole,
//! kept beside the kernel cache: a load that finds it holds that for its
//! bodies, not everything past one working set, and arms its lattice as
//! recorded, without the estimates a load that knows nothing prices by.
//!
//! The record is keyed on everything that shapes the bodies: the engine
//! build (its crate version and the binary's own size and time), the
//! deployment's checkpoint, the trace, the device profile and the budgets.
//! A key that differs in any part names another file, so a record never
//! reads across builds or deployments. The file is
//! `<cache_dir>/bodies/<key>`, text:
//!
//! ```text
//! pie-bodies 3
//! bytes <census bytes once every rung armed>
//! bodies <bodies armed>
//! captured <what device memory fell by across each composition's capture, summed>
//! ```

use std::path::{Path, PathBuf};

/// What the bodies took when every rung armed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Measured {
    /// The bodies' census.
    pub bytes: u64,
    pub bodies: u64,
    /// What device memory fell by across each composition's capture,
    /// summed (each clamped at zero): the bodies and what their captures
    /// keep beside them, never under the census.
    pub captured: u64,
}

impl Measured {
    /// What a load holds for its bodies: the recorded figure, exactly. It is
    /// a measurement of the same build and deployment, so no margin.
    #[must_use]
    pub fn hold(&self) -> u64 {
        self.captured
    }
}

const HEADER: &str = "pie-bodies 4";

/// The record's file for a deployment whose bodies `parts` shape, under
/// this build ([`build`]): what the bodies hold moves with the engine's code
/// as much as with its kernels', so the code they were built from keys it.
#[must_use]
pub(crate) fn path(cache_dir: &Path, parts: &str) -> PathBuf {
    let mut key = build().to_le_bytes().to_vec();
    key.extend_from_slice(parts.as_bytes());
    cache_dir
        .join("bodies")
        .join(format!("{:016x}", content_hash(&key)))
}

/// The code that shapes what the bodies hold, as this build was compiled
/// from it (`build.rs`: this crate's sources and its kernel crate's): a pie
/// version that changes none of it keeps every record.
fn build() -> u64 {
    u64::from_str_radix(env!("PIE_BODIES_BUILD"), 16).expect("build.rs writes 16 hex digits")
}

/// What names a checkpoint in a record's key: its artifact's own identity
/// (each object's and shard's digest, from its manifest, a few KB read),
/// so a copied or moved file keeps its records; a file with none is named
/// by its path and size.
pub(crate) fn checkpoint_identity(path: &Path) -> String {
    match checkpoint::file::zt::artifact_identity(path) {
        Ok(Some(identity)) => format!("zt:{:016x}", content_hash(&identity)),
        _ => format!(
            "{}:{:?}",
            path.display(),
            std::fs::metadata(path).map(|meta| meta.len()).ok()
        ),
    }
}

/// A fast 64-bit hash of `bytes`, a word at a time (a multiply and a
/// rotate a word), the length folded in: a key for files, not a defence.
fn content_hash(bytes: &[u8]) -> u64 {
    const K: u64 = 0x517c_c1b7_2722_0a95;
    let mut hash = bytes.len() as u64;
    let mut words = bytes.chunks_exact(8);
    for word in &mut words {
        let word = u64::from_le_bytes(word.try_into().expect("an 8-byte chunk"));
        hash = (hash.rotate_left(5) ^ word).wrapping_mul(K);
    }
    for &byte in words.remainder() {
        hash = (hash.rotate_left(5) ^ u64::from(byte)).wrapping_mul(K);
    }
    hash
}

/// The record at `path`, else the committed record of the same key
/// ([`COMMITTED`]: a measured profile's bodies, carried by the build); none
/// where neither is there and whole.
#[must_use]
pub(crate) fn read(path: &Path) -> Option<Measured> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| parse(&text))
        .or_else(|| committed(path))
}

/// The committed record keyed as `path` is (its file name), where the build
/// carries one.
fn committed(path: &Path) -> Option<Measured> {
    let name = path.file_name()?.to_str()?;
    let (_, text) = COMMITTED.iter().find(|(key, _)| *key == name)?;
    parse(text)
}

// The committed records (`profiles/bodies/<key>`), by key (`build.rs`).
include!(concat!(env!("OUT_DIR"), "/committed_bodies.rs"));

/// A record's text, read where it is whole and of this build.
fn parse(text: &str) -> Option<Measured> {
    let mut lines = text.lines();
    if lines.next()? != HEADER {
        return None;
    }
    let built = lines.next()?.strip_prefix("build ")?;
    if u64::from_str_radix(built.trim(), 16).ok()? != build() {
        return None;
    }
    let mut field =
        |name: &str| -> Option<u64> { lines.next()?.strip_prefix(name)?.trim().parse().ok() };
    let bytes = field("bytes ")?;
    let bodies = field("bodies ")?;
    let captured = field("captured ")?;
    (bodies > 0 && captured >= bytes).then_some(Measured {
        bytes,
        bodies,
        captured,
    })
}

/// Writes `measured` at `path`, through a file beside it so a reader never
/// sees half a record.
pub(crate) fn write(path: &Path, measured: Measured) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let partial = path.with_extension("partial");
    std::fs::write(
        &partial,
        format!(
            "{HEADER}\nbuild {:016x}\nbytes {}\nbodies {}\ncaptured {}\n",
            build(),
            measured.bytes,
            measured.bodies,
            measured.captured
        ),
    )?;
    std::fs::rename(partial, path)
}

/// What a load's arming ends with for the record at `path`: armed whole,
/// what the bodies took is written; short of whole (`armed` of `wanted`)
/// after holding a record's figure, the record under-held, and it is
/// dropped, so the next load measures them and records again. The
/// line to say, where there is one.
///
/// Armed whole, the record never shrinks: the key fixes the build and the
/// deployment, so the bound is what a cold process's captures cost, and a
/// load in a process that already holds what an earlier load of it built
/// (its modules, its graph resources), or a re-arm within one load, costs
/// less. The larger of the stored and the measured figure is kept, and a
/// measurement whose captures came to nothing, or to less than the bodies'
/// own census, measured nothing or something reused, and is not written.
pub(crate) fn conclude(
    path: &Path,
    held_from_record: bool,
    whole: bool,
    measured: Measured,
    armed: usize,
    wanted: usize,
) -> std::io::Result<Option<String>> {
    if whole {
        if measured.bodies == 0 || measured.captured == 0 || measured.captured < measured.bytes {
            return Ok(None);
        }
        let stored = read(path).map_or(0, |stored| stored.captured);
        write(
            path,
            Measured {
                captured: measured.captured.max(stored),
                ..measured
            },
        )?;
        return Ok(None);
    }
    if !held_from_record {
        return Ok(None);
    }
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(why) if why.kind() == std::io::ErrorKind::NotFound => {}
        Err(why) => return Err(why),
    }
    Ok(Some(format!(
        "engine-cuda: the bodies' record under-held: {armed} of {wanted} compositions armed; \
         the record is dropped and the next load measures them again"
    )))
}

#[cfg(test)]
mod tests {
    use super::{Measured, conclude, path, read, write};

    #[test]
    fn a_record_reads_back_and_holds_what_its_captures_took() {
        let dir = std::env::temp_dir().join(format!("pie-measured-{}", std::process::id()));
        let at = path(&dir, "deployment");
        assert_eq!(read(&at), None, "no record yet");
        let measured = Measured {
            bytes: 404 << 20,
            bodies: 161,
            captured: 500 << 20,
        };
        write(&at, measured).expect("the record is written");
        assert_eq!(read(&at), Some(measured));
        assert_eq!(measured.hold(), 500 << 20);
        std::fs::write(&at, "pie-bodies 1\nbytes 1\nbodies 1\n").unwrap();
        assert_eq!(read(&at), None, "an older record reads as none");
        assert_ne!(
            path(&dir, "another deployment"),
            at,
            "another key, another file"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_record_never_shrinks_and_skips_a_reused_measurement() {
        let dir = std::env::temp_dir().join(format!("pie-measured-keep-{}", std::process::id()));
        let at = path(&dir, "deployment");
        let cold = Measured {
            bytes: 6 << 20,
            bodies: 10,
            captured: 8 << 20,
        };
        assert_eq!(conclude(&at, false, true, cold, 10, 10).unwrap(), None);
        assert_eq!(read(&at), Some(cold));
        // A warmer load costs less: the record keeps the cold figure.
        let warm = Measured {
            captured: 7 << 20,
            ..cold
        };
        assert_eq!(conclude(&at, true, true, warm, 10, 10).unwrap(), None);
        assert_eq!(read(&at), Some(cold));
        // Captures under the census reused something: not written.
        let reused = Measured {
            captured: 1 << 20,
            ..cold
        };
        assert_eq!(conclude(&at, true, true, reused, 10, 10).unwrap(), None);
        assert_eq!(read(&at), Some(cold));
        // A warm process's captures that cost nothing measured nothing.
        let nothing = Measured {
            bytes: 0,
            captured: 0,
            ..cold
        };
        assert_eq!(conclude(&at, true, true, nothing, 10, 10).unwrap(), None);
        assert_eq!(read(&at), Some(cold));
        let fresh = path(&dir, "another deployment");
        assert_eq!(
            conclude(&fresh, false, true, nothing, 10, 10).unwrap(),
            None
        );
        assert_eq!(read(&fresh), None, "no record of nothing");
        // A colder one costs more: the record grows to it.
        let colder = Measured {
            captured: 9 << 20,
            ..cold
        };
        assert_eq!(conclude(&at, true, true, colder, 10, 10).unwrap(), None);
        assert_eq!(read(&at), Some(colder));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_record_that_under_held_is_dropped_and_said() {
        let dir = std::env::temp_dir().join(format!("pie-measured-drop-{}", std::process::id()));
        let at = path(&dir, "deployment");
        let measured = Measured {
            bytes: 404 << 20,
            bodies: 161,
            captured: 500 << 20,
        };
        // Armed whole: written, nothing said.
        assert_eq!(
            conclude(&at, false, true, measured, 161, 161).unwrap(),
            None
        );
        assert_eq!(read(&at), Some(measured));
        // Short of whole on the ceiling: the record stands.
        assert_eq!(
            conclude(&at, false, false, measured, 150, 161).unwrap(),
            None
        );
        assert_eq!(read(&at), Some(measured));
        // Short of whole on the record's figure: dropped, and said.
        let said = conclude(&at, true, false, measured, 150, 161).unwrap();
        assert!(said.is_some_and(|line| line.contains("150 of 161 compositions armed")));
        assert_eq!(read(&at), None, "the next load measures them again");
        // Dropped twice is no error.
        assert!(
            conclude(&at, true, false, measured, 150, 161)
                .unwrap()
                .is_some()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// The records under `cache_dir` (`$PIE_HOME/cache`) this build made,
/// copied into `out`'s `bodies/` for a build to carry (`pie tune export
/// --engine cuda`): the names copied. A record another build made keys no
/// load of this one, and is left.
pub(crate) fn export(cache_dir: &Path, out: &Path) -> std::io::Result<Vec<String>> {
    let from = cache_dir.join("bodies");
    let to = out.join("bodies");
    let mut copied = Vec::new();
    let Ok(entries) = std::fs::read_dir(&from) else {
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
        if parse(&text).is_none() {
            continue;
        }
        std::fs::create_dir_all(&to)?;
        std::fs::write(to.join(&name), text)?;
        copied.push(name);
    }
    copied.sort();
    Ok(copied)
}

#[cfg(test)]
mod key_tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("pie-bodies-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn a_record_of_this_build_reads_back_and_exports_and_another_builds_does_not() {
        let cache = tmp("cache");
        let record = path(&cache, "a deployment");
        let measured = Measured { bytes: 100, bodies: 3, captured: 120 };
        write(&record, measured).unwrap();
        let back = read(&record).expect("this build's record reads");
        assert_eq!((back.bytes, back.bodies, back.captured), (100, 3, 120));
        // Another build's record keys no load of this one, and is not carried.
        let other = cache.join("bodies").join("0000000000000000");
        std::fs::write(
            &other,
            format!("{HEADER}\nbuild 0123456789abcdef\nbytes 1\nbodies 1\ncaptured 1\n"),
        )
        .unwrap();
        assert!(read(&other).is_none());
        let out = tmp("out");
        let copied = export(&cache, &out).unwrap();
        assert_eq!(copied, vec![record.file_name().unwrap().to_str().unwrap().to_string()]);
        let _ = std::fs::remove_dir_all(cache);
        let _ = std::fs::remove_dir_all(out);
    }

    #[test]
    fn a_key_moves_with_the_deployment_and_not_with_the_cache_directory() {
        let a = path(Path::new("/a"), "one");
        let b = path(Path::new("/b"), "one");
        assert_eq!(a.file_name(), b.file_name());
        assert_ne!(a.file_name(), path(Path::new("/a"), "two").file_name());
    }
}
