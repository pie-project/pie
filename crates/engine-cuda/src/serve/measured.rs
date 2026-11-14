//! What a deployment's graph bodies took the last time it armed whole,
//! kept beside the kernel cache: a load that finds it holds that for its
//! bodies, not the `[engine] bodies_mem` ceiling, and arms its lattice as
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

const HEADER: &str = "pie-bodies 3";

/// The record's file for a deployment whose bodies `parts` shape.
#[must_use]
pub(crate) fn path(cache_dir: &Path, parts: &str) -> PathBuf {
    use std::hash::{Hash, Hasher};
    let build = std::env::current_exe()
        .and_then(std::fs::metadata)
        .map(|meta| {
            let time = meta
                .modified()
                .ok()
                .and_then(|at| at.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |at| at.as_nanos());
            format!("{}:{time}", meta.len())
        })
        .unwrap_or_default();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    (env!("CARGO_PKG_VERSION"), build, parts).hash(&mut hasher);
    cache_dir
        .join("bodies")
        .join(format!("{:016x}", hasher.finish()))
}

/// The record at `path`; none where it is missing or not whole.
#[must_use]
pub(crate) fn read(path: &Path) -> Option<Measured> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut lines = text.lines();
    if lines.next()? != HEADER {
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
            "{HEADER}\nbytes {}\nbodies {}\ncaptured {}\n",
            measured.bytes, measured.bodies, measured.captured
        ),
    )?;
    std::fs::rename(partial, path)
}

/// What a load's arming ends with for the record at `path`: armed whole,
/// what the bodies took is written; short of whole (`armed` of `wanted`)
/// after holding a record's figure, the record under-held, and it is
/// dropped, so the next load holds the ceiling and records again. The
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
         the record is dropped and the next load holds bodies_mem"
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
        assert_eq!(read(&at), None, "the next load holds bodies_mem");
        // Dropped twice is no error.
        assert!(
            conclude(&at, true, false, measured, 150, 161)
                .unwrap()
                .is_some()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
