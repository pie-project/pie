//! What XLA places on the device beside a load's weights and pool, measured
//! from its compiled programs (XLA's memory analysis of each, never a
//! timing): the largest temp (one reservation the runtime sizes to the
//! largest loaded program's), the largest program's code (the code cache
//! holds the program that runs; the rest reload, allowed after ready), and
//! the largest output not aliased into an argument.
//!
//! The pool's pages are program parameters, so they are fixed before any
//! program traces, and a fire's temp moves with them (XLA's memory-space
//! assignment evicts more of a hybrid's state blocks to HBM beside a larger
//! pool: v6e Qwen3.5's widest prompt 164 MiB at one pool, 229 MiB at four
//! times its pages). A fresh load measures it in a sizing pass: the fires
//! that bind it (`Shell::size_outside`) compiled against the pool sized
//! with none (T0), then against the pool T0 leaves (T1, the programs the
//! ladder keeps); the larger of the two, in bytes, sizes the load. The result is kept
//! beside the disk cache, keyed by what the load is sized from bar the
//! pool, so a warm load reads it and compiles nothing for it.

use std::path::{Path, PathBuf};

/// The measured outside of the pool (bytes) and the programs it was read
/// over.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Outside {
    pub temp: u64,
    pub code: u64,
    pub unaliased: u64,
    /// The programs compiled to measure it (0 where read from the record).
    pub programs: usize,
}

impl Outside {
    /// The device bytes it takes beside the weights and the pool.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.temp
            .saturating_add(self.code)
            .saturating_add(self.unaliased)
    }

    /// Over compiled programs' memory.
    #[must_use]
    pub fn of<'a>(memories: impl IntoIterator<Item = &'a crate::pjrt::CompiledMemory>) -> Outside {
        memories
            .into_iter()
            .fold(Outside::default(), |o, m| Outside {
                temp: o.temp.max(m.temp),
                code: o.code.max(m.code),
                unaliased: o.unaliased.max(m.output.saturating_sub(m.alias)),
                programs: o.programs + 1,
            })
    }

    fn line(&self) -> String {
        format!("{} {} {}", self.temp, self.code, self.unaliased)
    }

    fn parse(text: &str) -> Option<Outside> {
        let mut it = text.split_whitespace().map(|w| w.parse::<u64>().ok());
        let (temp, code, unaliased) = (it.next()??, it.next()??, it.next()??);
        Some(Outside {
            temp,
            code,
            unaliased,
            programs: 0,
        })
    }
}

/// Where the record of a load keyed `key` lies under the disk cache `dir`.
#[must_use]
pub fn record_path(dir: &Path, key: &str) -> PathBuf {
    dir.join("room").join(format!("{key}.outside"))
}

/// The recorded outside under `dir`, if one was kept for `key`.
#[must_use]
pub fn recorded(dir: &Path, key: &str) -> Option<Outside> {
    let text = std::fs::read_to_string(record_path(dir, key)).ok()?;
    Outside::parse(text.lines().next()?)
}

/// The step shapes the record kept for `key` states unservable.
#[must_use]
pub fn recorded_unservable(dir: &Path, key: &str) -> Option<Vec<u32>> {
    let text = std::fs::read_to_string(record_path(dir, key)).ok()?;
    let line = text.lines().nth(1).unwrap_or_default();
    line.split_whitespace().map(|w| w.parse().ok()).collect()
}

/// Keeps `outside`, and the step shapes the load states unservable by it,
/// for `key` under `dir` (a failed write is said, not fatal: the next load
/// measures again).
pub fn keep(dir: &Path, key: &str, outside: Outside, unservable: &[u32]) {
    let path = record_path(dir, key);
    let shapes: Vec<String> = unservable.iter().map(u32::to_string).collect();
    let written = path
        .parent()
        .map_or(Ok(()), std::fs::create_dir_all)
        .and_then(|()| {
            std::fs::write(&path, format!("{}\n{}\n", outside.line(), shapes.join(" ")))
        });
    if let Err(why) = written {
        tracing::warn!(path = %path.display(), %why, "xla outside record not kept: the next load measures it again");
    }
}

#[cfg(test)]
mod tests {
    use super::Outside;
    use crate::pjrt::CompiledMemory;

    #[test]
    fn the_outside_is_the_largest_of_each_term() {
        let m = |temp, code, output, alias| CompiledMemory {
            temp,
            code,
            output,
            alias,
            ..CompiledMemory::default()
        };
        let o = Outside::of(&[m(10, 1, 8, 8), m(4, 7, 9, 2), m(6, 3, 1, 0)]);
        assert_eq!((o.temp, o.code, o.unaliased, o.programs), (10, 7, 7, 3));
        assert_eq!(o.bytes(), 24);
    }

    #[test]
    fn a_record_reads_back_what_was_kept() {
        let dir = std::env::temp_dir().join(format!("pie-room-{}", std::process::id()));
        let o = Outside {
            temp: 5,
            code: 6,
            unaliased: 7,
            programs: 3,
        };
        super::keep(&dir, "k", o, &[93, 95]);
        assert_eq!(
            super::recorded(&dir, "k"),
            Some(Outside { programs: 0, ..o })
        );
        assert_eq!(super::recorded_unservable(&dir, "k"), Some(vec![93, 95]));
        super::keep(&dir, "none", o, &[]);
        assert_eq!(super::recorded_unservable(&dir, "none"), Some(vec![]));
        assert_eq!(super::recorded(&dir, "other"), None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
