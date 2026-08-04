//! `pie cache` — what pie has written under `$PIE_HOME`, and reclaiming it.
//!
//! The entries come from `pie_worker::state`, which is also what a future
//! `clear` will read: the point of the registry is that describing and
//! reclaiming cannot disagree about what exists.

use std::io::IsTerminal;

use anyhow::Result;
use clap::Subcommand;
use pie_worker::state::{self, Reclaim};

#[derive(Subcommand, Debug)]
pub enum CacheCmd {
    /// Show what pie has written, where, and how much of it there is.
    List,
}

pub fn run(cmd: CacheCmd) -> Result<()> {
    match cmd {
        CacheCmd::List => list(),
    }
}

/// Bytes as the largest binary unit that leaves a number worth reading.
fn human(bytes: u64) -> String {
    const UNITS: [(&str, u64); 4] = [
        ("GiB", 1 << 30),
        ("MiB", 1 << 20),
        ("KiB", 1 << 10),
        ("B", 1),
    ];
    for (suffix, scale) in UNITS {
        if bytes >= scale {
            let value = bytes as f64 / scale as f64;
            return if scale == 1 || value >= 100.0 {
                format!("{value:.0}{suffix}")
            } else {
                format!("{value:.1}{suffix}")
            };
        }
    }
    "0B".to_string()
}

fn list() -> Result<()> {
    let entries = state::entries();
    let home = pie_worker::paths::pie_home();

    let colorize = std::io::stdout().is_terminal();
    let (dim, bold, reset) = if colorize {
        ("\x1b[2m", "\x1b[1m", "\x1b[0m")
    } else {
        ("", "", "")
    };

    // Measured once and reused: the size is what decides whether a row is
    // worth a person's attention, and walking a weight-sized tree twice to
    // print it twice would be the slowest part of the command.
    let measured: Vec<(state::Entry, bool, u64)> = entries
        .into_iter()
        .map(|entry| {
            let exists = entry.path.exists();
            let size = if exists {
                state::disk_usage(&entry.path)
            } else {
                0
            };
            (entry, exists, size)
        })
        .collect();

    let name_width = measured
        .iter()
        .map(|(entry, _, _)| entry.name.len())
        .max()
        .unwrap_or(0);

    println!("{bold}{}{reset}", home.display());
    for (entry, exists, size) in &measured {
        let relative = entry
            .path
            .strip_prefix(&home)
            .unwrap_or(&entry.path)
            .display()
            .to_string();
        // An absent entry is reported rather than hidden: "pie has not written
        // this yet" and "pie does not know about this" are different answers,
        // and only the listing can tell them apart.
        let size_text = if *exists {
            human(*size)
        } else {
            "—".to_string()
        };
        let note = match entry.reclaim {
            Reclaim::Safe => "",
            Reclaim::OnRequest => " (kept unless asked)",
            Reclaim::Never => " (never reclaimed)",
        };
        println!(
            "  {:<name_width$}  {:>8}  {dim}{relative}{note}{reset}",
            entry.name,
            size_text,
            name_width = name_width,
        );
    }

    let reclaimable: u64 = measured
        .iter()
        .filter(|(entry, _, _)| entry.reclaim == Reclaim::Safe)
        .map(|(_, _, size)| *size)
        .sum();
    let on_request: u64 = measured
        .iter()
        .filter(|(entry, _, _)| entry.reclaim == Reclaim::OnRequest)
        .map(|(_, _, size)| *size)
        .sum();
    println!();
    println!(
        "  {} reclaimable, {} more if asked",
        human(reclaimable),
        human(on_request)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_read_as_the_unit_a_person_would_use() {
        assert_eq!(human(0), "0B");
        assert_eq!(human(512), "512B");
        assert_eq!(human(1 << 10), "1.0KiB");
        assert_eq!(human(3 << 30), "3.0GiB");
        // Past three digits the fraction is noise.
        assert_eq!(human(200 << 20), "200MiB");
    }
}
