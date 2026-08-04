//! `pie cache` — what pie has written under `$PIE_HOME`, and reclaiming it.
//!
//! `list` and `clear` both read `pie_worker::state`, which is the point of the
//! registry: describing and reclaiming cannot disagree about what exists.

use std::io::{IsTerminal, Write};

use anyhow::{Result, anyhow, bail};
use clap::Subcommand;
use pie_worker::state::{self, Reclaim};

use crate::ui::{self, Align, Mark, Palette, Row, Stream, Table};

#[derive(Subcommand, Debug)]
pub enum CacheCmd {
    /// Show what pie has written, where, and how much of it there is.
    List {
        /// Emit one JSON document instead of the table.
        #[arg(long)]
        json: bool,
    },

    /// Delete what pie can rebuild. With no names, everything safe to lose.
    Clear {
        /// Entries to clear, by the names `pie cache list` prints. Naming one
        /// is the only way to reach the ones kept unless asked.
        names: Vec<String>,
        /// Skip the confirmation prompt.
        #[arg(long)]
        yes: bool,
    },
}

pub fn run(cmd: CacheCmd) -> Result<()> {
    match cmd {
        CacheCmd::List { json } => list(json),
        CacheCmd::Clear { names, yes } => clear(names, yes),
    }
}

fn list(json: bool) -> Result<()> {
    let entries = state::entries();
    let home = pie_worker::paths::pie_home();

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

    if json {
        return ui::emit_json(&serde_json::json!({
            "home": home,
            "entries": measured
                .iter()
                .map(|(entry, exists, size)| serde_json::json!({
                    "name": entry.name,
                    "path": entry.path,
                    "what": entry.what,
                    // The word the CLI uses, not the enum's Rust spelling: a
                    // consumer should be able to pass it back to `clear`.
                    "reclaim": match entry.reclaim {
                        Reclaim::Safe => "safe",
                        Reclaim::OnRequest => "on-request",
                        Reclaim::Never => "never",
                    },
                    "exists": exists,
                    "bytes": size,
                }))
                .collect::<Vec<_>>(),
            "reclaimable_bytes": reclaimable,
            "on_request_bytes": on_request,
        }));
    }

    let palette = Palette::for_stream(Stream::Stdout);
    let (dim, bold, reset) = (palette.dim(), palette.bold(), palette.reset());

    println!("{dim}pie writes under{reset} {bold}{}{reset}", ui::short_path(&home));
    let mut table = Table::new([Align::Left, Align::Right, Align::Left], 2);
    for (entry, exists, size) in &measured {
        // An absent entry is reported rather than hidden: "pie has not written
        // this yet" and "pie does not know about this" are different answers,
        // and only the listing can tell them apart.
        let mark = if *exists { Mark::Plain } else { Mark::Absent };
        let note = match entry.reclaim {
            Reclaim::Safe => "",
            Reclaim::OnRequest => "kept unless asked",
            Reclaim::Never => "never reclaimed",
        };
        table.push(Row::new(
            mark,
            [
                entry.name.to_string(),
                if *exists { ui::bytes(*size) } else { String::new() },
                // The name already says which directory this is -- five of the
                // nine rows repeated the same word -- so the note column
                // carries what deleting it costs instead.
                note.to_string(),
            ],
        ));
    }
    table.print(&palette);

    println!();
    if reclaimable == 0 && on_request == 0 {
        // "0B reclaimable, 0B more if asked" reads as a failure to find
        // something rather than as there being nothing to reclaim.
        println!("  nothing to reclaim");
    } else {
        println!(
            "  {} reclaimable, {} more if asked",
            ui::bytes(reclaimable),
            ui::bytes(on_request)
        );
    }
    Ok(())
}

/// Resolve the entries a `clear` should act on.
///
/// No names means every `Safe` entry: the ones whose loss is rebuild time.
/// Reaching an `OnRequest` entry takes naming it, and there is deliberately no
/// flag that sweeps them in -- a single character should not stand between a
/// person and deleting weight-sized artifacts.
fn selected(names: &[String]) -> Result<Vec<state::Entry>> {
    let all = state::entries();
    if names.is_empty() {
        return Ok(all
            .into_iter()
            .filter(|e| e.reclaim == Reclaim::Safe)
            .collect());
    }
    let mut chosen = Vec::new();
    for name in names {
        let entry = all
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| {
                let known: Vec<&str> = all.iter().map(|e| e.name).collect();
                anyhow!("unknown entry {name:?}; `pie cache list` shows: {}", known.join(", "))
            })?
            .clone();
        if entry.reclaim == Reclaim::Never {
            bail!(
                "{name} is authored, not derived, and is never cleared; \
                 delete it yourself if you mean to"
            );
        }
        chosen.push(entry);
    }
    Ok(chosen)
}

fn clear(names: Vec<String>, skip_confirm: bool) -> Result<()> {
    let chosen = selected(&names)?;
    let present: Vec<(state::Entry, u64)> = chosen
        .into_iter()
        .filter(|entry| entry.path.exists())
        .map(|entry| {
            let size = state::disk_usage(&entry.path);
            (entry, size)
        })
        .collect();

    if present.is_empty() {
        println!("(nothing to clear)");
        return Ok(());
    }

    let total: u64 = present.iter().map(|(_, size)| *size).sum();
    for (entry, size) in &present {
        println!("  {:<12} {:>8}  {}", entry.name, ui::bytes(*size), entry.path.display());
    }
    println!();

    if !skip_confirm {
        // Same rule as `pie model remove`: without a terminal there is nobody
        // to ask, so refuse rather than assume consent for a delete.
        if !std::io::stdin().is_terminal() {
            bail!("clear requires confirmation; rerun with `pie cache clear --yes`");
        }
        eprint!("Delete {} from {} entries? [y/N] ", ui::bytes(total), present.len());
        let _ = std::io::stderr().flush();
        let mut answer = String::new();
        std::io::stdin()
            .read_line(&mut answer)
            .map_err(|e| anyhow!("read stdin: {e}"))?;
        if !matches!(answer.trim(), "y" | "Y" | "yes" | "YES") {
            println!("(aborted)");
            return Ok(());
        }
    }

    let mut freed = 0u64;
    for (entry, size) in &present {
        let result = if entry.path.is_dir() {
            std::fs::remove_dir_all(&entry.path)
        } else {
            std::fs::remove_file(&entry.path)
        };
        match result {
            Ok(()) => freed += size,
            // Reported, not fatal: one unreadable entry must not stop the rest
            // from being reclaimed, and the total says what actually went.
            Err(error) => eprintln!("  ! {}: {error}", entry.path.display()),
        }
    }
    println!("freed {}", ui::bytes(freed));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_names_selects_exactly_the_safe_entries() {
        let chosen = selected(&[]).unwrap();
        assert!(!chosen.is_empty());
        assert!(chosen.iter().all(|e| e.reclaim == Reclaim::Safe));
        // The expensive ones are reachable only by name. This is the whole
        // reason there is no --all.
        assert!(chosen.iter().all(|e| e.name != "optimized"));
    }

    #[test]
    fn an_authored_entry_cannot_be_named() {
        let err = selected(&["config".to_string()]).unwrap_err().to_string();
        assert!(err.contains("never cleared"), "got: {err}");
    }

    #[test]
    fn an_unknown_name_says_what_the_names_are() {
        let err = selected(&["ptir".to_string()]).unwrap_err().to_string();
        assert!(err.contains("unknown entry"), "got: {err}");
        assert!(err.contains("driver"), "should list the real names; got: {err}");
    }

    #[test]
    fn naming_an_on_request_entry_selects_it() {
        let chosen = selected(&["optimized".to_string()]).unwrap();
        assert_eq!(chosen.len(), 1);
        assert_eq!(chosen[0].reclaim, Reclaim::OnRequest);
    }

    #[test]
    fn sizes_read_as_the_unit_a_person_would_use() {
        assert_eq!(ui::bytes(0), "0B");
        assert_eq!(ui::bytes(512), "512B");
        assert_eq!(ui::bytes(1 << 10), "1.0KiB");
        assert_eq!(ui::bytes(3 << 30), "3.0GiB");
        // Past three digits the fraction is noise.
        assert_eq!(ui::bytes(200 << 20), "200MiB");
    }
}
