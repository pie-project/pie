//! `pie cache` — what pie has written under `$PIE_HOME`, and reclaiming it.
//!
//! `list` and `clear` both read `pie_worker::state`, which is the point of the
//! registry: describing and reclaiming cannot disagree about what exists.


use anyhow::{Result, anyhow, bail};
use clap::Subcommand;
use pie_worker::state::{self, Reclaim};

use crate::ui::{self, Align, Answer, Mark, Palette, Row, Table};

#[derive(Subcommand, Debug)]
pub enum CacheCmd {
    /// Show what pie has written, where, and how much of it there is.
    List,

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

pub fn run(cmd: CacheCmd) -> Result<Answer> {
    match cmd {
        CacheCmd::List => list(),
        CacheCmd::Clear { names, yes } => clear(names, yes),
    }
}

/// What pie has written, and how much of it a `clear` would take back.
#[derive(serde::Serialize)]
pub struct CacheReport {
    home: std::path::PathBuf,
    entries: Vec<CacheRow>,
    reclaimable_bytes: u64,
    on_request_bytes: u64,
}

#[derive(serde::Serialize)]
struct CacheRow {
    name: &'static str,
    path: std::path::PathBuf,
    what: &'static str,
    /// The word the CLI uses, not the enum's Rust spelling: a consumer should
    /// be able to pass it back to `clear`.
    reclaim: &'static str,
    exists: bool,
    bytes: u64,
}

impl ui::Report for CacheReport {
    fn render(&self, palette: &Palette) {
        println!(
            "{} {}",
            palette.dim("pie writes under"),
            palette.bold(ui::short_path(&self.home))
        );
        let mut table = Table::new([Align::Left, Align::Right, Align::Left], 2);
        for row in &self.entries {
            // An absent entry is reported rather than hidden: "pie has not
            // written this yet" and "pie does not know about this" are
            // different answers, and only the listing can tell them apart.
            let mark = if row.exists { Mark::Plain } else { Mark::Absent };
            table.push(Row::new(
                mark,
                [
                    row.name.to_string(),
                    if row.exists {
                        ui::bytes(row.bytes)
                    } else {
                        String::new()
                    },
                    // The name already says which directory this is -- five of
                    // the nine rows repeated the same word -- so the note
                    // column carries what deleting it costs instead.
                    match row.reclaim {
                        "safe" => "",
                        "on-request" => "kept unless asked",
                        _ => "never reclaimed",
                    }
                    .to_string(),
                ],
            ));
        }
        table.print(palette);

        println!();
        if self.reclaimable_bytes == 0 && self.on_request_bytes == 0 {
            // "0B reclaimable, 0B more if asked" reads as a failure to find
            // something rather than as there being nothing to reclaim.
            println!("  nothing to reclaim");
        } else {
            println!(
                "  {} reclaimable, {} more if asked",
                ui::bytes(self.reclaimable_bytes),
                ui::bytes(self.on_request_bytes)
            );
        }
    }
}

fn list() -> Result<Answer> {
    let entries = state::entries(Some(hf_hub::resolve_cache_dir()));
    let home = pie_worker::paths::pie_home();

    // Measured once and reused: the size is what decides whether a row is
    // worth a person's attention, and walking a weight-sized tree twice to
    // print it twice would be the slowest part of the command.
    let measured: Vec<(state::Entry, bool, u64)> = entries
        .into_iter()
        .map(|entry| {
            let exists = entry.path.exists();
            // `entry.size()`, not `disk_usage(entry.path)`: the driver cache
            // physically contains the weight cache and the planner profile,
            // and each of those is its own row. Summing the raw trees counted
            // their bytes twice, so the "reclaimable" total below overstated
            // what a clear would actually free.
            let size = if exists { entry.size() } else { 0 };
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

    Ok(Answer::report(CacheReport {
        home,
        entries: measured
            .into_iter()
            .map(|(entry, exists, bytes)| CacheRow {
                name: entry.name,
                path: entry.path,
                what: entry.what,
                reclaim: match entry.reclaim {
                    Reclaim::Safe => "safe",
                    Reclaim::OnRequest => "on-request",
                    Reclaim::Never => "never",
                },
                exists,
                bytes,
            })
            .collect(),
        reclaimable_bytes: reclaimable,
        on_request_bytes: on_request,
    }))
}

/// Resolve the entries a `clear` should act on.
///
/// No names means every `Safe` entry: the ones whose loss is rebuild time.
/// Reaching an `OnRequest` entry takes naming it, and there is deliberately no
/// flag that sweeps them in -- a single character should not stand between a
/// person and deleting weight-sized artifacts.
fn selected(names: &[String]) -> Result<Vec<state::Entry>> {
    let all = state::entries(Some(hf_hub::resolve_cache_dir()));
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

fn clear(names: Vec<String>, skip_confirm: bool) -> Result<Answer> {
    let chosen = selected(&names)?;
    let present: Vec<(state::Entry, u64)> = chosen
        .into_iter()
        .filter(|entry| entry.path.exists())
        .map(|entry| {
            let size = entry.size();
            (entry, size)
        })
        .collect();

    if present.is_empty() {
        return Ok(Answer::noop("nothing to clear"));
    }

    let total: u64 = present.iter().map(|(_, size)| *size).sum();
    for (entry, size) in &present {
        println!("  {:<12} {:>8}  {}", entry.name, ui::bytes(*size), entry.path.display());
    }
    println!();

    if !skip_confirm {
        let question = format!(
            "Delete {} from {} entries?",
            ui::bytes(total),
            present.len()
        );
        if !ui::confirm(&question, "pie cache clear --yes")? {
            return Ok(Answer::noop("aborted; nothing was deleted"));
        }
    }

    let mut freed = 0u64;
    for (entry, size) in &present {
        // `entry.remove()` rather than a blanket `remove_dir_all`, so clearing
        // `driver` leaves behind the two things inside it that are somebody
        // else's: the weight cache, which is its own row, and the planner
        // profile, which no boot rebuilds.
        match entry.remove() {
            Ok(()) => freed += size,
            // Reported, not fatal: one unreadable entry must not stop the rest
            // from being reclaimed, and the total says what actually went.
            Err(error) => eprintln!("  ! {}: {error}", entry.path.display()),
        }
    }
    Ok(Answer::did(format!("freed {}", ui::bytes(freed))))
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
        assert!(chosen.iter().all(|e| e.name != "models"));
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
        // `models` is the artifact store: reclaimable, but a re-download and a
        // re-convert rather than a reload, which is why it takes naming.
        let chosen = selected(&["models".to_string()]).unwrap();
        assert_eq!(chosen.len(), 1);
        assert_eq!(chosen[0].reclaim, Reclaim::OnRequest);
    }

    #[test]
    fn the_store_and_the_weight_cache_are_different_entries() {
        // They shared `$PIE_HOME/models` until the artifact store took it: the
        // store scans for `.zt` and silently ignored the `.weights` files
        // beside them, while `pie cache` reported their size under the store's
        // name. Different directories, different reclaim grades.
        let all = state::entries(None);
        let by = |n: &str| all.iter().find(|e| e.name == n).unwrap_or_else(|| panic!("no {n}"));
        assert_ne!(by("models").path, by("weights").path);
        assert_eq!(by("models").reclaim, Reclaim::OnRequest);
        assert_eq!(by("weights").reclaim, Reclaim::Safe);
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
