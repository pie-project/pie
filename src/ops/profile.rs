//! `pie profile`: the device profiles this build carries, the one this
//! machine resolves to, and the files to share.

use anyhow::{Result, anyhow, bail};
use clap::Subcommand;
use runtime::engine::profiles::{Backend, backends};

use crate::ui::{self, Answer, Palette};

#[derive(Subcommand, Debug)]
pub enum ProfileCmd {
    /// Every decision a backend exposes: kinds, alternatives, constants,
    /// variables and defaults.
    Schema { backend: Option<String> },

    /// The checked-in profiles, and the user's.
    List { backend: Option<String> },

    /// The profile this machine's device resolves to, as a complete file:
    /// the starting point for hand editing.
    Dump {
        backend: Option<String>,
        /// Resolve for this device name instead of probing the machine.
        #[arg(long)]
        device: Option<String>,
    },

    /// A profile's file by id (`nvidia-l40s`, `apple-m4-pro`, `apple9`): the
    /// user's if there is one, else the checked-in one. For contributing.
    Export { id: String, backend: Option<String> },
}

pub fn run(cmd: ProfileCmd) -> Result<Answer> {
    match cmd {
        ProfileCmd::Schema { backend } => {
            let b = pick(backend.as_deref())?;
            Ok(text(profile::render::schema((b.registry)())))
        }
        ProfileCmd::List { backend } => list(&pick(backend.as_deref())?),
        ProfileCmd::Dump { backend, device } => dump(&pick(backend.as_deref())?, device),
        ProfileCmd::Export { id, backend } => export(&pick(backend.as_deref())?, &id),
    }
}

fn pick(name: Option<&str>) -> Result<Backend> {
    let mut all = backends();
    let names: Vec<&str> = all.iter().map(|b| b.name).collect();
    match name {
        Some(name) => {
            let at = all.iter().position(|b| b.name == name).ok_or_else(|| {
                anyhow!("this build hosts no `{name}` backend; it hosts {names:?}")
            })?;
            Ok(all.swap_remove(at))
        }
        None if all.len() == 1 => Ok(all.remove(0)),
        None if all.is_empty() => bail!("this build hosts no backend with device profiles"),
        None => bail!("name a backend: this build hosts {names:?}"),
    }
}

fn profiles_dir() -> std::path::PathBuf {
    crate::paths::pie_home().join("profiles")
}

#[derive(serde::Serialize)]
struct Text {
    text: String,
}

impl ui::Report for Text {
    fn render(&self, _palette: &Palette) {
        print!("{}", self.text);
        if !self.text.ends_with('\n') {
            println!();
        }
    }
}

pub(crate) fn text(text: String) -> Answer {
    Answer::report(Text { text })
}

fn list(b: &Backend) -> Result<Answer> {
    let mut out = format!("{} profiles checked in with pie:\n", b.name);
    for (name, _) in b.shelf.seeded {
        out.push_str(&format!("  {}\n", name.trim_end_matches(".toml")));
    }
    let dir = profiles_dir().join(b.name).join("user");
    let mut users: Vec<String> = std::fs::read_dir(&dir)
        .map(|entries| {
            entries
                .filter_map(std::result::Result::ok)
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n.ends_with(".toml"))
                .map(|n| n.trim_end_matches(".toml").to_string())
                .collect()
        })
        .unwrap_or_default();
    users.sort();
    out.push_str(&format!("user profiles in {}:\n", ui::short_path(&dir)));
    if users.is_empty() {
        out.push_str("  none\n");
    }
    for id in users {
        out.push_str(&format!("  {id}\n"));
    }
    Ok(text(out))
}

fn dump(b: &Backend, device: Option<String>) -> Result<Answer> {
    let registry = (b.registry)();
    let dir = profiles_dir();
    let (device, facts) = match device {
        Some(name) => (name, profile::Facts::of(registry)),
        None => (b.probe)()
            .ok_or_else(|| anyhow!("no {} device answers here; name one with --device", b.name))?,
    };
    let probed = facts != profile::Facts::of(registry);
    let (profile, report) = if probed {
        (b.resolve)(&device, &facts, Some(&dir))
    } else {
        b.shelf.read(registry, &device, Some(&dir))
    };
    let mut out = format!("# {device}\n");
    if probed {
        out.push_str(&format!("# facts: {}\n", facts.describe().join(", ")));
    } else {
        out.push_str("# not probed: `target.assumes` was not checked against a device\n");
    }
    for (source, why) in &report.rejected {
        out.push_str(&format!("# {source} was not used: {why}\n"));
    }
    out.push_str(&format!("# layers: {}\n", report.describe()));
    out.push_str(&profile::render::dump(&profile));
    Ok(text(out))
}

fn export(b: &Backend, id: &str) -> Result<Answer> {
    let id = id.trim_end_matches(".toml");
    let user = profiles_dir()
        .join(b.name)
        .join("user")
        .join(format!("{id}.toml"));
    if let Ok(found) = std::fs::read_to_string(&user) {
        return Ok(text(found));
    }
    let file = format!("{id}.toml");
    let found =
        b.shelf.seeded.iter().find(|(name, _)| {
            *name == file || name.rsplit_once('/').is_some_and(|(_, n)| n == file)
        });
    match found {
        Some((_, found)) => Ok(text((*found).to_string())),
        None => bail!(
            "no {} profile `{id}`: neither {} nor a checked-in file",
            b.name,
            ui::short_path(&user)
        ),
    }
}
