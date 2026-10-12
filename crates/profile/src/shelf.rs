//! Where a backend's profiles live and how one load finds its layers: the
//! family file whose assumptions the device meets (or the backend's derived
//! profile where none does), the device file in that family's directory, and
//! the user's file, each laid over the registry defaults in turn.

use std::path::{Path, PathBuf};

use crate::profile::{Candidate, Facts, Profile, Report, Source};
use crate::registry::{File, Registry, Strict, Target};

/// A backend's checked-in profiles, and where its users keep theirs.
pub struct Shelf {
    pub backend: &'static str,
    /// `(path, text)` of each checked-in profile: `apple9.toml` is a family
    /// file, `apple9/apple-m4-pro.toml` a device file of that family.
    pub seeded: &'static [(&'static str, &'static str)],
    /// The profile a device no family file applies to runs, from its name
    /// and facts.
    pub derived: Option<fn(&str, &Facts) -> String>,
}

/// The id a device's profiles are filed under: its name, lower-cased, with
/// every run of other characters a dash. `Apple M4 Max` is `apple-m4-max`.
#[must_use]
pub fn id_of(device: &str) -> String {
    let mut out = String::new();
    for c in device.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    out.trim_end_matches('-').to_string()
}

impl Shelf {
    /// Where a user's profile for `device` lives under `profiles` (the
    /// `$PIE_HOME/profiles` directory).
    #[must_use]
    pub fn user_path(&self, profiles: &Path, device: &str) -> PathBuf {
        profiles
            .join(self.backend)
            .join("user")
            .join(format!("{}.toml", id_of(device)))
    }

    /// Resolve the profile one load on `device` runs under.
    #[must_use]
    pub fn resolve(
        &self,
        registry: &Registry,
        device: &str,
        facts: &Facts,
        profiles: Option<&Path>,
    ) -> (Profile, Report) {
        let mut layers = Vec::new();
        let family = self
            .families()
            .find(|(_, text)| target(registry, text).is_some_and(|t| facts.hold(&t.assumes)));
        match family {
            Some((path, text)) => {
                layers.push(seeded(path, text));
                for (path, text) in self.devices(path) {
                    let Some(t) = target(registry, text) else {
                        continue;
                    };
                    // A file that names the device is offered whatever it
                    // assumes, so a refusal says why.
                    let anonymous = t.device.is_none() && t.name_contains.is_empty();
                    if names(&t, device) || (anonymous && facts.hold(&t.assumes)) {
                        layers.push(seeded(path, text));
                    }
                }
            }
            None => {
                if let Some(derived) = self.derived {
                    layers.push(Candidate {
                        source: Source::Derived,
                        text: derived(device, facts),
                    });
                }
            }
        }
        layers.extend(self.user(device, profiles));
        registry.resolve(&layers, facts)
    }

    /// The profile `device` would resolve to without facts to check: the
    /// device file that names it and its family, then the user's file. For
    /// tools that show one by name.
    #[must_use]
    pub fn read(
        &self,
        registry: &Registry,
        device: &str,
        profiles: Option<&Path>,
    ) -> (Profile, Report) {
        let mut layers = Vec::new();
        let named = self.seeded.iter().find(|(path, text)| {
            path.contains('/') && target(registry, text).is_some_and(|t| names(&t, device))
        });
        if let Some((path, text)) = named {
            let dir = path.rsplit_once('/').map_or("", |(dir, _)| dir);
            if let Some((family, family_text)) = self.families().find(|(f, _)| stem(f) == dir) {
                layers.push(seeded(family, family_text));
            }
            layers.push(seeded(path, text));
        }
        layers.extend(self.user(device, profiles));
        registry.read(&layers)
    }

    fn families(&self) -> impl Iterator<Item = (&'static str, &'static str)> + '_ {
        self.seeded
            .iter()
            .copied()
            .filter(|(path, _)| !path.contains('/'))
    }

    fn devices(&self, family: &str) -> impl Iterator<Item = (&'static str, &'static str)> + '_ {
        let dir = format!("{}/", stem(family));
        self.seeded
            .iter()
            .copied()
            .filter(move |(path, _)| path.starts_with(&dir))
    }

    fn user(&self, device: &str, profiles: Option<&Path>) -> Option<Candidate> {
        let path = self.user_path(profiles?, device);
        let text = std::fs::read_to_string(&path).ok()?;
        Some(Candidate {
            source: Source::User(path.display().to_string()),
            text,
        })
    }

    /// Say once per device, on stderr, that it runs without a profile
    /// measured on it, and how to make one.
    pub fn tell_untuned(&self, device: &str, report: &Report) {
        use std::collections::BTreeSet;
        use std::sync::Mutex;
        static TOLD: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());
        if !report.untuned() {
            return;
        }
        let mut told = TOLD
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if told.insert(format!("{}/{device}", self.backend)) {
            eprintln!(
                "{}",
                crate::untuned_hint(device, self.backend, &id_of(device), report)
            );
        }
    }

    /// The checked-in profiles hold together, for each backend's tests:
    /// `dir`, the source directory, holds exactly the listed files, a family
    /// file at the top and its device files in a directory of its name;
    /// every file reads strictly against `registry` (a device file over its
    /// family); a family file assumes facts and names no device, and no two
    /// families can apply to one device; a device file that names a device
    /// is filed under its id, and no two device files of a family can apply
    /// to one device.
    pub fn check(&self, registry: &Registry, dir: &Path) -> Result<(), String> {
        let mut on_disk = Vec::new();
        walk(dir, "", &mut on_disk)?;
        on_disk.sort();
        let mut listed: Vec<String> = self.seeded.iter().map(|(n, _)| (*n).to_string()).collect();
        listed.sort();
        if on_disk != listed {
            return Err(format!(
                "{} lists {listed:?}, and the directory holds {on_disk:?}",
                self.backend
            ));
        }
        let read = |name: &str, text: &str, below: Option<&File>| {
            let (under, inherited) = match below {
                Some(f) => (f.entries.clone(), f.target.assumes.clone()),
                None => Default::default(),
            };
            registry
                .parse_over(text, name, Strict::CheckedIn, &under, &inherited)
                .map_err(|e| e.to_string())
        };
        let mut families: Vec<(&str, File)> = Vec::new();
        for (name, text) in self.families() {
            let file = read(name, text, None)?;
            if file.target.assumes.is_empty() {
                return Err(format!("{name} is a family file, and assumes no fact"));
            }
            if file.target.device.is_some() || !file.target.name_contains.is_empty() {
                return Err(format!(
                    "{name} is a family file, and names a device; a device file goes in {}/",
                    stem(name)
                ));
            }
            if let Some((other, _)) = families
                .iter()
                .find(|(_, f)| !apart(&f.target, &file.target))
            {
                return Err(format!(
                    "{other} and {name} can both apply to one device: no fact they assume differs"
                ));
            }
            families.push((name, file));
        }
        for (name, _) in self.seeded.iter().filter(|(n, _)| n.contains('/')) {
            let (dir, _) = name.split_once('/').expect("a device file");
            if name.matches('/').count() > 1 || !families.iter().any(|(f, _)| stem(f) == dir) {
                return Err(format!(
                    "{name} is not in the directory of a family file: a device file is <family>/<device>.toml beside <family>.toml"
                ));
            }
        }
        for (family, family_file) in &families {
            let mut devices: Vec<(&str, Target)> = Vec::new();
            for (name, text) in self.devices(family) {
                let file = read(name, text, Some(family_file))?;
                if let Some(device) = &file.target.device {
                    let filed = format!("{}/{}.toml", stem(family), id_of(device));
                    if filed != name {
                        return Err(format!("{name} names `{device}`, filed as {filed}"));
                    }
                }
                if let Some((other, _)) = devices.iter().find(|(_, t)| {
                    !apart(t, &file.target)
                        && !matches!((&t.device, &file.target.device), (Some(a), Some(b)) if a != b)
                }) {
                    return Err(format!(
                        "{other} and {name} can both apply to one device: they name no two devices and no fact they assume differs"
                    ));
                }
                devices.push((name, file.target));
            }
        }
        Ok(())
    }
}

fn seeded(path: &str, text: &str) -> Candidate {
    Candidate {
        source: Source::Seeded(path.to_string()),
        text: text.to_string(),
    }
}

/// `apple9` of `apple9.toml`.
fn stem(path: &str) -> &str {
    path.trim_end_matches(".toml")
}

/// A file's target; `None` where it does not read, and the file never
/// applies (`Shelf::check` keeps every checked-in target readable).
fn target(registry: &Registry, text: &str) -> Option<Target> {
    registry.target_of(text)
}

/// Whether a device file names `device`, by name or by a substring.
fn names(target: &Target, device: &str) -> bool {
    target.device.as_deref() == Some(device)
        || target
            .name_contains
            .iter()
            .any(|s| device.contains(s.as_str()))
}

/// Whether no device can meet both targets' assumptions.
fn apart(a: &Target, b: &Target) -> bool {
    a.assumes
        .iter()
        .any(|&(i, x)| b.assumes.iter().any(|&(j, y)| i == j && x != y))
}

/// Every `.toml` under `dir`, as a path relative to it with `/` between parts.
fn walk(dir: &Path, prefix: &str, out: &mut Vec<String>) -> Result<(), String> {
    for entry in std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let entry = entry.map_err(|e| format!("{}: {e}", dir.display()))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if entry.path().is_dir() {
            walk(&entry.path(), &format!("{prefix}{name}/"), out)?;
        } else if name.ends_with(".toml") {
            out.push(format!("{prefix}{name}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_device_name_files_under_its_slug() {
        assert_eq!(id_of("Apple M4 Max"), "apple-m4-max");
        assert_eq!(id_of("NVIDIA GeForce RTX 4090"), "nvidia-geforce-rtx-4090");
        assert_eq!(id_of("Apple A17 Pro GPU"), "apple-a17-pro-gpu");
        assert_eq!(id_of("  odd//name  "), "odd-name");
    }
}
