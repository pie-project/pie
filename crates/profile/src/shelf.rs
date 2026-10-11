//! Where a backend's profiles live and how one load finds its own: the
//! user's file for the device, then the one checked in with pie, then the
//! registry defaults.

use std::path::{Path, PathBuf};

use crate::profile::{Candidate, Facts, Profile, Report, Source};
use crate::registry::Registry;

/// A backend's checked-in profiles, and where its users keep theirs.
pub struct Shelf {
    pub backend: &'static str,
    /// `(file name, text)` of each checked-in profile, filed under the id of
    /// the device it names.
    pub seeded: &'static [(&'static str, &'static str)],
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
    /// The checked-in profile for `device`, if pie has one.
    #[must_use]
    pub fn seeded_for(&self, device: &str) -> Option<Candidate> {
        let file = format!("{}.toml", id_of(device));
        self.seeded
            .iter()
            .find(|(name, _)| *name == file)
            .map(|(name, text)| Candidate {
                source: Source::Seeded((*name).to_string()),
                text: (*text).to_string(),
            })
    }

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
        registry.resolve(&self.candidates(device, profiles), facts)
    }

    /// The profile `device` would resolve to, without facts to check:
    /// for tools that show one by name.
    #[must_use]
    pub fn read(
        &self,
        registry: &Registry,
        device: &str,
        profiles: Option<&Path>,
    ) -> (Profile, Report) {
        registry.read(&self.candidates(device, profiles))
    }

    fn candidates(&self, device: &str, profiles: Option<&Path>) -> Vec<Candidate> {
        let mut candidates = Vec::new();
        if let Some(dir) = profiles {
            let path = self.user_path(dir, device);
            if let Ok(text) = std::fs::read_to_string(&path) {
                candidates.push(Candidate {
                    source: Source::User(path.display().to_string()),
                    text,
                });
            }
        }
        candidates.extend(self.seeded_for(device));
        candidates
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

    /// Every checked-in profile reads strictly against `registry` and is
    /// filed under the device it names; `dir` is the source directory, which
    /// must hold exactly the listed files. For each backend's tests.
    pub fn check(&self, registry: &Registry, dir: &Path) -> Result<(), String> {
        let mut on_disk: Vec<String> = std::fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".toml"))
            .collect();
        on_disk.sort();
        let mut listed: Vec<String> = self.seeded.iter().map(|(n, _)| (*n).to_string()).collect();
        listed.sort();
        if on_disk != listed {
            return Err(format!(
                "{} lists {listed:?}, and the directory holds {on_disk:?}",
                self.backend
            ));
        }
        for (name, text) in self.seeded {
            let file = registry
                .parse(text, name, crate::Strict::CheckedIn)
                .map_err(|e| e.to_string())?;
            let device = file
                .target
                .device
                .ok_or_else(|| format!("{name} names no `target.device`"))?;
            if format!("{}.toml", id_of(&device)) != *name {
                return Err(format!(
                    "{name} names `{device}`, filed as {}",
                    id_of(&device)
                ));
            }
        }
        Ok(())
    }
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
