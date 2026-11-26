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
    /// file, `apple9/apple-m4-pro.toml` a device file of that family, and
    /// `<device>.measured.toml` a committed measured profile of a device
    /// ([`MEASURED_SUFFIX`]): its stored measurements, as `pie tune export`
    /// writes them.
    pub seeded: &'static [(&'static str, &'static str)],
    /// The profile a device no family file applies to runs, from its name
    /// and facts.
    pub derived: Option<fn(&str, &Facts) -> String>,
}

/// What a committed measured profile's file name ends in:
/// `<id>.measured.toml`, at the top of the backend's `profiles/`, `<id>` the
/// device's ([`id_of`]). It holds a device's stored measurements in the user
/// file's format, each keyed ([`crate::Strict::Measured`]), apart from the
/// hand-written decision profiles.
pub const MEASURED_SUFFIX: &str = ".measured.toml";

/// The file name a committed measured profile of `device` is filed under.
#[must_use]
pub fn measured_name(device: &str) -> String {
    format!("{}{MEASURED_SUFFIX}", id_of(device))
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
        self.resolve_with(registry, device, facts, profiles, None)
    }

    /// [`Shelf::resolve`], the user's file's measurements sifted by their
    /// keys against `now` ([`Registry::resolve_keyed`]).
    #[must_use]
    pub fn resolve_keyed(
        &self,
        registry: &Registry,
        device: &str,
        facts: &Facts,
        profiles: Option<&Path>,
        now: &dyn crate::rates::Now,
    ) -> (Profile, Report) {
        self.resolve_with(registry, device, facts, profiles, Some(now))
    }

    fn resolve_with(
        &self,
        registry: &Registry,
        device: &str,
        facts: &Facts,
        profiles: Option<&Path>,
        now: Option<&dyn crate::rates::Now>,
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
                // A device the derived profile serves keeps what is its
                // own (its rates and fire times) in `derived/<id>.toml`.
                for (path, text) in self.devices(DERIVED_DIR) {
                    if target(registry, text).is_some_and(|t| names(&t, device)) {
                        layers.push(seeded(path, text));
                    }
                }
            }
        }
        layers.extend(self.committed(device));
        layers.extend(self.user(device, profiles));
        match now {
            Some(now) => registry.resolve_keyed(&layers, facts, now),
            None => registry.resolve(&layers, facts),
        }
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
        layers.extend(self.committed(device));
        layers.extend(self.user(device, profiles));
        registry.read(&layers)
    }

    fn families(&self) -> impl Iterator<Item = (&'static str, &'static str)> + '_ {
        self.seeded
            .iter()
            .copied()
            .filter(|(path, _)| !path.contains('/') && !path.ends_with(MEASURED_SUFFIX))
    }

    /// Each committed measured profile ([`MEASURED_SUFFIX`]).
    fn measured(&self) -> impl Iterator<Item = (&'static str, &'static str)> + '_ {
        self.seeded
            .iter()
            .copied()
            .filter(|(path, _)| path.ends_with(MEASURED_SUFFIX))
    }

    /// The committed measured profile of `device`, where pie carries one.
    fn committed(&self, device: &str) -> Option<Candidate> {
        let name = measured_name(device);
        self.measured()
            .find(|(path, _)| *path == name)
            .map(|(path, text)| Candidate {
                source: Source::Committed(path.to_string()),
                text: text.to_string(),
            })
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

    /// What [`Shelf::tell_untuned`] says of `device`: that it runs without
    /// a profile measured on it, and how to make one; `None` where a layer
    /// measured what it runs.
    #[must_use]
    pub fn untuned(&self, device: &str, report: &Report) -> Option<String> {
        report
            .untuned()
            .then(|| crate::untuned_hint(device, self.backend, &id_of(device), report))
    }

    /// Say once per device, as a warning, that it runs without a profile
    /// measured on it, and how to make one.
    pub fn tell_untuned(&self, device: &str, report: &Report) {
        use std::collections::BTreeSet;
        use std::sync::Mutex;
        static TOLD: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());
        let Some(hint) = self.untuned(device, report) else {
            return;
        };
        let mut told = TOLD.lock().expect("told lock poisoned");
        if told.insert(format!("{}/{device}", self.backend)) {
            tracing::warn!("{hint}");
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
        self.check_beside(registry, dir, &[])
    }

    /// [`Shelf::check`] over a directory another build of the same engine
    /// shares: `beside` are the files it seeds, and this build leaves them out.
    pub fn check_beside(
        &self,
        registry: &Registry,
        dir: &Path,
        beside: &[&str],
    ) -> Result<(), String> {
        let mut on_disk = Vec::new();
        walk(dir, "", &mut on_disk)?;
        on_disk.retain(|name| !beside.contains(&name.as_str()));
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
        // A committed measured profile: at the top, filed by its device's
        // id, its measurements alone, each keyed.
        for (name, text) in self.measured() {
            if name.contains('/') {
                return Err(format!(
                    "{name} is a measured profile, filed at the top as <device>{MEASURED_SUFFIX}"
                ));
            }
            let file = registry
                .parse(text, name, Strict::Measured)
                .map_err(|e| e.to_string())?;
            if file.tuned.rates == crate::rates::Rates::default() {
                return Err(format!(
                    "{name} is a measured profile, and states no measurement"
                ));
            }
        }
        for (name, _) in self
            .seeded
            .iter()
            .filter(|(n, _)| n.contains('/') && !n.ends_with(MEASURED_SUFFIX))
        {
            let (dir, _) = name.split_once('/').expect("a device file");
            let filed = dir == DERIVED_DIR || families.iter().any(|(f, _)| stem(f) == dir);
            if name.matches('/').count() > 1 || !filed {
                return Err(format!(
                    "{name} is not in the directory of a family file: a device file is <family>/<device>.toml beside <family>.toml"
                ));
            }
        }
        // A device the derived profile serves: its file names it, is filed
        // under its id, and is read over nothing.
        for (name, text) in self.devices(DERIVED_DIR) {
            let file = read(name, text, None)?;
            let Some(device) = &file.target.device else {
                return Err(format!(
                    "{name} is a device file of the derived profile, and names no device"
                ));
            };
            let filed = format!("{DERIVED_DIR}/{}.toml", id_of(device));
            if filed != name {
                return Err(format!("{name} names `{device}`, filed as {filed}"));
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

    /// Every checked-in decision profile (not a committed measured one,
    /// which holds measurements alone) loads, dumps as `pie tune` writes a
    /// profile, and reads back strictly to the same rungs and assumed facts.
    pub fn check_dumps(&self, registry: &Registry) -> Result<(), String> {
        for (name, text) in self
            .seeded
            .iter()
            .filter(|(name, _)| !name.ends_with(MEASURED_SUFFIX))
        {
            let p = registry
                .load(text, name)
                .map_err(|e| format!("{name}: {e}"))?;
            let dumped = crate::render::dump(&p);
            let again = registry
                .parse(&dumped, "dump", Strict::CheckedIn)
                .map_err(|e| format!("{name}: {e}\n---\n{dumped}"))?;
            for (spec, rungs, _) in p.entries() {
                if again.entries.get(spec.id).map(Vec::as_slice) != Some(rungs) {
                    return Err(format!("{name}: `{}` dumps to other rungs", spec.id));
                }
            }
            if again.target.assumes != p.target().assumes {
                return Err(format!("{name}: the facts it assumes dump to others"));
            }
        }
        Ok(())
    }
}

/// What `pie tune export` writes of a device's stored measurements: the
/// `[rates]` of `user` (the device's user file, read from `origin`) in
/// committed form ([`crate::Strict::Measured`], filed as
/// [`measured_name`]): each entry with its key, in a fixed order
/// ([`crate::rates::Rates::ordered`]), no date, path or decision. An entry
/// stored without a key, or a file naming no device it was measured on, is
/// refused by name; so is a file that does not read, or states no
/// measurement. The text reads back to the same entries.
pub fn export(
    registry: &Registry,
    device: &str,
    user: &str,
    origin: &str,
) -> Result<String, String> {
    export_reporting(registry, device, user, origin).map(|(text, _)| text)
}

/// [`export`], with what a boot of each model the export carries a settle
/// of reads, model by model ([`settles_carried`]): each line says one
/// model's settle, the curves its planes take and whether each is carried;
/// a settle whose planes lack a curve is left out, by name (a boot of that
/// model would time its curves, so a settle over curves the export does not
/// carry would key nothing it serves).
pub fn export_reporting(
    registry: &Registry,
    device: &str,
    user: &str,
    origin: &str,
) -> Result<(String, Vec<String>), String> {
    let file = registry
        .parse(user, origin, Strict::User)
        .map_err(|e| e.to_string())?;
    export_rates(registry, device, file.tuned.rates, origin)
}

/// [`export_reporting`] of `rates` (a home's measured entries, merged over
/// what of a committed profile still holds: [`merge_committed`]).
pub fn export_rates(
    registry: &Registry,
    device: &str,
    rates: crate::rates::Rates,
    origin: &str,
) -> Result<(String, Vec<String>), String> {
    let mut rates = rates.ordered();
    let report = settles_carried(&mut rates);
    if rates == crate::rates::Rates::default() {
        return Err(format!("{origin}: stores no measurement of `{device}`"));
    }
    if let Some(why) = rates.unkeyed() {
        return Err(format!(
            "{origin}: {why}: a committed profile takes keyed entries alone; a load of this pie \
             measures it again under its key (or remove it), then export"
        ));
    }
    let text = format!(
        "api = 1\n\n# The stored measurements of {device}, each under its key: what a load of \
         this device\n# takes where the user's own file stores none under a matching key. \
         Written by\n# `pie tune export`; not edited by hand.\n{}\n[provenance]\nrates = \
         \"pie tune export: {}'s stored measurements, each under its key\"\n",
        rates.toml(),
        device.replace('"', "'"),
    );
    let read = registry
        .parse(&text, &measured_name(device), Strict::Measured)
        .map_err(|e| format!("the export of {origin} would not read back: {e}"))?;
    if read.tuned.rates != rates {
        return Err(format!(
            "the export of {origin} reads back other entries than it stores"
        ));
    }
    Ok((text, report))
}

/// A home's measured entries (`measured`, its user's file, sifted) laid over
/// the device's committed profile (`committed`, as it reads), each committed
/// entry checked against this build (`now`): one whose key still matches is
/// carried, one whose key no longer does is dropped. A re-export so never
/// shrinks a committed profile silently. Each entry is said, a line each:
/// measured (the home's), carried (the committed one's) or dropped (its key
/// no longer matches, and why). Refused by name where a rate the committed
/// profile stated is dropped and the home did not measure it again: a boot
/// would then measure it.
pub fn merge_committed(
    measured: crate::rates::Rates,
    committed: Option<crate::rates::Rates>,
    now: Option<&dyn crate::rates::Now>,
) -> Result<(crate::rates::Rates, Vec<String>), String> {
    use crate::rates::{DENSE_SETTLED, KERNEL_CURVES, KEYED};
    let mut report = Vec::new();
    // The home's own entries are checked as the committed ones are: one
    // stored under a key this build no longer matches is never written.
    let mut measured = measured;
    if let Some(now) = now {
        for line in measured.sift_as(now, "stored") {
            report.push(format!("dropped: {line}"));
        }
        measured.stale = None;
    }
    let Some(mut committed) = committed else {
        for key in KEYED.iter().filter(|key| measured.states(key)) {
            report.push(format!("`{key}`: measured"));
        }
        return Ok((measured, report));
    };
    let before = committed.clone();
    let dropped = match now {
        Some(now) => committed.sift_as(now, "committed"),
        None => Vec::new(),
    };
    for line in &dropped {
        report.push(format!("dropped: {line}"));
    }
    committed.stale = None;
    let mut refused: Vec<String> = Vec::new();
    for key in KEYED {
        match (
            measured.states(key),
            committed.states(key),
            before.states(key),
        ) {
            (true, _, _) => report.push(format!("`{key}`: measured")),
            (false, true, _) => report.push(format!("`{key}`: carried from the committed profile")),
            (false, false, true) => refused.push(format!("`{key}`")),
            (false, false, false) => {}
        }
    }
    let curve_ids = |r: &crate::rates::Rates| -> Vec<(u32, u32, u32, u32)> {
        r.kernel_curves
            .iter()
            .flatten()
            .map(crate::curves::KernelCurve::id)
            .collect()
    };
    // A committed curve whose key no longer matches, which the home did not
    // measure again: never carried, and a boot would time it.
    let (kept, mine_now) = (curve_ids(&committed), curve_ids(&measured));
    for curve in before.kernel_curves.iter().flatten() {
        let id = curve.id();
        if !kept.contains(&id) && !mine_now.contains(&id) {
            refused.push(format!("curve {}", curve.name()));
        }
    }
    // A committed fire term whose key no longer matches, which the home did
    // not measure again: never carried, as a curve is not.
    let term_ids = |r: &crate::rates::Rates| -> Vec<(u32, Vec<u32>)> {
        r.fire_terms
            .iter()
            .flatten()
            .map(crate::curves::FireTerm::id)
            .collect()
    };
    let (terms_kept, terms_now) = (term_ids(&committed), term_ids(&measured));
    for term in before.fire_terms.iter().flatten() {
        let id = term.id();
        if !terms_kept.contains(&id) && !terms_now.contains(&id) {
            refused.push(format!("fire term {}", term.name()));
        }
    }
    if !refused.is_empty() {
        return Err(format!(
            "the committed profile states {}, whose keys no longer match this build (a kernel's \
             code or the measuring method changed), and the home holds no current measurement of \
             them: an export never writes a stale entry. Measure them in this home first (boot \
             each model with `[engine] measure = true` and let it idle until its `behind serving \
             ... done` lines, or `pie tune --rates` for the rates), then export",
            refused.join(", ")
        ));
    }
    let (mine, carried) = (curve_ids(&measured), curve_ids(&committed));
    report.push(format!(
        "{KERNEL_CURVES}: {} measured, {} carried from the committed profile",
        mine.len(),
        carried.iter().filter(|id| !mine.contains(id)).count()
    ));
    report.push(format!(
        "{}: {} measured, {} carried from the committed profile",
        crate::rates::FIRE_TERMS,
        terms_now.len(),
        terms_kept
            .iter()
            .filter(|id| !terms_now.contains(id))
            .count()
    ));
    let settles_mine: Vec<u32> = measured.dense_settled.iter().map(|d| d.of).collect();
    report.push(format!(
        "{DENSE_SETTLED}: {} measured, {} carried from the committed profile",
        settles_mine.len(),
        committed
            .dense_settled
            .iter()
            .filter(|d| !settles_mine.contains(&d.of))
            .count()
    ));
    let measured_on = measured
        .measured_on
        .clone()
        .or(committed.measured_on.clone());
    let mut merged = measured.over(committed);
    merged.measured_on = measured_on;
    Ok((merged, report))
}

/// For each settle `rates` stores (one a model), what a boot of its model
/// reads beside it: the dense curves at each of its planes, by kernel. A
/// settle one of whose planes `rates` states no curve at is taken out of
/// `rates` and said: carried, a boot of the model would time that plane's
/// curves and settle again over them.
pub fn settles_carried(rates: &mut crate::rates::Rates) -> Vec<String> {
    let curves = rates.kernel_curves.clone().unwrap_or_default();
    let mut report = Vec::new();
    let mut kept = Vec::new();
    for settled in std::mem::take(&mut rates.dense_settled) {
        let at: Vec<String> = settled
            .planes
            .iter()
            .map(|&(n, k)| {
                let count = curves.iter().filter(|c| (c.n, c.k) == (n, k)).count();
                format!("{n}x{k}: {count} curve(s)")
            })
            .collect();
        let missing: Vec<String> = settled
            .planes
            .iter()
            .filter(|&&(n, k)| !curves.iter().any(|c| (c.n, c.k) == (n, k)))
            .map(|(n, k)| format!("{n}x{k}"))
            .collect();
        if missing.is_empty() {
            report.push(format!(
                "model settle {:#010x}: its settle and the curves at each of its planes are \
                 carried ({}); a boot of it measures none of them",
                settled.of,
                at.join(", ")
            ));
            kept.push(settled);
        } else {
            report.push(format!(
                "WARN model settle {:#010x}: left out: no curve is stored at {} ({}); a boot of \
                 it times those curves and settles again",
                settled.of,
                missing.join(", "),
                at.join(", ")
            ));
        }
    }
    rates.dense_settled = kept;
    report
}

fn seeded(path: &str, text: &str) -> Candidate {
    Candidate {
        source: Source::Seeded(path.to_string()),
        text: text.to_string(),
    }
}

/// Where a device the derived profile serves keeps its own file.
const DERIVED_DIR: &str = "derived";

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
