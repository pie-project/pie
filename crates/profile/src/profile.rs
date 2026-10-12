//! Resolution: from candidate files and probed facts to the one profile a
//! load runs under, with a report of how it got there.

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::ladder::{Anywhere, Decision, Ladder, Pick, Rung};
use crate::registry::{File, Registry, Strict};
use crate::spec::{Domain, Spec};
use crate::when::When;

/// What a backend measured about the device it opened. A fact it could not
/// measure is absent, never guessed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Facts {
    vars: &'static [crate::spec::Var],
    values: Vec<Option<i64>>,
}

impl Facts {
    #[must_use]
    pub fn of(registry: &Registry) -> Facts {
        Facts {
            vars: registry.facts,
            values: vec![None; registry.facts.len()],
        }
    }

    fn slot(&mut self, name: &str) -> &mut Option<i64> {
        let i = self
            .vars
            .iter()
            .position(|v| v.name == name)
            .unwrap_or_else(|| panic!("`{name}` is not a declared fact"));
        &mut self.values[i]
    }

    #[must_use]
    pub fn int(mut self, name: &str, value: i64) -> Facts {
        *self.slot(name) = Some(value);
        self
    }

    #[must_use]
    pub fn flag(mut self, name: &str, value: bool) -> Facts {
        *self.slot(name) = Some(i64::from(value));
        self
    }

    #[must_use]
    pub fn word(mut self, name: &str, value: &str) -> Facts {
        let i = self
            .vars
            .iter()
            .position(|v| v.name == name)
            .expect("a declared fact");
        let coord = self.vars[i]
            .domain
            .word(value)
            .unwrap_or_else(|| panic!("`{value}` is not a value of fact `{name}`"));
        self.values[i] = Some(coord);
        self
    }

    #[must_use]
    pub fn get(&self, name: &str) -> Option<i64> {
        let i = self.vars.iter().position(|v| v.name == name)?;
        self.values[i]
    }

    /// Each fact as `name = value`, absent ones as `name = ?`.
    #[must_use]
    pub fn describe(&self) -> Vec<String> {
        self.vars
            .iter()
            .zip(&self.values)
            .map(|(v, x)| match x {
                Some(c) => format!("{} = {}", v.name, v.domain.render(*c)),
                None => format!("{} = ?", v.name),
            })
            .collect()
    }

    /// Whether `requires` holds; `None` when a fact it names is absent.
    fn satisfies(&self, requires: &When) -> Option<bool> {
        let mut point = Vec::with_capacity(self.vars.len());
        for (v, x) in self.vars.iter().zip(&self.values) {
            match x {
                Some(c) => point.push(*c),
                None if requires.names.contains(&v.name) => return None,
                None => point.push(v.domain.set().min().unwrap_or(0)),
            }
        }
        Some(requires.region.contains(&point))
    }
}

/// Where a profile came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// `~/.pie/profiles/<backend>/user/<id>.toml`, written by hand or by `pie tune`.
    User(String),
    /// A profile checked in with pie.
    Seeded(String),
    /// Built by the backend from the facts.
    Derived,
    /// No file: every decision at its registry default.
    Defaults,
}

impl Source {
    /// Whether this is a profile someone measured on this device.
    #[must_use]
    pub fn is_tuned(&self) -> bool {
        matches!(self, Source::User(_) | Source::Seeded(_))
    }
}

impl std::fmt::Display for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Source::User(p) => write!(f, "user profile {p}"),
            Source::Seeded(p) => write!(f, "checked-in profile {p}"),
            Source::Derived => write!(f, "derived profile"),
            Source::Defaults => write!(f, "registry defaults"),
        }
    }
}

/// A file offered for resolution, most preferred first.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub source: Source,
    pub text: String,
}

/// How resolution went, for logs and doctor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Report {
    pub used: Source,
    /// Each candidate refused, and why.
    pub rejected: Vec<(Source, String)>,
    /// Decisions and constants the used file left to the registry.
    pub defaulted: Vec<String>,
    /// Whether the used file says it was measured on this device: a file
    /// carried over from another device's numbers says `measured = false`
    /// in its provenance.
    pub measured: bool,
}

impl Report {
    /// Whether the device runs untuned: nothing measured on it was used.
    #[must_use]
    pub fn untuned(&self) -> bool {
        !self.used.is_tuned() || !self.rejected.is_empty() || !self.measured
    }
}

/// One member of a census: an alternative with the constants compiled into it.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Pair {
    pub decision: &'static str,
    pub alternative: &'static str,
    /// Specialised constants as `(name, TOML value)`.
    pub baked: Vec<(&'static str, String)>,
}

struct Entry {
    spec: &'static Spec,
    rungs: Vec<Rung>,
    typed: Box<dyn Any + Send + Sync>,
    defaulted: bool,
    /// Raised when the decision is evaluated at a point.
    read: Arc<std::sync::atomic::AtomicBool>,
}

struct Inner {
    backend: &'static str,
    facts: &'static [crate::spec::Var],
    source: Source,
    target: crate::registry::Target,
    entries: BTreeMap<&'static str, Entry>,
}

/// The resolved profile a load runs under. Cheap to clone.
#[derive(Clone)]
pub struct Profile(Arc<Inner>);

impl std::fmt::Debug for Profile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Profile")
            .field("backend", &self.0.backend)
            .field("source", &self.0.source)
            .finish_non_exhaustive()
    }
}

impl Profile {
    #[must_use]
    pub fn source(&self) -> &Source {
        &self.0.source
    }

    #[must_use]
    pub fn backend(&self) -> &'static str {
        self.0.backend
    }

    #[must_use]
    pub fn facts(&self) -> &'static [crate::spec::Var] {
        self.0.facts
    }

    #[must_use]
    pub fn target(&self) -> &crate::registry::Target {
        &self.0.target
    }

    fn entry(&self, id: &str) -> &Entry {
        self.0.entries.get(id).unwrap_or_else(|| {
            panic!(
                "`{id}` is not registered with the {} registry",
                self.0.backend
            )
        })
    }

    #[must_use]
    pub fn ladder<T: Decision>(&self) -> &Ladder<T> {
        self.entry(T::spec().id)
            .typed
            .downcast_ref::<Ladder<T>>()
            .expect("an entry holds the ladder of the type that registered it")
    }

    /// The value a choice or constant takes at `at`.
    #[must_use]
    pub fn pick<T: Decision>(&self, at: &T::At) -> &T {
        self.ladder::<T>()
            .at(at)
            .unwrap_or_else(|| panic!("`{}` is total by lint, yet picked nothing", T::spec().id))
    }

    /// Whether a flag holds at `at`.
    #[must_use]
    pub fn on<T: Decision>(&self, at: &T::At) -> bool {
        self.ladder::<T>().at(at).is_some()
    }

    /// Whether `id` is registered with this profile's backend.
    #[must_use]
    pub fn has(&self, id: &str) -> bool {
        self.0.entries.contains_key(id)
    }

    /// Whether a flag with no shape variables holds, by id: for a pass that
    /// walks a list of flags by name. `None` when `id` is not a registered
    /// flag without variables.
    #[must_use]
    pub fn flag(&self, id: &str) -> Option<bool> {
        let entry = self.0.entries.get(id)?;
        if !matches!(entry.spec.shape, crate::spec::Shape::Flag) || !entry.spec.vars.is_empty() {
            return None;
        }
        entry.read.store(true, std::sync::atomic::Ordering::Relaxed);
        Some(!entry.rungs.is_empty())
    }

    /// The value of a decision with no shape variables.
    #[must_use]
    pub fn get<T: Decision<At = Anywhere>>(&self) -> &T {
        self.pick::<T>(&Anywhere)
    }

    /// The decisions a load has asked anything of so far: evaluated a
    /// ladder at a point, or read a flag by id. A decision a model never
    /// reads cannot change how that model serves.
    #[must_use]
    pub fn read(&self) -> Vec<&'static str> {
        self.0
            .entries
            .iter()
            .filter(|(_, e)| e.read.load(std::sync::atomic::Ordering::Relaxed))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Every decision with its untyped rungs, in id order.
    pub fn entries(&self) -> impl Iterator<Item = (&'static Spec, &[Rung], bool)> {
        self.0
            .entries
            .values()
            .map(|e| (e.spec, e.rungs.as_slice(), e.defaulted))
    }

    /// Every (decision, alternative, specialised constants) this profile can
    /// pick: what must be compiled and tested for it.
    #[must_use]
    pub fn census(&self) -> Vec<Pair> {
        let mut out = Vec::new();
        for (spec, rungs, _) in self.entries() {
            for rung in rungs {
                if let Pick::Alternative { index, constants } = &rung.pick {
                    let alt = &spec.alternatives()[index.to_owned()];
                    let baked: Vec<(&'static str, String)> = alt
                        .constants
                        .iter()
                        .zip(constants)
                        .filter(|(c, _)| c.sort == crate::spec::Sort::Specialize)
                        .map(|(c, v)| (c.name, v.toml()))
                        .collect();
                    let item = Pair {
                        decision: spec.id,
                        alternative: alt.name,
                        baked,
                    };
                    if !out.contains(&item) {
                        out.push(item);
                    }
                }
            }
        }
        out
    }
}

impl Registry {
    fn assemble(&self, file: Option<&File>, source: Source) -> Profile {
        let empty = File {
            origin: String::new(),
            target: Default::default(),
            provenance: Vec::new(),
            entries: BTreeMap::new(),
            defaulted: Vec::new(),
        };
        let file = file.unwrap_or(&empty);
        let entries = self
            .complete(file)
            .into_iter()
            .map(|(id, (rungs, defaulted))| {
                let reg = self.registered(id).expect("complete walks the registry");
                let (typed, read) = (reg.build)(&rungs);
                (
                    id,
                    Entry {
                        spec: reg.spec,
                        rungs,
                        typed,
                        defaulted,
                        read,
                    },
                )
            })
            .collect();
        Profile(Arc::new(Inner {
            backend: self.backend,
            facts: self.facts,
            source,
            target: file.target.clone(),
            entries,
        }))
    }

    /// Every decision at its registry default.
    #[must_use]
    pub fn defaults(&self) -> Profile {
        self.assemble(None, Source::Defaults)
    }

    /// Read a file strictly and build its profile without checking facts:
    /// for tests and tools that hold a known-good file.
    pub fn load(&self, text: &str, origin: &str) -> Result<Profile, crate::registry::Refusal> {
        let file = self.parse(text, origin, Strict::CheckedIn)?;
        Ok(self.assemble(Some(&file), Source::Seeded(origin.to_string())))
    }

    /// Read a partial file, filling what it leaves out from the registry,
    /// without checking facts: for tests that state the one decision they
    /// exercise.
    pub fn load_user(&self, text: &str, origin: &str) -> Result<Profile, crate::registry::Refusal> {
        let file = self.parse(text, origin, Strict::User)?;
        Ok(self.assemble(Some(&file), Source::User(origin.to_string())))
    }

    /// The registry defaults with `overrides` (profile TOML after `api = 1`)
    /// laid over them: for a test that exercises a decision on any device.
    /// Panics on an override the registry refuses.
    #[must_use]
    pub fn with_overrides(&self, overrides: &str) -> Profile {
        self.load_user(&format!("api = 1\n{overrides}"), "test overrides")
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// `base` with the decisions `overrides` spells (profile TOML after
    /// `api = 1`) replaced, under `target`: what `pie tune` writes, before
    /// it is read back strictly.
    pub fn overlay(
        &self,
        base: &Profile,
        overrides: &str,
        target: crate::registry::Target,
        source: Source,
    ) -> Result<Profile, crate::registry::Refusal> {
        // What the overrides may require is what the target assumes.
        let mut head = String::from("api = 1\n");
        if !target.assumes.is_empty() {
            let assumed: Vec<String> = target
                .assumes
                .iter()
                .map(|&(i, c)| {
                    let f = &self.facts[i];
                    let v = match f.domain {
                        Domain::Enum(_) => format!("\"{}\"", f.domain.render(c)),
                        _ => f.domain.render(c),
                    };
                    format!("{} = {v}", f.name)
                })
                .collect();
            head.push_str(&format!("[target]\nassumes = {{ {} }}\n", assumed.join(", ")));
        }
        let mut entries: BTreeMap<&'static str, Vec<Rung>> = base
            .0
            .entries
            .iter()
            .map(|(id, e)| (*id, e.rungs.clone()))
            .collect();
        let over = self.parse_over(
            &format!("{head}{overrides}"),
            "tuned decisions",
            Strict::User,
            Some(&entries),
        )?;
        for (id, rungs) in over.entries {
            entries.insert(id, rungs);
        }
        let file = File {
            origin: String::new(),
            target,
            provenance: Vec::new(),
            entries,
            defaulted: Vec::new(),
        };
        Ok(self.assemble(Some(&file), source))
    }

    /// Check a parsed file against the device: every assumed fact matches,
    /// and every alternative it picks has what it requires.
    fn admits(&self, file: &File, facts: &Facts) -> Result<(), String> {
        for &(i, want) in &file.target.assumes {
            let fact = &self.facts[i];
            match facts.values[i] {
                Some(have) if have == want => {}
                Some(have) => {
                    return Err(format!(
                        "it assumes {} = {}, and this device has {}",
                        fact.name,
                        fact.domain.render(want),
                        fact.domain.render(have)
                    ));
                }
                None => {
                    return Err(format!(
                        "it assumes {} = {}, and this device does not report {}",
                        fact.name,
                        fact.domain.render(want),
                        fact.name
                    ));
                }
            }
        }
        for (id, rungs) in &file.entries {
            let spec = self.spec(id).expect("registered");
            for rung in rungs {
                if let Pick::Alternative { index, .. } = rung.pick {
                    let alt = &spec.alternatives()[index];
                    if alt.requires.is_empty() {
                        continue;
                    }
                    let req =
                        When::compile(alt.requires, self.facts).expect("checked at registration");
                    match facts.satisfies(&req) {
                        Some(true) => {}
                        Some(false) => {
                            return Err(format!(
                                "`{id}` picks `{}`, which requires `{}`, and this device does not have it",
                                alt.name, alt.requires
                            ));
                        }
                        None => {
                            return Err(format!(
                                "`{id}` picks `{}`, which requires `{}`, and this device does not report what it names",
                                alt.name, alt.requires
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Resolve the profile a load runs under: the first candidate that reads
    /// cleanly and holds on this device, else the registry defaults. A
    /// candidate applies whole or not at all.
    #[must_use]
    pub fn resolve(&self, candidates: &[Candidate], facts: &Facts) -> (Profile, Report) {
        self.resolve_with(candidates, Some(facts))
    }

    /// Resolve without a device to check facts against: the first candidate
    /// that reads cleanly. For tools that show a profile by device name.
    #[must_use]
    pub fn read(&self, candidates: &[Candidate]) -> (Profile, Report) {
        self.resolve_with(candidates, None)
    }

    fn resolve_with(&self, candidates: &[Candidate], facts: Option<&Facts>) -> (Profile, Report) {
        let mut rejected = Vec::new();
        for candidate in candidates {
            let strict = match candidate.source {
                Source::Seeded(_) => Strict::CheckedIn,
                _ => Strict::User,
            };
            let origin = candidate.source.to_string();
            let file = match self.parse(&candidate.text, &origin, strict) {
                Ok(file) => file,
                Err(refusal) => {
                    rejected.push((candidate.source.clone(), refusal.to_string()));
                    continue;
                }
            };
            if let Some(facts) = facts
                && let Err(why) = self.admits(&file, facts)
            {
                rejected.push((candidate.source.clone(), why));
                continue;
            }
            // A derived profile states only what its facts change; leaving
            // the rest to the registry is its design, not an omission.
            let defaulted = if candidate.source == Source::Derived {
                Vec::new()
            } else {
                file.defaulted.clone()
            };
            let report = Report {
                used: candidate.source.clone(),
                rejected,
                defaulted,
                measured: file.measured(),
            };
            return (self.assemble(Some(&file), candidate.source.clone()), report);
        }
        let report = Report {
            used: Source::Defaults,
            rejected,
            defaulted: Vec::new(),
            measured: false,
        };
        (self.defaults(), report)
    }
}

/// The coordinate of an enumerated variable's value, for building a point.
/// Panics on a name the domain does not declare: a code error.
#[must_use]
pub fn coord(domain: &Domain, name: &str) -> i64 {
    domain
        .word(name)
        .unwrap_or_else(|| panic!("`{name}` is not a value of {}", domain.describe()))
}
