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

    fn index(&self, name: &str) -> usize {
        self.vars
            .iter()
            .position(|v| v.name == name)
            .unwrap_or_else(|| panic!("`{name}` is not a declared fact"))
    }

    #[must_use]
    pub fn int(mut self, name: &str, value: i64) -> Facts {
        let i = self.index(name);
        self.values[i] = Some(value);
        self
    }

    #[must_use]
    pub fn flag(self, name: &str, value: bool) -> Facts {
        self.int(name, i64::from(value))
    }

    #[must_use]
    pub fn word(self, name: &str, value: &str) -> Facts {
        let coord = self.vars[self.index(name)]
            .domain
            .word(value)
            .unwrap_or_else(|| panic!("`{value}` is not a value of fact `{name}`"));
        self.int(name, coord)
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

    /// Whether every assumed `(fact index, coordinate)` is what the device
    /// reports.
    pub(crate) fn hold(&self, assumes: &[(usize, i64)]) -> bool {
        assumes.iter().all(|&(i, c)| self.values[i] == Some(c))
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

/// Where a layer of a profile came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// `~/.pie/profiles/<backend>/user/<id>.toml`, written by hand or by `pie tune`.
    User(String),
    /// A family or device file checked in with pie, by its path under the
    /// backend's `profiles/`.
    Seeded(String),
    /// Built by the backend from the facts, for a device no family file
    /// applies to.
    Derived,
    /// No file: every decision at its registry default.
    Defaults,
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

/// A layer offered for resolution. Layers are offered lowest first.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub source: Source,
    pub text: String,
}

/// How resolution went, for logs and doctor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Report {
    /// The topmost layer applied, or [`Source::Defaults`] when none was.
    pub used: Source,
    /// Every layer applied, lowest first.
    pub layers: Vec<Source>,
    /// Each layer refused, and why.
    pub rejected: Vec<(Source, String)>,
    /// The decisions whose value a layer measured on this device.
    pub measured: Vec<&'static str>,
}

impl Report {
    /// Whether the device runs untuned: no decision it runs was measured
    /// on it, or a layer meant for it was refused.
    #[must_use]
    pub fn untuned(&self) -> bool {
        self.measured.is_empty() || !self.rejected.is_empty()
    }

    /// The layers applied, topmost first.
    #[must_use]
    pub fn describe(&self) -> String {
        if self.layers.is_empty() {
            return Source::Defaults.to_string();
        }
        let top_first: Vec<String> = self.layers.iter().rev().map(Source::to_string).collect();
        top_first.join(" over ")
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
    /// The profile `entries` spell, every other decision at its default.
    fn assemble(
        &self,
        entries: &BTreeMap<&'static str, Vec<Rung>>,
        target: crate::registry::Target,
        source: Source,
    ) -> Profile {
        let entries = self
            .decisions
            .iter()
            .map(|(id, reg)| {
                let (rungs, defaulted) = match entries.get(id) {
                    Some(rungs) => (rungs.clone(), false),
                    None => (reg.default.clone(), true),
                };
                let (typed, read) = (reg.build)(&rungs);
                (
                    *id,
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
            target,
            entries,
        }))
    }

    /// Every decision at its registry default.
    #[must_use]
    pub fn defaults(&self) -> Profile {
        self.assemble(
            &BTreeMap::new(),
            crate::registry::Target::default(),
            Source::Defaults,
        )
    }

    /// Read a file strictly and build its profile over the defaults without
    /// checking facts: for tests and tools that hold a known-good file.
    pub fn load(&self, text: &str, origin: &str) -> Result<Profile, crate::registry::Refusal> {
        let file = self.parse(text, origin, Strict::CheckedIn)?;
        Ok(self.assemble(
            &file.entries,
            file.target,
            Source::Seeded(origin.to_string()),
        ))
    }

    /// Read a partial file, filling what it leaves out from the registry,
    /// without checking facts: for tests that state the one decision they
    /// exercise.
    pub fn load_user(&self, text: &str, origin: &str) -> Result<Profile, crate::registry::Refusal> {
        let file = self.parse(text, origin, Strict::User)?;
        Ok(self.assemble(&file.entries, file.target, Source::User(origin.to_string())))
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
            head.push_str(&format!(
                "[target]\nassumes = {}\n",
                target.assumes_toml(self.facts)
            ));
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
            &entries,
            &[],
        )?;
        for (id, rungs) in over.entries {
            entries.insert(id, rungs);
        }
        Ok(self.assemble(&entries, target, source))
    }

    /// Admit a parsed layer on the device: every fact it assumes holds,
    /// every rung whose `when` reads a fact is settled by the fact's value,
    /// and every alternative a remaining rung picks has what it requires.
    /// The entries come back settled.
    fn admits(
        &self,
        file: &File,
        facts: &Facts,
    ) -> Result<BTreeMap<&'static str, Vec<Rung>>, String> {
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
        let mut entries = BTreeMap::new();
        for (id, rungs) in &file.entries {
            let spec = self.spec(id).expect("registered");
            let settled = self.settle(spec, rungs, facts)?;
            for rung in &settled {
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
            entries.insert(*id, settled);
        }
        Ok(entries)
    }

    /// `rungs` with every fact their `when`s read replaced by its value on
    /// this device: a rung that cannot hold here, that reads a fact the
    /// device does not report, or that the rungs above it now cover, goes,
    /// and what is left of its `when` names only the decision's variables.
    /// Refused where what is left no longer picks at every shape.
    fn settle(&self, spec: &Spec, rungs: &[Rung], facts: &Facts) -> Result<Vec<Rung>, String> {
        if !rungs.iter().any(|r| {
            r.when
                .as_ref()
                .is_some_and(|w| crate::registry::reads_facts(spec, w))
        }) {
            return Ok(rungs.to_vec());
        }
        let n = spec.vars.len();
        let vars = crate::registry::when_vars(spec, self.facts);
        let space = crate::spec::space(spec.vars);
        let mut covered = crate::region::Region::nothing();
        let mut unreported = None;
        let mut out = Vec::with_capacity(rungs.len());
        let mut keep = |rung: Rung, out: &mut Vec<Rung>| {
            if rung.region.is_subset(&covered, &space) {
                return;
            }
            let partial = matches!(rung.pick, Pick::Alternative { index, .. }
                if spec.alternatives()[index].partial);
            if !partial {
                covered = covered.union(&rung.region);
            }
            out.push(rung);
        };
        'rungs: for rung in rungs {
            let Some(when) = rung
                .when
                .as_ref()
                .filter(|w| crate::registry::reads_facts(spec, w))
            else {
                keep(rung.clone(), &mut out);
                continue;
            };
            let mut region = when.region.clone();
            for (i, fact) in self.facts.iter().enumerate() {
                if !when.names.contains(&fact.name) {
                    continue;
                }
                let Some(value) = facts.values[i] else {
                    unreported = Some(fact.name);
                    continue 'rungs;
                };
                region = region.pinned(n + i, value);
            }
            let region = region.project(0..n);
            if region.is_empty() {
                continue;
            }
            let when = crate::when::render(&region, &vars[..n])
                .map(|text| When::compile(&text, &vars).expect("a rendered region compiles"));
            keep(
                Rung {
                    region,
                    when,
                    pick: rung.pick.clone(),
                },
                &mut out,
            );
        }
        if let Some(name) = unreported
            && !matches!(spec.shape, crate::spec::Shape::Flag)
            && !crate::region::Region::full(&space).is_subset(&covered, &space)
        {
            return Err(format!(
                "`{}` reads `{name}`, which this device does not report, and picks nothing without it",
                spec.id
            ));
        }
        Ok(out)
    }

    /// Resolve the profile a load runs under from its layers, lowest first:
    /// each that reads cleanly and holds on this device replaces the
    /// decisions it spells, whole; one that does not is skipped and
    /// reported, and the layers below it stand.
    #[must_use]
    pub fn resolve(&self, layers: &[Candidate], facts: &Facts) -> (Profile, Report) {
        self.resolve_with(layers, Some(facts))
    }

    /// Resolve without a device to check facts against: every layer that
    /// reads cleanly, its rungs as written. For tools that show a profile by
    /// device name.
    #[must_use]
    pub fn read(&self, layers: &[Candidate]) -> (Profile, Report) {
        self.resolve_with(layers, None)
    }

    fn resolve_with(&self, layers: &[Candidate], facts: Option<&Facts>) -> (Profile, Report) {
        let mut entries: BTreeMap<&'static str, Vec<Rung>> = BTreeMap::new();
        let mut measured: BTreeMap<&'static str, bool> = BTreeMap::new();
        let mut target = crate::registry::Target::default();
        // What the checked-in layers below assume, which a checked-in
        // device file may rely on.
        let mut inherited: Vec<(usize, i64)> = Vec::new();
        let mut used = Vec::new();
        let mut rejected = Vec::new();
        for layer in layers {
            let seeded = matches!(layer.source, Source::Seeded(_));
            let strict = if seeded {
                Strict::CheckedIn
            } else {
                Strict::User
            };
            let origin = layer.source.to_string();
            let below: &[(usize, i64)] = if seeded { &inherited } else { &[] };
            let file = match self.parse_over(&layer.text, &origin, strict, &entries, below) {
                Ok(file) => file,
                Err(refusal) => {
                    rejected.push((layer.source.clone(), refusal.to_string()));
                    continue;
                }
            };
            let spelled = match facts {
                Some(facts) => match self.admits(&file, facts) {
                    Ok(spelled) => spelled,
                    Err(why) => {
                        rejected.push((layer.source.clone(), why));
                        continue;
                    }
                },
                None => file.entries.clone(),
            };
            for id in &file.measured {
                measured.insert(id, true);
            }
            for (id, rungs) in spelled {
                measured.insert(id, file.measured.contains(&id));
                entries.insert(id, rungs);
            }
            for &(i, c) in &file.target.assumes {
                if !target.assumes.iter().any(|&(j, _)| j == i) {
                    target.assumes.push((i, c));
                }
                if seeded && !inherited.contains(&(i, c)) {
                    inherited.push((i, c));
                }
            }
            target.device = file.target.device.or(target.device);
            target.tier = file.target.tier.or(target.tier);
            if !file.target.name_contains.is_empty() {
                target.name_contains = file.target.name_contains;
            }
            used.push(layer.source.clone());
        }
        let top = used.last().cloned().unwrap_or(Source::Defaults);
        let report = Report {
            used: top.clone(),
            layers: used,
            rejected,
            measured: measured
                .into_iter()
                .filter(|(_, m)| *m)
                .map(|(id, _)| id)
                .collect(),
        };
        (self.assemble(&entries, target, top), report)
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
