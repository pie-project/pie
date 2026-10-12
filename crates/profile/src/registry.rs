//! The registry: every decision a backend exposes, the TOML walker that reads
//! a profile against it, and the lints a profile must pass.

use std::any::Any;
use std::collections::BTreeMap;
use std::ops::Range;

use toml::Spanned;
use toml::de::{DeTable, DeValue};

use crate::ladder::{Decision, Ladder, Pick, Rung};
use crate::region::Region;
use crate::spec::{Alternative, Constant, Domain, Kind, Shape, Sort, Spec, Value, Var, space};
use crate::when::When;

/// The meta-model version a profile file declares as `api`.
pub const API: i64 = 1;

/// Top-level keys that are not decisions.
const RESERVED: [&str; 3] = ["api", "target", "provenance"];

/// A typed ladder, and the flag it raises when evaluated at a point.
type Build = fn(
    &[Rung],
) -> (
    Box<dyn Any + Send + Sync>,
    std::sync::Arc<std::sync::atomic::AtomicBool>,
);

pub(crate) struct Registered {
    pub(crate) spec: &'static Spec,
    pub(crate) build: Build,
    pub(crate) default: Vec<Rung>,
    /// The default as this registry spells it: the type's, or the backend's.
    pub(crate) default_text: String,
}

/// Every decision one backend exposes, with the facts its alternatives may
/// require and the recording axes its dynamic variables may name.
pub struct Registry {
    pub backend: &'static str,
    pub facts: &'static [Var],
    pub axes: &'static [&'static str],
    pub(crate) decisions: BTreeMap<&'static str, Registered>,
}

/// How strictly a file is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strict {
    /// A checked-in profile: every alternative it picks has each constant
    /// spelled out.
    CheckedIn,
    /// A user or autotune profile: a constant it leaves out takes its default.
    User,
}

/// One complaint about a file, at a line and column (both from 1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub line: usize,
    pub col: usize,
    pub message: String,
}

/// Why a file was not read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    pub origin: String,
    pub diagnostics: Vec<Diagnostic>,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, d) in self.diagnostics.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{}:{}:{}: {}", self.origin, d.line, d.col, d.message)?;
        }
        Ok(())
    }
}

impl std::error::Error for Refusal {}

/// How a file names the device it is for.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Target {
    /// The device name the backend reports: `Apple M4 Max`, `NVIDIA GeForce RTX 4090`.
    pub device: Option<String>,
    /// The CUDA tier, `sm_89`.
    pub tier: Option<String>,
    /// Device-name substrings a device file applies to, instead of `device`.
    pub name_contains: Vec<String>,
    /// Facts the file was tuned under, as (fact index, coordinate).
    pub assumes: Vec<(usize, i64)>,
}

/// A profile file read against a registry.
#[derive(Clone, Debug, Default)]
pub struct File {
    pub origin: String,
    pub target: Target,
    /// `[provenance]`, kept as written and never read by resolution.
    pub provenance: Vec<(String, String)>,
    /// `[provenance] measured`: the decisions measured on the devices this
    /// file targets. One it lists without spelling was measured, and the
    /// layer below it held. Only reporting reads it.
    pub measured: Vec<&'static str>,
    /// Decisions the file spells, by id.
    pub entries: BTreeMap<&'static str, Vec<Rung>>,
}

impl Registry {
    #[must_use]
    pub fn new(
        backend: &'static str,
        facts: &'static [Var],
        axes: &'static [&'static str],
    ) -> Registry {
        Registry {
            backend,
            facts,
            axes,
            decisions: BTreeMap::new(),
        }
    }

    /// Register `T`. Panics on a declaration the meta-model forbids: a code
    /// error any test that assembles the registry finds.
    pub fn add<T: Decision>(&mut self) -> &mut Registry {
        self.add_defaulting::<T>(T::spec().default)
    }

    /// Register `T` with this backend's own default, as TOML: for a shared
    /// decision whose safe default differs between backends.
    pub fn add_defaulting<T: Decision>(&mut self, default: &str) -> &mut Registry {
        let spec = T::spec();
        if let Err(why) = self.admit(spec) {
            panic!("decision `{}` cannot register: {why}", spec.id);
        }
        let rungs = self
            .default_rungs(spec, default)
            .unwrap_or_else(|why| panic!("decision `{}` has an unusable default: {why}", spec.id));
        self.decisions.insert(
            spec.id,
            Registered {
                spec,
                build: |rungs| {
                    let ladder = Ladder::<T>::of(rungs);
                    let read = ladder.read_flag();
                    (Box::new(ladder), read)
                },
                default: rungs,
                default_text: default.to_string(),
            },
        );
        self
    }

    /// The default `id` takes in this registry, as TOML.
    #[must_use]
    pub fn default_text(&self, id: &str) -> Option<&str> {
        self.decisions.get(id).map(|r| r.default_text.as_str())
    }

    #[must_use]
    pub fn spec(&self, id: &str) -> Option<&'static Spec> {
        self.decisions.get(id).map(|r| r.spec)
    }

    pub fn specs(&self) -> impl Iterator<Item = &'static Spec> + '_ {
        self.decisions.values().map(|r| r.spec)
    }

    pub(crate) fn registered(&self, id: &str) -> Option<&Registered> {
        self.decisions.get(id)
    }

    fn admit(&self, spec: &'static Spec) -> Result<(), String> {
        let id = spec.id;
        if !id.contains('.') || id.split('.').any(|s| s.is_empty()) {
            return Err("an id is `domain.name`".to_string());
        }
        if RESERVED.contains(&id.split('.').next().unwrap_or_default()) {
            return Err("its domain is a reserved key".to_string());
        }
        for other in self.decisions.keys() {
            if *other == id {
                return Err("the id is taken".to_string());
            }
            if other.starts_with(&format!("{id}.")) || id.starts_with(&format!("{other}.")) {
                return Err(format!("the id nests with `{other}`"));
            }
        }
        for (i, var) in spec.vars.iter().enumerate() {
            if spec.vars[..i].iter().any(|v| v.name == var.name) {
                return Err(format!("variable `{}` is declared twice", var.name));
            }
            if self.facts.iter().any(|f| f.name == var.name) {
                return Err(format!(
                    "variable `{}` is named as a fact of {}, which `when` may also read",
                    var.name, self.backend
                ));
            }
            if var.dynamic && !var.derived && !self.axes.contains(&var.name) {
                return Err(format!(
                    "`{}` is dynamic but not a recording axis of {} ({}); a dynamic variable \
                     is an axis or is derived from one",
                    var.name,
                    self.backend,
                    self.axes.join(", ")
                ));
            }
            if var.dynamic && matches!(spec.shape, Shape::Flag) {
                return Err(format!(
                    "flags act when a deployment is compiled, so `{}` cannot be dynamic",
                    var.name
                ));
            }
        }
        match spec.shape {
            Shape::Choice(alts) => {
                if alts.len() < 2 {
                    return Err("a choice needs two alternatives; one is a constant".to_string());
                }
                for (i, alt) in alts.iter().enumerate() {
                    if alts[..i].iter().any(|a| a.name == alt.name) {
                        return Err(format!("alternative `{}` is declared twice", alt.name));
                    }
                    if !alt.requires.is_empty() {
                        When::compile(alt.requires, self.facts).map_err(|e| {
                            format!("`{}` requires `{}`: {}", alt.name, alt.requires, e.message)
                        })?;
                    }
                    for c in alt.constants {
                        admit_constant(c).map_err(|e| format!("`{}.{}`: {e}", alt.name, c.name))?;
                    }
                    for rule in alt.consistent_with {
                        parse_consistent(rule).map_err(|e| format!("`{}`: {e}", alt.name))?;
                    }
                }
            }
            Shape::Constant(c) => admit_sort(&c)?,
            Shape::Flag => {}
        }
        Ok(())
    }

    fn default_rungs(&self, spec: &'static Spec, default: &str) -> Result<Vec<Rung>, String> {
        let text = format!("x = {default}");
        let mut walk = Walk::new(self, &text, Strict::CheckedIn);
        let table = DeTable::parse(&text).map_err(|e| e.message().to_string())?;
        let value = table.get_ref().iter().next().map(|(_, v)| v.clone());
        let Some(value) = value else {
            return Err("the default is empty".to_string());
        };
        let rungs = walk.entry(spec, &value);
        walk.ladder_lints(spec, &rungs, value.span().start);
        if let Some(d) = walk.diags.first() {
            return Err(d.message.clone());
        }
        for rung in &rungs {
            if rung.when.as_ref().is_some_and(|w| reads_facts(spec, w)) {
                return Err("its `when` reads a fact; a default must run anywhere".to_string());
            }
            if let Pick::Alternative { index, .. } = rung.pick {
                let alt = &spec.alternatives()[index];
                if !alt.requires.is_empty() {
                    return Err(format!(
                        "it picks `{}`, which requires `{}`; a default must run anywhere",
                        alt.name, alt.requires
                    ));
                }
            }
        }
        Ok(rungs)
    }

    /// Cross-decision checks that need the whole registry: every
    /// `consistent_with` names a registered alternative.
    #[must_use]
    pub fn check(&self) -> Vec<String> {
        let mut out = Vec::new();
        for spec in self.specs() {
            for alt in spec.alternatives() {
                for rule in alt.consistent_with {
                    let Ok((id, other)) = parse_consistent(rule) else {
                        continue;
                    };
                    match self.spec(id) {
                        None => out.push(format!(
                            "`{}` `{}` is consistent with `{rule}`, but `{id}` is not registered",
                            spec.id, alt.name
                        )),
                        Some(target) if !target.alternatives().iter().any(|a| a.name == other) => {
                            out.push(format!(
                                "`{}` `{}` is consistent with `{rule}`, but `{id}` has no `{other}`",
                                spec.id, alt.name
                            ));
                        }
                        Some(_) => {}
                    }
                }
            }
        }
        out
    }

    /// Read `text` as a profile. `origin` names it in diagnostics.
    pub fn parse(&self, text: &str, origin: &str, strict: Strict) -> Result<File, Refusal> {
        self.parse_over(text, origin, strict, &BTreeMap::new(), &[])
    }

    /// [`Registry::parse`] for a file laid over a lower layer: a decision the
    /// file leaves out is `under`'s where it has one, else the registry
    /// default, when a rule such as `consistent_with` looks at it; what an
    /// alternative requires may be promised by `inherited` assumptions as
    /// well as the file's own.
    pub(crate) fn parse_over(
        &self,
        text: &str,
        origin: &str,
        strict: Strict,
        under: &BTreeMap<&'static str, Vec<Rung>>,
        inherited: &[(usize, i64)],
    ) -> Result<File, Refusal> {
        let mut walk = Walk::new(self, text, strict);
        walk.under = under;
        walk.inherited = inherited;
        let refuse = |diagnostics| Refusal {
            origin: origin.to_string(),
            diagnostics,
        };
        let table = match DeTable::parse(text) {
            Ok(table) => table,
            Err(e) => {
                let at = e.span().map_or(0, |s| s.start);
                walk.err(at, e.message().trim().to_string());
                return Err(refuse(walk.diags));
            }
        };
        let mut file = File {
            origin: origin.to_string(),
            ..File::default()
        };
        let mut api_seen = false;
        for (key, value) in table.get_ref().iter() {
            match key.get_ref().as_ref() {
                "api" => {
                    api_seen = true;
                    match int(value.get_ref()) {
                        Some(API) => {}
                        Some(v) => walk.err(
                            value.span().start,
                            format!("this pie reads profile api {API}, the file is api {v}"),
                        ),
                        None => walk.err(value.span().start, "`api` is an integer"),
                    }
                }
                "target" => file.target = walk.target(value),
                "provenance" => match value.get_ref() {
                    DeValue::Table(t) => {
                        for (k, v) in t.iter() {
                            if k.get_ref() == "measured" {
                                file.measured = walk.measured(v);
                            }
                            let raw = &text[v.span()];
                            file.provenance
                                .push((k.get_ref().to_string(), raw.to_string()));
                        }
                    }
                    _ => walk.err(value.span().start, "`provenance` is a table"),
                },
                other => walk.node(other, value, &mut file.entries),
            }
        }
        if !api_seen {
            walk.err(0, format!("a profile starts with `api = {API}`"));
        }
        walk.file_lints(&file);
        if walk.diags.is_empty() {
            Ok(file)
        } else {
            Err(refuse(walk.diags))
        }
    }
}

impl Registry {
    /// The `[target]` of `text` alone, or `None` where it does not read:
    /// what picks the layers a device takes, before they are read whole.
    pub(crate) fn target_of(&self, text: &str) -> Option<Target> {
        let table = DeTable::parse(text).ok()?;
        let (_, value) = table
            .get_ref()
            .iter()
            .find(|(k, _)| k.get_ref() == "target")?;
        let mut walk = Walk::new(self, text, Strict::User);
        let target = walk.target(value);
        walk.diags.is_empty().then_some(target)
    }
}

impl Target {
    /// `assumes` as an inline TOML table.
    pub(crate) fn assumes_toml(&self, facts: &[Var]) -> String {
        let assumed: Vec<String> = self
            .assumes
            .iter()
            .map(|&(i, c)| {
                let f = &facts[i];
                let v = match f.domain {
                    Domain::Enum(_) => format!("\"{}\"", f.domain.render(c)),
                    _ => f.domain.render(c),
                };
                format!("{} = {v}", f.name)
            })
            .collect();
        format!("{{ {} }}", assumed.join(", "))
    }
}

fn admit_sort(c: &Constant) -> Result<(), String> {
    if c.sort == Sort::Specialize && matches!(c.kind, Kind::Int { .. } | Kind::Float { .. }) {
        return Err(
            "a specialized constant is compiled per value, so it declares a `set`, not a range"
                .to_string(),
        );
    }
    Ok(())
}

fn admit_constant(c: &Constant) -> Result<(), String> {
    admit_sort(c)?;
    let not = || format!("default `{}` is not {}", c.default, c.kind.describe());
    let v = default_literal(c)?.ok_or_else(not)?;
    if !c.kind.admits(&v) {
        return Err(not());
    }
    Ok(())
}

fn default_literal(c: &Constant) -> Result<Option<Value>, String> {
    let text = format!("x = {}", c.default);
    let table =
        DeTable::parse(&text).map_err(|e| format!("default `{}`: {}", c.default, e.message()))?;
    let (_, value) = table.get_ref().iter().next().ok_or("empty default")?;
    Ok(literal(&c.kind, value.get_ref()))
}

fn parse_consistent(rule: &str) -> Result<(&str, &str), String> {
    let (id, alt) = rule
        .split_once('=')
        .ok_or_else(|| format!("`{rule}` is not `<decision> = <alternative>`"))?;
    Ok((id.trim(), alt.trim()))
}

fn int(v: &DeValue<'_>) -> Option<i64> {
    match v {
        DeValue::Integer(i) => {
            let digits: String = i.as_str().chars().filter(|&c| c != '_').collect();
            let (sign, body) = match digits.strip_prefix('-') {
                Some(rest) => (-1, rest.to_string()),
                None => (1, digits.trim_start_matches('+').to_string()),
            };
            let body = body
                .trim_start_matches("0x")
                .trim_start_matches("0o")
                .trim_start_matches("0b")
                .to_string();
            i64::from_str_radix(&body, i.radix()).ok().map(|v| sign * v)
        }
        _ => None,
    }
}

fn literal(kind: &Kind, v: &DeValue<'_>) -> Option<Value> {
    match (kind, v) {
        (Kind::Int { .. } | Kind::Ints(_), _) => int(v).map(Value::Int),
        (Kind::Float { .. }, DeValue::Float(f)) => {
            f.as_str().replace('_', "").parse().ok().map(Value::Float)
        }
        (Kind::Float { .. }, DeValue::Integer(_)) => int(v).map(|i| Value::Float(i as f64)),
        (Kind::Bool, DeValue::Boolean(b)) => Some(Value::Bool(*b)),
        (Kind::Word(_), DeValue::String(s)) => Some(Value::Word(s.to_string())),
        (Kind::Words(_), DeValue::Array(items)) => items
            .iter()
            .map(|i| i.get_ref().as_str().map(str::to_string))
            .collect::<Option<Vec<_>>>()
            .map(Value::Words),
        _ => None,
    }
}

fn found(v: &DeValue<'_>) -> String {
    match v {
        DeValue::String(s) => format!("\"{s}\""),
        DeValue::Integer(i) => i.as_str().to_string(),
        DeValue::Float(f) => f.as_str().to_string(),
        DeValue::Boolean(b) => b.to_string(),
        other => format!("a {}", other.type_str()),
    }
}

fn names(list: impl IntoIterator<Item = &'static str>) -> String {
    let all: Vec<&str> = list.into_iter().collect();
    if all.is_empty() {
        "none".to_string()
    } else {
        all.join(", ")
    }
}

/// Whether `when` names a fact rather than only the decision's variables.
pub(crate) fn reads_facts(spec: &Spec, when: &When) -> bool {
    when.names
        .iter()
        .any(|n| !spec.vars.iter().any(|v| v.name == *n))
}

/// A decision's variables, then the backend's facts: what a `when` reads.
pub(crate) fn when_vars(spec: &Spec, facts: &[Var]) -> Vec<Var> {
    spec.vars.iter().chain(facts).copied().collect()
}

struct Walk<'a> {
    reg: &'a Registry,
    text: &'a str,
    strict: Strict,
    diags: Vec<Diagnostic>,
    /// Where each entry was spelled, for lints that point at it.
    spans: BTreeMap<&'static str, usize>,
    /// What a decision the file leaves out holds, when not the default.
    under: &'a BTreeMap<&'static str, Vec<Rung>>,
    /// Facts the layers below assume, which the file may rely on.
    inherited: &'a [(usize, i64)],
}

impl<'a> Walk<'a> {
    fn new(reg: &'a Registry, text: &'a str, strict: Strict) -> Walk<'a> {
        static NONE: BTreeMap<&'static str, Vec<Rung>> = BTreeMap::new();
        Walk {
            reg,
            text,
            strict,
            diags: Vec::new(),
            spans: BTreeMap::new(),
            under: &NONE,
            inherited: &[],
        }
    }

    fn measured(&mut self, v: &Spanned<DeValue<'_>>) -> Vec<&'static str> {
        let DeValue::Array(items) = v.get_ref() else {
            self.err(
                v.span().start,
                "`provenance.measured` is a list of decision ids",
            );
            return Vec::new();
        };
        let mut out = Vec::new();
        for item in items.iter() {
            let Some(id) = item.get_ref().as_str() else {
                self.err(
                    item.span().start,
                    "`provenance.measured` is a list of decision ids",
                );
                continue;
            };
            match self.reg.decisions.get_key_value(id) {
                Some((id, _)) if out.contains(id) => {
                    self.err(item.span().start, format!("`{id}` is listed twice"));
                }
                Some((id, _)) => out.push(*id),
                None => self.err(
                    item.span().start,
                    format!("`{id}` is not a decision of {}", self.reg.backend),
                ),
            }
        }
        out
    }

    fn err(&mut self, at: usize, message: impl Into<String>) {
        let at = at.min(self.text.len());
        let before = &self.text[..at];
        let line = before.matches('\n').count() + 1;
        let col = before.rsplit('\n').next().map_or(0, |l| l.chars().count()) + 1;
        self.diags.push(Diagnostic {
            line,
            col,
            message: message.into(),
        });
    }

    /// Ids registered under `prefix.`, one segment deeper, for "did you mean".
    fn children(&self, prefix: &str) -> Vec<&'static str> {
        let lead = if prefix.is_empty() {
            String::new()
        } else {
            format!("{prefix}.")
        };
        let mut out: Vec<&'static str> = self
            .reg
            .decisions
            .keys()
            .filter_map(|id| id.strip_prefix(lead.as_str()))
            .map(|rest| rest.split('.').next().unwrap_or(rest))
            .collect();
        out.dedup();
        out
    }

    fn is_flag_set(&self, path: &str) -> bool {
        let lead = format!("{path}.");
        let mut any = false;
        for (id, reg) in &self.reg.decisions {
            if let Some(rest) = id.strip_prefix(lead.as_str()) {
                if rest.contains('.') || !matches!(reg.spec.shape, Shape::Flag) {
                    return false;
                }
                any = true;
            }
        }
        any
    }

    fn node(
        &mut self,
        path: &str,
        value: &Spanned<DeValue<'_>>,
        out: &mut BTreeMap<&'static str, Vec<Rung>>,
    ) {
        if let Some(reg) = self.reg.registered(path) {
            let spec = reg.spec;
            self.spans.insert(spec.id, value.span().start);
            let rungs = self.entry(spec, value);
            self.ladder_lints(spec, &rungs, value.span().start);
            out.insert(spec.id, rungs);
            return;
        }
        if let DeValue::Array(items) = value.get_ref()
            && self.is_flag_set(path)
        {
            self.flag_set(path, value.span(), items, out);
            return;
        }
        if let DeValue::Table(t) = value.get_ref()
            && !self.children(path).is_empty()
        {
            for (key, child) in t.iter() {
                let sub = format!("{path}.{}", key.get_ref());
                self.node(&sub, child, out);
            }
            return;
        }
        let (parent, _) = path.rsplit_once('.').unwrap_or(("", path));
        let siblings = self.children(parent);
        let place = if parent.is_empty() {
            format!("domains of {}: {}", self.reg.backend, names(siblings))
        } else {
            format!("under `{parent}`: {}", names(siblings))
        };
        self.err(
            value.span().start,
            format!(
                "`{path}` is not a decision of {}; {place}",
                self.reg.backend
            ),
        );
    }

    fn flag_set(
        &mut self,
        path: &str,
        span: Range<usize>,
        items: &[Spanned<DeValue<'_>>],
        out: &mut BTreeMap<&'static str, Vec<Rung>>,
    ) {
        let lead = format!("{path}.");
        let flags: Vec<&'static Spec> = self
            .reg
            .decisions
            .iter()
            .filter(|(id, _)| id.starts_with(lead.as_str()))
            .map(|(_, r)| r.spec)
            .collect();
        for spec in &flags {
            self.spans.insert(spec.id, span.start);
            out.insert(spec.id, Vec::new());
        }
        let known = || names(flags.iter().map(|s| &s.id[lead.len()..]));
        for item in items {
            let (name, at, when) = match item.get_ref() {
                DeValue::String(s) => (s.to_string(), item.span().start, None),
                DeValue::Table(t) if t.len() == 1 => {
                    let (k, v) = t.iter().next().expect("one entry");
                    (k.get_ref().to_string(), k.span().start, Some(v))
                }
                _ => {
                    self.err(
                        item.span().start,
                        "a flag is a name, or `{ name = \"<when>\" }`",
                    );
                    continue;
                }
            };
            let id = format!("{lead}{name}");
            let Some(spec) = flags.iter().find(|s| s.id == id) else {
                self.err(
                    at,
                    format!("`{name}` is not a flag of `{path}`; declared: {}", known()),
                );
                continue;
            };
            if out.get(spec.id).is_some_and(|r| !r.is_empty()) {
                self.err(at, format!("`{name}` is listed twice"));
                continue;
            }
            let rung = match when {
                None => Some(self.always(Pick::On, spec)),
                Some(v) => self.when_rung(spec, v, Pick::On),
            };
            if let Some(rung) = rung {
                out.insert(spec.id, vec![rung]);
            }
        }
    }

    fn always(&self, pick: Pick, spec: &Spec) -> Rung {
        Rung {
            when: None,
            region: Region::full(&space(spec.vars)),
            pick,
        }
    }

    fn compile_when(&mut self, spec: &Spec, v: &Spanned<DeValue<'_>>) -> Option<When> {
        let Some(text) = v.get_ref().as_str() else {
            self.err(
                v.span().start,
                format!("`when` is a string, found {}", found(v.get_ref())),
            );
            return None;
        };
        match When::compile(text, &when_vars(spec, self.reg.facts)) {
            Ok(w) => Some(w),
            Err(e) => {
                self.err(
                    v.span().start + 1 + e.at,
                    format!("{} (in `{}`)", e.message, spec.id),
                );
                None
            }
        }
    }

    fn when_rung(&mut self, spec: &Spec, v: &Spanned<DeValue<'_>>, pick: Pick) -> Option<Rung> {
        let when = self.compile_when(spec, v)?;
        Some(Rung {
            region: when.region.project(0..spec.vars.len()),
            when: Some(when),
            pick,
        })
    }

    /// One decision's value: a scalar, a table, or a ladder.
    fn entry(&mut self, spec: &'static Spec, value: &Spanned<DeValue<'_>>) -> Vec<Rung> {
        match (spec.shape, value.get_ref()) {
            (Shape::Flag, DeValue::Boolean(true)) => vec![self.always(Pick::On, spec)],
            (Shape::Flag, DeValue::Boolean(false)) => Vec::new(),
            (Shape::Flag, DeValue::String(_)) => {
                self.when_rung(spec, value, Pick::On).into_iter().collect()
            }
            (Shape::Flag, other) => {
                self.err(
                    value.span().start,
                    format!(
                        "flag `{}` is true, false or a `when`, found {}",
                        spec.id,
                        found(other)
                    ),
                );
                Vec::new()
            }
            (_, DeValue::Array(items))
                if !matches!(
                    spec.shape,
                    Shape::Constant(Constant {
                        kind: Kind::Words(_),
                        ..
                    })
                ) =>
            {
                if items.is_empty() {
                    self.err(
                        value.span().start,
                        format!("`{}` is an empty ladder", spec.id),
                    );
                }
                items
                    .iter()
                    .filter_map(|item| self.rung(spec, item, true))
                    .collect()
            }
            _ => self.rung(spec, value, false).into_iter().collect(),
        }
    }

    fn rung(
        &mut self,
        spec: &'static Spec,
        v: &Spanned<DeValue<'_>>,
        in_ladder: bool,
    ) -> Option<Rung> {
        match spec.shape {
            Shape::Constant(c) => {
                if let (true, DeValue::Table(t)) = (in_ladder, v.get_ref()) {
                    let mut when = None;
                    let mut value = None;
                    for (k, x) in t.iter() {
                        match k.get_ref().as_ref() {
                            "when" => when = Some(x),
                            "value" => value = Some(x),
                            other => {
                                self.err(
                                    k.span().start,
                                    format!(
                                        "a constant's rung holds `when` and `value`, not `{other}`"
                                    ),
                                );
                                return None;
                            }
                        }
                    }
                    let Some(value) = value else {
                        self.err(v.span().start, "this rung has no `value`");
                        return None;
                    };
                    let value = self.constant(spec.id, &c, value)?;
                    return match when {
                        Some(w) => self.when_rung(spec, w, Pick::Value(value)),
                        None => Some(self.always(Pick::Value(value), spec)),
                    };
                }
                let value = self.constant(spec.id, &c, v)?;
                Some(self.always(Pick::Value(value), spec))
            }
            Shape::Choice(alts) => {
                let (name, name_at, table) = match v.get_ref() {
                    DeValue::String(s) => (s.to_string(), v.span().start, None),
                    DeValue::Table(t) => {
                        let Some((_, u)) = t.iter().find(|(k, _)| k.get_ref() == "use") else {
                            self.err(
                                v.span().start,
                                format!("this rung of `{}` names no `use`", spec.id),
                            );
                            return None;
                        };
                        let Some(s) = u.get_ref().as_str() else {
                            self.err(u.span().start, "`use` is a string");
                            return None;
                        };
                        (s.to_string(), u.span().start, Some(t))
                    }
                    other => {
                        self.err(
                            v.span().start,
                            format!(
                                "`{}` is an alternative name or a table, found {}",
                                spec.id,
                                found(other)
                            ),
                        );
                        return None;
                    }
                };
                let Some(index) = alts.iter().position(|a| a.name == name) else {
                    self.err(
                        name_at,
                        format!(
                            "`{name}` is not an alternative of `{}`; declared: {}",
                            spec.id,
                            names(alts.iter().map(|a| a.name))
                        ),
                    );
                    return None;
                };
                let alt = &alts[index];
                let mut constants: Vec<Option<Value>> = vec![None; alt.constants.len()];
                let mut when = None;
                let mut bad = false;
                if let Some(t) = table {
                    for (k, x) in t.iter() {
                        let key = k.get_ref().as_ref();
                        match key {
                            "use" => {}
                            "when" => when = Some(x),
                            _ => match alt.constants.iter().position(|c| c.name == key) {
                                Some(i) => match self.constant(
                                    &format!("{}.{}", spec.id, name),
                                    &alt.constants[i],
                                    x,
                                ) {
                                    Some(v) => constants[i] = Some(v),
                                    None => bad = true,
                                },
                                None => {
                                    self.err(
                                        k.span().start,
                                        format!(
                                            "`{key}` is not a constant of `{name}`; declared: {}",
                                            names(alt.constants.iter().map(|c| c.name))
                                        ),
                                    );
                                    return None;
                                }
                            },
                        }
                    }
                }
                if bad {
                    return None;
                }
                let mut filled = Vec::with_capacity(alt.constants.len());
                for (c, v) in alt.constants.iter().zip(constants) {
                    match v {
                        Some(v) => filled.push(v),
                        None => {
                            if self.strict == Strict::CheckedIn {
                                self.err(name_at, format!("`{name}` leaves `{}` unspecified; a checked-in profile spells every constant", c.name));
                            }
                            filled.push(constant_default(c));
                        }
                    }
                }
                let pick = Pick::Alternative {
                    index,
                    constants: filled,
                };
                match when {
                    Some(w) => self.when_rung(spec, w, pick),
                    None => Some(self.always(pick, spec)),
                }
            }
            Shape::Flag => unreachable!("flags are read by `entry`"),
        }
    }

    fn constant(&mut self, owner: &str, c: &Constant, v: &Spanned<DeValue<'_>>) -> Option<Value> {
        match literal(&c.kind, v.get_ref()) {
            Some(value) if c.kind.admits(&value) => Some(value),
            _ => {
                self.err(
                    v.span().start,
                    format!(
                        "`{owner}.{}` is {}, found {}",
                        c.name,
                        c.kind.describe(),
                        found(v.get_ref())
                    ),
                );
                None
            }
        }
    }

    fn target(&mut self, value: &Spanned<DeValue<'_>>) -> Target {
        let mut target = Target::default();
        let DeValue::Table(t) = value.get_ref() else {
            self.err(value.span().start, "`target` is a table");
            return target;
        };
        let text = |w: &mut Walk<'_>, v: &Spanned<DeValue<'_>>, what: &str| -> Option<String> {
            match v.get_ref().as_str() {
                Some(s) => Some(s.to_string()),
                None => {
                    w.err(v.span().start, format!("`target.{what}` is a string"));
                    None
                }
            }
        };
        for (k, v) in t.iter() {
            match k.get_ref().as_ref() {
                "device" => target.device = text(self, v, "device"),
                "tier" => target.tier = text(self, v, "tier"),
                "match" => {
                    let DeValue::Table(m) = v.get_ref() else {
                        self.err(v.span().start, "`target.match` is a table");
                        continue;
                    };
                    for (mk, mv) in m.iter() {
                        match mk.get_ref().as_ref() {
                            "name_contains" => match mv.get_ref() {
                                DeValue::Array(items) => {
                                    for i in items.iter() {
                                        if let Some(s) = text(self, i, "match.name_contains") {
                                            target.name_contains.push(s);
                                        }
                                    }
                                }
                                _ => self.err(
                                    mv.span().start,
                                    "`target.match.name_contains` is a list of strings",
                                ),
                            },
                            other => self.err(
                                mk.span().start,
                                format!("`target.match` holds `name_contains`, not `{other}`"),
                            ),
                        }
                    }
                }
                "assumes" => {
                    let DeValue::Table(a) = v.get_ref() else {
                        self.err(v.span().start, "`target.assumes` is a table of facts");
                        continue;
                    };
                    for (fk, fv) in a.iter() {
                        let name = fk.get_ref().as_ref();
                        let Some(index) = self.reg.facts.iter().position(|f| f.name == name) else {
                            self.err(
                                fk.span().start,
                                format!(
                                    "`{name}` is not a fact of {}; declared: {}",
                                    self.reg.backend,
                                    names(self.reg.facts.iter().map(|f| f.name))
                                ),
                            );
                            continue;
                        };
                        let fact = self.reg.facts[index];
                        let coord = match (fact.domain, fv.get_ref()) {
                            (Domain::Bool, DeValue::Boolean(b)) => Some(i64::from(*b)),
                            (Domain::Enum(_), DeValue::String(s)) => fact.domain.word(s),
                            (Domain::Int { .. } | Domain::Ints(_), x) => int(x),
                            _ => None,
                        }
                        .filter(|c| fact.domain.set().contains(*c));
                        match coord {
                            Some(c) => target.assumes.push((index, c)),
                            None => self.err(
                                fv.span().start,
                                format!(
                                    "fact `{name}` is {}, found {}",
                                    fact.domain.describe(),
                                    found(fv.get_ref())
                                ),
                            ),
                        }
                    }
                }
                other => self.err(
                    k.span().start,
                    format!("`target` holds device, tier, match and assumes, not `{other}`"),
                ),
            }
        }
        target
    }

    /// Lints on one decision's rungs: total, every rung reachable, over the
    /// decision's variables and the facts its `when`s may read.
    fn ladder_lints(&mut self, spec: &Spec, rungs: &[Rung], at: usize) {
        if let Shape::Flag = spec.shape {
            for rung in rungs {
                if let Some(w) = &rung.when
                    && let Some(name) = w
                        .names
                        .iter()
                        .find(|n| spec.vars.iter().any(|v| v.name == **n && v.dynamic))
                {
                    self.err(
                        at,
                        format!(
                            "flag `{}` acts at compile time and cannot read `{name}`",
                            spec.id
                        ),
                    );
                }
            }
            return;
        }
        let vars = when_vars(spec, self.reg.facts);
        let space = space(&vars);
        let alts = spec.alternatives();
        let partial =
            |r: &Rung| matches!(r.pick, Pick::Alternative { index, .. } if alts[index].partial);
        let mut covered = Region::nothing();
        for (i, rung) in rungs.iter().enumerate() {
            let region = held(rung, spec.vars.len(), &space);
            if region.is_empty() {
                self.err(
                    at,
                    format!("rung {} of `{}` holds no shape", i + 1, spec.id),
                );
                continue;
            }
            if region.is_subset(&covered, &space) {
                self.err(
                    at,
                    format!("rung {} of `{}` is unreachable: the rungs above it hold every shape it does", i + 1, spec.id),
                );
            }
            if !partial(rung) {
                covered = covered.union(&region);
            }
        }
        if rungs.is_empty() {
            return;
        }
        let gap = Region::full(&space).minus(&covered, &space);
        if !gap.is_empty() {
            let shown = crate::when::render(&gap, &vars).unwrap_or_default();
            let tail = if rungs.last().is_some_and(partial) {
                ", and it ends on an alternative that fits only some shapes"
            } else {
                ""
            };
            self.err(
                at,
                format!("`{}` picks nothing where `{shown}`{tail}", spec.id),
            );
        }
    }

    /// Lints across a whole file: requires against assumes (the file's and
    /// those it inherits, narrowed by any fact a rung's `when` reads), and
    /// consistency between decisions.
    fn file_lints(&mut self, file: &File) {
        let facts = self.reg.facts;
        let fact_space = space(facts);
        let mut assumed = fact_space.clone();
        for &(i, c) in self.inherited.iter().chain(&file.target.assumes) {
            assumed[i] = assumed[i].intersect(&crate::region::Set::point(c));
        }
        let assumed = Region::full(&assumed);
        let mut picks: Vec<(&'static str, &'static Alternative, Region)> = Vec::new();
        for (id, rungs) in &file.entries {
            let spec = self.reg.spec(id).expect("entries are registered");
            let n = spec.vars.len();
            let space = space(&when_vars(spec, facts));
            for rung in rungs {
                if let Pick::Alternative { index, .. } = rung.pick {
                    let on = held(rung, n, &space).project(n..n + facts.len());
                    picks.push((*id, &spec.alternatives()[index], on));
                }
            }
        }
        for (id, alt, on) in picks {
            let alt_name = alt.name;
            let at = self.spans.get(id).copied().unwrap_or(0);
            if !alt.requires.is_empty() {
                let req = When::compile(alt.requires, facts).expect("checked at registration");
                if !assumed.intersect(&on).is_subset(&req.region, &fact_space) {
                    self.err(
                        at,
                        format!(
                            "`{id}` picks `{alt_name}`, which requires `{}`; `target.assumes` does not promise it",
                            alt.requires
                        ),
                    );
                }
            }
            for rule in alt.consistent_with {
                let Ok((other_id, other_alt)) = parse_consistent(rule) else {
                    continue;
                };
                let other_has = match file
                    .entries
                    .get(other_id)
                    .or_else(|| self.under.get(other_id))
                {
                    Some(rungs) => {
                        let other = self.reg.spec(other_id);
                        rungs.iter().any(|r| {
                            matches!(r.pick, Pick::Alternative { index, .. }
                            if other.is_some_and(|s| s.alternatives()[index].name == other_alt))
                        })
                    }
                    None => self.reg.registered(other_id).is_some_and(|reg| {
                        reg.default.iter().any(|r| {
                            matches!(r.pick, Pick::Alternative { index, .. }
                            if reg.spec.alternatives()[index].name == other_alt)
                        })
                    }),
                };
                if !other_has {
                    self.err(at, format!("`{id}` picks `{alt_name}`, which needs `{rule}`, and no rung of `{other_id}` picks `{other_alt}`"));
                }
            }
        }
    }
}

/// Where `rung` holds over a decision's `n` variables and then the facts
/// (`space` spans both).
pub(crate) fn held(rung: &Rung, n: usize, space: &[crate::region::Set]) -> Region {
    match &rung.when {
        Some(w) => w.region.clone(),
        None => rung.region.lifted(&space[n..]),
    }
}

fn constant_default(c: &Constant) -> Value {
    default_literal(c)
        .ok()
        .flatten()
        .expect("checked at registration")
}
