//! Text for people: the schema a registry declares, and a resolved profile
//! spelled back as a complete, annotated file.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::ladder::{Pick, Rung};
use crate::profile::Profile;
use crate::registry::{API, Registry};
use crate::spec::{Shape, Sort, Spec, Var};

fn vars_line(vars: &[Var]) -> String {
    if vars.is_empty() {
        return "no variables".to_string();
    }
    let dynamic: Vec<&str> = vars.iter().filter(|v| v.dynamic).map(|v| v.name).collect();
    let fixed: Vec<String> = vars
        .iter()
        .filter(|v| !v.dynamic)
        .map(|v| match v.domain {
            crate::spec::Domain::Int {
                lo: 0,
                hi: i64::MAX,
            } => v.name.to_string(),
            d => format!("{} ({})", v.name, d.describe()),
        })
        .collect();
    format!(
        "varies with {}; fixed per op: {}",
        if dynamic.is_empty() {
            "nothing".to_string()
        } else {
            dynamic.join(", ")
        },
        if fixed.is_empty() {
            "nothing".to_string()
        } else {
            fixed.join(", ")
        }
    )
}

fn doc_lines(out: &mut String, doc: &str, indent: &str) {
    for line in doc.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let _ = writeln!(out, "{indent}{line}");
    }
}

/// Every decision, its kind, variables, alternatives and constants, and its
/// default: `pie profile schema`.
#[must_use]
pub fn schema(registry: &Registry) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "{} profile schema, api {API}", registry.backend);
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "facts (target.assumes, and what alternatives require):"
    );
    for f in registry.facts {
        let _ = writeln!(out, "  {:<16} {}", f.name, f.domain.describe());
    }
    for spec in registry.specs() {
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "{}  [{}]  {}",
            spec.id,
            spec.kind_word(),
            vars_line(spec.vars)
        );
        doc_lines(&mut out, spec.doc, "  ");
        let _ = writeln!(
            out,
            "  default: {}",
            registry.default_text(spec.id).unwrap_or(spec.default)
        );
        match spec.shape {
            Shape::Constant(c) => {
                let _ = writeln!(out, "  value: {}", c.kind.describe());
            }
            Shape::Choice(alts) => {
                for alt in alts {
                    let mut tags = Vec::new();
                    if !alt.requires.is_empty() {
                        tags.push(format!("requires {}", alt.requires));
                    }
                    if alt.partial {
                        tags.push("fits some shapes".to_string());
                    }
                    for rule in alt.consistent_with {
                        tags.push(format!("with {rule}"));
                    }
                    let tags = if tags.is_empty() {
                        String::new()
                    } else {
                        format!("  ({})", tags.join("; "))
                    };
                    let _ = writeln!(out, "  use = \"{}\"{tags}", alt.name);
                    doc_lines(&mut out, alt.doc, "      ");
                    for c in alt.constants {
                        let sort = match c.sort {
                            Sort::Specialize => "compiled",
                            Sort::Launch => "per launch",
                        };
                        let _ = writeln!(
                            out,
                            "    {:<18} {}, {sort}, default {}",
                            c.name,
                            c.kind.describe(),
                            c.default
                        );
                        doc_lines(&mut out, c.doc, "      ");
                    }
                }
            }
            Shape::Flag => {}
        }
    }
    out
}

fn rung_toml(spec: &Spec, rung: &Rung, bare_ok: bool) -> String {
    let when = rung.when.as_ref().map(|w| format!("when = \"{}\"", w.text));
    match &rung.pick {
        Pick::Alternative { index, constants } => {
            let alt = &spec.alternatives()[*index];
            if bare_ok && when.is_none() && constants.is_empty() {
                return format!("\"{}\"", alt.name);
            }
            let mut parts: Vec<String> = when.into_iter().collect();
            parts.push(format!("use = \"{}\"", alt.name));
            for (c, v) in alt.constants.iter().zip(constants) {
                parts.push(format!("{} = {}", c.name, v.toml()));
            }
            format!("{{ {} }}", parts.join(", "))
        }
        Pick::Value(v) => match when {
            None if bare_ok => v.toml(),
            None => format!("{{ value = {} }}", v.toml()),
            Some(w) => format!("{{ {w}, value = {} }}", v.toml()),
        },
        Pick::On => match &rung.when {
            None => "true".to_string(),
            Some(w) => format!("\"{}\"", w.text),
        },
    }
}

/// A decision's rungs as the TOML value of its key.
#[must_use]
pub fn value_toml(spec: &Spec, rungs: &[Rung]) -> String {
    if let Shape::Flag = spec.shape {
        return match rungs.first() {
            None => "false".to_string(),
            Some(r) => rung_toml(spec, r, true),
        };
    }
    if rungs.len() == 1 && rungs[0].when.is_none() {
        return rung_toml(spec, &rungs[0], true);
    }
    let mut s = String::from("[\n");
    for r in rungs {
        let bare_ok = matches!(r.pick, Pick::Alternative { .. });
        let _ = writeln!(s, "  {},", rung_toml(spec, r, bare_ok));
    }
    s.push(']');
    s
}

fn annotation(spec: &Spec, defaulted: bool) -> String {
    let mut out = String::new();
    doc_lines(&mut out, spec.doc, "# ");
    let what = match spec.shape {
        Shape::Constant(c) => c.kind.describe(),
        Shape::Choice(alts) => format!(
            "one of {}",
            alts.iter().map(|a| a.name).collect::<Vec<_>>().join(" | ")
        ),
        Shape::Flag => "flag".to_string(),
    };
    let _ = writeln!(
        out,
        "# {what}; {}{}",
        vars_line(spec.vars),
        if defaulted { "; registry default" } else { "" }
    );
    out
}

/// [`dump`], ending in a `[provenance]` table of `(key, TOML value)`.
#[must_use]
pub(crate) fn dump_with(profile: &Profile, provenance: &[(&str, String)]) -> String {
    let mut out = dump(profile);
    if !provenance.is_empty() {
        let _ = writeln!(out);
        let _ = writeln!(out, "[provenance]");
        for (key, value) in provenance {
            let _ = writeln!(out, "{key} = {value}");
        }
    }
    out
}

/// The profile as a complete file, each entry annotated with what it may
/// hold: `pie profile dump`, the starting point for hand editing.
#[must_use]
pub fn dump(profile: &Profile) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "# {} profile, resolved from {}",
        profile.backend(),
        profile.source()
    );
    let _ = writeln!(out, "api = {API}");

    // Flags whose parent path holds only flags are written as one list at
    // that path; any other flag is written as its own key.
    let all: Vec<&'static Spec> = profile.entries().map(|(spec, _, _)| spec).collect();
    let flag_set = |parent: &str| {
        let lead = format!("{parent}.");
        let children: Vec<&&'static Spec> =
            all.iter().filter(|s| s.id.starts_with(&lead)).collect();
        !children.is_empty()
            && children
                .iter()
                .all(|s| matches!(s.shape, Shape::Flag) && !s.id[lead.len()..].contains('.'))
    };
    let mut sets: BTreeMap<String, Vec<(&'static Spec, &[Rung])>> = BTreeMap::new();
    type Entries<'p> = Vec<(&'static Spec, &'p [Rung], bool)>;
    let mut plain: BTreeMap<&str, Entries<'_>> = BTreeMap::new();
    for (spec, rungs, defaulted) in profile.entries() {
        let parent = spec.id.rsplit_once('.').map_or("", |(p, _)| p).to_string();
        if matches!(spec.shape, Shape::Flag) && flag_set(&parent) {
            sets.entry(parent).or_default().push((spec, rungs));
        } else {
            let domain = spec.id.split('.').next().unwrap_or_default();
            plain
                .entry(domain)
                .or_default()
                .push((spec, rungs, defaulted));
        }
    }
    let set_toml = |flags: &[(&'static Spec, &[Rung])]| -> (String, String) {
        let on: Vec<String> = flags
            .iter()
            .filter_map(|(spec, rungs)| {
                let name = spec.id.rsplit_once('.').map_or(spec.id, |(_, n)| n);
                rungs.first().map(|r| match &r.when {
                    None => format!("\"{name}\""),
                    Some(w) => format!("{{ {name} = \"{}\" }}", w.text),
                })
            })
            .collect();
        let off: Vec<&str> = flags
            .iter()
            .filter(|(_, r)| r.is_empty())
            .map(|(s, _)| s.id.rsplit_once('.').map_or(s.id, |(_, n)| n))
            .collect();
        let comment = if off.is_empty() {
            String::new()
        } else {
            format!("# off: {}\n", off.join(", "))
        };
        (comment, format!("[{}]", on.join(", ")))
    };

    let target = profile.target();
    // Top-level flag sets must precede the first table header.
    for (path, flags) in sets.iter().filter(|(p, _)| !p.contains('.')) {
        let (comment, list) = set_toml(flags);
        let _ = writeln!(out);
        let _ = writeln!(out, "{comment}{path} = {list}");
    }
    let _ = writeln!(out);
    let _ = writeln!(out, "[target]");
    if let Some(d) = &target.device {
        let _ = writeln!(out, "device = \"{d}\"");
    }
    if let Some(t) = &target.tier {
        let _ = writeln!(out, "tier = \"{t}\"");
    }
    if !target.assumes.is_empty() {
        let _ = writeln!(out, "assumes = {}", target.assumes_toml(profile.facts()));
    }
    let mut domains: Vec<&str> = plain.keys().copied().collect();
    for path in sets.keys().filter(|p| p.contains('.')) {
        let domain = path.split('.').next().unwrap_or_default();
        if !domains.contains(&domain) {
            domains.push(domain);
        }
    }
    domains.sort_unstable();
    for domain in domains {
        let _ = writeln!(out);
        let _ = writeln!(out, "[{domain}]");
        for (spec, rungs, defaulted) in plain.get(domain).map(Vec::as_slice).unwrap_or_default() {
            let key = &spec.id[domain.len() + 1..];
            out.push_str(&annotation(spec, *defaulted));
            let _ = writeln!(out, "{key} = {}", value_toml(spec, rungs));
        }
        for (path, flags) in sets
            .iter()
            .filter(|(p, _)| p.split('.').next() == Some(domain) && p.contains('.'))
        {
            let key = &path[domain.len() + 1..];
            let (comment, list) = set_toml(flags);
            let _ = writeln!(out, "{comment}{key} = {list}");
        }
    }
    out
}
