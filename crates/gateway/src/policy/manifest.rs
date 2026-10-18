use std::collections::BTreeMap;

use serde::Deserialize;

use super::CAPABILITIES;

pub const SECTION: &str = "plex.manifest";

const DEFAULT_DEADLINE_MS: u64 = 50;
const MAX_DEADLINE_MS: u64 = 1000;
/// A gate longer than this would hold every launch on a stuck program.
const MAX_GATE_MS: u64 = 1000;

/// `timer` is delivered whether or not it is listed.
const EDGES: &[&str] = &[
    "arrived",
    "exited",
    "prop-changed",
    "closed",
    "timer",
    "dispatched",
    "worker-removed",
    "worker-draining",
];

const GRANTS: &[&str] = &["schedule", "placement", "terminate"];

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct Manifest {
    pub plane: String,
    #[serde(default)]
    pub subscribe: Vec<String>,
    #[serde(default)]
    pub grants: Vec<String>,
    #[serde(default)]
    pub requires: Vec<String>,
    #[serde(default)]
    pub uses: Vec<String>,
    #[serde(default = "default_deadline_ms")]
    pub deadline_ms: u64,
    /// 0 never waits; past it the router dispatches by its default.
    #[serde(default)]
    pub gate_ms: u64,
    /// For discovery; never enforced.
    #[serde(default)]
    pub props: BTreeMap<String, PropDoc>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PropDoc {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub doc: String,
}

fn default_deadline_ms() -> u64 {
    DEFAULT_DEADLINE_MS
}

#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("the component has no `{SECTION}` section: was it exported with `policy::router!`?")]
    Missing,
    #[error("`{SECTION}` is not UTF-8")]
    NotText,
    #[error("`{SECTION}` does not parse: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("plane is `{0}`, and this host runs router policies")]
    WrongPlane(String),
    #[error("a router host has no edge `{0}`")]
    UnknownEdge(String),
    #[error("no write is granted by `{0}`")]
    UnknownGrant(String),
    #[error("a router host has no capability `{0}`")]
    UnknownCapability(String),
    #[error("the program requires `{0}`, which this gateway does not have")]
    Unmet(String),
    #[error("deadline-ms {0} is outside 1..={MAX_DEADLINE_MS}")]
    Deadline(u64),
    #[error("gate-ms {0} is over {MAX_GATE_MS}")]
    Gate(u64),
}

impl Manifest {
    pub fn of(component: &[u8], capability: &dyn Fn(&str) -> bool) -> Result<Self, ManifestError> {
        let text = section(component).ok_or(ManifestError::Missing)?;
        let text = std::str::from_utf8(text).map_err(|_| ManifestError::NotText)?;
        let manifest: Manifest = toml::from_str(text)?;
        manifest.check(capability)?;
        Ok(manifest)
    }

    fn check(&self, capability: &dyn Fn(&str) -> bool) -> Result<(), ManifestError> {
        if self.plane != "router" {
            return Err(ManifestError::WrongPlane(self.plane.clone()));
        }
        if let Some(edge) = self.subscribe.iter().find(|e| !EDGES.contains(&e.as_str())) {
            return Err(ManifestError::UnknownEdge(edge.clone()));
        }
        if let Some(grant) = self.grants.iter().find(|g| !GRANTS.contains(&g.as_str())) {
            return Err(ManifestError::UnknownGrant(grant.clone()));
        }
        for name in self.requires.iter().chain(&self.uses) {
            if !CAPABILITIES.contains(&name.as_str()) {
                return Err(ManifestError::UnknownCapability(name.clone()));
            }
        }
        if let Some(name) = self.requires.iter().find(|name| !capability(name)) {
            return Err(ManifestError::Unmet(name.clone()));
        }
        if !(1..=MAX_DEADLINE_MS).contains(&self.deadline_ms) {
            return Err(ManifestError::Deadline(self.deadline_ms));
        }
        if self.gate_ms > MAX_GATE_MS {
            return Err(ManifestError::Gate(self.gate_ms));
        }
        Ok(())
    }

    pub fn subscribes(&self, edge: &str) -> bool {
        edge == "timer" || self.subscribe.iter().any(|e| e == edge)
    }

    pub fn grants(&self, write: &str) -> bool {
        self.grants.iter().any(|g| g == write)
    }
}

fn section(component: &[u8]) -> Option<&[u8]> {
    for payload in wasmparser::Parser::new(0).parse_all(component) {
        let Ok(wasmparser::Payload::CustomSection(section)) = payload else {
            continue;
        };
        if section.name() == SECTION {
            return Some(section.data());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Result<Manifest, ManifestError> {
        let manifest: Manifest = toml::from_str(text)?;
        manifest.check(&|name| name != "router.hold")?;
        Ok(manifest)
    }

    #[test]
    fn a_minimal_router_manifest_takes_the_defaults() {
        let manifest = parse(r#"plane = "router""#).unwrap();
        assert_eq!(manifest.deadline_ms, DEFAULT_DEADLINE_MS);
        assert_eq!(manifest.gate_ms, 0, "no gate unless asked");
        assert!(manifest.subscribes("timer"));
        assert!(!manifest.subscribes("arrived"));
    }

    #[test]
    fn what_a_router_host_cannot_honour_is_refused() {
        let cases = [
            (r#"plane = "engine""#, "WrongPlane"),
            (
                "plane = \"router\"\nsubscribe = [\"started\"]",
                "UnknownEdge",
            ),
            (
                "plane = \"router\"\ngrants = [\"retention\"]",
                "UnknownGrant",
            ),
            (
                "plane = \"router\"\nuses = [\"schedule.admit\"]",
                "UnknownCapability",
            ),
            ("plane = \"router\"\nrequires = [\"router.hold\"]", "Unmet"),
            ("plane = \"router\"\ndeadline-ms = 0", "Deadline"),
            ("plane = \"router\"\ngate-ms = 5000", "Gate"),
            ("plane = \"router\"\ngate-fail = \"open\"", "Parse"),
        ];
        for (text, want) in cases {
            let got = format!("{:?}", parse(text).unwrap_err());
            assert!(got.starts_with(want), "{text:?}: {got}");
        }
    }
}
