use std::collections::BTreeMap;

use serde::Deserialize;

use super::CAPABILITIES;

pub const SECTION: &str = "plex.manifest";

const DEFAULT_DEADLINE_MS: u64 = 50;
const MAX_DEADLINE_MS: u64 = 1000;

/// `timer` is delivered whether or not it is listed.
const EDGES: &[&str] = &[
    "arrived",
    "started",
    "paused",
    "exited",
    "retained",
    "discarded",
    "prop-changed",
    "closed",
    "timer",
    "offloaded",
    "offload-failed",
];

const GRANTS: &[&str] = &[
    "schedule",
    "retention",
    "offload",
    "terminate",
    "discard",
    "publish",
];

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
    /// Cluster only: present here is an error.
    pub gate_ms: Option<u64>,
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
    #[error("the component has no `{SECTION}` section: was it exported with `policy::worker!`?")]
    Missing,
    #[error("`{SECTION}` is not UTF-8")]
    NotText,
    #[error("`{SECTION}` does not parse: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("plane is `{0}`, and this host runs worker policies")]
    WrongPlane(String),
    #[error("`{0}` is a cluster setting")]
    ClusterOnly(&'static str),
    #[error("a worker host has no edge `{0}`")]
    UnknownEdge(String),
    #[error("no write is granted by `{0}`")]
    UnknownGrant(String),
    #[error("a worker host has no capability `{0}`")]
    UnknownCapability(String),
    #[error("the program requires `{0}`, which this worker does not have")]
    Unmet(String),
    #[error("deadline-ms {0} is outside 1..={MAX_DEADLINE_MS}")]
    Deadline(u64),
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
        if self.plane != "worker" {
            return Err(ManifestError::WrongPlane(self.plane.clone()));
        }
        if self.gate_ms.is_some() {
            return Err(ManifestError::ClusterOnly("gate-ms"));
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
        manifest.check(&|name| name != "retention.tier")?;
        Ok(manifest)
    }

    #[test]
    fn a_minimal_worker_manifest_takes_the_defaults() {
        let manifest = parse(r#"plane = "worker""#).unwrap();
        assert!(manifest.subscribe.is_empty());
        assert!(manifest.grants.is_empty());
        assert_eq!(manifest.deadline_ms, DEFAULT_DEADLINE_MS);
        assert!(manifest.subscribes("timer"), "timer answers set-alarm");
        assert!(!manifest.subscribes("arrived"));
    }

    #[test]
    fn what_a_worker_host_cannot_honour_is_refused() {
        let cases = [
            (r#"plane = "cluster""#, "WrongPlane"),
            ("plane = \"worker\"\ngate-ms = 5", "ClusterOnly"),
            (
                "plane = \"worker\"\nsubscribe = [\"dispatched\"]",
                "UnknownEdge",
            ),
            (
                "plane = \"worker\"\ngrants = [\"placement\"]",
                "UnknownGrant",
            ),
            (
                "plane = \"worker\"\nuses = [\"cluster.hold\"]",
                "UnknownCapability",
            ),
            (
                "plane = \"worker\"\nrequires = [\"retention.tier\"]",
                "Unmet",
            ),
            ("plane = \"worker\"\ndeadline-ms = 0", "Deadline"),
            ("plane = \"worker\"\nsurprise = 1", "Parse"),
        ];
        for (text, want) in cases {
            let got = format!("{:?}", parse(text).unwrap_err());
            assert!(got.starts_with(want), "{text:?}: {got}");
        }
    }
}
