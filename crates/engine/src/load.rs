use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::caps::Capabilities;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budgets {
    pub max_lanes: u32,
    pub max_tokens: u32,
    pub buckets: Vec<u32>,
    pub max_adapters: u32,
    pub page_size: u32,
    pub max_context: u32,
    pub slots: u32,
    pub pages: u32,
    #[serde(default)]
    pub max_patches: Option<u32>,
    #[serde(default)]
    pub max_images: Option<u32>,
    #[serde(default)]
    pub max_voxels: Option<u32>,
    #[serde(default)]
    pub max_clips: Option<u32>,
}

impl Default for Budgets {
    fn default() -> Budgets {
        Budgets {
            max_lanes: 256,
            max_tokens: 8192,
            buckets: Vec::new(),
            max_adapters: 0,
            page_size: 16,
            max_context: 4096,
            slots: 256,
            pages: 65536,
            max_patches: None,
            max_images: None,
            max_voxels: None,
            max_clips: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Checkpoint {
    Path(PathBuf),
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Residency {
    pub device_weight_budget: Option<u64>,
    pub host_weight_budget: Option<u64>,
    #[serde(default = "deferred_by_default")]
    pub deferred_tier: bool,
}

fn deferred_by_default() -> bool {
    true
}

impl Default for Residency {
    fn default() -> Residency {
        Residency::uncapped()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Tiers {
    pub device: u64,
    pub host: u64,
    pub spilled: u64,
    pub sourced: bool,
}

impl Residency {
    #[must_use]
    pub const fn uncapped() -> Residency {
        Residency {
            device_weight_budget: None,
            host_weight_budget: None,
            deferred_tier: true,
        }
    }

    #[must_use]
    pub const fn is_uncapped(&self) -> bool {
        self.device_weight_budget.is_none() && self.host_weight_budget.is_none()
    }

    pub fn admit(&self, device_demand: u64, host_demand: u64) -> crate::Result<()> {
        self.admit_tiers(Tiers {
            device: device_demand,
            host: host_demand,
            spilled: 0,
            sourced: false,
        })
    }

    pub fn admit_tiers(&self, tiers: Tiers) -> crate::Result<()> {
        for (budget, demand, tier, field) in [
            (
                self.device_weight_budget,
                tiers.device,
                "device",
                "device_weight_budget",
            ),
            (
                self.host_weight_budget,
                tiers.host,
                "pinned host",
                "host_weight_budget",
            ),
        ] {
            if let Some(budget) = budget
                && demand > budget
            {
                return Err(crate::Error::Impossible(format!(
                    "weight residency: `{field}` is {budget} bytes and this load demands \
                         {demand} bytes on the {tier} tier. That demand is what the engine \
                         has already reduced to as far as its tiers allow — routed expert \
                         banks stream and dense planes rotate through a ring, and what is \
                         left is what must stay resident — so the budget \
                         cannot be met by holding less of it. Raise the budget, or state \
                         `None` for uncapped."
                )));
            }
        }
        if tiers.spilled > 0 && !tiers.sourced {
            return Err(crate::Error::Impossible(format!(
                "a streamed plan spills {} bytes and this deployment has no source for them. \
                 The model's own `.zt` IS that source — `pie model import` writes every \
                 plane of the trace into it at a budget-free ranking, and a serve reads \
                 what the budgets cut out of it. So: import this checkpoint on this box, \
                 or raise one of the budgets, or state `None`.",
                tiers.spilled,
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadRequest {
    pub trace: model_ir::Trace,
    pub checkpoint: Checkpoint,
    pub budgets: Budgets,
    #[serde(default)]
    pub residency: Residency,
    pub ordinal: i32,
    pub frames_in_flight: u8,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Loaded {
    pub facts: LoadFacts,
    pub caps: Capabilities,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadFacts {
    pub trace_name: String,
    pub weight_bytes: u64,
    pub weights_resident: bool,
    #[serde(default)]
    pub weights_from_cache: bool,
    pub arena_bytes: u64,
    pub pool_bytes: u64,
    pub input_bytes: u64,
    #[serde(default)]
    pub pool_committed_bytes: u64,
    #[serde(default)]
    pub pool_high_water_bytes: u64,
}

#[cfg(test)]
mod residency_tests {
    use super::{Residency, Tiers};

    fn capped(device: u64, host: u64) -> Residency {
        Residency {
            device_weight_budget: Some(device),
            host_weight_budget: Some(host),
            ..Residency::uncapped()
        }
    }

    #[test]
    fn spilled_bytes_with_a_source_are_admitted_and_without_one_are_impossible() {
        let policy = capped(1_000, 500);
        let planned = |spilled, sourced| Tiers {
            device: 1_000,
            host: 500,
            spilled,
            sourced,
        };
        assert!(
            policy.admit_tiers(planned(4_000, true)).is_ok(),
            "bytes neither budget holds are SERVED when a source holds them — \
             that sentence is streaming §2's reason to exist"
        );
        let refused = policy
            .admit_tiers(planned(4_000, false))
            .expect_err("and refused when nothing does");
        let said = format!("{refused}");
        assert!(said.contains("4000"), "the refusal names the bytes: {said}");
        assert!(
            said.contains("spills"),
            "and what the plan wanted of them — the tier the budgets cut out, \
             which the refusal names by what it does rather than by a number: \
             {said}"
        );
        assert!(
            said.contains("pie model import"),
            "the one thing that would change the answer is a command, not a \
             differently-configured boot: {said}"
        );
        assert!(
            said.contains("state `None`"),
            "and the one case where a boot still comes into it — an uncapped \
             budget — because a deployment that has never been held whole has \
             nothing for a prepare to read its spilled planes out of: {said}"
        );
        assert!(
            matches!(refused, crate::Error::Impossible(_)),
            "statute, not exhaustion: freeing memory does not conjure a file"
        );
    }
}
