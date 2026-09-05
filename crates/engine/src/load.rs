//! `LoadRequest` in, `Loaded` out: the one door a model comes through. Only
//! the `serde`-able `Trace` crosses the socket; `CompiledModel` is built on
//! the shell side by `compile(trace, budgets, profile)`.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::caps::Capabilities;

/// The ceilings a load is baked against; mirrors `model_compiler::Budget`
/// without `engine` depending on `model-compiler`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budgets {
    /// The most requests one fire may carry (`Dim::Lanes`).
    pub max_lanes: u32,
    /// The most token rows one fire may carry (`Dim::Tokens`).
    pub max_tokens: u32,
    /// The shape lattice a fire's row count is rounded up to — one immutable
    /// graph per entry. Ascending, each entry at most `max_tokens`.
    pub buckets: Vec<u32>,
    /// How many adapter banks the device pool holds.
    pub max_adapters: u32,
    /// Tokens per KV page.
    pub page_size: u32,
    /// The most tokens one sequence may hold.
    pub max_context: u32,
    /// How many sequences the pools seat at once.
    pub slots: u32,
    /// KV pages the pool holds, drawn on by every live sequence.
    pub pages: u32,
    /// The most patch rows one fire may carry. `None` (default): the shell
    /// derives a ladder from the loaded text.
    #[serde(default)]
    pub max_patches: Option<u32>,
    /// The most images one fire may carry, over every lane. `None` derives a
    /// default; not derivable from `max_patches` alone since an image
    /// contributes at least one patch row.
    #[serde(default)]
    pub max_images: Option<u32>,
}

impl Default for Budgets {
    /// A deployment that runs, for a caller who has measured nothing.
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
        }
    }
}

/// Where a load's weights come from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Checkpoint {
    /// A snapshot directory, or one container file.
    Path(PathBuf),
    /// No weights: bind the device and bake the plan, but land nothing.
    None,
}

/// How much of the weight table this load may keep, as two tier budgets:
/// T0 device (`device_weight_budget`) and T1 pinned host
/// (`host_weight_budget`). Both `None` (the default) is full residency.
///
/// A budget can only be met by holding less of what the plane can shrink
/// (e.g. streaming routed expert banks); anything else refuses with
/// [`Error::Impossible`](crate::Error::Impossible), naming both numbers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Residency {
    /// Most weight bytes this load may hold on the device (tier T0). `None`
    /// is uncapped; met by holding fewer routed experts.
    pub device_weight_budget: Option<u64>,
    /// Most weight bytes this load may hold in the pinned host cache (tier
    /// T1) — read over UVA on a device miss instead of stalling on the
    /// checkpoint. `None` is uncapped.
    pub host_weight_budget: Option<u64>,
}

/// What a planned load will hold, tier by tier: T0 device, T1 pinned host,
/// T2 mapped (whatever neither tier holds, read from the artifact — no
/// budget, since it's just a file that may or may not exist).
///
/// Does not account for the elastic pool or the safety floor; those are
/// device facts the shell tracks, not this portable statute.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Tiers {
    /// T0: the weight bytes this load will hold on the device.
    pub device: u64,
    /// T1: the weight bytes it will hold in pinned host memory.
    pub host: u64,
    /// T2: the weight bytes neither budget holds, read from a mapped source.
    pub spilled: u64,
    /// Is there a T2 source for `spilled`? `false` with nonzero `spilled` is
    /// a load that cannot be served.
    pub sourced: bool,
}

impl Residency {
    /// Both budgets uncapped: the whole table resident.
    #[must_use]
    pub const fn uncapped() -> Residency {
        Residency {
            device_weight_budget: None,
            host_weight_budget: None,
        }
    }

    /// True when neither tier is capped.
    #[must_use]
    pub const fn is_uncapped(&self) -> bool {
        self.device_weight_budget.is_none() && self.host_weight_budget.is_none()
    }

    /// Does this policy admit a checkpoint that demands these bytes
    /// resident? Delegates to [`Residency::admit_tiers`] with nothing
    /// spilled.
    ///
    /// # Errors
    ///
    /// [`Error::Impossible`](crate::Error::Impossible) when either demand is
    /// past its budget.
    pub fn admit(&self, device_demand: u64, host_demand: u64) -> crate::Result<()> {
        self.admit_tiers(Tiers {
            device: device_demand,
            host: host_demand,
            spilled: 0,
            sourced: false,
        })
    }

    /// Does this policy admit a load planned across all three tiers?
    /// Spilled bytes are admitted only if `sourced` — otherwise nothing the
    /// deployment frees can conjure a file, so this is `Impossible`, not
    /// `Exhausted`.
    ///
    /// # Errors
    ///
    /// [`Error::Impossible`](crate::Error::Impossible) for a demand past
    /// either budget, or for spilled bytes with no source to spill to.
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
            if let Some(budget) = budget {
                if demand > budget {
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
        }
        // A model bigger than device+host can still serve from a mapping;
        // only spilled bytes with nowhere to spill from refuse.
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

/// Everything a load states.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadRequest {
    /// The traced supergraph, traced by the runtime.
    pub trace: model_ir::Trace,
    /// Where the weights are.
    pub checkpoint: Checkpoint,
    /// The ceilings every fire is baked against.
    pub budgets: Budgets,
    /// How much of the weight table this load may keep resident, per tier.
    /// `#[serde(default)]` so an older request still parses as uncapped.
    #[serde(default)]
    pub residency: Residency,
    /// Which device to bind, when the shell serves more than one.
    pub ordinal: i32,
    /// How many frames the caller will keep in flight; sizes the staging
    /// ring, carved once at load. A shell clamps out-of-range rather than
    /// refusing; zero reads as one.
    pub frames_in_flight: u8,
}

/// What a load answers with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Loaded {
    /// The facts this load carries.
    pub facts: LoadFacts,
    /// What it can do.
    pub caps: Capabilities,
}

/// What came of a load, as numbers a caller can log and act on.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadFacts {
    /// The trace's own name, as the model text declared it.
    pub trace_name: String,
    /// Bytes the weight tables occupy on the device.
    pub weight_bytes: u64,
    /// Is the whole weight table device-resident? `false`: the load is
    /// streaming some tier, and `weight_bytes` is what is resident rather
    /// than what the checkpoint holds.
    pub weights_resident: bool,
    /// Did this load's weight table come off a warm-boot artifact cache
    /// instead of the checkpoint?
    #[serde(default)]
    pub weights_from_cache: bool,
    /// Bytes the activation arena occupies.
    pub arena_bytes: u64,
    /// Bytes the pools occupy.
    pub pool_bytes: u64,
    /// Bytes the resident fire inputs occupy.
    pub input_bytes: u64,
    /// Bytes of the pools actually under a physical mapping; on an elastic
    /// engine this can be less than `pool_bytes`, its reserved ceiling.
    #[serde(default)]
    pub pool_committed_bytes: u64,
    /// The most that has ever been mapped, since load — what a trim is
    /// measured against.
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
            said.contains("third tier"),
            "and which tier they wanted: {said}"
        );
        assert!(
            said.contains("pie model import"),
            "the one thing that would change the answer is a command, not a \
             differently-configured boot: {said}"
        );
        assert!(
            said.contains("uncapped"),
            "and the one case where a boot still comes into it, because a \
             deployment that has never been held whole has nothing for a \
             prepare to read its spilled planes out of: {said}"
        );
        assert!(
            matches!(refused, crate::Error::Impossible(_)),
            "statute, not exhaustion: freeing memory does not conjure a file"
        );
    }

}
