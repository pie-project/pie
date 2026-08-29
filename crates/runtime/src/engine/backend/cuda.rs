//! Opening a CUDA device from a boot config.
//!
//! # What this file stopped being
//!
//! It was 415 lines of adapter: a `CudaEngineHandle` per rank, an
//! `impl Engine for CudaEngine` forwarding fourteen verbs to a leader shell,
//! a `status(Result<T, i32>, verb)` translating the shell's own `i32` ladder,
//! a `CompletionBroker` per handle, and a `load_model` that fanned a
//! `Vec<ModelLoadDesc>` across `std::thread::scope`.
//!
//! Every one of those was a shape the palo rewrite dissolved:
//!
//! * **The `Engine` impl is the shell's.** `engine_cuda::Cuda` implements the
//!   contract, in the crate that owns the device — which is what decision 13
//!   ("no `trait Backend`; shells are thin call-order crates") means when the
//!   trait in question is the contract itself. This module selects a device;
//!   it does not adapt one.
//! * **The `i32` ladder is gone** with `PIE_STATUS_*` (design §7). The shell
//!   answers a typed `Fault` and `engine_cuda::api` maps it to `Error`
//!   once.
//! * **The broker is the runtime's** (`crate::engine::completion`), so a
//!   per-handle one here was the wrong side of the boundary twice over.
//! * **A rank is not a load.** `load_model(Vec<ModelLoadDesc>)` shipped one
//!   descriptor per rank and cross-checked that they agreed;
//!   [`LoadRequest`](engine_api::LoadRequest) is one plan, `Shard::Cut` is in
//!   the plan, and which rank a shell is, is the shell's own.
//!
//! # What is left, and it is the whole file
//!
//! Read the boot TOML for the two things it says about the MACHINE — which
//! device, and how much of a fire to record — and hand the shell the load
//! door (`crate::engine::load::contract_for`) it cannot state for itself.

use anyhow::{Result, anyhow};
use engine_cuda::{Cuda, DeviceBoot, Graphs};

/// Open one device from a boot document.
///
/// # Errors
///
/// A boot document that is not TOML. Binding the device itself happens at
/// [`Engine::load`](engine_api::Engine::load), not here: `Shell::load` is one
/// call that binds, bakes and lands, and there is nothing to bind before a
/// plan says what to bake.
pub fn open(config_bytes: &[u8]) -> Result<Cuda> {
    let doc: toml::Table = std::str::from_utf8(config_bytes)
        .map_err(|error| anyhow!("the cuda boot config is not utf-8: {error}"))?
        .parse()
        .map_err(|error| anyhow!("the cuda boot config is not TOML: {error}"))?;
    Ok(Cuda::new(
        DeviceBoot {
            ordinal: ordinal(&doc),
            graphs: graphs(&doc),
        },
        crate::engine::load::contract_for,
    ))
}

/// How many descriptor-port envelopes the CUDA shell has resolved off guest
/// device rings in this process (`palo B3`).
///
/// **THE ONE OBSERVABLE OF A NEGATIVE.** Device-carried decode's whole claim
/// is that a chained fire's token did not travel to the host, and a round
/// trip that does not happen leaves no trace. What DOES happen is one
/// envelope resolved per attached device-carried lane per fire, so a serving
/// gate asserts on this: zero says every decode serialized through the host
/// plane, `>= decodes` says the shell read the token off the ring the
/// epilogue wrote.
///
/// Re-exported here rather than reached for directly because `engine-cuda` is
/// a private link of this crate — `_engine-cuda` is what gates it — and a
/// test that named the shell crate would be a test that could not build
/// without a GPU feature it does not select.
#[must_use]
pub fn envelopes_resolved() -> u64 {
    engine_cuda::Shell::envelopes_resolved()
}

/// The CUDA fold's motion counters —
/// `(folds, rebinds, rebind_us, swaps, prebinds, prebind_us, twins)` —
/// re-exported on `envelopes_resolved`'s argument exactly: the shell is a
/// private link of this crate, the instance lives behind `Box<dyn Engine>`
/// on a scheduler lane thread, and what a runtime-level fold gate diffs is
/// this process-global mirror before and after a serving loop. `prebinds`
/// moving is the one observable that the runtime's own next-fire hint
/// (`Engine::expect_fire`, stated from `scheduler::worker::fire_frame`)
/// reached the shell — nothing else in the runtime can say so, because a
/// hint that lands leaves no trace in any completion. The two micros
/// columns split the same binding work into its on-critical-path and
/// hidden halves, which is the number the `PIE_CUDA_PIPELINE` A/B moves.
#[must_use]
pub fn fold_observed() -> (u64, u64, u64, u64, u64, u64, u64) {
    engine_cuda::Shell::fold_observed()
}

/// Which device `[model] device` names.
///
/// `"cuda:1"`, `"1"` and an absent key all mean something, and the third
/// means zero — a single-GPU box is the deployment that writes the least.
fn ordinal(doc: &toml::Table) -> i32 {
    doc.get("model")
        .and_then(toml::Value::as_table)
        .and_then(|model| model.get("device"))
        .and_then(toml::Value::as_str)
        .and_then(|device| device.rsplit(':').next())
        .and_then(|ordinal| ordinal.trim().parse::<i32>().ok())
        .unwrap_or(0)
}

/// How much of a fire `[engine] graphs` asks to record.
///
/// `PIE_CUDA_GRAPHS` still overrides it inside the shell, at load, which is
/// where every other environment read on that plane happens.
fn graphs(doc: &toml::Table) -> Graphs {
    match doc
        .get("engine")
        .and_then(toml::Value::as_table)
        .and_then(|engine| engine.get("graphs"))
        .and_then(toml::Value::as_str)
    {
        Some("off" | "eager") => Graphs::Off,
        Some("shaped") => Graphs::Shaped,
        Some("on" | "graph") => Graphs::On,
        _ => Graphs::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_device_key_is_read_in_every_spelling_a_boot_config_uses() {
        let read = |text: &str| ordinal(&text.parse::<toml::Table>().expect("valid TOML"));
        assert_eq!(read(r#"[model]
device = "cuda:3""#), 3);
        assert_eq!(read(r#"[model]
device = "2""#), 2);
        assert_eq!(read("[model]"), 0, "a config that says nothing means device 0");
        assert_eq!(read(""), 0);
    }

    #[test]
    fn the_graph_mode_defaults_to_the_shells_own() {
        let read = |text: &str| graphs(&text.parse::<toml::Table>().expect("valid TOML"));
        assert_eq!(read(r#"[engine]
graphs = "on""#), Graphs::On);
        assert_eq!(read(r#"[engine]
graphs = "shaped""#), Graphs::Shaped);
        assert_eq!(read(""), Graphs::default());
    }
}
