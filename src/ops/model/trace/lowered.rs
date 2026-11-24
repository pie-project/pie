//! `lowered.json`: the trace as the compiler lowers it for each named
//! platform (pack, tile, shard, expert), before an engine's own kernel
//! fusions. Each entry says how many banks come through under the same
//! name, shape and dtype and how many products the lowered trace runs, or
//! why the compiler refused the platform.

use anyhow::{Result, anyhow};
use poem::{Linear, Operation, Param, ParamSource, Platform};
use runtime::catalog::Deployment;
use serde_json::{Value, json};

/// The entry of `deployment` lowered for `platform`, which should keep
/// `banks`; and the first bank it lost, if any.
pub fn entry(
    deployment: &Deployment,
    platform: Platform,
    banks: &[&Param],
) -> (Value, Option<String>) {
    let lowered = match deployment.try_trace(platform) {
        Ok(trace) => trace,
        Err(why) => return (json!({"refused": why.to_string()}), None),
    };
    let kept = |bank: &Param| {
        lowered.params.iter().any(|p| {
            p.source == ParamSource::Registered
                && p.name == bank.name
                && p.shape == bank.shape
                && p.dtype == bank.dtype
        })
    };
    let lost = banks
        .iter()
        .find(|bank| !kept(bank))
        .map(|bank| bank.name.clone());
    let matmuls = lowered
        .nodes
        .iter()
        .filter(|node| matches!(node.op, Operation::Linear(Linear::Matmul { .. })))
        .count();
    let entry = json!({
        "banks_kept": banks.iter().filter(|bank| kept(bank)).count(),
        "banks_total": banks.len(),
        "matmuls": matmuls,
    });
    (entry, lost)
}

/// The platform `name` spells: a backend's own name, or `gpu` for wgpu,
/// the platform engine-gpu serves.
pub fn platform(name: &str) -> Result<Platform> {
    if name == "gpu" {
        return Ok(Platform::Wgpu);
    }
    let all = [
        Platform::Cuda,
        Platform::Metal,
        Platform::Wgpu,
        Platform::Vulkan,
        Platform::Xla,
        Platform::Cerebras,
    ];
    all.into_iter()
        .find(|p| p.backend() == name)
        .ok_or_else(|| {
            anyhow!(
                "no platform `{name}`: pie lowers for cuda, metal, wgpu (gpu), vulkan, xla and \
                 cerebras"
            )
        })
}
