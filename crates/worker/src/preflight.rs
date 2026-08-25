//! Per-model translation helpers used by [`super::start_engine`]:
//! topology calculation, build-feature/driver-kind validation, and
//! the [`config::ModelConfig`] → [`DriverOptions`] projection.

use anyhow::{Result, anyhow};

#[cfg(feature = "_driver-cuda")]
use crate::config::CudaNativeDriverOptions;
#[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
use crate::config::MetalDriverOptions;
use crate::config::{self, DriverKind};
use crate::driver_ffi::Flavor;
use crate::embedded_driver::DriverOptions;

/// Partition `world_size` ranks into ONE tensor-parallel group.
///
/// Example: `world_size=2, tp_degree=2 → [[0, 1]]`.
///
/// A worker serves exactly one replica of its model. Data parallelism is a
/// CLUSTER shape, not an engine one: N replicas are N workers, and the
/// gateway already spreads requests over them by the coarse load each
/// reports (`gateway::admission` filters on `kv_pressure_bucket` and
/// `inflight`), which is both load-aware and the path that already exists.
///
/// Putting several replicas in one engine instead asks the CUDA driver to
/// serve two devices from one process, and almost nothing in it is written
/// that way: the driver holds ~96 process-global statics, and every one
/// that owns device memory or a stream is a fault waiting for the second
/// device. Three have already had to be made per-device — the cuBLASLt
/// context and plan cache, the frame carrier's copy stream, and the baked
/// buffer pool — each found only after it crashed. There is no way to know
/// the rest have been found. (Tensor parallelism does share a process
/// across devices, which is exactly why those three were bugs there too.)
pub fn calculate_topology(world_size: usize, tp_degree: usize) -> Result<Vec<Vec<usize>>> {
    if tp_degree == 0 {
        anyhow::bail!("tensor_parallel_size must be > 0");
    }
    if !world_size.is_multiple_of(tp_degree) {
        anyhow::bail!(
            "world_size ({world_size}) must be divisible by \
             tensor_parallel_size ({tp_degree})"
        );
    }
    let num_groups = world_size / tp_degree;
    if num_groups > 1 {
        anyhow::bail!(
            "model.driver.device lists {world_size} devices with \
             tensor_parallel_size = {tp_degree}, which asks for \
             {num_groups} data-parallel replicas in one engine. A worker \
             serves one replica: run {num_groups} workers, each with \
             {tp_degree} device(s), and let the gateway spread requests \
             over them."
        );
    }
    Ok((0..num_groups)
        .map(|g| (g * tp_degree..(g + 1) * tp_degree).collect())
        .collect())
}

/// Resolve the `[model].driver.type` to the [`Flavor`] that hosts it, naming
/// the model in the refusal when this binary hosts none.
///
/// A `ResolvedFlavor` enum STOOD HERE, wrapping the flavor in an `Embedded`
/// variant, and a four-arm match fed it the same expression from every arm.
/// It dated from a runtime with out-of-process drivers to dispatch to as
/// well; every driver is a static lib now, so "which of the ways of hosting
/// one" has one answer and does not need to be asked.
pub fn resolve_flavor(kind: DriverKind, model_name: &str) -> Result<Flavor> {
    Flavor::from_kind(kind).map_err(|msg| anyhow!("model {model_name:?}: {msg}"))
}

/// Project a [`config::ModelConfig`] into the typed [`DriverOptions`]
/// the embedded driver expects. Caller has already discriminated to an
/// embedded [`Flavor`].
///
/// The cuda variant's `device` is filled from the first device in the
/// model's list as a placeholder — the per-group spawn loop overwrites
/// it with the right device for each DP replica.
///
/// One arm, because one flavor: the `Metal`, `Vulkan` and `Wgpu` arms went
/// with their `Flavor` variants, and [`resolve_flavor`] refuses those kinds
/// before a `DriverOptions` is ever asked for.
#[cfg_attr(
    not(feature = "_driver-cuda"),
    allow(
        unused_variables,
        unreachable_code,
        reason = "with no `driver-*` feature `DriverOptions` is uninhabited, so \
                  every path that produces one diverges"
    )
)]
pub fn build_embedded_options(m: &config::ModelConfig, flavor: Flavor) -> Result<DriverOptions> {
    match flavor {
        #[cfg(feature = "_driver-cuda")]
        Flavor::Cuda => {
            let mut c: CudaNativeDriverOptions = m
                .driver
                .options
                .clone()
                .try_into()
                .map_err(|e| anyhow!("[model.driver.options] for {:?}: {e}", m.name))?;
            let device = m.driver.device.first().ok_or_else(|| {
                anyhow!(
                    "model {:?}: cuda_native requires at least one device",
                    m.name
                )
            })?;
            c.device = device.clone();
            Ok(DriverOptions::CudaNative(c))
        }
        // NO DEVICE SELECTOR, unlike the arm above. `Shell::open` takes the
        // DEFAULT Metal 4 device and offers no way to name another, so filling
        // one in here would be a setting nothing acts on —
        // `MetalDriverOptions::device` exists for the startup TOML an operator
        // reads, and is `#[serde(skip)]` for the same reason.
        #[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
        Flavor::Metal => {
            let p: MetalDriverOptions = m
                .driver
                .options
                .clone()
                .try_into()
                .map_err(|e| anyhow!("[model.driver.options] for {:?}: {e}", m.name))?;
            Ok(DriverOptions::Metal(p))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_single_rank() {
        assert_eq!(calculate_topology(1, 1).unwrap(), vec![vec![0]]);
    }

    #[test]
    fn topology_rejects_dp_two() {
        let err = calculate_topology(2, 1).unwrap_err().to_string();
        assert!(err.contains("run 2 workers"), "got: {err}");
    }

    #[test]
    fn topology_tp_two() {
        assert_eq!(
            calculate_topology(2, 2).unwrap(),
            vec![vec![0, 1]],
            "DP=1, TP=2 → one two-rank group"
        );
    }

    #[test]
    fn topology_rejects_dp2_tp2() {
        let err = calculate_topology(4, 2).unwrap_err().to_string();
        assert!(err.contains("run 2 workers"), "got: {err}");
    }

    #[test]
    fn topology_rejects_indivisible() {
        let err = calculate_topology(3, 2).unwrap_err().to_string();
        assert!(err.contains("must be divisible"), "got: {err}");
    }

    #[test]
    fn topology_rejects_zero_tp() {
        let err = calculate_topology(4, 0).unwrap_err().to_string();
        assert!(err.contains("must be > 0"), "got: {err}");
    }
}
