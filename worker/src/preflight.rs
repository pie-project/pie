//! Per-model translation helpers used by [`super::start_engine`]:
//! topology calculation, build-feature/driver-kind validation, and
//! the [`config::ModelConfig`] → [`DriverOptions`] projection.

use anyhow::{Result, anyhow};

#[cfg(feature = "driver-cuda")]
use crate::config::CudaNativeDriverOptions;
#[cfg(feature = "driver-metal")]
use crate::config::MetalDriverOptions;
use crate::config::{self, DriverKind, DummyDriverOptions};
use crate::driver_ffi::Flavor;
use crate::embedded_driver::DriverOptions;

/// Top-level dispatcher: the inbound `[model]` either binds to an
/// in-process [`Flavor`] (C++/Rust static lib).
///
/// Returned by [`resolve_flavor`]; consumed by [`super::start_engine`]
/// to pick which supervisor to spawn per group.
#[derive(Copy, Clone, Debug)]
pub enum ResolvedFlavor {
    Embedded(Flavor),
}

/// Partition `world_size` ranks into TP groups of size `tp_degree`.
///
/// Example: `world_size=4, tp_degree=2 → [[0,1], [2,3]]` — two DP
/// replicas, each with two TP-sharded ranks.
pub fn calculate_topology(world_size: usize, tp_degree: usize) -> Result<Vec<Vec<usize>>> {
    if tp_degree == 0 {
        anyhow::bail!("tensor_parallel_size must be > 0");
    }
    if world_size % tp_degree != 0 {
        anyhow::bail!(
            "world_size ({world_size}) must be divisible by \
             tensor_parallel_size ({tp_degree})"
        );
    }
    let num_groups = world_size / tp_degree;
    Ok((0..num_groups)
        .map(|g| (g * tp_degree..(g + 1) * tp_degree).collect())
        .collect())
}

/// Resolve the `[model].driver.type` to a [`ResolvedFlavor`]. Errors
/// out with a clear, model-named message when the requested embedded
/// flavor was not compiled into this binary.
pub fn resolve_flavor(kind: DriverKind, model_name: &str) -> Result<ResolvedFlavor> {
    match kind {
        DriverKind::CudaNative | DriverKind::Metal | DriverKind::Dummy => Flavor::from_kind(kind)
            .map(ResolvedFlavor::Embedded)
            .map_err(|msg| anyhow!("model {model_name:?}: {msg}")),
    }
}

/// Project a [`config::ModelConfig`] into the typed [`DriverOptions`]
/// the embedded driver expects. Caller has already discriminated to an
/// embedded [`Flavor`].
///
/// The cuda variant's `device` is filled from the first device in the
/// model's list as a placeholder — the per-group spawn loop overwrites
/// it with the right device for each DP replica.
pub fn build_embedded_options(m: &config::ModelConfig, flavor: Flavor) -> Result<DriverOptions> {
    match flavor {
        #[cfg(feature = "driver-cuda")]
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
        #[cfg(feature = "driver-metal")]
        Flavor::Metal => {
            let p: MetalDriverOptions = m
                .driver
                .options
                .clone()
                .try_into()
                .map_err(|e| anyhow!("[model.driver.options] for {:?}: {e}", m.name))?;
            Ok(DriverOptions::Metal(p))
        }
        Flavor::Dummy => {
            let d: DummyDriverOptions = m
                .driver
                .options
                .clone()
                .try_into()
                .map_err(|e| anyhow!("[model.driver.options] for {:?}: {e}", m.name))?;
            Ok(DriverOptions::Dummy {
                opts: d,
                random_seed: m.driver.random_seed,
                activation_dtype: m.driver.activation_dtype.clone(),
            })
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
    fn topology_dp_two() {
        assert_eq!(
            calculate_topology(2, 1).unwrap(),
            vec![vec![0], vec![1]],
            "DP=2, TP=1 → two single-rank groups"
        );
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
    fn topology_dp2_tp2() {
        assert_eq!(
            calculate_topology(4, 2).unwrap(),
            vec![vec![0, 1], vec![2, 3]],
            "DP=2, TP=2 → two two-rank groups"
        );
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
