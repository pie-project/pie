//! Opening a CUDA device from a [`DeviceBoot`] — the shell's boot, as the
//! shell's own typed struct.
//!
//! # The boot is a struct now, and the history is why
//!
//! This file used to read a boot TOML into [`DeviceBoot`]: the worker wrote
//! the document, this file parsed it back, and between the two sat a wire
//! format on which a key could quietly fail to arrive. Every key had that
//! failure at least once — `[model] weight_cache_dir` was written by the
//! worker and parsed by nobody for a whole rewrite, and
//! `[engine] gpu_mem_utilization` for four waves after that — and a key
//! nobody can read back is a key whose arrival nothing tests.
//!
//! So the seam inverted again. The caller constructs [`DeviceBoot`] and hands
//! it over; a field of a struct cannot fail to arrive, and the compiler is
//! the reader that never goes missing. What survives of the reader is the
//! one check that was never about spelling ([`open`]'s range refusal on the
//! memory fraction) and the one naming fact that is CUDA's rather than any
//! config format's ([`ordinal_of`]).

use crate::api::{ContractFor, Cuda, DeviceBoot};

/// Open one device from a typed boot.
///
/// The contract lookup is a PARAMETER and not something this crate could
/// find: how a checkpoint's tensors become a plan's params is the model's
/// declaration, resolved by the party that links the catalog. See
/// [`ContractFor`], and [`crate::api`]'s header for the diagram. It is also
/// why this function can live here at all — it is the one ingredient of an
/// open that points the wrong way up the dependency graph, and taking it as
/// an argument is what keeps `engine-cuda → runtime` from existing.
///
/// # Errors
///
/// A `gpu_mem_utilization` outside `(0.0, 1.0]`, as a sentence — the one
/// semantic check the boot needs, kept here because this shell is the party
/// that turns the fraction into bytes, and a fraction of `0` or `1.7` is a
/// deployment nobody meant. Clamping it would open a pool the operator did
/// not ask for and say nothing. `String` rather than
/// [`Fault`](crate::Fault): this is a seam between a crate whose errors are
/// `anyhow` and one whose errors are `Fault`, and neither should have to
/// name the other's error crate to open a device. Nothing here touches a
/// device, so no variant of `Fault` describes what can go wrong anyway.
///
/// Binding the device itself happens at
/// [`Engine::load`](engine::Engine::load), not here: `Shell::load` is one
/// call that binds, bakes and lands, and there is nothing to bind before a
/// plan says what to bake.
pub fn open(boot: DeviceBoot, contract_for: ContractFor) -> Result<Cuda, String> {
    let fraction = boot.knobs.gpu_mem_utilization;
    if !fraction.is_finite() || fraction <= 0.0 || fraction > 1.0 {
        return Err(format!(
            "`gpu_mem_utilization` must be finite and in (0.0, 1.0]; this boot \
             says {fraction}. It is the fraction of the whole card this \
             deployment lets pie hold, weights included — 1.0 is the whole \
             card, which is what the elastic pool took before the knob \
             reached a shell at all."
        ));
    }
    Ok(Cuda::new(boot, contract_for))
}

/// Which device ordinal a spelled device names.
///
/// `"cuda:1"` and `"1"` both mean device one, and the empty string means
/// zero — a single-GPU box is the deployment that writes the least. A CUDA
/// naming fact, which is why it lives in this crate rather than with
/// whoever assembles a [`DeviceBoot`].
#[must_use]
pub fn ordinal_of(device: &str) -> i32 {
    device
        .rsplit(':')
        .next()
        .and_then(|ordinal| ordinal.trim().parse::<i32>().ok())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::ordinal_of;

    #[test]
    fn the_device_spelling_is_read_in_every_form_a_deployment_writes() {
        assert_eq!(ordinal_of("cuda:3"), 3);
        assert_eq!(ordinal_of("2"), 2);
        assert_eq!(ordinal_of(""), 0, "saying nothing means device 0");
    }
}
