//! What a measurement a load stores in the user's profile is keyed by on
//! these backends ([`profile::fire::Now`]): the device (its vendor, device
//! id and driver, as the backend states them), the version of the method
//! that measured it (a constant beside the measuring code, bumped only when
//! the method changes) and the code of each kernel it fired (its SPIR-V or
//! WGSL module's fingerprint). Nothing else of the build keys it: a new pie
//! version, commit or day reuses every stored measurement whose key still
//! matches, and a changed kernel has its own measurements taken again,
//! alone.

use std::collections::BTreeSet;

use kernels_slang::Target;
use profile::fire::{MeasuredBy, Now};

use crate::error::{Fault, Result};
use crate::naming::Kernel;

/// This build on one device, as its stored measurements are checked
/// against.
#[derive(Clone, Debug)]
pub struct Keys {
    on: String,
    target: Target,
}

impl Keys {
    /// `on`, the device as the backend states it (vendor, device id,
    /// driver), its kernels built for `target`.
    #[must_use]
    pub fn new(on: String, target: Target) -> Keys {
        Keys { on, target }
    }

    /// The device as the stored keys name it.
    #[must_use]
    pub fn device(&self) -> &str {
        &self.on
    }
}

impl Now for Keys {
    fn on(&self) -> String {
        self.on.clone()
    }

    fn method(&self, what: &str) -> Option<u32> {
        method(what)
    }

    fn code(&self, kernel: &str) -> Option<u32> {
        code(self.target, kernel)
    }
}

/// The key of a measurement `what` (a `[rates]` key) that fired `kernels`
/// built for `target`: its method's version and each kernel's code. A
/// kernel this build does not carry, or a `what` no method measures, is
/// refused by name.
pub fn by<'k>(
    target: Target,
    what: &str,
    kernels: impl IntoIterator<Item = &'k Kernel>,
) -> Result<MeasuredBy> {
    let method = method(what).ok_or_else(|| Fault::Device {
        call: "stored key",
        why: format!("`{what}` is measured by no method of this build"),
    })?;
    let kernels = kernels
        .into_iter()
        .map(|kernel| {
            code(target, kernel.key)
                .map(|code| (kernel.key.to_string(), code))
                .ok_or_else(|| Fault::Device {
                    call: "stored key",
                    why: format!(
                        "`{what}` fired kernel `{}`, which this build does not carry",
                        kernel.key
                    ),
                })
        })
        .collect::<Result<Vec<_>>>()?;
    // Sorted by name, once each: the key is its content whatever order the
    // measurement fired its kernels in.
    let mut kernels = kernels;
    kernels.sort();
    kernels.dedup();
    Ok(MeasuredBy { method, kernels })
}

/// The fingerprint of `kernel`'s compiled module for `target` in this
/// build (`kernels_slang::key`'s key), `None` where it carries none.
#[must_use]
pub fn code(target: Target, kernel: &str) -> Option<u32> {
    let bytes = match target {
        Target::Spirv => kernels_slang::spirv(kernel)?,
        Target::Wgsl => kernels_slang::wgsl(kernel)?.as_bytes(),
    };
    Some(profile::fire::fingerprint(bytes))
}

/// The fingerprint of every kernel of `kernels` for `target`, each by its
/// key and code, in order: what a measurement of a whole served step (the
/// settle) is keyed by. A kernel the build does not carry is refused by
/// name.
pub fn code_of_all(target: Target, kernels: &BTreeSet<Kernel>) -> Result<u32> {
    // By name, once each: the key is the kernels' content, whatever order
    // their set is kept in.
    let mut keys: Vec<&str> = kernels.iter().map(|kernel| kernel.key).collect();
    keys.sort_unstable();
    keys.dedup();
    let mut bytes = Vec::new();
    for key in keys {
        let code = code(target, key).ok_or_else(|| Fault::Device {
            call: "stored key",
            why: format!("kernel `{key}` is not in this build"),
        })?;
        bytes.extend_from_slice(key.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(&code.to_le_bytes());
    }
    Ok(profile::fire::fingerprint(&bytes))
}

/// The version of the method that measures `what` (a `[rates]` key), as
/// the measuring code states it beside itself; `None` where no method of
/// this build measures it.
#[must_use]
pub fn method(what: &str) -> Option<u32> {
    use crate::rates;
    Some(match what {
        "launch_ns" | "hbm_gbps" => rates::LAUNCH_METHOD,
        "gemm_tflops_by_rows" => rates::GEMM_METHOD,
        "attn_tflops_by_keys" | profile::fire::ATTN_BY_ROWS_KEYS => rates::ATTN_METHOD,
        "elementwise_gbps" => rates::ELEMENTWISE_METHOD,
        "scan_gelems" => rates::SCAN_METHOD,
        profile::fire::RESIDENT_WORKGROUPS => rates::RESIDENT_METHOD,
        profile::fire::SUBMIT_WORST_RATIO => rates::SUBMIT_METHOD,
        profile::fire::KERNEL_CURVES => rates::CURVE_METHOD,
        profile::fire::DENSE_SETTLED => crate::serve::SETTLE_METHOD,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every key a load measures, its curves and its settle are measured by
    /// a method of this build; nothing else is.
    #[test]
    fn every_measured_key_has_its_method() {
        for key in crate::rates::keys_at(Some(std::time::Duration::from_millis(1))) {
            assert!(method(key).is_some(), "{key}");
        }
        assert!(method(profile::fire::KERNEL_CURVES).is_some());
        assert!(method(profile::fire::DENSE_SETTLED).is_some());
        assert_eq!(method("product_ns"), None);
    }

    /// A kernel's code is its module's fingerprint in this build; one the
    /// build does not carry has none, and a key naming it is refused.
    #[test]
    fn a_kernels_code_is_its_modules_fingerprint() {
        for (target, key) in kernels_slang::SPIRV
            .iter()
            .map(|(key, ..)| (Target::Spirv, *key))
            .chain(
                kernels_slang::WGSL
                    .iter()
                    .map(|(key, ..)| (Target::Wgsl, *key)),
            )
            .take(4)
        {
            let bytes = match target {
                Target::Spirv => kernels_slang::spirv(key).unwrap(),
                Target::Wgsl => kernels_slang::wgsl(key).unwrap().as_bytes(),
            };
            assert_eq!(code(target, key), Some(profile::fire::fingerprint(bytes)));
        }
        assert_eq!(code(Target::Spirv, "no_such_kernel"), None);
        let absent = Kernel {
            unit: "linear/gemm.slang",
            key: "no_such_kernel",
        };
        assert!(by(Target::Spirv, "launch_ns", [&absent]).is_err());
        let all: BTreeSet<Kernel> = [absent].into_iter().collect();
        assert!(code_of_all(Target::Spirv, &all).is_err());
    }
}
