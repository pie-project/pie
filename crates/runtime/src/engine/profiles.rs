//! The device-profile registries this build carries, one per backend it
//! hosts: what `pie profile` and doctor read.

use std::path::Path;

use profile::{Facts, Profile, Registry, Report, Shelf};

/// One backend's profiles: its registry, its checked-in files, and how to
/// read the device it would serve.
pub struct Backend {
    pub name: &'static str,
    pub registry: fn() -> &'static Registry,
    pub shelf: &'static Shelf,
    /// The device this machine would serve, by name, with its facts.
    pub probe: fn() -> Option<(String, Facts)>,
    /// Resolve the profile a load on that device runs under, as the engine
    /// does: with the user's profiles directory.
    pub resolve: fn(&str, &Facts, Option<&Path>) -> (Profile, Report),
}

/// Every backend this build hosts.
#[must_use]
pub fn backends() -> Vec<Backend> {
    #[allow(unused_mut)]
    let mut out = Vec::new();
    #[cfg(feature = "cuda")]
    out.push(Backend {
        name: "cuda",
        registry: engine_cuda::chip::registry,
        shelf: &engine_cuda::chip::SHELF,
        probe: || engine_cuda::chip::probe(0),
        resolve: engine_cuda::chip::resolve,
    });
    #[cfg(feature = "vulkan")]
    out.push(Backend {
        name: "vulkan",
        registry: engine_vulkan::chip::registry,
        shelf: &engine_vulkan::chip::SHELF,
        probe: engine_vulkan::chip::probe,
        resolve: engine_vulkan::chip::resolve_facts,
    });
    #[cfg(feature = "metal")]
    out.push(Backend {
        name: "metal",
        registry: engine_metal::chip::registry,
        shelf: &engine_metal::chip::SHELF,
        probe: engine_metal::chip::probe,
        resolve: engine_metal::chip::resolve,
    });
    out
}
