//! The pipeline cache: `(file, entrypoint, stamp)` in, a compiled compute
//! pipeline out, compiled once per load. Source (not `.metallib`) because
//! `xcrun metal` isn't available without full Xcode; `newLibraryWithSource:`
//! needs only the Metal framework. One library per file (may hold several
//! entrypoints), one pipeline per entrypoint. `stamp` is a jit specialization:
//! a macro invocation appended to the source, so a library is keyed
//! `(file, stamp)` and a pipeline `(file, entrypoint, stamp)`.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::error::{Fault, Result};

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2::runtime::ProtocolObject;
#[cfg(target_vendor = "apple")]
use objc2_metal::{MTLComputePipelineDescriptor, MTLComputePipelineState, MTLDevice, MTLLibrary};

#[cfg(target_vendor = "apple")]
type Library = Retained<ProtocolObject<dyn MTLLibrary>>;
#[cfg(not(target_vendor = "apple"))]
type Library = ();

#[cfg(target_vendor = "apple")]
pub(crate) type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
#[cfg(not(target_vendor = "apple"))]
pub(crate) type Pipeline = ();

/// Compiled shader state for one load. Interior-mutable because
/// `Encode::fire` takes `&self`, and a first-sighting pipeline is compiled
/// lazily on that path rather than all up front at load.
#[derive(Default)]
pub struct Pipelines {
    libraries: RefCell<HashMap<(&'static str, &'static str), Library>>,
    pipelines: RefCell<HashMap<(&'static str, &'static str, &'static str), Pipeline>>,
    /// Every compile this load performed, for the warm/cold gate.
    compiles: std::cell::Cell<u64>,
}

// SAFETY: `MTLLibrary` and `MTLComputePipelineState` are documented
// thread-safe; the `RefCell`s are only ever touched from the lane thread.
unsafe impl Send for Pipelines {}

impl std::fmt::Debug for Pipelines {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipelines")
            .field("libraries", &self.libraries.borrow().len())
            .field("pipelines", &self.pipelines.borrow().len())
            .field("compiles", &self.compiles.get())
            .finish()
    }
}

impl Pipelines {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Pipelines {
        Pipelines::default()
    }

    /// How many entrypoints this load has compiled.
    #[must_use]
    pub fn compiled(&self) -> u64 {
        self.compiles.get()
    }

    /// The pipeline for one `Fire`'s point, compiling it if this is its
    /// first sighting.
    ///
    /// # Errors
    ///
    /// [`Fault::Shader`] for a source this crate does not ship, an
    /// entrypoint the library does not hold — including one a jit stamp
    /// promised and its macro did not mint — or a source the Metal compiler
    /// refused.
    #[cfg(target_vendor = "apple")]
    pub(crate) fn at(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        fire: kernels_metal::Fire,
    ) -> Result<Pipeline> {
        let key = (fire.file, fire.entrypoint, fire.stamp);
        if let Some(pipeline) = self.pipelines.borrow().get(&key) {
            return Ok(pipeline.clone());
        }
        let library = self.library(device, fire)?;
        let name = super::ctx::nsstring(fire.entrypoint);
        let function = library
            .newFunctionWithName(&name)
            .ok_or_else(|| Fault::Shader {
                file: fire.file,
                entrypoint: fire.entrypoint,
                why: if fire.stamp.is_empty() {
                    "the compiled library holds no such entrypoint".to_string()
                } else {
                    format!(
                        "the stamp `{}` compiled and minted no entrypoint by that name — \
                         the macro and the symbol the driver composed do not agree",
                        fire.stamp
                    )
                },
            })?;
        // Every pipeline is built for an indirect command buffer:
        // supportIndirectCommandBuffers is false by default and can't be
        // turned on after the fact, and the flag costs nothing measurable.
        let descriptor = MTLComputePipelineDescriptor::new();
        descriptor.setComputeFunction(Some(&function));
        descriptor.setSupportIndirectCommandBuffers(true);
        descriptor.setLabel(Some(&super::ctx::nsstring(fire.entrypoint)));
        let pipeline = device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &descriptor,
                objc2_metal::MTLPipelineOption::None,
                None,
            )
            .map_err(|error| Fault::Shader {
                file: fire.file,
                entrypoint: fire.entrypoint,
                why: error.localizedDescription().to_string(),
            })?;
        self.compiles.set(self.compiles.get() + 1);
        self.pipelines.borrow_mut().insert(key, pipeline.clone());
        Ok(pipeline)
    }

    /// Compile one point and hold it, without firing it.
    ///
    /// # Errors
    ///
    /// As [`Pipelines::at`].
    pub fn warm(&self, device: &super::Context, fire: kernels_metal::Fire) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            self.at(device.device(), fire).map(|_| ())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (device, fire);
            Err(Fault::Deviceless)
        }
    }

    /// Every entrypoint one shipped source file publishes, as the compiled
    /// library itself says (not what this crate remembers), since
    /// `instantiate_*` macros mint host names by hand that can drift.
    ///
    /// # Errors
    ///
    /// [`Fault::Shader`] when the source does not ship or does not compile.
    pub fn entrypoints(&self, device: &super::Context, file: &'static str) -> Result<Vec<String>> {
        #[cfg(target_vendor = "apple")]
        {
            let library = self.library(device.device(), kernels_metal::Fire::at(file, ""))?;
            Ok(library
                .functionNames()
                .iter()
                .map(|name| name.to_string())
                .collect())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (device, file);
            Err(Fault::Deviceless)
        }
    }

    /// The compiled library for one source file.
    #[cfg(target_vendor = "apple")]
    fn library(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        fire: kernels_metal::Fire,
    ) -> Result<Library> {
        let key = (fire.file, fire.stamp);
        if let Some(library) = self.libraries.borrow().get(&key) {
            return Ok(library.clone());
        }
        let mut flat = kernels_metal::resolve(fire.file).map_err(|missing| Fault::Shader {
            file: fire.file,
            entrypoint: fire.entrypoint,
            why: format!("includes `{missing}`, which this crate does not ship"),
        })?;
        // Appended, not substituted: the stamp invokes a macro the file
        // itself declares, so it must follow every declaration it names.
        if !fire.stamp.is_empty() {
            flat.push('\n');
            flat.push_str(fire.stamp);
            flat.push('\n');
        }
        let source = super::ctx::nsstring(&flat);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|error| Fault::Shader {
                file: fire.file,
                entrypoint: fire.entrypoint,
                why: error.localizedDescription().to_string(),
            })?;
        self.libraries.borrow_mut().insert(key, library.clone());
        Ok(library)
    }
}
