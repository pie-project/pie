//! The pipeline cache: `(file, entrypoint, stamp)` in, a compiled compute
//! pipeline out, compiled once per load.
//!
//! **WHY SOURCE AND NOT A `.metallib`.** The obvious build-time answer is
//! `xcrun metal` into an offline library the rlib carries. It is not
//! available: a Mac with the Command Line Tools and no full Xcode has no
//! `metal` in `xcrun`'s tool list (measured on the M1 Max this shell is
//! gated on), so an offline build would make the crate un-buildable on the
//! ordinary developer machine. `newLibraryWithSource:` needs neither — the
//! compiler behind it ships with the Metal framework — so the sources travel
//! in the rlib (`kernels_metal::SOURCES`) and the engine compiles them on
//! the device it is about to fire on. That is also the honest ordering: a
//! pipeline is device-specific, and an offline library would have to be
//! re-specialized anyway.
//!
//! **One library per FILE, one pipeline per ENTRYPOINT.** A `.metal` file
//! holds several entrypoints — `elemwise/norm_layer_scalar.metal` holds two,
//! `attn/sdpa_paged.metal` holds ten — and compiling the file once for all
//! of them is the difference between one compile and ten. Includes are
//! flattened by `kernels_metal::resolve` before the source is handed over,
//! because `newLibraryWithSource:` has no header search path.
//!
//! **The `stamp` is a LINE OF SOURCE, and that is the whole specialization
//! path.** `Fire::stamp` is the jit instantiation point a specialized entry
//! names — `PIE_STAMP_qmm_t("affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32",
//! 64, 4, 32, 32, 32)`, composed by `kernels_metal::linear::quant`. The
//! shader declares the macro and instantiates NOTHING with it; the driver
//! appends the one invocation it selected to the flattened source and
//! compiles that. The reference driver spelled all 216 affine qmm points out
//! in the file instead, and paid the Metal compiler for every one of them on
//! a load that fires two.
//!
//! So the stamp is part of the source, and the cache keys say so: a library
//! is `(file, stamp)` and a pipeline is `(file, entrypoint, stamp)`. Two
//! stamps of one file are two libraries, which is the honest cost — a plan
//! reaches a handful of tile shapes, and a pipeline already compiled stays
//! valid when the next stamp arrives. A stamp that names no macro, or a
//! macro that mints no such entrypoint, is a compile refusal carrying the
//! Metal compiler's own paragraph, which is what the field being carried
//! rather than ignored was always for.

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

/// Compiled shader state for one load.
///
/// Interior-mutable because the encode sink holds it behind `&` — a
/// `kernels_metal::Encode::fire` takes `&self`, and a pipeline that has to
/// be compiled on first sight is a write on that path. The alternative,
/// compiling every point at load, is a real option and a worse default: the
/// catalog's 38 live points cost a second of compile each on a cold system
/// cache and a plan uses a fraction of them.
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
    ///
    /// The observable behind the warm-cache gate: a steady stream of fires
    /// over one composition compiles NOTHING, and an absence has no output
    /// unless something counts it.
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
        // **EVERY PIPELINE IS BUILT FOR AN INDIRECT COMMAND BUFFER, AND ONE
        // PATH SERVES BOTH CONSUMERS.** `supportIndirectCommandBuffers` is
        // false by default and cannot be turned on afterwards, and a pipeline
        // without it cannot be set into an `MTLIndirectComputeCommand` — so a
        // cache that built the plain form would have to build every point
        // twice the day `crate::icb` wants one. The flag costs a compute pass
        // nothing measurable here (`serve_smoke`'s ms/fire is unmoved), and
        // asking for it needs the DESCRIPTOR form of the constructor, which
        // is the only reason this is not one line.
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
    /// The warm-up door, and the census one: a caller that wants every point
    /// a plan can reach paid for before the first fire calls this over them,
    /// and a test that wants to know the shipped sources still compile calls
    /// it over [`entrypoints`](Pipelines::entrypoints).
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

    /// Every entrypoint one shipped source file publishes.
    ///
    /// What the library itself says, not what this crate remembers: the
    /// `instantiate_*` macros in the sources mint host names the Rust side
    /// spells by hand, and asking the compiled library is the only reading
    /// that cannot drift from them.
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
        // Appended and not substituted: the stamp is a macro invocation the
        // file itself declares, so it belongs after every declaration and
        // template it names.
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
