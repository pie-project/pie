use std::ffi::{CStr, CString, c_char};

use cudarc::driver::sys as dr;
use kernels::KernelSig;

use crate::runtime::{Args, Error, Launch, Stream, Ungeometric};

/// A loaded image, and every entry point the rows it was built from name.
pub struct KernelModule {
    /// The unit this image came from — [`crate::unit::Unit::name`].
    unit: &'static str,
    module: dr::CUmodule,
    /// Parallel to the row list the module was loaded against — one unit's
    entries: Vec<(&'static str, dr::CUfunction)>,
}

// SAFETY: `CUmodule` and `CUfunction` are context-scoped handles and a
unsafe impl Send for KernelModule {}
// SAFETY: as above -- every method here reads an immutable handle.
unsafe impl Sync for KernelModule {}

impl KernelModule {
    /// Load `image` and resolve every row, looking each one up by the
    pub fn load_mangled(
        unit: &'static str,
        image: &[u8],
        table: &[&'static KernelSig],
        lowered: &[(&'static str, String)],
    ) -> Result<Self, Error> {
        let module = Self::load_image(unit, image)?;
        let mut entries = Vec::with_capacity(table.len());
        for sig in table {
            let resolved = match lowered.iter().find(|(s, _)| *s == sig.symbol) {
                Some((_, mangled)) => Self::entry_by_name(unit, module, sig.symbol, mangled),
                None => Err(Error::Missing { unit, symbol: sig.symbol }),
            };
            match resolved {
                Ok(function) => entries.push((sig.symbol, function)),
                Err(why) => {
                    // SAFETY: `module` loaded, nothing else holds it, and no
                    unsafe { dr::cuModuleUnload(module) };
                    return Err(why);
                }
            }
        }
        Ok(Self { unit, module, entries })
    }

    /// `cuModuleLoadData`, with the empty image refused rather than handed
    fn load_image(unit: &'static str, image: &[u8]) -> Result<dr::CUmodule, Error> {
        if image.is_empty() {
            return Err(Error::Compile {
                unit,
                why: "the compile produced an empty image, so there is nothing to load".into(),
            });
        }
        let mut module: dr::CUmodule = std::ptr::null_mut();
        // SAFETY: `image` is a live byte image and `module` a live
        let code = unsafe { dr::cuModuleLoadData(&raw mut module, image.as_ptr().cast()) };
        if code == dr::CUresult::CUDA_SUCCESS {
            Ok(module)
        } else {
            Err(refused("cuModuleLoadData", code))
        }
    }

    /// One row's entry point, by the mangled name the compile gave it.
    fn entry_by_name(
        unit: &'static str,
        module: dr::CUmodule,
        symbol: &'static str,
        mangled: &str,
    ) -> Result<dr::CUfunction, Error> {
        let Ok(c_name) = CString::new(mangled) else {
            return Err(Error::Compile {
                unit,
                why: format!(
                    "the lowered name for `{symbol}` contains a NUL and cannot be looked up"
                ),
            });
        };
        let mut function: dr::CUfunction = std::ptr::null_mut();
        // SAFETY: `module` is loaded, `c_name` is NUL-terminated and outlives
        let code = unsafe { dr::cuModuleGetFunction(&raw mut function, module, c_name.as_ptr()) };
        match code {
            dr::CUresult::CUDA_SUCCESS => Ok(function),
            dr::CUresult::CUDA_ERROR_NOT_FOUND => Err(Error::Missing { unit, symbol }),
            other => Err(refused("cuModuleGetFunction", other)),
        }
    }

    /// The entry point for a row, by symbol.
    #[must_use]
    pub fn entry(&self, symbol: &str) -> Option<dr::CUfunction> {
        self.entries.iter().find(|(s, _)| *s == symbol).map(|(_, f)| *f)
    }

    /// Grant `sig`'s entry `bytes` of DYNAMIC shared memory, once per
    pub fn raise_dynamic_smem(&self, sig: &KernelSig, bytes: u32) -> Result<(), Error> {
        let Some(function) = self.entry(sig.symbol) else {
            return Err(Error::Missing { unit: self.unit, symbol: sig.symbol });
        };
        raise_dynamic_smem_cap(function, bytes)
    }

    /// How many entry points were resolved.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the row list was empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Fire `sig`'s entry with the geometry `launch` states.
    pub fn fire(
        &self,
        sig: &KernelSig,
        launch: Launch,
        args: &mut Args,
        stream: Stream<'_>,
    ) -> Result<(), Error> {
        let Some(function) = self.entry(sig.symbol) else {
            return Err(Error::Missing { unit: self.unit, symbol: sig.symbol });
        };
        if launch.grid.contains(&0) || launch.block.contains(&0) {
            return Err(Error::Geometry { symbol: sig.symbol, why: Ungeometric::Empty });
        }
        if launch.smem > DEFAULT_DYNAMIC_SMEM {
            raise_dynamic_smem_cap(function, launch.smem)?;
        }
        // SAFETY: `function` came from a module this value owns and outlives
        let code = unsafe {
            dr::cuLaunchKernel(
                function,
                launch.grid[0],
                launch.grid[1],
                launch.grid[2],
                launch.block[0],
                launch.block[1],
                launch.block[2],
                launch.smem,
                stream.as_raw().cast(),
                args.as_raw(),
                std::ptr::null_mut(),
            )
        };
        if code == dr::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(refused("cuLaunchKernel", code))
        }
    }

    /// Fire `symbol`'s entry with argument cells the CALLER owns.
    ///
    /// # Safety
    ///
    /// `args` must be a live array of pointers to argument cells, one per
    /// kernel parameter, each of the parameter's exact type and layout, all
    /// valid for the duration of the call. **Nothing checks this** — that is
    /// the whole difference between this and [`KernelModule::fire`].
    pub unsafe fn fire_raw(
        &self,
        symbol: &'static str,
        launch: Launch,
        args: &mut [*mut std::ffi::c_void],
        stream: Stream<'_>,
    ) -> Result<(), Error> {
        let Some(function) = self.entry(symbol) else {
            return Err(Error::Missing { unit: self.unit, symbol });
        };
        if launch.grid.contains(&0) || launch.block.contains(&0) {
            return Err(Error::Geometry { symbol, why: Ungeometric::Empty });
        }
        if launch.smem > DEFAULT_DYNAMIC_SMEM {
            raise_dynamic_smem_cap(function, launch.smem)?;
        }
        // SAFETY: `function` came from a module this value owns and outlives
        let code = unsafe {
            dr::cuLaunchKernel(
                function,
                launch.grid[0],
                launch.grid[1],
                launch.grid[2],
                launch.block[0],
                launch.block[1],
                launch.block[2],
                launch.smem,
                stream.as_raw().cast(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if code == dr::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(refused("cuLaunchKernel", code))
        }
    }

    /// Fire `symbol`'s entry through `cuLaunchKernelEx`, with a thread-block
    ///
    /// # Safety
    ///
    /// As [`KernelModule::fire_raw`]: `args` must be a live array of pointers
    /// to argument cells, one per kernel parameter, each of the parameter's
    /// exact type and layout, valid for the duration of the call. **Nothing
    /// checks this.**
    pub unsafe fn fire_ex(
        &self,
        symbol: &'static str,
        launch: Launch,
        cluster: Option<[u32; 3]>,
        programmatic_dependent: bool,
        cooperative: bool,
        args: &mut [*mut std::ffi::c_void],
        stream: Stream<'_>,
    ) -> Result<(), Error> {
        let Some(function) = self.entry(symbol) else {
            return Err(Error::Missing { unit: self.unit, symbol });
        };
        if launch.grid.contains(&0) || launch.block.contains(&0) {
            return Err(Error::Geometry { symbol, why: Ungeometric::Empty });
        }
        if let Some(dim) = cluster
            && dim.contains(&0)
        {
            return Err(Error::Geometry { symbol, why: Ungeometric::Empty });
        }
        if launch.smem > DEFAULT_DYNAMIC_SMEM {
            raise_dynamic_smem_cap(function, launch.smem)?;
        }

        let mut attrs: [dr::CUlaunchAttribute; 3] = unsafe { std::mem::zeroed() };
        let mut n = 0usize;
        if let Some(dim) = cluster {
            attrs[n].id = dr::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
            attrs[n].value.clusterDim = dr::CUlaunchAttributeValue_union__bindgen_ty_1 {
                x: dim[0],
                y: dim[1],
                z: dim[2],
            };
            n += 1;
        }
        if programmatic_dependent {
            attrs[n].id =
                dr::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
            attrs[n].value.programmaticStreamSerializationAllowed = 1;
            n += 1;
        }
        if cooperative {
            attrs[n].id = dr::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
            attrs[n].value.cooperative = 1;
            n += 1;
        }

        let config = dr::CUlaunchConfig {
            gridDimX: launch.grid[0],
            gridDimY: launch.grid[1],
            gridDimZ: launch.grid[2],
            blockDimX: launch.block[0],
            blockDimY: launch.block[1],
            blockDimZ: launch.block[2],
            sharedMemBytes: launch.smem,
            hStream: stream.as_raw().cast(),
            attrs: if n == 0 { std::ptr::null_mut() } else { attrs.as_mut_ptr() },
            numAttrs: n as std::ffi::c_uint,
        };

        // SAFETY: `function` came from a module this value owns and outlives
        let code = unsafe {
            dr::cuLaunchKernelEx(
                std::ptr::addr_of!(config),
                function,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if code == dr::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(refused("cuLaunchKernelEx", code))
        }
    }

    /// `cuOccupancyMaxActiveBlocksPerMultiprocessor` for one of this module's
    pub fn max_active_blocks_per_sm(
        &self,
        symbol: &'static str,
        block_threads: u32,
        dynamic_smem: u32,
    ) -> Result<u32, Error> {
        let Some(function) = self.entry(symbol) else {
            return Err(Error::Missing { unit: self.unit, symbol });
        };
        if dynamic_smem > DEFAULT_DYNAMIC_SMEM {
            raise_dynamic_smem_cap(function, dynamic_smem)?;
        }
        let mut blocks: std::ffi::c_int = 0;
        // SAFETY: `blocks` is a live out-parameter and `function` came from a
        let code = unsafe {
            dr::cuOccupancyMaxActiveBlocksPerMultiprocessor(
                &raw mut blocks,
                function,
                i32::try_from(block_threads).unwrap_or(i32::MAX),
                usize::try_from(dynamic_smem).unwrap_or(usize::MAX),
            )
        };
        if code != dr::CUresult::CUDA_SUCCESS {
            return Err(refused("cuOccupancyMaxActiveBlocksPerMultiprocessor", code));
        }
        Ok(u32::try_from(blocks).unwrap_or(0))
    }
}

impl Drop for KernelModule {
    fn drop(&mut self) {
        // SAFETY: the handle came from `cuModuleLoadData` and nothing else
        unsafe { dr::cuModuleUnload(self.module) };
    }
}

/// What a launch may ask for in dynamic shared memory before anyone has
const DEFAULT_DYNAMIC_SMEM: u32 = 48 * 1024;

/// `cuFuncSetAttribute`, once per (device, function), above the high-water
fn raise_dynamic_smem_cap(function: dr::CUfunction, bytes: u32) -> Result<(), Error> {
    if bytes <= DEFAULT_DYNAMIC_SMEM {
        return Ok(());
    }
    let mut device: dr::CUdevice = 0;
    // SAFETY: `device` is a live out-parameter. The call reads the calling
    let code = unsafe { dr::cuCtxGetDevice(&raw mut device) };
    if code != dr::CUresult::CUDA_SUCCESS {
        return Err(refused("cuCtxGetDevice", code));
    }
    let key = (device, function.addr());
    let mut granted = match GRANTED.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    if let Some((_, high_water)) = granted.iter().find(|(k, _)| *k == key) {
        if bytes <= *high_water {
            return Ok(());
        }
    }
    // SAFETY: `function` came from a loaded module and outlives the call.
    let value = i32::try_from(bytes).unwrap_or(i32::MAX);
    let code = unsafe {
        dr::cuFuncSetAttribute(
            function,
            dr::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            value,
        )
    };
    if code != dr::CUresult::CUDA_SUCCESS {
        return Err(refused("cuFuncSetAttribute", code));
    }
    match granted.iter_mut().find(|(k, _)| *k == key) {
        Some((_, high_water)) => *high_water = bytes,
        None => granted.push((key, bytes)),
    }
    Ok(())
}

/// Every (device, entry point) that has been granted more than
static GRANTED: std::sync::Mutex<Vec<((dr::CUdevice, usize), u32)>> =
    std::sync::Mutex::new(Vec::new());

/// The [`Error`] for a driver call that refused.
fn refused(what: &'static str, code: dr::CUresult) -> Error {
    Error::Driver { what, code: code as i32, why: describe(code) }
}

/// What the driver calls `code`, or the binding's name for it.
fn describe(code: dr::CUresult) -> String {
    let mut text: *const c_char = std::ptr::null();
    // SAFETY: `text` is a live out-parameter, and what is written into it is
    let got = unsafe { dr::cuGetErrorString(code, &raw mut text) };
    if got == dr::CUresult::CUDA_SUCCESS && !text.is_null() {
        // SAFETY: `text` is the driver's own NUL-terminated string, copied
        unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned()
    } else {
        format!("{code:?}")
    }
}

#[cfg(test)]
mod tests {
    use super::KernelModule;
    use crate::runtime::Error;
    use cudarc::driver::sys as dr;

    /// The unit a hand-built module claims to have come from.
    const UNIT: &str = "norm/altup_aux";

    /// A module with a null handle and entries that name nothing.
    fn stub(rows: &[&'static str]) -> std::mem::ManuallyDrop<KernelModule> {
        let unresolved: dr::CUfunction = std::ptr::null_mut();
        std::mem::ManuallyDrop::new(KernelModule {
            unit: UNIT,
            module: std::ptr::null_mut(),
            entries: rows.iter().map(|symbol| (*symbol, unresolved)).collect(),
        })
    }

    /// A row finds its own entry and no one else's. The lookup is by the
    #[test]
    fn an_entry_answers_to_the_row_that_named_it() {
        let module = stub(&["norm::tanh_bf16", "norm::compute_rms_bf16"]);
        assert_eq!(module.len(), 2);
        assert!(!module.is_empty());
        assert!(module.entry("norm::tanh_bf16").is_some());
        assert!(
            module.entry("norm::tanh_f16").is_none(),
            "a row this unit did not load must not resolve to a neighbour's entry"
        );
    }

    /// An empty row list is an empty module rather than a module of one
    #[test]
    fn a_module_of_no_rows_is_empty() {
        let module = stub(&[]);
        assert_eq!(module.len(), 0);
        assert!(module.is_empty());
    }

    /// An empty image is refused before the driver is called — which is what
    #[test]
    fn an_empty_image_never_reaches_the_driver() {
        let refusal = KernelModule::load_mangled(UNIT, &[], &[], &[])
            .err()
            .expect("an empty image has nothing to load");
        assert!(
            matches!(&refusal, Error::Compile { unit, .. } if *unit == UNIT),
            "an empty image is `{UNIT}` producing nothing, not a driver refusal: {refusal:?}"
        );
    }

    /// A module is `Send` and `Sync`, which the `unsafe impl`s above assert
    #[test]
    fn a_module_crosses_threads() {
        const fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<KernelModule>();
    }
}
