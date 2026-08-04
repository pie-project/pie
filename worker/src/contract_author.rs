//! The author-only bridge: ask the compiled driver what contract it would
//! write for a checkpoint, without booting a device.
//!
//! The C++ side (`driver/cuda/src/model/author_abi.cpp`) runs the same
//! `author_contract` family hook the load path runs and lends the result back
//! as the marshaled view the loader FFI already speaks; this side converts it
//! into the loader's own [`ModelContract`] and releases the lease. The caller
//! — `pie model convert` — then compiles and executes on the host.
//!
//! The two device-derived knobs the builder consumes (`fp8_native`,
//! `native_mxfp4_moe`) are parameters here: a convert run on the serving
//! machine answers them with a device query or its config, and an offline run
//! answers them with what the serving machine will be.

#![cfg(feature = "driver-cuda")]

use std::ffi::CString;
use std::os::raw::{c_char, c_void};
use std::path::Path;

use anyhow::{Result, anyhow};
use pie_loader::contract::ModelContract;
use pie_loader::ffi::contract::PieLoaderModelContractView;

/// The policy half of an authoring request; everything the load path reads
/// from its boot config, restated for a caller that has no boot.
pub struct AuthorRequest<'a> {
    pub snapshot_dir: &'a Path,
    /// Runtime quantization request (`""` for none), resolved against
    /// `fp8_native` exactly as the load path resolves it.
    pub runtime_quant: &'a str,
    /// `Mxfp4MoeRequest` as its wire value: 0 auto, 1 routed, 2 native, 3 bf16.
    pub mxfp4_moe_request: u32,
    /// `Component` as its wire value: 0 full, 1 text, 2 encode.
    pub component: u32,
    pub stream_routed_experts: bool,
    pub tp_rank: u32,
    pub tp_size: u32,
    pub fp8_native: bool,
    pub native_mxfp4_moe: bool,
}

unsafe extern "C" {
    fn pie_cuda_author_contract(
        snapshot_dir: *const c_char,
        runtime_quant: *const c_char,
        mxfp4_moe_request: u32,
        component: u32,
        stream_routed_experts: i32,
        tp_rank: u32,
        tp_size: u32,
        fp8_native: i32,
        native_mxfp4_moe: i32,
        out_owner: *mut *mut c_void,
        out_view: *mut PieLoaderModelContractView,
    ) -> i32;
    fn pie_cuda_author_release(owner: *mut c_void);
}

/// Author the load contract the CUDA driver would write for this request.
pub fn author_contract(request: &AuthorRequest<'_>) -> Result<ModelContract> {
    let snapshot = CString::new(request.snapshot_dir.to_string_lossy().as_bytes())
        .map_err(|_| anyhow!("snapshot path contains a NUL byte"))?;
    let runtime_quant = CString::new(request.runtime_quant)
        .map_err(|_| anyhow!("runtime_quant contains a NUL byte"))?;

    let mut owner: *mut c_void = std::ptr::null_mut();
    let mut view = std::mem::MaybeUninit::<PieLoaderModelContractView>::zeroed();
    let status = unsafe {
        pie_cuda_author_contract(
            snapshot.as_ptr(),
            runtime_quant.as_ptr(),
            request.mxfp4_moe_request,
            request.component,
            i32::from(request.stream_routed_experts),
            request.tp_rank,
            request.tp_size,
            i32::from(request.fp8_native),
            i32::from(request.native_mxfp4_moe),
            &mut owner,
            view.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(anyhow!(
            "the driver declined to author a contract (status {status}; its \
             reason is on stderr)"
        ));
    }
    // The view borrows from `owner`; convert before releasing, and release on
    // every path.
    let view = unsafe { view.assume_init() };
    let contract = unsafe { pie_loader::ffi::contract::read_contract(&view) };
    unsafe { pie_cuda_author_release(owner) };
    contract.map_err(|err| anyhow!("the authored contract did not read back: {err}"))
}
