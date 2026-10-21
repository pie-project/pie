//! The mangled C++ entry points of `libsdkruntime.so` and friends, resolved
//! at runtime into plain function pointers.
//!
//! Calling conventions (System V x86-64, Itanium C++ ABI):
//! - `this` is the first integer argument;
//! - by-value `std::string` parameters travel by invisible reference (the
//!   caller owns and destroys them);
//! - `Task` is a non-trivial return type, so it comes back through a hidden
//!   pointer passed before `this`.

use super::abi::{CxxString, CxxVecU32, MemcpyOptions, Task};
use libloading::Library;
use std::ffi::c_void;

pub type ArtifactsCtor = unsafe extern "C" fn(this: *mut c_void, dir: *const CxxString);
pub type RuntimeCtor = unsafe extern "C" fn(
    this: *mut c_void,
    artifacts: *const c_void,
    platform: *const c_void,
    msg_level: *mut CxxString,
    suppress_simfab_trace: bool,
    b2: bool,
    b3: bool,
    cslc_prefix: *mut CxxString,
    out_prefix: *mut CxxString,
);
pub type ThisFn = unsafe extern "C" fn(this: *mut c_void);
pub type PlatformCtor = unsafe extern "C" fn(this: *mut c_void, cmaddr: *const CxxString);
pub type CallFn = unsafe extern "C" fn(
    ret: *mut Task,
    this: *mut c_void,
    name: *const CxxString,
    args: *const CxxVecU32<'_>,
    opts: *const MemcpyOptions,
);
pub type H2dFn = unsafe extern "C" fn(
    ret: *mut Task,
    this: *mut c_void,
    sym: u16,
    buf: *mut c_void,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    elems_per_pe: i32,
    opts: *const MemcpyOptions,
);
pub type D2hFn = unsafe extern "C" fn(
    ret: *mut Task,
    this: *mut c_void,
    buf: *mut c_void,
    sym: u16,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    elems_per_pe: i32,
    opts: *const MemcpyOptions,
);
pub type TaskFn = unsafe extern "C" fn(this: *mut c_void, task: *const Task);
pub type TaskDoneFn = unsafe extern "C" fn(this: *mut c_void, task: *const Task) -> bool;
pub type TaskDtor = unsafe extern "C" fn(task: *mut Task);

pub const ARTIFACTS_CTOR: &[u8] =
    b"_ZN8cerebras19SdkCompileArtifactsC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE\0";
pub const RUNTIME_CTOR: &[u8] = b"_ZN8cerebras10SdkRuntimeC1ERKNS_19SdkCompileArtifactsERKNS_20SdkExecutionPlatformENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEbbbSC_SC_\0";
pub const PLATFORM_CTOR: &[u8] =
    b"_ZN8cerebras20SdkExecutionPlatformC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE\0";
pub const RUNTIME_DTOR: &[u8] = b"_ZN8cerebras10SdkRuntimeD1Ev\0";
pub const LOAD: &[u8] = b"_ZN8cerebras10SdkRuntime4loadEv\0";
pub const RUN: &[u8] = b"_ZN8cerebras10SdkRuntime3runEv\0";
pub const STOP: &[u8] = b"_ZN8cerebras10SdkRuntime4stopEv\0";
pub const CALL: &[u8] = b"_ZN8cerebras10SdkRuntime4callERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKSt6vectorIjSaIjEERKNS_13MemcpyOptionsE\0";
pub const MEMCPY_H2D: &[u8] =
    b"_ZN8cerebras10SdkRuntime10memcpy_h2dEtPviiiiiRKNS_13MemcpyOptionsE\0";
pub const MEMCPY_D2H: &[u8] =
    b"_ZN8cerebras10SdkRuntime10memcpy_d2hEPvtiiiiiRKNS_13MemcpyOptionsE\0";
pub const TASK_WAIT: &[u8] = b"_ZN8cerebras10SdkRuntime9task_waitERKNS0_4TaskE\0";
pub const IS_TASK_DONE: &[u8] = b"_ZN8cerebras10SdkRuntime12is_task_doneERKNS0_4TaskE\0";
pub const TASK_DTOR: &[u8] = b"_ZN8cerebras10SdkRuntime4TaskD1Ev\0";

/// Resolved entry points. Plain `Copy` function pointers; the `Library`
/// values that back them are kept alive by [`super::Sdk`].
#[derive(Clone, Copy)]
pub struct Api {
    pub artifacts_ctor: ArtifactsCtor,
    pub platform_ctor: PlatformCtor,
    pub runtime_ctor: RuntimeCtor,
    pub runtime_dtor: ThisFn,
    pub load: ThisFn,
    pub run: ThisFn,
    pub stop: ThisFn,
    pub call: CallFn,
    pub memcpy_h2d: H2dFn,
    pub memcpy_d2h: D2hFn,
    pub task_wait: TaskFn,
    pub is_task_done: TaskDoneFn,
    pub task_dtor: TaskDtor,
}

fn sym<T: Copy>(lib: &Library, name: &[u8]) -> Result<T, super::Error> {
    // SAFETY: the caller pairs each name with the function type recovered
    // from the SDK's exported signature (see the `type` aliases above).
    let s = unsafe { lib.get::<T>(name) }.map_err(|e| {
        super::Error::MissingSymbol(
            String::from_utf8_lossy(&name[..name.len() - 1]).into_owned(),
            e.to_string(),
        )
    })?;
    Ok(*s)
}

impl Api {
    pub fn resolve(
        artifacts: &Library,
        platform: &Library,
        runtime: &Library,
    ) -> Result<Self, super::Error> {
        Ok(Api {
            artifacts_ctor: sym(artifacts, ARTIFACTS_CTOR)?,
            platform_ctor: sym(platform, PLATFORM_CTOR)?,
            runtime_ctor: sym(runtime, RUNTIME_CTOR)?,
            runtime_dtor: sym(runtime, RUNTIME_DTOR)?,
            load: sym(runtime, LOAD)?,
            run: sym(runtime, RUN)?,
            stop: sym(runtime, STOP)?,
            call: sym(runtime, CALL)?,
            memcpy_h2d: sym(runtime, MEMCPY_H2D)?,
            memcpy_d2h: sym(runtime, MEMCPY_D2H)?,
            task_wait: sym(runtime, TASK_WAIT)?,
            is_task_done: sym(runtime, IS_TASK_DONE)?,
            task_dtor: sym(runtime, TASK_DTOR)?,
        })
    }
}
