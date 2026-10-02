//! `#[repr(C)]` mirrors of the libstdc++ and SDK types that cross the ABI.
//!
//! Every layout here was recovered empirically from SDK 2.10.1 (x86_64,
//! libstdc++ 13, `_GLIBCXX_USE_CXX11_ABI=1`) by dumping the objects the SDK's
//! own pybind module builds, and is checked by the simulator round trip in
//! `examples/gemv.rs`. Nothing in this file is a public API.

use std::ffi::c_void;

/// `std::__cxx11::basic_string<char>`: `{ ptr, len, sso[16] }`.
///
/// Boxed so the small-string self-pointer never moves. Long strings own a
/// `malloc` buffer; libstdc++ frees with `operator delete`, which is
/// `malloc`-compatible, so a callee that moves the string out is fine.
#[repr(C)]
pub struct CxxString {
    ptr: *mut u8,
    len: usize,
    sso: [u8; 16],
}

impl CxxString {
    pub fn new(s: &str) -> Box<Self> {
        let mut b = Box::new(CxxString {
            ptr: std::ptr::null_mut(),
            len: s.len(),
            sso: [0; 16],
        });
        if s.len() < 16 {
            b.sso[..s.len()].copy_from_slice(s.as_bytes());
            b.ptr = b.sso.as_mut_ptr();
        } else {
            // SAFETY: plain C allocation of len + 1 bytes, filled and NUL-terminated below.
            let p = unsafe { libc::malloc(s.len() + 1) } as *mut u8;
            assert!(!p.is_null(), "malloc failed");
            unsafe {
                std::ptr::copy_nonoverlapping(s.as_ptr(), p, s.len());
                *p.add(s.len()) = 0;
            }
            b.ptr = p;
        }
        b
    }
}

impl Drop for CxxString {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.ptr != self.sso.as_mut_ptr() {
            // SAFETY: ptr came from malloc in `new` and has not been freed.
            unsafe { libc::free(self.ptr as *mut c_void) };
        }
    }
}

/// `std::vector<uint32_t>`: `{ begin, end, cap }`. Borrows a Rust slice; the
/// callee only reads it, so no capacity or ownership is involved.
#[repr(C)]
pub struct CxxVecU32<'a> {
    begin: *const u32,
    end: *const u32,
    cap: *const u32,
    _marker: std::marker::PhantomData<&'a [u32]>,
}

impl<'a> CxxVecU32<'a> {
    pub fn borrow(v: &'a [u32]) -> Self {
        let begin = v.as_ptr();
        // SAFETY: one-past-the-end pointer of a live slice.
        let end = unsafe { begin.add(v.len()) };
        CxxVecU32 {
            begin,
            end,
            cap: end,
            _marker: std::marker::PhantomData,
        }
    }
}

/// `cerebras::MemcpyOptions`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MemcpyOptions {
    pub streaming: bool,
    _pad0: [u8; 3],
    /// 0 = `MEMCPY_32BIT`, 1 = `MEMCPY_16BIT`.
    pub data_type: i32,
    /// 0 = `ROW_MAJOR`, 1 = `COL_MAJOR`.
    pub order: i32,
    pub nonblock: bool,
    _pad1: [u8; 3],
}

impl MemcpyOptions {
    pub fn new(streaming: bool, data_type: i32, order: i32, nonblock: bool) -> Self {
        MemcpyOptions {
            streaming,
            _pad0: [0; 3],
            data_type,
            order,
            nonblock,
            _pad1: [0; 3],
        }
    }
}

/// `cerebras::SdkRuntime::Task`: a `std::shared_ptr<MemcpyTask>`.
/// Returned through a hidden first argument; released with the exported dtor.
#[repr(C)]
pub struct Task {
    pub ptr: *mut c_void,
    pub ctrl: *mut c_void,
}

impl Task {
    pub fn empty() -> Self {
        Task {
            ptr: std::ptr::null_mut(),
            ctrl: std::ptr::null_mut(),
        }
    }
}

/// Opaque storage for an SDK object whose constructor we call ourselves.
/// Over-allocated and zeroed; a constructor only writes within `sizeof`.
#[repr(C, align(16))]
pub struct Opaque<const WORDS: usize>(pub [u64; WORDS]);

impl<const WORDS: usize> Opaque<WORDS> {
    pub fn zeroed() -> Box<Self> {
        Box::new(Opaque([0; WORDS]))
    }
    pub fn as_ptr(&self) -> *mut c_void {
        self.0.as_ptr() as *mut c_void
    }
}

/// `cerebras::SdkCompileArtifacts`: 72 usable bytes observed; 128 reserved.
pub type ArtifactsMem = Opaque<16>;
/// `cerebras::SdkExecutionPlatform` built by its exported string constructor
/// (a CS system at `cmaddr`): writes up to offset 73 observed; 128 reserved.
pub type SystemPlatformMem = Opaque<16>;
/// `cerebras::SdkRuntime`: a pimpl pointer (24 usable bytes observed); 32 reserved.
pub type RuntimeMem = Opaque<4>;

/// `cerebras::SdkExecutionPlatform` for the fabric simulator, byte-compatible
/// with what the SDK's pybind `get_platform(None, SimfabConfig(..), target)`
/// builds (88 usable bytes observed):
///
/// ```text
/// +0   i32  SimfabConfig::num_threads
/// +4   bool SimfabConfig::suppress_trace
/// +5   bool SimfabConfig::dump_core
/// +8   std::optional<std::filesystem::path> SimfabConfig::core_path
///        +8  string { ptr -> +24, len = 8, sso = "out.core" }
///        +40 path::_List = 3 (type _Filename, no component list)
///        +48 engaged = 1
/// +56  i32  SdkTarget (0 = WSE2, 1 = WSE3)
/// +64  24 zero bytes
/// ```
#[repr(C, align(8))]
pub struct SimPlatform([u8; 88]);

impl SimPlatform {
    pub fn new(num_threads: u32, suppress_trace: bool, dump_core: bool, wse3: bool) -> Box<Self> {
        let mut b = Box::new(SimPlatform([0; 88]));
        let core_path = b"out.core";
        let sso = b.0.as_ptr().wrapping_add(24) as usize as u64;
        b.0[0..4].copy_from_slice(&num_threads.to_le_bytes());
        b.0[4] = suppress_trace as u8;
        b.0[5] = dump_core as u8;
        b.0[8..16].copy_from_slice(&sso.to_le_bytes());
        b.0[16..24].copy_from_slice(&(core_path.len() as u64).to_le_bytes());
        b.0[24..24 + core_path.len()].copy_from_slice(core_path);
        b.0[40..48].copy_from_slice(&3u64.to_le_bytes());
        b.0[48] = 1;
        b.0[56..60].copy_from_slice(&(wse3 as u32).to_le_bytes());
        b
    }

    pub fn as_ptr(&self) -> *const c_void {
        self.0.as_ptr() as *const c_void
    }
}
