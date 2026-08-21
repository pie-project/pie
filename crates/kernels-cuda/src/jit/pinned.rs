#[cfg(feature = "_cuda")]
use core::ffi::c_void;

#[cfg(feature = "_cuda")]
use cudarc::runtime::sys as rt;
use kernels::routine::Refusal;

pub struct PinnedBytes {
    ptr: *mut u8,
    len: usize,
    cap: usize,
}

unsafe impl Send for PinnedBytes {}

unsafe impl Sync for PinnedBytes {}

impl PinnedBytes {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            ptr: core::ptr::null_mut(),
            len: 0,
            cap: 0,
        }
    }

    pub fn with_capacity(cap: usize) -> Result<Self, Refusal> {
        if cap == 0 {
            return Ok(Self::empty());
        }
        #[cfg(feature = "_cuda")]
        {
            let mut p: *mut c_void = core::ptr::null_mut();

            let code = unsafe { rt::cudaMallocHost(&raw mut p, cap) };
            if code != rt::cudaError::cudaSuccess || p.is_null() {
                return Err(Refusal::Device {
                    why: "the pinned plan buffer could not be taken",
                });
            }
            Ok(Self {
                ptr: p.cast::<u8>(),
                len: 0,
                cap,
            })
        }

        #[cfg(not(feature = "_cuda"))]
        {
            let mut v = vec![0u8; cap];
            let p = v.as_mut_ptr();
            core::mem::forget(v);
            Ok(Self {
                ptr: p,
                len: 0,
                cap,
            })
        }
    }

    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.cap
    }

    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        if self.ptr.is_null() {
            return &[];
        }

        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }

    #[must_use]
    pub const fn as_ptr(&self) -> *const u8 {
        self.ptr.cast_const()
    }

    pub fn fill(&mut self, src: &[u8]) -> Result<bool, Refusal> {
        let mut moved = false;
        if src.len() > self.cap {
            *self = Self::with_capacity(src.len())?;
            moved = true;
        }
        if !src.is_empty() {
            unsafe { core::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr, src.len()) };
        }
        self.len = src.len();
        Ok(moved)
    }
}

impl Default for PinnedBytes {
    fn default() -> Self {
        Self::empty()
    }
}

impl core::fmt::Debug for PinnedBytes {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PinnedBytes")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .field("cap", &self.cap)
            .finish()
    }
}

impl Drop for PinnedBytes {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        #[cfg(feature = "_cuda")]
        unsafe {
            let _ = rt::cudaFreeHost(self.ptr.cast::<c_void>());
        }
        #[cfg(not(feature = "_cuda"))]
        unsafe {
            drop(Vec::from_raw_parts(self.ptr, self.len, self.cap));
        }
    }
}
