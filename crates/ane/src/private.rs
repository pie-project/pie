//! TEMPORARY: drives Apple's private AppleNeuralEngine.framework (`_ANEClient`,
//! `_ANEModel`, `_ANERequest`, ...), an undocumented interface. macOS updates may
//! change or remove it; `available()` checks every method first and pie falls
//! back to CoreML, then to the GPU. Replace once Apple ships a public API.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::{Path, PathBuf};

#[repr(C)]
struct RawSurface {
    surface: *mut c_void,
    base: *mut c_void,
    bytes: u64,
    stride: u32,
}

type Report = extern "C" fn(*mut c_void, c_int);

unsafe extern "C" {
    fn pie_ane_available(err: *mut c_char, cap: c_int) -> c_int;
    fn pie_ane_surface(
        rows: u32,
        width: u32,
        int8: c_int,
        out: *mut RawSurface,
        err: *mut c_char,
        cap: c_int,
    ) -> c_int;
    fn pie_ane_surface_free(surface: *mut c_void);
    fn pie_ane_program(
        directory: *const c_char,
        key: *const c_char,
        err: *mut c_char,
        cap: c_int,
    ) -> *mut c_void;
    fn pie_ane_program_free(program: *mut c_void);
    fn pie_ane_procedure(program: *mut c_void, function: *const c_char) -> c_int;
    fn pie_ane_input_count(program: *mut c_void, procedure: c_int) -> c_int;
    fn pie_ane_input_name(program: *mut c_void, procedure: c_int, input: c_int) -> *const c_char;
    fn pie_ane_bind(
        program: *mut c_void,
        procedure: c_int,
        inputs: *const *mut c_void,
        count: c_int,
        output: *mut c_void,
        err: *mut c_char,
        cap: c_int,
    ) -> *mut c_void;
    fn pie_ane_binding_free(binding: *mut c_void);
    fn pie_ane_enqueue(
        program: *mut c_void,
        binding: *mut c_void,
        event: *mut c_void,
        wait: u64,
        signal: u64,
        report: Report,
        context: *mut c_void,
        err: *mut c_char,
        cap: c_int,
    ) -> c_int;
}

const CAP: usize = 1024;

fn failure(err: &[c_char; CAP]) -> String {
    unsafe { CStr::from_ptr(err.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

pub fn available() -> Result<(), String> {
    let mut err = [0 as c_char; CAP];
    if unsafe { pie_ane_available(err.as_mut_ptr(), CAP as c_int) } == 1 {
        Ok(())
    } else {
        Err(failure(&err))
    }
}

pub struct Surface {
    raw: RawSurface,
    pub int8: bool,
}

unsafe impl Send for Surface {}
unsafe impl Sync for Surface {}

impl Surface {
    pub fn new(rows: u32, width: u32, int8: bool) -> Result<Surface, String> {
        let mut raw = RawSurface {
            surface: std::ptr::null_mut(),
            base: std::ptr::null_mut(),
            bytes: 0,
            stride: 0,
        };
        let mut err = [0 as c_char; CAP];
        if unsafe {
            pie_ane_surface(
                rows,
                width,
                c_int::from(int8),
                &raw mut raw,
                err.as_mut_ptr(),
                CAP as c_int,
            )
        } != 1
        {
            return Err(failure(&err));
        }
        Ok(Surface { raw, int8 })
    }

    #[must_use]
    pub fn base(&self) -> *mut u8 {
        self.raw.base.cast()
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.raw.bytes
    }

    #[must_use]
    pub fn stride(&self) -> u32 {
        self.raw.stride
    }

    fn element(&self) -> u32 {
        if self.int8 { 1 } else { 2 }
    }

    #[must_use]
    pub fn mil_type(&self) -> &'static str {
        if self.int8 { "int8" } else { "fp16" }
    }

    #[must_use]
    pub fn strides(&self, rows: u32) -> String {
        let stride = u64::from(self.raw.stride / self.element());
        let plane = u64::from(rows) * stride;
        format!("[{plane}, {plane}, {stride}, 1]")
    }

    #[must_use]
    pub fn buffer_type(&self, rows: u32, width: u32) -> String {
        format!(
            "tensor_buffer<{}, shape=[1, 1, {rows}, {width}], strides={}, interleave_factors=[1, 1, 1, 1]>",
            self.mil_type(),
            self.strides(rows)
        )
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe { pie_ane_surface_free(self.raw.surface) };
    }
}

pub const CONSTANT_OFFSET: u64 = 64;

#[must_use]
pub fn constant_blob(values: &[u16]) -> Vec<u8> {
    const DATA: usize = 128;
    let size = values.len() * 2;
    let mut blob = vec![0u8; DATA + size];
    blob[0..4].copy_from_slice(&1u32.to_le_bytes());
    blob[4..8].copy_from_slice(&2u32.to_le_bytes());
    let at = CONSTANT_OFFSET as usize;
    blob[at..at + 4].copy_from_slice(&0xdead_beef_u32.to_le_bytes());
    blob[at + 4..at + 8].copy_from_slice(&1u32.to_le_bytes());
    blob[at + 8..at + 16].copy_from_slice(&(size as u64).to_le_bytes());
    blob[at + 16..at + 24].copy_from_slice(&(DATA as u64).to_le_bytes());
    for (i, v) in values.iter().enumerate() {
        blob[DATA + 2 * i..DATA + 2 * i + 2].copy_from_slice(&v.to_le_bytes());
    }
    blob
}

#[must_use]
pub fn cache_directory() -> PathBuf {
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from);
    home.join(".cache/pie/ane-programs")
}

pub struct Program {
    raw: *mut c_void,
}

unsafe impl Send for Program {}
unsafe impl Sync for Program {}

impl Program {
    pub fn compile(mil: &str, weights: &[u8], cache: &Path) -> Result<Program, String> {
        let key = crate::fingerprint(&[mil.as_bytes(), weights].concat());
        let directory = cache.join(&key);
        std::fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
        for (name, bytes) in [("model.mil", mil.as_bytes()), ("weights.bin", weights)] {
            let path = directory.join(name);
            if std::fs::read(&path).ok().as_deref() != Some(bytes) {
                let partial = directory.join(format!("{name}.{}.partial", std::process::id()));
                std::fs::write(&partial, bytes).map_err(|e| e.to_string())?;
                std::fs::rename(&partial, &path).map_err(|e| e.to_string())?;
            }
        }
        let directory =
            CString::new(directory.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
        let key = CString::new(key).map_err(|e| e.to_string())?;
        let mut err = [0 as c_char; CAP];
        let raw = unsafe {
            pie_ane_program(
                directory.as_ptr(),
                key.as_ptr(),
                err.as_mut_ptr(),
                CAP as c_int,
            )
        };
        if raw.is_null() {
            return Err(failure(&err));
        }
        Ok(Program { raw })
    }

    #[must_use]
    pub fn procedure(&self, function: &str) -> Option<u32> {
        let function = CString::new(function).ok()?;
        let index = unsafe { pie_ane_procedure(self.raw, function.as_ptr()) };
        u32::try_from(index).ok()
    }

    #[must_use]
    pub fn inputs(&self, procedure: u32) -> Vec<String> {
        let count = unsafe { pie_ane_input_count(self.raw, procedure as c_int) };
        (0..count)
            .map(|i| {
                unsafe { CStr::from_ptr(pie_ane_input_name(self.raw, procedure as c_int, i)) }
                    .to_string_lossy()
                    .into_owned()
            })
            .collect()
    }

    pub fn bind(
        &self,
        procedure: u32,
        inputs: &[&Surface],
        output: &Surface,
    ) -> Result<Binding, String> {
        let raw: Vec<*mut c_void> = inputs.iter().map(|s| s.raw.surface).collect();
        let mut err = [0 as c_char; CAP];
        let binding = unsafe {
            pie_ane_bind(
                self.raw,
                procedure as c_int,
                raw.as_ptr(),
                raw.len() as c_int,
                output.raw.surface,
                err.as_mut_ptr(),
                CAP as c_int,
            )
        };
        if binding.is_null() {
            return Err(failure(&err));
        }
        Ok(Binding { raw: binding })
    }

    /// # Safety
    /// `event` is a live `id<MTLSharedEvent>`; the binding's surfaces outlive
    /// the evaluation.
    pub unsafe fn enqueue(
        &self,
        binding: &Binding,
        event: *mut c_void,
        wait: u64,
        signal: u64,
        report: Box<dyn FnOnce(bool) + Send>,
    ) -> Result<(), String> {
        extern "C" fn reported(context: *mut c_void, success: c_int) {
            let report: Box<Box<dyn FnOnce(bool) + Send>> =
                unsafe { Box::from_raw(context.cast()) };
            report(success == 1);
        }
        let context = Box::into_raw(Box::new(report)).cast::<c_void>();
        let mut err = [0 as c_char; CAP];
        let queued = unsafe {
            pie_ane_enqueue(
                self.raw,
                binding.raw,
                event,
                wait,
                signal,
                reported,
                context,
                err.as_mut_ptr(),
                CAP as c_int,
            )
        };
        match queued {
            1 => Ok(()),
            2 => {
                eprintln!("PIE_ANE: {}", failure(&err));
                Ok(())
            }
            _ => {
                drop(unsafe { Box::from_raw(context.cast::<Box<dyn FnOnce(bool) + Send>>()) });
                Err(failure(&err))
            }
        }
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        unsafe { pie_ane_program_free(self.raw) };
    }
}

pub struct Binding {
    raw: *mut c_void,
}

unsafe impl Send for Binding {}
unsafe impl Sync for Binding {}

impl Drop for Binding {
    fn drop(&mut self) {
        unsafe { pie_ane_binding_free(self.raw) };
    }
}
