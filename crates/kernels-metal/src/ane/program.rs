//! A MIL program compiled for and loaded on the Neural Engine, and the
//! bindings that tie its procedures to surfaces.

use std::ffi::{CStr, CString, c_int, c_void};
use std::path::Path;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLSharedEvent;

use super::surface::Surface;
use super::sys;

/// Where the constant's bytes begin inside the blob [`constant_blob`] builds,
/// so a MIL `BLOBFILE` reference knows its `offset`.
pub const CONSTANT_OFFSET: u64 = 64;

/// A `weights.bin` holding one fp16 constant, in the blob format the Neural
/// Engine compiler reads: a header, one descriptor at [`CONSTANT_OFFSET`],
/// then the values.
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

/// FNV-1a of `bytes`, as the name of a program's cache directory.
#[must_use]
pub fn fingerprint(bytes: &[u8]) -> String {
    let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    });
    format!("{hash:016x}")
}

/// The loaded program, shared: a hand-off's listener holds one past the
/// site's own borrow.
#[derive(Clone)]
pub struct Program(Arc<RawProgram>);

struct RawProgram(*mut c_void);

unsafe impl Send for RawProgram {}
unsafe impl Sync for RawProgram {}

impl Drop for RawProgram {
    fn drop(&mut self) {
        unsafe { sys::pie_ane_program_free(self.0) };
    }
}

impl Program {
    /// Writes `mil` and `weights` under `cache`, keyed by their fingerprint,
    /// and compiles and loads them. A program compiled earlier from the same
    /// text loads without compiling again.
    pub fn compile(mil: &str, weights: &[u8], cache: &Path) -> Result<Program, String> {
        let key = fingerprint(&[mil.as_bytes(), weights].concat());
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
        let mut err = sys::message();
        let raw = unsafe {
            sys::pie_ane_program(
                directory.as_ptr(),
                key.as_ptr(),
                err.as_mut_ptr(),
                sys::CAP as c_int,
            )
        };
        if raw.is_null() {
            return Err(sys::failure(&err));
        }
        Ok(Program(Arc::new(RawProgram(raw))))
    }

    fn raw(&self) -> *mut c_void {
        self.0.0
    }

    /// The index of the procedure named `function`, if the program has one.
    #[must_use]
    pub fn procedure(&self, function: &str) -> Option<u32> {
        let function = CString::new(function).ok()?;
        let index = unsafe { sys::pie_ane_procedure(self.raw(), function.as_ptr()) };
        u32::try_from(index).ok()
    }

    /// The procedure's input names, in the order a binding supplies them.
    #[must_use]
    pub fn inputs(&self, procedure: u32) -> Vec<String> {
        let count = unsafe { sys::pie_ane_input_count(self.raw(), procedure as c_int) };
        (0..count)
            .map(|i| {
                let name = unsafe { sys::pie_ane_input_name(self.raw(), procedure as c_int, i) };
                unsafe { CStr::from_ptr(name) }
                    .to_string_lossy()
                    .into_owned()
            })
            .collect()
    }

    /// Ties the procedure's inputs, in [`Program::inputs`] order, and its one
    /// output to surfaces.
    pub fn bind(
        &self,
        procedure: u32,
        inputs: &[&Surface],
        output: &Surface,
    ) -> Result<Binding, String> {
        let raw: Vec<*mut c_void> = inputs.iter().map(|s| s.handle()).collect();
        let mut err = sys::message();
        let binding = unsafe {
            sys::pie_ane_bind(
                self.raw(),
                procedure as c_int,
                raw.as_ptr(),
                raw.len() as c_int,
                output.handle(),
                err.as_mut_ptr(),
                sys::CAP as c_int,
            )
        };
        if binding.is_null() {
            return Err(sys::failure(&err));
        }
        Ok(Binding(Arc::new(RawBinding(binding))))
    }

    /// Queues one evaluation of `binding`: the Neural Engine waits for `event`
    /// to reach `wait`, runs, and signals `signal`. `report` is called exactly
    /// once with whether the evaluation succeeded; when the request could not
    /// be handed over at all this returns `Err` and `report` is dropped
    /// uncalled.
    ///
    /// # Safety
    /// The binding's surfaces outlive the evaluation.
    pub unsafe fn enqueue(
        &self,
        binding: &Binding,
        event: &Retained<ProtocolObject<dyn MTLSharedEvent>>,
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
        let event = Retained::as_ptr(event).cast_mut().cast::<c_void>();
        let mut err = sys::message();
        let queued = unsafe {
            sys::pie_ane_enqueue(
                self.raw(),
                binding.0.0,
                event,
                wait,
                signal,
                reported,
                context,
                err.as_mut_ptr(),
                sys::CAP as c_int,
            )
        };
        match queued {
            1 => Ok(()),
            // Handed over, then refused: `report` has already fired with
            // `false`, so the caller learns of it there; the reason is only
            // here.
            2 => {
                eprintln!("PIE_ANE: {}", sys::failure(&err));
                Ok(())
            }
            _ => {
                drop(unsafe { Box::from_raw(context.cast::<Box<dyn FnOnce(bool) + Send>>()) });
                Err(sys::failure(&err))
            }
        }
    }
}

/// One procedure's inputs and output, resolved to surfaces; shared like
/// the program.
#[derive(Clone)]
pub struct Binding(Arc<RawBinding>);

struct RawBinding(*mut c_void);

unsafe impl Send for RawBinding {}
unsafe impl Sync for RawBinding {}

impl Drop for RawBinding {
    fn drop(&mut self) {
        unsafe { sys::pie_ane_binding_free(self.0) };
    }
}
