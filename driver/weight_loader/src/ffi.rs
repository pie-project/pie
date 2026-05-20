use crate::error::{CompileError, clear_error, write_error};
use crate::ffi_arena::FfiArena;
use crate::ffi_types::{
    PieLoaderBackendKind, PieLoaderCompileInput, PieLoaderError, PieLoaderStatus,
    PieLoaderStorageProgramView,
};
use crate::source::{CheckpointMetadata, files_from_ffi, tensors_from_ffi};
use crate::storage::{StorageProgram, StorageTarget};
use crate::storage_compiler::compile_storage_program;
use crate::types::BackendKind;

pub struct PieLoaderProgramHandle {
    program: StorageProgram,
    arena: FfiArena,
}

impl PieLoaderProgramHandle {
    fn new(program: StorageProgram) -> Self {
        let arena = FfiArena::from_program(&program);
        Self { program, arena }
    }

    fn view(&self) -> PieLoaderStorageProgramView {
        self.arena.view(&self.program)
    }
}

fn compile(input: &PieLoaderCompileInput) -> Result<StorageProgram, CompileError> {
    validate_input(input)?;
    let files = unsafe { ffi_slice(input.files.ptr, input.files.len, "files")? };
    let tensors = unsafe { ffi_slice(input.tensors.ptr, input.tensors.len, "tensors")? };
    let metadata = CheckpointMetadata {
        files: files_from_ffi(files)?,
        tensors: tensors_from_ffi(tensors)?,
    };
    let model = crate::config::ModelConfig::from_ffi(&input.model)?;
    let target = StorageTarget {
        backend: backend_kind(input.target.backend),
        tp_rank: input.target.tp_rank,
        tp_size: input.target.tp_size.max(1),
        max_tile_bytes: input.target.max_tile_bytes,
        preferred_alignment: input.target.preferred_alignment.max(1),
        mxfp4_moe: mxfp4_policy(input.target.mxfp4_moe),
        native_mxfp4_moe: input.target.native_mxfp4_moe,
    };
    let abi = if input.runtime_abi.tensors.len == 0 {
        crate::abi::RuntimeAbi::default_for_target(&metadata, &model, &target)?
    } else {
        crate::abi::RuntimeAbi::from_ffi(&input.runtime_abi)?
    };
    compile_storage_program(&metadata, &model, &abi, target)
}

fn mxfp4_policy(kind: crate::ffi_types::PieLoaderMxfp4MoePolicy) -> crate::types::Mxfp4MoePolicy {
    match kind {
        crate::ffi_types::PieLoaderMxfp4MoePolicy::RoutedDecode => {
            crate::types::Mxfp4MoePolicy::RoutedDecode
        }
        crate::ffi_types::PieLoaderMxfp4MoePolicy::NativeGemm => {
            crate::types::Mxfp4MoePolicy::NativeGemm
        }
        crate::ffi_types::PieLoaderMxfp4MoePolicy::EagerBf16 => {
            crate::types::Mxfp4MoePolicy::EagerBf16
        }
    }
}

fn backend_kind(kind: PieLoaderBackendKind) -> BackendKind {
    match kind {
        PieLoaderBackendKind::Cuda => BackendKind::Cuda,
        PieLoaderBackendKind::Portable => BackendKind::Portable,
        PieLoaderBackendKind::Unknown => BackendKind::Unknown,
    }
}

fn validate_input(input: &PieLoaderCompileInput) -> Result<(), CompileError> {
    validate_slice(input.files.ptr.cast::<u8>(), input.files.len, "files")?;
    validate_slice(input.tensors.ptr.cast::<u8>(), input.tensors.len, "tensors")?;
    validate_slice(
        input.runtime_abi.tensors.ptr.cast::<u8>(),
        input.runtime_abi.tensors.len,
        "runtime_abi.tensors",
    )?;
    if input.target.tp_size == 0 {
        return Err(CompileError::InvalidInput(
            "target.tp_size must be non-zero".to_string(),
        ));
    }
    if input.target.tp_rank >= input.target.tp_size {
        return Err(CompileError::InvalidInput(format!(
            "target.tp_rank {} must be less than tp_size {}",
            input.target.tp_rank, input.target.tp_size
        )));
    }
    Ok(())
}

fn validate_slice(ptr: *const u8, len: usize, name: &'static str) -> Result<(), CompileError> {
    if len > 0 && ptr.is_null() {
        return Err(CompileError::NullArgument(name));
    }
    Ok(())
}

unsafe fn ffi_slice<'a, T>(
    ptr: *const T,
    len: usize,
    name: &'static str,
) -> Result<&'a [T], CompileError> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(CompileError::NullArgument(name));
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

#[unsafe(no_mangle)]
/// Compile a flat weight-loading storage program from C ABI input views.
///
/// # Safety
///
/// `input`, `out_program`, and all non-empty slices reachable from `input`
/// must be valid for reads for the duration of the call. `out_program` must be
/// valid for one pointer write. `out_error` may be null; when non-null it must
/// point to a valid `PieLoaderError` owned by the caller.
pub unsafe extern "C" fn pie_loader_compile(
    input: *const PieLoaderCompileInput,
    out_program: *mut *mut PieLoaderProgramHandle,
    out_error: *mut PieLoaderError,
) -> PieLoaderStatus {
    unsafe { clear_error(out_error) };
    if out_program.is_null() {
        let err = CompileError::NullArgument("out_program");
        unsafe { write_error(out_error, &err) };
        return err.status();
    }
    unsafe {
        *out_program = std::ptr::null_mut();
    }
    if input.is_null() {
        let err = CompileError::NullArgument("input");
        unsafe { write_error(out_error, &err) };
        return err.status();
    }

    match compile(unsafe { &*input }) {
        Ok(program) => {
            let handle = Box::new(PieLoaderProgramHandle::new(program));
            unsafe {
                *out_program = Box::into_raw(handle);
            }
            PieLoaderStatus::Ok
        }
        Err(err) => {
            let status = err.status();
            unsafe { write_error(out_error, &err) };
            status
        }
    }
}

#[unsafe(no_mangle)]
/// Borrow a flat view of a compiled program handle.
///
/// # Safety
///
/// `program` may be null. When non-null, it must be a handle returned by
/// `pie_loader_compile` that has not been freed. The returned view borrows
/// memory owned by the handle and becomes invalid when the handle is freed.
pub unsafe extern "C" fn pie_loader_program_view(
    program: *const PieLoaderProgramHandle,
) -> PieLoaderStorageProgramView {
    if program.is_null() {
        return PieLoaderStorageProgramView::default();
    }
    unsafe { (&*program).view() }
}

#[unsafe(no_mangle)]
/// Free a compiled program handle.
///
/// # Safety
///
/// `program` may be null. When non-null, it must be a handle returned by
/// `pie_loader_compile` and must not be freed more than once.
pub unsafe extern "C" fn pie_loader_program_free(program: *mut PieLoaderProgramHandle) {
    if !program.is_null() {
        drop(unsafe { Box::from_raw(program) });
    }
}

#[unsafe(no_mangle)]
/// Free the message currently owned by an FFI error object.
///
/// # Safety
///
/// `error` may be null. When non-null, it must point to a valid
/// `PieLoaderError` whose `message` is either null or was produced by this
/// crate.
pub unsafe extern "C" fn pie_loader_error_free(error: *mut PieLoaderError) {
    unsafe { clear_error(error) };
}
