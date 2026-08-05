//! The materialized-weight artifact, across the ABI.
//!
//! [`pie_loader::weight_store`] owns the format; this is how the driver drives it.
//! Two shapes, because the two directions are not symmetric.
//!
//! Writing is a **stream**. The payloads live in device memory and are copied
//! back a chunk at a time, so the driver opens the artifact, declares a tensor,
//! hands over its bytes as they arrive, and closes it. Nothing here holds a
//! whole tensor, which is the point: a store is tens of gigabytes.
//!
//! Reading is a **view**. The file is mapped, so each tensor's bytes are a
//! pointer into the mapping that stays valid until the handle is closed — the
//! driver stages them to the device straight from there, with no host copy in
//! between and no second parse of the file.
//!
//! Both directions carry two driver-owned strings the loader stores verbatim
//! and never interprets: a tensor's runtime dtype and a quantization entry's
//! scale kind. The driver's type vocabulary is larger than the loader's (it has
//! tile-packed types the loader has no name for) and it moves independently, so
//! mirroring it here would be a second copy to keep in step. What the loader
//! does interpret is `dtype`: the physical type the bytes are stored as, which
//! is what makes the object well-formed to any reader of the container.

use std::ffi::c_void;
use std::path::Path;

use pie_loader::types::DType;
use pie_loader::weight_store::{Artifact, ArtifactWriter, Quant, Tensor, View};

use super::entry::{PieLoaderDiagnostics, PieLoaderStatus};
use super::types::{PieLoaderBytes, PieLoaderDType, PieLoaderI64Slice, PieLoaderSlice};

// =======================================================================
// Vocabulary
// =======================================================================

/// A tensor with bytes of its own, as the driver declares it.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderWeightTensor {
    pub name: PieLoaderBytes,
    /// How the bytes are stored. The payload's length is
    /// `product(shape) × width(dtype)`, and the driver's own type must agree
    /// with that or the artifact would describe something else.
    pub dtype: PieLoaderDType,
    /// The driver's name for the type its kernels will read. Stored verbatim.
    pub runtime_dtype: PieLoaderBytes,
    pub shape: PieLoaderI64Slice,
}

/// A tensor as the artifact reports it back.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderWeightTensorView {
    pub name: PieLoaderBytes,
    pub runtime_dtype: PieLoaderBytes,
    pub shape: PieLoaderI64Slice,
    /// The payload, borrowed from the mapping. Valid until the handle closes.
    pub data: *const u8,
    pub nbytes: u64,
    /// Where the payload begins in the file. What a caller checks when it wants
    /// to know the artifact is mappable — a question the bytes cannot answer.
    pub file_offset: u64,
}

/// A tensor that is a window into another tensor's bytes.
///
/// Reconstructed as a window, never as a copy: views are how a fused projection
/// or a stacked expert weight is addressed, and copying them back would double
/// the memory they exist to save.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderWeightView {
    pub name: PieLoaderBytes,
    /// The tensor whose bytes this window lies in. Empty for a window with no
    /// bytes at all, which a store is allowed to hold.
    pub root: PieLoaderBytes,
    pub byte_offset: u64,
    pub dtype: PieLoaderDType,
    pub runtime_dtype: PieLoaderBytes,
    pub shape: PieLoaderI64Slice,
    /// What the store records this window as cut from, which is not always the
    /// root: a view of a view names the intermediate.
    pub backing: PieLoaderBytes,
}

/// A quantized weight's companion metadata.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieLoaderWeightQuant {
    pub name: PieLoaderBytes,
    /// The driver's name for the granularity its kernels expect. Stored verbatim.
    pub kind: PieLoaderBytes,
    pub scale: PieLoaderBytes,
    pub zero_point: PieLoaderBytes,
    pub group_size: i32,
    pub channel_axis: i32,
}

pub type PieLoaderWeightTensorSlice = PieLoaderSlice<PieLoaderWeightTensorView>;
pub type PieLoaderWeightViewSlice = PieLoaderSlice<PieLoaderWeightView>;
pub type PieLoaderWeightQuantSlice = PieLoaderSlice<PieLoaderWeightQuant>;

/// An artifact open for writing.
///
/// Freed by [`pie_loader_weight_store_publish`] or
/// [`pie_loader_weight_store_discard`]; nothing reaches the destination path
/// until the first of those succeeds.
#[repr(C)]
#[derive(Debug)]
pub struct PieLoaderWeightWriter {
    owner: *mut c_void,
}

/// An opened artifact. Freed with [`pie_loader_weight_store_close`].
#[repr(C)]
#[derive(Debug)]
pub struct PieLoaderWeightStore {
    pub tensors: PieLoaderWeightTensorSlice,
    pub views: PieLoaderWeightViewSlice,
    pub quants: PieLoaderWeightQuantSlice,
    owner: *mut c_void,
}

// =======================================================================
// Arenas
// =======================================================================

/// Owns the writer and the message from whatever last went wrong on it.
///
/// A streaming call has no room for diagnostics — it is called once per chunk —
/// so the message is kept here and fetched by
/// [`pie_loader_weight_store_error`] when a status says there is one.
struct WriterArena {
    writer: Option<ArtifactWriter>,
    error: Option<Box<[u8]>>,
}

impl WriterArena {
    fn fail(&mut self, message: String) -> PieLoaderStatus {
        self.error = Some(message.into_bytes().into_boxed_slice());
        PieLoaderStatus::InvalidCheckpoint
    }
}

/// Owns the mapping, the marshalled views into it, and their strings.
struct StoreArena {
    artifact: Artifact,
    strings: Vec<Box<[u8]>>,
    shapes: Vec<Box<[i64]>>,
    tensors: Vec<PieLoaderWeightTensorView>,
    views: Vec<PieLoaderWeightView>,
    quants: Vec<PieLoaderWeightQuant>,
}

impl StoreArena {
    fn store_str(&mut self, value: &str) -> PieLoaderBytes {
        let boxed: Box<[u8]> = value.as_bytes().to_vec().into_boxed_slice();
        let view = PieLoaderBytes {
            ptr: boxed.as_ptr(),
            len: boxed.len(),
        };
        self.strings.push(boxed);
        view
    }

    fn store_shape(&mut self, values: &[i64]) -> PieLoaderI64Slice {
        let boxed: Box<[i64]> = values.to_vec().into_boxed_slice();
        let view = PieLoaderI64Slice {
            ptr: boxed.as_ptr(),
            len: boxed.len(),
        };
        self.shapes.push(boxed);
        view
    }
}

// =======================================================================
// Marshalling
// =======================================================================

/// # Safety
///
/// `value` points at `value.len` initialized bytes live for the call.
unsafe fn string_of(value: &PieLoaderBytes, field: &str) -> Result<String, String> {
    unsafe { super::entry::as_str(value, field) }.map(str::to_string)
}

/// # Safety
///
/// `value.ptr` points at `value.len` initialized `i64`s live for the call.
unsafe fn shape_of(value: &PieLoaderI64Slice) -> Vec<i64> {
    if value.ptr.is_null() || value.len == 0 {
        return Vec::new();
    }
    unsafe { std::slice::from_raw_parts(value.ptr, value.len) }.to_vec()
}

/// # Safety
///
/// Every borrowed field of `decl` is live for the call.
unsafe fn read_tensor(decl: &PieLoaderWeightTensor) -> Result<Tensor, String> {
    Ok(Tensor {
        name: unsafe { string_of(&decl.name, "tensor.name") }?,
        dtype: DType::from(decl.dtype),
        runtime_dtype: unsafe { string_of(&decl.runtime_dtype, "tensor.runtime_dtype") }?,
        shape: unsafe { shape_of(&decl.shape) },
    })
}

/// # Safety
///
/// Every borrowed field of `decl` is live for the call.
unsafe fn read_view(decl: &PieLoaderWeightView) -> Result<View, String> {
    Ok(View {
        name: unsafe { string_of(&decl.name, "view.name") }?,
        root: unsafe { string_of(&decl.root, "view.root") }?,
        byte_offset: decl.byte_offset,
        dtype: DType::from(decl.dtype),
        runtime_dtype: unsafe { string_of(&decl.runtime_dtype, "view.runtime_dtype") }?,
        shape: unsafe { shape_of(&decl.shape) },
        backing: unsafe { string_of(&decl.backing, "view.backing") }?,
    })
}

/// # Safety
///
/// Every borrowed field of `decl` is live for the call.
unsafe fn read_quant(decl: &PieLoaderWeightQuant) -> Result<Quant, String> {
    Ok(Quant {
        name: unsafe { string_of(&decl.name, "quant.name") }?,
        kind: unsafe { string_of(&decl.kind, "quant.kind") }?,
        scale: unsafe { string_of(&decl.scale, "quant.scale") }?,
        zero_point: unsafe { string_of(&decl.zero_point, "quant.zero_point") }?,
        group_size: decl.group_size,
        channel_axis: decl.channel_axis,
    })
}

/// Marshal an opened artifact into a handle.
fn build(artifact: Artifact) -> Result<*mut PieLoaderWeightStore, String> {
    let mut arena = StoreArena {
        artifact,
        strings: Vec::new(),
        shapes: Vec::new(),
        tensors: Vec::new(),
        views: Vec::new(),
        quants: Vec::new(),
    };

    let tensors = arena.artifact.tensors().to_vec();
    for (index, tensor) in tensors.iter().enumerate() {
        let bytes = arena.artifact.bytes(index).map_err(|e| e.to_string())?;
        let (data, nbytes) = (bytes.as_ptr(), bytes.len() as u64);
        let file_offset = arena
            .artifact
            .file_offset(index)
            .map_err(|e| e.to_string())?;
        let name = arena.store_str(&tensor.name);
        let runtime_dtype = arena.store_str(&tensor.runtime_dtype);
        let shape = arena.store_shape(&tensor.shape);
        arena.tensors.push(PieLoaderWeightTensorView {
            name,
            runtime_dtype,
            shape,
            data,
            nbytes,
            file_offset,
        });
    }

    for view in arena.artifact.views().to_vec() {
        let name = arena.store_str(&view.name);
        let root = arena.store_str(&view.root);
        let runtime_dtype = arena.store_str(&view.runtime_dtype);
        let backing = arena.store_str(&view.backing);
        let shape = arena.store_shape(&view.shape);
        arena.views.push(PieLoaderWeightView {
            name,
            root,
            byte_offset: view.byte_offset,
            dtype: PieLoaderDType::from(view.dtype),
            runtime_dtype,
            shape,
            backing,
        });
    }

    for quant in arena.artifact.quants().to_vec() {
        let name = arena.store_str(&quant.name);
        let kind = arena.store_str(&quant.kind);
        let scale = arena.store_str(&quant.scale);
        let zero_point = arena.store_str(&quant.zero_point);
        arena.quants.push(PieLoaderWeightQuant {
            name,
            kind,
            scale,
            zero_point,
            group_size: quant.group_size,
            channel_axis: quant.channel_axis,
        });
    }

    let header = PieLoaderWeightStore {
        tensors: PieLoaderWeightTensorSlice {
            ptr: arena.tensors.as_ptr(),
            len: arena.tensors.len(),
        },
        views: PieLoaderWeightViewSlice {
            ptr: arena.views.as_ptr(),
            len: arena.views.len(),
        },
        quants: PieLoaderWeightQuantSlice {
            ptr: arena.quants.as_ptr(),
            len: arena.quants.len(),
        },
        owner: std::ptr::null_mut(),
    };
    let owner = Box::into_raw(Box::new(arena)).cast::<c_void>();
    Ok(Box::into_raw(Box::new(PieLoaderWeightStore {
        owner,
        ..header
    })))
}

/// # Safety
///
/// `handle` is a live pointer from [`pie_loader_weight_store_create`].
unsafe fn writer_of<'a>(handle: *mut PieLoaderWeightWriter) -> Option<&'a mut WriterArena> {
    if handle.is_null() {
        return None;
    }
    unsafe { (*handle).owner.cast::<WriterArena>().as_mut() }
}

// =======================================================================
// Entry points — writing
// =======================================================================

/// Open an artifact for `key` at `path`.
///
/// Nothing appears at `path` until [`pie_loader_weight_store_publish`]
/// succeeds: the bytes go to a neighbouring file that is renamed into place, so
/// a run that dies mid-write leaves nothing a later boot would try to read.
///
/// # Safety
///
/// `path` and `key` are valid [`PieLoaderBytes`] live for the call. `out` is a
/// writable slot. `out_diags` is null or a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_loader_weight_store_create(
    path: PieLoaderBytes,
    key: PieLoaderBytes,
    out: *mut *mut PieLoaderWeightWriter,
    out_diags: *mut *mut PieLoaderDiagnostics,
) -> PieLoaderStatus {
    unsafe { super::entry::emit_error(out_diags, None) };
    if out.is_null() {
        return PieLoaderStatus::InvalidRequest;
    }
    unsafe { *out = std::ptr::null_mut() };

    let opened = (|| {
        let path = unsafe { super::entry::as_str(&path, "path") }
            .map_err(|err| (PieLoaderStatus::InvalidRequest, err))?;
        let key = unsafe { super::entry::as_str(&key, "key") }
            .map_err(|err| (PieLoaderStatus::InvalidRequest, err))?;
        ArtifactWriter::create(Path::new(path), key)
            .map_err(|err| (PieLoaderStatus::InvalidCheckpoint, err.to_string()))
    })();

    match opened {
        Ok(writer) => {
            let arena = WriterArena {
                writer: Some(writer),
                error: None,
            };
            let owner = Box::into_raw(Box::new(arena)).cast::<c_void>();
            unsafe { *out = Box::into_raw(Box::new(PieLoaderWeightWriter { owner })) };
            PieLoaderStatus::Ok
        }
        Err((status, message)) => {
            unsafe { super::entry::emit_error(out_diags, Some(message)) };
            status
        }
    }
}

/// Declare a tensor and open it for writing.
///
/// Its payload is `product(shape) × width(dtype)` bytes, delivered by
/// [`pie_loader_weight_store_write`] and closed by
/// [`pie_loader_weight_store_end_tensor`].
///
/// # Safety
///
/// `writer` is a live handle and every borrowed field of `decl` is live for the
/// call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_loader_weight_store_begin_tensor(
    writer: *mut PieLoaderWeightWriter,
    decl: *const PieLoaderWeightTensor,
) -> PieLoaderStatus {
    let Some(arena) = (unsafe { writer_of(writer) }) else {
        return PieLoaderStatus::InvalidRequest;
    };
    let Some(decl) = (unsafe { decl.as_ref() }) else {
        return arena.fail("begin_tensor: null declaration".into());
    };
    let tensor = match unsafe { read_tensor(decl) } {
        Ok(tensor) => tensor,
        Err(message) => return arena.fail(message),
    };
    match arena.writer.as_mut() {
        Some(w) => match w.begin_tensor(tensor) {
            Ok(()) => PieLoaderStatus::Ok,
            Err(err) => arena.fail(err.to_string()),
        },
        None => arena.fail("begin_tensor: the artifact is already published".into()),
    }
}

/// Append bytes to the open tensor.
///
/// # Safety
///
/// `writer` is a live handle and `data` points at `len` initialized bytes live
/// for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_loader_weight_store_write(
    writer: *mut PieLoaderWeightWriter,
    data: *const u8,
    len: usize,
) -> PieLoaderStatus {
    let Some(arena) = (unsafe { writer_of(writer) }) else {
        return PieLoaderStatus::InvalidRequest;
    };
    if data.is_null() && len != 0 {
        return arena.fail("write: null pointer with non-zero length".into());
    }
    let chunk = if len == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(data, len) }
    };
    match arena.writer.as_mut() {
        Some(w) => match w.write(chunk) {
            Ok(()) => PieLoaderStatus::Ok,
            Err(err) => arena.fail(err.to_string()),
        },
        None => arena.fail("write: the artifact is already published".into()),
    }
}

/// Close the open tensor, which must have received its whole payload.
///
/// # Safety
///
/// `writer` is a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_loader_weight_store_end_tensor(
    writer: *mut PieLoaderWeightWriter,
) -> PieLoaderStatus {
    let Some(arena) = (unsafe { writer_of(writer) }) else {
        return PieLoaderStatus::InvalidRequest;
    };
    match arena.writer.as_mut() {
        Some(w) => match w.end_tensor() {
            Ok(()) => PieLoaderStatus::Ok,
            Err(err) => arena.fail(err.to_string()),
        },
        None => arena.fail("end_tensor: the artifact is already published".into()),
    }
}

/// Record a window into a tensor's bytes.
///
/// # Safety
///
/// `writer` is a live handle and every borrowed field of `decl` is live for the
/// call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_loader_weight_store_add_view(
    writer: *mut PieLoaderWeightWriter,
    decl: *const PieLoaderWeightView,
) -> PieLoaderStatus {
    let Some(arena) = (unsafe { writer_of(writer) }) else {
        return PieLoaderStatus::InvalidRequest;
    };
    let Some(decl) = (unsafe { decl.as_ref() }) else {
        return arena.fail("add_view: null declaration".into());
    };
    let view = match unsafe { read_view(decl) } {
        Ok(view) => view,
        Err(message) => return arena.fail(message),
    };
    match arena.writer.as_mut() {
        Some(w) => {
            w.add_view(view);
            PieLoaderStatus::Ok
        }
        None => arena.fail("add_view: the artifact is already published".into()),
    }
}

/// Record a quantized weight's companion metadata.
///
/// # Safety
///
/// `writer` is a live handle and every borrowed field of `decl` is live for the
/// call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_loader_weight_store_add_quant(
    writer: *mut PieLoaderWeightWriter,
    decl: *const PieLoaderWeightQuant,
) -> PieLoaderStatus {
    let Some(arena) = (unsafe { writer_of(writer) }) else {
        return PieLoaderStatus::InvalidRequest;
    };
    let Some(decl) = (unsafe { decl.as_ref() }) else {
        return arena.fail("add_quant: null declaration".into());
    };
    let quant = match unsafe { read_quant(decl) } {
        Ok(quant) => quant,
        Err(message) => return arena.fail(message),
    };
    match arena.writer.as_mut() {
        Some(w) => {
            w.add_quant(quant);
            PieLoaderStatus::Ok
        }
        None => arena.fail("add_quant: the artifact is already published".into()),
    }
}

/// Close the artifact and move it into place.
///
/// The handle is freed either way; a failure leaves nothing at the destination.
/// On failure the message is *not* retrievable — the handle it would have lived
/// in is gone — so it is written through `out_diags` instead.
///
/// # Safety
///
/// `writer` is a live handle, not used again after this call. `out_diags` is
/// null or a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_loader_weight_store_publish(
    writer: *mut PieLoaderWeightWriter,
    out_diags: *mut *mut PieLoaderDiagnostics,
) -> PieLoaderStatus {
    unsafe { super::entry::emit_error(out_diags, None) };
    if writer.is_null() {
        return PieLoaderStatus::InvalidRequest;
    }
    let boxed = unsafe { Box::from_raw(writer) };
    if boxed.owner.is_null() {
        return PieLoaderStatus::InvalidRequest;
    }
    let mut arena = unsafe { Box::from_raw(boxed.owner.cast::<WriterArena>()) };
    let Some(inner) = arena.writer.take() else {
        return PieLoaderStatus::InvalidRequest;
    };
    match inner.finish() {
        Ok(()) => PieLoaderStatus::Ok,
        Err(err) => {
            unsafe { super::entry::emit_error(out_diags, Some(err.to_string())) };
            PieLoaderStatus::InvalidCheckpoint
        }
    }
}

/// Abandon an artifact under construction, removing the partial file.
///
/// # Safety
///
/// `writer` is null, or a live handle not used again after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_loader_weight_store_discard(writer: *mut PieLoaderWeightWriter) {
    if writer.is_null() {
        return;
    }
    let boxed = unsafe { Box::from_raw(writer) };
    if !boxed.owner.is_null() {
        drop(unsafe { Box::from_raw(boxed.owner.cast::<WriterArena>()) });
    }
}

/// The message from whatever last failed on `writer`.
///
/// Empty when nothing has. Borrowed from the handle and valid until the next
/// failing call on it or until the handle is freed.
///
/// # Safety
///
/// `writer` is null or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_loader_weight_store_error(
    writer: *const PieLoaderWeightWriter,
) -> PieLoaderBytes {
    let empty = PieLoaderBytes::default();
    if writer.is_null() {
        return empty;
    }
    let owner = unsafe { (*writer).owner };
    let Some(arena) = (unsafe { owner.cast::<WriterArena>().as_ref() }) else {
        return empty;
    };
    match &arena.error {
        Some(message) => PieLoaderBytes {
            ptr: message.as_ptr(),
            len: message.len(),
        },
        None => empty,
    }
}

// =======================================================================
// Entry points — reading
// =======================================================================

/// Open the artifact at `path`, refusing one written for another key.
///
/// A key mismatch, a schema the loader does not know, and a file that is not an
/// artifact at all are all the same answer to the caller — recompute — so they
/// share a status and differ only in the diagnostic.
///
/// # Safety
///
/// `path` and `expected_key` are valid [`PieLoaderBytes`] live for the call.
/// `out` is a writable slot. `out_diags` is null or a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_loader_weight_store_open(
    path: PieLoaderBytes,
    expected_key: PieLoaderBytes,
    out: *mut *mut PieLoaderWeightStore,
    out_diags: *mut *mut PieLoaderDiagnostics,
) -> PieLoaderStatus {
    unsafe { super::entry::emit_error(out_diags, None) };
    if out.is_null() {
        return PieLoaderStatus::InvalidRequest;
    }
    unsafe { *out = std::ptr::null_mut() };

    let opened = (|| {
        let path = unsafe { super::entry::as_str(&path, "path") }
            .map_err(|err| (PieLoaderStatus::InvalidRequest, err))?;
        let key = unsafe { super::entry::as_str(&expected_key, "expected_key") }
            .map_err(|err| (PieLoaderStatus::InvalidRequest, err))?;
        let artifact = Artifact::open(Path::new(path), key)
            .map_err(|err| (PieLoaderStatus::InvalidCheckpoint, err.to_string()))?;
        build(artifact).map_err(|err| (PieLoaderStatus::InvalidCheckpoint, err))
    })();

    match opened {
        Ok(handle) => {
            unsafe { *out = handle };
            PieLoaderStatus::Ok
        }
        Err((status, message)) => {
            unsafe { super::entry::emit_error(out_diags, Some(message)) };
            status
        }
    }
}

/// Whether the tensor at `index` still matches the digest written with it.
///
/// Writes 1 or 0 through `out_ok`. A mismatch is not an error status: the
/// caller's move is to recompute, the same as for an artifact that was not
/// there. The check reads the mapping, so it can run on another thread while
/// the copies to the device are in flight — which is what the driver does.
///
/// # Safety
///
/// `store` is a live handle and `out_ok` is a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_loader_weight_store_verify(
    store: *const PieLoaderWeightStore,
    index: usize,
    out_ok: *mut bool,
) -> PieLoaderStatus {
    if store.is_null() || out_ok.is_null() {
        return PieLoaderStatus::InvalidRequest;
    }
    unsafe { *out_ok = false };
    let owner = unsafe { (*store).owner };
    let Some(arena) = (unsafe { owner.cast::<StoreArena>().as_ref() }) else {
        return PieLoaderStatus::InvalidRequest;
    };
    match arena.artifact.verify(index) {
        Ok(ok) => {
            unsafe { *out_ok = ok };
            PieLoaderStatus::Ok
        }
        Err(_) => PieLoaderStatus::InvalidCheckpoint,
    }
}

/// Free an artifact handle and unmap the file.
///
/// # Safety
///
/// `store` is null, or a handle from [`pie_loader_weight_store_open`] that has
/// not already been closed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_loader_weight_store_close(store: *mut PieLoaderWeightStore) {
    if store.is_null() {
        return;
    }
    let boxed = unsafe { Box::from_raw(store) };
    if !boxed.owner.is_null() {
        drop(unsafe { Box::from_raw(boxed.owner.cast::<StoreArena>()) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pie_loader_release_diagnostics;

    fn bytes(s: &str) -> PieLoaderBytes {
        PieLoaderBytes {
            ptr: s.as_ptr(),
            len: s.len(),
        }
    }

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("zt_wsffi_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The whole ABI, in the order a driver calls it: stream two payloads in
    /// uneven chunks, record a window and a quantization entry, publish, and
    /// read it all back through the mapping.
    #[test]
    fn the_abi_round_trips_a_store() {
        let dir = tmpdir("round");
        let path = dir.join("weights.zt").display().to_string();
        let key = "k-1";
        let payload = vec![0x5au8; 512];

        let mut writer: *mut PieLoaderWeightWriter = std::ptr::null_mut();
        let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
        unsafe {
            assert_eq!(
                pie_loader_weight_store_create(bytes(&path), bytes(key), &mut writer, &mut diags),
                PieLoaderStatus::Ok
            );
            assert!(!writer.is_null());

            let shape = [512i64];
            let decl = PieLoaderWeightTensor {
                name: bytes("arena"),
                dtype: PieLoaderDType::U8,
                runtime_dtype: bytes("mxfp4-packed"),
                shape: PieLoaderI64Slice {
                    ptr: shape.as_ptr(),
                    len: shape.len(),
                },
            };
            assert_eq!(
                pie_loader_weight_store_begin_tensor(writer, &decl),
                PieLoaderStatus::Ok
            );
            for chunk in payload.chunks(97) {
                assert_eq!(
                    pie_loader_weight_store_write(writer, chunk.as_ptr(), chunk.len()),
                    PieLoaderStatus::Ok
                );
            }
            assert_eq!(
                pie_loader_weight_store_end_tensor(writer),
                PieLoaderStatus::Ok
            );

            let window_shape = [128i64];
            let view = PieLoaderWeightView {
                name: bytes("expert.0"),
                root: bytes("arena"),
                byte_offset: 256,
                dtype: PieLoaderDType::U8,
                runtime_dtype: bytes("mxfp4-packed"),
                shape: PieLoaderI64Slice {
                    ptr: window_shape.as_ptr(),
                    len: window_shape.len(),
                },
                backing: bytes("arena"),
            };
            assert_eq!(
                pie_loader_weight_store_add_view(writer, &view),
                PieLoaderStatus::Ok
            );

            let quant = PieLoaderWeightQuant {
                name: bytes("expert.0"),
                kind: bytes("per_group"),
                scale: bytes("expert.0.scale"),
                zero_point: PieLoaderBytes::default(),
                group_size: 32,
                channel_axis: -1,
            };
            assert_eq!(
                pie_loader_weight_store_add_quant(writer, &quant),
                PieLoaderStatus::Ok
            );
            assert_eq!(
                pie_loader_weight_store_publish(writer, &mut diags),
                PieLoaderStatus::Ok
            );

            let mut store: *mut PieLoaderWeightStore = std::ptr::null_mut();
            assert_eq!(
                pie_loader_weight_store_open(bytes(&path), bytes(key), &mut store, &mut diags),
                PieLoaderStatus::Ok
            );
            let opened = &*store;
            assert_eq!(opened.tensors.len, 1);
            assert_eq!(opened.views.len, 1);
            assert_eq!(opened.quants.len, 1);

            let tensor = &*opened.tensors.ptr;
            assert_eq!(tensor.nbytes, 512);
            assert_eq!(tensor.file_offset % 65536, 0);
            assert_eq!(
                std::slice::from_raw_parts(tensor.data, tensor.nbytes as usize),
                &payload[..]
            );

            let view = &*opened.views.ptr;
            assert_eq!(view.byte_offset, 256);
            let quant = &*opened.quants.ptr;
            assert_eq!(quant.group_size, 32);
            assert_eq!(quant.channel_axis, -1);

            let mut ok = false;
            assert_eq!(
                pie_loader_weight_store_verify(store, 0, &mut ok),
                PieLoaderStatus::Ok
            );
            assert!(ok, "the payload did not verify");

            pie_loader_weight_store_close(store);
            pie_loader_release_diagnostics(diags);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A failing streaming call says why. There is nowhere else for the reason
    /// to go — the call is made once per chunk and returns a status only.
    #[test]
    fn a_failed_call_leaves_its_reason_on_the_handle() {
        let dir = tmpdir("error");
        let path = dir.join("weights.zt").display().to_string();

        let mut writer: *mut PieLoaderWeightWriter = std::ptr::null_mut();
        unsafe {
            assert_eq!(
                pie_loader_weight_store_create(
                    bytes(&path),
                    bytes("k"),
                    &mut writer,
                    std::ptr::null_mut()
                ),
                PieLoaderStatus::Ok
            );
            let shape = [8i64];
            let decl = PieLoaderWeightTensor {
                name: bytes("t"),
                dtype: PieLoaderDType::U8,
                runtime_dtype: bytes("u8"),
                shape: PieLoaderI64Slice {
                    ptr: shape.as_ptr(),
                    len: shape.len(),
                },
            };
            assert_eq!(
                pie_loader_weight_store_begin_tensor(writer, &decl),
                PieLoaderStatus::Ok
            );
            let too_much = [0u8; 16];
            assert_ne!(
                pie_loader_weight_store_write(writer, too_much.as_ptr(), too_much.len()),
                PieLoaderStatus::Ok
            );
            let message = pie_loader_weight_store_error(writer);
            let message =
                std::str::from_utf8(std::slice::from_raw_parts(message.ptr, message.len)).unwrap();
            assert!(
                message.contains("declared as 8"),
                "unhelpful message: {message}"
            );

            pie_loader_weight_store_discard(writer);
        }
        assert!(
            !std::path::Path::new(&path).exists(),
            "a discarded artifact was published"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
