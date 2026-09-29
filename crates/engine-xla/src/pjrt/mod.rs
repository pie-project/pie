//! A thin, owned wrapper over the PJRT C API.
//!
//! The plugin (`libtpu.so`, or any PJRT plugin) is dlopened at runtime, so the
//! crate builds and links on a box without one. Every C call goes through
//! [`Api::call`], which checks the function pointer is inside the plugin's
//! `struct_size` before calling it and turns a returned `PJRT_Error` into an
//! owned [`Error`]. Handles own their C object and destroy it on drop.

#![allow(unsafe_code)]

pub mod options;
pub mod profiler;
#[allow(clippy::all, unsafe_op_in_unsafe_fn, unnecessary_transmutes)]
pub mod sys;

use std::ffi::c_void;
use std::fmt;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Arc;

/// A failed PJRT call: the plugin's error code and message, or a refusal
/// raised on this side of the ABI (missing symbol, short vtable).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub code: u32,
    pub message: String,
}

impl Error {
    fn local(message: impl Into<String>) -> Self {
        Self {
            code: sys::PJRT_Error_Code_FAILED_PRECONDITION,
            message: message.into(),
        }
    }

    #[must_use]
    pub fn exhausted(&self) -> bool {
        self.code == sys::PJRT_Error_Code_RESOURCE_EXHAUSTED
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pjrt: {} (code {})", self.message, self.code)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Element types a buffer can hold, spelled the way PJRT spells them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementType {
    Pred,
    S8,
    S16,
    S32,
    S64,
    U8,
    U16,
    U32,
    U64,
    F16,
    F32,
    F64,
    Bf16,
    F8E5m2,
    F8E4m3fn,
    F8E8m0fnu,
    F4E2m1fn,
    S4,
    U4,
}

impl ElementType {
    #[must_use]
    pub const fn raw(self) -> sys::PJRT_Buffer_Type {
        match self {
            Self::Pred => sys::PJRT_Buffer_Type_PRED,
            Self::S8 => sys::PJRT_Buffer_Type_S8,
            Self::S16 => sys::PJRT_Buffer_Type_S16,
            Self::S32 => sys::PJRT_Buffer_Type_S32,
            Self::S64 => sys::PJRT_Buffer_Type_S64,
            Self::U8 => sys::PJRT_Buffer_Type_U8,
            Self::U16 => sys::PJRT_Buffer_Type_U16,
            Self::U32 => sys::PJRT_Buffer_Type_U32,
            Self::U64 => sys::PJRT_Buffer_Type_U64,
            Self::F16 => sys::PJRT_Buffer_Type_F16,
            Self::F32 => sys::PJRT_Buffer_Type_F32,
            Self::F64 => sys::PJRT_Buffer_Type_F64,
            Self::Bf16 => sys::PJRT_Buffer_Type_BF16,
            Self::F8E5m2 => sys::PJRT_Buffer_Type_F8E5M2,
            Self::F8E4m3fn => sys::PJRT_Buffer_Type_F8E4M3FN,
            Self::F8E8m0fnu => sys::PJRT_Buffer_Type_F8E8M0FNU,
            Self::F4E2m1fn => sys::PJRT_Buffer_Type_F4E2M1FN,
            Self::S4 => sys::PJRT_Buffer_Type_S4,
            Self::U4 => sys::PJRT_Buffer_Type_U4,
        }
    }

    #[must_use]
    pub const fn from_raw(raw: sys::PJRT_Buffer_Type) -> Option<Self> {
        Some(match raw {
            sys::PJRT_Buffer_Type_PRED => Self::Pred,
            sys::PJRT_Buffer_Type_S8 => Self::S8,
            sys::PJRT_Buffer_Type_S16 => Self::S16,
            sys::PJRT_Buffer_Type_S32 => Self::S32,
            sys::PJRT_Buffer_Type_S64 => Self::S64,
            sys::PJRT_Buffer_Type_U8 => Self::U8,
            sys::PJRT_Buffer_Type_U16 => Self::U16,
            sys::PJRT_Buffer_Type_U32 => Self::U32,
            sys::PJRT_Buffer_Type_U64 => Self::U64,
            sys::PJRT_Buffer_Type_F16 => Self::F16,
            sys::PJRT_Buffer_Type_F32 => Self::F32,
            sys::PJRT_Buffer_Type_F64 => Self::F64,
            sys::PJRT_Buffer_Type_BF16 => Self::Bf16,
            sys::PJRT_Buffer_Type_F8E5M2 => Self::F8E5m2,
            sys::PJRT_Buffer_Type_F8E4M3FN => Self::F8E4m3fn,
            sys::PJRT_Buffer_Type_F8E8M0FNU => Self::F8E8m0fnu,
            sys::PJRT_Buffer_Type_F4E2M1FN => Self::F4E2m1fn,
            sys::PJRT_Buffer_Type_S4 => Self::S4,
            sys::PJRT_Buffer_Type_U4 => Self::U4,
            _ => return None,
        })
    }

    /// Bits per element; sub-byte types pack little-end first.
    #[must_use]
    pub const fn bits(self) -> usize {
        match self {
            Self::Pred | Self::S8 | Self::U8 | Self::F8E5m2 | Self::F8E4m3fn | Self::F8E8m0fnu => 8,
            Self::S16 | Self::U16 | Self::F16 | Self::Bf16 => 16,
            Self::S32 | Self::U32 | Self::F32 => 32,
            Self::S64 | Self::U64 | Self::F64 => 64,
            Self::F4E2m1fn | Self::S4 | Self::U4 => 4,
        }
    }

    /// Host bytes for `elements` of this type, as PJRT lays a host buffer
    /// out: sub-byte types take one byte per element (unpacked), everything
    /// else is dense.
    #[must_use]
    pub const fn bytes(self, elements: usize) -> usize {
        if self.bits() < 8 {
            elements
        } else {
            elements * self.bits() / 8
        }
    }
}

/// The loaded plugin: its library handle (kept alive for the vtable) and the
/// `PJRT_Api` it returned.
pub struct Api {
    raw: *const sys::PJRT_Api,
    path: PathBuf,
    /// Never closed: a PJRT plugin keeps threads of its own running past
    /// its last client, and unmapping it under them faults at exit.
    _lib: std::mem::ManuallyDrop<libloading::Library>,
}

// SAFETY: the PJRT C API is thread-safe by contract; the vtable is immutable.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

impl fmt::Debug for Api {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (major, minor) = self.version();
        f.debug_struct("Api")
            .field("path", &self.path)
            .field("version", &format_args!("{major}.{minor}"))
            .finish()
    }
}

/// Calls the vtable entry `$f` with `&mut $args`, after checking the plugin's
/// vtable reaches it.
macro_rules! call {
    ($api:expr, $f:ident, $args:expr) => {{
        let api: &Api = $api;
        let entry = api.entry(
            std::mem::offset_of!(sys::PJRT_Api, $f),
            stringify!($f),
        );
        match entry {
            Err(e) => Err(e),
            Ok(()) => match unsafe { (*api.raw).$f } {
                None => Err(Error::local(concat!(stringify!($f), " is absent from the plugin"))),
                // SAFETY: the args struct is fully initialized with its
                // struct_size by the caller, and outlives the call.
                Some(f) => api.check(unsafe { f($args) }),
            },
        }
    }};
}

/// Zeroed args with `struct_size` set: every `PJRT_*_Args` is a plain C struct
/// whose all-zero bit pattern is the "unset" value the ABI expects.
macro_rules! args {
    ($ty:ident { $($field:ident : $value:expr),* $(,)? }) => {{
        // SAFETY: PJRT arg structs are POD; zero is null/0/false for every field.
        let mut a: sys::$ty = unsafe { std::mem::zeroed() };
        a.struct_size = std::mem::size_of::<sys::$ty>();
        $(a.$field = $value;)*
        a
    }};
}

impl Api {
    /// dlopens `path` and asks it for its `PJRT_Api`, then runs
    /// `PJRT_Plugin_Initialize`.
    pub fn load(path: &Path) -> Result<Arc<Self>> {
        // SAFETY: loading a PJRT plugin runs its static initializers; that is
        // the documented way to use one.
        let lib = unsafe { libloading::Library::new(path) }
            .map_err(|e| Error::local(format!("cannot load {}: {e}", path.display())))?;
        // SAFETY: `GetPjrtApi` is the one exported entry every PJRT plugin has.
        let get: libloading::Symbol<'_, unsafe extern "C" fn() -> *const sys::PJRT_Api> =
            unsafe { lib.get(b"GetPjrtApi\0") }
                .map_err(|e| Error::local(format!("{} is not a PJRT plugin: {e}", path.display())))?;
        let raw = unsafe { get() };
        if raw.is_null() {
            return Err(Error::local(format!("{} returned no PJRT_Api", path.display())));
        }
        let api = Self {
            raw,
            path: path.to_path_buf(),
            _lib: std::mem::ManuallyDrop::new(lib),
        };
        let (major, _) = api.version();
        if major != sys::PJRT_API_MAJOR as i32 {
            return Err(Error::local(format!(
                "{} serves PJRT {major}.x; this engine speaks {}.x",
                path.display(),
                sys::PJRT_API_MAJOR
            )));
        }
        let mut init = args!(PJRT_Plugin_Initialize_Args {});
        call!(&api, PJRT_Plugin_Initialize, &mut init)?;
        Ok(Arc::new(api))
    }

    /// Finds a plugin: `explicit`, else `PIE_XLA_PLUGIN`, else
    /// `TPU_LIBRARY_PATH`, else `libtpu.so` on the loader's search path.
    pub fn discover(explicit: Option<&Path>) -> Result<Arc<Self>> {
        if let Some(path) = explicit {
            return Self::load(path);
        }
        for var in ["PIE_XLA_PLUGIN", "TPU_LIBRARY_PATH"] {
            if let Some(path) = std::env::var_os(var).filter(|v| !v.is_empty()) {
                return Self::load(Path::new(&path));
            }
        }
        Self::load(Path::new("libtpu.so"))
    }

    #[must_use]
    pub fn version(&self) -> (i32, i32) {
        // SAFETY: `pjrt_api_version` is at a fixed offset in every version.
        let v = unsafe { (*self.raw).pjrt_api_version };
        (v.major_version, v.minor_version)
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn entry(&self, offset: usize, name: &str) -> Result<()> {
        // SAFETY: struct_size is the first field of every PJRT_Api.
        let size = unsafe { (*self.raw).struct_size };
        if offset + std::mem::size_of::<usize>() > size {
            return Err(Error::local(format!(
                "{name} is past the plugin's {size}-byte vtable"
            )));
        }
        Ok(())
    }

    fn check(&self, err: *mut sys::PJRT_Error) -> Result<()> {
        if err.is_null() {
            return Ok(());
        }
        let mut message = args!(PJRT_Error_Message_Args { error: err });
        let mut code = args!(PJRT_Error_GetCode_Args { error: err });
        // SAFETY: both entries predate every version we accept; `err` is live.
        let text = unsafe {
            if let Some(f) = (*self.raw).PJRT_Error_Message {
                f(&mut message);
            }
            let code_err = (*self.raw).PJRT_Error_GetCode.map(|f| f(&mut code));
            if let Some(e) = code_err.filter(|e| !e.is_null()) {
                self.destroy_error(e);
            }
            if message.message.is_null() {
                String::from("(no message)")
            } else {
                String::from_utf8_lossy(std::slice::from_raw_parts(
                    message.message.cast::<u8>(),
                    message.message_size,
                ))
                .into_owned()
            }
        };
        self.destroy_error(err);
        Err(Error {
            code: code.code,
            message: text,
        })
    }

    fn destroy_error(&self, err: *mut sys::PJRT_Error) {
        let mut d = args!(PJRT_Error_Destroy_Args { error: err });
        // SAFETY: `err` came from this plugin and is destroyed exactly once.
        unsafe {
            if let Some(f) = (*self.raw).PJRT_Error_Destroy {
                f(&mut d);
            }
        }
    }
}

/// A device the client addresses. Borrowed from the client; valid while it is.
#[derive(Clone, Copy)]
pub struct Device {
    raw: *mut sys::PJRT_Device,
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

/// What the device reports about its memory.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemoryStats {
    pub bytes_in_use: i64,
    pub bytes_limit: Option<i64>,
}

pub struct Client {
    api: Arc<Api>,
    raw: *mut sys::PJRT_Client,
    devices: Vec<Device>,
}

unsafe impl Send for Client {}
unsafe impl Sync for Client {}

impl fmt::Debug for Client {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Client")
            .field("platform", &self.platform_name().unwrap_or_default())
            .field("devices", &self.devices.len())
            .finish()
    }
}

impl Client {
    pub fn create(api: Arc<Api>) -> Result<Self> {
        let mut a = args!(PJRT_Client_Create_Args {});
        call!(&api, PJRT_Client_Create, &mut a)?;
        let mut client = Self {
            api,
            raw: a.client,
            devices: Vec::new(),
        };
        let mut d = args!(PJRT_Client_AddressableDevices_Args { client: client.raw });
        call!(&client.api, PJRT_Client_AddressableDevices, &mut d)?;
        // SAFETY: the plugin owns this array for the client's lifetime.
        let devices = unsafe {
            std::slice::from_raw_parts(d.addressable_devices, d.num_addressable_devices)
        };
        client.devices = devices.iter().map(|&raw| Device { raw }).collect();
        Ok(client)
    }

    #[must_use]
    pub fn api(&self) -> &Arc<Api> {
        &self.api
    }

    #[must_use]
    pub fn devices(&self) -> &[Device] {
        &self.devices
    }

    pub fn platform_name(&self) -> Result<String> {
        let mut a = args!(PJRT_Client_PlatformName_Args { client: self.raw });
        call!(&self.api, PJRT_Client_PlatformName, &mut a)?;
        Ok(unsafe { text(a.platform_name, a.platform_name_size) })
    }

    pub fn device_kind(&self, device: Device) -> Result<String> {
        let mut d = args!(PJRT_Device_GetDescription_Args { device: device.raw });
        call!(&self.api, PJRT_Device_GetDescription, &mut d)?;
        let mut k = args!(PJRT_DeviceDescription_Kind_Args {
            device_description: d.device_description
        });
        call!(&self.api, PJRT_DeviceDescription_Kind, &mut k)?;
        Ok(unsafe { text(k.device_kind, k.device_kind_size) })
    }

    pub fn memory_stats(&self, device: Device) -> Result<MemoryStats> {
        let mut a = args!(PJRT_Device_MemoryStats_Args { device: device.raw });
        call!(&self.api, PJRT_Device_MemoryStats, &mut a)?;
        Ok(MemoryStats {
            bytes_in_use: a.bytes_in_use,
            bytes_limit: a.bytes_limit_is_set.then_some(a.bytes_limit),
        })
    }

    /// Compiles a StableHLO (MLIR text or bytecode) module for one device.
    pub fn compile(&self, mlir: &str) -> Result<Executable> {
        let options = options::single_device();
        let format = "mlir";
        let program = args!(PJRT_Program {
            code: mlir.as_ptr().cast_mut().cast(),
            code_size: mlir.len(),
            format: format.as_ptr().cast(),
            format_size: format.len(),
        });
        let mut a = args!(PJRT_Client_Compile_Args {
            client: self.raw,
            program: &program,
            compile_options: options.as_ptr().cast(),
            compile_options_size: options.len(),
        });
        call!(&self.api, PJRT_Client_Compile, &mut a)?;
        Executable::adopt(self.api.clone(), a.executable)
    }

    /// Loads an executable `Executable::serialize` wrote (same plugin build).
    pub fn deserialize(&self, bytes: &[u8]) -> Result<Executable> {
        let mut a = args!(PJRT_Executable_DeserializeAndLoad_Args {
            client: self.raw,
            serialized_executable: bytes.as_ptr().cast(),
            serialized_executable_size: bytes.len(),
        });
        call!(&self.api, PJRT_Executable_DeserializeAndLoad, &mut a)?;
        Executable::adopt(self.api.clone(), a.loaded_executable)
    }

    /// Copies `data` (dense, row-major) onto `device` as an array of `dims`.
    /// The host bytes may be reused once this returns.
    pub fn upload(
        &self,
        device: Device,
        data: &[u8],
        ty: ElementType,
        dims: &[i64],
    ) -> Result<Buffer> {
        let elements: i64 = dims.iter().product();
        let want = ty.bytes(usize::try_from(elements).unwrap_or(usize::MAX));
        if data.len() != want {
            return Err(Error::local(format!(
                "upload of {dims:?} {ty:?} wants {want} bytes, was handed {}",
                data.len()
            )));
        }
        let mut a = args!(PJRT_Client_BufferFromHostBuffer_Args {
            client: self.raw,
            data: data.as_ptr().cast(),
            type_: ty.raw(),
            dims: dims.as_ptr(),
            num_dims: dims.len(),
            host_buffer_semantics: sys::PJRT_HostBufferSemantics_kImmutableOnlyDuringCall,
            device: device.raw,
        });
        call!(&self.api, PJRT_Client_BufferFromHostBuffer, &mut a)?;
        if !a.done_with_host_buffer.is_null() {
            Event::adopt(self.api.clone(), a.done_with_host_buffer).wait()?;
        }
        Ok(Buffer {
            api: self.api.clone(),
            raw: a.buffer,
        })
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let mut a = args!(PJRT_Client_Destroy_Args { client: self.raw });
        let _ = call!(&self.api, PJRT_Client_Destroy, &mut a);
    }
}

pub struct Buffer {
    api: Arc<Api>,
    raw: *mut sys::PJRT_Buffer,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl fmt::Debug for Buffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Buffer")
            .field("ty", &self.element_type().ok())
            .field("dims", &self.dims().ok())
            .finish()
    }
}

impl Buffer {
    pub fn dims(&self) -> Result<Vec<i64>> {
        let mut a = args!(PJRT_Buffer_Dimensions_Args { buffer: self.raw });
        call!(&self.api, PJRT_Buffer_Dimensions, &mut a)?;
        Ok(unsafe { slice(a.dims, a.num_dims) }.to_vec())
    }

    pub fn element_type(&self) -> Result<ElementType> {
        let mut a = args!(PJRT_Buffer_ElementType_Args { buffer: self.raw });
        call!(&self.api, PJRT_Buffer_ElementType, &mut a)?;
        ElementType::from_raw(a.type_)
            .ok_or_else(|| Error::local(format!("buffer holds PJRT type {}", a.type_)))
    }

    /// Dense host bytes this buffer copies out to.
    pub fn host_bytes(&self) -> Result<usize> {
        let order = self.row_major()?;
        let mut layout = row_major_layout(&order);
        let mut a = args!(PJRT_Buffer_ToHostBuffer_Args {
            src: self.raw,
            host_layout: &mut layout,
        });
        call!(&self.api, PJRT_Buffer_ToHostBuffer, &mut a)?;
        Ok(a.dst_size)
    }

    /// `minor_to_major` for the dense row-major host form of this buffer:
    /// the device may keep another layout (TPU picks per shape), so every
    /// copy out names this one.
    fn row_major(&self) -> Result<Vec<i64>> {
        let rank = self.dims()?.len() as i64;
        Ok((0..rank).rev().collect())
    }

    /// Starts a copy into `dst`; the returned event fires once it has landed.
    ///
    /// # Safety
    /// `dst` must stay valid and unaliased until the event completes.
    pub unsafe fn download_into(&self, dst: &mut [u8]) -> Result<Event> {
        let order = self.row_major()?;
        let mut layout = row_major_layout(&order);
        let mut a = args!(PJRT_Buffer_ToHostBuffer_Args {
            src: self.raw,
            host_layout: &mut layout,
            dst: dst.as_mut_ptr().cast(),
            dst_size: dst.len(),
        });
        call!(&self.api, PJRT_Buffer_ToHostBuffer, &mut a)?;
        Ok(Event::adopt(self.api.clone(), a.event))
    }

    /// Copies the whole buffer to the host and waits for it.
    pub fn download(&self) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.host_bytes()?];
        // SAFETY: `out` outlives the wait below.
        unsafe { self.download_into(&mut out) }?.wait()?;
        Ok(out)
    }

    /// An event that fires once the buffer's defining computation is done.
    pub fn ready(&self) -> Result<Event> {
        let mut a = args!(PJRT_Buffer_ReadyEvent_Args { buffer: self.raw });
        call!(&self.api, PJRT_Buffer_ReadyEvent, &mut a)?;
        Ok(Event::adopt(self.api.clone(), a.event))
    }

    #[must_use]
    pub fn raw(&self) -> *mut sys::PJRT_Buffer {
        self.raw
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let mut a = args!(PJRT_Buffer_Destroy_Args { buffer: self.raw });
        let _ = call!(&self.api, PJRT_Buffer_Destroy, &mut a);
    }
}

pub struct Executable {
    api: Arc<Api>,
    raw: *mut sys::PJRT_LoadedExecutable,
    outputs: usize,
}

unsafe impl Send for Executable {}
unsafe impl Sync for Executable {}

impl fmt::Debug for Executable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Executable")
            .field("outputs", &self.outputs)
            .finish()
    }
}

/// How an argument is handed to [`Executable::execute`].
pub enum Arg<'a> {
    /// Read only; the caller keeps the buffer.
    Keep(&'a Buffer),
    /// Donated: the executable may reuse its memory for an aliased output.
    /// The buffer is consumed; reading it afterwards is an error in PJRT.
    Donate(Buffer),
}

impl Executable {
    fn adopt(api: Arc<Api>, raw: *mut sys::PJRT_LoadedExecutable) -> Result<Self> {
        let mut g = args!(PJRT_LoadedExecutable_GetExecutable_Args {
            loaded_executable: raw
        });
        let mut exe = Self {
            api,
            raw,
            outputs: 0,
        };
        call!(&exe.api, PJRT_LoadedExecutable_GetExecutable, &mut g)?;
        let mut n = args!(PJRT_Executable_NumOutputs_Args {
            executable: g.executable
        });
        let counted = call!(&exe.api, PJRT_Executable_NumOutputs, &mut n);
        let mut d = args!(PJRT_Executable_Destroy_Args {
            executable: g.executable
        });
        let _ = call!(&exe.api, PJRT_Executable_Destroy, &mut d);
        counted?;
        exe.outputs = n.num_outputs;
        Ok(exe)
    }

    #[must_use]
    pub fn outputs(&self) -> usize {
        self.outputs
    }

    /// The executable's serialized form, for `Client::deserialize`.
    pub fn serialize(&self) -> Result<Vec<u8>> {
        let mut g = args!(PJRT_LoadedExecutable_GetExecutable_Args {
            loaded_executable: self.raw
        });
        call!(&self.api, PJRT_LoadedExecutable_GetExecutable, &mut g)?;
        let mut a = args!(PJRT_Executable_Serialize_Args {
            executable: g.executable,
        });
        let done = call!(&self.api, PJRT_Executable_Serialize, &mut a);
        let bytes = done.map(|()| {
            // SAFETY: the plugin owns these bytes until the deleter runs.
            let out = unsafe { slice(a.serialized_bytes.cast::<u8>(), a.serialized_bytes_size) }
                .to_vec();
            if let Some(deleter) = a.serialized_executable_deleter {
                unsafe { deleter(a.serialized_executable) };
            }
            out
        });
        let mut d = args!(PJRT_Executable_Destroy_Args {
            executable: g.executable
        });
        let _ = call!(&self.api, PJRT_Executable_Destroy, &mut d);
        bytes
    }

    /// Enqueues one run on `device`. Returns the outputs (ready when their
    /// events fire) and the device-complete event.
    pub fn execute(&self, device: Device, args: Vec<Arg<'_>>) -> Result<(Vec<Buffer>, Event)> {
        let mut raw_args: Vec<*mut sys::PJRT_Buffer> = Vec::with_capacity(args.len());
        let mut kept: Vec<i64> = Vec::new();
        let mut donated: Vec<Buffer> = Vec::new();
        for (i, arg) in args.into_iter().enumerate() {
            match arg {
                Arg::Keep(b) => {
                    raw_args.push(b.raw);
                    kept.push(i as i64);
                }
                Arg::Donate(b) => {
                    raw_args.push(b.raw);
                    donated.push(b);
                }
            }
        }
        let mut options = args!(PJRT_ExecuteOptions {
            non_donatable_input_indices: kept.as_ptr(),
            num_non_donatable_input_indices: kept.len(),
        });
        let list: *const *mut sys::PJRT_Buffer = raw_args.as_ptr();
        let mut outputs: Vec<*mut sys::PJRT_Buffer> = vec![ptr::null_mut(); self.outputs];
        let out_list: *mut *mut sys::PJRT_Buffer = outputs.as_mut_ptr();
        let mut done: *mut sys::PJRT_Event = ptr::null_mut();
        let mut a = args!(PJRT_LoadedExecutable_Execute_Args {
            executable: self.raw,
            options: &mut options,
            argument_lists: &list,
            num_devices: 1,
            num_args: raw_args.len(),
            output_lists: &out_list,
            device_complete_events: &mut done,
            execute_device: device.raw,
        });
        call!(&self.api, PJRT_LoadedExecutable_Execute, &mut a)?;
        // Donated inputs are now owned by the runtime: their handles still
        // need destroying, which releases nothing the outputs alias.
        drop(donated);
        let outputs = outputs
            .into_iter()
            .map(|raw| Buffer {
                api: self.api.clone(),
                raw,
            })
            .collect();
        Ok((outputs, Event::adopt(self.api.clone(), done)))
    }
}

impl Drop for Executable {
    fn drop(&mut self) {
        let mut a = args!(PJRT_LoadedExecutable_Destroy_Args {
            executable: self.raw
        });
        let _ = call!(&self.api, PJRT_LoadedExecutable_Destroy, &mut a);
    }
}

pub struct Event {
    api: Arc<Api>,
    raw: *mut sys::PJRT_Event,
}

unsafe impl Send for Event {}
unsafe impl Sync for Event {}

impl Event {
    fn adopt(api: Arc<Api>, raw: *mut sys::PJRT_Event) -> Self {
        Self { api, raw }
    }

    /// Blocks until the event fires; its error, if any, is returned.
    pub fn wait(self) -> Result<()> {
        if self.raw.is_null() {
            return Ok(());
        }
        let mut a = args!(PJRT_Event_Await_Args { event: self.raw });
        call!(&self.api, PJRT_Event_Await, &mut a)
    }

    /// Runs `f` with the event's outcome once it fires (maybe inline, maybe
    /// on a plugin thread).
    pub fn on_ready(self, f: impl FnOnce(Result<()>) + Send + 'static) -> Result<()> {
        if self.raw.is_null() {
            f(Ok(()));
            return Ok(());
        }
        struct Pending {
            api: Arc<Api>,
            f: Box<dyn FnOnce(Result<()>) + Send>,
        }
        unsafe extern "C" fn trampoline(err: *mut sys::PJRT_Error, user: *mut c_void) {
            // SAFETY: `user` is the Box leaked below, reclaimed exactly once.
            let pending = unsafe { Box::from_raw(user.cast::<Pending>()) };
            let outcome = pending.api.check(err);
            (pending.f)(outcome);
        }
        let user = Box::into_raw(Box::new(Pending {
            api: self.api.clone(),
            f: Box::new(f),
        }));
        let mut a = args!(PJRT_Event_OnReady_Args {
            event: self.raw,
            callback: Some(trampoline),
            user_arg: user.cast(),
        });
        let registered = call!(&self.api, PJRT_Event_OnReady, &mut a);
        if registered.is_err() {
            // SAFETY: the plugin refused the callback, so it never ran.
            drop(unsafe { Box::from_raw(user) });
        }
        registered
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }
        let mut a = args!(PJRT_Event_Destroy_Args { event: self.raw });
        let _ = call!(&self.api, PJRT_Event_Destroy, &mut a);
    }
}

/// A tiled layout with no tiles over `minor_to_major`: plain dense order.
/// Borrows `order`; the caller keeps it alive across the call.
fn row_major_layout(order: &[i64]) -> sys::PJRT_Buffer_MemoryLayout {
    let tiled = args!(PJRT_Buffer_MemoryLayout_Tiled {
        minor_to_major: order.as_ptr(),
        minor_to_major_size: order.len(),
    });
    let mut layout = args!(PJRT_Buffer_MemoryLayout {
        type_: sys::PJRT_Buffer_MemoryLayout_Type_Tiled,
    });
    layout.__bindgen_anon_1.tiled = tiled;
    layout
}

unsafe fn text(p: *const std::os::raw::c_char, n: usize) -> String {
    if p.is_null() {
        return String::new();
    }
    String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(p.cast::<u8>(), n) }).into_owned()
}

unsafe fn slice<'a, T>(p: *const T, n: usize) -> &'a [T] {
    if p.is_null() || n == 0 {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(p, n) }
}
