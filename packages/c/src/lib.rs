//! pie's C library (include/pie.h): a `worker::Server` in the caller's
//! process, on this build's engine.
// Each call's safety contract is its entry in include/pie.h.
#![allow(clippy::missing_safety_doc, non_camel_case_types)]

use std::ffi::{CStr, CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use worker::Server;
use worker::embedded::Settings;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum pie_status {
    PIE_OK = 0,
    PIE_ERR_INVALID_ARGUMENT = 1,
    PIE_ERR_SHUT_DOWN = 2,
    PIE_ERR_FAILED = 3,
    PIE_ERR_PANIC = 4,
}

pub struct pie_server {
    server: Server,
    summary: CString,
    listen_addr: Option<CString>,
}

pub type pie_frame_fn = extern "C" fn(ctx: *mut c_void, frame: *const u8, len: usize);

/// Why a call failed: the status it returns and the message for `*error`.
struct Failure(pie_status, String);

impl Failure {
    fn invalid(message: impl Into<String>) -> Self {
        Failure(pie_status::PIE_ERR_INVALID_ARGUMENT, message.into())
    }

    /// A call on `server` that failed: shut down if it no longer runs.
    fn of(server: &Server, what: &str, error: anyhow::Error) -> Self {
        if server.running() {
            Failure(pie_status::PIE_ERR_FAILED, format!("{what}: {error:#}"))
        } else {
            Failure(
                pie_status::PIE_ERR_SHUT_DOWN,
                "the server is shut down".into(),
            )
        }
    }
}

/// Runs `f` behind the C boundary: a failure or a panic becomes its status,
/// with its message in `*error` when the caller asked for one.
unsafe fn ffi(error: *mut *mut c_char, f: impl FnOnce() -> Result<(), Failure>) -> pie_status {
    let Failure(status, message) = match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => return pie_status::PIE_OK,
        Ok(Err(failure)) => failure,
        Err(panic) => Failure(
            pie_status::PIE_ERR_PANIC,
            panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "pie panicked".into()),
        ),
    };
    if !error.is_null() {
        unsafe { *error = to_c(message) };
    }
    status
}

fn to_c(text: String) -> *mut c_char {
    CString::new(text.replace('\0', "")).unwrap().into_raw()
}

unsafe fn str_arg<'a>(ptr: *const c_char, what: &str) -> Result<&'a str, Failure> {
    if ptr.is_null() {
        return Err(Failure::invalid(format!("{what} is NULL")));
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map_err(|e| Failure::invalid(format!("{what}: {e}")))
}

unsafe fn opt_str_arg<'a>(ptr: *const c_char, what: &str) -> Result<Option<&'a str>, Failure> {
    if ptr.is_null() {
        Ok(None)
    } else {
        unsafe { str_arg(ptr, what) }.map(Some)
    }
}

unsafe fn bytes_arg<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
    if ptr.is_null() || len == 0 {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

unsafe fn out_arg<'a, T>(ptr: *mut T, what: &str) -> Result<&'a mut T, Failure> {
    unsafe { ptr.as_mut() }.ok_or_else(|| Failure::invalid(format!("{what} is NULL")))
}

unsafe fn server<'a>(ptr: *const pie_server) -> Result<&'a Server, Failure> {
    unsafe { ptr.as_ref() }
        .map(|server| &server.server)
        .ok_or_else(|| Failure::invalid("server is NULL"))
}

fn boot(
    artifact: &Path,
    settings: Option<&str>,
    home: &Path,
    listen: Option<&str>,
) -> Result<pie_server, Failure> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .try_init();

    let settings = settings
        .map_or_else(|| Ok(Settings::default()), Settings::parse)
        .map_err(|e| Failure::invalid(format!("config: {e:#}")))?;
    let listen = listen
        .map(str::parse)
        .transpose()
        .map_err(|e| Failure::invalid(format!("listen: {e}")))?;
    let failed = |e: anyhow::Error| Failure(pie_status::PIE_ERR_FAILED, format!("{e:#}"));
    let server = Server::embed(artifact, &settings, home, listen).map_err(failed)?;
    let summary = serde_json::to_string(server.summary()).map_err(|e| failed(e.into()))?;
    Ok(pie_server {
        summary: CString::new(summary).map_err(|e| failed(e.into()))?,
        listen_addr: server
            .listen_addr()
            .map(|addr| CString::new(addr.to_string()).unwrap()),
        server,
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn pie_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_server_start(
    artifact: *const c_char,
    config: *const c_char,
    home: *const c_char,
    listen: *const c_char,
    out: *mut *mut pie_server,
    error: *mut *mut c_char,
) -> pie_status {
    unsafe {
        ffi(error, || {
            let out = out_arg(out, "out")?;
            let artifact = Path::new(str_arg(artifact, "artifact")?);
            let config = opt_str_arg(config, "config")?;
            let home = Path::new(str_arg(home, "home")?);
            let listen = opt_str_arg(listen, "listen")?;
            *out = Box::into_raw(Box::new(boot(artifact, config, home, listen)?));
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_server_summary(server: *const pie_server) -> *const c_char {
    match unsafe { server.as_ref() } {
        Some(server) => server.summary.as_ptr(),
        None => std::ptr::null(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_server_listen_addr(server: *const pie_server) -> *const c_char {
    match unsafe { server.as_ref() }.and_then(|server| server.listen_addr.as_ref()) {
        Some(addr) => addr.as_ptr(),
        None => std::ptr::null(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_server_install(
    server: *mut pie_server,
    bytes: *const u8,
    len: usize,
    file: *const c_char,
    version: *const c_char,
    name: *mut *mut c_char,
    error: *mut *mut c_char,
) -> pie_status {
    unsafe {
        ffi(error, || {
            let server = self::server(server)?;
            let name = out_arg(name, "name")?;
            let file = str_arg(file, "file")?;
            let version = opt_str_arg(version, "version")?;
            let installed = server
                .install(bytes_arg(bytes, len).to_vec(), file, version)
                .map_err(|e| Failure::of(server, "install", e))?;
            *name = to_c(installed);
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_server_install_language(
    server: *mut pie_server,
    language: *const c_char,
    bytes: *const u8,
    len: usize,
    error: *mut *mut c_char,
) -> pie_status {
    unsafe {
        ffi(error, || {
            let server = self::server(server)?;
            let language = str_arg(language, "language")?;
            server
                .install_language(language, bytes_arg(bytes, len).to_vec())
                .map_err(|e| Failure::of(server, "install language", e))?;
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_server_open_session(
    server: *mut pie_server,
    session: *mut u32,
    error: *mut *mut c_char,
) -> pie_status {
    unsafe {
        ffi(error, || {
            let server = self::server(server)?;
            let session = out_arg(session, "session")?;
            *session = server
                .open_session()
                .map_err(|e| Failure::of(server, "open session", e))?;
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_server_close_session(server: *mut pie_server, session: u32) {
    if let Ok(server) = unsafe { self::server(server) } {
        let _ = catch_unwind(AssertUnwindSafe(|| server.close_session(session)));
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_server_send_frame(
    server: *mut pie_server,
    session: u32,
    frame: *const u8,
    len: usize,
    error: *mut *mut c_char,
) -> pie_status {
    unsafe {
        ffi(error, || {
            let server = self::server(server)?;
            server
                .send_frame(session, bytes_arg(frame, len))
                .map_err(|e| Failure::of(server, "send frame", e))
        })
    }
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn pie_server_recv_frames(
    server: *mut pie_server,
    session: u32,
    max_wait_ms: u32,
    max_frames: usize,
    on_frame: pie_frame_fn,
    ctx: *mut c_void,
    received: *mut usize,
    error: *mut *mut c_char,
) -> pie_status {
    unsafe {
        ffi(error, || {
            let server = self::server(server)?;
            let received = out_arg(received, "received")?;
            let frames = server
                .recv_frames(session, u64::from(max_wait_ms), max_frames)
                .map_err(|e| Failure::of(server, "receive frames", e))?;
            for frame in &frames {
                on_frame(ctx, frame.as_ptr(), frame.len());
            }
            *received = frames.len();
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_server_shutdown(server: *mut pie_server) {
    if let Ok(server) = unsafe { self::server(server) } {
        let _ = catch_unwind(AssertUnwindSafe(|| server.shutdown()));
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_server_free(server: *mut pie_server) {
    if !server.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(server) })));
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(unsafe { CString::from_raw(s) });
    }
}
