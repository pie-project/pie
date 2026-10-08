//! The C core of the Swift `PieServer` (include/pie_server.h): a
//! `runtime::embed::Server` on the Metal engine.
#![cfg(target_vendor = "apple")]
// Each call's safety contract is its entry in include/pie_server.h.
#![allow(clippy::missing_safety_doc)]

use std::ffi::{CStr, CString, c_char, c_void};
use std::path::{Path, PathBuf};

use runtime::embed::{Host, Server};

pub struct PieServer {
    server: Server,
    summary: CString,
}

pub type FrameSink = extern "C" fn(ctx: *mut c_void, frame: *const u8, len: usize);

type Outcome<T> = Result<T, String>;

/// Runs `f`, storing its error in `*error` (when asked for) and returning
/// `failed` in its place.
unsafe fn ffi<T>(error: *mut *mut c_char, failed: T, f: impl FnOnce() -> Outcome<T>) -> T {
    match f() {
        Ok(value) => value,
        Err(message) => {
            if !error.is_null() {
                unsafe { *error = to_c(message) };
            }
            failed
        }
    }
}

fn to_c(text: String) -> *mut c_char {
    CString::new(text.replace('\0', "")).unwrap().into_raw()
}

fn message(error: anyhow::Error) -> String {
    format!("{error:#}")
}

unsafe fn str_arg<'a>(ptr: *const c_char, what: &str) -> Outcome<&'a str> {
    if ptr.is_null() {
        return Err(format!("{what} is NULL"));
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map_err(|e| format!("{what}: {e}"))
}

unsafe fn opt_str_arg<'a>(ptr: *const c_char, what: &str) -> Outcome<Option<&'a str>> {
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

unsafe fn server<'a>(ptr: *const PieServer) -> Outcome<&'a Server> {
    unsafe { ptr.as_ref() }
        .map(|server| &server.server)
        .ok_or_else(|| "server is NULL".to_string())
}

fn boot(artifact: &Path, config: Option<&str>, home: &Path) -> anyhow::Result<PieServer> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .try_init();

    let host = Host {
        name: "pie-swift".into(),
        home: home.to_path_buf(),
        worker_threads: 2,
    };
    let builtins = builtins::all()
        .iter()
        .map(|b| runtime::bootstrap::BuiltinProgram {
            name: b.name,
            version: b.version,
            component: b.component,
        })
        .collect();
    let server = Server::start(artifact, config, host, builtins)?;
    Ok(PieServer {
        summary: CString::new(serde_json::to_string(server.summary())?)?,
        server,
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_server_start(
    artifact: *const c_char,
    config: *const c_char,
    home: *const c_char,
    error: *mut *mut c_char,
) -> *mut PieServer {
    unsafe {
        ffi(error, std::ptr::null_mut(), || {
            let artifact = PathBuf::from(str_arg(artifact, "artifact")?);
            let config = opt_str_arg(config, "config")?;
            let home = PathBuf::from(str_arg(home, "home")?);
            let server = boot(&artifact, config, &home).map_err(message)?;
            Ok(Box::into_raw(Box::new(server)))
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_server_summary(server: *const PieServer) -> *const c_char {
    match unsafe { server.as_ref() } {
        Some(server) => server.summary.as_ptr(),
        None => std::ptr::null(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_server_install(
    server: *const PieServer,
    bytes: *const u8,
    len: usize,
    file: *const c_char,
    version: *const c_char,
    error: *mut *mut c_char,
) -> *mut c_char {
    unsafe {
        ffi(error, std::ptr::null_mut(), || {
            let file = str_arg(file, "file")?;
            let version = opt_str_arg(version, "version")?;
            let name = self::server(server)?
                .install(bytes_arg(bytes, len).to_vec(), file, version)
                .map_err(|e| format!("install: {e:#}"))?;
            Ok(to_c(name))
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_server_install_language(
    server: *const PieServer,
    language: *const c_char,
    bytes: *const u8,
    len: usize,
    error: *mut *mut c_char,
) -> i32 {
    unsafe {
        ffi(error, 1, || {
            let language = str_arg(language, "language")?;
            self::server(server)?
                .install_language(language, bytes_arg(bytes, len).to_vec())
                .map_err(|e| format!("install language: {e:#}"))?;
            Ok(0)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_server_open_session(
    server: *const PieServer,
    session: *mut u32,
    error: *mut *mut c_char,
) -> i32 {
    unsafe {
        ffi(error, 1, || {
            if session.is_null() {
                return Err("session is NULL".into());
            }
            *session = self::server(server)?
                .open_session()
                .map_err(|e| format!("open session: {e:#}"))?;
            Ok(0)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_server_close_session(server: *const PieServer, session: u32) {
    if let Ok(server) = unsafe { self::server(server) } {
        server.close_session(session);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_server_send_frame(
    server: *const PieServer,
    session: u32,
    frame: *const u8,
    len: usize,
    error: *mut *mut c_char,
) -> i32 {
    unsafe {
        ffi(error, 1, || {
            self::server(server)?
                .send_frame(session, bytes_arg(frame, len))
                .map_err(message)?;
            Ok(0)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_server_recv_frames(
    server: *const PieServer,
    session: u32,
    max_wait_ms: u32,
    max: u32,
    sink: FrameSink,
    ctx: *mut c_void,
    error: *mut *mut c_char,
) -> i32 {
    unsafe {
        ffi(error, -1, || {
            let frames = self::server(server)?
                .recv_frames(session, u64::from(max_wait_ms), max as usize)
                .map_err(message)?;
            for frame in &frames {
                sink(ctx, frame.as_ptr(), frame.len());
            }
            Ok(frames.len() as i32)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_server_shutdown(server: *const PieServer) {
    if let Ok(server) = unsafe { self::server(server) } {
        server.shutdown();
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_server_free(server: *mut PieServer) {
    if !server.is_null() {
        drop(unsafe { Box::from_raw(server) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(unsafe { CString::from_raw(s) });
    }
}
