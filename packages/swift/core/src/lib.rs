//! The C core of the Swift `PieServer` (include/pie_server.h): the runtime
//! booted in-process through `runtime::embed` on the Metal engine.
#![cfg(target_vendor = "apple")]
// Each call's safety contract is its entry in include/pie_server.h.
#![allow(clippy::missing_safety_doc)]

use std::ffi::{CStr, CString, c_char, c_void};
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use runtime::embed::{BootConfig, Embedded};
use runtime::inferlet::program;
use tokio::sync::watch;

pub struct PieServer {
    summary: CString,
    live: RwLock<Option<Live>>,
    stop: watch::Sender<bool>,
}

struct Live {
    embedded: Embedded,
    runtime: tokio::runtime::Runtime,
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

impl PieServer {
    /// Runs `f` on the live runtime; fails once the server is shut down.
    fn with<T>(&self, f: impl FnOnce(&Live) -> Outcome<T>) -> Outcome<T> {
        let live = self.live.read().unwrap();
        f(live.as_ref().ok_or("the server is shut down")?)
    }
}

unsafe fn server<'a>(ptr: *const PieServer) -> Outcome<&'a PieServer> {
    unsafe { ptr.as_ref() }.ok_or_else(|| "server is NULL".to_string())
}

fn boot(artifact: &Path, config: Option<&str>, home: &Path) -> anyhow::Result<PieServer> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .try_init();

    let config = config.map_or_else(|| Ok(BootConfig::default()), BootConfig::parse)?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_stack_size(8 << 20)
        .enable_all()
        .build()?;
    let model_name = artifact
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let engine = if config.engine {
        let doc = format!(
            "[metal]\ngpu_mem_utilization = {:?}\n",
            config.gpu_mem_utilization
        );
        Some((
            runtime::engine::backend::open::metal(doc.as_bytes())?,
            poem_ir::Platform::Metal,
        ))
    } else {
        None
    };
    let loaded = runtime::embed::load(&config, artifact, &model_name, engine)?;
    let host = runtime::embed::Host {
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
    let embedded = runtime.block_on(runtime::embed::start(
        &config,
        &host,
        artifact.to_path_buf(),
        loaded,
        builtins,
    ))?;
    Ok(PieServer {
        summary: CString::new(serde_json::to_string(&embedded.summary)?)?,
        live: RwLock::new(Some(Live { embedded, runtime })),
        stop: watch::channel(false).0,
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
            let server = boot(&artifact, config, &home).map_err(|e| format!("{e:#}"))?;
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
            let bytes = bytes_arg(bytes, len).to_vec();
            let name = self::server(server)?.with(|live| {
                live.runtime
                    .block_on(program::add(bytes, file, version, true))
                    .map_err(|e| format!("install: {e:#}"))
            })?;
            Ok(to_c(name.to_string()))
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
            let language = program::Language::parse(str_arg(language, "language")?)
                .map_err(|e| format!("language: {e:#}"))?;
            let component = bytes_arg(bytes, len).to_vec();
            self::server(server)?.with(|live| {
                live.runtime
                    .block_on(program::add_language(language, component))
                    .map_err(|e| format!("install language: {e:#}"))
            })?;
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
            let id = self::server(server)?.with(|live| {
                let _entered = live.runtime.enter();
                runtime::server::open_session().map_err(|e| format!("open session: {e:#}"))
            })?;
            *session = id;
            Ok(0)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_server_close_session(server: *const PieServer, session: u32) {
    if let Ok(server) = unsafe { self::server(server) } {
        let _ = server.with(|live| {
            let _entered = live.runtime.enter();
            runtime::server::close_session(session);
            Ok(())
        });
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
            self::server(server)?.with(|live| {
                let _entered = live.runtime.enter();
                runtime::embed::send_frame(session, bytes_arg(frame, len))
                    .map_err(|e| format!("{e:#}"))
            })?;
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
            let server = self::server(server)?;
            let mut stop = server.stop.subscribe();
            let frames = server.with(|live| {
                live.runtime.block_on(async {
                    tokio::select! {
                        frames = runtime::embed::recv_frames(
                            session,
                            u64::from(max_wait_ms),
                            max.max(1) as usize,
                        ) => frames.map_err(|e| format!("{e:#}")),
                        _ = stop.wait_for(|stopping| *stopping) => Ok(Vec::new()),
                    }
                })
            })?;
            for frame in &frames {
                sink(ctx, frame.as_ptr(), frame.len());
            }
            Ok(frames.len() as i32)
        })
    }
}

/// Wakes every waiting receive, then stops the runtime once no call is using it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_server_shutdown(server: *const PieServer) {
    let Ok(server) = (unsafe { self::server(server) }) else {
        return;
    };
    server.stop.send_replace(true);
    let live = server.live.write().unwrap().take();
    if let Some(Live { embedded, runtime }) = live
        && let Err(error) = runtime.block_on(embedded.shutdown())
    {
        tracing::warn!("shutdown: {error:#}");
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_server_free(server: *mut PieServer) {
    if !server.is_null() {
        unsafe { pie_server_shutdown(server) };
        drop(unsafe { Box::from_raw(server) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(unsafe { CString::from_raw(s) });
    }
}
