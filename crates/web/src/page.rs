//! The exports a page calls, and the imports it provides.
//!
//! Two calling conventions meet here. wasm-bindgen exports (`pie_boot`,
//! `pie_send`, …) are ordinary: they return at once, handing back a promise
//! for anything that takes time. `pie_tick` and `pie_fiber_entry` are raw
//! exports the page wraps in `WebAssembly.promising`, because a poll may
//! switch fibers and only a promising activation can be suspended.

use std::future::Future;

use wasm_bindgen::prelude::*;

use crate::boot::{self, BootConfig, Mount};

// Raw imports from the page's platform module — the same module the fiber
// hooks come from (`wasmtime-web`), so the page attaches it once.
#[link(wasm_import_module = "./platform.mjs")]
unsafe extern "C" {
    fn pie_now() -> f64;
    fn pie_wake();
}

fn clock_ms() -> f64 {
    unsafe { pie_now() }
}

fn clock_ns() -> u64 {
    (clock_ms() * 1_000_000.0) as u64
}

fn wake() {
    unsafe { pie_wake() }
}

/// Install the panic hook, the console logger, the clock and the wake hook.
/// Call once, before anything else.
#[wasm_bindgen]
pub fn pie_init(log_filter: Option<String>) {
    console_error_panic_hook::set_once();
    crate::log::install(log_filter.as_deref().unwrap_or("info"));
    web_rt::time::set_clock(clock_ms);
    web_rt::executor::set_wake_hook(wake);
    wasi_web::set_clock(clock_ns);
    tracing::info!("pie web host initialised");
}

/// One executor tick. Returns milliseconds until the next timer needs one,
/// or -1 when nothing is scheduled. The page calls this through
/// `WebAssembly.promising`.
#[unsafe(no_mangle)]
pub extern "C" fn pie_tick() -> f64 {
    let wasi_next = wasi_web::fire_due_timers();
    let next = web_rt::tick(100_000);
    let now = wasi_web::now();
    let wasi_ms = wasi_next.map(|deadline| deadline.saturating_sub(now) as f64 / 1_000_000.0);
    let rt_ms = next.map(|d| d.as_secs_f64() * 1000.0);
    match (rt_ms, wasi_ms) {
        (Some(a), Some(b)) => a.min(b),
        (Some(a), None) | (None, Some(a)) => a,
        (None, None) => -1.0,
    }
}

/// The fiber trampoline the page starts every fiber on (see `wasmtime-web`).
#[unsafe(no_mangle)]
pub extern "C" fn pie_fiber_entry(entry: usize, arg0: *mut u8, top: *mut u8) -> *mut u8 {
    wasmtime_web::fiber_entry(entry, arg0, top)
}

/// Run `future` as an executor task and hand the page a promise for it.
fn promise<F>(future: F) -> js_sys::Promise
where
    F: Future<Output = anyhow::Result<JsValue>> + 'static,
{
    let mut settle = None;
    let promise = js_sys::Promise::new(&mut |resolve, reject| {
        settle = Some((resolve, reject));
    });
    let (resolve, reject) = settle.expect("Promise constructor calls its executor");
    web_rt::spawn(async move {
        match future.await {
            Ok(value) => {
                let _ = resolve.call1(&JsValue::UNDEFINED, &value);
            }
            Err(error) => {
                tracing::error!("{error:#}");
                let _ = reject.call1(
                    &JsValue::UNDEFINED,
                    &JsValue::from_str(&format!("{error:#}")),
                );
            }
        }
    });
    promise
}

// ---- lazy model bytes -------------------------------------------------------
//
// A tab's wasm memory ends at 4 GiB, and the whole artifact in it would
// leave nothing for a model past about 3 GB. A boot keeps the
// artifact where it is — on the server, or in the origin's private file
// system — and the page serves byte ranges as the loader asks for them:
// the mount's fetch posts `{offset, len, request}` to the page
// (`__pieHost.fetchRange`), parks the green thread that reads, and the page
// answers with `pie_range_ready`/`pie_range_failed`, which wake it.

mod lazy {
    use std::cell::{Cell, RefCell};
    use std::collections::HashMap;
    use std::task::{Poll, Waker};

    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        /// The page's range server (installed on `globalThis.__pieHost` by
        /// worker.mjs): read `len` bytes at `offset` and answer `request`.
        #[wasm_bindgen(js_namespace = __pieHost, js_name = fetchRange)]
        fn fetch_range(offset: f64, len: f64, request: u32);
    }

    struct Slot {
        result: Option<Result<js_sys::Uint8Array, String>>,
        waker: Option<Waker>,
    }

    thread_local! {
        static RANGES: RefCell<HashMap<u32, Slot>> = RefCell::new(HashMap::new());
        static NEXT: Cell<u32> = const { Cell::new(1) };
    }

    /// `ztensor::memfs::Fetch` over the page: ask for the range, park until
    /// it is answered, copy it out.
    pub fn fetch(offset: u64, into: &mut [u8]) -> Result<(), String> {
        if !web_rt::thread::in_green_thread() {
            return Err(
                "a lazily mounted artifact was read off a green thread, where the page \
                 cannot serve it"
                    .into(),
            );
        }
        let request = NEXT.with(|next| {
            let id = next.get();
            next.set(id.wrapping_add(1).max(1));
            id
        });
        RANGES.with(|ranges| {
            ranges.borrow_mut().insert(
                request,
                Slot {
                    result: None,
                    waker: None,
                },
            )
        });
        tracing::debug!(request, offset, len = into.len(), "range requested");
        fetch_range(offset as f64, into.len() as f64, request);
        let answer = web_rt::block_on_poll(|cx| {
            RANGES.with(|ranges| {
                let mut ranges = ranges.borrow_mut();
                let slot = ranges
                    .get_mut(&request)
                    .expect("a parked range request keeps its slot");
                match slot.result.take() {
                    Some(answer) => Poll::Ready(answer),
                    None => {
                        slot.waker = Some(cx.waker().clone());
                        Poll::Pending
                    }
                }
            })
        });
        RANGES.with(|ranges| ranges.borrow_mut().remove(&request));
        let bytes = answer?;
        if bytes.length() as usize != into.len() {
            return Err(format!(
                "the page served {} bytes for the range {offset}+{}",
                bytes.length(),
                into.len()
            ));
        }
        bytes.copy_to(into);
        Ok(())
    }

    /// Deliver the page's answer and wake the reader parked on it.
    pub fn answer(request: u32, result: Result<js_sys::Uint8Array, String>) {
        let waker = RANGES.with(|ranges| {
            let mut ranges = ranges.borrow_mut();
            match ranges.get_mut(&request) {
                Some(slot) => {
                    slot.result = Some(result);
                    slot.waker.take()
                }
                None => {
                    tracing::warn!(request, "a range was answered for no pending request");
                    None
                }
            }
        });
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

/// The page's answer to a range request: the bytes, exactly as many as asked.
#[wasm_bindgen]
pub fn pie_range_ready(request: u32, bytes: js_sys::Uint8Array) {
    lazy::answer(request, Ok(bytes));
}

/// The page could not serve a range; the read that asked fails with `error`.
#[wasm_bindgen]
pub fn pie_range_failed(request: u32, error: String) {
    lazy::answer(request, Err(error));
}

/// Boot from an artifact of `len` bytes the page serves by range through
/// `__pieHost.fetchRange`: nothing is reserved in wasm memory, the weights
/// stream to the device tensor by tensor. Resolves with a JSON `BootSummary`.
#[wasm_bindgen]
pub fn pie_boot_lazy(config_toml: String, model_name: String, len: f64) -> js_sys::Promise {
    promise(async move {
        let config = BootConfig::parse(&config_toml)?;
        let device = if config.engine {
            Some(crate::engine::request(&config).await?)
        } else {
            None
        };
        let mount = Mount {
            len: len as u64,
            fetch: Box::new(lazy::fetch),
        };
        let summary = boot::boot(config, &model_name, mount, device).await?;
        Ok(JsValue::from_str(&serde_json::to_string(&summary)?))
    })
}

/// Install an inferlet from its component bytes and `Pie.toml`. Resolves
/// with the `name@version` to launch it by.
// (`component`, not `wasm`: a parameter named `wasm` shadows the module
// variable the generated glue reaches its allocator through.)
#[wasm_bindgen]
pub fn pie_install_program(component: Vec<u8>, manifest_toml: String) -> js_sys::Promise {
    promise(async move {
        let manifest = runtime::inferlet::program::Manifest::parse(&manifest_toml)?;
        let name = manifest.program_name().to_string();
        runtime::inferlet::program::add(component, manifest, true).await?;
        Ok(JsValue::from_str(&name))
    })
}

#[wasm_bindgen]
pub fn pie_open_session() -> Result<u32, JsError> {
    runtime::server::open_session().map_err(|e| JsError::new(&format!("{e:#}")))
}

#[wasm_bindgen]
pub fn pie_close_session(session: u32) {
    runtime::server::close_session(session);
}

/// Deliver one `ClientMessage` (JSON) to the session.
#[wasm_bindgen]
pub fn pie_send(session: u32, message_json: String) -> Result<(), JsError> {
    let message: client_api::ClientMessage = serde_json::from_str(&message_json)
        .map_err(|e| JsError::new(&format!("client message: {e}")))?;
    runtime::server::send_client_message(session, message)
        .map_err(|e| JsError::new(&format!("{e:#}")))
}

/// Wait up to `max_wait_ms` for up to `max` `ServerMessage`s. Resolves with
/// a JSON array (possibly empty).
#[wasm_bindgen]
pub fn pie_recv(session: u32, max_wait_ms: u32, max: u32) -> js_sys::Promise {
    promise(async move {
        let messages =
            runtime::server::recv_messages(session, u64::from(max_wait_ms), max as usize).await?;
        Ok(JsValue::from_str(&serde_json::to_string(&messages)?))
    })
}

/// The scheduler's own picture of engine `engine`: queues, in-flight frames,
/// the frame policy. For a page that wants to know why nothing is moving.
#[wasm_bindgen]
pub fn pie_debug(engine: u32) -> js_sys::Promise {
    promise(async move {
        let dump = runtime::scheduler::debug_dump(engine as usize).await?;
        Ok(JsValue::from_str(&dump))
    })
}

/// Executor and timer state, for the same page.
#[wasm_bindgen]
pub fn pie_executor_state() -> String {
    format!(
        "tasks={} next_timer={:?} parked={:?}",
        web_rt::executor::task_count(),
        web_rt::time::next_deadline(),
        engine_wgpu::device::host::parked()
    )
}

/// The wire form `pie serve`'s WebSocket carries: one MessagePack-encoded
/// `ClientMessage` in, for the JavaScript client SDK to speak unchanged.
#[wasm_bindgen]
pub fn pie_send_frame(session: u32, frame: js_sys::Uint8Array) -> Result<(), JsError> {
    let bytes = frame.to_vec();
    let message: client_api::ClientMessage =
        rmp_serde::from_slice(&bytes).map_err(|e| JsError::new(&format!("client frame: {e}")))?;
    runtime::server::send_client_message(session, message)
        .map_err(|e| JsError::new(&format!("{e:#}")))
}

/// The wire form out: resolves with an array of `Uint8Array`, one
/// MessagePack-encoded `ServerMessage` each (named fields, as the WebSocket
/// server encodes them).
#[wasm_bindgen]
pub fn pie_recv_frames(session: u32, max_wait_ms: u32, max: u32) -> js_sys::Promise {
    promise(async move {
        let messages =
            runtime::server::recv_messages(session, u64::from(max_wait_ms), max as usize).await?;
        let frames = js_sys::Array::new();
        for message in &messages {
            let bytes = rmp_serde::to_vec_named(message)?;
            frames.push(&js_sys::Uint8Array::from(bytes.as_slice()));
        }
        Ok(frames.into())
    })
}
