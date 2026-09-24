use std::future::Future;

use wasm_bindgen::prelude::*;

use crate::boot::{self, BootConfig, Mount};

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

#[wasm_bindgen]
pub fn pie_init(log_filter: Option<String>) {
    console_error_panic_hook::set_once();
    crate::log::install(log_filter.as_deref().unwrap_or("info"));
    web_std::time::set_clock(clock_ms);
    web_std::executor::set_wake_hook(wake);
    wasmtime_web::set_clock(clock_ns);
    tracing::info!("pie web host initialised");
}

#[unsafe(no_mangle)]
pub extern "C" fn pie_tick() -> f64 {
    let wasi_next = wasmtime_web::fire_due_timers();
    let next = web_std::tick(100_000);
    let now = wasmtime_web::now();
    let wasi_ms = wasi_next.map(|deadline| deadline.saturating_sub(now) as f64 / 1_000_000.0);
    let rt_ms = next.map(|d| d.as_secs_f64() * 1000.0);
    match (rt_ms, wasi_ms) {
        (Some(a), Some(b)) => a.min(b),
        (Some(a), None) | (None, Some(a)) => a,
        (None, None) => -1.0,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn pie_fiber_entry(entry: usize, arg0: *mut u8, top: *mut u8) -> *mut u8 {
    wasmtime_web::fiber_entry(entry, arg0, top)
}

fn promise<F>(future: F) -> js_sys::Promise
where
    F: Future<Output = anyhow::Result<JsValue>> + 'static,
{
    let mut settle = None;
    let promise = js_sys::Promise::new(&mut |resolve, reject| {
        settle = Some((resolve, reject));
    });
    let (resolve, reject) = settle.expect("Promise constructor calls its executor");
    web_std::spawn(async move {
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

mod lazy {
    use std::cell::{Cell, RefCell};
    use std::collections::HashMap;
    use std::task::{Poll, Waker};

    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
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

    pub fn fetch(offset: u64, into: &mut [u8]) -> Result<(), String> {
        if !web_std::thread::in_green_thread() {
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
        let answer = web_std::block_on_poll(|cx| {
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

#[wasm_bindgen]
pub fn pie_range_ready(request: u32, bytes: js_sys::Uint8Array) {
    lazy::answer(request, Ok(bytes));
}

#[wasm_bindgen]
pub fn pie_range_failed(request: u32, error: String) {
    lazy::answer(request, Err(error));
}

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

#[wasm_bindgen]
pub fn pie_install_program(
    bytes: Vec<u8>,
    file: String,
    version: Option<String>,
) -> js_sys::Promise {
    promise(async move {
        let name = runtime::inferlet::program::add(bytes, &file, version.as_deref(), true).await?;
        Ok(JsValue::from_str(&name.to_string()))
    })
}

/// Hand the runtime a language component (`python`, `javascript`) from
/// bytes: this host reads no files, so a script inferlet can only run once
/// the page has installed its language component this way.
#[wasm_bindgen]
pub fn pie_install_language(language: String, component: Vec<u8>) -> js_sys::Promise {
    promise(async move {
        let language = runtime::inferlet::program::Language::parse(&language)?;
        runtime::inferlet::program::add_language(language, component).await?;
        Ok(JsValue::from_str(language.name()))
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

#[wasm_bindgen]
pub fn pie_send_frame(session: u32, frame: js_sys::Uint8Array) -> Result<(), JsError> {
    let bytes = frame.to_vec();
    let message: client_api::ClientMessage =
        rmp_serde::from_slice(&bytes).map_err(|e| JsError::new(&format!("client frame: {e}")))?;
    runtime::server::send_client_message(session, message)
        .map_err(|e| JsError::new(&format!("{e:#}")))
}

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
