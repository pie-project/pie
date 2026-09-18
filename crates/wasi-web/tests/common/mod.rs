//! Shared test harness: a store type implementing `WasiWebView`, an engine
//! with async + component-model-async on, and a tiny executor that ticks the
//! wasi-web timers between polls the way a browser host would.

use std::future::Future;
use std::pin::pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use wasi_web::{WasiWebCtx, WasiWebCtxView, WasiWebView};
use wasmtime::component::ResourceTable;
use wasmtime::{Config, Engine};

pub struct State {
    ctx: WasiWebCtx,
    table: ResourceTable,
}

impl WasiWebView for State {
    fn wasi_web(&mut self) -> WasiWebCtxView<'_> {
        WasiWebCtxView {
            ctx: &mut self.ctx,
            table: &mut self.table,
        }
    }
}

pub fn engine() -> Engine {
    let mut config = Config::new();
    config.wasm_component_model(true);
    config.wasm_component_model_async(true);
    Engine::new(&config).expect("engine")
}

pub type Captured = Arc<Mutex<Vec<u8>>>;

/// A state whose stdout/stderr bytes land in the returned buffers.
pub fn state() -> (State, Captured, Captured) {
    let stdout = Arc::new(Mutex::new(Vec::new()));
    let stderr = Arc::new(Mutex::new(Vec::new()));
    let (out, err) = (stdout.clone(), stderr.clone());
    let ctx = WasiWebCtx::builder()
        .arg("guest")
        .env("PIE", "1")
        .stdout(move |bytes| out.lock().unwrap().extend_from_slice(bytes))
        .stderr(move |bytes| err.lock().unwrap().extend_from_slice(bytes))
        .build();
    let state = State {
        ctx,
        table: ResourceTable::new(),
    };
    (state, stdout, stderr)
}

pub fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(v) = future.as_mut().poll(&mut cx) {
            return v;
        }
        wasi_web::fire_due_timers();
        std::thread::yield_now();
    }
}
