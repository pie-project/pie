use std::fmt;
use std::sync::{Arc, Mutex};

use wasmtime::component::ResourceTable;

/// Where guest stdout/stderr bytes go.
pub type Sink = Box<dyn FnMut(&[u8]) + Send>;

/// Shared so every `get-stdout` call can hand out a fresh stream resource over
/// the same sink, as wasmtime-wasi does.
pub(crate) type SharedSink = Arc<Mutex<Sink>>;

/// Per-store WASI state: args, env, stdio sinks, and the insecure seed.
pub struct WasiWebCtx {
    pub(crate) args: Vec<String>,
    pub(crate) env: Vec<(String, String)>,
    pub(crate) stdout: SharedSink,
    pub(crate) stderr: SharedSink,
    pub(crate) insecure_seed: (u64, u64),
}

impl WasiWebCtx {
    pub fn builder() -> WasiWebCtxBuilder {
        WasiWebCtxBuilder::default()
    }
}

/// Builder for [`WasiWebCtx`]; output not routed to a sink is dropped.
#[derive(Default)]
pub struct WasiWebCtxBuilder {
    args: Vec<String>,
    env: Vec<(String, String)>,
    stdout: Option<Sink>,
    stderr: Option<Sink>,
}

impl WasiWebCtxBuilder {
    pub fn arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    pub fn args(mut self, args: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.args.extend(args.into_iter().map(Into::into));
        self
    }

    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.push((key.into(), value.into()));
        self
    }

    pub fn envs(
        mut self,
        vars: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Self {
        self.env
            .extend(vars.into_iter().map(|(k, v)| (k.into(), v.into())));
        self
    }

    pub fn stdout(mut self, sink: impl FnMut(&[u8]) + Send + 'static) -> Self {
        self.stdout = Some(Box::new(sink));
        self
    }

    pub fn stderr(mut self, sink: impl FnMut(&[u8]) + Send + 'static) -> Self {
        self.stderr = Some(Box::new(sink));
        self
    }

    pub fn build(self) -> WasiWebCtx {
        fn shared(sink: Option<Sink>) -> SharedSink {
            Arc::new(Mutex::new(sink.unwrap_or_else(|| Box::new(|_| {}))))
        }
        WasiWebCtx {
            args: self.args,
            env: self.env,
            stdout: shared(self.stdout),
            stderr: shared(self.stderr),
            // Hash-DoS seed; a failed getrandom only weakens hash maps, so
            // fall back to a fixed seed rather than fail construction.
            insecure_seed: crate::random::seed_pair()
                .unwrap_or((0x9e37_79b9_7f4a_7c15, 0xbf58_476d_1ce4_e5b9)),
        }
    }
}

/// Borrowed view of the store data the host functions operate on.
pub struct WasiWebCtxView<'a> {
    pub ctx: &'a mut WasiWebCtx,
    pub table: &'a mut ResourceTable,
}

/// Implemented by the store data type; mirrors `wasmtime_wasi::WasiView`.
pub trait WasiWebView: Send {
    fn wasi_web(&mut self) -> WasiWebCtxView<'_>;
}

/// The error `wasi:cli/exit` traps with; downcast it to recover the status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct I32Exit(pub i32);

impl fmt::Display for I32Exit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "exited with i32 exit status {}", self.0)
    }
}

impl std::error::Error for I32Exit {}
