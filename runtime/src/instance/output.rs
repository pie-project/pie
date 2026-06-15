//! Output streaming for WASM instance stdout/stderr.
//!
//! Provides `LogStream` — a WASI-compatible stream that routes output either
//! through the process actor (`process::stdout` / `process::stderr`, read by an
//! attached client) or to pie-server's own `tracing` log (daemon request path,
//! which has no attached client — see `daemon::Daemon::handle_request`).

use bytes::Bytes;
use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::io::AsyncWrite;
use wasmtime_wasi::async_trait;
use wasmtime_wasi::cli::IsTerminal;
use wasmtime_wasi::cli::StdoutStream;
use wasmtime_wasi::p2::{OutputStream, Pollable, StreamResult};

use crate::process;

use crate::process::ProcessId;

/// Where a `LogStream`'s bytes are routed.
#[derive(Clone)]
enum Dest {
    /// Per-process actor channel, drained by an attached client.
    Process(ProcessId),
    /// pie-server's `tracing` log, tagged with the program name for triage.
    /// Used by daemon components, which have no attached client.
    Log(Arc<str>),
}

/// A WASI-compatible output stream that routes guest stdout/stderr.
#[derive(Clone)]
pub struct LogStream {
    dest: Dest,
    is_stderr: bool,
}

impl LogStream {
    pub fn new_stdout(process_id: ProcessId) -> Self {
        LogStream { dest: Dest::Process(process_id), is_stderr: false }
    }

    pub fn new_stderr(process_id: ProcessId) -> Self {
        LogStream { dest: Dest::Process(process_id), is_stderr: true }
    }

    pub fn new_server_stdout(program: Arc<str>) -> Self {
        LogStream { dest: Dest::Log(program), is_stderr: false }
    }

    pub fn new_server_stderr(program: Arc<str>) -> Self {
        LogStream { dest: Dest::Log(program), is_stderr: true }
    }

    /// Dispatch output to its destination.
    fn write_bytes(&self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        match &self.dest {
            Dest::Process(process_id) => {
                let content = String::from_utf8_lossy(bytes).to_string();
                if self.is_stderr {
                    process::stderr(*process_id, content);
                } else {
                    process::stdout(*process_id, content);
                }
            }
            Dest::Log(program) => {
                // Strip the trailing newline so each `eprintln!`/`println!`
                // becomes one clean tracing event rather than an empty extra line.
                let content = String::from_utf8_lossy(bytes);
                let text = content.trim_end_matches(['\n', '\r']);
                if text.is_empty() {
                    return;
                }
                if self.is_stderr {
                    tracing::warn!(target: "pie::daemon::guest", program = %program, "{text}");
                } else {
                    tracing::info!(target: "pie::daemon::guest", program = %program, "{text}");
                }
            }
        }
    }
}

// =============================================================================
// WASI Trait Implementations
// =============================================================================

impl StdoutStream for LogStream {
    fn p2_stream(&self) -> Box<dyn OutputStream> {
        Box::new(self.clone())
    }
    fn async_stream(&self) -> Box<dyn AsyncWrite + Send + Sync> {
        Box::new(self.clone())
    }
}

impl IsTerminal for LogStream {
    fn is_terminal(&self) -> bool {
        false
    }
}

impl OutputStream for LogStream {
    fn write(&mut self, bytes: Bytes) -> StreamResult<()> {
        self.write_bytes(&bytes);
        Ok(())
    }

    fn flush(&mut self) -> StreamResult<()> {
        Ok(())
    }

    fn check_write(&mut self) -> StreamResult<usize> {
        Ok(1024 * 1024)
    }
}

#[async_trait]
impl Pollable for LogStream {
    async fn ready(&mut self) {
        // Always ready — no backpressure.
    }
}

impl AsyncWrite for LogStream {
    fn poll_write(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        self.write_bytes(buf);
        Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}
