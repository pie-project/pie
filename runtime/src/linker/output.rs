//! Output streaming for WASM instance stdout/stderr.
//!
//! Provides `LogStream` — a WASI-compatible stream that routes output
//! through the process actor via `process::stdout` / `process::stderr`.

use bytes::Bytes;
use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::io::AsyncWrite;
use wasmtime_wasi::async_trait;
use wasmtime_wasi::cli::IsTerminal;
use wasmtime_wasi::cli::StdoutStream;
use wasmtime_wasi::p2::{OutputStream, Pollable, StreamResult};

use crate::process;

type ProcessId = usize;

/// A WASI-compatible output stream that routes to the process actor.
#[derive(Clone)]
pub struct LogStream {
    process_id: ProcessId,
    is_stderr: bool,
}

impl LogStream {
    pub fn new_stdout(process_id: ProcessId) -> Self {
        LogStream { process_id, is_stderr: false }
    }

    pub fn new_stderr(process_id: ProcessId) -> Self {
        LogStream { process_id, is_stderr: true }
    }

    /// Dispatch output to the process actor.
    fn write_bytes(&self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        let content = String::from_utf8_lossy(bytes).to_string();
        if self.is_stderr {
            process::stderr(self.process_id, content);
        } else {
            process::stdout(self.process_id, content);
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
