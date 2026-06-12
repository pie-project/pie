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
        LogStream {
            dest: Dest::Process(process_id),
            is_stderr: false,
        }
    }

    pub fn new_stderr(process_id: ProcessId) -> Self {
        LogStream {
            dest: Dest::Process(process_id),
            is_stderr: true,
        }
    }

    pub fn new_server_stdout(program: Arc<str>) -> Self {
        LogStream {
            dest: Dest::Log(program),
            is_stderr: false,
        }
    }

    pub fn new_server_stderr(program: Arc<str>) -> Self {
        LogStream {
            dest: Dest::Log(program),
            is_stderr: true,
        }
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

// =============================================================================
// Tests — the `Server` destination (daemon request path). Regression cover for
// #300: daemon guest stderr was discarded by wasmtime's default sink because it
// was never wired to a destination.
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tracing_subscriber::fmt::MakeWriter;

    /// A `MakeWriter` that captures all tracing output into a shared buffer.
    #[derive(Clone)]
    struct CaptureWriter(Arc<Mutex<Vec<u8>>>);

    impl io::Write for CaptureWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'a> MakeWriter<'a> for CaptureWriter {
        type Writer = CaptureWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Run `f` with a tracing subscriber capturing into a buffer; return the text.
    fn capture_tracing(f: impl FnOnce()) -> String {
        let buf = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_writer(CaptureWriter(buf.clone()))
            .with_ansi(false)
            .with_max_level(tracing::Level::INFO)
            .finish();
        tracing::subscriber::with_default(subscriber, f);
        String::from_utf8(buf.lock().unwrap().clone()).unwrap()
    }

    #[test]
    fn server_stderr_surfaces_to_tracing_with_program_tag() {
        let out = capture_tracing(|| {
            let mut s = LogStream::new_server_stderr(Arc::from("chat-apc"));
            s.write_bytes(b"clock skew detected: 42s\n");
        });
        assert!(
            out.contains("clock skew detected: 42s"),
            "stderr text missing: {out:?}"
        );
        assert!(out.contains("chat-apc"), "program tag missing: {out:?}");
        assert!(out.contains("WARN"), "stderr should log at WARN: {out:?}");
    }

    #[test]
    fn server_stdout_surfaces_at_info() {
        let out = capture_tracing(|| {
            let mut s = LogStream::new_server_stdout(Arc::from("helloworld"));
            s.write_bytes(b"served request\n");
        });
        assert!(
            out.contains("served request"),
            "stdout text missing: {out:?}"
        );
        assert!(out.contains("INFO"), "stdout should log at INFO: {out:?}");
    }

    #[test]
    fn server_drops_empty_and_newline_only_writes() {
        let out = capture_tracing(|| {
            let mut s = LogStream::new_server_stderr(Arc::from("noisy"));
            s.write_bytes(b"");
            s.write_bytes(b"\n");
        });
        assert!(out.is_empty(), "blank writes should emit nothing: {out:?}");
    }
}
