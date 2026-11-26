#[cfg(not(target_arch = "wasm32"))]
use bytes::Bytes;
#[cfg(not(target_arch = "wasm32"))]
use std::io;
#[cfg(not(target_arch = "wasm32"))]
use std::pin::Pin;
#[cfg(not(target_arch = "wasm32"))]
use std::task::{Context, Poll};
#[cfg(not(target_arch = "wasm32"))]
use tokio::io::AsyncWrite;
#[cfg(not(target_arch = "wasm32"))]
use wasmtime_wasi::async_trait;
#[cfg(not(target_arch = "wasm32"))]
use wasmtime_wasi::cli::IsTerminal;
#[cfg(not(target_arch = "wasm32"))]
use wasmtime_wasi::cli::StdoutStream;
#[cfg(not(target_arch = "wasm32"))]
use wasmtime_wasi::p2::{OutputStream, Pollable, StreamResult};

use std::sync::{Arc, Mutex};

use crate::inferlet::process;

use super::ProcessId;

/// The end of what a guest wrote to stderr, kept whether or not its output
/// streams to a client: a Rust guest writes its panic message there just
/// before it traps, and the trap says it ([`StderrTail::said`]).
#[derive(Clone, Default)]
pub struct StderrTail(Arc<Mutex<String>>);

/// How much of a guest's stderr its tail keeps: a panic message with the
/// WIT error it carries, and the lines just before it.
const STDERR_TAIL_BYTES: usize = 4096;

impl StderrTail {
    fn push(&self, text: &str) {
        let mut tail = self.0.lock().expect("the stderr tail poisoned");
        tail.push_str(text);
        if tail.len() <= STDERR_TAIL_BYTES {
            return;
        }
        let mut cut = tail.len() - STDERR_TAIL_BYTES;
        while !tail.is_char_boundary(cut) {
            cut += 1;
        }
        // From the first whole line kept, when there is one.
        if let Some(newline) = tail[cut..].find('\n')
            && cut + newline + 1 < tail.len()
        {
            cut += newline + 1;
        }
        tail.drain(..cut);
    }

    /// What the guest last wrote to stderr, trimmed; `None` if nothing.
    pub fn said(&self) -> Option<String> {
        let tail = self.0.lock().expect("the stderr tail poisoned");
        let said = tail.trim();
        (!said.is_empty()).then(|| said.to_string())
    }
}

#[derive(Clone)]
pub struct LogStream {
    process_id: ProcessId,
    is_stderr: bool,
    /// Whether it goes on to the process's client as its output events.
    forward: bool,
    tail: Option<StderrTail>,
}

impl LogStream {
    pub fn new_stdout(process_id: ProcessId) -> Self {
        LogStream {
            process_id,
            is_stderr: false,
            forward: true,
            tail: None,
        }
    }

    /// Stderr into `tail`, and on to the client when `forward`.
    pub fn new_stderr(process_id: ProcessId, tail: StderrTail, forward: bool) -> Self {
        LogStream {
            process_id,
            is_stderr: true,
            forward,
            tail: Some(tail),
        }
    }

    pub(crate) fn write_bytes(&self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        let content = String::from_utf8_lossy(bytes).to_string();
        if let Some(tail) = &self.tail {
            tail.push(&content);
        }
        if !self.forward {
            return;
        }
        if self.is_stderr {
            process::stderr(self.process_id, content);
        } else {
            process::stdout(self.process_id, content);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl StdoutStream for LogStream {
    fn p2_stream(&self) -> Box<dyn OutputStream> {
        Box::new(self.clone())
    }
    fn async_stream(&self) -> Box<dyn AsyncWrite + Send + Sync> {
        Box::new(self.clone())
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl IsTerminal for LogStream {
    fn is_terminal(&self) -> bool {
        false
    }
}

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
impl Pollable for LogStream {
    async fn ready(&mut self) {}
}

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_guest_panic_message_is_its_stderr_tail() {
        let tail = StderrTail::default();
        let err = LogStream::new_stderr(ProcessId::new_v4(), tail.clone(), false);
        assert_eq!(tail.said(), None, "nothing written: nothing said");
        // As std's panic hook writes it: the report, then its note.
        err.write_bytes(b"\nthread '<unnamed>' panicked at src/eta.rs:164:14:\n");
        err.write_bytes(
            b"stage seed: channel is poisoned: device memory can never hold this need\n",
        );
        err.write_bytes(b"note: run with `RUST_BACKTRACE=1` to display a backtrace\n");
        let said = tail.said().expect("the panic was written");
        assert!(said.starts_with("thread '<unnamed>' panicked at src/eta.rs:164:14:"));
        assert!(said.contains("channel is poisoned: device memory can never hold this need"));
    }

    #[test]
    fn a_long_stderr_keeps_its_last_whole_lines() {
        let tail = StderrTail::default();
        let err = LogStream::new_stderr(ProcessId::new_v4(), tail.clone(), false);
        for line in 0..STDERR_TAIL_BYTES {
            err.write_bytes(format!("line {line} é\n").as_bytes());
        }
        err.write_bytes(b"panicked: the last word\n");
        let said = tail.said().unwrap();
        assert!(said.len() <= STDERR_TAIL_BYTES);
        assert!(
            said.starts_with("line "),
            "starts on a whole line: {said:.40}"
        );
        assert!(said.ends_with("panicked: the last word"));
    }
}
