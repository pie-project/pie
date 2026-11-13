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

use crate::inferlet::process;

use super::ProcessId;

#[derive(Clone)]
pub struct LogStream {
    process_id: ProcessId,
    is_stderr: bool,
}

impl LogStream {
    pub fn new_stdout(process_id: ProcessId) -> Self {
        LogStream {
            process_id,
            is_stderr: false,
        }
    }

    pub fn new_stderr(process_id: ProcessId) -> Self {
        LogStream {
            process_id,
            is_stderr: true,
        }
    }

    pub(crate) fn write_bytes(&self, bytes: &[u8]) {
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
