//! stdout/stderr as `wasi:io` output streams over a host sink; stdin is closed.

use wasmtime_wasi_io::async_trait;
use wasmtime_wasi_io::bytes::Bytes;
use wasmtime_wasi_io::poll::Pollable;
use wasmtime_wasi_io::streams::{InputStream, OutputStream, StreamError, StreamResult};

use crate::ctx::SharedSink;

/// Writes complete immediately: the sink is synchronous, so there is nothing
/// to wait for and `blocking-write-and-flush` never yields.
pub(crate) struct SinkStream(pub(crate) SharedSink);

impl OutputStream for SinkStream {
    fn write(&mut self, bytes: Bytes) -> StreamResult<()> {
        (self.0.lock().unwrap())(&bytes);
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
impl Pollable for SinkStream {
    async fn ready(&mut self) {}
}

/// A stdin with nothing behind it: every read reports end of stream.
pub(crate) struct ClosedInput;

impl InputStream for ClosedInput {
    fn read(&mut self, _size: usize) -> StreamResult<Bytes> {
        Err(StreamError::Closed)
    }
}

#[async_trait]
impl Pollable for ClosedInput {
    async fn ready(&mut self) {}
}
