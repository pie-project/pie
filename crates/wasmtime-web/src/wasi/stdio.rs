use wasmtime_wasi_io::async_trait;
use wasmtime_wasi_io::bytes::Bytes;
use wasmtime_wasi_io::poll::Pollable;
use wasmtime_wasi_io::streams::{InputStream, OutputStream, StreamError, StreamResult};

use crate::wasi::ctx::SharedSink;

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
