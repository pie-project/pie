//! Output streaming and buffering for instance stdout/stderr.
//!
//! This module provides the infrastructure for capturing and delivering
//! output from WASM instances to connected clients.

use bytes::Bytes;
use ringbuffer::{AllocRingBuffer, RingBuffer};
use std::io;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use tokio::io::AsyncWrite;
use tokio::sync::Notify;
use wasmtime_wasi::async_trait;
use wasmtime_wasi::cli::IsTerminal;
use wasmtime_wasi::cli::StdoutStream;
use wasmtime_wasi::p2::{OutputStream, Pollable, StreamResult};

use super::instance::InstanceId;

#[derive(Clone, Debug)]
pub enum OutputChannel {
    Stdout,
    Stderr,
}

impl OutputChannel {
    /// Send the output to the session actor for the client attached to this instance
    fn dispatch_output(&self, content: String, instance_id: InstanceId) {
        let output_type = self.clone();
        // Spawn async task since this is called from sync context
        tokio::spawn(async move {
            if let Ok(Some(client_id)) = crate::server::get_client_id(instance_id).await {
                crate::server::sessions::streaming_output(
                    client_id,
                    instance_id,
                    output_type,
                    content,
                ).ok();
            }
        });
    }
}

/// Output mode for LogStream
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum OutputDelivery {
    /// Buffer output in a circular buffer, discarding old content when full
    Buffered = 0,
    /// Stream buffered content via instance events
    Streamed = 1,
}

impl OutputDelivery {
    fn from_u8(value: u8) -> Self {
        match value {
            0 => OutputDelivery::Buffered,
            1 => OutputDelivery::Streamed,
            _ => OutputDelivery::Buffered, // Default to buffering for invalid values
        }
    }

    fn to_u8(self) -> u8 {
        self as u8
    }
}

/// Controller for controlling the output delivery mode of a running instance
#[derive(Clone)]
pub struct OutputDeliveryCtrl {
    stdout_stream: LogStream,
    stderr_stream: LogStream,
}

impl OutputDeliveryCtrl {
    /// Create a new output delivery controller with stdout and stderr streams.
    pub(super) fn new(stdout_stream: LogStream, stderr_stream: LogStream) -> Self {
        Self {
            stdout_stream,
            stderr_stream,
        }
    }

    /// Set output mode
    pub fn set_output_delivery(&self, output_delivery: OutputDelivery) {
        match output_delivery {
            OutputDelivery::Buffered => {
                self.stdout_stream.set_deliver_to_buffer();
                self.stderr_stream.set_deliver_to_buffer();
            }
            OutputDelivery::Streamed => {
                self.stdout_stream.set_deliver_to_stream();
                self.stderr_stream.set_deliver_to_stream();
            }
        }
    }

    /// Allow output to be written to the streams.
    /// This should be called after the instance ID has been communicated to the client
    /// to prevent a race condition where output arrives before the instance ID.
    pub fn allow_output(&self) {
        self.stdout_stream.allow_output();
        self.stderr_stream.allow_output();
    }
}

#[derive(Clone)]
pub struct LogStream {
    channel: OutputChannel,
    state: Arc<LogStreamState>,
}

struct LogStreamState {
    instance_id: InstanceId,
    mode: AtomicU8,
    buffer: Mutex<AllocRingBuffer<u8>>,
    /// Tracks whether output is allowed to be written.
    /// Starts as false to prevent output before the instance ID is sent to the client.
    output_allowed: AtomicBool,
    /// Notifies async waiters when output becomes allowed.
    output_allowed_notify: Notify,
}

impl LogStream {
    /// Default buffer capacity: 1MB
    const DEFAULT_BUFFER_CAPACITY: usize = 1024 * 1024;

    pub fn new(channel: OutputChannel, instance_id: InstanceId) -> LogStream {
        LogStream {
            channel,
            state: Arc::new(LogStreamState {
                instance_id,
                mode: AtomicU8::new(OutputDelivery::Buffered.to_u8()),
                buffer: Mutex::new(AllocRingBuffer::new(Self::DEFAULT_BUFFER_CAPACITY)),
                output_allowed: AtomicBool::new(false),
                output_allowed_notify: Notify::new(),
            }),
        }
    }

    /// Allow output to be written to this stream.
    /// This should be called after the instance ID has been communicated to the client.
    fn allow_output(&self) {
        self.state.output_allowed.store(true, Ordering::Release);
        self.state.output_allowed_notify.notify_waiters();
    }

    /// Set the delivery mode to buffering
    pub fn set_deliver_to_buffer(&self) {
        self.state
            .mode
            .store(OutputDelivery::Buffered.to_u8(), Ordering::Release);
    }

    /// Set the delivery mode to streaming
    ///
    /// When transitioning from buffering to streaming, any buffered content
    /// will be immediately flushed.
    pub fn set_deliver_to_stream(&self) {
        self.state
            .mode
            .store(OutputDelivery::Streamed.to_u8(), Ordering::Release);
        self.flush_buffer();
    }

    /// Flush any buffered content to output
    fn flush_buffer(&self) {
        let mut buffer = self.state.buffer.lock().unwrap();
        if !buffer.is_empty() {
            let content = String::from_utf8_lossy(&buffer.drain().collect::<Vec<u8>>()).to_string();
            self.channel
                .dispatch_output(content, self.state.instance_id);
        }
    }

    /// Write bytes according to the current mode
    fn write_bytes(&self, bytes: &[u8]) {
        let mode = OutputDelivery::from_u8(self.state.mode.load(Ordering::Acquire));

        match mode {
            // In buffering mode, append to the circular buffer
            OutputDelivery::Buffered => {
                let mut buffer = self.state.buffer.lock().unwrap();
                buffer.extend(bytes.iter().copied());
            }
            // In streaming mode, dispatch the new content immediately
            OutputDelivery::Streamed => {
                self.channel.dispatch_output(
                    String::from_utf8_lossy(&bytes).to_string(),
                    self.state.instance_id,
                );
            }
        }
    }
}

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
        match &self.channel {
            OutputChannel::Stdout => false,
            OutputChannel::Stderr => false,
        }
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
        // If output is not allowed yet, return 0 to signal backpressure.
        // This prevents writes until the instance ID has been sent to the client.
        if !self.state.output_allowed.load(Ordering::Acquire) {
            Ok(0)
        } else {
            Ok(1024 * 1024)
        }
    }
}

#[async_trait]
impl Pollable for LogStream {
    async fn ready(&mut self) {
        // IMPORTANT: Call notified() BEFORE checking the condition to avoid
        // missing the notification (lost wakeup problem).
        let notified = self.state.output_allowed_notify.notified();

        // Wait until output is allowed before becoming ready.
        // This prevents a race condition where output is sent before
        // the client receives the instance ID.
        if !self.state.output_allowed.load(Ordering::Acquire) {
            notified.await;
        }
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
