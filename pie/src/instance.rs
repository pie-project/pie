use super::api::core::Queue;
use super::utils;
use crate::model::resource::{ResourceId, ResourceTypeId};
use crate::server::InstanceEvent;
use anyhow::{Result, format_err};
use bytes::Bytes;
use ringbuffer::{AllocRingBuffer, RingBuffer};
use std::collections::HashMap;
use std::io;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use tokio::io::AsyncWrite;
use tokio::sync::Notify;
use uuid::Uuid;
use wasmtime::component::{Resource, ResourceTable};
use wasmtime_wasi::async_trait;
use wasmtime_wasi::cli::IsTerminal;
use wasmtime_wasi::cli::StdoutStream;
use wasmtime_wasi::p2::{OutputStream, Pollable, StreamResult};
use wasmtime_wasi::{WasiCtx, WasiCtxView, WasiView};
use wasmtime_wasi_http::{WasiHttpCtx, WasiHttpView};

pub type InstanceId = Uuid;

/// Controller for controlling the output delivery mode of a running instance
#[derive(Clone)]
pub struct OutputDeliveryCtrl {
    stdout_stream: LogStream,
    stderr_stream: LogStream,
}

impl OutputDeliveryCtrl {
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

/// Manages the mapping between virtual and physical resource identifiers.
#[derive(Debug)]
struct ResourceIdMapper {
    /// A pool for acquiring and releasing unique virtual IDs.
    virtual_id_pool: utils::IdPool<u32>,
    /// The map from a virtual ID to its corresponding physical ID.
    virtual_to_physical: HashMap<ResourceId, ResourceId>,
}

impl Default for ResourceIdMapper {
    fn default() -> Self {
        ResourceIdMapper {
            virtual_id_pool: utils::IdPool::new(u32::MAX),
            virtual_to_physical: HashMap::new(),
        }
    }
}

impl ResourceIdMapper {
    /// Creates new virtual IDs and maps them to the provided physical IDs.
    ///
    /// Returns the newly created virtual IDs in the same order as the provided physical IDs.
    fn map_resources(&mut self, physical_ids: &[ResourceId]) -> Vec<ResourceId> {
        let virtual_ids = self
            .virtual_id_pool
            .acquire_many(physical_ids.len())
            .unwrap();

        // Pre-allocate to prevent multiple rehashes when inserting new entries.
        self.virtual_to_physical.reserve(physical_ids.len());

        for (&virtual_id, &physical_id) in virtual_ids.iter().zip(physical_ids.iter()) {
            self.virtual_to_physical.insert(virtual_id, physical_id);
        }

        virtual_ids
    }

    /// Removes the mappings for the given virtual IDs and releases them back to the pool.
    fn unmap_resources(&mut self, virtual_ids: &[ResourceId]) {
        for &virtual_id in virtual_ids {
            self.virtual_to_physical.remove(&virtual_id);
        }
        self.virtual_id_pool.release_many(virtual_ids).unwrap();
    }

    /// Translates a single virtual ID to its corresponding physical ID.
    fn translate(&self, virtual_id: ResourceId) -> Option<ResourceId> {
        self.virtual_to_physical.get(&virtual_id).copied()
    }
}

pub struct InstanceState {
    // Wasm states
    id: InstanceId,
    arguments: Vec<String>,
    pub(crate) return_value: Option<String>,

    // WASI states
    wasi_ctx: WasiCtx,
    resource_table: ResourceTable,
    http_ctx: WasiHttpCtx,
    // virtual resources
    resources: HashMap<(usize, ResourceTypeId), ResourceIdMapper>,
}

impl WasiView for InstanceState {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.wasi_ctx,
            table: &mut self.resource_table,
        }
    }
}

impl WasiHttpView for InstanceState {
    fn ctx(&mut self) -> &mut WasiHttpCtx {
        &mut self.http_ctx
    }

    fn table(&mut self) -> &mut ResourceTable {
        &mut self.resource_table
    }
}

impl InstanceState {
    pub async fn new(id: InstanceId, arguments: Vec<String>) -> (Self, OutputDeliveryCtrl) {
        let mut builder = WasiCtx::builder();
        builder.inherit_network(); // TODO: Replace with socket_addr_check later.

        // Create LogStream instances and keep handles for delivery mode control
        let stdout_stream = LogStream::new(OutputChannel::Stdout, id);
        let stderr_stream = LogStream::new(OutputChannel::Stderr, id);

        // Clone the streams for the WASI context (LogStream is cheap to clone due to Arc)
        builder.stdout(stdout_stream.clone());
        builder.stderr(stderr_stream.clone());

        let streaming_ctrl = OutputDeliveryCtrl {
            stdout_stream,
            stderr_stream,
        };

        let state = InstanceState {
            id,
            arguments,
            return_value: None,
            wasi_ctx: builder.build(),
            resource_table: ResourceTable::new(),
            http_ctx: WasiHttpCtx::new(),
            resources: HashMap::new(),
        };

        (state, streaming_ctrl)
    }

    pub fn id(&self) -> InstanceId {
        self.id
    }

    pub fn arguments(&self) -> &[String] {
        &self.arguments
    }

    pub fn return_value(&self) -> Option<String> {
        self.return_value.clone()
    }

    pub fn read_queue(&self, queue: &Resource<Queue>) -> Result<(usize, u32, u32)> {
        let q = self.resource_table.get(&queue)?;
        Ok((q.service_id, q.uid, q.priority))
    }
    pub fn map_resources(
        &mut self,
        service_id: usize,
        resource_type: ResourceTypeId,
        phys_ids: &[ResourceId],
    ) -> Vec<ResourceId> {
        self.resources
            .entry((service_id, resource_type))
            .or_default()
            .map_resources(phys_ids)
    }

    pub fn unmap_resources(
        &mut self,
        service_id: usize,
        resource_type: ResourceTypeId,
        virt_ids: &[ResourceId],
    ) {
        let m = self.resources.get_mut(&(service_id, resource_type));
        if let Some(m) = m {
            m.unmap_resources(virt_ids);
        }
    }

    pub fn translate_resource_ptr(
        &self,
        service_id: usize,
        resource_type: ResourceTypeId,
        virt_id: ResourceId,
    ) -> Result<ResourceId> {
        let m = self
            .resources
            .get(&(service_id, resource_type))
            .ok_or(format_err!(
                "Failed to find resource mapper for service_id: {:?}, resource_type: {:?}",
                service_id,
                resource_type
            ))?;
        let phys_id = m.translate(virt_id).ok_or(format_err!(
            "Failed to translate resource pointer: {:?} -> {:?}",
            virt_id,
            m.virtual_to_physical
        ))?;
        Ok(phys_id)
    }
}

#[derive(Clone, Debug)]
pub enum OutputChannel {
    Stdout,
    Stderr,
}

impl OutputChannel {
    /// Send the output to the server so that it can be delivered to the client
    fn dispatch_output(&self, content: String, instance_id: InstanceId) {
        match self {
            OutputChannel::Stdout => {
                InstanceEvent::StreamingOutput {
                    inst_id: instance_id,
                    output_type: OutputChannel::Stdout,
                    content,
                }
                .dispatch()
                .unwrap();
            }
            OutputChannel::Stderr => {
                InstanceEvent::StreamingOutput {
                    inst_id: instance_id,
                    output_type: OutputChannel::Stderr,
                    content,
                }
                .dispatch()
                .unwrap();
            }
        }
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

#[derive(Clone)]
struct LogStream {
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

    fn new(channel: OutputChannel, instance_id: InstanceId) -> LogStream {
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
