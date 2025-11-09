use super::api::core::Queue;
use super::utils;
use crate::model::resource::{ResourceId, ResourceTypeId};
use crate::server::InstanceEvent;
use anyhow::{Result, format_err};
use bytes::Bytes;
use std::collections::HashMap;
use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::io::AsyncWrite;
use uuid::Uuid;
use wasmtime::component::{Resource, ResourceTable};
use wasmtime_wasi::async_trait;
use wasmtime_wasi::cli::IsTerminal;
use wasmtime_wasi::cli::StdoutStream;
use wasmtime_wasi::p2::{OutputStream, Pollable, StreamResult};
use wasmtime_wasi::{WasiCtx, WasiCtxView, WasiView};
use wasmtime_wasi_http::{WasiHttpCtx, WasiHttpView};

pub type InstanceId = Uuid;

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

// Define your custom stdout type.

impl InstanceState {
    pub async fn new(id: InstanceId, arguments: Vec<String>) -> Self {
        let mut builder = WasiCtx::builder();
        builder.inherit_network(); // TODO: Replace with socket_addr_check later.

        builder.stdout(LogStream::new(OutputType::Stdout, id));
        builder.stderr(LogStream::new(OutputType::Stderr, id));

        InstanceState {
            id,
            arguments,
            return_value: None,
            wasi_ctx: builder.build(),
            resource_table: ResourceTable::new(),
            http_ctx: WasiHttpCtx::new(),
            resources: HashMap::new(),
        }
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
pub enum OutputType {
    Stdout,
    Stderr,
}

impl OutputType {
    fn write_all(&self, buf: &[u8], instance_id: InstanceId) {
        match self {
            OutputType::Stdout => {
                InstanceEvent::StreamingOutput {
                    inst_id: instance_id,
                    output_type: OutputType::Stdout,
                    content: String::from_utf8_lossy(buf).to_string(),
                }
                .dispatch()
                .unwrap();
            }
            OutputType::Stderr => {
                InstanceEvent::StreamingOutput {
                    inst_id: instance_id,
                    output_type: OutputType::Stderr,
                    content: String::from_utf8_lossy(buf).to_string(),
                }
                .dispatch()
                .unwrap();
            }
        }
    }
}

#[derive(Clone)]
struct LogStream {
    output: OutputType,
    state: Arc<LogStreamState>,
}

struct LogStreamState {
    instance_id: InstanceId,
}

impl LogStream {
    fn new(output: OutputType, instance_id: InstanceId) -> LogStream {
        LogStream {
            output,
            state: Arc::new(LogStreamState { instance_id }),
        }
    }

    fn write_all(&mut self, bytes: &[u8]) {
        self.output.write_all(bytes, self.state.instance_id);
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
        match &self.output {
            OutputType::Stdout => std::io::stdout().is_terminal(),
            OutputType::Stderr => std::io::stderr().is_terminal(),
        }
    }
}

impl OutputStream for LogStream {
    fn write(&mut self, bytes: Bytes) -> StreamResult<()> {
        self.write_all(&bytes);
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
    async fn ready(&mut self) {}
}

impl AsyncWrite for LogStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        self.write_all(buf);
        Poll::Ready(Ok(buf.len()))
    }
    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}
