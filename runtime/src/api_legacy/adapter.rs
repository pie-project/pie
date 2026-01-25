use crate::api_legacy::core::forward::ForwardPass;
use crate::api_legacy::core::{Blob, Queue};
use crate::api_legacy::inferlet;
use crate::instance::InstanceState;
use crate::model::request::{DownloadAdapterRequest, Request, UploadAdapterRequest};
use crate::model::resource::{ADAPTER_TYPE_ID, ResourceId};
use crate::model::submit_request;
use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

impl inferlet::adapter::common::Host for InstanceState {
    async fn set_adapter(
        &mut self,
        pass: Resource<ForwardPass>,
        mut adapter_ptr: ResourceId,
    ) -> Result<()> {
        let svc_id = self.ctx().table.get(&pass)?.queue.service_id;
        adapter_ptr = self.translate_resource_ptr(svc_id, ADAPTER_TYPE_ID, adapter_ptr)?;

        let pass = self.ctx().table.get_mut(&pass)?;
        pass.adapter = Some(adapter_ptr);

        Ok(())
    }

    async fn download_adapter(
        &mut self,
        queue: Resource<Queue>,
        mut adapter_ptr: ResourceId,
        name: String,
    ) -> Result<Resource<crate::api_legacy::core::BlobResult>> {
        let (svc_id, queue_id, priority) = self.read_queue(&queue)?;

        adapter_ptr = self.translate_resource_ptr(svc_id, ADAPTER_TYPE_ID, adapter_ptr)?;

        let (tx, rx) = tokio::sync::oneshot::channel();
        let req = Request::DownloadAdapter(DownloadAdapterRequest { adapter_ptr, name });

        submit_request(svc_id, queue_id, priority, req)?;

        // TODO: The actual download should send the blob data through tx
        // For now, create a BlobResult that will receive the data
        let blob_result = crate::api_legacy::core::BlobResult {
            receiver: rx,
            result: None,
            done: false,
        };
        Ok(self.ctx().table.push(blob_result)?)
    }

    async fn upload_adapter(
        &mut self,
        queue: Resource<Queue>,
        mut adapter_ptr: ResourceId,
        name: String,
        blob: Resource<Blob>,
    ) -> Result<()> {
        let (svc_id, queue_id, priority) = self.read_queue(&queue)?;

        adapter_ptr = self.translate_resource_ptr(svc_id, ADAPTER_TYPE_ID, adapter_ptr)?;
        let blob = self.ctx().table.get(&blob)?;
        let req = Request::UploadAdapter(UploadAdapterRequest {
            adapter_ptr,
            name,
            adapter_data: blob.data.to_vec(),
        });

        submit_request(svc_id, queue_id, priority, req)?;

        Ok(())
    }
}
