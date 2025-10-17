use crate::api::core::{Blob, BlobResult, Queue};
use crate::api::inferlet;
use crate::instance::InstanceState;
use crate::model::request::{DownloadAdapterRequest, Request, UploadAdapterRequest};
use crate::model::resource::{ADAPTER_TYPE_ID, ResourceId};
use crate::model::submit_request;
use anyhow::Result;
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;
use crate::api::core::forward::ForwardPass;

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
    ) -> Result<Resource<BlobResult>> {
        let (svc_id, queue_id, priority) = self.read_queue(&queue)?;

        adapter_ptr = self.translate_resource_ptr(svc_id, ADAPTER_TYPE_ID, adapter_ptr)?;

        let (tx, rx) = oneshot::channel();

        let req = Request::DownloadAdapter(DownloadAdapterRequest { adapter_ptr, name }, tx);

        submit_request(svc_id, queue_id, priority, req)?;

        let res = BlobResult {
            receiver: rx,
            result: None,
            done: false,
        };

        Ok(self.ctx().table.push(res)?)
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
