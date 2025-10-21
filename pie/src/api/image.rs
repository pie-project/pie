use crate::api::core::Queue;
use crate::api::inferlet;
use crate::instance::InstanceState;
use crate::model::request::{EmbedImageRequest, Request};
use crate::model::resource::{EMBED_TYPE_ID, ResourceId};
use crate::model::submit_request;
use anyhow::Result;
use wasmtime::component::Resource;

impl inferlet::image::image::Host for InstanceState {
    async fn embed_image(
        &mut self,
        queue: Resource<Queue>,
        mut emb_ptrs: Vec<ResourceId>,
        image_blob: Vec<u8>,
        position_offset: u32,
    ) -> Result<()> {
        let (svc_id, queue_id, priority) = self.read_queue(&queue)?;
        emb_ptrs.iter_mut().try_for_each(|emb_ptr| {
            *emb_ptr = self.translate_resource_ptr(svc_id, EMBED_TYPE_ID, *emb_ptr)?;
            Ok::<_, anyhow::Error>(())
        })?;

        let req = Request::EmbedImage(EmbedImageRequest {
            embed_ptrs: emb_ptrs,
            image_blob,
            position_offset,
        });

        submit_request(svc_id, queue_id, priority, req)?;

        Ok(())
    }

    async fn calculate_embed_size(
        &mut self,
        _queue: Resource<Queue>,
        _image_width: u32,
        _image_height: u32,
    ) -> Result<u32> {
        // Placeholder implementation
        // This would typically involve a call to the model service
        // to determine the number of tokens/embeddings for a given image size.
        Ok(1024) // e.g., return a fixed size for now
    }
}
