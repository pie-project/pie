use crate::{Queue, api};

/// Provides the ability to embed images into model-compatible embeddings.
pub trait Image {
    /// Embeds an image blob into the provided embedding IDs.
    ///
    /// # Arguments
    ///
    /// * `embed_ids` - A slice of output object IDs where the image embeddings will be stored.
    /// * `image` - A slice of bytes representing the raw image data (e.g., JPEG or PNG).
    /// * `position_offset` - The positional offset in the embedding space.
    fn embed_image(&self, embed_ids: &[u32], image: &[u8], position_offset: u32);

    /// Calculates the number of embeddings required for an image of the given dimensions.
    ///
    /// This is useful for allocating the necessary space for the embeddings before calling `embed_image`.
    ///
    /// # Arguments
    ///
    /// * `image_width` - The width of the input image in pixels.
    /// * `image_height` - The height of the input image in pixels.
    ///
    /// # Returns
    ///
    /// The required number of embeddings for the image.
    fn calculate_embed_size(&self, image_width: u32, image_height: u32) -> u32;
}

impl Image for Queue {
    fn embed_image(&self, embed_ids: &[u32], image: &[u8], position_offset: u32) {
        api::image::image::embed_image(&self.inner, embed_ids, image, position_offset)
    }

    fn calculate_embed_size(&self, image_width: u32, image_height: u32) -> u32 {
        api::image::image::calculate_embed_size(&self.inner, image_width, image_height)
    }
}
