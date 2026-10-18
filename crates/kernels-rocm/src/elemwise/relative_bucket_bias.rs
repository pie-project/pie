use crate::tensor::Tensor;

crate::error::unsupported! {
    relative_bucket_bias = "elementwise.relative_bucket_bias" (embedding: Tensor, max_len: u32,
        num_buckets: u32, max_distance: f32, bidirectional: bool, y: &mut Tensor);
}
