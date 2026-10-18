use crate::tensor::Tensor;

crate::error::unsupported! {
    sinusoid = "elementwise.sinusoid" (t: Tensor, dim: u32, max_period: f32,
        flip_sin_cos: bool, scale: f32, o: &mut Tensor);
}
