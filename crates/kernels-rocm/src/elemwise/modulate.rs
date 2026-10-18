use crate::tensor::Tensor;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NormKind {
    LayerNormNoAffine { eps: f32 },
    RmsNormNoScale { eps: f32 },
    RmsNorm { weight: Tensor, eps: f32 },
}

crate::error::unsupported! {
    scale_shift = "elementwise.modulate" (x: Tensor, m: Tensor, lane_of_row: Option<Tensor>,
        o: &mut Tensor);
    scale = "elementwise.modulate" (x: Tensor, m: Tensor, lane_of_row: Option<Tensor>,
        o: &mut Tensor);
    tanh_gate = "elementwise.modulate" (x: Tensor, m: Tensor, lane_of_row: Option<Tensor>,
        o: &mut Tensor);
    gated_residual_add = "elementwise.gated_residual_add" (r: Tensor, g: Tensor, y: Tensor,
        lane_of_row: Option<Tensor>, r_out: &mut Tensor);
    norm_modulate = "elementwise.norm_modulate" (x: Tensor, m: Tensor,
        lane_of_row: Option<Tensor>, norm: NormKind, o: &mut Tensor);
    #[allow(clippy::too_many_arguments)]
    gated_residual_norm_modulate = "elementwise.gated_residual_norm_modulate" (r: Tensor,
        g: Tensor, y: Tensor, m: Tensor, lane_of_row: Option<Tensor>, norm: NormKind,
        r_out: &mut Tensor, o: &mut Tensor);
}
