use crate::plane::Ctx;

#[kernels_macros::claims]
impl kernels::points::Gemm for Ctx<'_> {}
