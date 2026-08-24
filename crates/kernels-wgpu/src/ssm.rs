use crate::plane::Ctx;

#[kernels_macros::claims]
impl kernels::points::Ssm for Ctx<'_> {}
