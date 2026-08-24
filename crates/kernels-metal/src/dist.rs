use crate::plane::Ctx;

#[kernels_macros::claims]
impl kernels::points::Dist for Ctx<'_> {}
