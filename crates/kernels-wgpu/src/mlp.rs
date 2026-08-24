use crate::plane::{Bind, Ctx, Fire, In, Out, elementwise};
use crate::points::{Payload, at_bf16};
use kernels::plane::Refusal;

#[kernels_macros::claims]
impl kernels::points::Mlp for Ctx<'_> {
    fn geglu_tanh<T: kernels::points::Scalar>(
        &self,
        gate: In<Payload<T>>,
        up: In<Payload<T>>,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("mlp.geglu_tanh at an element other than bf16")?;
        self.fire(
            Fire::at("mlp/gated.wgsl", "geglu_tanh_bfloat16")
                .apply(elementwise(gate.width, gate.rows)?),
            &[gate.arg(), up.arg(), y.arg()],
        )
    }
}
