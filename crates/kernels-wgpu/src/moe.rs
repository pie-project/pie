use crate::plane::{Bind, Ctx, Fire, In, Out};
use crate::points::{Payload, at_bf16};
use kernels::plane::Refusal;

#[kernels_macros::claims]
impl kernels::points::Moe for Ctx<'_> {
    fn sigmoid_gate_add<T: kernels::points::Scalar>(
        &self,
        routed: In<Payload<T>>,
        shared: In<Payload<T>>,
        gate: In<Payload<T>>,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("moe.sigmoid_gate_add at an element other than bf16")?;
        let width = routed.width;
        self.fire(
            Fire::at("moe/route.wgsl", "shared_expert_combine")
                .apply(rows_by_width(width, routed.rows)?),
            &[routed.arg(), shared.arg(), gate.arg(), y.arg(), width.arg()],
        )
    }
}

fn rows_by_width(width: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if width <= 0 {
        return Err(Refusal::Empty { what: "width" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    Ok([width.unsigned_abs(), rows.unsigned_abs(), 1])
}
