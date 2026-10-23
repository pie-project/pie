//! Ports fed from device cells (`runtime.port_cast`) from
//! `kernels/runtime/port.slang`: an f32 cell cast into its bf16 seat of the
//! input store, a seat that may start or end mid-word.

use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const PORT: &str = "runtime/port.slang";

/// The workgroup width every instantiation is built at.
const GROUP: u32 = 256;

/// `store[at + i] = bf16(x[i])` for each of `x`'s values, `store` the whole
/// input store as bf16 elements; the store's other elements are left alone,
/// the halves of the seat's edge words included.
pub fn cast(ctx: &Ctx<'_>, x: Tensor, store: Tensor, at: u32) -> Result<(), Error> {
    const OP: &str = "runtime.port_cast";
    if x.dtype != Dtype::F32 || store.dtype != Dtype::Bf16 {
        return Err(refuse(
            OP,
            format!(
                "a {:?} cell into a {:?} store; a cast port lands f32 values as bf16",
                x.dtype, store.dtype
            ),
        ));
    }
    let n = u64::from(x.rows) * u64::from(x.width);
    if n == 0 {
        return Ok(());
    }
    let held = u64::from(store.rows) * u64::from(store.width);
    let end = u64::from(at) + n;
    if end > held {
        return Err(refuse(
            OP,
            format!("a seat of {n} values at {at} runs past the store's {held}"),
        ));
    }
    let end =
        u32::try_from(end).map_err(|_| refuse(OP, format!("a seat ending at element {end}")))?;
    let words = end.div_ceil(2) - at / 2;
    ctx.fire(
        Fire::at(PORT, "runtime_port_cast_bf16").threads([words, 1, 1], [GROUP, 1, 1]),
        &[x.arg(), store.arg_mut(), at.arg(), (end - at).arg()],
    )
}
