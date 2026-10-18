use crate::error::Error;
use crate::jit::Ctx;
use crate::tensor::Tensor;

/// Columns side by side (DSpark's taps), which are CUDA's.
pub fn concat_cols(_ctx: &Ctx, _xs: &[Tensor], _y: &mut Tensor) -> Result<(), Error> {
    Err(Error::Unsupported {
        op: "layout.concat_cols",
    })
}
