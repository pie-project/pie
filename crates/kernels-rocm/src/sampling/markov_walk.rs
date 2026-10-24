//! DSpark's Markov draft walk, which is CUDA's.

use crate::error::Error;
use crate::jit::Ctx;
use crate::tensor::Tensor;

#[allow(clippy::too_many_arguments)]
pub fn markov_walk(
    _ctx: &Ctx,
    _logits: Tensor,
    _tokens: Tensor,
    _positions: Tensor,
    _ends: Tensor,
    _embed: Tensor,
    _head: Tensor,
    _block: u32,
    _biased: &mut Tensor,
    _picks: &mut Tensor,
) -> Result<(), Error> {
    Err(Error::Unsupported {
        op: "sampling.markov_walk",
    })
}
