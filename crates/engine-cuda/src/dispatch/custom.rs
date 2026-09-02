//! `CustomCuda`: the cuda-plane escape hatch's fused points.

use kernels_cuda::custom;
use model_exec::{DispatchCustomCuda, KernelError};
use model_ir::CustomCuda;

use crate::run::Run;

impl DispatchCustomCuda for Run<'_> {
    /// Write side uses the op's own `write_page`/`write_offset`; `positions` is the rope input.
    fn dispatch(&mut self, op: &CustomCuda) -> Result<(), KernelError> {
        self.custom_cuda(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    /// Returns `kernels_cuda::Error`, lifted by [`kernel`](crate::error::kernel) in `dispatch` above.
    fn custom_cuda(&mut self, op: &CustomCuda) -> Result<(), kernels_cuda::Error> {
        match op {
            CustomCuda::QkvFusedQknormRopeVnormWrite {
                packed,
                positions,
                q_norm_weight,
                q_norm_eps,
                k_norm_weight,
                k_norm_eps,
                cache,
                write_page,
                write_offset,
                kv_heads,
                head_dim,
                theta,
                q,
            } => custom::qkv_fused_qknorm_rope_vnorm_write(
                self.ctx(),
                self.tensor(*packed),
                self.tensor(*positions),
                self.tensor(*q_norm_weight),
                *q_norm_eps,
                self.tensor(*k_norm_weight),
                *k_norm_eps,
                &self.pool(*cache),
                self.tensor(*write_page),
                self.tensor(*write_offset),
                *kv_heads,
                *head_dim,
                *theta,
                &mut self.tensor(*q),
            ),
        }
    }
}
