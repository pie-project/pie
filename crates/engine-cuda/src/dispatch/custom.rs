//! `CustomCuda`: the cuda-plane escape hatch's fused points.

use kernels::{DispatchCustomCuda, KernelError};
use kernels_cuda::custom;
use model_ir::CustomCuda;

use crate::run::Run;

impl DispatchCustomCuda for Run<'_> {
    /// The cuda-plane fused family on its home `Run` — this is the shell
    /// the trace emitted it for, so the arm dispatches the real entry. The
    /// write side lands by the op's own `write_page`/`write_offset`
    /// descriptors; `positions` stays the rope input.
    fn dispatch(&mut self, op: &CustomCuda) -> Result<(), KernelError> {
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
