//! `Dist`: collectives on the metal plane — there are none. Metal serves a
//! single device, so every variant answers [`KernelError::Unsupported`]; the
//! entries exist so the driver arm stays destructure → resolve → call, and
//! the refusal stays typed (the old plane claimed nothing here either).

use new_kernels::KernelError;

use crate::encode::Ctx;
use crate::tensor::Tensor;

pub fn all_reduce(_ctx: &Ctx<'_>, _buf: Tensor) -> Result<(), KernelError> {
    Err(KernelError::Unsupported {
        op: "dist.all_reduce",
    })
}

pub fn all_gather(_ctx: &Ctx<'_>, _x: Tensor, _y: Tensor) -> Result<(), KernelError> {
    Err(KernelError::Unsupported {
        op: "dist.all_gather",
    })
}

pub fn reduce_scatter(_ctx: &Ctx<'_>, _x: Tensor, _y: Tensor) -> Result<(), KernelError> {
    Err(KernelError::Unsupported {
        op: "dist.reduce_scatter",
    })
}
