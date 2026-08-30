//! `Collective`: collectives on the metal plane — there are none. Metal serves a
//! single device, so every variant answers [`Error::Unsupported`]; the
//! entries exist so the driver arm stays destructure → resolve → call, and
//! the refusal stays typed (the old plane claimed nothing here either).

use crate::error::Error;

use crate::encode::Ctx;
use crate::tensor::Tensor;

pub fn all_reduce(_ctx: &Ctx<'_>, _buf: Tensor) -> Result<(), Error> {
    Err(Error::Unsupported {
        op: "collective.all_reduce",
    })
}

pub fn all_gather(_ctx: &Ctx<'_>, _x: Tensor, _y: Tensor) -> Result<(), Error> {
    Err(Error::Unsupported {
        op: "collective.all_gather",
    })
}

pub fn reduce_scatter(_ctx: &Ctx<'_>, _x: Tensor, _y: Tensor) -> Result<(), Error> {
    Err(Error::Unsupported {
        op: "collective.reduce_scatter",
    })
}
