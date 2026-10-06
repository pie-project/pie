pub mod cast;
pub mod e8m0;
pub mod fp8;
pub mod int4;
pub mod mlx;
pub mod mxfp4;
pub mod ptq1_0;
pub mod rows;

use crate::error::Error;

fn invalid(message: impl Into<String>) -> Error {
    Error::Contract(message.into())
}
