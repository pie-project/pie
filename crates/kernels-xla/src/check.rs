//! The checks a kernel makes of what it is handed: the element type a plain
//! dtype stores as, a dtype among those a kernel takes, a shape.

use dtype::Dtype;

use crate::error::{Error, refuse};
use crate::hlo::Elem;
use crate::tensor::Stored;

/// The element type a plain dtype stores as.
pub fn elem_of(op: &'static str, dtype: Dtype) -> Result<Elem, Error> {
    Ok(match dtype {
        Dtype::F32 => Elem::F32,
        Dtype::F16 => Elem::F16,
        Dtype::Bf16 => Elem::Bf16,
        Dtype::E4m3 => Elem::F8E4m3fn,
        Dtype::E5m2 => Elem::F8E5m2,
        Dtype::E8m0 => Elem::F8E8m0fnu,
        Dtype::I64 => Elem::I64,
        Dtype::I32 => Elem::I32,
        Dtype::I16 => Elem::I16,
        Dtype::I8 => Elem::I8,
        Dtype::U64 => Elem::U64,
        Dtype::U32 => Elem::U32,
        Dtype::U16 => Elem::U16,
        Dtype::U8 | Dtype::Bool => Elem::U8,
        other => {
            return Err(refuse(
                op,
                format!("{other:?} is a packed format; it is read through its bank"),
            ));
        }
    })
}

/// Refuses unless `t` is one of `dtypes`.
pub fn expect(op: &'static str, t: impl Stored, dtypes: &[Dtype]) -> Result<(), Error> {
    if dtypes.contains(&t.dtype()) {
        Ok(())
    } else {
        Err(Error::DtypeUnsupported {
            op,
            dtype: t.dtype(),
        })
    }
}

/// Refuses unless `t` is `[rows, width]`.
pub fn shaped(
    op: &'static str,
    what: &str,
    t: impl Stored,
    rows: u32,
    width: u32,
) -> Result<(), Error> {
    if t.rows() == rows && t.width() == width {
        Ok(())
    } else {
        Err(refuse(
            op,
            format!(
                "the {what} is {}x{}, and this op wants {rows}x{width}",
                t.rows(),
                t.width()
            ),
        ))
    }
}
