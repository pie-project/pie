use crate::plane::Refusal;
use crate::shader::ShaderValue;
use crate::{Kind, Lit, Source, Ty};

pub trait Holds {
    fn input(&mut self, n: usize) -> Result<u32, Refusal>;

    fn output(&mut self, n: usize) -> Result<u32, Refusal>;

    fn output_read(&mut self, n: usize) -> Result<u32, Refusal> {
        self.output(n)
    }

    fn weight(&mut self, n: usize) -> Result<u32, Refusal>;

    fn param(&self, n: usize) -> Result<i32, Refusal>;

    fn param_f32(&self, n: usize) -> Result<f32, Refusal>;

    fn null(&mut self) -> u32;

    fn in_width(&self, n: usize) -> Result<i32, Refusal> {
        let _ = n;
        Err(Refusal::Unstated {
            what: "an input's row width, which this backend does not carry",
        })
    }

    fn out_width(&self, n: usize) -> Result<i32, Refusal> {
        let _ = n;
        Err(Refusal::Unstated {
            what: "a result's row width, which this backend does not carry",
        })
    }

    fn out_elements(&self, n: usize) -> Result<i32, Refusal> {
        let _ = n;
        Err(Refusal::Unstated {
            what: "a result's element count, which this backend does not carry",
        })
    }

    fn rows(&mut self) -> i32 {
        0
    }
}

pub fn bind<V: ShaderValue, H: Holds + ?Sized>(
    args: &[Ty],
    sources: &[Option<Source>],
    h: &mut H,
) -> Result<Vec<V>, Refusal> {
    let mut out = Vec::with_capacity(args.len());
    for (at, ty) in args.iter().enumerate() {
        let source = sources
            .get(at)
            .copied()
            .flatten()
            .ok_or(Refusal::Unstated {
                what: "an argument whose signature does not say where it comes from",
            })?;
        out.push(one(*ty, source, h)?);
    }
    Ok(out)
}

pub fn one<V: ShaderValue, H: Holds + ?Sized>(
    ty: Ty,
    source: Source,
    h: &mut H,
) -> Result<V, Refusal> {
    match source {
        Source::Slot(Kind::In, n) => {
            let at = h.input(n.into())?;
            shaped(ty, at, rows(h), h.in_width(n.into()).unwrap_or(0))
        }
        Source::Slot(Kind::Out, n) => {
            let at = if matches!(ty, Ty::Buf | Ty::I32s | Ty::U32s | Ty::F32s) {
                h.output_read(n.into())?
            } else {
                h.output(n.into())?
            };
            shaped(ty, at, rows(h), h.out_width(n.into()).unwrap_or(0))
        }

        Source::Slot(Kind::Weight, n) => handle(ty, h.weight(n.into())?),

        Source::Slot(Kind::Param, n) if matches!(ty, Ty::F32) => Ok(V::f32(h.param_f32(n.into())?)),
        Source::Slot(Kind::Param, n) => number(ty, h.param(n.into())?),
        Source::Slot(Kind::ParamF32, n) => Ok(V::f32(h.param_f32(n.into())?)),

        Source::Slot(Kind::InWidth, n) => number(ty, h.in_width(n.into())?),
        Source::Slot(Kind::OutWidth, n) => number(ty, h.out_width(n.into())?),
        Source::Slot(Kind::OutElements, n) => number(ty, h.out_elements(n.into())?),
        Source::Slot(kind, _) => Err(unstated(kind)),

        Source::Alias(n, _) => {
            let at = h.input(n.into())?;
            shaped(ty, at, rows(h), h.in_width(n.into()).unwrap_or(0))
        }

        Source::Or(&Source::Slot(Kind::Param, n), fallback) => {
            match h.param(n.into()).ok().filter(|n| *n > 0) {
                Some(stated) => number(ty, stated),
                None => one(ty, *fallback, h),
            }
        }
        Source::Or(..) => Err(Refusal::Unstated {
            what: "a chain whose first half is neither one of the statement's \
                   scalars nor a fact the driver answers by name",
        }),

        Source::Times(a, b) => number(
            ty,
            count::<V, H>(*a, h)?.saturating_mul(count::<V, H>(*b, h)?),
        ),
        Source::Over(a, b) => {
            let d = count::<V, H>(*b, h)?;
            if d == 0 {
                return Err(Refusal::Empty { what: "a divisor" });
            }
            number(ty, count::<V, H>(*a, h)? / d)
        }

        Source::Lit(Lit::Null) => handle(ty, h.null()),
        Source::Lit(Lit::I32(n)) => number(ty, n),
        Source::Lit(_) => Err(Refusal::Unstated {
            what: "a literal argument at a carrier this binder has not met",
        }),
    }
}

fn count<V: ShaderValue, H: Holds + ?Sized>(source: Source, h: &mut H) -> Result<i32, Refusal> {
    one::<V, H>(Ty::I32, source, h)?
        .as_i32()
        .ok_or(Refusal::Unstated {
            what: "a side of an arithmetic source that is not a number",
        })
}

fn rows<H: Holds + ?Sized>(h: &mut H) -> i32 {
    h.rows()
}

fn shaped<V: ShaderValue>(ty: Ty, at: u32, rows: i32, width: i32) -> Result<V, Refusal> {
    let whole: V = handle(ty, at)?;
    Ok(match whole.as_buffer() {
        Some(_) if ty.binds() == crate::Binds::Writes => V::buffer_mut_at(at, rows, width),
        Some(_) => V::buffer_at(at, rows, width),
        None => whole,
    })
}

fn handle<V: ShaderValue>(ty: Ty, at: u32) -> Result<V, Refusal> {
    Ok(match ty {
        Ty::Buf
        | Ty::I32s
        | Ty::U32s
        | Ty::F32s
        | Ty::U8s
        | Ty::I64s
        | Ty::U16s
        | Ty::I8s
        | Ty::Bf16s
        | Ty::F16s
        | Ty::U8Array
        | Ty::I32Array
        | Ty::BufArray => V::buffer(at),
        Ty::BufMut
        | Ty::F32sMut
        | Ty::I32sMut
        | Ty::U32sMut
        | Ty::U8sMut
        | Ty::U16sMut
        | Ty::I8sMut
        | Ty::Bf16sMut
        | Ty::F16sMut
        | Ty::BufArrayMut
        | Ty::BufArrayOut
        | Ty::BufArrayOutMut => V::buffer_mut(at),
        _ => {
            return Err(Refusal::Unstated {
                what: "a buffer argument at a carrier this binder has not met",
            });
        }
    })
}

fn number<V: ShaderValue>(ty: Ty, n: i32) -> Result<V, Refusal> {
    Ok(match ty {
        Ty::I32 => V::i32(n),

        Ty::U32 | Ty::InPacked => V::u32(n.cast_unsigned()),
        Ty::Usize => V::usize(u64::from(n.cast_unsigned())),

        Ty::I64 => V::i64(i64::from(n)),
        Ty::Bool => V::bool(n != 0),
        _ => {
            return Err(Refusal::Unstated {
                what: "a scalar argument at a carrier this binder has not met",
            });
        }
    })
}

fn unstated(kind: Kind) -> Refusal {
    Refusal::Unstated {
        what: match kind {
            Kind::Aux => "an `Aux` slot, which this backend does not stage",
            _ => "an operand kind this binder has not met",
        },
    }
}
