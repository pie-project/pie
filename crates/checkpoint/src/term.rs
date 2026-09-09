use dtype::Dtype;
use dtype::{BIASES, Elem, Fmt, Group, Off, SCALES, spells};
use ztensor::{Group as ZGroup, Leaf, Offset, Plane, Term};

use crate::error::Error;
use crate::types::{QuantScheme, QuantSpec};

const F16: Fmt<'static> = Fmt::Elem(Elem::F16);
const F32: Fmt<'static> = Fmt::Elem(Elem::F32);

const G32_I4_F16_N: Fmt<'static> = Fmt::Q {
    g: Group::N(32),
    elem: Elem::I(4),
    gain: &F16,
    offset: None,
};
const _: () = assert!(spells(&G32_I4_F16_N, "g32_i4_f16_n"));

const G32_U4_F16_B_F16: Fmt<'static> = Fmt::Q {
    g: Group::N(32),
    elem: Elem::U(4),
    gain: &F16,
    offset: Some(Off::Post(&F16)),
};
const _: () = assert!(spells(&G32_U4_F16_B_F16, "g32_u4_f16_b_f16"));

const G32_I5_F16_N: Fmt<'static> = Fmt::Q {
    g: Group::N(32),
    elem: Elem::I(5),
    gain: &F16,
    offset: None,
};
const _: () = assert!(spells(&G32_I5_F16_N, "g32_i5_f16_n"));

const G32_U5_F16_B_F16: Fmt<'static> = Fmt::Q {
    g: Group::N(32),
    elem: Elem::U(5),
    gain: &F16,
    offset: Some(Off::Post(&F16)),
};
const _: () = assert!(spells(&G32_U5_F16_B_F16, "g32_u5_f16_b_f16"));

const G32_I8_F16_N: Fmt<'static> = Fmt::Q {
    g: Group::N(32),
    elem: Elem::I(8),
    gain: &F16,
    offset: None,
};
const _: () = assert!(spells(&G32_I8_F16_N, "g32_i8_f16_n"));

const GR_E5M2_F32_N: Fmt<'static> = Fmt::Q {
    g: Group::Row,
    elem: Elem::E { e: 5, m: 2 },
    gain: &F32,
    offset: None,
};
const _: () = assert!(spells(&GR_E5M2_F32_N, "gr_e5m2_f32_n"));

const GR_I8_F32_N: Fmt<'static> = Fmt::Q {
    g: Group::Row,
    elem: Elem::I(8),
    gain: &F32,
    offset: None,
};
const _: () = assert!(spells(&GR_I8_F32_N, "gr_i8_f32_n"));

fn post_affine_bf16(group: u32, elem: Elem) -> Fmt<'static> {
    const BF16: Fmt<'static> = Fmt::Elem(Elem::Bf16);
    Fmt::Q {
        g: Group::N(group),
        elem,
        gain: &BF16,
        offset: Some(Off::Post(&BF16)),
    }
}

fn pre_affine_u4(group: u32) -> Fmt<'static> {
    const U4: Fmt<'static> = Fmt::Elem(Elem::U(4));
    Fmt::Q {
        g: Group::N(group),
        elem: Elem::U(4),
        gain: &F16,
        offset: Some(Off::Pre(&U4)),
    }
}

impl QuantSpec {
    #[must_use]
    pub fn term(&self) -> Option<Fmt<'static>> {
        let bits = self.normalized_bits();
        let group = self.normalized_group_size();
        Some(match self.scheme {
            QuantScheme::None => return None,

            QuantScheme::Fp8E4M3 => *Dtype::E4m3row.repr(),
            QuantScheme::Fp8E5M2 => GR_E5M2_F32_N,
            QuantScheme::Int8Symmetric => GR_I8_F32_N,

            QuantScheme::Int8Asymmetric => return None,

            QuantScheme::AwqInt4 | QuantScheme::GptqInt4 => {
                if bits != 4 {
                    return None;
                }
                pre_affine_u4(group)
            }

            QuantScheme::MlxAffineU4 => match bits {
                2 | 4 | 8 => post_affine_bf16(group, Elem::U(bits)),
                _ => return None,
            },

            QuantScheme::Int4B8 => {
                if bits != 4 {
                    return None;
                }
                Fmt::Q {
                    g: Group::N(group),
                    elem: Elem::I(4),
                    gain: &F16,
                    offset: None,
                }
            }

            QuantScheme::Mxfp4E2M1E8M0 => {
                const E8M0: Fmt<'static> = Fmt::Elem(Elem::E { e: 8, m: 0 });
                Fmt::Q {
                    g: Group::N(group),
                    elem: Elem::E { e: 2, m: 1 },
                    gain: &E8M0,
                    offset: None,
                }
            }

            QuantScheme::GgufQ4_0 => G32_I4_F16_N,
            QuantScheme::GgufQ4_1 => G32_U4_F16_B_F16,
            QuantScheme::GgufQ5_0 => G32_I5_F16_N,
            QuantScheme::GgufQ5_1 => G32_U5_F16_B_F16,
            QuantScheme::GgufQ8_0 => G32_I8_F16_N,
            QuantScheme::GgufQ2K => *Dtype::U2g16k.repr(),
            QuantScheme::GgufQ3K => *Dtype::I3g16k.repr(),
            QuantScheme::GgufQ4K => *Dtype::U4g32k.repr(),
            QuantScheme::GgufQ5K => *Dtype::U5g32k.repr(),
            QuantScheme::GgufQ6K => *Dtype::I6g16k.repr(),

            QuantScheme::GgufMxfp4 => *Dtype::Mxfp4.repr(),

            QuantScheme::GgufIq4Nl | QuantScheme::GgufIq4Xs => return None,

            QuantScheme::GgufIq2Xxs
            | QuantScheme::GgufIq2Xs
            | QuantScheme::GgufIq2S
            | QuantScheme::GgufIq3Xxs
            | QuantScheme::GgufIq3S => return None,
        })
    }

    #[must_use]
    pub fn is_mxfp4(&self) -> bool {
        !self.scheme.is_self_contained()
            && matches!(self.term(), Some(t) if t == *Dtype::Mxfp4.repr())
    }

    #[must_use]
    pub fn affine_point(&self) -> Option<(u32, u32)> {
        if self.scheme.is_self_contained() {
            return None;
        }
        match self.term()? {
            Fmt::Q {
                g: Group::N(n),
                elem: Elem::U(b) | Elem::I(b),
                ..
            } => Some((n, u32::from(b))),
            _ => None,
        }
    }
}

const GGUF: &[(QuantScheme, &str)] = &[
    (QuantScheme::GgufQ4_0, "q4_0"),
    (QuantScheme::GgufQ4_1, "q4_1"),
    (QuantScheme::GgufQ5_0, "q5_0"),
    (QuantScheme::GgufQ5_1, "q5_1"),
    (QuantScheme::GgufQ8_0, "q8_0"),
    (QuantScheme::GgufQ2K, "q2_k"),
    (QuantScheme::GgufQ3K, "q3_k"),
    (QuantScheme::GgufQ4K, "q4_k"),
    (QuantScheme::GgufQ5K, "q5_k"),
    (QuantScheme::GgufQ6K, "q6_k"),
    (QuantScheme::GgufMxfp4, "mxfp4"),
    (QuantScheme::GgufIq4Nl, "iq4_nl"),
    (QuantScheme::GgufIq4Xs, "iq4_xs"),
    (QuantScheme::GgufIq2Xxs, "iq2_xxs"),
    (QuantScheme::GgufIq2Xs, "iq2_xs"),
    (QuantScheme::GgufIq2S, "iq2_s"),
    (QuantScheme::GgufIq3Xxs, "iq3_xxs"),
    (QuantScheme::GgufIq3S, "iq3_s"),
];

#[must_use]
pub fn spec_of_term(f: &Fmt<'_>) -> Option<QuantSpec> {
    GGUF.iter().find_map(|&(scheme, _)| {
        let (elems, _) = scheme.block_layout()?;
        let spec = QuantSpec {
            scheme,
            logical_dtype: Dtype::Bf16,
            bits_per_element: scheme.default_bits(),
            group_size: u32::try_from(elems).ok()?,
            channel_axis: None,
        };
        (spec.term() == Some(*f)).then_some(spec)
    })
}

pub const MMA_TILED: &str = "pie.mma_tiled/1";

#[must_use]
pub fn gguf_type_of(layout: &str) -> Option<&str> {
    layout
        .strip_prefix("gguf.")
        .and_then(|rest| rest.strip_suffix("/2"))
}

#[must_use]
pub fn gguf_scheme(name: &str) -> Option<QuantScheme> {
    GGUF.iter()
        .find(|(_, it)| *it == name)
        .map(|&(scheme, _)| scheme)
}

#[must_use]
pub fn gguf_name(scheme: QuantScheme) -> Option<&'static str> {
    GGUF.iter()
        .find(|(it, _)| *it == scheme)
        .map(|&(_, name)| name)
}

#[must_use]
pub fn spec_of_canonical(term: &Term) -> Option<QuantSpec> {
    let Term::Group {
        group,
        code,
        gain,
        offset,
    } = term
    else {
        return None;
    };
    let gain_leaf = gain.leaf()?;
    let offset_leaf = match offset {
        Offset::None => None,
        Offset::Post(t) | Offset::Pre(t) => Some(t.leaf()?),
    };
    let spec = |scheme, bits: u8, group: u32| QuantSpec {
        scheme,
        logical_dtype: Dtype::Bf16,
        bits_per_element: bits,
        group_size: group,
        channel_axis: None,
    };
    let n = |g: &ZGroup| match g {
        ZGroup::N(n) => u32::try_from(*n).ok(),
        _ => None,
    };
    Some(match (group, code, gain_leaf, offset) {
        (g, Leaf::U(b @ (2 | 4 | 8)), Leaf::BF16, Offset::Post(_))
            if offset_leaf == Some(Leaf::BF16) =>
        {
            spec(QuantScheme::MlxAffineU4, *b, n(g)?)
        }
        (g, Leaf::E2M1, Leaf::E8M0, Offset::None) => spec(QuantScheme::Mxfp4E2M1E8M0, 4, n(g)?),
        (g, Leaf::I(4), Leaf::F16, Offset::None) => spec(QuantScheme::Int4B8, 4, n(g)?),
        (ZGroup::Row, Leaf::E4M3, Leaf::F32, Offset::None) => spec(QuantScheme::Fp8E4M3, 8, 1),
        (ZGroup::Row, Leaf::E5M2, Leaf::F32, Offset::None) => spec(QuantScheme::Fp8E5M2, 8, 1),
        (ZGroup::Row, Leaf::I(8), Leaf::F32, Offset::None) => {
            spec(QuantScheme::Int8Symmetric, 8, 1)
        }
        _ => return None,
    })
}

#[must_use]
pub fn term_of(encoding: &crate::types::Encoding) -> Option<Term> {
    match encoding {
        crate::types::Encoding::Raw(dtype) => Term::parse(dtype.term()).ok(),
        crate::types::Encoding::Quant(spec) => {
            let term = Term::parse(spec.term()?.mangle().as_str()).ok()?;
            let read_back =
                || spec_of_canonical(&term).is_some_and(|read| read.scheme == spec.scheme);
            (spec.scheme.is_self_contained() || read_back()).then_some(term)
        }
    }
}

pub fn dtype_of_leaf(leaf: Leaf) -> Option<Dtype> {
    Some(match leaf {
        Leaf::F32 => Dtype::F32,
        Leaf::F16 => Dtype::F16,
        Leaf::BF16 => Dtype::Bf16,
        Leaf::E4M3 => Dtype::E4m3,
        Leaf::E5M2 => Dtype::E5m2,
        Leaf::E2M1 => Dtype::E2m1,
        Leaf::E8M0 => Dtype::E8m0,
        Leaf::Bool => Dtype::Bool,
        Leaf::I(64) => Dtype::I64,
        Leaf::I(32) => Dtype::I32,
        Leaf::I(16) => Dtype::I16,
        Leaf::I(8) => Dtype::I8,
        Leaf::U(64) => Dtype::U64,
        Leaf::U(32) => Dtype::U32,
        Leaf::U(16) => Dtype::U16,
        Leaf::U(8) => Dtype::U8,
        _ => return None,
    })
}

pub fn blob_planes(
    name: &str,
    layout: Option<&str>,
    term: Option<&Term>,
    shape: &[u64],
    nbytes: u64,
) -> Result<Vec<Plane>, Error> {
    match (layout, term) {
        (None | Some(MMA_TILED), Some(term)) => term.planes(shape).map_err(Error::from),
        (Some(_), _) => Ok(vec![Plane {
            path: "data".into(),
            leaf: Leaf::U8,
            shape: vec![nbytes],
            offset: 0,
            len: nbytes,
        }]),
        (None, None) => Err(Error::Checkpoint(format!("{name}: no type and no layout"))),
    }
}

#[must_use]
pub fn plane_name(object: &str, path: &str) -> String {
    match path {
        "data" | "code" => object.to_string(),
        "gain" => format!("{object}{SCALES}"),
        "offset" => format!("{object}{BIASES}"),
        other => format!("{object}.{other}"),
    }
}

#[must_use]
pub fn plane_of(name: &str) -> Vec<(String, &'static str)> {
    let mut out = vec![(name.to_string(), "code")];
    if let Some(object) = name.strip_suffix(SCALES) {
        out.push((object.to_string(), "gain"));
    }
    if let Some(object) = name.strip_suffix(BIASES) {
        out.push((object.to_string(), "offset"));
    }
    out
}
