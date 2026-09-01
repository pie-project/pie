//! The closed storage-type enum for the whole tree: elements and every
//! composite quantization format a kernel actually serves.
//! [`Dtype::repr`] expands each variant to its algebra, and every number
//! downstream derives from that; the wire form is the mangled spelling.

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

#[cfg(test)]
extern crate std;

mod repr;

pub use repr::{
    Elem, Fmt, G64_U4_F16_Z_F16, G128_U4_F16_Z_U4, GT_T3_F16_N, Group, MAX_PLANES, Mangled, Off,
    PlaneWidths, Quantum, spells,
};

/// Storage type as data: every kv page, weight plane, and traced value names
/// what it holds in one spelling, whether element or whole quantized format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    /// IEEE-754 binary32.
    F32,
    /// IEEE-754 binary16.
    F16,
    /// bfloat16: binary32's exponent range at binary16's width.
    Bf16,
    /// OCP FP8, 4-bit exponent, 3-bit mantissa.
    E4m3,
    /// OCP FP8, 5-bit exponent, 2-bit mantissa.
    E5m2,
    /// Bare 4-bit float codes, two per byte.
    E2m1,
    /// OCP Microscaling's 8-bit exponent-only scale byte.
    E8m0,
    /// Signed 64-bit.
    I64,
    /// Signed 32-bit.
    I32,
    /// Signed 16-bit.
    I16,
    /// Signed 8-bit.
    I8,
    /// Unsigned 64-bit.
    U64,
    /// Unsigned 32-bit.
    U32,
    /// Unsigned 16-bit.
    U16,
    /// Unsigned 8-bit.
    U8,
    /// One byte per element.
    Bool,

    /// OCP Microscaling FP4: `g32_e2m1_e8m0_n`.
    Mxfp4,
    /// NVIDIA NVFP4: `g16_e2m1_gt_e4m3_f32_n_n`.
    Nvfp4,
    /// Affine 4-bit, 64 per group: `g64_u4_bf16_b_bf16`.
    U4g64,
    /// Affine 8-bit, 64 per group: `g64_u8_bf16_b_bf16`.
    U8g64,
    /// Affine 4-bit, 32 per group: `g32_u4_bf16_b_bf16`.
    U4g32,
    /// Two-bit k-quant (ggml `q2_k`): `g16_u2_g16_u4_f16_n_b_g16_u4_f16_n`.
    U2g16k,
    /// Three-bit k-quant (ggml `q3_k`): `g16_i3_g16_i6_f16_n_n`.
    I3g16k,
    /// Four-bit k-quant (ggml `q4_k`): `g32_u4_g8_u6_f16_n_b_g8_u6_f16_n`.
    U4g32k,
    /// Five-bit k-quant (ggml `q5_k`): `g32_u5_g8_u6_f16_n_b_g8_u6_f16_n`.
    U5g32k,
    /// Six-bit k-quant (ggml `q6_k`): `g16_i6_g16_i8_f16_n_n`.
    I6g16k,
    /// FP8, one f32 scale per row: `gr_e4m3_f32_n`.
    E4m3row,
    /// FP8, one f32 scale per 128x128 tile: `g128x128_e4m3_f32_n`.
    E4m3tile128,
}

impl Dtype {
    /// Every variant, in declaration order.
    pub const ALL: [Self; 28] = [
        Self::F32,
        Self::F16,
        Self::Bf16,
        Self::E4m3,
        Self::E5m2,
        Self::E2m1,
        Self::E8m0,
        Self::I64,
        Self::I32,
        Self::I16,
        Self::I8,
        Self::U64,
        Self::U32,
        Self::U16,
        Self::U8,
        Self::Bool,
        Self::Mxfp4,
        Self::Nvfp4,
        Self::U4g64,
        Self::U8g64,
        Self::U4g32,
        Self::U2g16k,
        Self::I3g16k,
        Self::U4g32k,
        Self::U5g32k,
        Self::I6g16k,
        Self::E4m3row,
        Self::E4m3tile128,
    ];

    /// The variant's algebra, from which every derived number comes.
    #[must_use]
    pub const fn repr(self) -> &'static Fmt<'static> {
        const BF16: Fmt<'static> = Fmt::Elem(Elem::Bf16);
        const F16: Fmt<'static> = Fmt::Elem(Elem::F16);
        const F32: Fmt<'static> = Fmt::Elem(Elem::F32);
        const KQ_U6X8: Fmt<'static> = Fmt::Q {
            g: Group::N(8),
            elem: Elem::U(6),
            gain: &F16,
            offset: None,
        };
        const KQ_U4X16: Fmt<'static> = Fmt::Q {
            g: Group::N(16),
            elem: Elem::U(4),
            gain: &F16,
            offset: None,
        };
        match self {
            Self::F32 => &Fmt::Elem(Elem::F32),
            Self::F16 => &Fmt::Elem(Elem::F16),
            Self::Bf16 => &Fmt::Elem(Elem::Bf16),
            Self::E4m3 => &Fmt::Elem(Elem::E { e: 4, m: 3 }),
            Self::E5m2 => &Fmt::Elem(Elem::E { e: 5, m: 2 }),
            Self::E2m1 => &Fmt::Elem(Elem::E { e: 2, m: 1 }),
            Self::E8m0 => &Fmt::Elem(Elem::E { e: 8, m: 0 }),
            Self::I64 => &Fmt::Elem(Elem::I(64)),
            Self::I32 => &Fmt::Elem(Elem::I(32)),
            Self::I16 => &Fmt::Elem(Elem::I(16)),
            Self::I8 => &Fmt::Elem(Elem::I(8)),
            Self::U64 => &Fmt::Elem(Elem::U(64)),
            Self::U32 => &Fmt::Elem(Elem::U(32)),
            Self::U16 => &Fmt::Elem(Elem::U(16)),
            Self::U8 => &Fmt::Elem(Elem::U(8)),
            Self::Bool => &Fmt::Elem(Elem::Bool),
            Self::Mxfp4 => &Fmt::Q {
                g: Group::N(32),
                elem: Elem::E { e: 2, m: 1 },
                gain: &Fmt::Elem(Elem::E { e: 8, m: 0 }),
                offset: None,
            },
            Self::Nvfp4 => &Fmt::Q {
                g: Group::N(16),
                elem: Elem::E { e: 2, m: 1 },
                gain: &Fmt::Q {
                    g: Group::Tensor,
                    elem: Elem::E { e: 4, m: 3 },
                    gain: &F32,
                    offset: None,
                },
                offset: None,
            },
            Self::U4g64 => &Fmt::Q {
                g: Group::N(64),
                elem: Elem::U(4),
                gain: &BF16,
                offset: Some(Off::Post(&BF16)),
            },
            Self::U8g64 => &Fmt::Q {
                g: Group::N(64),
                elem: Elem::U(8),
                gain: &BF16,
                offset: Some(Off::Post(&BF16)),
            },
            Self::U4g32 => &Fmt::Q {
                g: Group::N(32),
                elem: Elem::U(4),
                gain: &BF16,
                offset: Some(Off::Post(&BF16)),
            },
            Self::U2g16k => &Fmt::Q {
                g: Group::N(16),
                elem: Elem::U(2),
                gain: &KQ_U4X16,
                offset: Some(Off::Post(&KQ_U4X16)),
            },
            Self::I3g16k => &Fmt::Q {
                g: Group::N(16),
                elem: Elem::I(3),
                gain: &Fmt::Q {
                    g: Group::N(16),
                    elem: Elem::I(6),
                    gain: &F16,
                    offset: None,
                },
                offset: None,
            },
            Self::U4g32k => &Fmt::Q {
                g: Group::N(32),
                elem: Elem::U(4),
                gain: &KQ_U6X8,
                offset: Some(Off::Post(&KQ_U6X8)),
            },
            Self::U5g32k => &Fmt::Q {
                g: Group::N(32),
                elem: Elem::U(5),
                gain: &KQ_U6X8,
                offset: Some(Off::Post(&KQ_U6X8)),
            },
            Self::I6g16k => &Fmt::Q {
                g: Group::N(16),
                elem: Elem::I(6),
                gain: &Fmt::Q {
                    g: Group::N(16),
                    elem: Elem::I(8),
                    gain: &F16,
                    offset: None,
                },
                offset: None,
            },
            Self::E4m3row => &Fmt::Q {
                g: Group::Row,
                elem: Elem::E { e: 4, m: 3 },
                gain: &F32,
                offset: None,
            },
            Self::E4m3tile128 => &Fmt::Q {
                g: Group::Tile(128, 128),
                elem: Elem::E { e: 4, m: 3 },
                gain: &F32,
                offset: None,
            },
        }
    }

    /// The canonical name: the repr's mangled spelling.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::E4m3 => "e4m3",
            Self::E5m2 => "e5m2",
            Self::E2m1 => "e2m1",
            Self::E8m0 => "e8m0",
            Self::I64 => "i64",
            Self::I32 => "i32",
            Self::I16 => "i16",
            Self::I8 => "i8",
            Self::U64 => "u64",
            Self::U32 => "u32",
            Self::U16 => "u16",
            Self::U8 => "u8",
            Self::Bool => "bool",
            Self::Mxfp4 => "g32_e2m1_e8m0_n",
            Self::Nvfp4 => "g16_e2m1_gt_e4m3_f32_n_n",
            Self::U4g64 => "g64_u4_bf16_b_bf16",
            Self::U8g64 => "g64_u8_bf16_b_bf16",
            Self::U4g32 => "g32_u4_bf16_b_bf16",
            Self::U2g16k => "g16_u2_g16_u4_f16_n_b_g16_u4_f16_n",
            Self::I3g16k => "g16_i3_g16_i6_f16_n_n",
            Self::U4g32k => "g32_u4_g8_u6_f16_n_b_g8_u6_f16_n",
            Self::U5g32k => "g32_u5_g8_u6_f16_n_b_g8_u6_f16_n",
            Self::I6g16k => "g16_i6_g16_i8_f16_n_n",
            Self::E4m3row => "gr_e4m3_f32_n",
            Self::E4m3tile128 => "g128x128_e4m3_f32_n",
        }
    }

    /// The variant whose repr is exactly this term, or `None` if unsupported.
    #[must_use]
    pub fn of_fmt(f: &Fmt<'_>) -> Option<Self> {
        for d in Self::ALL {
            if *d.repr() == *f {
                return Some(d);
            }
        }
        None
    }

    /// The element this dtype is, or `None` for a composite.
    #[must_use]
    pub const fn elem(self) -> Option<Elem> {
        match *self.repr() {
            Fmt::Elem(e) => Some(e),
            Fmt::Q { .. } => None,
        }
    }

    /// Bits one code occupies; for the five k-quants this is the code width,
    /// not the container size.
    #[must_use]
    pub const fn bits(self) -> u64 {
        match self.repr().code().rate() {
            Some((bits, 1)) => bits as u64,
            _ => panic!("every Dtype's code is a whole-bit element"),
        }
    }

    /// Bytes one element occupies, rounded up.
    #[must_use]
    pub const fn bytes_ceil(self) -> u64 {
        self.bits().div_ceil(8)
    }

    /// Bits per weight for a `k`-wide row.
    #[must_use]
    pub fn bpw(self, k: u32) -> Option<f64> {
        self.repr().bpw(k)
    }

    /// The leaf-per-plane byte rectangle.
    #[must_use]
    pub fn plane_widths(self, k: u32) -> Option<PlaneWidths> {
        self.repr().plane_widths(k)
    }

    /// The minimal k-split.
    #[must_use]
    pub fn quantum(self) -> Quantum {
        self.repr().quantum()
    }
}

impl core::fmt::Display for Dtype {
    /// The canonical name.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.spelling())
    }
}

const _: () = {
    let mut i = 0;
    while i < Dtype::ALL.len() {
        let d = Dtype::ALL[i];
        assert!(
            spells(d.repr(), d.spelling()),
            "a Dtype's spelling does not mangle its repr"
        );
        i += 1;
    }
};

#[cfg(feature = "serde")]
impl serde::Serialize for Dtype {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.spelling())
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Dtype {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl serde::de::Visitor<'_> for V {
            type Value = Dtype;

            fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str("a dtype spelling such as \"bf16\" or \"g64_u4_bf16_b_bf16\"")
            }

            fn visit_str<E: serde::de::Error>(self, text: &str) -> Result<Dtype, E> {
                for d in Dtype::ALL {
                    if d.spelling() == text {
                        return Ok(d);
                    }
                }
                Err(E::unknown_variant(text, &[]))
            }
        }
        d.deserialize_str(V)
    }
}

#[cfg(test)]
mod tests {
    use super::{Dtype, Fmt, Group};
    use std::string::{String, ToString};
    use std::vec::Vec;

    /// `repr` is injective, and `of_fmt` inverts it.
    #[test]
    fn repr_is_injective_and_of_fmt_inverts_it() {
        for (i, a) in Dtype::ALL.iter().enumerate() {
            for b in &Dtype::ALL[i + 1..] {
                assert_ne!(a.repr(), b.repr(), "{a:?} and {b:?} share a repr");
            }
        }
        for d in Dtype::ALL {
            assert_eq!(Dtype::of_fmt(d.repr()), Some(d), "{d:?}");
        }
        assert_eq!(Dtype::of_fmt(&super::G128_U4_F16_Z_U4), None);
        assert_eq!(Dtype::of_fmt(&super::GT_T3_F16_N), None);
    }

    /// A composite variant's identifier is derived from its repr.
    #[test]
    fn composite_names_are_derived_from_their_reprs() {
        fn cap(token: String) -> String {
            let mut c = token.chars();
            match c.next() {
                Some(first) => first.to_uppercase().chain(c).collect(),
                None => token,
            }
        }
        fn derived(f: &Fmt<'_>) -> Option<String> {
            let Fmt::Q { g, elem, gain, .. } = *f else {
                return None;
            };
            let group = match g {
                Group::N(n) => std::format!("g{n}"),
                Group::Tile(r, c) if r == c => std::format!("tile{r}"),
                Group::Tile(r, c) => std::format!("tile{r}x{c}"),
                Group::Row => "row".to_string(),
                Group::Tensor => "t".to_string(),
            };
            let nested = if matches!(gain, Fmt::Q { .. }) { "k" } else { "" };
            Some(std::format!("{}{group}{nested}", cap(elem.to_string())))
        }
        let mut exceptions = Vec::new();
        for d in Dtype::ALL {
            let Some(name) = derived(d.repr()) else {
                continue;
            };
            let ident = std::format!("{d:?}");
            if ident == name {
                continue;
            }
            exceptions.push(ident);
        }
        assert_eq!(exceptions, ["Mxfp4", "Nvfp4"]);
    }

    /// What it writes is the canonical spelling, and the trip is an identity.
    #[cfg(feature = "serde")]
    #[test]
    fn what_it_writes_is_the_spelling_and_it_round_trips() {
        let text = serde_json::to_string(&Dtype::Bf16).expect("write");
        assert_eq!(text, "\"bf16\"");
        let text = serde_json::to_string(&Dtype::U4g64).expect("write");
        assert_eq!(text, "\"g64_u4_bf16_b_bf16\"");
        for d in Dtype::ALL {
            let text = serde_json::to_string(&d).expect("write");
            let back: Dtype = serde_json::from_str(&text).expect("read");
            assert_eq!(back, d);
        }
        let bad: Result<Dtype, _> = serde_json::from_str("\"q4_k\"");
        assert!(bad.is_err(), "a vendor name is not a wire spelling");
    }

    /// The bits-per-weight numbers the published tables pin.
    #[test]
    fn bpw_matches_the_published_tables() {
        let k = 4096;
        assert_eq!(Dtype::Mxfp4.bpw(k), Some(4.25));
        assert_eq!(Dtype::Nvfp4.bpw(k), Some(4.5));
        assert_eq!(Dtype::U4g64.bpw(k), Some(4.5));
        assert_eq!(Dtype::U8g64.bpw(k), Some(8.5));
        assert_eq!(Dtype::U4g32.bpw(k), Some(5.0));
        assert_eq!(Dtype::U4g32k.bpw(k), Some(4.5));
        assert_eq!(Dtype::I6g16k.bpw(k), Some(6.5625));
        assert_eq!(Dtype::U2g16k.bpw(k), Some(2.625));
        assert_eq!(Dtype::U5g32k.bpw(k), Some(5.5));
        assert_eq!(Dtype::I3g16k.bpw(k), Some(3.4375));
        assert_eq!(Dtype::E4m3row.bpw(4096), Some(8.0 + 32.0 / 4096.0));
        assert_eq!(Dtype::E4m3row.bpw(1024), Some(8.0 + 32.0 / 1024.0));
        assert_eq!(Dtype::Bf16.bpw(k), Some(16.0));
    }
}
