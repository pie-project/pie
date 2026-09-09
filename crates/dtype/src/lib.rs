#![no_std]
#![forbid(unsafe_code)]

#[cfg(test)]
extern crate std;

mod repr;

pub use repr::{
    Elem, Fmt, G64_U4_F16_Z_F16, G128_U4_F16_Z_U4, GT_T3_F16_N, Group, MAX_PLANES, Mangled, Off,
    PlaneWidths, Quantum, spells,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    F32,
    F16,
    Bf16,
    E4m3,
    E5m2,
    E2m1,
    E8m0,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    Bool,

    Mxfp4,
    Nvfp4,
    U4g64,
    U8g64,
    U4g32,
    U2g32,
    U2g64,
    U2g128,
    U4g64tiled,
    U2g16k,
    I3g16k,
    U4g32k,
    U5g32k,
    I6g16k,
    E4m3row,
    E4m3tile128,
}

impl Dtype {
    pub const ALL: [Self; 32] = [
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
        Self::U2g32,
        Self::U2g64,
        Self::U2g128,
        Self::U4g64tiled,
        Self::U2g16k,
        Self::I3g16k,
        Self::U4g32k,
        Self::U5g32k,
        Self::I6g16k,
        Self::E4m3row,
        Self::E4m3tile128,
    ];

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
            Self::U2g32 => &Fmt::Q {
                g: Group::N(32),
                elem: Elem::U(2),
                gain: &BF16,
                offset: Some(Off::Post(&BF16)),
            },
            Self::U2g64 => &Fmt::Q {
                g: Group::N(64),
                elem: Elem::U(2),
                gain: &BF16,
                offset: Some(Off::Post(&BF16)),
            },
            Self::U2g128 => &Fmt::Q {
                g: Group::N(128),
                elem: Elem::U(2),
                gain: &BF16,
                offset: Some(Off::Post(&BF16)),
            },
            Self::U4g64tiled => Self::U4g64.repr(),
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
            Self::U2g32 => "g32_u2_bf16_b_bf16",
            Self::U2g64 => "g64_u2_bf16_b_bf16",
            Self::U2g128 => "g128_u2_bf16_b_bf16",
            Self::U4g64tiled => "g64_u4_bf16_b_bf16+tiled",
            Self::U2g16k => "g16_u2_g16_u4_f16_n_b_g16_u4_f16_n",
            Self::I3g16k => "g16_i3_g16_i6_f16_n_n",
            Self::U4g32k => "g32_u4_g8_u6_f16_n_b_g8_u6_f16_n",
            Self::U5g32k => "g32_u5_g8_u6_f16_n_b_g8_u6_f16_n",
            Self::I6g16k => "g16_i6_g16_i8_f16_n_n",
            Self::E4m3row => "gr_e4m3_f32_n",
            Self::E4m3tile128 => "g128x128_e4m3_f32_n",
        }
    }

    #[must_use]
    pub const fn term(self) -> &'static str {
        let spelling = self.spelling();
        let bytes = spelling.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'+' {
                let (term, _) = spelling.split_at(i);
                return term;
            }
            i += 1;
        }
        spelling
    }

    #[must_use]
    pub const fn placed(self) -> bool {
        self.term().len() != self.spelling().len()
    }

    #[must_use]
    pub fn canonical(self) -> Self {
        if !self.placed() {
            return self;
        }
        Self::of_fmt(self.repr()).expect("a placed variant's term is an unplaced variant's")
    }

    #[must_use]
    pub fn of_fmt(f: &Fmt<'_>) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|&d| !d.placed() && *d.repr() == *f)
    }

    #[must_use]
    pub const fn elem(self) -> Option<Elem> {
        match *self.repr() {
            Fmt::Elem(e) => Some(e),
            Fmt::Q { .. } => None,
        }
    }

    #[must_use]
    pub const fn bits(self) -> u64 {
        match self.repr().code().rate() {
            Some((bits, 1)) => bits as u64,
            _ => panic!("every Dtype's code is a whole-bit element"),
        }
    }

    #[must_use]
    pub const fn bytes_ceil(self) -> u64 {
        self.bits().div_ceil(8)
    }

    #[must_use]
    pub fn bpw(self, k: u32) -> Option<f64> {
        self.repr().bpw(k)
    }

    #[must_use]
    pub fn plane_widths(self, k: u32) -> Option<PlaneWidths> {
        self.repr().plane_widths(k)
    }

    #[must_use]
    pub fn row_bytes(self, k: u32) -> Option<u64> {
        self.repr().row_bytes(k)
    }

    #[must_use]
    pub fn quantum(self) -> Quantum {
        self.repr().quantum()
    }
}

impl core::fmt::Display for Dtype {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.spelling())
    }
}

const _: () = {
    let mut i = 0;
    while i < Dtype::ALL.len() {
        let d = Dtype::ALL[i];
        assert!(
            spells(d.repr(), d.term()),
            "a Dtype's spelling does not mangle its repr"
        );
        i += 1;
    }
};

pub const TILED_BAND: u32 = 16;

pub const TILED_STEP: u32 = 64;

pub const SCALES: &str = ".scales";

pub const BIASES: &str = ".biases";

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
    use super::Dtype;

    use std::vec::Vec;

    #[test]
    fn lib_every_case() {
        repr_is_injective_and_of_fmt_inverts_it();
        canonical_is_total_and_lands_on_an_unplaced_sibling();
    }

    fn repr_is_injective_and_of_fmt_inverts_it() {
        let plain: Vec<_> = Dtype::ALL.iter().filter(|d| !d.placed()).collect();
        for (i, a) in plain.iter().enumerate() {
            for b in &plain[i + 1..] {
                assert_ne!(a.repr(), b.repr(), "{a:?} and {b:?} share a repr");
            }
        }
        for d in plain {
            assert_eq!(Dtype::of_fmt(d.repr()), Some(*d), "{d:?}");
        }
        assert!(Dtype::U4g64tiled.placed());
        assert_eq!(Dtype::U4g64tiled.repr(), Dtype::U4g64.repr());
        assert_eq!(Dtype::of_fmt(Dtype::U4g64tiled.repr()), Some(Dtype::U4g64));
        assert_eq!(Dtype::of_fmt(&super::G128_U4_F16_Z_U4), None);
        assert_eq!(Dtype::of_fmt(&super::GT_T3_F16_N), None);
    }

    fn canonical_is_total_and_lands_on_an_unplaced_sibling() {
        for d in Dtype::ALL {
            let c = d.canonical();
            assert!(!c.placed(), "{d:?} canonicalizes to a placed {c:?}");
            assert_eq!(c.repr(), d.repr(), "{d:?} and {c:?} are not one algebra");
            if !d.placed() {
                assert_eq!(c, d, "{d:?} is unplaced and is its own canonical form");
            }
        }
        assert_eq!(Dtype::U4g64tiled.canonical(), Dtype::U4g64);
    }

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
}
