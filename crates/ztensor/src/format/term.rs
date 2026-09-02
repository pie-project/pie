//! Types (spec §4): the leaf table, the term grammar, and the planes a term
//! lays out under the canonical layout (§5.1).
//!
//! A [`Term`] is what an object's `type` field says. It has exactly one
//! spelling per tree, so two types are equal iff their strings are.

use std::fmt;

use crate::error::{Error, Result, Rule};
use crate::format::align_up;

/// Planes inside one blob start at multiples of this (spec §5.1 rule 3).
/// 256 because a consumer binds a plane straight out of a file mapping, and
/// the device APIs want a 256-byte-aligned operand.
pub const PLANE_ALIGN: u64 = 256;
/// Group nesting bound (spec §4.2).
pub const MAX_TERM_DEPTH: u32 = 8;

/// A leaf: a bit pattern of fixed width and the number it denotes (§4.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Leaf {
    /// Unsigned integer of `b` bits, `1..=64`.
    U(u8),
    /// Two's-complement signed integer of `b` bits, `1..=64`.
    I(u8),
    F16,
    BF16,
    F32,
    F64,
    /// OCP MX FP4.
    E2M1,
    /// OCP MX FP6 E2M3.
    E2M3,
    /// OCP MX FP6 E3M2.
    E3M2,
    /// OCP FP8 E4M3 (finite-only).
    E4M3,
    /// OCP FP8 E5M2.
    E5M2,
    /// OCP MX scale: `2^(code − 127)`, `0xFF` is NaN.
    E8M0,
    /// One byte, `0x00` or `0x01`.
    Bool,
}

impl Leaf {
    pub const U8: Leaf = Leaf::U(8);
    pub const U16: Leaf = Leaf::U(16);
    pub const U32: Leaf = Leaf::U(32);
    pub const U64: Leaf = Leaf::U(64);
    pub const I8: Leaf = Leaf::I(8);
    pub const I16: Leaf = Leaf::I(16);
    pub const I32: Leaf = Leaf::I(32);
    pub const I64: Leaf = Leaf::I(64);

    pub fn bits(self) -> u64 {
        match self {
            Leaf::U(b) | Leaf::I(b) => u64::from(b),
            Leaf::F16 | Leaf::BF16 => 16,
            Leaf::F32 => 32,
            Leaf::F64 => 64,
            Leaf::E2M1 => 4,
            Leaf::E2M3 | Leaf::E3M2 => 6,
            Leaf::E4M3 | Leaf::E5M2 | Leaf::E8M0 | Leaf::Bool => 8,
        }
    }

    /// Bytes of `n` packed elements: `⌈n · bits / 8⌉`.
    pub fn size(self, n: u64) -> Option<u64> {
        n.checked_mul(self.bits()).map(|bits| bits.div_ceil(8))
    }

    /// Byte width when the leaf is a whole number of bytes.
    pub fn width(self) -> Option<u64> {
        let bits = self.bits();
        (bits % 8 == 0).then_some(bits / 8)
    }

    pub fn parse(s: &str) -> Option<Leaf> {
        Some(match s {
            "f16" => Leaf::F16,
            "bf16" => Leaf::BF16,
            "f32" => Leaf::F32,
            "f64" => Leaf::F64,
            "e2m1" => Leaf::E2M1,
            "e2m3" => Leaf::E2M3,
            "e3m2" => Leaf::E3M2,
            "e4m3" => Leaf::E4M3,
            "e5m2" => Leaf::E5M2,
            "e8m0" => Leaf::E8M0,
            "bool" => Leaf::Bool,
            _ => {
                let bits = number(s.get(1..)?)?;
                if !(1..=64).contains(&bits) {
                    return None;
                }
                match s.as_bytes()[0] {
                    b'u' => Leaf::U(bits as u8),
                    b'i' => Leaf::I(bits as u8),
                    _ => return None,
                }
            }
        })
    }
}

impl fmt::Display for Leaf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Leaf::U(b) => write!(f, "u{b}"),
            Leaf::I(b) => write!(f, "i{b}"),
            Leaf::F16 => f.write_str("f16"),
            Leaf::BF16 => f.write_str("bf16"),
            Leaf::F32 => f.write_str("f32"),
            Leaf::F64 => f.write_str("f64"),
            Leaf::E2M1 => f.write_str("e2m1"),
            Leaf::E2M3 => f.write_str("e2m3"),
            Leaf::E3M2 => f.write_str("e3m2"),
            Leaf::E4M3 => f.write_str("e4m3"),
            Leaf::E5M2 => f.write_str("e5m2"),
            Leaf::E8M0 => f.write_str("e8m0"),
            Leaf::Bool => f.write_str("bool"),
        }
    }
}

impl std::str::FromStr for Leaf {
    type Err = Error;
    fn from_str(s: &str) -> Result<Leaf> {
        Leaf::parse(s).ok_or_else(|| Error::reject(Rule::Type, format!("unknown leaf {s:?}")))
    }
}

/// How many elements share one factor (§4.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Group {
    /// `g{N}`: `N` consecutive elements along the last axis.
    N(u64),
    /// `g{R}x{C}`: an `R × C` tile over the last two axes, `R ≥ 2`.
    Tile(u64, u64),
    /// `gr`: one row.
    Row,
    /// `gt`: the whole tensor.
    Tensor,
}

impl fmt::Display for Group {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Group::N(n) => write!(f, "g{n}"),
            Group::Tile(r, c) => write!(f, "g{r}x{c}"),
            Group::Row => f.write_str("gr"),
            Group::Tensor => f.write_str("gt"),
        }
    }
}

/// The offset of a group form.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Offset {
    /// `n`.
    None,
    /// `b_<term>`: added after scaling.
    Post(Box<Term>),
    /// `z_<term>`: subtracted before scaling.
    Pre(Box<Term>),
}

impl Offset {
    pub fn term(&self) -> Option<&Term> {
        match self {
            Offset::None => None,
            Offset::Post(t) | Offset::Pre(t) => Some(t),
        }
    }
}

/// A type (§4.2): a leaf, or codes of a leaf grouped under a gain term and an
/// offset.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Term {
    Leaf(Leaf),
    Group {
        group: Group,
        code: Leaf,
        gain: Box<Term>,
        offset: Offset,
    },
}

impl From<Leaf> for Term {
    fn from(leaf: Leaf) -> Term {
        Term::Leaf(leaf)
    }
}

/// One plane of a term under the canonical layout: where it is in the blob
/// and what it holds (§4.4, §5.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plane {
    /// `"code"`, `"gain"`, `"offset"`, `"gain.code"`, ...; `"data"` for a
    /// leaf term.
    pub path: String,
    pub leaf: Leaf,
    pub shape: Vec<u64>,
    /// Byte offset within the decoded blob.
    pub offset: u64,
    pub len: u64,
}

impl Plane {
    pub fn elements(&self) -> u64 {
        self.shape.iter().product()
    }

    pub fn range(&self) -> std::ops::Range<usize> {
        self.offset as usize..(self.offset + self.len) as usize
    }
}

fn number(s: &str) -> Option<u64> {
    if s.is_empty() || (s.len() > 1 && s.starts_with('0')) || !s.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    s.parse().ok()
}

fn group(tok: &str) -> Option<Group> {
    let rest = tok.strip_prefix('g')?;
    Some(match rest {
        "r" => Group::Row,
        "t" => Group::Tensor,
        _ => match rest.split_once('x') {
            Some((r, c)) => {
                let (r, c) = (number(r)?, number(c)?);
                if r < 2 || c < 1 {
                    return None;
                }
                Group::Tile(r, c)
            }
            None => {
                let n = number(rest)?;
                if n < 1 {
                    return None;
                }
                Group::N(n)
            }
        },
    })
}

struct Parser<'s> {
    text: &'s str,
    tokens: Vec<&'s str>,
    at: usize,
}

impl<'s> Parser<'s> {
    fn bad(&self, why: &str) -> Error {
        Error::reject(Rule::Type, format!("type {:?}: {why}", self.text))
    }

    fn next(&mut self) -> Result<&'s str> {
        let tok = self.tokens.get(self.at).copied();
        self.at += 1;
        tok.ok_or_else(|| self.bad("ends early"))
    }

    fn term(&mut self, depth: u32) -> Result<Term> {
        if depth > MAX_TERM_DEPTH {
            return Err(self.bad("nests deeper than 8 group forms"));
        }
        let tok = self.next()?;
        if let Some(g) = group(tok) {
            let code_tok = self.next()?;
            let code = Leaf::parse(code_tok)
                .ok_or_else(|| self.bad(&format!("{code_tok:?} is not a leaf")))?;
            let gain = Box::new(self.term(depth + 1)?);
            let offset = match self.next()? {
                "n" => Offset::None,
                "b" => Offset::Post(Box::new(self.term(depth + 1)?)),
                "z" => Offset::Pre(Box::new(self.term(depth + 1)?)),
                other => return Err(self.bad(&format!("{other:?} is not an offset"))),
            };
            return Ok(Term::Group {
                group: g,
                code,
                gain,
                offset,
            });
        }
        if tok.starts_with('g') {
            return Err(self.bad(&format!("{tok:?} is not a group")));
        }
        Leaf::parse(tok)
            .map(Term::Leaf)
            .ok_or_else(|| self.bad(&format!("{tok:?} is not a leaf")))
    }
}

impl Term {
    /// Parses a type string. Rejects under [`Rule::Type`] anything that is not
    /// exactly one well-formed term.
    pub fn parse(text: &str) -> Result<Term> {
        let mut p = Parser {
            text,
            tokens: text.split('_').collect(),
            at: 0,
        };
        let term = p.term(0)?;
        if p.at != p.tokens.len() {
            return Err(p.bad("has tokens after the term"));
        }
        Ok(term)
    }

    pub fn leaf(&self) -> Option<Leaf> {
        match self {
            Term::Leaf(l) => Some(*l),
            Term::Group { .. } => None,
        }
    }

    /// The planes this term lays out for an object of `shape`, in canonical
    /// order with canonical offsets (§4.4, §5.1).
    pub fn planes(&self, shape: &[u64]) -> Result<Vec<Plane>> {
        let mut out = Vec::new();
        walk(self, shape, "", &mut out)?;
        let mut cursor = 0u64;
        for plane in &mut out {
            let elements = plane
                .shape
                .iter()
                .try_fold(1u64, |a, &d| a.checked_mul(d))
                .ok_or_else(|| Error::reject(Rule::Shape, "plane element count overflows u64"))?;
            let len = plane
                .leaf
                .size(elements)
                .ok_or_else(|| Error::reject(Rule::Shape, "plane byte size overflows u64"))?;
            plane.offset = cursor;
            plane.len = len;
            cursor = cursor
                .checked_add(len)
                .and_then(|end| align_up(end, PLANE_ALIGN))
                .ok_or_else(|| Error::reject(Rule::Shape, "blob size overflows u64"))?;
        }
        Ok(out)
    }

    /// The decoded blob size under the canonical layout: the end of the last
    /// plane (§5.1 rule 4).
    pub fn canonical_size(&self, shape: &[u64]) -> Result<u64> {
        Ok(self
            .planes(shape)?
            .last()
            .map(|p| p.offset + p.len)
            .unwrap_or(0))
    }

    /// Content rules over a canonical blob: `bool` bytes are 0 or 1, the
    /// unused high bits of a packed plane's last byte are zero, and so are
    /// the bytes between planes (§4.1, §4.4, §5.1 rule 3).
    pub fn check_bytes(&self, shape: &[u64], bytes: &[u8]) -> Result<()> {
        let mut end = 0;
        for plane in self.planes(shape)? {
            let (Some(gap), Some(data)) = (
                bytes.get(end..plane.offset as usize),
                bytes.get(plane.range()),
            ) else {
                return Err(Error::reject(Rule::Size, "blob shorter than its planes"));
            };
            if gap.iter().any(|&b| b != 0) {
                return Err(Error::reject(
                    Rule::LayoutData,
                    format!("bytes before plane {:?} must be zero", plane.path),
                ));
            }
            end = plane.range().end;
            if plane.leaf == Leaf::Bool && data.iter().any(|&b| b > 1) {
                return Err(Error::reject(
                    Rule::LayoutData,
                    format!("plane {:?}: bool bytes must be 0x00 or 0x01", plane.path),
                ));
            }
            let used = (plane.elements() * plane.leaf.bits()) % 8;
            if used != 0 && data.last().is_some_and(|&b| b >> used != 0) {
                return Err(Error::reject(
                    Rule::LayoutData,
                    format!("plane {:?}: unused tail bits must be zero", plane.path),
                ));
            }
        }
        Ok(())
    }
}

fn factor_shape(g: Group, shape: &[u64]) -> Result<Vec<u64>> {
    let bad = |why: String| Error::reject(Rule::Type, why);
    Ok(match g {
        Group::N(n) => {
            let Some((&last, lead)) = shape.split_last() else {
                return Err(bad(format!("{g} needs rank ≥ 1")));
            };
            if last % n != 0 {
                return Err(bad(format!("{g} does not divide the last axis ({last})")));
            }
            let mut s = lead.to_vec();
            s.push(last / n);
            s
        }
        Group::Tile(r, c) => {
            let [lead @ .., rows, cols] = shape else {
                return Err(bad(format!("{g} needs rank ≥ 2")));
            };
            if rows % r != 0 || cols % c != 0 {
                return Err(bad(format!(
                    "{g} does not divide the last two axes ({rows}×{cols})"
                )));
            }
            let mut s = lead.to_vec();
            s.extend([rows / r, cols / c]);
            s
        }
        Group::Row => {
            let Some((_, lead)) = shape.split_last() else {
                return Err(bad("gr needs rank ≥ 1".into()));
            };
            lead.to_vec()
        }
        Group::Tensor => Vec::new(),
    })
}

fn walk(term: &Term, shape: &[u64], path: &str, out: &mut Vec<Plane>) -> Result<()> {
    let name = |leaf: &str| {
        if path.is_empty() {
            leaf.to_string()
        } else {
            format!("{path}.{leaf}")
        }
    };
    match term {
        Term::Leaf(leaf) => out.push(Plane {
            path: if path.is_empty() {
                "data".into()
            } else {
                path.into()
            },
            leaf: *leaf,
            shape: shape.to_vec(),
            offset: 0,
            len: 0,
        }),
        Term::Group {
            group,
            code,
            gain,
            offset,
        } => {
            out.push(Plane {
                path: name("code"),
                leaf: *code,
                shape: shape.to_vec(),
                offset: 0,
                len: 0,
            });
            let factor = factor_shape(*group, shape)?;
            walk(gain, &factor, &name("gain"), out)?;
            if let Some(t) = offset.term() {
                walk(t, &factor, &name("offset"), out)?;
            }
        }
    }
    Ok(())
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Leaf(l) => write!(f, "{l}"),
            Term::Group {
                group,
                code,
                gain,
                offset,
            } => {
                write!(f, "{group}_{code}_{gain}_")?;
                match offset {
                    Offset::None => f.write_str("n"),
                    Offset::Post(t) => write!(f, "b_{t}"),
                    Offset::Pre(t) => write!(f, "z_{t}"),
                }
            }
        }
    }
}

impl std::str::FromStr for Term {
    type Err = Error;
    fn from_str(s: &str) -> Result<Term> {
        Term::parse(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_is_canonical() {
        for s in [
            "bf16",
            "u4",
            "g64_u4_bf16_b_bf16",
            "g32_e2m1_e8m0_n",
            "g16_e2m1_gt_e4m3_f32_n_n",
            "g32_u4_g8_u6_f16_n_b_g8_u6_f16_n",
            "g128_u4_f16_z_u4",
            "gr_e4m3_f32_n",
            "g128x128_e4m3_f32_n",
        ] {
            assert_eq!(Term::parse(s).unwrap().to_string(), s);
        }
        for s in [
            "", "_", "g0_u4_bf16_n", "g1x64_u4_bf16_n", "g064_u4_bf16_n", "u0", "u65",
            "g64_u4_bf16", "g64_u4_bf16_b", "bf16_n", "g64_g8_u4_bf16_n", "G64_u4_bf16_n",
        ] {
            assert!(Term::parse(s).is_err(), "{s:?} parsed");
        }
    }

    #[test]
    fn planes_of_u4g64() {
        let t = Term::parse("g64_u4_bf16_b_bf16").unwrap();
        let planes = t.planes(&[4096, 4096]).unwrap();
        let got: Vec<(&str, u64, u64)> = planes
            .iter()
            .map(|p| (p.path.as_str(), p.offset, p.len))
            .collect();
        assert_eq!(
            got,
            [
                ("code", 0, 8_388_608),
                ("gain", 8_388_608, 524_288),
                ("offset", 8_912_896, 524_288)
            ]
        );
        assert_eq!(t.canonical_size(&[4096, 4096]).unwrap(), 9_437_184);
        assert!(t.planes(&[4096, 100]).is_err());
    }

    #[test]
    fn planes_align_and_nest() {
        let t = Term::parse("g32_u4_g8_u6_f16_n_b_g8_u6_f16_n").unwrap();
        let paths: Vec<String> = t
            .planes(&[2, 256])
            .unwrap()
            .into_iter()
            .map(|p| p.path)
            .collect();
        assert_eq!(
            paths,
            ["code", "gain.code", "gain.gain", "offset.code", "offset.gain"]
        );
        let t = Term::parse("g16_e2m1_gt_e4m3_f32_n_n").unwrap();
        let planes = t.planes(&[3, 16]).unwrap();
        assert_eq!(planes[0].len, 24);
        assert_eq!(planes[1].offset, 256);
        assert_eq!(planes[1].shape, [3, 1]);
        assert_eq!(planes[2].shape, Vec::<u64>::new());
        assert_eq!(planes[2].offset, 512);
        assert_eq!(t.canonical_size(&[3, 16]).unwrap(), 516);
    }

    #[test]
    fn content_rules() {
        let t = Term::Leaf(Leaf::U(3));
        assert!(t.check_bytes(&[2], &[0b0011_1111]).is_ok());
        assert!(t.check_bytes(&[2], &[0b1011_1111]).is_err());
        assert!(Term::Leaf(Leaf::Bool).check_bytes(&[1], &[2]).is_err());

        let t = Term::parse("g16_e2m1_e8m0_n").unwrap();
        let mut blob = vec![0u8; 257];
        assert!(t.check_bytes(&[16], &blob).is_ok());
        blob[100] = 1;
        assert_eq!(
            t.check_bytes(&[16], &blob).unwrap_err().rule(),
            Some(Rule::LayoutData)
        );
    }
}
