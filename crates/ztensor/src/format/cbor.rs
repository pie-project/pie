//! Minimal CBOR codec for the zTensor manifest.
//!
//! Implements exactly the subset the spec permits (§3.1): unsigned and
//! negative integers, byte strings, text strings, arrays, maps, false/true/
//! null, and floats. Definite lengths only, no tags, depth ≤ 32.
//!
//! RFC 8949 core deterministic encoding is enforced in both directions: the
//! encoder emits shortest-form heads, sorts map keys by their encoded bytes,
//! and rejects duplicates; the decoder requires shortest-form heads and
//! strictly ascending key order (which also catches duplicates).

use crate::error::{Error, Result, Rule};

pub const MAX_DEPTH: u32 = 32;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Uint(u64),
    /// Negative integer with value `-1 - n` (CBOR major type 1).
    Nint(u64),
    Bytes(Vec<u8>),
    Text(String),
    Array(Vec<Value>),
    /// Entries in any order; the encoder sorts by encoded key bytes.
    Map(Vec<(Value, Value)>),
    Bool(bool),
    Null,
    Float(f64),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Uint(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Value::Text(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_map(&self) -> Option<&[(Value, Value)]> {
        match self {
            Value::Map(m) => Some(m),
            _ => None,
        }
    }

    /// Looks up a key in a map value.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.as_map()?
            .iter()
            .find(|(k, _)| k.as_text() == Some(key))
            .map(|(_, v)| v)
    }
}

/// Builds an attributes map: `cbor::map([("group", 32u64), ("who", "me")])`.
///
/// Attributes are always a map of text keys (spec §3.1/§3.5), so this is the
/// shape every caller needs and the one they should not have to spell out.
pub fn map<K: Into<String>, V: Into<Value>>(entries: impl IntoIterator<Item = (K, V)>) -> Value {
    Value::Map(
        entries
            .into_iter()
            .map(|(k, v)| (Value::Text(k.into()), v.into()))
            .collect(),
    )
}

impl From<u64> for Value {
    fn from(n: u64) -> Value {
        Value::Uint(n)
    }
}

impl From<u32> for Value {
    fn from(n: u32) -> Value {
        Value::Uint(n as u64)
    }
}

impl From<i64> for Value {
    fn from(n: i64) -> Value {
        if n < 0 {
            Value::Nint((-1 - n) as u64)
        } else {
            Value::Uint(n as u64)
        }
    }
}

impl From<f64> for Value {
    fn from(x: f64) -> Value {
        Value::Float(x)
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Value {
        Value::Bool(b)
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Value {
        Value::Text(s.to_string())
    }
}

impl From<String> for Value {
    fn from(s: String) -> Value {
        Value::Text(s)
    }
}

impl From<Vec<u8>> for Value {
    fn from(b: Vec<u8>) -> Value {
        Value::Bytes(b)
    }
}

impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(items: Vec<T>) -> Value {
        Value::Array(items.into_iter().map(Into::into).collect())
    }
}

// =======================================================================
// Encoding
// =======================================================================

pub fn encode(v: &Value) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    encode_into(v, &mut out, 0)?;
    Ok(out)
}

fn head(major: u8, arg: u64, out: &mut Vec<u8>) {
    let mt = major << 5;
    if arg < 24 {
        out.push(mt | arg as u8);
    } else if arg <= 0xff {
        out.push(mt | 24);
        out.push(arg as u8);
    } else if arg <= 0xffff {
        out.push(mt | 25);
        out.extend((arg as u16).to_be_bytes());
    } else if arg <= 0xffff_ffff {
        out.push(mt | 26);
        out.extend((arg as u32).to_be_bytes());
    } else {
        out.push(mt | 27);
        out.extend(arg.to_be_bytes());
    }
}

fn encode_into(v: &Value, out: &mut Vec<u8>, depth: u32) -> Result<()> {
    if depth > MAX_DEPTH {
        return Err(Error::InvalidInput(format!(
            "CBOR nesting exceeds {MAX_DEPTH}"
        )));
    }
    match v {
        Value::Uint(n) => head(0, *n, out),
        Value::Nint(n) => head(1, *n, out),
        Value::Bytes(b) => {
            head(2, b.len() as u64, out);
            out.extend_from_slice(b);
        }
        Value::Text(s) => {
            head(3, s.len() as u64, out);
            out.extend_from_slice(s.as_bytes());
        }
        Value::Array(items) => {
            head(4, items.len() as u64, out);
            for item in items {
                encode_into(item, out, depth + 1)?;
            }
        }
        Value::Map(entries) => {
            let mut enc: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(entries.len());
            for (k, val) in entries {
                let mut kb = Vec::new();
                encode_into(k, &mut kb, depth + 1)?;
                let mut vb = Vec::new();
                encode_into(val, &mut vb, depth + 1)?;
                enc.push((kb, vb));
            }
            enc.sort_by(|a, b| a.0.cmp(&b.0));
            for w in enc.windows(2) {
                if w[0].0 == w[1].0 {
                    return Err(Error::InvalidInput("duplicate map key".into()));
                }
            }
            head(5, enc.len() as u64, out);
            for (kb, vb) in enc {
                out.extend_from_slice(&kb);
                out.extend_from_slice(&vb);
            }
        }
        Value::Bool(b) => out.push(if *b { 0xf5 } else { 0xf4 }),
        Value::Null => out.push(0xf6),
        Value::Float(x) => encode_float(*x, out),
    }
    Ok(())
}

/// Shortest float encoding that preserves the value (deterministic profile).
fn encode_float(x: f64, out: &mut Vec<u8>) {
    if x.is_nan() {
        out.push(0xf9);
        out.extend(0x7e00u16.to_be_bytes());
        return;
    }
    let as32 = x as f32;
    if as32 as f64 == x {
        let h = f32_to_f16_bits(as32);
        if f16_bits_to_f32(h) == as32 {
            out.push(0xf9);
            out.extend(h.to_be_bytes());
            return;
        }
        out.push(0xfa);
        out.extend(as32.to_bits().to_be_bytes());
        return;
    }
    out.push(0xfb);
    out.extend(x.to_bits().to_be_bytes());
}

fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x0200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflows half range; roundtrip check filters
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let man = man | 0x0080_0000;
        return sign | (man >> (14 - e)) as u16;
    }
    sign | ((e as u16) << 10) | (man >> 13) as u16
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = (h >> 10) & 0x1f;
    let man = (h & 0x03ff) as u32;
    let bits = match exp {
        0 => {
            if man == 0 {
                sign
            } else {
                let mut e = 127 - 15 + 1;
                let mut m = man;
                while m & 0x400 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (man << 13),
        e => sign | ((e as u32 + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

// =======================================================================
// Decoding
// =======================================================================

pub fn decode(input: &[u8]) -> Result<Value> {
    let mut d = Decoder { input, pos: 0 };
    let v = d.value(0)?;
    if d.pos != input.len() {
        return Err(Error::reject(Rule::CborSyntax, "trailing bytes"));
    }
    Ok(v)
}

struct Decoder<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    fn remaining(&self) -> usize {
        self.input.len() - self.pos
    }

    fn byte(&mut self) -> Result<u8> {
        let b = *self
            .input
            .get(self.pos)
            .ok_or_else(|| Error::reject(Rule::CborSyntax, "unexpected end of input"))?;
        self.pos += 1;
        Ok(b)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.remaining() < n {
            return Err(Error::reject(Rule::CborSyntax, "unexpected end of input"));
        }
        let s = &self.input[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    /// Reads a head for major types 0-5, enforcing shortest form.
    fn head_arg(&mut self, ai: u8) -> Result<u64> {
        let arg = match ai {
            0..=23 => ai as u64,
            24 => {
                let v = self.byte()? as u64;
                if v < 24 {
                    return Err(Error::reject(Rule::CborDeterminism, "non-shortest head"));
                }
                v
            }
            25 => {
                let v = u16::from_be_bytes(self.take(2)?.try_into().unwrap()) as u64;
                if v <= 0xff {
                    return Err(Error::reject(Rule::CborDeterminism, "non-shortest head"));
                }
                v
            }
            26 => {
                let v = u32::from_be_bytes(self.take(4)?.try_into().unwrap()) as u64;
                if v <= 0xffff {
                    return Err(Error::reject(Rule::CborDeterminism, "non-shortest head"));
                }
                v
            }
            27 => {
                let v = u64::from_be_bytes(self.take(8)?.try_into().unwrap());
                if v <= 0xffff_ffff {
                    return Err(Error::reject(Rule::CborDeterminism, "non-shortest head"));
                }
                v
            }
            _ => {
                return Err(Error::reject(
                    Rule::CborSyntax,
                    "indefinite length or reserved head",
                ))
            }
        };
        Ok(arg)
    }

    fn value(&mut self, depth: u32) -> Result<Value> {
        if depth > MAX_DEPTH {
            return Err(Error::reject(Rule::CborDepth, "nesting too deep"));
        }
        let first = self.byte()?;
        let major = first >> 5;
        let ai = first & 0x1f;

        if major == 7 {
            return match ai {
                20 => Ok(Value::Bool(false)),
                21 => Ok(Value::Bool(true)),
                22 => Ok(Value::Null),
                // Floats must be canonical: the single NaN 0xf9 0x7e00, and
                // the shortest width that preserves the value. Without this,
                // decode-accepted values are not closed under re-encoding
                // (e.g., two NaN payloads collapse into duplicate map keys).
                25 => {
                    let h = u16::from_be_bytes(self.take(2)?.try_into().unwrap());
                    let x = f16_bits_to_f32(h);
                    if x.is_nan() && h != 0x7e00 {
                        return Err(Error::reject(Rule::CborDeterminism, "non-canonical NaN"));
                    }
                    Ok(Value::Float(x as f64))
                }
                26 => {
                    let b = u32::from_be_bytes(self.take(4)?.try_into().unwrap());
                    let x = f32::from_bits(b);
                    if x.is_nan() {
                        return Err(Error::reject(Rule::CborDeterminism, "non-canonical NaN"));
                    }
                    if f16_bits_to_f32(f32_to_f16_bits(x)) == x {
                        return Err(Error::reject(Rule::CborDeterminism, "float not shortest"));
                    }
                    Ok(Value::Float(x as f64))
                }
                27 => {
                    let b = u64::from_be_bytes(self.take(8)?.try_into().unwrap());
                    let x = f64::from_bits(b);
                    if x.is_nan() {
                        return Err(Error::reject(Rule::CborDeterminism, "non-canonical NaN"));
                    }
                    if (x as f32) as f64 == x {
                        return Err(Error::reject(Rule::CborDeterminism, "float not shortest"));
                    }
                    Ok(Value::Float(x))
                }
                _ => Err(Error::reject(Rule::CborSyntax, "unsupported simple value")),
            };
        }
        if major == 6 {
            return Err(Error::reject(Rule::CborSyntax, "tags are not permitted"));
        }

        let arg = self.head_arg(ai)?;
        match major {
            0 => Ok(Value::Uint(arg)),
            1 => Ok(Value::Nint(arg)),
            2 => {
                let n = self.checked_len(arg, 1)?;
                Ok(Value::Bytes(self.take(n)?.to_vec()))
            }
            3 => {
                let n = self.checked_len(arg, 1)?;
                let s = std::str::from_utf8(self.take(n)?)
                    .map_err(|_| Error::reject(Rule::CborSyntax, "invalid UTF-8 in text"))?;
                Ok(Value::Text(s.to_string()))
            }
            4 => {
                let n = self.checked_len(arg, 1)?;
                let mut items = Vec::with_capacity(n);
                for _ in 0..n {
                    items.push(self.value(depth + 1)?);
                }
                Ok(Value::Array(items))
            }
            5 => {
                let n = self.checked_len(arg, 2)?;
                let mut entries = Vec::with_capacity(n);
                let mut prev: Option<(usize, usize)> = None;
                for _ in 0..n {
                    let ks = self.pos;
                    let key = self.value(depth + 1)?;
                    let ke = self.pos;
                    if let Some((ps, pe)) = prev {
                        use std::cmp::Ordering::*;
                        match self.input[ps..pe].cmp(&self.input[ks..ke]) {
                            Less => {}
                            Equal => {
                                return Err(Error::reject(
                                    Rule::CborDuplicateKey,
                                    "duplicate map key",
                                ))
                            }
                            Greater => {
                                return Err(Error::reject(
                                    Rule::CborDeterminism,
                                    "map keys not in canonical order",
                                ))
                            }
                        }
                    }
                    prev = Some((ks, ke));
                    let val = self.value(depth + 1)?;
                    entries.push((key, val));
                }
                Ok(Value::Map(entries))
            }
            _ => unreachable!(),
        }
    }

    /// Bounds a declared element count by the remaining input (each element
    /// occupies at least `min_bytes`), preventing huge pre-allocations.
    fn checked_len(&self, arg: u64, min_bytes: usize) -> Result<usize> {
        let max = (self.remaining() / min_bytes) as u64;
        if arg > max {
            return Err(Error::reject(
                Rule::CborSyntax,
                "declared length exceeds input",
            ));
        }
        Ok(arg as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(v: Value) {
        let bytes = encode(&v).unwrap();
        assert_eq!(decode(&bytes).unwrap(), v);
    }

    #[test]
    fn scalars() {
        roundtrip(Value::Uint(0));
        roundtrip(Value::Uint(23));
        roundtrip(Value::Uint(24));
        roundtrip(Value::Uint(u64::MAX));
        roundtrip(Value::Nint(0)); // -1
        roundtrip(Value::Bool(true));
        roundtrip(Value::Null);
        roundtrip(Value::Text("hello".into()));
        roundtrip(Value::Bytes(vec![1, 2, 3]));
        roundtrip(Value::Float(1.5));
        roundtrip(Value::Float(1.1));
        roundtrip(Value::Float(65504.0)); // max f16
    }

    #[test]
    fn maps_sorted_and_deduped() {
        let v = Value::Map(vec![
            (Value::Text("bb".into()), Value::Uint(1)),
            (Value::Text("a".into()), Value::Uint(2)),
        ]);
        let bytes = encode(&v).unwrap();
        // decoder accepts (encoder sorted: "a" < "bb" by encoded bytes)
        let back = decode(&bytes).unwrap();
        let m = back.as_map().unwrap();
        assert_eq!(m[0].0.as_text().unwrap(), "a");

        let dup = Value::Map(vec![
            (Value::Text("a".into()), Value::Uint(1)),
            (Value::Text("a".into()), Value::Uint(2)),
        ]);
        assert!(encode(&dup).is_err());
    }

    #[test]
    fn rejects_unsorted_and_dup_on_decode() {
        // {"b": 0, "a": 0}, which is the wrong order
        let bytes = [0xa2, 0x61, b'b', 0x00, 0x61, b'a', 0x00];
        assert!(matches!(
            decode(&bytes),
            Err(Error::Reject {
                rule: Rule::CborDeterminism,
                ..
            })
        ));
        // {"a": 0, "a": 0}
        let bytes = [0xa2, 0x61, b'a', 0x00, 0x61, b'a', 0x00];
        assert!(matches!(
            decode(&bytes),
            Err(Error::Reject {
                rule: Rule::CborDuplicateKey,
                ..
            })
        ));
    }

    #[test]
    fn rejects_tags_and_indefinite() {
        assert!(decode(&[0xc0, 0x00]).is_err()); // tag 0
        assert!(decode(&[0x9f, 0xff]).is_err()); // indefinite array
        assert!(decode(&[0x18, 0x05]).is_err()); // non-shortest uint 5
    }

    #[test]
    fn rejects_non_canonical_floats() {
        assert!(decode(&[0xf9, 0x7e, 0x01]).is_err()); // NaN with payload
        assert!(decode(&[0xf9, 0xfe, 0x00]).is_err()); // -NaN
        assert!(decode(&[0xfa, 0x7f, 0xc0, 0x00, 0x00]).is_err()); // f32 NaN
        assert!(decode(&[0xfa, 0x3f, 0xc0, 0x00, 0x00]).is_err()); // 1.5 fits f16
        assert!(decode(&[0xfb, 0x3f, 0xf8, 0, 0, 0, 0, 0, 0]).is_err()); // 1.5 fits f16
        assert!(decode(&[0xf9, 0x3e, 0x00]).is_ok()); // canonical 1.5
        assert!(decode(&[0xf9, 0x7e, 0x00]).is_ok()); // canonical NaN
    }

    /// A declared length never drives an allocation.
    ///
    /// The bug this forecloses is the one every binary decoder gets caught by:
    /// reading a length from the input and reserving that much before checking
    /// that the bytes are actually there, so nine bytes ask for sixteen
    /// exabytes. Every length here is bounded by what remains to be read.
    #[test]
    fn a_declared_length_is_checked_before_it_is_believed() {
        for major in [
            2u8, /* bytes */
            3,   /* text */
            4,   /* array */
            5,   /* map */
        ] {
            for declared in [u64::MAX, 1 << 40, 1 << 20, 300, 25] {
                // The shortest head for this length, so the determinism rule
                // cannot fire first and the length check is what is on trial.
                let mut input = Vec::new();
                head(major, declared, &mut input);
                let err = decode(&input).expect_err("a lie about length must be refused");
                assert_eq!(
                    err.rule(),
                    Some(Rule::CborSyntax),
                    "major {major}, declared {declared}: {err}"
                );
            }
        }
    }

    /// Nesting is bounded exactly at the documented depth, on both sides.
    #[test]
    fn nesting_stops_at_the_limit() {
        let nest = |depth: usize| {
            let mut v = vec![0x81u8; depth]; // array(1), repeated
            v.push(0x00); // and a uint at the bottom
            v
        };
        decode(&nest(MAX_DEPTH as usize)).expect("the limit itself is legal");
        let err = decode(&nest(MAX_DEPTH as usize + 1)).expect_err("one past it is not");
        assert_eq!(err.rule(), Some(Rule::CborDepth));
    }

    /// Regression: fuzz_cbor crash on a map with two distinct NaN-payload
    /// float keys decoded fine but re-encoded into duplicate keys.
    #[test]
    fn fuzz_regression_nan_map_keys() {
        let input = [
            0xa2, 0xfb, 0xff, 0xff, 0xff, 0x05, 0x3b, 0x8f, 0xfb, 0xfb, 0xfb, 0xfb, 0xfb, 0xfb,
            0xfb, 0xfb, 0xfb, 0x39, 0x38, 0xfb, 0xff, 0xff, 0xff, 0x0e, 0x39, 0x39, 0xa2, 0xa2,
            0x07,
        ];
        assert!(matches!(
            decode(&input),
            Err(Error::Reject {
                rule: Rule::CborDeterminism,
                ..
            })
        ));
    }
}
