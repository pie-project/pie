//! `QuantScheme::Int4B8`: signed nibbles biased by eight.
//!
//! One function, and it is here rather than folded into a `match` in the
//! walker for the reason the whole module is: the scheme is a claim about
//! bytes that a CUDA kernel makes the same claim about, and the two are
//! compared element for element.

/// Unpack `QuantScheme::Int4B8` nibbles, low nibble first.
///
/// The nibbles are stored eight to a 32-bit word, but a little-endian word's
/// nibbles run low-to-high across its bytes in exactly that order, so reading
/// bytes is reading words. An element is `nibble - 8`.
pub fn decode_int4b8_elements(bytes: &[u8]) -> Vec<f64> {
    let mut values = Vec::with_capacity(bytes.len() * 2);
    for byte in bytes {
        values.push(f64::from((byte & 0xF) as i8 - 8));
        values.push(f64::from((byte >> 4) as i8 - 8));
    }
    values
}
