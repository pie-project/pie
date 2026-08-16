//! `QuantScheme::Int4B8`: signed nibbles biased by eight.
//!
//! One function, and it is here rather than a `match` arm in the walker for the
//! reason the whole module is: the scheme is a claim about bytes that a CUDA
//! kernel makes the same claim about, compared element for element.

/// Unpack `QuantScheme::Int4B8` nibbles, low nibble first: eight to a 32-bit
/// word, whose nibbles run low-to-high across its bytes in that order, so
/// reading bytes is reading words. An element is `nibble - 8`.
pub fn decode_int4b8_elements(bytes: &[u8]) -> Vec<f64> {
    let mut values = Vec::with_capacity(bytes.len() * 2);
    for byte in bytes {
        values.push(f64::from((byte & 0xF) as i8 - 8));
        values.push(f64::from((byte >> 4) as i8 - 8));
    }
    values
}
