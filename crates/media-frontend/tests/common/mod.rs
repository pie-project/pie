//! **A REAL PNG, WRITTEN BY HAND** — and the reason it is written by hand.
//!
//! Both front-end gates end with a whole-pipe claim: real encoded bytes in,
//! the right shapes out. Encoding those bytes with the same crate that decodes
//! them would make the claim circular — an encoder and a decoder from one
//! library agree with each other by construction, and a gate that only proves
//! that has proved nothing about the file format. So this module emits PNG
//! from the specification: IHDR, one IDAT of STORED (uncompressed) deflate
//! blocks under a zlib wrapper, IEND, with CRC-32 per chunk and Adler-32 over
//! the raw stream.
//!
//! Stored blocks rather than a compressor for the same reason the whole crate
//! prefers transcription to cleverness: there is exactly one byte sequence this
//! can emit for a given image, so two runs and two machines produce identical
//! bytes and the digest gate downstream means what it says.

#![allow(dead_code)]

/// CRC-32 (IEEE), the polynomial PNG's chunk checksum names.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

/// Adler-32, zlib's own checksum over the UNCOMPRESSED stream.
fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &x in bytes {
        a = (a + u32::from(x)) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], body: &[u8]) {
    #[allow(clippy::cast_possible_truncation)]
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
    let mut crc_over = Vec::with_capacity(4 + body.len());
    crc_over.extend_from_slice(kind);
    crc_over.extend_from_slice(body);
    out.extend_from_slice(&crc32(&crc_over).to_be_bytes());
}

/// **A DETERMINISTIC `w × h` 8-BIT RGB PNG.**
///
/// `pixel(x, y)` names the colour; the caller picks a rule it can also assert
/// against, so a test can follow one source pixel all the way to a patch lane.
pub fn png_rgb(w: u32, h: u32, pixel: impl Fn(u32, u32) -> [u8; 3]) -> Vec<u8> {
    // Raw scanlines: PNG prefixes each with a filter byte, and 0 is "None".
    let mut raw = Vec::with_capacity((h * (1 + w * 3)) as usize);
    for y in 0..h {
        raw.push(0u8);
        for x in 0..w {
            raw.extend_from_slice(&pixel(x, y));
        }
    }

    // zlib: CMF/FLG, then stored deflate blocks of at most 65535 bytes, then
    // Adler-32 of the raw stream.
    let mut z = vec![0x78u8, 0x01];
    let mut at = 0usize;
    while at < raw.len() {
        let take = (raw.len() - at).min(0xffff);
        let last = u8::from(at + take == raw.len());
        z.push(last);
        #[allow(clippy::cast_possible_truncation)]
        let len = take as u16;
        z.extend_from_slice(&len.to_le_bytes());
        z.extend_from_slice(&(!len).to_le_bytes());
        z.extend_from_slice(&raw[at..at + take]);
        at += take;
    }
    z.extend_from_slice(&adler32(&raw).to_be_bytes());

    let mut out = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit, colour type 2 (RGB)
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &z);
    chunk(&mut out, b"IEND", &[]);
    out
}

/// A ramp that makes every pixel of a small image distinct, so a golden can
/// name which source pixel it expects in which patch lane.
#[must_use]
pub fn ramp(x: u32, y: u32) -> [u8; 3] {
    [
        ((x * 7 + y * 13) % 251) as u8,
        ((x * 31 + y * 3) % 251) as u8,
        ((x + y * 97) % 251) as u8,
    ]
}
