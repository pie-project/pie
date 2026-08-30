//! **A REAL PNG, WRITTEN BY HAND, INSIDE THE GUEST.**
//!
//! Lifted from `crates/media-frontend/tests/common/mod.rs` (wave MD-B), which
//! argues the idiom: PNG emitted from the SPECIFICATION — IHDR, one IDAT of
//! STORED (uncompressed) deflate blocks under a zlib wrapper, IEND, CRC-32 per
//! chunk and Adler-32 over the raw stream. Stored blocks rather than a
//! compressor because there is then exactly one byte sequence this can emit
//! for a given image, so two runs and two machines produce identical bytes.
//!
//! **AND THAT IS WHY THE GATE SYNTHESIZES ITS IMAGE RATHER THAN SHIPPING ONE.**
//! Determinism is what this gate is for: a checked-in photograph would be an
//! opaque blob whose content no reader of this file can verify, and a caption
//! asserted against it would be asserted against something nobody could check.
//! A solid square is a picture whose content is a sentence — "it is red" — and
//! the bytes that carry it are a function of four lines above.
//!
//! It costs the guest no dependency: this is `std` and arithmetic, and pixels
//! never cross into WASM in the other direction, so producing them here is the
//! only way a guest could hold an image at all.

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
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
    let mut crc_over = Vec::with_capacity(4 + body.len());
    crc_over.extend_from_slice(kind);
    crc_over.extend_from_slice(body);
    out.extend_from_slice(&crc32(&crc_over).to_be_bytes());
}

/// **A DETERMINISTIC `side × side` 8-BIT RGB PNG OF ONE COLOUR.**
pub fn solid(side: u32, rgb: [u8; 3]) -> Vec<u8> {
    let (w, h) = (side, side);
    // Raw scanlines: PNG prefixes each with a filter byte, and 0 is "None".
    let mut raw = Vec::with_capacity((h * (1 + w * 3)) as usize);
    for _ in 0..h {
        raw.push(0u8);
        for _ in 0..w {
            raw.extend_from_slice(&rgb);
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
