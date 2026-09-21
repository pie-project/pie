//! **A REAL PNG, WRITTEN BY HAND, INSIDE THE GUEST.**
//!
//! Lifted from the MD-B goldens' PNG writer (now `crates/runtime/tests/media_pipe_is_the_pinned_preprocessing.rs`), which
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
    image(side, |_, _| rgb)
}

/// **A DETERMINISTIC TEXTURED PNG**: a diagonal hue sweep over a checker of
/// 8-pixel cells, so every patch row is a different vector. Not a colour a
/// caption gate asserts on — a probe of how many routed experts a picture's
/// rows name at once, which a solid field (every patch identical) understates.
pub fn textured(side: u32) -> Vec<u8> {
    image(side, |x, y| {
        let t = (x + y) as f32 / (2 * side.max(1)) as f32;
        let checker = ((x / 8) + (y / 8)) % 2 == 0;
        let base = [
            (255.0 * (1.0 - t)) as u8,
            (255.0 * (0.5 - (t - 0.5).abs()) * 2.0) as u8,
            (255.0 * t) as u8,
        ];
        if checker {
            base
        } else {
            [base[0] / 2, base[1] / 2, base[2] / 2]
        }
    })
}

fn image(side: u32, pixel: impl Fn(u32, u32) -> [u8; 3]) -> Vec<u8> {
    let (w, h) = (side, side);
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

/// Standard-alphabet base64, padding optional, whitespace ignored — enough
/// to carry a picture through an argument list without a dependency.
pub fn base64_decode(text: &str) -> Result<Vec<u8>, String> {
    let value = |c: u8| -> Result<Option<u32>, String> {
        Ok(Some(match c {
            b'A'..=b'Z' => u32::from(c - b'A'),
            b'a'..=b'z' => u32::from(c - b'a') + 26,
            b'0'..=b'9' => u32::from(c - b'0') + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' | b'\n' | b'\r' | b' ' | b'\t' => return Ok(None),
            other => return Err(format!("base64: byte {other:#04x} is not in the alphabet")),
        }))
    };
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0u32);
    for &c in text.as_bytes() {
        let Some(v) = value(c)? else { continue };
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((acc >> bits) & 0xff) as u8);
        }
    }
    Ok(out)
}
