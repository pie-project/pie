//! The pictures a request carries inline: every API states an image as
//! base64 bytes, in a `data:` URL (OpenAI's chat and Responses) or a bare
//! payload beside its media type (Anthropic's `base64` source). The program
//! cannot fetch a remote one.

/// The bytes of a base64 `data:` URL.
pub fn data_url(url: &str) -> Result<Vec<u8>, String> {
    let Some((head, body)) = url.strip_prefix("data:").and_then(|u| u.split_once(',')) else {
        return Err("an image must be a `data:` URL; this server cannot fetch a remote one".into());
    };
    if !head.ends_with(";base64") {
        return Err("an image `data:` URL must be base64".into());
    }
    base64(body).map_err(|_| "an image `data:` URL is not valid base64".into())
}

/// Standard or URL-safe base64, padded or not, line breaks skipped.
pub fn base64(body: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(body.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in body.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' | b'\n' | b'\r' => continue,
            _ => return Err("not valid base64".into()),
        };
        acc = (acc << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_data_url_decodes_and_a_remote_one_is_refused() {
        assert_eq!(data_url("data:image/png;base64,aGVsbG8=").unwrap(), b"hello");
        assert_eq!(base64("aGVs\nbG8").unwrap(), b"hello");
        assert!(data_url("https://example.com/a.png").unwrap_err().contains("remote"));
        assert!(data_url("data:image/png,raw").unwrap_err().contains("base64"));
        assert!(data_url("data:image/png;base64,a*b").is_err());
    }
}
