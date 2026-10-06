#[cfg(target_vendor = "apple")]
pub mod coreml;
#[cfg(target_vendor = "apple")]
pub mod ffn;
#[cfg(target_vendor = "apple")]
pub mod handoff;
#[cfg(target_vendor = "apple")]
pub mod private;

pub fn requested() -> Option<String> {
    match std::env::var("PIE_ANE").as_deref() {
        Ok("0" | "off" | "false") => None,
        Ok(v) if !v.is_empty() && !matches!(v, "1" | "on" | "true") => Some(v.to_string()),
        _ => Some(String::new()),
    }
}

#[must_use]
pub fn fingerprint(bytes: &[u8]) -> String {
    let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    });
    format!("{hash:016x}")
}
