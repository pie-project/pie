#[cfg(target_vendor = "apple")]
pub mod ffn;
#[cfg(target_vendor = "apple")]
pub mod handoff;
#[cfg(target_vendor = "apple")]
pub mod private;

pub fn enabled() -> bool {
    !matches!(
        std::env::var("PIE_ANE").as_deref(),
        Ok("0" | "off" | "false")
    )
}

#[must_use]
pub fn units() -> Option<u32> {
    std::env::var("PIE_ANE_UNITS").ok()?.parse().ok()
}

#[must_use]
pub fn fingerprint(bytes: &[u8]) -> String {
    let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    });
    format!("{hash:016x}")
}
