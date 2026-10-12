//! GPU vendors, as a `vendor` fact spells them.

/// Every vendor a `vendor` fact may name, for its [`crate::Domain::Enum`].
pub const NAMES: &[&str] = &[
    "nvidia",
    "amd",
    "intel",
    "apple",
    "qualcomm",
    "imagination",
    "other",
];

/// The vendor a PCI vendor id stands for.
#[must_use]
pub fn of_pci(id: u32) -> &'static str {
    match id {
        0x10DE => "nvidia",
        0x1002 | 0x1022 => "amd",
        0x8086 => "intel",
        0x106B => "apple",
        0x5143 => "qualcomm",
        0x1010 => "imagination",
        _ => "other",
    }
}
