//! The GPU core count by model identifier, for devices whose IORegistry does
//! not say: iPhones and iPads. A row holds only where Apple's tech specs give
//! one count for every configuration the identifier names.

#![cfg_attr(not(target_os = "ios"), allow(dead_code))]

/// `(identifier, marketing name, chip, GPU cores)`.
pub(crate) const MODELS: &[(&str, &str, &str, u32)] = &[
    ("iPhone13,1", "iPhone 12 mini", "A14", 4),
    ("iPhone13,2", "iPhone 12", "A14", 4),
    ("iPhone13,3", "iPhone 12 Pro", "A14", 4),
    ("iPhone13,4", "iPhone 12 Pro Max", "A14", 4),
    ("iPhone14,2", "iPhone 13 Pro", "A15", 5),
    ("iPhone14,3", "iPhone 13 Pro Max", "A15", 5),
    ("iPhone14,4", "iPhone 13 mini", "A15", 4),
    ("iPhone14,5", "iPhone 13", "A15", 4),
    ("iPhone14,6", "iPhone SE (3rd generation)", "A15", 4),
    ("iPhone14,7", "iPhone 14", "A15", 5),
    ("iPhone14,8", "iPhone 14 Plus", "A15", 5),
    ("iPhone15,2", "iPhone 14 Pro", "A16", 5),
    ("iPhone15,3", "iPhone 14 Pro Max", "A16", 5),
    ("iPhone15,4", "iPhone 15", "A16", 5),
    ("iPhone15,5", "iPhone 15 Plus", "A16", 5),
    ("iPhone16,1", "iPhone 15 Pro", "A17 Pro", 6),
    ("iPhone16,2", "iPhone 15 Pro Max", "A17 Pro", 6),
    ("iPhone17,1", "iPhone 16 Pro", "A18 Pro", 6),
    ("iPhone17,2", "iPhone 16 Pro Max", "A18 Pro", 6),
    ("iPhone17,3", "iPhone 16", "A18", 5),
    ("iPhone17,4", "iPhone 16 Plus", "A18", 5),
    ("iPhone17,5", "iPhone 16e", "A18", 4),
    ("iPhone18,1", "iPhone 17 Pro", "A19 Pro", 6),
    ("iPhone18,2", "iPhone 17 Pro Max", "A19 Pro", 6),
    ("iPhone18,3", "iPhone 17", "A19", 5),
    ("iPhone18,4", "iPhone Air", "A19 Pro", 5),
    ("iPhone18,5", "iPhone 17e", "A19", 4),
    ("iPad13,1", "iPad Air (4th generation)", "A14", 4),
    ("iPad13,2", "iPad Air (4th generation)", "A14", 4),
    ("iPad13,4", "iPad Pro 11-inch (3rd generation)", "M1", 8),
    ("iPad13,5", "iPad Pro 11-inch (3rd generation)", "M1", 8),
    ("iPad13,6", "iPad Pro 11-inch (3rd generation)", "M1", 8),
    ("iPad13,7", "iPad Pro 11-inch (3rd generation)", "M1", 8),
    ("iPad13,8", "iPad Pro 12.9-inch (5th generation)", "M1", 8),
    ("iPad13,9", "iPad Pro 12.9-inch (5th generation)", "M1", 8),
    ("iPad13,10", "iPad Pro 12.9-inch (5th generation)", "M1", 8),
    ("iPad13,11", "iPad Pro 12.9-inch (5th generation)", "M1", 8),
    ("iPad13,16", "iPad Air (5th generation)", "M1", 8),
    ("iPad13,17", "iPad Air (5th generation)", "M1", 8),
    ("iPad13,18", "iPad (10th generation)", "A14", 4),
    ("iPad13,19", "iPad (10th generation)", "A14", 4),
    ("iPad14,1", "iPad mini (6th generation)", "A15", 5),
    ("iPad14,2", "iPad mini (6th generation)", "A15", 5),
    ("iPad14,3", "iPad Pro 11-inch (4th generation)", "M2", 10),
    ("iPad14,4", "iPad Pro 11-inch (4th generation)", "M2", 10),
    ("iPad14,5", "iPad Pro 12.9-inch (6th generation)", "M2", 10),
    ("iPad14,6", "iPad Pro 12.9-inch (6th generation)", "M2", 10),
    ("iPad14,8", "iPad Air 11-inch (M2)", "M2", 9),
    ("iPad14,9", "iPad Air 11-inch (M2)", "M2", 9),
    ("iPad14,10", "iPad Air 13-inch (M2)", "M2", 9),
    ("iPad14,11", "iPad Air 13-inch (M2)", "M2", 9),
    ("iPad15,3", "iPad Air 11-inch (M3)", "M3", 9),
    ("iPad15,4", "iPad Air 11-inch (M3)", "M3", 9),
    ("iPad15,5", "iPad Air 13-inch (M3)", "M3", 9),
    ("iPad15,6", "iPad Air 13-inch (M3)", "M3", 9),
    ("iPad15,7", "iPad (A16)", "A16", 4),
    ("iPad15,8", "iPad (A16)", "A16", 4),
    ("iPad16,1", "iPad mini (A17 Pro)", "A17 Pro", 5),
    ("iPad16,2", "iPad mini (A17 Pro)", "A17 Pro", 5),
    ("iPad16,3", "iPad Pro 11-inch (M4)", "M4", 10),
    ("iPad16,4", "iPad Pro 11-inch (M4)", "M4", 10),
    ("iPad16,5", "iPad Pro 13-inch (M4)", "M4", 10),
    ("iPad16,6", "iPad Pro 13-inch (M4)", "M4", 10),
    ("iPad16,8", "iPad Air 11-inch (M4)", "M4", 9),
    ("iPad16,9", "iPad Air 11-inch (M4)", "M4", 9),
    ("iPad16,10", "iPad Air 13-inch (M4)", "M4", 9),
    ("iPad16,11", "iPad Air 13-inch (M4)", "M4", 9),
    ("iPad17,1", "iPad Pro 11-inch (M5)", "M5", 10),
    ("iPad17,2", "iPad Pro 11-inch (M5)", "M5", 10),
    ("iPad17,3", "iPad Pro 13-inch (M5)", "M5", 10),
    ("iPad17,4", "iPad Pro 13-inch (M5)", "M5", 10),
];

/// Each chip's `MTLGPUFamily::AppleN`.
pub(crate) const CHIPS: &[(&str, u32)] = &[
    ("A14", 7),
    ("A15", 8),
    ("A16", 8),
    ("A17 Pro", 9),
    ("A18", 9),
    ("A18 Pro", 9),
    ("A19", 10),
    ("A19 Pro", 10),
    ("M1", 7),
    ("M2", 8),
    ("M3", 9),
    ("M4", 9),
    ("M5", 10),
];

/// The GPU cores of the model `identifier` names, with its row's identifier,
/// where the table holds it and its chip is of `family`: a device that runs
/// another family is not the one the row describes.
#[must_use]
pub(crate) fn gpu_cores(identifier: &str, family: u32) -> Option<(u32, &'static str)> {
    let &(id, _, chip, cores) = MODELS.iter().find(|row| row.0 == identifier)?;
    let &(_, of) = CHIPS.iter().find(|c| c.0 == chip)?;
    (of == family).then_some((cores, id))
}

/// `hw.machine`: on iOS and iPadOS, the model identifier.
#[cfg(target_os = "ios")]
pub(crate) fn machine() -> Option<String> {
    let mut buf = [0u8; 64];
    let mut len = buf.len();
    // SAFETY: `hw.machine` is a string sysctl; `len` states the out buffer's
    // size and the call writes at most that many bytes.
    let rc = unsafe {
        libc::sysctlbyname(
            c"hw.machine".as_ptr(),
            buf.as_mut_ptr().cast(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return None;
    }
    let text = buf.get(..len)?;
    let end = text.iter().position(|&b| b == 0).unwrap_or(text.len());
    String::from_utf8(text[..end].to_vec()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_identifier_is_one_row() {
        let mut ids: Vec<&str> = MODELS.iter().map(|row| row.0).collect();
        ids.sort_unstable();
        let before = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), before, "an identifier appears twice");
        for (id, name, _, cores) in MODELS {
            assert!(id.starts_with("iPhone") || id.starts_with("iPad"), "{id}");
            assert!(!name.is_empty() && *cores > 0, "{id}");
        }
    }

    #[test]
    fn every_model_has_a_chip_family() {
        for (id, _, chip, _) in MODELS {
            assert!(
                CHIPS.iter().any(|c| c.0 == *chip),
                "{id}: no family for {chip}"
            );
        }
    }

    #[test]
    fn an_iphone_14_pro_has_five_cores_and_a_mismatched_family_has_none() {
        assert_eq!(gpu_cores("iPhone15,2", 8), Some((5, "iPhone15,2")));
        assert_eq!(gpu_cores("iPhone15,2", 9), None);
        assert_eq!(gpu_cores("iPhone14,5", 8), Some((4, "iPhone14,5")));
        assert_eq!(gpu_cores("Mac16,10", 9), None);
    }
}
