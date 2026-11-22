//! The chip's profile: the CS-3's calibration (`profiles/cs3.toml`), which a load's programs are
//! priced under ([`kernels_cerebras::tuning`]).

/// The CS-3's checked-in profile.
pub const CS3: &str = include_str!("../profiles/cs3.toml");

/// Price this process's programs under the CS-3's calibration.
pub fn calibrate() -> Result<(), String> {
    kernels_cerebras::tuning::set(kernels_cerebras::tuning::load(CS3, "cs3.toml")?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernels_cerebras::tuning::{Tuning, registry};

    /// The checked-in profile reads strictly, states every decision, and is the defaults.
    #[test]
    fn the_cs3_profile_is_the_defaults() {
        let p = registry().load(CS3, "cs3.toml").unwrap();
        for s in registry().specs() {
            let key = s.id.trim_start_matches("kernel.");
            assert!(
                CS3.lines().any(|l| l.starts_with(&format!("{key} = "))),
                "{key}"
            );
        }
        assert_eq!(Tuning::of(&p), Tuning::DEFAULT);
        calibrate().unwrap();
        assert_eq!(*kernels_cerebras::tuning::get(), Tuning::DEFAULT);
    }
}
