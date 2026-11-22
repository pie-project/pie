//! The chip's profile: the CS-3's calibration (`profiles/cs3.toml`), which a load's programs are
//! priced under ([`kernels_cerebras::tuning`]).

use kernels_cerebras::tuning::{self, Tuning, registry};
use profile::Shelf;

/// The CS-3's checked-in profile.
pub const CS3: &str = include_str!("../profiles/cs3.toml");

pub static SHELF: Shelf = Shelf {
    backend: "cerebras",
    seeded: &[("cs3.toml", CS3)],
    derived: None,
};

/// Price this process's programs under the CS-3's calibration.
pub fn calibrate() -> Result<(), String> {
    let (profile, _) = SHELF.resolve(registry(), "Cerebras CS-3", &tuning::cs3(), None);
    tuning::set(Tuning::of(&profile))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The checked-in profile holds together, a CS-3 runs it, it states every decision, and they
    /// are the defaults.
    #[test]
    fn the_cs3_profile_is_the_defaults() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("profiles");
        SHELF
            .check(registry(), &dir)
            .unwrap_or_else(|e| panic!("{e}"));
        let (_, report) = SHELF.resolve(registry(), "Cerebras CS-3", &tuning::cs3(), None);
        assert_eq!(report.used, profile::Source::Seeded("cs3.toml".into()));
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
        assert_eq!(*tuning::get(), Tuning::DEFAULT);
    }
}
