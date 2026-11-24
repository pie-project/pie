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
