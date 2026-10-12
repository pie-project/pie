//! Device profiles.
//!
//! A profile maps the hardware-dependent **decisions** a backend's code
//! exposes to values: which alternative an op runs and with what constants,
//! which fusions hold, how pages are laid out, how the engine paces frames.
//! Each consuming crate declares its decisions with `#[derive(Decision)]`;
//! the engine crate assembles them into one [`Registry`] per backend; a
//! checked-in or user TOML file is read against it, linted, checked against
//! the facts the backend probed, and applied whole or not at all.
//!
//! `.wiki/device-profiles.md` holds the design and its rulings.

pub mod ladder;
pub mod profile;
pub mod region;
pub mod registry;
pub mod render;
pub mod search;
pub mod shelf;
pub mod spec;
pub mod tune;
pub mod vendor;
pub mod when;

pub use ladder::{Anywhere, Decision, Ladder, Pick, Point, Rung};
pub use profile::{Candidate, Facts, Pair, Profile, Report, Source, coord};
pub use profile_macros::Decision;
pub use registry::{API, Diagnostic, File, Refusal, Registry, Strict, Target};
pub use shelf::{Shelf, id_of};
pub use spec::{Alternative, Constant, Domain, Kind, Shape, Sort, Spec, Tune, Value, Var};
pub use when::When;

/// The backends `pie tune` can measure.
pub const TUNABLE: &[&str] = &["metal", "cuda", "vulkan"];

/// What `pie` prints when a device runs without a profile measured on it.
#[must_use]
pub fn untuned_hint(device: &str, backend: &str, id: &str, report: &Report) -> String {
    let mut out = if report.used.is_tuned() && report.rejected.is_empty() {
        format!(
            "{device}: running {}, whose numbers were carried over, not measured on it.",
            report.used
        )
    } else {
        format!("{device}: no tuned profile, running {}.", report.used)
    };
    for (source, why) in &report.rejected {
        out.push_str(&format!("\n  {source} was not used: {why}"));
    }
    if TUNABLE.contains(&backend) {
        out.push_str(&format!(
            "\nTo tune it (minutes; --full for a file to contribute):\n    \
             pie tune {backend}\n\
             and consider contributing the result:\n    \
             pie profile export {id} > {id}.toml\n\
             and open a PR at https://github.com/pie-project/pie adding it under \
             crates/engine-{backend}/profiles/. The file holds device facts and measured \
             numbers, nothing else."
        ));
        return out;
    }
    out.push_str(&format!(
        "\nTo tune it, start from the profile it runs now:\n    \
         pie profile dump > $PIE_HOME/profiles/{backend}/user/{id}.toml\n\
         change what you measure, and consider contributing the result:\n    \
         pie profile export {id} > {id}.toml\n\
         and open a PR at https://github.com/pie-project/pie adding it under \
         crates/engine-{backend}/profiles/. The file holds device facts and measured \
         numbers, nothing else."
    ));
    out
}
