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
pub mod spec;
pub mod when;

pub use ladder::{Anywhere, Decision, Ladder, Pick, Point, Rung};
pub use profile::{Candidate, Facts, Pair, Profile, Report, Source, coord};
pub use profile_macros::Decision;
pub use registry::{API, Diagnostic, File, Refusal, Registry, Strict, Target};
pub use spec::{Alternative, Constant, Domain, Kind, Shape, Sort, Spec, Value, Var};
pub use when::When;

/// What `pie` prints when a device runs without a profile measured on it.
#[must_use]
pub fn untuned_hint(device: &str, backend: &str, id: &str, report: &Report) -> String {
    let mut out = format!("{device}: no tuned profile, running {}.", report.used);
    for (source, why) in &report.rejected {
        out.push_str(&format!("\n  {source} was not used: {why}"));
    }
    out.push_str(&format!(
        "\nTo tune this device: pie tune --model <model>\n\
         Then consider contributing it:\n    \
         pie profile export {id} > {id}.toml\n\
         and open a PR at https://github.com/pie-project/pie adding it under \
         crates/engine-{backend}/profiles/. The file holds device facts and measured numbers, nothing else."
    ));
    out
}
