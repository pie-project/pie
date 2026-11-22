//! Device profiles: the hardware-dependent decisions a backend exposes, read
//! from layered TOML files (a checked-in family file, a checked-in device
//! file, the user's file) against a [`Registry`], each layer checked against
//! the probed facts and applied whole or not at all.

pub mod routes;
pub mod fire;
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

pub use ladder::{Anywhere, Decision, Ladder, Mark, Pick, Point, Rung};
pub use profile::{Candidate, Facts, Pair, Profile, Report, Source, coord};
pub use profile_macros::Decision;
pub use region::{Region, Set};
pub use registry::{API, Diagnostic, File, Refusal, Registry, Strict, Target};
pub use shelf::{Shelf, id_of};
pub use spec::{Alternative, Constant, Domain, Kind, Shape, Sort, Spec, Tune, Value, Var};
pub use when::When;

/// The tolerance two measurements of one thing agree within: two runs of a
/// rates step, a check's estimate against the measured best, and two
/// candidates of a search (the lowest of those within it of the fastest is
/// taken, so two tunes write the same profile). One definition, which
/// `engine::rates` re-exports.
pub const AGREEMENT: f64 = 0.05;

/// The backends `pie tune` can measure.
const TUNABLE: &[&str] = &["metal", "cuda", "vulkan"];

/// What `pie` prints when a device runs without a profile measured on it.
fn untuned_hint(device: &str, backend: &str, id: &str, report: &Report) -> String {
    let mut out = if report.layers.is_empty() {
        format!("{device}: no profile for it, running the registry defaults.")
    } else {
        format!(
            "{device}: running {}: {}.",
            report.describe(),
            report.what_was_measured()
        )
    };
    for (source, why) in &report.rejected {
        out.push_str(&format!("\n  {source} was not used: {why}"));
    }
    let how = if TUNABLE.contains(&backend) {
        format!("To tune it (minutes; --full for a file to contribute):\n    pie tune {backend}\n")
    } else {
        format!(
            "To tune it, start from the profile it runs now:\n    \
             pie profile dump > $PIE_HOME/profiles/{backend}/user/{id}.toml\n\
             change what you measure, "
        )
    };
    out.push_str(&format!(
        "\n{how}and consider contributing the result:\n    \
         pie profile export {id} > {id}.toml\n\
         and open a PR at https://github.com/pie-project/pie adding it under \
         crates/engine-{backend}/profiles/. The file holds device facts and measured \
         numbers, nothing else."
    ));
    out
}
