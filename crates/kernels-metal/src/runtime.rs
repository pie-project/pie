pub mod beam;
pub mod blit;
pub mod nan_check;
pub mod resident;

use crate::encode::Fire;

/// The files whose every kernel is a fixed pipeline ([`fixed_pipelines`]):
/// a kernel added to one and left off that list fails this crate's tests,
/// never a fire after ready.
pub const FIXED_FILES: &[&str] = &["runtime/beam.metal", "runtime/blit.metal"];

/// The pipelines a fire can name by its lanes' shape rather than its rung:
/// a beam group's head step and its page and row copies, and on a load that
/// buffers recurrent rows (`recurrent`) its row copies (one a launch, or up
/// to four side by side, by how many runs its lanes gather) and a long
/// carry's scans. A load's walk fires each rung's forms over one lane shape
/// each, so it reaches none of these in full; a load builds every one
/// before it seals, beside what its walk built, and is refused by name
/// where one does not build: a pipeline missing from a load is met at its
/// load, never at its first beam.
#[must_use]
pub fn fixed_pipelines(recurrent: bool) -> Vec<Fire> {
    let mut fires = beam::pipelines().to_vec();
    if recurrent {
        fires.extend(blit::pipelines());
        fires.extend(crate::ssm::carried_pipelines());
    }
    fires
}

#[cfg(test)]
mod tests {
    use super::{FIXED_FILES, fixed_pipelines};

    /// Every kernel of a fixed file is a fixed pipeline: the copy launching
    /// four copies (`rs_copy_words4`) was one the walk never fired and the
    /// load never built, so a beam's replay on a recurrent model was refused
    /// at its first fire.
    #[test]
    fn every_kernel_of_a_fixed_file_is_built_at_load() {
        let fixed = fixed_pipelines(true);
        for &file in FIXED_FILES {
            let source = crate::sources::source(file).expect("a fixed file is a source");
            let kernels: Vec<&str> = source
                .split("kernel void ")
                .skip(1)
                .filter_map(|rest| {
                    rest.split(|c: char| !(c.is_alphanumeric() || c == '_'))
                        .next()
                })
                .filter(|name| !name.is_empty())
                .collect();
            assert!(!kernels.is_empty(), "{file} names no kernel");
            for kernel in kernels {
                assert!(
                    fixed
                        .iter()
                        .any(|fire| fire.file == file && fire.entrypoint == kernel),
                    "`{kernel}` of {file} is no fixed pipeline: a fire naming it after ready is refused"
                );
            }
        }
    }
}
