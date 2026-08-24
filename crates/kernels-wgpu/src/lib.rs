mod capability;
pub use crate::capability::Capability;

pub mod preproc;
pub use crate::preproc::{Directive, Malformed, Variant, expand, instantiations};

pub mod source;
pub use crate::source::{Missing, SOURCES, entrypoint_source, source};

pub mod points;

pub mod plane;
pub mod points_dispatch;
pub mod views;

pub mod attn;
pub mod layout;
pub mod mlp;
pub mod moe;
pub mod norm;
pub mod ptir;
pub mod quant;
pub mod rope;
pub mod sample;
pub mod ssm;

#[must_use]
pub fn entrypoints() -> Vec<String> {
    static CENSUS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    CENSUS
        .get_or_init(|| {
            let mut out: Vec<String> = source::declared()
                .into_iter()
                .map(|(_, variant)| variant.entrypoint)
                .collect();
            out.sort();

            out.dedup();
            out
        })
        .clone()
}
