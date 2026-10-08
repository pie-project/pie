pub mod template;
pub mod tokenizer;

use poem_dsl::Dtype;

use crate::catalog::{Entry, Row};
use crate::star::{MUSE_GLIMMER, import, trace};

/// One entry of the family, its layout, forward and formats those of the
/// `muse_glimmer` package.
macro_rules! muse {
    ($id:literal, $mini:literal, [$( ($seq:literal, $w:expr) ),*]) => {
        Entry {
            id: $id,
            mini: $mini,
            parts: &[],
            drafters: &[],
            trace: |name, d, platform| trace(&MUSE_GLIMMER, $id, name, d, platform),
            import: |d, src, platform| import(&MUSE_GLIMMER, $id, d, src, platform),
            template: template::muse_glimmer,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: |_| None,
            generative: |_| None,
            rows: vec![$(Row {
                seq: $seq,
                deploy: crate::catalog::Deploy {
                    weights: vec![$w],
                    kv: Dtype::Bf16,
                    tp: 1,
                    parts: vec![],
                    drafter: None,
                },
            }),*],
        }
    };
}

pub fn entries() -> Vec<Entry> {
    vec![
        muse!(
            "muse-glimmer-30b",
            false,
            [(0, Dtype::Bf16), (2, Dtype::U4g64)]
        ),
        muse!("muse-glimmer-30b-mini-l8", true, [(4, Dtype::Bf16)]),
    ]
}
