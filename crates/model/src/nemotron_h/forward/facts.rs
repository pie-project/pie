//! nemotron_h's shape, re-exported so fixtures spelling
//! `forward::facts::NemotronHFacts` keep compiling. The semantic facts are
//! [`super::super::spec`]'s and ungated, so all three aspects can name them.

pub use super::super::spec::{
    NemotronAttnFacts, NemotronHFacts, NemotronLayerKind, NemotronMambaFacts, NemotronMoeFacts,
};
