pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::{Model, Routed};
use poem_dsl::Dtype;

use crate::catalog::{Drafter, Entry, Refused};

pub fn flash_u2g64() -> Model {
    Model::flash_mini(Dtype::U4g64, Routed::DQ_2BIT, Dtype::Bf16, Dtype::Bf16)
}

pub fn flash_u2g64_mtp() -> Model {
    Model::flash_mini_mtp(Dtype::U4g64, Routed::DQ_2BIT, Dtype::Bf16, Dtype::Bf16)
}

pub fn flash_u2g64_full_mtp() -> Model {
    Model::flash_mixed_mtp(Dtype::U4g64, Routed::DQ_2BIT_FULL, Dtype::Bf16, Dtype::Bf16)
}

pub fn flash_u2g64_full() -> Model {
    Model::flash_mixed(Dtype::U4g64, Routed::DQ_2BIT_FULL, Dtype::Bf16, Dtype::Bf16)
}

/// DeepSeek-V4.1-Flash as `mlx_lm` publishes it: every projection and the
/// routed experts at MLX affine 4-bit, the Engram tables read out to bf16.
pub fn flash41_u4g64() -> Model {
    Model::flash41(
        Dtype::U4g64,
        Routed::split(Dtype::U4g64),
        Dtype::Bf16,
        Dtype::Bf16,
    )
}

/// DeepSeek-V4.1-Flash with its fp8 planes read out to bf16 and its fp4
/// experts kept as MXFP4 codes (the released checkpoint's own expert format).
pub fn flash41_mxfp4() -> Model {
    Model::flash41(
        Dtype::Bf16,
        Routed::split(Dtype::Mxfp4),
        Dtype::Bf16,
        Dtype::Bf16,
    )
}

pub fn flash41_mini_mxfp4() -> Model {
    Model::flash41_mini(
        Dtype::Bf16,
        Routed::split(Dtype::Mxfp4),
        Dtype::Bf16,
        Dtype::Bf16,
    )
}

pub fn entries() -> Vec<Entry> {
    use Dtype::{Bf16, Mxfp4, U2g64, U4g64};
    vec![
        crate::entry! {
            id: "dsv41-flash",
            mini: false,
            parts: [],
            drafters: [],
            template: template::r1,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                match d.weights[..] {
                    [U4g64] => Ok(Model::flash41(U4g64, Routed::split(U4g64), Bf16, d.kv)),
                    [Bf16, Mxfp4] => Ok(Model::flash41(Bf16, Routed::split(Mxfp4), Bf16, d.kv)),
                    _ => Err(Refused::unsupported("dsv41-flash", d)),
                }
            },
            rows: [
                (0, [U4g64], Bf16, [], None),
                (1, [Bf16, Mxfp4], Bf16, [], None),
            ],
        },
        crate::entry! {
            id: "dsv41-flash-mini",
            mini: true,
            parts: [],
            drafters: [],
            template: template::r1,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                match d.weights[..] {
                    [Bf16, Mxfp4] => Ok(Model::flash41_mini(Bf16, Routed::split(Mxfp4), Bf16, d.kv)),
                    _ => Err(Refused::unsupported("dsv41-flash-mini", d)),
                }
            },
            rows: [(2, [Bf16, Mxfp4], Bf16, [], None)],
        },
        crate::entry! {
            id: "dsv4-flash",
            mini: false,
            parts: [],
            drafters: [Mtp],
            template: template::r1,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                match (&d.weights[..], d.drafter) {
                    ([Bf16], None) => Ok(Model::flash(Bf16, Bf16, d.kv)),
                    ([U4g64, U2g64], None) => {
                        Ok(Model::flash_mixed(U4g64, Routed::DQ_2BIT_FULL, Bf16, d.kv))
                    }
                    ([U4g64, U2g64, Mxfp4], Some(Drafter::Mtp)) => {
                        Ok(Model::flash_mixed_mtp(U4g64, Routed::DQ_2BIT_FULL, Bf16, d.kv))
                    }
                    _ => Err(Refused::unsupported("dsv4-flash", d)),
                }
            },
            rows: [
                (3, [U4g64, U2g64, Mxfp4], Bf16, [], Some(Drafter::Mtp)),
                (5, [U4g64, U2g64], Bf16, [], None),
                (10, [Bf16], Bf16, [], None),
            ],
        },
        crate::entry! {
            id: "dsv4-flash-mini",
            mini: true,
            parts: [],
            drafters: [Mtp],
            template: template::r1,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                match (&d.weights[..], d.drafter) {
                    ([Bf16], None) => Ok(Model::flash_mini(Bf16, Routed::uniform(Bf16), Bf16, d.kv)),
                    ([U4g64, U2g64], None) => {
                        Ok(Model::flash_mini(U4g64, Routed::DQ_2BIT, Bf16, d.kv))
                    }
                    ([U4g64, U2g64, Mxfp4], Some(Drafter::Mtp)) => {
                        Ok(Model::flash_mini_mtp(U4g64, Routed::DQ_2BIT, Bf16, d.kv))
                    }
                    _ => Err(Refused::unsupported("dsv4-flash-mini", d)),
                }
            },
            rows: [
                (4, [U4g64, U2g64, Mxfp4], Bf16, [], Some(Drafter::Mtp)),
                (6, [U4g64, U2g64], Bf16, [], None),
                (8, [Bf16], Bf16, [], None),
            ],
        },
        crate::entry! {
            id: "dsv4-base",
            mini: true,
            parts: [],
            drafters: [],
            template: template::r1,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                match d.weights[..] {
                    [Bf16] => Ok(Model::base(Bf16, Bf16, d.kv)),
                    _ => Err(Refused::unsupported("dsv4-base", d)),
                }
            },
            rows: [(9, [Bf16], Bf16, [], None)],
        },
    ]
}
