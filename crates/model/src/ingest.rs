use crate::shared::vocabulary::Vocab;
use model_loader::checkpoint::Attributes;
use model_loader::error::Error;

#[derive(Clone, Debug, PartialEq)]
pub enum Ingest {

    Rename(String),

    Unpermute { name: String, heads: u32 },

    Debias { name: String, by: f32 },

    Unstack { each: String },

    Drop,
}

impl Ingest {

    #[must_use]
    pub fn name(&self) -> Option<&str> {
        match self {
            Self::Rename(name) | Self::Unpermute { name, .. } | Self::Debias { name, .. } => {
                Some(name)
            }
            Self::Unstack { .. } | Self::Drop => None,
        }
    }
}

struct Pass {

    vocab: Vocab,

    derived: fn(&str) -> bool,

    regroup: fn(&Attributes, &str) -> Option<u32>,

    folded: fn(&str) -> Option<f32>,

    stacked: fn(&str) -> bool,
}

fn pass_for(architecture: &str) -> Result<Pass, Error> {
    match architecture {
        "qwen2" => Ok(Pass {
            vocab: crate::qwen_2::import::VOCAB,
            derived: |_| false,
            regroup: |_, _| None,
            folded: |_| None,
            stacked: |_| false,
        }),
        "qwen3" => Ok(Pass {
            vocab: crate::qwen_3::import::VOCAB,
            derived: |_| false,
            regroup: |_, _| None,
            folded: |_| None,
            stacked: |_| false,
        }),
        "qwen3moe" => Ok(Pass {
            vocab: crate::qwen_3::import_moe::VOCAB,
            derived: |_| false,
            regroup: |_, _| None,
            folded: |_| None,
            stacked: crate::qwen_3::import_moe::is_stacked,
        }),
        "gemma3" => Ok(Pass {
            vocab: crate::gemma_3::import::VOCAB,
            derived: |_| false,
            regroup: |_, _| None,
            folded: crate::gemma_3::import::folded_constant,
            stacked: |_| false,
        }),
        "llama" => Ok(Pass {
            vocab: crate::shared::llama_like::import::VOCAB,
            derived: crate::shared::llama_like::import::is_derived,
            regroup: crate::shared::llama_like::import::regroup_heads,
            folded: |_| None,
            stacked: |_| false,
        }),
        other => Err(Error::Contract(format!(
            "this is a `{other}` GGUF, and pie has no GGUF ingest pass for it. \
             Import the HuggingFace release instead, or add \
             `crates/model/src/<generation>/import.rs` and an arm in \
             `crates/model/src/ingest.rs`"
        ))),
    }
}

struct Family {

    generation: &'static str,

    arch: &'static str,

    vocabs: &'static [Vocab],

    #[allow(dead_code, reason = "read by the tests that ground this table")]
    rows: fn() -> &'static [&'static dyn crate::catalog::Variant],
}

const FAMILIES: &[Family] = &[
    Family {
        generation: "llama_3",
        arch: "llama",
        vocabs: &[crate::llama_3::import::VOCAB],
        rows: crate::llama_3::rows,
    },
    Family {
        generation: "qwen_2",
        arch: "qwen2",
        vocabs: &[crate::qwen_2::import::VOCAB],
        rows: crate::qwen_2::rows,
    },
    Family {
        generation: "qwen_3",
        arch: "qwen3",
        vocabs: &[
            crate::qwen_3::import::VOCAB,
            crate::qwen_3::import_moe::VOCAB,
        ],
        rows: crate::qwen_3::rows,
    },
    Family {
        generation: "qwen_3_5",
        arch: "qwen3_5",
        vocabs: &[crate::qwen_3_5::import::VOCAB],
        rows: crate::qwen_3_5::rows,
    },
    Family {
        generation: "gemma_2",
        arch: "gemma2",
        vocabs: &[crate::gemma_2::import::VOCAB],
        rows: crate::gemma_2::rows,
    },
    Family {
        generation: "gemma_3",
        arch: "gemma3",
        vocabs: &[crate::gemma_3::import::VOCAB],
        rows: crate::gemma_3::rows,
    },
    Family {
        generation: "gemma_3n",
        arch: "gemma3n",
        vocabs: &[crate::gemma_3n::import::VOCAB],
        rows: crate::gemma_3n::rows,
    },
    Family {
        generation: "gemma_4",
        arch: "gemma4",
        vocabs: &[crate::gemma_4::import::VOCAB],
        rows: crate::gemma_4::rows,
    },
    Family {
        generation: "glm_5",
        arch: "",
        vocabs: &[crate::glm_5::import::VOCAB],
        rows: crate::glm_5::rows,
    },
    Family {
        generation: "gpt_oss",
        arch: "gptoss",
        vocabs: &[crate::gpt_oss::import::VOCAB],
        rows: crate::gpt_oss::rows,
    },
    Family {
        generation: "kimi_k2",
        arch: "",
        vocabs: &[crate::kimi_k2::import::VOCAB],
        rows: crate::kimi_k2::rows,
    },
    Family {
        generation: "kimi_k3",
        arch: "",
        vocabs: &[crate::kimi_k3::import::VOCAB],
        rows: crate::kimi_k3::rows,
    },
    Family {
        generation: "deepseek_v4",
        arch: "",
        vocabs: &[crate::deepseek_v4::import::VOCAB],
        rows: crate::deepseek_v4::rows,
    },
    Family {
        generation: "nemotron_h",
        arch: "nemotron_h",
        vocabs: &[crate::nemotron_h::import::VOCAB],
        rows: crate::nemotron_h::rows,
    },
    Family {
        generation: "olmo_2",
        arch: "olmo2",
        vocabs: &[crate::olmo_2::import::VOCAB],
        rows: crate::olmo_2::rows,
    },
    Family {
        generation: "olmo_3",
        arch: "olmo3",
        vocabs: &[crate::olmo_3::import::VOCAB],
        rows: crate::olmo_3::rows,
    },
    Family {
        generation: "phi_3",
        arch: "phi3",
        vocabs: &[crate::phi_3::import::VOCAB],
        rows: crate::phi_3::rows,
    },
    Family {
        generation: "mistral_3",
        arch: "mistral",
        vocabs: &[crate::mistral_3::import::VOCAB],
        rows: crate::mistral_3::rows,
    },
    Family {
        generation: "csm",
        arch: "",
        vocabs: &[crate::csm::import::VOCAB],
        rows: crate::csm::rows,
    },
];

const MODEL_TYPES: &[(&str, &str)] = &[
    ("llama", "llama"),
    ("qwen2", "qwen2"),
    ("qwen3", "qwen3"),
    ("qwen3_moe", "qwen3"),
    ("qwen3_5", "qwen3_5"),

    ("qwen3_5_text", "qwen3_5"),
    ("qwen3_5_moe", "qwen3_5"),
    ("qwen3_5_moe_text", "qwen3_5"),
    ("qwen3_vl", "qwen3_5"),
    ("qwen3_vl_text", "qwen3_5"),
    ("gemma2", "gemma2"),
    ("gemma3", "gemma3"),
    ("gemma3_text", "gemma3"),
    ("gemma3n", "gemma3n"),
    ("gemma3n_text", "gemma3n"),
    ("gemma4", "gemma4"),
    ("gemma4_text", "gemma4"),
    ("gptoss", "gptoss"),
    ("gpt_oss", "gptoss"),
    ("nemotron_h", "nemotron_h"),
    ("olmo2", "olmo2"),
    ("olmo3", "olmo3"),
    ("phi3", "phi3"),
    ("mistral", "mistral"),
    ("mistral3", "mistral"),
];

#[must_use]
pub fn arch_for_model_type(model_type: &str) -> Option<&'static str> {
    MODEL_TYPES
        .iter()
        .find(|(hf, _)| *hf == model_type)
        .map(|(_, arch)| *arch)
}

fn hf_family(model_type: &str) -> Option<&'static Family> {
    let arch = arch_for_model_type(model_type)?;
    FAMILIES
        .iter()
        .find(|f| !f.arch.is_empty() && f.arch == arch)
}

pub enum Vocabulary<'a> {

    Gguf(&'a Attributes),

    HuggingFace(&'a str),
}

pub fn ingest(vocabulary: &Vocabulary<'_>, names: &[&str]) -> Result<Vec<Ingest>, Error> {
    match vocabulary {
        Vocabulary::Gguf(attributes) => gguf_ingest(attributes, names),
        Vocabulary::HuggingFace(model_type) => hf_ingest(model_type, names),
    }
}

fn hf_ingest(model_type: &str, names: &[&str]) -> Result<Vec<Ingest>, Error> {
    let family = hf_family(model_type);
    let vocabs = family.map_or(&[][..], |f| f.vocabs);
    let respells = vocabs.iter().any(Vocab::respells);
    let generation = family.map_or("<generation>", |f| f.generation);
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        match vocabs.iter().find_map(|v| v.from_hf(name)) {
            Some(pie) => out.push(Ingest::Rename(pie)),
            None if respells => {
                return Err(Error::Contract(format!(
                    "`{model_type}` publishes `{name}`, and this build's table for it has no row -- so pie has no name of its own to store it under. The table respells at least one tensor, so passing the name through would leave the artifact half in each vocabulary. Add the row in `crates/model/src/{generation}/import.rs`"
                )));
            }
            None => out.push(Ingest::Rename((*name).to_string())),
        }
    }
    Ok(out)
}

pub fn gguf_ingest(attributes: &Attributes, names: &[&str]) -> Result<Vec<Ingest>, Error> {
    let architecture = attributes.architecture().unwrap_or_default();
    let pass = pass_for(architecture)?;
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        if (pass.derived)(name) {
            out.push(Ingest::Drop);
            continue;
        }
        let Some(renamed) = pass.vocab.from_gguf(name) else {
            return Err(Error::Contract(format!(
                "`{architecture}` GGUF ingest has no name for `{name}`; the map in \
                 `crates/model/src/*/import.rs` predates this checkpoint"
            )));
        };
        let regroup = (pass.regroup)(attributes, name);
        let folded = (pass.folded)(name);
        let stacked = (pass.stacked)(name);

        if u8::from(regroup.is_some()) + u8::from(folded.is_some()) + u8::from(stacked) > 1 {
            return Err(Error::Contract(format!(
                "`{architecture}` GGUF ingest wants more than one of regroup, unfold and \
                 unstack for `{name}`; `Ingest` states one transform per tensor, so this \
                 needs a composition the enum does not have"
            )));
        }
        out.push(match (regroup, folded, stacked) {
            (Some(heads), _, _) => Ingest::Unpermute {
                name: renamed,
                heads,
            },
            (_, Some(by), _) => Ingest::Debias { name: renamed, by },
            (_, _, true) => Ingest::Unstack { each: renamed },
            _ => Ingest::Rename(renamed),
        });
    }
    Ok(out)
}

pub fn gguf_rename(architecture: &str, names: &[&str]) -> Result<Vec<String>, Error> {
    let attributes = Attributes::from_pairs([(
        "general.architecture".to_string(),
        model_loader::checkpoint::Attribute::Text(architecture.to_string()),
    )]);
    gguf_ingest(&attributes, names)?
        .into_iter()
        .map(|ingest| match ingest {
            Ingest::Rename(name) | Ingest::Unpermute { name, .. } | Ingest::Debias { name, .. } => {
                Ok(name)
            }
            Ingest::Unstack { each } => Err(Error::Contract(format!(
                "`{each}` is a template, not a name: this tensor reaches the artifact as \
                 one per instance. Ask `gguf_ingest`"
            ))),
            Ingest::Drop => Err(Error::Contract(
                "a dropped tensor has no name; ask `gguf_ingest`".to_string(),
            )),
        })
        .collect()
}

#[must_use]
pub fn can_ingest_gguf(architecture: &str) -> bool {
    gguf_rename(architecture, &[]).is_ok()
}
