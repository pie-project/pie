pub mod deployment;

#[cfg(feature = "chat")]
pub mod instruct;

pub mod shared;

pub mod manifest;

pub mod catalog;

pub mod encoding;

#[cfg(feature = "contract")]
pub mod boot;
#[cfg(feature = "contract")]
pub mod contract;

#[cfg(feature = "contract")]
pub mod ingest;
#[cfg(feature = "chat")]
pub mod multimodal;

pub mod csm;
pub mod deepseek_r1;
pub mod deepseek_v4;
pub mod gemma_2;
pub mod gemma_3;
pub mod gemma_3n;
pub mod gemma_4;
pub mod glm_5;
pub mod gpt_oss;
pub mod kimi_k2;
pub mod kimi_k3;
pub mod llama_2;
pub mod llama_3;
pub mod mistral_3;
pub mod nemotron_h;
pub mod olmo_2;
pub mod olmo_3;
pub mod phi_3;
pub mod qwen_2;
pub mod qwen_3;
pub mod qwen_3_5;

#[cfg(feature = "test-rows")]
pub mod test_rows;

mod metadata;
pub use metadata::ModelMetadata;
