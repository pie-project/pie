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

pub mod deepseek_v4;
pub mod gemma_4;
pub mod glm_5;
pub mod gpt_oss;
pub mod kimi_k3;
pub mod qwen_3_5;

mod metadata;
pub use metadata::ModelMetadata;
