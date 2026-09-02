//! Gemma vocabulary constants: image delimiters and tokenizer contracts.

use ::tokenizer::contract::Contract;

pub const IMAGE_PREFIX: &str = "<|image>";
pub const IMAGE_PAD: &str = "<|image|>";
pub const IMAGE_SUFFIX: &str = "<image|>";

/// The triple `media::Gemma4Vision` wraps a span in.
pub const VISION_DELIMITERS: &[&str] = &[IMAGE_PREFIX, IMAGE_PAD, IMAGE_SUFFIX];

pub const CONTRACT: Contract = Contract {
    markers: &[chat_template::gemma::STOP_TOKENS],
    pinned: &[],
};

pub const CONTRACT_VISION: Contract = Contract {
    markers: &[chat_template::gemma::STOP_TOKENS, VISION_DELIMITERS],
    pinned: &[],
};
