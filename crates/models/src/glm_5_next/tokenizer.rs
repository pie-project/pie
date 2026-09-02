//! GLM-5's tokenizer contract: the stop-token markers the serving row reads.

use ::tokenizer::contract::Contract;

pub const CONTRACT: Contract = Contract {
    markers: &[chat_template::glm::STOP_TOKENS],
    pinned: &[],
};

/// The vision rows' contract: the stop markers plus the image delimiters
/// `media::Glm5Vision` wraps a span in.
pub const CONTRACT_VISION: Contract = Contract {
    markers: &[chat_template::glm::STOP_TOKENS, super::media::VISION_DELIMITERS],
    pinned: &[],
};
