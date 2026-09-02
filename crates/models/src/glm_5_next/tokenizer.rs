//! GLM-5's tokenizer contract: the stop-token markers the serving row reads.

use ::tokenizer::contract::Contract;

pub const CONTRACT: Contract = Contract {
    markers: &[chat_template::glm::STOP_TOKENS],
    pinned: &[],
};
