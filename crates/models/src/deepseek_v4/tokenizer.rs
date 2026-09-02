//! Tokenizer contract: the stop-token markers deepseek rows read from their vocabulary.

use ::tokenizer::contract::Contract;

pub const CONTRACT: Contract = Contract {
    markers: &[chat_template::deepseek::STOP_TOKENS],
    pinned: &[],
};
