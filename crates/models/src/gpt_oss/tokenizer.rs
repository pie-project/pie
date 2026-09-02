//! gpt-oss tokenizer contract (stop tokens from Harmony).

use ::tokenizer::contract::Contract;

pub const CONTRACT: Contract = Contract {
    markers: &[chat_template::harmony::STOP_TOKENS],
    pinned: &[],
};
