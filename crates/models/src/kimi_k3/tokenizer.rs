//! What kimi rows read from their vocabulary — the grammar's stop list,
//! cited from where it lives.

use ::tokenizer::contract::Contract;

pub const CONTRACT: Contract = Contract {
    markers: &[chat_template::kimi::STOP_TOKENS],
    pinned: &[],
};
