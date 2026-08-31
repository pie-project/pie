//! What gpt-oss rows read from their vocabulary — Harmony's stop list,
//! cited from where it lives.

use ::tokenizer::contract::Contract;

pub const CONTRACT: Contract = Contract {
    markers: &[chat_template::harmony::STOP_TOKENS],
    pinned: &[],
};
