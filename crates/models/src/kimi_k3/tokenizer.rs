use ::tokenizer::contract::Contract;

pub const CONTRACT: Contract = Contract {
    markers: &[chat_template::kimi::STOP_TOKENS],
    pinned: &[],
};

/// The released Kimi-K3: XTML turns closed by `<|end_of_msg|>`.
pub const CONTRACT3: Contract = Contract {
    markers: &[chat_template::kimi3::STOP_TOKENS],
    pinned: &[],
};
