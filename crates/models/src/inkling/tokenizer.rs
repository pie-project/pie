use ::tokenizer::contract::Contract;

pub const CONTRACT: Contract = Contract {
    markers: &[
        chat_template::inkling::STOP_TOKENS,
        chat_template::inkling::MARKERS,
    ],
    pinned: &[],
};
