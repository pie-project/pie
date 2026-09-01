//! What glm rows read from their vocabulary — the stop list
//! `template::instruct` interrupts on, spelled once and cited from both.

use ::tokenizer::contract::Contract;

pub const IM_END: &str = "<|im_end|>";
pub const END_OF_TEXT: &str = "<|endoftext|>";
pub const USER: &str = "<|user|>";
pub const ASSISTANT: &str = "<|assistant|>";

/// GLM's ChatML reading stops on the role markers too: upstream's template
/// ends a turn at the next role header, not only at `<|im_end|>`.
pub const STOP_TOKENS: &[&str] = &[IM_END, END_OF_TEXT, USER, ASSISTANT];

pub const CONTRACT: Contract = Contract {
    markers: &[STOP_TOKENS],
    pinned: &[],
};
