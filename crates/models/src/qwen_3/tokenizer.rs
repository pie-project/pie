//! Vocabulary constants shared by the qwen template, media, and contracts.

use ::tokenizer::contract::Contract;

pub const IM_END: &str = "<|im_end|>";
pub const END_OF_TEXT: &str = "<|endoftext|>";

/// The stop list `template::chatml` interrupts on.
pub const STOP_TOKENS: &[&str] = &[IM_END, END_OF_TEXT];

pub const VISION_START: &str = "<|vision_start|>";
pub const IMAGE_PAD: &str = "<|image_pad|>";
pub const VISION_END: &str = "<|vision_end|>";

/// The triple `media::Qwen35Vision` wraps a span in — prefix, placeholder,
/// suffix, the placeholder being the reserved pad the run scan keys on.
pub const VISION_DELIMITERS: &[&str] = &[VISION_START, IMAGE_PAD, VISION_END];

/// The seven audio/tts specials that distinguish a 3.8 tokenizer from its
/// weight-identical 3.6 twin; used to reject a 3.6 artifact under a 3.8 row.
pub const AUDIO_SPECIALS: &[(&str, u32)] = &[
    ("<|audio_start|>", 248_070),
    ("<|audio_end|>", 248_071),
    ("<tts_pad>", 248_072),
    ("<tts_text_bos>", 248_073),
    ("<tts_text_eod>", 248_074),
    ("<tts_text_bos_single>", 248_075),
    ("<|audio_pad|>", 248_076),
];

pub const CONTRACT: Contract = Contract {
    markers: &[STOP_TOKENS],
    pinned: &[],
};

pub const CONTRACT_VISION: Contract = Contract {
    markers: &[STOP_TOKENS, VISION_DELIMITERS],
    pinned: &[],
};

pub const CONTRACT_38: Contract = Contract {
    markers: &[STOP_TOKENS],
    pinned: AUDIO_SPECIALS,
};

pub const CONTRACT_38_VISION: Contract = Contract {
    markers: &[STOP_TOKENS, VISION_DELIMITERS],
    pinned: AUDIO_SPECIALS,
};
