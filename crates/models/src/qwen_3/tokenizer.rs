//! What qwen rows read from their vocabulary — the family's citations,
//! spelled once and cited from the template, the media front-end, and the
//! contracts below.

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

/// **THE SEVEN TOKENS THAT TELL A 3.8 TOKENIZER FROM ITS 3.6 TWIN.**
///
/// Qwen3.8-27B's artifact surface is Qwen3.6-27B's tensor for tensor — no
/// weight contract can tell them apart (see `qwen_3.rs`'s shadowed-twin
/// paragraph). The one artifact-visible delta besides the chat template is
/// this: 3.8's `tokenizer.json` declares seven audio/tts specials over
/// reserved slots inside the same 248 320-token vocab, and 3.6's does not.
/// The ids are upstream's own (`added_tokens` diff of the two files,
/// 2026-08-30), pinned so a `qwen38` row refuses a 3.6 artifact at boot
/// instead of serving it under a reading it was never trained for — and
/// they are the audio door's forward signal (next.md §E).
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
