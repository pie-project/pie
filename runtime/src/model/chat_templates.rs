//! Chat template definitions for all supported model architectures.
//!
//! Each template is a static configuration looked up by architecture name.

/// How system messages are placed in the prompt.
#[derive(Debug, Clone, PartialEq)]
pub enum SystemHandling {
    /// System message rendered as its own turn (with role prefix/suffix).
    Standalone,
    /// System content merged into the first user message.
    MergeWithUser,
    /// System content prepended raw before all turns (no wrapping).
    BarePrepend,
}

/// Structured chat template — describes how to format a conversation.
#[derive(Debug, Clone)]
pub struct ChatTemplate {
    /// Token prepended at the start of the sequence (BOS).
    pub start_token: &'static str,
    /// Tokens that signal end of generation.
    pub stop_tokens: &'static [&'static str],

    /// Per-role prefix: `(role, prefix)` pairs.
    pub role_prefixes: &'static [(&'static str, &'static str)],
    /// Per-role suffix: `(role, suffix)` pairs.
    pub role_suffixes: &'static [(&'static str, &'static str)],

    /// How system messages are handled.
    pub system_handling: SystemHandling,
    /// Separator between system and user content when merging.
    pub system_separator: &'static str,

    /// Header appended to start the model's response turn.
    pub generation_header: &'static str,

    /// Wraps reasoning content.
    pub thinking_prefix: &'static str,
    pub thinking_suffix: &'static str,

    /// Tool call formatting: format string with `{name}`, `{arguments}`.
    pub tool_call_template: &'static str,
    pub tool_calls_prefix: &'static str,
    pub tool_calls_suffix: &'static str,

    /// Tool response formatting.
    pub tool_response_role: &'static str,
    pub tool_response_prefix: &'static str,
    pub tool_response_suffix: &'static str,
}

impl ChatTemplate {
    /// Look up the role prefix for a given role name.
    pub fn prefix_for(&self, role: &str) -> &str {
        self.role_prefixes
            .iter()
            .find(|(r, _)| *r == role)
            .map(|(_, p)| *p)
            .unwrap_or("")
    }

    /// Look up the role suffix for a given role name.
    pub fn suffix_for(&self, role: &str) -> &str {
        self.role_suffixes
            .iter()
            .find(|(r, _)| *r == role)
            .map(|(_, s)| *s)
            .unwrap_or("")
    }
}

/// Look up a chat template by architecture name.
///
/// Returns `None` for unknown architectures.
pub fn lookup(arch_name: &str) -> Option<&'static ChatTemplate> {
    match arch_name {
        "qwen3" => Some(&QWEN3),
        "qwen2" => Some(&QWEN2_5),
        "llama3" | "l4ma" => Some(&LLAMA3),
        "olmo3" => Some(&OLMO3),
        "gemma2" => Some(&GEMMA2),
        "gemma3" => Some(&GEMMA3),
        "mistral3" => Some(&MISTRAL3),
        "r1" | "deepseek_v3" => Some(&R1),
        "gptoss" | "gpt_oss" => Some(&GPTOSS),
        "dummy" => Some(&DUMMY),
        _ => None,
    }
}

// =============================================================================
// Template Definitions
// =============================================================================

/// Qwen 3 — ChatML with standalone system.
static QWEN3: ChatTemplate = ChatTemplate {
    start_token: "",
    stop_tokens: &["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
    role_prefixes: &[("system", "<|im_start|>system\n"), ("user", "<|im_start|>user\n"), ("assistant", "<|im_start|>assistant\n")],
    role_suffixes: &[("system", "<|im_end|>\n"), ("user", "<|im_end|>\n"), ("assistant", "<|im_end|>\n")],
    system_handling: SystemHandling::Standalone,
    system_separator: "",
    generation_header: "<|im_start|>assistant\n",
    thinking_prefix: "<think>\n",
    thinking_suffix: "</think>\n",
    tool_call_template: "<tool_call>\n{\"name\":\"{name}\",\"arguments\": {arguments}}\n</tool_call>",
    tool_calls_prefix: "",
    tool_calls_suffix: "",
    tool_response_role: "user",
    tool_response_prefix: "<tool_response>\n",
    tool_response_suffix: "\n</tool_response>",
};

/// Qwen 2.5 — Same ChatML format, fewer stop tokens.
static QWEN2_5: ChatTemplate = ChatTemplate {
    start_token: "",
    stop_tokens: &["<|im_end|>", "<|endoftext|>"],
    role_prefixes: &[("system", "<|im_start|>system\n"), ("user", "<|im_start|>user\n"), ("assistant", "<|im_start|>assistant\n")],
    role_suffixes: &[("system", "<|im_end|>\n"), ("user", "<|im_end|>\n"), ("assistant", "<|im_end|>\n")],
    system_handling: SystemHandling::Standalone,
    system_separator: "",
    generation_header: "<|im_start|>assistant\n",
    thinking_prefix: "",
    thinking_suffix: "",
    tool_call_template: "<tool_call>\n{\"name\": \"{name}\", \"arguments\": {arguments}}\n</tool_call>",
    tool_calls_prefix: "",
    tool_calls_suffix: "",
    tool_response_role: "user",
    tool_response_prefix: "<tool_response>\n",
    tool_response_suffix: "\n</tool_response>",
};

/// Llama 3 — Header-based format with standalone system.
static LLAMA3: ChatTemplate = ChatTemplate {
    start_token: "",
    stop_tokens: &["<|eot_id|>", "<|end_of_text|>"],
    role_prefixes: &[("system", "<|start_header_id|>system<|end_header_id|>\n"), ("user", "<|start_header_id|>user<|end_header_id|>\n"), ("assistant", "<|start_header_id|>assistant<|end_header_id|>\n"), ("ipython", "<|start_header_id|>ipython<|end_header_id|>\n")],
    role_suffixes: &[("system", "<|eot_id|>\n"), ("user", "<|eot_id|>\n"), ("assistant", "<|eot_id|>\n"), ("ipython", "<|eot_id|>\n")],
    system_handling: SystemHandling::Standalone,
    system_separator: "",
    generation_header: "<|start_header_id|>assistant<|end_header_id|>\n",
    thinking_prefix: "<think>\n",
    thinking_suffix: "</think>\n",
    tool_call_template: "{\"name\":\"{name}\",\"parameters\": {arguments}}",
    tool_calls_prefix: "",
    tool_calls_suffix: "",
    tool_response_role: "ipython",
    tool_response_prefix: "",
    tool_response_suffix: "",
};

/// OLMo 3 — ChatML format, standalone system.
static OLMO3: ChatTemplate = ChatTemplate {
    start_token: "",
    stop_tokens: &["<|im_end|>"],
    role_prefixes: &[("system", "<|im_start|>system\n"), ("user", "<|im_start|>user\n"), ("assistant", "<|im_start|>assistant\n")],
    role_suffixes: &[("system", "<|im_end|>\n"), ("user", "<|im_end|>\n"), ("assistant", "<|im_end|>\n")],
    system_handling: SystemHandling::Standalone,
    system_separator: "",
    generation_header: "<|im_start|>assistant\n",
    thinking_prefix: "<think>\n",
    thinking_suffix: "</think>\n",
    tool_call_template: "",
    tool_calls_prefix: "",
    tool_calls_suffix: "",
    tool_response_role: "",
    tool_response_prefix: "",
    tool_response_suffix: "",
};

/// Gemma 2 — Turn-based, system merged with first user message.
static GEMMA2: ChatTemplate = ChatTemplate {
    start_token: "<bos>\n",
    stop_tokens: &["<end_of_turn>", "<eos>"],
    role_prefixes: &[("user", "<start_of_turn>user\n"), ("assistant", "<start_of_turn>model\n")],
    role_suffixes: &[("user", "<end_of_turn>\n"), ("assistant", "<end_of_turn>\n")],
    system_handling: SystemHandling::MergeWithUser,
    system_separator: "\n\n",
    generation_header: "<start_of_turn>model\n",
    thinking_prefix: "",
    thinking_suffix: "",
    tool_call_template: "",
    tool_calls_prefix: "",
    tool_calls_suffix: "",
    tool_response_role: "",
    tool_response_prefix: "",
    tool_response_suffix: "",
};

/// Gemma 3 — Same format as Gemma 2.
static GEMMA3: ChatTemplate = ChatTemplate {
    start_token: "<bos>\n",
    stop_tokens: &["<end_of_turn>", "<eos>"],
    role_prefixes: &[("user", "<start_of_turn>user\n"), ("assistant", "<start_of_turn>model\n")],
    role_suffixes: &[("user", "<end_of_turn>\n"), ("assistant", "<end_of_turn>\n")],
    system_handling: SystemHandling::MergeWithUser,
    system_separator: "\n\n",
    generation_header: "<start_of_turn>model\n",
    thinking_prefix: "",
    thinking_suffix: "",
    tool_call_template: "",
    tool_calls_prefix: "",
    tool_calls_suffix: "",
    tool_response_role: "",
    tool_response_prefix: "",
    tool_response_suffix: "",
};

/// Mistral 3 — INST format, system merged with first user.
static MISTRAL3: ChatTemplate = ChatTemplate {
    start_token: "<s>\n",
    stop_tokens: &["</s>"],
    role_prefixes: &[("user", "[INST] "), ("assistant", "")],
    role_suffixes: &[("user", " [/INST]"), ("assistant", "</s>")],
    system_handling: SystemHandling::MergeWithUser,
    system_separator: "\n\n",
    generation_header: "",
    thinking_prefix: "",
    thinking_suffix: "",
    tool_call_template: "",
    tool_calls_prefix: "",
    tool_calls_suffix: "",
    tool_response_role: "",
    tool_response_prefix: "",
    tool_response_suffix: "",
};

/// DeepSeek R1 — Bare-prepend system, fullwidth tokens.
static R1: ChatTemplate = ChatTemplate {
    start_token: "<｜begin▁of▁sentence｜>",
    stop_tokens: &["<｜end▁of▁sentence｜>", "<|EOT|>"],
    role_prefixes: &[("user", "<｜User｜>"), ("assistant", "<｜Assistant｜>")],
    role_suffixes: &[("user", ""), ("assistant", "<｜end▁of▁sentence｜>")],
    system_handling: SystemHandling::BarePrepend,
    system_separator: "",
    generation_header: "<｜Assistant｜><think>\n",
    thinking_prefix: "",
    thinking_suffix: "",
    tool_call_template: "<｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\n```json\n{arguments}\n```<｜tool▁call▁end｜>",
    tool_calls_prefix: "<｜tool▁calls▁begin｜>",
    tool_calls_suffix: "<｜tool▁calls▁end｜><｜end▁of▁sentence｜>",
    tool_response_role: "tool",
    tool_response_prefix: "<｜tool▁output▁begin｜>",
    tool_response_suffix: "<｜tool▁output▁end｜>",
};

/// GPT-OSS — Channel-based format with developer role.
static GPTOSS: ChatTemplate = ChatTemplate {
    start_token: "",
    stop_tokens: &["<|endoftext|>", "<|return|>", "<|call|>"],
    role_prefixes: &[("system", "<|start|>system<|message|>"), ("developer", "<|start|>developer<|message|>"), ("user", "<|start|>user<|message|>"), ("assistant", "<|start|>assistant<|channel|>final<|message|>")],
    role_suffixes: &[("system", "<|end|>"), ("developer", "<|end|>"), ("user", "<|end|>"), ("assistant", "<|end|>")],
    system_handling: SystemHandling::Standalone,
    system_separator: "",
    generation_header: "<|start|>assistant",
    thinking_prefix: "<|start|>assistant<|channel|>analysis<|message|>",
    thinking_suffix: "<|end|>",
    tool_call_template: "",
    tool_calls_prefix: "",
    tool_calls_suffix: "",
    tool_response_role: "",
    tool_response_prefix: "",
    tool_response_suffix: "",
};

/// Dummy — Simple ChatML fallback.
static DUMMY: ChatTemplate = ChatTemplate {
    start_token: "",
    stop_tokens: &["<|im_end|>", "<|endoftext|>"],
    role_prefixes: &[("system", "<|im_start|>system\n"), ("user", "<|im_start|>user\n"), ("assistant", "<|im_start|>assistant\n")],
    role_suffixes: &[("system", "<|im_end|>\n"), ("user", "<|im_end|>\n"), ("assistant", "<|im_end|>\n")],
    system_handling: SystemHandling::Standalone,
    system_separator: "",
    generation_header: "<|im_start|>assistant\n",
    thinking_prefix: "",
    thinking_suffix: "",
    tool_call_template: "",
    tool_calls_prefix: "",
    tool_calls_suffix: "",
    tool_response_role: "",
    tool_response_prefix: "",
    tool_response_suffix: "",
};

