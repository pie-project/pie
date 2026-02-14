import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict


@dataclass
class ChatTemplate:
    """Structured chat template matching the WIT chat-template record.

    Each role's turn is rendered as: role_prefix + content + role_suffix.
    """

    start_token: str = ""
    stop_tokens: List[str] = field(default_factory=list)

    # Per-role wrapping
    role_prefixes: Dict[str, str] = field(default_factory=dict)
    role_suffixes: Dict[str, str] = field(default_factory=dict)

    # How system messages are handled
    system_handling: str = "standalone"  # "standalone" | "merge-with-user" | "bare-prepend"
    system_separator: str = ""

    # Header appended to start the model's response turn
    generation_header: str = ""

    # Reasoning/thinking wrapping
    thinking_prefix: str = ""
    thinking_suffix: str = ""

    # Tool call formatting
    tool_call_template: str = ""  # format string with {name}, {arguments}
    tool_calls_prefix: str = ""
    tool_calls_suffix: str = ""

    # Tool response formatting
    tool_response_role: str = ""
    tool_response_prefix: str = ""
    tool_response_suffix: str = ""

    def to_json(self) -> str:
        """Serialize to JSON for passing through pybindings."""
        d = asdict(self)
        # Convert dicts to list-of-tuples for WIT compatibility
        d["role_prefixes"] = list(d["role_prefixes"].items())
        d["role_suffixes"] = list(d["role_suffixes"].items())
        return json.dumps(d)


# =============================================================================
# Qwen 3 -- ChatML with standalone system
# =============================================================================

Qwen3Template = ChatTemplate(
    start_token='',
    stop_tokens=['<|im_end|>', '<|im_start|>', '<|endoftext|>'],
    role_prefixes={'system': '<|im_start|>system\n', 'user': '<|im_start|>user\n', 'assistant': '<|im_start|>assistant\n'},
    role_suffixes={'system': '<|im_end|>\n', 'user': '<|im_end|>\n', 'assistant': '<|im_end|>\n'},
    system_handling='standalone',
    generation_header='<|im_start|>assistant\n',
    thinking_prefix='<think>\n',
    thinking_suffix='</think>\n',
    tool_call_template='<tool_call>\\n{"name":"{name}","arguments": {arguments}}\\n</tool_call>',
    tool_calls_prefix='',
    tool_calls_suffix='',
    tool_response_role='user',
    tool_response_prefix='<tool_response>\n',
    tool_response_suffix='\n</tool_response>',
)

# =============================================================================
# Qwen 2.5 -- Same ChatML format as Qwen 3
# =============================================================================

Qwen2_5Template = ChatTemplate(
    start_token='',
    stop_tokens=['<|im_end|>', '<|endoftext|>'],
    role_prefixes={'system': '<|im_start|>system\n', 'user': '<|im_start|>user\n', 'assistant': '<|im_start|>assistant\n'},
    role_suffixes={'system': '<|im_end|>\n', 'user': '<|im_end|>\n', 'assistant': '<|im_end|>\n'},
    system_handling='standalone',
    generation_header='<|im_start|>assistant\n',
    tool_call_template='<tool_call>\\n{"name":"{name}","arguments": {arguments}}\\n</tool_call>',
    tool_calls_prefix='',
    tool_calls_suffix='',
    tool_response_role='user',
    tool_response_prefix='<tool_response>\n',
    tool_response_suffix='\n</tool_response>',
)

# =============================================================================
# Llama 3 -- Header-based format with standalone system
# =============================================================================

Llama3Template = ChatTemplate(
    start_token='',
    stop_tokens=['<|eot_id|>', '<|end_of_text|>'],
    role_prefixes={'system': '<|start_header_id|>system<|end_header_id|>\n', 'user': '<|start_header_id|>user<|end_header_id|>\n', 'assistant': '<|start_header_id|>assistant<|end_header_id|>\n', 'ipython': '<|start_header_id|>ipython<|end_header_id|>\n'},
    role_suffixes={'system': '<|eot_id|>\n', 'user': '<|eot_id|>\n', 'assistant': '<|eot_id|>\n', 'ipython': '<|eot_id|>\n'},
    system_handling='standalone',
    generation_header='<|start_header_id|>assistant<|end_header_id|>\n',
    thinking_prefix='<think>\n',
    thinking_suffix='</think>\n',
)

# =============================================================================
# OLMo 3 -- ChatML format (same structure as Qwen, standalone system)
# =============================================================================

Olmo3Template = ChatTemplate(
    start_token='',
    stop_tokens=['<|im_end|>'],
    role_prefixes={'system': '<|im_start|>system\n', 'user': '<|im_start|>user\n', 'assistant': '<|im_start|>assistant\n'},
    role_suffixes={'system': '<|im_end|>\n', 'user': '<|im_end|>\n', 'assistant': '<|im_end|>\n'},
    system_handling='standalone',
    generation_header='<|im_start|>assistant\n',
    thinking_prefix='<think>\n',
    thinking_suffix='</think>\n',
)

# =============================================================================
# Gemma 2 -- Turn-based, system merged with first user message
# =============================================================================

Gemma2Template = ChatTemplate(
    start_token='<bos>\n',
    stop_tokens=['<end_of_turn>', '<eos>'],
    role_prefixes={'user': '<start_of_turn>user\n', 'assistant': '<start_of_turn>model\n'},
    role_suffixes={'user': '<end_of_turn>\n', 'assistant': '<end_of_turn>\n'},
    system_handling='merge-with-user',
    system_separator='\n\n',
    generation_header='<start_of_turn>model\n',
)

# =============================================================================
# Gemma 3 -- Same format as Gemma 2
# =============================================================================

Gemma3Template = ChatTemplate(
    start_token='<bos>\n',
    stop_tokens=['<end_of_turn>', '<eos>'],
    role_prefixes={'user': '<start_of_turn>user\n', 'assistant': '<start_of_turn>model\n'},
    role_suffixes={'user': '<end_of_turn>\n', 'assistant': '<end_of_turn>\n'},
    system_handling='merge-with-user',
    system_separator='\n\n',
    generation_header='<start_of_turn>model\n',
)

# =============================================================================
# Mistral 3 -- INST format, system merged with first user message
# =============================================================================

Mistral3Template = ChatTemplate(
    start_token='<s>\n',
    stop_tokens=['</s>'],
    role_prefixes={'user': '[INST] ', 'assistant': ''},
    role_suffixes={'user': ' [/INST]', 'assistant': '</s>'},
    system_handling='merge-with-user',
    system_separator='\n\n',
    generation_header='',
)

# =============================================================================
# DeepSeek R1 -- Bare-prepend system, fullwidth tokens
# =============================================================================

R1Template = ChatTemplate(
    start_token='<｜begin▁of▁sentence｜>',
    stop_tokens=['<｜end▁of▁sentence｜>', '<|EOT|>'],
    role_prefixes={'user': '<｜User｜>', 'assistant': '<｜Assistant｜>'},
    role_suffixes={'user': '', 'assistant': '<｜end▁of▁sentence｜>'},
    system_handling='bare-prepend',
    generation_header='<｜Assistant｜><think>\n',
    tool_call_template='<｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\\n```json\\n{arguments}\\n```<｜tool▁call▁end｜>',
    tool_calls_prefix='<｜tool▁calls▁begin｜>',
    tool_calls_suffix='<｜tool▁calls▁end｜><｜end▁of▁sentence｜>',
    tool_response_role='tool',
    tool_response_prefix='<｜tool▁output▁begin｜>',
    tool_response_suffix='<｜tool▁output▁end｜>',
)

# =============================================================================
# GPT-OSS -- Channel-based format with developer role
# =============================================================================

GPTOSSTemplate = ChatTemplate(
    start_token='',
    stop_tokens=['<|endoftext|>', '<|return|>', '<|call|>'],
    role_prefixes={'system': '<|start|>system<|message|>', 'developer': '<|start|>developer<|message|>', 'user': '<|start|>user<|message|>', 'assistant': '<|start|>assistant<|channel|>final<|message|>'},
    role_suffixes={'system': '<|end|>', 'developer': '<|end|>', 'user': '<|end|>', 'assistant': '<|end|>'},
    system_handling='standalone',
    generation_header='<|start|>assistant',
    thinking_prefix='<|start|>assistant<|channel|>analysis<|message|>',
    thinking_suffix='<|end|>',
)
