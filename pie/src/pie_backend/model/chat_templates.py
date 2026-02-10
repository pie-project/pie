from dataclasses import dataclass
from typing import List


@dataclass
class ChatTemplate:
    template_type: str
    template: str
    stop_tokens: List[str]


_R1_TEMPLATE_CONTENT = """{%- if not add_generation_prompt is defined -%}
	{%- set add_generation_prompt = false -%}
{%- endif -%}
{%- if not begin_of_sequence is defined -%}
	{%- set begin_of_sequence = false -%}
{%- endif -%}
{%- if begin_of_sequence -%}
	{%- set bos_token = "<｜begin▁of▁sentence｜>" -%}
{%- else -%}
    {%- set bos_token = "" -%}
{%- endif -%}
{%- set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt="") -%}
{%- for message in messages -%}
	{%- if message["role"] == "system" -%}
		{%- set ns.system_prompt = message["content"] -%}
	{%- endif -%}
{%- endfor -%}
{{- bos_token -}}
{{- ns.system_prompt -}}
{%- for message in messages -%}
	{%- if message["role"] == "user" -%}
		{%- set ns.is_tool = false -%}
		{{- "<｜User｜>" + message["content"] -}}
	{%- endif -%}
	{%- if message["role"] == "assistant" and message["content"] is none -%}
		{%- set ns.is_tool = false -%}
		{%- for tool in message["tool_calls"] -%}
			{%- if not ns.is_first -%}
				{{- "<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>" + tool["type"] + "<｜tool▁sep｜>" + tool["function"]["name"] + "\\n" + "```json" + "\\n" + tool["function"]["arguments"] + "\\n" + "```" + "<｜tool▁call▁end｜>" -}}
				{%- set ns.is_first = true -%}
			{%- else -%}
				{{- "\\n" + "<｜tool▁call▁begin｜>" + tool["type"] + "<｜tool▁sep｜>" + tool["function"]["name"] + "\\n" + "```json" + "\\n" + tool["function"]["arguments"] + "\\n" + "```" + "<｜tool▁call▁end｜>" -}}
				{{- "<｜tool▁calls▁end｜><｜end▁of▁sentence｜>" -}}
			{%- endif -%}
		{%- endfor -%}
	{%- endif -%}
	{%- if message["role"] == "assistant" and message["content"] is not none -%}
		{%- if ns.is_tool -%}
			{{- "<｜tool▁outputs▁end｜>" + message["content"] + "<｜end▁of▁sentence｜>" -}}
			{%- set ns.is_tool = false -%}
		{%- else -%}
			{%- set content = message["content"] -%}
			{%- if "</think>" in content -%}
				{%- set content = content.split("</think>")[-1] -%}
			{%- endif -%}
			{{- "<｜Assistant｜>" + content + "<｜end▁of▁sentence｜>" -}}
		{%- endif -%}
	{%- endif -%}
	{%- if message["role"] == "tool" -%}
		{%- set ns.is_tool = true -%}
		{%- if ns.is_output_first -%}
			{{- "<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>" + message["content"] + "<｜tool▁output▁end｜>" -}}
			{%- set ns.is_output_first = false -%}
		{%- else -%}
			{{- "\\n<｜tool▁output▁begin｜>" + message["content"] + "<｜tool▁output▁end｜>" -}}
		{%- endif -%}
	{%- endif -%}
{%- endfor -%}
{%- if ns.is_tool -%}
	{{- "<｜tool▁outputs▁end｜>" -}}
{%- endif -%}
{%- if add_generation_prompt and not ns.is_tool -%}
	{{- "<｜Assistant｜><think>\\n" -}}
{%- endif -%}
"""

R1Template = ChatTemplate(
    template_type="minijinja",
    template=_R1_TEMPLATE_CONTENT,
    stop_tokens=["<｜end▁of▁sentence｜>", "<|EOT|>"],
)

_GPT_OSS_TEMPLATE_CONTENT = """
{#- Simplified Template that Removed Tool Calls to Work With MiniJinja #}
{#- Main Template Logic ================================================= #}
{#- Set defaults #}
{#- Render system message #}
{{- "<|start|>system<|message|>" }}
{%- if model_identity is not defined %}
    {%- set model_identity = "You are ChatGPT, a large language model trained by OpenAI." %}
{%- endif %}
{{- model_identity + "\\n" }}
{{- "Knowledge cutoff: 2024-06\\n" }}
{{- "Current date: %2025-09-24\\n\\n" }}
{%- if reasoning_effort is not defined %}
    {%- set reasoning_effort = "medium" %}
{%- endif %}
{{- "Reasoning: " + reasoning_effort + "\\n\\n" }}
{{- "# Valid channels: analysis, commentary, final. Channel must be included for every message." }}
{{- "<|end|>" }}
{#- Extract developer message #}
{%- if messages[0].role == "developer" or messages[0].role == "system" %}
    {%- set developer_message = messages[0].content %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set developer_message = "" %}
    {%- set loop_messages = messages %}
{%- endif %}
{#- Render developer message #}
{%- if developer_message %}
    {{- "<|start|>developer<|message|>" }}
    {%- if developer_message %}
        {{- "# Instructions\\n\\n" }}
        {{- developer_message }}
        {{- "\\n\\n" }}
    {%- endif %}
    {{- "<|end|>" }}
{%- endif %}
{#- Render messages #}
{%- for message in loop_messages -%}
    {#- At this point only assistant/user messages should remain #}
    {%- if message.role == \'assistant\' -%}
        {#- Checks to ensure the messages are being passed in the format we expect #}
        {%- if "content" in message %}
            {%- if "<|channel|>analysis<|message|>" in message.content or "<|channel|>final<|message|>" in message.content %}
                {{- raise("You have passed a message containing <|channel|> tags in the content field. Instead of doing this, you should pass analysis messages (the string between \'<|message|>\' and \'<|end|>\') in the \'thinking\' field, and final messages (the string between \'<|message|>\' and \'<|end|>\') in the \'content\' field.") }}
            {%- endif %}
        {%- endif %}
        {%- if "thinking" in message %}
            {%- if "<|channel|>analysis<|message|>" in message.thinking or "<|channel|>final<|message|>" in message.thinking %}
                {{- raise("You have passed a message containing <|channel|> tags in the thinking field. Instead of doing this, you should pass analysis messages (the string between \'<|message|>\' and \'<|end|>\') in the \'thinking\' field, and final messages (the string between \'<|message|>\' and \'<|end|>\') in the \'content\' field.") }}
            {%- endif %}
        {%- endif %}
        {%- if loop.last and not add_generation_prompt %}
            {#- Only render the CoT if the final turn is an assistant turn and add_generation_prompt is false #}
            {#- This is a situation that should only occur in training, never in inference. #}
            {%- if "thinking" in message %}
                {{- "<|start|>assistant<|channel|>analysis<|message|>" + message.thinking + "<|end|>" }}
            {%- endif %}
            {#- <|return|> indicates the end of generation, but <|end|> does not #}
            {#- <|return|> should never be an input to the model, but we include it as the final token #}
            {#- when training, so the model learns to emit it. #}
            {{- "<|start|>assistant<|channel|>final<|message|>" + message.content + "<|return|>" }}
        {%- else %}
            {#- CoT is dropped during all previous turns, so we never render it for inference #}
            {{- "<|start|>assistant<|channel|>final<|message|>" + message.content + "<|end|>" }}
        {%- endif %}
    {%- elif message.role == \'user\' -%}
        {{- "<|start|>user<|message|>" + message.content + "<|end|>" }}
    {%- endif -%}
{%- endfor -%}
{#- Generation prompt #}
{%- if add_generation_prompt -%}
<|start|>assistant
{%- endif -%}
"""

GPTOSSTemplate = ChatTemplate(
    template_type="ninja",
    template=_GPT_OSS_TEMPLATE_CONTENT,
    stop_tokens=["<|endoftext|>", "<|return|>", "<|call|>"],
)

_LLAMA_3_TEMPLATE_CONTENT = """
{%- for m in messages %}
{%- set hdr = (
    "system" if m.role=="system" else
    "user" if m.role=="user" else
    "assistant" if m.role=="assistant" else
    "ipython" if m.role=="tool" else "user"
) -%}
<|start_header_id|>{{ hdr }}<|end_header_id|>{{ "\\n" }}
{%- if m.role == "assistant" and m.reasoning_content %}<think>
{{ m.reasoning_content | trim }}
</think>
{%- endif -%}
{%- if m.role == "assistant" and m.tool_calls %}
{%- for tc in m.tool_calls %}
{"name":"{{ tc.name }}","parameters": {{ tc.arguments | tojson }}}
{%- endfor %}
{%- else %}
{{ m.content }}
{%- endif %}
<|eot_id|>
{%- endfor %}
{%- if add_generation_prompt and (messages | length == 0 or (messages | last).role != "assistant") %}<|start_header_id|>assistant<|end_header_id|>
{%- endif %}
"""

Llama3Template = ChatTemplate(
    template_type="minijinja",
    template=_LLAMA_3_TEMPLATE_CONTENT,
    stop_tokens=["<|eot_id|>", "<|end_of_text|>"],
)

_QWEN_2_5_TEMPLATE_CONTENT = """{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\\\\\"name\\\\\\\": <function-name>, \\\\\\\"arguments\\\\\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
    {%- else %}
        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\\n<tool_call>\\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- message.content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
{%- endif %}
"""

Qwen2_5Template = ChatTemplate(
    template_type="minijinja",
    template=_QWEN_2_5_TEMPLATE_CONTENT,
    stop_tokens=["<|im_end|>", "<|endoftext|>"],
)

_QWEN_3_TEMPLATE_CONTENT = """
{%- for m in messages %}
{%- if m.role == "user" %}
<|im_start|>user
{{ m.content }}<|im_end|>
{%- elif m.role == "system" %}<|im_start|>system
{{ m.content }}<|im_end|>
{%- elif m.role == "assistant" %}
<|im_start|>assistant
{%- if m.reasoning_content %}<think>
{{ m.reasoning_content | trim }}
</think>
{%- endif -%}
{%- if m.tool_calls %}
{%- for tc in m.tool_calls %}
<tool_call>
{"name":"{{ tc.name }}","arguments": {{ tc.arguments | tojson }}}
</tool_call>
{%- endfor %}
{%- else %}
{{ m.content }}
{%- endif -%}<|im_end|>
{%- elif m.role == "tool" %}<|im_start|>user
<tool_response>
{{ m.content }}
</tool_response><|im_end|>
{%- else %}
<|im_start|>user
{{ m.content }}<|im_end|>
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
<|im_start|>assistant
{%- endif %}
"""

Qwen3Template = ChatTemplate(
    template_type="minijinja",
    template=_QWEN_3_TEMPLATE_CONTENT,
    stop_tokens=["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
)

# Gemma 2 chat template
# Note: Gemma 2 doesn't support system messages natively.
# This template prepends system content to the first user message.
_GEMMA_2_TEMPLATE_CONTENT = """
{%- set system_message = "" -%}
{%- for m in messages -%}
{%- if m.role == "system" -%}
{%- set system_message = m.content -%}
{%- endif -%}
{%- endfor -%}
<bos>
{%- for m in messages -%}
{%- if m.role == "user" -%}
<start_of_turn>user
{%- if loop.first and system_message -%}
{{ system_message }}

{{ m.content }}<end_of_turn>
{%- else -%}
{{ m.content }}<end_of_turn>
{%- endif -%}
{%- elif m.role == "assistant" -%}
<start_of_turn>model
{{ m.content }}<end_of_turn>
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
<start_of_turn>model
{%- endif -%}
"""

Gemma2Template = ChatTemplate(
    template_type="minijinja",
    template=_GEMMA_2_TEMPLATE_CONTENT,
    stop_tokens=["<end_of_turn>", "<eos>"],
)

# Gemma 3 chat template
# Same format as Gemma 2 - uses the same template structure
Gemma3Template = ChatTemplate(
    template_type="minijinja",
    template=_GEMMA_2_TEMPLATE_CONTENT,  # Same template as Gemma 2
    stop_tokens=["<end_of_turn>", "<eos>"],
)

# Mistral 3 (Ministral) chat template
# Uses standard [INST] format with system messages supported
_MISTRAL_3_TEMPLATE_CONTENT = """
{%- set system_message = "" -%}
{%- for m in messages -%}
{%- if m.role == "system" -%}
{%- set system_message = m.content -%}
{%- endif -%}
{%- endfor -%}
<s>
{%- for m in messages -%}
{%- if m.role == "user" -%}
[INST] {% if loop.first and system_message %}{{ system_message }}

{% endif %}{{ m.content }} [/INST]
{%- elif m.role == "assistant" -%}
{{ m.content }}</s>
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt and (messages | length == 0 or (messages | last).role != "assistant") -%}
{%- endif -%}
"""

Mistral3Template = ChatTemplate(
    template_type="minijinja",
    template=_MISTRAL_3_TEMPLATE_CONTENT,
    stop_tokens=["</s>"],
)


# Olmo3 chat template
# FIXED: Uses ChatML format matching HuggingFace reference
_OLMO3_TEMPLATE_CONTENT = """
{% for m in messages %}
{% if m.role == "system" %}<|im_start|>system
{{ m.content }}<|im_end|>
{% elif m.role == "user" %}<|im_start|>user
{{ m.content }}<|im_end|>
{% elif m.role == "assistant" %}<|im_start|>assistant
{% if m.reasoning_content %}<think>
{{ m.reasoning_content | trim }}
</think>
{% endif -%}
{{ m.content }}<|im_end|>
{% endif %}
{% endfor %}
{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}
"""

Olmo3Template = ChatTemplate(
    template_type="minijinja",
    template=_OLMO3_TEMPLATE_CONTENT,
    stop_tokens=["<|im_end|>"],
)

