# text-completion

A small chat-style text generation inferlet.

## Usage

Accepts a prompt and optional sampling parameters.

## Inputs

- `prompt`: The user message to complete.
- `system` (optional): System prompt.
- `max_tokens` (optional): Max tokens to generate.
- `temperature` (optional): Sampling temperature.
- `top_p` (optional): Nucleus sampling threshold.

## Output

A string containing the generated text. Thinking tags are left in the text when the model emits them.
