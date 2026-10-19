# type-ahead-prefill-py

Prefill the prompt while the user is still typing it. The client sends the
draft after every keystroke; the inferlet prefills each finished word, so when
Enter is pressed only the last word, the end of the turn and the cue are left
to compute before the first token.

```bash
pie serve &
python examples/type-ahead-prefill-py/client.py             # type, Enter sends
python examples/type-ahead-prefill-py/client.py --compare   # scripted typing vs. pasting at Enter
```

`client.py` needs `pip install pie-client` and installs `main.py` on the
server itself.

On an M5 Max with Qwen3.5-0.8B, a 461-token prompt:

| | First token | Prompt tokens left at Enter |
| --- | --- | --- |
| Pasted at Enter | 20 ms | 461 |
| Prefilled while typing | 4–6 ms | 11 |

The gap grows with the model and the prompt: what is left at Enter stays
about one word, whatever the prompt's length.

## How it works

- The word being typed is held back: "Kyo" can become "Kyoto", and the
  tokenizer may split the two differently. Everything before the last space is
  prefilled.
- Cached slots are frozen, so before each word the working set is forked; the
  fork shares the frozen KV pages and copies any recurrent state. A backspace
  past what is cached resumes from the last checkpoint at or before the edit
  (up to 64 words back) and prefills only from there.
- The answer is the same as typing nothing and sending, except for the
  numerical difference of prefilling in word-sized passes instead of one,
  the same as any chunked prefill.

## Messages

| From | Message | Meaning |
| --- | --- | --- |
| client | `{"draft": "..."}` | The whole draft so far |
| client | `{"send": "..."}` | The final text; answer it |
| inferlet | `{"prefilled": n}` | Prompt tokens in the KV cache after a draft |
| inferlet | `{"ttft_ms": .., "tail": n}` | Time to the first token, and how many prompt tokens Enter still had to prefill |
| inferlet | `{"token": ".."}` | The answer, streamed |

| Input | Default | Meaning |
| --- | --- | --- |
| `max_tokens` | `128` | Most tokens the answer has |
| `system` | none | An optional system message, prefilled with the first keystroke |
