# pie-openai-adapter

An OpenAI-compatible HTTP adapter in front of the Pie engine. Any harness that
can point at a custom OpenAI base URL can talk to Pie through it, without
knowing Pie's WebSocket protocol.

> **Status: draft.** Verified against a local engine (Qwen3.5-0.8B on Metal)
> with the `chat-completion` inferlet. See *Relation to the built-in endpoint*
> before deciding whether this package should be merged.

## Layers

The system is split into three layers. Each depends only on the one below it.

```
GUI / agent harness      OpenCode, Aider, our own GUI
        │  OpenAI HTTP API (/v1/chat/completions, /v1/models)
        ▼
pie-openai-adapter       this package: translation only, no model knowledge
        │  Pie client protocol (WebSocket, msgpack): launch_process, Stdout/Return
        ▼
Pie engine               crates/ (gateway, worker, runtime, engines). Unchanged here.
```

- **Engine** owns models, KV cache, scheduling, and GPU kernels. It knows
  nothing about OpenAI.
- **Adapter** owns the mapping: OpenAI messages to an inferlet input, and
  inferlet output back to OpenAI chunks. It knows nothing about kernels.
- **Harness / GUI** knows only the OpenAI API. Swapping the engine or the
  harness does not touch the other side.

## Modules

| Module | Responsibility | Depends on |
| --- | --- | --- |
| `translate.py` | Pure mapping and validation of requests. No I/O. | nothing |
| `backend.py` | The only module that talks to Pie. Defines the `Backend` protocol. | `pie_client` |
| `server.py` | HTTP routes, SSE framing, OpenAI error shapes. | `aiohttp`, `Backend` protocol |
| `__main__.py` | CLI wiring. | all of the above |

The server depends on the `Backend` protocol, not on `PieClient`. Tests use a
fake backend, so the HTTP behaviour is tested without an engine.

## Usage

Start the engine, then the adapter:

```bash
pie serve                       # gateway on ws://127.0.0.1:8080
pie inferlet install <chat-completion artifact>   # the inferlet the adapter launches

pip install -e sdk/adapter/openai
pie-openai-adapter --pie-url ws://127.0.0.1:8080 --port 8000
```

Test it:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"pie","messages":[{"role":"user","content":"Say hello"}],"max_tokens":32}'
```

Add `"stream": true` for server-sent events.

Point a harness at `http://127.0.0.1:8000/v1` with any non-empty API key.

## Request mapping

| OpenAI field | Inferlet input | Notes |
| --- | --- | --- |
| `system` messages | `system` | Joined with newlines. Default is the inferlet's default system prompt. |
| last `user` message | `prompt` | Must be the last message. |
| earlier turns | folded into `prompt` | Rendered as `role: text` lines. See limitations. |
| `max_tokens` / `max_completion_tokens` | `max_tokens` | Default 256. Must be positive. |
| `temperature` | `temperature` | Default 0.6. Non-negative. |
| `top_p` | `top_p` | Default 0.95. In (0, 1]. |

Rejected with HTTP 400 in the OpenAI error shape: `tools`, `n != 1`, non-text
content parts, unknown roles, and out-of-range sampling parameters.

## Behaviour

- **Streaming:** each `Stdout` event from the inferlet becomes one
  `chat.completion.chunk`. The final `Return` value is used only when nothing
  was streamed, so text is never emitted twice.
- **Errors before output:** HTTP 502 with `{"error": {...}}`.
- **Errors mid-stream:** the error is sent as a final SSE event, then `[DONE]`,
  because the headers have already gone out.
- **Client disconnect:** the adapter terminates the inferlet process so it does
  not keep using the GPU.
- **Connections:** one engine WebSocket per request. This isolates failures
  between requests at the cost of a connect per request. Pooling is a possible
  follow-up.

## Limitations

- **No tool calling.** Requests with `tools` are rejected. Tool support needs
  an inferlet built on `tools.wit`, not a change here.
- **Multi-turn is lossy.** The `chat-completion` inferlet takes one prompt and
  one system string, so history is folded into text rather than rendered with
  the model's chat template.
- **Reasoning tags pass through.** Qwen3.5 emits `<think>…</think>`; the
  adapter does not strip or parse it.
- **Single model.** The `model` field is echoed, not used for routing.
- **Single-turn engine path.** Each request launches a new process.

## Relation to the built-in endpoint

The engine already serves `/v1/chat/completions` and `/v1/models` on its own
port, through the built-in `compat-openai@0.1.0` inferlet. Checked locally
against the same engine:

- Streaming works and emits real token deltas.
- Tool requests are accepted but ignored: the response is plain text, with no
  `tool_calls`.

So this adapter duplicates an endpoint the engine already exposes. Its only
differences are that it runs the `chat-completion` inferlet and rejects tool
requests explicitly instead of ignoring them. Whether to keep it, or to extend
`compat-openai` with tool support instead, is the open decision for this PR.
The `compat-openai` source is not in this repository, so that extension would
happen elsewhere.

## Tests

```bash
pip install -e "sdk/adapter/openai[test]"
pytest sdk/adapter/openai/tests
```

The tests cover request translation, every validation rule, both response
modes, the error shapes, and the health and model routes. They do not need an
engine. The end-to-end path was checked by hand (see the PR description).
