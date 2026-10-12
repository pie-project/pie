# RFC 0001: OpenAI-compatible surface for Pie

- **Status:** Draft
- **Scope:** How harnesses (OpenCode, Aider, our GUI) reach Pie through the OpenAI HTTP API.

## Summary

Pie already exposes OpenAI-compatible routes through the built-in
`compat-openai` inferlet. This RFC keeps that surface, closes the remaining
gaps, and defines how Pie-specific features reach clients. It does not add a
second front door.

## Current state

Source: `crates/builtins/inferlets/compat-openai` in the main Pie checkout. It
is not in this repository, so the findings below need to be checked against
that source before merge.

Routes, one program behind three paths:

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/responses`

Supported on chat completions: `messages` (system, user, assistant, tool; text
parts), `tools`, `tool_choice`, `response_format` (`text`, `json_object`,
`json_schema`), `stream`, `stream_options.include_usage`, `temperature`,
`top_p`, `top_k`, `max_tokens` / `max_completion_tokens`, `stop`, `seed`.

Refused with 400: `n > 1`, `logprobs`, non-text content parts.

Tool calls are returned to the client (`finish_reason: "tool_calls"`). The
client runs the tool and sends the result back. The engine does not run a tool
loop.

`model` is echoed as the loaded model's name. There is one model per engine.

## Decisions

1. **One surface.** `compat-openai` is the OpenAI front door. The outside
   adapter (PR #926) is superseded and should be closed.
2. **Client-executed tools are the standard mode.** This matches how OpenAI
   clients already work, and it is implemented.
3. **Server-side agent loops are a separate, explicit mode.** An inferlet that
   runs its own tool loop should be exposed under its own name, not overloaded
   onto `tools`. This is future work.
4. **Pie-specific parameters go in a namespaced `pie` object** (for example
   `"pie": {"beams": 4}`), passed through the OpenAI SDKs' `extra_body`. Not
   yet implemented. Unknown keys must stay rejected until it exists.
5. **Model selection by inferlet** is a proposal, not current behaviour.
   Today `model` names the loaded model. Selecting an inferlet would mean
   listing each one in `/v1/models`, generated from its `Pie.toml` parameters.

## Open questions

- **Prefix caching across turns.** Requests are stateless, so clients resend
  history. Pie's advantage depends on reusing KV cache across those turns. We
  have not verified that this happens through the OpenAI path.
- **Tool-call quality on small models.** Whether the 0.8B model produces valid
  tool calls is untested.
- **Extension versioning.** How the `pie` object is versioned and documented.
- **Refused features.** Whether `logprobs` and `n > 1` should be supported
  later or stay refused.

## Verification plan

- Run the official OpenAI Python SDK against the endpoint for each supported
  feature. Compatibility is measured, not assumed.
- Test streaming, tool round-trips (call, result, final answer), and JSON-schema
  output.
- Check cache reuse on repeated multi-turn requests.

## Non-goals

- A second OpenAI-compatible server outside the engine.
- Changing the wire format of Pie's native client protocol.
