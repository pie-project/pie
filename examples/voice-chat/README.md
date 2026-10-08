# voice-chat

One spoken turn of a voice conversation: the transcript in, a short speakable
reply streamed out, and the prompt's prefixes published so the next turn
prefills only what is new.

## What it does

A voice assistant sends the whole transcript every turn, and replaying it
through the model is what makes a phone assistant pause before it speaks.
This inferlet renders the transcript with the host chat template, looks the
prompt up in `PrefixCache` under the session's name, maps in the longest
published prefix (its KV pages, and on a hybrid model its recurrent state),
and prefills only the suffix past it. The prefill ends a chunk at the end of
the new user message, floored to a KV page, and publishes that one boundary:
the next turn's transcript extends this one, so it adopts everything up to
that message and prefills the rest. The entries a turn supersedes are retired
as it publishes, so a session holds one snapshot at a time and the rest of a
phone's few recurrent-state slots stay free for live work, instead of waiting
on the runtime's reclaim to pick which snapshots go.

Generation is a device-carried decode loop with temperature and nucleus
sampling. Only speakable text reaches stdout: a reasoning stripper removes
`<think>` blocks without ever emitting a partial tag, and with `think` off the
cue is followed by the model's own closed, empty thinking block so the reply
starts outside any reasoning region.

## Input

```json
{"messages": [{"role": "system", "content": "..."},
              {"role": "user", "content": "..."},
              {"role": "assistant", "content": "..."},
              {"role": "user", "content": "..."}],
 "session": "ios-voice-session", "max_tokens": 120,
 "temperature": 0.7, "top_p": 0.95, "think": false}
```

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `messages` | array, or that array as a JSON string | | The transcript so far, ending with the new user message |
| `text` | string | | A single user message, when there is no transcript |
| `system` | string | a short voice-assistant prompt | Prepended when the transcript has no leading system message |
| `session` | string | `voice-session` | Names the prefix index this conversation publishes to and adopts from |
| `max_tokens` | int | `120` | Maximum tokens generated this turn |
| `temperature` | float | `0.7` | `0` is greedy |
| `top_p` | float | `0.95` | Nucleus threshold |
| `think` | bool | `false` | Let the model reason first; the reasoning is stripped from stdout either way |
| `followup` | string, or an array of strings (also as a JSON string) | | Debug only: further user messages answered in the same process, so reuse can be seen from a shell |

## Output

stdout is the reply text as it is generated. The return value is one JSON
object:

```json
{"text": "...", "prompt_tokens": 113, "reused": 64, "new_prefill": 49,
 "generated": 14, "resumed": true, "note": ""}
```

`reused` is how many prompt tokens came from earlier turns' published state,
`new_prefill` how many this turn computed, `resumed` whether any prefix was
found, and `note` why a publish or a retirement was skipped, if one was. With
`followup` the object also carries `turns`, every turn's accounting.

## Run

`pie run` resolves curated examples by name once they are built:

```bash
cargo build --release --target wasm32-wasip2 -p voice-chat
pie run voice-chat -- --text "hello" --session demo
pie run voice-chat -- --messages '[{"role":"user","content":"hello"},{"role":"assistant","content":"Hi there."},{"role":"user","content":"How are you?"}]' --session demo
```

Every `pie run` boots a fresh engine, so a published prefix does not outlive
the command. To watch later turns reuse earlier ones within one engine:

```bash
pie run voice-chat -- --text "What is one good reason to run a language model on a phone?" --session demo --followup "Can you say that again in five words?"
pie run voice-chat -- --text "My name is Aarush." --session demo --followup '["What is my name?", "And what city is the Eiffel Tower in?", "Which country is that in?"]'
```
