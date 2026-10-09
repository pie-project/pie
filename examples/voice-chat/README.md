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
sampling. Only speakable text reaches stdout: a reasoning stripper separates
`<think>` blocks from the reply without ever emitting a partial tag, and with
`think` off the cue is followed by the model's own closed, empty thinking
block so the reply starts outside any reasoning region.

With `think` on, the cue is followed by the block's opening (`<think>` and a
newline, as the model's template writes it for `enable_thinking: true`), so
every thinking turn reasons and generation starts inside the block. The
reasoning is streamed as session messages, never to stdout. The program, not
the model, decides when thinking ends: the first decode loop is sized to stop
exactly where `thinking_budget` runs out. If the block is still open there,
the program prefills `"\n</think>\n\n"` and decodes the answer from that point
with what is left of `max_tokens`, at least 64 tokens; if the model closed it
sooner, the answer simply carries on in a second loop. A loop that runs its
exact count leaves the KV cache and the recurrent state holding precisely the
tokens it returned (one stopped early would leave run-ahead fires folding
discarded tokens into the state), so the continuation picks up where it left
off. A reply follows the reasoning even when a small model would have
reasoned until it ran out of tokens. The one exception is a model that ends
the turn itself, with its stop token, before the block closes: everything it
generated was reasoning, so `text` is empty and `note` says why, rather than
the reasoning standing in for the reply.

Without a `thinking_budget` the reasoning gets two thirds of `max_tokens`, at
most 256 tokens. Qwen3.5-0.8B practically never closes the block on its own,
so a thinking turn spends its whole budget before the first word of the
answer; at a phone's 60 to 70 tokens a second, 256 tokens is about four
seconds.

### Repetition loops

A small model can fall into a loop: asked to plan a three-day trip with
thinking on, Qwen3.5-0.8B has repeated the same three paragraphs until
`max_tokens` ran out, twenty seconds of it on a phone, and its reasoning
loops the same way until the thinking budget is spent. Two defenses work on
a *stretch* of the reply: the reasoning block the prompt opened, or the
answer after it. They are kept apart because an answer is expected to
restate what the reasoning settled on, and that is not a loop.

- **Repetition penalty, on the device, in the answer.** Every pass that
  samples answer tokens reads a vocabulary-sized histogram of the tokens the
  answer has generated, and divides the logit of each one already seen by
  `repetition_penalty` when it is positive (multiplies it when negative),
  before temperature and truncation, so greedy decoding at temperature 0 is
  still an exact argmax, of the penalized logits. The decode loop carries
  the histogram as a channel, one more count per token, the way the compat
  server's sampler does. Prompt tokens are never counted, so the prompt's
  prefill chunks carry no histogram (the answer has no tokens yet, and an
  empty one penalizes nothing); the pass that resumes a turn after the
  thinking budget gets one built on the host from the tokens it drained.
  The reasoning is not penalized: it restates what it has worked out while
  drafting the answer, and penalized it drifts instead (asked for the
  capital of France with a 150-token budget, the model named Paris in 1 of
  5 turns with its reasoning penalized, in 5 of 6 with only its answer
  penalized). The one answer that goes unpenalized for a while is one the
  model starts by closing the block itself, inside the loop that decodes
  the reasoning: until the budget's cut it has the loop guard alone.
  `1.0` turns the penalty off.
- **Loop guard, on the host, in both.** The host already drains every
  sampled token. When the newest 24 tokens (`LOOP_WINDOW`) already occurred,
  verbatim, earlier in the stretch (one set lookup per token), the model is
  looping. In the answer the turn ends there, as a stop token would end it,
  and `text` is cut back to the end of the last complete sentence before the
  repeat began; stdout has streamed the repeat already, so a client should
  show the returned `text` once the turn is over. In the reasoning, the
  reasoning ends, exactly as a spent thinking budget ends it, and the answer
  is decoded after the closing tag. Ending reasoning early needs the cache
  to hold exactly the tokens the host kept, so instead of dropping the fires
  the decode loop has run ahead with, the loop stops submitting and takes
  those too. `note` records either case: `stopped a repetition loop`, or
  `stopped a repetition loop in the reasoning`. A loop whose tokens are
  never quite the same, such as a list that counts up, is not verbatim and
  is left to the penalty and `max_tokens`.

## Input

```json
{"messages": [{"role": "system", "content": "..."},
              {"role": "user", "content": "..."},
              {"role": "assistant", "content": "..."},
              {"role": "user", "content": "..."}],
 "session": "ios-voice-session", "max_tokens": 120,
 "temperature": 0.7, "top_p": 0.95, "repetition_penalty": 1.1, "think": false}
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
| `repetition_penalty` | float | `1.1` | In the answer, divides a positive logit (multiplies a negative one) of every token the answer has already generated; prompt and reasoning tokens are not penalized; `1.0` is off |
| `think` | bool | `false` | Open a reasoning block after the cue so the model reasons first; the reasoning goes out as session messages, never to stdout |
| `thinking_budget` | int | two thirds of `max_tokens`, at most 256 | With `think`, the most tokens the reasoning may take before the program closes the block and the answer is decoded |
| `followup` | string, or an array of strings (also as a JSON string) | | Debug only: further user messages answered in the same process, so reuse can be seen from a shell |

## Output

stdout is the reply text as it is generated, and session messages are the
reasoning as it is generated (with `think`; `pie run` prints each message on
its own line). The return value is one JSON object:

```json
{"text": "...", "reasoning": "...", "thought_tokens": 40, "prompt_tokens": 95,
 "reused": 0, "new_prefill": 95, "generated": 93, "resumed": false, "note": ""}
```

| Field | Meaning |
| --- | --- |
| `text` | The spoken reply, reasoning excluded; after a repetition loop, cut back to the last complete sentence before the repeat |
| `reasoning` | The reasoning, trimmed; empty when the model did not reason |
| `thought_tokens` | Of `generated`, the tokens spent inside reasoning blocks |
| `prompt_tokens` | Tokens in the rendered transcript |
| `reused` | Prompt tokens that came from earlier turns' published state |
| `new_prefill` | Prompt tokens this turn computed |
| `generated` | Tokens sampled this turn, reasoning included; the closing tokens the program prefills are not counted |
| `resumed` | Whether any published prefix was found |
| `note` | Why a publish or a retirement was skipped, if one was, why `text` is empty after a turn that only reasoned, or that a repetition loop was stopped (`stopped a repetition loop`, or `... in the reasoning`) |

With `followup` the object also carries `turns`, every turn's accounting.

## Run

`pie run` resolves curated examples by name once they are built:

```bash
cargo build --release --target wasm32-wasip2 -p voice-chat
pie run voice-chat -- --text "hello" --session demo
pie run voice-chat -- --messages '[{"role":"user","content":"hello"},{"role":"assistant","content":"Hi there."},{"role":"user","content":"How are you?"}]' --session demo
pie run voice-chat -- --text "A bat and a ball cost 1.10 dollars in total. The bat costs 1 dollar more than the ball. How much does the ball cost?" --think --max_tokens 120 --thinking_budget 40
```

Every `pie run` boots a fresh engine, so a published prefix does not outlive
the command. To watch later turns reuse earlier ones within one engine:

```bash
pie run voice-chat -- --text "What is one good reason to run a language model on a phone?" --session demo --followup "Can you say that again in five words?"
pie run voice-chat -- --text "My name is Aarush." --session demo --followup '["What is my name?", "And what city is the Eiffel Tower in?", "Which country is that in?"]'
```
