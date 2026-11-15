# team-episode-py

s3-style team episodes on pie ([Jha et al., "s3: Simple Scaling of Swarms via
Cooperative Multi-agent RL"](https://arxiv.org/abs/2611.08615)): N copies of one
model work on one task at once, talk only through tools, and share one team
reward. The whole episode runs in one inferlet, next to the model:

- the shared prompt (team rules, task, tools) is prefilled once, and each agent
  is a fork of it plus its own private prompt;
- every forward pass decodes one token for every running agent, one lane each;
  idle and finished agents stay out of the pass;
- tool results and teammates' messages are appended to the agent's own working
  set, so no context is ever read twice;
- sampling (temperature, top-p, a seeded Gumbel draw) and the log-probability of
  each drawn token under the distribution it was drawn from are computed on the
  device.

Each episode returns per-agent trajectories (token ids, logprobs, and a mask that
is 1 only on generated tokens), also flattened to one `{tokens, mask, logprobs}`
list per agent, the form RL trainers such as PRIME-RL, SkyRL and verl take.

## Files

| file | what |
| --- | --- |
| `episode.py` | The episode, engine-agnostic text in and out: Qwen3 hermes tool prompts and parser, mailbox, both s3 tasks, team reward. Stdlib only, so it runs inside the inferlet. |
| `test_episode.py` | Tests with a scripted fake driver (`python test_episode.py`). |
| `main.py` | The inferlet: one team episode per process. The uploaded program is `episode.py + "\n" + main.py`. |
| `run.py` | Client: installs that program, launches K episodes concurrently, writes JSONL, prints a summary. |

## Run it

```bash
pie model import Qwen/Qwen3-0.6B
pie serve &
pip install -e packages/python/client            # from the repo root
cd examples/team-episode-py
python test_episode.py
python run.py --task aggregate --n-agents 4 --episodes 8 --concurrency 8 --out agg.jsonl
python run.py --task maze --depth 2 --n-agents 4 --episodes 1 --out maze.jsonl
```

`--verify` re-reads each agent's whole trajectory into a fresh working set and
re-scores the first tokens of every generated chunk, a check that injected text
really sits in the agent's context.

## The tasks

Every agent has `message(to, text)` (`to` is an agent id or `"all"`) and
`read_messages()`. An agent that ends its turn without a tool call idles until
mail arrives; a notice of unread mail rides on its next tool result.

- `aggregate` (s3's information aggregation): each agent privately holds one
  number from 1 to 99, written in words. `submit(answer)` sends the full sorted
  list, duplicates kept. The team succeeds if any agent submits it.
- `maze` (s3's decentralized exploration): a tree with branching 3 and corridors
  of 10 rooms (`--depth 2` is 121 rooms, `--depth 3` is 391), one exit at a leaf.
  `look()` and `move(room)`; any agent reaching the exit ends the episode.

Reward is s3's: `S * (1 - 0.25 * min(1, tokens / (n_agents * budget)))`, with S
the team's success.

## Measured

Apple M5 (Metal), untrained Qwen3-0.6B, 4 agents per team, temperature 1.0.
The untrained model solves neither task; the numbers are about the engine.

| run | result |
| --- | --- |
| 32 aggregate episodes, 8 at a time | 2,148 team episodes/h, 125 generated tokens/s, 4.5 s mean per episode |
| 1 maze episode, depth 2, budget 2,048 per agent | 160 s, 2,324 forward passes, 6,212 generated tokens |
| `--verify`, maze | tokens right after an injection score within 3e-5 to 3e-4 nats of the live run |

94.5% of an episode's time is spent waiting on forward passes. There is one host
round trip per pass, about 0.8 passes per generated token, and 1.3 lanes per
pass on average.

## Limits

- An agent's whole context stays live for the episode. A maze agent that keeps
  calling `look()` reaches about 9k tokens, so a 4 GiB KV pool holds one or two
  maze episodes at budget 2,048. When the pool runs out the episode ends with
  `"aborted"` and `run.py` counts it apart from successes and failures.
- Decoding takes a host round trip per token. Decoding several tokens per trip
  on the device, as `text-completion-py` does, is the obvious next step.
- Episodes run as separate processes. Running many episodes in one inferlet
  would put more lanes in each pass.

## Against an HTTP orchestrator

`baseline.py` runs the same episodes the way s3 and vLLM-based trainers do: an
outside orchestrator re-sends each running agent's whole context to
`/v1/completions` in chunks of up to 1,024 tokens and adds messages between
chunks. It takes vLLM (launch it with `--logprobs-mode processed_logprobs`, so
logprobs are after temperature, as here) or pie's own `/v1/completions`.
`bench.py` runs both drivers on the same seeds and prints team episodes per hour,
the pie / http ratio, and whether both generated about the same number of tokens
(within 15%; otherwise the ratio is marked not comparable).

```bash
pip install httpx tokenizers
python bench.py --task maze --n-agents 4 --episodes 2 --concurrency 2 --budget-tokens 1024 --out-dir bench-out
python bench.py --task aggregate --n-agents 4 --episodes 32 --concurrency 8 --out-dir bench-out
```

Same M5, same server: the http side is pie's own `/v1/completions`, so engine
and model are identical and only the driver differs.

| task | pie | http | pie / http | tokens per episode |
| --- | --- | --- | --- | --- |
| maze, depth 2, 2 episodes, budget 1,024 | 79.6 episodes/h, 74 tok/s | 57.1 episodes/h, 50 tok/s | 1.39x | 3,356 vs 3,144, matched |
| aggregate, 32 episodes, 8 at a time | 2,375 episodes/h, 139 tok/s | 6,579 episodes/h, 270 tok/s | 0.36x | 210 vs 148, not matched |

Long episodes favour the inferlet: the http side re-sent 567k prompt tokens per
maze episode, while the inferlet never re-reads context. Short episodes favour
the request path: the server batches every agent's request into each pass and
serves most re-sent context from its prefix cache, while the inferlet makes a
host round trip per token at about 1.3 lanes per pass.

pie's `/v1/completions` drops `<tool_call>` markup from raw completions and never
fires the `</tool_call>` stop, so `baseline.py` rebuilds each chunk's text from
the returned tokens and cuts it at the first stop itself; token ids come from the
local Qwen3 tokenizer.
