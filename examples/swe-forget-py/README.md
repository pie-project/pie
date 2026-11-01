# swe-forget-py

A coding agent (THOUGHT, one ```bash``` command, its output, repeat) that drops
stale tool output from its KV cache without reading anything again.

Most of a coding agent's context is old command output: test logs, search hits,
files it has finished with. Every later token pays attention over all of it. A
chat API can cut that text, but the cut changes the prefix, so the server reads
everything after it again. Here the program calls `WorkingSet.forget(start, end)`:

- later rows stop reading the whole pages in that span, so attention gets cheaper;
- nothing is prefilled again, and rows keep the positions they had;
- pages only this working set reads go back to the pool, so more agents fit.

`forget_after` turns after a command, the middle of its output is dropped, and with
`forget_replies` the middle of the agent's old replies too; `forget_keep` tokens stay
at each end, so the turn still reads as a command that printed something. The rest of
the loop is `swe-swarm-py`'s, as one branch.

Keep the opening of the context: a model that reads none of its first tokens loses the
thread (attention sinks). The system prompt holds them here.

Forgetting is approximate on purpose. A token written after an output was read
keeps what it learned from it, and a fresh read of the same context would not.

```bash
pie serve &
python harness.py linear --swe 97 --parallel 8                          # no forgetting
python harness.py forget --swe 97 --parallel 8 --config '{"forget_after":3,"forget_keep":32,"forget_replies":true}'
python harness.py openai --swe 97 --parallel 8 --base-url http://127.0.0.1:8000/v1
python harness.py openai --swe 97 --parallel 8 --config '{"clear_after":3,"forget_keep":32,"clear_replies":true}'  # the same cut, client side
python summarize.py results/*.jsonl
```

`harness.py` runs each command in a copy of the repository, scores the patch with
SWE-bench's hidden tests, and writes one JSON line per task to `--out`. Without
`--swe` it runs the three small tasks in `tasks.py`.

`forget` needs a model that rotates rows by the position table and has no recurrent
state; other models refuse the call. Hybrid models have folded the output into their
state already, so there is nothing to drop.

`examples/kv-forget-probe-py` checks the primitive: `--mode exact` compares it with an
attention mask that hides the same slots, and `--mode sweep` measures speed against
context length and the share of pages forgotten.
