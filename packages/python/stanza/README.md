# pie-stanza

Stanza by pie: any model, any adapter, one program.

A stanza is a few lines of Poem that say what is trainable in a model and
where it attaches. `pie-stanza` mounts stanzas on a pie model program, trains
one by differentiating that very program in PyTorch, and serves the result
with pie, next to other adapters in the same batch. The trainer and the
server read the same files under `$PIE_HOME/models`.

The import name is `pie_stanza` and the command is `stanza`.

## Python

```python
import pie_stanza as stanza

model = stanza.Model("qwen3-0.6b", weights="Qwen3-0.6B", pie_home=".pie-stanza")
model.mount("lora", "ia3", "examples/stanza-larkspur/steer.poem")
print(model.check())

train = stanza.chat("examples/stanza-larkspur/data/train.jsonl", model)
t = model.trainer("ia3", device="mps")
for batch in train.batches(rows=8, epochs=3, seed=0):
    t.forward_backward(batch)
    t.optim_step(lr=3e-2)
adapter = t.save("adapters/larkspur-ia3", name="larkspur-ia3")

with model.serve(port=18431) as pie:
    print(pie.chat(train.messages(0), adapter))
```

`examples/stanza-larkspur/train_and_serve.py` is the whole demo program.

| Name | What it is |
|---|---|
| `Model(id, weights=None, *, pie_home, models_dir, pie_bin)` | A pie model id, its Hugging Face checkpoint and its PIE_HOME. `weights=None` is enough to mount and check. |
| `Model.mount(*stanzas)` | Mounts standard-library stanzas (`"lora"`, `"ia3"`) or `.poem` files, in order, and traces the program. |
| `Model.check(weights=False)` | The map `stanza check` prints. |
| `Model.trainer(stanza, ...)` | A `Trainer` for slot 0 of one stanza's banks. Its `save` returns an `Adapter`. |
| `Model.serve(port=18431)` | A context manager that runs `pie serve` and gives a `Pie`. |
| `Pie.chat`, `Pie.chat_many`, `Pie.score` | Greedy replies (batched by the engine) and next-token log-probs, with or without an adapter. |
| `Adapter.load(dir)`, `Adapter.to_peft(dir)`, `from_peft(dir, model, name)` | Bundles on disk, and conversion to and from Hugging Face PEFT (LoRA, IA3). |
| `chat(path, model)`, `pairs(path, model)` | OpenAI chat JSONL and OpenAI preference JSONL, rendered once with the model's chat template. |
| `evaluate(pie, adapter, data, metric)`, `json_exact` | Served replies scored against the reference answers. |
| `skew(trainer, pie, data)` | Trained against served log-probs on answer tokens: P3 mean and max, P4 top-1 agreement, and the bf16 floor. |

## Command line

```
stanza mount lora ia3 examples/stanza-larkspur/steer.poem [--pie-home DIR]
stanza check --model qwen3-0.6b [--weights DIR]
stanza train --model qwen3-0.6b --stanza ia3 --data DATA.jsonl --epochs 3 --lr 3e-2 --out adapters/NAME
stanza serve --model qwen3-0.6b [--port 18431]
stanza eval  --adapter adapters/NAME --data TEST.jsonl [--port 18431]
stanza skew  --adapter adapters/NAME --data TEST.jsonl --n 32 [--port 18431]
```

`stanza train --loss dpo --data PREFS.jsonl` trains on OpenAI preference
JSONL with DPO, whose reference is the plain arm of the same program.

Every subcommand takes `--pie-home` (default `$PIE_HOME`, else
`~/.pie-stanza`), `--model` (default `qwen3-0.6b`) and `--weights` (default
`$STANZA_WEIGHTS`). `check` runs its four weight checks only when `--weights`
is given. `train` prints one JSON line per optimizer step:
`{"step", "loss", "tokens", "seconds"}`. `eval` and `skew` talk to a server
that `stanza serve` started; `skew` also records its numbers in the
adapter's `stanza.json`.

## Files

- `$PIE_HOME/models/lib/form.poem`: which stanzas are mounted. `mount` writes it.
- `$PIE_HOME/models/lib/stanzas/*.poem`: the stanza files.
- `$PIE_HOME/stanza/runs/<mount hash>/<model>/`: what `pie model trace` wrote for that mount.
- `$PIE_HOME/stanza/serve.log`: the output of the last `pie serve`.
- An adapter bundle: `adapter.safetensors` (one bf16 tensor per bank, keyed by the Poem bank name) and `stanza.json`.
