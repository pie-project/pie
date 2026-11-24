"""The whole Larkspur demo: mount three stanzas, train each on the bike shop's
tickets, then serve all three next to the base model in one engine batch.

    python examples/stanza-larkspur/train_and_serve.py WEIGHTS

Run it from the repository root; WEIGHTS is a Hugging Face Qwen3-0.6B directory.
"""
import sys

import pie_stanza as stanza

model = stanza.Model("qwen3-0.6b", weights=sys.argv[1], pie_home=".pie-stanza")
model.mount("lora", "ia3", "examples/stanza-larkspur/steer.poem")   # the program pie trains and serves
print(model.check())

train = stanza.chat("examples/stanza-larkspur/data/train.jsonl", model)   # OpenAI chat JSONL
test = stanza.chat("examples/stanza-larkspur/data/test.jsonl", model)

trainers, adapters = {}, {}
for form, lr in [("lora", 1e-3), ("ia3", 3e-2), ("steer", 1e-2)]:
    t = model.trainer(form, device="mps")
    for batch in train.batches(rows=8, epochs=3, seed=0):
        t.forward_backward(batch)                    # loss="cross_entropy" by default
        t.optim_step(lr=lr)
    trainers[form], adapters[form] = t, t.save(f"adapters/larkspur-{form}", name=f"larkspur-{form}")

with model.serve(port=18431) as pie:                 # pie on Metal, same mounted program
    ticket = test.messages(0)
    replies = pie.chat_many([(ticket, None)] + [(ticket, a) for a in adapters.values()])   # one engine batch
    for who, reply in zip(["base", *adapters], replies):
        print(f"{who:>6}: {reply}")
    for form, a in adapters.items():
        print(form, stanza.evaluate(pie, a, test, metric=stanza.json_exact),
              stanza.skew(trainers[form], pie, test.sample(32)))
