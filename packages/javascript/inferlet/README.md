# @pie-project/inferlet

JavaScript SDK for writing Pie inferlets.

```js
import { chat, eta, model } from '@pie-project/inferlet';
const { ForwardPass, Pipeline, WorkingSet, intrinsics, reduceArgmax } = eta;

export function main(input) {
  const tokens = [...chat.prefix(), ...model.encode(input.prompt ?? 'The capital of France is')];
  const maxTokens = Number(input.max_tokens ?? 8);
  const ws = new WorkingSet();
  const pipe = new Pipeline();

  let done = 0;
  for (let i = 0; i < maxTokens; i++) {
    const fwd = new ForwardPass();
    fwd.embed(tokens.slice(done));
    fwd.state([ws], { fold: 'all' });
    const out = fwd.epilogue(() => reduceArgmax(intrinsics.output(0, 'logits')));
    pipe.submit(fwd);
    done = tokens.length;
    tokens.push(out.takeScalar());
  }
  pipe.close();

  return { text: model.decode(tokens.slice(-maxTokens)) };
}
```

Exports `eta` (`Tensor`, `Channel`, `WorkingSet`, `StateGeometry`, `ForwardPass`, `Pipeline`), `model`, `tokenizer`, `grammar`, `mask`, `chat`, `reasoning`, `tools`, `media`, and `session`. Trace containers emitted by `eta` are byte-identical to the Rust [`inferlet`](../../../crates/inferlet) crate's.

## Build & test

```bash
npm ci --prefix ..
npm run generate-bindings && npm run build && npm test
```
