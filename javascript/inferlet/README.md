# @pie-project/inferlet (JavaScript)

The JavaScript library for writing Pie inferlets — the small programs that run
next to the model.

```js
import { chat, eta, model } from '@pie-project/inferlet';
const { ForwardPass, Pipeline, WorkingSet, intrinsics, reduceArgmax } = eta;

export function main(input) {
  const tokens = [...chat.prefix(), ...model.encode(input.prompt ?? 'The capital of France is')];
  const maxTokens = Number(input.max_tokens ?? 8);
  const ws = new WorkingSet({ tokens: tokens.length + maxTokens }); // KV pages (and recurrent state on a hybrid model)
  const pipe = new Pipeline();

  let done = 0; // tokens already in the KV cache
  for (let i = 0; i < maxTokens; i++) {
    const fwd = new ForwardPass();                       // the model's pass kind
    fwd.embed(tokens.slice(done));                       // the new tokens
    fwd.bindState(ws, ws.geometry(done, tokens.length));
    const out = fwd.epilogue(() => reduceArgmax(intrinsics.logits())); // traced once, runs on the device
    pipe.submit(fwd);
    done = tokens.length;
    tokens.push(out.takeScalar());                       // blocks until the fire settles
  }
  pipe.close();

  return { text: model.decode(tokens.slice(-maxTokens)) };
}
```

That is `examples/quickstart-js`. Every line above stands for device state:
`embed(array)` seeds the token and row channels, `ws.geometry(start, end)`
derives the six KV-geometry channels of one contiguous span (edit its fields
for anything else), and an `epilogue` that returns a tensor publishes it on a
channel the host reads. The explicit spellings — `Channel.from(...,
dtype.u32)`, a `{ kvLen, pages, ... }` geometry, `ch.put(tensor)` inside the
stage — are the same objects and stay available; `examples/text-completion-js`
uses them for chunked prefill and a device loop-carried decode.

## What is here

`eta` is the ETA authoring surface (a port of the Rust `eta-dsl`/`eta-ir`
crates and `inferlet::eta`): `Tensor` with `.add/.sub/.mul/.div/.rem/.neg/
.divCeil`, the op set (`reduceArgmax`, `gumbelMax`, `nucleusSample`,
`softmax`, `topK`, …), `Channel`, `WorkingSet`, `RsWorkingSet`,
`ForwardPass`, `Pipeline` (`submit`, `runAhead`), `prefillChunks`, and `eta.diffusion`
(the diffusion pass's `Mode` plus `entropyBoundAccept` / `stableAndConfident` /
`linearTemperature`). The container bytes
it emits are **byte-identical** to the Rust `inferlet` crate's for the same program
(`src/__tests__/eta_goldens.test.ts` pins them). `grammar`/`mask`, `chat`,
`reasoning`, `tools`, `media`, `session`, `model`/`tokenizer` wrap the other
host interfaces.

## Async, and why host reads block

The runtime's world is component-model async (`channel.take`,
`session.receive` are `async func`), which componentize-js cannot lower yet.
The host therefore also offers blocking twins (`take-blocking`,
`receive-blocking`), and this library is built against a **derived world** with
the async imports removed (`npm run derive-wit` → `wit/`). `takeHost()` /
`takeScalar()` and `session.receive()` are synchronous and block the guest's
task until the cell fills / the message arrives.

## Building

```
npm ci --prefix ..           # the javascript/ workspace: one install for every package
npm run generate-bindings    # derive the JS world + jco types + tsconfig paths
npm run build                # tsc
npm test                     # vitest (stubs in src/__tests__)
```

A JavaScript inferlet is its source: there is no build. `pie run index.js`
(or `pie inferlet install index.js`, or `client.installProgram("index.js")`,
which names the program by its directory) hands the module to the server,
which runs it under the
JavaScript language component -- one wasm bundling StarlingMonkey, this
library and acorn, built once by `language/build.sh` (componentize-js against
the derived world in `wit/`) and installed at
`~/.pie/languages/javascript.wasm`. The component rewrites the module's
top-level `import`/`export` declarations (`language/transform.js`) and
runs the body; a program imports `@pie-project/inferlet` or the host's
`pie:inferlet/*` interfaces (with their version: `'pie:inferlet/model@0.3.0'`)
and exports `main(input)`.
