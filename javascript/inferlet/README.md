# @pie-project/inferlet (JavaScript)

The JavaScript library for writing Pie inferlets — the small programs that run
next to the model.

```js
import { model, eta } from '@pie-project/inferlet';
const { Channel, ForwardPass, Pipeline, WorkingSet, dtype, intrinsics, reduceArgmax, reshape } = eta;

export function main(input) {
  const prompt = model.encode(input.prompt ?? 'The capital of France is');
  const n = prompt.length;
  const ws = new WorkingSet();
  ws.reserve(1);

  const tokens = Channel.from(prompt, dtype.i32);
  const indptr = Channel.from([0, n], dtype.u32);
  const tokOut = new Channel([1], dtype.i32);
  // ...kvLen / pages / pageIndptr / wSlot / wOff / positions as in
  // examples/text-completion-js/index.js

  const fwd = new ForwardPass();              // picks the model's pass kind
  fwd.embed(tokens, indptr);
  fwd.bindState(ws, { kvLen, pages, pageIndptr, wSlot, wOff, positions });
  fwd.epilogue(() => {                        // traced once, runs on the device
    tokOut.put(reshape(reduceArgmax(intrinsics.logits()), [1]));
  });

  const pipe = new Pipeline();
  fwd.submit(pipe);
  const first = tokOut.takeScalar();           // blocks until the fire settles
  pipe.close();
  return { text: model.decode([first]) };
}
```

## What is here

`eta` is the ETA authoring surface (a port of the Rust `eta-dsl`/`eta-ir`
crates and `inferlet::eta`): `Tensor` with `.add/.sub/.mul/.div/.rem/.neg/
.divCeil`, the op set (`reduceArgmax`, `gumbelMax`, `nucleusSample`,
`softmax`, `topK`, …), `Channel`, `WorkingSet`, `RsWorkingSet`,
`ForwardPass`, `Pipeline`, `runAhead`, `prefillChunks`, and `eta.diffusion`
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
(or `pie inferlet install index.js`, or `client.install_program(index_js,
pie_toml)`) hands the module to the server, which runs it under the
JavaScript language component -- one wasm bundling StarlingMonkey, this
library and acorn, built once by `language/build.sh` (componentize-js against
the derived world in `wit/`) and installed at
`~/.pie/languages/javascript.wasm`. The component rewrites the module's
top-level `import`/`export` declarations (`language/transform.js`) and
runs the body; a program imports `@pie-project/inferlet` or the host's
`pie:inferlet/*` interfaces (with their version: `'pie:inferlet/model@0.3.0'`)
and exports `main(input)`.
