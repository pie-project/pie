# @pie-project/inferlet (JavaScript)

The JavaScript library for writing Pie inferlets — the small programs that run
next to the model.

```js
import { chat, eta, model } from '@pie-project/inferlet';
const { ForwardPass, Pipeline, WorkingSet, intrinsics, reduceArgmax } = eta;

export function main(input) {
  const tokens = [...chat.prefix(), ...model.encode(input.prompt ?? 'The capital of France is')];
  const maxTokens = Number(input.max_tokens ?? 8);
  const ws = new WorkingSet(); // the sequence's state, whatever the model keeps
  const pipe = new Pipeline();

  let done = 0; // tokens already in the working set
  for (let i = 0; i < maxTokens; i++) {
    const fwd = new ForwardPass();
    fwd.embed(tokens.slice(done));                       // the new tokens
    fwd.state([ws], { fold: 'all' });                    // appended, and final
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
`embed(array)` seeds the token and row channels, `state([ws], ...)` says where
the rows land (by default after what the working set holds) and what freezes,
and an `epilogue` that returns a tensor publishes it on a channel the host
reads. The explicit spellings — `Channel.from(..., dtype.u32)`,
`{ slots, fold }` geometry channels, `ch.put(tensor)` inside the stage — are
the same objects and stay available; `examples/text-completion-js` uses them
for a one-pass prefill and a device loop-carried decode.

## What is here

`eta` is the ETA authoring surface (a port of the Rust `eta-dsl`/`eta-ir`
crates and `inferlet::eta`): `Tensor` with `.add/.sub/.mul/.div/.rem/.neg/
.divCeil`, the op set (`reduceArgmax`, `gumbelMax`, `nucleusSample`,
`softmax`, `topK`, …), `Channel`, `WorkingSet`, `StateGeometry`,
`ForwardPass`, `Pipeline` (`submit`, `runAhead`), `readoutSpans`, and
`eta.diffusion` (a block-diffusion text's samplers `entropyBoundAccept` /
`stableAndConfident` / `linearTemperature`). The container bytes
it emits are **byte-identical** to the Rust `inferlet` crate's for the same program
(`src/__tests__/eta_goldens.test.ts` pins them). `grammar`/`mask`, `chat`,
`reasoning`, `tools`, `media`, `session`, `model`/`tokenizer` wrap the other
host interfaces.

A model receives two things per fire: **inputs** (device channels) and
**switches** (a value from a finite set that picks the model's path). One
`ForwardPass` holds every lane of one computation (a CFG denoise step's text
and image lanes of both branches, say); `switch(lane, name, value)`,
`input(lane, name, ch)`, `media(lane, spans)` and `group(lane, id)` address a
lane by index, `mask(rows)` is pass-wide, and epilogue intrinsics take the lane
(`velocity(lane, …)`, `hidden(lane, …)`, `pixels(lane, …)`). `model.switches()`
lists a model's switches and `model.lane(switches)` says what a lane with those
values binds: its inputs, whether it binds a working set (`state`), whether it
embeds tokens, and its readout. Generative latents are the VAE's own
`[t, h, w, C]`; patching and positions live in the model. A causal VAE
decodes frame by frame into a working set, one latent frame per slot, and the
runtime's `fresh` switch (cursor 0) picks its first-frame branch. A `prologue`
runs before the forward, an `epilogue` after it.

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
npm ci --prefix ..           # the packages/javascript/ workspace: one install for every package
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
