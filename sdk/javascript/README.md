# inferlet JavaScript SDK

TypeScript/JavaScript API for writing Pie inferlets.

```typescript
import { Context, Model, Sampler, runtime } from 'inferlet';

export async function main(input: { prompt: string }) {
  const model = Model.load(runtime.models()[0]);

  using ctx = new Context(model);
  ctx.system('You are helpful.').user(input.prompt);

  return await ctx
    .generate(Sampler.topP(0.6, 0.95), { maxTokens: 256 })
    .collectText();
}
```

## Main pieces

- `Context`: owns KV-cache state and chat/raw token buffers.
- `Forward`: runs one explicit forward pass with samplers, probes, masks, and
  manual page control.
- `Generator`: multi-step generation loop with stop conditions, constraints,
  speculation, adapters, and JSON collection.
- `chat`, `reasoning`, `tools`: optional decoders and helpers for model-native
  formats.
- `runtime`, `session`, `messaging`: host services exposed to inferlets.

## Development

```bash
cd sdk/javascript
npm install
npm run build
npm test
```

Regenerate WIT bindings after runtime WIT changes:

```bash
npm run generate-bindings
```
