# inferlet-js

JavaScript/TypeScript library for writing Pie inferlets.

## Quick Start with Examples

Examples live under `sdk/examples/javascript`:

- **text-completion** - Basic text generation with sampling
- **beam-search** - Beam search decoding

Install dependencies and build an example:

```bash
cd sdk/javascript
npm install

# If needed, activate Python venv (e.g., sdk/python/.venv)
# cd ../python && source .venv/bin/activate && cd ../javascript

# Build example
bakery build "$PWD/../examples/javascript/text-completion" \
  -o "$PWD/../text-completion.wasm"
```

## How to Create My Own Inferlet

```bash
# Create a new inferlet with TypeScript
bakery create my-inferlet --ts
```

Note: The default is Rust. Use `--ts` for TypeScript/JavaScript projects.

This generates:
- `index.ts` - Your inferlet code (or `index.js` for JavaScript)
- `package.json` - Package manifest
- `tsconfig.json` - TypeScript configuration with path mappings (TypeScript only)

### Build

```bash
cd my-inferlet
# With venv activated
bakery build "$PWD" -o "$PWD/my-inferlet.wasm"
```

### Run

Make sure the Pie engine is running, then submit the compiled inferlet:

```bash
pie-cli submit my-inferlet.wasm
```

## Writing Inferlets

Inferlets use **top-level await**. Import the APIs you need from `'inferlet'`:

```typescript
// my-inferlet/index.ts

import { Context, getAutoModel, getArguments, send } from 'inferlet';

const args = getArguments();
const prompt = (args.prompt as string) ?? 'Hello, world!';

const model = getAutoModel();
const ctx = new Context(model);

ctx.fillSystem('You are a helpful assistant.');
ctx.fillUser(prompt);

const result = await ctx.generate({
  sampling: { topP: 0.95, temperature: 0.6 },
  stop: { maxTokens: 256, sequences: model.eosTokens }
});

send(result);
```

The build system automatically:
- Resolves imports from the `inferlet` package
- Wraps your code in the WIT interface
- Handles error reporting

## Available APIs

Import the APIs you need from the `'inferlet'` package:

```typescript
import {
  Context,
  getAutoModel,
  getArguments,
  send,
  // ... other APIs as needed
} from 'inferlet';
```

### Core Functions
- `getAutoModel()` - Returns the model instance
- `getArguments()` - Returns command-line arguments as an object
- `send(text)` - Sends output to the client

### Classes
- `Context` - Generation context with KV cache
- `Sampler` - Token sampling strategies (`.greedy()`, `.topP()`, `.topK()`)
- `ChatFormatter` - Chat template formatting
- `Tokenizer` - Text tokenization

### Stop Conditions
Stop conditions are configured in the `generate()` options object:

```typescript
const result = await ctx.generate({
  sampling: { topP: 0.95, temperature: 0.6 },
  stop: {
    maxTokens: 256,
    sequences: model.eosTokens  // Array of token sequences
  }
});
```

## CLI Reference

### Create

```bash
bakery create <name> [OPTIONS]

Options:
  --ts, -t           Create a TypeScript project instead of Rust
  -o, --output <dir> Output directory (default: current directory)
```

Note: TypeScript projects support both `.ts` and `.js` files. The default (without `--ts`) creates a Rust project.

### Build

```bash
bakery build <input> -o <output.wasm> [OPTIONS]

Options:
  --debug    Use debug build of StarlingMonkey runtime
```

## TypeScript Support

The generated `tsconfig.json` provides full IDE support:
- Auto-completion for all inferlet APIs
- Type checking for your code
- Import resolution via path mappings

Path mappings point to `inferlet-js/src/` for type definitions.

## Testing

```bash
npm test              # Unit tests + mock-based WASM tests
npm run test:unit     # Unit tests only (fast)
npm run test:wasm     # Mock-based WASM tests
npm run test:integration  # Real WASM execution tests (requires pie-cli)
npm run test:all      # Everything
npm run test:watch    # Watch mode
```

### Test Structure

- **Unit tests** (`src/__tests__/`) - Fast tests for individual modules (sampler, chat, args, etc.)
- **Mock WASM tests** (`test/wasm/__tests__/`) - Tests using vitest aliases to mock WIT imports
- **Integration tests** (`test/integration/__tests__/`) - Real WASM execution tests

### Integration Tests

Integration tests verify the full pipeline: TypeScript → WASM → JS execution.

```
TypeScript source → bakery build → .wasm → jco transpile → Node.js execution
```

These tests require `pie-cli` in PATH. They:
1. Build test fixtures to WASM using `bakery build`
2. Transpile WASM to JS using `jco transpile`
3. Execute the transpiled component with mock host functions
4. Verify outputs are captured correctly

Test fixtures are in `test/integration/fixtures/`. To add a new fixture:

```bash
mkdir test/integration/fixtures/my-test
# Create index.ts and package.json
```

Then add tests in `test/integration/__tests__/`.
